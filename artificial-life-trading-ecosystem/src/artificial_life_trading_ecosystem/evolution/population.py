"""Population-level evolution engine and reproduction logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from artificial_life_trading_ecosystem.agents.agent import Agent
from artificial_life_trading_ecosystem.config import settings
from artificial_life_trading_ecosystem.lineage.lineage import LineageRecord
from artificial_life_trading_ecosystem.models.genomes.decision_genome import DecisionGenome
from artificial_life_trading_ecosystem.trading.engine import TradingEngine


class GenerationStatus(str, Enum):
    READY = "READY"
    RUNNING = "RUNNING"
    EVALUATED = "EVALUATED"


@dataclass
class PopulationConfig:
    """Configuration for the artificial-life population layer."""

    population_size: int = 100
    generation: int = 0
    generations: int = 50
    generation_trading_days: int = 10
    starting_capital: float = 100000.0
    survival_threshold: float = 1.0
    mutation_rate: float = 0.05
    mutation_sigma: float = 0.10
    immigrant_fraction: float = 0.05
    elite_count: int = 1
    seed: int = 42
    metadata: dict[str, Any] = field(default_factory=dict)


class Population:
    """Concrete trading population that evaluates agents, selects parents, and reproduces them."""

    def __init__(self, config: PopulationConfig | None = None, *, forecaster: Any | None = None, seed: int | None = None) -> None:
        cfg = config or PopulationConfig()
        self.config = cfg
        self.population_size = int(cfg.population_size)
        self.starting_capital = float(cfg.starting_capital)
        self.generation = int(getattr(cfg, "generation", 0))
        self.generation_trading_days = int(getattr(cfg, "generation_trading_days", 10))
        self.generation_status = GenerationStatus.READY
        self.survival_threshold = 1.0
        self.mutation_rate = float(cfg.mutation_rate)
        self.mutation_sigma = float(cfg.mutation_sigma)
        self.immigrant_fraction = float(cfg.immigrant_fraction)
        self.elite_count = int(cfg.elite_count)
        self.seed = int(seed if seed is not None else getattr(cfg, "seed", settings.seed))
        self.rng = np.random.default_rng(self.seed)
        self.forecaster = forecaster
        self.engine = TradingEngine()
        self.lineage: dict[str, LineageRecord] = {}
        self.agents: list[Agent] = []
        self.generation_history: list[dict[str, Any]] = []
        self.daily_events: list[dict[str, Any]] = []
        self.day_count = 0
        self._agent_counter = 0
        self._init_population()

    def _new_agent_id(self) -> str:
        self._agent_counter += 1
        return f"A{self._agent_counter:03d}"

    def next_agent_id(self) -> str:
        return self._new_agent_id()

    def _init_population(self) -> None:
        for _ in range(self.population_size):
            genome = DecisionGenome.from_default(seed=int(self.rng.integers(0, 10_000_000)))
            agent = Agent(
                agent_id=self._new_agent_id(),
                generation=0,
                genome=genome,
                starting_capital=self.starting_capital,
                cash=self.starting_capital,
                current_capital=self.starting_capital,
                config={"seed": self.seed},
            )
            agent.alive = True
            agent.status = "READY"
            agent.statistics["immigrant"] = False
            self.agents.append(agent)
            self.lineage[agent.agent_id] = LineageRecord(
                agent_id=agent.agent_id,
                generation=0,
                reproduction_method="initial",
                mutation_applied=False,
                immigrant=False,
                metadata={"origin": "initial"},
            )

    def _shared_forecast(self, market: pd.DataFrame, *, feature_columns: list[str] | None = None) -> list[float]:
        if self.forecaster is None:
            return [0.0] * len(market)
        if feature_columns is None:
            feature_columns = [col for col in market.columns if col not in {"Date", "date"}]
        sequence_length = int(getattr(getattr(self.forecaster, "config", None), "sequence_length", self.engine.context_length + 1))
        outputs: list[float] = [0.0] * len(market)
        for idx, row in market.iterrows():
            history = market.loc[:row.name, feature_columns].copy()
            if len(history) < sequence_length:
                continue
            window = history.tail(sequence_length).to_numpy(dtype=float)
            prediction = self.forecaster.predict(window[np.newaxis, :, :])
            outputs[idx] = float(np.asarray(prediction).reshape(-1)[0])
        return outputs

    def evaluate_generation(
        self,
        market: pd.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        history_prefix: pd.DataFrame | None = None,
    ) -> list[Agent]:
        if market.empty:
            raise ValueError("A generation cannot be evaluated without trading days.")
        if self.generation_status != GenerationStatus.READY:
            raise RuntimeError("Only a READY generation can begin evaluation.")
        self.generation_status = GenerationStatus.RUNNING
        for agent in self.agents:
            agent.status = "RUNNING"
        prefix = history_prefix.copy() if history_prefix is not None else market.iloc[:0].copy()
        if feature_columns is None and self.forecaster is not None:
            model_columns = getattr(self.forecaster, "feature_names", None)
            feature_columns = list(model_columns) if model_columns else [col for col in market.columns if col not in {"Date", "date"}]
        for day_index, (_, row) in enumerate(market.iterrows()):
            prior_market = pd.concat([prefix, market.iloc[:day_index]], ignore_index=True)
            if self.forecaster is not None and not prior_market.empty:
                forecast = self.engine._forecast_for_day(
                    prior_market,
                    len(prior_market) - 1,
                    self.forecaster,
                    feature_columns=feature_columns,
                )
            else:
                forecast = 0.0
            context = self.engine._normalized_context(prior_market, len(prior_market) - 1) if not prior_market.empty else [0.0] * self.engine.context_length
            decisions = [self.engine.simulate_daily_step(agent, row, forecast, context) for agent in self.agents]
            counts = {action: sum(decision["action"] == action for decision in decisions) for action in ("BUY", "HOLD", "SELL")}
            capitals = [float(agent.cash) for agent in self.agents]
            self.daily_events.append(
                {
                    "generation": self.generation,
                    "day": day_index + 1,
                    "date": str(row.get("Date", row.get("date", row.name))),
                    "counts": counts,
                    "trade_count": sum(len(decision["trades"]) for decision in decisions),
                    "best_capital": max(capitals, default=0.0),
                    "mean_capital": float(np.mean(capitals)) if capitals else 0.0,
                    "worst_capital": min(capitals, default=0.0),
                    "agents": decisions,
                }
            )
            self.day_count = day_index + 1
        for agent in self.agents:
            agent.statistics["ending_capital"] = float(agent.cash)
            agent.statistics["survival_status"] = "EVALUATED"
        self.generation_status = GenerationStatus.EVALUATED
        return self.agents

    def reproduce_next_generation(self) -> list[Agent]:
        if self.generation_status != GenerationStatus.EVALUATED:
            raise RuntimeError("Reproduction requires an evaluated generation.")
        self.mark_dead_agents()
        survivors = self.surviving_agents()
        next_generation = self.generation + 1
        if not survivors:
            children = self.make_immigrants(count=self.population_size)
        else:
            ranked = sorted(survivors, key=lambda agent: float(agent.statistics["ending_capital"]), reverse=True)
            elite_count = min(max(1, self.elite_count), len(ranked), self.population_size)
            children = self.preserve_elite(ranked[:elite_count])
            parents = self.capital_weighted_selection(survivors)
            while len(children) < self.population_size:
                if len(parents) >= 2:
                    parent_a, parent_b = self.rng.choice(parents, size=2, replace=len(parents) < 2)
                    children.append(self.create_child(parent_a=parent_a, parent_b=parent_b, generation=next_generation))
                elif parents:
                    children.append(self.create_child(parent_a=parents[0], parent_b=None, generation=next_generation, method="asexual"))
                else:
                    children.extend(self.make_immigrants(count=self.population_size - len(children)))
        self.agents = children[:self.population_size]
        self.generation = next_generation
        self.generation_status = GenerationStatus.READY
        self.day_count = 0
        return self.agents

    def eligible_parents(self) -> list[Agent]:
        if self.generation_status != GenerationStatus.EVALUATED:
            return []
        profit_floor = self.starting_capital * self.survival_threshold
        return [agent for agent in self.agents if agent.status == "ALIVE" and bool(agent.alive) and float(agent.statistics.get("ending_capital", 0.0)) > profit_floor]

    def surviving_agents(self) -> list[Agent]:
        if self.generation_status != GenerationStatus.EVALUATED:
            return []
        profit_floor = self.starting_capital * self.survival_threshold
        return [agent for agent in self.agents if agent.status == "ALIVE" and bool(agent.alive) and float(agent.statistics.get("ending_capital", 0.0)) > profit_floor]

    def mark_dead_agents(self) -> int:
        if self.generation_status != GenerationStatus.EVALUATED:
            raise RuntimeError("Only evaluated generations can assign survival status.")
        dead_count = 0
        profit_floor = self.starting_capital * self.survival_threshold
        for agent in self.agents:
            ending = float(agent.statistics["ending_capital"])
            if agent.status == "DEAD" or not agent.alive or ending <= profit_floor:
                agent.alive = False
                agent.status = "DEAD"
                lineage_record = self.lineage.get(agent.agent_id)
                if lineage_record is not None:
                    lineage_record.status = "DEAD"
                agent.statistics["dead_reason"] = "no_profit"
                dead_count += 1
            else:
                agent.alive = True
                agent.status = "ALIVE"
                lineage_record = self.lineage.get(agent.agent_id)
                if lineage_record is not None:
                    lineage_record.status = "ALIVE"
        return dead_count

    @staticmethod
    def uniform_crossover(parent_a: DecisionGenome, parent_b: DecisionGenome, *, rng: np.random.Generator | None = None) -> DecisionGenome:
        rng = rng or np.random.default_rng()
        child = DecisionGenome(
            context_length=parent_a.context_length,
            hidden_units=parent_a.hidden_units,
            activation=parent_a.activation,
            input_to_hidden=np.where(rng.random(parent_a.input_to_hidden.shape) < 0.5, parent_a.input_to_hidden, parent_b.input_to_hidden).astype(np.float32),
            hidden_bias=np.where(rng.random(parent_a.hidden_bias.shape) < 0.5, parent_a.hidden_bias, parent_b.hidden_bias).astype(np.float32),
            hidden_to_output=np.where(rng.random(parent_a.hidden_to_output.shape) < 0.5, parent_a.hidden_to_output, parent_b.hidden_to_output).astype(np.float32),
            output_bias=float(parent_a.output_bias if rng.random() < 0.5 else parent_b.output_bias),
            signal_threshold=float(parent_a.signal_threshold if rng.random() < 0.5 else parent_b.signal_threshold),
            position_size=float(parent_a.position_size if rng.random() < 0.5 else parent_b.position_size),
            stop_loss=float(parent_a.stop_loss if rng.random() < 0.5 else parent_b.stop_loss),
            take_profit=float(parent_a.take_profit if rng.random() < 0.5 else parent_b.take_profit),
            transaction_cost=float(parent_a.transaction_cost if rng.random() < 0.5 else parent_b.transaction_cost),
        )
        return child

    @staticmethod
    def arithmetic_crossover(parent_a: DecisionGenome, parent_b: DecisionGenome, *, alpha: float = 0.5, rng: np.random.Generator | None = None) -> DecisionGenome:
        rng = rng or np.random.default_rng()
        alpha = float(alpha if alpha is not None else rng.uniform(0.10, 0.90))
        mixed = DecisionGenome(
            context_length=parent_a.context_length,
            hidden_units=parent_a.hidden_units,
            activation=parent_a.activation,
            input_to_hidden=(alpha * parent_a.input_to_hidden + (1.0 - alpha) * parent_b.input_to_hidden).astype(np.float32),
            hidden_bias=(alpha * parent_a.hidden_bias + (1.0 - alpha) * parent_b.hidden_bias).astype(np.float32),
            hidden_to_output=(alpha * parent_a.hidden_to_output + (1.0 - alpha) * parent_b.hidden_to_output).astype(np.float32),
            output_bias=float(alpha * parent_a.output_bias + (1.0 - alpha) * parent_b.output_bias),
            signal_threshold=float(np.clip(alpha * parent_a.signal_threshold + (1.0 - alpha) * parent_b.signal_threshold, 0.0, 1.0)),
            position_size=float(np.clip(alpha * parent_a.position_size + (1.0 - alpha) * parent_b.position_size, 1e-6, 1.0)),
            stop_loss=float(np.clip(alpha * parent_a.stop_loss + (1.0 - alpha) * parent_b.stop_loss, 1e-6, 0.999999)),
            take_profit=float(np.clip(alpha * parent_a.take_profit + (1.0 - alpha) * parent_b.take_profit, 1e-6, 0.999999)),
            transaction_cost=float(np.clip(alpha * parent_a.transaction_cost + (1.0 - alpha) * parent_b.transaction_cost, 0.0, 0.5)),
        )
        return mixed

    @staticmethod
    def mutate_genome(genome: DecisionGenome, *, mutation_rate: float = 0.05, mutation_sigma: float = 0.10, rng: np.random.Generator | None = None) -> DecisionGenome:
        rng = rng or np.random.default_rng()
        child = DecisionGenome(
            context_length=genome.context_length,
            hidden_units=genome.hidden_units,
            activation=genome.activation,
            input_to_hidden=genome.input_to_hidden.copy(),
            hidden_bias=genome.hidden_bias.copy(),
            hidden_to_output=genome.hidden_to_output.copy(),
            output_bias=float(genome.output_bias),
            signal_threshold=float(genome.signal_threshold),
            position_size=float(genome.position_size),
            stop_loss=float(genome.stop_loss),
            take_profit=float(genome.take_profit),
            transaction_cost=float(genome.transaction_cost),
        )
        if mutation_rate <= 0.0:
            return child
        for attr_name in ["input_to_hidden", "hidden_bias", "hidden_to_output"]:
            arr = getattr(child, attr_name).copy()
            mask = rng.random(arr.shape) < mutation_rate
            if np.any(mask):
                arr = arr.astype(np.float64)
                arr[mask] += rng.normal(0.0, mutation_sigma, size=np.count_nonzero(mask))
                setattr(child, attr_name, arr.astype(np.float32))
        if rng.random() < mutation_rate:
            child.output_bias = float(np.clip(child.output_bias + rng.normal(0.0, mutation_sigma), -1.0, 1.0))
        for param_name in ["signal_threshold", "position_size", "stop_loss", "take_profit", "transaction_cost"]:
            if rng.random() < mutation_rate:
                current = float(getattr(child, param_name))
                mutated = current + rng.normal(0.0, mutation_sigma)
                if param_name == "signal_threshold":
                    clipped = float(np.clip(mutated, 0.0, 1.0))
                elif param_name == "position_size":
                    clipped = float(np.clip(mutated, 1e-6, 1.0))
                elif param_name in {"stop_loss", "take_profit"}:
                    clipped = float(np.clip(mutated, 1e-6, 0.999999))
                else:
                    clipped = float(np.clip(mutated, 0.0, 0.5))
                setattr(child, param_name, clipped)
        child.trading_params = {
            "signal_threshold": float(child.signal_threshold),
            "position_size": float(child.position_size),
            "stop_loss": float(child.stop_loss),
            "take_profit": float(child.take_profit),
            "transaction_cost": float(child.transaction_cost),
        }
        return child

    def capital_weighted_selection(self, candidates: list[Agent]) -> list[Agent]:
        if not candidates:
            return []
        weights = np.asarray([max(0.0, float(agent.statistics.get("ending_capital", agent.cash))) for agent in candidates], dtype=np.float64)
        total = weights.sum()
        if total <= 0.0:
            return list(candidates)
        probs = weights / total
        selected = []
        for _ in range(len(candidates)):
            idx = int(self.rng.choice(len(candidates), p=probs))
            selected.append(candidates[idx])
        return selected

    def preserve_elite(self, elites: list[Agent]) -> list[Agent]:
        preserved: list[Agent] = []
        for elite in elites:
            cloner = DecisionGenome(
                context_length=elite.genome.context_length,
                hidden_units=elite.genome.hidden_units,
                activation=elite.genome.activation,
                input_to_hidden=elite.genome.input_to_hidden.copy(),
                hidden_bias=elite.genome.hidden_bias.copy(),
                hidden_to_output=elite.genome.hidden_to_output.copy(),
                output_bias=float(elite.genome.output_bias),
                signal_threshold=float(elite.genome.signal_threshold),
                position_size=float(elite.genome.position_size),
                stop_loss=float(elite.genome.stop_loss),
                take_profit=float(elite.genome.take_profit),
                transaction_cost=float(elite.genome.transaction_cost),
            )
            clone = Agent(
                agent_id=self._new_agent_id(),
                generation=self.generation + 1,
                parent_a=elite.agent_id,
                parent_b=None,
                genome=cloner,
                starting_capital=self.starting_capital,
                cash=self.starting_capital,
                current_capital=self.starting_capital,
                config={"seed": self.seed},
            )
            clone.alive = True
            clone.status = "READY"
            clone.statistics["immigrant"] = False
            clone.statistics["elite"] = True
            preserved.append(clone)
            self.lineage[clone.agent_id] = LineageRecord(
                agent_id=clone.agent_id,
                parent_a=elite.agent_id,
                generation=self.generation + 1,
                reproduction_method="elite",
                mutation_applied=False,
                immigrant=False,
                status="READY",
                metadata={"origin": "elite"},
            )
            self.lineage[elite.agent_id].offspring_ids.append(clone.agent_id)
        return preserved

    def make_immigrants(self, *, count: int | None = None) -> list[Agent]:
        number = int(count if count is not None else max(1, round(self.immigrant_fraction * self.population_size)))
        immigrants: list[Agent] = []
        for _ in range(number):
            genome = DecisionGenome.from_default(seed=int(self.rng.integers(0, 10_000_000)))
            agent = Agent(
                agent_id=self._new_agent_id(),
                generation=self.generation + 1,
                parent_a=None,
                parent_b=None,
                genome=genome,
                starting_capital=self.starting_capital,
                cash=self.starting_capital,
                current_capital=self.starting_capital,
                config={"seed": self.seed},
            )
            agent.alive = True
            agent.status = "READY"
            agent.statistics["immigrant"] = True
            immigrants.append(agent)
            self.lineage[agent.agent_id] = LineageRecord(
                agent_id=agent.agent_id,
                generation=self.generation + 1,
                reproduction_method="immigrant",
                mutation_applied=False,
                immigrant=True,
                status="READY",
                metadata={"origin": "immigrant"},
            )
        return immigrants

    def create_child(self, *, parent_a: Agent | None, parent_b: Agent | None, generation: int, method: str = "sexual") -> Agent:
        if parent_a is None and parent_b is None:
            base_genome = DecisionGenome.from_default(seed=int(self.rng.integers(0, 10_000_000)))
        elif parent_a is not None and parent_b is not None:
            if self.rng.random() < 0.5:
                base_genome = self.uniform_crossover(parent_a.genome, parent_b.genome, rng=self.rng)
            else:
                alpha = float(self.rng.uniform(0.10, 0.90))
                base_genome = self.arithmetic_crossover(parent_a.genome, parent_b.genome, alpha=alpha, rng=self.rng)
        else:
            parent = parent_a or parent_b
            base_genome = DecisionGenome.from_flattened(
                parent.genome.flatten(),
                context_length=parent.genome.context_length,
                hidden_units=parent.genome.hidden_units,
            )
        genome_before_mutation = base_genome.flatten().copy()
        if parent_a is not None or parent_b is not None:
            mutation_rate = max(self.mutation_rate * 2.0, 0.2) if parent_a is None or parent_b is None else self.mutation_rate
            base_genome = self.mutate_genome(
                base_genome,
                mutation_rate=mutation_rate,
                mutation_sigma=self.mutation_sigma,
                rng=self.rng,
            )
        mutation_applied = not np.array_equal(genome_before_mutation, base_genome.flatten())
        child = Agent(
            agent_id=self._new_agent_id(),
            generation=int(generation),
            parent_a=parent_a.agent_id if parent_a is not None else None,
            parent_b=parent_b.agent_id if parent_b is not None else None,
            genome=base_genome,
            starting_capital=self.starting_capital,
            cash=self.starting_capital,
            current_capital=self.starting_capital,
            config={"seed": self.seed},
        )
        child.alive = True
        child.status = "READY"
        child.statistics["immigrant"] = False
        self.lineage[child.agent_id] = LineageRecord(
            agent_id=child.agent_id,
            parent_a=child.parent_a,
            parent_b=child.parent_b,
            generation=int(generation),
            reproduction_method=method,
            mutation_applied=mutation_applied,
            immigrant=False,
            status="READY",
            metadata={"method": method},
        )
        for parent in (parent_a, parent_b):
            if parent is not None and child.agent_id not in self.lineage[parent.agent_id].offspring_ids:
                self.lineage[parent.agent_id].offspring_ids.append(child.agent_id)
        return child

    def asexual_fallback(self, survivors: list[Agent], *, target_count: int) -> list[Agent]:
        if not survivors:
            return self.zero_survivor_fallback(target_count=target_count)
        return [self.create_child(parent_a=survivors[0], parent_b=None, generation=self.generation + 1, method="asexual") for _ in range(target_count)]

    def aseuxal_fallback(self, survivors: list[Agent], *, target_count: int) -> list[Agent]:
        return self.asexual_fallback(survivors, target_count=target_count)

    def zero_survivor_fallback(self, *, target_count: int) -> list[Agent]:
        return self.make_immigrants(count=target_count)

    def genetic_diversity(self, agents: list[Agent]) -> float:
        if len(agents) < 2:
            return 0.0
        distances = []
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                distances.append(float(np.linalg.norm(agents[i].genome.flatten() - agents[j].genome.flatten())))
        return float(np.mean(distances)) if distances else 0.0

    def compute_generation_statistics(self, agents: list[Agent], generation: int) -> dict[str, Any]:
        ending_capitals = [float(agent.statistics.get("ending_capital", agent.cash)) for agent in agents]
        profit_floor = self.starting_capital * self.survival_threshold
        alive = [agent for agent in agents if agent.status == "ALIVE" and bool(agent.alive) and float(agent.statistics.get("ending_capital", 0.0)) > profit_floor]
        dead = [agent for agent in agents if agent.status == "DEAD"]
        stats = {
            "generation": int(generation),
            "population_size": len(agents),
            "alive_count": len(alive),
            "dead_count": len(dead),
            "survival_rate": float(len(alive) / len(agents)) if agents else 0.0,
            "zero_trade_count": sum(1 for agent in agents if len(agent.trade_history) == 0),
            "mean_ending_capital": float(np.mean(ending_capitals)) if ending_capitals else 0.0,
            "median_ending_capital": float(np.median(ending_capitals)) if ending_capitals else 0.0,
            "best_ending_capital": float(np.max(ending_capitals)) if ending_capitals else 0.0,
            "worst_ending_capital": float(np.min(ending_capitals)) if ending_capitals else 0.0,
            "genetic_diversity": float(self.genetic_diversity(agents)),
            "immigrant_count": sum(1 for agent in agents if agent.statistics.get("immigrant", False)),
            "sexual_children": 0,
            "asexual_children": 0,
            "mutation_count": 0,
            "elite_agent_id": "",
        }
        if alive:
            stats["elite_agent_id"] = max(alive, key=lambda agent: float(agent.statistics.get("ending_capital", agent.cash))).agent_id
        return stats


class EvolutionEngine:
    """Evolution orchestrator that evaluates generations on chronological epochs."""

    def __init__(self, *, seed: int = 42, generation_count: int = 1) -> None:
        self.seed = int(seed)
        self.generation_count = int(generation_count)

    def run_epoch_cycle(self, epochs: list[pd.DataFrame], *, population_size: int = 100, seed: int | None = None, forecaster: Any | None = None) -> Population:
        if not epochs:
            raise ValueError("At least one walk-forward epoch is required to evolve a population.")
        cfg = PopulationConfig(population_size=population_size, seed=int(seed if seed is not None else self.seed))
        population = Population(cfg, forecaster=forecaster)
        selected_epochs = epochs[: self.generation_count or len(epochs)]
        for epoch_index, epoch in enumerate(selected_epochs):
            if epoch.empty:
                raise ValueError(f"Epoch {epoch_index} is empty and cannot be used for evolution.")
            population.evaluate_generation(epoch)
            population.mark_dead_agents()
            if epoch_index + 1 < len(selected_epochs):
                population.reproduce_next_generation()
        return population

    def evaluate_population(self, population: Population, market: pd.DataFrame, forecaster: Any | None = None) -> Population:
        if market.empty:
            raise ValueError("The market epoch for evaluation must not be empty.")
        population.forecaster = forecaster
        population.evaluate_generation(market)
        return population

    def run_generations(self, epochs: list[pd.DataFrame], *, population_size: int = 100, seed: int | None = None, forecaster: Any | None = None) -> Population:
        if not epochs:
            raise ValueError("Evolution requires at least one epoch, and final-test data must never be used for population evolution.")
        cfg = PopulationConfig(population_size=population_size, seed=int(seed if seed is not None else self.seed))
        population = Population(cfg, forecaster=forecaster)
        for epoch_index, epoch in enumerate(epochs):
            if epoch.empty:
                raise ValueError(f"Epoch {epoch_index} is empty and cannot be used for evolution.")
            population.evaluate_generation(epoch)
            population.mark_dead_agents()
            if epoch_index + 1 < len(epochs):
                population.reproduce_next_generation()
        return population


__all__ = ["GenerationStatus", "PopulationConfig", "Population", "EvolutionEngine"]
