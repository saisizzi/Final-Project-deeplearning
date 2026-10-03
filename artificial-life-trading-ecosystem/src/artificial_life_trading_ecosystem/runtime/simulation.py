"""Persistent day-by-day evolution runtime isolated from canonical results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import pandas as pd

from artificial_life_trading_ecosystem.agents.agent import Agent
from artificial_life_trading_ecosystem.evolution.population import GenerationStatus, Population, PopulationConfig
from artificial_life_trading_ecosystem.lineage.lineage import LineageRecord
from artificial_life_trading_ecosystem.models.genomes.decision_genome import DecisionGenome


class InteractiveSimulation:
    """Advance one development-only population one trading day at a time."""

    generation_days = 10

    def __init__(
        self,
        *,
        development_market: pd.DataFrame,
        history_prefix: pd.DataFrame,
        forecaster: Any,
        state_path: str | Path,
        population_size: int = 10,
        starting_capital: float = 100000.0,
        survival_threshold: float = 1.0,
        mutation_rate: float = 0.05,
        mutation_sigma: float = 0.10,
        seed: int = 42,
        feature_columns: list[str] | None = None,
        experiment_id: str = "local-evolution-development:SimpleRNN",
        load_existing: bool = True,
    ) -> None:
        if len(development_market) < self.generation_days:
            raise ValueError("Evolution-development data must contain at least 10 trading days.")
        if development_market.attrs.get("split") == "test" or development_market.attrs.get("final_test"):
            raise ValueError("Interactive evolution cannot consume final-test data.")
        self.development_market = development_market.reset_index(drop=True).copy()
        self.history_prefix = history_prefix.reset_index(drop=True).copy()
        self.forecaster = forecaster
        self.experiment_id = str(experiment_id)
        self.state_path = Path(state_path)
        self.feature_columns = feature_columns or [
            name for name in getattr(forecaster, "feature_names", []) if name in self.development_market.columns
        ] or [name for name in self.development_market.columns if name not in {"Date", "date"}]
        self.population_config = {
            "population_size": int(population_size),
            "starting_capital": float(starting_capital),
            "survival_threshold": float(survival_threshold),
            "mutation_rate": float(mutation_rate),
            "mutation_sigma": float(mutation_sigma),
            "seed": int(seed),
        }
        self.lock = RLock()
        self.population = self._new_population()
        self.development_cursor = 0
        self.completed_generations: list[dict[str, Any]] = []
        self.current_day_events: list[dict[str, Any]] = []
        self.latest_decisions: dict[str, dict[str, Any]] = {}
        self.strategy_curves: dict[str, list[float]] = {"manual_baseline": [], "buy_and_hold": []}
        self._buy_hold_quantity: float | None = None
        self.legacy_mutations_through_generation = -1
        if load_existing and self.state_path.exists():
            self._load_state()

    def _new_population(self) -> Population:
        return Population(PopulationConfig(generation_trading_days=self.generation_days, **self.population_config), forecaster=self.forecaster, seed=self.population_config["seed"])

    @property
    def generation_day(self) -> int:
        return self.development_cursor % self.generation_days

    @property
    def generation_status(self) -> GenerationStatus:
        return self.population.generation_status

    def _serialize_genome(self, genome: DecisionGenome) -> dict[str, Any]:
        return {
            "context_length": genome.context_length,
            "hidden_units": genome.hidden_units,
            "activation": genome.activation,
            "input_to_hidden": genome.input_to_hidden.tolist(),
            "hidden_bias": genome.hidden_bias.tolist(),
            "hidden_to_output": genome.hidden_to_output.tolist(),
            "output_bias": genome.output_bias,
            "signal_threshold": genome.signal_threshold,
            "position_size": genome.position_size,
            "stop_loss": genome.stop_loss,
            "take_profit": genome.take_profit,
            "transaction_cost": genome.transaction_cost,
        }

    def _deserialize_genome(self, payload: dict[str, Any]) -> DecisionGenome:
        return DecisionGenome(**payload)

    def _serialize_agent(self, agent: Agent) -> dict[str, Any]:
        return {
            "agent_id": agent.agent_id,
            "generation": agent.generation,
            "parent_a": agent.parent_a,
            "parent_b": agent.parent_b,
            "genome": self._serialize_genome(agent.genome),
            "starting_capital": agent.starting_capital,
            "cash": agent.cash,
            "current_capital": agent.current_capital,
            "current_position": agent.current_position,
            "entry_price": agent.entry_price,
            "position_quantity": agent.position_quantity,
            "trade_history": self._json_value(agent.trade_history),
            "equity_history": agent.equity_history,
            "alive": agent.alive,
            "status": agent.status,
            "statistics": agent.statistics,
            "config": agent.config,
        }

    def _deserialize_agent(self, payload: dict[str, Any]) -> Agent:
        restored = dict(payload)
        restored["genome"] = self._deserialize_genome(restored["genome"])
        return Agent(**restored)

    def _serialize_lineage(self, *, display_legacy_status: bool = True) -> list[dict[str, Any]]:
        return [
            {
                "agent_id": record.agent_id,
                "parent_a": record.parent_a,
                "parent_b": record.parent_b,
                "parent_ids": record.parent_ids,
                "offspring_ids": record.offspring_ids,
                "generation": record.generation,
                "reproduction_method": record.reproduction_method,
                "mutation_applied": (
                    self._mutation_status(record.generation, record.mutation_applied)
                    if display_legacy_status
                    else record.mutation_applied
                ),
                "immigrant": record.immigrant,
                "status": record.status,
                "metadata": record.metadata,
            }
            for record in self.population.lineage.values()
        ]

    def _mutation_status(self, generation: int, mutation_applied: bool) -> bool | None:
        if 0 < generation <= self.legacy_mutations_through_generation:
            return None
        return mutation_applied

    def _persisted_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "data_split": "evolution-development",
            "final_test_access": False,
            "population_config": self.population_config,
            "population_generation": self.population.generation,
            "population_status": self.population.generation_status.value,
            "population_agent_counter": self.population._agent_counter,
            "development_cursor": self.development_cursor,
            "legacy_mutations_through_generation": self.legacy_mutations_through_generation,
            "completed_generations": self.completed_generations,
            "current_day_events": self.current_day_events,
            "latest_decisions": self.latest_decisions,
            "strategy_curves": self.strategy_curves,
            "buy_hold_quantity": self._buy_hold_quantity,
            "manual_baseline": self._serialize_agent(self.manual_baseline) if hasattr(self, "manual_baseline") else None,
            "agents": [self._serialize_agent(agent) for agent in self.population.agents],
            "lineage": self._serialize_lineage(display_legacy_status=False),
        }

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): InteractiveSimulation._json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [InteractiveSimulation._json_value(item) for item in value]
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, (pd.Timestamp, np.datetime64)):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def _persist(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(self._json_value(self._persisted_payload()), indent=2), encoding="utf-8")
        os.replace(temp_path, self.state_path)

    def _load_state(self) -> None:
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if (
            payload.get("data_split") != "evolution-development"
            or payload.get("final_test_access") is not False
            or payload.get("experiment_id") != self.experiment_id
        ):
            raise ValueError("Persisted interactive state has an invalid data-split boundary.")
        self.population_config = payload["population_config"]
        self.population = self._new_population()
        self.population.generation = int(payload["population_generation"])
        self.population.generation_status = GenerationStatus(payload["population_status"])
        self.population._agent_counter = int(payload["population_agent_counter"])
        self.population.agents = [self._deserialize_agent(agent) for agent in payload["agents"]]
        self.population.lineage = {
            item["agent_id"]: LineageRecord(**item) for item in payload.get("lineage", [])
        }
        self.legacy_mutations_through_generation = int(
            payload.get("legacy_mutations_through_generation", self.population.generation)
        )
        self.development_cursor = int(payload["development_cursor"])
        self.completed_generations = payload.get("completed_generations", [])
        for generation in self.completed_generations:
            generation_number = int(generation["generation"])
            if 0 < generation_number <= self.legacy_mutations_through_generation:
                for agent in generation.get("agents", []):
                    agent["mutation_applied"] = None
            if (
                0 < generation_number + 1 <= self.legacy_mutations_through_generation
                and generation.get("metrics") is not None
            ):
                generation["metrics"]["mutation_count"] = None
        self.current_day_events = payload.get("current_day_events", [])
        self.latest_decisions = payload.get("latest_decisions", {})
        self.strategy_curves = payload.get("strategy_curves", {"manual_baseline": [], "buy_and_hold": []})
        self._buy_hold_quantity = payload.get("buy_hold_quantity")
        if payload.get("manual_baseline"):
            from artificial_life_trading_ecosystem.agents.baseline import ManualBaselineAgent

            self.manual_baseline = ManualBaselineAgent(**{
                **payload["manual_baseline"],
                "genome": self._deserialize_genome(payload["manual_baseline"]["genome"]),
            })
        if self.generation_status == GenerationStatus.RUNNING:
            for agent in self.population.agents:
                if agent.statistics.get("dead_reason") == "threshold":
                    agent.alive = True
                    agent.status = "RUNNING"
                    agent.statistics.pop("dead_reason", None)
                    lineage_record = self.population.lineage.get(agent.agent_id)
                    if lineage_record is not None:
                        lineage_record.status = "RUNNING"
                    decision = self.latest_decisions.get(agent.agent_id)
                    if decision is not None:
                        decision.pop("dead_reason", None)
                        decision["status"] = "RUNNING"

    def reset(self) -> dict[str, Any]:
        with self.lock:
            self.population = self._new_population()
            self.development_cursor = 0
            self.completed_generations = []
            self.current_day_events = []
            self.latest_decisions = {}
            self.legacy_mutations_through_generation = -1
            self.strategy_curves = {"manual_baseline": [], "buy_and_hold": []}
            self._buy_hold_quantity = None
            if hasattr(self, "manual_baseline"):
                del self.manual_baseline
            self._persist()
            return self.get_state()

    def _current_history(self) -> pd.DataFrame:
        return pd.concat(
            [self.history_prefix, self.development_market.iloc[:self.development_cursor]],
            ignore_index=True,
        )

    def _generation_baseline_step(self, row: pd.Series, forecast: float, context: list[float]) -> None:
        if not hasattr(self, "manual_baseline") or self.manual_baseline.generation != self.population.generation:
            from artificial_life_trading_ecosystem.agents.baseline import ManualBaselineAgent

            self.manual_baseline = ManualBaselineAgent(
                agent_id="manual-baseline",
                generation=self.population.generation,
                starting_capital=self.population.starting_capital,
                config={"signal_threshold": 0.2},
            )
        if not self.strategy_curves["manual_baseline"]:
            self.strategy_curves["manual_baseline"].append(float(self.population.starting_capital))
        result = self.population.engine.simulate_daily_step(self.manual_baseline, row, forecast, context)
        self.strategy_curves["manual_baseline"].append(float(result["capital"]))
        if self._buy_hold_quantity is None:
            open_price = float(row.get("Open", row.get("Close", 0.0)))
            entry_cost = 1.0 + self.population.agents[0].transaction_cost if self.population.agents else 1.0
            self._buy_hold_quantity = self.population.starting_capital / (open_price * entry_cost) if open_price > 0 else 0.0
            self.strategy_curves["buy_and_hold"].append(float(self.population.starting_capital))
        self.strategy_curves["buy_and_hold"].append(float(self._buy_hold_quantity * float(row.get("Close", 0.0))))

    def next_day(self) -> dict[str, Any]:
        with self.lock:
            if self.generation_status == GenerationStatus.EVALUATED:
                raise RuntimeError("Evaluated generation must reproduce before another day can run.")
            if self.development_cursor >= len(self.development_market):
                raise RuntimeError("Evolution-development data is exhausted; final-test data is not available to interactive simulation.")
            if self.generation_status == GenerationStatus.READY and self.development_cursor + self.generation_days > len(self.development_market):
                raise RuntimeError("Not enough remaining evolution-development days to run a complete 10-day generation.")
            if self.generation_status == GenerationStatus.READY:
                self.population.generation_status = GenerationStatus.RUNNING
                for agent in self.population.agents:
                    if agent.status != "DEAD":
                        agent.status = "RUNNING"

            row = self.development_market.iloc[self.development_cursor]
            history = self._current_history()
            forecast = self.population.engine._forecast_for_day(
                history,
                len(history) - 1,
                self.forecaster,
                feature_columns=self.feature_columns,
            ) if len(history) else 0.0
            context = self.population.engine._normalized_context(history, len(history) - 1) if len(history) else [0.0] * self.population.engine.context_length
            decisions = []
            for agent in self.population.agents:
                if agent.status == "DEAD" or not agent.alive:
                    continue
                decision = self.population.engine.simulate_daily_step(agent, row, forecast, context)
                decision["status"] = "RUNNING"
                decisions.append(decision)
            self._generation_baseline_step(row, forecast, context)
            action_counts = {action: sum(entry["action"] == action for entry in decisions) for action in ("BUY", "HOLD", "SELL")}
            capitals = [float(agent.cash) for agent in self.population.agents]
            day_number = self.generation_day + 1
            daily_event = {
                "generation": self.population.generation,
                "day": day_number,
                "date": str(row.get("Date", row.get("date", row.name))),
                "counts": action_counts,
                "trade_count": sum(len(entry["trades"]) for entry in decisions),
                "best_capital": max(capitals, default=0.0),
                "mean_capital": float(np.mean(capitals)) if capitals else 0.0,
                "worst_capital": min(capitals, default=0.0),
                "agents": decisions,
            }
            self.current_day_events.append(daily_event)
            self.population.daily_events.append(daily_event)
            self.population.day_count = day_number
            self.latest_decisions = {item["agent_id"]: item for item in decisions}
            self.development_cursor += 1

            if day_number == self.generation_days:
                self._evaluate_and_reproduce()
            self._persist()
            return self.get_state()

    def _evaluate_and_reproduce(self) -> None:
        for agent in self.population.agents:
            agent.statistics["ending_capital"] = float(agent.cash)
            agent.statistics["survival_status"] = "EVALUATED"
        self.population.generation_status = GenerationStatus.EVALUATED
        self.population.mark_dead_agents()
        stats = self.population.compute_generation_statistics(self.population.agents, self.population.generation)
        stats["profit_making_parent_count"] = len(self.population.surviving_agents())
        ranked = sorted(self.population.agents, key=lambda item: float(item.statistics["ending_capital"]), reverse=True)
        completed_generation = {
            "generation": self.population.generation,
            "status": GenerationStatus.EVALUATED.value,
            "day": self.generation_days,
            "metrics": stats,
            "agents": [
                self._agent_display(
                    agent,
                    self.population.generation,
                    self.latest_decisions.get(agent.agent_id),
                    evaluated=True,
                )
                for agent in self.population.agents
            ],
            "daily_events": self.current_day_events,
            "best_agent_id": ranked[0].agent_id if ranked else None,
            "strategy_curves": self.strategy_curves,
        }
        self.completed_generations.append(completed_generation)
        self.current_day_events = []
        self.strategy_curves = {"manual_baseline": [], "buy_and_hold": []}
        self._buy_hold_quantity = None
        children = self.population.reproduce_next_generation()
        stats["mutation_count"] = sum(
            self.population.lineage[child.agent_id].mutation_applied
            for child in children
        )
        stats["offspring_count"] = len(children)
        stats["sexual_children"] = sum(
            self.population.lineage[child.agent_id].reproduction_method == "sexual"
            for child in children
        )
        stats["asexual_children"] = sum(
            self.population.lineage[child.agent_id].reproduction_method == "asexual"
            for child in children
        )
        stats["immigrant_count"] = sum(
            bool(child.statistics.get("immigrant", False))
            for child in children
        )
        self.latest_decisions = {}

    def run_current_generation(self) -> dict[str, Any]:
        with self.lock:
            start_generation = self.population.generation
            if self.generation_status == GenerationStatus.EVALUATED:
                return self.get_state()
            while self.population.generation == start_generation and self.generation_status != GenerationStatus.EVALUATED:
                self.next_day()
            return self.get_state()

    def run_generations(self, count: int) -> dict[str, Any]:
        if count <= 0 or count > 100:
            raise ValueError("Generation count must be between 1 and 100.")
        with self.lock:
            start_generation = self.population.generation
            target_generation = start_generation + count
            while len(self.completed_generations) < target_generation:
                if self.development_cursor + self.generation_days > len(self.development_market):
                    raise RuntimeError("Not enough remaining evolution-development days to run the requested generations.")
                self.run_current_generation()
            return self.get_state()

    def _agent_display(self, agent: Agent, generation: int, latest: dict[str, Any] | None, *, evaluated: bool) -> dict[str, Any]:
        lineage_record = self.population.lineage.get(agent.agent_id)
        return {
            "agent_id": agent.agent_id,
            "generation": generation,
            "status": agent.status,
            "alive": agent.status == "ALIVE",
            "starting_capital": agent.starting_capital,
            "ending_capital": float(agent.statistics["ending_capital"]) if evaluated else None,
            "capital": float(agent.cash),
            "forecast": latest.get("forecast") if latest else None,
            "decision_score": latest.get("decision_score") if latest else None,
            "threshold": float(agent.threshold),
            "action": latest.get("action", "READY" if not evaluated else "") if latest else ("READY" if not evaluated else ""),
            "position": float(agent.position_quantity),
            "daily_pnl": float(latest.get("daily_pnl", 0.0)) if latest else 0.0,
            "trade_count": len(agent.trade_history),
            "trade_history": agent.trade_history,
            "equity_curve": agent.equity_history,
            "parent_a": agent.parent_a,
            "parent_b": agent.parent_b,
            "immigrant": bool(agent.statistics.get("immigrant", False)),
            "elite": bool(agent.statistics.get("elite", False)),
            "dead_reason": agent.statistics.get("dead_reason"),
            "mutation_applied": self._mutation_status(
                lineage_record.generation,
                lineage_record.mutation_applied,
            ) if lineage_record else False,
        }

    def get_state(self) -> dict[str, Any]:
        with self.lock:
            current_generation = self.population.generation
            current_status = self.generation_status.value
            evaluated = current_status == GenerationStatus.EVALUATED.value
            day = self.generation_day if not evaluated else self.generation_days
            capitals = [float(agent.cash) for agent in self.population.agents]
            current_agents = [
                self._agent_display(agent, current_generation, self.latest_decisions.get(agent.agent_id), evaluated=evaluated)
                for agent in self.population.agents
            ]
            generations = list(self.completed_generations)
            if not evaluated or not generations or generations[-1]["generation"] != current_generation:
                generations.append(
                    {
                        "generation": current_generation,
                        "status": current_status,
                        "day": day,
                        "metrics": None,
                        "agents": current_agents,
                        "daily_events": self.current_day_events,
                    }
                )
            latest_day = self.current_day_events[-1] if self.current_day_events else None
            return {
                "experiment_id": self.experiment_id,
                "data_split": "evolution-development",
                "final_test_access": False,
                "generation": current_generation,
                "status": current_status,
                "day": day,
                "generation_days": self.generation_days,
                "date": latest_day["date"] if latest_day else None,
                "counts": latest_day["counts"] if latest_day else {"BUY": 0, "HOLD": 0, "SELL": 0},
                "trade_count": latest_day["trade_count"] if latest_day else 0,
                "best_capital": max(capitals, default=0.0),
                "mean_capital": float(np.mean(capitals)) if capitals else 0.0,
                "worst_capital": min(capitals, default=0.0),
                "agents": current_agents,
                "generations": generations,
                "daily_events": self.current_day_events,
                "lineage": self._serialize_lineage(),
                "strategy_curves": self.strategy_curves,
                "remaining_development_days": len(self.development_market) - self.development_cursor,
                "population_config": {
                    "population_size": self.population.population_size,
                    "starting_capital": self.population.starting_capital,
                    "survival_threshold": self.population.survival_threshold,
                    "mutation_rate": self.population.mutation_rate,
                    "mutation_sigma": self.population.mutation_sigma,
                    "immigrant_fraction": self.population.immigrant_fraction,
                    "seed": self.population.seed,
                },
            }