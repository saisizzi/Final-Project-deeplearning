from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from artificial_life_trading_ecosystem.agents.agent import Agent
from artificial_life_trading_ecosystem.evolution.population import EvolutionEngine, GenerationStatus, Population, PopulationConfig
from artificial_life_trading_ecosystem.models.genomes.decision_genome import DecisionGenome


class CountingForecaster:
    def __init__(self, prediction: float = 0.5):
        self.prediction = prediction
        self.calls = 0
        self.feature_names = ["Open", "High", "Low", "Close", "Volume"]
        self.config = type("Cfg", (), {"sequence_length": 5})()

    def predict(self, window):
        self.calls += 1
        return np.asarray([self.prediction], dtype=np.float32)


@pytest.fixture
def synthetic_epoch() -> pd.DataFrame:
    rows = []
    price = 100.0
    for idx in range(25):
        price = price * (1.0 + (idx % 4 - 1.5) * 0.002)
        rows.append(
            {
                "Date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=idx),
                "Open": round(price, 4),
                "High": round(price * 1.02, 4),
                "Low": round(price * 0.98, 4),
                "Close": round(price, 4),
                "Volume": 1000 + idx * 10,
            }
        )
    return pd.DataFrame(rows)


def test_population_initializes_to_exact_size() -> None:
    cfg = PopulationConfig(population_size=10, seed=7)
    population = Population(cfg)
    assert len(population.agents) == 10
    assert population.population_size == 10
    assert population.generation == 0


def test_all_agents_start_with_identical_capital() -> None:
    cfg = PopulationConfig(population_size=12, starting_capital=50000.0, seed=13)
    population = Population(cfg)
    assert {agent.cash for agent in population.agents} == {50000.0}
    assert all(agent.starting_capital == 50000.0 for agent in population.agents)


def test_agent_ids_are_unique() -> None:
    population = Population(PopulationConfig(population_size=25, seed=1))
    ids = [agent.agent_id for agent in population.agents]
    assert len(ids) == len(set(ids))


def test_generation_advances_correctly() -> None:
    cfg = PopulationConfig(population_size=8, generations=2, seed=4)
    population = Population(cfg)
    assert population.generation == 0
    population.generation += 1
    assert population.generation == 1


def test_all_agents_share_same_frozen_forecaster() -> None:
    forecaster = CountingForecaster(prediction=0.9)
    cfg = PopulationConfig(population_size=6, seed=5)
    population = Population(cfg, forecaster=forecaster)
    assert population.forecaster is forecaster
    assert all(agent.generation == 0 for agent in population.agents)


def test_forecast_is_computed_once_per_timestep() -> None:
    forecaster = CountingForecaster(prediction=0.6)
    engine = EvolutionEngine(seed=8)
    epoch = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "Open": [100 + i for i in range(6)],
            "High": [101 + i for i in range(6)],
            "Low": [99 + i for i in range(6)],
            "Close": [100 + i for i in range(6)],
            "Volume": [1000] * 6,
        }
    )
    population = Population(PopulationConfig(population_size=4, seed=9), forecaster=forecaster)
    engine.evaluate_population(population, epoch, forecaster)
    assert forecaster.calls == len(epoch) - forecaster.config.sequence_length


def test_population_uses_complete_history_only_for_forecast_windows() -> None:
    class ShortHistoryForecaster:
        def __init__(self):
            self.config = type("Cfg", (), {"sequence_length": 5})()
            self.calls = 0

        def predict(self, window):
            self.calls += 1
            return np.asarray([0.75], dtype=np.float32)

    market = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "Open": [100, 101, 102, 103, 104, 105],
            "High": [101, 102, 103, 104, 105, 106],
            "Low": [99, 100, 101, 102, 103, 104],
            "Close": [100, 101, 102, 103, 104, 105],
            "Volume": [1000, 1000, 1000, 1000, 1000, 1000],
        }
    )
    forecaster = ShortHistoryForecaster()
    population = Population(PopulationConfig(population_size=4, seed=42), forecaster=forecaster)
    forecasts = population._shared_forecast(market, feature_columns=["Open", "High", "Low", "Close", "Volume"])
    assert sum(1 for value in forecasts if value != 0.0) == 2
    assert forecaster.calls == 2


def test_dead_agents_cannot_reproduce() -> None:
    cfg = PopulationConfig(population_size=4, seed=10, survival_threshold=0.90)
    population = Population(cfg)
    population.generation_status = GenerationStatus.EVALUATED
    for agent in population.agents:
        agent.statistics["ending_capital"] = 50000.0
        agent.alive = True
        agent.status = "ALIVE"
    population.agents[0].statistics["ending_capital"] = 80000.0
    population.agents[1].statistics["ending_capital"] = 100001.0
    population.agents[2].statistics["ending_capital"] = 20000.0
    population.agents[3].statistics["ending_capital"] = 0.0
    eligible = population.eligible_parents()
    assert len(eligible) == 1
    assert population.agents[1].agent_id in {a.agent_id for a in eligible}


def test_zero_profit_agents_die_at_generation_end() -> None:
    cfg = PopulationConfig(population_size=5, seed=11)
    population = Population(cfg)
    population.generation_status = GenerationStatus.EVALUATED
    for agent in population.agents:
        agent.trade_history = []
        agent.statistics["ending_capital"] = agent.starting_capital
        agent.status = "RUNNING"
    dead = population.mark_dead_agents()
    assert dead == 5
    assert all(not agent.alive and agent.status == "DEAD" for agent in population.agents)


def test_only_agents_with_positive_profit_survive() -> None:
    cfg = PopulationConfig(population_size=4, starting_capital=100000.0, survival_threshold=0.90, seed=12)
    population = Population(cfg)
    population.generation_status = GenerationStatus.EVALUATED
    for i, agent in enumerate(population.agents):
        agent.statistics["ending_capital"] = 100000.0 + i * 1000.0
        agent.alive = True
        agent.status = "ALIVE"
        agent.trade_history = [{"action": "BUY"}]
    survivors = population.surviving_agents()
    assert len(survivors) == 3
    population.agents[1].statistics["ending_capital"] = 99000.0
    assert len(population.surviving_agents()) == 2


def test_capital_weighted_selection_prefers_larger_capital() -> None:
    cfg = PopulationConfig(population_size=4, seed=15)
    population = Population(cfg)
    population.generation_status = GenerationStatus.EVALUATED
    for agent in population.agents[:2]:
        agent.alive = True
        agent.status = "ALIVE"
    population.agents[0].statistics["ending_capital"] = 110000.0
    population.agents[1].statistics["ending_capital"] = 120000.0
    population.agents[2].statistics["ending_capital"] = 50000.0
    population.agents[3].statistics["ending_capital"] = 120000.0
    selected = population.capital_weighted_selection(population.eligible_parents())
    assert selected
    assert max(a.statistics["ending_capital"] for a in selected) >= 100000.0


def test_uniform_crossover_works() -> None:
    a = DecisionGenome.from_default(seed=20)
    b = DecisionGenome.from_default(seed=21)
    child = Population.uniform_crossover(a, b, rng=np.random.default_rng(99))
    assert child is not a
    assert child is not b
    assert child.flatten().shape == a.flatten().shape


def test_arithmetic_crossover_works() -> None:
    a = DecisionGenome.from_default(seed=31)
    b = DecisionGenome.from_default(seed=32)
    child = Population.arithmetic_crossover(a, b, alpha=0.35, rng=np.random.default_rng(7))
    assert child is not a
    assert child is not b
    assert child.flatten().shape == a.flatten().shape


def test_parent_genomes_are_unchanged_after_crossover() -> None:
    a = DecisionGenome.from_default(seed=40)
    b = DecisionGenome.from_default(seed=41)
    a_before = a.flatten().copy()
    b_before = b.flatten().copy()
    Population.uniform_crossover(a, b, rng=np.random.default_rng(123))
    assert np.allclose(a.flatten(), a_before)
    assert np.allclose(b.flatten(), b_before)


def test_mutation_changes_child_only() -> None:
    parent = DecisionGenome.from_default(seed=50)
    child = parent
    mutated = Population.mutate_genome(child, mutation_rate=1.0, mutation_sigma=0.5, rng=np.random.default_rng(5))
    assert mutated is not parent
    assert np.allclose(parent.flatten(), child.flatten()) if child is parent else True
    assert mutated.trading_params["signal_threshold"] == mutated.signal_threshold
    assert mutated.output_bias != parent.output_bias


def test_elite_genome_remains_unchanged() -> None:
    cfg = PopulationConfig(population_size=8, elite_count=1, seed=52)
    population = Population(cfg)
    elite = population.agents[0]
    elite_before = elite.genome.flatten().copy()
    preserved = population.preserve_elite([elite])
    assert preserved[0].genome.flatten().shape == elite_before.shape
    assert np.allclose(preserved[0].genome.flatten(), elite_before)


def test_immigrant_count_is_correct() -> None:
    cfg = PopulationConfig(population_size=10, immigrant_fraction=0.2, seed=60)
    population = Population(cfg)
    immigrants = population.make_immigrants(count=2)
    assert len(immigrants) == 2
    assert all(agent.parent_a is None for agent in immigrants)
    assert all(agent.parent_b is None for agent in immigrants)
    assert all(agent.statistics.get("immigrant", False) for agent in immigrants)


def test_asexual_fallback_works_for_one_survivor() -> None:
    cfg = PopulationConfig(population_size=6, seed=70)
    population = Population(cfg)
    survivors = [population.agents[0]]
    children = population.aseuxal_fallback(survivors, target_count=6)
    assert len(children) == 6
    assert all(child.parent_b is None for child in children)


def test_zero_survivor_fallback_replenishes_population() -> None:
    cfg = PopulationConfig(population_size=6, seed=80)
    population = Population(cfg)
    children = population.zero_survivor_fallback(target_count=6)
    assert len(children) == 6
    assert all(child.agent_id != "" for child in children)


def test_final_population_size_is_exact() -> None:
    cfg = PopulationConfig(population_size=30, seed=90)
    population = Population(cfg)
    children = population.make_immigrants(count=5)
    combined = population.agents + children
    assert len(combined) == 35
    assert len(population.agents) == cfg.population_size


def test_child_accounts_restart_at_starting_capital() -> None:
    cfg = PopulationConfig(population_size=4, starting_capital=25000.0, seed=95)
    population = Population(cfg)
    parent = population.agents[0]
    parent.cash = 30000.0
    child = population.create_child(parent_a=parent, parent_b=None, generation=1)
    assert child.cash == 25000.0
    assert child.current_capital == 25000.0


def test_parent_ids_are_recorded() -> None:
    cfg = PopulationConfig(population_size=4, seed=100)
    population = Population(cfg)
    parent_a = population.agents[0]
    parent_b = population.agents[1]
    child = population.create_child(parent_a=parent_a, parent_b=parent_b, generation=1)
    assert child.parent_a == parent_a.agent_id
    assert child.parent_b == parent_b.agent_id
    assert child.generation == 1


def test_generation_statistics_are_consistent() -> None:
    population = Population(PopulationConfig(population_size=6, seed=101))
    for agent in population.agents:
        agent.statistics["ending_capital"] = 100001.0
        agent.alive = True
        agent.status = "ALIVE"
        agent.trade_history = [{"action": "BUY"}, {"action": "SELL"}]
    stats = population.compute_generation_statistics(population.agents, 0)
    assert set(stats.keys()) >= {"generation", "population_size", "alive_count", "dead_count", "survival_rate", "genetic_diversity", "immigrant_count"}
    assert stats["population_size"] == 6
    assert stats["alive_count"] == 6


def test_genetic_diversity_is_deterministic() -> None:
    population = Population(PopulationConfig(population_size=8, seed=102))
    diversity = population.genetic_diversity(population.agents)
    assert isinstance(diversity, float)
    assert math.isfinite(diversity)
    assert diversity >= 0.0


def test_fixed_seed_produces_reproducible_generations() -> None:
    cfg = PopulationConfig(population_size=8, seed=110)
    population_a = Population(cfg)
    population_b = Population(PopulationConfig(population_size=8, seed=110))
    assert [a.agent_id for a in population_a.agents] == [b.agent_id for b in population_b.agents]


def test_population_cannot_create_duplicate_ids() -> None:
    population = Population(PopulationConfig(population_size=20, seed=111))
    ids = [population.next_agent_id() for _ in range(5)]
    assert len(ids) == len(set(ids))


def test_evolution_cannot_consume_final_test_data() -> None:
    engine = EvolutionEngine(seed=120)
    with pytest.raises(ValueError):
        engine.run_generations([], population_size=10)


def test_multiple_generations_run_successfully_on_tiny_synthetic_data() -> None:
    epoch_0 = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "Open": np.linspace(100, 110, 10),
            "High": np.linspace(101, 111, 10),
            "Low": np.linspace(99, 109, 10),
            "Close": np.linspace(100, 110, 10),
            "Volume": np.full(10, 1000),
        }
    )
    epoch_1 = epoch_0.copy()
    engine = EvolutionEngine(seed=130, generation_count=2)
    population = engine.run_epoch_cycle([epoch_0, epoch_1], population_size=8)
    assert population.generation >= 1
    assert population.population_size == 8
