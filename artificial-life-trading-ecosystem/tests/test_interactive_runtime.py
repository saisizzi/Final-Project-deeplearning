from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from artificial_life_trading_ecosystem.evolution.population import GenerationStatus
from artificial_life_trading_ecosystem.runtime.simulation import InteractiveSimulation


class SyntheticForecaster:
    feature_names = ["Open", "High", "Low", "Close", "Volume"]
    config = type("Config", (), {"sequence_length": 5})()

    def __init__(self) -> None:
        self.windows: list[np.ndarray] = []

    def predict(self, window: np.ndarray) -> np.ndarray:
        self.windows.append(np.asarray(window).copy())
        return np.asarray([0.5], dtype=np.float32)


def market_frame(count: int, *, start: int = 100) -> pd.DataFrame:
    close = np.arange(start, start + count, dtype=float)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=count, freq="B"),
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.full(count, 1000.0),
        }
    )


def make_simulation(tmp_path, days: int = 30, *, load_existing: bool = True) -> InteractiveSimulation:
    forecaster = SyntheticForecaster()
    prefix = market_frame(5, start=90)
    development = market_frame(days, start=100)
    simulation = InteractiveSimulation(
        development_market=development,
        history_prefix=prefix,
        forecaster=forecaster,
        state_path=tmp_path / "interactive" / "simulation-state.json",
        population_size=4,
        seed=19,
        load_existing=load_existing,
    )
    for agent in simulation.population.agents:
        agent.genome.input_to_hidden.fill(0.0)
        agent.genome.hidden_to_output.fill(0.0)
        agent.genome.output_bias = 0.5
    return simulation


def test_initial_state_is_ready_and_not_evaluated(tmp_path) -> None:
    state = make_simulation(tmp_path).get_state()

    assert state["status"] == "READY"
    assert state["day"] == 0
    assert state["generation_days"] == 10
    assert state["trade_count"] == 0
    assert all(agent["status"] == "READY" for agent in state["agents"])
    assert all(agent["ending_capital"] is None for agent in state["agents"])
    assert all(agent["alive"] is False for agent in state["agents"])


def test_next_day_uses_prior_data_and_records_real_decisions(tmp_path) -> None:
    simulation = make_simulation(tmp_path)
    state = simulation.next_day()

    assert state["status"] == "RUNNING"
    assert state["day"] == 1
    assert state["date"] == str(simulation.development_market.iloc[0]["Date"])
    assert state["counts"] == {"BUY": 4, "HOLD": 0, "SELL": 0}
    assert state["trade_count"] == 8
    assert simulation.forecaster.windows[0].shape == (1, 5, 5)
    assert float(simulation.forecaster.windows[0][0, -1, 3]) == 94.0
    assert all(agent["position"] == 0.0 for agent in state["agents"])
    assert all(agent["trade_count"] == 2 for agent in state["agents"])


def test_unprofitable_agents_die_at_generation_end_and_are_replaced(tmp_path) -> None:
    simulation = make_simulation(tmp_path)
    market = market_frame(30)
    market.loc[0, ["Open", "High", "Low", "Close"]] = [100.0, 101.0, 79.0, 80.0]
    market.loc[1:9, ["Open", "High", "Low", "Close"]] = 80.0
    simulation.development_market = market
    for agent in simulation.population.agents:
        agent.genome.position_size = 1.0
        agent.genome.stop_loss = 0.5
        agent.genome.trading_params["position_size"] = 1.0
        agent.genome.trading_params["stop_loss"] = 0.5

    first_day = simulation.next_day()
    assert all(agent["status"] == "RUNNING" for agent in first_day["agents"])
    assert all(agent["capital"] < agent["starting_capital"] * 0.9 for agent in first_day["agents"])
    trade_counts = {agent["agent_id"]: agent["trade_count"] for agent in first_day["agents"]}

    second_day = simulation.next_day()
    assert len(second_day["daily_events"][-1]["agents"]) == 4
    assert all(agent["status"] == "RUNNING" for agent in second_day["agents"])
    assert all(agent["trade_count"] > trade_counts[agent["agent_id"]] for agent in second_day["agents"])

    for _ in range(8):
        completed = simulation.next_day()
    generation_zero = next(item for item in completed["generations"] if item["generation"] == 0)
    assert generation_zero["metrics"]["alive_count"] == 0
    assert generation_zero["metrics"]["dead_count"] == 4
    assert all(agent["status"] == "DEAD" for agent in generation_zero["agents"])
    assert all(agent["immigrant"] for agent in completed["agents"])
    assert all(agent["dead_reason"] == "no_profit" for agent in generation_zero["agents"])


def test_reload_keeps_unprofitable_agents_active_until_generation_evaluation(tmp_path) -> None:
    simulation = make_simulation(tmp_path)
    simulation.development_cursor = 1
    simulation.population.generation_status = GenerationStatus.RUNNING
    agent = simulation.population.agents[0]
    agent.cash = 80000.0
    agent.current_capital = 80000.0
    agent.status = "RUNNING"
    agent.alive = True
    simulation._persist()

    restored = InteractiveSimulation(
        development_market=simulation.development_market,
        history_prefix=simulation.history_prefix,
        forecaster=SyntheticForecaster(),
        state_path=simulation.state_path,
        population_size=4,
        seed=19,
    )
    assert restored.get_state()["agents"][0]["status"] == "RUNNING"

    next_day = restored.next_day()
    assert "A001" in {decision["agent_id"] for decision in next_day["daily_events"][-1]["agents"]}
    assert next_day["agents"][0]["status"] == "RUNNING"
    assert next_day["agents"][0]["trade_count"] > 0


def test_profit_makers_parent_replacements_and_real_mutations_are_recorded(tmp_path) -> None:
    simulation = make_simulation(tmp_path)
    simulation.population.mutation_rate = 1.0
    profitable_ids = {simulation.population.agents[0].agent_id, simulation.population.agents[1].agent_id}
    for index, agent in enumerate(simulation.population.agents):
        ending_capital = 110000.0 - index * 5000.0
        agent.cash = ending_capital
        agent.current_capital = ending_capital
        agent.statistics["ending_capital"] = ending_capital
        agent.status = "RUNNING"
        agent.alive = True

    simulation._evaluate_and_reproduce()

    generation_zero = simulation.completed_generations[0]
    assert generation_zero["metrics"]["dead_count"] == 2
    assert generation_zero["metrics"]["profit_making_parent_count"] == 2
    children = simulation.population.agents
    assert len(children) == 4
    for child in children:
        lineage = simulation.population.lineage[child.agent_id]
        assert set(lineage.parent_ids).issubset(profitable_ids)
    mutated_children = [child for child in children if simulation.population.lineage[child.agent_id].mutation_applied]
    assert generation_zero["metrics"]["mutation_count"] == len(mutated_children)
    assert 0 <= len(mutated_children) < 4
    assert all(child["mutation_applied"] for child in simulation.get_state()["agents"] if child["agent_id"] in {agent.agent_id for agent in mutated_children})


def test_legacy_mutation_records_are_not_reported_as_verified(tmp_path) -> None:
    simulation = make_simulation(tmp_path, load_existing=False)
    parent = simulation.population.agents[0]
    child = simulation.population.create_child(parent_a=parent, parent_b=None, generation=1)
    simulation.population.lineage[child.agent_id].mutation_applied = True
    simulation.population.agents = [child]
    simulation.population.generation = 1
    simulation.completed_generations = [{
        "generation": 0,
        "status": "EVALUATED",
        "day": 10,
        "metrics": {"mutation_count": 0},
        "agents": [{"agent_id": child.agent_id, "mutation_applied": True}],
        "daily_events": [],
    }]
    simulation._persist()

    payload = json.loads(simulation.state_path.read_text(encoding="utf-8"))
    payload.pop("legacy_mutations_through_generation")
    simulation.state_path.write_text(json.dumps(payload), encoding="utf-8")

    restored = InteractiveSimulation(
        development_market=simulation.development_market,
        history_prefix=simulation.history_prefix,
        forecaster=SyntheticForecaster(),
        state_path=simulation.state_path,
        population_size=4,
        seed=19,
    )
    state = restored.get_state()

    assert state["agents"][0]["mutation_applied"] is None
    assert next(item for item in state["lineage"] if item["agent_id"] == child.agent_id)["mutation_applied"] is None
    assert state["generations"][0]["metrics"]["mutation_count"] is None
    restored._persist()
    persisted_lineage = json.loads(simulation.state_path.read_text(encoding="utf-8"))["lineage"]
    assert next(item for item in persisted_lineage if item["agent_id"] == child.agent_id)["mutation_applied"] is True


def test_generation_evaluates_then_reproduces_ready_children(tmp_path) -> None:
    simulation = make_simulation(tmp_path)
    state = simulation.run_current_generation()

    assert state["generation"] == 1
    assert state["status"] == "READY"
    assert state["day"] == 0
    previous = state["generations"][0]
    assert previous["status"] == "EVALUATED"
    assert previous["day"] == 10
    profitable = [
        agent for agent in previous["agents"]
        if agent["ending_capital"] > agent["starting_capital"]
    ]
    assert previous["metrics"]["alive_count"] == len(profitable)
    assert previous["metrics"]["dead_count"] == len(previous["agents"]) - len(profitable)
    assert all(agent["status"] == ("ALIVE" if agent in profitable else "DEAD") for agent in previous["agents"])
    assert all(agent["status"] == "READY" for agent in state["agents"])
    assert all(agent["ending_capital"] is None for agent in state["agents"])
    assert all(agent["trade_count"] == 0 for agent in state["agents"])
    assert len({agent["agent_id"] for agent in state["agents"]}) == 4
    lineage = {record["agent_id"]: record for record in state["lineage"]}
    assert all(lineage[agent["agent_id"]]["status"] == agent["status"] for agent in previous["agents"])
    assert all(lineage[agent["agent_id"]]["status"] == "READY" for agent in state["agents"])
    assert all(lineage[parent]["offspring_ids"] for parent in (agent["parent_a"] for agent in state["agents"] if agent["parent_a"]))


def test_run_n_generations_and_state_survives_reload(tmp_path) -> None:
    simulation = make_simulation(tmp_path, days=40)
    state = simulation.run_generations(3)
    assert state["generation"] == 3
    assert state["status"] == "READY"
    assert [item["generation"] for item in state["generations"] if item["status"] == "EVALUATED"] == [0, 1, 2]

    restored = InteractiveSimulation(
        development_market=simulation.development_market,
        history_prefix=simulation.history_prefix,
        forecaster=SyntheticForecaster(),
        state_path=simulation.state_path,
        population_size=4,
        seed=19,
    )
    assert restored.get_state()["generation"] == 3
    assert restored.get_state()["status"] == "READY"
    assert restored.get_state()["agents"] == state["agents"]


def test_reset_clears_only_interactive_state(tmp_path) -> None:
    simulation = make_simulation(tmp_path)
    simulation.next_day()
    state_path = simulation.state_path
    result = simulation.reset()

    assert state_path.exists()
    assert result["generation"] == 0
    assert result["day"] == 0
    assert result["status"] == "READY"
    assert result["daily_events"] == []


def test_refresh_mid_generation_preserves_baseline_and_daily_curves(tmp_path) -> None:
    simulation = make_simulation(tmp_path)
    simulation.next_day()
    baseline_before = simulation.manual_baseline.cash

    restored = InteractiveSimulation(
        development_market=simulation.development_market,
        history_prefix=simulation.history_prefix,
        forecaster=SyntheticForecaster(),
        state_path=simulation.state_path,
        population_size=4,
        seed=19,
    )
    assert restored.manual_baseline.cash == baseline_before
    restored.next_day()

    assert len(restored.strategy_curves["manual_baseline"]) == 3
    assert len(restored.strategy_curves["buy_and_hold"]) == 3
    assert len(restored.population.daily_events) == 1
    assert len(restored.current_day_events) == 2


def test_final_test_partition_is_rejected(tmp_path) -> None:
    development = market_frame(20)
    development.attrs["split"] = "test"
    with pytest.raises(ValueError, match="final-test"):
        InteractiveSimulation(
            development_market=development,
            history_prefix=market_frame(5),
            forecaster=SyntheticForecaster(),
            state_path=tmp_path / "interactive.json",
            population_size=4,
        )


def test_api_loader_truncates_before_feature_engineering(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import artificial_life_trading_ecosystem.api.app as api_module

    raw = market_frame(100)
    monkeypatch.setattr(api_module, "load_raw_market_data", lambda: raw.copy())
    monkeypatch.setattr(api_module, "_ensure_forecaster", lambda _name, allow_heuristic: SyntheticForecaster())

    simulation = api_module.build_development_simulation("GRU", state_path=tmp_path / "gru-state.json")
    from artificial_life_trading_ecosystem.config import settings
    validation_end = int(100 * (settings.train_proportion + settings.validation_proportion))
    evolution_end = int(100 * (settings.train_proportion + settings.validation_proportion + settings.evolution_proportion))

    assert simulation.experiment_id == "local-evolution-development:GRU"
    assert len(simulation.history_prefix) == validation_end
    assert len(simulation.development_market) == evolution_end - validation_end
    assert simulation.development_market["Close"].max() < raw.iloc[evolution_end:]["Close"].min()


def test_api_controls_events_generation_selection_and_final_test_boundary(tmp_path) -> None:
    from artificial_life_trading_ecosystem.api.app import create_app

    simulation = make_simulation(tmp_path, days=30)
    def select_model(model_name: str):
        selected = make_simulation(tmp_path / model_name)
        selected.experiment_id = f"local-evolution-development:{model_name}"
        return selected

    client = TestClient(create_app(simulation, simulation_factory=select_model))

    initial = client.get("/api/simulation").json()
    assert initial["generation"] == 0
    assert initial["status"] == "READY"
    assert initial["final_test_access"] is False
    experiments = client.get("/api/experiments").json()["experiments"]
    assert {entry["model_name"] for entry in experiments} == {"SimpleRNN", "LSTM", "GRU"}

    day = client.post("/api/simulation/next-day").json()
    assert day["status"] == "RUNNING"
    assert day["day"] == 1
    assert day["counts"]["BUY"] == 4

    evaluated = client.post("/api/simulation/run-generation").json()
    generation_zero = client.get("/api/generations/0").json()
    assert generation_zero["status"] == "EVALUATED"
    assert generation_zero["day"] == 10
    assert generation_zero["agents"][0]["ending_capital"] is not None
    assert generation_zero["agents"][0]["trade_count"] > 0
    assert len(generation_zero["agents"][0]["equity_curve"]) == 11
    assert evaluated["generation"] == 1
    assert evaluated["status"] == "READY"
    assert all(agent["status"] == "READY" for agent in evaluated["agents"])

    replay = client.get("/api/replay/0").json()
    assert len(replay["events"]) == 10
    assert replay["events"][0]["agents"][0]["trades"]

    backtest = client.get("/api/backtest/0").json()
    assert backtest["final_test_enabled"] is False
    assert len(backtest["manual_baseline"]) == 11
    assert len(backtest["buy_and_hold"]) == 11
    assert len(backtest["best_lineage"]["equity_curve"]) == 11
    assert client.get("/api/final-test").status_code == 404

    run_two = client.post("/api/simulation/run-generations", json={"count": 2}).json()
    assert run_two["generation"] == 3
    assert [item["generation"] for item in run_two["generations"] if item["status"] == "EVALUATED"] == [0, 1, 2]

    reset = client.post("/api/simulation/reset").json()
    assert reset["generation"] == 0
    assert reset["status"] == "READY"

    selected = client.post("/api/simulation/select-experiment", json={"model_name": "LSTM"}).json()
    assert selected["experiment_id"] == "local-evolution-development:LSTM"
    assert selected["generation"] == 0
    assert selected["status"] == "READY"