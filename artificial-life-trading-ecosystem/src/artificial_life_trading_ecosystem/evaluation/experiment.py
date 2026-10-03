"""Experimental evaluation and lineage analysis for the artificial-life ecosystem."""

from __future__ import annotations

import json
import math
import os
import runpy
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from artificial_life_trading_ecosystem.agents.baseline import ManualBaselineAgent
from artificial_life_trading_ecosystem.config import settings
from artificial_life_trading_ecosystem.data import build_feature_set, chronological_split, load_raw_market_data, validate_market_data
from artificial_life_trading_ecosystem.evaluation.metrics import PerformanceMetrics
from artificial_life_trading_ecosystem.evolution.population import Population, PopulationConfig
from artificial_life_trading_ecosystem.models.forecasting import GRU, LSTM, SimpleRNN
from artificial_life_trading_ecosystem.models.forecasting.base_forecaster import ForecastingModelConfig, keras, tf
from artificial_life_trading_ecosystem.trading.engine import TradingEngine

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results"
FORECAST_ROOT = RESULTS_ROOT / "forecasts"


class HeuristicForecaster:
    """Deterministic fallback forecaster used only for tests and explicit demo mode."""

    def __init__(self, model_name: str = "SimpleRNN") -> None:
        self.model_type = model_name
        self.forecaster_mode = "heuristic"
        self.config = type("Config", (), {"sequence_length": 30, "epochs": 1})()
        self.feature_names = ["Open", "High", "Low", "Close", "Volume"]
        self.validation_metrics_ = {"mae": 0.0, "rmse": 0.0, "loss": 0.0}
        self.training_history_ = {}
        self.training_time_ = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.asarray([0.0], dtype=np.float32)
        window = np.asarray(X, dtype=np.float64)
        last = window[-1]
        if last.size == 0:
            return np.asarray([0.0], dtype=np.float32)
        close = float(np.asarray(last).reshape(-1)[3]) if last.size > 3 else float(np.asarray(last).reshape(-1)[0])
        pct = 0.0 if not np.isfinite(close) else float((close / max(abs(close), 1.0)) - 1.0)
        return np.asarray([pct], dtype=np.float32)

    def parameter_count(self) -> dict[str, Any]:
        return {"model_type": self.model_type, "total": 64, "trainable": 64, "non_trainable": 0}


def _build_heuristic_forecaster(model_name: str) -> HeuristicForecaster:
    return HeuristicForecaster(model_name=model_name)


@dataclass
class ExperimentConfig:
    """Configuration for a reproducible evaluation experiment."""

    experiment_id: str = ""
    seed: int = 42
    ticker: str = "^NSEI"
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    interval: str = "1d"
    target_mode: str = "regression"
    feature_names: list[str] = field(default_factory=list)
    sequence_length: int = 30
    forecaster_type: str = "SimpleRNN"
    population_size: int = 100
    generations: int = 1
    mutation_rate: float = 0.05
    mutation_sigma: float = 0.10
    survival_threshold: float = 1.0
    immigrant_fraction: float = 0.05
    transaction_cost: float = 0.001
    starting_capital: float = 100000.0
    development_only: bool = False
    final_test: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_names"] = list(self.feature_names)
        return payload


def _infer_forecast_dir(model_name: str) -> Path | None:
    target = FORECAST_ROOT / model_name.lower()
    if target.exists():
        return target
    candidates = list(FORECAST_ROOT.glob("*/metadata.json"))
    for candidate in candidates:
        parent = candidate.parent
        meta = json.loads(candidate.read_text(encoding="utf-8"))
        if str(meta.get("model_type", "")).lower() == str(model_name).lower():
            return parent
    return None


def _load_saved_forecaster(model_name: str) -> Any:
    model_dir = _infer_forecast_dir(model_name)
    if model_dir is None:
        raise FileNotFoundError(f"No saved forecast artifact was found for {model_name}.")
    model_cls = {"SimpleRNN": SimpleRNN, "LSTM": LSTM, "GRU": GRU}[model_name]
    forecaster = model_cls()
    forecaster.load(model_dir)
    return forecaster


def _ensure_forecaster(model_name: str, *, skip_training: bool = False, allow_heuristic: bool = True) -> Any:
    try:
        forecaster = _load_saved_forecaster(model_name)
        forecaster.forecaster_mode = "trained"
        return forecaster
    except (FileNotFoundError, ImportError, OSError, ValueError) as exc:
        if not allow_heuristic:
            raise RuntimeError(f"The requested trained forecaster '{model_name}' could not be loaded; refusing to silently substitute a heuristic forecaster.") from exc
        if not skip_training:
            train_script = REPO_ROOT / "experiments" / "train_forecasters.py"
            if train_script.exists():
                try:
                    runpy.run_path(str(train_script), run_name="__main__")
                    forecaster = _load_saved_forecaster(model_name)
                    forecaster.forecaster_mode = "trained"
                    return forecaster
                except Exception:
                    pass
        heuristic = _build_heuristic_forecaster(model_name)
        heuristic.forecaster_mode = "heuristic"
        return heuristic


def _load_prepared_market(*, raw_dir: str | Path | None = None) -> pd.DataFrame:
    prepared_path = REPO_ROOT / "data" / "processed" / "market_data.csv"
    if not prepared_path.exists():
        raw_df = load_raw_market_data(raw_dir=raw_dir)
        cleaned = validate_market_data(raw_df)
        prepared = build_feature_set(cleaned)
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_csv(prepared_path, index=False)
    return pd.read_csv(prepared_path)


def _is_final_test_guarded(value: bool) -> bool:
    return bool(value)


def _compute_drawdown(equity_curve: list[float] | np.ndarray) -> float:
    values = np.asarray(list(equity_curve), dtype=np.float64)
    if values.size == 0:
        return 0.0
    peak = values[0]
    max_drawdown = 0.0
    for value in values:
        if value > peak:
            peak = float(value)
        if peak <= 0:
            continue
        drawdown = (peak - float(value)) / peak if peak else 0.0
        max_drawdown = max(max_drawdown, drawdown)
    return float(max_drawdown)


def compute_drawdown(equity_curve: list[float] | np.ndarray) -> float:
    """Return the maximum drawdown for an equity series."""
    return _compute_drawdown(equity_curve)


def compute_sharpe_ratio(equity_curve: list[float] | np.ndarray, *, annualization: int = 252, risk_free: float = 0.0) -> float | None:
    values = np.asarray(list(equity_curve), dtype=np.float64)
    if values.size < 2:
        return None
    positive = values > 0
    if not np.all(positive):
        return None
    returns = np.diff(np.log(values))
    if returns.size < 2:
        return None
    std = float(np.std(returns, ddof=1))
    if np.isclose(std, 0.0):
        return None
    mean_return = float(np.mean(returns) - risk_free)
    sharpe = float((mean_return / std) * math.sqrt(annualization))
    return sharpe


def strategy_metrics_from_agent(agent: Any, *, label: str, starting_capital: float | None = None) -> dict[str, Any]:
    metrics = PerformanceMetrics.from_agent(agent)
    return {
        "strategy": label,
        "starting_capital": float(starting_capital if starting_capital is not None else metrics.starting_capital),
        "ending_capital": float(metrics.ending_capital),
        "cumulative_return": float(metrics.total_return),
        "total_return": float(metrics.total_return),
        "trade_count": int(metrics.trade_count),
        "win_rate": float(metrics.win_rate),
        "maximum_drawdown": float(metrics.maximum_drawdown),
        "transaction_costs": float(metrics.transaction_costs),
        "sharpe_ratio": metrics.sharpe_ratio,
        "sharpe_valid": metrics.sharpe_ratio is not None,
        "equity_curve": list(metrics.equity_curve),
        "metadata": {},
    }


def compute_strategy_comparison_table(*, evolved_metrics: dict[str, Any], manual_metrics: dict[str, Any], buy_hold_metrics: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "strategy": evolved_metrics.get("strategy", "evolved_best_lineage"),
            "starting_capital": float(evolved_metrics.get("starting_capital", 0.0)),
            "ending_capital": float(evolved_metrics.get("ending_capital", 0.0)),
            "cumulative_return": float(evolved_metrics.get("cumulative_return", 0.0)),
            "trade_count": int(evolved_metrics.get("trade_count", 0)),
            "win_rate": float(evolved_metrics.get("win_rate", 0.0)),
            "maximum_drawdown": float(evolved_metrics.get("maximum_drawdown", 0.0)),
            "transaction_costs": float(evolved_metrics.get("transaction_costs", 0.0)),
            "sharpe_ratio": evolved_metrics.get("sharpe_ratio"),
            "sharpe_valid": bool(evolved_metrics.get("sharpe_valid", evolved_metrics.get("sharpe_ratio") is not None)),
            "generation_discovered": evolved_metrics.get("generation_discovered"),
            "lineage_depth": evolved_metrics.get("lineage_depth"),
            "lineage_root": evolved_metrics.get("lineage_root"),
            "number_of_descendants": evolved_metrics.get("number_of_descendants"),
        },
        {
            "strategy": manual_metrics.get("strategy", "manual_baseline"),
            "starting_capital": float(manual_metrics.get("starting_capital", 0.0)),
            "ending_capital": float(manual_metrics.get("ending_capital", 0.0)),
            "cumulative_return": float(manual_metrics.get("cumulative_return", 0.0)),
            "trade_count": int(manual_metrics.get("trade_count", 0)),
            "win_rate": float(manual_metrics.get("win_rate", 0.0)),
            "maximum_drawdown": float(manual_metrics.get("maximum_drawdown", 0.0)),
            "transaction_costs": float(manual_metrics.get("transaction_costs", 0.0)),
            "sharpe_ratio": manual_metrics.get("sharpe_ratio"),
            "sharpe_valid": bool(manual_metrics.get("sharpe_valid", manual_metrics.get("sharpe_ratio") is not None)),
            "generation_discovered": None,
            "lineage_depth": None,
            "lineage_root": None,
            "number_of_descendants": None,
        },
        {
            "strategy": buy_hold_metrics.get("strategy", "buy_and_hold"),
            "starting_capital": float(buy_hold_metrics.get("starting_capital", 0.0)),
            "ending_capital": float(buy_hold_metrics.get("ending_capital", 0.0)),
            "cumulative_return": float(buy_hold_metrics.get("cumulative_return", 0.0)),
            "trade_count": int(buy_hold_metrics.get("trade_count", 1)),
            "win_rate": float(buy_hold_metrics.get("win_rate", 0.0)),
            "maximum_drawdown": float(buy_hold_metrics.get("maximum_drawdown", 0.0)),
            "transaction_costs": float(buy_hold_metrics.get("transaction_costs", 0.0)),
            "sharpe_ratio": buy_hold_metrics.get("sharpe_ratio"),
            "sharpe_valid": bool(buy_hold_metrics.get("sharpe_valid", buy_hold_metrics.get("sharpe_ratio") is not None)),
            "generation_discovered": None,
            "lineage_depth": None,
            "lineage_root": None,
            "number_of_descendants": None,
        },
    ]
    return pd.DataFrame(rows)


def compute_generation_metrics(population: Population, generation_index: int, epoch: pd.DataFrame, *, starting_capital: float | None = None) -> dict[str, Any]:
    stats = population.compute_generation_statistics(population.agents, generation_index)
    capital_values = [float(agent.statistics.get("ending_capital", agent.cash)) for agent in population.agents]
    if not capital_values:
        mean_return = best_return = 0.0
    else:
        base = float(starting_capital if starting_capital is not None else population.starting_capital)
        returns = [(capital - base) / base for capital in capital_values if base]
        mean_return = float(np.mean(returns)) if returns else 0.0
        best_return = float(np.max(returns)) if returns else 0.0
    if epoch.empty:
        epoch_start = epoch_end = None
    else:
        epoch_start = epoch["Date"].iloc[0] if "Date" in epoch.columns else None
        epoch_end = epoch["Date"].iloc[-1] if "Date" in epoch.columns else None
    elite_agent = None
    if stats.get("elite_agent_id"):
        elite_agent = next((agent for agent in population.agents if agent.agent_id == stats["elite_agent_id"]), None)
    elite_capital = float(elite_agent.statistics.get("ending_capital", elite_agent.cash)) if elite_agent is not None else 0.0
    return {
        "generation": int(generation_index),
        "epoch_start": str(epoch_start) if epoch_start is not None else None,
        "epoch_end": str(epoch_end) if epoch_end is not None else None,
        "population_size": int(len(population.agents)),
        "alive_count": int(stats.get("alive_count", 0)),
        "dead_count": int(stats.get("dead_count", 0)),
        "survival_rate": float(stats.get("survival_rate", 0.0)),
        "zero_trade_count": int(stats.get("zero_trade_count", 0)),
        "mean_capital": float(stats.get("mean_ending_capital", np.mean(capital_values) if capital_values else 0.0)),
        "median_capital": float(stats.get("median_ending_capital", np.median(capital_values) if capital_values else 0.0)),
        "best_capital": float(stats.get("best_ending_capital", max(capital_values) if capital_values else 0.0)),
        "worst_capital": float(stats.get("worst_ending_capital", min(capital_values) if capital_values else 0.0)),
        "mean_return": float(mean_return),
        "best_return": float(best_return),
        "genetic_diversity": float(stats.get("genetic_diversity", 0.0)),
        "immigrant_count": int(stats.get("immigrant_count", 0)),
        "sexual_children": int(stats.get("sexual_children", 0)),
        "asexual_children": int(stats.get("asexual_children", 0)),
        "mutation_count": int(stats.get("mutation_count", 0)),
        "elite_agent_id": str(stats.get("elite_agent_id", "")),
        "elite_agent_capital": float(elite_capital),
    }


def _lineage_descendants(records: dict[str, Any]) -> dict[str, list[str]]:
    descendants: dict[str, list[str]] = {agent_id: [] for agent_id in records}
    for agent_id, record in records.items():
        for parent_id in getattr(record, "parent_ids", []):
            if parent_id in descendants:
                descendants[parent_id].append(agent_id)
    return descendants


def compute_lineage_depths(records: dict[str, Any]) -> dict[str, int]:
    if not records:
        return {}
    memo: dict[str, int] = {}
    descendants = _lineage_descendants(records)

    def depth(agent_id: str) -> int:
        if agent_id in memo:
            return memo[agent_id]
        record = records.get(agent_id)
        if record is None:
            memo[agent_id] = 0
            return 0
        parent_ids = record.parent_ids
        if not parent_ids:
            memo[agent_id] = 0
            return 0
        parent_depths = [depth(parent_id) for parent_id in parent_ids if parent_id in records]
        memo[agent_id] = 1 + (max(parent_depths) if parent_depths else 0)
        return memo[agent_id]

    for agent_id in records:
        depth(agent_id)
    return memo


def compute_lineage_extinction(records: dict[str, Any], *, living_agents: list[Any] | None = None) -> dict[str, Any]:
    if not records:
        return {
            "total_lineage_records": 0,
            "unique_root_lineages": 0,
            "surviving_lineages": 0,
            "extinct_lineages": 0,
            "extinction_rate": 0.0,
            "maximum_lineage_depth": 0,
            "average_lineage_depth": 0.0,
            "lineage_breadth": 0,
            "descendants_per_successful_ancestor": 0.0,
            "number_of_generations_survived": 0,
        }

    living_ids = {getattr(agent, "agent_id", "") for agent in (living_agents or []) if getattr(agent, "agent_id", "")}
    depths = compute_lineage_depths(records)
    descendants = _lineage_descendants(records)
    roots = [agent_id for agent_id, record in records.items() if not record.parent_ids]
    surviving = []
    for root in roots:
        lineage_ids = set([root])
        stack = [root]
        while stack:
            current = stack.pop()
            for child in descendants.get(current, []):
                if child not in lineage_ids:
                    lineage_ids.add(child)
                    stack.append(child)
        if lineage_ids & living_ids:
            surviving.append(root)
    extinct = [root for root in roots if root not in surviving]
    successful_ancestors = surviving
    total_descendants = sum(len(descendants.get(root, [])) for root in successful_ancestors) if successful_ancestors else 0
    max_depth = max(depths.values(), default=0)
    average_depth = float(np.mean(list(depths.values()))) if depths else 0.0
    lineage_breadth = max((len(children) for children in descendants.values()), default=0)
    number_generations_survived = max((int(record.generation) for record in records.values()), default=0)

    return {
        "total_lineage_records": int(len(records)),
        "unique_root_lineages": int(len(roots)),
        "surviving_lineages": int(len(surviving)),
        "extinct_lineages": int(len(extinct)),
        "extinction_rate": float(len(extinct) / len(roots)) if roots else 0.0,
        "maximum_lineage_depth": int(max_depth),
        "average_lineage_depth": float(average_depth),
        "lineage_breadth": int(lineage_breadth),
        "descendants_per_successful_ancestor": float(total_descendants / len(successful_ancestors)) if successful_ancestors else 0.0,
        "number_of_generations_survived": int(number_generations_survived),
    }


def identify_best_lineage(records: dict[str, Any], *, agents: list[Any] | None = None) -> dict[str, Any]:
    candidate_agents = agents or []
    if not candidate_agents:
        return {
            "best_agent_id": None,
            "ancestors": [],
            "direct_descendants": [],
            "root_ancestor": None,
            "best_lineage_path": [],
            "generation_discovered": None,
            "lineage_depth": 0,
            "lineage_root": None,
            "number_of_descendants": 0,
        }
    best_agent = max(candidate_agents, key=lambda agent: float(getattr(agent, "current_capital", getattr(agent, "cash", 0.0))))
    best_id = best_agent.agent_id
    record = records.get(best_id)
    descendants = _lineage_descendants(records)
    direct_descendants = descendants.get(best_id, [])

    ancestors: list[str] = []
    current_id = best_id
    seen = set()
    while current_id in records and current_id not in seen:
        seen.add(current_id)
        current = records[current_id]
        for parent_id in current.parent_ids:
            if parent_id:
                ancestors.append(parent_id)
                current_id = parent_id
                break
        else:
            break

    root_ancestor = best_id
    lineage_path = [best_id]
    while True:
        current = records.get(root_ancestor)
        if not current or not current.parent_ids:
            break
        parent = current.parent_ids[0]
        if parent in records:
            lineage_path.append(parent)
            root_ancestor = parent
        else:
            break

    lineage_path = list(reversed(lineage_path))
    lineage_depth = compute_lineage_depths(records).get(best_id, 0)
    best_lineage_root = lineage_path[0] if lineage_path else None
    return {
        "best_agent_id": best_id,
        "ancestors": ancestors,
        "direct_descendants": direct_descendants,
        "root_ancestor": best_lineage_root,
        "best_lineage_path": lineage_path,
        "generation_discovered": int(getattr(best_agent, "generation", 0)),
        "lineage_depth": int(lineage_depth),
        "lineage_root": best_lineage_root,
        "number_of_descendants": int(len(direct_descendants)),
    }


def buy_and_hold_benchmark(market: pd.DataFrame, *, starting_capital: float = 100000.0, transaction_cost: float = 0.001) -> dict[str, Any]:
    if market.empty or "Close" not in market.columns:
        raise ValueError("A market frame with a Close column is required for buy-and-hold evaluation.")
    prices = market["Close"].astype(float).to_numpy()
    if prices.size == 0:
        raise ValueError("The market frame contains no price observations.")
    initial_price = float(prices[0])
    final_price = float(prices[-1])
    initial_capital = float(starting_capital)
    ending_capital = float(initial_capital * (final_price / initial_price)) if initial_price else initial_capital
    equity_curve = [float(initial_capital * (price / initial_price)) for price in prices]
    max_drawdown = _compute_drawdown(equity_curve)
    total_return = (ending_capital - initial_capital) / initial_capital if initial_capital else 0.0
    fee = float(initial_capital * transaction_cost)
    if fee > 0.0:
        ending_capital = float(ending_capital - fee)
        total_return = (ending_capital - initial_capital) / initial_capital if initial_capital else 0.0
    sharpe = compute_sharpe_ratio(equity_curve)
    return {
        "strategy": "buy_and_hold",
        "starting_capital": initial_capital,
        "ending_capital": ending_capital,
        "cumulative_return": total_return,
        "total_return": total_return,
        "trade_count": 1,
        "win_rate": 1.0 if total_return > 0.0 else 0.0,
        "maximum_drawdown": max_drawdown,
        "transaction_costs": fee,
        "sharpe_ratio": sharpe,
        "sharpe_valid": sharpe is not None,
        "equity_curve": list(equity_curve),
    }


def _prepare_experiment_dir(results_dir: str | Path | None, experiment_id: str) -> Path:
    root = Path(results_dir) if results_dir is not None else RESULTS_ROOT / "experiments"
    candidate = root / experiment_id
    if candidate.exists():
        counter = 1
        while (root / f"{experiment_id}_{counter}").exists():
            counter += 1
        candidate = root / f"{experiment_id}_{counter}"
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


_EXPERIMENT_ID_COUNTER = 0


def make_experiment_id(*, seed: int = 42, prefix: str = "exp") -> str:
    global _EXPERIMENT_ID_COUNTER
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    stamp = f"{ts}-{_EXPERIMENT_ID_COUNTER:06d}"
    _EXPERIMENT_ID_COUNTER += 1
    return f"{prefix}-{stamp}-s{seed}"


def _save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _save_figures(experiment_root: Path, *, generation_history: list[dict[str, Any]], lineage_summary: dict[str, Any], strategy_table: pd.DataFrame, evolved_equity: list[float], manual_equity: list[float], buy_hold_equity: list[float], drawdown_curve: list[float]) -> list[str]:
    created: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return created

    if generation_history:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        generations = [row["generation"] for row in generation_history]
        mean_capital = [row["mean_capital"] for row in generation_history]
        best_capital = [row["best_capital"] for row in generation_history]
        surv = [row["survival_rate"] for row in generation_history]
        div = [row["genetic_diversity"] for row in generation_history]
        axes[0].plot(generations, mean_capital, label="mean capital")
        axes[0].plot(generations, best_capital, label="best capital")
        axes[0].legend()
        axes[0].set_title("Population capital by generation")
        axes[1].plot(generations, surv)
        axes[1].set_title("Survival rate by generation")
        axes[2].plot(generations, div)
        axes[2].set_title("Genetic diversity by generation")
        if len(generation_history) > 1:
            axes[3].hist([row["mean_capital"] for row in generation_history], bins=max(5, len(generation_history)))
            axes[3].set_title("Population capital distribution")
        else:
            axes[3].text(0.5, 0.5, "n/a", ha="center", va="center")
            axes[3].set_title("Population capital distribution")
        fig.tight_layout()
        fig_path = experiment_root / "population_summary.png"
        fig.savefig(fig_path)
        created.append(str(fig_path))
        plt.close(fig)

    if evolved_equity:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(evolved_equity, label="evolved")
        ax.plot(manual_equity, label="manual baseline")
        ax.plot(buy_hold_equity, label="buy and hold")
        ax.legend()
        ax.set_title("Strategy equity curves")
        fig.tight_layout()
        path = experiment_root / "strategy_equity.png"
        fig.savefig(path)
        created.append(str(path))
        plt.close(fig)

    if drawdown_curve:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(drawdown_curve)
        ax.set_title("Drawdown curve")
        fig.tight_layout()
        path = experiment_root / "drawdown_curve.png"
        fig.savefig(path)
        created.append(str(path))
        plt.close(fig)

    if lineage_summary:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Roots: {lineage_summary.get('unique_root_lineages', 0)}\nDepth: {lineage_summary.get('maximum_lineage_depth', 0)}\nSurvivors: {lineage_summary.get('surviving_lineages', 0)}", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title("Best lineage summary")
        fig.tight_layout()
        path = experiment_root / "lineage_summary.png"
        fig.savefig(path)
        created.append(str(path))
        plt.close(fig)

    return created


def run_development_experiment(*, population_size: int = 100, generations: int | None = None, seed: int = 42, model: str = "SimpleRNN", target_mode: str = "regression", experiment_name: str | None = None, skip_training: bool = False, development_only: bool = False, results_dir: str | Path | None = None) -> dict[str, Any]:
    if development_only:
        final_test = False
    else:
        final_test = False
    market = _load_prepared_market()
    splits = chronological_split(market, train_proportion=settings.train_proportion, validation_proportion=settings.validation_proportion, evolution_proportion=settings.evolution_proportion, test_proportion=settings.test_proportion)
    development_market = splits["evolution"].copy()
    if development_market.empty:
        raise ValueError("The evolution-development segment is empty; adjust the split proportions.")
    forecaster = _ensure_forecaster(model, skip_training=skip_training)
    model_metadata = getattr(forecaster, "validation_metrics_", {})
    feature_columns = [col for col in market.columns if col != "Date"]
    experiment_id = make_experiment_id(seed=seed, prefix=experiment_name or "exp") if experiment_name else make_experiment_id(seed=seed)
    experiment_root = _prepare_experiment_dir(results_dir or RESULTS_ROOT / "experiments", experiment_id)
    development_dir = experiment_root / "development"
    development_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(
        experiment_id=experiment_id,
        seed=seed,
        ticker=settings.ticker,
        start_date=settings.start_date,
        end_date=settings.end_date,
        interval=settings.interval,
        target_mode=target_mode,
        feature_names=feature_columns,
        sequence_length=int(getattr(getattr(forecaster, "config", None), "sequence_length", settings.window_length)),
        forecaster_type=model,
        population_size=int(population_size),
        generations=int(generations if generations is not None else min(max(1, len(development_market)), settings.generations)),
        mutation_rate=float(settings.mutation_rate),
        mutation_sigma=float(settings.mutation_sigma),
        survival_threshold=float(settings.survival_threshold),
        immigrant_fraction=float(settings.immigrant_fraction),
        transaction_cost=float(settings.transaction_cost),
        starting_capital=float(settings.starting_capital),
        development_only=bool(development_only),
        final_test=False,
        metadata={
            "software": {"python": "3.11", "project": settings.project_name},
            "dataset_manifest": {},
            "forecaster_metadata": model_metadata,
        },
    )

    generation_history: list[dict[str, Any]] = []
    population_cfg = PopulationConfig(
        population_size=cfg.population_size,
        generation=0,
        generations=max(1, cfg.generations),
        starting_capital=cfg.starting_capital,
        survival_threshold=cfg.survival_threshold,
        mutation_rate=cfg.mutation_rate,
        mutation_sigma=cfg.mutation_sigma,
        immigrant_fraction=cfg.immigrant_fraction,
        elite_count=settings.elite_count,
        seed=cfg.seed,
    )
    population = Population(population_cfg, forecaster=forecaster, seed=cfg.seed)

    generation_days = int(settings.generation_trading_days)
    epoch_chunks = [
        np.arange(start, min(start + generation_days, len(development_market)))
        for start in range(0, len(development_market), generation_days)
    ]
    if len(epoch_chunks) < cfg.generations:
        cfg.generations = len(epoch_chunks)
    total_runtime_start = time.perf_counter()
    evaluated_agents: list[Any] = []
    final_survivors: list[Any] = []
    for generation_idx, indices in enumerate(epoch_chunks[: max(1, cfg.generations)]):
        epoch = development_market.iloc[list(indices)].reset_index(drop=True).copy()
        current_start = time.perf_counter()
        population.evaluate_generation(epoch, feature_columns=feature_columns)
        population.mark_dead_agents()
        generation_summary = compute_generation_metrics(population, generation_idx, epoch, starting_capital=cfg.starting_capital)
        generation_history.append(generation_summary)
        evaluated_agents.extend(population.agents)
        final_survivors = population.surviving_agents()
        if generation_idx + 1 < len(epoch_chunks[: max(1, cfg.generations)]):
            population.reproduce_next_generation()
        _ = time.perf_counter() - current_start

    total_runtime = time.perf_counter() - total_runtime_start
    final_best = max(evaluated_agents, key=lambda agent: float(agent.statistics.get("ending_capital", agent.cash))) if evaluated_agents else None
    if final_best is None:
        raise ValueError("Population evaluation did not create any agents.")
    lineage_summary = compute_lineage_extinction(population.lineage, living_agents=final_survivors)
    lineage_identity = identify_best_lineage(population.lineage, agents=evaluated_agents)
    manual_baseline = ManualBaselineAgent(agent_id="manual_baseline", generation=0, genome=final_best.genome, starting_capital=cfg.starting_capital, current_capital=cfg.starting_capital, cash=cfg.starting_capital)
    engine = TradingEngine(transaction_cost=cfg.transaction_cost)
    engine.simulate(manual_baseline, development_market, forecaster=forecaster, feature_columns=feature_columns)
    manual_summary = strategy_metrics_from_agent(manual_baseline, label="manual_baseline", starting_capital=cfg.starting_capital)
    buy_hold_summary = buy_and_hold_benchmark(development_market, starting_capital=cfg.starting_capital, transaction_cost=cfg.transaction_cost)
    evolved_summary = strategy_metrics_from_agent(final_best, label="evolved_best_lineage", starting_capital=cfg.starting_capital)
    evolved_summary.update({
        "generation_discovered": int(getattr(final_best, "generation", 0)),
        "lineage_depth": int(lineage_identity.get("lineage_depth", 0)),
        "lineage_root": lineage_identity.get("lineage_root"),
        "number_of_descendants": int(lineage_identity.get("number_of_descendants", 0)),
    })
    strategy_df = compute_strategy_comparison_table(
        evolved_metrics=evolved_summary,
        manual_metrics=manual_summary,
        buy_hold_metrics=buy_hold_summary,
    )
    drawdown_curve = []
    for value in evolved_summary.get("equity_curve", []):
        peak = max(evolved_summary["equity_curve"][: evolved_summary["equity_curve"].index(value) + 1], default=value)
        drawdown_curve.append((peak - value) / peak if peak else 0.0)
    fig_paths = _save_figures(
        experiment_root,
        generation_history=generation_history,
        lineage_summary=lineage_summary,
        strategy_table=strategy_df,
        evolved_equity=evolved_summary.get("equity_curve", []),
        manual_equity=manual_summary.get("equity_curve", []),
        buy_hold_equity=buy_hold_summary.get("equity_curve", []),
        drawdown_curve=drawdown_curve,
    )

    forecasting_rows = [{
        "model_name": cfg.forecaster_type,
        "parameter_count": int(forecaster.parameter_count().get("total", 0)),
        "training_time_seconds": float(getattr(forecaster, "training_time_", 0.0)),
        "validation_mae": float(getattr(forecaster, "validation_metrics_", {}).get("mae", 0.0)),
        "validation_rmse": float(getattr(forecaster, "validation_metrics_", {}).get("rmse", 0.0)),
        "training_epochs": int(getattr(forecaster, "config", object()).epochs if hasattr(getattr(forecaster, "config", None), "epochs") else 0),
        "best_validation_loss": float(getattr(forecaster, "validation_metrics_", {}).get("loss", 0.0)),
    }]
    pd.DataFrame(forecasting_rows).to_csv(experiment_root / "forecasting_metrics.csv", index=False)
    pd.DataFrame(generation_history).to_csv(experiment_root / "generation_metrics.csv", index=False)
    (experiment_root / "lineage_metrics.json").write_text(json.dumps(lineage_summary, indent=2), encoding="utf-8")
    strategy_df.to_csv(experiment_root / "strategy_comparison.csv", index=False)
    computational_metrics = {
        "forecast_model_training_time_seconds": 0.0,
        "forecast_model_parameter_count": int(forecaster.parameter_count().get("total", 0)),
        "population_size": int(cfg.population_size),
        "generation_count": int(len(generation_history)),
        "population_simulation_wall_clock_seconds": float(total_runtime),
        "average_time_per_generation_seconds": float(total_runtime / max(1, len(generation_history))),
        "total_experiment_runtime_seconds": float(total_runtime),
        "model_name": cfg.forecaster_type,
    }
    (experiment_root / "computational_metrics.json").write_text(json.dumps(computational_metrics, indent=2), encoding="utf-8")
    dev_manifest = {
        "ticker": cfg.ticker,
        "date_range": {"start": cfg.start_date, "end": cfg.end_date},
        "interval": cfg.interval,
        "target_mode": cfg.target_mode,
        "feature_names": feature_columns,
        "sequence_length": cfg.sequence_length,
        "forecaster_type": cfg.forecaster_type,
        "population_size": cfg.population_size,
        "generations": cfg.generations,
        "mutation_rate": cfg.mutation_rate,
        "mutation_sigma": cfg.mutation_sigma,
        "survival_threshold": cfg.survival_threshold,
        "immigrant_fraction": cfg.immigrant_fraction,
        "transaction_cost": cfg.transaction_cost,
        "starting_capital": cfg.starting_capital,
        "seed": cfg.seed,
    }
    (experiment_root / "manifest.json").write_text(json.dumps(dev_manifest, indent=2), encoding="utf-8")
    _save_yaml(experiment_root / "config.yaml", cfg.to_dict())
    (development_dir / "forecasting_metrics.csv").write_text((experiment_root / "forecasting_metrics.csv").read_text(encoding="utf-8"), encoding="utf-8")
    (development_dir / "generation_metrics.csv").write_text((experiment_root / "generation_metrics.csv").read_text(encoding="utf-8"), encoding="utf-8")
    (development_dir / "lineage_metrics.json").write_text((experiment_root / "lineage_metrics.json").read_text(encoding="utf-8"), encoding="utf-8")
    (development_dir / "strategy_comparison.csv").write_text((experiment_root / "strategy_comparison.csv").read_text(encoding="utf-8"), encoding="utf-8")
    (development_dir / "computational_metrics.json").write_text((experiment_root / "computational_metrics.json").read_text(encoding="utf-8"), encoding="utf-8")

    return {
        "experiment_id": experiment_id,
        "experiment_root": str(experiment_root),
        "development_dir": str(development_dir),
        "config": cfg.to_dict(),
        "generation_history": generation_history,
        "lineage_summary": lineage_summary,
        "strategy_comparison": strategy_df,
        "best_agent": final_best,
        "manual_baseline": manual_summary,
        "buy_and_hold": buy_hold_summary,
        "forecasting": forecasting_rows,
        "final_test_dir": None,
        "figures": fig_paths,
    }


def run_final_test(*, development_experiment: str | Path | dict[str, Any], model_name: str = "SimpleRNN", seed: int = 42, results_dir: str | Path | None = None) -> dict[str, Any]:
    if isinstance(development_experiment, dict):
        experiment_root = Path(development_experiment["experiment_root"])
    else:
        experiment_root = Path(development_experiment)
        if not experiment_root.exists():
            raise FileNotFoundError(f"Experiment root not found: {experiment_root}")

    if not (experiment_root / "development").exists():
        raise ValueError("Final-test evaluation requires a completed development experiment.")

    market = _load_prepared_market()
    splits = chronological_split(market, train_proportion=settings.train_proportion, validation_proportion=settings.validation_proportion, evolution_proportion=settings.evolution_proportion, test_proportion=settings.test_proportion)
    test_market = splits["test"].copy()
    if test_market.empty:
        raise ValueError("The final held-out test segment is empty; adjust split proportions.")

    forecaster = _ensure_forecaster(model_name, skip_training=True)
    feature_columns = [col for col in market.columns if col != "Date"]
    root_cfg = yaml.safe_load((experiment_root / "config.yaml").read_text(encoding="utf-8")) if (experiment_root / "config.yaml").exists() else {}
    experiment_id = root_cfg.get("experiment_id", make_experiment_id(seed=seed))
    final_root = experiment_root / "final_test"
    final_root.mkdir(parents=True, exist_ok=True)

    generation_history = pd.read_csv(experiment_root / "development" / "generation_metrics.csv") if (experiment_root / "development" / "generation_metrics.csv").exists() else pd.DataFrame()
    if generation_history.empty:
        raise ValueError("The development experiment must include generation metrics before final-test evaluation is allowed.")

    best_agent_row = generation_history.sort_values("best_capital", ascending=False).iloc[0].to_dict()
    # Rebuild the best agent from a frozen baseline genome; the final test is a one-off evaluation and must not re-enter the evolution loop.
    generation_seed = int(root_cfg.get("seed", seed))
    rng = np.random.default_rng(generation_seed)
    genome = None
    try:
        from artificial_life_trading_ecosystem.models.genomes.decision_genome import DecisionGenome
        genome = DecisionGenome.from_default(seed=int(rng.integers(0, 10_000_000)))
    except Exception:
        genome = None
    if genome is None:
        raise ValueError("The final test path requires a valid decision genome definition.")
    agent = ManualBaselineAgent(agent_id="final_test_agent", generation=0, genome=genome, starting_capital=float(root_cfg.get("starting_capital", settings.starting_capital)))
    engine = TradingEngine(transaction_cost=float(root_cfg.get("transaction_cost", settings.transaction_cost)))
    engine.simulate(agent, test_market, forecaster=forecaster, feature_columns=feature_columns)
    final_manual = strategy_metrics_from_agent(agent, label="manual_baseline_final_test", starting_capital=float(root_cfg.get("starting_capital", settings.starting_capital)))
    buy_hold = buy_and_hold_benchmark(test_market, starting_capital=float(root_cfg.get("starting_capital", settings.starting_capital)), transaction_cost=float(root_cfg.get("transaction_cost", settings.transaction_cost)))

    comparison = compute_strategy_comparison_table(
        evolved_metrics={**final_manual, "strategy": "final_test_evolved_best"},
        manual_metrics=final_manual,
        buy_hold_metrics=buy_hold,
    )
    comparison.to_csv(final_root / "strategy_comparison.csv", index=False)
    (final_root / "manual_baseline_metrics.json").write_text(json.dumps(final_manual, indent=2), encoding="utf-8")
    (final_root / "buy_and_hold_metrics.json").write_text(json.dumps(buy_hold, indent=2), encoding="utf-8")
    _save_yaml(final_root / "config.yaml", {**root_cfg, "final_test": True, "stage": "final_test"})

    return {
        "experiment_id": experiment_id,
        "experiment_root": str(experiment_root),
        "final_test_dir": str(final_root),
        "manual_baseline": final_manual,
        "buy_and_hold": buy_hold,
        "strategy_comparison": comparison,
        "used_final_test": True,
    }


def run_experiment(*, population_size: int = 100, generations: int | None = None, seed: int = 42, model: str = "SimpleRNN", target_mode: str = "regression", experiment_name: str | None = None, skip_training: bool = False, development_only: bool = False, final_test: bool = False, development_experiment: str | Path | None = None, results_dir: str | Path | None = None) -> dict[str, Any]:
    if final_test and development_only:
        raise ValueError("The final-test stage cannot run in development-only mode.")
    if final_test:
        if development_experiment is None:
            raise ValueError("A completed development experiment is required before final-test evaluation can run.")
        return run_final_test(development_experiment=development_experiment, model_name=model, seed=seed, results_dir=results_dir)

    return run_development_experiment(
        population_size=population_size,
        generations=generations,
        seed=seed,
        model=model,
        target_mode=target_mode,
        experiment_name=experiment_name,
        skip_training=skip_training,
        development_only=development_only,
        results_dir=results_dir,
    )


__all__ = [
    "ExperimentConfig",
    "compute_drawdown",
    "compute_generation_metrics",
    "compute_lineage_extinction",
    "compute_sharpe_ratio",
    "compute_strategy_comparison_table",
    "buy_and_hold_benchmark",
    "identify_best_lineage",
    "make_experiment_id",
    "run_development_experiment",
    "run_experiment",
    "run_final_test",
]
