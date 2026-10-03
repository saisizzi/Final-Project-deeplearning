"""Reusable trading simulator for long-only single-agent runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from artificial_life_trading_ecosystem.agents.agent import Agent
from artificial_life_trading_ecosystem.config import settings


class TradingEngine:
    """Separate execution layer that owns trade accounting and order logic."""

    generation_trading_days: int = 10

    def __init__(
        self,
        *,
        signal_threshold: float | None = None,
        position_size: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        transaction_cost: float | None = None,
        context_length: int | None = None,
        allow_short: bool = False,
        leverage: float = 1.0,
        generation_trading_days: int | None = None,
    ) -> None:
        self.signal_threshold = float(signal_threshold if signal_threshold is not None else settings.signal_threshold)
        self.position_size_override = position_size
        self.position_size = float(position_size if position_size is not None else settings.position_size)
        self.stop_loss = float(stop_loss if stop_loss is not None else settings.stop_loss)
        self.take_profit = float(take_profit if take_profit is not None else settings.take_profit)
        self.transaction_cost = float(transaction_cost if transaction_cost is not None else settings.transaction_cost)
        self.context_length = int(context_length if context_length is not None else settings.context_length)
        self.allow_short = bool(allow_short)
        self.leverage = float(leverage)
        self.generation_trading_days = int(generation_trading_days if generation_trading_days is not None else getattr(settings, "generation_trading_days", 10))

    def _entry_price_for_trade(self, agent: Agent, row: pd.Series) -> float:
        return float(row.get("Open", row.get("open", row.get("Close", row.get("ClosePrice", 0.0)))))

    def chunk_market_by_generation(self, market: pd.DataFrame, *, generation_trading_days: int | None = None) -> list[pd.DataFrame]:
        if not isinstance(market, pd.DataFrame):
            raise TypeError("market must be a pandas.DataFrame.")
        if market.empty:
            return []
        chunk_size = int(generation_trading_days if generation_trading_days is not None else self.generation_trading_days)
        if chunk_size <= 0:
            raise ValueError("generation_trading_days must be positive.")
        ordered = market.sort_values("Date", kind="mergesort").reset_index(drop=True)
        return [ordered.iloc[idx: idx + chunk_size].reset_index(drop=True) for idx in range(0, len(ordered), chunk_size)]

    def _market_context(self, market: pd.DataFrame, idx: int, *, context_length: int) -> list[float]:
        if market.empty:
            return []
        slice_start = max(0, idx - context_length)
        context = market.iloc[slice_start:idx + 1].copy()
        if "Close" in context.columns:
            values = context["Close"].astype(float).to_numpy().tolist()
        elif "close" in context.columns:
            values = context["close"].astype(float).to_numpy().tolist()
        else:
            values = [0.0] * len(context)
        if len(values) == 0:
            return [0.0] * context_length
        if len(values) < context_length:
            values = [0.0] * (context_length - len(values)) + values
        return values[-context_length:]

    def _normalized_context(self, market: pd.DataFrame, idx: int) -> list[float]:
        context = self._market_context(market, idx, context_length=self.context_length)
        if not context:
            return [0.0] * self.context_length
        arr = np.asarray(context, dtype=np.float64)
        if arr.size == 0:
            return [0.0] * self.context_length
        if np.allclose(arr, 0.0):
            return [0.0] * self.context_length
        return (arr - arr.mean()) / (arr.std() if arr.std() != 0.0 else 1.0)

    def _trigger_exit(self, agent: Agent, row: pd.Series) -> tuple[str | None, float | None]:
        if agent.position_quantity <= 0 or agent.entry_price is None:
            return None, None
        sl = agent.entry_price * (1.0 - agent.stop_loss)
        tp = agent.entry_price * (1.0 + agent.take_profit)
        low = float(row.get("Low", row.get("low", agent.entry_price)))
        high = float(row.get("High", row.get("high", agent.entry_price)))

        if low <= sl and high >= tp:
            return "STOP_LOSS", sl
        if low <= sl:
            return "STOP_LOSS", sl
        if high >= tp:
            return "TAKE_PROFIT", tp
        return None, None

    def open_long(self, agent: Agent, row: pd.Series, *, action: str, reason: str = "ENTRY") -> None:
        if agent.position_quantity > 0:
            return
        price = float(row.get("Close", row.get("ClosePrice", 0.0)))
        if price <= 0:
            return
        position_size = self.position_size if self.position_size_override is not None else agent.position_size_fraction
        max_quantity = int(np.floor((agent.cash * position_size) / (price * (1.0 + agent.transaction_cost))))
        if max_quantity <= 0:
            return
        quantity = max_quantity
        notional = quantity * price
        fee = notional * agent.transaction_cost
        if agent.cash < notional + fee:
            quantity = max(0, int(np.floor((agent.cash / (price * (1.0 + agent.transaction_cost))))))
            notional = quantity * price
            fee = notional * agent.transaction_cost
        if quantity <= 0:
            return
        agent.cash -= notional + fee
        agent.position_quantity = float(quantity)
        agent.current_position = float(quantity)
        agent.entry_price = price
        agent.current_capital = float(agent.cash + agent.position_quantity * price)
        agent.record_trade(
            timestamp=row.get("Date", row.get("date", row.name)),
            action=action,
            price=price,
            quantity=float(quantity),
            transaction_cost=fee,
            realized_pnl=0.0,
            cash_after=agent.cash,
            equity_after=agent.current_capital,
            reason=reason,
        )

    def close_position(self, agent: Agent, row: pd.Series, *, action: str, reason: str) -> None:
        if agent.position_quantity <= 0 or agent.entry_price is None:
            return
        price = float(row.get("Close", row.get("ClosePrice", 0.0)))
        if reason == "STOP_LOSS":
            exit_price = agent.entry_price * (1.0 - agent.stop_loss)
        elif reason == "TAKE_PROFIT":
            exit_price = agent.entry_price * (1.0 + agent.take_profit)
        else:
            exit_price = price
        quantity = agent.position_quantity
        gross = quantity * exit_price
        fee = gross * agent.transaction_cost
        realized_pnl = (exit_price - agent.entry_price) * quantity - fee
        agent.cash += gross - fee
        agent.statistics["realized_pnl"] = float(agent.statistics.get("realized_pnl", 0.0) + realized_pnl)
        agent.position_quantity = 0.0
        agent.current_position = 0.0
        agent.entry_price = None
        agent.current_capital = float(agent.cash)
        agent.record_trade(
            timestamp=row.get("Date", row.get("date", row.name)),
            action=action,
            price=exit_price,
            quantity=float(quantity),
            transaction_cost=fee,
            realized_pnl=realized_pnl,
            cash_after=agent.cash,
            equity_after=agent.current_capital,
            reason=reason,
        )

    def step(self, agent: Agent, row: pd.Series, forecast: float, context: list[float] | None = None) -> dict[str, Any]:
        if context is None:
            context = []

        trigger_reason, trigger_price = self._trigger_exit(agent, row)
        if trigger_reason is not None:
            self.close_position(agent, row, action="SELL", reason=trigger_reason)
            agent._append_equity(price=trigger_price)
            return {"action": "SELL", "reason": trigger_reason, "score": float(forecast)}

        action_info = agent.decide(forecast, context)
        action = action_info["action"]
        if action == "BUY" and agent.position_quantity <= 0:
            self.open_long(agent, row, action="BUY", reason="ENTRY")
            agent._append_equity(price=float(row.get("Close", row.get("ClosePrice", 0.0))))
            return {"action": "BUY", "reason": "ENTRY", "score": float(action_info["score"])}
        if action == "SELL" and agent.position_quantity > 0:
            self.close_position(agent, row, action="SELL", reason="EXIT_SIGNAL")
            agent._append_equity(price=float(row.get("Close", row.get("ClosePrice", 0.0))))
            return {"action": "SELL", "reason": "EXIT_SIGNAL", "score": float(action_info["score"])}
        if action == "HOLD":
            agent._append_equity(price=float(row.get("Close", row.get("ClosePrice", 0.0))))
            return {"action": "HOLD", "reason": "HOLD", "score": float(action_info["score"])}
        agent._append_equity(price=float(row.get("Close", row.get("ClosePrice", 0.0))))
        return {"action": action, "reason": "HOLD", "score": float(action_info["score"]) }

    def simulate_daily_step(self, agent: Agent, row: pd.Series, forecast: float, context: list[float]) -> dict[str, Any]:
        """Execute a decision from prior-day inputs and close all exposure by today's close."""
        cash_before = float(agent.cash)
        trade_start = len(agent.trade_history)
        decision = agent.decide(forecast, context)
        action = str(decision["action"])
        entry: dict[str, Any] | None = None
        exit_event: dict[str, Any] | None = None

        if action == "BUY" and agent.position_quantity <= 0:
            entry_row = row.copy()
            entry_row["Close"] = float(row.get("Open", row.get("open", row.get("Close", 0.0))))
            self.open_long(agent, entry_row, action="BUY", reason="ENTRY")
            if len(agent.trade_history) > trade_start:
                entry = agent.trade_history[-1]

        if agent.position_quantity > 0:
            exit_reason, exit_price = self._trigger_exit(agent, row)
            reason = exit_reason or "END_OF_DAY"
            self.close_position(agent, row, action="SELL", reason=reason)
            if len(agent.trade_history) > trade_start:
                exit_event = agent.trade_history[-1]
            if exit_event is not None and exit_price is not None:
                exit_event["price"] = float(exit_price)
            agent._append_equity(price=float(row.get("Close", row.get("ClosePrice", 0.0))))
        else:
            agent._append_equity(price=float(row.get("Close", row.get("ClosePrice", 0.0))))

        trades = agent.trade_history[trade_start:]
        daily_pnl = float(agent.cash - cash_before)
        agent.statistics.setdefault("daily_pnl", []).append(daily_pnl)
        return {
            "agent_id": agent.agent_id,
            "forecast": float(forecast),
            "decision_score": float(decision["score"]),
            "threshold": float(decision["threshold"]),
            "action": action,
            "signal": float(decision.get("signal", decision["score"])),
            "position": float(agent.position_quantity),
            "daily_pnl": daily_pnl,
            "capital": float(agent.cash),
            "trade_count": len(agent.trade_history),
            "entry": self._trade_event_summary(entry),
            "exit": self._trade_event_summary(exit_event),
            "trades": [self._trade_event_summary(trade) for trade in trades],
        }

    @staticmethod
    def _trade_event_summary(trade: dict[str, Any] | None) -> dict[str, Any] | None:
        if trade is None:
            return None
        return {
            "timestamp": str(trade["timestamp"]),
            "action": trade["action"],
            "price": float(trade["price"]),
            "quantity": float(trade["quantity"]),
            "realized_pnl": float(trade["realized_pnl"]),
            "reason": trade["reason"],
        }

    def _forecast_for_day(self, market: pd.DataFrame, idx: int, forecaster: Any | None, *, feature_columns: list[str] | None = None) -> float:
        if forecaster is None:
            return 0.0
        if feature_columns is None:
            feature_columns = [col for col in market.columns if col not in {"Date", "date"}]
        model_sequence = int(getattr(getattr(forecaster, "config", None), "sequence_length", self.context_length + 1))
        model_feature_names = getattr(forecaster, "feature_names", None)
        if model_feature_names:
            selected = [name for name in model_feature_names if name in market.columns]
            if not selected:
                raise ValueError("The saved forecaster feature metadata does not match the prepared market dataset columns.")
            feature_columns = selected
        if not feature_columns:
            raise ValueError("No feature columns are available for forecasting.")
        history = market.iloc[:idx + 1].copy()
        if len(history) == 0:
            return 0.0
        if len(history) < model_sequence:
            return 0.0
        window = history.loc[:, feature_columns].tail(model_sequence).to_numpy(dtype=float)
        prediction = forecaster.predict(window[np.newaxis, :, :])
        values = np.asarray(prediction, dtype=np.float64).reshape(-1)
        return float(values[0]) if values.size else 0.0

    def simulate_daily(self, agent: Agent, market: pd.DataFrame, forecaster: Any | None = None, *, feature_columns: list[str] | None = None) -> Agent:
        if not isinstance(market, pd.DataFrame):
            raise TypeError("market must be a pandas.DataFrame.")
        if market.empty:
            return agent
        if agent.equity_history:
            agent.equity_history = [agent.equity_history[0]]
        else:
            agent.equity_history = [float(agent.current_capital)]

        for idx, row in market.iterrows():
            previous_history = market.iloc[:idx].copy()
            forecast = self._forecast_for_day(previous_history, len(previous_history) - 1, forecaster, feature_columns=feature_columns) if not previous_history.empty else 0.0
            context = self._normalized_context(previous_history, max(0, len(previous_history) - 1)) if not previous_history.empty else [0.0] * self.context_length
            self.simulate_daily_step(agent, row, forecast, context)

        agent.statistics["equity_curve"] = list(agent.equity_history)
        agent.statistics["ending_capital"] = float(agent.cash)
        return agent

    def simulate(self, agent: Agent, market: pd.DataFrame, forecaster: Any | None = None, *, feature_columns: list[str] | None = None) -> Agent:
        if not isinstance(market, pd.DataFrame):
            raise TypeError("market must be a pandas.DataFrame.")
        if agent.equity_history:
            agent.equity_history = [agent.equity_history[0]]
        else:
            agent.equity_history = [agent.current_capital]

        for idx, row in market.iterrows():
            recent_context = self._normalized_context(market.iloc[: idx + 1], idx)
            if forecaster is None:
                forecast = 0.0
            else:
                if feature_columns is None:
                    feature_columns = [c for c in market.columns if c not in {"Date", "date"}]
                model_sequence = int(getattr(getattr(forecaster, "config", None), "sequence_length", self.context_length + 1))
                model_feature_names = getattr(forecaster, "feature_names", None)
                if model_feature_names:
                    selected = [name for name in model_feature_names if name in market.columns]
                    if not selected:
                        raise ValueError("The saved forecaster feature metadata does not match the prepared market dataset columns.")
                    feature_columns = selected
                if not feature_columns:
                    raise ValueError("No feature columns are available for forecasting.")
                history = market.loc[:row.name, feature_columns].tail(model_sequence).copy()
                if len(history) < model_sequence:
                    continue
                window = history.to_numpy(dtype=float)
                prediction = forecaster.predict(window[np.newaxis, :, :])
                forecast = float(np.asarray(prediction).reshape(-1)[0]) if np.asarray(prediction).size else 0.0
            self.step(agent, row, forecast, recent_context)
        agent.statistics["equity_curve"] = agent.equity_history
        agent.statistics["ending_capital"] = float(agent.cash + (agent.position_quantity * (float(row.get("Close", row.get("ClosePrice", 0.0))) if not market.empty else 0.0)))
        return agent


__all__ = ["TradingEngine"]
