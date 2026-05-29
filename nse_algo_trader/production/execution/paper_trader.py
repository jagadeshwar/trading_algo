"""Paper trading engine — simulates order execution with realistic NSE costs.

Tracks open positions, applies ATR stops/targets on every bar,
logs every fill and risk event to the database, and integrates
the CircuitBreaker so paper mode behaviour matches live exactly.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from loguru import logger

from production.risk.circuit_breakers import CircuitBreaker
from production.risk.position_sizer import PositionSizer
from production.strategy.momentum import Signal


@dataclass
class Position:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol: str = ""
    direction: Literal[1, -1] = 1
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    qty: int = 0
    entry_time: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    atr: float = 0.0
    confidence: float = 0.0

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_price)

    @property
    def total_risk(self) -> float:
        return self.risk_per_share * self.qty

    @property
    def value(self) -> float:
        return self.entry_price * self.qty


@dataclass
class Fill:
    position_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    qty: int
    price: float
    time: pd.Timestamp
    pnl: float = 0.0
    costs: float = 0.0
    reason: str = ""


def _load_costs(path: str = "configs/broker.yaml") -> dict:
    try:
        return yaml.safe_load(Path(path).read_text()).get("transaction_costs", {})
    except Exception:
        return {}


def _compute_costs(qty: int, price: float, side: str, costs: dict) -> float:
    """Estimate total NSE transaction costs for one fill."""
    value = qty * price
    brokerage   = costs.get("brokerage_per_order", 20)
    stt         = value * costs.get("stt_equity_sell_pct", 0.025) / 100 if side == "SELL" else 0
    exch_fee    = value * costs.get("exchange_fee_pct", 0.00345) / 100
    gst         = (brokerage + exch_fee) * costs.get("gst_pct", 18) / 100
    stamp       = value * costs.get("stamp_duty_equity_buy_pct", 0.003) / 100 if side == "BUY" else 0
    slippage    = value * costs.get("slippage_pct", 0.02) / 100
    return brokerage + stt + exch_fee + gst + stamp + slippage


class PaperTrader:
    """Simulates a live trading session without sending real orders.

    Usage:
        trader = PaperTrader(capital=500_000)
        for bar in bars:
            signal = strategy.evaluate_bar(...)
            if signal:
                trader.on_signal(signal)
            trader.update_positions(current_prices)
        print(trader.summary())
    """

    def __init__(
        self,
        capital: float = 500_000,
        config_path: str = "configs/risk.yaml",
        strategy_version: str = "momentum_v1.0",
        model_version: str = "rule_based_v1",
        use_db: bool = True,
    ) -> None:
        self.capital = capital
        self.cash = capital
        self.strategy_version = strategy_version
        self.model_version = model_version
        self.use_db = use_db

        self._positions: dict[str, Position] = {}  # symbol → open position
        self._current_prices: dict[str, float] = {}  # live prices for MTM
        self._fills: list[Fill] = []
        self._peak_capital = capital
        self._daily_start_capital = capital
        self._costs_cfg = _load_costs()
        self.sizer = PositionSizer(capital, config_path)
        self.circuit = CircuitBreaker(
            alert_fn=lambda msg: logger.warning("ALERT: {}", msg)
        )
        self._db_available = False

        if use_db:
            try:
                from production.db.database import ping
                self._db_available = ping()
            except Exception:
                logger.warning("DB not available — paper trader running without DB logging")

    # ── Main hooks ────────────────────────────────────────────────────────────

    def on_signal(self, signal: Signal) -> Fill | None:
        """Process an entry signal. Returns Fill if position opened."""
        self.circuit.record_tick()

        state = self.circuit.check_all(
            daily_pnl_pct=self._daily_pnl_pct(),
            daily_dd_pct=self._daily_dd_pct(),
        )
        if state.broken:
            logger.warning("Signal blocked — circuit broken: {}", state.reason)
            return None

        if signal.symbol in self._positions:
            logger.debug("Already in position for {}, skipping signal", signal.symbol)
            return None

        if signal.direction == 0:
            return None

        qty = self.sizer.size(
            confidence=signal.confidence,
            entry_price=signal.entry_price,
            stop_price=signal.stop_price,
            current_capital=self.equity,
            current_exposure=self._total_exposure(),
        )
        if qty == 0:
            logger.debug("PositionSizer returned 0 shares for {}", signal.symbol)
            return None

        costs = _compute_costs(qty, signal.entry_price,
                               "BUY" if signal.direction == 1 else "SELL",
                               self._costs_cfg)
        position = Position(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_price=signal.stop_price,
            target_price=signal.target_price,
            qty=qty,
            entry_time=signal.time,
            atr=signal.atr,
            confidence=signal.confidence,
        )
        self._positions[signal.symbol] = position
        self.cash -= position.value + costs

        fill = Fill(
            position_id=position.id,
            symbol=signal.symbol,
            side="BUY" if signal.direction == 1 else "SELL",
            qty=qty,
            price=signal.entry_price,
            time=signal.time,
            costs=costs,
            reason=f"ENTRY confidence={signal.confidence:.2f} {signal.reason}",
        )
        self._fills.append(fill)
        self._log_fill(fill, signal)

        logger.info(
            "PAPER ENTRY  {} {} {} @ {:.2f}  stop={:.2f}  target={:.2f}  "
            "qty={}  conf={:.0%}",
            "LONG" if signal.direction == 1 else "SHORT",
            signal.symbol, signal.time.strftime("%H:%M"),
            signal.entry_price, signal.stop_price, signal.target_price,
            qty, signal.confidence,
        )
        return fill

    def update_positions(
        self,
        current_prices: dict[str, float],
        latest_features: dict[str, pd.Series] | None = None,
    ) -> list[Fill]:
        """Check each open position for stop/target/reversal. Returns any exit fills."""
        self.circuit.record_tick()
        self._current_prices.update(current_prices)   # keep MTM fresh
        exits: list[Fill] = []

        for symbol, pos in list(self._positions.items()):
            price = current_prices.get(symbol)
            if price is None:
                continue

            exit_now, reason = self._check_exit(pos, price, latest_features)
            if exit_now:
                fill = self._close_position(symbol, price, reason)
                if fill:
                    exits.append(fill)

        return exits

    # ── Metrics ───────────────────────────────────────────────────────────────

    @property
    def equity(self) -> float:
        """Cash + collateral locked in positions + unrealized P&L at current prices."""
        position_equity = 0.0
        for sym, pos in self._positions.items():
            current = self._current_prices.get(sym, pos.entry_price)
            # Collateral returned + unrealized P&L
            position_equity += pos.value + pos.direction * (current - pos.entry_price) * pos.qty
        return self.cash + position_equity

    @property
    def total_pnl(self) -> float:
        return self.equity - self.capital

    @property
    def total_pnl_pct(self) -> float:
        return self.total_pnl / self.capital * 100

    @property
    def closed_trades(self) -> list[Fill]:
        return [f for f in self._fills if f.pnl != 0]

    @property
    def win_rate(self) -> float:
        closed = self.closed_trades
        if not closed:
            return 0.0
        wins = sum(1 for f in closed if f.pnl > 0)
        return wins / len(closed)

    @property
    def profit_factor(self) -> float:
        closed = self.closed_trades
        gross_profit = sum(f.pnl for f in closed if f.pnl > 0)
        gross_loss   = abs(sum(f.pnl for f in closed if f.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    def summary(self) -> dict:
        closed = self.closed_trades
        total_costs = sum(f.costs for f in self._fills)
        return {
            "capital":        self.capital,
            "equity":         round(self.equity, 2),
            "total_pnl":      round(self.total_pnl, 2),
            "total_pnl_pct":  round(self.total_pnl_pct, 3),
            "total_trades":   len(closed),
            "open_positions": len(self._positions),
            "win_rate":       round(self.win_rate * 100, 1),
            "profit_factor":  round(self.profit_factor, 2),
            "total_costs":    round(total_costs, 2),
            "daily_pnl_pct":  round(self._daily_pnl_pct(), 3),
            "max_dd_pct":     round(self._daily_dd_pct(), 3),
            "sizer_stats":    self.sizer.stats,
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _check_exit(
        self, pos: Position, price: float, features: dict | None
    ) -> tuple[bool, str]:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now_ist = datetime.now(ZoneInfo("Asia/Kolkata")).time()
        from datetime import time as dtime
        if now_ist >= dtime(15, 25):
            return True, f"EOD_EXIT 15:25 forced close price={price:.2f}"

        if pos.direction == 1:
            if price <= pos.stop_price:
                return True, f"STOP price={price:.2f} stop={pos.stop_price:.2f}"
            if price >= pos.target_price:
                return True, f"TARGET price={price:.2f} target={pos.target_price:.2f}"
        else:
            if price >= pos.stop_price:
                return True, f"STOP price={price:.2f} stop={pos.stop_price:.2f}"
            if price <= pos.target_price:
                return True, f"TARGET price={price:.2f} target={pos.target_price:.2f}"

        if features and pos.symbol in features:
            f = features[pos.symbol]
            cross = f.get("ema_9_21_cross", 0)
            if pos.direction == 1 and cross < 0:
                return True, "REVERSAL bearish crossover"
            if pos.direction == -1 and cross > 0:
                return True, "REVERSAL bullish crossover"

        return False, ""

    def _close_position(self, symbol: str, price: float, reason: str) -> Fill | None:
        pos = self._positions.pop(symbol, None)
        if pos is None:
            return None

        gross_pnl = pos.direction * (price - pos.entry_price) * pos.qty
        costs = _compute_costs(pos.qty, price,
                               "SELL" if pos.direction == 1 else "BUY",
                               self._costs_cfg)
        net_pnl = gross_pnl - costs

        # Cash accounting (entry deducted full collateral for both long and short):
        #   Long  exit: return sell proceeds (exit_price × qty - costs)
        #   Short exit: return collateral (entry × qty) + PnL ((entry-exit) × qty) - costs
        if pos.direction == 1:
            self.cash += price * pos.qty - costs
        else:
            self.cash += pos.entry_price * pos.qty + (pos.entry_price - price) * pos.qty - costs

        self._peak_capital = max(self._peak_capital, self.equity)
        self.sizer.update_stats(net_pnl, pos.total_risk)

        fill = Fill(
            position_id=pos.id,
            symbol=symbol,
            side="SELL" if pos.direction == 1 else "BUY",
            qty=pos.qty,
            price=price,
            time=pd.Timestamp.now(tz="Asia/Kolkata"),
            pnl=net_pnl,
            costs=costs,
            reason=reason,
        )
        self._fills.append(fill)

        icon = "✓" if net_pnl > 0 else "✗"
        logger.info(
            "PAPER EXIT {} {} @ {:.2f}  PnL={:+.0f}  reason={}  {}",
            symbol, fill.time.strftime("%H:%M"),
            price, net_pnl, reason, icon,
        )
        return fill

    def _daily_pnl_pct(self) -> float:
        return (self.equity - self._daily_start_capital) / self._daily_start_capital * 100

    def _daily_dd_pct(self) -> float:
        if self._peak_capital <= 0:
            return 0.0
        return (self._peak_capital - self.equity) / self._peak_capital * 100

    def _total_exposure(self) -> float:
        return sum(pos.value for pos in self._positions.values())

    def _log_fill(self, fill: Fill, signal: Signal) -> None:
        if not self._db_available:
            return
        try:
            from production.db.database import log_signal, log_order
            sig_id = log_signal(
                time=signal.time,
                symbol=signal.symbol,
                interval_min=5,
                direction=signal.direction,
                confidence=signal.confidence,
                regime=signal.regime,
                model_version=self.model_version,
                strategy_version=self.strategy_version,
            )
            log_order(
                signal_id=sig_id,
                broker="paper",
                broker_order_id=fill.position_id,
                symbol=signal.symbol,
                side=fill.side,
                order_type="MARKET",
                qty=fill.qty,
                price=fill.price,
                trigger_price=None,
                status="FILLED",
                strategy_version=self.strategy_version,
                model_version=self.model_version,
            )
        except Exception as e:
            logger.debug("DB log failed (non-critical): {}", e)

    def reset_daily(self) -> None:
        """Call at start of each trading day."""
        self._daily_start_capital = self.equity
        self._peak_capital = self.equity
        self.circuit.record_tick()
        self.circuit.record_heartbeat()
