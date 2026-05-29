"""CircuitBreaker — automated safety shutdowns for live trading.

Manual reset required after any circuit break. Never auto-restart.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto

from loguru import logger


class BreakReason(Enum):
    WEBSOCKET_STALE = auto()
    WEBSOCKET_HALTED = auto()
    API_HEARTBEAT_LOST = auto()
    MODEL_CONFIDENCE_ANOMALY = auto()
    SLIPPAGE_SPIKE = auto()
    VIX_EXPLOSION = auto()
    TOO_MANY_REJECTIONS = auto()
    EXCHANGE_FREEZE = auto()
    DAILY_LOSS_LIMIT = auto()
    MAX_DRAWDOWN = auto()


@dataclass
class CircuitState:
    broken: bool = False
    reason: BreakReason | None = None
    triggered_at: datetime | None = None
    reduced_size: bool = False
    halt_new_entries: bool = False


@dataclass
class _RejectionTracker:
    count: int = 0
    window_start: datetime = field(default_factory=datetime.utcnow)
    window_seconds: int = 600

    def record(self) -> int:
        now = datetime.utcnow()
        if (now - self.window_start).total_seconds() > self.window_seconds:
            self.count = 0
            self.window_start = now
        self.count += 1
        return self.count

    def reset(self) -> None:
        self.count = 0
        self.window_start = datetime.utcnow()


class CircuitBreaker:
    """Check all safety conditions and trip circuit breaks when needed.

    Call check_all() on every bar. Trip states must be manually reset
    by calling reset() after investigating the cause.
    """

    def __init__(
        self,
        websocket_stale_sec: int = 30,
        websocket_halt_sec: int = 60,
        api_heartbeat_sec: int = 30,
        slippage_spike_multiplier: float = 3.0,
        vix_spike_pct: float = 30.0,
        max_order_rejections: int = 3,
        model_confidence_low: float = 0.05,
        model_confidence_high: float = 0.95,
        alert_fn=None,
    ) -> None:
        self.ws_stale_sec = websocket_stale_sec
        self.ws_halt_sec = websocket_halt_sec
        self.api_hb_sec = api_heartbeat_sec
        self.slippage_multiplier = slippage_spike_multiplier
        self.vix_spike_pct = vix_spike_pct
        self.max_rejections = max_order_rejections
        self.confidence_low = model_confidence_low
        self.confidence_high = model_confidence_high
        self._alert = alert_fn or self._log_alert

        self.state = CircuitState()
        self._rejections = _RejectionTracker()
        self._last_tick_time: float = time.monotonic()
        self._last_heartbeat_time: float = time.monotonic()
        self._recent_slippages: list[float] = []
        self._last_vix: float | None = None
        self._vix_check_start: datetime | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def check_all(
        self,
        model_confidences: list[float] | None = None,
        current_vix: float | None = None,
        daily_pnl_pct: float | None = None,
        daily_dd_pct: float | None = None,
    ) -> CircuitState:
        """Run all checks. Returns current state (broken or not)."""
        if self.state.broken:
            return self.state  # already broken, wait for manual reset

        self._check_websocket_health()
        self._check_api_heartbeat()
        if model_confidences:
            self._check_model_confidence(model_confidences)
        if current_vix is not None:
            self._check_vix_explosion(current_vix)
        if daily_pnl_pct is not None:
            self._check_daily_loss(daily_pnl_pct)
        if daily_dd_pct is not None:
            self._check_drawdown(daily_dd_pct)

        return self.state

    def record_tick(self) -> None:
        """Call on every WebSocket tick received."""
        self._last_tick_time = time.monotonic()

    def record_heartbeat(self) -> None:
        """Call on every Fyers API heartbeat received."""
        self._last_heartbeat_time = time.monotonic()

    def record_slippage(self, expected_pct: float, actual_pct: float) -> None:
        """Record a trade slippage observation."""
        self._recent_slippages.append(actual_pct)
        if len(self._recent_slippages) > 5:
            self._recent_slippages = self._recent_slippages[-5:]
        if not self.state.broken:
            self._check_slippage_spike(expected_pct)

    def record_order_rejection(self) -> None:
        """Call whenever a broker order is rejected."""
        count = self._rejections.record()
        if not self.state.broken and count >= self.max_rejections:
            self._trip(BreakReason.TOO_MANY_REJECTIONS,
                       f"{count} rejections in {self._rejections.window_seconds}s")

    def reset(self) -> None:
        """Manually reset after investigating the circuit break cause."""
        logger.warning("CircuitBreaker manually reset (was: {})", self.state.reason)
        self.state = CircuitState()
        self._rejections.reset()
        self._last_tick_time = time.monotonic()
        self._last_heartbeat_time = time.monotonic()
        self._recent_slippages.clear()

    @property
    def is_broken(self) -> bool:
        return self.state.broken

    @property
    def halt_entries(self) -> bool:
        return self.state.broken or self.state.halt_new_entries

    @property
    def size_multiplier(self) -> float:
        if self.state.broken:
            return 0.0
        if self.state.reduced_size:
            return 0.5
        return 1.0

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_websocket_health(self) -> None:
        elapsed = time.monotonic() - self._last_tick_time
        if elapsed > self.ws_halt_sec:
            self._trip(BreakReason.WEBSOCKET_HALTED,
                       f"No tick for {elapsed:.0f}s (halt threshold {self.ws_halt_sec}s)")
        elif elapsed > self.ws_stale_sec and not self.state.halt_new_entries:
            self.state.halt_new_entries = True
            msg = f"WebSocket stale {elapsed:.0f}s — halting new entries"
            logger.warning(msg)
            self._alert(msg)

    def _check_api_heartbeat(self) -> None:
        elapsed = time.monotonic() - self._last_heartbeat_time
        if elapsed > self.api_hb_sec:
            self._trip(BreakReason.API_HEARTBEAT_LOST,
                       f"Fyers heartbeat missing for {elapsed:.0f}s")

    def _check_model_confidence(self, confidences: list[float]) -> None:
        if all(c < self.confidence_low for c in confidences):
            self._trip(BreakReason.MODEL_CONFIDENCE_ANOMALY,
                       f"All confidences < {self.confidence_low} — model may be broken")
        elif all(c > self.confidence_high for c in confidences):
            self._trip(BreakReason.MODEL_CONFIDENCE_ANOMALY,
                       f"All confidences > {self.confidence_high} — model may be broken")

    def _check_slippage_spike(self, expected_pct: float) -> None:
        if len(self._recent_slippages) < 5:
            return
        avg_actual = sum(self._recent_slippages) / len(self._recent_slippages)
        if avg_actual > expected_pct * self.slippage_multiplier and not self.state.reduced_size:
            self.state.reduced_size = True
            msg = (f"Slippage spike: avg {avg_actual:.4f}% vs expected {expected_pct:.4f}% "
                   f"— reducing size 50%")
            logger.warning(msg)
            self._alert(msg)

    def _check_vix_explosion(self, current_vix: float) -> None:
        now = datetime.utcnow()
        if self._last_vix is None:
            self._last_vix = current_vix
            self._vix_check_start = now
            return

        if self._vix_check_start is None or (now - self._vix_check_start) > timedelta(minutes=15):
            self._last_vix = current_vix
            self._vix_check_start = now
            return

        if self._last_vix > 0:
            spike_pct = (current_vix - self._last_vix) / self._last_vix * 100
            if spike_pct > self.vix_spike_pct:
                self._trip(BreakReason.VIX_EXPLOSION,
                           f"India VIX spiked {spike_pct:.1f}% in 15min (current: {current_vix})")

    def _check_daily_loss(self, daily_pnl_pct: float) -> None:
        if daily_pnl_pct <= -2.0:
            self._trip(BreakReason.DAILY_LOSS_LIMIT,
                       f"Daily loss limit hit: {daily_pnl_pct:.2f}%")

    def _check_drawdown(self, daily_dd_pct: float) -> None:
        if daily_dd_pct >= 5.0:
            self._trip(BreakReason.MAX_DRAWDOWN,
                       f"Daily drawdown limit hit: {daily_dd_pct:.2f}%")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _trip(self, reason: BreakReason, detail: str) -> None:
        self.state.broken = True
        self.state.reason = reason
        self.state.triggered_at = datetime.utcnow()
        msg = f"CIRCUIT BREAK [{reason.name}]: {detail}. Manual reset required."
        logger.critical(msg)
        self._alert(msg)

    @staticmethod
    def _log_alert(msg: str) -> None:
        logger.warning("ALERT: {}", msg)
