"""Simulation tests — circuit breaker trigger and state verification."""

import time
import pytest

from production.risk.circuit_breakers import CircuitBreaker, BreakReason


@pytest.fixture
def cb() -> CircuitBreaker:
    return CircuitBreaker(
        websocket_stale_sec=1,
        websocket_halt_sec=2,
        api_heartbeat_sec=1,
        slippage_spike_multiplier=3.0,
        vix_spike_pct=30.0,
        max_order_rejections=3,
    )


class TestWebSocketStale:
    def test_trips_after_halt_threshold(self, cb):
        time.sleep(2.1)
        state = cb.check_all()
        assert state.broken
        assert state.reason == BreakReason.WEBSOCKET_HALTED

    def test_no_trip_when_ticks_arrive(self, cb):
        for _ in range(5):
            cb.record_tick()
            time.sleep(0.1)
        state = cb.check_all()
        assert not state.broken


class TestDailyLoss:
    def test_trips_at_2pct_loss(self, cb):
        state = cb.check_all(daily_pnl_pct=-2.0)
        assert state.broken
        assert state.reason == BreakReason.DAILY_LOSS_LIMIT

    def test_no_trip_below_threshold(self, cb):
        state = cb.check_all(daily_pnl_pct=-1.9)
        assert not state.broken


class TestDrawdown:
    def test_trips_at_5pct_drawdown(self, cb):
        state = cb.check_all(daily_dd_pct=5.0)
        assert state.broken
        assert state.reason == BreakReason.MAX_DRAWDOWN

    def test_no_trip_below_threshold(self, cb):
        state = cb.check_all(daily_dd_pct=4.9)
        assert not state.broken


class TestModelConfidence:
    def test_trips_on_all_low_confidence(self, cb):
        cb.record_tick()
        cb.record_heartbeat()
        state = cb.check_all(model_confidences=[0.03, 0.02, 0.04])
        assert state.broken
        assert state.reason == BreakReason.MODEL_CONFIDENCE_ANOMALY

    def test_trips_on_all_high_confidence(self, cb):
        cb.record_tick()
        cb.record_heartbeat()
        state = cb.check_all(model_confidences=[0.97, 0.98, 0.96])
        assert state.broken


class TestOrderRejections:
    def test_trips_after_max_rejections(self, cb):
        cb.record_tick()
        cb.record_heartbeat()
        for _ in range(3):
            cb.record_order_rejection()
        assert cb.is_broken
        assert cb.state.reason == BreakReason.TOO_MANY_REJECTIONS


class TestManualReset:
    def test_reset_clears_broken_state(self, cb):
        cb.check_all(daily_pnl_pct=-2.0)
        assert cb.is_broken
        cb.reset()
        assert not cb.is_broken
        assert cb.state.reason is None

    def test_broken_state_persists_across_checks(self, cb):
        cb.check_all(daily_pnl_pct=-2.0)
        cb.check_all()  # second call should still be broken
        assert cb.is_broken
