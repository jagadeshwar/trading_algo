"""Unit tests — ATR-based stop-loss placement accuracy."""

import pytest


def compute_stop(entry_price: float, atr: float, multiplier: float, direction: int) -> float:
    """direction: +1 for long, -1 for short."""
    return entry_price - direction * atr * multiplier


def compute_target(entry_price: float, atr: float, multiplier: float, direction: int) -> float:
    return entry_price + direction * atr * multiplier


class TestATRStop:
    def test_long_stop_below_entry(self):
        stop = compute_stop(18000, 100, 2.0, direction=1)
        assert stop < 18000

    def test_short_stop_above_entry(self):
        stop = compute_stop(18000, 100, 2.0, direction=-1)
        assert stop > 18000

    def test_stop_distance(self):
        atr = 150.0
        mult = 2.0
        stop = compute_stop(18000, atr, mult, direction=1)
        assert abs((18000 - stop) - atr * mult) < 1e-6

    def test_target_long(self):
        target = compute_target(18000, 100, 3.0, direction=1)
        assert target > 18000

    def test_reward_risk_ratio(self):
        entry = 18000.0
        atr = 100.0
        stop = compute_stop(entry, atr, 2.0, direction=1)
        target = compute_target(entry, atr, 3.0, direction=1)
        reward = target - entry
        risk = entry - stop
        assert abs(reward / risk - 1.5) < 1e-6

    def test_nse_tick_rounding():
        # TODO: test stop rounded to nearest 0.05 (options) or 0.10 (futures)
        pass
