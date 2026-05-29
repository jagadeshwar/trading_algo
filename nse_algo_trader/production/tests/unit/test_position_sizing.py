"""Unit tests — Kelly criterion position sizing math."""

import pytest


def kelly_size(win_rate: float, reward_risk: float, kelly_fraction: float = 0.5) -> float:
    """Half-Kelly position size as fraction of capital."""
    if reward_risk <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    full_kelly = win_rate - (1 - win_rate) / reward_risk
    return max(0.0, full_kelly * kelly_fraction)


class TestKelly:
    def test_positive_edge(self):
        # 55% win rate, 1.5 R:R → should give a positive size
        size = kelly_size(0.55, 1.5)
        assert size > 0

    def test_negative_edge_returns_zero(self):
        # 40% win rate, 1.0 R:R → negative Kelly → floor at 0
        size = kelly_size(0.40, 1.0)
        assert size == 0.0

    def test_half_kelly_applied(self):
        full = kelly_size(0.6, 2.0, kelly_fraction=1.0)
        half = kelly_size(0.6, 2.0, kelly_fraction=0.5)
        assert abs(half - full / 2) < 1e-9

    def test_size_capped(self):
        # Even with very high win rate, half-Kelly should stay < 0.5
        size = kelly_size(0.80, 3.0, kelly_fraction=0.5)
        assert size < 0.5

    def test_zero_win_rate(self):
        assert kelly_size(0.0, 2.0) == 0.0

    def test_zero_reward_risk(self):
        assert kelly_size(0.55, 0.0) == 0.0

    def test_max_position_cap():
        # TODO: test that position sizing respects max_position_pct from risk.yaml
        pass
