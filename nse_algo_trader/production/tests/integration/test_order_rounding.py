"""Integration tests — NSE tick-size rounding rules."""

import pytest


def round_to_tick(price: float, tick: float) -> float:
    """Round price to nearest NSE tick size."""
    return round(round(price / tick) * tick, 10)


class TestTickRounding:
    def test_options_tick_005(self):
        # NSE options tick = 0.05
        assert round_to_tick(45.03, 0.05) == 45.05
        assert round_to_tick(45.02, 0.05) == 45.00
        assert round_to_tick(45.075, 0.05) == 45.10

    def test_futures_tick_010(self):
        # NSE futures tick = 0.10
        assert round_to_tick(18234.14, 0.10) == 18234.10
        assert round_to_tick(18234.15, 0.10) == 18234.20

    def test_equity_tick_005(self):
        # NSE equity tick = 0.05
        assert round_to_tick(2543.03, 0.05) == 2543.05

    def test_zero_price_returns_zero(self):
        assert round_to_tick(0.0, 0.05) == 0.0

    def test_round_does_not_exceed_ask():
        # TODO: test limit buy order is never rounded above the ask
        pass

    def test_round_does_not_go_below_bid():
        # TODO: test limit sell order is never rounded below the bid
        pass
