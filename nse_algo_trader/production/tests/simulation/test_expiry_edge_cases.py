"""Simulation tests — NSE options expiry day behaviour."""

import pytest
from datetime import date


def is_expiry_day(d: date, expiry_dates: list[date]) -> bool:
    return d in expiry_dates


class TestExpiryDetection:
    def test_bank_nifty_weekly_expiry_wednesday(self):
        # TODO: verify weekly expiry detection for Bank Nifty (every Wednesday)
        pass

    def test_nifty_monthly_expiry_last_thursday(self):
        # TODO: verify monthly expiry detection for Nifty (last Thursday of month)
        pass

    def test_regime_switches_to_expiry_on_expiry_day(self):
        # TODO: on expiry day, regime label should be EXPIRY regardless of ADX/VIX
        pass

    def test_mean_reversion_strategy_disabled_on_expiry(self):
        # TODO: mean reversion strategy should be disabled in EXPIRY regime
        pass

    def test_position_size_reduced_on_expiry(self):
        # TODO: gamma risk → position size should be ≤ 50% of normal on expiry day
        pass

    def test_options_flatten_by_3pm_on_expiry(self):
        # TODO: all options positions must be closed by 15:00 on expiry day
        pass
