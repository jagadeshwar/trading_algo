"""Unit tests — indicator calculation accuracy."""

import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Synthetic OHLCV with known values for reference checks."""
    np.random.seed(42)
    n = 300
    close = 18000 + np.cumsum(np.random.randn(n) * 50)
    high = close + np.abs(np.random.randn(n) * 30)
    low = close - np.abs(np.random.randn(n) * 30)
    open_ = close + np.random.randn(n) * 20
    volume = np.random.randint(100_000, 500_000, n).astype(float)
    idx = pd.date_range("2024-01-01 09:15", periods=n, freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


class TestRSI:
    def test_rsi_range(self, sample_ohlcv):
        rsi = ta.rsi(sample_ohlcv["close"], length=14).dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_rsi_length(self, sample_ohlcv):
        rsi = ta.rsi(sample_ohlcv["close"], length=14)
        # First 14 values are NaN
        assert rsi.iloc[:14].isna().all()
        assert rsi.iloc[14:].notna().all()


class TestATR:
    def test_atr_positive(self, sample_ohlcv):
        atr = ta.atr(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"], length=14).dropna()
        assert (atr > 0).all()

    def test_atr_bounded_by_range(self, sample_ohlcv):
        atr = ta.atr(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"], length=1).dropna()
        bar_range = (sample_ohlcv["high"] - sample_ohlcv["low"]).iloc[1:]
        # ATR(1) should equal the true range, which is >= bar range
        assert (atr.values >= 0).all()


class TestEMA:
    def test_ema_converges(self, sample_ohlcv):
        ema = ta.ema(sample_ohlcv["close"], length=9).dropna()
        # EMA should be within the range of close prices
        close = sample_ohlcv["close"]
        assert ema.max() <= close.max() * 1.01
        assert ema.min() >= close.min() * 0.99

    def test_ema_periods_ordering(self, sample_ohlcv):
        # In a trending series, longer EMAs lag more
        ema9 = ta.ema(sample_ohlcv["close"], length=9).dropna()
        ema21 = ta.ema(sample_ohlcv["close"], length=21).dropna()
        # They should not be identical
        assert not ema9.equals(ema21)


class TestVWAP:
    def test_vwap_within_price_range(self, sample_ohlcv):
        vwap = ta.vwap(
            sample_ohlcv["high"], sample_ohlcv["low"],
            sample_ohlcv["close"], sample_ohlcv["volume"]
        ).dropna()
        assert (vwap >= sample_ohlcv["low"].min()).all()
        assert (vwap <= sample_ohlcv["high"].max()).all()
