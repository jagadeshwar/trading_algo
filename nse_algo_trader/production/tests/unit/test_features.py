"""Unit tests — FeatureEngineer output validation."""

import numpy as np
import pandas as pd
import pytest

from production.data.features import FeatureEngineer, FEATURE_NAMES


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    np.random.seed(0)
    n = 400
    close = 18000 + np.cumsum(np.random.randn(n) * 50)
    high = close + np.abs(np.random.randn(n) * 30)
    low = close - np.abs(np.random.randn(n) * 30)
    open_ = close + np.random.randn(n) * 20
    volume = np.random.randint(100_000, 500_000, n).astype(float)
    idx = pd.date_range("2024-01-01 09:15", periods=n, freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


@pytest.fixture
def features(sample_ohlcv) -> pd.DataFrame:
    return FeatureEngineer().compute(sample_ohlcv)


class TestFeatureShape:
    def test_no_nan_rows(self, features):
        assert not features.isnull().any().any(), "Feature DataFrame should have no NaN rows after dropna"

    def test_target_column_present(self, features):
        assert "target" in features.columns

    def test_expected_columns_present(self, features):
        core = ["ret_1b", "ret_5b", "rsi", "atr_pct", "adx", "target"]
        for col in core:
            assert col in features.columns, f"Missing column: {col}"


class TestTargetLabel:
    def test_target_values(self, features):
        assert set(features["target"].unique()).issubset({-1, 0, 1})

    def test_target_not_all_zero(self, features):
        assert features["target"].nunique() > 1, "Target should have more than one class"


class TestFeatureRanges:
    def test_rsi_range(self, features):
        assert features["rsi"].between(0, 100).all()

    def test_close_position_range(self, features):
        assert features["close_position"].between(0, 1).all()

    def test_bb_pct_b_finite(self, features):
        assert np.isfinite(features["bb_pct_b"]).all()

    def test_vol_ratio_positive(self, features):
        assert (features["vol_ratio"] > 0).all()

    def test_regime_heuristic_values(self, features):
        assert set(features["regime_heuristic"].unique()).issubset({0, 1, 2, 3})


class TestReturnFeatures:
    def test_ret_1b_magnitude(self, features):
        # Log returns on NSE should be small per bar
        assert features["ret_1b"].abs().max() < 0.5

    def test_lag_features_present(self, features):
        for n in [1, 2, 3, 5]:
            assert f"lag_{n}" in features.columns
