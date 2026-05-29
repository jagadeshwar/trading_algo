"""FeatureEngineer — computes all 35 ML features from OHLCV data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

FEATURES_DIR = Path("data/features")

FEATURE_NAMES = [
    # Price (8)
    "ret_1b", "ret_5b", "ret_10b", "ret_20b",
    "lag_1", "lag_2", "lag_3", "lag_5",
    # Bar (3)
    "bar_range_pct", "close_position", "gap_pct",
    # EMA (8)
    "dist_ema_9", "dist_ema_21", "dist_ema_50", "dist_ema_200",
    "ema_9_21_cross", "ema_21_50_cross", "golden_cross", "ema_21_slope",
    # Momentum (6)
    "rsi", "rsi_oversold", "rsi_overbought", "rsi_slope",
    "macd_norm", "macd_hist_norm", "macd_cross",
    # Volatility (5)
    "atr_pct", "atr_ratio", "bb_width", "bb_pct_b", "bb_squeeze",
    # Volume (4)
    "vwap_dev", "vol_ratio", "obv_norm", "vol_delta_ma",
    # Regime (4) — heuristic + HMM
    "adx", "di_diff", "regime_heuristic", "regime_hmm",
    # Target (1)
    "target",
]


class FeatureEngineer:
    """Compute all 35 ML features + target label from OHLCV DataFrame."""

    def __init__(
        self,
        ema_periods: list[int] | None = None,
        rsi_period: int = 14,
        atr_period: int = 14,
        adx_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        target_bars: int = 3,
        volume_ma_period: int = 20,
    ) -> None:
        self.ema_periods = ema_periods or [9, 21, 50, 200]
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.target_bars = target_bars
        self.volume_ma_period = volume_ma_period

    def compute(self, df: pd.DataFrame, hmm_model_path: str | None = None) -> pd.DataFrame:
        """Return a new DataFrame with all features appended. Drops NaN rows.

        If hmm_model_path is provided (or models/regime_hmm.pkl exists),
        the HMM regime label is appended as an additional feature.
        """
        d = df.copy()
        self._price_features(d)
        self._bar_features(d)
        self._ema_features(d)
        self._momentum_features(d)
        self._volatility_features(d)
        self._volume_features(d)
        self._regime_features(d)
        self._hmm_regime(d, hmm_model_path)
        self._target(d)
        keep = [c for c in FEATURE_NAMES if c in d.columns]
        result = d[keep].dropna()
        logger.debug("Features computed: {} rows × {} columns", len(result), len(result.columns))
        return result

    # ── Price features ────────────────────────────────────────────────────────

    def _price_features(self, d: pd.DataFrame) -> None:
        log_close = np.log(d["close"])
        for n in [1, 5, 10, 20]:
            d[f"ret_{n}b"] = log_close.diff(n)
        for n in [1, 2, 3, 5]:
            d[f"lag_{n}"] = d["close"].pct_change(n)

    # ── Bar features ─────────────────────────────────────────────────────────

    def _bar_features(self, d: pd.DataFrame) -> None:
        d["bar_range_pct"] = (d["high"] - d["low"]) / d["close"]
        hl_range = d["high"] - d["low"]
        d["close_position"] = np.where(
            hl_range > 0, (d["close"] - d["low"]) / hl_range, 0.5
        )
        d["gap_pct"] = (d["open"] - d["close"].shift(1)) / d["close"].shift(1)

    # ── EMA features ──────────────────────────────────────────────────────────

    def _ema_features(self, d: pd.DataFrame) -> None:
        for p in self.ema_periods:
            d[f"ema_{p}"] = ta.ema(d["close"], length=p)
            d[f"dist_ema_{p}"] = (d["close"] - d[f"ema_{p}"]) / d[f"ema_{p}"]

        e9, e21, e50, e200 = (
            d.get("ema_9"), d.get("ema_21"), d.get("ema_50"), d.get("ema_200")
        )
        if e9 is not None and e21 is not None:
            prev_diff = (e9 - e21).shift(1)
            curr_diff = e9 - e21
            d["ema_9_21_cross"] = np.sign(curr_diff) - np.sign(prev_diff)
        if e21 is not None and e50 is not None:
            prev_diff = (e21 - e50).shift(1)
            curr_diff = e21 - e50
            d["ema_21_50_cross"] = np.sign(curr_diff) - np.sign(prev_diff)
            d["ema_21_slope"] = e21.diff(3) / e21
        if e50 is not None and e200 is not None:
            d["golden_cross"] = (e50 > e200).astype(int)

        # Drop raw EMA columns — they are not ML features, only distances are
        for p in self.ema_periods:
            if f"ema_{p}" in d.columns:
                d.drop(columns=[f"ema_{p}"], inplace=True)

    # ── Momentum features ─────────────────────────────────────────────────────

    def _momentum_features(self, d: pd.DataFrame) -> None:
        rsi = ta.rsi(d["close"], length=self.rsi_period)
        d["rsi"] = rsi
        d["rsi_oversold"] = (rsi < 30).astype(int)
        d["rsi_overbought"] = (rsi > 70).astype(int)
        d["rsi_slope"] = rsi.diff(3)

        macd_df = ta.macd(d["close"])
        if macd_df is not None and not macd_df.empty:
            macd_col = [c for c in macd_df.columns if c.startswith("MACD_") and "s" not in c.lower() and "h" not in c.lower()]
            hist_col = [c for c in macd_df.columns if "h" in c.lower() or "MACDH" in c]
            if macd_col:
                d["macd_norm"] = macd_df[macd_col[0]] / d["close"]
            if hist_col:
                d["macd_hist_norm"] = macd_df[hist_col[0]] / d["close"]
                prev_hist = macd_df[hist_col[0]].shift(1)
                curr_hist = macd_df[hist_col[0]]
                d["macd_cross"] = np.sign(curr_hist) - np.sign(prev_hist)

    # ── Volatility features ───────────────────────────────────────────────────

    def _volatility_features(self, d: pd.DataFrame) -> None:
        atr = ta.atr(d["high"], d["low"], d["close"], length=self.atr_period)
        d["atr_pct"] = atr / d["close"]
        atr_ma = atr.rolling(self.volume_ma_period).mean()
        d["atr_ratio"] = atr / atr_ma

        bb = ta.bbands(d["close"], length=self.bb_period, std=self.bb_std)
        if bb is not None and not bb.empty:
            upper_col = [c for c in bb.columns if "U" in c][0]
            lower_col = [c for c in bb.columns if "L" in c][0]
            mid_col = [c for c in bb.columns if "M" in c][0]
            width = bb[upper_col] - bb[lower_col]
            d["bb_width"] = width / bb[mid_col]
            d["bb_pct_b"] = (d["close"] - bb[lower_col]) / (bb[upper_col] - bb[lower_col])
            width_ma = width.rolling(self.volume_ma_period).mean()
            d["bb_squeeze"] = (width < width_ma).astype(int)

    # ── Volume features ───────────────────────────────────────────────────────

    def _volume_features(self, d: pd.DataFrame) -> None:
        vwap = ta.vwap(d["high"], d["low"], d["close"], d["volume"])
        if vwap is not None:
            d["vwap_dev"] = (d["close"] - vwap) / vwap

        vol_ma = d["volume"].rolling(self.volume_ma_period).mean()
        d["vol_ratio"] = d["volume"] / vol_ma

        obv = ta.obv(d["close"], d["volume"])
        if obv is not None:
            obv_std = obv.rolling(self.volume_ma_period).std()
            obv_std = obv_std.replace(0, np.nan)
            d["obv_norm"] = (obv - obv.rolling(self.volume_ma_period).mean()) / obv_std

        # Buy/sell volume proxy: up-bar volume vs down-bar volume
        up_vol = d["volume"].where(d["close"] > d["open"], 0.0)
        dn_vol = d["volume"].where(d["close"] < d["open"], 0.0)
        delta = up_vol - dn_vol
        delta_ma = delta.rolling(self.volume_ma_period).mean()
        vol_ma_safe = vol_ma.replace(0, np.nan)
        d["vol_delta_ma"] = delta_ma / vol_ma_safe

    # ── Regime features ───────────────────────────────────────────────────────

    def _regime_features(self, d: pd.DataFrame) -> None:
        adx_df = ta.adx(d["high"], d["low"], d["close"], length=self.adx_period)
        if adx_df is not None and not adx_df.empty:
            adx_col = [c for c in adx_df.columns if c.startswith("ADX_")][0]
            dmp_col = [c for c in adx_df.columns if "DMP" in c][0]
            dmn_col = [c for c in adx_df.columns if "DMN" in c][0]
            d["adx"] = adx_df[adx_col]
            d["di_diff"] = adx_df[dmp_col] - adx_df[dmn_col]
            adx_vals = adx_df[adx_col]
            atr_pct = d.get("atr_pct", pd.Series(0.0, index=d.index))
            regime = np.where(
                atr_pct > atr_pct.rolling(20).mean() * 1.5, 3,
                np.where(adx_vals > 25,
                    np.where(d["di_diff"] > 0, 1, 2),
                    0
                )
            )
            d["regime_heuristic"] = regime

    def _hmm_regime(self, d: pd.DataFrame, model_path: str | None) -> None:
        """Append HMM regime label if a trained model is available."""
        path = Path(model_path) if model_path else Path("models/regime_hmm.pkl")
        if not path.exists():
            return
        try:
            from production.models.regime_hmm import RegimeDetector
            det    = RegimeDetector().load(path)
            labels = det.predict(d)
            d["regime_hmm"] = labels.reindex(d.index)
        except Exception as e:
            logger.debug("HMM regime skipped: {}", e)

    # ── Target label ──────────────────────────────────────────────────────────

    def _target(self, d: pd.DataFrame) -> None:
        fwd_ret = d["close"].shift(-self.target_bars) / d["close"] - 1
        d["target"] = np.where(fwd_ret > 0.002, 1, np.where(fwd_ret < -0.002, -1, 0))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, df: pd.DataFrame, symbol: str, interval_min: int) -> Path:
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        slug = symbol.replace(":", "_").replace("-", "_")
        path = FEATURES_DIR / f"{slug}_{interval_min}min_features.parquet"
        pq.write_table(pa.Table.from_pandas(df), path, compression="snappy")
        logger.info("Saved features → {}", path)
        return path

    def load(self, symbol: str, interval_min: int) -> pd.DataFrame:
        slug = symbol.replace(":", "_").replace("-", "_")
        path = FEATURES_DIR / f"{slug}_{interval_min}min_features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No feature file at {path}")
        df = pq.read_table(path).to_pandas()
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
        return df
