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
    # Phase 2.1 Strategy Features ─────────────────────────────────────────────
    # Breakout (Donchian Channel)
    "donchian_pos", "breakout_up", "breakout_dn",
    # Volatility Contraction (Keltner squeeze)
    "bb_kc_squeeze",
    # Support / Resistance (pivot-based)
    "pivot_resistance", "pivot_support",
    "near_resistance", "near_support",
    # Candlestick patterns (Price Action)
    "cdl_hammer", "cdl_shooting_star",
    "cdl_bullish_engulf", "cdl_bearish_engulf",
    "cdl_doji", "cdl_pin_bar_bull", "cdl_pin_bar_bear",
    # Relative Strength
    "rs_momentum",
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
        # Volume features: BankNifty 15min and all index daily bars have ~50% zero-volume
        # rows in Fyers historical data. vol_ratio = 0 on those bars means every
        # volume-gated strategy will be silent for half the dataset.
        # Fix: when >= 30% of bars have vol == 0 AND the dataset looks like an index
        # (whole-number volumes, all identical zeros), replace zero-volume bars with 1.0
        # so they are treated as "average volume" and strategies can fire on them.
        # NaN rows are always filled to prevent downstream NaN propagation into HMM.
        if "vol_ratio" in d.columns:
            zero_pct = (d["volume"] == 0).mean() if "volume" in d.columns else 0.0
            if zero_pct >= 0.30:
                # Index symbol with unreliable volume at this timeframe — neutralise zeros
                d["vol_ratio"] = d["vol_ratio"].replace(0, 1.0)
                logger.debug(
                    "vol_ratio: {:.0%} zero-volume bars detected — replacing with 1.0 "
                    "(index symbol or coarse timeframe). Volume-gated signals will use "
                    "ADX/RSI/price filters only.",
                    zero_pct,
                )

        for col in ["vol_ratio", "obv_norm", "vwap_dev", "vol_delta_ma"]:
            if col in d.columns and d[col].isna().any():
                fill = d[col].median() if d[col].notna().any() else (1.0 if col == "vol_ratio" else 0.0)
                d[col] = d[col].fillna(fill)
        self._hmm_regime(d, hmm_model_path)
        # Phase 2.1 strategy-specific features
        self._breakout_features(d)
        self._keltner_squeeze(d)
        self._pivot_sr(d)
        self._candlestick_patterns(d)
        self._relative_strength_features(d)
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

    # ── Phase 2.1 Strategy Features ───────────────────────────────────────────

    def _breakout_features(self, d: pd.DataFrame, period: int = 20) -> None:
        """Donchian Channel breakout detection.

        donchian_pos   : normalised 0–1 position within the channel
        breakout_up    : 1 on bar where close first breaks above prior Donchian high
        breakout_dn    : 1 on bar where close first breaks below prior Donchian low
        """
        don_high = d["high"].rolling(period).max()
        don_low  = d["low"].rolling(period).min()
        width    = (don_high - don_low).replace(0, np.nan)
        d["donchian_pos"] = (d["close"] - don_low) / width

        # Shift by 1: breakout occurs when today's close exceeds YESTERDAY'S channel high
        prev_high = don_high.shift(1)
        prev_low  = don_low.shift(1)
        prev_close = d["close"].shift(1)
        d["breakout_up"] = (
            (d["close"] > prev_high) & (prev_close <= prev_high)
        ).astype(int)
        d["breakout_dn"] = (
            (d["close"] < prev_low) & (prev_close >= prev_low)
        ).astype(int)

    def _keltner_squeeze(self, d: pd.DataFrame) -> None:
        """Keltner Channel squeeze: True when Bollinger Bands are entirely inside the KC.

        Keltner Channel: EMA(20) ± 1.5 × ATR(10)
        bb_kc_squeeze  : 1 when BB upper < KC upper AND BB lower > KC lower
        """
        if "bb_width" not in d.columns:
            return
        try:
            kc_mid   = d["close"].ewm(span=20, adjust=False).mean()
            kc_atr   = ta.atr(d["high"], d["low"], d["close"], length=10)
            if kc_atr is None:
                return
            kc_upper = kc_mid + 1.5 * kc_atr
            kc_lower = kc_mid - 1.5 * kc_atr

            # Recover BB upper/lower from bb_width and bb_pct_b:
            # bb_width = (BB_upper - BB_lower) / BB_mid
            # bb_pct_b = (close - BB_lower) / (BB_upper - BB_lower)
            if "bb_pct_b" not in d.columns:
                return
            bb_mid_approx   = d["close"].rolling(20).mean()
            half_width      = (d["bb_width"] * bb_mid_approx) / 2
            bb_upper_approx = bb_mid_approx + half_width
            bb_lower_approx = bb_mid_approx - half_width

            d["bb_kc_squeeze"] = (
                (bb_upper_approx < kc_upper) & (bb_lower_approx > kc_lower)
            ).astype(int)
        except Exception:
            d["bb_kc_squeeze"] = 0

    def _pivot_sr(self, d: pd.DataFrame, lookback: int = 5, window: int = 50) -> None:
        """Pivot point-based Support and Resistance levels.

        A pivot high is a bar where high > all highs in a ±lookback window.
        A pivot low  is a bar where low  < all lows  in a ±lookback window.
        pivot_resistance: nearest unbroken pivot high above current price
        pivot_support   : nearest unbroken pivot low  below current price
        near_resistance : within 0.5% of resistance
        near_support    : within 0.5% of support
        """
        pivot_high = pd.Series(np.nan, index=d.index)
        pivot_low  = pd.Series(np.nan, index=d.index)
        highs = d["high"].values
        lows  = d["low"].values
        n = len(d)

        for i in range(lookback, n - lookback):
            window_highs = highs[i - lookback: i + lookback + 1]
            window_lows  = lows[i - lookback: i + lookback + 1]
            if highs[i] == window_highs.max():
                pivot_high.iloc[i] = highs[i]
            if lows[i] == window_lows.min():
                pivot_low.iloc[i] = lows[i]

        # For each bar: look back `window` bars to find the most recent pivot high/low
        resistance = pd.Series(np.nan, index=d.index)
        support    = pd.Series(np.nan, index=d.index)
        close      = d["close"].values

        for i in range(window, n):
            recent_highs = pivot_high.iloc[i - window: i].dropna()
            recent_lows  = pivot_low.iloc[i - window: i].dropna()
            price = close[i]
            above = recent_highs[recent_highs > price]
            below = recent_lows[recent_lows < price]
            if not above.empty:
                resistance.iloc[i] = above.min()   # nearest resistance above
            if not below.empty:
                support.iloc[i] = below.max()      # nearest support below

        d["pivot_resistance"] = resistance
        d["pivot_support"]    = support

        prox = 0.005  # 0.5%
        d["near_resistance"] = (
            (d["pivot_resistance"].notna()) &
            ((d["close"] - d["pivot_resistance"]).abs() / d["pivot_resistance"] < prox)
        ).astype(int)
        d["near_support"] = (
            (d["pivot_support"].notna()) &
            ((d["close"] - d["pivot_support"]).abs() / d["pivot_support"] < prox)
        ).astype(int)

    def _candlestick_patterns(self, d: pd.DataFrame) -> None:
        """Candlestick pattern flags used by the Price Action strategy.

        Patterns based on bar body/wick geometry ratios:
          Hammer       : lower_wick > 2.5×body, upper_wick < 0.5×body (bullish reversal)
          Shooting Star: upper_wick > 2.5×body, lower_wick < 0.5×body (bearish reversal)
          Doji         : body < 5% of bar range (indecision)
          Bullish Engulf: current bullish bar fully engulfs previous bearish bar
          Bearish Engulf: current bearish bar fully engulfs previous bullish bar
          Pin Bar Bull : long lower wick (> 3×body), close in upper 30% of range
          Pin Bar Bear : long upper wick (> 3×body), close in lower 30% of range
        """
        o = d["open"]
        h = d["high"]
        l = d["low"]
        c = d["close"]

        body       = (c - o).abs()
        bar_range  = (h - l).replace(0, np.nan)
        upper_wick = h - d[["close", "open"]].max(axis=1)
        lower_wick = d[["close", "open"]].min(axis=1) - l
        body_to_range = body / bar_range

        # Hammer: long lower wick, small body, at/near lower end of range
        d["cdl_hammer"] = (
            (lower_wick > 2.5 * body) &
            (upper_wick < 0.5 * body.clip(lower=0.001)) &
            (body_to_range < 0.4)
        ).astype(int)

        # Shooting Star: long upper wick, small body, at/near upper end of range
        d["cdl_shooting_star"] = (
            (upper_wick > 2.5 * body) &
            (lower_wick < 0.5 * body.clip(lower=0.001)) &
            (body_to_range < 0.4)
        ).astype(int)

        # Doji: body < 5% of range
        d["cdl_doji"] = (body_to_range < 0.05).astype(int)

        # Bullish Engulfing: current bullish bar engulfs previous bearish bar
        prev_o = o.shift(1)
        prev_c = c.shift(1)
        d["cdl_bullish_engulf"] = (
            (c > o) &           # current bullish
            (prev_c < prev_o) & # previous bearish
            (o < prev_c) &      # current open below prev close
            (c > prev_o)        # current close above prev open
        ).astype(int)

        # Bearish Engulfing: current bearish bar engulfs previous bullish bar
        d["cdl_bearish_engulf"] = (
            (c < o) &           # current bearish
            (prev_c > prev_o) & # previous bullish
            (o > prev_c) &      # current open above prev close
            (c < prev_o)        # current close below prev open
        ).astype(int)

        # Pin Bar Bullish: long lower wick (> 3×body), close in upper 30% of bar
        close_pos = (c - l) / bar_range
        d["cdl_pin_bar_bull"] = (
            (lower_wick > 3.0 * body.clip(lower=0.001)) &
            (close_pos > 0.7)
        ).astype(int)

        # Pin Bar Bearish: long upper wick (> 3×body), close in lower 30% of bar
        d["cdl_pin_bar_bear"] = (
            (upper_wick > 3.0 * body.clip(lower=0.001)) &
            (close_pos < 0.3)
        ).astype(int)

    def _relative_strength_features(self, d: pd.DataFrame, period: int = 20) -> None:
        """RS momentum placeholder — requires external Nifty series for true RS.

        In standalone feature computation (without Nifty), we approximate RS momentum
        as the symbol's own 20-bar excess return vs its own average (self-relative).
        True RS vs Nifty is computed in the RelativeStrengthStrategy when nifty_close
        is provided externally.

        rs_momentum: 20-bar rate-of-change of a relative-strength proxy
        """
        # Proxy: excess return of price over its 20-bar mean return trend
        log_ret = np.log(d["close"]).diff(1)
        avg_ret = log_ret.rolling(period).mean()
        d["rs_momentum"] = log_ret.rolling(period).sum() - avg_ret * period

    # ── End Phase 2.1 features ─────────────────────────────────────────────────

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
