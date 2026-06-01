"""Volatility Contraction Trading Strategy — BB Squeeze + Keltner Channel breakout.

Entry logic:
  1. Squeeze detection: Bollinger Bands contract inside Keltner Channel
     (bb_kc_squeeze == 1 for >= 3 consecutive bars)
  2. Breakout entry: first bar where price breaks outside the BB after the squeeze
  3. Direction determined by momentum direction at breakout (MACD histogram / di_diff)
  4. Volume confirm: vol_ratio >= 1.5

Statistical basis:
  - Keltner Channel: EMA(20) ± 1.5 × ATR(10)  [standard Keltner parameters]
  - BB inside KC = true volatility compression ("squeeze on")
  - Post-squeeze moves are statistically 2-3× the average bar range
    (Bollinger band squeeze has ~65% follow-through rate after 3+ squeeze bars)
  - ATR ratio < 0.7 of 20-bar ATR mean: secondary squeeze confirmation
  - Min squeeze bars = 3: single-bar squeeze is noise
  - Stop  : 2.0 × ATR (wider — post-squeeze volatility is elevated)
  - Target: 3.0 × ATR (1.5:1 R:R, justified by high follow-through rate)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from production.strategy.base import BaseStrategy, Signal


class VolContractionConfig:
    def __init__(self, cfg: dict | None = None) -> None:
        c = (cfg or {}).get("volatility_contraction", {})
        s = (cfg or {}).get("session", {})
        self.min_squeeze_bars  = int(c.get("min_squeeze_bars",    3))
        self.volume_ratio      = float(c.get("volume_ratio",      1.5))
        self.atr_ratio_max     = float(c.get("atr_ratio_max",     0.75)) # ATR < 75% of ATR-MA
        self.stop_atr_mult     = float(c.get("stop_atr_mult",     2.0))
        self.target_atr_mult   = float(c.get("target_atr_mult",   3.0))
        self.min_confidence    = float(c.get("min_confidence",    0.55))
        self.vix_max           = float(c.get("vix_max",           30.0))
        self.session_start     = s.get("start", "09:45")
        self.session_end       = s.get("end",   "15:10")


def _load_cfg() -> dict:
    try:
        return yaml.safe_load(Path("configs/strategy.yaml").read_text()) or {}
    except Exception:
        return {}


class VolatilityContractionStrategy(BaseStrategy):
    """VCP / BB-squeeze breakout strategy for NSE intraday."""

    name = "volatility_contraction"

    def __init__(self, cfg: VolContractionConfig | None = None) -> None:
        self.cfg = cfg or VolContractionConfig(_load_cfg())

    # ── Vectorised ──────────────────────────────────────────────────────────────

    def generate_signals_df(
        self,
        features: pd.DataFrame,
        close: pd.Series,
        vix: pd.Series | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        f   = features.copy()
        c   = close.reindex(f.index)
        atr = f["atr_pct"] * c

        # ── Squeeze detection ──────────────────────────────────────────────────
        # bb_kc_squeeze: 1 when BB is inside KC (comes from features.py)
        # atr_ratio: current ATR / ATR-MA(20) — below 0.75 = compression
        squeeze     = f.get("bb_kc_squeeze", pd.Series(0, index=f.index)) == 1
        atr_squeeze = f.get("atr_ratio",     pd.Series(1.0, index=f.index)) < self.cfg.atr_ratio_max
        any_squeeze = squeeze | atr_squeeze

        # Count consecutive squeeze bars using a rolling sum
        squeeze_count = any_squeeze.astype(int).rolling(self.cfg.min_squeeze_bars).sum()
        was_in_squeeze = squeeze_count >= self.cfg.min_squeeze_bars

        # ── Breakout: first bar leaving the squeeze (bb_pct_b outside 0-1) ────
        bb_pct_b    = f.get("bb_pct_b", pd.Series(0.5, index=f.index))
        left_upper  = (bb_pct_b > 1.0) & was_in_squeeze.shift(1).fillna(False)
        left_lower  = (bb_pct_b < 0.0) & was_in_squeeze.shift(1).fillna(False)

        # ── Direction: MACD histogram + DI diff ───────────────────────────────
        macd_hist = f.get("macd_hist_norm", pd.Series(0.0, index=f.index))
        di_diff   = f.get("di_diff",        pd.Series(0.0, index=f.index))
        bullish   = (macd_hist > 0) | (di_diff > 0)
        bearish   = (macd_hist < 0) | (di_diff < 0)

        vol_ok = f["vol_ratio"] >= self.cfg.volume_ratio

        if vix is not None:
            vix_ok = vix.reindex(f.index, method="ffill") < self.cfg.vix_max
        else:
            vix_ok = pd.Series(True, index=f.index)

        import datetime as _dt
        _ss = _dt.time(*[int(x) for x in self.cfg.session_start.split(":")])
        _se = _dt.time(*[int(x) for x in self.cfg.session_end.split(":")])
        if hasattr(f.index, "time"):
            session_ok = pd.Series([_ss <= t < _se for t in f.index.time], index=f.index)
        else:
            session_ok = pd.Series(True, index=f.index)

        long_entry  = left_upper & bullish & vol_ok & vix_ok & session_ok
        short_entry = left_lower & bearish & vol_ok & vix_ok & session_ok

        direction = pd.Series(0, index=f.index, dtype=int)
        direction[long_entry]  =  1
        direction[short_entry] = -1

        # ── Confidence ────────────────────────────────────────────────────────
        confidence = pd.Series(0.55, index=f.index)  # base higher — squeeze has strong follow-through
        confidence += np.where(squeeze_count >= 5,       0.07, 0.0)  # longer squeeze = bigger move
        confidence += np.where(f["vol_ratio"] >= 2.0,    0.08, 0.0)  # volume confirms breakout
        confidence += np.where(f.get("atr_ratio", 1.0) < 0.6, 0.05, 0.0)  # tight ATR compression
        confidence = confidence.clip(0.0, 1.0).where(direction != 0, 0.0)

        stop_dist   = atr * self.cfg.stop_atr_mult
        target_dist = atr * self.cfg.target_atr_mult
        stop   = np.where(direction ==  1, c - stop_dist,
                 np.where(direction == -1, c + stop_dist, np.nan))
        target = np.where(direction ==  1, c + target_dist,
                 np.where(direction == -1, c - target_dist, np.nan))

        return pd.DataFrame({
            "direction":  direction,
            "confidence": confidence,
            "entry":      c,
            "stop":       stop,
            "target":     target,
            "atr":        atr,
            "adx":        f["adx"],
            "strategy":   "volatility_contraction",
            "regime":     f.get("regime_heuristic", pd.Series(0, index=f.index)),
        }, index=f.index)

    # ── Bar-by-bar live ──────────────────────────────────────────────────────

    def evaluate_bar(
        self,
        symbol: str,
        latest_features: pd.Series,
        close: float,
        vix: float | None = None,
        squeeze_bar_count: int = 0,
        **kwargs,
    ) -> Signal | None:
        f    = latest_features
        atr  = f.get("atr_pct", 0.0) * close

        squeeze = (f.get("bb_kc_squeeze", 0) == 1) or (f.get("atr_ratio", 1.0) < self.cfg.atr_ratio_max)
        if squeeze_bar_count < self.cfg.min_squeeze_bars:
            return None  # squeeze hasn't built up enough

        bb_pct_b  = f.get("bb_pct_b", 0.5)
        vol_ok    = f.get("vol_ratio", 0.0) >= self.cfg.volume_ratio
        macd_hist = f.get("macd_hist_norm", 0.0)
        di_diff   = f.get("di_diff", 0.0)
        vix_ok    = (vix is None) or (vix < self.cfg.vix_max)

        import datetime as _dt
        _now = pd.Timestamp.now(tz="Asia/Kolkata").time()
        _ss  = _dt.time(*[int(x) for x in self.cfg.session_start.split(":")])
        _se  = _dt.time(*[int(x) for x in self.cfg.session_end.split(":")])
        if not (_ss <= _now < _se) or not vol_ok or not vix_ok:
            return None

        direction = 0
        reason    = ""
        if bb_pct_b > 1.0 and (macd_hist > 0 or di_diff > 0):
            direction = 1
            reason = f"BB squeeze breakout UP | squeeze_bars={squeeze_bar_count} | vol_ratio={f.get('vol_ratio',0):.2f}"
        elif bb_pct_b < 0.0 and (macd_hist < 0 or di_diff < 0):
            direction = -1
            reason = f"BB squeeze breakdown DOWN | squeeze_bars={squeeze_bar_count} | vol_ratio={f.get('vol_ratio',0):.2f}"
        if direction == 0:
            return None

        confidence = 0.55
        if squeeze_bar_count >= 5:              confidence += 0.07
        if f.get("vol_ratio", 0) >= 2.0:       confidence += 0.08
        if f.get("atr_ratio", 1.0) < 0.6:      confidence += 0.05
        confidence = min(1.0, confidence)

        if confidence < self.cfg.min_confidence:
            return None

        stop   = close - direction * atr * self.cfg.stop_atr_mult
        target = close + direction * atr * self.cfg.target_atr_mult

        return Signal(
            time=pd.Timestamp.now(tz="Asia/Kolkata"),
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=close,
            stop_price=stop,
            target_price=target,
            atr=atr,
            strategy="volatility_contraction",
            reason=reason,
            vix=vix,
        )
