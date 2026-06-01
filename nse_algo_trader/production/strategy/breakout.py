"""Breakout Trading Strategy — Donchian Channel breakout with volume and ADX confirmation.

Entry logic:
  Long  : close breaks above 20-bar Donchian high  AND vol_ratio >= 1.5 AND ADX > 20
  Short : close breaks below 20-bar Donchian low   AND vol_ratio >= 1.5 AND ADX > 20

Statistical basis:
  - 20-period Donchian Channel (Turtle Trading standard)
  - Volume ≥ 1.5× MA(20): institution-level participation filter
  - ADX > 20: trend forming — breakouts during trending markets succeed ~60% vs ~40% in ranging
  - False-breakout filter: require close above/below channel (not just intrabar wick)
  - Stop  : 1.5 × ATR below breakout level (tight — invalidation of breakout)
  - Target: 3.0 × ATR (2:1 R:R — minimum for positive expectancy at 40% win rate)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from pathlib import Path

from production.strategy.base import BaseStrategy, Signal


class BreakoutConfig:
    channel_period: int   = 20      # Donchian channel lookback
    volume_ratio: float   = 1.5     # vol_ratio threshold for volume confirmation
    adx_min: float        = 20.0    # minimum ADX — trend must be forming
    adx_max: float        = 50.0    # cap — avoid entering late in extended trends
    stop_atr_mult: float  = 1.5     # stop = entry - 1.5×ATR
    target_atr_mult: float= 3.0     # target = entry + 3.0×ATR (2:1 R:R)
    min_confidence: float = 0.55
    vix_max: float        = 28.0    # slightly higher VIX tolerance than momentum (breakouts occur with fear)
    session_start: str    = "09:45"
    session_end: str      = "15:10"

    def __init__(self, cfg: dict | None = None) -> None:
        c = (cfg or {}).get("breakout", {})
        self.channel_period   = int(c.get("channel_period",    20))
        self.volume_ratio     = float(c.get("volume_ratio",   1.5))
        self.adx_min          = float(c.get("adx_min",        20.0))
        self.adx_max          = float(c.get("adx_max",        50.0))
        self.stop_atr_mult    = float(c.get("stop_atr_mult",  1.5))
        self.target_atr_mult  = float(c.get("target_atr_mult",3.0))
        self.min_confidence   = float(c.get("min_confidence", 0.55))
        self.vix_max          = float(c.get("vix_max",        28.0))
        s = (cfg or {}).get("session", {})
        self.session_start    = s.get("start", "09:45")
        self.session_end      = s.get("end",   "15:10")


def _load_cfg() -> dict:
    try:
        return yaml.safe_load(Path("configs/strategy.yaml").read_text()) or {}
    except Exception:
        return {}


class BreakoutStrategy(BaseStrategy):
    """Donchian Channel breakout strategy for trending NSE stocks and indices."""

    name = "breakout"

    def __init__(self, cfg: BreakoutConfig | None = None) -> None:
        self.cfg = cfg or BreakoutConfig(_load_cfg())

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

        # ── Breakout conditions ────────────────────────────────────────────────
        # breakout_up/breakout_dn come from FeatureEngineer (features.py)
        # Fall back to on-the-fly Donchian if feature not available
        if "breakout_up" in f.columns and "breakout_dn" in f.columns:
            brk_long  = f["breakout_up"] == 1
            brk_short = f["breakout_dn"] == 1
        else:
            don_high = c.rolling(self.cfg.channel_period).max().shift(1)
            don_low  = c.rolling(self.cfg.channel_period).min().shift(1)
            brk_long  = c > don_high
            brk_short = c < don_low

        vol_ok   = f["vol_ratio"] >= self.cfg.volume_ratio
        adx_ok   = (f["adx"] >= self.cfg.adx_min) & (f["adx"] <= self.cfg.adx_max)
        di_long  = f["di_diff"] > 0
        di_short = f["di_diff"] < 0

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

        long_entry  = brk_long  & vol_ok & adx_ok & di_long  & vix_ok & session_ok
        short_entry = brk_short & vol_ok & adx_ok & di_short & vix_ok & session_ok

        direction = pd.Series(0, index=f.index, dtype=int)
        direction[long_entry]  =  1
        direction[short_entry] = -1

        # ── Confidence: base 0.50, boosted by signal quality ──────────────────
        confidence = pd.Series(0.50, index=f.index)
        confidence += np.where(f["vol_ratio"] >= 1.5, 0.05, 0.0)
        confidence += np.where(f["vol_ratio"] >= 2.0, 0.05, 0.0)
        confidence += np.where(f["adx"] >= 25,        0.08, 0.0)
        confidence += np.where(f["adx"] >= 35,        0.05, 0.0)
        # breakout_pos in features = donchian position; extreme breakout = more confident
        if "donchian_pos" in f.columns:
            confidence += np.where(f["donchian_pos"] > 0.95, 0.05, 0.0)
            confidence += np.where(f["donchian_pos"] < 0.05, 0.05, 0.0)
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
            "strategy":   "breakout",
            "regime":     f.get("regime_heuristic", pd.Series(0, index=f.index)),
        }, index=f.index)

    # ── Bar-by-bar live ──────────────────────────────────────────────────────

    def evaluate_bar(
        self,
        symbol: str,
        latest_features: pd.Series,
        close: float,
        vix: float | None = None,
        **kwargs,
    ) -> Signal | None:
        f   = latest_features
        atr = f.get("atr_pct", 0.0) * close

        brk_long  = f.get("breakout_up",  0) == 1
        brk_short = f.get("breakout_dn",  0) == 1
        vol_ok    = f.get("vol_ratio", 0.0) >= self.cfg.volume_ratio
        adx       = f.get("adx", 0.0)
        adx_ok    = self.cfg.adx_min <= adx <= self.cfg.adx_max
        di_diff   = f.get("di_diff", 0.0)
        vix_ok    = (vix is None) or (vix < self.cfg.vix_max)

        import datetime as _dt
        _now = pd.Timestamp.now(tz="Asia/Kolkata").time()
        _ss  = _dt.time(*[int(x) for x in self.cfg.session_start.split(":")])
        _se  = _dt.time(*[int(x) for x in self.cfg.session_end.split(":")])
        if not (_ss <= _now < _se):
            return None

        if not (vol_ok and adx_ok and vix_ok):
            return None

        direction = 0
        reason    = ""
        if brk_long and di_diff > 0:
            direction = 1
            reason = f"Donchian breakout UP | ADX={adx:.1f} | vol_ratio={f.get('vol_ratio',0):.2f}"
        elif brk_short and di_diff < 0:
            direction = -1
            reason = f"Donchian breakdown DOWN | ADX={adx:.1f} | vol_ratio={f.get('vol_ratio',0):.2f}"
        if direction == 0:
            return None

        confidence = 0.50
        if f.get("vol_ratio", 0) >= 1.5: confidence += 0.05
        if f.get("vol_ratio", 0) >= 2.0: confidence += 0.05
        if adx >= 25: confidence += 0.08
        if adx >= 35: confidence += 0.05
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
            strategy="breakout",
            reason=reason,
            adx=adx,
            vix=vix,
        )
