"""Range Trading Strategy — Support/Resistance band trading in sideways markets.

Entry logic:
  Long  : close within 0.5% of pivot support AND ADX < 22 AND RSI < 55 (not overbought)
  Short : close within 0.5% of pivot resistance AND ADX < 22 AND RSI > 45 (not oversold)

Statistical basis:
  - Regime gate: ADX < 22 — range trading requires a sideways market.
    In trending regimes, support/resistance gets consistently broken.
  - Pivot S/R from features.py (5-bar lookback pivot points):
    Pivot High = bar where high is the highest in a 5-bar window centered on it.
    The most recent unbroken pivot highs form resistance; pivot lows form support.
  - Proximity filter: within 0.5% of S/R — price must be "testing" the level.
    Price needs to be close enough to the level to confirm it as a touch.
  - Range width filter: (resistance - support) / support >= 0.005 (min 0.5% range).
    Narrower ranges have too little reward potential relative to transaction costs.
  - Stop  : 0.7% beyond the S/R level (tight — if the level breaks, the trade is invalid)
  - Target: near the opposite band of the range (85% of the range width from entry)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from production.strategy.base import BaseStrategy, Signal


class RangeTradingConfig:
    def __init__(self, cfg: dict | None = None) -> None:
        c = (cfg or {}).get("range_trading", {})
        s = (cfg or {}).get("session", {})
        self.proximity_pct     = float(c.get("proximity_pct",    0.005))  # within 0.5% of S/R
        self.adx_max           = float(c.get("adx_max",          22.0))
        self.min_range_pct     = float(c.get("min_range_pct",    0.005))  # min 0.5% range width
        self.stop_pct          = float(c.get("stop_pct",         0.007))  # 0.7% beyond level
        self.target_range_frac = float(c.get("target_range_frac",0.85))   # 85% of range
        self.rsi_long_max      = float(c.get("rsi_long_max",     55.0))
        self.rsi_short_min     = float(c.get("rsi_short_min",    45.0))
        self.min_confidence    = float(c.get("min_confidence",   0.52))
        self.vix_max           = float(c.get("vix_max",          20.0))   # low VIX = stable range
        self.session_start     = s.get("start", "09:45")
        self.session_end       = s.get("end",   "15:10")


def _load_cfg() -> dict:
    try:
        return yaml.safe_load(Path("configs/strategy.yaml").read_text()) or {}
    except Exception:
        return {}


class RangeTradingStrategy(BaseStrategy):
    """Pivot-based support/resistance range trading for sideways NSE markets."""

    name = "range_trading"

    def __init__(self, cfg: RangeTradingConfig | None = None) -> None:
        self.cfg = cfg or RangeTradingConfig(_load_cfg())

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

        # ── Regime gate ────────────────────────────────────────────────────────
        ranging = f["adx"] < self.cfg.adx_max

        # ── S/R levels from features ───────────────────────────────────────────
        # pivot_resistance / pivot_support come from FeatureEngineer
        # Fall back to Donchian high/low if pivot features not present
        if "pivot_resistance" in f.columns and "pivot_support" in f.columns:
            resistance = f["pivot_resistance"]
            support    = f["pivot_support"]
        else:
            resistance = c.rolling(20).max()
            support    = c.rolling(20).min()

        # ── Range width filter ─────────────────────────────────────────────────
        range_width = (resistance - support) / support
        wide_enough = range_width >= self.cfg.min_range_pct

        # ── Proximity to S/R ──────────────────────────────────────────────────
        near_support    = (c - support).abs() / support < self.cfg.proximity_pct
        near_resistance = (c - resistance).abs() / resistance < self.cfg.proximity_pct

        # ── RSI quality ───────────────────────────────────────────────────────
        rsi = f.get("rsi", pd.Series(50.0, index=f.index))
        rsi_ok_long  = rsi < self.cfg.rsi_long_max
        rsi_ok_short = rsi > self.cfg.rsi_short_min

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

        long_entry  = near_support    & wide_enough & ranging & rsi_ok_long  & vix_ok & session_ok
        short_entry = near_resistance & wide_enough & ranging & rsi_ok_short & vix_ok & session_ok

        direction = pd.Series(0, index=f.index, dtype=int)
        direction[long_entry]  =  1
        direction[short_entry] = -1

        # ── Confidence ────────────────────────────────────────────────────────
        confidence = pd.Series(0.50, index=f.index)
        confidence += np.where(range_width >= 0.01, 0.07, 0.0)  # wider range = better reward
        confidence += np.where(f["adx"] < 15,       0.05, 0.0)  # very flat = strong range
        confidence += np.where(rsi < 35,             0.05, 0.0)  # oversold at support
        confidence += np.where(rsi > 65,             0.05, 0.0)  # overbought at resistance
        confidence = confidence.clip(0.0, 1.0).where(direction != 0, 0.0)

        # ── Target: 85% of range width toward opposite band ───────────────────
        range_w = resistance - support
        target_long  = support    + range_w * self.cfg.target_range_frac
        target_short = resistance - range_w * self.cfg.target_range_frac

        stop   = np.where(direction ==  1, c * (1 - self.cfg.stop_pct),
                 np.where(direction == -1, c * (1 + self.cfg.stop_pct), np.nan))
        target = np.where(direction ==  1, target_long,
                 np.where(direction == -1, target_short, np.nan))

        return pd.DataFrame({
            "direction":  direction,
            "confidence": confidence,
            "entry":      c,
            "stop":       stop,
            "target":     target,
            "atr":        atr,
            "adx":        f["adx"],
            "strategy":   "range_trading",
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
        adx = f.get("adx", 0.0)
        rsi = f.get("rsi", 50.0)

        if adx >= self.cfg.adx_max:
            return None

        resistance = f.get("pivot_resistance", None)
        support    = f.get("pivot_support",    None)
        if resistance is None or support is None:
            return None

        if pd.isna(resistance) or pd.isna(support) or support <= 0:
            return None

        range_width = (resistance - support) / support
        if range_width < self.cfg.min_range_pct:
            return None

        near_support    = abs(close - support)    / support < self.cfg.proximity_pct
        near_resistance = abs(close - resistance) / resistance < self.cfg.proximity_pct
        vix_ok = (vix is None) or (vix < self.cfg.vix_max)

        import datetime as _dt
        _now = pd.Timestamp.now(tz="Asia/Kolkata").time()
        _ss  = _dt.time(*[int(x) for x in self.cfg.session_start.split(":")])
        _se  = _dt.time(*[int(x) for x in self.cfg.session_end.split(":")])
        if not (_ss <= _now < _se) or not vix_ok:
            return None

        direction = 0
        reason    = ""
        if near_support and rsi < self.cfg.rsi_long_max:
            direction = 1
            reason = f"Price at support={support:.2f} | RSI={rsi:.1f} | range={range_width*100:.1f}%"
        elif near_resistance and rsi > self.cfg.rsi_short_min:
            direction = -1
            reason = f"Price at resistance={resistance:.2f} | RSI={rsi:.1f} | range={range_width*100:.1f}%"
        if direction == 0:
            return None

        confidence = 0.50
        if range_width >= 0.01: confidence += 0.07
        if adx < 15:            confidence += 0.05
        if rsi < 35 or rsi > 65: confidence += 0.05
        confidence = min(1.0, confidence)

        if confidence < self.cfg.min_confidence:
            return None

        range_w = resistance - support
        stop    = close * (1 - direction * self.cfg.stop_pct)
        target  = (support + range_w * self.cfg.target_range_frac) if direction == 1 \
                  else (resistance - range_w * self.cfg.target_range_frac)

        return Signal(
            time=pd.Timestamp.now(tz="Asia/Kolkata"),
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=close,
            stop_price=stop,
            target_price=target,
            atr=atr,
            strategy="range_trading",
            reason=reason,
            adx=adx,
            vix=vix,
        )
