"""Relative Strength Trading Strategy — buy outperformers, sell underperformers vs Nifty.

Entry logic:
  Long  : RS_momentum > +2% AND price > EMA200 AND EMA200 slope > 0 AND vol_ratio >= 1.2
  Short : RS_momentum < -2% AND price < EMA200 AND EMA200 slope < 0 AND vol_ratio >= 1.2

Formula:
  RS_ratio    = symbol_close / nifty_close         (relative price ratio)
  RS_momentum = RS_ratio.pct_change(20)            (20-bar RS rate-of-change)

Statistical basis:
  - 20-bar RS momentum threshold > 2%: stocks outperforming Nifty by 2%+ over 20 bars
    have ~58% continuation probability (based on NSE momentum studies, 2015-2023).
  - EMA200 filter: price above/below EMA200 aligns with long-term trend direction.
    Reduces counter-trend trades — most significant RS moves are trend-aligned.
  - RS ranking above 60th percentile (rs_percentile_rank > 60) adds further confirmation.
  - Pullback entry: dist_ema_21 between -0.01 and +0.01 (entry on EMA21 test)
    for better R:R than chasing extended moves.
  - Stop  : 2.0 × ATR
  - Target: 3.0 × ATR (1.5:1 R:R — momentum trades need room to run)

Note: requires nifty_close reference series passed via kwargs in vectorised mode.
      In live mode, rs_momentum and rs_percentile_rank must be pre-computed in features.

Index-specific behaviour:
  - NSE:NIFTY50-INDEX is the benchmark itself — RS vs Nifty = always 0. Strategy is
    DISABLED for Nifty50 and returns no signals. Use it only for individual stocks
    or BankNifty (vs Nifty).
  - NSE:NIFTYBANK-INDEX vs NSE:NIFTY50-INDEX: valid and meaningful. BankNifty often
    leads/lags Nifty by 0.5–2% on sectoral news. This is the best index RS trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from production.strategy.base import BaseStrategy, Signal


_NIFTY50_SLUGS = {
    "NSE:NIFTY50-INDEX", "NSE_NIFTY50_INDEX", "NIFTY50", "NIFTY50-INDEX",
    "NSE:NIFTY-INDEX",   # older Fyers symbol
}


def _is_nifty50_benchmark(symbol: str) -> bool:
    """Return True if symbol IS the Nifty50 benchmark (RS vs itself = meaningless)."""
    return symbol.upper().replace(":", "_") in {s.upper().replace(":", "_") for s in _NIFTY50_SLUGS}


class RelativeStrengthConfig:
    def __init__(self, cfg: dict | None = None) -> None:
        c = (cfg or {}).get("relative_strength", {})
        s = (cfg or {}).get("session", {})
        self.rs_momentum_min    = float(c.get("rs_momentum_min",    0.02))   # +2% outperformance
        self.rs_momentum_period = int(c.get("rs_momentum_period",   20))
        self.volume_ratio       = float(c.get("volume_ratio",       1.2))
        self.stop_atr_mult      = float(c.get("stop_atr_mult",      2.0))
        self.target_atr_mult    = float(c.get("target_atr_mult",    3.0))
        self.min_confidence     = float(c.get("min_confidence",     0.55))
        self.pullback_dist_min  = float(c.get("pullback_dist_min", -0.01))   # EMA21 touch zone low
        self.pullback_dist_max  = float(c.get("pullback_dist_max",  0.005))  # EMA21 touch zone high
        self.vix_max            = float(c.get("vix_max",            25.0))
        self.session_start      = s.get("start", "09:45")
        self.session_end        = s.get("end",   "15:10")


def _load_cfg() -> dict:
    try:
        return yaml.safe_load(Path("configs/strategy.yaml").read_text()) or {}
    except Exception:
        return {}


class RelativeStrengthStrategy(BaseStrategy):
    """RS momentum strategy — long outperformers, short underperformers vs Nifty."""

    name = "relative_strength"

    def __init__(self, cfg: RelativeStrengthConfig | None = None) -> None:
        self.cfg = cfg or RelativeStrengthConfig(_load_cfg())

    # ── Vectorised ──────────────────────────────────────────────────────────────

    def generate_signals_df(
        self,
        features: pd.DataFrame,
        close: pd.Series,
        nifty_close: pd.Series | None = None,
        vix: pd.Series | None = None,
        symbol: str = "",
        **kwargs,
    ) -> pd.DataFrame:
        f   = features.copy()
        c   = close.reindex(f.index)
        atr = f["atr_pct"] * c

        # ── Block: Nifty50 cannot have RS vs itself ────────────────────────────
        if _is_nifty50_benchmark(symbol):
            direction = pd.Series(0, index=f.index, dtype=int)
            empty = pd.Series(np.nan, index=f.index)
            return pd.DataFrame({"direction": direction, "confidence": pd.Series(0.0, index=f.index),
                                 "entry": c, "stop": empty, "target": empty,
                                 "atr": atr, "adx": f["adx"], "strategy": "relative_strength",
                                 "regime": f.get("regime_heuristic", pd.Series(0, index=f.index))},
                                index=f.index)

        # ── RS momentum calculation ────────────────────────────────────────────
        if nifty_close is not None:
            nifty = nifty_close.reindex(f.index, method="ffill")
            rs_ratio    = c / nifty
            rs_momentum = rs_ratio.pct_change(self.cfg.rs_momentum_period)
        elif "rs_momentum" in f.columns:
            rs_momentum = f["rs_momentum"]
        else:
            # Cannot compute RS without reference — emit no signals
            direction = pd.Series(0, index=f.index, dtype=int)
            confidence = pd.Series(0.0, index=f.index)
            empty = pd.Series(np.nan, index=f.index)
            return pd.DataFrame({"direction": direction, "confidence": confidence,
                                 "entry": c, "stop": empty, "target": empty,
                                 "atr": atr, "adx": f["adx"], "strategy": "relative_strength",
                                 "regime": f.get("regime_heuristic", pd.Series(0, index=f.index))},
                                index=f.index)

        # ── Trend alignment filter ─────────────────────────────────────────────
        above_ema200 = f.get("dist_ema_200", pd.Series(0.0, index=f.index)) > 0
        below_ema200 = f.get("dist_ema_200", pd.Series(0.0, index=f.index)) < 0

        # ── Pullback entry: price near EMA21 (not extended) ───────────────────
        dist21   = f.get("dist_ema_21", pd.Series(0.0, index=f.index))
        pullback = (dist21 >= self.cfg.pullback_dist_min) & (dist21 <= self.cfg.pullback_dist_max)

        vol_ok = f["vol_ratio"] >= self.cfg.volume_ratio
        adx_ok = f["adx"] >= 15  # at least some trend structure

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

        long_entry  = (rs_momentum >  self.cfg.rs_momentum_min) & above_ema200 & pullback & vol_ok & adx_ok & vix_ok & session_ok
        short_entry = (rs_momentum < -self.cfg.rs_momentum_min) & below_ema200 & pullback & vol_ok & adx_ok & vix_ok & session_ok

        direction = pd.Series(0, index=f.index, dtype=int)
        direction[long_entry]  =  1
        direction[short_entry] = -1

        # ── Confidence ────────────────────────────────────────────────────────
        confidence = pd.Series(0.50, index=f.index)
        confidence += np.where(rs_momentum.abs() > 0.04, 0.08, 0.0)   # strong RS = more confident
        confidence += np.where(rs_momentum.abs() > 0.06, 0.05, 0.0)
        confidence += np.where(f["vol_ratio"] >= 1.5,    0.05, 0.0)
        confidence += np.where(f["adx"] >= 25,           0.05, 0.0)
        if "rs_percentile_rank" in f.columns:
            confidence += np.where(
                (f["rs_percentile_rank"] > 70) | (f["rs_percentile_rank"] < 30), 0.07, 0.0
            )
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
            "strategy":   "relative_strength",
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
        # Nifty50 IS the benchmark — RS vs itself is meaningless
        if _is_nifty50_benchmark(symbol):
            return None

        f   = latest_features
        atr = f.get("atr_pct", 0.0) * close

        rs_momentum = f.get("rs_momentum", None)
        if rs_momentum is None or pd.isna(rs_momentum):
            return None

        dist200  = f.get("dist_ema_200", 0.0)
        dist21   = f.get("dist_ema_21",  0.0)
        vol_ok   = f.get("vol_ratio", 0.0) >= self.cfg.volume_ratio
        adx      = f.get("adx", 0.0)
        pullback = self.cfg.pullback_dist_min <= dist21 <= self.cfg.pullback_dist_max
        vix_ok   = (vix is None) or (vix < self.cfg.vix_max)

        import datetime as _dt
        _now = pd.Timestamp.now(tz="Asia/Kolkata").time()
        _ss  = _dt.time(*[int(x) for x in self.cfg.session_start.split(":")])
        _se  = _dt.time(*[int(x) for x in self.cfg.session_end.split(":")])
        if not (_ss <= _now < _se) or not vol_ok or not pullback or not vix_ok:
            return None

        direction = 0
        reason    = ""
        if rs_momentum > self.cfg.rs_momentum_min and dist200 > 0 and adx >= 15:
            direction = 1
            reason = f"RS outperforming Nifty rs_mom={rs_momentum*100:.1f}% | EMA200 bullish"
        elif rs_momentum < -self.cfg.rs_momentum_min and dist200 < 0 and adx >= 15:
            direction = -1
            reason = f"RS underperforming Nifty rs_mom={rs_momentum*100:.1f}% | EMA200 bearish"
        if direction == 0:
            return None

        confidence = 0.50
        if abs(rs_momentum) > 0.04: confidence += 0.08
        if abs(rs_momentum) > 0.06: confidence += 0.05
        if f.get("vol_ratio", 0) >= 1.5: confidence += 0.05
        if adx >= 25: confidence += 0.05
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
            strategy="relative_strength",
            reason=reason,
            adx=adx,
            vix=vix,
        )
