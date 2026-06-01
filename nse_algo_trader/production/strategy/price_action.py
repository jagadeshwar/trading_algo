"""Price Action Trading Strategy — candlestick patterns at key Support/Resistance levels.

Entry logic (pattern + location):
  Long  : Bullish reversal candle (Hammer / Bullish Engulfing / Doji)
           forms AT or NEAR a pivot support level
  Short : Bearish reversal candle (Shooting Star / Bearish Engulfing / Doji)
           forms AT or NEAR a pivot resistance level

Candlestick pattern definitions (from features.py):
  Hammer       : lower_wick > 2.5 × body, upper_wick < 0.5 × body, bullish close preferred
  Shooting Star: upper_wick > 2.5 × body, lower_wick < 0.5 × body, bearish close preferred
  Bullish Engulf: current bar (bullish) fully engulfs previous bearish bar
  Bearish Engulf: current bar (bearish) fully engulfs previous bullish bar
  Doji         : body < 5% of bar range (indecision) — only valid AT S/R levels
  Pin Bar      : any wick > 3 × body (rejection of price level)

Statistical basis:
  - Pattern at S/R: candle patterns in isolation have ~52% accuracy.
    At a confirmed S/R level, accuracy improves to ~62% (Baker, 2018; NSE backtests).
  - Confirmation requirement: pattern must appear within 0.5% of a pivot S/R level.
    This eliminates random mid-range patterns and focuses on level-reaction signals.
  - Min ADX independence: works in both trending and ranging markets — no ADX filter.
    Price action at key levels works regardless of broader trend context.
  - Volume confirmation boost: candles with vol_ratio > 1.2 on pattern bar = stronger.
  - Stop  : 1.0 × ATR (tight — pattern-based stops = pattern failure = immediate exit)
  - Target: 2.0 × ATR to next S/R level (2:1 R:R minimum)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from production.strategy.base import BaseStrategy, Signal


class PriceActionConfig:
    def __init__(self, cfg: dict | None = None) -> None:
        c = (cfg or {}).get("price_action", {})
        s = (cfg or {}).get("session", {})
        self.sr_proximity_pct  = float(c.get("sr_proximity_pct",  0.005))  # within 0.5% of level
        self.volume_confirm    = float(c.get("volume_confirm",    1.2))    # light volume OK for PA
        self.stop_atr_mult     = float(c.get("stop_atr_mult",     1.0))
        self.target_atr_mult   = float(c.get("target_atr_mult",   2.0))
        self.min_confidence    = float(c.get("min_confidence",    0.52))
        self.vix_max           = float(c.get("vix_max",           28.0))
        self.session_start     = s.get("start", "09:45")
        self.session_end       = s.get("end",   "15:10")


def _load_cfg() -> dict:
    try:
        return yaml.safe_load(Path("configs/strategy.yaml").read_text()) or {}
    except Exception:
        return {}


class PriceActionStrategy(BaseStrategy):
    """Candlestick pattern + S/R confluence strategy for NSE."""

    name = "price_action"

    def __init__(self, cfg: PriceActionConfig | None = None) -> None:
        self.cfg = cfg or PriceActionConfig(_load_cfg())

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

        # ── S/R levels ─────────────────────────────────────────────────────────
        if "pivot_resistance" in f.columns and "pivot_support" in f.columns:
            resistance = f["pivot_resistance"]
            support    = f["pivot_support"]
        else:
            resistance = c.rolling(20).max()
            support    = c.rolling(20).min()

        near_support    = (c - support).abs()    / support    < self.cfg.sr_proximity_pct
        near_resistance = (c - resistance).abs() / resistance < self.cfg.sr_proximity_pct

        # ── Candlestick pattern flags from features ────────────────────────────
        hammer        = f.get("cdl_hammer",         pd.Series(0, index=f.index)) == 1
        shooting_star = f.get("cdl_shooting_star",  pd.Series(0, index=f.index)) == 1
        bull_engulf   = f.get("cdl_bullish_engulf", pd.Series(0, index=f.index)) == 1
        bear_engulf   = f.get("cdl_bearish_engulf", pd.Series(0, index=f.index)) == 1
        doji          = f.get("cdl_doji",           pd.Series(0, index=f.index)) == 1
        pin_bar_bull  = f.get("cdl_pin_bar_bull",   pd.Series(0, index=f.index)) == 1
        pin_bar_bear  = f.get("cdl_pin_bar_bear",   pd.Series(0, index=f.index)) == 1

        # Bullish reversal patterns at support
        bullish_pattern = hammer | bull_engulf | (doji & near_support) | pin_bar_bull
        # Bearish reversal patterns at resistance
        bearish_pattern = shooting_star | bear_engulf | (doji & near_resistance) | pin_bar_bear

        vol_ok = f["vol_ratio"] >= self.cfg.volume_confirm

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

        long_entry  = bullish_pattern & near_support    & vix_ok & session_ok
        short_entry = bearish_pattern & near_resistance & vix_ok & session_ok

        direction = pd.Series(0, index=f.index, dtype=int)
        direction[long_entry]  =  1
        direction[short_entry] = -1

        # ── Confidence ────────────────────────────────────────────────────────
        confidence = pd.Series(0.50, index=f.index)
        confidence += np.where(bull_engulf | bear_engulf, 0.10, 0.0)  # engulfing is strongest
        confidence += np.where(hammer | shooting_star,    0.08, 0.0)
        confidence += np.where(vol_ok,                    0.05, 0.0)
        confidence += np.where(f["vol_ratio"] >= 1.5,     0.03, 0.0)
        # Confluence with trend
        dist200 = f.get("dist_ema_200", pd.Series(0.0, index=f.index))
        confidence += np.where((direction == 1)  & (dist200 > 0), 0.05, 0.0)
        confidence += np.where((direction == -1) & (dist200 < 0), 0.05, 0.0)
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
            "strategy":   "price_action",
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

        resistance = f.get("pivot_resistance", None)
        support    = f.get("pivot_support",    None)
        if resistance is None or support is None or pd.isna(resistance) or pd.isna(support):
            return None

        near_support    = abs(close - support)    / support    < self.cfg.sr_proximity_pct if support > 0 else False
        near_resistance = abs(close - resistance) / resistance < self.cfg.sr_proximity_pct if resistance > 0 else False

        hammer        = f.get("cdl_hammer",         0) == 1
        shooting_star = f.get("cdl_shooting_star",  0) == 1
        bull_engulf   = f.get("cdl_bullish_engulf", 0) == 1
        bear_engulf   = f.get("cdl_bearish_engulf", 0) == 1
        doji          = f.get("cdl_doji",           0) == 1
        pin_bar_bull  = f.get("cdl_pin_bar_bull",   0) == 1
        pin_bar_bear  = f.get("cdl_pin_bar_bear",   0) == 1
        vix_ok        = (vix is None) or (vix < self.cfg.vix_max)

        import datetime as _dt
        _now = pd.Timestamp.now(tz="Asia/Kolkata").time()
        _ss  = _dt.time(*[int(x) for x in self.cfg.session_start.split(":")])
        _se  = _dt.time(*[int(x) for x in self.cfg.session_end.split(":")])
        if not (_ss <= _now < _se) or not vix_ok:
            return None

        bullish_pat = hammer or bull_engulf or (doji and near_support) or pin_bar_bull
        bearish_pat = shooting_star or bear_engulf or (doji and near_resistance) or pin_bar_bear

        direction = 0
        reason    = ""
        pattern   = ""
        if bullish_pat and near_support:
            direction = 1
            pattern   = "hammer" if hammer else ("bull_engulf" if bull_engulf else "pin_bar" if pin_bar_bull else "doji")
            reason = f"Bullish {pattern} at support={support:.2f}"
        elif bearish_pat and near_resistance:
            direction = -1
            pattern   = "shooting_star" if shooting_star else ("bear_engulf" if bear_engulf else "pin_bar" if pin_bar_bear else "doji")
            reason = f"Bearish {pattern} at resistance={resistance:.2f}"
        if direction == 0:
            return None

        confidence = 0.50
        if bull_engulf or bear_engulf: confidence += 0.10
        if hammer or shooting_star:    confidence += 0.08
        if f.get("vol_ratio", 0) >= self.cfg.volume_confirm: confidence += 0.05
        dist200 = f.get("dist_ema_200", 0.0)
        if direction == 1 and dist200 > 0:  confidence += 0.05
        if direction == -1 and dist200 < 0: confidence += 0.05
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
            strategy="price_action",
            reason=reason,
            vix=vix,
        )
