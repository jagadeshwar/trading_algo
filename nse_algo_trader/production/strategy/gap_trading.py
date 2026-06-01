"""Gap Trading Strategy — trade stocks that gap up or gap down at market open.

Two modes:
  A) Continuation (strong gap):
     gap > +1.0% AND vol_ratio > 2.0 AND first bar bullish → long
     gap < -1.0% AND vol_ratio > 2.0 AND first bar bearish → short

  B) Fade (weak gap likely to fill):
     0.5% < gap < 1.0% AND vol_ratio <= 2.0 → short the gap-up / long the gap-down
     Target = gap fill level (prev_close)

Statistical basis:
  - Continuation mode: gaps > 1% with volume surge (2× average) continue in gap direction
    ~55% of the time on NSE (higher on earnings/event days).
  - Fade mode: gaps of 0.5%-1% without strong volume fill within the session ~60% of the time.
    The gap fill is a reliable statistical mean-reversion target on NSE intraday.
  - Session filter: gap trades are ONLY valid in the first 60 minutes (09:15–10:15).
    After that, the gap has either run or started filling — the edge disappears.
  - RSI at open: RSI < 45 at a gap-down = panic selling, likely fade opportunity.
    RSI > 55 at a gap-up = buying pressure, likely continuation.
  - Stop  : Continuation: 2.0 × ATR | Fade: 1.5 × ATR (tighter, gap fill is fast)
  - Target: Continuation: 2.5 × ATR | Fade: gap fill price (prev_close)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from production.strategy.base import BaseStrategy, Signal


class GapTradingConfig:
    def __init__(self, cfg: dict | None = None) -> None:
        c = (cfg or {}).get("gap_trading", {})
        self.gap_continuation_min  = float(c.get("gap_continuation_min",  0.010))  # 1.0%
        self.gap_fade_min          = float(c.get("gap_fade_min",          0.005))  # 0.5%
        self.gap_fade_max          = float(c.get("gap_continuation_min",  0.010))  # up to 1%
        self.vol_continuation_min  = float(c.get("vol_continuation_min",  2.0))    # 2× volume for continuation
        self.stop_cont_atr_mult    = float(c.get("stop_cont_atr_mult",    2.0))
        self.target_cont_atr_mult  = float(c.get("target_cont_atr_mult",  2.5))
        self.stop_fade_atr_mult    = float(c.get("stop_fade_atr_mult",    1.5))
        self.min_confidence        = float(c.get("min_confidence",        0.52))
        self.vix_max               = float(c.get("vix_max",               30.0))
        self.gap_session_end       = c.get("gap_session_end", "10:15")  # only first 60 min


def _load_cfg() -> dict:
    try:
        return yaml.safe_load(Path("configs/strategy.yaml").read_text()) or {}
    except Exception:
        return {}


class GapTradingStrategy(BaseStrategy):
    """Gap-open continuation and gap-fade strategy for NSE equities."""

    name = "gap_trading"

    def __init__(self, cfg: GapTradingConfig | None = None) -> None:
        self.cfg = cfg or GapTradingConfig(_load_cfg())

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

        gap = f.get("gap_pct", pd.Series(0.0, index=f.index))
        gap_up   = gap > 0
        gap_down = gap < 0
        gap_abs  = gap.abs()

        strong_gap = gap_abs >= self.cfg.gap_continuation_min
        weak_gap   = (gap_abs >= self.cfg.gap_fade_min) & (gap_abs < self.cfg.gap_continuation_min)
        vol_surge  = f["vol_ratio"] >= self.cfg.vol_continuation_min

        rsi = f.get("rsi", pd.Series(50.0, index=f.index))
        first_bar_bull = f.get("close_position", pd.Series(0.5, index=f.index)) > 0.5
        first_bar_bear = f.get("close_position", pd.Series(0.5, index=f.index)) < 0.5

        if vix is not None:
            vix_ok = vix.reindex(f.index, method="ffill") < self.cfg.vix_max
        else:
            vix_ok = pd.Series(True, index=f.index)

        # Gap session window: 09:15–10:15 only
        import datetime as _dt
        _gs = _dt.time(9, 15)
        _ge = _dt.time(*[int(x) for x in self.cfg.gap_session_end.split(":")])
        if hasattr(f.index, "time"):
            gap_session = pd.Series([_gs <= t < _ge for t in f.index.time], index=f.index)
        else:
            gap_session = pd.Series(True, index=f.index)

        # ── Mode A: Continuation ───────────────────────────────────────────────
        cont_long  = gap_up   & strong_gap & vol_surge & first_bar_bull & (rsi > 50) & vix_ok & gap_session
        cont_short = gap_down & strong_gap & vol_surge & first_bar_bear & (rsi < 50) & vix_ok & gap_session

        # ── Mode B: Fade ───────────────────────────────────────────────────────
        fade_short = gap_up   & weak_gap & ~vol_surge & (rsi > 55) & vix_ok & gap_session
        fade_long  = gap_down & weak_gap & ~vol_surge & (rsi < 45) & vix_ok & gap_session

        long_entry  = cont_long  | fade_long
        short_entry = cont_short | fade_short

        direction = pd.Series(0, index=f.index, dtype=int)
        direction[long_entry]  =  1
        direction[short_entry] = -1

        # ── Confidence ────────────────────────────────────────────────────────
        confidence = pd.Series(0.50, index=f.index)
        confidence += np.where(cont_long | cont_short, 0.08, 0.0)   # continuation > fade
        confidence += np.where(gap_abs > 0.015, 0.06, 0.0)          # larger gap = stronger
        confidence += np.where(f["vol_ratio"] >= 2.5, 0.06, 0.0)
        confidence = confidence.clip(0.0, 1.0).where(direction != 0, 0.0)

        # ── Targets: continuation = ATR-based; fade = gap fill (prev close) ──
        # gap_pct = (open - prev_close) / prev_close → prev_close = open / (1 + gap_pct)
        # Use open (close at gap bar) as proxy for open price
        prev_close_proxy = c / (1 + gap.replace(0, np.nan)).fillna(c)
        stop_cont   = atr * self.cfg.stop_cont_atr_mult
        target_cont = atr * self.cfg.target_cont_atr_mult
        stop_fade   = atr * self.cfg.stop_fade_atr_mult

        is_cont = (cont_long | cont_short)
        stop   = np.where(direction ==  1,
                     np.where(is_cont, c - stop_cont, c - stop_fade),
                 np.where(direction == -1,
                     np.where(is_cont, c + stop_cont, c + stop_fade),
                 np.nan))
        target = np.where(direction ==  1,
                     np.where(is_cont, c + target_cont, prev_close_proxy),
                 np.where(direction == -1,
                     np.where(is_cont, c - target_cont, prev_close_proxy),
                 np.nan))

        return pd.DataFrame({
            "direction":  direction,
            "confidence": confidence,
            "entry":      c,
            "stop":       stop,
            "target":     target,
            "atr":        atr,
            "adx":        f["adx"],
            "strategy":   "gap_trading",
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

        import datetime as _dt
        _now = pd.Timestamp.now(tz="Asia/Kolkata").time()
        _gs  = _dt.time(9, 15)
        _ge  = _dt.time(*[int(x) for x in self.cfg.gap_session_end.split(":")])
        if not (_gs <= _now < _ge):
            return None  # gap trading only in first hour

        gap = f.get("gap_pct", 0.0)
        if pd.isna(gap) or abs(gap) < self.cfg.gap_fade_min:
            return None

        vol_ratio  = f.get("vol_ratio", 1.0)
        rsi        = f.get("rsi", 50.0)
        close_pos  = f.get("close_position", 0.5)
        vix_ok     = (vix is None) or (vix < self.cfg.vix_max)
        if not vix_ok:
            return None

        strong = abs(gap) >= self.cfg.gap_continuation_min
        vol_surge = vol_ratio >= self.cfg.vol_continuation_min
        prev_close_proxy = close / (1 + gap) if gap != 0 else close

        direction = 0
        reason    = ""
        mode      = ""

        if gap > 0:  # gap up
            if strong and vol_surge and close_pos > 0.5 and rsi > 50:
                direction = 1
                mode = "CONTINUATION"
                reason = f"Gap-up continuation gap={gap*100:.1f}% vol={vol_ratio:.1f}x RSI={rsi:.0f}"
            elif not strong and not vol_surge and rsi > 55:
                direction = -1
                mode = "FADE"
                reason = f"Gap-up fade gap={gap*100:.1f}% weak vol={vol_ratio:.1f}x target=gap fill"
        elif gap < 0:  # gap down
            if strong and vol_surge and close_pos < 0.5 and rsi < 50:
                direction = -1
                mode = "CONTINUATION"
                reason = f"Gap-down continuation gap={gap*100:.1f}% vol={vol_ratio:.1f}x RSI={rsi:.0f}"
            elif not strong and not vol_surge and rsi < 45:
                direction = 1
                mode = "FADE"
                reason = f"Gap-down fade gap={gap*100:.1f}% weak vol={vol_ratio:.1f}x target=gap fill"

        if direction == 0:
            return None

        confidence = 0.50
        if mode == "CONTINUATION": confidence += 0.08
        if abs(gap) > 0.015:       confidence += 0.06
        if vol_ratio >= 2.5:       confidence += 0.06
        confidence = min(1.0, confidence)

        if confidence < self.cfg.min_confidence:
            return None

        if mode == "CONTINUATION":
            stop   = close - direction * atr * self.cfg.stop_cont_atr_mult
            target = close + direction * atr * self.cfg.target_cont_atr_mult
        else:
            stop   = close - direction * atr * self.cfg.stop_fade_atr_mult
            target = prev_close_proxy

        return Signal(
            time=pd.Timestamp.now(tz="Asia/Kolkata"),
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=close,
            stop_price=stop,
            target_price=target,
            atr=atr,
            strategy=f"gap_trading_{mode.lower()}",
            reason=reason,
            vix=vix,
        )
