"""Momentum strategy — EMA 9/21 crossover with volume, ADX, and VIX filters.

Signal logic:
  Long  entry: EMA9 crosses above EMA21 AND volume > 1.2× MA AND ADX > 25 AND VIX < 25
  Short entry: EMA9 crosses below EMA21 AND volume > 1.2× MA AND ADX > 25 AND VIX < 25
  Stop loss  : 2 × ATR from entry
  Take profit: 3 × ATR from entry  (1.5 R:R)

Confidence score drives position size via Kelly criterion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from loguru import logger


@dataclass
class Signal:
    time: pd.Timestamp
    symbol: str
    direction: Literal[1, -1, 0]   # 1=Long, -1=Short, 0=Flat
    confidence: float               # 0–1
    entry_price: float
    stop_price: float
    target_price: float
    atr: float
    adx: float
    vix: float | None
    regime: str
    reason: str                     # human-readable why signal fired


@dataclass
class StrategyConfig:
    fast_ema: int = 9
    slow_ema: int = 21
    volume_filter_ratio: float = 1.2
    adx_trending_threshold: float = 25.0
    stop_atr_multiplier: float = 2.0
    target_atr_multiplier: float = 3.0
    vix_max: float = 25.0
    vix_elevated: float = 20.0


def load_config(path: str = "configs/strategy.yaml") -> StrategyConfig:
    try:
        raw = yaml.safe_load(Path(path).read_text())
        m = raw.get("momentum", {})
        r = raw.get("regime", {})
        v = raw.get("vix", {})
        return StrategyConfig(
            fast_ema=m.get("fast_ema", 9),
            slow_ema=m.get("slow_ema", 21),
            volume_filter_ratio=m.get("volume_filter_ratio", 1.2),
            adx_trending_threshold=r.get("adx_trending_threshold", 25.0),
            stop_atr_multiplier=m.get("stop_atr_multiplier", 2.0),
            target_atr_multiplier=m.get("target_atr_multiplier", 3.0),
            vix_max=v.get("high", 25.0),
            vix_elevated=v.get("elevated", 20.0),
        )
    except Exception as e:
        logger.warning("Could not load strategy config ({}), using defaults", e)
        return StrategyConfig()


class MomentumStrategy:
    """EMA crossover momentum strategy for NSE indices and equities."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.cfg = config or load_config()

    # ── Vectorised signal generation (for backtesting) ────────────────────────

    def generate_signals_df(
        self,
        features: pd.DataFrame,
        close: pd.Series,
        vix: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame with columns: direction, confidence, stop, target, atr.

        Parameters
        ----------
        features : pre-computed feature DataFrame (output of FeatureEngineer)
        close    : raw close price series aligned to features index
        vix      : India VIX series (optional; disables VIX gate if None)
        """
        f = features.copy()
        close = close.reindex(f.index)

        # ── Absolute ATR ──────────────────────────────────────────────────────
        atr = f["atr_pct"] * close  # atr_pct = ATR/close → ATR = atr_pct × close

        # ── Filters ───────────────────────────────────────────────────────────
        cross_long  = f["ema_9_21_cross"] > 0   # EMA9 just crossed above EMA21
        cross_short = f["ema_9_21_cross"] < 0   # EMA9 just crossed below EMA21
        vol_ok      = f["vol_ratio"] >= self.cfg.volume_filter_ratio
        trending    = f["adx"] >= self.cfg.adx_trending_threshold
        di_long     = f["di_diff"] > 0          # +DI > -DI (bullish trend direction)
        di_short    = f["di_diff"] < 0

        if vix is not None:
            vix_aligned = vix.reindex(f.index, method="ffill")
            vix_ok = vix_aligned < self.cfg.vix_max
        else:
            vix_ok = pd.Series(True, index=f.index)

        # ── Raw entries ───────────────────────────────────────────────────────
        long_entry  = cross_long  & vol_ok & trending & di_long  & vix_ok
        short_entry = cross_short & vol_ok & trending & di_short & vix_ok

        direction = pd.Series(0, index=f.index, dtype=int)
        direction[long_entry]  =  1
        direction[short_entry] = -1

        # ── Confidence score ──────────────────────────────────────────────────
        confidence = pd.Series(0.50, index=f.index)
        confidence += np.where(f["adx"] > 25,  0.10, 0.0)
        confidence += np.where(f["adx"] > 35,  0.05, 0.0)
        confidence += np.where(f["vol_ratio"] > 1.5, 0.10, 0.0)
        confidence += np.where(f["vol_ratio"] > 2.0, 0.05, 0.0)
        if vix is not None:
            vix_aligned = vix.reindex(f.index, method="ffill")
            confidence += np.where(vix_aligned < 15, 0.05, 0.0)
        confidence += np.where(f["dist_ema_21"].abs() > 0.003, 0.05, 0.0)
        confidence = confidence.clip(0.0, 1.0)

        # Only apply confidence to actual signal bars
        confidence = confidence.where(direction != 0, 0.0)

        # ── Stop and target prices ─────────────────────────────────────────────
        stop_dist   = atr * self.cfg.stop_atr_multiplier
        target_dist = atr * self.cfg.target_atr_multiplier

        stop   = np.where(direction ==  1, close - stop_dist,
                 np.where(direction == -1, close + stop_dist, np.nan))
        target = np.where(direction ==  1, close + target_dist,
                 np.where(direction == -1, close - target_dist, np.nan))

        return pd.DataFrame({
            "direction":  direction,
            "confidence": confidence,
            "entry":      close,
            "stop":       stop,
            "target":     target,
            "atr":        atr,
            "adx":        f["adx"],
            "regime":     f["regime_heuristic"],
        }, index=f.index)

    # ── Bar-by-bar signal for live use ────────────────────────────────────────

    def evaluate_bar(
        self,
        symbol: str,
        latest_features: pd.Series,
        close: float,
        vix: float | None = None,
    ) -> Signal | None:
        """Evaluate a single bar. Returns Signal if entry condition met, else None."""
        f = latest_features
        atr = f.get("atr_pct", 0.0) * close

        cross_long  = f.get("ema_9_21_cross", 0) > 0
        cross_short = f.get("ema_9_21_cross", 0) < 0
        vol_ok      = f.get("vol_ratio", 0.0) >= self.cfg.volume_filter_ratio
        trending    = f.get("adx", 0.0) >= self.cfg.adx_trending_threshold
        di_long     = f.get("di_diff", 0.0) > 0
        di_short    = f.get("di_diff", 0.0) < 0
        vix_ok      = (vix is None) or (vix < self.cfg.vix_max)
        adx         = f.get("adx", 0.0)
        dist21      = f.get("dist_ema_21", 0.0)
        di_diff     = f.get("di_diff", 0.0)

        # Trend continuation: strong ADX + price on correct side of EMA21
        strong_trend_long  = adx >= 40 and dist21 > 0.001 and di_diff > 15
        strong_trend_short = adx >= 40 and dist21 < -0.001 and di_diff < -15

        vol_ok_relaxed = f.get("vol_ratio", 0.0) >= 0.9  # relaxed for continuation

        if not (vix_ok and trending):
            return None

        direction = 0
        reason = ""
        if vol_ok and cross_long and di_long:
            direction = 1
            reason = f"EMA9 crossed above EMA21 | ADX={adx:.1f} | vol_ratio={f.get('vol_ratio',0):.2f}"
        elif vol_ok and cross_short and di_short:
            direction = -1
            reason = f"EMA9 crossed below EMA21 | ADX={adx:.1f} | vol_ratio={f.get('vol_ratio',0):.2f}"
        elif vol_ok_relaxed and strong_trend_long:
            direction = 1
            reason = f"Strong trend continuation LONG | ADX={adx:.1f} | dist_ema21={dist21:.4f}"
        elif vol_ok_relaxed and strong_trend_short:
            direction = -1
            reason = f"Strong trend continuation SHORT | ADX={adx:.1f} | dist_ema21={dist21:.4f}"

        if direction == 0:
            return None

        confidence = 0.50
        if f.get("adx", 0) > 25:  confidence += 0.10
        if f.get("adx", 0) > 35:  confidence += 0.05
        if f.get("vol_ratio", 0) > 1.5: confidence += 0.10
        if f.get("vol_ratio", 0) > 2.0: confidence += 0.05
        if vix is not None and vix < 15:  confidence += 0.05
        if abs(f.get("dist_ema_21", 0)) > 0.003: confidence += 0.05

        # ── XGBoost confidence boost (Phase 2) ────────────────────────────────
        xgb_model_path = Path("models/xgb_direction.pkl")
        if xgb_model_path.exists():
            try:
                from production.models.xgb_classifier import DirectionClassifier
                if not hasattr(self, "_xgb_clf") or self._xgb_clf is None:
                    self._xgb_clf = DirectionClassifier().load(xgb_model_path)
                xgb_pred = self._xgb_clf.predict_bar(f)
                if xgb_pred["direction"] == direction:
                    confidence += xgb_pred["confidence"] * 0.15  # boost if aligned
                elif xgb_pred["direction"] == -direction:
                    confidence -= 0.10                            # penalise if opposed
            except Exception:
                pass

        confidence = min(1.0, max(0.0, confidence))

        stop_dist   = atr * self.cfg.stop_atr_multiplier
        target_dist = atr * self.cfg.target_atr_multiplier
        stop   = close - direction * stop_dist
        target = close + direction * target_dist

        regime_hmm_map = {0: "RANGING", 1: "TRENDING", 2: "HIGH_VOL"}
        regime_heur_map = {0: "RANGING", 1: "TRENDING_UP", 2: "TRENDING_DOWN", 3: "HIGH_VOL"}
        hmm_regime  = regime_hmm_map.get(int(f.get("regime_hmm", -1)), "")
        heur_regime = regime_heur_map.get(int(f.get("regime_heuristic", 0)), "UNKNOWN")
        regime = hmm_regime if hmm_regime else heur_regime

        return Signal(
            time=pd.Timestamp.now(tz="Asia/Kolkata"),
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=close,
            stop_price=stop,
            target_price=target,
            atr=atr,
            adx=f.get("adx", 0.0),
            vix=vix,
            regime=regime,
            reason=reason,
        )

    # ── Position management helpers ───────────────────────────────────────────

    def should_exit(
        self,
        direction: int,
        entry: float,
        stop: float,
        target: float,
        current_price: float,
        latest_features: pd.Series,
    ) -> tuple[bool, str]:
        """Return (exit_now, reason). Call every bar for open positions."""
        # Stop hit
        if direction == 1 and current_price <= stop:
            return True, f"STOP_HIT price={current_price:.2f} stop={stop:.2f}"
        if direction == -1 and current_price >= stop:
            return True, f"STOP_HIT price={current_price:.2f} stop={stop:.2f}"

        # Target hit
        if direction == 1 and current_price >= target:
            return True, f"TARGET_HIT price={current_price:.2f} target={target:.2f}"
        if direction == -1 and current_price <= target:
            return True, f"TARGET_HIT price={current_price:.2f} target={target:.2f}"

        # Signal reversal
        cross = latest_features.get("ema_9_21_cross", 0)
        if direction == 1 and cross < 0:
            return True, "SIGNAL_REVERSAL bearish crossover"
        if direction == -1 and cross > 0:
            return True, "SIGNAL_REVERSAL bullish crossover"

        return False, ""
