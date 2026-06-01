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

from production.strategy.base import Signal  # shared Signal dataclass


@dataclass
class StrategyConfig:
    fast_ema: int = 9
    slow_ema: int = 21
    volume_filter_ratio: float = 1.2
    adx_trending_threshold: float = 20.0   # was 25 — more signals on moderate trends
    stop_atr_multiplier: float = 2.0
    target_atr_multiplier: float = 3.0
    vix_max: float = 25.0
    vix_elevated: float = 20.0
    session_start: str = "09:45"   # skip opening noise
    session_end: str = "15:10"     # no new entries in final 20 min
    min_confidence: float = 0.55           # was 0.60
    require_bar_confirmation: bool = False # EMA crossovers often land on neutral bars
    rsi_long_min: float = 45.0             # RSI floor for longs
    rsi_long_max: float = 70.0             # RSI ceiling for longs (avoid overbought)
    rsi_short_min: float = 30.0            # RSI floor for shorts
    rsi_short_max: float = 55.0            # RSI ceiling for shorts (avoid oversold)


def load_config(path: str = "configs/strategy.yaml") -> StrategyConfig:
    try:
        raw = yaml.safe_load(Path(path).read_text())
        m = raw.get("momentum", {})
        r = raw.get("regime", {})
        v = raw.get("vix", {})
        s = raw.get("session", {})
        e = raw.get("entry", {})
        return StrategyConfig(
            fast_ema=m.get("fast_ema", 9),
            slow_ema=m.get("slow_ema", 21),
            volume_filter_ratio=m.get("volume_filter_ratio", 1.2),
            adx_trending_threshold=r.get("adx_trending_threshold", 20.0),
            stop_atr_multiplier=m.get("stop_atr_multiplier", 2.0),
            target_atr_multiplier=m.get("target_atr_multiplier", 3.0),
            vix_max=v.get("high", 25.0),
            vix_elevated=v.get("elevated", 20.0),
            session_start=s.get("start", "09:45"),
            session_end=s.get("end", "15:10"),
            min_confidence=e.get("min_confidence", 0.55),
            require_bar_confirmation=e.get("require_bar_confirmation", False),
            rsi_long_min=e.get("rsi_long_min", 45.0),
            rsi_long_max=e.get("rsi_long_max", 70.0),
            rsi_short_min=e.get("rsi_short_min", 30.0),
            rsi_short_max=e.get("rsi_short_max", 55.0),
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
        open_: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame with columns: direction, confidence, stop, target, atr.

        Parameters
        ----------
        features : pre-computed feature DataFrame (output of FeatureEngineer)
        close    : raw close price series aligned to features index
        vix      : India VIX series (optional; disables VIX gate if None)
        open_    : raw open price series (optional; used for bar confirmation filter)
        """
        f = features.copy()
        close = close.reindex(f.index)
        open_s = open_.reindex(f.index) if open_ is not None else None

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

        # ── Session time filter: skip opening noise and last 20 min ──────────
        import datetime as _dt
        _s_start = _dt.time(*[int(x) for x in self.cfg.session_start.split(":")])
        _s_end   = _dt.time(*[int(x) for x in self.cfg.session_end.split(":")])
        if hasattr(f.index, "time"):
            _times = f.index.time
            session_ok = pd.Series(
                [_s_start <= t < _s_end for t in _times],
                index=f.index,
            )
        else:
            session_ok = pd.Series(True, index=f.index)

        # ── RSI quality filter: avoid entering at overbought/oversold extremes ──
        rsi = f.get("rsi", pd.Series(50.0, index=f.index))
        rsi_long_ok  = (rsi >= self.cfg.rsi_long_min)  & (rsi <= self.cfg.rsi_long_max)
        rsi_short_ok = (rsi >= self.cfg.rsi_short_min) & (rsi <= self.cfg.rsi_short_max)

        # ── Raw entries — crossover only (strong_trend continuation path
        #    disabled: enters at trend exhaustion, reverses to stop) ──────────
        long_entry  = cross_long  & vol_ok & trending & di_long  & vix_ok & session_ok & rsi_long_ok
        short_entry = cross_short & vol_ok & trending & di_short & vix_ok & session_ok & rsi_short_ok

        direction = pd.Series(0, index=f.index, dtype=int)
        direction[long_entry]  =  1
        direction[short_entry] = -1

        # ── Confidence score ──────────────────────────────────────────────────
        confidence = pd.Series(0.50, index=f.index)
        confidence += np.where((f["adx"] >= 20) & (f["adx"] <= 25), 0.05, 0.0)  # moderate trend
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
        open_price: float | None = None,
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

        # Session time filter
        import datetime as _dt
        _now = pd.Timestamp.now(tz="Asia/Kolkata").time()
        _s_start = _dt.time(*[int(x) for x in self.cfg.session_start.split(":")])
        _s_end   = _dt.time(*[int(x) for x in self.cfg.session_end.split(":")])
        if not (_s_start <= _now < _s_end):
            return None

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
        if direction == 0:
            return None

        # ── RSI quality filter: avoid entering at overbought/oversold extremes ──
        rsi_val = f.get("rsi", 50.0)
        if direction == 1 and not (self.cfg.rsi_long_min <= rsi_val <= self.cfg.rsi_long_max):
            return None
        if direction == -1 and not (self.cfg.rsi_short_min <= rsi_val <= self.cfg.rsi_short_max):
            return None


        # ── News sentiment filter (Phase 2) ───────────────────────────────────
        try:
            from production.data.news_sentiment import get_sentiment
            ns = get_sentiment()
            if ns.should_block(direction, symbol):
                sent_score = ns.score(symbol)
                logger.info("Signal BLOCKED by news sentiment: {} score={:.2f}", symbol, sent_score)
                return None
        except Exception:
            pass  # never block on sentiment fetch failure

        confidence = 0.50
        if 20 <= f.get("adx", 0) <= 25: confidence += 0.05  # moderate trend boost
        if f.get("adx", 0) > 25:  confidence += 0.10
        if f.get("adx", 0) > 35:  confidence += 0.05
        if f.get("vol_ratio", 0) > 1.5: confidence += 0.10
        if f.get("vol_ratio", 0) > 2.0: confidence += 0.05
        if vix is not None and vix < 15:  confidence += 0.05
        if abs(f.get("dist_ema_21", 0)) > 0.003: confidence += 0.05

        # ── XGBoost confidence boost (Phase 2) ────────────────────────────────
        xgb_model_path = Path("models/xgb_direction.pkl")
        if xgb_model_path.exists():
            # _xgb_clf: None = unloaded, False = failed (skip retries), object = loaded
            if not hasattr(self, "_xgb_clf"):
                self._xgb_clf = None
            if self._xgb_clf is not False:
                try:
                    if self._xgb_clf is None:
                        from production.models.xgb_classifier import DirectionClassifier
                        self._xgb_clf = DirectionClassifier().load(xgb_model_path)
                    xgb_pred = self._xgb_clf.predict_bar(f)
                    if xgb_pred["direction"] == direction:
                        confidence += xgb_pred["confidence"] * 0.15
                    elif xgb_pred["direction"] == -direction:
                        confidence -= 0.10
                except Exception as e:
                    logger.warning("XGB boost failed, disabling for session: {}", e)
                    self._xgb_clf = False   # stop retrying this session

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
            strategy="momentum",
            reason=reason,
            adx=f.get("adx", 0.0),
            vix=vix,
            regime=regime,
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
