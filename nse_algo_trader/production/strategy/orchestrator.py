"""Phase 2.1 Strategy Orchestrator — routes to the best strategy based on market regime.

Regime → Primary Strategy mapping:
  TRENDING_UP   → TrendFollowing, Breakout, Swing (long bias), RelativeStrength
  TRENDING_DOWN → TrendFollowing, Breakout, Swing (short bias), RelativeStrength
  RANGING       → MeanReversion, RangeTrading, IronCondor, VolatilityContraction
  HIGH_VOL      → GapTrading (open only), VolatilityContraction (post-spike)
  ANY           → PriceAction (always active — pattern-based, no regime dependency)
  ANY (morning) → GapTrading (09:15–10:15 only)

Priority rules:
  1. Each regime has a ranked list of strategies.
  2. All enabled strategies generate signals for their applicable conditions.
  3. The orchestrator returns ALL active signals, ranked by confidence.
  4. The live engine picks the highest-confidence signal per symbol (no duplicates).
  5. Options conditions are reported separately (not directional positions).

Usage (backtesting):
    orch = StrategyOrchestrator()
    signals = orch.generate_signals_df(features, close, vix=vix_series)

Usage (live):
    signal = orch.evaluate_bar(symbol, features_row, close, vix=vix)
    # Returns Signal | None (best signal from all active strategies)
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger

from production.strategy.base import Signal
from production.strategy.breakout               import BreakoutStrategy
from production.strategy.gap_trading            import GapTradingStrategy
from production.strategy.mean_reversion         import MeanReversionStrategy
from production.strategy.momentum               import MomentumStrategy        # Phase 1 (kept)
from production.strategy.options_strategies     import ALL_OPTIONS_STRATEGIES, OPTIONS_REGISTRY
from production.strategy.price_action           import PriceActionStrategy
from production.strategy.range_trading          import RangeTradingStrategy
from production.strategy.relative_strength      import RelativeStrengthStrategy
from production.strategy.swing_trading          import SwingTradingStrategy
from production.strategy.trend_following        import TrendFollowingStrategy
from production.strategy.volatility_contraction import VolatilityContractionStrategy


REGIME_STRATEGY_MAP: dict[str, list[str]] = {
    "TRENDING_UP":   ["trend_following", "breakout", "swing_trading", "relative_strength", "price_action", "gap_trading"],
    "TRENDING_DOWN": ["trend_following", "breakout", "swing_trading", "relative_strength", "price_action", "gap_trading"],
    "RANGING":       ["mean_reversion",  "range_trading", "volatility_contraction", "price_action", "gap_trading"],
    "HIGH_VOL":      ["volatility_contraction", "gap_trading", "price_action"],
    "UNKNOWN":       ["trend_following", "momentum", "price_action", "gap_trading"],
}


def _load_cfg() -> dict:
    try:
        return yaml.safe_load(Path("configs/strategy.yaml").read_text()) or {}
    except Exception:
        return {}


class StrategyOrchestrator:
    """Runs all applicable strategies and returns ranked signals."""

    def __init__(self) -> None:
        cfg = _load_cfg()
        p21 = cfg.get("phase21", {})
        enabled = set(p21.get("enabled_strategies", [
            "momentum", "trend_following", "breakout", "mean_reversion",
            "volatility_contraction", "range_trading", "relative_strength",
            "gap_trading", "price_action", "swing_trading",
        ]))

        # Directional strategies
        self._strategies: dict[str, object] = {}
        if "momentum"              in enabled: self._strategies["momentum"]              = MomentumStrategy()
        if "trend_following"       in enabled: self._strategies["trend_following"]       = TrendFollowingStrategy()
        if "breakout"              in enabled: self._strategies["breakout"]              = BreakoutStrategy()
        if "mean_reversion"        in enabled: self._strategies["mean_reversion"]        = MeanReversionStrategy()
        if "volatility_contraction" in enabled: self._strategies["volatility_contraction"] = VolatilityContractionStrategy()
        if "range_trading"         in enabled: self._strategies["range_trading"]         = RangeTradingStrategy()
        if "relative_strength"     in enabled: self._strategies["relative_strength"]     = RelativeStrengthStrategy()
        if "gap_trading"           in enabled: self._strategies["gap_trading"]           = GapTradingStrategy()
        if "price_action"          in enabled: self._strategies["price_action"]          = PriceActionStrategy()
        if "swing_trading"         in enabled: self._strategies["swing_trading"]         = SwingTradingStrategy()

        # Options condition detectors (no directional trades)
        # Uses the global ALL_OPTIONS_STRATEGIES registry — any name that appears in
        # enabled_strategies and has a matching options strategy will be loaded.
        self._options_detectors = [
            s for s in ALL_OPTIONS_STRATEGIES if s.name in enabled
        ]

        logger.info("StrategyOrchestrator: {} directional strategies, {} options detectors loaded",
                    len(self._strategies), len(self._options_detectors))

    # ── Live bar-by-bar ──────────────────────────────────────────────────────

    def evaluate_bar(
        self,
        symbol: str,
        latest_features: pd.Series,
        close: float,
        vix: float | None = None,
        nifty_close: float | None = None,
        **kwargs,
    ) -> Signal | None:
        """Evaluate all applicable strategies. Return highest-confidence signal or None."""
        regime_code = int(latest_features.get("regime_heuristic", 0))
        regime_map  = {0: "RANGING", 1: "TRENDING_UP", 2: "TRENDING_DOWN", 3: "HIGH_VOL"}
        regime      = regime_map.get(regime_code, "UNKNOWN")

        applicable = REGIME_STRATEGY_MAP.get(regime, REGIME_STRATEGY_MAP["UNKNOWN"])
        candidates: list[Signal] = []

        for name in applicable:
            strat = self._strategies.get(name)
            if strat is None:
                continue
            try:
                extra: dict = {}
                if name == "relative_strength":
                    extra["nifty_close"] = nifty_close  # None is handled inside strategy
                sig = strat.evaluate_bar(symbol, latest_features, close, vix=vix, **extra)
                if sig is not None and sig.direction != 0:
                    candidates.append(sig)
            except Exception as e:
                logger.debug("Strategy {} error: {}", name, e)

        if not candidates:
            return None

        # Pick highest-confidence signal
        best = max(candidates, key=lambda s: s.confidence)
        if len(candidates) > 1:
            logger.debug("Orchestrator: {} candidates for {} → best={} conf={:.2f}",
                         len(candidates), symbol, best.strategy, best.confidence)
        return best

    def evaluate_options(
        self,
        symbol: str,
        latest_features: pd.Series,
        close: float,
        vix: float | None = None,
    ) -> list:
        """Return list of OptionsSignal objects for any active options conditions."""
        results = []
        for det in self._options_detectors:
            try:
                sig = det.evaluate_bar(symbol, latest_features, close, vix=vix)
                if sig is not None and sig.condition_met:
                    results.append(sig)
            except Exception as e:
                logger.debug("Options detector {} error: {}", det.name, e)
        return results

    # ── Vectorised backtesting ────────────────────────────────────────────────

    def generate_signals_df(
        self,
        features: pd.DataFrame,
        close: pd.Series,
        vix: pd.Series | None = None,
        nifty_close: pd.Series | None = None,
        strategy_name: str | None = None,
    ) -> pd.DataFrame:
        """Generate signals from a specific strategy or all strategies combined.

        If strategy_name is given, run only that strategy (for isolated backtesting).
        Otherwise, combine all strategies: for each bar, pick the signal from the
        applicable regime strategy with highest confidence.
        """
        if strategy_name:
            strat = self._strategies.get(strategy_name)
            if strat is None:
                raise ValueError(f"Strategy '{strategy_name}' not loaded. Check enabled_strategies in config.")
            extra = {}
            if strategy_name == "relative_strength" and nifty_close is not None:
                extra["nifty_close"] = nifty_close
            return strat.generate_signals_df(features, close, vix=vix, **extra)

        # Combined: run each strategy, then merge by taking max-confidence signal per bar
        all_dfs: list[pd.DataFrame] = []
        for name, strat in self._strategies.items():
            try:
                extra = {}
                if name == "relative_strength":
                    extra["symbol"] = strategy_name or ""  # let RS block Nifty50 itself
                    extra["nifty_close"] = nifty_close
                df = strat.generate_signals_df(features, close, vix=vix, **extra)
                df["strategy_name"] = name
                all_dfs.append(df)
            except Exception as e:
                logger.debug("Vectorised strategy {} failed: {}", name, e)

        if not all_dfs:
            empty = pd.DataFrame({"direction": 0, "confidence": 0.0, "entry": close,
                                   "stop": float("nan"), "target": float("nan"),
                                   "atr": 0.0, "adx": 0.0, "strategy": "none",
                                   "strategy_name": "none"}, index=features.index)
            return empty

        # For each timestamp, select the signal with the highest confidence
        # across all strategies that fired a non-zero direction
        import numpy as np
        combined = pd.concat(all_dfs, ignore_index=False)
        active   = combined[combined["direction"] != 0].copy()
        if active.empty:
            return all_dfs[0].copy().assign(direction=0, confidence=0.0)

        # Keep one signal per timestamp: highest confidence
        best_idx  = active.groupby(active.index)["confidence"].idxmax()
        best_rows = active.loc[best_idx.values]

        # Merge back into base frame (fill non-signal bars with 0 direction)
        base = all_dfs[0][["entry", "atr", "adx", "regime"]].copy()
        base["direction"]  = 0
        base["confidence"] = 0.0
        base["stop"]       = float("nan")
        base["target"]     = float("nan")
        base["strategy"]   = "none"

        for col in ["direction", "confidence", "stop", "target", "strategy"]:
            base.loc[best_rows.index, col] = best_rows[col].values if col != "strategy" else best_rows["strategy_name"].values

        return base
