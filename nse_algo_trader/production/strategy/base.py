"""Shared Signal dataclass and BaseStrategy interface for all Phase 2.1 strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


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
    strategy: str                   # e.g. "momentum", "breakout", "mean_reversion"
    reason: str                     # human-readable entry rationale
    adx: float = 0.0
    vix: float | None = None
    regime: str = ""


class BaseStrategy:
    """All Phase 2.1 strategies implement this interface."""

    name: str = "base"

    def generate_signals_df(
        self,
        features: pd.DataFrame,
        close: pd.Series,
        **kwargs,
    ) -> pd.DataFrame:
        """Vectorised signal generation for backtesting.

        Must return a DataFrame with at minimum:
          direction  : int   — 1 / -1 / 0
          confidence : float — 0–1
          entry      : float — entry price
          stop       : float — stop-loss price
          target     : float — take-profit price
          atr        : float — ATR at signal bar
          strategy   : str   — strategy name
        """
        raise NotImplementedError

    def evaluate_bar(
        self,
        symbol: str,
        latest_features: pd.Series,
        close: float,
        **kwargs,
    ) -> Signal | None:
        """Bar-by-bar live signal. Returns Signal if entry condition met, else None."""
        raise NotImplementedError
