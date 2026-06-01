"""Phase 2.1 multi-strategy package."""

from production.strategy.base                  import BaseStrategy, Signal
from production.strategy.momentum              import MomentumStrategy, StrategyConfig, load_config
from production.strategy.trend_following       import TrendFollowingStrategy
from production.strategy.breakout              import BreakoutStrategy
from production.strategy.mean_reversion        import MeanReversionStrategy
from production.strategy.volatility_contraction import VolatilityContractionStrategy
from production.strategy.range_trading         import RangeTradingStrategy
from production.strategy.relative_strength     import RelativeStrengthStrategy
from production.strategy.gap_trading           import GapTradingStrategy
from production.strategy.price_action          import PriceActionStrategy
from production.strategy.swing_trading         import SwingTradingStrategy
from production.strategy.options_strategies    import (
    # Bullish
    BuyCallStrategy, SellPutStrategy, BullCallSpreadStrategy, BullPutSpreadStrategy,
    CallBackSpreadStrategy, CallFrontSpreadStrategy,
    # Bearish
    BuyPutStrategy, SellCallStrategy, BearPutSpreadStrategy, BearCallSpreadStrategy,
    PutBackSpreadStrategy, PutFrontSpreadStrategy,
    # Neutral (original + new)
    IronCondorStrategy, ShortIronButterflyStrategy, ShortIronWonderStrategy,
    ShortStraddleStrategy, ShortStrangleStrategy,
    LongCallButterflyStrategy, LongCalendarStrategy,
    # Data classes
    OptionsSignal, OptionsLeg,
    # Registry
    ALL_OPTIONS_STRATEGIES, OPTIONS_REGISTRY,
)
from production.strategy.orchestrator          import StrategyOrchestrator

__all__ = [
    "BaseStrategy", "Signal",
    "MomentumStrategy", "StrategyConfig", "load_config",
    "TrendFollowingStrategy", "BreakoutStrategy",
    "MeanReversionStrategy", "VolatilityContractionStrategy",
    "RangeTradingStrategy", "RelativeStrengthStrategy",
    "GapTradingStrategy", "PriceActionStrategy", "SwingTradingStrategy",
    # Options — bullish
    "BuyCallStrategy", "SellPutStrategy", "BullCallSpreadStrategy", "BullPutSpreadStrategy",
    "CallBackSpreadStrategy", "CallFrontSpreadStrategy",
    # Options — bearish
    "BuyPutStrategy", "SellCallStrategy", "BearPutSpreadStrategy", "BearCallSpreadStrategy",
    "PutBackSpreadStrategy", "PutFrontSpreadStrategy",
    # Options — neutral
    "IronCondorStrategy", "ShortIronButterflyStrategy", "ShortIronWonderStrategy",
    "ShortStraddleStrategy", "ShortStrangleStrategy",
    "LongCallButterflyStrategy", "LongCalendarStrategy",
    "OptionsSignal", "OptionsLeg", "ALL_OPTIONS_STRATEGIES", "OPTIONS_REGISTRY",
    "StrategyOrchestrator",
]
