"""Phase 2.1 Options Strategies — full multi-strategy suite for NSE index options.

Covers 18 strategies across three market outlooks.

BULLISH (6):
  Buy Call          · Bull Call Spread  · Bull Put Spread
  Sell Put          · Call Back Spread  · Call Front Spread

BEARISH (6):
  Buy Put           · Bear Put Spread   · Bear Call Spread
  Sell Call         · Put Back Spread   · Put Front Spread

NEUTRAL (6):
  Long Call Butterfly  · Short Iron Butterfly  · Short Iron Wonder
  Short Straddle       · Short Strangle        · Long Calendar

All strategies:
  - Compute approximate strikes via Expected Move formula
  - Use symbol-aware strike rounding (Nifty=50pt, BankNifty=100pt, Stocks=5pt)
  - Return OptionsSignal with full leg structure and risk profile
  - Execution (live chain, Greeks, order routing) → Phase 4

Strike selection is based on the standard deviation formula:
  EM  = close × (VIX/100) × √(DTE/365)          [1-SD expected move]
  ATM = current close
  0.5 SD OTM call = close + 0.5 × EM
  1.0 SD OTM call = close + 1.0 × EM    (≈ 0.16 delta)
  1.5 SD OTM call = close + 1.5 × EM    (≈ 0.10 delta)
  2.0 SD OTM call = close + 2.0 × EM    (wing/protection)

Example — Nifty at 23,000, VIX=18, DTE=7:
  EM = 23000 × 0.18 × √(7/365) ≈ ±574 pts
  ATM         = 23,000
  0.5-SD OTM  = 23,287 → rounds to 23,300 (Nifty 50pt interval)
  1.0-SD OTM  = 23,574 → rounds to 23,550
  1.5-SD OTM  = 23,861 → rounds to 23,850
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import math

import numpy as np
import pandas as pd
import yaml
from pathlib import Path


# ── Core data classes ─────────────────────────────────────────────────────────

@dataclass
class OptionsLeg:
    """One leg of a multi-leg options strategy."""
    action:      Literal["buy", "sell"]
    option_type: Literal["call", "put"]
    strike:      float
    qty:         int    # lots (always positive; action determines direction)
    label:       str    # e.g. "ATM call", "1.0-SD OTM put wing"


@dataclass
class OptionsSignal:
    """Full options strategy condition signal with leg structure and risk profile."""
    time:          pd.Timestamp
    symbol:        str
    strategy:      str
    category:      Literal["bullish", "bearish", "neutral"]
    condition_met: bool
    legs:          list   # list[OptionsLeg]
    max_profit:    str    # textual description, e.g. "Net credit received"
    max_loss:      str    # e.g. "Spread width − net credit"
    break_even:    str    # e.g. "Short put strike − net credit"
    regime:        str
    adx:           float
    vix:           float | None
    expected_move: float  # ±EM in price points
    reason:        str
    confidence:    float  # 0–1

    # Legacy single-strike fields kept for backward compatibility
    sell_call_strike: float = float("nan")
    sell_put_strike:  float = float("nan")
    buy_call_strike:  float = float("nan")
    buy_put_strike:   float = float("nan")


# ── Shared utilities ──────────────────────────────────────────────────────────

def _expected_move(close: float, vix: float, dte: int) -> float:
    """1-SD expected move: EM = close × (VIX/100) × sqrt(DTE/365)."""
    return close * (vix / 100) * math.sqrt(dte / 365)


def _strike_interval(symbol: str) -> int:
    """NSE options strike interval by symbol:
    BankNifty = 100 pts, Nifty/FinNifty = 50 pts, Stocks = 5 pts (approx)."""
    sym = symbol.upper()
    if "BANKNIFTY" in sym or "NIFTYBANK" in sym:
        return 100
    if "NIFTY" in sym or "FINNIFTY" in sym:
        return 50
    return 5


def _rs(price: float, em: float, sd: float, symbol: str, side: Literal["call", "put"]) -> float:
    """Round a raw strike price (price ± sd×EM) to nearest valid NSE interval."""
    raw = price + sd * em if side == "call" else price - sd * em
    interval = _strike_interval(symbol)
    return round(raw / interval) * interval


def _load_cfg() -> dict:
    try:
        return yaml.safe_load(Path("configs/strategy.yaml").read_text()) or {}
    except Exception:
        return {}


# ── Config container ──────────────────────────────────────────────────────────

class AllOptionsConfig:
    """Single config object for all options strategies, loaded from strategy.yaml."""

    def __init__(self, cfg: dict | None = None) -> None:
        c = cfg or _load_cfg()
        # shared
        self.dte = int(c.get("options_common", {}).get("dte", 7))
        # Bullish
        buy_call  = c.get("buy_call",          {})
        sell_put  = c.get("sell_put",          {})
        bcs       = c.get("bull_call_spread",  {})
        bps       = c.get("bull_put_spread",   {})
        call_back = c.get("call_back_spread",  {})
        call_frt  = c.get("call_front_spread", {})
        # Bearish
        buy_put   = c.get("buy_put",           {})
        sell_call = c.get("sell_call",         {})
        brs       = c.get("bear_put_spread",   {})
        bcar      = c.get("bear_call_spread",  {})
        put_back  = c.get("put_back_spread",   {})
        put_frt   = c.get("put_front_spread",  {})
        # Neutral
        ic        = c.get("iron_condor",        {})
        ss_stg    = c.get("short_strangle",     {})
        sst       = c.get("short_straddle",     {})
        sib       = c.get("short_iron_butterfly",{})
        siw       = c.get("short_iron_wonder",  {})
        lcal      = c.get("long_calendar",      {})
        lcbf      = c.get("long_call_butterfly",{})

        # ── Bullish params ────────────────────────────────────────────────────
        self.buy_call_adx_min   = float(buy_call.get("adx_min",    20.0))
        self.buy_call_vix_max   = float(buy_call.get("vix_max",    20.0))
        self.buy_call_sd        = float(buy_call.get("strike_sd",   0.0))  # ATM

        self.sell_put_adx_max   = float(sell_put.get("adx_max",    30.0))
        self.sell_put_vix_min   = float(sell_put.get("vix_min",    16.0))
        self.sell_put_sd        = float(sell_put.get("strike_sd",   1.0))  # 1-SD OTM

        self.bcs_adx_min        = float(bcs.get("adx_min",         20.0))
        self.bcs_vix_max        = float(bcs.get("vix_max",         25.0))
        self.bcs_long_sd        = float(bcs.get("long_sd",          0.0))  # ATM
        self.bcs_short_sd       = float(bcs.get("short_sd",         1.0))  # 1-SD OTM call

        self.bps_vix_min        = float(bps.get("vix_min",         15.0))
        self.bps_adx_min        = float(bps.get("adx_min",         20.0))
        self.bps_short_sd       = float(bps.get("short_sd",         0.8))
        self.bps_long_sd        = float(bps.get("long_sd",          1.5))

        self.call_back_adx_min  = float(call_back.get("adx_min",   25.0))
        self.call_back_vix_max  = float(call_back.get("vix_max",   18.0))
        self.call_back_sell_sd  = float(call_back.get("sell_sd",    0.5))  # sell 0.5-SD OTM
        self.call_back_buy_sd   = float(call_back.get("buy_sd",     1.0))  # buy 1.0-SD OTM ×2
        self.call_back_ratio    = int(call_back.get("ratio",           2))  # 1 sell : 2 buy

        self.call_frt_adx_min   = float(call_frt.get("adx_min",    15.0))
        self.call_frt_vix_min   = float(call_frt.get("vix_min",    20.0))
        self.call_frt_buy_sd    = float(call_frt.get("buy_sd",      0.0))  # buy ATM
        self.call_frt_sell_sd   = float(call_frt.get("sell_sd",     1.0))  # sell 1-SD OTM ×2

        # ── Bearish params ────────────────────────────────────────────────────
        self.buy_put_adx_min    = float(buy_put.get("adx_min",     20.0))
        self.buy_put_vix_max    = float(buy_put.get("vix_max",     20.0))
        self.buy_put_sd         = float(buy_put.get("strike_sd",    0.0))  # ATM

        self.sell_call_adx_max  = float(sell_call.get("adx_max",   30.0))
        self.sell_call_vix_min  = float(sell_call.get("vix_min",   16.0))
        self.sell_call_sd       = float(sell_call.get("strike_sd",  1.0))  # 1-SD OTM

        self.brs_adx_min        = float(brs.get("adx_min",         20.0))
        self.brs_vix_max        = float(brs.get("vix_max",         25.0))
        self.brs_long_sd        = float(brs.get("long_sd",          0.0))  # ATM put
        self.brs_short_sd       = float(brs.get("short_sd",         1.0))  # 1-SD OTM put

        self.bcar_vix_min       = float(bcar.get("vix_min",        15.0))
        self.bcar_adx_min       = float(bcar.get("adx_min",        20.0))
        self.bcar_sell_sd       = float(bcar.get("sell_sd",         0.8))
        self.bcar_buy_sd        = float(bcar.get("buy_sd",          1.5))

        self.put_back_adx_min   = float(put_back.get("adx_min",    25.0))
        self.put_back_vix_max   = float(put_back.get("vix_max",    18.0))
        self.put_back_sell_sd   = float(put_back.get("sell_sd",     0.5))
        self.put_back_buy_sd    = float(put_back.get("buy_sd",      1.0))
        self.put_back_ratio     = int(put_back.get("ratio",            2))

        self.put_frt_adx_min    = float(put_frt.get("adx_min",     15.0))
        self.put_frt_vix_min    = float(put_frt.get("vix_min",     20.0))
        self.put_frt_buy_sd     = float(put_frt.get("buy_sd",       0.0))
        self.put_frt_sell_sd    = float(put_frt.get("sell_sd",      1.0))

        # ── Neutral params ────────────────────────────────────────────────────
        self.ic_adx_max         = float(ic.get("adx_max",          20.0))
        self.ic_vix_min         = float(ic.get("vix_min",          15.0))
        self.ic_vix_max         = float(ic.get("vix_max",          35.0))
        self.ic_sd_short        = float(ic.get("sd_short",          1.5))
        self.ic_sd_wing         = float(ic.get("sd_wing",           2.0))
        self.ic_dte             = int(ic.get("dte",                    7))

        self.ss_adx_max         = float(ss_stg.get("adx_max",      12.0))
        self.ss_vix_min         = float(ss_stg.get("vix_min",      22.0))
        self.ss_sd              = float(ss_stg.get("strike_sd",     1.5))

        self.sst_adx_max        = float(sst.get("adx_max",         12.0))
        self.sst_vix_min        = float(sst.get("vix_min",         25.0))

        self.sib_adx_max        = float(sib.get("adx_max",         15.0))
        self.sib_vix_min        = float(sib.get("vix_min",         20.0))
        self.sib_wing_sd        = float(sib.get("wing_sd",          1.0))

        self.siw_adx_max        = float(siw.get("adx_max",         20.0))
        self.siw_vix_min        = float(siw.get("vix_min",         18.0))
        self.siw_call_sd        = float(siw.get("call_sd",          1.5))  # call side wing
        self.siw_put_sd         = float(siw.get("put_sd",           1.0))  # put sold closer to ATM
        self.siw_put_wing_sd    = float(siw.get("put_wing_sd",      1.5))

        self.lcal_adx_max       = float(lcal.get("adx_max",        25.0))
        self.lcal_vix_max       = float(lcal.get("vix_max",        20.0))

        self.lcbf_adx_max       = float(lcbf.get("adx_max",        20.0))
        self.lcbf_vix_max       = float(lcbf.get("vix_max",        18.0))
        self.lcbf_wing_sd       = float(lcbf.get("wing_sd",         0.5))


_GLOBAL_CFG: AllOptionsConfig | None = None


def _cfg() -> AllOptionsConfig:
    global _GLOBAL_CFG
    if _GLOBAL_CFG is None:
        _GLOBAL_CFG = AllOptionsConfig()
    return _GLOBAL_CFG


def _reset_cfg() -> None:
    """Call after strategy.yaml changes to force reload."""
    global _GLOBAL_CFG
    _GLOBAL_CFG = None


def _sig(
    symbol: str, strategy: str, category: str, legs: list,
    max_profit: str, max_loss: str, break_even: str,
    regime: str, adx: float, vix: float, em: float, reason: str, confidence: float,
    **legacy_strikes,
) -> OptionsSignal:
    return OptionsSignal(
        time=pd.Timestamp.now(tz="Asia/Kolkata"),
        symbol=symbol, strategy=strategy, category=category,
        condition_met=True, legs=legs,
        max_profit=max_profit, max_loss=max_loss, break_even=break_even,
        regime=regime, adx=adx, vix=vix, expected_move=em, reason=reason,
        confidence=confidence,
        sell_call_strike=legacy_strikes.get("sc", float("nan")),
        sell_put_strike =legacy_strikes.get("sp", float("nan")),
        buy_call_strike =legacy_strikes.get("bc", float("nan")),
        buy_put_strike  =legacy_strikes.get("bp", float("nan")),
    )


def _leg(action, opt_type, strike, qty, label) -> OptionsLeg:
    return OptionsLeg(action=action, option_type=opt_type, strike=strike, qty=qty, label=label)


# ════════════════════════════════════════════════════════════════════════════
# BULLISH STRATEGIES
# ════════════════════════════════════════════════════════════════════════════

class BuyCallStrategy:
    """Long Call — buy ATM call when strong bullish trend + cheap IV.

    Entry: ADX > 20, DI+ > DI−, price above EMA21 and EMA50, VIX < 20 (buy cheap).
    Break-even: Strike + Premium paid.
    Max profit: Unlimited (price keeps rising).
    Max loss: Premium paid (if price below strike at expiry).

    Statistical basis: ~40-50% win rate but unlimited upside makes R:R positive.
    Best setup: ADX > 25 + recent EMA crossover + VIX below 15.
    """
    name = "buy_call"
    category = "bullish"

    def evaluate_bar(self, symbol, f, close, vix=None) -> OptionsSignal | None:
        if vix is None: return None
        c = _cfg()
        adx     = f.get("adx", 0.0)
        di_diff = f.get("di_diff", 0.0)
        dist21  = f.get("dist_ema_21",  0.0)
        dist50  = f.get("dist_ema_50",  0.0)
        if adx < c.buy_call_adx_min: return None
        if di_diff <= 0: return None
        if dist21 < 0 or dist50 < 0: return None  # price must be above EMA21 and EMA50
        if vix > c.buy_call_vix_max: return None   # don't buy expensive options

        em = _expected_move(close, vix, c.dte)
        atm = _rs(close, em, 0.0, symbol, "call")
        legs = [_leg("buy", "call", atm, 1, "ATM call")]

        conf = 0.50
        if adx > 25: conf += 0.08
        if vix < 15: conf += 0.07   # cheap premium
        if f.get("golden_cross", 0) == 1: conf += 0.05
        conf = min(1.0, conf)

        return _sig(symbol, "buy_call", "bullish", legs,
                    "Unlimited (price above strike + premium)",
                    "Premium paid",
                    f"ATM strike ({atm:.0f}) + Premium paid",
                    "TRENDING_UP", adx, vix, em,
                    f"Buy Call: ADX={adx:.1f} DI+>{abs(di_diff):.1f} VIX={vix:.1f} (cheap)",
                    conf, bc=atm)


class SellPutStrategy:
    """Short Put — sell OTM put when bullish + elevated IV for premium.

    Entry: Bullish trend, price above key support, VIX > 16 (sell expensive premium).
    Max profit: Net premium received.
    Max loss: Strike − Premium (large if price crashes).
    Break-even: Short put strike − Premium received.

    Strike: ~1.0 SD OTM (≈0.16 delta) — safe margin above support.
    Best setup: trending up, near support, VIX 16-25, ADX 20-35.
    """
    name = "sell_put"
    category = "bullish"

    def evaluate_bar(self, symbol, f, close, vix=None) -> OptionsSignal | None:
        if vix is None: return None
        c = _cfg()
        adx     = f.get("adx", 0.0)
        di_diff = f.get("di_diff", 0.0)
        dist200 = f.get("dist_ema_200", 0.0)
        if vix < c.sell_put_vix_min: return None
        if adx > c.sell_put_adx_max: return None
        if di_diff <= 0 and dist200 <= 0: return None

        em = _expected_move(close, vix, c.dte)
        sp = _rs(close, em, c.sell_put_sd, symbol, "put")
        legs = [_leg("sell", "put", sp, 1, f"{c.sell_put_sd:.1f}-SD OTM put")]

        conf = 0.52
        if vix >= 20: conf += 0.08
        if adx >= 20: conf += 0.05
        if dist200 > 0: conf += 0.05  # macro bullish
        conf = min(1.0, conf)

        return _sig(symbol, "sell_put", "bullish", legs,
                    "Net premium received",
                    f"Strike ({sp:.0f}) − Premium paid",
                    f"Strike ({sp:.0f}) − Premium received",
                    "TRENDING_UP", adx, vix, em,
                    f"Sell Put: VIX={vix:.1f}>{c.sell_put_vix_min} ADX={adx:.1f} bullish",
                    conf, sp=sp)


class BullCallSpreadStrategy:
    """Bull Call Spread — buy ATM call + sell OTM call. Defined-risk bullish.

    Entry: Moderate bullish trend (ADX 20-35), VIX moderate.
    Max profit: OTM strike − ATM strike − Net debit (achieved if price ≥ short strike).
    Max loss: Net debit paid.
    Break-even: ATM strike + Net debit.

    Strike selection: Buy ATM, sell 1.0-SD OTM call.
    Advantage over Buy Call: lower cost, defined risk; disadvantage: capped upside.
    """
    name = "bull_call_spread"
    category = "bullish"

    def evaluate_bar(self, symbol, f, close, vix=None) -> OptionsSignal | None:
        if vix is None: return None
        c = _cfg()
        adx = f.get("adx", 0.0)
        di_diff = f.get("di_diff", 0.0)
        if adx < c.bcs_adx_min: return None
        if di_diff <= 0: return None
        if vix > c.bcs_vix_max: return None

        em = _expected_move(close, vix, c.dte)
        bc_buy  = _rs(close, em, c.bcs_long_sd,  symbol, "call")   # ATM
        bc_sell = _rs(close, em, c.bcs_short_sd, symbol, "call")   # 1-SD OTM

        legs = [
            _leg("buy",  "call", bc_buy,  1, f"ATM call (long)"),
            _leg("sell", "call", bc_sell, 1, f"1-SD OTM call (short)"),
        ]
        conf = 0.52
        if adx > 25: conf += 0.07
        conf = min(1.0, conf)

        spread = bc_sell - bc_buy
        return _sig(symbol, "bull_call_spread", "bullish", legs,
                    f"Spread width (≈{spread:.0f} pts) − Net debit",
                    "Net debit paid",
                    f"Long strike ({bc_buy:.0f}) + Net debit",
                    "TRENDING_UP", adx, vix, em,
                    f"Bull Call Spread: ADX={adx:.1f} DI+>{abs(di_diff):.1f} VIX={vix:.1f}",
                    conf, bc=bc_buy, sc=bc_sell)


class BullPutSpreadStrategy:
    """Bull Put Spread — sell OTM put + buy further OTM put. Credit received.

    Entry: Bullish with support identified, VIX elevated for premium.
    Max profit: Net credit received (if price stays above short put at expiry).
    Max loss: Spread width − Net credit.
    Break-even: Short put strike − Net credit.

    Strike: Sell 0.8-SD OTM put, Buy 1.5-SD OTM put (protection).
    """
    name = "bull_put_spread"
    category = "bullish"

    def evaluate_bar(self, symbol, f, close, vix=None) -> OptionsSignal | None:
        if vix is None: return None
        c = _cfg()
        adx = f.get("adx", 0.0)
        di_diff = f.get("di_diff", 0.0)
        rsi = f.get("rsi", 50.0)
        if vix < c.bps_vix_min: return None
        if adx < c.bps_adx_min: return None
        if di_diff <= 0 and rsi < 45: return None  # need some bullish signal

        em = _expected_move(close, vix, c.dte)
        sp_sell = _rs(close, em, c.bps_short_sd, symbol, "put")
        sp_buy  = _rs(close, em, c.bps_long_sd,  symbol, "put")

        legs = [
            _leg("sell", "put", sp_sell, 1, f"{c.bps_short_sd:.1f}-SD OTM put (short)"),
            _leg("buy",  "put", sp_buy,  1, f"{c.bps_long_sd:.1f}-SD OTM put (wing)"),
        ]
        conf = 0.54
        if vix >= 20: conf += 0.07
        if adx >= 25: conf += 0.05
        conf = min(1.0, conf)

        spread = sp_sell - sp_buy
        return _sig(symbol, "bull_put_spread", "bullish", legs,
                    "Net credit received",
                    f"Spread width (≈{spread:.0f} pts) − Net credit",
                    f"Short put ({sp_sell:.0f}) − Net credit",
                    "TRENDING_UP", adx, vix, em,
                    f"Bull Put Spread: VIX={vix:.1f} ADX={adx:.1f} DI+={di_diff:.1f}",
                    conf, sp=sp_sell, bp=sp_buy)


class CallBackSpreadStrategy:
    """Call Ratio Back Spread — sell 1 OTM call + buy 2 further OTM calls.

    Entry: Very bullish, expecting large move (breakout, event).
    VIX < 18 — buy cheap OTM calls for the back-spread.
    ADX > 25 — strong directional momentum.

    Max profit: Unlimited above breakeven (large bull move pays).
    Max loss: Long strike − Short strike − Net credit (or + Net debit).
    Break-even (upper): Long strike + (Spread width ± Net premium) / (Ratio−1).

    Ratio: 1:2 (sell 1, buy 2).
    """
    name = "call_back_spread"
    category = "bullish"

    def evaluate_bar(self, symbol, f, close, vix=None) -> OptionsSignal | None:
        if vix is None: return None
        c = _cfg()
        adx = f.get("adx", 0.0)
        di_diff = f.get("di_diff", 0.0)
        if adx < c.call_back_adx_min: return None
        if di_diff <= 0: return None
        if vix > c.call_back_vix_max: return None  # need cheap back options

        em = _expected_move(close, vix, c.dte)
        c_sell = _rs(close, em, c.call_back_sell_sd, symbol, "call")
        c_buy  = _rs(close, em, c.call_back_buy_sd,  symbol, "call")

        legs = [
            _leg("sell", "call", c_sell, 1,              f"{c.call_back_sell_sd:.1f}-SD OTM call (1 lot)"),
            _leg("buy",  "call", c_buy,  c.call_back_ratio, f"{c.call_back_buy_sd:.1f}-SD OTM call ({c.call_back_ratio} lots)"),
        ]
        conf = 0.52
        if adx > 30: conf += 0.08
        if vix < 14: conf += 0.06   # extra cheap back options
        conf = min(1.0, conf)

        spread = c_buy - c_sell
        return _sig(symbol, "call_back_spread", "bullish", legs,
                    f"Unlimited above upper breakeven",
                    f"Spread width (≈{spread:.0f}) − Net credit | limited loss",
                    f"Upper: Long strike ({c_buy:.0f}) + (spread ± premium) / {c.call_back_ratio-1}",
                    "TRENDING_UP", adx, vix, em,
                    f"Call Back Spread 1:{c.call_back_ratio}: ADX={adx:.1f} VIX={vix:.1f} strong bull",
                    conf, bc=c_buy, sc=c_sell)


class CallFrontSpreadStrategy:
    """Call Ratio Front Spread — buy 1 ATM call + sell 2 OTM calls. Net credit.

    Entry: Moderately bullish, don't expect large move. High VIX (sell expensive OTM).
    ADX 15-30, VIX > 20.

    Max profit: At short strike = (short − long) + Net credit (moderate bull move).
    Max loss: Unlimited above 2× short strike (large bull move).
    Break-even upper: 2 × short strike − long strike + Net debit (or − Net credit).

    Risk warning: unlimited loss above upper breakeven — requires management.
    """
    name = "call_front_spread"
    category = "bullish"

    def evaluate_bar(self, symbol, f, close, vix=None) -> OptionsSignal | None:
        if vix is None: return None
        c = _cfg()
        adx = f.get("adx", 0.0)
        di_diff = f.get("di_diff", 0.0)
        if adx < c.call_frt_adx_min: return None
        if di_diff <= 0: return None
        if vix < c.call_frt_vix_min: return None  # need expensive OTM calls to sell

        em = _expected_move(close, vix, c.dte)
        c_buy  = _rs(close, em, c.call_frt_buy_sd,  symbol, "call")  # ATM
        c_sell = _rs(close, em, c.call_frt_sell_sd, symbol, "call")  # 1-SD OTM ×2

        legs = [
            _leg("buy",  "call", c_buy,  1, "ATM call (1 lot)"),
            _leg("sell", "call", c_sell, 2, f"1-SD OTM call (2 lots)"),
        ]
        conf = 0.52
        if vix > 25: conf += 0.07
        conf = min(1.0, conf)

        spread = c_sell - c_buy
        return _sig(symbol, "call_front_spread", "bullish", legs,
                    f"Spread width (≈{spread:.0f}) + Net credit (at short strike)",
                    "Unlimited above upper breakeven ⚠️ manage risk",
                    f"Upper: 2×{c_sell:.0f} − {c_buy:.0f} ± Net premium",
                    "TRENDING_UP", adx, vix, em,
                    f"Call Front Spread 1:2: VIX={vix:.1f} ADX={adx:.1f} moderate bull",
                    conf, bc=c_buy, sc=c_sell)


# ════════════════════════════════════════════════════════════════════════════
# BEARISH STRATEGIES
# ════════════════════════════════════════════════════════════════════════════

class BuyPutStrategy:
    """Long Put — buy ATM put when strong bearish trend + cheap IV.

    Entry: ADX > 20, DI− > DI+, price below EMA21 and EMA50, VIX < 20.
    Max profit: Strike − Premium (if index falls to zero, theoretical).
    Max loss: Premium paid.
    Break-even: ATM strike − Premium paid.
    """
    name = "buy_put"
    category = "bearish"

    def evaluate_bar(self, symbol, f, close, vix=None) -> OptionsSignal | None:
        if vix is None: return None
        c = _cfg()
        adx = f.get("adx", 0.0)
        di_diff = f.get("di_diff", 0.0)
        dist21 = f.get("dist_ema_21", 0.0)
        dist50 = f.get("dist_ema_50", 0.0)
        if adx < c.buy_put_adx_min: return None
        if di_diff >= 0: return None
        if dist21 > 0 or dist50 > 0: return None  # must be below EMAs
        if vix > c.buy_put_vix_max: return None

        em = _expected_move(close, vix, c.dte)
        atm = _rs(close, em, 0.0, symbol, "put")
        legs = [_leg("buy", "put", atm, 1, "ATM put")]

        conf = 0.50
        if adx > 25: conf += 0.08
        if vix < 15: conf += 0.07
        conf = min(1.0, conf)

        return _sig(symbol, "buy_put", "bearish", legs,
                    f"Strike ({atm:.0f}) − Premium (large if deep fall)",
                    "Premium paid",
                    f"ATM strike ({atm:.0f}) − Premium paid",
                    "TRENDING_DOWN", adx, vix, em,
                    f"Buy Put: ADX={adx:.1f} DI->{abs(di_diff):.1f} VIX={vix:.1f}",
                    conf, bp=atm)


class SellCallStrategy:
    """Short Call — sell OTM call when bearish + elevated IV.

    Entry: Bearish/neutral, price below resistance, VIX > 16.
    Max profit: Net premium received.
    Max loss: Unlimited above breakeven (manage with hard stop or hedge).
    Break-even: Short call strike + Premium received.

    Strike: ~1.0 SD OTM (≈0.16 delta) above current price.
    """
    name = "sell_call"
    category = "bearish"

    def evaluate_bar(self, symbol, f, close, vix=None) -> OptionsSignal | None:
        if vix is None: return None
        c = _cfg()
        adx = f.get("adx", 0.0)
        di_diff = f.get("di_diff", 0.0)
        dist200 = f.get("dist_ema_200", 0.0)
        if vix < c.sell_call_vix_min: return None
        if adx > c.sell_call_adx_max: return None
        if di_diff >= 0 and dist200 >= 0: return None  # need bearish signal

        em = _expected_move(close, vix, c.dte)
        sc = _rs(close, em, c.sell_call_sd, symbol, "call")
        legs = [_leg("sell", "call", sc, 1, f"{c.sell_call_sd:.1f}-SD OTM call")]

        conf = 0.52
        if vix >= 20: conf += 0.08
        if dist200 < 0: conf += 0.05  # macro bearish
        conf = min(1.0, conf)

        return _sig(symbol, "sell_call", "bearish", legs,
                    "Net premium received",
                    "Unlimited above breakeven ⚠️ manage risk",
                    f"Short call ({sc:.0f}) + Premium received",
                    "TRENDING_DOWN", adx, vix, em,
                    f"Sell Call: VIX={vix:.1f} ADX={adx:.1f} bearish",
                    conf, sc=sc)


class BearPutSpreadStrategy:
    """Bear Put Spread — buy ATM put + sell OTM put. Defined-risk bearish.

    Entry: Moderate bearish trend, moderate VIX.
    Max profit: Spread width − Net debit (achieved if price ≤ short put at expiry).
    Max loss: Net debit paid.
    Break-even: Long put strike − Net debit.
    """
    name = "bear_put_spread"
    category = "bearish"

    def evaluate_bar(self, symbol, f, close, vix=None) -> OptionsSignal | None:
        if vix is None: return None
        c = _cfg()
        adx = f.get("adx", 0.0)
        di_diff = f.get("di_diff", 0.0)
        if adx < c.brs_adx_min: return None
        if di_diff >= 0: return None
        if vix > c.brs_vix_max: return None

        em = _expected_move(close, vix, c.dte)
        bp_buy  = _rs(close, em, c.brs_long_sd,  symbol, "put")   # ATM
        bp_sell = _rs(close, em, c.brs_short_sd, symbol, "put")   # 1-SD OTM

        legs = [
            _leg("buy",  "put", bp_buy,  1, "ATM put (long)"),
            _leg("sell", "put", bp_sell, 1, "1-SD OTM put (short)"),
        ]
        conf = 0.52
        if adx > 25: conf += 0.07
        conf = min(1.0, conf)

        spread = bp_buy - bp_sell
        return _sig(symbol, "bear_put_spread", "bearish", legs,
                    f"Spread width (≈{spread:.0f} pts) − Net debit",
                    "Net debit paid",
                    f"Long put ({bp_buy:.0f}) − Net debit",
                    "TRENDING_DOWN", adx, vix, em,
                    f"Bear Put Spread: ADX={adx:.1f} DI->{abs(di_diff):.1f} VIX={vix:.1f}",
                    conf, bp=bp_buy, sp=bp_sell)


class BearCallSpreadStrategy:
    """Bear Call Spread — sell OTM call + buy further OTM call. Net credit.

    Entry: Bearish trend, VIX elevated for premium.
    Max profit: Net credit received (if price stays below short call at expiry).
    Max loss: Spread width − Net credit.
    Break-even: Short call strike + Net credit.
    """
    name = "bear_call_spread"
    category = "bearish"

    def evaluate_bar(self, symbol, f, close, vix=None) -> OptionsSignal | None:
        if vix is None: return None
        c = _cfg()
        adx = f.get("adx", 0.0)
        di_diff = f.get("di_diff", 0.0)
        rsi = f.get("rsi", 50.0)
        if vix < c.bcar_vix_min: return None
        if adx < c.bcar_adx_min: return None
        if di_diff >= 0 and rsi > 55: return None  # need bearish signal

        em = _expected_move(close, vix, c.dte)
        sc_sell = _rs(close, em, c.bcar_sell_sd, symbol, "call")
        sc_buy  = _rs(close, em, c.bcar_buy_sd,  symbol, "call")

        legs = [
            _leg("sell", "call", sc_sell, 1, f"{c.bcar_sell_sd:.1f}-SD OTM call (short)"),
            _leg("buy",  "call", sc_buy,  1, f"{c.bcar_buy_sd:.1f}-SD OTM call (wing)"),
        ]
        conf = 0.54
        if vix >= 20: conf += 0.07
        if adx >= 25: conf += 0.05
        conf = min(1.0, conf)

        spread = sc_buy - sc_sell
        return _sig(symbol, "bear_call_spread", "bearish", legs,
                    "Net credit received",
                    f"Spread width (≈{spread:.0f} pts) − Net credit",
                    f"Short call ({sc_sell:.0f}) + Net credit",
                    "TRENDING_DOWN", adx, vix, em,
                    f"Bear Call Spread: VIX={vix:.1f} ADX={adx:.1f} DI-={di_diff:.1f}",
                    conf, sc=sc_sell, bc=sc_buy)


class PutBackSpreadStrategy:
    """Put Ratio Back Spread — sell 1 OTM put + buy 2 further OTM puts.

    Entry: Very bearish, expecting large move down (event/breakdown).
    VIX < 18 — buy cheap OTM puts.
    ADX > 25 — strong bearish momentum.

    Max profit: Unlimited below lower breakeven.
    Max loss: Spread width ± Net premium (at long put strike at expiry).
    """
    name = "put_back_spread"
    category = "bearish"

    def evaluate_bar(self, symbol, f, close, vix=None) -> OptionsSignal | None:
        if vix is None: return None
        c = _cfg()
        adx = f.get("adx", 0.0)
        di_diff = f.get("di_diff", 0.0)
        if adx < c.put_back_adx_min: return None
        if di_diff >= 0: return None
        if vix > c.put_back_vix_max: return None

        em = _expected_move(close, vix, c.dte)
        p_sell = _rs(close, em, c.put_back_sell_sd, symbol, "put")
        p_buy  = _rs(close, em, c.put_back_buy_sd,  symbol, "put")

        legs = [
            _leg("sell", "put", p_sell, 1,               f"{c.put_back_sell_sd:.1f}-SD OTM put (1 lot)"),
            _leg("buy",  "put", p_buy,  c.put_back_ratio, f"{c.put_back_buy_sd:.1f}-SD OTM put ({c.put_back_ratio} lots)"),
        ]
        conf = 0.52
        if adx > 30: conf += 0.08
        if vix < 14: conf += 0.06
        conf = min(1.0, conf)

        spread = p_sell - p_buy
        return _sig(symbol, "put_back_spread", "bearish", legs,
                    "Unlimited below lower breakeven",
                    f"Spread (≈{spread:.0f}) ± Net premium | limited",
                    f"Lower: Long put ({p_buy:.0f}) − (spread ± premium) / {c.put_back_ratio-1}",
                    "TRENDING_DOWN", adx, vix, em,
                    f"Put Back Spread 1:{c.put_back_ratio}: ADX={adx:.1f} VIX={vix:.1f} strong bear",
                    conf, bp=p_buy, sp=p_sell)


class PutFrontSpreadStrategy:
    """Put Ratio Front Spread — buy 1 ATM put + sell 2 OTM puts. Net credit.

    Entry: Moderately bearish, don't expect large crash. VIX > 20.
    Max profit: At short put strike (moderate fall).
    Max loss: Unlimited below lower breakeven — manage strictly.
    """
    name = "put_front_spread"
    category = "bearish"

    def evaluate_bar(self, symbol, f, close, vix=None) -> OptionsSignal | None:
        if vix is None: return None
        c = _cfg()
        adx = f.get("adx", 0.0)
        di_diff = f.get("di_diff", 0.0)
        if adx < c.put_frt_adx_min: return None
        if di_diff >= 0: return None
        if vix < c.put_frt_vix_min: return None

        em = _expected_move(close, vix, c.dte)
        p_buy  = _rs(close, em, c.put_frt_buy_sd,  symbol, "put")  # ATM
        p_sell = _rs(close, em, c.put_frt_sell_sd, symbol, "put")  # 1-SD OTM ×2

        legs = [
            _leg("buy",  "put", p_buy,  1, "ATM put (1 lot)"),
            _leg("sell", "put", p_sell, 2, "1-SD OTM put (2 lots)"),
        ]
        conf = 0.52
        if vix > 25: conf += 0.07
        conf = min(1.0, conf)

        spread = p_buy - p_sell
        return _sig(symbol, "put_front_spread", "bearish", legs,
                    f"Spread (≈{spread:.0f}) + Net credit (at short strike)",
                    "Unlimited below lower breakeven ⚠️ manage risk",
                    f"Lower: 2×{p_sell:.0f} − {p_buy:.0f} ± Net premium",
                    "TRENDING_DOWN", adx, vix, em,
                    f"Put Front Spread 1:2: VIX={vix:.1f} ADX={adx:.1f} moderate bear",
                    conf, bp=p_buy, sp=p_sell)


# ════════════════════════════════════════════════════════════════════════════
# NEUTRAL STRATEGIES
# ════════════════════════════════════════════════════════════════════════════

class IronCondorStrategy:
    """Iron Condor — sell OTM call + OTM put, buy wings further out. Neutral range play.

    Entry: RANGING (ADX < 20), VIX 15-35 (need elevated IV for worthwhile premium).
    Max profit: Net credit (price stays between short strikes).
    Max loss: (Wing width − Net credit) per side.
    Break-even: Short call + Net credit / Short put − Net credit.
    """
    name = "iron_condor"
    category = "neutral"

    def evaluate_bar(self, symbol, f, close, vix=None) -> OptionsSignal | None:
        if vix is None: return None
        c = _cfg()
        adx = f.get("adx", 0.0)
        if adx >= c.ic_adx_max: return None
        if not (c.ic_vix_min <= vix <= c.ic_vix_max): return None

        em = _expected_move(close, vix, c.ic_dte)
        sc = _rs(close, em, c.ic_sd_short, symbol, "call")
        sp = _rs(close, em, c.ic_sd_short, symbol, "put")
        wc = _rs(close, em, c.ic_sd_wing,  symbol, "call")
        wp = _rs(close, em, c.ic_sd_wing,  symbol, "put")

        legs = [
            _leg("sell", "call", sc, 1, f"{c.ic_sd_short:.1f}-SD OTM call (short)"),
            _leg("buy",  "call", wc, 1, f"{c.ic_sd_wing:.1f}-SD OTM call (wing)"),
            _leg("sell", "put",  sp, 1, f"{c.ic_sd_short:.1f}-SD OTM put (short)"),
            _leg("buy",  "put",  wp, 1, f"{c.ic_sd_wing:.1f}-SD OTM put (wing)"),
        ]
        conf = 0.50
        if adx < 15:            conf += 0.10
        if 18 <= vix <= 28:     conf += 0.10
        if f.get("bb_squeeze", 0) == 1: conf += 0.05
        conf = min(1.0, conf)

        return _sig(symbol, "iron_condor", "neutral", legs,
                    "Net credit received",
                    f"Wing width (call: {wc-sc:.0f}, put: {sp-wp:.0f}) − Net credit per side",
                    f"Above: {sc:.0f} + Credit | Below: {sp:.0f} − Credit",
                    "RANGING", adx, vix, em,
                    f"Iron Condor: ADX={adx:.1f} VIX={vix:.1f} EM=±{em:.0f}",
                    conf, sc=sc, sp=sp, bc=wc, bp=wp)


class ShortIronButterflyStrategy:
    """Short Iron Butterfly — sell ATM call + ATM put, buy OTM wings. Neutral tight range.

    Entry: ADX < 15 (very flat), VIX > 20 (sell expensive premium near ATM).
    Max profit: Net credit (price pins at ATM strike at expiry).
    Max loss: Wing width − Net credit (price moves far from ATM).
    Break-even: ATM ± Net credit.

    More aggressive than Iron Condor — higher premium but narrower profit zone.
    """
    name = "short_iron_butterfly"
    category = "neutral"

    def evaluate_bar(self, symbol, f, close, vix=None) -> OptionsSignal | None:
        if vix is None: return None
        c = _cfg()
        adx = f.get("adx", 0.0)
        if adx >= c.sib_adx_max: return None
        if vix < c.sib_vix_min: return None

        em = _expected_move(close, vix, c.dte)
        atm  = _rs(close, em, 0.0,          symbol, "call")   # ATM
        wc   = _rs(close, em, c.sib_wing_sd, symbol, "call")  # OTM call wing
        wp   = _rs(close, em, c.sib_wing_sd, symbol, "put")   # OTM put wing

        legs = [
            _leg("sell", "call", atm, 1, "ATM call (short straddle leg)"),
            _leg("sell", "put",  atm, 1, "ATM put (short straddle leg)"),
            _leg("buy",  "call", wc,  1, f"{c.sib_wing_sd:.1f}-SD OTM call wing"),
            _leg("buy",  "put",  wp,  1, f"{c.sib_wing_sd:.1f}-SD OTM put wing"),
        ]
        conf = 0.52
        if adx < 10:    conf += 0.08
        if vix > 25:    conf += 0.08
        conf = min(1.0, conf)

        return _sig(symbol, "short_iron_butterfly", "neutral", legs,
                    "Net credit (price pins at ATM)",
                    f"Wing width (≈{wc-atm:.0f}) − Net credit",
                    f"{atm:.0f} ± Net credit",
                    "RANGING", adx, vix, em,
                    f"Short Iron Butterfly: ADX={adx:.1f} VIX={vix:.1f} ATM={atm:.0f}",
                    conf, sc=atm, sp=atm, bc=wc, bp=wp)


class ShortIronWonderStrategy:
    """Short Iron Wonder — asymmetric iron butterfly with skewed put/call wings.

    Structure:
      Sell ATM call + buy OTM call wing at 1.5 SD  [symmetric call side]
      Sell 0.5-SD OTM put + buy 1.5-SD OTM put wing  [put sold closer to ATM]

    This is bearish-skewed neutral: the put is sold closer to ATM than the call,
    collecting more put premium. Use when you expect mild downward drift but low volatility.

    Entry: ADX < 20, VIX > 18.
    Max profit: Net credit (price stays between short strikes).
    Max loss: Put side limited, call side limited, both defined.
    """
    name = "short_iron_wonder"
    category = "neutral"

    def evaluate_bar(self, symbol, f, close, vix=None) -> OptionsSignal | None:
        if vix is None: return None
        c = _cfg()
        adx = f.get("adx", 0.0)
        if adx >= c.siw_adx_max: return None
        if vix < c.siw_vix_min: return None

        em = _expected_move(close, vix, c.dte)
        atm   = _rs(close, em, 0.0,             symbol, "call")  # sell call at ATM
        wc    = _rs(close, em, c.siw_call_sd,    symbol, "call")  # buy call wing
        sp    = _rs(close, em, c.siw_put_sd,     symbol, "put")   # sell put closer (0.5 SD)
        wp    = _rs(close, em, c.siw_put_wing_sd, symbol, "put")  # buy put wing

        legs = [
            _leg("sell", "call", atm, 1, "ATM call (short)"),
            _leg("buy",  "call", wc,  1, f"{c.siw_call_sd:.1f}-SD OTM call wing"),
            _leg("sell", "put",  sp,  1, f"{c.siw_put_sd:.1f}-SD OTM put (short, closer ATM)"),
            _leg("buy",  "put",  wp,  1, f"{c.siw_put_wing_sd:.1f}-SD OTM put wing"),
        ]
        conf = 0.52
        if adx < 15: conf += 0.07
        if vix > 22: conf += 0.06
        conf = min(1.0, conf)

        return _sig(symbol, "short_iron_wonder", "neutral", legs,
                    "Net credit (price stays between short strikes)",
                    f"Call side: {wc-atm:.0f} − credit | Put side: {sp-wp:.0f} − credit",
                    f"Call side: {atm:.0f} + credit | Put side: {sp:.0f} − credit",
                    "RANGING", adx, vix, em,
                    f"Short Iron Wonder: ADX={adx:.1f} VIX={vix:.1f} skewed-put",
                    conf, sc=atm, sp=sp, bc=wc, bp=wp)


class ShortStraddleStrategy:
    """Short Straddle — sell ATM call + ATM put. Ultra-neutral, unlimited risk.

    Entry: ADX < 12 (extreme flat), VIX > 25 (sell very expensive ATM premium).
    Requires: post-event IV spike that is expected to collapse rapidly.
    Max profit: Net credit (price stays at ATM strike).
    Max loss: Unlimited in either direction.
    Break-even: ATM ± Net credit.

    WARNING: Use only when conviction is very high. Manage with delta hedge or tight stops.
    """
    name = "short_straddle"
    category = "neutral"

    def evaluate_bar(self, symbol, f, close, vix=None) -> OptionsSignal | None:
        if vix is None: return None
        c = _cfg()
        adx = f.get("adx", 0.0)
        if adx >= c.sst_adx_max: return None
        if vix < c.sst_vix_min: return None

        em = _expected_move(close, vix, c.dte)
        atm = _rs(close, em, 0.0, symbol, "call")

        legs = [
            _leg("sell", "call", atm, 1, "ATM call"),
            _leg("sell", "put",  atm, 1, "ATM put"),
        ]
        conf = 0.50
        if adx < 8:  conf += 0.10
        if vix > 30: conf += 0.10
        conf = min(1.0, conf)

        return _sig(symbol, "short_straddle", "neutral", legs,
                    "Net credit (price stays at ATM)",
                    "Unlimited in either direction ⚠️ use with delta hedge",
                    f"{atm:.0f} ± Net credit",
                    "RANGING", adx, vix, em,
                    f"Short Straddle: ADX={adx:.1f}<{c.sst_adx_max} VIX={vix:.1f}>{c.sst_vix_min}",
                    conf, sc=atm, sp=atm)


class ShortStrangleStrategy:
    """Short Strangle — sell OTM call + OTM put. Wider range than straddle.

    Entry: ADX < 12, VIX > 22 (post-event IV spike).
    Max profit: Net credit (price stays between short strikes).
    Max loss: Unlimited beyond breakeven — requires management.
    """
    name = "short_strangle"
    category = "neutral"

    def evaluate_bar(self, symbol, f, close, vix=None) -> OptionsSignal | None:
        if vix is None: return None
        c = _cfg()
        adx = f.get("adx", 0.0)
        if adx >= c.ss_adx_max: return None
        if vix < c.ss_vix_min: return None

        em = _expected_move(close, vix, c.dte)
        sc = _rs(close, em, c.ss_sd, symbol, "call")
        sp = _rs(close, em, c.ss_sd, symbol, "put")

        legs = [
            _leg("sell", "call", sc, 1, f"{c.ss_sd:.1f}-SD OTM call"),
            _leg("sell", "put",  sp, 1, f"{c.ss_sd:.1f}-SD OTM put"),
        ]
        conf = 0.50
        if adx < 8:  conf += 0.10
        if vix > 28: conf += 0.08
        conf = min(1.0, conf)

        return _sig(symbol, "short_strangle", "neutral", legs,
                    "Net credit (price stays between short strikes)",
                    "Unlimited beyond breakeven ⚠️",
                    f"Call: {sc:.0f} + credit | Put: {sp:.0f} − credit",
                    "RANGING", adx, vix, em,
                    f"Short Strangle: ADX={adx:.1f} VIX={vix:.1f}",
                    conf, sc=sc, sp=sp)


class LongCallButterflyStrategy:
    """Long Call Butterfly — buy lower call + sell 2 ATM calls + buy upper call.

    Entry: ADX < 20 (neutral/ranging), low VIX (cheap to enter), price expected to pin near ATM.
    Max profit: Wing width − Net debit (at ATM/middle strike at expiry).
    Max loss: Net debit paid (if price moves far from ATM in either direction).
    Break-even: Lower strike + Net debit (lower) / Upper strike − Net debit (upper).

    Wing size: 0.5 × EM (half the expected move — keeps profit zone realistic).
    Best in: Low VIX, range-bound market, specific expiry target level.
    """
    name = "long_call_butterfly"
    category = "neutral"

    def evaluate_bar(self, symbol, f, close, vix=None) -> OptionsSignal | None:
        if vix is None: return None
        c = _cfg()
        adx = f.get("adx", 0.0)
        if adx >= c.lcbf_adx_max: return None
        if vix > c.lcbf_vix_max: return None   # buy when IV is cheap

        em   = _expected_move(close, vix, c.dte)
        atm  = _rs(close, em, 0.0,             symbol, "call")
        low  = _rs(close, em, -c.lcbf_wing_sd, symbol, "call")   # lower wing (ITM call)
        high = _rs(close, em,  c.lcbf_wing_sd, symbol, "call")   # upper wing (OTM call)

        legs = [
            _leg("buy",  "call", low,  1, f"Lower call ({c.lcbf_wing_sd:.1f}-SD ITM)"),
            _leg("sell", "call", atm,  2, "ATM call (×2)"),
            _leg("buy",  "call", high, 1, f"Upper call ({c.lcbf_wing_sd:.1f}-SD OTM)"),
        ]
        conf = 0.52
        if adx < 12: conf += 0.07
        if vix < 14: conf += 0.06  # cheap entry
        conf = min(1.0, conf)

        wing = high - atm
        return _sig(symbol, "long_call_butterfly", "neutral", legs,
                    f"Wing width (≈{wing:.0f}) − Net debit (price pins at {atm:.0f})",
                    "Net debit paid (price moves far from ATM)",
                    f"Lower: {low:.0f} + debit | Upper: {high:.0f} − debit",
                    "RANGING", adx, vix, em,
                    f"Long Call Butterfly: ADX={adx:.1f} VIX={vix:.1f} target={atm:.0f}",
                    conf, bc=atm)


class LongCalendarStrategy:
    """Long Calendar Spread — sell near-month + buy far-month at same ATM strike.

    Entry: ADX < 25, low near-month VIX (cheap to sell), expecting IV expansion later.
    Max profit: Near-month expires worthless, far-month retains value + IV expansion.
    Max loss: Net debit (if near-month IV spikes unexpectedly or price moves far).
    Break-even: Near ATM at near-month expiry (price should stay near strike).

    Direction: Calendar works with both calls and puts; using calls here.
    DTE: near-month = 7 days, far-month = 30 days (standard weekly/monthly).
    """
    name = "long_calendar"
    category = "neutral"

    def evaluate_bar(self, symbol, f, close, vix=None) -> OptionsSignal | None:
        if vix is None: return None
        c = _cfg()
        adx = f.get("adx", 0.0)
        if adx >= c.lcal_adx_max: return None
        if vix > c.lcal_vix_max: return None  # don't buy far-month when VIX high (expensive)

        em  = _expected_move(close, vix, c.dte)
        atm = _rs(close, em, 0.0, symbol, "call")

        legs = [
            _leg("sell", "call", atm, 1, f"Near-month ATM call (DTE≈{c.dte})"),
            _leg("buy",  "call", atm, 1, f"Far-month ATM call (DTE≈{c.dte*4})"),
        ]
        conf = 0.52
        if adx < 15: conf += 0.06
        if vix < 14: conf += 0.07  # cheap far-month
        conf = min(1.0, conf)

        return _sig(symbol, "long_calendar", "neutral", legs,
                    f"Near-month expires worthless + far-month value + IV expansion",
                    "Net debit (both legs lose on large price move or near-month IV spike)",
                    f"Near ATM ({atm:.0f}) at near-month expiry",
                    "RANGING", adx, vix, em,
                    f"Long Calendar: ADX={adx:.1f} VIX={vix:.1f} ATM={atm:.0f}",
                    conf, bc=atm, sc=atm)


# ── Registry of all strategies ────────────────────────────────────────────────

ALL_OPTIONS_STRATEGIES: list = [
    # Bullish
    BuyCallStrategy(),
    SellPutStrategy(),
    BullCallSpreadStrategy(),
    BullPutSpreadStrategy(),
    CallBackSpreadStrategy(),
    CallFrontSpreadStrategy(),
    # Bearish
    BuyPutStrategy(),
    SellCallStrategy(),
    BearPutSpreadStrategy(),
    BearCallSpreadStrategy(),
    PutBackSpreadStrategy(),
    PutFrontSpreadStrategy(),
    # Neutral
    IronCondorStrategy(),
    ShortIronButterflyStrategy(),
    ShortIronWonderStrategy(),
    ShortStraddleStrategy(),
    ShortStrangleStrategy(),
    LongCallButterflyStrategy(),
    LongCalendarStrategy(),
]

# Name → instance mapping for quick lookup
OPTIONS_REGISTRY: dict[str, object] = {s.name: s for s in ALL_OPTIONS_STRATEGIES}
