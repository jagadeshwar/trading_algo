"""Options Executor — resolves strategy signals to real chain quotes and takes action.

Flow:
  OptionsSignal (EM-estimated strikes)
      ↓  OptionChainSnapshot (real LTP, bid/ask, IV, Greeks)
  ExecutableLeg  (real premium + Fyers symbol + Greeks)
      ↓
  PayoffResult   (exact max profit, max loss, break-evens at all price levels)
      ↓
  Order placement: paper trade → data/paper_options_trades.jsonl
                   live trade  → Fyers API

Strike matching:
  Each strategy leg specifies a theoretical strike. The executor finds the NEAREST
  available strike in the chain and returns the real bid/ask/LTP.
  For limit orders: mid-price = (bid + ask) / 2.
  For market orders: ask for buys, bid for sells (conservative).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from loguru import logger

from production.data.option_chain import OptionChainSnapshot, OptionQuote, bs_greeks
from production.strategy.options_strategies import OptionsLeg, OptionsSignal

IST             = ZoneInfo("Asia/Kolkata")
PAPER_LOG       = Path("data/paper_options_trades.jsonl")
POSITIONS_FILE  = Path("data/options_positions.json")


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ExecutableLeg:
    """OptionsLeg enriched with real chain data."""
    # From OptionsLeg
    action:       Literal["buy", "sell"]
    option_type:  Literal["call", "put"]
    strike:       float       # requested theoretical strike
    qty:          int
    label:        str
    # From chain
    actual_strike: float      # nearest available strike
    fyers_symbol:  str        # e.g. NSE:NIFTY2460613CE23000
    ltp:           float      # last traded price
    bid:           float
    ask:           float
    mid:           float      # execution price for limit orders
    iv:            float      # implied volatility %
    delta:         float
    theta:         float      # per day
    vega:          float      # per 1% IV change
    oi:            int        # open interest
    lot_size:      int = 25   # NSE lot size (Nifty=25, BankNifty=30)

    @property
    def premium_per_lot(self) -> float:
        return self.mid * self.lot_size

    @property
    def total_premium(self) -> float:
        return self.mid * self.lot_size * self.qty

    @property
    def signed_premium(self) -> float:
        """Positive = credit received, negative = debit paid."""
        return self.total_premium if self.action == "sell" else -self.total_premium


@dataclass
class PayoffResult:
    """P&L profile at expiry for a multi-leg options strategy."""
    net_premium:   float      # positive = net credit, negative = net debit
    max_profit:    float      # maximum possible profit (inf for unlimited)
    max_loss:      float      # maximum possible loss (inf for unlimited)
    break_even_up: float      # upper break-even price
    break_even_dn: float      # lower break-even price (0 if no lower BEP)
    prob_profit:   float      # probability of profit (from delta approximation)
    price_points:  list[float]  # range of underlying prices for payoff chart
    payoffs:       list[float]  # P&L at each price point (in ₹ per lot)


@dataclass
class ExecutionResult:
    """Result of placing an order."""
    success:    bool
    order_id:   str
    symbol:     str
    action:     str
    strike:     float
    qty:        int
    price:      float
    mode:       Literal["paper", "live"]
    message:    str = ""


# ── Lot sizes by symbol ────────────────────────────────────────────────────────

_LOT_SIZES: dict[str, int] = {
    "NIFTY":      25,
    "BANKNIFTY":  30,
    "FINNIFTY":   40,
    "MIDCPNIFTY": 75,
}


def _lot_size(fyers_symbol: str) -> int:
    for name, size in _LOT_SIZES.items():
        if name in fyers_symbol.upper():
            return size
    return 1  # equity options: 1 per share (lot size varies, default 1)


# ── Core executor ─────────────────────────────────────────────────────────────

class OptionsExecutor:
    """Resolve strategy signals to executable legs and manage orders."""

    def __init__(self, fyers_client=None) -> None:
        self._fyers = fyers_client
        PAPER_LOG.parent.mkdir(parents=True, exist_ok=True)

    # ── Resolve legs ──────────────────────────────────────────────────────────

    def resolve_legs(
        self,
        signal: OptionsSignal,
        chain: OptionChainSnapshot,
    ) -> list[ExecutableLeg]:
        """Match each signal leg to the nearest available chain strike."""
        executable = []
        for leg in signal.legs:
            quote = self._find_best_quote(leg, chain)
            if quote is None:
                logger.warning(
                    "No chain quote found for {} {} strike={:.0f}",
                    leg.option_type, leg.action, leg.strike
                )
                continue

            lot = _lot_size(quote.fyers_symbol)
            executable.append(ExecutableLeg(
                action=leg.action,
                option_type=leg.option_type,
                strike=leg.strike,
                qty=leg.qty,
                label=leg.label,
                actual_strike=quote.strike,
                fyers_symbol=quote.fyers_symbol,
                ltp=quote.ltp,
                bid=quote.bid,
                ask=quote.ask,
                mid=quote.mid,
                iv=quote.iv,
                delta=quote.greeks.delta,
                theta=quote.greeks.theta,
                vega=quote.greeks.vega,
                oi=quote.oi,
                lot_size=lot,
            ))
        return executable

    def _find_best_quote(
        self, leg: OptionsLeg, chain: OptionChainSnapshot
    ) -> OptionQuote | None:
        """Find the chain quote whose strike is nearest to leg.strike."""
        opt_type = "CE" if leg.option_type == "call" else "PE"
        candidates = [q for q in chain.quotes if q.option_type == opt_type and q.ltp > 0]
        if not candidates:
            return None
        return min(candidates, key=lambda q: abs(q.strike - leg.strike))

    # ── Payoff computation ────────────────────────────────────────────────────

    def compute_payoff(
        self,
        legs: list[ExecutableLeg],
        underlying: float,
        price_range_pct: float = 0.05,
        n_points: int = 100,
    ) -> PayoffResult:
        """Compute P&L profile at expiry across a price range around current spot."""
        if not legs:
            return PayoffResult(0, 0, 0, underlying, underlying, 0, [], [])

        net_premium = sum(l.signed_premium for l in legs)
        lo = underlying * (1 - price_range_pct)
        hi = underlying * (1 + price_range_pct)
        prices = np.linspace(lo, hi, n_points)

        payoffs = []
        for price in prices:
            pnl = net_premium  # start with premium already received/paid
            for l in legs:
                intrinsic = max(0.0, price - l.actual_strike) if l.option_type == "call" \
                            else max(0.0, l.actual_strike - price)
                lot_pnl = intrinsic * l.lot_size * l.qty
                pnl += lot_pnl if l.action == "buy" else -lot_pnl
            payoffs.append(pnl)

        payoffs_arr = np.array(payoffs)
        max_profit  = float(payoffs_arr.max())
        max_loss    = float(payoffs_arr.min())

        # Break-even: first price from left and right where payoff crosses 0
        bep_dn = bep_up = 0.0
        for i, (p, pnl) in enumerate(zip(prices, payoffs)):
            if i > 0 and payoffs[i - 1] * pnl < 0:  # sign change = BEP crossing
                bep_up = bep_dn if bep_dn else float(p)
                if not bep_dn:
                    bep_dn = float(p)

        # Prob of profit: approximate from delta sum
        if net_premium > 0:  # credit strategy — profit if price stays in range
            prob_profit = abs(sum(l.delta * (1 if l.action == "sell" else -1) for l in legs))
        else:
            prob_profit = 1 - abs(sum(l.delta for l in legs if l.action == "buy"))
        prob_profit = max(0.0, min(1.0, prob_profit))

        return PayoffResult(
            net_premium=round(net_premium, 2),
            max_profit=round(max_profit, 2),
            max_loss=round(max_loss, 2),
            break_even_up=round(bep_up, 0),
            break_even_dn=round(bep_dn, 0),
            prob_profit=round(prob_profit, 3),
            price_points=prices.tolist(),
            payoffs=payoffs_arr.tolist(),
        )

    # ── Paper trading ─────────────────────────────────────────────────────────

    def place_paper(
        self,
        legs: list[ExecutableLeg],
        signal: OptionsSignal,
        payoff: PayoffResult,
    ) -> list[ExecutionResult]:
        """Log paper trade to JSONL. No real orders placed."""
        results = []
        for leg in legs:
            price = leg.ask if leg.action == "buy" else leg.bid
            record = {
                "time":          pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
                "mode":          "paper",
                "strategy":      signal.strategy,
                "category":      signal.category,
                "action":        leg.action,
                "option_type":   leg.option_type,
                "requested_strike": leg.strike,
                "actual_strike": leg.actual_strike,
                "fyers_symbol":  leg.fyers_symbol,
                "qty":           leg.qty,
                "lots":          leg.qty,
                "lot_size":      leg.lot_size,
                "price":         price,
                "iv":            leg.iv,
                "delta":         leg.delta,
                "oi":            leg.oi,
                "net_premium":   payoff.net_premium,
                "max_profit":    payoff.max_profit,
                "max_loss":      payoff.max_loss,
                "break_even_up": payoff.break_even_up,
                "break_even_dn": payoff.break_even_dn,
                "underlying":    signal.expected_move,
                "vix":           signal.vix,
                "regime":        signal.regime,
                "status":        "open",
            }
            with PAPER_LOG.open("a") as f:
                f.write(json.dumps(record) + "\n")

            results.append(ExecutionResult(
                success=True,
                order_id=f"PAPER_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}",
                symbol=leg.fyers_symbol,
                action=leg.action,
                strike=leg.actual_strike,
                qty=leg.qty,
                price=price,
                mode="paper",
                message=f"{leg.action.upper()} {leg.option_type.upper()} {leg.actual_strike:.0f} @ {price:.2f}",
            ))
            logger.info("PAPER OPTION | {} {} {} {} @ {:.2f} | strategy={}",
                        leg.action.upper(), leg.option_type.upper(),
                        leg.actual_strike, leg.fyers_symbol, price, signal.strategy)
        return results

    # ── Live trading ──────────────────────────────────────────────────────────

    def place_live(
        self,
        legs: list[ExecutableLeg],
        signal: OptionsSignal,
        order_type: Literal["limit", "market"] = "limit",
        product_type: str = "INTRADAY",
    ) -> list[ExecutionResult]:
        """Place real orders via Fyers API. Requires authenticated fyers_client."""
        if self._fyers is None:
            raise RuntimeError("No Fyers client — cannot place live orders")

        results = []
        for leg in legs:
            # Conservative pricing: buy at ask, sell at bid (avoid partial fills)
            price = leg.ask if leg.action == "buy" else leg.bid
            fyers_side = 1 if leg.action == "buy" else -1

            order_data = {
                "symbol":       leg.fyers_symbol,
                "qty":          leg.qty * leg.lot_size,
                "type":         2 if order_type == "market" else 1,  # 1=Limit, 2=Market
                "side":         fyers_side,
                "productType":  product_type,
                "limitPrice":   round(price, 1) if order_type == "limit" else 0,
                "stopPrice":    0,
                "validity":     "DAY",
                "disclosedQty": 0,
                "offlineOrder": False,
            }

            try:
                resp = self._fyers.place_order(data=order_data)
                ok   = resp.get("s") == "ok"
                oid  = resp.get("id", "")
                msg  = resp.get("message", "")
                logger.info("LIVE ORDER {} | {} | id={} ok={} msg={}",
                            leg.fyers_symbol, leg.action.upper(), oid, ok, msg)
                results.append(ExecutionResult(
                    success=ok, order_id=oid, symbol=leg.fyers_symbol,
                    action=leg.action, strike=leg.actual_strike,
                    qty=leg.qty, price=price, mode="live", message=msg,
                ))
            except Exception as e:
                logger.error("Fyers place_order failed: {}", e)
                results.append(ExecutionResult(
                    success=False, order_id="", symbol=leg.fyers_symbol,
                    action=leg.action, strike=leg.actual_strike,
                    qty=leg.qty, price=price, mode="live", message=str(e),
                ))
        return results

    # ── Convenience: full flow ────────────────────────────────────────────────

    def execute_signal(
        self,
        signal: OptionsSignal,
        chain: OptionChainSnapshot,
        mode: Literal["paper", "live"] = "paper",
        order_type: Literal["limit", "market"] = "limit",
    ) -> dict:
        """Full flow: signal → resolve legs → payoff → place order."""
        legs   = self.resolve_legs(signal, chain)
        if not legs:
            return {"success": False, "message": "No legs resolved from chain"}

        payoff = self.compute_payoff(legs, chain.underlying)

        if mode == "paper":
            orders = self.place_paper(legs, signal, payoff)
        else:
            orders = self.place_live(legs, signal, order_type)

        return {
            "success":     all(o.success for o in orders),
            "legs":        legs,
            "payoff":      payoff,
            "orders":      orders,
            "net_premium": payoff.net_premium,
            "max_profit":  payoff.max_profit,
            "max_loss":    payoff.max_loss,
        }

    # ── Open positions ────────────────────────────────────────────────────────

    def open_positions(self) -> pd.DataFrame:
        """Load all open paper option positions from JSONL."""
        if not PAPER_LOG.exists():
            return pd.DataFrame()
        rows = [json.loads(l) for l in PAPER_LOG.read_text().splitlines()
                if l.strip() and json.loads(l).get("status") == "open"]
        return pd.DataFrame(rows) if rows else pd.DataFrame()
