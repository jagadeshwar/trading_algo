"""Option Chain — live NSE option chain with Greeks, Max Pain, PCR, and market context.

Data sources (in priority order):
  1. Fyers API       — fyers.optionchain(). Primary source. Authenticated, reliable,
                        returns oi/volume/bid/ask/ltp/symbol. IV is back-solved via
                        Black-Scholes bisection since Fyers does not return IV directly.
  2. NSE direct API  — Fallback only. NSE now uses Cloudflare protection (returns 403
                        on most non-browser requests). Used as secondary when Fyers
                        is unavailable (no active session).

Provides per-strike:
  LTP · Bid/Ask · OI · OI Change · Volume · IV · Greeks (Δ Γ Θ V) · Fyers symbol

Computed:
  Max Pain · Put-Call Ratio · ATM IV · Expected Move (±SD table)

Market context (enrichment):
  India VIX · BankNifty vs Nifty performance · Top Nifty50 movers / RS rankings

Usage:
    fetcher = OptionChainFetcher(fyers_client=fyers)   # fyers_client optional
    snap    = fetcher.fetch("NSE:NIFTY50-INDEX")       # OptionChainSnapshot
    snap    = fetcher.fetch("NSE:NIFTYBANK-INDEX", expiry_index=1)

    ctx = fetcher.fetch_market_context()               # VIX + Nifty50 movers
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from loguru import logger

IST = ZoneInfo("Asia/Kolkata")
RISK_FREE = 0.065   # India 10-year gilt ≈ 6.5%

# ── NSE symbol mapping ─────────────────────────────────────────────────────────

NSE_CHAIN_SYMBOL: dict[str, str] = {
    "NSE:NIFTY50-INDEX":     "NIFTY",
    "NSE:NIFTYBANK-INDEX":   "BANKNIFTY",
    "NSE:FINNIFTY-INDEX":    "FINNIFTY",
    "NSE:MIDCPNIFTY-INDEX":  "MIDCPNIFTY",
}

# Full Nifty 50 constituent list
NIFTY50_SYMBOLS = [
    "NSE:RELIANCE-EQ",   "NSE:TCS-EQ",        "NSE:HDFCBANK-EQ",   "NSE:BHARTIARTL-EQ",
    "NSE:ICICIBANK-EQ",  "NSE:INFY-EQ",        "NSE:SBIN-EQ",       "NSE:HINDUNILVR-EQ",
    "NSE:ITC-EQ",        "NSE:LT-EQ",          "NSE:KOTAKBANK-EQ",  "NSE:BAJFINANCE-EQ",
    "NSE:AXISBANK-EQ",   "NSE:ASIANPAINT-EQ",  "NSE:MARUTI-EQ",     "NSE:HCLTECH-EQ",
    "NSE:SUNPHARMA-EQ",  "NSE:TITAN-EQ",       "NSE:WIPRO-EQ",      "NSE:ULTRACEMCO-EQ",
    "NSE:NTPC-EQ",       "NSE:ONGC-EQ",        "NSE:POWERGRID-EQ",  "NSE:ADANIENT-EQ",
    "NSE:ADANIPORTS-EQ", "NSE:BAJAJFINSV-EQ",  "NSE:BPCL-EQ",       "NSE:COALINDIA-EQ",
    "NSE:DIVISLAB-EQ",   "NSE:DRREDDY-EQ",     "NSE:EICHERMOT-EQ",  "NSE:GRASIM-EQ",
    "NSE:HDFCLIFE-EQ",   "NSE:HEROMOTOCO-EQ",  "NSE:HINDALCO-EQ",   "NSE:INDUSINDBK-EQ",
    "NSE:JSWSTEEL-EQ",   "NSE:NESTLEIND-EQ",   "NSE:SBILIFE-EQ",    "NSE:SHRIRAMFIN-EQ",
    "NSE:TATAMOTORS-EQ", "NSE:TATASTEEL-EQ",   "NSE:TECHM-EQ",      "NSE:TRENT-EQ",
    "NSE:CIPLA-EQ",      "NSE:APOLLOHOSP-EQ",  "NSE:BAJAJ-AUTO-EQ", "NSE:BEL-EQ",
    "NSE:MM-EQ",         "NSE:BRITANNIA-EQ",
]

BANK_NIFTY_CONSTITUENTS = [
    "NSE:HDFCBANK-EQ", "NSE:ICICIBANK-EQ", "NSE:AXISBANK-EQ", "NSE:SBIN-EQ",
    "NSE:KOTAKBANK-EQ", "NSE:INDUSINDBK-EQ", "NSE:BANDHANBNK-EQ", "NSE:FEDERALBNK-EQ",
    "NSE:IDFCFIRSTB-EQ", "NSE:AUBANK-EQ", "NSE:PNB-EQ", "NSE:BANKBARODA-EQ",
]


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class OptionGreeks:
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0   # per day
    vega:  float = 0.0   # per 1% IV change


@dataclass
class OptionQuote:
    """Single option strike quote — one row of the chain."""
    strike:      float
    expiry:      str
    option_type: str      # "CE" or "PE"
    fyers_symbol: str     # e.g. NSE:NIFTY2460613CE23000
    ltp:         float
    bid:         float
    ask:         float
    mid:         float    # (bid + ask) / 2
    iv:          float    # implied volatility %
    oi:          int
    oi_change:   int
    volume:      int
    greeks:      OptionGreeks = field(default_factory=OptionGreeks)
    underlying:  float = 0.0


@dataclass
class OptionChainSnapshot:
    """Full option chain for one symbol at one moment in time."""
    symbol:          str
    nse_symbol:      str          # e.g. "NIFTY"
    expiry:          str
    all_expiries:    list[str]
    underlying:      float
    vix:             float        # India VIX at time of fetch (0 if unavailable)
    timestamp:       pd.Timestamp
    quotes:          list[OptionQuote]   # all strikes, both CE and PE
    # Computed
    max_pain:        float = 0.0
    pcr:             float = 0.0   # OI-based PCR
    atm_iv:          float = 0.0   # IV at nearest ATM strike
    atm_strike:      float = 0.0
    em_1sd:          float = 0.0   # 1-SD expected move in points
    source:          str   = "nse" # "nse" | "fyers"

    @property
    def calls(self) -> list[OptionQuote]:
        return [q for q in self.quotes if q.option_type == "CE"]

    @property
    def puts(self) -> list[OptionQuote]:
        return [q for q in self.quotes if q.option_type == "PE"]

    def to_df(self) -> pd.DataFrame:
        """Return chain as a pivot DataFrame: strikes as rows, CE/PE as column groups."""
        rows = []
        for q in self.quotes:
            rows.append({
                "strike": q.strike,
                "type":   q.option_type,
                "ltp":    q.ltp,
                "bid":    q.bid,
                "ask":    q.ask,
                "iv":     q.iv,
                "oi":     q.oi,
                "oi_chg": q.oi_change,
                "vol":    q.volume,
                "delta":  q.greeks.delta,
                "theta":  q.greeks.theta,
                "vega":   q.greeks.vega,
                "symbol": q.fyers_symbol,
            })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        pivot = df.pivot(index="strike", columns="type")
        pivot.columns = [f"{col[1]}_{col[0]}" for col in pivot.columns]
        return pivot.reset_index().sort_values("strike")


@dataclass
class MarketContext:
    """Real-time market context: VIX, index levels, top Nifty50 movers."""
    timestamp:          pd.Timestamp
    india_vix:          float
    nifty_last:         float
    banknifty_last:     float
    nifty_change_pct:   float
    banknifty_change_pct: float
    banknifty_vs_nifty: float    # BankNifty / Nifty ratio change (sector performance)
    em_nifty_daily:     float    # daily EM for Nifty based on VIX
    em_nifty_weekly:    float    # weekly EM
    em_banknifty_daily: float
    em_banknifty_weekly: float
    nifty50_quotes:     dict     # symbol → {ltp, change_pct, rs_vs_nifty}
    top_gainers:        list     # top 5 Nifty50 stocks
    top_losers:         list     # bottom 5 Nifty50 stocks


# ── Black-Scholes Greeks (pure Python, no scipy) ──────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via error function (math.erf — stdlib)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_greeks(
    S: float,   # spot price
    K: float,   # strike
    T: float,   # time to expiry in years
    sigma: float,  # IV as decimal (e.g. 0.18 for 18%)
    option_type: str = "CE",
    r: float = RISK_FREE,
) -> OptionGreeks:
    """Black-Scholes Greeks for European options (NSE index options are European)."""
    if T <= 1e-6 or sigma <= 1e-6 or S <= 0 or K <= 0:
        return OptionGreeks()

    try:
        sqrt_T  = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        pdf_d1  = _norm_pdf(d1)
        cdf_d1  = _norm_cdf(d1)
        cdf_d2  = _norm_cdf(d2)
        cdf_nd1 = _norm_cdf(-d1)
        cdf_nd2 = _norm_cdf(-d2)

        # Delta
        delta = cdf_d1 if option_type == "CE" else cdf_d1 - 1.0

        # Gamma (same for call and put)
        gamma = pdf_d1 / (S * sigma * sqrt_T)

        # Theta (per calendar day)
        common_theta = -(S * pdf_d1 * sigma) / (2.0 * sqrt_T)
        if option_type == "CE":
            theta = (common_theta - r * K * math.exp(-r * T) * cdf_d2) / 365.0
        else:
            theta = (common_theta + r * K * math.exp(-r * T) * cdf_nd2) / 365.0

        # Vega (per 1% change in IV)
        vega = S * pdf_d1 * sqrt_T / 100.0

        return OptionGreeks(
            delta=round(delta, 4),
            gamma=round(gamma, 6),
            theta=round(theta, 2),
            vega=round(vega, 2),
        )
    except Exception:
        return OptionGreeks()


# ── Implied Volatility (back-solve from price via bisection) ──────────────────

def compute_iv(
    price: float,
    S: float,     # spot
    K: float,     # strike
    T: float,     # time to expiry (years)
    option_type: str = "CE",
    r: float = RISK_FREE,
) -> float:
    """Back-solve IV% from option market price using bisection (50 iterations).

    Returns IV as a percentage (e.g. 18.5 for 18.5% IV).
    Returns 0.0 if the price is below intrinsic value or T <= 0.
    """
    if T <= 1e-6 or price <= 0:
        return 0.0

    def _bs_price(sigma: float) -> float:
        if sigma <= 0:
            return 0.0
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        if option_type == "CE":
            return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        else:
            return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

    lo, hi = 0.001, 10.0   # 0.1% → 1000% IV search range
    for _ in range(60):
        mid = (lo + hi) / 2
        p   = _bs_price(mid)
        diff = p - price
        if abs(diff) < 0.05:
            return round(mid * 100, 2)
        if diff < 0:
            lo = mid
        else:
            hi = mid
    return round(mid * 100, 2)


# ── Auto-create Fyers client from saved token ──────────────────────────────────

def _make_fyers_client():
    """Create a FyersModel instance from the saved daily token file.
    Returns None if the token is missing or expired.
    """
    try:
        import os, json
        from pathlib import Path as _P
        from dotenv import load_dotenv
        load_dotenv()
        token_path = _P("fyers_token.txt")
        if not token_path.exists():
            return None
        data     = json.loads(token_path.read_text())
        from datetime import date
        if data.get("date") != date.today().isoformat():
            logger.debug("Fyers token expired — re-login required")
            return None
        access_token = data["access_token"]
        app_id       = os.environ.get("FYERS_APP_ID", "")
        if not app_id:
            return None
        from fyers_apiv3 import fyersModel
        return fyersModel.FyersModel(
            client_id=app_id, is_async=False,
            token=access_token, log_path=""
        )
    except Exception as e:
        logger.debug("Could not build Fyers client: {}", e)
        return None


# ── Max Pain computation ───────────────────────────────────────────────────────

def compute_max_pain(quotes: list[OptionQuote]) -> float:
    """Max Pain = strike where aggregate option sellers' loss is minimised.

    For each possible expiry price (= each strike), compute the total loss for:
      call writers: sum of max(0, strike_price - expiry_price) × OI for all calls where
                    strike_price < expiry_price  [ITM calls = loss for call sellers]
      put writers:  sum of max(0, expiry_price - strike_price) × OI for all puts where
                    strike_price > expiry_price  [ITM puts = loss for put sellers]
    The expiry price with minimum total writer loss = Max Pain.
    """
    calls = {q.strike: q.oi for q in quotes if q.option_type == "CE"}
    puts  = {q.strike: q.oi for q in quotes if q.option_type == "PE"}
    strikes = sorted(set(calls) | set(puts))

    if not strikes:
        return 0.0

    min_loss  = float("inf")
    max_pain  = strikes[0]

    for exp_price in strikes:
        call_loss = sum(max(0.0, exp_price - k) * oi for k, oi in calls.items())
        put_loss  = sum(max(0.0, k - exp_price) * oi for k, oi in puts.items())
        total     = call_loss + put_loss
        if total < min_loss:
            min_loss = total
            max_pain = exp_price

    return max_pain


def compute_pcr(quotes: list[OptionQuote]) -> float:
    """Put-Call Ratio = total put OI / total call OI."""
    call_oi = sum(q.oi for q in quotes if q.option_type == "CE")
    put_oi  = sum(q.oi for q in quotes if q.option_type == "PE")
    return round(put_oi / call_oi, 3) if call_oi > 0 else 0.0


# ── NSE direct fetcher ────────────────────────────────────────────────────────

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         "https://www.nseindia.com/option-chain",
    "X-Requested-With": "XMLHttpRequest",
    "Sec-Fetch-Dest":  "empty",
    "Sec-Fetch-Mode":  "cors",
    "Sec-Fetch-Site":  "same-origin",
    "Connection":      "keep-alive",
}

_nse_session: requests.Session | None = None
_nse_session_ts: float = 0.0
_NSE_SESSION_TTL = 300   # re-establish cookies every 5 min


def _get_nse_session() -> requests.Session:
    """Return a requests Session with valid NSE cookies (refreshed every 5 min)."""
    global _nse_session, _nse_session_ts
    if _nse_session is None or (time.time() - _nse_session_ts) > _NSE_SESSION_TTL:
        s = requests.Session()
        try:
            # Two warm-up requests to establish cookies
            s.get("https://www.nseindia.com", headers=_NSE_HEADERS, timeout=10)
            time.sleep(0.5)
            s.get("https://www.nseindia.com/option-chain", headers=_NSE_HEADERS, timeout=10)
        except Exception as e:
            logger.debug("NSE session warm-up error: {}", e)
        _nse_session = s
        _nse_session_ts = time.time()
    return _nse_session


def _nse_fetch_raw(nse_symbol: str, timeout: int = 10) -> dict:
    """Fetch raw JSON from NSE option chain API."""
    s = _get_nse_session()
    is_index = nse_symbol in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY")
    endpoint = (
        f"https://www.nseindia.com/api/option-chain-indices?symbol={nse_symbol}"
        if is_index else
        f"https://www.nseindia.com/api/option-chain-equities?symbol={nse_symbol}"
    )
    resp = s.get(endpoint, headers=_NSE_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _days_to_expiry(expiry_str: str) -> float:
    """Return calendar days until expiry (for Greeks computation)."""
    try:
        exp = pd.to_datetime(expiry_str, format="%d-%b-%Y", errors="coerce")
        if pd.isna(exp):
            exp = pd.to_datetime(expiry_str, errors="coerce")
        if pd.isna(exp):
            return 7.0
        now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
        return max((exp - now).days + (exp - now).seconds / 86400, 0.5)
    except Exception:
        return 7.0


def _parse_nse_chain(raw: dict, expiry_index: int = 0) -> OptionChainSnapshot | None:
    """Parse NSE API JSON → OptionChainSnapshot with Greeks computed."""
    records = raw.get("records", {})
    filtered = raw.get("filtered", {})

    expiries    = records.get("expiryDates", [])
    all_data    = records.get("data", [])
    underlying  = float(records.get("underlyingValue", 0))
    timestamp   = pd.Timestamp.now(tz="Asia/Kolkata")

    if not expiries or not all_data:
        return None

    expiry_idx  = min(expiry_index, len(expiries) - 1)
    expiry      = expiries[expiry_idx]
    T           = _days_to_expiry(expiry) / 365.0

    quotes: list[OptionQuote] = []
    for item in all_data:
        if item.get("expiryDate") != expiry:
            continue
        strike = float(item.get("strikePrice", 0))

        for opt_type, key in (("CE", "CE"), ("PE", "PE")):
            leg = item.get(key)
            if not leg:
                continue
            ltp    = float(leg.get("lastPrice",  0) or 0)
            bid    = float(leg.get("bidprice",   0) or 0)
            ask    = float(leg.get("askPrice",   0) or 0)
            iv_pct = float(leg.get("impliedVolatility", 0) or 0)
            oi     = int(leg.get("openInterest",       0) or 0)
            oi_chg = int(leg.get("changeinOpenInterest", 0) or 0)
            vol    = int(leg.get("totalTradedVolume",   0) or 0)
            mid    = (bid + ask) / 2.0 if bid and ask else ltp

            sigma  = iv_pct / 100.0
            greeks = bs_greeks(underlying, strike, T, sigma, opt_type)

            quotes.append(OptionQuote(
                strike=strike, expiry=expiry, option_type=opt_type,
                fyers_symbol="",  # filled after
                ltp=ltp, bid=bid, ask=ask, mid=mid,
                iv=iv_pct, oi=oi, oi_change=oi_chg, volume=vol,
                greeks=greeks, underlying=underlying,
            ))

    if not quotes:
        return None

    # ATM strike and ATM IV
    atm_strike = min((q.strike for q in quotes), key=lambda k: abs(k - underlying))
    atm_calls  = [q for q in quotes if q.option_type == "CE" and q.strike == atm_strike]
    atm_iv     = atm_calls[0].iv if atm_calls else 0.0

    # Expected move from ATM IV (DTE in days)
    dte_days = _days_to_expiry(expiry)
    em_1sd   = underlying * (atm_iv / 100) * math.sqrt(dte_days / 365) if atm_iv > 0 else 0.0

    snap = OptionChainSnapshot(
        symbol=f"NSE:{[k for k,v in NSE_CHAIN_SYMBOL.items() if v == 'NIFTY'][0].split(':')[1]}",
        nse_symbol="NIFTY",
        expiry=expiry,
        all_expiries=expiries,
        underlying=underlying,
        vix=0.0,
        timestamp=timestamp,
        quotes=quotes,
        max_pain=compute_max_pain(quotes),
        pcr=compute_pcr(quotes),
        atm_iv=atm_iv,
        atm_strike=atm_strike,
        em_1sd=em_1sd,
        source="nse",
    )
    return snap


# ── Fyers fallback ────────────────────────────────────────────────────────────

def _parse_fyers_chain(
    response: dict, fyers_symbol: str, expiry_index: int = 0
) -> OptionChainSnapshot | None:
    """Parse Fyers optionchain response → OptionChainSnapshot.

    Actual Fyers API structure (verified 2026-06):
      response['data']['optionsChain']  — flat list: underlying row (strike=-1) + all options
      response['data']['expiryData']    — list of {date, expiry (unix timestamp), expiry_flag}
      response['data']['indiavixData']  — VIX quote dict with 'ltp'
      Each option item: {ask, bid, ltp, oi, oich, oichp, prev_oi, volume,
                         option_type (CE/PE), strike_price, symbol, fyToken, ...}
      NOTE: Fyers does NOT return IV — it is back-solved via Black-Scholes bisection.
    """
    data        = response.get("data", {})
    expiry_data = data.get("expiryData", [])
    all_chain   = data.get("optionsChain", [])
    vix_data    = data.get("indiavixData", {})

    if not all_chain:
        return None

    expiry_index   = min(expiry_index, max(0, len(expiry_data) - 1))
    all_expiry_dates = [e.get("date", "") for e in expiry_data]

    # The optionchain is returned for the single expiry requested via timestamp.
    # Extract underlying price from the row where strike_price == -1
    underlying = 0.0
    for item in all_chain:
        if item.get("strike_price", 0) == -1:
            underlying = float(item.get("ltp", 0) or item.get("fp", 0) or 0)
            break

    # Target expiry date string (e.g. "02-06-2026")
    expiry = all_expiry_dates[expiry_index] if all_expiry_dates else ""
    T      = _days_to_expiry(expiry) / 365.0

    # VIX from indiavixData
    vix = float(vix_data.get("ltp", 0) or 0)

    quotes: list[OptionQuote] = []
    for item in all_chain:
        opt_type = item.get("option_type", "")
        if opt_type not in ("CE", "PE"):
            continue
        strike = float(item.get("strike_price", 0) or 0)
        if strike <= 0:
            continue

        ltp    = float(item.get("ltp",    0) or 0)
        bid    = float(item.get("bid",    0) or 0)
        ask    = float(item.get("ask",    0) or 0)
        oi     = int(item.get("oi",       0) or 0)
        oi_chg = int(item.get("oich",     0) or 0)
        vol    = int(item.get("volume",   0) or 0)
        fyrsym = item.get("symbol",       "")
        mid    = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else ltp

        # Back-solve IV from market price (Fyers doesn't return IV)
        iv_pct = compute_iv(ltp, underlying, strike, T, opt_type) if underlying > 0 and T > 0 and ltp > 0 else 0.0
        sigma  = iv_pct / 100.0
        greeks = bs_greeks(underlying, strike, T, sigma, opt_type)

        quotes.append(OptionQuote(
            strike=strike, expiry=expiry, option_type=opt_type,
            fyers_symbol=fyrsym,
            ltp=ltp, bid=bid, ask=ask, mid=mid,
            iv=iv_pct, oi=oi, oi_change=oi_chg, volume=vol,
            greeks=greeks, underlying=underlying,
        ))

    if not quotes:
        return None

    atm_strike = min((q.strike for q in quotes), key=lambda k: abs(k - underlying))
    atm_calls  = [q for q in quotes if q.option_type == "CE" and q.strike == atm_strike]
    atm_iv     = atm_calls[0].iv if atm_calls else (vix or 0.0)
    dte_days   = _days_to_expiry(expiry)
    em_1sd     = underlying * (atm_iv / 100) * math.sqrt(dte_days / 365) if atm_iv > 0 else 0.0

    return OptionChainSnapshot(
        symbol=fyers_symbol,
        nse_symbol=NSE_CHAIN_SYMBOL.get(fyers_symbol, fyers_symbol.split(":")[1]),
        expiry=expiry,
        all_expiries=all_expiry_dates,
        underlying=underlying,
        vix=vix,
        timestamp=pd.Timestamp.now(tz="Asia/Kolkata"),
        quotes=quotes,
        max_pain=compute_max_pain(quotes),
        pcr=compute_pcr(quotes),
        atm_iv=atm_iv,
        atm_strike=atm_strike,
        em_1sd=em_1sd,
        source="fyers",
    )


# ── Main fetcher class ─────────────────────────────────────────────────────────

class OptionChainFetcher:
    """Fetch live option chain + market context.

    Priority: Fyers API (primary, authenticated) → NSE direct (fallback).
    If no fyers_client is passed, auto-creates one from the saved daily token.
    """

    _CACHE: dict[str, tuple[float, OptionChainSnapshot]] = {}
    _CACHE_TTL = 30  # seconds

    def __init__(self, fyers_client=None) -> None:
        # Use provided client or auto-create from saved token
        self._fyers = fyers_client or _make_fyers_client()
        if self._fyers:
            logger.debug("OptionChainFetcher: Fyers client ready")
        else:
            logger.warning("OptionChainFetcher: no Fyers client — will attempt NSE direct (may be blocked)")

    def fetch(
        self,
        symbol: str = "NSE:NIFTY50-INDEX",
        expiry_index: int = 0,
        strike_count: int = 30,
        force_refresh: bool = False,
    ) -> OptionChainSnapshot:
        """Return OptionChainSnapshot, using cache if recent enough."""
        cache_key = f"{symbol}_{expiry_index}"
        now = time.time()
        if not force_refresh and cache_key in self._CACHE:
            ts, cached = self._CACHE[cache_key]
            if now - ts < self._CACHE_TTL:
                return cached

        # Fyers first (primary), then NSE
        snap = self._fetch_fyers(symbol, expiry_index, strike_count) or \
               self._fetch_nse(symbol, expiry_index)

        if snap is None:
            raise RuntimeError(
                f"Could not fetch option chain for {symbol}. "
                "Fyers session may be expired — run: python auth.py"
            )

        self._CACHE[cache_key] = (now, snap)
        logger.info("Option chain: {} expiry={} strikes={} VIX={:.2f} source={}",
                    symbol, snap.expiry, len(snap.quotes) // 2, snap.vix, snap.source)
        return snap

    def _fetch_nse(self, symbol: str, expiry_index: int) -> OptionChainSnapshot | None:
        nse_sym = NSE_CHAIN_SYMBOL.get(symbol)
        if nse_sym is None:
            # Equity: extract symbol from "NSE:RELIANCE-EQ" → "RELIANCE"
            nse_sym = symbol.split(":")[1].replace("-EQ", "").replace("-INDEX", "")
        try:
            raw  = _nse_fetch_raw(nse_sym)
            snap = _parse_nse_chain(raw, expiry_index)
            if snap:
                snap.symbol     = symbol
                snap.nse_symbol = nse_sym
                # Fill Fyers symbols post-parse (NSE doesn't provide them)
                self._fill_fyers_symbols(snap)
                return snap
        except Exception as e:
            logger.warning("NSE option chain failed ({}): {}", symbol, e)
        return None

    def _fetch_fyers(
        self, symbol: str, expiry_index: int, strike_count: int
    ) -> OptionChainSnapshot | None:
        if self._fyers is None:
            return None
        try:
            # First call with no timestamp to get the expiry list
            resp0 = self._fyers.optionchain(data={
                "symbol": symbol, "strikecount": 2, "timestamp": "",
            })
            if resp0.get("s") != "ok":
                logger.warning("Fyers optionchain error: {}", resp0.get("message", resp0))
                return None

            expiry_data = resp0.get("data", {}).get("expiryData", [])
            if not expiry_data:
                return None

            # For expiry_index > 0, re-fetch with the specific expiry timestamp
            if expiry_index > 0 and expiry_index < len(expiry_data):
                ts = expiry_data[expiry_index].get("expiry", "")
                resp = self._fyers.optionchain(data={
                    "symbol": symbol, "strikecount": strike_count, "timestamp": ts,
                })
                if resp.get("s") != "ok":
                    resp = resp0  # fallback to first expiry
            else:
                # Fetch nearest expiry with full strike count
                resp = self._fyers.optionchain(data={
                    "symbol": symbol, "strikecount": strike_count, "timestamp": "",
                })
                if resp.get("s") != "ok":
                    return None

            return _parse_fyers_chain(resp, symbol, expiry_index)
        except Exception as e:
            logger.warning("Fyers option chain failed ({}): {}", symbol, e)
        return None

    def _fill_fyers_symbols(self, snap: OptionChainSnapshot) -> None:
        """Construct Fyers option symbols from expiry + strike (NSE doesn't provide them)."""
        try:
            exp = pd.to_datetime(snap.expiry, format="%d-%b-%Y", errors="coerce")
            if pd.isna(exp):
                return
            yy  = exp.strftime("%y")        # 2-digit year
            mmm = exp.strftime("%b").upper() # 3-letter month (for monthly)
            dd  = exp.strftime("%d")         # 2-digit day
            mm  = exp.strftime("%m")         # 2-digit month
            # Fyers weekly format: NSE:NIFTY{YY}{DD}{MM}{STRIKE}{CE/PE}
            # Fyers monthly format: NSE:NIFTY{YY}{MMM}{STRIKE}{CE/PE}
            # We use the date format (works for both)
            base = snap.nse_symbol  # NIFTY / BANKNIFTY
            for q in snap.quotes:
                st = int(q.strike)
                q.fyers_symbol = f"NSE:{base}{yy}{mm}{dd}{st}{q.option_type}"
        except Exception as e:
            logger.debug("Fyers symbol fill failed: {}", e)

    # ── Market context ────────────────────────────────────────────────────────

    def fetch_market_context(self, vix_value: float = 0.0) -> MarketContext | None:
        """Fetch live Nifty50 quotes and compute market context."""
        if self._fyers is None:
            logger.debug("No Fyers client — market context unavailable")
            return None

        try:
            # Batch quotes for indices + all Nifty50 stocks
            index_syms  = "NSE:NIFTY50-INDEX,NSE:NIFTYBANK-INDEX,NSE:INDIAVIX-INDEX"
            stock_syms  = ",".join(NIFTY50_SYMBOLS)
            all_syms    = f"{index_syms},{stock_syms}"

            resp = self._fyers.quotes({"symbols": all_syms})
            if resp.get("s") != "ok":
                return None

            quotes_raw = {
                q["n"]: q for q in resp.get("d", [])
                if isinstance(q, dict) and "n" in q
            }

            def _get(sym: str, key: str, default: float = 0.0) -> float:
                q = quotes_raw.get(sym, {})
                v = q.get("v", {})
                return float(v.get(key, default) or default)

            nifty_last    = _get("NSE:NIFTY50-INDEX",   "lp")
            nifty_prev    = _get("NSE:NIFTY50-INDEX",   "prev_close_price")
            banknifty_last= _get("NSE:NIFTYBANK-INDEX", "lp")
            banknifty_prev= _get("NSE:NIFTYBANK-INDEX", "prev_close_price")
            vix_live      = _get("NSE:INDIAVIX-INDEX",  "lp") or vix_value

            n_chg  = (nifty_last    - nifty_prev)    / nifty_prev    * 100 if nifty_prev    else 0.0
            bn_chg = (banknifty_last - banknifty_prev) / banknifty_prev * 100 if banknifty_prev else 0.0

            # BankNifty outperformance vs Nifty
            bn_vs_n = bn_chg - n_chg

            # Expected moves (EM = S × VIX/100 × √(DTE/365))
            em_n_daily  = nifty_last    * (vix_live / 100) * math.sqrt(1 / 365) if vix_live else 0
            em_n_weekly = nifty_last    * (vix_live / 100) * math.sqrt(7 / 365) if vix_live else 0
            em_bn_daily = banknifty_last * (vix_live / 100) * math.sqrt(1 / 365) if vix_live else 0
            em_bn_weekly= banknifty_last * (vix_live / 100) * math.sqrt(7 / 365) if vix_live else 0

            # Nifty50 individual stock performance
            stock_perf = {}
            for sym in NIFTY50_SYMBOLS:
                lp   = _get(sym, "lp")
                prev = _get(sym, "prev_close_price")
                chg  = (lp - prev) / prev * 100 if prev else 0.0
                rs   = chg - n_chg  # RS vs Nifty (excess return)
                stock_perf[sym] = {"ltp": lp, "change_pct": round(chg, 2), "rs_vs_nifty": round(rs, 2)}

            sorted_stocks = sorted(stock_perf.items(), key=lambda x: x[1]["change_pct"])
            top_losers  = [{"symbol": k.split(":")[1], **v} for k, v in sorted_stocks[:5]]
            top_gainers = [{"symbol": k.split(":")[1], **v} for k, v in sorted_stocks[-5:]][::-1]

            return MarketContext(
                timestamp=pd.Timestamp.now(tz="Asia/Kolkata"),
                india_vix=round(vix_live, 2),
                nifty_last=round(nifty_last, 2),
                banknifty_last=round(banknifty_last, 2),
                nifty_change_pct=round(n_chg, 2),
                banknifty_change_pct=round(bn_chg, 2),
                banknifty_vs_nifty=round(bn_vs_n, 2),
                em_nifty_daily=round(em_n_daily, 0),
                em_nifty_weekly=round(em_n_weekly, 0),
                em_banknifty_daily=round(em_bn_daily, 0),
                em_banknifty_weekly=round(em_bn_weekly, 0),
                nifty50_quotes=stock_perf,
                top_gainers=top_gainers,
                top_losers=top_losers,
            )
        except Exception as e:
            logger.warning("Market context fetch failed: {}", e)
            return None
