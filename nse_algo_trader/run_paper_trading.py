"""Live paper trading runner — polls every 5 minutes during market hours.

Loads the latest OHLCV + features, generates signals via the momentum
strategy, simulates fills via PaperTrader, and prints a live PnL table.

Run from within the project directory with the venv active:
    python run_paper_trading.py

Options:
    --symbol    NSE:NIFTY50-INDEX     primary symbol to trade
    --interval  5                     bar size in minutes
    --capital   500000                paper capital in INR
    --backtest                        run backtest first, then start paper mode
    --no-db                           skip database logging
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

Path("logs").mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
logger.add("logs/paper_trading_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days")

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
SYMBOLS_ALL  = [
    "NSE:NIFTY50-INDEX",
    "NSE:NIFTYBANK-INDEX",
    "NSE:RELIANCE-EQ",
    "NSE:HDFCBANK-EQ",
    "NSE:ICICIBANK-EQ",
]


def is_market_open() -> bool:
    now = datetime.now(IST).time()
    return MARKET_OPEN <= now <= MARKET_CLOSE


def load_latest_bar(symbol: str, interval_min: int) -> tuple[float, object] | None:
    """Load most recent bar's close price and feature row from Parquet."""
    import pyarrow.parquet as pq
    slug = symbol.replace(":", "_").replace("-", "_")
    ohlcv_path = Path(f"data/ohlcv/{slug}_{interval_min}min.parquet")
    feat_path  = Path(f"data/features/{slug}_{interval_min}min_features.parquet")

    if not ohlcv_path.exists() or not feat_path.exists():
        return None

    try:
        import pandas as pd
        ohlcv = pq.read_table(ohlcv_path).to_pandas()
        ohlcv.index = pd.to_datetime(ohlcv.index, utc=True).tz_convert("Asia/Kolkata")
        feats = pq.read_table(feat_path).to_pandas()
        feats.index = pd.to_datetime(feats.index, utc=True).tz_convert("Asia/Kolkata")

        last_close = float(ohlcv["close"].iloc[-1])
        last_feats = feats.iloc[-1]
        return last_close, last_feats
    except Exception as e:
        logger.warning("Could not load data for {}: {}", symbol, e)
        return None


def fetch_live_price(symbol: str, fyers_client) -> float | None:
    """Fetch current LTP from Fyers (used during market hours)."""
    try:
        data = {"symbols": symbol}
        resp = fyers_client.quotes(data=data)
        if resp.get("s") == "ok":
            return float(resp["d"][0]["v"]["lp"])
    except Exception as e:
        logger.debug("Live price fetch failed for {}: {}", symbol, e)
    return None


def fetch_vix(fyers_client) -> float | None:
    try:
        data = {"symbols": "NSE:INDIAVIX-INDEX"}
        resp = fyers_client.quotes(data=data)
        if resp.get("s") == "ok":
            return float(resp["d"][0]["v"]["lp"])
    except Exception:
        pass
    return None


def print_status_table(trader, symbols: list[str], prices: dict[str, float]) -> None:
    summary = trader.summary()
    now = datetime.now(IST).strftime("%H:%M:%S")
    sep = "─" * 70

    print(f"\n{sep}")
    print(f"  PAPER TRADING  {now} IST")
    print(sep)
    print(f"  Capital : ₹{summary['capital']:>12,.0f}")
    print(f"  Equity  : ₹{summary['equity']:>12,.0f}   PnL: {summary['total_pnl']:+,.0f} ({summary['total_pnl_pct']:+.2f}%)")
    print(f"  Trades  : {summary['total_trades']}  Win Rate: {summary['win_rate']}%  PF: {summary['profit_factor']}")
    print(f"  Daily PnL: {summary['daily_pnl_pct']:+.2f}%   Max DD: {summary['max_dd_pct']:.2f}%")

    if trader._positions:
        print(f"\n  Open Positions:")
        for sym, pos in trader._positions.items():
            price = prices.get(sym, pos.entry_price)
            mtm = pos.direction * (price - pos.entry_price) * pos.qty
            pct = mtm / pos.value * 100
            side = "LONG " if pos.direction == 1 else "SHORT"
            print(f"    {side} {sym:<28} qty={pos.qty:<5} "
                  f"entry={pos.entry_price:.2f}  now={price:.2f}  mtm={mtm:+.0f} ({pct:+.1f}%)")
    else:
        print("\n  No open positions")

    print(sep)


def run_paper_session(
    symbols: list[str],
    interval_min: int,
    capital: float,
    use_db: bool,
) -> None:
    from auth import get_fyers_client
    from production.strategy.momentum import MomentumStrategy, load_config
    from production.execution.paper_trader import PaperTrader

    logger.info("Initialising paper trading session")
    logger.info("Symbols: {}", symbols)
    logger.info("Capital: ₹{:,.0f}", capital)
    logger.info("Interval: {}min", interval_min)

    try:
        fyers = get_fyers_client()
        logger.info("Fyers client ready")
    except Exception as e:
        logger.error("Fyers auth failed: {}", e)
        fyers = None

    strategy = MomentumStrategy(load_config())
    trader   = PaperTrader(capital=capital, use_db=use_db)
    trader.reset_daily()

    poll_seconds = interval_min * 60
    bar_count = 0

    logger.info("Paper trading started — polling every {}s", poll_seconds)
    logger.info("Market hours: 09:15–15:30 IST. Press Ctrl+C to stop.\n")

    try:
        while True:
            now = datetime.now(IST)

            if not is_market_open():
                next_check = 60
                if now.time() < MARKET_OPEN:
                    mins = (datetime.combine(now.date(), MARKET_OPEN, tzinfo=IST) - now).seconds // 60
                    logger.info("Market opens in {} min. Waiting...", mins)
                else:
                    logger.info("Market closed. Session summary:")
                    summary = trader.summary()
                    logger.info("  Trades: {}  Win rate: {}%  PnL: {}{:+,.0f} ({:+.2f}%)",
                                summary["total_trades"], summary["win_rate"],
                                "₹", summary["total_pnl"], summary["total_pnl_pct"])
                    break
                time.sleep(next_check)
                continue

            bar_count += 1
            logger.info("── Bar {} @ {} ──", bar_count, now.strftime("%H:%M"))

            prices: dict[str, float] = {}
            features_map: dict[str, object] = {}

            # ── Fetch prices and features for each symbol ─────────────────────
            vix = fetch_vix(fyers) if fyers else None

            for symbol in symbols:
                # Try live price first, fall back to last Parquet bar
                price = fetch_live_price(symbol, fyers) if fyers else None
                result = load_latest_bar(symbol, interval_min)

                if result is None:
                    continue
                last_close, last_feats = result

                if price is None:
                    price = last_close

                prices[symbol] = price
                features_map[symbol] = last_feats

                # ── Generate signal ────────────────────────────────────────────
                signal = strategy.evaluate_bar(symbol, last_feats, price, vix=vix)
                if signal and signal.confidence >= 0.6:
                    trader.on_signal(signal)

            # ── Update open positions ──────────────────────────────────────────
            trader.update_positions(prices, features_map)

            # ── Check circuit breakers ─────────────────────────────────────────
            state = trader.circuit.check_all(
                daily_pnl_pct=trader.summary()["daily_pnl_pct"],
                daily_dd_pct=trader.summary()["max_dd_pct"],
            )
            if state.broken:
                logger.critical("CIRCUIT BREAK: {}. Stopping paper session.", state.reason)
                break

            # ── Print status ───────────────────────────────────────────────────
            print_status_table(trader, symbols, prices)

            time.sleep(poll_seconds)

    except KeyboardInterrupt:
        logger.info("Paper trading stopped by user")

    # ── Final summary ──────────────────────────────────────────────────────────
    summary = trader.summary()
    print(f"\n{'='*58}")
    print("  FINAL PAPER TRADING SUMMARY")
    print(f"{'='*58}")
    for k, v in summary.items():
        if k != "sizer_stats":
            print(f"  {k:<22} {v}")
    print(f"{'='*58}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="NSE Algo — Paper Trading")
    parser.add_argument("--symbol",    default="NSE:NIFTY50-INDEX",
                        help="Primary symbol (default: NSE:NIFTY50-INDEX)")
    parser.add_argument("--symbols",   nargs="+", default=None,
                        help="Multiple symbols (overrides --symbol)")
    parser.add_argument("--interval",  default=5, type=int,
                        help="Bar interval in minutes (default: 5)")
    parser.add_argument("--capital",   default=500_000, type=float,
                        help="Paper capital in INR (default: 500000)")
    parser.add_argument("--backtest",  action="store_true",
                        help="Run backtest first before paper trading")
    parser.add_argument("--no-db",     action="store_true",
                        help="Skip database logging")
    args = parser.parse_args()

    symbols = args.symbols or [args.symbol]

    if args.backtest:
        from production.strategy.backtest import run_backtest
        logger.info("Running backtest before paper session...")
        for sym in symbols:
            result = run_backtest(symbol=sym, interval_min=args.interval,
                                  capital=args.capital, verbose=True)
            if not result.get("live_ready"):
                logger.warning("{} did not pass live-ready thresholds. Proceeding anyway (paper mode).", sym)

    run_paper_session(
        symbols=symbols,
        interval_min=args.interval,
        capital=args.capital,
        use_db=not args.no_db,
    )


if __name__ == "__main__":
    main()
