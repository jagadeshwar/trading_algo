"""Day 1 — Full historical data bootstrap.

Downloads 2 years of OHLCV data for all symbols and intervals,
computes all 35 ML features, and saves everything to both
Parquet files and the TimescaleDB database.

Run once after setting up credentials:
    python day1_run.py

Re-run safely — existing data is merged, not overwritten.
Options:
    --symbols NSE:NIFTY50-INDEX NSE:NIFTYBANK-INDEX   # subset of symbols
    --intervals 5 15                                   # subset of intervals
    --skip-db                                          # save to Parquet only
    --skip-features                                    # skip feature computation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
logger.add("logs/day1_bootstrap_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days")


def run_bootstrap(
    symbols: list[str],
    intervals: list[int],
    skip_db: bool,
    skip_features: bool,
) -> None:
    from auth import get_fyers_client
    from production.data.fetcher import DataFetcher, HISTORY_DAYS
    from production.data.features import FeatureEngineer

    if not skip_db:
        from production.db.database import ping, upsert_ohlcv, upsert_features
        if not ping():
            logger.error("Cannot reach database. Start PostgreSQL or use --skip-db.")
            sys.exit(1)
        logger.info("Database connection OK")

    logger.info("Authenticating with Fyers...")
    client = get_fyers_client()
    fetcher = DataFetcher(client)
    engineer = FeatureEngineer()

    total = len(symbols) * len(intervals)
    done = 0
    errors: list[str] = []
    t_start = time.monotonic()

    logger.info("Bootstrap: {} symbols × {} intervals = {} downloads", len(symbols), len(intervals), total)
    logger.info("Estimated time: {:.0f}–{:.0f} minutes", total * 0.3 / 60, total * 1 / 60)

    for symbol in symbols:
        for interval_min in intervals:
            days = HISTORY_DAYS.get(interval_min, 365)
            done += 1
            tag = f"[{done}/{total}] {symbol} {interval_min}min ({days}d)"

            try:
                # ── Fetch ──────────────────────────────────────────────────
                logger.info("{} — fetching...", tag)
                df = fetcher.fetch_historical_chunked(symbol, interval_min)

                if df.empty:
                    logger.warning("{} — no data returned, skipping", tag)
                    errors.append(f"EMPTY: {tag}")
                    continue

                logger.info("{} — {} candles fetched", tag, len(df))

                # ── Save OHLCV to Parquet ──────────────────────────────────
                fetcher.save_ohlcv(df, symbol, interval_min)

                # ── Save OHLCV to DB ───────────────────────────────────────
                if not skip_db:
                    inserted = upsert_ohlcv(df, symbol, interval_min)
                    logger.info("{} — {} rows inserted into DB", tag, inserted)

                # ── Feature engineering ────────────────────────────────────
                if not skip_features:
                    try:
                        features = engineer.compute(df)
                        engineer.save(features, symbol, interval_min)
                        logger.info("{} — {} feature rows saved", tag, len(features))

                        if not skip_db:
                            upsert_features(features, symbol, interval_min)
                    except Exception as fe:
                        logger.warning("{} — feature computation failed: {}", tag, fe)
                        errors.append(f"FEATURES: {tag}: {fe}")

                time.sleep(0.3)  # respect Fyers rate limits

            except Exception as e:
                logger.error("{} — FAILED: {}", tag, e)
                errors.append(f"ERROR: {tag}: {e}")
                time.sleep(1)  # back off on error

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    logger.info("="*60)
    logger.info("Bootstrap complete in {:.1f} minutes", elapsed / 60)
    logger.info("  Successful: {}/{}", total - len(errors), total)

    if errors:
        logger.warning("  {} errors:", len(errors))
        for e in errors:
            logger.warning("    {}", e)
    else:
        logger.success("  All downloads succeeded ✓")

    # ── Verify DB row counts ──────────────────────────────────────────────────
    if not skip_db:
        try:
            from production.db.database import get_engine
            from sqlalchemy import text
            with get_engine().connect() as conn:
                ohlcv_count = conn.execute(text("SELECT COUNT(*) FROM ohlcv")).scalar()
                feat_count = conn.execute(text("SELECT COUNT(*) FROM features")).scalar()
            logger.info("DB row counts — ohlcv: {:,}  features: {:,}", ohlcv_count, feat_count)
        except Exception as e:
            logger.warning("Could not query DB row counts: {}", e)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    from production.data.fetcher import SYMBOLS, INTERVAL_MAP

    parser = argparse.ArgumentParser(description="NSE Algo — Day 1 OHLCV bootstrap")
    parser.add_argument(
        "--symbols", nargs="+", default=SYMBOLS,
        help="Symbols to download (default: all 13+VIX)"
    )
    parser.add_argument(
        "--intervals", nargs="+", type=int, default=list(INTERVAL_MAP.keys()),
        help="Intervals in minutes (default: 1 5 15 1440)"
    )
    parser.add_argument(
        "--skip-db", action="store_true",
        help="Save to Parquet only, skip TimescaleDB"
    )
    parser.add_argument(
        "--skip-features", action="store_true",
        help="Skip feature computation (download OHLCV only)"
    )
    args = parser.parse_args()

    logger.info("NSE Algo Trader — Day 1 Bootstrap")
    logger.info("Symbols: {}", args.symbols)
    logger.info("Intervals: {} min", args.intervals)
    logger.info("DB write: {}", "disabled" if args.skip_db else "enabled")
    logger.info("Features: {}", "disabled" if args.skip_features else "enabled")
    print()

    run_bootstrap(
        symbols=args.symbols,
        intervals=args.intervals,
        skip_db=args.skip_db,
        skip_features=args.skip_features,
    )


if __name__ == "__main__":
    main()
