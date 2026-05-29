"""APScheduler — daily IST task schedule for live trading."""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

Path("logs").mkdir(exist_ok=True)
logger.add("logs/scheduler_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days")

IST = "Asia/Kolkata"
scheduler = BlockingScheduler(timezone=IST)


# ── Task functions ─────────────────────────────────────────────────────────────

def task_refresh_token() -> None:
    logger.info("08:50 — Refreshing Fyers access token")
    try:
        from auth import get_access_token
        get_access_token()
        logger.info("Token refreshed successfully")
    except Exception as e:
        logger.error("Token refresh failed: {}", e)


def task_premarket_warmup() -> None:
    logger.info("09:00 — Pre-market data warm-up")
    try:
        from auth import get_fyers_client
        from production.data.fetcher import DataFetcher
        client = get_fyers_client()
        fetcher = DataFetcher(client)
        # Pull last 2 days to ensure features are current
        fetcher.incremental_update(days=2)
        logger.info("Pre-market warm-up complete")
    except Exception as e:
        logger.error("Pre-market warm-up failed: {}", e)


def task_market_open() -> None:
    logger.info("09:15 — Market open: starting live feed and signal generation")
    # TODO Phase 2: start WebSocket, run ML inference loop


def task_market_close() -> None:
    logger.info("15:30 — Market close: stopping live feed")
    # TODO Phase 2: stop WebSocket, log session summary


def task_eod_data_update() -> None:
    logger.info("16:00 — End-of-day incremental data update")
    try:
        from auth import get_fyers_client
        from production.data.fetcher import DataFetcher
        client = get_fyers_client()
        fetcher = DataFetcher(client)
        fetcher.incremental_update(days=2)
        logger.info("EOD data update complete")
    except Exception as e:
        logger.error("EOD data update failed: {}", e)


def task_recompute_features() -> None:
    logger.info("16:30 — Recomputing features for all symbols")
    try:
        from production.data.fetcher import DataFetcher, SYMBOLS, INTERVAL_MAP
        from production.data.features import FeatureEngineer
        engineer = FeatureEngineer()
        # Fetcher needs a client; skip if no token available (e.g. weekends)
        from auth import get_fyers_client
        client = get_fyers_client()
        fetcher = DataFetcher(client)
        for symbol in SYMBOLS:
            for interval_min in INTERVAL_MAP:
                try:
                    df = fetcher.load_ohlcv(symbol, interval_min)
                    features = engineer.compute(df)
                    engineer.save(features, symbol, interval_min)
                except FileNotFoundError:
                    pass
                except Exception as e:
                    logger.error("Feature recompute failed {} {}min: {}", symbol, interval_min, e)
        logger.info("Feature recompute complete")
    except Exception as e:
        logger.error("Feature recompute task failed: {}", e)


def task_weekly_retrain() -> None:
    """Trigger model retraining on Sunday 18:00 if Sharpe has dropped."""
    logger.info("18:00 Sunday — Checking if model retraining is needed")
    # TODO Phase 3: check alpha decay, trigger Optuna + retraining if Sharpe < 1.0


# ── Schedule ──────────────────────────────────────────────────────────────────

def register_jobs() -> None:
    scheduler.add_job(task_refresh_token,    CronTrigger(hour=8,  minute=50, timezone=IST), id="refresh_token")
    scheduler.add_job(task_premarket_warmup, CronTrigger(hour=9,  minute=0,  timezone=IST), id="premarket_warmup")
    scheduler.add_job(task_market_open,      CronTrigger(hour=9,  minute=15, timezone=IST), id="market_open")
    scheduler.add_job(task_market_close,     CronTrigger(hour=15, minute=30, timezone=IST), id="market_close")
    scheduler.add_job(task_eod_data_update,  CronTrigger(hour=16, minute=0,  timezone=IST), id="eod_update",
                      day_of_week="mon-fri")
    scheduler.add_job(task_recompute_features, CronTrigger(hour=16, minute=30, timezone=IST), id="recompute_features",
                      day_of_week="mon-fri")
    scheduler.add_job(task_weekly_retrain,   CronTrigger(hour=18, minute=0, day_of_week="sun", timezone=IST),
                      id="weekly_retrain")


def _shutdown(signum, frame) -> None:
    logger.info("Shutdown signal received — stopping scheduler")
    scheduler.shutdown(wait=False)
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    register_jobs()
    logger.info("Scheduler started (IST timezone). Jobs registered:")
    for job in scheduler.get_jobs():
        logger.info("  {} — next run: {}", job.id, job.next_run_time)
    scheduler.start()
