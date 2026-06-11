"""Database connection and helper utilities for nse_algo PostgreSQL + TimescaleDB."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        url = os.environ.get(
            "DATABASE_URL",
            "postgresql://jthileti@localhost:5432/nse_algo",
        )
        _engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)
        logger.info("Database engine created: {}", url.split("@")[-1])
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    return _SessionLocal


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager yielding a database session with automatic rollback on error."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── OHLCV helpers ─────────────────────────────────────────────────────────────

def upsert_ohlcv(df: pd.DataFrame, symbol: str, interval_min: int) -> int:
    """Insert OHLCV rows, skipping duplicates. Returns number of rows inserted."""
    if df.empty:
        return 0

    records = df.reset_index().rename(columns={"datetime": "time"})
    records["symbol"] = symbol
    records["interval_min"] = interval_min

    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                INSERT INTO ohlcv (time, symbol, interval_min, open, high, low, close, volume)
                VALUES (:time, :symbol, :interval_min, :open, :high, :low, :close, :volume)
                ON CONFLICT (time, symbol, interval_min) DO NOTHING
            """),
            records[["time", "symbol", "interval_min", "open", "high", "low", "close", "volume"]]
            .to_dict(orient="records"),
        )
        conn.commit()
        inserted = result.rowcount
    logger.debug("upsert_ohlcv: {} rows → {} inserted ({} {}min)", len(df), inserted, symbol, interval_min)
    return inserted


def load_ohlcv_db(symbol: str, interval_min: int, limit: int | None = None) -> pd.DataFrame:
    """Load OHLCV from TimescaleDB, most recent first."""
    sql = """
        SELECT time, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = :symbol AND interval_min = :interval_min
        ORDER BY time ASC
    """
    if limit:
        sql += f" LIMIT {limit}"
    engine = get_engine()
    df = pd.read_sql(text(sql), engine, params={"symbol": symbol, "interval_min": interval_min})
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Asia/Kolkata")
    return df.set_index("time")


# ── Features helpers ──────────────────────────────────────────────────────────

def upsert_features(df: pd.DataFrame, symbol: str, interval_min: int) -> int:
    if df.empty:
        return 0

    records = df.reset_index().rename(columns={"datetime": "time", "index": "time"})
    records["symbol"] = symbol
    records["interval_min"] = interval_min

    cols = list(records.columns)
    col_list = ", ".join(cols)
    val_list = ", ".join(f":{c}" for c in cols)

    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text(f"""
                INSERT INTO features ({col_list})
                VALUES ({val_list})
                ON CONFLICT (time, symbol, interval_min) DO NOTHING
            """),
            records.to_dict(orient="records"),
        )
        conn.commit()
    return result.rowcount


# ── Signal logging ────────────────────────────────────────────────────────────

def log_signal(
    time,
    symbol: str,
    interval_min: int,
    direction: int,
    confidence: float,
    regime: str | None,
    model_version: str,
    strategy_version: str,
    git_commit: str | None = None,
) -> str:
    """Insert a signal record and return the generated signal_id UUID."""
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                INSERT INTO signals
                    (time, symbol, interval_min, direction, confidence, regime,
                     model_version, strategy_version, git_commit)
                VALUES
                    (:time, :symbol, :interval_min, :direction, :confidence, :regime,
                     :model_version, :strategy_version, :git_commit)
                RETURNING signal_id
            """),
            {
                "time": time, "symbol": symbol, "interval_min": interval_min,
                "direction": direction, "confidence": confidence, "regime": regime,
                "model_version": model_version, "strategy_version": strategy_version,
                "git_commit": git_commit,
            },
        ).fetchone()
        conn.commit()
    return str(row[0])


# ── Order / fill logging ──────────────────────────────────────────────────────

def log_order(
    signal_id: str | None,
    broker: str,
    broker_order_id: str | None,
    symbol: str,
    side: str,
    order_type: str,
    qty: int,
    price: float | None,
    trigger_price: float | None,
    status: str,
    strategy_version: str,
    model_version: str,
    git_commit: str | None = None,
    reject_reason: str | None = None,
) -> int:
    """Insert an order record and return its DB id."""
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                INSERT INTO orders
                    (signal_id, broker, broker_order_id, symbol, side, order_type,
                     qty, price, trigger_price, status, strategy_version, model_version,
                     git_commit, reject_reason)
                VALUES
                    (:signal_id, :broker, :broker_order_id, :symbol, :side, :order_type,
                     :qty, :price, :trigger_price, :status, :strategy_version, :model_version,
                     :git_commit, :reject_reason)
                RETURNING id
            """),
            {
                "signal_id": signal_id, "broker": broker, "broker_order_id": broker_order_id,
                "symbol": symbol, "side": side, "order_type": order_type,
                "qty": qty, "price": price, "trigger_price": trigger_price, "status": status,
                "strategy_version": strategy_version, "model_version": model_version,
                "git_commit": git_commit, "reject_reason": reject_reason,
            },
        ).fetchone()
        conn.commit()
    return row[0]


def log_risk_event(event_type: str, detail: str, daily_pnl_pct: float | None = None,
                   daily_dd_pct: float | None = None) -> None:
    """Persist a circuit breaker trigger to the risk_events table."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO risk_events (event_type, detail, daily_pnl_pct, daily_dd_pct)
                VALUES (:event_type, :detail, :daily_pnl_pct, :daily_dd_pct)
            """),
            {"event_type": event_type, "detail": detail,
             "daily_pnl_pct": daily_pnl_pct, "daily_dd_pct": daily_dd_pct},
        )
        conn.commit()


# ── Schema auto-init ──────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they don't exist (plain PostgreSQL, no TimescaleDB needed).

    Safe to call multiple times — all statements are CREATE … IF NOT EXISTS.
    Called automatically by ping() on first successful connection so Streamlit
    Cloud deployments self-provision without a manual psql step.
    """
    schema = Path(__file__).parent / "schema_plain.sql"
    sql = schema.read_text()
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()
    logger.info("init_db: schema applied (plain PostgreSQL)")


# ── Health check ──────────────────────────────────────────────────────────────

def ping() -> bool:
    """Return True if the database is reachable; auto-creates tables on first connect."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        # Tables may not exist yet on a brand-new cloud DB — create them silently
        try:
            init_db()
        except Exception as e:
            logger.debug("init_db skipped: {}", e)
        return True
    except Exception as e:
        logger.error("DB ping failed: {}", e)
        return False
