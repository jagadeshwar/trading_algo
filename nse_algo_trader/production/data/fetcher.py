"""DataFetcher — historical OHLCV, options chain, Parquet storage."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

DATA_DIR = Path("data")
OHLCV_DIR = DATA_DIR / "ohlcv"
OPTIONS_DIR = DATA_DIR / "options"

# Fyers interval codes
INTERVAL_MAP: dict[int, str] = {
    1: "1",
    5: "5",
    15: "15",
    1440: "D",
}

# Symbols and history depth per timeframe (days)
SYMBOLS = [
    "NSE:NIFTY50-INDEX",
    "NSE:NIFTYBANK-INDEX",
    "NSE:FINNIFTY-INDEX",
    "NSE:RELIANCE-EQ",
    "NSE:TCS-EQ",
    "NSE:HDFCBANK-EQ",
    "NSE:INFY-EQ",
    "NSE:ICICIBANK-EQ",
    "NSE:SBIN-EQ",
    "NSE:AXISBANK-EQ",
    "NSE:KOTAKBANK-EQ",
    "NSE:LT-EQ",
    "NSE:ITC-EQ",
    "NSE:INDIAVIX-INDEX",
]

HISTORY_DAYS: dict[int, int] = {
    1: 60,
    5: 180,
    15: 365,
    1440: 730,
}


class DataFetcher:
    """Fetch and store OHLCV data via Fyers API v3."""

    def __init__(self, fyers_client) -> None:
        self._fyers = fyers_client
        OHLCV_DIR.mkdir(parents=True, exist_ok=True)
        OPTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Historical OHLCV ──────────────────────────────────────────────────────

    def fetch_historical(
        self,
        symbol: str,
        interval_min: int,
        days: int | None = None,
    ) -> pd.DataFrame:
        """Pull OHLCV candles from Fyers and return as DataFrame."""
        if interval_min not in INTERVAL_MAP:
            raise ValueError(f"Unsupported interval: {interval_min}. Use {list(INTERVAL_MAP)}")

        lookback = days or HISTORY_DAYS.get(interval_min, 365)
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=lookback)

        data = {
            "symbol": symbol,
            "resolution": INTERVAL_MAP[interval_min],
            "date_format": "1",
            "range_from": start_dt.strftime("%Y-%m-%d"),
            "range_to": end_dt.strftime("%Y-%m-%d"),
            "cont_flag": "1",
        }

        logger.info("Fetching {} {}min ({}d)", symbol, interval_min, lookback)
        response = self._fyers.history(data=data)

        if response.get("s") != "ok":
            raise RuntimeError(f"Fyers history error for {symbol}: {response.get('message')}")

        candles = response.get("candles", [])
        if not candles:
            logger.warning("No candles returned for {} {}min", symbol, interval_min)
            return pd.DataFrame()

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(
            "Asia/Kolkata"
        )
        df = df.set_index("datetime").drop(columns=["timestamp"])
        df = df.sort_index()
        df.attrs["symbol"] = symbol
        df.attrs["interval_min"] = interval_min
        return df

    def fetch_historical_chunked(
        self, symbol: str, interval_min: int, days: int | None = None
    ) -> pd.DataFrame:
        """Fetch long histories in 100-day chunks to respect API limits."""
        lookback = days or HISTORY_DAYS.get(interval_min, 365)
        chunk_size = 100
        frames: list[pd.DataFrame] = []
        end_dt = datetime.now()

        while lookback > 0:
            fetch_days = min(chunk_size, lookback)
            start_dt = end_dt - timedelta(days=fetch_days)

            data = {
                "symbol": symbol,
                "resolution": INTERVAL_MAP[interval_min],
                "date_format": "1",
                "range_from": start_dt.strftime("%Y-%m-%d"),
                "range_to": end_dt.strftime("%Y-%m-%d"),
                "cont_flag": "1",
            }
            response = self._fyers.history(data=data)
            if response.get("s") == "ok" and response.get("candles"):
                frames.append(
                    pd.DataFrame(
                        response["candles"],
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                )
            time.sleep(0.2)  # respect rate limits
            end_dt = start_dt
            lookback -= fetch_days

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(
            "Asia/Kolkata"
        )
        df = df.set_index("datetime").drop(columns=["timestamp"])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.attrs["symbol"] = symbol
        df.attrs["interval_min"] = interval_min
        return df

    # ── Options chain ─────────────────────────────────────────────────────────

    def fetch_options_chain(self, symbol: str, strike_count: int = 20) -> pd.DataFrame:
        """Fetch live options chain — OI, IV, Greeks, bid/ask."""
        data = {"symbol": symbol, "strikecount": strike_count, "timestamp": ""}
        response = self._fyers.optionchain(data=data)

        if response.get("s") != "ok":
            raise RuntimeError(f"Options chain error for {symbol}: {response.get('message')}")

        records = []
        for item in response.get("optionsChain", []):
            records.append(
                {
                    "strike": item.get("strikePrice"),
                    "expiry": item.get("expiryDate"),
                    "option_type": item.get("option_type"),
                    "ltp": item.get("ltp"),
                    "oi": item.get("oi"),
                    "oi_change": item.get("oiChange"),
                    "iv": item.get("iv"),
                    "delta": item.get("delta"),
                    "gamma": item.get("gamma"),
                    "theta": item.get("theta"),
                    "vega": item.get("vega"),
                    "bid": item.get("bid"),
                    "ask": item.get("ask"),
                    "volume": item.get("volume"),
                }
            )

        df = pd.DataFrame(records)
        df["fetched_at"] = pd.Timestamp.now(tz="Asia/Kolkata")
        return df

    # ── Parquet storage ───────────────────────────────────────────────────────

    def save_ohlcv(self, df: pd.DataFrame, symbol: str, interval_min: int) -> Path:
        """Append-merge DataFrame into Parquet; returns file path."""
        path = self._ohlcv_path(symbol, interval_min)
        if path.exists():
            existing = pq.read_table(path).to_pandas()
            existing.index = pd.to_datetime(existing.index, utc=True).tz_convert("Asia/Kolkata")
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = df

        table = pa.Table.from_pandas(combined)
        pq.write_table(table, path, compression="snappy")
        logger.info("Saved {} {}min → {} rows to {}", symbol, interval_min, len(combined), path)
        return path

    def load_ohlcv(self, symbol: str, interval_min: int) -> pd.DataFrame:
        """Load OHLCV Parquet for a symbol/interval."""
        path = self._ohlcv_path(symbol, interval_min)
        if not path.exists():
            raise FileNotFoundError(f"No data for {symbol} {interval_min}min at {path}")
        df = pq.read_table(path).to_pandas()
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
        return df.sort_index()

    def save_options(self, df: pd.DataFrame, symbol: str) -> Path:
        """Append options chain snapshot to Parquet."""
        slug = symbol.replace(":", "_").replace("-", "_")
        path = OPTIONS_DIR / f"{slug}_options.parquet"
        if path.exists():
            existing = pq.read_table(path).to_pandas()
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df
        pq.write_table(pa.Table.from_pandas(combined), path, compression="snappy")
        return path

    # ── Bootstrap ─────────────────────────────────────────────────────────────

    def bootstrap_all(self) -> None:
        """Download full history for all symbols and intervals. Run once."""
        total = len(SYMBOLS) * len(INTERVAL_MAP)
        done = 0
        for symbol in SYMBOLS:
            for interval_min in INTERVAL_MAP:
                try:
                    df = self.fetch_historical_chunked(symbol, interval_min)
                    if not df.empty:
                        self.save_ohlcv(df, symbol, interval_min)
                    done += 1
                    logger.info("Bootstrap {}/{}: {} {}min", done, total, symbol, interval_min)
                    time.sleep(0.3)
                except Exception as e:
                    logger.error("Bootstrap failed for {} {}min: {}", symbol, interval_min, e)

    def incremental_update(self, days: int = 2) -> None:
        """Pull last N days and merge — run daily after market close."""
        for symbol in SYMBOLS:
            for interval_min in INTERVAL_MAP:
                try:
                    df = self.fetch_historical(symbol, interval_min, days=days)
                    if not df.empty:
                        self.save_ohlcv(df, symbol, interval_min)
                except Exception as e:
                    logger.error("Incremental update failed {} {}min: {}", symbol, interval_min, e)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _ohlcv_path(symbol: str, interval_min: int) -> Path:
        slug = symbol.replace(":", "_").replace("-", "_")
        return OHLCV_DIR / f"{slug}_{interval_min}min.parquet"
