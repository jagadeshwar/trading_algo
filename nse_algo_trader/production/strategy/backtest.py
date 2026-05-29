"""Vectorbt backtester for the momentum strategy.

Runs on historical data with full NSE transaction costs.
Reports Sharpe, max drawdown, win rate, profit factor.

Usage:
    python -m production.strategy.backtest
    python -m production.strategy.backtest --symbol NSE:NIFTYBANK-INDEX --interval 5
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

warnings.filterwarnings("ignore")

RISK_FREE_RATE = 0.065   # 6.5% India 10-year gilt


def load_ohlcv_and_features(symbol: str, interval_min: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from production.data.fetcher import DataFetcher
    from production.data.features import FeatureEngineer

    slug = symbol.replace(":", "_").replace("-", "_")
    ohlcv_path = Path(f"data/ohlcv/{slug}_{interval_min}min.parquet")
    feat_path  = Path(f"data/features/{slug}_{interval_min}min_features.parquet")

    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV not found: {ohlcv_path}. Run day1_run.py first.")
    if not feat_path.exists():
        raise FileNotFoundError(f"Features not found: {feat_path}. Run day1_run.py first.")

    import pyarrow.parquet as pq
    ohlcv = pq.read_table(ohlcv_path).to_pandas()
    ohlcv.index = pd.to_datetime(ohlcv.index, utc=True).tz_convert("Asia/Kolkata")

    feats = pq.read_table(feat_path).to_pandas()
    feats.index = pd.to_datetime(feats.index, utc=True).tz_convert("Asia/Kolkata")

    return ohlcv, feats


def run_backtest(
    symbol: str = "NSE:NIFTY50-INDEX",
    interval_min: int = 5,
    capital: float = 500_000,
    verbose: bool = True,
) -> dict:
    from production.strategy.momentum import MomentumStrategy, load_config

    logger.info("Loading data for {} {}min...", symbol, interval_min)
    ohlcv, feats = load_ohlcv_and_features(symbol, interval_min)

    # Align on common index
    common = feats.index.intersection(ohlcv.index)
    feats  = feats.loc[common]
    ohlcv  = ohlcv.loc[common]
    close  = ohlcv["close"]

    # Load VIX for regime gate (if available)
    vix_series = None
    try:
        import pyarrow.parquet as pq
        vix_path = Path(f"data/ohlcv/NSE_INDIAVIX_INDEX_{interval_min}min.parquet")
        if vix_path.exists():
            vix_df = pq.read_table(vix_path).to_pandas()
            vix_df.index = pd.to_datetime(vix_df.index, utc=True).tz_convert("Asia/Kolkata")
            vix_series = vix_df["close"].reindex(common)
    except Exception:
        pass

    strategy = MomentumStrategy(load_config())
    logger.info("Generating signals on {} bars...", len(feats))
    signals_df = strategy.generate_signals_df(feats, close, vix=vix_series)

    # ── Load NSE transaction costs ─────────────────────────────────────────────
    costs_cfg = {}
    try:
        costs_cfg = yaml.safe_load(Path("configs/broker.yaml").read_text()).get("transaction_costs", {})
    except Exception:
        pass
    fee_pct     = costs_cfg.get("exchange_fee_pct",          0.00345) / 100
    slippage_pct= costs_cfg.get("slippage_pct",               0.02)   / 100
    brokerage   = costs_cfg.get("brokerage_per_order",        20)
    total_fee   = fee_pct + slippage_pct + (brokerage / (capital * 0.05))

    # ── Vectorbt portfolio ─────────────────────────────────────────────────────
    try:
        import vectorbt as vbt

        entries = signals_df["direction"] == 1
        exits   = signals_df["direction"] == -1

        # ATR-based SL/TP as price fractions (per-bar, aligned to entries)
        sl_stop = (signals_df["atr"] * 2.0 / close).clip(0.005, 0.05)
        tp_stop = (signals_df["atr"] * 3.0 / close).clip(0.005, 0.08)

        # Fixed fractional sizing: 5% of capital per trade
        size = capital * 0.05 / close

        pf = vbt.Portfolio.from_signals(
            close         = close,
            entries       = entries,
            exits         = exits,
            short_entries = signals_df["direction"] == -1,
            short_exits   = signals_df["direction"] ==  1,
            size          = size,
            size_type     = "amount",
            sl_stop       = sl_stop,
            tp_stop       = tp_stop,
            fees          = total_fee,
            slippage      = slippage_pct,
            init_cash     = capital,
            freq          = f"{interval_min}min",
        )

        stats = pf.stats()

        # ── Extract key metrics ────────────────────────────────────────────────
        sharpe       = float(stats.get("Sharpe Ratio", 0))
        max_dd       = float(stats.get("Max Drawdown [%]", 100))
        total_ret    = float(stats.get("Total Return [%]", 0))
        total_trades = int(stats.get("Total Trades", 0))
        win_rate     = float(stats.get("Win Rate [%]", 0))

        # Profit factor from returns
        rets = pf.returns()
        gross_profit = rets[rets > 0].sum()
        gross_loss   = abs(rets[rets < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        result = {
            "symbol":        symbol,
            "interval_min":  interval_min,
            "bars":          len(close),
            "signals":       int((signals_df["direction"] != 0).sum()),
            "total_trades":  total_trades,
            "sharpe":        round(sharpe, 3),
            "max_dd_pct":    round(max_dd, 2),
            "total_ret_pct": round(total_ret, 2),
            "win_rate_pct":  round(win_rate, 1),
            "profit_factor": round(profit_factor, 3),
            "live_ready":    sharpe > 1.5 and max_dd < 10 and win_rate > 52,
        }

    except Exception as e:
        logger.warning("Vectorbt error ({}), falling back to manual calculation", e)
        result = _manual_backtest(signals_df, close, capital, total_fee, interval_min)
        result["symbol"] = symbol
        result["interval_min"] = interval_min

    if verbose:
        _print_report(result, signals_df, close)

    return result


def _manual_backtest(
    signals_df: pd.DataFrame,
    close: pd.Series,
    capital: float,
    fee_pct: float,
    interval_min: int = 5,
) -> dict:
    """Fallback bar-by-bar backtest when vectorbt has API issues."""
    equity = capital
    peak   = capital
    max_dd = 0.0
    trades: list[dict] = []

    position = None
    for ts, row in signals_df.iterrows():
        price = close.loc[ts]

        if position:
            d = position["direction"]
            if (d == 1 and price <= position["stop"]) or \
               (d == -1 and price >= position["stop"]):
                pnl = d * (price - position["entry"]) * position["qty"]
                pnl -= abs(price * position["qty"]) * fee_pct
                trades.append({"pnl": pnl, "exit": "STOP"})
                equity += pnl
                position = None
            elif (d == 1 and price >= position["target"]) or \
                 (d == -1 and price <= position["target"]):
                pnl = d * (price - position["entry"]) * position["qty"]
                pnl -= abs(price * position["qty"]) * fee_pct
                trades.append({"pnl": pnl, "exit": "TARGET"})
                equity += pnl
                position = None

        if position is None and row["direction"] != 0:
            qty = max(1, int(capital * 0.05 / price))
            cost = price * qty * fee_pct
            position = {
                "direction": int(row["direction"]),
                "entry": price,
                "stop":  float(row["stop"]),
                "target": float(row["target"]),
                "qty":   qty,
            }
            equity -= cost

        peak = max(peak, equity)
        dd   = (peak - equity) / peak * 100
        max_dd = max(max_dd, dd)

    if not trades:
        return {"total_trades": 0, "sharpe": 0, "max_dd_pct": max_dd,
                "total_ret_pct": 0, "win_rate_pct": 0, "profit_factor": 0,
                "live_ready": False, "bars": len(close), "signals": 0}

    pnls  = [t["pnl"] for t in trades]
    wins  = [p for p in pnls if p > 0]
    losses= [p for p in pnls if p < 0]
    bars_per_day = {1: 375, 5: 75, 15: 26, 1440: 1}.get(interval_min, 75)
    daily_rets = pd.Series(pnls) / capital
    sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(252 * bars_per_day)) if daily_rets.std() > 0 else 0

    return {
        "total_trades":  len(trades),
        "sharpe":        round(sharpe, 3),
        "max_dd_pct":    round(max_dd, 2),
        "total_ret_pct": round((equity - capital) / capital * 100, 2),
        "win_rate_pct":  round(len(wins) / len(trades) * 100, 1),
        "profit_factor": round(sum(wins) / abs(sum(losses)), 3) if losses else float("inf"),
        "live_ready":    sharpe > 1.5 and max_dd < 10 and len(wins)/len(trades) > 0.52,
        "bars":          len(close),
        "signals":       int((signals_df["direction"] != 0).sum()),
    }


def _print_report(result: dict, signals_df: pd.DataFrame, close: pd.Series) -> None:
    targets = {"sharpe": 1.5, "max_dd_pct": 10.0, "win_rate_pct": 52.0, "profit_factor": 1.4}
    sep = "=" * 58

    print(f"\n{sep}")
    print(f"  BACKTEST REPORT — {result['symbol']} {result['interval_min']}min")
    print(sep)
    print(f"  Bars analysed : {result['bars']:,}")
    print(f"  Signals fired : {result.get('signals', 'N/A')}")
    print(f"  Total trades  : {result['total_trades']}")
    print()

    def fmt(key, label, fmt_str, target, higher_is_better=True):
        val = result.get(key, 0)
        ok  = (val >= target) if higher_is_better else (val <= target)
        mark = "✓" if ok else "✗"
        print(f"  {label:<22} {fmt_str.format(val):<12} target={target}  {mark}")

    fmt("sharpe",        "Sharpe Ratio",    "{:.3f}", targets["sharpe"])
    fmt("max_dd_pct",    "Max Drawdown",    "{:.2f}%", targets["max_dd_pct"], higher_is_better=False)
    fmt("win_rate_pct",  "Win Rate",        "{:.1f}%", targets["win_rate_pct"])
    fmt("profit_factor", "Profit Factor",   "{:.3f}", targets["profit_factor"])
    print(f"  {'Total Return':<22} {result['total_ret_pct']:.2f}%")
    print()
    status = "LIVE READY ✓" if result.get("live_ready") else "NOT READY YET ✗"
    print(f"  Status: {status}")
    print(sep + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",   default="NSE:NIFTY50-INDEX")
    parser.add_argument("--interval", default=5, type=int)
    parser.add_argument("--capital",  default=500_000, type=float)
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    run_backtest(symbol=args.symbol, interval_min=args.interval,
                 capital=args.capital, verbose=True)
