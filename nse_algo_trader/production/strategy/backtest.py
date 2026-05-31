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
    """Load OHLCV and recompute features fresh (includes HMM regime label)."""
    from production.data.features import FeatureEngineer
    import pyarrow.parquet as pq

    slug       = symbol.replace(":", "_").replace("-", "_")
    ohlcv_path = Path(f"data/ohlcv/{slug}_{interval_min}min.parquet")

    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV not found: {ohlcv_path}. Run the data bootstrap first.")

    ohlcv = pq.read_table(ohlcv_path).to_pandas()
    ohlcv.index = pd.to_datetime(ohlcv.index, utc=True).tz_convert("Asia/Kolkata")

    hmm_path = str(Path("models/regime_hmm.pkl")) if Path("models/regime_hmm.pkl").exists() else None
    feats    = FeatureEngineer().compute(ohlcv, hmm_model_path=hmm_path)

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
    open_series = ohlcv["open"] if "open" in ohlcv.columns else None
    signals_df = strategy.generate_signals_df(feats, close, vix=vix_series, open_=open_series)

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

        # Apply same confidence + session filters as _manual_backtest
        import datetime as _dt
        _cfg = strategy.cfg
        _s = _dt.time(*[int(x) for x in _cfg.session_start.split(":")])
        _e = _dt.time(*[int(x) for x in _cfg.session_end.split(":")])
        session_mask = pd.Series(
            [_s <= t < _e for t in signals_df.index.time],
            index=signals_df.index,
        )
        conf_mask = signals_df["confidence"] >= _cfg.min_confidence
        valid_long  = (signals_df["direction"] == 1)  & conf_mask & session_mask
        valid_short = (signals_df["direction"] == -1) & conf_mask & session_mask

        entries = valid_long
        exits   = valid_short

        # ATR-based SL/TP using config multipliers
        sl_stop = (signals_df["atr"] * _cfg.stop_atr_multiplier  / close).clip(0.002, 0.05)
        tp_stop = (signals_df["atr"] * _cfg.target_atr_multiplier / close).clip(0.002, 0.10)

        # Fixed fractional sizing: 5% of capital per trade
        size = capital * 0.05 / close

        pf = vbt.Portfolio.from_signals(
            close         = close,
            entries       = entries,
            exits         = exits,
            short_entries = valid_short,
            short_exits   = valid_long,
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
            "live_ready":    sharpe > 1.5 and max_dd < 10 and win_rate > 52 and profit_factor > 1.4,
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
    """Bar-by-bar backtest with correct Sharpe computed on equity curve returns."""
    equity       = capital
    peak         = capital
    max_dd       = 0.0
    trades:  list[dict] = []
    equity_curve: list[float] = []   # equity at every bar — used for Sharpe

    position = None
    for ts, row in signals_df.iterrows():
        price = float(close.loc[ts])

        # ── Check exits ────────────────────────────────────────────────────────
        if position:
            d     = position["direction"]
            entry = position["entry"]
            qty   = position["qty"]
            stop  = position["stop"]
            tgt   = position["target"]

            hit_stop   = (d == 1 and price <= stop)   or (d == -1 and price >= stop)
            hit_target = (d == 1 and price >= tgt)    or (d == -1 and price <= tgt)

            if hit_stop or hit_target:
                gross = d * (price - entry) * qty
                costs = price * qty * fee_pct
                pnl   = gross - costs
                trades.append({"pnl": pnl, "exit": "TARGET" if hit_target else "STOP"})
                equity  += pnl
                position = None

        # ── EOD forced exit at 15:25 ───────────────────────────────────────────
        if position and hasattr(ts, 'time') and ts.time() >= __import__('datetime').time(15, 25):
            d     = position["direction"]
            gross = d * (price - position["entry"]) * position["qty"]
            costs = price * position["qty"] * fee_pct
            pnl   = gross - costs
            trades.append({"pnl": pnl, "exit": "EOD"})
            equity  += pnl
            position = None

        # ── New entry: session window 09:45–15:10, confidence >= 0.55 ─────────
        import datetime as _dt
        _t = ts.time() if hasattr(ts, 'time') else None
        in_session = _t is not None and _dt.time(9, 45) <= _t < _dt.time(15, 10)
        conf_ok    = float(row.get("confidence", 1.0)) >= 0.55
        if position is None and row.get("direction", 0) != 0 and conf_ok and in_session:
            qty  = max(1, int(capital * 0.05 / price))
            cost = price * qty * fee_pct
            equity -= cost
            position = {
                "direction": int(row["direction"]),
                "entry":     price,
                "stop":      float(row["stop"])   if pd.notna(row.get("stop"))   else price * 0.98,
                "target":    float(row["target"]) if pd.notna(row.get("target")) else price * 1.03,
                "qty":       qty,
            }

        # ── Live equity (mark open position to market) ─────────────────────────
        if position:
            d   = position["direction"]
            mtm = d * (price - position["entry"]) * position["qty"]
            live_eq = equity + mtm
        else:
            live_eq = equity

        equity_curve.append(live_eq)
        peak   = max(peak, live_eq)
        max_dd = max(max_dd, (peak - live_eq) / peak * 100)

    if not trades:
        return {"total_trades": 0, "sharpe": 0, "max_dd_pct": round(max_dd, 2),
                "total_ret_pct": 0, "win_rate_pct": 0, "profit_factor": 0,
                "live_ready": False, "bars": len(close), "signals": 0}

    # ── Sharpe on per-trade returns (correct for intraday) ────────────────────
    # Use each trade's PnL as a fraction of capital at entry.
    # Annualise by estimated trades per year from the observed rate.
    pnl_series  = pd.Series([t["pnl"] for t in trades])
    trade_rets  = pnl_series / capital           # fraction of capital per trade
    n_days      = (signals_df.index[-1] - signals_df.index[0]).days or 1
    trades_pa   = max(len(trades) / n_days * 252, 1)   # estimated trades per year
    if len(trade_rets) > 1 and trade_rets.std() > 0:
        excess  = trade_rets.mean() - RISK_FREE_RATE / trades_pa
        sharpe  = float(excess / trade_rets.std() * np.sqrt(trades_pa))
    else:
        sharpe  = 0.0

    pnls   = [t["pnl"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    return {
        "total_trades":  len(trades),
        "sharpe":        round(sharpe, 3),
        "max_dd_pct":    round(max_dd, 2),
        "total_ret_pct": round((equity - capital) / capital * 100, 2),
        "win_rate_pct":  round(len(wins) / len(trades) * 100, 1),
        "profit_factor": round(sum(wins) / abs(sum(losses)), 3) if losses else float("inf"),
        "live_ready":    sharpe > 1.5 and max_dd < 10 and len(wins) / len(trades) > 0.52 and (sum(wins) / abs(sum(losses)) if losses else float("inf")) > 1.4,
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
