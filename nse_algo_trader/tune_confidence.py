"""Confidence threshold grid search — finds min_confidence that maximises Sharpe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

SYMBOL   = "NSE:NIFTY50-INDEX"
INTERVAL = 5
CAPITAL  = 1_000_000

THRESHOLDS = [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75]

def run_grid():
    from production.strategy.backtest import load_ohlcv_and_features, _manual_backtest
    from production.strategy.momentum import MomentumStrategy, load_config

    print(f"Loading {SYMBOL} {INTERVAL}min data...")
    ohlcv, feats = load_ohlcv_and_features(SYMBOL, INTERVAL)
    common = feats.index.intersection(ohlcv.index)
    feats  = feats.loc[common]
    close  = ohlcv.loc[common, "close"]

    vix_series = None
    vpath = Path(f"data/ohlcv/NSE_INDIAVIX_INDEX_{INTERVAL}min.parquet")
    if vpath.exists():
        vdf = pq.read_table(vpath).to_pandas()
        vdf.index = pd.to_datetime(vdf.index, utc=True).tz_convert("Asia/Kolkata")
        vix_series = vdf["close"].reindex(common)

    costs = yaml.safe_load(Path("configs/broker.yaml").read_text()).get("transaction_costs", {})
    fee   = costs.get("exchange_fee_pct", 0.00345)/100 + costs.get("slippage_pct", 0.02)/100

    cfg = load_config()
    # generate all signals once (unfiltered), filter by confidence in backtest
    cfg.min_confidence = 0.0
    all_signals = MomentumStrategy(cfg).generate_signals_df(feats, close, vix=vix_series)

    results = []
    for thresh in THRESHOLDS:
        # mask signals below threshold
        signals = all_signals.copy()
        mask = signals["confidence"] < thresh
        signals.loc[mask, "direction"] = 0

        r = _manual_backtest(signals, close, CAPITAL, fee, INTERVAL)
        results.append({
            "conf":    thresh,
            "sharpe":  r["sharpe"],
            "pf":      r["profit_factor"],
            "wr":      r["win_rate_pct"],
            "ret":     r["total_ret_pct"],
            "dd":      r["max_dd_pct"],
            "trades":  r["total_trades"],
            "signals": r["signals"],
        })
        s = results[-1]
        targets_met = sum([s["sharpe"] >= 1.5, s["dd"] <= 10,
                           s["wr"] >= 52.0, s["pf"] >= 1.4])
        print(f"  conf={thresh:.2f}  trades={s['trades']:>3}  "
              f"sharpe={s['sharpe']:>7.3f}  pf={s['pf']:.3f}  "
              f"wr={s['wr']:.1f}%  ret={s['ret']:+.2f}%  {targets_met}/4")

    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)

    print("\n" + "="*75)
    print("  CONFIDENCE GRID — sorted by Sharpe")
    print("="*75)
    print(f"  {'conf':>5}  {'trades':>6}  {'sharpe':>8}  {'PF':>6}  {'WR%':>5}  {'ret%':>6}  {'DD%':>5}  pass")
    print("-"*75)
    targets = {"sharpe": 1.5, "dd": 10.0, "wr": 52.0, "pf": 1.4}
    for _, row in df.iterrows():
        passed = sum([row.sharpe >= targets["sharpe"], row.dd <= targets["dd"],
                      row.wr >= targets["wr"], row.pf >= targets["pf"]])
        flag = " ◄ BEST" if row.name == df.index[0] else ""
        star = " ★ READY" if passed == 4 else f" {passed}/4"
        print(f"  {row.conf:>5.2f}  {int(row.trades):>6}  {row.sharpe:>8.3f}  "
              f"{row.pf:>6.3f}  {row.wr:>5.1f}  {row.ret:>+6.2f}  {row.dd:>5.2f} {star}{flag}")

    best = df.iloc[0]
    best_passed = sum([best.sharpe >= 1.5, best.dd <= 10, best.wr >= 52.0, best.pf >= 1.4])
    print("\n" + "="*75)
    print(f"  RECOMMENDATION: min_confidence={best.conf}")
    print(f"  Sharpe {best.sharpe:.3f}  PF {best.pf:.3f}  WR {best.wr:.1f}%  "
          f"Ret {best.ret:+.2f}%  DD {best.dd:.2f}%  Trades {int(best.trades)}  ({best_passed}/4 targets)")
    print("="*75)
    return df

if __name__ == "__main__":
    run_grid()
