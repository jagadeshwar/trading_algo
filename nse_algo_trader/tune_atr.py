"""ATR multiplier grid search — finds best stop/target combo for momentum strategy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import yaml

SYMBOL       = "NSE:NIFTY50-INDEX"
INTERVAL     = 5
CAPITAL      = 1_000_000
CONFIG_PATH  = Path("configs/strategy.yaml")

STOP_MULTS   = [1.5, 2.0, 2.5, 3.0, 3.5]
TARGET_MULTS = [3.0, 4.0, 5.0, 6.0]

def run_grid():
    from production.strategy.backtest import load_ohlcv_and_features, _manual_backtest
    from production.strategy.momentum import MomentumStrategy, load_config
    import pyarrow.parquet as pq

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

    costs_cfg = yaml.safe_load(Path("configs/broker.yaml").read_text()).get("transaction_costs", {})
    fee = costs_cfg.get("exchange_fee_pct", 0.00345) / 100 + costs_cfg.get("slippage_pct", 0.02) / 100

    cfg = load_config()

    results = []
    total = len(STOP_MULTS) * len(TARGET_MULTS)
    done  = 0

    for stop_m in STOP_MULTS:
        for tgt_m in TARGET_MULTS:
            if tgt_m <= stop_m:          # skip inverted R:R
                done += 1
                continue
            rr = tgt_m / stop_m

            cfg.stop_atr_multiplier   = stop_m
            cfg.target_atr_multiplier = tgt_m
            strategy = MomentumStrategy(cfg)
            signals  = strategy.generate_signals_df(feats, close, vix=vix_series)
            r        = _manual_backtest(signals, close, CAPITAL, fee, INTERVAL)
            results.append({
                "stop":    stop_m,
                "target":  tgt_m,
                "rr":      round(rr, 2),
                "sharpe":  r["sharpe"],
                "pf":      r["profit_factor"],
                "wr":      r["win_rate_pct"],
                "ret":     r["total_ret_pct"],
                "dd":      r["max_dd_pct"],
                "trades":  r["total_trades"],
            })
            done += 1
            s = results[-1]
            print(f"  [{done:>2}/{total}] stop={stop_m:.1f} tgt={tgt_m:.1f}  "
                  f"R:R={rr:.1f}  sharpe={s['sharpe']:>7.3f}  pf={s['pf']:.3f}  "
                  f"wr={s['wr']:.1f}%  ret={s['ret']:+.2f}%  trades={s['trades']}")

    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)

    print("\n" + "="*80)
    print("  ATR MULTIPLIER GRID RESULTS — sorted by Sharpe")
    print("="*80)
    print(f"  {'stop':>5}  {'target':>6}  {'R:R':>4}  {'sharpe':>8}  {'PF':>6}  {'WR%':>5}  {'ret%':>6}  {'DD%':>5}  trades")
    print("-"*80)
    for _, row in df.iterrows():
        flag = " ◄ BEST" if row.name == df.index[0] else ""
        print(f"  {row.stop:>5.1f}  {row.target:>6.1f}  {row.rr:>4.1f}  "
              f"{row.sharpe:>8.3f}  {row.pf:>6.3f}  {row.wr:>5.1f}  "
              f"{row.ret:>+6.2f}  {row.dd:>5.2f}  {int(row.trades):>6}{flag}")

    best = df.iloc[0]
    print("\n" + "="*80)
    print(f"  RECOMMENDATION: stop_atr_multiplier={best.stop}  target_atr_multiplier={best.target}")
    print(f"  Sharpe {best.sharpe:.3f}  PF {best.pf:.3f}  WR {best.wr:.1f}%  "
          f"Ret {best.ret:+.2f}%  DD {best.dd:.2f}%  Trades {int(best.trades)}")
    print("="*80)
    return df

if __name__ == "__main__":
    run_grid()
