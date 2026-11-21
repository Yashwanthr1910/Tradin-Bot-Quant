from __future__ import annotations
import argparse, json, uuid
from pathlib import Path
import pandas as pd
from quant_platform.data import canonicalize
from quant_platform.strategy import StrategyConfig, backtest_symbol
from quant_platform.config import load_settings
from quant_platform.metrics import summarize
from quant_platform.provenance import git_state,sha256_file

def main():
    p=argparse.ArgumentParser(description="Run baseline on a validated local OHLCV CSV (no network access)")
    p.add_argument("csv", type=Path); p.add_argument("--symbol", default="LOCAL"); p.add_argument("--capital", type=float)
    p.add_argument("--risk", type=float); p.add_argument("--reward-risk", type=float); p.add_argument("--rsi-period", type=int)
    p.add_argument("--config", type=Path, default=Path("config/settings.json"))
    a=p.parse_args()
    raw=pd.read_csv(a.csv)
    bars, report=canonicalize(raw)
    _, defaults=load_settings(a.config)
    cfg=StrategyConfig(initial_capital=a.capital if a.capital is not None else defaults.initial_capital,risk_fraction=a.risk if a.risk is not None else defaults.risk_fraction,reward_risk=a.reward_risk if a.reward_risk is not None else defaults.reward_risk,rsi_period=a.rsi_period if a.rsi_period is not None else defaults.rsi_period, rsi_long=defaults.rsi_long,rsi_short=defaults.rsi_short,entry_start=defaults.entry_start,last_entry=defaults.last_entry,square_off=defaults.square_off,interval_minutes=defaults.interval_minutes)
    trades,rejected=backtest_symbol(bars,cfg,a.symbol)
    exp_id=uuid.uuid4().hex[:12]; out=Path("results/backtests")/exp_id; out.mkdir(parents=True)
    metrics=summarize(trades,cfg.initial_capital)
    payload={"experiment_id":exp_id,"strategy":"legacy_15m_pivot_rsi_corrected_v1","config":cfg.__dict__,"input":str(a.csv),"dataset_sha256":sha256_file(a.csv),"code":git_state(Path.cwd()),"validation":report.to_dict(),"metrics":metrics,"limitations":["Single-symbol simulation only; shared portfolio capital and transaction costs are not modeled.","Trade-close equity frequency; no daily mark-to-market series.","No live data download was performed."]}
    (out/"experiment.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    trades.to_csv(out/"trades.csv",index=False); rejected.to_csv(out/"rejected_signals.csv",index=False)
    print(json.dumps(payload,indent=2))

if __name__ == "__main__": main()
