from __future__ import annotations
import argparse,json,uuid
from pathlib import Path
import pandas as pd
from quant_platform.config import load_settings
from quant_platform.data import canonicalize
from quant_platform.portfolio import PortfolioConfig,backtest_portfolio
from quant_platform.metrics import summarize_portfolio
from quant_platform.provenance import git_state,sha256_file

def main():
    p=argparse.ArgumentParser(description="Run shared-capital portfolio simulation on local CSV bars")
    p.add_argument("--data",action="append",required=True,metavar="SYMBOL=CSV",help="Repeat once per symbol")
    p.add_argument("--config",type=Path,default=Path("config/settings.json")); p.add_argument("--max-positions",type=int)
    p.add_argument("--fee-bps",type=float); p.add_argument("--slippage-bps",type=float)
    a=p.parse_args(); raw_config,strategy=load_settings(a.config); data={}; validation={}
    for item in a.data:
        symbol,filename=item.split("=",1); raw=pd.read_csv(filename); data[symbol],report=canonicalize(raw); validation[symbol]=report.to_dict()
    pvalues=raw_config.get("portfolio",{}).copy()
    if a.max_positions is not None: pvalues["max_open_positions"]=a.max_positions
    if a.fee_bps is not None: pvalues["fee_bps_per_side"]=a.fee_bps
    if a.slippage_bps is not None: pvalues["slippage_bps_per_side"]=a.slippage_bps
    pcfg=PortfolioConfig(**pvalues)
    trades,rejected,equity=backtest_portfolio(data,strategy,pcfg); metrics=summarize_portfolio(trades,equity,strategy.initial_capital)
    run=uuid.uuid4().hex[:12]; out=Path("results/backtests")/run; out.mkdir(parents=True)
    payload={"experiment_id":run,"strategy":"legacy_15m_pivot_rsi_corrected_v1","strategy_config":strategy.__dict__,"portfolio_config":pcfg.__dict__,"symbols":sorted(data),"input_files":{symbol:{"path":str(filename),"sha256":sha256_file(filename)} for symbol,filename in (x.split("=",1) for x in a.data)},"code":git_state(Path.cwd()),"validation":validation,"metrics":metrics,"execution_policy":"chronological; exits before entries at each timestamp; symbol sort order for simultaneous entries; stop first on same-bar collision; adverse slippage on fills"}
    (out/"experiment.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    trades.to_csv(out/"trades.csv",index=False); rejected.to_csv(out/"rejected_signals.csv",index=False); equity.to_csv(out/"equity_curve.csv",index=False)
    print(json.dumps(payload,indent=2))

if __name__=="__main__":main()
