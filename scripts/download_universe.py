from __future__ import annotations
import argparse,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd
from quant_platform.config import load_settings
from quant_platform.downloads import download_symbol
from quant_platform.providers import YahooFinanceProvider

def main():
    p=argparse.ArgumentParser(description="Download configured symbols sequentially and retain per-symbol status")
    p.add_argument("--config",type=Path,default=Path("config/settings.json")); p.add_argument("--interval"); p.add_argument("--start"); p.add_argument("--end"); p.add_argument("--symbols",nargs="*"); p.add_argument("--workers",type=int,default=4)
    a=p.parse_args(); settings,_=load_settings(a.config)
    if a.symbols: symbols=a.symbols
    else:
        constituents=pd.read_csv(settings["universe"]["constituents_file"])
        column="ProviderSymbol" if "ProviderSymbol" in constituents.columns else "Symbol"
        symbols=constituents[column].dropna().astype(str).unique().tolist()
    if a.workers<1 or a.workers>12: p.error("--workers must be between 1 and 12")
    interval=a.interval or settings["data"]["interval"]; start=a.start or settings["data"]["start"]; end=a.end or settings["data"]["end_exclusive"]
    provider=YahooFinanceProvider(); results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(download_symbol,provider,symbol,interval,start,end):symbol for symbol in symbols}
        for future in as_completed(futures):
            try: result=future.result()
            except Exception as exc: result={"symbol":futures[future],"status":"FAILED","error":f"{type(exc).__name__}: {exc}"}
            results.append(result); print(json.dumps(result),flush=True)
    results.sort(key=lambda row:row.get("symbol",""))
    summary={"universe":settings.get("universe",{}),"interval":interval,"requested_start":start,"requested_end_exclusive":end,"symbols_requested":len(symbols),"completed":sum(x["status"]=="COMPLETED" for x in results),"failed":sum(x["status"]!="COMPLETED" for x in results),"results":results}
    out=Path("data/metadata")/"last_download_batch.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="results"},indent=2))

if __name__=="__main__":main()
