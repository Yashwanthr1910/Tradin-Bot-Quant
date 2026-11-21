from __future__ import annotations
import argparse
from pathlib import Path
from quant_platform.providers import YahooFinanceProvider
from quant_platform.downloads import download_symbol
from quant_platform.config import load_settings

def main():
    p=argparse.ArgumentParser(description="Optional yfinance acquisition; provider limits may truncate intraday history")
    p.add_argument("symbol"); p.add_argument("--interval"); p.add_argument("--start"); p.add_argument("--end")
    p.add_argument("--output",type=Path)
    a=p.parse_args(); provider=YahooFinanceProvider(); config,_=load_settings()
    interval=a.interval or config["data"]["interval"]; start=a.start or config["data"]["start"]; end=a.end or config["data"]["end_exclusive"]
    result=download_symbol(provider,a.symbol,interval,start,end,root=(a.output.parent if a.output else Path("data/raw")))
    if a.output and result.get("status")=="COMPLETED":
        source=(a.output.parent/a.symbol/interval/"bars.csv") if a.output else Path("data/raw")/a.symbol/interval/"bars.csv"
        if source.resolve()!=a.output.resolve(): a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_bytes(source.read_bytes())
    print(result)

if __name__ == "__main__": main()
