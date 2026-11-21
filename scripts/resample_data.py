from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from quant_platform.data import canonicalize,coverage
from quant_platform.resampling import resample_complete

def main():
    p=argparse.ArgumentParser(description="Build complete session-aligned bars from supported lower-frequency data")
    p.add_argument("csv",type=Path); p.add_argument("--source-minutes",type=int,required=True); p.add_argument("--target-minutes",type=int,required=True); p.add_argument("--symbol",required=True); p.add_argument("--output",type=Path)
    a=p.parse_args(); raw=pd.read_csv(a.csv); clean,quality=canonicalize(raw); result=resample_complete(clean,a.source_minutes,a.target_minutes)
    output=a.output or Path("data/resampled")/a.symbol/f"{a.target_minutes}m"/"bars.csv"; output.parent.mkdir(parents=True,exist_ok=True); result.to_csv(output)
    meta={"symbol":a.symbol,"source_file":str(a.csv),"source_interval_minutes":a.source_minutes,"target_interval_minutes":a.target_minutes,"method":"session anchored, complete source groups only","source_validation":quality.to_dict(),"output_coverage":coverage(result),"output_file":str(output)}
    output.with_suffix(".metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); print(json.dumps(meta,indent=2))

if __name__=="__main__":main()
