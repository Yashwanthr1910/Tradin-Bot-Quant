"""Retryable, checkpointed local downloads; never substitutes fabricated data."""
from __future__ import annotations
import json,time
import hashlib
from pathlib import Path
import pandas as pd
from .data import canonicalize,coverage

def download_symbol(provider,symbol:str,interval:str,start:str,end:str,root:Path=Path("data/raw"),retries:int=3,throttle_seconds:float=1.0)->dict:
    folder=root/symbol/interval; folder.mkdir(parents=True,exist_ok=True)
    bars_path=folder/"bars.csv"; status_path=folder/"status.json"
    old=pd.read_csv(bars_path,index_col=0,parse_dates=True) if bars_path.exists() else pd.DataFrame()
    request_start=start
    if not old.empty:
        try:
            old.index=pd.DatetimeIndex(pd.to_datetime(old.index,utc=True)).tz_convert("Asia/Kolkata")
            old,_=canonicalize(old)
            requested_start = pd.Timestamp(start).normalize()
            # Append only when the existing file already reaches the requested start.
            # If this run asks for older history, fetch from its true start so that a
            # short legacy file cannot masquerade as a complete long-range dataset.
            if old.index.min().normalize() <= requested_start:
                # Re-fetch the last observed local date to cover provider end-point quirks;
                # merge below deduplicates the overlap.
                request_start=max(start,str(old.index.max().date()))
        except Exception:
            old=pd.DataFrame()
    error=None
    for attempt in range(retries+1):
        try:
            time.sleep(throttle_seconds if attempt==0 else 0)
            fetched=provider.fetch(symbol,interval,request_start,end)
            merged=pd.concat([old,fetched]).sort_index() if not old.empty else fetched
            clean,report=canonicalize(merged)
            clean.to_csv(bars_path)
            digest=hashlib.sha256(bars_path.read_bytes()).hexdigest()
            result={"symbol":symbol,"interval":interval,"requested_start":start,"request_start_this_run":request_start,"requested_end_exclusive":end,"downloaded_at_utc":pd.Timestamp.now(tz="UTC").isoformat(),"status":"COMPLETED","coverage":coverage(clean),"validation":report.to_dict(),"provider":type(provider).__name__,"adjusted_prices":False,"timestamp_label":"bar_open","timezone":"Asia/Kolkata","dataset_sha256":digest}
            status_path.write_text(json.dumps(result,indent=2),encoding="utf-8")
            return result
        except Exception as exc:
            error=f"{type(exc).__name__}: {exc}"
            if attempt<retries: time.sleep(min(2**attempt,16))
    result={"symbol":symbol,"interval":interval,"requested_start":start,"requested_end_exclusive":end,"status":"FAILED","error":error,"checkpoint":str(bars_path) if bars_path.exists() else None}
    status_path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result
