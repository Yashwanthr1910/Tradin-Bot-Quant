"""Strict session-anchored resampling of complete intraday bars only."""
from __future__ import annotations
import pandas as pd

SUPPORTED_TARGETS={10:5,120:60,240:60}

def resample_complete(frame:pd.DataFrame,source_minutes:int,target_minutes:int,session_open:str="09:15")->pd.DataFrame:
    """Aggregate complete, aligned bar-open groups; partial groups are dropped, never padded."""
    if target_minutes not in SUPPORTED_TARGETS or source_minutes!=SUPPORTED_TARGETS[target_minutes]:
        raise ValueError(f"Only source intervals 5->10, 60->120, and 60->240 minutes are currently supported (got {source_minutes}->{target_minutes})")
    if target_minutes%source_minutes: raise ValueError("Target interval must be divisible by source interval")
    data=frame.copy().sort_index()
    if not isinstance(data.index,pd.DatetimeIndex): raise ValueError("DatetimeIndex required")
    data.index=data.index.tz_localize("Asia/Kolkata") if data.index.tz is None else data.index.tz_convert("Asia/Kolkata")
    count=target_minutes//source_minutes
    open_minutes=int(session_open[:2])*60+int(session_open[3:])
    result=[]
    for day,group in data.groupby(data.index.date):
        offset=group.index.hour*60+group.index.minute-open_minutes
        bucket=(offset//target_minutes).astype(int)
        for _,part in group.groupby(bucket):
            first=part.index[0]; start_offset=first.hour*60+first.minute-open_minutes
            if start_offset<0 or start_offset%target_minutes or start_offset+target_minutes>375 or len(part)!=count: continue
            expected=pd.date_range(first,periods=count,freq=f"{source_minutes}min",tz=first.tz)
            if not part.index.equals(expected): continue
            result.append({"timestamp":first,"Open":part.Open.iloc[0],"High":part.High.max(),"Low":part.Low.min(),"Close":part.Close.iloc[-1],"Volume":part.Volume.sum()})
    if not result: return pd.DataFrame(columns=["Open","High","Low","Close","Volume"],index=pd.DatetimeIndex([],tz="Asia/Kolkata",name="timestamp"))
    return pd.DataFrame(result).set_index("timestamp").sort_index()
