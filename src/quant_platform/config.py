"""Load typed runtime configuration from the checked-in JSON config."""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import fields
from .strategy import StrategyConfig

def load_settings(path: str | Path = "config/settings.json") -> tuple[dict, StrategyConfig]:
    raw=json.loads(Path(path).read_text(encoding="utf-8"))
    data=raw.get("data",{})
    if data.get("timestamp_semantics") != "bar_open":
        raise ValueError("Only timestamp_semantics='bar_open' is currently supported")
    allowed={field.name for field in fields(StrategyConfig)}
    values={key:value for key,value in raw.get("strategy",{}).items() if key in allowed}
    interval=data.get("interval","15m")
    values["interval_minutes"]=60 if interval in ("1h","60m") else int(interval[:-1]) if interval.endswith("m") else values.get("interval_minutes",15)
    strategy=StrategyConfig(**values)
    return raw, strategy
