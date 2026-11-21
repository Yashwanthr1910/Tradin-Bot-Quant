"""Canonical OHLCV validation and provider-neutral storage helpers."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Protocol
import pandas as pd

REQUIRED = ("Open", "High", "Low", "Close", "Volume")


class MarketDataProvider(Protocol):
    def fetch(self, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame: ...


@dataclass
class ValidationReport:
    input_rows: int
    output_rows: int
    duplicate_timestamps: int
    invalid_ohlc: int
    missing_values: int
    negative_prices: int
    negative_volume: int
    zero_volume: int
    timezone_assumed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def canonicalize(frame: pd.DataFrame, timezone: str = "Asia/Kolkata") -> tuple[pd.DataFrame, ValidationReport]:
    """Normalize an OHLCV frame; naive timestamps are explicitly treated as local exchange time."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "Datetime" in frame.columns:
            frame = frame.set_index("Datetime")
        elif "Date" in frame.columns:
            frame = frame.set_index("Date")
        elif len(frame.columns) and (str(frame.columns[0]).startswith("Unnamed") or frame.columns[0] == "timestamp"):
            # Accept canonical CSVs written with the datetime index as the first field.
            frame = frame.set_index(frame.columns[0])
        else:
            raise ValueError("Input needs a DatetimeIndex or Datetime/Date column")
    data = frame.copy()
    data.index = pd.DatetimeIndex(pd.to_datetime(data.index))
    assumed = data.index.tz is None
    data.index = data.index.tz_localize(timezone) if assumed else data.index.tz_convert(timezone)
    data = data.rename(columns={str(c).lower(): str(c).title() for c in data.columns})
    missing_cols = set(REQUIRED) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing_cols)}")
    data = data.loc[:, list(REQUIRED)].apply(pd.to_numeric, errors="coerce").sort_index()
    duplicates = int(data.index.duplicated(keep="last").sum())
    data = data.loc[~data.index.duplicated(keep="last")]
    na_mask = data.isna().any(axis=1)
    invalid = (data.High < data.Low) | (data.Open < data.Low) | (data.Open > data.High) | (data.Close < data.Low) | (data.Close > data.High)
    negative = (data[list(REQUIRED[:4])] <= 0).any(axis=1)
    negative_volume = data.Volume < 0
    report = ValidationReport(len(frame), 0, duplicates, int(invalid.sum()), int(na_mask.sum()), int(negative.sum()), int(negative_volume.sum()), int((data.Volume == 0).sum()), assumed)
    # Invalid observations remain out of canonical output; no interpolation or silent repair.
    clean = data.loc[~(na_mask | invalid | negative | negative_volume)].copy()
    report.output_rows = len(clean)
    clean.index.name = "timestamp"
    return clean, report


def coverage(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"rows": 0, "start": None, "end": None}
    return {"rows": len(frame), "start": frame.index.min().isoformat(), "end": frame.index.max().isoformat()}
