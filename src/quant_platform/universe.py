"""Load local OHLCV files and run a shared-capital universe backtest."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from .data import canonicalize, coverage
from .portfolio import PortfolioConfig, backtest_portfolio
from .strategy import StrategyConfig


def _bound(value: str | None, *, is_end: bool) -> pd.Timestamp | None:
    if not value:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Kolkata")
    else:
        timestamp = timestamp.tz_convert("Asia/Kolkata")
    if is_end and len(value) == 10:
        timestamp += pd.Timedelta(days=1)
    return timestamp


def load_local_universe(
    root: str | Path = "data/raw",
    interval: str = "15m",
    start: str | None = None,
    end: str | None = None,
    symbol_limit: int | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, dict[str, Any]]]:
    """Load local files and return a coverage inventory for every discovered symbol."""
    base = Path(root)
    lower, upper = _bound(start, is_end=False), _bound(end, is_end=True)
    files = sorted(base.glob(f"*/{interval}/bars.csv")) if base.exists() else []
    if symbol_limit is not None:
        files = files[:symbol_limit]
    data: dict[str, pd.DataFrame] = {}
    validation: dict[str, dict[str, Any]] = {}
    inventory: list[dict[str, Any]] = []
    for path in files:
        symbol = path.parent.parent.name
        try:
            bars, report = canonicalize(pd.read_csv(path))
            if lower is not None:
                bars = bars.loc[bars.index >= lower]
            if upper is not None:
                bars = bars.loc[bars.index < upper]
            item = {"symbol": symbol, "path": str(path), **coverage(bars), **report.to_dict()}
            if bars.empty:
                item.update(status="SKIPPED", reason="no_rows_in_selected_date_range")
            else:
                data[symbol] = bars
                validation[symbol] = report.to_dict()
                item.update(status="READY", reason=None)
        except Exception as exc:
            item = {"symbol": symbol, "path": str(path), "rows": 0, "start": None, "end": None,
                    "status": "SKIPPED", "reason": f"{type(exc).__name__}: {exc}"}
        inventory.append(item)
    return data, pd.DataFrame(inventory), validation


def run_local_universe(
    root: str | Path,
    interval: str,
    strategy: StrategyConfig,
    portfolio: PortfolioConfig,
    start: str | None = None,
    end: str | None = None,
    symbol_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    data, inventory, validation = load_local_universe(root, interval, start, end, symbol_limit)
    if not data:
        raise ValueError("No valid local OHLCV files match the selected interval and date range.")
    trades, rejected, equity = backtest_portfolio(data, strategy, portfolio)
    return trades, rejected, equity, inventory, validation


def strategy_snapshot(strategy: StrategyConfig, portfolio: PortfolioConfig) -> dict[str, dict[str, Any]]:
    return {"strategy_config": asdict(strategy), "portfolio_config": asdict(portfolio)}


def input_data_manifest(inventory: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Return an immutable content manifest for the files used by a saved experiment."""
    rows: list[dict[str, Any]] = []
    for item in inventory.loc[inventory.status.eq("READY")].sort_values("symbol").itertuples():
        path = Path(item.path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"symbol": item.symbol, "path": str(path), "rows": item.rows,
                     "start": item.start, "end": item.end, "sha256": digest})
    manifest = pd.DataFrame(rows)
    canonical = manifest.to_csv(index=False, lineterminator="\n") if not manifest.empty else ""
    return manifest, hashlib.sha256(canonical.encode("utf-8")).hexdigest()
