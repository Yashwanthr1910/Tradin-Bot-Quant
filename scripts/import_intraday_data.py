"""Import and merge local OHLCV exports into the universe data layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_platform.data import canonicalize, coverage
from quant_platform.provenance import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and merge local intraday CSV exports into data/raw/<SYMBOL>/<INTERVAL>/bars.csv"
    )
    parser.add_argument("--data", action="append", required=True, metavar="SYMBOL=CSV", help="Repeat once per symbol")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--source-name", default="local_import", help="Provider or vendor label recorded in status metadata")
    args = parser.parse_args()

    results = []
    for item in args.data:
        if "=" not in item:
            raise ValueError("Each --data value must use SYMBOL=CSV format")
        symbol, source_name = item.split("=", 1)
        source = Path(source_name)
        if not source.exists():
            placeholder = "path\\to" in source_name.lower().replace("/", "\\") or "your_" in source_name.lower()
            hint = (
                "Replace the example placeholder with a real CSV export path, for example: "
                "--data RELIANCE.NS=C:\\MarketData\\RELIANCE_15m.csv"
                if placeholder else
                "Check the file path and export the provider's 15-minute OHLCV CSV before importing it."
            )
            raise FileNotFoundError(f"CSV file was not found: {source}. {hint}")
        imported, import_report = canonicalize(pd.read_csv(source))
        folder = args.root / symbol / args.interval
        folder.mkdir(parents=True, exist_ok=True)
        bars_path = folder / "bars.csv"
        if bars_path.exists():
            current, _ = canonicalize(pd.read_csv(bars_path))
            merged, merge_report = canonicalize(pd.concat([current, imported]))
        else:
            merged, merge_report = imported, import_report
        merged.to_csv(bars_path)
        status = {
            "symbol": symbol, "interval": args.interval, "status": "COMPLETED",
            "provider": args.source_name, "imported_from": str(source.resolve()),
            "imported_rows": int(len(imported)), "coverage": coverage(merged),
            "validation": merge_report.to_dict(), "timestamp_label": "bar_open",
            "timezone": "Asia/Kolkata", "dataset_sha256": sha256_file(bars_path),
        }
        (folder / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        results.append(status)
    print(json.dumps({"imported": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
