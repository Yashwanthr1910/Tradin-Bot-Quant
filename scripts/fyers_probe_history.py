"""Probe actual FYERS history availability before scheduling a large universe job."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_platform.data import coverage
from quant_platform.providers import FyersProvider
from quant_platform.secrets import load_local_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe FYERS candle coverage for one NSE symbol and all requested intervals.")
    parser.add_argument("--symbol", default="SBIN.NS")
    parser.add_argument("--intervals", nargs="+", default=["1m", "5m", "15m", "30m", "1h", "2h", "4h"])
    parser.add_argument("--years", nargs="+", type=int, default=[2010, 2020, 2021, 2022, 2023, 2024, 2025, 2026])
    parser.add_argument("--output", type=Path, default=Path("data/metadata/fyers_coverage_probe.json"))
    parser.add_argument("--request-delay-seconds", type=float, default=0.25,
                        help="Pause between probe requests; FYERS Standard permits at most 5 requests per second.")
    args = parser.parse_args()
    settings = load_local_env()
    if args.request_delay_seconds < 0.2:
        parser.error("--request-delay-seconds must be at least 0.2 to stay within FYERS Standard's request limit")
    provider = FyersProvider(settings.get("FYERS_APP_ID", ""), settings.get("FYERS_ACCESS_TOKEN", ""))
    results = []
    for interval in args.intervals:
        for year in args.years:
            try:
                bars = provider.fetch(args.symbol, interval, f"{year}-01-01", f"{year}-01-31")
                results.append({"symbol": args.symbol, "interval": interval, "year": year, "status": "AVAILABLE", "coverage": coverage(bars)})
            except Exception as exc:
                results.append({"symbol": args.symbol, "interval": interval, "year": year, "status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"})
            time.sleep(args.request_delay_seconds)
    earliest: dict[str, str | None] = {}
    for interval in args.intervals:
        available = [item for item in results if item["interval"] == interval and item["status"] == "AVAILABLE"]
        earliest[interval] = min((item["coverage"]["start"] for item in available), default=None)
    report = {"symbol": args.symbol, "years_probed": args.years, "probe": results, "earliest_returned_bar_by_interval": earliest}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "earliest_returned_bar_by_interval": earliest}, indent=2))


if __name__ == "__main__":
    main()
