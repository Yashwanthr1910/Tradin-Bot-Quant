"""Download locally authenticated FYERS historical candles for the reference universe."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_platform.config import load_settings
from quant_platform.downloads import download_symbol
from quant_platform.providers import FyersProvider
from quant_platform.secrets import load_local_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FYERS candles for the configured NIFTY 500 reference universe.")
    parser.add_argument("--config", type=Path, default=Path("config/settings.json"))
    parser.add_argument("--interval", required=True, choices=sorted(FyersProvider.RESOLUTIONS))
    parser.add_argument("--start", required=True); parser.add_argument("--end", required=True)
    parser.add_argument("--symbols", nargs="*"); parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--request-delay-seconds", type=float, default=1.6, help="Delay between FYERS 100-day requests")
    args = parser.parse_args()
    if args.workers != 1:
        parser.error("FYERS Standard permits 50 history requests per minute; use --workers 1 for a complete, reliable universe download")
    if args.request_delay_seconds < 1.5:
        parser.error("--request-delay-seconds must be at least 1.5 for FYERS Standard's 50-requests-per-minute limit")
    settings, _ = load_settings(args.config)
    symbols = args.symbols
    if not symbols:
        constituents = pd.read_csv(settings["universe"]["constituents_file"])
        column = "ProviderSymbol" if "ProviderSymbol" in constituents.columns else "Symbol"
        symbols = constituents[column].dropna().astype(str).unique().tolist()
    secrets = load_local_env()
    provider = FyersProvider(secrets.get("FYERS_APP_ID", ""), secrets.get("FYERS_ACCESS_TOKEN", ""), args.request_delay_seconds)
    metadata = Path("data/metadata")
    metadata.mkdir(parents=True, exist_ok=True)
    lock = metadata / f"fyers_{args.interval}_download_in_progress.json"
    lock.write_text(json.dumps({"provider": "FYERS", "interval": args.interval, "started_at_utc": datetime.now(timezone.utc).isoformat(), "symbols_requested": len(symbols)}, indent=2), encoding="utf-8")
    try:
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download_symbol, provider, symbol, args.interval, args.start, args.end): symbol for symbol in symbols}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"symbol": symbol, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
                results.append(result)
                print(json.dumps(result), flush=True)
        results.sort(key=lambda item: item["symbol"])
        summary = {"provider": "FYERS", "interval": args.interval, "requested_start": args.start, "requested_end_exclusive": args.end, "symbols_requested": len(symbols), "completed": sum(item["status"] == "COMPLETED" for item in results), "failed": sum(item["status"] != "COMPLETED" for item in results), "results": results}
        output = metadata / f"fyers_{args.interval}_last_download_batch.json"
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
