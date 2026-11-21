"""Run resumable FYERS universe downloads for one or more supported timeframes."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quant_platform.providers import FyersProvider


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FYERS candles for all selected timeframes, one interval at a time.")
    parser.add_argument("--start", required=True, help="Inclusive date, first tried as supplied")
    parser.add_argument("--end", required=True, help="Exclusive date")
    parser.add_argument("--intervals", nargs="+", default=["1m", "5m", "15m", "30m", "1h", "2h", "4h"], choices=sorted(FyersProvider.RESOLUTIONS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--request-delay-seconds", type=float, default=1.6)
    parser.add_argument("--symbols", nargs="*", help="Optional .NS symbols; omit for the reference NIFTY 500 snapshot")
    args = parser.parse_args()
    script = Path(__file__).with_name("download_fyers_universe.py")
    for interval in args.intervals:
        command = [sys.executable, str(script), "--interval", interval, "--start", args.start, "--end", args.end,
                   "--workers", str(args.workers), "--request-delay-seconds", str(args.request_delay_seconds)]
        if args.symbols:
            command.extend(["--symbols", *args.symbols])
        print(f"Starting FYERS {interval} universe job.", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            raise SystemExit(f"FYERS {interval} job stopped with exit code {completed.returncode}.")


if __name__ == "__main__":
    main()
