from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from quant_platform.data import canonicalize, coverage

def main():
    parser = argparse.ArgumentParser(description="Validate a local OHLCV CSV and report actual coverage")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = pd.read_csv(args.csv)
    clean, report = canonicalize(raw)
    result = {"source": str(args.csv), "coverage": coverage(clean), "validation": report.to_dict()}
    out = args.output or Path("results/validation") / f"{args.csv.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    clean.to_csv(out.with_suffix(".validated.csv"))
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
