"""Train a research-only RL gate for existing baseline signals; no live orders."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from quant_platform.config import load_settings
from quant_platform.provenance import git_state
from quant_platform.rl import RLConfig, save_policy, train_offline_policy
from quant_platform.strategy import StrategyConfig
from quant_platform.universe import input_data_manifest, load_local_universe

INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an offline Q-learning gate for baseline signals. No live orders are sent.")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--interval", choices=INTERVAL_MINUTES, default="15m")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--train-fraction", type=float, default=.70)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=.10)
    parser.add_argument("--exploration-rate", type=float, default=.10)
    parser.add_argument("--min-state-visits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--symbol-limit", type=int)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    _, baseline = load_settings(project / "config" / "settings.json")
    strategy = StrategyConfig(**{**asdict(baseline), "interval_minutes": INTERVAL_MINUTES[args.interval]})
    config = RLConfig(args.train_fraction, args.epochs, args.learning_rate, args.exploration_rate, args.min_state_visits, args.seed)
    data, inventory, _ = load_local_universe(args.data_root, args.interval, args.start, args.end, args.symbol_limit)
    if not data:
        raise ValueError("No valid local files match the requested interval and range.")
    metadata, train, evaluation = train_offline_policy(data, strategy, config)
    manifest, fingerprint = input_data_manifest(inventory)
    run_id = uuid.uuid4().hex[:12]
    destination = project / "results" / "rl_models" / run_id
    metadata.update({
        "run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(), "interval": args.interval,
        "selected_start": args.start, "selected_end_inclusive": args.end, "symbols_loaded": len(data),
        "data_fingerprint_sha256": fingerprint, "code": git_state(project),
        "portfolio_note": "This evaluates a per-signal gate only. It is not connected to the shared-capital simulator.",
    })
    save_policy(destination, metadata, train, evaluation)
    manifest.to_csv(destination / "input_data_manifest.csv", index=False)
    inventory.to_csv(destination / "universe_inventory.csv", index=False)
    print(json.dumps({"run_id": run_id, "saved_to": str(destination), "evaluation": metadata["evaluation"], "research_only": True}, indent=2))


if __name__ == "__main__":
    main()