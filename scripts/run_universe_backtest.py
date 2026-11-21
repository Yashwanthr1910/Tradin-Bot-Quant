from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_platform.config import load_settings
from quant_platform.metrics import summarize_portfolio
from quant_platform.portfolio import PortfolioConfig
from quant_platform.provenance import git_state
from quant_platform.strategy import StrategyConfig
from quant_platform.universe import input_data_manifest, run_local_universe


INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the baseline strategy across every available local universe file.")
    parser.add_argument("--config", type=Path, default=Path("config/settings.json"))
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--start", help="Inclusive local date or timestamp, for example 2026-09-01")
    parser.add_argument("--end", help="Inclusive local date or timestamp, for example 2026-10-23")
    parser.add_argument("--symbol-limit", type=int, help="Optional deterministic limit for a quick smoke test")
    parser.add_argument("--capital", type=float); parser.add_argument("--risk", type=float)
    parser.add_argument("--reward-risk", type=float); parser.add_argument("--rsi-period", type=int)
    parser.add_argument("--rsi-long", type=float); parser.add_argument("--rsi-short", type=float)
    parser.add_argument("--entry-start"); parser.add_argument("--last-entry"); parser.add_argument("--square-off")
    parser.add_argument("--max-trades-per-stock-per-day", type=int)
    parser.add_argument("--max-positions", type=int); parser.add_argument("--fee-bps", type=float)
    parser.add_argument("--slippage-bps", type=float)
    args = parser.parse_args()

    raw, defaults = load_settings(args.config)
    values = asdict(defaults)
    if args.interval not in INTERVAL_MINUTES:
        parser.error(f"Unsupported interval {args.interval!r}; choose one of {sorted(INTERVAL_MINUTES)}")
    for field, value in {"initial_capital": args.capital, "risk_fraction": args.risk,
                         "reward_risk": args.reward_risk, "rsi_period": args.rsi_period,
                         "rsi_long": args.rsi_long, "rsi_short": args.rsi_short,
                         "entry_start": args.entry_start, "last_entry": args.last_entry,
                         "square_off": args.square_off,
                         "max_trades_per_symbol_per_day": args.max_trades_per_stock_per_day}.items():
        if value is not None:
            values[field] = value
    values["interval_minutes"] = INTERVAL_MINUTES[args.interval]
    strategy = StrategyConfig(**values)
    pvalues = raw.get("portfolio", {}).copy()
    for field, value in {"max_open_positions": args.max_positions, "fee_bps_per_side": args.fee_bps,
                         "slippage_bps_per_side": args.slippage_bps}.items():
        if value is not None:
            pvalues[field] = value
    portfolio = PortfolioConfig(**pvalues)

    trades, rejected, equity, inventory, validation = run_local_universe(
        args.root, args.interval, strategy, portfolio, args.start, args.end, args.symbol_limit
    )
    manifest, data_fingerprint = input_data_manifest(inventory)
    metrics = summarize_portfolio(trades, equity, strategy.initial_capital)
    run_id = uuid.uuid4().hex[:12]
    output = Path("results/backtests") / run_id
    output.mkdir(parents=True, exist_ok=False)
    ready = inventory.loc[inventory.status.eq("READY"), "symbol"].sort_values().tolist()
    payload = {
        "experiment_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "legacy_15m_pivot_rsi_corrected_v1", "mode": "local_universe_shared_capital",
        "strategy_config": asdict(strategy), "portfolio_config": asdict(portfolio),
        "universe": raw.get("universe", {}), "data_root": str(args.root), "interval": args.interval,
        "selected_start": args.start, "selected_end_inclusive": args.end, "symbols": ready,
        "symbols_loaded": len(ready), "symbols_skipped": int((inventory.status != "READY").sum()),
        "validation": validation, "data_fingerprint_sha256": data_fingerprint,
        "metrics": metrics, "code": git_state(Path.cwd()),
        "execution_policy": "shared capital; exits precede entries; simultaneous candidates resolve by symbol order; one entry per symbol per session; next-bar-open fills; stop wins same-bar collisions; adverse slippage",
    }
    (output / "experiment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest.to_csv(output / "input_data_manifest.csv", index=False)
    inventory.to_csv(output / "universe_inventory.csv", index=False)
    trades.to_csv(output / "trades.csv", index=False)
    rejected.to_csv(output / "rejected_signals.csv", index=False)
    equity.to_csv(output / "equity_curve.csv", index=False)
    print(json.dumps({"experiment_id": run_id, "output": str(output), "symbols_loaded": len(ready), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
