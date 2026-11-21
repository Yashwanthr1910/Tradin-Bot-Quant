"""Small deterministic shared-capital portfolio simulator for research use."""
from __future__ import annotations
from dataclasses import dataclass
import math
import pandas as pd
from .strategy import StrategyConfig, prepare_signals, _time


@dataclass(frozen=True)
class PortfolioConfig:
    max_open_positions: int = 5
    max_gross_exposure_fraction: float = 1.0
    max_symbol_exposure_fraction: float = 0.25
    fee_bps_per_side: float = 0.0
    slippage_bps_per_side: float = 0.0
    short_margin_fraction: float = 1.0

    def __post_init__(self):
        if self.max_open_positions < 1 or not 0 < self.max_gross_exposure_fraction <= 1 or not 0 < self.max_symbol_exposure_fraction <= 1:
            raise ValueError("Invalid portfolio limits")
        if self.fee_bps_per_side < 0 or self.slippage_bps_per_side < 0 or not 0 < self.short_margin_fraction <= 1:
            raise ValueError("Invalid cost or short-margin settings")


def backtest_portfolio(data: dict[str, pd.DataFrame], strategy: StrategyConfig = StrategyConfig(), portfolio: PortfolioConfig = PortfolioConfig()) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate shared equity chronologically; open positions exit same session only."""
    prepared = {symbol: prepare_signals(frame, strategy) for symbol, frame in data.items()}
    row_positions = {symbol: {timestamp: i for i, timestamp in enumerate(frame.index)} for symbol, frame in prepared.items()}
    row_values = {symbol: list(frame.itertuples()) for symbol, frame in prepared.items()}
    timeline = sorted(set().union(*(set(frame.index) for frame in prepared.values()))) if prepared else []
    last_eligible = {}
    for sym, frame in prepared.items():
        eligible = frame.loc[(frame.index.hour * 60 + frame.index.minute + strategy.interval_minutes) <= _time(strategy.square_off).hour * 60 + _time(strategy.square_off).minute]
        for day, group in eligible.groupby(eligible.index.date):
            last_eligible[(sym, day)] = group.index[-1]
    capital = float(strategy.initial_capital)
    realized = 0.0
    positions = {}
    entries_by_symbol_day = {}
    trades, rejected, equity = [], [], []
    fee = portfolio.fee_bps_per_side / 10000
    slip = portfolio.slippage_bps_per_side / 10000
    sqoff = _time(strategy.square_off)

    def close_position(sym, ts, raw_exit, exit_reason):
        nonlocal realized
        pos = positions[sym]
        fill = float(raw_exit) * (1 - slip * pos["side"])
        exit_fee = abs(fill * pos["qty"]) * fee
        gross = pos["side"] * (float(raw_exit) - pos["entry_reference"]) * pos["qty"]
        slippage_cost = pos["entry_slippage"] + abs(fill - float(raw_exit)) * pos["qty"]
        pnl = gross - slippage_cost - pos["entry_fee"] - exit_fee
        realized += pnl
        trades.append({"symbol": sym, "side": "LONG" if pos["side"] == 1 else "SHORT", "signal_time": pos["signal_time"].isoformat(), "entry_time": pos["entry_time"].isoformat(), "entry_reference": pos["entry_reference"], "entry_price": pos["entry"], "stop_price": pos["stop"], "target_price": pos["target"], "exit_time": ts.isoformat(), "exit_reference": float(raw_exit), "exit_price": fill, "exit_reason": exit_reason, "quantity": pos["qty"], "gross_pnl": gross, "slippage_cost": slippage_cost, "fees": pos["entry_fee"] + exit_fee, "net_pnl": pnl, "initial_risk": pos["risk"] * pos["qty"]})
        del positions[sym]

    for ts in timeline:
        active = {sym: locations[ts] for sym, locations in row_positions.items() if ts in locations}
        bars = {sym: row_values[sym][position] for sym, position in active.items()}
        for sym in sorted(list(positions)):
            if sym not in bars:
                continue
            pos = positions[sym]
            bar = bars[sym]
            if ts.date() != pos["day"]:
                continue
            stop_hit = bar.Low <= pos["stop"] if pos["side"] == 1 else bar.High >= pos["stop"]
            target_hit = bar.High >= pos["target"] if pos["side"] == 1 else bar.Low <= pos["target"]
            if stop_hit:
                raw_exit = min(pos["stop"], bar.Open) if pos["side"] == 1 else max(pos["stop"], bar.Open)
                close_position(sym, ts, raw_exit, "stop_and_target_same_bar_conservative_stop" if target_hit else "stop")
            elif target_hit:
                close_position(sym, ts, pos["target"], "target")
            elif last_eligible.get((sym, pos["day"])) == ts:
                close_position(sym, ts, bar.Close, "square_off")
        for sym in sorted(bars):
            i = active[sym]
            if i < 1:
                continue
            signal = row_values[sym][i - 1]
            if not signal.Signal:
                continue
            day = ts.date()
            reason = None
            if entries_by_symbol_day.get((sym, day), 0) >= strategy.max_trades_per_symbol_per_day: reason = "max_trades_per_symbol_day"
            elif sym in positions: reason = "symbol_already_open"
            elif signal.Index.date() != day: reason = "signal_from_prior_session"
            elif last_eligible.get((sym, day)) is None or ts > last_eligible[(sym, day)]: reason = "no_completed_bar_before_square_off"
            elif len(positions) >= portfolio.max_open_positions: reason = "max_open_positions"
            if reason:
                rejected.append({"symbol": sym, "timestamp": signal.Index.isoformat(), "reason": reason})
                continue
            side = 1 if signal.Signal == "LONG" else -1
            raw_entry = float(bars[sym].Open)
            entry = raw_entry * (1 + slip * side)
            stop = float(signal.Stop)
            risk = side * (entry - stop)
            if risk <= 0:
                rejected.append({"symbol": sym, "timestamp": signal.Index.isoformat(), "reason": "non_positive_entry_risk"})
                continue
            unrealized = sum(p["side"] * (float(bars[s].Open if s in bars else prepared[s].Close.asof(ts)) - p["entry"]) * p["qty"] - p["entry_fee"] for s, p in positions.items())
            equity_now = max(0.0, capital + realized + unrealized)
            risk_budget = equity_now * strategy.risk_fraction
            gross = sum(abs(p["entry"] * p["qty"]) for p in positions.values())
            used_symbol = sum(abs(p["entry"] * p["qty"]) for s, p in positions.items() if s == sym)
            qty = min(math.floor(risk_budget / risk), math.floor(max(0, equity_now * portfolio.max_gross_exposure_fraction - gross) / entry), math.floor(max(0, equity_now * portfolio.max_symbol_exposure_fraction - used_symbol) / entry))
            if side == 1:
                long_notional = sum(p["entry"] * p["qty"] for p in positions.values() if p["side"] == 1)
                qty = min(qty, math.floor(max(0, equity_now - long_notional) / entry))
            else:
                short_notional = sum(p["entry"] * p["qty"] for p in positions.values() if p["side"] == -1)
                qty = min(qty, math.floor(max(0, equity_now * portfolio.short_margin_fraction - short_notional) / entry))
            if qty < 1:
                rejected.append({"symbol": sym, "timestamp": signal.Index.isoformat(), "reason": "shared_capital_or_exposure_limit"})
                continue
            target = entry + side * strategy.reward_risk * risk
            entry_fee = entry * qty * fee
            positions[sym] = {"side": side, "entry": entry, "entry_reference": raw_entry, "entry_slippage": abs(entry - raw_entry) * qty, "stop": stop, "target": target, "qty": qty, "risk": risk, "entry_fee": entry_fee, "entry_time": ts, "signal_time": signal.Index, "day": day}
            entries_by_symbol_day[(sym, day)] = entries_by_symbol_day.get((sym, day), 0) + 1
            bar = bars[sym]
            stop_hit = bar.Low <= stop if side == 1 else bar.High >= stop
            target_hit = bar.High >= target if side == 1 else bar.Low <= target
            if stop_hit:
                close_position(sym, ts, stop, "stop_and_target_same_bar_conservative_stop" if target_hit else "stop")
            elif target_hit:
                close_position(sym, ts, target, "target")
            elif last_eligible.get((sym, day)) == ts:
                close_position(sym, ts, bar.Close, "square_off")
        unrealized = 0.0
        gross = 0.0
        for sym, pos in positions.items():
            px = float(bars[sym].Close) if sym in bars else float(prepared[sym].Close.asof(ts))
            unrealized += pos["side"] * (px - pos["entry"]) * pos["qty"] - pos["entry_fee"]
            gross += abs(px * pos["qty"])
        equity.append({"timestamp": ts, "equity": capital + realized + unrealized, "realized_pnl": realized, "unrealized_pnl": unrealized, "gross_exposure": gross, "open_positions": len(positions)})
    return pd.DataFrame(trades), pd.DataFrame(rejected), pd.DataFrame(equity)
