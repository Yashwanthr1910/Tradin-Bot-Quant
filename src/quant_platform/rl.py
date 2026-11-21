"""Offline, research-only tabular Q-learning gate for baseline strategy signals.

A valid baseline signal is one episode: the agent accepts or rejects it. State
features are known at the signal close. The reward is observed only after the
unchanged baseline stop/target/square-off exit, so future prices are never inputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .strategy import StrategyConfig, _time, prepare_signals

MODEL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RLConfig:
    """Reproducible configuration for offline Q-learning research."""
    train_fraction: float = 0.70
    epochs: int = 20
    learning_rate: float = 0.10
    exploration_rate: float = 0.10
    min_state_visits: int = 3
    seed: int = 7

    def __post_init__(self) -> None:
        if not 0.50 <= self.train_fraction < 1:
            raise ValueError("train_fraction must be in [0.50, 1.0)")
        if self.epochs < 1 or not 0 < self.learning_rate <= 1 or not 0 <= self.exploration_rate <= 1:
            raise ValueError("Invalid RL training configuration")
        if self.min_state_visits < 1:
            raise ValueError("min_state_visits must be at least one")


def _bucket(value: float, edges: tuple[float, ...]) -> int:
    return -1 if not np.isfinite(value) else int(np.searchsorted(np.asarray(edges), value, side="right"))


def _state(row: pd.Series, cfg: StrategyConfig) -> str:
    """Encode only values known when the signal candle closes."""
    side = str(row["Signal"])
    direction = 1.0 if side == "LONG" else -1.0
    trigger = float(row["Prev_High"] if side == "LONG" else row["Prev_Low"])
    cpr = float(row["TC"] if side == "LONG" else row["BC"])
    close = float(row["Close"])
    rsi_strength = direction * (float(row["RSI"]) - 50.0)
    breakout_bps = direction * (close - trigger) / close * 10_000 if close else np.nan
    cpr_bps = direction * (close - cpr) / close * 10_000 if close else np.nan
    minute = row.name.hour * 60 + row.name.minute
    start = _time(cfg.entry_start).hour * 60 + _time(cfg.entry_start).minute
    last = _time(cfg.last_entry).hour * 60 + _time(cfg.last_entry).minute
    session_progress = (minute - start) / max(1, last - start)
    return "|".join((
        side,
        f"rsi{_bucket(rsi_strength, (0, 5, 15, 30))}",
        f"break{_bucket(breakout_bps, (0, 5, 15, 40))}",
        f"cpr{_bucket(cpr_bps, (0, 5, 15, 40))}",
        f"atr{_bucket(float(row['ATR_pct']) * 10_000, (20, 50, 100, 200))}",
        f"volume{_bucket(float(row['Volume_ratio']), (0.7, 1.0, 1.5, 2.5))}",
        f"time{_bucket(session_progress, (0.33, 0.67))}",
    ))


def signal_outcomes(bars: pd.DataFrame, strategy: StrategyConfig, symbol: str) -> pd.DataFrame:
    """Build unit-risk outcomes while preserving the baseline entry and exit rules."""
    frame = prepare_signals(bars, strategy)
    true_range = pd.concat((frame.High - frame.Low, (frame.High - frame.Close.shift()).abs(), (frame.Low - frame.Close.shift()).abs()), axis=1).max(axis=1)
    frame["ATR_pct"] = true_range.rolling(14, min_periods=14).mean().div(frame.Close)
    frame["Volume_ratio"] = frame.Volume.div(frame.Volume.rolling(20, min_periods=5).mean())
    square_off = _time(strategy.square_off)
    records: list[dict[str, Any]] = []
    for i, row in enumerate(frame.itertuples()):
        if not row.Signal or i + 1 >= len(frame) or pd.isna(row.Entry_Open):
            continue
        entry_time = frame.index[i + 1]
        if entry_time.date() != row.Index.date():
            continue
        entry, stop = float(row.Entry_Open), float(row.Stop)
        side = 1.0 if row.Signal == "LONG" else -1.0
        risk = side * (entry - stop)
        if risk <= 0:
            continue
        target = entry + side * strategy.reward_risk * risk
        exit_price: float | None = None
        exit_reason: str | None = None
        for j in range(i + 1, len(frame)):
            timestamp, bar = frame.index[j], frame.iloc[j]
            if timestamp.date() != entry_time.date():
                break
            if timestamp.hour * 60 + timestamp.minute + strategy.interval_minutes > square_off.hour * 60 + square_off.minute:
                continue
            stop_hit = bar.Low <= stop if side == 1 else bar.High >= stop
            target_hit = bar.High >= target if side == 1 else bar.Low <= target
            if stop_hit:
                exit_price = min(stop, float(bar.Open)) if side == 1 else max(stop, float(bar.Open))
                exit_reason = "stop" if not target_hit else "stop_and_target_same_bar_conservative_stop"
                break
            if target_hit:
                exit_price, exit_reason = target, "target"
                break
        if exit_price is None:
            valid = [j for j in range(i + 1, len(frame)) if frame.index[j].date() == entry_time.date() and frame.index[j].hour * 60 + frame.index[j].minute + strategy.interval_minutes <= square_off.hour * 60 + square_off.minute]
            if not valid:
                continue
            exit_price, exit_reason = float(frame.iloc[valid[-1]].Close), "square_off"
        records.append({
            "symbol": symbol, "signal_time": row.Index, "entry_time": entry_time, "side": row.Signal,
            "state": _state(frame.iloc[i], strategy), "reward_r": float(side * (exit_price - entry) / risk),
            "exit_reason": exit_reason,
        })
    return pd.DataFrame(records)


def build_outcomes(data: dict[str, pd.DataFrame], strategy: StrategyConfig) -> pd.DataFrame:
    rows = [signal_outcomes(bars, strategy, symbol) for symbol, bars in sorted(data.items())]
    rows = [item for item in rows if not item.empty]
    if not rows:
        return pd.DataFrame(columns=["symbol", "signal_time", "entry_time", "side", "state", "reward_r", "exit_reason"])
    return pd.concat(rows, ignore_index=True).sort_values(["signal_time", "symbol"], kind="stable").reset_index(drop=True)


def chronological_split(outcomes: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if outcomes.empty:
        raise ValueError("No valid baseline signal outcomes are available for RL training.")
    days = pd.Index(pd.to_datetime(outcomes.signal_time).dt.date.unique()).sort_values()
    if len(days) < 3:
        raise ValueError("RL training needs at least three trading sessions with valid signals.")
    cutoff = days[min(len(days) - 1, max(1, int(len(days) * train_fraction)))]
    signal_days = pd.to_datetime(outcomes.signal_time).dt.date
    train, evaluation = outcomes.loc[signal_days < cutoff].copy(), outcomes.loc[signal_days >= cutoff].copy()
    if train.empty or evaluation.empty:
        raise ValueError("Chronological split produced an empty train or evaluation period.")
    return train, evaluation, str(cutoff)


def train_q_learning(train: pd.DataFrame, config: RLConfig) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Train an epsilon-greedy, gamma=0 tabular Q-learning contextual bandit."""
    rng = np.random.default_rng(config.seed)
    values: dict[str, dict[str, float]] = {}
    visits: dict[str, int] = {}
    records = train.reset_index(drop=True)
    for _ in range(config.epochs):
        for index in rng.permutation(len(records)):
            record = records.iloc[int(index)]
            state = str(record.state)
            q = values.setdefault(state, {"REJECT": 0.0, "ACCEPT": 0.0})
            visits[state] = visits.get(state, 0) + 1
            action = ("REJECT", "ACCEPT")[int(rng.integers(0, 2))] if rng.random() < config.exploration_rate else ("ACCEPT" if q["ACCEPT"] > q["REJECT"] else "REJECT")
            reward = float(record.reward_r) if action == "ACCEPT" else 0.0
            q[action] += config.learning_rate * (reward - q[action])
    return values, visits


def evaluate_policy(outcomes: pd.DataFrame, values: dict[str, dict[str, float]], visits: dict[str, int], min_state_visits: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    evaluated = outcomes.copy()
    evaluated["state_visits"] = evaluated.state.map(visits).fillna(0).astype(int)
    evaluated["q_reject"] = evaluated.state.map(lambda state: values.get(str(state), {}).get("REJECT", 0.0))
    evaluated["q_accept"] = evaluated.state.map(lambda state: values.get(str(state), {}).get("ACCEPT", 0.0))
    evaluated["policy_action"] = np.where((evaluated.state_visits >= min_state_visits) & (evaluated.q_accept > evaluated.q_reject), "ACCEPT", "REJECT")
    accepted = evaluated.loc[evaluated.policy_action.eq("ACCEPT")]
    baseline_r, policy_r = float(evaluated.reward_r.sum()), float(accepted.reward_r.sum())
    return evaluated, {
        "candidate_signals": int(len(evaluated)), "accepted_signals": int(len(accepted)),
        "acceptance_rate_pct": float(len(accepted) / len(evaluated) * 100) if len(evaluated) else 0.0,
        "baseline_total_r": baseline_r, "policy_total_r": policy_r,
        "baseline_average_r": float(evaluated.reward_r.mean()) if len(evaluated) else None,
        "policy_average_r": float(accepted.reward_r.mean()) if len(accepted) else None,
        "baseline_win_rate_pct": float((evaluated.reward_r > 0).mean() * 100) if len(evaluated) else None,
        "policy_win_rate_pct": float((accepted.reward_r > 0).mean() * 100) if len(accepted) else None,
        "policy_outperformed_baseline_r": policy_r > baseline_r,
        "evaluation_note": "Unit-risk candidate comparison only; not a shared-capital portfolio backtest and not evidence of live-trading suitability.",
    }


def train_offline_policy(data: dict[str, pd.DataFrame], strategy: StrategyConfig, config: RLConfig) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    outcomes = build_outcomes(data, strategy)
    train, evaluation, cutoff = chronological_split(outcomes, config.train_fraction)
    values, visits = train_q_learning(train, config)
    evaluated, report = evaluate_policy(evaluation, values, visits, config.min_state_visits)
    metadata: dict[str, Any] = {
        "schema_version": MODEL_SCHEMA_VERSION, "model_type": "offline_tabular_q_learning_contextual_bandit",
        "research_only": True, "live_order_execution": False,
        "continuous_learning_policy": "Manual or scheduled retraining from validated local historical data. A candidate must pass a fresh chronological evaluation before review; no model is auto-promoted to live trading.",
        "strategy_config": asdict(strategy), "rl_config": asdict(config), "trained_candidate_count": int(len(train)),
        "evaluation_candidate_count": int(len(evaluation)), "train_end_exclusive_session": cutoff,
        "state_count": int(len(values)), "q_values": values, "state_visits": visits, "evaluation": report,
        "methodology": "State uses signal-time side, RSI strength, breakout and CPR distance, ATR percentage, relative volume, and session timing. Reward is the later R-multiple under unchanged baseline exits. Reject reward is zero. Gamma is zero because each episode is one accept/reject decision.",
    }
    return metadata, train, evaluated


def save_policy(output_root: str | Path, metadata: dict[str, Any], train: pd.DataFrame, evaluation: pd.DataFrame) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    train.to_csv(root / "training_candidates.csv", index=False)
    evaluation.to_csv(root / "evaluation_candidates.csv", index=False)
    return root