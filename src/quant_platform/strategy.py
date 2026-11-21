"""Configurable reproduction of the 15-minute pivot/RSI baseline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    initial_capital: float = 10_000
    risk_fraction: float = .005
    reward_risk: float = 1.85
    rsi_period: int = 14
    rsi_long: float = 45
    rsi_short: float = 40
    entry_start: str = "10:00"
    last_entry: str = "14:00"
    square_off: str = "15:15"
    interval_minutes: int = 15
    max_trades_per_symbol_per_day: int = 1

    def __post_init__(self):
        if self.initial_capital <= 0 or not 0 < self.risk_fraction <= 1 or self.reward_risk <= 0 or self.rsi_period < 2:
            raise ValueError("Invalid capital, risk, reward/risk, or RSI configuration")
        if (not 0 <= self.rsi_short <= 100 or not 0 <= self.rsi_long <= 100
                or self.interval_minutes < 1 or self.max_trades_per_symbol_per_day < 1):
            raise ValueError("Invalid RSI threshold, interval, or maximum daily trades configuration")
        start = _time(self.entry_start); last = _time(self.last_entry); squareoff = _time(self.square_off)
        if not start <= last < squareoff:
            raise ValueError("Require entry_start <= last_entry < square_off")


def _time(value: str) -> time:
    return time.fromisoformat(value)


def prepare_signals(bars: pd.DataFrame, cfg: StrategyConfig = StrategyConfig()) -> pd.DataFrame:
    """Create signal-at-close rows with next-bar execution references; input timestamps denote bar opens."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(bars.columns):
        raise ValueError(f"Required columns: {sorted(required)}")
    df = bars.copy().sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Bars must have a DatetimeIndex")
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")
    delta = df.Close.diff().fillna(0.0)
    gain = delta.clip(lower=0).rolling(cfg.rsi_period, min_periods=cfg.rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(cfg.rsi_period, min_periods=cfg.rsi_period).mean()
    ratio = gain.div(loss.where(loss != 0))
    rsi = 100 - 100 / (1 + ratio)
    rsi = rsi.mask((loss == 0) & (gain > 0), 100).mask((loss == 0) & (gain == 0), 50)
    df["RSI"] = rsi
    session = pd.Series(df.index.date, index=df.index)
    daily = df.groupby(session).agg(High=("High", "max"), Low=("Low", "min"), Close=("Close", "last"))
    prev = daily.shift(1)
    prev.index = pd.Index(daily.index)
    keys = pd.Series(df.index.date, index=df.index)
    df["Prev_High"] = keys.map(prev.High).to_numpy()
    df["Prev_Low"] = keys.map(prev.Low).to_numpy()
    df["Prev_Close"] = keys.map(prev.Close).to_numpy()
    pivot = (df.Prev_High + df.Prev_Low + df.Prev_Close) / 3
    df["BC"] = (df.Prev_High + df.Prev_Low) / 2
    df["TC"] = 2 * pivot - df.BC
    prev_close = df.Close.shift(1)
    hm = df.index.hour * 60 + df.index.minute
    start = _time(cfg.entry_start).hour * 60 + _time(cfg.entry_start).minute
    last = _time(cfg.last_entry).hour * 60 + _time(cfg.last_entry).minute
    eligible = (hm >= start) & (hm <= last)
    long = (df.RSI > cfg.rsi_long) & (df.Close > df.TC) & (df.Close > df.Prev_High) & (prev_close <= df.Prev_High)
    short = (df.RSI < cfg.rsi_short) & (df.Close < df.BC) & (df.Close < df.Prev_Low) & (prev_close >= df.Prev_Low)
    df["Signal"] = ""
    df.loc[eligible & long, "Signal"] = "LONG"
    df.loc[eligible & short, "Signal"] = "SHORT"
    df["Entry_Time"] = df.index.to_series().shift(-1)
    df["Entry_Open"] = df.Open.shift(-1)
    df["Stop"] = df.Low.shift(1).where(df.Signal.eq("LONG"), df.High.shift(1))
    df.loc[df.Signal.eq(""), "Stop"] = float("nan")
    return df


def backtest_symbol(bars: pd.DataFrame, cfg: StrategyConfig = StrategyConfig(), symbol: str = "UNKNOWN") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic single-symbol simulation; conservatively resolves same-bar stop/target and models stop gaps."""
    df = prepare_signals(bars, cfg)
    trades, rejected = [], []
    entries_by_day: dict[object, int] = {}
    for i, row in enumerate(df.itertuples()):
        if not row.Signal:
            continue
        day = row.Index.date()
        reason = None
        if entries_by_day.get(day, 0) >= cfg.max_trades_per_symbol_per_day: reason = "max_trades_per_symbol_day"
        elif i + 1 >= len(df) or pd.isna(row.Entry_Open): reason = "no_next_bar"
        elif df.index[i + 1].date() != day: reason = "next_bar_outside_session"
        if reason:
            rejected.append({"symbol": symbol, "timestamp": row.Index.isoformat(), "reason": reason})
            continue
        entry_i = i + 1
        entry = float(row.Entry_Open)
        stop = float(row.Stop)
        side = 1 if row.Signal == "LONG" else -1
        risk = side * (entry - stop)
        if not risk > 0:
            rejected.append({"symbol": symbol, "timestamp": row.Index.isoformat(), "reason": "non_positive_entry_risk"})
            continue
        qty = int(cfg.initial_capital * cfg.risk_fraction // risk)
        qty = min(qty, int(cfg.initial_capital // entry)) if side == 1 else qty
        if qty < 1:
            rejected.append({"symbol": symbol, "timestamp": row.Index.isoformat(), "reason": "quantity_below_one"})
            continue
        target = entry + side * cfg.reward_risk * risk
        exit_i, exit_px, exit_reason = None, None, None
        entry_day = df.index[entry_i].date()
        sqoff = _time(cfg.square_off)
        for j in range(entry_i, len(df)):
            ts, bar = df.index[j], df.iloc[j]
            if ts.date() != entry_day:
                break
            if (ts.hour * 60 + ts.minute + cfg.interval_minutes) > sqoff.hour * 60 + sqoff.minute:
                continue
            stop_hit = bar.Low <= stop if side == 1 else bar.High >= stop
            target_hit = bar.High >= target if side == 1 else bar.Low <= target
            if stop_hit:
                exit_i, exit_px, exit_reason = j, (min(stop, float(bar.Open)) if side == 1 else max(stop, float(bar.Open))), "stop" if not target_hit else "stop_and_target_same_bar_conservative_stop"
                break
            if target_hit:
                exit_i, exit_px, exit_reason = j, target, "target"
                break
        if exit_i is None:
            valid = [j for j in range(entry_i, len(df)) if df.index[j].date() == entry_day and (df.index[j].hour * 60 + df.index[j].minute + cfg.interval_minutes) <= sqoff.hour * 60 + sqoff.minute]
            if not valid:
                rejected.append({"symbol": symbol, "timestamp": row.Index.isoformat(), "reason": "no_completed_bar_before_square_off"})
                continue
            exit_i = valid[-1]
            exit_px, exit_reason = float(df.iloc[exit_i].Close), "square_off"
        gross = side * (exit_px - entry) * qty
        trades.append({"symbol": symbol, "side": row.Signal, "signal_time": row.Index.isoformat(), "entry_time": df.index[entry_i].isoformat(), "entry_price": entry, "stop_price": stop, "target_price": target, "exit_time": df.index[exit_i].isoformat(), "exit_price": exit_px, "exit_reason": exit_reason, "quantity": qty, "gross_pnl": gross, "net_pnl": gross, "initial_risk": qty * risk})
        entries_by_day[day] = entries_by_day.get(day, 0) + 1
    return pd.DataFrame(trades), pd.DataFrame(rejected)
