"""Local Streamlit interface for reproducible, shared-capital research runs."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, time, timezone
import json
from pathlib import Path
import sys
import uuid

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import plotly.express as px
    import streamlit as st
except ImportError as exc:
    raise SystemExit("Install dashboard extras: python -m pip install -e .[dashboard]") from exc

from quant_platform.config import load_settings
from quant_platform.metrics import summarize_portfolio
from quant_platform.portfolio import PortfolioConfig
from quant_platform.provenance import git_state
from quant_platform.rl import RLConfig, save_policy, train_offline_policy
from quant_platform.strategy import StrategyConfig
from quant_platform.universe import input_data_manifest, load_local_universe, run_local_universe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "backtests"
RL_MODELS_ROOT = PROJECT_ROOT / "results" / "rl_models"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240}

st.set_page_config(page_title="NSE Quant Research", page_icon="▦", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1450px; padding-top: 2.2rem;}
[data-testid="stMetric"] {background: #101826; border: 1px solid #26364d; border-radius: 12px; padding: .9rem;}
.eyebrow {color: #61d4b3; font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;}
.hero {font-size: 2.35rem; font-weight: 700; margin: .1rem 0 .3rem 0;}
</style>""", unsafe_allow_html=True)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def local_interval_bounds(root: Path, interval: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Read persisted sidecars so date controls reflect actual local availability."""
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for path in root.glob(f"*/{interval}/status.json"):
        try:
            status = _read_json(path)
            if status.get("status") != "COMPLETED":
                continue
            span = status.get("coverage", {})
            if span.get("start") and span.get("end"):
                starts.append(pd.Timestamp(span["start"]))
                ends.append(pd.Timestamp(span["end"]))
        except (OSError, ValueError, TypeError):
            continue
    return (min(starts), max(ends)) if starts and ends else None


def active_download_locks() -> list[Path]:
    metadata = PROJECT_ROOT / "data" / "metadata"
    return sorted(metadata.glob("fyers_*_download_in_progress.json")) if metadata.exists() else []


@st.cache_data(show_spinner=False)
def experiment_index() -> pd.DataFrame:
    rows: list[dict] = []
    for path in RESULTS_ROOT.glob("*/experiment.json") if RESULTS_ROOT.exists() else []:
        try:
            run = _read_json(path)
            metrics = run.get("metrics", {})
            rows.append({
                "experiment_id": run.get("experiment_id", path.parent.name),
                "created_at_utc": run.get("created_at_utc", pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat()),
                "mode": run.get("mode", "single_symbol"),
                "symbols_loaded": run.get("symbols_loaded", len(run.get("symbols", []))),
                "trades": metrics.get("trades"), "net_pnl": metrics.get("net_pnl"),
                "total_return_pct": metrics.get("total_return_pct"), "max_drawdown_pct": metrics.get("max_drawdown_pct"),
            })
        except (OSError, ValueError, TypeError):
            continue
    return pd.DataFrame(rows).sort_values("created_at_utc", ascending=False) if rows else pd.DataFrame()



def metric_text(value: object, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.2f}{suffix}" if isinstance(value, float) else f"{value}{suffix}"


def selected_payload(run_id: str | None) -> tuple[dict | None, Path | None]:
    if not run_id:
        return None, None
    path = RESULTS_ROOT / run_id / "experiment.json"
    try:
        return _read_json(path), path.parent
    except (OSError, ValueError):
        return None, None


def save_universe_experiment(data_root: Path, interval: str, start: str, end: str, strategy: StrategyConfig, portfolio: PortfolioConfig, universe_metadata: dict) -> str:
    trades, rejected, equity, inventory, validation = run_local_universe(data_root, interval, strategy, portfolio, start, end)
    manifest, data_fingerprint = input_data_manifest(inventory)
    metrics = summarize_portfolio(trades, equity, strategy.initial_capital)
    run_id = uuid.uuid4().hex[:12]
    output = RESULTS_ROOT / run_id
    output.mkdir(parents=True, exist_ok=False)
    ready = inventory.loc[inventory.status.eq("READY"), "symbol"].sort_values().tolist()
    payload = {
        "experiment_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "legacy_15m_pivot_rsi_corrected_v1", "mode": "local_universe_shared_capital",
        "strategy_config": asdict(strategy), "portfolio_config": asdict(portfolio),
        "universe": universe_metadata, "data_root": str(data_root), "interval": interval,
        "selected_start": start, "selected_end_inclusive": end, "symbols": ready,
        "symbols_loaded": len(ready), "symbols_skipped": int((inventory.status != "READY").sum()),
        "validation": validation, "data_fingerprint_sha256": data_fingerprint,
        "metrics": metrics, "code": git_state(PROJECT_ROOT),
        "execution_policy": "Shared capital. Exits are processed before entries. Simultaneous candidates use symbol order. One entry per symbol per session. Entries fill at the next bar open. Stop wins if stop and target both occur inside one bar. Slippage is adverse.",
    }
    (output / "experiment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest.to_csv(output / "input_data_manifest.csv", index=False)
    inventory.to_csv(output / "universe_inventory.csv", index=False)
    trades.to_csv(output / "trades.csv", index=False)
    rejected.to_csv(output / "rejected_signals.csv", index=False)
    equity.to_csv(output / "equity_curve.csv", index=False)
    return run_id


raw_settings, defaults = load_settings(PROJECT_ROOT / "config" / "settings.json")
index = experiment_index()
run_ids = index.experiment_id.tolist() if not index.empty else []
if "pending_selected_run" in st.session_state:
    st.session_state["selected_run_picker"] = st.session_state.pop("pending_selected_run")
if run_ids and st.session_state.get("selected_run_picker") not in run_ids:
    st.session_state["selected_run_picker"] = run_ids[0]
available_bounds = local_interval_bounds(DEFAULT_DATA_ROOT, "15m")
available_start = available_bounds[0].date() if available_bounds else pd.Timestamp("2026-09-01").date()
available_end = available_bounds[1].date() if available_bounds else pd.Timestamp("2026-10-23").date()

with st.sidebar:
    st.markdown("### Research control")
    selected_run = st.selectbox("Saved experiment", options=run_ids, key="selected_run_picker") if run_ids else None
    st.caption("All results use local files. No live orders are sent.")
    if st.button("Refresh saved experiments", use_container_width=True):
        experiment_index.clear()
        st.rerun()

payload, folder = selected_payload(selected_run)
metrics = payload.get("metrics", {}) if payload else {}
st.markdown('<div class="eyebrow">Indian equities · systematic research</div>', unsafe_allow_html=True)
st.markdown('<div class="hero">NSE universe strategy lab</div>', unsafe_allow_html=True)
st.caption("15-minute pivot/RSI baseline · shared-capital simulation · simulated results, not live execution")
tabs = st.tabs(["Overview", "Run universe", "Performance", "Trades", "Data quality", "Experiments", "RL research"])

with tabs[0]:
    if not payload:
        st.info("No experiment is selected. Run the local universe after checking that market-data files are present.")
    else:
        a, b, c, d, e = st.columns(5)
        a.metric("Final equity", f"₹{metric_text(metrics.get('final_equity'))}")
        b.metric("Net P&L", f"₹{metric_text(metrics.get('net_pnl'))}")
        c.metric("Return", metric_text(metrics.get("total_return_pct"), "%"))
        d.metric("Maximum drawdown", metric_text(metrics.get("max_drawdown_pct"), "%"))
        e.metric("Closed trades", metric_text(metrics.get("trades")))
        st.subheader("Research run")
        left, right = st.columns(2)
        left.write({"experiment": payload.get("experiment_id"), "mode": payload.get("mode"), "symbols loaded": payload.get("symbols_loaded", len(payload.get("symbols", []))), "period": f"{payload.get('selected_start', 'full local file')} to {payload.get('selected_end_inclusive', 'full local file')}", "interval": payload.get("interval", "15m"), "input data fingerprint": (payload.get("data_fingerprint_sha256") or "legacy run: unavailable")[:16]})
        right.caption(payload.get("execution_policy", "Execution policy unavailable."))
        with st.expander("Exact saved parameters and data identity"):
            st.json({
                "strategy_config": payload.get("strategy_config", {}),
                "portfolio_config": payload.get("portfolio_config", {}),
                "data_fingerprint_sha256": payload.get("data_fingerprint_sha256", "legacy run: unavailable"),
            })
        if metrics.get("cagr_pct") is None:
            st.warning("CAGR and Calmar require at least 365 calendar days. Sharpe and Sortino use daily returns and require at least 30 daily observations.")

with tabs[1]:
    st.subheader("Run the whole local universe")
    interval = st.selectbox("Bar interval", options=list(INTERVAL_MINUTES), index=list(INTERVAL_MINUTES).index("15m"), key="run_interval")
    selected_bounds = local_interval_bounds(DEFAULT_DATA_ROOT, interval)
    has_data = selected_bounds is not None
    run_start = selected_bounds[0].date() if selected_bounds else pd.Timestamp("2026-09-01").date()
    run_end = selected_bounds[1].date() if selected_bounds else pd.Timestamp("2026-10-23").date()
    active_locks = active_download_locks()
    st.caption(f"Every discovered `data/raw/<symbol>/{interval}/bars.csv` file is considered. Signals compete for one shared capital account; portfolio limits select the entries that can be executed.")
    range_a, range_b, range_c = st.columns(3)
    range_a.metric(f"Earliest local {interval} bar", run_start.strftime("%d %b %Y") if has_data else "—")
    range_b.metric(f"Latest local {interval} bar", run_end.strftime("%d %b %Y") if has_data else "—")
    range_c.metric("Usable backtest window", f"{(run_end - run_start).days + 1} days" if has_data else "No local files")
    if active_locks:
        st.warning("FYERS data download is active. Backtests are disabled until its local files are complete, so an experiment always uses one fixed data snapshot.")
    elif has_data:
        st.info("The date controls use the local file coverage. Each saved experiment records a SHA-256 fingerprint of its exact input candles.")
    else:
        st.warning(f"No validated local {interval} files are available yet.")
    with st.expander("How to enable a one-year 15-minute backtest"):
        st.markdown("Export one year of 15-minute OHLCV data for each symbol from a licensed provider, then import it into the project. The importer validates timestamps and OHLC values, merges overlap with the existing files, and refreshes the dashboard range.")
        st.code("""$env:PYTHONPATH = \"src\"
& .\\.venv\\Scripts\\python.exe scripts\\import_intraday_data.py `
  --interval 15m `
  --source-name YOUR_PROVIDER `
  --data RELIANCE.NS=C:\\path\\to\\RELIANCE_15m.csv `
  --data TCS.NS=C:\\path\\to\\TCS_15m.csv""", language="powershell")
        st.caption("Required columns: Datetime (or Date), Open, High, Low, Close, Volume. The 15-minute strategy must be evaluated on 15-minute data; daily bars are a different strategy test.")
    with st.form("universe_run"):
        col1, col2, col3 = st.columns(3)
        data_root = col1.text_input("Local data root", value=str(DEFAULT_DATA_ROOT))
        max_positions = col2.number_input("Maximum open positions", min_value=1, max_value=100, value=int(raw_settings["portfolio"]["max_open_positions"]))
        max_trades = col3.number_input("Max trades per stock per day", min_value=1, max_value=50, value=int(defaults.max_trades_per_symbol_per_day))
        from_date = col1.date_input("Start date within local data", value=run_start, min_value=run_start, max_value=run_end, disabled=not has_data)
        to_date = col2.date_input("End date within local data", value=run_end, min_value=run_start, max_value=run_end, disabled=not has_data)
        capital = col3.number_input("Initial capital (₹)", min_value=1.0, value=float(defaults.initial_capital), step=1000.0)
        risk = col1.number_input("Risk per trade", min_value=0.001, max_value=0.10, value=float(defaults.risk_fraction), step=0.005, format="%.3f")
        reward_risk = col2.number_input("Reward / risk", min_value=0.1, value=float(defaults.reward_risk), step=0.1)
        rsi_period = col3.number_input("RSI period", min_value=2, max_value=100, value=int(defaults.rsi_period))
        rsi_long = col1.number_input("RSI long threshold", min_value=0.0, max_value=100.0, value=float(defaults.rsi_long), step=1.0)
        rsi_short = col2.number_input("RSI short threshold", min_value=0.0, max_value=100.0, value=float(defaults.rsi_short), step=1.0)
        entry_start = col3.time_input("Entry start time", value=time.fromisoformat(defaults.entry_start), step=60)
        last_entry = col1.time_input("Last entry time", value=time.fromisoformat(defaults.last_entry), step=60)
        square_off = col2.time_input("Square-off time", value=time.fromisoformat(defaults.square_off), step=60)
        fee = col3.number_input("Fee, bps per side", min_value=0.0, value=float(raw_settings["portfolio"]["fee_bps_per_side"]), step=0.1, format="%.2f")
        slippage = col1.number_input("Slippage, bps per side", min_value=0.0, value=float(raw_settings["portfolio"]["slippage_bps_per_side"]), step=0.1, format="%.2f")
        submit = st.form_submit_button("Run shared-capital universe backtest", type="primary", disabled=not has_data or bool(active_locks))
    if submit:
        try:
            if from_date > to_date:
                raise ValueError("Start date must not be after end date.")
            strategy = StrategyConfig(initial_capital=float(capital), risk_fraction=float(risk), reward_risk=float(reward_risk), rsi_period=int(rsi_period), rsi_long=float(rsi_long), rsi_short=float(rsi_short), entry_start=entry_start.strftime("%H:%M"), last_entry=last_entry.strftime("%H:%M"), square_off=square_off.strftime("%H:%M"), interval_minutes=INTERVAL_MINUTES[interval], max_trades_per_symbol_per_day=int(max_trades))
            portfolio = PortfolioConfig(max_open_positions=int(max_positions), max_gross_exposure_fraction=float(raw_settings["portfolio"]["max_gross_exposure_fraction"]), max_symbol_exposure_fraction=float(raw_settings["portfolio"]["max_symbol_exposure_fraction"]), fee_bps_per_side=float(fee), slippage_bps_per_side=float(slippage), short_margin_fraction=float(raw_settings["portfolio"]["short_margin_fraction"]))
            with st.spinner("Loading local bars, evaluating signals, and applying shared-capital limits…"):
                run_id = save_universe_experiment(Path(data_root), interval, str(from_date), str(to_date), strategy, portfolio, raw_settings.get("universe", {}))
            st.session_state["pending_selected_run"] = run_id
            experiment_index.clear()
            st.success(f"Saved universe experiment {run_id}.")
            st.rerun()
        except Exception as exc:
            st.error(f"Universe run failed: {type(exc).__name__}: {exc}")

with tabs[2]:
    if not payload or not folder:
        st.info("Select an experiment to inspect its equity curve and risk-adjusted metrics.")
    else:
        saved_metrics = dict(metrics)
        trade_file = folder / "trades.csv"
        if trade_file.exists() and ({"gross_profit", "gross_loss", "average_win", "average_loss", "average_win_loss_ratio", "expectancy_per_trade"} - saved_metrics.keys()):
            ledger = pd.read_csv(trade_file)
            pnl = ledger["net_pnl"] if "net_pnl" in ledger else pd.Series(dtype=float)
            wins, losses = pnl[pnl > 0], pnl[pnl < 0]
            saved_metrics.update({
                "gross_profit": float(wins.sum()), "gross_loss": float(losses.sum()),
                "average_win": float(wins.mean()) if not wins.empty else None,
                "average_loss": float(-losses.mean()) if not losses.empty else None,
                "average_win_loss_ratio": float(wins.mean() / -losses.mean()) if not wins.empty and not losses.empty else None,
                "expectancy_per_trade": float(pnl.mean()) if not pnl.empty else None,
            })
        metric_rows = pd.DataFrame([{
            "Metric": label,
            "Value": f"₹{metric_text(saved_metrics.get(key))}" if style == "currency" else metric_text(saved_metrics.get(key), style),
        } for label, key, style in [("Gross profit", "gross_profit", "currency"), ("Gross loss", "gross_loss", "currency"), ("Profit factor", "profit_factor", ""), ("Win rate", "win_rate_pct", "%"), ("Average win", "average_win", "currency"), ("Average loss", "average_loss", "currency"), ("Average win/loss ratio", "average_win_loss_ratio", ""), ("Expectancy per trade", "expectancy_per_trade", "currency"), ("CAGR", "cagr_pct", "%"), ("Sharpe", "sharpe", ""), ("Sortino", "sortino", ""), ("Calmar", "calmar", ""), ("Costs", "total_costs", "currency"), ("Turnover", "turnover", "currency"), ("Value at Risk, 95%", "var_95_daily", "%"), ("Expected shortfall, 95%", "expected_shortfall_95_daily", "%")]])
        st.dataframe(metric_rows, hide_index=True, use_container_width=True)
        curve_file = folder / "equity_curve.csv"
        if curve_file.exists():
            curve = pd.read_csv(curve_file)
            if not curve.empty:
                curve["timestamp"] = pd.to_datetime(curve["timestamp"], errors="coerce", utc=True)
                curve = curve.dropna(subset=["timestamp", "equity"]).sort_values("timestamp")
                curve["drawdown_pct"] = (curve.equity / curve.equity.cummax() - 1) * 100
                st.plotly_chart(px.line(curve, x="timestamp", y="equity", title="Marked-to-market portfolio equity"), use_container_width=True)
                st.plotly_chart(px.area(curve, x="timestamp", y="drawdown_pct", title="Drawdown (%)"), use_container_width=True)
                daily_equity = curve.set_index("timestamp")["equity"].resample("1D").last().dropna()
                daily_returns = daily_equity.pct_change().dropna()
                expanding_sharpe = (daily_returns.expanding(min_periods=2).mean().div(daily_returns.expanding(min_periods=2).std(ddof=1).replace(0, float("nan"))) * (252 ** 0.5)).dropna()
                if expanding_sharpe.empty:
                    st.caption("Sharpe through the full period needs at least three daily equity observations; this run does not yet have enough.")
                else:
                    sharpe_chart = px.line(expanding_sharpe.rename("Sharpe since start").reset_index(), x="timestamp", y="Sharpe since start", title="Sharpe ratio from backtest start through each trading day")
                    sharpe_chart.add_hline(y=0, line_dash="dash", line_color="#8b98a9")
                    st.plotly_chart(sharpe_chart, use_container_width=True)
                    st.caption(f"Calculated from every daily marked-to-market return since the run began, annualized using 252 trading days. Final chart value: {expanding_sharpe.iloc[-1]:.2f}.")
                st.subheader("Rolling Sharpe analysis")
                st.caption("Use this to compare recent 30-day and 60-day risk-adjusted performance and identify deterioration or regime changes.")
                earliest_day, latest_day = daily_equity.index[0].date(), daily_equity.index[-1].date()
                range_left, range_right = st.columns(2)
                analysis_start = range_left.date_input("Rolling-analysis start date", value=earliest_day, min_value=earliest_day, max_value=latest_day, key=f"rolling_sharpe_start_{selected_run}")
                analysis_end = range_right.date_input("Rolling-analysis end date", value=latest_day, min_value=earliest_day, max_value=latest_day, key=f"rolling_sharpe_end_{selected_run}")
                if analysis_start > analysis_end:
                    st.error("Rolling-analysis start date must not be after its end date.")
                else:
                    selected_equity = daily_equity.loc[(daily_equity.index.date >= analysis_start) & (daily_equity.index.date <= analysis_end)]
                    selected_returns = selected_equity.pct_change().dropna()
                    rolling_lines = []
                    insufficient_windows = []
                    for window in (30, 60):
                        series = (selected_returns.rolling(window, min_periods=window).mean().div(selected_returns.rolling(window, min_periods=window).std(ddof=1).replace(0, float("nan"))) * (252 ** 0.5)).dropna()
                        if series.empty:
                            insufficient_windows.append(window)
                        else:
                            line = series.rename("Sharpe").rename_axis("timestamp").reset_index()
                            line["Window"] = f"{window}-day"
                            rolling_lines.append(line)
                    if rolling_lines:
                        rolling_chart_data = pd.concat(rolling_lines, ignore_index=True)
                        rolling_chart = px.line(rolling_chart_data, x="timestamp", y="Sharpe", color="Window", title="30-day and 60-day rolling Sharpe ratio (annualized)")
                        rolling_chart.add_hline(y=0, line_dash="dash", line_color="#8b98a9")
                        st.plotly_chart(rolling_chart, use_container_width=True)
                    if insufficient_windows:
                        st.caption(f"The selected range does not have enough daily returns for: {', '.join(f'{window}-day' for window in insufficient_windows)} Sharpe.")
                    st.caption("Each point uses only the preceding 30 or 60 trading-day returns inside the chosen date range. Values above zero indicate positive risk-adjusted returns for that recent window.")
            else:
                st.info("The selected run produced no marked-to-market observations.")

with tabs[3]:
    if not folder:
        st.info("Select an experiment to inspect its trade ledger.")
    else:
        trade_file = folder / "trades.csv"
        trades = pd.read_csv(trade_file) if trade_file.exists() else pd.DataFrame()
        if trades.empty:
            st.info("No trades were executed for this configuration and data window.")
        else:
            symbols = ["All symbols", *sorted(trades.symbol.unique())]
            choice = st.selectbox("Symbol", symbols)
            shown = trades if choice == "All symbols" else trades.loc[trades.symbol.eq(choice)]
            a, b, c = st.columns(3)
            a.metric("Executed trades", len(shown)); b.metric("Net P&L", f"₹{shown.net_pnl.sum():,.2f}"); c.metric("Win rate", f"{(shown.net_pnl.gt(0).mean() * 100):.1f}%")
            st.dataframe(shown, use_container_width=True, hide_index=True)
            st.download_button("Download selected trades CSV", shown.to_csv(index=False).encode("utf-8"), "trades.csv", "text/csv")
        rejected_file = folder / "rejected_signals.csv"
        if rejected_file.exists():
            with st.expander("Rejected signals and capacity constraints"):
                st.dataframe(pd.read_csv(rejected_file), use_container_width=True, hide_index=True)

with tabs[4]:
    data, inventory, _ = load_local_universe(DEFAULT_DATA_ROOT, "15m")
    batch_path = PROJECT_ROOT / "data" / "metadata" / "last_download_batch.json"
    if batch_path.exists():
        batch = _read_json(batch_path)
        a, b, c = st.columns(3)
        a.metric("Snapshot symbols requested", batch.get("symbols_requested", "—")); b.metric("Provider downloads completed", batch.get("completed", "—")); c.metric("Provider failures", batch.get("failed", "—"))
    st.caption(f"{len(data)} local 15-minute files currently pass canonical validation. The displayed constituent snapshot date is {raw_settings.get('universe', {}).get('snapshot_page_updated', 'unknown')}.")
    st.dataframe(inventory, use_container_width=True, hide_index=True)

with tabs[5]:
    st.subheader("Saved strategy experiments")
    if index.empty:
        st.info("No saved experiments yet.")
    else:
        st.dataframe(index, use_container_width=True, hide_index=True)
        st.caption("Runs are append-only. Select an experiment in the sidebar to inspect its artifacts.")

with tabs[6]:
    st.subheader("Offline reinforcement-learning research")
    st.info("Research-only Q-learning signal gate. It can accept or reject a valid baseline signal, but cannot create signals, alter stops/targets, bypass capital limits, place orders, or self-promote itself to live trading.")
    model_rows = []
    for model_file in sorted(RL_MODELS_ROOT.glob("*/model.json"), reverse=True) if RL_MODELS_ROOT.exists() else []:
        try:
            model = _read_json(model_file)
            evaluation = model.get("evaluation", {})
            model_rows.append({"run_id": model.get("run_id", model_file.parent.name), "created_at_utc": model.get("created_at_utc"), "interval": model.get("interval"), "symbols": model.get("symbols_loaded"), "train candidates": model.get("trained_candidate_count"), "evaluation candidates": model.get("evaluation_candidate_count"), "accepted": evaluation.get("accepted_signals"), "policy total R": evaluation.get("policy_total_r"), "baseline total R": evaluation.get("baseline_total_r")})
        except (OSError, ValueError, TypeError):
            continue
    if model_rows:
        st.dataframe(pd.DataFrame(model_rows), hide_index=True, use_container_width=True)
    else:
        st.caption("No RL research models have been trained yet.")
    rl_interval = st.selectbox("RL research interval", list(INTERVAL_MINUTES), index=list(INTERVAL_MINUTES).index("15m"), key="rl_interval")
    rl_bounds = local_interval_bounds(DEFAULT_DATA_ROOT, rl_interval)
    if not rl_bounds:
        st.warning(f"No validated local {rl_interval} data is available for offline RL research.")
    else:
        rl_start, rl_end = rl_bounds[0].date(), rl_bounds[1].date()
        with st.form("rl_training"):
            first, second, third = st.columns(3)
            rl_from = first.date_input("RL start date", value=rl_start, min_value=rl_start, max_value=rl_end)
            rl_to = second.date_input("RL end date", value=rl_end, min_value=rl_start, max_value=rl_end)
            train_fraction = third.number_input("Training fraction", min_value=.50, max_value=.90, value=.70, step=.05)
            epochs = first.number_input("Q-learning epochs", min_value=1, max_value=200, value=20)
            learning_rate = second.number_input("Learning rate", min_value=.01, max_value=1.0, value=.10, step=.01)
            exploration_rate = third.number_input("Exploration rate", min_value=.0, max_value=1.0, value=.10, step=.01)
            min_state_visits = first.number_input("Minimum state visits", min_value=1, max_value=100, value=3)
            seed = second.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=7)
            train_rl = st.form_submit_button("Train and evaluate offline RL gate", type="primary")
        if train_rl:
            try:
                if rl_from > rl_to:
                    raise ValueError("RL start date must not be after end date.")
                data, inventory, _ = load_local_universe(DEFAULT_DATA_ROOT, rl_interval, str(rl_from), str(rl_to))
                rl_strategy = StrategyConfig(**{**asdict(defaults), "interval_minutes": INTERVAL_MINUTES[rl_interval]})
                rl_config = RLConfig(train_fraction=float(train_fraction), epochs=int(epochs), learning_rate=float(learning_rate), exploration_rate=float(exploration_rate), min_state_visits=int(min_state_visits), seed=int(seed))
                with st.spinner("Building causal signal states, training Q-values, and evaluating on later unseen sessions…"):
                    model, train_rows, evaluation_rows = train_offline_policy(data, rl_strategy, rl_config)
                    manifest, fingerprint = input_data_manifest(inventory)
                    run_id = uuid.uuid4().hex[:12]
                    destination = RL_MODELS_ROOT / run_id
                    model.update({"run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(), "interval": rl_interval, "selected_start": str(rl_from), "selected_end_inclusive": str(rl_to), "symbols_loaded": len(data), "data_fingerprint_sha256": fingerprint, "code": git_state(PROJECT_ROOT), "portfolio_note": "This is a per-signal research gate, not a shared-capital portfolio backtest."})
                    save_policy(destination, model, train_rows, evaluation_rows)
                    manifest.to_csv(destination / "input_data_manifest.csv", index=False)
                    inventory.to_csv(destination / "universe_inventory.csv", index=False)
                st.success(f"Saved RL research model {run_id}. It is not enabled for live trading or the portfolio backtest.")
                st.json(model["evaluation"])
            except Exception as exc:
                st.error(f"RL training failed: {type(exc).__name__}: {exc}")
    st.caption("Continuous learning is controlled retraining: after new validated candles arrive, train a fresh candidate, preserve its data fingerprint, and review a fresh chronological holdout. The system deliberately never updates a live policy from new trades.")
