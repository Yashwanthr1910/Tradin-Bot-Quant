# Indian Equities Quant Research Platform

Research-only platform for validating local market data and evaluating the existing 15-minute RSI/pivot strategy. It does not place live orders. The prior strategy scripts and CSV are preserved in `archive/legacy/`; their reported returns are not treated as verified results.

## Current implementation status

Working first milestone:

- `src/quant_platform/data.py`: provider-neutral OHLCV interface, timezone normalization, duplicate/invalid-row reporting, and coverage helpers. Naive timestamps are assumed to be Asia/Kolkata and reported as such.
- `src/quant_platform/providers.py`: optional yfinance adapter, imported only when used.
- `src/quant_platform/strategy.py`: configurable RSI(14) rolling arithmetic mean and prior-session pivot/BC/TC breakout baseline; signal at candle close, entry at next candle open; one accepted trade per symbol/session; stop/target simulation and end-of-day square-off.
- `src/quant_platform/metrics.py`: safe trade-close summary metrics. Sharpe/Sortino are explicitly trade-frequency annualized here; this is not a portfolio mark-to-market equity series.
- `scripts/validate_data.py`, `scripts/download_data.py`, and `scripts/run_backtest.py`: local validation, opt-in acquisition, and reproducible run artifacts.
- `scripts/download_universe.py`, `scripts/run_universe_backtest.py`, and `scripts/resample_data.py`: universe acquisition, shared-capital universe backtests, and complete session-aligned resampling.
- `dashboard/app.py`: saved-run overview, parameterized local backtest form, portfolio metrics/equity plots, trade ledger, and data inventory.
- `tests/test_core.py`: offline deterministic checks.

The shared-capital multi-symbol simulator and configurable proportional fees/slippage are available through `scripts/run_portfolio.py`. Downloads now have per-symbol CSV checkpoints, overlap deduplication, retry/backoff, and status sidecars. A constituent snapshot from the [NSE NIFTY 500 page](https://www.nseindia.com/static/products-services/indices-nifty500-index) and [published CSV](https://archives.nseindia.com/content/indices/ind_nifty500list.csv) is stored under `data/reference/`; its page date is April 22, 2026, and it predates the reported September 2026 rebalance. On October 24, 2026, the snapshot's 501 entries were requested from Yahoo Finance for 15-minute data from September 1 through October 24: 497 returned bars and 4 returned no data. See `data/metadata/last_download_batch.json` and each symbol's `status.json` for exact results. This is a provider response for that dated snapshot, not a verified current or historical membership database. The data pipeline still needs production-grade resume orchestration, parquet/partitioned storage, comprehensive calendar-aware gap classification, historical NIFTY 500 membership, ML/regime models, walk-forward evaluation, optimization, paper execution, and expanded dashboard pages. Costs default to zero and must be configured. Research output is not a deployable strategy result.

## Baseline and correctness conventions

Defaults are stored in `config/settings.json`: ₹10,000 reference capital, 1% risk, 1.4 reward/risk, RSI(14), thresholds 45/40, entry signal window 10:00–14:00, square-off 15:15, and the original 23-symbol strategy basket. `config/universe.csv` records that strategy basket. The NIFTY 500 download universe is read from `data/reference/nifty500_constituents.csv` and is tracked separately from the strategy basket.

The original scripts are retained unchanged in `archive/legacy/`. The corrected simulator makes these explicit choices:

- Timestamps label bar opens in Asia/Kolkata. The signal uses that bar only after its close and executes at the next same-session bar open.
- Daily levels use the previous observed session's completed OHLC; exchange-calendar and missing-session interpretation still need a proper calendar/data-coverage layer.
- RSI uses rolling arithmetic means of gains/losses. A flat window returns 50 and a gain-only window returns 100.
- If stop and target are both touched inside one bar, stop wins. A gap through a stop fills at the worse bar open.
- Square-off uses the close of the latest bar completed by 15:15. Trades are not carried into another session.
- Invalid OHLC rows are excluded and counted; no price interpolation is performed.

These conventions can change legacy output. The legacy preliminary figures (401 trades, ₹3,704.31 net P&L, and related metrics) remain unverified and are not hardcoded.

## Windows PowerShell setup

Install Python 3.11 or newer, then from the repository root:

```powershell
cd C:\Trading-Bot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dashboard,yahoo,dev]"
$env:PYTHONPATH = "src"
```

For only core validation/backtesting, install `python -m pip install -e .` instead. The current [yfinance API documentation](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html) lists native intervals and says intraday history cannot extend beyond the latest 60 days. Do not assume it can supply 15-minute data back to 2010. No download or market-data response is claimed until a run produces its own report.

## Commands

Validate a local OHLCV CSV (requires a `Datetime` or `Date` column and Open/High/Low/Close/Volume):

```powershell
$env:PYTHONPATH = "src"
python scripts/validate_data.py .\path\to\bars.csv
```

Optional provider download (inspect actual returned coverage; provider may truncate the requested range):

```powershell
$env:PYTHONPATH = "src"
python scripts/download_data.py RELIANCE.NS --interval 15m --start 2026-09-01 --end 2026-10-25
```

Download the configured NIFTY 500 reference snapshot (15-minute intraday history is limited by the provider), or resample complete aligned bars (supported source/target pairs are 5→10 minutes and 60→120/240 minutes):

```powershell
$env:PYTHONPATH = "src"
python scripts/download_universe.py --interval 15m --start 2026-09-01 --end 2026-10-25 --workers 4
python scripts/download_data.py RELIANCE.NS --interval 5m --start 2026-09-01 --end 2026-10-25
python scripts/resample_data.py .\data\raw\RELIANCE.NS\5m\bars.csv --symbol RELIANCE.NS --source-minutes 5 --target-minutes 10
```

Run baseline on a local CSV; each run writes a unique directory under `results/backtests/` with configuration, validation report, trades, and rejected signals:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_backtest.py .\data\validated\RELIANCE_15m.csv --symbol RELIANCE.NS
```

Shared-capital multi-symbol simulation (repeat `--data` for each symbol):

```powershell
$env:PYTHONPATH = "src"
python scripts/run_portfolio.py --data RELIANCE.NS=.\data\validated\RELIANCE_15m.csv --data SBIN.NS=.\data\validated\SBIN_15m.csv --fee-bps 5 --slippage-bps 2
```

Run the strategy across every available local 15-minute universe file with one shared capital account. It writes an append-only experiment, full trade and rejection ledgers, equity curve, and data inventory under `results/backtests/`:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_universe_backtest.py --interval 15m --start 2026-09-01 --end 2026-10-23 --max-positions 5 --fee-bps 5 --slippage-bps 2
```

Yahoo Finance cannot supply one year of 15-minute history. Do not relabel a daily-data test as a test of this intraday strategy. Supply a licensed or local intraday source with one-year coverage before drawing annual performance conclusions.

The dashboard date inputs use the actual coverage in `data/raw/<SYMBOL>/15m/status.json`; the current local files run from 1 September to 23 October 2026. To run a one-year universe test, populate `data/raw/<SYMBOL>/15m/bars.csv` for every intended symbol with one year of valid intraday data. The universe runner discovers those files automatically and will then expose the full range.

Import a provider's local CSV export without discarding the current overlap:

```powershell
$env:PYTHONPATH = "src"
python scripts/import_intraday_data.py --interval 15m --source-name YOUR_PROVIDER --data RELIANCE.NS=C:\path\to\RELIANCE_15m.csv
```

The import requires `Datetime` (or `Date`), `Open`, `High`, `Low`, `Close`, and `Volume` columns. Repeat `--data` for each available symbol.

## FYERS historical data

Copy `.env.example` to `.env` and enter your FYERS App ID, regenerated secret, and the registered redirect URL locally. Do not commit or share `.env`. Authenticate to generate the daily local access token, probe coverage for one liquid stock, then start the universe download using the earliest date that probe actually returns:

```powershell
$env:PYTHONPATH = "src"
python -m pip install -e ".[fyers]"
python scripts/fyers_auth.py
# Complete FYERS login in the opened browser, then copy its redirect URL.
python scripts/fyers_auth.py --auth-code "PASTE_REDIRECT_URL_OR_AUTH_CODE_HERE"
python scripts/fyers_probe_history.py --symbol SBIN.NS
python scripts/download_fyers_all_timeframes.py --start 2020-01-01 --end 2026-10-25
```

FYERS accepts at most 100 days per intraday request. The downloader chunks longer ranges, preserves per-symbol checkpoints, and waits between requests. The coverage probe is authoritative: FYERS plan and instrument availability determine the earliest usable date. Its current documentation states intraday history is available from 3 July 2017, while Standard and Prime plan windows may further constrain that coverage.

Run offline tests and launch the dashboard:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
streamlit run dashboard/app.py
```

## Research interpretation

Portfolio Sharpe and Sortino use daily closing observations with 252 trading periods/year and zero risk-free rate. CAGR uses elapsed calendar days; Calmar is CAGR divided by absolute maximum drawdown percentage. Drawdown duration and recovery are measured in daily observed equity points. CAGR is suppressed below 365 calendar days; volatility, Sharpe, and Sortino below 30 daily returns; historical VaR/expected shortfall below 20 daily returns. Fees and slippage are reported separately, and net trade P&L includes both. The simpler single-symbol command produces trade-close metrics and does not calculate Calmar/CAGR. Metrics from either command are only as good as the supplied data and assumptions. Simulated results are not forecasts, and positive historical P&L would not establish profitability.

## Roadmap

1. Data acquisition, checkpoint/resume, source metadata, coverage, and quality reports.
2. Expand and validate baseline behavior using deterministic synthetic scenarios, including next-bar fills and stop-gap handling.
3. Extend portfolio accounting with daily loss/drawdown controls, order lifecycle records, and richer exposure attribution.
4. Expand portfolio equity metrics, experiment comparison, and editable dashboard controls.
5. Chronological supervised ML and regime research, then walk-forward/out-of-sample analysis.
6. Paper simulation and resumable orchestration. Live broker execution is out of scope for the initial research platform.
