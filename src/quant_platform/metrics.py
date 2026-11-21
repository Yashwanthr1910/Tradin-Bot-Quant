"""Portfolio equity metrics with explicit sample and annualization conventions."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def summarize(trades: pd.DataFrame, initial_capital: float = 10_000, periods_per_year: int = 252) -> dict:
    pnl = trades["net_pnl"].astype(float) if not trades.empty else pd.Series(dtype=float)
    # Trade-close equity is a trade-time series, not a daily mark-to-market curve.
    equity = initial_capital + pnl.cumsum()
    returns = equity.pct_change().dropna()
    peak = equity.cummax()
    dd = equity - peak
    max_dd = float(dd.min()) if len(dd) else 0.0
    max_dd_pct = float((equity / peak - 1).min()) if len(equity) else 0.0
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(np.sqrt(periods_per_year) * returns.mean() / std) if std > 0 else None
    downside = returns[returns < 0]
    downside_std = float(np.sqrt((downside.pow(2)).mean())) if len(downside) else 0.0
    sortino = float(np.sqrt(periods_per_year) * returns.mean() / downside_std) if downside_std else None
    wins = pnl[pnl > 0]; losses = pnl[pnl < 0]
    gross_profit, gross_loss = float(wins.sum()), float(-losses.sum())
    return {"initial_equity": float(initial_capital), "final_equity": float(initial_capital+pnl.sum()), "net_pnl": float(pnl.sum()), "total_return_pct": float(pnl.sum()/initial_capital*100), "trades": int(len(pnl)), "wins": int(len(wins)), "losses": int(len(losses)), "gross_profit": gross_profit, "gross_loss": -gross_loss, "win_rate_pct": float(len(wins)/len(pnl)*100) if len(pnl) else 0.0, "profit_factor": gross_profit/gross_loss if gross_loss else (math.inf if gross_profit else None), "max_drawdown_rupees": max_dd, "max_drawdown_pct": max_dd_pct*100, "sharpe_trade_frequency_annualized": sharpe, "sortino_trade_frequency_annualized": sortino, "return_frequency_note": "Trade-close returns, annualized using configured periods_per_year; not a daily marked-to-market portfolio series.", "transaction_costs": 0.0}


def summarize_portfolio(trades: pd.DataFrame, equity_curve: pd.DataFrame, initial_capital: float = 10_000) -> dict:
    """Daily-close portfolio metrics from a timestamped marked-to-market equity curve."""
    base={"initial_equity":float(initial_capital),"trades":int(len(trades)),"return_frequency":"daily close","annualization_periods":252,"risk_free_rate":0.0}
    if equity_curve.empty:
        return {**base,"final_equity":float(initial_capital),"net_pnl":0.0,"total_return_pct":0.0,"cagr_pct":None,"annualized_volatility_pct":None,"sharpe":None,"sortino":None,"max_drawdown_rupees":0.0,"max_drawdown_pct":0.0,"calmar":None,"drawdown_duration_days":0,"recovery_days":None,"gross_profit":0.0,"gross_loss":0.0,"profit_factor":None,"win_rate_pct":0.0,"average_win":None,"average_loss":None,"average_win_loss_ratio":None,"expectancy_per_trade":None,"total_costs":0.0}
    curve=equity_curve.copy()
    curve["timestamp"]=pd.to_datetime(curve.timestamp,utc=True)
    curve=curve.sort_values("timestamp").drop_duplicates("timestamp",keep="last").set_index("timestamp")
    daily=curve.equity.resample("1D").last().dropna()
    if daily.empty: daily=pd.Series([initial_capital],index=[curve.index[0]])
    returns=daily.pct_change().dropna()
    peak=daily.cummax(); dd=daily-peak; dd_pct=daily/peak-1
    elapsed_days=max((daily.index[-1]-daily.index[0]).total_seconds()/86400,0)
    total_return=float(daily.iloc[-1]/initial_capital-1)
    cagr=(float((daily.iloc[-1]/initial_capital)**(365.25/elapsed_days)-1) if elapsed_days>=365 and daily.iloc[-1]>0 else None)
    vol=float(returns.std(ddof=1)*np.sqrt(252)) if len(returns)>=30 else None
    sharpe=float(np.sqrt(252)*returns.mean()/returns.std(ddof=1)) if len(returns)>=30 and returns.std(ddof=1)>0 else None
    downside=returns[returns<0]
    sortino=float(np.sqrt(252)*returns.mean()/np.sqrt((downside**2).mean())) if len(returns)>=30 and len(downside) and (downside**2).mean()>0 else None
    max_dd=float(dd.min()) if len(dd) else 0.0; max_dd_pct=float(dd_pct.min()) if len(dd_pct) else 0.0
    underwater=dd_pct<0; max_duration=run=0
    for active in underwater.tolist():
        run=run+1 if active else 0; max_duration=max(max_duration,run)
    trough=dd_pct.idxmin(); later= daily.loc[trough:]; prior_peak=float(peak.loc[trough]); rec=later[later>=prior_peak]
    recovery_days=int((rec.index[0]-trough).days) if not rec.empty else None
    pnl=trades.net_pnl.astype(float) if not trades.empty and "net_pnl" in trades else pd.Series(dtype=float)
    wins=pnl[pnl>0]; losses=pnl[pnl<0]; gross_loss=float(-losses.sum())
    gross_pnl=float(trades.gross_pnl.sum()) if not trades.empty and "gross_pnl" in trades else float(pnl.sum())
    fees=float(trades.fees.sum()) if not trades.empty and "fees" in trades else 0.0
    slip_cost=float(trades.slippage_cost.sum()) if not trades.empty and "slippage_cost" in trades else 0.0
    avg_win=float(wins.mean()) if len(wins) else None; avg_loss=float(-losses.mean()) if len(losses) else None
    monthly=(daily.resample("ME").last().pct_change().dropna()*100).to_dict()
    yearly=(daily.resample("YE").last().pct_change().dropna()*100).to_dict()
    monthly={str(k.date()):float(v) for k,v in monthly.items()}; yearly={str(k.year):float(v) for k,v in yearly.items()}
    turnover=float(((trades.entry_price.abs()+trades.exit_price.abs())*trades.quantity).sum()) if not trades.empty and {"entry_price","exit_price","quantity"}.issubset(trades.columns) else 0.0
    var95=float(-returns.quantile(.05)) if len(returns)>=20 else None
    es95=float(-returns[returns<=returns.quantile(.05)].mean()) if len(returns)>=20 and (returns<=returns.quantile(.05)).any() else None
    return {**base,"final_equity":float(daily.iloc[-1]),"net_pnl":float(daily.iloc[-1]-initial_capital),"gross_pnl":gross_pnl,"total_return_pct":total_return*100,"cagr_pct":cagr*100 if cagr is not None else None,"annualized_volatility_pct":vol*100 if vol is not None else None,"sharpe":sharpe,"sortino":sortino,"max_drawdown_rupees":max_dd,"max_drawdown_pct":max_dd_pct*100,"calmar":cagr/abs(max_dd_pct) if cagr is not None and max_dd_pct<0 else None,"drawdown_duration_trading_days":max_duration,"recovery_days":recovery_days,"gross_profit":float(wins.sum()),"gross_loss":-gross_loss,"profit_factor":float(wins.sum()/gross_loss) if gross_loss else (math.inf if len(wins) else None),"win_rate_pct":float((pnl>0).sum()/len(pnl)*100) if len(pnl) else 0.0,"winning_trades":int((pnl>0).sum()),"losing_trades":int((pnl<0).sum()),"breakeven_trades":int((pnl==0).sum()),"average_win":avg_win,"average_loss":avg_loss,"average_win_loss_ratio":avg_win/avg_loss if avg_win is not None and avg_loss else None,"expectancy_per_trade":float(pnl.mean()) if len(pnl) else None,"fees":fees,"slippage_cost":slip_cost,"total_costs":fees+slip_cost,"turnover":turnover,"var_95_daily":var95,"expected_shortfall_95_daily":es95,"monthly_returns_pct":monthly,"yearly_returns_pct":yearly,"minimum_history_note":"CAGR requires at least 365 calendar days; annualized volatility, Sharpe, and Sortino require at least 30 daily returns; historical VaR/ES require at least 20 daily returns."}
