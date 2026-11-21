import unittest
import pandas as pd
from io import StringIO
from tempfile import TemporaryDirectory
from pathlib import Path
from quant_platform.data import canonicalize
from quant_platform.strategy import StrategyConfig, prepare_signals, backtest_symbol
from quant_platform.metrics import summarize, summarize_portfolio
from quant_platform.portfolio import PortfolioConfig, backtest_portfolio
from quant_platform.resampling import resample_complete
from quant_platform.downloads import download_symbol
from quant_platform.universe import load_local_universe
from quant_platform.providers import YahooFinanceProvider

class CoreTests(unittest.TestCase):
  def test_canonicalize_timezone_duplicates_and_invalid_rows(self):
    idx=pd.to_datetime(["2024-01-02 09:15","2024-01-02 09:15","2024-01-02 09:30"])
    frame=pd.DataFrame({"open":[10,10,12],"high":[11,11,11],"low":[9,9,13],"close":[10,10,12],"volume":[5,6,1]},index=idx)
    clean,report=canonicalize(frame)
    self.assertTrue(report.timezone_assumed); self.assertEqual(report.duplicate_timestamps,1); self.assertEqual(report.invalid_ohlc,1)
    self.assertEqual(len(clean),1); self.assertEqual(str(clean.index.tz),"Asia/Kolkata")

  def test_canonical_csv_roundtrip(self):
    idx=pd.date_range("2024-01-02 09:15",periods=1,freq="15min",tz="Asia/Kolkata")
    bars=pd.DataFrame({"Open":[10],"High":[11],"Low":[9],"Close":[10],"Volume":[1]},index=idx)
    loaded=pd.read_csv(StringIO(bars.to_csv()))
    clean,_=canonicalize(loaded)
    self.assertEqual(len(clean),1)
    self.assertEqual(clean.index[0],idx[0])

  def test_rsi_constant_series_is_neutral_and_timestamps_are_aware(self):
    idx=pd.date_range("2024-01-01 09:15",periods=20,freq="15min",tz="Asia/Kolkata")
    bars=pd.DataFrame({"Open":100.,"High":101.,"Low":99.,"Close":100.,"Volume":1},index=idx)
    out=prepare_signals(bars,StrategyConfig(rsi_period=3))
    self.assertTrue((out.RSI.dropna()==50).all())
    self.assertEqual(str(out.index.tz),"Asia/Kolkata")

  def test_empty_metrics_are_safe(self):
    result=summarize(pd.DataFrame(columns=["net_pnl"]))
    self.assertEqual(result["trades"],0); self.assertEqual(result["max_drawdown_rupees"],0); self.assertIsNone(result["sharpe_trade_frequency_annualized"])

  def test_portfolio_metrics_use_timestamped_daily_equity(self):
    curve=pd.DataFrame({"timestamp":pd.date_range("2024-01-01",periods=4,freq="D",tz="UTC"),"equity":[1000,1100,900,1100]})
    trades=pd.DataFrame({"net_pnl":[100,-200,200],"fees":[1,1,1]})
    result=summarize_portfolio(trades,curve,1000)
    self.assertEqual(result["max_drawdown_rupees"],-200)
    self.assertAlmostEqual(result["max_drawdown_pct"],(900/1100-1)*100)
    self.assertEqual(result["recovery_days"],1)
    self.assertEqual(result["total_costs"],3)
    self.assertEqual(result["gross_profit"],300)
    self.assertEqual(result["gross_loss"],-200)
    self.assertEqual(result["average_win"],150)
    self.assertEqual(result["average_loss"],200)
    self.assertEqual(result["average_win_loss_ratio"],.75)
    self.assertEqual(result["expectancy_per_trade"],100 / 3)
    self.assertIsNone(result["cagr_pct"])
    self.assertIsNone(result["sharpe"])

  def test_signal_enters_next_bar_and_stop_gap_fills_at_open(self):
    prior=pd.date_range("2024-01-02 09:15",periods=24,freq="15min",tz="Asia/Kolkata")
    today=pd.date_range("2024-01-03 09:15",periods=24,freq="15min",tz="Asia/Kolkata")
    idx=prior.append(today)
    bars=pd.DataFrame({"Open":99.,"High":100.,"Low":95.,"Close":99.,"Volume":10},index=idx)
    bars.loc[today[3], ["Open","High","Low","Close"]]=[100.,103.,99.,102.]
    bars.loc[today[4], ["Open","High","Low","Close"]]=[102.,103.,99.,102.]
    bars.loc[today[5], ["Open","High","Low","Close"]]=[94.,96.,93.,95.]
    cfg=StrategyConfig(rsi_period=3)
    prepared=prepare_signals(bars,cfg)
    self.assertEqual(prepared.loc[today[3],"Signal"],"LONG")
    trades,_=backtest_symbol(bars,cfg,"TEST")
    self.assertEqual(len(trades),1)
    self.assertEqual(trades.iloc[0].entry_time,today[4].isoformat())
    self.assertEqual(trades.iloc[0].exit_price,94.)
    self.assertIn("stop",trades.iloc[0].exit_reason)

  def test_portfolio_limits_simultaneous_entries_and_marks_daily_equity(self):
    prior=pd.date_range("2024-01-02 09:15",periods=24,freq="15min",tz="Asia/Kolkata")
    today=pd.date_range("2024-01-03 09:15",periods=24,freq="15min",tz="Asia/Kolkata")
    idx=prior.append(today)
    bars=pd.DataFrame({"Open":99.,"High":100.,"Low":95.,"Close":99.,"Volume":10},index=idx)
    bars.loc[today[3], ["Open","High","Low","Close"]]=[100.,103.,99.,102.]
    bars.loc[today[4], ["Open","High","Low","Close"]]=[102.,103.,99.,102.]
    data={"AAA":bars,"BBB":bars.copy()}
    trades,rejected,equity=backtest_portfolio(data,StrategyConfig(rsi_period=3),PortfolioConfig(max_open_positions=1,fee_bps_per_side=5))
    self.assertEqual(len(trades),1)
    self.assertTrue((rejected.reason=="max_open_positions").any())
    self.assertFalse(equity.empty)
    metrics=summarize_portfolio(trades,equity)
    self.assertEqual(metrics["return_frequency"],"daily close")
    self.assertGreater(metrics["total_costs"],0)


  def test_resampler_drops_incomplete_session_bucket(self):
    idx=pd.date_range("2024-01-02 09:15",periods=5,freq="5min",tz="Asia/Kolkata")
    bars=pd.DataFrame({"Open":[10,11,12,13,14],"High":[11,12,13,14,15],"Low":[9,10,11,12,13],"Close":[10.5,11.5,12.5,13.5,14.5],"Volume":[1,2,3,4,5]},index=idx)
    result=resample_complete(bars,5,10)
    self.assertEqual(len(result),2)
    self.assertEqual(result.iloc[0].Volume,3)
    self.assertEqual(result.iloc[0].High,12)

  def test_download_checkpoint_and_failure_status(self):
    class Provider:
      def fetch(self,symbol,interval,start,end):
        idx=pd.date_range("2024-01-02 09:15",periods=2,freq="15min",tz="Asia/Kolkata")
        return pd.DataFrame({"Open":[10,11],"High":[11,12],"Low":[9,10],"Close":[10,11],"Volume":[1,2]},index=idx)
    class Broken:
      def fetch(self,*args): raise RuntimeError("offline")
    with TemporaryDirectory(dir=Path.cwd()) as temp:
      first=download_symbol(Provider(),"TEST","15m","2024-01-01","2024-02-01",root=Path(temp),retries=0,throttle_seconds=0)
      self.assertEqual(first["status"],"COMPLETED")
      again=download_symbol(Provider(),"TEST","15m","2024-01-01","2024-02-01",root=Path(temp),retries=0,throttle_seconds=0)
      self.assertEqual(again["coverage"]["rows"],2)
      failed=download_symbol(Broken(),"FAIL","15m","2024-01-01","2024-02-01",root=Path(temp),retries=0,throttle_seconds=0)
      self.assertEqual(failed["status"],"FAILED")

  def test_local_universe_loader_discovers_and_filters_files(self):
    with TemporaryDirectory(dir=Path.cwd()) as temp:
      base=Path(temp)
      folder=base/"AAA.NS"/"15m"; folder.mkdir(parents=True)
      pd.DataFrame({"Datetime":["2024-01-02 09:15","2024-01-03 09:15"],"Open":[10,11],"High":[11,12],"Low":[9,10],"Close":[10,11],"Volume":[1,1]}).to_csv(folder/"bars.csv",index=False)
      data,inventory,validation=load_local_universe(base,"15m",start="2024-01-03",end="2024-01-03")
      self.assertEqual(list(data),["AAA.NS"]); self.assertEqual(len(data["AAA.NS"]),1)
      self.assertEqual(inventory.iloc[0].status,"READY"); self.assertIn("AAA.NS",validation)

  def test_yahoo_rejects_one_year_intraday_before_network(self):
    with self.assertRaisesRegex(ValueError, "latest 60 days"):
      YahooFinanceProvider().fetch("TEST.NS", "15m", "2025-01-01", "2026-01-02")

  def test_fyers_long_range_is_chunked_and_deduplicated(self):
    class FakeClient:
      def __init__(self): self.calls=[]
      def history(self,data):
        self.calls.append(data)
        return {"s":"ok", "candles":[[1577866500,10,11,9,10.5,100]]}
    from quant_platform.providers import FyersProvider
    provider=object.__new__(FyersProvider); provider._client=FakeClient(); provider.request_delay_seconds=0
    bars=provider.fetch("TEST.NS","15m","2020-01-01","2020-08-01")
    self.assertEqual(len(provider._client.calls),3)
    self.assertEqual(provider._client.calls[0]["range_from"],"2020-01-01")
    self.assertEqual(provider._client.calls[-1]["range_to"],"2020-07-31")
    self.assertEqual(len(bars),1)

if __name__ == "__main__": unittest.main()
