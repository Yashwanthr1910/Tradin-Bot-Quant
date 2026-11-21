"""Optional external data adapters; importing the core package does not require yfinance."""
from datetime import timedelta
import time
import pandas as pd


class FyersProvider:
    """FYERS historical-candle adapter for locally authenticated, data-only research."""
    RESOLUTIONS = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "2h": "120", "4h": "240"}
    MAX_DAYS_PER_REQUEST = 100

    def __init__(self, app_id: str, access_token: str, request_delay_seconds: float = 1.3):
        if not app_id or not access_token:
            raise ValueError("FYERS_APP_ID and FYERS_ACCESS_TOKEN must be set in .env. Run scripts/fyers_auth.py first.")
        if request_delay_seconds < 0:
            raise ValueError("request_delay_seconds cannot be negative")
        try:
            from fyers_apiv3 import fyersModel
        except ImportError as exc:
            raise RuntimeError("Install FYERS support: python -m pip install -e .[fyers]") from exc
        self._client = fyersModel.FyersModel(client_id=app_id, token=access_token, is_async=False, log_path="")
        self.request_delay_seconds = request_delay_seconds

    @staticmethod
    def fyers_symbol(symbol: str) -> str:
        value = symbol.upper().strip()
        if value.startswith("NSE:"):
            return value
        if value.endswith(".NS"):
            value = value[:-3]
        return f"NSE:{value}-EQ"

    def fetch(self, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
        if interval not in self.RESOLUTIONS:
            raise ValueError(f"Unsupported FYERS interval {interval!r}; use one of {sorted(self.RESOLUTIONS)}")
        start_day, end_day = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        if end_day <= start_day:
            raise ValueError("FYERS end date must be later than the start date")
        frames: list[pd.DataFrame] = []
        failures: list[str] = []
        chunk_start = start_day
        while chunk_start < end_day:
            chunk_end = min(chunk_start + timedelta(days=self.MAX_DAYS_PER_REQUEST), end_day)
            # The project's end dates are exclusive. FYERS date-format requests are date-inclusive.
            request_end = chunk_end - timedelta(days=1)
            response = self._client.history(data={
                "symbol": self.fyers_symbol(symbol), "resolution": self.RESOLUTIONS[interval],
                "date_format": "1", "range_from": chunk_start.date().isoformat(),
                "range_to": request_end.date().isoformat(), "cont_flag": "0",
            })
            if isinstance(response, dict) and response.get("s") == "ok" and response.get("candles"):
                frame = pd.DataFrame(response["candles"], columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
                frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
                frames.append(frame.set_index("timestamp"))
            else:
                message = response.get("message", "no candle data returned") if isinstance(response, dict) else "invalid FYERS response"
                failures.append(f"{chunk_start.date()}:{request_end.date()} {message}")
            chunk_start = chunk_end
            if chunk_start < end_day and self.request_delay_seconds:
                time.sleep(self.request_delay_seconds)
        # A partial response can be valid for a recently listed symbol, but a
        # throttle or malformed-request response means the requested range was
        # not reliably retrieved.  Raise so the caller retries rather than
        # saving a deceptively complete-looking partial file.
        fatal_failures = [item for item in failures if any(
            marker in item.lower() for marker in ("request limit", "requests/second", "requests/minute", "bad request")
        )]
        if fatal_failures:
            raise RuntimeError(f"FYERS returned incomplete data for {symbol} {interval} {start}:{end}: {'; '.join(fatal_failures[:3])}")
        if not frames:
            detail = "; ".join(failures[:3]) or "no candle data returned"
            raise RuntimeError(f"FYERS returned no data for {symbol} {interval} {start}:{end}: {detail}")
        return pd.concat(frames).sort_index().loc[lambda frame: ~frame.index.duplicated(keep="last")]

class YahooFinanceProvider:
    NATIVE_INTERVALS={"1m","2m","5m","15m","30m","60m","1h","90m","1d","5d","1wk","1mo","3mo"}
    INTRADAY_INTERVALS={"1m","2m","5m","15m","30m","60m","1h","90m"}

    def fetch(self, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
        if interval not in self.NATIVE_INTERVALS:
            raise ValueError(f"yfinance adapter does not provide {interval!r} natively; resample validated lower-frequency bars instead")
        if interval in self.INTRADAY_INTERVALS:
            requested_days = (pd.Timestamp(end) - pd.Timestamp(start)).days
            if requested_days > 60:
                raise ValueError(
                    f"Yahoo Finance cannot provide {requested_days} days of {interval} bars. "
                    "Its intraday history is limited to the latest 60 days. "
                    "Use a licensed/local intraday source for a one-year 15-minute backtest."
                )
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("Install the optional provider with: python -m pip install -e .[yahoo]") from exc
        frame = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False, actions=False)
        if frame is None or frame.empty:
            raise RuntimeError(f"Provider returned no data for {symbol} {interval} {start}:{end}")
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        return frame
