from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

from quantbrief.scenarios import SCENARIOS


def utc_today() -> date:
    """Current date in UTC — deterministic across hosts/timezones."""
    return datetime.now(UTC).date()


class MarketDataError(Exception):
    """Raised when market data cannot be retrieved or is unusable.

    ``status_code`` lets the API layer map the failure to the right HTTP error:
    400 when tickers are unknown/delisted, 503 when the network is the problem.
    """

    def __init__(self, message: str, *, missing: list[str] | None = None, network: bool = False) -> None:
        super().__init__(message)
        self.missing: list[str] = missing or []
        self.network: bool = network

    @property
    def status_code(self) -> int:
        return 503 if self.network else 400


@dataclass(frozen=True)
class MarketData:
    """Aligned, timezone-naive adjusted-close history for the analysis."""

    prices: pd.DataFrame  # columns = tickers, DatetimeIndex (Adj Close)
    benchmark: pd.Series  # benchmark Adj Close
    metrics_start: date  # inclusive start of the metrics window
    metrics_end: date  # inclusive end of the metrics window


def download_window(lookback_days: int) -> tuple[date, date]:
    """Start/end dates covering the metrics window *and* every shock scenario.

    The download must reach back further than the metrics window so that
    historical stress windows (e.g. 2008) can be replayed.
    """
    today = utc_today()
    end = today + timedelta(days=1)  # small buffer past today
    earliest_shock = min(s.start for s in SCENARIOS) - timedelta(days=15)
    metrics_start = today - timedelta(days=int(lookback_days))
    start = min(metrics_start, earliest_shock)
    return start, end


def _as_tz_naive(series: pd.Series) -> pd.Series:
    """yfinance returns NY-timezone indices; normalize to naive dates."""
    s = series.copy()
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _fetch_close_history(symbol: str, start: date, end: date) -> tuple[pd.Series | None, bool]:
    """Download adjusted close for one symbol.

    Returns ``(close_series, network_error)``. Any non-price outcome (empty
    history, unknown ticker, non-network exception) is treated as "no data"
    rather than a hard crash, so callers can report missing tickers cleanly.
    """
    try:
        history = yf.Ticker(symbol).history(
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
            actions=False,
        )
    except requests.RequestException:  # network-level failure
        return None, True
    except Exception:  # noqa: BLE001 - any non-network failure means "no data" for this symbol
        return None, False
    if history is None or history.empty or "Close" not in history.columns:
        return None, False
    close = _as_tz_naive(history["Close"].dropna())
    if close.empty:
        return None, False
    return close, False


def fetch_market_data(tickers: list[str], benchmark: str, lookback_days: int) -> MarketData:
    """Download and align price history for the requested portfolio.

    Raises :class:`MarketDataError` with a precise message when tickers are
    missing or the network fails — the API layer never sees a raw exception.
    """
    start, end = download_window(lookback_days)
    symbols = [benchmark, *tickers]

    closes: dict[str, pd.Series] = {}
    any_network_error = False
    for symbol in symbols:
        series, network_error = _fetch_close_history(symbol, start, end)
        any_network_error = any_network_error or network_error
        if series is not None:
            closes[symbol] = series

    missing = [s for s in symbols if s not in closes]
    if missing:
        flavor = "network error while fetching" if any_network_error else "no price data"
        raise MarketDataError(
            f"Unable to fetch market data: {flavor} for {', '.join(sorted(missing))}. "
            "Verify the ticker symbols and your network connection, then retry.",
            missing=missing,
            network=any_network_error and set(missing) == set(symbols),
        )

    prices = pd.DataFrame({t: closes[t] for t in tickers}).sort_index()
    benchmark_series = closes[benchmark]
    if prices.empty or benchmark_series.empty:
        raise MarketDataError("No usable price history returned for the requested symbols.")

    today = utc_today()
    return MarketData(
        prices=prices,
        benchmark=benchmark_series,
        metrics_start=today - timedelta(days=int(lookback_days)),
        metrics_end=today,
    )