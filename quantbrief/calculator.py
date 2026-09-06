from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quantbrief.config import Settings
from quantbrief.market_data import MarketData
from quantbrief.scenarios import SCENARIOS, ShockScenario
from quantbrief.schemas import AssetMetrics, PortfolioMetrics, ShockScenarioResult


class InsufficientDataError(Exception):
    """Raised when there are not enough observations to compute reliable metrics."""


# ---------------------------------------------------------------------------
# Pure, deterministic math primitives. These functions never talk to the
# network and never invoke an LLM — they are the single source of truth for
# every number the API returns.
# ---------------------------------------------------------------------------


def annualized_return(prices: pd.Series, trading_days: int = 252) -> float:
    """Geometric annualized return from a price series (end/start compounded)."""
    prices = prices.dropna()
    if len(prices) < 2:
        raise InsufficientDataError(f"need >= 2 price observations to annualize, got {len(prices)}")
    periods = len(prices) - 1
    total_growth = float(prices.iloc[-1]) / float(prices.iloc[0])
    if total_growth <= 0.0:
        return -1.0
    return float(total_growth ** (trading_days / periods) - 1.0)


def annualized_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    """Annualized sample standard deviation of daily returns."""
    returns = returns.dropna()
    if len(returns) < 2:
        raise InsufficientDataError(f"need >= 2 return observations to annualize volatility, got {len(returns)}")
    return float(returns.std(ddof=1) * math.sqrt(trading_days))


def sharpe_ratio(annualized_return: float, annualized_vol: float, risk_free_rate: float) -> float | None:
    """Sharpe ratio; None when volatility is zero (undefined)."""
    if annualized_vol <= 0.0:
        return None
    return float((annualized_return - risk_free_rate) / annualized_vol)


def beta(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float | None:
    """CAPM beta = cov(stock, benchmark) / var(benchmark). None if undefined."""
    frame = pd.concat([stock_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(frame) < 2:
        return None
    benchmark_var = float(frame.iloc[:, 1].var(ddof=1))
    if benchmark_var == 0.0:
        return None
    covariance = float(frame.iloc[:, 0].cov(frame.iloc[:, 1]))
    return float(covariance / benchmark_var)


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value at Risk as a *positive* daily loss fraction.

    At ``confidence=0.95`` the daily loss is not expected to exceed the returned
    magnitude more than 5% of the time (empirical percentile method).
    """
    returns = returns.dropna()
    if returns.empty:
        raise InsufficientDataError("cannot compute VaR from an empty return series")
    quantile = (1.0 - confidence) * 100.0
    return float(-np.percentile(returns.to_numpy(dtype=float), quantile))


def weighted_portfolio_returns(asset_returns: pd.DataFrame, weights: list[float]) -> pd.Series:
    """Daily portfolio returns as the weighted sum of asset returns.

    Rows where an asset has no data (e.g. IPO gaps) contribute zero that day;
    this is documented as an estimation convention for shock replay.
    """
    weight_series = pd.Series(weights, index=asset_returns.columns, dtype=float)
    return asset_returns.mul(weight_series, axis=1).sum(axis=1, skipna=True)


def scenario_result(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    weights: list[float],
    scenario: ShockScenario,
) -> ShockScenarioResult:
    """Replay one historical shock window against the portfolio and benchmark."""
    start_ts, end_ts = pd.Timestamp(scenario.start), pd.Timestamp(scenario.end)
    window = asset_returns.loc[start_ts:end_ts]
    bench_window = benchmark_returns.loc[start_ts:end_ts]

    base = ShockScenarioResult(
        id=scenario.id,
        name=scenario.name,
        description=scenario.description,
        start_date=scenario.start,
        end_date=scenario.end,
    )
    if window.empty:
        return base

    portfolio_daily = weighted_portfolio_returns(window, weights)
    if portfolio_daily.notna().any():
        base.portfolio_return = float((1.0 + portfolio_daily).prod(skipna=True) - 1.0)
    if bench_window.notna().any():
        base.benchmark_return = float((1.0 + bench_window.dropna()).prod() - 1.0)
    base.coverage_pct = float(window.notna().mean().mean() * 100.0)
    return base


# ---------------------------------------------------------------------------
# Orchestration: turn aligned price history into the full metrics bundle.
# ---------------------------------------------------------------------------


def compute_metrics(
    data: MarketData,
    weights: list[float],
    settings: Settings,
) -> tuple[PortfolioMetrics, list[AssetMetrics], list[ShockScenarioResult]]:
    """Compute portfolio metrics, asset breakdown, and shock scenarios.

    All math here is deterministic (NumPy/Pandas); nothing is delegated to an LLM.
    """
    tickers = list(data.prices.columns)
    if len(tickers) != len(weights):
        raise ValueError("tickers/weights length mismatch (must be validated upstream)")

    # --- Metrics window: complete rows for every ticker ---------------------
    metrics_window = data.prices.loc[data.prices.index >= pd.Timestamp(data.metrics_start)].dropna()
    if len(metrics_window) < settings.min_observations:
        raise InsufficientDataError(
            f"only {len(metrics_window)} complete trading days available in the metrics window "
            f"(need >= {settings.min_observations}); increase lookback_days or check ticker history"
        )

    asset_returns = metrics_window.pct_change().dropna(how="all")
    benchmark_window = data.benchmark.reindex(metrics_window.index).ffill()
    benchmark_returns = benchmark_window.pct_change().dropna()
    benchmark_returns_aligned = benchmark_returns.reindex(asset_returns.index)

    # --- Portfolio-level metrics --------------------------------------------
    portfolio_daily = weighted_portfolio_returns(asset_returns, weights)
    portfolio_price = (1.0 + portfolio_daily).cumprod()
    port_annual_return = annualized_return(portfolio_price, settings.trading_days_per_year)
    port_annual_vol = annualized_volatility(portfolio_daily, settings.trading_days_per_year)

    portfolio_metrics = PortfolioMetrics(
        annualized_return=port_annual_return,
        annualized_volatility=port_annual_vol,
        sharpe_ratio=sharpe_ratio(port_annual_return, port_annual_vol, settings.risk_free_rate),
        beta=beta(portfolio_daily, benchmark_returns_aligned),
        var_95_daily=historical_var(portfolio_daily, settings.var_confidence),
        risk_free_rate=settings.risk_free_rate,
        benchmark_annualized_return=annualized_return(benchmark_window.dropna(), settings.trading_days_per_year),
        observation_days=len(portfolio_daily),
        start_date=metrics_window.index[0].date(),
        end_date=metrics_window.index[-1].date(),
    )

    # --- Asset-level breakdown ----------------------------------------------
    asset_metrics: list[AssetMetrics] = []
    for ticker, weight in zip(tickers, weights):
        asset_metrics.append(
            AssetMetrics(
                ticker=ticker,
                weight=weight,
                annualized_return=annualized_return(metrics_window[ticker], settings.trading_days_per_year),
                annualized_volatility=annualized_volatility(asset_returns[ticker], settings.trading_days_per_year),
                beta=beta(asset_returns[ticker], benchmark_returns_aligned),
            )
        )

    # --- Shock scenarios over the full history ------------------------------
    full_asset_returns = data.prices.pct_change().dropna(how="all")
    full_benchmark_returns = data.benchmark.pct_change().dropna()
    shocks = [
        scenario_result(full_asset_returns, full_benchmark_returns, weights, scenario) for scenario in SCENARIOS
    ]

    return portfolio_metrics, asset_metrics, shocks