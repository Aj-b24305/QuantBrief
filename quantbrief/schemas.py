from __future__ import annotations

import math
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AnalysisRequest(BaseModel):
    """Client payload for ``POST /analyze``. Validated strictly (Pydantic v2)."""

    model_config = ConfigDict(extra="forbid")

    tickers: list[str] = Field(
        min_length=1,
        max_length=50,
        description="Asset tickers, e.g. ['AAPL', 'MSFT'].",
    )
    weights: list[float] = Field(
        min_length=1,
        max_length=50,
        description="Portfolio weights; must sum to 1.0 and contain no short positions.",
    )
    lookback_days: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="Calendar days of history used for the metrics window.",
    )
    benchmark: str = Field(default="SPY", description="Benchmark ticker used for beta and scenario context.")

    @field_validator("tickers")
    @classmethod
    def _normalize_tickers(cls, value: list[str]) -> list[str]:
        cleaned = [t.strip().upper() for t in value]
        if any(not t for t in cleaned):
            raise ValueError("tickers must be non-empty strings")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("duplicate tickers are not allowed")
        return cleaned

    @field_validator("benchmark")
    @classmethod
    def _normalize_benchmark(cls, value: str) -> str:
        value = value.strip().upper()
        if not value:
            raise ValueError("benchmark must be a non-empty ticker")
        return value

    @model_validator(mode="after")
    def _validate_portfolio(self) -> AnalysisRequest:
        if len(self.tickers) != len(self.weights):
            raise ValueError(
                f"tickers ({len(self.tickers)}) and weights ({len(self.weights)}) must have identical length"
            )
        if any(w < 0.0 for w in self.weights):
            raise ValueError("short positions are not supported; every weight must be >= 0")
        total = math.fsum(self.weights)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"weights must sum to 1.0, got {total:.8f}")
        return self


class PortfolioMetrics(BaseModel):
    """Deterministic portfolio-level statistics (computed, never LLM-derived)."""

    model_config = ConfigDict(extra="forbid")

    annualized_return: float = Field(description="Geometric annualized return over the metrics window.")
    annualized_volatility: float = Field(description="Annualized daily return volatility (252 trading days).")
    sharpe_ratio: float | None = Field(default=None, description="(annualized return - risk-free rate) / annualized vol.")
    beta: float | None = Field(default=None, description="Portfolio beta vs the benchmark.")
    var_95_daily: float = Field(
        description="Historical 95% daily Value at Risk, expressed as a positive loss fraction."
    )
    risk_free_rate: float = Field(description="Annualized risk-free rate used for the Sharpe ratio.")
    benchmark_annualized_return: float = Field(description="Benchmark annualized return over the same window.")
    observation_days: int = Field(description="Number of daily returns used for the metrics window.")
    start_date: date
    end_date: date


class AssetMetrics(BaseModel):
    """Per-asset breakdown (weight, return, volatility, beta)."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    weight: float
    annualized_return: float
    annualized_volatility: float
    beta: float | None = None


class ShockScenarioResult(BaseModel):
    """Result of replaying a historical shock window against the portfolio."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str
    start_date: date
    end_date: date
    portfolio_return: float | None = Field(
        default=None,
        description="Compounded portfolio return over the window (None if no data in window).",
    )
    benchmark_return: float | None = None
    coverage_pct: float | None = Field(
        default=None,
        description="Average fraction of asset-days with price data inside the window (0-100).",
    )


class AgentSynthesizedMemo(BaseModel):
    """Strict LLM output contract.

    ``extra="forbid"`` rejects any hallucinated field; the LLM is only allowed to
    restate pre-computed metrics and add qualitative commentary.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str
    summary: str
    key_risks: list[str] = Field(min_length=1)
    recommendations: list[str] = Field(min_length=1)
    stress_test_outlook: str
    confidence: Literal["low", "medium", "high"]
    caveats: list[str] = []


class MemoMeta(BaseModel):
    """Metadata about how the memo was produced (LLM vs deterministic fallback)."""

    model_config = ConfigDict(extra="forbid")

    synthesized_by_llm: bool
    model: str | None = None
    elapsed_ms: int | None = None
    note: str | None = None


class MemoContext(BaseModel):
    """Everything the LLM is allowed to see.

    Contains only pre-computed metrics — never raw price series. This is the
    enforcement point of the separation-of-concerns constraint.
    """

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    tickers: list[str]
    weights: list[float]
    benchmark: str
    lookback_days: int
    risk_free_rate: float
    portfolio: PortfolioMetrics
    assets: list[AssetMetrics]
    shock_scenarios: list[ShockScenarioResult]


class AnalysisResponse(BaseModel):
    """Full response of ``POST /analyze``: deterministic metrics + synthesized memo."""

    model_config = ConfigDict(extra="forbid")

    request: AnalysisRequest
    portfolio: PortfolioMetrics
    assets: list[AssetMetrics]
    shock_scenarios: list[ShockScenarioResult]
    memo: AgentSynthesizedMemo
    memo_meta: MemoMeta


class MetricsOnlyResponse(BaseModel):
    """Fast response from ``POST /analyze`` — deterministic metrics only, no LLM.

    Returned immediately (~10 s) so the dashboard can render charts while
    the memo is synthesised asynchronously via ``POST /session/{id}/memo/stream``.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="Opaque UUID for this session; pass to /memo/stream and /chat/stream.")
    request: AnalysisRequest
    portfolio: PortfolioMetrics
    assets: list[AssetMetrics]
    shock_scenarios: list[ShockScenarioResult]
    cumulative_returns: dict[str, list[object]] = Field(
        description="Keys: 'dates' (ISO strings), 'portfolio' (pct), 'benchmark' (pct).",
    )
    correlation_matrix: dict[str, object] = Field(
        description="Pearson correlation matrix. Keys: 'tickers' (list[str]), 'matrix' (list[list[float]]).",
    )


class SessionAnalysisResponse(AnalysisResponse):
    """Extended response — kept for backward-compat and test fixtures."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="Opaque UUID identifying this analysis session.")
    cumulative_returns: dict[str, list[object]] = Field(
        description=(
            "Daily cumulative returns for the line chart. "
            "Keys: 'dates' (ISO strings), 'portfolio' (pct), 'benchmark' (pct)."
        ),
    )
    correlation_matrix: dict[str, object] = Field(
        description="Pearson correlation matrix. Keys: 'tickers' (list[str]), 'matrix' (list[list[float]]).",
    )


class ChatRequest(BaseModel):
    """Payload for ``POST /chat/stream`` and ``POST /session/{id}/memo/stream``."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="Session ID returned by /analyze.")
    message: str = Field(min_length=1, max_length=4000, description="User's follow-up question.")