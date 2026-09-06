from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient
from pydantic import ValidationError

from quantbrief.agent import SYSTEM_PROMPT, AgentSynthesizer, build_fallback_memo, parse_memo
from quantbrief.calculator import (
    InsufficientDataError,
    annualized_return,
    annualized_volatility,
    beta,
    compute_metrics,
    historical_var,
    sharpe_ratio,
    weighted_portfolio_returns,
)
from quantbrief.config import Settings
from quantbrief.main import app
from quantbrief.market_data import MarketData, MarketDataError, download_window, fetch_market_data
from quantbrief.schemas import (
    AgentSynthesizedMemo,
    AnalysisRequest,
    AnalysisResponse,
    MemoContext,
    MemoMeta,
    MetricsOnlyResponse,
    SessionAnalysisResponse,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers (fully synthetic — tests never touch the network)
# ---------------------------------------------------------------------------


def utc_today() -> date:
    return datetime.now(UTC).date()


def make_prices(
    tickers: tuple[str, ...] = ("AAPL", "MSFT"),
    start: str = "2008-06-01",
    seed: int = 7,
) -> pd.DataFrame:
    """Random-walk adjusted-close prices covering every shock scenario window."""
    index = pd.bdate_range(start=start, end=pd.Timestamp(utc_today()))
    rng = np.random.default_rng(seed)
    prices: dict[str, np.ndarray] = {}
    for i, ticker in enumerate(tickers):
        rets = rng.normal(0.0003 * (i + 1), 0.012 * (i + 1), len(index) - 1)
        prices[ticker] = 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + rets)])
    return pd.DataFrame(prices, index=index)


def make_benchmark(seed: int = 3, start: str = "2008-06-01") -> pd.Series:
    index = pd.bdate_range(start=start, end=pd.Timestamp(utc_today()))
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.009, len(index) - 1)
    return pd.Series(100.0 * np.concatenate([[1.0], np.cumprod(1.0 + rets)]), index=index, name="SPY")


def make_market_data(
    tickers: tuple[str, ...] = ("AAPL", "MSFT"),
    seed: int = 7,
    start: str = "2008-06-01",
) -> MarketData:
    prices = make_prices(tickers, start=start, seed=seed)
    benchmark = make_benchmark(seed=seed + 1, start=start)
    return MarketData(
        prices=prices,
        benchmark=benchmark,
        metrics_start=utc_today() - timedelta(days=365),
        metrics_end=utc_today(),
    )


@pytest.fixture
def settings() -> Settings:
    # gemini_api_key pinned to None so tests are hermetic even if a real key
    # exists in the local .env / environment.
    return Settings(llm_provider="ollama", llm_model="test-model", gemini_api_key=None)


@pytest.fixture
def market_data() -> MarketData:
    return make_market_data()


@pytest.fixture
def memo_context(settings: Settings, market_data: MarketData) -> MemoContext:
    portfolio, assets, shocks = compute_metrics(market_data, [0.6, 0.4], settings)
    return MemoContext(
        generated_at=datetime.now(UTC),
        tickers=["AAPL", "MSFT"],
        weights=[0.6, 0.4],
        benchmark="SPY",
        lookback_days=365,
        risk_free_rate=settings.risk_free_rate,
        portfolio=portfolio,
        assets=assets,
        shock_scenarios=shocks,
    )


# ---------------------------------------------------------------------------
# Schemas / validation (Pydantic v2, strict)
# ---------------------------------------------------------------------------


def test_request_accepts_valid_portfolio() -> None:
    request = AnalysisRequest(tickers=["AAPL", "MSFT"], weights=[0.5, 0.5])
    assert request.weights == [0.5, 0.5]


def test_request_normalizes_tickers_and_benchmark() -> None:
    request = AnalysisRequest(tickers=[" aapl ", "msft"], weights=[0.5, 0.5], benchmark=" spy ")
    assert request.tickers == ["AAPL", "MSFT"]
    assert request.benchmark == "SPY"


def test_request_rejects_length_mismatch() -> None:
    with pytest.raises(ValidationError, match="identical length"):
        AnalysisRequest(tickers=["AAPL", "MSFT"], weights=[1.0])


def test_request_rejects_short_positions() -> None:
    with pytest.raises(ValidationError, match="short positions"):
        AnalysisRequest(tickers=["AAPL", "MSFT"], weights=[1.5, -0.5])


def test_request_rejects_weights_not_summing_to_one() -> None:
    with pytest.raises(ValidationError, match="sum to 1.0"):
        AnalysisRequest(tickers=["AAPL", "MSFT"], weights=[0.3, 0.3])


def test_request_rejects_duplicate_tickers() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        AnalysisRequest(tickers=["AAPL", "AAPL"], weights=[0.5, 0.5])


def test_request_rejects_empty_ticker_string() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        AnalysisRequest(tickers=["AAPL", "  "], weights=[0.5, 0.5])


def test_request_rejects_out_of_range_lookback() -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest(tickers=["AAPL"], weights=[1.0], lookback_days=10)


def test_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        AnalysisRequest(tickers=["AAPL"], weights=[1.0], unexpected="x")  # type: ignore[call-arg]


def test_memo_model_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        AgentSynthesizedMemo(
            title="t",
            summary="s",
            key_risks=["r"],
            recommendations=["rec"],
            stress_test_outlook="o",
            confidence="medium",
            hallucinated_field="no",  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# Calculator — pure math primitives
# ---------------------------------------------------------------------------


def test_annualized_return_doubling_over_one_year() -> None:
    index = pd.bdate_range("2020-01-01", periods=253)  # 252 return periods
    prices = pd.Series(np.linspace(100.0, 200.0, 253), index=index)
    assert annualized_return(prices) == pytest.approx(1.0, abs=1e-9)


def test_annualized_return_flat_series_is_zero() -> None:
    prices = pd.Series([100.0] * 60)
    assert annualized_return(prices) == pytest.approx(0.0, abs=1e-12)


def test_annualized_return_requires_two_observations() -> None:
    with pytest.raises(InsufficientDataError):
        annualized_return(pd.Series([100.0]))


def test_annualized_volatility_constant_returns_is_zero() -> None:
    assert annualized_volatility(pd.Series([0.001] * 100)) == pytest.approx(0.0, abs=1e-12)


def test_annualized_volatility_scales_with_sqrt_252() -> None:
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0, 0.01, 1000))
    assert annualized_volatility(returns) == pytest.approx(0.01 * np.sqrt(252), rel=0.05)


def test_sharpe_ratio_formula() -> None:
    assert sharpe_ratio(0.10, 0.20, 0.045) == pytest.approx((0.10 - 0.045) / 0.20)


def test_sharpe_ratio_undefined_for_zero_vol() -> None:
    assert sharpe_ratio(0.10, 0.0, 0.045) is None


def test_beta_linear_exposure() -> None:
    rng = np.random.default_rng(0)
    benchmark_returns = pd.Series(rng.normal(0.001, 0.01, 300))
    stock_returns = 2.0 * benchmark_returns
    assert beta(stock_returns, benchmark_returns) == pytest.approx(2.0, abs=1e-9)


def test_beta_undefined_for_constant_benchmark() -> None:
    assert beta(pd.Series([0.01, 0.02, 0.03]), pd.Series([0.001, 0.001, 0.001])) is None


def test_beta_undefined_for_insufficient_data() -> None:
    assert beta(pd.Series([0.01]), pd.Series([0.02])) is None


def test_historical_var_matches_empirical_percentile() -> None:
    values = np.linspace(-0.05, 0.05, 100)
    assert historical_var(pd.Series(values), 0.95) == pytest.approx(-np.percentile(values, 5))


def test_historical_var_is_positive_loss() -> None:
    assert historical_var(pd.Series(np.linspace(-0.1, 0.0, 50)), 0.95) > 0


def test_historical_var_empty_raises() -> None:
    with pytest.raises(InsufficientDataError):
        historical_var(pd.Series(dtype=float))


def test_weighted_portfolio_returns() -> None:
    returns = pd.DataFrame({"A": [0.01, -0.02], "B": [0.005, 0.01]})
    result = weighted_portfolio_returns(returns, [0.5, 0.5])
    assert list(result) == pytest.approx([0.0075, -0.005])


def test_weighted_portfolio_returns_skips_missing_asset_days() -> None:
    returns = pd.DataFrame({"A": [0.01, np.nan], "B": [0.02, 0.01]})
    result = weighted_portfolio_returns(returns, [0.5, 0.5])
    assert result.iloc[0] == pytest.approx(0.015)
    assert result.iloc[1] == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# Calculator — compute_metrics integration on synthetic data
# ---------------------------------------------------------------------------


def test_compute_metrics_shape_and_sanity(settings: Settings, market_data: MarketData) -> None:
    portfolio, assets, shocks = compute_metrics(market_data, [0.6, 0.4], settings)

    assert portfolio.observation_days >= 30
    assert portfolio.var_95_daily > 0
    assert portfolio.annualized_volatility > 0
    assert portfolio.sharpe_ratio is not None
    assert portfolio.start_date <= portfolio.end_date

    assert len(assets) == 2
    assert [a.weight for a in assets] == [0.6, 0.4]
    assert all(a.annualized_volatility > 0 for a in assets)

    assert len(shocks) == 3  # matches SCENARIOS
    for shock in shocks:
        assert shock.id in {"covid_2020", "rate_shock_2022", "gfc_2008"}
        assert shock.coverage_pct is not None and shock.coverage_pct > 0


def test_compute_metrics_beta_reflects_exposure(settings: Settings) -> None:
    index = pd.bdate_range(end=pd.Timestamp(utc_today()), periods=600)
    rng = np.random.default_rng(11)
    benchmark_ret = rng.normal(0.0004, 0.01, len(index) - 1)
    a_ret = 1.5 * benchmark_ret + rng.normal(0.0, 0.0005, len(index) - 1)
    b_ret = 0.5 * benchmark_ret + rng.normal(0.0, 0.0005, len(index) - 1)

    def series_from(rets: np.ndarray) -> np.ndarray:
        return 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + rets)])

    data = MarketData(
        prices=pd.DataFrame({"AAPL": series_from(a_ret), "MSFT": series_from(b_ret)}, index=index),
        benchmark=pd.Series(series_from(benchmark_ret), index=index),
        metrics_start=(index[-1] - pd.Timedelta(days=365)).date(),
        metrics_end=index[-1].date(),
    )
    portfolio, assets, _ = compute_metrics(data, [0.6, 0.4], settings)

    assert assets[0].beta == pytest.approx(1.5, abs=0.1)
    assert assets[1].beta == pytest.approx(0.5, abs=0.1)
    assert portfolio.beta == pytest.approx(0.6 * 1.5 + 0.4 * 0.5, abs=0.1)


def test_scenario_replays_known_shock(settings: Settings) -> None:
    crash_date = pd.Timestamp("2020-03-20")
    baseline = make_market_data(seed=7)
    crashed = make_market_data(seed=7)  # identical synthetic data, then crashed
    for column in crashed.prices.columns:
        crashed.prices.loc[crashed.prices.index > crash_date, column] *= 0.7  # -30% jump

    _, _, baseline_shocks = compute_metrics(baseline, [0.5, 0.5], settings)
    _, _, crash_shocks = compute_metrics(crashed, [0.5, 0.5], settings)
    baseline_covid = next(s for s in baseline_shocks if s.id == "covid_2020")
    crash_covid = next(s for s in crash_shocks if s.id == "covid_2020")

    assert crash_covid.portfolio_return is not None
    assert crash_covid.portfolio_return < 0
    # A one-day -30% crash must move the window return by ~25-45 percentage points
    # (compounding of the surrounding drift dilutes the exact -30%).
    delta = baseline_covid.portfolio_return - crash_covid.portfolio_return
    assert 0.25 <= delta <= 0.45


def test_compute_metrics_raises_on_insufficient_window(settings: Settings) -> None:
    index = pd.bdate_range(end=pd.Timestamp(utc_today()), periods=40)
    rng = np.random.default_rng(5)
    rets = rng.normal(0.0, 0.01, len(index) - 1)
    values = 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + rets)])
    prices = pd.DataFrame({"A": values}, index=index)
    data = MarketData(
        prices=prices,
        benchmark=prices["A"].copy(),
        metrics_start=utc_today() - timedelta(days=2),  # tiny window -> few rows
        metrics_end=utc_today(),
    )
    with pytest.raises(InsufficientDataError):
        compute_metrics(data, [1.0], settings)


# ---------------------------------------------------------------------------
# Market data ingestion (yfinance mocked)
# ---------------------------------------------------------------------------


def fake_history_frame(close: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"Close": close})
    frame.index.name = "Date"
    return frame


class FakeTicker:
    """Stands in for yfinance.Ticker; behavior driven by ``registry``."""

    registry: ClassVar[dict[str, object]] = {}

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def history(self, **kwargs: object) -> pd.DataFrame:
        entry = FakeTicker.registry.get(self.symbol)
        if isinstance(entry, Exception):
            raise entry
        if entry is None:
            return pd.DataFrame()
        return entry  # type: ignore[return-value]


@pytest.fixture
def fake_yfinance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("quantbrief.market_data.yf.Ticker", FakeTicker)


def test_fetch_market_data_ok(fake_yfinance: None) -> None:
    prices = make_prices(start="2023-01-01", seed=9)
    benchmark = make_benchmark(seed=10, start="2023-01-01")
    FakeTicker.registry = {
        "SPY": fake_history_frame(benchmark),
        "AAPL": fake_history_frame(prices["AAPL"]),
        "MSFT": fake_history_frame(prices["MSFT"]),
    }
    data = fetch_market_data(["AAPL", "MSFT"], "SPY", 365)
    assert list(data.prices.columns) == ["AAPL", "MSFT"]
    assert data.prices.index.is_monotonic_increasing
    assert not data.benchmark.empty


def test_fetch_market_data_reports_missing_ticker(fake_yfinance: None) -> None:
    benchmark = make_benchmark(seed=10, start="2023-01-01")
    FakeTicker.registry = {
        "SPY": fake_history_frame(benchmark),
        "AAPL": fake_history_frame(make_prices(start="2023-01-01", seed=9)["AAPL"]),
        # MSFT deliberately absent -> empty history
    }
    with pytest.raises(MarketDataError) as exc_info:
        fetch_market_data(["AAPL", "MSFT"], "SPY", 365)
    assert "MSFT" in str(exc_info.value)
    assert exc_info.value.status_code == 400


def test_fetch_market_data_network_failure_maps_to_503(fake_yfinance: None) -> None:
    FakeTicker.registry = {
        "SPY": requests.exceptions.ConnectionError("boom"),
        "AAPL": requests.exceptions.ConnectionError("boom"),
        "MSFT": requests.exceptions.ConnectionError("boom"),
    }
    with pytest.raises(MarketDataError) as exc_info:
        fetch_market_data(["AAPL", "MSFT"], "SPY", 365)
    assert exc_info.value.network is True
    assert exc_info.value.status_code == 503


def test_download_window_covers_metrics_and_shocks() -> None:
    start, end = download_window(365)
    assert start <= date(2008, 8, 17)  # before the earliest shock window (GFC 2008, minus buffer)
    assert end > utc_today()


# ---------------------------------------------------------------------------
# Agent — memo synthesis (OpenAI client mocked)
# ---------------------------------------------------------------------------


def fake_openai_client(content: str) -> tuple[SimpleNamespace, dict[str, object]]:
    captured: dict[str, object] = {}

    def create(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), captured


def valid_memo_payload() -> dict[str, object]:
    return {
        "title": "AAPL/MSFT Risk Memo",
        "summary": "Moderate volatility with a positive Sharpe ratio.",
        "key_risks": ["Single-name concentration in AAPL.", "Elevated beta to SPY."],
        "recommendations": ["Rebalance toward lower-beta assets.", "Set VaR-based stop-loss limits."],
        "stress_test_outlook": "The portfolio would lose roughly 10-15% in a 2022-style shock.",
        "confidence": "medium",
        "caveats": ["Memo is illustrative only."],
    }


def test_agent_parses_llm_json(settings: Settings, memo_context: MemoContext) -> None:
    client, captured = fake_openai_client(json.dumps(valid_memo_payload()))
    synthesizer = AgentSynthesizer(settings)
    synthesizer._client = client  # type: ignore[attr-defined]
    synthesizer._preflight = lambda *_: True  # model is "active" in the registry

    memo, meta = synthesizer.synthesize(memo_context)

    assert meta.synthesized_by_llm is True
    assert meta.model == "test-model"
    assert memo.title == "AAPL/MSFT Risk Memo"
    assert memo.confidence == "medium"

    # The endpoint contract is honored: JSON object mode is enforced.
    assert captured["response_format"] == {"type": "json_object"}
    # The LLM only ever receives pre-computed metrics, never raw prices.
    user_prompt = str(captured["messages"])  # type: ignore[index]
    assert "prices" not in user_prompt


def test_agent_falls_back_on_invalid_json(settings: Settings, memo_context: MemoContext) -> None:
    client, _ = fake_openai_client("this is not json")
    synthesizer = AgentSynthesizer(settings)
    synthesizer._client = client  # type: ignore[attr-defined]
    synthesizer._preflight = lambda *_: True  # model is "active" in the registry

    memo, meta = synthesizer.synthesize(memo_context)

    assert meta.synthesized_by_llm is False
    assert isinstance(memo, AgentSynthesizedMemo)
    assert meta.note is not None


def test_agent_falls_back_on_schema_violation(settings: Settings, memo_context: MemoContext) -> None:
    client, _ = fake_openai_client(json.dumps({"title": "missing everything else"}))
    synthesizer = AgentSynthesizer(settings)
    synthesizer._client = client  # type: ignore[attr-defined]
    synthesizer._preflight = lambda *_: True  # model is "active" in the registry

    memo, meta = synthesizer.synthesize(memo_context)
    assert meta.synthesized_by_llm is False
    assert isinstance(memo, AgentSynthesizedMemo)


def test_agent_falls_back_on_llm_error(settings: Settings, memo_context: MemoContext) -> None:
    def boom(**kwargs: object) -> None:
        raise requests.exceptions.ConnectionError("ollama down")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=boom)))
    synthesizer = AgentSynthesizer(settings)
    synthesizer._client = client  # type: ignore[attr-defined]
    synthesizer._preflight = lambda *_: True  # model is "active" in the registry

    _, meta = synthesizer.synthesize(memo_context)
    assert meta.synthesized_by_llm is False
    assert "ollama down" in (meta.note or "")


def test_parse_memo_handles_code_fences() -> None:
    content = f"```json\n{json.dumps(valid_memo_payload())}\n```"
    memo = parse_memo(content)
    assert memo.title == "AAPL/MSFT Risk Memo"


def test_fallback_memo_is_deterministic_and_uses_metrics(memo_context: MemoContext) -> None:
    first = build_fallback_memo(memo_context)
    second = build_fallback_memo(memo_context)
    assert first == second
    assert any("VaR" in risk for risk in first.key_risks)
    assert first.confidence in {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# Separation of concerns — the LLM never receives raw math inputs
# ---------------------------------------------------------------------------


def test_memo_context_contains_only_precomputed_metrics(memo_context: MemoContext) -> None:
    payload = json.dumps(memo_context.model_dump(mode="json"))
    assert "prices" not in payload
    assert "Close" not in payload
    assert "returns" not in payload  # raw return series never leave the engine


def test_system_prompt_forbids_calculation() -> None:
    prompt = SYSTEM_PROMPT.lower()
    assert "never calculate" in prompt
    assert "quote them as given" in prompt


# ---------------------------------------------------------------------------
# API endpoints (TestClient; market data + LLM fully mocked)
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class FakeAgentSynthesizer:
    """Deterministic stand-in for the LLM agent used by the API tests."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def synthesize(self, context: MemoContext) -> tuple[AgentSynthesizedMemo, MemoMeta]:
        return (
            AgentSynthesizedMemo(
                title="Fake memo",
                summary=f"Analyzed {', '.join(context.tickers)}.",
                key_risks=["Fake risk"],
                recommendations=["Fake recommendation"],
                stress_test_outlook="Fake outlook",
                confidence="medium",
                caveats=["Fake caveat"],
            ),
            MemoMeta(synthesized_by_llm=True, model="fake-model", elapsed_ms=1),
        )


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "quantbrief"
    assert "version" in body


def test_analyze_success(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "quantbrief.main.fetch_market_data",
        lambda tickers, benchmark, lookback: make_market_data(tuple(tickers)),
    )
    monkeypatch.setattr("quantbrief.main.AgentSynthesizer", FakeAgentSynthesizer)

    response = client.post(
        "/analyze",
        json={"tickers": ["AAPL", "MSFT"], "weights": [0.6, 0.4], "lookback_days": 365, "benchmark": "SPY"},
    )
    assert response.status_code == 200
    body = response.json()

    # /analyze now returns MetricsOnlyResponse (fast path, no LLM)
    MetricsOnlyResponse.model_validate(body)

    assert body["request"]["tickers"] == ["AAPL", "MSFT"]
    assert body["portfolio"]["var_95_daily"] > 0
    assert body["portfolio"]["annualized_volatility"] > 0
    assert len(body["assets"]) == 2
    assert [asset["weight"] for asset in body["assets"]] == [0.6, 0.4]
    assert len(body["shock_scenarios"]) == 3
    assert "session_id" in body
    assert "cumulative_returns" in body
    assert "correlation_matrix" in body


def test_analyze_rejects_length_mismatch(client: TestClient) -> None:
    response = client.post("/analyze", json={"tickers": ["AAPL", "MSFT"], "weights": [1.0]})
    assert response.status_code == 422


def test_analyze_rejects_short_positions(client: TestClient) -> None:
    response = client.post("/analyze", json={"tickers": ["AAPL", "MSFT"], "weights": [1.5, -0.5]})
    assert response.status_code == 422
    assert "short" in response.json()["detail"][0]["msg"]


def test_analyze_rejects_weights_not_summing_to_one(client: TestClient) -> None:
    response = client.post("/analyze", json={"tickers": ["AAPL", "MSFT"], "weights": [0.3, 0.3]})
    assert response.status_code == 422
    assert "sum to 1.0" in response.json()["detail"][0]["msg"]


def test_analyze_missing_ticker_returns_400(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(tickers: list[str], benchmark: str, lookback: int) -> MarketData:
        raise MarketDataError("no price data for BADTICK", missing=["BADTICK"])

    monkeypatch.setattr("quantbrief.main.fetch_market_data", boom)
    response = client.post("/analyze", json={"tickers": ["AAPL"], "weights": [1.0]})
    assert response.status_code == 400
    assert "BADTICK" in response.json()["detail"]


def test_analyze_network_failure_returns_503(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(tickers: list[str], benchmark: str, lookback: int) -> MarketData:
        raise MarketDataError("network failure", missing=["AAPL"], network=True)

    monkeypatch.setattr("quantbrief.main.fetch_market_data", boom)
    response = client.post("/analyze", json={"tickers": ["AAPL"], "weights": [1.0]})
    assert response.status_code == 503


def test_analyze_insufficient_data_returns_422(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantbrief.calculator import InsufficientDataError as CalcError

    def boom(tickers: list[str], benchmark: str, lookback: int) -> MarketData:
        raise CalcError("only 5 complete trading days available")

    monkeypatch.setattr("quantbrief.main.fetch_market_data", boom)
    response = client.post("/analyze", json={"tickers": ["AAPL"], "weights": [1.0]})
    assert response.status_code == 422
    assert "5 complete trading days" in response.json()["detail"]