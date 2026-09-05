from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException

from risksentry import __version__
from risksentry.agent import AgentSynthesizer
from risksentry.calculator import InsufficientDataError, compute_metrics
from risksentry.config import get_settings
from risksentry.market_data import MarketDataError, fetch_market_data
from risksentry.schemas import AnalysisRequest, AnalysisResponse, MemoContext

logger = logging.getLogger("risksentry")

settings = get_settings()

app = FastAPI(
    title="RiskSentry",
    version=__version__,
    description=(
        "Deterministic portfolio risk engine (NumPy/Pandas) with LLM memo synthesis. "
        "All quantitative metrics are computed locally; the LLM only restates them in a memo."
    ),
)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {"service": "RiskSentry", "docs": "/docs", "health": "/health", "analyze": "/analyze"}


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "risksentry",
        "version": __version__,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.post("/analyze", response_model=AnalysisResponse, tags=["risk"])
def analyze(payload: AnalysisRequest) -> AnalysisResponse:
    """Compute deterministic portfolio risk metrics and synthesize a risk memo.

    The endpoint is synchronous; FastAPI runs it in a threadpool so the blocking
    I/O (yfinance download, LLM call) never stalls the event loop.
    """
    try:
        data = fetch_market_data(payload.tickers, payload.benchmark, payload.lookback_days)
        portfolio, assets, shocks = compute_metrics(data, payload.weights, settings)

        context = MemoContext(
            generated_at=datetime.now(UTC),
            tickers=payload.tickers,
            weights=payload.weights,
            benchmark=payload.benchmark,
            lookback_days=payload.lookback_days,
            risk_free_rate=settings.risk_free_rate,
            portfolio=portfolio,
            assets=assets,
            shock_scenarios=shocks,
        )
        memo, memo_meta = AgentSynthesizer(settings).synthesize(context)

        return AnalysisResponse(
            request=payload,
            portfolio=portfolio,
            assets=assets,
            shock_scenarios=shocks,
            memo=memo,
            memo_meta=memo_meta,
        )
    except MarketDataError as exc:
        logger.warning("market data error: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except InsufficientDataError as exc:
        logger.warning("insufficient data: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive boundary
        logger.exception("unexpected failure in /analyze")
        raise HTTPException(status_code=500, detail="internal error while running analysis") from exc


def main() -> None:
    """Run the development server: ``risksentry`` or ``uvicorn risksentry.main:app``."""
    import uvicorn

    uvicorn.run("risksentry.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()