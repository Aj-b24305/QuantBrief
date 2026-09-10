from __future__ import annotations

import logging
from datetime import UTC, datetime

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from quantbrief import __version__
from quantbrief.agent import AgentSynthesizer
from quantbrief.calculator import InsufficientDataError, compute_metrics, weighted_portfolio_returns
from quantbrief.chat import build_chat_messages, stream_chat
from quantbrief.config import Settings, get_settings
from quantbrief.market_data import MarketDataError, fetch_market_data
from quantbrief.schemas import (
    AgentSynthesizedMemo,
    AnalysisRequest,
    AnalysisResponse,
    ChatRequest,
    MemoContext,
    MetricsOnlyResponse,
    SessionAnalysisResponse,
)
from quantbrief.state import (
    ChatMessage,
    InMemorySessionRepository,
    PortfolioSession,
    build_correlation_matrix,
    build_cumulative_returns,
    get_session_repo,
    new_session_id,
)

logger = logging.getLogger("quantbrief")

settings = get_settings()

app = FastAPI(
    title="QuantBrief",
    version=__version__,
    description=(
        "Deterministic portfolio risk engine (NumPy/Pandas) with LLM memo synthesis and "
        "interactive streaming chat grounded in pre-computed quantitative metrics."
    ),
)

# Allow Streamlit (port 8501) to reach the API during local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {
        "service": "QuantBrief",
        "docs": "/docs",
        "health": "/health",
        "analyze": "POST /analyze  (fast — metrics + charts, no LLM)",
        "memo": "POST /session/{id}/memo/stream  (streams LLM memo)",
        "chat": "POST /chat/stream  (streaming conversational chat)",
    }


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "quantbrief",
        "version": __version__,
        "timestamp": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# FAST: /analyze — deterministic metrics only (~10 s, no LLM)
# ---------------------------------------------------------------------------


@app.post("/analyze", response_model=MetricsOnlyResponse, tags=["risk"])
def analyze(
    payload: AnalysisRequest,
    repo: InMemorySessionRepository = Depends(get_session_repo),
) -> MetricsOnlyResponse:
    """Compute deterministic risk metrics and open a session.

    Returns immediately (~10 s) with all quantitative data and chart payloads.
    The risk memo is synthesised separately via ``POST /session/{id}/memo/stream``
    so charts render without waiting for the LLM.
    """
    try:
        # 1. Market data (yfinance) — typically 5–10 s
        data = fetch_market_data(payload.tickers, payload.benchmark, payload.lookback_days)
        portfolio, assets, shocks = compute_metrics(data, payload.weights, settings)

        # 2. Build MemoContext (metrics only — stored for later LLM + chat use)
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

        # 3. Chart payloads (pure NumPy/Pandas, deterministic)
        metrics_window = data.prices.loc[
            data.prices.index >= pd.Timestamp(data.metrics_start)
        ].dropna()
        asset_returns = metrics_window.pct_change().dropna(how="all")
        benchmark_window = data.benchmark.reindex(metrics_window.index).ffill()
        benchmark_returns = benchmark_window.pct_change().dropna()
        portfolio_daily = weighted_portfolio_returns(asset_returns, payload.weights)
        benchmark_daily = benchmark_returns.reindex(portfolio_daily.index)

        cum_returns = build_cumulative_returns(portfolio_daily, benchmark_daily)
        corr_matrix = build_correlation_matrix(asset_returns)

        # 4. Cache session (memo will be filled in by /memo/stream later)
        session_id = new_session_id()
        # Use a temporary AnalysisResponse with a placeholder memo so the
        # session is cache-able immediately; /memo/stream overwrites this.
        from quantbrief.agent import build_fallback_memo
        from quantbrief.schemas import MemoMeta

        placeholder_memo = build_fallback_memo(context)
        placeholder_meta = MemoMeta(synthesized_by_llm=False, note="pending — call /memo/stream")
        base_response = AnalysisResponse(
            request=payload,
            portfolio=portfolio,
            assets=assets,
            shock_scenarios=shocks,
            memo=placeholder_memo,
            memo_meta=placeholder_meta,
        )
        session = PortfolioSession(
            session_id=session_id,
            request=payload,
            response=base_response,
            cumulative_returns=cum_returns,
            correlation_matrix=corr_matrix,
        )
        session.__dict__["_context"] = context
        repo.save(session)

        return MetricsOnlyResponse(
            session_id=session_id,
            request=payload,
            portfolio=portfolio,
            assets=assets,
            shock_scenarios=shocks,
            cumulative_returns=cum_returns,
            correlation_matrix=corr_matrix,
        )

    except MarketDataError as exc:
        logger.warning("market data error: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except InsufficientDataError as exc:
        logger.warning("insufficient data: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("unexpected failure in /analyze")
        raise HTTPException(status_code=500, detail="internal error while running analysis") from exc


# ---------------------------------------------------------------------------
# STREAMING: /session/{id}/memo/stream — LLM memo as a stream of JSON chunks
# ---------------------------------------------------------------------------


@app.post("/session/{session_id}/memo/stream", tags=["risk"])
def memo_stream(
    session_id: str,
    repo: InMemorySessionRepository = Depends(get_session_repo),
) -> StreamingResponse:
    """Stream the LLM risk memo for an existing session.

    Delegates to :meth:`AgentSynthesizer.stream_synthesis` which calls
    ``create(stream=True)`` — tokens are forwarded to the client *as the model
    generates them*, not word-split from an already-complete response.

    Event types (NDJSON, one JSON object per line):
    - ``{"type": "token", "text": "..."}``   — raw LLM token
    - ``{"type": "done",  "memo": {...}, "memo_meta": {...}}``  — final validated object
    - ``{"type": "error", "detail": "..."}`` — unrecoverable error
    """
    import json

    session = repo.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    context: MemoContext | None = session.__dict__.get("_context")
    if context is None:
        raise HTTPException(status_code=500, detail="Session context missing — re-run /analyze.")

    synthesizer = AgentSynthesizer(settings)

    def _generate() -> object:
        try:
            for event in synthesizer.stream_synthesis(context):
                if event["type"] == "done":
                    # Update cached session with the validated memo
                    from quantbrief.schemas import AgentSynthesizedMemo as _Memo, MemoMeta as _Meta
                    memo = _Memo.model_validate(event["memo"])
                    memo_meta = _Meta.model_validate(event["memo_meta"])
                    session.response = AnalysisResponse(
                        request=session.request,
                        portfolio=session.response.portfolio,
                        assets=session.response.assets,
                        shock_scenarios=session.response.shock_scenarios,
                        memo=memo,
                        memo_meta=memo_meta,
                    )
                    repo.save(session)
                yield json.dumps(event) + "\n"
        except Exception as exc:  # noqa: BLE001
            logger.warning("memo_stream error: %s", exc)
            yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"

    return StreamingResponse(_generate(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# STREAMING: /chat/stream — conversational follow-up
# ---------------------------------------------------------------------------


@app.post("/chat/stream", tags=["chat"])
def chat_stream(
    payload: ChatRequest,
    repo: InMemorySessionRepository = Depends(get_session_repo),
) -> StreamingResponse:
    """Stream a conversational response grounded in the cached quantitative report.

    Re-uses ``AgentSynthesizer``'s pre-built OpenAI clients (local + Gemini) via
    the module-level ``stream_chat`` helper — no duplicate client construction.
    """
    session = repo.get(payload.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{payload.session_id}' not found. Run /analyze first.",
        )

    context: MemoContext | None = session.__dict__.get("_context")
    context_payload: dict = (
        context.model_dump(mode="json")
        if context is not None
        else {
            "portfolio": session.response.portfolio.model_dump(mode="json"),
            "assets": [a.model_dump(mode="json") for a in session.response.assets],
            "shock_scenarios": [s.model_dump(mode="json") for s in session.response.shock_scenarios],
            "memo": session.response.memo.model_dump(mode="json"),
        }
    )

    session.chat_history.append(ChatMessage(role="user", content=payload.message))
    messages = build_chat_messages(context_payload, session.chat_history[:-1], payload.message)

    # Re-use AgentSynthesizer's client — it already selected local vs Gemini.
    synth = AgentSynthesizer(settings)
    client = synth._client  # noqa: SLF001 — deliberate internal access to avoid duplicate construction
    model = synth.model
    timeout = settings.ollama_timeout_seconds

    def _token_generator() -> object:
        collected: list[str] = []
        for chunk in stream_chat(client, model, messages, timeout=timeout):
            collected.append(chunk)
            yield chunk
        full_reply = "".join(collected)
        session.chat_history.append(ChatMessage(role="assistant", content=full_reply))
        repo.save(session)

    return StreamingResponse(_token_generator(), media_type="text/plain; charset=utf-8")


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------


@app.get("/session/{session_id}/ping", tags=["session"])
def ping_session(
    session_id: str,
    repo: InMemorySessionRepository = Depends(get_session_repo),
) -> dict[str, str]:
    """Lightweight liveness check — returns 200 if session exists, 404 if not.

    Used by the Streamlit frontend to detect backend restarts without
    making an expensive LLM call.
    """
    if repo.get(session_id) is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"status": "alive", "session_id": session_id}


@app.post("/session/{session_id}/save", tags=["session"])
def save_session(
    session_id: str,
    repo: InMemorySessionRepository = Depends(get_session_repo),
) -> dict[str, str]:
    """Mark a session for durable persistence (Postgres placeholder)."""
    session = repo.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    repo.mark_for_persistence(session_id)
    return {"status": "queued", "session_id": session_id, "message": "Session flagged for persistence."}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the development server."""
    import uvicorn

    uvicorn.run("quantbrief.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()