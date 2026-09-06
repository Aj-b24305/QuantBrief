from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from risksentry.schemas import AnalysisRequest, AnalysisResponse


@dataclass
class ChatMessage:
    """A single turn in a chat conversation."""

    role: str  # "user" | "assistant"
    content: str


@dataclass
class PortfolioSession:
    """Everything cached per user session.

    Keyed by ``session_id`` (UUID string). Holds the full analysis
    response, raw time-series data for charts, and the running chat
    history. Design is intentionally flat so a future Postgres row
    can be added without restructuring the domain model.
    """

    session_id: str
    request: AnalysisRequest
    response: AnalysisResponse
    # Pre-formatted chart payloads (JSON-serialisable dicts)
    cumulative_returns: dict[str, list[Any]]  # {date: [...], portfolio: [...], benchmark: [...]}
    correlation_matrix: dict[str, Any]  # {tickers: [...], matrix: [[...]]}
    chat_history: list[ChatMessage] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Repository interface — swap InMemorySessionRepository for a
# PostgresSessionRepository without touching any endpoint code.
# ---------------------------------------------------------------------------


class SessionRepository(ABC):
    """Abstract repository — enables future Postgres swap."""

    @abstractmethod
    def get(self, session_id: str) -> PortfolioSession | None:
        """Fetch a session by ID, or ``None`` if not found."""

    @abstractmethod
    def save(self, session: PortfolioSession) -> None:
        """Persist (create or overwrite) a session."""

    @abstractmethod
    def delete(self, session_id: str) -> None:
        """Remove a session."""

    @abstractmethod
    def mark_for_persistence(self, session_id: str) -> None:
        """Placeholder: flag the session for export to a durable store."""


class InMemorySessionRepository(SessionRepository):
    """Process-local dict-backed store.

    Thread-safe for CPython's GIL; if you add async workers or
    multiple Uvicorn workers, swap this for a Redis-backed or
    SQLAlchemy implementation behind the same interface.
    """

    def __init__(self) -> None:
        self._store: dict[str, PortfolioSession] = {}
        self._pending_persist: set[str] = set()

    def get(self, session_id: str) -> PortfolioSession | None:
        return self._store.get(session_id)

    def save(self, session: PortfolioSession) -> None:
        self._store[session.session_id] = session

    def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)
        self._pending_persist.discard(session_id)

    def mark_for_persistence(self, session_id: str) -> None:
        """Flag a session as ready to be written to a durable store.

        In a real implementation this would enqueue a background task
        that serialises the session to Postgres. For now it is a no-op
        placeholder so the Streamlit "Save Session" button has an
        actual backend hook.
        """
        if session_id in self._store:
            self._pending_persist.add(session_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def pending_ids(self) -> set[str]:
        """Return IDs currently flagged for persistence."""
        return set(self._pending_persist)


# ---------------------------------------------------------------------------
# Helpers to build chart payloads from raw market data
# ---------------------------------------------------------------------------


def build_cumulative_returns(
    portfolio_daily: pd.Series,
    benchmark_daily: pd.Series,
) -> dict[str, list[Any]]:
    """Produce a JSON-serialisable cumulative return time-series.

    Both series are aligned to their common index and expressed as
    percentage gain/loss from the first common observation.
    """
    combined = pd.DataFrame(
        {"portfolio": portfolio_daily, "benchmark": benchmark_daily}
    ).dropna()
    cum = (1.0 + combined).cumprod() - 1.0  # fractional cumulative return
    return {
        "dates": [str(d.date()) for d in cum.index],
        "portfolio": [round(v * 100, 4) for v in cum["portfolio"]],  # pct
        "benchmark": [round(v * 100, 4) for v in cum["benchmark"]],  # pct
    }


def build_correlation_matrix(asset_returns: pd.DataFrame) -> dict[str, Any]:
    """Produce a JSON-serialisable correlation matrix from daily returns."""
    corr = asset_returns.corr(method="pearson")
    return {
        "tickers": list(corr.columns),
        "matrix": [[round(v, 4) for v in row] for row in corr.values.tolist()],
    }


# ---------------------------------------------------------------------------
# Module-level singleton — FastAPI endpoints use this via dependency injection
# ---------------------------------------------------------------------------

_repo: InMemorySessionRepository | None = None


def get_session_repo() -> InMemorySessionRepository:
    """Return the process-wide session repository singleton."""
    global _repo
    if _repo is None:
        _repo = InMemorySessionRepository()
    return _repo


def new_session_id() -> str:
    """Generate a URL-safe session identifier."""
    return str(uuid.uuid4())
