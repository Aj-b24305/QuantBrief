from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import ClassVar, Self

import httpx
import pytest

# Reuse the hermetic synthetic market-data builders from the core test suite.
from test_risk_engine import make_market_data, valid_memo_payload

from quantbrief.agent import CORRECTION_PROMPT_TEMPLATE, AgentSynthesizer
from quantbrief.calculator import compute_metrics
from quantbrief.config import Settings
from quantbrief.llm.model_checker import is_model_available, ollama_native_tags_url
from quantbrief.llm.sanitizer import MalformedJSONError, extract_json_from_thinking
from quantbrief.schemas import AgentSynthesizedMemo, MemoContext

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_provider="ollama",
        llm_model="test-model",
        gemini_api_key=None,
        gemini_model="gemini-fake",
    )


@pytest.fixture
def memo_context(settings: Settings) -> MemoContext:
    data = make_market_data()
    portfolio, assets, shocks = compute_metrics(data, [0.6, 0.4], settings)
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


def valid_memo_json() -> str:
    return json.dumps(valid_memo_payload())


class ScriptedCompletions:
    """Scripted OpenAI-compatible completions that record every call.

    Each ``create()`` pops the next queued content string (or raises the
    configured error). ``calls`` keeps the kwargs of every invocation so tests
    can assert on temperature, prompts, and call ordering.
    """

    def __init__(self, *contents: str, error: Exception | None = None) -> None:
        self._queue = list(contents)
        self._error = error
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        if not self._queue:
            raise AssertionError("ScriptedCompletions ran out of scripted responses")
        content = self._queue.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def as_openai_client(completions: ScriptedCompletions) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completions.create)))


def make_synthesizer(
    settings: Settings,
    local: ScriptedCompletions,
    gemini: ScriptedCompletions | None = None,
    preflight: bool = True,
) -> AgentSynthesizer:
    synthesizer = AgentSynthesizer(settings)
    synthesizer._client = as_openai_client(local)  # type: ignore[attr-defined]
    synthesizer._preflight = lambda *_: preflight  # type: ignore[attr-defined]
    if gemini is not None:
        synthesizer._gemini_client = as_openai_client(gemini)  # type: ignore[attr-defined]
    return synthesizer


# ---------------------------------------------------------------------------
# 1) Pre-flight model checker (mocked Ollama /api/tags responses)
# ---------------------------------------------------------------------------


class FakeAsyncClient:
    """Stand-in for ``httpx.AsyncClient`` driven by ``behavior``."""

    behavior: ClassVar[dict[str, object]] = {}

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    async def get(self, url: str) -> object:
        behavior = FakeAsyncClient.behavior
        if "exc" in behavior:
            raise behavior["exc"]  # type: ignore[misc]
        return SimpleNamespace(
            status_code=behavior.get("status", 200),
            json=lambda: behavior.get("payload", {}),
        )


def _registry_payload(*names: str) -> dict[str, object]:
    return {"models": [{"name": name, "model": name} for name in names]}


@pytest.fixture
def fake_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    import quantbrief.llm.model_checker as checker

    fake_module = SimpleNamespace(AsyncClient=FakeAsyncClient, ConnectError=httpx.ConnectError)
    monkeypatch.setattr(checker, "httpx", fake_module)
    FakeAsyncClient.behavior = {"status": 200, "payload": _registry_payload("qwen3.5:9b")}


def test_model_available_when_registry_lists_model(fake_ollama: None) -> None:
    available = asyncio.run(is_model_available("qwen3.5:9b", "http://localhost:11434/v1"))
    assert available is True


def test_model_available_matches_prefixed_equivalent(fake_ollama: None) -> None:
    # Requesting the family name "qwen3.5" is satisfied by registry "qwen3.5:9b".
    available = asyncio.run(is_model_available("qwen3.5", "http://localhost:11434/v1"))
    assert available is True


def test_model_missing_from_registry_returns_false(fake_ollama: None) -> None:
    FakeAsyncClient.behavior = {"status": 200, "payload": _registry_payload("llama3.1:8b")}
    available = asyncio.run(is_model_available("qwen3.5:9b", "http://localhost:11434/v1"))
    assert available is False


def test_non_200_response_returns_false(fake_ollama: None) -> None:
    FakeAsyncClient.behavior = {"status": 500, "payload": {}}
    available = asyncio.run(is_model_available("qwen3.5:9b", "http://localhost:11434/v1"))
    assert available is False


def test_unreachable_ollama_returns_false_without_raising(fake_ollama: None) -> None:
    FakeAsyncClient.behavior = {"exc": httpx.ConnectError("connection refused")}
    available = asyncio.run(is_model_available("qwen3.5:9b", "http://localhost:11434/v1"))
    assert available is False


def test_empty_model_name_returns_false(fake_ollama: None) -> None:
    available = asyncio.run(is_model_available("   ", "http://localhost:11434/v1"))
    assert available is False


def test_ollama_native_tags_url_normalizes_v1_suffix() -> None:
    assert ollama_native_tags_url("http://localhost:11434/v1") == "http://localhost:11434/api/tags"
    assert ollama_native_tags_url("http://localhost:11434") == "http://localhost:11434/api/tags"
    assert ollama_native_tags_url("http://localhost:11434/v1/") == "http://localhost:11434/api/tags"


# ---------------------------------------------------------------------------
# 2) Regex extractor — thinking tags, fences, trailing text, schema checks
# ---------------------------------------------------------------------------


def test_extract_strips_think_block_with_nested_braces() -> None:
    raw = (
        "<think>The user wants a memo. I should restate, not compute.\n"
        "Internal note: {this must never reach the parser} and [arrays] too.</think>\n"
        f"{valid_memo_json()}\n"
        "Hope this helps!"
    )
    data = extract_json_from_thinking(raw)
    assert data["title"] == "AAPL/MSFT Risk Memo"
    assert data["confidence"] == "medium"


def test_extract_handles_think_then_code_fence() -> None:
    raw = f"<think>Reasoning here.</think>\n```json\n{valid_memo_json()}\n```"
    data = extract_json_from_thinking(raw)
    assert data["title"] == "AAPL/MSFT Risk Memo"


def test_extract_handles_trailing_second_json_object() -> None:
    raw = f"{valid_memo_json()}{json.dumps({'noise': True})}"
    data = extract_json_from_thinking(raw)
    assert data["title"] == "AAPL/MSFT Risk Memo"


def test_extract_handles_plain_json_without_think_tags() -> None:
    data = extract_json_from_thinking(valid_memo_json())
    assert data["confidence"] == "medium"


def test_extract_raises_on_non_json_output() -> None:
    with pytest.raises(MalformedJSONError):
        extract_json_from_thinking("I am sorry, I cannot produce JSON today.")


def test_extract_raises_on_schema_violation() -> None:
    with pytest.raises(MalformedJSONError):
        extract_json_from_thinking(json.dumps({"title": "missing the rest of the schema"}))


def test_extract_raises_on_empty_output() -> None:
    with pytest.raises(MalformedJSONError):
        extract_json_from_thinking("   ")


# ---------------------------------------------------------------------------
# 3) Local self-correction loop (mocked sequential completions)
# ---------------------------------------------------------------------------


def test_self_correction_repairs_malformed_local_output(
    settings: Settings,
    memo_context: MemoContext,
) -> None:
    malformed = "<think>Let me draft the memo...</think>\nHere is my draft: {\"title\": \"incomplete\"}"
    local = ScriptedCompletions(malformed, valid_memo_json())
    synthesizer = make_synthesizer(settings, local)

    memo, meta = synthesizer.synthesize(memo_context)

    assert meta.synthesized_by_llm is True
    assert meta.model == "test-model"
    assert memo.title == "AAPL/MSFT Risk Memo"
    assert isinstance(memo, AgentSynthesizedMemo)
    assert len(local.calls) == 2  # initial generation + one repair pass
    assert local.calls[0]["temperature"] == pytest.approx(0.2)
    # Repair pass is zero-temperature and targets the same local model.
    assert local.calls[1]["temperature"] == 0.0
    repair_prompt = str(local.calls[1]["messages"])
    assert "Convert the following financial analysis into strictly valid JSON" in repair_prompt
    assert "incomplete" in repair_prompt  # raw (malformed) output is fed back


def test_failed_self_correction_falls_back_to_deterministic_memo(
    settings: Settings,
    memo_context: MemoContext,
) -> None:
    # Both the initial output AND the repaired output stay malformed.
    local = ScriptedCompletions("still not json", "nope, not json either")
    synthesizer = make_synthesizer(settings, local)

    memo, meta = synthesizer.synthesize(memo_context)

    assert meta.synthesized_by_llm is False
    assert isinstance(memo, AgentSynthesizedMemo)
    assert len(local.calls) == 2
    assert "deterministic fallback" in (meta.note or "")


def test_correction_prompt_template_matches_spec() -> None:
    assert CORRECTION_PROMPT_TEMPLATE.startswith(
        "Convert the following financial analysis into strictly valid JSON matching this schema: "
    )


# ---------------------------------------------------------------------------
# 4) Graceful fallback routing to Gemini
# ---------------------------------------------------------------------------


def test_gemini_fallback_when_local_ollama_times_out(
    settings: Settings,
    memo_context: MemoContext,
) -> None:
    local = ScriptedCompletions(error=TimeoutError("local Ollama request timed out after 30s"))
    gemini = ScriptedCompletions(valid_memo_json())
    synthesizer = make_synthesizer(settings, local, gemini=gemini)

    memo, meta = synthesizer.synthesize(memo_context)

    assert meta.synthesized_by_llm is True  # Gemini produced the memo
    assert meta.model == "gemini-fake"
    assert memo.title == "AAPL/MSFT Risk Memo"
    assert len(local.calls) == 1
    assert len(gemini.calls) == 1
    note = meta.note or ""
    assert "timed out" in note
    assert "Gemini fallback" in note


def test_gemini_skipped_when_local_preflight_fails(
    settings: Settings,
    memo_context: MemoContext,
) -> None:
    local = ScriptedCompletions(valid_memo_json())  # must NEVER be called
    gemini = ScriptedCompletions(valid_memo_json())
    synthesizer = make_synthesizer(settings, local, gemini=gemini, preflight=False)

    _, meta = synthesizer.synthesize(memo_context)

    assert meta.synthesized_by_llm is True
    assert meta.model == "gemini-fake"
    assert local.calls == []  # pre-flight gate prevented the local call entirely
    assert len(gemini.calls) == 1
    assert "pre-flight" in (meta.note or "")


def test_deterministic_fallback_when_gemini_also_fails(
    settings: Settings,
    memo_context: MemoContext,
) -> None:
    local = ScriptedCompletions(error=TimeoutError("local Ollama request timed out after 30s"))
    gemini = ScriptedCompletions(error=RuntimeError("gemini quota exhausted"))
    synthesizer = make_synthesizer(settings, local, gemini=gemini)

    memo, meta = synthesizer.synthesize(memo_context)

    assert meta.synthesized_by_llm is False  # never an HTTP 500 — a memo always comes back
    assert isinstance(memo, AgentSynthesizedMemo)
    assert "deterministic fallback" in (meta.note or "")
    assert "quota exhausted" in (meta.note or "")


def test_deterministic_fallback_when_gemini_not_configured(
    settings: Settings,
    memo_context: MemoContext,
) -> None:
    local = ScriptedCompletions("broken output", "still broken")
    synthesizer = make_synthesizer(settings, local)  # _gemini_client stays None

    memo, meta = synthesizer.synthesize(memo_context)

    assert meta.synthesized_by_llm is False
    assert isinstance(memo, AgentSynthesizedMemo)
    assert "GEMINI_API_KEY" in (meta.note or "")
