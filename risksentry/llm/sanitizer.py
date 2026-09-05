"""Deterministic extraction of a strict JSON memo from LLM output.

Thinking-enabled local models (deepseek-r1, qwen3, ...) emit a reasoning block
enclosed in ``<think>...</think>`` tags before their answer. This module strips
that noise with regex, locates the outermost JSON object, parses it, and
validates it against the strict ``AgentSynthesizedMemo`` Pydantic schema. Any
failure raises :class:`MalformedJSONError` so the orchestrator can self-correct
or fall back — never a silent, schema-less result.
"""

from __future__ import annotations

import json
import re

from risksentry.schemas import AgentSynthesizedMemo

#: ``<think>...</think>`` reasoning blocks, non-greedy so multiple blocks and
#: nested angle brackets survive; ``re.DOTALL`` lets ``.*?`` cross newlines.
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

#: Optional markdown code fences the model may wrap the JSON in.
CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

#: Greedy outer-object candidate (fallback when the balanced scan is not needed).
GREEDY_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class MalformedJSONError(ValueError):
    """Raised when model output cannot be parsed into a valid risk memo.

    Subclasses :class:`ValueError` so callers that used to catch a generic
    parsing failure keep working; ``str(exc)`` is safe for log lines and notes.
    """


def _candidate_parses(candidate: str) -> dict[str, object] | None:
    """Try to ``json.loads`` + schema-validate a candidate; ``None`` on any failure."""
    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        AgentSynthesizedMemo.model_validate(data)
    except Exception:  # noqa: BLE001 - schema violations are MalformedJSONError territory
        return None
    return data


def extract_json_from_thinking(raw_text: str) -> dict[str, object]:
    """Extract and validate the JSON memo out of a possibly thinking output.

    Steps:
    a. Strip ``<think>...</think>`` blocks (``re.DOTALL``).
    b. Locate the outermost JSON object: first try a greedy ``\\{.*\\}`` match,
       then fall back to a balanced outer-object scan over every ``{``/``}`` pair.
    c. ``json.loads`` the extracted text.
    d. Validate against ``AgentSynthesizedMemo`` (Pydantic v2, ``extra="forbid"``).
    e. Raise :class:`MalformedJSONError` when any step fails.
    """
    if not raw_text or not raw_text.strip():
        raise MalformedJSONError("model returned an empty response")

    text = raw_text.strip()

    # Pull JSON out of markdown fences first (if any), otherwise keep the text.
    fence = CODE_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

    # a) Drop reasoning tokens. Loop to catch multiple/adjacent think blocks.
    cleaned = text
    while True:
        stripped = THINK_BLOCK_RE.sub("", cleaned)
        if stripped == cleaned:
            break
        cleaned = stripped
    cleaned = cleaned.strip()

    # b/c/d) Greedy attempt first — the common case (one JSON object, trailing noise).
    greedy = GREEDY_OBJECT_RE.search(cleaned)
    if greedy is not None:
        parsed = _candidate_parses(greedy.group(0))
        if parsed is not None:
            return parsed

    # Balanced outer-object scan: try each '{' with every '}' after it, from the
    # outermost pairing inward, until one candidate parses AND validates.
    brace_starts = [i for i, ch in enumerate(cleaned) if ch == "{"]
    brace_ends = [i for i, ch in enumerate(cleaned) if ch == "}"]
    for start in brace_starts:
        for end in reversed(brace_ends):
            if end <= start:
                break
            parsed = _candidate_parses(cleaned[start : end + 1])
            if parsed is not None:
                return parsed

    preview = (raw_text.strip()[:120] + "…") if len(raw_text.strip()) > 120 else raw_text.strip()
    raise MalformedJSONError(f"no valid AgentSynthesizedMemo JSON object found in model output: {preview!r}")
