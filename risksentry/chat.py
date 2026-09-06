from __future__ import annotations

"""Chat engine for RiskSentry's conversational follow-up feature."""

import json
import logging
from collections.abc import Generator
from typing import Any

from openai import OpenAI

from risksentry.state import ChatMessage

logger = logging.getLogger("risksentry.chat")

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CHAT_SYSTEM_PROMPT = """\
You are RiskSentry's portfolio risk advisor. A quantitative risk engine has
already computed all the hard numbers for this portfolio — your job is to
help the portfolio manager understand and act on them through natural
conversation.

ABSOLUTE RULES:
1. You NEVER calculate or recalculate any number. Every figure you discuss
   must come from the quantitative report supplied in this conversation.
2. If asked about a scenario not in the report (e.g. "what if I drop NVDA?"),
   respond qualitatively — describe the direction of change without inventing
   new numbers.
3. Speak in plain, concise English. Short paragraphs work best.
4. If the question is unrelated to the portfolio or financial analysis,
   politely redirect back to the portfolio.
5. The pre-computed quantitative report is your ONLY factual source.
"""


def build_chat_messages(
    context_payload: dict[str, Any],
    history: list[ChatMessage],
    user_message: str,
) -> list[dict[str, str]]:
    """Build the full message list for a chat completion call."""
    context_json = json.dumps(context_payload, indent=2, default=str)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Here is the pre-computed quantitative risk report. "
                "Use it as your ONLY factual source:\n\n"
                f"{context_json}"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Understood. I have the quantitative report. "
                "I will ground all answers in those figures without recalculating anything. "
                "How can I help?"
            ),
        },
    ]
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": user_message})
    return messages


# ---------------------------------------------------------------------------
# Streaming generator — uses standard create(stream=True), compatible with
# both Ollama's OpenAI-compatible endpoint and the real Gemini/OpenAI APIs.
# ---------------------------------------------------------------------------


def stream_chat(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    *,
    timeout: float = 120.0,
) -> Generator[str, None, None]:
    """Yield text token chunks from the LLM via standard streaming.

    Uses ``client.chat.completions.create(stream=True)`` which is compatible
    with Ollama, Gemini (OpenAI-compat endpoint), and OpenAI itself.
    """
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.3,
            stream=True,
            timeout=timeout,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as exc:  # noqa: BLE001
        logger.warning("stream_chat error (%s): %s", type(exc).__name__, exc)
        yield f"\n\n⚠️ Chat error: {type(exc).__name__} — {exc}"
