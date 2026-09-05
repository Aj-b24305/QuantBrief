from __future__ import annotations

import asyncio
import json
import logging
from time import perf_counter
from typing import Literal

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from risksentry.config import Settings, get_settings
from risksentry.llm.model_checker import is_model_available as _async_is_model_available
from risksentry.llm.sanitizer import MalformedJSONError, extract_json_from_thinking
from risksentry.schemas import AgentSynthesizedMemo, MemoContext, MemoMeta

# Semantic alias used by the orchestration pipeline: the pre-computed,
# authoritative quantitative report the LLM is allowed to read.
QuantRiskReport = MemoContext

logger = logging.getLogger("risksentry.agent")

#: JSON shape the model must reproduce. Used both in the system prompt and in
#: the self-correction prompt so the repair pass knows the target contract.
SCHEMA_SHAPE = (
    '{"title": string, "summary": string, "key_risks": [string, ...], '
    '"recommendations": [string, ...], "stress_test_outlook": string, '
    '"confidence": "low" | "medium" | "high", "caveats": [string, ...]}'
)

SYSTEM_PROMPT = (
    "You are RiskSentry's risk-memo synthesizer. You translate pre-computed quantitative risk "
    "metrics into a concise, professional risk memo for portfolio managers.\n\n"
    "HARD RULES:\n"
    "1. You NEVER calculate or recalculate any number. No return, volatility, Sharpe, beta, VaR, "
    "or scenario math. All figures are provided pre-computed and authoritative; quote them as given.\n"
    "2. Formatting: you may present each figure as a percentage or a 2-decimal float, whichever "
    "reads most naturally (e.g. write 28.18% or 0.28 instead of 0.28178354610878165). Never output "
    "raw long decimals, and keep the meaning unchanged — do not round away magnitude or sign.\n"
    "3. When comparing returns, sign and magnitude matter: a more negative return represents a "
    "larger loss. For example, -38% is worse (more loss, underperformance) than -24% — never "
    "describe -38% as outperforming -24%.\n"
    "4. If a metric is null, do not invent a value; describe the gap qualitatively.\n"
    "5. Only discuss shock scenarios that appear in the provided data.\n"
    "6. Your entire reply must be a single JSON object matching this exact schema:\n"
    f"{SCHEMA_SHAPE}\n"
    "7. No markdown, no prose, no commentary outside the JSON object. If you perform any internal "
    "reasoning, emit it only inside a single <think>...</think> block immediately before the JSON; "
    "the JSON object itself must still be complete and valid."
)

CORRECTION_PROMPT_TEMPLATE = (
    "Convert the following financial analysis into strictly valid JSON matching this schema: "
    "{schema}. Do not output thoughts or markdown, only the raw JSON string: {raw_text}"
)


def _build_user_prompt(context: MemoContext) -> str:
    """Serialize the pre-computed metrics for the LLM (never raw price series)."""
    payload = json.dumps(context.model_dump(mode="json"), indent=2)
    return (
        "Below is the pre-computed, authoritative risk analysis produced by the RiskSentry "
        "quantitative engine. Do NOT recompute anything — use these numbers verbatim in your memo.\n\n"
        f"{payload}\n\n"
        "Synthesize the risk memo now. Reply with a single JSON object only."
    )


def parse_memo(content: str) -> AgentSynthesizedMemo:
    """Parse and strictly validate the LLM's JSON into ``AgentSynthesizedMemo``.

    Thin wrapper over :func:`risksentry.llm.sanitizer.extract_json_from_thinking`
    (which already strips ``<think>`` blocks, code fences and trailing noise).
    Raises :class:`MalformedJSONError` when the output does not conform.
    """
    data = extract_json_from_thinking(content)
    return AgentSynthesizedMemo.model_validate(data)


def build_fallback_memo(context: MemoContext) -> AgentSynthesizedMemo:
    """Deterministic, rule-based memo used when every LLM path is unavailable.

    Built exclusively from pre-computed metrics — identical inputs always yield
    identical output, and no math is performed here.
    """
    portfolio = context.portfolio

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2%}"

    sharpe_txt = "n/a" if portfolio.sharpe_ratio is None else f"{portfolio.sharpe_ratio:.2f}"
    beta_txt = "n/a" if portfolio.beta is None else f"{portfolio.beta:.2f}"

    key_risks = [
        f"Daily loss potential: historical 95% VaR of {pct(portfolio.var_95_daily)} over the lookback window.",
        f"Market sensitivity: portfolio beta of {beta_txt} versus {context.benchmark}.",
        f"Return efficiency: Sharpe ratio of {sharpe_txt} at a {pct(context.risk_free_rate)} risk-free rate.",
    ]
    worst_shock = min(
        (s.portfolio_return for s in context.shock_scenarios if s.portfolio_return is not None),
        default=None,
    )
    if worst_shock is not None:
        key_risks.append(
            f"Historical stress: the worst replay among the included shock scenarios is {pct(worst_shock)}."
        )

    shock_outlook = "; ".join(
        f"{s.name}: portfolio {pct(s.portfolio_return)}, benchmark {pct(s.benchmark_return)}"
        for s in context.shock_scenarios
    )

    if portfolio.sharpe_ratio is not None and portfolio.sharpe_ratio >= 1.0:
        confidence: Literal["low", "medium", "high"] = "high"
    elif portfolio.sharpe_ratio is not None and portfolio.sharpe_ratio >= 0.0:
        confidence = "medium"
    else:
        confidence = "low"

    summary = (
        f"Over the {context.lookback_days}-day window the portfolio returned "
        f"{pct(portfolio.annualized_return)} annualized with {pct(portfolio.annualized_volatility)} "
        f"volatility. Its 95% daily VaR is {pct(portfolio.var_95_daily)} and its beta to "
        f"{context.benchmark} is {beta_txt}."
    )

    return AgentSynthesizedMemo(
        title=f"Portfolio Risk Memo — {', '.join(context.tickers)}",
        summary=summary,
        key_risks=key_risks,
        recommendations=[
            "Review positions with the largest contribution to portfolio volatility.",
            "Reassess allocations if the portfolio beta is materially above the benchmark.",
            "Stress-test the portfolio against the shock scenarios with the largest losses.",
        ],
        stress_test_outlook=shock_outlook or "No scenario data available for this portfolio.",
        confidence=confidence,
        caveats=[
            (
                "This memo was generated deterministically by the RiskSentry engine because the "
                "LLM synthesizer was unavailable or returned invalid output. All figures remain "
                "authoritative."
            ),
        ],
    )


def _default_preflight(model_name: str, base_url: str) -> bool:
    """Sync bridge to the async :func:`is_model_available` probe.

    Intended to run from FastAPI's sync (threadpool) request handlers. If it is
    ever invoked from inside a running event loop, the probe cannot be awaited —
    returning ``True`` is the safe choice because the generation attempt itself
    surfaces any real unavailability, which then routes to the fallbacks.
    """
    try:
        return asyncio.run(_async_is_model_available(model_name, base_url))
    except RuntimeError:
        return True


class AgentSynthesizer:
    """Resilient LLM memo synthesizer with local-first, cloud-second fallbacks.

    Pipeline (see :meth:`synthesize_risk_memo`):

    1. Pre-flight: :func:`is_model_available` against Ollama's ``/api/tags``.
    2. Local generation on Ollama with a 30s request timeout.
    3. Deterministic regex extraction + Pydantic validation of the memo JSON.
    4. Zero-temperature self-correction on the *same* local model when the
       first output is malformed.
    5. Cloud fallback to Google Gemini (OpenAI-compatible endpoint).
    6. Deterministic fallback memo — LLM failures never surface as HTTP 500s.

    The LLM only ever sees pre-computed metrics (see ``MemoContext``); it never
    receives price history, so it structurally cannot calculate anything.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        # --- Local (Ollama / OpenAI-compatible) endpoint -------------------
        base_url, api_key, model = settings.resolve_llm_config()
        self.base_url = base_url
        self.model = model
        # Hard 30s cap on local Ollama HTTP requests (OLLAMA_TIMEOUT_SECONDS).
        self.local_timeout = settings.ollama_timeout_seconds
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or "not-set",
            timeout=self.local_timeout,
            max_retries=settings.llm_max_retries,
        )

        # --- Gemini cloud fallback (None when unconfigured) ----------------
        gemini_base_url, gemini_api_key, gemini_model = settings.resolve_gemini_config()
        self.gemini_base_url = gemini_base_url
        self.gemini_model = gemini_model
        self._gemini_client: OpenAI | None = None
        if gemini_api_key:
            self._gemini_client = OpenAI(
                base_url=gemini_base_url,
                api_key=gemini_api_key,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
            )

        # Test seam: default hits Ollama's /api/tags; tests override it.
        self._preflight = _default_preflight

        # Outcome of the most recent pipeline run, consumed by synthesize().
        self._last_outcome: dict[str, str | None] = {}

    # --- Public API ---------------------------------------------------------

    def synthesize(self, context: MemoContext) -> tuple[AgentSynthesizedMemo, MemoMeta]:
        """Run the full pipeline and report the memo plus provenance metadata."""
        started = perf_counter()
        memo = self.synthesize_risk_memo(context)
        elapsed_ms = int((perf_counter() - started) * 1000)
        outcome = self._last_outcome
        source = outcome.get("source") or "deterministic"
        return memo, MemoMeta(
            synthesized_by_llm=source in ("local", "gemini"),
            model=outcome.get("model"),
            elapsed_ms=elapsed_ms,
            note=outcome.get("note"),
        )

    # --- Orchestration (spec steps 1-6) -------------------------------------

    def synthesize_risk_memo(self, quant_report: QuantRiskReport) -> AgentSynthesizedMemo:
        """Synthesize a memo, never raising on LLM failures.

        Returns the validated ``AgentSynthesizedMemo`` from whichever stage of
        the local → self-correction → Gemini → deterministic ladder succeeds.
        """
        memo, outcome = self._run_pipeline(quant_report)
        self._last_outcome = outcome
        return memo

    def _run_pipeline(self, context: MemoContext) -> tuple[AgentSynthesizedMemo, dict[str, str | None]]:
        local_error: str | None = None

        # Step 1 — pre-flight: is the local model actually pullable?
        try:
            available = self._preflight(self.model, self.base_url)
        except Exception as exc:  # noqa: BLE001 - a broken probe must not crash the app
            available = False
            local_error = f"pre-flight probe raised ({type(exc).__name__}: {exc})"
        if not available:
            local_error = local_error or (
                f"local model {self.model} unavailable (pre-flight check against "
                f"{self.base_url}/api/tags failed)"
            )
            logger.warning("RiskSentry LLM: %s — jumping to Gemini fallback.", local_error)
        else:
            try:
                # Step 2 — local generation with the enforced 30s timeout.
                user_prompt = _build_user_prompt(context)
                raw_output = self._complete(
                    self._client,
                    self.model,
                    user_prompt,
                    temperature=self.settings.llm_temperature,
                    json_object=True,
                    timeout=self.local_timeout,
                )
            except Exception as exc:  # noqa: BLE001 - timeout/connection/refusal
                local_error = f"local generation failed ({type(exc).__name__}: {exc})"
                logger.warning("RiskSentry LLM: %s — jumping to Gemini fallback.", local_error)
            else:
                # Step 3 — deterministic extraction + strict schema validation.
                try:
                    return self._validated_memo(
                        extract_json_from_thinking(raw_output),
                        source="local",
                        model=self.model,
                        note=None,
                    )
                except MalformedJSONError:
                    # Step 4 — self-correction: same local model, zero temperature.
                    try:
                        correction_prompt = CORRECTION_PROMPT_TEMPLATE.format(
                            schema=SCHEMA_SHAPE,
                            raw_text=raw_output[:4000],
                        )
                        repaired = self._complete(
                            self._client,
                            self.model,
                            correction_prompt,
                            temperature=0.0,
                            json_object=True,
                            timeout=self.local_timeout,
                        )
                        return self._validated_memo(
                            extract_json_from_thinking(repaired),
                            source="local",
                            model=self.model,
                            note="local output was malformed; memo repaired via one zero-temperature self-correction pass",
                        )
                    except MalformedJSONError as exc:
                        local_error = f"self-correction failed ({type(exc).__name__}: {str(exc)[:200]})"
                        logger.warning("RiskSentry LLM: %s — jumping to Gemini fallback.", local_error)
                    except Exception as exc:  # noqa: BLE001 - repair request itself failed
                        local_error = f"self-correction request failed ({type(exc).__name__}: {exc})"
                        logger.warning("RiskSentry LLM: %s — jumping to Gemini fallback.", local_error)

        # Step 5 — Gemini cloud fallback (OpenAI-compatible endpoint).
        gemini_error: str | None = None
        if self._gemini_client is not None:
            try:
                user_prompt = _build_user_prompt(context)
                raw_output = self._complete(
                    self._gemini_client,
                    self.gemini_model,
                    user_prompt,
                    temperature=0.0,
                    json_object=False,
                    timeout=self.settings.llm_timeout_seconds,
                )
                note = (
                    f"Local Ollama synthesis unavailable ({local_error or 'unknown'}); "
                    f"memo produced by Gemini fallback ({self.gemini_model})."
                )
                return self._validated_memo(
                    extract_json_from_thinking(raw_output),
                    source="gemini",
                    model=self.gemini_model,
                    note=note,
                )
            except Exception as exc:  # noqa: BLE001 - any Gemini failure ends at the memo
                gemini_error = f"{type(exc).__name__}: {exc}"
        else:
            gemini_error = "Gemini fallback not configured (no GEMINI_API_KEY)"

        # Step 6 — deterministic fallback memo. LLM failures must never 500.
        logger.warning(
            "RiskSentry LLM: local error [%s]; Gemini error [%s]; returning deterministic fallback memo.",
            local_error,
            gemini_error,
        )
        note = (
            f"LLM synthesis failed (local: {local_error}; gemini: {gemini_error}); "
            "deterministic fallback memo returned."
        )
        return build_fallback_memo(context), {"source": "deterministic", "model": self.model, "note": note}

    # --- Helpers ------------------------------------------------------------

    def _validated_memo(
        self,
        data: dict[str, object],
        *,
        source: str,
        model: str,
        note: str | None,
    ) -> tuple[AgentSynthesizedMemo, dict[str, str | None]]:
        return AgentSynthesizedMemo.model_validate(data), {"source": source, "model": model, "note": note}

    def _complete(
        self,
        client: OpenAI,
        model: str,
        prompt: str,
        *,
        temperature: float,
        json_object: bool,
        timeout: float,
    ) -> str:
        """Single chat completion; returns the raw text content."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if json_object:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
                timeout=timeout,
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
        return response.choices[0].message.content or ""


def synthesize_risk_memo(quant_report: QuantRiskReport) -> AgentSynthesizedMemo:
    """Module-level convenience: run the resilient pipeline with app settings.

    Equivalent to ``AgentSynthesizer(get_settings()).synthesize_risk_memo(...)``
    but drops the provenance metadata (use the class when you need ``MemoMeta``).
    """
    return AgentSynthesizer(get_settings()).synthesize_risk_memo(quant_report)
