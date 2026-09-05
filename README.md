# RiskSentry

A production-grade FastAPI service for **deterministic portfolio risk calculations** (NumPy/Pandas)
with **resilient LLM-based risk memo synthesis**.

> **Core constraint:** the LLM never calculates anything. Every number in the response — returns,
> volatility, Sharpe ratio, VaR, beta, shock returns — is computed by a dedicated, deterministic
> calculator module. The LLM only receives those pre-computed metrics and restates them in a memo.

## Architecture

```
Client ──▶ POST /analyze
              │
              ▼
      ┌─ market_data.py ─┐        yfinance download (graceful: missing tickers / network)
      │  fetch_market_data()      → aligned, tz-naive adjusted-close DataFrame
      └───────────────────┘
              │
              ▼
      ┌─ calculator.py ──┐        DETERMINISTIC MATH (the only place numbers are produced)
      │  compute_metrics()│       annualized return/vol (252d), Sharpe (default rf=5.25%),
      └───────────────────┘       beta vs benchmark (^NSEI or SPY), historical 95% daily VaR,
              │                   asset-level breakdown, shock scenario replays
              ▼
      ┌─ schemas.py ─────┐        MemoContext: metrics ONLY — raw prices never leave the engine
      │  MemoContext      │
      └───────────────────┘
              │
              ▼
      ┌─ agent.py ───────┐        Resilient LLM memo synthesis, local-first:
      │  AgentSynthesizer│        ① pre-flight model check (Ollama /api/tags)
      └───────────────────┘        ② local generation (Ollama @ 120s timeout)
              │                    ③ regex extraction (<think> stripped) → strict schema
              │                    ④ zero-temp self-correction on the same model
              │                    ⑤ Gemini cloud fallback   ⑥ deterministic memo
              ▼                      (never an HTTP 500)
        AnalysisResponse
```

## File structure

```
risksentry/
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
├── risksentry/
│   ├── __init__.py          # package metadata / version
│   ├── main.py              # FastAPI app: GET /health, POST /analyze
│   ├── config.py            # pydantic-settings config + local/Gemini resolution
│   ├── schemas.py           # strict Pydantic v2 request/response/memo models
│   ├── market_data.py       # yfinance ingestion with graceful failure handling
│   ├── calculator.py        # deterministic quantitative engine (no LLM, no I/O)
│   ├── scenarios.py         # historical shock windows (2020 COVID, 2022 rates, 2008 GFC)
│   ├── llm/
│   │   ├── model_checker.py # async pre-flight check: GET {host}/api/tags
│   │   └── sanitizer.py     # <think> stripping + regex JSON extraction → strict schema
│   └── agent.py             # LLM pipeline orchestrator + deterministic fallback
└── tests/
    ├── test_risk_engine.py       # core engine/API coverage (hermetic — no network)
    └── test_synthesis_pipeline.py # pre-flight, sanitizer, self-correction, Gemini fallback
```

## Quickstart

```bash
# 1. Install (Python >= 3.11)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Run the API
uvicorn risksentry.main:app --reload        # or: risksentry

# 3. Health check
curl http://localhost:8000/health

# 4. Analyze a portfolio (US Equities example)
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT", "NVDA"], "weights": [0.4, 0.3, 0.3], "lookback_days": 365, "benchmark": "SPY"}'

# 5. Analyze a portfolio (Indian Equities example)
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["RELIANCE.NS", "TCS.NS", "INFY.NS"], "weights": [0.4, 0.3, 0.3], "lookback_days": 365, "benchmark": "^NSEI"}'
```

Interactive docs: <http://localhost:8000/docs>

## LLM Setup

The synthesizer uses the `openai` SDK pointed at OpenAI-compatible endpoints. Defaults:

| Endpoint                | Base URL                                                   | Recommended Model   | Key              |
|-------------------------|------------------------------------------------------------|---------------------|------------------|
| `ollama` (local, first) | `http://localhost:11434/v1`                                | `granite4.2:3b` / `llama3.1:8b` | ignored          |
| `gemini` (cloud fallback) | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-3.6-flash`  | `GEMINI_API_KEY` |
| `openai` (alternative)  | `https://api.openai.com/v1`                                | `gpt-4o-mini`       | `OPENAI_API_KEY` |

```bash
# Recommended local models (fast, structured JSON, no heavy thinking-token bloat):
ollama pull granite4.2:3b    # IBM Granite (enterprise JSON tuned, ~2.0 GB)
# or:
ollama pull llama3.1:8b      # Meta Llama 3.1 (high formatting quality, ~4.9 GB)

# Run the API
uvicorn risksentry.main:app
```

**Inference pipeline (thinking-enabled models supported):**

1. **Pre-flight** — the async checker queries Ollama's native `GET {host}/api/tags`; a missing
   model or unreachable daemon is detected in ~ms and never blocks the request.
2. **Local generation** on Ollama, capped at `OLLAMA_TIMEOUT_SECONDS` (default: 120s).
3. **Deterministic extraction** — `sanitizer.py` strips `<think>...</think>` reasoning blocks with
   regex (`re.DOTALL`), finds the outermost JSON object, parses it, and validates it against the
   strict `AgentSynthesizedMemo` schema (Pydantic v2, `extra="forbid"`).
4. **Self-correction** — on a malformed reply the raw output is sent back to the *same local model*
   at zero temperature with a "convert to strict JSON" prompt, then re-extracted.
5. **Gemini fallback** — timeouts, connection failures, or failed self-corrections route to the
   OpenAI-compatible Gemini endpoint (`GEMINI_BASE_URL`/`GEMINI_API_KEY`/`GEMINI_MODEL`).
6. **Deterministic fallback** — if Gemini is also unavailable/unconfigured, a rule-based memo is
   built from the same authoritative metrics. **LLM failures never produce an HTTP 500**;
   `memo_meta.synthesized_by_llm` and `memo_meta.note` report exactly how the memo was produced.

## Validation (Pydantic v2, strict)

`POST /analyze` rejects, with a `422` and a precise message:

- ticker/weight lists of different lengths
- weights that do not sum to `1.0` (tolerance `1e-6`)
- negative weights (short positions)
- empty or duplicate tickers, empty ticker strings
- `lookback_days` outside `[30, 3650]`
- unknown request fields (`extra="forbid"`)

Unknown/delisted tickers → `400` with the offending symbols; network failures → `503`.

## Quantitative Engine

- **Annualized return** — geometric: `(P_end / P_start)^(252/n) - 1`
- **Annualized volatility** — sample std of daily returns × `√252`
- **Sharpe ratio** — `(r_p − rf) / σ_p`, default rf = **5.25%** (configurable via `RISKSENTRY_RISK_FREE_RATE`)
- **Beta** — `cov(portfolio, benchmark) / var(benchmark)` (default benchmark `^NSEI`, configurable via `RISKSENTRY_DEFAULT_BENCHMARK`)
- **Historical 95% daily VaR** — empirical 5th-percentile of daily returns, reported as a positive loss
- **Asset-level breakdown** — per-ticker weight, annualized return, volatility, beta
- **Shock scenarios** — replays of `covid_2020` (Feb 19 – Mar 23, 2020), `rate_shock_2022`
  (Jan 3 – Oct 12, 2022), and `gfc_2008` (Sep 1 – Nov 20, 2008), with portfolio and benchmark
  returns plus data-coverage percentages

## Tests

```bash
pytest                      # hermetic: mocks yfinance + the LLM, no network required
mypy risksentry             # static type checking
ruff check risksentry tests # lint
```

`tests/test_synthesis_pipeline.py` covers the inference pipeline: pre-flight checks against mocked
`/api/tags` responses (active/missing/non-200/unreachable), regex extraction from `<think>`-wrapped
output, the zero-temperature self-correction loop, and graceful routing to Gemini/deterministic
fallbacks.