# RiskSentry

A **portfolio risk analytics platform** combining a deterministic quantitative engine (NumPy/Pandas)
with an LLM-powered risk advisor — built as a full-stack Python project for the resume.

> **Core constraint:** the LLM never calculates anything. Every number — returns, volatility,
> Sharpe ratio, VaR, beta, shock returns — is computed deterministically. The LLM only receives
> those pre-computed metrics to write a memo or answer chat questions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Streamlit Dashboard  (port 8501)                                   │
│  frontend/app.py                                                    │
│                                                                     │
│  ┌──────────────┐  ┌───────────────────────┐  ┌─────────────────┐  │
│  │ Sidebar      │  │ Metric Cards + Charts │  │ Streaming Chat  │  │
│  │ (inputs)     │  │ Pie · Line · Heatmap  │  │ st.chat_input   │  │
│  └──────┬───────┘  └───────────────────────┘  └────────┬────────┘  │
│         │  POST /analyze (fast)                         │  POST /chat/stream
└─────────┼─────────────────────────────────────────────-┼───────────┘
          │                                               │
          ▼                                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FastAPI Backend  (port 8000)                                       │
│  risksentry/main.py                                                 │
│                                                                     │
│  POST /analyze ──────────────────────────────────────────────────▶ │
│    market_data.py  →  calculator.py  →  state.py (session cache)   │
│    Returns: metrics + charts + session_id  (~10–20 s, no LLM)      │
│                                                                     │
│  POST /session/{id}/memo/stream ──────────────────────────────────▶ │
│    agent.py (AgentSynthesizer) → streams memo tokens               │
│    Pipeline: Ollama local → Gemini fallback → deterministic memo   │
│                                                                     │
│  POST /chat/stream ───────────────────────────────────────────────▶ │
│    chat.py  →  LLM grounded in cached MemoContext (no raw prices)  │
│    Streams plain-prose response tokens via create(stream=True)     │
│                                                                     │
│  GET  /session/{id}/ping   (staleness check, <1 ms)               │
│  POST /session/{id}/save   (Postgres placeholder)                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Two-phase UX design

| Phase | Endpoint | Time | Result |
|-------|----------|------|--------|
| **1 — Fast metrics** | `POST /analyze` | ~10–20 s | KPI cards, 3 charts, stress table appear immediately |
| **2 — Streaming memo** | `POST /session/{id}/memo/stream` | 30–60 s | LLM risk memo streams in word-by-word |
| **Chat** | `POST /chat/stream` | depends on model | Conversational answers grounded in cached metrics |

---

## File Structure

```
tryingFreebuff/
├── pyproject.toml
├── README.md
├── .env                     ← local config (git-ignored)
├── .env.example             ← template
├── .gitignore
│
├── risksentry/              ← FastAPI backend package
│   ├── __init__.py          # package metadata / version
│   ├── main.py              # FastAPI app: /analyze, /memo/stream, /chat/stream, /ping
│   ├── config.py            # pydantic-settings + LLM provider resolution
│   ├── schemas.py           # strict Pydantic v2 models (MetricsOnlyResponse, ChatRequest…)
│   ├── state.py             # InMemorySessionRepository (repo pattern, Postgres-swappable)
│   ├── chat.py              # streaming chat engine (grounded in MemoContext)
│   ├── agent.py             # AgentSynthesizer: resilient LLM memo pipeline
│   ├── market_data.py       # yfinance ingestion, graceful failure, tz-normalisation
│   ├── calculator.py        # deterministic quant engine (the only place numbers are made)
│   ├── scenarios.py         # historical shock windows (COVID-2020, rates-2022, GFC-2008)
│   ├── _dashboard_runner.py # `risksentry-dashboard` entry-point shim
│   └── llm/
│       ├── model_checker.py # Ollama pre-flight: GET {host}/api/tags
│       └── sanitizer.py     # <think> stripping + regex JSON extraction → strict schema
│
├── frontend/
│   └── app.py               ← Streamlit dashboard
│
└── tests/
    ├── test_risk_engine.py          # 70 tests — hermetic (no network, mocked LLM)
    └── test_synthesis_pipeline.py   # pre-flight, sanitizer, self-correction, fallbacks
```

---

## Quickstart

```bash
# 1. Install (Python >= 3.11)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Copy config and add your Gemini key (optional but recommended for fallback)
cp .env.example .env
# Edit .env: set GEMINI_API_KEY, RISKSENTRY_LLM_MODEL, etc.

# 3. Pull a local model (pick one)
ollama pull granite4.2:3b     # IBM Granite — fast, structured JSON (~2 GB)
ollama pull llama3.1:8b       # Meta Llama 3.1 — higher quality, slower (~5 GB)

# 4a. Start the backend (stable — no --reload, keeps sessions in memory)
uvicorn risksentry.main:app --host 0.0.0.0 --port 8000

# 4b. Start the dashboard (in a second terminal)
streamlit run frontend/app.py
#   or: risksentry-dashboard
```

Open **http://localhost:8501** in your browser.

---

## LLM Configuration

All settings are in `.env` (prefix `RISKSENTRY_`):

| Provider | Base URL | Model | Key |
|----------|----------|-------|-----|
| `ollama` (local, default) | `http://localhost:11434/v1` | `granite4.2:3b` | — |
| `gemini` (cloud fallback) | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.0-flash` | `GEMINI_API_KEY` |
| `openai` (alternative) | `https://api.openai.com/v1` | `gpt-4o-mini` | `OPENAI_API_KEY` |

**Memo synthesis pipeline** (6-stage, never returns HTTP 500):

1. **Pre-flight** — checks Ollama `/api/tags` in milliseconds; skips local if model missing
2. **Local generation** — Ollama at `OLLAMA_TIMEOUT_SECONDS` (default 120 s)
3. **JSON extraction** — strips `<think>…</think>` reasoning blocks, finds outermost JSON object
4. **Self-correction** — malformed reply → same model at zero temperature with a fix prompt
5. **Gemini fallback** — timeout / failure → cloud endpoint
6. **Deterministic fallback** — if everything fails, a rule-based memo is built from the metrics

`memo_meta.synthesized_by_llm` and `memo_meta.model` always report exactly how the memo was produced.

---

## Quantitative Engine

All math is in [`calculator.py`](risksentry/calculator.py) — deterministic, no LLM involved.

| Metric | Formula |
|--------|---------|
| Annualized return | Geometric: `(P_end/P_start)^(252/n) − 1` |
| Annualized volatility | `std(daily_returns) × √252` |
| Sharpe ratio | `(r_p − rf) / σ_p` — default rf = **5.25%** |
| Beta | `cov(portfolio, benchmark) / var(benchmark)` |
| Historical 95% VaR | Empirical 5th-percentile of daily returns (positive loss) |
| Shock scenarios | COVID-2020 · Rate Shock 2022 · GFC-2008 |

Chart payloads (cumulative returns series, Pearson correlation matrix) are also computed
deterministically and returned by `/analyze` — the frontend does zero math.

> **Currency note:** all metrics use daily percentage returns (dimensionless), so mixed-currency
> portfolios compute correct relative returns. Absolute prices are never compared across tickers.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service liveness |
| `POST` | `/analyze` | Fast metrics + charts + `session_id` (no LLM) |
| `POST` | `/session/{id}/memo/stream` | Stream LLM risk memo (NDJSON tokens) |
| `POST` | `/chat/stream` | Stream conversational answer (plain text chunks) |
| `GET` | `/session/{id}/ping` | Lightweight session liveness check |
| `POST` | `/session/{id}/save` | Flag session for Postgres export (placeholder) |

Interactive docs: **http://localhost:8000/docs**

---

## Validation

`POST /analyze` returns `422` for:

- Ticker/weight list length mismatch
- Weights not summing to `1.0` (tolerance `1e-6`)
- Negative weights (short positions not supported)
- Duplicate or empty tickers
- `lookback_days` outside `[30, 3650]`
- Unknown request fields (`extra="forbid"`)

Unknown/delisted tickers → `400`; network failures → `503`.

---

## Tests

```bash
pytest                       # 70 tests — hermetic, no network, no real LLM
mypy risksentry              # static type checking
ruff check risksentry tests  # lint
```