# QuantBrief

A **portfolio risk analytics platform** combining a deterministic quantitative engine (NumPy/Pandas)
with an LLM-powered risk advisor — built as a full-stack Python platform for production-grade portfolio risk auditing.

> **Core constraint:** the LLM never calculates anything. Every number — returns, volatility,
> Sharpe ratio, VaR, beta, shock returns — is computed deterministically. The LLM only receives
> those pre-computed metrics to write an executive risk memo or answer conversational questions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Streamlit Dashboard  (port 8501)                                   │
│  frontend/app.py                                                    │
│                                                                     │
│  ┌──────────────┐  ┌───────────────────────┐  ┌─────────────────┐  │
│  │ Sidebar      │  │ Metric Cards + Charts │  │ Streaming Chat  │  │
│  │ CSV Popover  │  │ Pie · Line · Heatmap  │  │ st.chat_input   │  │
│  └──────┬───────┘  └───────────────────────┘  └────────┬────────┘  │
│         │  POST /analyze (fast)                         │  POST /chat/stream
└─────────┼─────────────────────────────────────────────-┼───────────┘
          │                                               │
          ▼                                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FastAPI Backend  (port 8000)                                       │
│  quantbrief/main.py                                                 │
│                                                                     │
│  POST /analyze ──────────────────────────────────────────────────▶ │
│    market_data.py  →  calculator.py  →  state.py (session cache)   │
│    Returns: metrics + charts + session_id  (~10–20 s, no LLM)      │
│                                                                     │
│  POST /session/{id}/memo/stream ──────────────────────────────────▶ │
│    agent.py (AgentSynthesizer.stream_synthesis)                     │
│    True token-by-token LLM streaming + live Markdown JSON parsing   │
│    Pipeline: Ollama local → Gemini fallback → deterministic memo   │
│                                                                     │
│  POST /chat/stream ───────────────────────────────────────────────▶ │
│    chat.py  →  LLM grounded strictly in cached MemoContext         │
│    Streams plain-prose response tokens via create(stream=True)     │
│                                                                     │
│  GET  /session/{id}/ping   (staleness check, <1 ms)               │
│  POST /session/{id}/save   (Postgres persistence flag)             │
└─────────────────────────────────────────────────────────────────────┘
```

### Two-Phase UX Design

| Phase | Endpoint | Latency | UX Behavior |
|-------|----------|---------|-------------|
| **1 — Fast Metrics** | `POST /analyze` | ~5–15 s | Fetches market data via `yfinance`, calculates quantitative KPIs, and displays pie, line, and heatmap charts immediately. |
| **2 — True Streaming Memo** | `POST /session/{id}/memo/stream` | Real-time tokens | True LLM token streaming. Tokens stream into a live client-side parser that renders clean, formatted Markdown bullets as they arrive (no raw JSON exposed). |
| **Grounded Advisor Chat** | `POST /chat/stream` | Instant first token | Conversational advisor strictly grounded in the session's computed metrics. Displays an immediate thinking indicator while connecting. |

---

## Key Features

1. **Deterministic Quant Engine:**
   - Calculations for Geometric Annualized Return, Volatility, Sharpe Ratio (rf = 5.25%), Market Beta, and Empirical 95% Daily VaR.
   - 3 Historical crisis shock replays: COVID-2020 Crash, 2022 Fed Rate Shock, 2008 Global Financial Crisis.

2. **True LLM Token Streaming:**
   - Real-time token streaming (`create(stream=True)`) directly from Ollama or Gemini.
   - Frontend `_format_partial_memo_json` intercepts `<think>` blocks and live-parses partial JSON into formatted Markdown.

3. **CSV Portfolio Import via Popover:**
   - Dedicated `📂 Import Portfolio CSV` popover button in the sidebar.
   - Upload any CSV containing `ticker` (or `symbol`) and `weight` columns to auto-fill asset allocations.

4. **Financial Consistency & Anomaly Guardrails:**
   - **Upside Verification:** Prohibits declaring "underperformance" or "lagged returns" when portfolio return exceeds the benchmark.
   - **Anomaly Checks:** Alerts if Sharpe > 2.0 or low beta combined with extreme outperformance indicates data skew or corporate action anomalies.
   - **Caveat Dependencies:** Enforces "Further Analysis Required" recommendations if critical price gaps or missing parameters exist.

5. **Fault-Tolerant Charting & Resilient UI:**
   - Correlation heatmap safely handles `NaN`/`None` values resulting from non-overlapping market schedules or illiquid tickers without throwing format errors.
   - In-memory session ping detects backend restarts and prompts clean re-runs.

---

## File Structure

```
tryingFreebuff/
├── pyproject.toml
├── README.md
├── .env                     ← local config (git-ignored)
├── .env.example             ← template (QUANTBRIEF_ prefixes)
├── .gitignore
│
├── quantbrief/              ← FastAPI backend package
│   ├── __init__.py          # package metadata / version
│   ├── main.py              # FastAPI app: /analyze, /memo/stream, /chat/stream, /ping
│   ├── config.py            # pydantic-settings + LLM provider resolution
│   ├── schemas.py           # strict Pydantic v2 models (MetricsOnlyResponse, ChatRequest…)
│   ├── state.py             # InMemorySessionRepository (repo pattern, Postgres-swappable)
│   ├── chat.py              # streaming chat engine (grounded in MemoContext)
│   ├── agent.py             # AgentSynthesizer: stream_synthesis + resilient LLM memo pipeline
│   ├── market_data.py       # yfinance ingestion, graceful failure, tz-normalisation
│   ├── calculator.py        # deterministic quant engine (the only place numbers are made)
│   ├── scenarios.py         # historical shock windows (COVID-2020, rates-2022, GFC-2008)
│   ├── _dashboard_runner.py # `quantbrief-dashboard` entry-point shim
│   └── llm/
│       ├── model_checker.py # Ollama pre-flight: GET {host}/api/tags
│       └── sanitizer.py     # <think> stripping + regex JSON extraction → strict schema
│
├── frontend/
│   └── app.py               ← Streamlit dashboard with CSV popover & live streaming parser
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

# 2. Copy configuration and configure Gemini key (recommended for cloud fallback)
cp .env.example .env
# Edit .env: set GEMINI_API_KEY, QUANTBRIEF_LLM_MODEL, etc.

# 3. Pull a local model via Ollama (optional, default is local)
ollama pull granite4.2:3b     # IBM Granite — fast, structured JSON (~2 GB)
ollama pull llama3.1:8b       # Meta Llama 3.1 — higher quality (~5 GB)

# 4a. Start the FastAPI backend
uvicorn quantbrief.main:app --host 0.0.0.0 --port 8000

# 4b. Start the Streamlit dashboard (in a second terminal)
streamlit run frontend/app.py
#   or using the entry point: quantbrief-dashboard
```

Open **http://localhost:8501** in your browser.

---

## LLM Configuration

All settings are configured via `.env` (prefix `QUANTBRIEF_`):

| Provider | Base URL | Model | Key |
|----------|----------|-------|-----|
| `ollama` (local, default) | `http://localhost:11434/v1` | `granite4.2:3b` | — |
| `gemini` (cloud fallback) | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.0-flash` | `GEMINI_API_KEY` |
| `openai` (alternative) | `https://api.openai.com/v1` | `gpt-4o-mini` | `OPENAI_API_KEY` |

**Resilient 6-stage Memo Synthesis Pipeline:**

1. **Pre-flight** — checks Ollama `/api/tags` in milliseconds; skips local if model is missing or offline.
2. **Local generation** — Ollama at `QUANTBRIEF_LLM_TIMEOUT_SECONDS`.
3. **JSON extraction** — strips `<think>…</think>` reasoning blocks, isolates outermost JSON object.
4. **Self-correction** — malformed reply → re-prompts same model at zero temperature with a fix prompt.
5. **Gemini fallback** — timeout or failure seamlessly falls back to cloud endpoint.
6. **Deterministic fallback** — if all LLM options fail, an executive rule-based memo is synthesized from metrics.

`memo_meta.synthesized_by_llm` and `memo_meta.model` always transparently report how the memo was generated.

---

## Quantitative Engine

All math is in [`calculator.py`](quantbrief/calculator.py) — deterministic, zero LLM involvement.

| Metric | Formula / Definition |
|--------|----------------------|
| Annualized return | Geometric: `(P_end/P_start)^(252/n) − 1` |
| Annualized volatility | `std(daily_returns) × √252` |
| Sharpe ratio | `(r_p − rf) / σ_p` — default risk-free rate rf = **5.25%** |
| Beta | `cov(portfolio, benchmark) / var(benchmark)` |
| Historical 95% VaR | Empirical 5th-percentile of daily returns (expressed as positive loss) |
| Shock scenarios | Compounded returns during COVID-2020, Rate Shock 2022, and GFC-2008 |

Chart payloads (cumulative return time-series and Pearson correlation matrix) are computed deterministically and returned by `/analyze`.

> **Currency Note:** All calculations operate on daily percentage returns (dimensionless). Portfolios with mixed currencies (e.g. US and Indian equities) compute accurate relative returns without erroneous cross-currency price comparisons.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service liveness check |
| `POST` | `/analyze` | Fast metrics + charts + `session_id` (no LLM latency) |
| `POST` | `/session/{id}/memo/stream` | Stream LLM risk memo (token-by-token NDJSON) |
| `POST` | `/chat/stream` | Stream grounded conversational advisor answers (plain text chunks) |
| `GET` | `/session/{id}/ping` | Session liveness check (<1 ms) |
| `POST` | `/session/{id}/save` | Flag session for persistence |

Interactive Swagger documentation is available at: **http://localhost:8000/docs**

---

## Validation & Error Handling

`POST /analyze` returns `422 Unprocessable Entity` for:
- Ticker/weight list length mismatch.
- Weights not summing to `1.0` (tolerance `1e-6`).
- Negative weights (short positions prohibited).
- Duplicate or empty tickers.
- `lookback_days` outside `[30, 3650]`.
- Unknown request fields (`extra="forbid"`).

Unknown/delisted tickers return `400 Bad Request`; upstream data provider failures return `503 Service Unavailable`.

---

## Testing & Quality Assurance

```bash
# Run unit test suite (70 hermetic tests — no network, mocked LLMs)
pytest

# Static type checking
mypy quantbrief

# Code style and linting
ruff check quantbrief tests
```