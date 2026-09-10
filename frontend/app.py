"""QuantBrief · Streamlit Portfolio Dashboard
==============================================
Run: streamlit run frontend/app.py
Requires backend: uvicorn quantbrief.main:app --reload
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="QuantBrief · Portfolio Risk Dashboard",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
for key, default in [
    ("session_id", None),
    ("metrics", None),      # MetricsOnlyResponse payload (from /analyze)
    ("memo", None),         # AgentSynthesizedMemo dict (from /memo/stream)
    ("memo_meta", None),    # MemoMeta dict
    ("chat_history", []),
    ("saved", False),
    ("memo_streaming", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pct(v: float | None, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v * 100:.{decimals}f}%"


def _f2(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.2f}"


def _parse_weights(raw: str) -> list[float] | None:
    try:
        parts = [float(x.strip()) for x in raw.split(",") if x.strip()]
        return parts if parts else None
    except ValueError:
        return None


def _format_partial_memo_json(text: str) -> str:
    """Format raw streaming memo text (JSON / think blocks) into clean Markdown."""
    if "<think>" in text:
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        else:
            think_content = text.split("<think>", 1)[1].strip()
            return f"🧠 *Analyzing rules & metrics...*\n\n_{think_content}_"

    text = text.strip()
    if not text.startswith("{"):
        return text

    # Try full JSON parse first
    try:
        data = json.loads(text)
        md = f"### {data.get('title', 'Portfolio Risk Memo')}\n\n"
        if summary := data.get("summary"):
            md += f"> {summary}\n\n"
        if key_risks := data.get("key_risks"):
            md += "**Key Risks:**\n" + "\n".join(f"- {r}" for r in key_risks) + "\n\n"
        if recs := data.get("recommendations"):
            md += "**Recommendations:**\n" + "\n".join(f"- {r}" for r in recs) + "\n\n"
        if outlook := data.get("stress_test_outlook"):
            md += f"**Stress Test Outlook:**\n{outlook}\n\n"
        if caveats := data.get("caveats"):
            md += "**Caveats:**\n" + "\n".join(f"- ⚠️ {c}" for c in caveats) + "\n\n"
        return md
    except json.JSONDecodeError:
        pass

    # Partial JSON extraction via regex
    md_parts = []
    m_title = re.search(r'"title"\s*:\s*"([^"]*)', text)
    if m_title and m_title.group(1):
        md_parts.append(f"### {m_title.group(1)}")

    m_sum = re.search(r'"summary"\s*:\s*"([^"]*)', text)
    if m_sum and m_sum.group(1):
        md_parts.append(f"> {m_sum.group(1)}")

    m_r_partial = re.search(r'"key_risks"\s*:\s*\[([^\]]*)', text, re.DOTALL)
    if m_r_partial:
        items = re.findall(r'"([^"]+)"', m_r_partial.group(1))
        if items:
            md_parts.append("**Key Risks:**\n" + "\n".join(f"- {item}" for item in items))

    m_rec_partial = re.search(r'"recommendations"\s*:\s*\[([^\]]*)', text, re.DOTALL)
    if m_rec_partial:
        items = re.findall(r'"([^"]+)"', m_rec_partial.group(1))
        if items:
            md_parts.append("**Recommendations:**\n" + "\n".join(f"- {item}" for item in items))

    m_out = re.search(r'"stress_test_outlook"\s*:\s*"([^"]*)', text)
    if m_out and m_out.group(1):
        md_parts.append(f"**Stress Test Outlook:**\n{m_out.group(1)}")

    if md_parts:
        return "\n\n".join(md_parts)

    return "⏳ *Formatting risk memo...*"


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Portfolio Setup")

    # CSV File Upload Option in Popover Box
    with st.popover("📂 Import Portfolio CSV", width="stretch"):
        st.markdown("### Upload CSV File")
        st.caption("Upload a CSV containing `ticker` (or `symbol`) and `weight` columns.")
        uploaded_csv = st.file_uploader(
            "Select CSV file",
            type=["csv"],
            help="CSV with columns: ticker and weight",
        )
        if uploaded_csv is not None:
            try:
                df = pd.read_csv(uploaded_csv)
                cols = {str(c).lower().strip(): c for c in df.columns}
                t_col = cols.get("ticker") or cols.get("symbol") or cols.get("asset")
                w_col = cols.get("weight") or cols.get("weights") or cols.get("allocation")

                if t_col and w_col:
                    csv_t = [str(x).strip().upper() for x in df[t_col] if str(x).strip()]
                    csv_w = [float(x) for x in df[w_col] if pd.notnull(x)]
                    if len(csv_t) == len(csv_w) and len(csv_t) > 0:
                        st.session_state["csv_tickers"] = ", ".join(csv_t)
                        st.session_state["csv_weights"] = ", ".join(str(w) for w in csv_w)
                        st.success(f"✅ Loaded {len(csv_t)} assets from CSV!")
                    else:
                        st.error("Row count mismatch between tickers and weights.")
                else:
                    st.error("CSV requires 'ticker' and 'weight' header columns.")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    default_tickers = st.session_state.get("csv_tickers", "RELIANCE.NS, TCS.NS, INFY.NS")
    default_weights = st.session_state.get("csv_weights", "0.4, 0.35, 0.25")

    raw_tickers = st.text_input(
        "Tickers (comma-separated)",
        value=default_tickers,
        help="Yahoo Finance symbols, e.g. RELIANCE.NS, TCS.NS",
    )
    raw_weights = st.text_input(
        "Weights (comma-separated, must sum to 1.0)",
        value=default_weights,
    )
    benchmark = st.text_input("Benchmark ticker", value="^NSEI")
    lookback = st.slider("Lookback (days)", 90, 1825, 365, step=30)

    st.divider()

    weights = _parse_weights(raw_weights)
    tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]

    if weights is not None and tickers:
        total = math.fsum(weights)
        if abs(total - 1.0) > 1e-4:
            st.error(f"Weights sum to {total:.4f} — must equal 1.0")
        elif len(weights) != len(tickers):
            st.error(f"{len(tickers)} tickers but {len(weights)} weights")
        else:
            st.success(f"✅ {len(tickers)} assets · weights sum = {total:.4f}")

    run_btn = st.button("🚀 Run Audit", type="primary", width="stretch")

    st.divider()

    # Save session button
    if st.session_state["session_id"]:
        if st.button("💾 Save Session", width="stretch"):
            try:
                resp = requests.post(
                    f"{BACKEND}/session/{st.session_state['session_id']}/save", timeout=5
                )
                if resp.status_code == 200:
                    st.session_state["saved"] = True
                    st.toast("✅ Session flagged for persistence!", icon="💾")
                else:
                    st.toast("⚠️ Save returned an error.", icon="⚠️")
            except Exception:
                st.toast("⚠️ Could not reach backend.", icon="⚠️")
        if st.session_state["saved"]:
            st.caption("🟢 Marked for Postgres export")

    st.divider()
    st.caption("Backend: `http://localhost:8000` · [Swagger docs](http://localhost:8000/docs)")


# ---------------------------------------------------------------------------
# PHASE 1: Run Audit — call /analyze (fast, no LLM)
# ---------------------------------------------------------------------------
if run_btn:
    if not tickers:
        st.sidebar.error("Enter at least one ticker.")
    elif weights is None or len(weights) != len(tickers):
        st.sidebar.error("Ticker / weight mismatch — fix inputs first.")
    elif abs(math.fsum(weights) - 1.0) > 1e-4:
        st.sidebar.error("Weights must sum to 1.0.")
    else:
        with st.spinner("⏳ Fetching market data & computing risk metrics…"):
            try:
                resp = requests.post(
                    f"{BACKEND}/analyze",
                    json={
                        "tickers": tickers,
                        "weights": weights,
                        "benchmark": benchmark.strip().upper(),
                        "lookback_days": lookback,
                    },
                    timeout=90,  # fast endpoint — only yfinance + NumPy
                )
                if resp.status_code == 200:
                    data: dict[str, Any] = resp.json()
                    st.session_state["session_id"] = data["session_id"]
                    st.session_state["metrics"] = data
                    st.session_state["memo"] = None  # will arrive via /memo/stream
                    st.session_state["memo_meta"] = None
                    st.session_state["chat_history"] = []
                    st.session_state["saved"] = False
                    st.session_state["memo_streaming"] = True
                    st.toast("✅ Metrics ready! Streaming risk memo…", icon="📊")
                    st.rerun()
                else:
                    detail = resp.json().get("detail", resp.text)
                    st.error(f"Backend error {resp.status_code}: {detail}")
            except requests.exceptions.Timeout:
                st.error(
                    "⏱️ Backend timed out fetching market data. "
                    "Check your internet connection or try fewer tickers."
                )
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach backend at `http://localhost:8000`. Is uvicorn running?")
            except Exception as e:
                st.error(f"Unexpected error: {e}")


metrics: dict[str, Any] | None = st.session_state["metrics"]

# ---------------------------------------------------------------------------
# Session staleness check — detects if backend was restarted (in-memory wiped)
# ---------------------------------------------------------------------------
if metrics is not None and st.session_state["session_id"]:
    try:
        _ping = requests.get(
            f"{BACKEND}/session/{st.session_state['session_id']}/ping",
            timeout=3,
        )
        if _ping.status_code == 404:
            st.warning(
                "⚠️ **Session expired** — the backend was restarted and in-memory sessions were cleared.  "
                "Click **Run Audit** again to reload your portfolio.",
                icon="⚠️",
            )
            if st.button("🔄 Clear & Re-run", type="primary"):
                for _k in ["session_id", "metrics", "memo", "memo_meta", "chat_history", "saved", "memo_streaming"]:
                    st.session_state[_k] = None if _k != "chat_history" else []
                st.session_state["memo_streaming"] = False
                st.rerun()
            st.stop()
    except Exception:
        pass  # backend unreachable — surfaces as connection error on next user action

if metrics is None:
    st.markdown(
        """
        <div style='text-align:center; margin-top:80px; opacity:0.55;'>
            <h1>📊 QuantBrief</h1>
            <p style='font-size:1.1rem'>
                Enter your portfolio in the sidebar and click <b>Run Audit</b> to begin.
            </p>
            <p style='font-size:0.9rem'>Charts appear quickly · Risk memo streams in afterward</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ---------------------------------------------------------------------------
# Unpack data
# ---------------------------------------------------------------------------
portfolio = metrics["portfolio"]
assets: list[dict] = metrics["assets"]
shocks: list[dict] = metrics["shock_scenarios"]
cum_ret: dict = metrics["cumulative_returns"]
corr: dict = metrics["correlation_matrix"]
session_id: str = metrics["session_id"]
bm_label: str = metrics["request"]["benchmark"]

memo: dict | None = st.session_state["memo"]
memo_meta: dict | None = st.session_state["memo_meta"]

# ---------------------------------------------------------------------------
# Header (before memo arrives use a placeholder)
# ---------------------------------------------------------------------------
if memo:
    st.markdown(f"## {memo.get('title', 'Portfolio Risk Dashboard')}")
    lm = memo_meta or {}
    st.caption(
        f"Session `{session_id[:8]}…` · "
        f"Memo via {'🤖 LLM' if lm.get('synthesized_by_llm') else '⚙️ deterministic'}"
        f" ({lm.get('model', 'n/a')} · {lm.get('elapsed_ms', '?')} ms)"
    )
    st.markdown(f"> {memo.get('summary', '')}")
else:
    st.markdown("## 📊 Portfolio Risk Dashboard")
    st.caption(f"Session `{session_id[:8]}…` · Metrics loaded · Risk memo loading…")

st.divider()

# ---------------------------------------------------------------------------
# KPI metric cards
# ---------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📈 Ann. Return", _pct(portfolio["annualized_return"]))
col2.metric("📉 Ann. Volatility", _pct(portfolio["annualized_volatility"]))
col3.metric("⚡ Sharpe Ratio", _f2(portfolio["sharpe_ratio"]), help="(Return − Risk-Free) / Volatility")
col4.metric("β Beta", _f2(portfolio["beta"]))
col5.metric("🔴 95% Daily VaR", _pct(portfolio["var_95_daily"]), help="Max expected daily loss at 95% confidence")

st.divider()

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
chart_left, chart_mid, chart_right = st.columns([1, 2, 2])

with chart_left:
    st.subheader("Asset Allocation")
    fig_pie = px.pie(
        names=[a["ticker"] for a in assets],
        values=[a["weight"] for a in assets],
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=300)
    st.plotly_chart(fig_pie, width="stretch")

with chart_mid:
    st.subheader("Cumulative Returns")
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=cum_ret["dates"], y=cum_ret["portfolio"],
        mode="lines", name="Portfolio",
        line=dict(color="#00CC96", width=2),
    ))
    fig_line.add_trace(go.Scatter(
        x=cum_ret["dates"], y=cum_ret["benchmark"],
        mode="lines", name=bm_label,
        line=dict(color="#636EFA", width=2, dash="dash"),
    ))
    fig_line.update_layout(
        yaxis_title="Cumulative Return (%)", xaxis_title="",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=10, b=0), height=300,
    )
    st.plotly_chart(fig_line, width="stretch")

with chart_right:
    st.subheader("Correlation Heatmap")
    corr_tickers: list[str] = corr["tickers"]
    corr_vals: list[list[float]] = corr["matrix"]  # type: ignore[assignment]
    fig_heat = go.Figure(go.Heatmap(
        z=corr_vals, x=corr_tickers, y=corr_tickers,
        colorscale="RdBu", zmin=-1, zmax=1,
        text=[[_f2(v) for v in row] for row in corr_vals],
        texttemplate="%{text}", showscale=True,
    ))
    fig_heat.update_layout(margin=dict(t=10, b=0), height=300)
    st.plotly_chart(fig_heat, width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Stress scenario table
# ---------------------------------------------------------------------------
with st.expander("📋 Historical Stress Scenarios", expanded=True):
    st.caption("Portfolio and benchmark returns during major market crises.")
    st.table([
        {
            "Scenario": s["name"],
            "Period": f"{s['start_date']} → {s['end_date']}",
            "Portfolio": _pct(s.get("portfolio_return")),
            "Benchmark": _pct(s.get("benchmark_return")),
            "Coverage": f"{s.get('coverage_pct', 0):.0f}%",
        }
        for s in shocks
    ])

# ---------------------------------------------------------------------------
# PHASE 2: Stream LLM memo (fires once after /analyze completes)
# ---------------------------------------------------------------------------
if st.session_state["memo_streaming"] and memo is None:
    st.subheader("📝 Risk Memo")
    memo_placeholder = st.empty()
    memo_placeholder.info("⏳ Generating risk memo with AI advisor…")

    collected_tokens: list[str] = []
    final_memo: dict | None = None
    final_meta: dict | None = None

    try:
        with requests.get(
            f"{BACKEND}/session/{session_id}/memo/stream",
            stream=True,
            timeout=180,
            # Use GET-like pattern via POST with no body
        ) if False else requests.post(
            f"{BACKEND}/session/{session_id}/memo/stream",
            stream=True,
            timeout=180,
        ) as r:
            if r.status_code != 200:
                memo_placeholder.error(f"Memo error {r.status_code}: {r.text[:200]}")
                st.session_state["memo_streaming"] = False
            else:
                partial_text = ""
                for raw_line in r.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        evt = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    if evt.get("type") == "token":
                        token = evt.get("text", "")
                        partial_text += token
                        memo_placeholder.markdown(_format_partial_memo_json(partial_text) + " ▌")
                    elif evt.get("type") == "done":
                        final_memo = evt.get("memo")
                        final_meta = evt.get("memo_meta")
                        memo_placeholder.markdown(partial_text)
                        st.session_state["memo"] = final_memo
                        st.session_state["memo_meta"] = final_meta
                        st.session_state["memo_streaming"] = False
                        break
                    elif evt.get("type") == "error":
                        memo_placeholder.warning(f"⚠️ Memo generation issue: {evt.get('detail')}")
                        st.session_state["memo_streaming"] = False
                        break

    except requests.exceptions.Timeout:
        memo_placeholder.warning("⏱️ Memo generation timed out. You can still use the chat below.")
        st.session_state["memo_streaming"] = False
    except Exception as e:
        memo_placeholder.warning(f"Memo stream error: {e}")
        st.session_state["memo_streaming"] = False

    if final_memo:
        st.rerun()  # re-render header with real memo title/summary

elif memo:
    # Show full memo detail expander once we have it
    with st.expander("📝 Full Risk Memo", expanded=False):
        st.markdown("**Key Risks**")
        for risk in memo.get("key_risks", []):
            st.markdown(f"- {risk}")
        st.markdown("**Recommendations**")
        for rec in memo.get("recommendations", []):
            st.markdown(f"- {rec}")
        st.markdown("**Stress Test Outlook**")
        st.write(memo.get("stress_test_outlook", ""))
        if memo.get("caveats"):
            st.markdown("**Caveats**")
            for cav in memo["caveats"]:
                st.caption(f"⚠️ {cav}")

st.divider()

# ---------------------------------------------------------------------------
# Streaming Chat Interface
# ---------------------------------------------------------------------------
st.subheader("💬 Ask the Risk Advisor")
st.caption(
    "Ask follow-up questions about this portfolio. "
    "The advisor is grounded strictly in the quantitative metrics — it will not invent numbers."
)

for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("e.g. What's driving the high VaR? Which stock is most volatile?"):
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state["chat_history"].append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("⏳ *Thinking and analyzing portfolio...*")
        collected_chunks: list[str] = []

        try:
            with requests.post(
                f"{BACKEND}/chat/stream",
                json={"session_id": session_id, "message": user_input},
                stream=True,
                timeout=180,
            ) as r:
                if r.status_code == 404:
                    # Session wiped (backend restart) — clear state so user can re-run
                    response_placeholder.empty()
                    st.error(
                        "⚠️ **Session expired** — the backend was restarted and lost in-memory state.  "
                        "Click **Run Audit** in the sidebar to reload your portfolio."
                    )
                    for _k in ["session_id", "metrics", "memo", "memo_meta", "chat_history", "saved"]:
                        st.session_state[_k] = None if _k != "chat_history" else []
                    st.session_state["memo_streaming"] = False
                elif r.status_code != 200:
                    error_detail = r.json().get("detail", r.text)
                    response_placeholder.error(f"Backend error {r.status_code}: {error_detail}")
                else:
                    partial = ""
                    for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            collected_chunks.append(chunk)
                            partial += chunk
                            response_placeholder.markdown(partial + "▌")
                    full_reply = "".join(collected_chunks)
                    response_placeholder.markdown(full_reply)
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": full_reply}
                    )
        except requests.exceptions.Timeout:
            response_placeholder.error("⏱️ Chat response timed out — the model may still be loading.")
        except requests.exceptions.ConnectionError:
            response_placeholder.error("❌ Cannot reach backend. Is uvicorn running?")
        except Exception as e:
            response_placeholder.error(f"Streaming error: {e}")
