"""
Financial AI Agent — Streamlit App
Calls the pipeline directly (no FastAPI server needed).
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.safety import check as safety_check
from src.classifier import classify
from src.router import route

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial AI Agent",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Financial AI Agent")
st.caption("Powered by LLaMA 3.3 70B via Groq • Portfolio health, market research, financial calculations")

# ---------------------------------------------------------------------------
# Sidebar — Portfolio Input
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("👤 Your Profile")

    user_name    = st.text_input("Name", value="Alex Chen")
    risk_profile = st.selectbox("Risk Profile", ["aggressive", "moderate", "conservative"], index=1)
    benchmark    = st.selectbox("Benchmark", ["S&P 500", "FTSE 100", "NIKKEI 225", "MSCI World"], index=0)

    st.subheader("📊 Portfolio Holdings")

    if "positions" not in st.session_state:
        st.session_state.positions = [
            {"ticker": "AAPL", "quantity": 10, "avg_cost": 150.0, "currency": "USD", "purchased_at": "2023-01-01"},
        ]

    positions = []
    for i, pos in enumerate(st.session_state.positions):
        with st.expander(f"Position {i+1}: {pos['ticker']}", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                ticker   = st.text_input("Ticker", value=pos["ticker"], key=f"ticker_{i}").upper()
                quantity = st.number_input("Quantity", value=pos["quantity"], min_value=1, key=f"qty_{i}")
            with col2:
                avg_cost = st.number_input("Avg Cost ($)", value=pos["avg_cost"], min_value=0.01, key=f"cost_{i}")
                currency = st.selectbox("Currency", ["USD", "GBP", "EUR", "JPY", "SGD"], key=f"cur_{i}")
            purchased_at = st.date_input("Purchase Date", key=f"date_{i}")
            positions.append({
                "ticker":       ticker,
                "quantity":     quantity,
                "avg_cost":     avg_cost,
                "currency":     currency,
                "purchased_at": str(purchased_at),
            })

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Add Position"):
            st.session_state.positions.append(
                {"ticker": "MSFT", "quantity": 5, "avg_cost": 300.0, "currency": "USD", "purchased_at": "2023-01-01"}
            )
            st.rerun()
    with col2:
        if st.button("🗑️ Remove Last") and len(st.session_state.positions) > 1:
            st.session_state.positions.pop()
            st.rerun()

    # Quick load fixture users
    st.subheader("⚡ Load Sample User")
    fixture = st.selectbox(
        "Sample profiles",
        ["None", "Active Trader", "Concentrated (NVDA)", "Empty Portfolio", "Retiree"],
    )
    if st.button("Load Profile") and fixture != "None":
        profiles = {
            "Active Trader": {
                "name": "Alex Chen", "risk": "aggressive",
                "positions": [
                    {"ticker": "AAPL",  "quantity": 60, "avg_cost": 142.30, "currency": "USD", "purchased_at": "2023-08-04"},
                    {"ticker": "MSFT",  "quantity": 40, "avg_cost": 305.10, "currency": "USD", "purchased_at": "2023-09-01"},
                    {"ticker": "NVDA",  "quantity": 35, "avg_cost": 412.85, "currency": "USD", "purchased_at": "2023-11-15"},
                    {"ticker": "GOOGL", "quantity": 25, "avg_cost": 132.40, "currency": "USD", "purchased_at": "2023-10-22"},
                ],
            },
            "Concentrated (NVDA)": {
                "name": "Marcus Webb", "risk": "moderate",
                "positions": [
                    {"ticker": "NVDA", "quantity": 180, "avg_cost": 218.40, "currency": "USD", "purchased_at": "2023-04-12"},
                    {"ticker": "AAPL", "quantity": 8,   "avg_cost": 168.20, "currency": "USD", "purchased_at": "2024-05-20"},
                ],
            },
            "Empty Portfolio": {
                "name": "Jamie Patel", "risk": "moderate",
                "positions": [],
            },
            "Retiree": {
                "name": "Robert Chen", "risk": "conservative",
                "positions": [
                    {"ticker": "JNJ", "quantity": 100, "avg_cost": 155.0, "currency": "USD", "purchased_at": "2020-01-15"},
                    {"ticker": "PG",  "quantity": 80,  "avg_cost": 135.0, "currency": "USD", "purchased_at": "2020-03-10"},
                ],
            },
        }
        profile = profiles[fixture]
        st.session_state.positions = profile["positions"] if profile["positions"] else [
            {"ticker": "", "quantity": 1, "avg_cost": 1.0, "currency": "USD", "purchased_at": "2023-01-01"}
        ]
        st.rerun()

# ---------------------------------------------------------------------------
# Build user object
# ---------------------------------------------------------------------------
user = {
    "user_id":      "usr_streamlit",
    "name":         user_name,
    "risk_profile": risk_profile,
    "positions":    [p for p in positions if p["ticker"]],
    "preferences":  {"preferred_benchmark": benchmark},
}

# ---------------------------------------------------------------------------
# Pipeline function
# ---------------------------------------------------------------------------
def run_pipeline(query: str, prior_turns: list[str], user: dict) -> dict:
    """Run the full pipeline directly."""
    # 1. Safety check
    safety_verdict = safety_check(query)
    if safety_verdict.blocked:
        return {
            "blocked":  True,
            "category": safety_verdict.category,
            "message":  safety_verdict.message,
        }

    # 2. Classify
    result = classify(query=query, prior_turns=prior_turns)

    # 3. Route
    response = route(result=result, user=user)

    return {
        "blocked":    False,
        "agent":      result.agent,
        "entities":   result.entities,
        "confidence": result.confidence,
        "fallback":   result.fallback,
        "response":   response,
    }

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
if "messages"    not in st.session_state: st.session_state.messages    = []
if "prior_turns" not in st.session_state: st.session_state.prior_turns = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("structured"):
            with st.expander("📊 Structured Data"):
                st.json(msg["structured"])

# Quick action buttons
st.subheader("Quick Actions")
col1, col2, col3, col4 = st.columns(4)
quick_queries = {
    "Portfolio Health":  "how is my portfolio doing?",
    "Market Overview":   "what happened in markets today?",
    "Stock Research":    "what is the price of AAPL?",
    "DCA Calculator":    "if i invest 1000 monthly for 10 years at 8%, what will i have?",
}
for (label, qquery), col in zip(quick_queries.items(), [col1, col2, col3, col4]):
    if col.button(label):
        st.session_state["pending_query"] = qquery

# Chat input
query = st.chat_input("Ask anything — portfolio health, market research, calculations...")
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            output = run_pipeline(query, st.session_state.prior_turns, user)

        full_text      = ""
        structured_data = None

        if output.get("blocked"):
            full_text = f"⚠️ **Blocked ({output.get('category', '')}):** {output.get('message', '')}"
            st.markdown(full_text)

        else:
            agent    = output.get("agent", "")
            response = output.get("response", {})
            fallback = output.get("fallback", False)

            st.caption(f"🎯 Agent: **{agent.replace('_', ' ').title()}** | "
                       f"Confidence: {output.get('confidence', 0):.0%} | "
                       f"{'⚡ Groq LLaMA' if not fallback else '🔄 Local fallback'}")

            if agent == "portfolio_health":
                perf  = response.get("performance", {})
                conc  = response.get("concentration_risk", {})
                bench = response.get("benchmark_comparison", {})
                obs   = response.get("observations", [])
                narr  = response.get("narrative", "")

                # Metrics row
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Return",     f"{perf.get('total_return_pct', 0):.1f}%")
                c2.metric("Portfolio Value",  f"${perf.get('total_value', 0):,.0f}")
                c3.metric(f"Alpha vs {bench.get('benchmark', 'benchmark')}",
                          f"{bench.get('alpha_pct', 0):+.1f}%")

                # Concentration
                flag       = conc.get("flag", "low")
                flag_icon  = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(flag, "⚪")
                st.markdown(f"**Concentration Risk:** {flag_icon} {flag.upper()} — "
                            f"Top position: {conc.get('top_position_pct', 0):.1f}%")

                # Observations
                for o in obs:
                    icon = "⚠️" if o.get("severity") == "warning" else "ℹ️"
                    st.markdown(f"{icon} {o.get('text', '')}")

                if narr:
                    full_text = narr
                    st.markdown(narr)

                disclaimer = response.get("disclaimer", "")
                if disclaimer:
                    st.caption(f"*{disclaimer}*")

                structured_data = {k: v for k, v in response.items()
                                   if k not in ("disclaimer", "_meta", "narrative")}

            elif agent == "financial_calculator":
                result  = response.get("result", {})
                summary = response.get("summary", "")
                full_text = summary
                st.markdown(summary)
                if result:
                    st.json(result)
                disclaimer = response.get("disclaimer", "")
                if disclaimer:
                    st.caption(f"*{disclaimer}*")

            elif agent == "market_research":
                summary = response.get("summary", "")
                full_text = summary
                tickers = response.get("tickers", [])
                indices = response.get("indices", [])

                if tickers:
                    for t in tickers:
                        if "error" not in t:
                            c1, c2, c3 = st.columns(3)
                            c1.metric(t.get("name", t["ticker"]), f"${t.get('current_price', 0):,.2f}")
                            c2.metric("Day Change",   f"{t.get('day_change_pct', 0):+.2f}%")
                            c3.metric("Month Change", f"{t.get('month_change_pct', 0):+.2f}%")

                if indices:
                    for idx in indices:
                        if "error" not in idx:
                            c1, c2, c3 = st.columns(3)
                            c1.metric(idx["index"],   f"{idx.get('current_value', 0):,.2f}")
                            c2.metric("Day Change",   f"{idx.get('day_change_pct', 0):+.2f}%")
                            c3.metric("Month Change", f"{idx.get('month_change_pct', 0):+.2f}%")

                st.markdown(summary)
                disclaimer = response.get("disclaimer", "")
                if disclaimer:
                    st.caption(f"*{disclaimer}*")

            else:
                # Stub or general
                message   = response.get("message", "")
                full_text = f"ℹ️ {message}" if message else str(response)
                st.markdown(full_text)

    st.session_state.messages.append({
        "role":       "assistant",
        "content":    full_text,
        "structured": structured_data,
    })
    st.session_state.prior_turns.append(query)
    if len(st.session_state.prior_turns) > 10:
        st.session_state.prior_turns = st.session_state.prior_turns[-10:]

# Clear chat
if st.session_state.messages:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages    = []
        st.session_state.prior_turns = []
        st.rerun()