"""
Valura AI — Streamlit Frontend

Chat interface for the financial AI agent.
Connects to the FastAPI backend running on localhost:8000.
"""

import streamlit as st
import requests
import json

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

    user_name = st.text_input("Name", value="Alex Chen")
    risk_profile = st.selectbox(
        "Risk Profile",
        ["aggressive", "moderate", "conservative"],
        index=1,
    )
    benchmark = st.selectbox(
        "Benchmark",
        ["S&P 500", "FTSE 100", "NIKKEI 225", "MSCI World"],
        index=0,
    )

    st.subheader("📊 Portfolio Holdings")
    st.caption("Add your stock positions below")

    # Dynamic position input
    if "positions" not in st.session_state:
        st.session_state.positions = [
            {"ticker": "AAPL", "quantity": 10, "avg_cost": 150.0, "currency": "USD", "purchased_at": "2023-01-01"},
        ]

    positions = []
    for i, pos in enumerate(st.session_state.positions):
        with st.expander(f"Position {i+1}: {pos['ticker']}", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                ticker = st.text_input("Ticker", value=pos["ticker"], key=f"ticker_{i}").upper()
                quantity = st.number_input("Quantity", value=pos["quantity"], min_value=1, key=f"qty_{i}")
            with col2:
                avg_cost = st.number_input("Avg Cost ($)", value=pos["avg_cost"], min_value=0.01, key=f"cost_{i}")
                currency = st.selectbox("Currency", ["USD", "GBP", "EUR", "JPY", "SGD"], key=f"cur_{i}")
            purchased_at = st.date_input("Purchase Date", key=f"date_{i}")
            positions.append({
                "ticker": ticker,
                "quantity": quantity,
                "avg_cost": avg_cost,
                "currency": currency,
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
        ["None", "Active Trader (usr_001)", "Concentrated (usr_003)", "Empty Portfolio (usr_004)", "Retiree (usr_008)"],
    )
    if st.button("Load Profile") and fixture != "None":
        profiles = {
            "Active Trader (usr_001)": {
                "name": "Alex Chen", "risk": "aggressive",
                "positions": [
                    {"ticker": "AAPL", "quantity": 60, "avg_cost": 142.30, "currency": "USD", "purchased_at": "2023-08-04"},
                    {"ticker": "MSFT", "quantity": 40, "avg_cost": 305.10, "currency": "USD", "purchased_at": "2023-09-01"},
                    {"ticker": "NVDA", "quantity": 35, "avg_cost": 412.85, "currency": "USD", "purchased_at": "2023-11-15"},
                    {"ticker": "GOOGL", "quantity": 25, "avg_cost": 132.40, "currency": "USD", "purchased_at": "2023-10-22"},
                ]
            },
            "Concentrated (usr_003)": {
                "name": "Marcus Webb", "risk": "moderate",
                "positions": [
                    {"ticker": "NVDA", "quantity": 180, "avg_cost": 218.40, "currency": "USD", "purchased_at": "2023-04-12"},
                    {"ticker": "AAPL", "quantity": 8, "avg_cost": 168.20, "currency": "USD", "purchased_at": "2024-05-20"},
                ]
            },
            "Empty Portfolio (usr_004)": {
                "name": "Jamie Patel", "risk": "moderate",
                "positions": []
            },
            "Retiree (usr_008)": {
                "name": "Robert Chen", "risk": "conservative",
                "positions": [
                    {"ticker": "JNJ", "quantity": 100, "avg_cost": 155.0, "currency": "USD", "purchased_at": "2020-01-15"},
                    {"ticker": "PG", "quantity": 80, "avg_cost": 135.0, "currency": "USD", "purchased_at": "2020-03-10"},
                ]
            },
        }
        profile = profiles[fixture]
        user_name = profile["name"]
        st.session_state.positions = [
            {**p, "purchased_at": p.get("purchased_at", "2023-01-01")}
            for p in profile["positions"]
        ] if profile["positions"] else [{"ticker": "", "quantity": 1, "avg_cost": 1.0, "currency": "USD", "purchased_at": "2023-01-01"}]
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
# Chat interface
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prior_turns" not in st.session_state:
    st.session_state.prior_turns = []

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
    "Portfolio Health": "how is my portfolio doing?",
    "Market Overview": "what happened in markets today?",
    "NVDA Research": "what is the price of NVDA?",
    "DCA Calculator": "if i invest 1000 monthly for 10 years at 8%, what will i have?",
}
for (label, query), col in zip(quick_queries.items(), [col1, col2, col3, col4]):
    if col.button(label):
        st.session_state["pending_query"] = query

# Chat input
query = st.chat_input("Ask anything — portfolio health, market research, calculations...")

# Handle quick action
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Call API
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        structured_data = None
        full_text = ""
        agent_name = ""

        try:
            with requests.post(
                "http://localhost:8000/query",
                json={
                    "query": query,
                    "user": user,
                    "prior_turns": st.session_state.prior_turns,
                },
                stream=True,
                timeout=60,
            ) as resp:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8")
                    if line.startswith("data:"):
                        try:
                            data = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            continue

                        event_type = data.get("type")

                        if event_type == "status":
                            stage = data.get("stage", "")
                            status_placeholder.caption(f"⏳ {stage.replace('_', ' ').title()}...")

                        elif event_type == "blocked":
                            full_text = f"⚠️ **Blocked:** {data.get('message', '')}"
                            response_placeholder.markdown(full_text)
                            status_placeholder.empty()

                        elif event_type == "classified":
                            agent_name = data.get("agent", "")
                            status_placeholder.caption(f"🎯 Routing to: **{agent_name.replace('_', ' ').title()}**")

                        elif event_type == "text_chunk":
                            full_text += data.get("content", "")
                            response_placeholder.markdown(full_text + "▌")

                        elif event_type == "structured_response":
                            response = data.get("response", {})
                            structured_data = {k: v for k, v in response.items()
                                             if k not in ("disclaimer", "_meta", "narrative")}
                            disclaimer = response.get("disclaimer", "")

                            # Format response based on agent
                            if agent_name == "portfolio_health":
                                perf = response.get("performance", {})
                                conc = response.get("concentration_risk", {})
                                bench = response.get("benchmark_comparison", {})
                                obs = response.get("observations", [])

                                if not full_text:
                                    full_text = ""

                                # Show metrics
                                col1, col2, col3 = st.columns(3)
                                col1.metric(
                                    "Total Return",
                                    f"{perf.get('total_return_pct', 0):.1f}%",
                                )
                                col2.metric(
                                    "Portfolio Value",
                                    f"${perf.get('total_value', 0):,.0f}",
                                )
                                col3.metric(
                                    "Alpha vs " + bench.get("benchmark", ""),
                                    f"{bench.get('alpha_pct', 0):+.1f}%",
                                )

                                # Concentration flag
                                flag = conc.get("flag", "low")
                                flag_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(flag, "⚪")
                                st.markdown(f"**Concentration Risk:** {flag_color} {flag.upper()} — Top position: {conc.get('top_position_pct', 0):.1f}%")

                                # Observations
                                for obs_item in obs:
                                    severity = obs_item.get("severity", "info")
                                    icon = "⚠️" if severity == "warning" else "ℹ️"
                                    st.markdown(f"{icon} {obs_item.get('text', '')}")

                                if disclaimer:
                                    st.caption(f"*{disclaimer}*")

                            elif agent_name == "financial_calculator":
                                result = response.get("result", {})
                                summary = response.get("summary", "")
                                if summary:
                                    full_text = summary
                                if result:
                                    st.json(result)
                                if disclaimer:
                                    st.caption(f"*{disclaimer}*")

                            elif agent_name == "market_research":
                                summary = response.get("summary", "")
                                if summary:
                                    full_text = summary
                                tickers = response.get("tickers", [])
                                if tickers:
                                    for t in tickers:
                                        if "error" not in t:
                                            col1, col2, col3 = st.columns(3)
                                            col1.metric(t.get("name", t["ticker"]),
                                                       f"${t.get('current_price', 0):,.2f}")
                                            col2.metric("Day Change",
                                                       f"{t.get('day_change_pct', 0):+.2f}%")
                                            col3.metric("Month Change",
                                                       f"{t.get('month_change_pct', 0):+.2f}%")
                                if disclaimer:
                                    st.caption(f"*{disclaimer}*")

                            else:
                                # Stub or other agent
                                message = response.get("message", "")
                                if message:
                                    full_text = f"ℹ️ {message}"

                        elif event_type == "done":
                            status_placeholder.empty()
                            latency = data.get("latency_ms", 0)
                            st.caption(f"⚡ Response time: {latency/1000:.1f}s")

            if full_text:
                response_placeholder.markdown(full_text)

        except requests.exceptions.ConnectionError:
            full_text = "❌ Cannot connect to the API server. Make sure `uvicorn src.main:app --reload` is running."
            response_placeholder.markdown(full_text)
        except Exception as e:
            full_text = f"❌ Error: {str(e)}"
            response_placeholder.markdown(full_text)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_text,
        "structured": structured_data,
    })
    st.session_state.prior_turns.append(query)
    if len(st.session_state.prior_turns) > 10:
        st.session_state.prior_turns = st.session_state.prior_turns[-10:]

# Clear chat button
if st.session_state.messages:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.prior_turns = []
        st.rerun()