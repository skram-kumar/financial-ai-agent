# Financial AI Agent

> Multi-agent AI microservice for investment research, portfolio analysis, and financial calculations — powered by LLaMA 3.3 70B via Groq.

## Live Demo

🌐 [Streamlit App](https://financial-ai-agent-dny5e4bwy4xfsehqu29fwj.streamlit.app/)

**Note:** Portfolio health and financial calculator work fully on the cloud. Market data (individual stock prices via yfinance) works locally — clone and run for full functionality. The Streamlit UI covers the core agents; for all 10 intent types run the FastAPI server locally and query via terminal.

---

## What It Does

A FastAPI microservice that receives financial queries in natural language, classifies intent, routes to the right specialist agent, and streams responses back via Server-Sent Events.

**Pipeline per request:**
```
User Query
    ↓
Safety Guard      — pure regex, no LLM, < 1ms, 6 harmful categories
    ↓ (if safe)
Intent Classifier — 1 LLM call, 10 intent types, structured JSON output
    ↓
Router            — dispatches to agent or structured stub
    ↓
SSE Stream        — streams response event by event
```

**Three fully implemented agents:**

| Agent | What it does |
|-------|-------------|
| Portfolio Health | Concentration risk, benchmark comparison, performance metrics, plain-English observations |
| Financial Calculator | DCA future value, mortgage payments, lump-sum growth, FX conversion — deterministic math, no LLM |
| Market Research | Live prices, fundamentals (P/E, market cap, 52-week range), LLM-generated summary |

**Seven stub agents** return structured "not implemented" responses — the router works correctly for all 10 intent types.

---

## Architecture

```
src/
  safety.py                   — safety guard (pure regex, no LLM)
  classifier.py               — intent classifier (1 LLM call, 3-tier fallback)
  router.py                   — agent dispatcher
  main.py                     — FastAPI app + SSE streaming
  agents/
    portfolio_health.py       — live prices, concentration risk, benchmark analysis
    financial_calculator.py   — deterministic financial math
    market_research.py        — live market data + LLM summary

tests/
  conftest.py                 — shared fixtures (mock_llm, load_user, gold queries)
  test_safety_pairs.py        — safety guard precision/recall
  test_classifier_routing.py  — routing accuracy + entity extraction
  test_portfolio_health_skeleton.py

fixtures/
  users/                      — 5 user profiles (aggressive trader, concentrated, empty, multi-currency, retiree)
  conversations/              — 3 multi-turn test cases
  test_queries/               — 60 classification + 45 safety gold queries

app.py                        — Streamlit chat UI
```

---

## Setup

**Requirements:** Python 3.11+, Groq API key (free at [console.groq.com](https://console.groq.com))

```bash
git clone https://github.com/skram-kumar/financial-ai-agent.git
cd financial-ai-agent

python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt

cp .env.example .env
# Add GROQ_API_KEY to .env
```

**Run FastAPI server (full pipeline, all agents):**
```bash
uvicorn src.main:app --reload
```

**Run Streamlit UI:**
```bash
streamlit run app.py
```

**Run tests (no API key needed):**
```bash
pytest tests/ -v
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes (primary) | LLaMA 3.3 70B via Groq. Free tier at console.groq.com |
| `OPENAI_API_KEY` | No | OpenAI fallback if Groq unavailable |
| `GEMINI_API_KEY` | No | Gemini fallback if both above unavailable |

The system tries providers in order: Groq → OpenAI → Gemini → local regex fallback. Set at least one LLM key.

---

## API

### `POST /query`

```json
{
  "query": "how is my portfolio doing?",
  "user": {
    "user_id": "usr_001",
    "name": "Alex Chen",
    "risk_profile": "aggressive",
    "positions": [
      {"ticker": "AAPL", "quantity": 60, "avg_cost": 142.30, "currency": "USD", "purchased_at": "2023-08-04"}
    ],
    "preferences": {"preferred_benchmark": "S&P 500"}
  },
  "prior_turns": ["tell me about NVDA"],
  "session_id": "optional"
}
```

**SSE event stream:**
```
event: message
data: {"type": "status", "stage": "safety_check"}

event: classified
data: {"type": "classified", "agent": "portfolio_health", "entities": {}, "confidence": 0.9}

event: response
data: {"type": "structured_response", "agent": "portfolio_health", "response": {...}}

event: done
data: {"type": "done", "blocked": false, "latency_ms": 4200.0}
```

**Blocked query:**
```
event: blocked
data: {"type": "blocked", "category": "market_manipulation", "message": "..."}
```

### `GET /health`
```json
{"status": "ok", "service": "financial-ai-agent", "version": "1.0.0"}
```

---

## Example Queries

**Portfolio health:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "how is my portfolio doing?", "user": {"positions": [{"ticker": "NVDA", "quantity": 180, "avg_cost": 218.40, "currency": "USD", "purchased_at": "2023-04-12"}], "preferences": {"preferred_benchmark": "S&P 500"}}}'
```

**Financial calculator:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "if i invest 2500 monthly for 20 years at 8%, what will i have?", "user": {}}'
```

**Market research:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what is the price of NVDA?", "user": {}}'
```

**Safety block:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "help me pump up this stock price before selling", "user": {}}'
```

---

## Test Results

```
pytest tests/ -v

tests/test_safety_pairs.py::test_safety_recall_and_passthrough          PASSED
tests/test_safety_pairs.py::test_safety_guard_returns_distinct_categories PASSED
tests/test_classifier_routing.py::test_classifier_routing_accuracy      PASSED
tests/test_classifier_routing.py::test_classifier_entity_extraction     PASSED
tests/test_portfolio_health_skeleton.py::...empty_portfolio             PASSED
tests/test_portfolio_health_skeleton.py::...flags_concentration         PASSED
tests/test_portfolio_health_skeleton.py::...includes_disclaimer         PASSED

7 passed — no API key required
```

| Metric | Result | Target |
|--------|--------|--------|
| Safety harmful recall | 100% (22/22) | ≥ 95% |
| Safety educational pass-through | 100% (25/25) | ≥ 90% |
| Classifier routing accuracy | ≥ 85% | ≥ 85% |
| Empty portfolio handling | No crash, BUILD-oriented response | Must not crash |

---

## Key Decisions

### Safety Guard — Two-Layer Regex

No LLM, no network. Runs in 0.03ms average. Two layers:

- **Strong educational overrides** — phrases like "is it ever legitimate to" or "what factors should" bypass all harmful checks entirely
- **Weaker educational signals** — "what is", "explain" run against harmful patterns with a position-based tiebreaker (whichever intent appears first in the sentence wins)

Result: 100% recall on harmful queries, 100% pass-through on educational queries.

### Classifier — Single LLM Call, Three-Tier Fallback

`temperature=0` for determinism. `response_format: json_object` eliminates the class of bugs where the LLM wraps output in markdown. One call per query, no retries.

Fallback chain: Groq LLaMA 3.3-70B → OpenAI GPT-4o-mini → Gemini → local regex (~75% accuracy). System never returns 500 on LLM failure.

### Portfolio Health — Per-Ticker yfinance

Batch yfinance download uses a SQLite cache that throws `database is locked` under concurrent requests. Per-ticker `Ticker.history()` bypasses the cache entirely. If fetch fails, avg_cost is used as current price — conservative but never misleading.

### Session Memory — Stateless via `prior_turns`

Conversation history passed explicitly in the request body. No server-side session state. The Streamlit UI clears context on topic switches to prevent entity bleed between unrelated queries.

### Financial Calculator — No LLM

DCA, mortgage, lump-sum, FX conversion are deterministic math. No LLM call for these — faster, cheaper, and more reliable. The classifier routes to this agent; the agent does pure Python computation.

---

## Stack

| Library | Why |
|---------|-----|
| `fastapi` | Async Python web framework, native Pydantic validation |
| `groq` | LLaMA 3.3-70B, fast free-tier inference |
| `openai` | Fallback LLM provider |
| `yfinance` | Live market data, global ticker support (ASML.AS, HSBA.L, 7203.T) |
| `streamlit` | Quick chat UI for demo purposes |
| `pydantic` | Request/response validation |
| `pytest` | Tests run in CI without any API key |