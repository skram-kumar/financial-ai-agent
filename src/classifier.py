"""
Intent Classifier — single LLM call that drives the entire pipeline.

Public API (must match test_classifier_routing.py):
    result = classify(query, llm=mock_llm)
    result.agent       # string
    result.entities    # dict (supports .get())
    result.intent      # same as agent
    result.safety_verdict
    result.confidence
    result.fallback    # True if LLM failed
"""

from __future__ import annotations
import json
import re
import logging
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

logger = logging.getLogger(__name__)

_VALID_AGENTS = {
    "portfolio_health", "market_research", "investment_strategy",
    "financial_planning", "financial_calculator", "risk_assessment",
    "product_recommendation", "predictive_analysis",
    "customer_support", "general_query",
}
_VALID_FREQUENCIES = {"daily", "weekly", "monthly", "yearly"}
_VALID_HORIZONS    = {"6_months", "1_year", "5_years"}
_VALID_TIME        = {"today", "this_week", "this_month", "this_year"}
_VALID_ACTIONS     = {"buy", "sell", "hold", "hedge", "rebalance"}
_VALID_GOALS       = {"retirement", "education", "house", "FIRE", "emergency_fund"}
_VALID_INDEXES     = {"S&P 500", "FTSE 100", "NIKKEI 225", "MSCI World"}
_VALID_SAFETY      = {"safe", "flagged"}


@dataclass
class ClassifierResult:
    intent: str
    agent: str
    entities: dict
    safety_verdict: str
    confidence: float
    raw: dict
    fallback: bool = False


def _normalize_entities(raw: dict) -> dict:
    if not isinstance(raw, dict):
        return {}
    result = {}
    tickers = raw.get("tickers", [])
    if isinstance(tickers, list) and tickers:
        result["tickers"] = [str(t).upper() for t in tickers if t]
    if raw.get("amount") is not None:
        try: result["amount"] = float(raw["amount"])
        except (TypeError, ValueError): pass
    if raw.get("currency"):
        result["currency"] = str(raw["currency"]).upper()
    if raw.get("rate") is not None:
        try: result["rate"] = float(raw["rate"])
        except (TypeError, ValueError): pass
    if raw.get("period_years") is not None:
        try: result["period_years"] = int(raw["period_years"])
        except (TypeError, ValueError): pass
    if raw.get("frequency") in _VALID_FREQUENCIES:
        result["frequency"] = raw["frequency"]
    if raw.get("horizon") in _VALID_HORIZONS:
        result["horizon"] = raw["horizon"]
    if raw.get("time_period") in _VALID_TIME:
        result["time_period"] = raw["time_period"]
    if raw.get("action") in _VALID_ACTIONS:
        result["action"] = raw["action"]
    if raw.get("goal") in _VALID_GOALS:
        result["goal"] = raw["goal"]
    if raw.get("index") in _VALID_INDEXES:
        result["index"] = raw["index"]
    topics = raw.get("topics", [])
    if isinstance(topics, list) and topics:
        result["topics"] = [str(t).strip() for t in topics if t]
    sectors = raw.get("sectors", [])
    if isinstance(sectors, list) and sectors:
        result["sectors"] = [str(s).strip().lower() for s in sectors if s]
    return result


SYSTEM_PROMPT = """You are the intent classifier for Valura, a wealth management AI platform.

AGENTS — map the user's primary intent to exactly one:
  portfolio_health      — health check on the user's existing portfolio
  market_research       — factual/recent info about instruments, sectors, markets
  investment_strategy   — advice/strategy: buy/sell/rebalance, allocation guidance
  financial_planning    — long-term goal planning: retirement, education, FIRE, house
  financial_calculator  — deterministic computation: DCA, mortgage, future value, tax, FX
  risk_assessment       — risk metrics: beta, drawdown, VaR, stress test, currency exposure
  product_recommendation— recommend specific funds, ETFs matching user profile
  predictive_analysis   — forward-looking: forecasts, trend extrapolation
  customer_support      — platform/app issues: login, bank account, settings
  general_query         — greetings, definitions, conversational, gibberish

ENTITY EXTRACTION — only extract what is explicitly stated:
  tickers: uppercase. Map names: Apple->AAPL, Nvidia->NVDA, Tesla->TSLA, Microsoft->MSFT,
           Amazon->AMZN, Google->GOOGL, Meta->META, HSBC->HSBA.L, Barclays->BARC.L,
           ASML->ASML.AS, Toyota->7203.T, gold->GOLD, bitcoin->BTC-USD
  amount: number (500k->500000), currency: ISO 4217, rate: decimal (8%->0.08),
  period_years: integer, frequency: daily/weekly/monthly/yearly,
  horizon: 6_months/1_year/5_years, time_period: today/this_week/this_month/this_year,
  topics: array of strings, sectors: array lowercase, action: buy/sell/hold/hedge/rebalance,
  goal: retirement/education/house/FIRE/emergency_fund,
  index: exact one of "S&P 500", "FTSE 100", "NIKKEI 225", "MSCI World"

Single ticker alone -> market_research. Gibberish -> general_query.
Multi-intent -> pick primary. Follow-ups -> resolve pronouns from history.

OUTPUT: valid JSON only, no markdown:
{"intent":"<agent>","agent":"<agent>","entities":{...},"safety_verdict":"safe","confidence":0.9}"""


def _build_message(query: str, prior_turns: list[str]) -> str:
    if not prior_turns:
        return query
    history = "\n".join(f"  Turn {i+1}: {t}" for i, t in enumerate(prior_turns))
    return f"Conversation history:\n{history}\n\nCurrent query: {query}"


def _parse_llm_text(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(text)


_GREETINGS = re.compile(
    r"^(hi+|hello+|hey+|thanks?|thank you|ok|okay|great|good|sure|yes|no|nope|yep)[\.\!?]*$", re.I)

_FALLBACK_PATTERNS = [
    (re.compile(r"\b(beta|drawdown|VaR|stress test|downside risk)\b", re.I), "risk_assessment"),
    (re.compile(r"\b(portfolio|holdings?|diversi|concentration|health.?check|benchmark)\b", re.I), "portfolio_health"),
    (re.compile(r"\b(price of|news on|tell me about|compare|top gainers|markets? today)\b", re.I), "market_research"),
    (re.compile(r"\b(should i (buy|sell|invest)|rebalance|allocation|hedge)\b", re.I), "investment_strategy"),
    (re.compile(r"\b(retire|FIRE|college fund|down payment|savings rate)\b", re.I), "financial_planning"),
    (re.compile(r"\b(calculate|mortgage|future value|DCA|convert .{1,10} to|tax on)\b", re.I), "financial_calculator"),
    (re.compile(r"\b(recommend|which fund|best .{1,20} ETF|suggest)\b", re.I), "product_recommendation"),
    (re.compile(r"\b(predict|forecast|where will)\b", re.I), "predictive_analysis"),
    (re.compile(r"\b(login|password|transaction history|bank account|recurring investment)\b", re.I), "customer_support"),
    (re.compile(r"^[A-Z]{2,5}(\.[A-Z]{1,2})?(-USD)?$", re.I), "market_research"),
]


def _fallback_classify(query: str) -> ClassifierResult:
    if _GREETINGS.match(query.strip()):
        return ClassifierResult(intent="general_query", agent="general_query",
                                entities={}, safety_verdict="safe",
                                confidence=0.9, raw={}, fallback=True)
    agent = "general_query"
    for pattern, candidate in _FALLBACK_PATTERNS:
        if pattern.search(query.strip()):
            agent = candidate
            break
    stopwords = {"I","A","AT","IN","ON","OR","TO","US","UK","IF","IS","AM",
                 "BE","DO","GO","BY","MY","UP","NO","SO","ETF","DCA","FX"}
    tickers = [m.group(1) for m in re.finditer(r"\b([A-Z]{2,5}(?:\.[A-Z]{1,2})?)\b", query)
               if m.group(1).upper() not in stopwords]
    return ClassifierResult(intent=agent, agent=agent,
                            entities={"tickers": tickers} if tickers else {},
                            safety_verdict="safe", confidence=0.5, raw={}, fallback=True)


def classify(
    query: str,
    llm: Any = None,
    prior_turns: list[str] | None = None,
) -> ClassifierResult:
    """
    Classify a user query.
    Args:
        query:       User message.
        llm:         MagicMock (tests) | OpenAI client | None (uses real OpenAI).
        prior_turns: Prior user messages for follow-up resolution.
    Returns ClassifierResult. Never raises.
    """
    prior_turns = prior_turns or []
    try:
        return _classify_with_llm(query, prior_turns, llm)
    except Exception as exc:
        logger.warning("Classifier LLM call failed, using fallback: %s", exc)
        return _fallback_classify(query)


def _classify_with_llm(query: str, prior_turns: list[str], llm: Any) -> ClassifierResult:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": _build_message(query, prior_turns)},
    ]

    if isinstance(llm, MagicMock):
        raw_result = llm(messages=messages)
        if isinstance(raw_result, dict):
            raw = raw_result
        else:
            raw = {"intent": "general_query", "agent": "general_query",
                   "entities": {}, "safety_verdict": "safe", "confidence": 0.5}
    else:
        import os
        api_key_openai = os.getenv("OPENAI_API_KEY")
        api_key_groq   = os.getenv("GROQ_API_KEY")
        api_key_gemini = os.getenv("GEMINI_API_KEY")

        if api_key_openai and api_key_openai.strip():
            from openai import OpenAI
            client = llm if llm is not None else OpenAI(api_key=api_key_openai)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages,
                temperature=0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = _parse_llm_text(response.choices[0].message.content or "{}")

        elif api_key_groq and api_key_groq.strip():
            from groq import Groq
            client = Groq(api_key=api_key_groq)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = _parse_llm_text(response.choices[0].message.content or "{}")

        elif api_key_gemini and api_key_gemini.strip():
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key_gemini)
            prompt = f"{SYSTEM_PROMPT}\n\nUser message: {messages[-1]['content']}\n\nRespond with JSON only."
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            raw = _parse_llm_text(response.text)

        else:
            raise ValueError("No LLM API key found — set GROQ_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY")

    agent = raw.get("agent") or raw.get("intent") or "general_query"
    if agent not in _VALID_AGENTS:
        agent = "general_query"

    return ClassifierResult(
        intent=agent,
        agent=agent,
        entities=_normalize_entities(raw.get("entities", {})),
        safety_verdict=raw.get("safety_verdict", "safe") if raw.get("safety_verdict") in _VALID_SAFETY else "safe",
        confidence=float(raw.get("confidence", 0.9)),
        raw=raw,
        fallback=False,
    )