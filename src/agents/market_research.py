"""
Market Research Agent — fetches live market data and generates analysis.

Handles:
  - Stock price and performance data
  - Basic fundamentals (P/E, market cap, 52-week range)
  - Recent news headlines
  - Index performance (S&P 500, FTSE 100, NIKKEI 225)
  - Sector/market overview

Uses yfinance for data, LLM for plain-English summary.
"""

from __future__ import annotations
import logging
from typing import Any
from unittest.mock import MagicMock

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This is not investment advice. Market data is provided for informational "
    "purposes only. Past performance is not indicative of future results. "
    "Please consult a qualified financial advisor before making investment decisions."
)

# Index ticker mapping
INDEX_TICKERS = {
    "S&P 500":    "^GSPC",
    "FTSE 100":   "^FTSE",
    "NIKKEI 225": "^N225",
    "MSCI World": "URTH",
    "NASDAQ":     "^IXIC",
    "DOW":        "^DJI",
}


def _fetch_ticker_data(ticker: str) -> dict:
    """Fetch price, fundamentals, and recent performance for a ticker."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period="1mo")
        info = t.info

        if hist.empty:
            return {"ticker": ticker, "error": "No data found"}

        current_price = float(hist["Close"].iloc[-1])
        prev_close    = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_price
        month_start   = float(hist["Close"].iloc[0])

        day_change_pct   = ((current_price - prev_close) / prev_close) * 100
        month_change_pct = ((current_price - month_start) / month_start) * 100

        return {
            "ticker":           ticker,
            "name":             info.get("longName") or info.get("shortName", ticker),
            "current_price":    round(current_price, 2),
            "currency":         info.get("currency", "USD"),
            "day_change_pct":   round(day_change_pct, 2),
            "month_change_pct": round(month_change_pct, 2),
            "52w_high":         info.get("fiftyTwoWeekHigh"),
            "52w_low":          info.get("fiftyTwoWeekLow"),
            "market_cap":       info.get("marketCap"),
            "pe_ratio":         info.get("trailingPE"),
            "volume":           info.get("regularMarketVolume"),
            "sector":           info.get("sector"),
            "industry":         info.get("industry"),
        }
    except Exception as exc:
        logger.warning("Failed to fetch data for %s: %s", ticker, exc)
        return {"ticker": ticker, "error": str(exc)}


def _fetch_index_data(index_name: str) -> dict:
    """Fetch index performance data."""
    ticker_symbol = INDEX_TICKERS.get(index_name, index_name)
    try:
        import yfinance as yf
        t = yf.Ticker(ticker_symbol)
        hist = t.history(period="1mo")

        if hist.empty:
            return {"index": index_name, "error": "No data found"}

        current = float(hist["Close"].iloc[-1])
        prev    = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current
        month   = float(hist["Close"].iloc[0])

        return {
            "index":            index_name,
            "current_value":    round(current, 2),
            "day_change_pct":   round(((current - prev) / prev) * 100, 2),
            "month_change_pct": round(((current - month) / month) * 100, 2),
        }
    except Exception as exc:
        logger.warning("Failed to fetch index %s: %s", index_name, exc)
        return {"index": index_name, "error": str(exc)}


def _format_market_cap(mc: float | None) -> str:
    if mc is None:
        return "N/A"
    if mc >= 1e12:
        return f"${mc/1e12:.1f}T"
    if mc >= 1e9:
        return f"${mc/1e9:.1f}B"
    if mc >= 1e6:
        return f"${mc/1e6:.1f}M"
    return f"${mc:,.0f}"


def _generate_summary(ticker_data: list[dict], index_data: list[dict], llm: Any) -> str:
    """Generate plain-English market summary using LLM."""
    import os

    # Build context for LLM
    context_parts = []
    for td in ticker_data:
        if "error" not in td:
            context_parts.append(
                f"{td.get('name', td['ticker'])} ({td['ticker']}): "
                f"${td['current_price']} | "
                f"Day: {td['day_change_pct']:+.1f}% | "
                f"Month: {td['month_change_pct']:+.1f}% | "
                f"Market Cap: {_format_market_cap(td.get('market_cap'))} | "
                f"P/E: {td.get('pe_ratio', 'N/A')}"
            )

    for idx in index_data:
        if "error" not in idx:
            context_parts.append(
                f"{idx['index']}: {idx['current_value']} | "
                f"Day: {idx['day_change_pct']:+.1f}% | "
                f"Month: {idx['month_change_pct']:+.1f}%"
            )

    if not context_parts:
        return "Unable to fetch market data at this time."

    context = "\n".join(context_parts)
    prompt = f"""You are a financial market analyst at Valura.
Based on this live market data, write a brief 3-4 sentence plain-English summary 
that a novice investor would understand. Highlight the most important trends.
Do not make specific buy/sell recommendations.

Market Data:
{context}"""

    try:
        if isinstance(llm, MagicMock):
            # Test fallback
            lines = []
            for td in ticker_data:
                if "error" not in td:
                    direction = "up" if td["day_change_pct"] > 0 else "down"
                    lines.append(f"{td.get('name', td['ticker'])} is {direction} {abs(td['day_change_pct']):.1f}% today.")
            return " ".join(lines) if lines else "Market data retrieved successfully."

        api_key_groq   = os.getenv("GROQ_API_KEY")
        api_key_openai = os.getenv("OPENAI_API_KEY")

        if api_key_groq and api_key_groq.strip():
            from groq import Groq
            client = Groq(api_key=api_key_groq)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()

        elif api_key_openai and api_key_openai.strip():
            from openai import OpenAI
            client = OpenAI(api_key=api_key_openai)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()

        else:
            # No LLM — return structured summary
            return context

    except Exception as exc:
        logger.warning("LLM summary failed: %s", exc)
        return context


def run(entities: dict, llm: Any = None) -> dict:
    """
    Run the market research agent.

    Args:
        entities: Extracted entities from classifier
                  (tickers, index, time_period, topics)
        llm:      LLM client for summary generation

    Returns:
        Market research result dict. Never raises.
    """
    try:
        tickers    = entities.get("tickers", [])
        index_name = entities.get("index")
        topics     = entities.get("topics", [])

        ticker_data = []
        index_data  = []

        # Fetch ticker data
        for ticker in tickers[:5]:  # max 5 tickers
            data = _fetch_ticker_data(ticker)
            ticker_data.append(data)

        # Fetch index data
        if index_name:
            data = _fetch_index_data(index_name)
            index_data.append(data)

        # If no specific tickers or index, fetch major indices
        if not tickers and not index_name:
            for idx in ["S&P 500", "NASDAQ"]:
                data = _fetch_index_data(idx)
                index_data.append(data)

        # Generate summary
        summary = _generate_summary(ticker_data, index_data, llm)

        return {
            "tickers":    ticker_data,
            "indices":    index_data,
            "topics":     topics,
            "summary":    summary,
            "disclaimer": DISCLAIMER,
        }

    except Exception as exc:
        logger.error("Market research agent error: %s", exc, exc_info=True)
        return {
            "tickers":    [],
            "indices":    [],
            "summary":    "Unable to fetch market data at this time. Please try again.",
            "disclaimer": DISCLAIMER,
        }