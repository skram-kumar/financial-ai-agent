"""
Router — takes a ClassifierResult and dispatches to the right agent.
Fully implemented agents:
  - portfolio_health → src.agents.portfolio_health.run()
  - financial_calculator → src.agents.financial_calculator.run()
All other agents return a structured "not implemented" stub response.
The router never crashes — unknown agents get the stub too.
"""
from __future__ import annotations
import logging
from typing import Any

from src.classifier import ClassifierResult
from src.agents.portfolio_health import run as run_portfolio_health
from src.agents.financial_calculator import run as run_financial_calculator
from src.agents.market_research import run as run_market_research

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stub response for unimplemented agents
# ---------------------------------------------------------------------------

def _stub_response(result: ClassifierResult) -> dict:
    """
    Structured 'not implemented' response.
    Includes classified intent and entities so the caller knows what was detected.
    """
    return {
        "status":  "not_implemented",
        "agent":   result.agent,
        "intent":  result.intent,
        "entities": result.entities,
        "message": (
            f"The '{result.agent}' agent is not yet implemented in this build. "
            f"Intent was classified correctly and would be handled by this agent "
            f"in the full system."
        ),
        "disclaimer": (
            "This is not investment advice. For informational purposes only."
        ),
    }


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

IMPLEMENTED_AGENTS = {"portfolio_health", "financial_calculator", "market_research"}

STUB_AGENTS = {
    "market_research",
    "investment_strategy",
    "financial_planning",
    "financial_calculator",
    "risk_assessment",
    "product_recommendation",
    "predictive_analysis",
    "customer_support",
    "general_query",
}


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

def route(
    result: ClassifierResult,
    user: dict,
    llm: Any = None,
) -> dict:
    """
    Route a classified query to the appropriate agent.

    Args:
        result:  ClassifierResult from the classifier.
        user:    User profile dict (portfolio, risk profile, etc.)
        llm:     LLM client passed through to agents.

    Returns:
        Agent response dict. Never raises.
    """
    agent = result.agent
    logger.info("Routing to agent: %s (confidence: %.2f)", agent, result.confidence)

    try:
        if agent == "portfolio_health":
            response = run_portfolio_health(user=user, llm=llm)
            # Attach routing metadata
            response["_meta"] = {
                "agent":          agent,
                "intent":         result.intent,
                "entities":       result.entities,
                "safety_verdict": result.safety_verdict,
                "confidence":     result.confidence,
                "fallback_used":  result.fallback,
            }
            return response
        
        elif agent == "financial_calculator":
            response = run_financial_calculator(
                entities=result.entities,
                llm=llm,
            )
            response["_meta"] = {
                "agent": agent,
                "intent": result.intent,
                "entities": result.entities,
                "confidence": result.confidence,
            }
            return response
        
        elif agent == "market_research":
            response = run_market_research(
                entities=result.entities,
                llm=llm,
            )
            response["_meta"] = {
                "agent": agent,
                "intent": result.intent,
                "entities": result.entities,
                "confidence": result.confidence,
            }
            return response

        # All other agents → structured stub
        return _stub_response(result)

    except Exception as exc:
        logger.error("Router error for agent '%s': %s", agent, exc, exc_info=True)
        return {
            "status":  "error",
            "agent":   agent,
            "message": "An unexpected error occurred. Please try again.",
            "disclaimer": "This is not investment advice. For informational purposes only.",
        }