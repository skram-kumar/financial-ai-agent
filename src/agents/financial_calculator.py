"""
Financial Calculator Agent — deterministic math, no LLM needed.

Handles:
  - DCA (dollar cost averaging) future value
  - Mortgage payment calculation
  - Future value of lump sum
  - FX conversion (via yfinance)
  - Compound interest

No LLM call — all calculations are deterministic.
"""

from __future__ import annotations
import math
import logging
from typing import Any

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This is not financial advice. Calculations are estimates based on "
    "provided inputs and assume constant rates. Actual results may vary. "
    "Consult a qualified financial advisor before making investment decisions."
)


def _future_value_dca(
    amount: float,
    rate: float,
    period_years: int,
    frequency: str = "monthly",
) -> dict:
    """Future value of regular contributions (DCA)."""
    periods_per_year = {"daily": 365, "weekly": 52, "monthly": 12, "yearly": 1}
    n = periods_per_year.get(frequency, 12)
    total_periods = period_years * n
    rate_per_period = rate / n

    if rate_per_period == 0:
        fv = amount * total_periods
    else:
        fv = amount * ((1 + rate_per_period) ** total_periods - 1) / rate_per_period

    total_contributed = amount * total_periods
    total_interest = fv - total_contributed

    return {
        "calculation": "dca_future_value",
        "inputs": {
            "contribution": amount,
            "frequency": frequency,
            "annual_rate_pct": round(rate * 100, 2),
            "period_years": period_years,
        },
        "result": {
            "future_value": round(fv, 2),
            "total_contributed": round(total_contributed, 2),
            "total_interest_earned": round(total_interest, 2),
            "return_on_investment_pct": round((total_interest / total_contributed) * 100, 1),
        },
        "summary": (
            f"Investing {frequency} contributions of ${amount:,.0f} at {rate*100:.1f}% "
            f"annual return for {period_years} years gives a future value of "
            f"${fv:,.0f}. You contribute ${total_contributed:,.0f} and earn "
            f"${total_interest:,.0f} in returns."
        ),
    }


def _future_value_lump_sum(
    amount: float,
    rate: float,
    period_years: int,
) -> dict:
    """Future value of a one-time lump sum investment."""
    fv = amount * (1 + rate) ** period_years
    gain = fv - amount

    return {
        "calculation": "lump_sum_future_value",
        "inputs": {
            "principal": amount,
            "annual_rate_pct": round(rate * 100, 2),
            "period_years": period_years,
        },
        "result": {
            "future_value": round(fv, 2),
            "total_gain": round(gain, 2),
            "return_pct": round((gain / amount) * 100, 1),
        },
        "summary": (
            f"${amount:,.0f} invested at {rate*100:.1f}% annual return "
            f"for {period_years} years grows to ${fv:,.0f} "
            f"(a gain of ${gain:,.0f})."
        ),
    }


def _mortgage_payment(
    amount: float,
    rate: float,
    period_years: int,
) -> dict:
    """Monthly mortgage payment calculation."""
    monthly_rate = rate / 12
    n_payments = period_years * 12

    if monthly_rate == 0:
        monthly_payment = amount / n_payments
    else:
        monthly_payment = amount * (monthly_rate * (1 + monthly_rate) ** n_payments) / \
                          ((1 + monthly_rate) ** n_payments - 1)

    total_paid = monthly_payment * n_payments
    total_interest = total_paid - amount

    return {
        "calculation": "mortgage_payment",
        "inputs": {
            "loan_amount": amount,
            "annual_rate_pct": round(rate * 100, 2),
            "period_years": period_years,
        },
        "result": {
            "monthly_payment": round(monthly_payment, 2),
            "total_paid": round(total_paid, 2),
            "total_interest": round(total_interest, 2),
            "interest_to_principal_ratio": round(total_interest / amount, 2),
        },
        "summary": (
            f"A ${amount:,.0f} mortgage at {rate*100:.1f}% for {period_years} years "
            f"costs ${monthly_payment:,.2f}/month. Total paid: ${total_paid:,.0f} "
            f"(${total_interest:,.0f} in interest)."
        ),
    }


def _fx_conversion(amount: float, from_currency: str, to_currency: str = "USD") -> dict:
    """FX conversion using live rates from yfinance."""
    try:
        import yfinance as yf
        pair = f"{from_currency}{to_currency}=X"
        ticker = yf.Ticker(pair)
        hist = ticker.history(period="1d")
        if hist.empty:
            raise ValueError(f"No rate found for {pair}")
        rate = float(hist["Close"].iloc[-1])
        converted = amount * rate

        return {
            "calculation": "fx_conversion",
            "inputs": {
                "amount": amount,
                "from_currency": from_currency,
                "to_currency": to_currency,
            },
            "result": {
                "converted_amount": round(converted, 2),
                "exchange_rate": round(rate, 4),
            },
            "summary": (
                f"{amount:,.2f} {from_currency} = {converted:,.2f} {to_currency} "
                f"(rate: {rate:.4f})"
            ),
        }
    except Exception as exc:
        logger.warning("FX conversion failed: %s", exc)
        return {
            "calculation": "fx_conversion",
            "error": f"Could not fetch live rate for {from_currency}/{to_currency}",
            "summary": f"Unable to convert {from_currency} to {to_currency} at this time.",
        }


def run(entities: dict, llm: Any = None) -> dict:
    """
    Run the financial calculator agent.

    Args:
        entities: Extracted entities from classifier
                  (amount, rate, period_years, frequency, currency)
        llm:      Not used — all calculations are deterministic

    Returns:
        Calculation result dict. Never raises.
    """
    try:
        amount      = entities.get("amount")
        rate        = entities.get("rate")
        period_years = entities.get("period_years")
        frequency   = entities.get("frequency", "monthly")
        currency    = entities.get("currency")
        topics      = entities.get("topics", [])

        # Determine calculation type from entities and topics
        topics_lower = [t.lower() for t in topics]

        # FX conversion
        if currency and currency.upper() != "USD" and amount:
            result = _fx_conversion(amount, currency.upper())

        # Mortgage
        elif any(t in topics_lower for t in ["mortgage", "loan", "home loan"]):
            if not all([amount, rate, period_years]):
                return {
                    "calculation": "mortgage_payment",
                    "error": "Need loan amount, interest rate, and period in years.",
                    "summary": "Please provide the loan amount, annual interest rate, and loan term.",
                    "disclaimer": DISCLAIMER,
                }
            result = _mortgage_payment(amount, rate, period_years)

        # DCA with frequency
        elif frequency and frequency != "yearly" and amount and rate and period_years:
            result = _future_value_dca(amount, rate, period_years, frequency)

        # Lump sum future value
        elif amount and rate and period_years:
            result = _future_value_lump_sum(amount, rate, period_years)

        # Not enough info
        else:
            return {
                "calculation": "unknown",
                "error": "Not enough information to perform calculation.",
                "summary": (
                    "I need more details — please provide the amount, rate, "
                    "and time period for the calculation."
                ),
                "disclaimer": DISCLAIMER,
            }

        result["disclaimer"] = DISCLAIMER
        return result

    except Exception as exc:
        logger.error("Financial calculator error: %s", exc, exc_info=True)
        return {
            "calculation": "error",
            "error": str(exc),
            "summary": "An error occurred during calculation. Please try again.",
            "disclaimer": DISCLAIMER,
        }