"""
Safety Guard — runs before any LLM call.
Pure local computation, no network calls, must complete < 10ms.
 
Logic:
  1. Check for HARMFUL intent patterns → block immediately
  2. Check for EDUCATIONAL signals → pass immediately
  3. Default → pass (avoid over-blocking)
 
Each blocked category returns a distinct professional response.
"""
 
from __future__ import annotations
import re
import time
from dataclasses import dataclass
 
 
@dataclass
class SafetyVerdict:
    blocked: bool
    category: str | None       # e.g. "insider_trading", None if not blocked
    reason: str | None         # human-readable reason
    message: str | None       # professional refusal message if blocked
    latency_ms: float = 0.0
 
 
# ---------------------------------------------------------------------------
# Educational signal patterns — if these match, we lean toward passing.
# Used as a secondary check to avoid over-blocking legitimate questions.
# ---------------------------------------------------------------------------
EDUCATIONAL_SIGNALS = re.compile(
    r"\b("
    r"what is|what are|what does|what's the|"
    r"how does|how do regulators|how is|how was|"
    r"explain|describe|define|definition of|"
    r"why is|why are|why would|"
    r"history of|historical|"
    r"penalty|penalties|punishment|illegal|legal\?|lawful|"
    r"regulation|regulations|regulatory|compliance|"
    r"fca|sec|ofac|aml|kyc|"
    r"detect|investigation|investigate|caught|catch|"
    r"difference between|distinguish|compare|"
    r"red flag|warning sign|risk of|risks of|"
    r"average return|historical return|"
    r"how.*works?|how.*scheme works?|"
    r"is it (ever |)legitimate|"
    r"what factors|considerations?|should.*consider|"
    r"obligations?|screening|"
    r"how should|appropriate allocation"
    r")\b",
    re.IGNORECASE,
)
 
# Strong educational overrides — these phrases are unambiguously educational
# even if they trigger a harmful pattern. No tiebreaker needed.
STRONG_EDUCATIONAL_OVERRIDES = re.compile(
    r"("
    r"is it (ever |)legitimate to"
    r"|what factors should"
    r"|what (compliance |)obligations"
    r"|how should .{0,30} (be allocated|consider|approach)"
    r"|what.*consider when (allocating|investing|trading)"
    r")",
    re.IGNORECASE,
)
 
# ---------------------------------------------------------------------------
# Harmful intent patterns, per category
# ---------------------------------------------------------------------------
 
# Pattern: first-person or imperative + non-public info + action
_INSIDER_HARMFUL = re.compile(
    r"("
    # Direct access to MNPI + intent to trade
    r"(i (work|am employed)|my (friend|colleague|contact|boss|lawyer|attorney)).*"
    r"(tip|told me|gave me|know about|confidential|unannounced|merger|acquisition|earnings).*"
    r"(buy|sell|load|trade|invest|shares|stock)"
    r"|"
    # Explicit MNPI trading requests
    r"(trade|invest|buy|sell|load up).*(confidential|non.?public|insider|mnpi|merger|acquisition)"
    r"|"
    r"(confidential|merger|acquisition|unannounced).*(trade|invest|buy|sell|shares|stock)"
    r"|"
    # Need earnings before announcement
    r"(need|want|get).*(earnings|results|numbers).*(before|prior to).*(announcement|release|call|report)"
    r"|"
    r"i know about an? (unannounced|confidential|secret|private)"
    r")",
    re.IGNORECASE | re.DOTALL,
)
 
_MARKET_MANIP_HARMFUL = re.compile(
    r"("
    r"(pump|pump up|inflate|boost).*(price|stock|share|volume)"
    r"|"
    r"(coordinated|coordinate).*(buy|buying|scheme|campaign).*(move|push|lift).*(stock|price|percent|%)"
    r"|"
    r"(wash trade|wash.?trading).*(between|two|my|accounts?)"
    r"|"
    r"(help me|i want to|how (do i|can i)).*(wash trade|manipulate|coordinate.*buy)"
    r"|"
    r"move this stock \d"
    r")",
    re.IGNORECASE,
)
 
_MONEY_LAUNDERING_HARMFUL = re.compile(
    r"("
    r"(move|transfer|send|route).*(cash|funds?|money).*(without report|without trigger|avoid report)"
    r"|"
    r"(structure|structuring).*(deposit|withdraw|payment).*(avoid|under|below).*(threshold|limit|report)"
    r"|"
    r"(layer|layering|obscure|hide).*(trades?|funds?|money|profits?|source)"
    r"|"
    r"(avoid|evade|hide).*(tax|reporting|authorities|irs|hmrc)"
    r"|"
    r"(shell company|nominee).*(bypass|evade|avoid|hide)"
    r")",
    re.IGNORECASE,
)
 
_GUARANTEED_RETURNS_HARMFUL = re.compile(
    r"("
    # Requests for guarantees
    r"(guarantee|guaranteed|promise|certain|100%).*(return|profit|gain|double|triple)"
    r"|"
    r"(return|profit|gain).*(guarantee|guaranteed|promise|certain|100%)"
    r"|"
    # Specific foolproof / certain gain phrasing
    r"(foolproof|fool.proof|risk.free|no.risk|zero.risk).*(make|earn|return|profit|gain)"
    r"|"
    r"(stock|investment).*(100%|certain|definitely|for sure).*(go up|rise|increase|profit)"
    r"|"
    # Double / triple in unrealistic time frame
    r"(double|triple|10x|100x).*(in a (month|week|day|year))"
    r"|"
    r"(make|earn|return|gain) \d{2,3}%.*(in a (month|week|day))"
    r")",
    re.IGNORECASE,
)
 
_RECKLESS_ADVICE_HARMFUL = re.compile(
    r"("
    # All-in on volatile assets for vulnerable people
    r"(all|everything|entire|whole).*(retirement|pension|savings|life savings|emergency fund).*(crypto|bitcoin|options?|derivatives?|single stock|margin)"
    r"|"
    r"(70|75|80|retired|retirement).*(all|everything|entire).*(crypto|bitcoin|options?)"
    r"|"
    # Leverage on single positions
    r"(margin loan|leverage).*(buy more|all in|single stock)"
    r"|"
    r"(mortgage|remortgage|home equity|heloc).*(stock|invest|trade|buy)"
    r"|"
    # Emergency fund into high-risk
    r"(emergency fund|emergency savings).*(options?|crypto|bitcoin|margin|derivatives?)"
    r"|"
    # Explicit "tell me to do reckless thing"
    r"tell me (to take|to put|to invest|to buy|which stock to)"
    r")",
    re.IGNORECASE,
)
 
_SANCTIONS_EVASION_HARMFUL = re.compile(
    r"("
    r"(bypass|evade|circumvent|avoid).*(sanction|ofac|embargo)"
    r"|"
    r"(sanction|sanctioned).*(company|entity|person|country).*(invest|trade|buy|route|without.*trace)"
    r"|"
    r"(route|hide|obscure).*(trades?|funds?|money).*(sanction|ofac|embargo)"
    r"|"
    r"(shell company|offshore|nominee).*(sanction|ofac)"
    r")",
    re.IGNORECASE,
)
 
_FRAUD_HARMFUL = re.compile(
    r"("
    r"(fake|forged?|fabricate|falsify|counterfeit).*(document|contract|statement|note|record|report)"
    r"|"
    r"(draft|create|make).*(fake|false|fraudulent).*(contract|document|statement|loss|gain)"
    r")",
    re.IGNORECASE,
)
 
# ---------------------------------------------------------------------------
# Category registry: (pattern, category_key)
# Order matters — more specific patterns first
# ---------------------------------------------------------------------------
HARMFUL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (_INSIDER_HARMFUL,          "insider_trading"),
    (_MARKET_MANIP_HARMFUL,     "market_manipulation"),
    (_MONEY_LAUNDERING_HARMFUL, "money_laundering"),
    (_GUARANTEED_RETURNS_HARMFUL,"guaranteed_returns"),
    (_RECKLESS_ADVICE_HARMFUL,  "reckless_advice"),
    (_SANCTIONS_EVASION_HARMFUL,"sanctions_evasion"),
    (_FRAUD_HARMFUL,            "fraud"),
]
 
# ---------------------------------------------------------------------------
# Professional refusal messages — one per category, distinct tone
# ---------------------------------------------------------------------------
REFUSAL_MESSAGES: dict[str, str] = {
    "insider_trading": (
        "I'm not able to assist with trading on material non-public information. "
        "Acting on insider knowledge is a serious criminal offence in most jurisdictions — "
        "it carries significant fines and imprisonment, and regulators actively monitor for it. "
        "If you have questions about what constitutes permissible trading or how disclosure rules work, "
        "I'm happy to help with that."
    ),
    "market_manipulation": (
        "I can't help with coordinating trades, inflating prices, or any form of market manipulation. "
        "These activities are illegal under securities law globally and carry severe criminal penalties. "
        "If you're interested in understanding how manipulation is detected or regulated, "
        "I can assist with that instead."
    ),
    "money_laundering": (
        "I'm unable to assist with structuring transactions to avoid reporting requirements, "
        "layering trades to obscure fund origins, or any activity that constitutes money laundering. "
        "These are serious financial crimes. If you'd like to understand AML regulations or "
        "compliance obligations, I can help with that."
    ),
    "guaranteed_returns": (
        "No investment can guarantee returns — any claim to the contrary is a hallmark of fraud. "
        "I won't provide advice framed around guaranteed profits or certain gains. "
        "If you'd like to understand realistic return expectations, historical market data, "
        "or how to evaluate investment risk, I'm here to help."
    ),
    "reckless_advice": (
        "I can't recommend actions that would put your financial security at serious risk — "
        "such as placing retirement savings entirely in highly volatile assets or using leverage "
        "on concentrated positions. My role is to help you make informed, appropriate decisions. "
        "If you'd like to understand the risks involved or explore more suitable strategies, "
        "I'm happy to do that."
    ),
    "sanctions_evasion": (
        "I'm not able to help with circumventing sanctions, OFAC restrictions, or other "
        "regulatory embargoes. Sanctions evasion is a federal crime with severe consequences. "
        "If you have questions about what sanctions apply or how compliance screening works, "
        "I can help with that."
    ),
    "fraud": (
        "I can't assist with creating false, fake, or fraudulent financial documents or records. "
        "Document fraud is a criminal offence. If you need help with legitimate record-keeping "
        "or understanding financial reporting requirements, I'm here to help."
    ),
}
 
 
def check(query: str) -> SafetyVerdict:
    """
    Run the safety guard on a user query.
    Returns a SafetyVerdict in well under 10ms for any input.
    """
    start = time.perf_counter()
 
    # Step 1: Check for strong educational overrides first.
    # Phrases like "is it ever legitimate to..." or "what factors should..."
    # are unambiguously educational — bypass all harmful pattern checks.
    if STRONG_EDUCATIONAL_OVERRIDES.search(query):
        latency_ms = (time.perf_counter() - start) * 1000
        return SafetyVerdict(blocked=False, category=None, reason=None,
                             message=None, latency_ms=latency_ms)
 
    # Step 2: Check for weaker educational signals.
    # If present, we still run harmful checks but use a tiebreaker.
    is_educational = bool(EDUCATIONAL_SIGNALS.search(query))
 
    # Step 2: Run all harmful patterns
    matched_category = None
    for pattern, category in HARMFUL_PATTERNS:
        if pattern.search(query):
            matched_category = category
            break
 
    latency_ms = (time.perf_counter() - start) * 1000
 
    if matched_category and not is_educational:
        # Clearly harmful — block
        return SafetyVerdict(
            blocked=True,
            category=matched_category,
            reason=f"Matched harmful pattern: {matched_category}",
            message=REFUSAL_MESSAGES[matched_category],
            latency_ms=latency_ms,
        )
 
    if matched_category and is_educational:
        # Edge case: query has both educational signals AND harmful patterns.
        # Example: "explain how to pump a stock and sell it" — educational phrasing
        # but clearly instructional for harm.
        # We apply a tiebreaker: if the educational signal appears BEFORE the
        # harmful signal, it's likely a legit question. Otherwise block.
        edu_match = EDUCATIONAL_SIGNALS.search(query)
        harm_match = next(
            (p.search(query) for p, c in HARMFUL_PATTERNS if p.search(query)), None
        )
        if harm_match and edu_match:
            if harm_match.start() < edu_match.start():
                # Harmful intent leads — block
                return SafetyVerdict(
                    blocked=True,
                    category=matched_category,
                    reason=f"Harmful intent precedes educational framing: {matched_category}",
                    message=REFUSAL_MESSAGES[matched_category],
                    latency_ms=latency_ms,
                )
 
    # Default: pass
    return SafetyVerdict(
        blocked=False,
        category=None,
        reason=None,
        message=None,
        latency_ms=latency_ms,
    )
 