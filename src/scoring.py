"""
scoring.py

The opportunity score is deliberately a pure function over provided fields
-- no LLM in the loop -- so ranking is deterministic, explainable, and
re-runnable in CI. Each component is 0-100 before weighting, so the banker
can see exactly which signal drove a client to the top.

Composite = weighted sum of:
  - urgency        : how soon does something lapse (exception expiry, maturity)
  - balance_value   : deposit balance at risk / opportunity size
  - aus_value       : total relationship value (cross-sell headroom)
  - rate_sensitivity: how likely a nudge is needed vs. auto-retention
  - liquidity_signal: fresh, uninvested money sitting in the account
  - engagement      : prior positive/neutral reaction (response-conversion boost)

Weights are a starting point, intentionally documented so they're a config
change, not a code change (see WEIGHTS below).
"""
from __future__ import annotations

from dataclasses import dataclass

from data_access import DecisionMaker, Account, Transaction, BankerNote, TODAY

WEIGHTS = {
    "urgency": 0.30,
    "balance_value": 0.20,
    "aus_value": 0.10,
    "rate_sensitivity": 0.15,
    "liquidity_signal": 0.15,
    "engagement": 0.10,
}

RATE_SENSITIVITY_SCORE = {"HIGH": 100, "MEDIUM": 55, "LOW": 20}
TIER_SCORE = {"PLATINUM": 100, "GOLD": 75, "SILVER": 45, "BRONZE": 20}


def _urgency_from_days(days: int | None, horizon: int) -> float:
    """Closer to expiry/maturity -> higher urgency. Linear decay to 0 at horizon."""
    if days is None:
        return 0.0
    days = max(days, 0)
    return max(0.0, 100.0 * (1 - days / horizon))


def _log_scale(value: float, cap: float) -> float:
    """Diminishing-returns scaling for dollar amounts so a single $500M
    outlier doesn't blow out the 0-100 range for everyone else."""
    import math
    if value <= 0:
        return 0.0
    return min(100.0, 100.0 * math.log10(1 + value) / math.log10(1 + cap))


@dataclass
class ScoredOpportunity:
    client: DecisionMaker
    intent: str
    trigger: Account | Transaction | BankerNote
    score: float
    components: dict[str, float]
    reason: str


def _days_phrase(days: int) -> str:
    """Phrase a day-delta naturally whether it's upcoming or already past due."""
    if days < 0:
        return f"{-days}d ago"
    return f"{days}d"


def score_maturity(dm: DecisionMaker, acct: Account, horizon: int = 30, live_reaction: str | None = None) -> ScoredOpportunity:
    days = (acct.end_date - TODAY).days if acct.end_date else None
    components = {
        "urgency": _urgency_from_days(days, horizon),
        "balance_value": _log_scale(acct.balance_usd, cap=20_000_000),
        "aus_value": _log_scale(dm.total_aus_usd, cap=100_000_000),
        "rate_sensitivity": RATE_SENSITIVITY_SCORE.get(dm.rate_sensitivity, 40),
        "liquidity_signal": 0.0,
        "engagement": _engagement_score(dm, live_reaction),
    }
    total = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)
    verb = "matured" if days is not None and days < 0 else "matures in"
    reason = f"{acct.product_type} of ${acct.balance_usd:,.0f} {verb} {_days_phrase(days)}" if days is not None else f"{acct.product_type} of ${acct.balance_usd:,.0f}"
    return ScoredOpportunity(dm, "maturity", acct, round(total, 1), components, reason)


def score_exception_expiry(dm: DecisionMaker, acct: Account, horizon: int = 45, live_reaction: str | None = None) -> ScoredOpportunity:
    days = acct.days_to_exception_expiry
    components = {
        "urgency": _urgency_from_days(days, horizon),
        "balance_value": _log_scale(acct.balance_usd, cap=20_000_000),
        "aus_value": _log_scale(dm.total_aus_usd, cap=100_000_000),
        "rate_sensitivity": RATE_SENSITIVITY_SCORE.get(dm.rate_sensitivity, 40),
        "liquidity_signal": 0.0,
        "engagement": _engagement_score(dm, live_reaction),
    }
    total = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)
    verb = "expired" if days is not None and days < 0 else "expires in"
    reason = (
        f"Exception rate {acct.exception_rate}% on ${acct.balance_usd:,.0f} "
        f"{verb} {_days_phrase(days)} (reverts to {acct.rate}%)"
    )
    return ScoredOpportunity(dm, "exception_expiry", acct, round(total, 1), components, reason)


def score_liquidity_event(dm: DecisionMaker, txn: Transaction, horizon: int = 21, live_reaction: str | None = None) -> ScoredOpportunity:
    components = {
        "urgency": _urgency_from_days(txn.days_ago, horizon),
        "balance_value": _log_scale(txn.amount, cap=50_000_000),
        "aus_value": _log_scale(dm.total_aus_usd, cap=100_000_000),
        "rate_sensitivity": RATE_SENSITIVITY_SCORE.get(dm.rate_sensitivity, 40),
        "liquidity_signal": 100.0,
        "engagement": _engagement_score(dm, live_reaction),
    }
    total = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)
    reason = f"${txn.amount:,.0f} inbound via {txn.type} ({txn.liquidity_event_type}) {txn.days_ago}d ago"
    return ScoredOpportunity(dm, "liquidity_event", txn, round(total, 1), components, reason)


def score_new_account(dm: DecisionMaker, acct: Account, horizon: int = 30, live_reaction: str | None = None) -> ScoredOpportunity:
    days_open = (TODAY - acct.start_date).days if acct.start_date else horizon
    components = {
        "urgency": _urgency_from_days(days_open, horizon),
        "balance_value": _log_scale(acct.balance_usd, cap=20_000_000),
        "aus_value": _log_scale(dm.total_aus_usd, cap=100_000_000),
        "rate_sensitivity": RATE_SENSITIVITY_SCORE.get(dm.rate_sensitivity, 40),
        "liquidity_signal": 0.0,
        "engagement": _engagement_score(dm, live_reaction),
    }
    total = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)
    reason = f"New {acct.product_type} account opened {days_open}d ago -- cross-sell window"
    return ScoredOpportunity(dm, "new_account", acct, round(total, 1), components, reason)


def score_response_conversion(dm: DecisionMaker, note: BankerNote, horizon: int = 60, live_reaction: str | None = None) -> ScoredOpportunity:
    # prefer the in-app-recorded reaction (more current) over the static
    # note's reaction when one has been logged for this client
    effective_reaction = live_reaction or note.client_reaction
    components = {
        "urgency": _urgency_from_days(note.days_ago, horizon),
        "balance_value": _log_scale(dm.total_deposit_balance, cap=20_000_000),
        "aus_value": _log_scale(dm.total_aus_usd, cap=100_000_000),
        "rate_sensitivity": RATE_SENSITIVITY_SCORE.get(dm.rate_sensitivity, 40),
        "liquidity_signal": 0.0,
        "engagement": 100.0 if effective_reaction == "POSITIVE" else 60.0,
    }
    total = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)
    reason = f"{note.client_reaction} reply on {note.topic} ({note.days_ago}d ago) -- close the loop"
    return ScoredOpportunity(dm, "response_conversion", note, round(total, 1), components, reason)


def _engagement_score(dm: DecisionMaker, live_reaction: str | None = None) -> float:
    """Small boost/penalty based on tier + most recent reaction, so two
    otherwise-identical opportunities break ties toward the more valuable
    or more receptive client.

    live_reaction is the most recent reaction recorded IN-APP (memory.
    latest_reaction_value), i.e. the funnel feedback loop: send -> respond
    -> convert, recording each reaction to improve the next round. It takes
    priority over the static banker_notes.csv reaction, since it reflects
    what actually happened on THIS system's prior outreach, not historical
    mock data. Falls back to the static note only if nothing's been
    recorded live yet.
    """
    tier_component = TIER_SCORE.get(dm.tier, 40) * 0.5
    reaction = live_reaction
    if reaction is None and dm.notes:
        reaction = dm.notes[0].client_reaction
    reaction_component = {"POSITIVE": 50.0, "NEUTRAL": 25.0, "NO_RESPONSE": 5.0, "NEGATIVE": -20.0}.get(
        reaction, 15.0
    )
    return max(0.0, min(100.0, tier_component + reaction_component))
