"""
skills.py

The skill registry. A "skill" is data, not code: a trigger intent, a set
of tool bindings the harness will call (in order), a drafting instruction
template, and a default channel. Adding capability #6 ("rate-review
courtesy check-in", say) means appending an entry here -- the harness,
router, and MCP tools don't change.

Kept as a Python dict of dataclasses (rather than YAML) so it's still
one file to read end-to-end, but nothing stops you from serializing
this to YAML/JSON and loading it at runtime -- same shape either way.
Each skill's `instruction` doubles as the MCP *prompt* registered for that
skill (see mcp_server.py), so the same text is what's inspectable over MCP
and what actually drives the draft.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from data_access import DataStore
from scoring import (
    ScoredOpportunity,
    score_maturity,
    score_exception_expiry,
    score_liquidity_event,
    score_new_account,
    score_response_conversion,
)


@dataclass
class Skill:
    intent: str
    description: str
    fetch_tools: list[str]           # MCP tool names the harness calls to gather candidates
    default_channel: str             # email | teams | call_brief | meeting_notes -- default only;
                                      # the banker can override per-request or per-item (see harness.py)
    instruction: str                 # drafting instruction template (also an MCP prompt)
    horizon_days: int                # lookback/lookahead window for this intent
    query_fn: Callable[[DataStore, int], list]   # (store, horizon) -> [(DecisionMaker, trigger), ...]
    score_fn: Callable[..., ScoredOpportunity]    # (dm, trigger, horizon, live_reaction=...) -> ScoredOpportunity


SKILL_REGISTRY: dict[str, Skill] = {
    "maturity": Skill(
        intent="maturity",
        description="Deposits maturing soon; draft retention outreach before roll-off.",
        fetch_tools=["get_clients_by_intent", "get_client_profile", "get_banker_notes"],
        default_channel="email",
        horizon_days=30,
        query_fn=lambda store, horizon: store.clients_with_maturity_in_window(horizon),
        score_fn=score_maturity,
        instruction=(
            "This client's {product_type} account (${balance_usd:,.0f}) matures on "
            "{end_date}, currently earning {rate}%. Goal: retention -- prompt a "
            "renewal conversation before the funds roll off or leave the bank. "
            "Reference the current rate, grounded in the account data, and use the "
            "banker notes for relationship/tone context. Offer to discuss renewal "
            "options. Do not state or imply any rate that is not present in the "
            "account data."
        ),
    ),
    "new_account": Skill(
        intent="new_account",
        description="Client recently opened an account; cross-sell deposit products.",
        fetch_tools=["get_clients_by_intent", "get_client_profile", "get_banker_notes"],
        default_channel="email",
        horizon_days=30,
        query_fn=lambda store, horizon: store.clients_with_new_account(horizon),
        score_fn=score_new_account,
        instruction=(
            "This client opened a {product_type} account {days_open} days ago. "
            "Goal: a light-touch cross-sell -- suggest one complementary deposit "
            "product based on their segment and rate sensitivity, without being "
            "salesy. Ground any figures strictly in provided data."
        ),
    ),
    "liquidity_event": Skill(
        intent="liquidity_event",
        description="New money arrived (wire, IPO proceeds, asset sale); invite it into deposits.",
        fetch_tools=["get_clients_by_intent", "get_client_profile", "get_banker_notes"],
        default_channel="call_brief",
        horizon_days=21,
        query_fn=lambda store, horizon: store.clients_with_liquidity_event(horizon),
        score_fn=score_liquidity_event,
        instruction=(
            "This client had a recent liquidity event (${amount:,.0f} via {txn_type}, "
            "{liquidity_event_type}). Goal: invite the funds into a deposit before "
            "they leave -- summarize the event, suggest an opening line, and list "
            "2-3 talking points grounded in banker notes and account context."
        ),
    ),
    "exception_expiry": Skill(
        intent="exception_expiry",
        description="An exception-priced rate is expiring; reach out to renew before it reverts.",
        fetch_tools=["get_clients_by_intent", "get_client_profile", "get_banker_notes"],
        default_channel="teams",
        horizon_days=45,
        query_fn=lambda store, horizon: store.clients_with_exception_expiring(horizon),
        score_fn=score_exception_expiry,
        instruction=(
            "This client's exception-priced rate ({exception_rate}%) on "
            "${balance_usd:,.0f} expires on {end_date} and will revert to {rate}% if "
            "not renewed. Goal: flag this and recommend whether to propose renewal, "
            "and at what rate. Note the original justification on file and any "
            "competitor rate on record, using only rate figures present in the "
            "account record (current, exception, competitor, proposed)."
        ),
    ),
    "response_conversion": Skill(
        intent="response_conversion",
        description="A prior nudge got a reply; convert it into a booked deposit.",
        fetch_tools=["get_clients_by_intent", "get_client_profile", "get_banker_notes"],
        default_channel="call_brief",
        horizon_days=60,
        query_fn=lambda store, horizon: store.clients_with_open_response(horizon),
        score_fn=score_response_conversion,
        instruction=(
            "The client (or banker) responded {reaction} on the topic of {topic} "
            "{days_ago} days ago. Goal: close the loop -- a reply plus a short "
            "plan with 2-3 talking points, grounded in the most recent banker "
            "notes, account pricing, and the client's stated reaction. Do not "
            "invent any commitment or rate not already on record."
        ),
    ),
}


def get_skill(intent: str) -> Skill | None:
    return SKILL_REGISTRY.get(intent)


def all_intents() -> list[str]:
    return list(SKILL_REGISTRY.keys())
