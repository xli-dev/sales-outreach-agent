"""
routing_eval_cases.py

The labeled ask set used to score intent_router's routing accuracy --
pure data, no logic, imported by BOTH tests/test_routing_accuracy.py (the
pytest eval) and streamlit_app.py (the sidebar "Run routing eval"
diagnostic), so the two never drift into scoring against different cases.

There's no ground-truth labeled set provided (unlike the CSVs) -- this
is a hand-labeled set instead, inferring expected intents from
skills.py's own intent list and
intent_router.ROUTER_SYSTEM_PROMPT's documented guidance.
"""
from skills import all_intents

ALL_INTENTS = set(all_intents())

# (ask, expected intents). A case passes if the router's returned intent
# set exactly matches (single-intent cases) or is a superset of (the
# full-funnel case, where the router's own prompt says to return everything
# rather than guess narrowly) the expected set.
CASES = [
    ("Which clients have deposits maturing soon?", {"maturity"}),
    ("Anyone rolling off a term deposit in the next few weeks?", {"maturity"}),
    ("Who recently opened a new investment account?", {"new_account"}),
    ("Which clients just got a large inbound wire from another bank?", {"liquidity_event"}),
    ("A client sold a business recently, who has liquidity events?", {"liquidity_event"}),
    ("Which clients have exception pricing about to expire?", {"exception_expiry"}),
    ("Which clients have exception rates expiring in the next day?", {"exception_expiry"}),
    ("A client replied positively last week, help me follow up and convert it", {"response_conversion"}),
    ("Who should I reach out to this week?", ALL_INTENTS),
    ("What are my opportunities today, across everything?", ALL_INTENTS),
]


def score_case(expected: set, actual: set) -> bool:
    """A case passes on an exact match, except the full-funnel cases
    (expected == ALL_INTENTS), which pass on a superset -- the router is
    told to return everything for a vague "this week" style ask rather
    than guess narrowly, so returning extra intents there is correct,
    not a miss."""
    return actual == expected if expected != ALL_INTENTS else actual >= expected
