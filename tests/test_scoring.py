"""
Smoke tests: data loads and joins correctly, scoring returns sane bounded
values, and every intent's fetch path returns something on the real mock
dataset. Run in CI so a bad data-schema assumption fails the pipeline
before it fails at runtime.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_access
import scoring


def test_data_loads():
    store = data_access.get_store()
    clients = store.all_clients()
    assert len(clients) == 100
    assert all(c.client_eci for c in clients)


def test_accounts_joined_to_clients():
    store = data_access.get_store()
    total_accounts = sum(len(c.accounts) for c in store.all_clients())
    assert total_accounts == 164


def test_scores_are_bounded():
    store = data_access.get_store()
    for dm, acct in store.clients_with_exception_expiring(days=365):
        opp = scoring.score_exception_expiry(dm, acct, horizon=365)
        assert 0 <= opp.score <= 100
        for v in opp.components.values():
            assert 0 <= v <= 100


def test_live_reaction_changes_engagement_score():
    """Regression test: a reaction recorded via the app (memory.reactions)
    must actually move the score on the next ranking pass -- this is the
    'recording each reaction to improve the next round' requirement, and it
    was silently a no-op before scoring.py accepted live_reaction."""
    store = data_access.get_store()
    dm, acct = store.clients_with_exception_expiring(days=3650)[0]

    baseline = scoring.score_exception_expiry(dm, acct, horizon=365, live_reaction=None)
    positive = scoring.score_exception_expiry(dm, acct, horizon=365, live_reaction="POSITIVE")
    negative = scoring.score_exception_expiry(dm, acct, horizon=365, live_reaction="NEGATIVE")

    assert positive.components["engagement"] > negative.components["engagement"]
    assert positive.score != baseline.score or negative.score != baseline.score


def test_each_intent_returns_candidates_on_full_dataset():
    """With a wide enough horizon, every intent should find at least one
    candidate in the mock dataset -- guards against a schema/field typo
    silently making an intent always-empty."""
    store = data_access.get_store()
    assert len(store.clients_with_maturity_in_window(days=3650)) > 0
    assert len(store.clients_with_exception_expiring(days=3650)) > 0
    assert len(store.clients_with_liquidity_event(days=3650)) > 0
    assert len(store.clients_with_new_account(days=3650)) > 0
    assert len(store.clients_with_open_response(days=3650)) > 0


def test_sixth_skill_works_via_registry_alone():
    """Adding a capability should be configuration, not a new code path.
    Prove it: register a 6th skill at runtime using only
    existing data_access/scoring primitives and confirm mcp_server's
    get_clients_by_intent handles it with zero changes to its dispatch
    logic (it's driven entirely by skill.query_fn/score_fn)."""
    import skills
    import mcp_server
    import json as _json

    skills.SKILL_REGISTRY["_test_only_sixth_skill"] = skills.Skill(
        intent="_test_only_sixth_skill",
        description="test-only",
        fetch_tools=["get_clients_by_intent"],
        default_channel="email",
        horizon_days=9999,
        query_fn=lambda store, horizon: [
            (dm, dm.accounts[0]) for dm in store.all_clients() if dm.accounts
        ][:5],
        score_fn=scoring.score_maturity,
        instruction="test",
    )
    try:
        result = _json.loads(mcp_server.get_clients_by_intent("_test_only_sixth_skill"))
        assert isinstance(result, list) and len(result) > 0
    finally:
        del skills.SKILL_REGISTRY["_test_only_sixth_skill"]
