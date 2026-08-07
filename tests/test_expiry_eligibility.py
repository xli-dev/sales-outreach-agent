"""
clients_with_exception_expiring and clients_with_maturity_in_window used to
require `0 <= delta <= days`, which silently dropped any account whose
end_date had already passed (delta < 0) from the eligibility pool -- even
though scoring._urgency_from_days clamps negative days to 0 and treats that
as *maximum* urgency. An unrenewed exception or a matured TD that nobody
had gotten to yet would just vanish from ranking instead of surfacing as
the most urgent item. This locks in the fix: already-past-due accounts
must remain eligible (bounded only above, by the horizon).
"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_access
import scoring
from data_access import Account, DataStore, DecisionMaker


def _account(**overrides) -> Account:
    defaults = dict(
        account_id="A1", client_eci="TEST123", product_type="TD",
        account_class="RETAIL", platform="CORE", balance_usd=1_000_000.0,
        currency="USD", rate=2.0, libor_rate=None,
        is_exception_priced=False, exception_rate=None,
        start_date=dt.date(2025, 1, 1), end_date=None,
        last_exception_request_id=None, proposed_rate=None,
        request_reason_type=None, competitor_rate=None,
        competitor_product=None, justification_comment=None,
        decision_date=None,
    )
    defaults.update(overrides)
    return Account(**defaults)


def _store_with(account: Account) -> DataStore:
    store = object.__new__(DataStore)
    dm = DecisionMaker(
        client_eci="TEST123", name="Test Client", segment="PRIVATE",
        tier="1", total_aus_usd=5_000_000.0, rate_sensitivity="MEDIUM",
        banker_id="BKR0001", banker_name="Test Banker", team="T1", region="R1",
    )
    dm.accounts.append(account)
    store._dms = {"TEST123": dm}
    return store


def test_already_past_due_maturity_is_still_eligible():
    past_due = _account(product_type="TD", end_date=data_access.TODAY - dt.timedelta(days=10))
    store = _store_with(past_due)
    results = store.clients_with_maturity_in_window(days=30)
    assert len(results) == 1


def test_already_expired_exception_is_still_eligible():
    expired = _account(is_exception_priced=True, exception_rate=1.5, end_date=data_access.TODAY - dt.timedelta(days=10))
    store = _store_with(expired)
    results = store.clients_with_exception_expiring(days=45)
    assert len(results) == 1


def test_past_due_scores_as_max_urgency_not_zero():
    expired = _account(is_exception_priced=True, exception_rate=1.5, end_date=data_access.TODAY - dt.timedelta(days=10))
    store = _store_with(expired)
    dm, acct = store.clients_with_exception_expiring(days=45)[0]
    opp = scoring.score_exception_expiry(dm, acct, horizon=45)
    assert opp.components["urgency"] == 100.0
    assert "ago" in opp.reason


def test_far_beyond_horizon_still_excluded():
    """The upper bound must still work -- this isn't a blanket removal of
    the window, just of the incorrect lower bound."""
    far_out = _account(product_type="TD", end_date=data_access.TODAY + dt.timedelta(days=365))
    store = _store_with(far_out)
    results = store.clients_with_maturity_in_window(days=30)
    assert len(results) == 0
