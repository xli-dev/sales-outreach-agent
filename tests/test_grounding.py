"""
Tests for the hallucination guard (_grounding_check in mcp_server.py). This
is the core "no invented client facts" safety mechanism -- it needs to
actually catch hallucinated figures, and needs to NOT flood the banker with
false positives on harmless prose, or they'll learn to ignore the warning.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcp_server import _grounding_check

PROFILE = {
    "name": "Test Client",
    "total_aus_usd": 12500000.50,
    "accounts": [
        {
            "balance_usd": 1659581.55, "rate": 4.33, "exception_rate": 4.38,
            "competitor_rate": 4.5, "end_date": "2026-09-15",
        },
    ],
}


def test_grounded_dollar_figure_not_flagged():
    draft = "Your balance of $1,659,581.55 is earning a competitive rate."
    assert _grounding_check(draft, PROFILE) is None


def test_grounded_rate_not_flagged():
    draft = "Your current rate of 4.33% could be improved."
    assert _grounding_check(draft, PROFILE) is None


def test_hallucinated_multi_digit_rate_is_flagged():
    draft = "We could offer you an exceptional 12.5% promotional rate."
    result = _grounding_check(draft, PROFILE)
    assert result is not None
    assert "12.5%" in str(result)


def test_hallucinated_single_digit_rate_is_flagged():
    """Regression test for the specific bug found in review: a single-digit
    invented rate (e.g. a fabricated '9%') used to slip through unflagged
    because the old filter exempted anything <=1 digit after stripping
    $/%. Proven broken with a concrete example before this fix."""
    draft = "We could offer you a special 9% rate on your account."
    result = _grounding_check(draft, PROFILE)
    assert result is not None
    assert "9%" in str(result)


def test_hallucinated_dollar_figure_is_flagged():
    draft = "You could deposit an additional $500,000 to unlock this tier."
    result = _grounding_check(draft, PROFILE)
    assert result is not None
    assert "$500,000" in str(result)


def test_harmless_bare_numbers_not_flagged():
    """Bare counts with no $/% marker (list items, talking-point counts)
    must not trip the guard -- that's noise that would train bankers to
    ignore real warnings."""
    draft = "Here are 3 talking points to cover across your 2 accounts."
    assert _grounding_check(draft, PROFILE) is None


def test_no_numbers_at_all_not_flagged():
    draft = "It was great catching up with you recently."
    assert _grounding_check(draft, PROFILE) is None


def test_dollar_figure_rounded_to_whole_dollar_not_flagged():
    """Regression test for a real false-positive found by running actual
    drafts through this checker (test_draft_hallucination_eval.py): a
    correct balance of $1,659,581.55 written in prose as "$1,659,581" (the
    normal, reasonable way to state a balance without cents) used to fail
    exact-string matching and get flagged as unverified -- it's the same
    number, just rounded, not a hallucination."""
    draft = "Your balance of $1,659,581 is earning a competitive rate."
    assert _grounding_check(draft, PROFILE) is None


def test_percent_with_trailing_zero_not_flagged():
    """Regression test: a correct rate of 4.33% written as "4.330%" (or a
    stored rate like 4.1 written as "4.10%") is the same number with
    different formatting, not a hallucination -- used to fail exact-string
    matching."""
    draft = "Your current rate of 4.330% could be improved."
    assert _grounding_check(draft, PROFILE) is None


def test_dollar_figure_rounded_too_coarsely_still_flagged():
    """A materially coarser rounding than dropping cents (nearest hundred
    thousand, not nearest dollar) is a different, less precise claim --
    still worth flagging, not the same tolerance as normal cent-rounding."""
    draft = "You're sitting on roughly $1,700,000 in this account."
    result = _grounding_check(draft, PROFILE)
    assert result is not None
    assert "$1,700,000" in str(result)


def test_grounded_date_not_flagged():
    draft = "Your account matures on 2026-09-15, so let's discuss renewal."
    assert _grounding_check(draft, PROFILE) is None


def test_hallucinated_date_is_flagged():
    """A wrong maturity/meeting date is exactly the kind of invented
    client fact a $/%-only check would silently miss -- dates are just as
    mechanically detectable (same YYYY-MM-DD format CONTEXT already uses
    throughout) as dollar/percent figures, so there's no reason to leave
    them uncovered."""
    draft = "Your account matures on 2026-11-01, so let's discuss renewal."
    result = _grounding_check(draft, PROFILE)
    assert result is not None
    assert "2026-11-01" in str(result)


def test_date_has_no_rounding_tolerance():
    """Unlike dollar figures, there's no legitimate 'rounding' of a date --
    a date one day off from the real one is still a wrong date, not a
    reasonable approximation, so this must be exact-string, not
    numeric-with-tolerance like the dollar check."""
    draft = "Your account matures on 2026-09-16, so let's discuss renewal."
    result = _grounding_check(draft, PROFILE)
    assert result is not None
    assert "2026-09-16" in str(result)


def test_date_digit_groups_dont_leak_into_known_numbers():
    """Regression test for a real bug found immediately after adding date
    coverage: PROFILE's end_date "2026-09-15" contains digit groups "2026",
    "09", "15" that used to get parsed as standalone known numbers (the
    same regex that extracts $/% figures from the profile blob doesn't
    know a date isn't a number) -- so a fabricated "9%" rate coincidentally
    matched the "09" from the month and silently passed unflagged. Proven
    broken with this exact case before the fix (strip date substrings from
    the blob before extracting known numbers)."""
    draft = "We could offer you a special 9% rate on your account."
    result = _grounding_check(draft, PROFILE)
    assert result is not None
    assert "9%" in str(result)
