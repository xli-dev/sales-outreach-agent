"""
Unit tests for _note_faithfulness_check (mcp_server.py) -- the permanent,
always-on guardrail promoted from tests/test_note_faithfulness_eval.py
after that eval reached a clean 100% (12/12) result. These tests inject a
fake llm_call (no real API call, fully deterministic) so the guardrail's
own logic -- parsing the judge's verdict, building the flag message,
handling a malformed response -- has fast CI coverage independent of the
real, slower, real-API-calling eval.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcp_server import _note_faithfulness_check

PROFILE = {"name": "Test Client", "banker_name": "Test Banker", "recent_notes": []}


def test_faithful_verdict_returns_none():
    fake_llm = lambda sp, up: json.dumps({"faithful": True, "issues": []})
    assert _note_faithfulness_check("Some draft text.", PROFILE, llm_call=fake_llm) is None


def test_unfaithful_verdict_is_flagged():
    fake_llm = lambda sp, up: json.dumps({
        "faithful": False,
        "issues": ["Claims the client is highly rate-sensitive; not supported by profile."],
    })
    result = _note_faithfulness_check("Some draft text.", PROFILE, llm_call=fake_llm)
    assert result is not None
    assert "highly rate-sensitive" in result


def test_malformed_judge_response_is_flagged_not_silently_passed():
    """If the judge doesn't return parseable JSON, this must NOT silently
    treat the draft as faithful -- that would make a judge failure
    indistinguishable from a clean pass, defeating the guardrail's
    purpose. Flagging for manual review on a parse failure is the safe
    default, same principle as _grounding_check never silently correcting
    or dropping a suspect figure."""
    fake_llm = lambda sp, up: "not json"
    result = _note_faithfulness_check("Some draft text.", PROFILE, llm_call=fake_llm)
    assert result is not None
    assert "unparseable" in result.lower()


def test_llm_call_receives_full_profile_and_draft():
    """Regression guard for the exact bug found while building the eval
    this guardrail was promoted from: the judge must see the COMPLETE
    profile (not a hand-picked subset), or it flags real, correctly-
    grounded facts (a banker's own name, an account's justification
    comment) as fabricated simply because it wasn't shown where they
    actually came from."""
    captured = {}

    def fake_llm(system_prompt, user_prompt):
        captured["user_prompt"] = user_prompt
        return json.dumps({"faithful": True, "issues": []})

    rich_profile = {
        "name": "Kenji Chen", "banker_name": "Grace Osei", "rate_sensitivity": "HIGH",
        "accounts": [{"justification_comment": "quarterly bake-off for company TD"}],
        "recent_notes": [],
    }
    _note_faithfulness_check("Some draft text.", rich_profile, llm_call=fake_llm)
    assert "Grace Osei" in captured["user_prompt"]
    assert "quarterly bake-off for company TD" in captured["user_prompt"]
    assert "HIGH" in captured["user_prompt"]
