"""
Guardrail enforcement tests. These specifically test that the grounding-flag
block is enforced at the MCP tool layer (log_contact), not just displayed
as a UI warning -- i.e. that it can't be bypassed by any caller, not only
the Streamlit app.
"""
import os
import sys
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import memory
import mcp_server


@pytest.fixture(autouse=True)
def isolated_memory_db(tmp_path, monkeypatch):
    """log_contact (and now, since it creates follow-ups, its side effects)
    writes through memory.DB_PATH -- point that at a throwaway file for the
    duration of each test instead of the real data/memory.sqlite3, so
    running the test suite doesn't leave TEST123/TEST456/TEST789 rows in
    actual demo data. monkeypatch reverts DB_PATH automatically after each
    test; tmp_path is cleaned up by pytest."""
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "test_memory.sqlite3")
    memory.init_db()


def test_log_contact_blocks_flagged_draft_without_override():
    draft_id = memory.save_draft(
        "TEST123", "maturity", "email", "Some drafted text",
        grounding_flag="UNVERIFIED figures in draft: ['$999,999,999']",
    )
    result = json.loads(mcp_server.log_contact("TEST123", "maturity", "email", draft_id=draft_id))
    assert "error" in result
    assert "BLOCKED" in result["error"]
    # and it must NOT have actually logged the contact
    assert memory.recently_contacted("TEST123", within_days=1) is False


def test_log_contact_allows_flagged_draft_with_explicit_override():
    draft_id = memory.save_draft(
        "TEST456", "maturity", "email", "Some drafted text",
        grounding_flag="UNVERIFIED figures in draft: ['$999,999,999']",
    )
    result = json.loads(mcp_server.log_contact(
        "TEST456", "maturity", "email", draft_id=draft_id, override_grounding_flag=True
    ))
    assert result.get("status") == "logged"
    assert memory.recently_contacted("TEST456", within_days=1) is True


def test_log_contact_allows_clean_draft_without_override():
    draft_id = memory.save_draft("TEST789", "maturity", "email", "Clean text", grounding_flag=None)
    result = json.loads(mcp_server.log_contact("TEST789", "maturity", "email", draft_id=draft_id))
    assert result.get("status") == "logged"


def test_log_contact_followup_text_uses_real_date_not_pinned_demo_date():
    """The follow-up's own "Sent {date}... check back by {date}" text must
    agree with contact_log/reactions timestamps, which use real wall-clock
    time -- not data_access.TODAY, which is pinned purely for reproducible
    opportunity SCORING against the static mock dataset. That pin never
    changes regardless of when the app runs; follow-ups/History/contact_log
    track a genuinely different thing (what a banker actually did, and
    when) and must stay on the real clock for both the initial send and
    any resend -- otherwise recently_contacted's 7-day guardrail (which
    reads contact_log.contacted_at directly) breaks, since a pinned date
    only gets further in the past as real time moves on."""
    import datetime as dt
    import data_access

    draft_id = memory.save_draft("TEST_DATE", "maturity", "email", "text", grounding_flag=None)
    result = json.loads(mcp_server.log_contact("TEST_DATE", "maturity", "email", draft_id=draft_id))
    assert result.get("status") == "logged"

    todo = memory.open_todos("TEST_DATE")[0]
    real_today = dt.datetime.now().date().isoformat()
    assert real_today in todo["action"]
    assert data_access.TODAY.isoformat() not in todo["action"]


def test_followup_history_survives_a_channel_switch():
    """get_followup_history must not filter by channel: a retry from an
    existing follow-up can go out on a different channel than the
    original send (that's the point of letting a banker redraft for a
    different channel), and it's still the same ongoing thread about the
    same opportunity. A channel-scoped query would silently make the
    switch look like the conversation just started, hiding every earlier
    send/reaction recorded under the old channel."""
    draft_id = memory.save_draft("TEST_HIST", "liquidity_event", "call_brief", "text", grounding_flag=None)
    mcp_server.log_contact("TEST_HIST", "liquidity_event", "call_brief", draft_id=draft_id)
    todo_id = memory.open_todos("TEST_HIST")[0]["id"]
    memory.record_reaction("TEST_HIST", "call_brief", "NO_RESPONSE", intent="liquidity_event")

    draft_id2 = memory.save_draft("TEST_HIST", "liquidity_event", "meeting_notes", "text2", grounding_flag=None)
    result = json.loads(mcp_server.log_contact(
        "TEST_HIST", "liquidity_event", "meeting_notes", draft_id=draft_id2,
        source_todo_id=todo_id, override_recent_contact=True,
    ))
    assert result.get("status") == "logged"

    history = memory.get_followup_history("TEST_HIST", "liquidity_event")
    channels_seen = {h["channel"] for h in history}
    assert channels_seen == {"call_brief", "meeting_notes"}
    assert len(history) == 3  # sent (call_brief), reaction (call_brief), sent (meeting_notes)
