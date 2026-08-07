"""
engagement_channel_hint is a deterministic keyword match over a client's
RELATIONSHIP notes -- it must only fire on the two note patterns that map
onto a channel this system can actually produce (see data_access.py for
why a "wants Teams messages" note is deliberately NOT mapped: this
system's `teams` channel is an internal nudge to the banker, not
client-facing copy, so honoring that note literally would misrepresent
what gets produced).
"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_access import BankerNote, engagement_channel_hint


def _note(text: str, topic: str = "RELATIONSHIP") -> BankerNote:
    return BankerNote(
        note_id="N1", client_eci="TEST123", date=dt.date(2026, 1, 1),
        author_banker_id="BKR0001", channel="NOTE", text=text,
        client_reaction="NEUTRAL", topic=topic, linked_request_id=None,
    )


def test_prefers_call_maps_to_call_brief():
    notes = [_note("Engagement preference — prefers a short call over email.")]
    channel, note = engagement_channel_hint(notes)
    assert channel == "call_brief"
    assert "short call" in note


def test_in_person_maps_to_meeting_notes():
    notes = [_note("Engagement preference — likes an in-person quarterly review.")]
    channel, note = engagement_channel_hint(notes)
    assert channel == "meeting_notes"


def test_wants_teams_is_not_mapped():
    """Deliberately unmapped: our `teams` channel produces an internal
    banker note, not client-facing copy, so this note can't be honored
    without misrepresenting what the system actually sends."""
    notes = [_note("Engagement preference — wants concise Teams messages, not long emails.")]
    channel, note = engagement_channel_hint(notes)
    assert channel is None


def test_non_relationship_topic_ignored():
    notes = [_note("Engagement preference — prefers a short call over email.", topic="LIQUIDITY")]
    channel, _ = engagement_channel_hint(notes)
    assert channel is None


def test_no_matching_note_returns_none():
    notes = [_note("Client asked about wire cutoff times.")]
    channel, note = engagement_channel_hint(notes)
    assert channel is None
    assert note is None


def test_empty_notes_returns_none():
    assert engagement_channel_hint([]) == (None, None)
