"""
Tests for harness.py's planning & recovery logic (_fetch_with_replan),
extracted specifically so it's testable without a live MCP subprocess or
OpenAI key. Uses a minimal mock session (anything with an async call_tool)
rather than a real ClientSession -- these test the CONTROL FLOW (widen on
empty, retry once, skip-and-report on failure), not the actual data/LLM
behind it, which is covered separately by test_scoring.py against the real
dataset.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from harness import Harness


class _FakeContent:
    def __init__(self, text):
        self.text = text


class _FakeResult:
    def __init__(self, payload):
        self.content = [_FakeContent(json.dumps(payload))]


class _MockSession:
    """call_tool returns whatever the queue for that call index says to,
    or raises if the queued value is an Exception instance."""
    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return _FakeResult(response)


def _run(coro):
    return asyncio.run(coro)


def test_no_replan_needed_when_first_fetch_has_results():
    session = _MockSession([[{"client_eci": "1", "score": 90}]])
    candidates, notes = _run(Harness._fetch_with_replan(session, "maturity", 30, None))
    assert len(candidates) == 1
    assert len(session.calls) == 1  # no widening call made
    assert not any("widening" in n for n in notes)


def test_widens_once_on_empty_then_succeeds():
    session = _MockSession([[], [{"client_eci": "2", "score": 70}]])
    candidates, notes = _run(Harness._fetch_with_replan(session, "exception_expiry", 45, None))
    assert len(candidates) == 1
    assert len(session.calls) == 2
    # second call used the widened horizon (2x)
    assert session.calls[1][1]["timeframe_days"] == 90
    assert any("widening to 90d" in n for n in notes)


def test_widens_once_then_reports_still_empty_not_fabricated():
    session = _MockSession([[], []])
    candidates, notes = _run(Harness._fetch_with_replan(session, "liquidity_event", 21, None))
    assert candidates == []  # never fabricates a candidate
    assert len(session.calls) == 2  # exactly one retry, not infinite
    assert any("still empty" in n for n in notes)


def test_fetch_failure_is_skipped_not_crashed():
    session = _MockSession([RuntimeError("MCP subprocess died")])
    candidates, notes = _run(Harness._fetch_with_replan(session, "new_account", 30, None))
    assert candidates == []
    assert len(session.calls) == 1  # does not retry on a hard failure, only on empty
    assert any("fetch failed" in n and "RuntimeError" in n for n in notes)


def test_widened_fetch_failure_also_recovers_gracefully():
    session = _MockSession([[], ConnectionError("timeout")])
    candidates, notes = _run(Harness._fetch_with_replan(session, "maturity", 30, None))
    assert candidates == []
    assert any("widened fetch also failed" in n for n in notes)
