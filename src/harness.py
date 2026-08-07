"""
harness.py

The orchestrator. This is the key boundary in the system:
"where the model ends and the harness begins." The model is invoked at
exactly three call sites -- intent_router.route, and two calls inside
draft_outreach on the MCP server (the draft itself, then
_note_faithfulness_check's judge call) -- everywhere else, this file is
plain deterministic Python deciding what tools to call, in what order,
and what to do when a call comes back empty or stale. From the
harness's own point of view it only ever triggers the model in two
places -- route(), and one draft_outreach call per row -- the second
model call inside draft_outreach is an implementation detail of that
tool, invisible to the harness itself.

Agent loop per request:
  1. GATHER-CONTEXT: route the ask to intent(s) via intent_router
  2. ACT: for each intent, call get_clients_by_intent (MCP tool) via the
     skill's fetch_tools binding
  3. VERIFY / RE-PLAN: if a fetch comes back empty, widen the horizon once
     (skill.horizon_days * 2) before concluding "no opportunities" -- this
     is the "re-plan gracefully on empty or stale results" requirement.
     If still empty, the intent is reported as empty, not silently dropped.
  4. Merge + globally re-rank all (intent, client) candidates by score
  5. Return the full ranked list immediately -- no drafting yet

Drafting is lazy, not part of this loop: draft_outreach only runs when a
banker actually expands a row in the UI (Harness.draft_for_channel), one
client at a time. A broad ask can rank 50+ candidates; eagerly drafting a
fixed top-N up front means either spending LLM calls on candidates nobody
looks at, or capping some rows out of "click a row to expand the drafted
outreach" entirely. On-demand drafting satisfies that for every row while
never spending a call on one that goes unopened.

Follow-ups are lazier still: creating one is not a side effect of drafting
at all, only of actually sending (mcp_server.log_contact creates it).
Drafting a message a banker never sends needs no follow-up -- the
follow-up represents a real open question ("did they respond?") that only
exists once something was actually sent.

The harness talks to mcp_server.py exclusively through an MCP client over
stdio -- it does not import data_access/scoring/memory directly. That's
what makes "tool execution through a harness, over MCP" literally true
rather than an in-process shortcut.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client, get_default_environment

from intent_router import route, RoutedIntent
from openai_client import call_router_llm
from skills import get_skill, all_intents

SERVER_SCRIPT = str(Path(__file__).resolve().parent / "mcp_server.py")

# mcp's stdio_client only passes a minimal default env (HOME/PATH/TERM) to the
# spawned server subprocess -- it does NOT inherit the parent's environment.
# The MCP server needs OPENAI_API_KEY (and model overrides) to draft outreach,
# so we forward them explicitly rather than relying on inheritance.
_FORWARDED_ENV_VARS = ("OPENAI_API_KEY", "ROUTER_MODEL", "DRAFT_MODEL", "PREMIUM_DRAFT_MODEL")


def _server_env() -> dict:
    env = get_default_environment()
    for var in _FORWARDED_ENV_VARS:
        if var in os.environ:
            env[var] = os.environ[var]
    return env


@dataclass
class OutreachItem:
    client_eci: str
    client_name: str
    segment: str
    tier: str
    banker_name: str
    intent: str
    channel: str
    score: float
    reason: str
    components: dict
    draft_text: str | None = None
    draft_id: int | None = None
    grounding_flag: str | None = None
    draft_error: str | None = None
    channel_override_note: str | None = None  # set when engagement-preference data overrode the skill default
    model_used: str | None = None            # which model drafted the current text -- gpt-5.4 vs premium gpt-5.5
    personalization_tier: str | None = None  # "standard" or "premium", from the same draft call
    sent: bool = False  # set once "Mark sent" succeeds for this item -- session-local, not re-derived from contact_log


@dataclass
class HarnessResult:
    items: list[OutreachItem] = field(default_factory=list)
    empty_intents: list[str] = field(default_factory=list)  # intents that returned nothing even after widening
    notes: list[str] = field(default_factory=list)           # human-readable trace of re-planning decisions


class Harness:
    def __init__(self):
        self._server_params = StdioServerParameters(
            command=sys.executable, args=[SERVER_SCRIPT], env=_server_env()
        )

    async def _run(self, user_ask: str, channel_override: str | None = None) -> HarnessResult:
        """channel_override: if set, the banker has chosen a channel preference
        for this request (e.g. 'always give me Teams nudges'), which wins over
        each skill's default_channel. None means 'use each intent's default'."""
        result = HarnessResult()

        async with stdio_client(self._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # 1. GATHER-CONTEXT: intent routing
                try:
                    routed = route(user_ask, call_router_llm)
                    result.notes.append(f"Routed to intents: {routed.intents}")
                except Exception as e:
                    # Routing is a single LLM call; if the network/API call
                    # itself fails (not just malformed JSON -- that's handled
                    # inside route() already), degrade to the full funnel
                    # rather than losing the whole request to one failed call.
                    result.notes.append(
                        f"Intent routing failed ({type(e).__name__}: {e}) -- "
                        f"falling back to full-funnel (all intents)"
                    )
                    routed = RoutedIntent(
                        intents=all_intents(), client_name=None, banker_or_team=None, timeframe_days=None
                    )

                all_candidates: list[dict] = []

                for intent in routed.intents:
                    skill = get_skill(intent)
                    horizon = routed.timeframe_days or skill.horizon_days

                    candidates, notes = await self._fetch_with_replan(
                        session, intent, horizon, routed.banker_or_team
                    )
                    result.notes.extend(notes)
                    if not candidates and intent not in result.empty_intents:
                        result.empty_intents.append(intent)

                    if routed.client_name:
                        needle = routed.client_name.lower()
                        candidates = [c for c in candidates if needle in c["client_name"].lower()]

                    all_candidates.extend(candidates)

                # 4. Merge + globally re-rank across all requested intents.
                # Ranking unit is (client, intent, trigger) -- each row is a
                # distinct, independently actionable opportunity, matching
                # a literal "ranked by opportunity score" design: an
                # opportunity, not an aggregated client. A
                # client with two live opportunities (e.g. a maturing CD and
                # an expiring exception) appears as two separate rows, since
                # each may need a different draft, channel, and urgency --
                # not bundled into one relationship-level entry.
                all_candidates.sort(key=lambda c: c["score"], reverse=True)

                # 5. Build the ranked list -- no drafting here, that's lazy
                # (see draft_for_channel). The engagement-channel-hint check
                # still runs for every row eagerly, though: it's a cheap
                # deterministic note lookup, not an LLM call, so there's no
                # cost reason to defer it -- doing it now means the channel
                # badge is accurate immediately, for every row, not only
                # once a banker happens to draft that one.
                for cand in all_candidates:
                    skill = get_skill(cand["intent"])
                    item = OutreachItem(
                        client_eci=cand["client_eci"],
                        client_name=cand["client_name"],
                        segment=cand["segment"],
                        tier=cand["tier"],
                        banker_name=cand["banker_name"],
                        intent=cand["intent"],
                        channel=channel_override or skill.default_channel,
                        score=cand["score"],
                        reason=cand["reason"],
                        components=cand["components"],
                    )
                    if channel_override is None:
                        hint_channel, hint_note = await self._engagement_hint(session, item.client_eci)
                        # Log the reason whenever the hint fires at all, not
                        # only when it changes the channel -- if the skill's
                        # default already happens to match the hint, there's
                        # still a real reason for that channel; staying
                        # silent there looks identical to "no reason, just
                        # the arbitrary default," which it isn't.
                        if hint_channel:
                            if hint_channel != item.channel:
                                result.notes.append(
                                    f"Channel for {item.client_name} overridden to {hint_channel} "
                                    f"based on engagement-preference note: \"{hint_note}\""
                                )
                            else:
                                result.notes.append(
                                    f"Channel for {item.client_name} ({hint_channel}) matches "
                                    f"engagement-preference note: \"{hint_note}\""
                                )
                            item.channel = hint_channel
                            item.channel_override_note = hint_note
                    result.items.append(item)

        return result

    @staticmethod
    async def _fetch_with_replan(session, intent: str, horizon: int, banker_or_team) -> tuple[list[dict], list[str]]:
        """ACT + VERIFY/RE-PLAN for a single intent, extracted as a standalone,
        independently testable unit (takes any object with an async call_tool
        method -- a real MCP ClientSession or a mock). Returns (candidates,
        trace_notes) rather than mutating a HarnessResult directly, so it can
        be unit tested without spinning up a real MCP subprocess or an
        OpenAI-backed router call."""
        notes: list[str] = []
        try:
            candidates = await Harness._fetch_candidates(session, intent, horizon, banker_or_team)
        except Exception as e:
            # MCP tool call failed (subprocess/data error, not just "no
            # matches" -- that's a valid empty list, handled below). Skip
            # this intent rather than crash the whole request over one.
            notes.append(f"'{intent}' fetch failed ({type(e).__name__}: {e}) -- skipping this intent")
            return [], notes

        if not candidates:
            widened = horizon * 2
            notes.append(f"'{intent}' empty at {horizon}d window -- widening to {widened}d")
            try:
                candidates = await Harness._fetch_candidates(session, intent, widened, banker_or_team)
            except Exception as e:
                notes.append(f"'{intent}' widened fetch also failed ({type(e).__name__}: {e})")
                candidates = []
            if not candidates:
                notes.append(f"'{intent}' still empty at {widened}d -- reporting as no matches")

        return candidates, notes

    @staticmethod
    async def _fetch_candidates(session: ClientSession, intent: str, horizon: int, banker_or_team) -> list[dict]:
        res = await session.call_tool(
            "get_clients_by_intent",
            {"intent": intent, "timeframe_days": horizon, "banker_or_team": banker_or_team},
        )
        text = res.content[0].text
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []

    @staticmethod
    async def _engagement_hint(session: ClientSession, client_eci: str) -> tuple[str | None, str | None]:
        res = await session.call_tool("get_engagement_channel_hint", {"client_eci": client_eci})
        parsed = json.loads(res.content[0].text)
        return parsed.get("channel_hint"), parsed.get("note")

    @staticmethod
    async def _draft(session: ClientSession, item: OutreachItem, cand: dict) -> dict:
        res = await session.call_tool(
            "draft_outreach",
            {
                "client_eci": item.client_eci,
                "intent": item.intent,
                "channel": item.channel,
                "trigger_context": {"reason": cand["reason"], "components": cand["components"]},
            },
        )
        text = res.content[0].text
        return json.loads(text)

    @staticmethod
    async def record_reaction(client_eci: str, channel: str, reaction: str, note: str = "", intent: str | None = None):
        harness = Harness()
        async with stdio_client(harness._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool(
                    "record_reaction",
                    {"client_eci": client_eci, "channel": channel, "reaction": reaction, "note": note, "intent": intent},
                )

    @staticmethod
    async def get_followup_history(client_eci: str, intent: str) -> list:
        harness = Harness()
        async with stdio_client(harness._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.call_tool(
                    "get_followup_history",
                    {"client_eci": client_eci, "intent": intent},
                )
                return json.loads(res.content[0].text)

    @staticmethod
    async def get_engagement_hint(client_eci: str) -> dict:
        """Public entry point for the same engagement-preference check
        _run() applies eagerly to every ranked-list row -- exposed
        separately because a follow-up-driven draft never goes through
        _run() at all (it's constructed straight from a todos row), so it
        would otherwise never get this check applied and never explain
        why a channel is what it is."""
        harness = Harness()
        async with stdio_client(harness._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                channel, note = await harness._engagement_hint(session, client_eci)
                return {"channel_hint": channel, "note": note}

    @staticmethod
    async def get_open_todos(client_eci: str | None = None) -> list:
        """Wraps the get_open_todos MCP tool -- keeps the UI's "Open
        follow-ups" list on the MCP boundary like every other memory read,
        instead of streamlit_app.py importing memory.py directly."""
        harness = Harness()
        async with stdio_client(harness._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.call_tool("get_open_todos", {"client_eci": client_eci})
                return json.loads(res.content[0].text)

    @staticmethod
    async def get_client_profile(client_eci: str) -> dict:
        """Wraps the get_client_profile MCP tool -- lets the UI look up a
        client's profile (e.g. for a banker-name label on a follow-up-
        driven draft) without importing data_access.py directly."""
        harness = Harness()
        async with stdio_client(harness._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.call_tool("get_client_profile", {"client_eci": client_eci})
                return json.loads(res.content[0].text)

    @staticmethod
    async def complete_todo(todo_id: int) -> dict:
        harness = Harness()
        async with stdio_client(harness._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.call_tool("complete_todo", {"todo_id": todo_id})
                return json.loads(res.content[0].text)

    @staticmethod
    async def reset_memory() -> dict:
        harness = Harness()
        async with stdio_client(harness._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.call_tool("reset_memory", {})
                return json.loads(res.content[0].text)

    @staticmethod
    async def log_contact(
        client_eci: str, intent: str, channel: str, draft_id: int | None = None,
        override_grounding_flag: bool = False, override_recent_contact: bool = False,
        source_todo_id: int | None = None,
    ) -> dict:
        harness = Harness()
        async with stdio_client(harness._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.call_tool(
                    "log_contact",
                    {
                        "client_eci": client_eci, "intent": intent, "channel": channel,
                        "draft_id": draft_id, "override_grounding_flag": override_grounding_flag,
                        "override_recent_contact": override_recent_contact,
                        "source_todo_id": source_todo_id,
                    },
                )
                return json.loads(res.content[0].text)

    async def _draft_with_channel(self, item: OutreachItem, new_channel: str, cand: dict) -> dict:
        async with stdio_client(self._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                item.channel = new_channel
                return await self._draft(session, item, cand)

    def draft_for_channel(self, item: OutreachItem, new_channel: str, reason: str, components: dict) -> dict:
        """Draft (or redraft) outreach for one ranked item in the given
        channel -- used identically for a row's first draft (triggered when
        a banker expands it, not eagerly for a fixed top-N up front) and
        for any later redraft in a different channel. No follow-up is
        created here: a draft nobody sends needs no follow-up. The
        follow-up gets created when the banker actually clicks Mark sent
        (see log_contact on the MCP server), not when a draft merely
        exists -- drafting and committing to sending are different
        actions and shouldn't share a side effect."""
        cand = {"reason": reason, "components": components}
        try:
            return asyncio.run(self._draft_with_channel(item, new_channel, cand))
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, user_ask: str, channel_override: str | None = None) -> HarnessResult:
        """Sync entry point for Streamlit."""
        return asyncio.run(self._run(user_ask, channel_override=channel_override))
