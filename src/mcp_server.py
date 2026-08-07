"""
mcp_server.py

All data and model access is exposed as MCP tools. The harness talks
to this server via an MCP client -- it never touches data_access.py or
the OpenAI client directly. This keeps "the harness performs tool
calls" honest: swap this server for a real deposits API tomorrow and
nothing above this layer changes.

Run standalone for local dev:
    python src/mcp_server.py

Exposes:
  Tools:
    - get_clients_by_intent(intent, timeframe_days, banker_or_team)
    - get_client_profile(client_eci)
    - get_banker_notes(client_eci, topic)
    - get_engagement_channel_hint(client_eci)
    - draft_outreach(client_eci, intent, channel, trigger_context)
    - create_todo(client_eci, action, due_date, client_name, intent, channel)
    - complete_todo(todo_id)
    - reset_memory()  -- wipes all persistent memory, for demo resets
    - record_reaction(client_eci, channel, reaction, note, intent)
    - get_followup_history(client_eci, intent)  -- merged, sorted
      contact_log + reactions history for one opportunity
    - recently_contacted(client_eci, within_days)
    - log_contact(client_eci, intent, channel)  -- also updates the matching
      follow-up's text to reflect it was sent, without closing it
    - get_open_todos(client_eci)  -- backs the UI's "Open follow-ups" list
  Resources:
    - data://decision_makers  (the raw joined dataset, for inspection/debug)
  Prompts:
    - one prompt per skill, sourced from skills.SKILL_REGISTRY, so
      drafting instructions are versioned and inspectable over MCP
      rather than buried in application code.
"""
import json
import os
import datetime as dt
from typing import Optional

from mcp.server.fastmcp import FastMCP

import data_access
import memory
import scoring
from skills import SKILL_REGISTRY, get_skill

mcp = FastMCP("sales-outreach-agent")
memory.init_db()


def _real_today() -> dt.date:
    """The actual current date, in local server time -- deliberately
    distinct from data_access.TODAY, which is pinned for reproducible
    opportunity scoring against the static mock dataset. Local, not UTC:
    this is a single-instance demo app, not a distributed multi-timezone
    service, so there's no benefit to UTC's usual justification -- only
    the cost of "today" flipping to tomorrow in the late afternoon/evening
    for anyone west of Greenwich, which reads as a bug to whoever's
    actually looking at the app. Session events (sent/due/retry dates on
    follow-ups) need to agree with contact_log and reactions timestamps,
    which use the same local-time convention -- mixing UTC and local (or
    the pinned demo date and real time) produced follow-up text dated
    differently than the History entry for the exact same event."""
    return dt.datetime.now().date()


def _channel_recipient_label(channel: str, banker_name: Optional[str], client_name: Optional[str]) -> Optional[str]:
    """Human-readable "who this draft is actually for" label for banker-
    facing channels, matching the header draft_outreach's channel_guidance
    asks the model to open the draft itself with (see below) -- used
    outside the draft too (follow-up text, harness trace) so that fact is
    stated wherever this outreach shows up, not only inside the draft.
    None for "email": that's client-facing, no banker relabeling needed."""
    banker_name = banker_name or "the covering banker"
    client_name = client_name or "the client"
    return {
        "teams": f"Teams message for {banker_name} -- re: {client_name}",
        "call_brief": f"Call brief for {banker_name} -- call with {client_name}",
        "meeting_notes": f"Meeting prep for {banker_name} -- meeting with {client_name}",
    }.get(channel)


# ---------------------------------------------------------------------------
# Tools: data access (grounding layer -- no invented facts, ever)
# ---------------------------------------------------------------------------

@mcp.tool()
def get_clients_by_intent(
    intent: str, timeframe_days: Optional[int] = None, banker_or_team: Optional[str] = None
) -> str:
    """Fetch candidate (client, trigger) pairs for a given intent, scored by
    opportunity. Returns JSON list sorted highest score first. Empty list
    means genuinely no matches in the window -- the harness should widen
    the window or report that clearly, never fabricate a candidate."""
    store = data_access.get_store()
    skill = get_skill(intent)
    if skill is None:
        return json.dumps({"error": f"unknown intent '{intent}'"})
    horizon = timeframe_days or skill.horizon_days

    # Dispatch is entirely driven by the skill registry's query_fn/score_fn
    # bindings -- this function has zero knowledge of specific intent names.
    # Adding intent #6 means a new SKILL_REGISTRY entry in skills.py with its
    # own query_fn/score_fn; nothing here changes.
    results: list[scoring.ScoredOpportunity] = []
    for dm, trigger in skill.query_fn(store, horizon):
        live = memory.latest_reaction_value(dm.client_eci)
        results.append(skill.score_fn(dm, trigger, horizon, live_reaction=live))

    if banker_or_team:
        needle = banker_or_team.lower()
        results = [
            r for r in results
            if needle in r.client.banker_name.lower() or needle in r.client.team.lower()
        ]

    # suppress recently-contacted clients (memory-informed filtering)
    results = [r for r in results if not memory.recently_contacted(r.client.client_eci)]

    results.sort(key=lambda r: r.score, reverse=True)

    return json.dumps([
        {
            "client_eci": r.client.client_eci,
            "client_name": r.client.name,
            "segment": r.client.segment,
            "tier": r.client.tier,
            "banker_name": r.client.banker_name,
            "intent": r.intent,
            "score": r.score,
            "components": r.components,
            "reason": r.reason,
        }
        for r in results
    ])


@mcp.tool()
def get_client_profile(client_eci: str) -> str:
    """Full client profile: DM info, accounts (with rates), recent transactions,
    recent banker notes, and any open to-dos. This is the grounding context
    passed to drafting -- every number quoted downstream must trace to this."""
    store = data_access.get_store()
    dm = store.get_client(client_eci)
    if dm is None:
        return json.dumps({"error": f"no client with CLIENT_ECI={client_eci}"})

    todos = [dict(t) for t in memory.open_todos(client_eci)]

    return json.dumps({
        "client_eci": dm.client_eci,
        "name": dm.name,
        "segment": dm.segment,
        "tier": dm.tier,
        "total_aus_usd": dm.total_aus_usd,
        "rate_sensitivity": dm.rate_sensitivity,
        "banker_name": dm.banker_name,
        "team": dm.team,
        "region": dm.region,
        "accounts": [
            {
                "account_id": a.account_id,
                "product_type": a.product_type,
                "balance_usd": a.balance_usd,
                "rate": a.rate,
                "libor_rate": a.libor_rate,
                "is_exception_priced": a.is_exception_priced,
                "exception_rate": a.exception_rate,
                "end_date": a.end_date.isoformat() if a.end_date else None,
                "proposed_rate": a.proposed_rate,
                "competitor_rate": a.competitor_rate,
                "competitor_product": a.competitor_product,
                "justification_comment": a.justification_comment,
            }
            for a in dm.accounts
        ],
        "recent_transactions": [
            {
                "date": t.date.isoformat(),
                "direction": t.direction,
                "amount": t.amount,
                "type": t.type,
                "counterparty": t.counterparty,
                "liquidity_event_type": t.liquidity_event_type,
            }
            for t in dm.transactions[:10]
        ],
        "recent_notes": [
            {
                "date": n.date.isoformat(),
                "channel": n.channel,
                "text": n.text,
                "client_reaction": n.client_reaction,
                "topic": n.topic,
            }
            for n in dm.notes[:10]
        ],
        "open_todos": todos,
    })


@mcp.tool()
def get_banker_notes(client_eci: str, topic: Optional[str] = None) -> str:
    """Banker notes for a client, optionally filtered by topic
    (RELATIONSHIP, LIQUIDITY, RATE_NEGOTIATION, MATURITY_PLANNING, RETENTION)."""
    store = data_access.get_store()
    dm = store.get_client(client_eci)
    if dm is None:
        return json.dumps({"error": f"no client with CLIENT_ECI={client_eci}"})
    notes = dm.notes
    if topic:
        notes = [n for n in notes if n.topic == topic]
    return json.dumps([
        {"date": n.date.isoformat(), "channel": n.channel, "text": n.text,
         "client_reaction": n.client_reaction, "topic": n.topic}
        for n in notes
    ])


@mcp.tool()
def get_engagement_channel_hint(client_eci: str) -> str:
    """Check this client's RELATIONSHIP notes for an engagement-preference
    signal (e.g. 'prefers a short call over email') that maps onto a channel
    this system can produce. Returns {"channel_hint": ..., "note": ...},
    both null if no note gives a clear, actionable signal -- used by the
    harness to override a skill's static default_channel per client,
    grounded in real note data rather than a blanket per-intent guess."""
    store = data_access.get_store()
    dm = store.get_client(client_eci)
    if dm is None:
        return json.dumps({"error": f"no client with CLIENT_ECI={client_eci}"})
    channel, note = data_access.engagement_channel_hint(dm.notes)
    return json.dumps({"channel_hint": channel, "note": note})


# ---------------------------------------------------------------------------
# Tools: memory (persisted across requests)
# ---------------------------------------------------------------------------

@mcp.tool()
def recently_contacted(client_eci: str, within_days: int = 7) -> bool:
    """True if this client was contacted (any intent/channel) within N days."""
    return memory.recently_contacted(client_eci, within_days)


@mcp.tool()
def get_open_todos(client_eci: Optional[str] = None) -> str:
    """List open follow-ups, optionally filtered to one client. Backs the
    UI's "Open follow-ups" list -- kept on the MCP boundary like every
    other memory read, rather than the UI importing memory.py directly to
    query the todos table itself."""
    rows = memory.open_todos(client_eci)
    return json.dumps([dict(r) for r in rows])


@mcp.tool()
def log_contact(
    client_eci: str, intent: str, channel: str, draft_id: Optional[int] = None,
    override_grounding_flag: bool = False, override_recent_contact: bool = False,
    source_todo_id: Optional[int] = None,
) -> str:
    """Record that outreach was sent, so future requests suppress this client.
    GUARDRAIL 1: if draft_id references a draft with an unresolved grounding
    flag (unverified $/% figures the model wasn't confirmed to have grounded
    correctly), this refuses to log the send unless override_grounding_flag
    is explicitly set.
    GUARDRAIL 2: if this client was already contacted (any intent/channel)
    within the last 7 days, this refuses unless override_recent_contact is
    set. The ranked list already filters out recently-contacted clients,
    but only at fetch time -- on a shared deployment, two bankers can each
    hold the same client in their own already-fetched ranked list, and
    without this check both could independently send. This catches that at
    the last possible moment, not just when the list was generated.
    Both guardrails are enforced HERE, not just in the UI, so no caller
    (this app's UI, a future integration, a script) can silently bypass
    them by skipping a client-side check.

    source_todo_id: set when this send was drafted directly from an
    existing follow-up (e.g. a "convert" reminder, or a "no response,
    retry" one) rather than from the ranked list. In that case, THAT exact
    row is updated in place -- text and due date refreshed, reclassified
    to kind="outreach" -- rather than running the generic outreach
    find-or-create below. Without this, a send from a "convert" follow-up
    would leave it open (untouched, since upsert_sent_todo only ever
    matches kind="outreach") AND create an unrelated new "outreach,
    waiting" entry -- two open follow-ups for what the banker experienced
    as one action, confirmed exactly this way in practice. Kept OPEN
    rather than closed, too: an earlier version of this fix closed it
    outright, which silently dropped the fact that a new outreach had just
    gone out and something was now legitimately being waited on."""
    if draft_id is not None:
        draft = memory.get_draft(draft_id)
        if draft and draft["grounding_flag"] and not override_grounding_flag:
            return json.dumps({
                "error": "BLOCKED: draft has an unresolved grounding flag -- "
                         "review the flagged figures and resend with override_grounding_flag=True to confirm.",
                "grounding_flag": draft["grounding_flag"],
            })
    if memory.recently_contacted(client_eci) and not override_recent_contact:
        return json.dumps({
            "error": "BLOCKED: this client was already contacted within the last 7 days -- "
                     "possibly by someone else since your list was generated. Confirm this "
                     "outreach is still needed and resend with override_recent_contact=True.",
            "already_contacted": True,
        })
    memory.log_contact(client_eci, intent, channel, draft_id=draft_id)
    # The follow-up stays OPEN through this -- it's about what happens
    # AFTER sending, and only closes once a reaction is recorded (see
    # record_reaction), since that's when the outcome is actually known.
    # A draft nobody sends needs no follow-up, which is why this is where
    # one first appears (not at draft time) if none already exists.
    #
    # Every follow-up date here -- initial send or resend alike -- uses the
    # REAL clock, not data_access.TODAY (which is pinned for reproducible
    # opportunity SCORING against the static mock dataset, and stays fixed
    # regardless of when the app runs). Follow-ups/History/contact_log
    # track what a banker actually did and when -- that's a genuinely
    # different clock from the scoring snapshot, and the two are not meant
    # to agree: forcing them to (by pinning follow-up dates, or by
    # backdating contact_log to match) either misrepresents when a real
    # action happened or breaks recently_contacted's 7-day guardrail, which
    # reads contact_log.contacted_at directly.
    today = _real_today()
    sent_date = today.isoformat()
    due_date = (today + dt.timedelta(days=7)).isoformat()
    store = data_access.get_store()
    dm = store.get_client(client_eci)
    # Banker-facing channels are drafted FOR the covering banker (see
    # channel_guidance's header instruction in draft_outreach) -- state that
    # same fact here too, so the follow-up list and any trace built from
    # this text says who it's actually for, not just the draft itself.
    label = _channel_recipient_label(channel, dm.banker_name if dm else None, dm.name if dm else None)
    prefix = f"{label} -- " if label else ""
    todo_action = (
        f"{prefix}Sent {sent_date} ({intent.replace('_', ' ')}, {channel}) -- "
        f"check back by {due_date} if no response"
    )
    if source_todo_id is not None:
        # This send was drafted directly from an existing follow-up (a
        # "convert" reminder, or a "no response, retry" one) -- update
        # THAT exact row in place rather than the generic outreach lookup
        # below, and reclassify it to kind="outreach" now that it's
        # functioning as a normal "sent, waiting" follow-up. Kept OPEN,
        # not closed: a fresh send creates a new open question, it doesn't
        # resolve one -- closing it here would just make the follow-up
        # disappear as if nothing happened, which is exactly the
        # complaint that led to source_todo_id resolving via completion
        # in the first place, before this correction.
        memory.retry_todo(source_todo_id, todo_action, due_date, kind="outreach", channel=channel)
        return json.dumps({"status": "logged"})
    # Fresh send from the ranked list, not tied to an existing follow-up --
    # upsert_sent_todo handles the find-existing-or-create atomically, see
    # its docstring for why a naive find-then-create here was a real,
    # reproduced race condition.
    memory.upsert_sent_todo(
        client_eci, todo_action, due_date, dm.name if dm else None, intent, channel,
    )
    return json.dumps({"status": "logged"})


@mcp.tool()
def create_todo(
    client_eci: str, action: str, due_date: Optional[str] = None, client_name: Optional[str] = None,
    intent: Optional[str] = None, channel: Optional[str] = None,
) -> int:
    """Create a follow-up to-do, carried forward into future planning passes.
    client_name is stored alongside the ECI purely for display. intent and
    channel are stored so log_contact/record_reaction can later find this
    exact todo again (a client can have multiple live opportunities, each
    with its own follow-up -- matching by client_eci alone isn't enough)."""
    return memory.create_todo(client_eci, action, due_date, client_name, intent, channel)


@mcp.tool()
def complete_todo(todo_id: int) -> str:
    """Mark a follow-up to-do DONE so it stops showing in Open follow-ups."""
    memory.complete_todo(todo_id)
    return json.dumps({"status": "done", "todo_id": todo_id})


@mcp.tool()
def get_followup_history(client_eci: str, intent: str) -> str:
    """Chronological history for one (client, intent) opportunity thread
    -- every send and every recorded reaction, merged and sorted, across
    every channel it was ever contacted through. All of this was already
    permanently tracked in contact_log/reactions; this just surfaces it,
    since a follow-up's own text gets overwritten in place on each update
    and otherwise shows only its latest state."""
    return json.dumps(memory.get_followup_history(client_eci, intent))


@mcp.tool()
def reset_memory() -> str:
    """Wipe all persistent memory (contact_log, todos, reactions,
    draft_history) -- an explicit demo-reset action, never called as part
    of normal request handling."""
    memory.reset_all()
    return json.dumps({"status": "reset"})


@mcp.tool()
def record_reaction(client_eci: str, channel: str, reaction: str, note: str = "", intent: Optional[str] = None) -> str:
    """Record a client/banker reaction to outreach (POSITIVE/NEUTRAL/NEGATIVE/NO_RESPONSE).
    Feeds response_conversion scoring on subsequent requests. Also resolves
    the follow-up this outreach created, if intent is given -- but the
    three outcomes aren't equally final, so "resolves" means something
    different for each:
      - NEGATIVE: closes outright. Nothing to actively pursue right now
        (the engagement-score penalty already handles ranking it lower).
      - POSITIVE: closes this follow-up (the "did they respond" question
        is answered), but creates a NEW one -- a positive reply is the
        start of a task (convert it into a booked deposit), not the end
        of one; closing with nothing further would drop that thread.
      - anything else (NO_RESPONSE, NEUTRAL, ...): NOT treated as
        resolved -- "no response by the due date" is a window that
        passed, not an answer. The follow-up is rewritten with a fresh
        due date instead of closed, so it keeps nagging rather than
        silently disappearing as if the question had been settled.
    """
    memory.record_reaction(client_eci, channel, reaction, note, intent=intent)
    if intent:
        if reaction == "POSITIVE":
            # Atomic: finds+closes the open outreach follow-up and creates
            # the conversion one in a single transaction, so two
            # near-simultaneous Positive reactions can't both create a
            # duplicate conversion follow-up (see memory.resolve_positive_reaction).
            new_due = (_real_today() + dt.timedelta(days=3)).isoformat()
            store = data_access.get_store()
            dm = store.get_client(client_eci)
            memory.resolve_positive_reaction(
                client_eci, intent, channel,
                f"Positive response on {intent.replace('_', ' ')} ({channel}) -- "
                f"schedule a follow-up call to convert",
                new_due, dm.name if dm else None,
            )
            return "recorded"
        todo = memory.find_open_todo(client_eci, intent=intent, channel=channel)
        if todo:
            if reaction == "NEGATIVE":
                memory.complete_todo(todo["id"])
            else:
                retry_due = (_real_today() + dt.timedelta(days=7)).isoformat()
                memory.retry_todo(
                    todo["id"],
                    f"No response yet on {intent.replace('_', ' ')} ({channel}) -- "
                    f"retry via a different channel, or follow up again by {retry_due}",
                    retry_due,
                )
    return "recorded"


# ---------------------------------------------------------------------------
# Tools: drafting (the one place the LLM writes prose, always grounded)
# ---------------------------------------------------------------------------

@mcp.tool()
def draft_outreach(client_eci: str, intent: str, channel: str, trigger_context: dict) -> str:
    """Draft outreach text for one client/intent/channel, grounded in the
    client's profile and the skill's instruction template. trigger_context
    is the specific trigger (account/txn/note) that justified this
    opportunity -- passed through from get_clients_by_intent. Deliberately
    typed as dict, not str: FastMCP's stdio transport pre-parses any
    string argument that looks like JSON (a workaround for a Claude
    Desktop quirk), so a JSON-encoded string here gets silently turned
    into a dict before pydantic validates it against the declared type --
    tripping a validation error against `str`. Declaring the real type
    avoids fighting that pre-parse step.
    """
    from openai_client import call_llm  # local import: keeps mcp_server importable without the key set

    store = data_access.get_store()
    dm = store.get_client(client_eci)
    if dm is None:
        return json.dumps({"error": f"no client with CLIENT_ECI={client_eci}"})
    skill = get_skill(intent)
    if skill is None:
        return json.dumps({"error": f"unknown intent '{intent}'"})

    profile = json.loads(get_client_profile(client_eci))

    channel_guidance = {
        "email": "Client-facing email. Professional, warm, concise. Sign off as the covering banker.",
        "teams": "Internal Teams message to the covering banker. Short, scannable, action-oriented -- "
                 "flag the opportunity and recommend next step, not client-facing copy. Start with a "
                 "one-line header naming the banker this is for (from CONTEXT's banker_name), so it's "
                 "unambiguous this is internal even skimmed on its own.",
        "call_brief": "Internal call brief for the banker's next phone call with the client -- this "
                      "document is FOR the banker (from CONTEXT's banker_name), not the client, even "
                      "though its suggested opening line is written as first-person dialogue addressed "
                      "to the client by name (that's what the banker will actually say on the call). "
                      "Start with a one-line header making that audience explicit, e.g. 'Call brief for "
                      "{banker_name} -- call with {client_name}', so the document reads unambiguously as "
                      "internal prep even before reaching the quoted opening line. Then the opening line, "
                      "then 2-3 talking points, grounded in notes and account context.",
        "meeting_notes": "Internal meeting-prep notes for the banker's upcoming Zoom/in-person meeting "
                         "with the client -- this document is FOR the banker (from CONTEXT's "
                         "banker_name), not the client. Start with a one-line header making that "
                         "audience explicit, e.g. 'Meeting prep for {banker_name} -- meeting with "
                         "{client_name}'. Slightly more thorough than a call brief -- include "
                         "relationship context, agenda-style talking points, and a suggested close.",
    }.get(channel, "Internal note to the banker.")

    # Value-tiered personalization depth: higher-balance / higher-potential
    # clients warrant richer personalization -- not just the same
    # template with a name swapped in.
    is_high_value = profile.get("tier") in ("PLATINUM", "GOLD") or profile.get("total_aus_usd", 0) >= 25_000_000
    depth_guidance = (
        "This is a high-value relationship -- write a fuller, more personalized draft: "
        "reference specific relationship history from the notes, acknowledge the client's "
        "situation specifically (not generically), and go beyond the minimum talking points."
        if is_high_value else
        "Keep this efficient and to the point -- a shorter, templated draft is appropriate "
        "given the relationship size."
    )

    system_prompt = (
        "You are a private banking outreach assistant. You write outreach copy "
        "using ONLY the facts given to you in CONTEXT below. Never state a rate, "
        "balance, date, or fact that is not explicitly present in CONTEXT. If a "
        "detail would help but isn't in CONTEXT, omit it rather than invent it."
        f"\n\nCONTEXT:\n{json.dumps(profile, indent=2)}\n\nTRIGGER:\n{json.dumps(trigger_context)}"
        f"\n\nCHANNEL: {channel}. {channel_guidance}"
        f"\n\nPERSONALIZATION DEPTH: {depth_guidance}"
    )
    draft_text, model_used = call_llm(
        system_prompt, skill.instruction, tier=profile.get("tier"), total_aus_usd=profile.get("total_aus_usd", 0)
    )

    # lightweight hallucination guard: flag (don't silently strip) any $/%
    # figure or date in the draft that doesn't trace back to CONTEXT
    numeric_flag = _grounding_check(draft_text, profile)
    # Second, independent guard: an LLM-as-judge check for the class of
    # hallucination the regex-based check above structurally can't catch
    # -- a fabricated relationship/preference/history claim. Always-on,
    # not an opt-in eval, per an explicit decision to accept the added
    # per-draft cost/latency for the coverage (see Known Scoping Calls).
    note_flag = _note_faithfulness_check(draft_text, profile)
    guard_flag = " | ".join(f for f in (numeric_flag, note_flag) if f) or None

    draft_id = memory.save_draft(client_eci, intent, channel, draft_text, grounding_flag=guard_flag)
    return json.dumps({
        "draft_id": draft_id, "text": draft_text, "grounding_flag": guard_flag,
        "model_used": model_used, "personalization_tier": "premium" if is_high_value else "standard",
    })


def _grounding_check(draft_text: str, profile: dict) -> Optional[str]:
    """Hallucination guard: flags (does not silently strip) any dollar
    figure, percent figure, or date in the draft that doesn't trace back
    to the client's provided context. Deliberately matches ONLY figures
    with an explicit $/% marker or an ISO date pattern -- not bare
    numbers -- so it doesn't false-positive on harmless prose like "3
    talking points" or "2 accounts", while still catching every actual
    rate/dollar/date claim regardless of digit count. (Previously filtered
    by digit length instead, which let single-digit hallucinated figures
    like a fabricated "9%" rate through unflagged -- fixed after that was
    demonstrated concretely, not just suspected.)

    Dollar/percent figures compare NUMERICALLY, not by exact string, with
    a tolerance sized to each type -- exact-string matching produced a
    real, high false-positive rate against actual drafts (see
    tests/test_draft_hallucination_eval.py, 58% flagged on real
    generations, every one traced to reasonable rounding: a $3,815,941.87
    balance written as "$3,815,942", or a 4.1% rate written as "4.10%").
    Dollar figures get a $1 tolerance (rounding to the nearest whole
    dollar is normal prose, dropping cents changes the value by less than
    $1 by definition); percentages get exact numeric equality (no
    legitimate reason to round a rate, and "4.10" vs "4.1" are the same
    float regardless). A coarser rounding like "approximately $44.3
    million" still won't match and still gets flagged -- correctly, since
    that's a materially different, less precise claim than dropping cents.

    Dates match on the exact YYYY-MM-DD string CONTEXT already uses
    throughout (account end_date, transaction/note dates) -- there's no
    legitimate "rounding" of a date the way there is a dollar figure, so
    unlike numbers this is exact-string, not numeric-with-tolerance. A
    hallucinated date (wrong maturity date, invented meeting date) is
    exactly the kind of "invented client fact" a $/%-only check would
    have silently missed -- this is scoped to structured dates specifically
    because they're mechanically detectable the same way $/% are; it does
    NOT catch a hallucinated non-numeric, non-date fact (a fabricated
    relationship detail, a wrong account type) -- see Known Scoping Calls
    for why that's a materially harder, different kind of check
    (LLM-as-judge territory, not a regex)."""
    import re
    blob = json.dumps(profile)
    known_dates = set(re.findall(r"\d{4}-\d{2}-\d{2}", blob))
    # Strip date substrings before extracting known_numbers -- otherwise a
    # date's own digit groups (e.g. "09" or "15" from "2026-09-15") get
    # parsed as standalone known numbers, and a fabricated rate that
    # happens to coincide with a day/month component (e.g. a hallucinated
    # "9%" matching "09") would silently pass. Confirmed as a real
    # regression, not just a theoretical risk, when adding date coverage.
    blob_without_dates = re.sub(r"\d{4}-\d{2}-\d{2}", "", blob)
    known_numbers = set()
    for m in re.findall(r"[\d,]+\.\d+|\d+", blob_without_dates):
        known_numbers.add(float(m.replace(",", "")))
    suspects = []
    for m in re.findall(r"\$[\d,]+(?:\.\d+)?|\d+(?:\.\d+)?%", draft_text):
        is_dollar = m.startswith("$")
        cleaned = m.replace("$", "").replace(",", "").replace("%", "")
        if not cleaned:
            continue
        value = float(cleaned)
        tolerance = 1.0 if is_dollar else 0.0
        if not any(abs(value - k) <= tolerance for k in known_numbers):
            suspects.append(m)
    for m in re.findall(r"\d{4}-\d{2}-\d{2}", draft_text):
        if m not in known_dates:
            suspects.append(m)
    if suspects:
        return f"UNVERIFIED figures in draft, review before sending: {suspects}"
    return None


_NOTE_FAITHFULNESS_JUDGE_PROMPT = """You are auditing a piece of client outreach for factual
faithfulness to a client's real profile, specifically the RELATIONSHIP,
PREFERENCE, and HISTORY claims a banker's free-text notes would cover --
NOT pricing/dollar/date figures, which are checked separately by a
different, deterministic mechanism and are out of scope here.

You will be given:
1. PROFILE: the client's COMPLETE real profile as JSON -- the exact same
   context the drafting model itself was given. This includes name,
   tier, segment, banker_name, rate_sensitivity, every account (with
   fields like justification_comment, request_reason_type,
   competitor_product), transactions, and notes. A claim in DRAFT can be
   legitimately grounded in ANY of these fields, not only recent_notes --
   check the WHOLE PROFILE before concluding something is unsupported.
2. DRAFT: outreach text generated for this client.

Your job: does DRAFT state, imply, or exaggerate any RELATIONSHIP,
PREFERENCE, or HISTORY claim (rate sensitivity, stated preferences, past
reactions, personal details, relationship risk, engagement style,
justification for a prior pricing decision, etc.) that is NOT actually
supported by ANYTHING in PROFILE? Do NOT flag: dollar/percent/date
figures (checked elsewhere), or a claim that traces to any real PROFILE
field even if it's not in "recent_notes" specifically (e.g. rate
sensitivity from the top-level field, or a justification_comment on an
account). Ordinary paraphrasing or summarizing of real profile content is
fine and expected -- flag only real misrepresentation, exaggeration, or
fabrication that isn't supported ANYWHERE in PROFILE.

Be strict about SPECIFIC circumstantial details, not just overall
sentiment: a claim referencing a particular interaction (e.g. "our last
call," "when we met," "you mentioned in our conversation") must match
what PROFILE's notes actually record about that interaction -- the
channel it happened through, what it was actually about, what the
outcome actually was. A note recording a real NEGATIVE reaction on
Teams about a retention/pricing risk does NOT support a draft claiming
"our last call didn't go well when rates came up" -- the valence being
directionally right (something negative did happen) does not make a
fabricated channel or fabricated specific topic faithful. Check the
specific circumstance, not just whether similar-flavored sentiment
exists somewhere in PROFILE.

Respond ONLY with JSON, no prose, no markdown fences:
{"faithful": true or false, "issues": ["<short description>", ...]}
(issues is an empty list when faithful is true)"""


def _note_faithfulness_check(draft_text: str, profile: dict, llm_call=None) -> Optional[str]:
    """LLM-as-judge guardrail: flags any relationship/preference/history
    claim in a draft that isn't supported anywhere in the client's real
    profile -- the class of hallucination _grounding_check's $/%/date
    regex structurally cannot catch (a fabricated relationship detail, an
    invented preference, an exaggerated reaction). Unlike _grounding_check,
    this necessarily costs a real LLM call on every draft -- there's no
    deterministic way to verify a semantic paraphrase. Promoted from an
    on-demand eval (tests/test_note_faithfulness_eval.py) to an always-on
    guardrail after that eval reached a clean 100% (12/12) result -- see
    README's Known Scoping Calls for the two rounds of self-inflicted
    false positives (an incomplete judge context) that preceded that
    result, and for the explicit cost/latency tradeoff this promotion
    accepts (~+30-50% per draft, roughly the same order of magnitude as
    the eval's own observed draft-vs-judge call timing).

    llm_call is injectable (signature: (system_prompt, user_prompt) -> str)
    so this is unit-testable without a real API call -- see
    test_note_faithfulness_guard.py. Defaults to the real router-tier
    model: this is a classification task, not open-ended generation, so
    the cheap/fast tier used for intent routing is the right fit here too."""
    if llm_call is None:
        from openai_client import call_llm, ROUTER_MODEL  # local import: keeps mcp_server importable without the key set

        def llm_call(system_prompt, user_prompt):
            text, _ = call_llm(system_prompt, user_prompt, model=ROUTER_MODEL)
            return text
    user_prompt = f"PROFILE:\n{json.dumps(profile, indent=2)}\n\nDRAFT:\n{draft_text}"
    raw = llm_call(_NOTE_FAITHFULNESS_JUDGE_PROMPT, user_prompt)
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        return f"Note-faithfulness check returned an unparseable response -- review manually: {raw!r}"
    if verdict.get("faithful") is False:
        issues = verdict.get("issues", [])
        return f"Possible relationship/preference claim(s) not supported by profile, review before sending: {issues}"
    return None


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource("data://decision_makers")
def decision_makers_resource() -> str:
    """Raw joined dataset for inspection/debugging over MCP."""
    with open(data_access.DATA_DIR / "decision_makers.json") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Prompts (one per skill, sourced from the registry -- config, not code)
# ---------------------------------------------------------------------------

def _register_skill_prompts():
    for intent, skill in SKILL_REGISTRY.items():
        def make_prompt(s=skill):
            def _prompt() -> str:
                return s.instruction
            return _prompt
        mcp.prompt(name=f"draft_{intent}")(make_prompt())


_register_skill_prompts()


if __name__ == "__main__":
    mcp.run()
