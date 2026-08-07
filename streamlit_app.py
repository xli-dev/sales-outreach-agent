"""
streamlit_app.py

Ranked-list UI: single chat box in, ranked opportunity list out, click a
row to expand the drafted outreach and record a reaction. Session state
holds the current result (short-term memory); SQLite (via harness/memory)
holds everything persistent across runs.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import asyncio

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Bridge Streamlit secrets -> env vars, since the harness/MCP server run as a
# subprocess and read config via os.environ, not st.secrets.
for key in ("OPENAI_API_KEY", "ROUTER_MODEL", "DRAFT_MODEL", "PREMIUM_DRAFT_MODEL"):
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

from harness import Harness, OutreachItem  # noqa: E402
import memory  # noqa: E402
import data_access  # noqa: E402 -- only for the pinned TODAY constant, to label the
                     # ranked list with the fixed snapshot date it's scored against
import intent_router  # noqa: E402 -- only for the sidebar's on-demand routing eval
from openai_client import call_router_llm  # noqa: E402
from routing_eval_cases import CASES as ROUTING_EVAL_CASES, score_case  # noqa: E402

st.set_page_config(page_title="Deposit Sales Outreach Agent", layout="wide")
memory.init_db()


def _channel_recipient_label(channel: str, banker_name: str | None, client_name: str | None) -> str | None:
    """Mirrors mcp_server._channel_recipient_label -- kept as a separate
    copy since the UI and the MCP server are different processes (stdio
    subprocess boundary), not an in-process import. Used so the harness
    trace states the same "who this draft is actually for" fact that's
    baked into the draft's own header and the follow-up's persisted text,
    not just inside the draft itself."""
    banker_name = banker_name or "the covering banker"
    client_name = client_name or "the client"
    return {
        "teams": f"Teams message for {banker_name} -- re: {client_name}",
        "call_brief": f"Call brief for {banker_name} -- call with {client_name}",
        "meeting_notes": f"Meeting prep for {banker_name} -- meeting with {client_name}",
    }.get(channel)

# One running trace for the whole browser session, not scoped to whichever
# query happens to be the current st.session_state.result -- a draft
# triggered from the follow-ups panel (reachable before any query has ever
# been run this session) needs somewhere to log to just as much as a
# ranked-list draft does. Query-time routing/re-planning notes (from
# harness._run) get folded in here too, right after each Run click, so
# this expander is the one place that shows every LLM-driven step in
# order, regardless of where in the UI it happened.
if "trace" not in st.session_state:
    st.session_state.trace = []

with st.sidebar:
    st.subheader("Demo controls")
    st.caption(
        "This app's memory (follow-ups, sent-suppression, reactions, draft "
        "history) is one shared SQLite file. On a public deployment, every "
        "visitor hits the same instance -- use this to start clean before "
        "a live demo rather than showing a previous visitor's state."
    )
    if st.button("Reset demo data", type="primary"):
        asyncio.run(Harness.reset_memory())
        st.session_state.result = None
        st.session_state.trace = []
        st.session_state.pop("todo_items", None)
        st.success("Memory reset -- follow-ups, suppression, reactions, and draft history all cleared.")
        st.rerun()

    st.divider()
    st.caption(
        "Dev/demo diagnostic, not a product feature -- scores intent_router "
        "against a hand-labeled ask set (see routing_eval_cases.py). "
        "Makes real OpenAI calls."
    )
    if st.button("Run routing eval"):
        with st.spinner(f"Routing {len(ROUTING_EVAL_CASES)} labeled asks..."):
            eval_rows = []
            for ask, expected in ROUTING_EVAL_CASES:
                routed = intent_router.route(ask, call_router_llm)
                actual = set(routed.intents)
                eval_rows.append((ask, expected, actual, score_case(expected, actual)))
        correct = sum(1 for *_, ok in eval_rows if ok)
        st.session_state.routing_eval = (correct, eval_rows)

    if "routing_eval" in st.session_state:
        correct, eval_rows = st.session_state.routing_eval
        st.metric("Routing accuracy", f"{correct}/{len(eval_rows)}")
        with st.expander("Per-case results"):
            for ask, expected, actual, ok in eval_rows:
                icon = "✅" if ok else "❌"
                st.caption(f"{icon} \"{ask}\"  \nexpected: {sorted(expected)} · got: {sorted(actual)}")

    st.divider()
    total_drafts, flagged_drafts = memory.hallucination_stats()
    _, flagged_notes = memory.note_faithfulness_stats()
    st.caption(
        "Dev/demo diagnostic, not a product feature -- both guardrails "
        "already run on every real draft this app generates (see "
        "\"checks passed\" under each draft); this just reads that history "
        "instead of drafting fresh candidates, so it's instant and makes "
        "no API calls. For a proactive check against candidates nobody's "
        "drafted yet, run `pytest tests/test_draft_hallucination_eval.py "
        "-v -s` or `pytest tests/test_note_faithfulness_eval.py -v -s`."
    )
    if total_drafts:
        st.metric("Grounding-flag rate (all drafts, combined)", f"{flagged_drafts}/{total_drafts} flagged")
        st.metric("↳ of which, note-faithfulness issues", f"{flagged_notes}/{total_drafts} flagged")
    else:
        st.caption("No drafts generated yet this deployment.")

st.title("Deposit Sales Outreach Agent")
st.caption("Ask in plain language who to reach out to. The harness routes intent, "
           "fetches grounded context via MCP, ranks by opportunity, and drafts outreach.")

open_todos = asyncio.run(Harness.get_open_todos())
if open_todos:
    with st.expander(f"Open follow-ups ({len(open_todos)})"):
        for t in open_todos:
            # include the ECI alongside the name -- names aren't unique in
            # the mock dataset (e.g. two different "Elena Haddad" clients),
            # so the name alone can't disambiguate which client this is
            who = f"{t['client_name']} ({t['client_eci']})" if t["client_name"] else f"client {t['client_eci']}"
            # Nothing automatic happens when a due date passes -- no
            # background job, no auto-expiry -- so without this, an overdue
            # follow-up looks identical to one due next week. String
            # comparison of ISO dates is safe here (YYYY-MM-DD sorts
            # lexicographically the same as chronologically).
            # due_date is always a REAL date (real_today + 7 days, set in
            # mcp_server.log_contact) -- it must be compared against the
            # real clock here too, not data_access.TODAY (pinned to
            # 2026-08-03 for reproducible opportunity scoring). Comparing
            # against the pinned date made this permanently unreachable,
            # since due_date is never earlier than 2026-08-03.
            real_today = dt.datetime.now(dt.UTC).date().isoformat()
            is_overdue = bool(t["due_date"]) and t["due_date"] < real_today
            if is_overdue:
                st.markdown(
                    f"- **{who}** -- {t['action']} "
                    f"(:red[**⚠ OVERDUE** -- was due {t['due_date']}])"
                )
            else:
                st.write(f"- **{who}** -- {t['action']} (due {t['due_date']})")

            # This is the durable, cross-session anchor for closing the loop --
            # the ranked-list row that originally had these buttons is gone
            # the moment recently_contacted suppresses this client (up to 7
            # days), so this is often the ONLY place left to ever record what
            # happened. Reaction buttons need intent+channel to find and
            # auto-close this exact todo (see record_reaction) -- both are
            # only present on todos created after that column was added, so
            # older rows just get "Mark done" with no reaction recorded.
            if t["intent"] and t["channel"]:
                # The follow-up's own text only ever shows its LATEST state --
                # each update overwrites the last. The full history was
                # already permanently tracked in contact_log/reactions the
                # whole time; this just surfaces it instead of leaving it
                # invisible. Not scoped by channel: a retry can go out on a
                # different channel than the original send, and it's still
                # the same ongoing thread about the same opportunity.
                history = asyncio.run(Harness.get_followup_history(t["client_eci"], t["intent"]))
                if history:
                    with st.expander(f"History ({len(history)})", icon="🕑"):
                        icons = {"sent": "📤", "POSITIVE": "🙂", "NEGATIVE": "🙁", "NO_RESPONSE": "😐"}
                        for event in history:
                            label = (
                                f"Sent via {event['channel']}" if event["kind"] == "sent"
                                else f"Reaction: {event['reaction']}"
                            )
                            icon = icons.get(event["reaction"] or "sent", "•")
                            st.caption(f"{icon} {label} -- {event['at'][:10]}")

                # A follow-up often explicitly calls for a NEW outreach --
                # "retry via a different channel," "schedule a call to
                # convert" -- but until now there was no way to actually
                # draft one from here; the only path was the ranked list,
                # which this client may not even appear in anymore
                # (recently_contacted suppression, or a "convert"/"retry"
                # follow-up was never a ranked candidate to begin with).
                # Reuses draft_for_channel exactly as the ranked list does;
                # the OutreachItem here is minimal (no score/tier/segment --
                # those are ranking-display fields _draft never reads), kept
                # in session_state per todo id so the draft survives reruns.
                if "todo_items" not in st.session_state:
                    st.session_state.todo_items = {}
                if t["id"] not in st.session_state.todo_items:
                    # banker_name isn't on the todos row itself -- look it up
                    # via the same get_client_profile MCP tool the ranked
                    # list uses, so the "Call brief for {banker} -- call
                    # with {client}" label below can name the actual
                    # covering banker instead of falling back to a generic
                    # placeholder.
                    profile = asyncio.run(Harness.get_client_profile(t["client_eci"]))
                    banker_name = profile.get("banker_name", "") if "error" not in profile else ""
                    new_item = OutreachItem(
                        client_eci=t["client_eci"],
                        client_name=t["client_name"] or t["client_eci"],
                        segment="", tier="", banker_name=banker_name,
                        intent=t["intent"], channel=t["channel"],
                        score=0.0, reason=t["action"], components={},
                    )
                    # _run() applies this check eagerly to every ranked-list
                    # row, but a follow-up-driven item is built straight from
                    # a todos row and never goes through _run() at all -- so
                    # without this, that client's banker notes could say
                    # "meeting_notes" while a retry drafted from their
                    # follow-up silently defaulted to whatever channel that
                    # follow-up happened to already have, with no explanation.
                    # Record the reason whenever the hint fires at all, not
                    # only when it actually changes the channel -- if the
                    # stored channel already happens to match (e.g. it was
                    # set correctly upstream, from the ranked list's own
                    # eager check), there's still a real reason for it, and
                    # silently saying nothing looks identical to "this is
                    # just the arbitrary default," which it isn't.
                    hint = asyncio.run(Harness.get_engagement_hint(t["client_eci"]))
                    if hint.get("channel_hint"):
                        new_item.channel = hint["channel_hint"]
                        new_item.channel_override_note = hint["note"]
                    st.session_state.todo_items[t["id"]] = new_item
                todo_item = st.session_state.todo_items[t["id"]]

                if todo_item.channel_override_note:
                    st.caption(
                        f"📌 Channel **{todo_item.channel}** is based on a note on file: "
                        f"\"{todo_item.channel_override_note}\""
                    )
                todo_recipient_label = _channel_recipient_label(todo_item.channel, todo_item.banker_name, todo_item.client_name)
                if todo_recipient_label:
                    st.caption(f"📋 {todo_recipient_label}")

                chan_col, draft_col = st.columns([2, 1])
                new_channel = chan_col.selectbox(
                    "Channel",
                    options=["email", "teams", "call_brief", "meeting_notes"],
                    index=["email", "teams", "call_brief", "meeting_notes"].index(todo_item.channel)
                    if todo_item.channel in ("email", "teams", "call_brief", "meeting_notes") else 0,
                    key=f"todo_chan_{t['id']}",
                    label_visibility="collapsed",
                )
                todo_has_draft = bool(todo_item.draft_text)
                if draft_col.button(
                    "Redraft for this channel" if todo_has_draft else "Draft outreach",
                    key=f"todo_draft_btn_{t['id']}",
                ):
                    with st.spinner("Redrafting..." if todo_has_draft else "Drafting..."):
                        draft_result = Harness().draft_for_channel(
                            todo_item, new_channel, todo_item.reason, todo_item.components
                        )
                    if draft_result.get("error"):
                        st.error(f"Draft failed: {draft_result['error']}")
                    else:
                        todo_item.channel = new_channel
                        todo_item.draft_text = draft_result.get("text")
                        todo_item.draft_id = draft_result.get("draft_id")
                        todo_item.grounding_flag = draft_result.get("grounding_flag")
                        todo_item.model_used = draft_result.get("model_used")
                        todo_item.personalization_tier = draft_result.get("personalization_tier")
                        todo_item.channel_override_note = None  # banker made an explicit choice; the auto-note no longer applies
                        # Same per-draft audit note as the ranked list (see
                        # below) -- a draft triggered from the follow-ups
                        # list is just as real an action and belongs in the
                        # same trace, not a second, invisible audit path.
                        # Goes to the session-wide trace (not any specific
                        # query's result.notes), since the follow-ups panel
                        # is reachable before any query has ever been run.
                        recipient_label = _channel_recipient_label(todo_item.channel, todo_item.banker_name, todo_item.client_name)
                        subject = recipient_label if recipient_label else f"email for {todo_item.client_name}"
                        st.session_state.trace.append(
                            f"{'Redrafted' if todo_has_draft else 'Drafted'} {subject} "
                            f"({todo_item.intent}, {todo_item.channel}) using {todo_item.model_used} "
                            f"[{todo_item.personalization_tier} personalization] (from follow-ups)"
                        )
                    st.rerun()

                if todo_item.draft_text:
                    st.text_area(
                        "Drafted outreach", todo_item.draft_text, height=150,
                        key=f"todo_draft_text_{t['id']}",
                    )
                    if todo_item.model_used:
                        st.caption(f"Drafted with {todo_item.model_used} ({todo_item.personalization_tier} personalization)")
                    if todo_item.grounding_flag:
                        st.error(f"Grounding check: {todo_item.grounding_flag}")
                    else:
                        st.caption("✓ Grounding + note-faithfulness checks passed")

                    todo_send_blocked = bool(todo_item.grounding_flag)
                    todo_confirm = True
                    if todo_send_blocked:
                        todo_confirm = st.checkbox(
                            "I've reviewed the flagged figures and confirm this draft is accurate",
                            key=f"todo_override_{t['id']}",
                        )
                    # This retry is, by definition, contacting a client who
                    # was already contacted recently -- that's the entire
                    # reason this follow-up exists. So this WILL trip the
                    # already-contacted guardrail almost every time, unlike
                    # the ranked list where it's the exception. Same
                    # confirm-then-retry pattern as there, just expected to
                    # fire routinely here rather than rarely.
                    todo_recent_key = f"todo_recent_block_{t['id']}"
                    todo_recently_blocked = st.session_state.get(todo_recent_key, False)
                    todo_confirm_recent = True
                    if todo_recently_blocked:
                        st.warning(
                            "**Send blocked** -- this client was already contacted within the "
                            "last 7 days (expected, since that's why this follow-up exists). "
                            "Confirm this retry is intentional."
                        )
                        todo_confirm_recent = st.checkbox(
                            "I've confirmed this retry should still go out",
                            key=f"todo_override_recent_{t['id']}",
                        )
                    if st.button(
                        "Mark sent", key=f"todo_sent_{t['id']}",
                        disabled=not todo_confirm or (todo_recently_blocked and not todo_confirm_recent),
                    ):
                        log_result = asyncio.run(Harness.log_contact(
                            todo_item.client_eci, todo_item.intent, todo_item.channel,
                            draft_id=todo_item.draft_id, override_grounding_flag=todo_send_blocked,
                            override_recent_contact=todo_recently_blocked and todo_confirm_recent,
                            source_todo_id=t["id"],
                        ))
                        if log_result.get("already_contacted"):
                            st.session_state[todo_recent_key] = True
                            st.rerun()
                        elif log_result.get("error"):
                            st.error(log_result["error"])
                        else:
                            st.session_state.pop(todo_recent_key, None)
                            st.success("Logged -- this follow-up's text and due date are refreshed above, "
                                       "not closed; it's now waiting on this new outreach.")
                            del st.session_state.todo_items[t["id"]]
                            st.rerun()

                done_col, pos_col, nr_col, neg_col = st.columns(4)
                if done_col.button("Mark done", key=f"todo_done_{t['id']}"):
                    asyncio.run(Harness.complete_todo(t["id"]))
                    st.rerun()
                if pos_col.button("Reaction: Positive", key=f"todo_pos_{t['id']}"):
                    asyncio.run(Harness.record_reaction(
                        t["client_eci"], t["channel"], "POSITIVE", intent=t["intent"]
                    ))
                    # st.toast survives the rerun below, unlike st.success --
                    # worth using here since the three outcomes now behave
                    # differently and that's not obvious from the click alone
                    st.toast("Closed, and a new follow-up was created to schedule a call and convert.")
                    st.rerun()
                if nr_col.button("Reaction: No response", key=f"todo_nr_{t['id']}"):
                    asyncio.run(Harness.record_reaction(
                        t["client_eci"], t["channel"], "NO_RESPONSE", intent=t["intent"]
                    ))
                    st.toast("Not closed -- no response isn't a resolved outcome. Due date pushed for a retry.")
                    st.rerun()
                if neg_col.button("Reaction: Negative", key=f"todo_neg_{t['id']}"):
                    asyncio.run(Harness.record_reaction(
                        t["client_eci"], t["channel"], "NEGATIVE", intent=t["intent"]
                    ))
                    st.toast("Closed -- nothing further to pursue right now.")
                    st.rerun()
            else:
                if st.button("Mark done", key=f"todo_done_{t['id']}"):
                    asyncio.run(Harness.complete_todo(t["id"]))
                    st.rerun()
            st.divider()

if "result" not in st.session_state:
    st.session_state.result = None

CHANNEL_LABELS = {
    "auto": "Auto (use each intent's default)",
    "email": "Email",
    "teams": "Teams message",
    "call_brief": "Call brief",
    "meeting_notes": "Meeting-notes prep (Zoom)",
}

with st.form("ask_form"):
    ask = st.text_input(
        "What do you need?",
        placeholder="e.g. Who should I reach out to this week, and draft the outreach",
    )
    channel_pref = st.selectbox(
        "Channel preference",
        options=list(CHANNEL_LABELS.keys()),
        format_func=lambda k: CHANNEL_LABELS[k],
        help="Overrides every intent's default channel for this request. "
             "You can still override a single client's channel below after the list is generated.",
    )
    submitted = st.form_submit_button("Run")

if submitted and ask:
    with st.spinner("Routing intent, fetching candidates, scoring, drafting..."):
        harness = Harness()
        override = None if channel_pref == "auto" else channel_pref
        st.session_state.result = harness.run(ask, channel_override=override)
        # Fold this run's routing/re-planning notes into the session-wide
        # trace (see its init above) rather than leaving them isolated on
        # this one result object -- a later query replacing
        # st.session_state.result would otherwise silently drop them from
        # view instead of just adding to the running record.
        st.session_state.trace.extend(st.session_state.result.notes)

result = st.session_state.result

if st.session_state.trace:
    with st.expander("Harness trace (planning / re-planning / drafting)"):
        for n in st.session_state.trace:
            st.write(f"- {n}")

if result:
    if result.empty_intents:
        st.warning(f"No matches found for: {', '.join(result.empty_intents)} (even after widening the window)")

    if not result.items:
        st.info("No opportunities found for this request.")
    else:
        st.subheader(f"Ranked opportunities ({len(result.items)})")
        st.caption(
            f"Scored as of {data_access.TODAY.isoformat()} -- a fixed snapshot of the "
            "deposit book, not today's real date, so results are reproducible. "
            "Follow-ups and history below use real timestamps."
        )
        # Every per-row widget key below is scoped by run_id (this result
        # object's identity), not just the row index i. Streamlit persists
        # keyed-widget state (selectbox index, checkbox value, text_area
        # value) in session_state across ALL reruns, including a brand new
        # "Run" click with completely different candidates -- a bare
        # f"chan_{i}" key means row 0 in a fresh exception_expiry query
        # inherits whatever channel was left selected for row 0 of the
        # PREVIOUS query (e.g. a maturity item defaulting to "email"),
        # showing "email" for a client whose actual default is "teams".
        # Confirmed live: same bug would let a stale checked
        # override_/override_recent_ checkbox silently satisfy "Mark
        # sent"'s guard for a new, unrelated flagged draft. run_id changes
        # every time a new query replaces st.session_state.result, so old
        # keys are simply orphaned instead of leaking in; it stays constant
        # across reruns of the SAME result (e.g. after clicking "Draft
        # outreach"), so a banker's in-progress choices aren't lost either.
        run_id = id(result)
        for i, item in enumerate(result.items):
            header = (
                f"{'✓ SENT · ' if item.sent else ''}#{i+1} · {item.client_name} "
                f"({item.tier}, {item.segment}) · {item.intent.replace('_', ' ')} · "
                f"score {item.score} · via {item.banker_name}"
            )
            with st.expander(header):
                st.write(f"**Why:** {item.reason}")
                cols = st.columns(len(item.components))
                for col, (k, v) in zip(cols, item.components.items()):
                    col.metric(k, f"{v:.0f}")

                if item.channel_override_note:
                    st.caption(
                        f"📌 Channel **{item.channel}** is based on a note on file: "
                        f"\"{item.channel_override_note}\""
                    )

                chan_col1, chan_col2 = st.columns([2, 1])
                new_channel = chan_col1.selectbox(
                    "Channel",
                    options=["email", "teams", "call_brief", "meeting_notes"],
                    index=["email", "teams", "call_brief", "meeting_notes"].index(item.channel)
                    if item.channel in ("email", "teams", "call_brief", "meeting_notes") else 0,
                    key=f"chan_{run_id}_{i}",
                    label_visibility="collapsed",
                )
                has_draft = bool(item.draft_text)
                button_label = "Redraft for this channel" if has_draft else "Draft outreach"
                if chan_col2.button(button_label, key=f"draft_btn_{run_id}_{i}"):
                    with st.spinner("Redrafting..." if has_draft else "Drafting..."):
                        harness = Harness()
                        draft_result = harness.draft_for_channel(item, new_channel, item.reason, item.components)
                        if draft_result.get("error"):
                            st.error(f"{'Redraft' if has_draft else 'Draft'} failed: {draft_result['error']}")
                        else:
                            item.channel = new_channel
                            item.draft_text = draft_result.get("text")
                            item.draft_id = draft_result.get("draft_id")
                            item.grounding_flag = draft_result.get("grounding_flag")
                            item.draft_error = None
                            item.channel_override_note = None  # banker made an explicit choice; the auto-note no longer applies
                            item.model_used = draft_result.get("model_used")
                            item.personalization_tier = draft_result.get("personalization_tier")
                            # Per-draft audit note, same spirit as the old
                            # eager loop's trace entry -- just logged now,
                            # at click time, since drafting itself moved here.
                            # Goes to the session-wide trace, alongside this
                            # query's own routing notes and any follow-up-
                            # driven drafts, so it's one ordered record of
                            # every LLM-driven step regardless of where in
                            # the UI it happened.
                            recipient_label = _channel_recipient_label(item.channel, item.banker_name, item.client_name)
                            subject = recipient_label if recipient_label else f"email for {item.client_name}"
                            st.session_state.trace.append(
                                f"{'Redrafted' if has_draft else 'Drafted'} {subject} "
                                f"({item.intent}, {item.channel}) using {item.model_used} "
                                f"[{item.personalization_tier} personalization]"
                            )
                    st.rerun()

                if item.draft_error:
                    st.error(f"Drafting failed for this client: {item.draft_error}")
                elif item.draft_text:
                    st.text_area("Drafted outreach", item.draft_text, height=180, key=f"draft_{run_id}_{i}")
                    if item.model_used:
                        st.caption(f"Drafted with {item.model_used} ({item.personalization_tier} personalization)")
                    if item.grounding_flag:
                        st.error(f"Grounding check: {item.grounding_flag}")
                    else:
                        st.caption("✓ Grounding + note-faithfulness checks passed")
                else:
                    st.info("Not drafted yet -- click \"Draft outreach\" above to generate it for this client.")

                # Sending a draft that doesn't exist yet makes no sense --
                # gate this on a draft actually being present. Reactions
                # aren't offered here at all: recording one only makes sense
                # once something's been sent (that's what creates the
                # follow-up now), and the durable, cross-session place to
                # close that loop is the Open follow-ups list above, not
                # this ephemeral ranked-list row -- see log_contact/
                # record_reaction and the follow-ups section.
                if item.sent:
                    # Kept visible rather than removed from the ranked list --
                    # a banker working through a long list benefits from
                    # seeing what they've already done this session, not
                    # having rows disappear as they go. The header's
                    # "✓ SENT" prefix is visible even collapsed; this is the
                    # expanded-view confirmation.
                    st.success(
                        "✓ Sent this session -- suppressed from future ranking for 7 days. "
                        "Follow-up is in Open follow-ups above; record the outcome there."
                    )
                elif has_draft:
                    send_blocked = bool(item.grounding_flag)
                    if send_blocked:
                        st.warning(
                            "**Send blocked** -- this draft has unverified figures flagged above. "
                            "Review the draft, then check the box to confirm it's safe to send."
                        )
                        confirm_override = st.checkbox(
                            "I've reviewed the flagged figures and confirm this draft is accurate",
                            key=f"override_{run_id}_{i}",
                        )
                    else:
                        confirm_override = True

                    # "Already contacted recently" can only be discovered by
                    # actually trying to send -- unlike the grounding flag,
                    # it depends on shared state (someone else's action)
                    # that isn't known until log_contact checks it
                    # server-side. Remember the block across the rerun it
                    # takes to show this checkbox, then retry with the
                    # override once confirmed.
                    recent_block_key = f"recent_contact_block_{run_id}_{i}"
                    recently_blocked = st.session_state.get(recent_block_key, False)
                    confirm_recent = True
                    if recently_blocked:
                        st.warning(
                            "**Send blocked** -- this client was already contacted within the "
                            "last 7 days, possibly by someone else since this list was "
                            "generated. Confirm this outreach is still needed."
                        )
                        confirm_recent = st.checkbox(
                            "I've confirmed this client should still be contacted",
                            key=f"override_recent_{run_id}_{i}",
                        )

                    if st.button(
                        "Mark sent", key=f"sent_{run_id}_{i}",
                        disabled=not confirm_override or (recently_blocked and not confirm_recent),
                    ):
                        log_result = asyncio.run(
                            Harness.log_contact(
                                item.client_eci, item.intent, item.channel,
                                draft_id=item.draft_id, override_grounding_flag=send_blocked,
                                override_recent_contact=recently_blocked and confirm_recent,
                            )
                        )
                        if log_result.get("already_contacted"):
                            st.session_state[recent_block_key] = True
                            st.rerun()
                        elif log_result.get("error"):
                            st.error(log_result["error"])
                        else:
                            st.session_state.pop(recent_block_key, None)
                            item.sent = True
                            st.success("Logged -- suppressed from future ranking for 7 days. "
                                       "Follow-up created in Open follow-ups above; record the outcome there.")
                            st.rerun()
else:
    st.info("Enter a request above to generate a ranked outreach list.")
