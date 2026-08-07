# Deposit Sales Outreach Agent

Agentic system for deposit sales outreach: one chat-box ask in, a ranked,
actionable target list out, with per-client drafted outreach and tracked
follow-ups.

## Contents

1. [Orchestration Loop](#s1)
2. [Tool Design](#s2)
3. [Error Handling](#s3)
4. [Context Management](#s4)
5. [State Management](#s5)
6. [Guardrails & Permissions](#s6)
7. [Audit Trail Design](#s7)
8. [Repo Layout](#repo-layout)
9. [Running Locally](#running-locally)
10. [Deployment](#deployment)
11. [Known Scoping Calls](#known-scoping-calls-documented-rather-than-silently-made)

```
User ask --> Intent Router (LLM, gpt-5.4-mini) --> Skill Registry (config)
   --> Harness (deterministic orchestrator, gather-context/act/verify)
   --> MCP Server (tools: data access, scoring, drafting, memory)
   --> Ranked list + drafts --> Streamlit UI (click-to-expand)
```

<a name="s1"></a>
## 1. Orchestration Loop

`src/harness.py` is the entire agent loop. The model is invoked at exactly
three call sites: intent classification (`intent_router.route`), outreach
drafting, and note-faithfulness judging -- the latter two both inside
`draft_outreach` on the MCP server (see §3/§6). From the harness's own
point of view it only ever triggers the model in two places -- `route()`
and one `draft_outreach` call per row -- the second model call inside
`draft_outreach` is an implementation detail of that tool. Everything else
in this file is plain deterministic Python deciding what to call, in what
order, and how to react to what comes back.

```
1. GATHER-CONTEXT  route(user_ask) -> intent(s) + params (client/banker/timeframe)
2. ACT              per intent, call get_clients_by_intent via that skill's
                     fetch_tools binding
3. VERIFY / RE-PLAN  empty result? widen horizon 2x, retry once; still
                     empty -> report as empty, never fabricate a candidate
4. MERGE + RANK      pool all (intent, client) candidates, sort by score
                     across the whole requested funnel, not per-intent
5. RETURN            the full ranked list comes back immediately -- every
                     row shows score + reason, no drafting done yet
6. DRAFT (lazy)      draft_outreach only runs when a banker expands a
                     specific row and clicks "Draft outreach" -- one
                     client at a time, on demand, never eagerly for a
                     fixed top-N (see Known Scoping Calls)
7. PERSIST           harness never writes memory directly -- it calls
                     log_contact / record_reaction / complete_todo as MCP
                     tools, same as any other action. Follow-ups are
                     created at SEND time (log_contact), not draft time --
                     a draft nobody sends needs no follow-up
```

The harness talks to `mcp_server.py` exclusively as an MCP client over
stdio -- never an in-process import. That is what makes "tool execution
through a harness, over MCP" literally true rather than a shortcut: swap
the server for a real deposits API tomorrow and the harness/UI don't change.

<a name="s2"></a>
## 2. Tool Design

Everything is exposed as MCP tools/resources/prompts (`mcp.server.fastmcp`).
`mcp_server.py` is the *only* module that touches the CSVs, the OpenAI
client, or the SQLite memory store.

**Tools**
| Tool | Purpose |
|---|---|
| `get_clients_by_intent(intent, timeframe_days, banker_or_team)` | Scored, ranked candidates for one intent -- grounding layer, no invented candidates |
| `get_client_profile(client_eci)` | Full joined profile (DM + accounts + recent txns + recent notes + open todos) -- the context passed into drafting |
| `get_banker_notes(client_eci, topic)` | Notes filtered by topic |
| `draft_outreach(client_eci, intent, channel, trigger_context)` | The one place the LLM writes prose; returns text + grounding flag + model used + personalization tier |
| `get_engagement_channel_hint(client_eci)` | Deterministic keyword match over the banker notes on file for that client, tagged RELATIONSHIP; overrides a skill's static `default_channel` when those notes give a clear, actionable signal (see §4) |
| `log_contact(client_eci, intent, channel, draft_id, override_grounding_flag, override_recent_contact, source_todo_id)` | Records a send; enforces two guardrails (unresolved grounding flag, client already contacted within 7 days) unless explicitly overridden; creates/updates the follow-up (see §6) |
| `record_reaction(client_eci, channel, reaction, note, intent)` | Records an outcome; NEGATIVE closes the follow-up outright, POSITIVE closes it but opens a new `kind="convert"` follow-up, anything else (NO_RESPONSE, NEUTRAL) just pushes the due date rather than closing |
| `get_followup_history(client_eci, intent, channel)` | Merged, chronologically-sorted send + reaction history for one follow-up thread -- backs the UI's "History" expander |
| `get_open_todos(client_eci)` | Lists open follow-ups, optionally filtered to one client -- backs the UI's "Open follow-ups" list |
| `create_todo` / `complete_todo` / `recently_contacted` / `reset_memory` | Memory read/write -- `reset_memory` wipes all four tables in one call, for resetting a shared/public deployment between demos |

**Resources**: `data://decision_makers` -- the raw joined dataset, for
inspection/debugging over MCP.

**Prompts**: one per skill (`draft_maturity`, `draft_exception_expiry`,
etc.), sourced directly from `skills.SKILL_REGISTRY` so the drafting
instruction is the same text whether you're inspecting it over MCP or it's
actually driving a draft -- not two copies to keep in sync.

Tool granularity is deliberately coarse (one call fetches all candidates
for an intent, not one call per client) so the harness's tool-call count
scales with intents-requested, not clients-in-dataset.

<a name="s3"></a>
## 3. Error Handling

- **Empty/stale results**: `get_clients_by_intent` returning `[]` triggers
  one re-plan -- widen the horizon to 2x and retry -- before the intent is
  reported as genuinely empty. Never silently dropped, never fabricated.
- **Malformed routing output**: if the router LLM returns invalid JSON or
  an intent outside the registry, `intent_router.route` retries once with a
  stricter prompt; if that still fails, it falls back to *all* intents
  (full-funnel) rather than failing closed and returning nothing.
- **Unknown intent at the tool layer**: `get_clients_by_intent` returns a
  structured `{"error": ...}` payload rather than raising, so a bad routing
  decision degrades gracefully instead of crashing the harness mid-request.
- **Grounding failures aren't silently corrected**: `_grounding_check`
  flags (doesn't strip or auto-fix) any $/%/date figure in a draft that
  doesn't trace back to the client's provided context, and
  `_note_faithfulness_check` (an LLM-as-judge) separately flags any
  relationship/preference/history claim the profile doesn't support -- both
  surfaced to the banker as a visible warning rather than silently edited.
- **Concurrent sends for the same opportunity**: a naive "find the open
  follow-up, then update or create" sequence in `log_contact` was a
  confirmed race (reproduced with two simultaneous threads -> 2 duplicate
  follow-ups). Fixed with a single `BEGIN IMMEDIATE` SQLite transaction
  (`memory.upsert_sent_todo`) that acquires the write lock before the check
  runs; re-running the same reproduction after the fix gives exactly one row.
- **Already-past-due accounts were silently excluded from ranking**:
  `clients_with_exception_expiring`/`clients_with_maturity_in_window`
  required `0 <= delta <= days`, dropping any account whose `end_date` had
  already passed -- even though `scoring._urgency_from_days` treats that as
  *maximum* urgency. Fixed by dropping the lower bound (`delta <= days`);
  see `tests/test_expiry_eligibility.py` for the regression coverage
  (synthetic accounts, since the provided dataset has no past-due rows).
- **The grounding-flag check itself had a real false-positive rate**:
  hand-crafted test strings proved the checker catches a hallucination when
  one exists, but not how often it wrongly flags a *correct* figure.
  Running real `draft_outreach` calls against real clients
  (`tests/test_draft_hallucination_eval.py`) found a 58% flag rate (7/12),
  every one traced to normal rounding (e.g. a 4.1% rate written as
  "4.10%") failing an exact-string match. Fixed by comparing numerically
  instead, with a $1 tolerance for dollars and exact equality for
  percentages. Re-run after the fix: 0% (0/12).
- **`_grounding_check` extended to dates, then immediately found its own
  regression**: a hallucinated maturity/meeting date is exactly the kind of
  invented fact a $/%-only check would miss, so dates got the same
  treatment (exact-string match, no rounding tolerance). Adding it broke an
  existing test: a date's own digit groups (e.g. "09" from "2026-09-15")
  were being parsed as standalone known numbers by the same regex used for
  $/% grounding, letting a fabricated "9%" coincidentally match. Fixed by
  stripping date substrings from the profile blob before extracting known
  numbers -- caught within the same session it was introduced.
- **Non-numeric fact hallucination now has a permanent guardrail too**:
  `_note_faithfulness_check` runs an LLM-as-judge on every `draft_outreach`
  call, checking for any relationship/preference/history claim not
  supported by the client's real profile -- the class of hallucination a
  $/%/date regex structurally can't catch. This roughly doubles the LLM
  calls per draft (+30-50% latency/cost), accepted explicitly for the
  coverage. It started as an on-demand eval to gather evidence before
  committing to that ongoing cost, and was only promoted to always-on after
  reaching a clean result (next bullet).
- **Building that eval surfaced two rounds of its own false positives**:
  given only raw notes text, the judge "found" an 83% (10/12)
  misrepresentation rate -- e.g. flagging the client's real covering
  banker's name as an invented identity, simply because the judge wasn't
  shown that field. Given a hand-picked profile summary instead of the full
  profile, still 50% (6/12) for the same reason at a smaller scale. Both
  were bugs in the eval's own context, not in drafting -- fixed by giving
  the judge the exact same full profile JSON `draft_outreach` itself
  receives. Result after that fix: **100% (12/12) faithful**, which
  justified promoting it to the permanent guardrail above (`§3`); both now
  share one prompt in `mcp_server.py`. Fast, mocked unit coverage of the
  guardrail's own logic lives in `tests/test_note_faithfulness_guard.py`,
  independent of real API calls.
- **"100% faithful" alone didn't prove the judge worked**: that number only
  measures specificity (no false positives), not recall (does it actually
  catch a real violation) -- a judge that always says "faithful" would
  score the same 100%. Validated recall by deliberately injecting 3 known
  fabricated claims into real drafts
  (`test_note_faithfulness_catches_injected_hallucinations`). First run:
  2/3 caught -- the miss matched the *valence* of a real note (something
  negative did happen) without checking the specific circumstance was
  fabricated. Tightened the prompt to require specific-circumstance
  matching: fixed recall to 3/3, but cost precision -- the specificity eval
  dropped from 12/12 to 10/12. Decision: kept the stricter prompt. A missed
  hallucination reaching a client is worse than an occasional unnecessary
  review prompt, so recall was prioritized over precision deliberately.

<a name="s4"></a>
## 4. Context Management

**In context for drafting**: the single client's full profile (accounts
with rates/exception terms, last 10 transactions, last 10 notes, open
todos), the specific trigger that justified the opportunity, channel-
specific guidance (email vs. Teams vs. call brief vs. meeting-notes prep
read differently), and a personalization-depth instruction gated on client
tier/AUS.

**Channel selection is three-tier, in priority order**: (1) a banker's
explicit request-level channel preference, if set, always wins outright;
(2) absent that, `get_engagement_channel_hint` checks that client's banker
notes tagged RELATIONSHIP for a real signal (currently: "prefers a short
call over email" -> `call_brief`, "likes an in-person quarterly review" ->
`meeting_notes`) and overrides the skill's default when it disagrees --
logged in the harness trace and surfaced in the UI (📌 badge); (3)
otherwise, the skill's static `default_channel` in `skills.py` applies. One
free-text pattern ("wants concise Teams messages") is deliberately *not*
mapped: this system's `teams` channel produces an internal nudge to the
banker, not client-facing copy, so honoring that note literally would
misrepresent what the system actually sends.

Each skill's `instruction` template is deliberately channel-*neutral* -- it
states only the facts and the goal, never the medium; format and audience
come entirely from `channel_guidance` in `mcp_server.py`. This was a real
bug until the channel-hint feature exposed it: templates used to bake in an
assumed channel ("drafting a retention **email**...") that happened to
always match the skill's default -- invisible until the engagement hint
legitimately picked a different channel, at which point the model followed
the instruction's hardcoded medium over the channel guidance and produced
client-facing email copy under a `meeting_notes` label.

**Deliberately kept out of context**: the other 99 clients' data (each
drafting call is scoped to one `client_eci`, fetched fresh -- never the
full dataset dump), and the router's context is symmetrically minimal
(just the ask + the intent taxonomy, no client data at all). The router and
drafting also use different model tiers (`call_llm` takes a `model`
override) since the context size and task shape differ enough that one
config for both would be wrong in either direction.

<a name="s5"></a>
## 5. State Management

Two tiers, for managing memory across requests:

- **Short-term (session)**: the current ranked list, drafts, and harness
  trace live in Streamlit `session_state` -- gone when the session ends,
  not written to disk.
- **Persistent (cross-session)**: SQLite (`data/memory.sqlite3`,
  gitignored), four tables --
  - `contact_log`: suppresses recently-contacted clients from future
    `get_clients_by_intent` results (7-day default window)
  - `todos`: follow-ups, created at SEND time (`log_contact`), not draft
    time -- a draft nobody sends needs no follow-up. Tagged with
    `intent`/`channel` so a client with multiple live opportunities gets
    one follow-up each, not one shared row. Stays OPEN through "sent,
    waiting" and only closes once an outcome is recorded via
    `record_reaction`, or a banker explicitly clicks "Mark done." Tagged
    with a `kind` column (`outreach` vs. `convert`) so a POSITIVE
    reaction's new conversion follow-up can't collide with the original
    "sent, waiting" row for the same `(client_eci, intent, channel)`.
  - `reactions`: every recorded reaction, timestamped -- the *most recent*
    reaction per client is read back by `scoring.py` and overrides the
    static CSV reaction in the engagement component, closing the funnel
    loop (send -> respond -> convert). This changes that client's score the
    next time they're eligible to be ranked again -- after the 7-day
    contact-suppression window clears, not on the very next query, since
    `recently_contacted` (see §6) deliberately blocks the same client from
    reappearing immediately regardless of how they reacted.
  - `draft_history`: every draft ever generated, for audit (see §7)

Chosen over a vector/graph store because retrieval here is exact-match over
structured fields on ~100 clients / ~366 notes -- a vector index over note
text is a reasonable next step if note volume grew large enough for
semantic search to matter; noted as a scoping call, not built here.

<a name="s6"></a>
## 6. Guardrails & Permissions

- **No auto-send**: the system drafts; it never contacts a client directly.
  A banker must explicitly click "Mark sent" in the UI, which is the only
  action that logs to `contact_log` and suppresses the client going
  forward. Drafting and sending are deliberately separate permissions.
- **Grounding, enforced three ways**: the drafting system prompt instructs
  the model to use *only* supplied facts and omit rather than invent
  missing details; `_grounding_check` independently re-verifies every
  $/%/date figure against that same context as a deterministic, code-level
  check; and `_note_faithfulness_check` separately verifies (via a second
  LLM, acting as judge) that no relationship/preference/history claim goes
  beyond what the profile supports. None trust the prompt instruction
  alone -- the second and third are independent checks specifically
  because prompt-following isn't a guarantee.
- **Blocks sending to an already-contacted client, not just filtering it
  from the list**: on a shared deployment, two bankers can each hold the
  same client in their own already-stale fetched list. `log_contact`
  re-checks `recently_contacted` at the actual moment of sending and
  refuses unless `override_recent_contact` is explicitly set.
- **No direct data/API access outside the MCP boundary**: the harness (and
  therefore the Streamlit UI, for every real product feature) has no
  filesystem or OpenAI client access of its own -- every read or write
  goes through `mcp_server.py`'s tool surface, including the "Open
  follow-ups" list and the banker-name lookup for a follow-up-driven
  draft (both used to import `memory`/`data_access` directly -- a real
  gap, closed by adding `get_open_todos` and routing that lookup through
  the existing `get_client_profile` tool). Swapping in a real deposits
  API means changing one module. The one documented exception: the
  sidebar's grounding-flag stats read `draft_history` directly, since
  they're explicitly a dev/demo diagnostic, not a product feature (see
  Deployment).
- **Secrets never touch the repo**: `OPENAI_API_KEY` (and model overrides)
  are read from environment/Streamlit-secrets only, explicitly forwarded
  into the MCP server's subprocess environment (stdio transport doesn't
  inherit the parent's env by default), and `data/memory.sqlite3` /
  `.streamlit/secrets.toml` / `.env` are gitignored.

<a name="s7"></a>
## 7. Audit Trail Design

Two complementary audit surfaces:

- **Per-request trace** (`HarnessResult.notes`, the UI's "Harness trace"
  expander): a human-readable log of what the harness decided and why --
  which intents it routed to, every re-planning widen-and-retry, every
  engagement-based channel override -- appended at ranking time, with each
  draft's "drafted using `gpt-5.4`/premium `gpt-5.5` [tier]" line appended
  later, the moment that specific draft happens. Nothing here is a debug
  flag; nothing to strip out before a demo.
- **Persistent history** (`memory.draft_history`, `contact_log`,
  `reactions`): every draft, send, and recorded reaction is timestamped and
  kept indefinitely in SQLite, independent of the UI trace -- so "what did
  we tell this client, when, and how did they react" is answerable outside
  any single session.

<a name="repo-layout"></a>
## Repo Layout

```
src/
  data_access.py     # loads + joins the CSVs; no LLM, no invented facts
  scoring.py          # opportunity score (pure function)
  memory.py            # SQLite persistence; hallucination_stats() reads
                        # draft_history for the sidebar's live grounding-flag rate
  skills.py            # skill registry (declarative)
  intent_router.py     # LLM-based intent classification
  openai_client.py     # thin OpenAI wrapper, model config + tiering
  routing_eval_cases.py # labeled ask set, shared by the pytest eval and the
                         # sidebar diagnostic so they can't drift apart
  hallucination_eval_cases.py # samples real top-scoring candidates per intent
                               # live from the dataset, for the proactive eval
  mcp_server.py         # MCP tools/resources/prompts -- the only I/O layer
  harness.py            # orchestrator; MCP client; agent loop
streamlit_app.py         # ranked-list UI
tests/
  test_scoring.py          # opportunity-score unit tests
  test_guardrails.py        # grounding-flag block enforced at the tool layer
  test_grounding.py          # hallucination-guard regex behavior
  test_harness_recovery.py    # re-plan/fallback paths (empty results, bad routing)
  test_engagement_channel.py   # engagement_channel_hint pattern matching
  test_expiry_eligibility.py   # already-past-due accounts stay eligible/max-urgency
  test_routing_accuracy.py      # LLM routing-accuracy eval, hand-labeled ask set --
                                  # not a unit test, real API calls, skips without a key
  test_draft_hallucination_eval.py # real drafts against real clients, scored by
                                     # _grounding_check's flag rate
  test_note_faithfulness_eval.py # broader real-API evidence for
                                   # _note_faithfulness_check; skips without a key
  test_note_faithfulness_guard.py # fast, mocked unit tests for that guardrail's
                                    # own logic -- no API calls
.github/workflows/ci.yml   # test on every push
data/                       # provided mock CSVs + JSON (memory.sqlite3 gitignored)
```

<a name="running-locally"></a>
## Running Locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml with your OPENAI_API_KEY

streamlit run streamlit_app.py
```

Try: *"Who should I reach out to this week, and draft the outreach"*

<a name="deployment"></a>
## Deployment

- **Streamlit Community Cloud**: point at this repo, `streamlit_app.py` as
  the entrypoint, set `OPENAI_API_KEY` (and optionally `ROUTER_MODEL` /
  `DRAFT_MODEL` / `PREMIUM_DRAFT_MODEL`) in the app's Secrets settings.
- **CI**: `.github/workflows/ci.yml` runs `pytest` and a module smoke-import
  on every push/PR to `main`. `test_routing_accuracy.py` self-skips in CI
  (no `OPENAI_API_KEY` secret configured there).
- **Every visitor to a Community Cloud deployment hits the same running
  instance** -- `data/memory.sqlite3` is one shared file, not per-visitor,
  so one person's follow-ups/suppression/reactions are visible to the
  next. The sidebar's "Demo controls" -> "Reset demo data" button
  (`reset_memory`) wipes all four tables in one click, from the browser --
  use it before presenting live.

<a name="known-scoping-calls-documented-rather-than-silently-made"></a>
## Known Scoping Calls (documented rather than silently made)

- **Evaluation splits into two tiers, by determinism**: the main suite
  (scoring, guardrails, grounding, harness recovery, engagement-channel,
  expiry-eligibility) is fully deterministic -- exact-match assertions, no
  API key, safe on every push. Routing and drafting are LLM calls and get a
  separate treatment: `test_routing_accuracy.py` is a hand-labeled ~10-case
  set (current result: **10/10, 100%**), and
  `test_draft_hallucination_eval.py` samples real top-scoring candidates
  live from the dataset and drafts them for real (current result after the
  false-positive fix in §3: **0%, 0/12**). Both self-skip without
  `OPENAI_API_KEY` rather than blocking every push on a paid API call. The
  sidebar's grounding-flag metric reads `draft_history` directly instead
  (`memory.hallucination_stats()`) -- an instant read of every real draft
  this deployment has generated, not a fresh eval run. Draft *quality*
  beyond groundedness (well-written, appropriately toned) has no automated
  check here -- the honest next step would be an LLM-as-judge pass, not
  built for this scope.
- **Two intentionally different clocks, never reconciled**: `TODAY` is
  pinned to 2026-08-03 in `data_access.py` (matching the provided
  dataset's `as_of_date`) for reproducible scoring against the static mock
  data -- not `datetime.now()`. The ranked list is labeled with this
  snapshot date in the UI so it's never an implicit assumption. Follow-ups,
  `contact_log`, and `reactions` use real wall-clock time instead
  (`mcp_server._real_today`), since they track what a banker actually did
  -- backdating them to match the pin would silently defeat the 7-day
  contact-suppression guardrail once real time passed it.
- **Drafting is fully on-demand, not a fixed top-N** (replaced an earlier
  `TOP_N_TO_DRAFT = 15` design): eagerly drafting a fixed top-15 meant
  lower-ranked rows could never satisfy "click a row to expand the drafted
  outreach." Deferring to the moment a banker actually expands a row
  satisfies that for every row while spending fewer LLM calls on average.
- Contact suppression window defaults to 7 days (`memory.recently_contacted`)
  -- a reasonable default for weekly outreach cadence, exposed as a
  parameter rather than a hardcoded assumption.
- Premium personalization threshold (`PREMIUM_AUS_THRESHOLD = $25M`,
  PLATINUM/GOLD tier) in `openai_client.py` -- roughly the 75th percentile
  of AUS in the provided dataset, not tuned against any held-out signal.
- `mcp==1.9.4` is pinned deliberately: 2.0.0 restructured
  `mcp.server.fastmcp`/`mcp.types` with breaking changes; 1.9.4 is the last
  version confirmed compatible with the FastMCP API this project uses.
- **FastMCP silently pre-parses JSON-looking string arguments**: its stdio
  transport runs every string argument through `json.loads` and swaps in
  the parsed value if it succeeds, regardless of declared type.
  `draft_outreach`'s `trigger_context` is therefore typed `dict`, not
  `str` -- worth remembering before adding any new tool parameter that's a
  string containing structured data.
- **Engagement-preference channel mapping only covers 2 of ~6 free-text
  note patterns**, by design: `get_engagement_channel_hint` only acts on
  notes that map onto a channel this system can actually produce (routing/
  tone guidance and timing preferences stay as prose context only; "wants
  concise Teams messages" is deliberately left unmapped since this
  system's `teams` channel is an internal note, not client-facing copy).
- **The "already contacted" guardrail isn't atomic with the follow-up
  upsert**: `upsert_sent_todo`'s transaction closes the duplicate-follow-up
  race, but the `recently_contacted` check that gates `log_contact` is a
  separate, non-atomic read before that transaction starts. Not hardened
  further -- the guardrail's real purpose (catching a banker acting on a
  list that went stale minutes or hours earlier) doesn't need
  microsecond-level atomicity.
- The opportunity score ranks 4 of 5 intents on a shared urgency/value
  composite; `response_conversion` uses the same composite, with its
  engagement component driven by the reaction (live if recorded, else the
  static note) rather than a separate always-priority queue.
- **Ranking unit is `(client, intent, trigger)`, not client**: a client
  with two live opportunities appears as two separate rows, since each may
  warrant a different draft, channel, and urgency. Client-level grouping
  was considered and rejected -- it would require an arbitrary rule for
  aggregating multiple opportunity scores into one ranking number.
- **Single-service architecture, not microservices**: one Streamlit
  process and one MCP server process, communicating over stdio. The MCP
  server is already a real service boundary (the harness only ever talks
  to it through the MCP protocol, never an in-process import), so
  decomposing it into a network service later is a config change (swap
  stdio for HTTP/SSE), not a rewrite. Full microservice decomposition would
  be premature for a 100-client mock dataset and a single banker-facing UI.
- **A2A not used**: A2A exists for multiple *coordinating* agents. This
  system has one harness orchestrating one skill registry -- there's no
  second agent to coordinate with, so introducing A2A here would be
  protocol for its own sake.
