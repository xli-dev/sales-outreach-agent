"""
memory.py

Persistent (cross-session) memory, backed by SQLite for reproducibility --
no external service needed to run the demo from a fresh clone.

Two things live here that the harness reads on every request:
  1. contact_log   -- suppress clients touched too recently
  2. todos         -- open follow-ups carried into the next planning pass

reactions and draft_history exist to support response-conversion (a reaction
recorded on a prior outreach is itself an input signal next time) and to give
the UI something to show under "history" for a client.

Design note: this is intentionally a relational store, not a vector store.
Given the dataset (100 clients, few hundred notes), exact-match/filter
retrieval over structured fields covers everything required. A vector index
over banker_notes.text would help if note volume grew large enough that
semantic search materially beat "give me this client's last N notes" --
noted as a follow-on, not built here, to avoid unnecessary complexity for
~366 rows of text.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "memory.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS contact_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_eci TEXT NOT NULL,
    intent TEXT NOT NULL,
    channel TEXT NOT NULL,
    contacted_at TEXT NOT NULL,
    draft_id INTEGER
);

CREATE TABLE IF NOT EXISTS draft_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_eci TEXT NOT NULL,
    intent TEXT NOT NULL,
    channel TEXT NOT NULL,
    content TEXT NOT NULL,
    grounding_flag TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS todos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_eci TEXT NOT NULL,
    client_name TEXT,
    intent TEXT,
    channel TEXT,
    kind TEXT NOT NULL DEFAULT 'outreach',
    action TEXT NOT NULL,
    due_date TEXT,
    status TEXT NOT NULL DEFAULT 'OPEN',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_eci TEXT NOT NULL,
    channel TEXT NOT NULL,
    intent TEXT,
    reaction TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    note TEXT
);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as c:
        c.executescript(SCHEMA)
        # migrations: todos predates client_name, then intent/channel -- add
        # whichever are missing for DBs created before those columns
        # existed, so old rows still work (NULL values; find_open_todo just
        # won't match old rows against a specific intent/channel).
        existing_cols = {row["name"] for row in c.execute("PRAGMA table_info(todos)")}
        for col in ("client_name", "intent", "channel"):
            if col not in existing_cols:
                c.execute(f"ALTER TABLE todos ADD COLUMN {col} TEXT")
        if "kind" not in existing_cols:
            c.execute("ALTER TABLE todos ADD COLUMN kind TEXT NOT NULL DEFAULT 'outreach'")
            # One-time best-effort backfill: rows created by an earlier
            # version, before "kind" existed, that are actually conversion
            # follow-ups (identifiable only by their action text, since
            # there was nowhere else to record it at the time) would
            # otherwise be misclassified as "outreach" -- which is exactly
            # the collision this column exists to prevent. This is a
            # one-time migration cleanup on existing rows, not runtime
            # logic (record_reaction never matches on action text).
            c.execute(
                "UPDATE todos SET kind='convert' WHERE action LIKE 'Positive response on%'"
            )
        reaction_cols = {row["name"] for row in c.execute("PRAGMA table_info(reactions)")}
        if "intent" not in reaction_cols:
            c.execute("ALTER TABLE reactions ADD COLUMN intent TEXT")
            # Old rows have no way to recover which intent they were about
            # (never recorded) -- left NULL rather than guessed. This only
            # affects the History display's precision for reactions logged
            # before this column existed, not any matching/scoring logic
            # (record_reaction's todo-closing behavior keys off client_eci
            # + channel + the intent passed in at call time, not this
            # stored value).


def reset_all():
    """Wipe every persistent table -- for demo purposes (e.g. a public
    Streamlit Community Cloud deployment, where a banker giving a live demo
    has no terminal access to just delete memory.sqlite3 between runs, and
    the single shared file otherwise leaks one visitor's state into the
    next visitor's view). Never called automatically; only from an explicit
    UI action."""
    with _conn() as c:
        has_sequence_table = c.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'"
        ).fetchone()
        for table in ("contact_log", "draft_history", "todos", "reactions"):
            c.execute(f"DELETE FROM {table}")
            if has_sequence_table:
                c.execute("DELETE FROM sqlite_sequence WHERE name=?", (table,))


def recently_contacted(client_eci: str, within_days: int = 7) -> bool:
    cutoff = (dt.datetime.now() - dt.timedelta(days=within_days)).isoformat()
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM contact_log WHERE client_eci=? AND contacted_at >= ? LIMIT 1",
            (client_eci, cutoff),
        ).fetchone()
        return row is not None


def log_contact(client_eci: str, intent: str, channel: str, draft_id: int | None = None):
    with _conn() as c:
        c.execute(
            "INSERT INTO contact_log (client_eci, intent, channel, contacted_at, draft_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (client_eci, intent, channel, dt.datetime.now().isoformat(), draft_id),
        )


def save_draft(client_eci: str, intent: str, channel: str, content: str, grounding_flag: str | None = None) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO draft_history (client_eci, intent, channel, content, grounding_flag, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (client_eci, intent, channel, content, grounding_flag, dt.datetime.now().isoformat()),
        )
        return cur.lastrowid


def get_draft(draft_id: int) -> sqlite3.Row | None:
    with _conn() as c:
        return c.execute("SELECT * FROM draft_history WHERE id=?", (draft_id,)).fetchone()


def hallucination_stats() -> tuple[int, int]:
    """(total drafts, flagged drafts) from every real draft this
    deployment has ever generated -- combines BOTH guardrails now folded
    into grounding_flag by mcp_server.draft_outreach: _grounding_check
    ($/%/date, regex) and _note_faithfulness_check (relationship/
    preference/history claims, LLM-as-judge). Both already run on every
    draft_outreach call and the result is already persisted here, so this
    is an instant read of existing data, not a new eval that has to
    generate fresh drafts (and their real API calls/wait time) to answer
    "how often does this actually happen." A separate on-demand eval
    (tests/test_draft_hallucination_eval.py) still exists for proactively
    sampling candidates nobody has drafted yet -- this is for "what's
    actually been happening," not "let's go check right now." See
    note_faithfulness_stats() below to isolate just the note-faithfulness
    portion of this combined number."""
    with _conn() as c:
        row = c.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN grounding_flag IS NOT NULL THEN 1 ELSE 0 END) AS flagged "
            "FROM draft_history"
        ).fetchone()
        return row["total"], row["flagged"] or 0


def note_faithfulness_stats() -> tuple[int, int]:
    """(total drafts, drafts with a note-faithfulness issue) -- isolates
    the _note_faithfulness_check portion of the combined grounding_flag
    (see hallucination_stats() above for the combined number). The marker
    string must match _note_faithfulness_check's own flag prefix exactly
    (mcp_server.py) -- duplicated here rather than imported, since
    mcp_server.py already imports this module and importing back would
    be circular."""
    with _conn() as c:
        row = c.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN grounding_flag LIKE '%Possible relationship/preference claim(s)%' "
            "THEN 1 ELSE 0 END) AS flagged "
            "FROM draft_history"
        ).fetchone()
        return row["total"], row["flagged"] or 0


def create_todo(
    client_eci: str, action: str, due_date: str | None = None,
    client_name: str | None = None, intent: str | None = None, channel: str | None = None,
    kind: str = "outreach",
) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO todos (client_eci, client_name, intent, channel, kind, action, due_date, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)",
            (client_eci, client_name, intent, channel, kind, action, due_date, dt.datetime.now().isoformat()),
        )
        return cur.lastrowid


def open_todos(client_eci: str | None = None) -> list[sqlite3.Row]:
    with _conn() as c:
        if client_eci:
            return c.execute(
                "SELECT * FROM todos WHERE client_eci=? AND status='OPEN' ORDER BY due_date",
                (client_eci,),
            ).fetchall()
        return c.execute("SELECT * FROM todos WHERE status='OPEN' ORDER BY due_date").fetchall()


def find_open_todo(
    client_eci: str, intent: str | None = None, channel: str | None = None, kind: str = "outreach",
) -> sqlite3.Row | None:
    """Find the most recent open follow-up for this (client, intent, channel)
    -- used to update or close the SAME todo a draft created, rather than
    guessing by client alone (a client can have multiple live opportunities,
    each with its own follow-up). Defaults to kind="outreach" so this never
    matches a "convert" follow-up (created after a Positive reaction) --
    those share the same (client, intent, channel) key but represent a
    different task; without this, a later send or reaction could silently
    overwrite or duplicate the wrong row (confirmed reproducible before
    this column existed)."""
    query = "SELECT * FROM todos WHERE client_eci=? AND status='OPEN' AND kind=?"
    params: list = [client_eci, kind]
    if intent:
        query += " AND intent=?"
        params.append(intent)
    if channel:
        query += " AND channel=?"
        params.append(channel)
    query += " ORDER BY created_at DESC LIMIT 1"
    with _conn() as c:
        return c.execute(query, params).fetchone()


def retry_todo(todo_id: int, action: str, due_date: str, kind: str | None = None, channel: str | None = None) -> None:
    """Rewrite a follow-up's text and push its due date, without closing
    it -- used when an outcome isn't actually resolved yet (e.g. no
    response by the original due date isn't an answer, just a window that
    passed), so the follow-up keeps nagging with a fresh deadline instead
    of either closing (implying resolution) or sitting there stale.
    Optionally also reclassifies its kind -- e.g. a "convert" follow-up
    that just got a new outreach sent from it becomes a normal "outreach,
    waiting" follow-up going forward, so a later send for the same
    (client, intent, channel) matches it correctly via upsert_sent_todo.
    Also optionally updates channel: a resend from this follow-up on a
    DIFFERENT channel than it was created with (e.g. switching call_brief
    -> meeting_notes) must update this row's own channel column, not just
    its action text -- callers like get_followup_history filter strictly
    by this column, so leaving it stale silently hides the new send from
    History (it's still there in contact_log, just invisible here) and
    left any cached "who is this for" label describing the old channel."""
    sets = ["action=?", "due_date=?"]
    params: list = [action, due_date]
    if kind is not None:
        sets.append("kind=?")
        params.append(kind)
    if channel is not None:
        sets.append("channel=?")
        params.append(channel)
    params.append(todo_id)
    with _conn() as c:
        c.execute(f"UPDATE todos SET {', '.join(sets)} WHERE id=?", params)


def upsert_sent_todo(
    client_eci: str, action: str, due_date: str, client_name: str | None,
    intent: str, channel: str,
) -> None:
    """Atomically find-or-create the follow-up when marking outreach sent --
    wrapped in a single BEGIN IMMEDIATE transaction, which acquires SQLite's
    write lock upfront rather than only at the final INSERT/UPDATE. Without
    this, two near-simultaneous log_contact calls for the same (client,
    intent, channel) can both run find_open_todo before either has written,
    both conclude "no existing todo," and both create one -- confirmed
    reproducible with two concurrent threads before this fix existed.
    A public/shared deployment makes this a real scenario (two bankers'
    stale ranked lists both containing the same client), not just a
    theoretical one. Only ever matches/creates kind="outreach" rows -- a
    "convert" follow-up (from a Positive reaction) shares the same
    (client, intent, channel) key but is a different task; without this
    filter, a later legitimate send would silently overwrite an
    unresolved conversion reminder (confirmed reproducible before this
    filter existed, by simulating the 7-day contact-suppression window
    expiring)."""
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT id FROM todos WHERE client_eci=? AND intent=? AND channel=? AND status='OPEN' AND kind='outreach' "
            "ORDER BY created_at DESC LIMIT 1",
            (client_eci, intent, channel),
        ).fetchone()
        if existing:
            conn.execute("UPDATE todos SET action=? WHERE id=?", (action, existing["id"]))
        else:
            conn.execute(
                "INSERT INTO todos (client_eci, client_name, intent, channel, kind, action, due_date, status, created_at) "
                "VALUES (?, ?, ?, ?, 'outreach', ?, ?, 'OPEN', ?)",
                (client_eci, client_name, intent, channel, action, due_date, dt.datetime.now().isoformat()),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def resolve_positive_reaction(
    client_eci: str, intent: str, channel: str,
    convert_action: str, convert_due_date: str, client_name: str | None,
) -> bool:
    """Atomically close the open outreach follow-up (if any) and create the
    conversion follow-up, in one BEGIN IMMEDIATE transaction. Returns True
    if it actually closed something and created a new row, False if there
    was nothing open to resolve.

    Without this atomicity, two near-simultaneous Positive reactions on
    the same follow-up can both see it as open and both create a
    duplicate conversion follow-up -- confirmed reproducible by forcing
    the timing window open (natural thread scheduling rarely hits it, but
    the code path was genuinely unguarded). The False-return case matters
    for the same reason: a second, racing caller that finds nothing open
    (because the first already closed it) must NOT create its own
    conversion row, or the duplicate happens anyway just one step later."""
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT id FROM todos WHERE client_eci=? AND intent=? AND channel=? AND status='OPEN' AND kind='outreach' "
            "ORDER BY created_at DESC LIMIT 1",
            (client_eci, intent, channel),
        ).fetchone()
        created = False
        if existing:
            conn.execute("UPDATE todos SET status='DONE' WHERE id=?", (existing["id"],))
            conn.execute(
                "INSERT INTO todos (client_eci, client_name, intent, channel, kind, action, due_date, status, created_at) "
                "VALUES (?, ?, ?, ?, 'convert', ?, ?, 'OPEN', ?)",
                (client_eci, client_name, intent, channel, convert_action, convert_due_date, dt.datetime.now().isoformat()),
            )
            created = True
        conn.commit()
        return created
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def complete_todo(todo_id: int):
    with _conn() as c:
        c.execute("UPDATE todos SET status='DONE' WHERE id=?", (todo_id,))


def record_reaction(client_eci: str, channel: str, reaction: str, note: str = "", intent: str | None = None):
    with _conn() as c:
        c.execute(
            "INSERT INTO reactions (client_eci, channel, intent, reaction, recorded_at, note) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (client_eci, channel, intent, reaction, dt.datetime.now().isoformat(), note),
        )


def client_reaction_history(client_eci: str) -> list[sqlite3.Row]:
    with _conn() as c:
        return c.execute(
            "SELECT * FROM reactions WHERE client_eci=? ORDER BY recorded_at DESC",
            (client_eci,),
        ).fetchall()


def get_followup_history(client_eci: str, intent: str) -> list[dict]:
    """Chronological history for one (client, intent) opportunity thread,
    combining contact_log (every send) and reactions (every recorded
    outcome) -- both append-only and already permanently tracked, just
    never surfaced together anywhere. This is what the "History" UI
    reads; nothing here is new data, just a merged, sorted view of two
    tables that already existed. Deliberately NOT filtered by channel: a
    retry from a follow-up can go out on a different channel than the
    original send (that's the whole point of letting a banker redraft for
    a different channel), and the thread is still the same ongoing
    conversation about the same opportunity -- filtering by channel here
    would hide the very history a channel switch is trying to build on top
    of. Each event still carries its own channel for display."""
    with _conn() as c:
        sends = c.execute(
            "SELECT contacted_at AS at, 'sent' AS kind, NULL AS reaction, channel FROM contact_log "
            "WHERE client_eci=? AND intent=?",
            (client_eci, intent),
        ).fetchall()
        reactions = c.execute(
            "SELECT recorded_at AS at, 'reaction' AS kind, reaction, channel FROM reactions "
            "WHERE client_eci=? AND (intent=? OR intent IS NULL)",
            (client_eci, intent),
        ).fetchall()
    events = [dict(r) for r in sends] + [dict(r) for r in reactions]
    events.sort(key=lambda e: e["at"])
    return events


def latest_reaction_value(client_eci: str) -> str | None:
    """Most recent reaction recorded IN-APP (via the UI's reaction buttons),
    as opposed to the static banker_notes.csv reaction history. This is what
    closes the funnel loop: send -> respond -> convert, recording each
    reaction to improve the next round -- scoring.py reads this so a
    reaction logged this session actually changes next round's ranking,
    not just sits in a table nobody reads."""
    with _conn() as c:
        row = c.execute(
            "SELECT reaction FROM reactions WHERE client_eci=? ORDER BY recorded_at DESC LIMIT 1",
            (client_eci,),
        ).fetchone()
        return row["reaction"] if row else None
