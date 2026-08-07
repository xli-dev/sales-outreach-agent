"""
data_access.py

Single, boring source of truth for reading the provided mock dataset.
Nothing in here talks to an LLM. Nothing in here invents a value that
isn't in the CSVs. This module is what the MCP tools call into, and
it's kept separate so it can be unit-tested without any agent machinery.
"""
from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TODAY = dt.date(2026, 8, 3)  # matches decision_makers.json "as_of_date" -- pinned for reproducible demo runs


def _read_csv(name: str) -> list[dict]:
    with open(DATA_DIR / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_date(s: str) -> Optional[dt.date]:
    if not s:
        return None
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def _parse_float(s: str) -> Optional[float]:
    if s in (None, ""):
        return None
    return float(s)


def _parse_bool(s: str) -> bool:
    return s.strip().lower() == "true"


@dataclass
class Account:
    account_id: str
    client_eci: str
    product_type: str
    account_class: str
    platform: str
    balance_usd: float
    currency: str
    rate: Optional[float]
    libor_rate: Optional[float]
    is_exception_priced: bool
    exception_rate: Optional[float]
    start_date: Optional[dt.date]
    end_date: Optional[dt.date]
    last_exception_request_id: Optional[str]
    proposed_rate: Optional[float]
    request_reason_type: Optional[str]
    competitor_rate: Optional[float]
    competitor_product: Optional[str]
    justification_comment: Optional[str]
    decision_date: Optional[dt.date]

    @property
    def days_to_exception_expiry(self) -> Optional[int]:
        if not (self.is_exception_priced and self.end_date):
            return None
        return (self.end_date - TODAY).days


@dataclass
class Transaction:
    txn_id: str
    client_eci: str
    account_id: str
    date: dt.date
    direction: str
    amount: float
    type: str
    counterparty: str
    liquidity_event_flag: bool
    liquidity_event_type: Optional[str]

    @property
    def days_ago(self) -> int:
        return (TODAY - self.date).days


@dataclass
class BankerNote:
    note_id: str
    client_eci: str
    date: dt.date
    author_banker_id: str
    channel: str
    text: str
    client_reaction: Optional[str]
    topic: Optional[str]
    linked_request_id: Optional[str]

    @property
    def days_ago(self) -> int:
        return (TODAY - self.date).days


@dataclass
class DecisionMaker:
    client_eci: str
    name: str
    segment: str
    tier: str
    total_aus_usd: float
    rate_sensitivity: str
    banker_id: str
    banker_name: str
    team: str
    region: str
    accounts: list[Account] = field(default_factory=list)
    transactions: list[Transaction] = field(default_factory=list)
    notes: list[BankerNote] = field(default_factory=list)

    @property
    def total_deposit_balance(self) -> float:
        return sum(a.balance_usd for a in self.accounts)


class DataStore:
    """Loads all CSVs once and joins them in memory. Reproducible, no
    external services -- appropriate for a static mock dataset."""

    def __init__(self):
        self._dms: dict[str, DecisionMaker] = {}
        self._load()

    def _load(self):
        for row in _read_csv("decision_makers.csv"):
            eci = row["CLIENT_ECI"]
            self._dms[eci] = DecisionMaker(
                client_eci=eci,
                name=row["name"],
                segment=row["segment"],
                tier=row["tier"],
                total_aus_usd=_parse_float(row["total_aus_usd"]) or 0.0,
                rate_sensitivity=row["rate_sensitivity"],
                banker_id=row["banker_id"],
                banker_name=row["banker_name"],
                team=row["team"],
                region=row["region"],
            )

        for row in _read_csv("accounts.csv"):
            eci = row["CLIENT_ECI"]
            if eci not in self._dms:
                continue
            acct = Account(
                account_id=row["ACCOUNT_ID"],
                client_eci=eci,
                product_type=row["product_type"],
                account_class=row["account_class"],
                platform=row["platform"],
                balance_usd=_parse_float(row["balance_usd"]) or 0.0,
                currency=row["currency"],
                rate=_parse_float(row["RATE"]),
                libor_rate=_parse_float(row["LIBOR_RATE"]),
                is_exception_priced=_parse_bool(row["is_exception_priced"]),
                exception_rate=_parse_float(row["EXCEPTION_RATE"]),
                start_date=_parse_date(row["START_DATE"]),
                end_date=_parse_date(row["END_DATE"]),
                last_exception_request_id=row["last_exception_REQUEST_ID"] or None,
                proposed_rate=_parse_float(row["PROPOSED_RATE"]),
                request_reason_type=row["REQUEST_REASON_TYPE"] or None,
                competitor_rate=_parse_float(row["COMPETITOR_RATE"]),
                competitor_product=row["COMPETITOR_PRODUCT"] or None,
                justification_comment=row["BUSINESS_JUSTIFICATION_COMMENT"] or None,
                decision_date=_parse_date(row["decision_date"]),
            )
            self._dms[eci].accounts.append(acct)

        for row in _read_csv("transactions.csv"):
            eci = row["CLIENT_ECI"]
            if eci not in self._dms:
                continue
            txn = Transaction(
                txn_id=row["txn_id"],
                client_eci=eci,
                account_id=row["ACCOUNT_ID"],
                date=_parse_date(row["date"]),
                direction=row["direction"],
                amount=_parse_float(row["AMOUNT"]) or 0.0,
                type=row["type"],
                counterparty=row["counterparty"],
                liquidity_event_flag=_parse_bool(row["liquidity_event_flag"]),
                liquidity_event_type=row["liquidity_event_type"] or None,
            )
            self._dms[eci].transactions.append(txn)

        for row in _read_csv("banker_notes.csv"):
            eci = row["CLIENT_ECI"]
            if eci not in self._dms:
                continue
            note = BankerNote(
                note_id=row["note_id"],
                client_eci=eci,
                date=_parse_date(row["date"]),
                author_banker_id=row["author_banker_id"],
                channel=row["channel"],
                text=row["text"],
                client_reaction=row["client_reaction"] or None,
                topic=row["topic"] or None,
                linked_request_id=row["linked_REQUEST_ID"] or None,
            )
            self._dms[eci].notes.append(note)

        # keep notes/txns newest-first per client for easy "most recent" access
        for dm in self._dms.values():
            dm.notes.sort(key=lambda n: n.date, reverse=True)
            dm.transactions.sort(key=lambda t: t.date, reverse=True)

    # ---- query methods used by MCP tools ---------------------------------

    def get_client(self, client_eci: str) -> Optional[DecisionMaker]:
        return self._dms.get(client_eci)

    def all_clients(self) -> list[DecisionMaker]:
        return list(self._dms.values())

    def clients_with_maturity_in_window(self, days: int = 30) -> list[tuple[DecisionMaker, Account]]:
        """Accounts (term deposits / CDs) whose END_DATE falls within the window."""
        out = []
        for dm in self._dms.values():
            for a in dm.accounts:
                if a.product_type in ("TD", "CD") and a.end_date:
                    delta = (a.end_date - TODAY).days
                    if delta <= days:
                        out.append((dm, a))
        return out

    def clients_with_exception_expiring(self, days: int = 45) -> list[tuple[DecisionMaker, Account]]:
        out = []
        for dm in self._dms.values():
            for a in dm.accounts:
                if a.is_exception_priced and a.end_date:
                    delta = (a.end_date - TODAY).days
                    if delta <= days:
                        out.append((dm, a))
        return out

    def clients_with_liquidity_event(self, days: int = 21) -> list[tuple[DecisionMaker, Transaction]]:
        out = []
        for dm in self._dms.values():
            for t in dm.transactions:
                if t.liquidity_event_flag and t.days_ago <= days:
                    out.append((dm, t))
        return out

    def clients_with_new_account(self, days: int = 30) -> list[tuple[DecisionMaker, Account]]:
        out = []
        for dm in self._dms.values():
            for a in dm.accounts:
                if a.start_date and (TODAY - a.start_date).days <= days:
                    out.append((dm, a))
        return out

    def clients_with_open_response(self, days: int = 60) -> list[tuple[DecisionMaker, BankerNote]]:
        """Notes with a reaction other than NO_RESPONSE, recent, not yet converted."""
        out = []
        for dm in self._dms.values():
            for n in dm.notes:
                if n.client_reaction in ("POSITIVE", "NEUTRAL") and n.days_ago <= days:
                    out.append((dm, n))
        return out


# Free-text "Engagement preference" RELATIONSHIP notes carry real signal
# about how a client wants to be reached, but only these two patterns map
# cleanly onto this system's channel set. `email` is the only client-facing
# channel here; `teams`/`call_brief`/`meeting_notes` are all internal
# banker-prep documents (see channel_guidance in mcp_server.py). A note
# saying a client wants Teams messages can't be honored by our `teams`
# channel -- that produces an internal nudge to the banker, not
# client-facing copy -- so it's deliberately left unmapped rather than
# silently mislabeled as something this system doesn't actually do.
_ENGAGEMENT_CHANNEL_PATTERNS = (
    ("prefers a short call over email", "call_brief"),
    ("likes an in-person quarterly review", "meeting_notes"),
)


def engagement_channel_hint(notes: list[BankerNote]) -> tuple[Optional[str], Optional[str]]:
    """Scan a client's RELATIONSHIP notes for an engagement-preference
    signal that maps onto a channel this system can actually produce.
    Returns (channel, raw_note_text), or (None, None) if no note matches --
    deterministic keyword match, not an LLM call, same grounding principle
    as everything else in this module."""
    for n in notes:
        if n.topic != "RELATIONSHIP" or "Engagement preference" not in n.text:
            continue
        lowered = n.text.lower()
        for phrase, channel in _ENGAGEMENT_CHANNEL_PATTERNS:
            if phrase in lowered:
                return channel, n.text
    return None, None


_store: Optional[DataStore] = None


def get_store() -> DataStore:
    global _store
    if _store is None:
        _store = DataStore()
    return _store
