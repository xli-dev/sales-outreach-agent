"""
intent_router.py

Maps the free-text chat ask to one or more skill intents plus any params
(timeframe override, named client, named banker/team). This is the ONE
place an LLM is used for "understanding" rather than "drafting" -- kept
small and cheap (gpt-5.4-mini) since it's a classification task, not a
generation task. Upgraded from gpt-5-mini to handle ambiguous/overlapping
intent cases (e.g. a client both maturing and liquidity-flagged) more
reliably.

The router returns structured JSON; the harness (not the model) decides
what to do with it. If the model returns an intent that isn't in the
registry, or malformed JSON, the harness treats it as a routing failure
and re-prompts once before falling back to "ask the user to clarify" --
this is the harness's re-planning behavior for a bad/empty routing result,
mirrored from how it re-plans on empty data results (see harness.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from skills import all_intents

ROUTER_SYSTEM_PROMPT = f"""You route a banker's free-text request to one or more
outreach intents. Valid intents: {all_intents()}.

Respond ONLY with JSON, no prose, no markdown fences:
{{
  "intents": ["<intent>", ...],
  "client_name": "<string or null, if a specific client was named>",
  "banker_or_team": "<string or null, if a specific banker/team was named>",
  "timeframe_days": <int or null, if the user specified a window>
}}

Guidance:
- "who should I reach out to" / "this week" with no other detail -> return
  ALL intents so the harness ranks across the full funnel.
- Mentions of "maturing", "rolling off" -> maturity
- Mentions of "new account", "just opened" -> new_account
- Mentions of "wire", "IPO", "sold", "liquidity" -> liquidity_event
- Mentions of "exception", "expiring rate", "reverts" -> exception_expiry
- Mentions of "replied", "responded", "got back to us", "follow up on reply" -> response_conversion
"""


@dataclass
class RoutedIntent:
    intents: list[str]
    client_name: str | None
    banker_or_team: str | None
    timeframe_days: int | None


def route(user_ask: str, llm_call) -> RoutedIntent:
    """llm_call: callable(system_prompt, user_prompt) -> str, injected so this
    module has no hard dependency on the OpenAI client (testable, swappable)."""
    raw = llm_call(ROUTER_SYSTEM_PROMPT, user_ask)
    try:
        parsed = json.loads(raw)
        intents = [i for i in parsed.get("intents", []) if i in all_intents()]
        if not intents:
            raise ValueError("no valid intents returned")
        return RoutedIntent(
            intents=intents,
            client_name=parsed.get("client_name"),
            banker_or_team=parsed.get("banker_or_team"),
            timeframe_days=parsed.get("timeframe_days"),
        )
    except (json.JSONDecodeError, ValueError):
        # Re-planning: retry once with a stricter nudge before giving up
        retry_raw = llm_call(
            ROUTER_SYSTEM_PROMPT + "\nSTRICT: return valid JSON matching the schema exactly.",
            user_ask,
        )
        parsed = json.loads(retry_raw)  # let this raise -> harness surfaces a clarify-with-user response
        intents = [i for i in parsed.get("intents", []) if i in all_intents()]
        if not intents:
            intents = all_intents()  # fall back to full-funnel rather than fail closed
        return RoutedIntent(
            intents=intents,
            client_name=parsed.get("client_name"),
            banker_or_team=parsed.get("banker_or_team"),
            timeframe_days=parsed.get("timeframe_days"),
        )
