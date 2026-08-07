"""
Note-faithfulness eval, not a unit test: broader-sample evidence for
mcp_server._note_faithfulness_check, the permanent guardrail promoted
into draft_outreach after this eval's design was fixed and it reached a
clean result (see README's Known Scoping Calls for the two rounds of
self-inflicted false positives that preceded that). Fast, mocked
unit coverage of the guardrail's own logic lives in
test_note_faithfulness_guard.py; this file is the slower, real-API
complement -- real clients, real drafts, real judge calls -- run
on-demand to gather more evidence than the always-on guardrail alone
provides on any single draft.

draft_outreach now runs _note_faithfulness_check internally on every
draft (it's a permanent guardrail, not opt-in) and folds its verdict into
the combined grounding_flag -- this eval reads that instead of calling
the judge a second time per case, so it costs no more than drafting
already does, and there is exactly one copy of the judge prompt, in
mcp_server.py, so this eval and the production guardrail can never
silently drift apart.

Two tests, measuring the two different ways a binary classifier can be
wrong: test_note_faithfulness measures specificity (does it stay quiet on
drafts that are already faithful -- a false positive here means a banker
gets an unnecessary review prompt); test_note_faithfulness_catches_
injected_hallucinations measures recall (does it actually catch a real
violation -- a false negative here means a hallucination ships to a
client undetected, which is the failure mode that actually matters). A
judge that always says "faithful" would pass the first test perfectly
and fail the second one completely -- neither test alone would reveal
that, which is why both exist.

Deliberately separate from the rest of the suite: it calls the real
OpenAI API (two calls per case: one to draft, one internally to judge) --
skipped automatically when no OPENAI_API_KEY is available, so it never
blocks or slows down the free/deterministic CI run the other test files
are part of. Run explicitly and locally:
`pytest tests/test_note_faithfulness_eval.py -v -s`
(the -s shows the printed rate and any flagged issues; pytest hides
print() output on a pass otherwise).

The recall test found a real gap on its first run (2/3 caught): a
fabricated claim anchored to a REAL negative interaction ("our last call
didn't go well when rates came up") slipped through, because a real
negative-reaction note existed for that client (via Teams, about
retention risk) and the judge treated the matching *valence* as
sufficient, without checking that the *specific circumstance* (channel,
topic) was fabricated. Tightening the judge prompt to require specific-
circumstance matching (not just matching sentiment) fixed this to 3/3 --
but cost precision: test_note_faithfulness went from a clean 12/12 to
10/12. Both new flags were checked concretely, not taken at face value:
one was the judge being pedantic about substantively-true paraphrasing
(it admitted as much in its own reasoning); the other surfaced a genuine
inconsistency in the *mock dataset itself* -- an account's
justification_comment says "Morgan Stanley," but a real call note
independently says "HSBC," for the same rate on the same CD. A draft
citing "HSBC" is actually grounded (in the note), just not in the field
the judge happened to check first. Decision: kept the stricter prompt.
Missing a real hallucination (ships to a client) is a worse failure than
an occasional unnecessary review prompt (costs a banker a few seconds) --
this is a precision/recall tradeoff, not a clean win, and is documented
as a deliberate choice rather than an accident.
"""
import json
import os
import sys
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Mirror streamlit_app.py's secrets->env bridge for a standalone pytest run,
# where nothing has already populated the environment the way the Streamlit
# process does via st.secrets.
if "OPENAI_API_KEY" not in os.environ:
    secrets_path = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
        for k, v in secrets.items():
            if isinstance(v, str):
                os.environ[k] = v

pytestmark = pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="note-faithfulness eval calls the real OpenAI API; needs OPENAI_API_KEY",
)

import mcp_server  # noqa: E402
from hallucination_eval_cases import sample_cases  # noqa: E402


NOTE_ISSUE_MARKER = "Possible relationship/preference claim(s)"


def test_note_faithfulness():
    cases = sample_cases()
    results = []
    for client_eci, intent, channel in cases:
        # draft_outreach already runs _note_faithfulness_check internally
        # (it's a permanent guardrail now, not opt-in) and folds its result
        # into the combined grounding_flag -- read that instead of calling
        # the judge a second time per case, which would double the real API
        # cost of this eval for no benefit. NOTE_ISSUE_MARKER isolates note-
        # specific flags from $/%/date ones also possibly present in the
        # same combined string (see mcp_server.draft_outreach).
        draft_result = json.loads(mcp_server.draft_outreach(
            client_eci, intent, channel, {"reason": "eval", "components": {}},
        ))
        combined_flag = draft_result.get("grounding_flag") or ""
        note_flag = combined_flag if NOTE_ISSUE_MARKER in combined_flag else None
        results.append((client_eci, intent, channel, note_flag))

    flagged = [r for r in results if r[3]]
    print(f"\nnote faithfulness: {len(results) - len(flagged)}/{len(results)} faithful, {len(flagged)} flagged")
    for client_eci, intent, channel, flag in flagged:
        print(f"  {client_eci} {intent} {channel} -- {flag}")

    assert not flagged, f"{len(flagged)}/{len(results)} drafts misrepresented banker notes: {flagged}"


# Deliberately fabricated relationship/preference/history claims, none of
# which appear in this dataset's notes for any client -- injected into
# real drafts to test whether the judge actually catches a violation when
# one exists, not just whether it stays quiet on drafts that are already
# faithful. Varied in kind (a life event, a stated preference, a past
# reaction) so a pass isn't just luck against one phrasing.
INJECTED_HALLUCINATIONS = [
    "I remember you mentioning your daughter's engagement last month -- congratulations again!",
    "As you've told us before, you'd rather receive all future updates by handwritten letter than any digital channel.",
    "I recall our last call didn't go well when rates came up, so I'll keep this brief and avoid the topic.",
]


def test_note_faithfulness_catches_injected_hallucinations():
    """Validates the judge's OTHER failure mode: test_note_faithfulness
    above only proves the judge doesn't false-positive on drafts that are
    already faithful (specificity) -- a judge that always says "faithful"
    regardless of input would score identically on that eval. This proves
    the judge actually has recall: given a real draft with one deliberately
    fabricated relationship/preference/history claim appended, verified to
    not appear anywhere in that client's real profile, does
    _note_faithfulness_check actually flag it? Without this, "100%
    faithful" on the eval above is indistinguishable from a broken judge
    that never flags anything."""
    cases = sample_cases()[:3]
    results = []
    for (client_eci, intent, channel), injected_claim in zip(cases, INJECTED_HALLUCINATIONS):
        profile = json.loads(mcp_server.get_client_profile(client_eci))
        assert injected_claim not in json.dumps(profile), (
            f"sanity check failed: injected claim already appears in {client_eci}'s real profile"
        )
        draft_result = json.loads(mcp_server.draft_outreach(
            client_eci, intent, channel, {"reason": "eval", "components": {}},
        ))
        tampered_draft = draft_result.get("text", "") + "\n\n" + injected_claim
        flag = mcp_server._note_faithfulness_check(tampered_draft, profile)
        results.append((client_eci, injected_claim, flag))

    caught = [r for r in results if r[2]]
    print(f"\ninjected-hallucination recall: {len(caught)}/{len(results)} caught")
    for client_eci, claim, flag in results:
        status = "CAUGHT" if flag else "MISSED"
        print(f"  {status}: {client_eci} -- injected {claim!r}")

    missed = [r for r in results if not r[2]]
    assert not missed, f"{len(missed)}/{len(results)} injected hallucinations went undetected: {missed}"
