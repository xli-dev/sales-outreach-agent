"""
Draft-hallucination eval, not a unit test: draft_outreach makes real LLM
calls, so whether it actually stays grounded in practice can't be
asserted the way test_grounding.py asserts _grounding_check's own regex
logic (hand-crafted draft strings, known profile -- that proves the
CHECKER works, not that real generation rarely trips it). This runs the
real drafting pipeline against hallucination_eval_cases.sample_cases() --
the top-scoring real candidates per intent, pulled live from the actual
mock dataset (no hand-picked or hand-labeled list; unlike routing
accuracy, whether a draft hallucinates is objective, so there's nothing
to label) -- and measures how often the combined grounding_flag actually
fires. That flag is now two guardrails folded into one string:
_grounding_check ($/%/date, regex-based) AND _note_faithfulness_check
(relationship/preference/history claims, LLM-as-judge-based, promoted to
always-on after tests/test_note_faithfulness_eval.py reached a clean
result) -- see that file if you need to isolate which mechanism fired.

Deliberately separate from the rest of the suite: it calls the real
OpenAI API (now two calls per case -- drafting, plus the note-faithfulness
judge that draft_outreach runs internally) and isn't perfectly
deterministic run to run -- skipped automatically when no
OPENAI_API_KEY is available, so it never blocks or slows down the
free/deterministic CI run the other test files are part of. Run
explicitly and locally:
`pytest tests/test_draft_hallucination_eval.py -v -s`
(the -s shows the printed rate; pytest hides it on a pass otherwise).
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
    reason="hallucination eval calls the real OpenAI API; needs OPENAI_API_KEY",
)

import mcp_server  # noqa: E402
from hallucination_eval_cases import sample_cases  # noqa: E402


def test_draft_hallucination_rate():
    cases = sample_cases()
    flagged = []
    for client_eci, intent, channel in cases:
        result = json.loads(mcp_server.draft_outreach(
            client_eci, intent, channel, {"reason": "eval", "components": {}},
        ))
        if result.get("grounding_flag"):
            flagged.append((client_eci, intent, channel, result["grounding_flag"]))
    rate = len(flagged) / len(cases)
    print(f"\nhallucination (grounding-flag) rate: {rate:.0%} ({len(flagged)}/{len(cases)} flagged)")
    for client_eci, intent, channel, flag in flagged:
        print(f"  flagged: {client_eci} {intent} {channel} -- {flag}")
    # Every case is drafted from a real, fully-populated client profile --
    # there's no legitimate missing-data reason for the model to invent a
    # figure here, unlike a hypothetical client with sparse data. A nonzero
    # rate on this sample is a real regression signal (prompt/model
    # change made drafting less grounded), not expected noise to tolerate.
    assert rate == 0, f"{len(flagged)}/{len(cases)} real drafts had unverified figures: {flagged}"
