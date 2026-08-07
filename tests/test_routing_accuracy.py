"""
Routing-ACCURACY eval, not a unit test: intent_router.route() is an LLM
call, so its correctness can't be asserted the way the rest of this suite
does (exact input -> exact output). This scores it against
routing_eval_cases.CASES -- a small hand-labeled set of realistic asks,
shared with the Streamlit sidebar's "Run routing eval" diagnostic so the
two never drift into scoring against different cases.

Deliberately separate from the rest of the suite: it calls the real
OpenAI API (costs real tokens, ~1-2 calls per case, and the router's
exact output isn't perfectly deterministic run to run) -- skipped
automatically when no OPENAI_API_KEY is available, so it never blocks or
slows down the free/deterministic CI run the other test files are part
of. Run explicitly and locally: `pytest tests/test_routing_accuracy.py -v -s`
(the -s shows the printed score; pytest hides it on a pass otherwise).
"""
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
    reason="routing-accuracy eval calls the real OpenAI API; needs OPENAI_API_KEY",
)

import intent_router  # noqa: E402
from openai_client import call_router_llm  # noqa: E402
from routing_eval_cases import CASES, score_case  # noqa: E402


def test_routing_accuracy():
    correct = 0
    failures = []
    for ask, expected in CASES:
        result = intent_router.route(ask, call_router_llm)
        actual = set(result.intents)
        if score_case(expected, actual):
            correct += 1
        else:
            failures.append((ask, sorted(expected), sorted(actual)))
    accuracy = correct / len(CASES)
    # Printed unconditionally (run with -s to see it) -- pytest otherwise
    # only surfaces this via the assertion message, i.e. only on failure,
    # which hides the actual score on every run that happens to pass.
    print(f"\nrouting accuracy: {accuracy:.0%} ({correct}/{len(CASES)})")
    assert accuracy >= 0.8, (
        f"routing accuracy {accuracy:.0%} ({correct}/{len(CASES)}) below 80% threshold.\n"
        f"Failures (ask, expected, actual): {failures}"
    )
