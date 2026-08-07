"""
hallucination_eval_cases.py

Builds the sample used to measure how often draft_outreach actually
produces an ungrounded figure in practice -- pulled live from the real
mock dataset via data_access + skills, not a hand-picked or hand-labeled
list. Unlike routing accuracy (which needs a human judgment call about
the "right" intent for an ask), whether a draft hallucinates is objective
-- _grounding_check either flags it or it doesn't -- so there's nothing
to label here, only real candidates to sample from.

Imported by BOTH tests/test_draft_hallucination_eval.py (the pytest eval)
and streamlit_app.py (the sidebar "Run hallucination eval" diagnostic),
so the two always score against the same live sample logic.
"""
from skills import SKILL_REGISTRY


def sample_cases(n_per_intent: int = 3) -> list[tuple[str, str, str]]:
    """Top N real (by score) candidates for each intent that currently has
    any, paired with that intent's own default channel -- e.g.
    (client_eci, "exception_expiry", "teams"). new_account is skipped
    automatically if it has zero candidates in the current dataset (see
    README's Known Scoping Calls -- true today, not assumed)."""
    import data_access
    import scoring

    store = data_access.get_store()
    cases = []
    for intent, skill in SKILL_REGISTRY.items():
        scored = []
        for dm, trigger in skill.query_fn(store, skill.horizon_days):
            scored.append(skill.score_fn(dm, trigger, skill.horizon_days))
        scored.sort(key=lambda s: s.score, reverse=True)
        for s in scored[:n_per_intent]:
            cases.append((s.client.client_eci, intent, skill.default_channel))
    return cases
