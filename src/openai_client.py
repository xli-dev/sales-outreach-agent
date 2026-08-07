"""
openai_client.py

Single place the OpenAI key/model are configured. Never hardcode the key --
read from environment (Streamlit secrets inject as env vars, GitHub Actions
as repo/workflow secrets). Three models used deliberately for cost/latency:
  - ROUTER_MODEL:   cheap/fast, used for intent classification (small task)
  - DRAFT_MODEL:    stronger, used for outreach drafting (quality matters)
  - PREMIUM_DRAFT_MODEL: reserved for PLATINUM/GOLD tier or >=$25M AUS
    clients -- higher-balance, higher-potential clients warrant richer
    personalization; we tier the MODEL to match, not just the prompt,
    since draft quality is part of what "richer" means for a $250M
    relationship vs a $500K one.
"""
from __future__ import annotations

import os

from openai import OpenAI

ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "gpt-5.4-mini")
DRAFT_MODEL = os.environ.get("DRAFT_MODEL", "gpt-5.4")
PREMIUM_DRAFT_MODEL = os.environ.get("PREMIUM_DRAFT_MODEL", "gpt-5.5")
PREMIUM_TIERS = {"PLATINUM", "GOLD"}
PREMIUM_AUS_THRESHOLD = 25_000_000

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Add it to Streamlit secrets / GitHub Actions "
                "secrets -- never commit it to the repo."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    tier: str | None = None,
    total_aus_usd: float = 0,
) -> tuple[str, str]:
    """Returns (text, model_used) -- the model used is returned (not just
    logged) so callers can surface it in the harness trace as a permanent,
    auditable record of which model tier was applied and why, rather than
    a throwaway debug print."""
    if model is None:
        is_premium = tier in PREMIUM_TIERS or total_aus_usd >= PREMIUM_AUS_THRESHOLD
        model = PREMIUM_DRAFT_MODEL if is_premium else DRAFT_MODEL

    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content, model


def call_router_llm(system_prompt: str, user_prompt: str) -> str:
    text, _ = call_llm(system_prompt, user_prompt, model=ROUTER_MODEL)
    return text
