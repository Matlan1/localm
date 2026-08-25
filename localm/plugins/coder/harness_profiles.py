# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-model harness profiles for the coder agent."""

from __future__ import annotations

# Conservative seed. Keyed by prompts.detect_model_family() output.
# gemma / default deliberately have no overrides (baseline behavior).
_PROFILES: dict = {
    "small":    {"temperature": 0.3},
    "thinking": {"max_tokens": 4096},
}

# Generation kwargs safe to FILL in any path (additive, no truncation risk).
# max_tokens is intentionally NOT here - see the module docstring.
_AGENT_SAFE_GEN_KEYS = ("temperature",)

# The CLI's prior hardcoded default, kept here so the baseline lives in one place.
CLI_DEFAULT_MAX_TOKENS = 2048


def _family(model_name: str) -> str:
    """The model family for *model_name* ('default' when unknown/empty)."""
    if not model_name:
        return "default"
    from .prompts import detect_model_family
    return detect_model_family(model_name)


def family_profile(model_name: str) -> dict:
    """The full profile override dict for *model_name*'s family ({} if none)."""
    return dict(_PROFILES.get(_family(model_name), {}))


def agent_gen_overrides(model_name: str) -> dict:
    """Gen-kwarg defaults safe to fill in the Agent for any path (explicit wins)."""
    prof = family_profile(model_name)
    return {k: v for k, v in prof.items() if k in _AGENT_SAFE_GEN_KEYS}


def cli_max_tokens(model_name: str, baseline: int = CLI_DEFAULT_MAX_TOKENS) -> int:
    """The CLI's default max_tokens for *model_name* - the family override, or the baseline cap."""
    val = family_profile(model_name).get("max_tokens")
    return int(val) if val is not None else int(baseline)
