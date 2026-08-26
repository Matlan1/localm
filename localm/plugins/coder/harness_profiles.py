# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Per-model harness profiles for the coder agent.

The system prompt already adapts to the model family (prompts.detect_model_family
-> gemma / thinking / small / default). This module extends that idea to the
HARNESS knobs - the numeric/behavioral settings around the loop - so a weak small
model and a long-reasoning model can run with settings tuned to them instead of a
single one-size default.

It tunes generation kwargs only:
  - small models: a steadier ``temperature``.
  - thinking models: a larger ``max_tokens`` so the answer is not truncated after
    a long <think> scratchpad.

PRECEDENCE: a profile value is only a DEFAULT - an explicit caller value (a CLI
flag or a ``.localcoder/config.toml`` key) always wins.

Two accessors reflect a real asymmetry between the CLI and the GUI:
  - ``agent_gen_overrides`` returns only keys safe to apply in ANY construction
    path (the Agent, used by CLI + GUI + sub-agents). ``max_tokens`` is excluded
    here because the GUI passes no max_tokens by default (it relies on the
    backend default), so filling one could LOWER/truncate it.
  - ``cli_max_tokens`` is applied in the CLI, where a 2048 baseline cap already
    exists, so raising it for thinking models is strictly safe.
"""

from __future__ import annotations

# Keyed by prompts.detect_model_family() output. gemma and default have no
# overrides, so they keep baseline behavior.
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
    """Gen-kwarg defaults safe to fill in the Agent for any path (explicit wins).

    Excludes ``max_tokens`` (see module docstring) so this never truncates a
    GUI/sub-agent call that relied on the backend's own default.
    """
    prof = family_profile(model_name)
    return {k: v for k, v in prof.items() if k in _AGENT_SAFE_GEN_KEYS}


def cli_max_tokens(model_name: str, baseline: int = CLI_DEFAULT_MAX_TOKENS) -> int:
    """The CLI's default max_tokens for *model_name* - the family override, or the
    baseline cap. Applied only when the user did not set max_tokens explicitly."""
    val = family_profile(model_name).get("max_tokens")
    return int(val) if val is not None else int(baseline)
