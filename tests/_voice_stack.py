# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gating for tests that need the optional voice stack (PyAV, faster-whisper).

A test that needs the voice stack must not pass by being skipped where the stack
is missing: that is how a broken decoder went unseen. With
``LOCALM_REQUIRE_VOICE_STACK=1`` (set by the CI job that installs the voice
extra) a missing stack is a FAILURE; without it, the test is skipped so a
contributor without the extra can still run the rest of the suite.
"""
from __future__ import annotations

import os
from typing import Optional

import pytest

REQUIRE_ENV = "LOCALM_REQUIRE_VOICE_STACK"


def required() -> bool:
    return os.environ.get(REQUIRE_ENV) == "1"


def missing_reason() -> Optional[str]:
    """Why the voice stack cannot be used here, or None when it can."""
    for module, label in (("av", "PyAV"), ("faster_whisper.audio", "faster-whisper")):
        try:
            __import__(module)
        except (ImportError, OSError) as e:
            return f"{label} cannot be imported ({type(e).__name__}: {e})"
    return None


def unavailable(reason: str) -> None:
    """Fail when the voice stack is required, otherwise skip."""
    if required():
        pytest.fail(f"{REQUIRE_ENV}=1 but {reason}", pytrace=False)
    pytest.skip(reason)


@pytest.fixture
def voice_stack():
    reason = missing_reason()
    if reason is not None:
        unavailable(reason)
