# SPDX-License-Identifier: AGPL-3.0-or-later
"""_generate()'s stderr-context selection (#952/#963 follow-up)."""

import contextlib

from localm.debuglog import dedup_native_stderr
from localm.inference.backends.llamacpp.llama import _stderr_ctx_for_generate


def test_verbose_uses_nullcontext():
    assert _stderr_ctx_for_generate(True) is contextlib.nullcontext


def test_non_verbose_uses_dedup_native_stderr():
    """Grammar/grammar_lazy are no longer inputs to this decision at all - the whole point of the fix is that both branches now agree."""
    assert _stderr_ctx_for_generate(False) is dedup_native_stderr
