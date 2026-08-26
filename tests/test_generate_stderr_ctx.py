# SPDX-License-Identifier: AGPL-3.0-or-later
"""_generate()'s stderr-context selection.

Grammar and plain generation share one decision (_stderr_ctx_for_generate):
both use dedup_native_stderr (grouped, still visible) rather than _quiet_stderr
(full suppression), now that _LineGrouper collapses the lazy grammar sampler's
per-token "still awaiting trigger" spam. Tested here directly as a pure
function, so it needs no real native model."""

import contextlib

from localm.debuglog import dedup_native_stderr
from localm.inference.backends.llamacpp.llama import _stderr_ctx_for_generate


def test_verbose_uses_nullcontext():
    assert _stderr_ctx_for_generate(True) is contextlib.nullcontext


def test_non_verbose_uses_dedup_native_stderr():
    """Grammar/grammar_lazy are not inputs to this decision at all - both
    branches agree."""
    assert _stderr_ctx_for_generate(False) is dedup_native_stderr
