# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for _stderr_ctx_for_generate selection."""

import contextlib

from localm.debuglog import dedup_native_stderr
from localm.inference.backends.llamacpp.llama import (
    _quiet_stderr,
    _stderr_ctx_for_generate,
)


def test_verbose_uses_nullcontext():
    assert _stderr_ctx_for_generate(True) is contextlib.nullcontext
    assert _stderr_ctx_for_generate(True, grammar_active=True) is contextlib.nullcontext
    assert _stderr_ctx_for_generate(True, grammar_active=False) is contextlib.nullcontext


def test_non_verbose_plain_uses_dedup_native_stderr():
    assert _stderr_ctx_for_generate(False) is dedup_native_stderr
    assert _stderr_ctx_for_generate(False, grammar_active=False) is dedup_native_stderr


def test_non_verbose_grammar_uses_quiet_stderr():
    assert _stderr_ctx_for_generate(False, grammar_active=True) is _quiet_stderr
