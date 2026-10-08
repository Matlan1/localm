# SPDX-License-Identifier: AGPL-3.0-or-later
"""A voice test must fail, not skip, where the environment says the voice stack
is required. These tests need no voice stack themselves."""
import pytest

from tests import _voice_stack


def test_a_missing_stack_skips_by_default(monkeypatch):
    monkeypatch.delenv(_voice_stack.REQUIRE_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception):
        _voice_stack.unavailable("PyAV is not installed")


def test_a_missing_stack_fails_when_the_environment_requires_it(monkeypatch):
    monkeypatch.setenv(_voice_stack.REQUIRE_ENV, "1")
    with pytest.raises(pytest.fail.Exception, match="LOCALM_REQUIRE_VOICE_STACK=1"):
        _voice_stack.unavailable("PyAV is not installed")


def test_the_fixture_fails_a_test_when_required_and_the_stack_is_missing(monkeypatch):
    monkeypatch.setenv(_voice_stack.REQUIRE_ENV, "1")
    monkeypatch.setattr(_voice_stack, "missing_reason", lambda: "PyAV cannot be imported")
    with pytest.raises(pytest.fail.Exception, match="PyAV cannot be imported"):
        _voice_stack.voice_stack.__wrapped__()


def test_the_fixture_passes_when_the_stack_is_present(monkeypatch):
    monkeypatch.setenv(_voice_stack.REQUIRE_ENV, "1")
    monkeypatch.setattr(_voice_stack, "missing_reason", lambda: None)
    assert _voice_stack.voice_stack.__wrapped__() is None


def test_an_unimportable_module_is_reported_with_its_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def refuse_av(name, *a, **k):
        if name == "av":
            raise ImportError("no PyAV here")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", refuse_av)
    reason = _voice_stack.missing_reason()
    assert reason is not None
    assert "PyAV" in reason and "no PyAV here" in reason
