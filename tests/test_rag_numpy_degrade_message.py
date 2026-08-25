# SPDX-License-Identifier: AGPL-3.0-or-later
"""The numpy fallback message must say WHICH of three states it hit."""

from __future__ import annotations

import logging
import types

import pytest

import localm.rag.store as store


@pytest.fixture(autouse=True)
def _fresh_once_per_process_flag(monkeypatch):
    """The warning fires once per process; give every test a clean slate."""
    monkeypatch.setattr(store, "_NUMPY_DEGRADE_LOGGED", set())


def _messages(caplog):
    return [r.getMessage() for r in caplog.records]


def test_absent_numpy_is_not_reported_as_a_broken_install(monkeypatch, caplog):
    """State 1."""
    monkeypatch.setattr(store, "_numpy", None)
    monkeypatch.setattr(store, "_NUMPY_IS_STUB", False)

    with caplog.at_level(logging.DEBUG, logger="localm"):
        store._warn_numpy_degrade(
            ImportError("numpy is not installed"), "cosine similarity")

    joined = " ".join(_messages(caplog))
    assert "present but unusable" not in joined, (
        "absence was reported as a broken install - the message contradicts "
        "itself, and it is what every numpy-less user saw")
    assert "EMPTY NAMESPACE PACKAGE" not in joined
    assert "not installed" in joined
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], (
        "a default install with no numpy must not raise a WARNING")


def test_a_namespace_stub_is_still_reported_loudly(monkeypatch, caplog):
    """State 2 must keep its loud, specific message - the fix for state 1 must not quieten the case that actually indicates a broken environment."""
    fake = types.ModuleType("numpy")          # no __file__ -> looks like a stub
    fake.__path__ = ["/somewhere/stray/numpy"]
    monkeypatch.setattr(store, "_numpy", fake)
    monkeypatch.setattr(store, "_NUMPY_IS_STUB", True)

    with caplog.at_level(logging.DEBUG, logger="localm"):
        store._warn_numpy_degrade(
            AttributeError("module 'numpy' has no attribute 'asarray'"),
            "cosine similarity")

    joined = " ".join(_messages(caplog))
    assert "EMPTY NAMESPACE PACKAGE" in joined
    assert "/somewhere/stray/numpy" in joined, "it must name the artefact to delete"
    assert "present but unusable" not in joined
    assert [r for r in caplog.records if r.levelno >= logging.WARNING], (
        "a stub numpy IS a broken install and must warn")


def test_a_real_but_unusable_numpy_still_says_so(monkeypatch, caplog):
    """State 3: numpy is a real install (has __file__) that failed anyway."""
    fake = types.ModuleType("numpy")
    fake.__file__ = "/real/site-packages/numpy/__init__.py"
    monkeypatch.setattr(store, "_numpy", fake)
    monkeypatch.setattr(store, "_NUMPY_IS_STUB", False)

    with caplog.at_level(logging.DEBUG, logger="localm"):
        store._warn_numpy_degrade(ValueError("something odd"), "cosine similarity")

    joined = " ".join(_messages(caplog))
    assert "present but unusable" in joined
    assert "EMPTY NAMESPACE PACKAGE" not in joined
    assert [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_the_three_states_do_not_share_a_message(monkeypatch, caplog):
    """The whole point, asserted directly: three states, three DISTINCT messages."""
    seen = []
    fake_stub = types.ModuleType("numpy")
    fake_stub.__path__ = ["/stray/numpy"]
    fake_real = types.ModuleType("numpy")
    fake_real.__file__ = "/real/numpy/__init__.py"

    for mod, is_stub, exc in (
            (None, False, ImportError("numpy is not installed")),
            (fake_stub, True, AttributeError("no attribute 'asarray'")),
            (fake_real, False, ValueError("odd")),
    ):
        monkeypatch.setattr(store, "_numpy", mod)
        monkeypatch.setattr(store, "_NUMPY_IS_STUB", is_stub)
        monkeypatch.setattr(store, "_NUMPY_DEGRADE_LOGGED", set())
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="localm"):
            store._warn_numpy_degrade(exc, "cosine similarity")
        seen.append(" ".join(_messages(caplog)))

    assert len(set(seen)) == 3, (
        "two of the three numpy states produced the SAME message:\n  "
        + "\n  ".join(seen))
