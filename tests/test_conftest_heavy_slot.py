# SPDX-License-Identifier: AGPL-3.0-or-later
"""The heavy_slot lock helpers: a stale slot is taken over, a live one is waited
on, and a release removes only the slot its own acquire created.

The helpers are reached through the tests/conftest.py module already loaded in
this session (found by its ``__file__`` in ``sys.modules``) rather than by
importing the file again, which would re-execute it and re-arm every guard in
it.

Every slot here lives under the test's own ``tmp_path``. The real shared slot
under ``tempfile.gettempdir()`` is never touched: other runs on the same
machine are serialised on it.
"""

from __future__ import annotations

import inspect
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

_REAL_CONFTEST = os.path.normcase(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "conftest.py")))


def _live_conftest():
    """The tests/conftest.py module object loaded in THIS session."""
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if isinstance(file, str) and os.path.normcase(os.path.abspath(file)) == _REAL_CONFTEST:
            return module
    raise AssertionError("tests/conftest.py is not loaded in this session")


@pytest.fixture
def helpers():
    """``(_acquire_heavy_slot, _release_heavy_slot)`` from the live conftest."""
    module = _live_conftest()
    return module._acquire_heavy_slot, module._release_heavy_slot


def _age_to(slot: Path, seconds: float) -> None:
    """Move ``slot``'s mtime ``seconds`` into the past and check that it moved."""
    past = time.time() - seconds
    os.utime(slot, (past, past))
    assert time.time() - slot.stat().st_mtime > seconds - 5, "utime did not age the slot"


def _content(slot: Path) -> str:
    return slot.read_text(encoding="utf-8")


def test_stale_leftover_is_taken_over(tmp_path, helpers):
    """A slot older than stale_after is taken at once, not waited out."""
    acquire, _ = helpers
    slot = tmp_path / "slot"
    slot.write_text("crashed-holder", encoding="utf-8")
    _age_to(slot, 300)

    started = time.monotonic()
    token = acquire(slot, stale_after=5.0)
    elapsed = time.monotonic() - started

    assert _content(slot) == token, "the stale leftover was not taken over"
    assert token is not None
    assert elapsed < 2.5, f"a stale slot was waited on for {elapsed:.2f}s instead of being taken"


def test_live_holder_past_stale_after_is_reclaimed_and_cannot_release_the_new_owner(
        tmp_path, helpers):
    """A holder still on the slot past stale_after is taken over, and its own
    release then leaves the new owner's slot in place."""
    acquire, release = helpers
    slot = tmp_path / "slot"
    first = acquire(slot, stale_after=0.5)
    assert _content(slot) == first

    started = time.monotonic()
    second = acquire(slot, stale_after=0.5)
    elapsed = time.monotonic() - started

    assert _content(slot) == second, "the holder past stale_after was not taken over"
    assert second is not None and second != first
    assert elapsed >= 0.4, f"a live holder was taken over after only {elapsed:.2f}s"

    release(slot, first)
    assert _content(slot) == second, "the reclaimed holder's release removed the new owner's slot"
    release(slot, second)
    assert not slot.exists()


def test_release_unlinks_only_an_owned_slot(tmp_path, helpers):
    acquire, release = helpers
    slot = tmp_path / "slot"
    token = acquire(slot, stale_after=5.0)
    assert token is not None

    release(slot, "not-the-owner")
    assert _content(slot) == token, "a non-owner release removed the slot"
    release(slot, None)
    assert _content(slot) == token, "a token-less release removed the slot"

    release(slot, token)
    assert not slot.exists(), "the owner's release left its own slot behind"
    release(slot, token)
    assert not slot.exists()


def test_release_is_silent_when_the_unlink_is_refused(tmp_path, helpers, monkeypatch):
    acquire, release = helpers
    slot = tmp_path / "slot"
    token = acquire(slot, stale_after=5.0)
    assert _content(slot) == token

    def refuse(self, missing_ok=False):
        raise PermissionError(13, "unlink refused by the test", str(self))

    monkeypatch.setattr(Path, "unlink", refuse)
    release(slot, token)
    assert _content(slot) == token


def test_second_waiter_waits_behind_a_live_holder_until_release(tmp_path, helpers):
    """With the holder well inside stale_after, the only way in is the release."""
    acquire, release = helpers
    slot = tmp_path / "slot"
    first = acquire(slot, stale_after=60.0)
    assert first is not None

    result: list = []
    waiter = threading.Thread(
        target=lambda: result.append(acquire(slot, stale_after=60.0)), daemon=True)
    waiter.start()
    try:
        waiter.join(timeout=0.3)
        assert _content(slot) == first, "the second waiter took a live holder's slot"
        assert waiter.is_alive() and not result, "the second waiter did not wait"

        release(slot, first)
        waiter.join(timeout=10.0)
        assert not waiter.is_alive(), "the second waiter did not acquire after the release"
        assert result and result[0] is not None
        assert _content(slot) == result[0]
    finally:
        try:
            slot.unlink()
        except OSError:
            pass
        waiter.join(timeout=10.0)


def test_un_owned_timeout_leaves_the_slot_alone(tmp_path, helpers, monkeypatch):
    """When the stale slot cannot be unlinked, acquire gives up within its
    budget without ownership, and the release of that outcome touches nothing."""
    acquire, release = helpers
    slot = tmp_path / "slot"
    slot.write_text("crashed-holder", encoding="utf-8")
    _age_to(slot, 300)

    refused: list = []

    def refuse(self, missing_ok=False):
        refused.append(self)
        raise PermissionError(13, "unlink refused by the test", str(self))

    monkeypatch.setattr(Path, "unlink", refuse)
    started = time.monotonic()
    token = acquire(slot, stale_after=0.5)
    elapsed = time.monotonic() - started

    assert refused, "the reclaim never tried to unlink the stale slot"
    assert _content(slot) == "crashed-holder", "a slot that could not be unlinked was rewritten"
    assert token is None, "reported ownership of a slot it never created"
    assert elapsed < 5.0, f"waited {elapsed:.2f}s against a 0.5s budget"

    attempts = len(refused)
    release(slot, token)
    assert _content(slot) == "crashed-holder"
    assert len(refused) == attempts, "an un-owned release tried to unlink the slot"


def test_slot_whose_stat_fails_is_waited_on_within_the_budget(tmp_path, helpers, monkeypatch):
    """A slot that exists but cannot be stat-ed is neither reclaimed nor spun
    on: acquire gives up within its budget without ownership."""
    acquire, _ = helpers
    slot = tmp_path / "slot"
    slot.write_text("unreadable-holder", encoding="utf-8")

    def refuse(self, *args, **kwargs):
        raise PermissionError(13, "stat refused by the test", str(self))

    monkeypatch.setattr(Path, "stat", refuse)
    result: list = []
    thread = threading.Thread(
        target=lambda: result.append(acquire(slot, stale_after=0.5)), daemon=True)
    thread.start()
    thread.join(timeout=10.0)

    assert _content(slot) == "unreadable-holder", "a slot that could not be stat-ed was taken"
    assert not thread.is_alive(), "acquire did not give up within its budget"
    assert result == [None], "reported ownership of a slot it never created"


def test_fixture_takes_and_releases_the_shared_slot_name(tmp_path, monkeypatch, request):
    """The fixture takes ONE slot named localm-subprocess-heavy.slot under
    tempfile.gettempdir() with the 240 s default, and its teardown releases it.

    gettempdir is pointed at this test's tmp_path before the fixture runs, so
    the real shared slot is never touched."""
    conftest = _live_conftest()
    default = inspect.signature(conftest._acquire_heavy_slot).parameters["stale_after"].default
    assert default == 240.0

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    slot = tmp_path / "localm-subprocess-heavy.slot"

    def after_the_fixture_tore_down():
        assert not slot.exists(), "heavy_slot's teardown left its slot behind"

    request.addfinalizer(after_the_fixture_tore_down)
    request.getfixturevalue("heavy_slot")

    assert slot.is_file(), "heavy_slot did not take localm-subprocess-heavy.slot under gettempdir()"
    assert re.fullmatch(r"\d+:[0-9a-f]{32}", _content(slot)), "the slot does not carry an owner token"
