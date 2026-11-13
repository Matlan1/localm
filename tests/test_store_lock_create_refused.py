# SPDX-License-Identifier: AGPL-3.0-or-later
"""storekit.store_lock when creating its lock file is refused with access denied.

On Windows a lock file another process is deleting (delete pending) or that a
scanner holds open refuses the O_CREAT|O_EXCL create with a PermissionError
instead of FileExistsError. store_lock waits that out on its own deadline; a
refusal that never clears ends in a clear error, never an endless wait. On
POSIX the refusal is a real denial and is raised at once.

The refusal is injected at os.open for the lock path only; every other path,
and every later create once the injection is spent, is the real call."""

import ast
import os
import time
from pathlib import Path

import pytest

from localm import storekit
from localm.storekit import ConfigLockTimeout, store_lock

_windows_only = pytest.mark.skipif(os.name != "nt", reason="the wait is Windows-only")


def _refuse_lock_create(monkeypatch, lockpath, times, *, first_real=0):
    """Patch os.open so the exclusive create of *lockpath* raises
    PermissionError(13) *times* times, after letting the first *first_real*
    creates of it through to the real call. Returns the counters."""
    real = os.open
    state = {"left": times, "real_left": first_real, "creates": 0}

    def fake(path, flags, *a, **kw):
        if str(path) == str(lockpath) and (flags & os.O_CREAT) and (flags & os.O_EXCL):
            state["creates"] += 1
            if state["real_left"]:
                state["real_left"] -= 1
                return real(path, flags, *a, **kw)
            if state["left"]:
                state["left"] -= 1
                raise PermissionError(13, "Permission denied", str(path))
        return real(path, flags, *a, **kw)

    monkeypatch.setattr(os, "open", fake)
    return state


@_windows_only
def test_store_lock_waits_out_access_denied_on_the_lock_create(tmp_path, monkeypatch):
    target = tmp_path / "auth.json"
    lockpath = tmp_path / "auth.json.lock"
    state = _refuse_lock_create(monkeypatch, lockpath, 3)

    with store_lock(target, timeout=30) as held:
        held.assert_held()
        assert lockpath.exists()

    assert state["left"] == 0, "the injected access-denied never fired"
    assert state["creates"] == 4
    assert not lockpath.exists(), "the released lock file was left behind"


@_windows_only
def test_store_lock_access_denied_that_never_clears_raises_permission_error(
        tmp_path, monkeypatch):
    target = tmp_path / "auth.json"
    lockpath = tmp_path / "auth.json.lock"
    state = _refuse_lock_create(monkeypatch, lockpath, 10_000)

    started = time.time()
    with pytest.raises(OSError) as info:
        with store_lock(target, timeout=0.5):
            pass
    elapsed = time.time() - started
    caught = info.value

    assert not lockpath.exists()
    assert isinstance(caught, PermissionError), repr(caught)
    assert not isinstance(caught, TimeoutError), repr(caught)
    assert "auth.json.lock" in str(caught)
    assert "not writable" in str(caught)
    assert isinstance(caught.__cause__, PermissionError)
    assert state["creates"] > 1, "the refusal was not retried"
    assert 0.5 <= elapsed < 30, f"the wait was not bounded by the timeout ({elapsed:.2f}s)"


@_windows_only
def test_store_lock_held_then_access_denied_reports_a_lock_timeout(tmp_path, monkeypatch):
    target = tmp_path / "auth.json"
    lockpath = tmp_path / "auth.json.lock"
    lockpath.write_bytes(storekit._lock_token(os.getppid(), "a-live-holder"))
    state = _refuse_lock_create(monkeypatch, lockpath, 10_000, first_real=1)
    with pytest.raises(OSError) as info:
        with store_lock(target, timeout=0.5):
            pass
    caught = info.value

    assert lockpath.exists(), "the live holder's lock was removed"
    assert isinstance(caught, ConfigLockTimeout), repr(caught)
    assert isinstance(caught.__cause__, PermissionError)
    assert state["creates"] > 2


@pytest.mark.skipif(os.name == "nt", reason="POSIX raises the denial at once")
def test_store_lock_access_denied_on_posix_is_raised_at_once(tmp_path, monkeypatch):
    target = tmp_path / "auth.json"
    lockpath = tmp_path / "auth.json.lock"
    state = _refuse_lock_create(monkeypatch, lockpath, 10_000)

    started = time.time()
    with pytest.raises(PermissionError) as info:
        with store_lock(target, timeout=5):
            pass

    assert not isinstance(info.value, TimeoutError)
    assert state["creates"] == 1
    assert time.time() - started < 2


def _excl_flag(node):
    return any(isinstance(n, ast.Attribute) and n.attr == "O_EXCL" for n in ast.walk(node))


def _is_os_open_excl(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "open" and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os" and any(_excl_flag(a) for a in node.args))


def _handler_names(handler):
    t = handler.type
    elts = t.elts if isinstance(t, ast.Tuple) else [t]
    return {e.id for e in elts if isinstance(e, ast.Name)}


def test_every_lock_file_create_decides_access_denied_through_storekit():
    """Every os.open(..., O_EXCL) in localm sits in a try whose PermissionError
    handler calls storekit.lock_create_refusal_is_transient."""
    root = Path(storekit.__file__).resolve().parent
    sites, missing = [], []
    for py in sorted(root.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        parent = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for call in [n for n in ast.walk(tree) if _is_os_open_excl(n)]:
            site = f"{py.relative_to(root.parent)}:{call.lineno}"
            sites.append(site)
            node, enclosing = call, None
            while node in parent:
                up = parent[node]
                if isinstance(up, ast.Try) and node in up.body:
                    enclosing = up
                    break
                node = up
            decided = enclosing is not None and any(
                "PermissionError" in _handler_names(h) and any(
                    isinstance(c, ast.Call)
                    and getattr(c.func, "id", getattr(c.func, "attr", None))
                    == "lock_create_refusal_is_transient"
                    for c in ast.walk(h))
                for h in enclosing.handlers)
            if not decided:
                missing.append(site)
    assert len(sites) >= 2, f"expected store_lock and collection_lock, found {sites}"
    assert not missing, f"lock-file creates that do not decide through storekit: {missing}"
