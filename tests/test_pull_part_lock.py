# SPDX-License-Identifier: AGPL-3.0-or-later
"""Two pulls of the same URL must not interleave into one .part file.

The destination and its ``.part`` are derived from the URL, so two pulls of the
same URL target one file. Each reads the ``.part``'s current size to decide
append-or-truncate, and with no lock the second one reads a size the first is
still changing. They then write into the same handle: the download "succeeds"
and fails its hash, or - with no ``--sha256`` to check it against - registers
as a working model that is not one.

THE CONTENDERS ARE PROCESSES, NOT THREADS. The GUI starts a pull by spawning
``localm pull`` as a child, and a user can run the same command in a terminal
at the same time. So a ``threading.Lock`` would serialise nothing here while
looking, in review, exactly like a correct fix - which is why the central test
in this file drives TWO REAL INTERPRETERS. A test that monkeypatched the
liveness check could not demonstrate atomicity across processes, and atomicity
across processes is the entire claim.

STALENESS IS DECIDED BY PID LIVENESS, NEVER BY ELAPSED TIME. Any fixed timeout
eventually reclaims a live holder's lock and recreates the corruption, and a
large model on a slow link is exactly the download that outlives a generous
one. Every uncertainty therefore KEEPS the lock, and the tests below pin both
directions: a live holder is never evicted, a proven-dead one always is.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import localm.config as config
import localm.model_manager as model_manager
from localm.model_manager.pull import (
    PullInFlight,
    _part_lock,
    _part_lock_dir,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    (h / "models").mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(model_manager, "MODELS_DIR", h / "models")
    monkeypatch.setattr(config, "HOME_DIR", h)
    monkeypatch.setattr(config, "MODELS_DIR", h / "models")
    monkeypatch.setattr(config, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(config, "REGISTRY_FILE", h / "registry.json")
    return h


def _worktree_root() -> str:
    """The checkout THIS test is running from.

    A child started with ``python -c`` resolves ``localm`` through the venv's
    editable-install .pth, which points at the main checkout - so without this
    the subprocesses below would exercise a different tree than the one under
    test and pass or fail for reasons unrelated to this diff.
    """
    import localm
    return str(Path(localm.__file__).resolve().parent.parent)


# --------------------------------------------------------------------------
#  The claim: atomicity across real processes
# --------------------------------------------------------------------------

CONTEND = textwrap.dedent('''
    import json, os, sys, time
    import localm
    root = os.environ["EXPECT_ROOT"]
    # Prove the child is running the tree under test, not the venv's editable
    # install of the main checkout. A child on the wrong tree would report a
    # perfectly plausible result about the wrong code.
    assert os.path.normcase(os.path.dirname(os.path.dirname(
        os.path.abspath(localm.__file__)))) == os.path.normcase(root), (
        "child imported localm from " + localm.__file__)
    from localm.model_manager.pull import _part_lock, PullInFlight
    try:
        with _part_lock(sys.argv[1]):
            # Hold it long enough that the sibling is certainly contending for
            # a HELD lock rather than arriving after it was released.
            print("WON", flush=True)
            time.sleep(float(sys.argv[2]))
    except PullInFlight as e:
        print("LOST", flush=True)
''')


def _spawn(script: str, home_dir, *args):
    env = dict(os.environ)
    env["LOCALM_HOME"] = str(home_dir)
    env["EXPECT_ROOT"] = _worktree_root()
    env["PYTHONPATH"] = _worktree_root()
    return subprocess.Popen(
        [sys.executable, "-c", script, *[str(a) for a in args]],
        cwd=_worktree_root(), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_two_real_interpreters_cannot_both_hold_the_lock(home):
    """Exactly one of two real processes may write the .part.

    This is the test the fix exists to pass. It uses no mocks and no patched
    liveness: two OS processes race for the same lock and the OS decides.
    """
    a = _spawn(CONTEND, home, "m.gguf", 2.0)
    b = _spawn(CONTEND, home, "m.gguf", 2.0)
    out_a, err_a = a.communicate(timeout=60)
    out_b, err_b = b.communicate(timeout=60)

    results = sorted([out_a.strip(), out_b.strip()])
    assert a.returncode == 0, err_a
    assert b.returncode == 0, err_b
    assert results == ["LOST", "WON"], (
        f"both processes reported {results} - the lock did not serialise two "
        f"real interpreters.\nA stderr: {err_a}\nB stderr: {err_b}")


def test_two_real_interpreters_on_different_files_both_proceed(home):
    """The lock is per DESTINATION, not global.

    Without this, a lock that simply refused everything would pass the test
    above while making concurrent downloads of unrelated models impossible.
    """
    a = _spawn(CONTEND, home, "one.gguf", 0.2)
    b = _spawn(CONTEND, home, "two.gguf", 0.2)
    out_a, err_a = a.communicate(timeout=60)
    out_b, err_b = b.communicate(timeout=60)

    assert out_a.strip() == "WON", err_a
    assert out_b.strip() == "WON", err_b


def test_the_lock_is_released_when_the_holder_exits(home):
    """A lock taken and dropped by a real process leaves nothing behind."""
    p = _spawn(CONTEND, home, "m.gguf", 0.05)
    out, err = p.communicate(timeout=60)
    assert out.strip() == "WON", err
    assert not _part_lock_dir("m.gguf").exists(), (
        "the lock directory outlived the process that held it")
    with _part_lock("m.gguf"):
        pass


# --------------------------------------------------------------------------
#  Staleness: liveness, never elapsed time
# --------------------------------------------------------------------------

def test_a_live_holders_lock_is_never_reclaimed(home):
    """The direction that matters. A timeout-based rule eventually evicts a
    slow-but-healthy download; this must not, however long it runs."""
    holder = subprocess.Popen([sys.executable, "-c",
                               "import time; time.sleep(30)"])
    try:
        d = _part_lock_dir("m.gguf")
        d.mkdir(parents=True)
        (d / "owner.json").write_text(
            json.dumps({"pid": holder.pid, "filename": "m.gguf",
                        "started": 0.0}), encoding="utf-8")
        # The injection took: a REAL live process owns this lock, and its
        # recorded start time is the epoch - so any elapsed-time rule would
        # call it stale immediately.
        assert holder.poll() is None, "the holder process died before the test"

        with pytest.raises(PullInFlight) as e:
            with _part_lock("m.gguf"):
                pass
        assert str(holder.pid) in str(e.value)
        assert (d / "owner.json").exists(), (
            "the live holder's own lock record was destroyed by the refusal")
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_a_dead_holders_lock_is_reclaimed(home):
    """The permissive direction. A crashed download must not wedge the
    destination forever."""
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait(timeout=30)

    d = _part_lock_dir("m.gguf")
    d.mkdir(parents=True)
    (d / "owner.json").write_text(
        json.dumps({"pid": dead.pid, "filename": "m.gguf",
                    "started": 0.0}), encoding="utf-8")
    assert dead.poll() is not None, "the supposedly dead holder is still alive"

    with _part_lock("m.gguf"):
        rec = json.loads((d / "owner.json").read_text(encoding="utf-8"))
    assert rec["pid"] == os.getpid(), (
        "the lock was not actually re-taken by this process")


@pytest.mark.parametrize("body", [
    None,                     # no owner record at all
    "{not json",              # unreadable
    '{"pid": "banana"}',      # a pid that is not a pid
])
def test_an_unidentifiable_holder_keeps_the_lock(home, body):
    """Uncertainty KEEPS the lock.

    An owner record that cannot be read is not evidence its owner died. The
    refusal names the directory, so a user who is certain can clear it - which
    is the recoverable direction; silently stealing a lock from a live download
    is not.
    """
    d = _part_lock_dir("m.gguf")
    d.mkdir(parents=True)
    if body is not None:
        (d / "owner.json").write_text(body, encoding="utf-8")

    with pytest.raises(PullInFlight) as e:
        with _part_lock("m.gguf"):
            pass
    assert str(d) in str(e.value), str(e.value)


def test_the_lock_is_dropped_even_when_the_download_raises(home):
    """A failing download must not wedge the destination for its own pid."""
    with pytest.raises(RuntimeError):
        with _part_lock("m.gguf"):
            raise RuntimeError("download blew up")
    assert not _part_lock_dir("m.gguf").exists()
    with _part_lock("m.gguf"):
        pass


# --------------------------------------------------------------------------
#  Placement and wiring
# --------------------------------------------------------------------------

def test_the_lock_lives_beside_the_models_dir_not_inside_it(home):
    """A lock under models/ would sit in the path of the thing it guards:
    sync_models_dir walks that tree, and a tidy-up of stray files there would
    be free to delete a live lock."""
    d = _part_lock_dir("m.gguf")
    models = home / "models"
    assert models not in d.parents, f"{d} is inside the models dir"
    assert d.parent.parent == models.parent


def test_pull_url_refuses_and_leaves_the_part_file_untouched(home):
    """The wiring, and a data-first assertion.

    Holding the lock, a second _pull_url must not open the .part at all. The
    bytes are the property; the return value is a proxy for it.
    """
    from unittest.mock import MagicMock

    from localm.model_manager import pull as _pull

    part = home / "models" / "m.gguf.part"
    part.write_bytes(b"FIRST-DOWNLOAD-BYTES")
    before = part.read_bytes()

    # Asserted from OUTSIDE the call, never by raising from a stub: _pull_url
    # catches NetworkPolicyError around exactly this region, and an assertion
    # raised inside code that catches is an input to that code rather than a
    # failure the runner sees.
    reached_network = MagicMock(side_effect=RuntimeError("unreachable"))
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(_pull, "_ssrf_resolve_final_url", reached_network)
    try:
        with _part_lock("m.gguf"):
            # The injection took: the lock is held, so the call below is
            # genuinely contending rather than arriving at a free destination.
            assert _part_lock_dir("m.gguf").exists()
            ok = _pull._pull_url("https://example.invalid/m.gguf", "m")
    finally:
        monkeypatch.undo()

    assert part.read_bytes() == before, (
        "a second pull wrote into the .part file while another download held "
        "it - this is the interleaving that corrupts the download")
    reached_network.assert_not_called()
    assert ok is False
