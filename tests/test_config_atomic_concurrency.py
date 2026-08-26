# SPDX-License-Identifier: AGPL-3.0-or-later
"""The config/registry "atomic write" (temp file + os.replace) is atomic, but on
Windows ``os.replace`` raises PermissionError (WinError 5) when ANOTHER handle has
the destination open at that instant - a second localm process reading the file,
an antivirus / indexer / backup scanner, Windows Search. The window is
microseconds, so a bare os.replace makes a config/registry SAVE crash (and a
concurrent read spuriously fall back to .bak/defaults) whenever a reader touches
the file mid-write.

Both sides ride out the transient sharing violation with a bounded retry, while
a PERSISTENT permission problem still surfaces. These tests inject the transient
fault deterministically at the OS boundary, so they exercise the real
_atomic_write_json / _read_json code paths on every platform."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import localm.config as cfg


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    h.mkdir()
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    # keep retries fast in tests
    monkeypatch.setattr(cfg, "_REPLACE_BACKOFF", 0.001, raising=False)
    return h


def test_atomic_write_rides_out_transient_replace_error(home, monkeypatch):
    """A few transient PermissionErrors from os.replace must not fail the write;
    the file ends up correct."""
    # The retry is Windows-only: _is_transient_permission_error returns False when
    # os.name != nt. This test stubs the classifier rather than pinning os.name;
    # cfg.os IS the os module, so a pin would also switch os.path/tempfile to
    # Windows semantics and break the tmp path _atomic_write_json builds.
    monkeypatch.setattr(cfg, "_is_transient_permission_error", lambda e: True)
    real_replace = cfg.os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        # Fail the final tmp->path replace (dst ends with the real name) a few
        # times, then let it through. The .bak replace is left alone.
        if str(dst).endswith("config.json") and calls["n"] < 3:
            calls["n"] += 1
            err = PermissionError(13, "Access is denied")
            err.winerror = 5                # ERROR_ACCESS_DENIED
            raise err
        return real_replace(src, dst)

    monkeypatch.setattr(cfg.os, "replace", flaky_replace)
    cfg._atomic_write_json(cfg.CONFIG_FILE, {"port": 4242})
    assert calls["n"] == 3, "should have hit and retried the transient error"
    assert json.loads(cfg.CONFIG_FILE.read_text())["port"] == 4242


def test_atomic_write_reraises_persistent_permission_error(home, monkeypatch):
    """A PERSISTENT permission failure is NOT hidden - it surfaces after retries."""
    def always_denied(src, dst):
        if str(dst).endswith("config.json"):
            raise PermissionError(13, "Access is denied")
        return None  # swallow the .bak replace
    monkeypatch.setattr(cfg.os, "replace", always_denied)
    with pytest.raises(PermissionError):
        cfg._atomic_write_json(cfg.CONFIG_FILE, {"port": 1})


def test_read_json_rides_out_transient_permission_error(home, monkeypatch, capsys):
    """A transient PermissionError on read must be retried, not treated as a
    corrupt file - the live data is returned and no scary warning is printed."""
    cfg.REGISTRY_FILE.write_text(json.dumps({"m": {"path": "Z:/x.gguf"}}), encoding="utf-8")
    real_open = cfg.open if hasattr(cfg, "open") else open
    import builtins
    real_open = builtins.open
    state = {"n": 0}
    # Windows-only retry; same classifier stub as above rather than an os.name pin.
    monkeypatch.setattr(cfg, "_is_transient_permission_error", lambda e: True)

    def flaky_open(file, *a, **k):
        if str(file).endswith("registry.json") and state["n"] < 2:
            state["n"] += 1
            err = PermissionError(13, "Access is denied")
            err.winerror = 5                # ERROR_ACCESS_DENIED
            raise err
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", flaky_open)
    reg = cfg._read_json(cfg.REGISTRY_FILE, {})
    assert reg == {"m": {"path": "Z:/x.gguf"}}, "should return the live data after retry"
    assert state["n"] == 2
    err = capsys.readouterr().err.lower()
    assert "unreadable" not in err, "a transient, recovered blip must not warn"


def test_read_json_corrupt_falls_back_without_retry(home, capsys):
    """A genuinely corrupt (non-transient) file falls back immediately."""
    cfg.CONFIG_FILE.write_text("{ not json", encoding="utf-8")
    out = cfg._read_json(cfg.CONFIG_FILE, {"fallback": True})
    assert out == {"fallback": True}
    assert "unreadable" in capsys.readouterr().err.lower()


def test_replace_atomic_rides_out_a_real_file_lock(home):
    """Deterministic REAL lock (not a mock, not a timing race): hold an ACTUAL OS
    read handle on the destination - which makes os.replace fail with WinError 5
    on Windows - and release it from another thread partway through the retries.
    _replace_atomic must ride out the transient lock and land the new content.

    Part 1 proves the held handle really does block a bare replace on Windows, so
    the test exercises a real sharing violation rather than a no-op. On POSIX a
    held read handle does not block os.replace, so Part 1 is skipped and Part 2
    just confirms the write lands without crashing."""
    import sys as _sys
    import threading
    import time as _t

    p = cfg.REGISTRY_FILE
    p.write_text('{"old": 1}', encoding="utf-8")

    if _sys.platform == "win32":
        tmp0 = p.with_name(p.name + ".t0")
        tmp0.write_text("{}", encoding="utf-8")
        fh0 = open(p, "rb")
        try:
            with pytest.raises(PermissionError):
                cfg.os.replace(tmp0, p)  # a bare replace really does fail while held
        finally:
            fh0.close()
            tmp0.unlink()

    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text('{"new": 2}', encoding="utf-8")
    fh = open(p, "rb")

    def releaser():
        _t.sleep(0.05)  # << the ~1 s retry budget; released mid-retry
        fh.close()

    t = threading.Thread(target=releaser)
    t.start()
    cfg._replace_atomic(tmp, p)  # must NOT raise: it retries until the handle frees
    t.join()
    fh.close()  # idempotent; ensures the handle is closed on POSIX too
    assert json.loads(p.read_text()) == {"new": 2}


# --------------------------------------------------------------------------- #
#  Unique per-write temp: two concurrent writers must not collide on one shared
#  temp source.
# --------------------------------------------------------------------------- #

def test_atomic_write_uses_unique_temp_per_write(tmp_path, monkeypatch):
    target = tmp_path / "reg.json"
    names = []
    real_mkstemp = cfg.tempfile.mkstemp

    def spy_mkstemp(*a, **k):
        fd, name = real_mkstemp(*a, **k)
        names.append(name)
        return fd, name

    monkeypatch.setattr(cfg.tempfile, "mkstemp", spy_mkstemp)
    cfg._atomic_write_json(target, {"a": 1})
    cfg._atomic_write_json(target, {"a": 2})

    assert len(names) == 2
    assert names[0] != names[1]                            # unique per write
    assert all(Path(n).name != "reg.json.tmp" for n in names)   # never the old fixed name
    assert json.loads(target.read_text()) == {"a": 2}      # last write wins, intact
    assert not list(tmp_path.glob("reg.json*.tmp"))        # no orphan temp left behind


def test_atomic_write_leaves_no_orphan_temp(tmp_path):
    target = tmp_path / "c.json"
    cfg._atomic_write_json(target, {"x": 1})
    cfg._atomic_write_json(target, {"x": 2})
    assert json.loads(target.read_text()) == {"x": 2}
    assert not list(tmp_path.glob("*.tmp"))


# Each worker process hammers the same file through the real cross-process path,
# with no mocks.
_CONC_NPROC = 5
_CONC_NWRITES = 25
_CONC_WORKER = (
    "import sys\n"
    "from pathlib import Path\n"
    "import localm.config as c\n"
    "p = Path(sys.argv[1]); idx = int(sys.argv[2]); m = int(sys.argv[3])\n"
    "for n in range(m):\n"
    "    c._atomic_write_json(p, {'w': idx, 'n': n})\n"
)


def test_concurrent_processes_no_crash_no_corruption(tmp_path):
    target = tmp_path / "shared.json"
    target.write_text("{}", encoding="utf-8")
    env = dict(os.environ)
    env["LOCALM_HOME"] = str(tmp_path / "home")
    # Make the child import the same localm this test runs, not the venv install.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)

    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _CONC_WORKER, str(target), str(i), str(_CONC_NWRITES)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for i in range(_CONC_NPROC)
    ]
    for i, pr in enumerate(procs):
        out, err = pr.communicate(timeout=90)
        assert pr.returncode == 0, (
            f"writer {i} crashed (rc={pr.returncode}): "
            f"{err.decode('utf-8', 'replace')[-500:]}")

    # The final file is valid JSON (never a torn write) and no temp leaked.
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data.get("w") in range(_CONC_NPROC)
    assert not list(tmp_path.glob("shared.json*.tmp"))
