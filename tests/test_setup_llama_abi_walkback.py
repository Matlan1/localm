# SPDX-License-Identifier: AGPL-3.0-or-later
"""The installer must never hand the user a runtime our OWN ABI gate rejects.

`_latest_tag()` resolves upstream's newest release at RUNTIME, so localm can be
handed a build it refuses to load without any localm change at all - upstream
publishes, and the next `setup-llama` on any installed version picks it up. That
happened three times in one week.

This is a SAFETY NET, not the remedy. The remedy is binding the new struct
layout. Nothing here names a specific tag, and none of it should: the property
is "never ship a runtime our gate rejects", whichever tag and whatever the cause.
Once a layout is bound, the rejection stops and this code simply never runs.

Why the walk is over TAGS and not backends: an ABI rejection means the BUILD is
wrong for this code, so every backend from that release fails identically (field
issue 1208 reports cuda, vulkan AND cpu all AbiMismatch together). The existing
backend fallback cannot help, and running it first would move the user off the
backend they asked for to fix something that was never the backend's fault.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from localm import config as cfg
from localm import setup_llama as sl


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Throwaway LOCALM_HOME with config.py's frozen paths redirected, so the
    pin these tests read is never the real user's."""
    h = tmp_path / ".localm"
    h.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    return h


def _flat(capsys) -> str:
    """Captured console output with whitespace collapsed.

    Rich hard-wraps to the terminal width, so a phrase can arrive split across a
    line break - measured here as "does not match this " / "build of localm".
    Asserting on the raw text would pin the WRAPPING rather than the message, and
    would go red on a different console width: a test failing for a reason
    unrelated to the property it defends."""
    return " ".join(capsys.readouterr().out.split())


# --------------------------------------------------------------------------- #
#  The probe: an ABI rejection is told apart STRUCTURALLY, not by text          #
# --------------------------------------------------------------------------- #

def _stub_localm_tree(root: Path, loader_body: str) -> None:
    """A minimal importable `localm.inference.backends.llamacpp` whose load_lib
    does whatever *loader_body* says. Lets the REAL probe string run in a REAL
    subprocess against a controlled failure."""
    pkg = root / "localm" / "inference" / "backends" / "llamacpp"
    pkg.mkdir(parents=True)
    for d in (root / "localm", root / "localm" / "inference",
              root / "localm" / "inference" / "backends", pkg):
        (d / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "_abi.py").write_text(
        "class AbiMismatch(Exception):\n    pass\n", encoding="utf-8")
    (pkg / "_loader.py").write_text(loader_body, encoding="utf-8")


def _run_probe(root: Path):
    """Run the ACTUAL probe string the installer uses. `python -c` puts cwd on
    sys.path, so the stub tree above shadows the real package."""
    return subprocess.run([sys.executable, "-c", sl._LOAD_PROBE_CODE],
                          cwd=str(root), capture_output=True, text=True, timeout=60)


def test_probe_exits_abi_code_when_load_lib_raises_abimismatch(tmp_path):
    """The load-bearing link, exercised for real rather than with a faked exit
    code: if the runtime's layout is refused, the probe must say so in a way the
    parent can act on. Matching upstream's wording in a traceback would depend on
    a string that is not ours to control."""
    _stub_localm_tree(tmp_path, (
        "from ._abi import AbiMismatch\n"
        "def load_lib():\n"
        "    raise AbiMismatch('llama ABI mismatch: model_params.load_mode = -1')\n"
        "def compute_backends_available():\n"
        "    return True\n"))
    r = _run_probe(tmp_path)
    assert r.returncode == sl._PROBE_ABI_MISMATCH, r.stderr
    assert "load_mode" in r.stderr, "the cause must reach the parent"


def test_probe_exits_zero_on_a_healthy_load(tmp_path):
    """The control for the test above: the probe must be able to report the
    OPPOSITE outcome, or its ABI verdict means nothing."""
    _stub_localm_tree(tmp_path, (
        "def load_lib():\n    return object()\n"
        "def compute_backends_available():\n    return True\n"))
    assert _run_probe(tmp_path).returncode == 0


def test_probe_exits_no_backends_code_when_nothing_registers(tmp_path):
    """And the third outcome stays distinct - a build that loads but registers
    no compute backend is not an ABI rejection and must not trigger a walk-back."""
    _stub_localm_tree(tmp_path, (
        "def load_lib():\n    return object()\n"
        "def compute_backends_available():\n    return False\n"))
    assert _run_probe(tmp_path).returncode == sl._PROBE_NO_BACKENDS


def test_probe_lets_an_ordinary_load_failure_through_as_a_crash(tmp_path):
    """A non-ABI exception must NOT be laundered into the ABI code - that would
    make the walk-back fire on a machine problem an older release cannot fix."""
    _stub_localm_tree(tmp_path, (
        "def load_lib():\n    raise OSError('libcuda.so.1: cannot open shared object file')\n"
        "def compute_backends_available():\n    return True\n"))
    r = _run_probe(tmp_path)
    assert r.returncode not in (0, sl._PROBE_ABI_MISMATCH, sl._PROBE_NO_BACKENDS)


class _Completed:
    def __init__(self, returncode, stderr="", stdout=""):
        self.returncode, self.stderr, self.stdout = returncode, stderr, stdout


def test_native_loads_ok_reports_an_abi_rejection_recognisably(monkeypatch):
    monkeypatch.setattr(sl.subprocess, "run",
                        lambda *a, **k: _Completed(
                            sl._PROBE_ABI_MISMATCH,
                            stderr="AbiMismatch: llama ABI mismatch: load_mode = -1"))
    ok, detail = sl._native_loads_ok()
    assert ok is False
    assert sl._is_abi_rejection(detail), detail
    assert "load_mode" in detail, "the cause must survive into the detail"


@pytest.mark.parametrize("rc,stderr", [
    (1, "OSError: libcuda.so.1: cannot open shared object file"),
    (sl._PROBE_NO_BACKENDS, ""),
])
def test_other_failures_are_not_abi_rejections(monkeypatch, rc, stderr):
    """The discriminator has to say NO too. A driver problem must reach the
    backend fallback, because no older release can fix this machine."""
    monkeypatch.setattr(sl.subprocess, "run",
                        lambda *a, **k: _Completed(rc, stderr=stderr))
    ok, detail = sl._native_loads_ok()
    assert ok is False
    assert not sl._is_abi_rejection(detail), detail


# --------------------------------------------------------------------------- #
#  _recent_tags: the candidate list, from the call that already happened        #
# --------------------------------------------------------------------------- #

def _releases(*entries):
    out = []
    for tag, assets, draft, pre in entries:
        out.append({"tag_name": tag, "assets": [{"name": "x"}] if assets else [],
                    "draft": draft, "prerelease": pre})
    return out


def _fake_releases_api(monkeypatch, payload, seen=None):
    import io
    import json as _json

    class _R:
        def read(self):
            return _json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _open(req, timeout=None):
        if seen is not None:
            seen.append(getattr(req, "full_url", str(req)))
        return _R()

    monkeypatch.setattr(sl, "verified_urlopen", _open)
    return io


def test_recent_tags_is_newest_first_and_skips_unusable_releases(monkeypatch):
    _fake_releases_api(monkeypatch, _releases(
        ("b300", False, False, False),   # published but assets not uploaded yet
        ("b299", True, True, False),     # draft
        ("b298", True, False, True),     # prerelease
        ("b297", True, False, False),
        ("b296", True, False, False),
    ))
    assert sl._recent_tags() == ["b297", "b296"]


def test_recent_tags_is_empty_when_the_lookup_fails(monkeypatch):
    def _boom(req, timeout=None):
        raise OSError("no network")
    monkeypatch.setattr(sl, "verified_urlopen", _boom)
    assert sl._recent_tags() == []


def test_latest_tag_still_takes_the_newest_usable_release(monkeypatch):
    _fake_releases_api(monkeypatch, _releases(
        ("b300", False, False, False), ("b297", True, False, False)))
    assert sl._latest_tag() == "b297"


def test_latest_tag_falls_back_when_nothing_is_usable(monkeypatch, capsys):
    monkeypatch.setattr(sl, "_recent_tags", lambda *a, **k: [])
    assert sl._latest_tag() == sl._FALLBACK_TAG
    assert "rerun later" in _flat(capsys), "the fallback must be visible"


# --------------------------------------------------------------------------- #
#  The walk-back                                                                #
# --------------------------------------------------------------------------- #

def _walk(monkeypatch, home, *, recent, loads, rejected="b300", pin=None):
    """Drive _walk_back_tags with a recorded try/load sequence.
    Returns (result, tags_tried)."""
    if pin:
        sl.set_pinned_tag(pin)
    monkeypatch.setattr(sl, "_recent_tags", lambda *a, **k: list(recent))
    tried: list = []

    def try_fn(backend, cudart, tag=None):
        tried.append(tag)

    seq = iter(loads)
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: next(seq))
    got = sl._walk_back_tags("vulkan", False, rejected, try_fn, "layout drift")
    return got, tried


def test_walk_back_installs_the_first_older_release_that_loads(monkeypatch, home, capsys):
    (ok, tag), tried = _walk(monkeypatch, home,
                             recent=["b300", "b299", "b298"],
                             loads=[(False, "still wrong"), (True, "")])
    assert (ok, tag) == (True, "b298")
    assert tried == ["b299", "b298"], "oldest-last, and the rejected tag skipped"
    out = _flat(capsys)
    assert "b298" in out and "b300" in out, "say which build, and instead of what"
    assert "does not match this build of localm" in out, "say WHY it is not the newest"


def test_walk_back_is_bounded(monkeypatch, home):
    """An unbounded scan would download a runtime per iteration against a
    genuinely broken environment."""
    recent = [f"b{300 - i}" for i in range(12)]
    (ok, _tag), tried = _walk(monkeypatch, home, recent=recent,
                              loads=[(False, "no")] * 12)
    assert ok is False
    assert len(tried) == sl._ABI_WALKBACK_MAX, tried


def test_walk_back_never_moves_off_a_user_pin(monkeypatch, home, capsys):
    """A pin is an explicit choice, so walking off it is the exact override the
    project forbids. Report it and stop, naming both escape commands."""
    (ok, tag), tried = _walk(monkeypatch, home, recent=["b300", "b299"],
                             loads=[], pin="b300")
    assert (ok, tag) == (False, None)
    assert tried == [], "a pinned build must not be walked back from"
    out = _flat(capsys)
    assert "--rollback" in out and "--tag latest" in out
    assert sl.pinned_tag() == "b300", "and the pin itself is untouched"


def test_walk_back_reports_when_no_older_release_exists(monkeypatch, home, capsys):
    (ok, _t), tried = _walk(monkeypatch, home, recent=["b300"], loads=[])
    assert ok is False and tried == []
    assert "No older llama.cpp release" in _flat(capsys)


def test_walk_back_survives_a_provision_error_and_keeps_going(monkeypatch, home):
    """One release failing to download must not end the recovery."""
    sl.set_pinned_tag(None)
    monkeypatch.setattr(sl, "_recent_tags", lambda *a, **k: ["b300", "b299", "b298"])
    tried: list = []

    def try_fn(backend, cudart, tag=None):
        tried.append(tag)
        if tag == "b299":
            raise OSError("404")

    monkeypatch.setattr(sl, "_native_loads_ok", lambda: (True, ""))
    ok, tag = sl._walk_back_tags("vulkan", False, "b300", try_fn, "drift")
    assert (ok, tag) == (True, "b298")
    assert tried == ["b299", "b298"]


# --------------------------------------------------------------------------- #
#  Wiring: the walk-back runs, and runs BEFORE the backend fallback             #
# --------------------------------------------------------------------------- #

def _wire(monkeypatch, tmp_path, loads):
    lib = sl._lib_name()
    provisioned: list = []

    def fake_provision(backend, target, sha256, with_cudart,
                       cuda_line=sl._CUDA_LINE, tag=None):
        provisioned.append((backend, tag))
        (target / lib).write_bytes(b"x")
        return tag or "b300"

    monkeypatch.setattr(sl, "_provision_backend", fake_provision)
    monkeypatch.setattr(sl, "_clear_target_or_refuse", lambda t: None)
    monkeypatch.setattr(sl, "_install_runtime_wheel", lambda d: True)
    monkeypatch.setattr(sl, "_recent_tags", lambda *a, **k: ["b300", "b299"])
    seq = iter(loads)
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: next(seq))
    monkeypatch.setattr(sl.sys.stdin, "isatty", lambda: False)
    return provisioned


def test_an_abi_rejection_walks_tags_and_keeps_the_users_backend(
        monkeypatch, tmp_path, home):
    """The property, end to end: the user asked for cuda, the newest release is
    refused by our gate, and they end on cuda from an older release - NOT on
    vulkan. Walking backends first would have swapped their pick to fix
    something that was never the backend's fault."""
    abi = f"{sl._ABI_REJECT_PREFIX}: load_mode = -1"
    provisioned = _wire(monkeypatch, tmp_path, [(False, abi), (True, "")])

    backend, tag = sl._provision_with_fallback("cuda", tmp_path, None, True,
                                               assume_yes=True)

    assert backend == "cuda", "the user's backend must survive a tag walk-back"
    assert tag == "b299"
    assert provisioned == [("cuda", None), ("cuda", "b299")]
    assert all(b == "cuda" for b, _ in provisioned), "no backend fallback ran"


def test_a_non_abi_failure_still_falls_back_by_backend(monkeypatch, tmp_path, home):
    """The discriminator's other side, and the reason it is not 'any load
    failure': a too-old driver is about this MACHINE, so an older release cannot
    help and the vulkan fallback must still happen."""
    provisioned = _wire(monkeypatch, tmp_path,
                        [(False, "libcuda.so.1: cannot open shared object file"),
                         (True, "")])

    backend, _tag = sl._provision_with_fallback("cuda", tmp_path, None, True,
                                                assume_yes=True)

    assert backend == "vulkan"
    assert [b for b, _ in provisioned] == ["cuda", "vulkan"]
    assert all(t is None for _b, t in provisioned), "no tag walk-back was attempted"


def test_walk_back_failing_still_reaches_the_backend_fallback(
        monkeypatch, tmp_path, home):
    """If no older release loads either, it was not release drift after all, so
    the existing recovery must still get its turn rather than being swallowed."""
    abi = f"{sl._ABI_REJECT_PREFIX}: load_mode = -1"
    provisioned = _wire(monkeypatch, tmp_path,
                        [(False, abi), (False, abi), (True, "")])

    backend, _tag = sl._provision_with_fallback("cuda", tmp_path, None, True,
                                                assume_yes=True)

    assert backend == "vulkan"
    assert [b for b, _ in provisioned] == ["cuda", "cuda", "vulkan"]
