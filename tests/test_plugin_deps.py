# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for localm.plugins.deps - the host-side pip-extra installer.

Covers the resolver (extra -> requirement strings from metadata), the
already-satisfied check, and install orchestration including the NEGATIVE paths:
a failing installer surfaces the real error and never reports success, and a pip
that exits 0 without actually providing the package is still reported failed.
"""

import subprocess

import pytest

from localm.plugins import deps


# --------------------------------------------------------------------------- #
#  Resolver: extra -> concrete requirement strings                            #
# --------------------------------------------------------------------------- #

class _FakeMeta:
    def __init__(self, reqs):
        self._reqs = reqs

    def get_all(self, key):
        return list(self._reqs) if key == "Requires-Dist" else []


def _patch_metadata(monkeypatch, reqs):
    import importlib.metadata as md
    monkeypatch.setattr(md, "metadata", lambda name: _FakeMeta(reqs))


def test_extra_requirements_reads_marker(monkeypatch):
    _patch_metadata(monkeypatch, [
        "click>=8.4.1",                              # core, no extra
        "faster-whisper>=1.0; extra == 'voice'",
        "pypdf>=4.0; extra == 'rag'",
    ])
    assert deps.extra_requirements("voice") == ["faster-whisper>=1.0"]
    assert deps.extra_requirements("rag") == ["pypdf>=4.0"]


def test_extra_requirements_double_quote_marker(monkeypatch):
    _patch_metadata(monkeypatch, ['soundfile>=0.12; extra == "audio"'])
    assert deps.extra_requirements("audio") == ["soundfile>=0.12"]


def test_extra_requirements_unknown_extra_falls_back(monkeypatch):
    _patch_metadata(monkeypatch, ["faster-whisper>=1.0; extra == 'voice'"])
    # An extra with no matching Requires-Dist falls back to the localm[extra] form
    assert deps.extra_requirements("nope") == ["localm[nope]"]


def test_extra_requirements_metadata_unreadable_falls_back(monkeypatch):
    import importlib.metadata as md

    def boom(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "metadata", boom)
    assert deps.extra_requirements("voice") == ["localm[voice]"]


def test_plugin_requirements_dedups(monkeypatch):
    _patch_metadata(monkeypatch, [
        "faster-whisper>=1.0; extra == 'voice'",
        "shared-lib>=2.0; extra == 'voice'",
        "shared-lib>=2.0; extra == 'other'",
    ])
    reqs = deps.plugin_requirements(["voice", "other"])
    assert reqs == ["faster-whisper>=1.0", "shared-lib>=2.0"]


def test_plugin_requirements_empty():
    assert deps.plugin_requirements([]) == []
    assert deps.plugin_requirements(None) == []


# --------------------------------------------------------------------------- #
#  Requirement-name parsing + satisfied check                                 #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("req,name", [
    ("faster-whisper>=1.0", "faster-whisper"),
    ("pypdf", "pypdf"),
    ("torch==2.9.1", "torch"),
    ("pillow>=10.0,<11", "pillow"),
    ("localm[voice]", "localm"),
    ("pkg @ https://example/x.whl", "pkg"),
])
def test_req_name(req, name):
    assert deps._req_name(req) == name


def test_is_satisfied_present_no_specifier(monkeypatch):
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda n: "1.2.3")
    assert deps.is_satisfied("somepkg") is True


def test_is_satisfied_absent(monkeypatch):
    import importlib.metadata as md

    def boom(n):
        raise md.PackageNotFoundError(n)

    monkeypatch.setattr(md, "version", boom)
    assert deps.is_satisfied("somepkg>=1.0") is False


def test_is_satisfied_version_specifier(monkeypatch):
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda n: "0.9.0")
    # packaging is available in the env; 0.9.0 does NOT satisfy >=1.0
    assert deps.is_satisfied("faster-whisper>=1.0") is False
    monkeypatch.setattr(md, "version", lambda n: "1.5.0")
    assert deps.is_satisfied("faster-whisper>=1.0") is True


def test_is_satisfied_extra_form_never_satisfied(monkeypatch):
    # localm itself is installed, but localm[voice] must NOT count as satisfied
    # just because localm is present - the concrete deps are unknown here.
    assert deps.is_satisfied("localm[voice]") is False


def test_missing_requirements(monkeypatch):
    monkeypatch.setattr(deps, "is_satisfied",
                        lambda r: r == "here>=1.0")
    assert deps.missing_requirements(["here>=1.0", "gone>=2.0"]) == ["gone>=2.0"]


# --------------------------------------------------------------------------- #
#  install_requirements orchestration (mocked pip)                            #
# --------------------------------------------------------------------------- #

def test_install_all_satisfied_skips_pip(monkeypatch):
    monkeypatch.setattr(deps, "is_satisfied", lambda r: True)
    called = {"n": 0}
    monkeypatch.setattr(deps, "_run_pip",
                        lambda reqs, on_progress=None: called.__setitem__("n", 1))
    res = deps.install_requirements(["a>=1", "b>=2"])
    assert res.ok and res.installed == [] and res.skipped == ["a>=1", "b>=2"]
    assert called["n"] == 0            # pip never invoked


def test_install_success(monkeypatch):
    # Missing before, satisfied after the (fake) install.
    state = {"installed": False}
    monkeypatch.setattr(deps, "is_satisfied",
                        lambda r: state["installed"])

    def fake_pip(reqs, on_progress=None):
        state["installed"] = True
        return True, "Successfully installed x"

    monkeypatch.setattr(deps, "_run_pip", fake_pip)
    res = deps.install_requirements(["x>=1"])
    assert res.ok and res.installed == ["x>=1"] and res.failed == []


def test_install_pip_fails_surfaces_error(monkeypatch):
    """NEGATIVE: a failing installer must report ok=False with the real tail,
    never a hollow success."""
    monkeypatch.setattr(deps, "is_satisfied", lambda r: False)
    monkeypatch.setattr(deps, "_run_pip",
                        lambda reqs, on_progress=None: (False, "ERROR: could not build wheel"))
    res = deps.install_requirements(["x>=1"])
    assert res.ok is False
    assert res.failed == ["x>=1"]
    assert "could not build wheel" in res.error


def test_install_pip_lies_still_missing(monkeypatch):
    """NEGATIVE: installer exits 0 but the package is STILL missing afterwards
    -> reported failed, not installed."""
    monkeypatch.setattr(deps, "is_satisfied", lambda r: False)  # never satisfied
    monkeypatch.setattr(deps, "_run_pip",
                        lambda reqs, on_progress=None: (True, "Successfully installed (not really)"))
    res = deps.install_requirements(["x>=1"])
    assert res.ok is False
    assert res.failed == ["x>=1"]
    assert "still missing" in res.error


def test_install_plugin_extras_no_extras_is_noop(monkeypatch):
    monkeypatch.setattr(deps, "_run_pip",
                        lambda *a, **k: pytest.fail("pip must not run for no extras"))
    res = deps.install_plugin_extras([])
    assert res.ok and res.installed == []


def test_progress_sink_receives_lines(monkeypatch):
    monkeypatch.setattr(deps, "is_satisfied", lambda r: True)
    seen = []
    deps.install_requirements(["a>=1"], on_progress=seen.append)
    assert any("already satisfied" in ln for ln in seen)


def test_progress_sink_that_raises_is_ignored(monkeypatch):
    monkeypatch.setattr(deps, "is_satisfied", lambda r: True)

    def bad(_):
        raise RuntimeError("boom")

    # Must not propagate out of the install.
    res = deps.install_requirements(["a>=1"], on_progress=bad)
    assert res.ok


# --------------------------------------------------------------------------- #
#  _run_pip: uv -> pip fallback                                               #
# --------------------------------------------------------------------------- #

class _FakeProc:
    def __init__(self, lines, rc):
        self.stdout = iter(lines)
        self.returncode = rc

    def wait(self):
        pass


def test_run_pip_uv_missing_falls_back_to_pip(monkeypatch):
    calls = []

    def fake_popen(cmd, **kw):
        calls.append(cmd[0])
        if cmd[0] == "uv":
            raise FileNotFoundError("no uv")
        return _FakeProc(["collecting x\n", "installed x\n"], 0)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    ok, out = deps._run_pip(["x>=1"])
    assert ok is True
    assert calls[0] == "uv"                 # tried uv first
    assert "pip" in "".join(str(c) for c in calls[1:]) or calls[1].endswith("python") \
        or "python" in calls[1]             # then the python -m pip fallback
    assert "installed x" in out


def test_run_pip_both_fail_returns_false(monkeypatch):
    def fake_popen(cmd, **kw):
        raise FileNotFoundError("nothing here")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    ok, out = deps._run_pip(["x>=1"])
    assert ok is False and out == ""


def test_run_pip_nonzero_returncode(monkeypatch):
    def fake_popen(cmd, **kw):
        return _FakeProc(["ERROR: boom\n"], 1)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    ok, out = deps._run_pip(["x>=1"])
    assert ok is False and "boom" in out
