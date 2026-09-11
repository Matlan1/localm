# SPDX-License-Identifier: AGPL-3.0-or-later
"""_provision_with_fallback's vulkan/cpu terminal branch: when the user's
own pick is provisioned but does not load, there is no different backend to
fall back to (vulkan/cpu ARE the universal builds). This must:

  - name the missing OS library in plain words when the failure looks like a
    dlopen "cannot open shared object file" (defect 4);
  - never call it a "self-contained fallback" for a PRIMARY pick (defect 2);
  - offer an interactive retry before giving up, and actually retry the
    SAME backend, never a different one (defect 1/3's setup_llama half -
    setup.sh's own continue-without-a-runtime offer is covered separately in
    tests/test_setup_entrypoint_retry.py);
  - always mention --from as an escape hatch in the final message.
"""

from __future__ import annotations

import pytest

from localm import bugreport
from localm import setup_llama as sl


def _wire(monkeypatch, tmp_path, *, loads, retry_ok: bool = True):
    """Same shape as test_setup_llama_abi_walkback.py's _wire: fake
    _provision_backend writes the lib file and records each attempt; the
    load-test sequence is driven from *loads*. _bundle_missing_native_deps is
    stubbed out - this file tests the messaging/retry logic, not bundling
    (covered in test_setup_llama_libgomp_bundle.py)."""
    lib = sl._lib_name()
    provisioned: list = []

    def fake_provision(backend, target, sha256, with_cudart,
                       cuda_line=sl._CUDA_LINE, tag=None):
        provisioned.append(backend)
        if retry_ok:
            (target / lib).write_bytes(b"x")
        return tag or "bTEST"

    monkeypatch.setattr(sl, "_provision_backend", fake_provision)
    monkeypatch.setattr(sl, "_clear_target_or_refuse", lambda t: None)
    monkeypatch.setattr(sl, "_install_runtime_wheel", lambda d: True)
    monkeypatch.setattr(sl, "_bundle_missing_native_deps", lambda t: None)
    seq = iter(loads)
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: next(seq))
    return provisioned


def _flat(capsys) -> str:
    return " ".join(capsys.readouterr().out.split())


_LIBGOMP_DETAIL = ("RuntimeError: Failed to load libllama.so from /x/libllama.so: "
                   "libgomp.so.1: cannot open shared object file: No such file or directory")


# --------------------------------------------------------------------------- #
#  Non-interactive: no prompt, straight to a clear, correctly-worded error    #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("chosen", ["vulkan", "cpu"])
def test_non_interactive_failure_names_the_library_and_never_retries(
        monkeypatch, tmp_path, chosen, capsys):
    _wire(monkeypatch, tmp_path, loads=[(False, _LIBGOMP_DETAIL)])
    monkeypatch.setattr(sl.sys.stdin, "isatty", lambda: False)

    with pytest.raises(bugreport.LocalmError) as exc:
        sl._provision_with_fallback(chosen, tmp_path, None, False, assume_yes=True)

    reason = exc.value.reason
    assert "libgomp.so.1" in reason
    assert "libgomp1" in reason
    assert "self-contained fallback" not in reason, "defect 2: wrong terminology for a primary pick"
    assert "--from" in reason, "defect 3(c): the escape hatch must be mentioned"
    assert f"--backend {chosen} --force" in reason


def test_non_interactive_unknown_cause_falls_back_to_raw_detail(monkeypatch, tmp_path):
    """When the failure is not a recognisable dlopen 'cannot open shared
    object file' shape, the raw detail still reaches the message - the
    library-naming helper is a refinement, not the only source of truth."""
    _wire(monkeypatch, tmp_path, loads=[(False, "no compute backends are loaded")])
    monkeypatch.setattr(sl.sys.stdin, "isatty", lambda: False)

    with pytest.raises(bugreport.LocalmError) as exc:
        sl._provision_with_fallback("cpu", tmp_path, None, False, assume_yes=True)

    assert "no compute backends are loaded" in exc.value.reason


# --------------------------------------------------------------------------- #
#  Interactive: retry offered, and it retries the SAME backend                #
# --------------------------------------------------------------------------- #

def test_interactive_accepted_retry_that_succeeds_returns_normally(
        monkeypatch, tmp_path, capsys):
    provisioned = _wire(monkeypatch, tmp_path,
                        loads=[(False, _LIBGOMP_DETAIL), (True, "")])
    monkeypatch.setattr(sl.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sl, "_flush_stdin", lambda: None)
    monkeypatch.setattr(sl.click, "confirm", lambda *a, **k: True)   # accept the retry

    backend, tag = sl._provision_with_fallback("vulkan", tmp_path, None, False,
                                                assume_yes=False)

    assert backend == "vulkan"
    assert provisioned == ["vulkan", "vulkan"], "retried the SAME backend, never a different one"
    out = _flat(capsys)
    assert "libgomp.so.1" in out, "the missing piece must be shown before the retry prompt"
    assert "OK - vulkan runtime loads" in out


def test_interactive_declined_retry_raises_with_the_same_message_shape(
        monkeypatch, tmp_path):
    provisioned = _wire(monkeypatch, tmp_path, loads=[(False, _LIBGOMP_DETAIL)])
    monkeypatch.setattr(sl.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sl, "_flush_stdin", lambda: None)
    monkeypatch.setattr(sl.click, "confirm", lambda *a, **k: False)   # decline

    with pytest.raises(bugreport.LocalmError) as exc:
        sl._provision_with_fallback("cpu", tmp_path, None, False, assume_yes=False)

    assert provisioned == ["cpu"], "declining must not attempt a second provision"
    assert "libgomp.so.1" in exc.value.reason


def test_interactive_retry_that_fails_again_then_declined(monkeypatch, tmp_path):
    """A retry that still does not load loops back to the same prompt with
    the NEW failure's detail, not the original one."""
    provisioned = _wire(monkeypatch, tmp_path,
                        loads=[(False, _LIBGOMP_DETAIL),
                              (False, "libvulkan.so.1: cannot open shared object file")])
    monkeypatch.setattr(sl.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sl, "_flush_stdin", lambda: None)
    answers = iter([True, False])   # retry once, then decline
    monkeypatch.setattr(sl.click, "confirm", lambda *a, **k: next(answers))

    with pytest.raises(bugreport.LocalmError) as exc:
        sl._provision_with_fallback("vulkan", tmp_path, None, False, assume_yes=False)

    assert provisioned == ["vulkan", "vulkan"]
    assert "libvulkan.so.1" in exc.value.reason, "the SECOND failure's cause must be reported"


def test_interactive_retry_that_raises_during_provisioning_is_reported_and_stops(
        monkeypatch, tmp_path, capsys):
    """If the retry attempt itself blows up (not merely fails to load), that
    is reported and treated as the end of the recovery, not swallowed. Only
    the SECOND call (the retry) fails - the first must succeed normally, or
    this would be testing the unrelated 'not provisioned' branch instead."""
    lib = sl._lib_name()
    calls: list = []

    def fake_provision(backend, target, sha256, with_cudart,
                       cuda_line=sl._CUDA_LINE, tag=None):
        calls.append(backend)
        if len(calls) == 1:
            (target / lib).write_bytes(b"x")
            return "bTEST"
        raise sl.ArtifactError("disk full")

    monkeypatch.setattr(sl, "_provision_backend", fake_provision)
    monkeypatch.setattr(sl, "_clear_target_or_refuse", lambda t: None)
    monkeypatch.setattr(sl, "_install_runtime_wheel", lambda d: True)
    monkeypatch.setattr(sl, "_bundle_missing_native_deps", lambda t: None)
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: (False, _LIBGOMP_DETAIL))
    monkeypatch.setattr(sl.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sl, "_flush_stdin", lambda: None)
    monkeypatch.setattr(sl.click, "confirm", lambda *a, **k: True)

    with pytest.raises(bugreport.LocalmError):
        sl._provision_with_fallback("cpu", tmp_path, None, False, assume_yes=False)

    assert calls == ["cpu", "cpu"], "the first call must have succeeded before the retry ran"
    assert "disk full" in _flat(capsys)


# --------------------------------------------------------------------------- #
#  Regression: the "not provisioned" (download/validation failure) branch     #
#  is a DIFFERENT code path and must be unaffected by the retry addition.     #
# --------------------------------------------------------------------------- #

def test_download_failure_still_exits_without_the_retry_prompt(monkeypatch, tmp_path):
    def fail_provision(backend, target, sha256, with_cudart,
                       cuda_line=sl._CUDA_LINE, tag=None):
        raise sl.ArtifactError("connection reset")
    monkeypatch.setattr(sl, "_provision_backend", fail_provision)
    monkeypatch.setattr(sl, "_clear_target_or_refuse", lambda t: None)
    monkeypatch.setattr(sl, "_bundle_missing_native_deps", lambda t: None)
    monkeypatch.setattr(sl.click, "confirm",
                        lambda *a, **k: pytest.fail("no retry prompt for a download failure"))

    with pytest.raises(SystemExit) as exc:
        sl._provision_with_fallback("vulkan", tmp_path, None, False, assume_yes=False)
    assert exc.value.code == 1
