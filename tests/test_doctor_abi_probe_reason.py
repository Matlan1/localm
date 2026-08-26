# SPDX-License-Identifier: AGPL-3.0-or-later
"""doctor's ABI line: "not verified" must not invent WHY.

``abi_report()`` populates detail on every path it can return from ("runtime not
loadable: ...", "loader import failed: ..."), so the only result with no detail
is ``_run_probe_subprocess`` returning None - the probe timed out, crashed, or
printed nothing. That case must not be reported as "runtime not loadable", whose
remedy ('setup-llama --force') cannot fix a subprocess that timed out.
"""

from __future__ import annotations

import importlib

doctor_mod = importlib.import_module("localm.cli.doctor")


def _probe(monkeypatch, result):
    """Control the ABI probe subprocess. None models 'the probe never ran'.

    Patched on localm.diagnostics, NOT on the doctor module: the probe lives
    there (doctor's _check_native_abi is a renderer over
    diagnostics.check_native_abi), and check_native_abi resolves
    run_probe_subprocess from its own module globals on every call, so the name
    does not exist as a doctor attribute at all - patching doctor_mod raises
    AttributeError.

    The assertions below drive doctor's RENDERED OUTPUT rather than the core's
    return value, since the wrong reason guarded against here is one a user
    reads."""
    import localm.diagnostics as diagnostics_mod
    monkeypatch.setattr(diagnostics_mod, "run_probe_subprocess",
                        lambda code, prefix, **kw: result)


def test_a_probe_that_never_ran_does_not_claim_the_runtime_is_unloadable(
        monkeypatch, capsys):
    """None means the cause is unknown, so no cause is named."""
    _probe(monkeypatch, None)

    doctor_mod._check_native_abi()
    out = capsys.readouterr().out

    assert "not verified" in out
    assert "runtime not loadable" not in out, out


def test_a_probe_that_never_ran_says_so(monkeypatch, capsys):
    """The user is told the probe itself failed, which is a different thing to
    investigate."""
    _probe(monkeypatch, None)

    doctor_mod._check_native_abi()
    out = capsys.readouterr().out

    assert "did not run" in out


def test_a_real_reason_from_the_probe_is_preserved(monkeypatch, capsys):
    """When the probe DID run and reported why, that reason survives verbatim."""
    _probe(monkeypatch, {"status": "unchecked",
                         "detail": "runtime not loadable: OSError(126)"})

    doctor_mod._check_native_abi()
    out = capsys.readouterr().out

    assert "runtime not loadable: OSError(126)" in out
    assert "did not run" not in out


def test_a_probe_that_ran_but_reported_no_reason_is_not_given_one(
        monkeypatch, capsys):
    """A ran-but-detail-less result is a third case, and is not handed a
    fabricated cause either."""
    _probe(monkeypatch, {"status": "unchecked", "detail": ""})

    doctor_mod._check_native_abi()
    out = capsys.readouterr().out

    assert "not verified" in out
    assert "no reason reported" in out
    assert "runtime not loadable" not in out, out


def test_the_ok_and_mismatch_verdicts_are_untouched(monkeypatch, capsys):
    """The ok and mismatch verdicts render their own text unchanged."""
    _probe(monkeypatch, {"status": "ok", "layout": "v2"})
    doctor_mod._check_native_abi()
    out = capsys.readouterr().out
    assert "struct layout matches this build" in out
    assert "v2" in out

    _probe(monkeypatch, {"status": "mismatch", "failures": ["f1"]})
    doctor_mod._check_native_abi()
    out = capsys.readouterr().out
    assert "native ABI MISMATCH" in out
