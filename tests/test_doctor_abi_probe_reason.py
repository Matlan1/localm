# SPDX-License-Identifier: AGPL-3.0-or-later
"""doctor's ABI line: 'not verified' must not invent WHY (ADR-0008)."""

from __future__ import annotations

import importlib

doctor_mod = importlib.import_module("localm.cli.doctor")


def _probe(monkeypatch, result):
    """Control the ABI probe subprocess."""
    import localm.diagnostics as diagnostics_mod
    monkeypatch.setattr(diagnostics_mod, "run_probe_subprocess",
                        lambda code, prefix, **kw: result)


def test_a_probe_that_never_ran_does_not_claim_the_runtime_is_unloadable(
        monkeypatch, capsys):
    """The defect."""
    _probe(monkeypatch, None)

    doctor_mod._check_native_abi()
    out = capsys.readouterr().out

    assert "not verified" in out
    assert "runtime not loadable" not in out, out


def test_a_probe_that_never_ran_says_so(monkeypatch, capsys):
    """Surface, do not silence: the user should learn the probe itself failed, which is a different thing to investigate."""
    _probe(monkeypatch, None)

    doctor_mod._check_native_abi()
    out = capsys.readouterr().out

    assert "did not run" in out


def test_a_real_reason_from_the_probe_is_preserved(monkeypatch, capsys):
    """No regression: when the probe DID run and reported why, that reason is what the user needs and must survive verbatim."""
    _probe(monkeypatch, {"status": "unchecked",
                         "detail": "runtime not loadable: OSError(126)"})

    doctor_mod._check_native_abi()
    out = capsys.readouterr().out

    assert "runtime not loadable: OSError(126)" in out
    assert "did not run" not in out


def test_a_probe_that_ran_but_reported_no_reason_is_not_given_one(
        monkeypatch, capsys):
    """A ran-but-detail-less result is a third case."""
    _probe(monkeypatch, {"status": "unchecked", "detail": ""})

    doctor_mod._check_native_abi()
    out = capsys.readouterr().out

    assert "not verified" in out
    assert "no reason reported" in out
    assert "runtime not loadable" not in out, out


def test_the_ok_and_mismatch_verdicts_are_untouched(monkeypatch, capsys):
    """This change touches only the unverified branch's reason text."""
    _probe(monkeypatch, {"status": "ok", "layout": "v2"})
    doctor_mod._check_native_abi()
    out = capsys.readouterr().out
    assert "struct layout matches this build" in out
    assert "v2" in out

    _probe(monkeypatch, {"status": "mismatch", "failures": ["f1"]})
    doctor_mod._check_native_abi()
    out = capsys.readouterr().out
    assert "native ABI MISMATCH" in out
