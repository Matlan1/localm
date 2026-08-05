# SPDX-License-Identifier: AGPL-3.0-or-later
"""doctor's ABI line: "not verified" must not invent WHY (ADR-0008).

The verdict was always honest. The reason was not: the line defaulted to
"runtime not loadable" whenever the probe result lacked a detail, and
``abi_report()`` populates detail on every path it can return from
("runtime not loadable: ...", "loader import failed: ..."). So the default was
reachable ONLY when ``_run_probe_subprocess`` returned None - the probe timed
out, crashed, or printed nothing - which is precisely the case where "runtime
not loadable" is least likely to be true.

The cost is a wrong remedy: a user told the runtime is not loadable re-runs
'setup-llama --force', which cannot fix a subprocess that timed out.

Same family as the smi fix (a check answering an adjacent question), one level
smaller: here the pass/fail verdict is right and only the explanation is
fabricated.
"""

from __future__ import annotations

import importlib

doctor_mod = importlib.import_module("localm.cli.doctor")


def _probe(monkeypatch, result):
    """Control the ABI probe subprocess. None models 'the probe never ran'.

    Patched on localm.cli.errors, NOT on the doctor module: _check_native_abi
    does `from .errors import _run_probe_subprocess` INSIDE the function, so the
    name is resolved from that module on every call and never exists as a doctor
    attribute at all (patching doctor_mod raises AttributeError, which is how
    this was caught)."""
    import localm.cli.errors as errors_mod
    monkeypatch.setattr(errors_mod, "_run_probe_subprocess",
                        lambda code, prefix: result)


def test_a_probe_that_never_ran_does_not_claim_the_runtime_is_unloadable(
        monkeypatch, capsys):
    """The defect. None means we do not know why, so we must not name a cause -
    least of all one whose remedy cannot help."""
    _probe(monkeypatch, None)

    doctor_mod._check_native_abi()
    out = capsys.readouterr().out

    assert "not verified" in out
    assert "runtime not loadable" not in out, out


def test_a_probe_that_never_ran_says_so(monkeypatch, capsys):
    """Surface, do not silence: the user should learn the probe itself failed,
    which is a different thing to investigate."""
    _probe(monkeypatch, None)

    doctor_mod._check_native_abi()
    out = capsys.readouterr().out

    assert "did not run" in out


def test_a_real_reason_from_the_probe_is_preserved(monkeypatch, capsys):
    """No regression: when the probe DID run and reported why, that reason is
    what the user needs and must survive verbatim."""
    _probe(monkeypatch, {"status": "unchecked",
                         "detail": "runtime not loadable: OSError(126)"})

    doctor_mod._check_native_abi()
    out = capsys.readouterr().out

    assert "runtime not loadable: OSError(126)" in out
    assert "did not run" not in out


def test_a_probe_that_ran_but_reported_no_reason_is_not_given_one(
        monkeypatch, capsys):
    """A ran-but-detail-less result is a third case. It still must not be
    handed a fabricated cause."""
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
