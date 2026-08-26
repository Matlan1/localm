# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm/diagnostics.py - the callable core behind `localm doctor`'s five ACTIVE
probes and the GUI's diagnostics card.

The rendering half is covered by the test_doctor_*.py files, which drive `localm
doctor`'s printed output through the same core. This file covers what only the
core can express: the aggregate verdict, which finding a compact surface leads
with, and - the part with the sharpest failure mode - that a run which could NOT
be completed is never renderable as a clean bill of health.
"""

from __future__ import annotations

import json

import pytest

from localm import diagnostics as d


def _check(status, key="k", label="L", findings=()):
    return d.CheckResult(key=key, label=label, status=status,
                         summary="s", findings=tuple(findings))


# --------------------------------------------------------------------------- #
#  The aggregate                                                              #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("statuses,expected", [
    ([d.OK, d.OK], d.OK),
    ([d.OK, d.WARN], d.WARN),
    ([d.OK, d.WARN, d.FAIL], d.FAIL),
    ([d.WARN, d.FAIL], d.FAIL),
])
def test_the_verdict_is_the_worst_status_present(statuses, expected):
    assert d.verdict([_check(s) for s in statuses]) == expected


def test_a_skipped_check_neither_drags_a_clean_report_down_nor_lifts_a_broken_one():
    """SKIPPED sits BELOW ok in the severity order.

    An absent optional backend is the common case (no torch installed), so if
    "did not run" ranked as a warning every ordinary box would render a warning
    verdict and the word would stop meaning anything. It must not round UP
    either: a skipped check beside a failing one is still a failing report."""
    assert d.verdict([_check(d.OK), _check(d.SKIPPED)]) == d.OK
    assert d.verdict([_check(d.FAIL), _check(d.SKIPPED)]) == d.FAIL
    assert d.verdict([_check(d.WARN), _check(d.SKIPPED)]) == d.WARN


def test_a_report_of_only_skipped_checks_does_not_claim_ok():
    """Nothing was measured, so the verdict is SKIPPED rather than ok."""
    assert d.verdict([_check(d.SKIPPED), _check(d.SKIPPED)]) == d.SKIPPED


# --------------------------------------------------------------------------- #
#  Which finding a one-line surface leads with                                #
# --------------------------------------------------------------------------- #

def test_the_summary_leads_with_the_finding_carrying_the_checks_own_verdict():
    """The llama-library shape, and why _result does not just take findings[0]:
    that check reports a GREEN "found the library" line and then the BLAS kernel
    failures underneath it. A card showing one line per check would otherwise
    render the reassuring half of a failure."""
    res = d._result("llama_lib", "llama.cpp library", [
        d.Finding(d.OK, "llama.dll found in /somewhere"),
        d.Finding(d.FAIL, "rocblas is installed but its rocblas/ kernel "
                          "directory is empty - GPU matrix ops will crash"),
    ])
    assert res.status == d.FAIL
    assert res.summary.startswith("rocblas is installed")
    assert "found in" not in res.summary


def test_the_summary_carries_the_inline_note_a_terminal_would_print_dim():
    res = d._result("native_abi", "Native ABI",
                    [d.Finding(d.WARN, "native ABI not verified",
                               note="the ABI probe did not run")])
    assert res.summary == "native ABI not verified (the ABI probe did not run)"


def test_healthy_is_true_only_for_a_clean_pass():
    """`doctor` gates the ABI load-test on this. A truncated library reports
    WARN, and load-testing it anyway is exactly what must not happen."""
    assert _check(d.OK).healthy is True
    assert _check(d.WARN).healthy is False
    assert _check(d.FAIL).healthy is False
    assert _check(d.SKIPPED).healthy is False


# --------------------------------------------------------------------------- #
#  The library check, on a real filesystem                                    #
# --------------------------------------------------------------------------- #

def _bindir(tmp_path, size):
    lib = tmp_path / "llama.dll"
    lib.write_bytes(b"\0" * size)
    return lambda: tmp_path


def test_a_zero_byte_library_is_a_failure_not_a_pass(tmp_path):
    """It EXISTS, which is all a presence check would ask. It cannot load."""
    res = d.check_llama_lib(_bindir(tmp_path, 0))
    assert res.status == d.FAIL
    assert "0 bytes" in res.summary
    assert res.healthy is False


def test_a_truncated_library_is_reported_rather_than_passed(tmp_path):
    res = d.check_llama_lib(_bindir(tmp_path, 1024))
    assert res.status == d.WARN
    assert "suspiciously small" in res.summary


def test_a_plausible_library_with_no_vendor_blas_passes(tmp_path):
    res = d.check_llama_lib(_bindir(tmp_path, d.TINY_LIB_BYTES + 1))
    assert res.status == d.OK
    assert res.healthy is True


def test_a_rocblas_install_with_no_kernel_data_fails_despite_a_good_library(tmp_path):
    """The silent one: the library is present and the right size, so every
    presence check passes, and the first GEMM hard-crashes the native process.

    Note the assertion: `"rocblas" in res.summary` is USELESS here, because
    pytest derives tmp_path's basename from this test's own name - so the string
    appears in the healthy "llama.dll found in <tmp_path>" line too, and the test
    passes even when the summary picks the wrong finding. It asserts on wording
    only the FAILING finding can produce, and that the green line is not what
    leads."""
    (tmp_path / "llama.dll").write_bytes(b"\0" * (d.TINY_LIB_BYTES + 1))
    (tmp_path / "rocblas.dll").write_bytes(b"\0" * 4096)
    res = d.check_llama_lib(lambda: tmp_path)
    assert res.status == d.FAIL
    assert res.summary.startswith("rocblas is installed but its rocblas/ kernel")
    assert "GPU matrix ops will crash" in res.summary
    assert "found in" not in res.summary
    # The green line is still THERE - the terminal prints both - it is simply
    # not what a one-line surface leads with.
    assert [f.status for f in res.findings] == [d.OK, d.FAIL]


def test_a_missing_binary_dir_fails(tmp_path):
    res = d.check_llama_lib(lambda: None)
    assert res.status == d.FAIL
    assert "not found" in res.summary


# --------------------------------------------------------------------------- #
#  JSON round trip (the child -> parent boundary)                             #
# --------------------------------------------------------------------------- #

def test_a_report_survives_the_json_boundary_intact():
    """run_report_isolated parses exactly this, so a field that does not
    round-trip is a field the GUI silently never sees."""
    original = d.build_report([
        d._result("llama_lib", "llama.cpp library",
                  [d.Finding(d.OK, "llama.dll found in /x")]),
        d._result("native_abi", "Native ABI",
                  [d.Finding(d.FAIL, "native ABI MISMATCH",
                             hints=("field a", "field b"))]),
        d.CheckResult(key="hf_backend", label="HF (transformers) backend",
                      status=d.SKIPPED, summary="not installed", findings=()),
    ])
    back = d.DiagnosticsReport.from_dict(json.loads(json.dumps(original.as_dict())))
    assert back.verdict == d.FAIL == original.verdict
    assert [c.key for c in back.checks] == ["llama_lib", "native_abi", "hf_backend"]
    assert back.checks[1].findings[0].hints == ("field a", "field b")
    assert back.checks[2].status == d.SKIPPED
    assert back.checks[2].summary == "not installed"


# --------------------------------------------------------------------------- #
#  The isolated run: a run that could not happen must never look clean         #
# --------------------------------------------------------------------------- #

@pytest.fixture
def child(monkeypatch):
    """Swap the child program so the REAL Popen/parse path runs against a
    controlled process. Patching subprocess would test the mock instead."""
    def _set(code):
        monkeypatch.setattr(d, "_CHILD_CODE", code)
    return _set


def test_a_child_that_prints_nothing_is_an_error_not_an_empty_pass(child):
    """The failure this guards is specific: an empty `checks` list aggregates to
    a clean-looking report, so "we could not check" would render identically to
    "we checked and found nothing wrong"."""
    child("pass")
    rep = d.run_report_isolated(timeout=60)
    assert rep.verdict == d.ERROR
    assert rep.checks == ()
    assert "nothing usable" in rep.error


def test_a_child_that_crashes_reports_its_own_stderr(child):
    """Surface, do not silence: a diagnostics run that failed still reports why
    it failed."""
    child("import sys; sys.stderr.write('boom: no runtime\\n'); sys.exit(3)")
    rep = d.run_report_isolated(timeout=60)
    assert rep.verdict == d.ERROR
    assert "boom: no runtime" in rep.error
    assert "exit 3" in rep.error


def test_a_child_that_hangs_is_killed_and_reported_as_a_timeout(child):
    child("import time; time.sleep(120)")
    rep = d.run_report_isolated(timeout=1.0)
    assert rep.verdict == d.ERROR
    assert "did not finish within 1s" in rep.error


def test_unparseable_result_json_is_an_error_not_a_silent_empty_report(child):
    child(r"import sys; sys.stdout.write('LOCALM_DIAGNOSTICS:{not json}\n')")
    rep = d.run_report_isolated(timeout=60)
    assert rep.verdict == d.ERROR
    assert "could not be read" in rep.error


def test_progress_lines_are_delivered_as_they_arrive(child):
    """Progress that only shows up once the run has finished is not progress.
    The parent reads line by line for exactly this."""
    child(
        "import sys, json;"
        "w=lambda o: (sys.stdout.write("
        "'LOCALM_DIAGNOSTICS_PROGRESS:'+json.dumps(o)+'\\n'), sys.stdout.flush());"
        "w({'key':'llama_lib','label':'llama.cpp library','done':0,'total':2});"
        "w({'key':'venv','label':'Nested venv creation','done':1,'total':2});"
        "sys.stdout.write('LOCALM_DIAGNOSTICS:'+json.dumps("
        "{'verdict':'ok','checks':[]})+'\\n')"
    )
    seen = []
    rep = d.run_report_isolated(
        timeout=60, on_progress=lambda k, l, done, total: seen.append((k, done, total)))
    assert seen == [("llama_lib", 0, 2), ("venv", 1, 2)]
    assert rep.verdict == d.OK


def test_a_broken_progress_callback_never_costs_the_report(child):
    """The report is the deliverable; a surface that cannot render an update is
    not a reason to lose it."""
    child(
        "import sys, json;"
        "sys.stdout.write('LOCALM_DIAGNOSTICS_PROGRESS:'+json.dumps("
        "{'key':'x','label':'X','done':0,'total':1})+'\\n');"
        "sys.stdout.write('LOCALM_DIAGNOSTICS:'+json.dumps("
        "{'verdict':'ok','checks':[]})+'\\n')"
    )

    def _explode(*a):
        raise RuntimeError("the card went away")

    rep = d.run_report_isolated(timeout=60, on_progress=_explode)
    assert rep.verdict == d.OK
    assert rep.error == ""


def test_run_checks_reports_each_check_before_it_starts(monkeypatch):
    """`done` counts what has actually FINISHED and `phase` names what is
    running now, so a card never shows a number nothing has earned and never
    attributes the wait to the wrong check."""
    monkeypatch.setattr(d, "check_llama_lib",
                        lambda *a, **k: _check(d.OK, key="llama_lib"))
    monkeypatch.setattr(d, "check_native_abi",
                        lambda: _check(d.OK, key="native_abi"))
    monkeypatch.setattr(d, "check_worker_spawn",
                        lambda: _check(d.OK, key="worker_spawn"))
    monkeypatch.setattr(d, "check_venv_creation",
                        lambda: _check(d.OK, key="venv"))
    monkeypatch.setattr(d, "check_hf_backend",
                        lambda *a, **k: _check(d.SKIPPED, key="hf_backend"))
    seen = []
    d.run_checks(lambda key, label, done, total: seen.append((key, done, total)))
    assert seen == [("llama_lib", 0, 5), ("native_abi", 1, 5),
                    ("worker_spawn", 2, 5), ("venv", 3, 5), ("hf_backend", 4, 5)]


def test_the_abi_check_is_skipped_rather_than_omitted_when_the_library_is_broken(
        monkeypatch):
    """An omitted row reads as a check that passed. It gets a named SKIPPED
    result carrying the reason instead - and it must not be RUN, because
    load-testing a library already known to be truncated costs a 120s timeout
    to learn nothing."""
    monkeypatch.setattr(d, "check_llama_lib",
                        lambda *a, **k: _check(d.FAIL, key="llama_lib"))
    monkeypatch.setattr(d, "check_native_abi",
                        lambda: pytest.fail("the ABI probe must not run"))
    monkeypatch.setattr(d, "check_worker_spawn", lambda: _check(d.OK))
    monkeypatch.setattr(d, "check_venv_creation", lambda: _check(d.OK))
    monkeypatch.setattr(d, "check_hf_backend", lambda *a, **k: _check(d.SKIPPED))
    results = d.run_checks()
    abi = results[1]
    assert abi.key == "native_abi"
    assert abi.status == d.SKIPPED
    assert "no healthy llama.cpp library" in abi.summary


# --------------------------------------------------------------------------- #
#  Bound to the REAL child process                                            #
# --------------------------------------------------------------------------- #

def test_a_real_isolated_run_returns_all_five_checks_in_order():
    """The one test here with no fixture between it and the shipped path: a real
    child interpreter, the real checks, the real JSON boundary.

    Asserts STRUCTURE, not verdicts - a machine with no provisioned runtime is
    entitled to fail llama_lib, and pinning a verdict here would make this test
    a statement about the box rather than about the code. What it does insist on
    is that the run COMPLETED: an ERROR verdict means the isolation path itself
    is broken, which no fixture-driven test above can tell you."""
    rep = d.run_report_isolated(timeout=300)
    assert rep.verdict != d.ERROR, rep.error
    assert [c.key for c in rep.checks] == list(d.CHECK_KEYS)
    for c in rep.checks:
        assert c.label == d.CHECK_LABELS[c.key]
        assert c.status in (d.OK, d.WARN, d.FAIL, d.SKIPPED)
        assert c.summary, f"{c.key} produced no summary line"


def test_the_outer_deadline_fits_around_every_inner_bound():
    """The RELATION, not the literal.

    ``run_report_isolated``'s deadline has to be larger than everything the child
    can legitimately spend, or the outer timer becomes the first thing to fire and
    every slow-but-working box reports "the diagnostics run did not finish" - a
    fabricated failure on a healthy machine. Neither number is wrong alone; the
    relation is the thing, and it cannot be reviewed one number at a time."""
    inner = (d.PROBE_TIMEOUT_S + d.VENV_TIMEOUT_S + d.VENV_PIP_TIMEOUT_S
             + d.SPAWN_REPLY_TIMEOUT_S + 2 * d.SPAWN_JOIN_TIMEOUT_S)
    assert d.worst_case_run_seconds() > inner, (
        "the default deadline must leave room for the steps that have no timeout "
        "of their own (interpreter startup, importing torch and transformers)")
    assert d.worst_case_run_seconds() - inner >= d.UNBOUNDED_HEADROOM_S


def test_a_child_that_floods_its_output_still_yields_its_result(child):
    """stderr is merged into stdout rather than given its own pipe: reading one
    pipe to EOF while the other fills its buffer is the classic subprocess
    deadlock, and this child's dependencies do write to stderr. 400 KiB is well
    past a 64 KiB pipe buffer, so this hangs against a two-pipe implementation
    and returns against the merged one."""
    # chr(10) rather than a backslash escape: this string is Python source that
    # becomes a `-c` program, so an escape would have to survive two levels of
    # quoting.
    child(
        "import sys, json;"
        "nl = chr(10);"
        "sys.stderr.write(('x' * 200 + nl) * 2000);"
        "sys.stderr.flush();"
        "sys.stdout.write('LOCALM_DIAGNOSTICS:' + json.dumps("
        "{'verdict': 'ok', 'checks': []}) + nl)"
    )
    rep = d.run_report_isolated(timeout=90)
    # The injection took: an ERROR verdict here would mean the child never ran.
    assert rep.verdict == d.OK, rep.error
    assert rep.error == ""
