# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm doctor`: the exit code follows the failure lines, and the GPU section
says what it knows when the runtime is broken or the adapter probe did not run.
The worker spawn check names a test-only fault switch left in the environment."""

import ast
import importlib
import re
from pathlib import Path

import localm.cli as cli
from localm import diagnostics, hwdetect

doctor_mod = importlib.import_module("localm.cli.doctor")

_REPO = Path(__file__).resolve().parents[1]


def _quiet_checks(monkeypatch):
    """Every doctor check replaced by a silent one with a neutral return."""
    for name in dir(doctor_mod):
        if name.startswith("_check_"):
            monkeypatch.setattr(doctor_mod, name, lambda *a, **k: None)
    monkeypatch.setattr(doctor_mod, "_check_llama_lib", lambda *a, **k: True)
    monkeypatch.setattr(doctor_mod, "_check_gpu_driver", lambda *a, **k: False)
    monkeypatch.setattr(doctor_mod, "_check_vram_torch", lambda *a, **k: False)
    monkeypatch.setattr(doctor_mod, "_check_packages", lambda *a, **k: {})


def _one_check_result(monkeypatch, status):
    result = diagnostics.CheckResult(
        key="worker_spawn", label="Worker spawn", status=status, summary="",
        findings=(diagnostics.Finding(status, f"STATUS-LINE-{status}"),))
    monkeypatch.setattr(doctor_mod, "_check_worker_spawn",
                        lambda: doctor_mod._render(result))


def test_a_failure_line_makes_doctor_exit_1(cli_runner, monkeypatch):
    _quiet_checks(monkeypatch)
    _one_check_result(monkeypatch, diagnostics.FAIL)

    result = cli_runner.invoke(cli.doctor, [])

    assert "STATUS-LINE-fail" in result.output
    assert result.exit_code == 1, result.output


def test_warnings_alone_leave_the_exit_code_at_0(cli_runner, monkeypatch):
    _quiet_checks(monkeypatch)
    _one_check_result(monkeypatch, diagnostics.WARN)

    result = cli_runner.invoke(cli.doctor, [])

    assert "STATUS-LINE-warn" in result.output
    assert result.exit_code == 0, result.output


def test_a_second_run_does_not_inherit_the_first_runs_failures(cli_runner, monkeypatch):
    _quiet_checks(monkeypatch)
    _one_check_result(monkeypatch, diagnostics.FAIL)
    assert cli_runner.invoke(cli.doctor, []).exit_code == 1

    _one_check_result(monkeypatch, diagnostics.OK)
    assert cli_runner.invoke(cli.doctor, []).exit_code == 0


def test_every_failure_symbol_goes_through_line():
    """A console.print that can show the red cross bypasses the exit code."""
    tree = ast.parse(Path(doctor_mod.__file__).read_text(encoding="utf-8"))
    risky = {"_FAIL_SYM", "_SYM_FOR", "sym"}
    offenders = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or fn.name == "_line":
            continue
        for call in ast.walk(fn):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "print"
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "console"):
                continue
            names = {n.id for a in call.args for n in ast.walk(a) if isinstance(n, ast.Name)}
            if names & risky:
                offenders.append(f"{fn.name}:{call.lineno}")
    assert offenders == []


def test_a_broken_library_with_a_visible_card_still_gets_a_gpu_verdict(capsys):
    doctor_mod._check_gpu_verdict(lambda: None, False, True)

    out = " ".join(capsys.readouterr().out.split())
    assert "GPU: not used for GGUF models until the llama.cpp library above is fixed" in out
    assert "the card listed above is there" in out


def test_no_library_and_no_card_keeps_the_cpu_only_verdict(capsys):
    doctor_mod._check_gpu_verdict(lambda: None, False, False)

    out = capsys.readouterr().out
    assert "CPU mode only" in out
    assert "GPU: not used for GGUF models" not in out


def test_an_adapter_probe_that_did_not_run_is_reported(monkeypatch, capsys):
    monkeypatch.setattr(hwdetect, "detect", lambda: hwdetect.Detection(
        probe_ok=False, probe_error="the display-adapter enumeration did not run"))

    doctor_mod._check_adapter_probe()

    out = " ".join(capsys.readouterr().out.split())
    assert "graphics adapters could not be listed" in out
    assert "the display-adapter enumeration did not run" in out


def test_an_adapter_probe_that_answered_prints_nothing(monkeypatch, capsys):
    monkeypatch.setattr(hwdetect, "detect", lambda: hwdetect.Detection(probe_ok=True))
    doctor_mod._check_adapter_probe()
    monkeypatch.setattr(hwdetect, "detect", lambda: hwdetect.Detection(
        vendors=["nvidia"], probe_ok=False, probe_error="x"))
    doctor_mod._check_adapter_probe()

    assert capsys.readouterr().out == ""


def test_a_set_fault_switch_turns_the_worker_spawn_check_into_a_warning(monkeypatch):
    monkeypatch.setenv("LOCALM_GGUF_FAULT_FOR_TEST", "crash")

    result = diagnostics.check_worker_spawn()

    assert result.status == diagnostics.WARN, result
    texts = [f.text for f in result.findings]
    assert any(t.startswith("LOCALM_GGUF_FAULT_FOR_TEST is set") for t in texts), texts


def test_fault_switch_findings_ignore_unset_and_empty_variables():
    assert diagnostics.fault_switch_findings({}) == []
    assert diagnostics.fault_switch_findings({"LOCALM_HF_FAULT_FOR_TEST": ""}) == []
    found = diagnostics.fault_switch_findings({"LOCALM_HF_FAULT_FOR_TEST": "hang"})
    assert [f.status for f in found] == [diagnostics.WARN]


def test_every_fault_switch_is_listed():
    """A new test-only switch in product code must reach the doctor check."""
    pattern = re.compile(r"[\"'](LOCALM_[A-Z0-9_]+_FOR_TEST)[\"']")
    seen = set()
    for path in (_REPO / "localm").rglob("*.py"):
        if path.name == "diagnostics.py" and path.parent.name == "localm":
            continue
        seen.update(pattern.findall(path.read_text(encoding="utf-8", errors="replace")))
    assert seen, "the scan found no switch at all"
    assert sorted(seen) == sorted(diagnostics.FAULT_SWITCH_VARIABLES)
