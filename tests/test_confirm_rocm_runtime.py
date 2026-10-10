# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/confirm_rocm_runtime.py: does a lemonade-sdk ROCm build work with localm.

The script's job is to turn what its stages measured into a verdict nobody can
round up. These tests cover that on real data shapes:

  * the verdict logic: PASS only when every required check PASSED, FAIL when one
    FAILED, INCONCLUSIVE for anything that could not be measured, a receipt on
    every path including an uncaught exception, and a receipt the bump script
    accepts;
  * every evaluation separates a bad build (FAIL) from a missing measurement
    (SKIP), and a "GPU" run that was really the CPU cannot pass;
  * isolation: the child environment, the path checks, the process plumbing and
    the candidate-pins preload are exercised for real in child processes.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

confirm = importlib.import_module("confirm_rocm_runtime")
bump = importlib.import_module("bump_rocm_pin")
fake_api = importlib.import_module("tests.test_bump_rocm_pin")

PASS, FAIL, SKIP = confirm.PASS, confirm.FAIL, confirm.SKIP
CPU_COMMIT = "71ad0590f4808b6202f9213d166913858c73b1bc"
CAND = bump.Candidate(
    tag="b1342", lemonade_commit="71ad0", rocm_version="10.2.0a20261008",
    published_at=bump.parse_date("2026-10-08T20:46:47Z"),
    rocm_assets={"llama-b1342-windows-rocm-gfx103X-x64.zip": "2" * 64,
                 "llama-b1342-ubuntu-rocm-gfx103X-x64.zip": "3" * 64},
    cpu_tag="b11513", cpu_commit=CPU_COMMIT, cpu_asset="llama-b11513-bin-win-cpu-x64.zip",
    cpu_sha256="4" * 64)


# --------------------------------------------------------------------------- #
#  Verdict and receipt                                                         #
# --------------------------------------------------------------------------- #

def _all(receipt, status=PASS):
    for n in confirm.CHECK_NAMES:
        confirm.set_check(receipt, n, status, "x")
    return receipt


def test_a_fresh_receipt_is_inconclusive_with_every_check_required_and_unrun():
    r = confirm.new_receipt("b1", False)
    assert confirm.verdict_of(r)[0] == "INCONCLUSIVE"
    assert set(r["checks"]) == set(bump.REQUIRED_CHECKS)
    assert all(c["required"] and c["status"] == SKIP for c in r["checks"].values())


def test_pass_needs_every_required_check_to_pass():
    r = _all(confirm.new_receipt("b1", False))
    assert confirm.verdict_of(r)[0] == "PASS"
    assert confirm.finalize(r) == 0 and r["not_measured"] == []
    confirm.set_check(r, "gpu_generate", SKIP, "no GPU")
    assert confirm.verdict_of(r) == ("INCONCLUSIVE", "gpu_generate: no GPU")
    assert confirm.finalize(r) == 2 and r["not_measured"] == ["gpu_generate"]


def test_fail_dominates_skip_and_names_the_failed_check():
    r = confirm.new_receipt("b1", False)
    confirm.set_check(r, "abi", FAIL, "layout moved")
    verdict, why = confirm.verdict_of(r)
    assert verdict == "FAIL" and why == "abi: layout moved"
    assert confirm.finalize(r) == 1


def test_an_optional_check_never_changes_the_verdict():
    r = _all(confirm.new_receipt("b1", False))
    confirm.set_check(r, "extra", FAIL, "ignored", required=False)
    assert confirm.verdict_of(r)[0] == "PASS"


def test_write_receipt_is_atomic_and_creates_parents(tmp_path):
    p = tmp_path / "a" / "b" / "r.json"
    confirm.write_receipt(p, {"k": 1})
    assert json.loads(p.read_text(encoding="utf-8")) == {"k": 1}
    assert [f.name for f in p.parent.iterdir()] == ["r.json"]


# --------------------------------------------------------------------------- #
#  Evaluations                                                                 #
# --------------------------------------------------------------------------- #

HW_OK = {"vendors": ["amd"], "gpu_names": "amd radeon rx 6900 xt", "amd_gfx_family": "gfx103x",
         "probe_ok": True}


def test_hardware_passes_only_on_an_amd_gfx103x_gpu():
    assert confirm.evaluate_hardware(HW_OK)[0] == PASS
    for report in ({**HW_OK, "vendors": ["nvidia"]}, {**HW_OK, "vendors": []},
                   {**HW_OK, "amd_gfx_family": "gfx110x"}, {**HW_OK, "amd_gfx_family": ""},
                   {**HW_OK, "probe_ok": False, "probe_error": "wmi"}):
        status, detail = confirm.evaluate_hardware(report)
        assert status == SKIP and detail


@pytest.mark.parametrize("log, kind", [
    ("Download failed: Remote end closed connection without response", "network"),
    ("download stalled (no data for 60s, after 12 MB)", "network"),
    ("urllib.error.URLError: <urlopen error [Errno 11001] getaddrinfo failed>", "network"),
    ("HTTP Error 503: Service Unavailable", "network"),
    ("Refusing to install: sha256 mismatch (expected a, got b)", "build"),
    ("the archive did not contain llama.dll", "build"),
    ("", "build"),
])
def test_install_failures_are_split_into_transport_and_build(log, kind):
    assert confirm.classify_install_failure(log) == kind


URL = "https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/b1342/llama-b1342-windows-rocm-gfx103X-x64.zip"


def test_resolve_compares_the_installers_url_and_digest_with_the_release():
    ok = {"url": URL, "sha256": "2" * 64}
    assert confirm.evaluate_resolve(ok, True, CAND)[0] == PASS
    status, detail = confirm.evaluate_resolve(ok, False, CAND)
    assert status == PASS and "pinned fallback" in detail, "a rate limited listing still proves the fallback pair"
    assert confirm.evaluate_resolve({**ok, "sha256": "9" * 64}, False, CAND)[0] == FAIL
    assert confirm.evaluate_resolve({"url": None, "sha256": None}, False, CAND)[0] == FAIL
    assert confirm.evaluate_resolve({**ok, "sha256": None}, True, CAND)[0] == FAIL
    assert confirm.evaluate_resolve({**ok, "sha256": "9" * 64}, True, CAND)[0] == FAIL
    assert confirm.evaluate_resolve({**ok, "url": URL.replace("b1342", "b1307")}, True, CAND)[0] == FAIL


INSTALL_OK = {"exit_code": None, "exception": None, "marker": {"backend": "amd-rocm", "build": "b1342-cpu-b11513"},
              "overlay": {"tag": "b11513", "variant": "ggml-cpu-haswell.dll"}, "llama_dll": True, "log_tail": ""}


def test_install_passes_only_for_the_exact_build_with_the_cpu_overlay():
    assert confirm.evaluate_install(INSTALL_OK, CAND)[0] == PASS
    assert confirm.evaluate_install({**INSTALL_OK, "exit_code": 0}, CAND)[0] == PASS


@pytest.mark.parametrize("change", [
    {"marker": {"backend": "vulkan", "build": "b11513"}},
    {"marker": {"backend": "cpu", "build": None}},
    {"marker": {"backend": "amd-rocm", "build": "b1342"}},
    {"marker": {"backend": "amd-rocm", "build": "b1307-cpu-b10270"}},
    {"overlay": None},
    {"overlay": {"tag": "b10270", "variant": "ggml-cpu-haswell.dll"}},
    {"overlay": {"tag": "b11513", "variant": ""}},
    {"llama_dll": False},
    {"exit_code": 1, "log_tail": "Provisioning amd-rocm failed: the archive did not contain llama.dll"},
])
def test_install_fails_when_the_runtime_is_not_the_requested_build(change):
    status, detail = confirm.evaluate_install({**INSTALL_OK, **change}, CAND)
    assert status == FAIL and detail


def test_a_fallback_to_vulkan_is_a_failure_of_the_build_not_a_pass():
    out = {**INSTALL_OK, "exit_code": None, "marker": {"backend": "vulkan", "build": "b11513"}}
    status, detail = confirm.evaluate_install(out, CAND)
    assert status == FAIL and "fell back" in detail


def test_an_install_cut_off_by_the_network_is_unmeasured_not_failed():
    out = {**INSTALL_OK, "exit_code": 1, "log_tail": "Download failed: Remote end closed connection"}
    assert confirm.evaluate_install(out, CAND)[0] == SKIP
    out = {**INSTALL_OK, "exception": "ConnectionError: timed out", "log_tail": ""}
    assert confirm.evaluate_install(out, CAND)[0] == SKIP
    out = {**INSTALL_OK, "exception": "KeyError: 'x'", "log_tail": ""}
    assert confirm.evaluate_install(out, CAND)[0] == FAIL


@pytest.mark.parametrize("text, ok", [
    ("The color of a clear daytime sky is blue.", True),
    ("Blue.", False),
    ("", False),
    ("#### $$$$ ???? ^^^^ @@@@ %%%%", False),
    ("the the the the the the the the the", False),
    ("\u00e9\u00e9\u00e9 \u4e2d\u6587\u4e2d\u6587 \u0436\u0436", False),
    ("It is blue because air scatters blue light more than red light.", True),
])
def test_coherent_text_is_prose_not_empty_or_degenerate_output(text, ok):
    assert confirm.coherent(text) is ok


def test_common_prefix():
    assert confirm.common_prefix("blue sky", "blue sea") == 6
    assert confirm.common_prefix("", "x") == 0 and confirm.common_prefix("ab", "ab") == 2


def _lib(tmp_path):
    d = tmp_path / "runtime-pkg" / "localm_llama_runtime" / "lib"
    d.mkdir(parents=True)
    return d


def _mods(d, extra=(), skip=()):
    names = ["llama.dll", "ggml-base.dll", "ggml-hip.dll", "ggml-cpu.dll", "amdhip64_7.dll"]
    mods = [str(d / n) for n in names if n not in skip]
    return mods + [str(d.parents[2] / "system32" / "kernel32.dll"), *extra]


def test_worker_libraries_must_come_from_the_runtime_under_test(tmp_path):
    d = _lib(tmp_path)
    assert confirm.worker_problems(_mods(d), d) == []
    assert "did not map llama.dll" in confirm.worker_problems(_mods(d, skip=("llama.dll",)), d)[0]
    foreign = str(tmp_path / "shared" / "ggml-base.dll")
    problems = confirm.worker_problems(_mods(d, extra=(foreign,)), d)
    assert len(problems) == 1 and "outside" in problems[0]
    other = str(tmp_path / "shared" / "llama.dll")
    assert confirm.worker_problems(_mods(d, skip=("llama.dll",), extra=(other,)), d)


def probe_ok(d, **over):
    p = {"gpu_type": 1, "load": {"status": "ok"}, "runtime_dir": str(d), "runtime_dir_ok": True,
         "abi": {"status": "ok", "layout": "v3", "context_layout": "ctx_v3",
                 "failures": [], "diagnostics": [], "detail": "ok"},
         "identity": {"ggml_commit": "71ad0590", "ggml_version": "0.26.0"},
         "system_info": ("ROCm : NO_VMM = 1 | FA_QUANTS = q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16 | CPU : SSE3 = 1 "
                         "| SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | LLAMAFILE = 1 "
                         "| OPENMP = 1 | REPACK = 1 | "),
         "devices": [{"index": 0, "name": "ROCm0", "description": "AMD Radeon RX 6900 XT",
                      "type": 1, "free": 17015111680, "total": 17163091968}],
         "gpu": {"requested_gpu_layers": 99, "effective_gpu_layers": 99, "modules": _mods(d),
                 "worker_vram": 420 << 20, "text": "The color of a clear daytime sky is blue."},
         "gpu_ref": {"requested_gpu_layers": 0, "effective_gpu_layers": 0, "modules": _mods(d),
                     "worker_vram": 150 << 20, "text": "The color of a clear daytime sky is blue."}}
    p.update(over)
    return p


MODEL_BYTES = 105454432


def test_abi_evaluation(tmp_path):
    d = _lib(tmp_path)
    assert confirm.evaluate_abi(probe_ok(d))[0] == PASS
    assert confirm.evaluate_abi(probe_ok(d, load={"status": "abi_mismatch", "detail": "x"}))[0] == FAIL
    assert confirm.evaluate_abi(probe_ok(d, load={"status": "load_error", "detail": "dll"}))[0] == FAIL
    assert confirm.evaluate_abi(probe_ok(d, abi={"status": "mismatch", "failures": ["a"]}))[0] == FAIL
    assert confirm.evaluate_abi(probe_ok(d, abi={"status": "skipped", "detail": "env"}))[0] == SKIP
    assert confirm.evaluate_abi(probe_ok(d, abi={"status": "unchecked", "detail": "err"}))[0] == SKIP
    assert confirm.evaluate_abi({})[0] == SKIP


def test_loading_a_different_runtime_than_the_one_under_test_measures_nothing(tmp_path):
    d = _lib(tmp_path)
    status, detail = confirm.evaluate_abi(probe_ok(d, runtime_dir_ok=False, runtime_dir="/shared/lib"))
    assert status == SKIP and "not the runtime under test" in detail


def test_identity_ties_the_loaded_binary_to_the_release_and_the_cpu_tag(tmp_path):
    d = _lib(tmp_path)
    assert confirm.evaluate_identity(probe_ok(d), CAND)[0] == PASS
    wrong = probe_ok(d, identity={"ggml_commit": "07132750", "ggml_version": "0.18.1"})
    assert confirm.evaluate_identity(wrong, CAND)[0] == FAIL
    assert confirm.evaluate_identity(probe_ok(d, identity={}), CAND)[0] == SKIP
    assert confirm.evaluate_identity(probe_ok(d, identity={"ggml_commit": "71ad"}), CAND)[0] == SKIP
    notes_disagree = bump.Candidate(**{**CAND.__dict__, "lemonade_commit": "99999"})
    assert confirm.evaluate_identity(probe_ok(d), notes_disagree)[0] == FAIL


def test_gpu_device_requires_a_registered_rocm_gpu(tmp_path):
    d = _lib(tmp_path)
    assert confirm.evaluate_gpu_device(probe_ok(d))[0] == PASS
    cpu_only = [{"name": "CPU", "description": "x", "type": 0}]
    assert confirm.evaluate_gpu_device(probe_ok(d, devices=[]))[0] == FAIL
    vulkan = [{"name": "Vulkan0", "description": "AMD", "type": 1}]
    assert confirm.evaluate_gpu_device(probe_ok(d, devices=vulkan))[0] == FAIL
    assert confirm.evaluate_gpu_device(probe_ok(d, devices=cpu_only))[0] == FAIL
    assert confirm.evaluate_gpu_device(probe_ok(d, devices=None))[0] == SKIP


def test_model_evaluation():
    ok = {"architecture": "llama", "refusal": None, "bytes": MODEL_BYTES, "source": "from the cache"}
    assert confirm.evaluate_model(ok)[0] == PASS
    assert confirm.evaluate_model({"error": "offline"}) == (SKIP, "offline")
    assert confirm.evaluate_model({**ok, "architecture": "bert", "refusal": "embedding model"})[0] == FAIL
    assert confirm.evaluate_model({**ok, "architecture": None})[0] == FAIL


def _gpu(d, **over):
    g = probe_ok(d)["gpu"]
    g.update(over)
    return g


def _ref(d, **over):
    g = probe_ok(d)["gpu_ref"]
    g.update(over)
    return g


def _evaluate(d, gpu=None, ref=None):
    return confirm.evaluate_gpu_generate(_gpu(d) if gpu is None else gpu, _ref(d) if ref is None else ref,
                                         MODEL_BYTES, d)


def test_gpu_generation_passes_with_offload_residency_and_the_rocm_backend_mapped(tmp_path):
    d = _lib(tmp_path)
    status, detail = _evaluate(d)
    assert status == PASS and "420 MiB" in detail and "+270 MiB" in detail


def test_a_run_that_only_looks_like_it_used_the_gpu_cannot_pass(tmp_path):
    d = _lib(tmp_path)
    same = _gpu(d, worker_vram=_ref(d)["worker_vram"])
    assert _evaluate(d, gpu=same)[0] == FAIL, (
        "layers requested and text generated, but the worker holds no more GPU memory than with no offload")
    barely = _gpu(d, worker_vram=_ref(d)["worker_vram"] + MODEL_BYTES // 3)
    assert _evaluate(d, gpu=barely)[0] == FAIL
    enough = _gpu(d, worker_vram=_ref(d)["worker_vram"] + MODEL_BYTES // 2)
    assert _evaluate(d, gpu=enough)[0] == PASS
    no_hip = {"modules": _mods(d, skip=("ggml-hip.dll",))}
    assert _evaluate(d, gpu=_gpu(d, **no_hip))[0] == SKIP
    assert _evaluate(d, gpu=_gpu(d, effective_gpu_layers=0))[0] == FAIL


def test_gpu_residency_is_unproven_when_the_per_process_counter_cannot_be_read(tmp_path):
    d = _lib(tmp_path)
    for gpu, ref in ((_gpu(d, worker_vram=None), None), (None, _ref(d, worker_vram=None)),
                     (_gpu(d, worker_vram=None), _ref(d, worker_vram=None)), (None, {})):
        status, detail = _evaluate(d, gpu=gpu, ref=ref)
        assert status == SKIP and "counter" in detail


def test_gpu_generation_failures_and_skips(tmp_path):
    d = _lib(tmp_path)
    assert confirm.evaluate_gpu_generate({}, {}, MODEL_BYTES, d)[0] == SKIP
    assert confirm.evaluate_gpu_generate({"skip": "busy"}, {}, MODEL_BYTES, d) == (SKIP, "busy")
    assert _evaluate(d, gpu=_gpu(d, error="RuntimeError: load"))[0] == FAIL
    assert _evaluate(d, gpu=_gpu(d, text="@@@@ #### ????"))[0] == FAIL
    assert _evaluate(d, gpu=_gpu(d, modules=[]))[0] == SKIP
    foreign = {"modules": _mods(d, extra=(str(tmp_path / "other" / "ggml-hip.dll"),))}
    assert _evaluate(d, gpu=_gpu(d, **foreign))[0] == SKIP


HEADER = ('"(PDH-CSV 4.0)","\\\\HOST\\GPU Process Memory(pid_10032_luid_0x00000000_0x0000F397_phys_0)\\Dedicated Usage",'
          '"\\\\HOST\\GPU Process Memory(pid_4242_luid_0x00000000_0x0000F397_phys_0)\\Dedicated Usage",'
          '"\\\\HOST\\GPU Process Memory(pid_4242_luid_0x00000000_0x0000F398_phys_0)\\Dedicated Usage",'
          '"\\\\HOST\\GPU Process Memory(pid_42421_luid_0x00000000_0x0000F397_phys_0)\\Dedicated Usage"')


def test_gpu_process_memory_counter_output_is_read_per_pid():
    rows = HEADER + '\n"10/10/2026 15:56:42.743","53248.000000","440401920.000000","1048576.000000","7.000000"\n'
    assert confirm.parse_gpu_process_memory(rows, 4242) == 440401920 + 1048576
    assert confirm.parse_gpu_process_memory(rows, 10032) == 53248
    assert confirm.parse_gpu_process_memory(rows, 42421) == 7, "pid 4242 is not pid 42421"
    assert confirm.parse_gpu_process_memory(rows, 99) == 0, "a process with no instance holds no GPU memory"


@pytest.mark.parametrize("text", ["", "Error: No valid counters.", '"(PDH-CSV 4.0)","\\\\HOST\\Memory\\Available"\n"t","1"',
                                  HEADER])
def test_output_that_is_not_the_gpu_counter_set_is_unreadable_not_zero(text):
    assert confirm.parse_gpu_process_memory(text, 4242) is None


def test_a_garbled_counter_value_is_unreadable_not_zero():
    assert confirm.parse_gpu_process_memory(HEADER + '\n"t","1","oops","2","3"\n', 4242) is None


def test_the_counter_is_unavailable_off_windows(monkeypatch):
    monkeypatch.setattr(confirm.sys, "platform", "linux")
    assert confirm.dedicated_gpu_bytes(os.getpid()) is None


@pytest.mark.skipif(sys.platform != "win32", reason="the GPU Process Memory counter set is Windows-only")
def test_the_real_counter_reports_no_gpu_memory_for_this_process():
    assert confirm.dedicated_gpu_bytes(os.getpid()) in (None, 0)


def test_cpu_backend_needs_the_pinned_overlay_and_a_simd_capable_cpu(tmp_path):
    d = _lib(tmp_path)
    assert confirm.evaluate_cpu_backend(INSTALL_OK, CAND, probe_ok(d))[0] == PASS
    assert confirm.evaluate_cpu_backend({**INSTALL_OK, "overlay": None}, CAND, probe_ok(d))[0] == FAIL
    stale = {**INSTALL_OK, "overlay": {"tag": "b10270", "variant": "ggml-cpu-haswell.dll"}}
    assert confirm.evaluate_cpu_backend(stale, CAND, probe_ok(d))[0] == FAIL
    base = {**INSTALL_OK, "overlay": {"tag": "b11513", "variant": "ggml-cpu-x64.dll"}}
    assert confirm.evaluate_cpu_backend(base, CAND, probe_ok(d))[0] == SKIP
    scalar = probe_ok(d, system_info="CPU : SSE3 = 1 | OPENMP = 1 |")
    assert confirm.evaluate_cpu_backend(INSTALL_OK, CAND, scalar)[0] == SKIP


def _cpu(d, **over):
    c = {"requested_gpu_layers": 0, "effective_gpu_layers": 0, "gpu_visible": False, "devices": [("CPU", 0)],
         "worker_vram": 0,
         "modules": _mods(d, skip=("ggml-hip.dll", "amdhip64_7.dll")),
         "text": "The color of a clear daytime sky is blue."}
    c.update(over)
    return c


def test_cpu_generation_must_be_cpu_only(tmp_path):
    d = _lib(tmp_path)
    status, detail = confirm.evaluate_cpu_generate(_cpu(d), d, "The color of a clear sky is blue.")
    assert status == PASS and "agrees with the GPU output" in detail
    assert confirm.evaluate_cpu_generate(_cpu(d, gpu_visible=True), d)[0] == SKIP
    assert confirm.evaluate_cpu_generate(_cpu(d, worker_vram=300 << 20), d)[0] == SKIP
    assert confirm.evaluate_cpu_generate(_cpu(d, worker_vram=None), d)[0] == PASS
    assert confirm.evaluate_cpu_generate(_cpu(d, effective_gpu_layers=99), d)[0] == FAIL
    assert confirm.evaluate_cpu_generate(_cpu(d, error="RuntimeError: x"), d)[0] == FAIL
    assert confirm.evaluate_cpu_generate(_cpu(d, text="???"), d)[0] == FAIL
    assert confirm.evaluate_cpu_generate(_cpu(d, modules=[]), d)[0] == SKIP
    assert confirm.evaluate_cpu_generate(_cpu(d, modules=_mods(d, skip=("ggml-cpu.dll",))), d)[0] == SKIP
    assert confirm.evaluate_cpu_generate({}, d)[0] == SKIP
    assert confirm.evaluate_cpu_generate({"skip": "x"}, d) == (SKIP, "x")


# --------------------------------------------------------------------------- #
#  Isolation                                                                   #
# --------------------------------------------------------------------------- #

def _report(work, repo):
    return {"home_dir": str(work / "home"), "runtime_lib": str(work / "runtime-pkg" / "localm_llama_runtime" / "lib"),
            "tmp_dir": str(work / "tmp"), "localm_file": str(repo / "localm" / "__init__.py"),
            "llama_cpp_lib_env": ""}


def test_isolation_passes_only_when_every_path_is_under_the_work_directory(tmp_path):
    work, repo = tmp_path / "work", tmp_path / "repo"
    assert confirm.isolation_problems(_report(work, repo), work, repo) == []
    for key in ("home_dir", "runtime_lib", "tmp_dir"):
        bad = {**_report(work, repo), key: str(tmp_path / "elsewhere")}
        problems = confirm.isolation_problems(bad, work, repo)
        assert len(problems) == 1 and key in problems[0]
        missing = {k: v for k, v in _report(work, repo).items() if k != key}
        assert "not reported" in confirm.isolation_problems(missing, work, repo)[0]
    foreign = {**_report(work, repo), "localm_file": str(tmp_path / "main" / "localm" / "__init__.py")}
    assert "not from this checkout" in confirm.isolation_problems(foreign, work, repo)[0]
    leaked = {**_report(work, repo), "llama_cpp_lib_env": "/shared/lib/llama.dll"}
    assert "LLAMA_CPP_LIB" in confirm.isolation_problems(leaked, work, repo)[0]


def test_a_sibling_directory_with_the_same_prefix_is_not_under_the_work_directory(tmp_path):
    assert confirm.under(tmp_path / "work" / "a", tmp_path / "work")
    assert not confirm.under(tmp_path / "work-other" / "a", tmp_path / "work")
    assert not confirm.under(tmp_path / "work" / ".." / "x", tmp_path / "work")


def test_child_environment_points_everything_under_the_work_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_CPP_LIB", "/shared/llama.dll")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("LOCALM_HOME", "/real/home")
    env = confirm.child_env(tmp_path, {"HIP_VISIBLE_DEVICES": "-1"})
    assert "LLAMA_CPP_LIB" not in env
    assert env["HIP_VISIBLE_DEVICES"] == "-1", "the caller's extra variables are applied last"
    for key in ("LOCALM_HOME", "TEMP", "TMP", "TMPDIR", "HF_HOME"):
        assert confirm.under(env[key], tmp_path), key
    first, second = env["PYTHONPATH"].split(";" if sys.platform == "win32" else ":")[:2]
    assert Path(first) == tmp_path / "runtime-pkg" and Path(second) == confirm.REPO


def test_prepare_workdir_copies_the_products_runtime_package(tmp_path):
    confirm.prepare_workdir(tmp_path)
    pkg = tmp_path / "runtime-pkg" / "localm_llama_runtime" / "__init__.py"
    assert pkg.read_bytes() == (confirm.REPO / "runtime" / "localm_llama_runtime" / "__init__.py").read_bytes()
    assert not (tmp_path / "runtime-pkg" / "localm_llama_runtime" / "lib").exists()
    for name in ("home", "tmp", "hf", "logs", "stages"):
        assert (tmp_path / name).is_dir()


def test_cleanup_removes_only_the_directories_the_run_created(tmp_path):
    confirm.prepare_workdir(tmp_path)
    (tmp_path / "keepme.txt").write_text("x", encoding="utf-8")
    (tmp_path / "model-cache").mkdir()
    confirm.cleanup_workdir(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keepme.txt", "logs", "model-cache", "stages"]


# --------------------------------------------------------------------------- #
#  Real child processes                                                        #
# --------------------------------------------------------------------------- #

def test_a_child_stage_reports_its_result_through_the_real_process_plumbing(tmp_path):
    confirm.prepare_workdir(tmp_path)
    res = confirm.run_child("no-such-stage", {}, tmp_path, timeout=120)
    assert res.returncode == 0 and not res.timed_out
    assert res.out == {"error": "unknown stage 'no-such-stage'"}
    assert "unknown stage" in confirm.child_failure("no-such-stage", res)
    assert (tmp_path / "logs" / "no-such-stage.log").is_file()


def test_a_stage_that_overruns_is_killed_and_reported_as_a_timeout(tmp_path):
    confirm.prepare_workdir(tmp_path)
    res = confirm.run_child("no-such-stage", {}, tmp_path, timeout=0)
    assert res.timed_out
    assert "timed out" in confirm.child_failure("no-such-stage", res)
    assert not confirm._LIVE_PIDS


def test_kill_tree_stops_a_process_and_its_children():
    import psutil
    parent = subprocess.Popen([sys.executable, "-c", textwrap.dedent("""
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        print(child.pid, flush=True)
        time.sleep(60)
    """)], stdout=subprocess.PIPE, text=True)
    try:
        child_pid = int(parent.stdout.readline())
        assert psutil.pid_exists(child_pid)
        confirm.kill_tree(parent.pid)
        parent.wait(timeout=30)
        for _ in range(100):
            if not psutil.pid_exists(child_pid):
                break
            time.sleep(0.1)
        assert not psutil.pid_exists(child_pid)
    finally:
        if parent.poll() is None:
            parent.kill()


def test_a_stage_that_wrote_nothing_is_reported_with_the_log_tail(tmp_path):
    res = confirm.ChildResult(returncode=3, out=None, log_tail="Traceback ... boom")
    assert "wrote no result (exit 3)" in confirm.child_failure("probe", res)
    assert "boom" in confirm.child_failure("probe", res)
    assert confirm.child_failure("probe", confirm.ChildResult(0, {"ok": 1}, "")) is None


def _pins_probe(tmp_path, preload: Path | None) -> dict:
    code = textwrap.dedent(f"""
        import importlib.util, json, sys
        spec = importlib.util.spec_from_file_location("confirm", r"{_SCRIPTS / 'confirm_rocm_runtime.py'}")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        pre = {str(preload)!r}
        if pre != "None":
            mod._preload_pins(pre)
        from localm import setup_llama as sl
        from localm.setup_llama import assets, rocm_cpu, cli
        print(json.dumps({{"tag": sl._ROCM_TAG, "build": sl._ROCM_BUILD, "cpu": sl._ROCM_CPU_TAG,
            "asset": sl._ROCM_CPU_ASSET, "url": sl.DEFAULT_URL, "sha": sl.DEFAULT_URL_SHA256,
            "assets_tag": assets._ROCM_TAG, "cli_build": cli._ROCM_BUILD,
            "overlay_url": rocm_cpu.rocm_cpu_overlay_url(),
            "table_has_cpu": sl._ROCM_CPU_ASSET in sl._PINNED_FALLBACK_SHA256,
            "rocm_cpu_build": rocm_cpu._ROCM_BUILD}}))
    """)
    env = confirm.child_env(tmp_path)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_candidate_pins_reach_every_module_that_copies_them(tmp_path):
    confirm.prepare_workdir(tmp_path)
    api = fake_api.FakeApi()
    cand = bump.fetch_candidate(fake_api.TAG, api)
    pins = tmp_path / "candidate_pins.py"
    pins.write_text(bump.rewrite_pins(fake_api.REAL_PINS, cand), encoding="utf-8")

    control = _pins_probe(tmp_path, None)
    assert control["tag"] == fake_api.REAL_VIEW.tag, "without the preload the repo's own pins apply"

    got = _pins_probe(tmp_path, pins)
    assert got["tag"] == got["assets_tag"] == fake_api.TAG
    assert got["cpu"] == fake_api.CPU_TAG and got["asset"] == f"llama-{fake_api.CPU_TAG}-bin-win-cpu-x64.zip"
    assert got["build"] == got["cli_build"] == got["rocm_cpu_build"] == f"{fake_api.TAG}-cpu-{fake_api.CPU_TAG}"
    assert got["url"].endswith(f"{fake_api.TAG}/llama-{fake_api.TAG}-windows-rocm-gfx103X-x64.zip")
    assert got["sha"] == cand.gfx103x_sha256
    assert got["overlay_url"].endswith(f"/{fake_api.CPU_TAG}/llama-{fake_api.CPU_TAG}-bin-win-cpu-x64.zip")
    assert got["table_has_cpu"] is True


# --------------------------------------------------------------------------- #
#  The test model                                                              #
# --------------------------------------------------------------------------- #

class _Resp:
    def __init__(self, data: bytes):
        self._data, self._pos = data, 0

    def read(self, n):
        chunk = self._data[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _opener(data: bytes, calls: list):
    def opener(req, timeout=None):
        calls.append(req.full_url)
        return _Resp(data)
    return opener


def test_the_model_is_downloaded_once_verified_and_then_taken_from_the_cache(tmp_path):
    data = b"gguf-bytes" * 1000
    sha = hashlib.sha256(data).hexdigest()
    calls: list = []
    path, source, err = confirm.ensure_model(tmp_path / "c", sha256=sha, opener=_opener(data, calls))
    assert (source, err) == ("downloaded", "") and path.read_bytes() == data
    assert [p.name for p in path.parent.iterdir()] == [confirm.MODEL_FILE]
    path, source, err = confirm.ensure_model(tmp_path / "c", sha256=sha, opener=_opener(data, calls))
    assert source == "from the cache" and len(calls) == 1


def test_a_download_with_the_wrong_digest_is_discarded(tmp_path):
    path, source, err = confirm.ensure_model(tmp_path / "c", sha256="0" * 64, opener=_opener(b"xx", []))
    assert path is None and "expected" in err
    assert list((tmp_path / "c").iterdir()) == []


def test_a_corrupt_cached_model_is_replaced_not_trusted(tmp_path):
    data = b"real" * 100
    sha = hashlib.sha256(data).hexdigest()
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / confirm.MODEL_FILE).write_bytes(b"truncated")
    calls: list = []
    path, source, err = confirm.ensure_model(tmp_path / "c", sha256=sha, opener=_opener(data, calls))
    assert source == "downloaded" and path.read_bytes() == data and len(calls) == 1


def test_an_unreachable_model_host_is_an_error_string_not_an_exception(tmp_path):
    def opener(req, timeout=None):
        raise OSError("network is unreachable")
    path, source, err = confirm.ensure_model(tmp_path / "c", opener=opener)
    assert path is None and "network is unreachable" in err


def test_the_pinned_model_url_names_an_immutable_revision():
    assert confirm.MODEL_REVISION in confirm.MODEL_URL and "/main/" not in confirm.MODEL_URL
    assert len(confirm.MODEL_SHA256) == 64


def test_cache_location(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALM_PIN_CACHE_DIR", raising=False)
    assert confirm.cache_location(None, tmp_path) == tmp_path / "model-cache"
    monkeypatch.setenv("LOCALM_PIN_CACHE_DIR", str(tmp_path / "shared"))
    assert confirm.cache_location(None, tmp_path) == tmp_path / "shared" / "rocm"
    assert confirm.cache_location(str(tmp_path / "x"), tmp_path) == tmp_path / "x"


# --------------------------------------------------------------------------- #
#  The whole run, with canned stage results                                    #
# --------------------------------------------------------------------------- #

class Stages:
    """Stands in for run_child: returns what each stage would have written for
    candidate *cand*, and records the order and payloads. Everything else in the
    run is the script."""

    def __init__(self, work: Path, cand: bump.Candidate, **over):
        self.work, self.cand = work, cand
        self.calls: list = []
        self.lib = work / "runtime-pkg" / "localm_llama_runtime" / "lib"
        self.over = over

    def hw(self, workdir):
        return {**HW_OK, "localm_file": str(confirm.REPO / "localm" / "__init__.py"),
                "home_dir": str(workdir / "home"), "runtime_lib": str(self.lib),
                "tmp_dir": str(workdir / "tmp"), "llama_cpp_lib_env": ""}

    def install(self):
        c = self.cand
        return {**INSTALL_OK, "target": str(self.lib), "listing_ok": True, "override_ok": True,
                "marker": {"backend": "amd-rocm", "build": f"{c.tag}-cpu-{c.cpu_tag}"},
                "overlay": {"tag": c.cpu_tag, "variant": "ggml-cpu-haswell.dll"},
                "resolved": {"url": ("https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/"
                                     f"{c.tag}/{c.gfx103x_asset}"), "sha256": c.gfx103x_sha256}}

    def probe(self):
        return probe_ok(self.lib, identity={"ggml_commit": self.cand.cpu_commit[:8], "ggml_version": "0.26.0"})

    def __call__(self, stage, payload, workdir, *, extra_env=None, timeout=None):
        self.calls.append((stage, payload, extra_env))
        if stage in self.over:
            value = self.over[stage]
            if isinstance(value, Exception):
                raise value
            return value
        if stage == "hw":
            out = self.hw(workdir)
        elif stage == "model":
            out = {"architecture": "llama", "refusal": None, "bytes": MODEL_BYTES}
        elif stage == "install":
            out = self.install()
        elif stage == "probe":
            out = self.probe()
        elif stage == "probe-cpu":
            out = {**self.probe(), "cpu": _cpu(self.lib), "gpu": None}
        else:
            raise AssertionError(stage)
        return confirm.ChildResult(0, out, "")


@pytest.fixture
def world(tmp_path, monkeypatch):
    """main() against the real pins file with a fake GitHub API and canned stages."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    work = tmp_path / "work"
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x" * 10)
    monkeypatch.setattr(confirm, "ensure_model", lambda cache_dir: (model, "from the cache", ""))
    pins = bump.read_pins(fake_api.REAL_PINS)
    api = fake_api.FakeApi(tag=pins.tag)
    body = fake_api.lemonade_body(pins.tag, commit="71ad0")
    body["published_at"] = "2026-10-08T20:46:47Z"
    for a in body["assets"]:
        if a["name"] in pins.rocm_table:
            a["digest"] = "sha256:" + pins.rocm_table[a["name"]]
    api.routes = {
        f"repos/{bump.LEMONADE_REPO}/releases/tags/{pins.tag}": body,
        f"repos/{bump.UPSTREAM_REPO}/releases?per_page=100&page=1": [
            fake_api.upstream_release(pins.cpu_tag, CPU_COMMIT, "2026-10-08T16:00:00Z",
                                      cpu_sha=next(iter(pins.cpu_table.values())))],
        f"repos/{bump.UPSTREAM_REPO}/releases?per_page=100&page=2": [],
        f"repos/{bump.UPSTREAM_REPO}/git/ref/tags/{pins.cpu_tag}": {
            "object": {"type": "commit", "sha": CPU_COMMIT}},
    }

    class World:
        pass

    w = World()
    w.work, w.receipt, w.api, w.pins, w.tmp = work, tmp_path / "r.json", api, pins, tmp_path
    w.cand = bump.fetch_candidate(pins.tag, api)
    w.stages = Stages(work, w.cand)

    def go(*argv, stages=None, fetch=None):
        args = [*argv, "--workdir", str(work), "--receipt", str(w.receipt)]
        code = confirm.main(args, fetch=fetch or api, runner=stages or w.stages)
        return code, json.loads(w.receipt.read_text(encoding="utf-8"))

    w.go = go
    return w


def _ran(stages):
    return [s for s, _, _ in stages.calls]


def test_a_current_run_where_everything_passes_writes_a_pass_receipt(world):
    code, r = world.go("--current")
    assert code == 0, r["why"]
    assert (r["verdict"], r["current"], r["tag"], r["schema"], r["component"]) == (
        "PASS", True, world.pins.tag, 1, "rocm")
    assert all(c["status"] == "PASS" and c["required"] for c in r["checks"].values())
    assert _ran(world.stages) == ["hw", "model", "install", "probe", "probe-cpu"]
    assert r["candidate"]["cpu_tag"] == world.pins.cpu_tag and r["not_measured"] == []
    assert r["hardware"]["amd_gfx_family"] == "gfx103x" and r["not_covered"]


def test_a_current_run_applies_no_pins_override_but_a_candidate_run_does(world):
    world.go("--current")
    assert "pins_file" not in world.stages.calls[2][1]
    fake = fake_api.FakeApi()
    cand = bump.fetch_candidate(fake_api.TAG, fake)
    stages = Stages(world.work, cand)
    code, r = world.go("--tag", fake_api.TAG, stages=stages, fetch=fake)
    assert code == 0, r["why"]
    install = stages.calls[2][1]
    pinned = Path(install["pins_file"]).read_text(encoding="utf-8")
    assert pinned.count(f'_ROCM_TAG = "{fake_api.TAG}"') == 1
    assert (install["expected_tag"], install["expected_cpu_tag"]) == (fake_api.TAG, fake_api.CPU_TAG)
    assert r["current"] is False and r["tag"] == fake_api.TAG


def test_the_probe_stages_get_the_runtime_and_model_and_the_cpu_stage_hides_the_gpu(world):
    world.go("--current")
    calls = {s: (p, e) for s, p, e in world.stages.calls}
    assert calls["probe"][0]["expected_lib_dir"] == str(world.stages.lib)
    assert calls["probe"][0]["model"] == str(world.tmp / "model.gguf")
    assert calls["probe"][1] is None
    assert calls["probe-cpu"][1] == {"HIP_VISIBLE_DEVICES": "-1", "ROCR_VISIBLE_DEVICES": "-1"}


def test_the_receipt_of_a_passing_candidate_run_is_accepted_by_the_bump(world, tmp_path):
    fake = fake_api.FakeApi()
    stages = Stages(world.work, bump.fetch_candidate(fake_api.TAG, fake))
    code, r = world.go("--tag", fake_api.TAG, stages=stages, fetch=fake)
    assert code == 0, r["why"]
    assert bump.load_receipt(world.receipt, fake_api.TAG)["candidate"]["tag"] == fake_api.TAG
    pins = tmp_path / "pins.py"
    pins.write_bytes(fake_api.REAL_PINS.encode("utf-8"))
    assert bump.main(["--tag", fake_api.TAG, "--receipt", str(world.receipt)], fetch=fake, pins_path=pins) == 0


def test_the_receipt_exists_and_says_inconclusive_before_any_stage_runs(world):
    seen = {}

    def runner(stage, payload, workdir, **kw):
        seen[stage] = json.loads(world.receipt.read_text(encoding="utf-8"))["verdict"]
        raise SystemExit(3)

    code, r = world.go("--current", stages=runner)
    assert seen == {"hw": "INCONCLUSIVE"}
    assert code == 2 and r["verdict"] == "INCONCLUSIVE" and "uncaught" in r


def test_an_uncaught_exception_is_inconclusive_never_a_fail_or_a_pass(world):
    stages = Stages(world.work, world.cand, hw=RuntimeError("boom"))
    code, r = world.go("--current", stages=stages)
    assert code == 2 and r["verdict"] == "INCONCLUSIVE"
    assert "boom" in r["uncaught"]
    assert all(c["status"] == "SKIP" for c in r["checks"].values())
    assert "raised an exception" in r["checks"]["hardware"]["detail"]


def test_an_exception_after_a_failed_check_keeps_the_fail(world):
    stages = Stages(world.work, world.cand, **{"probe-cpu": RuntimeError("late")})
    refused = {**stages.probe(), "load": {"status": "abi_mismatch", "detail": "context params moved"}}
    stages.over["probe"] = confirm.ChildResult(0, refused, "")
    code, r = world.go("--current", stages=stages)
    assert code == 1 and r["verdict"] == "FAIL" and "late" in r["uncaught"]


def test_a_box_without_an_amd_gfx103x_gpu_is_inconclusive_and_spends_nothing(world):
    stages = Stages(world.work, world.cand)
    stages.over["hw"] = confirm.ChildResult(0, {
        **stages.hw(world.work), "vendors": ["nvidia"], "gpu_names": "nvidia rtx", "amd_gfx_family": ""}, "")
    code, r = world.go("--current", stages=stages)
    assert code == 2 and r["checks"]["hardware"]["status"] == "SKIP"
    assert _ran(stages) == ["hw"], "no download, no install on hardware that cannot run it"


def test_paths_outside_the_work_directory_stop_the_run_before_any_install(world):
    stages = Stages(world.work, world.cand)
    stages.over["hw"] = confirm.ChildResult(0, {
        **stages.hw(world.work), "home_dir": str(world.tmp / "real-home")}, "")
    code, r = world.go("--current", stages=stages)
    assert code == 2 and "home_dir" in r["checks"]["isolation"]["detail"]
    assert _ran(stages) == ["hw"]


def test_an_unreachable_api_is_inconclusive(world):
    def down(path):
        raise bump.UpstreamUnreadable("HTTP 403")

    code, r = world.go("--current", fetch=down)
    assert code == 2 and r["checks"]["candidate"]["status"] == "SKIP" and "403" in r["checks"]["candidate"]["detail"]
    assert _ran(world.stages) == ["hw"]


def test_a_candidate_that_cannot_be_paired_with_an_upstream_cpu_archive_fails(world):
    fake = fake_api.FakeApi()
    fake.routes[f"repos/{bump.LEMONADE_REPO}/releases/tags/{fake_api.TAG}"]["body"] = (
        "**Llama.cpp Commit Hash**: 00000")
    code, r = world.go("--tag", fake_api.TAG, fetch=fake)
    assert code == 1 and "no upstream" in r["checks"]["candidate"]["detail"]


def test_a_current_pin_that_disagrees_with_the_published_release_fails(world):
    release = world.api.routes[f"repos/{bump.LEMONADE_REPO}/releases/tags/{world.pins.tag}"]
    release["assets"][0]["digest"] = "sha256:" + "9" * 64
    code, r = world.go("--current")
    assert code == 1 and "disagrees with the upstream release" in r["checks"]["candidate"]["detail"]
    assert _ran(world.stages) == ["hw"]


def test_a_missing_test_model_is_inconclusive(world, monkeypatch):
    monkeypatch.setattr(confirm, "ensure_model",
                        lambda cache_dir: (None, "", "could not obtain the test model: offline"))
    code, r = world.go("--current")
    assert code == 2 and "offline" in r["checks"]["model"]["detail"]
    assert "install" not in _ran(world.stages)


def test_a_test_model_that_is_not_a_causal_chat_model_fails(world):
    bad = confirm.ChildResult(0, {"architecture": "bert", "refusal": "embedding", "bytes": 5}, "")
    code, r = world.go("--current", stages=Stages(world.work, world.cand, model=bad))
    assert code == 1 and "not a causal chat model" in r["checks"]["model"]["detail"]


def test_a_failed_install_stops_the_run_and_nothing_is_probed(world):
    stages = Stages(world.work, world.cand)
    bad = {**stages.install(), "marker": {"backend": "amd-rocm", "build": world.pins.tag}, "overlay": None}
    stages.over["install"] = confirm.ChildResult(0, bad, "")
    code, r = world.go("--current", stages=stages)
    assert code == 1 and r["checks"]["install"]["status"] == "FAIL"
    assert _ran(stages) == ["hw", "model", "install"]
    assert r["checks"]["abi"]["detail"].startswith("not run:")


def test_a_network_failed_install_is_retried_then_inconclusive(world):
    stages = Stages(world.work, world.cand)
    net = {**stages.install(), "exit_code": 1, "marker": {}, "overlay": None, "llama_dll": False}
    stages.over["install"] = confirm.ChildResult(0, net, "Download failed: Remote end closed connection")
    code, r = world.go("--current", stages=stages)
    assert code == 2 and r["checks"]["install"]["status"] == "SKIP"
    assert _ran(stages).count("install") == confirm.INSTALL_ATTEMPTS


def test_a_build_failed_install_is_not_retried(world):
    stages = Stages(world.work, world.cand)
    bad = {**stages.install(), "exit_code": 1, "marker": {}, "overlay": None, "llama_dll": False}
    stages.over["install"] = confirm.ChildResult(0, bad, "the archive did not contain llama.dll")
    code, r = world.go("--current", stages=stages)
    assert code == 1 and _ran(stages).count("install") == 1


def test_a_stage_that_timed_out_or_wrote_nothing_is_inconclusive(world):
    for result, word in ((confirm.ChildResult(None, None, "", timed_out=True), "timed out"),
                         (confirm.ChildResult(1, None, "Traceback"), "wrote no result"),
                         (confirm.ChildResult(0, {"error": "Traceback: ImportError"}, ""), "raised")):
        code, r = world.go("--current", stages=Stages(world.work, world.cand, probe=result))
        assert code == 2, word
        assert word in r["checks"]["abi"]["detail"]


def test_an_abi_refusal_is_a_fail_and_the_other_checks_still_report(world):
    stages = Stages(world.work, world.cand)
    refused = {**stages.probe(), "load": {"status": "abi_mismatch", "detail": "context params moved"}}
    stages.over["probe"] = confirm.ChildResult(0, refused, "")
    code, r = world.go("--current", stages=stages)
    assert code == 1 and r["checks"]["abi"]["status"] == "FAIL" and "moved" in r["why"]
    assert r["checks"]["cpu_generate"]["status"] == "PASS"


def test_a_gpu_run_that_never_used_video_memory_fails_the_run(world):
    stages = Stages(world.work, world.cand)
    probe = stages.probe()
    probe["gpu"]["worker_vram"] = probe["gpu_ref"]["worker_vram"]
    stages.over["probe"] = confirm.ChildResult(0, probe, "")
    code, r = world.go("--current", stages=stages)
    assert code == 1 and r["checks"]["gpu_generate"]["status"] == "FAIL"
    assert r["checks"]["cpu_generate"]["status"] == "PASS"


def test_low_disk_space_is_inconclusive_before_anything_is_created(world, monkeypatch):
    monkeypatch.setattr(confirm.shutil, "disk_usage", lambda p: type("U", (), {"free": 1 << 20})())
    code, r = world.go("--current")
    assert code == 2 and "free under the work directory" in r["checks"]["isolation"]["detail"]
    assert world.stages.calls == [] and not (world.work / "runtime-pkg").exists()


def test_the_work_directory_is_cleaned_unless_kept(world):
    world.go("--current")
    assert not (world.work / "runtime-pkg").exists() and not (world.work / "home").exists()
    world.go("--current", "--keep")
    assert (world.work / "runtime-pkg").is_dir() and (world.work / "home").is_dir()


def test_arguments_are_validated():
    for argv in (["--tag", "b1", "--current", "--workdir", "x", "--receipt", "y"],
                 ["--tag", "latest", "--workdir", "x", "--receipt", "y"],
                 ["--workdir", "x", "--receipt", "y"],
                 ["--current", "--receipt", "y"]):
        with pytest.raises(SystemExit):
            confirm.main(argv)
