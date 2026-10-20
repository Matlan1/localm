# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline tests for scripts/confirm_llama_runtime.py: the known-answer check, the
verdict a probe result earns (GPU use is decided by llama.cpp's own load report),
the exit code, the receipt pin_weekly.py reads, the archive bookkeeping and the PE
import reader. The probe itself needs a real runtime and runs on hardware."""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


confirm = _load("confirm_llama_runtime")

GOOD = ["The capital of France is Paris.", "The planet we live on is called Earth."]
CPU_ONLY = [{"backend": "CPU", "mib": 91.27, "is_ram": True},
            {"backend": "CPU_REPACK", "mib": 7.59, "is_ram": True}]
ON_VULKAN = [{"backend": "Vulkan0", "mib": 98.87, "is_ram": False},
             {"backend": "Vulkan_Host", "mib": 28.69, "is_ram": True}]
VULKAN_DEVICES = [{"name": "Vulkan0", "gpu": True}, {"name": "CPU", "gpu": False}]
ROCM_DEVICES = [{"name": "ROCm0", "gpu": True}, {"name": "CPU", "gpu": False}]


def _generated(**kw):
    res = {"verdict": "GENERATED", "why": "loaded and generated", "abi": "ok",
           "devices": VULKAN_DEVICES, "answers": list(GOOD), "placement": ON_VULKAN}
    res.update(kw)
    return res


# --------------------------------------------------------------------------- #
#  Known answers                                                              #
# --------------------------------------------------------------------------- #

def test_real_answers_pass_the_known_answer_check():
    ok, why = confirm.answer_check(GOOD)
    assert ok, why


@pytest.mark.parametrize("answers", [
    ["aaaaaaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaaa"],
    ["Paris Paris Paris", "zzzzzzzzzzzzzz"],
    ["The capital of France is Lyon.", "Earth."],
    ["", ""],
    ["Comparison is the parisian way.", "Earthly delights."],
    ["Paris."],
    None,
])
def test_garbage_or_wrong_answers_fail_the_known_answer_check(answers):
    ok, why = confirm.answer_check(answers)
    assert not ok and why


def test_the_old_length_and_letter_bar_would_have_passed_garbage_this_one_does_not():
    garbage = "a" * 40
    assert len(garbage) >= 10 and any(c.isalpha() for c in garbage)
    assert not confirm.answer_check([garbage, garbage])[0]


# --------------------------------------------------------------------------- #
#  judge: GPU use is shown by the load report, not by the device list          #
# --------------------------------------------------------------------------- #

def test_vulkan_with_weights_on_the_registered_device_passes():
    verdict, why = confirm.judge("vulkan", _generated(), require_gpu=True)
    assert verdict == confirm.PASS and "Vulkan0 98.87 MiB" in why


@pytest.mark.parametrize("require_gpu, expected", [(True, "FAIL"), (False, "INCONCLUSIVE")])
def test_a_gpu_device_registered_but_weights_left_in_ram_is_never_a_pass(require_gpu, expected):
    res = _generated(devices=ROCM_DEVICES, placement=CPU_ONLY)
    verdict, why = confirm.judge("hip", res, require_gpu=require_gpu)
    assert verdict == expected and "generated on the CPU" in why and "ROCm0" in why


def test_weights_only_in_ram_are_inconclusive_for_an_also_measured_backend():
    res = _generated(devices=ROCM_DEVICES, placement=CPU_ONLY)
    for require_gpu in (True, False):
        verdict, why = confirm.judge("hip", res, require_gpu=require_gpu, best_effort=True)
        assert verdict == confirm.INCONCLUSIVE and "generated on the CPU" in why


def test_a_gpu_buffer_named_after_a_device_that_did_not_register_is_not_a_pass():
    res = _generated(devices=ROCM_DEVICES, placement=ON_VULKAN)
    assert confirm.judge("hip", res, require_gpu=True)[0] == confirm.FAIL


def test_a_zero_size_gpu_buffer_is_not_gpu_use():
    res = _generated(placement=[{"backend": "Vulkan0", "mib": 0.0, "is_ram": False}, *CPU_ONLY])
    assert confirm.judge("vulkan", res, require_gpu=True)[0] == confirm.FAIL


@pytest.mark.parametrize("placement", [None, []])
def test_no_load_report_is_inconclusive_never_pass(placement):
    verdict, why = confirm.judge("vulkan", _generated(placement=placement), require_gpu=True)
    assert verdict == confirm.INCONCLUSIVE and "not captured" in why


@pytest.mark.parametrize("best_effort, require_gpu, expected", [
    (False, True, "FAIL"), (False, False, "INCONCLUSIVE"),
    (True, True, "NOT MEASURED"), (True, False, "NOT MEASURED")])
def test_a_gpu_build_that_registered_no_gpu_device(best_effort, require_gpu, expected):
    res = {"verdict": "LOADED", "why": "no GPU device registered", "abi": "ok",
           "devices": [{"name": "CPU", "gpu": False}],
           "plugin": "did not load: Could not find module", "missing_imports": ["nvcuda.dll"]}
    verdict, why = confirm.judge("cuda", res, require_gpu=require_gpu, best_effort=best_effort)
    assert verdict == expected
    assert "registered no GPU device" in why and "nvcuda.dll" in why


def test_a_wrong_answer_fails_even_on_the_gpu():
    res = _generated(answers=["aaaaaaaa", "aaaaaaaa"])
    assert confirm.judge("vulkan", res, require_gpu=True)[0] == confirm.FAIL


def test_cpu_needs_the_known_answers_and_no_gpu():
    assert confirm.judge("cpu", _generated(devices=[{"name": "CPU", "gpu": False}],
                                           placement=CPU_ONLY))[0] == confirm.PASS
    assert confirm.judge("cpu", _generated(answers=["x" * 30, "y" * 30]))[0] == confirm.FAIL


def test_the_probes_own_fail_and_inconclusive_are_kept():
    for v in (confirm.FAIL, confirm.INCONCLUSIVE):
        assert confirm.judge("vulkan", {"verdict": v, "why": "w"}) == (v, "w")
    assert confirm.judge("vulkan", {"verdict": "WHAT"})[0] == confirm.INCONCLUSIVE


def test_no_model_is_inconclusive_for_cpu():
    res = {"verdict": "LOADED", "why": "no model to generate with: offline", "devices": []}
    assert confirm.judge("cpu", res)[0] == confirm.INCONCLUSIVE


# --------------------------------------------------------------------------- #
#  Exit code, plan, checks and the receipt pin_weekly.py reads                 #
# --------------------------------------------------------------------------- #

def _summary(results, also=None):
    return {"tag": "b200", "results": results, "also": also or {}, "lib_sha256": {},
            "core_sha256": {}}


def test_exit_code_counts_an_also_measured_fail_but_not_its_not_measured():
    ok = {"cpu": {"verdict": "PASS"}, "vulkan": {"verdict": "PASS"}}
    assert confirm.exit_code(_summary(ok, {"sycl": {"verdict": "NOT MEASURED"}}), ["cpu", "vulkan"]) == 0
    assert confirm.exit_code(_summary(ok, {"hip": {"verdict": "INCONCLUSIVE"}}), ["cpu", "vulkan"]) == 0
    assert confirm.exit_code(_summary(ok, {"cuda-12": {"verdict": "FAIL"}}), ["cpu", "vulkan"]) == 1
    half = {"cpu": {"verdict": "PASS"}, "vulkan": {"verdict": "INCONCLUSIVE"}}
    assert confirm.exit_code(_summary(half), ["cpu", "vulkan"]) == 2
    assert confirm.exit_code(_summary({"cpu": {"verdict": "PASS"}}), ["cpu", "vulkan"]) == 2


def test_also_measure_expands_cuda_to_both_lines_and_skips_required_backends():
    plan = confirm._also_plan(["cuda", "cpu", "hip", "hip"], ["cpu", "vulkan"])
    assert plan == [("cuda-12", "cuda", "cuda-12"), ("cuda-13", "cuda", "cuda-13"),
                    ("hip", "hip", None)]


def test_the_receipt_carries_the_verdict_pin_weekly_requires(tmp_path):
    pw = _load("pin_weekly")
    results = {"cpu": _generated(devices=[{"name": "CPU", "gpu": False}], placement=CPU_ONLY,
                                 verdict="PASS", fetched=[{"asset": "a.zip", "sha256": "1" * 64}]),
               "vulkan": _generated(verdict="PASS", fetched=[{"asset": "b.zip", "sha256": "2" * 64}])}
    also = {"sycl": {"verdict": "NOT MEASURED", "why": "no GPU", "abi": "ok",
                     "devices": [{"name": "CPU", "gpu": False}],
                     "fetched": [{"asset": "c.zip", "sha256": "3" * 64}]}}
    for rc, verdict in ((0, "PASS"), (1, "FAIL"), (2, "INCONCLUSIVE")):
        p = tmp_path / f"r{rc}.json"
        confirm._write_receipt(p, _summary(results, also), ["cpu", "vulkan"], rc)
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["verdict"] == verdict and data["schema"] == 1
        assert pw.receipt_problem(p, verdict) == ""
        assert all({"status", "required", "detail"} <= set(c) for c in data["checks"].values())
    names = list(data["checks"])
    assert names[:3] == ["cpu:generation", "vulkan:generation", "sycl:generation"]
    assert data["checks"]["sycl:generation"]["required"] is False
    assert data["checks"]["vulkan:generation"]["required"] is True
    assert data["checks"]["vulkan:gpu_device"]["detail"].startswith("Vulkan0")


def test_an_unverified_archive_is_inconclusive_and_a_mismatch_fails():
    assert confirm._archive_check({"fetched": [{"asset": "a", "sha256": None}]})[0] == confirm.INCONCLUSIVE
    assert confirm._archive_check({"fetched": [{"asset": "a", "sha256": "f" * 64}]})[0] == confirm.PASS
    why = f"could not place: ArtifactError: download sha256 {confirm._SHA_MISMATCH_TEXT} (x)"
    assert confirm._archive_check({"fetched": [{"asset": "a", "sha256": "f" * 64}],
                                   "why": why})[0] == confirm.FAIL
    assert confirm._archive_check({})[0] == confirm.NOT_MEASURED


def test_the_mismatch_text_is_what_setup_llamas_own_validator_raises(tmp_path):
    from localm.setup_llama import download
    arc = tmp_path / "a.zip"
    with zipfile.ZipFile(arc, "w") as z:
        z.writestr("llama.dll", b"x" * 64)
    with pytest.raises(download.ArtifactError) as info:
        download._validate_archive(arc, expected_sha256="0" * 64, min_size=1)
    assert confirm._SHA_MISMATCH_TEXT in str(info.value)


def _fake_provision(monkeypatch, behaviour):
    from localm import setup_llama as sl
    calls = []

    def provision(backend, dest, sha256, with_cudart, cuda_line=None, tag=None):
        calls.append((backend, cuda_line, tag, with_cudart))
        return behaviour(sl, dest)
    monkeypatch.setattr(sl, "_provision_backend", provision)
    monkeypatch.setattr(sl, "_fetch_and_place", lambda url, target, sha256=None: 1)
    return calls


def test_place_build_records_every_archive_and_whether_it_had_a_digest(monkeypatch, tmp_path):
    def behaviour(sl, dest):
        sl._fetch_and_place("https://h/r/llama-b1-bin-win-cuda-12.4-x64.zip", dest, "a" * 64)
        sl._fetch_and_place("https://h/r/cudart-llama-bin-win-cuda-12.4-x64.zip", dest, None)
        (dest / sl._lib_name()).write_bytes(b"lib")
    calls = _fake_provision(monkeypatch, behaviour)
    lib, fetched, err, kind = confirm._place_build("cuda", "b1", tmp_path / "d", "cuda-12")
    assert err == "" and kind == "" and Path(lib).name
    assert fetched == [{"asset": "llama-b1-bin-win-cuda-12.4-x64.zip", "sha256": "a" * 64},
                       {"asset": "cudart-llama-bin-win-cuda-12.4-x64.zip", "sha256": None}]
    assert calls == [("cuda", "cuda-12", "b1", True)]


def test_place_build_stops_at_a_digest_mismatch_without_retrying(monkeypatch, tmp_path):
    from localm.setup_llama.download import ArtifactError

    def behaviour(sl, dest):
        sl._fetch_and_place("https://h/r/x.zip", dest, "b" * 64)
        raise ArtifactError(f"download sha256 {confirm._SHA_MISMATCH_TEXT} (expected b, got c)")
    calls = _fake_provision(monkeypatch, behaviour)
    lib, fetched, err, kind = confirm._place_build("vulkan", "b1", tmp_path / "d")
    assert lib == "" and kind == "mismatch" and len(calls) == 1
    assert fetched == [{"asset": "x.zip", "sha256": "b" * 64}]


def test_place_build_retries_a_transient_failure(monkeypatch, tmp_path):
    attempts = []

    def behaviour(sl, dest):
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("Remote end closed connection without response")
        (dest / sl._lib_name()).write_bytes(b"lib")
    _fake_provision(monkeypatch, behaviour)
    lib, _fetched, err, _kind = confirm._place_build("cpu", "b1", tmp_path / "d")
    assert lib and err == "" and len(attempts) == 3


# --------------------------------------------------------------------------- #
#  PE imports                                                                 #
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(sys.platform != "win32", reason="reads a Windows extension module")
def test_pe_imports_reads_a_real_extension_modules_import_table():
    import _ctypes
    names = [n.lower() for n in confirm.pe_imports(Path(_ctypes.__file__))]
    assert f"python3{sys.version_info.minor}.dll" in names
    assert any(n.startswith("libffi") for n in names)
    assert len(names) == len(set(names))


def test_pe_imports_is_empty_for_anything_that_is_not_a_pe_image(tmp_path):
    (tmp_path / "x.dll").write_bytes(b"MZ" + b"\0" * 100)
    (tmp_path / "y.dll").write_bytes(b"not an image")
    assert confirm.pe_imports(tmp_path / "x.dll") == []
    assert confirm.pe_imports(tmp_path / "y.dll") == []
    assert confirm.pe_imports(tmp_path / "missing.dll") == []


def test_identity_compares_every_core_library_with_the_cpu_build():
    core = {"cpu": {"llama.dll": "1", "ggml.dll": "2"},
            "hip": {"llama.dll": "1", "ggml.dll": "2"},
            "sycl": {"llama.dll": "1", "ggml.dll": "9"}}
    assert confirm._identity(core, "hip")[0] == confirm.PASS
    status, detail = confirm._identity(core, "sycl")
    assert status == confirm.NOT_MEASURED and "ggml.dll" in detail
    assert confirm._identity({"hip": {"llama.dll": "1"}}, "hip")[0] == confirm.NOT_MEASURED
