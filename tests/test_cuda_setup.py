# SPDX-License-Identifier: AGPL-3.0-or-later
"""CUDA preflight, self-assembly, and the load-test + graceful fallback for
``localm setup-llama``.

CUDA is the visible "peak NVIDIA performance" option, so picking it has to land
or fall back cleanly. These tests pin, with no network and no real GPU:
  * the nvidia-smi banner is parsed into a driver / CUDA-capability report;
  * the CUDA build + matching cudart bundle are resolved from a release listing;
  * the dialogue picks the right action per preflight state;
  * a chosen backend that does not load falls back vulkan -> cpu;
  * an explicit --sha256 pin is NEVER bypassed by the fallback.
"""

from __future__ import annotations

import pytest

from localm import setup_llama as sl


# --------------------------- preflight parsing ---------------------------- #

def test_ver_tuple_parses_and_tolerates_junk():
    assert sl._ver_tuple("12.4") == (12, 4)
    assert sl._ver_tuple("13") == (13,)
    assert sl._ver_tuple("") == (0, 0)
    assert sl._ver_tuple("not.a.version") == (0, 0)


def test_nvidia_preflight_parses_banner(monkeypatch):
    def fake_smi(*args):
        if args and str(args[0]).startswith("--query-gpu"):
            return "NVIDIA GeForce RTX 4070\n"
        return ("|  NVIDIA-SMI 552.22   Driver Version: 552.22   "
                "CUDA Version: 12.4  |\n")
    monkeypatch.setattr(sl, "_nvidia_smi", fake_smi)
    info = sl.nvidia_preflight()
    assert info.present
    assert info.driver_version == "552.22"
    assert info.cuda_capability == "12.4"
    assert info.gpu_name == "NVIDIA GeForce RTX 4070"
    assert info.driver_ok is True


def test_nvidia_preflight_absent_when_no_smi(monkeypatch):
    monkeypatch.setattr(sl, "_nvidia_smi", lambda *a: "")
    info = sl.nvidia_preflight()
    assert info.present is False
    assert info.driver_ok is True   # unknown capability never blocks


def test_driver_ok_false_for_old_capability():
    assert sl.NvidiaInfo(present=True, cuda_capability="11.2").driver_ok is False
    assert sl.NvidiaInfo(present=True, cuda_capability="12.4").driver_ok is True
    assert sl.NvidiaInfo(present=True, cuda_capability="13.3").driver_ok is True


# --------------------------- asset resolution ----------------------------- #

_FAKE_ASSETS = [
    {"name": "llama-b9685-bin-win-cuda-12.4-x64.zip",
     "browser_download_url": "https://x/llama-cuda-12.4.zip", "size": 200_000_000},
    {"name": "llama-b9685-bin-win-cuda-13.3-x64.zip",
     "browser_download_url": "https://x/llama-cuda-13.3.zip", "size": 200_000_000},
    {"name": "cudart-llama-bin-win-cuda-12.4-x64.zip",
     "browser_download_url": "https://x/cudart-12.4.zip", "size": 400_000_000},
    {"name": "cudart-llama-bin-win-cuda-13.3-x64.zip",
     "browser_download_url": "https://x/cudart-13.3.zip", "size": 400_000_000},
    {"name": "llama-b9685-bin-win-vulkan-x64.zip",
     "browser_download_url": "https://x/vulkan.zip", "size": 100_000_000},
]


def test_resolve_cuda_pair_prefers_12_line(monkeypatch):
    monkeypatch.setattr(sl, "_release_assets", lambda tag: _FAKE_ASSETS)
    build, cudart = sl._resolve_cuda_pair("b9685")
    assert build["name"] == "llama-b9685-bin-win-cuda-12.4-x64.zip"
    assert cudart["name"] == "cudart-llama-bin-win-cuda-12.4-x64.zip"


def test_resolve_cuda_pair_none_when_listing_empty(monkeypatch):
    monkeypatch.setattr(sl, "_release_assets", lambda tag: [])
    build, cudart = sl._resolve_cuda_pair("b9685")
    assert build is None and cudart is None


def test_pick_asset_requires_all_needles():
    assert sl._pick_asset(_FAKE_ASSETS, "vulkan")["name"].endswith("vulkan-x64.zip")
    assert sl._pick_asset(_FAKE_ASSETS, "cudart", "win-cuda-12") is not None
    assert sl._pick_asset(_FAKE_ASSETS, "rocm") is None


# --------------------------- the CUDA dialogue ----------------------------- #

def _confirm(monkeypatch, answer: bool):
    monkeypatch.setattr(sl.click, "confirm", lambda *a, **k: answer)


def test_dialogue_driver_ok_confirm_fetches_cuda(monkeypatch):
    _confirm(monkeypatch, True)
    info = sl.NvidiaInfo(present=True, gpu_name="RTX 4070",
                         driver_version="552.22", cuda_capability="12.4")
    assert sl._cuda_setup_dialogue(info, assume_yes=False) == ("cuda", True)


def test_dialogue_driver_ok_decline_falls_back_to_vulkan(monkeypatch):
    _confirm(monkeypatch, False)
    info = sl.NvidiaInfo(present=True, cuda_capability="12.4")
    assert sl._cuda_setup_dialogue(info, assume_yes=False) == ("vulkan", False)


def test_dialogue_old_driver_recommends_vulkan(monkeypatch):
    # No confirm should be needed; an old driver cannot be self-assembled.
    _confirm(monkeypatch, True)
    info = sl.NvidiaInfo(present=True, driver_version="460.0", cuda_capability="11.2")
    assert sl._cuda_setup_dialogue(info, assume_yes=False) == ("vulkan", False)


def test_dialogue_no_nvidia_continue_is_users_choice(monkeypatch):
    _confirm(monkeypatch, True)
    info = sl.NvidiaInfo(present=False)
    assert sl._cuda_setup_dialogue(info, assume_yes=False) == ("cuda", True)


def test_dialogue_no_nvidia_decline_uses_vulkan(monkeypatch):
    _confirm(monkeypatch, False)
    info = sl.NvidiaInfo(present=False)
    assert sl._cuda_setup_dialogue(info, assume_yes=False) == ("vulkan", False)


def test_dialogue_assume_yes_fetches_when_driver_ok():
    info = sl.NvidiaInfo(present=True, cuda_capability="12.4")
    assert sl._cuda_setup_dialogue(info, assume_yes=True) == ("cuda", True)


def test_dialogue_assume_yes_uses_vulkan_when_no_nvidia():
    info = sl.NvidiaInfo(present=False)
    assert sl._cuda_setup_dialogue(info, assume_yes=True) == ("vulkan", False)


# --------------------------- off-profile warning -------------------------- #

def _fake_vendors(monkeypatch, vendors):
    from localm import hwdetect
    monkeypatch.setattr(hwdetect, "detect",
                        lambda: hwdetect.Detection(vendors=list(vendors)))


def test_warn_off_profile_flags_mismatch(monkeypatch, capsys):
    _fake_vendors(monkeypatch, ["amd"])
    sl._warn_off_profile("cuda")                 # cuda on an AMD-only box
    assert "Heads up" in capsys.readouterr().out


def test_warn_off_profile_quiet_when_matched(monkeypatch, capsys):
    _fake_vendors(monkeypatch, ["nvidia"])
    sl._warn_off_profile("cuda")
    assert "Heads up" not in capsys.readouterr().out


def test_warn_off_profile_quiet_for_universal_backend(monkeypatch, capsys):
    _fake_vendors(monkeypatch, ["amd"])
    sl._warn_off_profile("vulkan")               # not a vendor-specific backend
    assert "Heads up" not in capsys.readouterr().out


# --------------------------- load-test + fallback ------------------------- #

def _stub_provision(monkeypatch):
    """_provision_backend writes the expected lib name into target; wheel install
    is a no-op. Lets us drive the fallback purely via _native_loads_ok."""
    lib = sl._lib_name()

    def fake_provision(backend, target, sha256, with_cudart):
        (target / lib).write_bytes(b"x")
    monkeypatch.setattr(sl, "_provision_backend", fake_provision)
    monkeypatch.setattr(sl, "_install_runtime_wheel", lambda pkg: True)


def test_fallback_chosen_loads_returns_chosen(monkeypatch, tmp_path):
    _stub_provision(monkeypatch)
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: (True, ""))
    assert sl._provision_with_fallback("cuda", tmp_path, None, True) == "cuda"


def test_fallback_to_vulkan_when_chosen_does_not_load(monkeypatch, tmp_path):
    _stub_provision(monkeypatch)
    seq = iter([(False, "cuda failed to load"), (True, "")])
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: next(seq))
    assert sl._provision_with_fallback("cuda", tmp_path, None, True) == "vulkan"


def test_fallback_to_cpu_when_vulkan_also_fails(monkeypatch, tmp_path):
    _stub_provision(monkeypatch)
    seq = iter([(False, "cuda no"), (False, "vulkan no"), (True, "")])
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: next(seq))
    assert sl._provision_with_fallback("cuda", tmp_path, None, True) == "cpu"


def test_nothing_loads_raises_reportable_error(monkeypatch, tmp_path):
    _stub_provision(monkeypatch)
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: (False, "nope"))
    from localm.bugreport import LocalmError
    with pytest.raises(LocalmError) as ei:
        sl._provision_with_fallback("cuda", tmp_path, None, True)
    assert "no llama.cpp backend" in ei.value.summary


def test_pinned_sha256_never_falls_back(monkeypatch, tmp_path):
    """A --sha256 pin means 'exactly this artifact' - a validation failure must
    stop, not silently swap to an unpinned vulkan build."""
    def boom(*a, **k):
        raise sl.ArtifactError("sha mismatch")
    monkeypatch.setattr(sl, "_provision_backend", boom)
    monkeypatch.setattr(sl, "_install_runtime_wheel", lambda pkg: True)
    # If a fallback were attempted it would call _native_loads_ok; make that loud.
    monkeypatch.setattr(sl, "_native_loads_ok",
                        lambda: pytest.fail("must not load-test after a pin failure"))
    with pytest.raises(SystemExit):
        sl._provision_with_fallback("cuda", tmp_path, "deadbeef", True)


def test_selfcontained_provisioned_but_unloadable_is_reportable(monkeypatch, tmp_path):
    """vulkan/cpu are the universal fallbacks; if one provisions but will not
    load, that is an unexpected fault - raise (report-worthy), not exit-0."""
    _stub_provision(monkeypatch)
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: (False, "broken binary"))
    from localm.bugreport import LocalmError
    with pytest.raises(LocalmError) as ei:
        sl._provision_with_fallback("vulkan", tmp_path, None, False)
    assert "did not load" in ei.value.summary


# --------------------------- _native_loads_ok ----------------------------- #

def test_native_loads_ok_handles_subprocess_error(monkeypatch):
    import subprocess
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="python", timeout=120)
    monkeypatch.setattr(subprocess, "run", boom)
    ok, detail = sl._native_loads_ok()
    assert ok is False
    assert detail            # a non-empty, human-readable reason


# --------------------------- _clear_target -------------------------------- #

def test_clear_target_removes_files_keeps_subdirs(tmp_path):
    (tmp_path / "a.dll").write_bytes(b"x")
    (tmp_path / "b.so").write_bytes(b"y")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "keep.txt").write_bytes(b"z")
    sl._clear_target(tmp_path)
    assert not (tmp_path / "a.dll").exists()
    assert not (tmp_path / "b.so").exists()
    assert (sub / "keep.txt").exists()           # subdirectories are left alone


def test_clear_target_missing_dir_does_not_raise(tmp_path):
    sl._clear_target(tmp_path / "nope")          # must not raise


# --------------------------- _provision_backend (cuda) -------------------- #

def test_provision_backend_cuda_fetches_build_and_cudart(monkeypatch, tmp_path):
    monkeypatch.setattr(sl.sys, "platform", "win32")
    monkeypatch.setattr(sl, "_latest_tag", lambda: "b9999")
    build = {"name": "llama-cuda-12.4.zip", "browser_download_url": "https://b", "size": 1}
    cudart = {"name": "cudart-12.4.zip", "browser_download_url": "https://c", "size": 1}
    monkeypatch.setattr(sl, "_resolve_cuda_pair", lambda tag: (build, cudart))
    fetched = []
    monkeypatch.setattr(sl, "_fetch_and_place",
                        lambda url, target, sha256=None: fetched.append(url))
    sl._provision_backend("cuda", tmp_path, None, with_cudart=True)
    assert fetched == ["https://b", "https://c"]


def test_provision_backend_cuda_no_assets_uses_templated_url(monkeypatch, tmp_path):
    monkeypatch.setattr(sl.sys, "platform", "win32")
    monkeypatch.setattr(sl, "_latest_tag", lambda: "b9999")
    monkeypatch.setattr(sl, "_resolve_cuda_pair", lambda tag: (None, None))
    monkeypatch.setattr(sl, "_resolve_backend_url", lambda backend: "https://templated/cuda.zip")
    calls = []
    monkeypatch.setattr(sl, "_fetch_and_place",
                        lambda url, target, sha256=None: calls.append((url, sha256)))
    sl._provision_backend("cuda", tmp_path, "PIN", with_cudart=True)
    assert calls == [("https://templated/cuda.zip", "PIN")]   # build only, no cudart


# --------------------------- _warn_off_profile robustness ----------------- #

def test_warn_off_profile_survives_hwdetect_failure(monkeypatch):
    from localm import hwdetect
    def boom():
        raise RuntimeError("detect failed")
    monkeypatch.setattr(hwdetect, "detect", boom)
    sl._warn_off_profile("cuda")                 # must not raise
