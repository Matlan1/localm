# SPDX-License-Identifier: AGPL-3.0-or-later
"""An explicit llama.cpp backend choice is kept when a different backend has to
be installed: setup-llama records the choice and the reason, says so, and the
record survives a re-provision of the installed backend by `localm update`.
A cuda build that loads but cannot use the GPU does not count as cuda working.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from localm import setup_llama as sl


@pytest.fixture()
def box(tmp_path, monkeypatch):
    """A runtime lib dir in *tmp_path* and setup-llama with every download,
    hardware probe and native load replaced. ``box.loads`` and ``box.gpu`` are
    the per-backend answers of the two load probes; ``box.provisioned`` lists
    each backend provisioned, in order."""
    target = tmp_path / "lib"
    target.mkdir()
    (target / ".gitignore").write_text("*\n", encoding="utf-8")

    class Box:
        pass
    b = Box()
    b.target = target
    b.provisioned = []
    b.current = [None]
    b.loads = {}
    b.gpu = {}

    def fake_provision(backend, tgt, sha256, with_cudart, cuda_line=sl._CUDA_LINE, tag=None):
        b.provisioned.append(backend)
        b.current[0] = backend
        (tgt / sl._lib_name()).write_bytes(b"x")
        return "bTEST"

    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: target)
    monkeypatch.setattr(sl, "_provision_backend", fake_provision)
    monkeypatch.setattr(sl, "_install_runtime_wheel", lambda d: True)
    monkeypatch.setattr(sl, "_bundle_missing_native_deps", lambda t: None)
    monkeypatch.setattr(sl, "_native_loads_ok",
                        lambda: b.loads.get(b.current[0], (True, "")))
    monkeypatch.setattr(sl, "_native_gpu_loads_ok",
                        lambda: b.gpu.get(b.current[0], (True, "")))
    monkeypatch.setattr(sl, "_verify", lambda: None)
    monkeypatch.setattr(sl, "_ensure_importable", lambda: None)
    monkeypatch.setattr(sl, "_refresh_install_record", lambda t: None)
    monkeypatch.setattr(sl, "_warn_off_profile", lambda chosen: None)
    monkeypatch.setattr(sl, "_record_runtime_history", lambda backend, build: None)
    monkeypatch.setattr(sl, "nvidia_preflight",
                        lambda: sl.NvidiaInfo(present=True, gpu_name="RTX 4070",
                                              driver_version="552.22",
                                              cuda_capability="12.4"))
    monkeypatch.setattr(sl.sys.stdin, "isatty", lambda: False)
    return b


def _run(args):
    return CliRunner().invoke(sl.main, args)


def _flat(text: str) -> str:
    return " ".join(text.split())


_NO_GPU = (False, "the runtime loaded but registered no GPU device (registered: CPU)")


def test_cuda_that_cannot_use_the_gpu_is_not_recorded_as_working_cuda(box):
    box.gpu["cuda"] = _NO_GPU
    result = _run(["--backend", "cuda", "--force", "--yes"])
    assert result.exit_code == 0, result.output
    assert box.provisioned == ["cuda", "vulkan"]
    assert sl._provisioned_backend(box.target) == "vulkan"


def test_gui_path_fallback_keeps_the_cuda_choice_with_the_reason(box):
    box.gpu["cuda"] = _NO_GPU
    result = _run(["--backend", "cuda", "--force", "--yes"])
    assert result.exit_code == 0, result.output
    choice = sl.backend_choice()
    assert choice is not None
    assert choice["requested"] == "cuda"
    assert choice["installed"] == "vulkan"
    assert "registered no GPU device" in choice["reason"]
    out = _flat(result.output)
    assert "Your choice 'cuda' is kept" in out
    assert "--backend cuda --force" in out


def test_interactive_fallback_accepted_keeps_the_choice_it_promised_to_keep(box, monkeypatch):
    box.loads["cuda"] = (False, "OSError: could not load ggml-cuda.dll")
    monkeypatch.setattr(sl.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sl.click, "confirm", lambda *a, **k: True)
    result = _run(["--backend", "cuda", "--force"])
    assert result.exit_code == 0, result.output
    choice = sl.backend_choice()
    assert choice and choice["requested"] == "cuda"
    assert "ggml-cuda.dll" in choice["reason"]


def test_old_driver_with_yes_records_the_driver_reason(box, monkeypatch):
    monkeypatch.setattr(sl, "nvidia_preflight",
                        lambda: sl.NvidiaInfo(present=True, driver_version="460.0",
                                              cuda_capability="11.2"))
    result = _run(["--backend", "cuda", "--force", "--yes"])
    assert result.exit_code == 0, result.output
    assert box.provisioned == ["vulkan"]
    choice = sl.backend_choice()
    assert choice and choice["requested"] == "cuda"
    assert "460.0" in choice["reason"]


def test_choice_that_loads_clears_an_earlier_record(box):
    box.gpu["cuda"] = _NO_GPU
    assert _run(["--backend", "cuda", "--force", "--yes"]).exit_code == 0
    assert sl.backend_choice() is not None
    box.gpu["cuda"] = (True, "")
    result = _run(["--backend", "cuda", "--force", "--yes"])
    assert result.exit_code == 0, result.output
    assert sl._provisioned_backend(box.target) == "cuda"
    assert not (box.target / sl._CHOICE_MARKER).exists()
    assert sl.backend_choice() is None


def test_update_reprovision_of_the_fallback_keeps_the_choice(box):
    box.gpu["cuda"] = _NO_GPU
    assert _run(["--backend", "cuda", "--force", "--yes"]).exit_code == 0
    from localm import _apply_update
    argv = _apply_update.post_swap_command("runtime", backend="vulkan")
    args = argv[argv.index("setup-llama") + 1:]
    result = _run(args)
    assert result.exit_code == 0, result.output
    choice = sl.backend_choice()
    assert choice and choice["requested"] == "cuda" and choice["installed"] == "vulkan"


def test_explicit_vulkan_reprovision_without_keep_choice_drops_the_cuda_record(box):
    box.gpu["cuda"] = _NO_GPU
    assert _run(["--backend", "cuda", "--force", "--yes"]).exit_code == 0
    assert _run(["--backend", "vulkan", "--force", "--yes"]).exit_code == 0
    assert sl.backend_choice() is None


def test_asking_for_the_installed_backend_accepts_it_without_a_download(box):
    box.gpu["cuda"] = _NO_GPU
    assert _run(["--backend", "cuda", "--force", "--yes"]).exit_code == 0
    before = list(box.provisioned)
    result = _run(["--backend", "vulkan", "--yes"])
    assert result.exit_code == 0, result.output
    assert box.provisioned == before
    assert sl.backend_choice() is None
    assert "Keeping vulkan" in _flat(result.output)


def test_bare_setup_llama_keeps_the_record_and_says_so(box):
    box.gpu["cuda"] = _NO_GPU
    assert _run(["--backend", "cuda", "--force", "--yes"]).exit_code == 0
    result = _run(["--yes"])
    assert result.exit_code == 0, result.output
    assert sl.backend_choice() is not None
    assert "Your choice 'cuda' is not in use" in _flat(result.output)


def test_auto_records_no_choice(box, monkeypatch):
    monkeypatch.setattr(sl, "_auto_backend", lambda: "cuda")
    box.gpu["cuda"] = _NO_GPU
    result = _run(["--force", "--yes"])
    assert result.exit_code == 0, result.output
    assert sl._provisioned_backend(box.target) == "vulkan"
    assert not (box.target / sl._CHOICE_MARKER).exists()
    assert "auto chose vulkan" in _flat(result.output)


def test_from_dir_build_drops_the_record(box, tmp_path):
    box.gpu["cuda"] = _NO_GPU
    assert _run(["--backend", "cuda", "--force", "--yes"]).exit_code == 0
    build = tmp_path / "build"
    build.mkdir()
    (build / sl._lib_name()).write_bytes(b"own")
    result = _run(["--from", str(build), "--force", "--yes"])
    assert result.exit_code == 0, result.output
    assert not (box.target / sl._CHOICE_MARKER).exists()


def test_clear_target_keeps_the_choice_record(tmp_path):
    (tmp_path / sl._lib_name()).write_bytes(b"x")
    sl._record_backend_choice(tmp_path, "cuda", "why")
    assert sl._clear_target(tmp_path) == []
    assert not (tmp_path / sl._lib_name()).exists()
    assert sl._read_backend_choice(tmp_path) == {"requested": "cuda", "reason": "why"}


@pytest.mark.parametrize("content", [
    "", "not json", "[]", json.dumps({"requested": "nope"}),
    json.dumps({"requested": 5}), json.dumps({"reason": "x"}),
])
def test_unreadable_or_unknown_choice_record_reads_as_none(tmp_path, content):
    (tmp_path / sl._CHOICE_MARKER).write_text(content, encoding="utf-8")
    assert sl._read_backend_choice(tmp_path) is None


def test_choice_reason_is_one_bounded_line(tmp_path):
    sl._record_backend_choice(tmp_path, "cuda", "a\nb\r\n  c" + "x" * 5000)
    rec = sl._read_backend_choice(tmp_path)
    assert rec["reason"].startswith("a b c")
    assert len(rec["reason"]) == sl._CHOICE_REASON_MAX


def test_backend_choice_is_none_when_the_choice_is_installed(box):
    sl._record_provisioned_backend(box.target, "cuda")
    sl._record_backend_choice(box.target, "cuda", "stale")
    assert sl.backend_choice() is None


def test_backend_choice_is_none_when_nothing_is_installed(box):
    sl._record_backend_choice(box.target, "cuda", "why")
    assert sl.backend_choice() is None


# ------------------------- which runtime is loaded ------------------------- #

def _fake_runtime(d, backend_lib):
    d.mkdir(parents=True, exist_ok=True)
    (d / sl._lib_name()).write_bytes(b"x")
    (d / backend_lib).write_bytes(b"x")


def test_runtime_in_use_names_a_binary_dir_override(box, tmp_path, monkeypatch):
    sl._record_provisioned_backend(box.target, "cuda")
    other = tmp_path / "other"
    _fake_runtime(other, "ggml-vulkan.dll")
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr(_loader, "runtime_binary_dir", lambda: other)
    monkeypatch.delenv("LLAMA_CPP_LIB", raising=False)
    use = sl.runtime_in_use()
    assert use == {"dir": str(other), "source": "binary_dir", "backend": "vulkan"}


def test_runtime_in_use_names_an_env_override(box, tmp_path, monkeypatch):
    other = tmp_path / "env"
    _fake_runtime(other, "ggml-hip.dll")
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr(_loader, "runtime_binary_dir", lambda: other)
    monkeypatch.setenv("LLAMA_CPP_LIB", str(other / sl._lib_name()))
    assert sl.runtime_in_use()["source"] == "LLAMA_CPP_LIB"
    assert sl.runtime_in_use()["backend"] == "hip"


def test_runtime_in_use_reads_the_provisioned_marker(box, monkeypatch):
    sl._record_provisioned_backend(box.target, "amd-rocm")
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr(_loader, "runtime_binary_dir", lambda: box.target)
    assert sl.runtime_in_use() == {"dir": str(box.target), "source": "provisioned",
                                   "backend": "amd-rocm"}


def test_runtime_in_use_with_nothing_resolvable(monkeypatch):
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr(_loader, "runtime_binary_dir", lambda: None)
    assert sl.runtime_in_use() == {"dir": None, "source": None, "backend": None}


def test_setup_llama_warns_when_a_binary_dir_override_wins(box, tmp_path, monkeypatch,
                                                           capsys):
    other = tmp_path / "other"
    _fake_runtime(other, "ggml-vulkan.dll")
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr(_loader, "runtime_binary_dir", lambda: other)
    monkeypatch.delenv("LLAMA_CPP_LIB", raising=False)
    from localm.setup_llama import cli
    cli._warn_if_overridden()
    out = _flat(capsys.readouterr().out)
    assert "localm does not load this runtime" in out
    assert "binary_dir setting" in out and "(vulkan)" in out


def test_setup_llama_silent_when_the_provisioned_runtime_is_loaded(box, monkeypatch, capsys):
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr(_loader, "runtime_binary_dir", lambda: box.target)
    from localm.setup_llama import cli
    cli._warn_if_overridden()
    assert capsys.readouterr().out == ""


def test_backend_from_files(tmp_path):
    assert sl._backend_from_files(tmp_path) is None
    (tmp_path / "ggml-cpu-haswell.dll").write_bytes(b"x")
    assert sl._backend_from_files(tmp_path) == "cpu"
    (tmp_path / "ggml-cuda.dll").write_bytes(b"x")
    assert sl._backend_from_files(tmp_path) == "cuda"
    assert sl._backend_from_files(tmp_path / "missing") is None


def test_gpu_probe_detail_names_the_library_that_did_not_load(monkeypatch):
    from localm.setup_llama import load_probe
    import subprocess

    def fake_run(*a, **k):
        return subprocess.CompletedProcess(
            a, sl._PROBE_NO_GPU_DEVICE,
            stdout="CPU\nggml-cuda.dll did not load: Could not find module\n"
                   "nvcuda.dll is missing: the NVIDIA driver provides it\n", stderr="")
    monkeypatch.setattr(load_probe.subprocess, "run", fake_run)
    ok, detail = sl._native_gpu_loads_ok()
    assert not ok
    assert "registered no GPU device (registered: CPU)" in detail
    assert "ggml-cuda.dll did not load" in detail
    assert "nvcuda.dll is missing" in detail


def test_gpu_probe_detail_without_a_named_cause(monkeypatch):
    from localm.setup_llama import load_probe
    import subprocess
    monkeypatch.setattr(load_probe.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, sl._PROBE_NO_GPU_DEVICE, stdout="CPU", stderr=""))
    ok, detail = sl._native_gpu_loads_ok()
    assert not ok and "GPU driver libraries may be missing" in detail


# ------------------------- review follow-ups ------------------------------- #

def test_update_that_has_to_fall_back_records_the_choice(box):
    """cuda installed and in use, no record; the update's re-provision of cuda
    cannot use the GPU: the cuda choice is recorded, not silently replaced."""
    sl._record_provisioned_backend(box.target, "cuda")
    box.gpu["cuda"] = _NO_GPU
    from localm import _apply_update
    argv = _apply_update.post_swap_command("runtime", backend="cuda")
    result = _run(argv[argv.index("setup-llama") + 1:])
    assert result.exit_code == 0, result.output
    choice = sl.backend_choice()
    assert choice and choice["requested"] == "cuda" and choice["installed"] == "vulkan"


def test_update_of_the_chosen_backend_that_loads_writes_no_record(box):
    sl._record_provisioned_backend(box.target, "cuda")
    from localm import _apply_update
    argv = _apply_update.post_swap_command("runtime", backend="cuda")
    assert _run(argv[argv.index("setup-llama") + 1:]).exit_code == 0
    assert not (box.target / sl._CHOICE_MARKER).exists()


@pytest.mark.parametrize("version_args", [["--tag", "b1"], ["--rollback"]])
def test_a_build_change_of_the_installed_backend_keeps_the_record(box, monkeypatch,
                                                                  version_args):
    box.gpu["cuda"] = _NO_GPU
    assert _run(["--backend", "cuda", "--force", "--yes"]).exit_code == 0
    monkeypatch.setattr(sl, "_apply_version_request", lambda *a, **k: None)
    result = _run(["--backend", "vulkan", *version_args, "--yes"])
    assert result.exit_code == 0, result.output
    choice = sl.backend_choice()
    assert choice and choice["requested"] == "cuda"


def test_auto_on_an_old_driver_uses_vulkan_without_asking(box, monkeypatch):
    monkeypatch.setattr(sl, "_auto_backend", lambda: "cuda")
    monkeypatch.setattr(sl, "nvidia_preflight",
                        lambda: sl.NvidiaInfo(present=True, driver_version="460.0",
                                              cuda_capability="11.2"))
    monkeypatch.setattr(sl.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sl.click, "confirm",
                        lambda *a, **k: pytest.fail("auto must not ask"))
    result = _run(["--force"])
    assert result.exit_code == 0, result.output
    assert box.provisioned == ["vulkan"]
    assert not (box.target / sl._CHOICE_MARKER).exists()
    out = _flat(result.output)
    assert "choice is kept" not in out
    assert "auto chose vulkan" in out


def test_auto_fallback_does_not_claim_a_kept_choice(box, monkeypatch):
    monkeypatch.setattr(sl, "_auto_backend", lambda: "cuda")
    box.gpu["cuda"] = _NO_GPU
    result = _run(["--force", "--yes"])
    assert result.exit_code == 0, result.output
    assert "choice is kept" not in _flat(result.output)


def test_pinned_floor_build_must_register_a_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(sl, "tracks_latest", lambda: True)
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: (True, ""))
    monkeypatch.setattr(sl, "_native_gpu_loads_ok", lambda: _NO_GPU)
    from localm.setup_llama import provision
    monkeypatch.setattr(provision, "tracks_latest", lambda: True)
    ok, tag = provision._floor_at_pinned_tag(
        "cuda", True, "b99999", lambda backend, cudart, tag=None: None,
        f"{sl._ABI_REJECT_PREFIX}: drift")
    assert (ok, tag) == (False, None)
