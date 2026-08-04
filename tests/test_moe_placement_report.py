# SPDX-License-Identifier: AGPL-3.0-or-later
"""MoE expert placement OBSERVABILITY: n_cpu_moe (test_moe_cpu_placement.py) had
no way to confirm it actually moved anything - the worker never reported
llama.cpp's own load placement back to the parent, and even the load-FAILURE
diagnostic was silently broken. This file covers:

  * _MODEL_BUFFER_RE / _CapturedStderr.model_buffers() - parsing llama.cpp's
    own "load_tensors: <backend> model buffer size = N MiB" report, the ONLY
    source for a per-backend weight-placement split (no llama.h API exists
    for it - verified: no buffer-size introspection function is bound in
    llamacpp/_api.py, and none exists to bind).
  * _capture_stderr's temp-file lifetime: a real, pre-existing bug found
    while building this - the temp file was unlinked in the context
    manager's OWN finally block, before the caller (outside the `with`)
    ever read it, so BOTH the existing load-failure detail (captured.tail())
    and this file's new success-path placement report would silently return
    ""/[] forever. Verified live against a real load-failure RuntimeError
    before the fix (message carried none of the native reason) and after
    (the real "invalid magic characters" detail appeared). Fixed by reading
    inside the `with` block instead of after it exits.
  * GgufWorker.load()'s meta dict carries weight_placement through the
    isolated-worker process boundary (see test_gguf_worker.py for the
    worker-level wiring test).
  * GgufBackend._load_native() prints a one-line placement summary, gated on
    n_cpu_moe>0 so an ordinary load stays as quiet as before.
  * A REAL end-to-end load (@pytest.mark.integration) of a genuine tiny MoE
    GGUF through the full GgufBackend -> isolated worker pipeline, proving
    the printed numbers are real and non-trivial - a field-presence test
    alone proves plumbing, not truth.
"""

import os
import re
from unittest.mock import patch

import pytest

from localm.inference.backends.llamacpp import llama as llama_mod


# --------------------------------------------------------------------------- #
#  _MODEL_BUFFER_RE / _CapturedStderr.model_buffers()                          #
# --------------------------------------------------------------------------- #

# Captured verbatim from a real load of mradermacher/tiny-random-granite-moe-GGUF
# (Q8_0, n_cpu_moe=1) on this platform's ROCm build - see dev-notes for the full
# capture. Confirms the regex against genuine llama.cpp output, not a guess.
_REAL_LOAD_TENSORS_EXCERPT = """\
load_tensors: loading model tensors, this can take a while... (mmap = false, direct_io = false)
load_tensors: layer   0 assigned to device ROCm0, is_swa = 0
load_tensors: offloading output layer to GPU
load_tensors: offloading 5 repeating layers to GPU
load_tensors: offloaded 7/7 layers to GPU
load_tensors:        ROCm0 model buffer size =     3.35 MiB
load_tensors:    ROCm_Host model buffer size =     3.20 MiB
"""


class TestModelBufferParsing:
    def test_parses_real_captured_output(self, tmp_path):
        p = tmp_path / "captured.log"
        p.write_text(_REAL_LOAD_TENSORS_EXCERPT, encoding="utf-8")
        buffers = llama_mod._CapturedStderr(str(p)).model_buffers()
        assert buffers == [
            {"backend": "ROCm0", "mib": 3.35, "is_ram": False},
            {"backend": "ROCm_Host", "mib": 3.20, "is_ram": True},
        ]

    @pytest.mark.parametrize("name,is_ram", [
        ("CPU", True),
        ("CPU_Mapped", True),
        ("CPU_REPACK", True),
        ("ROCm_Host", True),
        ("CUDA_Host", True),
        ("Vulkan_Host", True),
        ("ROCm0", False),
        ("CUDA0", False),
        ("Vulkan0", False),
        ("Metal", False),
    ])
    def test_classifies_ram_vs_vram_backends(self, tmp_path, name, is_ram):
        p = tmp_path / "captured.log"
        p.write_text(f"load_tensors:  {name} model buffer size =   1.00 MiB\n",
                      encoding="utf-8")
        buffers = llama_mod._CapturedStderr(str(p)).model_buffers()
        assert buffers == [{"backend": name, "mib": 1.0, "is_ram": is_ram}]

    def test_no_matching_lines_returns_empty_list(self, tmp_path):
        p = tmp_path / "captured.log"
        p.write_text("nothing relevant in here\n", encoding="utf-8")
        assert llama_mod._CapturedStderr(str(p)).model_buffers() == []

    def test_missing_file_returns_empty_list_not_raise(self, tmp_path):
        missing = tmp_path / "does-not-exist.log"
        assert llama_mod._CapturedStderr(str(missing)).model_buffers() == []

    def test_adversarial_input_stays_linear_time(self, tmp_path):
        """CodeQL (py/polynomial-redos, PR #1007) correctly flagged an earlier
        version of _MODEL_BUFFER_RE that used \\S+ for the backend-name group:
        captured native stderr is technically uncontrolled data, and a string
        with many "load_tensors:" restart points that never complete the rest
        of the pattern let \\S+ backtrack across the whole remaining text at
        EVERY restart point - O(n^2) total. Measured live before the fix:
        0.019s/0.081s/0.330s/1.140s for n=500/1000/2000/4000 repetitions (a
        clean quadratic curve, up to 1853x slower than the fixed version at
        n=4000). [A-Za-z0-9_]+ has no character overlap with "load_tensors:"'s
        colon or the following literal's leading space, so a failed attempt
        terminates immediately with no backtracking - this test proves that
        property directly (wall-clock IS the security property here, not a
        proxy for one) rather than merely asserting the regex text changed."""
        import time
        p = tmp_path / "adversarial.log"
        # 20_000 repetitions, no valid completion anywhere - extrapolating the
        # measured quadratic curve above, the pre-fix \S+ pattern would take
        # roughly 25-30s here; the fixed pattern finishes in well under 0.1s.
        p.write_text("load_tensors:" * 20_000 + "!", encoding="utf-8")
        start = time.perf_counter()
        result = llama_mod._CapturedStderr(str(p)).model_buffers()
        elapsed = time.perf_counter() - start
        assert result == []
        assert elapsed < 2.0, (
            f"model_buffers() took {elapsed:.2f}s on adversarial input - "
            "this is the exact shape of the py/polynomial-redos regression "
            "CodeQL flagged on the earlier \\S+ version of _MODEL_BUFFER_RE")


# --------------------------------------------------------------------------- #
#  _capture_stderr temp-file lifetime (the bug found while building this)      #
# --------------------------------------------------------------------------- #

class TestCaptureStderrLifetime:
    def test_reading_inside_the_block_sees_the_written_text(self):
        """The correct usage - .tail()/.model_buffers() called INSIDE the
        `with` block, before the finally clause's unlink fires."""
        with llama_mod._capture_stderr() as captured:
            os.write(2, b"load_tensors: ROCm0 model buffer size =   9.00 MiB\n")
            tail = captured.tail()
            buffers = captured.model_buffers()
        assert "ROCm0" in tail
        assert buffers == [{"backend": "ROCm0", "mib": 9.0, "is_ram": False}]

    def test_reading_after_the_block_exits_is_empty_not_stale(self):
        """Regression lock for the bug this file's module docstring describes:
        _capture_stderr unlinks its temp file in its OWN finally, so a caller
        that reads captured.tail()/.model_buffers() AFTER the `with` block has
        exited must see ""/[] - never raise, and never silently look like a
        successful-but-empty read. This is exactly the shape llama.py's load
        call used to have (reads happened after the block) and no longer does
        (see the _load_ctx restructuring in llama.py's __init__)."""
        with llama_mod._capture_stderr() as captured:
            os.write(2, b"load_tensors: ROCm0 model buffer size =   9.00 MiB\n")
        assert captured.tail() == ""
        assert captured.model_buffers() == []

    def test_debug_mode_tees_captured_text_into_the_debug_log(self, tmp_path):
        """The other half of the fix: when debug mode is on, the captured text
        must ALSO reach the debug log file before the temp file is removed -
        matching _quiet_stderr's "debug mode sees the native stream" contract
        at its other call sites. _capture_stderr's own docstring notes this
        load span is the one _quiet_stderr does not cover, which is why
        without this the load's own native report was invisible even under
        LOCALM_DEBUG=1."""
        log_path = tmp_path / "debug.log"
        log_path.write_bytes(b"")

        def _fake_target():
            return os.open(str(log_path), os.O_WRONLY | os.O_APPEND)

        with patch("localm.debuglog.native_stderr_target", side_effect=_fake_target):
            with llama_mod._capture_stderr():
                os.write(2, b"load_tensors: ROCm0 model buffer size =   9.00 MiB\n")
        assert "ROCm0 model buffer size" in log_path.read_text(encoding="utf-8")

    def test_debug_mode_off_does_not_touch_the_log(self):
        """No debug log target (the normal case) must not attempt anything
        beyond the existing temp-file capture/cleanup - never raise."""
        with patch("localm.debuglog.native_stderr_target", return_value=None):
            with llama_mod._capture_stderr():
                os.write(2, b"hello\n")   # must not raise


# --------------------------------------------------------------------------- #
#  Load-failure detail actually reaches the raised error (the bug's other     #
#  visible symptom, at the layer a real caller sees)                          #
# --------------------------------------------------------------------------- #

class TestLoadFailureDetailSurvivesUnlink:
    def test_null_model_ptr_with_captured_detail_is_included_in_the_error(self, monkeypatch):
        """Drives LlamaCpp.__init__'s actual restructured load block (not just
        _capture_stderr in isolation) with the native call faked to return
        NULL after writing a diagnosable reason to fd 2 - proving the
        RuntimeError's message carries that reason, not just the generic
        '(run with LOCALM_DEBUG=1 ...)' hint the pre-fix code always fell
        back to (verified live against this exact code path: a corrupted
        GGUF's RuntimeError contained zero native detail before this fix)."""
        from localm.inference.backends.llamacpp._structs import LlamaModelParams

        def _fake_backend_init():
            pass

        def _fake_default_params():
            return LlamaModelParams()

        def _fake_load_model_from_file(path, params):
            os.write(2, b"gguf_init_from_reader: invalid magic characters: "
                        b"'NOPE', expected 'GGUF'\n")
            return None   # NULL - a real load failure

        monkeypatch.setattr(llama_mod.api, "llama_backend_init", _fake_backend_init)
        monkeypatch.setattr(llama_mod.api, "llama_model_default_params",
                            _fake_default_params)
        monkeypatch.setattr(llama_mod.api, "llama_load_model_from_file",
                            _fake_load_model_from_file)
        with patch("localm.discover.apply_main_gpu", lambda mp: None), \
             patch("localm.discover.apply_gpu_split", lambda mp, ratios_override=None: None):
            with pytest.raises(RuntimeError) as exc_info:
                llama_mod.LlamaCpp("bad.gguf", n_ctx=64, n_gpu_layers=99,
                                   verbose=False)
        message = str(exc_info.value)
        assert "invalid magic characters" in message, (
            f"the native failure reason did not reach the raised error - "
            f"this is exactly the unlink-before-read bug: {message!r}")
        assert "run with LOCALM_DEBUG=1" not in message, (
            "a real captured reason exists but the generic fallback hint "
            "was shown anyway")


# --------------------------------------------------------------------------- #
#  GgufBackend._load_native() - the printed placement summary                 #
# --------------------------------------------------------------------------- #

def _backend(tmp_path, *, n_cpu_moe=0):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * 4096)
    from localm.inference.backends.gguf import GgufBackend
    return GgufBackend(str(f), n_ctx=512, n_cpu_moe=n_cpu_moe)


def _load(backend, *, weight_placement):
    with patch("localm.discover.list_gpus", return_value=([], "ok")), \
         patch("localm.inference.backends.llamacpp._runner.ModelRunner."
               "spawn_and_load",
               return_value={"n_layers": 8, "kv_bytes_per_token": 0,
                             "supports_images": False,
                             "weight_placement": weight_placement}):
        backend._load_native()


class TestPlacementSummaryPrint:
    def test_prints_summary_when_n_cpu_moe_set_and_reported(self, tmp_path, capsys):
        b = _backend(tmp_path, n_cpu_moe=2)
        _load(b, weight_placement=[
            {"backend": "ROCm0", "mib": 800.0, "is_ram": False},
            {"backend": "ROCm_Host", "mib": 200.0, "is_ram": True},
        ])
        out = capsys.readouterr().out
        assert "moe placement" in out
        assert "200.00 MiB system RAM" in out
        assert "800.00 MiB VRAM" in out
        assert "n_cpu_moe=2" in out

    def test_silent_when_n_cpu_moe_is_off(self, tmp_path, capsys):
        """The default (off) load must stay exactly as quiet as before this
        change - the summary is opt-in observability, not new noise."""
        b = _backend(tmp_path, n_cpu_moe=0)
        _load(b, weight_placement=[
            {"backend": "ROCm0", "mib": 800.0, "is_ram": False},
        ])
        out = capsys.readouterr().out
        assert "moe placement" not in out

    def test_honest_when_n_cpu_moe_set_but_not_reported(self, tmp_path, capsys):
        """verbose mode or a parse miss reports [] - which must never be
        silently read as "0 bytes everywhere" (AGENTS.md rule 5): say
        plainly that it was not reported."""
        b = _backend(tmp_path, n_cpu_moe=2)
        _load(b, weight_placement=[])
        out = capsys.readouterr().out
        assert "moe placement" in out
        assert "not reported" in out


# --------------------------------------------------------------------------- #
#  REAL end-to-end: does the reported placement match what llama.cpp did?     #
# --------------------------------------------------------------------------- #

_MOE_REPO = "mradermacher/tiny-random-granite-moe-GGUF"
_MOE_FILE = "tiny-random-granite-moe.Q8_0.gguf"


@pytest.mark.integration
@pytest.mark.real_gguf
def test_real_moe_load_reports_nontrivial_placement(capsys):
    """No mocks: loads a genuine tiny MoE GGUF (a few MB) through the full
    GgufBackend -> isolated worker pipeline with n_cpu_moe set, and confirms
    the printed summary carries REAL, non-trivial numbers - not just that
    the line is present (a field-presence test alone proves plumbing, not
    truth). Checks the SHAPE (positive figures, a plausible backend count)
    rather than hardcoding exact floats, which would break on any other GPU
    vendor/build than the one this was verified on by hand."""
    try:
        from localm.inference.backends.llamacpp._loader import load_lib
        load_lib()
    except Exception as e:
        pytest.skip(f"native llama runtime not provisioned: {e}")

    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id=_MOE_REPO, filename=_MOE_FILE)
    except Exception as e:
        pytest.skip(f"could not fetch {_MOE_REPO}/{_MOE_FILE}: {e}")

    from localm.inference.backends.gguf import GgufBackend
    backend = GgufBackend(path, n_ctx=64, n_gpu_layers=99, n_cpu_moe=1)
    try:
        backend.load()
    except Exception as e:
        pytest.skip(f"MoE GGUF failed to load on this machine: {e}")
    try:
        out = capsys.readouterr().out
        assert "not reported" not in out, (
            f"n_cpu_moe was requested but llama.cpp's own load report was "
            f"not captured/parsed - output was:\n{out}")
        m = re.search(
            r"moe placement: ([\d.]+) MiB system RAM / ([\d.]+) MiB VRAM "
            r"across (\d+) backend", out)
        assert m, f"placement line did not match the expected shape:\n{out}"
        ram_mib, vram_mib, n_buffers = (
            float(m.group(1)), float(m.group(2)), int(m.group(3)))
        assert n_buffers >= 1
        assert ram_mib > 0 or vram_mib > 0, (
            "both figures were exactly zero - almost certainly a parse "
            "failure silently reporting nothing rather than a genuine "
            "all-zero model")
    finally:
        backend.unload()
