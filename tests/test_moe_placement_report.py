# SPDX-License-Identifier: AGPL-3.0-or-later
"""MoE expert placement OBSERVABILITY: without a report from the worker,
n_cpu_moe gives no way to confirm it actually moved anything. This file covers:

  * _MODEL_BUFFER_RE / _CapturedStderr.model_buffers() - parsing llama.cpp's
    own "load_tensors: <backend> model buffer size = N MiB" report, the ONLY
    source for a per-backend weight-placement split (no llama.h API exists
    for it: no buffer-size introspection function is bound in
    llamacpp/_api.py, and none exists to bind).
  * _capture_stderr's temp-file lifetime: unlinking the temp file in the
    context manager's OWN finally block, before a caller outside the `with`
    ever reads it, makes BOTH the load-failure detail (captured.tail()) and
    the success-path placement report silently return ""/[] forever. The read
    happens inside the `with` block instead.
  * GgufWorker.load()'s meta dict carries weight_placement through the
    isolated-worker process boundary.
  * GgufBackend._load_native() prints a one-line placement summary, gated on
    n_cpu_moe>0 so an ordinary load stays quiet.
  * A REAL end-to-end load (@pytest.mark.integration) of a genuine tiny MoE
    GGUF through the full GgufBackend -> isolated worker pipeline, proving
    the printed numbers are real and non-trivial - a field-presence test
    alone proves plumbing, not truth.
"""

import os
import re
import struct
from unittest.mock import patch

import pytest

from localm.inference.backends.llamacpp import llama as llama_mod

_T_UINT32 = 4
_T_STRING = 8


def _s(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _dense_gguf(path):
    """A REAL minimal GGUF header for an architecture with NO expert_count
    key - the "no_experts" skip path _apply_cpu_moe reports via
    MOE_SKIP_MESSAGES."""
    kv = [
        ("general.architecture", _T_STRING, "llama"),
        ("llama.block_count", _T_UINT32, 32),
    ]
    out = [b"GGUF", struct.pack("<I", 3), struct.pack("<QQ", 0, len(kv))]
    for key, vtype, val in kv:
        out.append(_s(key))
        out.append(struct.pack("<I", vtype))
        if vtype == _T_STRING:
            out.append(_s(val))
        else:
            out.append(struct.pack("<I", val))
    path.write_bytes(b"".join(out))
    return path


# --------------------------------------------------------------------------- #
#  _MODEL_BUFFER_RE / _CapturedStderr.model_buffers()                          #
# --------------------------------------------------------------------------- #

# Captured verbatim from a real load of mradermacher/tiny-random-granite-moe-GGUF
# (Q8_0, n_cpu_moe=1) on this platform's ROCm build.
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
        """A ``\\S+`` backend-name group is a polynomial-ReDoS hazard here:
        captured native stderr is technically uncontrolled data, and a string
        with many "load_tensors:" restart points that never complete the rest
        of the pattern lets ``\\S+`` backtrack across the whole remaining text
        at EVERY restart point - O(n^2) total, and measurably quadratic in the
        number of repetitions. ``[A-Za-z0-9_]+`` has no character overlap with
        "load_tensors:"'s colon or the following literal's leading space, so a
        failed attempt terminates immediately with no backtracking. This test
        proves that property directly (wall-clock IS the security property
        here, not a proxy for one) rather than merely asserting the regex text
        changed."""
        import time
        p = tmp_path / "adversarial.log"
        # 20_000 repetitions with no valid completion anywhere; the pattern still
        # finishes in well under 0.1s.
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
#  _capture_stderr temp-file lifetime                                         #
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
        """_capture_stderr unlinks its temp file in its OWN finally, so a
        caller that reads captured.tail()/.model_buffers() AFTER the `with`
        block has exited must see ""/[] - never raise, and never silently look
        like a successful-but-empty read. llama.py's load call reads inside the
        block instead (see the _load_ctx restructuring in llama.py's
        __init__)."""
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
#  Load-failure detail reaches the raised error, at the layer a caller sees   #
# --------------------------------------------------------------------------- #

class TestLoadFailureDetailSurvivesUnlink:
    def test_null_model_ptr_with_captured_detail_is_included_in_the_error(self, monkeypatch):
        """Drives LlamaCpp.__init__'s actual restructured load block (not just
        _capture_stderr in isolation) with the native call faked to return
        NULL after writing a diagnosable reason to fd 2 - proving the
        RuntimeError's message carries that reason, not just the generic
        '(run with LOCALM_DEBUG=1 ...)' hint it would otherwise fall back
        to."""
        # Layout-agnostic here: only tensor_buft_overrides is touched, and it is
        # at the same offset in both llama_model_params layouts (guarded by
        # test_moe_cpu_placement.test_tensor_buft_overrides_offset_is_layout_agnostic).
        from localm.inference.backends.llamacpp._structs import (
            LlamaModelParamsV1 as LlamaModelParams)

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
#  ONE contiguous native-call scope covering llama_backend_init() and         #
#  _apply_cpu_moe, so neither fd 2 nor the debug-mode console mirror is       #
#  left unmanaged between them and no _dbg.info() output reaches the          #
#  shared terminal raw.                                                       #
# --------------------------------------------------------------------------- #

class TestMergedNativeCallScope:
    def _drive(self, monkeypatch, events, *, n_cpu_moe=0, verbose=False):
        import contextlib

        # Layout-agnostic here too (see TestLoadFailureDetailSurvivesUnlink's
        # own comment above) - only tensor_buft_overrides is touched.
        from localm.inference.backends.llamacpp._structs import (
            LlamaModelParamsV1 as LlamaModelParams)

        @contextlib.contextmanager
        def spy_capture():
            events.append("capture_enter")
            yield None
            events.append("capture_exit")

        @contextlib.contextmanager
        def spy_mirror():
            events.append("mirror_enter")
            yield
            events.append("mirror_exit")

        def _fake_backend_init():
            events.append("backend_init")

        def _fake_default_params():
            return LlamaModelParams()

        def _fake_apply_cpu_moe(mp, n, model_path):
            events.append("apply_cpu_moe")
            return None, None

        def _fake_load_model_from_file(path, params):
            events.append("load_model")
            return None   # NULL - a clean load "failure", same safe shape as
                           # TestLoadFailureDetailSurvivesUnlink above: stops
                           # __init__ right after this scope, before any
                           # native call needs a REAL model pointer.

        monkeypatch.setattr(llama_mod, "_capture_stderr", spy_capture)
        monkeypatch.setattr("localm.debuglog.suppress_console_mirror", spy_mirror)
        monkeypatch.setattr(llama_mod, "_apply_cpu_moe", _fake_apply_cpu_moe)
        monkeypatch.setattr(llama_mod.api, "llama_backend_init", _fake_backend_init)
        monkeypatch.setattr(llama_mod.api, "llama_model_default_params",
                            _fake_default_params)
        monkeypatch.setattr(llama_mod.api, "llama_load_model_from_file",
                            _fake_load_model_from_file)
        with patch("localm.discover.apply_main_gpu", lambda mp: None), \
             patch("localm.discover.apply_gpu_split", lambda mp, ratios_override=None: None):
            with pytest.raises(RuntimeError):
                llama_mod.LlamaCpp("bad.gguf", n_ctx=64, n_gpu_layers=99,
                                   verbose=verbose, n_cpu_moe=n_cpu_moe)

    def test_backend_init_and_load_share_one_scope_not_two(self, monkeypatch):
        """The regression this whole file guards: before the fix,
        llama_backend_init() ran inside its OWN _quiet_stderr() that had
        already exited by the time _capture_stderr's scope began - two
        separate enter/exit pairs with a gap between them, not one."""
        events = []
        self._drive(monkeypatch, events, n_cpu_moe=0)
        assert events == [
            "mirror_enter", "capture_enter",
            "backend_init", "load_model",
            "capture_exit", "mirror_exit",
        ], events

    def test_apply_cpu_moe_runs_inside_the_same_scope(self, monkeypatch):
        """The concrete bug: _apply_cpu_moe (and the _dbg.info call inside it
        - see TestMoeSkipReasonPrint above for what it can log) must run
        strictly BETWEEN the one mirror/capture enter and the one exit, never
        outside it."""
        events = []
        self._drive(monkeypatch, events, n_cpu_moe=2)
        assert events == [
            "mirror_enter", "capture_enter",
            "backend_init", "apply_cpu_moe", "load_model",
            "capture_exit", "mirror_exit",
        ], events

    def test_apply_cpu_moe_skipped_entirely_when_n_cpu_moe_is_zero(self, monkeypatch):
        """The default (off) load must not pay for or touch MoE placement at
        all - matches _apply_cpu_moe's own opt-in contract, unchanged by
        this scope merge."""
        events = []
        self._drive(monkeypatch, events, n_cpu_moe=0)
        assert "apply_cpu_moe" not in events

    def test_verbose_mode_skips_both_wraps(self, monkeypatch):
        """verbose=True means "let native output through unfiltered" - ALL
        of it, including via the console mirror - so neither wrap engages,
        exactly like the pre-merge code's own contextlib.nullcontext branch
        for both _quiet_stderr and _capture_stderr individually."""
        events = []
        self._drive(monkeypatch, events, n_cpu_moe=2, verbose=True)
        assert "mirror_enter" not in events
        assert "capture_enter" not in events
        # the underlying native/log calls still happen - only the two
        # wraps are skipped, not the load itself.
        assert "backend_init" in events
        assert "apply_cpu_moe" in events
        assert "load_model" in events


class TestConsoleMirrorGenuinelySilentDuringMoeLoad:
    """The end-to-end proof, not just "the wrap was entered": with a REAL
    console-mirror handler attached to the shared logger (the exact
    debug-mode state _add_console_handler produces) and n_cpu_moe set high
    enough that _apply_cpu_moe's own _dbg.info call fires for real, the
    mirror stream must receive NOTHING while LlamaCpp.__init__'s merged scope
    is open - this is the actual observable fix for the stuck "0:00:00"
    spinner line, not just a spy-order assertion."""

    def test_mirror_stream_receives_nothing_from_apply_cpu_moe(self, monkeypatch, tmp_path):
        import io
        import logging

        from localm import debuglog
        from localm.inference.backends.llamacpp._structs import (
            LlamaModelParamsV1 as LlamaModelParams)

        mirror_stream = io.StringIO()
        mirror = logging.StreamHandler(mirror_stream)
        mirror.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
        saved_level = debuglog.logger.level
        # suppress_console_mirror() identifies "the mirror" as the FIRST
        # non-FileHandler StreamHandler on the logger. A sibling test in the same
        # xdist worker can leave one attached, in which case the suppressor detaches
        # THAT one and this stand-in keeps receiving. Detach any pre-existing ones
        # so this stand-in IS unambiguously the mirror.
        preexisting = [h for h in list(debuglog.logger.handlers)
                       if isinstance(h, logging.StreamHandler)
                       and not isinstance(h, logging.FileHandler)]
        for _h in preexisting:
            debuglog.logger.removeHandler(_h)
        debuglog.logger.addHandler(mirror)
        debuglog.logger.setLevel(logging.DEBUG)
        try:
            def _fake_backend_init():
                pass

            def _fake_default_params():
                return LlamaModelParams()

            def _fake_load_model_from_file(path, params):
                return None

            # A real, parseable dense-model GGUF header (no expert_count key) so
            # _apply_cpu_moe's own real code (not mocked here) reaches its "no
            # experts" skip path and logs.
            model_path = _dense_gguf(tmp_path / "tiny.gguf")

            monkeypatch.setattr(llama_mod.api, "llama_backend_init", _fake_backend_init)
            monkeypatch.setattr(llama_mod.api, "llama_model_default_params",
                                _fake_default_params)
            monkeypatch.setattr(llama_mod.api, "llama_load_model_from_file",
                                _fake_load_model_from_file)
            with patch("localm.discover.apply_main_gpu", lambda mp: None), \
                 patch("localm.discover.apply_gpu_split", lambda mp, ratios_override=None: None):
                with pytest.raises(RuntimeError):
                    llama_mod.LlamaCpp(str(model_path), n_ctx=64, n_gpu_layers=99,
                                       verbose=False, n_cpu_moe=2)

            assert mirror_stream.getvalue() == "", (
                "a debug-mode log record reached the shared terminal mirror "
                "during the model-load span - this is the exact mechanism "
                f"that stranded a stuck 0:00:00 spinner line: "
                f"{mirror_stream.getvalue()!r}")
        finally:
            debuglog.logger.removeHandler(mirror)
            for _h in preexisting:
                debuglog.logger.addHandler(_h)
            debuglog.logger.setLevel(saved_level)


# --------------------------------------------------------------------------- #
#  GgufBackend._load_native() - the printed placement summary                 #
# --------------------------------------------------------------------------- #

def _backend(tmp_path, *, n_cpu_moe=0):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * 4096)
    from localm.inference.backends.gguf import GgufBackend
    return GgufBackend(str(f), n_ctx=512, n_cpu_moe=n_cpu_moe)


def _load(backend, *, weight_placement, moe_skip_reason=None):
    with patch("localm.discover.list_gpus", return_value=([], "ok")), \
         patch("localm.inference.backends.llamacpp._runner.ModelRunner."
               "spawn_and_load",
               return_value={"n_layers": 8, "kv_bytes_per_token": 0,
                             "supports_images": False,
                             "weight_placement": weight_placement,
                             "moe_skip_reason": moe_skip_reason}):
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
        """The default (off) load must stay quiet - the summary is opt-in
        observability, not new noise."""
        b = _backend(tmp_path, n_cpu_moe=0)
        _load(b, weight_placement=[
            {"backend": "ROCm0", "mib": 800.0, "is_ram": False},
        ])
        out = capsys.readouterr().out
        assert "moe placement" not in out

    def test_honest_when_n_cpu_moe_set_but_not_reported(self, tmp_path, capsys):
        """verbose mode or a parse miss reports [] - which must never be
        silently read as "0 bytes everywhere": say plainly that it was not
        reported."""
        b = _backend(tmp_path, n_cpu_moe=2)
        _load(b, weight_placement=[])
        out = capsys.readouterr().out
        assert "moe placement" in out
        assert "not reported" in out


class TestMoeSkipReasonPrint:
    """The skip message renders from the PARENT, from the moe_skip_reason fact
    carried through GgufWorker.load()'s metadata; _apply_cpu_moe printing it
    directly from the ISOLATED CHILD garbles the parent's spinner. The
    worker-metadata leg and the child's own silence are covered elsewhere;
    this class closes the loop - given the fact arrives, does the PARENT
    actually print the right message."""

    def test_no_experts_reason_prints_the_exact_message(self, tmp_path, capsys):
        b = _backend(tmp_path, n_cpu_moe=2)
        _load(b, weight_placement=[], moe_skip_reason="no_experts")
        out = capsys.readouterr().out
        assert "this model has no experts" in out
        assert "the setting does nothing here" in out

    def test_buffer_unresolved_reason_prints_the_exact_message(self, tmp_path, capsys):
        b = _backend(tmp_path, n_cpu_moe=2)
        _load(b, weight_placement=[], moe_skip_reason="buffer_unresolved")
        out = capsys.readouterr().out
        assert "the CPU buffer type could not be resolved" in out
        assert "experts were NOT moved" in out

    def test_no_skip_reason_prints_neither_message(self, tmp_path, capsys):
        """The success path (override applied) must not ALSO print a skip
        message - the two are mutually exclusive facts about the same load."""
        b = _backend(tmp_path, n_cpu_moe=2)
        _load(b, weight_placement=[
            {"backend": "ROCm0", "mib": 800.0, "is_ram": False},
        ], moe_skip_reason=None)
        out = capsys.readouterr().out
        assert "has no experts" not in out
        assert "could not be resolved" not in out

    def test_skip_reason_uses_the_shared_message_table(self, tmp_path, monkeypatch,
                                                        capsys):
        """Regression lock for message drift: the parent must render from
        llama.MOE_SKIP_MESSAGES, the SAME table the child-side test
        (test_every_skip_reason_has_a_rendered_message) checks is complete -
        not a second, independently-typed copy of the strings that could
        silently diverge from it."""
        from localm.inference.backends.llamacpp import llama as llama_mod
        monkeypatch.setitem(llama_mod.MOE_SKIP_MESSAGES, "no_experts",
                            "[yellow]  n_cpu_moe:[/yellow] CANARY-MESSAGE")
        b = _backend(tmp_path, n_cpu_moe=2)
        _load(b, weight_placement=[], moe_skip_reason="no_experts")
        out = capsys.readouterr().out
        assert "CANARY-MESSAGE" in out, (
            "the parent did not read from the shared MOE_SKIP_MESSAGES table")

    def test_skipped_override_never_prints_a_placement_line(self, tmp_path, capsys):
        """The trap this class exists to close: a load can report a
        moe_skip_reason AND a non-empty weight_placement in the SAME
        metadata dict (the placement report is populated from every
        load_tensors line regardless of whether n_cpu_moe applied - see
        _worker.py's load() docstring). When the override was skipped, the
        placement numbers describe an ordinary, unrelated load and must
        never be printed alongside the skip message - a reader sees
        "n_cpu_moe: ... does nothing here" immediately followed by
        "moe placement: ... across N backend buffer(s) (n_cpu_moe=2)" and
        reasonably concludes the setting moved something. It did not run at
        all."""
        b = _backend(tmp_path, n_cpu_moe=2)
        _load(b, weight_placement=[
            {"backend": "ROCm0", "mib": 630.59, "is_ram": False},
            {"backend": "ROCm_Host", "mib": 7669.77, "is_ram": True},
        ], moe_skip_reason="no_experts")
        out = capsys.readouterr().out
        assert "this model has no experts" in out
        assert "moe placement" not in out, (
            "printed a placement summary for a load where the n_cpu_moe "
            "override never ran")


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
    vendor/build."""
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
