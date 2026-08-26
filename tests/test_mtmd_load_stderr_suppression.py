# SPDX-License-Identifier: AGPL-3.0-or-later
"""LlamaCpp._load_mmproj stderr handling.

Before this fix, MtmdContext(mmproj_path, self._model_ptr) - unlike every other
native call in this file - ran with NO stderr redirect at all: not
_capture_stderr, not _quiet_stderr, not suppress_console_mirror. Real field logs
showed two consequences: (1) llama.cpp's own CLIP loader tensor dump (roughly
half of every vision-load log) landed straight on the user's console, mid-line
over the live load spinner, and (2) the module's own internal mtmd_input_text
ABI probe (_PROBE_CONTROL / _PROBE_EMBEDDED_NUL in mtmd.py) leaked its raw probe
bytes ("add_text: aaaa...") the same way. And on a genuine NULL mmproj open, the
real native reason was never captured, so the raised MtmdUnavailable's generic
message ("mmproj incompatible with this model or build") was all a caller - and
the debug log - ever saw.

_load_mmproj was pulled out of LlamaCpp.__init__ specifically so these are
testable without a real native model/GPU (same reasoning _stderr_ctx_for_generate
documents for its own extraction in llama.py). Each test here drives the REAL
method, with only MtmdContext itself swapped for a controllable stand-in that
mimics the native library's behavior of writing raw bytes straight to fd 2 - the
same trick test_moe_placement_report.py uses for llama_load_model_from_file."""

import logging
import os

from localm.inference.backends.llamacpp import mtmd as mtmd_mod
from tests._bare_llama import make_bare_llama


def _bare_instance():
    """A LlamaCpp instance with native __init__ bypassed - _load_mmproj only
    touches self._model_ptr and self._mtmd, both set here."""
    return make_bare_llama(_model_ptr=0xDEADBEEF)  # any truthy "pointer"; never dereferenced


def _redirect_fd2_to(path):
    """Point the REAL fd 2 at *path* and return the saved fd to restore it
    with. Standing in for "the user's terminal" - only what actually reaches
    THIS fd, after _load_mmproj's own internal redirect/restore cycle, counts
    as reaching the console."""
    target_fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    saved = os.dup(2)
    os.dup2(target_fd, 2)
    os.close(target_fd)
    return saved


def _restore_fd2(saved):
    os.dup2(saved, 2)
    os.close(saved)


class TestVisionLoadNeverReachesConsole:
    """Required proof 1: a vision load does not emit CLIP loader output to
    the console."""

    def test_clip_loader_dump_is_suppressed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "localm.debuglog.native_stderr_target", lambda: None)

        class _FakeMtmdPrintsLikeClip:
            def __init__(self, mmproj_path, model_ptr, gpu_index=0):
                # Verbatim shape from the field logs (issues/log_1.txt:1-314):
                # a real load prints one such line per tensor.
                os.write(2, b"clip_model_loader: tensor[0]: mm.0.weight\n")
                os.write(2, b"clip_model_loader: tensor[1]: mm.0.bias\n")
                os.write(2, b"load_hparams: has_vision_encoder = 1\n")
                self.supports_vision = True

            def free(self):
                pass

        monkeypatch.setattr(mtmd_mod, "MtmdContext", _FakeMtmdPrintsLikeClip)

        console = tmp_path / "console.log"
        saved = _redirect_fd2_to(console)
        try:
            inst = _bare_instance()
            inst._load_mmproj("fake.mmproj", verbose=False)
        finally:
            _restore_fd2(saved)

        assert inst._mtmd is not None, "the fake load should have succeeded"
        seen = console.read_text(encoding="utf-8", errors="replace")
        assert "clip_model_loader" not in seen, (
            f"CLIP loader output reached the console: {seen!r}")
        assert "has_vision_encoder" not in seen, (
            f"native load_hparams output reached the console: {seen!r}")

    def test_fires_control_unwrapped_call_does_leak(self, tmp_path, monkeypatch):
        """Proves the test above can actually fail: calling the SAME fake
        directly, with no suppression at all, must leak to fd 2 - otherwise
        test_clip_loader_dump_is_suppressed would pass even with the fix
        reverted, for a reason that has nothing to do with the fix."""
        class _FakeMtmdPrintsLikeClip:
            def __init__(self, mmproj_path, model_ptr, gpu_index=0):
                os.write(2, b"clip_model_loader: tensor[0]: mm.0.weight\n")
                self.supports_vision = True

        console = tmp_path / "console.log"
        saved = _redirect_fd2_to(console)
        try:
            _FakeMtmdPrintsLikeClip("fake.mmproj", 1)
        finally:
            _restore_fd2(saved)

        seen = console.read_text(encoding="utf-8", errors="replace")
        assert "clip_model_loader" in seen, (
            "the fake did not actually write to fd 2 - this test's whole "
            "premise (that suppression is what stops the leak, not the "
            "fake being inert) would be unproven")


class TestAbiProbeNeverReachesConsole:
    """Required proof 3: the internal mtmd_input_text ABI probe's output does
    not reach the console. Mirrors the real shape from mtmd.py's
    _PROBE_CONTROL/_PROBE_EMBEDDED_NUL, which llama.cpp echoes back via its
    own "add_text: ..." tokenizer trace (issues/log_1.txt:348-349)."""

    def test_probe_bytes_are_suppressed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "localm.debuglog.native_stderr_target", lambda: None)

        class _FakeMtmdRunsAbiProbe:
            def __init__(self, mmproj_path, model_ptr, gpu_index=0):
                # The exact leaked shape from the field log: the control probe
                # ("a"*256) followed by the embedded-NUL probe (prints empty
                # after the NUL truncates it).
                os.write(2, b"add_text: " + b"a" * 256 + b"\n")
                os.write(2, b"add_text: \n")
                self.supports_vision = True

            def free(self):
                pass

        monkeypatch.setattr(mtmd_mod, "MtmdContext", _FakeMtmdRunsAbiProbe)

        console = tmp_path / "console.log"
        saved = _redirect_fd2_to(console)
        try:
            inst = _bare_instance()
            inst._load_mmproj("fake.mmproj", verbose=False)
        finally:
            _restore_fd2(saved)

        seen = console.read_text(encoding="utf-8", errors="replace")
        assert "add_text:" not in seen, (
            f"the ABI probe's raw payload reached the console: {seen!r}")
        assert "a" * 256 not in seen, (
            f"the ABI probe control payload reached the console: {seen!r}")


class TestFailedMmprojOpenSurfacesNativeReason:
    """Required proof 2: a failed mmproj open surfaces the native reason
    rather than only the generic message."""

    def test_native_reason_reaches_the_warning(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(
            "localm.debuglog.native_stderr_target", lambda: None)

        class _FakeMtmdFailsWithRealReason:
            def __init__(self, mmproj_path, model_ptr, gpu_index=0):
                os.write(
                    2,
                    b"clip_model_loader: unknown projector type: 'nope'\n"
                    b"clip_init: failed to load vision model\n")
                raise mtmd_mod.MtmdUnavailable(
                    f"mtmd_init_from_file returned NULL for {mmproj_path} "
                    "(mmproj incompatible with this model or build)")

        monkeypatch.setattr(mtmd_mod, "MtmdContext", _FakeMtmdFailsWithRealReason)

        inst = _bare_instance()
        with caplog.at_level(logging.WARNING, logger="localm"):
            inst._load_mmproj("fake.mmproj", verbose=False)

        assert inst._mtmd is None
        assert "unknown projector type: 'nope'" in caplog.text, (
            f"the native failure reason did not reach the log - only the "
            f"generic message survived: {caplog.text!r}")
        assert "clip_init: failed to load vision model" in caplog.text

    def test_generic_message_alone_is_not_good_enough(
            self, tmp_path, monkeypatch, caplog):
        """Fires-control for the assertion shape above: a fake that raises
        the SAME generic exception but writes NOTHING to fd 2 (simulating a
        build with no diagnosable native reason) must NOT fabricate detail -
        proving the test above is checking real captured text, not a fixed
        string this method always appends."""
        monkeypatch.setattr(
            "localm.debuglog.native_stderr_target", lambda: None)

        class _FakeMtmdFailsSilently:
            def __init__(self, mmproj_path, model_ptr, gpu_index=0):
                raise mtmd_mod.MtmdUnavailable(
                    f"mtmd_init_from_file returned NULL for {mmproj_path} "
                    "(mmproj incompatible with this model or build)")

        monkeypatch.setattr(mtmd_mod, "MtmdContext", _FakeMtmdFailsSilently)

        inst = _bare_instance()
        with caplog.at_level(logging.WARNING, logger="localm"):
            inst._load_mmproj("fake.mmproj", verbose=False)

        assert inst._mtmd is None
        assert "mmproj incompatible with this model or build" in caplog.text
        assert "unknown projector type" not in caplog.text

    def test_warning_level_not_debug(self, tmp_path, monkeypatch, caplog):
        """A vision model silently dropping to text-only is a real capability
        loss (AGENTS.md rule 5) and must reach a level a user actually sees,
        not only LOCALM_DEBUG=1's debug log."""
        monkeypatch.setattr(
            "localm.debuglog.native_stderr_target", lambda: None)

        class _FakeMtmdFails:
            def __init__(self, mmproj_path, model_ptr, gpu_index=0):
                raise mtmd_mod.MtmdUnavailable("NULL for fake.mmproj")

        monkeypatch.setattr(mtmd_mod, "MtmdContext", _FakeMtmdFails)

        inst = _bare_instance()
        with caplog.at_level(logging.DEBUG, logger="localm"):
            inst._load_mmproj("fake.mmproj", verbose=False)

        matching = [r for r in caplog.records if "mmproj load failed" in r.message]
        assert matching, f"no 'mmproj load failed' record at all: {caplog.text!r}"
        assert matching[0].levelno == logging.WARNING, (
            f"expected WARNING, got {logging.getLevelName(matching[0].levelno)} - "
            "a debug-only record is invisible without LOCALM_DEBUG=1")


class TestVerboseModeSkipsTheWrap:
    """verbose=True means "let native output through unfiltered" everywhere
    else in this file (_stderr_ctx_for_generate, the main model load's
    _load_ctx/_mirror_ctx) - _load_mmproj must honour the same contract
    rather than silently suppressing even the mode that explicitly asked not
    to be suppressed."""

    def test_verbose_true_does_not_redirect_fd2(self, tmp_path, monkeypatch):
        class _FakeMtmdPrints:
            def __init__(self, mmproj_path, model_ptr, gpu_index=0):
                os.write(2, b"clip_model_loader: tensor[0]: mm.0.weight\n")
                self.supports_vision = True

            def free(self):
                pass

        monkeypatch.setattr(mtmd_mod, "MtmdContext", _FakeMtmdPrints)

        console = tmp_path / "console.log"
        saved = _redirect_fd2_to(console)
        try:
            inst = _bare_instance()
            inst._load_mmproj("fake.mmproj", verbose=True)
        finally:
            _restore_fd2(saved)

        seen = console.read_text(encoding="utf-8", errors="replace")
        assert "clip_model_loader" in seen, (
            "verbose=True must let native mmproj output through, same as "
            "every other native call in this file")
