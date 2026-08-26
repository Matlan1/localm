# SPDX-License-Identifier: AGPL-3.0-or-later
"""A ComfyUI node whose weights only partly apply (a LoRA key mismatch, missing
VAE/UNet keys, ...) is not an execution_error - ComfyUI logs a console warning
and the run still "succeeds". Covers the capture mechanism in comfy_client.py
(comfy_launch_log_path / comfy_console_tail_start /
comfy_console_warnings_since) and its wiring into generate_image()'s returned
message + reproducibility sidecar.

Three properties of that mechanism are pinned separately: comfy_launch_log_path()
is PER-INSTANCE, so independently-configured image/video/music instances do not
corrupt or misattribute each other's log; the guard checks process IDENTITY, not
just liveness, so a process that died and was relaunched under the same api_url
is not read as the same one; and comfy_console_checked is derived from whether
the read after polling actually happened, not from whether an offset was
captured before submission.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from localm.image_gen import comfy
from localm.media import comfy_client as cc
from localm.music_gen import comfy as music_comfy
from localm.video_gen import comfy as video_comfy


def _minimal_png(width: int, height: int) -> bytes:
    import struct
    import zlib
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = (struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data
            + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data)))
    return sig + ihdr


class _FakeProc:
    def __init__(self, pid=4321, alive=True):
        self.pid = pid
        self._alive = alive

    def poll(self):
        return None if self._alive else 0


URL = "http://127.0.0.1:8188"
URL_2 = "http://127.0.0.1:8288"  # a DIFFERENT localm-launched ComfyUI instance


@pytest.fixture
def spawned():
    """Register a fake localm-launched ComfyUI process so spawned_pid() /
    comfy_console_tail_start() treat it as ours, and drop the registration
    afterwards - _spawned_procs is module-level global state shared across
    the whole test session (see test_stopcomfy_2026_07_01.py), so a test
    that forgets to clean up leaks into whichever test runs next. Yields the
    url."""
    cc._remember_spawned(URL, _FakeProc(pid=4321))
    yield URL
    cc._take_spawned(URL)


class TestLaunchLogPathScoping:
    """The HIGH-severity finding: one shared log path corrupts/misattributes
    across independently self-launched ComfyUI instances (image/video/music
    can each point at a different api_url)."""

    def test_different_urls_get_different_paths(self):
        p1 = cc.comfy_launch_log_path(URL)
        p2 = cc.comfy_launch_log_path(URL_2)
        assert p1 != p2

    def test_same_url_is_deterministic(self):
        assert cc.comfy_launch_log_path(URL) == cc.comfy_launch_log_path(URL)

    def test_trailing_slash_does_not_change_the_path(self):
        assert cc.comfy_launch_log_path(URL) == cc.comfy_launch_log_path(URL + "/")


class TestConsoleTailStart:
    def test_none_when_not_localm_launched(self):
        # No _remember_spawned entry for this URL: nothing to tail.
        assert cc.comfy_console_tail_start("http://127.0.0.1:19999") is None

    def test_offset_and_pid_captured(self, spawned):
        log = cc.comfy_launch_log_path(spawned)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_bytes(b"already here\n")
        tail = cc.comfy_console_tail_start(spawned)
        assert tail.offset == len(b"already here\n")
        assert tail.pid == 4321

    def test_none_when_log_file_absent(self, spawned):
        # Spawned but nothing logged yet: .stat() raises FileNotFoundError and the
        # offset must come back None, not 0.
        assert not cc.comfy_launch_log_path(spawned).exists()
        assert cc.comfy_console_tail_start(spawned) is None


class TestConsoleWarningsSince:
    @staticmethod
    def _fresh_log(url):
        log = cc.comfy_launch_log_path(url)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_bytes(b"")

    def test_no_tail_short_circuits(self, spawned):
        assert cc.comfy_console_warnings_since(spawned, None) == (False, [])

    def test_not_localm_launched_short_circuits(self):
        # A real-looking tail token with no spawned_pid.
        fake_tail = cc.ComfyConsoleTail(offset=0, pid=999999)
        assert cc.comfy_console_warnings_since(
            "http://127.0.0.1:19999", fake_tail) == (False, [])

    def test_matches_known_pattern_with_count(self, spawned):
        self._fresh_log(spawned)
        tail = cc.comfy_console_tail_start(spawned)
        with open(cc.comfy_launch_log_path(spawned), "a", encoding="utf-8") as f:
            for _ in range(3):
                f.write("WARNING:root:lora key not loaded: lora_unet_x\n")
        checked, warnings = cc.comfy_console_warnings_since(spawned, tail)
        assert checked is True
        assert warnings == [
            "a LoRA patch key did not match the model and was skipped (x3)"]

    def test_ignores_content_before_the_offset(self, spawned):
        """The exact scenario the offset exists for: a PRIOR generation's own
        warning must not bleed into the NEXT generation's report."""
        log = cc.comfy_launch_log_path(spawned)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("lora key not loaded: pre-existing noise\n", encoding="utf-8")
        tail = cc.comfy_console_tail_start(spawned)
        with open(log, "a", encoding="utf-8") as f:
            f.write("nothing interesting here\n")
        assert cc.comfy_console_warnings_since(spawned, tail) == (True, [])

    def test_multiple_distinct_patterns(self, spawned):
        self._fresh_log(spawned)
        tail = cc.comfy_console_tail_start(spawned)
        with open(cc.comfy_launch_log_path(spawned), "a", encoding="utf-8") as f:
            f.write("lora key not loaded: a\n")
            f.write("Missing VAE keys ['x', 'y']\n")
            f.write("Missing VAE keys ['z']\n")
        checked, warnings = cc.comfy_console_warnings_since(spawned, tail)
        assert checked is True
        assert "a LoRA patch key did not match the model and was skipped" in warnings
        assert "some VAE weights were not found in the checkpoint (x2)" in warnings

    def test_pattern_must_not_straddle_two_lines(self, spawned):
        """The LOW-severity anchoring fix: a pattern split across a line
        break (an artifact of byte-offset slicing, not a real single-line
        console message) must not count as a match."""
        self._fresh_log(spawned)
        tail = cc.comfy_console_tail_start(spawned)
        with open(cc.comfy_launch_log_path(spawned), "a", encoding="utf-8") as f:
            f.write("lora key not loaded\n")  # not the unet missing: marker
            f.write("unet missing")            # split across the line break
            f.write("\n: on the next line, should not match\n")
        checked, warnings = cc.comfy_console_warnings_since(spawned, tail)
        assert checked is True
        assert warnings == [
            "a LoRA patch key did not match the model and was skipped"]

    def test_no_match_returns_empty_but_checked(self, spawned):
        """The fires-control for every test above: ordinary ComfyUI startup
        chatter must never be mistaken for a silent-partial-apply warning -
        but the read DID happen, so checked must be True."""
        self._fresh_log(spawned)
        tail = cc.comfy_console_tail_start(spawned)
        with open(cc.comfy_launch_log_path(spawned), "a", encoding="utf-8") as f:
            f.write("Total VRAM 24576 MB, total RAM 65536 MB\n")
            f.write("got prompt\n")
        assert cc.comfy_console_warnings_since(spawned, tail) == (True, [])

    def test_process_gone_before_check_yields_unchecked(self, spawned):
        """If the process we were tailing was stopped and never replaced,
        checked must be False - the window was never actually read."""
        self._fresh_log(spawned)
        tail = cc.comfy_console_tail_start(spawned)
        with open(cc.comfy_launch_log_path(spawned), "a", encoding="utf-8") as f:
            f.write("lora key not loaded: a\n")
        cc._take_spawned(spawned)
        assert cc.comfy_console_warnings_since(spawned, tail) == (False, [])

    def test_relaunch_with_new_pid_yields_unchecked_not_a_false_clean(self, spawned):
        """The ORIGINAL process dies and a DIFFERENT one is launched for the
        SAME api_url before the read happens (ensure_comfy's fresh "w" open
        truncates the log and re-registers a new pid under the same
        _spawned_procs[api_url] key). A liveness-only check ("is spawned_pid()
        not None") would pass here and either under-report (seek past a
        truncated file) or misattribute the new process's own output. The
        identity check (tail.pid == the CURRENT spawned_pid) reports unchecked
        instead."""
        self._fresh_log(spawned)
        tail = cc.comfy_console_tail_start(spawned)
        assert tail.pid == 4321
        with open(cc.comfy_launch_log_path(spawned), "a", encoding="utf-8") as f:
            f.write("lora key not loaded: a genuine warning about to be lost\n")
        # ensure_comfy relaunching for the same api_url: a fresh spawn registers a
        # new pid and truncates the log.
        cc._remember_spawned(spawned, _FakeProc(pid=9999))
        cc.comfy_launch_log_path(spawned).write_bytes(b"")  # the real "w" reopen
        assert cc.comfy_console_warnings_since(spawned, tail) == (False, [])
        cc._take_spawned(spawned)  # extra cleanup: this test re-registered mid-run

    def test_independent_instances_do_not_cross_contaminate(self, spawned):
        """Companion to TestLaunchLogPathScoping: two DIFFERENT api_urls
        (e.g. image on :8188, video on :8288) must read from - and only
        from - their own log file."""
        cc._remember_spawned(URL_2, _FakeProc(pid=5555))
        try:
            self._fresh_log(spawned)
            self._fresh_log(URL_2)
            tail_1 = cc.comfy_console_tail_start(spawned)
            tail_2 = cc.comfy_console_tail_start(URL_2)
            with open(cc.comfy_launch_log_path(spawned), "a", encoding="utf-8") as f:
                f.write("lora key not loaded: from instance 1\n")
            with open(cc.comfy_launch_log_path(URL_2), "a", encoding="utf-8") as f:
                f.write("Missing VAE keys ['from instance 2']\n")
            checked_1, warnings_1 = cc.comfy_console_warnings_since(spawned, tail_1)
            checked_2, warnings_2 = cc.comfy_console_warnings_since(URL_2, tail_2)
            assert checked_1 is True
            assert warnings_1 == [
                "a LoRA patch key did not match the model and was skipped"]
            assert checked_2 is True
            assert warnings_2 == [
                "some VAE weights were not found in the checkpoint"]
        finally:
            cc._take_spawned(URL_2)


class TestComfyConsoleWarningWiring:
    """Drives generate_image() fully (same fake-urlopen pattern as
    TestSidecarContent in test_image_gen_robustness.py) to prove the
    capture mechanism is actually wired in, not just correct in isolation."""

    @staticmethod
    def _fake_urlopen(responses, log_path, extra_on_prompt=""):
        def fake_urlopen(req, timeout=None):
            url = req if isinstance(req, str) else req.full_url
            if "/prompt" in url and extra_on_prompt:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(extra_on_prompt)
            for key, body in responses.items():
                if key in url:
                    m = MagicMock()
                    m.read.return_value = body
                    m.__enter__ = lambda s=m: s
                    m.__exit__ = MagicMock(return_value=False)
                    return m
            raise AssertionError(f"unexpected url {url}")
        return fake_urlopen

    def test_warning_surfaces_in_message_and_sidecar(self, tmp_path, monkeypatch, spawned):
        monkeypatch.setattr(comfy, "workflow_path",
                            lambda: comfy._WORKFLOW_EXAMPLE_PATH)
        out = tmp_path / "art.png"
        log = cc.comfy_launch_log_path(spawned)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_bytes(b"")

        responses = {
            "/system_stats": b"{}",
            "/object_info": b"{}",
            "/prompt": b'{"prompt_id": "p1"}',
            "/history/p1": json.dumps({
                "p1": {"outputs": {"9": {"images": [
                    {"filename": "f.png", "subfolder": "", "type": "output"}
                ]}}}
            }).encode(),
            "/view": _minimal_png(8, 8),
        }
        fake_urlopen = self._fake_urlopen(
            responses, log,
            extra_on_prompt="WARNING:root:lora key not loaded: proj_lora1\n" * 5)

        with patch.object(cc, "_comfy_urlopen", side_effect=fake_urlopen), \
             patch.object(comfy, "_localm_unload"):
            ok, msg = comfy.generate_image(
                "a fox", out, seed=1, lora_name="bad.safetensors")

        assert ok, msg
        assert "ComfyUI's own console reported" in msg
        assert "LoRA patch key did not match" in msg
        assert "(x5)" in msg

        sidecar = json.loads((tmp_path / "art.png.json").read_text(encoding="utf-8"))
        assert "did not match the model" in sidecar["comfy_console_warning"]
        assert "(x5)" in sidecar["comfy_console_warning"]
        assert sidecar["comfy_console_checked"] is True
        # lora_name/strengths still record what was REQUESTED, unchanged.
        assert sidecar["lora_name"] == "bad.safetensors"

    def test_clean_run_never_warns(self, tmp_path, monkeypatch, spawned):
        """A successful, compatible generation must not warn or grow a sidecar
        field it never observed, and MUST still record
        comfy_console_checked=True, distinguishing "checked, found nothing"
        from the remote-ComfyUI "could not check" case below."""
        monkeypatch.setattr(comfy, "workflow_path",
                            lambda: comfy._WORKFLOW_EXAMPLE_PATH)
        out = tmp_path / "art2.png"
        log = cc.comfy_launch_log_path(spawned)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_bytes(b"")

        responses = {
            "/system_stats": b"{}",
            "/object_info": b"{}",
            "/prompt": b'{"prompt_id": "p2"}',
            "/history/p2": json.dumps({
                "p2": {"outputs": {"9": {"images": [
                    {"filename": "g.png", "subfolder": "", "type": "output"}
                ]}}}
            }).encode(),
            "/view": _minimal_png(8, 8),
        }
        fake_urlopen = self._fake_urlopen(responses, log, extra_on_prompt="")

        with patch.object(cc, "_comfy_urlopen", side_effect=fake_urlopen), \
             patch.object(comfy, "_localm_unload"):
            ok, msg = comfy.generate_image("a fox", out, seed=2)

        assert ok, msg
        assert "ComfyUI's own console reported" not in msg
        sidecar = json.loads((tmp_path / "art2.png.json").read_text(encoding="utf-8"))
        assert "comfy_console_warning" not in sidecar
        assert sidecar["comfy_console_checked"] is True

    def test_remote_comfy_never_claims_to_have_checked(self, tmp_path, monkeypatch):
        """No _remember_spawned registration at all - the 'localm did not
        launch this ComfyUI' case (already running, or remote/LAN per
        sanitize_comfy_url). No warning either, same as the clean-run case -
        but comfy_console_checked must read False here, NOT True: this run
        never looked, and the two silences have to be distinguishable."""
        monkeypatch.setattr(comfy, "workflow_path",
                            lambda: comfy._WORKFLOW_EXAMPLE_PATH)
        out = tmp_path / "art3.png"

        responses = {
            "/system_stats": b"{}",
            "/object_info": b"{}",
            "/prompt": b'{"prompt_id": "p3"}',
            "/history/p3": json.dumps({
                "p3": {"outputs": {"9": {"images": [
                    {"filename": "h.png", "subfolder": "", "type": "output"}
                ]}}}
            }).encode(),
            "/view": _minimal_png(8, 8),
        }

        def fake_urlopen(req, timeout=None):
            url = req if isinstance(req, str) else req.full_url
            for key, body in responses.items():
                if key in url:
                    m = MagicMock()
                    m.read.return_value = body
                    m.__enter__ = lambda s=m: s
                    m.__exit__ = MagicMock(return_value=False)
                    return m
            raise AssertionError(f"unexpected url {url}")

        assert cc.spawned_pid("http://127.0.0.1:8188") is None
        with patch.object(cc, "_comfy_urlopen", side_effect=fake_urlopen), \
             patch.object(comfy, "_localm_unload"):
            ok, msg = comfy.generate_image("a fox", out, seed=3)

        assert ok, msg
        assert "ComfyUI's own console reported" not in msg
        sidecar = json.loads((tmp_path / "art3.png.json").read_text(encoding="utf-8"))
        assert "comfy_console_warning" not in sidecar
        assert sidecar["comfy_console_checked"] is False


def _fake_music_urlopen(log_path, extra_on_prompt=""):
    """Mirrors TestComfyConsoleWarningWiring._fake_urlopen, shaped for
    generate_music's /history outputs (an 'audio' entry, not 'images')."""
    def fake_urlopen(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        if "/prompt" in url:
            if extra_on_prompt:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(extra_on_prompt)
            m = MagicMock()
            m.read.return_value = b'{"prompt_id": "p1"}'
            m.__enter__ = lambda s=m: s
            m.__exit__ = MagicMock(return_value=False)
            return m
        m = MagicMock()
        m.__enter__ = lambda s=m: s
        m.__exit__ = MagicMock(return_value=False)
        if "/history/" in url:
            m.read.return_value = json.dumps({"p1": {"outputs": {
                "8": {"audio": [{"filename": "t.flac", "subfolder": "", "type": "output"}]}
            }}}).encode()
        elif "/view" in url:
            m.read.return_value = b"FAKE-FLAC-BYTES"
        else:
            m.read.return_value = b"{}"
        return m
    return fake_urlopen


class TestMusicConsoleWarningWiring:
    """generate_music() consumes the SAME shared comfy_console_tail_start /
    comfy_console_warnings_since as generate_image(): image/music/video share
    the same submit -> poll -> fetch transport. Mirrors
    TestComfyConsoleWarningWiring above, adapted to music's inline sidecar
    dict (no separate _write_music_sidecar function)."""

    def test_warning_surfaces_in_message_and_sidecar(self, tmp_path, monkeypatch, spawned):
        out = tmp_path / "t.flac"
        log = cc.comfy_launch_log_path(spawned)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_bytes(b"")
        fake = _fake_music_urlopen(
            log, extra_on_prompt="WARNING:root:unet missing: some.weight\n" * 3)

        with patch.object(music_comfy, "ensure_comfy", return_value=(True, "up")), \
             patch.object(music_comfy, "_localm_unload"), \
             patch.object(cc, "_comfy_urlopen", fake), \
             patch.object(music_comfy.time, "sleep"):
            ok, msg = music_comfy.generate_music("synthwave", out, seed=1)

        assert ok, msg
        assert "ComfyUI's own console reported" in msg
        assert "UNet weights were not found" in msg
        assert "(x3)" in msg

        sidecar = json.loads(out.with_suffix(".flac.json").read_text(encoding="utf-8"))
        assert "were not found" in sidecar["comfy_console_warning"]
        assert sidecar["comfy_console_checked"] is True

    def test_clean_run_checked_true_no_warning(self, tmp_path, monkeypatch, spawned):
        out = tmp_path / "t2.flac"
        log = cc.comfy_launch_log_path(spawned)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_bytes(b"")
        fake = _fake_music_urlopen(log, extra_on_prompt="")

        with patch.object(music_comfy, "ensure_comfy", return_value=(True, "up")), \
             patch.object(music_comfy, "_localm_unload"), \
             patch.object(cc, "_comfy_urlopen", fake), \
             patch.object(music_comfy.time, "sleep"):
            ok, msg = music_comfy.generate_music("synthwave", out, seed=2)

        assert ok, msg
        assert "ComfyUI's own console reported" not in msg
        sidecar = json.loads(out.with_suffix(".flac.json").read_text(encoding="utf-8"))
        assert "comfy_console_warning" not in sidecar
        assert sidecar["comfy_console_checked"] is True

    def test_remote_comfy_never_claims_to_have_checked(self, tmp_path, monkeypatch):
        # No spawned/_remember_spawned registration - the "localm did not
        # launch this ComfyUI" case.
        out = tmp_path / "t3.flac"
        fake = _fake_music_urlopen(tmp_path / "unused.log", extra_on_prompt="")

        assert cc.spawned_pid(URL) is None
        with patch.object(music_comfy, "ensure_comfy", return_value=(True, "up")), \
             patch.object(music_comfy, "_localm_unload"), \
             patch.object(cc, "_comfy_urlopen", fake), \
             patch.object(music_comfy.time, "sleep"):
            ok, msg = music_comfy.generate_music("synthwave", out, seed=3)

        assert ok, msg
        assert "ComfyUI's own console reported" not in msg
        sidecar = json.loads(out.with_suffix(".flac.json").read_text(encoding="utf-8"))
        assert "comfy_console_warning" not in sidecar
        assert sidecar["comfy_console_checked"] is False


def _fake_video_urlopen(log_path, extra_on_prompt=""):
    """Mirrors _fake_music_urlopen, shaped for generate_video's /history
    outputs (an 'images' entry with animated=True, per test_video_gen.py)."""
    def fake_urlopen(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        if "/prompt" in url:
            if extra_on_prompt:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(extra_on_prompt)
            m = MagicMock()
            m.read.return_value = b'{"prompt_id": "p1"}'
            m.__enter__ = lambda s=m: s
            m.__exit__ = MagicMock(return_value=False)
            return m
        m = MagicMock()
        m.__enter__ = lambda s=m: s
        m.__exit__ = MagicMock(return_value=False)
        if "/history/" in url:
            m.read.return_value = json.dumps({"p1": {"outputs": {
                "11": {"images": [{"filename": "c.mp4", "subfolder": "", "type": "output"}],
                       "animated": [True]}
            }}}).encode()
        elif "/view" in url:
            m.read.return_value = b"FAKE-MP4-BYTES"
        else:
            m.read.return_value = b"{}"
        return m
    return fake_urlopen


class TestVideoConsoleWarningWiring:
    """generate_video() consumes the same shared machinery - see
    TestMusicConsoleWarningWiring's docstring. Also confirms video_gen's
    submit path reaches comfy_submit_prompt/comfy_poll_until_done rather than
    merely importing them."""

    def test_warning_surfaces_in_message_and_sidecar(self, tmp_path, monkeypatch, spawned):
        import os
        out = tmp_path / "c.mp4"
        log = cc.comfy_launch_log_path(spawned)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_bytes(b"")
        fake = _fake_video_urlopen(
            log, extra_on_prompt="WARNING:root:clip missing: text_projection.weight\n" * 2)

        comfy_out = tmp_path / "comfy_out"
        with patch.object(video_comfy, "ensure_comfy", return_value=(True, "up")), \
             patch.object(video_comfy, "_localm_unload"), \
             patch.object(cc, "_comfy_urlopen", fake), \
             patch.object(video_comfy.time, "sleep"), \
             patch.dict(os.environ, {"COMFY_OUTPUT_DIR": str(comfy_out)}):
            ok, msg = video_comfy.generate_video("a red fox running", out, seed=1)

        assert ok, msg
        assert "ComfyUI's own console reported" in msg
        assert "CLIP/text-encoder weights" in msg
        assert "(x2)" in msg

        sidecar = json.loads(out.with_suffix(".mp4.json").read_text(encoding="utf-8"))
        assert "were not found" in sidecar["comfy_console_warning"]
        assert sidecar["comfy_console_checked"] is True

    def test_clean_run_checked_true_no_warning(self, tmp_path, monkeypatch, spawned):
        import os
        out = tmp_path / "c2.mp4"
        log = cc.comfy_launch_log_path(spawned)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_bytes(b"")
        fake = _fake_video_urlopen(log, extra_on_prompt="")

        comfy_out = tmp_path / "comfy_out2"
        with patch.object(video_comfy, "ensure_comfy", return_value=(True, "up")), \
             patch.object(video_comfy, "_localm_unload"), \
             patch.object(cc, "_comfy_urlopen", fake), \
             patch.object(video_comfy.time, "sleep"), \
             patch.dict(os.environ, {"COMFY_OUTPUT_DIR": str(comfy_out)}):
            ok, msg = video_comfy.generate_video("a red fox running", out, seed=2)

        assert ok, msg
        assert "ComfyUI's own console reported" not in msg
        sidecar = json.loads(out.with_suffix(".mp4.json").read_text(encoding="utf-8"))
        assert "comfy_console_warning" not in sidecar
        assert sidecar["comfy_console_checked"] is True

    def test_remote_comfy_never_claims_to_have_checked(self, tmp_path, monkeypatch):
        import os
        out = tmp_path / "c3.mp4"
        fake = _fake_video_urlopen(tmp_path / "unused.log", extra_on_prompt="")

        assert cc.spawned_pid(URL) is None
        comfy_out = tmp_path / "comfy_out3"
        with patch.object(video_comfy, "ensure_comfy", return_value=(True, "up")), \
             patch.object(video_comfy, "_localm_unload"), \
             patch.object(cc, "_comfy_urlopen", fake), \
             patch.object(video_comfy.time, "sleep"), \
             patch.dict(os.environ, {"COMFY_OUTPUT_DIR": str(comfy_out)}):
            ok, msg = video_comfy.generate_video("a red fox running", out, seed=3)

        assert ok, msg
        assert "ComfyUI's own console reported" not in msg
        sidecar = json.loads(out.with_suffix(".mp4.json").read_text(encoding="utf-8"))
        assert "comfy_console_warning" not in sidecar
        assert sidecar["comfy_console_checked"] is False
