# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the Wan video generation module (video_gen/comfy.py)."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from localm.media import comfy_client
from localm.video_gen import comfy


class TestSnapFrames:
    """Wan requires 4k+1 frames; durations snap to the nearest valid count."""

    @pytest.mark.parametrize("seconds,fps,expected", [
        (5.0, 24, 121),     # the native length: 24*5=120 -> 121
        (5.0, 16, 81),      # Wan 2.1 style: 16*5=80 -> 81
        (1.0, 24, 25),
        (0.1, 24, 5),       # floor: at least 5 frames
        (10.0, 24, 241),
        (4.9, 24, 117),     # 117.6 -> nearest 4k+1 is 117
    ])
    def test_snapping(self, seconds, fps, expected):
        frames = comfy._snap_frames(seconds, fps)
        assert frames == expected
        assert (frames - 1) % 4 == 0

    def test_always_valid(self):
        for tenth in range(1, 200):
            frames = comfy._snap_frames(tenth / 10, 24)
            assert frames >= 5 and (frames - 1) % 4 == 0


class TestFailFast:
    def test_dead_comfy_errors_without_unloading_llm(self, tmp_path):
        """A dead server must not cost the user an LLM unload+reload."""
        unload_spy = MagicMock()
        with patch.object(comfy, "ensure_comfy",
                          return_value=(False, "ComfyUI is not reachable (stub)")), \
             patch.object(comfy, "_localm_unload", unload_spy):
            ok, msg = comfy.generate_video("a fox", tmp_path / "out.mp4")
        assert ok is False
        assert "not reachable" in msg
        unload_spy.assert_not_called()

    def test_bad_duration_rejected(self, tmp_path):
        ok, msg = comfy.generate_video("a fox", tmp_path / "out.mp4",
                                       seconds=0)
        assert ok is False and "Duration" in msg

    def test_bad_fps_rejected(self, tmp_path):
        ok, msg = comfy.generate_video("a fox", tmp_path / "out.mp4", fps=0)
        assert ok is False and "FPS" in msg

    def test_missing_input_image_rejected(self, tmp_path):
        with patch.object(comfy, "ensure_comfy",
                          return_value=(True, "ComfyUI is running.")), \
             patch.object(comfy, "_localm_unload"):
            ok, msg = comfy.generate_video(
                "a fox", tmp_path / "out.mp4",
                input_image=tmp_path / "nope.png")
        assert ok is False and "not found" in msg


def _fake_comfy(captured, *, output_key="images", history_outputs=None):
    """urlopen stub that records the queued workflow and plays a ComfyUI
    happy path: /prompt -> prompt_id, /history -> one finished output,
    /view -> fake video bytes."""
    outputs = history_outputs if history_outputs is not None else {
        "11": {output_key: [
            {"filename": "clip.mp4", "subfolder": "", "type": "output"}
        ], "animated": [True]}
    }

    def fake_urlopen(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        m = MagicMock()
        m.__enter__ = lambda s=m: s
        m.__exit__ = MagicMock(return_value=False)
        if "/prompt" in url:
            captured["workflow"] = json.loads(req.data.decode())["prompt"]
            m.read.return_value = b'{"prompt_id": "p1"}'
        elif "/history/" in url:
            m.read.return_value = json.dumps({"p1": {"outputs": outputs}}).encode()
        elif "/view" in url:
            m.read.return_value = b"FAKE-MP4-BYTES"
        else:
            m.read.return_value = b"{}"
        return m

    return fake_urlopen


class TestGenerateVideo:
    def _run(self, tmp_path, output_key="images", history_outputs=None,
             **kwargs):
        captured = {}
        fake = _fake_comfy(captured, output_key=output_key,
                           history_outputs=history_outputs)
        # Pin COMFY_OUTPUT_DIR to a sandbox so the post-download cleanup can
        # never consult the developer's real config (and delete a real file).
        comfy_out = tmp_path / "comfy_out"
        with patch.object(comfy, "ensure_comfy",
                          return_value=(True, "ComfyUI is running.")), \
             patch.object(comfy, "_localm_unload"), \
             patch.object(comfy_client, "_comfy_urlopen", fake), \
             patch.object(comfy.time, "sleep"), \
             patch.dict(os.environ,
                        {"COMFY_OUTPUT_DIR": str(comfy_out)}):
            ok, msg = comfy.generate_video(
                "a red fox running", tmp_path / "out.mp4", **kwargs)
        return ok, msg, captured

    def test_instance_token_reaches_localm_unload(self, tmp_path):
        """The video route's instance_token (its own attach token, for
        keyless-mode auth on the localm_url unload call) must reach
        _localm_unload, never be dropped between the plug.py route and
        comfy.py's call site."""
        captured = {}
        fake = _fake_comfy(captured)
        unload_spy = MagicMock(return_value=None)
        comfy_out = tmp_path / "comfy_out"
        with patch.object(comfy, "ensure_comfy",
                          return_value=(True, "ComfyUI is running.")), \
             patch.object(comfy, "_localm_unload", unload_spy), \
             patch.object(comfy_client, "_comfy_urlopen", fake), \
             patch.object(comfy.time, "sleep"), \
             patch.dict(os.environ, {"COMFY_OUTPUT_DIR": str(comfy_out)}):
            ok, msg = comfy.generate_video(
                "a red fox running", tmp_path / "out.mp4",
                localm_url="http://127.0.0.1:9/v1", instance_token="tok-abc")
        assert ok, msg
        unload_spy.assert_called_once_with("http://127.0.0.1:9/v1", "tok-abc")

    def test_happy_path_saves_clip_and_sidecar(self, tmp_path):
        ok, msg, captured = self._run(tmp_path, seconds=5.0, fps=24, seed=42)
        assert ok, msg
        out = tmp_path / "out.mp4"
        assert out.read_bytes() == b"FAKE-MP4-BYTES"
        wf = captured["workflow"]
        assert wf["4"]["inputs"]["text"] == "a red fox running"
        assert wf["6"]["inputs"]["length"] == 121
        assert wf["8"]["inputs"]["seed"] == 42
        assert wf["10"]["inputs"]["fps"] == 24
        sidecar = json.loads(
            out.with_suffix(".mp4.json").read_text(encoding="utf-8"))
        assert sidecar["prompt"] == "a red fox running"
        assert sidecar["seed"] == 42
        assert sidecar["frames"] == 121

    def test_privacy_skips_sidecar(self, tmp_path):
        ok, _, _ = self._run(tmp_path, write_sidecar=False)
        assert ok
        assert (tmp_path / "out.mp4").is_file()
        assert not (tmp_path / "out.mp4.json").exists()

    def test_overrides_injected(self, tmp_path):
        ok, _, captured = self._run(
            tmp_path, negative_prompt="text, watermark",
            width=640, height=368, steps=20, cfg=4.0)
        assert ok
        wf = captured["workflow"]
        assert wf["5"]["inputs"]["text"] == "text, watermark"
        assert wf["6"]["inputs"]["width"] == 640
        assert wf["6"]["inputs"]["height"] == 368
        assert wf["8"]["inputs"]["steps"] == 20
        assert wf["8"]["inputs"]["cfg"] == 4.0

    def test_defaults_keep_template_values(self, tmp_path):
        """None overrides must not clobber the template's tuned defaults."""
        ok, _, captured = self._run(tmp_path)
        assert ok
        wf = captured["workflow"]
        assert wf["5"]["inputs"]["text"]          # template negative intact
        assert wf["6"]["inputs"]["width"] == 1280  # native 720p - see docs
        assert wf["8"]["inputs"]["steps"] == 30

    @pytest.mark.parametrize("key", ["videos", "gifs", "images"])
    def test_all_save_node_output_keys_found(self, tmp_path, key):
        """SaveVideo reports under 'images', VHS under 'gifs' - the poller
        must find the clip regardless of which save node the graph uses."""
        ok, msg, _ = self._run(tmp_path, output_key=key)
        assert ok, msg

    def test_no_output_is_a_clear_error(self, tmp_path):
        ok, msg, _ = self._run(tmp_path, history_outputs={"11": {}})
        assert not ok
        assert "no video output" in msg

    def test_comfy_side_copy_deleted_when_delete_outputs(self, tmp_path):
        """With delete_outputs=True, the duplicate in ComfyUI's own output dir is
        removed when COMFY_OUTPUT_DIR / comfy_output_dir points at it."""
        comfy_out = tmp_path / "comfy_out"
        comfy_out.mkdir()
        orig = comfy_out / "clip.mp4"          # matches the fake history
        orig.write_bytes(b"comfy-side copy")
        ok, _, _ = self._run(tmp_path, delete_outputs=True)
        assert ok
        assert not orig.exists()
        assert (tmp_path / "out.mp4").is_file()   # local save unaffected

    def test_comfy_side_copy_kept_by_default(self, tmp_path):
        """By default (delete_outputs off) ComfyUI's own copy is LEFT in place -
        a user may run ComfyUI for its own gallery and want the file."""
        comfy_out = tmp_path / "comfy_out"
        comfy_out.mkdir()
        orig = comfy_out / "clip.mp4"
        orig.write_bytes(b"comfy-side copy")
        ok, _, _ = self._run(tmp_path)
        assert ok
        assert orig.exists()                       # kept by default
        assert (tmp_path / "out.mp4").is_file()

    def test_image_to_video_wires_start_image(self, tmp_path):
        img = tmp_path / "start.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        with patch.object(comfy, "_upload_image",
                          return_value="uploaded.png") as up:
            ok, _, captured = self._run(tmp_path, input_image=img)
        assert ok
        up.assert_called_once()
        wf = captured["workflow"]
        # The LoadImage node is injected with a fresh (role-resolved) id, not a
        # hardcoded one - locate it by class and confirm the latent's start_image
        # points at it.
        load_nodes = [(nid, n) for nid, n in wf.items()
                      if n.get("class_type") == "LoadImage"]
        assert len(load_nodes) == 1
        load_id, load_node = load_nodes[0]
        assert load_node["inputs"]["image"] == "uploaded.png"
        assert wf["6"]["inputs"]["start_image"] == [load_id, 0]


class TestWorkflowTemplate:
    def test_committed_template_has_expected_nodes(self):
        """Sanity-check the committed template is well-formed. Injection is by
        ROLE (resolve_sampler_roles + find_node_by_class), so a local override
        need not preserve these ids; the shipped template does."""
        wf = json.loads(
            (comfy._WORKFLOW_PATH).read_text(encoding="utf-8"))
        assert wf["4"]["class_type"] == "CLIPTextEncode"      # positive
        assert wf["5"]["class_type"] == "CLIPTextEncode"      # negative
        assert "length" in wf["6"]["inputs"]                  # video latent
        assert wf["8"]["class_type"] == "KSampler"
        assert "fps" in wf["10"]["inputs"]                    # CreateVideo
        # 4k+1 default and /16 dimensions
        assert (wf["6"]["inputs"]["length"] - 1) % 4 == 0
        assert wf["6"]["inputs"]["width"] % 16 == 0
        assert wf["6"]["inputs"]["height"] % 16 == 0
