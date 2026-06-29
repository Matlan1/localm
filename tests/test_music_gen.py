# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the ACE-Step music generation module (music_gen/comfy.py).

The generator had zero coverage; these mirror tests/test_video_gen.py and pin
the workflow-injection contract, the fail-fast-before-unload behaviour, the
instrumental default, privacy sidecar suppression, and output handling.
"""

import json
from unittest.mock import MagicMock, patch

from localm.music_gen import comfy


def _fake_comfy(captured, *, outputs=None):
    """urlopen stub playing a ComfyUI happy path: /prompt -> prompt_id,
    /history -> one finished audio output, /view -> fake FLAC bytes."""
    outputs = outputs if outputs is not None else {
        "8": {"audio": [
            {"filename": "track.flac", "subfolder": "", "type": "output"}
        ]}
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
            m.read.return_value = b"FAKE-FLAC-BYTES"
        else:
            m.read.return_value = b"{}"
        return m

    return fake_urlopen


class TestFailFast:
    def test_dead_comfy_errors_without_unloading_llm(self, tmp_path):
        """A dead server must not cost the user an LLM unload+reload."""
        unload_spy = MagicMock()
        with patch.object(comfy, "ensure_comfy",
                          return_value=(False, "ComfyUI is not reachable (stub)")), \
             patch.object(comfy, "_localm_unload", unload_spy):
            ok, msg = comfy.generate_music("synthwave", tmp_path / "out.flac")
        assert ok is False
        assert "not reachable" in msg
        unload_spy.assert_not_called()

    def test_bad_duration_rejected(self, tmp_path):
        ok, msg = comfy.generate_music("synthwave", tmp_path / "out.flac",
                                       duration_seconds=0)
        assert ok is False and "Duration" in msg


class TestGenerateMusic:
    def _run(self, tmp_path, outputs=None, **kwargs):
        captured = {}
        fake = _fake_comfy(captured, outputs=outputs)
        with patch.object(comfy, "ensure_comfy",
                          return_value=(True, "ComfyUI is running.")), \
             patch.object(comfy, "_localm_unload"), \
             patch.object(comfy.urllib.request, "urlopen", fake), \
             patch.object(comfy.time, "sleep"):
            ok, msg = comfy.generate_music(
                "synthwave, 80s", tmp_path / "out.flac", **kwargs)
        return ok, msg, captured

    def test_happy_path_saves_track_and_sidecar(self, tmp_path):
        ok, msg, captured = self._run(
            tmp_path, lyrics="[verse]\nhello", duration_seconds=90.0,
            seed=42, steps=40, cfg=4.5)
        assert ok, msg
        out = tmp_path / "out.flac"
        assert out.read_bytes() == b"FAKE-FLAC-BYTES"
        wf = captured["workflow"]
        assert wf["2"]["inputs"]["seconds"] == 90.0
        assert wf["3"]["inputs"]["tags"] == "synthwave, 80s"
        assert wf["3"]["inputs"]["lyrics"] == "[verse]\nhello"
        assert wf["6"]["inputs"]["seed"] == 42
        assert wf["6"]["inputs"]["steps"] == 40
        assert wf["6"]["inputs"]["cfg"] == 4.5
        sidecar = json.loads(
            out.with_suffix(".flac.json").read_text(encoding="utf-8"))
        assert sidecar["tags"] == "synthwave, 80s"
        assert sidecar["seed"] == 42
        assert sidecar["duration_seconds"] == 90.0

    def test_no_lyrics_uses_instrumental_marker(self, tmp_path):
        ok, _, captured = self._run(tmp_path, seed=1)
        assert ok
        assert captured["workflow"]["3"]["inputs"]["lyrics"] == "[inst]"

    def test_instrumental_sidecar_omits_lyrics(self, tmp_path):
        ok, _, _ = self._run(tmp_path, seed=1)
        assert ok
        sidecar = json.loads(
            (tmp_path / "out.flac.json").read_text(encoding="utf-8"))
        assert "lyrics" not in sidecar      # None values are dropped

    def test_privacy_skips_sidecar(self, tmp_path):
        ok, _, _ = self._run(tmp_path, seed=1, write_sidecar=False)
        assert ok
        assert (tmp_path / "out.flac").is_file()
        assert not (tmp_path / "out.flac.json").exists()

    def test_no_audio_output_is_a_clear_error(self, tmp_path):
        ok, msg, _ = self._run(tmp_path, outputs={"8": {}})
        assert not ok
        assert "no audio output" in msg

    def test_custom_checkpoint_injected(self, tmp_path):
        ok, _, captured = self._run(tmp_path, seed=1, ckpt_name="my_ace.safetensors")
        assert ok
        assert captured["workflow"]["1"]["inputs"]["ckpt_name"] == "my_ace.safetensors"


class TestWorkflowTemplate:
    def test_committed_template_has_expected_nodes(self):
        wf = json.loads(comfy._WORKFLOW_PATH.read_text(encoding="utf-8"))
        assert wf["2"]["class_type"] == "EmptyAceStepLatentAudio"
        assert "seconds" in wf["2"]["inputs"]
        assert wf["3"]["class_type"] == "TextEncodeAceStepAudio"
        assert wf["6"]["class_type"] == "KSampler"
        assert wf["8"]["class_type"] == "SaveAudio"


class TestAudioQualityWarning:
    """RULE 5: localm flags a band-limited track instead of reporting it as a
    clean success. Best-effort - never raises, skips if deps unavailable."""

    def _write(self, path, sig, sr=44100):
        import numpy as np
        import soundfile as sf
        sf.write(str(path), np.stack([sig, sig], axis=1).astype("float32"), sr)

    def test_flags_bandlimited(self, tmp_path):
        import numpy as np
        sr = 44100
        t = np.linspace(0, 3, 3 * sr, endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 200 * t)   # all energy far below 4 kHz
        p = tmp_path / "muffled.flac"
        self._write(p, sig, sr)
        warn = comfy._audio_quality_warning(p)
        assert warn and "band-limited" in warn

    def test_silent_on_fullband(self, tmp_path):
        import numpy as np
        sr = 44100
        rng = np.random.default_rng(0)
        sig = 0.3 * rng.standard_normal(3 * sr)   # broadband -> no warning
        p = tmp_path / "fullband.flac"
        self._write(p, sig, sr)
        assert comfy._audio_quality_warning(p) == ""

    def test_silent_audio_not_flagged(self, tmp_path):
        import numpy as np
        sr = 44100
        p = tmp_path / "silent.flac"
        self._write(p, np.zeros(3 * sr), sr)      # silence is a different fault
        assert comfy._audio_quality_warning(p) == ""

    def test_missing_file_is_best_effort(self, tmp_path):
        assert comfy._audio_quality_warning(tmp_path / "nope.flac") == ""
