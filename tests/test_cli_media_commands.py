# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm image` / `localm music` / `localm video` (localm/cli/media.py) privacy
parity with the GUI/API plugin routes: privacy mode folds into BOTH
write_sidecar (prompt sidecar suppressed) AND delete_outputs (ComfyUI's own
on-disk output copy, which embeds the full prompt/workflow as PNG metadata per
localm/image_gen/comfy.py, plus any img2img source image, is removed).
"""

from contextlib import ExitStack
from unittest.mock import patch

from click.testing import CliRunner

from localm.audit import SessionMode
from localm.cli import main


def _invoke(args, generate_target, *, mode, default_api_url_target,
           extra_patches=()):
    """Invoke `localm <args>` with the generator patched, returning the kwargs
    it was called with (list, since the func can be called more than once -
    e.g. image_cmd's retry path) plus the click Result."""
    calls = []

    def fake_gen(*a, **kwargs):
        calls.append(kwargs)
        return (True, "ok")

    with ExitStack() as stack:
        stack.enter_context(patch(generate_target, fake_gen))
        stack.enter_context(patch(default_api_url_target,
                                  return_value="http://127.0.0.1:9999"))
        stack.enter_context(patch("localm.audit.effective_mode", return_value=mode))
        for p in extra_patches:
            stack.enter_context(p)
        result = CliRunner().invoke(main, args)
    return calls, result


class TestImageCmdPrivacy:
    def test_privacy_mode_deletes_outputs(self):
        calls, result = _invoke(
            ["image", "a cat"],
            "localm.image_gen.comfy.generate_image",
            mode=SessionMode.PRIVACY,
            default_api_url_target="localm.image_gen.comfy.default_api_url",
            extra_patches=[patch("localm.image_gen.comfy.free_comfy_vram")],
        )
        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert calls[0].get("write_sidecar") is False
        assert calls[0].get("delete_outputs") is True

    def test_log_mode_keeps_outputs(self):
        calls, result = _invoke(
            ["image", "a cat"],
            "localm.image_gen.comfy.generate_image",
            mode=SessionMode.LOG,
            default_api_url_target="localm.image_gen.comfy.default_api_url",
            extra_patches=[patch("localm.image_gen.comfy.free_comfy_vram")],
        )
        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert calls[0].get("write_sidecar") is True
        assert not calls[0].get("delete_outputs")

    def test_retry_path_also_deletes_outputs_in_privacy_mode(self):
        """image_cmd builds the retry call as a SEPARATE lambda from the
        initial call, unlike music/video, which reuse one closure. Forces the
        retry path and asserts the same privacy handling there."""
        calls = []

        def fake_gen(*a, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return (False, "boom")
            return (True, "ok on retry")

        def fake_retry_helper(message, api_url, retry):
            return retry()

        with patch("localm.image_gen.comfy.generate_image", fake_gen), \
             patch("localm.image_gen.comfy.default_api_url",
                   return_value="http://127.0.0.1:9999"), \
             patch("localm.image_gen.comfy.free_comfy_vram"), \
             patch("localm.audit.effective_mode", return_value=SessionMode.PRIVACY), \
             patch("localm.cli.media._maybe_apply_func_shim_and_retry",
                   fake_retry_helper):
            result = CliRunner().invoke(main, ["image", "a cat"])
        assert result.exit_code == 0, result.output
        assert len(calls) == 2
        assert calls[0].get("delete_outputs") is True
        assert calls[1].get("delete_outputs") is True


class TestMusicCmdPrivacy:
    def test_privacy_mode_deletes_outputs(self):
        calls, result = _invoke(
            ["music", "synthwave"],
            "localm.music_gen.generate_music",
            mode=SessionMode.PRIVACY,
            default_api_url_target="localm.media.comfy_client.default_api_url",
        )
        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert calls[0].get("write_sidecar") is False
        assert calls[0].get("delete_outputs") is True

    def test_log_mode_keeps_outputs(self):
        calls, result = _invoke(
            ["music", "synthwave"],
            "localm.music_gen.generate_music",
            mode=SessionMode.LOG,
            default_api_url_target="localm.media.comfy_client.default_api_url",
        )
        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert calls[0].get("write_sidecar") is True
        assert not calls[0].get("delete_outputs")


class TestVideoCmdPrivacy:
    def test_privacy_mode_deletes_outputs(self):
        calls, result = _invoke(
            ["video", "a cat surfing"],
            "localm.video_gen.generate_video",
            mode=SessionMode.PRIVACY,
            default_api_url_target="localm.media.comfy_client.default_api_url",
        )
        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert calls[0].get("write_sidecar") is False
        assert calls[0].get("delete_outputs") is True

    def test_log_mode_keeps_outputs(self):
        calls, result = _invoke(
            ["video", "a cat surfing"],
            "localm.video_gen.generate_video",
            mode=SessionMode.LOG,
            default_api_url_target="localm.media.comfy_client.default_api_url",
        )
        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert calls[0].get("write_sidecar") is True
        assert not calls[0].get("delete_outputs")
