# SPDX-License-Identifier: AGPL-3.0-or-later
"""A ComfyUI/media-backend message or progress line shown by the `image`/
`music`/`video` CLI commands (localm/cli/media.py) must survive verbatim -
Rich's ``Console.print()`` parses ``[...]`` in ANY interpolated string as
markup, not just inside a command's own literal ``[style]`` tags. Reproduced
directly against this venv's rich (see test_rag_cli_markup_escaping.py's
module docstring for the same repro against ``rag``):

    Console().print('render[draft] failed')       -> prints "render failed"
    Console().print('render[bold red] failed')     -> prints "render failed"

The bracketed span is either dropped outright or consumed as a (bogus) style
directive, in both cases silently. The outcome ``message`` shown by
``image``/``music``/``video`` and the streamed ``on_progress`` text both
originate from the ComfyUI backend (a local but SEPARATE process this repo
does not control the content of - comparable in exposure to
maintenance.py's GitHub API text, per dev-notes/RAG-CLI-MARKUP-ESCAPING-
2026-08-20.md) - e.g. ``comfy_exec_error_message()`` echoes a POLL_EXEC_ERROR
payload straight from ComfyUI, and a preflight substitution notice embeds
model filenames read from ComfyUI's own ``/object_info``.

Each ``generate_image``/``generate_music``/``generate_video`` call is faked
at the same boundary test_cli_media_commands.py already uses (the CLI's own
generator entry point - the "lower-level call that actually talks to the
ComfyUI backend" from image_cmd/music_cmd/video_cmd's point of view), so the
real print statements in media.py execute unmodified; only the ComfyUI
round-trip itself is replaced with a realistic payload.

``_open_file``'s ``except Exception as e`` escape (media.py's 7th site) is
deliberately left without a dedicated test here: it is best-effort UI sugar
reachable only through an interactive terminal AND a failing OS file-open
call, and shares the identical, already-exercised ``escape(str(e))`` shape
covered by TestGenerateOrAbortExceptionMarkupEscaping below.
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from localm.audit import SessionMode
from localm.cli import main

# One text Rich DROPS outright, one it consumes as a (bogus) style tag - the
# two distinct failure shapes described in the module docstring above.
BRACKET_DROP_TEXT = "render[draft] failed"
BRACKET_STYLE_TEXT = "render[bold red] failed"


class TestImageCmdMessageMarkupEscaping:
    def test_bracket_drop_message_survives_verbatim(self):
        with patch("localm.image_gen.comfy.generate_image",
                   return_value=(False, BRACKET_DROP_TEXT)), \
             patch("localm.image_gen.comfy.default_api_url",
                   return_value="http://127.0.0.1:9999"), \
             patch("localm.audit.effective_mode", return_value=SessionMode.LOG):
            result = CliRunner().invoke(main, ["image", "a cat"])
        assert result.exit_code == 1, result.output
        assert BRACKET_DROP_TEXT in result.output, (
            f"a failed generation's message must survive verbatim, not be "
            f"silently mangled by Rich markup parsing: {result.output!r}")

    def test_bracket_style_message_survives_verbatim(self):
        with patch("localm.image_gen.comfy.generate_image",
                   return_value=(False, BRACKET_STYLE_TEXT)), \
             patch("localm.image_gen.comfy.default_api_url",
                   return_value="http://127.0.0.1:9999"), \
             patch("localm.audit.effective_mode", return_value=SessionMode.LOG):
            result = CliRunner().invoke(main, ["image", "a cat"])
        assert result.exit_code == 1, result.output
        assert BRACKET_STYLE_TEXT in result.output, (
            f"a message segment that happens to look like a style tag must "
            f"be shown as literal text, not consumed as Rich styling: "
            f"{result.output!r}")


class TestMusicCmdMarkupEscaping:
    def test_message_survives_verbatim(self):
        with patch("localm.music_gen.generate_music",
                   return_value=(False, BRACKET_STYLE_TEXT)), \
             patch("localm.media.comfy_client.default_api_url",
                   return_value="http://127.0.0.1:9999"), \
             patch("localm.audit.effective_mode", return_value=SessionMode.LOG):
            result = CliRunner().invoke(main, ["music", "synthwave"])
        assert result.exit_code == 1, result.output
        assert BRACKET_STYLE_TEXT in result.output, (
            f"'music' failure message must survive verbatim: {result.output!r}")

    def test_progress_text_survives_verbatim(self):
        """generate_music is faked wholesale, so the on_progress callback is
        never invoked by any real ComfyUI-talking code - simulate what a real
        preflight substitution notice does (comfy_client.py's
        preflight_models(), which embeds ComfyUI-reported model filenames)
        by calling the SAME on_progress kwarg music_cmd actually passed in,
        which is media.py's own real lambda."""
        def fake_music(*a, **kwargs):
            on_progress = kwargs.get("on_progress")
            assert on_progress is not None, "music_cmd must pass on_progress"
            on_progress(BRACKET_DROP_TEXT)
            return (True, "ok")

        with patch("localm.music_gen.generate_music", fake_music), \
             patch("localm.media.comfy_client.default_api_url",
                   return_value="http://127.0.0.1:9999"), \
             patch("localm.audit.effective_mode", return_value=SessionMode.LOG):
            result = CliRunner().invoke(main, ["music", "synthwave"])
        assert result.exit_code == 0, result.output
        assert BRACKET_DROP_TEXT in result.output, (
            f"streamed progress text must survive verbatim: {result.output!r}")


class TestVideoCmdMarkupEscaping:
    def test_message_survives_verbatim(self):
        with patch("localm.video_gen.generate_video",
                   return_value=(False, BRACKET_DROP_TEXT)), \
             patch("localm.media.comfy_client.default_api_url",
                   return_value="http://127.0.0.1:9999"), \
             patch("localm.audit.effective_mode", return_value=SessionMode.LOG):
            result = CliRunner().invoke(main, ["video", "a cat surfing"])
        assert result.exit_code == 1, result.output
        assert BRACKET_DROP_TEXT in result.output, (
            f"'video' failure message must survive verbatim: {result.output!r}")

    def test_progress_text_survives_verbatim(self):
        def fake_video(*a, **kwargs):
            on_progress = kwargs.get("on_progress")
            assert on_progress is not None, "video_cmd must pass on_progress"
            on_progress(BRACKET_STYLE_TEXT)
            return (True, "ok")

        with patch("localm.video_gen.generate_video", fake_video), \
             patch("localm.media.comfy_client.default_api_url",
                   return_value="http://127.0.0.1:9999"), \
             patch("localm.audit.effective_mode", return_value=SessionMode.LOG):
            result = CliRunner().invoke(main, ["video", "a cat surfing"])
        assert result.exit_code == 0, result.output
        assert BRACKET_STYLE_TEXT in result.output, (
            f"streamed progress text must survive verbatim: {result.output!r}")


class TestGenerateOrAbortExceptionMarkupEscaping:
    def test_interrupt_failure_message_survives_verbatim(self):
        """_generate_or_abort's except KeyboardInterrupt handler re-reports a
        failure to reach ComfyUI (interrupt_comfy/free_comfy_vram raising) via
        the same unescaped-f-string shape every other site here had. Force a
        real KeyboardInterrupt out of the generator and a real exception out
        of interrupt_comfy, the same way test_rag_cli_markup_escaping.py's
        TestLockMessageEscaping forces a real CollectionLockedError rather
        than mocking the display layer."""
        exc_text = "could not reach host[bold red]:8188"

        def fake_gen(*a, **kwargs):
            raise KeyboardInterrupt()

        def fake_interrupt(api_url):
            raise RuntimeError(exc_text)

        with patch("localm.image_gen.comfy.generate_image", fake_gen), \
             patch("localm.image_gen.comfy.default_api_url",
                   return_value="http://127.0.0.1:9999"), \
             patch("localm.media.comfy_client.interrupt_comfy", fake_interrupt), \
             patch("localm.audit.effective_mode", return_value=SessionMode.LOG):
            result = CliRunner().invoke(main, ["image", "a cat"])
        assert result.exit_code != 0, result.output
        assert exc_text in result.output, (
            f"a failed-interrupt exception message must survive verbatim, "
            f"not be silently mangled by Rich markup parsing: {result.output!r}")
