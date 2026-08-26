# SPDX-License-Identifier: AGPL-3.0-or-later
"""REPL media-generation parity: /generate-music and /generate-video mirror
/generate-image - they unload the chat model, resolve the configured ComfyUI
api_url (not a hardcoded default), and honour the privacy contract by
suppressing the on-disk sidecar in privacy mode.

These mirror the /generate-image tests so the three media commands stay
behaviourally identical.
"""

from unittest.mock import MagicMock, patch

import pytest


def _run_repl_media(command: str, generate_target: str, *, mode, api_url,
                    home_dir, engine=None):
    """Drive cli._handle_command(command) with the generator patched, returning
    the kwargs the generator was called with (empty dict if never called)."""
    from localm import cli

    if engine is None:
        engine = MagicMock()
    calls = {}

    def fake_gen(prompt, out, **kwargs):
        calls["prompt"] = prompt
        calls["out"] = out
        calls.update(kwargs)
        return (True, "ok")

    with patch(generate_target, fake_gen), \
         patch("localm.media.comfy_client._comfy_alive", return_value=True), \
         patch("localm.image_gen.comfy.default_api_url", return_value=api_url), \
         patch("localm.image_gen.comfy.free_comfy_vram"), \
         patch("localm.audit.effective_mode", return_value=mode):
        cli._handle_command(command, [], {}, engine=engine)
    return calls, engine


@pytest.mark.parametrize("command,target,ext,subdir", [
    ("/generate-music happy lo-fi beats",
     "localm.music_gen.comfy.generate_music", ".flac", "gui_music"),
    ("/generate-video a cat surfing",
     "localm.video_gen.comfy.generate_video", ".mp4", "gui_video"),
])
def test_repl_media_privacy_no_sidecar(tmp_path, monkeypatch, command, target,
                                       ext, subdir):
    from localm import cli
    from localm.audit import SessionMode

    monkeypatch.setattr(cli, "HOME_DIR", tmp_path)
    calls, engine = _run_repl_media(
        command, target, mode=SessionMode.PRIVACY,
        api_url="http://127.0.0.1:9999", home_dir=tmp_path)

    # Generator actually ran, sidecar suppressed, resolved api_url passed through.
    assert calls.get("write_sidecar") is False
    assert calls.get("api_url") == "http://127.0.0.1:9999"
    # Privacy mode must also delete ComfyUI's own on-disk output copy (it
    # embeds the full prompt/workflow as metadata), not just the sidecar.
    assert calls.get("delete_outputs") is True
    # Output is routed into the per-medium GUI dir with the right extension.
    assert calls["out"].parent == tmp_path / subdir
    assert calls["out"].suffix == ext
    # The chat model is unloaded to free VRAM before generation.
    engine.unload.assert_called_once()


@pytest.mark.parametrize("command,target", [
    ("/generate-music happy lo-fi", "localm.music_gen.comfy.generate_music"),
    ("/generate-video a cat", "localm.video_gen.comfy.generate_video"),
])
def test_repl_media_logmode_keeps_sidecar(tmp_path, monkeypatch, command, target):
    from localm import cli
    from localm.audit import SessionMode

    monkeypatch.setattr(cli, "HOME_DIR", tmp_path)
    calls, _ = _run_repl_media(
        command, target, mode=SessionMode.LOG,
        api_url="http://127.0.0.1:8188", home_dir=tmp_path)
    assert calls.get("write_sidecar") is True
    assert not calls.get("delete_outputs")


@pytest.mark.parametrize("command,target", [
    ("/generate-music x", "localm.music_gen.comfy.generate_music"),
    ("/generate-video x", "localm.video_gen.comfy.generate_video"),
])
def test_repl_media_no_engine_is_graceful(tmp_path, monkeypatch, command, target):
    """With no engine (e.g. model-less chat) the command must decline cleanly
    and never reach the generator."""
    from localm import cli

    monkeypatch.setattr(cli, "HOME_DIR", tmp_path)
    gen = MagicMock()
    with patch(target, gen):
        # engine=None is the model-less case.
        cli._handle_command(command, [], {}, engine=None)
    gen.assert_not_called()


@pytest.mark.parametrize("command,target", [
    ("/generate-music", "localm.music_gen.comfy.generate_music"),
    ("/generate-video", "localm.video_gen.comfy.generate_video"),
])
def test_repl_media_no_arg_shows_usage_no_unload(tmp_path, monkeypatch,
                                                 command, target):
    """An argument-less invocation prints usage and must NOT unload the model
    or call the generator."""
    from localm import cli

    monkeypatch.setattr(cli, "HOME_DIR", tmp_path)
    engine = MagicMock()
    gen = MagicMock()
    with patch(target, gen):
        cli._handle_command(command, [], {}, engine=engine)
    gen.assert_not_called()
    engine.unload.assert_not_called()
