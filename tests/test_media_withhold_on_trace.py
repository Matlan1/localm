# SPDX-License-Identifier: AGPL-3.0-or-later
"""The media backend facade's strict generation (``privacy`` or
``withhold_on_trace``): a ComfyUI backend is asked to fail on a copy it could
not remove, a failed strict generation leaves no local file, and a file that
cannot be removed is named. Also: a media plugin whose config source cannot be
checked because the plugins folder is unreadable is told so."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from localm.plugins import media_config


def _facade(impl):
    return media_config.make_backend_facade("localm.plugins.builtin.image",
                                            impl, "image")


def _impl(calls, *, ok=True, writes=True, comfy=True, embeds=True):
    def generate(s, prompt, out_path, **kw):
        calls.append(kw)
        if writes:
            Path(out_path).write_bytes(b"data")
        return ok, "done" if ok else "it broke"
    return SimpleNamespace(generate=generate, NEEDS_COMFY_SERVER=comfy,
                           EMBEDS_OUTPUT_METADATA=embeds)


@pytest.mark.parametrize("privacy, withhold, comfy, expected", [
    (False, False, True, None),
    (True, False, True, True),
    (False, True, True, True),
    (True, True, False, None),
])
def test_containment_is_required_only_of_a_comfy_backend_in_a_strict_generation(
        tmp_path, privacy, withhold, comfy, expected):
    calls = []
    _facade(_impl(calls, comfy=comfy)).generate(
        {}, "p", tmp_path / "o.bin", privacy=privacy, withhold_on_trace=withhold)
    assert calls[0].get("containment_required") is expected


def test_a_caller_cannot_set_containment_itself(tmp_path):
    with pytest.raises(TypeError):
        _facade(_impl([])).generate({}, "p", tmp_path / "o.bin",
                                    containment_required=False)


@pytest.mark.parametrize("strict", [{"privacy": True}, {"withhold_on_trace": True}])
def test_a_failed_strict_generation_leaves_no_file(tmp_path, strict):
    out = tmp_path / "o.bin"
    ok, msg = _facade(_impl([], ok=False)).generate({}, "p", out, **strict)
    assert (ok, msg) == (False, "it broke")
    assert not out.exists()


@pytest.mark.parametrize("strict", [{"privacy": True}, {"withhold_on_trace": True}])
def test_a_failed_strict_generation_keeps_a_file_it_did_not_write(tmp_path, strict):
    out = tmp_path / "photo.png"
    out.write_bytes(b"the user's photo")
    ok, _msg = _facade(_impl([], ok=False, writes=False)).generate({}, "p", out, **strict)
    assert not ok
    assert out.read_bytes() == b"the user's photo"


def test_a_failed_strict_generation_of_a_backend_that_embeds_nothing_keeps_its_file(
        tmp_path):
    out = tmp_path / "o.bin"
    ok, _msg = _facade(_impl([], ok=False, embeds=False)).generate(
        {}, "p", out, privacy=True)
    assert not ok
    assert out.read_bytes() == b"data"


def test_a_failed_generation_outside_strict_keeps_its_file(tmp_path):
    out = tmp_path / "o.bin"
    ok, _msg = _facade(_impl([], ok=False)).generate({}, "p", out)
    assert not ok
    assert out.read_bytes() == b"data"


def _unlink_raising(monkeypatch, exc):
    def unlink(self, *a, **k):
        raise exc
    monkeypatch.setattr(Path, "unlink", unlink)


def test_withheld_file_already_gone_is_not_reported(tmp_path, monkeypatch):
    out = tmp_path / "o.bin"
    _unlink_raising(monkeypatch, FileNotFoundError(2, "gone"))
    ok, msg = _facade(_impl([], ok=False)).generate({}, "p", out, privacy=True)
    assert (ok, msg) == (False, "it broke")
    assert "by hand" not in msg


def test_withheld_file_that_cannot_be_removed_is_named(tmp_path, monkeypatch):
    out = tmp_path / "o.bin"
    _unlink_raising(monkeypatch, PermissionError(13, "denied"))
    ok, msg = _facade(_impl([], ok=False)).generate({}, "p", out, privacy=True)
    assert not ok
    assert msg.startswith("it broke The file could not be removed")
    assert f"delete {out} by hand." in msg


def _cfg():
    return {"plugins_enabled": ["image", "video"],
            "plugins": {"video": {"use_config_from": "image"}, "image": {}}}


def test_an_unreadable_plugins_folder_is_named_not_called_inactive(monkeypatch):
    monkeypatch.setattr(media_config, "_installed_plugins",
                        lambda: (set(), PermissionError(13, "denied")))
    block, warning = media_config.resolve_config("video", _cfg())
    assert block.get("use_config_from") == "image"
    assert "could not be read" in warning and "not active" not in warning


def test_an_inactive_source_is_still_called_inactive(monkeypatch):
    monkeypatch.setattr(media_config, "_installed_plugins", lambda: (set(), None))
    _block, warning = media_config.resolve_config("video", _cfg())
    assert "'image' is not active" in warning


def test_active_plugins_keeps_its_shape(monkeypatch):
    monkeypatch.setattr(media_config, "_installed_plugins",
                        lambda: ({"image", "music"}, PermissionError(13, "denied")))
    assert media_config.active_plugins({"plugins_enabled": ["image"]}) == {"image"}
