# SPDX-License-Identifier: AGPL-3.0-or-later
"""Media backends declare capabilities; callers test them, never a backend name.

Covers the shared backend facade in ``localm.plugins.media_config``: capability
attributes on the backend modules, the typed metadata strip at the facade's
output (and its privacy-mode withholding), the ``steps`` / ``clip_skip``
refusal for a backend that does not take them, and an AST sentinel on
comparisons of a ``backend`` setting with a string literal."""

from __future__ import annotations

import ast
import shutil
import struct
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from localm.media import output_metadata
from localm.media.output_metadata import StripStatus, strip_output
from localm.plugins import media_config

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures" / "media"
PNG_SIG = b"\x89PNG\r\n\x1a\n"
SECRET = b"prompt\x00super secret prompt do not leak"


def _chunk(ctype: bytes, data: bytes) -> bytes:
    return (len(data).to_bytes(4, "big") + ctype + data
            + zlib.crc32(ctype + data).to_bytes(4, "big"))


def _png(with_text: bool = True) -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\x00\x00")
    return (PNG_SIG + _chunk(b"IHDR", ihdr)
            + (_chunk(b"tEXt", SECRET) if with_text else b"")
            + _chunk(b"IDAT", idat) + _chunk(b"IEND", b""))


# --------------------------------------------------------------------------- #
#  Typed strip result                                                         #
# --------------------------------------------------------------------------- #

def test_png_with_text_chunk_is_stripped(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(_png())
    result = strip_output("image", path)
    assert result.status is StripStatus.STRIPPED
    assert result.warning == ""
    assert SECRET not in path.read_bytes()


def test_png_without_metadata_is_nothing_to_strip(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(_png(with_text=False))
    before = path.read_bytes()
    result = strip_output("image", path)
    assert result.status is StripStatus.NOTHING_TO_STRIP
    assert path.read_bytes() == before


def test_not_a_png_fails_with_a_reason(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(b"\xff\xd8\xff\xe1Exif prompt")
    result = strip_output("image", path)
    assert result.status is StripStatus.FAILED
    assert result.failed
    assert "not PNG" in result.reason
    assert result.warning == result.reason


@pytest.mark.parametrize("kind,name", [("audio", "prompt.flac"), ("audio", "prompt.mp3"),
                                       ("video", "prompt.mp4")])
def test_container_strip_goes_stripped_then_nothing_to_strip(tmp_path, kind, name):
    path = tmp_path / name
    shutil.copy(FIXTURES / name, path)
    assert strip_output(kind, path).status is StripStatus.STRIPPED
    assert strip_output(kind, path).status is StripStatus.NOTHING_TO_STRIP


def test_missing_file_fails_instead_of_raising(tmp_path):
    result = strip_output("audio", tmp_path / "gone.flac")
    assert result.status is StripStatus.FAILED
    assert result.reason.startswith("WARNING: could not strip audio metadata")


def test_unknown_kind_is_a_key_error(tmp_path):
    with pytest.raises(KeyError):
        strip_output("text", tmp_path / "a.txt")


# --------------------------------------------------------------------------- #
#  The facade strips once, at its output                                      #
# --------------------------------------------------------------------------- #

def _facade(write: bytes, *, embeds: bool = True, ok: bool = True):
    calls = []

    def generate(s, prompt, out_path, **kwargs):
        calls.append(kwargs)
        Path(out_path).write_bytes(write)
        return ok, "generated"

    ref = SimpleNamespace(
        ensure_available=lambda s, on_progress=None: (True, "ready"),
        free_vram=lambda s: True, generate=generate,
        EMBEDS_OUTPUT_METADATA=embeds)
    return media_config.make_backend_facade("localm.plugins.builtin.image", ref,
                                            "image"), calls


def test_a_failed_strip_withholds_the_file_in_privacy_mode(tmp_path):
    facade, calls = _facade(b"\xff\xd8\xff\xe1Exif prompt: secret")
    out = tmp_path / "out.png"
    ok, message = facade.generate({"backend": "comfy"}, "p", out, privacy=True)
    assert not out.exists()
    assert ok is False
    assert message.startswith("Output withheld: WARNING: generated file is not PNG")
    assert "privacy" not in calls[0], "privacy is the facade's argument, not the backend's"


def test_a_failed_strip_outside_privacy_mode_keeps_the_file_and_says_so(tmp_path):
    facade, _calls = _facade(b"\xff\xd8\xff\xe1Exif prompt: secret")
    out = tmp_path / "out.png"
    ok, message = facade.generate({"backend": "comfy"}, "p", out)
    assert ok is True
    assert out.exists()
    assert message.startswith("generated\nWARNING: generated file is not PNG")


def test_a_clean_strip_reports_the_backend_message_unchanged(tmp_path):
    facade, _calls = _facade(_png())
    out = tmp_path / "out.png"
    assert facade.generate({"backend": "comfy"}, "p", out, privacy=True) == (
        True, "generated")
    assert SECRET not in out.read_bytes()


def test_a_successful_generation_that_left_no_file_is_not_a_strip_failure(tmp_path):
    facade, _calls = _facade(b"x")
    out = tmp_path / "sub" / "gone.png"
    facade_ref = facade.resolve({"backend": "comfy"})
    facade_ref.generate = lambda s, prompt, out_path, **k: (True, "generated")
    assert facade.generate({"backend": "comfy"}, "p", out, privacy=True) == (
        True, "generated")


def test_an_empty_output_file_has_nothing_to_strip(tmp_path):
    facade, _calls = _facade(b"")
    out = tmp_path / "out.png"
    assert facade.generate({"backend": "comfy"}, "p", out, privacy=True) == (
        True, "generated")
    assert out.exists()


def test_generate_media_hands_the_callers_progress_function_to_the_backend(
        monkeypatch, tmp_path):
    from localm.plugins.builtin.music import backend
    seen = {}

    def progress(text):
        pass

    def ensure(s, on_progress=None):
        seen["ensure"] = on_progress
        return True, "ready"

    def generate(s, tags, out_path, **kwargs):
        seen["generate"] = kwargs.get("on_progress")
        return True, "done"

    monkeypatch.setattr(backend._COMFY_REF, "ensure_available", ensure)
    monkeypatch.setattr(backend._COMFY_REF, "free_vram", lambda s: True)
    monkeypatch.setattr(backend._COMFY_REF, "generate", generate)
    monkeypatch.setattr(backend._COMFY_REF, "EMBEDS_OUTPUT_METADATA", False)
    media_config.generate_media(backend, {"backend": "comfy"}, "t",
                                tmp_path / "t.flac", on_progress=progress,
                                write_sidecar=True)
    assert seen == {"ensure": progress, "generate": progress}


def test_the_strip_runs_exactly_once_per_generation(tmp_path, monkeypatch):
    seen = []
    real = output_metadata.strip_output
    monkeypatch.setattr(output_metadata, "strip_output",
                        lambda kind, path: seen.append(kind) or real(kind, path))
    facade, _calls = _facade(_png())
    facade.generate({"backend": "comfy"}, "p", tmp_path / "out.png")
    assert seen == ["image"]


def test_a_failed_generation_is_not_stripped_or_removed(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(output_metadata, "strip_output",
                        lambda kind, path: seen.append(kind))
    facade, _calls = _facade(b"partial", ok=False)
    out = tmp_path / "out.png"
    assert facade.generate({"backend": "comfy"}, "p", out, privacy=True) == (
        False, "generated")
    assert seen == []


def test_a_backend_that_embeds_nothing_keeps_its_own_png_text(tmp_path):
    facade, _calls = _facade(_png(), embeds=False)
    out = tmp_path / "out.png"
    facade.generate({"backend": "comfy"}, "p", out, privacy=True)
    assert SECRET in out.read_bytes()


# --------------------------------------------------------------------------- #
#  Capability attributes                                                      #
# --------------------------------------------------------------------------- #

PACKAGES = {"image": "localm.plugins.builtin.image",
            "music": "localm.plugins.builtin.music",
            "video": "localm.plugins.builtin.video"}


@pytest.mark.parametrize("plugin", sorted(PACKAGES))
def test_comfy_needs_a_server_and_native_embeds_metadata_only_for_video(plugin):
    from importlib import import_module
    backend = import_module(f"{PACKAGES[plugin]}.backend")
    comfy = media_config.backend_capabilities(PACKAGES[plugin], "comfy",
                                              backend._COMFY_REF)
    native = media_config.backend_capabilities(PACKAGES[plugin], "native",
                                               backend._COMFY_REF)
    assert comfy["NEEDS_COMFY_SERVER"] and comfy["EMBEDS_OUTPUT_METADATA"]
    assert not comfy["SUPPORTS_JOB_OPTIONS"]
    assert not native["NEEDS_COMFY_SERVER"]
    assert native["EMBEDS_OUTPUT_METADATA"] is (plugin == "video")
    assert native["SUPPORTS_JOB_OPTIONS"]


def test_only_the_native_image_backend_takes_clip_skip_and_upscales():
    from localm.plugins.builtin.image import backend
    comfy = backend.supports({"backend": "comfy"}, "SUPPORTS_CLIP_SKIP")
    native = backend.supports({"backend": "native"}, "SUPPORTS_CLIP_SKIP")
    assert (comfy, native) == (False, True)
    assert backend.supports({"backend": "native"}, "SUPPORTS_UPSCALE")
    assert not backend.supports({"backend": "comfy"}, "SUPPORTS_UPSCALE")
    assert not backend.supports({"backend": "comfy"}, "SUPPORTS_STEPS")
    assert backend.supports({"backend": "native"}, "SUPPORTS_STEPS")


def test_an_unknown_capability_name_is_a_key_error():
    from localm.plugins.builtin.image import backend
    with pytest.raises(KeyError):
        backend.supports({"backend": "comfy"}, "SUPPORTS_TYPOS")


def test_an_unloadable_backend_name_reads_as_comfy():
    from localm.plugins.builtin.image import backend
    assert backend.supports({"backend": "no-such-backend"}, "NEEDS_COMFY_SERVER")


# --------------------------------------------------------------------------- #
#  steps / clip_skip are refused through the facade, for every caller         #
# --------------------------------------------------------------------------- #

def _comfy_image_settings():
    from localm.plugins.builtin.image import backend
    return backend, {"backend": "comfy", "api_url": "http://127.0.0.1:1"}


@pytest.mark.parametrize("kwargs,named", [({"steps": 4}, "steps"),
                                          ({"clip_skip": 2}, "clip_skip")])
def test_a_backend_without_the_capability_refuses_before_loading_anything(
        monkeypatch, tmp_path, kwargs, named):
    backend, s = _comfy_image_settings()
    touched = []
    for name in ("ensure_available", "generate", "free_vram"):
        monkeypatch.setattr(backend._COMFY_REF, name,
                            lambda *a, _n=name, **k: touched.append(_n))
    ok, message, path = media_config.generate_media(
        backend, s, "p", tmp_path / "o.png", write_sidecar=True, **kwargs)
    assert ok is False
    assert named in message
    assert path == tmp_path / "o.png"
    assert touched == []


def test_a_backend_with_the_capability_passes_steps_to_the_generator(
        monkeypatch, tmp_path):
    from localm.plugins.builtin.music import backend
    seen = {}

    def generate(s, tags, out_path, **kwargs):
        seen.update(kwargs)
        return True, "done"

    monkeypatch.setattr(backend._COMFY_REF, "ensure_available",
                        lambda s, on_progress=None: (True, "ready"))
    monkeypatch.setattr(backend._COMFY_REF, "free_vram", lambda s: True)
    monkeypatch.setattr(backend._COMFY_REF, "generate", generate)
    monkeypatch.setattr(backend._COMFY_REF, "EMBEDS_OUTPUT_METADATA", False)
    ok, message, _path = media_config.generate_media(
        backend, {"backend": "comfy"}, "tags", tmp_path / "t.flac",
        write_sidecar=True, steps=7)
    assert (ok, message) == (True, "done")
    assert seen["steps"] == 7


@pytest.mark.parametrize("plugin,out", [("image", "o.png"), ("music", "o.flac"),
                                        ("video", "o.mp4")])
def test_a_caller_that_gives_no_self_url_reaches_the_comfyui_generator(
        monkeypatch, tmp_path, plugin, out):
    from importlib import import_module
    backend = import_module(f"{PACKAGES[plugin]}.backend")
    seen = {}

    def generate(prompt, out_path, **kwargs):
        seen.update(kwargs)
        return True, "done"

    monkeypatch.setattr(backend._COMFY_REF, "ensure_available",
                        lambda s, on_progress=None: (True, "ready"))
    monkeypatch.setattr(backend._COMFY_REF, "free_vram", lambda s: True)
    monkeypatch.setattr(backend._COMFY_REF, "generate", backend._comfy_generate)
    monkeypatch.setattr(backend._COMFY_REF, "EMBEDS_OUTPUT_METADATA", False)
    gen_module = {"image": "localm.image_gen.comfy.generate_image",
                  "music": "localm.music_gen.generate_music",
                  "video": "localm.video_gen.generate_video"}[plugin]
    from unittest.mock import patch
    s = {"backend": "comfy", "api_url": "http://127.0.0.1:1", "launch_cmd": "",
         "workdir": "", "output_dir": ""}
    with patch(gen_module, generate):
        ok, message, _path = media_config.generate_media(
            backend, s, "p", tmp_path / out, write_sidecar=True)
    assert (ok, message) == (True, "done")
    assert seen["localm_url"] is None


# --------------------------------------------------------------------------- #
#  Sentinel: no new comparison of a "backend" setting with a string literal   #
# --------------------------------------------------------------------------- #

_ALLOWED = (
    "localm/plugins/media_config.py",
    "localm/media/backend_choice.py",
)
_ALLOWED_GLOBS = (
    "localm/plugins/builtin/*/backend.py",
    "localm/plugins/builtin/*/backends/*.py",
)
# Sites outside the media plugins that still test the backend name, by file.
# The numbers only go down: lowering one is the ratchet, raising one fails.
_BASELINE = {
    "localm/plugins/gui/routes/models/acquisition.py": 2,
    "localm/vram.py": 2,
}


def _is_backend_value(node: ast.AST) -> bool:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr == "get" and node.args:
        first = node.args[0]
        return isinstance(first, ast.Constant) and first.value == "backend"
    if isinstance(node, ast.Subscript):
        key = node.slice
        return isinstance(key, ast.Constant) and key.value == "backend"
    return False


def _is_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return bool(node.elts) and all(_is_literal(e) for e in node.elts)
    return False


_MEDIA_BACKEND_NAMES = {"native", "comfy", "auto"}


def _names_a_media_backend(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return node.value in _MEDIA_BACKEND_NAMES
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(_names_a_media_backend(e) for e in node.elts)
    return False


def _name_comparisons(source: str) -> int:
    count = 0
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        if any(_is_backend_value(o) for o in operands) \
                and any(_is_literal(o) and _names_a_media_backend(o)
                        for o in operands):
            count += 1
    return count


def _scan() -> dict:
    found = {}
    allowed = {REPO / p for p in _ALLOWED}
    for glob in _ALLOWED_GLOBS:
        allowed.update(REPO.glob(glob))
    for path in sorted((REPO / "localm").rglob("*.py")):
        if path in allowed:
            continue
        n = _name_comparisons(path.read_text(encoding="utf-8"))
        if n:
            found[path.relative_to(REPO).as_posix()] = n
    return found


def test_no_comparison_of_the_backend_setting_with_a_string_literal_is_added():
    found = _scan()
    added = {f: n for f, n in found.items() if n > _BASELINE.get(f, 0)}
    assert not added, (
        "These files compare a 'backend' setting with a string literal; test a "
        f"capability attribute through the plugin's backend.supports instead: {added}")


def test_the_backend_name_comparison_baseline_only_goes_down():
    found = _scan()
    stale = {f: (n, found.get(f, 0)) for f, n in _BASELINE.items()
             if found.get(f, 0) < n}
    assert not stale, (
        f"A baseline entry is higher than the sites left; lower it to the "
        f"second number: {stale}")


def test_the_sentinel_sees_the_comparison_shapes_it_is_for():
    assert _name_comparisons('s.get("backend") == "native"') == 1
    assert _name_comparisons('"comfy" != s["backend"]') == 1
    assert _name_comparisons('settings.get("backend") in ("native", "auto")') == 1
    assert _name_comparisons('s.get("backend_choice") == "auto"') == 0
    assert _name_comparisons('name == "native"') == 0
    assert _name_comparisons('s.get("backend") == other') == 0


# --------------------------------------------------------------------------- #
#  Sentinel: only the plugins' backend modules call the ComfyUI generators    #
# --------------------------------------------------------------------------- #

_GENERATORS = {"generate_image", "generate_music", "generate_video"}
_GENERATOR_HOMES = ("localm/image_gen/", "localm/music_gen/", "localm/video_gen/")
_GENERATOR_CALLERS = {
    "localm/plugins/builtin/image/backend.py",
    "localm/plugins/builtin/music/backend.py",
    "localm/plugins/builtin/video/backend.py",
}


def _generator_references(source: str) -> list:
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            hits += [a.name for a in node.names if a.name in _GENERATORS]
        elif isinstance(node, ast.Attribute) and node.attr in _GENERATORS:
            hits.append(node.attr)
    return hits


def test_only_the_backend_modules_reach_the_comfyui_generators():
    offenders = {}
    for path in sorted((REPO / "localm").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if (rel.startswith(_GENERATOR_HOMES) or rel in _GENERATOR_CALLERS
                or rel.startswith("localm/media/sdcpp/") or "/backends/" in rel):
            continue
        hits = _generator_references(path.read_text(encoding="utf-8"))
        if hits:
            offenders[rel] = hits
    assert not offenders, (
        "Generate through the plugin's backend.generate_media; only the three "
        f"backend modules call the ComfyUI generators: {offenders}")


def test_the_generator_sentinel_sees_both_reference_shapes():
    assert _generator_references("from x.comfy import generate_image") == ["generate_image"]
    assert _generator_references("m.generate_music(a)") == ["generate_music"]
    assert _generator_references("generate_media(a)") == []


def _failing_unlink(exc):
    def unlink(self, *args, **kwargs):
        raise exc
    return unlink


def test_a_withheld_file_that_is_already_gone_is_still_withheld(tmp_path, monkeypatch):
    facade, _ = _facade(b"\xff\xd8\xff\xe1Exif prompt: secret")
    out = tmp_path / "out.png"
    monkeypatch.setattr(Path, "unlink", _failing_unlink(FileNotFoundError("gone")))
    ok, message = facade.generate({"backend": "comfy"}, "p", out, privacy=True)
    assert ok is False
    assert message.startswith("Output withheld: WARNING: generated file is not PNG")
    assert "by hand" not in message


def test_a_withheld_file_that_cannot_be_removed_says_to_delete_it_by_hand(tmp_path, monkeypatch):
    facade, _ = _facade(b"\xff\xd8\xff\xe1Exif prompt: secret")
    out = tmp_path / "out.png"
    monkeypatch.setattr(Path, "unlink", _failing_unlink(PermissionError("denied")))
    ok, message = facade.generate({"backend": "comfy"}, "p", out, privacy=True)
    assert ok is False
    assert message.startswith("Output withheld: WARNING: generated file is not PNG")
    assert f"delete {out} by hand" in message and "denied" in message
