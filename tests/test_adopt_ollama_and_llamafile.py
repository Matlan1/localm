# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm add` over an Ollama model store registers every GGUF model in it, and
over a llamafile unpacks the model out of the archive appended to the executable
and registers that copy."""

import hashlib
import json
import struct
import zipfile
from pathlib import Path

import pytest

from localm.config import load_registry
from localm.model_manager import add_local

_T_STRING = 8
_MIN_BYTES = 2048


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    import localm.model_manager as mm
    (home / "models").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mm, "MODELS_DIR", home / "models")
    monkeypatch.setattr(mm, "ensure_dirs", lambda: (home / "models").mkdir(parents=True, exist_ok=True))
    return home


def _s(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf(arch: str = "llama", tag: str = "") -> bytes:
    """A valid GGUF header declaring *arch*, padded past the size floor; *tag*
    makes the bytes unique."""
    kv = [("general.architecture", arch), ("general.name", tag or arch)]
    out = [b"GGUF", struct.pack("<I", 3), struct.pack("<QQ", 0, len(kv))]
    for key, val in kv:
        out += [_s(key), struct.pack("<I", _T_STRING), _s(val)]
    body = b"".join(out)
    return body + b"\0" * (_MIN_BYTES - len(body))


# ----------------------------------------------------------------- Ollama

_MODEL = "application/vnd.ollama.image.model"
_PROJECTOR = "application/vnd.ollama.image.projector"


def _blob(root: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    (root / "blobs" / f"sha256-{digest}").write_bytes(data)
    return digest


def _manifest(root: Path, rel: str, layers: list) -> None:
    f = root / "manifests" / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"schemaVersion": 2, "layers": layers}), encoding="utf-8")


def _layer(media: str, digest: str) -> dict:
    return {"mediaType": media, "digest": f"sha256:{digest}", "size": 1}


def _store(tmp_path, *models) -> Path:
    """An Ollama store with one text model per (relative manifest path, tag-bytes)."""
    root = tmp_path / "ollama"
    for rel, data in models:
        _manifest(root, rel, [_layer(_MODEL, _blob(root, data)),
                              _layer("application/vnd.ollama.image.template", "0" * 64)])
    return root


class TestOllamaRoot:
    def test_every_model_registers_named_model_tag(self, tmp_path, isolated_home):
        a, b = _gguf(tag="a"), _gguf(tag="b")
        root = _store(tmp_path,
                      ("registry.ollama.ai/library/llama3/latest", a),
                      ("registry.ollama.ai/library/Qwen/7B", b))
        assert add_local(str(root)) is True
        reg = load_registry()
        assert set(reg) == {"llama3-latest", "qwen-7b"}
        entry = reg["llama3-latest"]
        digest = hashlib.sha256(a).hexdigest()
        assert Path(entry["path"]) == (root / "blobs" / f"sha256-{digest}").resolve()
        assert entry["source"] == "ollama" and entry["sha256"] == digest
        assert entry["model_type"] == "llm"

    def test_dot_ollama_folder_resolves_to_its_models_folder(self, tmp_path, isolated_home):
        root = _store(tmp_path, ("registry.ollama.ai/library/m/latest", _gguf(tag="m")))
        dot = tmp_path / "dotollama"
        dot.mkdir()
        root.rename(dot / "models")
        assert add_local(str(dot)) is True
        assert set(load_registry()) == {"m-latest"}

    def test_owner_namespace_is_part_of_the_name(self, tmp_path, isolated_home):
        root = _store(tmp_path, ("hf.co/someuser/somerepo/q4_k_m", _gguf(tag="x")))
        assert add_local(str(root)) is True
        assert set(load_registry()) == {"someuser-somerepo-q4_k_m"}

    def test_two_tags_of_one_blob_are_two_names_for_one_file(self, tmp_path, isolated_home):
        data = _gguf(tag="shared")
        root = _store(tmp_path,
                      ("registry.ollama.ai/library/m/8b", data),
                      ("registry.ollama.ai/library/m/latest", data))
        assert add_local(str(root)) is True
        reg = load_registry()
        assert set(reg) == {"m-8b", "m-latest"}
        assert reg["m-8b"]["path"] == reg["m-latest"]["path"]

    def test_embedding_architecture_is_typed(self, tmp_path, isolated_home):
        root = _store(tmp_path, ("registry.ollama.ai/library/embed/latest",
                                 _gguf(arch="bert", tag="e")))
        assert add_local(str(root)) is True
        assert load_registry()["embed-latest"]["model_type"] == "embedding"

    def test_name_applies_to_a_single_model_only(self, tmp_path, isolated_home):
        root = _store(tmp_path, ("registry.ollama.ai/library/m/latest", _gguf(tag="m")))
        assert add_local(str(root), name="mine") is True
        assert set(load_registry()) == {"mine"}

    def test_empty_store_is_not_a_model(self, tmp_path, isolated_home):
        root = tmp_path / "ollama"
        (root / "manifests").mkdir(parents=True)
        (root / "blobs").mkdir()
        assert add_local(str(root)) is False
        assert load_registry() == {}


class TestOllamaProjector:
    def _vision(self, tmp_path, projector: bytes) -> Path:
        root = tmp_path / "ollama"
        model = _blob(root, _gguf(tag="vision"))
        proj = _blob(root, projector)
        _manifest(root, "registry.ollama.ai/library/llava/latest",
                  [_layer(_MODEL, model), _layer(_PROJECTOR, proj)])
        return root

    def test_gguf_projector_layer_is_attached(self, tmp_path, isolated_home):
        proj = _gguf(arch="clip", tag="proj")
        root = self._vision(tmp_path, proj)
        assert add_local(str(root)) is True
        entry = load_registry()["llava-latest"]
        expect = (root / "blobs" / f"sha256-{hashlib.sha256(proj).hexdigest()}").resolve()
        assert Path(entry["mmproj"]) == expect

    def test_projector_layer_that_is_not_a_projector_is_left_out(self, tmp_path, isolated_home, capsys):
        root = self._vision(tmp_path, _gguf(arch="llama", tag="notproj"))
        assert add_local(str(root)) is True
        assert "mmproj" not in load_registry()["llava-latest"]
        assert "no vision projector attached" in capsys.readouterr().out


class TestOllamaProblems:
    def test_a_non_gguf_model_layer_is_reported_and_skipped(self, tmp_path, isolated_home, capsys):
        root = _store(tmp_path,
                      ("registry.ollama.ai/library/good/latest", _gguf(tag="good")),
                      ("registry.ollama.ai/library/st/latest", b"\x08\0\0\0\0\0\0\0{}" + b"x" * 2048))
        assert add_local(str(root)) is True
        assert set(load_registry()) == {"good-latest"}
        assert "st-latest: the model is not a GGUF file" in capsys.readouterr().out

    def test_a_missing_blob_is_reported_and_skipped(self, tmp_path, isolated_home, capsys):
        root = _store(tmp_path, ("registry.ollama.ai/library/good/latest", _gguf(tag="good")))
        _manifest(root, "registry.ollama.ai/library/gone/latest", [_layer(_MODEL, "f" * 64)])
        assert add_local(str(root)) is True
        assert set(load_registry()) == {"good-latest"}
        assert "gone-latest" in capsys.readouterr().out

    def test_a_hostile_digest_never_leaves_the_blobs_folder(self, tmp_path, isolated_home, capsys):
        root = _store(tmp_path, ("registry.ollama.ai/library/good/latest", _gguf(tag="good")))
        outside = tmp_path / "secret.gguf"
        outside.write_bytes(_gguf(tag="secret"))
        _manifest(root, "registry.ollama.ai/library/evil/latest",
                  [{"mediaType": _MODEL, "digest": "sha256:../../../secret.gguf"}])
        assert add_local(str(root)) is True
        assert set(load_registry()) == {"good-latest"}
        assert "malformed digest" in capsys.readouterr().out

    def test_junk_files_in_the_manifest_tree_are_ignored(self, tmp_path, isolated_home):
        root = _store(tmp_path, ("registry.ollama.ai/library/good/latest", _gguf(tag="good")))
        (root / "manifests" / ".DS_Store").write_bytes(b"\0\1")
        (root / "manifests" / "registry.ollama.ai" / "library" / "good" / "notes.txt").write_text("hi")
        assert add_local(str(root)) is True
        assert set(load_registry()) == {"good-latest"}


class TestOllamaControls:
    def test_a_single_manifest_folder_still_registers_one_model(self, tmp_path, isolated_home):
        root = _store(tmp_path, ("registry.ollama.ai/library/m/latest", _gguf(tag="m")))
        assert add_local(str(root / "manifests" / "registry.ollama.ai" / "library" / "m")) is True
        assert set(load_registry()) == {"m-latest"}

    def test_copy_brings_the_blob_into_the_models_folder(self, tmp_path, isolated_home):
        root = _store(tmp_path, ("registry.ollama.ai/library/m/latest", _gguf(tag="m")))
        assert add_local(str(root), store="copy") is True
        path = Path(load_registry()["m-latest"]["path"])
        assert path.parent == (isolated_home / "models").resolve()
        assert any((root / "blobs").iterdir())


# -------------------------------------------------------------- llamafile

def _llamafile(path: Path, members: dict, *, compression=zipfile.ZIP_STORED) -> Path:
    """An executable stub with a ZIP appended, the way llamafile ships weights."""
    path.write_bytes(b"MZ" + b"\x90" * 4096)
    with zipfile.ZipFile(path, "a") as z:
        for name, data in members.items():
            z.writestr(zipfile.ZipInfo(name), data, compress_type=compression)
    return path


class TestLlamafile:
    @pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED],
                             ids=["stored", "deflated"])
    def test_model_member_is_extracted_and_registered(self, tmp_path, isolated_home, compression):
        data = _gguf(tag="inside")
        lf = _llamafile(tmp_path / "Tiny.llamafile",
                        {"Tiny.gguf": data, ".args": b"-m\nTiny.gguf\n", "LICENSE": b"x"},
                        compression=compression)
        original = lf.read_bytes()
        assert add_local(str(lf)) is True
        reg = load_registry()
        assert set(reg) == {"Tiny"}
        entry = reg["Tiny"]
        path = Path(entry["path"])
        assert path.parent == (isolated_home / "models").resolve() and path.name == "Tiny.gguf"
        assert path.read_bytes() == data
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert entry["model_type"] == "llm"
        assert lf.read_bytes() == original
        assert not list((isolated_home / "models").glob(".*.part"))

    def test_projector_member_is_attached(self, tmp_path, isolated_home):
        proj = _gguf(arch="clip", tag="p")
        lf = _llamafile(tmp_path / "v.llamafile",
                        {"v-Q4.gguf": _gguf(tag="v"), "v-mmproj-Q4.gguf": proj})
        assert add_local(str(lf)) is True
        reg = load_registry()
        assert set(reg) == {"v-Q4"}
        assert Path(reg["v-Q4"]["mmproj"]).read_bytes() == proj

    def test_name_override(self, tmp_path, isolated_home):
        lf = _llamafile(tmp_path / "t.llamafile", {"t.gguf": _gguf(tag="t")})
        assert add_local(str(lf), name="mine") is True
        assert set(load_registry()) == {"mine"}

    def test_adding_again_extracts_nothing_new(self, tmp_path, isolated_home):
        lf = _llamafile(tmp_path / "t.llamafile", {"t.gguf": _gguf(tag="t")})
        assert add_local(str(lf)) is True
        assert add_local(str(lf)) is True
        assert set(load_registry()) == {"t"}
        assert [p.name for p in (isolated_home / "models").iterdir()] == ["t.gguf"]

    def test_a_different_file_at_the_name_gets_a_new_name(self, tmp_path, isolated_home):
        models = isolated_home / "models"
        models.mkdir(parents=True, exist_ok=True)
        (models / "t.gguf").write_bytes(_gguf(tag="other"))
        lf = _llamafile(tmp_path / "t.llamafile", {"t.gguf": _gguf(tag="mine")})
        assert add_local(str(lf)) is True
        path = Path(load_registry()["t-2"]["path"])
        assert path.name == "t-2.gguf"
        assert (models / "t.gguf").read_bytes() == _gguf(tag="other")

    def test_traversing_member_name_stays_in_the_models_folder(self, tmp_path, isolated_home):
        lf = _llamafile(tmp_path / "t.llamafile", {"../evil.gguf": _gguf(tag="evil")})
        assert add_local(str(lf)) is True
        assert not (tmp_path / "evil.gguf").exists()
        assert Path(load_registry()["evil"]["path"]).parent == (isolated_home / "models").resolve()

    def test_reserved_character_in_member_name_is_refused(self, tmp_path, isolated_home):
        lf = _llamafile(tmp_path / "t.llamafile", {"model:stream.gguf": _gguf(tag="x")})
        assert add_local(str(lf)) is False
        assert load_registry() == {}

    def test_member_without_gguf_magic_is_refused_and_leaves_nothing(self, tmp_path, isolated_home):
        lf = _llamafile(tmp_path / "t.llamafile", {"t.gguf": b"NOPE" + b"\0" * _MIN_BYTES})
        assert add_local(str(lf)) is False
        assert load_registry() == {}
        assert list((isolated_home / "models").iterdir()) == []

    def test_corrupt_member_is_refused_and_leaves_nothing(self, tmp_path, isolated_home):
        data = _gguf(tag="c")
        lf = _llamafile(tmp_path / "t.llamafile", {"t.gguf": data})
        raw = bytearray(lf.read_bytes())
        at = raw.index(data) + 100
        raw[at] ^= 0xFF
        lf.write_bytes(bytes(raw))
        assert add_local(str(lf)) is False
        assert load_registry() == {}
        assert list((isolated_home / "models").iterdir()) == []

    def test_not_enough_disk_extracts_nothing(self, tmp_path, isolated_home, monkeypatch):
        import localm.model_manager as mm
        monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: False)
        lf = _llamafile(tmp_path / "t.llamafile", {"t.gguf": _gguf(tag="t")})
        assert add_local(str(lf)) is False
        assert load_registry() == {}
        assert list((isolated_home / "models").iterdir()) == []

    def test_gui_mode_reports_extraction_progress(self, tmp_path, isolated_home, monkeypatch, capsys):
        from localm.model_manager import PROGRESS_SENTINEL
        monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")
        lf = _llamafile(tmp_path / "t.llamafile", {"t.gguf": _gguf(tag="t")})
        assert add_local(str(lf)) is True
        frames = [json.loads(line[len(PROGRESS_SENTINEL):])
                  for line in capsys.readouterr().out.splitlines()
                  if line.startswith(PROGRESS_SENTINEL)]
        extract = [f for f in frames if f["phase"] == "extract"]
        assert extract and extract[-1]["downloaded"] == extract[-1]["total"] == _MIN_BYTES


class TestNotALlamafile:
    def test_plain_executable_is_not_a_model(self, tmp_path, isolated_home):
        exe = tmp_path / "tool.exe"
        exe.write_bytes(b"MZ" + b"\0" * 4096)
        assert add_local(str(exe)) is False
        assert load_registry() == {}

    def test_zip_without_a_gguf_member_is_not_a_llamafile(self, tmp_path, isolated_home):
        lf = _llamafile(tmp_path / "t.llamafile", {"readme.txt": b"hello"})
        assert add_local(str(lf)) is False
        assert load_registry() == {}

    def test_zip_of_a_gguf_with_another_extension_is_left_alone(self, tmp_path, isolated_home):
        z = _llamafile(tmp_path / "bundle.zip", {"t.gguf": _gguf(tag="t")})
        assert add_local(str(z)) is False
        assert load_registry() == {}

    def test_a_plain_gguf_still_registers_in_place(self, tmp_path, isolated_home):
        f = tmp_path / "plain.gguf"
        f.write_bytes(_gguf(tag="plain"))
        assert add_local(str(f)) is True
        assert Path(load_registry()["plain"]["path"]) == f.resolve()
