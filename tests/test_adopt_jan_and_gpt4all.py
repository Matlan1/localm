# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm add` over the model stores of the Jan desktop app and GPT4All.

Jan keeps each model in ``llamacpp/models/<id>/`` beside a ``model.yml``; GPT4All
keeps flat GGUF files and names a download in progress ``incomplete-<file>``. A
GGUF whose header declares tensor data beyond the end of the file is an unfinished
copy and is never registered by a folder import (a single-file add is unchanged).
"""

import struct
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


def _gguf(arch: str = "llama", tag: str = "", *, tensor_offset: int = 0) -> bytes:
    """A valid GGUF header declaring *arch*, padded past the size floor. With
    *tensor_offset* the header declares one tensor whose data starts that many
    bytes into the tensor-data section; *tag* makes the bytes unique."""
    kv = [("general.architecture", arch), ("general.name", tag or arch)]
    tensors = 1 if tensor_offset else 0
    out = [b"GGUF", struct.pack("<I", 3), struct.pack("<QQ", tensors, len(kv))]
    for key, val in kv:
        out += [_s(key), struct.pack("<I", _T_STRING), _s(val)]
    if tensor_offset:
        out += [_s("blk.0.weight"), struct.pack("<I", 1), struct.pack("<Q", 8),
                struct.pack("<I", 0), struct.pack("<Q", tensor_offset)]
    body = b"".join(out)
    return body + b"\0" * max(0, _MIN_BYTES - len(body))


def _jan_unit(root: Path, model_id: str, *, yml: str = "", weights: bytes | None = None,
              projector: bytes | None = None, extra: dict | None = None) -> Path:
    unit = root / "llamacpp" / "models" / model_id
    unit.mkdir(parents=True, exist_ok=True)
    (unit / "model.yml").write_text(yml, encoding="utf-8")
    if weights is not None:
        (unit / "model.gguf").write_bytes(weights)
    if projector is not None:
        (unit / "mmproj.gguf").write_bytes(projector)
    for name, data in (extra or {}).items():
        (unit / name).write_bytes(data)
    return unit


class TestJanStore:
    def _store(self, tmp_path) -> Path:
        data = tmp_path / "Jan" / "data"
        _jan_unit(data, "janhq/Jan-v3-4b-Q4_K_XL", weights=_gguf(tag="a"),
                  projector=_gguf("clip", "proj"), extra={"mtp.gguf": _gguf(tag="draft")})
        _jan_unit(data, "janhq/Jan-v3-4b-Q8_0", weights=_gguf(tag="b"))
        return data

    def test_models_register_under_their_jan_id_with_the_projector_recorded(
            self, tmp_path, isolated_home):
        data = self._store(tmp_path)
        assert add_local(str(data)) is True
        reg = load_registry()
        assert set(reg) == {"janhq-Jan-v3-4b-Q4_K_XL", "janhq-Jan-v3-4b-Q8_0"}
        vision = reg["janhq-Jan-v3-4b-Q4_K_XL"]
        assert vision["model_type"] == "llm"
        assert Path(vision["mmproj"]) == (
            data / "llamacpp" / "models" / "janhq" / "Jan-v3-4b-Q4_K_XL" / "mmproj.gguf").resolve()
        assert not reg["janhq-Jan-v3-4b-Q8_0"].get("mmproj")

    @pytest.mark.parametrize("sub", ["llamacpp", "llamacpp/models"])
    def test_the_llamacpp_and_models_folders_are_accepted(self, tmp_path, isolated_home, sub):
        data = self._store(tmp_path)
        assert add_local(str(data / sub)) is True
        assert len(load_registry()) == 2

    def test_a_download_in_progress_is_skipped_and_the_rest_registers(
            self, tmp_path, isolated_home, capsys):
        data = self._store(tmp_path)
        _jan_unit(data, "qwen/Qwen3-1.7B-Q4_K_M",
                  extra={"model.gguf.tmp": b"partial", "model.gguf.url": b"https://x"})
        assert add_local(str(data)) is True
        assert "qwen-Qwen3-1.7B-Q4_K_M" not in load_registry()
        assert len(load_registry()) == 2
        assert "download has not finished" in " ".join(capsys.readouterr().out.split())

    def test_a_model_linked_by_absolute_path_registers_the_linked_file(
            self, tmp_path, isolated_home):
        data = tmp_path / "Jan" / "data"
        elsewhere = tmp_path / "elsewhere" / "Mistral-7B.gguf"
        elsewhere.parent.mkdir()
        elsewhere.write_bytes(_gguf(tag="linked"))
        _jan_unit(data, "Mistral-7B", yml=f"model_path: {elsewhere}\n")
        assert add_local(str(data)) is True
        reg = load_registry()
        assert set(reg) == {"Mistral-7B"}
        assert Path(reg["Mistral-7B"]["path"]) == elsewhere.resolve()

    def test_a_relative_model_path_is_taken_from_the_jan_data_folder(
            self, tmp_path, isolated_home):
        data = tmp_path / "Jan" / "data"
        _jan_unit(data, "org/orig-name",
                  yml="model_path: llamacpp/models/org/orig-name/Original.Q4.gguf\n")
        (data / "llamacpp" / "models" / "org" / "orig-name" / "Original.Q4.gguf").write_bytes(
            _gguf(tag="o"))
        assert add_local(str(data)) is True
        assert Path(load_registry()["org-orig-name"]["path"]).name == "Original.Q4.gguf"

    def test_a_dangling_link_is_reported_by_the_name_it_links_to(
            self, tmp_path, isolated_home, capsys):
        data = self._store(tmp_path)
        target = tmp_path / "nowhere.gguf"
        _jan_unit(data, "gone/Linked", yml=f"model_path: {target}\n")
        assert add_local(str(data)) is True
        assert "gone-Linked" not in load_registry()
        out = " ".join(capsys.readouterr().out.split())
        assert "linked file is missing" in out and "nowhere.gguf" in out

    def test_a_model_path_climbing_out_of_the_data_folder_is_not_followed(
            self, tmp_path, isolated_home, capsys):
        data = tmp_path / "Jan" / "data"
        _jan_unit(data, "org/escape", yml="model_path: ../../secret.gguf\n")
        (tmp_path / "secret.gguf").write_bytes(_gguf(tag="secret"))
        assert add_local(str(data)) is False
        assert load_registry() == {}
        assert "climbs out" in " ".join(capsys.readouterr().out.split())

    @pytest.mark.parametrize("link", [
        r"\\host\share\x.gguf", "//host/share/x.gguf", "C:x.gguf", r"\rooted\x.gguf"])
    def test_network_drive_relative_and_rooted_links_are_not_followed(
            self, tmp_path, isolated_home, link):
        from localm.model_manager.jan_store import _resolve
        assert _resolve(link, tmp_path) is None

    def test_a_huge_whitespace_run_in_model_yml_does_not_hang_the_scan(
            self, tmp_path, isolated_home):
        import time
        data = tmp_path / "Jan" / "data"
        _jan_unit(data, "org/hostile", weights=_gguf(tag="h"),
                  yml="model_path: a" + " " * 3000 + "b\n" + "# " + "x" * 5_000_000 + "\n")
        started = time.monotonic()
        assert add_local(str(data)) is True
        assert time.monotonic() - started < 10

    def test_bom_and_trailing_comments_in_model_yml_are_read(self, tmp_path, isolated_home):
        data = tmp_path / "Jan" / "data"
        unit = _jan_unit(data, "org/odd", weights=None,
                         yml="model_path: custom.gguf # the weights\n")
        (unit / "model.yml").write_bytes(
            b"\xef\xbb\xbfmodel_path: llamacpp/models/org/odd/custom.gguf # the weights\n")
        (unit / "custom.gguf").write_bytes(_gguf(tag="odd"))
        assert add_local(str(data)) is True
        assert Path(load_registry()["org-odd"]["path"]).name == "custom.gguf"

    def test_a_download_with_no_model_yml_yet_is_reported(
            self, tmp_path, isolated_home, capsys):
        data = self._store(tmp_path)
        pending = data / "llamacpp" / "models" / "qwen" / "Q4"
        pending.mkdir(parents=True)
        (pending / "model.gguf.tmp").write_bytes(b"partial")
        assert add_local(str(data)) is True
        assert "qwen-Q4: its download has not finished" in " ".join(capsys.readouterr().out.split())

    @pytest.mark.parametrize("answer", ["copy", "move"])
    def test_a_duplicate_is_never_copied_or_moved_out_of_the_store(
            self, tmp_path, isolated_home, answer, capsys):
        data = self._store(tmp_path)
        assert add_local(str(data)) is True
        before = {p for p in (data / "llamacpp").rglob("*.gguf")}
        assert add_local(str(data), on_duplicate=answer) is False
        assert {p for p in (data / "llamacpp").rglob("*.gguf")} == before
        assert not list(isolated_home.joinpath("models").glob("model*.gguf"))
        assert "Cannot" in " ".join(capsys.readouterr().out.split())

    def test_a_duplicate_under_ask_is_skipped_with_a_note(
            self, tmp_path, isolated_home, capsys):
        data = self._store(tmp_path)
        assert add_local(str(data)) is True
        capsys.readouterr()
        assert add_local(str(data), on_duplicate="ask") is True
        assert "registered in place" in " ".join(capsys.readouterr().out.split())
        assert len(load_registry()) == 2

    def test_quoted_yaml_values_are_unquoted(self, tmp_path, isolated_home):
        data = tmp_path / "Jan" / "data"
        _jan_unit(data, "org/quoted", yml='model_path: "llamacpp/models/org/quoted/w.gguf"\n')
        (data / "llamacpp" / "models" / "org" / "quoted" / "w.gguf").write_bytes(_gguf(tag="q"))
        assert add_local(str(data)) is True
        assert Path(load_registry()["org-quoted"]["path"]).name == "w.gguf"

    def test_a_bad_projector_leaves_the_model_without_one(
            self, tmp_path, isolated_home, capsys):
        data = tmp_path / "Jan" / "data"
        _jan_unit(data, "org/m", weights=_gguf(tag="m"), projector=b"not a gguf")
        assert add_local(str(data)) is True
        assert not load_registry()["org-m"].get("mmproj")
        assert "not a GGUF vision projector" in " ".join(capsys.readouterr().out.split())

    def test_copy_and_move_are_refused(self, tmp_path, isolated_home, capsys):
        data = self._store(tmp_path)
        for store in ("copy", "move"):
            assert add_local(str(data), store=store) is False
        assert load_registry() == {}
        assert "named model.gguf" in " ".join(capsys.readouterr().out.split())

    def test_a_llamacpp_folder_without_model_yml_is_an_ordinary_folder(
            self, tmp_path, isolated_home):
        models = tmp_path / "x" / "llamacpp" / "models"
        models.mkdir(parents=True)
        (models / "Plain.gguf").write_bytes(_gguf(tag="p"))
        assert add_local(str(tmp_path / "x")) is True
        assert set(load_registry()) == {"Plain"}


class TestGpt4allFolder:
    def _folder(self, tmp_path) -> Path:
        folder = tmp_path / "GPT4All"
        folder.mkdir()
        (folder / "Meta-Llama-3-8B-Instruct.Q4_0.gguf").write_bytes(_gguf(tag="llama"))
        (folder / "nomic-embed-text-v1.5.f16.gguf").write_bytes(_gguf("nomic-bert", "emb"))
        (folder / "gpt4all-key.rmodel").write_text('{"apiKey": "secret"}', encoding="utf-8")
        (folder / "test_write.txt").write_text("x", encoding="utf-8")
        return folder

    def test_complete_models_register_and_the_rmodel_file_is_left_alone(self, tmp_path, isolated_home):
        folder = self._folder(tmp_path)
        assert add_local(str(folder)) is True
        reg = load_registry()
        assert set(reg) == {"Meta-Llama-3-8B-Instruct.Q4_0", "nomic-embed-text-v1.5.f16"}
        assert reg["nomic-embed-text-v1.5.f16"]["model_type"] == "embedding"

    def test_a_file_named_incomplete_is_skipped_with_a_reason(
            self, tmp_path, isolated_home, capsys):
        folder = self._folder(tmp_path)
        (folder / "incomplete-Phi-3-mini.Q4_0.gguf").write_bytes(_gguf(tag="part"))
        assert add_local(str(folder)) is True
        assert "incomplete-Phi-3-mini.Q4_0" not in load_registry()
        assert "starts with 'incomplete'" in " ".join(capsys.readouterr().out.split())

    def test_a_file_named_incomplete_still_registers_when_added_by_its_own_path(
            self, tmp_path, isolated_home):
        folder = self._folder(tmp_path)
        target = folder / "incomplete-Phi-3-mini.Q4_0.gguf"
        target.write_bytes(_gguf(tag="part"))
        assert add_local(str(target)) is True
        assert "incomplete-Phi-3-mini.Q4_0" in load_registry()


class TestUnfinishedCopyIsNotRegistered:
    def _truncated(self, folder: Path) -> Path:
        folder.mkdir(exist_ok=True)
        f = folder / "Half.Q4_0.gguf"
        f.write_bytes(_gguf(tensor_offset=50_000_000))
        return f

    def test_a_folder_import_skips_a_header_that_declares_data_past_the_end(
            self, tmp_path, isolated_home, capsys):
        folder = tmp_path / "models-in"
        self._truncated(folder)
        (folder / "Whole.Q4_0.gguf").write_bytes(_gguf(tensor_offset=64) + b"\0" * 4096)
        assert add_local(str(folder)) is True
        assert set(load_registry()) == {"Whole.Q4_0"}
        assert "not finished copying" in " ".join(capsys.readouterr().out.split())
