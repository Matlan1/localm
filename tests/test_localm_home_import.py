# SPDX-License-Identifier: AGPL-3.0-or-later
"""Importing another localm install's home reuses the sha256 its registry.json
recorded instead of hashing every file again.

The foreign registry is untrusted input: a bad entry, a path outside the
file's identity, a stale file or a corrupt file must fall back to hashing,
never crash and never register a wrong digest silently for a file that changed.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

import localm.model_manager as mm
from localm.config import load_registry
from localm.model_manager import add_local
from localm.model_manager.localm_home import (
    RecordedDigests, find_localm_home, recorded_digests)
from localm.model_manager.registry import _entry_path


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    models = home / "models"
    models.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", models)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setattr(mm, "MODELS_DIR", models)
    monkeypatch.setattr(mm, "HOME_DIR", home)
    return home


@pytest.fixture
def hash_calls(monkeypatch):
    """Every file the real hasher is asked to read, in order."""
    real = mm._hash_with_progress
    calls: list[str] = []

    def counting(path, *args, **kwargs):
        calls.append(Path(path).name)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(mm, "_hash_with_progress", counting)
    return calls


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _content(name: str) -> bytes:
    return b"GGUF\x00\x00\x00\x00" + name.encode()


def _other_home(tmp_path, *names, registry_mtime_offset=60):
    """A localm home holding *names* as registered GGUFs. Returns
    ``(home, {name: path})``. The registry is stamped *registry_mtime_offset*
    seconds after the files, as if written after them."""
    home = tmp_path / "other"
    models = home / "models"
    models.mkdir(parents=True)
    reg = {}
    paths = {}
    for n in names:
        f = models / f"{n}.gguf"
        f.write_bytes(_content(n))
        paths[n] = f
        reg[n] = {"path": str(f.resolve()), "source": "local", "model_type": "llm",
                  "sha256": _sha(_content(n))}
    _write_registry(home, reg, after=list(paths.values()), offset=registry_mtime_offset)
    return home, paths


def _write_registry(home, reg, after=(), offset=60):
    rf = home / "registry.json"
    rf.write_text(json.dumps(reg), encoding="utf-8")
    newest = max([os.stat(p).st_mtime for p in after] or [os.stat(rf).st_mtime])
    os.utime(rf, (newest + offset, newest + offset))


class TestReuse:
    def test_matching_file_reuses_recorded_digest_without_reading_it(
            self, tmp_path, isolated_home, hash_calls):
        home, _ = _other_home(tmp_path, "alpha", "beta")
        assert add_local(str(home)) is True
        reg = load_registry()
        assert hash_calls == []
        assert reg["alpha"]["sha256"] == _sha(_content("alpha"))
        assert reg["beta"]["sha256"] == _sha(_content("beta"))

    def test_pointing_at_models_folder_reuses_too(self, tmp_path, isolated_home, hash_calls):
        home, _ = _other_home(tmp_path, "alpha")
        assert add_local(str(home / "models")) is True
        assert hash_calls == []
        assert load_registry()["alpha"]["sha256"] == _sha(_content("alpha"))

    def test_status_line_says_the_hash_was_reused(self, tmp_path, isolated_home, monkeypatch):
        lines: list[str] = []
        monkeypatch.setattr(mm, "_import_status",
                            lambda text, **kw: lines.append(text))
        home, _ = _other_home(tmp_path, "alpha")
        assert add_local(str(home)) is True
        assert any("Using the hash recorded by the other install" in t for t in lines)

    def test_recorded_digest_is_trusted_even_when_it_is_wrong(
            self, tmp_path, isolated_home, hash_calls):
        # Trust decision: the other install's record is taken as the file's hash;
        # a record that disagrees with the bytes is registered as recorded.
        home, paths = _other_home(tmp_path, "alpha")
        wrong = "ab" * 32
        reg = json.loads((home / "registry.json").read_text())
        reg["alpha"]["sha256"] = wrong
        _write_registry(home, reg, after=list(paths.values()))
        assert add_local(str(home)) is True
        assert hash_calls == []
        assert load_registry()["alpha"]["sha256"] == wrong
        assert wrong != _sha(_content("alpha"))

    def test_uppercase_digest_is_stored_lowercase(self, tmp_path, isolated_home, hash_calls):
        home, paths = _other_home(tmp_path, "alpha")
        reg = json.loads((home / "registry.json").read_text())
        reg["alpha"]["sha256"] = reg["alpha"]["sha256"].upper()
        _write_registry(home, reg, after=list(paths.values()))
        assert add_local(str(home)) is True
        assert load_registry()["alpha"]["sha256"] == _sha(_content("alpha"))


class TestFallsBackToHashing:
    def test_file_changed_after_registry_was_written_is_hashed(
            self, tmp_path, isolated_home, hash_calls):
        home, paths = _other_home(tmp_path, "alpha", "beta")
        new = _content("alpha") + b"-edited-later"
        paths["alpha"].write_bytes(new)
        st = os.stat(home / "registry.json")
        os.utime(paths["alpha"], (st.st_mtime + 30, st.st_mtime + 30))
        assert add_local(str(home)) is True
        reg = load_registry()
        assert hash_calls == ["alpha.gguf"]
        assert reg["alpha"]["sha256"] == _sha(new)
        assert reg["beta"]["sha256"] == _sha(_content("beta"))

    def test_file_not_in_registry_is_hashed(self, tmp_path, isolated_home, hash_calls):
        home, paths = _other_home(tmp_path, "alpha")
        extra = home / "models" / "extra.gguf"
        extra.write_bytes(_content("extra"))
        os.utime(extra, (1, 1))
        assert add_local(str(home)) is True
        assert hash_calls == ["extra.gguf"]
        assert load_registry()["extra"]["sha256"] == _sha(_content("extra"))

    def test_entry_without_digest_is_hashed(self, tmp_path, isolated_home, hash_calls):
        home, paths = _other_home(tmp_path, "alpha")
        reg = json.loads((home / "registry.json").read_text())
        del reg["alpha"]["sha256"]
        _write_registry(home, reg, after=list(paths.values()))
        assert add_local(str(home)) is True
        assert hash_calls == ["alpha.gguf"]

    @pytest.mark.parametrize("bad", ["", "zz" * 32, "ab" * 31, "ab" * 33, 123, None, ["ab" * 32]],
                             ids=["empty", "nonhex", "short", "long", "int", "null", "list"])
    def test_malformed_digest_is_hashed(self, tmp_path, isolated_home, hash_calls, bad):
        home, paths = _other_home(tmp_path, "alpha")
        reg = json.loads((home / "registry.json").read_text())
        reg["alpha"]["sha256"] = bad
        _write_registry(home, reg, after=list(paths.values()))
        assert add_local(str(home)) is True
        assert hash_calls == ["alpha.gguf"]
        assert load_registry()["alpha"]["sha256"] == _sha(_content("alpha"))

    def test_two_entries_for_one_file_with_different_digests_are_hashed(
            self, tmp_path, isolated_home, hash_calls):
        home, paths = _other_home(tmp_path, "alpha")
        reg = json.loads((home / "registry.json").read_text())
        reg["alias"] = dict(reg["alpha"], sha256="cd" * 32)
        _write_registry(home, reg, after=list(paths.values()))
        assert add_local(str(home)) is True
        assert hash_calls == ["alpha.gguf"]
        assert load_registry()["alpha"]["sha256"] == _sha(_content("alpha"))

    def test_junk_entries_do_not_crash_and_good_ones_still_count(
            self, tmp_path, isolated_home, hash_calls):
        home, paths = _other_home(tmp_path, "alpha")
        reg = json.loads((home / "registry.json").read_text())
        reg.update({
            "n1": None, "n2": 5, "n3": "text", "n4": [], "n5": {},
            "n6": {"path": None, "sha256": "ab" * 32},
            "n7": {"path": 12, "sha256": "ab" * 32},
            "n8": {"path": "", "sha256": "ab" * 32},
            "n9": {"path": "relative/x.gguf", "sha256": "ab" * 32},
        })
        _write_registry(home, reg, after=list(paths.values()))
        assert add_local(str(home)) is True
        assert hash_calls == []
        assert load_registry()["alpha"]["sha256"] == _sha(_content("alpha"))

    def test_path_traversal_entry_cannot_vouch_for_another_file(
            self, tmp_path, isolated_home, hash_calls):
        home, paths = _other_home(tmp_path, "alpha", "beta")
        reg = json.loads((home / "registry.json").read_text())
        beta = paths["beta"]
        reg["beta"]["path"] = str(beta.parent / ".." / "models" / "beta.gguf")
        reg["beta"]["sha256"] = "ef" * 32
        _write_registry(home, reg, after=list(paths.values()))
        assert add_local(str(home)) is True
        assert hash_calls == ["beta.gguf"]
        assert load_registry()["beta"]["sha256"] == _sha(_content("beta"))

    def test_entry_naming_a_different_directory_does_not_apply(
            self, tmp_path, isolated_home, hash_calls):
        home, paths = _other_home(tmp_path, "alpha")
        reg = json.loads((home / "registry.json").read_text())
        reg["alpha"]["path"] = str(tmp_path / "elsewhere" / "alpha.gguf")
        _write_registry(home, reg, after=list(paths.values()))
        assert add_local(str(home)) is True
        assert hash_calls == ["alpha.gguf"]

    @pytest.mark.parametrize("text", ["", "{not json", "[1, 2]", '"x"', "null", "\x00\x01"],
                             ids=["empty", "truncated", "list", "string", "null", "binary"])
    def test_unreadable_registry_falls_back_and_says_so(
            self, tmp_path, isolated_home, hash_calls, capsys, text):
        home, paths = _other_home(tmp_path, "alpha")
        (home / "registry.json").write_text(text, encoding="utf-8")
        assert add_local(str(home)) is True
        assert hash_calls == ["alpha.gguf"]
        assert load_registry()["alpha"]["sha256"] == _sha(_content("alpha"))
        said = "".join(capsys.readouterr().out.split())
        assert "Foundanotherlocalminstall" in said
        assert "hashingitsfilesinstead" in said

    def test_oversized_registry_falls_back(self, tmp_path, isolated_home, hash_calls, monkeypatch):
        import localm.model_manager.localm_home as lh
        home, paths = _other_home(tmp_path, "alpha")
        monkeypatch.setattr(lh, "_REGISTRY_MAX_BYTES", 10)
        assert add_local(str(home)) is True
        assert hash_calls == ["alpha.gguf"]

    def test_deeply_nested_registry_falls_back(self, tmp_path, isolated_home, hash_calls):
        home, paths = _other_home(tmp_path, "alpha")
        (home / "registry.json").write_text("[" * 200000 + "]" * 200000, encoding="utf-8")
        assert add_local(str(home)) is True
        assert hash_calls == ["alpha.gguf"]

    def test_nul_and_relative_stored_paths_are_ignored(self, tmp_path):
        home = tmp_path / "h"
        (home / "models").mkdir(parents=True)
        (home / "registry.json").write_text(json.dumps({
            "a": {"path": "/x/a\u0000.gguf", "sha256": "ab" * 32},
            "b": {"path": "models/b.gguf", "sha256": "ab" * 32},
            "c": {"path": "b.gguf", "sha256": "ab" * 32},
        }))
        recorded, problem = recorded_digests(home, _entry_path)
        assert problem is None
        assert len(recorded) == 0

    def test_registry_that_is_a_directory_falls_back(self, tmp_path, isolated_home, hash_calls):
        home = tmp_path / "other"
        (home / "models").mkdir(parents=True)
        (home / "models" / "alpha.gguf").write_bytes(_content("alpha"))
        (home / "registry.json").mkdir()
        assert find_localm_home(home) is None
        assert add_local(str(home)) is True
        assert hash_calls == ["alpha.gguf"]


class TestScopeOfReuse:
    def test_no_hash_flag_is_unchanged(self, tmp_path, isolated_home, hash_calls):
        home, _ = _other_home(tmp_path, "alpha")
        assert add_local(str(home), no_hash=True) is True
        assert hash_calls == []
        assert "sha256" not in load_registry()["alpha"]

    def test_fast_flag_is_unchanged(self, tmp_path, isolated_home, hash_calls):
        home, _ = _other_home(tmp_path, "alpha")
        assert add_local(str(home), fast=True) is True
        assert hash_calls == []
        assert "sha256" not in load_registry()["alpha"]

    def test_plain_folder_without_a_registry_still_hashes(
            self, tmp_path, isolated_home, hash_calls):
        d = tmp_path / "drop"
        d.mkdir()
        (d / "alpha.gguf").write_bytes(_content("alpha"))
        assert add_local(str(d)) is True
        assert hash_calls == ["alpha.gguf"]

    def test_store_copy_still_verifies_by_hashing(self, tmp_path, isolated_home, hash_calls):
        home, _ = _other_home(tmp_path, "alpha")
        assert add_local(str(home), store="copy") is True
        assert hash_calls
        assert load_registry()["alpha"]["sha256"] == _sha(_content("alpha"))


class TestUnits:
    def test_find_localm_home_needs_registry_and_models(self, tmp_path):
        a = tmp_path / "a"
        (a / "models").mkdir(parents=True)
        assert find_localm_home(a) is None
        (a / "registry.json").write_text("{}")
        assert find_localm_home(a) == a
        assert find_localm_home(a / "models") == a
        assert find_localm_home(tmp_path) is None

    @pytest.mark.skipif(os.name != "nt", reason="UNC paths are a Windows shape")
    def test_network_path_entry_is_never_resolved(self, tmp_path, monkeypatch):
        home = tmp_path / "h"
        (home / "models").mkdir(parents=True)
        (home / "registry.json").write_text(json.dumps({
            "x": {"path": r"\\attacker.invalid\share\x.gguf", "sha256": "ab" * 32},
            "y": {"path": "//attacker.invalid/share/y.gguf", "sha256": "ab" * 32},
        }))
        resolved: list = []
        real = Path.resolve
        monkeypatch.setattr(Path, "resolve",
                            lambda self, *a, **k: resolved.append(str(self)) or real(self, *a, **k))
        recorded, problem = recorded_digests(home, _entry_path)
        assert problem is None
        assert len(recorded) == 0
        assert resolved == []

    def test_lookup_rejects_empty_file(self, tmp_path):
        f = tmp_path / "m.gguf"
        f.write_bytes(b"")
        from localm.model_manager.localm_home import _digest_key
        rec = RecordedDigests({_digest_key(f): "ab" * 32}, written_at=f.stat().st_mtime + 10)
        assert rec.lookup(f) is None
        f.write_bytes(b"x")
        os.utime(f, (1, 1))
        assert rec.lookup(f) == "ab" * 32
