# SPDX-License-Identifier: AGPL-3.0-or-later
"""ADR-0009 P17: the store-by-copy path hashes a model TWICE, silently."""

import json

import pytest

from localm import model_manager as mm
from localm.model_manager import _shared


def _events(capsys):
    """Progress payloads emitted so far."""
    out = capsys.readouterr().out
    return [json.loads(line.split(mm.PROGRESS_SENTINEL, 1)[1])
            for line in out.splitlines() if mm.PROGRESS_SENTINEL in line]


@pytest.fixture()
def gui(monkeypatch):
    monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")


class TestTheRelocationThatMakesThisPossible:
    def test_the_emitters_live_where_registry_can_import_them(self):
        """`registry` cannot import from `pull` (pull imports registry), so if these ever move back the copy path silently loses its reporting or the package stops importing."""
        assert _shared._verify_digest.__module__ == "localm.model_manager._shared"
        assert _shared._emit_progress.__module__ == "localm.model_manager._shared"

    def test_shared_imports_nothing_from_its_own_package(self):
        """What makes _shared safe for both sides."""
        src = open(_shared.__file__, encoding="utf-8").read()
        for bad in ("from .pull import", "from .registry import",
                    "from .gguf import"):
            assert bad not in src, f"_shared.py reintroduced a package import: {bad}"


class TestBothHashesReport:
    def test_a_copy_reports_two_separate_verify_runs(self, gui, tmp_path,
                                                     monkeypatch, capsys):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        models = tmp_path / "models"
        models.mkdir()
        gguf = src_dir / "m.gguf"
        gguf.write_bytes(b"GGUF" + b"\0" * 4096)

        monkeypatch.setattr(mm, "MODELS_DIR", models, raising=False)
        monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **kw: True,
                            raising=False)
        monkeypatch.setattr(mm, "_HASH_BLOCK_BYTES", 64, raising=False)

        from localm.model_manager import registry as _reg
        _reg._store_into_models_dir(gguf, "copy")

        evs = [e for e in _events(capsys) if e.get("phase") == "verify"]
        assert evs, "the copy path hashed twice and reported nothing"

        # Two runs, not one: each ends at the file size, so the count of
        # terminal events is the count of hashes that reported.
        terminals = [e for e in evs if e["downloaded"] == e["total"] != 0]
        assert len(terminals) >= 2, (
            f"only one hash reported; the other ran silent: {evs}")

    def test_the_copy_still_verifies_and_still_lands(self, gui, tmp_path,
                                                     monkeypatch, capsys):
        """Reporting must not change the outcome."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        models = tmp_path / "models"
        models.mkdir()
        gguf = src_dir / "m.gguf"
        gguf.write_bytes(b"GGUF" + b"\0" * 4096)

        monkeypatch.setattr(mm, "MODELS_DIR", models, raising=False)
        monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **kw: True,
                            raising=False)

        from localm.model_manager import registry as _reg
        dest = _reg._store_into_models_dir(gguf, "copy")
        _events(capsys)                      # drain

        assert dest.exists() and dest.read_bytes() == gguf.read_bytes()

    def test_a_corrupted_copy_is_still_refused(self, gui, tmp_path, monkeypatch,
                                               capsys):
        """The negative case, without which the test above passes for any implementation that copies at all."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        models = tmp_path / "models"
        models.mkdir()
        gguf = src_dir / "m.gguf"
        gguf.write_bytes(b"GGUF" + b"\0" * 4096)

        monkeypatch.setattr(mm, "MODELS_DIR", models, raising=False)
        monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **kw: True,
                            raising=False)

        from localm.model_manager import registry as _reg
        real = _shared._verify_digest
        calls = {"n": 0}

        def _second_call_lies(path, **kw):
            calls["n"] += 1
            if calls["n"] == 2:              # the post-copy hash
                return "0" * 64
            return real(path, **kw)

        monkeypatch.setattr(_reg, "_verify_digest", _second_call_lies)

        with pytest.raises(RuntimeError):
            _reg._store_into_models_dir(gguf, "copy")
        _events(capsys)                      # drain

        assert calls["n"] == 2, "the fixture never reached the post-copy hash"
        assert not (models / "m.gguf").exists(), (
            "a copy that failed its integrity check was left on disk")
