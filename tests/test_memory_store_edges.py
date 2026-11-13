# SPDX-License-Identifier: AGPL-3.0-or-later
"""Edge cases of the memory store, its routes and its CLI: a records file with
bytes that are not UTF-8, a switched embedding model, an empty edit, a fact added
twice, the legacy chat-memory.md after a clear, and the CLI's own checks."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi import HTTPException

from localm.memory import MemoryRecord, MemoryStore
from localm.plugins.builtin.memory import plug
from tests._shared_model_guard import (  # noqa: F401
    no_embedder_download,
    no_shared_model_left_resident,
)


@pytest.fixture
def memhome(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setattr(plug, "_persist_enabled", lambda: True)
    monkeypatch.setattr(plug, "_embed_fn", lambda: None)
    monkeypatch.setenv("LOCALM_MODE", "log")
    return tmp_path


@pytest.fixture()
def clihome(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    h.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setenv("LOCALM_MODE", "log")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", h)
    monkeypatch.setattr(_cfg, "home_dir", lambda: h)
    monkeypatch.setattr(plug, "_home", lambda: h)
    return h


def _run(*args):
    from localm.plugins.builtin.memory.cli import main
    return CliRunner().invoke(main, list(args), catch_exceptions=False)


def _store(root) -> MemoryStore:
    return MemoryStore("owner", "chat", root=root / "memory")


def test_a_line_that_is_not_utf8_is_kept_byte_for_byte(tmp_path):
    s = _store(tmp_path)
    s.add(MemoryRecord(text="User likes tea", source="user"))
    bad = b'{"v": 1, "id": "x1", "text": "caf\xe9 au lait"}'
    with open(s.path, "ab") as fh:
        fh.write(bad + b"\n")
    again = _store(tmp_path)
    assert [r.text for r in again.all()] == ["User likes tea"]
    again.add(MemoryRecord(text="User lives in Graz", source="user"))
    raw = s.path.read_bytes()
    lines = [ln.rstrip(b"\r") for ln in raw.split(b"\n")]
    assert bad in lines, "the undecodable line is written back unchanged"
    assert sorted(r.text for r in _store(tmp_path).all()) == [
        "User likes tea", "User lives in Graz"]


def test_a_query_vector_of_another_width_reads_as_a_changed_model(tmp_path):
    s = _store(tmp_path)
    for t in ("tea", "coffee", "water", "juice", "milk", "soda"):
        s.add(MemoryRecord(text=f"User drinks {t}", source="user"),
              embed_fn=lambda texts: [[1.0, 0.0, 0.0] for _ in texts])
    diag: dict = {}
    s._vector_relevance("drinks", lambda texts: [[1.0, 0.0] for _ in texts],
                        diagnostics=diag)
    assert diag["degrade_reason"] == "dim_mismatch"
    diag = {}
    s._vector_relevance("drinks", lambda texts: [[]], diagnostics=diag)
    assert diag["degrade_reason"] == "query_embed_failed"


def test_an_empty_edit_is_refused_and_changes_nothing(tmp_path, memhome):
    s = _store(memhome)
    rec = s.add(MemoryRecord(text="User likes tea", source="user"))
    with pytest.raises(ValueError):
        s.update(rec.id, text="   ")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(plug.memory_patch(rec.id, plug.MemoryPatch(text=" - "), None))
    assert exc.value.status_code == 400
    assert [r.text for r in _store(memhome).all()] == ["User likes tea"]


def test_remember_twice_stores_one_fact(memhome):
    first = asyncio.run(plug.memory_append(plug.MemoryAppend(text="User likes tea"), None))
    assert first["status"] == "appended"
    again = asyncio.run(plug.memory_append(plug.MemoryAppend(text="  user LIKES   tea"), None))
    assert again == {"status": "exists", "id": first["id"]}
    assert [r.text for r in _store(memhome).all()] == ["User likes tea"]


def test_a_fact_longer_than_the_cap_is_still_found_the_second_time(memhome):
    long_fact = "User likes " + "tea " * 200
    for _ in range(2):
        asyncio.run(plug.memory_append(plug.MemoryAppend(text=long_fact), None))
    assert len(_store(memhome).all()) == 1


def test_adding_a_synthesised_fact_yourself_makes_it_yours(memhome):
    s = _store(memhome)
    synth = s.add(MemoryRecord(text="User prefers Zig", source="synth", importance=0.4))
    out = asyncio.run(plug.memory_append(plug.MemoryAppend(text="user prefers zig"), None))
    assert out == {"status": "appended", "id": synth.id}
    (rec,) = _store(memhome).all()
    assert rec.source == "user" and rec.importance == 0.8


def test_a_record_with_undecodable_text_is_refused_not_hidden(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(UnicodeEncodeError):
        s.add(MemoryRecord(text="a\udc80b", source="user"))
    assert _store(tmp_path).all() == []
    assert not s.path.exists() or b"\x80" not in s.path.read_bytes()


def test_unreadable_lines_count_as_something_to_clear(clihome):
    from localm.plugins.builtin.memory.cli import _store as cli_store
    path = cli_store().path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'{"text": "caf\xe9"}\n')
    assert "unreadable record lines" in cli_store().remnants()
    res = _run("clear", "--yes")
    assert res.exit_code == 0, res.output
    assert "Nothing to clear" not in res.output
    assert cli_store().remnants() == []


def test_the_legacy_file_is_not_shown_once_imported(memhome):
    (memhome / "chat-memory.md").write_text("- User likes tea\n", encoding="utf-8")
    s = _store(memhome)
    assert "User likes tea" in plug._rendered_text(s)
    plug._legacy_marker(s).parent.mkdir(parents=True, exist_ok=True)
    plug._legacy_marker(s).write_text("1", encoding="utf-8")
    assert plug._rendered_text(s) == ""


def test_clear_erases_the_legacy_file_too(clihome):
    legacy = clihome / "chat-memory.md"
    legacy.write_text("- User likes tea\n", encoding="utf-8")
    res = _run("clear", "--yes")
    assert res.exit_code == 0, res.output
    assert not legacy.exists()


def test_cli_add_refuses_an_importance_outside_0_to_1(clihome):
    for bad in ("5", "nan"):
        res = _run("add", "User likes tea", "--importance", bad)
        assert res.exit_code != 0, bad
        assert "0" in res.output and "1" in res.output


def test_cli_add_twice_says_it_is_already_there(clihome):
    assert _run("add", "User likes tea").exit_code == 0
    res = _run("add", "user likes TEA")
    assert res.exit_code == 0
    assert "Already in memory" in res.output
    from localm.plugins.builtin.memory.cli import _store as cli_store
    assert len(cli_store().all()) == 1


def test_forget_of_a_fact_another_process_removed_says_so(clihome, monkeypatch):
    from localm.plugins.builtin.memory import cli
    rec = cli._store().add(MemoryRecord(text="User likes tea", source="user"))
    real = cli._store

    def racing_store():
        s = real()
        orig = s.delete

        def delete(mem_id):
            real().delete(mem_id)
            return orig(mem_id)
        s.delete = delete
        return s

    monkeypatch.setattr(cli, "_store", racing_store)
    res = _run("forget", rec.id, "--yes")
    assert res.exit_code == 0, res.output
    assert "no longer in memory" in res.output
