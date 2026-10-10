# SPDX-License-Identifier: AGPL-3.0-or-later
"""State, marker, lock and model-directory JSON files that a user, a pulled model
or a crashed process can leave over-nested or with an over-long integer: each
reader falls back exactly as it does for any other unreadable file, instead of
raising ``RecursionError`` / the integer digit-limit ``ValueError``."""

from __future__ import annotations

import json

import pytest


from localm import auth, instances, sessions
from localm.inference.backends import _hf_fp8
from localm.media import managed_comfy, managed_comfy_fresh, managed_comfy_update
from localm.model_manager import pull, unsupported
from localm import cpu_backend_select
from localm.rag import collection_lock
from localm.setup_llama import rocm_cpu, runtime_dir
from tests._hostile_json import HOSTILE


# ------------------------------------------------------------------- auth

@HOSTILE
def test_hostile_keystore_reads_as_empty(doc, tmp_path, monkeypatch):
    path = tmp_path / "auth.json"
    path.write_text(doc, encoding="utf-8")
    monkeypatch.setattr(auth, "keystore_file", lambda: path)
    assert auth._load_keystore() == []


@HOSTILE
def test_hostile_keystore_counts_as_configured_so_auth_fails_closed(
        doc, tmp_path, monkeypatch):
    path = tmp_path / "auth.json"
    path.write_text(doc, encoding="utf-8")
    monkeypatch.setattr(auth, "keystore_file", lambda: path)
    assert auth._keystore_configured() is True


@HOSTILE
def test_hostile_owner_kdf_file_reads_as_no_records(doc, tmp_path, monkeypatch):
    path = tmp_path / "auth.kdf.json"
    path.write_text(doc, encoding="utf-8")
    monkeypatch.setattr(auth, "owner_kdf_file", lambda: path)
    assert auth._load_owner_kdf() == []


# -------------------------------------------------------------- instances

@HOSTILE
def test_hostile_registry_entry_is_unreadable_not_fatal(doc, tmp_path):
    path = tmp_path / "entry.json"
    path.write_text(doc, encoding="utf-8")
    assert instances.read_entry(path) is None


@HOSTILE
def test_hostile_registry_entry_does_not_hide_the_others(doc, tmp_path):
    run = instances.run_dir(tmp_path)
    run.mkdir(parents=True)
    (run / "bad.json").write_text(doc, encoding="utf-8")
    (run / "good.json").write_text(json.dumps({"pid": 1}), encoding="utf-8")
    assert [e["pid"] for e in instances.list_entries(tmp_path)] == [1]


# ------------------------------------------------------------------- pull

@HOSTILE
def test_hostile_partial_owner_record_is_unreadable(doc, tmp_path):
    partial = tmp_path / "model.gguf.part"
    pull._partial_owner_path(partial).write_text(doc, encoding="utf-8")
    assert pull._read_partial_owner(partial) is None


@HOSTILE
def test_hostile_lock_record_is_unreadable(doc, tmp_path):
    (tmp_path / "owner.json").write_text(doc, encoding="utf-8")
    assert pull._read_lock_record(tmp_path) is None


@HOSTILE
def test_hostile_part_record_is_not_resumable(doc, tmp_path):
    part = tmp_path / "model.gguf.part"
    part.write_bytes(b"x")
    pull._part_record_path(part).write_text(doc, encoding="utf-8")
    assert pull._part_is_resumable(part, {"size": 1}) is False


# ------------------------------------------------- model directories / hf

@HOSTILE
def test_hostile_config_json_is_not_a_readable_config(doc, tmp_path):
    (tmp_path / "config.json").write_text(doc, encoding="utf-8")
    assert unsupported._read_config(tmp_path) is None


@HOSTILE
def test_hostile_config_json_has_no_fp8_quant_method(doc, tmp_path):
    (tmp_path / "config.json").write_text(doc, encoding="utf-8")
    assert _hf_fp8.quant_method(str(tmp_path)) is None


@HOSTILE
def test_hostile_safetensors_index_names_no_weight_files(doc, tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(doc, encoding="utf-8")
    assert _hf_fp8.weight_files(str(tmp_path)) == []


# --------------------------------------------------- locks and markers

@HOSTILE
def test_hostile_rag_lock_record_is_held_not_free(doc, tmp_path):
    lock = tmp_path / "lock.json"
    lock.write_text(doc, encoding="utf-8")
    rec, mtime = collection_lock._read_record(lock)
    assert rec is None and mtime is not None


@HOSTILE
def test_hostile_cpu_overlay_marker_is_no_overlay(doc, tmp_path):
    (tmp_path / rocm_cpu.CPU_OVERLAY_MARKER).write_text(doc, encoding="utf-8")
    assert rocm_cpu.installed_cpu_overlay(tmp_path) is None


@HOSTILE
def test_hostile_provision_lock_owner_has_no_holder_pid(doc, tmp_path):
    (tmp_path / runtime_dir._PROVISION_LOCK_OWNER).write_text(doc, encoding="utf-8")
    assert runtime_dir._provision_lock_holder_pid(tmp_path) is None


@HOSTILE
def test_hostile_comfy_lock_owner_reads_as_unknown(doc, tmp_path):
    (tmp_path / managed_comfy._LOCK_OWNER).write_text(doc, encoding="utf-8")
    assert managed_comfy._lock_holder(tmp_path) == (None, "update")


@HOSTILE
def test_hostile_comfy_workflow_contributes_no_class_types(doc, tmp_path):
    workflow = tmp_path / "wf.json"
    workflow.write_text(doc, encoding="utf-8")
    assert managed_comfy_fresh._class_types_in(workflow) == set()


@HOSTILE
def test_hostile_comfy_marker_is_replaced_by_the_update_record(doc, tmp_path):
    marker = tmp_path / managed_comfy_update.MARKER_FILENAME
    marker.write_text(doc, encoding="utf-8")
    managed_comfy_update._update_marker(tmp_path, "abc123", "9.9", "prev", [])
    written = json.loads(marker.read_text(encoding="utf-8"))
    assert written["commit"] == "abc123" and written["stage"] == "S4"


@pytest.fixture
def session_store(tmp_path, monkeypatch):
    path = tmp_path / "sessions.json"
    monkeypatch.setattr(sessions, "sessions_file", lambda: path)
    monkeypatch.setitem(sessions._CACHE, "mtime", None)
    monkeypatch.setitem(sessions._CACHE, "records", None)
    return path


@HOSTILE
def test_hostile_session_store_refuses_every_session(doc, session_store):
    session_store.write_text(doc, encoding="utf-8")
    assert sessions.lookup("some-session-id") is None


@HOSTILE
def test_hostile_session_store_does_not_break_logout(doc, session_store):
    session_store.write_text(doc, encoding="utf-8")
    assert sessions.revoke("some-session-id") is None
    assert session_store.read_text(encoding="utf-8") == doc


@HOSTILE
def test_hostile_safetensors_shard_header_has_no_size(doc, tmp_path):
    body = doc.encode("utf-8")
    (tmp_path / "model.safetensors").write_bytes(
        len(body).to_bytes(8, "little") + body)
    assert _hf_fp8.expanded_bf16_bytes(str(tmp_path)) is None


@HOSTILE
def test_hostile_cpu_tier_lock_owner_does_not_stop_selection(
        doc, tmp_path, monkeypatch):
    monkeypatch.setattr(cpu_backend_select, "_LOCK_WAIT_SECONDS", 0.3)
    monkeypatch.setattr(cpu_backend_select, "_LOCK_POLL_SECONDS", 0.05)
    lock = tmp_path / cpu_backend_select._LOCK_NAME
    lock.mkdir()
    (lock / cpu_backend_select._LOCK_OWNER_FILE).write_text(doc, encoding="utf-8")
    with cpu_backend_select._lock(tmp_path) as acquired:
        assert acquired is False
    assert lock.is_dir()
