# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression pin for #621: a managed-ComfyUI git clone must not fail with Windows' legacy 260-char MAX_PATH limit."""

from __future__ import annotations


from localm.media import managed_comfy_fresh as fresh
from localm.media import managed_comfy_provision as prov


def _record_run(monkeypatch, target_module):
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return True, ""

    monkeypatch.setattr(target_module, "_run", _fake_run)
    return calls


def test_clone_at_commit_uses_longpaths(monkeypatch, tmp_path):
    calls = _record_run(monkeypatch, fresh)
    fresh._clone_at_commit("https://example/repo.git", tmp_path / "dest",
                          "abc123", on_progress=None)
    clone_call = calls[0]
    assert clone_call[0] == "git"
    assert clone_call[1:3] == ["-c", "core.longpaths=true"]
    assert "clone" in clone_call


def test_clone_source_uses_longpaths(monkeypatch, tmp_path):
    calls = _record_run(monkeypatch, prov)
    prov._clone_source(tmp_path / "user_workdir", tmp_path / "managed",
                       "abc123", on_progress=None)
    clone_call = calls[0]
    assert clone_call[0] == "git"
    assert clone_call[1:3] == ["-c", "core.longpaths=true"]
    assert "clone" in clone_call
