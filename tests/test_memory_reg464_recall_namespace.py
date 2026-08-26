# SPDX-License-Identifier: AGPL-3.0-or-later
"""The recall inlet must read the SAME namespace the write path writes to.

Every WRITE path (memory_get/put/append/patch/delete + memory_consolidate) and
the auto-consolidate outlet collapse to the shared "owner" namespace for an
ADMIN/owner caller (memory_principal -> None -> "owner"), so an owner's memories
survive a key rotation. The recall INLET must apply the same ADMIN->owner
collapse rather than reading the raw per-key-hash principal (ctx.principal): in
protected mode (an API key configured, owner authed with an ADMIN key) the owner
SAVES into "owner", so an inlet reading the key-hash namespace injects nothing.

The case guarded here: a store populated through the owner-collapsed write path,
recalled through a realistic protected-mode ctx (ADMIN scope + key-hash
principal).
"""

from __future__ import annotations

import types

from localm import scopes
from localm.memory import MemoryRecord
from localm.plugins.builtin.memory import plug


def _home(tmp_path, monkeypatch):
    monkeypatch.setattr(plug, "_home", lambda: tmp_path)
    monkeypatch.setenv("LOCALM_MODE", "log")           # writes allowed (not privacy)


def _owner_ctx():
    """A protected-mode owner ctx as the chat route builds it: ctx.principal is the
    key hash (principal_id) and ctx.scopes carries ADMIN (routes/chat.py)."""
    return types.SimpleNamespace(model_id="", principal="owner-key-hash",
                                 stream=False, request_id="r1", state={},
                                 scopes=(scopes.ADMIN,))


def test_owner_recall_reads_owner_namespace_in_protected_mode(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    # The write path collapses ADMIN -> "owner", so the owner's memory lives in the
    # shared "owner" namespace. _chat_store(None) opens exactly that namespace.
    plug._chat_store(None).add(MemoryRecord(
        text="User prefers Python and pytest", source="user", importance=0.9))

    messages = [{"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "help me test my python code"}]
    out = plug._memory_inlet(messages, _owner_ctx())

    assert out is messages, "owner recall injected nothing (wrong namespace)"
    assert "Python and pytest" in messages[0]["content"]


def test_owner_recall_matches_write_namespace(tmp_path, monkeypatch):
    """Read/write namespace consistency, stated directly: whatever namespace the
    inlet resolves for an owner ctx MUST equal the one the write path uses."""
    _home(tmp_path, monkeypatch)
    write_principal = None                              # memory_principal collapses ADMIN -> None
    read_principal = plug._ctx_principal(_owner_ctx())  # the inlet's resolution
    assert read_principal == write_principal
    assert plug._chat_store(read_principal).path == plug._chat_store(write_principal).path


def test_scoped_non_owner_keeps_its_own_namespace(tmp_path, monkeypatch):
    """A non-owner scoped key (no ADMIN) must NOT collapse to owner - its recall
    stays bound to its own key-hash namespace, matching principal_id on the write
    path for a scoped key."""
    _home(tmp_path, monkeypatch)
    ctx = types.SimpleNamespace(model_id="", principal="scoped-hash", stream=False,
                               request_id="r1", state={}, scopes=(scopes.CHAT,))
    assert plug._ctx_principal(ctx) == "scoped-hash"
