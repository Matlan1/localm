# SPDX-License-Identifier: AGPL-3.0-or-later
"""The owner's chat memory must live in the shared "owner" namespace.

``principal_id`` has no ADMIN/owner short-circuit (a job must stay bound to the
key that created it), so using it directly to namespace chat memory puts the
owner's memories in a per-key-hash namespace in protected mode - invisible in
the GUI and orphaned again on every key rotation.

The memory namespace routes through memory_principal(), which collapses an
ADMIN/owner caller to the "owner" namespace while leaving principal_id (job
ownership) unchanged.
"""

from localm.inference import http_server as hs
from localm.memory import principal_of
from localm import scopes


def test_owner_memory_namespace_is_owner(monkeypatch):
    monkeypatch.setattr(hs, "caller_scopes", lambda r: {scopes.ADMIN})
    monkeypatch.setattr(hs, "principal_id", lambda r: "owner-key-hash")
    # ADMIN -> memory principal None -> the shared "owner" namespace.
    assert hs.memory_principal(object()) is None
    assert principal_of(hs.memory_principal(object())) == "owner"


def test_scoped_key_memory_keeps_its_own_namespace(monkeypatch):
    monkeypatch.setattr(hs, "caller_scopes", lambda r: {scopes.CHAT})
    monkeypatch.setattr(hs, "principal_id", lambda r: "scoped-hash")
    assert hs.memory_principal(object()) == "scoped-hash"
    assert principal_of(hs.memory_principal(object())) == "scoped-hash"


def test_open_mode_memory_namespace_is_owner(monkeypatch):
    monkeypatch.setattr(hs, "caller_scopes", lambda r: None)
    monkeypatch.setattr(hs, "principal_id", lambda r: None)
    assert hs.memory_principal(object()) is None
    assert principal_of(hs.memory_principal(object())) == "owner"


def test_job_ownership_principal_unchanged_for_admin(monkeypatch):
    """principal_id (job binding, KEY-SCOPE-2) must NOT collapse ADMIN to owner -
    only the memory principal does."""
    monkeypatch.setattr(hs, "caller_scopes", lambda r: {scopes.ADMIN})
    monkeypatch.setattr(hs, "_request_token", lambda r: ("tok", "header"))
    monkeypatch.setattr("localm.auth.any_key_configured", lambda: True)
    monkeypatch.setattr(hs, "_principal_from_token",
                        lambda t, s: ({scopes.ADMIN}, "owner-key-hash", None, []))
    # principal_id still returns the real hash (job stays bound to the key).
    assert hs.principal_id(object()) == "owner-key-hash"
