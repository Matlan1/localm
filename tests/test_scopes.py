# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the capability/scope taxonomy (localm/scopes.py)."""

from localm import scopes


def test_kernel_scopes_documented():
    for s in (scopes.MODELS_READ, scopes.MODELS_WRITE, scopes.CONFIG_READ,
              scopes.CONFIG_WRITE, scopes.PLUGINS_READ, scopes.PLUGINS_ADMIN,
              scopes.KEYS_ADMIN, scopes.ADMIN):
        assert s in scopes.KERNEL_SCOPES


def test_builtin_plugin_scopes_documented():
    for s in (scopes.CHAT, scopes.CODER, scopes.IMAGE, scopes.MUSIC,
              scopes.VIDEO, scopes.RAG, scopes.WEB, scopes.VOICE, scopes.MCP):
        assert s in scopes.BUILTIN_PLUGIN_SCOPES


def test_all_scope_strings_unique():
    vals = list(scopes.KERNEL_SCOPES) + list(scopes.BUILTIN_PLUGIN_SCOPES)
    assert len(vals) == len(set(vals))


def test_is_valid_scope():
    assert scopes.is_valid_scope(scopes.CODER)
    assert scopes.is_valid_scope(scopes.MODELS_WRITE)     # colon kernel scope
    assert scopes.is_valid_scope("my_third_party")        # plugin-name scope
    assert scopes.is_valid_scope("x", extra={"x"})        # registered third-party
    assert not scopes.is_valid_scope("")
    assert not scopes.is_valid_scope("bad name")          # space
    assert not scopes.is_valid_scope(None)


def test_privileged_scopes_are_known():
    # Every privileged scope must be a real, known scope: a kernel scope or an
    # EXTRA capability variant like coder:full.
    for s in scopes.PRIVILEGED_SCOPES:
        assert s in scopes.all_known_scopes()


def test_grants_admin_implies_all():
    assert scopes.grants({scopes.ADMIN}, scopes.CODER)
    assert scopes.grants({scopes.CODER}, scopes.CODER)
    assert not scopes.grants({scopes.CHAT}, scopes.CODER)


def test_normalize():
    assert scopes.normalize([" coder ", "coder", "", "chat", None]) == ["chat", "coder"]
