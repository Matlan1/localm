# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the plugin contract interfaces (localm/plugins/contract.py)."""

from localm.plugins import contract as c


def test_api_version_is_int():
    assert isinstance(c.API_VERSION, int) and c.API_VERSION >= 1


def test_spec_defaults_scope_to_name():
    s = c.PluginSpec(name="coder")
    assert s.scope == "coder"
    assert s.api_version == c.API_VERSION
    assert s.builtin is False and s.protected is False
    assert s.compatible() is True


def test_spec_explicit_scope_kept():
    s = c.PluginSpec(name="img", scope="image")
    assert s.scope == "image"


def test_spec_incompatible_api_version():
    s = c.PluginSpec(name="future", api_version=c.API_VERSION + 1)
    assert s.compatible() is False


def test_surface_defaults():
    s = c.PluginSpec(name="x")
    assert isinstance(s.surface, c.Surface)
    assert s.surface.tab_id == ""


def test_plugin_protocol_runtime_checkable():
    class P:
        def register(self, host):
            ...

        def unregister(self):
            ...

    assert isinstance(P(), c.Plugin)

    class NotAPlugin:
        def register(self, host):
            ...

    assert not isinstance(NotAPlugin(), c.Plugin)
