# SPDX-License-Identifier: AGPL-3.0-or-later
"""Validate every bundled (store) plugin manifest so a broken plugin.toml, a
missing register module, or a client-asset plugin shipping a missing entry module
fails CI rather than the user at install time.

Unlike the engine tests (which use synthetic manifests), this iterates the REAL
store under localm/plugins/builtin/, located repo-relative so it validates the
checkout under test.
"""

from pathlib import Path

import pytest

from localm.plugins.engine import parse_spec

_BUILTIN = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "builtin"


def _builtin_dirs():
    if not _BUILTIN.is_dir():
        return []
    return sorted(d for d in _BUILTIN.iterdir()
                  if d.is_dir() and (d / "plugin.toml").is_file())


def test_store_has_plugins():
    assert _builtin_dirs(), f"no bundled plugins found under {_BUILTIN}"


@pytest.mark.parametrize("plugin_dir", _builtin_dirs(), ids=lambda d: d.name)
def test_builtin_manifest_valid(plugin_dir):
    spec = parse_spec(plugin_dir, builtin=True)
    assert spec.name == plugin_dir.name
    assert spec.scope
    assert spec.compatible(), f"{spec.name}: api_version {spec.api_version} unsupported"

    # The module named by register_entry is what the engine imports at load time.
    mod = (spec.register_entry or "plugin").split(":", 1)[0]
    assert (plugin_dir / f"{mod}.py").is_file() \
        or (plugin_dir / mod / "__init__.py").is_file(), \
        f"{spec.name}: register module {mod!r} not found"

    # Client-asset contract: a client_entry requires an assets_dir that holds it
    # (so the SPA can import /plugins/<name>/<client_entry>).
    if spec.surface.client_entry:
        assert spec.surface.assets_dir, \
            f"{spec.name}: client_entry set but assets_dir is empty"
        entry = plugin_dir / spec.surface.assets_dir / spec.surface.client_entry
        assert entry.is_file(), f"{spec.name}: client_entry {entry} missing"
