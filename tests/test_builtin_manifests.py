# SPDX-License-Identifier: AGPL-3.0-or-later
"""Validate every bundled (store) plugin manifest so a broken plugin.toml, a
missing register module, or a client-asset plugin shipping a missing entry module
fails CI rather than the user at install time.

Unlike the engine tests (which use synthetic manifests), this iterates the REAL
store under localm/plugins/builtin/, located repo-relative so it validates the
checkout under test.
"""

import tomllib
from pathlib import Path

import pytest

from localm.plugins.engine import parse_spec

_ROOT = Path(__file__).resolve().parents[1]
_BUILTIN = _ROOT / "localm" / "plugins" / "builtin"


def _declared_extras() -> set:
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return set(data.get("project", {}).get("optional-dependencies", {}))


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

    # Client-asset contract: a client_entry requires an assets_dir that holds it.
    if spec.surface.client_entry:
        assert spec.surface.assets_dir, \
            f"{spec.name}: client_entry set but assets_dir is empty"
        entry = plugin_dir / spec.surface.assets_dir / spec.surface.client_entry
        assert entry.is_file(), f"{spec.name}: client_entry {entry} missing"


@pytest.mark.parametrize("plugin_dir", _builtin_dirs(), ids=lambda d: d.name)
def test_builtin_requires_extras_are_real_pyproject_extras(plugin_dir):
    """A builtin's requires_extras must name a real [project.optional-dependencies]
    extra, so a typo cannot silently turn the plugin dep self-installer into a
    no-op (the host resolves the extra name against localm's own metadata)."""
    spec = parse_spec(plugin_dir, builtin=True)
    extras = _declared_extras()
    for e in spec.requires_extras:
        assert e in extras, \
            f"{spec.name}: requires_extras '{e}' is not a pyproject extra {sorted(extras)}"


def test_rag_declares_its_pdf_extra():
    """RAG's PDF extraction needs pypdf (the [rag] extra). It must self-provision
    that on enable - like voice does for faster-whisper - or PDFs silently fail on
    a default install (which ships .[coder,voice,monitor], no rag)."""
    spec = parse_spec(_BUILTIN / "rag", builtin=True)
    assert spec.requires_extras == ["rag"]
    from localm.plugins import deps
    reqs = deps.plugin_requirements(["rag"])
    assert reqs, "the rag extra must resolve to at least one requirement"
    # Normally the concrete pypdf specifier; the localm[rag] fallback is accepted
    # when the installed-metadata read is unavailable.
    assert any("pypdf" in r for r in reqs) or reqs == ["localm[rag]"], reqs
