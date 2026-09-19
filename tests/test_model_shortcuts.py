# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for curated MODEL_SHORTCUTS and _SHORTCUT_SIZES in model_manager."""

import pytest
from localm.model_manager.registry import (
    MODEL_SHORTCUTS,
    _SHORTCUT_SIZES,
    resolve_spec,
)


class TestModelShortcuts:
    def test_shortcuts_not_empty(self):
        assert MODEL_SHORTCUTS, "MODEL_SHORTCUTS must contain curated shortcuts"
        assert _SHORTCUT_SIZES, "_SHORTCUT_SIZES must contain sizes for shortcuts"

    def test_every_shortcut_has_a_size(self):
        for alias in MODEL_SHORTCUTS:
            assert alias in _SHORTCUT_SIZES, f"Missing size mapping for shortcut alias: {alias}"
            size_str = _SHORTCUT_SIZES[alias]
            assert size_str.startswith("~"), f"Size string should start with '~': {size_str}"
            assert size_str.endswith("GB"), f"Size string should end with 'GB': {size_str}"

    def test_every_shortcut_spec_is_well_formed(self):
        for alias, spec in MODEL_SHORTCUTS.items():
            assert ":" in spec, f"Spec for {alias} must be in 'owner/repo:filename' format, got: {spec}"
            repo, filename = spec.split(":", 1)
            assert "/" in repo, f"Repo in {spec} must have an owner/repo format"
            assert not repo.startswith("/"), f"Repo in {spec} must not start with slash"
            assert not repo.endswith("/"), f"Repo in {spec} must not end with slash"
            assert filename.endswith(".gguf"), f"Filename in {spec} must end with .gguf"

    def test_resolve_spec(self):
        for alias, spec in MODEL_SHORTCUTS.items():
            assert resolve_spec(alias) == spec
        # Non-shortcut specs pass through untouched
        assert resolve_spec("custom/my-model:model.gguf") == "custom/my-model:model.gguf"
        assert resolve_spec("nonexistent") == "nonexistent"

    def test_upstream_author_prefixes_present(self):
        # Specific regression checks: bartowski repos require upstream author prefixes
        # for microsoft phi and google gemma models.
        assert MODEL_SHORTCUTS["phi4-mini"].startswith("bartowski/microsoft_Phi-4-mini-instruct-GGUF:")
        assert MODEL_SHORTCUTS["gemma3-4b"].startswith("bartowski/google_gemma-3-4b-it-GGUF:")
        assert MODEL_SHORTCUTS["gemma3-12b"].startswith("bartowski/google_gemma-3-12b-it-GGUF:")
