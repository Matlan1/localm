# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for curated MODEL_SHORTCUTS and _SHORTCUT_SIZES in model_manager."""

import re

from localm.model_manager.registry import (
    MODEL_SHORTCUTS,
    _SHORTCUT_SIZES,
    resolve_spec,
)

# Matches a repo of the form "<owner>/<prefix>_<rest>-GGUF", where <prefix> is
# an upstream author name folded into the repo id (e.g. "microsoft", "google").
_PREFIXED_REPO_RE = re.compile(r"^[^/]+/([A-Za-z0-9]+)_.+-GGUF$")


class TestModelShortcuts:
    def test_shortcuts_not_empty(self):
        assert MODEL_SHORTCUTS, "MODEL_SHORTCUTS must contain curated shortcuts"
        assert _SHORTCUT_SIZES, "_SHORTCUT_SIZES must contain sizes for shortcuts"

    def test_every_shortcut_has_a_size(self):
        for alias in MODEL_SHORTCUTS:
            assert alias in _SHORTCUT_SIZES, f"Missing size mapping for shortcut alias: {alias}"
            size_str = _SHORTCUT_SIZES[alias]
            assert size_str.startswith("~"), f"Size string should start with '~': {size_str}"
            assert re.match(r"^~[\d.]+ GB(\Z| \+ )", size_str), (
                f"Size string should start with '~<number> GB', optionally followed "
                f"by ' + <extra download>': {size_str}")

    def test_every_shortcut_spec_is_well_formed(self):
        for alias, spec in MODEL_SHORTCUTS.items():
            assert ":" in spec, f"Spec for {alias} must be in 'owner/repo:filename' format, got: {spec}"
            repo, filename = spec.split(":", 1)
            assert "/" in repo, f"Repo in {spec} must have an owner/repo format"
            assert not repo.startswith("/"), f"Repo in {spec} must not start with slash"
            assert not repo.endswith("/"), f"Repo in {spec} must not end with slash"
            assert filename.endswith(".gguf"), f"Filename in {spec} must end with .gguf"

    def test_prefixed_repo_implies_prefixed_filename(self):
        # For every shortcut whose repo carries an upstream-author prefix, the
        # filename after ':' must carry the same prefix.
        for alias, spec in MODEL_SHORTCUTS.items():
            repo, filename = spec.split(":", 1)
            match = _PREFIXED_REPO_RE.match(repo)
            if match is None:
                continue
            prefix = match.group(1)
            assert filename.startswith(f"{prefix}_"), (
                f"{alias}: repo {repo!r} carries prefix {prefix!r} but filename "
                f"{filename!r} does not"
            )

    def test_resolve_spec(self):
        # Non-shortcut specs pass through untouched.
        assert resolve_spec("custom/my-model:model.gguf") == "custom/my-model:model.gguf"
        assert resolve_spec("nonexistent") == "nonexistent"

    def test_upstream_author_prefixes_present(self):
        # Full-spec pins for the repos requiring upstream author prefixes on
        # both the repo id and the filename.
        assert MODEL_SHORTCUTS["phi4-mini"] == (
            "bartowski/microsoft_Phi-4-mini-instruct-GGUF:"
            "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
        )
        assert MODEL_SHORTCUTS["gemma3-4b"] == (
            "bartowski/google_gemma-3-4b-it-GGUF:"
            "google_gemma-3-4b-it-Q4_K_M.gguf"
        )
        assert MODEL_SHORTCUTS["gemma3-12b"] == (
            "bartowski/google_gemma-3-12b-it-GGUF:"
            "google_gemma-3-12b-it-Q4_K_M.gguf"
        )
