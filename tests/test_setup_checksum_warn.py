# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup-llama must not silently skip download integrity verification.

On the main dynamic (latest-release) path the expected sha256 comes from the
GitHub asset's optional `digest` field, falling back to a small pinned table.
When neither is available the sha resolves to None and _validate_archive skips
the provenance check with NO warning - inconsistent with the --url path, which
warns "Custom URL download is unverified". A security step that silently does
not run is a hidden problem.
"""

from pathlib import Path

from localm import setup_llama


def test_provision_warns_when_asset_has_no_checksum(monkeypatch, capsys):
    monkeypatch.setattr(setup_llama, "_resolve_backend_asset",
                        lambda backend, cuda_line=None, tag=None: ("http://example/vulkan.zip", None, "bTEST"))
    placed = {}
    monkeypatch.setattr(setup_llama, "_fetch_and_place",
                        lambda url, target, sha: placed.update(sha=sha))

    setup_llama._provision_backend("vulkan", Path("target"), None, False)

    out = capsys.readouterr().out.lower()
    assert placed["sha"] is None, "the no-checksum path must have been taken"
    assert "checksum" in out or ("not" in out and "verif" in out), (
        f"a missing checksum must be surfaced, not silently skipped; got: {out!r}")


def test_provision_does_not_warn_when_checksum_present(monkeypatch, capsys):
    monkeypatch.setattr(setup_llama, "_resolve_backend_asset",
                        lambda backend, cuda_line=None, tag=None: ("http://example/vulkan.zip", "a" * 64, "bTEST"))
    monkeypatch.setattr(setup_llama, "_fetch_and_place",
                        lambda url, target, sha: None)

    setup_llama._provision_backend("vulkan", Path("target"), None, False)

    out = capsys.readouterr().out.lower()
    assert "no checksum" not in out and "not cryptographically verified" not in out


def test_explicit_sha_overrides_missing_asset_checksum(monkeypatch, capsys):
    """--sha256 provides the verification even when the asset publishes none."""
    monkeypatch.setattr(setup_llama, "_resolve_backend_asset",
                        lambda backend, cuda_line=None, tag=None: ("http://example/vulkan.zip", None, "bTEST"))
    placed = {}
    monkeypatch.setattr(setup_llama, "_fetch_and_place",
                        lambda url, target, sha: placed.update(sha=sha))

    setup_llama._provision_backend("vulkan", Path("target"), "b" * 64, False)

    out = capsys.readouterr().out.lower()
    assert placed["sha"] == "b" * 64
    assert "no checksum" not in out
