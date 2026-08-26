# SPDX-License-Identifier: AGPL-3.0-or-later
"""Guards on the [gpu] extra's pins: the HF/torch backend must stay importable.

transformers 5.14/5.14.1 imports `transformers/distributed/fsdp.py` on the
ordinary `from transformers import AutoTokenizer` path (via generation ->
GenerationMixin), and fsdp needs torch's distributed C extension
`torch._C._distributed_c10d`. The ROCm Windows torch this project pins
(torch==2.9.1+rocm7.13.0) is built WITHOUT distributed, so that makes EVERY HF
model load die at "loading processor..." - and the lazy-import layer reports it
as "Could not import module 'AutoTokenizer'", hiding the real cause.

The assertions target uv.lock, NOT the installed venv: a lock refresh can move
transformers while a dev venv stays on an older pin, so every dev machine and
the whole test suite stay green while the artifact a real user installs is
broken. uv.lock is what ships, so uv.lock is what gets asserted.

The cap moves version by version rather than being lifted outright: only the
range actually verified against the pinned ROCm torch is allowed. See
pyproject.toml's [gpu] extra for the current note.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# The first version not yet verified against the pinned ROCm torch. Raise only
# after a cold-install re-verify.
_TRANSFORMERS_BREAKS_AT = "5.16"


def _gpu_requirements() -> list[str]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]["gpu"]


def _locked_version(name: str) -> str | None:
    """The version uv.lock resolves *name* to (the version that actually ships)."""
    data = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    for pkg in data.get("package", []):
        if pkg.get("name") == name:
            return pkg.get("version")
    return None


def test_pyproject_transformers_spec_excludes_the_fsdp_breaking_line():
    """The [gpu] pin must not ADMIT transformers 5.14+. Asserted semantically, so a
    reformatting or an equivalent rewrite of the specifier still counts - and so the
    plain `~=5.13` that caused this (it resolves to the newest 5.x) fails."""
    packaging_specifiers = pytest.importorskip("packaging.specifiers")
    packaging_requirements = pytest.importorskip("packaging.requirements")

    specs = [packaging_requirements.Requirement(r) for r in _gpu_requirements()]
    transformers = [s for s in specs if s.name == "transformers"]
    assert transformers, "the [gpu] extra must pin transformers"

    spec: packaging_specifiers.SpecifierSet = transformers[0].specifier
    assert not spec.contains(_TRANSFORMERS_BREAKS_AT), (
        f"pyproject [gpu] admits transformers {_TRANSFORMERS_BREAKS_AT} "
        f"(spec: '{spec}'), which breaks the HF backend on the pinned ROCm/Windows "
        "torch (no torch._C._distributed_c10d). Cap below it; see this module's "
        "docstring."
    )


def test_locked_transformers_cannot_break_the_hf_backend():
    """uv.lock ships inside the release zip, so the LOCKED version is what a real
    user installs. This is the assertion that catches a lock drift while every dev
    venv stays green on an older pin."""
    packaging_version = pytest.importorskip("packaging.version")

    locked = _locked_version("transformers")
    assert locked, "transformers must be present in uv.lock"
    assert packaging_version.Version(locked) < packaging_version.Version(
        _TRANSFORMERS_BREAKS_AT
    ), (
        f"uv.lock resolves transformers=={locked}, at/above {_TRANSFORMERS_BREAKS_AT}, "
        "which breaks EVERY HF model load on the pinned ROCm/Windows torch. Re-lock "
        "with the pyproject cap; do not raise the cap on a dev-venv check alone."
    )


def test_locked_torch_is_the_pinned_rocm_wheel_on_windows():
    """Pins the OTHER half of the incompatibility: if torch ever moves to a build
    that HAS distributed, the transformers cap can be revisited, and this test is
    where that is recorded."""
    packaging_requirements = pytest.importorskip("packaging.requirements")

    specs = [packaging_requirements.Requirement(r) for r in _gpu_requirements()]
    torch = [s for s in specs if s.name == "torch"]
    assert torch, "the [gpu] extra must pin torch"
    assert "rocm" in str(torch[0].specifier), (
        "the win32 torch pin is no longer the ROCm wheel - re-check whether the "
        "transformers cap in this module is still needed (it exists only because that "
        "wheel lacks torch._C._distributed_c10d)."
    )
