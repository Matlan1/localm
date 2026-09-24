# SPDX-License-Identifier: AGPL-3.0-or-later
"""Guards on the [gpu] and [grammar] extras' pins: the HF/torch backend and its
grammar support must stay importable.

transformers 5.14/5.14.1 imports `transformers/distributed/fsdp.py` on the
ordinary `from transformers import AutoTokenizer` path (via generation ->
GenerationMixin), and fsdp needs torch's distributed C extension
`torch._C._distributed_c10d`. The ROCm Windows torch this project pins
(torch==2.9.1+rocm7.13.0) is built WITHOUT distributed, so that makes EVERY HF
model load die at "loading processor..." - and the lazy-import layer reports it
as "Could not import module 'AutoTokenizer'", hiding the real cause.

The assertions target pyproject.toml, which installs resolve from, and uv.lock,
NOT the installed venv: a lock refresh can move transformers while a dev venv
stays on an older pin, so every dev machine and the whole test suite stay green
while the locked set is broken.

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


def _extra_requirements(extra: str) -> list[str]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"][extra]


def _gpu_requirements() -> list[str]:
    return _extra_requirements("gpu")


def _locked_version(name: str) -> str | None:
    """The version uv.lock resolves *name* to (the version that actually ships)."""
    data = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    for pkg in data.get("package", []):
        if pkg.get("name") == name:
            return pkg.get("version")
    return None


@pytest.mark.parametrize("extra", ["gpu", "hf"])
def test_pyproject_transformers_spec_excludes_the_fsdp_breaking_line(extra):
    """Neither [gpu] nor [hf] may ADMIT transformers 5.14+. Asserted
    semantically, so a reformatting or an equivalent rewrite of the specifier
    still counts - and so a plain `~=5.13` (it resolves to the newest 5.x)
    fails."""
    packaging_specifiers = pytest.importorskip("packaging.specifiers")
    packaging_requirements = pytest.importorskip("packaging.requirements")

    specs = [packaging_requirements.Requirement(r) for r in _extra_requirements(extra)]
    transformers = [s for s in specs if s.name == "transformers"]
    assert transformers, f"the [{extra}] extra must pin transformers"

    spec: packaging_specifiers.SpecifierSet = transformers[0].specifier
    assert not spec.contains(_TRANSFORMERS_BREAKS_AT), (
        f"pyproject [{extra}] admits transformers {_TRANSFORMERS_BREAKS_AT} "
        f"(spec: '{spec}'), which breaks the HF backend on the pinned ROCm/Windows "
        "torch (no torch._C._distributed_c10d). Cap below it; see this module's "
        "docstring."
    )


def test_hf_extra_matches_gpu_extra_on_the_shared_hf_pins():
    """[hf] and [gpu] must never drift apart on the pins they share
    (transformers, tokenizers, accelerate, psutil) - [hf] exists so a
    non-ROCm install gets the identical HF stack [gpu] gives a ROCm/Windows
    one."""
    packaging_requirements = pytest.importorskip("packaging.requirements")

    def _specifiers(extra):
        specs = [packaging_requirements.Requirement(r) for r in _extra_requirements(extra)]
        return {s.name: str(s.specifier) for s in specs}

    gpu_specs = _specifiers("gpu")
    hf_specs = _specifiers("hf")
    for name in ("transformers", "tokenizers", "accelerate", "psutil"):
        assert name in hf_specs, f"[hf] is missing {name}"
        assert hf_specs[name] == gpu_specs[name], (
            f"[hf] pins {name} as '{hf_specs[name]}' but [gpu] pins it as "
            f"'{gpu_specs[name]}' - keep the two in sync"
        )


def test_locked_transformers_cannot_break_the_hf_backend():
    """uv.lock ships inside the release zip and is what a locked environment
    installs. This is the assertion that catches a lock drift while every dev
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


# (first xgrammar release that needs it, the apache-tvm-ffi floor it loads with)
_XGRAMMAR_NEEDS_TVM_FFI = ("0.2.6", "0.1.13")


def test_grammar_extra_pairs_xgrammar_with_a_loadable_tvm_ffi():
    """xgrammar 0.2.6 and later fail at import with an apache-tvm-ffi older than
    0.1.13. The [grammar] extra must require that floor, and uv.lock must not
    pair such an xgrammar with an older apache-tvm-ffi."""
    packaging_requirements = pytest.importorskip("packaging.requirements")
    packaging_version = pytest.importorskip("packaging.version")
    xgrammar_from, tvm_ffi_floor = map(packaging_version.Version, _XGRAMMAR_NEEDS_TVM_FFI)

    specs = {r.name: r.specifier for r in
             map(packaging_requirements.Requirement, _extra_requirements("grammar"))}
    assert "apache-tvm-ffi" in specs, "the [grammar] extra must pin apache-tvm-ffi"
    below = f"{tvm_ffi_floor.major}.{tvm_ffi_floor.minor}.{tvm_ffi_floor.micro - 1}"
    assert not specs["apache-tvm-ffi"].contains(below), (
        f"[grammar] admits apache-tvm-ffi {below}, which xgrammar {xgrammar_from}+ "
        "cannot load")

    xgrammar, tvm_ffi = _locked_version("xgrammar"), _locked_version("apache-tvm-ffi")
    assert xgrammar and tvm_ffi, "xgrammar and apache-tvm-ffi must be present in uv.lock"
    if packaging_version.Version(xgrammar) >= xgrammar_from:
        assert packaging_version.Version(tvm_ffi) >= tvm_ffi_floor, (
            f"uv.lock pairs xgrammar {xgrammar} with apache-tvm-ffi {tvm_ffi}, "
            "which it cannot load")
