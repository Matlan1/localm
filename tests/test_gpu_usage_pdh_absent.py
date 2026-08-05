# SPDX-License-Identifier: AGPL-3.0-or-later
"""pywin32 must be DECLARED, and its absence must degrade honestly.

Two separate properties, and they are separate on purpose.

THE DECLARATION. gpu_usage.py's PDH source reads the vendor-neutral WDDM
counter through ``win32pdh``. It is the ONLY device-global VRAM source on a
non-AMD Windows GPU, because the preferred source (AMD's ADL) is AMD-only by
construction. pywin32 was installed in the maintainer's venv and declared in
neither pyproject.toml nor uv.lock, so a clean install on an NVIDIA or Intel
Windows box silently lost the cross-process correction. The gap was structurally
invisible where it was written: on an AMD box ADL answers first and the PDH path
is never reached.

THE DEGRADATION. Declaring it does not make its absence safe by itself, and a
missing probe path is exactly where a confident wrong number gets invented. So
this also pins what happens with no source at all: "unmeasurable", never a
figure. That property held before the declaration and must keep holding, since
a user can still uninstall the package.
"""

from __future__ import annotations

import sys

import pytest

from localm import gpu_usage


class _BlockWin32Pdh:
    """Make ``import win32pdh`` fail exactly as on a clean install.

    Uses find_spec (PEP 451). The older find_module hook was REMOVED in Python
    3.12, so a blocker written that way is silently inert - it would let the
    real module import and the test would pass while proving nothing. That is
    why every test below asserts the blocker fired before trusting a result.
    """

    def find_spec(self, name, path=None, target=None):
        if name == "win32pdh":
            raise ImportError("No module named 'win32pdh'")
        return None


@pytest.fixture
def no_pywin32(monkeypatch):
    blocker = _BlockWin32Pdh()
    sys.meta_path.insert(0, blocker)
    monkeypatch.setattr(gpu_usage, "_pdh_state", None, raising=False)
    monkeypatch.delitem(sys.modules, "win32pdh", raising=False)
    try:
        # The guard must be proven live, not assumed. A blocker that cannot fire
        # makes every assertion below meaningless.
        with pytest.raises(ImportError):
            import win32pdh  # noqa: F401
        yield
    finally:
        sys.meta_path.remove(blocker)
        gpu_usage._pdh_state = None


# ------------------------------------------------------------- the declaration

def test_pywin32_is_declared_for_windows():
    """The actual gap. gpu_usage.py reaches for win32pdh, so something has to
    put it on a clean Windows install."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    matches = [d for d in deps if d.split(">=")[0].split(";")[0].strip() == "pywin32"]
    assert matches, (
        "pywin32 is imported by localm/gpu_usage.py but declared nowhere; a "
        "clean non-AMD Windows install would lose device-global VRAM reporting")
    assert "sys_platform == 'win32'" in matches[0], (
        f"pywin32 must be Windows-gated, got: {matches[0]!r}")


def test_pywin32_is_in_the_lockfile():
    """pyproject alone is not enough for a locked install to get it."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    lock = (root / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "pywin32"' in lock, "pywin32 missing from uv.lock"


# ------------------------------------------------------------- the degradation

def test_the_pdh_source_reports_nothing_rather_than_raising(no_pywin32):
    assert gpu_usage._pdh_adapter_used() == []


def test_the_unavailable_result_is_latched(no_pywin32):
    """Second call must not retry the import on every VRAM read."""
    assert gpu_usage._pdh_adapter_used() == []
    assert gpu_usage._pdh_state == {}
    assert gpu_usage._pdh_adapter_used() == []


def test_with_no_source_at_all_the_answer_is_unmeasurable_not_a_number(
        no_pywin32, monkeypatch):
    """THE RULE-5 CHECK. Simulates a non-AMD Windows box: ADL does not exist
    (it is AMD-only) and pywin32 is absent, so neither device-global source is
    available.

    The result must be an EMPTY MAPPING, meaning "no device-global figure",
    which callers render as unknown. A fabricated number here would be worse
    than the blind reading it replaces, because it would be confidently wrong
    and indistinguishable from a real measurement.
    """
    monkeypatch.setattr(gpu_usage, "_adl_used_by_bus", lambda: {})
    monkeypatch.setattr(gpu_usage, "_adl_state", {}, raising=False)
    gpus = [{"index": 0, "name": "Non-AMD GPU", "total": 8 * 1024 ** 3}]
    assert gpu_usage.device_global_used_bytes(gpus) == {}


def test_a_missing_source_does_not_become_a_zero_reading(no_pywin32, monkeypatch):
    """The specific wrong answer worth naming: 0 bytes used reads as "the card
    is empty", which is the opposite of unknown and would let a caller load a
    model onto a full GPU."""
    monkeypatch.setattr(gpu_usage, "_adl_used_by_bus", lambda: {})
    monkeypatch.setattr(gpu_usage, "_adl_state", {}, raising=False)
    gpus = [{"index": 0, "name": "Non-AMD GPU", "total": 8 * 1024 ** 3}]
    out = gpu_usage.device_global_used_bytes(gpus)
    assert 0 not in out.values(), f"reported 0 bytes used with no source: {out}"
