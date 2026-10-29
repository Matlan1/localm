# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fixtures for tests that fake a system program by faking ``subprocess``."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _nvidia_smi_found(monkeypatch):
    """nvidia-smi resolves to a full path (the real one when installed); whether
    it runs is up to each test's faked subprocess call."""
    from localm import sysprog
    real = sysprog.find
    monkeypatch.setattr(sysprog, "find", lambda name: (real(name) or "/usr/bin/nvidia-smi")
                        if name == "nvidia-smi" else real(name))
