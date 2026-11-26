# SPDX-License-Identifier: AGPL-3.0-or-later
"""``setup-llama --sha256`` and ``--only`` stop without trying another build.
When they stop, the reason the build could not be used is printed and logged:
the load check's own detail for a build that was placed and did not load, the
provisioning error for one that could not be placed.

The provisioning and the load check are stubbed (see
tests/test_setup_llama_vulkan_cpu_recovery.py); the branch that reports runs
for real.
"""

from __future__ import annotations

import logging

import pytest

from localm import setup_llama as sl
from localm.setup_llama.download import ArtifactError
from localm.setup_llama.provision import BUILD_UNUSABLE_EXIT
from tests.test_setup_llama_vulkan_cpu_recovery import _flat, _wire

_SHA = "0" * 64
_HIPBLAS_DETAIL = ("RuntimeError: Failed to load llama.dll from rt\\llama.dll: "
                   "hipblas.dll is missing (needed by ggml-hip.dll)")


def _records(caplog) -> str:
    return " ".join(r.getMessage() for r in caplog.records if r.name.startswith("localm"))


def test_a_pinned_build_that_does_not_load_names_the_missing_library(
        monkeypatch, tmp_path, capsys, caplog):
    provisioned = _wire(monkeypatch, tmp_path, loads=[(False, _HIPBLAS_DETAIL)])
    monkeypatch.setattr(sl, "vc_runtime_blocks_load", lambda target, text: False)
    with caplog.at_level(logging.INFO, logger="localm"), pytest.raises(SystemExit) as exc:
        sl._provision_with_fallback("hip", tmp_path, _SHA, False, assume_yes=True)
    assert exc.value.code == 1
    assert provisioned == ["hip"], "an explicit --sha256 never tries another build"
    out = _flat(capsys)
    assert "was provisioned but did not load: hipblas.dll (needed by ggml-hip.dll)" in out
    assert "Not falling back" in out
    logged = _records(caplog)
    assert "pinned hip artifact was provisioned but did not load" in logged
    assert _HIPBLAS_DETAIL in logged


def test_a_reason_of_another_shape_is_printed_as_it_is(
        monkeypatch, tmp_path, capsys, caplog):
    detail = "RuntimeError: llama_backend_init failed [code 3]"
    _wire(monkeypatch, tmp_path, loads=[(False, detail)])
    monkeypatch.setattr(sl, "vc_runtime_blocks_load", lambda target, text: False)
    with caplog.at_level(logging.INFO, logger="localm"), pytest.raises(SystemExit):
        sl._provision_with_fallback("hip", tmp_path, _SHA, False, assume_yes=True)
    assert f"did not load: {detail}" in _flat(capsys)
    assert detail in _records(caplog)


def test_a_pinned_build_that_could_not_be_placed_logs_why(
        monkeypatch, tmp_path, capsys, caplog):
    _wire(monkeypatch, tmp_path, loads=[])

    def refuse(*args, **kwargs):
        raise ArtifactError("sha256 mismatch for llama-b1-bin-win-hip-x64.zip")

    monkeypatch.setattr(sl, "_provision_backend", refuse)
    with caplog.at_level(logging.INFO, logger="localm"), pytest.raises(SystemExit) as exc:
        sl._provision_with_fallback("hip", tmp_path, _SHA, False, assume_yes=True)
    assert exc.value.code == 1
    out = _flat(capsys)
    assert "sha256 mismatch" in out
    assert "The pinned artifact failed validation." in out
    assert "provisioning: sha256 mismatch for llama-b1-bin-win-hip-x64.zip" in _records(caplog)


def test_only_logs_why_the_build_is_not_used(monkeypatch, tmp_path, capsys, caplog):
    _wire(monkeypatch, tmp_path, loads=[(False, _HIPBLAS_DETAIL)])
    monkeypatch.setattr(sl, "vc_runtime_blocks_load", lambda target, text: False)
    with caplog.at_level(logging.INFO, logger="localm"), pytest.raises(SystemExit) as exc:
        sl._provision_with_fallback("hip", tmp_path, None, False, assume_yes=True,
                                    only=True)
    assert exc.value.code == BUILD_UNUSABLE_EXIT
    assert "hipblas.dll" in _flat(capsys)
    assert "the hip build is not used (--only)" in _records(caplog)
    assert "hipblas.dll" in _records(caplog)
