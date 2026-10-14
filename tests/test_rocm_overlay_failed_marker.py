# SPDX-License-Identifier: AGPL-3.0-or-later
"""An amd-rocm install whose SIMD CPU overlay could not be installed is not reported
as out of date for as long as the overlay is missing.

Provisioning records the plain amd-rocm build in that case, and the update check
compares against the build with the overlay, so without a record of the failed
attempt the "runtime update available" card and the "Upgrading" line of
``setup-llama --force`` repeat forever and cannot be cleared.
"""

from __future__ import annotations

import json

import pytest

import localm.setup_llama as sl
from localm.setup_llama import cli, rocm_cpu


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: tmp_path)
    monkeypatch.setattr(sl, "_latest_tag",
                        lambda: pytest.fail("amd-rocm must not query releases"))
    return tmp_path


def _failed_marker(target, tag):
    (target / sl.CPU_OVERLAY_FAILED_MARKER).write_text(json.dumps({"tag": tag}), encoding="utf-8")


class TestTheAttemptIsRecorded:
    def test_a_runtime_with_no_cpu_dll_to_replace_records_a_failed_attempt(self, tmp_path):
        assert sl.install_rocm_simd_cpu(tmp_path) is None
        assert sl.cpu_overlay_failed(tmp_path)
        assert sl.rocm_build(tmp_path) == sl._ROCM_TAG

    def test_a_later_success_clears_the_record(self, tmp_path, monkeypatch):
        _failed_marker(tmp_path, sl._ROCM_CPU_TAG)
        monkeypatch.setattr(rocm_cpu, "_install_overlay", lambda target: "ggml-cpu-x64.dll")
        assert sl.install_rocm_simd_cpu(tmp_path) == "ggml-cpu-x64.dll"
        assert not sl.cpu_overlay_failed(tmp_path)

    def test_a_record_for_another_overlay_release_does_not_count(self, tmp_path):
        _failed_marker(tmp_path, "b1")
        assert not sl.cpu_overlay_failed(tmp_path)

    @pytest.mark.parametrize("text", ["", "not json", "[]", "null", '{"tag": 5}'])
    def test_an_unreadable_record_counts_as_no_failed_attempt(self, tmp_path, text):
        (tmp_path / sl.CPU_OVERLAY_FAILED_MARKER).write_text(text, encoding="utf-8")
        assert not sl.cpu_overlay_failed(tmp_path)

    def test_a_record_that_cannot_be_written_is_logged(self, tmp_path, monkeypatch, caplog):
        (tmp_path / sl.CPU_OVERLAY_FAILED_MARKER).mkdir()
        with caplog.at_level("WARNING", logger="localm"):
            assert sl.install_rocm_simd_cpu(tmp_path) is None
        assert any("could not record the overlay attempt" in r.getMessage()
                   for r in caplog.records)


class TestTheUpdateCheck:
    def test_a_failed_overlay_is_not_an_available_update(self, runtime):
        sl._record_provisioned_backend(runtime, "amd-rocm", build=sl._ROCM_TAG)
        _failed_marker(runtime, sl._ROCM_CPU_TAG)
        result = sl.check_runtime_update()
        assert result["current"] == sl._ROCM_TAG
        assert result["newer"] is False

    def test_an_install_that_never_tried_the_overlay_is_offered_it(self, runtime):
        sl._record_provisioned_backend(runtime, "amd-rocm", build=sl._ROCM_TAG)
        assert sl.check_runtime_update()["newer"] is True

    def test_a_failure_against_an_older_overlay_release_is_offered_a_retry(self, runtime):
        sl._record_provisioned_backend(runtime, "amd-rocm", build=sl._ROCM_TAG)
        _failed_marker(runtime, "b1")
        assert sl.check_runtime_update()["newer"] is True

    def test_an_older_amd_rocm_build_is_still_an_update(self, runtime):
        sl._record_provisioned_backend(runtime, "amd-rocm", build="b1288")
        _failed_marker(runtime, sl._ROCM_CPU_TAG)
        assert sl.check_runtime_update()["newer"] is True


class TestSetupLlamaWording:
    def _say(self, target, capsys, monkeypatch):
        class _Tty:
            def isatty(self):
                return True

        monkeypatch.setattr(cli.sys, "stdin", _Tty())
        monkeypatch.setattr(cli.click, "confirm", lambda *a, **k: True)
        kept = cli._keeps_existing_install(target, "llama.dll", "amd-rocm", False, False)
        assert kept is False
        return capsys.readouterr().out

    def _installed(self, target):
        (target / "llama.dll").write_text("x", encoding="utf-8")
        sl._record_provisioned_backend(target, "amd-rocm", build=sl._ROCM_TAG)

    def test_a_failed_overlay_is_not_announced_as_an_upgrade(self, tmp_path, capsys, monkeypatch):
        self._installed(tmp_path)
        _failed_marker(tmp_path, sl._ROCM_CPU_TAG)
        out = self._say(tmp_path, capsys, monkeypatch)
        assert "Upgrading" not in out
        assert "Re-downloading" in out

    def test_a_build_without_the_overlay_attempt_is_announced_as_an_upgrade(
            self, tmp_path, capsys, monkeypatch):
        self._installed(tmp_path)
        out = self._say(tmp_path, capsys, monkeypatch)
        assert "Upgrading" in out and sl._ROCM_BUILD in out
