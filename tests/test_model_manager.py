# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for localm.model_manager disk-space preflight, resumable download helpers,
and SHA256 verification.
"""

import hashlib
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from localm.model_manager import _check_disk_space, _sha256_file


# ---------------------------------------------------------------------------
#  _check_disk_space
# ---------------------------------------------------------------------------

class TestCheckDiskSpace:
    def _patch_usage(self, free: int):
        mock_usage = MagicMock()
        mock_usage.free = free
        return patch("localm.model_manager.shutil.disk_usage", return_value=mock_usage)

    def test_sufficient_space_returns_true(self, tmp_path):
        with self._patch_usage(free=10_000_000_000):  # 10 GB free
            result = _check_disk_space(tmp_path, required_bytes=5_000_000_000)
        assert result is True

    def test_insufficient_space_returns_false(self, tmp_path):
        with self._patch_usage(free=1_000_000):  # 1 MB free
            result = _check_disk_space(tmp_path, required_bytes=5_000_000_000)
        assert result is False

    def test_exact_space_returns_true(self, tmp_path):
        """Edge: exactly enough space should pass."""
        required = 500_000_000
        with self._patch_usage(free=required):
            result = _check_disk_space(tmp_path, required_bytes=required)
        assert result is True

    def test_zero_required_always_returns_true(self, tmp_path):
        """required_bytes=0 skips the check and always returns True."""
        with self._patch_usage(free=0):
            result = _check_disk_space(tmp_path, required_bytes=0)
        assert result is True

    def test_disk_usage_exception_returns_true(self, tmp_path):
        """If disk_usage raises, check is non-fatal and returns True."""
        with patch("localm.model_manager.shutil.disk_usage", side_effect=OSError("no access")):
            result = _check_disk_space(tmp_path, required_bytes=1_000_000)
        assert result is True

    def test_prints_warning_on_insufficient_space(self, tmp_path, capsys):
        with self._patch_usage(free=1_000_000), \
             patch("localm.model_manager.console") as mock_console:
            _check_disk_space(tmp_path, required_bytes=2_000_000_000)
            mock_console.print.assert_called_once()
            msg = mock_console.print.call_args[0][0]
            assert "disk space" in msg.lower() or "not enough" in msg.lower()


# ---------------------------------------------------------------------------
#  _sha256_file
# ---------------------------------------------------------------------------

class TestSha256File:
    def test_returns_correct_hash(self, tmp_path):
        data = b"hello world\n"
        f = tmp_path / "test.bin"
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert _sha256_file(f) == expected

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert _sha256_file(f) == expected

    def test_large_file(self, tmp_path):
        data = b"x" * 200_000
        f = tmp_path / "large.bin"
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert _sha256_file(f) == expected
