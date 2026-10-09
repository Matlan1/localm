# SPDX-License-Identifier: AGPL-3.0-or-later
"""sysstats.system_ram: the (total, available) RAM reading the mmap decision
uses. Read through psutil when installed, else the platform's own interface;
a cgroup memory limit caps both on Linux. A failed read is (None, None), never
an exception and never a made-up figure.

The platform paths run for real on the platform the test runs on; the
/proc/meminfo and cgroup parsers also run against files in their real formats.
"""

import sys

import pytest

from localm import sysstats

GIB = 1024 ** 3


def test_the_reading_on_this_machine_is_plausible():
    total, available = sysstats.system_ram()
    assert isinstance(total, int) and total > 256 * 1024 ** 2
    assert isinstance(available, int) and 0 < available <= total


def test_without_psutil_the_platform_interface_answers(monkeypatch):
    if sys.platform not in ("win32",) and not sys.platform.startswith("linux"):
        pytest.skip("available RAM has no psutil-free reading on this platform")
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert sysstats._psutil_ram() == (None, None)
    total, available = sysstats.system_ram()
    assert isinstance(total, int) and total > 256 * 1024 ** 2
    assert isinstance(available, int) and 0 < available <= total


@pytest.mark.skipif(sys.platform != "win32", reason="GlobalMemoryStatusEx is Windows-only")
def test_global_memory_status_total_matches_psutil():
    psutil = pytest.importorskip("psutil")
    total, available = sysstats._windows_ram()
    assert total == psutil.virtual_memory().total
    assert 0 < available <= total


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc/meminfo is Linux-only")
def test_proc_meminfo_total_matches_psutil():
    psutil = pytest.importorskip("psutil")
    total, available = sysstats._meminfo_ram(sysstats._PROC_MEMINFO)
    assert total == psutil.virtual_memory().total
    assert 0 < available <= total


def test_meminfo_parser_reads_kib_lines(tmp_path):
    p = tmp_path / "meminfo"
    p.write_text("MemTotal:       32768000 kB\n"
                 "MemFree:         1000000 kB\n"
                 "MemAvailable:   16384000 kB\n"
                 "Buffers:          200000 kB\n", encoding="ascii")
    assert sysstats._meminfo_ram(p) == (32768000 * 1024, 16384000 * 1024)


def test_meminfo_without_mem_available_reports_it_unknown(tmp_path):
    p = tmp_path / "meminfo"
    p.write_text("MemTotal:       32768000 kB\nMemFree:  1 kB\n", encoding="ascii")
    assert sysstats._meminfo_ram(p) == (32768000 * 1024, None)


def test_a_platform_read_failure_is_unknown_not_an_exception(monkeypatch):
    monkeypatch.setattr(sysstats.sys, "platform", "sunos5")

    def _boom(_name):
        raise ValueError("unsupported")

    monkeypatch.setattr(sysstats.os, "sysconf", _boom, raising=False)
    assert sysstats._platform_ram() == (None, None)


def test_with_nothing_readable_the_whole_reading_is_unknown(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(sysstats, "_platform_ram", lambda: (None, None))
    monkeypatch.setattr(sysstats, "_CGROUP_ROOT", sysstats.Path("/nonexistent-cgroup-root"))
    assert sysstats.system_ram() == (None, None)


class TestCgroup:
    def test_v2_limit_and_usage(self, tmp_path):
        (tmp_path / "memory.max").write_text("8589934592\n")
        (tmp_path / "memory.current").write_text("2147483648\n")
        assert sysstats._cgroup_memory(tmp_path) == (8 * GIB, 2 * GIB)

    def test_v2_max_means_no_limit(self, tmp_path):
        (tmp_path / "memory.max").write_text("max\n")
        (tmp_path / "memory.current").write_text("2147483648\n")
        assert sysstats._cgroup_memory(tmp_path) == (None, None)

    def test_v1_limit_and_usage(self, tmp_path):
        (tmp_path / "memory").mkdir()
        (tmp_path / "memory" / "memory.limit_in_bytes").write_text("4294967296\n")
        (tmp_path / "memory" / "memory.usage_in_bytes").write_text("1073741824\n")
        assert sysstats._cgroup_memory(tmp_path) == (4 * GIB, 1 * GIB)

    def test_no_controller_files(self, tmp_path):
        assert sysstats._cgroup_memory(tmp_path) == (None, None)

    def test_a_limit_below_the_host_caps_both(self):
        assert sysstats._cap_to_cgroup(64 * GIB, 40 * GIB, (8 * GIB, 6 * GIB)) == (8 * GIB, 2 * GIB)

    def test_the_host_reading_wins_when_it_is_smaller(self):
        assert sysstats._cap_to_cgroup(64 * GIB, 1 * GIB, (8 * GIB, 2 * GIB)) == (8 * GIB, 1 * GIB)

    def test_usage_above_the_limit_leaves_nothing(self):
        assert sysstats._cap_to_cgroup(64 * GIB, 40 * GIB, (8 * GIB, 9 * GIB)) == (8 * GIB, 0)

    def test_v1_unlimited_sentinel_changes_nothing(self):
        assert sysstats._cap_to_cgroup(64 * GIB, 40 * GIB, (9223372036854771712, 1)) == (64 * GIB, 40 * GIB)

    def test_unknown_usage_caps_available_at_the_limit(self):
        assert sysstats._cap_to_cgroup(64 * GIB, 40 * GIB, (8 * GIB, None)) == (8 * GIB, 8 * GIB)

    def test_an_unknown_host_reading_takes_the_cgroup(self):
        assert sysstats._cap_to_cgroup(None, None, (8 * GIB, 2 * GIB)) == (8 * GIB, 6 * GIB)

    def test_no_limit_changes_nothing(self):
        assert sysstats._cap_to_cgroup(64 * GIB, 40 * GIB, (None, None)) == (64 * GIB, 40 * GIB)

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="cgroups are Linux-only")
    def test_system_ram_applies_the_cgroup_limit(self, tmp_path, monkeypatch):
        (tmp_path / "memory.max").write_text(f"{512 * 1024 ** 2}\n")
        (tmp_path / "memory.current").write_text(f"{128 * 1024 ** 2}\n")
        monkeypatch.setattr(sysstats, "_CGROUP_ROOT", tmp_path)
        assert sysstats.system_ram() == (512 * 1024 ** 2, 384 * 1024 ** 2)
