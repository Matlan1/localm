# SPDX-License-Identifier: AGPL-3.0-or-later
"""sysstats.system_ram: the (total, available) RAM reading the mmap decision
uses. Read through psutil when installed, else the platform's own interface;
a memory limit on this process's cgroup or an ancestor caps a known reading on
Linux. A failed read is (None, None), never an exception and never a made-up
figure.

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


def test_with_nothing_readable_the_whole_reading_is_unknown(monkeypatch, tmp_path):
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    (tmp_path / "memory.max").write_text(f"{8 * GIB}\n")
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(sysstats, "_platform_ram", lambda: (None, None))
    monkeypatch.setattr(sysstats, "_CGROUP_ROOT", tmp_path)
    monkeypatch.setattr(sysstats, "_PROC_SELF_CGROUP", tmp_path / "no-such-proc-file")
    monkeypatch.setattr(sysstats.sys, "platform", "linux")
    assert sysstats.system_ram() == (None, None)


def _proc(tmp_path, text):
    p = tmp_path / "proc_self_cgroup"
    p.write_text(text, encoding="ascii")
    return p


def _level(d, *, limit=None, usage=None, inactive=None, v2=True):
    d.mkdir(parents=True, exist_ok=True)
    if limit is not None:
        (d / ("memory.max" if v2 else "memory.limit_in_bytes")).write_text(f"{limit}\n")
    if usage is not None:
        (d / ("memory.current" if v2 else "memory.usage_in_bytes")).write_text(f"{usage}\n")
    if inactive is not None:
        key = "inactive_file" if v2 else "total_inactive_file"
        (d / "memory.stat").write_text(f"anon 1\nfile 2\n{key} {inactive}\nactive_file 3\n")


class TestCgroupLevels:
    def test_v2_own_cgroup_and_every_ancestor_innermost_first(self, tmp_path):
        proc = _proc(tmp_path, "0::/docker/abc\n")
        assert sysstats._cgroup_levels(tmp_path, proc) == [
            (tmp_path / "docker" / "abc", True), (tmp_path / "docker", True), (tmp_path, True)]

    def test_v1_memory_controller_entry(self, tmp_path):
        proc = _proc(tmp_path, "5:cpu,cpuacct:/x\n4:memory:/user.slice\n")
        base = tmp_path / "memory"
        assert sysstats._cgroup_levels(tmp_path, proc) == [
            (base / "user.slice", False), (base, False)]

    def test_a_private_namespace_root_is_the_mount_root(self, tmp_path):
        assert sysstats._cgroup_levels(tmp_path, _proc(tmp_path, "0::/\n")) == [(tmp_path, True)]

    def test_a_path_above_the_namespace_root_falls_back_to_the_mounts(self, tmp_path):
        proc = _proc(tmp_path, "0::/../../user.slice\n")
        assert sysstats._cgroup_levels(tmp_path, proc) == [
            (tmp_path, True), (tmp_path / "memory", False)]

    def test_an_unreadable_proc_file_falls_back_to_the_mounts(self, tmp_path):
        assert sysstats._cgroup_levels(tmp_path, tmp_path / "gone") == [
            (tmp_path, True), (tmp_path / "memory", False)]


class TestCgroupMemory:
    def test_reclaimable_page_cache_is_not_usage(self, tmp_path):
        _level(tmp_path / "c", limit=8 * GIB, usage=5 * GIB, inactive=3 * GIB)
        proc = _proc(tmp_path, "0::/c\n")
        assert sysstats._cgroup_memory(tmp_path, proc) == (8 * GIB, 6 * GIB)

    def test_a_tighter_ancestor_limit_wins(self, tmp_path):
        _level(tmp_path / "slice", limit=4 * GIB, usage=3 * GIB)
        _level(tmp_path / "slice" / "unit", limit=8 * GIB, usage=2 * GIB)
        proc = _proc(tmp_path, "0::/slice/unit\n")
        assert sysstats._cgroup_memory(tmp_path, proc) == (4 * GIB, 1 * GIB)

    def test_a_limit_only_on_the_own_cgroup_is_found(self, tmp_path):
        _level(tmp_path / "system.slice" / "app.service", limit=2 * GIB, usage=1 * GIB)
        proc = _proc(tmp_path, "0::/system.slice/app.service\n")
        assert sysstats._cgroup_memory(tmp_path, proc) == (2 * GIB, 1 * GIB)

    def test_v2_max_everywhere_is_no_limit(self, tmp_path):
        (tmp_path / "c").mkdir()
        (tmp_path / "c" / "memory.max").write_text("max\n")
        (tmp_path / "c" / "memory.current").write_text(f"{GIB}\n")
        assert sysstats._cgroup_memory(tmp_path, _proc(tmp_path, "0::/c\n")) == (None, None)

    def test_v1_limit_usage_and_cache(self, tmp_path):
        _level(tmp_path / "memory" / "u", limit=4 * GIB, usage=2 * GIB, inactive=GIB, v2=False)
        proc = _proc(tmp_path, "4:memory:/u\n")
        assert sysstats._cgroup_memory(tmp_path, proc) == (4 * GIB, 3 * GIB)

    def test_v1_unlimited_sentinel_is_no_limit(self, tmp_path):
        _level(tmp_path / "memory", limit=9223372036854771712, usage=GIB, v2=False)
        proc = _proc(tmp_path, "4:memory:/\n")
        assert sysstats._cgroup_memory(tmp_path, proc) == (None, None)

    def test_unreadable_usage_leaves_the_whole_limit(self, tmp_path):
        _level(tmp_path / "c", limit=3 * GIB)
        assert sysstats._cgroup_memory(tmp_path, _proc(tmp_path, "0::/c\n")) == (3 * GIB, 3 * GIB)

    def test_usage_above_the_limit_leaves_nothing(self, tmp_path):
        _level(tmp_path / "c", limit=GIB, usage=2 * GIB)
        assert sysstats._cgroup_memory(tmp_path, _proc(tmp_path, "0::/c\n")) == (GIB, 0)

    def test_no_controller_files(self, tmp_path):
        assert sysstats._cgroup_memory(tmp_path, _proc(tmp_path, "0::/\n")) == (None, None)


class TestCapToCgroup:
    def test_a_limit_below_the_host_caps_both(self):
        assert sysstats._cap_to_cgroup(64 * GIB, 40 * GIB, (8 * GIB, 2 * GIB)) == (8 * GIB, 2 * GIB)

    def test_the_host_reading_wins_when_it_is_smaller(self):
        assert sysstats._cap_to_cgroup(64 * GIB, 1 * GIB, (8 * GIB, 6 * GIB)) == (8 * GIB, 1 * GIB)

    def test_a_limit_at_or_above_the_host_changes_nothing(self):
        assert sysstats._cap_to_cgroup(64 * GIB, 40 * GIB, (64 * GIB, 1)) == (64 * GIB, 40 * GIB)

    @pytest.mark.parametrize("cgroup", [(8 * GIB, 6 * GIB),
                                        (9223372036854771712, 9223372036854771711)])
    def test_an_unknown_host_reading_stays_unknown(self, cgroup):
        assert sysstats._cap_to_cgroup(None, None, cgroup) == (None, None)

    def test_an_unknown_available_takes_the_room(self):
        assert sysstats._cap_to_cgroup(64 * GIB, None, (8 * GIB, 6 * GIB)) == (8 * GIB, 6 * GIB)

    def test_no_limit_changes_nothing(self):
        assert sysstats._cap_to_cgroup(64 * GIB, 40 * GIB, (None, None)) == (64 * GIB, 40 * GIB)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="cgroups are Linux-only")
def test_system_ram_applies_the_own_cgroup_limit(tmp_path, monkeypatch):
    _level(tmp_path / "app", limit=512 * 1024 ** 2, usage=128 * 1024 ** 2)
    monkeypatch.setattr(sysstats, "_CGROUP_ROOT", tmp_path)
    monkeypatch.setattr(sysstats, "_PROC_SELF_CGROUP", _proc(tmp_path, "0::/app\n"))
    assert sysstats.system_ram() == (512 * 1024 ** 2, 384 * 1024 ** 2)
