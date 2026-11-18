"""list_gpus keeps the device list the out-of-process torch probe answered with.

A model load and unload each read the GPUs several times. On a box where torch
is installed but not imported in the server (every Windows AMD install with a
ROCm torch), each of those reads used to start a child that imports torch, which
takes seconds and can run past its time limit while the disk is busy loading the
model. Only the free VRAM changes between reads, and it has device-global
sources that never import torch, so the child is asked once and asked again only
when a re-enumeration is requested, the device-visibility environment changes,
or a free reading cannot be re-read for every device.
"""
from __future__ import annotations

import logging
import os
import pathlib
import threading

import pytest

from localm import discover, gpu_usage

GB = 1024 ** 3

_STUB_TORCH = '''
import os
with open(os.environ["PROBE_COUNT_FILE"], "a", encoding="utf-8") as _fh:
    _fh.write("import\\n")
_N = int(os.environ.get("PROBE_STUB_DEVICES", "1"))


class _Props:
    def __init__(self, i):
        self.pci_bus_id = 3 + i
        self.is_integrated = 0


class _Cuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return _N

    @staticmethod
    def mem_get_info(i):
        return (6 * 2**30, 8 * 2**30)

    @staticmethod
    def get_device_name(i):
        return "STUB GPU %d" % i

    @staticmethod
    def get_device_properties(i):
        return _Props(i)


cuda = _Cuda()
'''


@pytest.fixture
def torch_free_sources(monkeypatch):
    """The device-global free source answers ``used[index]`` for every device
    asked about; nvidia-smi is absent; torch is not resident and not doomed."""
    used = {"bytes": 2 * GB}
    monkeypatch.setattr(discover, "_torch_is_resident", lambda: False)
    monkeypatch.setattr(discover, "_torch_gpu_probe_known_doomed", lambda: False)
    monkeypatch.setattr(gpu_usage, "raw_reading_is_process_scoped", lambda: False)
    monkeypatch.setattr(gpu_usage, "source_is_warm", lambda: True)
    monkeypatch.setattr(
        gpu_usage, "device_global_used_bytes",
        lambda gpus: {g["index"]: used["bytes"] for g in gpus})
    from localm import sysprog
    monkeypatch.setattr(sysprog, "find", lambda name: None)
    return used


@pytest.fixture
def stub_torch_child(monkeypatch, tmp_path):
    """The real torch probe child, importing a stub torch that records each
    import in a file. Returns a callable giving the number of children run."""
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    (stub_dir / "torch.py").write_text(_STUB_TORCH, encoding="utf-8")
    repo = str(pathlib.Path(discover.__file__).resolve().parent.parent)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(stub_dir), repo]))
    count_file = tmp_path / "imports.txt"
    monkeypatch.setenv("PROBE_COUNT_FILE", str(count_file))
    monkeypatch.setenv("PROBE_STUB_DEVICES", "1")

    def spawns() -> int:
        if not count_file.exists():
            return 0
        return len(count_file.read_text(encoding="utf-8").splitlines())
    return spawns


def _child_answering(monkeypatch, answers):
    """Replace the torch probe child with one answering from *answers* in
    turn; returns the list of calls made."""
    calls = []

    def _isolated():
        calls.append(1)
        value = answers[min(len(calls), len(answers)) - 1]
        if isinstance(value, Exception):
            raise value
        return [dict(d) for d in value] if isinstance(value, list) else value
    monkeypatch.setattr(discover, "_torch_gpus_isolated", _isolated)
    return calls


_ONE = [{"index": 0, "name": "GPU 0", "total": 8 * GB, "free": 6 * GB,
         "pci_bus_id": 3}]


class TestKeptDeviceList:

    def test_three_load_unload_reads_start_one_real_torch_child(
            self, torch_free_sources, stub_torch_child):
        """The load/unload path's repeated list_gpus calls start the real torch
        probe child once, and every read still follows the device's live used
        VRAM. Break: drop the kept-list return in _list_gpus_probe and this
        reports three children. The first read uses the stub's own 2 GB used,
        which the driver reading off Windows and the device-global correction
        on Windows both give."""
        seen = []
        for used in (2, 3, 5):
            torch_free_sources["bytes"] = used * GB
            gpus, status = discover.list_gpus(return_status=True)
            assert status == discover.GPU_PROBE_OK
            seen.append(gpus[0]["free"])
            assert gpus[0]["free_scope"] == discover.FREE_SCOPE_DEVICE
            assert gpus[0]["name"] == "STUB GPU 0"
        assert stub_torch_child() == 1
        assert seen == [6 * GB, 5 * GB, 3 * GB]

    def test_a_requested_re_enumeration_asks_the_real_child_again(
            self, monkeypatch, torch_free_sources, stub_torch_child):
        discover.list_gpus()
        monkeypatch.setenv("PROBE_STUB_DEVICES", "2")
        assert len(discover.list_gpus()) == 1
        discover.request_gpu_reenumeration()
        assert [g["index"] for g in discover.list_gpus()] == [0, 1]
        assert stub_torch_child() == 2
        discover.list_gpus()
        assert stub_torch_child() == 2

    def test_a_child_that_cannot_answer_a_re_enumeration_keeps_the_list(
            self, monkeypatch, torch_free_sources):
        calls = _child_answering(monkeypatch, [_ONE, discover._IsolatedTorchWedged()])
        discover.list_gpus()
        discover.request_gpu_reenumeration()
        torch_free_sources["bytes"] = 5 * GB
        gpus, status = discover.list_gpus(return_status=True)
        assert len(calls) == 2
        assert status == discover.GPU_PROBE_OK
        assert gpus[0]["name"] == "GPU 0" and gpus[0]["free"] == 3 * GB

    def test_an_honest_empty_answer_drops_the_list(self, monkeypatch,
                                                   torch_free_sources):
        calls = _child_answering(monkeypatch, [_ONE, []])
        discover.list_gpus()
        discover.request_gpu_reenumeration()
        assert discover.list_gpus() == []
        discover.list_gpus()
        assert len(calls) == 3

    def test_a_changed_device_visibility_asks_the_child_again(
            self, monkeypatch, torch_free_sources):
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
        calls = _child_answering(monkeypatch, [_ONE])
        discover.list_gpus()
        discover.list_gpus()
        assert len(calls) == 1
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
        discover.list_gpus()
        assert len(calls) == 2

    def test_a_device_whose_free_cannot_be_re_read_asks_every_round(
            self, monkeypatch, torch_free_sources):
        monkeypatch.setattr(gpu_usage, "device_global_used_bytes", lambda gpus: {})
        calls = _child_answering(monkeypatch, [_ONE])
        for _ in range(3):
            discover.list_gpus()
        assert len(calls) == 3

    def test_a_failed_torch_answer_does_not_replace_the_list(
            self, monkeypatch, torch_free_sources):
        """A child that exited non-zero answers [] for "torch failed", which
        says nothing about the devices."""
        calls = _child_answering(monkeypatch, [_ONE, []])

        def _failed():
            calls.append(1)
            discover._note_torch_round_unanswered()
            return []
        discover.list_gpus()
        monkeypatch.setattr(discover, "_torch_gpus_isolated", _failed)
        discover.request_gpu_reenumeration()
        assert discover.list_gpus()[0]["name"] == "GPU 0"
        assert discover.list_gpus()[0]["name"] == "GPU 0"
        assert len(calls) == 2

    def test_the_reset_drops_the_list(self, monkeypatch, torch_free_sources):
        calls = _child_answering(monkeypatch, [_ONE])
        discover.list_gpus()
        discover._reset_gpu_probe_cache()
        discover.list_gpus()
        assert len(calls) == 2


class TestProbeLogging:

    def test_the_real_child_logs_how_long_it_took(
            self, caplog, torch_free_sources, stub_torch_child):
        with caplog.at_level(logging.DEBUG, logger="localm"):
            discover.list_gpus()
        lines = [r.getMessage() for r in caplog.records
                 if "torch probe child (pid" in r.getMessage()]
        assert len(lines) == 1
        assert " took " in lines[0] and lines[0].endswith(": 1 device(s)")

    def test_a_busy_answer_is_logged(self, monkeypatch, caplog):
        release = threading.Event()
        entered = threading.Event()

        def _slow_probe():
            entered.set()
            release.wait(10)
            return _ONE
        monkeypatch.setattr(discover, "_list_gpus_probe", _slow_probe)
        starter = threading.Thread(target=discover.list_gpus, daemon=True)
        starter.start()
        assert entered.wait(10)
        try:
            with caplog.at_level(logging.DEBUG, logger="localm"):
                _gpus, status = discover.list_gpus(return_status=True)
        finally:
            release.set()
            starter.join(10)
        assert status == discover.GPU_PROBE_BUSY
        assert any("answering busy" in r.getMessage() for r in caplog.records)


class TestUnloadReading:

    def test_the_unload_reading_joins_a_probe_in_flight(self, monkeypatch):
        """Break: give _vram_free_reading the plain (non-joining) vram_capacity
        call back and this reads busy, not fresh."""
        import time

        from localm import vram
        release = threading.Event()
        entered = threading.Event()
        reader_in = threading.Event()

        def _slow_probe():
            entered.set()
            release.wait(10)
            return [dict(_ONE[0], free_scope=discover.FREE_SCOPE_DEVICE)]
        monkeypatch.setattr(discover, "_list_gpus_probe", _slow_probe)
        starter = threading.Thread(target=discover.list_gpus, daemon=True)
        starter.start()
        assert entered.wait(10)
        real_list_gpus = discover.list_gpus

        def _spy(*args, **kwargs):
            if threading.current_thread().name == "unload-reader":
                reader_in.set()
            return real_list_gpus(*args, **kwargs)
        monkeypatch.setattr(discover, "list_gpus", _spy)
        got = {}
        reader = threading.Thread(
            target=lambda: got.setdefault("r", vram._vram_free_reading()),
            name="unload-reader", daemon=True)
        reader.start()
        try:
            assert reader_in.wait(10)
            time.sleep(0.2)
        finally:
            release.set()
        reader.join(20)
        starter.join(20)
        _free, fresh, _scope = got["r"]
        assert vram.free_reading_status(got["r"]) == discover.GPU_PROBE_OK
        assert fresh is True

    def test_a_plain_tuple_has_no_status(self):
        from localm import vram
        assert vram.free_reading_status((None, True, None)) is None

    @pytest.mark.parametrize("status, words", [
        (discover.GPU_PROBE_BUSY, "probe status busy"),
        (discover.GPU_PROBE_TIMEOUT, "the GPU probe timed out when"),
        (discover.GPU_PROBE_INCONCLUSIVE, "probe status inconclusive"),
        (None, "timed out or was busy"),
    ])
    def test_the_vram_note_names_the_probe_status(self, status, words):
        from localm.inference.http_server import _add_vram_fields
        result: dict = {}
        _add_vram_fields(result, before=4 * GB, released=True, after=6 * GB,
                         before_fresh=False,
                         before_scope=discover.FREE_SCOPE_DEVICE,
                         before_status=status)
        assert result["vram_reading_uncertain"] is True
        assert words in result["vram_note"]
