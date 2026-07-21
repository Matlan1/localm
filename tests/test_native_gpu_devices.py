# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native (ggml-registry) GPU device enumeration for the vulkan build's GUI
selectors - GPU-SPLIT-VKINDEX follow-up.

On the vulkan build the split/main-GPU indices the loader consumes live in
ggml-vulkan's own index space, which ``list_gpus()`` (torch.cuda / nvidia-smi)
is structurally blind to - so the Settings selectors, which are populated from
``list_gpus()``, could never offer or display a split there. The fix chain
under test:

- ``_vram_probe`` (the crash-isolated probe daemon) answers a new ``devices``
  request with a one-line JSON inventory of the ggml registry's non-CPU
  devices, alongside the existing any-other-line -> "<free> <total>" memory
  reply.
- ``_loader.gpu_devices_isolated()`` performs that round-trip via the same
  daemon plumbing as ``gpu_memory_isolated()`` (shared lock/spawn/timeout/
  desync handling).
- ``discover.native_gpu_devices()`` shapes the inventory for the GUI/selector
  consumers ({"index", "name", "total", "free"}), preferring the human
  description over the backend's terse name.

The daemon round-trip here runs against a FAKE daemon object (no native code,
no subprocess): the real native enumeration is exercised by the live check
documented in dev-notes/checkup/REPORT-2026-07-21-gpu-split.md.
"""

from __future__ import annotations

import io
import json
import sys

GB = 1024 ** 3


# ------------------------------------------------------------------ #
#  Daemon dispatch (_vram_probe.main)                                 #
# ------------------------------------------------------------------ #

def _run_daemon_main(monkeypatch, capsys, stdin_text, *, inventory, memory=(1, 2)):
    from localm.inference.backends.llamacpp import _loader, _vram_probe
    monkeypatch.setattr(_loader, "load_lib", lambda: None)
    if isinstance(inventory, Exception):
        def _raise():
            raise inventory
        monkeypatch.setattr(_loader, "native_device_inventory", _raise)
    else:
        monkeypatch.setattr(_loader, "native_device_inventory", lambda: inventory)
    monkeypatch.setattr(_loader, "gpu_memory", lambda: memory)
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_text))
    assert _vram_probe.main() == 0
    return capsys.readouterr().out.strip().splitlines()


def test_probe_daemon_answers_devices_and_memory(monkeypatch, capsys):
    """A "devices" line gets the JSON inventory; any other line still gets the
    legacy "<free> <total>" memory reply (protocol back-compat)."""
    fake = [{"index": 0, "name": "Vulkan0",
             "description": "AMD Radeon RX 6900 XT (RADV NAVI21)",
             "type": 1, "free": 15 * GB, "total": 16 * GB}]
    out = _run_daemon_main(monkeypatch, capsys, "devices\nq\n",
                           inventory=fake, memory=(5, 7))
    assert json.loads(out[0]) == fake
    assert out[1] == "5 7"


def test_probe_daemon_devices_err_on_failure(monkeypatch, capsys):
    """An inventory that cannot be taken (None, or a raise) answers ERR - the
    daemon stays alive for later queries, mirroring the memory reply's
    contract."""
    out = _run_daemon_main(monkeypatch, capsys, "devices\ndevices\n",
                           inventory=RuntimeError("boom"))
    assert out == ["ERR", "ERR"]


def test_probe_daemon_devices_none_is_err(monkeypatch, capsys):
    out = _run_daemon_main(monkeypatch, capsys, "devices\n", inventory=None)
    assert out == ["ERR"]


# ------------------------------------------------------------------ #
#  Isolated round-trip (_loader.gpu_devices_isolated)                 #
# ------------------------------------------------------------------ #

class _FakeDaemon:
    """Popen-shaped double: stdin/stdout point back at this object; replies are
    served one per readline. No subprocess, no native code."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.requests: list = []
        self.killed = False
        self.stdin = self
        self.stdout = self

    def write(self, s):
        self.requests.append(s)

    def flush(self):
        pass

    def poll(self):
        return None

    def kill(self):
        self.killed = True

    def readline(self):
        return self._replies.pop(0) if self._replies else ""


def _install_fake_daemon(monkeypatch, replies):
    from localm.inference.backends.llamacpp import _loader
    fake = _FakeDaemon(replies)
    monkeypatch.setattr(_loader, "_PROBE_PROC", None)
    monkeypatch.setattr(_loader, "_spawn_probe_daemon", lambda: fake)
    return fake


def test_gpu_devices_isolated_parses_inventory(monkeypatch):
    from localm.inference.backends.llamacpp import _loader
    inv = [{"index": 0, "name": "Vulkan0", "description": "AMD X", "type": 1,
            "free": 1, "total": 2},
           {"index": 1, "name": "Vulkan1", "description": "llvmpipe", "type": 1,
            "free": 3, "total": 4}]
    fake = _install_fake_daemon(monkeypatch, [json.dumps(inv) + "\n"])
    assert _loader.gpu_devices_isolated() == inv
    assert fake.requests == ["devices\n"]


def test_gpu_devices_isolated_err_keeps_daemon(monkeypatch):
    """ERR = the daemon is alive and genuinely cannot answer - same contract as
    gpu_memory_isolated: return None WITHOUT killing the daemon."""
    from localm.inference.backends.llamacpp import _loader
    fake = _install_fake_daemon(monkeypatch, ["ERR\n"])
    assert _loader.gpu_devices_isolated() is None
    assert not fake.killed
    assert _loader._PROBE_PROC is fake


def test_gpu_devices_isolated_desync_kills_daemon(monkeypatch):
    """A non-JSON (or wrong-shape) reply is a protocol desync: do not trust
    this daemon again - kill and clear so the next call respawns fresh."""
    from localm.inference.backends.llamacpp import _loader
    fake = _install_fake_daemon(monkeypatch, ["garbage not json\n"])
    assert _loader.gpu_devices_isolated() is None
    assert fake.killed
    assert _loader._PROBE_PROC is None


def test_gpu_devices_isolated_wrong_shape_is_desync(monkeypatch):
    from localm.inference.backends.llamacpp import _loader
    fake = _install_fake_daemon(monkeypatch, ['{"not": "a list"}\n'])
    assert _loader.gpu_devices_isolated() is None
    assert fake.killed


def test_gpu_memory_isolated_still_parses_memory(monkeypatch):
    """Regression guard for the shared-roundtrip refactor: the existing memory
    query still parses "<free> <total>" through the same plumbing."""
    from localm.inference.backends.llamacpp import _loader
    fake = _install_fake_daemon(monkeypatch, ["5 7\n"])
    assert _loader.gpu_memory_isolated() == (5, 7)
    assert fake.requests == ["q\n"]


# ------------------------------------------------------------------ #
#  GUI shaping (discover.native_gpu_devices)                          #
# ------------------------------------------------------------------ #

def test_native_gpu_devices_shapes_for_selectors(monkeypatch):
    """Selector shape: {"index","name"} plus total/free when positive ints;
    the human description wins over the backend's terse name; a device with
    no usable description keeps the name; non-positive memory is omitted (a
    key the JS checks by type, not by zero)."""
    from localm import discover
    from localm.inference.backends.llamacpp import _loader
    raw = [
        {"index": 0, "name": "Vulkan0",
         "description": "AMD Radeon RX 6900 XT (RADV NAVI21)",
         "type": 1, "free": 15 * GB, "total": 16 * GB},
        {"index": 1, "name": "Vulkan1", "description": "", "type": 1,
         "free": 0, "total": 0},
    ]
    monkeypatch.setattr(_loader, "gpu_devices_isolated", lambda: raw)
    assert discover.native_gpu_devices() == [
        {"index": 0, "name": "AMD Radeon RX 6900 XT (RADV NAVI21)",
         "total": 16 * GB, "free": 15 * GB},
        {"index": 1, "name": "Vulkan1"},
    ]


def test_native_gpu_devices_none_when_daemon_cannot_answer(monkeypatch):
    from localm import discover
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr(_loader, "gpu_devices_isolated", lambda: None)
    assert discover.native_gpu_devices() is None
