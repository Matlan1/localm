# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native (ggml-registry) GPU device enumeration for the vulkan build's GUI selectors - GPU-SPLIT-VKINDEX follow-up."""

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
    """A 'devices' line gets the JSON inventory; any other line still gets the legacy '<free> <total>' memory reply (protocol back-compat)."""
    fake = [{"index": 0, "name": "Vulkan0",
             "description": "AMD Radeon RX 6900 XT (RADV NAVI21)",
             "type": 1, "free": 15 * GB, "total": 16 * GB}]
    out = _run_daemon_main(monkeypatch, capsys, "devices\nq\n",
                           inventory=fake, memory=(5, 7))
    assert json.loads(out[0]) == fake
    assert out[1] == "5 7"


def test_probe_daemon_devices_err_on_failure(monkeypatch, capsys):
    """An inventory that cannot be taken (None, or a raise) answers ERR - the daemon stays alive for later queries, mirroring the memory reply's contract."""
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
    """Popen-shaped double: stdin/stdout point back at this object; replies are served one per readline."""

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
    """ERR = the daemon is alive and genuinely cannot answer - same contract as gpu_memory_isolated: return None WITHOUT killing the daemon."""
    from localm.inference.backends.llamacpp import _loader
    fake = _install_fake_daemon(monkeypatch, ["ERR\n"])
    assert _loader.gpu_devices_isolated() is None
    assert not fake.killed
    assert _loader._PROBE_PROC is fake


def test_gpu_devices_isolated_desync_kills_daemon(monkeypatch):
    """A non-JSON (or wrong-shape) reply is a protocol desync: do not trust this daemon again - kill and clear so the next call respawns fresh."""
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
    """Regression guard for the shared-roundtrip refactor: the existing memory query still parses '<free> <total>' through the same plumbing."""
    from localm.inference.backends.llamacpp import _loader
    fake = _install_fake_daemon(monkeypatch, ["5 7\n"])
    assert _loader.gpu_memory_isolated() == (5, 7)
    assert fake.requests == ["q\n"]


# ------------------------------------------------------------------ #
#  GUI shaping (discover.native_gpu_devices)                          #
# ------------------------------------------------------------------ #

def test_native_gpu_devices_shapes_for_selectors(monkeypatch):
    """Selector shape: {'index','name'} plus total/free when positive ints and the ggml device type when reported; the human description wins over the backend's terse name; a device with no usable description keeps the name; non-positive memory is omitted (a key the JS checks by type, not by zero)."""
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
         "total": 16 * GB, "free": 15 * GB, "type": 1},
        {"index": 1, "name": "Vulkan1", "type": 1},
    ]


def test_native_gpu_devices_none_when_daemon_cannot_answer(monkeypatch):
    from localm import discover
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr(_loader, "gpu_devices_isolated", lambda: None)
    assert discover.native_gpu_devices() is None


# ------------------------------------------------------------------ #
#  llama.cpp's OWN device list (discover._llama_visible_devices)      #
# ------------------------------------------------------------------ #
#
# native_device_inventory numbers EVERY non-CPU registry device. llama.cpp
# indexes main_gpu / tensor_split into model->devices, which is built by
# llama_prepare_model_devices (src/llama.cpp, read at b10361) as: RPC devices
# hoisted to the front, then GPU-type devices in registry order deduped by
# device_id, then at most ONE integrated GPU and only when no discrete GPU was
# found; CPU and ACCEL are skipped and META aborts fatally. So on a box with a
# discrete card beside integrated graphics the two numberings diverge, and a
# configured split can name cards the user never chose.
#
# NONE OF THIS IS REPRODUCIBLE ON THIS PROJECT'S DEV BOX, which has one
# discrete GPU and no iGPU (compute_devices() -> [("ROCm0", 1), ("CPU", 0)]).
# These are synthetic inventories against a construction read from upstream
# source, which is why the single-GPU and CPU-last shapes below are pinned
# too: they are the arrangement that IS observed here.

# The ggml device-type enum has GROWN: IGPU was inserted AHEAD of ACCEL, so the
# value 2 means ACCEL on a runtime around b6000 and INTEGRATED GPU on anything
# since roughly b8100. Only CPU=0 and GPU=1 have held at every tag sampled, so
# the code allowlists GPU rather than excluding the others by value, and these
# tests deliberately do NOT name 2 as one member or the other - a non-GPU type
# is dropped either way, which is the property under test.
_TYPE_GPU = 1
_TYPE_NOT_GPU = 2


def _native(monkeypatch, raw):
    """Drive the REAL native_gpu_devices() over a synthetic raw inventory."""
    from localm import discover
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr(_loader, "gpu_devices_isolated", lambda: raw)
    return discover.native_gpu_devices()


def test_igpu_before_a_discrete_gpu_shifts_every_index(monkeypatch):
    """THE case that bites: an iGPU enumerating FIRST. llama.cpp drops it (a discrete GPU exists), so its model->devices is [discrete] and the discrete card is device 0 - while the raw inventory calls it device 1."""
    got = _native(monkeypatch, [
        {"index": 0, "name": "Vulkan0", "description": "Intel UHD Graphics",
         "type": _TYPE_NOT_GPU, "free": 2 * GB, "total": 4 * GB},
        {"index": 1, "name": "Vulkan1", "description": "NVIDIA RTX 4090",
         "type": _TYPE_GPU, "free": 20 * GB, "total": 24 * GB},
    ])
    assert got == [{"index": 0, "name": "NVIDIA RTX 4090",
                    "total": 24 * GB, "free": 20 * GB, "type": _TYPE_GPU}]


def test_igpu_after_a_discrete_gpu_is_still_dropped(monkeypatch):
    """Same drop when the iGPU enumerates SECOND."""
    got = _native(monkeypatch, [
        {"index": 0, "name": "Vulkan0", "description": "NVIDIA RTX 4090",
         "type": _TYPE_GPU, "free": 20 * GB, "total": 24 * GB},
        {"index": 1, "name": "Vulkan1", "description": "Intel UHD Graphics",
         "type": _TYPE_NOT_GPU, "free": 2 * GB, "total": 4 * GB},
    ])
    assert [d["index"] for d in got] == [0]
    assert [d["name"] for d in got] == ["NVIDIA RTX 4090"]


def test_second_discrete_card_becomes_reachable_across_an_igpu(monkeypatch):
    """An iGPU BETWEEN two discrete cards."""
    got = _native(monkeypatch, [
        {"index": 0, "name": "Vulkan0", "description": "RTX 4090",
         "type": _TYPE_GPU, "free": 20 * GB, "total": 24 * GB},
        {"index": 1, "name": "Vulkan1", "description": "Intel UHD Graphics",
         "type": _TYPE_NOT_GPU, "free": 2 * GB, "total": 4 * GB},
        {"index": 2, "name": "Vulkan2", "description": "RTX 3090",
         "type": _TYPE_GPU, "free": 18 * GB, "total": 24 * GB},
    ])
    assert [(d["index"], d["name"]) for d in got] == [
        (0, "RTX 4090"), (1, "RTX 3090")]


def test_igpu_only_box_is_left_exactly_as_it_was(monkeypatch):
    """NO REGRESSION on the commonest affected machine."""
    raw = [{"index": 0, "name": "Vulkan0", "description": "Intel Iris Xe",
            "type": _TYPE_NOT_GPU, "free": 6 * GB, "total": 8 * GB}]
    assert _native(monkeypatch, raw) == [
        {"index": 0, "name": "Intel Iris Xe", "total": 8 * GB,
         "free": 6 * GB, "type": _TYPE_NOT_GPU}]


def test_device_with_no_reported_type_is_not_assumed_discrete(monkeypatch):
    """The probe omits 'type' when the registry did not report an int."""
    got = _native(monkeypatch, [
        {"index": 0, "name": "Vulkan0", "description": "mystery",
         "free": 1 * GB, "total": 2 * GB},
    ])
    assert [d["index"] for d in got] == [0]
    assert "type" not in got[0]


def test_untyped_device_beside_a_discrete_gpu_is_dropped(monkeypatch):
    """The discriminating half of the case above: once a GPU-type device DOES exist, an untyped one is excluded rather than counted."""
    got = _native(monkeypatch, [
        {"index": 0, "name": "Vulkan0", "description": "mystery",
         "free": 1 * GB, "total": 2 * GB},
        {"index": 1, "name": "Vulkan1", "description": "RTX 4090",
         "type": _TYPE_GPU, "free": 20 * GB, "total": 24 * GB},
    ])
    assert [(d["index"], d["name"]) for d in got] == [(0, "RTX 4090")]


def test_all_discrete_multi_gpu_is_untouched(monkeypatch):
    """The ordinary multi-GPU board: every device is a discrete GPU, so the derivation is the identity and a configured [0, 1] keeps meaning what it always did."""
    got = _native(monkeypatch, [
        {"index": 0, "name": "Vulkan0", "description": "RTX 4090",
         "type": _TYPE_GPU, "free": 20 * GB, "total": 24 * GB},
        {"index": 1, "name": "Vulkan1", "description": "RTX 3090",
         "type": _TYPE_GPU, "free": 18 * GB, "total": 24 * GB},
    ])
    assert [(d["index"], d["name"]) for d in got] == [
        (0, "RTX 4090"), (1, "RTX 3090")]


def test_dev_box_single_discrete_gpu_shape_is_unchanged(monkeypatch):
    """Pinned to the shape actually MEASURED on this project's dev box, where compute_devices() reports [('ROCm0', 1), ('CPU', 0)] - the GPU FIRST and the CPU LAST, which is the opposite of the instinctive fixture."""
    assert _native(monkeypatch, [
        {"index": 0, "name": "ROCm0",
         "description": "AMD Radeon RX 6900 XT", "type": _TYPE_GPU,
         "free": 15 * GB, "total": 16 * GB},
    ]) == [{"index": 0, "name": "AMD Radeon RX 6900 XT",
            "total": 16 * GB, "free": 15 * GB, "type": _TYPE_GPU}]


def test_empty_inventory_stays_empty(monkeypatch):
    """An empty list is a real answer (the runtime registers no non-CPU device) and must not become None, which means 'could not look'."""
    assert _native(monkeypatch, []) == []
