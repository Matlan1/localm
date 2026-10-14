# SPDX-License-Identifier: AGPL-3.0-or-later
"""A native media job frees this process's OTHER native media runtime when the
card is short, and names another localm instance that holds VRAM.

Music (KoboldCpp) and image/video (the shared stable-diffusion.cpp worker) each
stay loaded after their job. These tests drive the real helpers in
``localm.vram`` against the real runtime modules' state functions
(``server.running_pid`` / ``server.stop``, ``shared.worker_pid`` /
``shared.free``), faked at that seam, so a wrong runtime being stopped or a stop
called when the card is fine shows up as a failed assertion.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from localm import vram
from localm.discover import FREE_SCOPE_DEVICE, FREE_SCOPE_PROCESS
from localm.media.koboldcpp import server as kobold_server
from localm.media.sdcpp import shared as sd_shared

GB = 1024 ** 3


class _Job:
    def __init__(self):
        self.lines = []
        self.cancel_requested = False
        self.outcomes = []
        self.result = None

    def push(self, ev):
        self.lines.append(ev["text"])

    def mark_outcome(self, status):
        self.outcomes.append(status)


class _Runtimes:
    """Fake the two native runtimes at the module functions the helper calls."""

    def __init__(self, monkeypatch, *, music_loaded=False, sd_loaded=False,
                 stop_returns=True):
        self.music_loaded = music_loaded
        self.sd_loaded = sd_loaded
        self.calls = []
        self.free_returns = stop_returns

        def music_pid():
            return 4242 if self.music_loaded else None

        def music_stop(cancel_check=None):
            self.calls.append("koboldcpp")
            self.music_loaded = False
            return True

        def sd_pid():
            return 4343 if self.sd_loaded else None

        def sd_free(cancel_check=None):
            self.calls.append("sdcpp")
            if self.free_returns:
                self.sd_loaded = False
            return self.free_returns

        monkeypatch.setattr(kobold_server, "running_pid", music_pid)
        monkeypatch.setattr(kobold_server, "stop", music_stop)
        monkeypatch.setattr(sd_shared, "worker_pid", sd_pid)
        monkeypatch.setattr(sd_shared, "free", sd_free)


def _reading(free_gb, *, fresh=True, scope=FREE_SCOPE_DEVICE):
    free = None if free_gb is None else int(free_gb * GB)
    return lambda: (free, fresh, scope)


def _settings(need_gb=12.0, backend="native"):
    return {"backend": backend, "vram_estimate_bytes": int(need_gb * GB)}


def _release(job, label, settings, reading, **kw):
    kw.setdefault("read_free", lambda: 64 * GB)
    return vram.release_other_native_media(job, settings, label,
                                           read_reading=reading, **kw)


# --- music runtime freed before an image / video job --------------------------

@pytest.mark.parametrize("label", ["image", "video"])
def test_loaded_music_server_is_stopped_before_an_image_or_video_job(monkeypatch, label):
    rt = _Runtimes(monkeypatch, music_loaded=True)
    job = _Job()
    out = _release(job, label, _settings(12), _reading(4))
    assert rt.calls == ["koboldcpp"]
    assert out == ["koboldcpp"]
    assert len(job.lines) == 1
    assert "native music runtime" in job.lines[0]
    assert "12.0 GB" in job.lines[0] and "4.0 GB is free" in job.lines[0]


def test_nothing_is_stopped_when_free_vram_suffices(monkeypatch):
    rt = _Runtimes(monkeypatch, music_loaded=True, sd_loaded=True)
    job = _Job()
    assert _release(job, "image", _settings(12), _reading(40)) == []
    assert rt.calls == []
    assert job.lines == []


def test_headroom_counts_like_the_chat_swap_gate(monkeypatch):
    rt = _Runtimes(monkeypatch, music_loaded=True)
    job = _Job()
    just_short = 12.0 + vram._DEFAULT_HEADROOM / GB - 0.1
    _release(job, "image", _settings(12), _reading(just_short))
    assert rt.calls == ["koboldcpp"]


# --- the reverse direction -----------------------------------------------------

def test_loaded_sdcpp_worker_is_stopped_before_a_music_job(monkeypatch):
    rt = _Runtimes(monkeypatch, sd_loaded=True)
    job = _Job()
    out = _release(job, "music", _settings(10), _reading(3))
    assert rt.calls == ["sdcpp"]
    assert out == ["sdcpp"]
    assert "native image/video worker" in job.lines[0]


def test_reverse_direction_stops_nothing_when_vram_suffices(monkeypatch):
    rt = _Runtimes(monkeypatch, sd_loaded=True)
    assert _release(_Job(), "music", _settings(10), _reading(30)) == []
    assert rt.calls == []


# --- never the job's own runtime ------------------------------------------------

@pytest.mark.parametrize("label,loaded", [
    ("image", {"sd_loaded": True}),
    ("video", {"sd_loaded": True}),
    ("music", {"music_loaded": True}),
])
def test_a_job_does_not_stop_its_own_runtime(monkeypatch, label, loaded):
    rt = _Runtimes(monkeypatch, **loaded)
    job = _Job()
    assert _release(job, label, _settings(12), _reading(1)) == []
    assert rt.calls == []
    assert job.lines == []


def test_only_the_other_runtime_is_stopped_when_both_are_loaded(monkeypatch):
    rt = _Runtimes(monkeypatch, music_loaded=True, sd_loaded=True)
    _release(_Job(), "image", _settings(12), _reading(1))
    assert rt.calls == ["koboldcpp"]


# --- readings that cannot see another process -----------------------------------

@pytest.mark.parametrize("reading", [
    _reading(None),
    _reading(64, fresh=False),
    _reading(64, scope=FREE_SCOPE_PROCESS),
], ids=["unreadable", "stale", "process-scoped"])
def test_a_blind_reading_counts_as_short_when_a_peer_runtime_is_loaded(monkeypatch, reading):
    rt = _Runtimes(monkeypatch, music_loaded=True)
    job = _Job()
    assert _release(job, "image", _settings(12), reading) == ["koboldcpp"]
    assert rt.calls == ["koboldcpp"]
    assert "could not be read" in job.lines[0]


def test_a_reading_that_raises_counts_as_blind(monkeypatch):
    rt = _Runtimes(monkeypatch, music_loaded=True)

    def boom(cancel_check=None):
        raise RuntimeError("driver hung")

    assert _release(_Job(), "image", _settings(12), boom) == ["koboldcpp"]
    assert rt.calls == ["koboldcpp"]


# --- jobs this does not apply to --------------------------------------------------

def test_comfy_backend_and_missing_estimate_are_left_alone(monkeypatch):
    rt = _Runtimes(monkeypatch, music_loaded=True, sd_loaded=True)
    assert _release(_Job(), "image", _settings(12, backend="comfy"), _reading(1)) == []
    assert _release(_Job(), "image", {"backend": "native"}, _reading(1)) == []
    assert _release(_Job(), "image",
                    {"backend": "native", "vram_estimate_bytes": 0}, _reading(1)) == []
    assert rt.calls == []


# --- failures are reported, not swallowed -----------------------------------------

def test_a_runtime_that_kept_its_models_is_reported(monkeypatch):
    rt = _Runtimes(monkeypatch, sd_loaded=True, stop_returns=False)
    job = _Job()
    assert _release(job, "music", _settings(12), _reading(1)) == []
    assert rt.calls == ["sdcpp"]
    assert any("kept its models in VRAM" in line for line in job.lines)


def test_a_stop_that_raises_is_reported_and_does_not_abort(monkeypatch):
    _Runtimes(monkeypatch, sd_loaded=True)

    def boom(cancel_check=None):
        raise OSError("access denied")

    monkeypatch.setattr(sd_shared, "free", boom)
    job = _Job()
    assert _release(job, "music", _settings(12), _reading(1)) == []
    assert any("Could not stop the native image/video worker" in line
               and "access denied" in line for line in job.lines)


def test_free_vram_is_polled_after_a_stop(monkeypatch):
    _Runtimes(monkeypatch, music_loaded=True)
    polls = []
    monkeypatch.setattr(vram, "wait_for_vram_release",
                        lambda read, **kw: polls.append(kw) or (True, 0))
    _release(_Job(), "image", _settings(12), _reading(4))
    assert [p["before_bytes"] for p in polls] == [4 * GB]


# --- loaded_native_media_runtimes (what this instance tells its peers) ------------

def test_loaded_runtimes_are_named_for_the_peer_status(monkeypatch):
    _Runtimes(monkeypatch, music_loaded=True, sd_loaded=True)
    assert vram.loaded_native_media_runtimes() == [
        "native image/video worker", "native music runtime"]
    _Runtimes(monkeypatch)
    assert vram.loaded_native_media_runtimes() == []


# --- naming another localm instance --------------------------------------------------

def _peer(port, *, model=None, media=(), gpu_index=0):
    return {"port": port, "model": model, "media": list(media), "gpu_index": gpu_index}


def _note(job, peers, reading, *, label="image", settings=None, calls=None):
    def list_peers():
        if calls is not None:
            calls.append(1)
        return peers

    return vram.note_other_vram_holders(
        job, settings or _settings(12), label, read_reading=reading,
        list_peers=list_peers, gpu_index=0)


def test_a_peer_holding_vram_is_named_in_the_log():
    job = _Job()
    named = _note(job, [_peer(8765, model="qwen-8b", media=["native music runtime"])],
                  _reading(3))
    assert [p["port"] for p in named] == [8765]
    assert len(job.lines) == 1
    line = job.lines[0]
    assert "port 8765" in line and "qwen-8b" in line
    assert "native music runtime" in line
    assert "12.0 GB" in line and "3.0 GB is free" in line


def test_an_idle_media_runtime_alone_makes_a_peer_worth_naming():
    job = _Job()
    _note(job, [_peer(8766, media=["native image/video worker"])], _reading(3))
    assert "port 8766" in job.lines[0]


def test_the_peer_scan_is_skipped_when_free_vram_suffices():
    calls = []
    job = _Job()
    assert _note(job, [_peer(8765, model="m")], _reading(40), calls=calls) == []
    assert calls == [] and job.lines == []


@pytest.mark.parametrize("reading", [
    _reading(None), _reading(3, fresh=False), _reading(3, scope=FREE_SCOPE_PROCESS),
], ids=["unreadable", "stale", "process-scoped"])
def test_no_peer_is_blamed_on_a_reading_that_cannot_be_trusted(reading):
    calls = []
    job = _Job()
    assert _note(job, [_peer(8765, model="m")], reading, calls=calls) == []
    assert calls == [] and job.lines == []


def test_peers_holding_nothing_or_on_another_gpu_are_not_named():
    job = _Job()
    peers = [_peer(8765), _peer(8766, model="m", gpu_index=1)]
    assert _note(job, peers, _reading(3)) == []
    assert job.lines == []


def test_comfy_job_and_peer_scan_failure_are_silent():
    job = _Job()
    assert _note(job, [_peer(8765, model="m")], _reading(3),
                 settings=_settings(12, backend="comfy")) == []

    def boom():
        raise OSError("socket table unreadable")

    assert vram.note_other_vram_holders(
        job, _settings(12), "image", read_reading=_reading(3),
        list_peers=boom, gpu_index=0) == []
    assert job.lines == []


# --- wiring: each plugin runs the release before the chat-swap gate ----------------

def _drive_plug(monkeypatch, plug, route_name, request_obj):
    """Run the plugin's real job body with a recorded call order."""
    order = []
    s = {"reload_after": False, "warning": "", "api_url": "http://127.0.0.1:8188",
         "backend": "comfy", "vram_estimate_bytes": 12 * GB}
    monkeypatch.setattr(plug._backend, "settings", lambda cfg: s)
    monkeypatch.setattr(plug._backend, "ensure_available",
                        lambda s, on_progress=None: (True, "up"))

    def fake_generate(*a, **k):
        order.append("generate")
        return (False, "no")

    monkeypatch.setattr(plug._backend, "generate", fake_generate)
    monkeypatch.setattr("localm.vram.release_other_native_media",
                        lambda job, s, label: order.append(("release", label)))
    monkeypatch.setattr("localm.vram.decide_media_swap",
                        lambda s: order.append("decide") or False)
    monkeypatch.setattr("localm.vram.media_single_device_shortfall", lambda s: None)
    monkeypatch.setattr("localm.vram.note_other_vram_holders",
                        lambda job, s, label: order.append(("note", label)))
    captured = {}

    class _FakeJobs:
        def start_fn(self, kind, fn, result_path=None, owner=None):
            captured["fn"] = fn
            return MagicMock(id="job1")

    request = MagicMock()
    request.app.state.jobs = _FakeJobs()
    request.app.state.self_url = "http://127.0.0.1:8642/v1"
    request.headers = {}
    request.cookies = {}
    asyncio.run(getattr(plug, route_name)(request_obj, request))
    captured["fn"](_Job())
    return order


def test_image_job_releases_before_the_gate_and_notes_before_generating(monkeypatch):
    from localm.plugins.builtin.image import plug
    order = _drive_plug(monkeypatch, plug, "imagine", plug.ImagineRequest(prompt="a cat"))
    assert order == [("release", "image"), "decide", ("note", "image"), "generate"]


def test_music_job_releases_before_the_gate_and_notes_before_generating(monkeypatch):
    from localm.plugins.builtin.music import plug
    order = _drive_plug(monkeypatch, plug, "music", plug.MusicRequest(tags="lofi"))
    assert order == [("release", "music"), "decide", ("note", "music"), "generate"]


def test_video_job_releases_before_the_gate_and_notes_before_generating(monkeypatch):
    from localm.plugins.builtin.video import plug
    order = _drive_plug(monkeypatch, plug, "video", plug.VideoRequest(prompt="a cat"))
    assert order == [("release", "video"), "decide", ("note", "video"), "generate"]
