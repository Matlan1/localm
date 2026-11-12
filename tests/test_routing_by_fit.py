# SPDX-License-Identifier: AGPL-3.0-or-later
"""Capability routing chooses by how a candidate would run on this machine
(its fit class) and asks before any move that would read weights from disk,
run on the CPU only beside a GPU, could not be sized, or unload a model the
user chose. The rule holds in the ranking, at the HTTP route, under the load
lease (a fresh measurement before the native load) and in which models a
routed load may unload."""

from __future__ import annotations

import itertools
import json

import pytest

import localm.inference.http_server as hs
from localm.inference import capability_routing as cr
from localm.inference import routing_fit
from localm.inference import switch_admission
from tests.test_capability_routing import (_IMAGE_MSG, _answering_model,  # noqa: F401
                                           vision_server)

_USABLE = (cr.FIT_GPU, cr.FIT_EXPERTS_RAM, cr.FIT_PARTIAL)
_ASKED = (cr.FIT_CPU_ONLY, cr.FIT_UNSIZED, cr.FIT_DISK)
_BIG = 1 << 50


# --------------------------------------------------------------------------- #
#  Ranking: exhaustive over fits of three candidates                          #
# --------------------------------------------------------------------------- #

_ALL_FITS = [cr.Fit(c, evicts_user=e) for c in sorted(cr.FIT_NAMES) for e in ((), ("mine",))]


def _decide(fits_by_name, resident=()):
    reg = {n: {"path": f"Z:/{n}.gguf", "model_type": "llm", "tool_use": True}
           for n in ("current", *fits_by_name)}
    reg["current"]["tool_use"] = False
    return cr.plan_route("current", cr.CapabilityNeeds(capabilities=("tool_use",)),
                         pinned=False, fit=lambda names: {n: fits_by_name[n] for n in names},
                         resident=resident, reg=reg, mode="auto")


def test_the_ranking_never_prefers_a_move_that_needs_asking():
    """Every combination of fit (7 classes, with or without unloading the
    user's model) for three candidates: a candidate usable without asking is
    always chosen over one that needs asking, the chosen one is the best
    usable class, and no candidate that needs asking is ever in
    ``candidates``."""
    checked = 0
    for a, b, c in itertools.product(_ALL_FITS, repeat=3):
        fits = {"a": a, "b": b, "c": c}
        d = _decide(fits)
        usable = {n: f for n, f in fits.items() if not f.needs_ask}
        assert all(not fits[n].needs_ask for n in d.candidates)
        if usable:
            assert d.routed and d.resolved in usable
            assert fits[d.resolved].cls == min(f.cls for f in usable.values())
            assert d.suggested is None
        else:
            assert not d.routed and d.resolved == "current"
            assert d.candidates == ()
            assert d.suggested in fits
            assert d.ask_reason == fits[d.suggested].ask_reason() != ""
        checked += 1
    assert checked == len(_ALL_FITS) ** 3


def test_a_fitting_candidate_beats_a_disk_reading_one_with_a_larger_window():
    """The maintainer's case: the bigger window no longer wins."""
    reg = {"current": {"path": "Z:/c.gguf", "model_type": "llm", "tool_use": False},
           "huge": {"path": "Z:/h.gguf", "model_type": "llm", "tool_use": True,
                    "context_length": 262144},
           "small": {"path": "Z:/s.gguf", "model_type": "llm", "tool_use": True,
                     "context_length": 8192}}
    fits = {"huge": cr.Fit(cr.FIT_DISK, disk_read_bytes=22 << 30),
            "small": cr.Fit(cr.FIT_GPU)}
    d = cr.plan_route("current", cr.CapabilityNeeds(capabilities=("tool_use",)),
                      pinned=False, fit=lambda names: {n: fits[n] for n in names},
                      reg=reg, mode="auto")
    assert d.resolved == "small" and d.candidates == ("small",)


def test_a_whole_gpu_fit_beats_a_partial_one_with_a_larger_window():
    reg = {"current": {"path": "Z:/c.gguf", "model_type": "llm", "tool_use": False},
           "big": {"path": "Z:/b.gguf", "model_type": "llm", "tool_use": True,
                   "context_length": 262144},
           "small": {"path": "Z:/s.gguf", "model_type": "llm", "tool_use": True,
                     "context_length": 8192}}
    fits = {"big": cr.Fit(cr.FIT_PARTIAL), "small": cr.Fit(cr.FIT_GPU)}
    d = cr.plan_route("current", cr.CapabilityNeeds(capabilities=("tool_use",)),
                      pinned=False, fit=lambda names: {n: fits[n] for n in names},
                      reg=reg, mode="auto")
    assert d.resolved == "small" and d.candidates == ("small", "big")


def test_only_a_disk_reading_candidate_is_suggested_with_the_reason():
    d = _decide({"a": cr.Fit(cr.FIT_DISK, disk_read_bytes=22 << 30)})
    assert not d.routed and d.suggested == "a"
    assert "22.0 GB" in d.ask_reason and "disk" in d.ask_reason
    assert "would answer if you switch" in d.describe() and d.ask_reason in d.describe()


def test_a_candidate_that_cannot_be_prepared_is_a_load_error_never_a_suggestion():
    d = _decide({"a": cr.Fit(cr.FIT_UNSIZED, error="ValueError: gone")})
    assert d.suggested is None and not d.routed
    assert d.load_errors == ("a: ValueError: gone",)


def test_every_caller_must_pass_a_fit():
    with pytest.raises(TypeError):
        cr.plan_route("current", cr.CapabilityNeeds(capabilities=("tool_use",)),  # type: ignore[call-arg]
                      pinned=False, reg={})


# --------------------------------------------------------------------------- #
#  routing_fit.assess                                                         #
# --------------------------------------------------------------------------- #

class _Sized:
    """An engine that fits the GPU only with at least *need* bytes free."""

    def __init__(self, need, model_path=None):
        self.need = need
        self.model_path = model_path

    def predict_placement(self, free, total, ram_available, ram_total):
        if free >= self.need:
            return {"layers": 99, "full": True, "n_cpu_moe": 0, "reads_from_disk": False,
                    "cpu_read_bytes": 0}
        return {"layers": 0, "full": False, "n_cpu_moe": 0, "reads_from_disk": False,
                "cpu_read_bytes": self.need}


def _reading(free=10, devices=1, usable=True, present=True):
    return routing_fit.VramReading(free=free, total=1000, usable=usable,
                                   devices=devices, gpu_present=present)


def _assess(names, residents=(), *, need=100, reading=None, policy="always"):
    return routing_fit.assess(
        names, residents={r.name: r for r in residents}, build=lambda n: _Sized(need),
        reading=reading or _reading(), ram_available=_BIG, ram_total=_BIG,
        unload_policy=policy)


def test_a_candidate_that_fits_beside_needs_no_unload():
    assert _assess(["v"], need=5)["v"] == cr.Fit(cr.FIT_GPU)


def test_unloading_a_model_routing_loaded_is_not_asked():
    routed = routing_fit.Resident("old-route", hold=200, user_chosen=False)
    fit = _assess(["v"], [routed])["v"]
    assert fit.cls == cr.FIT_GPU and fit.evicts_user == () and not fit.needs_ask


def test_unloading_the_users_model_is_asked():
    mine = routing_fit.Resident("mine", hold=200, user_chosen=True)
    fit = _assess(["v"], [mine])["v"]
    assert fit.cls == cr.FIT_GPU and fit.evicts_user == ("mine",)
    assert fit.ask_reason() == "loading it unloads mine"


def test_an_unknown_hold_does_not_count_as_freed_vram():
    mine = routing_fit.Resident("mine", hold=0, user_chosen=True)
    assert _assess(["v"], [mine])["v"].cls == cr.FIT_CPU_ONLY


def test_a_pinned_model_is_never_counted_as_freed():
    mine = routing_fit.Resident("mine", hold=200, user_chosen=True, evictable=False)
    assert _assess(["v"], [mine])["v"].cls == cr.FIT_CPU_ONLY


@pytest.mark.parametrize("policy", ["never", "ask"])
def test_unload_for_room_never_or_ask_unloads_nothing_for_a_route(policy):
    routed = routing_fit.Resident("old-route", hold=200, user_chosen=False)
    assert _assess(["v"], [routed], policy=policy)["v"].cls == cr.FIT_CPU_ONLY


def test_on_two_gpus_the_backend_sizes_itself_and_no_unload_is_counted():
    asked = []

    class TwoGpu(_Sized):
        def predict_placement(self, free, total, ram_available, ram_total):
            asked.append(free)
            return None if free is None else super().predict_placement(
                free, total, ram_available, ram_total)
    routed = routing_fit.Resident("old-route", hold=200, user_chosen=False)
    fit = routing_fit.assess(["v"], residents={"old-route": routed},
                             build=lambda n: TwoGpu(100), reading=_reading(devices=2),
                             ram_available=_BIG, ram_total=_BIG, unload_policy="always")["v"]
    assert asked == [None], "only the backend's own split reading is used, once"
    assert fit.evicts_user == () and fit.cls == cr.FIT_UNSIZED


def test_an_unusable_reading_is_unsized():
    assert _assess(["v"], reading=_reading(usable=False))["v"].cls == cr.FIT_UNSIZED


def test_a_resident_and_a_peer_load_nothing():
    here = routing_fit.Resident("v", hold=1, user_chosen=True)
    assert _assess(["v"], [here])["v"].cls == cr.FIT_RESIDENT
    fits = routing_fit.assess(["p"], residents={}, build=lambda n: _Sized(1),
                              reading=_reading(), ram_available=1, ram_total=1,
                              unload_policy="always", peers=["p"])
    assert fits["p"].cls == cr.FIT_RESIDENT


def test_a_resident_that_reads_from_disk_is_disk():
    here = routing_fit.Resident("v", hold=1, user_chosen=True, reads_from_disk=True)
    assert _assess(["v"], [here])["v"].cls == cr.FIT_DISK


def test_a_model_that_cannot_be_built_carries_its_error():
    def build(name):
        raise ValueError("Model files not found")
    fit = routing_fit.assess(["v"], residents={}, build=build, reading=_reading(),
                             ram_available=1, ram_total=1, unload_policy="always")["v"]
    assert fit.error == "ValueError: Model files not found"


class TestClassifyWithoutAPrediction:
    """A backend that cannot predict its placement is judged by its size."""

    def _engine(self, tmp_path, size, needs=None):
        f = tmp_path / "m.bin"
        f.write_bytes(b"\0" * 16)

        class E:
            model_path = str(f)

            def gpu_offload_needs(self):
                return needs
        return E(), size

    def test_size_beyond_vram_and_ram_reads_from_disk(self, tmp_path, monkeypatch):
        eng, _ = self._engine(tmp_path, 0)
        monkeypatch.setattr(routing_fit.residency, "model_footprint_bytes", lambda p: 40 << 30)
        cls, disk = routing_fit.classify(eng, 16 << 30, _reading(free=16 << 30),
                                         ram_available=20 << 30, ram_total=32 << 30)
        assert cls == cr.FIT_DISK and disk == 24 << 30

    def test_offload_needs_within_free_is_gpu(self, tmp_path, monkeypatch):
        eng, _ = self._engine(tmp_path, 0, needs=(4 << 30, 4 << 30))
        monkeypatch.setattr(routing_fit.residency, "model_footprint_bytes", lambda p: 3 << 30)
        cls, _ = routing_fit.classify(eng, 8 << 30, _reading(free=8 << 30),
                                      ram_available=20 << 30, ram_total=32 << 30)
        assert cls == cr.FIT_GPU

    def test_unknown_size_is_unsized(self, tmp_path, monkeypatch):
        eng, _ = self._engine(tmp_path, 0)
        monkeypatch.setattr(routing_fit.residency, "model_footprint_bytes", lambda p: 0)
        cls, _ = routing_fit.classify(eng, 8 << 30, _reading(free=8 << 30),
                                      ram_available=20 << 30, ram_total=32 << 30)
        assert cls == cr.FIT_UNSIZED


# --------------------------------------------------------------------------- #
#  The chat route: one vision candidate in every fit class                    #
# --------------------------------------------------------------------------- #

def _fix_classes(monkeypatch, plan_cls, *, plain_hold=0, need=None):
    """Plan-time measurement: the vision model is *plan_cls*; with *need*,
    it reaches FIT_GPU only once that many bytes are free (the user's plain
    model holds *plain_hold*). The load-time re-check agrees with the plan."""
    monkeypatch.setattr(hs, "routing_vram_reading", lambda: routing_fit.VramReading(
        free=1, total=_BIG, usable=True))
    from localm.inference.backends.llamacpp import _sizing
    monkeypatch.setattr(_sizing, "ram_reading", lambda: (_BIG, _BIG))

    def classify(engine, free, *a, **k):
        if need is not None:
            return (cr.FIT_GPU, 0) if (free or 0) >= need else (cr.FIT_CPU_ONLY, 0)
        return plan_cls, (7 << 30 if plan_cls == cr.FIT_DISK else 0)
    monkeypatch.setattr(routing_fit, "classify", classify)
    if plain_hold:
        hs._engines["plain"].vram_load_bytes = plain_hold


def _post_image(client):
    return client.post("/v1/chat/completions", json={"messages": _IMAGE_MSG, "stream": False})


@pytest.mark.parametrize("cls", _USABLE)
def test_a_usable_vision_model_answers_beside_the_users_model(vision_server, monkeypatch, cls):
    client, engines, _ = vision_server
    _fix_classes(monkeypatch, cls)
    r = _post_image(client)
    assert r.status_code == 200, r.text
    assert _answering_model(engines) == ["seer"]
    assert engines["seer"].loaded and engines["plain"].loaded
    assert "seer" in hs._routing_loaded
    blob = json.loads(r.headers["X-Localm-Model-Routing"])
    assert blob["routed"] is True and blob["fit"] == cr.FIT_NAMES[cls]


@pytest.mark.parametrize("cls", _ASKED)
def test_a_vision_model_that_needs_asking_is_offered_not_loaded(vision_server, monkeypatch, cls):
    client, engines, _ = vision_server
    _fix_classes(monkeypatch, cls)
    r = _post_image(client)
    assert r.status_code == 400
    assert r.headers[hs.MEDIA_REFUSAL_HEADER] == "image"
    blob = json.loads(r.headers["X-Localm-Model-Routing"])
    assert blob["suggested"] == "seer" and blob["routed"] is False
    assert blob["ask_reason"] == cr.Fit(cls, disk_read_bytes=7 << 30 if cls == cr.FIT_DISK
                                        else 0).ask_reason()
    assert blob["fit"] == cr.FIT_NAMES[cls]
    assert not getattr(engines.get("seer"), "loaded", False)
    assert engines["plain"].loaded and _answering_model(engines) == []


def test_a_vision_model_that_needs_the_users_model_unloaded_is_offered(
        vision_server, monkeypatch):
    client, engines, _ = vision_server
    _fix_classes(monkeypatch, None, plain_hold=500, need=100)
    r = _post_image(client)
    assert r.status_code == 400
    blob = json.loads(r.headers["X-Localm-Model-Routing"])
    assert blob["suggested"] == "seer"
    assert blob["ask_reason"] == "loading it unloads plain"
    assert engines["plain"].loaded
    assert not getattr(engines.get("seer"), "loaded", False)


def test_a_resident_vision_model_answers_without_a_load(vision_server, monkeypatch):
    client, engines, _ = vision_server
    _fix_classes(monkeypatch, cr.FIT_DISK)
    hs._engine_factory("seer").load()
    hs._engines["seer"] = engines["seer"]
    hs._engines_lru.insert(0, "seer")
    r = _post_image(client)
    assert r.status_code == 200
    assert _answering_model(engines) == ["seer"]


def test_the_load_time_measurement_can_refuse_a_planned_route(vision_server, monkeypatch):
    """Planned as fitting, measured again under the lease as reading from
    disk: nothing loads, and the request gets the offer with that reason."""
    client, engines, _ = vision_server
    _fix_classes(monkeypatch, cr.FIT_GPU)
    late = cr.Fit(cr.FIT_DISK, disk_read_bytes=9 << 30)
    monkeypatch.setattr(hs, "_routed_load_fit", lambda name, engine: late)
    r = _post_image(client)
    assert r.status_code == 400
    blob = json.loads(r.headers["X-Localm-Model-Routing"])
    assert blob["suggested"] == "seer" and blob["ask_reason"] == late.ask_reason()
    assert blob["fit"] == "disk", "the header carries the fit measured at load time"
    assert not getattr(engines.get("seer"), "loaded", False)
    assert "seer" not in hs._routing_loaded


def test_a_routed_load_may_unload_only_models_routing_loaded(vision_server, monkeypatch):
    """The eviction ladder gets every model routing did not load as
    never-evict, so it cannot pick the user's model whatever VRAM says."""
    client, engines, _ = vision_server
    _fix_classes(monkeypatch, cr.FIT_GPU)
    seen = []
    real = hs._switch_make_room

    async def capture(loop, budget, **kw):
        seen.append(budget)
        return await real(loop, budget, **kw)
    monkeypatch.setattr(hs, "_switch_make_room", capture)
    monkeypatch.setattr(hs, "_switch_load_budget", lambda name: switch_admission.LoadBudget(
        name=name, vram_required=1, headroom=0, resident_cap=None, pinned=frozenset(),
        check_split_fit=False))
    r = _post_image(client)
    assert r.status_code == 200, r.text
    assert seen and "plain" in seen[0].pinned


class TestTheRoutingLoadedSet:
    def test_a_request_that_names_the_model_makes_it_the_users(self, vision_server, monkeypatch):
        client, engines, _ = vision_server
        _fix_classes(monkeypatch, cr.FIT_GPU)
        assert _post_image(client).status_code == 200
        assert "seer" in hs._routing_loaded
        r = client.post("/v1/chat/completions",
                        json={"model": "seer", "messages": [{"role": "user", "content": "hi"}],
                              "stream": False})
        assert r.status_code == 200
        assert "seer" not in hs._routing_loaded

    def test_an_unloaded_routed_model_leaves_the_set(self, vision_server, monkeypatch):
        client, engines, _ = vision_server
        _fix_classes(monkeypatch, cr.FIT_GPU)
        assert _post_image(client).status_code == 200
        hs._switch_detach_victim("seer", hs._engines["seer"], activate=False)
        assert "seer" not in hs._routing_loaded


def test_count_tokens_never_loads_a_routed_model(vision_server, monkeypatch):
    client, engines, _ = vision_server
    _fix_classes(monkeypatch, cr.FIT_GPU)
    r = client.post("/v1/messages/count_tokens", json={
        "model": "localm",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": "iVBORw0KGgo="}}]}]})
    assert r.status_code == 200, r.text
    assert not getattr(engines.get("seer"), "loaded", False)


def test_an_ollama_chat_inherits_the_offer(vision_server, monkeypatch):
    client, engines, _ = vision_server
    _fix_classes(monkeypatch, cr.FIT_DISK)
    r = client.post("/api/chat", json={
        "model": "localm", "stream": False,
        "messages": [{"role": "user", "content": "what is this?", "images": ["iVBORw0KGgo="]}]})
    assert r.status_code == 400, r.text
    blob = json.loads(r.headers["X-Localm-Model-Routing"])
    assert blob["suggested"] == "seer"
    assert not getattr(engines.get("seer"), "loaded", False)


# --------------------------------------------------------------------------- #
#  Scheduled jobs                                                             #
# --------------------------------------------------------------------------- #

@pytest.fixture
def job_world(monkeypatch, tmp_path):
    from types import SimpleNamespace
    registry = {
        "plain": {"path": str(tmp_path / "plain.gguf"), "model_type": "llm", "tool_use": False},
        "tooly": {"path": str(tmp_path / "tooly.gguf"), "model_type": "llm", "tool_use": True},
    }
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.model_manager.load_registry", lambda: registry)
    monkeypatch.setattr("localm.plugins.builtin.jobs.webtool.web_enabled", lambda: True)
    monkeypatch.setattr("localm.plugins.builtin.jobs.runner._default_chat_model", lambda: "plain")
    monkeypatch.setattr(cr, "configured_mode", lambda: "auto")
    monkeypatch.setattr(hs, "_engine_factory",
                        lambda name: SimpleNamespace(model_path=registry[name]["path"]))
    hs._engines.clear()
    hs._routing_loaded.clear()
    from localm.plugins.builtin.jobs.store import Job
    return Job(name="nightly", prompt="summarise")


@pytest.mark.parametrize("cls,expected", [(cr.FIT_GPU, "tooly"), (cr.FIT_DISK, "plain"),
                                          (cr.FIT_UNSIZED, "plain")])
def test_a_headless_job_routes_only_to_a_model_that_fits(job_world, monkeypatch, cls, expected):
    from localm.plugins.builtin.jobs import runner
    _fix_classes(monkeypatch, cls)
    assert runner._headless_routed_model(job_world) == expected


def test_a_served_job_loads_its_routed_model_as_a_routed_load(job_world, monkeypatch):
    import asyncio
    import threading
    from localm.plugins.builtin.jobs import runner
    _fix_classes(monkeypatch, cr.FIT_GPU)
    calls = []

    async def fake_get_engine(name, **kw):
        calls.append((name, kw.get("routed")))
        return f"engine-{name}"
    monkeypatch.setattr(hs, "get_engine", fake_get_engine)
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    monkeypatch.setattr(hs, "_server_loop", loop)
    try:
        assert runner._served_engine(job_world, None) == "engine-tooly"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(5)
        loop.close()
    assert calls == [("tooly", True)]


def test_a_routed_request_for_a_model_the_user_loaded_does_not_claim_it(vision_server, monkeypatch):
    client, engines, _ = vision_server
    _fix_classes(monkeypatch, cr.FIT_GPU)
    named = client.post("/v1/chat/completions", json={
        "model": "seer", "messages": [{"role": "user", "content": "hi"}], "stream": False})
    assert named.status_code == 200 and engines["seer"].loaded
    hs._active_model_name = "plain"
    r = client.post("/v1/chat/completions", json={
        "model": "plain", "pin_model": False, "messages": _IMAGE_MSG, "stream": False})
    assert r.status_code == 200
    assert engines["seer"].answered == 2
    assert "seer" not in hs._routing_loaded, "the user's own model stays the user's"


def test_a_routed_load_reports_which_model_and_why(vision_server, monkeypatch):
    client, engines, _ = vision_server
    _fix_classes(monkeypatch, cr.FIT_GPU)
    built = []
    real = hs.loading_routed_status
    monkeypatch.setattr(hs, "loading_routed_status",
                        lambda *a: built.append(a) or real(*a))
    assert _post_image(client).status_code == 200
    assert built == [("seer", "plain", ["vision"])]
    text = real("seer", "plain", ["vision"])
    assert text == "Loading 'seer' for this request: 'plain' lacks image input..."
    assert text.code == "loading_routed"


def test_a_context_overflow_carries_the_routing_header(vision_server, monkeypatch):
    client, engines, registry = vision_server
    _fix_classes(monkeypatch, cr.FIT_GPU)
    registry["plain"]["context_length"] = 4096
    engines["plain"].count_messages_tokens = lambda messages: 10 ** 6
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "stream": False,
        "min_context": 100000, "model_autoswitch": "ask"})
    assert r.status_code == 413, r.text
    assert "X-Localm-Model-Routing" in r.headers


def test_the_load_time_check_counts_no_unload_that_did_not_happen(monkeypatch):
    """An idle model routing loaded would make room if unloaded, but room-making
    left it resident: the load-time check judges the state as it is."""
    monkeypatch.setattr(hs, "routing_vram_reading", lambda: routing_fit.VramReading(
        free=1, total=_BIG, usable=True))
    from localm.inference.backends.llamacpp import _sizing
    monkeypatch.setattr(_sizing, "ram_reading", lambda: (_BIG, _BIG))
    monkeypatch.setattr(hs, "_routing_residents", lambda exclude="": {
        "old-route": routing_fit.Resident("old-route", hold=500, user_chosen=False)})
    monkeypatch.setattr(routing_fit, "classify",
                        lambda engine, free, *a, **k: ((cr.FIT_GPU, 0) if (free or 0) >= 100
                                                       else (cr.FIT_CPU_ONLY, 0)))
    fit = hs._routed_load_fit("seer", object())
    assert fit.cls == cr.FIT_CPU_ONLY and fit.needs_ask
