# SPDX-License-Identifier: AGPL-3.0-or-later
# ruff: noqa: F811
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
from tests._routing_fit import pin_vram_reading, reset_routing_state
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
    pin_vram_reading(monkeypatch, routing_fit.VramReading(free=1, total=_BIG, usable=True))
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
    reset_routing_state()
    monkeypatch.setattr(hs, "_active_model_name", None)
    monkeypatch.setattr(hs, "_last_active_model_name", None)
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
    pin_vram_reading(monkeypatch, routing_fit.VramReading(free=1, total=_BIG, usable=True))
    from localm.inference.backends.llamacpp import _sizing
    monkeypatch.setattr(_sizing, "ram_reading", lambda: (_BIG, _BIG))
    monkeypatch.setattr(hs, "_routing_residents", lambda exclude="": {
        "old-route": routing_fit.Resident("old-route", hold=500, user_chosen=False)})
    monkeypatch.setattr(routing_fit, "classify",
                        lambda engine, free, *a, **k: ((cr.FIT_GPU, 0) if (free or 0) >= 100
                                                       else (cr.FIT_CPU_ONLY, 0)))
    fit = hs._routed_load_fit("seer", object())
    assert fit.cls == cr.FIT_CPU_ONLY and fit.needs_ask


def test_a_routed_switch_that_finds_the_users_model_loaded_does_not_claim_it(
        vision_server, monkeypatch):
    """A routed load that waited while the user's own load of the same model
    finished reuses it and leaves it the user's."""
    import asyncio
    client, engines, _ = vision_server
    _fix_classes(monkeypatch, cr.FIT_GPU)
    named = client.post("/v1/chat/completions", json={
        "model": "seer", "messages": [{"role": "user", "content": "hi"}], "stream": False})
    assert named.status_code == 200 and "seer" not in hs._routing_loaded
    result = asyncio.run(hs.switch_engine("seer", None, preempt=False, activate=False,
                                          routed=True))
    assert result["status"] == "already_active"
    assert "seer" not in hs._routing_loaded


def test_planning_reuses_a_recent_reading_until_a_load_ends(monkeypatch):
    readings = []

    def measure():
        readings.append(1)
        return routing_fit.VramReading(free=len(readings), total=10, usable=True)
    pin_vram_reading(monkeypatch, measure)
    first = hs.planning_vram_reading()
    assert hs.planning_vram_reading() is first and len(readings) == 1
    hs._forget_routing_reading()
    assert hs.planning_vram_reading().free == 2
    monkeypatch.setattr(hs, "ROUTING_READING_MAX_AGE_S", -1.0)
    assert hs.planning_vram_reading().free == 3
    hs._forget_routing_reading()


def test_a_routed_request_never_reloads_a_model_by_unloading_the_users(
        vision_server, monkeypatch):
    """The user's vision model sits partly on the CPU because the user's
    current model holds the VRAM. A routed image request is answered by it as
    loaded: no reload that would unload the current model."""
    client, engines, _ = vision_server
    _fix_classes(monkeypatch, cr.FIT_GPU)
    monkeypatch.setattr(hs, "_switch_probe_vram", _cannot_measure)
    _seer_resident(engines)
    seer = engines["seer"]
    assert seer.loaded and engines["plain"].loaded
    assert "seer" not in hs._routing_loaded
    seer.gpu_placement = {"degraded": True}
    seer.placement_heal = hs.PlacementHeal(frozenset({"plain"}), False)
    assert hs._placement_heal_due("seer", seer)
    switches = []
    real = hs.switch_engine

    async def spy(name, *a, **k):
        switches.append(name)
        return await real(name, *a, **k)
    monkeypatch.setattr(hs, "switch_engine", spy)
    r = _post_image(client)
    assert r.status_code == 200, r.text
    assert _answering_model(engines) == ["seer"]
    assert switches == []
    assert engines["plain"].loaded


def test_a_protected_blocker_never_makes_a_reload_due():
    class _Eng:
        loaded = True
        active_requests = 0
        unloading = False
        gpu_placement = {"degraded": True}
        placement_heal = hs.PlacementHeal(frozenset({"mine"}), False)
    hs._engines["mine"] = _Eng()
    try:
        assert hs._placement_heal_due("x", _Eng())
        assert not hs._placement_heal_due("x", _Eng(), frozenset({"mine"}))
    finally:
        hs._engines.pop("mine", None)


def test_an_explicit_switch_that_waited_on_a_routed_load_makes_the_model_the_users(
        vision_server, monkeypatch):
    """A routed load of the model holds its semaphore and marks it routing
    loaded while the user's explicit switch to it waits: once the switch runs
    the model is the user's."""
    import asyncio

    from localm.inference.inference_gate import InferenceGate, exclusively
    client, engines, _ = vision_server
    _fix_classes(monkeypatch, cr.FIT_GPU)

    async def scenario():
        sem = hs._inference_sems.setdefault("seer", InferenceGate())
        async with exclusively(sem):
            switch = asyncio.create_task(
                hs.switch_engine("seer", None, preempt=True, activate=False))
            for _ in range(5):
                await asyncio.sleep(0)
            hs._engine_factory("seer").load()
            hs._engines["seer"] = engines["seer"]
            hs._engines_lru.append("seer")
            hs._routing_loaded.add("seer")
        return await switch
    result = asyncio.run(scenario())
    assert result["status"] in ("already_active", "loaded"), result
    assert "seer" not in hs._routing_loaded


_COUNT_IMAGE = {"model": "localm", "messages": [{"role": "user", "content": [
    {"type": "text", "text": "what is this?"},
    {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                 "data": "iVBORw0KGgo="}}]}]}


def _counted_by(engines, monkeypatch):
    counted = []
    for name in ("plain", "seer"):
        eng = hs._engine_factory(name)
        monkeypatch.setattr(eng, "count_messages_tokens",
                            lambda messages, _n=name: counted.append(_n) or 5)
    return counted


def _seer_resident(engines):
    hs._engine_factory("seer").load()
    hs._engines["seer"] = engines["seer"]
    hs._engines_lru.insert(0, "seer")


def test_count_tokens_keeps_the_configured_autoswitch_mode(vision_server, monkeypatch):
    """With routing off, a resident vision model does not count an image
    request's tokens in place of the user's model."""
    client, engines, _ = vision_server
    _fix_classes(monkeypatch, cr.FIT_GPU)
    _seer_resident(engines)
    counted = _counted_by(engines, monkeypatch)
    monkeypatch.setattr(cr, "configured_mode", lambda: "off")
    r = client.post("/v1/messages/count_tokens", json=_COUNT_IMAGE)
    assert r.status_code == 200, r.text
    assert counted == ["plain"]


def test_count_tokens_routes_to_a_resident_vision_model(vision_server, monkeypatch):
    client, engines, _ = vision_server
    _fix_classes(monkeypatch, cr.FIT_GPU)
    _seer_resident(engines)
    counted = _counted_by(engines, monkeypatch)
    r = client.post("/v1/messages/count_tokens", json=_COUNT_IMAGE)
    assert r.status_code == 200, r.text
    assert counted == ["seer"]


def test_count_tokens_loads_nothing_when_the_routed_model_left(vision_server, monkeypatch):
    """Planned on a resident vision model that is unloaded before the count:
    the user's model counts, and the vision model is not loaded again."""
    client, engines, _ = vision_server
    _fix_classes(monkeypatch, cr.FIT_GPU)
    _seer_resident(engines)
    counted = _counted_by(engines, monkeypatch)
    real = hs.plan_capability_route

    def plan_then_unload(*a, **k):
        decision = real(*a, **k)
        engines["seer"].unload()
        return decision
    monkeypatch.setattr(hs, "plan_capability_route", plan_then_unload)
    r = client.post("/v1/messages/count_tokens", json=_COUNT_IMAGE)
    assert r.status_code == 200, r.text
    assert counted == ["plain"]
    assert not engines["seer"].loaded


def test_a_headless_job_names_the_model_it_did_not_load_and_why(job_world, monkeypatch):
    from localm.plugins.builtin.jobs import runner
    _fix_classes(monkeypatch, cr.FIT_DISK)
    notes = []
    assert runner._headless_routed_model(job_world, notes=notes) == "plain"
    reason = cr.Fit(cr.FIT_DISK, disk_read_bytes=7 << 30).ask_reason()
    assert notes == [f"tooly could answer this job (plain lacks tool_use); "
                     f"it was not loaded: {reason}"]


def test_a_served_job_refused_at_load_time_runs_on_its_model_and_says_why(
        job_world, monkeypatch):
    import asyncio
    import threading
    from localm.plugins.builtin.jobs import runner
    _fix_classes(monkeypatch, cr.FIT_GPU)
    late = cr.Fit(cr.FIT_DISK, disk_read_bytes=9 << 30)
    calls = []

    async def fake_get_engine(name, **kw):
        calls.append(name)
        if name == "tooly":
            raise hs.RouteRefused(name, late)
        return f"engine-{name}"
    monkeypatch.setattr(hs, "get_engine", fake_get_engine)
    warned = []
    monkeypatch.setattr(runner.logger, "warning", lambda *a, **k: warned.append(a))
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    monkeypatch.setattr(hs, "_server_loop", loop)
    notes = []
    try:
        assert runner._served_engine(job_world, "live-engine", notes=notes) == "live-engine"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(5)
        loop.close()
    assert calls == ["tooly"]
    assert notes == [f"tooly was not loaded: {late.ask_reason()}"]
    assert warned == []


def test_a_job_record_carries_the_routing_note(job_world, monkeypatch):
    from localm.plugins.builtin.jobs import runner
    monkeypatch.setattr(runner, "_creator_refusal", lambda job: None)
    monkeypatch.setattr(runner, "_served_engine",
                        lambda job, live, notes=None: notes.append("why not") or "eng")
    monkeypatch.setattr(runner, "_run_chat", lambda job, engine=None: "answer")
    record = runner.run_job(job_world)
    assert record["status"] == "ok", record
    assert record["routing"] == "why not"


def test_eviction_considers_models_routing_loaded_first(monkeypatch):
    monkeypatch.setattr(hs, "_engines_lru", ["mine-old", "routed", "mine-new"])
    monkeypatch.setattr(hs, "_routing_loaded", {"routed"})
    assert hs._eviction_order() == ["routed", "mine-old", "mine-new"]


def test_the_eviction_loop_picks_a_routed_model_before_the_users(monkeypatch):
    """Every resident is idle and VRAM is short: the idle model routing
    loaded is the first victim, even though the user's model was used less
    recently."""
    import asyncio
    from types import SimpleNamespace
    engines = {n: SimpleNamespace(loaded=True, active_requests=0, unloading=False)
               for n in ("mine", "routed")}
    monkeypatch.setattr(hs, "_engines", engines)
    monkeypatch.setattr(hs, "_engines_lru", ["mine", "routed"])
    monkeypatch.setattr(hs, "_routing_loaded", {"routed"})
    seen = []
    real_decide = switch_admission.decide_admission
    short = switch_admission.VramProbe(free=0, probe_ok=True, process_scoped=False,
                                       shortfall=None, shares_adaptive=False)

    def decide(probe, budget, lru, live):
        seen.append(real_decide(short, budget, lru, live).victim)
        return switch_admission.AdmissionDecision(switch_admission.ADMIT, True, False)

    async def probe(loop, budget, reading=None):
        return None
    monkeypatch.setattr(hs, "_switch_probe_vram", probe)
    monkeypatch.setattr(switch_admission, "decide_admission", decide)
    budget = switch_admission.LoadBudget(
        name="new", vram_required=100, headroom=0, resident_cap=None, pinned=frozenset(),
        check_split_fit=False)

    async def run():
        return await hs._switch_make_room(asyncio.get_running_loop(), budget,
                                          preempt=False, force=False, activate=False)
    assert asyncio.run(run()) is None
    assert seen == ["routed"]


class _Placed:
    def __init__(self, placement):
        self.placement = placement

    def predict_placement(self, free, total, ram_available, ram_total):
        return self.placement


@pytest.mark.parametrize("placement,expected", [
    ({"layers": 10, "full": False, "n_cpu_moe": 0, "reads_from_disk": False}, cr.FIT_UNSIZED),
    ({"layers": 40, "full": True, "n_cpu_moe": 20, "reads_from_disk": False}, cr.FIT_UNSIZED),
    ({"layers": 40, "full": True, "n_cpu_moe": 0, "reads_from_disk": False}, cr.FIT_GPU),
])
def test_without_a_ram_reading_only_a_whole_gpu_placement_is_sized(placement, expected):
    cls, _ = routing_fit.classify(_Placed(placement), 8 << 30, _reading(free=8 << 30),
                                  ram_available=None, ram_total=None)
    assert cls == expected


class TestTwoGpusWithoutAPrediction:
    def _engine(self, tmp_path, monkeypatch, size):
        f = tmp_path / "m.bin"
        f.write_bytes(b"\0" * 16)
        monkeypatch.setattr(routing_fit.residency, "model_footprint_bytes", lambda p: size)

        class E:
            model_path = str(f)
        return E()

    def test_a_model_that_fits_the_free_vram_of_both_whole_is_gpu(self, tmp_path, monkeypatch):
        eng = self._engine(tmp_path, monkeypatch, 4 << 30)
        cls, _ = routing_fit.classify(eng, 24 << 30, _reading(free=24 << 30, devices=2),
                                      ram_available=20 << 30, ram_total=32 << 30)
        assert cls == cr.FIT_GPU

    def test_any_other_placement_is_unsized(self, tmp_path, monkeypatch):
        eng = self._engine(tmp_path, monkeypatch, 30 << 30)
        cls, _ = routing_fit.classify(eng, 24 << 30, _reading(free=24 << 30, devices=2),
                                      ram_available=20 << 30, ram_total=32 << 30)
        assert cls == cr.FIT_UNSIZED


def test_the_image_refusal_text_names_the_suggested_model_first(monkeypatch):
    """Two vision models are installed and routing suggests the second: the
    refusal text tells the user to run the suggested one."""
    from localm.model_manager import registry
    monkeypatch.setattr(registry, "vision_capable_models", lambda: ["big-35b", "small-8b"])
    text = registry.vision_input_guidance(suggested="small-8b")
    assert "`localm run small-8b`" in text
    assert text.index("small-8b") < text.index("big-35b")
    assert "`localm run big-35b`" in registry.vision_input_guidance()


def test_the_audio_refusal_text_names_the_suggested_model_first(monkeypatch):
    from localm.model_manager import registry
    monkeypatch.setattr(registry, "audio_capable_models", lambda: ["big-asr", "small-asr"])
    assert "`localm run small-asr" in registry.audio_input_guidance(suggested="small-asr")
    assert "`localm run big-asr" in registry.audio_input_guidance()


def test_once_the_blocker_is_gone_a_routed_request_reloads_and_unloads_nothing_else(
        vision_server, monkeypatch):
    """A box that cannot measure free VRAM: the user's explicit switch to the
    vision model unloads the other model to make room. A routed image request
    then reloads the vision model for a better placement (its blocker is gone);
    the reload is the only switch, and no other model is unloaded or loaded."""
    import asyncio
    client, engines, _ = vision_server
    _fix_classes(monkeypatch, cr.FIT_GPU)

    monkeypatch.setattr(hs, "_switch_probe_vram", _cannot_measure)
    asyncio.run(hs.switch_engine("seer", None, preempt=True, activate=False))
    assert not engines["plain"].loaded and list(hs._engines) == ["seer"]
    seer = engines["seer"]
    seer.gpu_placement = {"degraded": True}
    seer.placement_heal = hs.PlacementHeal(frozenset({"plain"}), False)
    switches = []
    real = hs.switch_engine

    async def spy(name, *a, **k):
        switches.append((name, k.get("routed"), k.get("preempt")))
        return await real(name, *a, **k)
    monkeypatch.setattr(hs, "switch_engine", spy)
    r = _post_image(client)
    assert r.status_code == 200, r.text
    assert switches == [("seer", True, False)]
    assert list(hs._engines) == ["seer"] and not engines["plain"].loaded
    assert "seer" not in hs._routing_loaded


async def _cannot_measure(loop, budget, reading=None):
    """A VRAM probe on a box that reports no free figure (a CI runner)."""
    return switch_admission.VramProbe(free=None, probe_ok=True, process_scoped=False,
                                      shortfall=None, shares_adaptive=False)


def test_reset_routing_state_drops_every_kind_of_routing_state(monkeypatch):
    from tests._routing_fit import reset_routing_state
    monkeypatch.setattr(hs, "_routing_reading_cache", (0.0, object()))
    hs._routing_loaded.add("x")
    hs._loads_in_flight["x"] = 1
    hs._routing_latch.record_failure("x", "no VRAM")
    reset_routing_state()
    assert hs._routing_reading_cache is None
    assert not hs._routing_loaded and not hs._loads_in_flight
    assert hs._routing_latch.failure("x") is None


# --------------------------------------------------------------------------- #
#  Why a candidate would run on the CPU only                                  #
# --------------------------------------------------------------------------- #

class TestCpuOnlyReason:
    """A real GGUF backend, sized by ``assess``: with n_gpu_layers set to 0
    the reason names that setting; with room for no layer under auto sizing
    it says no layer fits."""

    def _fit(self, tmp_path, free, **kw):
        from tests.test_moe_auto_placement import _backend, _dense_model, _Vram
        backend = _backend(_dense_model(tmp_path), **kw)
        with _Vram(free):
            return routing_fit.assess(
                ["v"], residents={}, build=lambda n: backend,
                reading=_reading(free=free), ram_available=_BIG, ram_total=_BIG,
                unload_policy="always")["v"]

    def test_n_gpu_layers_zero_names_the_setting(self, tmp_path):
        fit = self._fit(tmp_path, 1 << 40, n_gpu_layers=0)
        assert fit.cls == cr.FIT_CPU_ONLY and fit.configured_layers == 0
        assert fit.ask_reason() == ("it would run on the CPU only, because the GPU "
                                    "layers setting (n_gpu_layers) is 0")

    def test_n_gpu_layers_zero_with_auto_off_names_the_setting(self, tmp_path):
        fit = self._fit(tmp_path, 1 << 40, n_gpu_layers=0, n_gpu_layers_auto=False)
        assert fit.configured_layers == 0 and "n_gpu_layers" in fit.ask_reason()

    def test_no_room_under_auto_says_no_layer_fits(self, tmp_path):
        fit = self._fit(tmp_path, 1)
        assert fit.cls == cr.FIT_CPU_ONLY and fit.configured_layers is None
        assert fit.ask_reason() == ("it would run on the CPU only, because no layer "
                                    "fits on the GPU")

    def test_a_backend_that_does_not_report_its_setting_says_no_layer_fits(self):
        fit = _assess(["v"], need=_BIG)["v"]
        assert fit.cls == cr.FIT_CPU_ONLY and fit.configured_layers is None
        assert "no layer fits" in fit.ask_reason()
