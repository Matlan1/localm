# SPDX-License-Identifier: AGPL-3.0-or-later
"""Capability routing does not load a model again right after its load failed.

The latch key is (model name, load fingerprint). A record stops applying when
the fingerprint changes (model file, load-related config key, provisioned
runtime), when the model loads, or when its backoff elapses. A load the user
asked for by name never consults it, and a cancelled load is not a failure.

The end-to-end tests drive the real /v1/chat/completions route with an engine
whose load raises the text a native crash produces, and assert on how many times
that load was attempted, read from the engine itself.
"""

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import localm.inference.http_server as hs
from localm.inference import capability_routing as cr
from localm.inference import routing_latch as rl
from localm.inference.backends.base import ModelLoadCancelled

CRASH_TEXT = ("The native model-loading process crashed (exit code -11 "
              "(killed by signal SIGSEGV)) while loading the model file.")


class _Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def _latch(fp="fp-1"):
    clock = _Clock()
    box = {"fp": fp}
    latch = rl.RoutingLatch(clock=clock, fingerprint=lambda name: box["fp"])
    return latch, clock, box


# --------------------------------------------------------------------------- #
#  RoutingLatch on its own                                                     #
# --------------------------------------------------------------------------- #

class TestRoutingLatch:
    def test_a_recorded_failure_is_skipped_until_its_backoff_elapses(self):
        latch, clock, _ = _latch()
        latch.record_failure("big", CRASH_TEXT)
        assert list(latch.skipped()) == ["big"]
        clock.now += rl.BACKOFF_BASE_S - 1
        assert list(latch.skipped()) == ["big"]
        clock.now += 2
        assert latch.skipped() == {}

    def test_the_skip_carries_when_it_failed_why_and_when_it_is_retried(self):
        latch, clock, _ = _latch()
        latch.record_failure("big", CRASH_TEXT)
        skipped = latch.skipped()["big"]
        assert skipped.failed_at == 1_000_000.0
        assert skipped.retry_at == 1_000_000.0 + rl.BACKOFF_BASE_S
        assert "SIGSEGV" in skipped.reason

    def test_each_consecutive_failure_doubles_the_delay_up_to_the_cap(self):
        latch, clock, _ = _latch()
        delays = []
        for _ in range(12):
            rec = latch.record_failure("big", "boom")
            delays.append(rec.retry_at - rec.failed_at)
        assert delays[:3] == [rl.BACKOFF_BASE_S, rl.BACKOFF_BASE_S * 2,
                              rl.BACKOFF_BASE_S * 4]
        assert delays[-1] == rl.BACKOFF_MAX_S
        assert max(delays) == rl.BACKOFF_MAX_S

    def test_a_failed_retry_keeps_counting_after_the_backoff_elapsed(self):
        latch, clock, _ = _latch()
        latch.record_failure("big", "boom")
        clock.now += rl.BACKOFF_BASE_S + 1
        assert latch.skipped() == {}
        second = latch.record_failure("big", "boom")
        assert second.attempts == 2
        assert second.retry_at - second.failed_at == rl.BACKOFF_BASE_S * 2

    def test_a_changed_fingerprint_stops_the_skip_and_restarts_the_count(self):
        latch, clock, box = _latch()
        latch.record_failure("big", "boom")
        latch.record_failure("big", "boom")
        box["fp"] = "fp-2"
        assert latch.skipped() == {}
        assert latch.failure("big") is None
        assert latch.record_failure("big", "boom").attempts == 1

    def test_a_failure_under_a_new_fingerprint_does_not_inherit_the_old_count(self):
        latch, clock, box = _latch()
        latch.record_failure("big", "boom")
        latch.record_failure("big", "boom")
        rec = latch.record_failure("big", "boom", fingerprint="fp-other")
        assert rec.attempts == 1

    def test_a_successful_load_clears_the_record(self):
        latch, clock, _ = _latch()
        latch.record_failure("big", "boom")
        latch.record_success("big")
        assert latch.skipped() == {}
        assert latch.failure("big") is None

    def test_one_models_success_leaves_another_models_record_alone(self):
        latch, clock, _ = _latch()
        latch.record_failure("a", "boom")
        latch.record_failure("b", "boom")
        latch.record_success("a")
        assert list(latch.skipped()) == ["b"]

    def test_with_nothing_recorded_no_fingerprint_is_computed(self):
        calls = []
        latch = rl.RoutingLatch(fingerprint=lambda n: calls.append(n) or "x")
        assert latch.skipped() == {}
        assert calls == []

    def test_the_reason_is_one_bounded_line(self):
        latch, clock, _ = _latch()
        rec = latch.record_failure("big", "line one\n\n   line two " + "x" * 500)
        assert "\n" not in rec.reason
        assert len(rec.reason) <= rl.REASON_MAX_CHARS
        assert rec.reason.startswith("line one line two")
        assert rec.reason.endswith("...")


# --------------------------------------------------------------------------- #
#  load_fingerprint: what counts as "something relevant changed"               #
# --------------------------------------------------------------------------- #

class TestLoadFingerprint:
    @pytest.fixture
    def world(self, monkeypatch, tmp_path):
        model = tmp_path / "m.gguf"
        model.write_bytes(b"GGUF" + b"\0" * 64)
        w = SimpleNamespace(model=model, mmproj=None,
                            cfg={"n_ctx": 4096, "n_gpu_layers": 99,
                                 "max_tokens": 512, "llama_runtime_history": []})
        monkeypatch.setattr("localm.config.load_config", lambda: dict(w.cfg))
        monkeypatch.setattr("localm.model_manager.get_model_info",
                            lambda name: (str(w.model), "hint"))
        monkeypatch.setattr("localm.model_manager.get_model_mmproj",
                            lambda name: w.mmproj)
        return w

    def test_it_is_stable_when_nothing_changed(self, world):
        assert rl.load_fingerprint("m") == rl.load_fingerprint("m")

    @pytest.mark.parametrize("key,value", [
        ("n_ctx", 8192), ("n_gpu_layers", 20), ("n_cpu_moe", 4),
        ("mtp_enabled", True), ("gpu_split_ratios", [1, 2]),
        ("main_gpu_index", 1), ("binary_dir", "x"), ("llama_runtime_pin", "b1"),
        ("vram_overhead_mb", 3000), ("ctx_auto", False),
    ])
    def test_a_load_setting_changes_it(self, world, key, value):
        before = rl.load_fingerprint("m")
        world.cfg[key] = value
        assert rl.load_fingerprint("m") != before

    def test_a_setting_that_does_not_affect_loading_does_not_change_it(self, world):
        before = rl.load_fingerprint("m")
        world.cfg["max_tokens"] = 99
        world.cfg["temperature"] = 0.1
        assert rl.load_fingerprint("m") == before

    def test_a_new_runtime_provision_changes_it(self, world):
        before = rl.load_fingerprint("m")
        world.cfg["llama_runtime_history"] = [
            {"backend": "cuda", "tag": "b11118", "at": 1.0}]
        assert rl.load_fingerprint("m") != before

    def test_a_replaced_model_file_changes_it(self, world):
        before = rl.load_fingerprint("m")
        world.model.write_bytes(b"GGUF" + b"\0" * 4096)
        assert rl.load_fingerprint("m") != before

    def test_a_model_file_at_a_different_path_changes_it(self, world, tmp_path):
        before = rl.load_fingerprint("m")
        other = tmp_path / "other.gguf"
        other.write_bytes(world.model.read_bytes())
        world.model = other
        assert rl.load_fingerprint("m") != before

    def test_a_projector_appearing_changes_it(self, world, tmp_path):
        before = rl.load_fingerprint("m")
        proj = tmp_path / "mmproj.gguf"
        proj.write_bytes(b"GGUF")
        world.mmproj = str(proj)
        assert rl.load_fingerprint("m") != before

    def test_an_unreadable_model_file_still_yields_a_digest(self, world):
        world.model = world.model.parent / "gone.gguf"
        assert len(rl.load_fingerprint("m")) == 16

    def test_a_reader_that_raises_still_yields_a_digest(self, world, monkeypatch):
        def malformed(*args, **kwargs):
            raise AttributeError("malformed registry entry")

        monkeypatch.setattr("localm.model_manager.get_model_info", malformed)
        monkeypatch.setattr("localm.config.load_config", malformed)
        assert len(rl.load_fingerprint("m")) == 16


# --------------------------------------------------------------------------- #
#  plan_route with a skip set                                                  #
# --------------------------------------------------------------------------- #

def _reg(**entries):
    out = {}
    for name, spec in entries.items():
        entry = {"path": f"Z:/models/{name}.gguf", "source": "local",
                 "model_type": "llm"}
        entry.update(spec)
        out[name] = entry
    return out


REG = _reg(
    plain={"tool_use": False, "context_length": 8192},
    big={"tool_use": True, "context_length": 65536},
    small={"tool_use": True, "context_length": 32768},
)
TOOLS = cr.CapabilityNeeds(capabilities=("tool_use",))
SKIP_BIG = {"big": cr.SkippedCandidate("big", 1_000_000.0, 1_000_600.0, "boom")}


class TestPlanRouteSkip:
    def test_a_skipped_model_is_not_a_candidate(self):
        d = cr.plan_route("plain", TOOLS, pinned=False, reg=REG, skip=SKIP_BIG)
        assert d.candidates == ("small",)
        assert d.resolved == "small"

    def test_a_skipped_model_that_would_have_qualified_is_reported(self):
        d = cr.plan_route("plain", TOOLS, pinned=False, reg=REG, skip=SKIP_BIG)
        assert [s.model for s in d.skipped] == ["big"]

    def test_when_every_capable_model_is_skipped_the_current_model_answers(self):
        skip = dict(SKIP_BIG)
        skip["small"] = cr.SkippedCandidate("small", 1.0, 2.0, "boom")
        d = cr.plan_route("plain", TOOLS, pinned=False, reg=REG, skip=skip)
        assert d.candidates == ()
        assert d.resolved == "plain" and d.routed is False
        assert d.unmet == ("tool_use",)
        assert [s.model for s in d.skipped] == ["big", "small"]

    def test_a_skipped_model_that_could_not_serve_the_request_is_not_reported(self):
        skip = {"plain": cr.SkippedCandidate("plain", 1.0, 2.0, "boom")}
        d = cr.plan_route("tooly", TOOLS, pinned=False,
                          reg=_reg(plain={"tool_use": False}, tooly={"tool_use": True}),
                          skip=skip)
        assert d.skipped == ()

    def test_a_pinned_request_ignores_the_skip_set(self):
        d = cr.plan_route("plain", TOOLS, pinned=True, reg=REG, skip=SKIP_BIG)
        assert d.resolved == "plain" and d.skipped == ()

    def test_without_a_skip_set_nothing_changes(self):
        d = cr.plan_route("plain", TOOLS, pinned=False, reg=REG)
        assert d.candidates == ("big", "small") and d.skipped == ()

    def test_the_partial_path_skips_too(self):
        """No installed model has both tool calls and the window the prompt
        needs, so the roomy models are the candidates; the roomiest is skipped."""
        reg = _reg(plain={"tool_use": False, "context_length": 4096},
                   toolsmall={"tool_use": True, "context_length": 4096},
                   roomy={"tool_use": False, "context_length": 200000},
                   roomy2={"tool_use": False, "context_length": 131072})
        needs = cr.CapabilityNeeds(capabilities=("tool_use",), min_context=100000)
        skip = {"roomy": cr.SkippedCandidate("roomy", 1.0, 2.0, "boom")}
        d = cr.plan_route("plain", needs, pinned=False, reg=reg, skip=skip)
        assert d.candidates == ("roomy2",)
        assert d.unmet == ("tool_use",)
        assert [s.model for s in d.skipped] == ["roomy"]

    def test_the_partial_path_without_a_skip_prefers_the_roomiest(self):
        reg = _reg(plain={"tool_use": False, "context_length": 4096},
                   toolsmall={"tool_use": True, "context_length": 4096},
                   roomy={"tool_use": False, "context_length": 200000},
                   roomy2={"tool_use": False, "context_length": 131072})
        needs = cr.CapabilityNeeds(capabilities=("tool_use",), min_context=100000)
        d = cr.plan_route("plain", needs, pinned=False, reg=reg)
        assert d.candidates == ("roomy", "roomy2") and d.skipped == ()


class TestDescribeSkipped:
    def test_it_names_the_model_when_it_failed_why_and_when_it_is_retried(self):
        skip = {n: cr.SkippedCandidate(n, 1_000_000.0, 1_000_600.0, "it crashed")
                for n in ("big", "small")}
        d = cr.plan_route("plain", TOOLS, pinned=False, reg=REG, skip=skip)
        text = d.describe()
        assert "big was skipped because its last load failed at" in text
        assert "(it crashed)" in text
        assert cr._clock_text(1_000_000.0) in text
        assert f"tried again after {cr._clock_text(1_000_600.0)}" in text
        assert "kept plain (tool_use=absent)" in text
        assert "no installed model provides" not in text

    def test_a_route_that_skipped_a_model_still_says_so(self):
        d = cr.plan_route("plain", TOOLS, pinned=False, reg=REG, skip=SKIP_BIG)
        text = d.describe()
        assert text.startswith("routed plain -> small")
        assert "big was skipped" in text

    def test_a_decision_with_nothing_skipped_reads_as_before(self):
        d = cr.plan_route("plain", TOOLS, pinned=False, reg=REG)
        assert d.describe() == "routed plain -> big (tool_use=absent)"


# --------------------------------------------------------------------------- #
#  End to end over HTTP                                                        #
# --------------------------------------------------------------------------- #

class _Engine:
    def __init__(self, name, world):
        self.display_name = name
        self.loaded = False
        self.supports_images = False
        self.can_be_multimodal = False
        self.last_finish_reason = "stop"
        self.unloading = False
        self.answered = 0
        self.load_attempts = 0
        self._world = world

    def load(self):
        self.load_attempts += 1
        behaviour = self._world.load_behaviour.get(self.display_name)
        if behaviour == "crash":
            raise RuntimeError(CRASH_TEXT)
        if behaviour == "cancel":
            raise ModelLoadCancelled("superseded by a newer selection")
        self.loaded = True

    def unload(self):
        self.loaded = False

    def chat_stream(self, messages, **kw):
        self.answered += 1
        yield f"answered-by-{self.display_name}"

    def count_tokens(self, text):
        return 3

    def count_messages_tokens(self, messages):
        return 5

    def context_capacity(self):
        return 8192


def _build_world(monkeypatch, tmp_path, registry):
    world = SimpleNamespace(
        registry=registry, engines={}, load_behaviour={}, clock=_Clock(),
        cfg={})
    files = {}
    for name in registry:
        f = tmp_path / f"{name}.gguf"
        f.write_bytes(b"GGUF" + b"\0" * 64)
        files[name] = f
        registry[name]["path"] = str(f)
    world.files = files

    def factory(name):
        return world.engines.setdefault(name, _Engine(name, world))

    import localm.config as _cfg
    real_load_config = _cfg.load_config
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {**real_load_config(), **world.cfg})
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.model_manager.load_registry", lambda: registry)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (str(files[name]), "hint"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
    monkeypatch.setattr(hs, "_engine_factory", factory)
    monkeypatch.setattr(hs._routing_latch, "_clock", world.clock)
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._last_activity_per_model.clear()
    hs._active_model_name = None
    hs._engine = None
    hs._inference_sem = None
    world.factory = factory
    return world


def _mount_explicit_load(app):
    """The model-load route the GUI mounts, backed by the same switch_engine
    call the server's launcher hands it."""
    from localm.plugins.gui.routes import models as routes_models

    async def switch_model(name, *, force=False):
        return await hs.switch_engine(name, hs._engine_factory, force=force)

    ctx = SimpleNamespace(active_model=lambda: hs._active_model_name or "",
                          switch_model=switch_model, jobs=None)
    routes_models.register(app, ctx)


@pytest.fixture
def crashing(monkeypatch, tmp_path):
    """plain (no tool calls) is loaded; big (tool calls, the roomiest) crashes
    when loaded. No other capable model is installed."""
    world = _build_world(monkeypatch, tmp_path, _reg(
        plain={"tool_use": False, "context_length": 8192},
        big={"tool_use": True, "context_length": 65536},
    ))
    world.load_behaviour["big"] = "crash"
    startup = world.factory("plain")
    startup.load()
    app = hs.create_app(startup)
    _mount_explicit_load(app)
    shell = {"Authorization": f"Bearer {app.state.shell_token}"}
    with TestClient(app, headers=shell) as client:
        world.client = client
        yield world


@pytest.fixture
def crashing_with_alternative(monkeypatch, tmp_path):
    """As above, plus small: a second capable model that loads fine but ranks
    behind big."""
    world = _build_world(monkeypatch, tmp_path, _reg(
        plain={"tool_use": False, "context_length": 8192},
        big={"tool_use": True, "context_length": 65536},
        small={"tool_use": True, "context_length": 32768},
    ))
    world.load_behaviour["big"] = "crash"
    startup = world.factory("plain")
    startup.load()
    with TestClient(hs.create_app(startup)) as client:
        world.client = client
        yield world


def _ask(world, **body):
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    body.setdefault("stream", False)
    body.setdefault("required_capabilities", ["tool_use"])
    return world.client.post("/v1/chat/completions", json=body)


def _attempts(world, name):
    eng = world.engines.get(name)
    return eng.load_attempts if eng is not None else 0


def _blob(response):
    return json.loads(response.headers["X-Localm-Model-Routing"])


class TestACrashedCandidateIsNotRetried:
    def test_the_second_request_does_not_attempt_the_load(self, crashing):
        r1 = _ask(crashing)
        assert r1.status_code == 200
        assert _attempts(crashing, "big") == 1
        r2 = _ask(crashing)
        assert r2.status_code == 200
        assert _attempts(crashing, "big") == 1, (
            "the model whose load just crashed was loaded again")
        assert crashing.engines["plain"].answered == 2

    def test_the_first_fallback_reports_the_load_error(self, crashing):
        blob = _blob(_ask(crashing))
        assert blob["routed"] is False
        assert "SIGSEGV" in blob["load_errors"][0]

    def test_the_later_fallback_says_the_model_was_skipped_and_why(self, crashing):
        _ask(crashing)
        blob = _blob(_ask(crashing))
        assert blob["routed"] is False
        assert blob["resolved"] == "plain"
        [skipped] = blob["skipped"]
        assert skipped["model"] == "big"
        assert "SIGSEGV" in skipped["reason"]
        assert skipped["retry_at"] > skipped["failed_at"]
        assert "big was skipped because its last load failed" in blob["note"]

    def test_the_debug_log_says_the_model_was_skipped(self, crashing, caplog):
        _ask(crashing)
        with caplog.at_level(logging.INFO, logger="localm"):
            _ask(crashing)
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "big was skipped because its last load failed" in text
        assert "SIGSEGV" in text

    def test_a_request_that_needs_nothing_reports_nothing(self, crashing):
        _ask(crashing)
        r = _ask(crashing, required_capabilities=[])
        assert "X-Localm-Model-Routing" not in r.headers


class TestTheNextCapableModelStillAnswers:
    def test_the_crashed_model_is_bypassed_and_the_alternative_answers(
            self, crashing_with_alternative):
        world = crashing_with_alternative
        _ask(world)
        assert _attempts(world, "big") == 1
        assert world.engines["small"].answered == 1
        r = _ask(world)
        assert _attempts(world, "big") == 1
        assert world.engines["small"].answered == 2
        blob = _blob(r)
        assert blob["routed"] is True and blob["resolved"] == "small"
        assert blob["skipped"][0]["model"] == "big"


class TestALoadTheUserAskedForStillHappens:
    def test_a_request_that_names_the_model_attempts_the_load(self, crashing):
        _ask(crashing)
        r = _ask(crashing, model="big")
        assert _attempts(crashing, "big") == 2
        assert r.status_code == 503
        assert "SIGSEGV" in r.json()["detail"]

    def test_an_explicit_load_attempts_the_load(self, crashing):
        _ask(crashing)
        r = crashing.client.post("/api/models/load", json={"model": "big"})
        assert _attempts(crashing, "big") == 2
        assert r.status_code >= 400

    def test_a_failed_explicit_load_latches_the_model_for_routing(self, crashing):
        crashing.client.post("/api/models/load", json={"model": "big"})
        assert _attempts(crashing, "big") == 1
        _ask(crashing)
        assert _attempts(crashing, "big") == 1
        assert _blob(_ask(crashing))["skipped"][0]["model"] == "big"


class TestWhatClearsTheLatch:
    def test_a_successful_explicit_load_clears_it(self, crashing):
        _ask(crashing)
        crashing.load_behaviour["big"] = None
        r = crashing.client.post("/api/models/load", json={"model": "big"})
        assert r.status_code == 200
        assert hs._routing_latch.failure("big") is None
        r2 = _ask(crashing)
        assert crashing.engines["big"].answered == 1
        assert "X-Localm-Model-Routing" not in r2.headers

    def test_a_changed_load_setting_clears_it(self, crashing):
        _ask(crashing)
        _ask(crashing)
        assert _attempts(crashing, "big") == 1
        crashing.cfg["n_gpu_layers"] = 12
        _ask(crashing)
        assert _attempts(crashing, "big") == 2

    def test_a_replaced_model_file_clears_it(self, crashing):
        _ask(crashing)
        crashing.files["big"].write_bytes(b"GGUF" + b"\0" * 8192)
        _ask(crashing)
        assert _attempts(crashing, "big") == 2

    def test_a_new_runtime_provision_clears_it(self, crashing):
        _ask(crashing)
        crashing.cfg["llama_runtime_history"] = [
            {"backend": "cuda", "tag": "b11118", "at": 5.0}]
        _ask(crashing)
        assert _attempts(crashing, "big") == 2

    def test_the_backoff_elapsing_allows_one_retry_and_a_failed_retry_waits_longer(
            self, crashing):
        _ask(crashing)
        crashing.clock.now += rl.BACKOFF_BASE_S + 1
        _ask(crashing)
        assert _attempts(crashing, "big") == 2
        crashing.clock.now += rl.BACKOFF_BASE_S + 1
        _ask(crashing)
        assert _attempts(crashing, "big") == 2, (
            "the second failure must wait longer than the first")
        crashing.clock.now += rl.BACKOFF_BASE_S * 2
        _ask(crashing)
        assert _attempts(crashing, "big") == 3

    def test_a_fresh_app_starts_with_nothing_latched(self, crashing):
        _ask(crashing)
        assert hs._routing_latch.failure("big") is not None
        hs._init_engine_state(None)
        assert hs._routing_latch.failure("big") is None


class TestACancelledLoadIsNotAFailure:
    def test_a_cancelled_load_is_attempted_again_next_time(self, crashing):
        crashing.load_behaviour["big"] = "cancel"
        _ask(crashing)
        assert hs._routing_latch.failure("big") is None
        _ask(crashing)
        assert _attempts(crashing, "big") == 2


class TestOtherLoadFailuresLatchToo:
    def test_a_load_that_raises_an_oserror_falls_back_and_latches(
            self, monkeypatch, tmp_path):
        world = _build_world(monkeypatch, tmp_path, _reg(
            plain={"tool_use": False, "context_length": 8192},
            big={"tool_use": True, "context_length": 65536},
        ))
        engines = world.engines

        class OsErrorEngine(_Engine):
            def load(self):
                self.load_attempts += 1
                raise OSError("the disk went away")

        monkeypatch.setattr(
            hs, "_engine_factory",
            lambda name: engines.setdefault(
                name, (OsErrorEngine if name == "big" else _Engine)(name, world)))
        startup = hs._engine_factory("plain")
        startup.load()
        body = {"messages": [{"role": "user", "content": "hi"}],
                "required_capabilities": ["tool_use"]}
        with TestClient(hs.create_app(startup)) as client:
            first = client.post("/v1/chat/completions", json=body)
            assert first.status_code == 200
            assert "the disk went away" in _blob(first)["load_errors"][0]
            assert engines["big"].load_attempts == 1
            rec = hs._routing_latch.failure("big")
            assert rec is not None and "the disk went away" in rec.reason
            second = client.post("/v1/chat/completions", json=body)
            assert second.status_code == 200
            assert engines["big"].load_attempts == 1


# --------------------------------------------------------------------------- #
#  The reported scenario: the coder's backend against a real server            #
# --------------------------------------------------------------------------- #

def _wait(cond, timeout=10.0):
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


@pytest.fixture
def live_crashing(monkeypatch, tmp_path):
    """A real uvicorn server with "plain" (no tool calls) loaded and "big"
    (tool calls) installed, whose load crashes."""
    import socket
    import threading

    import uvicorn

    world = _build_world(monkeypatch, tmp_path, _reg(
        plain={"tool_use": False, "context_length": 8192},
        big={"tool_use": True, "context_length": 65536},
    ))
    world.load_behaviour["big"] = "crash"
    startup = world.factory("plain")
    startup.load()
    app = hs.create_app(startup)
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    port = lsock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
    th = threading.Thread(target=lambda: asyncio.run(server.serve(sockets=[lsock])),
                          daemon=True)
    th.start()
    assert _wait(lambda: server.started), "uvicorn did not start"
    try:
        yield f"http://127.0.0.1:{port}/v1", world
    finally:
        server.should_exit = True
        th.join(timeout=5.0)


class TestTheCoderIsNotMadeToWaitOnACrashingLoad:
    def _backend(self, base):
        from localm.plugins.coder.backends.http import HTTPBackend
        be = HTTPBackend(base, model="plain", api_key="localm", localm_server=True,
                         model_pinned=False, required_capabilities=("tool_use",))
        be.notes = []
        be.on_routing_note = be.notes.append
        return be

    def test_every_request_is_answered_and_the_load_is_attempted_once(
            self, live_crashing):
        base, world = live_crashing
        be = self._backend(base)
        for _ in range(3):
            text = be.chat([{"role": "user", "content": "list files"}])
            assert text == "answered-by-plain"
        assert _attempts(world, "big") == 1
        assert world.engines["plain"].answered == 3

    def test_streaming_requests_are_not_retried_either(self, live_crashing):
        base, world = live_crashing
        be = self._backend(base)
        for _ in range(3):
            text = "".join(be.chat_stream([{"role": "user", "content": "x"}]))
            assert text == "answered-by-plain"
        assert _attempts(world, "big") == 1

    def test_the_session_is_told_when_the_load_failed_and_again_when_it_is_skipped(
            self, live_crashing):
        base, world = live_crashing
        be = self._backend(base)
        for _ in range(3):
            be.chat([{"role": "user", "content": "list files"}])
        assert len(be.notes) == 2, be.notes
        assert "SIGSEGV" in be.notes[0]
        assert "no capable model could be loaded" in be.notes[0]
        assert "big was skipped because its last load failed" in be.notes[1]
        assert "SIGSEGV" in be.notes[1]
        assert be.routing_note == be.notes[1]
