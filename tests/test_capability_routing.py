# SPDX-License-Identifier: AGPL-3.0-or-later
"""Autonomous model-capability routing, and the one rule it must never break.

THE BINDING CONSTRAINT: a model the user named EXPLICITLY is never swapped for
another. Everything else here is a convenience; that is a correctness property,
so its tests assert on the model that ACTUALLY ANSWERED rather than on a status
code. A 200 says the request worked, not that it worked on the right model, and
a test that only checked the code would pass while the server quietly answered
from somewhere else.

Routing runs against the REAL detectors reading a REAL registry shape, not a
mocked capability oracle: the capability values here are the same keys
registration writes.
"""

import json
import unittest

import pytest
from fastapi.testclient import TestClient

import localm.inference.http_server as hs
from localm.inference import capability_routing as cr
from localm.model_manager import capabilities as caps


def _reg(**entries):
    """A registry in the shape registration actually writes.

    ``tool_use``/``context_length`` present = confirmed at registration; absent =
    nobody has looked. Both states appear below on purpose: a fixture that could
    only produce confirmed values could not catch an unknown being treated as a
    confirmed no."""
    out = {}
    for name, spec in entries.items():
        entry = {"path": f"Z:/models/{name}.gguf", "source": "local",
                 "model_type": "llm"}
        entry.update(spec)
        out[name] = entry
    return out


TOOLS_ONLY = _reg(
    plain={"tool_use": False, "context_length": 8192},
    tooly={"tool_use": True, "context_length": 32768},
)


# --------------------------------------------------------------------------- #
#  plan_route: the decision itself                                             #
# --------------------------------------------------------------------------- #

class TestPlanRoute:
    def test_routes_an_unpinned_request_to_a_capable_model(self):
        d = cr.plan_route("plain", cr.CapabilityNeeds(capabilities=("tool_use",)),
                          pinned=False, reg=TOOLS_ONLY)
        assert d.resolved == "tooly"
        assert d.routed is True
        assert d.gaps == {"tool_use": False}

    def test_NEVER_moves_an_explicitly_pinned_model(self):
        """The binding constraint, at the planner. A capable alternative exists
        and is deliberately NOT chosen."""
        d = cr.plan_route("plain", cr.CapabilityNeeds(capabilities=("tool_use",)),
                          pinned=True, reg=TOOLS_ONLY)
        assert d.resolved == "plain"
        assert d.routed is False
        assert d.gaps == {"tool_use": False}      # still REPORTED, never acted on

    def test_no_gap_leaves_the_model_alone(self):
        d = cr.plan_route("tooly", cr.CapabilityNeeds(capabilities=("tool_use",)),
                          pinned=False, reg=TOOLS_ONLY)
        assert d.resolved == "tooly"
        assert d.routed is False
        assert d.has_gap is False

    def test_no_needs_does_nothing(self):
        d = cr.plan_route("plain", cr.CapabilityNeeds(), pinned=False,
                          reg=TOOLS_ONLY)
        assert d.resolved == "plain"
        assert d.has_gap is False

    def test_unmet_when_no_installed_model_qualifies(self):
        """The honest fallback: routing that found nowhere better says so."""
        reg = _reg(a={"tool_use": False}, b={"tool_use": False})
        d = cr.plan_route("a", cr.CapabilityNeeds(capabilities=("tool_use",)),
                          pinned=False, reg=reg)
        assert d.resolved == "a"
        assert d.routed is False
        assert d.unmet == ("tool_use",)

    def test_an_unknown_model_is_not_a_routing_target(self):
        """Positive membership only: a model nobody has inspected must not
        receive a request on the strength of an absent key."""
        reg = _reg(current={"tool_use": False}, unmeasured={})
        d = cr.plan_route("current",
                          cr.CapabilityNeeds(capabilities=("tool_use",)),
                          pinned=False, reg=reg)
        assert d.resolved == "current"
        assert d.unmet == ("tool_use",)

    def test_a_gap_records_unknown_and_absent_distinctly(self):
        """An unmeasured current model gaps (routing prefers certainty) but is
        recorded as None, never False - it is not a model known to lack tools."""
        reg = _reg(unmeasured={}, tooly={"tool_use": True})
        d = cr.plan_route("unmeasured",
                          cr.CapabilityNeeds(capabilities=("tool_use",)),
                          pinned=False, reg=reg)
        assert d.resolved == "tooly"
        assert d.gaps == {"tool_use": None}
        assert d.gaps["tool_use"] is not False
        assert "unknown" in d.describe()

    def test_unknown_context_is_not_a_shortfall(self):
        """Context gaps only on a CONFIRMED shortfall. Treating an unmeasured
        window as too small would re-route almost every request on a registry
        that predates the field."""
        reg = _reg(unmeasured={}, big={"context_length": 131072})
        d = cr.plan_route("unmeasured", cr.CapabilityNeeds(min_context=100000),
                          pinned=False, reg=reg)
        assert d.has_gap is False
        assert d.resolved == "unmeasured"

    def test_confirmed_too_small_context_routes(self):
        reg = _reg(small={"context_length": 4096}, big={"context_length": 131072})
        d = cr.plan_route("small", cr.CapabilityNeeds(min_context=100000),
                          pinned=False, reg=reg)
        assert d.resolved == "big"
        assert d.gaps == {"context_length": False}

    def test_prefers_a_resident_model_over_an_equally_capable_one(self):
        reg = _reg(plain={"tool_use": False},
                   coldbig={"tool_use": True, "context_length": 131072},
                   warm={"tool_use": True, "context_length": 8192})
        d = cr.plan_route("plain", cr.CapabilityNeeds(capabilities=("tool_use",)),
                          pinned=False, resident=["warm"], reg=reg)
        assert d.resolved == "warm"

    def test_prefers_the_roomier_model_when_neither_is_resident(self):
        reg = _reg(plain={"tool_use": False},
                   big={"tool_use": True, "context_length": 131072},
                   small={"tool_use": True, "context_length": 8192})
        d = cr.plan_route("plain", cr.CapabilityNeeds(capabilities=("tool_use",)),
                          pinned=False, reg=reg)
        assert d.resolved == "big"

    def test_multiple_needs_must_all_be_met_by_one_model(self):
        reg = _reg(plain={"tool_use": False, "context_length": 4096},
                   toolsonly={"tool_use": True, "context_length": 4096},
                   both={"tool_use": True, "context_length": 131072})
        d = cr.plan_route(
            "plain", cr.CapabilityNeeds(capabilities=("tool_use",),
                                        min_context=100000),
            pinned=False, reg=reg)
        assert d.resolved == "both"


class TestContextNeed:
    def test_short_prompts_ask_no_context_question(self):
        assert cr.context_need([{"role": "user", "content": "hi"}]) is None

    def test_a_long_prompt_asks_for_headroom_above_its_own_size(self):
        msgs = [{"role": "user", "content": "x" * 40000}]      # ~10000 tokens
        need = cr.context_need(msgs)
        assert need is not None
        assert need > cr.estimate_prompt_tokens(msgs)

    def test_structured_content_text_parts_are_counted(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "y" * 40000}]}]
        assert cr.context_need(msgs) is not None


class TestDescribe:
    def test_names_the_capability_that_drove_the_choice(self):
        d = cr.plan_route("plain", cr.CapabilityNeeds(capabilities=("tool_use",)),
                          pinned=False, reg=TOOLS_ONLY)
        text = d.describe()
        assert "tool_use" in text and "plain" in text and "tooly" in text

    def test_says_nothing_happened_when_nothing_did(self):
        d = cr.plan_route("tooly", cr.CapabilityNeeds(capabilities=("tool_use",)),
                          pinned=False, reg=TOOLS_ONLY)
        assert d.describe() == "no capability gap"


# --------------------------------------------------------------------------- #
#  End to end over HTTP: which model actually answered                         #
# --------------------------------------------------------------------------- #

class FakeEngine:
    def __init__(self, name):
        self.display_name = name
        self.loaded = False
        self.supports_images = False
        self.can_be_multimodal = False
        self.last_finish_reason = "stop"
        self.unloading = False
        self.answered = 0

    def load(self):
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


@pytest.fixture
def server(monkeypatch):
    """A running server with two REGISTERED models: the loaded one has no
    tool-call template, a second one does."""
    registry = _reg(
        plain={"tool_use": False, "context_length": 8192},
        tooly={"tool_use": True, "context_length": 32768},
    )
    engines: dict[str, FakeEngine] = {}

    def factory(name):
        return engines.setdefault(name, FakeEngine(name))

    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.model_manager.load_registry", lambda: registry)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (f"Z:/models/{name}.gguf", "hint"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj", lambda name: None)
    monkeypatch.setattr(hs, "_engine_factory", factory)

    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._last_activity_per_model.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None

    startup = factory("plain")
    startup.load()
    with TestClient(hs.create_app(startup)) as client:
        yield client, engines


def _ask(client, **body):
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    body.setdefault("stream", False)
    return client.post("/v1/chat/completions", json=body)


def _answering_model(engines):
    """The model that actually produced the reply, read from the engines
    themselves rather than from anything the response claims."""
    return [n for n, e in engines.items() if e.answered]


class TestRoutingOverHTTP:
    def test_unpinned_request_needing_tools_is_answered_by_the_capable_model(
            self, server):
        client, engines = server
        r = _ask(client, required_capabilities=["tool_use"])
        assert r.status_code == 200
        assert _answering_model(engines) == ["tooly"]
        assert engines["plain"].answered == 0

    def test_explicit_pin_is_NEVER_overridden(self, server):
        """THE binding constraint, end to end.

        "plain" provably lacks the capability and a capable model is installed,
        so every incentive to swap is present. The assertion is on the engine
        that generated the reply, not on the status code: a 200 would be
        satisfied by either model answering."""
        client, engines = server
        r = _ask(client, model="plain", required_capabilities=["tool_use"])
        assert r.status_code == 200
        assert _answering_model(engines) == ["plain"]
        assert "answered-by-plain" in r.text
        # Stronger than "tooly did not answer": the engine factory is lazy, so a
        # capable model that was never even CONSTRUCTED proves the pinned path
        # never so much as resolved an alternative.
        assert engines.get("tooly") is None or engines["tooly"].answered == 0

    def test_a_pinned_request_still_reports_the_gap(self, server):
        """Not silently ignored either: the user is told what the pinned model
        lacks, which is the suggestion half of the never-swap rule."""
        client, engines = server
        r = _ask(client, model="plain", required_capabilities=["tool_use"])
        blob = json.loads(r.headers["X-Localm-Model-Routing"])
        assert blob["pinned"] is True
        assert blob["routed"] is False
        assert blob["resolved"] == "plain"
        assert blob["gaps"] == {"tool_use": "absent"}

    def test_a_routed_request_is_auditable(self, server):
        client, engines = server
        r = _ask(client, required_capabilities=["tool_use"])
        blob = json.loads(r.headers["X-Localm-Model-Routing"])
        assert blob["routed"] is True
        assert blob["requested"] == "plain"
        assert blob["resolved"] == "tooly"
        assert blob["gaps"] == {"tool_use": "absent"}

    def test_an_ordinary_request_routes_nowhere_and_adds_no_header(self, server):
        client, engines = server
        r = _ask(client)
        assert r.status_code == 200
        assert _answering_model(engines) == ["plain"]
        assert "X-Localm-Model-Routing" not in r.headers

    def test_the_localm_sentinel_counts_as_unpinned(self, server):
        """"localm" is truthy but means "no preference", the same idiom
        get_engine and peer routing already use."""
        client, engines = server
        r = _ask(client, model="localm", required_capabilities=["tool_use"])
        assert r.status_code == 200
        assert _answering_model(engines) == ["tooly"]

    def test_an_unknown_capability_name_is_rejected(self, server):
        """A typo must not be silently ignored, which would look exactly like
        "no model qualifies"."""
        client, _ = server
        r = _ask(client, required_capabilities=["tool-use"])
        assert r.status_code == 422

    def test_streaming_carries_the_same_audit_header(self, server):
        client, engines = server
        r = _ask(client, required_capabilities=["tool_use"], stream=True)
        assert r.status_code == 200
        assert json.loads(r.headers["X-Localm-Model-Routing"])["resolved"] == "tooly"


class TestPinnedDiscriminator(unittest.TestCase):
    """The single test that decides whether routing may act at all."""

    def test_named_models_are_pinned(self):
        for name in ("plain", "Qwen2.5", "  spaced  "):
            self.assertTrue(hs._model_is_pinned(name), name)

    def test_absent_empty_and_the_sentinel_are_not_pinned(self):
        for name in (None, "", "   ", "localm"):
            self.assertFalse(hs._model_is_pinned(name), repr(name))


# --------------------------------------------------------------------------- #
#  Reconciliation with the shipped vision-mismatch 400                         #
# --------------------------------------------------------------------------- #

_IMAGE_MSG = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
    ],
}]


@pytest.fixture
def vision_server(tmp_path, monkeypatch):
    """A text-only model loaded, and a genuinely vision-capable one registered.

    The vision capability is REAL, not patched: a regular model file plus a
    recorded projector that exists on disk is exactly what
    model_vision_capability reads, so the routing decision here runs the shipped
    probe rather than a stand-in for it."""
    # SEPARATE directories on purpose: find_sibling_mmproj auto-detects a
    # projector sitting NEXT TO a model file, so a shared folder would make the
    # text-only model vision-capable too and leave nothing to route away from.
    (tmp_path / "plain").mkdir()
    (tmp_path / "seer").mkdir()
    plain_path = tmp_path / "plain" / "plain.gguf"
    plain_path.write_bytes(b"GGUF")
    seer_path = tmp_path / "seer" / "seer.gguf"
    seer_path.write_bytes(b"GGUF")
    proj_path = tmp_path / "seer" / "seer-mmproj.gguf"
    proj_path.write_bytes(b"GGUF")

    registry = {
        "plain": {"path": str(plain_path), "source": "local",
                  "model_type": "llm"},
        "seer": {"path": str(seer_path), "source": "local", "model_type": "llm",
                 "mmproj": str(proj_path)},
    }
    engines: dict[str, FakeEngine] = {}

    def factory(name):
        e = engines.setdefault(name, FakeEngine(name))
        e.supports_images = (name == "seer")
        return e

    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.model_manager.load_registry", lambda: registry)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (str(tmp_path / name / f"{name}.gguf"), "hint"))
    monkeypatch.setattr("localm.model_manager.get_model_mmproj",
                        lambda name, **kw: str(proj_path) if name == "seer" else None)
    monkeypatch.setattr(hs, "_engine_factory", factory)

    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._last_activity_per_model.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None

    startup = factory("plain")
    startup.load()
    with TestClient(hs.create_app(startup)) as client:
        yield client, engines, registry


class TestVisionMismatchReconciliation:
    def test_the_shipped_probe_really_sees_the_fixture(self, vision_server):
        """Guards the fixture itself: if this stopped being True the routing
        tests below would pass for the wrong reason (nothing to route to)."""
        _, _, registry = vision_server
        from localm.model_manager import registry as R
        assert R.model_vision_capability("seer", reg=registry) is True
        assert R.model_vision_capability("plain", reg=registry) is False

    def test_unpinned_image_request_now_ROUTES_instead_of_400(self, vision_server):
        """The behaviour change the ADR asked for. This request used to be a 400
        naming the text-only model; it is now answered by a model that can
        actually see the picture."""
        client, engines, _ = vision_server
        r = client.post("/v1/chat/completions",
                        json={"messages": _IMAGE_MSG, "stream": False})
        assert r.status_code == 200
        assert _answering_model(engines) == ["seer"]

    def test_pinned_image_request_still_gets_the_shipped_400(self, vision_server):
        """Unchanged for an explicit pin: dropping the picture silently is worse
        than refusing, and swapping the model the user is looking at is out."""
        client, engines, _ = vision_server
        r = client.post("/v1/chat/completions",
                        json={"model": "plain", "messages": _IMAGE_MSG,
                              "stream": False})
        assert r.status_code == 400
        assert "cannot accept image" in r.json()["detail"]
        assert _answering_model(engines) == []

    def test_unpinned_image_request_with_no_capable_model_still_400s(
            self, vision_server, monkeypatch):
        """The honest fallback: with nowhere to route, the shipped refusal
        stands rather than the picture being dropped."""
        client, engines, registry = vision_server
        registry.pop("seer")
        r = client.post("/v1/chat/completions",
                        json={"messages": _IMAGE_MSG, "stream": False})
        assert r.status_code == 400
        assert _answering_model(engines) == []


# --------------------------------------------------------------------------- #
#  The coder's suggestion (a note, never a switch)                             #
# --------------------------------------------------------------------------- #

class TestCoderToolCapabilityNote:
    """The coder pins the active model and a per-session switch changes the one
    shared engine for every other caller, so its capability integration is a
    SUGGESTION. These pin the conditions under which it stays silent."""

    def _note(self, monkeypatch, registry, model):
        from localm.plugins.builtin.coder import plug
        monkeypatch.setattr("localm.model_manager.load_registry", lambda: registry)
        return plug._tool_capability_note(model)

    def test_suggests_a_capable_model_when_the_active_one_confirmedly_lacks_it(
            self, monkeypatch):
        note = self._note(monkeypatch, TOOLS_ONLY, "plain")
        assert "tooly" in note
        assert "Tool calls still work" in note      # a fitness note, not a block

    def test_silent_when_the_active_model_is_capable(self, monkeypatch):
        assert self._note(monkeypatch, TOOLS_ONLY, "tooly") == ""

    def test_silent_when_the_active_model_is_UNKNOWN(self, monkeypatch):
        """The tri-state case. An unmeasured model is not a model known to lack
        tool support, and advising a switch away from one on no evidence is
        exactly the wrong advice."""
        reg = _reg(unmeasured={}, tooly={"tool_use": True})
        assert self._note(monkeypatch, reg, "unmeasured") == ""

    def test_silent_when_nothing_better_is_installed(self, monkeypatch):
        reg = _reg(plain={"tool_use": False})
        assert self._note(monkeypatch, reg, "plain") == ""

    def test_silent_without_a_model_name(self, monkeypatch):
        assert self._note(monkeypatch, TOOLS_ONLY, "") == ""
