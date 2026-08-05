# SPDX-License-Identifier: AGPL-3.0-or-later
"""A model that cannot see images must REJECT image input, not silently drop it.

Regression guard for the audit finding that GGUF (and text-only HF) accepted an
image_url part, discarded it, and answered about a picture the model never
received. The contract now: raise UnsupportedInputError at the backend, and
return a clean 400 at the HTTP route.
"""

import importlib.util
import json
import unittest
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localm.inference.backends.base import (
    IMAGE_UNSUPPORTED_MESSAGE,
    UnsupportedInputError,
    messages_contain_image,
)
from localm.inference.http_server import create_app


_TEXT_MSG = [{"role": "user", "content": "describe this"}]
_IMAGE_MSG = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
    ],
}]


# --------------------------------------------------------------------------- #
#  Detection helper                                                            #
# --------------------------------------------------------------------------- #

class TestMessagesContainImage:
    def test_plain_text_string(self):
        assert messages_contain_image(_TEXT_MSG) is False

    def test_text_only_parts(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        assert messages_contain_image(msgs) is False

    def test_image_part_detected(self):
        assert messages_contain_image(_IMAGE_MSG) is True

    def test_ignores_non_dict_parts(self):
        msgs = [{"role": "user", "content": ["loose string", {"type": "text", "text": "x"}]}]
        assert messages_contain_image(msgs) is False


# --------------------------------------------------------------------------- #
#  Backend-level guard (the source-of-truth guarantee, covers every caller)   #
# --------------------------------------------------------------------------- #

class TestGgufRejectsImages:
    def test_chat_stream_raises_on_image(self):
        from localm.inference.backends.gguf import GgufBackend
        backend = GgufBackend("does-not-need-to-exist.gguf")
        with pytest.raises(UnsupportedInputError):
            # Generator: the guard fires on first iteration, before any DLL use.
            next(backend.chat_stream(_IMAGE_MSG))

    def test_gguf_never_supports_images(self):
        from localm.inference.backends.gguf import GgufBackend
        backend = GgufBackend("does-not-need-to-exist.gguf")
        assert backend.supports_images is False
        assert backend.can_be_multimodal is False


class TestHFRejectsImagesWhenTextOnly:
    def test_supports_images_tracks_multimodal_flag(self):
        from localm.inference.backends.hf import HFBackend
        backend = HFBackend("does-not-need-to-exist")
        assert backend.can_be_multimodal is True
        assert backend.supports_images is False        # not loaded / text-only
        # supports_images is cached from the isolated child's load response
        # (see hf.py) rather than a live self._is_multimodal read - the real
        # HFWorker instance now lives in a separate process. Fake a completed
        # load the same way GgufBackend's own tests do: setting _loaded=True
        # with no real self._runner hits the defensive "is_alive is None ->
        # True" branch of the loaded property (mirrors GgufBackend.loaded).
        backend._loaded = True
        backend._supports_images = True
        assert backend.supports_images is True

    def test_text_only_chat_stream_raises_on_image(self):
        from localm.inference.backends.hf import HFBackend
        backend = HFBackend("does-not-need-to-exist")
        # Never loaded (supports_images is therefore False); the backend-level
        # guard fires before self._runner is ever touched, importing nothing
        # heavier than base.py - see HFBackend.chat_stream's ordering.
        with pytest.raises(UnsupportedInputError):
            next(backend.chat_stream(_IMAGE_MSG))


# --------------------------------------------------------------------------- #
#  HTTP route returns a clean 400 (the good-UX layer)                          #
# --------------------------------------------------------------------------- #

def _mock_engine(*, supports_images, can_be_multimodal, loaded=True):
    engine = MagicMock()
    _state = {"loaded": loaded}
    engine.load.side_effect = lambda: _state.__setitem__("loaded", True)

    def _chat_stream(messages, **kwargs):
        yield "ok"

    engine.chat_stream.side_effect = _chat_stream
    engine.display_name = "test-model"
    engine.supports_images = supports_images
    engine.can_be_multimodal = can_be_multimodal
    engine.last_finish_reason = "stop"
    engine.count_tokens.return_value = 2
    engine.count_messages_tokens.return_value = 3
    type(engine).loaded = property(lambda self: _state["loaded"])
    return engine


class TestRouteRejectsImage(unittest.TestCase):
    def _post(self, engine, messages, stream=False):
        # Enter the app via `with` so the lifespan runs and the inference
        # semaphore is created on the event loop.
        with TestClient(create_app(engine)) as client:
            return client.post("/v1/chat/completions", json={
                "model": "test-model", "messages": messages, "stream": stream,
            })

    def test_text_only_engine_rejects_image_with_400(self):
        engine = _mock_engine(supports_images=False, can_be_multimodal=False)
        r = self._post(engine, _IMAGE_MSG)
        self.assertEqual(r.status_code, 400)
        self.assertIn("cannot accept image", r.json()["detail"])
        engine.load.assert_not_called()          # GGUF: no wasteful load to decide
        engine.chat_stream.assert_not_called()   # never reached the model

    def test_text_request_still_succeeds(self):
        engine = _mock_engine(supports_images=False, can_be_multimodal=False)
        r = self._post(engine, _TEXT_MSG)
        self.assertEqual(r.status_code, 200)

    def test_multimodal_engine_lets_image_through(self):
        engine = _mock_engine(supports_images=True, can_be_multimodal=True)
        r = self._post(engine, _IMAGE_MSG)
        self.assertEqual(r.status_code, 200)
        engine.chat_stream.assert_called_once()

    def test_unloaded_multimodal_candidate_loads_then_rejects(self):
        # An HF model that is not yet loaded: support is unknown until load.
        # The route must load it, then still reject because it is text-only.
        engine = _mock_engine(supports_images=False, can_be_multimodal=True,
                              loaded=False)
        r = self._post(engine, _IMAGE_MSG)
        self.assertEqual(r.status_code, 400)
        engine.load.assert_called_once()

    def test_message_constant_is_clear(self):
        # The user-facing string names the cause and the remedy.
        self.assertIn("vision", IMAGE_UNSUPPORTED_MESSAGE.lower())
        self.assertIn("GGUF", IMAGE_UNSUPPORTED_MESSAGE)


# --------------------------------------------------------------------------- #
#  Capability-aware vision guidance (no dead-end: route or guide per install)  #
# --------------------------------------------------------------------------- #

class TestVisionGuidance:
    def test_no_vision_model_gives_actionable_guidance(self, monkeypatch):
        import localm.model_manager as mm
        monkeypatch.setattr(mm, "load_registry", lambda: {})
        assert mm.vision_capable_models() == []
        msg = mm.vision_input_guidance()
        assert "cannot accept image" in msg               # legacy phrase preserved
        assert ("localm pull" in msg) or ("pip install" in msg)  # install-specific remedy

    def test_no_transformers_stack_offers_gguf_route_not_a_false_claim(
            self, monkeypatch):
        """The final fallback (no HF vision model registered, no transformers
        stack installed) used to claim 'the built-in GGUF backend is
        text-only' - flatly false: mtmd GGUF vision IS implemented (this same
        function's own mmproj_failed docstring says so, and #957's own live
        E2E proves it). The message must offer the GGUF+mmproj route instead
        of denying GGUF vision exists at all."""
        import localm.model_manager as mm
        import importlib.util
        monkeypatch.setattr(mm, "load_registry", lambda: {})
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        msg = mm.vision_input_guidance()
        assert "GGUF backend is text-only" not in msg     # the retracted false claim
        assert "mmproj" in msg
        assert "localm pull" in msg

    def test_mmproj_failed_message_is_honest_about_gguf(self, monkeypatch):
        # The user loaded a GGUF model WITH an mmproj expecting vision but the
        # projector did not load. GGUF vision IS implemented (#200), so the message
        # must blame the failed projector, NOT claim vision is unimplemented.
        import localm.model_manager as mm
        monkeypatch.setattr(mm, "load_registry", lambda: {})
        msg = mm.vision_input_guidance(mmproj_failed=True)
        assert "cannot accept image" in msg               # legacy phrase still present
        assert "mmproj" in msg
        assert "failed to load" in msg
        assert "not implemented" not in msg               # the old stale claim is gone
        # The default (text-only) message must NOT mention mmproj - but ONLY on the
        # branch where the HuggingFace stack IS installed. With transformers absent,
        # vision_input_guidance deliberately points at the GGUF route and names
        # --mmproj, which the sibling test above asserts is correct. Unpinned, this
        # assertion silently depended on whether the runner happened to have
        # transformers: it passed in a dev venv and failed in CI's `.[dev,rag]`,
        # which has none - so it could never pass on the only environment that gates
        # a merge.
        import importlib.util
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
        assert "mmproj" not in mm.vision_input_guidance()

    def test_registered_vlm_is_named_and_routed(self, tmp_path, monkeypatch):
        import localm.model_manager as mm
        d = tmp_path / "qwen-vl"
        d.mkdir()
        (d / "config.json").write_text(json.dumps(
            {"architectures": ["Qwen2VLForConditionalGeneration"], "vision_config": {}}),
            encoding="utf-8")
        monkeypatch.setattr(mm, "load_registry",
                            lambda: {"qwen-vl": {"path": str(d), "source": "local"}})
        assert "qwen-vl" in mm.vision_capable_models()
        msg = mm.vision_input_guidance()
        assert "qwen-vl" in msg and "localm run" in msg

    def test_gguf_entry_is_not_vision(self, tmp_path, monkeypatch):
        import localm.model_manager as mm
        f = tmp_path / "m.gguf"
        f.write_bytes(b"GGUF")
        monkeypatch.setattr(mm, "load_registry",
                            lambda: {"m": {"path": str(f), "source": "local"}})
        assert mm.vision_capable_models() == []           # a .gguf file is text-only

    def test_hf_vision_detected_via_preprocessor(self, tmp_path, monkeypatch):
        import localm.model_manager as mm
        d = tmp_path / "vlm2"
        d.mkdir()
        (d / "config.json").write_text('{"architectures": ["LlamaForCausalLM"]}',
                                       encoding="utf-8")
        (d / "preprocessor_config.json").write_text(
            '{"image_processor_type": "CLIPImageProcessor"}', encoding="utf-8")
        monkeypatch.setattr(mm, "load_registry",
                            lambda: {"vlm2": {"path": str(d), "source": "local"}})
        assert "vlm2" in mm.vision_capable_models()

    def test_gguf_with_recorded_mmproj_is_vision(self, tmp_path, monkeypatch):
        # ADR-0008 U9 / Theme 3(c): a GGUF chat model with a recorded, resolvable
        # mmproj is a genuinely working vision model - get_model_mmproj() is the
        # SAME lookup the real load path uses. vision_capable_models() must not
        # be blind to it just because it is a file, not an HF directory.
        import localm.model_manager as mm
        model_f = tmp_path / "vision-model.gguf"
        model_f.write_bytes(b"GGUF")
        mmproj_f = tmp_path / "vision-model-mmproj.gguf"
        mmproj_f.write_bytes(b"GGUF")
        monkeypatch.setattr(mm, "load_registry", lambda: {
            "vision-model": {
                "path": str(model_f), "source": "local", "model_type": "llm",
                "mmproj": str(mmproj_f),
            },
        })
        assert "vision-model" in mm.vision_capable_models()
        msg = mm.vision_input_guidance()
        assert "vision-model" in msg and "localm run" in msg
        assert "could be confirmed" not in msg

    def test_gguf_mmproj_entry_itself_is_not_a_vision_model(self, tmp_path, monkeypatch):
        # A standalone mmproj (projector) registry entry is not something a
        # user can switch to - it must never be listed even though it is a
        # GGUF file, and even in the pathological case where it happens to sit
        # beside another mmproj file get_model_mmproj's sibling scan could match.
        import localm.model_manager as mm
        proj_f = tmp_path / "some-mmproj.gguf"
        proj_f.write_bytes(b"GGUF")
        monkeypatch.setattr(mm, "load_registry", lambda: {
            "some-mmproj": {"path": str(proj_f), "source": "local",
                             "model_type": "mmproj"},
        })
        assert mm.vision_capable_models() == []

    def test_transformers_absent_does_not_claim_gguf_is_text_only(self, monkeypatch):
        # ADR-0008 U9 / Theme 3(c), third branch: when no vision model is
        # registered at all and transformers is absent, the message must not
        # claim "the built-in GGUF backend is text-only" - that contradicts
        # the mmproj_failed branch of this same function, which already treats
        # GGUF vision (the built-in mtmd path) as real and transformers-independent.
        import localm.model_manager as mm
        monkeypatch.setattr(mm, "load_registry", lambda: {})
        real_find_spec = importlib.util.find_spec

        def _no_transformers(name, *args, **kwargs):
            if name == "transformers":
                return None
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", _no_transformers)
        msg = mm.vision_input_guidance()
        # "this model ... is text-only" (the head, about the model that just
        # rejected the image) is a separate, true claim and stays. The bug was
        # the backend-wide claim - assert that exact false sentence is gone.
        assert "GGUF backend is text-only" not in msg
        assert "localm pull" in msg
        assert "mtmd" in msg

    def test_no_vlms_message_reports_the_search_not_absolute_absence(self, monkeypatch):
        # ADR-0008 R3: vision_capable_models() is a best-effort CONFIRMATION
        # check, not an exhaustive one - a genuinely vision-capable GGUF whose
        # mmproj is neither recorded nor a detectable sibling (e.g. supplied
        # only via the CLI --mmproj flag at run time, never written back to
        # the registry) is a real, unclosable blind spot. The message built on
        # an empty vlms list must report what was checked, not assert the user
        # has no vision model at all - or it launders a non-exhaustive check
        # into an absolute claim.
        import localm.model_manager as mm
        monkeypatch.setattr(mm, "load_registry", lambda: {})
        msg = mm.vision_input_guidance()
        assert "is registered yet" not in msg
        assert "could be confirmed" in msg


# --------------------------------------------------------------------------- #
#  #957: already-pulled GGUF vision model with no mmproj recorded yet          #
# --------------------------------------------------------------------------- #

class TestVisionGuidanceActiveModelBackfillPending:
    def _entry(self, source="hf:o/repo", model_type="llm", mmproj=None):
        return {
            "path": self.gguf_path,
            "source": source,
            "model_type": model_type,
            **({"mmproj": mmproj} if mmproj else {}),
        }

    def test_active_gguf_from_hf_with_no_mmproj_is_named(self, tmp_path, monkeypatch):
        import localm.model_manager as mm
        self.gguf_path = str(tmp_path / "m.gguf")
        (tmp_path / "m.gguf").write_bytes(b"GGUF")
        monkeypatch.setattr(mm, "load_registry",
                            lambda: {"my-vlm": self._entry()})
        msg = mm.vision_input_guidance(active_model_path=self.gguf_path)
        assert "cannot accept image" in msg               # legacy phrase preserved
        assert "my-vlm" in msg
        assert "o/repo" in msg
        assert "not implemented" not in msg
        # This is the #957 case, distinct from "no vision model registered":
        assert "no vision model is registered" not in msg.lower()

    def test_entry_already_carrying_mmproj_falls_through(self, tmp_path, monkeypatch):
        import localm.model_manager as mm
        self.gguf_path = str(tmp_path / "m.gguf")
        (tmp_path / "m.gguf").write_bytes(b"GGUF")
        monkeypatch.setattr(
            mm, "load_registry",
            lambda: {"my-vlm": self._entry(mmproj="o/repo:proj.gguf")})
        msg = mm.vision_input_guidance(active_model_path=self.gguf_path)
        assert "my-vlm" not in msg
        assert "o/repo" not in msg

    def test_non_hf_source_is_not_a_candidate(self, tmp_path, monkeypatch):
        import localm.model_manager as mm
        self.gguf_path = str(tmp_path / "m.gguf")
        (tmp_path / "m.gguf").write_bytes(b"GGUF")
        monkeypatch.setattr(mm, "load_registry",
                            lambda: {"my-vlm": self._entry(source="local")})
        msg = mm.vision_input_guidance(active_model_path=self.gguf_path)
        assert "my-vlm" not in msg

    def test_non_llm_model_type_is_not_a_candidate(self, tmp_path, monkeypatch):
        import localm.model_manager as mm
        self.gguf_path = str(tmp_path / "m.gguf")
        (tmp_path / "m.gguf").write_bytes(b"GGUF")
        monkeypatch.setattr(mm, "load_registry",
                            lambda: {"my-vlm": self._entry(model_type="mmproj")})
        msg = mm.vision_input_guidance(active_model_path=self.gguf_path)
        assert "my-vlm" not in msg

    def test_mmproj_failed_still_takes_priority_over_backfill_branch(
            self, tmp_path, monkeypatch):
        # A candidate entry is registered AND mmproj_failed=True is passed (the
        # projector WAS resolved and attached but failed to load at runtime) -
        # that is a more specific, more recent diagnosis than "never backfilled"
        # and must win.
        import localm.model_manager as mm
        self.gguf_path = str(tmp_path / "m.gguf")
        (tmp_path / "m.gguf").write_bytes(b"GGUF")
        monkeypatch.setattr(mm, "load_registry",
                            lambda: {"my-vlm": self._entry()})
        msg = mm.vision_input_guidance(mmproj_failed=True,
                                       active_model_path=self.gguf_path)
        assert "failed to load" in msg
        assert "my-vlm" not in msg

    def test_unregistered_path_falls_through(self, tmp_path, monkeypatch):
        import localm.model_manager as mm
        monkeypatch.setattr(mm, "load_registry", lambda: {})
        msg = mm.vision_input_guidance(active_model_path=str(tmp_path / "ghost.gguf"))
        assert "cannot accept image" in msg


if __name__ == "__main__":
    unittest.main()
