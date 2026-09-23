# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm run MODEL` in this process: a turn that needs something MODEL does
not provide is answered by an installed model that has it, unless
--pin-model; and a turn MODEL refused for an image does not leave that image
in the conversation to be refused again on every later turn.

The capability answers come from the real registry readers over a real
registry shape (a recorded projector that exists on disk for the vision
model), not from a stubbed capability oracle."""

from __future__ import annotations

import pytest

from localm.cli import chat as chat_mod
from localm.inference.backends.base import UnsupportedInputError


class _Engine:
    def __init__(self, name, *, images=False, log=None):
        self.display_name = name
        self.loaded = False
        self.supports_images = images
        self.answered = 0
        self.seen = []
        self._log = log if log is not None else []

    def load(self):
        self.loaded = True
        self._log.append(("load", self.display_name))

    def unload(self):
        self.loaded = False
        self._log.append(("unload", self.display_name))

    def chat_stream(self, messages, **kw):
        from localm.inference.backends.base import messages_contain_image
        self.seen.append([dict(m) for m in messages])
        if messages_contain_image(messages) and not self.supports_images:
            raise UnsupportedInputError("cannot accept image input")
        self.answered += 1
        yield f"answered-by-{self.display_name}"

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def context_capacity(self):
        return 4096

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *_):
        self.unload()


@pytest.fixture
def reg(tmp_path, monkeypatch):
    for n in ("plain", "seer", "roomy"):
        (tmp_path / n).mkdir()
        (tmp_path / n / f"{n}.gguf").write_bytes(b"GGUF")
    proj = tmp_path / "seer" / "seer-mmproj.gguf"
    proj.write_bytes(b"GGUF")
    registry = {
        "plain": {"path": str(tmp_path / "plain" / "plain.gguf"), "source": "local",
                  "model_type": "llm", "context_length": 4096},
        "seer": {"path": str(tmp_path / "seer" / "seer.gguf"), "source": "local",
                 "model_type": "llm", "context_length": 8192, "mmproj": str(proj)},
        "roomy": {"path": str(tmp_path / "roomy" / "roomy.gguf"), "source": "local",
                  "model_type": "llm", "context_length": 65536},
    }
    monkeypatch.setattr("localm.config.load_registry", lambda: registry)
    monkeypatch.setattr("localm.model_manager.load_registry", lambda: registry)
    img = tmp_path / "cat.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    return registry, img


def _router(pinned=False):
    log = []
    engines = {}

    def build(name):
        return engines.setdefault(name, _Engine(name, images=(name == "seer"), log=log))

    primary = build("plain")
    primary.load()
    return chat_mod._TurnRouter(primary, "plain", pinned=pinned, build=build), engines, log


def _image_turn(img):
    return [chat_mod._build_user_message("what is this?", [str(img)])]


class TestInProcessRouting:
    def test_an_image_turn_is_answered_by_the_installed_vision_model(self, reg):
        _, img = reg
        router, engines, log = _router()
        eng = router.engine_for(_image_turn(img))
        assert eng is engines["seer"]
        assert eng.loaded and not engines["plain"].loaded
        assert log.index(("unload", "plain")) < log.index(("load", "seer")), \
            "the loaded model is freed before the routed one loads"
        assert "reading images" in router.note(eng)

    def test_a_plain_turn_goes_back_to_the_loaded_model(self, reg):
        _, img = reg
        router, engines, _ = _router()
        router.engine_for(_image_turn(img))
        eng = router.engine_for([{"role": "user", "content": "hi"}])
        assert eng is engines["plain"] and eng.loaded
        assert not engines["seer"].loaded
        assert router.note(eng) is None

    def test_pin_model_keeps_the_loaded_model(self, reg):
        _, img = reg
        router, engines, _ = _router(pinned=True)
        eng = router.engine_for(_image_turn(img))
        assert eng is engines["plain"]
        assert "seer" not in engines, "a pinned run never even builds another model"

    def test_a_conversation_outgrowing_the_model_goes_to_a_roomier_one(self, reg):
        router, engines, _ = _router()
        long = [{"role": "user", "content": "word " * 4000}]
        eng = router.engine_for(long)
        assert eng is engines["roomy"]
        assert "a longer conversation" in router.note(eng)

    def test_a_routed_model_that_fails_to_load_falls_back(self, reg, monkeypatch):
        _, img = reg
        router, engines, _ = _router()
        broken = _Engine("seer", images=True)
        broken.load = lambda: (_ for _ in ()).throw(RuntimeError("no VRAM"))
        engines["seer"] = broken
        eng = router.engine_for(_image_turn(img))
        assert eng is engines["plain"]
        assert engines["plain"].loaded


class TestFallbackKeepsCompaction:
    def test_a_roomier_model_that_fails_to_load_leaves_compaction_on(self, reg):
        router, engines, _ = _router()
        broken = _Engine("roomy")
        broken.load = lambda: (_ for _ in ()).throw(RuntimeError("no VRAM"))
        engines["roomy"] = broken
        long = [{"role": "user", "content": "word " * 12000}]
        assert router.plan(long).candidates == ("roomy",)
        eng = router.engine_for(long)
        assert eng is engines["plain"]
        assert router.min_context is None,             "the small model answers, so the conversation must still be compacted"


def _drive(monkeypatch, inputs, engine, router):
    it = iter(inputs)

    def fake_input(_prompt):
        v = next(it)
        if isinstance(v, BaseException):
            raise v
        return v

    monkeypatch.setattr(chat_mod.console, "input", fake_input)
    chat_mod._interactive(engine, None, {}, router=router)


class TestInteractive:
    def test_an_attached_image_is_answered_by_the_vision_model(self, reg, monkeypatch):
        _, img = reg
        router, engines, _ = _router()
        _drive(monkeypatch, [f"/image {img}", "what is this?", KeyboardInterrupt()],
               engines["plain"], router)
        assert engines["seer"].answered == 1
        assert engines["plain"].answered == 0

    def test_a_refused_image_is_removed_so_the_chat_keeps_working(self, reg, monkeypatch):
        """Pinned to a model that cannot read images: the refusal is shown,
        and the next turn is not refused for the same image."""
        _, img = reg
        router, engines, _ = _router(pinned=True)
        _drive(monkeypatch,
               [f"/image {img}", "what is this?", "and now text only", KeyboardInterrupt()],
               engines["plain"], router)
        plain = engines["plain"]
        assert plain.answered == 1, "the second turn is answered"
        assert plain.seen[-1] == [{"role": "user", "content": "and now text only"}], \
            "the refused turn is withdrawn, so no image and no unanswered turn remain"


class TestLoadFailures:
    def test_a_single_prompt_falls_back_when_the_routed_model_fails_to_load(
            self, reg, monkeypatch):
        from click.testing import CliRunner
        log = []
        plain = _Engine("plain", log=log)
        monkeypatch.setattr("localm.inference.engine.Engine", lambda *a, **k: plain)

        def build(name, **_):
            eng = _Engine(name, log=log)
            eng.load = lambda: (_ for _ in ()).throw(RuntimeError("no VRAM"))
            return eng
        monkeypatch.setattr(chat_mod, "_build_cli_engine", build)
        monkeypatch.setattr(chat_mod, "_maybe_persist_cli_mmproj", lambda *a, **k: None)
        result = CliRunner().invoke(
            chat_mod.run, ["plain", "--no-server", "-p", "word " * 12000])
        assert result.exception is None, result.output
        assert result.exit_code == 0
        assert plain.answered == 1, "the loaded model answers the prompt"
        assert "Could not load roomy" in result.output

    def test_a_model_that_fails_to_load_mid_chat_does_not_end_the_session(
            self, reg, monkeypatch):
        router, engines, _ = _router()
        plain = engines["plain"]
        plain.unload()
        plain.load = lambda: (_ for _ in ()).throw(RuntimeError("no VRAM"))
        _drive(monkeypatch, ["hi", "again", KeyboardInterrupt()], plain, router)
        assert plain.answered == 0
        assert plain.seen == []

