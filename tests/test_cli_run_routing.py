# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm run MODEL` in this process: a turn that needs something MODEL does
not provide is answered by an installed model that has it, unless
--pin-model; the conversation is compacted against the context window of the
model that answers it (attached to a server, a turn routed for context is sent
whole instead); a model that fails to load is reported, and only a model that
answered is named; and a turn MODEL refused for an image does not leave that
image in the conversation to be refused again on every later turn.

The capability answers come from the real registry readers over a real
registry shape (a recorded projector that exists on disk for the vision
model), not from a stubbed capability oracle."""

from __future__ import annotations

import pytest

from localm.cli import chat as chat_mod
from localm.inference.backends.base import (ContextCapacityExceededError,
                                            UnsupportedInputError)
from localm.inference.compact import estimate_tokens


class _Engine:
    def __init__(self, name, *, images=False, log=None, capacity=4096):
        self.display_name = name
        self.loaded = False
        self.supports_images = images
        self.capacity = capacity
        self.answered = 0
        self.seen = []
        self.opts = []
        self._log = log if log is not None else []

    def load(self):
        self.loaded = True
        self._log.append(("load", self.display_name))

    def unload(self):
        self.loaded = False
        self._log.append(("unload", self.display_name))

    def chat_stream(self, messages, **kw):
        from localm.inference.backends.base import messages_contain_image
        # Engine.chat_stream reloads an unloaded model.
        if not self.loaded:
            self.load()
        self.seen.append([dict(m) for m in messages])
        self.opts.append(dict(kw))
        if messages_contain_image(messages) and not self.supports_images:
            raise UnsupportedInputError("cannot accept image input")
        self.answered += 1
        yield f"answered-by-{self.display_name}"

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def context_capacity(self):
        return self.capacity

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *_):
        self.unload()


class _Bounded(_Engine):
    """An engine loaded with room for *capacity* tokens that refuses a prompt
    leaving less than 96 of them, as the GGUF backend does."""

    def __init__(self, name, *, capacity, log=None):
        super().__init__(name, log=log, capacity=capacity)
        self.overflows = 0

    def chat_stream(self, messages, **kw):
        n_prompt = estimate_tokens(messages, self.count_tokens)
        if n_prompt > self.capacity - 96:
            self.overflows += 1
            raise ContextCapacityExceededError(
                f"Conversation ({n_prompt} tokens) has outgrown "
                f"n_ctx_max={self.capacity}")
        yield from super().chat_stream(messages, **kw)


def _fails_once(engine, error="no VRAM"):
    """Make the next load of *engine* raise RuntimeError(*error*); every load
    after it succeeds."""
    real = engine.load
    failed = []

    def load():
        if not failed:
            failed.append(True)
            raise RuntimeError(error)
        real()
    engine.load = load


def _always_fails(engine, error="no VRAM"):
    engine.load = lambda: (_ for _ in ()).throw(RuntimeError(error))
    return engine


def _printed(monkeypatch):
    """The text localm.cli.chat prints from here on, one entry per print, with
    its markup removed."""
    from rich.text import Text
    lines = []

    def record(*args, **_):
        lines.append(" ".join(Text.from_markup(a).plain if isinstance(a, str)
                              else str(a) for a in args))
    monkeypatch.setattr(chat_mod.console, "print", record)
    return lines


def _max_resident(log):
    """The most models loaded at the same time over *log*'s load and unload
    events."""
    loaded, peak = set(), 0
    for event, name in log:
        if event == "load":
            loaded.add(name)
        elif event == "unload":
            loaded.discard(name)
        peak = max(peak, len(loaded))
    return peak


def _answered_turns(engines):
    """The last message of every conversation any engine answered."""
    return [conv[-1]["content"] for e in engines.values() for conv in e.seen]


def _words(n, word="word"):
    """*n* copies of *word* joined by spaces: about 5*n/4 estimated tokens, and
    unchanged by the REPL's strip of what is typed."""
    return " ".join([word] * n)


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


def _router(pinned=False, plain_capacity=None):
    """A router for plain, loaded in this process, whose other engines come
    from the returned dict. With *plain_capacity* plain is a _Bounded engine
    with that loaded window."""
    log = []
    engines = {}

    def build(name):
        return engines.setdefault(name, _Engine(name, images=(name == "seer"), log=log))

    if plain_capacity:
        engines["plain"] = _Bounded("plain", capacity=plain_capacity, log=log)
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

    def test_a_failed_candidate_does_not_name_a_model_that_does_not_answer(
            self, reg, monkeypatch):
        router, engines, _ = _router()
        engines["roomy"] = _always_fails(_Engine("roomy"))
        turn = [{"role": "user", "content": "word " * 4000}]
        assert router.plan(turn).candidates == ("roomy", "seer")
        out = _printed(monkeypatch)
        eng = router.engine_for(turn)
        assert eng is engines["seer"]
        assert out == ["Could not load roomy: no VRAM"], \
            "the failure is printed, and nothing names plain, which does not answer"
        assert router.note(eng).startswith("answered by seer: ")

    def test_when_every_candidate_fails_the_note_names_the_loaded_model(
            self, reg, monkeypatch):
        router, engines, _ = _router()
        for name in ("roomy", "seer"):
            engines[name] = _always_fails(_Engine(name))
        out = _printed(monkeypatch)
        eng = router.engine_for([{"role": "user", "content": "word " * 4000}])
        assert eng is engines["plain"] and eng.loaded
        assert out == ["Could not load roomy: no VRAM", "Could not load seer: no VRAM"]
        assert router.note(eng) == "answered by plain: could not load roomy, seer"

    def test_a_candidate_that_answers_leaves_failed_to_load_empty(self, reg):
        router, engines, _ = _router()
        engines["roomy"] = _always_fails(_Engine("roomy"))
        seer = engines["seer"] = _Engine("seer", images=True)
        _fails_once(seer)
        turn = [{"role": "user", "content": "word " * 4000}]
        router.engine_for(turn)
        assert router.failed_to_load == ("roomy", "seer")
        assert router.engine_for(turn) is seer
        assert router.failed_to_load == ()


class TestNamingTheAnsweringModel:
    """The model that answers a turn is named once, after it has answered."""

    def test_the_loaded_model_is_named_after_answering_for_models_that_failed(
            self, reg, monkeypatch):
        router, engines, _ = _router()
        for name in ("roomy", "seer"):
            engines[name] = _always_fails(_Engine(name))
        out = _printed(monkeypatch)
        _drive(monkeypatch, [_words(4000), "/clear", "hi", KeyboardInterrupt()],
               engines["plain"], router)
        assert engines["plain"].answered == 2
        named = [i for i, line in enumerate(out) if "answered by" in line]
        assert [out[i] for i in named] == \
            ["(answered by plain: could not load roomy, seer)"], out
        assert named[0] > out.index("Could not load seer: no VRAM")

    def test_nothing_is_named_when_the_loaded_model_fails_to_load_too(
            self, reg, monkeypatch):
        router, engines, _ = _router()
        for name in ("roomy", "seer"):
            engines[name] = _always_fails(_Engine(name))
        _always_fails(engines["plain"])
        out = _printed(monkeypatch)
        _drive(monkeypatch, [_words(4000), KeyboardInterrupt()], engines["plain"], router)
        assert "\nCould not load plain: no VRAM" in out
        assert [line for line in out
                if "answered by" in line or "answering with" in line.lower()] == []

    def test_a_loaded_model_that_refuses_the_image_is_not_named(self, reg, monkeypatch):
        _, img = reg
        router, engines, _ = _router()
        engines["seer"] = _always_fails(_Engine("seer", images=True))
        out = _printed(monkeypatch)
        _drive(monkeypatch, [f"/image {img}", "what is this?", KeyboardInterrupt()],
               engines["plain"], router)
        assert engines["plain"].answered == 0
        assert "Could not load seer: no VRAM" in out
        assert [line for line in out
                if "answered by" in line or "answering with" in line.lower()] == []


class TestFallbackKeepsCompaction:
    def test_the_loaded_model_compacts_when_the_roomier_ones_fail_to_load(
            self, reg, monkeypatch):
        """plain, loaded with room for 4096 tokens, answers a turn routed to
        models that failed to load, and compacts the conversation to its own
        window first."""
        router, engines, _ = _router(plain_capacity=4096)
        for name in ("roomy", "seer"):
            engines[name] = _always_fails(_Engine(name))
        turns = [f"turn {i} " + _words(500) for i in range(5)]
        out = _printed(monkeypatch)
        _drive(monkeypatch, [*turns, KeyboardInterrupt()], engines["plain"], router)
        assert "Could not load roomy: no VRAM" in out, "the last turn is routed"
        plain = engines["plain"]
        answered_last = [c for c in plain.seen if c[-1]["content"] == turns[-1]]
        assert answered_last, "plain answers the last turn"
        assert answered_last[0][0]["content"].startswith("[Conversation summary]"), \
            "the conversation is compacted before plain answers it"
        assert plain.overflows == 0


class TestRoutedTurnsAreCompacted:
    """plain is loaded with room for 4096 tokens; roomy was trained on 65536
    and is loaded with room for 8192."""

    def test_a_conversation_routed_for_context_keeps_answering_past_the_loaded_window(
            self, reg, monkeypatch):
        router, engines, log = _router(plain_capacity=4096)
        roomy = engines["roomy"] = _Bounded("roomy", capacity=8192, log=log)
        turns = [_words(4000)] + [f"turn {i} " + _words(600) for i in range(6)]
        out = _printed(monkeypatch)
        _drive(monkeypatch, [*turns, KeyboardInterrupt()], engines["plain"], router)
        answered = _answered_turns(engines)
        assert [t[:7] for t in turns if t not in answered] == [], \
            "turns went unanswered once the conversation outgrew roomy's loaded window"
        assert roomy.overflows == 0
        assert any("older conversation summarised" in line for line in out)

    def test_it_is_compacted_against_the_window_of_the_model_answering_it(
            self, reg, monkeypatch):
        router, engines, log = _router(plain_capacity=4096)
        roomy = engines["roomy"] = _Bounded("roomy", capacity=8192, log=log)
        routed, nearing = _words(3200), _words(1600, "more")
        _drive(monkeypatch, ["a", "b", "c", routed, nearing, KeyboardInterrupt()],
               engines["plain"], router)
        first = {conv[-1]["content"]: conv[0]["content"] for conv in roomy.seen}
        assert first.get(routed) == "a", \
            "a conversation under 70% of roomy's loaded window is sent whole"
        assert (first.get(nearing) or "").startswith("[Conversation summary]"), \
            "a conversation reaching 70% of roomy's loaded window is compacted"


class TestAttachedRouting:
    def test_a_turn_routed_for_context_is_sent_whole_with_the_window_it_needs(
            self, reg, monkeypatch):
        """Attached to a server, a turn that outgrows plain's trained window is
        sent whole, with min_context naming the window it needs."""
        server = _Engine("plain")
        server.load()
        router = chat_mod._TurnRouter(server, "plain", pinned=False)
        assert not router.in_process
        _drive(monkeypatch, ["a", "b", "c", _words(4000), KeyboardInterrupt()],
               server, router)
        assert server.seen[-1][0]["content"] == "a", "the conversation is sent whole"
        sent = server.opts[-1].get("min_context")
        assert sent is not None and sent > 4096, server.opts[-1]


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
        assert "(answered by plain: could not load roomy)" in result.output

    def test_a_single_prompt_the_loaded_model_refuses_names_no_model(
            self, reg, monkeypatch):
        from click.testing import CliRunner
        _, img = reg
        plain = _Engine("plain")
        monkeypatch.setattr("localm.inference.engine.Engine", lambda *a, **k: plain)
        monkeypatch.setattr(chat_mod, "_build_cli_engine",
                            lambda name, **_: _always_fails(_Engine(name, images=True)))
        monkeypatch.setattr(chat_mod, "_maybe_persist_cli_mmproj", lambda *a, **k: None)
        result = CliRunner().invoke(
            chat_mod.run,
            ["plain", "--no-server", "-p", "what is this?", "--image", str(img)])
        assert result.exception is None, result.output
        assert plain.answered == 0
        assert "Could not load seer: no VRAM" in result.output
        assert "answered by" not in result.output, result.output

    def test_a_single_prompt_answered_by_the_next_candidate_names_only_it(
            self, reg, monkeypatch):
        from click.testing import CliRunner
        log = []
        plain = _Engine("plain", log=log)
        monkeypatch.setattr("localm.inference.engine.Engine", lambda *a, **k: plain)
        built = {}

        def build(name, **_):
            eng = built[name] = _Engine(name, log=log)
            return _always_fails(eng) if name == "roomy" else eng
        monkeypatch.setattr(chat_mod, "_build_cli_engine", build)
        monkeypatch.setattr(chat_mod, "_maybe_persist_cli_mmproj", lambda *a, **k: None)
        result = CliRunner().invoke(
            chat_mod.run, ["plain", "--no-server", "-p", "word " * 4000])
        assert result.exception is None, result.output
        assert built["seer"].answered == 1 and plain.answered == 0
        assert "Could not load roomy: no VRAM" in result.output
        assert "answering with" not in result.output.lower(), result.output
        assert "answered by seer" in result.output

    def test_a_model_that_fails_to_load_mid_chat_does_not_end_the_session(
            self, reg, monkeypatch):
        router, engines, _ = _router()
        plain = engines["plain"]
        plain.unload()
        _fails_once(plain)
        _drive(monkeypatch, ["hi", "again", KeyboardInterrupt()], plain, router)
        assert plain.seen == [[{"role": "user", "content": "again"}]], \
            "the next turn is answered, without the withdrawn one"
        assert plain.answered == 1

    def test_after_a_failed_load_the_unloaded_model_is_not_brought_back(
            self, reg, monkeypatch):
        """seer answers the image turns. The next turn needs vision and a
        window no installed model has, so it goes to plain, whose load fails
        once. /compact and the turn after it use plain; seer stays unloaded."""
        _, img = reg
        router, engines, log = _router()
        plain = engines["plain"]
        _fails_once(plain)
        needs_what_no_model_has = _words(12000)
        _drive(monkeypatch,
               [f"/image {img}", "what is this?", "t1", "t2",
                needs_what_no_model_has, "/compact", "ok", KeyboardInterrupt()],
               plain, router)
        assert _max_resident(log) == 1, f"two models were loaded at once: {log}"
        assert not engines["seer"].loaded
        assert plain.seen[-1][-1] == {"role": "user", "content": "ok"}

