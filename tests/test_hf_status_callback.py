# SPDX-License-Identifier: AGPL-3.0-or-later
"""on_status must reach a caller of every concrete inference backend.

HFBackend.chat_stream declared on_status in its signature but never forwarded
it to HFRunner.chat_stream, so a client of an HF (transformers) model never
saw a status update advance past the initial guess - see gguf.py's chat_stream
for the working GGUF twin these tests mirror.

Five layers, each catching a different way this regresses:
  - HFBackend.chat_stream forwards on_status to self._runner.chat_stream (the
    exact line that was missing).
  - HFRunner.chat_stream (parent side) relays a "status" envelope from the
    isolated child, via a fake req_q/resp_q channel, before any chunk.
  - HFWorker.chat_stream (child side, in-process, no subprocess/real model)
    calls on_status at the prefill/decode and image-encoding boundaries.
  - A contract test enumerating every concrete BaseBackend subclass in
    localm.inference.backends, so a future backend that forgets to wire
    on_status through fails here instead of shipping silently.
  - A real end-to-end round trip through an actual isolated worker process
    (no mocks), proving the dispatch-loop closure that glues the two mocked
    halves above together actually works.
"""

from __future__ import annotations

import inspect
import multiprocessing as mp
import threading
from typing import List
from unittest.mock import MagicMock

import pytest

from localm.inference.backends._hf_runner import HFRunner
from localm.inference.backends._hf_worker import HFWorker
from localm.inference.backends.base import BaseBackend
from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.hf import HFBackend

_MESSAGES = [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------- #
# 1. HFBackend forwards on_status to its runner - the exact omission this
#    regression targets (hf.py's chat_stream -> self._runner.chat_stream).
# --------------------------------------------------------------------------- #

class _FakeHFRunner:
    """Stand-in for HFRunner: calls the on_status it was given before
    yielding, exactly like the real runner relays a "status" envelope ahead
    of the first chunk."""

    last_done: dict = {}

    def chat_stream(self, *, on_status=None, **_kwargs):
        if on_status:
            on_status("Generating response...")
        yield "hi"


class TestHFBackendForwardsOnStatus:
    def test_on_status_reaches_the_runner(self):
        backend = HFBackend("does-not-need-to-exist")
        backend._loaded = True
        backend._runner = _FakeHFRunner()

        received: List[str] = []
        tokens = list(backend.chat_stream(_MESSAGES, on_status=received.append))

        assert tokens == ["hi"]
        assert received == ["Generating response..."], (
            "HFBackend.chat_stream did not forward on_status to "
            "self._runner.chat_stream")

    def test_a_missing_on_status_is_still_fine(self):
        # The default/no-callback case must not regress: chat_stream still
        # runs and yields tokens with on_status left at its default of None.
        backend = HFBackend("does-not-need-to-exist")
        backend._loaded = True
        backend._runner = _FakeHFRunner()

        assert list(backend.chat_stream(_MESSAGES)) == ["hi"]


# --------------------------------------------------------------------------- #
# 2. HFRunner.chat_stream (parent side) relays a "status" envelope from a
#    fake child channel before any chunk. Mirrors
#    test_runner_stream_status.py's identical test for GGUF's ModelRunner.
# --------------------------------------------------------------------------- #

class _FakeHFProc:
    def __init__(self):
        self.terminated = False
        self.exitcode = 0

    def is_alive(self):
        return not self.terminated

    def terminate(self):
        self.terminated = True

    def join(self, timeout=None):
        return None


def _make_hf_runner() -> HFRunner:
    ctx = mp.get_context("spawn")
    r = HFRunner()
    r._req_q, r._resp_q, r._ctrl_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    r._proc = _FakeHFProc()
    return r


def _fake_hf_child_with_status(r, stop, *, statuses: List[str], tokens: List[str]):
    while not stop.is_set():
        try:
            cmd = r._req_q.get(timeout=0.05)
        except Exception:
            continue
        if cmd[0] != "chat_stream":
            continue
        for s in statuses:
            r._resp_q.put(("status", s))
        for t in tokens:
            r._resp_q.put(("chunk", t))
        r._resp_q.put(("done", {"finish_reason": "stop"}))
        return


class TestHFRunnerChatStreamRelaysStatus:
    def test_relays_status_before_chunks(self):
        r = _make_hf_runner()
        stop = threading.Event()
        child = threading.Thread(
            target=_fake_hf_child_with_status,
            args=(r, stop),
            kwargs=dict(
                statuses=["Processing prompt...", "Generating response..."],
                tokens=["Hello", " world"],
            ),
            daemon=True,
        )
        child.start()
        received: List[str] = []
        try:
            tokens = list(r.chat_stream(messages=[], on_status=received.append))
        finally:
            stop.set()
            child.join(2)

        assert tokens == ["Hello", " world"]
        assert received == ["Processing prompt...", "Generating response..."]

    def test_on_status_exception_does_not_abort_the_stream(self):
        r = _make_hf_runner()
        stop = threading.Event()
        child = threading.Thread(
            target=_fake_hf_child_with_status,
            args=(r, stop),
            kwargs=dict(statuses=["Processing prompt..."], tokens=["token1"]),
            daemon=True,
        )
        child.start()

        def _exploding_callback(s):
            raise ValueError("callback exploded")

        try:
            tokens = list(r.chat_stream(messages=[], on_status=_exploding_callback))
        finally:
            stop.set()
            child.join(2)

        assert tokens == ["token1"]


# --------------------------------------------------------------------------- #
# 3. HFWorker.chat_stream (child side) calls on_status at the prefill/decode
#    and image-encoding boundaries. In-process, no subprocess and no real
#    model/tokenizer - same mocking shape as
#    test_hf_worker_generate_thread_exception.py and
#    test_hf_max_tokens_unlimited.py.
# --------------------------------------------------------------------------- #

class _FakeStreamer:
    """Replaces transformers.TextIteratorStreamer so the yielded tokens are
    controlled directly, independent of the background generate() thread."""

    def __init__(self, *_a, **_k):
        pass

    def __iter__(self):
        return iter(["Hello", " world"])

    def end(self):
        pass


def _make_hf_worker(model, tokenizer, *, processor=None, is_multimodal=False):
    worker = HFWorker.__new__(HFWorker)
    worker._processor = processor
    worker._is_multimodal = is_multimodal
    worker._supports_image = is_multimodal
    worker._supports_audio = False
    worker.context_capacity = None
    worker._model = model
    worker._tokenizer = tokenizer
    worker.last_finish_reason = "stop"
    return worker


def _fake_model_and_tokenizer():
    model = MagicMock()
    model.device = "cpu"
    model.generation_config = None
    tokenizer = MagicMock()
    tokenizer.eos_token_id = None
    tokenizer.apply_chat_template.return_value = "fake prompt"
    tokenizer.return_value.to.return_value = {
        "input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]],
    }
    return model, tokenizer


@pytest.fixture(autouse=True)
def _skip_if_native_runtime_already_loaded():
    # chat_stream's `from transformers import StoppingCriteriaList, ...`
    # triggers a fresh `import torch`, the known-doomed DLL-identity conflict
    # with llama.cpp's native runtime if that already loaded earlier in this
    # same pytest worker - see test_hf_prompt_tokenization.py.
    from localm.inference.backends.llamacpp import _loader
    if _loader.native_lib_loaded():
        pytest.skip("llama.cpp's native runtime is already loaded in this "
                     "process; a fresh torch import here is the known-doomed "
                     "DLL-identity conflict, not this test's own subject")


class TestHFWorkerChatStreamCallsOnStatus:
    def test_generating_response_fires_before_the_first_yielded_token(self, monkeypatch):
        pytest.importorskip("transformers", exc_type=ImportError)
        import transformers

        monkeypatch.setattr(transformers, "TextIteratorStreamer", _FakeStreamer)
        monkeypatch.setattr(
            "localm.inference.backends._hf_worker._grammar_processor",
            lambda *a, **k: None)

        model, tokenizer = _fake_model_and_tokenizer()
        worker = _make_hf_worker(model, tokenizer)

        received: List[str] = []
        gen = worker.chat_stream(_MESSAGES, on_status=received.append)
        first = next(gen)

        assert first == "Hello"
        assert received == ["Processing prompt...", "Generating response..."], (
            "on_status must have received both stages before the first "
            "token was yielded")
        assert list(gen) == [" world"]
        assert model.generate.called

    def test_status_calls_are_in_order_and_generation_still_completes(self, monkeypatch):
        pytest.importorskip("transformers", exc_type=ImportError)
        import transformers

        monkeypatch.setattr(transformers, "TextIteratorStreamer", _FakeStreamer)
        monkeypatch.setattr(
            "localm.inference.backends._hf_worker._grammar_processor",
            lambda *a, **k: None)

        model, tokenizer = _fake_model_and_tokenizer()
        worker = _make_hf_worker(model, tokenizer)

        received: List[str] = []
        out = list(worker.chat_stream(_MESSAGES, on_status=received.append))

        assert out == ["Hello", " world"]
        assert received == ["Processing prompt...", "Generating response..."]

    def test_no_on_status_callback_is_still_fine(self, monkeypatch):
        # The default (on_status=None) path used by every direct caller that
        # does not care about status must not regress.
        pytest.importorskip("transformers", exc_type=ImportError)
        import transformers

        monkeypatch.setattr(transformers, "TextIteratorStreamer", _FakeStreamer)
        monkeypatch.setattr(
            "localm.inference.backends._hf_worker._grammar_processor",
            lambda *a, **k: None)

        model, tokenizer = _fake_model_and_tokenizer()
        worker = _make_hf_worker(model, tokenizer)

        assert list(worker.chat_stream(_MESSAGES)) == ["Hello", " world"]

    def test_encoding_image_fires_before_the_processor_runs(self, monkeypatch):
        pytest.importorskip("transformers", exc_type=ImportError)
        import transformers

        monkeypatch.setattr(transformers, "TextIteratorStreamer", _FakeStreamer)
        monkeypatch.setattr(
            "localm.inference.backends._hf_worker._grammar_processor",
            lambda *a, **k: None)
        monkeypatch.setattr(
            "localm.inference.media.decode_image_url", lambda url: object())

        model, tokenizer = _fake_model_and_tokenizer()
        processor = MagicMock()
        processor.apply_chat_template.return_value = "fake prompt"
        processor.return_value.to.return_value = {
            "input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]],
        }
        worker = _make_hf_worker(
            model, tokenizer, processor=processor, is_multimodal=True)

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            ],
        }]

        received: List[str] = []
        out = list(worker.chat_stream(messages, on_status=received.append))

        assert out == ["Hello", " world"]
        assert received == [
            "Encoding image...", "Processing prompt...", "Generating response...",
        ]

    def test_no_status_fires_for_a_text_only_multimodal_capable_model(self, monkeypatch):
        # A model whose processor CAN see images gets no "Encoding image..."
        # status on a turn that carries none.
        pytest.importorskip("transformers", exc_type=ImportError)
        import transformers

        monkeypatch.setattr(transformers, "TextIteratorStreamer", _FakeStreamer)
        monkeypatch.setattr(
            "localm.inference.backends._hf_worker._grammar_processor",
            lambda *a, **k: None)

        model, tokenizer = _fake_model_and_tokenizer()
        processor = MagicMock()
        worker = _make_hf_worker(
            model, tokenizer, processor=processor, is_multimodal=True)

        received: List[str] = []
        out = list(worker.chat_stream(_MESSAGES, on_status=received.append))

        assert out == ["Hello", " world"]
        assert "Encoding image..." not in received
        assert received == ["Processing prompt...", "Generating response..."]


# --------------------------------------------------------------------------- #
# 4. Contract: every concrete BaseBackend subclass in localm's own
#    inference.backends package invokes on_status during chat_stream, or is
#    explicitly exempted with a reason.
# --------------------------------------------------------------------------- #

# A backend that genuinely cannot report generation-stage status. Empty today:
# both concrete backends (GgufBackend, HFBackend) support on_status.
_ON_STATUS_EXEMPT: set = set()


def _concrete_production_backend_classes():
    """Every concrete (non-abstract) BaseBackend subclass defined in localm's
    own localm.inference.backends package, found by walking
    BaseBackend.__subclasses__() recursively (so a subclass of a subclass is
    still found).

    Filtered to that one package rather than accepting every subclass
    __subclasses__() reports: a test-only stand-in defined in another test
    module (e.g. test_backend_capability_error_contract.py's
    _MinimalBackend) registers itself on BaseBackend too, once that module
    has been collected, and never invokes on_status on purpose - it is not a
    production backend and must not make this contract test flap depending
    on collection order.
    """
    seen = set()
    found = []
    stack = [BaseBackend]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub in seen:
                continue
            seen.add(sub)
            stack.append(sub)
            if (sub.__module__.startswith("localm.inference.backends.")
                    and not inspect.isabstract(sub)):
                found.append(sub)
    return found


class _FakeContractRunner:
    """A minimal stand-in shared by every backend under test: chat_stream
    calls on_status once (if given) before yielding one token."""

    last_done: dict = {}

    def is_alive(self):
        return True

    def chat_stream(self, *, on_status=None, **_kwargs):
        if on_status:
            on_status("stage")
        yield "token"


def _stub_backend_for_contract(cls):
    """A loaded instance of *cls* wired to a fake runner, using the
    ``_loaded``/``_runner`` convention every current concrete backend shares
    (see GgufBackend.loaded / HFBackend.loaded, both gated on exactly those
    two attributes)."""
    backend = cls("does-not-need-to-exist")
    backend._loaded = True
    backend._runner = _FakeContractRunner()
    return backend


class TestEveryBackendInvokesOnStatus:
    def test_the_enumeration_finds_both_known_backends(self):
        # A canary for the walk itself: if this ever drops below the two
        # backends known to exist, the parametrized test below would pass by
        # finding nothing to check, silently losing its own coverage.
        classes = _concrete_production_backend_classes()
        assert GgufBackend in classes
        assert HFBackend in classes

    @pytest.mark.parametrize("cls", _concrete_production_backend_classes(),
                             ids=lambda c: c.__name__)
    def test_chat_stream_invokes_on_status(self, cls):
        if cls in _ON_STATUS_EXEMPT:
            pytest.skip(f"{cls.__name__} is on the explicit on_status exempt list")
        backend = _stub_backend_for_contract(cls)
        received: List[str] = []
        tokens = list(backend.chat_stream(_MESSAGES, on_status=received.append))
        assert tokens == ["token"]
        assert received, (
            f"{cls.__name__}.chat_stream never invoked on_status during a "
            "stubbed generation - wire it through to the runner, or add "
            f"{cls.__name__} to _ON_STATUS_EXEMPT with a reason")


# --------------------------------------------------------------------------- #
# 5. Real end-to-end round trip: an actual isolated worker process, no mocks.
#    Proves the dispatch-loop closure connecting HFWorker's on_status calls to
#    HFRunner's status-envelope relay (layers 2 and 3 above, each mocked
#    separately) actually works together. Marked @integration so the default
#    `pytest -m "not integration"` skips it (loads a real, tiny model).
# --------------------------------------------------------------------------- #

_TINY_MODEL = "sshleifer/tiny-gpt2"
_TINY_CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }}\n{% endfor %}"


@pytest.mark.integration
class TestRealWorkerRelaysStatusEndToEnd:
    @pytest.fixture(scope="class")
    def hf_backend(self, tmp_path_factory):
        pytest.importorskip("torch", exc_type=ImportError)
        pytest.importorskip("transformers", exc_type=ImportError)
        import json
        import shutil

        from huggingface_hub import snapshot_download

        try:
            local_dir = snapshot_download(_TINY_MODEL)
        except Exception as e:
            pytest.skip(f"could not fetch {_TINY_MODEL}: {e}")

        model_dir = tmp_path_factory.mktemp("tiny_gpt2_status")
        shutil.copytree(local_dir, model_dir, dirs_exist_ok=True)
        config_path = model_dir / "tokenizer_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["chat_template"] = _TINY_CHAT_TEMPLATE
        config_path.write_text(json.dumps(config), encoding="utf-8")

        be = HFBackend(str(model_dir), device="cpu")
        be.load()
        yield be
        be.unload()

    def test_on_status_receives_real_stage_updates_before_the_first_token(self, hf_backend):
        received: List[str] = []
        gen = hf_backend.chat_stream(
            [{"role": "user", "content": "Say hi."}],
            max_tokens=8, temperature=0.0,
            on_status=received.append,
        )
        first = next(gen)
        list(gen)   # drain the rest

        assert isinstance(first, str)
        assert received == ["Processing prompt...", "Generating response..."], received
