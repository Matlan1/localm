# SPDX-License-Identifier: AGPL-3.0-or-later
"""A model whose pre-tokenizer cannot hold a conversation is refused at LOAD.

``pretokenizer_guard.load_refusal`` decides it (pinned in
``tests/test_pretokenizer_guard.py::TestUnusablePolicies``); this file pins the
wiring that carries the decision to a user:

* the parent reads ``tokenizer.ggml.pre`` out of the GGUF header before any
  VRAM probe or worker spawn and refuses instantly;
* the worker's own metadata read refuses again after the native load, as the
  backstop for a header the bounded read cannot reach, and frees the model
  before a context is allocated for it;
* the refusal crosses the worker IPC as a TYPED error, so the parent reports
  it as written instead of wrapping it in runtime-repair advice;
* the embedder's two load sites do the same;
* the server maps it to the same 503 every other load failure gets.

Every "did it reach native code" assertion is made from OUTSIDE the call with
``assert_not_called`` on a plain mock, never by raising from a ``side_effect``.
"""

import asyncio
import gc
import queue
import struct
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import regex

from localm.inference import pretokenizer_guard as guard
from localm.inference.backends.base import (
    PretokenizerUnusableModelError, PretokenizerUnsafeInputError)

# --------------------------------------------------------------------------- #
#  GGUF header fixtures                                                        #
# --------------------------------------------------------------------------- #

_T_STRING = 8
_T_ARRAY = 9


def _kv_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<I", _T_STRING) + struct.pack("<Q", len(raw)) + raw


def _kv_string_array(values) -> bytes:
    out = struct.pack("<I", _T_ARRAY) + struct.pack("<I", _T_STRING)
    out += struct.pack("<Q", len(values))
    for v in values:
        raw = v.encode("utf-8")
        out += struct.pack("<Q", len(raw)) + raw
    return out


def _write_gguf(path, kv, *, version=3):
    body = b""
    for key, encoded in kv:
        kb = key.encode("utf-8")
        body += struct.pack("<Q", len(kb)) + kb + encoded
    header = (b"GGUF" + struct.pack("<I", version) + struct.pack("<Q", 0)
              + struct.pack("<Q", len(kv)))
    path.write_bytes(header + body)
    return path


def _model_gguf(path, *, pre="exaone-moe", arch="exaone4"):
    """The key order llama.cpp's converter writes: general.*, then
    ``tokenizer.ggml.model``, ``tokenizer.ggml.pre``, then the vocabulary."""
    kv = [("general.architecture", _kv_string(arch)),
          ("tokenizer.ggml.model", _kv_string("gpt2"))]
    if pre is not None:
        kv.append(("tokenizer.ggml.pre", _kv_string(pre)))
    kv.append(("tokenizer.ggml.tokens", _kv_string_array(["a", "b", "c"])))
    return _write_gguf(path, kv)


REFUSAL = guard.load_refusal("exaone-moe")


def test_the_fixture_pre_type_is_one_the_guard_refuses():
    assert REFUSAL is not None


# --------------------------------------------------------------------------- #
#  Why fragmenting is not an option                                            #
# --------------------------------------------------------------------------- #

class TestTheRefusedRunIsOnePreToken:
    """The alternative to refusing the model, tokenising in fragments at
    pre-tokenizer boundaries, needs a boundary INSIDE the refused run. There
    is none: the ``exaone-moe`` pattern matches a whole single-spaced sentence
    as one pre-token, so any split lands inside one BPE word and changes the
    tokens the model sees."""

    # The single regex llama.cpp runs for LLAMA_VOCAB_PRE_TYPE_EXAONE_MOE at
    # the pinned tag (src/llama-vocab.cpp, b10375), copied verbatim.
    PATTERN = (r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
               r"|[^\r\n\p{L}\p{N}]?(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+"
               r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]?|\s*[\r\n]|\s+(?!\S)|\s+")

    @staticmethod
    def _split(text):
        return [m.group() for m in regex.finditer(
            TestTheRefusedRunIsOnePreToken.PATTERN, text)]

    def test_a_single_spaced_sentence_is_one_pre_token(self):
        sentence = " ".join(["word"] * 40)
        assert len(sentence) > guard.UNSAFE_PRE_TYPES["exaone-moe"].max_run
        assert self._split(sentence) == [sentence]

    def test_what_does_end_the_run_is_what_the_guard_already_breaks_on(self):
        # Punctuation, a digit, a line break and a double space each end the
        # pre-token, and none of them is in the guard's run class, so the
        # guard never refuses a run that has a boundary inside it.
        assert self._split("hello, world foo") == ["hello", ",", " world foo"]
        assert self._split("hello 2 world") == ["hello", " ", "2", " world"]
        assert self._split("hello\nworld") == ["hello", "\n", "world"]
        assert self._split("hello  world") == ["hello", " ", " world"]
        for ch in ",2\n":
            assert not regex.fullmatch(guard._CLASS_LETTER_SPACE, ch)

    def test_a_fragment_started_at_a_space_is_a_different_pre_token(self):
        # Splitting "hello world" at the space does not reproduce the
        # original pre-token: the fragment's leading space is absorbed into a
        # new word, so the BPE merge domain changes at the split.
        whole = self._split("hello world")
        assert whole == ["hello world"]
        assert self._split("hello") + self._split(" world") == ["hello", " world"]


# --------------------------------------------------------------------------- #
#  The header read                                                             #
# --------------------------------------------------------------------------- #

class TestGgufPretokenizerRead:
    def _read(self, path):
        from localm.model_manager.gguf import gguf_pretokenizer
        return gguf_pretokenizer(path)

    def test_reads_the_declared_value(self, tmp_path):
        assert self._read(_model_gguf(tmp_path / "m.gguf")) == "exaone-moe"
        assert self._read(_model_gguf(tmp_path / "n.gguf", pre="qwen2")) == "qwen2"

    def test_a_file_declaring_none_reads_none(self, tmp_path):
        assert self._read(_model_gguf(tmp_path / "m.gguf", pre=None)) is None

    def test_a_key_past_the_bounded_read_reads_none(self, tmp_path):
        from localm.model_manager import gguf as mm
        big = _kv_string_array(["x" * 1000] * 5000)
        assert len(big) > mm._GGUF_META_PROBE_BYTES
        f = _write_gguf(tmp_path / "m.gguf", [
            ("general.architecture", _kv_string("exaone4")),
            ("tokenizer.ggml.tokens", big),
            ("tokenizer.ggml.pre", _kv_string("exaone-moe")),
        ])
        assert self._read(f) is None

    def test_not_a_gguf_reads_none(self, tmp_path):
        f = tmp_path / "m.gguf"
        f.write_bytes(b"not a gguf at all")
        assert self._read(f) is None

    def test_a_missing_file_reads_none(self, tmp_path):
        assert self._read(tmp_path / "absent.gguf") is None

    def test_a_v1_header_reads_none(self, tmp_path):
        f = _write_gguf(tmp_path / "m.gguf",
                        [("tokenizer.ggml.pre", _kv_string("exaone-moe"))],
                        version=1)
        assert self._read(f) is None


# --------------------------------------------------------------------------- #
#  GgufBackend.load: the parent refuses before VRAM or a worker is touched      #
# --------------------------------------------------------------------------- #

def _backend(path):
    from localm.inference.backends.gguf import GgufBackend
    b = GgufBackend.__new__(GgufBackend)
    b.model_path = str(path)
    b._ram_kv_hint_shown = False
    b.effective_gpu_layers = None
    return b


class TestGgufBackendLoadPreflight:
    def test_an_unusable_model_is_refused_before_vram_and_worker(self, tmp_path):
        b = _backend(_model_gguf(tmp_path / "m.gguf"))
        with patch.object(b, "_effective_gpu_layers", MagicMock(return_value=99)), \
                patch.object(b, "_check_vram", MagicMock()) as vram, \
                patch.object(b, "_load_native", MagicMock()) as native:
            with pytest.raises(PretokenizerUnusableModelError) as ei:
                b.load()
        assert str(ei.value) == REFUSAL
        vram.assert_not_called()
        native.assert_not_called()

    def test_an_unaffected_model_loads(self, tmp_path):
        b = _backend(_model_gguf(tmp_path / "m.gguf", pre="qwen2"))
        with patch.object(b, "_effective_gpu_layers", MagicMock(return_value=99)), \
                patch.object(b, "_check_vram", MagicMock()), \
                patch.object(b, "_load_native", MagicMock()) as native:
            b.load()
        native.assert_called_once()

    def test_an_unreadable_pre_type_defers_to_the_worker(self, tmp_path):
        # The header read reports None; the load proceeds and the worker's
        # own read is the backstop.
        b = _backend(_model_gguf(tmp_path / "m.gguf", pre=None))
        with patch.object(b, "_effective_gpu_layers", MagicMock(return_value=99)), \
                patch.object(b, "_check_vram", MagicMock()), \
                patch.object(b, "_load_native", MagicMock()) as native:
            b.load()
        native.assert_called_once()

    def test_the_worker_refusal_is_reported_as_written(self, tmp_path):
        # A RuntimeError from _load_native gets the runtime-repair advice
        # appended; the typed refusal must come through untouched.
        b = _backend(_model_gguf(tmp_path / "m.gguf", pre=None))
        with patch.object(b, "_effective_gpu_layers", MagicMock(return_value=99)), \
                patch.object(b, "_check_vram", MagicMock()), \
                patch.object(b, "_load_native",
                             MagicMock(side_effect=PretokenizerUnusableModelError(REFUSAL))):
            with pytest.raises(PretokenizerUnusableModelError) as ei:
                b.load()
        assert str(ei.value) == REFUSAL
        assert "setup-llama" not in str(ei.value)


# --------------------------------------------------------------------------- #
#  LlamaCpp.__init__: the worker-side backstop                                 #
# --------------------------------------------------------------------------- #

class TestLlamaCppBackstop:
    def _api(self, pre_type):
        mock_api = MagicMock()
        mock_api.llama_model_default_params.return_value = SimpleNamespace(
            main_gpu=0, n_gpu_layers=0, use_mmap=True)
        mock_api.llama_load_model_from_file.return_value = 0xB00
        mock_api.llama_model_meta_val_str.return_value = pre_type
        return mock_api

    def _build(self, mock_api):
        from localm.inference.backends.llamacpp.llama import LlamaCpp
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            llm = LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=99, verbose=True)
            llm.close()
            return llm

    def test_an_unusable_model_is_refused_and_freed_before_any_context(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config", lambda: {"main_gpu_index": None})
        mock_api = self._api("exaone-moe")
        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            from localm.inference.backends.llamacpp.llama import LlamaCpp
            with pytest.raises(PretokenizerUnusableModelError) as ei:
                LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=99, verbose=True)
            message = str(ei.value)
            # Drop the traceback's reference to the half-built instance and
            # collect it while the api is still patched, so its __del__ runs
            # here and a second free would be counted below.
            del ei
            gc.collect()
        assert message == REFUSAL
        mock_api.llama_free_model.assert_called_once_with(0xB00)
        mock_api.llama_init_from_model.assert_not_called()

    def test_an_unaffected_model_is_loaded(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config", lambda: {"main_gpu_index": None})
        mock_api = self._api("qwen2")
        self._build(mock_api)
        mock_api.llama_init_from_model.assert_called()


# --------------------------------------------------------------------------- #
#  The worker IPC carries the refusal typed                                    #
# --------------------------------------------------------------------------- #

class TestRunnerCarriesTheRefusalTyped:
    TAG = "PretokenizerUnusableModelError"

    def test_the_worker_tags_the_load_refusal(self):
        import inspect

        import localm.inference.backends.llamacpp._runner as runner
        assert f'"error", str(e), "{self.TAG}"' in inspect.getsource(runner)

    def _runner_with_reply(self, reply):
        from localm.inference.backends.llamacpp._runner import ModelRunner
        r = ModelRunner()

        def fake_spawn():
            r._req_q, r._resp_q, r._ctrl_q = queue.Queue(), queue.Queue(), queue.Queue()
            r._resp_q.put(reply)
            r._proc = MagicMock()
            r._proc.is_alive.return_value = True
        r._spawn = fake_spawn
        return r

    def test_the_parent_re_raises_the_tag_as_the_typed_error(self):
        r = self._runner_with_reply(("error", REFUSAL, self.TAG))
        with pytest.raises(PretokenizerUnusableModelError) as ei:
            r.spawn_and_load({}, timeout=5.0)
        assert str(ei.value) == REFUSAL

    def test_an_untagged_load_error_is_still_a_runtime_error(self):
        r = self._runner_with_reply(("error", "failed to load model"))
        with pytest.raises(RuntimeError) as ei:
            r.spawn_and_load({}, timeout=5.0)
        assert type(ei.value) is RuntimeError

    def test_the_error_is_importable_where_the_runner_expects_it(self):
        from localm.inference.backends.llamacpp._runner import (
            PretokenizerUnusableModelError as imported,
        )
        assert imported is PretokenizerUnusableModelError

    def test_the_dispatch_loop_emits_the_tagged_envelope(self, monkeypatch):
        # The real _runner_main on a thread, with the worker's load refusing
        # the way LlamaCpp.__init__ does; the envelope is read off the queue.
        import localm._mp_spawn as mp_spawn
        from localm.inference.backends.llamacpp import _runner, _worker

        monkeypatch.setattr(mp_spawn, "install_parent_death_watchdog", lambda *a: None)
        monkeypatch.setattr(mp_spawn, "suppress_native_error_dialogs", lambda *a: None)

        class _RefusingWorker:
            def __init__(self, **kw):
                pass

            def load(self):
                raise PretokenizerUnusableModelError(REFUSAL)

            def close(self):
                pass

        monkeypatch.setattr(_worker, "GgufWorker", _RefusingWorker)
        req_q, resp_q, ctrl_q = queue.Queue(), queue.Queue(), queue.Queue()
        died = []

        def _run():
            try:
                _runner._runner_main(req_q, resp_q, ctrl_q)
            except BaseException as e:      # noqa: BLE001 - escaping = the worker dies
                died.append(e)

        t = threading.Thread(target=_run, name="dispatch-under-test", daemon=True)
        t.start()
        try:
            req_q.put(("load", {}))
            envelope = resp_q.get(timeout=5)
            assert envelope[:2] == ("error", REFUSAL), envelope
            assert len(envelope) > 2 and envelope[2] == self.TAG, (
                f"the refusal crossed the IPC untagged ({envelope!r}), so the parent "
                f"would wrap it in runtime-repair advice")
            assert not died, died
        finally:
            req_q.put(None)
            t.join(timeout=5)


# --------------------------------------------------------------------------- #
#  The embedder's two load sites                                               #
# --------------------------------------------------------------------------- #

class TestEmbedderParentPreflight:
    def test_an_unusable_model_is_refused_before_vram_and_spawn(self, tmp_path):
        from localm.inference import embedder as emb
        with patch.object(emb.IsolatedEmbedder, "_preflight_vram", MagicMock()) as vram, \
                patch("localm.inference._embedder_runner.EmbedderRunner",
                      MagicMock()) as runner:
            with pytest.raises(PretokenizerUnusableModelError) as ei:
                emb.IsolatedEmbedder(str(_model_gguf(tmp_path / "e.gguf")),
                                     n_gpu_layers=0)
        assert str(ei.value) == REFUSAL
        vram.assert_not_called()
        runner.assert_not_called()

    def test_get_embedder_records_the_reason(self, tmp_path, monkeypatch):
        from localm.inference import embedder as emb
        path = str(_model_gguf(tmp_path / "e.gguf"))
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"embedding_model": "e.gguf",
                                     "n_gpu_layers": 0, "net_mode": "off"})
        monkeypatch.setattr(emb, "resolve_embedding_model_path",
                            lambda *, allow_download=None: path)
        monkeypatch.setattr(emb, "_maybe_swap_for_embedder", lambda *a, **k: None)
        monkeypatch.setattr(emb, "_choose_embedder_gpu_layers",
                            lambda *a, **k: (0, None))
        monkeypatch.setattr(emb, "_EMBEDDER", None)
        monkeypatch.setattr(emb, "_LOAD_FAILED_SPEC", None)
        monkeypatch.setattr(emb, "_LAST_ERROR", None)
        with patch.object(emb.IsolatedEmbedder, "_preflight_vram", MagicMock()), \
                patch("localm.inference._embedder_runner.EmbedderRunner",
                      MagicMock()) as runner:
            assert emb.get_embedder() is None
        assert emb.last_error() == REFUSAL
        runner.assert_not_called()


class TestEmbedderChildBackstop:
    def _patch_native(self, monkeypatch, pre_type):
        import localm.inference.backends.llamacpp._api as api_module
        calls = {"free": [], "init": 0}
        monkeypatch.setattr(api_module, "has_embeddings_api", lambda: True)
        monkeypatch.setattr(api_module, "llama_backend_init", lambda: None)
        monkeypatch.setattr(api_module, "llama_model_default_params",
                            lambda: SimpleNamespace())
        monkeypatch.setattr(api_module, "llama_load_model_from_file",
                            lambda path, mp: "model_ptr")
        monkeypatch.setattr(api_module, "llama_model_get_vocab", lambda model: "vocab")
        monkeypatch.setattr(api_module, "llama_model_n_embd", lambda model: 384)
        monkeypatch.setattr(api_module, "llama_model_n_ctx_train", lambda model: 512)
        monkeypatch.setattr(api_module, "has_model_meta_api", lambda: True)
        monkeypatch.setattr(api_module, "llama_model_meta_val_str",
                            lambda model, key: pre_type)
        monkeypatch.setattr(api_module, "llama_context_default_params",
                            lambda: SimpleNamespace())

        def _init(model, cp):
            calls["init"] += 1
            return "ctx_ptr"
        monkeypatch.setattr(api_module, "llama_init_from_model", _init)
        monkeypatch.setattr(api_module, "has_memory_api", lambda: False)
        monkeypatch.setattr(api_module, "llama_free", lambda ctx: None)
        monkeypatch.setattr(api_module, "llama_free_model",
                            lambda model: calls["free"].append(model))
        monkeypatch.setattr("localm.discover.apply_main_gpu", lambda *a, **k: None)
        monkeypatch.setattr("localm.discover.apply_gpu_split", lambda *a, **k: None)
        return calls

    def test_an_unusable_model_is_refused_and_freed_before_any_context(self, monkeypatch):
        from localm.inference import embedder as emb
        calls = self._patch_native(monkeypatch, "exaone-moe")
        with pytest.raises(PretokenizerUnusableModelError) as ei:
            emb.GGUFEmbedder("<stub-path>", n_gpu_layers=0)
        message = str(ei.value)
        # Collect the half-built instance while the api is still patched, so
        # its __del__ runs here and a second free would be counted below.
        del ei
        gc.collect()
        assert message == REFUSAL
        assert calls["free"] == ["model_ptr"]
        assert calls["init"] == 0

    def test_an_unaffected_model_is_loaded(self, monkeypatch):
        from localm.inference import embedder as emb
        calls = self._patch_native(monkeypatch, "bert")
        embedder = emb.GGUFEmbedder("<stub-path>", n_gpu_layers=0)
        try:
            assert calls["init"] == 1
        finally:
            embedder.close()


# --------------------------------------------------------------------------- #
#  The server reports it like any other load failure                           #
# --------------------------------------------------------------------------- #

class _RefusingEngine:
    def __init__(self):
        self.display_name = "model-a"
        self._loaded = False
        self.active_requests = 0

    @property
    def loaded(self):
        return self._loaded

    def set_load_cancel(self, ev):
        pass

    def load(self):
        raise PretokenizerUnusableModelError(REFUSAL)

    def unload(self):
        pass


class TestSwitchEngineReportsTheRefusal:
    def test_the_refusal_is_a_503_carrying_the_message(self, monkeypatch):
        import localm.inference.http_server as hs
        from fastapi import HTTPException
        from tests.conftest import probe_double

        reg = {"model-a": {"path": "models/model-a.gguf", "source": "local"}}
        monkeypatch.setattr("localm.config.load_registry", lambda: reg)
        monkeypatch.setattr("localm.model_manager.get_model_info",
                            lambda n: (f"models/{n}.gguf", "h"))
        monkeypatch.setattr("localm.discover.vram_info",
                            probe_double({"free": 10 * 1024 ** 3,
                                          "total": 16 * 1024 ** 3}))
        for d in (hs._engines, hs._engines_lru, hs._inference_sems,
                  hs._last_activity_per_model):
            d.clear()
        monkeypatch.setattr(hs, "_active_model_name", None)
        monkeypatch.setattr(hs, "_engine", None)
        monkeypatch.setattr(hs, "_inference_sem", None)
        monkeypatch.setattr(hs, "_switch_desired", None)
        monkeypatch.setattr(hs, "_switch_loading", None)
        monkeypatch.setattr(hs, "_switch_cancel", None)

        with pytest.raises(HTTPException) as ei:
            asyncio.run(hs.switch_engine("model-a", lambda n: _RefusingEngine()))
        assert ei.value.status_code == 503
        assert REFUSAL in ei.value.detail
        assert "model-a" not in hs._engines


class TestTheRefusalIsNotAPerRequestRefusal:
    def test_the_two_errors_are_distinct_types(self):
        # A per-request refusal is a ValueError the worker keeps serving after;
        # a load refusal is a RuntimeError that ends the load. Neither may be
        # caught by a handler written for the other.
        assert issubclass(PretokenizerUnusableModelError, RuntimeError)
        assert not issubclass(PretokenizerUnusableModelError, ValueError)
        assert not issubclass(PretokenizerUnsafeInputError, RuntimeError)
