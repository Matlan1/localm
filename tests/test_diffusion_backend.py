# SPDX-License-Identifier: AGPL-3.0-or-later
"""Diffusion language models (Dream, LLaDA, LLaDA-MoE, RND1) through the GGUF
backend: they load as chat models, refuse a grammar, charge no KV cache, carry
their settings to the worker, answer through the denoising path with step
status, and stop between steps when the stream is cancelled."""

from __future__ import annotations

import queue
import struct
import threading
from unittest.mock import patch

import pytest

from localm.inference.backends.base import (
    GRAMMAR_DIFFUSION_UNSUPPORTED_MESSAGE, ContextCapacityExceededError,
    GrammarUnsupportedError)
from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp import _diffusion
from localm.inference.backends.llamacpp import llama as llama_mod
from localm.inference.backends.llamacpp.llama import LlamaCpp

_T_STRING = 8


def _gguf(path, arch: str):
    raw = arch.encode()
    key = b"general.architecture"
    body = (struct.pack("<Q", len(key)) + key + struct.pack("<I", _T_STRING)
            + struct.pack("<Q", len(raw)) + raw)
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                     + struct.pack("<Q", 1) + body + b"\0" * 4096)
    return path


class TestBackendCapabilities:
    def test_diffusion_model_refuses_grammar_loaded_or_not(self, tmp_path):
        b = GgufBackend(str(_gguf(tmp_path / "m.gguf", "llada-moe")))
        assert b.is_diffusion is True
        assert b.supports_grammar is False
        with pytest.raises(GrammarUnsupportedError) as caught:
            b.validate_grammar("root ::= \"a\"")
        assert str(caught.value) == GRAMMAR_DIFFUSION_UNSUPPORTED_MESSAGE
        b.validate_grammar(None)

    def test_autoregressive_model_keeps_grammar(self, tmp_path):
        b = GgufBackend(str(_gguf(tmp_path / "m.gguf", "qwen3")))
        assert b.is_diffusion is False
        assert b.supports_grammar is True

    def test_diffusion_model_reaches_the_loader(self, tmp_path, monkeypatch):
        b = GgufBackend(str(_gguf(tmp_path / "m.gguf", "dream")))
        called = []
        monkeypatch.setattr(b, "_effective_gpu_layers", lambda: 99)
        monkeypatch.setattr(b, "_check_vram", lambda: called.append("vram"))
        monkeypatch.setattr(b, "_load_native", lambda: called.append("load"))
        b.load()
        assert called == ["vram", "load"]


class TestSizing:
    def test_no_kv_cache_is_charged(self, tmp_path):
        b = GgufBackend(str(_gguf(tmp_path / "m.gguf", "llada")), n_ctx=8192)
        assert b._kv_bytes_per_token() == 0
        assert b._full_offload_parts(1)[1] == 0

    def test_autoregressive_model_is_still_charged(self, tmp_path):
        b = GgufBackend(str(_gguf(tmp_path / "m.gguf", "qwen3")), n_ctx=8192)
        assert b._kv_bytes_per_token() > 0

    def test_auto_context_ceiling_does_not_divide_by_zero(self, tmp_path, monkeypatch):
        b = GgufBackend(str(_gguf(tmp_path / "m.gguf", "rnd1")), n_ctx=4096)
        monkeypatch.setattr(b, "_split_free_total_bytes", lambda: (8 << 30, 16 << 30, 1))
        assert b._auto_ctx_max() == max(4096, b._AUTO_CTX_MIN)

    def test_loaded_model_answer_wins_over_the_file(self, tmp_path):
        b = GgufBackend(str(_gguf(tmp_path / "m.gguf", "qwen3")))

        class _Llm:
            is_diffusion = True
            kv_bytes_per_token = 1234
        b._llm = _Llm()
        assert b._kv_bytes_per_token() == 0


class TestWorkerParams:
    def _load(self, b, cfg):
        from localm.config import DEFAULT_CONFIG
        merged = {**DEFAULT_CONFIG, **cfg}
        with patch("localm.config.load_config", return_value=merged), \
             patch("localm.discover.list_gpus", return_value=([], "ok")), \
             patch("localm.inference.backends.llamacpp._runner.ModelRunner."
                   "spawn_and_load", return_value={"diffusion": True}) as spawn:
            b.effective_gpu_layers = 0
            b._load_native()
        return spawn.call_args[0][0]

    def test_diffusion_settings_reach_the_worker(self, tmp_path):
        b = GgufBackend(str(_gguf(tmp_path / "m.gguf", "dream")))
        params = self._load(b, {"diffusion_steps": 77, "diffusion_max_tokens": 64})
        assert params["diffusion_steps"] == 77
        assert params["diffusion_max_tokens"] == 64

    def test_unset_steps_are_not_sent(self, tmp_path):
        b = GgufBackend(str(_gguf(tmp_path / "m.gguf", "dream")))
        params = self._load(b, {"diffusion_steps": None})
        assert "diffusion_steps" not in params

    def test_autoregressive_model_gets_no_diffusion_settings(self, tmp_path):
        b = GgufBackend(str(_gguf(tmp_path / "m.gguf", "qwen3")))
        params = self._load(b, {"diffusion_steps": 77})
        assert "diffusion_steps" not in params and "diffusion_max_tokens" not in params


class _FakeCanvas:
    """Stands in for NativeCanvas: fills the canvas with a scripted reply."""

    instances: list = []
    reply = [11, 12, 2, 13]

    def __init__(self, api, ctx, vocab, n_vocab, params, guard):
        self.params = params
        self.guard = guard
        self.closed = False
        self.decodes = 0
        _FakeCanvas.instances.append(self)

    def decode(self, tokens):
        with self.guard():
            self.decodes += 1
            return 0

    def sample(self, row, algorithm, greedy):
        pos = row + 1 if self.params.shift_logits else row
        offset = pos - self._n_input
        token = self.reply[offset] if 0 <= offset < len(self.reply) else 2
        return token, 0.5

    def close(self):
        self.closed = True


def _bare_llama(monkeypatch, *, capacity=256, steps=8, max_tokens=None, arch="dream"):
    from tests._bare_llama import make_bare_llama

    class _Tok:
        _vocab = 1

        @staticmethod
        def is_eog(t):
            return t == 2
    llm = make_bare_llama(
        _model_ptr=1, _ctx_ptr=1, _seed=0, _verbose=True, _tokenizer=_Tok(),
        is_diffusion=True, architecture=arch, _diffusion_mask=99,
        _diffusion_shift_logits=False, _diffusion_capacity=capacity,
        _diffusion_steps=steps, _diffusion_max_tokens=max_tokens)
    native_calls = []
    monkeypatch.setattr(llama_mod.api, "llama_set_causal_attn",
                        lambda ctx, causal: native_calls.append(("causal", causal)))
    monkeypatch.setattr(llama_mod.api, "llama_vocab_n_tokens", lambda vocab: 100)
    _FakeCanvas.instances = []
    monkeypatch.setattr(_diffusion, "NativeCanvas", _FakeCanvas)
    return llm, native_calls


def _run(llm, prompt, **kw):
    _FakeCanvas._n_input = len(prompt)
    return list(llm._generate_diffusion(prompt, **{
        "max_new_tokens": 64, "temperature": 0.5, "top_k": 40, "top_p": 0.95, **kw}))


class TestGenerateDiffusion:
    def test_reply_ends_at_the_end_token_with_status(self, monkeypatch):
        llm, calls = _bare_llama(monkeypatch)
        statuses = []
        out = _run(llm, [5, 6, 7], on_status=statuses.append)
        assert out == [11, 12]
        assert llm.last_finish_reason == "stop"
        assert calls == []
        assert statuses[0] == "Denoising reply (0%)..."
        assert all(s.startswith("Denoising reply (") for s in statuses)
        assert len(statuses) == len(set(statuses)) <= 11
        assert _FakeCanvas.instances[0].closed is True

    def test_request_max_tokens_cuts_a_block_rounded_reply(self, monkeypatch):
        llm, _ = _bare_llama(monkeypatch, arch="llada")
        _FakeCanvas.reply = [11, 12, 13, 14, 2]
        try:
            out = _run(llm, [5, 6, 7], max_new_tokens=2)
        finally:
            _FakeCanvas.reply = [11, 12, 2, 13]
        assert out == [11, 12]
        assert llm.last_finish_reason == "length"

    def test_grammar_is_refused_before_any_native_call(self, monkeypatch):
        llm, calls = _bare_llama(monkeypatch)
        with pytest.raises(GrammarUnsupportedError):
            _run(llm, [5, 6, 7], grammar="root ::= \"a\"")
        assert calls == [] and _FakeCanvas.instances == []

    def test_prompt_too_long_is_a_capacity_error(self, monkeypatch):
        llm, calls = _bare_llama(monkeypatch, capacity=20)
        with pytest.raises(ContextCapacityExceededError, match="diffusion window"):
            _run(llm, list(range(10)))
        assert calls == []

    def test_stream_cancel_stops_between_steps_with_nothing_yielded(self, monkeypatch):
        llm, _ = _bare_llama(monkeypatch, steps=8)
        polls = []

        def should_stop():
            polls.append(1)
            return len(polls) > 2
        out = _run(llm, [5, 6, 7], should_stop=should_stop)
        assert out == []
        assert llm.last_finish_reason == "stop"
        assert _FakeCanvas.instances[0].decodes == 2
        assert _FakeCanvas.instances[0].closed is True

    def test_unload_mid_run_stops_and_reports_an_error(self, monkeypatch):
        llm, _ = _bare_llama(monkeypatch, steps=8)
        original = _FakeCanvas.decode

        def decode_then_unload(self, tokens):
            code = original(self, tokens)
            llm._stop.set()
            return code
        monkeypatch.setattr(_FakeCanvas, "decode", decode_then_unload)
        assert _run(llm, [5, 6, 7]) == []
        assert llm.last_finish_reason == "error"
        assert _FakeCanvas.instances[0].closed is True

    def test_dispatch_from_create_chat_completion(self, monkeypatch):
        llm, _ = _bare_llama(monkeypatch)
        seen = {}

        def fake_generate(tokens, **kw):
            seen.update(kw)
            yield from ()
        monkeypatch.setattr(llm, "_generate_diffusion", fake_generate)
        monkeypatch.setattr(llama_mod, "_apply_model_template",
                            lambda model, messages: ("hi", None))
        monkeypatch.setattr(llama_mod, "_untrusted_prompt_ranges", lambda *a: ())
        llm._tokenizer.encode = lambda text, add_bos=True, untrusted_ranges=(): [1, 2]
        llm._mtmd = None
        stop = threading.Event().is_set
        out = llm.create_chat_completion(
            [{"role": "user", "content": "hi"}], grammar="g", should_stop=stop)
        assert out["choices"][0]["message"]["content"] == ""
        assert seen["grammar"] == "g" and seen["should_stop"] is stop


class TestWorkerPlumbing:
    def test_chat_stream_hands_the_cancel_check_to_the_model(self):
        from localm.inference.backends.llamacpp._worker import GgufWorker
        w = GgufWorker("m.gguf", None, 512, 0, None, 512)
        captured = {}

        class _Llm:
            def create_chat_completion(self, **kw):
                captured.update(kw)
                return iter([{"choices": [{"delta": {}, "finish_reason": "stop"}]}])
        w._llm = _Llm()
        ev = threading.Event()
        w.stream_cancel = ev
        assert list(w.chat_stream([{"role": "user", "content": "x"}])) == []
        assert captured["should_stop"] == ev.is_set

    def test_no_cancel_event_sends_no_cancel_check(self):
        from localm.inference.backends.llamacpp._worker import GgufWorker
        w = GgufWorker("m.gguf", None, 512, 0, None, 512)
        captured = {}

        class _Llm:
            def create_chat_completion(self, **kw):
                captured.update(kw)
                return iter([])
        w._llm = _Llm()
        list(w.chat_stream([{"role": "user", "content": "x"}]))
        assert "should_stop" not in captured

    def test_load_forwards_diffusion_settings_and_reports_the_role(self, monkeypatch):
        from localm.inference.backends.llamacpp import _worker
        seen = {}

        class _FakeLlama:
            def __init__(self, **kw):
                seen.update(kw)
                self.is_diffusion = True
                self.supports_images = False
        monkeypatch.setattr("localm.inference.backends.llamacpp._loader.load_lib", lambda: None)
        monkeypatch.setattr("localm.inference.backends.llamacpp.LlamaCpp", _FakeLlama)
        w = _worker.GgufWorker("m.gguf", None, 512, 0, None, 512,
                               diffusion_steps=40, diffusion_max_tokens=96)
        meta = w.load()
        assert seen["diffusion_steps"] == 40 and seen["diffusion_max_tokens"] == 96
        assert meta["diffusion"] is True


class TestRunnerCancel:
    def test_cancel_stream_reaches_a_running_diffusion_generation(self, monkeypatch):
        from localm.inference.backends.llamacpp import _runner
        for name in ("install_parent_death_watchdog", "ignore_interrupt_signals",
                     "suppress_native_error_dialogs"):
            monkeypatch.setattr(f"localm._mp_spawn.{name}", lambda: None)
        monkeypatch.setattr("localm.debuglog.attach_child_logging", lambda: None)
        req_q, resp_q, ctrl_q = queue.Queue(), queue.Queue(), queue.Queue()
        observed = {}

        class _Worker:
            last_finish_reason = "stop"
            grammar_unsupported_this_call = False
            chatml_fallback_reason = None
            mtp_status = None
            mtp_active_this_call = False
            mtp_call_status = ""
            mtp_drafted = mtp_accepted = mtp_steps = mtp_paused_steps = 0
            mtp_skipped = ""
            spec_report = None

            def __init__(self, cancel_event=None, **payload):
                self.stream_cancel = None

            def load(self):
                return {}

            def chat_stream(self, on_status=None, **payload):
                ctrl_q.put(("cancel_stream",))
                stop = self.stream_cancel
                observed["stopped"] = stop is not None and stop.wait(5.0)
                return iter(())

            def close(self):
                pass
        monkeypatch.setattr("localm.inference.backends.llamacpp._worker.GgufWorker", _Worker)
        req_q.put(("load", {}))
        req_q.put(("chat_stream", {"messages": []}))
        req_q.put(None)
        _runner._runner_main(req_q, resp_q, ctrl_q)
        ctrl_q.put(None)
        assert resp_q.get_nowait()[0] == "ok"
        assert resp_q.get_nowait()[0] == "done"
        assert observed["stopped"] is True


class _FakeProc:
    exitcode = 0

    def __init__(self):
        self.terminated = False

    def is_alive(self):
        return not self.terminated

    def terminate(self):
        self.terminated = True

    def join(self, timeout=None):
        return None


def _runner_with_denoising_child(steps=10, step_seconds=0.05, status_every=1):
    """A ModelRunner whose fake child sends a status every *status_every*
    steps and stops at the first step after a cancel_stream arrives, like the
    diffusion worker."""
    import multiprocessing as mp
    from localm.inference.backends.llamacpp._runner import ModelRunner
    ctx = mp.get_context("spawn")
    r = ModelRunner()
    r._req_q, r._resp_q, r._ctrl_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    r._proc = _FakeProc()
    seen = {"cancel": False, "steps": 0}

    def child():
        cmd = r._req_q.get(timeout=5)
        assert cmd[0] == "chat_stream"
        for step in range(steps):
            try:
                if r._ctrl_q.get(timeout=step_seconds)[0] == "cancel_stream":
                    seen["cancel"] = True
                    r._resp_q.put(("done", {"finish_reason": "stop", "cancelled": True}))
                    return
            except Exception:
                pass
            seen["steps"] += 1
            if step % status_every == 0:
                r._resp_q.put(("status", f"Denoising reply ({step * 10}%)..."))
        r._resp_q.put(("chunk", "full reply"))
        r._resp_q.put(("done", {"finish_reason": "stop"}))
    t = threading.Thread(target=child, daemon=True)
    t.start()
    return r, seen, t


class TestRunnerStopRequest:
    def test_a_stop_request_cancels_the_child_between_statuses(self):
        from localm.inference.backends.base import stream_stop_check
        r, seen, t = _runner_with_denoising_child(steps=60, step_seconds=0.05,
                                                  status_every=1000)
        stop = threading.Event()
        threading.Timer(0.4, stop.set).start()
        with stream_stop_check(stop.is_set):
            out = list(r.chat_stream(messages=[], stop_on_request=True))
        t.join(5)
        assert out == []
        assert seen["cancel"] is True and seen["steps"] < 60
        assert r.last_done == {"finish_reason": "stop", "cancelled": True}
        assert r._proc.terminated is False

    def test_without_the_flag_a_stop_request_is_ignored(self):
        from localm.inference.backends.base import stream_stop_check
        r, seen, t = _runner_with_denoising_child()
        with stream_stop_check(lambda: True):
            out = list(r.chat_stream(messages=[]))
        t.join(5)
        assert out == ["full reply"]
        assert seen["cancel"] is False

    def test_keyboard_interrupt_cancels_and_drains_before_propagating(self):
        r, seen, t = _runner_with_denoising_child()

        def on_status(s):
            if s == "Denoising reply (30%)...":
                raise KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            list(r.chat_stream(messages=[], on_status=on_status))
        t.join(5)
        assert seen["cancel"] is True
        assert r._resp_q.empty()
        assert r._proc.terminated is False

    def test_keyboard_interrupt_waits_a_whole_step_for_a_diffusion_child(self, monkeypatch):
        from localm.inference.backends.llamacpp import _runner
        waits = []
        monkeypatch.setattr(_runner.ModelRunner, "_cancel_stream_and_drain",
                            lambda self, timeout=_runner._CANCEL_DRAIN_TIMEOUT: waits.append(timeout))
        for flag in (True, False):
            r, seen, t = _runner_with_denoising_child()

            def on_status(s):
                raise KeyboardInterrupt()
            with pytest.raises(KeyboardInterrupt):
                list(r.chat_stream(messages=[], on_status=on_status, stop_on_request=flag))
        assert waits == [_runner._STREAM_CHUNK_TIMEOUT, _runner._CANCEL_DRAIN_TIMEOUT]

    def test_backend_asks_for_stop_requests_only_for_diffusion(self, tmp_path):
        seen = {}

        class _Runner:
            last_done = {"finish_reason": "stop"}

            def chat_stream(self, **kw):
                seen.update(kw)
                return iter(())
        for arch, expected in (("dream", True), ("qwen3", False)):
            b = GgufBackend(str(_gguf(tmp_path / f"{arch}.gguf", arch)))
            b._runner = _Runner()
            b._loaded = True
            list(b.chat_stream([{"role": "user", "content": "x"}]))
            assert seen["stop_on_request"] is expected

    def test_stop_check_scoping(self):
        from localm.inference.backends.base import stream_stop_check, stream_stop_requested
        assert stream_stop_requested() is False
        with stream_stop_check(lambda: True):
            assert stream_stop_requested() is True
            with stream_stop_check(lambda: False):
                assert stream_stop_requested() is False
            assert stream_stop_requested() is True
        assert stream_stop_requested() is False

        def broken():
            raise OSError("check failed")
        with stream_stop_check(broken):
            assert stream_stop_requested() is False


class TestLoadSetup:
    """LlamaCpp.__init__ over a mocked native API, the pattern of
    tests/test_main_gpu_wiring.py: what a diffusion model's load sets up."""

    def _api(self, *, mask=126336, meta=None):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        values = {"general.architecture": "llada-moe", "diffusion.shift_logits": "false"}
        values.update(meta or {})
        api = MagicMock()
        api.llama_model_default_params.return_value = SimpleNamespace(
            main_gpu=0, n_gpu_layers=0, use_mmap=True)
        api.llama_model_is_diffusion.return_value = True
        api.llama_model_has_encoder.return_value = False
        api.has_diffusion_api.return_value = True
        api.has_model_meta_api.return_value = True
        api.llama_model_meta_val_str.side_effect = lambda model, key: values.get(key)
        api.llama_vocab_mask.return_value = mask
        api.llama_n_ubatch.return_value = 512
        api.llama_model_n_ctx_train.return_value = 4096
        return api

    def _build(self, api, monkeypatch, **kw):
        monkeypatch.setattr(LlamaCpp, "_cache_can_drop_a_speculative_token",
                            lambda self: pytest.fail("a speculation probe ran at load"))
        monkeypatch.setattr(LlamaCpp, "_load_mmproj",
                            lambda self, *a: pytest.fail("a vision projector was loaded"))
        with patch("localm.inference.backends.llamacpp.llama.api", api):
            llm = LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=99, verbose=True, **kw)
            report = llm.speculation_report()
            llm.close()
        return llm, report

    @pytest.mark.parametrize("spec", [dict(spec_source="ngram"), dict(mtp_enabled=True),
                                      dict(spec_source="mtp")])
    def test_no_speculation_is_set_up_and_the_refusal_is_reported(self, monkeypatch, spec):
        api = self._api()
        llm, report = self._build(api, monkeypatch, mmproj_path="proj.gguf", **spec)
        assert llm.is_diffusion is True and llm.architecture == "llada-moe"
        assert llm._spec_source_name == "off" and llm.supports_mtp is False
        assert llm.mtp_status == "diffusion-model"
        assert report["status"] == "diffusion-model"
        api.llama_model_mtp_support.assert_not_called()
        assert llm.kv_bytes_per_token == 0
        assert llm._diffusion_capacity == 512
        assert llm._diffusion_shift_logits is False
        assert llm._diffusion_mask == 126336
        api.llama_set_causal_attn.assert_called_once_with(
            api.llama_init_from_model.return_value, False)

    def test_shift_logits_defaults_to_true_when_undeclared(self, monkeypatch):
        llm, _ = self._build(self._api(meta={"diffusion.shift_logits": None}), monkeypatch)
        assert llm._diffusion_shift_logits is True

    def test_capacity_is_bounded_by_the_trained_context(self, monkeypatch):
        api = self._api()
        api.llama_model_n_ctx_train.return_value = 256
        llm, _ = self._build(api, monkeypatch)
        assert llm._diffusion_capacity == 256

    def test_no_mask_token_is_refused_and_frees_the_model(self, monkeypatch):
        api = self._api(mask=-1)
        with pytest.raises(RuntimeError, match="no mask token"):
            self._build(api, monkeypatch)
        api.llama_free_model.assert_called_once()

    def test_runtime_without_the_diffusion_calls_is_refused(self, monkeypatch):
        api = self._api()
        api.has_diffusion_api.return_value = False
        with pytest.raises(RuntimeError, match="setup-llama"):
            self._build(api, monkeypatch)
        api.llama_free_model.assert_called_once()

    def test_a_mocked_non_bool_answer_is_not_diffusion(self, monkeypatch):
        from unittest.mock import MagicMock
        api = self._api(meta={"general.architecture": "qwen3"})
        api.llama_model_is_diffusion.return_value = MagicMock()
        monkeypatch.setattr(LlamaCpp, "_read_kv_bytes_per_token", lambda self: 4096)
        with patch("localm.inference.backends.llamacpp.llama.api", api):
            llm = LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=99, verbose=True)
            llm.close()
        assert llm.is_diffusion is False and llm.kv_bytes_per_token == 4096

    def test_a_rebuilt_context_stays_bidirectional(self):
        import ctypes
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from tests._bare_llama import make_bare_llama
        api = MagicMock()
        api.llama_context_default_params.return_value = SimpleNamespace(
            n_ctx=0, n_batch=0, n_ubatch=0, offload_kqv=True)
        api.llama_init_from_model.return_value = ctypes.c_void_p(77)
        llm = make_bare_llama(_model_ptr=ctypes.c_void_p(1), _ctx_ptr=ctypes.c_void_p(2),
                              _mtp_enabled=False, is_diffusion=True)
        llm._target_ctx = lambda needed: 1024
        with patch("localm.inference.backends.llamacpp.llama.api", api):
            llm._prefill_fresh_context([], 10)
        api.llama_set_causal_attn.assert_called_once_with(
            api.llama_init_from_model.return_value, False)
        api.llama_set_causal_attn.assert_not_called()


class TestWindowAndReplyReserve:
    def _loaded(self, tmp_path, meta, arch="dream"):
        b = GgufBackend(str(_gguf(tmp_path / f"{arch}.gguf", arch)))
        with patch("localm.discover.list_gpus", return_value=([], "ok")), \
             patch("localm.inference.backends.llamacpp._runner.ModelRunner."
                   "spawn_and_load", return_value=meta):
            b.effective_gpu_layers = 0
            b._load_native()
        return b

    def test_capacity_and_reply_reserve_come_from_the_worker(self, tmp_path):
        from localm.inference.engine import Engine
        meta = {"diffusion": True, "diffusion_capacity": 2048,
                "diffusion_reply_tokens": 256}
        b = self._loaded(tmp_path, meta)
        with patch("localm.inference.backends.llamacpp._runner.ModelRunner.is_alive",
                   return_value=True):
            assert b.effective_ctx_max == 2048
            assert b.reply_reserve == 256
            eng = Engine.__new__(Engine)
            eng._backend = b
            assert eng.context_capacity() == 2048
            assert eng.reply_reserve == 256

    def test_a_worker_reported_diffusion_model_is_diffusion_whatever_the_header(self, tmp_path):
        meta = {"diffusion": True, "diffusion_capacity": 1024, "diffusion_reply_tokens": 128}
        b = self._loaded(tmp_path, meta, arch="future-diffusion")
        assert b.is_diffusion is True
        assert b.supports_grammar is False
        assert b._kv_bytes_per_token() == 0

    def test_autoregressive_model_has_no_reply_reserve(self, tmp_path):
        b = self._loaded(tmp_path, {"diffusion": False}, arch="qwen3")
        with patch("localm.inference.backends.llamacpp._runner.ModelRunner.is_alive",
                   return_value=True):
            assert b.reply_reserve is None

    def test_pre_load_ceiling_is_left_to_the_worker_with_no_auto_line(self, tmp_path, capsys):
        b = GgufBackend(str(_gguf(tmp_path / "m.gguf", "llada")), n_ctx=4096,
                        ctx_auto=True)
        assert b._effective_ctx_max() is None
        assert "ctx auto" not in capsys.readouterr().out


class TestCompactionGate:
    def test_reply_reserve_replaces_the_reply_buffer(self):
        from localm.inference.http_server import _needs_compaction
        msgs = [{"role": "user", "content": "x"}] * 4
        assert _needs_compaction(2048, 1700, msgs, reply_reserve=256) is False
        assert _needs_compaction(2048, 1800, msgs, reply_reserve=256) is True
        assert _needs_compaction(2048, 100, msgs) is True
        assert _needs_compaction(2048, 1800, msgs[:3], reply_reserve=256) is False

    def test_engine_reply_reserve_accepts_only_a_positive_int(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from localm.inference.http_server import _engine_reply_reserve
        assert _engine_reply_reserve(SimpleNamespace(reply_reserve=256)) == 256
        for bad in (None, 0, -1, True, 2.5, MagicMock()):
            assert _engine_reply_reserve(SimpleNamespace(reply_reserve=bad)) is None
        assert _engine_reply_reserve(object()) is None

    def test_overflow_text_names_the_fixed_window(self):
        from localm.inference.http_server import context_overflow_detail
        text = context_overflow_detail(2100, 2048, reply_reserve=256)
        assert "diffusion model" in text and "2048" in text
        assert "n_ctx_max" not in text
        assert "n_ctx_max" in context_overflow_detail(5000, 4096)


class TestMaskTokenRefusal:
    def test_the_worker_tags_it_and_the_parent_raises_the_role_error(self):
        import multiprocessing as mp
        from localm.inference.backends.base import UnsupportedModelRoleError
        from localm.inference.backends.llamacpp._runner import ModelRunner
        ctx = mp.get_context("spawn")
        r = ModelRunner()
        r._spawn = lambda: None
        r._req_q, r._resp_q, r._ctrl_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
        r._proc = _FakeProc()
        r._resp_q.put(("error", "declares no mask token", "UnsupportedModelRoleError"))
        with pytest.raises(UnsupportedModelRoleError, match="no mask token"):
            r.spawn_and_load({"model_path": "m.gguf"}, timeout=5)

    def test_the_load_reports_it_as_is(self, tmp_path, monkeypatch):
        from localm.inference.backends.base import UnsupportedModelRoleError
        b = GgufBackend(str(_gguf(tmp_path / "m.gguf", "dream")))
        monkeypatch.setattr(b, "_effective_gpu_layers", lambda: 0)
        monkeypatch.setattr(b, "_check_vram", lambda: None)

        def refuse():
            raise UnsupportedModelRoleError(
                "This diffusion language model declares no mask token, so it "
                "cannot be run.")
        monkeypatch.setattr(b, "_load_native", refuse)
        with pytest.raises(UnsupportedModelRoleError) as caught:
            b.load()
        assert "setup-llama" not in str(caught.value)

    def test_llama_raises_the_role_error(self, monkeypatch):
        from localm.inference.backends.base import UnsupportedModelRoleError
        setup = TestLoadSetup()
        api = setup._api(mask=-1)
        with pytest.raises(UnsupportedModelRoleError, match="no mask token"):
            setup._build(api, monkeypatch)


class TestVramEstimate:
    def test_no_kv_cache_is_estimated_for_a_diffusion_model(self):
        from localm.sysstats import estimate_vram
        est = estimate_vram(3 << 30, n_ctx=8192, keeps_kv=False)
        assert est["kv_cache"] == 0
        assert estimate_vram(3 << 30, n_ctx=8192)["kv_cache"] > 0


class TestEntropyFastPath:
    def test_large_candidate_sets_sum_in_double_close_to_float32(self):
        import math
        import random
        rng = random.Random(1)
        raw = [rng.random() for _ in range(_diffusion.EXACT_ENTROPY_MAX + 1)]
        total = sum(raw)
        probs = [_diffusion.f32(x / total) for x in raw]
        fast = _diffusion.entropy_confidence(probs)
        exact = 0.0
        for p in probs:
            exact = _diffusion.f32(exact + _diffusion.f32(
                p * _diffusion.f32(math.log(_diffusion.f32(p + _diffusion._ENTROPY_EPS)))))
        assert fast == pytest.approx(-exact, rel=1e-4)
        assert fast == _diffusion.f32(fast)

    def test_large_candidate_sets_are_not_summed_element_by_element(self, monkeypatch):
        calls = []
        real = _diffusion.f32
        monkeypatch.setattr(_diffusion, "f32", lambda x: calls.append(1) or real(x))
        _diffusion.entropy_confidence([1.0 / 5000] * 5000)
        assert len(calls) <= 2

    def test_small_candidate_sets_stay_float32_exact(self):
        probs = [_diffusion.f32(0.5), _diffusion.f32(0.25), _diffusion.f32(0.25)]
        assert _diffusion.entropy_confidence(probs) == _diffusion.f32(1.0397207736968994)


class TestWorkerLoadMeta:
    def test_window_and_reply_length_are_reported(self, monkeypatch):
        from localm.inference.backends.llamacpp import _worker

        class _FakeLlama:
            def __init__(self, **kw):
                self.is_diffusion = True
                self.supports_images = False
                self._diffusion_capacity = 1536
                self._diffusion_max_tokens = kw.get("diffusion_max_tokens")
        monkeypatch.setattr("localm.inference.backends.llamacpp._loader.load_lib", lambda: None)
        monkeypatch.setattr("localm.inference.backends.llamacpp.LlamaCpp", _FakeLlama)
        meta = _worker.GgufWorker("m.gguf", None, 512, 0, None, 512).load()
        assert meta["diffusion_capacity"] == 1536
        assert meta["diffusion_reply_tokens"] == _diffusion.DEFAULT_MAX_TOKENS
        meta = _worker.GgufWorker("m.gguf", None, 512, 0, None, 512,
                                  diffusion_max_tokens=64).load()
        assert meta["diffusion_reply_tokens"] == 64
