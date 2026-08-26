# SPDX-License-Identifier: AGPL-3.0-or-later
"""GGUF vision plumbing that is testable WITHOUT a model and GPU: the image ->
marker message rewrite, and that a text-only GGUF (no mmproj) still refuses
images while one with an mmproj advertises it is worth loading."""

import base64
import io

import pytest

from localm.inference.backends.base import VisionInputError

# Pillow ships only with the [gpu] extra, which CI does not install (the ci.yml
# Tests step uses [dev,rag] only), so skip cleanly when it is absent rather than
# breaking collection.
Image = pytest.importorskip("PIL.Image")


def _data_url(w=8, h=8):
    im = Image.new("RGB", (w, h), (10, 20, 30))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_messages_with_markers_extracts_images_and_inserts_marker():
    from localm.inference.backends.llamacpp.llama import LlamaCpp
    msgs = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _data_url()}},
            {"type": "text", "text": "what is this?"},
        ]},
    ]
    text_msgs, images = LlamaCpp._messages_with_markers(msgs, "<MARK>")
    assert len(images) == 1
    w, h, rgb = images[0]
    assert (w, h) == (8, 8)
    assert len(rgb) == 8 * 8 * 3                 # raw RGB, 3 bytes/pixel
    assert text_msgs[0]["content"] == "be terse"  # non-image message untouched
    assert "<MARK>" in text_msgs[1]["content"]    # marker spliced in where the image was
    assert "what is this?" in text_msgs[1]["content"]


def test_two_images_yield_two_markers_in_order():
    from localm.inference.backends.llamacpp.llama import LlamaCpp
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _data_url(4, 4)}},
        {"type": "image_url", "image_url": {"url": _data_url(6, 6)}},
        {"type": "text", "text": "compare"},
    ]}]
    text_msgs, images = LlamaCpp._messages_with_markers(msgs, "<M>")
    assert [(w, h) for (w, h, _) in images] == [(4, 4), (6, 6)]
    assert text_msgs[0]["content"].count("<M>") == 2


def test_gguf_without_mmproj_is_text_only_and_refuses_images():
    from localm.inference.backends.base import UnsupportedInputError
    from localm.inference.backends.gguf import GgufBackend
    be = GgufBackend("/fake/model.gguf")          # no mmproj, not loaded
    assert be.can_be_multimodal is False
    assert be.supports_images is False
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _data_url()}}]}]
    with pytest.raises(UnsupportedInputError):
        list(be.chat_stream(msgs))


def test_gguf_with_mmproj_advertises_can_be_multimodal():
    from localm.inference.backends.gguf import GgufBackend
    be = GgufBackend("/fake/model.gguf", mmproj_path="/fake/mmproj.gguf")
    assert be.can_be_multimodal is True           # worth loading to check vision
    assert be.supports_images is False            # not loaded yet -> not confirmed


class _FakeMtmdLib:
    """A stand-in for the loaded mtmd.dll's bound functions, controllable per
    test - real enough to drive MtmdContext.eval_into()'s Python-side logic
    (n_batch derivation, the new_n_past sanity check) without a real native
    library or GPU."""

    def __init__(self, new_n_past: int, rc_tokenize: int = 0, rc_eval: int = 0):
        self._new_n_past = new_n_past
        self._rc_tokenize = rc_tokenize
        self._rc_eval = rc_eval
        self.last_eval_n_batch = None

    def mtmd_bitmap_init(self, w, h, rgb):
        return 1  # any truthy "pointer"

    def mtmd_bitmap_free(self, bmp):
        pass

    def mtmd_input_chunks_init(self):
        return 1

    def mtmd_input_chunks_free(self, chunks):
        pass

    def mtmd_tokenize(self, ctx, chunks, itext, arr, n_bitmaps):
        return self._rc_tokenize

    def mtmd_helper_eval_chunks(self, ctx, llama_ctx, chunks, a, b, n_batch, c, out_ptr):
        self.last_eval_n_batch = n_batch
        out_ptr._obj.value = self._new_n_past
        return self._rc_eval


def _fake_mtmd_context(new_n_past: int, **kwargs):
    """A real MtmdContext instance with its native-loading __init__ bypassed
    (no DLL/GPU needed), _m swapped for a controllable fake."""
    from localm.inference.backends.llamacpp.mtmd import MtmdContext
    ctx = MtmdContext.__new__(MtmdContext)
    ctx._m = _FakeMtmdLib(new_n_past, **kwargs)
    ctx._ctx = 1
    return ctx


class TestMtmdEvalIntoSanity:
    """eval_into() must derive n_batch from the LIVE context (not a hardcoded
    512) and must refuse an implausible new_n_past rather than silently hand a
    likely-corrupted KV position to the generation loop.

    The refusal is a VisionInputError (a ValueError), not a RuntimeError:
    _runner.py treats an escaping non-grammar exception as a native fault, which
    kills the whole gguf worker process and evicts the model. Every failure
    eval_into reports is a checked status code from a native call that returned
    normally, so it stays a per-request refusal."""

    def test_default_n_batch_derives_from_real_context_size(self, monkeypatch):
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp.mtmd.api.llama_n_ctx",
            lambda llama_ctx: 4096)
        ctx = _fake_mtmd_context(new_n_past=10)
        ctx.eval_into(llama_ctx=1, prompt="hi", images=[], add_special=True)
        assert ctx._m.last_eval_n_batch == 2048   # min(4096, 2048)

    def test_default_n_batch_capped_at_2048_for_a_larger_context(self, monkeypatch):
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp.mtmd.api.llama_n_ctx",
            lambda llama_ctx: 8192)
        ctx = _fake_mtmd_context(new_n_past=10)
        ctx.eval_into(llama_ctx=1, prompt="hi", images=[], add_special=True)
        assert ctx._m.last_eval_n_batch == 2048

    def test_explicit_n_batch_overrides_the_derived_default(self, monkeypatch):
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp.mtmd.api.llama_n_ctx",
            lambda llama_ctx: 4096)
        ctx = _fake_mtmd_context(new_n_past=10)
        ctx.eval_into(llama_ctx=1, prompt="hi", images=[], add_special=True, n_batch=128)
        assert ctx._m.last_eval_n_batch == 128

    def test_a_zero_or_negative_new_n_past_is_refused(self, monkeypatch):
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp.mtmd.api.llama_n_ctx",
            lambda llama_ctx: 4096)
        ctx = _fake_mtmd_context(new_n_past=0)
        with pytest.raises(VisionInputError, match="implausible position"):
            ctx.eval_into(llama_ctx=1, prompt="hi", images=[], add_special=True)

    def test_a_new_n_past_beyond_the_context_size_is_refused(self, monkeypatch):
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp.mtmd.api.llama_n_ctx",
            lambda llama_ctx: 4096)
        ctx = _fake_mtmd_context(new_n_past=999_999)   # far beyond a 4096 context
        with pytest.raises(VisionInputError, match="implausible position"):
            ctx.eval_into(llama_ctx=1, prompt="hi", images=[], add_special=True)

    def test_a_plausible_new_n_past_is_accepted(self, monkeypatch):
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp.mtmd.api.llama_n_ctx",
            lambda llama_ctx: 4096)
        ctx = _fake_mtmd_context(new_n_past=120)
        pos = ctx.eval_into(llama_ctx=1, prompt="hi", images=[], add_special=True)
        assert pos == 120
