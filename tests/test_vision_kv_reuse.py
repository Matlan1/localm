# SPDX-License-Identifier: AGPL-3.0-or-later
"""Image chat turns reuse the KV cache and the encoded image embeddings.

Driven through the real MtmdContext and LlamaCpp code against the fakes in
tests/_fake_mtmd.py: a follow-up turn must not encode an unchanged image again,
must decode only the positions that changed, and must leave exactly the cache a
from-scratch evaluation of the same prompt leaves.
"""
from unittest.mock import MagicMock, patch

import pytest

from localm.inference.backends.base import VisionInputError
from localm.inference.backends.llamacpp import mtmd as lmtmd

from tests._bare_llama import make_bare_llama
from tests._fake_mtmd import (
    MARKER,
    FakeKV,
    FakeLlamaApi,
    FakeMtmdLib,
    fake_create_batch,
    make_mtmd_context,
    solid_image,
    word_token,
)

IMG_A = solid_image(10)
IMG_B = solid_image(200)

TURN1 = f"system be terse user {MARKER} describe this image assistant"
TURN2 = TURN1 + " a red square user what colour is it assistant"
TURN3 = TURN2 + " red user anything else assistant"


class _Rig:
    """One LlamaCpp with a fake KV cache, fake llama api and fake mtmd."""

    def __init__(self, *, n_ctx=4096, mrope=False, **lib_kwargs):
        self.kv = FakeKV()
        self.api = FakeLlamaApi(self.kv, n_ctx=n_ctx, mrope=mrope)
        self.lib = FakeMtmdLib(self.kv, **lib_kwargs)
        self.llm = make_bare_llama(_model_ptr=0x11, _ctx_ptr=0x22, _ctx_capacity=n_ctx,
                                   _n_ctx=n_ctx, _n_ctx_grow=256)
        self.llm._mtmd = make_mtmd_context(self.lib)
        self.llm._create_batch = fake_create_batch

    def prefill(self, prompt, images, *, needed=None, on_status=None):
        vprompt = self.llm._mtmd.tokenize(prompt, images, add_special=True)
        try:
            with patch("localm.inference.backends.llamacpp.llama.api", self.api):
                return self.llm._prefill_vision(
                    vprompt, needed or vprompt.n_tokens + 8, on_status)
        finally:
            vprompt.free()


def _fresh_cells(prompt, images, **lib_kwargs):
    """The cache a brand-new instance leaves after evaluating *prompt*."""
    rig = _Rig(**lib_kwargs)
    rig.prefill(prompt, images)
    return list(rig.kv.cells)


def _suffix_tokens(old_prompt, new_prompt):
    return [word_token(w) for w in new_prompt[len(old_prompt):].split()]


class TestFollowUpTurns:
    def test_a_follow_up_turn_neither_encodes_nor_redecodes_the_image(self):
        rig = _Rig()
        rig.prefill(TURN1, [IMG_A])
        assert rig.lib.encode_calls == 1
        n_turn1 = len(rig.kv.cells)
        rig.api.decoded.clear()

        pos, reused = rig.prefill(TURN2, [IMG_A])

        assert rig.lib.encode_calls == 1, "the unchanged image was encoded again"
        assert len(rig.lib.image_decodes) == 1, "the unchanged image was decoded again"
        assert reused == n_turn1
        assert rig.api.decoded == [(n_turn1, _suffix_tokens(TURN1, TURN2))], (
            "a follow-up turn must decode only the new text, at the end of the cache")
        assert rig.kv.cells == _fresh_cells(TURN2, [IMG_A])
        assert pos == len(rig.kv.cells)

    def test_several_follow_ups_keep_reusing(self):
        rig = _Rig()
        rig.prefill(TURN1, [IMG_A])
        rig.prefill(TURN2, [IMG_A])
        n_turn2 = len(rig.kv.cells)
        rig.api.decoded.clear()

        rig.prefill(TURN3, [IMG_A])

        assert rig.lib.encode_calls == 1
        assert rig.api.decoded == [(n_turn2, _suffix_tokens(TURN2, TURN3))]
        assert rig.kv.cells == _fresh_cells(TURN3, [IMG_A])

    def test_a_changed_image_is_encoded_and_everything_after_it_redecoded(self):
        rig = _Rig()
        rig.prefill(TURN1, [IMG_A])
        rig.prefill(TURN2, [IMG_B])

        assert rig.lib.encode_calls == 2, "a different image must be encoded"
        assert rig.kv.cells == _fresh_cells(TURN2, [IMG_B])

    def test_a_new_image_later_in_the_chat_encodes_only_that_image(self):
        rig = _Rig()
        rig.prefill(TURN1, [IMG_A])
        turn2 = TURN1 + f" fine user {MARKER} and this one assistant"
        rig.prefill(turn2, [IMG_A, IMG_B])

        assert rig.lib.encode_calls == 2
        assert [c for c, _ in rig.lib.image_decodes].count(rig.lib.image_decodes[0][0]) == 1
        assert rig.kv.cells == _fresh_cells(turn2, [IMG_A, IMG_B])

    def test_resending_the_same_prompt_decodes_only_its_last_token(self):
        rig = _Rig()
        rig.prefill(TURN1, [IMG_A])
        n = len(rig.kv.cells)
        rig.api.decoded.clear()

        pos, reused = rig.prefill(TURN1, [IMG_A])

        assert reused == n - 1
        assert rig.api.decoded == [(n - 1, [word_token("assistant")])]
        assert pos == n
        assert rig.kv.cells == _fresh_cells(TURN1, [IMG_A])

    def test_tiled_image_slices_are_told_apart(self):
        rig = _Rig(slices=3)
        rig.prefill(TURN1, [IMG_A])
        assert rig.lib.encode_calls == 3
        rig.llm._cached_tokens = [1, 2, 3]       # a text turn rewrote the cache

        rig.prefill(TURN2, [IMG_A])

        assert rig.lib.encode_calls == 3, "cached slice embeddings were not reused"
        assert rig.kv.cells == _fresh_cells(TURN2, [IMG_A], slices=3), (
            "a slice was decoded with another slice's embeddings")

    def test_status_names_encoding_only_when_an_image_is_encoded(self):
        rig = _Rig()
        statuses = []
        rig.prefill(TURN1, [IMG_A], on_status=statuses.append)
        rig.prefill(TURN2, [IMG_A], on_status=statuses.append)
        rig.llm._mtmd.on_gpu = False
        rig.prefill(TURN2, [IMG_B], on_status=statuses.append)
        assert statuses == ["Encoding image (GPU)...", "Processing prompt...",
                            "Encoding image (CPU)..."]


class TestWhenTheCacheCannotBeReused:
    def test_a_text_turn_drops_the_record_but_not_the_embeddings(self):
        rig = _Rig()
        rig.prefill(TURN1, [IMG_A])
        rig.llm._cached_tokens = [5, 6, 7]
        assert rig.llm._vision_kv is None
        rig.kv.cells = [("text", 5), ("text", 6), ("text", 7)]
        clears = rig.api.clears

        rig.prefill(TURN2, [IMG_A])

        assert rig.api.clears == clears + 1, "the text turn's cache was not wiped"
        assert rig.lib.encode_calls == 1, "the image was encoded again"
        assert rig.kv.cells == _fresh_cells(TURN2, [IMG_A])

    def test_the_text_prefill_drops_the_record_before_touching_the_cache(self):
        rig = _Rig()
        rig.prefill(TURN1, [IMG_A])
        rig.api.llama_decode = MagicMock(side_effect=RuntimeError("decode failed"))
        with patch("localm.inference.backends.llamacpp.llama.api", rig.api), \
             pytest.raises(RuntimeError):
            rig.llm._prefill_with_reuse([1, 2, 3])
        assert rig.llm._vision_kv is None

    def test_growing_the_context_rebuilds_without_encoding(self):
        rig = _Rig(n_ctx=256)
        rig.prefill(TURN1, [IMG_A])
        rig.prefill(TURN2, [IMG_A], needed=400)

        assert rig.api.inits == [512]
        assert rig.lib.encode_calls == 1, "growing the context re-encoded the image"
        assert rig.kv.cells == _fresh_cells(TURN2, [IMG_A])

    def test_an_mrope_model_reevaluates_but_does_not_reencode(self):
        rig = _Rig(mrope=True, image_pos=2)
        rig.prefill(TURN1, [IMG_A])
        clears = rig.api.clears
        rig.prefill(TURN2, [IMG_A])

        assert rig.api.clears == clears + 1
        assert rig.api.seq_rm_calls == [], "partial removal attempted on M-RoPE"
        assert rig.lib.encode_calls == 1
        assert rig.kv.cells == _fresh_cells(TURN2, [IMG_A], mrope=True, image_pos=2)

    def test_a_cache_that_cannot_drop_its_tail_is_rebuilt(self):
        rig = _Rig()
        rig.prefill(TURN1, [IMG_A])
        rig.api.seq_rm_ok = False
        rig.prefill(TURN2, [IMG_A])
        assert rig.api.seq_rm_calls, "partial removal was never tried"
        assert rig.kv.cells == _fresh_cells(TURN2, [IMG_A])

    def test_a_failed_prefill_leaves_no_record(self):
        rig = _Rig()
        rig.prefill(TURN1, [IMG_A])
        rig.lib.fail_decode = True
        with pytest.raises(lmtmd.MtmdGpuEncodeFailed):
            rig.prefill(TURN2, [IMG_B])
        assert rig.llm._vision_kv is None

        rig.lib.fail_decode = False
        rig.prefill(TURN2, [IMG_B])
        assert rig.kv.cells == _fresh_cells(TURN2, [IMG_B])

    def test_closing_the_model_drops_the_record_and_the_embeddings(self):
        rig = _Rig()
        rig.prefill(TURN1, [IMG_A])
        mt = rig.llm._mtmd
        key = next(k for k, _ in rig.llm._vision_kv if isinstance(k, tuple))
        assert mt.has_embedding(key)

        with patch("localm.inference.backends.llamacpp.llama.api", rig.api):
            rig.llm.close()

        assert rig.llm._vision_kv is None
        assert not mt.has_embedding(key)
        assert mt._embd_bytes == 0


class TestBatching:
    @pytest.mark.parametrize("n_ctx, expected", [(1000, 1000), (4096, 2048), (8192, 2048)])
    def test_images_are_decoded_with_the_context_batch_size(self, n_ctx, expected):
        rig = _Rig(n_ctx=n_ctx)
        rig.prefill(TURN1, [IMG_A])
        assert rig.lib.image_n_batch == [expected]

    def test_long_text_is_decoded_in_batch_sized_pieces(self):
        rig = _Rig()
        words = " ".join(f"w{i}" for i in range(2500))
        rig.prefill(f"{MARKER} {words}", [IMG_A])
        assert [(start, len(toks)) for start, toks in rig.api.decoded] == [
            (4, 2048), (4 + 2048, 452)]

    def test_a_prompt_ending_past_the_context_is_refused(self):
        rig = _Rig()
        rig.api.n_ctx = 6
        with pytest.raises(VisionInputError, match="implausible position"):
            rig.prefill(TURN1, [IMG_A])
        assert rig.llm._vision_kv is None


class TestGenerateImageAcrossTurns:
    """Two turns through the real _generate_image, including its decode loop."""

    def _turn(self, rig, prompt, images, sampled):
        rig.api.llama_sampler_sample = MagicMock(side_effect=sampled)
        rig.llm._tokenizer.is_eog.side_effect = lambda t: t == -1
        with patch("localm.inference.backends.llamacpp.llama.api", rig.api), \
             patch("localm.inference.backends.llamacpp.llama._apply_model_template",
                   return_value=(prompt, None)), \
             patch("localm.inference.backends.llamacpp.llama._build_sampler",
                   return_value=MagicMock()), \
             patch.object(type(rig.llm), "_messages_with_markers",
                          return_value=([], images)):
            return list(rig.llm._generate_image(
                [], max_new_tokens=8, temperature=0.0, top_k=40, top_p=0.95,
                repeat_penalty=1.0))

    def test_the_reply_is_reevaluated_and_the_image_is_not(self):
        rig = _Rig()
        rig.api.llama_sampler_free = lambda s: None
        assert self._turn(rig, TURN1, [IMG_A], [7, 8, -1]) == [7, 8]
        n_prompt1 = len(_fresh_cells(TURN1, [IMG_A]))
        assert len(rig.kv.cells) == n_prompt1 + 2, "the reply was not decoded into the cache"
        rig.api.decoded.clear()

        self._turn(rig, TURN2, [IMG_A], [-1])

        assert rig.lib.encode_calls == 1
        assert rig.api.decoded[0][0] == n_prompt1, (
            "the next turn must resume right after the previous prompt")
        assert rig.kv.cells == _fresh_cells(TURN2, [IMG_A])


class TestEmbeddingCache:
    def _chunk(self, ctx, image):
        prompt = ctx.tokenize(f"look {MARKER} here", [image], add_special=True)
        return prompt, next(c for c in prompt.chunks if c.tokens is None)

    def test_a_cached_image_is_decoded_from_the_same_embeddings(self):
        kv = FakeKV()
        lib = FakeMtmdLib(kv)
        ctx = make_mtmd_context(lib)
        p1, c1 = self._chunk(ctx, IMG_A)
        ctx.eval_media_chunk(0x22, c1, 0, 512)
        kv.clear()
        p2, c2 = self._chunk(ctx, IMG_A)
        ctx.eval_media_chunk(0x22, c2, 0, 512)
        p1.free()
        p2.free()

        assert lib.encode_calls == 1
        assert ctx.encode_count == 1
        first = _fresh_image_cells(IMG_A)
        assert kv.cells == first

    def test_a_different_image_misses(self):
        lib = FakeMtmdLib(FakeKV())
        ctx = make_mtmd_context(lib)
        _, ca = self._chunk(ctx, IMG_A)
        _, cb = self._chunk(ctx, IMG_B)
        assert ca.key != cb.key
        ctx.eval_media_chunk(0x22, ca, 0, 512)
        assert not ctx.has_embedding(cb.key)
        ctx.eval_media_chunk(0x22, cb, ca.n_pos, 512)
        assert lib.encode_calls == 2

    def test_the_content_id_follows_pixels_and_size(self):
        w, h, rgb = IMG_A
        assert lmtmd._image_content_id(w, h, rgb) == lmtmd._image_content_id(w, h, bytes(rgb))
        assert lmtmd._image_content_id(w, h, rgb) != lmtmd._image_content_id(h * 2, w // 2, rgb)
        assert lmtmd._image_content_id(*IMG_A) != lmtmd._image_content_id(*IMG_B)

    def test_retain_drops_images_no_longer_in_the_prompt(self):
        lib = FakeMtmdLib(FakeKV())
        ctx = make_mtmd_context(lib)
        _, ca = self._chunk(ctx, IMG_A)
        _, cb = self._chunk(ctx, IMG_B)
        ctx.eval_media_chunk(0x22, ca, 0, 512)
        ctx.eval_media_chunk(0x22, cb, ca.n_pos, 512)
        ctx.retain_embeddings([cb.key])
        assert not ctx.has_embedding(ca.key)
        assert ctx.has_embedding(cb.key)
        assert ctx._embd_bytes == 4 * 3 * 4

    def test_free_and_cpu_retry_empty_the_cache(self):
        for empty in (lambda c: c.free(), lambda c: c.retry_on_cpu()):
            lib = FakeMtmdLib(FakeKV())
            ctx = make_mtmd_context(lib)
            _, ca = self._chunk(ctx, IMG_A)
            ctx.eval_media_chunk(0x22, ca, 0, 512)
            assert ctx.has_embedding(ca.key)
            empty(ctx)
            assert not ctx.has_embedding(ca.key)
            assert ctx._embd_bytes == 0

    def test_least_recently_used_entries_go_first_within_the_budget(self, monkeypatch):
        entry = 4 * 3 * 4
        monkeypatch.setattr(lmtmd, "_EMBD_CACHE_MAX_BYTES", 2 * entry)
        lib = FakeMtmdLib(FakeKV())
        ctx = make_mtmd_context(lib)
        chunks = [self._chunk(ctx, solid_image(v))[1] for v in (1, 2, 3)]
        ctx.eval_media_chunk(0x22, chunks[0], 0, 512)
        ctx.eval_media_chunk(0x22, chunks[1], 4, 512)
        ctx.eval_media_chunk(0x22, chunks[0], 8, 512)     # hit: 0 is now newest
        ctx.eval_media_chunk(0x22, chunks[2], 12, 512)
        assert ctx.has_embedding(chunks[0].key)
        assert not ctx.has_embedding(chunks[1].key)
        assert ctx.has_embedding(chunks[2].key)
        assert ctx._embd_bytes == 2 * entry

    def test_an_entry_larger_than_the_budget_is_not_kept(self, monkeypatch):
        monkeypatch.setattr(lmtmd, "_EMBD_CACHE_MAX_BYTES", 8)
        ctx = make_mtmd_context(FakeMtmdLib(FakeKV()))
        _, ca = self._chunk(ctx, IMG_A)
        ctx.eval_media_chunk(0x22, ca, 0, 512)
        assert not ctx.has_embedding(ca.key)
        assert ctx._embd_bytes == 0

    def test_a_failed_encode_is_retryable_on_gpu_only_and_caches_nothing(self):
        for on_gpu, expected in ((True, lmtmd.MtmdGpuEncodeFailed), (False, VisionInputError)):
            lib = FakeMtmdLib(FakeKV())
            lib.fail_encode = True
            ctx = make_mtmd_context(lib, on_gpu=on_gpu)
            _, ca = self._chunk(ctx, IMG_A)
            exc = None
            try:
                ctx.eval_media_chunk(0x22, ca, 0, 512)
            except VisionInputError as e:
                exc = e
            assert type(exc) is expected
            assert not ctx.has_embedding(ca.key)

    def test_a_decode_reporting_the_wrong_position_is_refused(self):
        lib = FakeMtmdLib(FakeKV())
        lib.pos_skew = 1
        ctx = make_mtmd_context(lib, on_gpu=False)
        _, ca = self._chunk(ctx, IMG_A)
        with pytest.raises(VisionInputError, match="implausible position"):
            ctx.eval_media_chunk(0x22, ca, 0, 512)


class TestTokenize:
    def test_chunks_carry_ids_ordinals_and_tokens(self):
        lib = FakeMtmdLib(FakeKV(), slices=2)
        ctx = make_mtmd_context(lib)
        prompt = ctx.tokenize(f"a b {MARKER} c {MARKER} d", [IMG_A, IMG_A], add_special=True)
        text = [c.tokens for c in prompt.chunks if c.tokens is not None]
        keys = [c.key for c in prompt.chunks if c.tokens is None]
        assert text == [(word_token("a"), word_token("b")), (word_token("c"),),
                        (word_token("d"),)]
        content_id = lmtmd._image_content_id(*IMG_A)
        assert [k[:2] for k in keys] == [(content_id, i) for i in range(4)]
        assert prompt.n_tokens == 4 + 4 * 4
        assert prompt.n_images == 4
        prompt.free()

    def test_free_releases_everything_once(self):
        lib = FakeMtmdLib(FakeKV())
        ctx = make_mtmd_context(lib)
        prompt = ctx.tokenize(f"x {MARKER} {MARKER}", [IMG_A, IMG_B], add_special=True)
        prompt.free()
        prompt.free()
        assert lib.bitmaps_freed == 2
        assert lib.chunk_lists_freed == 1

    def test_a_tokenize_failure_releases_everything(self):
        lib = FakeMtmdLib(FakeKV())
        lib.rc_tokenize = 2
        ctx = make_mtmd_context(lib)
        with pytest.raises(VisionInputError, match="mtmd_tokenize rc=2"):
            ctx.tokenize(f"x {MARKER}", [IMG_A], add_special=True)
        assert lib.bitmaps_freed == 1
        assert lib.chunk_lists_freed == 1


def _fresh_image_cells(image):
    kv = FakeKV()
    lib = FakeMtmdLib(kv)
    ctx = make_mtmd_context(lib)
    prompt = ctx.tokenize(f"look {MARKER} here", [image], add_special=True)
    chunk = next(c for c in prompt.chunks if c.tokens is None)
    ctx.eval_media_chunk(0x22, chunk, 0, 512)
    prompt.free()
    return list(kv.cells)
