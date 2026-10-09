# SPDX-License-Identifier: AGPL-3.0-or-later
"""The diffusion language model step loop (llamacpp/_diffusion.py), driven by a
fake native layer: schedules, transfer counts, selection, block bookkeeping,
shift_logits, cancellation and the request parameters."""

from __future__ import annotations

from typing import Sequence

import pytest

from localm.inference.backends.llamacpp import _diffusion as d

MASK = 99


class FakeNative:
    """Records every decode and sample. ``confidence[(row, call)]`` or
    ``confidence[row]`` gives the confidence returned for a row; the token
    sampled for row r is ``1000 + r``."""

    def __init__(self, confidence=None, decode_code=0):
        self.confidence: dict = confidence or {}
        self.decode_code = decode_code
        self.decodes: list[list[int]] = []
        self.samples: list[tuple[int, int]] = []

    def decode(self, tokens: Sequence[int]) -> int:
        self.decodes.append(list(tokens))
        return self.decode_code

    def sample(self, row: int, algorithm: int, greedy: bool) -> tuple[int, float]:
        step = len(self.decodes) - 1
        self.samples.append((step, row))
        conf = self.confidence.get((row, step), self.confidence.get(row, 0.5))
        return 1000 + row, conf


def _params(**kw) -> d.DiffusionParams:
    base = dict(steps=4, mask_token_id=MASK, max_length=8, seed=0,
                algorithm=d.ALGORITHM_CONFIDENCE, schedule=d.SCHEDULE_TIMESTEP)
    base.update(kw)
    return d.DiffusionParams(**base)


class TestRandomStream:
    def test_mt19937_matches_the_cpp_standard(self):
        rng = d.Mt19937(5489)
        assert [rng() for _ in range(3)] == [3499211612, 581869302, 3890346734]
        rng = d.Mt19937(5489)
        for _ in range(9999):
            rng()
        assert rng() == 4123659995

    def test_uniform_float_rounds_the_draw_to_float32(self):
        class Fixed:
            def __init__(self, v):
                self.v = v

            def __call__(self):
                return self.v
        assert d.uniform_float(Fixed(0)) == 0.0
        assert d.uniform_float(Fixed(2 ** 31)) == 0.5
        assert d.uniform_float(Fixed(2 ** 31 + 1)) == 0.5
        assert d.uniform_float(Fixed(0xFFFFFFFF)) == 1.0 - 2.0 ** -24


class TestTransferCount:
    def test_timestep_counts_are_computed_in_float32(self):
        assert d.transfer_count(1, 8, 7, d.SCHEDULE_TIMESTEP, 0.0) == 0
        assert d.transfer_count(1, 8, 14, d.SCHEDULE_TIMESTEP, 0.0) == 1

    def test_timestep_last_step_takes_every_remaining_mask(self):
        assert d.transfer_count(7, 8, 13, d.SCHEDULE_TIMESTEP, 0.001) == 13

    def test_timestep_first_step_with_eps(self):
        assert d.transfer_count(0, 4, 100, d.SCHEDULE_TIMESTEP, 0.0) == 25

    def test_block_uses_the_per_step_table_then_falls_back(self):
        assert d.transfer_count(1, 4, 9, d.SCHEDULE_BLOCK, 0.0, [3, 2, 2, 2]) == 2
        assert d.transfer_count(2, 4, 9, d.SCHEDULE_BLOCK, 0.0, []) == 4

    def test_num_transfer_tokens_spreads_the_remainder_first(self):
        assert d.num_transfer_tokens(10, 4) == [3, 3, 2, 2]
        assert d.num_transfer_tokens(3, 4) == [1, 1, 1, 0]
        assert sum(d.num_transfer_tokens(37, 8)) == 37


class TestDenoise:
    def test_fills_every_mask_and_keeps_the_prompt(self):
        native = FakeNative()
        canvas = d.denoise(native, [1, 2, 3], _params())
        assert canvas[:3] == [1, 2, 3]
        assert MASK not in canvas
        assert canvas[3:] == [1003, 1004, 1005, 1006, 1007]

    @staticmethod
    def _filled_sets(confidence):
        seen = []
        final = d.denoise(FakeNative(confidence=confidence), [1, 2, 3], _params(steps=5),
                          on_step=lambda s, t, c: seen.append(list(c)) or True)
        seen.append(final)
        return [{i for i in range(3, 8) if c[i] != MASK} for c in seen]

    def test_higher_confidence_positions_are_filled_first(self):
        order = [6, 4, 5, 7, 3]
        filled = self._filled_sets({3: 0.1, 4: 0.9, 5: 0.5, 6: 0.95, 7: 0.2})
        assert [len(f) for f in filled] == [0, 0, 1, 2, 3, 5]
        for f in filled:
            assert f == set(order[:len(f)])

    def test_equal_confidence_fills_the_lowest_position_first(self):
        for f in self._filled_sets({}):
            assert f == set(range(3, 3 + len(f)))

    def test_shift_logits_reads_the_previous_row(self):
        native = FakeNative()
        canvas = d.denoise(native, [1, 2, 3], _params(shift_logits=True))
        assert canvas[3:] == [1002, 1003, 1004, 1005, 1006]
        assert all(row == pos - 1 for (_step, row), pos in
                   zip(native.samples[:5], range(3, 8), strict=True))

    def test_logit_row(self):
        assert d.logit_row(0, True) == 0
        assert d.logit_row(5, True) == 4
        assert d.logit_row(5, False) == 5

    def test_block_schedule_samples_only_inside_the_current_block(self):
        native = FakeNative()
        params = _params(max_length=8, schedule=d.SCHEDULE_BLOCK, block_length=4,
                         steps=4)
        canvas = d.denoise(native, [1, 2], params)
        assert MASK not in canvas
        first_block_rows = {row for step, row in native.samples if step < 2}
        assert first_block_rows <= {2, 3, 4, 5}
        later_rows = {row for step, row in native.samples if step >= 2}
        assert later_rows <= {6, 7}

    def test_a_block_with_no_masks_is_not_decoded(self):
        native = FakeNative()
        params = _params(max_length=8, schedule=d.SCHEDULE_BLOCK, block_length=4,
                         steps=4)
        steps = []
        canvas = d.denoise(native, [1, 2, 3, 4, 5, 6], params,
                           on_step=lambda s, t, c: steps.append(s) or True)
        assert MASK not in canvas
        assert len(native.decodes) == 2
        assert steps == [0, 1, 2]

    def test_random_algorithm_orders_by_the_seeded_stream(self):
        draws_a = d.denoise(FakeNative(), [1], _params(algorithm=d.ALGORITHM_RANDOM,
                                                       steps=7, seed=1))
        draws_b = d.denoise(FakeNative(), [1], _params(algorithm=d.ALGORITHM_RANDOM,
                                                       steps=7, seed=1))
        assert draws_a == draws_b
        seen_1, seen_2 = [], []
        d.denoise(FakeNative(), [1], _params(algorithm=d.ALGORITHM_RANDOM, steps=7, seed=1),
                  on_step=lambda s, t, c: seen_1.append(list(c)) or True)
        d.denoise(FakeNative(), [1], _params(algorithm=d.ALGORITHM_RANDOM, steps=7, seed=2),
                  on_step=lambda s, t, c: seen_2.append(list(c)) or True)
        assert seen_1 != seen_2

    def test_origin_algorithm_transfers_by_uniform_draws(self):
        native = FakeNative()
        canvas = d.denoise(native, [1, 2, 3], _params(algorithm=d.ALGORITHM_ORIGIN))
        assert MASK not in canvas

    def test_on_step_false_abandons_the_run(self):
        native = FakeNative()
        out = d.denoise(native, [1, 2, 3], _params(),
                        on_step=lambda s, t, c: s < 1)
        assert out is None
        assert len(native.decodes) == 1

    def test_failed_decode_raises_with_the_step_and_code(self):
        with pytest.raises(d.DiffusionDecodeError) as caught:
            d.denoise(FakeNative(decode_code=-3), [1, 2, 3], _params())
        assert caught.value.step == 0 and caught.value.code == -3

    @pytest.mark.parametrize("kw,msg", [
        (dict(mask_token_id=d.LLAMA_TOKEN_NULL), "mask token"),
        (dict(max_length=3), "no room"),
        (dict(steps=0), "steps"),
        (dict(algorithm=7), "algorithm"),
        (dict(schedule=d.SCHEDULE_BLOCK, block_length=3), "multiple of block_length"),
        (dict(schedule=d.SCHEDULE_BLOCK, block_length=4, steps=3), "multiple of the"),
        (dict(schedule=d.SCHEDULE_BLOCK, block_length=0), "positive block_length"),
    ])
    def test_invalid_parameters_are_refused_before_any_native_call(self, kw, msg):
        native = FakeNative()
        with pytest.raises(d.DiffusionConfigError, match=msg):
            d.denoise(native, [1, 2, 3], _params(**kw))
        assert native.decodes == []


class TestReplyTokens:
    def test_reply_ends_at_the_first_end_token(self):
        out, ended = d.reply_tokens([1, 2, 7, 8, 0, 9], 2, MASK, lambda t: t == 0)
        assert out == [7, 8] and ended is True

    def test_no_end_token_is_not_ended(self):
        assert d.reply_tokens([1, 7, 8], 1, MASK, lambda t: False) == ([7, 8], False)

    def test_a_leftover_mask_ends_the_reply_unfinished(self):
        assert d.reply_tokens([1, 7, MASK, 8], 1, MASK, lambda t: False) == ([7], False)


class TestResolveParams:
    def _resolve(self, **kw):
        base = dict(architecture="dream", n_input=50, max_tokens=4096,
                    canvas_tokens=None, steps=None, capacity=2048,
                    mask_token_id=MASK, shift_logits=True, temperature=0.7,
                    top_k=40, top_p=0.95, seed=7)
        base.update(kw)
        return d.resolve_params(**base)

    def test_dream_uses_the_timestep_schedule_with_entropy(self):
        p = self._resolve()
        assert p.schedule == d.SCHEDULE_TIMESTEP and p.eps == pytest.approx(0.001)
        assert p.algorithm == d.ALGORITHM_ENTROPY
        assert p.max_length == 50 + d.DEFAULT_MAX_TOKENS
        assert p.steps == d.DEFAULT_STEPS
        p.validate(50)

    def test_request_max_tokens_caps_the_canvas(self):
        p = self._resolve(max_tokens=40)
        assert p.max_length == 90 and p.steps == 40

    def test_explicit_steps_are_kept(self):
        assert self._resolve(steps=300).steps == 300

    def test_capacity_cuts_the_canvas(self):
        p = self._resolve(capacity=100)
        assert p.max_length == 100

    def test_too_little_room_is_refused(self):
        with pytest.raises(d.DiffusionConfigError, match="diffusion window"):
            self._resolve(n_input=2040, capacity=2048)

    def test_a_short_requested_reply_is_not_refused(self):
        p = self._resolve(max_tokens=2)
        assert p.max_length == 52
        p.validate(50)

    def test_block_rounding_that_leaves_no_room_is_refused(self):
        with pytest.raises(d.DiffusionConfigError, match="diffusion window"):
            self._resolve(architecture="llada", n_input=300, capacity=319)

    @pytest.mark.parametrize("arch", ["llada", "llada-moe"])
    def test_llada_canvas_is_whole_blocks_and_steps_divide(self, arch):
        p = self._resolve(architecture=arch)
        assert p.schedule == d.SCHEDULE_BLOCK and p.block_length == 32
        assert p.algorithm == d.ALGORITHM_CONFIDENCE
        assert p.max_length % 32 == 0
        p.validate(50)
        assert d.reply_steps(p, 50) >= d.DEFAULT_STEPS

    def test_llada_canvas_rounds_down_at_capacity(self):
        p = self._resolve(architecture="llada", capacity=300)
        assert p.max_length == 288
        p.validate(50)

    def test_zero_temperature_is_greedy(self):
        p = self._resolve(temperature=0.0)
        assert p.greedy is True and p.temperature == 0.0
        assert self._resolve(temperature=0.5).greedy is False

    def test_every_diffusion_architecture_has_defaults(self):
        from localm.model_manager.gguf import GGUF_DIFFUSION_ARCHITECTURES
        assert set(d._ARCH_DEFAULTS) == set(GGUF_DIFFUSION_ARCHITECTURES)


class TestConfidence:
    def test_entropy_follows_the_upstream_sign_in_float32(self):
        probs = [d.f32(0.5), d.f32(0.25), d.f32(0.25)]
        value = d.entropy_confidence(probs)
        assert value == d.f32(value)
        assert value == pytest.approx(1.0397, abs=1e-3)

    def test_flatter_distribution_ranks_first(self):
        flat = d.entropy_confidence([d.f32(0.5), d.f32(0.5)])
        peaked = d.entropy_confidence([d.f32(0.98), d.f32(0.02)])
        assert flat > peaked
        assert d.entropy_confidence([d.f32(1.0)]) == pytest.approx(0.0, abs=1e-6)
