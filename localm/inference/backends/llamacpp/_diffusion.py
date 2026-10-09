# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Text generation for diffusion language models (Dream, LLaDA, LLaDA-MoE, RND1).

A diffusion model does not decode one token after another. The reply is a
canvas of mask tokens appended to the prompt; every step decodes the whole
canvas with bidirectional attention and replaces some masks with sampled
tokens, until none are left.

The step loop is a port of ``diffusion_generate`` from llama.cpp's
``examples/diffusion/diffusion.cpp``: the same schedules, selection
algorithms, tie-breaking, float32 transfer-count arithmetic and random number
stream, so a run with a fixed seed reproduces the upstream example token for
token, with one deliberate difference: the ENTROPY algorithm ranks positions
by negative entropy, most certain first, as Dream's reference sampler does
(the upstream example negates it). Upstream's classifier-free guidance,
``alg_temp`` and gumbel-noise branches are not ported
(``llama-diffusion-cli`` never sets the first two).
``DiffusionParams.greedy`` is a localm addition: the highest-probability
candidate is taken instead of a sampled one.

The loop talks to the model only through :class:`NativeCanvas` (or any object
with the same ``decode``/``sample`` methods), so it runs against a fake in
tests.
"""

from __future__ import annotations

import ctypes
import math
import struct
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, Sequence, Tuple

ALGORITHM_ORIGIN = 0
ALGORITHM_ENTROPY = 1
ALGORITHM_MARGIN = 2
ALGORITHM_RANDOM = 3
ALGORITHM_CONFIDENCE = 4
ALGORITHMS = (ALGORITHM_ORIGIN, ALGORITHM_ENTROPY, ALGORITHM_MARGIN,
              ALGORITHM_RANDOM, ALGORITHM_CONFIDENCE)

SCHEDULE_TIMESTEP = 0
SCHEDULE_BLOCK = 1

LLAMA_TOKEN_NULL = -1

# Steps for a whole reply when neither the config nor the request names a count.
DEFAULT_STEPS = 128
# Reply canvas length, in tokens, when the config does not set one.
DEFAULT_MAX_TOKENS = 256
# The smallest reply canvas a request may be cut down to before it is refused.
MIN_CANVAS = 16

# Per-architecture schedule and selection defaults: (schedule, eps,
# block_length, algorithm).
_ARCH_DEFAULTS = {
    "dream": (SCHEDULE_TIMESTEP, 0.001, 0, ALGORITHM_ENTROPY),
    "rnd1": (SCHEDULE_TIMESTEP, 0.001, 0, ALGORITHM_ENTROPY),
    "llada": (SCHEDULE_BLOCK, 0.0, 32, ALGORITHM_CONFIDENCE),
    "llada-moe": (SCHEDULE_BLOCK, 0.0, 32, ALGORITHM_CONFIDENCE),
}


class DiffusionConfigError(ValueError):
    """A parameter combination the step loop cannot run (each one is a native
    assertion, i.e. a process abort, in the upstream example)."""


class DiffusionDecodeError(RuntimeError):
    """``llama_decode`` returned non-zero during a denoising step."""

    def __init__(self, step: int, code: int) -> None:
        super().__init__(
            f"diffusion step {step} failed: llama_decode returned {code}")
        self.step = step
        self.code = code


@dataclass
class DiffusionParams:
    """One generation's settings. ``max_length`` counts prompt plus canvas."""

    steps: int
    mask_token_id: int
    max_length: int
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int = 0
    shift_logits: bool = False
    algorithm: int = ALGORITHM_CONFIDENCE
    schedule: int = SCHEDULE_TIMESTEP
    eps: float = 0.0
    block_length: int = 0
    greedy: bool = False

    def validate(self, n_input: int) -> None:
        """Raise :class:`DiffusionConfigError` for any combination the loop
        cannot run. Never touches native state."""
        if self.mask_token_id == LLAMA_TOKEN_NULL:
            raise DiffusionConfigError("the model declares no mask token")
        if n_input <= 0:
            raise DiffusionConfigError("the prompt is empty")
        if self.max_length <= n_input:
            raise DiffusionConfigError(
                f"canvas length {self.max_length} leaves no room after a "
                f"{n_input}-token prompt")
        if self.steps <= 0:
            raise DiffusionConfigError(f"steps must be positive, got {self.steps}")
        if self.algorithm not in ALGORITHMS:
            raise DiffusionConfigError(f"unknown algorithm {self.algorithm}")
        if self.schedule == SCHEDULE_BLOCK:
            if self.block_length <= 0:
                raise DiffusionConfigError("block schedule needs a positive block_length")
            if self.max_length % self.block_length:
                raise DiffusionConfigError(
                    f"canvas length {self.max_length} is not a multiple of "
                    f"block_length {self.block_length}")
            if self.steps % (self.max_length // self.block_length):
                raise DiffusionConfigError(
                    f"steps {self.steps} is not a multiple of the "
                    f"{self.max_length // self.block_length} blocks")
        elif self.schedule != SCHEDULE_TIMESTEP:
            raise DiffusionConfigError(f"unknown schedule {self.schedule}")


# --------------------------------------------------------------------------- #
#  float32 arithmetic and the C++ random number stream                         #
# --------------------------------------------------------------------------- #

_F32 = struct.Struct("<f")


def f32(x: float) -> float:
    """*x* rounded to the nearest IEEE float32, as a Python float."""
    return _F32.unpack(_F32.pack(x))[0]


class Mt19937:
    """``std::mt19937`` seeded with one 32-bit value (``init_genrand``)."""

    _N = 624
    _M = 397

    def __init__(self, seed: int) -> None:
        mt = [0] * self._N
        mt[0] = seed & 0xFFFFFFFF
        for i in range(1, self._N):
            prev = mt[i - 1]
            mt[i] = (1812433253 * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF
        self._mt = mt
        self._index = self._N

    def _twist(self) -> None:
        mt = self._mt
        n, m = self._N, self._M
        for i in range(n):
            y = (mt[i] & 0x80000000) | (mt[(i + 1) % n] & 0x7FFFFFFF)
            v = mt[(i + m) % n] ^ (y >> 1)
            if y & 1:
                v ^= 0x9908B0DF
            mt[i] = v
        self._index = 0

    def __call__(self) -> int:
        if self._index >= self._N:
            self._twist()
        y = self._mt[self._index]
        self._index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF


_ONE_BELOW_F32 = 1.0 - 2.0 ** -24


def uniform_float(rng: Mt19937) -> float:
    """``std::uniform_real_distribution<float>(0, 1)(rng)``: one draw, the
    draw rounded to float32 and scaled by 2**-32, kept below 1.0."""
    value = f32(float(rng())) / 4294967296.0
    return value if value < 1.0 else _ONE_BELOW_F32


# --------------------------------------------------------------------------- #
#  Schedules                                                                   #
# --------------------------------------------------------------------------- #

def num_transfer_tokens(mask_count: int, steps: int) -> List[int]:
    """Masks to fill at each of *steps* steps of one block: an even split, the
    remainder going to the first steps."""
    base, remainder = divmod(mask_count, steps)
    return [base + (1 if i < remainder else 0) for i in range(steps)]


def transfer_count(step: int, total_steps: int, remaining_masked: int,
                   schedule: int, eps: float,
                   per_step: Sequence[int] = ()) -> int:
    """How many of *remaining_masked* masks step *step* fills, computed in
    float32 like the upstream example."""
    if schedule == SCHEDULE_TIMESTEP:
        span = f32(1.0 - f32(eps))
        t = f32(1.0 - f32(f32(f32(step) / f32(total_steps)) * span))
        s = f32(1.0 - f32(f32(f32(step + 1) / f32(total_steps)) * span))
        p_transfer = f32(1.0 - f32(s / t)) if step < total_steps - 1 else 1.0
        return int(f32(f32(remaining_masked) * p_transfer))
    if per_step and step < len(per_step):
        return per_step[step]
    return remaining_masked // (total_steps - step)


# --------------------------------------------------------------------------- #
#  The step loop                                                               #
# --------------------------------------------------------------------------- #

class Native(Protocol):
    """What the step loop needs from the model."""

    def decode(self, tokens: Sequence[int]) -> int:
        """Decode the whole canvas; 0 on success, the native code otherwise."""

    def sample(self, row: int, algorithm: int, greedy: bool) -> Tuple[int, float]:
        """Sample a token from logit row *row* of the last decode and return it
        with its confidence under *algorithm* (any value for RANDOM and
        ORIGIN, which do not use it)."""


StepCallback = Callable[[int, int, List[int]], bool]


def logit_row(pos: int, shift_logits: bool) -> int:
    """The logit row that predicts canvas position *pos*."""
    if shift_logits:
        return 0 if pos == 0 else pos - 1
    return pos


def denoise(native: Native, input_tokens: Sequence[int], params: DiffusionParams,
            on_step: Optional[StepCallback] = None) -> Optional[List[int]]:
    """Run the denoising loop and return the full canvas (prompt included).

    *on_step* is called at the start of every step with ``(step, total_steps,
    canvas)``; a False return abandons the run and this returns None.

    Raises :class:`DiffusionConfigError` before any native call when *params*
    cannot run, and :class:`DiffusionDecodeError` when a decode fails."""
    n_input = len(input_tokens)
    params.validate(n_input)
    length = params.max_length
    mask = params.mask_token_id
    canvas = list(input_tokens) + [mask] * (length - n_input)
    rng = Mt19937(params.seed & 0xFFFFFFFF)
    block = params.schedule == SCHEDULE_BLOCK

    num_blocks = length // params.block_length if block else 1
    steps_per_block = params.steps // num_blocks

    for block_num in range(num_blocks):
        if block:
            block_start = n_input + block_num * params.block_length
            block_end = min(n_input + (block_num + 1) * params.block_length, length)
            per_step = num_transfer_tokens(
                sum(1 for t in canvas[block_start:block_end] if t == mask),
                steps_per_block)
        else:
            block_start, block_end = 0, length
            per_step = []

        for step in range(steps_per_block):
            global_step = block_num * steps_per_block + step
            if on_step is not None and not on_step(global_step, params.steps, canvas):
                return None

            positions = [i for i in range(block_start, block_end) if canvas[i] == mask]
            if not positions:
                break

            code = native.decode(canvas)
            if code != 0:
                raise DiffusionDecodeError(global_step, code)

            count = transfer_count(step, steps_per_block, len(positions),
                                   params.schedule, params.eps, per_step)

            if params.algorithm == ALGORITHM_ORIGIN:
                p_transfer = f32(count / len(positions))
                for pos in positions:
                    if uniform_float(rng) < p_transfer:
                        canvas[pos] = native.sample(
                            logit_row(pos, params.shift_logits),
                            params.algorithm, params.greedy)[0]
                continue

            sampled: List[int] = []
            ranked: List[Tuple[float, int]] = []
            for i, pos in enumerate(positions):
                token, confidence = native.sample(
                    logit_row(pos, params.shift_logits),
                    params.algorithm, params.greedy)
                if params.algorithm == ALGORITHM_RANDOM:
                    confidence = uniform_float(rng)
                sampled.append(token)
                ranked.append((-confidence, i))

            if count > 0:
                ranked.sort()
                for _neg, i in ranked[:count]:
                    canvas[positions[i]] = sampled[i]

    return canvas


def reply_tokens(canvas: Sequence[int], n_input: int, mask_token_id: int,
                 is_eog: Callable[[int], bool]) -> Tuple[List[int], bool]:
    """The reply part of a finished *canvas*: the tokens after the prompt up to
    the first end-of-generation token, and whether one was found. A mask token
    left in the canvas ends the reply too and counts as not found."""
    out: List[int] = []
    for token in canvas[n_input:]:
        if token == mask_token_id:
            return out, False
        if is_eog(token):
            return out, True
        out.append(token)
    return out, False


# --------------------------------------------------------------------------- #
#  Parameters for one chat request                                             #
# --------------------------------------------------------------------------- #

def arch_defaults(architecture: Optional[str]) -> Tuple[int, float, int, int]:
    """``(schedule, eps, block_length, algorithm)`` for *architecture*."""
    return _ARCH_DEFAULTS.get(architecture or "", _ARCH_DEFAULTS["dream"])


def resolve_params(*, architecture: Optional[str], n_input: int, max_tokens: int,
                   canvas_tokens: Optional[int], steps: Optional[int],
                   capacity: int, mask_token_id: int, shift_logits: bool,
                   temperature: float, top_k: int, top_p: float,
                   seed: int) -> DiffusionParams:
    """Settings for one chat reply.

    The reply canvas is ``min(canvas_tokens or DEFAULT_MAX_TOKENS,
    max_tokens)`` tokens, cut down to what *capacity* (the most tokens one
    decode may hold) leaves after the prompt; under a block schedule it is
    then rounded so the whole canvas is a whole number of blocks, and the step
    count is spread evenly over the blocks the reply covers (rounded up to a
    whole number of steps per block). *steps* defaults to the smaller of
    :data:`DEFAULT_STEPS` and the canvas length. A *temperature* of 0 or below
    selects greedy sampling.

    Raises :class:`DiffusionConfigError` when fewer than
    ``min(MIN_CANVAS, requested reply length)`` tokens fit after the prompt."""
    schedule, eps, block_length, algorithm = arch_defaults(architecture)
    wanted = min(canvas_tokens or DEFAULT_MAX_TOKENS, max(1, max_tokens))
    room = capacity - n_input
    canvas = min(wanted, room)
    if schedule == SCHEDULE_BLOCK:
        length = ((n_input + canvas + block_length - 1) // block_length) * block_length
        if length > capacity:
            length = (capacity // block_length) * block_length
        canvas = length - n_input
    needed = min(MIN_CANVAS, wanted)
    if canvas < needed:
        raise DiffusionConfigError(
            f"the prompt ({n_input} tokens) leaves {max(canvas, 0)} of this model's "
            f"{capacity}-token diffusion window for the reply; at least "
            f"{needed} are needed")
    total = int(steps) if steps and steps > 0 else min(DEFAULT_STEPS, canvas)
    length = n_input + canvas
    if schedule == SCHEDULE_BLOCK:
        num_blocks = length // block_length
        reply_blocks = -(-canvas // block_length)
        total = max(1, -(-total // reply_blocks)) * num_blocks
    greedy = temperature <= 0.0
    return DiffusionParams(
        steps=total, mask_token_id=mask_token_id, max_length=length,
        temperature=0.0 if greedy else float(temperature),
        top_k=int(top_k), top_p=float(top_p), seed=int(seed) & 0xFFFFFFFF,
        shift_logits=shift_logits, algorithm=algorithm, schedule=schedule,
        eps=eps, block_length=block_length, greedy=greedy)


def reply_steps(params: DiffusionParams, n_input: int) -> int:
    """The steps that can fill a reply token: under a block schedule, blocks
    that start past the canvas end are skipped at once."""
    if params.schedule != SCHEDULE_BLOCK:
        return params.steps
    num_blocks = params.max_length // params.block_length
    reply_blocks = -(-(params.max_length - n_input) // params.block_length)
    return (params.steps // num_blocks) * reply_blocks


# --------------------------------------------------------------------------- #
#  Native side                                                                 #
# --------------------------------------------------------------------------- #

class NativeCanvas:
    """The ``Native`` the step loop uses on a real context: one batch holding
    the whole canvas, one sampler chain built like the upstream example's
    (top_k when > 0, top_p when < 1, temperature when > 0, then dist), and a
    reusable candidate array.

    *guard* is a context manager entered around every native call; it raises
    to stop the run (the model is being unloaded). Call :meth:`close` when
    done. Not thread-safe."""

    def __init__(self, api, ctx, vocab, n_vocab: int, params: DiffusionParams,
                 guard) -> None:
        from ._structs import LlamaTokenData, LlamaTokenDataArray, llama_token
        self._api = api
        self._ctx = ctx
        self._guard = guard
        self._n_vocab = n_vocab
        self._length = params.max_length
        self._batch = None
        self._chain = None
        try:
            self._setup(params, LlamaTokenData, LlamaTokenDataArray, llama_token)
        except BaseException:
            self.close()
            raise

    def _setup(self, params, LlamaTokenData, LlamaTokenDataArray, llama_token) -> None:
        api = self._api
        n_vocab = self._n_vocab
        chain_params = api.llama_sampler_chain_default_params()
        chain_params.no_perf = True
        self._chain = api.llama_sampler_chain_init(chain_params)
        if params.top_k > 0:
            api.llama_sampler_chain_add(self._chain, api.llama_sampler_init_top_k(params.top_k))
        if params.top_p < 1.0:
            api.llama_sampler_chain_add(self._chain, api.llama_sampler_init_top_p(params.top_p, 1))
        if params.temperature > 0.0:
            api.llama_sampler_chain_add(self._chain, api.llama_sampler_init_temp(params.temperature))
        api.llama_sampler_chain_add(self._chain, api.llama_sampler_init_dist(params.seed))

        n = self._length
        self._batch = api.llama_batch_init(n, 0, 1)
        self._batch.n_tokens = n
        pos = ctypes.cast(self._batch.pos, ctypes.POINTER(ctypes.c_int32))
        n_seq = ctypes.cast(self._batch.n_seq_id, ctypes.POINTER(ctypes.c_int32))
        seq = ctypes.cast(self._batch.seq_id, ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)))
        out = ctypes.cast(self._batch.logits, ctypes.POINTER(ctypes.c_int8))
        for i in range(n):
            pos[i] = i
            n_seq[i] = 1
            seq[i][0] = 0
            out[i] = 1
        self._token_bytes = ctypes.sizeof(llama_token) * n
        self._tokens_t = llama_token * n

        self._template = (LlamaTokenData * n_vocab)()
        for i in range(n_vocab):
            self._template[i].id = i
        self._cand = (LlamaTokenData * n_vocab)()
        self._cand_ptr = ctypes.cast(self._cand, ctypes.POINTER(LlamaTokenData))
        self._cand_floats = memoryview(self._cand).cast("B").cast("f")
        self._cand_bytes = ctypes.sizeof(self._cand)
        self._row_t = ctypes.c_float * n_vocab
        self._array = LlamaTokenDataArray()
        self._array_ref = ctypes.byref(self._array)
        self._logits = 0

    def decode(self, tokens: Sequence[int]) -> int:
        with self._guard():
            ctypes.memmove(self._batch.token, self._tokens_t(*tokens), self._token_bytes)
            code = self._api.llama_decode(self._ctx, self._batch)
            if code == 0:
                ptr = self._api.llama_get_logits(self._ctx)
                self._logits = ctypes.cast(ptr, ctypes.c_void_p).value or 0
                if not self._logits:
                    return -1
            return code

    def sample(self, row: int, algorithm: int, greedy: bool) -> Tuple[int, float]:
        if not 0 <= row < self._length:
            raise IndexError(f"logit row {row} outside the {self._length}-token canvas")
        with self._guard():
            ctypes.memmove(self._cand, self._template, self._cand_bytes)
            source = self._row_t.from_address(self._logits + row * self._n_vocab * 4)
            self._cand_floats[1::3] = memoryview(source).cast("B").cast("f")
            arr = self._array
            arr.data = self._cand_ptr
            arr.size = self._n_vocab
            arr.selected = -1
            arr.sorted = False
            self._api.llama_sampler_apply(self._chain, self._array_ref)
            data = arr.data
            size = int(arr.size)
            selected = int(arr.selected)
            probs = None
            if greedy or not 0 <= selected < size:
                if arr.sorted or size <= 1:
                    selected = 0
                else:
                    probs = _probs(data, size)
                    selected = probs.index(max(probs))
            token = int(data[selected].id)
            if algorithm == ALGORITHM_ENTROPY:
                return token, entropy_confidence(
                    probs if probs is not None else _probs(data, size))
            return token, _confidence(data, size, selected, algorithm)

    def close(self) -> None:
        if self._batch is not None:
            self._api.llama_batch_free(self._batch)
            self._batch = None
        if self._chain is not None:
            self._api.llama_sampler_free(self._chain)
            self._chain = None


def _probs(data, size: int) -> List[float]:
    """The ``p`` of the first *size* candidates at *data*, in order."""
    from ._structs import LlamaTokenData
    address = ctypes.cast(data, ctypes.c_void_p).value
    view = (LlamaTokenData * size).from_address(address)
    return memoryview(view).cast("B").cast("f")[2::3].tolist()


_ENTROPY_EPS = f32(1e-10)


def entropy_confidence(probs: Sequence[float]) -> float:
    """Negative entropy ``sum(p * log(p + 1e-10))`` of *probs*, accumulated in
    float32 in order: 0 for a certain prediction, lower for a flatter one, so
    the most certain positions rank first. This is the sign of Dream's
    reference sampler; the upstream example returns its negation."""
    entropy = 0.0
    for p in probs:
        entropy = f32(entropy + f32(p * f32(math.log(f32(p + _ENTROPY_EPS)))))
    return entropy


def _confidence(data, size: int, selected: int, algorithm: int) -> float:
    """The upstream example's ``calculate_confidence`` for CONFIDENCE, ORIGIN
    and MARGIN, in float32. RANDOM returns 0.0 (the loop draws its value)."""
    if algorithm in (ALGORITHM_CONFIDENCE, ALGORITHM_ORIGIN):
        return float(data[selected].p)
    if algorithm == ALGORITHM_MARGIN:
        if size > 1:
            return f32(float(data[0].p) - float(data[1].p))
        return float(data[0].p)
    return 0.0
