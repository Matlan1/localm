# SPDX-License-Identifier: AGPL-3.0-or-later
"""Speech synthesis with a llama.cpp text-to-speech GGUF through libmtmd.

A text-to-speech GGUF (``general.architecture`` ``qwen3tts``) is a text backbone
plus an mmproj holding the speech stages (speaker encoder, code predictor and
codec decoder). libmtmd's audio-generation helper builds the backbone prompt,
runs those stages and writes the WAV. :class:`SpeechSynthesizer` loads the
backbone in an embeddings-mode context plus the mmproj, samples the backbone's
codebook-0 token for every frame with the sampler chain llama.cpp's
``llama-tts`` uses, and drives the helper one frame at a time so a caller can
report progress and stop between frames.

The helper's input struct and ``step_gen`` signature are bound for the mtmd
builds that export ``mtmd_gen_inp_default``; an mtmd without that export lays
the input struct out differently and is refused with :class:`SpeechUnavailable`.

Runs inside the isolated speech worker (``localm/inference/_speech_runner.py``),
never in the server process.
"""

from __future__ import annotations

import ctypes
import re
import secrets
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from localm.debuglog import dedup_native_stderr, logger

from . import _api as api

# mtmd_gen_audio_type values.
PIPELINE_NONE = 0
PIPELINE_QWEN3TTS = 1
PIPELINE_POCKETTTS = 2
PIPELINE_NAMES = {PIPELINE_QWEN3TTS: "qwen3tts", PIPELINE_POCKETTTS: "pockettts"}
SUPPORTED_PIPELINES = frozenset({PIPELINE_QWEN3TTS})

# mtmd_helper_gen_audio_outtype: 16-bit little-endian mono WAV.
_OUTTYPE_WAV = 1

# Frames per second of audio for the supported pipelines.
FRAMES_PER_SECOND = 12.5

# llama-tts's sampling defaults before a model's own general.sampling.* keys
# override them (common_params_sampling plus the TTS example's penalty pair).
_DEFAULT_TOP_K = 40
_DEFAULT_TOP_P = 0.95
_DEFAULT_MIN_P = 0.05
_DEFAULT_TEMP = 0.8
_DEFAULT_PENALTY_REPEAT = 1.05
_DEFAULT_PENALTY_LAST_N = -1

# GGUF sampling keys this module applies, and the ones it does not.
_META_TOP_K = "general.sampling.top_k"
_META_TOP_P = "general.sampling.top_p"
_META_MIN_P = "general.sampling.min_p"
_META_TEMP = "general.sampling.temp"
_META_PENALTY_LAST_N = "general.sampling.penalty_last_n"
_META_PENALTY_REPEAT = "general.sampling.penalty_repeat"
_META_UNAPPLIED = ("general.sampling.sequence", "general.sampling.xtc_probability",
                   "general.sampling.xtc_threshold", "general.sampling.mirostat",
                   "general.sampling.mirostat_tau", "general.sampling.mirostat_eta")

DEFAULT_N_CTX = 8192
_N_BATCH = 2048
_N_UBATCH = 512

# Frame budget for one synthesis: a fixed allowance plus frames per text token,
# capped by what the context can hold.
_BUDGET_BASE_FRAMES = 125
_BUDGET_FRAMES_PER_TOKEN = 10
# A text whose context room is below this many frames per token cannot be spoken
# at an ordinary rate and is refused before synthesis starts.
_MIN_FRAMES_PER_TOKEN = 3
# Backbone positions the helper's prompt adds around the text tokens (role
# tokens, codec control tokens, the speaker embedding) plus slack.
_PROMPT_OVERHEAD = 24

# The helper's language codes and the codec language names they stand for.
LANGUAGE_CODES = {
    "zh": "chinese", "en": "english", "de": "german", "it": "italian",
    "pt": "portuguese", "es": "spanish", "ja": "japanese", "ko": "korean",
    "fr": "french", "ru": "russian",
}
_LANGUAGE_NAME_RE = re.compile(r"^[a-z][a-z_]{1,31}$")

_UINT32_MAX = 0xFFFFFFFF
_WAV_HEADER_BYTES = 44


class SpeechUnavailable(RuntimeError):
    """The runtime or the model files cannot synthesize speech: an mtmd build
    without the supported generation helper, an mmproj with no speech stages, or
    a pipeline localm does not run."""


class SpeechInputError(ValueError):
    """The request cannot be synthesized as given (unsupported language, a
    reference voice the model cannot use, text that does not fit)."""


class SpeechCancelled(Exception):
    """The caller asked the synthesis to stop; the synthesizer stays usable."""


class SpeechBudgetExceeded(RuntimeError):
    """The model kept generating past the frame budget without ending speech."""


class SpeechStageFailed(RuntimeError):
    """A native generation stage reported failure."""


@dataclass(frozen=True)
class SpeechResult:
    """One finished synthesis: a 16-bit mono WAV file and its facts."""
    wav: bytes
    sample_rate: int
    n_samples: int
    frames: int
    seed: int


class _GenAudioInfo(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int),
                ("sample_rate", ctypes.c_int32),
                ("model_variant", ctypes.c_char_p)]


class _HelperInput(ctypes.Structure):
    """``mtmd_helper_gen_audio_inp``."""
    _fields_ = [("seq_id", ctypes.c_int32),
                ("prompt", ctypes.c_char_p),
                ("prompt_len", ctypes.c_size_t),
                ("speaker_ref", ctypes.c_void_p),
                ("lang", ctypes.c_char_p),
                ("top_k", ctypes.c_int32),
                ("top_p", ctypes.c_float),
                ("seed", ctypes.c_uint32),
                ("out_type", ctypes.c_int)]


class _GenInp(ctypes.Structure):
    """``mtmd_gen_inp``."""
    _fields_ = [("type", ctypes.c_int),
                ("code0", ctypes.c_int32),
                ("embd", ctypes.POINTER(ctypes.c_float)),
                ("top_k", ctypes.c_int32),
                ("top_p", ctypes.c_float),
                ("seed", ctypes.c_uint32),
                ("temp", ctypes.c_float),
                ("codes", ctypes.POINTER(ctypes.c_int32)),
                ("n_codes", ctypes.c_size_t),
                ("feats", ctypes.POINTER(ctypes.c_float)),
                ("n_feats", ctypes.c_size_t),
                ("state_data", ctypes.c_char_p),
                ("state_size", ctypes.c_size_t)]


class _GenOut(ctypes.Structure):
    """``mtmd_gen_out``."""
    _fields_ = [("codes", ctypes.POINTER(ctypes.c_int32)),
                ("n_codes", ctypes.c_size_t),
                ("feats", ctypes.POINTER(ctypes.c_float)),
                ("n_feats", ctypes.c_size_t),
                ("embd", ctypes.POINTER(ctypes.c_float)),
                ("is_eos", ctypes.c_bool),
                ("audio", ctypes.POINTER(ctypes.c_float)),
                ("n_samples", ctypes.c_size_t),
                ("state_data", ctypes.c_char_p),
                ("state_size", ctypes.c_size_t)]


# mtmd_gen_process_type: hidden state to codes.
_GEN_PROCESS_CODE = 0


class _LogitBias(ctypes.Structure):
    _fields_ = [("token", ctypes.c_int32), ("bias", ctypes.c_float)]


_FloatPtr = ctypes.POINTER(ctypes.c_float)

_SIGNATURES = {
    "mtmd_gen_audio_get_info": (_GenAudioInfo, [ctypes.c_void_p]),
    "mtmd_gen_inp_default": (_GenInp, [ctypes.c_void_p]),
    "mtmd_gen_audio_process": (ctypes.c_int32, [
        ctypes.c_void_p, ctypes.POINTER(_GenInp), ctypes.POINTER(_GenOut)]),
    "mtmd_support_audio": (ctypes.c_bool, [ctypes.c_void_p]),
    "mtmd_get_audio_sample_rate": (ctypes.c_int, [ctypes.c_void_p]),
    "mtmd_bitmap_init_from_audio": (ctypes.c_void_p, [ctypes.c_size_t, _FloatPtr]),
    "mtmd_bitmap_free": (None, [ctypes.c_void_p]),
    "mtmd_helper_gen_audio_init": (ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p]),
    "mtmd_helper_gen_audio_free": (None, [ctypes.c_void_p]),
    "mtmd_helper_gen_audio_reset": (None, [ctypes.c_void_p]),
    "mtmd_helper_gen_audio_set_input": (ctypes.c_int32,
                                        [ctypes.c_void_p, ctypes.POINTER(_HelperInput)]),
    "mtmd_helper_gen_audio_step_prompt": (ctypes.c_int32, [ctypes.c_void_p, ctypes.c_int32]),
    "mtmd_helper_gen_audio_step_gen": (ctypes.c_int32, [
        ctypes.c_void_p, ctypes.c_int32, _FloatPtr,
        ctypes.POINTER(_FloatPtr), ctypes.POINTER(ctypes.c_bool)]),
    "mtmd_helper_gen_audio_get_output": (ctypes.c_int32, [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_int64)]),
}


class GenerationApi:
    """The speech generation functions of one loaded mtmd library, each its own
    function object: the signatures declared here never change, or depend on,
    the library's shared attributes that other bindings set."""


_bind_lock = threading.Lock()
_bound: dict[int, tuple] = {}


def bind_generation_api(m) -> GenerationApi:
    """The speech generation functions of the loaded mtmd library *m*.

    Raises :class:`SpeechUnavailable` when *m* lacks a required function or
    predates the supported helper layout (no ``mtmd_gen_inp_default``)."""
    with _bind_lock:
        cached = _bound.get(id(m))
        if cached is not None and cached[0] is m:
            return cached[1]
        try:
            m["mtmd_gen_inp_default"]
        except (AttributeError, KeyError) as e:
            raise SpeechUnavailable(
                "The installed llama.cpp runtime predates the speech generation "
                "interface localm supports. Install the runtime localm pins with "
                "'localm setup-llama'.") from e
        ns = GenerationApi()
        try:
            for name, (restype, argtypes) in _SIGNATURES.items():
                fn = m[name]
                fn.restype = restype
                fn.argtypes = argtypes
                setattr(ns, name, fn)
        except (AttributeError, KeyError) as e:
            raise SpeechUnavailable(
                f"The installed llama.cpp runtime lacks a speech generation "
                f"function ({e}). Install the runtime localm pins with "
                "'localm setup-llama'.") from e
        _bound[id(m)] = (m, ns)
        return ns


@dataclass(frozen=True)
class SamplingParams:
    """The backbone sampler settings for one model."""
    top_k: int
    top_p: float
    min_p: float
    temp: float
    penalty_last_n: int
    penalty_repeat: float


def _parse_int(raw: Optional[str], default: int) -> int:
    if raw is None:
        return default
    m = re.match(r"\s*[-+]?\d+", raw)
    return int(m.group(0)) if m else default


def _parse_float(raw: Optional[str], default: float) -> float:
    if raw is None:
        return default
    m = re.match(r"\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", raw)
    return float(m.group(0)) if m else default


def sampling_params_from_meta(read: Callable[[str], Optional[str]]) -> SamplingParams:
    """The sampler settings llama-tts would use for a model whose GGUF metadata
    is read by *read* (``key -> value string or None``): the TTS defaults,
    overridden by the model's ``general.sampling.*`` keys."""
    return SamplingParams(
        top_k=_parse_int(read(_META_TOP_K), _DEFAULT_TOP_K),
        top_p=_parse_float(read(_META_TOP_P), _DEFAULT_TOP_P),
        min_p=_parse_float(read(_META_MIN_P), _DEFAULT_MIN_P),
        temp=_parse_float(read(_META_TEMP), _DEFAULT_TEMP),
        penalty_last_n=_parse_int(read(_META_PENALTY_LAST_N), _DEFAULT_PENALTY_LAST_N),
        penalty_repeat=_parse_float(read(_META_PENALTY_REPEAT), _DEFAULT_PENALTY_REPEAT),
    )


def resolve_language(language: Optional[str]) -> Optional[str]:
    """The codec language name for *language* (an ISO 639-1 code from
    :data:`LANGUAGE_CODES` or a language name), or None for the model default.
    Raises :class:`SpeechInputError` for a value that cannot name a language."""
    if language is None or not language.strip():
        return None
    value = language.strip().lower()
    if value in LANGUAGE_CODES:
        return LANGUAGE_CODES[value]
    if not _LANGUAGE_NAME_RE.match(value):
        raise SpeechInputError(
            f"language must be a code such as 'en' or a name such as 'english' "
            f"(got {language[:40]!r}).")
    return value


def frame_budget(n_ctx: int, n_text_tokens: int, *, has_reference: bool) -> int:
    """The most frames one synthesis of *n_text_tokens* text tokens may generate
    in an *n_ctx* context. Raises :class:`SpeechInputError` when the context
    cannot hold the text spoken at an ordinary rate."""
    prompt = n_text_tokens + _PROMPT_OVERHEAD + (1 if has_reference else 0)
    room = n_ctx - prompt
    if room < _MIN_FRAMES_PER_TOKEN * max(1, n_text_tokens):
        raise SpeechInputError(
            f"The text is too long to speak in one request ({n_text_tokens} "
            f"tokens; this model's context holds about "
            f"{max(0, (n_ctx - _PROMPT_OVERHEAD) // (_MIN_FRAMES_PER_TOKEN + 1))}). "
            "Split it into shorter requests.")
    return min(room, _BUDGET_BASE_FRAMES + _BUDGET_FRAMES_PER_TOKEN * n_text_tokens)


def wav_pcm_payload(wav: bytes) -> bytes:
    """The sample bytes of a canonical 44-byte-header PCM WAV (as the helper
    writes it). Raises ValueError for any other layout."""
    if (len(wav) < _WAV_HEADER_BYTES or wav[0:4] != b"RIFF" or wav[8:12] != b"WAVE"
            or wav[12:16] != b"fmt " or wav[36:40] != b"data"):
        raise ValueError("not a canonical PCM WAV")
    (fmt_size,) = struct.unpack_from("<I", wav, 16)
    (data_size,) = struct.unpack_from("<I", wav, 40)
    if fmt_size != 16 or _WAV_HEADER_BYTES + data_size != len(wav):
        raise ValueError("not a canonical PCM WAV")
    return wav[_WAV_HEADER_BYTES:]


class SpeechSynthesizer:
    """A text-to-speech GGUF and its mmproj, loaded for synthesis.

    Not thread-safe: one :meth:`synthesize` at a time. The isolated speech
    worker owns one instance and serves requests in order."""

    def __init__(self, model_path: str, mmproj_path: str, *,
                 n_gpu_layers: int = 0, n_ctx: int = DEFAULT_N_CTX,
                 n_threads: Optional[int] = None, main_gpu: Optional[int] = None) -> None:
        from .mtmd import MtmdContext, _encode_threads, _load_lib
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.n_ctx = int(n_ctx)
        self._model = None
        self._ctx = None
        self._mtmd = None
        self._helper = None
        self._vocab = None
        self._mem = None
        self._last_gen_seed: Optional[int] = None
        self._m = bind_generation_api(_load_lib())
        if not api.has_embeddings_api() or not api.has_memory_api():
            raise SpeechUnavailable(
                "The installed llama.cpp runtime lacks the embeddings or memory "
                "interface speech synthesis needs.")
        threads = int(n_threads) if n_threads else _encode_threads()
        api.llama_backend_init()
        mp = api.llama_model_default_params()
        mp.n_gpu_layers = int(n_gpu_layers)
        if n_gpu_layers >= 99:
            from ._structs import set_use_mmap
            set_use_mmap(mp, False)
        from localm.discover import apply_main_gpu
        if main_gpu is not None:
            apply_main_gpu(mp, slot=main_gpu)
        else:
            apply_main_gpu(mp)
        try:
            with dedup_native_stderr():
                self._model = api.llama_load_model_from_file(model_path, mp)
                if not self._model:
                    raise SpeechUnavailable(
                        f"could not load the speech model {Path(model_path).name}")
                cp = api.llama_context_default_params()
                cp.n_ctx = self.n_ctx
                cp.n_batch = _N_BATCH
                cp.n_ubatch = _N_UBATCH
                cp.n_threads = threads
                cp.n_threads_batch = threads
                cp.embeddings = True
                self._ctx = api.llama_init_from_model(self._model, cp)
                if not self._ctx:
                    raise SpeechUnavailable("could not create the speech model's context")
                self._mtmd = MtmdContext(mmproj_path, self._model,
                                         gpu_index=int(mp.main_gpu))
            self._vocab = api.llama_model_get_vocab(self._model)
            self._mem = api.llama_get_memory(self._ctx)
            info = self._m.mtmd_gen_audio_get_info(self._mtmd._ctx)
            self.pipeline = int(info.type)
            self.sample_rate = int(info.sample_rate)
            if self.pipeline == PIPELINE_NONE:
                raise SpeechUnavailable(
                    f"{Path(mmproj_path).name} has no speech generation stages; "
                    "a text-to-speech model needs the mmproj published with it.")
            if self.pipeline not in SUPPORTED_PIPELINES:
                name = PIPELINE_NAMES.get(self.pipeline, str(self.pipeline))
                raise SpeechUnavailable(
                    f"localm does not synthesize speech with {name} models.")
            self.encoder_sample_rate = (
                int(self._m.mtmd_get_audio_sample_rate(self._mtmd._ctx))
                if self._m.mtmd_support_audio(self._mtmd._ctx) else 0)
            self._n_vocab = int(api.llama_vocab_n_tokens(self._vocab))
            self._n_embd = int(api.llama_model_n_embd(self._model))
            self._sampling = sampling_params_from_meta(
                lambda key: api.llama_model_meta_val_str(self._model, key))
            unapplied = [k for k in _META_UNAPPLIED
                         if api.llama_model_meta_val_str(self._model, k) is not None]
            if unapplied:
                logger.warning(
                    "speech model %s declares sampler settings the speech path does "
                    "not apply: %s", Path(model_path).name, ", ".join(unapplied))
            self._suppress = self._read_suppress_tokens()
            from localm.inference import pretokenizer_guard
            self._pre_type = pretokenizer_guard.read_pre_type(self._model, api)
            refusal = pretokenizer_guard.load_refusal(self._pre_type)
            if refusal is not None:
                raise SpeechUnavailable(refusal)
            self._helper = self._new_helper()
        except BaseException:
            self.close()
            raise
        logger.info("speech model ready: %s (%s, %d Hz, mmproj on %s)",
                    Path(model_path).name, PIPELINE_NAMES[self.pipeline],
                    self.sample_rate, "GPU" if self._mtmd.on_gpu else "CPU")

    @property
    def sampling(self) -> SamplingParams:
        return self._sampling

    @property
    def projector_on_gpu(self) -> bool:
        return bool(self._mtmd is not None and self._mtmd.on_gpu)

    def _new_helper(self) -> int:
        helper = self._m.mtmd_helper_gen_audio_init(self._ctx, self._mtmd._ctx)
        if not helper:
            raise SpeechUnavailable("could not create the speech generation helper")
        return helper

    def _read_suppress_tokens(self) -> list[int]:
        lib = api.load_lib()
        fn = getattr(lib, "llama_vocab_get_suppress_tokens", None)
        if fn is None:
            return []
        fn.restype = ctypes.POINTER(ctypes.c_int32)
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
        n = ctypes.c_int32(0)
        ptr = fn(self._vocab, ctypes.byref(n))
        if not ptr or n.value <= 0:
            return []
        return [int(ptr[i]) for i in range(n.value)]

    def _tokenize(self, text: str, *, parse_special: bool) -> list[int]:
        raw = text.encode("utf-8")
        cap = len(raw) + 16
        buf = (ctypes.c_int32 * cap)()
        n = api.llama_tokenize(self._vocab, raw, len(raw), buf, cap, False, parse_special)
        if n < 0:
            cap = -n
            buf = (ctypes.c_int32 * cap)()
            n = api.llama_tokenize(self._vocab, raw, len(raw), buf, cap, False, parse_special)
        if n < 0:
            raise SpeechInputError("the text could not be tokenized")
        return list(buf[:n])

    def has_language(self, name: str) -> bool:
        """True when the model's vocabulary has the codec token for language
        *name* (a resolved name such as ``english``)."""
        return len(self._tokenize(f"<|codec_language_{name}|>", parse_special=True)) == 1

    def check_text(self, text: str) -> int:
        """Validate *text* for synthesis and return its token count. Raises
        :class:`SpeechInputError` (or ``PretokenizerUnsafeInputError``) for text
        that is empty, unsafe for the tokenizer, or contains control-token text."""
        from localm.inference import pretokenizer_guard
        if not text.strip():
            raise SpeechInputError("The text to speak is empty.")
        pretokenizer_guard.check_text(self._pre_type, text)
        plain = self._tokenize(text, parse_special=False)
        if self._tokenize(text, parse_special=True) != plain:
            raise SpeechInputError(
                "The text contains one of the model's control-token strings (for "
                "example '<|im_end|>'), which would change the prompt instead of "
                "being spoken. Remove it and try again.")
        return len(plain)

    def _build_chain(self, seed: int) -> int:
        s = self._sampling
        chain = api.llama_sampler_chain_init(api.llama_sampler_chain_default_params())
        if self._suppress:
            fn = api.load_lib().llama_sampler_init_logit_bias
            fn.restype = ctypes.c_void_p
            fn.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(_LogitBias)]
            biases = (_LogitBias * len(self._suppress))(
                *[_LogitBias(t, float("-inf")) for t in self._suppress])
            api.llama_sampler_chain_add(chain, fn(self._n_vocab, len(self._suppress), biases))
        if s.penalty_last_n > 0 and s.penalty_repeat != 1.0 and api.has_penalties_sampler():
            api.llama_sampler_chain_add(chain, api.llama_sampler_init_penalties(
                s.penalty_last_n, s.penalty_repeat, 0.0, 0.0, n_vocab=self._n_vocab))
        api.llama_sampler_chain_add(chain, api.llama_sampler_init_top_k(s.top_k))
        api.llama_sampler_chain_add(chain, api.llama_sampler_init_top_p(s.top_p, 0))
        api.llama_sampler_chain_add(chain, api.llama_sampler_init_min_p(s.min_p, 0))
        api.llama_sampler_chain_add(chain, api.llama_sampler_init_temp(s.temp))
        api.llama_sampler_chain_add(chain, api.llama_sampler_init_dist(seed))
        return chain

    def synthesize(self, text: str, *, language: Optional[str] = None,
                   reference: Optional[bytes] = None,
                   seed: Optional[int] = None,
                   on_progress: Optional[Callable[[int], None]] = None,
                   should_stop: Optional[Callable[[], bool]] = None) -> SpeechResult:
        """Speak *text* and return the WAV.

        *language* is a code or name (:func:`resolve_language`); None uses the
        model default. *reference* is mono float32 little-endian samples at
        :attr:`encoder_sample_rate` whose voice the speech imitates. *seed*
        makes the output reproducible; None picks one at random (reported in
        the result). *on_progress* receives the frame count after each frame;
        *should_stop* is polled between frames and raises
        :class:`SpeechCancelled` once it returns True.

        Raises :class:`SpeechInputError`, ``PretokenizerUnsafeInputError``,
        :class:`SpeechCancelled`, :class:`SpeechBudgetExceeded` or
        :class:`SpeechStageFailed`. A projector GPU failure is retried once on
        the CPU; when the projector cannot be reopened there,
        :class:`SpeechUnavailable` is raised and the synthesizer is unusable."""
        if self._helper is None:
            raise SpeechUnavailable("the speech model is closed")
        n_tokens = self.check_text(text)
        lang = resolve_language(language)
        if lang is not None and not self.has_language(lang):
            raise SpeechInputError(f"This model does not speak '{language}'.")
        if reference is not None:
            if not self.encoder_sample_rate:
                raise SpeechInputError(
                    "This model has no speaker encoder, so it cannot imitate a "
                    "reference voice.")
            if len(reference) < 4 or len(reference) % 4:
                raise SpeechInputError("The reference voice recording is empty.")
        budget = frame_budget(self.n_ctx, n_tokens, has_reference=reference is not None)
        if seed is None:
            seed = secrets.randbelow(_UINT32_MAX)
        seed = int(seed) & _UINT32_MAX
        if seed == _UINT32_MAX:
            seed = 0
        try:
            return self._run(text, lang, reference, seed, budget, on_progress, should_stop)
        except SpeechStageFailed:
            if not self._mtmd.on_gpu:
                raise
            if not self._retry_projector_on_cpu():
                raise SpeechUnavailable(
                    "The speech model's projector failed on the GPU and could not "
                    "be reopened on the CPU; the model is unloaded and loads again "
                    "on the next request.") from None
            return self._run(text, lang, reference, seed, budget, on_progress, should_stop)

    def _reseed_generation_rng(self, seed: int) -> None:
        """Make the projector's generation RNG start from *seed* on the next
        generation call. The projector reseeds only when the seed differs from
        the one its previous call used, so a request that repeats the previous
        request's seed first runs one code-generation call with another seed."""
        if self._last_gen_seed != seed:
            return
        m = self._m
        inp = m.mtmd_gen_inp_default(self._mtmd._ctx)
        inp.type = _GEN_PROCESS_CODE
        inp.code0 = 0
        zeros = (ctypes.c_float * self._n_embd)()
        inp.embd = ctypes.cast(zeros, ctypes.POINTER(ctypes.c_float))
        inp.seed = (seed ^ 1) & _UINT32_MAX
        out = _GenOut()
        self._last_gen_seed = None
        if m.mtmd_gen_audio_process(self._mtmd._ctx, ctypes.byref(inp), ctypes.byref(out)) != 0:
            raise SpeechStageFailed("the speech generator could not be reset")
        self._last_gen_seed = int(inp.seed)

    def _retry_projector_on_cpu(self) -> bool:
        self._m.mtmd_helper_gen_audio_free(self._helper)
        self._helper = None
        self._last_gen_seed = None
        if not self._mtmd.retry_on_cpu():
            return False
        self._helper = self._new_helper()
        return True

    def _run(self, text: str, lang: Optional[str], reference: Optional[bytes],
             seed: int, budget: int, on_progress, should_stop) -> SpeechResult:
        m = self._m
        helper = self._helper
        api.llama_memory_clear(self._mem, True)
        self._reseed_generation_rng(seed)
        chain = self._build_chain(seed)
        bitmap = None
        try:
            if reference is not None:
                n = len(reference) // 4
                samples = (ctypes.c_float * n).from_buffer_copy(reference)
                bitmap = m.mtmd_bitmap_init_from_audio(n, samples)
                if not bitmap:
                    raise SpeechInputError("The reference voice could not be prepared.")
            raw = text.encode("utf-8")
            sp = self._sampling
            inp = _HelperInput(0, raw, len(raw), bitmap,
                               (lang or "").encode("ascii"), sp.top_k, sp.top_p,
                               seed, _OUTTYPE_WAV)
            if m.mtmd_helper_gen_audio_set_input(helper, ctypes.byref(inp)) != 0:
                raise SpeechStageFailed("the speech prompt could not be prepared")
            if bitmap is not None:
                m.mtmd_bitmap_free(bitmap)
                bitmap = None
            while True:
                if should_stop is not None and should_stop():
                    raise SpeechCancelled()
                left = m.mtmd_helper_gen_audio_step_prompt(helper, _N_BATCH)
                if left < 0:
                    raise SpeechStageFailed("the speech prompt could not be processed")
                if left == 0:
                    break
            sampled = api.llama_sampler_sample(chain, self._ctx, -1)
            h_state = api.llama_get_embeddings_ith(self._ctx, -1)
            if not h_state:
                raise SpeechStageFailed("the speech model produced no hidden state")
            frames = 0
            stop = ctypes.c_bool(False)
            while not stop.value:
                if should_stop is not None and should_stop():
                    raise SpeechCancelled()
                if frames >= budget:
                    raise SpeechBudgetExceeded(
                        f"The model did not finish speaking within {budget} frames "
                        f"(about {budget / FRAMES_PER_SECOND:.0f} s of audio); it may be "
                        "repeating itself. Try again, or with a different seed.")
                h_next = _FloatPtr()
                if m.mtmd_helper_gen_audio_step_gen(
                        helper, int(sampled), h_state, ctypes.byref(h_next),
                        ctypes.byref(stop)) != 0:
                    raise SpeechStageFailed("a speech generation stage failed")
                if not h_next:
                    break
                frames += 1
                h_state = h_next
                if on_progress is not None:
                    on_progress(frames)
                sampled = api.llama_sampler_sample(chain, self._ctx, -1)
            rate = ctypes.c_int32()
            data = ctypes.c_void_p()
            size = ctypes.c_size_t()
            n_samples = ctypes.c_int64()
            if m.mtmd_helper_gen_audio_get_output(
                    helper, ctypes.byref(rate), ctypes.byref(data), ctypes.byref(size),
                    ctypes.byref(n_samples)) != 0 or not data:
                raise SpeechStageFailed("the speech audio could not be decoded")
            wav = ctypes.string_at(data, size.value)
            return SpeechResult(wav=wav, sample_rate=int(rate.value),
                                n_samples=int(n_samples.value), frames=frames, seed=seed)
        finally:
            self._last_gen_seed = seed
            if bitmap is not None:
                m.mtmd_bitmap_free(bitmap)
            api.llama_sampler_free(chain)
            m.mtmd_helper_gen_audio_reset(helper)
            api.llama_memory_clear(self._mem, True)

    def close(self) -> None:
        """Free the helper, the mmproj, the context and the model. Safe to call
        more than once."""
        if self._helper is not None:
            try:
                self._m.mtmd_helper_gen_audio_free(self._helper)
            finally:
                self._helper = None
        if self._mtmd is not None:
            try:
                self._mtmd.free()
            finally:
                self._mtmd = None
        if self._ctx is not None:
            try:
                api.llama_free(self._ctx)
            finally:
                self._ctx = None
        if self._model is not None:
            try:
                api.llama_free_model(self._model)
            finally:
                self._model = None
