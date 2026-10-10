# SPDX-License-Identifier: AGPL-3.0-or-later
"""On-device speech synthesis with a text-to-speech GGUF.

A registered model of type ``tts`` (a Qwen3-TTS GGUF plus its mmproj) speaks
text into a WAV file. It runs in its own isolated worker process
(``_speech_runner.py``), so a native fault costs the worker, never the server.

One speech model is resident at a time, loaded on first use and replaced when a
request names a different one. Loading serialises on the engine's
process-global load lock, so it never races a chat-model load onto the GPU.

``resolve_speech_model`` maps a request's model name to a registered speech
model; a request can never name a filesystem path. ``synthesize`` speaks.
Reference voices are WAV files in :func:`voices_dir`, named by file stem.
"""

from __future__ import annotations

import atexit
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from localm import pathscrub
from localm.debuglog import logger

# LOCK ORDER: engine._LOAD_LOCK (outer) -> _LOCK (inner), never the reverse, as
# in reranker.py. _LOCK is held for a whole worker spawn and model load, so no
# reader below may be called from an `async def` handler.
_LOCK = threading.RLock()
_ENGINE: Optional[SpeechEngine] = None
_ENGINE_KEY: Optional[tuple] = None
_LOAD_FAILED: dict[tuple, tuple[str, float]] = {}
_LOAD_RETRY_AFTER_S = 60.0

OPENAI_MODEL_ALIASES = ("tts-1", "tts-1-hd", "gpt-4o-mini-tts")
DEFAULT_VOICE = "default"
MAX_INPUT_CHARS = 4096
MAX_REFERENCE_SECONDS = 30.0
MAX_REFERENCE_BYTES = 16 * 1024 * 1024
_VOICE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class SpeechModelError(Exception):
    """A request names a model or voice that cannot be used. *status* is the
    HTTP status the route reports."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class SpeechUnavailableError(RuntimeError):
    """The speech model is registered but could not be loaded."""


@dataclass(frozen=True)
class SpeechModel:
    """A registered speech model: its name, GGUF and mmproj paths."""
    name: str
    path: str
    mmproj: str


@dataclass(frozen=True)
class SpeechOutput:
    """A finished synthesis: a 16-bit mono WAV and its facts."""
    wav: bytes
    sample_rate: int
    n_samples: int
    frames: int
    seed: int

    @property
    def seconds(self) -> float:
        return self.n_samples / self.sample_rate if self.sample_rate else 0.0


# A progress event: {"stage": "loading" | "waiting" | "speaking",
# "frames": int, "seconds": float} (frames and seconds while speaking).
ProgressFn = Callable[[dict], None]


def registered_speech_models() -> list[str]:
    """Names of the registered models of type ``tts``, sorted."""
    from localm.config import load_registry
    reg = load_registry()
    return sorted(name for name, entry in reg.items()
                  if isinstance(entry, dict) and entry.get("model_type") == "tts")


def _kind_words(kind: str) -> str:
    from localm.inference.reranker import _KIND_WORDS
    return _KIND_WORDS.get(kind, f"a {kind} model")


def resolve_speech_model(model: Optional[str]) -> SpeechModel:
    """The speech model a request names.

    An omitted model, ``localm`` or one of :data:`OPENAI_MODEL_ALIASES` resolves
    to the only registered speech model. Raises :class:`SpeechModelError` with
    the HTTP status to report: 404 when nothing matching is registered, 400 when
    an alias is ambiguous, 422 when the model is not a speech model or its files
    are missing."""
    from localm.config import load_registry
    from localm.model_manager.registry import get_model_mmproj
    from localm.pathsafe import is_unc_or_device_path
    name = (model or "").strip()
    if not name or name == "localm" or name.lower() in OPENAI_MODEL_ALIASES:
        names = registered_speech_models()
        if not names:
            raise SpeechModelError(
                "No text-to-speech model is registered. Add a Qwen3-TTS GGUF with "
                "its mmproj (for example 'localm pull "
                "ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF:"
                "Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf') and name it in the "
                "request's model field.", 404)
        if len(names) > 1:
            raise SpeechModelError(
                "Several text-to-speech models are registered (" + ", ".join(names)
                + "); name one in the request's model field.", 400)
        name = names[0]
    reg = load_registry()
    entry = reg.get(name)
    if not isinstance(entry, dict):
        raise SpeechModelError(f"Model {name!r} is not registered.", 404)
    kind = entry.get("model_type") or "llm"
    if not isinstance(kind, str):
        kind = "unknown"
    if kind != "tts":
        raise SpeechModelError(
            f"Model {name!r} is {_kind_words(kind)}, not a text-to-speech model.", 422)
    path = entry.get("path")
    if (not isinstance(path, str) or not path or is_unc_or_device_path(path)
            or not Path(path).is_file()):
        raise SpeechModelError(
            f"Model {name!r} is registered but its GGUF file is missing or is not "
            "a single local file.", 422)
    mmproj = get_model_mmproj(name, reg=reg)
    if not mmproj or is_unc_or_device_path(mmproj) or not Path(mmproj).is_file():
        raise SpeechModelError(
            f"Model {name!r} needs its mmproj (the file holding the speech "
            "stages), and none is attached. Pull the model again so its mmproj is "
            "fetched, or attach one with 'localm pull --mmproj'.", 422)
    return SpeechModel(name, path, mmproj)


def voices_dir() -> Path:
    """The directory whose ``<name>.wav`` files are the named reference voices."""
    from localm.config import home_dir
    return home_dir() / "voices"


def list_voices() -> list[str]:
    """``default`` plus the names of the WAV files in :func:`voices_dir`."""
    names = []
    try:
        for p in voices_dir().glob("*.wav"):
            if p.is_file() and _VOICE_NAME_RE.match(p.stem) and p.stem != DEFAULT_VOICE:
                names.append(p.stem)
    except OSError as e:
        logger.debug("could not list the voices directory: %s", e)
    return [DEFAULT_VOICE, *sorted(names)]


def voice_reference(voice: Optional[str]) -> Optional[bytes]:
    """The reference WAV bytes for the named *voice*, or None for the model's
    own default voice. Raises :class:`SpeechModelError` (400) for a voice that
    does not exist."""
    name = (voice or DEFAULT_VOICE).strip()
    if name == DEFAULT_VOICE:
        return None
    if _VOICE_NAME_RE.match(name):
        path = voices_dir() / f"{name}.wav"
        try:
            if path.is_file():
                if path.stat().st_size > MAX_REFERENCE_BYTES:
                    raise SpeechModelError(
                        f"The voice file for {name!r} is larger than "
                        f"{MAX_REFERENCE_BYTES // (1024 * 1024)} MB.", 400)
                return path.read_bytes()
        except OSError as e:
            raise SpeechModelError(f"The voice file for {name!r} could not be read "
                                   f"({type(e).__name__}).", 400) from e
    raise SpeechModelError(
        f"Voice {name[:64]!r} is not available. Available voices: "
        + ", ".join(list_voices()) + f". Add one by saving a WAV recording of up "
        f"to {MAX_REFERENCE_SECONDS:.0f} s as <name>.wav in the voices folder.", 400)


def _file_key(model: SpeechModel) -> tuple:
    def one(path: str) -> tuple:
        try:
            st = Path(path).stat()
            return (str(Path(path).resolve()), st.st_mtime_ns, st.st_size)
        except OSError:
            return (str(path), 0, 0)
    return one(model.path) + one(model.mmproj)


def _latched_failure(key: tuple) -> Optional[str]:
    failed = _LOAD_FAILED.get(key)
    if failed is None:
        return None
    reason, at = failed
    if time.monotonic() - at < _LOAD_RETRY_AFTER_S:
        return reason
    del _LOAD_FAILED[key]
    return None


def _estimate_bytes(model: SpeechModel, n_ctx: int) -> int:
    from localm.model_manager.gguf import gguf_kv_bytes_per_token
    size = 0
    for p in (model.path, model.mmproj):
        try:
            size += Path(p).stat().st_size
        except OSError:
            pass
    try:
        kv = gguf_kv_bytes_per_token(Path(model.path)) * n_ctx
    except Exception as e:   # noqa: BLE001 - an estimate falls back to weights only
        logger.debug("speech: no KV estimate for %s (%s)", Path(model.path).name, e)
        kv = 0
    return int(size * 1.2) + kv


def _choose_gpu_layers(model: SpeechModel, n_ctx: int) -> tuple[int, Optional[str]]:
    """``(n_gpu_layers, reason)``: the configured ``n_gpu_layers`` when it is
    moved off its 99 default, else everything on the GPU unless free VRAM was
    measured and cannot hold the model, the mmproj and the context even after
    the chat model is swapped out under the ``model_swap_policy``."""
    from localm.config import load_config
    cfg = load_config()
    raw = cfg.get("n_gpu_layers", 99)
    if isinstance(raw, int) and not isinstance(raw, bool) and raw != 99:
        return int(raw), None
    estimate = _estimate_bytes(model, n_ctx)
    try:
        from localm.discover import GPU_PROBE_OK, vram_capacity
        info, status = vram_capacity(return_status=True)
        free = info.get("free") if status == GPU_PROBE_OK else None
    except Exception as e:   # noqa: BLE001 - unmeasurable VRAM keeps the full offload
        logger.debug("speech: free VRAM unreadable (%s)", e)
        free = None
    if free is None or free >= estimate:
        return 99, None
    from localm.vram import decide_embedder_swap, evict_chat_for_embedder, resolve_swap_policy
    if decide_embedder_swap(estimate, policy=resolve_swap_policy({}, cfg)):
        evict_chat_for_embedder()
        try:
            info, status = vram_capacity(return_status=True)
            free = info.get("free") if status == GPU_PROBE_OK else None
        except Exception:   # noqa: BLE001 - see above
            free = None
        if free is None or free >= estimate:
            return 99, None
    return 0, (f"not enough free VRAM for the speech model (~{estimate // 1024 ** 2} "
               f"MB needed, {int(free) // 1024 ** 2} MB free); it runs on the CPU")


class SpeechEngine:
    """A loaded speech model in its isolated worker."""

    def __init__(self, model: SpeechModel, *, n_gpu_layers: int,
                 n_ctx: Optional[int] = None, n_threads: Optional[int] = None) -> None:
        from localm.inference._speech_runner import SpeechRunner
        from localm.inference.backends.llamacpp.mtmd_gen import DEFAULT_N_CTX
        self.model = model
        self.n_ctx = int(n_ctx or DEFAULT_N_CTX)
        self._runner = SpeechRunner()
        self._rpc_lock = threading.Lock()
        self._count_lock = threading.Lock()
        self.active_requests = 0
        meta = self._runner.spawn_and_load({
            "model_path": model.path, "mmproj_path": model.mmproj,
            "n_gpu_layers": int(n_gpu_layers), "n_ctx": self.n_ctx,
            "n_threads": n_threads, "cpu_only": int(n_gpu_layers) <= 0})
        self.sample_rate = int(meta["sample_rate"])
        self.encoder_sample_rate = int(meta["encoder_sample_rate"])
        self.projector_on_gpu = bool(meta["projector_on_gpu"])

    @property
    def alive(self) -> bool:
        return self._runner.is_alive()

    def speak(self, text: str, *, language: Optional[str] = None,
              reference_wav: Optional[bytes] = None, seed: Optional[int] = None,
              on_progress: Optional[ProgressFn] = None,
              should_cancel: Optional[Callable[[], bool]] = None) -> SpeechOutput:
        """Speak *text*. *reference_wav* is a WAV recording whose voice to
        imitate. Raises the speech errors of ``mtmd_gen`` and RuntimeError when
        the worker crashed or hung (it is gone afterwards)."""
        from localm.inference.backends.llamacpp.mtmd_gen import (
            FRAMES_PER_SECOND, SpeechInputError)
        reference = None
        if reference_wav is not None:
            if not self.encoder_sample_rate:
                raise SpeechInputError(
                    "This model has no speaker encoder, so it cannot imitate a "
                    "reference voice.")
            from localm.wav_audio import WavError, to_mono_float32
            try:
                reference = to_mono_float32(reference_wav, self.encoder_sample_rate,
                                            max_seconds=MAX_REFERENCE_SECONDS)
            except WavError as e:
                raise SpeechInputError(f"The reference voice could not be read: {e}.") from e
        with self._count_lock:
            self.active_requests += 1
        try:
            if not self._rpc_lock.acquire(blocking=False):
                if on_progress is not None:
                    on_progress({"stage": "waiting"})
                while not self._rpc_lock.acquire(timeout=0.5):
                    if should_cancel is not None and should_cancel():
                        from localm.inference.backends.llamacpp.mtmd_gen import SpeechCancelled
                        raise SpeechCancelled("The speech request was cancelled while waiting.")
            try:
                if on_progress is not None:
                    on_progress({"stage": "speaking", "frames": 0, "seconds": 0.0})

                def _frames(n: int) -> None:
                    if on_progress is not None:
                        on_progress({"stage": "speaking", "frames": n,
                                     "seconds": n / FRAMES_PER_SECOND})

                result = self._runner.speak(
                    {"text": text, "language": language, "reference": reference,
                     "seed": seed},
                    on_progress=_frames, should_cancel=should_cancel)
            finally:
                self._rpc_lock.release()
        finally:
            with self._count_lock:
                self.active_requests -= 1
        return SpeechOutput(wav=result["wav"], sample_rate=int(result["sample_rate"]),
                            n_samples=int(result["n_samples"]),
                            frames=int(result["frames"]), seed=int(result["seed"]))

    def close(self, grace: float = 5.0) -> None:
        self._runner.shutdown(grace=grace)


def get_engine(model: SpeechModel, *, on_progress: Optional[ProgressFn] = None) -> SpeechEngine:
    """The resident engine for *model*, loading it first (and releasing a
    different resident one). Raises :class:`SpeechUnavailableError` when the
    load fails (a failed file pair is not retried for a minute unless it
    changes) or a different speech model is still busy."""
    global _ENGINE, _ENGINE_KEY
    key = _file_key(model)
    with _LOCK:
        if _ENGINE is not None and _ENGINE_KEY == key and _ENGINE.alive:
            return _ENGINE
        failed = _latched_failure(key)
        if failed is not None:
            raise SpeechUnavailableError(failed)
    if on_progress is not None:
        on_progress({"stage": "loading"})
    from localm.inference.backends.llamacpp.mtmd_gen import DEFAULT_N_CTX
    ngl, reason = _choose_gpu_layers(model, DEFAULT_N_CTX)
    if reason is not None:
        logger.warning("speech placement: %s", reason)
    from localm.inference.engine import _LOAD_LOCK
    with _LOAD_LOCK:
        with _LOCK:
            if _ENGINE is not None and _ENGINE_KEY == key and _ENGINE.alive:
                return _ENGINE
            failed = _latched_failure(key)
            if failed is not None:
                raise SpeechUnavailableError(failed)
            current = _ENGINE
            if current is not None and current.alive and current.active_requests > 0:
                raise SpeechUnavailableError(
                    "another speech model is still speaking a request; retry shortly")
            if current is not None:
                _ENGINE = None
                _ENGINE_KEY = None
                current.close()
            try:
                _ENGINE = SpeechEngine(model, n_gpu_layers=ngl)
            except Exception as e:
                reason_text = pathscrub.scrub_paths(str(e))
                _LOAD_FAILED[key] = (reason_text, time.monotonic())
                logger.warning("could not load speech model %s (%s)", model.name, e)
                raise SpeechUnavailableError(reason_text) from e
            _ENGINE_KEY = key
            logger.info("speech model ready: %s (%d Hz, projector on %s)", model.name,
                        _ENGINE.sample_rate, "GPU" if _ENGINE.projector_on_gpu else "CPU")
            return _ENGINE


def synthesize(model: SpeechModel, text: str, *, language: Optional[str] = None,
               reference_wav: Optional[bytes] = None, seed: Optional[int] = None,
               on_progress: Optional[ProgressFn] = None,
               should_cancel: Optional[Callable[[], bool]] = None) -> SpeechOutput:
    """Speak *text* with *model* (loading it when needed). A worker that
    crashed or hung is dropped, so the next request starts a fresh one."""
    global _ENGINE, _ENGINE_KEY
    engine = get_engine(model, on_progress=on_progress)
    logger.info("speech: %d characters with %s", len(text), model.name)
    try:
        out = engine.speak(text, language=language, reference_wav=reference_wav,
                           seed=seed, on_progress=on_progress,
                           should_cancel=should_cancel)
    except RuntimeError:
        if not engine.alive:
            with _LOCK:
                if _ENGINE is engine:
                    _ENGINE = None
                    _ENGINE_KEY = None
        raise
    logger.info("speech: %.2f s of audio (%d frames)", out.seconds, out.frames)
    return out


def speech_info() -> Optional[dict]:
    """``{"name", "path", "sample_rate"}`` of the resident speech model, or
    None. Does not load."""
    with _LOCK:
        if _ENGINE is None:
            return None
        return {"name": _ENGINE.model.name, "path": _ENGINE.model.path,
                "sample_rate": _ENGINE.sample_rate}


def is_loaded() -> bool:
    """True while a speech model is resident. Does not load."""
    with _LOCK:
        return _ENGINE is not None


def is_resident() -> bool:
    """True while a speech model is resident, without taking the lock (a
    snapshot for exit paths that must not wait on a load)."""
    return _ENGINE is not None


def active_requests() -> int:
    """In-flight requests on the resident speech model, or 0."""
    with _LOCK:
        return _ENGINE.active_requests if _ENGINE is not None else 0


def reset_speech(*, force: bool = True) -> bool:
    """Release the resident speech model and clear the failed-load memory.
    Returns True when a model was released; with ``force=False`` a model with a
    request in flight is left alone (and nothing is cleared), checked and
    released in one locked step."""
    global _ENGINE, _ENGINE_KEY
    with _LOCK:
        if not force and _ENGINE is not None and _ENGINE.active_requests > 0:
            return False
        current = _ENGINE
        if current is not None:
            current.close()
        _ENGINE = None
        _ENGINE_KEY = None
        _LOAD_FAILED.clear()
        return current is not None


def release_for_exit() -> bool:
    """Release the speech worker for a caller about to ``os._exit()`` /
    ``os.execv()``, which bypass atexit and would orphan it. Takes no lock: a
    busy worker is terminated without waiting, an idle one closed politely."""
    engine = _ENGINE
    if engine is None:
        return False
    engine.close(grace=0 if engine.active_requests > 0 else 5.0)
    return True


atexit.register(reset_speech)
