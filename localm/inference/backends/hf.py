# SPDX-License-Identifier: AGPL-3.0-or-later
"""HuggingFace Transformers backend - parent-side proxy."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional

from localm.console import console
from localm.debuglog import logger

from ._hf_runner import (
    EMBED_MAX_CHARS_DEFAULT,
    EMBED_MAX_TEXTS_DEFAULT,
    EMBED_TIMEOUT_DEFAULT,
    FIRST_TOKEN_TIMEOUT_DEFAULT,
    LOAD_TIMEOUT_DEFAULT,
    HFRunner,
    RunnerBusy,
)
from .base import (
    IMAGE_UNSUPPORTED_MESSAGE,
    BaseBackend,
    EmbedBatchTooLargeError,
    UnsupportedInputError,
    messages_contain_image,
)


def _trust_remote_code_enabled() -> bool:
    """Whether transformers may import and execute a model directory's own .py."""
    try:
        from localm.config import load_config
        return bool(load_config().get("hf_trust_remote_code", False))
    except Exception as e:
        # Fail CLOSED: an unreadable config must never be read as permission to
        # execute a model's bundled code. Logged rather than silent (rule 5).
        logger.debug("hf: could not read hf_trust_remote_code, assuming off: %s", e)
        return False


def _declares_custom_code(model_path: str) -> bool:
    """True when this model directory asks transformers to run its own Python."""
    import json
    p = Path(model_path)
    if not p.is_dir():
        return False
    # Every file transformers actually consults for auto_map. NOTE the spelling:
    # it is preprocessor_config.json (with the "pre"), which AutoProcessor reads -
    # an earlier version of this list said "processor_config.json" and so missed
    # the most common multimodal case entirely. A miss here is not an execution
    # bypass (trust_remote_code is genuinely passed as False, so transformers still
    # refuses), but it costs the user the clear message this function exists to
    # give: transformers' own refusal for a processor gets caught by the broad
    # handler in load() and relabelled "processor load failed ... falling back to
    # text-only tokenizer", so the model quietly loads WITHOUT the capability and
    # nobody is told it asked to run its own code.
    for fname in ("config.json", "tokenizer_config.json",
                  "preprocessor_config.json", "processor_config.json",
                  "video_preprocessor_config.json"):
        f = p / fname
        if not f.is_file():
            continue
        try:
            # Any non-empty auto_map counts. transformers still accepts the legacy
            # list/tuple form, so an isinstance(dict) test alone under-detects;
            # this asks "does it declare one" rather than "is it the shape I
            # expected", which is the safe direction for a refuse-gate.
            if json.loads(f.read_text(encoding="utf-8")).get("auto_map"):
                return True
        except Exception as e:
            # A malformed/unreadable file is NOT evidence of safety, but it is also
            # not evidence of custom code - the real load will fail on it later with
            # a better message. Surface it instead of deciding silently (rule 5).
            logger.debug("hf: could not parse %s while checking for custom "
                         "code: %s", f, e)
    return False


def _check_custom_code_allowed(model_path: str) -> None:
    """Refuse, with an actionable message, a model that needs custom code when ``hf_trust_remote_code`` is off."""
    if not _declares_custom_code(model_path):
        return
    if _trust_remote_code_enabled():
        return
    raise RuntimeError(
        f"'{Path(model_path).name}' requires custom code: it asks to import and "
        "run Python bundled inside the model directory (auto_map). That is "
        "arbitrary code execution as the user running localm, so it is refused by "
        "default.\n"
        "If you trust the source of this model, enable it with:\n"
        "  localm config hf_trust_remote_code true\n"
        "Only do that for a model you obtained from a source you trust.")


class HFBackend(BaseBackend):
    """Parent-side handle to a HuggingFace-format model loaded in an isolated child process."""

    # An HF checkpoint may ship an image processor; whether this instance can
    # actually see images is only known after load() (see supports_images).
    can_be_multimodal = True

    def __init__(self, model_path: str, device: Optional[str] = None) -> None:
        self.model_path = str(Path(model_path).resolve())
        self._device = device      # None = auto-detect
        # The isolated worker process holding the real model - see
        # _hf_runner.py. None until load() succeeds.
        self._runner: Optional[HFRunner] = None
        self._loaded = False
        # Cached from the child's load response - the real HFWorker instance
        # now lives in the child, so these can no longer be read live off it.
        self._supports_images = False
        self.effective_ctx_max: Optional[int] = None
        self.n_ctx_max: Optional[int] = None
        # Unloaded -> True ("unknown, load to find out"): unlike
        # supports_images, this is NOT gated on current liveness once known -
        # see the can_embed property below for why that distinction matters.
        self._can_embed = True

    @property
    def loaded(self) -> bool:
        """A backend whose isolated worker is GONE is not loaded, whatever ``_loaded`` says - mirrors ``GgufBackend.loaded`` exactly (REG-606): the worker can die out from under this object on a timeout or a crash without any exception ever reaching a caller that would clear ``_loaded`` (a killed ``chat_stream..."""
        if not self._loaded:
            return False
        is_alive = getattr(self._runner, "is_alive", None)
        return True if is_alive is None else bool(is_alive())

    @property
    def supports_images(self) -> bool:
        """True once loaded with a working multimodal processor."""
        return bool(self.loaded and self._supports_images)

    @property
    def can_embed(self) -> bool:
        """True only when the LOADED model is a GENUINE embedding model (an architectural fact about the checkpoint, computed once by the child right after load - see HFWorker.can_embed for the full reasoning and the measurements behind it)."""
        return self._can_embed

    @property
    def supports_grammar(self) -> bool:
        """True only when xgrammar (the optional ``[grammar]`` extra) is installed, because that package IS this backend's grammar support."""
        import importlib.util
        try:
            return importlib.util.find_spec("xgrammar") is not None
        except (ImportError, ValueError):
            # A broken/partial install: the parent package is missing or the
            # module has no spec. Either way the child's own import would fail
            # too, so the honest answer is the same one it would produce.
            return False

    def validate_grammar(self, grammar: Optional[str], *, lazy: bool = False) -> None:
        """Refuse a LAZY grammar up front; defer everything else to the base."""
        from .base import GRAMMAR_LAZY_UNSUPPORTED_MESSAGE, GrammarUnsupportedError
        super().validate_grammar(grammar, lazy=lazy)
        if grammar and lazy:
            raise GrammarUnsupportedError(GRAMMAR_LAZY_UNSUPPORTED_MESSAGE)

    # ------------------------------------------------------------------ #
    #  Load / unload                                                       #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        # Two pre-flight refusals, BEFORE a child is ever spawned - neither
        # needs torch/transformers or anything only the child can see, so a
        # refused model never pays the cost of a child spawn:
        #   1. Custom code (auto_map) the user has not explicitly trusted.
        #   2. A tokenizer.json regex pattern that fails the Oniguruma safety
        #      probe (NEW-HF-TOKENIZER-REDOS - see hf_tokenizer_safety.py's
        #      module docstring for the full mechanism and the measurements
        #      behind probing Oniguruma specifically, not Python re).
        _check_custom_code_allowed(self.model_path)
        from localm.inference.hf_tokenizer_safety import validate_tokenizer_json
        validate_tokenizer_json(self.model_path)

        self._runner = HFRunner()
        params = {"model_path": self.model_path, "device": self._device}
        meta = self._runner.spawn_and_load(params, timeout=self._load_timeout_seconds())
        self._supports_images = bool(meta.get("supports_images"))
        self._can_embed = bool(meta.get("can_embed", True))
        self.effective_ctx_max = meta.get("context_capacity")
        self.n_ctx_max = self.effective_ctx_max
        self._loaded = True
        # Printed HERE, in the parent, not inside the child - mirrors
        # GgufBackend.load()'s identical choice (gguf.py's "Model loaded"
        # line runs in the parent too; GgufWorker prints nothing at all).
        # The child's own stdout inherits whatever codepage the spawn gave
        # it, not the parent's console configuration - a rich/Unicode
        # console.print from inside the child hard-failed with
        # UnicodeEncodeError against a 'charmap' codec during real-model
        # verification. See _hf_worker.py's load() for the fuller note.
        mm_note = " (multimodal)" if self._supports_images else ""
        device = meta.get("device") or "?"
        console.print(f"[green]✓[/green] Model loaded{mm_note} (device: {device})")

    @staticmethod
    def _load_timeout_seconds() -> float:
        """Model-load timeout, from config (``hf_load_timeout_s``) or the generous built-in default."""
        from localm.config import load_config
        raw = load_config().get("hf_load_timeout_s")
        try:
            return float(raw or LOAD_TIMEOUT_DEFAULT)
        except (TypeError, ValueError):
            # A present-but-unparseable value (e.g. "abc", a list) is a real
            # misconfiguration, distinct from the benign missing/empty case
            # (None/0 -> the default above with no exception). Surface it
            # under --debug rather than silently masking a typo'd config
            # (rule 5).
            logger.warning("hf_load_timeout_s is set but not a valid number "
                           "(%r); using the default %.0fs", raw, LOAD_TIMEOUT_DEFAULT)
            return LOAD_TIMEOUT_DEFAULT

    @staticmethod
    def _first_token_timeout_seconds() -> float:
        """How long to wait for a reply's FIRST token, from config (``hf_first_token_timeout_s``) or the generous built-in default."""
        from localm.config import load_config
        raw = load_config().get("hf_first_token_timeout_s")
        try:
            return float(raw or FIRST_TOKEN_TIMEOUT_DEFAULT)
        except (TypeError, ValueError):
            logger.warning("hf_first_token_timeout_s is set but not a valid "
                           "number (%r); using the default %.0fs",
                           raw, FIRST_TOKEN_TIMEOUT_DEFAULT)
            return FIRST_TOKEN_TIMEOUT_DEFAULT

    @staticmethod
    def _embed_timeout_seconds() -> float:
        """Bounded wait for one embed() RPC, from config (``hf_embed_timeout_s``) or the generous built-in default - see ``_hf_runner.EMBED_TIMEOUT_DEFAULT`` for why this is sized independently from both the GGUF simple-RPC bound and the dedicated embedder's own timeout."""
        from localm.config import load_config
        raw = load_config().get("hf_embed_timeout_s")
        try:
            return float(raw or EMBED_TIMEOUT_DEFAULT)
        except (TypeError, ValueError):
            logger.warning("hf_embed_timeout_s is set but not a valid number "
                           "(%r); using the default %.0fs", raw, EMBED_TIMEOUT_DEFAULT)
            return EMBED_TIMEOUT_DEFAULT

    @staticmethod
    def _embed_max_texts() -> int:
        """Max texts accepted in one embed() call, from config (``hf_embed_max_texts``) or the built-in default - see ``_hf_runner.EMBED_MAX_TEXTS_DEFAULT`` for why this exists."""
        from localm.config import load_config
        raw = load_config().get("hf_embed_max_texts")
        try:
            return int(raw or EMBED_MAX_TEXTS_DEFAULT)
        except (TypeError, ValueError):
            logger.warning("hf_embed_max_texts is set but not a valid number "
                           "(%r); using the default %d", raw, EMBED_MAX_TEXTS_DEFAULT)
            return EMBED_MAX_TEXTS_DEFAULT

    @staticmethod
    def _embed_max_chars() -> int:
        """Max total characters, summed across every text, accepted in one embed() call, from config (``hf_embed_max_chars``) or the built-in default - see ``_hf_runner.EMBED_MAX_CHARS_DEFAULT`` for why this exists."""
        from localm.config import load_config
        raw = load_config().get("hf_embed_max_chars")
        try:
            return int(raw or EMBED_MAX_CHARS_DEFAULT)
        except (TypeError, ValueError):
            logger.warning("hf_embed_max_chars is set but not a valid number "
                           "(%r); using the default %d", raw, EMBED_MAX_CHARS_DEFAULT)
            return EMBED_MAX_CHARS_DEFAULT

    def unload(self) -> None:
        # Ask the isolated worker to close cleanly, killing it if it does not
        # exit promptly. Safe to call when the worker already crashed or was
        # never spawned - HFRunner.shutdown() is a no-op in both cases.
        if self._runner is not None:
            try:
                self._runner.shutdown()
            except Exception as e:
                # Leftover VRAM/context here can make a later load fail
                # mysteriously, so log a correlatable line rather than
                # swallowing it. Teardown is best-effort, so we still drop
                # the reference below instead of escalating to a hard
                # failure.
                logger.debug("hf worker shutdown failed (%s); its process "
                             "may not be fully torn down", type(e).__name__)
        self._runner = None
        self._loaded = False

    # ------------------------------------------------------------------ #
    #  Tokenisation                                                        #
    # ------------------------------------------------------------------ #

    def count_tokens(self, text: str) -> int:
        """Return exact token count using the loaded model's tokenizer (an RPC to the isolated worker), or the chars/4 heuristic when the worker is busy streaming or the model is not loaded yet."""
        if self.loaded and self._runner is not None:
            try:
                return self._runner.count_tokens(text)
            except RunnerBusy:
                logger.debug("hf count_tokens: worker busy with a live "
                             "stream; using the chars/4 estimate")
        return max(1, len(text) // 4)

    def count_messages_tokens(self, messages: List[dict]) -> int:
        """Return exact token count of the structured messages formatted with the HF tokenizer/processor's chat template (an RPC), or the base heuristic when the worker is busy or not loaded."""
        if self.loaded and self._runner is not None:
            try:
                return self._runner.count_messages_tokens(messages)
            except RunnerBusy:
                logger.debug("hf count_messages_tokens: worker busy with a "
                             "live stream; using the base heuristic")
        return super().count_messages_tokens(messages)

    # ------------------------------------------------------------------ #
    #  Embeddings                                                          #
    # ------------------------------------------------------------------ #

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return embedding vectors for *texts* via the isolated worker."""
        if self._runner is None or not self.loaded:
            raise RuntimeError("Model not loaded - call load() first")
        # Checked HERE, before the batch ever reaches self._runner - i.e.
        # before it crosses into the isolated worker process at all, so a
        # rejected request costs nothing beyond this check. This is the only
        # caller of HFRunner.embed()/HFWorker.embed() in production, so one
        # enforcement point is complete; no duplicate check is needed on the
        # other side of the IPC boundary.
        max_texts = self._embed_max_texts()
        if len(texts) > max_texts:
            raise EmbedBatchTooLargeError(
                f"Too many texts in one /v1/embeddings request: {len(texts)} "
                f"(max {max_texts}). Split the batch across multiple requests.")
        max_chars = self._embed_max_chars()
        total_chars = sum(len(t) for t in texts)
        if total_chars > max_chars:
            raise EmbedBatchTooLargeError(
                f"/v1/embeddings request too large: {total_chars} characters "
                f"across {len(texts)} texts (max {max_chars}). Split the "
                f"batch across multiple requests.")
        return self._runner.embed(texts, timeout=self._embed_timeout_seconds())

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    def chat_stream(
        self,
        messages: List[dict],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        grammar: Optional[str] = None,
        grammar_lazy: bool = False,
        grammar_triggers: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> Iterator[str]:
        # Checked BEFORE the loaded-state gate below, and before ever touching
        # self._runner - mirrors GgufBackend.chat_stream's identical ordering
        # exactly (gguf.py:586). This is the backend-level "source of truth"
        # guarantee the image-rejection regression test suite documents:
        # ANY caller gets a clean UnsupportedInputError for an image against
        # a text-only model, even one that never bothered to load it first,
        # not just "model not loaded". supports_images is False whenever
        # not self.loaded, so this fires correctly pre-load too.
        if messages_contain_image(messages) and not self.supports_images:
            raise UnsupportedInputError(IMAGE_UNSUPPORTED_MESSAGE)
        # Same guarantee, same reason, for a LAZY grammar this backend cannot
        # apply: xgrammar has no trigger mode, so honouring the request is
        # impossible and the worker would otherwise generate UNCONSTRAINED text
        # behind a DEBUG line (NEW-LAZY-GRAMMAR-SILENT-UNCONSTRAINED). Placed here,
        # beside the image check and BEFORE the loaded-state gate and the runner,
        # so ANY caller is refused - including one that never went through the
        # chat routes' up-front validate_grammar, and one whose model is not loaded
        # yet. validate_grammar remains the path that turns this into a clean 400
        # before either response is committed; this is the backend-level backstop.
        if grammar and grammar_lazy:
            from .base import GRAMMAR_LAZY_UNSUPPORTED_MESSAGE, GrammarUnsupportedError
            raise GrammarUnsupportedError(GRAMMAR_LAZY_UNSUPPORTED_MESSAGE)
        if self._runner is None or not self.loaded:
            raise RuntimeError("Model not loaded - call load() first")
        # Mirrors GgufBackend.chat_stream: the isolated worker computes a
        # real finish_reason (HFWorker.chat_stream, via _hf_worker.py's
        # _FinishReasonObserver) and reports it in the "done" envelope,
        # which HFRunner.chat_stream already caches as self._runner.last_done
        # (see _hf_runner.py's module docstring for the envelope shape).
        self.last_finish_reason = "stop"
        yield from self._runner.chat_stream(
            first_chunk_timeout=self._first_token_timeout_seconds(),
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            grammar=grammar,
            grammar_lazy=grammar_lazy,
            grammar_triggers=grammar_triggers,
            seed=seed,
        )
        done = getattr(self._runner, "last_done", None) or {}
        self.last_finish_reason = done.get("finish_reason", "stop")
