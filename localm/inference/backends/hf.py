# SPDX-License-Identifier: AGPL-3.0-or-later
"""HuggingFace Transformers backend - parent-side proxy.

Drives an isolated child process (see ``_hf_runner.py``'s module docstring)
that owns one real ``HFWorker`` (``_hf_worker.py``) for its whole lifetime -
this class never imports torch/transformers or touches a model handle
itself. Mirrors ``backends/gguf.py``'s ``GgufBackend``. Every HF native call
(tokenizer regex, a torch forward pass, ``model.generate()``) is
uninterruptible from Python, so a hang in this process would burn a slot in
the server's shared thread pool PERMANENTLY (see ``_hf_runner.py``'s module
docstring for the full mechanism). Isolating it is what makes a hang killable
without a restart.

``create_backend()`` (``engine.py``) needs no changes: this class keeps the
import path, class name and constructor signature that function expects - the
whole ``BaseBackend`` public contract is preserved.
"""

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
    """Whether transformers may import and execute a model directory's own .py.

    Config-driven and DEFAULT OFF (config key ``hf_trust_remote_code``,
    owner-only to set) - see ``_check_custom_code_allowed`` below for the full
    contract. A second, independent copy of this exact function lives in
    ``_hf_worker.py``, needed there for the ``trust_remote_code=`` kwarg every
    ``from_pretrained`` call takes; see that copy's docstring for the
    duplication contract between the parent and child modules."""
    try:
        from localm.config import load_config
        return bool(load_config().get("hf_trust_remote_code", False))
    except Exception as e:
        # Fail CLOSED: an unreadable config is not permission to execute a model's
        # bundled code. Logged rather than silent.
        logger.debug("hf: could not read hf_trust_remote_code, assuming off: %s", e)
        return False


def _declares_custom_code(model_path: str) -> bool:
    """True when this model directory asks transformers to run its own Python.

    That is what an ``auto_map`` entry means: a class is loaded from a .py inside
    the directory. Read from the on-disk JSON only - nothing is imported here.
    """
    import json
    p = Path(model_path)
    if not p.is_dir():
        return False
    # Every file transformers consults for auto_map. preprocessor_config.json
    # (with the pre) is the one AutoProcessor reads.
    for fname in ("config.json", "tokenizer_config.json",
                  "preprocessor_config.json", "processor_config.json",
                  "video_preprocessor_config.json"):
        f = p / fname
        if not f.is_file():
            continue
        try:
            # Any non-empty auto_map counts, including the legacy list/tuple form
            # transformers still accepts.
            if json.loads(f.read_text(encoding="utf-8")).get("auto_map"):
                return True
        except Exception as e:
            # A malformed or unreadable file is evidence neither of safety nor of
            # custom code; the real load fails on it later with a better message.
            # Logged rather than decided silently.
            logger.debug("hf: could not parse %s while checking for custom "
                         "code: %s", f, e)
    return False


def _check_custom_code_allowed(model_path: str) -> None:
    """Refuse, with an actionable message, a model that needs custom code when
    ``hf_trust_remote_code`` is off. No-op for an ordinary model.

    Called at the TOP of load(), BEFORE spawning a child, so a refused model
    never pays the cost of spawning one. It needs no torch/transformers and
    nothing only the child can see, so it runs in the PARENT - the same
    placement as ``validate_tokenizer_json`` right below it in ``load()``. A
    torch-less GGUF-only build refuses identically.
    """
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
    """
    Parent-side handle to a HuggingFace-format model loaded in an isolated
    child process.

    Multimodal detection is automatic: if the model directory ships a processor
    that handles images/audio, multimodal content in messages is handled.
    If the model only has a tokenizer, image/audio parts are silently dropped
    and only text is passed to the model.
    """

    # An HF checkpoint may ship an image processor; whether this instance can
    # actually see images is only known after load().
    can_be_multimodal = True

    def __init__(self, model_path: str, device: Optional[str] = None) -> None:
        self.model_path = str(Path(model_path).resolve())
        self._device = device      # None = auto-detect
        # The isolated worker process holding the real model. None until load()
        # succeeds.
        self._runner: Optional[HFRunner] = None
        self._loaded = False
        # Cached from the child's load response; the real HFWorker lives in the
        # child and cannot be read live.
        self._supports_images = False
        self.effective_ctx_max: Optional[int] = None
        self.n_ctx_max: Optional[int] = None
        # Unloaded means True (unknown, load to find out). Unlike supports_images,
        # this is not gated on current liveness once known.
        self._can_embed = True

    @property
    def loaded(self) -> bool:
        """A backend whose isolated worker is GONE is not loaded, whatever
        ``_loaded`` says - mirrors ``GgufBackend.loaded`` exactly: the worker
        can die out from under this object on a timeout or a crash without any
        exception ever reaching a caller that would clear ``_loaded`` (a killed
        ``chat_stream`` generator raises ``GeneratorExit``, not a catchable
        error). Reporting the truth here is what makes
        ``Engine.chat_stream``/``Engine.embed``'s ``if not
        self._backend.loaded: self._backend.load()`` auto-reload actually fire,
        instead of calling straight into a dead runner."""
        if not self._loaded:
            return False
        is_alive = getattr(self._runner, "is_alive", None)
        return True if is_alive is None else bool(is_alive())

    @property
    def supports_images(self) -> bool:
        """True once loaded with a working multimodal processor.

        Cached from the child's load response (self._supports_images): the real
        HFWorker instance lives in the isolated worker process, not here. Gated
        on ``loaded`` (a live worker check, mirroring
        ``GgufBackend.supports_images``), so a dead worker reports no vision
        rather than a stale cached True. The caller (routes/chat.py) reloads and
        rechecks when `not engine.loaded and engine.can_be_multimodal`, so this
        never strands a genuinely multimodal model and costs at most one extra
        reload."""
        return bool(self.loaded and self._supports_images)

    @property
    def can_embed(self) -> bool:
        """True only when the LOADED model is a GENUINE embedding model (an
        architectural fact about the checkpoint, computed once by the child
        right after load - see HFWorker.can_embed).

        NOT gated on current liveness, unlike supports_images above. The
        architectural fact does not change when the worker dies, so the cached
        value stays valid across a crash, and Engine.embed()'s own
        `not self._backend.loaded` check is what triggers the reload. Gating
        this on liveness would report False at Engine.embed()'s FIRST check -
        before the reload attempt ever runs - and permanently reroute a
        confirmed HF embedder to the dedicated on-device embedder."""
        return self._can_embed

    @property
    def supports_grammar(self) -> bool:
        """True only when xgrammar (the optional ``[grammar]`` extra) is
        installed, because that package IS this backend's grammar support.

        Without it ``_grammar_processor`` in the worker logs a warning and
        returns no logits processor, so generation runs UNCONSTRAINED while the
        caller still gets a normal 200. Reporting True here regardless of the
        extra would hide that silent degrade behind an honest-looking capability
        flag.

        NOT gated on ``loaded``, unlike ``supports_images`` above: this is a
        fact about the INSTALL, not about the checkpoint or the worker, and the
        up-front request check in the chat routes runs before a model is
        necessarily loaded. Gating it on liveness would refuse a grammar request
        that the backend can serve perfectly well after the auto-reload.

        ``find_spec`` rather than a real import: importing xgrammar pulls in
        torch and costs seconds, and this runs on the event loop for every
        grammar request. The worker child runs the same interpreter out of the
        same environment, so importability here predicts importability there.
        """
        import importlib.util
        try:
            return importlib.util.find_spec("xgrammar") is not None
        except (ImportError, ValueError):
            # A broken or partial install: the parent package is missing or the
            # module has no spec. The child's own import would fail too.
            return False

    def validate_grammar(self, grammar: Optional[str], *, lazy: bool = False) -> None:
        """Refuse a LAZY grammar up front; defer everything else to the base.

        xgrammar - which IS this backend's grammar support, see
        :attr:`supports_grammar` - has no lazy/trigger mode at all. Its compiled
        matcher masks logits from the first token or not at all, so there is
        nothing to feed a trigger pattern to. That is a static fact about the
        library, not about the checkpoint, the install or the worker, so it is
        knowable HERE, in the parent, with no probe and no side effect - which is
        what the GGUF backend cannot do for its own lazy support (see
        ``BaseBackend.validate_grammar``).

        Without this refusal the worker drops the grammar and generates
        UNCONSTRAINED text behind a DEBUG line, so a caller that asked for
        constrained output gets a normal 200 it cannot tell from a
        grammar-conformant answer. Raising here happens before a byte of either
        the streaming or the non-streaming response is committed, so both paths
        get the identical status and the identical reason.

        NOT folded into the ``supports_grammar`` denial in the base: this backend
        may well support plain grammar (with the extra installed), and telling the
        caller to install an extra they already have would send them to fix the
        wrong thing. ``GRAMMAR_LAZY_UNSUPPORTED_MESSAGE`` names the lazy mode and
        offers the two recoveries that actually apply.
        """
        from .base import GRAMMAR_LAZY_UNSUPPORTED_MESSAGE, GrammarUnsupportedError
        super().validate_grammar(grammar, lazy=lazy)
        if grammar and lazy:
            raise GrammarUnsupportedError(GRAMMAR_LAZY_UNSUPPORTED_MESSAGE)

    # ------------------------------------------------------------------ #
    #  Load / unload                                                       #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        # Two pre-flight refusals, before a child is ever spawned:
        #   1. Custom code (auto_map) the user has not explicitly trusted.
        #   2. A tokenizer.json regex pattern that fails the Oniguruma safety
        #      probe.
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
        # Printed in the parent, not the child: the child's stdout inherits the
        # codepage the spawn gave it rather than the parent's console
        # configuration, and a rich/Unicode console.print from the child raises
        # UnicodeEncodeError.
        mm_note = " (multimodal)" if self._supports_images else ""
        device = meta.get("device") or "?"
        console.print(f"[green]✓[/green] Model loaded{mm_note} (device: {device})")

    @staticmethod
    def _load_timeout_seconds() -> float:
        """Model-load timeout, from config (``hf_load_timeout_s``) or the
        generous built-in default. Mirrors ``GgufBackend._load_timeout_seconds``
        exactly: a stalled load has no safe "unmeasurable" fallback, so this
        always raises rather than silently reporting not-loaded. Configurable
        because HF loads read full-precision safetensors from disk (no
        quantized-mmap fast path), so a large checkpoint on slow storage can
        legitimately need longer than the built-in default."""
        from localm.config import load_config
        raw = load_config().get("hf_load_timeout_s")
        try:
            return float(raw or LOAD_TIMEOUT_DEFAULT)
        except (TypeError, ValueError):
            # A present-but-unparseable value is a misconfiguration, distinct from
            # the benign missing/empty case (None or 0 uses the default above).
            logger.warning("hf_load_timeout_s is set but not a valid number "
                           "(%r); using the default %.0fs", raw, LOAD_TIMEOUT_DEFAULT)
            return LOAD_TIMEOUT_DEFAULT

    @staticmethod
    def _first_token_timeout_seconds() -> float:
        """How long to wait for a reply's FIRST token, from config
        (``hf_first_token_timeout_s``) or the generous built-in default.
        Mirrors ``GgufBackend._first_token_timeout_seconds`` exactly - see
        its docstring for why this covers prompt PREFILL, not one token's
        decode."""
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
        """Bounded wait for one embed() RPC, from config
        (``hf_embed_timeout_s``) or the generous built-in default - see
        ``_hf_runner.EMBED_TIMEOUT_DEFAULT`` for why this is sized
        independently from both the GGUF simple-RPC bound and the dedicated
        embedder's own timeout."""
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
        """Max texts accepted in one embed() call, from config
        (``hf_embed_max_texts``) or the built-in default - see
        ``_hf_runner.EMBED_MAX_TEXTS_DEFAULT``."""
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
        """Max total characters, summed across every text, accepted in one
        embed() call, from config (``hf_embed_max_chars``) or the built-in
        default - see ``_hf_runner.EMBED_MAX_CHARS_DEFAULT`` for why this
        exists."""
        from localm.config import load_config
        raw = load_config().get("hf_embed_max_chars")
        try:
            return int(raw or EMBED_MAX_CHARS_DEFAULT)
        except (TypeError, ValueError):
            logger.warning("hf_embed_max_chars is set but not a valid number "
                           "(%r); using the default %d", raw, EMBED_MAX_CHARS_DEFAULT)
            return EMBED_MAX_CHARS_DEFAULT

    def unload(self) -> None:
        # Ask the isolated worker to close cleanly, killing it if it does not exit
        # promptly. A no-op when the worker already crashed or was never spawned.
        if self._runner is not None:
            try:
                self._runner.shutdown()
            except Exception as e:
                # Teardown is best-effort: log a correlatable line and drop the
                # reference below rather than escalating to a hard failure.
                logger.debug("hf worker shutdown failed (%s); its process "
                             "may not be fully torn down", type(e).__name__)
        self._runner = None
        self._loaded = False

    # ------------------------------------------------------------------ #
    #  Tokenisation                                                        #
    # ------------------------------------------------------------------ #

    def count_tokens(self, text: str) -> int:
        """Return exact token count using the loaded model's tokenizer (an
        RPC to the isolated worker), or the chars/4 heuristic when the
        worker is busy streaming or the model is not loaded yet. Mirrors
        ``GgufBackend.count_tokens`` exactly."""
        if self.loaded and self._runner is not None:
            try:
                return self._runner.count_tokens(text)
            except RunnerBusy:
                logger.debug("hf count_tokens: worker busy with a live "
                             "stream; using the chars/4 estimate")
        return max(1, len(text) // 4)

    def count_messages_tokens(self, messages: List[dict]) -> int:
        """Return exact token count of the structured messages formatted
        with the HF tokenizer/processor's chat template (an RPC), or the
        base heuristic when the worker is busy or not loaded. Mirrors
        ``GgufBackend.count_messages_tokens``'s shape."""
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
        """
        Return embedding vectors for *texts* via the isolated worker.
        Callers must gate on ``can_embed`` above: this is NOT a valid
        embedding path for a chat decoder (see HFWorker.embed's docstring) -
        Engine.embed routes those to the dedicated on-device embedder instead.
        """
        if self._runner is None or not self.loaded:
            raise RuntimeError("Model not loaded - call load() first")
        # Checked before the batch reaches self._runner, so a rejected request
        # never crosses into the isolated worker process. This is the only caller
        # of HFRunner.embed() in production, so it is the single enforcement point.
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
        # Checked before the loaded-state gate below and before touching
        # self._runner, so any caller gets a clean UnsupportedInputError for an
        # image against a text-only model rather than a not-loaded error.
        # supports_images is False whenever not self.loaded, so this fires pre-load
        # too.
        if messages_contain_image(messages) and not self.supports_images:
            raise UnsupportedInputError(IMAGE_UNSUPPORTED_MESSAGE)
        # The same check for a LAZY grammar this backend cannot apply: xgrammar has
        # no trigger mode, so the worker would otherwise generate UNCONSTRAINED
        # text. Placed before the loaded-state gate and the runner, so any caller is
        # refused, including one that skipped the routes' up-front
        # validate_grammar and one whose model is not loaded yet.
        if grammar and grammar_lazy:
            from .base import GRAMMAR_LAZY_UNSUPPORTED_MESSAGE, GrammarUnsupportedError
            raise GrammarUnsupportedError(GRAMMAR_LAZY_UNSUPPORTED_MESSAGE)
        if self._runner is None or not self.loaded:
            raise RuntimeError("Model not loaded - call load() first")
        # The isolated worker computes a real finish_reason and reports it in the
        # done envelope, which HFRunner.chat_stream caches as
        # self._runner.last_done.
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
