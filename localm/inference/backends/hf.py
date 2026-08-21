# SPDX-License-Identifier: AGPL-3.0-or-later
"""HuggingFace Transformers backend - parent-side proxy.

Drives an isolated child process (see ``_hf_runner.py``'s module docstring)
that owns one real ``HFWorker`` (``_hf_worker.py``) for its whole lifetime -
this class never imports torch/transformers or touches a model handle
itself. Mirrors ``backends/gguf.py``'s ``GgufBackend`` (PR #606) applied to
the one backend that was still fully in-process: every HF native call
(tokenizer regex, a torch forward pass, ``model.generate()``) is
uninterruptible from Python, so a hang used to burn a slot in the server's
shared thread pool PERMANENTLY (see ``dev-notes/decisions-2026-07-30-release-
gate.md``, Q2, and ``_hf_runner.py``'s module docstring for the full
mechanism). Isolating it is what makes a hang killable without a restart.

``create_backend()`` (``engine.py``) needs no changes: this class keeps the
exact same import path, class name, and constructor signature the in-process
version had - the whole ``BaseBackend`` public contract is preserved.
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
    owner-only to set) - see ``_check_custom_code_allowed`` below for the
    full history/rationale. A second, independent copy of this exact
    function lives in ``_hf_worker.py``, needed there for the
    ``trust_remote_code=`` kwarg every ``from_pretrained`` call takes - see
    that copy's docstring for why this is deliberately duplicated rather
    than cross-imported between the parent and child modules."""
    try:
        from localm.config import load_config
        return bool(load_config().get("hf_trust_remote_code", False))
    except Exception as e:
        # Fail CLOSED: an unreadable config must never be read as permission to
        # execute a model's bundled code. Logged rather than silent (rule 5).
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
    """Refuse, with an actionable message, a model that needs custom code when
    ``hf_trust_remote_code`` is off. No-op for an ordinary model.

    Called at the TOP of load(), deliberately BEFORE spawning a child (and,
    when this ran in-process, before ``_require_torch()`` - a torch-less
    GGUF-only build must refuse identically, not differ by build). Now runs
    in the PARENT rather than inside the isolated worker: this needs no
    torch/transformers and nothing only the child can see, so gatekeeping it
    here means a refused model never pays the cost of spawning a child at
    all - the same reasoning as ``validate_tokenizer_json`` right below it in
    ``load()``.
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
        """A backend whose isolated worker is GONE is not loaded, whatever
        ``_loaded`` says - mirrors ``GgufBackend.loaded`` exactly (REG-606):
        the worker can die out from under this object on a timeout or a
        crash without any exception ever reaching a caller that would clear
        ``_loaded`` (a killed ``chat_stream`` generator raises
        ``GeneratorExit``, not a catchable error). Reporting the truth here
        is what makes ``Engine.chat_stream``/``Engine.embed``'s ``if not
        self._backend.loaded: self._backend.load()`` auto-reload actually
        fire, instead of calling straight into a dead runner."""
        if not self._loaded:
            return False
        is_alive = getattr(self._runner, "is_alive", None)
        return True if is_alive is None else bool(is_alive())

    @property
    def supports_images(self) -> bool:
        """True once loaded with a working multimodal processor.

        Cached from the child's load response (self._supports_images) rather
        than read live off a real HFWorker instance - that instance now
        lives in the isolated worker process, not here. Gated on ``loaded``
        (a live worker check, mirroring ``GgufBackend.supports_images``): a
        dead worker safely reports no vision rather than a stale cached
        True - the caller (routes/chat.py) already reloads-then-rechecks
        when `not engine.loaded and engine.can_be_multimodal`, so this never
        strands a genuinely multimodal model, it only ever costs one extra
        reload."""
        return bool(self.loaded and self._supports_images)

    @property
    def can_embed(self) -> bool:
        """True only when the LOADED model is a GENUINE embedding model (an
        architectural fact about the checkpoint, computed once by the child
        right after load - see HFWorker.can_embed for the full reasoning
        and the measurements behind it).

        Deliberately NOT gated on current liveness, unlike supports_images
        above - this is a real behavioral difference, not an oversight.
        Engine.embed()'s flow is: check can_embed -> reload if not loaded ->
        check can_embed again -> embed(). If this gated on liveness, a
        confirmed real HF embedder whose worker merely crashed between calls
        would report can_embed=False at the FIRST check (before the reload
        attempt ever runs) and get silently, permanently rerouted to the
        dedicated on-device embedder instead of ever being reloaded - a
        worse outcome than the transient crash itself, since the user's
        actually-configured embedding model would stop being used. The
        underlying architectural fact (does this checkpoint's structure
        support embedding) does not change when the worker dies, so the
        cached value stays valid across a crash; Engine.embed()'s own
        `not self._backend.loaded` check is what triggers the reload."""
        return self._can_embed

    @property
    def supports_grammar(self) -> bool:
        """True only when xgrammar (the optional ``[grammar]`` extra) is
        installed, because that package IS this backend's grammar support.

        Without it ``_grammar_processor`` in the worker logs a warning and
        returns no logits processor, so generation runs UNCONSTRAINED while the
        caller still gets a normal 200. Reporting True here regardless of the
        extra would keep that silent degrade alive behind an honest-looking
        capability flag, which is worse than the original bug.

        Deliberately NOT gated on ``loaded``, unlike ``supports_images`` above:
        this is a fact about the INSTALL, not about the checkpoint or the worker,
        and the up-front request check in the chat routes runs before a model is
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
            # A broken/partial install: the parent package is missing or the
            # module has no spec. Either way the child's own import would fail
            # too, so the honest answer is the same one it would produce.
            return False

    def validate_grammar(self, grammar: Optional[str], *, lazy: bool = False) -> None:
        """Refuse a LAZY grammar up front; defer everything else to the base.

        xgrammar - which IS this backend's grammar support, see
        :attr:`supports_grammar` - has no lazy/trigger mode at all. Its compiled
        matcher masks logits from the first token or not at all, so there is
        nothing to feed a trigger pattern to. That is a static fact about the
        library, not about the checkpoint, the install or the worker, so it is
        knowable HERE, in the parent, with no probe and no side effect. That is
        exactly what the GGUF backend cannot do for its own lazy support (see
        ``BaseBackend.validate_grammar`` for the measurement), which is why this
        refusal lives at the one site that can prove its answer.

        Refusing beats the alternative it replaces. The worker used to drop the
        grammar and generate UNCONSTRAINED text behind a DEBUG line, so a caller
        that asked for constrained output got a normal 200 it could not tell from
        a grammar-conformant answer. Raising here happens before a byte of either
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
        """Model-load timeout, from config (``hf_load_timeout_s``) or the
        generous built-in default. Mirrors ``GgufBackend._load_timeout_seconds``
        exactly - a stalled load has no safe "unmeasurable" fallback, so this
        always raises rather than silently reporting not-loaded. Configurable
        because HF loads read full-precision safetensors from disk (no
        quantized-mmap fast path), so a large checkpoint on slow storage can
        legitimately need longer than the generous default."""
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
        ``_hf_runner.EMBED_MAX_TEXTS_DEFAULT`` for why this exists."""
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
        embedding path for a chat decoder (see HFWorker.embed's docstring
        for the measurements) - Engine.embed routes those to the dedicated
        on-device embedder instead.
        """
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
