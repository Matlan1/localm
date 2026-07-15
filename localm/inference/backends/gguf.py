# SPDX-License-Identifier: AGPL-3.0-or-later
"""GGUF backend - drives our native ctypes wrapper around llama.dll through an
isolated worker PROCESS (see llamacpp/_runner.py and llamacpp/_worker.py).

The model's whole lifecycle (load, generate, tokenize, grammar-check, unload)
runs in a disposable child process, not here: llama_load_model_from_file (and
every later context-grow, which hits the same native call class) can
hard-abort the WHOLE PROCESS on a native CUDA/HIP driver failure - no Python
try/except can catch that. Isolating just the load call is not enough (a
model must go on to serve many later requests, and context growth is just as
abort-prone as the initial load), and isolating just a native handle back to
this process is not possible (a ctypes.c_void_p model/context pointer is
meaningless outside the process that created it) - so the isolation boundary
wraps the model's entire lifecycle. A crash in the child kills only the
child; this process reports it as a clean, catchable error and the backend
reloads fresh on the next request, exactly like today's in-process contract.

This class itself (GgufBackend) stays a thin, parent-side proxy: preflight
VRAM sizing (VramSizingMixin, shared with the child) still runs here, before
a child is even spawned, so a load that can never fit still fails fast
without paying a process-spawn cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional

from rich.console import Console

from .base import BaseBackend, ModelLoadCancelled
from .llamacpp._runner import RunnerBusy
from .llamacpp._sizing import VramSizingMixin

console = Console()


class GgufBackend(VramSizingMixin, BaseBackend):
    """
    Inference backend for GGUF model files.

    Drives our own ctypes binding to llama.dll inside an isolated worker
    process (see the module docstring) - this class never imports LlamaCpp or
    touches a native pointer itself. If the native runtime cannot be loaded,
    load() raises rather than degrading to a slower, lower-fidelity path.
    """

    def __init__(
        self,
        model_path: str,
        mmproj_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 99,
        n_ctx_max: Optional[int] = None,
        n_ctx_grow: int = 4096,
        ctx_auto: bool = False,
        n_gpu_layers_auto: bool = False,
    ) -> None:
        self.model_path = str(Path(model_path).resolve())
        self.mmproj_path = mmproj_path   # multimodal projection GGUF
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx_max = n_ctx_max       # ceiling for dynamic growth (0/None = unlimited)
        self.n_ctx_grow = n_ctx_grow
        self.ctx_auto = ctx_auto         # derive n_ctx_max from free VRAM at load
        # Auto-size how many layers go on the GPU from free VRAM at load, but only
        # when n_gpu_layers is left at its "everything" default (see
        # _effective_gpu_layers - an explicit -g is never overridden).
        self.n_gpu_layers_auto = n_gpu_layers_auto
        self.effective_ctx_max: Optional[int] = None   # resolved ceiling of the last load
        self.effective_gpu_layers: Optional[int] = None  # resolved gpu layers of the last load
        # The isolated worker process holding the real model - see
        # llamacpp/_runner.py. None until load() succeeds.
        self._runner = None
        # True once loaded, from the child's load response - supports_images
        # used to read self._llm.supports_images directly; the real LlamaCpp
        # instance now lives in the child, so this is cached instead.
        self._supports_images = False
        # Kept only for the VramSizingMixin test-monkeypatch surface (e.g.
        # test_kv_bytes_offload.py assigns a stub here to drive
        # _check_context_fit directly) - never set to a real object in
        # production; the real LlamaCpp instance lives in GgufWorker, inside
        # the child process, not here.
        self._llm = None
        self._loaded = False
        self._grammar_unsupported = False
        self._load_cancel = None         # threading.Event to abort a load mid-flight
        # One-time guard for the RAM-offload notice in _check_context_fit: a
        # card-filling model with the default grow step overflows free VRAM on
        # EVERY grow, so without this the "kept in system RAM" warning would repeat
        # on each grow of one conversation. load() clears it, so the hint fires once
        # per loaded-model session even when the SAME backend instance is reloaded
        # (Engine.chat_stream's auto-reload reuses the instance, unlike a fresh
        # switch_engine load).
        self._ram_kv_hint_shown = False

    def set_load_cancel(self, event) -> None:
        """Install (or clear with None) the cancel event honoured by load() via
        llama.cpp's native load-progress callback, so a superseded switch aborts
        the load instead of running it to completion."""
        self._load_cancel = event

    @property
    def can_be_multimodal(self) -> bool:
        """A vision GGUF needs an mmproj; only then is it worth loading the model
        to discover whether vision actually works (the HTTP route uses this to load
        before deciding to reject an image)."""
        return bool(self.mmproj_path)

    @property
    def supports_images(self) -> bool:
        """True once loaded with a working mmproj (mtmd vision, C1).

        Cached from the child's load response (self._supports_images) rather
        than read live off a real LlamaCpp instance - that instance now lives
        in the isolated worker process, not here."""
        return bool(self.loaded and self._supports_images)   # property: a dead worker has no vision

    # ------------------------------------------------------------------ #
    #  Load / unload                                                       #
    # ------------------------------------------------------------------ #
    # VRAM measurement, preflight checks, and GPU-layer/context auto-sizing
    # (_check_vram, _check_context_fit, _effective_gpu_layers, _effective_ctx_max,
    # etc.) are inherited from VramSizingMixin - see llamacpp/_sizing.py. None of
    # them touch the abort-prone native call, so they stay usable from both this
    # class and GgufWorker (the isolated child) without duplication.

    def load(self) -> None:
        # Fresh loaded-model session: let the RAM-offload notice fire once again
        # (the same instance can be reloaded via Engine.chat_stream's auto-reload).
        self._ram_kv_hint_shown = False
        # Split GGUF pre-flight: all sibling parts must be present, otherwise
        # llama.cpp fails with a cryptic native error mid-load.
        from localm.model_manager import missing_split_parts
        missing = missing_split_parts(Path(self.model_path))
        if missing:
            names = ", ".join(p.name for p in missing)
            raise FileNotFoundError(
                f"Split GGUF is incomplete - missing part(s): {names}. "
                f"Re-run 'localm pull' to download all parts."
            )
        # Resolve the effective GPU-layer count ONCE (it may probe free VRAM and
        # print a notice), then let _check_vram and _load_native both read it - so
        # the preflight and the actual load agree on how many layers go on the GPU.
        self.effective_gpu_layers = self._effective_gpu_layers()
        self._check_vram()
        try:
            self._load_native()
        except ModelLoadCancelled:
            # Deliberate abort (a newer model selection superseded this load).
            # Not a failure - propagate as-is so the caller reports "superseded",
            # never the "native runtime failed to load" error below.
            raise
        except Exception as exc:
            free = self._free_vram_bytes()
            vram_hint = ""
            if free is not None and free < self._model_bytes() + self._VRAM_OVERHEAD_BYTES:
                vram_hint = (
                    " The GPU is low on memory - free VRAM or retry with "
                    "fewer GPU layers (-g 24, or -g 0 for CPU)."
                )
            # The isolated worker itself already failed (or crashed) loading
            # the model - fail loud and actionable here too, rather than
            # silently degrading to a slower, lower-fidelity path.
            raise RuntimeError(
                f"Native llama runtime failed to load: {exc}.{vram_hint}\n"
                "Provision or repair it with  localm setup-llama  "
                "(or set LLAMA_CPP_LIB to a working llama.dll)."
            ) from exc

    def _load_native(self) -> None:
        """Load by spawning an isolated worker process and handing it the
        already-resolved parameters (see the module docstring for why this
        runs out-of-process). Preflight sizing (ctx_max/gpu_layers) stays
        here, exactly as when the native call was made in-process - none of
        it touches the abort-prone call, so it can safely run before a child
        even exists."""
        from localm.inference.backends.llamacpp._runner import ModelRunner
        from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

        vram_before = self._vram_levels()

        ctx_max = self._effective_ctx_max()
        self.effective_ctx_max = ctx_max
        # load() resolves this before _check_vram; fall back for a direct
        # _load_native() call (e.g. in tests) so the value is never None here.
        gpu_layers = self.effective_gpu_layers
        if gpu_layers is None:
            gpu_layers = self._effective_gpu_layers()
            self.effective_gpu_layers = gpu_layers

        params = dict(
            model_path=self.model_path,
            mmproj_path=self.mmproj_path,       # C1: vision via mtmd, in the child
            n_ctx=self.n_ctx,
            n_gpu_layers=gpu_layers,
            n_ctx_max=ctx_max,
            n_ctx_grow=self.n_ctx_grow,
        )
        timeout = self._load_timeout_seconds()

        cap_label = f"→{ctx_max}" if ctx_max else "→∞"
        self._runner = ModelRunner()
        with Progress(
            SpinnerColumn(),
            TextColumn("[dim]{task.description}[/dim]"),
            TimeElapsedColumn(),
            transient=True,
            console=console,
        ) as progress:
            progress.add_task(
                f"Loading model  (ctx={self.n_ctx}{cap_label}, "
                f"gpu_layers={gpu_layers})",
                total=None,
            )
            # cancel_event abort mid-load if superseded (relayed to the child
            # over its control queue - see ModelRunner.spawn_and_load).
            meta = self._runner.spawn_and_load(
                params, cancel_event=self._load_cancel, timeout=timeout)

        self._loaded = True
        self._supports_images = bool(meta.get("supports_images"))

        # Remember the model's true transformer layer count (reported once by
        # the child, the only place it is knowable) so the next load and the
        # GUI VRAM estimate can size a partial GPU offload precisely instead of
        # from _ASSUMED_LAYERS. Static model metadata, not chat/session content -
        # written regardless of privacy mode (see localm.model_meta).
        n_layers = meta.get("n_layers")
        if isinstance(n_layers, int) and n_layers > 0:
            from localm.model_meta import store_n_layers
            store_n_layers(self.model_path, n_layers)

        # VRAM usage after load - device-level driver numbers (a global,
        # cross-process view - measured from THIS process exactly as
        # accurately as from the child, since it reflects the whole GPU, not
        # a single process's allocations). torch's allocator counters
        # (memory_allocated/reserved) can only see torch's own allocations and
        # always read 0.00 for llama.dll. "in use" therefore includes every
        # process on the GPU; the delta is what this load itself consumed.
        for i, (free, total) in enumerate(self._vram_levels()):
            used = (total - free) / 1024**3
            line = (f"  vram     : {used:.2f} GB in use / "
                    f"{total / 1024**3:.2f} GB total (device {i}")
            if i < len(vram_before):
                delta = (vram_before[i][0] - free) / 1024**3
                line += f", {delta:+.2f} GB this load"
            console.print(f"[dim]{line})[/dim]")

        console.print("[green]✓[/green] Model loaded")

    @staticmethod
    def _load_timeout_seconds() -> float:
        """Model-load timeout, from config (``gguf_load_timeout_s``) or the
        generous built-in default. Unlike the VRAM-probe daemon's short
        bounded wait (which has a safe "unmeasurable, skip" fallback), a
        stalled model load has no safe default - see ModelRunner.spawn_and_load
        for why this always raises rather than silently reporting not-loaded.
        Configurable because a multi-GB model on a slow disk can legitimately
        take minutes, and that varies far more by install than a fixed
        constant could ever cover."""
        from localm.inference.backends.llamacpp._runner import LOAD_TIMEOUT_DEFAULT
        from localm.config import load_config
        raw = load_config().get("gguf_load_timeout_s")
        try:
            return float(raw or LOAD_TIMEOUT_DEFAULT)
        except (TypeError, ValueError):
            # A present-but-unparseable value (e.g. "abc", a list) is a real
            # misconfiguration, distinct from the benign missing/empty case
            # (None/0 -> the default above with no exception). Surface it under
            # --debug rather than silently masking a typo'd config as if the
            # user had simply not set it (rule 5).
            from localm.debuglog import logger as _dbg
            _dbg.warning("gguf_load_timeout_s is set but not a valid number "
                         "(%r); using the default %.0fs", raw, LOAD_TIMEOUT_DEFAULT)
            return LOAD_TIMEOUT_DEFAULT

    @staticmethod
    def _first_token_timeout_seconds() -> float:
        """How long to wait for a reply's FIRST token, from config
        (``gguf_first_token_timeout_s``) or the generous built-in default.
        Configurable for the same reason as the load timeout above: it covers
        prompt PREFILL, whose duration varies enormously by install (CPU vs GPU,
        partial offload, prompt length) - far more than a fixed constant could
        cover. See FIRST_TOKEN_TIMEOUT_DEFAULT for why this is not the per-token
        ceiling."""
        from localm.inference.backends.llamacpp._runner import FIRST_TOKEN_TIMEOUT_DEFAULT
        from localm.config import load_config
        raw = load_config().get("gguf_first_token_timeout_s")
        try:
            return float(raw or FIRST_TOKEN_TIMEOUT_DEFAULT)
        except (TypeError, ValueError):
            # Same rule-5 reasoning as _load_timeout_seconds above: a typo'd
            # value is a real misconfiguration, not a silent fall-through.
            from localm.debuglog import logger as _dbg
            _dbg.warning("gguf_first_token_timeout_s is set but not a valid "
                         "number (%r); using the default %.0fs",
                         raw, FIRST_TOKEN_TIMEOUT_DEFAULT)
            return FIRST_TOKEN_TIMEOUT_DEFAULT

    def unload(self) -> None:
        # Ask the isolated worker to close cleanly (native teardown + its
        # stderr suppression happen there), killing it if it does not exit
        # promptly. Safe to call when the worker already crashed or was never
        # spawned - ModelRunner.shutdown() is a no-op in both cases.
        if self._runner is not None:
            try:
                self._runner.shutdown()
            except Exception as e:
                # Leftover VRAM/context here can make a later load fail
                # mysteriously, so log a correlatable line rather than
                # swallowing it. Teardown is best-effort, so we still drop the
                # reference below instead of escalating to a hard failure.
                from localm.debuglog import logger as _dbg
                _dbg.debug("gguf worker shutdown failed (%s); its process may "
                           "not be fully torn down", type(e).__name__)
        self._runner = None
        self._llm = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        # A backend whose isolated worker is GONE is not loaded, whatever
        # _loaded says. The worker can be killed out from under this object on
        # paths that never run unload(): _cancel_stream_and_drain kills it when
        # a mid-stream cancel is not confirmed within its drain timeout, and
        # _simple_request kills it on its own timeout - both call
        # ModelRunner.shutdown(), which nulls the runner's queues and process.
        # Neither raises RuntimeError into chat_stream's handler below
        # (GeneratorExit is not a RuntimeError), so _loaded stayed True next to
        # a dead runner. Engine.chat_stream then skipped its auto-reload and
        # called straight into the dead runner, whose first act is
        # self._req_q.put(...) on None -> AttributeError, which is not caught or
        # unloaded either, so EVERY later request to this model repeated it
        # until the model was manually evicted or the server restarted.
        # Reporting the truth here makes the next request reload cleanly instead
        # (load() builds a fresh ModelRunner) - REG-606.
        if not self._loaded:
            return False
        is_alive = getattr(self._runner, "is_alive", None)
        return True if is_alive is None else bool(is_alive())

    def validate_grammar(self, grammar: Optional[str]) -> None:
        """Raise :class:`InvalidGrammarError` for a malformed GBNF string, up front,
        so a bad grammar is a clean 400 rather than a native fault that would latch
        _grammar_unsupported and silently strip grammar from later requests. No-op
        when not loaded (no vocab to parse against) or when *grammar* is empty."""
        if grammar and self.loaded and self._runner is not None:   # property, see count_tokens
            try:
                self._runner.check_grammar(grammar)
            except RunnerBusy:
                # A generation is streaming on this model right now and holds the
                # worker's response queue. validate_grammar is called
                # synchronously on the server's async event loop (before the
                # per-model inference semaphore), so blocking here would freeze
                # the whole loop for the length of that generation (HON-02
                # review finding). Defer the check: when generation builds the
                # sampler it rejects a malformed grammar with the SAME clean
                # InvalidGrammarError (llama.py raises it on the native NULL
                # return, before any token) - so deferring only moves a bad
                # grammar from an up-front 400 to a generation-time error, never
                # a native fault or a latched degrade.
                from localm.debuglog import logger as _dbg
                _dbg.debug("gguf validate_grammar: worker busy with a live "
                           "stream; deferring grammar validation to generation time")

    # ------------------------------------------------------------------ #
    #  Tokenisation                                                        #
    # ------------------------------------------------------------------ #

    def count_tokens(self, text: str) -> int:
        """Return exact token count using the loaded model's vocabulary (an
        RPC to the isolated worker), or the chars/4 heuristic when the worker
        is busy streaming or the model is not loaded yet."""
        # `self.loaded`, NOT the raw `self._loaded`: the property is what knows the
        # worker is gone. _simple_request kills it on its own timeout (and the
        # cancel-drain does the same), nulling the queues while _loaded stays True -
        # so gating on the attribute here called straight into a dead runner and
        # `self._req_q.put(...)` raised AttributeError on None, which is not
        # RunnerBusy and so escaped uncaught, on every later count (REG-606).
        if self.loaded and self._runner is not None:
            try:
                return self._runner.count_tokens(text)
            except RunnerBusy:
                # A generation is streaming on this model right now and holds
                # the worker's response queue (HON-02). Rather than block this
                # request's token accounting behind a whole generation - or risk
                # its RPC reply racing the live stream's envelopes - fall back to
                # the documented chars/4 estimate immediately and say so under
                # --debug.
                from localm.debuglog import logger as _dbg
                _dbg.debug("gguf count_tokens: worker busy with a live stream; "
                           "using the chars/4 estimate")
        # Not loaded yet, or the worker is mid-stream - chars/4 heuristic. A
        # genuine RPC failure (worker crash/timeout) is NOT swallowed here: it
        # propagates, preserving this method's existing contract.
        return max(1, len(text) // 4)

    def count_messages_tokens(self, messages: List[dict]) -> int:
        """Return exact token count of the structured messages formatted with
        the model's embedded chat template (an RPC to the isolated worker,
        which alone holds the native model pointer the template needs)."""
        if self.loaded and self._runner is not None:   # the property - see count_tokens
            try:
                return self._runner.count_messages_tokens(messages)
            except RunnerBusy:
                # Worker busy with a live stream (HON-02) - use the heuristic
                # rather than queue behind the generation. See count_tokens.
                from localm.debuglog import logger as _dbg
                _dbg.debug("gguf count_messages_tokens: worker busy with a live "
                           "stream; using the heuristic estimate")
            except Exception as e:
                # An unexpected RPC failure (worker crash/timeout, an encode
                # error): the super() return below is then a heuristic ESTIMATE,
                # not an exact count, and context-budgeting downstream is
                # trusting an approximation - so surface it under --debug rather
                # than swallowing it silently (rule 5).
                from localm.debuglog import logger as _dbg
                _dbg.debug("gguf count_messages_tokens RPC failed (%s); using "
                           "the heuristic estimate", type(e).__name__)
        return super().count_messages_tokens(messages)

    # ------------------------------------------------------------------ #
    #  Embeddings                                                          #
    # ------------------------------------------------------------------ #

    # The native ctypes binding (localm.inference.backends.llamacpp) does not
    # expose create_embedding, so this backend cannot produce embeddings. This
    # flag lets callers (the MCP tools/list and /v1/embeddings) decide NOT to
    # advertise an embed capability that would always raise (FAC-6), without
    # having to load a model first.
    can_embed: bool = False

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self._loaded:
            raise RuntimeError("Model not loaded - call load() first")
        # The isolated worker's GgufWorker never exposes create_embedding (the
        # ctypes binding does not implement it) - this call always raises, so
        # there is no RPC to make; see can_embed above and the GGUFEmbedder
        # class (localm/inference/embedder.py) for the real embedding path.
        raise NotImplementedError(
            "Embeddings are not supported by the built-in GGUF binding yet. "
            "Use a HuggingFace-format model for /v1/embeddings."
        )

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
        grammar_triggers: Optional[list] = None,
        seed: Optional[int] = None,
    ) -> Iterator[str]:
        # Image input: when an mmproj is loaded (mtmd vision, C1) it flows through
        # to create_chat_completion's image path. Otherwise the model is text-only,
        # so refuse the image rather than drop it and answer about a picture the
        # model never received.
        from .base import IMAGE_UNSUPPORTED_MESSAGE, UnsupportedInputError, messages_contain_image
        if messages_contain_image(messages) and not self.supports_images:
            raise UnsupportedInputError(IMAGE_UNSUPPORTED_MESSAGE)

        # Safety net for a native build whose grammar sampler genuinely faults
        # at sample time (a C++ exception across the C ABI) without harming
        # the loaded model: skip grammar up-front once seen and generate
        # unconstrained - the same soft-degrade contract the HF backend
        # offers - so a grammar request never breaks chat. NOTE: the fault
        # this path was originally written for turned out to be OUR bug (a
        # redundant llama_sampler_accept after llama_sampler_sample, fixed in
        # llama.py's _generate) - grammar works on the bundled build now; this
        # stays as a fallback for truly grammar-less builds.
        if grammar and getattr(self, "_grammar_unsupported", False):
            console.print(
                "[yellow]grammar is not supported by this native llama build; "
                "generating without constraint.[/yellow]"
            )
            grammar = None

        kwargs: dict = dict(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            grammar=grammar,
            grammar_lazy=grammar_lazy,
            grammar_triggers=grammar_triggers,
        )
        if seed is not None:
            kwargs["seed"] = seed

        # The grammar-fault retry-without-grammar logic (a genuinely
        # recoverable native OSError, not a process abort) now runs INSIDE the
        # isolated worker (GgufWorker.chat_stream) - it already has the real
        # model, so retrying there is a plain in-process call, no round trip.
        # This method just relays the resulting stream and, on a normal
        # finish, re-applies the worker's report of that decision to this
        # instance's OWN persistent policy state (_grammar_unsupported must
        # survive across many calls on the same backend, so it stays here).
        #
        # yield from (not a manual for-loop) is required so that closing THIS
        # generator (http_server.py's mid-stream cancel, unchanged) correctly
        # forwards GeneratorExit into the runner's generator too - that is
        # what triggers ModelRunner.chat_stream's own cancel-and-drain cleanup
        # instead of abandoning the runner's generator mid-flight.
        self.last_finish_reason = "stop"
        try:
            yield from self._runner.chat_stream(
                first_chunk_timeout=self._first_token_timeout_seconds(), **kwargs)
        except RuntimeError:
            # The isolated worker crashed or stalled (a real native abort, or
            # an unrecoverable fault GgufWorker.chat_stream deliberately left
            # uncaught) - the model is gone; without this, every later
            # request would hit a dead process. Drop it so the next request
            # triggers a clean reload, matching the in-process contract this
            # replaces exactly (same user-facing message and effect).
            self.last_finish_reason = "error"
            from localm.debuglog import logger as _dbg
            _dbg.exception("native inference fault - dropping model instance")
            try:
                self.unload()
            except Exception:
                self._runner = None
                self._llm = None
                self._loaded = False
            raise
        else:
            done = getattr(self._runner, "last_done", None) or {}
            self.last_finish_reason = done.get("finish_reason", "stop")
            if done.get("grammar_unsupported") and not self._grammar_unsupported:
                # First time this model has shown it can't do grammar: latch
                # it (future calls skip sending grammar at all, see the
                # up-front check above) and surface the same degrade notice
                # the in-process path always printed - a degrade must stay
                # visible even after crossing a process boundary.
                self._grammar_unsupported = True
                from localm.debuglog import logger as _dbg
                _dbg.warning("native grammar sampler faulted; degrading to "
                             "unconstrained generation")
                console.print(
                    "[yellow]grammar is not supported by this native llama "
                    "build; generating without constraint.[/yellow]"
                )
