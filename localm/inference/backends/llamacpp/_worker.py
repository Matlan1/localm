# SPDX-License-Identifier: AGPL-3.0-or-later
"""``GgufWorker`` - owns the real native model."""

from __future__ import annotations

from typing import List, Optional

from ._sizing import VramSizingMixin


class GgufWorker(VramSizingMixin):
    """The real, native-call-owning half of the GGUF backend."""

    def __init__(
        self,
        model_path: str,
        mmproj_path: Optional[str],
        n_ctx: int,
        n_gpu_layers: int,
        n_ctx_max: Optional[int],
        n_ctx_grow: int,
        cancel_event=None,
        vram_overhead_bytes: Optional[int] = None,
        gpu_split_ratios: Optional[list] = None,
        n_cpu_moe: int = 0,
        mtp_enabled: bool = True,
    ) -> None:
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.mtp_enabled = mtp_enabled
        # Already resolved by the parent - VramSizingMixin's _check_context_fit
        # reads this in preference to n_gpu_layers, matching GgufBackend's shape.
        self.effective_gpu_layers = n_gpu_layers
        self.n_cpu_moe = n_cpu_moe
        self.n_ctx_max = n_ctx_max
        self.n_ctx_grow = n_ctx_grow
        self.cancel_event = cancel_event
        # Mirrors the parent's already-resolved overhead (GgufBackend passes its
        # own self._VRAM_OVERHEAD_BYTES here - see gguf.py's _load_native), so
        # _check_context_fit's mid-session grow-time VRAM check reasons about the
        # SAME reserved overhead the initial load's layer-sizing used, not the
        # class-level default a config override would otherwise silently miss.
        if vram_overhead_bytes is not None:
            self._VRAM_OVERHEAD_BYTES = vram_overhead_bytes
        # The parent's already-resolved effective split ratios (auto
        # free-VRAM-proportional distribution) - forwarded verbatim to
        # LlamaCpp, never recomputed here: this process must not probe
        # (see discover.resolve_auto_split_ratios).
        self.gpu_split_ratios = gpu_split_ratios
        self._llm = None
        self._loaded = False
        self._ram_kv_hint_shown = False
        # Set by chat_stream() when the grammar-fault retry-without-grammar path
        # is taken, so the runner dispatch loop can report it in the "done"
        # envelope - the PARENT owns the persistent "stop sending grammar to
        # this model" latch (policy state that must survive across many calls
        # on one backend instance), this is just this call's outcome.
        self.grammar_unsupported_this_call = False
        self.last_finish_reason = "stop"

    @property
    def chatml_fallback_reason(self) -> Optional[str]:
        """Non-None once this model's own embedded chat template could not be used (see llama.py's _apply_model_template)."""
        return self._llm.chat_template_fallback_reason if self._llm is not None else None

    def load(self) -> dict:
        """Construct the real native model."""
        from localm.inference.backends.llamacpp._loader import load_lib
        from localm.inference.backends.llamacpp.llama import _capture_stdio
        from localm.debuglog import suppress_console_mirror

        # load_lib() (native CDLL open + ggml backend registration) can print
        # a native banner with no capture scope of its own - the only one
        # that exists opens inside LlamaCpp.__init__ below, strictly AFTER
        # load_lib() has already run. Left unredirected it lands mid-line in
        # whatever this process's inherited console is currently rendering -
        # the parent's live Rich load spinner - corrupting it. Paired with
        # suppress_console_mirror() for the same reason llama.py's own merged
        # native-call scope needs it: load_lib()'s logger.warning calls (e.g.
        # "no ggml compute backends registered") go through Python's logging
        # module, not fd 1/2, and the debug-mode console mirror is BY DESIGN
        # immune to an fd redirect (see suppress_console_mirror's docstring).
        with suppress_console_mirror(), _capture_stdio() as captured:
            try:
                load_lib()   # ensure DLLs are loaded before importing the class
            except Exception as e:
                # A genuine failure's native cause (e.g. the OS loader's own
                # dlopen error text) may be sitting in the very output just
                # suppressed - never let the capture swallow it (AGENTS.md
                # rule 5). Append it to load_lib()'s own already-actionable
                # message rather than raising a fresh exception, so this
                # stays the same type callers already handle.
                detail = captured.tail()
                if detail:
                    extra = f"\n\nCaptured native output during load:\n{detail}"
                    e.args = ((f"{e.args[0]}{extra}",) + tuple(e.args[1:])
                              if e.args else (extra,))
                raise

        from localm.inference.backends.llamacpp import LlamaCpp

        self._llm = LlamaCpp(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.effective_gpu_layers,
            n_ctx_max=self.n_ctx_max,
            n_ctx_grow=self.n_ctx_grow,
            mmproj_path=self.mmproj_path,        # C1: in-process vision via mtmd
            cancel_event=self.cancel_event,       # abort mid-load if superseded
            vram_check=self._check_context_fit,   # guard context GROWTH too
            gpu_split_ratios=self.gpu_split_ratios,
            n_cpu_moe=self.n_cpu_moe,
            mtp_enabled=self.mtp_enabled,
            verbose=False,
        )
        self._loaded = True
        return {
            "n_layers": getattr(self._llm, "n_layers", None),
            "kv_bytes_per_token": getattr(self._llm, "kv_bytes_per_token", 0),
            "supports_images": bool(self._llm.supports_images),
            "supports_mtp": bool(getattr(self._llm, "supports_mtp", False)),
            "weight_placement": getattr(self._llm, "weight_placement", []),
            "moe_skip_reason": getattr(self._llm, "moe_skip_reason", None),
        }

    def close(self) -> None:
        if self._llm is not None and hasattr(self._llm, "close"):
            try:
                self._llm.close()
            except Exception as e:
                # Surface the failed native close: leftover VRAM/context here can
                # make a later load fail mysteriously, so log a correlatable line
                # rather than swallowing it. Teardown is best-effort, so we still
                # drop the instance below instead of escalating to a hard failure.
                from localm.debuglog import logger as _dbg
                _dbg.debug("llama close() failed (%s); context may not be fully freed",
                           type(e).__name__)
        self._llm = None
        self._loaded = False

    # ------------------------------------------------------------------ #
    #  Tokenisation / grammar                                              #
    # ------------------------------------------------------------------ #

    def count_tokens(self, text: str) -> int:
        return len(self._llm.tokenize(text, add_bos=False))

    def count_messages_tokens(self, messages: List[dict]) -> int:
        """Exact token count of the structured messages formatted with the model's embedded chat template."""
        from .llama import _apply_model_template
        text_messages = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                text = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            else:
                text = content or ""
            text_messages.append({"role": m.get("role", "user"), "content": text})
        # Fallback status not propagated from here - this is a token-count-only
        # call, not a generation request; the real chat_stream() call below
        # will report the same model's fallback if a generation is ever made.
        prompt, _fallback_reason = _apply_model_template(self._llm._model_ptr, text_messages)
        bos_markers = ("<bos>", "<s>", "﻿")
        add_bos = not any(prompt.startswith(m) for m in bos_markers)
        return len(self._llm.tokenize(prompt, add_bos=add_bos))

    def check_grammar(self, grammar: str) -> None:
        """Raises InvalidGrammarError for a malformed GBNF string - see LlamaCpp.check_grammar."""
        if grammar:
            self._llm.check_grammar(grammar)

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
    ):
        """Yield text tokens one at a time."""

        def _make_kwargs(g: Optional[str]) -> dict:
            kw: dict = dict(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                grammar=g,
                grammar_lazy=grammar_lazy,
                grammar_triggers=grammar_triggers,
                stream=True,
            )
            if seed is not None:
                kw["seed"] = seed
            return kw

        def _stream(g: Optional[str]):
            for chunk in self._llm.create_chat_completion(**_make_kwargs(g)):
                choice = chunk["choices"][0]
                if choice.get("finish_reason"):
                    # "length" = the max_tokens budget ran out mid-reply
                    self.last_finish_reason = choice["finish_reason"]
                token = choice.get("delta", {}).get("content", "")
                if token:
                    yield token

        self.last_finish_reason = "stop"
        self.grammar_unsupported_this_call = False
        yielded = False
        try:
            for token in _stream(grammar):
                yielded = True
                yield token
        except OSError as e:
            # A grammar sampler fault on a build that does not implement it: the
            # model decode context is unharmed (only the sampler threw), so if
            # nothing was emitted yet, remember it and retry once WITHOUT the
            # grammar instead of failing the request.
            if grammar and not yielded:
                self.grammar_unsupported_this_call = True
                from localm.debuglog import logger as _dbg
                _dbg.warning("native grammar sampler faulted (%s); "
                             "degrading to unconstrained generation", e)
                self.last_finish_reason = "stop"
                yield from _stream(None)
                return
            # Any other native fault (access violation etc.): do not soften it,
            # let this process die - see the docstring above.
            raise
