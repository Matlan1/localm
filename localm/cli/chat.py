# SPDX-License-Identifier: AGPL-3.0-or-later
import json
import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.panel import Panel

from ..config import load_config
from ..model_manager import get_model_info
from ..console import show_url
from ._core import console, err_console, main, _complete_model_name


def _attach_fallback_note(no_server: bool, attach_error: Optional[BaseException],
                          autostart_attempted: bool = False) -> Optional[str]:
    """The note to print when `localm run` is about to load the model in THIS
    process instead of attaching to a background server, so the fallback is never
    silent. None when the user opted out with --no-server (stay quiet).

    When an auto-start WAS launched but did not come up in time,
    ``autostart_attempted`` is True so the note acknowledges that timeout instead
    of telling the user "no server is serving this directory; start one", which
    would contradict the `Starting one in the background...` line they just
    saw."""
    if no_server:
        return None
    if attach_error is not None:
        return (f"Could not attach to a localm server ({attach_error}); loading "
                "the model in this process.")
    if autostart_attempted:
        return ("The background server did not come up in time (~20s; a large "
                "model can take longer to load); loading the model in this "
                "process instead. Try again shortly, or start one yourself with "
                "`localm serve`.")
    return ("No localm server is serving this directory; loading the model in "
            "this process. Start one with `localm serve` so clients share a "
            "single load.")


class _TurnRouter:
    """Picks the engine that answers each turn of ``localm run MODEL``.

    MODEL is the loaded model. A turn that needs something it does not provide
    (reading an image, a longer conversation than it was trained for) is
    answered by an installed model that has it, unless *pinned*.

    Attached to a server (*build* is None) the server routes the request
    itself; this only works out the context window to ask for, so the
    conversation is not compacted when a roomier model can hold it. Loaded in
    this process, it swaps engines: the current one is unloaded before another
    is loaded."""

    def __init__(self, engine, model_name: str, *, pinned: bool, build=None,
                 known: Optional[dict] = None):
        self.primary = engine
        self._known = dict(known or {})
        self.model_name = model_name
        self.pinned = pinned
        self._build = build
        self._engines = {model_name: engine}
        self.current = model_name
        self.min_context: Optional[int] = None
        self.last_decision = None
        self.failed_to_load: tuple = ()

    @property
    def in_process(self) -> bool:
        return self._build is not None

    @property
    def engine(self):
        """The engine of ``current``: the model this router last selected,
        unloaded when that selection's load failed."""
        return self._engines[self.current]

    def plan(self, messages: list):
        """The routing decision for a turn with *messages*, without acting on
        it. Sets ``min_context`` to the window to ask the server for when the
        decision routes for context, else None."""
        from localm.config import load_registry
        from localm.inference import capability_routing as cr
        from localm.inference.compact import COMPACT_RATIO
        from localm.model_manager import capabilities as caps
        reg = load_registry()
        trained = caps.model_context_length(self.model_name, reg=reg)
        comp = cr.compaction_context_need(messages, trained, COMPACT_RATIO)
        needs = cr.request_needs(messages, min_context=comp)
        known = dict(self._known)
        cur = self._engines.get(self.model_name)
        if (cur is not None and getattr(cur, "loaded", False)
                and getattr(cur, "supports_images", False) is True):
            known[caps.VISION] = True
        decision = cr.plan_route(self.model_name, needs, pinned=self.pinned,
                                 resident=[self.current], reg=reg,
                                 current_known=known)
        self.last_decision = decision
        self.min_context = (comp if decision.routed
                            and caps.CONTEXT_LENGTH in decision.gaps else None)
        return decision

    def engine_for(self, messages: list):
        """The engine to answer a turn with *messages*: MODEL's, or in this
        process the routed model's, loaded in its place. A routed model that
        fails to load is skipped for the next capable one, then MODEL. Each
        failure is printed; ``failed_to_load`` names the routed models MODEL
        answers in place of, else is empty. Raises what MODEL's own load raised
        when that fails too."""
        decision = self.plan(messages)
        self.failed_to_load = ()
        if not self.in_process:
            return self.primary
        names = list(decision.candidates or (decision.resolved,)) if decision.routed else []
        failed = []
        for name in names:
            try:
                return self._use(name)
            except Exception as e:
                from rich.markup import escape
                console.print(f"[yellow]Could not load {escape(name)}: "
                              f"{escape(str(e))}[/yellow]")
                failed.append(name)
        eng = self._use(self.model_name)
        self.failed_to_load = tuple(failed)
        return eng

    def _use(self, name: str):
        if name == self.current:
            eng = self._engines[name]
            if not getattr(eng, "loaded", True):
                eng.load()
            return eng
        prev = self._engines[self.current]
        if getattr(prev, "loaded", True):
            prev.unload()
        eng = self._engines.get(name) or self._build(name)
        self._engines[name] = eng
        self.current = name
        eng.load()
        return eng

    def answered_by(self, engine) -> str:
        """The model that answered the last turn on *engine*."""
        if self.in_process:
            return self.current
        return getattr(engine, "answered_model", None) or self.model_name

    def note(self, engine) -> Optional[str]:
        """One line naming the model that answered and why, when it was not
        MODEL or MODEL answered in place of routed models that failed to load;
        else None."""
        answered = self.answered_by(engine)
        if not answered:
            return None
        if answered == self.model_name:
            if self.in_process and self.failed_to_load:
                return (f"answered by {answered}: could not load "
                        f"{', '.join(self.failed_to_load)}")
            return None
        gaps = sorted((self.last_decision.gaps if self.last_decision else {}) or {})
        why = {"vision": "reading images", "tool_use": "structured tool calls",
               "reasoning": "reasoning",
               "context_length": "a longer conversation"}
        needs = ", ".join(why.get(g, g) for g in gaps) or "this request"
        return (f"answered by {answered}: {self.model_name} lacks {needs} "
                f"(--pin-model always answers with {self.model_name})")

    def close(self) -> None:
        """Unload every engine this router loaded besides the one the session
        opened with, which its own ``with`` block unloads."""
        for name, eng in self._engines.items():
            if eng is not self.primary and getattr(eng, "loaded", False):
                try:
                    eng.unload()
                except Exception as e:
                    from localm.debuglog import logger as _dbg
                    _dbg.warning("unloading %s after the session failed: %s", name, e)


def _build_cli_engine(name: str, *, n_ctx=None, n_gpu_layers=None, device=None,
                      mmproj=None):
    """An unloaded in-process Engine for the registered model *name*, with its
    recorded or sibling projector unless *mmproj* is given. Registry names
    only: raises ValueError when *name* is not registered or its file is
    gone."""
    from ..inference.engine import Engine
    from ..model_manager import get_model_info, get_model_mmproj
    info = get_model_info(name)
    if info is None:
        raise ValueError(f"Model not found: {name}")
    return Engine(
        str(info[0]),
        mmproj_path=mmproj or get_model_mmproj(name),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        device=device,
        display_name=name,
    )


def _maybe_persist_cli_mmproj(model: str, mmproj: Optional[str],
                              is_registered: bool, engine) -> None:
    """Record an explicit --mmproj onto the registry entry once the backend has
    CONFIRMED supports_images for this load, so a later `localm run model` -
    including the one vision_input_guidance itself suggests - keeps seeing it
    instead of losing it the moment the flag is left off.

    Gated on is_registered, since a bare direct-path run has no entry to write
    to, and on the CONFIRMED load rather than the flag merely being present, so a
    projector that failed to load can never be recorded as working."""
    if not (mmproj and is_registered):
        return
    backend = getattr(engine, "_backend", None)
    if not getattr(backend, "supports_images", False):
        return
    from rich.markup import escape

    from ..model_manager import persist_cli_mmproj
    note = persist_cli_mmproj(model, mmproj)
    if note:
        console.print(f"[dim]{escape(note)}[/dim]")




# ------------------------------------------------------------------ #
#  run                                                                 #
# ------------------------------------------------------------------ #

@main.command()
@click.argument("model", shell_complete=_complete_model_name)
@click.option("-p", "--prompt",       default=None,  help="Single prompt (non-interactive).")
@click.option("-s", "--system",       default=None,  help="System prompt.")
@click.option("-m", "--max-tokens",   default=None,  type=int,   help="Max tokens to generate.")
@click.option("-t", "--temperature",  default=None,  type=float, help="Sampling temperature.")
@click.option("-c", "--ctx",          default=None,  type=int,   help="Context window (GGUF only).")
@click.option("-g", "--gpu-layers",   default=None,  type=click.IntRange(0, 1000),   help="GPU layers (GGUF only, 99=all).")
@click.option("--mmproj",             default=None,
              help="Path to a multimodal projector (mmproj) GGUF, enabling vision "
                   "on a vision-capable GGUF model. Pair with --image to attach "
                   "pictures.")
@click.option("--device",             default=None,  help="HF device override (cuda / cpu).")
@click.option("--image", "images",   multiple=True, type=click.Path(exists=True),
              help="Local image file to include (repeat for multiple). Use with -p.")
@click.option("--debug", is_flag=True,
              help="Write a debug log (<data dir>/logs/), capture native llama.cpp "
                   "stderr, and record raw model output (with markers) in the log.")
@click.option("--mode", default=None,
              type=click.Choice(["privacy", "log", "full"], case_sensitive=False),
              help="Session persistence [default: config 'chat_mode'/'mode', else privacy]. "
                   "privacy = nothing saved automatically; "
                   "log = JSONL audit trail to <data dir>/sessions/; "
                   "full = log + markdown transcript.")
@click.option("--no-server", is_flag=True,
              help="Load the model in THIS process instead of attaching to a localm "
                   "server already serving this directory (default: attach, like the "
                   "GUI, so you do not load a second copy).")
@click.option("--pin-model", "pin_model", is_flag=True,
              help="Always answer with MODEL. Without it, a message MODEL cannot "
                   "handle (an attached image it cannot read, a longer "
                   "conversation than it was trained for) is answered by an "
                   "installed model that can, and says so.")
def run(model, prompt, system, max_tokens, temperature, ctx, gpu_layers,
        mmproj, device, images, debug, mode, no_server, pin_model):
    """Run a model - interactive chat or single prompt.

    \b
    MODEL can be a registered name OR a direct path:
      localm run gemma4-12b
      localm run D:\\models\\llama3.gguf
      localm run D:\\hf-models\\gemma-3-4b-it

    \b
    Image input (multimodal models):
      localm run gemma4-12b -p "What is in this photo?" --image photo.jpg
      localm run gemma4-12b -p "Compare" --image a.png --image b.png

    \b
    In interactive mode, attach images with the /image command:
      /image D:\\photos\\cat.jpg

    \b
    Pipe a prompt from stdin:
      echo "Explain RDNA2" | localm run qwen2.5-7b
    """
    from rich.markup import escape

    if debug:
        from ..debuglog import enable_debug
        console.print(f"[yellow]debug log:[/yellow] {escape(str(enable_debug()))}")

    from ..audit import MODE_ENV_VAR, SessionMode, effective_mode
    if mode:
        os.environ[MODE_ENV_VAR] = mode.lower()
    session_mode = effective_mode("chat")
    if session_mode == SessionMode.PRIVACY:
        from ..readline_privacy import suppress_readline_history
        suppress_readline_history()
        if debug:
            console.print(
                "[yellow]⚠  privacy mode + --debug:[/yellow] the debug log "
                "still records operational lines (requests, timings, errors) "
                "- never raw model output or chat content, even with this "
                "flag on. Delete it after analysis if the operational detail "
                "matters to you.")
    else:
        console.print(f"[dim]session mode: {session_mode.value} "
                      f"(audit trail in <data dir>/sessions/)[/dim]")

    # Thin client: ATTACH to the localm server already serving this directory
    # (route chat through its /v1 over HTTP) instead of loading a SECOND in-process
    # copy of the model, mirroring `localm gui`. Default is to attach when a verified
    # server exists; --no-server forces an in-process load.
    engine = None
    is_registered = False
    attach_error: Optional[BaseException] = None
    if not no_server:
        from .. import instances
        from ..config import home_dir
        try:
            target = instances.attach_target(
                home_dir(), instances.resolve_root_dir())
        except Exception as e:
            # Log the failure at debug level and remember it, so the in-process
            # fallback below can say WHY it did not attach.
            from localm.debuglog import logger as _dbg
            _dbg.exception("attach_target failed; falling back to in-process load")
            attach_error = e
            target = None
        if target:
            from ..auth import resolve_bearer_token
            from ..inference.http_engine import HttpEngine, remote_model_status
            # The discovered instance's own per-instance token is only
            # meaningful in OPEN mode (it satisfies the origin guard, not
            # _enforce_request's key check). Once an owner key is configured,
            # _principal_from_token has no notion of instance tokens at all and
            # 401s every request that presents one - the CLI and the server
            # share the same data dir, so the owner key on disk is exactly the
            # credential this process is entitled to use.
            attach_token = resolve_bearer_token(target.get("token"))
            state, active = remote_model_status(
                target["base_url"], attach_token)
            # Never let `localm run X` answer with a DIFFERENT model. When the
            # running server serves a KNOWN model that is not the one the user
            # asked for, attaching would stream a reply generated by that other
            # model - indistinguishable from X answering. Refuse instead of
            # silently overriding the user's explicit choice; --no-server is the
            # way out.
            from ..model_manager import names_same_model
            if active and not names_same_model(active, model):
                console.print(
                    f"[red]The localm server here serves "
                    f"[bold]{escape(active)}[/bold], not "
                    f"the requested [bold]{escape(model)}[/bold].[/red]\n"
                    f"[dim]Attaching would answer with {escape(active)}, so localm "
                    f"will not do that silently. To run {escape(model)}, re-run "
                    f"with [bold]--no-server"
                    f"[/bold] (loads it in this process).[/dim]")
                sys.exit(1)
            if state == "empty":
                # The server is up but has no model loaded, so it cannot serve the
                # requested model either. Refuse cleanly rather than attach and fail
                # mid-stream; --no-server loads the model in this process.
                console.print(
                    f"[red]A localm server is running for this directory but has no "
                    f"model loaded, so it cannot serve [bold]{escape(model)}[/bold]."
                    f"[/red]\n"
                    f"[dim]Load a model on its Models page/API, or re-run with "
                    f"[bold]--no-server[/bold] to run {escape(model)} in this "
                    f"process.[/dim]")
                sys.exit(1)
            # state == "unknown": the server answered our attach but we could not
            # read /v1/models (it needs the models scope, so a chat-scoped attach
            # token gets a 403). Do NOT claim it has no model - it very likely does;
            # attach quietly and let the reply come from whatever it serves.
            engine = HttpEngine(
                target["base_url"], token=attach_token,
                model=active or model, display_name=active or model,
                pin_model=pin_model)
            # show_url(): target["base_url"] can carry a bracketed IPv6 host
            # (RFC 3986), which show_url()'s own docstring documents.
            console.print(
                f"[dim]connected to the localm server at "
                f"{show_url(target['base_url'])} (no second model load)[/dim]")

    autostart_attempted = False
    if engine is None and not no_server:
        autostart_attempted = True
        console.print("[dim]No server running. Starting one in the background...[/dim]")
        import subprocess
        import time
        
        cmd = [sys.executable, "-m", "localm", "gui", "--no-browser", "--api-mode"]
        # Use no_model if no model was provided, else pass it
        if not model:
            cmd.append("--no-model")
        else:
            cmd.append(model)
        
        if ctx is not None: cmd.extend(["-c", str(ctx)])
        if gpu_layers is not None: cmd.extend(["-g", str(gpu_layers)])
        if mmproj: cmd.extend(["--mmproj", mmproj])
        if device: cmd.extend(["--device", device])
        
        kwargs = {}
        env = os.environ.copy()
        env["LOCALM_OWN_CONSOLE"] = "1"
        kwargs["env"] = env
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
        else:
            kwargs["start_new_session"] = True
            
        try:
            subprocess.Popen(cmd, **kwargs)
            # Poll for the server to come up
            for _ in range(40):
                time.sleep(0.5)
                target = instances.attach_target(home_dir(), instances.resolve_root_dir())
                if target:
                    from ..auth import resolve_bearer_token
                    from ..inference.http_engine import HttpEngine
                    engine = HttpEngine(
                        target["base_url"],
                        token=resolve_bearer_token(target.get("token")),
                        model=model, display_name=model, pin_model=pin_model)
                    console.print(f"[dim]connected to newly started server at "
                                  f"{show_url(target['base_url'])}[/dim]")
                    break
        except Exception as e:
            attach_error = e

    if engine is None:
        _note = _attach_fallback_note(no_server, attach_error, autostart_attempted)
        if _note:
            console.print(f"[dim]{escape(_note)}[/dim]")
        # allow_direct_path: `localm run /full/path` is a documented feature (the
        # help text right below advertises it), and *model* here is typed by the
        # operator on their own command line, not received over the wire.
        info = get_model_info(model, allow_direct_path=True)
        if info is None:
            console.print(f"[red]Model not found:[/red] {escape(model)}")
            console.print("  [dim]localm list[/dim]              - downloaded models")
            console.print("  [dim]localm models[/dim]            - GGUF shortcuts")
            console.print("  [dim]localm pull owner/repo[/dim]   - download HF model")
            console.print("  [dim]localm pull name[/dim]         - download GGUF shortcut")
            console.print("  [dim]localm add <path>[/dim]        - register local file/dir")
            console.print("  [dim]localm run /full/path[/dim]    - use path directly")
            sys.exit(1)

        model_path, _display_hint = info

        from ..inference.engine import Engine
        from ..model_manager import get_model_mmproj
        from ..model_manager import load_registry as _reg

        # Priority: registered alias > Ollama manifest hint > engine auto-derive
        is_registered = model in _reg()
        if is_registered:
            display_name = model
        else:
            display_name = _display_hint  # None or Ollama suggested name

        # An explicit --mmproj always wins; otherwise fall back to the model's
        # own recorded/sibling projector, or a pulled vision GGUF run straight
        # from the CLI (no --mmproj flag given) silently loses image support.
        # allow_direct_path=True matches the get_model_info call above: *model*
        # is operator-typed on this command line.
        mmproj_path = mmproj or get_model_mmproj(model, allow_direct_path=True)

        engine = Engine(
            str(model_path),
            mmproj_path=mmproj_path,
            n_ctx=ctx,
            n_gpu_layers=gpu_layers,
            device=device,
            display_name=display_name,
        )
        router = _TurnRouter(
            engine, model, pinned=pin_model,
            build=lambda name: _build_cli_engine(
                name, n_gpu_layers=gpu_layers, device=device),
            known={"vision": True} if mmproj_path else None)
    else:
        router = _TurnRouter(engine, getattr(engine, "_model", None) or model,
                             pinned=pin_model)

    cfg = load_config()
    # settings_schema calls chat_system_prompt 'the system prompt every new chat
    # starts with', with a chat's own System field overriding it. --system is that
    # override here, so a run without one inherits the setting. Without this the
    # setting reached the web GUI only and did nothing on the command line.
    if system is None:
        system = (cfg.get("chat_system_prompt") or "").strip() or None
    gen_opts = {
        "max_tokens":     max_tokens  if max_tokens  is not None else cfg["max_tokens"],
        "temperature":    temperature if temperature is not None else cfg["temperature"],
        "top_p":          cfg["top_p"],
        "top_k":          cfg["top_k"],
        "repeat_penalty": cfg["repeat_penalty"],
    }

    # Accept piped stdin as the prompt
    if not sys.stdin.isatty() and prompt is None:
        prompt = sys.stdin.read().strip()

    from ..audit import make_audit_log, make_transcript
    audit = make_audit_log(session_mode, label="chat")
    transcript = make_transcript(session_mode, label="chat")

    if prompt is not None and router.in_process:
        # A single prompt is known before anything loads: when it routes, the
        # routed model is the only one loaded.
        _first = []
        if system:
            _first.append({"role": "system", "content": system})
        _first.append(_build_user_message(prompt, list(images)))
        if router.plan(_first).routed:
            engine = router.engine_for(_first)
            if router.current != model:
                is_registered = False
            _routed_note = router.note(engine)
        else:
            _routed_note = None
    else:
        _routed_note = None

    try:
        with engine:
            _maybe_persist_cli_mmproj(model, mmproj, is_registered, engine)
            if prompt is not None:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append(_build_user_message(prompt, list(images)))
                audit.user(prompt)
                stream_opts = dict(gen_opts)
                if not router.in_process:
                    router.plan(messages)
                    if router.min_context:
                        stream_opts["min_context"] = router.min_context
                response = _stream_once(engine, messages, **stream_opts)
                note = _routed_note or router.note(engine)
                if note and response:
                    console.print(f"[dim]({escape(note)})[/dim]")
                # The JSONL audit log is an INTERNAL consumer (see
                # textnorm.strip_think's docstring), so it gets the visible
                # answer only, not the raw <think> scratchpad - matching what
                # transcript.exchange does via its own split_think() call.
                from ..textnorm import strip_think
                audit.llm(strip_think(response))
                if transcript:
                    transcript.exchange(prompt, response)
            else:
                _interactive(engine, system, gen_opts,
                             audit=audit, transcript=transcript, router=router)
    finally:
        router.close()
        audit.close()




def _withdraw(messages: list, msg: dict) -> None:
    """Remove the unanswered turn *msg* from the end of *messages* and say so."""
    if messages and messages[-1] is msg:
        messages.pop()
        console.print("[dim](that message was withdrawn from the conversation; "
                      "you can keep chatting)[/dim]")


def _build_user_message(text: str, image_paths: list) -> dict:
    """Build a user message dict, embedding local images as base64 data-URIs."""
    if not image_paths:
        return {"role": "user", "content": text}

    parts: list = []
    for path in image_paths:
        parts.append({
            "type": "image_url",
            "image_url": {"url": _file_to_data_uri(path)},
        })
    if text:
        parts.append({"type": "text", "text": text})
    return {"role": "user", "content": parts}




def _file_to_data_uri(path: str) -> str:
    """Read a local image file and return a base64 data-URI."""
    import base64
    import mimetypes
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"




class _ThinkPrinter:
    """Stream a reply to stdout, dimming the model's ``<think>`` reasoning so it
    reads as an aside rather than raw ``<think>`` tags inline with the answer.
    The full raw text (tags included) is still returned by the caller for the
    audit/transcript, which separate it themselves.

    Streaming here is APPEND-ONLY plain ``print`` plus SGR styling (dim/colour):
    no alternate screen, no cursor repositioning, no spinner/Live region and no
    ``\\r`` redraw. The terminal emulator therefore owns scrolling, so a user who
    scrolls up to re-read mid-stream keeps their position while output continues
    to append below."""

    def __init__(self) -> None:
        import sys as _sys
        from localm.textnorm import ThinkSplitter
        tty = _sys.stdout.isatty()
        self._dim, self._reset = ("\033[2m", "\033[22m") if tty else ("", "")
        self._think = ThinkSplitter()

    def _emit(self, content: str, reasoning: str) -> None:
        if reasoning:
            print(f"{self._dim}{reasoning}{self._reset}", end="", flush=True)
        if content:
            print(content, end="", flush=True)

    def feed(self, token: str) -> None:
        self._emit(*self._think.feed(token))

    def flush(self) -> None:
        self._emit(*self._think.flush())




# A floor on plausible per-token decode time (mirrors http_server.py's
# _MIN_SEC_PER_TOKEN - kept local rather than imported, so this module does not
# pull in the HTTP server's FastAPI/uvicorn import surface for a single
# constant). first_at is a SINGLE sample: if the GPU scheduler delays the first
# token under contention and then delivers the rest in an uncontended burst, the
# decode window shrinks toward zero and the rate runs to tens of thousands of
# tok/s even though every timestamp is real. 1ms/token (a 1000 tok/s ceiling) is
# generous for single-stream autoregressive decode, which is memory-bandwidth-
# bound at a full weight read per token. Below it, the rate is omitted rather
# than printed.
_MIN_SEC_PER_TOKEN = 0.001


def _perf_line(n_tokens: int, t0: float, first_at: Optional[float],
               end: float) -> Optional[str]:
    """One-line perf readout for the REPL, or None when there is nothing worth
    showing. tok/s is computed over the DECODE window only (first token -> end);
    the model-load + prompt-prefill time (the wait before the first token) is shown
    separately as `load` rather than folded into the rate, which would make the
    first call after a load read around 100x too slow.

    Matches the server's _tokens_per_sec and the `localm bench` convention: tok/s
    measures pure generation after the first token. tok/s is omitted for a single
    token (no decode interval to time) or an implausibly short decode window (see
    _MIN_SEC_PER_TOKEN)."""
    total = end - t0
    if first_at is None or total <= 0.5:
        return None
    load = first_at - t0          # model load + prompt prefill (TTFT)
    gen = end - first_at          # decode window
    plausible = n_tokens >= 2 and gen >= n_tokens * _MIN_SEC_PER_TOKEN
    rate = f"{n_tokens / gen:.1f} tok/s  " if plausible else ""
    # Show the load/gen split only when the load is a meaningful slice (a cold
    # start); a warm call's sub-100ms prefill keeps the single-time form.
    if load >= 0.1:
        return f"{n_tokens} tokens  {rate}(load {load:.1f}s, gen {gen:.1f}s)"
    return f"{n_tokens} tokens  {rate}({gen:.1f}s)"


def _stream_once(engine, messages: list, **kwargs) -> str:
    """Stream response to stdout, print tok/s on completion, and return the full text."""
    import time as _time
    from rich.markup import escape

    from localm.inference.backends.base import (
        ImageDecodeUnavailable,
        UnsupportedInputError,
        VISION_CPU_FALLBACK_STATUS,
    )
    parts: list[str] = []
    printer = _ThinkPrinter()
    t0 = _time.monotonic()
    first_at: Optional[float] = None
    stream_kwargs = dict(kwargs)
    if "on_status" not in stream_kwargs:
        def _cli_status(s: str) -> None:
            if s == VISION_CPU_FALLBACK_STATUS:
                err_console.print(f"\n[yellow]{escape(s)}[/yellow]")
        stream_kwargs["on_status"] = _cli_status
    try:
        for token in engine.chat_stream(messages, **stream_kwargs):
            if first_at is None:
                first_at = _time.monotonic()
            parts.append(token)
            printer.feed(token)
        printer.flush()
    except ImageDecodeUnavailable as e:
        # BEFORE the UnsupportedInputError arm below, which DISCARDS the message
        # and prints vision-capability guidance in its place. On this path the
        # model is vision-capable and the picture is fine, the environment simply
        # has no image decoder, so "pick or download a vision model" would send
        # the user after a problem they do not have. This arm keeps the real
        # message, which names the missing library and the fix.
        console.print(f"\n[red]{escape(str(e))}[/red]")
        return ""
    except UnsupportedInputError:
        # Capability-aware guidance instead of a flat "can't do that": name a
        # vision model this install has, or how to get one.
        from localm.model_manager import vision_input_guidance
        backend = getattr(engine, "_backend", None)
        mmproj_failed = bool(getattr(backend, "mmproj_path", None))
        active_model_path = getattr(backend, "model_path", None)
        guidance = vision_input_guidance(
            mmproj_failed=mmproj_failed, active_model_path=active_model_path)
        console.print(f"\n[yellow]{escape(guidance)}[/yellow]")
        return ""
    except RuntimeError as e:
        # An attached server returned an error (no model loaded, unreachable, ...).
        # Surface it cleanly instead of a traceback (the interactive path already
        # catches Exception; this is the single-prompt path).
        console.print(f"\n[red]{escape(str(e))}[/red]")
        sys.exit(1)
    end = _time.monotonic()
    print()
    full = "".join(parts)
    if full:
        line = _perf_line(engine.count_tokens(full), t0, first_at, end)
        if line:
            console.print(f"[dim]{line}[/dim]")
    return full




def _interactive(engine, system_prompt: Optional[str], gen_opts: dict,
                 audit=None, transcript=None, router=None) -> None:  # noqa: C901
    from rich.markup import escape
    from localm.inference.backends.base import (ImageDecodeUnavailable,
                                                UnsupportedInputError)
    if router is None:
        router = _TurnRouter(engine, engine.display_name, pinned=True)

    console.print(Panel(
        f"[bold cyan]localm[/bold cyan] - {escape(engine.display_name)}\n"
        "[dim]Ctrl+C or [bold]/exit[/bold] to quit  ·  "
        "[bold]/clear[/bold] history  ·  [bold]/image <path>[/bold] attach image  ·  "
        "[bold]/help[/bold][/dim]",
        border_style="dim cyan",
        padding=(0, 1),
    ))

    messages: list = []
    pending_images: list = []   # image paths queued for the next user message

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        console.print(f"[dim]system: {escape(system_prompt)}[/dim]\n")

    while True:
        img_hint = f" [dim][{len(pending_images)} image(s) queued][/dim]" if pending_images else ""
        try:
            user_input = console.input(f"\n[bold green]You[/bold green]{img_hint}: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not user_input:
            continue

        # Catch bare-word exits typed without the leading slash
        if user_input.lower() in ("exit", "quit", "q", "bye"):
            console.print("[dim]Bye.[/dim]")
            break

        if user_input.startswith("/"):
            stop = _handle_command(user_input, messages, gen_opts, pending_images,
                                   engine=engine)
            if stop:
                break
            continue

        msg = _build_user_message(user_input, pending_images)
        pending_images.clear()
        messages.append(msg)
        if audit:
            audit.user(user_input)

        # The engine that answers this turn: MODEL's, or one that has what the
        # turn needs (see _TurnRouter). After a failed load it is the router's
        # current engine.
        try:
            engine = router.engine_for(messages)
        except Exception as e:
            engine = router.engine
            console.print(f"\n[red]Could not load {escape(router.model_name)}: "
                          f"{escape(str(e))}[/red]")
            if messages and messages[-1] is msg:
                messages.pop()
            continue

        # Seamless compaction: summarise older turns before the history
        # collides with the context ceiling. Never fails - falls back to a
        # visible hard trim when summarisation is unavailable.
        from ..inference.compact import maybe_compact
        # Budget against the LOADED model's RESOLVED ceiling (VRAM-derived under
        # ctx_auto), not the static config n_ctx_max, which both over-compacts a
        # small-window model and under-protects a large one. Fall back to the
        # config only when the engine cannot report a capacity (not loaded).
        limit = engine.context_capacity() or load_config().get("n_ctx_max", 16384) or 0
        if router.min_context and not router.in_process:
            # Attached, a turn routed for context is sent whole, with
            # min_context naming the window it needs.
            compacted_msgs, did_compact = messages, False
        else:
            compacted_msgs, did_compact = maybe_compact(
                messages,
                limit_tokens=limit,
                count_tokens=engine.count_tokens,
                generate=lambda m, max_tok: "".join(
                    engine.chat_stream(m, max_tokens=max_tok, temperature=0.3)),
            )
        if did_compact:
            messages[:] = compacted_msgs
            console.print("[dim](older conversation summarised to free context)[/dim]")

        console.print("\n[bold blue]Assistant[/bold blue]: ", end="")

        parts: list[str] = []
        printer = _ThinkPrinter()
        import time as _time
        t0 = _time.monotonic()
        first_at: Optional[float] = None
        interactive_opts = dict(gen_opts)
        if router.min_context and not router.in_process:
            interactive_opts["min_context"] = router.min_context
        if "on_status" not in interactive_opts:
            from localm.inference.backends.base import VISION_CPU_FALLBACK_STATUS

            def _cli_interactive_status(s: str) -> None:
                if s == VISION_CPU_FALLBACK_STATUS:
                    err_console.print(f"\n[yellow]{escape(s)}[/yellow]")
            interactive_opts["on_status"] = _cli_interactive_status
        try:
            for token in engine.chat_stream(messages, **interactive_opts):
                if first_at is None:
                    first_at = _time.monotonic()
                parts.append(token)
                printer.feed(token)
            printer.flush()
        except KeyboardInterrupt:
            printer.flush()
            console.print("\n[dim](interrupted)[/dim]")
        except ImageDecodeUnavailable as e:
            # The model can read images but this environment cannot decode
            # them. The message is withdrawn from the conversation.
            console.print(f"\n[red]{escape(str(e))}[/red]")
            _withdraw(messages, msg)
            continue
        except UnsupportedInputError:
            # The answering model cannot read the attached image. The message is
            # withdrawn from the conversation.
            from localm.model_manager import vision_input_guidance
            backend = getattr(engine, "_backend", None)
            console.print("\n[yellow]" + escape(vision_input_guidance(
                mmproj_failed=bool(getattr(backend, "mmproj_path", None)),
                active_model_path=getattr(backend, "model_path", None)))
                + "[/yellow]")
            _withdraw(messages, msg)
            continue
        except Exception as e:
            console.print(f"\n[red]Inference error: {escape(str(e))}[/red]")
            # The failed message is withdrawn so the next one is not sent after
            # an unanswered turn.
            if messages and messages[-1] is msg:
                messages.pop()
            continue

        response = "".join(parts) or "(interrupted)"
        end = _time.monotonic()
        print()
        if parts:
            line = _perf_line(engine.count_tokens(response), t0, first_at, end)
            if line:
                console.print(f"[dim]{line}[/dim]")
            note = router.note(engine)
            if note:
                console.print(f"[dim]({escape(note)})[/dim]")
        if response:
            # Resend and log only the visible answer, never the raw <think>
            # scratchpad (textnorm.strip_think's docstring: "the one helper every
            # INTERNAL consumer of model output must run before storing").
            # transcript.exchange is exempt - it splits `response` itself and
            # keeps the reasoning in a collapsed block.
            from ..textnorm import strip_think
            visible = strip_think(response)
            messages.append({"role": "assistant", "content": visible})
            if audit:
                audit.llm(visible)
            if transcript:
                transcript.exchange(user_input, response)




# The REPL media-generation commands (/generate-image, /generate-music,
# /generate-video) all share one shape: ensure ComfyUI is reachable (auto-
# launching it from the configured comfy_launch_cmd/comfy_workdir if needed),
# unload the chat model to free VRAM, generate one file into <home>/<subdir>,
# then free ComfyUI's VRAM. Only the generator, output subdir/extension and the
# argument label differ, so the three paths are table-driven to stay identical.
# Each helper imports its generator from the `.comfy` submodule at call time so a
# test that patches e.g. localm.image_gen.comfy.generate_image is honoured.
def _media_generate_image():
    from ..image_gen.comfy import generate_image
    return generate_image


def _media_generate_music():
    from ..music_gen.comfy import generate_music
    return generate_music


def _media_generate_video():
    from ..video_gen.comfy import generate_video
    return generate_video


# The gallery directory names come from localm.media.paths, which is also what
# the image/music/video plugins and the input_image confinement policy use. One
# list, not a duplicated set of literals here: it decides which directories an
# img2img input may be read from.
from ..media import paths as _media_paths  # noqa: E402


_MEDIA_REPL = {
    "generate-image": {
        "subdir": _media_paths.IMAGE_DIR_NAME, "ext": ".png", "arg": "prompt",
        "get_generate": _media_generate_image, "plugin": "image",
    },
    "generate-music": {
        "subdir": _media_paths.MUSIC_DIR_NAME, "ext": ".flac", "arg": "tags",
        "get_generate": _media_generate_music, "plugin": "music",
    },
    "generate-video": {
        "subdir": _media_paths.VIDEO_DIR_NAME, "ext": ".mp4", "arg": "prompt",
        "get_generate": _media_generate_video, "plugin": "video",
    },
}


def _cmd_generate_media(label: str, arg: str, engine, console, home_dir) -> None:
    """REPL /generate-image|/generate-music|/generate-video: generate one media
    file via the configured ComfyUI backend, unloading the chat model first to
    free VRAM. console + home_dir are passed in so a caller that resolved the
    (monkeypatchable) localm.cli.console / localm.cli.HOME_DIR is honoured here."""
    from rich.markup import escape

    spec = _MEDIA_REPL[label]
    if engine is None:
        console.print(f"[dim]/{label} not available in this mode[/dim]")
        return
    if not arg:
        console.print(f"[dim]Usage: /{label} <{spec['arg']}>[/dim]")
        return
    from ..image_gen.comfy import ensure_comfy, free_comfy_vram
    from .media import _plugin_api_url
    api = _plugin_api_url(spec["plugin"])
    # Auto-launch ComfyUI from the configured comfy_launch_cmd/comfy_workdir.
    # Only unload the chat model once ComfyUI is actually available. escape(t):
    # progress text embeds the user's own comfy_launch_cmd/comfy_workdir config
    # values verbatim.
    ok, msg = ensure_comfy(
        api, on_progress=lambda t: console.print(f"[dim]{escape(t)}[/dim]"))
    if not ok:
        console.print(f"[yellow]{escape(msg)}[/yellow]")
        return
    import time as _t
    out_dir = home_dir / spec["subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{_t.strftime('%Y%m%d_%H%M%S')}_cli{spec['ext']}"
    console.print("[dim]Freeing VRAM (chat model unloads, "
                  "reloads on your next message)...[/dim]")
    engine.unload()
    from ..audit import SessionMode, effective_mode
    generate = spec["get_generate"]()
    is_privacy = effective_mode("chat") == SessionMode.PRIVACY
    ok, message = generate(
        arg, out,
        api_url=api,
        write_sidecar=not is_privacy,
        delete_outputs=is_privacy,
    )
    # escape(): message is a bare string passed straight to console.print - Rich
    # parses "[...]" in ANY printed string, not just inside literal markup tags -
    # and it embeds the generated output path plus, on failure, arbitrary
    # exception/config text (see generate_image's own return sites).
    console.print(escape(message))
    if ok:
        free_comfy_vram(api)


def _cmd_generate_image(cmd: str, arg: str, engine, console, home_dir) -> None:
    """REPL /generate-image (and the /imagine alias): generate one image via the
    configured ComfyUI backend. Thin wrapper over the shared media path that
    keeps the /imagine rename notice."""
    if cmd == "imagine":
        console.print("[dim]/imagine was renamed to /generate-image[/dim]")
    _cmd_generate_media("generate-image", arg, engine, console, home_dir)


def _cmd_save(arg: str, messages: list, console) -> None:
    """REPL /save: write the conversation to JSON, confined to the cwd."""
    from rich.markup import escape

    target = arg or "chat.json"
    # Confine writes to the current working directory: a path like "../x.json" or
    # an absolute path elsewhere must not let the REPL write outside cwd.
    cwd = Path.cwd().resolve()
    try:
        resolved = (cwd / target).resolve()
    except (OSError, ValueError) as e:
        console.print(f"[red]Invalid save path: {escape(str(e))}[/red]", soft_wrap=True)
        return
    if resolved == cwd or cwd not in resolved.parents:
        console.print(
            f"[red]Refusing to save outside the current directory:[/red] "
            f"{escape(target)}\n[dim]Use a path inside {escape(str(cwd))}[/dim]",
            soft_wrap=True,
        )
        return
    _save_chat(messages, str(resolved))


def _handle_command(
    raw: str,
    messages: list,
    gen_opts: dict,
    pending_images: Optional[list] = None,
    engine=None,
) -> bool:
    """Handle a /command. Returns True if the session should exit."""
    from rich.markup import escape

    # Resolve console + HOME_DIR from the package at call time so tests that
    # monkeypatch localm.cli.console / localm.cli.HOME_DIR affect this call site.
    from localm import cli as _cli
    console = _cli.console
    HOME_DIR = _cli.HOME_DIR
    parts = raw[1:].split(" ", 1)
    cmd  = parts[0].lower()
    arg  = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("exit", "quit", "q", "bye"):
        console.print("[dim]Bye.[/dim]")
        return True
    elif cmd in ("generate-image", "imagine"):
        _cmd_generate_image(cmd, arg, engine, console, HOME_DIR)
    elif cmd == "generate-music":
        _cmd_generate_media("generate-music", arg, engine, console, HOME_DIR)
    elif cmd == "generate-video":
        _cmd_generate_media("generate-video", arg, engine, console, HOME_DIR)
    elif cmd == "compact":
        if engine is None:
            console.print("[dim]/compact not available in this mode[/dim]")
        else:
            from ..inference.compact import compact_messages
            new_messages, changed = compact_messages(
                messages,
                generate=lambda m, max_tok: "".join(
                    engine.chat_stream(m, max_tokens=max_tok, temperature=0.3)),
            )
            if changed:
                messages[:] = new_messages
                console.print("[dim]Older conversation summarised.[/dim]")
            else:
                console.print("[dim]Nothing to compact yet.[/dim]")
    elif cmd == "clear":
        messages[:] = [m for m in messages if m["role"] == "system"]
        if pending_images is not None:
            pending_images.clear()
        console.print("[dim]Cleared.[/dim]")
    elif cmd == "image":
        if pending_images is None:
            console.print("[dim]/image not available in this mode[/dim]")
        elif not arg:
            console.print("[dim]Usage: /image <file path>[/dim]")
        else:
            p = Path(arg)
            if not p.exists():
                console.print(f"[red]File not found:[/red] {escape(arg)}", soft_wrap=True)
            else:
                pending_images.append(str(p.resolve()))
                console.print(
                    f"[dim]Queued {escape(p.name)} - will attach to your next "
                    f"message[/dim]",
                    soft_wrap=True,
                )
    elif cmd == "images":
        if pending_images:
            for f in pending_images:
                console.print(f"[dim]  {escape(f)}[/dim]", soft_wrap=True)
        else:
            console.print("[dim]No images queued.[/dim]")
    elif cmd == "system":
        if arg:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "system":
                    messages.pop(i)
            messages.insert(0, {"role": "system", "content": arg})
            console.print("[dim]System prompt updated.[/dim]")
        else:
            console.print("[dim]Usage: /system <text>[/dim]")
    elif cmd == "save":
        _cmd_save(arg, messages, console)
    elif cmd == "temp":
        try:
            requested = float(arg)
        except ValueError:
            console.print("[dim]Usage: /temp 0.7[/dim]")
        else:
            # Clamp to a sane sampling range; negative or huge temps are not
            # meaningful and would otherwise be stored verbatim.
            clamped = max(0.0, min(2.0, requested))
            gen_opts["temperature"] = clamped
            if clamped != requested:
                console.print(
                    f"[dim]temperature clamped to {clamped} (valid range 0..2)[/dim]"
                )
            else:
                console.print(f"[dim]temperature = {clamped}[/dim]")
    elif cmd == "tokens":
        try:
            requested = int(arg)
        except ValueError:
            console.print("[dim]Usage: /tokens 2048[/dim]")
        else:
            # max_tokens must be at least 1; cap absurd values so a typo cannot
            # request an effectively unbounded generation.
            MAX_TOKENS_CAP = 1_000_000
            clamped = max(1, min(MAX_TOKENS_CAP, requested))
            gen_opts["max_tokens"] = clamped
            if clamped != requested:
                console.print(
                    f"[dim]max_tokens clamped to {clamped} "
                    f"(valid range 1..{MAX_TOKENS_CAP})[/dim]"
                )
            else:
                console.print(f"[dim]max_tokens = {clamped}[/dim]")
    elif cmd == "help":
        console.print(
            "[dim]"
            "/exit                   quit\n"
            "/clear                  clear chat history\n"
            "/image <path>           queue a local image for the next message\n"
            "/images                 list queued images\n"
            "/system <text>          set system prompt\n"
            "/save [file]            save conversation to JSON\n"
            "/compact                summarise older turns to free context\n"
            "/generate-image <prompt> generate an image via ComfyUI FLUX\n"
            "/generate-music <tags>  generate music via ComfyUI ACE-Step\n"
            "/generate-video <prompt> generate a clip via ComfyUI Wan\n"
            "/temp <float>           sampling temperature\n"
            "/tokens <int>           max response tokens"
            "[/dim]"
        )
    else:
        hint = None
        from ..config import load_config
        if load_config().get("suggest_plugins", True):
            from ..plugins import catalog
            hint = catalog.suggestion(cmd)
        if hint:
            # escape(): hint is only ever non-None for a command the catalog
            # itself recognises (a fixed, hardcoded set), so no bracket can reach
            # here. Escaped anyway as defense-in-depth.
            console.print(f"[yellow]{escape(hint)}[/yellow]")
        else:
            console.print(f"[dim]Unknown: /{escape(cmd)} -- try /help[/dim]")
    return False




def _save_chat(messages: list, filepath: str) -> None:
    from rich.markup import escape

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        console.print(f"[green]✓[/green] Saved: {escape(filepath)}", soft_wrap=True)
    except Exception as e:
        console.print(f"[red]Save failed: {escape(str(e))}[/red]", soft_wrap=True)
