# SPDX-License-Identifier: AGPL-3.0-or-later
"""MCP tools that run a local generation pipeline: ``generate_image`` (the
image plugin's backend: built-in stable-diffusion.cpp or ComfyUI) and
``run_coder_task`` (the coder agent, in this process).

``run_coder_task`` reads the timeout cap and the server log function from the
server module at call time, so they resolve to whatever that module currently
binds.
"""

from __future__ import annotations

import dataclasses
import os
import secrets
import threading
import time
from pathlib import Path

from localm import pathsafe
from localm.pathsafe import is_unc_or_device_path

from .. import server as _srv
from ..server import (EngineCache, ModelEvictedError, ModelLoadSkipped, _quiet_stdout,
                      _text_result)
from ._common import MODEL_PARAM


def _peer_unusable(error: BaseException) -> bool:
    """True for a failure that means another instance's copy of a model cannot
    be used: that instance could not be reached or did not answer, or it
    refused the credential. An HTTP error it answered with is not one."""
    import requests
    from localm.plugins.coder.backends.http import CoderAuthError
    if isinstance(error, CoderAuthError):
        return True
    return (isinstance(error, requests.RequestException)
            and not isinstance(error, requests.HTTPError))


class PeerCoderBackend:
    """The coder backend for a model another localm instance has loaded.

    Every call goes to that instance through *http* until one fails in a way
    ``_peer_unusable`` accepts; then the instance is dropped from *engines*,
    the model is loaded here and pinned until ``release()``, and that call and
    every later one are answered by this server's own copy. No switch happens
    after ``cancel()`` or ``release()``. A sub-agent cannot ask for a
    different model on this backend. Any other attribute is the current
    backend's."""

    supports_model_override = False

    def __init__(self, engines: EngineCache, name: str, peer, http) -> None:
        self._engines = engines
        self._name = name
        self._peer = peer
        self._http = http
        self._current = http
        self._local = None
        self._released = False
        self._cancel_reason = None
        self._switch_lock = threading.Lock()

    def __getattr__(self, attr):
        current = self.__dict__.get("_current")
        if current is None:
            raise AttributeError(attr)
        return getattr(current, attr)

    @property
    def local_engine(self):
        """This server's engine the run moved to, or None while it has not."""
        return self._local[0] if self._local else None

    def chat(self, messages: list, **kwargs) -> str:
        backend = self._current
        try:
            return backend.chat(messages, **kwargs)
        except Exception as e:
            if not self._switch(backend, e):
                raise
        return self._current.chat(messages, **kwargs)

    def chat_stream(self, messages: list, on_reasoning=None, **kwargs):
        backend = self._current
        started = False

        def _reasoning(piece):
            nonlocal started
            started = True
            if on_reasoning is not None:
                on_reasoning(piece)

        try:
            for piece in backend.chat_stream(messages, on_reasoning=_reasoning, **kwargs):
                started = True
                yield piece
            return
        except Exception as e:
            if started or not self._switch(backend, e):
                raise
        yield from self._current.chat_stream(messages, on_reasoning=on_reasoning, **kwargs)

    def _switch(self, failed, error: BaseException) -> bool:
        """Move the run to a copy loaded here when *error*, raised by *failed*,
        means the peer can no longer be used. Returns whether later calls go
        to that copy. Raises when loading it here failed."""
        if failed is not self._http or not _peer_unusable(error):
            return False
        with self._switch_lock:
            if self._local is not None:
                return True
            if self._released or self._cancel_reason is not None:
                return False
            _srv._log(f"the localm instance answering {self._name} for a coder "
                      f"task failed ({error}); loading it here")
            try:
                engine = self._engines.load_here_instead_of(self._name, self._peer)
            except Exception as e:
                raise RuntimeError(
                    f"the localm instance answering {self._name} failed ({error}), "
                    f"and loading {self._name} here failed: {e}") from e
            try:
                from localm.plugins.coder.backends.shared_engine import SharedEngineBackend
                local = SharedEngineBackend(
                    engine, self._name, lock=self._engines.generation_lock(engine),
                    still_resident=lambda: self._engines.is_resident(self._name, engine))
            except Exception:
                self._engines.unpin(engine)
                raise
            self._local = (engine, local)
            self._current = local
        if self._cancel_reason is not None:
            local.cancel(self._cancel_reason)
        return True

    def cancel(self, reason: str = "cancelled") -> None:
        """Refuse any later switch, and cancel the current backend when it can
        be cancelled."""
        self._cancel_reason = reason or "cancelled"
        abort = getattr(self._current, "cancel", None)
        if callable(abort):
            abort(self._cancel_reason)

    def release(self) -> None:
        """Refuse any later switch, and unpin the copy loaded here, if any."""
        with self._switch_lock:
            if self._released:
                return
            self._released = True
            local = self._local
        if local is not None:
            self._engines.unpin(local[0])


def coder_engine(engines: EngineCache, decision):
    """The loaded engine for a coder task under *decision*: each routed
    candidate in order, then the model the task would otherwise use. A
    candidate that cannot be built or loaded is skipped. Returns
    ``(engine, name, decision)``, the decision rewritten to name the model
    whose engine is returned. Raises what getting that last model raised."""
    names = list(decision.candidates or (decision.resolved,)) if decision.routed else []
    load_errors = []
    skipped = []
    for name in names:
        try:
            with _quiet_stdout():
                engine = engines.get_loaded_chat(
                    name, skip_if_latched=name != decision.current)
        except ModelLoadSkipped as e:
            skipped.append(e.skipped)
            _srv._log(f"not loading {name} for a coder task: {e}")
            continue
        except Exception as e:
            load_errors.append(f"{name}: {e}")
            _srv._log(f"warning: could not load {name} for a coder task: {e}")
            continue
        if skipped:
            decision = decision.with_skipped(skipped)
        if name != decision.resolved:
            decision = decision.answered_by(name)
        return engine, name, decision
    with _quiet_stdout():
        engine = engines.get_loaded_chat(decision.current)
    if skipped:
        decision = decision.with_skipped(skipped)
    if decision.routed:
        decision = decision.without_route(load_errors)
    return engine, decision.current, decision


_MCP_IMAGES_DIR = "mcp-images"
_OUTPUT_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def _inside(path: Path, root: Path) -> bool:
    """True when the resolved *path* is *root* or below it."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def build(engines: EngineCache, coder_shell: bool = False,
          coder_root: Path | None = None) -> dict[str, dict]:
    """``generate_image`` and ``run_coder_task``. The coder agent runs on an
    engine from *engines*, pinned for the length of the run.

    *coder_shell*: the operator allowed the coder to run shell commands
    unattended; without it ``run_coder_task`` never auto-approves shell, whatever
    ``yes`` or a project config say. *coder_root*: when set, ``cwd`` must resolve
    inside it."""
    def generate_image(args: dict) -> dict:
        prompt = args.get("prompt", "")
        if not prompt:
            return _text_result("'prompt' is required", is_error=True)
        from localm.audit import SessionMode, effective_mode
        from localm.config import home_dir, load_config
        from localm.image_gen.comfy import generate_image as gen_img
        from localm.media import paths as _media_paths

        home = home_dir().resolve()

        try:
            from localm.plugins.builtin.image import backend as _image_backend
            api_url = _image_backend.settings(load_config()).get("api_url") or None
        except Exception as exc:                     # noqa: BLE001
            from localm.debuglog import logger
            logger.debug("could not resolve the image plugin ComfyUI url (%s); "
                         "using the shared default", exc)
            api_url = None
        if not api_url:
            from localm.media.comfy_client import default_api_url
            api_url = default_api_url()

        def _confine_output(raw: str) -> Path:
            """Resolve an MCP ``output_path`` to a NEW image file in a
            generated-media folder of the data dir: ``mcp-images`` (where a
            relative path is placed) or the image gallery (``gui_images``),
            given as an absolute path.

            Raises ``ValueError`` when the path is outside those folders, is a
            UNC/device path or carries a reserved character (via
            ``pathsafe.confined_absolute_or_under``), does not end in an image
            extension, or names a file (or its ``.json`` sidecar) that already
            exists. The message never echoes the client-supplied string. The
            path is not created here; ``_reserve`` does that."""
            refused = ValueError(
                "output_path must name a new image file ("
                + ", ".join(_OUTPUT_IMAGE_SUFFIXES)
                + f") in the data dir's {_MCP_IMAGES_DIR} or "
                f"{_media_paths.IMAGE_DIR_NAME} folder")
            expanded = str(Path(raw).expanduser())
            mcp_root = home / _MCP_IMAGES_DIR
            absolute = Path(expanded).is_absolute()
            roots = ([mcp_root, home / _media_paths.IMAGE_DIR_NAME] if absolute
                     else [mcp_root])
            resolved = None
            for root in roots:
                try:
                    resolved = pathsafe.confined_absolute_or_under(root, expanded)
                except ValueError:
                    continue
                if resolved != root.resolve():
                    break
                resolved = None
            if resolved is None:
                raise refused
            if resolved.suffix.lower() not in _OUTPUT_IMAGE_SUFFIXES:
                raise refused
            sidecar = resolved.with_suffix(resolved.suffix + ".json")
            named = Path(expanded) if absolute else mcp_root / expanded
            if any(os.path.lexists(c) for c in (named, resolved, sidecar)):
                raise refused
            return resolved

        def _reserve(path: Path) -> None:
            """Create *path* as an empty file, failing if anything exists there,
            so the backend only ever replaces a file this call created. Raises
            ``ValueError`` when the name is taken and ``OSError`` when the file
            cannot be created."""
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(path, "xb"):
                    pass
            except FileExistsError as e:
                raise ValueError("output_path names a file that already exists") from e

        def _release(path: Path) -> None:
            """Remove the reservation of *path* when nothing was written into it."""
            try:
                if path.stat().st_size == 0:
                    path.unlink()
            except OSError as e:
                from localm.debuglog import logger
                logger.warning("could not remove the empty image placeholder (%s)", e)

        try:
            out_arg = args.get("output_path")
            out = (_confine_output(out_arg) if out_arg
                   else home / _MCP_IMAGES_DIR
                   / f"mcp-{int(time.time())}-{secrets.token_hex(4)}.png")
            # input_image is a READ that is then UPLOADED to ComfyUI, over an
            # api_url sanitize_comfy_url permits to be a LAN or public host on
            # plaintext http, so it is read-AND-TRANSMIT and the data dir is the
            # wrong boundary for it. Same policy the image/video HTTP routes use
            # (uploads inbox + the generated-media galleries), via the non-HTTP
            # entry point so a refusal becomes an error REPLY here rather than an
            # HTTPException escaping the stdio handler. InputImageRefused is a
            # ValueError, so the existing except catches it.
            input_p = (_media_paths.check_input_image(args["input_image"])
                       if args.get("input_image") else None)
        except ValueError as e:
            return _text_result(str(e), is_error=True)

        is_privacy = effective_mode("mcp") == SessionMode.PRIVACY
        # comfy.generate_image builds its own rich Console / Progress on stdout;
        # the JSON-RPC frame stream lives on stdout too, so route any stray
        # output to stderr or it corrupts the protocol.
        from localm.plugins.builtin.image import backend as image_backend

        def _generate():
            with _quiet_stdout():
                native = image_backend.generate_unless_comfy(
                    prompt, out,
                    guidance=args.get("guidance"),
                    negative_prompt=args.get("negative_prompt"),
                    seed=args.get("seed"),
                    input_image=input_p,
                    denoise=args.get("denoise"),
                    write_sidecar=not is_privacy,
                    delete_outputs=is_privacy,
                    on_progress=_srv.report_progress,
                )
            if native is not None:
                return native
            with _quiet_stdout():
                return gen_img(
                    prompt, out,
                    api_url=api_url,
                    guidance=args.get("guidance"),
                    negative_prompt=args.get("negative_prompt"),
                    seed=args.get("seed"),
                    input_image=input_p,
                    denoise=args.get("denoise"),
                    write_sidecar=not is_privacy,
                    delete_outputs=is_privacy,
                    on_progress=_srv.report_progress,
                )

        try:
            _reserve(out)
        except ValueError as e:
            return _text_result(str(e), is_error=True)
        except OSError as e:
            return _text_result(
                f"could not create the output file: {e.strerror or type(e).__name__}",
                is_error=True)
        ok = False
        try:
            ok, message = _generate()
        finally:
            if not ok:
                _release(out)
        return _text_result(message, is_error=not ok)

    def run_coder_task(args: dict) -> dict:
        task = args.get("task", "")
        if not task:
            return _text_result("'task' is required", is_error=True)
        cwd = args.get("cwd", "")
        if not cwd:
            return _text_result("'cwd' is required (the project directory to work in)",
                                 is_error=True)
        # `cwd` is MCP-client-supplied, same as pull_model's `repo` above:
        # refuse UNC/device syntax unconditionally, BEFORE is_dir() below ever
        # runs. is_dir() dials SMB for a UNC target exactly like exists() does.
        if is_unc_or_device_path(cwd):
            return _text_result(
                "'cwd' must be a local directory path, not a UNC or device path.",
                is_error=True)
        cwd_path = Path(cwd).expanduser()
        if not cwd_path.is_dir():
            return _text_result(f"cwd is not a directory: {cwd_path}", is_error=True)
        if coder_root is not None and not _inside(cwd_path.resolve(), coder_root):
            return _text_result(
                "'cwd' must be inside the folder this server was started with "
                "(--coder-root).", is_error=True)

        # The coder Agent runs IN THIS PROCESS on an engine from this server's
        # own EngineCache: one resident model serves chat, embed and every
        # coder task, and no per-project server is spawned. The project's own
        # .localcoder/config.toml is honoured exactly as `localm coder` does.
        from localm.plugins.coder import runner as coder_runner
        from localm.plugins.coder.backends.shared_engine import SharedEngineBackend
        from localm.plugins.coder.project_config import ProjectConfigUnreadable

        work_dir = cwd_path.resolve()
        max_turns = args.get("max_turns")
        if max_turns is not None:
            try:
                max_turns = int(max_turns)
            except (TypeError, ValueError):
                return _text_result("'max_turns' must be an integer", is_error=True)
        try:
            cfg = coder_runner.resolve_task_config(
                work_dir, model=args.get("model") or None, max_turns=max_turns,
                yes=bool(args.get("yes")) and coder_shell, unattended=True)
        except ProjectConfigUnreadable as e:
            return _text_result(f"Project config could not be read: {e}", is_error=True)
        except coder_runner.InvalidSessionMode as e:
            return _text_result(str(e), is_error=True)
        timeout = args.get("timeout_seconds") or 900
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or timeout <= 0):
            return _text_result("'timeout_seconds' must be a positive number",
                                is_error=True)
        if timeout > _srv.MAX_CODER_TIMEOUT_SECONDS:
            return _text_result(
                f"'timeout_seconds' must be at most {_srv.MAX_CODER_TIMEOUT_SECONDS:g}",
                is_error=True)
        timeout = float(timeout)
        # One deadline for the whole call: the model load and the agent's
        # construction below spend from the same budget as the run.
        started = time.monotonic()

        # A model name from the client or the project config is registry-gated
        # by resolve_model; the operator's own --model default is the only path
        # allowed through. A named model is used as named; without one, a
        # default model that cannot emit structured tool calls, or whose
        # trained window is too small for the task text, gives way to an
        # installed model that has what it lacks.
        try:
            decision = engines.route(cfg.model, [{"role": "user", "content": task}],
                                     required=("tool_use",), pinned=bool(cfg.model))
        except ValueError as e:
            return _text_result(str(e), is_error=True)
        try:
            engine, model_name, decision = coder_engine(engines, decision)
        except ValueError as e:
            return _text_result(str(e), is_error=True)
        except Exception as e:
            return _text_result(f"coder task failed to start: {e}", is_error=True)
        peer_backend = None
        if engines.is_peer(engine):
            # Another localm instance's loaded copy, reached over its own API;
            # the model is loaded here if that instance stops answering.
            from localm.plugins.coder.backends.http import HTTPBackend
            peer_http = HTTPBackend(
                engine._base, model=getattr(engine, "_model", None) or model_name,
                api_key=getattr(engine, "_token", None) or "localm",
                localm_server=True, trusted_peer=True)
            peer_http.session_mode = cfg.session_mode
            backend = peer_backend = PeerCoderBackend(engines, model_name, engine,
                                                      peer_http)
        else:
            backend = SharedEngineBackend(
                engine, model_name, lock=engines.generation_lock(engine),
                still_resident=lambda: engines.is_resident(model_name, engine))

        # Shell approval is the operator's launch decision (--coder-shell):
        # without it neither `yes` nor a project's auto_approve is honoured, and
        # file writes still happen while run_shell is denied.
        if not coder_shell and cfg.auto_approve:
            cfg = dataclasses.replace(cfg, auto_approve=False)
        if coder_runner.unattended_shell_gated(task, cfg.auto_approve):
            _srv._log("coder task: run_shell is denied for this run "
                      + ("(no 'yes')" if coder_shell
                         else "(server started without --coder-shell)"))

        def _release():
            engines.unpin(engine)
            if peer_backend is not None:
                peer_backend.release()

        # Pinned for the whole run; released on the worker thread when the run
        # ends, even after a timeout has abandoned it.
        try:
            engines.pin(engine)
        except ModelEvictedError as e:
            return _text_result(f"coder task failed to start: {e}", is_error=True)
        try:
            with _quiet_stdout():
                agent = coder_runner.build_agent(
                    backend, work_dir, task=task, max_turns=cfg.max_turns,
                    auto_approve=cfg.auto_approve,
                    always_confirm=cfg.always_confirm,
                    session_mode=cfg.session_mode, gen_kw=cfg.gen_kw,
                    browser_enabled=coder_runner.browser_enabled(),
                    max_tokens_explicit=cfg.max_tokens_explicit,
                    server_side=True)
        except Exception as e:
            _release()
            return _text_result(f"coder task failed to start: {e}", is_error=True)
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            try:
                with _quiet_stdout():
                    coder_runner.finish_agent(agent)
            finally:
                _release()
            return _text_result(
                f"coder task timed out after {timeout:g}s before it could start: "
                "loading the model and preparing the agent used the whole budget",
                is_error=True)
        try:
            with _quiet_stdout():
                result = coder_runner.run_task_with_timeout(
                    agent, task, remaining, on_finished=_release,
                    requested_timeout=timeout)
        except Exception as e:
            return _text_result(f"coder task failed to run: {e}", is_error=True)

        denied = coder_runner.describe_denied(result.denied)
        meta = (f"\n\n[turns={result.turns} tokens={result.total_tokens} "
                f"success={result.success} denied={len(result.denied)}]")
        from .chat import routing_note
        note = routing_note(decision)
        if note:
            meta += "\n" + note
        text = result.response + ("\n\n[denied] " + denied if denied else "") + meta
        return _text_result(text, is_error=not result.success)

    if coder_shell:
        yes_description = (
            "Auto-approve shell commands too, not just file writes (default "
            "false: shell calls are denied without this since there is no TTY "
            "to confirm them)")
    else:
        yes_description = (
            "Accepted for compatibility. This server was started without "
            "--coder-shell, so shell calls are always denied and file writes "
            "are the only thing auto-approved")

    return {
        "generate_image": {
            "description": ("Generate an image locally with the image plugin's backend "
                            "(the built-in stable-diffusion.cpp runtime, or ComfyUI). "
                            "Returns the saved file path and seed."),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt":          {"type": "string", "description": "Image description"},
                    "output_path":     {"type": "string", "description": "New image file to write: a relative path under <data dir>/mcp-images/, or an absolute path inside <data dir>/mcp-images/ or <data dir>/gui_images/; .png, .jpg, .jpeg or .webp; an existing file is refused (default: a new file in <data dir>/mcp-images/)"},
                    "negative_prompt": {"type": "string", "description": "Things to avoid"},
                    "seed":            {"type": "integer", "description": "Reproducibility seed"},
                    "guidance":        {"type": "number", "description": "Guidance scale (default 3.5)"},
                    "input_image":     {"type": "string", "description": "Existing image for img2img"},
                    "denoise":         {"type": "number", "description": "img2img change amount 0-1"},
                },
                "required": ["prompt"],
            },
            "handler": generate_image,
        },
        "run_coder_task": {
            "description": (
                "Delegate a whole coding task (read/edit files, run shell commands, "
                "git, tests) to localm's own offline agent, running entirely on a "
                "local model. Blocks until the task finishes or times out, then "
                "returns the agent's final result - use this to hand off a "
                "self-contained sub-task instead of doing it turn-by-turn yourself. "
                "Nobody can confirm a tool call during the run: a call the agent "
                "makes that needs a confirmation is denied, the result names it "
                "and reports success=False."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The task to perform"},
                    "cwd":  {"type": "string",
                             "description": "Project directory the agent should work in"},
                    "model": MODEL_PARAM,
                    "max_turns": {"type": "integer",
                                  "description": "Safety cap on agent iterations (default: 40)"},
                    "yes": {"type": "boolean", "description": yes_description},
                    "timeout_seconds": {"type": "integer",
                                        "description": ("Give up after this long (default 900, "
                                                        "at most 3600). Loading the model and "
                                                        "preparing the agent count against it: "
                                                        "a load that outlasts it ends the call "
                                                        "as soon as the load finishes. On expiry "
                                                        "during the run, the run is cancelled")},
                },
                "required": ["task", "cwd"],
            },
            "handler": run_coder_task,
        },
    }
