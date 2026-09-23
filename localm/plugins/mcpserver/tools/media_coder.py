# SPDX-License-Identifier: AGPL-3.0-or-later
"""MCP tools that run a local generation pipeline: ``generate_image`` (FLUX
via ComfyUI) and ``run_coder_task`` (the coder agent, in this process).

``run_coder_task`` reads the timeout cap and the server log function from the
server module at call time, so they resolve to whatever that module currently
binds.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

from localm import pathsafe
from localm.pathsafe import is_unc_or_device_path

from .. import server as _srv
from ..server import EngineCache, _quiet_stdout, _text_result
from ._common import MODEL_PARAM


def build(engines: EngineCache) -> Dict[str, dict]:
    """``generate_image`` and ``run_coder_task``. The coder agent runs on an
    engine from *engines*, pinned for the length of the run."""
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

        def _confine(raw: str, label: str):
            """Keep an MCP OUTPUT path inside the localm data dir - this tool is
            driven by an LLM client, so an arbitrary output_path could overwrite
            anything on disk.

            WRITE targets only. ``input_image`` does NOT use this: confining a
            READ to the data dir is far too wide, because the data dir is the
            credential store (auth.key is the plaintext owner key, plus
            auth.json, sessions.json, rag/, coder/).

            Delegates to ``pathsafe.confined_absolute_or_under``, which carries
            the UNC/device guard and additionally closes an NTFS Alternate Data
            Stream / short-name-alias gap. Every rejection reason is folded into
            the SAME message, never echoing the client-supplied string back."""
            expanded = str(Path(raw).expanduser())
            try:
                return pathsafe.confined_absolute_or_under(home, expanded)
            except ValueError:
                raise ValueError(
                    f"{label} must stay within the localm data dir ({home})")

        try:
            out_arg = args.get("output_path")
            out = (_confine(out_arg, "output_path") if out_arg
                   else home / "mcp-images" / f"mcp-{int(time.time())}.png")
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
        with _quiet_stdout():
            ok, message = gen_img(
                prompt, out,
                api_url=api_url,
                guidance=args.get("guidance"),
                negative_prompt=args.get("negative_prompt"),
                seed=args.get("seed"),
                input_image=input_p,
                denoise=args.get("denoise"),
                write_sidecar=not is_privacy,
                delete_outputs=is_privacy,
            )
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
                yes=bool(args.get("yes")))
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
        # default model that cannot emit structured tool calls gives way to an
        # installed model that can.
        try:
            decision = engines.route(cfg.model, [], required=("tool_use",),
                                     pinned=bool(cfg.model))
        except ValueError as e:
            return _text_result(str(e), is_error=True)
        names = list(decision.candidates or (decision.resolved,)) if decision.routed else []
        names.append(decision.current)
        engine = model_name = None
        load_errors = []
        for name in names:
            try:
                with _quiet_stdout():
                    engine = engines.get_chat(name)
                model_name = name
                break
            except ValueError as e:
                return _text_result(str(e), is_error=True)
            except Exception as e:
                if name == decision.current:
                    return _text_result(f"coder task failed to start: {e}",
                                        is_error=True)
                load_errors.append(f"{name}: {e}")
                _srv._log(f"warning: could not load {name} for a coder task: {e}")
        if model_name != decision.resolved:
            decision = decision.without_route(load_errors)
        if engines.is_peer(engine):
            # Another localm instance's loaded copy, reached over its own API.
            from localm.plugins.coder.backends.http import HTTPBackend
            backend = HTTPBackend(
                getattr(engine, "_base"), model=getattr(engine, "_model", None) or model_name,
                api_key=getattr(engine, "_token", None) or "localm",
                localm_server=True)
        else:
            backend = SharedEngineBackend(
                engine, model_name, lock=engines.generation_lock(engine),
                still_resident=lambda: engines.is_resident(model_name, engine))

        # Default OFF, matching the CLI's own fail-closed default: without
        # `yes` file writes still happen but run_shell is denied, since there
        # is nobody to confirm it.
        if coder_runner.unattended_shell_gated(task, cfg.auto_approve):
            _srv._log("coder task: run_shell is denied for this run (no 'yes')")

        # Pinned for the whole run; released on the worker thread when the run
        # ends, even after a timeout has abandoned it.
        engines.pin(engine)
        try:
            with _quiet_stdout():
                agent = coder_runner.build_agent(
                    backend, work_dir, task=task, max_turns=cfg.max_turns,
                    auto_approve=cfg.auto_approve,
                    always_confirm=cfg.always_confirm,
                    session_mode=cfg.session_mode, gen_kw=cfg.gen_kw,
                    browser_enabled=coder_runner.browser_enabled(),
                    max_tokens_explicit=cfg.max_tokens_explicit)
        except Exception as e:
            engines.unpin(engine)
            return _text_result(f"coder task failed to start: {e}", is_error=True)
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            try:
                with _quiet_stdout():
                    coder_runner.finish_agent(agent)
            finally:
                engines.unpin(engine)
            return _text_result(
                f"coder task timed out after {timeout:g}s before it could start: "
                "loading the model and preparing the agent used the whole budget",
                is_error=True)
        try:
            with _quiet_stdout():
                result = coder_runner.run_task_with_timeout(
                    agent, task, remaining,
                    on_finished=lambda: engines.unpin(engine))
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

    return {
        "generate_image": {
            "description": ("Generate an image with the local FLUX model via ComfyUI. "
                            "Returns the saved file path and seed."),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt":          {"type": "string", "description": "Image description"},
                    "output_path":     {"type": "string", "description": "Where to save (default: <data dir>/mcp-images/)"},
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
                    "yes": {"type": "boolean",
                            "description": ("Auto-approve shell commands too, not just file "
                                             "writes (default false: shell calls are denied "
                                             "without this since there is no TTY to confirm "
                                             "them)")},
                    "timeout_seconds": {"type": "integer",
                                        "description": ("Give up after this long (default 900, "
                                                        "at most 3600). Covers loading the "
                                                        "model and preparing the agent as "
                                                        "well as the run; on expiry the run is "
                                                        "cancelled")},
                },
                "required": ["task", "cwd"],
            },
            "handler": run_coder_task,
        },
    }
