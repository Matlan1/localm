# SPDX-License-Identifier: AGPL-3.0-or-later
"""
MCP server over stdio - exposes localm to any MCP client.

Protocol: JSON-RPC 2.0, newline-delimited JSON on stdin/stdout (the MCP
stdio transport - the mirror image of plugins/coder/mcp.py, which is the
client side).

CRITICAL INVARIANT: stdout carries ONLY protocol messages. Everything in
this process that would normally print (model loading banners, VRAM info,
rich progress) must go to stderr - see _redirect_consoles_to_stderr().

Tools exposed:
    chat            - generate a response with a local model
    list_models     - registered model names with type and size
    embed           - embedding vectors (models that support it)
    generate_image  - local FLUX via ComfyUI (omit with --no-images)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROTOCOL_VERSION = "2025-03-26"
SERVER_NAME = "localm"
SERVER_VERSION = "0.1.0"


def _log(msg: str) -> None:
    """Server-side logging - stderr only, stdout belongs to the protocol."""
    print(f"[localm-mcp] {msg}", file=sys.stderr, flush=True)


def _redirect_consoles_to_stderr() -> None:
    """
    Re-point every rich Console used during model loading at stderr.
    A single stray print to stdout corrupts the JSON-RPC stream.
    """
    from rich.console import Console
    err = Console(stderr=True)
    import localm.inference.engine as _engine_mod
    import localm.inference.backends.gguf as _gguf_mod
    import localm.model_manager as _mm_mod
    _engine_mod.console = err
    _gguf_mod.console = err
    _mm_mod.console = err
    try:
        import localm.inference.backends.llamacpp.llama as _llama_mod
        if hasattr(_llama_mod, "console"):
            _llama_mod.console = err
    except Exception:
        pass


# ---------------------------------------------------------------------------
#  Tool implementations
# ---------------------------------------------------------------------------

class EngineCache:
    """Lazy, per-model engine cache. Only one model loaded at a time."""

    def __init__(self, default_model: Optional[str] = None,
                 engine_factory: Optional[Callable] = None) -> None:
        self.default_model = default_model
        self._engine = None
        self._loaded_name: Optional[str] = None
        # Injection point for tests - real factory builds a localm Engine
        self._factory = engine_factory or self._build_engine

    @staticmethod
    def _build_engine(model_name: str):
        from localm.inference.engine import Engine
        from localm.model_manager import get_model_info
        info = get_model_info(model_name)
        if info is None:
            raise ValueError(f"Model not found: {model_name!r}. "
                             f"Run 'localm list' to see registered models.")
        path, _hint = info
        return Engine(str(path), display_name=model_name)

    def resolve_model(self, requested: Optional[str]) -> str:
        name = requested or self.default_model
        if name:
            return name
        from localm.config import load_registry
        reg = load_registry()
        if not reg:
            raise ValueError("No models registered. Run 'localm pull <name>' first.")
        return sorted(reg)[0]

    def get(self, requested: Optional[str]):
        name = self.resolve_model(requested)
        if self._engine is not None and self._loaded_name == name:
            return self._engine
        if self._engine is not None:
            _log(f"switching model {self._loaded_name} -> {name}")
            try:
                self._engine.unload()
            except Exception:
                pass
        _log(f"loading model {name}")
        self._engine = self._factory(name)
        self._loaded_name = name
        return self._engine


def _text_result(text: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _backend_can_embed(engines: "EngineCache") -> bool:
    """True unless the active/default backend explicitly cannot embed.

    The common GGUF backend has no embedding support and would make the 'embed'
    tool error on every call (FAC-6), so we hide it. We resolve the default
    engine WITHOUT loading the model (constructing the backend is cheap) and
    read its ``can_embed`` flag; a backend that does not set the flag is treated
    as embed-capable (e.g. the HuggingFace backend), and any failure to resolve
    a model leaves the tool advertised rather than silently dropping it."""
    try:
        engine = engines.get(None)
    except Exception:
        return True  # cannot determine - do not hide a possibly-working tool
    backend = getattr(engine, "_backend", None)
    return getattr(backend, "can_embed", True) is not False


def build_tools(engines: EngineCache, enable_images: bool = True) -> Dict[str, dict]:
    """Return {tool_name: {schema, handler}} for everything this server offers."""

    def chat(args: dict) -> dict:
        prompt = args.get("prompt", "")
        if not prompt:
            return _text_result("'prompt' is required", is_error=True)
        engine = engines.get(args.get("model"))
        messages = []
        if args.get("system"):
            messages.append({"role": "system", "content": args["system"]})
        messages.append({"role": "user", "content": prompt})
        gen: dict = {}
        for key in ("max_tokens", "temperature", "seed"):
            if args.get(key) is not None:
                gen[key] = args[key]
        text = "".join(engine.chat_stream(messages, **gen))
        return _text_result(text)

    def list_models(args: dict) -> dict:
        from localm.config import load_registry
        reg = load_registry()
        if not reg:
            return _text_result("No models registered.")
        lines = []
        for name, info in sorted(reg.items()):
            p = Path(info.get("path", ""))
            if p.is_dir():
                size = "dir (HF format)"
            elif p.is_file():
                b = p.stat().st_size
                size = f"{b/1024**3:.2f} GB" if b >= 1024**3 else f"{b/1024**2:.0f} MB"
            else:
                size = "missing"
            lines.append(f"{name}  [{size}]  {info.get('source', 'local')}")
        return _text_result("\n".join(lines))

    def embed(args: dict) -> dict:
        texts = args.get("texts")
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return _text_result("'texts' is required (string or list)", is_error=True)
        engine = engines.get(args.get("model"))
        try:
            vecs = engine.embed(texts)
        except NotImplementedError as e:
            return _text_result(str(e), is_error=True)
        return _text_result(json.dumps(vecs))

    def generate_image(args: dict) -> dict:
        prompt = args.get("prompt", "")
        if not prompt:
            return _text_result("'prompt' is required", is_error=True)
        import contextlib
        from localm.audit import SessionMode, effective_mode
        from localm.config import home_dir
        from localm.image_gen.comfy import generate_image as gen_img

        home = home_dir().resolve()

        def _confine(raw: str, label: str):
            """Keep MCP file paths inside the localm data dir - this tool is
            driven by an LLM client, so an arbitrary output_path/input_image
            could read or overwrite anything on disk (SEC-7)."""
            p = Path(raw).expanduser()
            p = p if p.is_absolute() else home / p
            p = p.resolve()
            if not p.is_relative_to(home):
                raise ValueError(
                    f"{label} must stay within the localm data dir ({home})")
            return p

        try:
            out_arg = args.get("output_path")
            out = (_confine(out_arg, "output_path") if out_arg
                   else home / "mcp-images" / f"mcp-{int(time.time())}.png")
            input_p = _confine(args["input_image"], "input_image") if args.get("input_image") else None
        except ValueError as e:
            return _text_result(str(e), is_error=True)

        write_sidecar = effective_mode("mcp") != SessionMode.PRIVACY
        # comfy.generate_image builds its own rich Console / Progress on stdout;
        # the JSON-RPC frame stream lives on stdout too, so route any stray
        # output to stderr or it corrupts the protocol (BUG-11).
        with contextlib.redirect_stdout(sys.stderr):
            ok, message = gen_img(
                prompt, out,
                guidance=args.get("guidance"),
                negative_prompt=args.get("negative_prompt"),
                seed=args.get("seed"),
                input_image=input_p,
                denoise=args.get("denoise"),
                write_sidecar=write_sidecar,
            )
        return _text_result(message, is_error=not ok)

    _model_param = {"type": "string",
                    "description": "Registered model name (default: server's configured model)"}

    tools: Dict[str, dict] = {
        "chat": {
            "description": "Generate a response with a local LLM (fully offline).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt":      {"type": "string", "description": "User prompt"},
                    "system":      {"type": "string", "description": "Optional system prompt"},
                    "model":       _model_param,
                    "max_tokens":  {"type": "integer", "description": "Max tokens to generate"},
                    "temperature": {"type": "number", "description": "Sampling temperature"},
                    "seed":        {"type": "integer", "description": "Seed for reproducible output"},
                },
                "required": ["prompt"],
            },
            "handler": chat,
        },
        "list_models": {
            "description": "List locally registered models with size and source.",
            "inputSchema": {"type": "object", "properties": {}},
            "handler": list_models,
        },
    }

    # Only advertise embed when the active backend can actually produce vectors
    # (FAC-6). The handler still degrades gracefully if invoked anyway, but a
    # tool that always errors should not appear in tools/list.
    if _backend_can_embed(engines):
        tools["embed"] = {
            "description": "Compute embedding vectors for one or more texts with a local model.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "texts": {"type": "array", "description": "Texts to embed",
                              "items": {"type": "string"}},
                    "model": _model_param,
                },
                "required": ["texts"],
            },
            "handler": embed,
        }

    if enable_images:
        tools["generate_image"] = {
            "description": ("Generate an image with the local FLUX model via ComfyUI. "
                            "Returns the saved file path and seed."),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt":          {"type": "string", "description": "Image description"},
                    "output_path":     {"type": "string", "description": "Where to save (default: ~/.localm/mcp-images/)"},
                    "negative_prompt": {"type": "string", "description": "Things to avoid"},
                    "seed":            {"type": "integer", "description": "Reproducibility seed"},
                    "guidance":        {"type": "number", "description": "Guidance scale (default 3.5)"},
                    "input_image":     {"type": "string", "description": "Existing image for img2img"},
                    "denoise":         {"type": "number", "description": "img2img change amount 0-1"},
                },
                "required": ["prompt"],
            },
            "handler": generate_image,
        }

    return tools


# ---------------------------------------------------------------------------
#  JSON-RPC dispatch
# ---------------------------------------------------------------------------

class MCPStdioServer:
    """Dispatches MCP JSON-RPC messages to tool handlers."""

    def __init__(self, tools: Dict[str, dict]) -> None:
        self.tools = tools

    def handle(self, msg: dict) -> Optional[dict]:
        """Process one message. Returns the response dict, or None for
        notifications (which get no reply)."""
        if not isinstance(msg, dict):
            # A JSON-RPC batch array, a bare scalar, or null all parse fine but
            # are not a request object - reply Invalid Request instead of
            # crashing on msg.get(...).
            return self._error(None, -32600, "Invalid Request: expected a JSON object")
        method = msg.get("method", "")
        mid = msg.get("id")

        if mid is None:
            return None   # notification (e.g. notifications/initialized)

        if method == "initialize":
            return self._result(mid, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            })

        if method == "ping":
            return self._result(mid, {})

        if method == "tools/list":
            listed = [
                {"name": name,
                 "description": spec["description"],
                 "inputSchema": spec["inputSchema"]}
                for name, spec in self.tools.items()
            ]
            return self._result(mid, {"tools": listed})

        if method == "tools/call":
            params = msg.get("params", {}) or {}
            name = params.get("name", "")
            spec = self.tools.get(name)
            if spec is None:
                return self._error(mid, -32602, f"Unknown tool: {name}")
            try:
                result = spec["handler"](params.get("arguments", {}) or {})
            except Exception as e:
                _log(f"tool {name} crashed: {e}")
                result = _text_result(f"Tool failed: {e}", is_error=True)
            return self._result(mid, result)

        return self._error(mid, -32601, f"Method not found: {method}")

    @staticmethod
    def _result(mid: Any, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    @staticmethod
    def _error(mid: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": code, "message": message}}

    # ------------------------------------------------------------------ #

    def run_stdio(self, stdin=None, stdout=None) -> None:
        """Blocking loop: read newline-delimited JSON until EOF."""
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        _log("ready - waiting for MCP client")
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                _log(f"skipping non-JSON input line")
                continue
            # A JSON-RPC payload may be a single request object or a batch
            # array; a bare scalar / null is invalid. handle() replies -32600
            # for any non-dict element rather than crashing the loop.
            if isinstance(msg, list):
                batch = msg or [None]      # empty batch -> one Invalid Request
            else:
                batch = [msg]
            for one in batch:
                response = self.handle(one)
                if response is not None:
                    stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                    stdout.flush()
        _log("stdin closed - shutting down")


def serve_stdio(model: Optional[str] = None, enable_images: bool = True) -> None:
    """Entry point used by the CLI: build everything and block on stdio."""
    _redirect_consoles_to_stderr()
    engines = EngineCache(default_model=model)
    server = MCPStdioServer(build_tools(engines, enable_images=enable_images))
    try:
        server.run_stdio()
    finally:
        if engines._engine is not None:
            try:
                engines._engine.unload()
            except Exception:
                pass
