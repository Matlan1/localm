# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Minimal MCP (Model Context Protocol) client - stdio transport, stdlib only.

Lets the coder agent use tools from any MCP server. Servers are declared in
``.localcoder/config.toml``:

    [mcp.servers.weather]
    command = "python"
    args = ["path/to/weather_server.py"]

    [mcp.servers.db]
    command = "npx"
    args = ["-y", "@modelcontextprotocol/server-sqlite", "app.db"]
    trusted = true          # skip the destructive-tool confirmation

No server is started until the user has trusted the project config's commands
(hook_trust.py), the same trust that lets its tool hooks run; ``trusted = true``
only skips the per-call confirmation once the server runs.

Each server is spawned as a child process speaking JSON-RPC 2.0 over
newline-delimited JSON on stdin/stdout (the MCP stdio transport). Its tools
are registered into TOOL_REGISTRY as ``mcp_<server>_<tool>`` so the agent
can call them like any built-in tool. Everything is local and offline -
whether a given server talks to the network is up to that server.

Servers are pooled per declared spec (name, command, args, env, trusted) for
the life of the process: a second registration of the same spec, by a later
one-shot run or a second session on the same project, reuses the live server
and its registry bindings instead of spawning another child. A registration
whose spec differs from the live binding under the same tool name is refused
with a warning, and the agent that was refused does not get that tool.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
from pathlib import Path
from typing import Optional

from localm import __version__ as _LOCALM_VERSION

from .proc_tail import StderrTail
from .provenance import neutralise
from .tool_registration import register_foreign_tool
from .tools import ToolResult

PROTOCOL_VERSION = "2025-03-26"

_INIT_TIMEOUT = 15      # seconds for initialize + tools/list
_CALL_TIMEOUT = 120     # seconds for a tool call

# Live servers by spec key, shared by every registration in this process.
_POOL: dict[tuple, MCPServer] = {}
_POOL_LOCK = threading.Lock()


class MCPError(Exception):
    """Raised when an MCP server misbehaves or a request fails."""


class MCPServer:
    """One running MCP server process and its JSON-RPC session."""

    def __init__(self, name: str, command: str, args: Optional[list[str]] = None,
                 env: Optional[dict[str, str]] = None, trusted: bool = False) -> None:
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env
        self.trusted = trusted
        self.tools: list[dict] = []
        self._proc: Optional[subprocess.Popen] = None
        self._stderr: Optional[StderrTail] = None
        self._responses: queue.Queue[dict] = queue.Queue()
        self._next_id = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Spawn the server, run the MCP handshake, and fetch its tool list."""
        import os
        from localm.config import user_command_env
        full_env = user_command_env({**os.environ, **self.env} if self.env else None)
        try:
            self._proc = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,   # drained by StderrTail, never DEVNULL
                text=True,
                encoding="utf-8",
                env=full_env,
            )
        except FileNotFoundError as e:
            raise MCPError(f"MCP server '{self.name}': command not found: {self.command}") from e
        self._stderr = StderrTail(self._proc)

        reader = threading.Thread(target=self._read_loop, daemon=True)
        reader.start()

        init = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "localcoder", "version": _LOCALM_VERSION},
        }, timeout=_INIT_TIMEOUT)
        if "error" in init:
            raise MCPError(self._with_tail(
                f"MCP server '{self.name}' rejected initialize: {init['error']}"))

        self._notify("notifications/initialized")

        listed = self._request("tools/list", {}, timeout=_INIT_TIMEOUT)
        if "error" in listed:
            raise MCPError(self._with_tail(
                f"MCP server '{self.name}' tools/list failed: {listed['error']}"))
        self.tools = listed.get("result", {}).get("tools", [])

    def _with_tail(self, msg: str) -> str:
        """*msg* plus the server's captured stderr tail, if it wrote anything."""
        tail = self._stderr.tail() if self._stderr else ""
        return f"{msg}:\n{tail}" if tail else msg

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                    # A killed child stays in the process table until it is
                    # waited on, and a zombie answers os.kill(pid, 0), so every
                    # pid-liveness check in the codebase reads it as running.
                    self._proc.wait(timeout=5)
                except Exception:
                    pass
        self._proc = None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ------------------------------------------------------------------ #
    #  JSON-RPC plumbing (newline-delimited JSON over stdio)              #
    # ------------------------------------------------------------------ #

    def _read_loop(self) -> None:
        """Background reader: queue every response object the server emits."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except (ValueError, RecursionError):
                continue   # servers may log junk to stdout - skip it
            if not isinstance(msg, dict):
                continue
            if "id" in msg and ("result" in msg or "error" in msg):
                self._responses.put(msg)
            # Server-initiated requests/notifications are ignored - this
            # client offers no capabilities for the server to call back on.

    def _request(self, method: str, params: dict, timeout: float) -> dict:
        with self._lock:
            self._next_id += 1
            req_id = self._next_id
            self._send({"jsonrpc": "2.0", "id": req_id, "method": method,
                        "params": params})
            # One request in flight at a time - wait for OUR id
            import time
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MCPError(self._with_tail(
                        f"MCP server '{self.name}': no response to {method} "
                        f"within {timeout:.0f}s"))
                try:
                    msg = self._responses.get(timeout=remaining)
                except queue.Empty:
                    continue
                if msg.get("id") == req_id:
                    return msg
                # Stale response from an earlier timed-out call - drop it

    def _notify(self, method: str, params: Optional[dict] = None) -> None:
        body: dict = {"jsonrpc": "2.0", "method": method}
        if params:
            body["params"] = params
        self._send(body)

    def _send(self, obj: dict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise MCPError(f"MCP server '{self.name}' is not running")
        proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        proc.stdin.flush()

    # ------------------------------------------------------------------ #
    #  Tool calls                                                          #
    # ------------------------------------------------------------------ #

    def call_tool(self, tool_name: str, arguments: dict) -> ToolResult:
        """Invoke one of this server's tools and convert the reply."""
        if not self.alive:
            return ToolResult.error(
                f"MCP server '{self.name}' has exited - tool unavailable")
        try:
            resp = self._request("tools/call",
                                 {"name": tool_name, "arguments": arguments},
                                 timeout=_CALL_TIMEOUT)
        except MCPError as e:
            return ToolResult.error(str(e))

        if "error" in resp:
            err = resp["error"]
            return ToolResult.error(
                f"MCP error {err.get('code')}: {err.get('message')}")

        result = resp.get("result", {})
        parts = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(f"[{item.get('type', 'unknown')} content omitted]")
        text = "\n".join(parts) or "(empty result)"

        if result.get("isError"):
            return ToolResult.error(text)
        return ToolResult.success(
            text, summary=f"mcp:{self.name}/{tool_name} ok")


# ---------------------------------------------------------------------------
#  Registry integration
# ---------------------------------------------------------------------------

_JSON_TO_PARAM_TYPE = {
    "integer": "int", "number": "float", "boolean": "bool", "array": "array",
}


def _schema_to_params(schema: dict) -> dict:
    """Convert an MCP inputSchema (JSON Schema) to our ToolDef params format."""
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = set(schema.get("required", []) or [])
    params = {}
    for pname, meta in props.items():
        # Param names and descriptions are server-controlled and land in the
        # system prompt, so neutralise() them. Membership in *required* is
        # checked on the original name.
        params[neutralise(str(pname))] = {
            "type": _JSON_TO_PARAM_TYPE.get(meta.get("type", "string"), "string"),
            "description": neutralise(str(meta.get("description", ""))),
            "required": pname in required,
        }
    return params


def _make_tool_fn(server: MCPServer, tool_name: str):
    """Wrap an MCP tool as a registry-compatible fn(cwd, **args). The function
    carries the server it calls (``_mcp_server``) so a later registration of
    the same spec can recognise its own binding."""
    def _fn(cwd: Path, **args) -> ToolResult:
        return server.call_tool(tool_name, args)
    _fn._mcp_server = server
    return _fn


def _spec_key(name: str, spec: dict) -> tuple:
    env = spec.get("env") or {}
    env_items: tuple = ()
    if isinstance(env, dict):
        env_items = tuple(sorted((str(k), str(v)) for k, v in env.items()))
    return (name, spec["command"], tuple(str(a) for a in spec.get("args", [])),
            env_items, bool(spec.get("trusted", False)))


def pooled_server(name: str, spec: dict) -> MCPServer:
    """The live server for *spec*, started on first use and reused after.
    Raises MCPError (or the start failure) when a fresh start fails; nothing
    is pooled then. The pool is process-wide and its entries stop at exit."""
    key = _spec_key(name, spec)
    with _POOL_LOCK:
        server = _POOL.get(key)
        if server is not None and server.alive:
            return server
        if server is not None:
            _POOL.pop(key, None)
        server = MCPServer(
            name=name,
            command=spec["command"],
            args=spec.get("args", []),
            env=spec.get("env"),
            trusted=bool(spec.get("trusted", False)),
        )
        try:
            server.start()
        except BaseException:
            server.stop()
            raise
        _POOL[key] = server
        import atexit
        atexit.register(server.stop)
        return server


def stop_pooled_servers() -> None:
    """Stop and forget every pooled server."""
    with _POOL_LOCK:
        servers = list(_POOL.values())
        _POOL.clear()
    for server in servers:
        try:
            server.stop()
        except Exception:
            pass


def mcp_servers_from(raw: dict) -> dict[str, dict]:
    """The [mcp.servers.*] tables of a parsed project config *raw* that have a
    string ``command``, by server name. Empty when there are none."""
    mcp_table = raw.get("mcp", {})
    servers = mcp_table.get("servers", {}) if isinstance(mcp_table, dict) else {}
    if not isinstance(servers, dict):
        return {}
    out = {}
    for name, spec in servers.items():
        if isinstance(spec, dict) and isinstance(spec.get("command"), str):
            out[name] = spec
    return out


def mcp_argv(spec: dict) -> list[str]:
    """The command line a server *spec* is started with, as strings."""
    return [str(spec.get("command", "")), *(str(a) for a in (spec.get("args") or []))]


def load_mcp_config(cwd: Path) -> dict[str, dict]:
    """Read the [mcp.servers.*] tables from the nearest project config.

    Returns an empty dict when there is no project config or it declares no
    servers. Raises ProjectConfigUnreadable when the config file exists but
    cannot be read or parsed. Says nothing about whether they are trusted."""
    from .project_config import find_project_config, read_project_config_file
    path = find_project_config(cwd)
    if path is None:
        return {}
    return mcp_servers_from(read_project_config_file(path))


def untrusted_servers_message(config_path: Path, names) -> str:
    """The notice that the MCP servers *names* in *config_path* were not
    started because they are not trusted."""
    listed = ", ".join(f"'{n}'" for n in names)
    return (f"MCP servers {listed} in {config_path} are not trusted, so none were "
            "started. Review them and trust them with /hooks trust or --trust-hooks "
            "(or the coder's hooks button in the GUI), or remove them from the file.")


def load_trusted_mcp_config(cwd: Path) -> tuple[dict[str, dict], list[str]]:
    """The MCP servers of the project config nearest *cwd* when the trust store
    trusts that config as it is now, else none and a warning naming them.

    The servers returned come from the same read the digest was taken of.
    Raises ProjectConfigUnreadable like load_mcp_config."""
    from .hook_trust import commands_digest, is_trusted
    from .project_config import find_project_config, read_project_config_file
    path = find_project_config(cwd)
    if path is None:
        return {}, []
    raw = read_project_config_file(path)
    servers = mcp_servers_from(raw)
    if not servers or is_trusted(path, commands_digest(raw)):
        return servers, []
    return {}, [untrusted_servers_message(path, servers)]


def register_mcp_tools(cwd: Path, servers: Optional[dict[str, dict]] = None,
                       ) -> tuple[list[str], list[str]]:
    """
    Start MCP servers and register their tools.

    *servers* are started as given; the caller has established that the user
    trusts them. Without *servers*, the servers of the project config nearest
    *cwd* are started only when the trust store trusts that config as it is now
    (load_trusted_mcp_config); otherwise none start and the warnings say so.

    Returns (registered_tool_names, warnings). A failing server produces a
    warning, never an exception - MCP problems must not break the agent.
    """
    from .project_config import ProjectConfigUnreadable

    registered: list[str] = []
    warnings: list[str] = []

    if servers is not None:
        configured = servers
    else:
        try:
            configured, warnings = load_trusted_mcp_config(cwd)
        except ProjectConfigUnreadable as e:
            return registered, [f"MCP servers not loaded: project config unreadable ({e})"]

    for name, spec in configured.items():
        try:
            server = pooled_server(name, spec)
        except MCPError as e:
            warnings.append(str(e))
            continue
        except Exception as e:
            warnings.append(f"MCP server '{name}' failed to start: {e}")
            continue

        for tool in server.tools:
            tool_name = tool.get("name", "")
            if not tool_name:
                continue
            # The server fully controls its tool names and descriptions, and
            # both end up in the system prompt. Defang any chat-template control
            # token / frame marker in them. The registered name uses the
            # neutralised form (a no-op for ordinary names); the ACTUAL call
            # keeps the original tool_name via the closure.
            reg_name = f"mcp_{name}_{neutralise(tool_name)}"
            register_foreign_tool(
                reg_name,
                fn=_make_tool_fn(server, tool_name),
                description=f"[MCP:{name}] {tool.get('description', '')}",
                params=_schema_to_params(tool.get("inputSchema", {})),
                # External code - confirm unless the config marks it trusted
                destructive=not server.trusted,
                source_label="MCP",
                registered=registered,
                warnings=warnings,
                reuse_if_already_ours=(
                    lambda td, s=server: getattr(td.fn, "_mcp_server", None) is s),
            )

    return registered, warnings
