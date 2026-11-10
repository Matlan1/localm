# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every raw outbound network call in ``localm/`` is classified, by enumeration.

Walks the AST of every module under ``localm/`` and collects each call to
``requests.<verb>``/``requests.Session``, ``httpx.*``, ``urllib.request.urlopen``,
``build_opener`` and ``socket.create_connection`` (import aliases resolved per
module). Each (file, enclosing function) site must appear in ``_CLASSIFIED`` with
one class word; a new site fails until it is classified, and an entry whose site
no longer exists fails too.

Class words:
  self          this machine's own server (loopback, via ``selfclient``)
  peer          another localm instance on this machine (loopback, redirects off)
  operator-cli  a command the operator runs against a server they name
  comfy         the ComfyUI / managed media backends the user configured
  fixed-host    a socket probe or opener to a fixed local address, or a
                transport primitive whose callers gate with ``netpolicy``
"""

import ast
import pathlib

import pytest

import localm

_ROOT = pathlib.Path(localm.__file__).resolve().parent
_REQUESTS_VERBS = {"get", "post", "put", "delete", "head", "patch", "request",
                   "options", "Session"}
_CLASSES = {"self", "peer", "operator-cli", "comfy", "fixed-host"}

_CLASSIFIED = {
    ("cli/_core.py", "server_call"): "operator-cli",
    ("cli/models.py", "_rename_on_running_server"): "operator-cli",
    ("cli/models.py", "unload_cmd"): "operator-cli",
    ("cli/models.py", "stop_cmd"): "operator-cli",
    ("cli/plugins.py", "_runtime_fields"): "operator-cli",
    ("cli/plugins.py", "_runtime_plugin_config"): "operator-cli",
    ("cli/rag.py", "_cli_rag_embed_fn._embed"): "operator-cli",
    ("gpu_registry.py", "_get_status"): "peer",
    ("gpu_registry.py", "_post_json"): "peer",
    ("http_ssl.py", "_open"): "fixed-host",
    ("inference/_hang_alarm.py", "_probe_once"): "self",
    ("inference/http_engine.py", "HttpEngine.context_capacity"): "operator-cli",
    ("inference/http_engine.py", "HttpEngine.chat_stream"): "operator-cli",
    ("inference/http_engine.py", "remote_model_status"): "operator-cli",
    ("instances.py", "fetch_any_whoami"): "peer",
    ("media/comfy_client.py", "comfy_port_holder"): "comfy",
    ("media/koboldcpp/server.py", "_Server.__init__"): "fixed-host",
    ("netpin.py", "pinned_session"): "fixed-host",
    ("plugins/coder/backends/http.py", "_post_with_retry"): "operator-cli",
    ("plugins/coder/backends/http.py", "HTTPBackend._get"): "operator-cli",
    ("plugins/coder/backends/http.py", "HTTPBackend.probe"): "operator-cli",
    ("plugins/coder/backends/http.py", "HTTPBackend.load_model"): "operator-cli",
    ("plugins/coder/server.py", "_port_open"): "fixed-host",
    ("plugins/gui/cli.py", "_mount_remote_gui"): "operator-cli",
    ("plugins/gui/cli.py", "_open_when_ready"): "fixed-host",
    ("plugins/gui/cli.py", "_mark_ready_when_listening"): "fixed-host",
    ("plugins/gui/cli.py", "_serve_beside_native_window"): "fixed-host",
    ("selfclient.py", "read_activity"): "self",
    ("selfclient.py", "read_model_file_hold"): "self",
    ("selfclient.py", "local_request"): "self",
}


def _aliases(tree):
    """Local names bound to the ``requests``, ``httpx``, ``socket`` and
    ``urllib.request`` modules, and to the call names imported directly."""
    mods, funcs = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in ("requests", "httpx", "socket", "urllib.request"):
                    mods[a.asname or a.name.split(".")[0]] = a.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                local = a.asname or a.name
                if node.module == "urllib" and a.name == "request":
                    mods[local] = "urllib.request"
                elif node.module == "urllib.request" and a.name in ("urlopen", "build_opener"):
                    funcs[local] = a.name
                elif node.module == "socket" and a.name == "create_connection":
                    funcs[local] = a.name
                elif node.module in ("requests", "httpx"):
                    funcs[local] = f"{node.module}.{a.name}"
    return mods, funcs


def _kind(call, mods, funcs):
    f = call.func
    if isinstance(f, ast.Name) and f.id in funcs:
        return funcs[f.id]
    if not isinstance(f, ast.Attribute):
        return None
    base = f.value
    if isinstance(base, ast.Name) and base.id in mods:
        mod = mods[base.id]
        if mod == "requests" and f.attr in _REQUESTS_VERBS:
            return f"requests.{f.attr}"
        if mod == "httpx":
            return f"httpx.{f.attr}"
        if mod == "socket" and f.attr == "create_connection":
            return "socket.create_connection"
        if mod == "urllib.request" and f.attr in ("urlopen", "build_opener"):
            return f"urllib.request.{f.attr}"
    if (isinstance(base, ast.Attribute) and base.attr == "request"
            and isinstance(base.value, ast.Name)
            and base.value.id in mods and mods[base.value.id] == "urllib.request"
            and f.attr in ("urlopen", "build_opener")):
        return f"urllib.request.{f.attr}"
    if (isinstance(base, ast.Attribute) and base.attr == "request"
            and isinstance(base.value, ast.Name) and base.value.id == "urllib"
            and f.attr in ("urlopen", "build_opener")):
        return f"urllib.request.{f.attr}"
    return None


class _Walker(ast.NodeVisitor):
    def __init__(self, mods, funcs):
        self.mods, self.funcs, self.stack, self.sites = mods, funcs, [], set()

    def _scope(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scope

    def visit_Call(self, node):
        if _kind(node, self.mods, self.funcs):
            self.sites.add(".".join(self.stack) or "<module>")
        self.generic_visit(node)


def _sites_in(source: str):
    tree = ast.parse(source)
    walker = _Walker(*_aliases(tree))
    walker.visit(tree)
    return walker.sites


def _all_sites():
    found = set()
    for path in sorted(_ROOT.rglob("*.py")):
        rel = path.relative_to(_ROOT).as_posix()
        for fn in _sites_in(path.read_text(encoding="utf-8")):
            found.add((rel, fn))
    return found


def test_every_raw_outbound_call_site_is_classified():
    unclassified = sorted(_all_sites() - set(_CLASSIFIED))
    assert not unclassified, (
        "raw outbound call sites with no class word in _CLASSIFIED (route the "
        "request through netpolicy, selfclient.local_request or verified_urlopen, "
        f"or classify it here): {unclassified}")


def test_no_classification_outlives_its_call_site():
    stale = sorted(set(_CLASSIFIED) - _all_sites())
    assert not stale, f"classified sites that no longer exist: {stale}"


def test_every_class_word_is_a_known_one():
    assert set(_CLASSIFIED.values()) <= _CLASSES


@pytest.mark.parametrize("source, expected", [
    ("import requests\ndef f():\n    requests.get('x')\n", {"f"}),
    ("import requests as r\ndef f():\n    r.post('x')\n", {"f"}),
    ("from requests import get\ndef f():\n    get('x')\n", {"f"}),
    ("import urllib.request\ndef f():\n    urllib.request.urlopen('x')\n", {"f"}),
    ("from urllib.request import urlopen\nclass C:\n    def m(self):\n        urlopen('x')\n", {"C.m"}),
    ("import urllib.request as u\ndef f():\n    u.build_opener()\n", {"f"}),
    ("import httpx\nasync def f():\n    httpx.AsyncClient()\n", {"f"}),
    ("import socket\ndef f():\n    socket.create_connection(('h', 1))\n", {"f"}),
    ("import requests\nrequests.get('x')\n", {"<module>"}),
    ("import json\ndef f():\n    json.get('x')\n", set()),
])
def test_the_scanner_finds_each_call_shape(source, expected):
    assert _sites_in(source) == expected
