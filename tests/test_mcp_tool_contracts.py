# SPDX-License-Identifier: AGPL-3.0-or-later
"""Semantic contracts for the MCP tool table that ``build_tools()`` returns.

Every tool an MCP client can see is pinned here as a structured contract: its
name, the argument names and JSON types of its input schema, its required
arguments, its MCP annotations, the feature gate that decides whether it is
advertised, and the reply its handler gives on the argument-validation path.
These are contracts about the PUBLIC surface (what ``tools/list`` and
``tools/call`` expose), never about how ``build_tools()`` is written, so they
must pass unchanged across an internal reorganisation of the server module.

A change to any of these is a change to what every MCP client sees and must
show up here first.
"""

from __future__ import annotations

import importlib
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

import localm.plugins.mcpserver.server as srv
import localm.plugins.mcpserver.tools as toolpkg
from localm.plugins.mcpserver.server import EngineCache, MCPStdioServer, build_tools

# A UNC path at a non-routable RFC5737 documentation address (TEST-NET-1), so
# a regression in the client-string guards still cannot dial a real host.
_UNC = r"\\192.0.2.1\share"

# --------------------------------------------------------------------- #
#  The contracts                                                        #
# --------------------------------------------------------------------- #

_MODEL = ("string", None)
_STR = ("string", None)
_INT = ("integer", None)
_NUM = ("number", None)
_BOOL = ("boolean", None)
_STR_ARRAY = ("array", "string")

_READ_ONLY = {"readOnlyHint": True}
_DESTRUCTIVE = {"destructiveHint": True}

# name -> (properties {arg: (json type, array item type or None)},
#          required (tuple, or None when the schema declares no required key),
#          annotations (dict of the hint keys, or None when none are declared))
CONTRACTS = {
    "chat": (
        {"prompt": _STR, "system": _STR, "images": _STR_ARRAY, "model": _MODEL,
         "max_tokens": _INT, "temperature": _NUM, "seed": _INT},
        ("prompt",), None),
    "embed": (
        {"texts": _STR_ARRAY, "model": _MODEL},
        ("texts",), None),
    "memory_recall": (
        {"query": _STR, "limit": _INT},
        ("query",), _READ_ONLY),
    "memory_append": (
        {"text": _STR},
        ("text",), {}),
    "list_models": ({}, None, _READ_ONLY),
    "system_stats": ({}, None, _READ_ONLY),
    "search_models": (
        {"query": _STR, "limit": _INT},
        None, _READ_ONLY),
    "list_model_files": (
        {"repo": _STR},
        ("repo",), _READ_ONLY),
    "pull_model": (
        {"repo": _STR, "file": _STR, "name": _STR, "load": _BOOL},
        ("repo", "name"), None),
    "setup_embeddings": (
        {"model": _STR},
        None, None),
    "remove_model": (
        {"model": _STR},
        ("model",), _DESTRUCTIVE),
    "generate_image": (
        {"prompt": _STR, "output_path": _STR, "negative_prompt": _STR, "seed": _INT,
         "guidance": _NUM, "input_image": _STR, "denoise": _NUM},
        ("prompt",), None),
    "run_coder_task": (
        {"task": _STR, "cwd": _STR, "model": _MODEL, "max_turns": _INT, "yes": _BOOL,
         "timeout_seconds": _INT},
        ("task", "cwd"), None),
    "server_activity": ({}, None, _READ_ONLY),
    "run_doctor": ({}, None, _READ_ONLY),
    "list_plugins": ({}, None, _READ_ONLY),
    "install_plugin": (
        {"plugin": _STR, "with_deps": _BOOL},
        ("plugin",), None),
    "enable_plugin": ({"plugin": _STR}, ("plugin",), None),
    "disable_plugin": ({"plugin": _STR}, ("plugin",), None),
    "uninstall_plugin": (
        {"plugin": _STR, "delete_data": _BOOL},
        ("plugin",), _DESTRUCTIVE),
}

ALL_TOOLS = frozenset(CONTRACTS)

# Tools whose advertisement depends on a flag or a probe. Everything else is
# always advertised.
GATED = frozenset({"embed", "run_coder_task", "generate_image",
                   "memory_recall", "memory_append"})
UNCONDITIONAL = ALL_TOOLS - GATED

# Every annotated tool carries a human-readable title next to its hint.
ANNOTATED = frozenset(name for name, (_, _, ann) in CONTRACTS.items()
                      if ann is not None)

# Behavioural promises a client reads out of the description and relies on.
DESCRIPTION_PROMISES = {
    "pull_model": ("blocks until ready",),
    "run_coder_task": ("denied", "success=False"),
    "memory_recall": ("Read-only",),
    "memory_append": ("UNVERIFIED", "never overwrites"),
    "server_activity": ("no server is running", "could not reach it"),
    "remove_model": ("delete the file",),
    "uninstall_plugin": ("Uninstall",),
    "setup_embeddings": ("embedding model",),
    "list_model_files": ("fits", "tight", "too-big"),
    "system_stats": ("VRAM",),
}


# --------------------------------------------------------------------- #
#  Harness                                                              #
# --------------------------------------------------------------------- #

def _stub_engine_factory(model_name):
    engine = MagicMock()
    engine.display_name = model_name
    engine.chat_stream.side_effect = lambda messages, **kw: iter(
        [f"reply-from-{model_name}"])
    engine.embed.return_value = [[0.1, 0.2]]
    engine.active_requests = 0
    engine.unloading = False
    return engine


def _engines(default_model="stub-model"):
    return EngineCache(default_model=default_model,
                       engine_factory=_stub_engine_factory)


def _force_gates(monkeypatch, *, embed=True, coder=True, memory=True):
    monkeypatch.setattr(srv, "_backend_can_embed", lambda *a, **k: embed)
    monkeypatch.setattr(srv, "_coder_available", lambda: coder)
    monkeypatch.setattr(srv, "_memory_available", lambda: memory)
    monkeypatch.setattr(srv, "_memory_embed_fn", lambda: None)


@pytest.fixture
def all_tools(monkeypatch):
    """Every tool build_tools() can ever return: all gates forced open."""
    _force_gates(monkeypatch)
    monkeypatch.setenv("LOCALM_MODE", "full")
    engines = _engines()
    tools = build_tools(engines, enable_images=True, enable_coder=True,
                        enable_memory=True, enable_memory_write=True)
    tools["_engines"] = engines
    return tools


def _call(tools, name, args):
    return tools[name]["handler"](args)


def _text(reply):
    return reply["content"][0]["text"]


# --------------------------------------------------------------------- #
#  Inventory                                                            #
# --------------------------------------------------------------------- #

class TestInventory:
    def test_all_gates_open_advertises_exactly_the_contracted_tools(self, all_tools):
        names = set(all_tools) - {"_engines"}
        assert names == set(ALL_TOOLS)

    def test_all_gates_closed_advertises_exactly_the_unconditional_tools(self, monkeypatch):
        _force_gates(monkeypatch, embed=False, coder=False, memory=False)
        tools = build_tools(_engines(), enable_images=False, enable_coder=False,
                            enable_memory=False, enable_memory_write=False)
        assert set(tools) == set(UNCONDITIONAL)

    def test_default_flags_match_the_documented_defaults(self, monkeypatch):
        """build_tools(engines) alone means images on, coder on, memory on,
        memory writes OFF."""
        _force_gates(monkeypatch)
        by_default = set(build_tools(_engines()))
        explicit = set(build_tools(_engines(), enable_images=True, enable_coder=True,
                                   enable_memory=True, enable_memory_write=False))
        assert by_default == explicit
        assert "generate_image" in by_default
        assert "run_coder_task" in by_default
        assert "memory_recall" in by_default
        assert "memory_append" not in by_default

    def test_every_spec_carries_exactly_the_dispatcher_keys(self, all_tools):
        for name in ALL_TOOLS:
            spec = all_tools[name]
            extra = set(spec) - {"description", "inputSchema", "annotations", "handler"}
            assert not extra, f"{name} spec has unexpected keys {extra}"
            assert {"description", "inputSchema", "handler"} <= set(spec), name
            assert callable(spec["handler"]), name


# --------------------------------------------------------------------- #
#  Schemas                                                              #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("name", sorted(ALL_TOOLS))
class TestSchemaContract:
    def test_input_schema_is_an_object_with_only_the_contracted_keys(self, all_tools, name):
        schema = all_tools[name]["inputSchema"]
        assert schema["type"] == "object", name
        assert set(schema) <= {"type", "properties", "required"}, name

    def test_argument_names_and_types(self, all_tools, name):
        props, _required, _ann = CONTRACTS[name]
        schema = all_tools[name]["inputSchema"]
        assert set(schema["properties"]) == set(props), name
        for arg, (jtype, item_type) in props.items():
            decl = schema["properties"][arg]
            assert decl["type"] == jtype, f"{name}.{arg}"
            if item_type is None:
                assert "items" not in decl, f"{name}.{arg}"
            else:
                assert decl["items"] == {"type": item_type}, f"{name}.{arg}"

    def test_every_argument_is_described(self, all_tools, name):
        for arg, decl in all_tools[name]["inputSchema"]["properties"].items():
            assert isinstance(decl.get("description"), str) and decl["description"].strip(), \
                f"{name}.{arg} has no description"

    def test_required_arguments(self, all_tools, name):
        _props, required, _ann = CONTRACTS[name]
        schema = all_tools[name]["inputSchema"]
        if required is None:
            assert "required" not in schema, name
        else:
            assert schema["required"] == list(required), name
            # A required argument must be a declared one.
            assert set(required) <= set(schema["properties"]), name

    def test_annotations(self, all_tools, name):
        _props, _required, ann = CONTRACTS[name]
        spec = all_tools[name]
        if ann is None:
            assert "annotations" not in spec, name
            return
        got = dict(spec["annotations"])
        title = got.pop("title", None)
        assert isinstance(title, str) and title.strip(), f"{name} has no title"
        assert got == ann, name

    def test_description_is_a_non_empty_string(self, all_tools, name):
        desc = all_tools[name]["description"]
        assert isinstance(desc, str) and desc.strip(), name
        for promise in DESCRIPTION_PROMISES.get(name, ()):
            assert promise in desc, f"{name} description lost the promise {promise!r}"


class TestAnnotationInvariants:
    def test_a_destructive_tool_is_never_also_read_only(self, all_tools):
        for name in ALL_TOOLS:
            ann = all_tools[name].get("annotations") or {}
            assert not (ann.get("destructiveHint") and ann.get("readOnlyHint")), name

    def test_the_two_file_deleting_tools_are_the_only_destructive_ones(self, all_tools):
        destructive = {name for name in ALL_TOOLS
                       if (all_tools[name].get("annotations") or {}).get("destructiveHint")}
        assert destructive == {"remove_model", "uninstall_plugin"}

    def test_tools_list_projects_the_same_annotations(self, all_tools):
        tools = {n: s for n, s in all_tools.items() if n != "_engines"}
        listed = MCPStdioServer(tools).handle(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
        by_name = {t["name"]: t for t in listed}
        assert set(by_name) == set(ALL_TOOLS)
        for name in ALL_TOOLS:
            entry = by_name[name]
            assert set(entry) <= {"name", "description", "inputSchema", "annotations"}, name
            assert "handler" not in entry, name
            assert entry["inputSchema"] == tools[name]["inputSchema"], name
            if name in ANNOTATED:
                assert entry["annotations"] == tools[name]["annotations"], name
            else:
                assert "annotations" not in entry, name


# --------------------------------------------------------------------- #
#  Feature gates                                                        #
# --------------------------------------------------------------------- #

class TestFeatureGates:
    @pytest.mark.parametrize("can_embed", [True, False])
    def test_embed_is_advertised_only_when_the_backend_can_embed(self, monkeypatch, can_embed):
        seen = []

        def probe(engines):
            seen.append(engines)
            return can_embed

        _force_gates(monkeypatch)
        monkeypatch.setattr(srv, "_backend_can_embed", probe)
        engines = _engines()
        tools = build_tools(engines)
        assert ("embed" in tools) is can_embed
        # The probe is asked once, about this server's own engine cache.
        assert seen == [engines]

    @pytest.mark.parametrize("flag,available,expected", [
        (True, True, True), (True, False, False),
        (False, True, False), (False, False, False)])
    def test_coder_tool_needs_the_flag_and_the_plugin(self, monkeypatch, flag, available, expected):
        _force_gates(monkeypatch, coder=available)
        tools = build_tools(_engines(), enable_coder=flag)
        assert ("run_coder_task" in tools) is expected

    def test_coder_probe_is_not_run_when_the_flag_is_off(self, monkeypatch):
        _force_gates(monkeypatch)

        def boom():
            raise AssertionError("probe must not run with --no-coder")

        monkeypatch.setattr(srv, "_coder_available", boom)
        assert "run_coder_task" not in build_tools(_engines(), enable_coder=False)

    @pytest.mark.parametrize("flag", [True, False])
    def test_generate_image_follows_the_images_flag(self, monkeypatch, flag):
        _force_gates(monkeypatch)
        assert ("generate_image" in build_tools(_engines(), enable_images=flag)) is flag

    @pytest.mark.parametrize("flag,available,expected", [
        (True, True, True), (True, False, False),
        (False, True, False), (False, False, False)])
    def test_memory_recall_needs_the_flag_and_the_plugin(self, monkeypatch, flag, available, expected):
        _force_gates(monkeypatch, memory=available)
        tools = build_tools(_engines(), enable_memory=flag)
        assert ("memory_recall" in tools) is expected

    def test_memory_probe_is_not_run_when_the_flag_is_off(self, monkeypatch):
        _force_gates(monkeypatch)

        def boom():
            raise AssertionError("probe must not run with --no-memory")

        monkeypatch.setattr(srv, "_memory_available", boom)
        tools = build_tools(_engines(), enable_memory=False, enable_memory_write=True)
        assert "memory_recall" not in tools and "memory_append" not in tools

    @pytest.mark.parametrize("memory,write,expected", [
        (True, True, True), (True, False, False),
        (False, True, False), (False, False, False)])
    def test_memory_append_needs_recall_plus_the_write_flag(self, monkeypatch, memory, write, expected):
        _force_gates(monkeypatch, memory=memory)
        tools = build_tools(_engines(), enable_memory=True, enable_memory_write=write)
        assert ("memory_append" in tools) is expected
        if expected:
            assert "memory_recall" in tools

    @pytest.mark.parametrize("images,coder,memory,write", [
        (True, True, True, True), (False, False, False, False),
        (True, False, True, False), (False, True, False, True)])
    def test_unconditional_tools_are_advertised_under_every_gate_state(
            self, monkeypatch, images, coder, memory, write):
        _force_gates(monkeypatch, embed=coder, coder=coder, memory=memory)
        tools = build_tools(_engines(), enable_images=images, enable_coder=coder,
                            enable_memory=memory, enable_memory_write=write)
        assert set(UNCONDITIONAL) <= set(tools)
        assert set(tools) <= set(ALL_TOOLS)


# --------------------------------------------------------------------- #
#  Handler contracts: argument validation                               #
# --------------------------------------------------------------------- #

def _complete_args(name, tmp_dir):
    """A well-formed argument set for every tool with required arguments,
    used to knock one required argument out at a time."""
    return {
        "chat": {"prompt": "hi"},
        "embed": {"texts": ["a"]},
        "memory_recall": {"query": "q"},
        "memory_append": {"text": "t"},
        "list_model_files": {"repo": "owner/name"},
        "pull_model": {"repo": "owner/name", "name": "m"},
        "remove_model": {"model": "m"},
        "generate_image": {"prompt": "p"},
        "run_coder_task": {"task": "t", "cwd": tmp_dir},
        "install_plugin": {"plugin": "p"},
        "enable_plugin": {"plugin": "p"},
        "disable_plugin": {"plugin": "p"},
        "uninstall_plugin": {"plugin": "p"},
    }[name]


_REQUIRED_CASES = [(name, arg) for name, (_p, req, _a) in sorted(CONTRACTS.items())
                   if req for arg in req]


class TestRequiredArgumentsAreEnforced:
    @pytest.mark.parametrize("name,arg", _REQUIRED_CASES)
    def test_a_missing_required_argument_is_a_tool_error_naming_it(
            self, all_tools, tmp_path, name, arg):
        args = dict(_complete_args(name, str(tmp_path)))
        del args[arg]
        reply = _call(all_tools, name, args)
        assert reply["isError"] is True, (name, arg, reply)
        assert f"'{arg}'" in _text(reply), (name, arg, reply)

    @pytest.mark.parametrize("name,arg", _REQUIRED_CASES)
    def test_an_empty_required_argument_is_refused_like_a_missing_one(
            self, all_tools, tmp_path, name, arg):
        args = dict(_complete_args(name, str(tmp_path)))
        args[arg] = "" if not isinstance(args[arg], list) else []
        reply = _call(all_tools, name, args)
        assert reply["isError"] is True, (name, arg, reply)
        assert f"'{arg}'" in _text(reply), (name, arg, reply)

    def test_a_tool_error_is_a_text_content_reply(self, all_tools):
        reply = _call(all_tools, "chat", {})
        assert set(reply) == {"content", "isError"}
        assert reply["content"] == [{"type": "text", "text": _text(reply)}]


# --------------------------------------------------------------------- #
#  Handler contracts: selected behaviour that clients rely on           #
# --------------------------------------------------------------------- #

class TestChatAndEmbed:
    def test_chat_reaches_the_configured_engine_and_returns_its_text(self, all_tools):
        reply = _call(all_tools, "chat", {"prompt": "hi"})
        assert reply["isError"] is False
        assert _text(reply) == "reply-from-stub-model"

    def test_chat_model_argument_selects_the_engine(self, all_tools):
        reply = _call(all_tools, "chat", {"prompt": "hi", "model": "other-model"})
        assert _text(reply) == "reply-from-other-model"

    def test_chat_forwards_system_prompt_and_generation_knobs(self, all_tools):
        _call(all_tools, "chat", {"prompt": "hi", "system": "be brief",
                                  "max_tokens": 7, "temperature": 0.2, "seed": 3})
        engine = all_tools["_engines"]._engines["stub-model"]
        messages, kwargs = engine.chat_stream.call_args.args[0], engine.chat_stream.call_args.kwargs
        assert messages == [{"role": "system", "content": "be brief"},
                            {"role": "user", "content": "hi"}]
        assert kwargs == {"max_tokens": 7, "temperature": 0.2, "seed": 3}

    def test_chat_omits_unset_generation_knobs(self, all_tools):
        _call(all_tools, "chat", {"prompt": "hi"})
        engine = all_tools["_engines"]._engines["stub-model"]
        assert engine.chat_stream.call_args.kwargs == {}
        assert engine.chat_stream.call_args.args[0] == [{"role": "user", "content": "hi"}]

    def test_embed_returns_the_vectors_as_json(self, all_tools):
        reply = _call(all_tools, "embed", {"texts": ["a", "b"]})
        assert reply["isError"] is False
        assert json.loads(_text(reply)) == [[0.1, 0.2]]

    def test_embed_accepts_a_single_string(self, all_tools):
        _call(all_tools, "embed", {"texts": "one"})
        engine = all_tools["_engines"]._engines["stub-model"]
        engine.embed.assert_called_once_with(["one"])

    def test_embed_reports_a_backend_that_cannot_embed_as_a_tool_error(self, all_tools):
        engine = all_tools["_engines"].get("stub-model")
        engine.embed.side_effect = NotImplementedError("this backend cannot embed")
        reply = _call(all_tools, "embed", {"texts": ["a"]})
        assert reply["isError"] is True
        assert "cannot embed" in _text(reply)


class TestClientSuppliedStringsAreGuarded:
    """Strings an MCP client supplies are never handed to a filesystem probe
    when they carry UNC/device syntax, and are refused with a message that
    does not echo them back."""

    def test_pull_model_refuses_a_unc_repo(self, all_tools):
        with patch("localm.model_manager.pull.pull_model") as pull:
            reply = _call(all_tools, "pull_model", {"repo": _UNC, "name": "m"})
        assert reply["isError"] is True
        assert "UNC or device" in _text(reply)
        assert _UNC not in _text(reply)
        pull.assert_not_called()

    def test_pull_model_refuses_a_repo_that_is_neither_known_nor_registered(self, all_tools):
        with patch("localm.config.load_registry", return_value={}), \
             patch("localm.model_manager.pull.pull_model") as pull:
            reply = _call(all_tools, "pull_model", {"repo": "nobody/unknown", "name": "m"})
        assert reply["isError"] is True
        assert "Refusing to pull" in _text(reply)
        assert "localm pull" in _text(reply)
        pull.assert_not_called()

    def test_pull_model_accepts_a_repo_matching_a_registered_source(self, all_tools):
        reg = {"m": {"path": "x", "source": "hf:nobody/registered"}}
        with patch("localm.config.load_registry", return_value=reg), \
             patch("localm.model_manager.pull.pull_model", return_value=True) as pull:
            reply = _call(all_tools, "pull_model",
                          {"repo": "nobody/registered", "name": "m", "load": False})
        assert reply["isError"] is False
        pull.assert_called_once_with("nobody/registered", name="m")
        assert "not loaded" in _text(reply)

    def test_run_coder_task_refuses_a_unc_cwd(self, all_tools):
        reply = _call(all_tools, "run_coder_task", {"task": "x", "cwd": _UNC})
        assert reply["isError"] is True
        assert "UNC or device" in _text(reply)
        assert _UNC not in _text(reply)

    def test_run_coder_task_refuses_a_cwd_that_is_not_a_directory(self, all_tools, tmp_path):
        reply = _call(all_tools, "run_coder_task",
                      {"task": "x", "cwd": str(tmp_path / "nope")})
        assert reply["isError"] is True
        assert "not a directory" in _text(reply)

    def test_run_coder_task_refuses_a_timeout_above_the_cap(self, all_tools, tmp_path):
        reply = _call(all_tools, "run_coder_task",
                      {"task": "x", "cwd": str(tmp_path),
                       "timeout_seconds": srv.MAX_CODER_TIMEOUT_SECONDS + 1})
        assert reply["isError"] is True
        assert "at most" in _text(reply)
        assert all_tools["_engines"]._engines == {}, "nothing may be loaded first"

    def test_the_coder_timeout_cap_is_the_servers_current_value(
            self, all_tools, monkeypatch, tmp_path):
        """The cap is read from the server module on every call."""
        monkeypatch.setattr(srv, "MAX_CODER_TIMEOUT_SECONDS", 1.0)
        reply = _call(all_tools, "run_coder_task",
                      {"task": "x", "cwd": str(tmp_path), "timeout_seconds": 2})
        assert reply["isError"] is True
        assert "at most 1" in _text(reply)
        assert all_tools["_engines"]._engines == {}

    @pytest.mark.parametrize("bad", [-1, True, "soon"])
    def test_run_coder_task_refuses_a_non_positive_timeout(self, all_tools, tmp_path, bad):
        reply = _call(all_tools, "run_coder_task",
                      {"task": "x", "cwd": str(tmp_path), "timeout_seconds": bad})
        assert reply["isError"] is True
        assert "positive number" in _text(reply)

    @pytest.mark.parametrize("arg_key", ["output_path", "input_image"])
    def test_generate_image_keeps_paths_out_of_the_filesystem_at_large(
            self, all_tools, tmp_path, arg_key):
        outside = str(tmp_path.parent / "outside" / "x.png")
        with patch("localm.image_gen.comfy.generate_image") as gen:
            reply = _call(all_tools, "generate_image", {"prompt": "p", arg_key: outside})
        assert reply["isError"] is True
        gen.assert_not_called()

    def test_generate_image_refuses_a_unc_output_path(self, all_tools):
        with patch("localm.image_gen.comfy.generate_image") as gen:
            reply = _call(all_tools, "generate_image", {"prompt": "p", "output_path": _UNC})
        assert reply["isError"] is True
        assert "output_path" in _text(reply)
        assert _UNC not in _text(reply)
        gen.assert_not_called()

    def test_setup_embeddings_refuses_a_path_and_writes_nothing(self, all_tools):
        import localm.config as cfg
        before = cfg.load_config().get("embedding_model")
        with patch("localm.config.load_registry", return_value={}), \
             patch("localm.inference.embedder.resolve_embedding_model_path") as resolve:
            reply = _call(all_tools, "setup_embeddings", {"model": "C:/somewhere/x.gguf"})
        assert reply["isError"] is True
        assert "Refusing" in _text(reply)
        resolve.assert_not_called()
        assert cfg.load_config().get("embedding_model") == before


class TestMemoryGatesAtCallTime:
    """Advertising a memory tool is UX; the handler re-checks every gate on
    each call, so a stale client tool list cannot reach the store."""

    @pytest.mark.parametrize("name,args", [
        ("memory_recall", {"query": "x"}), ("memory_append", {"text": "y"})])
    def test_plugin_inactive_at_call_time_is_refused(self, all_tools, monkeypatch, name, args):
        monkeypatch.setattr(srv, "_memory_available", lambda: False)
        reply = _call(all_tools, name, args)
        assert reply["isError"] is True
        assert "memory plugin is not active" in _text(reply)
        assert "localm plugin install memory" in _text(reply)

    @pytest.mark.parametrize("name,args", [
        ("memory_recall", {"query": "x"}), ("memory_append", {"text": "y"})])
    def test_privacy_mode_at_call_time_is_refused(self, all_tools, monkeypatch, name, args):
        monkeypatch.setenv("LOCALM_MODE", "privacy")
        reply = _call(all_tools, name, args)
        assert reply["isError"] is True
        assert "privacy mode" in _text(reply)

    @pytest.mark.parametrize("name,args", [
        ("memory_recall", {"query": "x"}), ("memory_append", {"text": "y"})])
    def test_the_privacy_gate_is_the_chat_surface_not_the_global_mode(
            self, all_tools, monkeypatch, name, args):
        """The memory tools read and write the chat namespace, so the mode that
        gates them is chat_mode: a privacy chat_mode refuses even under a
        permissive global mode, and a permissive chat_mode allows even under a
        privacy global mode."""
        from localm.config import update_config
        monkeypatch.delenv("LOCALM_MODE", raising=False)
        update_config(lambda c: c.update({"mode": "full", "chat_mode": "privacy"}))
        reply = _call(all_tools, name, args)
        assert reply["isError"] is True, reply
        assert "privacy mode" in _text(reply)

        update_config(lambda c: c.update({"mode": "privacy", "chat_mode": "full"}))
        reply = _call(all_tools, name, args)
        assert reply["isError"] is False, reply
        assert "privacy mode" not in _text(reply)

    def test_recall_of_an_empty_store_says_so_without_an_error(self, all_tools):
        reply = _call(all_tools, "memory_recall", {"query": "anything"})
        assert reply["isError"] is False
        assert "No remembered facts" in _text(reply)
        assert "0 fact(s)" in _text(reply)

    def test_append_stores_an_unverified_record_and_says_so(self, all_tools):
        reply = _call(all_tools, "memory_append", {"text": "the user prefers tabs"})
        assert reply["isError"] is False
        assert "unverified" in _text(reply)
        assert "the user prefers tabs" in _text(reply)
        from localm import memory as _mem
        from localm.config import home_dir
        store = _mem.open_store(None, "chat", "", root=home_dir() / "memory")
        recs = store.all()
        assert [r.text for r in recs] == ["the user prefers tabs"]
        assert recs[0].source == "synth"


class TestInventoryAndDiagnosticTools:
    def test_list_models_reports_an_empty_registry(self, all_tools):
        with patch("localm.config.load_registry", return_value={}):
            reply = _call(all_tools, "list_models", {})
        assert reply["isError"] is False
        assert _text(reply) == "No models registered."

    def test_list_models_marks_a_malformed_entry_instead_of_failing(self, all_tools, tmp_path):
        good = tmp_path / "good.gguf"
        good.write_bytes(b"x" * 10)
        reg = {"good": {"path": str(good), "source": "local"}, "bad": {"path": None}}
        with patch("localm.config.load_registry", return_value=reg):
            reply = _call(all_tools, "list_models", {})
        assert reply["isError"] is False
        lines = _text(reply).splitlines()
        assert any(ln.startswith("bad") and "corrupt" in ln for ln in lines)
        assert any(ln.startswith("good") and "MB" in ln and "local" in ln for ln in lines)

    def test_remove_model_refuses_an_unregistered_name(self, all_tools):
        with patch("localm.config.load_registry", return_value={}), \
             patch("localm.model_manager.remove_model") as rm:
            reply = _call(all_tools, "remove_model", {"model": "ghost"})
        assert reply["isError"] is True
        assert "Model not found" in _text(reply)
        rm.assert_not_called()

    def test_system_stats_returns_a_live_reading_with_vram_waited_for(self, all_tools):
        stats = {"cpu_percent": 1.0, "vram": {"free": 5}}
        with patch("localm.sysstats.system_stats", return_value=stats) as fn:
            reply = _call(all_tools, "system_stats", {})
        assert json.loads(_text(reply)) == stats
        assert fn.call_args.kwargs.get("wait_first_vram") is True

    def test_search_models_returns_hf_results_as_json(self, all_tools):
        results = [{"id": "owner/repo"}]
        with patch("localm.discover.hf_search", return_value=results) as fn:
            reply = _call(all_tools, "search_models", {"query": "qwen", "limit": 3})
        assert json.loads(_text(reply)) == results
        fn.assert_called_once_with("qwen", limit=3)

    def test_search_models_surfaces_a_discover_error(self, all_tools):
        from localm.discover import DiscoverError
        with patch("localm.discover.hf_search", side_effect=DiscoverError("offline")):
            reply = _call(all_tools, "search_models", {})
        assert reply["isError"] is True
        assert "offline" in _text(reply)

    def test_list_model_files_adds_a_fit_label_per_file(self, all_tools):
        files = [{"filename": "a.gguf", "size_bytes": 1}]
        with patch("localm.discover.hf_gguf_files", return_value=files), \
             patch("localm.discover.vram_capacity", return_value={"total": 8}), \
             patch("localm.discover.fit_label", return_value="fits"):
            reply = _call(all_tools, "list_model_files", {"repo": "owner/repo"})
        assert json.loads(_text(reply))[0]["fit"] == "fits"

    def test_run_doctor_runs_this_installs_doctor_with_a_pinned_home(self, all_tools):
        proc = MagicMock(stdout="all good", stderr="")
        with patch("subprocess.run", return_value=proc) as run:
            reply = _call(all_tools, "run_doctor", {})
        assert reply["isError"] is False
        assert _text(reply) == "all good"
        assert run.call_args.args[0] == [sys.executable, "-m", "localm", "doctor"]
        env = run.call_args.kwargs["env"]
        from localm.config import home_dir
        assert env["LOCALM_HOME"] == str(home_dir())
        assert env["PYTHONSAFEPATH"] == "1"
        assert run.call_args.kwargs["timeout"] == 60

    def test_server_activity_with_no_server_is_not_reported_as_idle(self, all_tools, monkeypatch):
        from localm import instances
        monkeypatch.setattr(instances, "snapshot", lambda *a, **kw: [])
        reply = _call(all_tools, "server_activity", {})
        assert reply["isError"] is False
        assert "No localm server" in _text(reply)
        assert "idle, nothing running" not in _text(reply)


class TestPluginAdministration:
    @pytest.mark.parametrize("name", ["enable_plugin", "disable_plugin", "uninstall_plugin"])
    def test_an_unknown_plugin_is_a_no_such_plugin_error(self, all_tools, name):
        reply = _call(all_tools, name, {"plugin": "no-such-plugin-xyz"})
        assert reply["isError"] is True
        assert "No such plugin: no-such-plugin-xyz" in _text(reply)

    def test_list_plugins_reports_each_state(self, all_tools):
        state = {"plugins": [
            {"name": "a", "active": True, "installed": True, "description": "A"},
            {"name": "b", "active": False, "installed": True},
            {"name": "c", "active": False, "installed": False},
        ]}
        with patch("localm.plugins.engine.PluginManager.api_state", return_value=state):
            reply = _call(all_tools, "list_plugins", {})
        lines = _text(reply).splitlines()
        assert lines[0] == "a  [enabled] - A"
        assert lines[1] == "b  [disabled]"
        assert lines[2] == "c  [available]"

    def test_install_plugin_installs_then_resolves_deps_by_default(self, all_tools):
        with patch("localm.plugins.engine.PluginManager.set_installed_state") as install, \
             patch("localm.plugins.engine.PluginManager.plugin_missing_deps",
                   return_value=False) as missing:
            reply = _call(all_tools, "install_plugin", {"plugin": "p"})
        assert reply["isError"] is False
        install.assert_called_once_with("p", True)
        missing.assert_called_once_with("p")
        assert "installed and enabled" in _text(reply)

    def test_uninstall_plugin_forwards_delete_data(self, all_tools):
        with patch("localm.plugins.engine.PluginManager.uninstall", return_value=True) as un:
            reply = _call(all_tools, "uninstall_plugin", {"plugin": "p", "delete_data": True})
        assert reply["isError"] is False
        un.assert_called_once_with("p", delete_data=True)


# --------------------------------------------------------------------- #
#  Composition                                                          #
# --------------------------------------------------------------------- #

# Family module -> the tools it owns. Together the families partition the
# whole table; build_tools() only merges them and applies the gates.
FAMILIES = {
    "chat": {"chat", "embed"},
    "memory": {"memory_recall", "memory_append"},
    "models": {"list_models", "search_models", "list_model_files", "pull_model",
               "setup_embeddings", "remove_model"},
    "media_coder": {"generate_image", "run_coder_task"},
    "diagnostics": {"server_activity", "system_stats", "run_doctor"},
    "plugin_admin": {"list_plugins", "install_plugin", "enable_plugin", "disable_plugin",
                     "uninstall_plugin"},
}


def _build_family(name, engines):
    mod = importlib.import_module(f"localm.plugins.mcpserver.tools.{name}")
    if name == "memory":
        return mod.build(enable_memory_write=True)
    if name in ("diagnostics", "plugin_admin"):
        return mod.build()
    return mod.build(engines)


class TestComposition:
    def test_the_families_partition_the_tool_table(self):
        engines = _engines()
        seen = set()
        for family, expected in FAMILIES.items():
            built = _build_family(family, engines)
            assert set(built) == expected, family
            assert not (seen & set(built)), f"{family} re-defines {seen & set(built)}"
            seen |= set(built)
        assert seen == set(ALL_TOOLS)

    def test_a_family_builds_every_tool_it_owns_regardless_of_the_gates(self):
        """Gating is the server's job: a family never hides its own tools."""
        assert set(_build_family("chat", _engines())) == {"chat", "embed"}
        mod = importlib.import_module("localm.plugins.mcpserver.tools.memory")
        assert set(mod.build(enable_memory_write=False)) == {"memory_recall", "memory_append"}

    def test_a_name_defined_by_two_families_is_a_build_time_error(self):
        with pytest.raises(toolpkg.ToolNameCollision) as excinfo:
            toolpkg.merge_tool_groups([("first", {"dup": {}}), ("second", {"dup": {}})])
        message = str(excinfo.value)
        assert "'dup'" in message and "'first'" in message and "'second'" in message

    def test_merge_keeps_family_order_and_the_original_spec_objects(self):
        first = {"x": {"description": "x"}, "y": {"description": "y"}}
        second = {"z": {"description": "z"}}
        merged = toolpkg.merge_tool_groups([("a", first), ("b", second)])
        assert list(merged) == ["x", "y", "z"]
        assert merged["x"] is first["x"] and merged["z"] is second["z"]

    def test_build_tools_refuses_a_colliding_family_instead_of_overwriting(self, monkeypatch):
        from localm.plugins.mcpserver.tools import plugin_admin
        real_build = plugin_admin.build

        def colliding():
            tools = real_build()
            tools["chat"] = {"description": "impostor", "inputSchema": {"type": "object",
                             "properties": {}}, "handler": lambda args: None}
            return tools

        monkeypatch.setattr(plugin_admin, "build", colliding)
        _force_gates(monkeypatch)
        with pytest.raises(toolpkg.ToolNameCollision, match="'chat'"):
            build_tools(_engines())

    def test_build_tools_advertises_the_families_in_order(self, all_tools):
        names = [n for n in all_tools if n != "_engines"]
        expected = []
        for family in FAMILIES:
            expected.extend(n for n in names if n in FAMILIES[family])
        assert names == expected
