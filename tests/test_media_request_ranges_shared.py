# SPDX-License-Identifier: AGPL-3.0-or-later
"""A denoise outside 0 to 1, a music track outside 1 to 3600 seconds and a video
clip outside 1 to 20 seconds are refused by the one generation path every
caller shares (CLI, chat REPL, coder tool, MCP tool), before a backend loads."""

from __future__ import annotations

import io
import json
import math
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from localm.plugins import media_config


class _Backend:
    """A media plugin's backend facade that counts loads and generations."""

    def __init__(self):
        self.loads = 0
        self.generated = []

    def settings(self, cfg):
        return {"backend": "fake"}

    def prepare_for_job(self, s, cfg):
        return s

    def _impl(self, s):
        return SimpleNamespace()

    def supports(self, s, capability):
        return False

    def ensure_available(self, s, on_progress=None):
        self.loads += 1
        return True, "ready"

    def generate(self, s, prompt, path, **kwargs):
        self.generated.append(kwargs)
        return True, "generated"

    def free_vram(self, s):
        return True


OUT_OF_RANGE = [
    ("denoise", 1.5), ("denoise", -0.1), ("denoise", math.nan), ("denoise", math.inf),
    ("denoise", True), ("denoise", "much"), ("denoise", [0.5]),
    ("duration_seconds", 0.5), ("duration_seconds", 0), ("duration_seconds", 3600.5),
    ("seconds", 0.5), ("seconds", 20.01), ("seconds", -1),
]

IN_RANGE = [
    ("denoise", 0), ("denoise", 0.6), ("denoise", 1), ("denoise", "0.6"),
    ("duration_seconds", 1), ("duration_seconds", 3600),
    ("seconds", 1.0), ("seconds", 20),
]


@pytest.mark.parametrize("name,value", OUT_OF_RANGE)
def test_the_shared_path_refuses_out_of_range_before_any_load(tmp_path, name, value):
    backend = _Backend()
    ok, message, _path = media_config.generate_media(
        backend, {"backend": "fake"}, "p", tmp_path / "o.png", **{name: value})
    assert (ok, message) == (False, media_config.REQUEST_RANGES[name][2])
    assert backend.loads == 0 and backend.generated == []


@pytest.mark.parametrize("name,value", IN_RANGE)
def test_the_shared_path_generates_in_range(tmp_path, name, value):
    backend = _Backend()
    ok, message, _path = media_config.generate_media(
        backend, {"backend": "fake"}, "p", tmp_path / "o.png", **{name: value})
    assert ok, message
    assert backend.loads == 1 and backend.generated[0][name] == value


def test_arguments_without_a_range_are_ignored():
    assert media_config.range_refusal(seed=-5, steps=10_000, fps=0) is None


def _facade(backend):
    return SimpleNamespace(generate_media=lambda prompt, out, **kw: media_config.generate_media(
        backend, {"backend": "fake"}, prompt, out, **kw))


def test_the_coder_tool_refuses_a_denoise_above_1(tmp_path, monkeypatch):
    from localm.plugins import lookup
    from localm.plugins.coder.tools.media import tool_generate_image
    backend = _Backend()
    monkeypatch.setattr(lookup, "plugin_api", lambda plugin, sub: _facade(backend))
    result = tool_generate_image(tmp_path, "a fox", output_path="fox.png", denoise=1.5)
    assert not result.ok
    assert result.output == "Denoise must be between 0 and 1"
    assert backend.loads == 0


def test_the_mcp_tool_refuses_a_denoise_above_1(monkeypatch):
    import localm.plugins.mcpserver.server as srv
    from localm.plugins import lookup
    backend = _Backend()
    monkeypatch.setattr(lookup, "plugin_api", lambda plugin, sub: _facade(backend))
    engines = srv.EngineCache(default_model="stub", engine_factory=lambda name: None)
    tools = srv.build_tools(engines, enable_images=True, enable_coder=False,
                            enable_memory=False)
    server = srv.MCPStdioServer({"generate_image": tools["generate_image"]})
    call = json.dumps({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                       "params": {"name": "generate_image",
                                  "arguments": {"prompt": "a fox", "denoise": 1.5}}}) + "\n"
    out = io.StringIO()
    server.run_stdio(stdin=io.StringIO(call), stdout=out)
    reply = [json.loads(line) for line in out.getvalue().splitlines()][-1]
    assert reply["id"] == 4 and reply["result"]["isError"] is True, reply
    assert "Denoise must be between 0 and 1" in reply["result"]["content"][0]["text"]
    assert backend.loads == 0


@pytest.mark.parametrize("command,limit", [("music", 3600), ("video", 20)])
@pytest.mark.parametrize("duration", ["0.5", "4000"])
def test_the_cli_refuses_a_duration_out_of_range(tmp_path, monkeypatch, command, limit,
                                                duration):
    import localm.cli.media as cli_media
    from localm.cli import main
    backend = _Backend()
    monkeypatch.setattr(cli_media, "_media_plugin", lambda plugin, sub: backend)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, [command, "x", "-d", duration])
    assert result.exit_code == 1, result.output
    assert f"Duration must be between 1 and {limit} seconds" in result.output
    assert backend.loads == 0 and backend.generated == []
    assert list(tmp_path.glob(f"{command}_*")) == []
