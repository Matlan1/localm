# SPDX-License-Identifier: AGPL-3.0-or-later
"""The configured delete-outputs setting (``comfy_delete_outputs``) reaches
ComfyUI from every way of generating media, and privacy mode forces it on.

The backend facade (``media_config.make_backend_facade``'s ``generate``) is
the one place that decides ``delete_outputs``; these tests drive each caller
(``localm image``, the REPL ``/generate-image``, the coder tool, the MCP tool)
down to the patched ComfyUI generator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner



@pytest.fixture(autouse=True)
def _comfy_image_backend(monkeypatch):
    """Every first-party plugin is available and the image plugin's ``auto``
    backend resolves to a running ComfyUI."""
    monkeypatch.setattr("localm.plugins.lookup.is_available", lambda name: True)
    monkeypatch.setattr("localm.media.backend_choice.comfy_is_set_up",
                        lambda *a, **k: True)
    monkeypatch.setattr("localm.image_gen.comfy.ensure_comfy",
                        lambda *a, **k: (True, "up"))
    monkeypatch.setattr("localm.image_gen.comfy.free_comfy_vram",
                        lambda *a, **k: True)
    monkeypatch.setattr("localm.image_gen.comfy.default_api_url",
                        lambda *a, **k: "http://127.0.0.1:9999")
    monkeypatch.setattr("localm.media.comfy_client._comfy_alive",
                        lambda *a, **k: True)


def _set_delete_outputs(value: bool) -> None:
    from localm.config import update_config
    update_config(lambda c: c.update(comfy_delete_outputs=value))


def _recording_generator(calls: list):
    def fake_generate_image(prompt, out, **kwargs):
        calls.append(kwargs)
        return True, "Image saved"
    return fake_generate_image


def _via_cli(calls, tmp_path, privacy):
    from localm.cli import main
    result = CliRunner().invoke(main, ["image", "a cat", "-o", str(tmp_path / "o.png")])
    assert result.exit_code == 0, result.output


def _via_repl(calls, tmp_path, privacy):
    from localm import cli
    cli._handle_command("/generate-image a cat", [], {}, engine=MagicMock())


def _via_coder_tool(calls, tmp_path, privacy):
    """The agent injects ``_privacy`` from its session mode."""
    from localm.plugins.coder.tools import tool_generate_image
    result = tool_generate_image(tmp_path, "a cat", "o.png", _privacy=privacy)
    assert result.ok, result.output


def _via_mcp_tool(calls, tmp_path, privacy):
    from localm.plugins.mcpserver.server import EngineCache, MCPStdioServer, build_tools
    engines = EngineCache(default_model="stub", engine_factory=lambda *a, **k: MagicMock())
    server = MCPStdioServer(build_tools(engines, enable_images=True))
    r = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": "generate_image", "arguments": {"prompt": "x"}}})
    assert r["result"].get("isError") is not True, r["result"]


CALLERS = [_via_cli, _via_repl, _via_coder_tool, _via_mcp_tool]


@pytest.mark.parametrize("caller", CALLERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("configured", [True, False])
def test_configured_setting_reaches_comfyui_outside_privacy(
        caller, configured, tmp_path, monkeypatch):
    from localm import cli
    monkeypatch.setattr(cli, "HOME_DIR", tmp_path)
    monkeypatch.setenv("LOCALM_MODE", "log")
    _set_delete_outputs(configured)
    calls: list = []
    with patch("localm.image_gen.comfy.generate_image", _recording_generator(calls)):
        caller(calls, tmp_path, False)
    assert len(calls) == 1
    assert calls[0]["delete_outputs"] is configured


@pytest.mark.parametrize("caller", CALLERS, ids=lambda f: f.__name__)
def test_privacy_mode_forces_delete_outputs(caller, tmp_path, monkeypatch):
    from localm import cli
    monkeypatch.setattr(cli, "HOME_DIR", tmp_path)
    monkeypatch.setenv("LOCALM_MODE", "privacy")
    _set_delete_outputs(False)
    calls: list = []
    with patch("localm.image_gen.comfy.generate_image", _recording_generator(calls)):
        caller(calls, tmp_path, True)
    assert len(calls) == 1
    assert calls[0]["delete_outputs"] is True
    assert calls[0]["write_sidecar"] is False


class _Impl:
    def __init__(self):
        self.kwargs = None

    def generate(self, s, prompt, out_path, **kwargs):
        self.kwargs = kwargs
        return True, "ok"


def _facade(impl):
    from localm.plugins import media_config
    return media_config.make_backend_facade("localm.plugins.builtin.image", impl, "image")


@pytest.mark.parametrize("configured,privacy,expected", [
    (True, False, True), (False, False, False), (False, True, True), (True, True, True),
])
def test_facade_decides_delete_outputs(configured, privacy, expected, tmp_path):
    impl = _Impl()
    _facade(impl).generate({"delete_outputs": configured}, "p", tmp_path / "o.png",
                           privacy=privacy)
    assert impl.kwargs["delete_outputs"] is expected


def test_facade_refuses_a_caller_supplied_delete_outputs(tmp_path):
    impl = _Impl()
    with pytest.raises(TypeError, match="delete_outputs"):
        _facade(impl).generate({"delete_outputs": True}, "p", tmp_path / "o.png",
                               delete_outputs=False)
    assert impl.kwargs is None


def test_every_media_call_site_leaves_delete_outputs_to_the_facade():
    """No caller of a media backend passes ``delete_outputs`` itself."""
    import ast
    root = Path(__file__).resolve().parent.parent / "localm"
    offenders = []
    facade = root / "plugins" / "media_config.py"
    for path in root.rglob("*.py"):
        if path == facade:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("generate", "generate_media")
                    and any(k.arg == "delete_outputs" for k in node.keywords)):
                offenders.append(f"{path.relative_to(root.parent)}:{node.lineno}")
    assert offenders == []


def _media_backend_calls():
    """Every call in localm/ of a media backend's ``generate`` or of
    ``generate_media``, outside the facade module, as (location, call)."""
    import ast
    root = Path(__file__).resolve().parent.parent / "localm"
    facade = root / "plugins" / "media_config.py"
    for path in root.rglob("*.py"):
        if path == facade:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            base = node.func.value
            if node.func.attr == "generate_media" or (
                    node.func.attr == "generate" and isinstance(base, ast.Name)
                    and base.id in ("_backend", "media_backend", "image_backend")):
                yield f"{path.relative_to(root.parent)}:{node.lineno}", node


def _is_plugin_forwarder(where: str) -> bool:
    parts = Path(where.split(":")[0]).parts
    return parts[:3] == ("localm", "plugins", "builtin") and parts[-1] == "backend.py"


def test_every_media_backend_call_states_privacy():
    """Each caller passes ``privacy=`` itself, so the facade can force
    delete_outputs in privacy mode. A plugin backend's ``generate_media``
    forwards its caller's ``**kwargs`` instead."""
    calls = list(_media_backend_calls())
    assert len(calls) >= 7
    missing = []
    for where, call in calls:
        names = {k.arg for k in call.keywords}
        if "privacy" in names:
            continue
        if _is_plugin_forwarder(where) and None in names:
            continue
        missing.append(where)
    assert missing == []


@pytest.fixture
def image_app(monkeypatch, tmp_path):
    from fastapi import FastAPI
    from localm.jobmanager import JobManager
    from localm.plugins.engine import PluginManager
    monkeypatch.setenv("LOCALM_TMPDIR", str(tmp_path / "scratch"))
    (tmp_path / "scratch").mkdir()
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr("localm.vram.decide_media_swap", lambda s: False)
    monkeypatch.setattr("localm.vram.media_single_device_shortfall", lambda s: None)
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("image")
    app.state.jobs = JobManager()
    app.state.self_url = "http://127.0.0.1:8642/v1"
    return app


def _png_writing_generator(calls: list):
    def fake_generate_image(prompt, out, **kwargs):
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, "PNG")
        Path(out).write_bytes(buf.getvalue())
        calls.append(kwargs)
        return True, f"Image saved to {out}"
    return fake_generate_image


@pytest.mark.parametrize("mode,configured,expected", [
    ("privacy", False, True), ("log", True, True), ("log", False, False),
])
def test_gui_image_route_reaches_comfyui_with_the_decided_value(
        image_app, monkeypatch, mode, configured, expected):
    """POST /v1/images/generations runs the same job as /api/imagine, through
    the real backend facade down to the ComfyUI generator."""
    from fastapi.testclient import TestClient
    monkeypatch.setenv("LOCALM_MODE", mode)
    _set_delete_outputs(configured)
    calls: list = []
    with patch("localm.image_gen.comfy.generate_image", _png_writing_generator(calls)), \
         TestClient(image_app) as client:
        r = client.post("/v1/images/generations",
                        json={"prompt": "a cat", "response_format": "b64_json"})
    assert r.status_code == 200, r.text
    assert len(calls) == 1
    assert calls[0]["delete_outputs"] is expected
    assert calls[0]["write_sidecar"] is (mode != "privacy")
