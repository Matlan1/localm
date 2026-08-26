# SPDX-License-Identifier: AGPL-3.0-or-later
"""FOUR different call sites build the inference Engine for a named model, and a
site that only passes a vision projector (mmproj_path) when the caller names one
with --mmproj leaves a model pulled with an auto-detected mmproj (see
model_manager/pull.py) unable to see an image once actually served or run - the
registry field written and never read.

This pins all four so a fifth new engine-construction site cannot ship the
same gap silently:
  - localm/inference/http_server.py:_default_engine_factory
  - localm/inference/http_server.py:mount_gui_surface's _build_engine
  - localm/plugins/gui/cli.py's _make_engine
  - localm/cli/chat.py's `run` command's Engine construction
"""

import asyncio

import pytest
from click.testing import CliRunner


class _FakeEngine:
    """A stand-in for inference.engine.Engine that only records the kwargs it
    was constructed with. Supports the context-manager protocol (cli/chat.py
    uses `with engine:`) and answers any other attribute access with a no-op
    callable, so callers that poke at display_name/chat_stream/count_tokens/
    etc. after construction do not need each one hand-modelled here - this
    test only cares what mmproj_path the REAL constructor call received."""

    def __init__(self, *args, **kwargs):
        self.captured = kwargs
        self.display_name = kwargs.get("display_name") or "vision-model"
        # switch_engine's _gpu_placement_fields does getattr(engine,
        # 'gpu_placement', None) and dict()s it when truthy, so this stub needs
        # a real not-available shape.
        self.gpu_placement = None

    def chat_stream(self, *args, **kwargs):
        return iter(["ok"])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture()
def registry_with_mmproj(tmp_path, monkeypatch):
    """A registry with one 'llm' entry whose 'mmproj' field is set - the exact
    shape model_manager/pull.py's auto-fetch writes. get_model_mmproj only trusts
    a RECORDED mmproj that still exists on disk (else it falls through to sibling
    auto-detect), so the projector must be a real file, not just a string.
    Returns (model_path, mmproj_path) as strings."""
    model_path = tmp_path / "vision-model.gguf"
    model_path.write_bytes(b"GGUF" + b"\x00" * 64)
    mmproj_path = tmp_path / "mmproj-vision-f16.gguf"
    mmproj_path.write_bytes(b"GGUF" + b"\x00" * 64)
    reg = {
        "vision-model": {
            "path": str(model_path),
            "source": "hf:owner/repo",
            "model_type": "llm",
            "mmproj": str(mmproj_path),
        }
    }
    monkeypatch.setattr("localm.model_manager.load_registry", lambda: dict(reg))
    monkeypatch.setattr("localm.config.load_registry", lambda: dict(reg))
    return str(model_path), str(mmproj_path)


class TestHttpServerFactoriesAlreadyCorrect:
    """Control cases: these two already call get_model_mmproj - pinned so a
    future edit cannot silently drop it."""

    def test_default_engine_factory_threads_mmproj(self, registry_with_mmproj, monkeypatch):
        _model_path, mmproj_path = registry_with_mmproj
        from localm.inference import http_server as hs

        captured = {}

        def _spy(*a, **kw):
            captured.update(kw)
            return _FakeEngine(*a, **kw)

        monkeypatch.setattr(hs, "Engine", _spy)
        hs._default_engine_factory("vision-model")
        assert captured.get("mmproj_path") == mmproj_path

    def test_mount_gui_surface_build_engine_threads_mmproj(
            self, registry_with_mmproj, tmp_path, monkeypatch):
        _model_path, mmproj_path = registry_with_mmproj
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        from localm.inference.http_server import create_app, mount_gui_surface

        app = create_app(None)
        app.state.instance_id = "iid-test"
        app.state.instance_token = "inst-secret-token"
        app.state.instance_mode = "api"
        app.state.instance_port = 8642
        app.state.instance_scheme = "http"
        app.state.bind_host = "127.0.0.1"

        assert mount_gui_surface(app) is True

        captured = {}

        def _spy(*a, **kw):
            captured.update(kw)
            return _FakeEngine(*a, **kw)

        import localm.inference.http_server as hs
        monkeypatch.setattr(hs, "Engine", _spy)

        # switch_model is the closure mount_gui_surface wired into attach_gui
        # and stashed on app.state; reach it the way routes/models.py does.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(app.state.switch_model("vision-model"))
        finally:
            loop.close()
        assert captured.get("mmproj_path") == mmproj_path


class TestPreviouslyBrokenFactoriesNowFixed:
    """Regression coverage for the two sites that did not consult get_model_mmproj."""

    def test_gui_cli_make_engine_threads_mmproj(
            self, registry_with_mmproj, tmp_path, monkeypatch):
        _model_path, mmproj_path = registry_with_mmproj
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        monkeypatch.setattr("localm.winconsole.disable_quickedit", lambda: None)
        monkeypatch.setattr("localm.instances.resolve_root_dir", lambda *a, **k: "/proj")
        monkeypatch.setattr("localm.instances.find_attachable", lambda *a, **k: None)
        monkeypatch.setattr("localm.portmux.run_server", lambda *a, **k: None)

        captured = {}

        def _spy(*a, **kw):
            captured.update(kw)
            return _FakeEngine(*a, **kw)

        monkeypatch.setattr("localm.inference.engine.Engine", _spy)

        from localm.plugins.gui import cli as guicli
        result = CliRunner().invoke(
            guicli.main,
            ["vision-model", "--no-browser", "--no-tls", "--new", "--isolated"],
        )
        assert result.exit_code == 0, result.output
        assert captured.get("mmproj_path") == mmproj_path, result.output

    def test_cli_run_engine_threads_mmproj(self, registry_with_mmproj, tmp_path, monkeypatch):
        _model_path, mmproj_path = registry_with_mmproj
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))

        captured = {}

        def _spy(*a, **kw):
            captured.update(kw)
            return _FakeEngine(*a, **kw)

        monkeypatch.setattr("localm.inference.engine.Engine", _spy)

        from localm.cli import chat as chatcli
        result = CliRunner().invoke(
            chatcli.run, ["vision-model", "-p", "hello", "--no-server"])
        assert result.exit_code == 0, result.output
        assert captured.get("mmproj_path") == mmproj_path, result.output


class TestExplicitMmprojStillWins:
    """An explicit --mmproj flag must not be silently overridden by the
    registry auto-detect on either just-fixed site."""

    def test_gui_cli_explicit_mmproj_wins(self, registry_with_mmproj, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        monkeypatch.setattr("localm.winconsole.disable_quickedit", lambda: None)
        monkeypatch.setattr("localm.instances.resolve_root_dir", lambda *a, **k: "/proj")
        monkeypatch.setattr("localm.instances.find_attachable", lambda *a, **k: None)
        monkeypatch.setattr("localm.portmux.run_server", lambda *a, **k: None)

        captured = {}

        def _spy(*a, **kw):
            captured.update(kw)
            return _FakeEngine(*a, **kw)

        monkeypatch.setattr("localm.inference.engine.Engine", _spy)

        from localm.plugins.gui import cli as guicli
        other = "/explicit/other-mmproj.gguf"
        result = CliRunner().invoke(
            guicli.main,
            ["vision-model", "--mmproj", other, "--no-browser", "--no-tls",
             "--new", "--isolated"],
        )
        assert result.exit_code == 0, result.output
        assert captured.get("mmproj_path") == other, result.output

    def test_cli_run_explicit_mmproj_wins(self, registry_with_mmproj, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))

        captured = {}

        def _spy(*a, **kw):
            captured.update(kw)
            return _FakeEngine(*a, **kw)

        monkeypatch.setattr("localm.inference.engine.Engine", _spy)

        from localm.cli import chat as chatcli
        other = "/explicit/other-mmproj.gguf"
        result = CliRunner().invoke(
            chatcli.run,
            ["vision-model", "-p", "hello", "--no-server", "--mmproj", other])
        assert result.exit_code == 0, result.output
        assert captured.get("mmproj_path") == other, result.output
