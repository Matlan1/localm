# SPDX-License-Identifier: AGPL-3.0-or-later
"""plugins/gui/cli.py's _make_engine closes over the gui/serve command's own
--mmproj CLI value for the SERVER'S ENTIRE LIFETIME, and switch_model reuses
that same closure for every subsequent model switch. Unscoped, --mmproj X given
for a startup model reaches every later switch: an unrelated model gets X too,
and a model with its OWN correctly-recorded projector has it overridden by X.

--mmproj is therefore scoped to the model it was given for at startup; every
other name falls through to its own registry lookup exactly as if --mmproj had
never been given.
"""
import asyncio

import pytest
from click.testing import CliRunner


class _FakeEngine:
    """Mirrors test_engine_factory_mmproj.py's _FakeEngine - records only the
    kwargs it was constructed with."""

    def __init__(self, *args, **kwargs):
        self.captured = kwargs
        self.display_name = kwargs.get("display_name") or "test-model"
        self._loaded = False
        self.active_requests = 0
        self.gpu_placement = None

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def set_load_cancel(self, ev):
        pass

    def chat_stream(self, *args, **kwargs):
        return iter(["ok"])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture()
def three_models(tmp_path, monkeypatch):
    """model-a (the startup model, no own mmproj), model-b (switch target, no
    own mmproj), model-c (switch target WITH its own correctly-recorded
    projector)."""
    def _gguf(name):
        f = tmp_path / name
        f.write_bytes(b"GGUF" + b"\x00" * 64)
        return f

    model_a = _gguf("model-a.gguf")
    model_b = _gguf("model-b.gguf")
    model_c = _gguf("model-c.gguf")
    model_c_proj = _gguf("model-c-proj.gguf")

    reg = {
        "model-a": {"path": str(model_a), "source": "local", "model_type": "llm"},
        "model-b": {"path": str(model_b), "source": "local", "model_type": "llm"},
        "model-c": {"path": str(model_c), "source": "local", "model_type": "llm",
                    "mmproj": str(model_c_proj)},
    }
    monkeypatch.setattr("localm.model_manager.load_registry", lambda: dict(reg))
    monkeypatch.setattr("localm.config.load_registry", lambda: dict(reg))
    return reg


@pytest.fixture()
def gui_harness(monkeypatch):
    """Boots the real `gui` CLI command (via CliRunner) with only the
    heavy/side-effecting seams mocked - the server-attach probe, TLS/mDNS/
    console setup, the real Engine constructor, and the VRAM probe a real
    model switch consults. Returns (constructed, get_app) where constructed
    maps model name -> the kwargs its _FakeEngine was built with, and
    get_app() returns the FastAPI app instance main() created (captured via a
    spy on create_app), for driving a real post-startup switch through
    app.state.switch_model."""
    from localm.discover import vram_info as _real_vram_info  # noqa: F401
    from tests.conftest import probe_double

    monkeypatch.setattr("localm.winconsole.disable_quickedit", lambda: None)
    monkeypatch.setattr("localm.instances.resolve_root_dir", lambda *a, **k: "/proj")
    monkeypatch.setattr("localm.instances.find_attachable", lambda *a, **k: None)
    monkeypatch.setattr("localm.portmux.run_server", lambda *a, **k: None)
    monkeypatch.setattr("localm.discover.vram_info",
                        probe_double({"free": 10 * 1024 ** 3, "total": 16 * 1024 ** 3}))

    constructed = {}

    def _spy_engine(*a, **kw):
        engine = _FakeEngine(*a, **kw)
        # positional model_path is always given; the name is only known via
        # display_name (registered name) here, matching what the real Engine
        # constructor's caller passes.
        name = kw.get("display_name") or (a[0] if a else None)
        constructed[name] = kw
        return engine

    monkeypatch.setattr("localm.inference.engine.Engine", _spy_engine)

    captured_app = {}
    from localm.inference import http_server as hs
    real_create_app = hs.create_app

    def _spy_create_app(*a, **kw):
        app = real_create_app(*a, **kw)
        captured_app["app"] = app
        return app

    monkeypatch.setattr(hs, "create_app", _spy_create_app)

    def _get_app():
        return captured_app["app"]

    return constructed, _get_app


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestMmprojScopedToStartupModel:
    def test_startup_model_gets_the_override(self, three_models, gui_harness, tmp_path):
        constructed, _get_app = gui_harness
        startup_proj = str(tmp_path / "startup-proj.gguf")

        result = CliRunner().invoke(
            __import__("localm.plugins.gui.cli", fromlist=["main"]).main,
            ["model-a", "--mmproj", startup_proj, "--no-browser", "--no-tls",
             "--new", "--isolated"],
        )
        assert result.exit_code == 0, result.output
        assert constructed["model-a"]["mmproj_path"] == startup_proj

    def test_switching_to_unrelated_model_does_not_inherit_the_override(
            self, three_models, gui_harness, tmp_path):
        constructed, get_app = gui_harness
        startup_proj = str(tmp_path / "startup-proj.gguf")

        result = CliRunner().invoke(
            __import__("localm.plugins.gui.cli", fromlist=["main"]).main,
            ["model-a", "--mmproj", startup_proj, "--no-browser", "--no-tls",
             "--new", "--isolated"],
        )
        assert result.exit_code == 0, result.output

        switch_result = _run(get_app().state.switch_model("model-b"))
        assert switch_result.get("status") == "loaded", switch_result

        assert constructed["model-b"]["mmproj_path"] is None, (
            "model-b must not inherit model-a's --mmproj override")

    def test_switching_to_a_model_with_its_own_projector_uses_its_own(
            self, three_models, gui_harness, tmp_path):
        constructed, get_app = gui_harness
        startup_proj = str(tmp_path / "startup-proj.gguf")

        result = CliRunner().invoke(
            __import__("localm.plugins.gui.cli", fromlist=["main"]).main,
            ["model-a", "--mmproj", startup_proj, "--no-browser", "--no-tls",
             "--new", "--isolated"],
        )
        assert result.exit_code == 0, result.output

        switch_result = _run(get_app().state.switch_model("model-c"))
        assert switch_result.get("status") == "loaded", switch_result

        own_proj = str((tmp_path / "model-c-proj.gguf").resolve())
        assert constructed["model-c"]["mmproj_path"] == own_proj, (
            "model-c's own recorded projector must not be overridden by "
            "model-a's --mmproj")
        assert constructed["model-c"]["mmproj_path"] != startup_proj
