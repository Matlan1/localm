# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm mcp: the printed client config, the refusal for an installed but
disabled plugin, and run_doctor's child process."""

import json
import subprocess
import sys

import pytest
from click.testing import CliRunner


@pytest.fixture
def cfg_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(_cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return _cfg


def test_print_config_stdout_is_one_json_document_naming_this_interpreter(cfg_env):
    from localm.plugins.mcpserver.cli import main
    result = CliRunner().invoke(main, ["--print-config", "--no-images"])
    assert result.exit_code == 0
    block = json.loads(result.stdout)
    entry = block["mcpServers"]["localm"]
    assert entry["command"] == sys.executable
    assert entry["args"] == ["-m", "localm", "mcp", "--no-images"]
    assert "claude_desktop_config.json" in result.stderr
    assert "localm plugin install mcp" in result.stderr


def test_installed_but_disabled_is_told_to_enable_not_install(cfg_env):
    from localm.plugins.engine import PluginManager
    from localm.plugins.mcpserver.cli import main
    mgr = PluginManager(None)
    mgr.set_installed_state("mcp", True)
    mgr.set_enabled_state("mcp", False)
    result = CliRunner().invoke(main, [])
    assert result.exit_code == 1
    assert "localm plugin enable mcp" in result.stderr
    assert "plugin install" not in result.stderr


def test_not_installed_is_told_to_install(cfg_env):
    from localm.plugins.mcpserver.cli import main
    result = CliRunner().invoke(main, [])
    assert result.exit_code == 1
    assert "localm plugin install mcp" in result.stderr


def test_run_doctor_child_opens_no_console_window(monkeypatch):
    from localm.plugins.mcpserver.tools import diagnostics
    seen: dict = {}

    def _run(cmd, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(diagnostics.subprocess, "run", _run)
    result = diagnostics.build()["run_doctor"]["handler"]({})
    assert result["content"][0]["text"].startswith("ok")
    assert seen["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
