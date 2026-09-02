# SPDX-License-Identifier: AGPL-3.0-or-later
"""chat_system_prompt must reach the command line, not only the web GUI.

settings_schema describes it as "the system prompt every new chat starts with",
with a chat's own System prompt field overriding it. On the CLI that override is
--system. The setting used to be read in exactly one place, the browser's
chat.js, so a terminal chat silently started with no system prompt however the
setting was set.

The OpenAI-compatible endpoint is deliberately NOT covered: there the caller owns
the message list, and injecting a system message into a third-party client's
request would change an API contract rather than apply a chat default.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner


@pytest.fixture
def cli_runner():
    return CliRunner()


def _stub_engine(monkeypatch, captured):
    inst = MagicMock(name="EngineInstance")
    inst.__enter__ = MagicMock(return_value=inst)
    inst.__exit__ = MagicMock(return_value=False)

    def _stream(messages, **kw):
        captured.append(messages)
        return iter(["ok"])

    inst.chat_stream = MagicMock(side_effect=_stream)
    monkeypatch.setattr("localm.inference.engine.Engine",
                        MagicMock(name="Engine", return_value=inst))
    monkeypatch.setattr("localm.instances.attach_target", lambda *a, **k: None)
    return inst


def _roles(messages):
    return [m["role"] for m in messages]


def test_a_terminal_run_inherits_the_configured_system_prompt(
        cli_runner, tmp_path, monkeypatch):
    model_f = tmp_path / "m.gguf"
    model_f.write_bytes(b"GGUF")
    captured = []
    _stub_engine(monkeypatch, captured)

    import localm.cli.chat as chat_mod
    real = chat_mod.load_config
    monkeypatch.setattr(chat_mod, "load_config",
                        lambda: {**real(), "chat_system_prompt": "  Be terse.  "})

    from localm.cli.chat import run
    cli_runner.invoke(run, [str(model_f), "--no-server", "-p", "hi"])

    assert captured, "the engine was never asked to generate"
    msgs = captured[0]
    assert "system" in _roles(msgs), (
        f"a terminal run ignored chat_system_prompt entirely: {msgs}")
    assert msgs[0]["content"] == "Be terse.", (
        f"the configured prompt must be used, trimmed: {msgs[0]}")


def test_an_explicit_system_flag_overrides_the_setting(
        cli_runner, tmp_path, monkeypatch):
    model_f = tmp_path / "m.gguf"
    model_f.write_bytes(b"GGUF")
    captured = []
    _stub_engine(monkeypatch, captured)

    import localm.cli.chat as chat_mod
    real = chat_mod.load_config
    monkeypatch.setattr(chat_mod, "load_config",
                        lambda: {**real(), "chat_system_prompt": "from settings"})

    from localm.cli.chat import run
    cli_runner.invoke(run, [str(model_f), "--no-server", "-p", "hi",
                            "-s", "from the flag"])

    assert captured
    assert captured[0][0]["content"] == "from the flag", (
        "--system is the per-chat override and must win over the setting")


def test_an_empty_setting_still_means_no_system_prompt(
        cli_runner, tmp_path, monkeypatch):
    model_f = tmp_path / "m.gguf"
    model_f.write_bytes(b"GGUF")
    captured = []
    _stub_engine(monkeypatch, captured)

    import localm.cli.chat as chat_mod
    real = chat_mod.load_config
    monkeypatch.setattr(chat_mod, "load_config",
                        lambda: {**real(), "chat_system_prompt": "   "})

    from localm.cli.chat import run
    cli_runner.invoke(run, [str(model_f), "--no-server", "-p", "hi"])

    assert captured
    assert "system" not in _roles(captured[0]), (
        "a blank setting must not inject an empty system message")
