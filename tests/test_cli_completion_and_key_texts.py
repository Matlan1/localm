# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm __complete` (the PowerShell completion's source) answers from Click's
own completion: every command that completes model names, and options. The
PowerShell completer quotes a candidate with a space. `key create --help` names
only the tiers it accepts."""

import re
import shutil
import subprocess
import sys

import click
import pytest

from localm.cli import _core, main
from localm.cli import completion as completion_mod


def _complete(cli_runner, *words):
    res = cli_runner.invoke(main, ["__complete", *words])
    assert res.exit_code == 0, res.output
    return res.output.splitlines()


@pytest.fixture
def registered(cli_runner, monkeypatch):
    monkeypatch.setattr("localm.config.load_registry",
                        lambda *a, **k: {"my model": {}, "qwen": {}})
    return cli_runner


def _model_name_commands():
    out = []
    for name, cmd in main.commands.items():
        args = [p for p in cmd.params if isinstance(p, click.Argument)]
        if args and getattr(args[0], "_custom_shell_complete", None) is _core._complete_model_name:
            out.append(name)
    return sorted(out)


def test_every_model_name_command_completes_model_names(registered):
    commands = _model_name_commands()
    assert {"run", "serve", "benchmark", "rm"} <= set(commands), commands
    for name in commands:
        assert _complete(registered, name, "") == ["my model", "qwen"], name


def test_a_prefix_narrows_the_names(registered):
    assert _complete(registered, "run", "q") == ["qwen"]


def test_options_complete(registered):
    assert _complete(registered, "run", "--ct") == ["--ctx"]


def test_command_names_complete_without_hidden_ones(registered):
    names = _complete(registered, "")
    assert "benchmark" in names
    assert "__complete" not in names


@pytest.mark.skipif(sys.platform != "win32" or not shutil.which("powershell"),
                    reason="needs Windows PowerShell")
def test_the_powershell_completer_quotes_a_name_with_a_space():
    line = next(ln.strip() for ln in completion_mod._POWERSHELL_COMPLETION.splitlines()
                if ln.strip().startswith("if ($text -match"))
    script = "\n".join(f"$text = '{name}'; {line}; Write-Output $text"
                       for name in ("my model", "qwen", "it's"))
    out = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=60,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == ["'my model'", "qwen", "'it''s'"]


def test_key_create_help_names_only_the_tiers_it_accepts(cli_runner):
    res = cli_runner.invoke(main, ["key", "create", "--help"])
    assert res.exit_code == 0
    assert "shared" not in re.sub(r"a shared key", "", " ".join(res.output.split()))
