# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every `localm <command> ...` line in a fenced block of README.md or docs/*.md names
a command that exists and flags that command accepts."""
import re
from pathlib import Path

import click
import pytest

from localm import cli as localm_cli

ROOT = Path(__file__).resolve().parents[1]
_LINE = re.compile(r"\s*(?:\$ )?localm((?: [a-z][a-z0-9-]*)+)((?: .*)?)$")
_QUOTED = re.compile(r'"[^"]*"|\'[^\']*\'')
_FLAG = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")


def _group() -> click.Group:
    group = getattr(localm_cli, "main", None)
    assert isinstance(group, click.Group)
    return group


def _options(command) -> set[str]:
    return {opt for param in command.params
            for opt in getattr(param, "opts", []) + getattr(param, "secondary_opts", [])}


def documented_command_lines():
    """``(file, line number, text)`` of every `localm ...` line inside a fenced block."""
    files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    for path in files:
        fenced = False
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip().startswith("```"):
                fenced = not fenced
            elif fenced and _LINE.match(line):
                yield path.relative_to(ROOT).as_posix(), number, line


def problems_with(line: str, group: click.Group) -> list[str]:
    """What is wrong with one documented `localm` line: an unknown command or a flag
    the resolved command does not accept."""
    match = _LINE.match(line)
    words, rest = match.group(1).split(), _QUOTED.sub("", match.group(2))
    command, used = group, []
    for word in words:
        if isinstance(command, click.Group) and word in command.commands:
            command = command.commands[word]
            used.append(word)
        else:
            break
    if not used:
        return [f"unknown command {words[0]!r}"]
    allowed = _options(command) | _options(group) | {"--help", "-h"}
    return [f"{' '.join(used)} has no flag {flag}" for flag in _FLAG.findall(rest)
            if flag not in allowed]


def test_the_documented_command_lines_are_found():
    assert sum(1 for _ in documented_command_lines()) > 100


def test_every_documented_command_line_uses_real_commands_and_flags():
    group = _group()
    found = [f"{name}:{number}: {problem}"
             for name, number, line in documented_command_lines()
             for problem in problems_with(line, group)]
    assert not found, "\n".join(found)


@pytest.mark.parametrize("line,fragment", [
    ("localm serve --no-such-flag", "serve has no flag --no-such-flag"),
    ("localm nosuchcommand --port 1", "unknown command 'nosuchcommand'"),
])
def test_a_wrong_line_is_reported(line, fragment):
    assert problems_with(line, _group()) == [fragment]


def test_a_flag_inside_a_quoted_argument_is_not_a_flag_of_the_command():
    assert problems_with('localm coder --until "pytest -x"', _group()) == []
