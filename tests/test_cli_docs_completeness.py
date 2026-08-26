# SPDX-License-Identifier: AGPL-3.0-or-later
"""docs/cli.md claims to document "the full localm command-line interface".
Bind that claim to the actual registered Click commands, so a new top-level
command added under localm/cli/*.py without a matching doc row fails here.

Reads the REAL docs/cli.md, not a fixture string.
"""

import re
from pathlib import Path

from localm.cli import main

_DOC = Path(__file__).resolve().parents[1] / "docs" / "cli.md"


def _doc_text() -> str:
    return _DOC.read_text(encoding="utf-8")


def _visible_top_level_commands() -> set:
    """Every non-hidden top-level command name actually registered on `main`.

    Plugin-contributed verbs (coder/job/mcp) register conditionally on
    whether their optional extras import cleanly, so this set is
    environment-dependent. The assertion below only ever checks "what IS
    registered is documented", never a fixed count.
    """
    return {name for name, cmd in main.commands.items() if not cmd.hidden}


def _documents_invocation(doc: str, name: str) -> bool:
    """True if `doc` shows `localm <name>` as an invocation, not merely as a
    substring.

    Requires `localm` immediately before the name, with only whitespace
    between, so a SUBcommand mention (`localm rag add`) can never satisfy a
    top-level assertion.
    """
    return re.search(r"localm\s+" + re.escape(name) + r"\b", doc) is not None


def test_matcher_does_not_accept_an_absent_command():
    """The matcher must be able to report a command as UNdocumented, or the
    test that uses it proves nothing."""
    assert not _documents_invocation(_doc_text(), "totally-not-a-real-cli-command")


def test_every_visible_cli_command_is_documented_in_cli_reference():
    """docs/cli.md claims to document the full CLI, so every non-hidden
    top-level command must appear in it as an invocation."""
    doc = _doc_text()
    missing = sorted(
        name for name in _visible_top_level_commands()
        if not _documents_invocation(doc, name)
    )
    assert not missing, f"commands missing from docs/cli.md: {missing}"
