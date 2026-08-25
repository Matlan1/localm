# SPDX-License-Identifier: AGPL-3.0-or-later
"""docs/cli.md claims to document "the full localm command-line interface"
(docs/cli.md:3). Bind that claim to the actual registered Click commands so a
new top-level command added under localm/cli/*.py without a matching doc row
fails here instead of silently leaving the page short.

This deliberately reads the REAL docs/cli.md rather than a fixture string.
The defect this test exists to catch (`make-launcher`, registered at
localm/cli/maintenance.py:234, absent from docs/cli.md) lives entirely in the
real file's content - a synthetic fixture would have to invent the same gap
by hand, and would pass today, with the bug fully in place, for no reason
other than nobody thought to reproduce it there too.
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
    whether their optional extras import cleanly (see
    localm/cli/maintenance.py's _wire_plugin_cli_entries), so this set is
    environment-dependent - a CI install of `.[dev,rag]` may register fewer
    of them than a full dev install. That is deliberate: the assertion below
    only ever checks "what IS registered is documented", never a fixed
    count, so it stays correct in both environments.
    """
    return {name for name, cmd in main.commands.items() if not cmd.hidden}


def _documents_invocation(doc: str, name: str) -> bool:
    """True if `doc` shows `localm <name>` as an invocation, not merely as a
    substring.

    A bare `name in doc` check would be satisfied by ordinary prose and by
    SUBcommands that happen to share a word with a top-level command -
    `localm rag add`, `localm key list`, `localm job add` would each satisfy
    a naive "add" / "list" substring check even if the top-level `localm
    add` / `localm list` command were never mentioned anywhere. Requiring
    `localm` immediately before the name (only whitespace between) means a
    subcommand mention can never satisfy a top-level assertion, because
    there is always another word sitting between `localm` and the
    subcommand's own name.
    """
    return re.search(r"localm\s+" + re.escape(name) + r"\b", doc) is not None


def test_matcher_does_not_accept_an_absent_command():
    """Fires-control on the matcher itself, before trusting it below: it must
    be able to report a command as UNdocumented, or the test that uses it
    proves nothing (diff-review-discipline.md item 19 - a matcher that
    cannot produce a negative result is not a matcher, it is a tautology)."""
    assert not _documents_invocation(_doc_text(), "totally-not-a-real-cli-command")


def test_every_visible_cli_command_is_documented_in_cli_reference():
    """docs/cli.md:3 claims to document the full CLI. Every non-hidden
    top-level command must therefore appear in it as an invocation."""
    doc = _doc_text()
    missing = sorted(
        name for name in _visible_top_level_commands()
        if not _documents_invocation(doc, name)
    )
    assert not missing, f"commands missing from docs/cli.md: {missing}"
