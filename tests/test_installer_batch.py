# SPDX-License-Identifier: AGPL-3.0-or-later
"""Static guard against the cmd.exe parenthesis crash class in .bat installers.

The crash class
---------------
cmd.exe counts UNescaped parens when it matches a parenthesised block, so a
``)`` inside a command *inside* a block terminates the enclosing ``if ... (``
block early and the rest of the line is parsed as a brand-new command. That
produces ``+ was unexpected at this time.``, or ``: was unexpected at this
time.`` for the same shape with a trailing ``:``, and the installer dies.

So an unquoted paren in any command *inside* a block MUST be escaped as
``^(`` / ``^)``. Parens at the top level (depth 0), e.g. the backend menu, are
harmless and are not flagged; neither is a paren inside a double-quoted string
(cmd.exe's block parser does not count those) or the FOR /F ``in ("...")``
idiom, which wraps a quoted literal in cmd.exe's own grammar.

This is a cheap static lint, not a full cmd parser.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BAT_FILES = sorted(REPO_ROOT.glob("*.bat"))


def find_unescaped_block_parens(text: str) -> list[tuple[int, str]]:
    """Return (line_number, line) for any non-comment, non-label line inside a
    block that contains an unescaped, unquoted ``(`` or ``)``.

    Block depth is tracked structurally: a block opens when a line ENDS with
    ``(`` (``if ... (`` / ``else (`` / ``for ... (``) and closes when a line
    STARTS with ``)``. That structural opener/closer is excluded from the
    hazard scan, as is a paren inside a double-quoted segment (no nested
    quoting in cmd) and the FOR /F ``in ("...")`` parenthesised-literal idiom.
    """
    offenders: list[tuple[int, str]] = []
    depth = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        low = stripped.lower()
        is_comment = low.startswith("rem ") or low == "rem" or stripped.startswith("::")
        is_label = stripped.startswith(":") and not stripped.startswith("::")

        if depth > 0 and not is_comment and not is_label:
            body = stripped.replace("^(", "").replace("^)", "")
            # The structural block opener/closer is legitimate cmd.exe syntax.
            if body.endswith("("):
                body = body[:-1]
            if body.startswith(")"):
                body = body[1:]
            # `("...")` - the FOR /F `in (...)` idiom - wraps a quoted literal
            # in cmd.exe's own grammar, not a stray paren.
            body = re.sub(r'\("[^"]*"\)', "", body)
            # Anything else quoted is likewise outside the block parser's
            # count (no nested quoting in a .bat file).
            unquoted = "".join(seg for i, seg in enumerate(body.split('"')) if i % 2 == 0)
            if "(" in unquoted or ")" in unquoted:
                offenders.append((lineno, raw))

        # Update structural depth AFTER classifying this line.
        if stripped.endswith("("):
            depth += 1
        if stripped.startswith(")"):
            depth = max(0, depth - 1)
    return offenders


def test_installer_batch_files_are_found() -> None:
    """An empty BAT_FILES collects the parametrized test below as SKIPPED, not
    FAILED, so this is the only thing that notices the installers moving or
    the glob going stale and silently turning off the lint."""
    assert BAT_FILES, "no *.bat at the repo root; the installer lint is not running"
    assert REPO_ROOT / "setup.bat" in BAT_FILES


@pytest.mark.parametrize("bat", BAT_FILES, ids=[p.name for p in BAT_FILES])
def test_no_unescaped_parens_inside_batch_blocks(bat: Path) -> None:
    text = bat.read_text(encoding="utf-8", errors="replace")
    offenders = find_unescaped_block_parens(text)
    if offenders:
        detail = "\n".join(f"  {bat.name}:{n}: {line.strip()}" for n, line in offenders)
        pytest.fail(
            f"Unescaped parens inside a cmd block in {bat.name} - cmd.exe will "
            f"close the block early and crash ('X was unexpected at this time').\n"
            f"Escape them as ^( and ^):\n{detail}"
        )


def test_checker_flags_a_known_bad_snippet() -> None:
    """The checker FLAGS an in-block echo carrying unescaped parens."""
    bad = (
        '@echo off\r\n'
        'if /i "%VENDOR%"=="amd" (\r\n'
        '    echo  Installing PyTorch (AMD ROCm) + transformers ...\r\n'
        ')\r\n'
    )
    offenders = find_unescaped_block_parens(bad)
    assert offenders, "checker failed to flag a known-bad in-block paren echo"
    assert offenders[0][0] == 3


def test_checker_ignores_safe_patterns() -> None:
    """Top-level parens and properly escaped in-block parens are not flagged."""
    ok = (
        '@echo off\r\n'
        'echo    [1] amd-rocm   (recommended for your hardware)\r\n'   # depth 0: safe
        'if /i "%VENDOR%"=="amd" (\r\n'
        '    echo  Installing PyTorch ^(AMD ROCm^) + transformers ...\r\n'  # escaped
        ') else (\r\n'
        '    echo  Skipping ^(not needed^).\r\n'
        ')\r\n'
    )
    assert find_unescaped_block_parens(ok) == []


def test_checker_flags_an_unquoted_non_echo_paren() -> None:
    """The widened scope covers any command, not only echo: an unescaped
    paren in a `set` (or any other) command inside a block is the same
    cmd.exe hazard."""
    bad = (
        'if x (\r\n'
        '    set FOO=value (unquoted)\r\n'
        ')\r\n'
    )
    offenders = find_unescaped_block_parens(bad)
    assert offenders, "checker failed to flag an unquoted non-echo paren inside a block"
    assert offenders[0][0] == 2


def test_checker_ignores_quoted_and_for_f_parens() -> None:
    """A paren inside a double-quoted string (a `set /p` prompt) and the
    FOR /F `in ("...")` idiom are cmd.exe quoting/grammar, not a hazard."""
    ok = (
        'if x (\r\n'
        '    set /p "X=prompt (quoted)"\r\n'
        '    for /f "usebackq" %%i in ("f") do (\r\n'
        '        echo hi\r\n'
        '    )\r\n'
        ')\r\n'
    )
    assert find_unescaped_block_parens(ok) == []
