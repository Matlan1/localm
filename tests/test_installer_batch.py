# SPDX-License-Identifier: AGPL-3.0-or-later
"""Static guard against the cmd.exe parenthesis crash class in .bat installers.

The crash class
---------------
cmd.exe counts UNescaped parens when it matches a parenthesised block, so a
``)`` inside an ``echo`` terminates the enclosing ``if ... (`` block early and
the rest of the line is parsed as a brand-new command. That produces
``+ was unexpected at this time.``, or ``: was unexpected at this time.`` for
the same shape with a trailing ``:``, and the installer dies.

So parens appearing in an ``echo`` (or any command) *inside* a block MUST be
escaped as ``^(`` / ``^)``. Parens at the top level (depth 0), e.g. the backend
menu, are harmless and are not flagged.

This is a cheap static lint, not a full cmd parser.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BAT_FILES = sorted(REPO_ROOT.glob("*.bat"))


def find_unescaped_block_parens(text: str) -> list[tuple[int, str]]:
    """Return (line_number, line) for echo lines inside a block that contain an
    unescaped ``(`` or ``)``.

    Block depth is tracked structurally: a block opens when a line ENDS with
    ``(`` (``if ... (`` / ``else (`` / ``for ... (``) and closes when a line
    STARTS with ``)``. Parens embedded mid-line in an echo do not move the
    structural counter.
    """
    offenders: list[tuple[int, str]] = []
    depth = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        low = stripped.lower()
        is_comment = low.startswith("rem ") or low == "rem" or stripped.startswith("::")
        is_label = stripped.startswith(":") and not stripped.startswith("::")

        if depth > 0 and not is_comment and not is_label and "echo" in low:
            # Only consider it an echo statement, not the literal word in prose.
            # In these scripts an echo is either at line start or guarded by a
            # single-line `if ... echo ...`.
            looks_like_echo = (
                low.startswith("echo")
                or low.startswith("@echo")
                or " echo " in (" " + low)
            )
            if looks_like_echo:
                scrubbed = stripped.replace("^(", "").replace("^)", "")
                if "(" in scrubbed or ")" in scrubbed:
                    offenders.append((lineno, raw))

        # Update structural depth AFTER classifying this line.
        if stripped.endswith("("):
            depth += 1
        if stripped.startswith(")"):
            depth = max(0, depth - 1)
    return offenders


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
