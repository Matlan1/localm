"""Static guard against the cmd.exe parenthesis crash class in .bat installers.

Why this exists
---------------
`setup.bat` once shipped lines like::

    if /i "%VENDOR%"=="amd" (
        echo  Installing PyTorch (AMD ROCm) + transformers ...
    )

cmd.exe counts UNescaped parens when it matches a parenthesised block, so the
`)` inside the echo terminates the `if (` block early and the rest of the line
(`+ transformers ...`) is parsed as a brand-new command. The result is the
infamous ``+ was unexpected at this time.`` and the installer dies before the
user can do anything. The same shape with a trailing ``:`` produced
``: was unexpected at this time.`` in the uninstall path.

Nothing in CI executed the installers, so this shipped to every AMD and NVIDIA
user. This test closes that gap: parens that appear in an ``echo`` (or any
command) *inside* a block MUST be escaped as ``^(`` / ``^)``. Parens at the top
level (depth 0), e.g. the backend menu, are harmless and are not flagged.

This is a cheap static lint, not a full cmd parser, but it catches the entire
known family and enforces the safe style going forward.
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
    structural counter, which is exactly why cmd mis-handles them and why we
    flag them.
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
    """Negative-test the oracle: it must FLAG the exact pattern we just fixed."""
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
    """Top-level parens and properly escaped in-block parens must NOT be flagged."""
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
