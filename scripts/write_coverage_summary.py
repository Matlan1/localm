#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render the coverage.json produced by the Tests step as a GitHub Actions
step summary, so the measured numbers survive after the runner is gone.

Read-only and never fails the build: this is a display step, not a gate. The
gates themselves are pyproject.toml's [tool.coverage.report] fail_under and
scripts/check_coverage_floors.py.

Run:  python scripts/write_coverage_summary.py [path/to/coverage.json]
"""

from __future__ import annotations

import json
import os
import platform as platform_module
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def render_summary(
    platform: str,
    percent_covered: float,
    fail_under: float | None,
    module_rows: list[tuple[str, float | None, int]] | None,
) -> str:
    """Markdown for one platform's coverage step summary."""
    lines = [f"### Coverage ({platform})", ""]
    if fail_under is None:
        lines.append(f"**{percent_covered:.2f}%** measured (floor unavailable).")
    else:
        headroom = percent_covered - fail_under
        lines.append(
            f"**{percent_covered:.2f}%** measured, ratchet floor **{fail_under:g}%** "
            f"(headroom {headroom:+.2f}).")
    if module_rows:
        lines += [
            "",
            "<details><summary>Per-module trust-boundary floors</summary>",
            "",
            "| module | measured | floor | headroom |",
            "|---|---:|---:|---:|",
        ]
        for module, actual, floor in module_rows:
            if actual is None:
                lines.append(f"| `{module}` | absent | {floor}% | - |")
            else:
                lines.append(
                    f"| `{module}` | {actual:.2f}% | {floor}% | {actual - floor:+.2f} |")
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def _read_fail_under() -> float | None:
    """Best-effort read of pyproject.toml's [tool.coverage.report] fail_under."""
    try:
        import tomllib
        data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        return data["tool"]["coverage"]["report"]["fail_under"]
    except (ModuleNotFoundError, OSError, KeyError, TypeError, ValueError) as e:
        print(f"note: could not read fail_under from pyproject.toml: {e}", file=sys.stderr)
        return None


def _read_module_rows(data: dict) -> list[tuple[str, float | None, int]] | None:
    """Best-effort per-module rows via the sibling trust-boundary floor script."""
    try:
        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import check_coverage_floors as ccf
        return ccf.report_rows(data)
    except (ImportError, ModuleNotFoundError) as e:
        print(f"note: could not load per-module coverage floors: {e}", file=sys.stderr)
        return None


def _publish(text: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv else REPO / "coverage.json"
    plat = os.environ.get("RUNNER_OS") or platform_module.system() or "unknown"

    if not path.is_file():
        _publish(f"### Coverage ({plat})\n\nNo coverage report at `{path}` - the "
                  "Tests step likely did not produce one.\n")
        return 0

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        _publish(f"### Coverage ({plat})\n\nCould not read `{path.name}`: {e}\n")
        return 0

    try:
        percent_covered = data["totals"]["percent_covered"]
    except (KeyError, TypeError) as e:
        _publish(f"### Coverage ({plat})\n\n`{path.name}` has no totals.percent_covered: {e}\n")
        return 0

    fail_under = _read_fail_under()
    module_rows = _read_module_rows(data) if plat == "Windows" else None
    _publish(render_summary(plat, percent_covered, fail_under, module_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
