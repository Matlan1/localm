# SPDX-License-Identifier: AGPL-3.0-or-later
"""Review guard: flag coder edits that a passing check cannot vouch for."""

from __future__ import annotations

# JS/TS test files by suffix.
_TEST_SUFFIXES = (
    ".test.js", ".spec.js", ".test.jsx", ".spec.jsx",
    ".test.ts", ".spec.ts", ".test.tsx", ".spec.tsx",
    ".test.mjs", ".spec.mjs",
)

# CI / lint / format config by exact filename. pyproject.toml is deliberately NOT
# here: it also holds dependencies and project metadata, so flagging every edit to
# it would be noise. Files here exist primarily to define a gate.
_CI_CONFIG_NAMES = frozenset({
    ".gitlab-ci.yml", ".travis.yml", "azure-pipelines.yml",
    ".pre-commit-config.yaml", ".pre-commit-config.yml",
    "ruff.toml", ".ruff.toml", ".flake8", "tox.ini", "setup.cfg",
    ".eslintrc", ".eslintrc.js", ".eslintrc.json", ".eslintrc.cjs",
})


def _norm(path: str) -> str:
    return str(path).replace("\\", "/").strip("/")


def _is_test_file(path: str) -> bool:
    p = _norm(path)
    parts = p.split("/")
    name = parts[-1] if parts else p
    if "tests" in parts or "test" in parts:
        return True
    if name == "conftest.py":
        return True
    if name.startswith("test_") and name.endswith(".py"):
        return True
    if name.endswith("_test.py"):
        return True
    if any(name.endswith(s) for s in _TEST_SUFFIXES):
        return True
    return False


def _is_ci_config(path: str) -> bool:
    p = _norm(path)
    parts = p.split("/")
    name = parts[-1] if parts else p
    if ".github" in parts and "workflows" in parts:
        return True
    if name in _CI_CONFIG_NAMES:
        return True
    return False


def classify_sensitive_changes(changed_files: list) -> dict:
    """Split a changed-file list into the categories a green check cannot vouch for."""
    tests: list[str] = []
    ci: list[str] = []
    for entry in changed_files or []:
        path = entry.get("path") if isinstance(entry, dict) else str(entry)
        if not path:
            continue
        # Emit a normalised (forward-slash) path so the output is consistent
        # regardless of the OS separator the change was tracked with.
        norm = _norm(path)
        if _is_test_file(norm):
            tests.append(norm)
        elif _is_ci_config(norm):
            ci.append(norm)
    return {"tests": sorted(set(tests)), "ci_config": sorted(set(ci))}


def render_warning(flags: dict) -> str:
    """A human-readable warning for the classified changes, or '' if there are none."""
    tests = flags.get("tests") or []
    ci = flags.get("ci_config") or []
    if not tests and not ci:
        return ""
    lines = ["This session changed files a passing check cannot vouch for - "
             "review these by hand before trusting the result:"]
    if tests:
        lines.append("  Test files (a green run over rewritten assertions proves "
                     "nothing - confirm the new assertions are correct):")
        lines.extend("    - " + p for p in tests)
    if ci:
        lines.append("  CI / lint config (weakening the gate makes the gate lie):")
        lines.extend("    - " + p for p in ci)
    return "\n".join(lines)
