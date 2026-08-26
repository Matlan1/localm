# SPDX-License-Identifier: AGPL-3.0-or-later
"""Static tripwire against the GUI nav render/fetch loop.

reconcileActiveView re-asserts the active highlight via the _applyActiveClasses
helper WITHOUT re-firing onViewShown for the current view; it may only call
showView for the genuine fallback to chat. refreshSettingsPage guards
overlapping renders with a token.

The GUI has no JS test harness (jsdom etc.), so these are source-level guards:
they fail loudly if the loop or the unguarded-render pattern is reintroduced."""

import re
from pathlib import Path

_STATIC = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "gui" / "static"


def _all_js() -> str:
    """The shipped GUI JS as one string: EVERY .js under static/ (recursively,
    minus vendored third-party libs). The guarded functions are unique by name,
    so which module holds one does not matter."""
    return "\n".join(
        p.read_text(encoding="utf-8")
        for p in sorted(_STATIC.rglob("*.js"))
        if "vendor" not in p.parts
    )


def _func_body(src: str, name: str) -> str:
    """Return the {...} body of `function <name>(...)` by brace matching."""
    marker = f"function {name}("
    start = src.index(marker)
    brace = src.index("{", start)
    depth = 0
    for i in range(brace, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[brace:i + 1]
    raise AssertionError(f"unbalanced braces in function {name}")


def _strip_line_comments(src: str) -> str:
    """Drop // line comments so we match real code, not explanatory comments
    (which legitimately mention the very patterns we guard against)."""
    return "\n".join(line.split("//", 1)[0] for line in src.splitlines())


def test_apply_active_classes_helper_exists():
    src = _all_js()
    assert "function _applyActiveClasses(" in src, (
        "_applyActiveClasses (highlight-only re-assert, split out of showView) is "
        "missing - without it the nav rebuild re-fires onViewShown and loops")


def test_showview_uses_apply_active_classes():
    body = _func_body(_all_js(), "showView")
    assert "_applyActiveClasses(" in body


def test_reconcile_active_view_does_not_reenter_showview():
    """reconcileActiveView must NOT call showView for the current/valid view -
    that re-enters onViewShown and recreates the /api/plugins loop. Only the
    fallback showView("chat") is allowed."""
    body = _strip_line_comments(
        _func_body(_all_js(), "reconcileActiveView"))
    assert "_applyActiveClasses(" in body, (
        "reconcileActiveView must re-assert the highlight via _applyActiveClasses")
    for arg in re.findall(r"showView\(([^)]*)\)", body):
        arg = arg.strip()
        assert arg in ('"chat"', "'chat'"), (
            f'reconcileActiveView calls showView({arg}); only the "chat" fallback '
            "is allowed - calling showView for the current view re-enters "
            "onViewShown and recreates the /api/plugins render loop")


def test_settings_render_is_guarded_against_overlap():
    """refreshSettingsPage must guard overlapping renders with a token so two
    concurrent calls cannot double-render the form."""
    body = _func_body(_all_js(), "refreshSettingsPage")
    assert "_settingsRenderToken" in body, (
        "refreshSettingsPage must use a render token to drop superseded renders")
