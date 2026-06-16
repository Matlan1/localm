"""Static tripwire against the GUI nav render/fetch loop.

Bug (2026-06-16): loading the GUI / opening Settings fired ~230 GET /api/plugins
in a burst and rendered the settings form twice (60 inputs instead of 30). Root
cause was a re-entrant cycle:

    onViewShown(chat|coder) -> refreshPluginCommands -> renderNav
    -> reconcileActiveView -> showView -> onViewShown -> ...

The fix: reconcileActiveView re-asserts the active highlight via the
_applyActiveClasses helper WITHOUT re-firing onViewShown for the current view;
it may only call showView for the genuine fallback to chat. And
refreshSettingsPage guards overlapping renders with a token.

The GUI has no JS test harness (jsdom etc.), so these are source-level guards:
not a substitute for executing the JS, but they fail loudly if the specific loop
or the unguarded-render pattern is reintroduced. A real JS test harness is a
separate, larger piece of work."""

import re
from pathlib import Path

_STATIC = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "gui" / "static"
APP_JS = _STATIC / "app.js"
PAGES_JS = _STATIC / "pages.js"


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
    src = APP_JS.read_text(encoding="utf-8")
    assert "function _applyActiveClasses(" in src, (
        "_applyActiveClasses (highlight-only re-assert, split out of showView) is "
        "missing - without it the nav rebuild re-fires onViewShown and loops")


def test_showview_uses_apply_active_classes():
    body = _func_body(APP_JS.read_text(encoding="utf-8"), "showView")
    assert "_applyActiveClasses(" in body


def test_reconcile_active_view_does_not_reenter_showview():
    """reconcileActiveView must NOT call showView for the current/valid view -
    that re-enters onViewShown and recreates the /api/plugins loop. Only the
    fallback showView("chat") is allowed."""
    body = _strip_line_comments(
        _func_body(APP_JS.read_text(encoding="utf-8"), "reconcileActiveView"))
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
    concurrent calls can't double-render the form (the 60-vs-30 inputs bug)."""
    body = _func_body(PAGES_JS.read_text(encoding="utf-8"), "refreshSettingsPage")
    assert "_settingsRenderToken" in body, (
        "refreshSettingsPage must use a render token to drop superseded renders")
