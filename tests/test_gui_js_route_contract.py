# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contract test: every HTTP path the GUI's JavaScript fetches must be declared
as a real route somewhere in the Python source.

Nothing else ties the frontend to the backend, so a route deleted while its
caller stays behind 404s for every user while the suite stays green.

The check is static - it reads source, it does not build an app: the route table
of a live app depends on which plugins happen to be installed and loaded, which
would make the test's coverage depend on fixture state. Route paths are declared
as literals repo-wide (``@_router.get("/api/coder/sessions")``, no router
prefixes), so scanning the source sees all of them, deterministically.

A fetch with no matching ``@app.get()``-style declaration is also checked
against the GUI's static directory: a bare filename interpolation with a fixed
extension (``/i18n/${id}.json``) is what the catch-all static mount serves, so
it counts as declared only while a matching file actually exists on disk.

Limits: a ``fetch(url)`` whose URL is built in a variable is not seen, and a
route that is declared in source but never registered on the app still counts as
declared. This catches the "route deleted, caller left behind" class.
"""

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PKG = _ROOT / "localm"
_STATIC = _PKG / "plugins" / "gui" / "static"

# @app.get("/x") / @_router.post(f"/api/{media}/workflows") / @router.delete('/y')
_ROUTE_DECL = re.compile(
    r"""@[A-Za-z_][A-Za-z_0-9.]*\.(?:get|post|put|delete|patch)\(\s*f?(["'])(/[^"']*)\1""")

# fetch(`...`) needs its own pattern: a template literal may contain quotes
# inside an interpolation, e.g. `/x/${a ? '?b=1' : ''}`.
_FETCH_TMPL = re.compile(r"fetch\(\s*`([^`]*)`")
_FETCH_STR = re.compile(r"""fetch\(\s*(["'])(/[^"']*)\1""")

_JS_INTERP = re.compile(r"\$\{[^{}]*\}")     # ${expr} in a JS template literal
_PY_PARAM = re.compile(r"\{[^{}]*\}")        # {name} in a route path / f-string

_WILD = "\x00"                                # normalized wildcard segment marker


def _norm_route(path: str) -> str:
    """A declared route path -> comparable form ({name} and f-string {media} both
    become a wildcard: the JS interpolates exactly where the route varies)."""
    return _PY_PARAM.sub(_WILD, path).split("?")[0].rstrip("/") or "/"


def _norm_fetch(url: str) -> str:
    """A JS fetch URL literal -> comparable form."""
    prev = None
    while prev != url:                        # nested ${...} settle in a few passes
        prev = url
        url = _JS_INTERP.sub(_WILD, url)
    url = url.split("?")[0].split("#")[0]
    segs = []
    for seg in [s for s in url.split("/") if s]:
        if _WILD in seg and seg != _WILD:
            # A wildcard glued onto a literal is a query/suffix expression, not a
            # path segment: `/api/x/events${replay ? '?replay=true' : ''}`.
            head = seg.split(_WILD)[0]
            if head:
                segs.append(head)
            break
        segs.append(seg)
    return "/" + "/".join(segs) if segs else "/"


def _matches(path: str, route: str) -> bool:
    p = [s for s in path.split("/") if s]
    r = [s for s in route.split("/") if s]
    if len(p) != len(r):
        return False
    return all(a == b or _WILD in (a, b) for a, b in zip(p, r))


def _declared_routes() -> set:
    routes = set()
    for f in _PKG.rglob("*.py"):
        for m in _ROUTE_DECL.finditer(f.read_text(encoding="utf-8")):
            routes.add(_norm_route(m.group(2)))
    return routes


# A fetch whose filename is a bare interpolation with a fixed extension:
# `/dir/${expr}.ext`. No @app.get() can declare this shape; it is served by the
# catch-all static mount (web.py's `app.mount("/", _RevalidatingStatic(...))`),
# which serves whatever file already exists under _STATIC.
_STATIC_DYNAMIC_FETCH = re.compile(
    r"^/([A-Za-z0-9_-]+)/\$\{[^{}]*\}(\.[A-Za-z0-9]+)$")


def _served_by_static_mount(raw: str) -> bool:
    """Whether *raw* (a fetch() literal) is a dynamic-filename request the
    static mount can serve, checked against the files actually on disk."""
    m = _STATIC_DYNAMIC_FETCH.match(raw.split("?")[0])
    if not m:
        return False
    directory, suffix = m.groups()
    d = _STATIC / directory
    return d.is_dir() and any(p.suffix == suffix for p in d.iterdir() if p.is_file())


def _js_fetches() -> dict:
    """{normalized path: {"raw": literal, "files": [...]}} for every GUI fetch."""
    found: dict = {}

    def _add(raw: str, f: Path):
        entry = found.setdefault(_norm_fetch(raw), {"raw": raw, "files": set()})
        entry["files"].add(f.relative_to(_ROOT).as_posix())
        # A literal ending in "/" is ambiguous: a bare path with a trailing slash
        # ("/api/prompts/") or a concatenation prefix
        # (fetch("/api/video/file/" + encodeURIComponent(n))). Accept either.
        entry["trailing_slash"] = raw.split("?")[0].endswith("/")

    for f in sorted(_STATIC.rglob("*.js")):
        txt = f.read_text(encoding="utf-8")
        for m in _FETCH_TMPL.finditer(txt):
            if m.group(1).startswith("/"):
                _add(m.group(1), f)
        for m in _FETCH_STR.finditer(txt):
            _add(m.group(2), f)
    return found


def test_matcher_rejects_an_undeclared_path():
    """Negative case: the matcher must be able to FAIL, or it proves nothing."""
    routes = _declared_routes()
    assert routes, "no routes were collected - the scanner is broken"
    assert not any(_matches("/v1/definitely-not-a-real-route", r) for r in routes)
    # ...and it must still accept a route that really is declared.
    assert any(_matches("/api/plugins", r) for r in routes)


def test_matcher_handles_params_and_interpolation():
    assert _matches(_norm_fetch("/api/coder/sessions/${id}/undo"),
                    _norm_route("/api/coder/sessions/{session_id}/undo"))
    assert _matches(_norm_fetch("/api/${media}/workflows"),
                    _norm_route(r"/api/{media}/workflows"))
    assert not _matches(_norm_fetch("/api/coder/sessions/${id}/nope"),
                        _norm_route("/api/coder/sessions/{session_id}/undo"))


def test_static_mount_matcher_rejects_a_nonexistent_asset():
    """Negative case: the matcher must be able to FAIL, or it proves nothing."""
    assert not _served_by_static_mount("/i18n/${id}.nope")
    assert not _served_by_static_mount("/does-not-exist/${id}.json")
    assert not _served_by_static_mount("/i18n/de.json")  # no interpolation at all
    # ...and it must still accept a fetch the static mount really does serve.
    assert _served_by_static_mount("/i18n/${id}.json")


def test_i18n_catalog_fetch_is_seen_by_the_static_mount_matcher():
    """Guards the scanner itself: if this fallback silently stopped matching,
    the German-language catalog fetch would read as an undeclared route."""
    assert _served_by_static_mount("/i18n/${id}.json")


def test_every_gui_fetch_hits_a_declared_route():
    routes = _declared_routes()
    fetches = _js_fetches()
    assert fetches, "no fetch() calls were found - the scanner is broken"

    broken = []
    for path, info in sorted(fetches.items()):
        candidates = [path]
        if info["trailing_slash"]:
            candidates.append(path.rstrip("/") + "/" + _WILD)
        if any(_matches(c, r) for c in candidates for r in routes):
            continue
        if _served_by_static_mount(info["raw"]):
            continue
        broken.append(f"  {info['raw']!r} -> no such route "
                      f"(called from {', '.join(sorted(info['files']))})")
    assert not broken, (
        "GUI JavaScript fetches paths that no Python route declares:\n"
        + "\n".join(broken))


@pytest.mark.parametrize("path", ["/api/plugins", "/api/models", "/v1/chat/completions"])
def test_known_live_routes_are_seen_by_the_scanner(path):
    """Guards the scanner itself: if the route regex silently stopped matching,
    every fetch would 'pass' vacuously and this test would go quiet."""
    assert any(_matches(path, r) for r in _declared_routes())
