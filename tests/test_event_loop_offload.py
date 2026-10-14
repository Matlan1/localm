# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/check_event_loop_offload.py: the sweep for `async def` route handlers
that block the event loop.

Behavioural tests (test_embedder_event_loop_freeze.py, test_image_proxy.py,
test_bug_report_endpoint.py) each pin ONE known route. Neither coverage nor ruff
can find the next one: the offending line executes, it is simply executing on
the wrong thread. This is the sweep that looks.

WHAT THESE TESTS PIN:

1. It FIRES. Four synthetic trees, one per mechanism the analyzer claims to
   detect.
2. It does NOT fire on correctly-offloaded code, or on a plain `def` handler
   (which FastAPI already runs in a worker thread), or on a short-hold lock.
   About 180 of this repo's 208 async routes flag if "any lock" counts, so the
   bounded/unbounded split is load-bearing.
3. The UNCONFIRMED marking survives. The analyzer resolves `x.method(...)` on
   an unknown receiver by unique method name (`sidecar.rename` is
   pathlib.Path.rename, and matches the CLI's `rename` command). Marked
   findings must never gate.
4. The real tree passes --gate. This is the recurrence guard, and the only
   assertion here that a future change can trip.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_event_loop_offload", REPO / "scripts" / "check_event_loop_offload.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    """Build a throwaway localm/ package from {relative path: source}."""
    for rel, src in files.items():
        path = tmp_path / "localm" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(src, encoding="utf-8")
    return tmp_path


def _routes(mod, root: Path) -> dict[str, list]:
    out: dict[str, list] = {}
    for finding in mod.Analyzer(root).findings():
        out.setdefault(finding.route, []).append(finding)
    return out


# --------------------------------------------------------------------------- #
#  1. IT FIRES - one synthetic tree per detected mechanism                     #
# --------------------------------------------------------------------------- #

_HELPERS = '''
import socket
import threading

_LOCK = threading.RLock()
_SHORT_LOCK = threading.RLock()
_CACHE = {}


class Isolated:
    def __init__(self):
        self._runner.spawn_and_load({})


def loaded_dim():
    """Does NOT trigger a load - safe for a cheap status probe."""
    with _LOCK:
        return 384


def get_embedder():
    with _LOCK:
        return Isolated()


def cached(key):
    with _SHORT_LOCK:
        return _CACHE.get(key)


def fetch(url):
    return socket.getaddrinfo(url, None)
'''


def test_fires_on_an_unbounded_lock_reached_from_an_async_handler(tmp_path):
    mod = _load()
    root = _tree(tmp_path, {
        "helpers.py": _HELPERS,
        "routes.py": '''
from localm.helpers import loaded_dim

def register(app):
    @app.get("/api/status")
    async def status():
        return {"dim": loaded_dim()}
''',
    })
    found = _routes(mod, root)
    assert "GET /api/status" in found, (
        "the analyzer did not report an async handler calling loaded_dim(), "
        "which takes a lock held across a spawn - that is the exact shape of "
        "the defect measured at 47s on POST /api/embedding/warmup")
    kinds = {f.kind for f in found["GET /api/status"]}
    assert "unbounded-lock" in kinds, kinds
    assert any(f.gating for f in found["GET /api/status"])


def test_fires_on_a_network_call_reached_from_an_async_handler(tmp_path):
    mod = _load()
    root = _tree(tmp_path, {
        "helpers.py": _HELPERS,
        "routes.py": '''
from localm.helpers import fetch

def register(app):
    @app.get("/api/proxy")
    async def proxy(url: str):
        return fetch(url)
''',
    })
    found = _routes(mod, root)
    assert "GET /api/proxy" in found
    assert {f.kind for f in found["GET /api/proxy"]} == {"network"}
    assert any(f.gating for f in found["GET /api/proxy"])


def test_fires_through_a_helper_chain_a_grep_would_not_follow(tmp_path):
    """The reason this is not a grep: in every real instance the blocking call
    was two or three hops below the handler, in a module the handler's own file
    does not mention."""
    mod = _load()
    root = _tree(tmp_path, {
        "helpers.py": _HELPERS,
        "middle.py": '''
from localm.helpers import loaded_dim

def describe():
    return {"dim": loaded_dim()}
''',
        "routes.py": '''
from localm.middle import describe

def register(app):
    @app.get("/api/deep")
    async def deep():
        return describe()
''',
    })
    found = _routes(mod, root)
    assert "GET /api/deep" in found
    assert any(f.kind == "unbounded-lock" for f in found["GET /api/deep"])


def test_a_bounded_lock_is_reported_but_does_not_gate(tmp_path):
    """A lock whose longest hold is a short subprocess is real and is worth
    reporting, and it is a different severity from one held across a model
    load. Conflating them is what would make this gate unusable: ~46 routes in
    the real tree reach sessions._LOCK through the auth path alone."""
    mod = _load()
    root = _tree(tmp_path, {
        "helpers.py": '''
import subprocess
import threading

_FILE_LOCK = threading.RLock()


def save(rec):
    with _FILE_LOCK:
        subprocess.run(["icacls", "x"], capture_output=True, check=False)
''',
        "routes.py": '''
from localm.helpers import save

def register(app):
    @app.post("/api/save")
    async def do_save():
        return save({})
''',
    })
    found = _routes(mod, root)
    assert "POST /api/save" in found
    assert {f.kind for f in found["POST /api/save"]} == {"bounded-lock"}
    assert not any(f.gating for f in found["POST /api/save"]), (
        "a bounded lock must not fail the gate - a gate that fires on ~46 "
        "routes at once is one people switch off")


# --------------------------------------------------------------------------- #
#  2. IT DOES NOT FIRE on the legal shapes                                     #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("offload,extra", [
    ("await run_in_threadpool_bounded(_work, timeout=5.0)", ""),
    ("await run_in_threadpool(_work)", ""),
    ("await loop.run_in_executor(None, _work)", "loop = None\n        "),
])
def test_does_not_fire_when_the_work_is_handed_to_a_worker_thread(tmp_path, offload, extra):
    mod = _load()
    root = _tree(tmp_path, {
        "helpers.py": _HELPERS,
        "routes.py": '''
from localm.helpers import loaded_dim

def register(app):
    @app.get("/api/status")
    async def status():
        def _work():
            return loaded_dim()
        %s%s
''' % (extra, offload),
    })
    assert _routes(mod, root) == {}, (
        "reported a handler that hands the blocking call to a worker thread - "
        "the false-positive direction, and the one that gets a gate disabled")


def test_does_not_fire_on_a_plain_def_handler(tmp_path):
    """FastAPI runs a non-async handler in a worker thread itself, so a
    blocking call there is not a defect and must not be reported."""
    mod = _load()
    root = _tree(tmp_path, {
        "helpers.py": _HELPERS,
        "routes.py": '''
from localm.helpers import loaded_dim

def register(app):
    @app.get("/api/status")
    def status():
        return {"dim": loaded_dim()}
''',
    })
    assert _routes(mod, root) == {}


def test_does_not_fire_on_a_short_hold_lock(tmp_path):
    """A dict guard held for microseconds. Every authenticated request in the
    real tree takes several; flagging them is the difference between a 5-row
    report and a 180-row one."""
    mod = _load()
    root = _tree(tmp_path, {
        "helpers.py": _HELPERS,
        "routes.py": '''
from localm.helpers import cached

def register(app):
    @app.get("/api/cached")
    async def read_cache(key: str):
        return cached(key)
''',
    })
    assert _routes(mod, root) == {}


def test_a_job_callback_is_not_walked(tmp_path):
    """The body of a function handed to jobs.start_fn runs on a job thread, so
    the analyzer does not walk into it. POST /api/embedding/warmup calls
    get_embedder() inside exactly such a closure, correctly."""
    mod = _load()
    root = _tree(tmp_path, {
        "helpers.py": _HELPERS,
        "routes.py": '''
from localm.helpers import get_embedder

def register(app, jobs):
    @app.post("/api/warmup")
    async def warmup():
        def _warm(job):
            return get_embedder()
        return {"job_id": jobs.start_fn("warmup", _warm).id}
''',
    })
    assert _routes(mod, root) == {}


# --------------------------------------------------------------------------- #
#  3. THE HEURISTIC STAYS MARKED                                               #
# --------------------------------------------------------------------------- #

def test_an_unknown_receiver_is_reported_as_unconfirmed_and_never_gates(tmp_path):
    """`x.fetch(...)` where x is a local of unknown type. Resolved by unique
    method name, because dropping that lost a real finding - and marked, because
    trusting it invented one."""
    mod = _load()
    root = _tree(tmp_path, {
        "helpers.py": _HELPERS,
        "routes.py": '''
def register(app):
    @app.get("/api/guess")
    async def guess(thing):
        return thing.fetch("example.invalid")
''',
    })
    found = _routes(mod, root)
    assert "GET /api/guess" in found, "the heuristic edge was dropped entirely"
    assert not any(f.confident for f in found["GET /api/guess"])
    assert not any(f.gating for f in found["GET /api/guess"]), (
        "an UNCONFIRMED finding must never fail the gate: it is a guess about "
        "a receiver's type, and this repo has a measured false positive of "
        "exactly that shape")


# --------------------------------------------------------------------------- #
#  4. THE REAL TREE - the recurrence guard                                     #
# --------------------------------------------------------------------------- #

def test_the_real_tree_has_no_unoffloaded_blocking_route():
    """The gate. A new `async def` handler that reaches a network call or a
    lock held across a model load fails HERE.

    An exception is landed by adding an entry to ALLOWED in the script."""
    mod = _load()
    rows = mod._shortest(mod.Analyzer(REPO).findings())
    gating = [f for f in rows if f.gating]
    # Built eagerly, not as a lazy expression after the comma, so the message
    # names the call chain rather than only the route.
    report = []
    for f in gating:
        report.append("%s blocks the event loop via %s (%s)"
                      % (f.route, f.sink, f.sink_at))
        report.extend("      " + step for step in f.chain)
    assert not gating, "\n".join(report)


def test_an_unparseable_file_is_surfaced_and_fails_the_gate(tmp_path):
    """A file the analyzer cannot parse contributes nothing, so it must be
    reported and must fail the gate rather than counting as clean."""
    mod = _load()
    root = _tree(tmp_path, {
        "helpers.py": _HELPERS,
        # `this is not python` would PARSE as an `is not` comparison; this does not.
        "broken.py": "def register(app:\n",
        "routes.py": '''
from localm.helpers import loaded_dim

def register(app):
    @app.get("/api/status")
    async def status():
        return {"dim": loaded_dim()}
''',
    })
    analyzer = mod.Analyzer(root)
    assert any("broken.py" in item for item in analyzer.unparseable)
    assert mod.main(["--gate", "--root", str(root)]) == 1


def test_the_real_tree_parses_completely():
    """The gate above is only meaningful if every file was actually read."""
    mod = _load()
    analyzer = mod.Analyzer(REPO)
    assert analyzer.unparseable == []


def test_the_two_worked_examples_are_still_recognised_shapes():
    """A guard against the gate passing for the WRONG reason - an analyzer that
    stopped seeing these routes at all would also report zero findings. Both
    are now offloaded, so they must be KNOWN as routes and CLEAN as findings."""
    mod = _load()
    analyzer = mod.Analyzer(REPO)
    routes = {m + " " + p
              for mod_ in analyzer.mods.values() for _q, m, p in mod_.routes}
    assert "POST /api/embedding/warmup" in routes
    assert "GET /api/image-proxy" in routes
    assert "GET /api/rag/embedding" in routes
    blocked = {f.route for f in analyzer.findings() if f.gating}
    assert blocked.isdisjoint(
        {"POST /api/embedding/warmup", "GET /api/image-proxy",
         "GET /api/rag/embedding"})


def test_the_embedder_load_lock_is_classified_unbounded():
    """The classification the whole report rests on. embedder._LOCK is held
    across an IsolatedEmbedder construction - a process spawn plus a native
    load, ceiling 300s - and if that stopped being computed as UNBOUNDED the
    gate would silently stop covering the defect it was built for."""
    mod = _load()
    analyzer = mod.Analyzer(REPO)
    kind = analyzer.lock_kind.get("localm.inference.embedder._LOCK")
    assert kind is not None, "embedder._LOCK was not classified at all"
    assert kind[0] == "unbounded", kind
