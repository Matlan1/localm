# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.inference.http_server.rekey_loaded_model: re-keys every in-memory
record of a loaded model's identity after its registry entry is renamed, so a
still-loaded/serving engine is not orphaned under its old name. See
tests/test_driving_engine.py for the sibling http_server module-state test
pattern this reuses.

The second half of this file covers what happens when NOBODY re-keys, which is
not a hypothetical: `localm rename` runs in a separate process and physically
cannot reach into a running server's memory, so it leaves the engine map keyed
on the old name while the registry holds the new one. Every name-keyed guard on
the remove route then missed, and the route deleted a loaded model's GGUF.
Those tests use a REAL file inside a REAL models dir, because the guard turns
on `resolve_deletion_target`, and a fixture with a fictional path like
"x/a.gguf" can never make it return anything - it would execute the guard and
be structurally unable to fail on the bug.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import localm.config as config
import localm.model_manager as model_manager
from localm.inference import http_server as hs


class _FakeEngine:
    def __init__(self, name):
        self.display_name = name
        self.loaded = True


def _reset():
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._last_activity_per_model.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._last_active_model_name = None
    hs._engine = None


def test_rekey_moves_engine_and_updates_display_name():
    _reset()
    eng = _FakeEngine("old")
    hs._engines["old"] = eng
    hs._engines_lru.append("old")
    hs._active_model_name = "old"
    hs._inference_sems["old"] = object()
    hs._last_activity_per_model["old"] = 123.0

    assert hs.rekey_loaded_model("old", "new") is True

    assert "old" not in hs._engines
    assert hs._engines["new"] is eng
    assert eng.display_name == "new", "active_model() reads engine.display_name directly"
    assert hs._engines_lru == ["new"]
    assert hs._active_model_name == "new"
    assert "old" not in hs._inference_sems
    assert "new" in hs._inference_sems
    assert hs._last_activity_per_model.get("new") == 123.0
    assert "old" not in hs._last_activity_per_model
    _reset()


def test_rekey_is_a_noop_when_the_old_name_is_not_loaded():
    _reset()
    assert hs.rekey_loaded_model("nowhere", "elsewhere") is False
    assert hs._engines == {}
    assert hs._active_model_name is None
    _reset()


def test_rekey_also_moves_the_startup_and_last_active_pointers():
    """Found by the live repro, not by reasoning: after renaming the model a
    server was STARTED with, GET /v1/models still listed a row for the old
    name, because list_models adds _default_model_name whenever the registry
    lacks it. The same stale pointer also makes switch_engine's registration
    check accept a name the registry no longer has, and
    _resolve_unnamed_model_name falls back to _last_active_model_name after an
    eviction. All three are records of a loaded model's identity, so all three
    move."""
    _reset()
    eng = _FakeEngine("old")
    hs._engines["old"] = eng
    hs._default_model_name = "old"
    hs._last_active_model_name = "old"

    assert hs.rekey_loaded_model("old", "new") is True

    assert hs._default_model_name == "new"
    assert hs._last_active_model_name == "new"
    _reset()


def test_rekey_leaves_another_models_pointers_alone():
    """The pointers move only when they NAME the renamed model - renaming a
    background model must not steal the startup identity of a different one."""
    _reset()
    hs._engines["bg"] = _FakeEngine("bg")
    hs._default_model_name = "startup-one"
    hs._last_active_model_name = "startup-one"

    assert hs.rekey_loaded_model("bg", "bg-renamed") is True

    assert hs._default_model_name == "startup-one"
    assert hs._last_active_model_name == "startup-one"
    _reset()


def test_rekey_does_not_touch_active_model_name_for_a_background_model():
    # Renaming a BACKGROUND-loaded (not active) model must not steal the
    # active-model pointer or disturb the actually-active engine's entry.
    _reset()
    active_eng = _FakeEngine("active-one")
    bg_eng = _FakeEngine("bg-one")
    hs._engines["active-one"] = active_eng
    hs._engines["bg-one"] = bg_eng
    hs._engines_lru.extend(["active-one", "bg-one"])
    hs._active_model_name = "active-one"

    assert hs.rekey_loaded_model("bg-one", "bg-renamed") is True

    assert hs._active_model_name == "active-one"
    assert hs._engines["bg-renamed"] is bg_eng
    assert bg_eng.display_name == "bg-renamed"
    assert hs._engines["active-one"] is active_eng
    _reset()


def test_rekey_fixes_the_stale_active_model_guard_hazard():
    """The concrete hazard this function exists to close: active_model()
    reads _engine.display_name, and the GUI's remove-model guard is exactly
    `req.model == active_model()`. Without the rekey, renaming the active
    model would leave that comparison checking the NEW registry name against
    the engine's stale OLD display_name, so it would never match - and the
    GUI could delete the file out from under the model still serving
    requests. Reproduces active_model()'s own read directly (it is a closure
    built per-app, but it always reduces to exactly this)."""
    _reset()
    eng = _FakeEngine("old")
    hs._engine = eng
    hs._engines["old"] = eng
    hs._engines_lru.append("old")
    hs._active_model_name = "old"

    assert hs.rekey_loaded_model("old", "new") is True

    assert hs._engine.display_name == "new"
    _reset()


# ---------------------------------------------------------------------------
#  The other half: when the re-key CANNOT happen (a rename from another
#  process), the file must still not be deletable. Guard by FILE IDENTITY.
# ---------------------------------------------------------------------------


class _FileEngine:
    """An engine holding a real file, the way a loaded one does."""

    def __init__(self, name, path, loaded=True):
        self.display_name = name
        self.model_path = str(path)
        self.loaded = loaded


@pytest.fixture
def models_home(tmp_path, monkeypatch):
    """A throwaway data dir whose models root every call site resolves through.

    model_manager.MODELS_DIR is what is_owned_model_path (and therefore
    resolve_deletion_target) reads; config.MODELS_DIR is pinned to the same
    directory so nothing can silently answer against the session's real home
    and make these tests pass vacuously. Same reasoning as
    tests/test_cli_rm_prompt.py's `home` fixture, which documents why leaving
    the second one unpinned once made a whole file pass against the very bug it
    existed to catch.
    """
    home = tmp_path / ".localm"
    models = home / "models"
    models.mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(model_manager, "MODELS_DIR", models)
    monkeypatch.setattr(config, "HOME_DIR", home)
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(config, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(config, "REGISTRY_FILE", home / "registry.json")
    return home


def _make_model_file(models_home, filename="m.gguf", data=b"GGUF" + b"\0" * 64):
    path = models_home / "models" / filename
    path.write_bytes(data)
    return path


def _register(models_home, entries):
    (models_home / "registry.json").write_text(json.dumps(entries), encoding="utf-8")


@pytest.fixture
def gui_client(monkeypatch):
    """The REAL GUI router, and a start_cli that runs the REAL removal.

    Substituting an in-process ``remove_model(name)`` for the spawned
    ``localm rm <name> --yes`` is faithful: that subprocess does exactly this
    and nothing else (``--yes`` skips the only other step, the prompt). It is
    also the point of the harness - the property under test is whether the
    user's FILE survives, and a start_cli that merely records its arguments
    could never observe a deletion, so it could never fail on the bug either.
    """
    from localm.plugins.gui.web import attach_gui

    app = FastAPI()
    started = []

    class _FakeJob:
        id = "job-test"

    def fake_start_cli(self, kind, cli_args, **kw):
        started.append(list(cli_args))
        args = list(cli_args)
        if args and args[0] == "rm":
            model_manager.remove_model(args[1])
        return _FakeJob()

    monkeypatch.setattr("localm.plugins.gui.jobs.JobManager.start_cli", fake_start_cli)

    async def switch_model(name):
        return {"status": "loaded", "model": name}

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model,
               # No engine is ACTIVE in these tests: the whole point is the
               # loaded-but-not-active, wrongly-named case the two name-keyed
               # guards were blind to. An active_model() that named something
               # would let the FIRST guard answer, and the test would pass
               # without ever reaching the one under test.
               active_model=lambda: "")
    return app, started


def test_remove_refuses_a_model_whose_file_a_live_engine_still_holds(
        models_home, gui_client):
    """THE regression: the shape `localm rename` leaves behind.

    The registry has been renamed old-name -> new-name by another process, so
    the running server's engine map is still keyed on "old-name" while the
    request names "new-name". Both name-keyed guards miss. Without the
    file-identity guard the route removes it and the GGUF is gone for good.
    """
    app, started = gui_client
    gguf = _make_model_file(models_home)
    _register(models_home, {"new-name": {"path": str(gguf), "source": "local"}})

    hs._engines.clear()
    hs._engines_lru.clear()
    hs._engine = None
    hs._engines["old-name"] = _FileEngine("old-name", gguf)
    try:
        with TestClient(app) as client:
            r = client.post("/api/models/remove", json={"model": "new-name"})
        # The FILE first, deliberately. A status-code assertion placed ahead of
        # it would short-circuit on a regression and report only "409 != 200",
        # which is a statement about the guard, not about the user's data. The
        # property being defended is that the GGUF is still there.
        assert gguf.exists(), (
            "the user's model file was DELETED out from under a live engine")
        assert started == [], "the removal job must not even be started"
        assert r.status_code == 409, (
            "a file a live engine is serving must never be deletable, whatever "
            f"name the registry or the engine currently uses: {r.text}")
        assert "old-name" in r.json()["detail"], (
            "the refusal must name the key it is actually loaded under, or the "
            "user cannot work out what to unload")
        assert "new-name" in config.load_registry()
    finally:
        hs._engines.clear()
        hs._engines_lru.clear()


def test_remove_still_deletes_an_unloaded_model_for_real(models_home, gui_client):
    """The control for the test above, and it has to be here.

    If this harness could not delete a file, the assertion that the file
    survives would hold for reasons that have nothing to do with the guard.
    Nothing is loaded, so the removal proceeds and the GGUF really goes.
    """
    app, started = gui_client
    gguf = _make_model_file(models_home)
    _register(models_home, {"spare": {"path": str(gguf), "source": "local"}})

    hs._engines.clear()
    hs._engine = None
    with TestClient(app) as client:
        r = client.post("/api/models/remove", json={"model": "spare"})
    assert r.status_code == 200, r.text
    assert started == [["rm", "spare", "--yes"]]
    assert not gguf.exists(), (
        "the harness cannot observe a deletion, so the sibling test's "
        "'file survives' assertion would be vacuous")
    assert "spare" not in config.load_registry()


def test_remove_still_allows_an_alias_of_a_loaded_file(models_home, gui_client):
    """Not over-refusing: while a SECOND registered name points at the file,
    removing one of them keeps the bytes (remove_model returns early), so there
    is no data loss to guard against and the request must still succeed. A
    guard that refused here would break `localm alias` cleanup for every loaded
    model."""
    app, started = gui_client
    gguf = _make_model_file(models_home)
    _register(models_home, {
        "keeper": {"path": str(gguf), "source": "local"},
        "spare-name": {"path": str(gguf), "source": "local"},
    })

    hs._engines.clear()
    hs._engine = None
    hs._engines["keeper"] = _FileEngine("keeper", gguf)
    try:
        with TestClient(app) as client:
            r = client.post("/api/models/remove", json={"model": "spare-name"})
        assert r.status_code == 200, r.text
        assert started == [["rm", "spare-name", "--yes"]]
        assert gguf.exists(), "an alias removal must never delete the file"
        reg = config.load_registry()
        assert "keeper" in reg and "spare-name" not in reg
    finally:
        hs._engines.clear()


def test_guard_ignores_an_unloaded_engine_and_an_unowned_path(models_home):
    """Two ways the guard must answer None, both of which would otherwise make
    it refuse removals that destroy nothing."""
    gguf = _make_model_file(models_home)
    outside = models_home / "elsewhere.gguf"      # NOT under <data dir>/models
    outside.write_bytes(b"GGUF")
    _register(models_home, {
        "unloaded": {"path": str(gguf), "source": "local"},
        "external": {"path": str(outside), "source": "local"},
    })

    hs._engines.clear()
    hs._engine = None
    try:
        hs._engines["unloaded"] = _FileEngine("unloaded", gguf, loaded=False)
        assert hs.loaded_engine_holding_model_file("unloaded") is None, (
            "an engine that is not loaded is not holding the file open")

        hs._engines.clear()
        hs._engines["external"] = _FileEngine("external", outside)
        assert hs.loaded_engine_holding_model_file("external") is None, (
            "remove_model never deletes a path outside <data dir>/models, so "
            "there is nothing to refuse")
    finally:
        hs._engines.clear()


# ---------------------------------------------------------------------------
#  FAIL CLOSED. The question is not "do these paths compare equal" but "can I
#  establish nothing live is using this file". Every input where the comparison
#  cannot be MADE was answered "nothing holds it, delete away".
#
#  None of these cases can be expressed by the fixtures above: they all build a
#  real, resolvable temp file, so the comparison always succeeds and the error
#  paths are unreachable. That is precisely why they slipped past six
#  fires-controls and a green suite.
# ---------------------------------------------------------------------------


class _UnresolvableEngine:
    """A loaded engine whose path will not resolve - a UNC share that is
    momentarily unreachable, a permission error, an embedded NUL."""

    def __init__(self, name, exc=OSError("network path not found")):
        self.display_name = name
        self.loaded = True
        self._exc = exc

    @property
    def model_path(self):
        return "//unreachable-share/models/m.gguf"


def _raise_on_resolve(monkeypatch, bad: str, exc=OSError("unreachable")):
    """Make Path.resolve() raise for one specific path and behave normally for
    every other, so the test does not have to disable resolution process-wide
    (which would take the target side down with the engine side).

    Compared through Path + normcase, NOT as the raw string handed in: on
    Windows ``str(Path("//share/x"))`` comes back with BACKSLASHES, so a raw
    equality test silently never matches. The first version of this helper did
    exactly that - the fault was never injected, nothing refused, and the file
    was deleted. The test caught it, but only because it asserts on the FILE.
    """
    import os as _os
    from pathlib import Path as _P
    real = _P.resolve
    want = _os.path.normcase(str(_P(bad)))

    def fake(self, *a, **kw):
        if _os.path.normcase(str(self)) == want:
            raise exc
        return real(self, *a, **kw)

    monkeypatch.setattr(_P, "resolve", fake)
    # Prove the injection took. A patch that silently fails to match looks
    # exactly like a guard that correctly found nothing to refuse.
    try:
        _P(bad).resolve()
    except type(exc):
        pass
    else:
        raise AssertionError(f"the resolve fault was not injected for {bad!r}")


def test_remove_refuses_when_a_loaded_engines_path_will_not_resolve(
        models_home, gui_client, monkeypatch):
    """THE fail-open: `except (OSError, ValueError): continue` skipped a LOADED
    engine and let the loop fall through to "nothing holds this file". A model
    served off a momentarily-unreachable UNC share therefore had its GGUF
    deleted out from under it - the same unrecoverable outcome the guard
    exists to prevent, reached through the error path."""
    app, started = gui_client
    gguf = _make_model_file(models_home)
    _register(models_home, {"victim": {"path": str(gguf), "source": "local"}})

    bad = "//unreachable-share/models/m.gguf"
    _raise_on_resolve(monkeypatch, bad)

    hs._engines.clear()
    hs._engine = None
    hs._engines["serving"] = _UnresolvableEngine("serving")
    try:
        with TestClient(app) as client:
            r = client.post("/api/models/remove", json={"model": "victim"})
        assert gguf.exists(), (
            "a file was deleted while an engine that could not be ruled out "
            "was loaded")
        assert started == []
        assert r.status_code == 409, r.text
        detail = r.json()["detail"]
        assert "could not be resolved" in detail, (
            f"the refusal must say WHY, or a user hunts a phantom in-use "
            f"model: {detail}")
        assert "serving" in detail
    finally:
        hs._engines.clear()


def test_remove_refuses_when_a_loaded_engine_has_no_recorded_path(
        models_home, gui_client):
    """The second fail-open, three lines up from the first: `if not mpath:
    continue`. An engine with no recorded model_path has not been shown to be
    free of this file either."""
    app, started = gui_client
    gguf = _make_model_file(models_home)
    _register(models_home, {"victim": {"path": str(gguf), "source": "local"}})

    pathless = _FileEngine("pathless", gguf)
    pathless.model_path = ""

    hs._engines.clear()
    hs._engine = None
    hs._engines["pathless"] = pathless
    try:
        with TestClient(app) as client:
            r = client.post("/api/models/remove", json={"model": "victim"})
        assert gguf.exists(), (
            "a file was deleted while a loaded engine with no recorded path "
            "was resident - it was never shown to be free of this file")
        assert started == []
        assert r.status_code == 409, r.text
        assert "no recorded model file path" in r.json()["detail"]
    finally:
        hs._engines.clear()


def test_remove_refuses_when_the_models_own_path_will_not_resolve(
        models_home, gui_client, monkeypatch):
    """The third case, and it is NOT a data-loss one - stated precisely
    because the other two are and it would be easy to bundle them.

    MEASURED against the previous code: an unresolvable registry path made
    find_aliases_by_path raise (it resolves the path it is handed and catches
    only for SIBLING entries), so the guard blew up before reaching
    resolve_deletion_target and the route answered 500. Nothing was deleted.
    So the collapse inside resolve_deletion_target - which returns None for
    "unresolvable" exactly as it does for "outside the models dir" and
    "already gone" - was LATENT, not live.

    Two reasons it is fixed anyway. A guard whose job is to answer a question
    calmly should not crash the request (rule 5: a failure gets reported at
    its own altitude, not as a stack trace). And the latent hole goes LIVE the
    moment anything reorders those two calls - which this very change does,
    moving the resolve probe ahead of find_aliases_by_path.
    """
    app, started = gui_client
    gguf = _make_model_file(models_home)
    _register(models_home, {"victim": {"path": str(gguf), "source": "local"}})

    _raise_on_resolve(monkeypatch, str(gguf))

    hs._engines.clear()
    hs._engine = None
    hs._engines["serving"] = _FileEngine("serving", models_home / "models" / "other.gguf")
    try:
        with TestClient(app) as client:
            r = client.post("/api/models/remove", json={"model": "victim"})
        assert gguf.exists()
        assert started == []
        assert r.status_code == 409, r.text
        assert "registered file path could not be resolved" in r.json()["detail"]
    finally:
        hs._engines.clear()


def test_an_unresolvable_path_with_nothing_loaded_is_not_a_refusal(
        models_home, monkeypatch):
    """Failing closed must not become failing always. With NOTHING resident,
    nothing can be holding the file, so the answer is a certain no however
    unresolvable the paths are - the user can still tidy their library on a box
    with a flaky network share.

    Asserted on the guard directly rather than through the route: with the
    target unresolvable, remove_model itself raises out of find_aliases_by_path,
    and this harness runs that removal synchronously where production spawns it
    as a job. Driving the route here would measure the harness's own shape
    rather than the property.
    """
    gguf = _make_model_file(models_home)
    _register(models_home, {"spare": {"path": str(gguf), "source": "local"}})
    _raise_on_resolve(monkeypatch, str(gguf))

    hs._engines.clear()
    hs._engine = None
    assert hs.loaded_engine_holding_model_file("spare") is None


def test_a_proven_holder_is_named_ahead_of_an_unresolvable_one(models_home):
    """Both refuse, but only one of them can say truthfully WHICH engine has
    the file, so the scan completes rather than returning the first unknown it
    trips over."""
    gguf = _make_model_file(models_home)
    _register(models_home, {"victim": {"path": str(gguf), "source": "local"}})

    hs._engines.clear()
    hs._engine = None
    hs._engines["broken"] = _UnresolvableEngine("broken")
    hs._engines["the-real-holder"] = _FileEngine("the-real-holder", gguf)
    try:
        hold = hs.loaded_engine_holding_model_file("victim")
        assert hold is not None
        assert hold.key == "the-real-holder"
        assert hold.reason is None, "a proven holder is not a cautious guess"
    finally:
        hs._engines.clear()


def test_an_unloaded_engine_with_a_broken_path_is_not_a_refusal(models_home):
    """Only LOADED engines can hold a file open, so a broken path on an
    unloaded one is not an unknown - it is irrelevant."""
    gguf = _make_model_file(models_home)
    _register(models_home, {"spare": {"path": str(gguf), "source": "local"}})

    dead = _UnresolvableEngine("dead")
    dead.loaded = False

    hs._engines.clear()
    hs._engine = None
    hs._engines["dead"] = dead
    try:
        assert hs.loaded_engine_holding_model_file("spare") is None
    finally:
        hs._engines.clear()


def test_guard_sees_the_startup_engine_that_is_not_in_the_engine_map(models_home):
    """`localm serve <path.gguf>` can leave the running engine as the module
    singleton without an _engines entry. It is holding the file exactly as
    hard, so a scan of _engines alone would miss it."""
    gguf = _make_model_file(models_home)
    _register(models_home, {"served": {"path": str(gguf), "source": "local"}})

    hs._engines.clear()
    hs._engine = _FileEngine("started-as-this", gguf)
    try:
        assert hs.loaded_engine_holding_model_file("served").key == "started-as-this"
    finally:
        hs._engine = None
        hs._engines.clear()


# ---------------------------------------------------------------------------
#  Every test above proves the POLICY. This one proves the WIRING: that
#  loaded_engine_holding_model_file answers with registry.engine_holding_model_file's
#  OWN result rather than an independently maintained copy of the same logic.
#  A fixture built from real files can never distinguish those two - both
#  implementations agree on every real input today by construction, so agreement
#  proves nothing about whether one is still a second, driftable copy of the
#  other. Forcing the registry policy to answer with a fabricated result no
#  inline computation could produce is what makes the two distinguishable.
# ---------------------------------------------------------------------------


def test_loaded_engine_holding_model_file_delegates_to_registry_policy(
        models_home, monkeypatch):
    """The registry policy's answer must be the wrapper's answer, verbatim."""
    import localm.model_manager.registry as registry_mod

    gguf = _make_model_file(models_home)
    _register(models_home, {"victim": {"path": str(gguf), "source": "local"}})

    sentinel = registry_mod.ModelFileHold(
        "SENTINEL-FROM-REGISTRY-POLICY",
        "a marker no inline computation over real paths could produce")
    captured = {}

    def fake_engine_holding_model_file(model, reg, candidates):
        captured["model"] = model
        captured["candidates"] = list(candidates)
        return sentinel

    monkeypatch.setattr(registry_mod, "engine_holding_model_file",
                        fake_engine_holding_model_file)

    hs._engines.clear()
    hs._engine = None
    hs._engines["holder"] = _FileEngine("holder", gguf)
    try:
        result = hs.loaded_engine_holding_model_file("victim")
    finally:
        hs._engines.clear()

    assert result is sentinel, (
        "loaded_engine_holding_model_file must return the registry policy's "
        "own object, not a value it computed independently - a copy that "
        "merely agrees with the policy today can silently drift from it "
        "tomorrow, which is the defect this test guards against")
    assert captured["model"] == "victim"
    assert captured["candidates"] == [("holder", str(gguf))], (
        "the wrapper's only remaining job is turning this process's "
        "residents into (key, model_path) pairs for the shared policy")


# ---------------------------------------------------------------------------
#  POST /v1/models/rename - the always-present route the CLI drives, so a
#  rename from another process re-keys the live engine instead of stranding it.
# ---------------------------------------------------------------------------


_INSTANCE_TOKEN = "rename-instance-token-0123456789"


def _core_app(engine):
    """The real core app, carrying the per-instance attach token a local
    management client presents. Without it _origin_guard's open-mode gate
    refuses every unsafe method with 403 before the route is ever reached -
    which is what a management route on this surface really faces, and what a
    test that patched `requests` would never see."""
    app = hs.create_app(engine)
    app.state.instance_token = _INSTANCE_TOKEN
    return app


_AUTH = {"Authorization": f"Bearer {_INSTANCE_TOKEN}"}


def test_v1_rename_route_moves_the_registry_and_rekeys_the_engine(models_home):
    """On the /v1 surface, not only the GUI's /api one: a headless
    `localm serve` has no GUI routes, and it is exactly as capable of holding a
    model open."""
    gguf = _make_model_file(models_home)
    _register(models_home, {"old-name": {"path": str(gguf), "source": "local"}})

    _reset()
    eng = _FileEngine("old-name", gguf)
    hs._engines["old-name"] = eng
    hs._engines_lru.append("old-name")
    hs._active_model_name = "old-name"
    hs._engine = eng
    try:
        with TestClient(_core_app(eng)) as client:
            r = client.post("/v1/models/rename", headers=_AUTH,
                            params={"model": "old-name", "new_name": "new-name"})
        assert r.status_code == 200, r.text
        assert r.json()["new_name"] == "new-name"
        assert "new-name" in config.load_registry()
        assert "old-name" not in config.load_registry()
        assert hs._engines["new-name"] is eng
        assert eng.display_name == "new-name"
        # The whole reason the CLI routes through here: after this, the file is
        # not deletable under EITHER name, because the engine map agrees with
        # the registry again.
        assert hs.loaded_engine_holding_model_file("new-name").key == "new-name"
    finally:
        _reset()


def test_v1_rename_route_reports_an_unregistered_model(models_home):
    _register(models_home, {})
    _reset()
    eng = _FileEngine("only", models_home / "models" / "nothing.gguf")
    try:
        with TestClient(_core_app(eng)) as client:
            r = client.post("/v1/models/rename", headers=_AUTH,
                            params={"model": "ghost", "new_name": "x"})
        assert r.status_code == 404, r.text
    finally:
        _reset()


# ---------------------------------------------------------------------------
#  `localm rename` drives that route when a server is up, and falls back
#  LOUDLY (never silently) when it cannot.
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._payload = payload if payload is not None else {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


def _patch_instance(monkeypatch, entry):
    from localm import instances
    monkeypatch.setattr(instances, "find_attachable", lambda *a, **kw: entry)
    monkeypatch.setattr(instances, "resolve_root_dir", lambda *a, **kw: "root")
    monkeypatch.delenv("LOCALM_URL", raising=False)


def test_cli_rename_asks_a_running_server_instead_of_renaming_behind_its_back(
        monkeypatch):
    """The fix for the data loss: this process cannot re-key another process's
    engine map, so it must not move the registry entry on its own while a
    server is up."""
    import requests

    from localm.cli import models as cli_models

    _patch_instance(monkeypatch, {"scheme": "http", "host": "127.0.0.1",
                                  "port": 1234, "token": "tok"})
    seen = {}

    def fake_post(url, **kw):
        seen["url"] = url
        seen["params"] = kw.get("params")
        return _FakeResponse(200, {"status": "renamed", "model": "a",
                                   "new_name": "b", "notes": ["note one"]})

    monkeypatch.setattr(requests, "post", fake_post)
    # The local rename must NOT also run: two renames would leave the second
    # reporting "Not found" and, worse, imply the registry move happened twice.
    monkeypatch.setattr("localm.model_manager.rename_model",
                        lambda *a: pytest.fail("renamed locally as well"))

    assert cli_models._rename_on_running_server("a", "b") is True
    assert seen["url"] == "http://127.0.0.1:1234/v1/models/rename"
    assert seen["params"] == {"model": "a", "new_name": "b"}


def test_cli_rename_falls_back_locally_when_no_server_is_running(monkeypatch):
    """The ordinary offline case must be untouched: no server, no network call,
    plain local rename."""
    import requests

    from localm.cli import models as cli_models

    _patch_instance(monkeypatch, None)
    monkeypatch.setattr(requests, "post",
                        lambda *a, **kw: pytest.fail("dialled with no server"))
    assert cli_models._rename_on_running_server("a", "b") is None


def test_cli_rename_does_not_redo_a_rename_that_a_lost_reply_hid(
        models_home, monkeypatch, capsys):
    """A read timeout means the request WAS sent and the answer was lost, so
    the rename may well have been applied. Renaming locally on top of that
    reports "Not found" for work that actually completed. The registry is the
    ground truth, so it gets read instead of guessed at."""
    import requests

    from localm.cli import models as cli_models

    _patch_instance(monkeypatch, {"port": 1234})
    # The state a server that DID apply the rename leaves behind.
    _register(models_home, {"b": {"path": "x/b.gguf", "source": "local"}})

    def timeout(*a, **kw):
        raise requests.ReadTimeout("timed out")

    monkeypatch.setattr(requests, "post", timeout)
    assert cli_models._rename_on_running_server("a", "b") is True
    assert "Renamed" in capsys.readouterr().out


def test_cli_rename_falls_back_when_a_lost_reply_hid_nothing(
        models_home, monkeypatch, capsys):
    """The other half of the same unknown: the registry shows the rename was
    NOT applied, so the user's rename is still to do. Fall back, and say why."""
    import requests

    from localm.cli import models as cli_models

    _patch_instance(monkeypatch, {"port": 1234})
    _register(models_home, {"a": {"path": "x/a.gguf", "source": "local"}})

    def timeout(*a, **kw):
        raise requests.ReadTimeout("timed out")

    monkeypatch.setattr(requests, "post", timeout)
    assert cli_models._rename_on_running_server("a", "b") is None
    assert "No reply" in capsys.readouterr().out


def test_cli_rename_stops_when_the_server_rejects_the_rename_itself(monkeypatch):
    """A 409 "name already taken" is the server's verdict on the rename, not a
    transport problem - retrying locally would move the entry the server just
    refused to move."""
    import requests

    from localm.cli import models as cli_models

    _patch_instance(monkeypatch, {"port": 1234})
    monkeypatch.setattr(requests, "post", lambda *a, **kw: _FakeResponse(
        409, {"detail": "Name already taken: b"}))
    assert cli_models._rename_on_running_server("a", "b") is False


def test_cli_rename_treats_a_missing_route_as_fall_back_not_as_a_refusal(
        monkeypatch, capsys):
    """A server too old to have /v1/models/rename answers 404 with FastAPI's
    bare "Not Found", which by STATUS alone is indistinguishable from this
    route's own "Model not registered". Reading it as a verdict would refuse a
    rename the user is entitled to, i.e. trade the data-loss bug for a broken
    feature. It must fall back and warn instead."""
    import requests

    from localm.cli import models as cli_models

    _patch_instance(monkeypatch, {"port": 1234})
    monkeypatch.setattr(requests, "post",
                        lambda *a, **kw: _FakeResponse(404, {"detail": "Not Found"}))
    assert cli_models._rename_on_running_server("a", "b") is None
    assert "old name" in capsys.readouterr().out


def test_cli_rename_warns_out_loud_before_falling_back_past_a_live_server(
        monkeypatch, capsys):
    """A 401 (a keyed server this CLI has no credential for) still leaves the
    user's rename to do, so it happens locally - but that strands the running
    server on the old name, and saying nothing about it is exactly the silent
    degradation that produced this bug. The warning must name the remedy."""
    import requests

    from localm.cli import models as cli_models

    _patch_instance(monkeypatch, {"port": 1234})
    monkeypatch.setattr(requests, "post", lambda *a, **kw: _FakeResponse(
        401, {"detail": "Unauthorized"}))
    assert cli_models._rename_on_running_server("a", "b") is None
    out = capsys.readouterr().out
    assert "401" in out
    assert "old name" in out
    assert "localm unload" in out


# ---------------------------------------------------------------------------
#  Real HTTP: the whole CLI path, through the real _origin_guard.
#
#  The four tests above patch `requests.post`, so they cannot see the gate that
#  actually stands in front of this route. It is not hypothetical: the first
#  version of the /v1 tests above passed a mocked client and 403'd the moment a
#  real one was used ("Open-mode management requires ..."). A rename that
#  cannot authenticate is a rename that silently falls back and strands the
#  server, which is the bug, so this has to be exercised for real.
# ---------------------------------------------------------------------------


def _start_real_server(engine):
    import asyncio
    import socket as _socket
    import threading
    import time as _time

    import uvicorn

    app = _core_app(engine)
    lsock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    lsock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    port = lsock.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
    thread = threading.Thread(target=lambda: asyncio.run(server.serve(sockets=[lsock])),
                              daemon=True)
    thread.start()
    deadline = _time.monotonic() + 10.0
    while not server.started and _time.monotonic() < deadline:
        _time.sleep(0.02)
    assert server.started, "uvicorn did not start"
    return app, port, server, thread


def test_cli_rename_over_real_http_rekeys_the_live_engine(models_home, monkeypatch):
    """End to end: `localm rename` against a REAL running server moves the
    registry entry AND re-keys the engine, so the file the engine is serving is
    not left orphaned under a name the registry no longer has.

    The server runs in a thread in this process, so its engine map IS the one
    asserted on here - the same reason the fix has to happen server-side.
    """
    from click.testing import CliRunner

    from localm import instances
    from localm.cli import main

    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_URL", raising=False)
    gguf = _make_model_file(models_home)
    _register(models_home, {"old-name": {"path": str(gguf), "source": "local"}})

    _reset()
    eng = _FileEngine("old-name", gguf)
    hs._engines["old-name"] = eng
    hs._engines_lru.append("old-name")
    hs._active_model_name = "old-name"
    hs._engine = eng
    app, port, server, thread = _start_real_server(eng)
    try:
        monkeypatch.setattr(instances, "find_attachable", lambda *a, **k: {
            "scheme": "http", "host": "127.0.0.1", "port": port,
            "token": _INSTANCE_TOKEN})
        monkeypatch.setattr(instances, "resolve_root_dir", lambda *a, **k: "root")
        res = CliRunner().invoke(main, ["rename", "old-name", "new-name"])

        assert res.exit_code == 0, res.output
        assert "403" not in res.output and "401" not in res.output, res.output
        assert "new-name" in config.load_registry()
        assert hs._engines.get("new-name") is eng, (
            "the server that holds the file must end up keyed on the new name")
        assert "old-name" not in hs._engines
        assert eng.display_name == "new-name"
        assert hs.loaded_engine_holding_model_file("new-name").key == "new-name"
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)
        _reset()
