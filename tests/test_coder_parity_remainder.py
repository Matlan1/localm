# SPDX-License-Identifier: AGPL-3.0-or-later
"""Web forms for the localcoder options that had none:

  --seed                  seed on CreateSessionRequest -> gen_kwargs
  --interactive-confirm   interactive_confirm -> Agent.always_confirm
  --episodes-archive      GET  /api/coder/episodes/archive
  --forget-episode        POST /api/coder/episodes/{id}/forget
  --restore-episode       POST /api/coder/episodes/{id}/restore
  --forget-episodes       DELETE /api/coder/episodes
  --consolidate-episodes  POST /api/coder/episodes/consolidate

The property most of these tests exist to pin is HONESTY, because the CLI draws
distinctions here that are easy to flatten and expensive to get wrong: an
unreadable archive is not an empty one, a restore that is immediately re-evicted
is not a plain success, and an erase that only half happened must never be
reported as cleared.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_UNC = "\\\\192.0.2.1\\share"


# --------------------------------------------------------------------------- #
#  Harness                                                                     #
# --------------------------------------------------------------------------- #

class _StubBackend:
    model_id = "stub-model"
    native_tools = False
    supports_native_tools = True
    supports_grammar = False

    def set_tools(self, defs):
        pass

    def chat(self, messages, **kw):
        return ""


def _coder_app(tmp_path, monkeypatch, *, api_key):
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setenv("LOCALM_API_KEY", api_key)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("coder")

    async def switch_model(name):
        pass

    from localm.plugins.gui.web import attach_gui
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "m")
    return app


def _owner(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(proj)
    return app, proj, {"Authorization": "Bearer ownersecret"}


def _seed(proj, lesson, **kw):
    from localm.plugins.coder.episodes import Episode, EpisodeStore
    return EpisodeStore(Path(proj)).add(
        Episode(task=kw.get("task", "t"), outcome="ok", summary="",
                lesson=lesson, turns=1))


# --------------------------------------------------------------------------- #
#  --seed                                                                      #
# --------------------------------------------------------------------------- #

def test_seed_reaches_the_agents_generation_kwargs(tmp_path, monkeypatch):
    """A plain generation kwarg beside temperature and max_tokens, which were
    already web fields. It reaches gen_kwargs, which is what every LLM call
    forwards."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "seed": 4242})
        assert r.status_code == 200, r.text
        sess = app.state.coder_sessions.get(r.json()["id"])
        assert sess.agent.gen_kwargs.get("seed") == 4242


def test_seed_omitted_leaves_sampling_unpinned(tmp_path, monkeypatch):
    """Blank must mean NO seed, not seed 0 - which is a real, reproducible
    value. Coercing the two together would silently pin every session that left
    the field alone."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj)})
        sess = app.state.coder_sessions.get(r.json()["id"])
        assert "seed" not in sess.agent.gen_kwargs

        z = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "seed": 0})
        zs = app.state.coder_sessions.get(z.json()["id"])
        assert zs.agent.gen_kwargs.get("seed") == 0, (
            "seed 0 is a real seed and must survive")


# --------------------------------------------------------------------------- #
#  --interactive-confirm                                                       #
# --------------------------------------------------------------------------- #

def test_interactive_confirm_keeps_shell_gated_under_auto_approve(tmp_path,
                                                                  monkeypatch):
    """The whole point: auto-approve lets file writes through, and shell
    execution STILL stops for a human. Asserted on the agent's always_confirm
    set, which is what the dispatch gate actually consults."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    from localm.plugins.coder.agent.constants import _SHELL_EXEC_TOOLS
    with TestClient(app) as client:
        r = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "auto_approve": True,
                              "interactive_confirm": True})
        assert r.status_code == 200, r.text
        assert r.json()["interactive_confirm"] is True
        sess = app.state.coder_sessions.get(r.json()["id"])
        assert set(_SHELL_EXEC_TOOLS) <= set(sess.agent.always_confirm)


def test_interactive_confirm_defaults_off_and_adds_nothing(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        r = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "auto_approve": True})
        assert r.json()["interactive_confirm"] is False
        sess = app.state.coder_sessions.get(r.json()["id"])
        assert not sess.agent.always_confirm


def test_interactive_confirm_names_no_tools_a_restricted_session_lacks(
        tmp_path, monkeypatch):
    """A scoped-key session has no shell tools at all, so listing them as
    always-confirm would advertise a gate on capabilities it does not have."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    from localm import auth
    with TestClient(app) as client:
        scoped = auth.create_key("phone", ["coder"])
        r = client.post("/api/coder/sessions",
                        headers={"Authorization": "Bearer %s" % scoped["key"]},
                        json={"cwd": str(proj), "auto_approve": True,
                              "interactive_confirm": True})
        sess = app.state.coder_sessions.get(r.json()["id"])
        assert sess.restricted is True
        assert not sess.agent.always_confirm


# --------------------------------------------------------------------------- #
#  --episodes-archive                                                          #
# --------------------------------------------------------------------------- #

def test_archive_lists_dropped_lessons(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        ep = _seed(proj, "the dropped one")
        client.post("/api/coder/episodes/%s/forget" % ep.id, headers=owner,
                    json={"cwd": str(proj)})
        r = client.get("/api/coder/episodes/archive", headers=owner,
                       params={"cwd": str(proj)})
        assert r.status_code == 200, r.text
        rows = r.json()["archived"]
        assert [x["id"] for x in rows] == [ep.id]
        assert rows[0]["reason"] == "forget"


def test_an_unreadable_archive_is_not_reported_as_an_empty_one(tmp_path,
                                                               monkeypatch):
    """The distinction this endpoint exists for. A 200 with an empty list would
    say the lesson is gone, when it may be sitting in the archive, recoverable.
    The CLI refuses that collapse by exiting non-zero; here it is a 503."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        ep = _seed(proj, "still recoverable")
        client.post("/api/coder/episodes/%s/forget" % ep.id, headers=owner,
                    json={"cwd": str(proj)})
        from localm.plugins.coder import episodes as _eps

        def _boom(self):
            self.last_forgotten_ok = False
            return []

        monkeypatch.setattr(_eps.EpisodeStore, "forgotten", _boom)
        r = client.get("/api/coder/episodes/archive", headers=owner,
                       params={"cwd": str(proj)})
        assert r.status_code == 503, r.text
        assert "INCOMPLETE" in r.json()["detail"]


# --------------------------------------------------------------------------- #
#  --forget-episode / --restore-episode                                        #
# --------------------------------------------------------------------------- #

def test_forget_then_restore_round_trips(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    from localm.plugins.coder.episodes import EpisodeStore
    with TestClient(app) as client:
        ep = _seed(proj, "run the tests first")
        f = client.post("/api/coder/episodes/%s/forget" % ep.id, headers=owner,
                        json={"cwd": str(proj)})
        assert f.status_code == 200, f.text
        assert f.json()["recoverable"] is True and f.json()["warning"] is None
        assert [e.id for e in EpisodeStore(Path(proj)).all()] == []

        r = client.post("/api/coder/episodes/%s/restore" % ep.id, headers=owner,
                        json={"cwd": str(proj)})
        assert r.status_code == 200, r.text
        assert r.json()["restored"] == ep.id
        assert r.json()["notes"] == []
        assert [e.id for e in EpisodeStore(Path(proj)).all()] == [ep.id]


def test_forget_reports_when_the_drop_is_not_recoverable(tmp_path, monkeypatch):
    """The lesson IS gone from recall, so this is a caveat on a real outcome, not
    a failure. Saying nothing would leave the user believing they can restore
    it - which they would discover at the worst possible moment."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        ep = _seed(proj, "unarchivable")
        from localm.plugins.coder import episodes as _eps
        monkeypatch.setattr(_eps.EpisodeStore, "_archive",
                            lambda self, eps, reason: False)
        r = client.post("/api/coder/episodes/%s/forget" % ep.id, headers=owner,
                        json={"cwd": str(proj)})
        assert r.status_code == 200
        assert r.json()["recoverable"] is False
        assert "cannot be restored" in r.json()["warning"]


def test_forget_and_restore_404_on_an_unknown_id(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        _seed(proj, "present")
        assert client.post("/api/coder/episodes/nope/forget", headers=owner,
                           json={"cwd": str(proj)}).status_code == 404
        assert client.post("/api/coder/episodes/nope/restore", headers=owner,
                           json={"cwd": str(proj)}).status_code == 404


def test_restore_does_not_claim_no_such_id_when_it_could_not_look(tmp_path,
                                                                  monkeypatch):
    """An unreadable archive means the id could not be LOOKED UP. Answering 404
    would assert the episode does not exist, which is a claim we cannot make and
    which sends the user away from a lesson that is still there."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        from localm.plugins.coder import episodes as _eps

        def _blind(self, episode_id):
            self.last_forgotten_ok = False
            return None

        monkeypatch.setattr(_eps.EpisodeStore, "restore", _blind)
        r = client.post("/api/coder/episodes/whatever/restore", headers=owner,
                        json={"cwd": str(proj)})
        assert r.status_code == 503, r.text
        assert "Nothing was changed" in r.json()["detail"]


def test_restore_surfaces_the_immediately_re_evicted_case(tmp_path, monkeypatch):
    """A restore that lands and is dropped again at the cap SUCCEEDED, and
    reporting only "restored" would be contradicted by the very next read."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        from localm.plugins.coder import episodes as _eps
        from localm.plugins.coder.episodes import Episode

        ghost = Episode(task="t", outcome="ok", lesson="back then gone")
        ghost.id = "ghost1"

        def _evicting(self, episode_id):
            self.last_restore_archive_ok = True
            self.last_evicted = [ghost]
            return ghost

        monkeypatch.setattr(_eps.EpisodeStore, "restore", _evicting)
        r = client.post("/api/coder/episodes/ghost1/restore", headers=owner,
                        json={"cwd": str(proj)})
        assert r.json()["evicted_again"] is True
        assert any("dropped again immediately" in n for n in r.json()["notes"])


# --------------------------------------------------------------------------- #
#  --forget-episodes (erase everything)                                        #
# --------------------------------------------------------------------------- #

def test_erase_removes_the_archive_too_and_counts_what_it_destroyed(
        tmp_path, monkeypatch):
    """The archive has to go: "cleared episodic memory" while the lesson text
    still sat in a sidecar would be a privacy claim that is not true."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    from localm.plugins.coder.episodes import EpisodeStore
    with TestClient(app) as client:
        keep = _seed(proj, "live one")
        drop = _seed(proj, "dropped one")
        client.post("/api/coder/episodes/%s/forget" % drop.id, headers=owner,
                    json={"cwd": str(proj)})

        r = client.request("DELETE", "/api/coder/episodes", headers=owner,
                           json={"cwd": str(proj)})
        assert r.status_code == 200, r.text
        assert r.json() == {"erased": 1, "erased_archived": 1}

        store = EpisodeStore(Path(proj))
        assert store.all() == []
        assert store.forgotten() == []
        assert not store.path.exists() and not store.archive_path.exists()
        assert keep.id and drop.id      # ids were real, not empty strings


def test_a_partial_erase_is_never_reported_as_cleared(tmp_path, monkeypatch):
    """The one episode operation with no undo, so "erased" has to be a MEASURED
    claim. A clear() that left records behind while answering 200 would be a
    privacy promise that was not kept."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        _seed(proj, "stubborn")
        from localm.plugins.coder import episodes as _eps
        monkeypatch.setattr(_eps.EpisodeStore, "clear", lambda self: None)
        r = client.request("DELETE", "/api/coder/episodes", headers=owner,
                           json={"cwd": str(proj)})
        assert r.status_code == 500, r.text
        assert "NOT reported as cleared" in r.json()["detail"]


# --------------------------------------------------------------------------- #
#  --consolidate-episodes                                                      #
# --------------------------------------------------------------------------- #

def test_consolidate_reports_what_it_did(tmp_path, monkeypatch):
    """Memory that rewrites itself without saying so is how a bad merge becomes
    invisible, so the report is the feature, not decoration."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        from localm.plugins.coder import episodes as _eps
        # The route imports consolidate inside its worker, so patching the module
        # attribute is seen at call time. The stub never calls `complete`.
        monkeypatch.setattr(
            _eps, "consolidate",
            lambda store, **kw: {"groups": 2, "merged": 2, "replaced": 5,
                                 "archived": 5, "skipped": 1,
                                 "warning": "one group was left alone"})
        r = client.post("/api/coder/episodes/consolidate", headers=owner,
                        json={"cwd": str(proj)})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["groups"] == 2 and body["replaced"] == 5
        assert body["skipped"] == 1, "a group with no usable merge is COUNTED"
        assert body["warning"]


def test_consolidate_backend_uses_the_persisted_owner_key(tmp_path, monkeypatch):
    """The self-call backend consolidate builds must present the real owner
    key, not the hardcoded "localm" placeholder, once one is configured -
    the same precedence make_localm_backend already uses."""
    app, proj, _owner_header = _owner(tmp_path, monkeypatch)
    from localm import auth
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    auth.set_api_key("file-key-persisted")

    captured = {}
    import localm.plugins.coder.backends.http as _http
    real_init = _http.HTTPBackend.__init__

    def _spy_init(self, *a, **kw):
        captured["api_key"] = kw.get("api_key")
        return real_init(self, *a, **kw)

    monkeypatch.setattr(_http.HTTPBackend, "__init__", _spy_init)
    from localm.plugins.coder import episodes as _eps
    # The route imports consolidate inside its worker, so patching the module
    # attribute is seen at call time. The stub never calls `complete`.
    monkeypatch.setattr(
        _eps, "consolidate",
        lambda store, **kw: {"groups": 0, "merged": 0, "replaced": 0,
                             "archived": 0, "skipped": 0})
    with TestClient(app) as client:
        r = client.post("/api/coder/episodes/consolidate",
                        headers={"Authorization": "Bearer file-key-persisted"},
                        json={"cwd": str(proj)})
    assert r.status_code == 200, r.text
    assert captured.get("api_key") == "file-key-persisted"


def test_consolidate_needs_the_gui_server(tmp_path, monkeypatch):
    """It takes a model turn, so without the shared services there is nothing to
    ask - said plainly rather than answered with an empty report."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        app.state.self_url = None
        r = client.post("/api/coder/episodes/consolidate", headers=owner,
                        json={"cwd": str(proj)})
        assert r.status_code == 503


# --------------------------------------------------------------------------- #
#  Ownership and path safety, on every write route                             #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method,path_tpl", [
    ("POST", "/api/coder/episodes/{id}/forget"),
    ("DELETE", "/api/coder/episodes"),
    ("POST", "/api/coder/episodes/consolidate"),
])
def test_every_episode_write_is_owner_only(tmp_path, monkeypatch, method,
                                           path_tpl):
    """A scoped key is excluded from episodic memory entirely - it neither
    recalls a lesson nor writes one - so it must not be able to destroy the
    owner's.

    The id in the path is the REAL seeded one, not a made-up string: with a
    placeholder id the route answers 404 for "no such episode" whether or not the
    owner gate is present, and the survival assertion is toothless because a
    nonexistent lesson cannot be destroyed either way."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    from localm import auth
    from localm.plugins.coder.episodes import EpisodeStore
    with TestClient(app) as client:
        ep = _seed(proj, "the owner's")
        path = path_tpl.format(id=ep.id)
        scoped = auth.create_key("phone", ["coder"])
        r = client.request(method, path,
                           headers={"Authorization": "Bearer %s" % scoped["key"]},
                           json={"cwd": str(proj)})
        assert r.status_code in (403, 404), r.text
        # Assert on the data, not only the status.
        assert [e.id for e in EpisodeStore(Path(proj)).all()] == [ep.id], (
            "the owner's lesson did not survive a scoped key's write")


def test_restore_is_owner_only_and_leaks_no_archived_text(tmp_path, monkeypatch):
    """restore gets its own test because it is not DESTRUCTIVE, so the
    lesson-survived property every other write is checked with cannot detect a
    missing gate here.

    Its two real risks are resurrecting a lesson the owner deliberately dropped,
    and leaking the archived lesson TEXT, which the success body carries."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    from localm import auth
    from localm.plugins.coder.episodes import EpisodeStore
    with TestClient(app) as client:
        ep = _seed(proj, "a secret the owner dropped on purpose")
        client.post("/api/coder/episodes/%s/forget" % ep.id, headers=owner,
                    json={"cwd": str(proj)})
        assert EpisodeStore(Path(proj)).all() == []

        scoped = auth.create_key("phone", ["coder"])
        r = client.post("/api/coder/episodes/%s/restore" % ep.id,
                        headers={"Authorization": "Bearer %s" % scoped["key"]},
                        json={"cwd": str(proj)})
        assert r.status_code in (403, 404), r.text
        assert "secret the owner dropped" not in r.text, "archived text leaked"
        assert EpisodeStore(Path(proj)).all() == [], (
            "a scoped key resurrected a lesson the owner had dropped")


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/coder/episodes/archive"),
    ("POST", "/api/coder/episodes/abc/forget"),
    ("DELETE", "/api/coder/episodes"),
    ("POST", "/api/coder/episodes/consolidate"),
])
def test_every_episode_route_refuses_unc(tmp_path, monkeypatch, method, path):
    """The spy covers resolve AND is_dir rather than whichever one the shared
    guard happens to call today: a spy pointed at a single method goes
    structurally dead the moment the code reaches for the other, and a dead
    fault injector looks exactly like a guard that found nothing to refuse."""
    real = {"resolve": Path.resolve, "is_dir": Path.is_dir}

    def make_spy(name):
        def spy(self, *a, **kw):
            s = str(self)
            if s[:2] in ("\\\\", "//", "\\/", "/\\"):
                raise AssertionError(
                    "Path.%s() reached the filesystem with %r" % (name, s))
            return real[name](self, *a, **kw)
        return spy

    for name in real:
        monkeypatch.setattr(Path, name, make_spy(name))
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        if method == "GET":
            r = client.get(path, headers=owner, params={"cwd": _UNC})
        else:
            r = client.request(method, path, headers=owner, json={"cwd": _UNC})
        assert r.status_code == 400, r.text
        assert "UNC or device" in r.json()["detail"]
