# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every route a running app serves has a pinned privilege tier.

The app is ``create_app`` with ``attach_gui`` applied and every bundled plugin
loaded through the real engine, so a plugin router is walked with the same
scope gate the engine mounts it under. The pin of a route is the dependency
chain its handler runs behind (``scope:<s>`` is a ``require_scope`` gate,
``owner`` a ``require_owner`` gate, ``auth`` any valid key, ``fs_host`` the
host-filesystem gate). A route that is served and has no entry here fails, and
so does an entry whose route is gone or whose gate changed.

The coder plugin has three tiers on top of its scope gate: a session-scoped
route resolves the caller's own session before the handler runs, an owner-only
route refuses a non-owner before the handler runs, and the remaining routes are
either restricted-safe or answer a non-owner with an empty body themselves.
"""

import pathlib
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_create_app_characterization import _expected_routes, _route_table
from tests import _routing_fit as _rf

@pytest.fixture(autouse=True)
def _every_candidate_fits(monkeypatch):
    """Routing fit is not this file's subject: every candidate fits the GPU."""
    _rf.make_every_candidate_fit(monkeypatch)


# (kind, method, path) -> gate chain, for the GUI surface attach_gui adds to
# create_app, then one table per bundled plugin.
_GUI_ROUTES = {
    ('mount', '*', ''): (),
    ('api', 'GET', '/'): (),
    ('api', 'POST', '/api/app/rebuild-launcher'): ('scope:config:write',),
    ('api', 'GET', '/api/backend'): ('scope:config:read',),
    ('api', 'GET', '/api/capabilities'): ('auth', 'caller_scopes'),
    ('api', 'GET', '/api/comfy/managed-status'): ('scope:config:read',),
    ('api', 'POST', '/api/comfy/remove'): ('scope:config:write',),
    ('api', 'POST', '/api/comfy/repair'): ('scope:config:write',),
    ('api', 'POST', '/api/comfy/setup'): ('scope:config:write',),
    ('api', 'POST', '/api/comfy/update'): ('scope:config:write',),
    ('api', 'POST', '/api/comfyui/create-launcher'): ('scope:config:write',),
    ('api', 'GET', '/api/companion'): ('scope:config:read',),
    ('api', 'GET', '/api/discover/files'): ('scope:models:read',),
    ('api', 'GET', '/api/discover/search'): ('scope:models:read',),
    ('api', 'GET', '/api/doctor'): ('scope:config:read',),
    ('api', 'POST', '/api/doctor/run'): ('scope:config:write',),
    ('api', 'POST', '/api/embedding/warmup'): ('scope:models:write',),
    ('api', 'GET', '/api/fs/dirs'): ('fs_host',),
    ('api', 'POST', '/api/fs/mkdir'): ('scope:config:write', 'fs_host'),
    ('api', 'GET', '/api/fs/places'): ('fs_host',),
    ('api', 'POST', '/api/fs/rename'): ('scope:config:write', 'fs_host'),
    ('api', 'POST', '/api/fs/save-text'): ('scope:config:write', 'fs_host'),
    ('api', 'GET', '/api/gpus'): ('scope:models:read',),
    ('api', 'POST', '/api/gui-error'): ('auth',),
    ('api', 'GET', '/api/image-proxy'): ('scope:chat',),
    ('api', 'GET', '/api/instances'): ('scope:config:read',),
    ('api', 'POST', '/api/instances/{instance_id}/stop'): ('scope:admin',),
    ('api', 'GET', '/api/live'): ('auth',),
    ('api', 'POST', '/api/logs/export'): ('scope:config:write', 'fs_host'),
    ('api', 'POST', '/api/media/{kind}/preflight'): ('scope:models:write',),
    ('api', 'GET', '/api/models'): ('scope:models:read',),
    ('api', 'POST', '/api/models/adapters/attach'): ('scope:models:write',),
    ('api', 'POST', '/api/models/adapters/detach'): ('scope:models:write',),
    ('api', 'POST', '/api/models/alias'): ('scope:models:write',),
    ('api', 'POST', '/api/models/comfy-source/lookup'): ('scope:models:write',),
    ('api', 'POST', '/api/models/load'): ('scope:models:write',),
    ('api', 'POST', '/api/models/packages/install'): ('scope:plugins:admin',),
    ('api', 'GET', '/api/models/packages/install/events'): ('scope:plugins:admin',),
    ('api', 'POST', '/api/models/packages/plan'): ('scope:plugins:admin',),
    ('api', 'POST', '/api/models/pull'): ('scope:models:write',),
    ('api', 'POST', '/api/models/pull-comfy-source'): ('scope:models:write',),
    ('api', 'POST', '/api/models/pull-token/redeem'): ('scope:models:write',),
    ('api', 'POST', '/api/models/pull/discard'): ('scope:models:write',),
    ('api', 'GET', '/api/models/pulls/interrupted'): ('scope:models:write',),
    ('api', 'POST', '/api/models/pulls/interrupted/{pull_id}/delete'): ('scope:models:write',),
    ('api', 'POST', '/api/models/pulls/interrupted/{pull_id}/dismiss'): ('scope:models:write',),
    ('api', 'POST', '/api/models/pulls/interrupted/{pull_id}/resume'): ('scope:models:write',),
    ('api', 'POST', '/api/models/pulls/orphans/{orphan_id}/delete'): ('scope:models:write',),
    ('api', 'POST', '/api/models/relocate'): ('scope:models:write',),
    ('api', 'POST', '/api/models/remove'): ('scope:models:write',),
    ('api', 'POST', '/api/models/rename'): ('scope:models:write',),
    ('api', 'GET', '/api/models/roles'): ('scope:models:read',),
    ('api', 'POST', '/api/models/scan'): ('scope:models:write',),
    ('api', 'GET', '/api/models/shortcuts'): ('scope:models:read',),
    ('api', 'POST', '/api/models/type'): ('scope:models:write',),
    ('api', 'POST', '/api/models/unload'): ('scope:models:write',),
    ('api', 'GET', '/api/pairing/qr'): ('scope:admin',),
    ('api', 'POST', '/api/pairing/qr'): ('scope:keys:admin',),
    ('api', 'GET', '/api/runtime/check'): ('scope:config:read',),
    ('api', 'POST', '/api/runtime/keep'): ('scope:config:write',),
    ('api', 'POST', '/api/runtime/update'): ('scope:config:write',),
    ('api', 'POST', '/api/share/clear'): ('auth',),
    ('api', 'GET', '/api/share/pending'): ('auth',),
    ('api', 'GET', '/api/stats'): ('auth',),
    ('api', 'POST', '/api/upload'): ('scope:config:write',),
    ('api', 'GET', '/api/uploads'): ('scope:config:read',),
    ('api', 'DELETE', '/api/uploads/{name}'): ('scope:config:write',),
    ('api', 'GET', '/api/vram-estimate'): ('scope:models:read',),
    ('api', 'GET', '/index.html'): (),
    ('api', 'POST', '/share-target'): ('auth',),
    ('api', 'GET', '/sw.js'): (),
}

_BROWSER_PLUGIN_ROUTES = {
    ('api', 'GET', '/api/browser/agent'): ('scope:browser',),
    ('api', 'POST', '/api/browser/agent'): ('scope:browser',),
    ('api', 'POST', '/api/browser/click'): ('scope:browser',),
    ('api', 'POST', '/api/browser/download'): ('scope:browser',),
    ('api', 'GET', '/api/browser/engine'): ('scope:browser',),
    ('api', 'POST', '/api/browser/key'): ('scope:browser',),
    ('api', 'POST', '/api/browser/navigate'): ('scope:browser',),
    ('api', 'POST', '/api/browser/scroll'): ('scope:browser',),
    ('api', 'POST', '/api/browser/session'): ('scope:browser',),
    ('api', 'GET', '/api/browser/state'): ('scope:browser',),
    ('api', 'POST', '/api/browser/stop'): ('scope:browser',),
    ('api', 'POST', '/api/browser/type'): ('scope:browser',),
    ('mount', '*', '/plugins/browser'): (),
}

_CODER_PLUGIN_ROUTES = {
    ('api', 'DELETE', '/api/coder/checkpoints'): ('scope:coder', '_require_owner'),
    ('api', 'GET', '/api/coder/dormant'): ('scope:coder',),
    ('api', 'DELETE', '/api/coder/episodes'): ('scope:coder', '_require_owner'),
    ('api', 'GET', '/api/coder/episodes'): ('scope:coder',),
    ('api', 'GET', '/api/coder/episodes/archive'): ('scope:coder',),
    ('api', 'POST', '/api/coder/episodes/consolidate'): ('scope:coder', '_require_owner'),
    ('api', 'POST', '/api/coder/episodes/{episode_id}/forget'): ('scope:coder', '_require_owner'),
    ('api', 'POST', '/api/coder/episodes/{episode_id}/restore'): ('scope:coder', '_require_owner'),
    ('api', 'GET', '/api/coder/history'): ('scope:coder',),
    ('api', 'GET', '/api/coder/history/{name}'): ('scope:coder', '_require_owner'),
    ('api', 'GET', '/api/coder/hooks'): ('scope:coder',),
    ('api', 'POST', '/api/coder/hooks/forget'): ('scope:coder', '_require_owner'),
    ('api', 'POST', '/api/coder/hooks/trust'): ('scope:coder', '_require_owner'),
    ('api', 'DELETE', '/api/coder/projects'): ('scope:coder', '_require_owner'),
    ('api', 'GET', '/api/coder/resumable'): ('scope:coder',),
    ('api', 'GET', '/api/coder/sessions'): ('scope:coder',),
    ('api', 'POST', '/api/coder/sessions'): ('scope:coder',),
    ('api', 'DELETE', '/api/coder/sessions/{session_id}'): ('scope:coder', '_bind_session'),
    ('api', 'GET', '/api/coder/sessions/{session_id}/background'): ('scope:coder', '_bind_session'),
    ('api', 'POST', '/api/coder/sessions/{session_id}/compact'): ('scope:coder', '_bind_session'),
    ('api', 'POST', '/api/coder/sessions/{session_id}/confirm'): ('scope:coder', '_bind_session'),
    ('api', 'POST', '/api/coder/sessions/{session_id}/cwd'): ('scope:coder', '_bind_session'),
    ('api', 'POST', '/api/coder/sessions/{session_id}/estimate'): ('scope:coder', '_bind_session'),
    ('api', 'GET', '/api/coder/sessions/{session_id}/events'): ('scope:coder', '_bind_session'),
    ('api', 'GET', '/api/coder/sessions/{session_id}/files'): ('scope:coder', '_bind_session'),
    ('api', 'GET', '/api/coder/sessions/{session_id}/files/diff'): ('scope:coder', '_bind_session'),
    ('api', 'GET', '/api/coder/sessions/{session_id}/files/download'): ('scope:coder', '_bind_session'),
    ('api', 'GET', '/api/coder/sessions/{session_id}/log'): ('scope:coder', '_bind_session'),
    ('api', 'GET', '/api/coder/sessions/{session_id}/memory'): ('scope:coder', '_bind_session'),
    ('api', 'POST', '/api/coder/sessions/{session_id}/memory'): ('scope:coder', '_bind_session'),
    ('api', 'POST', '/api/coder/sessions/{session_id}/memory/forget'): ('scope:coder', '_bind_session'),
    ('api', 'POST', '/api/coder/sessions/{session_id}/message'): ('scope:coder', '_bind_session'),
    ('api', 'POST', '/api/coder/sessions/{session_id}/model'): ('scope:coder', '_bind_session'),
    ('api', 'GET', '/api/coder/sessions/{session_id}/patch'): ('scope:coder', '_bind_session'),
    ('api', 'GET', '/api/coder/sessions/{session_id}/patch/download'): ('scope:coder', '_bind_session'),
    ('api', 'GET', '/api/coder/sessions/{session_id}/result'): ('scope:coder', '_bind_session'),
    ('api', 'POST', '/api/coder/sessions/{session_id}/settings'): ('scope:coder', '_bind_session'),
    ('api', 'POST', '/api/coder/sessions/{session_id}/stop'): ('scope:coder', '_bind_session'),
    ('api', 'POST', '/api/coder/sessions/{session_id}/undo'): ('scope:coder', '_bind_session'),
}

_EXLLAMA_PLUGIN_ROUTES = {

}

_IMAGE_PLUGIN_ROUTES = {
    ('api', 'GET', '/api/image/native-workflows'): ('scope:image',),
    ('api', 'POST', '/api/image/native-workflows'): ('scope:image',),
    ('api', 'POST', '/api/image/native-workflows/import'): ('scope:image',),
    ('api', 'DELETE', '/api/image/native-workflows/{wid}'): ('scope:image',),
    ('api', 'GET', '/api/image/native-workflows/{wid}'): ('scope:image',),
    ('api', 'POST', '/api/image/native-workflows/{wid}/rename'): ('scope:image',),
    ('api', 'POST', '/api/image/native-workflows/{wid}/resolve'): ('scope:image',),
    ('api', 'GET', '/api/image/workflows'): ('scope:image',),
    ('api', 'POST', '/api/image/workflows'): ('scope:image',),
    ('api', 'POST', '/api/image/workflows/select'): ('scope:image',),
    ('api', 'DELETE', '/api/image/workflows/{name}'): ('scope:image',),
    ('api', 'POST', '/api/imagine'): ('scope:image',),
    ('api', 'GET', '/api/imagine/backend'): ('scope:image',),
    ('api', 'POST', '/api/imagine/comfy-launch'): ('scope:image',),
    ('api', 'GET', '/api/imagine/comfy-models'): ('scope:image',),
    ('api', 'DELETE', '/api/imagine/file/{name}'): ('scope:image', 'owner', '_resolve'),
    ('api', 'GET', '/api/imagine/file/{name}'): ('scope:image', 'owner', '_resolve'),
    ('api', 'POST', '/api/imagine/file/{name}/move'): ('scope:image', 'owner', '_resolve'),
    ('api', 'POST', '/api/imagine/file/{name}/rename'): ('scope:image', 'owner', '_resolve'),
    ('api', 'POST', '/api/imagine/file/{name}/upscale'): ('scope:image', 'owner', '_resolve'),
    ('api', 'GET', '/api/imagine/history'): ('scope:image',),
    ('api', 'POST', '/api/imagine/import-params'): ('scope:image',),
    ('api', 'GET', '/api/imagine/native-models'): ('scope:image',),
    ('api', 'POST', '/v1/images/generations'): ('scope:image',),
}

_JOBS_PLUGIN_ROUTES = {
    ('api', 'GET', '/api/jobs'): ('scope:jobs',),
    ('api', 'POST', '/api/jobs'): ('scope:jobs',),
    ('api', 'DELETE', '/api/jobs/{job_id}'): ('scope:jobs', 'owned_job'),
    ('api', 'GET', '/api/jobs/{job_id}'): ('scope:jobs', 'owned_job'),
    ('api', 'PUT', '/api/jobs/{job_id}'): ('scope:jobs', 'owned_job'),
    ('api', 'GET', '/api/jobs/{job_id}/results'): ('scope:jobs', 'owned_job'),
    ('api', 'POST', '/api/jobs/{job_id}/run'): ('scope:jobs', 'owned_job'),
    ('mount', '*', '/plugins/jobs'): (),
}

_MCP_PLUGIN_ROUTES = {

}

_MEMORY_PLUGIN_ROUTES = {
    ('api', 'GET', '/api/memory'): ('scope:memory',),
    ('api', 'PUT', '/api/memory'): ('scope:memory',),
    ('api', 'POST', '/api/memory/append'): ('scope:memory',),
    ('api', 'POST', '/api/memory/consolidate'): ('scope:memory',),
    ('api', 'POST', '/api/memory/corrections/{cid}/accept'): ('scope:memory',),
    ('api', 'POST', '/api/memory/corrections/{cid}/reject'): ('scope:memory',),
    ('api', 'GET', '/api/memory/forgotten'): ('scope:memory',),
    ('api', 'POST', '/api/memory/forgotten/{mem_id}/restore'): ('scope:memory',),
    ('api', 'DELETE', '/api/memory/{mem_id}'): ('scope:memory',),
    ('api', 'PATCH', '/api/memory/{mem_id}'): ('scope:memory',),
}

_MUSIC_PLUGIN_ROUTES = {
    ('api', 'POST', '/api/music'): ('scope:music',),
    ('api', 'GET', '/api/music/backend'): ('scope:music',),
    ('api', 'POST', '/api/music/comfy-launch'): ('scope:music',),
    ('api', 'GET', '/api/music/comfy-models'): ('scope:music',),
    ('api', 'DELETE', '/api/music/file/{name}'): ('scope:music', 'owner', '_resolve'),
    ('api', 'GET', '/api/music/file/{name}'): ('scope:music', 'owner', '_resolve'),
    ('api', 'POST', '/api/music/file/{name}/move'): ('scope:music', 'owner', '_resolve'),
    ('api', 'POST', '/api/music/file/{name}/rename'): ('scope:music', 'owner', '_resolve'),
    ('api', 'GET', '/api/music/history'): ('scope:music',),
    ('api', 'GET', '/api/music/native-models'): ('scope:music',),
    ('api', 'GET', '/api/music/native-workflows'): ('scope:music',),
    ('api', 'POST', '/api/music/native-workflows'): ('scope:music',),
    ('api', 'POST', '/api/music/native-workflows/import'): ('scope:music',),
    ('api', 'DELETE', '/api/music/native-workflows/{wid}'): ('scope:music',),
    ('api', 'GET', '/api/music/native-workflows/{wid}'): ('scope:music',),
    ('api', 'POST', '/api/music/native-workflows/{wid}/rename'): ('scope:music',),
    ('api', 'POST', '/api/music/native-workflows/{wid}/resolve'): ('scope:music',),
    ('api', 'GET', '/api/music/workflows'): ('scope:music',),
    ('api', 'POST', '/api/music/workflows'): ('scope:music',),
    ('api', 'POST', '/api/music/workflows/select'): ('scope:music',),
    ('api', 'DELETE', '/api/music/workflows/{name}'): ('scope:music',),
}

_RAG_PLUGIN_ROUTES = {
    ('api', 'GET', '/api/rag/collections'): ('scope:rag',),
    ('api', 'POST', '/api/rag/collections'): ('scope:rag',),
    ('api', 'DELETE', '/api/rag/collections/{name}'): ('scope:rag',),
    ('api', 'GET', '/api/rag/collections/{name}'): ('scope:rag',),
    ('api', 'POST', '/api/rag/collections/{name}/add'): ('scope:rag',),
    ('api', 'POST', '/api/rag/collections/{name}/query'): ('scope:rag',),
    ('api', 'POST', '/api/rag/collections/{name}/reembed'): ('scope:rag',),
    ('api', 'POST', '/api/rag/collections/{name}/remove-doc'): ('scope:rag',),
    ('api', 'POST', '/api/rag/collections/{name}/repair'): ('scope:rag',),
    ('api', 'POST', '/api/rag/collections/{name}/upload'): ('scope:rag',),
    ('api', 'GET', '/api/rag/embedding'): ('scope:rag',),
    ('api', 'POST', '/api/rag/embedding'): ('scope:rag',),
    ('api', 'POST', '/api/rag/embedding/download'): ('scope:rag',),
    ('api', 'POST', '/api/rag/extract'): ('scope:rag',),
    ('api', 'GET', '/api/rag/rerank'): ('scope:rag',),
}

_TTS_PLUGIN_ROUTES = {
    ('api', 'GET', '/api/tts/config'): ('scope:tts',),
    ('api', 'GET', '/api/tts/status'): ('scope:tts',),
    ('mount', '*', '/plugins/tts'): (),
    ('api', 'POST', '/v1/audio/speech'): ('scope:tts',),
}

_VIDEO_PLUGIN_ROUTES = {
    ('api', 'POST', '/api/video'): ('scope:video',),
    ('api', 'GET', '/api/video/backend'): ('scope:video',),
    ('api', 'POST', '/api/video/comfy-launch'): ('scope:video',),
    ('api', 'GET', '/api/video/comfy-models'): ('scope:video',),
    ('api', 'DELETE', '/api/video/file/{name}'): ('scope:video', 'owner', '_resolve'),
    ('api', 'GET', '/api/video/file/{name}'): ('scope:video', 'owner', '_resolve'),
    ('api', 'POST', '/api/video/file/{name}/move'): ('scope:video', 'owner', '_resolve'),
    ('api', 'POST', '/api/video/file/{name}/rename'): ('scope:video', 'owner', '_resolve'),
    ('api', 'GET', '/api/video/history'): ('scope:video',),
    ('api', 'GET', '/api/video/native-models'): ('scope:video',),
    ('api', 'GET', '/api/video/native-workflows'): ('scope:video',),
    ('api', 'POST', '/api/video/native-workflows'): ('scope:video',),
    ('api', 'POST', '/api/video/native-workflows/import'): ('scope:video',),
    ('api', 'DELETE', '/api/video/native-workflows/{wid}'): ('scope:video',),
    ('api', 'GET', '/api/video/native-workflows/{wid}'): ('scope:video',),
    ('api', 'POST', '/api/video/native-workflows/{wid}/rename'): ('scope:video',),
    ('api', 'POST', '/api/video/native-workflows/{wid}/resolve'): ('scope:video',),
    ('api', 'GET', '/api/video/workflows'): ('scope:video',),
    ('api', 'POST', '/api/video/workflows'): ('scope:video',),
    ('api', 'POST', '/api/video/workflows/select'): ('scope:video',),
    ('api', 'DELETE', '/api/video/workflows/{name}'): ('scope:video',),
}

_VOICE_PLUGIN_ROUTES = {
    ('api', 'POST', '/api/voice/model/download'): ('scope:voice',),
    ('api', 'GET', '/api/voice/status'): ('scope:voice',),
    ('api', 'POST', '/api/voice/transcribe'): ('scope:voice',),
    ('api', 'POST', '/v1/audio/transcriptions'): ('scope:voice',),
}

_WEB_PLUGIN_ROUTES = {
    ('api', 'POST', '/api/web/fetch'): ('scope:web',),
    ('api', 'POST', '/api/web/retrieve'): ('scope:web',),
    ('api', 'POST', '/api/web/search'): ('scope:web',),
}


_PLUGIN_TABLES = (
    _BROWSER_PLUGIN_ROUTES, _CODER_PLUGIN_ROUTES, _EXLLAMA_PLUGIN_ROUTES,
    _IMAGE_PLUGIN_ROUTES, _JOBS_PLUGIN_ROUTES, _MCP_PLUGIN_ROUTES,
    _MEMORY_PLUGIN_ROUTES, _MUSIC_PLUGIN_ROUTES, _RAG_PLUGIN_ROUTES,
    _TTS_PLUGIN_ROUTES, _VIDEO_PLUGIN_ROUTES, _VOICE_PLUGIN_ROUTES,
    _WEB_PLUGIN_ROUTES,
)

_CODER_SAFE = {
    ("GET", "/api/coder/sessions"),
    ("POST", "/api/coder/sessions"),
}

_CODER_ANSWERS_NON_OWNER_ITSELF = {
    ("GET", "/api/coder/history"),
    ("GET", "/api/coder/resumable"),
    ("GET", "/api/coder/dormant"),
    ("GET", "/api/coder/episodes"),
    ("GET", "/api/coder/episodes/archive"),
    ("GET", "/api/coder/hooks"),
}


@pytest.fixture(scope="module")
def full_app(tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    mp = pytest.MonkeyPatch()
    mp.setenv("LOCALM_HOME", str(home))
    import localm.config as cfg
    mp.setattr(cfg, "HOME_DIR", home)
    mp.setattr(cfg, "MODELS_DIR", home / "models")
    mp.setattr(cfg, "CONFIG_FILE", home / "config.json")
    mp.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.inference.http_server import create_app
    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.web import attach_gui

    app = create_app(None)

    async def _switch(model, **kwargs):
        return None

    attach_gui(app, self_url="http://127.0.0.1:9/v1", switch_model=_switch,
               active_model=lambda: "")
    import localm
    builtin = pathlib.Path(localm.__file__).parent / "plugins" / "builtin"
    manager = PluginManager(app, installed_root=builtin, store_root=None)
    live = app.state.plugin_manager
    for name, spec in sorted(manager.discover().items()):
        if name not in live._loaded:
            manager._load(spec)
    try:
        yield app
    finally:
        for name in list(manager._loaded):
            manager._unload(name)
        mp.undo()


def _expected_full_table() -> dict:
    table = dict(_expected_routes(False))
    table.update(_GUI_ROUTES)
    for plugin_table in _PLUGIN_TABLES:
        table.update(plugin_table)
    return table


def test_every_served_route_has_a_pinned_gate(full_app):
    actual = _route_table(full_app)
    expected = _expected_full_table()
    untiered = sorted(f"    {k}: {actual[k]}" for k in actual.keys() - expected.keys())
    gone = sorted(f"    {k}" for k in expected.keys() - actual.keys())
    changed = sorted(f"    {k}: pinned {expected[k]}, got {actual[k]}"
                     for k in actual.keys() & expected.keys() if actual[k] != expected[k])
    assert not untiered, "served routes with no tier entry:\n" + "\n".join(untiered)
    assert not gone, "pinned routes no longer served:\n" + "\n".join(gone)
    assert not changed, "routes whose gate changed:\n" + "\n".join(changed)


def test_every_coder_route_has_a_tier(full_app):
    coder = {k: v for k, v in _route_table(full_app).items()
             if k[0] == "api" and k[2].startswith("/api/coder/")}
    assert coder, "no coder routes were mounted"
    for (_, method, path), labels in coder.items():
        tiers = [t for t, label in (("session", "_bind_session"),
                                    ("owner", "_require_owner")) if label in labels]
        listed = (method, path) in _CODER_SAFE | _CODER_ANSWERS_NON_OWNER_ITSELF
        assert len(tiers) + int(listed) == 1, (
            f"{method} {path} must sit in exactly one coder tier, "
            f"has session/owner gates {tiers} and listed={listed}")


def _coder_app(monkeypatch, *, owner: bool, principal: str | None = None):
    from localm.plugins.builtin.coder import plug

    monkeypatch.setattr(plug, "_principal_from_request",
                        lambda request: (owner, None if owner else principal))
    app = FastAPI()
    app.include_router(plug._router)
    sessions = {"s1": SimpleNamespace(principal="p-owner")}
    app.state.coder_sessions = SimpleNamespace(get=sessions.get)
    return TestClient(app, raise_server_exceptions=False)


_OWNER_REFUSALS = {
    ("DELETE", "/api/coder/checkpoints"): (403, "Owner only"),
    ("DELETE", "/api/coder/projects"): (403, "Owner only"),
    ("GET", "/api/coder/history/x_localcoder.jsonl"): (404, "No such log"),
    ("POST", "/api/coder/episodes/e1/forget"): (404, "No such episode"),
    ("POST", "/api/coder/episodes/e1/restore"): (404, "No such episode"),
    ("DELETE", "/api/coder/episodes"): (403, "Owner only"),
    ("POST", "/api/coder/episodes/consolidate"): (403, "Owner only"),
    ("POST", "/api/coder/hooks/trust"): (404, "No tool hooks"),
    ("POST", "/api/coder/hooks/forget"): (404, "No tool hooks"),
}


@pytest.mark.parametrize("method,path", sorted(_OWNER_REFUSALS))
def test_owner_only_coder_routes_refuse_a_scoped_key_before_the_handler(
        monkeypatch, method, path):
    client = _coder_app(monkeypatch, owner=False, principal="p-other")
    status, detail = _OWNER_REFUSALS[(method, path)]
    response = client.request(method, path, json={})
    assert (response.status_code, response.json()["detail"]) == (status, detail)


@pytest.mark.parametrize("method,path", sorted(_OWNER_REFUSALS))
def test_owner_only_coder_routes_let_the_owner_reach_the_handler(
        monkeypatch, method, path):
    client = _coder_app(monkeypatch, owner=True)
    status, detail = _OWNER_REFUSALS[(method, path)]
    response = client.request(method, path, json={})
    body = response.json() if response.content else {}
    assert not (response.status_code == status and body.get("detail") == detail)


_SESSION_ROUTES = sorted(
    (method, path) for (kind, method, path), labels in _CODER_PLUGIN_ROUTES.items()
    if "_bind_session" in labels)


@pytest.mark.parametrize("method,path", _SESSION_ROUTES)
def test_session_routes_hide_another_principals_session(monkeypatch, method, path):
    client = _coder_app(monkeypatch, owner=False, principal="p-other")
    response = client.request(method, path.replace("{session_id}", "s1"), json={})
    assert response.status_code == 404
    assert response.json()["detail"] == "No such session: s1"


@pytest.mark.parametrize("method,path", _SESSION_ROUTES)
def test_session_routes_hide_an_unknown_session_from_the_owner(
        monkeypatch, method, path):
    client = _coder_app(monkeypatch, owner=True)
    response = client.request(method, path.replace("{session_id}", "nope"), json={})
    assert response.status_code == 404
    assert response.json()["detail"] == "No such session: nope"


@pytest.mark.parametrize("path,key", [
    ("/api/coder/history", "authorized"),
    ("/api/coder/resumable", "resumable"),
    ("/api/coder/dormant", "projects"),
    ("/api/coder/episodes", "episodes"),
    ("/api/coder/episodes/archive", "archived"),
    ("/api/coder/hooks", "configured"),
])
def test_routes_that_answer_a_scoped_key_themselves_return_an_empty_body(
        monkeypatch, path, key):
    client = _coder_app(monkeypatch, owner=False, principal="p-other")
    response = client.get(path)
    assert response.status_code == 200
    assert key in response.json()
    assert not response.json()[key]
