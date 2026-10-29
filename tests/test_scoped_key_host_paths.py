# SPDX-License-Identifier: AGPL-3.0-or-later
"""A key without the owner scope gets no host folder layout: host paths outside
the data dir, install root, env prefix and home (a model registered in place,
any other absolute path) reach it with their leading directories replaced by
``<path>``, in a bug report's log tail, in ``/api/models``'s ``last_path`` and
in the other routes such a key reaches. The owner keeps the full paths, and an
error text shown to every caller (``scrub_paths``) hides only the data dir,
install root, env prefix and home: the owner and the local CLI see any other
folder in it unchanged."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import pathscrub
from localm.pathscrub import PATH_LABEL, scrub_host_paths
from localm.plugins.gui.web import attach_gui

_WIN = sys.platform == "win32"


@pytest.fixture(autouse=True)
def _comfyui_is_set_up(monkeypatch):
    monkeypatch.setattr("localm.media.backend_choice.comfy_is_set_up",
                        lambda *a, **k: True)


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "user" / ".localm"
    h.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "user"))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", h)
    monkeypatch.setattr(_cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", h / "registry.json")
    return h


@pytest.fixture
def outside(tmp_path):
    """A folder outside the data dir, install root, env prefix and home."""
    d = tmp_path / "elsewhere" / "bin"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def spaced(tmp_path):
    """A model folder outside the data dir, install root, env prefix and home,
    with a space in one of its names."""
    d = tmp_path / "Model Store" / "gguf"
    d.mkdir(parents=True)
    return d


def _h(key):
    return {"Authorization": f"Bearer {key}"}


def _key(scopes):
    from localm import auth
    return auth.create_key("k", scopes, allow_privileged=True)["key"]


def _register(home, entries):
    (home / "registry.json").write_text(json.dumps(entries), encoding="utf-8")


def _gui_app(tmp_path, *plugins):
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    pm = PluginManager(app, external_root=tmp_path / "noplugins")
    for p in plugins:
        pm.install(p)

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "")
    return app


def _forms(folder: Path) -> list:
    raw = str(folder)
    return [raw, raw.replace("\\", "/"), raw.replace("\\", "\\\\")]


def _assert_no_folder(text: str, folder: Path) -> None:
    for form in _forms(folder):
        assert form not in text, f"host folder {form!r} reached a scoped key"


# --------------------------------------------------------------------------- #
#  The scrubber                                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,want", [
    (r"model ready: D:\My Models\gguf\bge.gguf", r"model ready: <path>\bge.gguf"),
    (json.dumps({"p": r"D:\a b\c\x.gguf"}), json.dumps({"p": r"<path>\x.gguf"})),
    (r"copied D:\a\b.gguf to E:\c\d.gguf", r"copied <path>\b.gguf to <path>\d.gguf"),
    ("D:/x/y/z.gguf failed", "<path>/z.gguf failed"),
    (r"C:\Users\<redacted>\Downloads\m.gguf", r"<path>\m.gguf"),
    (r"see \\nas\models\q\m.gguf ok", r"see <path>\m.gguf ok"),
    (r"\\?\D:\deep\x.gguf", r"\\?\<path>\x.gguf"),
    (r"D:\x.gguf", r"<path>\x.gguf"),
    ("/mnt/models/x.gguf", "<path>/x.gguf"),
    ("PATH=/usr/bin:/usr/local/bin", "PATH=<path>/bin:<path>/bin"),
])
def test_host_paths_lose_their_folders(text, want):
    out = scrub_host_paths(text, {"mnt", "usr"})
    assert out == want
    assert scrub_host_paths(out, {"mnt", "usr"}) == out


@pytest.mark.parametrize("text", [
    r"<data>\models\x.gguf", r"<data>\\models\\x.gguf", r'File "<install>\localm\x.py"',
    "GET /api/models HTTP/1.1", "http://127.0.0.1:8080/v1/models", "~/foo/bar",
    "localm/inference/x.py", "/x.gguf", "a/b/c",
])
def test_text_that_is_not_a_host_path_is_kept(text):
    assert scrub_host_paths(text, {"mnt", "usr"}) == text


def test_an_unlistable_root_treats_every_posix_path_as_a_host_path(monkeypatch):
    def _boom(_p):
        raise OSError("no root")

    monkeypatch.setattr(pathscrub.os, "listdir", _boom)
    assert scrub_host_paths("GET /api/models/x") == "GET <path>/x"


def test_scoped_scrubber_drops_a_registry_folder_outside_the_known_roots(home, spaced):
    model = spaced / "bge-small-en-v1.5-q4_k_m.gguf"
    _register(home, {"bge": {"path": str(model), "source": "local"},
                     "inside": {"path": str(home / "models" / "i.gguf")}})
    text = f"ready: {model} and {home / 'models' / 'i.gguf'}"
    out = pathscrub.scoped_path_scrubber()(text)
    _assert_no_folder(out, spaced)
    assert f"{PATH_LABEL}{os.sep}{model.name}" in out
    assert f"<data>{os.sep}models{os.sep}i.gguf" in out


# --------------------------------------------------------------------------- #
#  POST /api/bug-report with include_log                                       #
# --------------------------------------------------------------------------- #

def test_bug_report_log_tail_holds_no_host_path_for_a_scoped_key(home, tmp_path, spaced):
    from localm.inference.http_server import create_app
    model = spaced / "bge-small-en-v1.5-q4_k_m.gguf"
    _register(home, {"bge": {"path": str(model), "source": "local",
                             "model_type": "embedding"}})
    other = (Path(r"E:\Other Stuff\cache") if _WIN else Path("/usr/lib/other-cache"))
    logs = home / "logs"
    logs.mkdir()
    (logs / f"localm_2026-10-29_120000_{os.getpid()}.log").write_text(
        f"2026-10-29 12:00:00,001 INFO     localm: embedding model ready: {model}\n"
        f"2026-10-29 12:00:01,002 WARNING  localm.media: no cache at {other / 'thing.bin'}\n",
        encoding="utf-8")
    engine = MagicMock()
    engine.display_name = "m"
    engine.loaded = True
    scoped, admin = _key(["config:write"]), _key(["admin"])
    app = create_app(engine)
    with TestClient(app) as c:
        mine = c.post("/api/bug-report", headers=_h(scoped),
                      json={"message": "embedder", "include_log": True}).json()
        saved = (home / "bug-reports" / mine["filename"]).read_text(encoding="utf-8")
        full = c.post("/api/bug-report", headers=_h(admin),
                      json={"message": "embedder", "include_log": True}).json()
    text = mine["report_markdown"]
    assert saved == text
    assert "embedding model ready:" in text, "the log tail was not attached"
    _assert_no_folder(text, spaced)
    _assert_no_folder(text, other)
    assert f"embedding model ready: {PATH_LABEL}{os.sep}{model.name}" in text
    assert f"{PATH_LABEL}{os.sep}thing.bin" in text
    assert str(model) in full["report_markdown"], "the owner's report keeps the path"


# --------------------------------------------------------------------------- #
#  GET /api/models: last_path of a missing model                               #
# --------------------------------------------------------------------------- #

def test_models_last_path_holds_no_host_path_for_a_scoped_key(home, tmp_path, spaced):
    gone = spaced / "gone.gguf"
    _register(home, {"gone": {"path": str(gone), "source": "local"}})
    app = _gui_app(tmp_path)
    scoped, admin = _key(["models:read"]), _key(["admin"])
    with TestClient(app) as c:
        mine = c.get("/api/models", headers=_h(scoped))
        full = c.get("/api/models", headers=_h(admin))
    assert mine.status_code == 200, mine.text
    row = next(m for m in mine.json()["models"] if m["name"] == "gone")
    assert row["missing"] is True
    assert row["last_path"] == f"{PATH_LABEL}{os.sep}gone.gguf"
    _assert_no_folder(mine.text, spaced)
    owner_row = next(m for m in full.json()["models"] if m["name"] == "gone")
    assert owner_row["last_path"] == str(gone)


# --------------------------------------------------------------------------- #
#  Siblings                                                                    #
# --------------------------------------------------------------------------- #

def test_rebuild_launcher_holds_no_host_path_for_a_scoped_key(home, tmp_path, outside,
                                                              monkeypatch):
    from localm import applaunch
    exe = outside / "LocaLM.exe"
    shortcut = outside / "Desktop" / "LocaLM.lnk"
    monkeypatch.setattr(applaunch, "make_launcher", lambda force=False: applaunch.LauncherResult(
        ok=True, path=exe, desktop_file=shortcut, notes=[f"wrote {exe}"]))
    app = _gui_app(tmp_path)
    scoped, admin = _key(["config:write"]), _key(["admin"])
    with TestClient(app) as c:
        mine = c.post("/api/app/rebuild-launcher", headers=_h(scoped))
        full = c.post("/api/app/rebuild-launcher", headers=_h(admin)).json()
    assert mine.status_code == 200, mine.text
    _assert_no_folder(mine.text, outside)
    body = mine.json()
    assert body["path"] == f"{PATH_LABEL}{os.sep}LocaLM.exe"
    assert body["desktop_file"] == f"{PATH_LABEL}{os.sep}LocaLM.lnk"
    assert body["notes"] == [f"wrote {PATH_LABEL}{os.sep}LocaLM.exe"]
    assert full["path"] == str(exe) and full["desktop_file"] == str(shortcut)


def test_native_models_preview_holds_no_host_path_for_a_scoped_key(home, tmp_path, outside,
                                                                    monkeypatch):
    from localm.media.sdcpp import pairing
    hint = outside / "hint.bin"
    _register(home, {"m": {"path": str(outside / "m.gguf"), "source": "local",
                           "model_type": "diffusion"}})
    monkeypatch.setattr(pairing, "preview", lambda *a, **k: {
        "family": None, "variant": None, "defaults": None,
        "error": f"a helper file could not be written: {hint}"})
    app = _gui_app(tmp_path, "image")
    scoped, admin = _key(["image"]), _key(["admin"])
    with TestClient(app) as c:
        mine = c.get("/api/imagine/native-models?model=m", headers=_h(scoped))
        full = c.get("/api/imagine/native-models?model=m", headers=_h(admin)).json()
    assert mine.status_code == 200, mine.text
    _assert_no_folder(mine.text, outside)
    assert mine.json()["preview"]["error"].endswith(f"{PATH_LABEL}{os.sep}hint.bin")
    assert full["preview"]["error"].endswith(str(hint))


def test_scoped_scrubber_without_a_readable_registry_still_scrubs_drive_paths(home, monkeypatch):
    import localm.config as _cfg

    def _boom():
        raise OSError("registry unreadable")

    monkeypatch.setattr(_cfg, "peek_registry", _boom)
    out = pathscrub.scoped_path_scrubber()(r"load D:\Somewhere Else\m.gguf")
    assert out == r"load <path>\m.gguf"


# --------------------------------------------------------------------------- #
#  Error texts shown to every caller: the owner keeps other folders            #
# --------------------------------------------------------------------------- #

def _holds(text: str, fragment: str) -> bool:
    """True when *fragment* is in *text* as written or JSON-escaped."""
    return fragment in text or json.dumps(fragment)[1:-1] in text


def _owner_form(path) -> str:
    """*path* as the owner saw it before host folders were hidden: only the
    account name is replaced (``scrub_user_paths``)."""
    return pathscrub.scrub_user_paths(str(path))


def _scoped_form(name: str) -> str:
    return f"{PATH_LABEL}{os.sep}{name}"


def test_scrub_paths_keeps_folders_outside_the_known_roots(home, outside, spaced):
    model = spaced / "bge.gguf"
    _register(home, {"bge": {"path": str(model), "source": "local"}})
    other = outside / "m.gguf"
    inside = home / "models" / "i.gguf"
    posix = "/usr/lib/x.so"
    text = f"load {other} and {model} and {inside} and {posix}"
    want = (f"load {_owner_form(other)} and {_owner_form(model)} and "
            f"<data>{os.sep}models{os.sep}i.gguf and {posix}")
    assert pathscrub.scrub_paths(text) == want
    assert pathscrub.path_scrubber()(text) == want
    scoped = pathscrub.scoped_path_scrubber()
    assert scoped(f"load {other}") == f"load {_scoped_form('m.gguf')}"
    assert scoped(f"load {model}") == f"load {_scoped_form('bge.gguf')}"


def _raising_engine(message: str):
    engine = MagicMock()

    def _chat_stream(messages, **kwargs):
        raise RuntimeError(message)
        yield  # pragma: no cover - makes this a generator function

    engine.chat_stream.side_effect = _chat_stream
    engine.count_tokens.return_value = 7
    engine.display_name = "test-model"
    engine.supports_images = False
    engine.can_be_multimodal = False
    engine.last_finish_reason = "error"
    engine.loaded = True
    return engine


@pytest.mark.parametrize("route,body", [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                              "stream": True}),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                              "stream": False}),
    ("/v1/completions", {"prompt": "hi", "stream": True}),
    ("/v1/completions", {"prompt": "hi", "stream": False}),
])
def test_inference_error_owner_keeps_a_host_folder(home, outside, route, body):
    from localm.inference.http_server import create_app
    target = outside / "m.gguf"
    app = create_app(_raising_engine(f"Failed to load model: {target}"))
    scoped, admin = _key(["chat"]), _key(["admin"])
    body = {"model": "test-model", **body}
    with TestClient(app) as c:
        mine = c.post(route, headers=_h(scoped), json=body)
        full = c.post(route, headers=_h(admin), json=body)
    assert mine.status_code == 200 and full.status_code == 200, (mine.text, full.text)
    assert "[inference error" in full.text and "[inference error" in mine.text
    assert _holds(full.text, _owner_form(target)), full.text
    _assert_no_folder(mine.text, outside)
    assert _holds(mine.text, _scoped_form("m.gguf")), mine.text


def test_rerank_unavailable_owner_keeps_a_host_folder(home, outside, monkeypatch):
    from localm.inference import reranker as rr
    from localm.inference.http_server import create_app
    target = outside / "rr.gguf"
    monkeypatch.setattr(rr, "resolve_reranker", lambda model: ("rr", str(target)))

    def _boom(path, query, texts):
        raise rr.RerankerUnavailableError(pathscrub.scrub_paths(f"could not load {target}"))

    monkeypatch.setattr(rr, "rerank", _boom)
    scoped, admin = _key(["chat"]), _key(["admin"])
    body = {"query": "q", "documents": ["a", "b"]}
    with TestClient(create_app(MagicMock())) as c:
        mine = c.post("/v1/rerank", headers=_h(scoped), json=body)
        full = c.post("/v1/rerank", headers=_h(admin), json=body)
    assert mine.status_code == 503 and full.status_code == 503, (mine.text, full.text)
    assert full.json()["detail"] == f"Reranker unavailable: could not load {_owner_form(target)}"
    _assert_no_folder(mine.text, outside)
    assert mine.json()["detail"].endswith(_scoped_form("rr.gguf"))


def test_embedding_errors_owner_keeps_a_host_folder(home, outside, monkeypatch):
    from localm.inference import embedder as emb
    from localm.inference import reranker as rr
    from localm.inference.http_server import create_app
    target = outside / "e.gguf"
    (home / "config.json").write_text(json.dumps({"embedding_model": "emb"}),
                                      encoding="utf-8")
    _register(home, {"emb2": {"path": str(target), "source": "local",
                              "model_type": "embedding"}})
    monkeypatch.setattr(emb, "embed_texts", lambda texts: None)
    monkeypatch.setattr(emb, "last_error", lambda: f"no embedding model at {target}")
    monkeypatch.setattr("localm.model_manager.registry.entry_is_reranker",
                        lambda name, entry: False)
    monkeypatch.setattr(rr, "named_embedding_path", lambda name, entry: str(target))

    def _boom(path, texts):
        raise rr.RerankerUnavailableError(f"could not load {target}")

    monkeypatch.setattr(rr, "embed", _boom)
    scoped, admin = _key(["chat"]), _key(["admin"])
    out = {}
    with TestClient(create_app(MagicMock())) as c:
        for m in ("emb", "emb2"):
            for who, key in (("owner", admin), ("scoped", scoped)):
                out[m, who] = c.post("/v1/embeddings", headers=_h(key),
                                     json={"model": m, "input": "hi"})
    assert out["emb", "owner"].status_code == 422, out["emb", "owner"].text
    assert out["emb2", "owner"].status_code == 503, out["emb2", "owner"].text
    assert str(target) in out["emb", "owner"].json()["detail"]
    assert str(target) in out["emb2", "owner"].json()["detail"]
    for m in ("emb", "emb2"):
        r = out[m, "scoped"]
        assert r.status_code == out[m, "owner"].status_code, r.text
        _assert_no_folder(r.text, outside)
        assert _scoped_form("e.gguf") in r.json()["detail"]


def test_rag_embedding_status_owner_keeps_a_host_folder(home, tmp_path, outside,
                                                        monkeypatch):
    from localm.inference import embedder as emb
    target = outside / "e.gguf"
    monkeypatch.setattr(emb, "resolve_embedding_model_path", lambda **k: None)
    monkeypatch.setattr(emb, "loaded_dim", lambda: None)
    monkeypatch.setattr(emb, "last_error", lambda: f"could not load {target}")
    monkeypatch.setattr(emb, "gpu_fallback_reason", lambda: f"GPU crash in {target}")
    app = _gui_app(tmp_path, "rag")
    scoped, admin = _key(["rag"]), _key(["admin"])
    with TestClient(app) as c:
        mine = c.get("/api/rag/embedding", headers=_h(scoped))
        full = c.get("/api/rag/embedding", headers=_h(admin))
    assert mine.status_code == 200 and full.status_code == 200, (mine.text, full.text)
    assert full.json()["error"] == f"could not load {target}"
    assert full.json()["gpu_fallback_reason"] == f"GPU crash in {target}"
    _assert_no_folder(mine.text, outside)
    assert mine.json()["error"] == f"could not load {_scoped_form('e.gguf')}"
    assert mine.json()["gpu_fallback_reason"] == f"GPU crash in {_scoped_form('e.gguf')}"


def test_embedder_reasons_keep_a_host_folder(home, outside, monkeypatch):
    from types import SimpleNamespace

    from localm.inference import embedder as emb
    target = outside / "e.gguf"
    monkeypatch.setattr(emb, "_LAST_ERROR", f"could not load {target}")
    monkeypatch.setattr(emb, "_EMBEDDER",
                        SimpleNamespace(gpu_fallback_reason=f"GPU crash in {target}"))
    assert emb.last_error() == f"could not load {_owner_form(target)}"
    assert emb.gpu_fallback_reason() == f"GPU crash in {_owner_form(target)}"


def test_speech_load_failure_owner_keeps_a_host_folder(home, tmp_path, outside,
                                                       monkeypatch):
    from localm.inference import speech
    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.jobs import JobManager
    target = outside / "tts.gguf"
    models = home / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "voice-1.gguf").write_bytes(b"GGUF")
    (models / "mmproj-voice-1.gguf").write_bytes(b"GGUF")
    _register(home, {"voice-1": {"path": str(models / "voice-1.gguf"), "source": "local",
                                 "model_type": "tts",
                                 "mmproj": str(models / "mmproj-voice-1.gguf")}})

    def _boom(*a, **k):
        raise speech.SpeechUnavailableError(pathscrub.scrub_paths(f"load failed: {target}"))

    monkeypatch.setattr(speech, "synthesize", _boom)
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("tts")
    app.state.jobs = JobManager()
    scoped, admin = _key(["tts"]), _key(["admin"])
    body = {"input": "Hello there."}
    with TestClient(app) as c:
        mine = c.post("/v1/audio/speech", headers=_h(scoped), json=body)
        full = c.post("/v1/audio/speech", headers=_h(admin), json=body)
    assert mine.status_code == 503 and full.status_code == 503, (mine.text, full.text)
    assert full.json()["detail"].endswith(f"load failed: {_owner_form(target)}")
    _assert_no_folder(mine.text, outside)
    assert mine.json()["detail"].endswith(f"load failed: {_scoped_form('tts.gguf')}")


def test_transcription_failure_owner_keeps_a_host_folder(home, tmp_path, outside,
                                                         monkeypatch):
    import localm.voice as voice
    from localm.plugins.engine import PluginManager
    target = outside / "w.bin"
    monkeypatch.setattr(voice, "prefetch_stt_model",
                        lambda allow_download=None: (False, "stubbed in tests"))
    monkeypatch.setattr(voice, "_stt_plan", lambda language: {
        "name": "base", "language": language, "timeout": 5.0,
        "local_files_only": True, "blocked_reason": None})

    def _worker(*a, **k):
        raise RuntimeError(f"engine blew up at {target}")

    monkeypatch.setattr(voice, "_run_in_worker_detailed", _worker)
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("voice")
    scoped, admin = _key(["voice"]), _key(["admin"])
    files = {"file": ("a.wav", b"RIFF\x24\x00\x00\x00WAVEfmt fake-audio-bytes", "audio/wav")}
    with TestClient(app) as c:
        mine = c.post("/v1/audio/transcriptions", headers=_h(scoped), files=files,
                      data={"model": "whisper-1"})
        full = c.post("/v1/audio/transcriptions", headers=_h(admin), files=files,
                      data={"model": "whisper-1"})
    assert mine.status_code == 502 and full.status_code == 502, (mine.text, full.text)
    assert full.json()["detail"] == f"Transcription failed: engine blew up at {_owner_form(target)}"
    _assert_no_folder(mine.text, outside)
    assert mine.json()["detail"].endswith(_scoped_form("w.bin"))


def test_debug_stacks_owner_keeps_a_host_folder(home, outside, monkeypatch):
    import traceback

    from localm import auth
    from localm.inference.http_server import create_app
    target = outside / "x.py"
    monkeypatch.setattr(traceback, "format_stack",
                        lambda frame: [f'  File "{target}", line 1, in f\n'])
    scoped = auth.create_key("k", ["chat"], fs_access="host")["key"]
    admin = _key(["admin"])
    with TestClient(create_app(MagicMock())) as c:
        mine = c.get("/debug/stacks", headers=_h(scoped))
        full = c.get("/debug/stacks", headers=_h(admin))
    assert mine.status_code == 200 and full.status_code == 200, (mine.text, full.text)
    assert _holds(full.text, _owner_form(target)), full.text
    _assert_no_folder(mine.text, outside)
    assert _holds(mine.text, _scoped_form("x.py")), mine.text


def test_cli_failed_turn_keeps_a_host_folder(home, outside):
    from localm.cli.chat import _record_failed_turn
    target = outside / "m.gguf"
    audit = MagicMock()
    _record_failed_turn(audit, None, "hi", "part", RuntimeError(f"Failed to load {target}"))
    reply = audit.llm.call_args.args[0]
    assert reply == f"part\n[inference error: Failed to load {_owner_form(target)}]"
