# SPDX-License-Identifier: AGPL-3.0-or-later
"""Privacy mode keeps no generated media unless ``privacy_keep`` lists it.

``localm.media.held`` (the folder of one running process, its removal and the
removal of what a killed process left), the image, music and video routes, the
command line, the chat REPL and the MCP tool."""

from __future__ import annotations

import contextlib
import io
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from tests import media_bytes
from localm.media import held

_REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("LOCALM_MODE", raising=False)
    monkeypatch.setattr("localm.plugins.lookup.is_available", lambda name: True)
    monkeypatch.setattr("localm.media.backend_choice.comfy_is_set_up", lambda *a, **k: True)
    for mod in ("image_gen.comfy", "music_gen.comfy", "video_gen.comfy"):
        monkeypatch.setattr(f"localm.{mod}.ensure_comfy", lambda *a, **k: (True, "up"))
    monkeypatch.setattr("localm.media.comfy_client.ensure_comfy", lambda *a, **k: (True, "up"))
    monkeypatch.setattr("localm.media.comfy_client.free_comfy_vram", lambda *a, **k: True)
    monkeypatch.setattr("localm.image_gen.comfy.free_comfy_vram", lambda *a, **k: True)
    monkeypatch.setattr("localm.vram.decide_media_swap", lambda s: False)
    monkeypatch.setattr("localm.vram.media_single_device_shortfall", lambda s: None)
    yield
    held.CLOSING.clear()
    held.discard_all()


def _config(**kw):
    from localm.config import update_config
    update_config(lambda c: c.update(kw))


def _home() -> Path:
    from localm.config import home_dir
    return home_dir()


# --------------------------------------------------------------------------- #
#  the setting                                                                #
# --------------------------------------------------------------------------- #

def test_privacy_keeps_nothing_by_default_and_media_when_listed():
    from localm.audit import privacy_keeps
    assert privacy_keeps("media") is False
    _config(privacy_keep=["media"])
    assert privacy_keeps("media") is True
    assert privacy_keeps("chats") is False


@pytest.mark.parametrize("value", ["media", True, {"media": True}, None])
def test_a_malformed_setting_keeps_nothing(value):
    from localm.audit import privacy_keeps
    _config(privacy_keep=value)
    assert privacy_keeps("media") is False


def test_an_unreadable_config_keeps_nothing():
    from localm import config
    from localm.audit import privacy_keeps
    _config(privacy_keep=["media"])
    config.CONFIG_FILE.write_text("{ not json", encoding="utf-8")
    (config.CONFIG_FILE.with_suffix(".json.bak")).write_text("{ nor this", encoding="utf-8")
    assert privacy_keeps("media") is False


def test_the_setting_accepts_its_options_only():
    from localm.settings_schema import validate_update
    assert validate_update({"privacy_keep": "media"})["privacy_keep"] == ["media"]
    assert validate_update({"privacy_keep": []})["privacy_keep"] == []
    with pytest.raises(ValueError, match="is not one of"):
        validate_update({"privacy_keep": ["media", "chats"]})


def test_the_setting_is_the_owners_and_sits_with_the_session_modes():
    from localm.settings_schema import CORE_FIELDS
    fields = {f.key: f for f in CORE_FIELDS}
    assert fields["privacy_keep"].admin_only
    assert fields["privacy_keep"].group == fields["mode"].group == "Privacy"


# --------------------------------------------------------------------------- #
#  the held folder                                                            #
# --------------------------------------------------------------------------- #

def test_a_held_folder_is_removed_with_everything_in_it():
    d = held.folder("image")
    (d / "a.png").write_bytes(b"x")
    assert held.is_held(d / "a.png") and held.files("image") == [d / "a.png"]
    assert held.discard_all() is True
    assert not held.root_dir().exists()
    assert held.files("image") == []


_HOLDER = """
import os, sys
from localm.media import held
d = held.folder("image")
(d / "left.png").write_bytes(b"x")
held.note_comfy_job("t", "http://127.0.0.1:8188", "job-1")
print(os.getpid(), d.parent, flush=True)
sys.stdin.readline()
"""


@contextlib.contextmanager
def _other_process():
    env = dict(os.environ, LOCALM_HOME=str(_home()), PYTHONPATH=str(_REPO))
    proc = subprocess.Popen([sys.executable, "-c", _HOLDER], env=env, text=True,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    pid, folder = proc.stdout.readline().strip().split(" ", 1)

    def kill():
        """Kill the process that holds the folder, which on Windows is the child
        of the venv launcher ``proc`` stands for."""
        with contextlib.suppress(OSError):
            os.kill(int(pid), signal.SIGTERM if os.name == "nt" else signal.SIGKILL)
        proc.wait(30)

    try:
        yield kill, Path(folder)
    finally:
        kill()


def test_a_killed_process_leaves_its_media_until_the_next_start_removes_it():
    with _other_process() as (kill, folder):
        assert held.sweep_leftovers() == 0
        assert (folder / "image" / "left.png").is_file()
        kill()
        assert (folder / "image" / "left.png").is_file()
        assert held.sweep_leftovers() == 0
        assert not (folder / "image").exists()
        assert not folder.with_name(folder.name + ".lock").exists()
        told: list = []
        held.sweep_leftovers(lambda *a: told.append(a))
        held.sweep_leftovers(lambda *a: told.append(a))
        assert len(told) == 2 and told[0] == told[1]
        assert "http://127.0.0.1:8188" in told[0][0] and "job-1" in told[0][0]
        held.dismiss_note(told[0][1])
        assert not held.root_dir().exists()


def test_a_sweep_holds_the_lock_of_the_folder_it_removes(monkeypatch):
    with _other_process() as (kill, folder):
        kill()
        lock = folder.with_name(folder.name + ".lock")
        real, taken = held._clear, []

        def clear(path):
            with open(lock, "a+b") as f:
                taken.append(held._try_lock(f))
            return real(path)
        monkeypatch.setattr(held, "_clear", clear)
        held.sweep_leftovers()
    assert taken == [False]


def test_the_folder_of_this_process_is_made_again_when_it_vanished():
    import shutil
    d = held.folder("image")
    shutil.rmtree(d.parent)
    assert held.folder("image") == d and d.is_dir()
    shutil.rmtree(d.parent)
    note = held.note_comfy_job("t", "http://127.0.0.1:8188", "job-1")
    assert note is not None and note.is_file()


def _locked(monkeypatch, note):
    real = Path.unlink

    def unlink(self, missing_ok=False):
        if self == note:
            raise PermissionError(32, "in use")
        return real(self, missing_ok=missing_ok)
    monkeypatch.setattr(Path, "unlink", unlink)


def test_a_note_that_cannot_be_deleted_is_not_a_cut_off_job(monkeypatch):
    note = held.note_comfy_job("t", "http://127.0.0.1:8188", "job-1")
    other = held.note_comfy_job("u", "http://127.0.0.1:8188", "job-2")
    with monkeypatch.context() as m:
        _locked(m, note)
        held.forget_note(note)
        assert note.is_file()
        held.discard_all()
        assert note.is_file()
        told: list = []
        held.sweep_leftovers(lambda *a: told.append(a[1]))
    assert told == [other]
    held.sweep_leftovers()
    assert not note.exists() and other.is_file()
    held.dismiss_note(other)
    assert not held.root_dir().exists()


def test_a_note_that_could_not_be_deleted_goes_when_localm_stops(monkeypatch):
    note = held.note_comfy_job("t", "http://127.0.0.1:8188", "job-1")
    with monkeypatch.context() as m:
        _locked(m, note)
        held.forget_note(note)
    assert held.discard_all() is True
    assert not held.root_dir().exists()


def test_a_live_folder_whose_lock_the_listing_missed_is_left_alone(monkeypatch):
    with _other_process() as (_kill, folder):
        real = Path.iterdir
        monkeypatch.setattr(Path, "iterdir", lambda self: [
            p for p in real(self) if p.suffix != ".lock"])
        assert held.sweep_leftovers() == 0
        assert (folder / "image" / "left.png").is_file()


def test_what_is_no_held_folder_is_neither_removed_nor_reported(tmp_path, caplog):
    held.root_dir().mkdir(parents=True)
    stray = held.root_dir() / "Thumbs.db"
    stray.write_bytes(b"x")
    target = tmp_path / "target"
    target.mkdir()
    link = held.root_dir() / "1-link"
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)
    with caplog.at_level("WARNING", logger="localm"):
        assert held.sweep_leftovers() == 0
    assert stray.is_file() and target.is_dir() and os.path.lexists(link)
    assert caplog.records == []


@pytest.mark.skipif(os.name == "nt", reason="an open file cannot be removed on Windows")
def test_a_lock_file_that_was_replaced_is_not_the_one_locked(monkeypatch):
    real = held._try_lock

    def lock_after_a_sweep(f):
        os.unlink(f.name)
        Path(f.name).write_bytes(b"")
        return real(f)
    monkeypatch.setattr(held, "_try_lock", lock_after_a_sweep)
    with pytest.raises(OSError, match="cannot lock"):
        held.folder("image")


def test_the_entries_of_files_that_were_never_kept_go_with_them():
    from localm.media import gallery
    stray = held.root_dir() / "123-abc" / "image"
    stray.mkdir(parents=True)
    kept = _home() / "gui_images"
    kept.mkdir()
    for name in ("gone.png", "saved.png"):
        (stray / name).write_bytes(b"x")
        gallery.stamp_owner("image", name, "key:ka")
    (kept / "saved.png").write_bytes(b"x")
    gallery.stamp_owner("image", "elsewhere.png", "key:ka")
    assert held.sweep_leftovers() == 1
    assert gallery.owner_of("image", "gone.png") is None
    assert gallery.owner_of("image", "saved.png") == "key:ka"
    assert gallery.owner_of("image", "elsewhere.png") == "key:ka"


def test_a_result_whose_folder_is_gone_gets_no_entry():
    from localm.media import gallery
    d = held.folder("image")
    held.discard_all()
    assert gallery.record_owner("image", d / "late.png", "key:ka") is None
    assert not (_home() / "gallery_index").exists()


def test_a_copy_that_fails_leaves_the_original_and_no_file(tmp_path, monkeypatch):
    d = held.folder("image")
    (d / "a.png").write_bytes(b"xyz")

    def full(src, dst):
        dst.write(b"x")
        raise OSError("No space left on device")
    monkeypatch.setattr(held.shutil, "copyfileobj", full)
    with pytest.raises(OSError, match="No space"):
        held.save_copy(d / "a.png", tmp_path / "a.png")
    assert (d / "a.png").read_bytes() == b"xyz" and not (tmp_path / "a.png").exists()
    (tmp_path / "b.png").write_bytes(b"mine")
    with pytest.raises(FileExistsError):
        held.save_copy(d / "a.png", tmp_path / "b.png")
    assert (tmp_path / "b.png").read_bytes() == b"mine"


def test_a_copy_that_differs_is_not_kept(tmp_path, monkeypatch):
    d = held.folder("image")
    (d / "a.png").write_bytes(b"xyz")
    monkeypatch.setattr(held.shutil, "copyfileobj", lambda src, dst: dst.write(b"xy"))
    with pytest.raises(OSError, match="differs"):
        held.save_copy(d / "a.png", tmp_path / "a.png")
    assert not (tmp_path / "a.png").exists()


def test_the_root_stays_while_another_process_holds_media():
    with _other_process() as (_kill, folder):
        held.folder("music")
        assert held.discard_all() is True
        assert (folder / "image" / "left.png").is_file()


def test_a_folder_without_a_lock_file_is_a_leftover():
    stray = held.root_dir() / "123-abc" / "video"
    stray.mkdir(parents=True)
    (stray / "a.mp4").write_bytes(b"x")
    assert held.sweep_leftovers() == 1
    assert not stray.parent.exists()


def test_a_file_that_cannot_be_removed_is_reported_and_tried_again(monkeypatch, caplog):
    d = held.folder("image")
    (d / "a.png").write_bytes(b"x")
    monkeypatch.setattr(held, "_REMOVE_RETRY_SECONDS", 0)
    import shutil
    real = shutil.rmtree
    monkeypatch.setattr(held.shutil, "rmtree",
                        lambda p: (_ for _ in ()).throw(PermissionError("in use")))
    with caplog.at_level("WARNING", logger="localm"):
        assert held.discard_all() is False
    assert any("could not be removed" in r.getMessage() for r in caplog.records)
    assert (d / "a.png").is_file()
    monkeypatch.setattr(held.shutil, "rmtree", real)
    assert held.sweep_leftovers() == 1
    assert not d.exists()


def test_stopping_the_server_removes_held_media():
    from localm.inference import http_server
    d = held.folder("image")
    (d / "a.png").write_bytes(b"x")
    with patch.object(http_server, "_engines", {}), patch.object(http_server, "_engine", None):
        http_server._shutdown_teardown()
    assert not d.exists()


def test_a_start_tells_the_owner_of_a_comfyui_job_that_was_cut_off(capsys):
    from localm.inference import http_server
    from localm.jobmanager import JobManager
    note = held.root_dir() / "123-abc" / "comfy" / "t"
    note.parent.mkdir(parents=True)
    note.write_text('["http://127.0.0.1:8188", "job-1"]', encoding="utf-8")
    held.sweep_leftovers()
    written: list = []
    for _start in range(2):
        jobs = JobManager(reconcile=False)
        jobs._TTL_S = -1
        jobs._store = SimpleNamespace(write=written.append, clear=lambda: None)
        with patch("localm.auth.any_key_configured", return_value=True):
            http_server._remove_leftovers(SimpleNamespace(state=SimpleNamespace(jobs=jobs)))
        jobs._gc()
        jobs._persist()
        rows = jobs.snapshot()
        assert len(rows) == 1 and rows[0]["status"] == "interrupted"
        assert "job-1" in rows[0]["label"] and rows[0]["note"] is True
        assert jobs.snapshot(visible=lambda owner: owner in (None, "key:ka")) == []
        assert "cut off" in capsys.readouterr().out
    assert written == []
    job = jobs.get(rows[0]["id"])
    assert jobs.note_shown(job) is True and jobs.note_shown(job) is False
    assert "note" not in jobs.snapshot()[0] and not held.root_dir().exists()
    jobs._gc()
    assert jobs.snapshot() == []


def test_the_app_says_that_it_showed_a_note(server):
    app, client = server
    shown: list = []
    app.state.jobs.add_note("imagine", "A private job was cut off.", None,
                            lambda: shown.append(1))
    op = next(o for o in client.get("/api/activity").json()["operations"] if o.get("note"))
    url = f"/api/jobs/{op['id']}/cancel"
    assert client.post(url).json() == {"status": "shown"} and shown == [1]
    assert client.post(url).json()["status"] == "not_running" and shown == [1]


def test_a_damaged_note_is_still_told():
    note = held.root_dir() / "123-abc" / "comfy" / "t"
    note.parent.mkdir(parents=True)
    note.write_bytes(b"[" * 4000)
    told: list = []
    held.sweep_leftovers(lambda *a: told.append(a))
    assert len(told) == 1 and "unknown" in told[0][0]


def test_starting_the_server_removes_what_a_killed_one_left():
    from localm.inference.http_server import create_app
    stray = held.root_dir() / "123-abc" / "image"
    stray.mkdir(parents=True)
    (stray / "a.png").write_bytes(b"x")
    engine = MagicMock()
    engine.display_name = "test-model"
    with TestClient(create_app(engine)):
        assert not stray.parent.exists()


# --------------------------------------------------------------------------- #
#  the routes                                                                 #
# --------------------------------------------------------------------------- #

BYTES = {"image": media_bytes.png_bytes, "music": media_bytes.flac_bytes,
         "video": media_bytes.mp4_bytes}


KINDS = {
    "image": ("/api/imagine", {"prompt": "a secret cat"}, "localm.image_gen.comfy.generate_image",
              "gui_images", "images"),
    "music": ("/api/music", {"tags": "a secret song", "duration_seconds": 5},
              "localm.music_gen.generate_music", "gui_music", "tracks"),
    "video": ("/api/video", {"prompt": "a secret clip", "seconds": 1},
              "localm.video_gen.generate_video", "gui_video", "videos"),
}


@pytest.fixture
def server(tmp_path):
    from localm.inference.http_server import create_app
    from localm.plugins.engine import PluginManager
    engine = MagicMock()
    engine.display_name = "test-model"
    app = create_app(engine)
    manager = PluginManager(app, external_root=tmp_path / "noplugins")
    for name in KINDS:
        manager.install(name)
    app.state.self_url = "http://127.0.0.1:8642/v1"
    with TestClient(app) as client:
        client.headers["Authorization"] = f"Bearer {app.state.shell_token}"
        yield app, client


def _generate(server, kind, headers=None):
    app, client = server
    path, body, target, _dir, _key = KINDS[kind]
    calls = []

    def fake(prompt, out, **kwargs):
        calls.append((Path(out), kwargs))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(BYTES[kind]())
        return True, f"Saved to {out}"

    with patch(target, fake):
        r = client.post(path, json=body, headers=headers or {})
        assert r.status_code == 200, r.text
        job = app.state.jobs.get(r.json()["job_id"])
        deadline = time.monotonic() + 30
        while job.finished_at is None and time.monotonic() < deadline:
            time.sleep(0.02)
    assert job.status == "done", [e.get("text") for e in job._history]
    return job, calls[0]


def _media_files() -> list[str]:
    return sorted(str(p.relative_to(_home())) for p in _home().rglob("*")
                  if p.suffix in (".png", ".flac", ".mp4") or p.name.endswith(".json")
                  and "gallery_index" in str(p))


@pytest.mark.parametrize("kind", list(KINDS))
def test_privacy_mode_shows_the_result_and_keeps_nothing(server, kind):
    _app, client = server
    path, _body, _t, folder, key = KINDS[kind]
    job, (out, kwargs) = _generate(server, kind)
    assert held.is_held(out) and kwargs["write_sidecar"] is False
    assert any(e.get("type") == "held" for e in job._history)
    shown = client.get(f"{path}/file/{job.result}")
    assert shown.status_code == 200 and shown.headers["cache-control"] == "no-store"
    listing = client.get(f"{path}/history").json()
    assert listing[key] == []
    assert [i["name"] for i in listing["held"]] == [job.result]
    assert not (_home() / folder).exists() and not (_home() / "gallery_index").exists()
    held.discard_all()
    assert _media_files() == []
    assert client.get(f"{path}/file/{job.result}").status_code == 404


@pytest.mark.parametrize("kind", list(KINDS))
def test_with_media_kept_the_result_is_in_the_history(server, kind):
    _app, client = server
    _config(privacy_keep=["media"])
    path, _body, _t, folder, key = KINDS[kind]
    job, (out, kwargs) = _generate(server, kind)
    assert out.parent == _home() / folder and kwargs["write_sidecar"] is False
    listing = client.get(f"{path}/history").json()
    assert [i["name"] for i in listing[key]] == [job.result] and listing["held"] == []
    assert not any(e.get("type") == "held" for e in job._history)


@pytest.mark.parametrize("kind", list(KINDS))
def test_outside_privacy_mode_nothing_changes(server, kind, monkeypatch):
    _app, client = server
    monkeypatch.setenv("LOCALM_MODE", "full")
    path, _body, _t, folder, key = KINDS[kind]
    job, (out, kwargs) = _generate(server, kind)
    assert out.parent == _home() / folder and kwargs["write_sidecar"] is True
    listing = client.get(f"{path}/history").json()
    assert [i["name"] for i in listing[key]] == [job.result] and listing["held"] == []
    assert "cache-control" not in client.get(f"{path}/file/{job.result}").headers


def _no_exec(monkeypatch):
    from localm.inference import http_server
    monkeypatch.setattr(http_server, "_execv_restart", lambda *a, **k: None)
    monkeypatch.setattr(http_server, "_mark_fds_noninheritable", lambda: None)
    monkeypatch.setattr(http_server, "_set_restart_env", lambda: None)
    monkeypatch.setattr(http_server, "_finish_pending_runtime_update", lambda: False)
    monkeypatch.setattr(http_server, "_finish_comgr_temp", lambda: None)


def _ended(job):
    deadline = time.monotonic() + 20
    while job.finished_at is None and time.monotonic() < deadline:
        time.sleep(0.02)
    return job.status


@pytest.mark.parametrize("stop", ["_shutdown_teardown", "_do_restart"])
@pytest.mark.parametrize("kind", list(KINDS))
def test_a_stop_ends_a_private_job_and_leaves_nothing(server, kind, stop, monkeypatch):
    """The job is ended before held media is removed, also when its writer finishes
    the file after the cancel."""
    from localm.inference import http_server
    app, client = server
    path, body, target, folder, _key = KINDS[kind]
    monkeypatch.setattr(f"localm.plugins.builtin.{kind}.plug.principal_id", lambda r: "key:ka")
    _no_exec(monkeypatch)
    started = threading.Event()
    cancelled = []

    def fake(prompt, out, cancel_check=None, **kwargs):
        started.set()
        deadline = time.monotonic() + 20
        while not cancel_check() and time.monotonic() < deadline:
            time.sleep(0.02)
        cancelled.append(cancel_check())
        time.sleep(0.5)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(BYTES[kind]())
        return True, "saved"

    with patch(target, fake), patch.object(http_server, "_engines", {}), \
            patch.object(http_server, "_engine", None):
        job = app.state.jobs.get(client.post(path, json=body).json()["job_id"])
        assert started.wait(20) and held.private_run_wait() == held._END_LIMIT_S
        getattr(http_server, stop)()
    assert cancelled == [True] and held.private_run_wait() == 0
    assert _ended(job) == "cancelled"
    assert _media_files() == [] and not (_home() / "gallery_index").exists()
    assert not (_home() / folder).exists() and not held.CLOSING.is_set()


@pytest.mark.parametrize("kind,stop,mode,keep", [
    *[(kind, stop, "log", []) for kind in KINDS for stop in ("_shutdown_teardown", "_do_restart")],
    ("image", "_shutdown_teardown", "privacy", ["media"]),
    ("image", "_shutdown_teardown", "full", [])])
def test_a_stop_neither_ends_nor_awaits_a_job_whose_result_is_kept(
        server, kind, stop, mode, keep, monkeypatch):
    from localm.inference import http_server
    app, client = server
    path, body, target, _folder, _key = KINDS[kind]
    _config(mode=mode, privacy_keep=keep)
    _no_exec(monkeypatch)
    monkeypatch.setattr(held, "_END_LIMIT_S", 60.0)
    started, finish = threading.Event(), threading.Event()
    cancelled = []

    def fake(prompt, out, cancel_check=None, **kwargs):
        started.set()
        while not finish.wait(0.02):
            cancelled.append(cancel_check())
        cancelled.append(cancel_check())
        Path(out).write_bytes(BYTES[kind]())
        return True, "saved"

    stopped = threading.Event()
    with patch(target, fake), patch.object(http_server, "_engines", {}), \
            patch.object(http_server, "_engine", None):
        job = app.state.jobs.get(client.post(path, json=body).json()["job_id"])
        assert started.wait(20) and held.private_run_wait() == 0
        t = threading.Thread(target=lambda: (getattr(http_server, stop)(), stopped.set()))
        t.start()
        try:
            assert stopped.wait(40)
            assert job.status == "running"
        finally:
            finish.set()
            t.join(30)
        assert _ended(job) == "done"
    assert not any(cancelled)


def test_a_job_that_turns_private_before_the_stop_is_ended(server, monkeypatch):
    from localm.inference import http_server
    app, client = server
    _config(mode="log")
    started = threading.Event()

    def fake(prompt, out, cancel_check=None, **kwargs):
        started.set()
        deadline = time.monotonic() + 20
        while not cancel_check() and time.monotonic() < deadline:
            time.sleep(0.02)
        return not cancel_check(), "Generation cancelled."
    with patch(KINDS["image"][2], fake), patch.object(http_server, "_engines", {}), \
            patch.object(http_server, "_engine", None):
        job = app.state.jobs.get(client.post("/api/imagine", json={"prompt": "x"}).json()["job_id"])
        assert started.wait(20) and held.private_run_wait() == 0
        _config(mode="privacy")
        assert held.private_run_wait() == held._END_LIMIT_S
        http_server._shutdown_teardown()
        assert held.private_run_wait() == 0 and _ended(job) == "cancelled"


def test_a_job_that_only_reloads_the_chat_model_is_left_alone(server, monkeypatch):
    app, client = server
    reloading, go_on = threading.Event(), threading.Event()

    @contextlib.contextmanager
    def restore(*a, **k):
        yield
        reloading.set()
        go_on.wait(20)
    monkeypatch.setattr("localm.plugins.builtin.image.plug.restore_chat_after_media", restore)

    def fake(prompt, out, **kwargs):
        Path(out).write_bytes(media_bytes.png_bytes())
        return True, "saved"
    with patch(KINDS["image"][2], fake):
        job = app.state.jobs.get(client.post("/api/imagine", json={"prompt": "x"}).json()["job_id"])
        assert reloading.wait(20)
        assert held.private_run_wait() == 0 and held.end_private_runs() is True
        assert not job.cancel_requested
        go_on.set()
        assert _ended(job) == "done"


def test_a_restart_that_fails_lets_media_jobs_run_again(monkeypatch):
    from localm.inference import http_server
    _no_exec(monkeypatch)
    closing = []

    def refuse(*a, **k):
        closing.append(held.CLOSING.is_set())
        raise OSError("cannot start")
    monkeypatch.setattr(http_server, "_execv_restart", refuse)
    with patch.object(http_server, "_engines", {}), patch.object(http_server, "_engine", None), \
            pytest.raises(OSError, match="cannot start"):
        http_server._do_restart()
    assert closing == [True] and not held.CLOSING.is_set()
    assert held.folder("image").is_dir()


@pytest.mark.parametrize("stop", ["_shutdown_teardown", "_do_restart"])
def test_held_media_goes_before_the_gpu_temp_folders(stop, monkeypatch):
    from localm.inference import http_server
    d = held.folder("image")
    _no_exec(monkeypatch)
    there = []
    monkeypatch.setattr(http_server, "_finish_comgr_temp", lambda: there.append(d.exists()))
    with patch.object(http_server, "_engines", {}), patch.object(http_server, "_engine", None):
        getattr(http_server, stop)()
    assert there == [False]


def test_the_last_gpu_temp_cleanup_does_not_wait_for_a_running_probe(monkeypatch):
    from localm import comgr_temp
    from localm.inference.backends.llamacpp import _loader
    killed = []
    monkeypatch.setattr(_loader, "_kill_and_clear_probe", lambda: killed.append(1))
    done = threading.Event()
    with _loader._PROBE_LOCK:
        t = threading.Thread(target=lambda: (comgr_temp.finish(), done.set()), daemon=True)
        t.start()
        assert done.wait(15)
    assert killed == []


def test_the_stop_route_says_how_long_a_private_generation_may_take(server, monkeypatch):
    from localm import instances
    from localm.inference import http_server
    _app, client = server
    asked = []
    monkeypatch.setattr(http_server, "_do_shutdown", lambda **k: asked.append(1))
    with held.run(lambda: False):
        reply = client.post("/v1/server/shutdown")
        assert reply.json() == {"stopping": True}
        assert instances.shutdown_wait(reply) == 10
    with held.run(lambda: True):
        reply = client.post("/v1/server/shutdown")
        assert reply.json() == {"stopping": True, "wait_s": 12.0}
        assert instances.shutdown_wait(reply) == 22
    deadline = time.monotonic() + 10
    while len(asked) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert asked == [1, 1]
    for value in ("x", None, [], -5, float("nan"), 1e9):
        assert 10 <= instances.shutdown_wait(SimpleNamespace(
            json=lambda v=value: {"wait_s": v})) <= 70

    def nested():
        import json
        return json.loads("[" * 200_000)
    assert instances.shutdown_wait(SimpleNamespace(json=nested)) == 10


def test_no_private_job_starts_and_no_chat_model_reloads_while_localm_stops(server):
    app, client = server
    from localm import vram
    held.CLOSING.set()
    with patch(KINDS["image"][2]) as gen, patch("localm.selfclient.self_request") as req:
        job = app.state.jobs.get(client.post("/api/imagine", json={"prompt": "x"}).json()["job_id"])
        deadline = time.monotonic() + 20
        while job.finished_at is None and time.monotonic() < deadline:
            time.sleep(0.02)
        vram.reload_chat_after_media(MagicMock(), "http://127.0.0.1:1/v1", {"reload_after": True},
                                     MagicMock(), "image")
    gen.assert_not_called()
    req.assert_not_called()
    assert job.status == "failed" and not held.root_dir().exists()


def test_a_job_that_does_not_end_is_reported_and_stops_nothing_else(monkeypatch, caplog):
    monkeypatch.setattr(held, "_END_LIMIT_S", 0.1)
    with caplog.at_level("WARNING", logger="localm"), held.run(lambda: True):
        assert held.end_private_runs() is False
    assert any("did not end within" in r.getMessage() for r in caplog.records)


def test_the_native_worker_is_ended_soon_when_localm_stops(monkeypatch):
    from localm.media.sdcpp import runner
    monkeypatch.setattr(runner, "STOPPING_GRACE", 0.3)
    r = runner.SdRunner()
    r._proc = MagicMock()
    r._proc.is_alive.return_value = True
    r._resp_q, r._cancel = runner._queue.Queue(), threading.Event()
    ended = []
    monkeypatch.setattr(r, "shutdown", lambda grace=None: ended.append(grace))
    held.CLOSING.set()
    with pytest.raises(runner.SdCancelled, match="worker was stopped"):
        r._wait("generate", 60, cancel_check=lambda: True)
    assert ended == [0]


def _acl(path: Path) -> str:
    if os.name != "nt":
        return oct(path.stat().st_mode)
    out = subprocess.run(["icacls", path.name], cwd=path.parent, capture_output=True, text=True,
                         creationflags=subprocess.CREATE_NO_WINDOW).stdout
    return out.replace(path.name, "")


def test_a_moved_held_file_has_the_rights_of_its_new_folder(server, tmp_path):
    _app, client = server
    job, (out, _k) = _generate(server, "image")
    dest = tmp_path / "saved"
    r = client.post(f"/api/imagine/file/{job.result}/move", json={"dest": str(dest)})
    assert r.status_code == 200, r.text
    (dest / "plain.png").write_bytes(b"x")
    assert _acl(dest / job.result).split() == _acl(dest / "plain.png").split()


def test_a_held_file_that_cannot_be_moved_stays(server, tmp_path, monkeypatch):
    _app, client = server
    job, (out, _k) = _generate(server, "image")

    def full(src, dst):
        raise OSError("No space left on device")
    monkeypatch.setattr(held.shutil, "copyfileobj", full)
    r = client.post(f"/api/imagine/file/{job.result}/move", json={"dest": str(tmp_path / "s")})
    assert r.status_code == 400 and "No space" in r.text
    assert out.is_file() and list((tmp_path / "s").iterdir()) == []
    r = client.post(f"/api/imagine/file/{job.result}/move",
                    json={"dest": str(_home() / "gui_images")})
    assert r.status_code == 400 and held.tracks("image", job.result)
    assert not (_home() / "gallery_index").exists() or "{}" == (
        _home() / "gallery_index" / "image.json").read_text()


@pytest.mark.parametrize("was_held", [True, False])
def test_a_move_never_drops_the_row_of_a_file_that_came_first(server, was_held, monkeypatch):
    from localm.media import gallery
    _app, client = server
    if not was_held:
        _config(privacy_keep=["media"])
    job, (out, _k) = _generate(server, "image")
    target = _home() / "gui_video" / job.result
    target.parent.mkdir(parents=True, exist_ok=True)
    real = Path.exists

    def first(self):
        if self == target and not real(self):
            target.write_bytes(b"theirs")
            gallery.stamp_owner("video", target.name, "key:kb")
            return False
        return real(self)
    with monkeypatch.context() as m:
        m.setattr(Path, "exists", first)
        r = client.post(f"/api/imagine/file/{job.result}/move", json={"dest": str(target.parent)})
    assert r.status_code == 400, r.text
    assert target.read_bytes() == b"theirs" and out.is_file()
    assert gallery.owner_of("video", target.name) == "key:kb"


def test_a_kept_file_moved_to_another_gallery_belongs_to_its_owner_there(server, monkeypatch):
    from localm.media import gallery
    _app, client = server
    _config(privacy_keep=["media"])
    monkeypatch.setattr("localm.plugins.builtin.image.plug.principal_id", lambda r: "key:ka")
    monkeypatch.setattr("localm.inference.http_server.job_owner_ok", lambda r, owner: True)
    job, (out, _k) = _generate(server, "image")
    data = out.read_bytes()
    r = client.post(f"/api/imagine/file/{job.result}/move",
                    json={"dest": str(_home() / "gui_video")})
    assert r.status_code == 200, r.text
    assert (_home() / "gui_video" / job.result).read_bytes() == data and not out.exists()
    assert gallery.owner_of("video", job.result) == "key:ka"
    assert gallery.owner_of("image", job.result) is None


def test_a_result_made_before_any_key_is_the_owners_once_there_is_one(monkeypatch):
    from localm import auth
    from localm.media import gallery
    d = held.folder("image")
    (d / "a.png").write_bytes(b"x")
    gallery.record_owner("image", d / "a.png", None)
    monkeypatch.setattr(auth, "any_key_configured", lambda: False)
    assert gallery.owner_of("image", "a.png") is None
    monkeypatch.setattr(auth, "any_key_configured", lambda: True)
    assert gallery.owner_of("image", "a.png") == held.OWNER_ONLY
    assert held.owner_of("image", "untracked.png") is None
    monkeypatch.setattr("localm.inference.http_server.caller_is_admin_key", lambda r: False)
    monkeypatch.setattr("localm.inference.http_server.principal_id", lambda r: "key:kb")
    assert gallery.held_items(None, "image", str) == []


@pytest.mark.parametrize("kind", list(KINDS))
def test_a_result_moved_into_the_gallery_keeps_its_owner(server, kind, monkeypatch):
    from localm.media import gallery
    _app, client = server
    monkeypatch.setattr(f"localm.plugins.builtin.{kind}.plug.principal_id", lambda r: "key:ka")
    monkeypatch.setattr("localm.inference.http_server.job_owner_ok", lambda r, owner: True)
    path, _body, _t, folder, key = KINDS[kind]
    job, _c = _generate(server, kind)
    r = client.post(f"{path}/file/{job.result}/move", json={"dest": str(_home() / folder)})
    assert r.status_code == 200, r.text
    assert gallery.owner_of(kind, job.result) == "key:ka" and not held.tracks(kind, job.result)
    listing = client.get(f"{path}/history").json()
    assert [i["name"] for i in listing[key]] == [job.result] and listing["held"] == []


def test_a_result_written_after_the_mode_turned_private_is_not_kept():
    from localm.media import gallery
    d = _home() / "gui_images"
    d.mkdir(parents=True)
    for name in ("a.png", "a.png.json", "b.png"):
        (d / name).write_bytes(b"x")
    (d / "b.png.json").write_bytes(b"x")
    job = MagicMock()
    settle = gallery.settle
    assert settle(job, "image", d / "b.png", None, True, False) == (d / "b.png", "")
    assert (d / "b.png.json").is_file()
    assert settle(job, "image", d / "b.png", None, True, True) == (d / "b.png", "")
    moved, warning = settle(job, "image", d / "a.png", None, False, True)
    assert held.is_held(moved) and moved.read_bytes() == b"x" and warning == ""
    assert list(held.root_dir().glob("*/comfy/*")) == []
    assert [p.name for p in d.iterdir()] == ["b.png"]
    held.CLOSING.set()
    with pytest.raises(OSError, match="stopping"):
        settle(job, "image", d / "b.png", None, False, True)
    assert list(d.iterdir()) == []


def _open_elsewhere(monkeypatch, *paths):
    real = Path.unlink

    def unlink(self, missing_ok=False):
        if self in paths:
            raise PermissionError(32, "in use")
        return real(self, missing_ok=missing_ok)
    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(held, "_REMOVE_RETRY_SECONDS", 0)


@pytest.mark.parametrize("copies", [True, False])
def test_a_file_open_elsewhere_after_the_switch_is_said_and_removed_later(monkeypatch, copies):
    from localm.media import gallery
    d = _home() / "gui_images"
    d.mkdir(parents=True)
    (d / "a.png").write_bytes(b"x")
    (d / "a.png.json").write_text('{"prompt": "SECRET"}')
    job = MagicMock()
    with monkeypatch.context() as m:
        _open_elsewhere(m, d / "a.png")
        if copies:
            _moved, said = gallery.settle(job, "image", d / "a.png", "key:ka", False, True)
        else:
            m.setattr(held.shutil, "copyfileobj",
                      lambda s, t: (_ for _ in ()).throw(OSError("disk full")))
            with pytest.raises(OSError, match="disk full") as e:
                gallery.settle(job, "image", d / "a.png", "key:ka", False, True)
            said = str(e.value)
        assert "WARNING" in said and str(d / "a.png") in said
        assert [p.name for p in d.iterdir()] == ["a.png"]
        held.forget("image", "a.png")
        assert gallery.owner_of("image", "a.png") == "key:ka"
        held.discard_all()
        assert (d / "a.png").is_file()
    held.sweep_leftovers()
    assert list(d.iterdir()) == [] and gallery.owner_of("image", "a.png") is None
    assert not held.root_dir().exists()


def test_a_sidecar_open_elsewhere_is_said_and_a_file_saved_since_is_kept(monkeypatch):
    from localm.media import gallery
    d = _home() / "gui_images"
    d.mkdir(parents=True)
    (d / "a.png").write_bytes(b"x")
    (d / "a.png.json").write_text('{"prompt": "SECRET"}')
    with monkeypatch.context() as m:
        _open_elsewhere(m, d / "a.png.json")
        moved, said = gallery.settle(MagicMock(), "image", d / "a.png", None, False, True)
        assert str(d / "a.png.json") in said and str(d / "a.png") + " " not in said
    time.sleep(0.05)
    held.save_copy(moved, d / "a.png")
    held.discard_all()
    assert [p.name for p in d.iterdir()] == ["a.png"]


def test_a_noted_file_that_is_gone_is_nothing_to_remove():
    note = held.note_gallery_file("image", _home() / "gui_images" / "gone.png")
    for name, text in (("file-bad", "[[["), ("file-up", '["image", "../x.png", 0]')):
        note.with_name(name).write_text(text)
    assert held.discard_all() is True and not held.root_dir().exists()


@pytest.mark.parametrize("kind", list(KINDS))
def test_a_stop_waits_for_the_move_of_a_result_that_turned_private(server, kind, monkeypatch):
    from localm.inference import http_server
    app, client = server
    path, body, target, folder, _key = KINDS[kind]
    _config(mode="log")
    moving, go_on = threading.Event(), threading.Event()
    real = held.save_copy

    def slow(*a, **k):
        moving.set()
        go_on.wait(20)
        return real(*a, **k)
    monkeypatch.setattr(held, "save_copy", slow)

    def fake(prompt, out, **kwargs):
        Path(out).write_bytes(BYTES[kind]())
        Path(out).with_name(Path(out).name + ".json").write_text('{"prompt": "SECRET"}')
        _config(mode="privacy")
        return True, "saved"
    with patch(target, fake), patch.object(http_server, "_engines", {}), \
            patch.object(http_server, "_engine", None):
        job = app.state.jobs.get(client.post(path, json=body).json()["job_id"])
        assert moving.wait(20)
        assert held.private_run_wait() == held._END_LIMIT_S
        assert not [p for p in (_home() / folder).iterdir() if p.suffix == ".json"]
        threading.Timer(0.5, go_on.set).start()
        http_server._shutdown_teardown()
    _ended(job)
    assert _media_files() == [] and list((_home() / folder).iterdir()) == []


def test_a_stop_killed_during_the_move_leaves_a_note_for_the_next_start(monkeypatch):
    from localm.media import gallery
    d = _home() / "gui_images"
    d.mkdir(parents=True)
    (d / "a.png").write_bytes(b"x")

    def killed(*a, **k):
        raise KeyboardInterrupt
    with monkeypatch.context() as m:
        m.setattr(held, "save_copy", killed)
        with pytest.raises(KeyboardInterrupt):
            gallery.settle(MagicMock(), "image", d / "a.png", None, False, True)
    assert (d / "a.png").is_file()
    held.discard_all()
    assert list(d.iterdir()) == []


@pytest.mark.parametrize("headers", [{"X-Localm-Session-Mode": "privacy"},
                                     {"X-Client-Id": "chat"}])
def test_a_private_stop_request_does_not_make_a_kept_job_private(server, headers, monkeypatch):
    from localm.inference import http_server
    app, client = server
    _config(mode="log", chat_mode="privacy")
    stopping = threading.Event()
    monkeypatch.setattr(http_server, "_do_shutdown", lambda **k: stopping.set())
    started, finish = threading.Event(), threading.Event()

    def fake(prompt, out, **kwargs):
        started.set()
        finish.wait(20)
        Path(out).write_bytes(media_bytes.png_bytes())
        return True, "saved"
    with patch(KINDS["image"][2], fake):
        job = app.state.jobs.get(client.post("/api/imagine", json={"prompt": "x"}).json()["job_id"])
        assert started.wait(20)
        assert client.post("/v1/server/shutdown", headers=headers).json() == {"stopping": True}
        assert stopping.wait(10)
        finish.set()
        assert _ended(job) == "done"
    assert (_home() / "gui_images" / job.result).is_file() and held.files("image") == []


def test_a_look_at_a_job_never_decides_for_it():
    from localm.plugins.media_config import keep_check
    answers = iter([True, False, True])
    keeps = keep_check(lambda: next(answers))
    assert keeps(peek=True) is False and keeps() is True


@pytest.mark.parametrize("kind", list(KINDS))
def test_each_job_asks_again_when_its_result_is_written(server, kind, monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "log")
    written: list = []
    monkeypatch.setattr(f"localm.plugins.builtin.{kind}.plug.media_keep_check",
                        lambda request: lambda peek=False: not written)
    real = BYTES[kind]
    monkeypatch.setitem(BYTES, kind, lambda: written.append(1) or real())
    job, (out, _k) = _generate(server, kind)
    assert not held.is_held(out) and not out.exists()
    assert [p.name for p in held.files(kind)] == [job.result]
    assert any(e.get("type") == "held" for e in job._history)


def test_each_image_of_one_api_request_is_decided_when_it_is_written(server, monkeypatch):
    from localm.plugins.builtin.image import plug
    monkeypatch.setenv("LOCALM_MODE", "log")
    written = []

    def fake(prompt, out, **kwargs):
        written.append(Path(out))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(media_bytes.png_bytes())
        return True, "saved"
    monkeypatch.setattr(plug, "_is_private_session", lambda: len(written) >= 2)
    monkeypatch.setenv("LOCALM_API_KEY", "owner-key-for-this-test")
    _app, client = server
    with patch(KINDS["image"][2], fake):
        r = client.post("/v1/images/generations", json={"prompt": "x", "n": 3},
                        headers={"Authorization": "Bearer owner-key-for-this-test"})
    assert r.status_code == 200, r.text
    assert ["url" in i for i in r.json()["data"]] == [True, False, False]
    assert any("turned private" in w for w in r.json()["warnings"])
    assert [p.name for p in (_home() / "gui_images").glob("*.png")] == [written[0].name]
    assert held.is_held(written[2]) and not written[2].exists()


def test_a_held_result_can_be_renamed_moved_and_deleted(server, tmp_path):
    _app, client = server
    job, (out, _k) = _generate(server, "image")
    r = client.post(f"/api/imagine/file/{job.result}/rename", json={"new_name": "mine"})
    assert r.json() == {"status": "renamed", "name": "mine.png"}
    dest = tmp_path / "saved"
    r = client.post("/api/imagine/file/mine.png/move", json={"dest": str(dest)})
    assert r.status_code == 200, r.text
    assert (dest / "mine.png").is_file() and not (out.parent / "mine.png").exists()
    job, (out, _k) = _generate(server, "image")
    assert client.delete(f"/api/imagine/file/{job.result}").status_code == 200
    assert not out.exists()
    assert client.get("/api/imagine/history").json()["held"] == []


def test_a_held_result_can_be_an_input_image(server):
    from localm.media.paths import check_input_image
    _job, (out, _k) = _generate(server, "image")
    assert check_input_image(str(out)) == out.resolve()
    assert check_input_image(out.name) == out.resolve()


def test_a_held_result_is_an_input_only_for_its_owner_and_once_it_is_finished():
    from localm.media.paths import InputImageRefused, check_input_image
    d = held.folder("image")
    (d / "api-x").mkdir()
    for name in ("mine.png", "unfinished.png", "api-x/nested.png"):
        (d / name).write_bytes(b"x")
    held.set_owner("image", "mine.png", "key:ka")
    assert check_input_image("mine.png", lambda o: o == "key:ka") == (d / "mine.png").resolve()
    for raw, key in (("mine.png", "key:kb"), (str(d / "mine.png"), "key:kb"),
                     ("unfinished.png", "key:ka"), (str(d / "unfinished.png"), "key:ka"),
                     (str(d / "api-x" / "nested.png"), "key:ka")):
        with pytest.raises(InputImageRefused):
            check_input_image(raw, lambda o, key=key: o == key)


@pytest.mark.parametrize("path", ["/api/imagine", "/api/video"])
def test_the_routes_ask_whether_the_caller_may_use_the_input(server, path, monkeypatch):
    _app, client = server
    d = held.folder("image")
    (d / "theirs.png").write_bytes(media_bytes.png_bytes())
    held.set_owner("image", "theirs.png", "key:kb")
    r = client.post(path, json={"prompt": "x", "input_image": "theirs.png"})
    assert r.status_code == 400, r.text


def test_a_held_result_is_private_to_the_key_that_made_it():
    from localm.media import gallery
    d = held.folder("image")
    (d / "a.png").write_bytes(b"x")
    assert gallery.record_owner("image", d / "a.png", "key:ka") is None
    assert gallery.owner_of("image", "a.png") == "key:ka"
    assert not (_home() / "gallery_index").exists()
    gallery.rename_owner("image", "a.png", "b.png")
    assert gallery.owner_of("image", "b.png") == "key:ka"
    gallery.forget_owner("image", "b.png")
    assert not held.tracks("image", "b.png")


def test_another_keys_held_results_are_not_listed(monkeypatch):
    from localm.media import gallery
    d = held.folder("image")
    for name, owner in (("mine.png", "key:ka"), ("other.png", "key:kb")):
        (d / name).write_bytes(b"x")
        gallery.record_owner("image", d / name, owner)
    (d / "unfinished.png").write_bytes(b"x")
    monkeypatch.setattr(gallery, "_caller_may_see", lambda req, owner: owner == "key:ka")
    assert [i["name"] for i in gallery.held_items(None, "image", str)] == ["mine.png"]


def test_a_saved_chat_keeps_the_media_made_in_it(server):
    _config(mode="privacy", chat_mode="full")
    _job, (out, kwargs) = _generate(server, "image", {"X-Client-Id": "chat"})
    assert out.parent == _home() / "gui_images" and kwargs["write_sidecar"] is False
    _job, (out, _k) = _generate(server, "image")
    assert held.is_held(out)


def test_a_private_chat_under_a_recording_server_keeps_nothing(server):
    _config(mode="log", chat_mode="privacy")
    _job, (out, _k) = _generate(server, "image", {"X-Client-Id": "chat"})
    assert held.is_held(out)
    _job, (out, _k) = _generate(server, "image")
    assert out.parent == _home() / "gui_images"


def test_once_a_job_is_not_kept_it_stays_so():
    from localm.plugins.media_config import keep_check
    answers = iter([True, True, False, True])
    keeps = keep_check(lambda: next(answers))
    assert [keeps() for _ in range(4)] == [True, False, False, False]
    assert next(answers) is True
    assert keep_check(lambda: False)() is False


def test_a_job_whose_mode_turns_private_before_it_starts_keeps_nothing():
    from localm.plugins.builtin.image.plug import ImagineRequest, _make_generate
    job = MagicMock()
    paths = []
    body = _make_generate(
        ImagineRequest(prompt="x"), input_image=None, lora_name=None, out_path=Path("a.png"),
        owner=None, self_url="http://127.0.0.1:1/v1", instance_token=None,
        keeps=lambda peek=False: False)
    with patch("localm.media.gallery.job_output_path",
               side_effect=lambda j, kind, name, kept: paths.append((name, kept))):
        body(job)
    assert paths == [("a.png", False)]


def test_a_held_folder_that_cannot_be_made_fails_the_job(server, monkeypatch):
    def refuse(kind):
        raise OSError("disk full")
    monkeypatch.setattr(held, "folder", refuse)
    app, client = server
    with patch("localm.image_gen.comfy.generate_image") as gen:
        r = client.post("/api/imagine", json={"prompt": "x"})
        job = app.state.jobs.get(r.json()["job_id"])
        deadline = time.monotonic() + 30
        while job.finished_at is None and time.monotonic() < deadline:
            time.sleep(0.02)
    assert job.status == "failed"
    assert any("disk full" in str(e.get("text")) for e in job._history)
    gen.assert_not_called()


# --------------------------------------------------------------------------- #
#  command line, chat REPL, MCP                                               #
# --------------------------------------------------------------------------- #

def _fake_generator(calls):
    def fake(prompt, out, **kwargs):
        calls.append(Path(out))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(media_bytes.png_bytes())
        return True, f"Image saved to {out}"
    return fake


@pytest.mark.parametrize("command", ["image", "music", "video"])
def test_the_cli_refuses_to_throw_a_result_away(command, tmp_path, monkeypatch):
    from localm.cli import main
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    with patch(KINDS[command][2]) as gen:
        result = CliRunner().invoke(main, [command, "a secret"])
    assert result.exit_code == 2, result.output
    assert "-o" in result.output and "privacy_keep media" in result.output
    gen.assert_not_called()
    assert list(cwd.iterdir()) == []


def test_the_cli_writes_the_file_it_was_told_to(tmp_path):
    from localm.cli import main
    calls: list = []
    out = tmp_path / "o.png"
    with patch("localm.image_gen.comfy.generate_image", _fake_generator(calls)):
        result = CliRunner().invoke(main, ["image", "a cat", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert calls == [out] and out.is_file()


@pytest.mark.parametrize("answer,kept", [("y", True), ("", True), ("n", False)])
def test_the_cli_asks_whether_to_save(answer, kept, tmp_path, monkeypatch):
    from localm.cli import main, media
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(media, "_is_interactive", lambda: True)
    monkeypatch.setattr(media, "_offer_open", lambda path: None)
    monkeypatch.setattr(media.console, "input", lambda *a, **k: answer)
    calls: list = []
    with patch("localm.image_gen.comfy.generate_image", _fake_generator(calls)):
        result = CliRunner().invoke(main, ["image", "a cat"])
    assert result.exit_code == 0, result.output
    assert held.root_dir() in calls[0].parents and not calls[0].exists()
    assert bool(list(tmp_path.glob("image_*.png"))) is kept


def test_the_cli_asks_for_another_path_when_the_copy_fails(tmp_path, monkeypatch):
    from localm.cli import main, media
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(media, "_is_interactive", lambda: True)
    monkeypatch.setattr(media, "_offer_open", lambda path: None)
    (tmp_path / "taken.png").write_bytes(b"mine")
    calls: list = []
    asked: list = []
    answers = iter([str(tmp_path / "taken.png"), str(tmp_path / "other.png")])

    def answer(question, **kw):
        asked.append((question, calls[0].is_file()))
        return next(answers)
    monkeypatch.setattr(media.console, "input", answer)
    with patch("localm.image_gen.comfy.generate_image", _fake_generator(calls)):
        result = CliRunner().invoke(main, ["image", "a cat"])
    assert result.exit_code == 0, result.output
    assert len(asked) == 2 and all(there for _q, there in asked)
    assert (tmp_path / "taken.png").read_bytes() == b"mine"
    assert (tmp_path / "other.png").is_file() and not calls[0].exists()


def test_the_cli_keeps_the_default_file_when_media_is_kept(tmp_path, monkeypatch):
    from localm.cli import main
    monkeypatch.chdir(tmp_path)
    _config(privacy_keep=["media"])
    calls: list = []
    with patch("localm.image_gen.comfy.generate_image", _fake_generator(calls)):
        result = CliRunner().invoke(main, ["image", "a cat"])
    assert result.exit_code == 0, result.output
    assert len(list(tmp_path.glob("image_*.png"))) == 1


def _repl(command, console):
    from localm import cli
    with patch.object(cli, "console", console):
        cli._handle_command(command, [], {}, engine=MagicMock())
    return console.file.getvalue()


def test_the_repl_holds_the_result_and_saves_it_on_request(tmp_path, monkeypatch):
    from rich.console import Console
    monkeypatch.chdir(tmp_path)
    console = Console(file=io.StringIO(), width=300)
    calls: list = []
    with patch("localm.image_gen.comfy.generate_image", _fake_generator(calls)):
        text = _repl("/generate-image a cat", console)
    assert held.is_held(calls[0]) and "/save-media" in text
    assert "Saved:" in _repl("/save-media mine", console)
    assert (tmp_path / "mine.png").is_file()
    assert "Already exists" in _repl("/save-media mine", console)
    assert "Refusing to save outside" in _repl("/save-media ../x.png", console)
    held.discard_all()
    assert (tmp_path / "mine.png").is_file() and not calls[0].exists()


def _mcp(args):
    from localm.plugins.mcpserver.server import EngineCache, MCPStdioServer, build_tools
    engines = EngineCache(default_model="stub", engine_factory=lambda *a, **k: MagicMock())
    server = MCPStdioServer(build_tools(engines, enable_images=True))
    r = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": "generate_image", "arguments": args}})
    return r["result"]["content"][0]["text"]


def test_the_mcp_tool_holds_an_image_nobody_named_a_file_for():
    calls: list = []
    with patch("localm.image_gen.comfy.generate_image", _fake_generator(calls)):
        text = _mcp({"prompt": "x"})
        assert held.is_held(calls[0]) and "not kept" in text
        assert str(calls[0]) in text and "when this MCP server stops" in text
        assert not (_home() / "mcp-images").exists()
        text = _mcp({"prompt": "x", "output_path": "mine.png"})
    assert calls[1] == _home().resolve() / "mcp-images" / "mine.png"
    assert "not kept" not in text


def test_the_mcp_tool_keeps_its_images_when_media_is_kept():
    _config(privacy_keep=["media"])
    calls: list = []
    with patch("localm.image_gen.comfy.generate_image", _fake_generator(calls)):
        _mcp({"prompt": "x"})
    assert calls[0].parent == _home().resolve() / "mcp-images"
