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
        told: list = []
        assert held.sweep_leftovers(told.append) == 1
        assert not folder.exists() and not folder.with_name(folder.name + ".lock").exists()
        assert len(told) == 1 and "http://127.0.0.1:8188" in told[0] and "job-1" in told[0]
        assert held.sweep_leftovers(told.append) == 0 and len(told) == 1


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
    jobs = JobManager(reconcile=False)
    with patch("localm.auth.any_key_configured", return_value=True):
        http_server._remove_leftovers(SimpleNamespace(state=SimpleNamespace(jobs=jobs)))
        http_server._remove_leftovers(SimpleNamespace(state=SimpleNamespace(jobs=jobs)))
    rows = jobs.snapshot()
    assert len(rows) == 1 and rows[0]["status"] == "interrupted" and "job-1" in rows[0]["label"]
    assert rows[0]["note"] is True
    assert jobs.snapshot(visible=lambda owner: owner in (None, "key:ka")) == []
    assert "cut off" in capsys.readouterr().out


def test_a_damaged_note_is_still_told():
    note = held.root_dir() / "123-abc" / "comfy" / "t"
    note.parent.mkdir(parents=True)
    note.write_bytes(b"[" * 4000)
    told: list = []
    held.sweep_leftovers(told.append)
    assert len(told) == 1 and "unknown" in told[0]


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


@pytest.mark.parametrize("stop", ["_shutdown_teardown", "_do_restart"])
@pytest.mark.parametrize("kind", list(KINDS))
def test_a_stop_ends_a_running_job_and_leaves_nothing(server, kind, stop, monkeypatch):
    """The job is cancelled and ended before held media is removed, also when its
    writer finishes the file after the cancel."""
    from localm.inference import http_server
    app, client = server
    path, body, target, folder, _key = KINDS[kind]
    monkeypatch.setattr(f"localm.plugins.builtin.{kind}.plug.principal_id", lambda r: "key:ka")
    monkeypatch.setattr(http_server, "_execv_restart", lambda *a, **k: None)
    monkeypatch.setattr(http_server, "_mark_fds_noninheritable", lambda: None)
    monkeypatch.setattr(http_server, "_set_restart_env", lambda: None)
    monkeypatch.setattr(http_server, "_finish_pending_runtime_update", lambda: False)
    monkeypatch.setattr(http_server, "_finish_comgr_temp", lambda: None)
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
        assert started.wait(20)
        getattr(http_server, stop)()
    assert cancelled == [True] and job.status == "cancelled"
    assert _media_files() == [] and not (_home() / "gallery_index").exists()
    assert not (_home() / folder).exists() and not held.CLOSING.is_set()


def test_no_media_job_starts_and_no_chat_model_reloads_while_localm_stops(server):
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
    from localm import vram
    monkeypatch.setattr(held, "_END_LIMIT_S", 0.1)
    monkeypatch.setattr(vram, "_handover_inflight", [MagicMock()])
    with caplog.at_level("WARNING", logger="localm"):
        assert held.end_media_work() is False
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
    assert gallery.settle("image", d / "b.png", lambda: True) == d / "b.png"
    moved = gallery.settle("image", d / "a.png", lambda: False)
    assert held.is_held(moved) and moved.read_bytes() == b"x"
    assert [p.name for p in d.iterdir()] == ["b.png"]
    held.CLOSING.set()
    with pytest.raises(OSError, match="stopping"):
        gallery.settle("image", d / "b.png", lambda: False)
    assert list(d.iterdir()) == []


@pytest.mark.parametrize("kind", list(KINDS))
def test_each_job_asks_again_when_its_result_is_written(server, kind, monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "log")
    answers = iter([True, True])
    monkeypatch.setattr(f"localm.plugins.builtin.{kind}.plug.media_keep_check",
                        lambda request: lambda: next(answers, False))
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


def test_a_job_whose_mode_turns_private_before_it_starts_keeps_nothing():
    answers = iter([True, False])
    from localm.plugins.builtin.image.plug import ImagineRequest, _make_generate
    job = MagicMock()
    paths = []
    body = _make_generate(
        ImagineRequest(prompt="x"), input_image=None, lora_name=None, out_path=Path("a.png"),
        owner=None, self_url="http://127.0.0.1:1/v1", instance_token=None,
        keeps=lambda: next(answers))
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
