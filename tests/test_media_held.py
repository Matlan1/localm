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
import time
from pathlib import Path
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
    assert held.files("image") == [] and not held.is_held(d / "a.png")


_HOLDER = """
import os, sys
from localm.media import held
d = held.folder("image")
(d / "left.png").write_bytes(b"x")
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
        assert held.sweep_leftovers() == 1
        assert not folder.exists() and not folder.with_name(folder.name + ".lock").exists()


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
