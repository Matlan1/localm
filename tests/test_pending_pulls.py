# SPDX-License-Identifier: AGPL-3.0-or-later
"""Records of model downloads that did not finish (model_manager.pending_pulls):
written when a transfer starts, removed on success, kept after a stop with
partial data on disk, detected after a restart, and resumed, deleted or
dismissed from the CLI and the GUI routes.

The direct-URL and Hugging Face pulls run through the real ``pull_model`` with
the network doubled at the transport seam; the liveness checks use real child
processes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from localm import model_manager as mm
from localm.model_manager import pending_pulls
from localm.model_manager.pull import PullInFlight, _partial_owner_path
from tests._process_identity import spawn_on_this_tree, this_pid_space

URL = "http://host1.example/tiny.gguf"
BODY = b"G" * 4000


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture()
def home(tmp_path, monkeypatch):
    """A data folder whose models folder the pull code uses."""
    models = tmp_path / "home" / "models"
    models.mkdir(parents=True)
    monkeypatch.setattr(mm, "MODELS_DIR", models)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    monkeypatch.setattr(mm, "find_by_sha256", lambda *a, **k: [])
    monkeypatch.setattr(mm, "_register", MagicMock())
    monkeypatch.setattr(mm, "_register_with_dedup", MagicMock(return_value=True))
    monkeypatch.setattr(mm, "gguf_unusable_reason", lambda *a, **k: None)
    monkeypatch.setattr(
        "socket.getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    return models


def _resp(status, body: bytes, total=None):
    r = MagicMock()
    r.status_code = status
    r.raise_for_status = MagicMock()
    r.headers = {"content-length": str(len(body) if total is None else total)}

    def _iter(chunk_size):
        for i in range(0, len(body), chunk_size):
            yield body[i:i + chunk_size]
    r.iter_content = _iter
    return r


class _Server:
    """A fake ``pinned_request`` serving *body* with Range support. With
    *stop_after*, a GET body stops after that many bytes by raising *stop*
    (a dropped connection by default)."""

    def __init__(self, body: bytes, stop_after=None, stop=None, etag='"v1"'):
        self.body = body
        self.stop_after = stop_after
        self.stop = stop or requests.ConnectionError("connection dropped")
        self.etag = etag
        self.gets: list[dict] = []

    def __call__(self, method, url, **kw):
        if method == "HEAD":
            h = MagicMock()
            h.status_code = 200
            h.headers = {"content-length": str(len(self.body)), "etag": self.etag}
            return h
        headers = dict(kw.get("headers") or {})
        self.gets.append(headers)
        rng = headers.get("Range")
        start = int(rng.removeprefix("bytes=").removesuffix("-")) if rng else 0
        r = _resp(206 if rng else 200, self.body[start:])
        if self.stop_after is not None:
            keep, err = self.body[start:start + self.stop_after], self.stop

            def _iter(chunk_size):
                for i in range(0, len(keep), chunk_size):
                    yield keep[i:i + chunk_size]
                raise err
            r.iter_content = _iter
        return r


def _records(models: Path) -> list[dict]:
    d = models.parent / pending_pulls.RECORDS_DIR_NAME
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))]


# ------------------------------------------------------------------ #
#  Direct URL: record lifecycle, resume                                #
# ------------------------------------------------------------------ #

class TestUrlPullRecord:
    def test_a_finished_pull_leaves_no_record(self, home, monkeypatch):
        monkeypatch.setattr("localm.netpolicy.pinned_request", _Server(BODY))
        assert mm.pull_model(URL, name="tiny") is True
        assert _records(home) == []

    def test_a_dropped_transfer_keeps_an_ended_record_with_its_source(self, home, monkeypatch):
        monkeypatch.setattr("localm.netpolicy.pinned_request",
                            _Server(BODY, stop_after=1500))
        assert mm.pull_model(URL, name="tiny") is False
        [rec] = _records(home)
        assert rec["ended"] is True
        assert rec["source"] == "url" and rec["url"] == URL
        assert rec["request"]["spec"] == URL and rec["request"]["name"] == "tiny"
        [view] = pending_pulls.list_interrupted()
        assert view["urls"] == [URL]
        assert view["bytes_on_disk"] == 1500
        assert view["expected_size"] == len(BODY)
        assert view["resumable"] is True

    def test_ctrl_c_mid_transfer_keeps_the_record(self, home, monkeypatch):
        monkeypatch.setattr("localm.netpolicy.pinned_request",
                            _Server(BODY, stop_after=700, stop=KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            mm.pull_model(URL, name="tiny")
        [rec] = _records(home)
        assert rec["ended"] is True
        assert pending_pulls.bytes_on_disk(rec) == 700

    def test_a_failure_before_any_byte_leaves_no_record(self, home, monkeypatch):
        server = _Server(BODY)

        def refuse(method, url, **kw):
            if method == "GET":
                raise requests.ConnectionError("refused")
            return server(method, url, **kw)
        monkeypatch.setattr("localm.netpolicy.pinned_request", refuse)
        assert mm.pull_model(URL, name="tiny") is False
        assert _records(home) == []

    def test_resume_continues_from_the_partial_and_removes_the_record(self, home, monkeypatch):
        monkeypatch.setattr("localm.netpolicy.pinned_request",
                            _Server(BODY, stop_after=1500))
        assert mm.pull_model(URL, name="tiny") is False
        [view] = pending_pulls.list_interrupted()

        server = _Server(BODY)
        monkeypatch.setattr("localm.netpolicy.pinned_request", server)
        from localm.cli.models import _resume_interrupted
        assert _resume_interrupted(view) == 0
        assert [g.get("Range") for g in server.gets] == ["bytes=1500-"]
        assert (home / "tiny.gguf").read_bytes() == BODY
        assert _records(home) == []

    def test_a_second_attempt_replaces_the_first_record(self, home, monkeypatch):
        monkeypatch.setattr("localm.netpolicy.pinned_request",
                            _Server(BODY, stop_after=1000))
        assert mm.pull_model(URL, name="tiny") is False
        first = _records(home)[0]["id"]
        monkeypatch.setattr("localm.netpolicy.pinned_request",
                            _Server(BODY, stop_after=1000))
        assert mm.pull_model(URL, name="tiny") is False
        [rec] = _records(home)
        assert rec["id"] != first
        assert pending_pulls.bytes_on_disk(rec) == 2000

    def test_delete_removes_the_part_its_identity_and_the_record(self, home, monkeypatch):
        monkeypatch.setattr("localm.netpolicy.pinned_request",
                            _Server(BODY, stop_after=1500))
        assert mm.pull_model(URL, name="tiny") is False
        other = home / "other.gguf"
        other.write_bytes(b"keep me")
        [view] = pending_pulls.list_interrupted()
        assert pending_pulls.delete_interrupted(view["id"]) == 1500
        assert not (home / "tiny.gguf.part").exists()
        assert not (home / "tiny.gguf.part.json").exists()
        assert other.read_bytes() == b"keep me"
        assert _records(home) == []

    def test_changed_size_at_the_source_refuses_resume(self, home, monkeypatch):
        monkeypatch.setattr("localm.netpolicy.pinned_request",
                            _Server(BODY, stop_after=1500))
        assert mm.pull_model(URL, name="tiny") is False
        [view] = pending_pulls.list_interrupted()
        monkeypatch.setattr("localm.netpolicy.pinned_request", _Server(BODY + b"x"))
        rec = pending_pulls.get_interrupted(view["id"])
        assert pending_pulls.check_upstream(rec)["changed"] is True
        called = []
        monkeypatch.setattr("localm.cli.models.pull_model",
                            lambda *a, **k: called.append(a) or True)
        from localm.cli.models import _resume_interrupted
        assert _resume_interrupted(view) == 1
        assert called == []
        assert (home / "tiny.gguf.part").stat().st_size == 1500

    def test_changed_etag_at_the_source_is_detected(self, home, monkeypatch):
        monkeypatch.setattr("localm.netpolicy.pinned_request",
                            _Server(BODY, stop_after=1500, etag='"v1"'))
        assert mm.pull_model(URL, name="tiny") is False
        rec = pending_pulls.get_interrupted(pending_pulls.list_interrupted()[0]["id"])
        monkeypatch.setattr("localm.netpolicy.pinned_request", _Server(BODY, etag='"v1"'))
        assert pending_pulls.check_upstream(rec)["changed"] is False
        monkeypatch.setattr("localm.netpolicy.pinned_request", _Server(BODY, etag='"v2"'))
        assert pending_pulls.check_upstream(rec)["changed"] is True

    def test_dismiss_hides_it_and_keeps_its_data(self, home, monkeypatch):
        monkeypatch.setattr("localm.netpolicy.pinned_request",
                            _Server(BODY, stop_after=1500))
        assert mm.pull_model(URL, name="tiny") is False
        [view] = pending_pulls.list_interrupted()
        pending_pulls.set_dismissed(view["id"], True)
        assert pending_pulls.list_interrupted() == []
        [hidden] = pending_pulls.list_interrupted(include_dismissed=True)
        assert hidden["dismissed"] is True
        assert (home / "tiny.gguf.part").stat().st_size == 1500
        pending_pulls.set_dismissed(view["id"], False)
        assert [v["id"] for v in pending_pulls.list_interrupted()] == [view["id"]]


# ------------------------------------------------------------------ #
#  Hugging Face single file                                            #
# ------------------------------------------------------------------ #

def _fake_hf_head(monkeypatch, size: int, etag: str):
    first = MagicMock()
    first.headers = {"X-Linked-Etag": f'"{etag}"'}
    head = MagicMock(status_code=200, headers={"content-length": str(size)})
    head.history = [first]
    monkeypatch.setattr("localm.model_manager.pull._policy_head", lambda *a, **kw: head)


def _hf_temp(base: Path, rel: str, etag: str, data: bytes, owner_pid: int, start) -> Path:
    from huggingface_hub._local_folder import get_local_download_paths
    paths = get_local_download_paths(base, rel)
    inc = paths.incomplete_path(etag)
    t = inc.with_name(f"{inc.stem}.deadbeef.incomplete")
    t.write_bytes(data)
    _partial_owner_path(t).write_text(json.dumps(
        {"pid": owner_pid, "space": this_pid_space(), "start": start}), encoding="utf-8")
    return t


@pytest.fixture()
def dead_pid():
    import subprocess
    from localm.instances import process_start_identity
    p = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
                         stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    start = process_start_identity(p.pid)
    p.stdin.close()
    p.wait()
    return p.pid, start


class TestHfFileRecord:
    def test_a_stopped_hf_pull_records_repo_file_url_size_and_etag(
            self, home, monkeypatch, dead_pid):
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda *a: None)
        monkeypatch.setattr("localm.model_manager.pull._published_digest", lambda *a: None)
        _fake_hf_head(monkeypatch, 5000, "etag1")

        def stopped_download(repo_id, filename, local_dir, **kw):
            _hf_temp(Path(local_dir), filename, "etag1", b"x" * 1234, *dead_pid)
            raise requests.ConnectionError("connection dropped")
        monkeypatch.setattr("huggingface_hub.hf_hub_download", stopped_download)

        assert mm.pull_model("owner/repo:tiny-Q4.gguf", name="tiny") is False
        [rec] = _records(home)
        assert rec["ended"] is True
        assert rec["source"] == "hf-file" and rec["repo"] == "owner/repo"
        [f] = rec["files"]
        assert f["rel"] == "tiny-Q4.gguf" and f["size"] == 5000 and f["etag"] == "etag1"
        assert f["url"].startswith("https://huggingface.co/owner/repo/resolve/")
        assert f["url"].endswith("/tiny-Q4.gguf")
        [view] = pending_pulls.list_interrupted()
        assert view["bytes_on_disk"] == 1234 and view["expected_size"] == 5000

    def test_delete_removes_only_this_pulls_temp_files(self, home, dead_pid):
        mine = _hf_temp(home, "a.gguf", "e1", b"a" * 10, *dead_pid)
        other = _hf_temp(home, "b.gguf", "e1", b"b" * 20, *dead_pid)
        with pending_pulls.tracking({"spec": "o/r:a.gguf"}):
            pending_pulls.note_source(source="hf-file", repo="o/r")
            pending_pulls.note_hf_files(home, [{"rel": "a.gguf", "url": "u", "size": 50,
                                                "etag": "e1"}])
        [view] = pending_pulls.list_interrupted()
        assert pending_pulls.delete_interrupted(view["id"]) == 10
        assert not mine.exists() and not _partial_owner_path(mine).exists()
        assert other.read_bytes() == b"b" * 20

    def test_a_record_naming_a_folder_outside_the_models_folder_deletes_nothing(
            self, home, tmp_path, dead_pid):
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        victim = _hf_temp(outside, "a.gguf", "e1", b"v" * 10, *dead_pid)
        with pending_pulls.tracking({"spec": "o/r:a.gguf"}):
            pending_pulls.note_source(source="hf-file")
            pending_pulls.note_hf_files(outside, [{"rel": "a.gguf"}])
        [view] = pending_pulls.list_interrupted()
        assert pending_pulls.delete_interrupted(view["id"]) == 0
        assert victim.read_bytes() == b"v" * 10

    def test_changed_etag_at_the_hub_is_detected(self, home, monkeypatch):
        rec = {"files": [{"rel": "a.gguf", "url": "https://huggingface.co/o/r/resolve/main/a.gguf",
                          "size": 10, "etag": "old"}]}
        _fake_hf_head(monkeypatch, 10, "new")
        assert pending_pulls.check_upstream(rec)["changed"] is True
        _fake_hf_head(monkeypatch, 10, "old")
        assert pending_pulls.check_upstream(rec)["changed"] is False


# ------------------------------------------------------------------ #
#  Full repo                                                           #
# ------------------------------------------------------------------ #

class TestSnapshotRecord:
    def _stopped(self, home: Path, created: bool, dead_pid) -> Path:
        folder = home / "Repo"
        (folder / ".cache" / "huggingface" / "download").mkdir(parents=True)
        (folder / "config.json").write_text("{}", encoding="utf-8")
        _hf_temp(folder, "model.safetensors", "e1", b"p" * 30, *dead_pid)
        with pending_pulls.tracking({"spec": "o/Repo"}):
            pending_pulls.note_source(source="hf-repo", repo="o/Repo", commit="a" * 40)
            pending_pulls.note_snapshot(folder, "Repo", created, 1000)
        return folder

    def test_delete_removes_a_folder_this_pull_created(self, home, monkeypatch, dead_pid):
        monkeypatch.setattr(mm, "load_registry", lambda: {})
        folder = self._stopped(home, True, dead_pid)
        [view] = pending_pulls.list_interrupted()
        assert pending_pulls.delete_interrupted(view["id"]) > 0
        assert not folder.exists()

    def test_delete_keeps_a_folder_a_registered_model_uses(self, home, monkeypatch, dead_pid):
        folder = self._stopped(home, True, dead_pid)
        monkeypatch.setattr(mm, "load_registry",
                            lambda: {"repo": {"path": str(folder), "source": "hf:o/Repo"}})
        [view] = pending_pulls.list_interrupted()
        assert pending_pulls.delete_interrupted(view["id"]) == 30
        assert (folder / "config.json").exists()

    def test_delete_keeps_a_folder_that_existed_before(self, home, monkeypatch, dead_pid):
        monkeypatch.setattr(mm, "load_registry", lambda: {})
        folder = self._stopped(home, False, dead_pid)
        [view] = pending_pulls.list_interrupted()
        assert pending_pulls.delete_interrupted(view["id"]) == 30
        assert (folder / "config.json").exists()


# ------------------------------------------------------------------ #
#  Liveness: a running pull is never offered or deleted                #
# ------------------------------------------------------------------ #

def _kill_owner(pid: int) -> None:
    """Terminate *pid* (the interpreter that owns the record, not a venv
    launcher in front of it) and wait until it is gone."""
    import os
    import signal
    import time
    from localm.instances import pid_alive
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not pid_alive(pid)


HOLD_A_PULL = '''
    import sys
    from pathlib import Path
    import localm.model_manager as mm
    from localm.model_manager import pending_pulls
    part = mm.MODELS_DIR / "held.gguf.part"
    mm.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with pending_pulls.tracking({"spec": "http://h.example/held.gguf", "name": "held"}):
        pending_pulls.note_source(source="url", url="http://h.example/held.gguf")
        part.write_bytes(b"z" * 64)
        pending_pulls.note_part(part, "held.gguf", 128)
        print("HOLDING", flush=True)
        sys.stdin.read()
'''


class TestLiveness:
    def test_a_live_pull_is_not_offered_and_a_killed_one_is(self, tmp_path, monkeypatch):
        import subprocess
        home_dir = tmp_path / "home"
        models = home_dir / "models"
        models.mkdir(parents=True)
        monkeypatch.setattr(mm, "MODELS_DIR", models)
        child = spawn_on_this_tree(HOLD_A_PULL, home_dir, stdin=subprocess.PIPE)
        try:
            line = child.stdout.readline()
            assert line.strip() == "HOLDING", child.stderr.read()
            [rec] = _records(models)
            assert rec["ended"] is False
            assert pending_pulls.list_interrupted() == []
            with pytest.raises(PullInFlight):
                pending_pulls.delete_interrupted(rec["id"])
            assert (models / "held.gguf.part").stat().st_size == 64
            assert pending_pulls.list_orphans() == []
            _kill_owner(rec["owner"]["pid"])
        finally:
            child.kill()
            child.wait()
        [view] = pending_pulls.list_interrupted()
        assert view["spec"] == "http://h.example/held.gguf"
        assert view["bytes_on_disk"] == 64
        assert pending_pulls.delete_interrupted(view["id"]) == 64
        assert not (models / "held.gguf.part").exists()


# ------------------------------------------------------------------ #
#  Orphans                                                             #
# ------------------------------------------------------------------ #

class TestOrphans:
    def test_an_unclaimed_temp_file_of_a_dead_owner_is_offered_and_deleted(self, home, dead_pid):
        orphan = _hf_temp(home, "old.gguf", "e9", b"o" * 77, *dead_pid)
        [o] = pending_pulls.list_orphans()
        assert o["size"] == 77 and o["file"].endswith(".incomplete")
        assert pending_pulls.delete_orphan(o["id"]) == 77
        assert not orphan.exists()
        assert pending_pulls.list_orphans() == []

    def test_a_temp_file_of_a_live_owner_is_not_offered(self, home):
        import subprocess
        from localm.instances import process_start_identity
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            _hf_temp(home, "live.gguf", "e9", b"l" * 5, p.pid, process_start_identity(p.pid))
            assert pending_pulls.list_orphans() == []
        finally:
            p.kill()
            p.wait()

    def test_a_temp_file_a_record_names_is_not_an_orphan(self, home, dead_pid):
        _hf_temp(home, "a.gguf", "e1", b"a" * 10, *dead_pid)
        with pending_pulls.tracking({"spec": "o/r:a.gguf"}):
            pending_pulls.note_hf_files(home, [{"rel": "a.gguf"}])
        assert pending_pulls.list_orphans() == []

    def test_an_unknown_orphan_id_deletes_nothing(self, home, dead_pid):
        f = _hf_temp(home, "old.gguf", "e9", b"o" * 7, *dead_pid)
        bogus = "orphan-" + hashlib.sha256(b"x").hexdigest()[:24]
        with pytest.raises(pending_pulls.UnknownPull):
            pending_pulls.delete_orphan(bogus)
        with pytest.raises(pending_pulls.UnknownPull):
            pending_pulls.delete_orphan("../../etc")
        assert f.exists()


# ------------------------------------------------------------------ #
#  Bad records                                                         #
# ------------------------------------------------------------------ #

def test_a_malformed_record_is_ignored_and_reported(home, caplog):
    d = home.parent / pending_pulls.RECORDS_DIR_NAME
    d.mkdir()
    (d / ("0" * 32 + ".json")).write_text("{not json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert pending_pulls.list_interrupted() == []
    assert "unreadable download record" in caplog.text
    with pytest.raises(pending_pulls.UnknownPull):
        pending_pulls.get_interrupted("0" * 32)
    with pytest.raises(pending_pulls.UnknownPull):
        pending_pulls.get_interrupted("../x")


def test_resume_args_repeat_the_request():
    rec = {"request": {"spec": "o/r:f.gguf", "name": "n", "sha256": "ab", "mmproj": None,
                       "type": "llm", "redownload": False, "dest_dir": "D:/c",
                       "register": False}}
    assert pending_pulls.resume_args(rec) == [
        "--name", "n", "--sha256", "ab", "--type", "llm", "--comfy-dest-dir", "D:/c",
        "--no-register", "--", "o/r:f.gguf"]


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

class TestCli:
    def _interrupt(self, home, monkeypatch):
        monkeypatch.setattr("localm.netpolicy.pinned_request",
                            _Server(BODY, stop_after=1500))
        assert mm.pull_model(URL, name="tiny") is False
        return pending_pulls.list_interrupted()[0]

    def test_list_interrupted_shows_spec_and_url(self, home, monkeypatch):
        from click.testing import CliRunner
        from localm.cli import main
        view = self._interrupt(home, monkeypatch)
        out = CliRunner().invoke(main, ["pull", "--list-interrupted"])
        assert out.exit_code == 0, out.output
        assert view["id"][:8] in out.output and URL in out.output

    def test_resume_by_id_runs_the_recorded_pull(self, home, monkeypatch):
        from click.testing import CliRunner
        from localm.cli import main
        view = self._interrupt(home, monkeypatch)
        server = _Server(BODY)
        monkeypatch.setattr("localm.netpolicy.pinned_request", server)
        out = CliRunner().invoke(main, ["pull", "--resume", view["id"][:8]])
        assert out.exit_code == 0, out.output
        assert [g.get("Range") for g in server.gets] == ["bytes=1500-"]
        assert _records(home) == []

    def test_resume_is_refused_with_the_network_off(self, home, monkeypatch):
        from click.testing import CliRunner
        from localm.cli import main
        view = self._interrupt(home, monkeypatch)
        monkeypatch.setenv("LOCALM_NET_MODE", "off")
        monkeypatch.setattr("localm.netpolicy.downloads_allowed_when_off", lambda: False)
        out = CliRunner().invoke(main, ["pull", "--resume", view["id"][:8]])
        assert out.exit_code == 1
        assert "net_mode=off" in out.output
        assert len(_records(home)) == 1

    def test_delete_and_dismiss_flags(self, home, monkeypatch):
        from click.testing import CliRunner
        from localm.cli import main
        view = self._interrupt(home, monkeypatch)
        out = CliRunner().invoke(main, ["pull", "--dismiss-interrupted", view["id"][:8]])
        assert out.exit_code == 0, out.output
        assert pending_pulls.list_interrupted() == []
        out = CliRunner().invoke(main, ["pull", "--delete-interrupted", view["id"][:8]])
        assert out.exit_code == 0, out.output
        assert not (home / "tiny.gguf.part").exists() and _records(home) == []

    def test_a_non_interactive_pull_names_interrupted_downloads_in_one_line(
            self, home, monkeypatch):
        from localm.cli.models import _offer_interrupted
        self._interrupt(home, monkeypatch)
        monkeypatch.setattr("localm.model_manager.registry.can_prompt", lambda: False)
        printed = []
        monkeypatch.setattr("localm.cli.models.console.print",
                            lambda *a, **k: printed.append(str(a[0])))
        _offer_interrupted("owner/other:x.gguf", None)
        assert len(printed) == 1 and "--list-interrupted" in printed[0]


# ------------------------------------------------------------------ #
#  GUI routes                                                          #
# ------------------------------------------------------------------ #

class _FakeJob:
    id = "job-stub"


class TestGuiRoutes:
    @pytest.fixture()
    def client(self, home, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient

        from fastapi import FastAPI

        from localm.plugins.gui.web import attach_gui
        app = FastAPI()

        async def switch_model(name):
            return {"status": "loaded", "model": name}
        attach_gui(app, self_url="http://127.0.0.1:9/v1", switch_model=switch_model,
                   active_model=lambda: "m")
        started = []

        def fake_start_cli(self, kind, cli_args, **kw):
            started.append((kind, list(cli_args), kw.get("host_label")))
            return _FakeJob()
        monkeypatch.setattr("localm.plugins.gui.jobs.JobManager.start_cli", fake_start_cli)
        monkeypatch.setattr("localm.netpolicy.pinned_request",
                            _Server(BODY, stop_after=1500))
        assert mm.pull_model(URL, name="tiny") is False
        return TestClient(app), started

    def test_list_resume_dismiss_delete(self, client, home, monkeypatch):
        c, started = client
        r = c.get("/api/models/pulls/interrupted")
        assert r.status_code == 200, r.text
        [d] = r.json()["downloads"]
        assert d["spec"] == URL and d["bytes_on_disk"] == 1500

        monkeypatch.setattr("localm.netpolicy.pinned_request", _Server(BODY))
        r = c.post(f"/api/models/pulls/interrupted/{d['id']}/resume", json={})
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "started"
        assert started == [("pull", ["pull", "--name", "tiny", "--", URL],
                            f"Model pull {URL}")]

        r = c.post(f"/api/models/pulls/interrupted/{d['id']}/dismiss", json={"dismissed": True})
        assert r.status_code == 200 and r.json()["dismissed"] is True
        assert c.get("/api/models/pulls/interrupted").json()["downloads"] == []
        listed = c.get("/api/models/pulls/interrupted?include_dismissed=true").json()
        assert listed["downloads"][0]["dismissed"] is True

        r = c.post(f"/api/models/pulls/interrupted/{d['id']}/delete")
        assert r.status_code == 200 and r.json()["removed_bytes"] == 1500
        assert c.get("/api/models/pulls/interrupted").json()["downloads"] == []

    def test_resume_of_a_changed_source_starts_nothing(self, client, monkeypatch):
        c, started = client
        [d] = c.get("/api/models/pulls/interrupted").json()["downloads"]
        monkeypatch.setattr("localm.netpolicy.pinned_request", _Server(BODY + b"more"))
        r = c.post(f"/api/models/pulls/interrupted/{d['id']}/resume", json={})
        assert r.status_code == 200 and r.json()["status"] == "changed"
        assert started == []
        r = c.post(f"/api/models/pulls/interrupted/{d['id']}/resume", json={"fresh": True})
        assert r.json()["status"] == "started"
        assert len(started) == 1
        assert not (mm.MODELS_DIR / "tiny.gguf.part").exists()

    def test_resume_with_the_network_off_is_refused(self, client, monkeypatch):
        c, started = client
        [d] = c.get("/api/models/pulls/interrupted").json()["downloads"]
        monkeypatch.setenv("LOCALM_NET_MODE", "off")
        monkeypatch.setattr("localm.netpolicy.downloads_allowed_when_off", lambda: False)
        r = c.post(f"/api/models/pulls/interrupted/{d['id']}/resume", json={})
        assert r.status_code == 403
        assert "Network access is off" in r.json()["detail"]
        assert started == []

    def test_unknown_ids_are_404(self, client):
        c, _ = client
        assert c.post("/api/models/pulls/interrupted/" + "f" * 32 + "/delete").status_code == 404
        assert c.post("/api/models/pulls/interrupted/nope/resume", json={}).status_code == 404
        assert c.post("/api/models/pulls/orphans/orphan-zz/delete").status_code == 404

    def test_delete_of_a_complete_unregistered_folder_is_refused_with_409(
            self, client, home, monkeypatch):
        c, _ = client
        monkeypatch.setattr(mm, "load_registry", lambda: {})
        folder = home / "Repo"
        (folder / ".cache" / "huggingface" / "download").mkdir(parents=True)
        (folder / "model.safetensors").write_bytes(b"w" * 1000)
        with pending_pulls.tracking({"spec": "o/Repo"}):
            pending_pulls.note_source(source="hf-repo", repo="o/Repo", commit="a" * 40)
            pending_pulls.note_snapshot(folder, "Repo", True, 1000)
        [d] = [x for x in c.get("/api/models/pulls/interrupted").json()["downloads"]
               if x["spec"] == "o/Repo"]
        assert (folder / "model.safetensors").stat().st_size == 1000
        r = c.post(f"/api/models/pulls/interrupted/{d['id']}/delete")
        assert r.status_code == 409 and "complete" in r.json()["detail"]
        assert (folder / "model.safetensors").stat().st_size == 1000
        r = c.post(f"/api/models/pulls/interrupted/{d['id']}/resume", json={"fresh": True})
        assert r.status_code == 409
        assert (folder / "model.safetensors").stat().st_size == 1000

    def test_a_refused_second_pull_of_a_running_file_adds_no_row(self, client, home, monkeypatch):
        import subprocess
        import textwrap
        c, _ = client
        before = c.get("/api/models/pulls/interrupted").json()["downloads"]
        TestSecondPullOfOneHfFile()._fakes(monkeypatch)
        child = subprocess.Popen(
            [sys.executable, "-I", "-c", textwrap.dedent(HOLD_A_HF_LOCK),
             str(home), "tiny-Q4.gguf"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            assert child.stdout.readline().strip() == "HOLDING", child.stderr.read()
            assert mm.pull_model("owner/repo:tiny-Q4.gguf", name="tiny2") is False
            assert c.get("/api/models/pulls/interrupted").json()["downloads"] == before
        finally:
            child.stdin.close()
            child.wait(timeout=30)


# ------------------------------------------------------------------ #
#  Review findings                                                     #
# ------------------------------------------------------------------ #

class TestDeleteScope:
    def test_delete_keeps_another_etags_temp_file_of_the_same_file(self, home, dead_pid):
        mine = _hf_temp(home, "f.gguf", "e1", b"a" * 10, *dead_pid)
        other = _hf_temp(home, "f.gguf", "e2", b"b" * 20, *dead_pid)
        with pending_pulls.tracking({"spec": "o/r:f.gguf"}):
            pending_pulls.note_hf_files(home, [{"rel": "f.gguf", "etag": "e1"}])
        [view] = pending_pulls.list_interrupted()
        assert view["bytes_on_disk"] == 10
        assert pending_pulls.delete_interrupted(view["id"]) == 10
        assert not mine.exists()
        assert other.read_bytes() == b"b" * 20

    def test_a_part_that_cannot_be_deleted_keeps_the_record_and_says_so(self, home, monkeypatch):
        monkeypatch.setattr("localm.netpolicy.pinned_request",
                            _Server(BODY, stop_after=1500))
        assert mm.pull_model(URL, name="tiny") is False
        [view] = pending_pulls.list_interrupted()
        part = home / "tiny.gguf.part"
        real_unlink = Path.unlink

        def stuck(self, missing_ok=False):
            if self == part:
                raise PermissionError("in use")
            return real_unlink(self, missing_ok=missing_ok)
        monkeypatch.setattr(Path, "unlink", stuck)
        with pytest.raises(pending_pulls.DeleteFailed):
            pending_pulls.delete_interrupted(view["id"])
        monkeypatch.setattr(Path, "unlink", real_unlink)
        assert part.stat().st_size == 1500
        assert [r["id"] for r in _records(home)] == [view["id"]]

    def test_a_hidden_part_of_an_import_is_not_an_orphan(self, home):
        (home / ".model.gguf.part").write_bytes(b"i" * 9)
        (home / ".model.gguf.part.json").write_text(json.dumps({"source": "url"}),
                                                    encoding="utf-8")
        (home / "nojson.gguf.part").write_bytes(b"n" * 9)
        assert pending_pulls.list_orphans() == []

    def test_an_unclaimed_url_part_with_its_identity_is_an_orphan(self, home):
        (home / "old.gguf.part").write_bytes(b"u" * 9)
        (home / "old.gguf.part.json").write_text(json.dumps({"source": "url"}), encoding="utf-8")
        [o] = pending_pulls.list_orphans()
        assert o["file"] == "old.gguf.part"
        assert pending_pulls.delete_orphan(o["id"]) == 9
        assert not (home / "old.gguf.part").exists()
        assert not (home / "old.gguf.part.json").exists()


class TestRecordHygiene:
    def test_a_record_whose_data_is_gone_is_pruned(self, home, monkeypatch):
        monkeypatch.setattr("localm.netpolicy.pinned_request",
                            _Server(BODY, stop_after=1500))
        assert mm.pull_model(URL, name="tiny") is False
        (home / "tiny.gguf.part").unlink()
        assert pending_pulls.list_interrupted() == []
        assert _records(home) == []

    def test_own_pid_in_another_pid_space_counts_as_running(self, home):
        import os
        with pending_pulls.tracking({"spec": "o/r:x.gguf"}):
            part = home / "x.gguf.part"
            part.write_bytes(b"x")
            pending_pulls.note_part(part, "x.gguf", 2)
        [rec] = _records(home)
        rec.update(ended=False, owner={"pid": os.getpid(), "space": "another-space",
                                       "host": "elsewhere", "local": False, "start": None})
        (home.parent / pending_pulls.RECORDS_DIR_NAME / f"{rec['id']}.json").write_text(
            json.dumps(rec), encoding="utf-8")
        assert pending_pulls.list_interrupted() == []
        with pytest.raises(PullInFlight):
            pending_pulls.delete_interrupted(rec["id"])

    def test_a_url_with_nothing_to_compare_and_no_digest_is_not_resumed(self, home, monkeypatch):
        class Bare(_Server):
            def __call__(self, method, url, **kw):
                r = super().__call__(method, url, **kw)
                if method == "HEAD":
                    r.headers = {}
                return r
        monkeypatch.setattr("localm.netpolicy.pinned_request", Bare(BODY, stop_after=1500))
        assert mm.pull_model(URL, name="tiny") is False
        rec = pending_pulls.get_interrupted(pending_pulls.list_interrupted()[0]["id"])
        check = pending_pulls.check_upstream(rec)
        assert check["changed"] is True and "no size or etag" in check["detail"]

    def test_the_view_does_not_name_a_comfyui_folder(self, home):
        with pending_pulls.tracking({"spec": "o/r:x.gguf", "dest_dir": r"\\server\share\comfy"}):
            part = home / "x.gguf.part"
            part.write_bytes(b"x")
            pending_pulls.note_part(part, "x.gguf", 2)
        [view] = pending_pulls.list_interrupted()
        assert "dest_dir" not in view["options"] and view["comfy_dest"] is True
        assert "share" not in json.dumps(view)


def test_a_failed_delete_at_the_pull_prompt_still_runs_the_requested_pull(home, monkeypatch):
    from localm.cli import models as cli
    monkeypatch.setattr("localm.netpolicy.pinned_request", _Server(BODY, stop_after=1500))
    assert mm.pull_model(URL, name="tiny") is False
    monkeypatch.setattr("localm.model_manager.registry.can_prompt", lambda: True)
    monkeypatch.setattr("click.prompt", lambda *a, **k: "d")

    def busy(_id):
        raise PullInFlight("busy")
    monkeypatch.setattr(pending_pulls, "delete_interrupted", busy)
    cli._offer_interrupted("owner/other:x.gguf", None)
    assert len(_records(home)) == 1


# ------------------------------------------------------------------ #
#  U12: audit of the interrupted-download feature                       #
# ------------------------------------------------------------------ #

def _fs_snapshot(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def _link_or_skip(link: Path, target: Path) -> None:
    """Make *link* a directory link to *target*: a symlink, else (Windows
    without the symlink privilege) a junction."""
    import subprocess
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if sys.platform != "win32":
            pytest.skip("cannot create a directory link here")
        out = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                             capture_output=True)
        if out.returncode != 0:
            pytest.skip("cannot create a directory link here")


@pytest.fixture()
def net_policy(tmp_path, monkeypatch):
    """A throwaway config the tests write net_* keys into."""
    cfg_home = tmp_path / "cfg" / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(cfg_home))
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", cfg_home)
    monkeypatch.setattr(_cfg, "CONFIG_FILE", cfg_home / "config.json")
    cfg_home.mkdir(parents=True)

    def _set(**values):
        cfg = _cfg.load_config()
        cfg.update(values)
        _cfg.save_config(cfg)
    return _set


class TestUpstreamCheckHonoursNetworkPolicy:
    REC = {"source": "hf-repo", "repo": "o/Repo", "revision": None, "commit": "a" * 40,
           "files": [], "parts": [], "snapshot": {"dir": "x", "name": "Repo"}}

    def _count_model_info(self, monkeypatch, sha):
        calls = []

        def model_info(repo, **kw):
            calls.append(repo)
            return MagicMock(sha=sha)
        monkeypatch.setattr("localm.model_manager.pull._hf_api",
                            lambda: MagicMock(model_info=model_info))
        return calls

    def test_a_denied_hub_endpoint_gets_no_request_and_is_reported(
            self, home, monkeypatch, net_policy):
        net_policy(net_mode="allow", net_deny=["huggingface.co"])
        calls = self._count_model_info(monkeypatch, "b" * 40)
        check = pending_pulls.check_upstream(dict(self.REC))
        assert calls == []
        assert check["changed"] is None
        assert "o/Repo" in check["detail"] and "network policy" in check["detail"]

    def test_an_allowed_hub_endpoint_is_asked_and_a_new_commit_is_seen(
            self, home, monkeypatch, net_policy):
        net_policy(net_mode="allow", net_deny=[])
        calls = self._count_model_info(monkeypatch, "b" * 40)
        check = pending_pulls.check_upstream(dict(self.REC))
        assert calls == ["o/Repo"]
        assert check["changed"] is True


class TestListingWritesNothing:
    def _comfy_record(self, home, tmp_path, dead_pid):
        comfy = tmp_path / "comfy"
        (comfy / ".cache" / "localm-staging").mkdir(parents=True)
        _hf_temp(comfy / ".cache" / "localm-staging", "sub/a.gguf", "e1", b"t" * 4, *dead_pid)
        with pending_pulls.tracking({"spec": "o/r:sub/a.gguf", "dest_dir": str(comfy)}):
            pending_pulls.note_source(source="hf-file", repo="o/r")
            pending_pulls.note_hf_files(comfy / ".cache" / "localm-staging",
                                        [{"rel": "sub/a.gguf", "etag": "e1", "size": 9}])
        return comfy

    def test_listing_a_record_whose_folder_is_gone_leaves_it_gone(
            self, home, tmp_path, dead_pid):
        import shutil
        comfy = self._comfy_record(home, tmp_path, dead_pid)
        assert len(_records(home)) == 1
        shutil.rmtree(comfy)
        assert pending_pulls.list_interrupted() == []
        assert pending_pulls.list_orphans() == []
        assert not comfy.exists()

    def test_deleting_a_record_whose_folder_is_gone_creates_nothing(
            self, home, tmp_path, dead_pid):
        import shutil
        comfy = self._comfy_record(home, tmp_path, dead_pid)
        shutil.rmtree(comfy)
        [rec] = _records(home)
        assert pending_pulls.delete_interrupted(rec["id"]) == 0
        assert not comfy.exists()
        assert _records(home) == []

    def test_listing_an_existing_folder_adds_no_file_or_folder(self, home, tmp_path, dead_pid):
        comfy = self._comfy_record(home, tmp_path, dead_pid)
        before = _fs_snapshot(comfy)
        [view] = pending_pulls.list_interrupted()
        assert view["bytes_on_disk"] == 4
        assert _fs_snapshot(comfy) == before

    def test_a_record_outside_the_allowed_folders_is_shown_invalid_and_untouched(
            self, home, tmp_path, dead_pid):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        victim = _hf_temp(elsewhere, "a.gguf", "e1", b"v" * 10, *dead_pid)
        with pending_pulls.tracking({"spec": "o/r:a.gguf"}):
            pending_pulls.note_source(source="hf-file", repo="o/r")
            pending_pulls.note_hf_files(elsewhere, [{"rel": "a.gguf", "etag": "e1"}])
        before = _fs_snapshot(elsewhere)
        [view] = pending_pulls.list_interrupted()
        assert view["invalid"] is True and view["resumable"] is False
        assert view["bytes_on_disk"] == 0
        assert _fs_snapshot(elsewhere) == before and victim.read_bytes() == b"v" * 10
        assert pending_pulls.list_orphans() == [] or all(
            o["file"] != victim.name for o in pending_pulls.list_orphans())
        assert pending_pulls.delete_interrupted(view["id"]) == 0
        assert victim.read_bytes() == b"v" * 10 and _records(home) == []

    def test_the_layout_matches_the_one_huggingface_hub_uses(self, tmp_path):
        import os
        from huggingface_hub._local_folder import get_local_download_paths
        deep = tmp_path / ("d" * 120) / ("e" * 120)
        for base in (tmp_path / "plain", deep):
            os.makedirs(pending_pulls._as_extended_path(base, 0), exist_ok=True)
            for rel in ("a.gguf", "sub/dir/b.safetensors", "x" * 90 + ".bin"):
                real = get_local_download_paths(base, rel)
                mine = pending_pulls._hf_layout(base, rel)
                assert (mine.metadata_path, mine.lock_path) == (
                    real.metadata_path, real.lock_path)

    def test_the_layout_refuses_a_path_that_is_not_plain(self, tmp_path):
        for rel in ("../a.gguf", "/etc/a", "a//b", ""):
            assert pending_pulls._hf_layout(tmp_path, rel) is None
        assert list(tmp_path.iterdir()) == []


class TestFinishedDownloadIsNotInterrupted:
    def _stop(self, home, dead_pid, request, folder):
        (folder / ".cache" / "huggingface" / "download").mkdir(parents=True)
        (folder / "config.json").write_text("{}", encoding="utf-8")
        _hf_temp(folder, "model.safetensors", "e1", b"p" * 30, *dead_pid)
        with pending_pulls.tracking(request):
            pending_pulls.note_source(source="hf-repo", repo="o/Repo", commit="a" * 40)
            pending_pulls.note_snapshot(folder, "Repo", True, 1000)

    def test_a_pull_of_the_same_repo_with_other_options_clears_the_record(
            self, home, dead_pid):
        folder = home / "Repo"
        self._stop(home, dead_pid, {"spec": "o/Repo", "type": "llm"}, folder)
        assert len(_records(home)) == 1
        with pending_pulls.tracking({"spec": "o/Repo", "type": "auto"}) as outcome:
            pending_pulls.note_source(source="hf-repo", repo="o/Repo", commit="a" * 40)
            pending_pulls.note_snapshot(folder, "Repo", False, 1000)
            outcome["ok"] = True
        assert _records(home) == []
        assert pending_pulls.list_interrupted() == []

    def test_a_pull_that_found_the_folder_complete_clears_the_record(self, home, dead_pid):
        folder = home / "Repo"
        self._stop(home, dead_pid, {"spec": "o/Repo", "type": "llm"}, folder)
        with pending_pulls.tracking({"spec": "o/Repo", "type": "auto"}) as outcome:
            pending_pulls.note_snapshot_complete(folder, "o/repo", None)
            outcome["ok"] = True
        assert _records(home) == []

    def test_another_repo_or_revision_in_the_list_is_kept(self, home, dead_pid):
        folder = home / "Repo"
        self._stop(home, dead_pid, {"spec": "o/Repo"}, folder)
        with pending_pulls.tracking({"spec": "o/Other"}) as outcome:
            pending_pulls.note_source(source="hf-repo", repo="o/Other", commit="a" * 40)
            pending_pulls.note_snapshot(home / "Other", "Other", True, 5)
            outcome["ok"] = True
        with pending_pulls.tracking({"spec": "o/Repo@dev"}) as outcome:
            pending_pulls.note_source(source="hf-repo", repo="o/Repo", revision="dev")
            pending_pulls.note_snapshot(folder, "Repo", False, 1000)
            outcome["ok"] = True
        assert [r["request"]["spec"] for r in _records(home)] == ["o/Repo"]

    def test_a_second_stopped_pull_replaces_the_record_of_the_first(self, home, dead_pid):
        folder = home / "Repo"
        self._stop(home, dead_pid, {"spec": "o/Repo", "type": "llm"}, folder)
        first = _records(home)[0]["id"]
        with pending_pulls.tracking({"spec": "o/Repo", "type": "auto"}):
            pending_pulls.note_source(source="hf-repo", repo="o/Repo", commit="a" * 40)
            pending_pulls.note_snapshot(folder, "Repo", False, 1000)
        [rec] = _records(home)
        assert rec["id"] != first and rec["snapshot"]["created"] is True

    def _complete_folder(self, home, size=1000):
        folder = home / "Repo"
        (folder / ".cache" / "huggingface" / "download").mkdir(parents=True)
        (folder / "model.safetensors").write_bytes(b"w" * size)
        return folder

    def test_delete_refuses_a_complete_unregistered_folder(self, home, monkeypatch):
        monkeypatch.setattr(mm, "load_registry", lambda: {})
        folder = self._complete_folder(home)
        with pending_pulls.tracking({"spec": "o/Repo"}):
            pending_pulls.note_source(source="hf-repo", repo="o/Repo", commit="a" * 40)
            pending_pulls.note_snapshot(folder, "Repo", True, 1000)
        [rec] = _records(home)
        assert (folder / "model.safetensors").stat().st_size == 1000
        with pytest.raises(pending_pulls.DownloadComplete):
            pending_pulls.delete_interrupted(rec["id"])
        assert (folder / "model.safetensors").stat().st_size == 1000
        assert [r["id"] for r in _records(home)] == [rec["id"]]

    def test_delete_refuses_a_folder_with_finished_files_when_the_size_is_unknown(
            self, home, monkeypatch):
        monkeypatch.setattr(mm, "load_registry", lambda: {})
        folder = self._complete_folder(home)
        with pending_pulls.tracking({"spec": "o/Repo"}):
            pending_pulls.note_source(source="hf-repo", repo="o/Repo", commit="a" * 40)
            pending_pulls.note_snapshot(folder, "Repo", True, None)
        [rec] = _records(home)
        with pytest.raises(pending_pulls.DownloadComplete):
            pending_pulls.delete_interrupted(rec["id"])
        assert (folder / "model.safetensors").stat().st_size == 1000

    def test_delete_still_removes_a_folder_with_a_temp_file_left(
            self, home, monkeypatch, dead_pid):
        monkeypatch.setattr(mm, "load_registry", lambda: {})
        folder = self._complete_folder(home)
        _hf_temp(folder, "model2.safetensors", "e1", b"p" * 30, *dead_pid)
        with pending_pulls.tracking({"spec": "o/Repo"}):
            pending_pulls.note_source(source="hf-repo", repo="o/Repo", commit="a" * 40)
            pending_pulls.note_snapshot(folder, "Repo", True, 1000)
        [rec] = _records(home)
        assert pending_pulls.delete_interrupted(rec["id"]) > 0
        assert not folder.exists()

    def test_delete_does_not_follow_a_folder_swapped_for_a_link(
            self, home, tmp_path, monkeypatch, dead_pid):
        import shutil
        from localm.model_manager import pull
        monkeypatch.setattr(mm, "load_registry", lambda: {})
        folder = home / "Repo"
        (folder / ".cache" / "huggingface" / "download").mkdir(parents=True)
        _hf_temp(folder, "model.safetensors", "e1", b"p" * 30, *dead_pid)
        with pending_pulls.tracking({"spec": "o/Repo"}):
            pending_pulls.note_source(source="hf-repo", repo="o/Repo", commit="a" * 40)
            pending_pulls.note_snapshot(folder, "Repo", True, 1000)
        [rec] = _records(home)
        outside = tmp_path / "precious"
        outside.mkdir()
        (outside / "keep.txt").write_text("keep", encoding="utf-8")
        real_lock = pull._snapshot_dest_lock

        import contextlib

        @contextlib.contextmanager
        def swap_then_lock(name):
            with real_lock(name):
                shutil.rmtree(folder)
                _link_or_skip(folder, outside)
                yield
        monkeypatch.setattr(pull, "_snapshot_dest_lock", swap_then_lock)
        assert pending_pulls.delete_interrupted(rec["id"]) == 0
        assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"


class TestSourceUrlsOfEveryShape:
    def test_a_full_repo_row_shows_the_repo_address_and_its_revision(self, home, dead_pid):
        folder = home / "Repo"
        (folder / ".cache" / "huggingface" / "download").mkdir(parents=True)
        _hf_temp(folder, "m.safetensors", "e1", b"p" * 3, *dead_pid)
        with pending_pulls.tracking({"spec": "o/Repo@dev"}):
            pending_pulls.note_source(source="hf-repo", repo="o/Repo", revision="dev",
                                      commit="a" * 40)
            pending_pulls.note_snapshot(folder, "Repo", False, 10)
        [view] = pending_pulls.list_interrupted()
        assert view["urls"] == ["https://huggingface.co/o/Repo/tree/dev"]

    def test_a_civitai_row_shows_the_model_version_address(self, home):
        part = home / "m.safetensors.part"
        part.write_bytes(b"x" * 3)
        with pending_pulls.tracking({"spec": "civitai:456"}):
            pending_pulls.note_source(source="civitai", repo="civitai:456")
            pending_pulls.note_part(part, "m.safetensors", 10)
        [view] = pending_pulls.list_interrupted()
        assert view["urls"] == ["https://civitai.com/api/v1/model-versions/456"]


HOLD_A_HF_LOCK = '''
    import sys
    from pathlib import Path
    from filelock import FileLock
    from huggingface_hub._local_folder import get_local_download_paths
    lock = FileLock(str(get_local_download_paths(Path(sys.argv[1]), sys.argv[2]).lock_path))
    with lock:
        print("HOLDING", flush=True)
        sys.stdin.read()
'''


class TestSecondPullOfOneHfFile:
    def _fakes(self, monkeypatch):
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda *a: None)
        monkeypatch.setattr("localm.model_manager.pull._published_digest", lambda *a: None)
        _fake_hf_head(monkeypatch, 5000, "etag1")
        downloads = []

        def download(repo_id, filename, local_dir, **kw):
            downloads.append(filename)
            raise requests.ConnectionError("connection dropped")
        monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
        return downloads

    def test_a_pull_of_a_file_another_process_downloads_is_refused_up_front(
            self, home, monkeypatch):
        import subprocess
        downloads = self._fakes(monkeypatch)
        child = subprocess.Popen(
            [sys.executable, "-I", "-c", __import__("textwrap").dedent(HOLD_A_HF_LOCK),
             str(home), "tiny-Q4.gguf"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            assert child.stdout.readline().strip() == "HOLDING", child.stderr.read()
            assert mm.pull_model("owner/repo:tiny-Q4.gguf", name="tiny") is False
            assert downloads == []
            assert _records(home) == []
            assert pending_pulls.list_interrupted() == []
        finally:
            child.stdin.close()
            child.wait(timeout=30)
        assert mm.pull_model("owner/repo:tiny-Q4.gguf", name="tiny") is False
        assert downloads == ["tiny-Q4.gguf"]
