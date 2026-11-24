# SPDX-License-Identifier: AGPL-3.0-or-later
"""``comfy_client.run_leaving_nothing``: a private job's files leave ComfyUI's
output and temp folders and its history, however the job ends. A real HTTP
server stands in for ComfyUI, with real folders."""

from __future__ import annotations

import http.server
import json
import threading
import urllib.parse

import pytest

from localm.media import comfy_client as cc


class _Comfy:
    def __init__(self, root):
        self.output, self.temp = root / "output", root / "temp"
        self.output.mkdir()
        self.temp.mkdir()
        self.history: dict = {}
        self.running: list = []
        self.stops = True

    def job(self, pid, *files, status="success"):
        outputs = {}
        for n, (kind, name) in enumerate(files):
            (self.output if kind == "output" else self.temp).joinpath(name).write_bytes(
                f"{pid} {name} prompt text".encode())
            outputs[str(n)] = {"images": [{"filename": name, "subfolder": "", "type": kind}]}
        self.history[pid] = {"outputs": outputs, "status": {
            "status_str": status, "completed": status == "success", "messages": []}}


@pytest.fixture
def comfy(tmp_path):
    state = _Comfy(tmp_path)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            query = dict(urllib.parse.parse_qsl(url.query))
            if url.path == "/view":
                path = (state.output if query["type"] == "output" else state.temp) / query["filename"]
                if not path.is_file():
                    return self._json({}, 404)
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return self.wfile.write(data)
            if url.path.startswith("/history/"):
                pid = url.path.rsplit("/", 1)[1]
                return self._json({pid: state.history[pid]} if pid in state.history else {})
            if url.path == "/queue":
                return self._json({"queue_running": [[0, p] for p in state.running],
                                   "queue_pending": []})
            return self._json({}, 404)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path == "/history":
                for pid in json.loads(body).get("delete", []):
                    state.history.pop(pid, None)
                return self._json({})
            if self.path == "/interrupt":
                if state.stops:
                    state.running.clear()
                return self._json({}, 200 if state.stops else 500)
            if self.path == "/prompt":
                return self._json({"prompt_id": "p1"})
            return self._json({}, 404)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state.url = f"http://127.0.0.1:{srv.server_address[1]}"
    yield state
    srv.shutdown()


def _submit(comfy):
    kind, pid = cc.comfy_submit_prompt(comfy.url, {"1": {"class_type": "SaveImage", "inputs": {}}})
    assert (kind, pid) == (cc.SUBMIT_OK, "p1")


def _run(comfy, job, output_dir=None):
    return cc.run_leaving_nothing(job, comfy_output_dir=str(output_dir or comfy.output))


def _left(comfy):
    return sorted(p.name for d in (comfy.output, comfy.temp) for p in d.iterdir())


def test_every_file_of_a_finished_job_is_removed(comfy, tmp_path):
    def job():
        _submit(comfy)
        comfy.job("p1", ("output", "a.png"), ("output", "b.png"), ("temp", "preview.png"))
        (comfy.output / "other.png").write_bytes(b"someone else's")
        status, entry = cc.comfy_poll_until_done(comfy.url, "p1", max_poll_seconds=5,
                                                 sleep_seconds=0.01)
        assert status == cc.POLL_FINISHED
        info = cc.select_output_info(entry, ("images",))
        sha = cc.comfy_fetch_output(comfy.url, info, tmp_path / "mine.png", timeout=5)
        assert cc.contain_comfy_artifacts(
            comfy.url, "p1", info, comfy_output_dir=str(comfy.output), output_sha256=sha,
            delete_outputs=True) == ""
        return True, "saved"
    assert _run(comfy, job) == (True, "saved")
    assert _left(comfy) == ["other.png"] and comfy.history == {}
    assert (tmp_path / "mine.png").is_file()


def test_a_failed_job_leaves_nothing(comfy):
    def job():
        _submit(comfy)
        comfy.job("p1", ("temp", "preview.png"), status="error")
        return False, "ComfyUI execution failed"
    assert _run(comfy, job) == (False, "ComfyUI execution failed")
    assert _left(comfy) == [] and comfy.history == {}


def test_a_job_that_finished_as_it_was_cancelled_leaves_nothing(comfy):
    def job():
        _submit(comfy)
        comfy.job("p1", ("output", "a.png"))
        status, _v = cc.comfy_poll_until_done(comfy.url, "p1", max_poll_seconds=5,
                                              cancel_check=lambda: True)
        assert status == cc.POLL_CANCELLED
        return False, "Generation cancelled."
    assert _run(comfy, job) == (False, "Generation cancelled.")
    assert _left(comfy) == [] and comfy.history == {}


def test_a_running_job_is_stopped_before_its_files_are_read(comfy):
    def job():
        _submit(comfy)
        comfy.running.append("p1")
        comfy.job("p1", ("temp", "preview.png"), status="error")
        raise RuntimeError("the connection broke")
    with pytest.raises(RuntimeError, match="the connection broke"):
        _run(comfy, job)
    assert comfy.running == [] and _left(comfy) == []


def test_a_job_that_cannot_be_stopped_is_said_not_to_be_clean(comfy):
    def job():
        _submit(comfy)
        comfy.running.append("p1")
        comfy.stops = False
        return False, "Generation cancelled."
    ok, message = _run(comfy, job)
    assert ok is False and "could not be asked which files" in message


def test_an_output_that_stays_withholds_the_result(comfy, tmp_path):
    elsewhere = tmp_path / "elsewhere" / "output"
    elsewhere.mkdir(parents=True)

    def job():
        _submit(comfy)
        comfy.job("p1", ("output", "b.png"))
        cc.comfy_poll_until_done(comfy.url, "p1", max_poll_seconds=5, sleep_seconds=0.01)
        return True, "saved"
    ok, message = _run(comfy, job, elsewhere)
    assert ok is False and message.startswith("Output withheld: WARNING:")
    assert (comfy.output / "b.png").is_file()


def test_a_preview_that_stays_is_named_but_does_not_withhold(comfy, tmp_path):
    elsewhere = tmp_path / "elsewhere" / "output"
    elsewhere.mkdir(parents=True)

    def job():
        _submit(comfy)
        comfy.job("p1", ("temp", "preview.png"))
        cc.comfy_poll_until_done(comfy.url, "p1", max_poll_seconds=5, sleep_seconds=0.01)
        return True, "saved"
    ok, message = _run(comfy, job, elsewhere)
    assert ok is True and message.startswith("saved\nWARNING:") and "temp folder" in message


def test_a_file_with_other_content_than_comfyui_serves_is_left_alone(comfy, tmp_path):
    elsewhere = tmp_path / "elsewhere" / "output"
    elsewhere.mkdir(parents=True)
    (elsewhere / "b.png").write_bytes(b"the user's own picture")

    def job():
        _submit(comfy)
        comfy.job("p1", ("output", "b.png"))
        cc.comfy_poll_until_done(comfy.url, "p1", max_poll_seconds=5, sleep_seconds=0.01)
        return True, "saved"
    ok, _message = _run(comfy, job, elsewhere)
    assert ok is False
    assert (elsewhere / "b.png").read_bytes() == b"the user's own picture"


def test_a_filename_outside_the_folder_is_never_followed(comfy, tmp_path):
    victim = tmp_path / "victim.png"
    victim.write_bytes(b"x")

    def job():
        _submit(comfy)
        comfy.history["p1"] = {"outputs": {"1": {"images": [
            {"filename": "../victim.png", "subfolder": "", "type": "output"}]}},
            "status": {"status_str": "success", "completed": True, "messages": []}}
        cc.comfy_poll_until_done(comfy.url, "p1", max_poll_seconds=5, sleep_seconds=0.01)
        return True, "saved"
    ok, message = _run(comfy, job)
    assert ok is False and "out-of-bounds" in message and victim.is_file()


def test_outside_a_private_run_nothing_is_removed(comfy):
    _submit(comfy)
    comfy.job("p1", ("output", "a.png"), ("temp", "preview.png"))
    status, _entry = cc.comfy_poll_until_done(comfy.url, "p1", max_poll_seconds=5,
                                              sleep_seconds=0.01)
    assert status == cc.POLL_FINISHED
    assert _left(comfy) == ["a.png", "preview.png"] and "p1" in comfy.history


@pytest.mark.parametrize("privacy,kept,expected", [
    (True, False, True), (True, True, False), (False, False, False)])
def test_the_facade_asks_for_it_in_privacy_mode_that_keeps_no_media(
        privacy, kept, expected, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from localm.plugins import media_config
    asked = []

    def spy(job, **kw):
        asked.append(kw)
        return job()
    monkeypatch.setattr(cc, "run_leaving_nothing", spy)
    monkeypatch.setattr("localm.audit.privacy_keeps", lambda kind: kept)
    ref = SimpleNamespace(generate=lambda *a, **k: (True, "ok"), NEEDS_COMFY_SERVER=True)
    facade = media_config.make_backend_facade("localm.plugins.builtin.image", ref, "image")
    s = {"output_dir": "o", "workdir": ""}
    assert facade.generate(s, "p", tmp_path / "x.png", privacy=privacy) == (True, "ok")
    assert asked == ([{"comfy_output_dir": "o", "comfy_workdir": None}] if expected else [])
