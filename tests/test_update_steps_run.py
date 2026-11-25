# SPDX-License-Identifier: AGPL-3.0-or-later
"""The steps of an update carried out, its children, its probe and the small data files
it can put back."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from localm.update_mechanism import disk, job, mover, steps, steps_data, steps_probe, steps_update
from localm.update_mechanism.mover import Unreadable
from tests import _update_mechanism as um

ID = um.UPDATE_ID


# ------------------------------------------------------------------ children

def _child(code, **kw):
    said: list = []
    result = steps_update.run_child([sys.executable, "-c", code], cwd=os.getcwd(),
                                    env=dict(os.environ), say=said.append, **kw)
    return result, said


def test_a_childs_lines_are_passed_on_and_its_last_one_is_the_reason():
    result, said = _child("import sys; print('one', flush=True); print('two', file=sys.stderr); sys.exit(3)")
    assert result == (3, "two") and said == ["one", "two"]


def test_a_child_that_goes_silent_is_ended():
    started = time.monotonic()
    (status, reason), said = _child("import time; print('x', flush=True); time.sleep(120)",
                                    silence_s=10.0)
    assert status is None and "printed nothing for 10 s" in reason and said == ["x"]
    assert time.monotonic() - started < 60


def test_a_program_that_is_missing_is_a_reason(tmp_path):
    status, reason = steps_update.run_child([tmp_path / "none"], cwd=tmp_path, env={},
                                            say=lambda _t: None)
    assert status is None and "could not be started" in reason


# --------------------------------------------------------------------- parts

def _steps(**kw):
    spec = {"format": steps_update.FORMAT, "sync": None, "torch": [], "uv": None, "parts": [],
            "kept": [], "notes": []}
    return {**spec, **kw}


@pytest.fixture
def update(tmp_path):
    """An install in phase STEPS: ``(root, home, lock)``."""
    root, home = tmp_path / "root", tmp_path / "data"
    um.install(root)
    um.stage(root)
    home.mkdir()
    with job.Lock(root) as lock:
        job.start(lock, ID, old="1", new="2")
        job.advance(lock, job.MOVING)
        job.advance(lock, job.STEPS)
        yield root, home, lock


def _run(update, spec, env=None):
    root, home, lock = update
    disk.write_json(mover.work_folder(root, ID) / steps_update.STEPS_FILE, spec)
    said: list = []
    done = steps_update.run(lock, ID, uv="uv", home=home, env=env or dict(os.environ),
                            say=said.append)
    return done, said


def _folders_part(update, label="music runtime"):
    root, home, _lock = update
    base = home / "runtimes" / "koboldcpp"
    um.write(base, {"v1-nocuda/launcher": "old"})
    job.own_part(root, ID, base)
    um.write(steps.Place(base, ID).new(mover.WHOLE), {"v2-nocuda/launcher": "new"})
    return base, {"kind": "folders", "label": label, "root": job.part_path(root, base)}


def test_a_new_tag_folder_is_added_beside_the_old_one_and_taken_out_again(update):
    root, _home, lock = update
    base, part = _folders_part(update)
    _done, said = _run(update, _steps(parts=[part]))
    assert (base / "v2-nocuda" / "launcher").exists() and (base / "v1-nocuda" / "launcher").exists()
    assert said == ["Replacing the music runtime..."]
    assert job.restore(lock).errors == []
    assert sorted(p.name for p in base.iterdir()) == [".localm-update", "v1-nocuda"]
    assert um.left(base) == []


def test_a_step_that_fails_takes_the_earlier_ones_back(update, monkeypatch):
    root, _home, lock = update
    base, part = _folders_part(update)
    before = um.tree(root)

    def fails(*_a):
        raise mover.Refused("the managed ComfyUI does not exist")
    monkeypatch.setattr(steps_update.Run, "_comfyui", fails)
    (base.parent / "comfyui").mkdir()
    comfy = {"kind": "comfyui", "label": "managed ComfyUI", "removed": [], "packages": None,
             "root": job.part_path(root, base.parent / "comfyui")}
    with pytest.raises(steps.Failed, match="managed ComfyUI does not exist"):
        _run(update, _steps(parts=[part, comfy]))
    assert (base / "v2-nocuda").exists()
    assert job.restore(lock).errors == []
    assert um.tree(root) == before
    assert not (base / "v2-nocuda").exists()


def _runtime_part(update):
    root, _home, _lock = update
    folder = root / "runtime" / "localm_llama_runtime"
    um.write(folder, {"lib/llama.dll": "old", "lib/.gitignore": "*", "lib/rocblas/a.dat": "a",
                      "lib/mine/kept.txt": "the user's"})
    job.own_part(root, ID, folder)
    staged = steps.Place(folder, ID).beside("lib.incoming")
    um.write(staged, {"llama.dll": "new", "extra.dll": "new"})
    return folder, {"kind": "runtime", "label": "cpu runtime", "root": job.part_path(root, folder),
                    "name": "lib", "beside": "lib.incoming", "backend": "cpu", "from": "b1",
                    "to": "b2", "unproven": False, "keep": [".gitignore", ".gitkeep"],
                    "folders": ["rocblas"]}


def _proofs(monkeypatch, *codes):
    answers = list(codes)
    asked: list = []

    def localm(self, *args):
        asked.append(args)
        return answers.pop(0)
    monkeypatch.setattr(steps_update.Run, "_localm", localm)
    return asked


def test_a_runtime_that_works_replaces_the_old_one(update, monkeypatch):
    _root, _home, lock = update
    folder, part = _runtime_part(update)
    asked = _proofs(monkeypatch, (0, "OK"))
    _run(update, _steps(parts=[part]))
    assert asked == [("setup-llama", "--prove", str(folder / "lib.incoming"))]
    assert um.tree(folder / "lib").keys() == {"llama.dll", "extra.dll", ".gitignore", "mine",
                                              "mine/kept.txt"}
    assert not (folder / "lib.incoming").exists()
    job.restore(lock)
    assert (folder / "lib" / "llama.dll").read_text() == "old"
    assert (folder / "lib" / "rocblas" / "a.dat").exists()
    assert not (folder / "lib" / "extra.dll").exists()


def test_a_runtime_that_does_not_work_here_is_not_moved_in(update, monkeypatch):
    folder, part = _runtime_part(update)
    _proofs(monkeypatch, (steps_update.UNUSABLE_EXIT, "no driver"), (0, "OK"))
    done, _said = _run(update, _steps(parts=[part]))
    assert (folder / "lib" / "llama.dll").read_text() == "old"
    assert done.kept == [{"backend": "cpu", "kept": "b1", "target": "b2", "why": "no driver"}]
    assert "kept the cpu runtime b1" in done.notes[0]
    assert not (folder / "lib.incoming").exists()


def test_no_runtime_that_works_fails_the_update(update, monkeypatch):
    folder, part = _runtime_part(update)
    _proofs(monkeypatch, (steps_update.UNUSABLE_EXIT, "no driver"),
            (steps_update.UNUSABLE_EXIT, "wrong layout"))
    with pytest.raises(steps.Failed, match="no driver.*wrong layout"):
        _run(update, _steps(parts=[part]))
    assert (folder / "lib" / "llama.dll").read_text() == "old"


def test_steps_that_cannot_be_read_fail_the_update(update):
    with pytest.raises(steps.Failed, match="steps of an update"):
        _run(update, {"format": "else"})


def test_what_was_prepared_is_discarded_with_the_update(update):
    root, _home, _lock = update
    folder, runtime = _runtime_part(update)
    base, part = _folders_part(update)
    disk.write_json(mover.work_folder(root, ID) / steps_update.STEPS_FILE,
                    _steps(parts=[part, runtime]))
    steps_update.discard(root, ID)
    assert not (folder / "lib.incoming").exists() and um.left(base) == [] == um.left(folder)


def test_what_an_update_prepared_goes_once_the_update_is_gone(update):
    _root, _home, lock = update
    base, _part = _folders_part(update)
    steps.sweep(base)
    assert um.left(base) == [ID]
    job.restore(lock)
    steps.sweep(base)
    assert um.left(base) == []


def test_the_exit_status_of_a_proof_is_the_one_setup_llama_uses():
    from localm import setup_llama
    assert steps_update.UNUSABLE_EXIT == setup_llama.BUILD_UNUSABLE_EXIT


# ---------------------------------------------------------------------- data

def test_the_copied_files_are_the_small_ones_an_older_build_may_not_read():
    from localm import updater
    assert set(steps_data.FILES) == {n for n in updater._DOWNGRADE_DATA if n.endswith(".json")}
    assert set(steps_data.FOLDERS) < set(updater._DOWNGRADE_DATA)


def _data(home):
    return {p.name: p.read_text() for p in home.iterdir() if p.is_file()}


def test_data_files_the_new_version_wrote_are_put_back(update):
    root, home, lock = update
    um.write(home, {"config.json": "{}", "registry.json": "{1}", "auth.json": "secret"})
    before = _data(home)
    steps_data.keep(root, ID, home)
    (home / "registry.json").write_text("{migrated, longer}")
    (home / "sessions.json").write_text("made by the new version")
    (home / "auth.json").unlink()
    assert job.restore(lock).errors == []
    assert _data(home) == before
    kept = root / ".localm-update" / "kept" / ID / "data"
    assert _data(kept) == {"registry.json": "{migrated, longer}",
                           "sessions.json": "made by the new version"}


def test_plugins_the_new_version_changed_are_put_back(update):
    root, home, lock = update
    um.write(home, {"plugins/chat/plugin.toml": "1", "plugins/mine/code.py": "x"})
    before = um.tree(home)
    steps_data.keep(root, ID, home)
    um.write(home, {"plugins/chat/plugin.toml": "2", "plugins/added/plugin.toml": "new"})
    (home / "plugins" / "mine" / "code.py").unlink()
    assert job.restore(lock).errors == []
    assert um.tree(home) == before


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_a_data_file_that_went_comes_back_with_its_permissions(update):
    root, home, lock = update
    um.write(home, {"config.json": "{}", "auth.json": "secret", "plugins/mine/run.sh": "x"})
    (home / "auth.json").chmod(0o600)
    (home / "plugins" / "mine" / "run.sh").chmod(0o755)
    before = um.tree(home)
    steps_data.keep(root, ID, home)
    for name in ("config.json", "auth.json", "plugins/mine/run.sh"):
        (home / name).unlink()
    assert job.restore(lock).errors == []
    assert um.tree(home) == before


def _access(path) -> list:
    done = subprocess.run([os.path.join(os.environ["SystemRoot"], "System32", "icacls.exe"),
                           str(path)], capture_output=True, text=True, **um.NO_WINDOW)
    assert done.returncode == 0, done.stderr
    return sorted(line.replace(str(path), "").strip() for line in done.stdout.splitlines()[:-2])


@pytest.mark.skipif(os.name != "nt", reason="Windows permissions")
def test_a_secret_stays_the_users_alone_in_its_copy_and_when_it_comes_back(update):
    from localm import storekit
    root, home, lock = update
    um.write(home, {"auth.json": "secret", "config.json": "{}"})
    assert storekit.restrict_file_perms(home / "auth.json")
    before = {n: _access(home / n) for n in ("auth.json", "config.json")}
    assert before["auth.json"] != before["config.json"]
    steps_data.keep(root, ID, home)
    assert _access(mover.work_folder(root, ID) / "data" / "auth.json") == before["auth.json"]
    for name in before:
        (home / name).unlink()
    assert job.restore(lock).errors == []
    assert {n: _access(home / n) for n in before} == before
    assert (home / "auth.json").read_text() == "secret"


def test_a_plugins_folder_that_came_is_taken_out(update):
    root, home, lock = update
    steps_data.keep(root, ID, home)
    um.write(home, {"plugins/chat/plugin.toml": "2"})
    assert job.restore(lock).errors == []
    assert um.tree(home) == {}


def test_a_large_plugins_folder_is_left_alone(update, monkeypatch):
    root, home, lock = update
    monkeypatch.setattr(steps_data, "MAX_FOLDER_FILES", 1)
    um.write(home, {"plugins/a": "1", "plugins/b": "2"})
    steps_data.keep(root, ID, home)
    um.write(home, {"plugins/a": "changed"})
    assert job.restore(lock).errors == []
    assert (home / "plugins" / "a").read_text() == "changed"


def test_data_files_that_did_not_change_are_left_alone(update):
    root, home, lock = update
    um.write(home, {"config.json": "{}"})
    steps_data.keep(root, ID, home)
    stamp = os.stat(home / "config.json").st_mtime_ns
    (mover.work_folder(root, ID) / "data" / "config.json").write_text("damaged copy")
    assert job.restore(lock).errors == []
    assert os.stat(home / "config.json").st_mtime_ns == stamp


def test_a_damaged_copy_is_not_put_back(update):
    root, home, lock = update
    um.write(home, {"config.json": "{}"})
    steps_data.keep(root, ID, home)
    (home / "config.json").write_text("{changed}")
    (mover.work_folder(root, ID) / "data" / "config.json").write_text("damaged copy")
    result = job.restore(lock)
    assert result.state == "failed" and "is damaged" in result.errors[0]


def test_a_data_folder_that_moved_is_left_alone(update):
    root, home, lock = update
    um.write(home, {"config.json": "{}"})
    steps_data.keep(root, ID, home)
    home.rename(home.with_name("elsewhere"))
    assert job.restore(lock).errors == []


def test_a_list_of_data_files_with_another_path_is_refused(update):
    root, home, lock = update
    disk.write_json(mover.work_folder(root, ID) / "data.json",
                    {"home": str(home), "files": {"../x": None}, "modes": {}, "folders": []})
    with pytest.raises(Unreadable):
        job.restore(lock)


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_the_copies_are_the_users_alone(update):
    root, home, _lock = update
    um.write(home, {"auth.json": "secret"})
    steps_data.keep(root, ID, home)
    folder = mover.work_folder(root, ID) / "data"
    assert folder.stat().st_mode & 0o77 == 0 and (folder / "auth.json").stat().st_mode & 0o77 == 0


# --------------------------------------------------------------------- probe

_APP = """
import json, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"version": %r}).encode()
        self.send_response(200); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
s = HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=lambda: (sys.stdin.read(), s.shutdown()), daemon=True).start()
print("starting")
print(json.dumps({"port": s.server_port}), flush=True)
s.serve_forever()
"""


def _probe(tmp_path, code, version="2", **kw):
    package = tmp_path / "tree" / "localm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(code)
    return steps_probe.probe(sys.executable, tmp_path / "tree", version=version,
                             home=tmp_path / "home", env=dict(os.environ), **kw)


def test_the_probe_passes_when_the_version_answers(tmp_path):
    assert _probe(tmp_path, _APP % "2") is None


def test_the_probe_names_another_version(tmp_path):
    assert _probe(tmp_path, _APP % "1") == "localm 1 answered instead of 2"


def test_the_probe_gives_the_reason_a_version_does_not_start(tmp_path):
    reason = _probe(tmp_path, "raise RuntimeError('a module is missing')")
    assert reason.startswith("localm 2 does not start") and "a module is missing" in reason


def test_the_probe_ends_a_version_that_hangs(tmp_path):
    started = time.monotonic()
    reason = _probe(tmp_path, "import time; print('loading', flush=True); time.sleep(120)",
                    limit_s=3)
    assert "did not start within 3 seconds: loading" in reason
    assert time.monotonic() - started < 60


def test_a_port_that_does_not_answer_is_named(tmp_path):
    code = "import json, time; print(json.dumps({'port': 9}), flush=True); time.sleep(120)"
    assert "does not answer" in _probe(tmp_path, code, limit_s=2)


def test_the_real_app_answers_the_probe_and_writes_no_session(tmp_path):
    from localm import _version
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.json").write_text(json.dumps(
        {"mode": "full", "update_url": "", "bugreport_upload_url": ""}), encoding="utf-8")
    reason = steps_probe.probe(sys.executable, um.REPO, version=_version.read_version(),
                               home=home, env=dict(os.environ))
    assert reason is None
    assert not (home / "sessions").exists() and not (home / "bug-reports").exists()
