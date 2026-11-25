# SPDX-License-Identifier: AGPL-3.0-or-later
"""A re-provision downloads and verifies the new build in a staging directory
before it touches the installed runtime, so a failed download (offline, a bad
archive) leaves the working runtime in place."""

import os
import subprocess
import sys
import types

import pytest
from click.testing import CliRunner

import localm.setup_llama as sl
from localm.inference.backends.llamacpp import _loader
from localm.setup_llama.download import ArtifactError
from localm.update_mechanism import disk, job, mover, steps


def _installed(target):
    target.mkdir(parents=True)
    (target / sl._lib_name()).write_bytes(b"working runtime")
    (target / "ggml-base.dll").write_bytes(b"old ggml")
    (target / ".localm-backend").write_text("cpu b11757", encoding="utf-8")
    (target / ".localm-backend-choice").write_text('{"requested": "vulkan"}', encoding="utf-8")
    (target / ".gitkeep").write_text("", encoding="utf-8")
    return {p.name: p.read_bytes() for p in target.iterdir()}


def _offline(monkeypatch):
    def no_network(url, dest, *a, **k):
        raise ArtifactError(f"could not download {url}: network is unreachable")
    monkeypatch.setattr(sl, "_download", no_network)
    monkeypatch.setattr(sl, "_resolve_backend_asset",
                        lambda backend, cuda_line=None, tag=None:
                        (f"https://example.invalid/{backend}.zip", None, "b11928"))
    monkeypatch.setattr(sl, "_install_runtime_wheel", lambda pkg: True)
    monkeypatch.setattr(sl.sys.stdin, "isatty", lambda: False)


def test_failed_download_keeps_the_runtime(tmp_path, monkeypatch):
    target = tmp_path / "lib"
    before = _installed(target)
    _offline(monkeypatch)
    with pytest.raises(SystemExit):
        sl._provision_with_fallback("cpu", target, None, False, assume_yes=True)
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before
    assert not sl._staging_path(target).exists()


def test_failed_url_download_keeps_the_runtime(tmp_path, monkeypatch):
    target = tmp_path / "lib"
    before = _installed(target)
    _offline(monkeypatch)
    with pytest.raises(SystemExit):
        sl.cli._provision_from_url("https://example.invalid/x.zip", target,
                                   sl._lib_name(), None)
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before


def test_a_staged_build_replaces_the_runtime_and_keeps_the_choice_record(tmp_path):
    target = tmp_path / "lib"
    _installed(target)
    (target / "rocblas").mkdir()
    (target / "rocblas" / "old.dat").write_bytes(b"old")

    def fill(staging):
        (staging / sl._lib_name()).write_bytes(b"new runtime")
        (staging / "rocblas").mkdir()
        (staging / "rocblas" / "new.dat").write_bytes(b"new")
        return "b11928"

    assert sl._replace_target(target, fill) == "b11928"
    names = sorted(p.name for p in target.iterdir())
    assert names == sorted([sl._lib_name(), ".localm-backend-choice", ".gitkeep", "rocblas"])
    assert (target / sl._lib_name()).read_bytes() == b"new runtime"
    assert [p.name for p in (target / "rocblas").iterdir()] == ["new.dat"]
    assert not sl._staging_path(target).exists()


def test_a_fill_that_raises_changes_nothing(tmp_path):
    target = tmp_path / "lib"
    before = _installed(target)

    def fill(staging):
        (staging / sl._lib_name()).write_bytes(b"half")
        raise ArtifactError("sha256 mismatch")

    with pytest.raises(ArtifactError):
        sl._replace_target(target, fill)
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before
    assert not sl._staging_path(target).exists()


def _snapshot(target):
    return {p.name: p.read_bytes() for p in target.iterdir() if p.is_file()}


def _fill_with(content):
    def fill(staging):
        (staging / sl._lib_name()).write_bytes(content)
        (staging / ".localm-backend").write_text("vulkan b99999", encoding="utf-8")
        return "b99999"
    return fill


def _put(target, content=b"new working build", into=None):
    """What setup-llama does with a build: provision it aside, then put it in place."""
    with sl.staged_runtime(target, into) as staged:
        sl._replace_target(staged, _fill_with(content))
        if into is None:
            sl.put_in_place(target, staged)


def test_a_build_that_does_not_load_changes_nothing(tmp_path):
    target = tmp_path / "lib"
    before = _installed(target)
    with pytest.raises(SystemExit) as exc:
        with sl.staged_runtime(target) as staged:
            sl._replace_target(staged, _fill_with(b"new build that does not load"))
            sl._replace_target(staged, _fill_with(b"fallback that does not load either"))
            sl.sys.exit(1)
    assert exc.value.code == 1
    assert _snapshot(target) == before
    assert [p.name for p in tmp_path.iterdir()] == ["lib"]


def test_a_build_that_loads_replaces_the_old_one_whole(tmp_path):
    target = tmp_path / "lib"
    _installed(target)
    _put(target)
    assert (target / sl._lib_name()).read_bytes() == b"new working build"
    assert (target / ".localm-backend").read_text(encoding="utf-8") == "vulkan b99999"
    assert not (target / "ggml-base.dll").exists()
    assert (target / ".gitkeep").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [".localm-update", "lib"]
    assert [p.name for p in (tmp_path / ".localm-update").iterdir()] == ["lock"]


def test_the_build_aside_is_the_one_that_is_tested(tmp_path, monkeypatch):
    target = tmp_path / "lib"
    _installed(target)
    monkeypatch.setenv("LLAMA_CPP_LIB", "the user's")
    seen = []
    with sl.staged_runtime(target) as staged:
        seen.append(os.environ["LLAMA_CPP_LIB"])
    assert seen == [str(staged / sl._lib_name())]
    assert os.environ["LLAMA_CPP_LIB"] == "the user's"


def test_the_users_choice_goes_with_the_new_build(tmp_path):
    target = tmp_path / "lib"
    _installed(target)
    sl._record_backend_choice(target, "cuda", "no driver")
    _put(target)
    assert sl._read_backend_choice(target)["requested"] == "cuda"
    with sl.staged_runtime(target) as staged:
        assert sl._read_backend_choice(staged)["requested"] == "cuda"
        sl._record_backend_choice(staged, None)
        sl._replace_target(staged, _fill_with(b"another"))
        sl.put_in_place(target, staged)
    assert sl._read_backend_choice(target) is None


def test_a_build_for_an_update_stays_aside(tmp_path):
    target, into = tmp_path / "lib", tmp_path / "aside"
    before = _installed(target)
    _put(target, into=into)
    assert _snapshot(target) == before
    assert (into / sl._lib_name()).read_bytes() == b"new working build"


def test_a_runtime_in_use_is_refused_before_anything_changes(tmp_path, monkeypatch):
    target = tmp_path / "lib"
    before = _installed(target)
    monkeypatch.setattr(sl, "_files_in_use",
                        lambda t: [target / sl._lib_name()] if t.name == target.name else [])
    with pytest.raises(sl.RuntimeInUseError):
        _put(target)
    assert _snapshot(target) == before


_KILLED = """
import os, sys
from pathlib import Path
from localm.update_mechanism import disk, steps
count, at = 0, int(sys.argv[2])
real = os.rename
def rename(*args):
    global count
    if count == at:
        os._exit(9)
    count += 1
    return real(*args)
from localm import setup_llama as sl
target = Path(sys.argv[1])
staged = target.with_name("lib.incoming")
staged.mkdir()
for name in (sl._lib_name(), "ggml-base.dll", "new.dll", ".localm-backend"):
    (staged / name).write_bytes(b"new " + name.encode())
os.rename = rename
sl.put_in_place(target, staged)
"""


def test_a_setup_killed_while_it_replaces_the_runtime_is_undone_by_the_next_use(tmp_path):
    states = []
    for at in range(40):
        target = tmp_path / str(at) / "lib"
        target.parent.mkdir()
        before = _installed(target)
        done = subprocess.run([sys.executable, "-c", _KILLED, str(target), str(at)],
                              capture_output=True, text=True, timeout=120,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if done.returncode == 0:
            states.append("done")
            break
        assert done.returncode == 9, done.stderr
        assert steps.unsettled(target.parent) is None
        new = _snapshot(target) if _snapshot(target) != before else None
        assert new is None or (len(new) == 5 and new["new.dll"] == b"new new.dll"), at
        states.append("old" if new is None else "new")
    assert states == sorted(states, key=["old", "new", "done"].index)
    assert states[-1] == "done" and states.count("old") > 6


def test_a_runtime_that_cannot_be_put_back_is_installed_again(tmp_path, monkeypatch):
    target = tmp_path / "lib"
    _installed(target)
    place = steps.Place(tmp_path, "ab" * 16)
    (place.new(mover.WHOLE) / "lib").mkdir()
    (place.new(mover.WHOLE) / "lib" / sl._lib_name()).write_bytes(b"half")
    with job.Lock(tmp_path) as lock:
        job.start(lock, place.id, old="1", new="2")
        job.advance(lock, job.MOVING)
        job.advance(lock, job.STEPS)
        steps.move_in(tmp_path, place.id, mover.WHOLE)
    assert disk.remove_tree(mover.area(tmp_path, place.id, mover.WHOLE, 1).old)
    assert "is lost" in (steps.unsettled(tmp_path) or "")
    assert sl.cli._settle(target) is True

    def loads():
        found = _loader._settled_binary_dir(os.environ["LLAMA_CPP_LIB"])
        return (found / sl._lib_name()).read_bytes() == b"new working build", "another runtime"
    build = tmp_path / "build"
    build.mkdir()
    (build / sl._lib_name()).write_bytes(b"new working build")
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: target)
    monkeypatch.setattr(sl, "_native_loads_ok", loads)
    monkeypatch.setattr(sl, "_install_runtime_wheel", lambda pkg: True)
    monkeypatch.setattr(sl, "_verify", lambda: None)
    monkeypatch.setattr(sl.cli, "_refresh_install_record", lambda target: None)
    monkeypatch.setitem(sys.modules, "localm_llama_runtime",
                        types.SimpleNamespace(LIB_DIR=str(target), lib_dir=lambda: str(target)))
    done = CliRunner().invoke(sl.main, ["--from", str(build), "--yes"])
    assert done.exit_code == 0, done.output
    assert "is installed again" in done.output
    assert (target / sl._lib_name()).read_bytes() == b"new working build"
    assert steps.unsettled(tmp_path) is None and sl.cli._settle(target) is False


def test_a_failed_provision_puts_the_pin_back(monkeypatch):
    import localm.config as config
    store = {"llama_runtime_pin": "b11000"}
    monkeypatch.setattr(config, "load_config", lambda *a, **k: dict(store))

    def _update(fn, *a, **k):
        fn(store)
    monkeypatch.setattr(config, "update_config", _update)
    before = sl.cli._stored_pin()
    store["llama_runtime_pin"] = "b12345"
    sl.cli._restore_pin(before)
    assert store["llama_runtime_pin"] == "b11000"
    store.pop("llama_runtime_pin")
    sl.cli._restore_pin(None)
    assert "llama_runtime_pin" not in store


def test_a_folder_that_is_there_is_not_taken_for_an_update(tmp_path):
    target, into = tmp_path / "lib", tmp_path / "aside"
    _installed(target)
    into.mkdir()
    (into / "mine.txt").write_text("the user's")
    with pytest.raises(FileExistsError), sl.staged_runtime(target, into):
        pass
    assert (into / "mine.txt").read_text() == "the user's"


def _prove(folder):
    return CliRunner().invoke(sl.main, ["--prove", str(folder)], standalone_mode=True).exit_code


def test_only_a_build_that_does_not_work_is_called_unusable(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_CPP_LIB", "")
    folder = tmp_path / "lib"
    assert _prove(folder) == 1
    _installed(folder)
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: (False, "no driver"))
    assert _prove(folder) == sl.BUILD_UNUSABLE_EXIT

    def not_run():
        raise sl.LoadTestNotRun("no temporary folder")
    monkeypatch.setattr(sl, "_native_loads_ok", not_run)
    assert _prove(folder) == 1
