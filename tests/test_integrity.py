# SPDX-License-Identifier: AGPL-3.0-or-later
"""The integrity check (``localm.integrity``): localm's files against the list of
the installed release, the record that spares a start from reading unchanged files,
the user's kept files, the native runtime's list, the startup check and its two
settings, the repair from a signed build, the routes and the bug-report count."""
from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import pytest

from localm import integrity, runtime_files, update_package, updater

FILES = {
    "VERSION": "1.0\n",
    "localm/a.py": "A = 1\n",
    "localm/b.py": "B = 2\n",
    "docs/c.md": "# c\n",
    "runtime/localm_llama_runtime/lib/.gitignore": "*\n",
}


def _statement(root: Path) -> dict:
    manifest = {"format": update_package.FORMAT, "version": "1.0",
                "files": update_package.files(root), "components": {}}
    (root / update_package.NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


@pytest.fixture
def install(tmp_path, monkeypatch):
    """An install of release 1.0 with its statement, and a data dir of its own."""
    import localm.config as cfg
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    root = tmp_path / "install"
    for rel, text in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode("utf-8"))
    _statement(root)
    monkeypatch.setattr(updater, "repo_root", lambda: root)
    monkeypatch.setattr(integrity, "_last", None)
    monkeypatch.setattr(integrity, "_pending", False)
    monkeypatch.setattr(integrity, "_running", False)
    return root


@pytest.fixture
def reads(monkeypatch):
    """Every path whose content the check reads in full."""
    seen: list = []
    real = runtime_files.sha256

    def spy(path):
        seen.append(Path(path).name)
        return real(path)

    monkeypatch.setattr(runtime_files, "sha256", spy)
    return seen


def _settings(monkeypatch, **values):
    from tests._net_config import set_config
    set_config(monkeypatch, lambda: dict(values))


# ------------------------------------------------------------------ the check

def test_an_untouched_install_matches_and_the_runtime_folder_is_left_to_its_list(install):
    r = integrity.check()
    assert r["available"] and r["version"] == "1.0"
    assert (r["differ"], r["missing"], r["unreadable"], r["count"]) == ([], [], [], 0)
    assert r["checked"] == 4, "the stated file under the runtime folder is not compared"
    assert r["runtime"]["present"] is False
    assert integrity.summary(r) == "every file matches localm 1.0"


def test_an_edited_a_shortened_and_a_deleted_file_are_each_named(install):
    (install / "localm" / "a.py").write_bytes(b"A = 9\n")
    (install / "localm" / "b.py").write_bytes(b"B\n")
    (install / "docs" / "c.md").unlink()
    r = integrity.check()
    assert r["differ"] == ["localm/a.py", "localm/b.py"]
    assert r["missing"] == ["docs/c.md"]
    assert r["count"] == 3
    assert "differs from the release: localm/a.py" in integrity.lines(r)
    assert "missing: docs/c.md" in integrity.lines(r)


def test_a_file_the_release_does_not_list_is_not_reported(install):
    (install / "localm" / "mine.py").write_bytes(b"x\n")
    assert integrity.check()["count"] == 0


@pytest.mark.parametrize("case", ["no statement", "other version"])
def test_without_a_matching_list_nothing_is_compared_and_the_reason_is_given(install, case):
    if case == "no statement":
        (install / update_package.NAME).unlink()
    else:
        (install / "VERSION").write_bytes(b"2.0\n")
    r = integrity.check()
    assert r["available"] is False and r["count"] == 0
    assert ("git clone" in r["reason"]) if case == "no statement" else ("2.0" in r["reason"])


def test_a_second_check_with_nothing_changed_reads_no_file_in_full(install, reads):
    first = integrity.check()
    assert first["hashed"] == 4 and len(reads) == 4
    reads.clear()
    second = integrity.check()
    assert second["count"] == 0
    assert second["hashed"] == 0 and reads == [], (
        "an unchanged file was read again although the record holds its size and time")


def test_a_file_changed_since_the_record_is_read_again_and_reported(install, reads):
    integrity.check()
    reads.clear()
    p = install / "localm" / "a.py"
    p.write_bytes(b"A = 9\n")
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    r = integrity.check()
    assert reads == ["a.py"]
    assert r["differ"] == ["localm/a.py"]


def test_a_full_check_reads_every_file_whatever_the_record_says(install, reads):
    integrity.check()
    p = install / "localm" / "a.py"
    st = p.stat()
    p.write_bytes(b"A = 9\n")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    reads.clear()
    assert integrity.check()["differ"] == [], "same size and time: the record stands"
    reads.clear()
    r = integrity.check(full=True)
    assert len(reads) == 4 and r["differ"] == ["localm/a.py"]


def test_a_record_of_another_version_is_not_used(install, reads):
    integrity.check()
    record = install.parent / "home" / "updates" / integrity.RECORD
    data = json.loads(record.read_text(encoding="utf-8"))
    data["version"] = "0.9"
    record.write_text(json.dumps(data), encoding="utf-8")
    reads.clear()
    integrity.check()
    assert len(reads) == 4


# ------------------------------------------------------------------- keeping

def test_a_kept_file_is_listed_apart_until_the_release_changes_it(install):
    p = install / "localm" / "a.py"
    p.write_bytes(b"A = 9\n")
    assert integrity.keep(["localm/a.py"]) == ["localm/a.py"]
    r = integrity.check()
    assert r["differ"] == [] and r["kept"] == ["localm/a.py"] and r["count"] == 0
    assert "1 kept as yours" in integrity.summary(r)
    manifest = json.loads((install / update_package.NAME).read_text(encoding="utf-8"))
    manifest["files"]["localm/a.py"]["sha256"] = "0" * 64
    (install / update_package.NAME).write_text(json.dumps(manifest), encoding="utf-8")
    assert integrity.check(full=True)["differ"] == ["localm/a.py"], (
        "a new release of the kept file must be reported again")


def test_a_deleted_file_kept_as_yours_is_listed_apart_and_no_longer_counted(install, release):
    from click.testing import CliRunner

    from localm.cli import main
    (install / "docs" / "c.md").unlink()
    assert integrity.check()["count"] == 1
    r = CliRunner().invoke(main, ["integrity", "--keep", "docs/c.md"])
    assert r.exit_code == 0 and "kept as yours: docs/c.md" in r.output, r.output
    r = integrity.check()
    assert (r["missing"], r["kept"], r["count"]) == ([], ["docs/c.md"], 0)
    assert integrity.repair(opener=release["opener"])["repaired"] == [], (
        "a kept deleted file is not repaired unless named")
    assert integrity.repair(["docs/c.md"], opener=release["opener"])["repaired"] == ["docs/c.md"]
    assert (install / "docs" / "c.md").read_bytes() == b"# c\n" and integrity.kept() == {}


def test_keep_refuses_a_file_the_release_does_not_state(install):
    with pytest.raises(integrity.IntegrityError, match="not a file of the release"):
        integrity.keep(["localm/zzz.py"])


# ------------------------------------------------------------------- runtime

def _runtime(install: Path) -> Path:
    lib = install / "runtime" / "localm_llama_runtime" / "lib"
    (lib / "rocblas").mkdir(parents=True, exist_ok=True)
    (lib / "llama.dll").write_bytes(b"L" * 1000)
    (lib / "rocblas" / "kernel.dat").write_bytes(b"K" * 500)
    (lib / ".localm-backend").write_bytes(b"cpu\n")
    return lib


def test_the_runtime_list_holds_its_files_and_not_its_own_records(install):
    lib = _runtime(install)
    assert integrity.write_runtime_list(lib) == 2
    data = json.loads((lib / runtime_files.NAME).read_text(encoding="utf-8"))
    assert set(data["files"]) == {"llama.dll", "rocblas/kernel.dat"}
    r = integrity.check()
    assert r["runtime"] == {"present": True, "listed": True, "full": False, "checked": 2,
                            "differ": [], "missing": [], "unreadable": [], "hashed": 0,
                            "error": None}


def test_every_record_of_localm_in_the_runtime_folder_stays_out_of_the_list(install):
    lib = _runtime(install)
    for name in (".localm-cuda-staged", ".localm-cuda-runtime.json", ".localm-cpu-overlay",
                 ".localm-cpu-overlay-failed", ".localm-kernel-packs", ".localm-cpu-tier",
                 ".localm-backend-choice", ".gitkeep"):
        (lib / name).write_bytes(b"{}")
    (lib / ".localm-cpu-tier.lock").mkdir()
    (lib / ".localm-cpu-tier.lock" / "owner.json").write_bytes(b"{}")
    integrity.write_runtime_list(lib)
    data = json.loads((lib / runtime_files.NAME).read_text(encoding="utf-8"))
    assert set(data["files"]) == {"llama.dll", "rocblas/kernel.dat"}


def test_a_cpu_tier_the_loader_set_aside_is_no_difference(install, monkeypatch):
    from localm import cpu_backend_select
    lib = _runtime(install)
    for tier in ("alderlake", "haswell", "zen4"):
        (lib / f"libggml-cpu-{tier}.so").write_bytes(tier.encode() * 100)
    integrity.write_runtime_list(lib)
    monkeypatch.setattr(cpu_backend_select, "_probe_score",
                        lambda c, d: {"libggml-cpu-haswell.so": 64}.get(c.name, 0))
    assert cpu_backend_select.ensure_cpu_tier_selected(lib) == "libggml-cpu-haswell.so"
    assert (lib / "_unused-libggml-cpu-zen4.so").is_file()
    assert not (lib / "libggml-cpu-zen4.so").exists()
    for full in (False, True):
        rt = integrity.check(full=full)["runtime"]
        assert (rt["differ"], rt["missing"], rt["checked"]) == ([], [], 5), (full, rt)
    (lib / "_unused-libggml-cpu-zen4.so").write_bytes(b"x")
    assert integrity.check()["runtime"]["differ"] == ["libggml-cpu-zen4.so"]
    (lib / "_unused-libggml-cpu-zen4.so").unlink()
    assert integrity.check()["runtime"]["missing"] == ["libggml-cpu-zen4.so"]


def test_a_runtime_file_or_list_that_cannot_be_read_is_reported_not_raised(
        install, monkeypatch):
    lib = _runtime(install)
    integrity.write_runtime_list(lib)
    real = os.stat

    def stat(p, *a, **k):
        if str(p).endswith("llama.dll"):
            raise PermissionError(13, "Permission denied", str(p))
        return real(p, *a, **k)

    monkeypatch.setattr(runtime_files.os, "stat", stat)
    r = integrity.check()
    assert r["runtime"]["unreadable"] == ["llama.dll"] and r["runtime"]["differ"] == []
    assert "native runtime file could not be read: llama.dll" in integrity.lines(r)
    monkeypatch.setattr(runtime_files.os, "stat", real)
    real_read = Path.read_text

    def read_text(self, *a, **k):
        if self.name == runtime_files.NAME:
            raise PermissionError(13, "Permission denied", str(self))
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", read_text)
    rt = integrity.check()["runtime"]
    assert rt["listed"] is False and rt["error"] == "PermissionError: Permission denied"
    assert str(lib) not in rt["error"]


def test_a_runtime_file_of_another_size_is_reported_without_reading_it(install, reads):
    lib = _runtime(install)
    integrity.write_runtime_list(lib)
    integrity.check()
    reads.clear()
    (lib / "llama.dll").write_bytes(b"L" * 999)
    (lib / "rocblas" / "kernel.dat").unlink()
    r = integrity.check()
    assert r["runtime"]["differ"] == ["llama.dll"]
    assert r["runtime"]["missing"] == ["rocblas/kernel.dat"]
    assert r["count"] == 2 and reads == []


def test_a_runtime_file_whose_time_changed_is_read_and_passes_when_equal(install, reads):
    lib = _runtime(install)
    integrity.write_runtime_list(lib)
    integrity.check()
    p = lib / "llama.dll"
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    reads.clear()
    r = integrity.check()
    assert reads == ["llama.dll"] and r["runtime"]["differ"] == []
    p.write_bytes(b"X" * 1000)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert integrity.check()["runtime"]["differ"] == []
    assert integrity.check(full=True)["runtime"]["differ"] == ["llama.dll"]


def test_a_runtime_without_a_list_is_said_to_have_none(install):
    _runtime(install)
    rt = integrity.check()["runtime"]
    assert rt["present"] is True and rt["listed"] is False and rt["checked"] == 0


def test_provisioning_writes_the_runtime_list(install):
    from localm.setup_llama import _record_provisioned_backend
    lib = _runtime(install)
    _record_provisioned_backend(lib, "cpu", build="b1")
    data = json.loads((lib / runtime_files.NAME).read_text(encoding="utf-8"))
    assert set(data["files"]) == {"llama.dll", "rocblas/kernel.dat"}


# --------------------------------------------------------------- the startup

def test_with_the_setting_off_a_start_reads_nothing(install, monkeypatch, reads):
    _settings(monkeypatch, integrity_check_at_start=False, integrity_notice=True)
    looked: list = []
    monkeypatch.setattr(integrity, "statement", lambda *a, **k: looked.append(a) or (None, ""))
    monkeypatch.setattr(integrity, "check", lambda *a, **k: looked.append(a) or {})
    assert integrity.startup(delay=0) is None
    assert looked == [] and reads == []
    assert integrity.state()["pending"] is False


def _start(command, monkeypatch):
    """The CLI start of *command* with its update steps and the command itself
    replaced: returns what ``integrity.startup`` was called with."""
    import click
    from click.testing import CliRunner

    from localm.cli import _core, main
    seen: list = []
    monkeypatch.setattr(integrity, "startup", lambda **k: seen.append(k))
    monkeypatch.setattr(_core, "_update_steps_at_start", lambda command: None)
    monkeypatch.setitem(main.commands, command, click.Command(command, callback=lambda: None))
    r = CliRunner().invoke(main, [command])
    assert r.exit_code == 0, r.output
    return seen


@pytest.mark.parametrize("command", ["gui", "serve"])
def test_a_server_start_starts_the_check_and_hands_it_the_end_of_its_command(
        install, monkeypatch, command):
    (call,) = _start(command, monkeypatch)
    assert set(call) == {"say", "on_close"}


@pytest.mark.parametrize("command", ["chat", "integrity", "run"])
def test_a_start_that_is_not_a_server_does_not_start_the_check(install, monkeypatch, command):
    assert _start(command, monkeypatch) == []


def test_the_end_of_the_command_ends_the_waiting_check_and_reads_nothing(
        install, monkeypatch, reads):
    _settings(monkeypatch, integrity_check_at_start=True, integrity_notice=True)
    ends: list = []
    thread = integrity.startup(delay=3600, on_close=ends.append)
    assert thread.is_alive() and integrity.state()["pending"] is True
    (end,) = ends
    end()
    assert not thread.is_alive(), "the startup check outlived its command"
    assert reads == [] and integrity.state()["pending"] is False
    assert integrity.state()["result"] is None


def test_the_end_of_the_command_ends_a_check_that_is_reading(install, monkeypatch):
    import threading
    _settings(monkeypatch, integrity_check_at_start=True)
    stop = threading.Event()
    stop.set()
    with pytest.raises(integrity.Stopped):
        integrity.check(stop=stop)
    reading = threading.Event()
    release_read = threading.Event()
    real = runtime_files.sha256
    read: list = []

    def slow(path):
        read.append(Path(path).name)
        reading.set()
        release_read.wait(10)
        return real(path)

    monkeypatch.setattr(runtime_files, "sha256", slow)
    ends: list = []
    thread = integrity.startup(delay=0, on_close=ends.append)
    assert reading.wait(10)
    (stopping,) = [c.cell_contents for c in ends[0].__closure__
                   if isinstance(c.cell_contents, threading.Event)]
    stopper = threading.Thread(target=ends[0])
    stopper.start()
    assert stopping.wait(10)
    release_read.set()
    stopper.join(10)
    thread.join(10)
    assert not thread.is_alive()
    assert len(read) == 1, f"the check read on after its end: {read}"
    assert integrity.state()["result"] is None, "a check ended midway keeps no result"


@pytest.mark.parametrize("notice", [True, False])
def test_the_startup_check_keeps_its_result_and_says_it_only_with_the_notice_on(
        install, monkeypatch, notice):
    _settings(monkeypatch, integrity_check_at_start=True, integrity_notice=notice)
    (install / "localm" / "a.py").write_bytes(b"A = 9\n")
    said: list = []
    thread = integrity.startup(say=said.append, delay=0)
    assert thread is not None and thread.daemon
    thread.join(30)
    state = integrity.state()
    assert state["result"]["differ"] == ["localm/a.py"] and state["pending"] is False
    assert state["notice"] is notice
    assert len(said) == (1 if notice else 0)
    if notice:
        assert "1 file(s) differ from localm 1.0" in said[0]


@pytest.mark.parametrize("mode", ["privacy", "full"])
def test_the_startup_check_writes_its_record_only_where_automatic_writes_are_allowed(
        install, monkeypatch, mode):
    _settings(monkeypatch, integrity_check_at_start=True, integrity_notice=False, mode=mode)
    monkeypatch.setattr("localm.audit.automatic_write_allowed",
                        lambda surface, cwd=None: mode != "privacy")
    integrity.startup(delay=0).join(30)
    record = install.parent / "home" / "updates" / integrity.RECORD
    assert record.exists() is (mode != "privacy")


def test_a_server_help_does_not_start_the_check(install, monkeypatch):
    from click.testing import CliRunner

    from localm.cli import _core, main
    seen: list = []
    monkeypatch.setattr(integrity, "startup", lambda **k: seen.append(k))
    monkeypatch.setattr(_core, "_update_steps_at_start", lambda command: None)
    CliRunner().invoke(main, ["serve", "--help"])
    assert seen == []


# ------------------------------------------------------------------ repair

def _build(files: dict, statement, modes: dict | None = None) -> bytes:
    """A build zip of *files* and *statement* (a dict, or the text of a broken one),
    each entry with the permission *modes* gives it (default 0o600, as a release
    zip records a plain file)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for rel, text in sorted(files.items()):
            info = zipfile.ZipInfo(rel)
            info.external_attr = (0o100000 | (modes or {}).get(rel, 0o600)) << 16
            z.writestr(info, text)
        z.writestr(update_package.NAME,
                   statement if isinstance(statement, str) else json.dumps(statement))
    return buf.getvalue()


@pytest.fixture
def release(install, monkeypatch):
    """The signed build of 1.0 behind a stand-in of the update server, signed with a
    test release key pinned for the test. Returns a setter for the build's bytes and
    the download opener."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from tests._update_signing import envelope, keyset
    key = Ed25519PrivateKey.generate()
    monkeypatch.setattr(updater, "_PINNED_KEYSET", keyset(
        1, [key], [Ed25519PrivateKey.generate() for _ in range(3)]))
    _settings(monkeypatch, update_url="https://updates.invalid")
    statement = json.loads((install / update_package.NAME).read_text(encoding="utf-8"))
    box = {"data": _build(FILES, statement), "key": key}

    def selection(path, params, opener=None):
        assert path == "/update/release" and params == {"version": "1.0"}
        return {"found": True, "version": "1.0", "asset": {"id": 7},
                "signature": envelope(box["data"], [box["key"]])}

    monkeypatch.setattr(updater, "_selection_request", selection)

    def opener(url, headers, timeout, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(box["data"])

    box["opener"] = opener
    return box


def _tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_repair_writes_exactly_the_differing_files_from_the_signed_build(install, release):
    (install / "localm" / "a.py").write_bytes(b"A = 9\n")
    (install / "docs" / "c.md").unlink()
    (install / "localm" / "b.py").write_bytes(b"B = 7\n")
    integrity.keep(["localm/b.py"])
    before = _tree(install)
    said: list = []
    res = integrity.repair(say=said.append, opener=release["opener"])
    assert res == {"version": "1.0", "repaired": ["localm/a.py", "docs/c.md"], "failed": []}
    after = _tree(install)
    assert after["localm/a.py"] == b"A = 1\n" and after["docs/c.md"] == b"# c\n"
    assert {k for k in after if after[k] != before.get(k)} == {
        "localm/a.py", "docs/c.md", ".localm-update/lock"}, (
        "besides the install's update lock, only the repaired files changed")
    assert after["localm/b.py"] == b"B = 7\n", "a kept file is not repaired unless named"
    assert not (install.parent / "home" / "updates" / "integrity-repair").exists()
    assert integrity.check()["count"] == 0


def test_repair_of_a_named_kept_file_puts_it_back_and_ends_keeping_it(install, release):
    (install / "localm" / "b.py").write_bytes(b"B = 7\n")
    integrity.keep(["localm/b.py"])
    res = integrity.repair(["localm/b.py"], opener=release["opener"])
    assert res["repaired"] == ["localm/b.py"]
    assert (install / "localm" / "b.py").read_bytes() == b"B = 2\n"
    assert integrity.kept() == {}


def test_a_build_that_does_not_verify_repairs_nothing(install, release):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from localm.bugreport import LocalmError
    (install / "localm" / "a.py").write_bytes(b"A = 9\n")
    release["key"] = Ed25519PrivateKey.generate()
    before = _tree(install)
    with pytest.raises(LocalmError):
        integrity.repair(opener=release["opener"])
    assert _tree(install) == before


def test_a_file_the_signed_build_states_otherwise_is_not_written(install, release):
    files = dict(FILES, **{"localm/a.py": "A = 5\n"})
    tmp = install.parent / "other"
    for rel, text in files.items():
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp / rel).write_bytes(text.encode("utf-8"))
    statement = {"format": update_package.FORMAT, "version": "1.0",
                 "files": update_package.files(tmp), "components": {}}
    release["data"] = _build(files, statement)
    (install / "localm" / "a.py").write_bytes(b"A = 9\n")
    res = integrity.repair(opener=release["opener"])
    assert res["repaired"] == [] and res["failed"][0]["path"] == "localm/a.py"
    assert "other content" in res["failed"][0]["reason"]
    assert (install / "localm" / "a.py").read_bytes() == b"A = 9\n"


def test_repair_refuses_a_path_outside_the_release(install, release):
    with pytest.raises(integrity.IntegrityError):
        integrity.repair(["../outside.py"], opener=release["opener"])
    with pytest.raises(integrity.IntegrityError):
        integrity.repair(["runtime/localm_llama_runtime/lib/.gitignore"],
                         opener=release["opener"])


def test_repair_waits_for_an_update_in_progress(install, release):
    from localm.update_mechanism import job
    record = job.record_path(install)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_bytes(b"{}")
    (install / "localm" / "a.py").write_bytes(b"A = 9\n")
    with pytest.raises(integrity.IntegrityError, match="update"):
        integrity.repair(opener=release["opener"])
    assert (install / "localm" / "a.py").read_bytes() == b"A = 9\n"


def _with_scripts(install: Path, release: dict) -> None:
    """Add a launcher (one of ``_apply_update.EXECUTABLES``) and a script the build
    marks executable to the install and its signed build, both recorded 0600 and
    0755 as a release zip records them."""
    files = dict(FILES, **{"localm.sh": "#!/bin/sh\necho launch\n",
                           "tools/x.sh": "#!/bin/sh\necho x\n"})
    for rel in ("localm.sh", "tools/x.sh"):
        (install / rel).parent.mkdir(parents=True, exist_ok=True)
        (install / rel).write_bytes(files[rel].encode())
    statement = _statement(install)
    release["data"] = _build(files, statement, {"tools/x.sh": 0o755})


def test_a_repaired_script_keeps_the_execute_permission_the_build_gives_it(install, release):
    _with_scripts(install, release)
    for rel in ("localm.sh", "tools/x.sh", "localm/a.py"):
        (install / rel).write_bytes(b"edited\n")
    res = integrity.repair(opener=release["opener"])
    assert sorted(res["repaired"]) == ["localm.sh", "localm/a.py", "tools/x.sh"], res
    assert (install / "localm.sh").read_bytes() == b"#!/bin/sh\necho launch\n"
    with zipfile.ZipFile(io.BytesIO(release["data"])) as z:
        decided = {rel: integrity._executable(z.getinfo(rel), rel)
                   for rel in ("localm.sh", "tools/x.sh", "localm/a.py")}
    assert decided == {"localm.sh": True, "tools/x.sh": True, "localm/a.py": False}
    if os.name != "nt":
        modes = {rel: os.stat(install / rel).st_mode & 0o111
                 for rel in ("localm.sh", "tools/x.sh", "localm/a.py")}
        assert modes["localm.sh"] and modes["tools/x.sh"], modes
        assert modes["localm/a.py"] == 0, modes


def test_a_failed_write_leaves_no_unfinished_copy(install, release, monkeypatch):
    (install / "localm" / "a.py").write_bytes(b"A = 9\n")
    real = Path.write_bytes

    def half(self, data):
        if self.name.endswith(".localm-repair"):
            real(self, data[:2])
            raise OSError(28, "No space left on device")
        return real(self, data)

    monkeypatch.setattr(Path, "write_bytes", half)
    res = integrity.repair(opener=release["opener"])
    assert res["repaired"] == [] and res["failed"][0]["path"] == "localm/a.py"
    assert list(install.rglob("*.localm-repair")) == []
    assert (install / "localm" / "a.py").read_bytes() == b"A = 9\n"


def test_repair_does_not_write_through_a_folder_that_leads_outside(install, release, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "c.md").write_bytes(b"theirs\n")
    import shutil
    shutil.rmtree(install / "docs")
    try:
        (install / "docs").symlink_to(outside, target_is_directory=True)
    except OSError as e:
        pytest.skip(f"this system cannot make a folder link here: {e}")
    res = integrity.repair(["docs/c.md"], opener=release["opener"])
    assert res["repaired"] == [] and "outside" in res["failed"][0]["reason"], res
    assert (outside / "c.md").read_bytes() == b"theirs\n"


def test_an_update_that_starts_during_the_download_stops_the_repair(
        install, release, monkeypatch):
    from localm.update_mechanism import job
    (install / "localm" / "a.py").write_bytes(b"A = 9\n")
    real = updater.verify_signature

    def verify_while_an_update_starts(data, signature):
        record = job.record_path(install)
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_bytes(b"{}")
        return real(data, signature)

    monkeypatch.setattr(updater, "verify_signature", verify_while_an_update_starts)
    with pytest.raises(integrity.IntegrityError, match="update"):
        integrity.repair(opener=release["opener"])
    assert (install / "localm" / "a.py").read_bytes() == b"A = 9\n"


def test_a_build_whose_list_cannot_be_read_repairs_nothing_and_says_so(install, release):
    from localm.bugreport import LocalmError
    (install / "localm" / "a.py").write_bytes(b"A = 9\n")
    release["data"] = _build(FILES, "{not json")
    with pytest.raises(LocalmError, match="cannot be read"):
        integrity.repair(opener=release["opener"])
    assert (install / "localm" / "a.py").read_bytes() == b"A = 9\n"
    assert not (install.parent / "home" / "updates" / "integrity-repair").exists()


# ------------------------------------------------------------ CLI and report

def test_the_command_lists_keeps_and_reports_its_exit_code(install):
    from click.testing import CliRunner

    from localm.cli import main
    (install / "localm" / "a.py").write_bytes(b"A = 9\n")
    r = CliRunner().invoke(main, ["integrity"])
    assert r.exit_code == 1 and "localm/a.py" in r.output, r.output
    r = CliRunner().invoke(main, ["integrity", "--keep"])
    assert r.exit_code == 0 and "Kept as yours: localm/a.py" in r.output, r.output
    r = CliRunner().invoke(main, ["integrity"])
    assert r.exit_code == 0 and "kept as yours: localm/a.py" in r.output, r.output


def test_keep_or_repair_with_only_runtime_differences_still_exits_1(install):
    from click.testing import CliRunner

    from localm.cli import main
    lib = _runtime(install)
    integrity.write_runtime_list(lib)
    (lib / "llama.dll").write_bytes(b"L" * 999)
    for flag in ("--keep", "--repair"):
        r = CliRunner().invoke(main, ["integrity", flag])
        assert r.exit_code == 1, (flag, r.output)
        assert "nothing to" in r.output and "llama.dll" in r.output, r.output


def test_localm_bug_report_carries_the_counts_without_a_check_before_and_writes_nothing(
        install, monkeypatch):
    from click.testing import CliRunner

    from localm import bugreport
    from localm.bugreport import diagnostics
    from localm.cli import main
    (install / "localm" / "a.py").write_bytes(b"A = 9\n")
    (install / "docs" / "c.md").unlink()
    integrity.keep(["docs/c.md"])
    monkeypatch.setattr(bugreport, "stdin_is_interactive", lambda: False)
    home = install.parent / "home"
    updates = home / "updates"
    before = sorted(p.name for p in updates.iterdir())
    r = CliRunner().invoke(main, ["bug-report", "-m", "it broke", "--no-log"])
    assert r.exit_code == 0, r.output
    (saved,) = [p for p in home.rglob("*.md")]
    report = saved.read_text(encoding="utf-8")
    assert ("- Files differing from the release: 1 (1 changed, 0 missing, 0 native "
            "runtime); 1 kept as the user's") in report, report
    (line,) = [ln for ln in report.splitlines() if "Files differing" in ln]
    assert "a.py" not in line and "c.md" not in line
    assert integrity.state()["result"] is None, "no check had run in this process"
    assert sorted(p.name for p in updates.iterdir()) == before, "the report wrote a record"
    assert diagnostics.collect_diagnostics()["files_differing"] == {
        "checked": True, "differ": 1, "missing": 0, "kept": 1, "runtime_differ": 0}


def test_a_failing_source_of_the_bug_report_is_named_not_raised(install, monkeypatch):
    import localm.bugreport as br
    from localm.bugreport import diagnostics

    def broken():
        raise RuntimeError("no")

    monkeypatch.setitem(br._diagnostic_sources, "files_differing", broken)
    assert diagnostics.collect_diagnostics()["files_differing"] == {"error": "RuntimeError"}


# ------------------------------------------------------------------ routes

OWNER_KEY = "owner-admin-key-integrity-abc123"


@pytest.fixture
def client(install, monkeypatch):
    import localm.config as cfg
    from fastapi.testclient import TestClient

    from localm import auth
    from localm.inference.http_server import create_app
    home = install.parent / "home"
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setenv("LOCALM_API_KEY", OWNER_KEY)
    scoped = auth.create_key("dev", ["config:read", "config:write"],
                             allow_privileged=True)["key"]
    with TestClient(create_app(None)) as c:
        yield c, scoped


def test_the_routes_check_and_keep_for_the_owner_only(install, client):
    c, scoped = client
    owner = {"Authorization": f"Bearer {OWNER_KEY}"}
    r = c.get("/api/integrity", headers=owner)
    assert r.status_code == 200 and r.json()["result"] is None, r.text
    (install / "localm" / "a.py").write_bytes(b"A = 9\n")
    r = c.post("/api/integrity/check", headers=owner, json={"full": True})
    assert r.status_code == 200 and r.json()["differ"] == ["localm/a.py"], r.text
    assert c.get("/api/integrity", headers=owner).json()["result"]["count"] == 1
    r = c.post("/api/integrity/keep", headers={"Authorization": f"Bearer {scoped}"})
    assert r.status_code == 403 and integrity.kept() == {}, r.text
    r = c.post("/api/integrity/repair", headers={"Authorization": f"Bearer {scoped}"})
    assert r.status_code == 403, r.text
    r = c.post("/api/integrity/keep", headers=owner, json={"paths": ["localm/a.py"]})
    assert r.status_code == 200 and r.json()["kept"] == ["localm/a.py"], r.text
    assert r.json()["result"]["kept"] == ["localm/a.py"]
    r = c.post("/api/integrity/keep", headers=owner, json={"paths": "localm/a.py"})
    assert r.status_code == 400


def test_a_failing_check_keep_or_repair_answers_without_exception_text(
        install, client, release, monkeypatch):
    c, _scoped = client
    owner = {"Authorization": f"Bearer {OWNER_KEY}"}
    secret = "SECRET-DETAIL-4711"
    statement = (install / update_package.NAME).read_bytes()
    broken = dict(json.loads(statement), format=secret)
    (install / update_package.NAME).write_bytes(json.dumps(broken).encode())
    for r in (c.post("/api/integrity/check", headers=owner),
              c.post("/api/integrity/repair", headers=owner),
              c.post("/api/integrity/keep", headers=owner, json={"paths": ["localm/a.py"]})):
        assert secret not in r.text and "Traceback" not in r.text, r.text
        assert "PackageError" not in r.text and "Error(" not in r.text, r.text
        assert "cannot be read" in r.text, r.text
    (install / update_package.NAME).write_bytes(statement)
    (install / "localm" / "a.py").write_bytes(b"A = 9\n")
    monkeypatch.setattr(integrity, "_HELD_S", 0.0)
    real = os.replace

    def held(src, dst):
        if str(dst).endswith("a.py"):
            raise PermissionError(f"[WinError 32] {secret}")
        return real(src, dst)

    monkeypatch.setattr(integrity.os, "replace", held)
    monkeypatch.setattr(updater, "download", lambda aid, dest, opener=None:
                        release["opener"](None, None, None, dest))
    r = c.post("/api/integrity/repair", headers=owner)
    assert r.status_code == 200 and r.json()["repaired"] is True, r.text
    assert r.json()["failed"][0]["path"] == "localm/a.py"
    assert secret not in r.text and "WinError" not in r.text, r.text


def test_the_repair_route_says_why_nothing_was_repaired(install, client):
    c, _scoped = client
    (install / update_package.NAME).unlink()
    r = c.post("/api/integrity/repair", headers={"Authorization": f"Bearer {OWNER_KEY}"})
    assert r.status_code == 200 and r.json()["repaired"] is False
    assert "git clone" in r.json()["error"]
