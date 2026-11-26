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
    from localm.bugreport import diagnostics
    monkeypatch.setattr(diagnostics, "_integrity", {"checked": False})
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
    (install / "localm" / "a.py").write_text("A = 9\n", encoding="utf-8")
    (install / "localm" / "b.py").write_text("B\n", encoding="utf-8")
    (install / "docs" / "c.md").unlink()
    r = integrity.check()
    assert r["differ"] == ["localm/a.py", "localm/b.py"]
    assert r["missing"] == ["docs/c.md"]
    assert r["count"] == 3
    assert "differs from the release: localm/a.py" in integrity.lines(r)
    assert "missing: docs/c.md" in integrity.lines(r)


def test_a_file_the_release_does_not_list_is_not_reported(install):
    (install / "localm" / "mine.py").write_text("x\n", encoding="utf-8")
    assert integrity.check()["count"] == 0


@pytest.mark.parametrize("case", ["no statement", "other version"])
def test_without_a_matching_list_nothing_is_compared_and_the_reason_is_given(install, case):
    if case == "no statement":
        (install / update_package.NAME).unlink()
    else:
        (install / "VERSION").write_text("2.0\n", encoding="utf-8")
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
    p.write_text("A = 9\n", encoding="utf-8")
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    r = integrity.check()
    assert reads == ["a.py"]
    assert r["differ"] == ["localm/a.py"]


def test_a_full_check_reads_every_file_whatever_the_record_says(install, reads):
    integrity.check()
    p = install / "localm" / "a.py"
    st = p.stat()
    p.write_text("A = 9\n", encoding="utf-8")
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
    p.write_text("A = 9\n", encoding="utf-8")
    assert integrity.keep(["localm/a.py"]) == ["localm/a.py"]
    r = integrity.check()
    assert r["differ"] == [] and r["kept"] == ["localm/a.py"] and r["count"] == 0
    assert "1 kept as yours" in integrity.summary(r)
    manifest = json.loads((install / update_package.NAME).read_text(encoding="utf-8"))
    manifest["files"]["localm/a.py"]["sha256"] = "0" * 64
    (install / update_package.NAME).write_text(json.dumps(manifest), encoding="utf-8")
    assert integrity.check(full=True)["differ"] == ["localm/a.py"], (
        "a new release of the kept file must be reported again")


def test_keep_refuses_a_file_the_release_does_not_state(install):
    with pytest.raises(integrity.IntegrityError, match="not a file of the release"):
        integrity.keep(["localm/zzz.py"])


# ------------------------------------------------------------------- runtime

def _runtime(install: Path) -> Path:
    lib = install / "runtime" / "localm_llama_runtime" / "lib"
    (lib / "rocblas").mkdir(parents=True, exist_ok=True)
    (lib / "llama.dll").write_bytes(b"L" * 1000)
    (lib / "rocblas" / "kernel.dat").write_bytes(b"K" * 500)
    (lib / ".localm-backend").write_text("cpu\n", encoding="utf-8")
    return lib


def test_the_runtime_list_holds_its_files_and_not_its_own_records(install):
    lib = _runtime(install)
    assert integrity.write_runtime_list(lib) == 2
    data = json.loads((lib / runtime_files.NAME).read_text(encoding="utf-8"))
    assert set(data["files"]) == {"llama.dll", "rocblas/kernel.dat"}
    r = integrity.check()
    assert r["runtime"] == {"present": True, "listed": True, "full": False, "checked": 2,
                            "differ": [], "missing": [], "hashed": 0}


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
    assert integrity.startup("gui", delay=0) is None
    assert looked == [] and reads == []
    assert integrity.state()["pending"] is False


def test_a_start_that_is_not_a_server_reads_nothing(install, monkeypatch, reads):
    _settings(monkeypatch, integrity_check_at_start=True)
    assert integrity.startup("chat", delay=0) is None
    assert reads == []


@pytest.mark.parametrize("notice", [True, False])
def test_the_startup_check_keeps_its_result_and_says_it_only_with_the_notice_on(
        install, monkeypatch, notice):
    _settings(monkeypatch, integrity_check_at_start=True, integrity_notice=notice)
    (install / "localm" / "a.py").write_text("A = 9\n", encoding="utf-8")
    said: list = []
    thread = integrity.startup("serve", say=said.append, delay=0)
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
    integrity.startup("gui", delay=0).join(30)
    record = install.parent / "home" / "updates" / integrity.RECORD
    assert record.exists() is (mode != "privacy")


def test_every_cli_start_but_a_help_hands_its_command_to_the_startup_check(
        install, monkeypatch):
    from click.testing import CliRunner

    from localm.cli import _core, main
    seen: list = []
    monkeypatch.setattr(integrity, "startup", lambda command, say=None: seen.append(command))
    monkeypatch.setattr(_core, "_update_steps_at_start", lambda command: None)
    CliRunner().invoke(main, ["integrity", "--help"])
    assert seen == []
    CliRunner().invoke(main, ["integrity"])
    assert seen == ["integrity"]


# ------------------------------------------------------------------ repair

def _build(files: dict, statement: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for rel, text in sorted(files.items()):
            z.writestr(rel, text)
        z.writestr(update_package.NAME, json.dumps(statement))
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
    (install / "localm" / "a.py").write_text("A = 9\n", encoding="utf-8")
    (install / "docs" / "c.md").unlink()
    (install / "localm" / "b.py").write_text("B = 7\n", encoding="utf-8")
    integrity.keep(["localm/b.py"])
    before = _tree(install)
    said: list = []
    res = integrity.repair(say=said.append, opener=release["opener"])
    assert res == {"version": "1.0", "repaired": ["localm/a.py", "docs/c.md"], "failed": []}
    after = _tree(install)
    assert after["localm/a.py"] == b"A = 1\n" and after["docs/c.md"] == b"# c\n"
    assert {k for k in after if after[k] != before.get(k)} == {"localm/a.py", "docs/c.md"}
    assert after["localm/b.py"] == b"B = 7\n", "a kept file is not repaired unless named"
    assert not (install.parent / "home" / "updates" / "integrity-repair").exists()
    assert integrity.check()["count"] == 0


def test_repair_of_a_named_kept_file_puts_it_back_and_ends_keeping_it(install, release):
    (install / "localm" / "b.py").write_text("B = 7\n", encoding="utf-8")
    integrity.keep(["localm/b.py"])
    res = integrity.repair(["localm/b.py"], opener=release["opener"])
    assert res["repaired"] == ["localm/b.py"]
    assert (install / "localm" / "b.py").read_bytes() == b"B = 2\n"
    assert integrity.kept() == {}


def test_a_build_that_does_not_verify_repairs_nothing(install, release):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from localm.bugreport import LocalmError
    (install / "localm" / "a.py").write_text("A = 9\n", encoding="utf-8")
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
    (install / "localm" / "a.py").write_text("A = 9\n", encoding="utf-8")
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
    record.write_text("{}", encoding="utf-8")
    (install / "localm" / "a.py").write_text("A = 9\n", encoding="utf-8")
    with pytest.raises(integrity.IntegrityError, match="update"):
        integrity.repair(opener=release["opener"])
    assert (install / "localm" / "a.py").read_bytes() == b"A = 9\n"


# ------------------------------------------------------------ CLI and report

def test_the_command_lists_keeps_and_reports_its_exit_code(install):
    from click.testing import CliRunner

    from localm.cli import main
    (install / "localm" / "a.py").write_text("A = 9\n", encoding="utf-8")
    r = CliRunner().invoke(main, ["integrity"])
    assert r.exit_code == 1 and "localm/a.py" in r.output, r.output
    r = CliRunner().invoke(main, ["integrity", "--keep"])
    assert r.exit_code == 0 and "Kept as yours: localm/a.py" in r.output, r.output
    r = CliRunner().invoke(main, ["integrity"])
    assert r.exit_code == 0 and "kept as yours: localm/a.py" in r.output, r.output


def test_the_bug_report_carries_counts_and_no_paths(install):
    from localm.bugreport import diagnostics
    assert diagnostics._integrity_counts() == {"checked": False}
    (install / "localm" / "a.py").write_text("A = 9\n", encoding="utf-8")
    integrity.run()
    counts = diagnostics.collect_diagnostics()["files_differing"]
    assert counts == {"checked": True, "differ": 1, "missing": 0, "kept": 0,
                      "runtime_differ": 0}
    assert "a.py" not in json.dumps(counts)


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
    (install / "localm" / "a.py").write_text("A = 9\n", encoding="utf-8")
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


def test_the_repair_route_says_why_nothing_was_repaired(install, client):
    c, _scoped = client
    (install / update_package.NAME).unlink()
    r = c.post("/api/integrity/repair", headers={"Authorization": f"Bearer {OWNER_KEY}"})
    assert r.status_code == 200 and r.json()["repaired"] is False
    assert "git clone" in r.json()["error"]
