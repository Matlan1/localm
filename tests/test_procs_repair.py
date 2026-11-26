# SPDX-License-Identifier: AGPL-3.0-or-later
"""A start whose process record (``localm/procs.py``) cannot be imported goes on, and
``localm.procs_repair`` repairs it: a damaged file from the signed build of the
installed release, a missing package by the repair of the installed parts, and
otherwise a warning that says why and what repairs it.

Each test starts a real ``import localm`` in a fresh interpreter, in a copy of
localm's code with a release statement of its own. The update server is a stand-in
loaded before localm (the release request, the download and the package install);
the signature check, the update lock and the writes are localm's own."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

import localm
from localm import procs_repair, update_package
from tests._fake_clone import CREATE_NO_WINDOW

REPO = Path(__file__).resolve().parents[1]
HIDDEN = {"creationflags": CREATE_NO_WINDOW} if os.name == "nt" else {}
BROKEN = "import localm_no_such_module_for_the_gate\n"
PACKAGE = "localm_gate_test_pkg"

# Loaded before localm in the started interpreter: it puts the stand-in of the update
# server into localm.updater, and the stand-in of uv's install into
# localm.updater_parts, as each is imported, then imports localm and says whether the
# process is recorded.
_START = r'''
import importlib.machinery, json, os, sys
SPEC = json.loads(os.environ["PROCS_REPAIR_TEST"])

def standin_updater(m):
    m._PINNED_KEYSET = SPEC["keyset"]
    def selection(path, params, opener=None):
        assert path == "/update/release" and params == {"version": SPEC["version"]}
        return {"found": True, "version": SPEC["version"], "asset": {"id": 7},
                "signature": SPEC["signature"]}
    m._selection_request = selection
    real = m.download
    def download(asset_id, dest, *, timeout=120.0, opener=None):
        with open(SPEC["seen"], "a", encoding="utf-8") as f:
            f.write(json.dumps({"download_timeout": timeout}) + "\n")
        def serve(url, headers, t, d):
            d.parent.mkdir(parents=True, exist_ok=True)
            with open(SPEC["build"], "rb") as src:
                d.write_bytes(src.read())
        return real(asset_id, dest, timeout=timeout, opener=serve)
    m.download = download
    m._uv_for_deps = lambda target: SPEC["uv"]

def standin_parts(m):
    real = m._reinstall_deps
    def reinstall(root, *, runner, say):
        def install(cmd):
            with open(SPEC["seen"], "a", encoding="utf-8") as f:
                f.write(json.dumps({"uv": cmd, "UV_HTTP_TIMEOUT": os.environ.get(
                    "UV_HTTP_TIMEOUT")}) + "\n")
            site = SPEC["site"]
            with open(os.path.join(site, SPEC["package"] + ".py"), "w") as f:
                f.write("HERE = True\n")
            info = os.path.join(site, SPEC["package"] + "-1.0.dist-info")
            os.makedirs(info)
            with open(os.path.join(info, "METADATA"), "w") as f:
                f.write("Metadata-Version: 2.1\nName: " + SPEC["package"] + "\nVersion: 1.0\n\n")
            return 0
        return real(root, runner=install, say=say)
    m._reinstall_deps = reinstall

HOOKS = {"localm.updater": standin_updater, "localm.updater_parts": standin_parts}

class Finder:
    def find_spec(self, name, path=None, target=None):
        if name not in HOOKS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        run = spec.loader.exec_module
        def exec_module(module):
            run(module)
            HOOKS[name](module)
        spec.loader.exec_module = exec_module
        return spec

sys.meta_path.insert(0, Finder())
import localm
try:
    import localm.procs as p
    record = p.current() is not None
except Exception as e:
    record = False
print("STARTED " + json.dumps({"file": localm.__file__, "record": record, "pid": os.getpid(),
                               "repair_loaded": "localm.procs_repair" in sys.modules}),
      flush=True)
'''


def _copy_code(root: Path) -> None:
    def skip(folder, names):
        return [n for n in names if n == "__pycache__" or n.endswith(".pyc")
                or (n == "static" and Path(folder).name == "gui")]
    shutil.copytree(REPO / "localm", root / "localm", ignore=skip)


class Install:
    """A copy of localm's code at *root* with the statement of release *version*."""

    def __init__(self, tmp_path: Path, *, procs_imports: str = "",
                 with_package: bool = False) -> None:
        self.root = root = tmp_path / "inst"
        self.home = tmp_path / "home"
        self.seen = tmp_path / "seen.jsonl"
        self.version = localm.__version__
        _copy_code(root)
        (root / "VERSION").write_text(self.version, encoding="utf-8")
        (root / "pyproject.toml").write_text(
            f'[project]\nname = "localm"\nversion = "{self.version}"\n'
            f'dependencies = ["{PACKAGE.replace("_", "-")}>=1"]\n', encoding="utf-8")
        (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        self.procs = root / "localm" / "procs.py"
        if procs_imports:
            self.procs.write_text(procs_imports + self.procs.read_text(encoding="utf-8"),
                                  encoding="utf-8")
        self.good = self.procs.read_bytes()
        self.statement = {"format": update_package.FORMAT, "version": self.version,
                          "files": update_package.files(root), "components": {}}
        (root / update_package.NAME).write_text(json.dumps(self.statement), encoding="utf-8")
        self.site = self._venv()
        if with_package:
            (self.site / f"{PACKAGE}.py").write_text("HERE = True\n", encoding="utf-8")
            info = self.site / f"{PACKAGE}-1.0.dist-info"
            info.mkdir()
            (info / "METADATA").write_text(
                f"Metadata-Version: 2.1\nName: {PACKAGE}\nVersion: 1.0\n\n", encoding="utf-8")
        self.home.mkdir()
        (self.home / "config.json").write_text(json.dumps({
            "update_url": "https://updates.invalid", "bugreport_upload_url": ""}),
            encoding="utf-8")

    def _venv(self) -> Path:
        """A venv record of localm at this version, without the package."""
        site = (self.root / ".venv" / "Lib" / "site-packages" if os.name == "nt" else
                self.root / ".venv" / "lib" / "python3.12" / "site-packages")
        info = site / f"localm-{self.version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: localm\nVersion: {self.version}\n\n", encoding="utf-8")
        return site

    def signed_build(self) -> dict:
        """The signed build of this release, holding the good ``localm/procs.py``."""
        import io
        import zipfile

        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        from tests._update_signing import envelope, keyset
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("localm/procs.py", self.good)
            z.writestr(update_package.NAME, json.dumps(self.statement))
        data = buf.getvalue()
        build = self.root.parent / "build.zip"
        build.write_bytes(data)
        key = Ed25519PrivateKey.generate()
        uv = self.root.parent / "uv-bin" / "uv.exe"
        uv.parent.mkdir()
        uv.write_bytes(b"")
        uv.chmod(0o755)
        return {"keyset": keyset(1, [key], [Ed25519PrivateKey.generate() for _ in range(3)]),
                "signature": envelope(data, [key]), "version": self.version,
                "build": str(build), "seen": str(self.seen), "uv": str(uv),
                "site": str(self.site), "package": PACKAGE}

    def start(self, spec: dict):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("LOCALM_") and k not in (
                   "PYTHONHOME", "VIRTUAL_ENV", "PYTHONDONTWRITEBYTECODE", "UV_HTTP_TIMEOUT")}
        env.update(PYTHONPATH=os.pathsep.join([str(self.root), str(self.site)]),
                   PYTHONNOUSERSITE="1", LOCALM_HOME=str(self.home),
                   PROCS_REPAIR_TEST=json.dumps(spec))
        proc = subprocess.run([sys.executable, "-c", _START], cwd=str(self.root), env=env,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=180, **HIDDEN)
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("STARTED ")), None)
        return types.SimpleNamespace(code=proc.returncode, out=proc.stdout, err=proc.stderr,
                                     said=json.loads(line[8:]) if line else None)

    def recorded(self, pid: int) -> bool:
        return any(p.is_file() and f"{pid}" in p.name
                   for p in (self.root / ".localm-run").glob("*"))

    def seen_lines(self) -> list:
        if not self.seen.exists():
            return []
        return [json.loads(ln) for ln in self.seen.read_text(encoding="utf-8").splitlines()]


def _ran(done, inst) -> bool:
    return (done.code == 0 and done.said is not None
            and Path(done.said["file"]).resolve() == (inst.root / "localm" / "__init__.py").resolve())


@pytest.mark.parametrize("damage, error", [
    (BROKEN, "ModuleNotFoundError: No module named 'localm_no_such_module_for_the_gate'"),
    ("def (\n", "SyntaxError: invalid syntax"),
])
def test_a_damaged_process_record_is_repaired_from_the_signed_release(tmp_path, damage, error):
    inst = Install(tmp_path)
    spec = inst.signed_build()
    inst.procs.write_text(damage + inst.good.decode("utf-8"), encoding="utf-8")
    done = inst.start(spec)
    assert _ran(done, inst), (done.out, done.err)
    assert done.said["record"] and inst.recorded(done.said["pid"]), done.err
    assert inst.procs.read_bytes() == inst.good
    lines = [ln for ln in done.err.splitlines() if ln.startswith("localm: ")]
    assert len(lines) == 1 and lines[0].startswith(
        f"localm: repaired localm/procs.py from the signed release {inst.version} "
        f"(it could not be imported: {error}"), done.err
    assert inst.seen_lines() == [{"download_timeout": float(procs_repair.NET_S)}]
    assert not (inst.home / "updates" / "integrity-repair").exists()


@pytest.mark.parametrize("case, reason", [
    ("no update server", "the updater is not configured"),
    ("lock busy", "another repair is already being applied"),
])
def test_a_repair_that_cannot_run_says_why_and_the_start_goes_on(tmp_path, case, reason):
    inst = Install(tmp_path, with_package=True)
    spec = inst.signed_build()
    if case == "no update server":
        (inst.home / "config.json").write_text(json.dumps(
            {"update_url": "", "bugreport_upload_url": ""}), encoding="utf-8")
    else:
        lock = inst.home / "updates" / "apply.lock"
        lock.mkdir(parents=True)
        (lock / "pid").write_text(str(os.getpid()), encoding="utf-8")
    inst.procs.write_text(BROKEN + inst.good.decode("utf-8"), encoding="utf-8")
    done = inst.start(spec)
    assert _ran(done, inst) and done.said["record"] is False, (done.out, done.err)
    assert "No module named 'localm_no_such_module_for_the_gate'" in done.err
    assert reason in done.err and "This start goes on without it" in done.err, done.err
    assert done.err.endswith("to repair it: 'localm integrity --repair localm/procs.py'.\n"), \
        done.err
    assert inst.procs.read_bytes() != inst.good and inst.seen_lines() == []


def test_a_normal_start_loads_no_repair(tmp_path, monkeypatch):
    inst = Install(tmp_path)
    done = inst.start(inst.signed_build())
    assert _ran(done, inst) and done.said["record"], (done.out, done.err)
    assert done.said["repair_loaded"] is False and done.err == "" and inst.seen_lines() == []


def test_without_a_release_statement_the_start_runs_and_says_why_and_the_fix(
        tmp_path, monkeypatch):
    inst = Install(tmp_path, with_package=True)
    spec = inst.signed_build()
    (inst.root / update_package.NAME).unlink()
    inst.procs.write_text(BROKEN + inst.good.decode("utf-8"), encoding="utf-8")
    done = inst.start(spec)
    assert _ran(done, inst), (done.out, done.err)
    assert done.said["record"] is False and not (inst.root / ".localm-run").exists()
    assert "ModuleNotFoundError: No module named 'localm_no_such_module_for_the_gate'" in done.err
    assert "this install has no list of its release's files" in done.err
    assert "the installed parts are complete" in done.err
    assert "This start goes on without it" in done.err
    assert (f"to repair it: unzip the release of this version over {inst.root}; when the "
            "package localm_no_such_module_for_the_gate is missing: Settings > System > "
            "Updates > Installed parts > Repair") in done.err
    assert inst.seen_lines() == [], "nothing was downloaded or installed"


def test_a_git_clone_is_told_to_check_the_file_out(tmp_path, monkeypatch):
    inst = Install(tmp_path, with_package=True)
    spec = inst.signed_build()
    (inst.root / update_package.NAME).unlink()
    (inst.root / ".git").mkdir()
    inst.procs.write_text(BROKEN + inst.good.decode("utf-8"), encoding="utf-8")
    done = inst.start(spec)
    assert _ran(done, inst) and done.said["record"] is False, (done.out, done.err)
    assert f"to repair it: 'git checkout -- localm/procs.py' in {inst.root};" in done.err


def test_a_missing_package_is_installed_by_the_repair_of_the_installed_parts(
        tmp_path, monkeypatch):
    inst = Install(tmp_path, procs_imports=f"import {PACKAGE}\n")
    spec = inst.signed_build()
    done = inst.start(spec)
    assert _ran(done, inst), (done.out, done.err)
    assert done.said["record"] and inst.recorded(done.said["pid"]), done.err
    assert (inst.site / f"{PACKAGE}.py").is_file()
    seen = inst.seen_lines()
    assert len(seen) == 1 and seen[0]["uv"][0] == spec["uv"], seen
    assert seen[0]["UV_HTTP_TIMEOUT"] == str(procs_repair.NET_S)
    lines = [ln for ln in done.err.splitlines() if ln.startswith("localm: ")]
    assert lines == [f"localm: installed the missing package {PACKAGE} (reinstalled localm and "
                     f"its dependencies) (it could not be imported: ModuleNotFoundError: No "
                     f"module named '{PACKAGE}')"], done.err
    assert inst.procs.read_bytes() == inst.good, "the release's own file is not written"


def test_a_missing_package_the_repair_cannot_install_is_named_with_the_fix(
        tmp_path, monkeypatch):
    inst = Install(tmp_path, procs_imports=f"import {PACKAGE}\n")
    spec = inst.signed_build()
    (inst.root / "uv.lock").unlink()
    done = inst.start(spec)
    assert _ran(done, inst) and done.said["record"] is False, (done.out, done.err)
    assert f"No module named '{PACKAGE}'" in done.err
    assert "the build has no uv.lock" in done.err
    assert "Settings > System > Updates > Installed parts > Repair" in done.err
    assert not (inst.site / f"{PACKAGE}.py").exists()


def test_the_repair_imports_without_the_process_record(tmp_path):
    """The modules the repair loads import while localm.procs cannot be imported."""
    code = ("import sys; sys.modules['localm.procs'] = None\n"
            "import localm.procs_repair, localm.integrity, localm.updater_parts, "
            "localm.bugreport, localm.updater_runtime, localm.updater_components\n"
            "print('IMPORTED')")
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOCALM_")}
    env["LOCALM_HOME"] = str(tmp_path / "home")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env,
                          capture_output=True, text=True, timeout=120, **HIDDEN)
    assert proc.stdout.strip().endswith("IMPORTED"), proc.stderr


@pytest.mark.parametrize("argv, said", [
    (["--version"], f"localm, version {localm.__version__}"),
    (["integrity", "--help"], "--repair"),
    (["update", "--help"], "--repair"),
])
def test_the_commands_that_repair_it_run_without_the_process_record(tmp_path, argv, said):
    """What the warning tells the user to run works while localm.procs cannot be
    imported: the command line loads no module that imports it at module level."""
    code = ("import sys; sys.modules['localm.procs'] = None\n"
            "from localm.cli import main\n"
            f"sys.argv = ['localm', *{argv!r}]\n"
            "main()")
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOCALM_")}
    env["LOCALM_HOME"] = str(tmp_path / "home")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env,
                          capture_output=True, text=True, timeout=120, **HIDDEN)
    assert proc.returncode == 0 and said in proc.stdout, (proc.stdout, proc.stderr)


@pytest.mark.parametrize("error, files", [
    (ModuleNotFoundError("No module named 'localm.gone'", name="localm.gone"),
     ["localm/gone.py", "localm/procs.py"]),
    (ModuleNotFoundError("No module named 'yaml'", name="yaml"), ["localm/procs.py"]),
])
def test_the_files_an_import_error_names(error, files):
    assert procs_repair.named_files(error, REPO) == files
    assert procs_repair.missing_package(error) == ("" if error.name.startswith("localm") else
                                                   "yaml")
