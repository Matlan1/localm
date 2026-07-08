# SPDX-License-Identifier: AGPL-3.0-or-later
"""The release smoke gate (scripts/make_release.py smoke_test): a built release must
IMPORT AND RUN on its own, so a runtime-needed file omitted from the build (mis-
classified dev-only, or gitignored) is caught at build time, not by a user later.

The manifest gate proves every file is CLASSIFIED; this proves the included set is
actually COMPLETE. These are integration tests (they build a real zip and spawn a
subprocess), so they run when invoked directly / locally; the gate itself runs in
scripts/make_release.py on every release cut.
"""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cm = _load("check_manifest")
build_release = _load("build_release")
make_release = _load("make_release")


def _rebuild_without(src_zip: Path, dst_zip: Path, drop: str) -> None:
    """Copy *src_zip* to *dst_zip* omitting the member *drop* - simulates a release
    that lost one file to a mis-classification."""
    with zipfile.ZipFile(src_zip) as zin, zipfile.ZipFile(dst_zip, "w") as zout:
        for item in zin.infolist():
            if item.filename != drop:
                zout.writestr(item, zin.read(item.filename))


@pytest.mark.integration
class TestReleaseSmokeGate:
    def test_real_build_imports_and_runs(self, tmp_path):
        """The actual release built from the tree passes the smoke gate."""
        out = tmp_path / "build.zip"
        build_release.build(out)
        make_release.smoke_test(out)   # must not raise

    def test_catches_missing_runtime_module(self, tmp_path):
        """Drop a runtime .py the server needs -> the import check fails the gate."""
        out = tmp_path / "build.zip"
        build_release.build(out)
        assert "localm/inference/http_server.py" in set(zipfile.ZipFile(out).namelist())
        stripped = tmp_path / "stripped.zip"
        _rebuild_without(out, stripped, "localm/inference/http_server.py")
        with pytest.raises(SystemExit, match="release smoke FAILED"):
            make_release.smoke_test(stripped)

    def test_catches_missing_critical_asset(self, tmp_path):
        """Drop a critical non-.py asset -> the asset spot-check fails the gate."""
        out = tmp_path / "build.zip"
        build_release.build(out)
        stripped = tmp_path / "stripped.zip"
        _rebuild_without(out, stripped, "localm/plugins/gui/static/index.html")
        with pytest.raises(SystemExit, match="missing critical file"):
            make_release.smoke_test(stripped)
