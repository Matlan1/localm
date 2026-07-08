# SPDX-License-Identifier: AGPL-3.0-or-later
"""The standalone rollback (scripts/rollback_update.py): restore the previous build
after a bad update WITHOUT importing the (possibly broken) localm package. Every test
runs against a THROWAWAY install + home, never the real repo, and fails loud paths
must return non-zero and keep the backup (do-not-hide-problems)."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_rb():
    spec = importlib.util.spec_from_file_location(
        "rollback_update", REPO / "scripts" / "rollback_update.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stage(tmp_path, monkeypatch, *, manifest, with_helper=True):
    """A fake install (existing.txt replaced by the update, brand_new/ added) and a
    fake home whose updates/backup holds the pre-apply existing.txt."""
    inst = tmp_path / "install"
    inst.mkdir()
    if with_helper:   # the stdlib-only restore helper, loaded by file path
        (inst / "localm").mkdir()
        shutil.copy2(REPO / "localm" / "_apply_update.py",
                     inst / "localm" / "_apply_update.py")
    home = tmp_path / "home"
    (home / "updates" / "backup").mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    (inst / "existing.txt").write_text("NEW-from-update", encoding="utf-8")
    (home / "updates" / "backup" / "existing.txt").write_text("OLD-preapply", encoding="utf-8")
    (inst / "brand_new").mkdir()
    (inst / "brand_new" / "f.txt").write_text("added by update", encoding="utf-8")
    if manifest is not None:
        (home / "updates" / "applied_names.json").write_text(
            json.dumps(manifest), encoding="utf-8")
    return inst, home


def test_rollback_restores_and_removes_added(tmp_path, monkeypatch):
    rb = _load_rb()
    inst, _ = _stage(tmp_path, monkeypatch, manifest=["existing.txt", "brand_new"])
    assert rb.main(["--yes"], install_root=inst) == 0
    assert (inst / "existing.txt").read_text(encoding="utf-8") == "OLD-preapply"  # restored
    assert not (inst / "brand_new").exists()   # added entry removed -> pre-apply state


def test_rollback_fallback_without_manifest(tmp_path, monkeypatch):
    rb = _load_rb()
    inst, _ = _stage(tmp_path, monkeypatch, manifest=None)
    assert rb.main(["--yes"], install_root=inst) == 0
    assert (inst / "existing.txt").read_text(encoding="utf-8") == "OLD-preapply"  # restored
    assert (inst / "brand_new").exists()   # no manifest -> unrecorded new entry survives


def test_rollback_no_backup_exits_1(tmp_path, monkeypatch):
    rb = _load_rb()
    home = tmp_path / "home"
    (home / "updates").mkdir(parents=True)      # updates dir but no backup
    monkeypatch.setenv("LOCALM_HOME", str(home))
    inst = tmp_path / "install"
    inst.mkdir()
    assert rb.main(["--yes"], install_root=inst) == 1


def test_rollback_missing_helper_exits_2_and_keeps_backup(tmp_path, monkeypatch):
    """A build so broken the restore helper cannot even load must fail loud (exit 2),
    keep the backup, and NOT touch the install - never a false success."""
    rb = _load_rb()
    inst, home = _stage(tmp_path, monkeypatch, manifest=["existing.txt"], with_helper=False)
    assert rb.main(["--yes"], install_root=inst) == 2
    assert (home / "updates" / "backup" / "existing.txt").exists()          # backup kept
    assert (inst / "existing.txt").read_text(encoding="utf-8") == "NEW-from-update"  # untouched


def test_rollback_restore_failure_exits_2_and_keeps_backup(tmp_path, monkeypatch):
    """If a restore step fails mid-way, fail loud (exit 2) and keep the backup - never
    report a clean rollback over a still-broken install."""
    rb = _load_rb()
    inst, home = _stage(tmp_path, monkeypatch, manifest=["existing.txt"])

    class _BoomAu:
        @staticmethod
        def rollback(backup, installed, names):
            raise RuntimeError("disk full mid-restore")

    monkeypatch.setattr(rb, "_load_apply_update", lambda _ir: _BoomAu)
    assert rb.main(["--yes"], install_root=inst) == 2
    assert (home / "updates" / "backup" / "existing.txt").exists()   # backup kept


def test_detect_home_honors_localm_home(tmp_path, monkeypatch):
    rb = _load_rb()
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "custom"))
    assert rb._detect_home(tmp_path / "install") == tmp_path / "custom"


def test_shims_invoke_the_rollback_script():
    """The root shims must actually run scripts/rollback_update.py (else the recovery
    entry points are dead)."""
    bat = (REPO / "rollback.bat").read_text(encoding="utf-8")
    sh = (REPO / "rollback.sh").read_text(encoding="utf-8")
    assert "scripts\\rollback_update.py" in bat
    assert "scripts/rollback_update.py" in sh
