# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-plugin media workflow management (localm.media_workflows).

Users upload ComfyUI API-format workflows and select which one a media plugin
uses. These tests pin: upload validation (reject non-JSON / non-workflow),
the select/clear config marker, active-file resolution + the generator fallback
chain, delete refusing the active file, and path-traversal safety.
"""

from __future__ import annotations

import json

import pytest

from localm import media_workflows as mw


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    import localm.config as cfg
    h = tmp_path / ".localm"
    h.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    return h


_WF = json.dumps({"3": {"class_type": "KSampler", "inputs": {}},
                  "4": {"class_type": "SaveImage", "inputs": {}}}).encode()


def test_is_workflow_json():
    assert mw.is_workflow_json({"1": {"class_type": "X"}})
    assert not mw.is_workflow_json({})
    assert not mw.is_workflow_json([1, 2])
    assert not mw.is_workflow_json({"a": {"no": "class_type"}})


def test_save_validates_and_normalizes_name():
    name = mw.save_workflow("image", "my-flux", _WF)   # no .json suffix
    assert name == "my-flux.json"
    assert (mw.workflows_dir("image") / "my-flux.json").is_file()


def test_save_rejects_non_json_and_non_workflow():
    with pytest.raises(ValueError):
        mw.save_workflow("image", "bad", b"not json{{")
    with pytest.raises(ValueError):
        mw.save_workflow("image", "plain", json.dumps({"hello": "world"}).encode())


def test_select_sets_and_clears_marker_and_resolves_active():
    mw.save_workflow("image", "a.json", _WF)
    assert mw.active_workflow_path("image") is None      # nothing selected yet
    mw.select_workflow("image", "a.json")
    assert mw.selected_name("image") == "a.json"
    p = mw.active_workflow_path("image")
    assert p is not None and p.name == "a.json"
    # clearing falls back (None) so the generator uses its legacy/example template
    mw.select_workflow("image", None)
    assert mw.selected_name("image") is None
    assert mw.active_workflow_path("image") is None


def test_select_unknown_rejected():
    with pytest.raises(ValueError):
        mw.select_workflow("image", "ghost.json")


def test_list_flags_active():
    mw.save_workflow("video", "one.json", _WF)
    mw.save_workflow("video", "two.json", _WF)
    mw.select_workflow("video", "two.json")
    by_name = {w["name"]: w for w in mw.list_workflows("video")}
    assert by_name["two.json"]["is_active"] is True
    assert by_name["one.json"]["is_active"] is False


def test_delete_refuses_active_and_removes_inactive():
    mw.save_workflow("music", "keep.json", _WF)
    mw.save_workflow("music", "drop.json", _WF)
    mw.select_workflow("music", "keep.json")
    with pytest.raises(ValueError):
        mw.delete_workflow("music", "keep.json")          # active -> refused
    mw.delete_workflow("music", "drop.json")              # inactive -> ok
    assert not (mw.workflows_dir("music") / "drop.json").is_file()


def test_delete_unknown_rejected():
    with pytest.raises(ValueError):
        mw.delete_workflow("image", "nope.json")


@pytest.mark.parametrize("bad", ["../escape.json", "a/b.json", "..", ""])
def test_path_traversal_rejected(bad):
    from fastapi import HTTPException
    with pytest.raises((HTTPException, ValueError)):
        mw.save_workflow("image", bad, _WF)


def test_router_endpoints_list_upload_select_delete():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(mw.make_workflow_router("image"))
    c = TestClient(app)

    assert c.get("/api/image/workflows").json() == {"workflows": [], "selected": None}

    wf = {"3": {"class_type": "KSampler", "inputs": {}}}
    r = c.post("/api/image/workflows",
               json={"name": "x", "workflow": wf, "activate": True})
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "x.json" and r.json()["selected"] == "x.json"

    # bad uploads -> 400 with a reason
    assert c.post("/api/image/workflows",
                  json={"name": "y", "workflow": {"no": "nodes"}}).status_code == 400
    assert c.post("/api/image/workflows", json={"name": "y"}).status_code == 400

    # cannot delete the active one; clear then delete
    assert c.delete("/api/image/workflows/x.json").status_code == 400
    assert c.post("/api/image/workflows/select", json={"name": None}).json()["selected"] is None
    assert c.delete("/api/image/workflows/x.json").status_code == 200
    assert c.get("/api/image/workflows").json()["workflows"] == []


def test_generator_uses_selected_workflow(monkeypatch):
    # The image generator's _workflow_path() resolves the selected file first.
    from pathlib import Path

    from localm.image_gen import comfy
    # Isolate from a personal flux_workflow.json that may exist in a dev checkout
    # (it is gitignored and takes precedence over the example), so the cleared-
    # selection case deterministically falls back to the committed example.
    monkeypatch.setattr(comfy, "_WORKFLOW_PATH",
                        Path(__file__).parent / "_no_personal_flux_workflow.json")
    custom = mw.save_workflow("image", "custom.json", _WF)
    mw.select_workflow("image", custom)
    assert comfy._workflow_path() == mw.active_workflow_path("image")
    # cleared -> falls back to the committed example (no personal override here)
    mw.select_workflow("image", None)
    assert comfy._workflow_path() == comfy._WORKFLOW_EXAMPLE_PATH


# ---------------------------------------------------------------------------
#  Legacy in-package override migration (rescued from the self-update wipe)
# ---------------------------------------------------------------------------


def _legacy(tmp_path, name="flux_workflow.json"):
    """A fake legacy override sitting 'inside the package', at a temp path so the
    test never touches the real repo checkout."""
    p = tmp_path / "pkg" / "image_gen" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(_WF)
    return p


def test_migration_moves_legacy_override_into_home_and_selects_it(tmp_path):
    legacy = _legacy(tmp_path)
    ok, note = mw._migrate_one("image", legacy)
    assert ok and "migrated" in note

    dest = mw.workflows_dir("image") / "flux_workflow.json"
    assert dest.is_file()
    assert dest.read_bytes() == _WF                          # content preserved
    assert mw.selected_name("image") == "flux_workflow.json"  # kept active
    assert not legacy.exists()                               # moved out of the package
    # Resolves as the active workflow through the home-preferring resolution.
    assert mw.active_workflow_path("image").resolve() == dest.resolve()


def test_migration_is_noop_when_no_legacy_file(tmp_path):
    assert mw._migrate_one("image", tmp_path / "ghost" / "flux_workflow.json") is None


def test_migration_keeps_an_existing_selection(tmp_path):
    # The user already picked a workflow -> migration rescues the legacy file into
    # the list but must NOT override their choice.
    mw.save_workflow("image", "chosen.json", _WF)
    mw.select_workflow("image", "chosen.json")

    legacy = _legacy(tmp_path)
    ok, _ = mw._migrate_one("image", legacy)
    assert ok
    assert mw.selected_name("image") == "chosen.json"                 # unchanged
    assert (mw.workflows_dir("image") / "flux_workflow.json").is_file()  # rescued
    assert not legacy.exists()


def test_migration_dedups_on_name_collision(tmp_path):
    # A DIFFERENT workflow already saved under the legacy's name -> keep BOTH,
    # never clobber the user's existing file.
    other = json.dumps({"9": {"class_type": "Other", "inputs": {}}}).encode()
    mw.save_workflow("image", "flux_workflow.json", other)

    legacy = _legacy(tmp_path)
    ok, _ = mw._migrate_one("image", legacy)
    assert ok
    d = mw.workflows_dir("image")
    assert (d / "flux_workflow.json").read_bytes() != _WF     # existing untouched
    assert (d / "flux_workflow-1.json").read_bytes() == _WF   # legacy under a fresh name
    assert not legacy.exists()


def test_migration_idempotent_second_run_removes_duplicate(tmp_path):
    # After a successful migration the source is gone, so a second run is a no-op;
    # and a fresh identical legacy copy (same bytes already in home) is retired,
    # not duplicated on every startup.
    legacy = _legacy(tmp_path)
    assert mw._migrate_one("image", legacy)[0]
    assert mw._migrate_one("image", legacy) is None          # source moved away

    again = _legacy(tmp_path)                                 # identical bytes reappear
    ok, note = mw._migrate_one("image", again)
    assert ok and not again.exists()
    # Still exactly one home copy - no flux_workflow-1.json spawned.
    names = [w["name"] for w in mw.list_workflows("image")]
    assert names == ["flux_workflow.json"]


def test_migration_reactivates_legacy_when_selection_is_dangling(tmp_path):
    # The selection marker points at a workflow whose file has since vanished, so
    # the legacy override is the EFFECTIVE active workflow (the generators fall
    # back to it). Migration must keep it active - gating on the marker alone
    # would let generation silently drop to the committed example.
    mw.save_workflow("image", "myflow.json", _WF)
    mw.select_workflow("image", "myflow.json")
    (mw.workflows_dir("image") / "myflow.json").unlink()     # selected file vanishes
    assert mw.selected_name("image") == "myflow.json"        # marker still set
    assert mw.active_workflow_path("image") is None          # but nothing resolves

    legacy = _legacy(tmp_path)
    ok, _ = mw._migrate_one("image", legacy)
    assert ok
    dest = mw.workflows_dir("image") / "flux_workflow.json"
    assert mw.selected_name("image") == "flux_workflow.json"   # reactivated
    assert mw.active_workflow_path("image").resolve() == dest.resolve()
    assert not legacy.exists()


def test_migration_no_copy_proliferation_when_removal_keeps_failing(tmp_path, monkeypatch):
    # A different-bytes file owns the base name AND the in-package original cannot
    # be removed (read-only/locked install dir). Each startup must NOT spawn a
    # fresh flux_workflow-N.json: once the bytes exist under home, later runs find
    # them by content and only retry the removal.
    import pathlib
    other = json.dumps({"9": {"class_type": "Other", "inputs": {}}}).encode()
    mw.save_workflow("image", "flux_workflow.json", other)   # occupies the base name

    legacy = _legacy(tmp_path)
    real_unlink = pathlib.Path.unlink

    def _unlink_fail_on_legacy(self, *a, **k):
        if self == legacy:
            raise OSError("locked")               # the original can never be removed
        return real_unlink(self, *a, **k)
    monkeypatch.setattr(pathlib.Path, "unlink", _unlink_fail_on_legacy)

    for _ in range(3):                            # three startups
        ok, _note = mw._migrate_one("image", legacy)
        assert ok is False                        # removal fails, surfaced as a note
        assert legacy.exists()                    # original still stuck in place
    d = mw.workflows_dir("image")
    names = sorted(w["name"] for w in mw.list_workflows("image"))
    assert names == ["flux_workflow-1.json", "flux_workflow.json"]   # no -2/-3 spawned
    assert (d / "flux_workflow-1.json").read_bytes() == _WF          # the rescued bytes


def test_migrate_legacy_override_is_skipped_under_pytest():
    # The public startup entry is a no-op during the suite so it never moves a
    # developer's real in-package workflow; the logic is covered via _migrate_one.
    assert mw.migrate_legacy_override("image") is None


def test_subprocess_migration_guard_flag_is_set_by_conftest():
    # conftest sets this so a localm SUBPROCESS a test spawns (no pytest in its
    # sys.modules) still skips the destructive in-package migration. Losing it
    # would let the suite move a developer's real flux_workflow.json/etc. out of
    # their working tree.
    import os
    assert os.environ.get("LOCALM_SKIP_LEGACY_WORKFLOW_MIGRATION") == "1"


def test_legacy_override_survives_self_update_via_migration(tmp_path, monkeypatch):
    # End to end: a legacy override present BEFORE an update is preserved AFTER it
    # because migration moved it under home/ (the updater's NEVER_TOUCH tree).
    import localm.config as cfg
    from localm import _apply_update as up

    install = tmp_path / "install"
    home = install / "home"
    pkg_img = install / "localm" / "image_gen"
    home.mkdir(parents=True)
    pkg_img.mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")

    # 1. A personal override the user dropped inside the package (pre-migration).
    legacy = pkg_img / "flux_workflow.json"
    legacy.write_bytes(_WF)

    # 2. Startup migration moves it under home/workflows and keeps it active.
    assert mw._migrate_one("image", legacy)[0]
    dest = mw.workflows_dir("image") / "flux_workflow.json"
    assert dest.is_file() and not legacy.exists()

    # 3. A real self-update swaps the whole localm/ package dir for a staged build
    #    that (like a real build) does not carry the personal override.
    staged = tmp_path / "staged"
    (staged / "localm" / "image_gen").mkdir(parents=True)
    (staged / "localm" / "image_gen" / "flux_workflow.example.json").write_text(
        "{}", encoding="utf-8")
    swapped = up.swap_with_backup(staged, install, tmp_path / "backup")
    assert "localm" in swapped and "home" not in swapped     # home is NEVER_TOUCH

    # 4. The override survived the update and still resolves as active.
    assert dest.is_file()
    assert mw.active_workflow_path("image").resolve() == dest.resolve()


def test_in_package_override_is_destroyed_by_update_without_migration(tmp_path):
    # Documents the bug the migration prevents: an override left INSIDE localm/ is
    # wiped by the whole-tree package swap. This is the oracle's negative case -
    # if the swap did NOT delete it, the survival test above would prove nothing.
    from localm import _apply_update as up

    install = tmp_path / "install"
    pkg_img = install / "localm" / "image_gen"
    pkg_img.mkdir(parents=True)
    legacy = pkg_img / "flux_workflow.json"
    legacy.write_bytes(_WF)

    staged = tmp_path / "staged"
    (staged / "localm" / "image_gen").mkdir(parents=True)
    up.swap_with_backup(staged, install, tmp_path / "backup")
    assert not legacy.exists()
