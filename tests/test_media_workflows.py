# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-plugin media workflow management (localm.media_workflows).

Users upload ComfyUI API-format workflows and select which one a media plugin
uses. These tests pin: upload validation (reject non-JSON / non-workflow),
the select/clear config marker, active-file resolution + the generator fallback
chain, delete refusing the active file, and path-traversal safety.
"""

from __future__ import annotations

import json
import threading
import time

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


def test_upload_route_504s_when_the_write_hangs_past_budget(monkeypatch):
    """A wedged save_workflow() call must not leave the HTTP request hanging:
    the route returns a clear 504 within the configured budget, proven end to
    end through the real route rather than only the underlying wrapper."""
    import time as _time

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(mw, "_WORKFLOW_RMW_TIMEOUT_S", 0.2)

    def _hangs(media, name, content):
        _time.sleep(2.0)
        return "should never get here in time"

    monkeypatch.setattr(mw, "save_workflow", _hangs)

    app = FastAPI()
    app.include_router(mw.make_workflow_router("image"))
    c = TestClient(app)

    start = _time.monotonic()
    r = c.post("/api/image/workflows",
              json={"name": "x", "workflow": {"3": {"class_type": "K"}}})
    elapsed = _time.monotonic() - start

    assert r.status_code == 504, r.text
    assert "timed out" in r.json()["detail"].lower()
    assert elapsed < 1.5, (
        f"the route waited {elapsed:.2f}s despite a 0.2s budget - the "
        "timeout did not actually bound the request")


def test_rmw_timeout_has_headroom_over_a_single_holders_own_work_ceiling():
    """Structural guard, asserting the ARITHMETIC rather than the literal:
    _WORKFLOW_RMW_TIMEOUT_S is the budget every route actually uses, but all
    four routes share ONE _lock_for(media) lock acquired INSIDE that same
    bounded call, so a request's own clock also covers however long it waits
    behind another holder. If this regresses to matching
    _WORKFLOW_OWN_WORK_TIMEOUT_S exactly, a merely-slow (not hung) writer can
    collaterally 504 a concurrently-queued, otherwise-instant reader sharing
    the same lock (see test_a_merely_slow_write_does_not_collaterally_
    504_a_queued_read below)."""
    assert mw._WORKFLOW_RMW_TIMEOUT_S >= 2 * mw._WORKFLOW_OWN_WORK_TIMEOUT_S


@pytest.mark.anyio
async def test_a_merely_slow_write_does_not_collaterally_504_a_queued_read(monkeypatch):
    """A write that is merely slow (not hung, and over a naive single-holder
    budget) must not push a concurrently-queued, trivially-fast read past ITS
    OWN budget purely from waiting on the shared per-media lock. Driven end to
    end over real concurrent HTTP requests, not just at the function level."""
    import asyncio

    import httpx
    from fastapi import FastAPI

    # Scaled down for a fast test while preserving the >=2x relationship, with
    # generous margins.
    monkeypatch.setattr(mw, "_WORKFLOW_RMW_TIMEOUT_S", 3.0)

    write_entered = threading.Event()

    def _slow_but_legitimate_write(media, name, content):
        write_entered.set()
        # 1.0s: a third of the 3.0s budget, but over a naive single-holder ceiling
        # of 1.5s - the merely-slow, not-hung case.
        time.sleep(1.0)
        return "x.json"

    monkeypatch.setattr(mw, "save_workflow", _slow_but_legitimate_write)

    app = FastAPI()
    app.include_router(mw.make_workflow_router("image"))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        write_task = asyncio.ensure_future(
            client.post("/api/image/workflows",
                       json={"name": "x", "workflow": {"3": {"class_type": "K"}}}))
        # Deterministic hand-off: wait until the write is genuinely inside its
        # critical section before firing the read.
        for _ in range(500):
            if write_entered.is_set():
                break
            await asyncio.sleep(0.005)
        assert write_entered.is_set(), "the write never reached its critical section"

        read_resp = await client.get("/api/image/workflows")
        write_resp = await write_task

    assert write_resp.status_code == 200, write_resp.text
    assert read_resp.status_code == 200, (
        f"a merely-slow (not hung) write starved a queued fast read: {read_resp.text}")


def test_generator_uses_selected_workflow(monkeypatch):
    # The image generator's workflow_path() resolves the selected file first.
    from pathlib import Path

    from localm.image_gen import comfy
    # Isolate from a personal flux_workflow.json that may exist in a dev checkout
    # (gitignored, and it takes precedence over the example), so the cleared-
    # selection case falls back to the committed example.
    monkeypatch.setattr(comfy, "_WORKFLOW_PATH",
                        Path(__file__).parent / "_no_personal_flux_workflow.json")
    custom = mw.save_workflow("image", "custom.json", _WF)
    mw.select_workflow("image", custom)
    assert comfy.workflow_path() == mw.active_workflow_path("image")
    # cleared -> falls back to the committed example (no personal override here)
    mw.select_workflow("image", None)
    assert comfy.workflow_path() == comfy._WORKFLOW_EXAMPLE_PATH


# ---------------------------------------------------------------------------
#  Legacy in-package override migration
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
    # the legacy override is the effective active workflow. Migration keeps it
    # active.
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
    # sys.modules) still skips the destructive in-package migration.
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
    # An override left inside localm/ is wiped by the whole-tree package swap.
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


# --------------------------------------------------------------------------- #
#  _lock_for: per-media mutual exclusion. These tests drive the check-then-act #
#  sequences from real OS threads, so what is proven is the lock's actual      #
#  mutual-exclusion behavior.                                                  #
# --------------------------------------------------------------------------- #

def test_lock_for_serializes_concurrent_operations_on_the_same_media():
    """Direct proof of the property every route now relies on: two threads
    racing for the SAME media's lock must never be inside the critical
    section at the same time. A sleep widens the window so a race would
    reliably show up rather than being missed by luck."""
    concurrent_now = []
    max_concurrent = [0]
    guard = threading.Lock()

    def _critical_section():
        with mw._lock_for("test-media"):
            with guard:
                concurrent_now.append(1)
                max_concurrent[0] = max(max_concurrent[0], len(concurrent_now))
            time.sleep(0.05)
            with guard:
                concurrent_now.pop()

    threads = [threading.Thread(target=_critical_section) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert max_concurrent[0] == 1, (
        f"max concurrent threads inside the lock was {max_concurrent[0]}, "
        "expected 1 - the lock is not actually serializing")


def test_lock_for_does_not_serialize_different_media():
    """image/music/video are independent directory trees and independent
    config blocks - only genuinely shared state needs mutual exclusion.
    Serializing across DIFFERENT media would be pure unnecessary contention,
    not a correctness requirement."""
    both_held = threading.Event()

    def _hold(media):
        with mw._lock_for(media):
            both_held.wait(timeout=2)

    t1 = threading.Thread(target=_hold, args=("image",))
    t2 = threading.Thread(target=_hold, args=("music",))
    t1.start()
    t2.start()
    # If the two locks were the same object, the second thread would block
    # here and this next line would never run before the join timeout.
    time.sleep(0.1)
    both_held.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert not t1.is_alive() and not t2.is_alive(), (
        "a lock for one media blocked a different media's lock - they must "
        "be independent")


def test_concurrent_uploads_to_the_same_name_no_longer_corrupt_the_file(home):
    """Two real OS threads racing to save the SAME filename with no lock
    produce torn/invalid JSON on disk. Under the per-media lock the writes must
    serialize, so the file on disk must ALWAYS be valid JSON matching one of the
    two payloads, in full, never a mix of both."""
    small_data = {"3": {"class_type": "KSampler", "inputs": {"a": "x" * 500}}}
    large_data = {"3": {"class_type": "KSampler", "inputs": {"a": "y" * 50_000}}}
    small = json.dumps(small_data).encode()
    large = json.dumps(large_data).encode()

    def _upload(content):
        with mw._lock_for("image"):
            mw.save_workflow("image", "race.json", content)

    corruptions = 0
    for _ in range(30):
        threads = [threading.Thread(target=_upload, args=(small,)),
                   threading.Thread(target=_upload, args=(large,))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        raw = (mw.workflows_dir("image") / "race.json").read_bytes()
        try:
            # save_workflow re-serializes (indent=2, platform newlines), so parsed
            # content is what gets compared.
            data = json.loads(raw)
        except json.JSONDecodeError:
            corruptions += 1
            continue
        # Must match ONE writer's PARSED content completely, not a torn mix
        # (e.g. one writer's keys with another's values, or a truncated tail).
        assert data == small_data or data == large_data, (
            f"parsed content did not match either writer's payload in full "
            f"(a torn, mixed write): {data!r}")

    assert corruptions == 0, (
        f"{corruptions}/30 trials produced invalid JSON on disk - the lock "
        "is not preventing torn writes")


def test_concurrent_list_survives_a_racing_delete(home):
    """A DELETE landing between list_workflows' is_file() check and its later
    stat() calls raises an unhandled FileNotFoundError -> 500. Under the lock, a
    listing and a delete for the same media cannot interleave at all:
    list_workflows must never raise, no matter how many concurrent deletes are
    racing it."""
    d = mw.workflows_dir("image")
    d.mkdir(parents=True, exist_ok=True)
    names = [f"wf{i}.json" for i in range(20)]
    for n in names:
        (d / n).write_bytes(_WF)

    errors = []

    def _list():
        try:
            with mw._lock_for("image"):
                mw.list_workflows("image")
        except Exception as e:  # noqa: BLE001 - the property under test is "never raises"
            errors.append(e)

    def _delete(n):
        try:
            with mw._lock_for("image"):
                (d / n).unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=_list) for _ in range(10)]
    threads += [threading.Thread(target=_delete, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors, f"list_workflows raised under a racing delete: {errors!r}"


def test_concurrent_select_and_delete_never_orphans_the_selection(home):
    """A select and a delete of the SAME file, racing, can leave
    config["plugins"]["image"]["workflow"] pointing at a file that no longer
    exists on disk, violating the documented invariant ("a generation is never
    left pointing at a missing file"). Under the lock the two operations fully
    serialize, so after both finish whatever is selected (if anything) must
    still exist on disk."""
    d = mw.workflows_dir("image")
    d.mkdir(parents=True, exist_ok=True)
    (d / "x.json").write_bytes(_WF)

    def _select():
        with mw._lock_for("image"):
            try:
                mw.select_workflow("image", "x.json")
            except ValueError:
                pass   # lost the race to delete - fine, that is the point

    def _delete():
        with mw._lock_for("image"):
            try:
                mw.delete_workflow("image", "x.json")
            except ValueError:
                pass   # lost the race to select (file now active) - fine

    for _ in range(20):
        (d / "x.json").write_bytes(_WF)   # restore for the next trial if deleted
        threads = [threading.Thread(target=_select), threading.Thread(target=_delete)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        sel = mw.selected_name("image")
        if sel is not None:
            assert (d / sel).is_file(), (
                f"config selected {sel!r} but the file does not exist on disk - "
                "orphaned selection, the invariant was violated")
