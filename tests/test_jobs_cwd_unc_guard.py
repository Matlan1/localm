# SPDX-License-Identifier: AGPL-3.0-or-later
"""A coder job's ``cwd`` reaches Path.expanduser()/.is_dir()/.resolve() in
runner._run_coder with only a non-empty check in front of it (store.py's "coder
jobs require a cwd"). is_dir()/.resolve() are not read-only: on Windows a UNC
target dials outbound SMB and auto-authenticates with the host's net-NTLMv2
credential, before any status or error is chosen.

Reachability is what makes this worth a dedicated test file rather than a note:
the plain ``jobs`` scope reaches it via POST /api/jobs/{id}/run (that route only
re-checks the caller when allow_shell=true), and the scheduler auto-tick runs it
UNATTENDED roughly every 30s with no caller at all, so a hostile cwd stored once
keeps firing on a timer without anyone touching the API again.

The gate has two halves, mirroring the model-name gate already in this same
plugin:

  * write-time (plug.py's _check_cwd for POST/PUT, and the CLI's own check) -
    early rejection, so a NEW or edited job's bad cwd never persists.
  * run-time (runner.py's _run_coder) - the AUTHORITATIVE check, because it
    covers a row persisted before the write-time check existed, or written
    straight to the store, which is exactly the scenario the scheduler tick
    tests below drive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localm.pathsafe import is_unc_or_device_path

# A UNC path at a non-routable RFC 5737 documentation address, so it never
# routes anywhere and cannot reach a real host.
UNC = r"\\192.0.2.1\share"
UNC_FWD = "//192.0.2.1/share"
DEVICE = r"\\.\PhysicalDrive0"

# Every Path method that reaches the filesystem; on Windows each dials SMB for a
# UNC target (is_dir() alone reproduces the full stall, not just resolve()).
_FS_METHODS = ("resolve", "is_dir", "is_file", "exists", "stat", "iterdir")


@pytest.fixture
def fs_spy(monkeypatch):
    """Record every path string that reaches a filesystem call, and hard-fail
    the call when it is UNC or device syntax.

    A local copy rather than an import across test modules: pytest fixtures do
    not share cleanly across unrelated files without a conftest entry, and this
    file's UNC contract is judged by the same predicate
    (pathsafe.is_unc_or_device_path) the product itself uses, so the two cannot
    drift apart.

    Recording AND raising: a raise alone could be swallowed by a caller's broad
    except-and-record-error handling (run_job never lets an exception escape),
    while the recorded list is checked independently below and cannot be
    swallowed that way."""
    seen: list = []
    reals = {m: getattr(Path, m) for m in _FS_METHODS}

    def make(method_name):
        real = reals[method_name]

        def spy(self, *a, **kw):
            s = str(self)
            seen.append(s)
            if is_unc_or_device_path(s):
                raise AssertionError(
                    f"Path.{method_name}() reached the filesystem with a UNC "
                    f"or device string: {s!r} - this is the SMB dial (and the "
                    f"net-NTLMv2 leak), which happens before any status or "
                    f"error is chosen")
            return real(self, *a, **kw)

        return spy

    for m in _FS_METHODS:
        monkeypatch.setattr(Path, m, make(m))
    return seen


def _unc_calls(seen) -> list:
    return [s for s in seen if is_unc_or_device_path(s)]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _make_job(**kw):
    from localm.plugins.builtin.jobs.store import Job
    base = dict(name="j", task_kind="coder", prompt="do x", cwd=".",
                schedule_kind="interval", schedule=60)
    base.update(kw)
    return Job(**base)


def _fake_agent_capture(monkeypatch):
    """Patch the coder backend and Agent so _run_coder runs for real up through
    the cwd guard, without needing a real coder backend or LLM."""
    from localm.plugins.builtin.jobs import runner
    captured: dict = {}

    class _FakeAgent:
        def __init__(self, backend, cwd, **kw):
            captured.update(kw)
            captured["cwd"] = cwd

        def run_task(self, prompt):
            return "ran"

        def close(self):
            return None

    monkeypatch.setattr(runner, "_coder_backend", lambda job: object())
    import localm.plugins.coder.agent as agent_mod
    monkeypatch.setattr(agent_mod, "Agent", _FakeAgent)
    return runner, captured


def _jobs_client(home, monkeypatch):
    """A bare app with the bundled jobs plugin installed, open mode, no key."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from localm.plugins.engine import PluginManager

    store_root = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "builtin"
    app = FastAPI()
    mgr = PluginManager(app, store_root=store_root, installed_root=home / "plugins")
    mgr.install("jobs")
    return TestClient(app)


# --------------------------------------------------------------------------- #
#  The predicate this whole file relies on                                    #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [UNC, UNC_FWD, DEVICE])
def test_predicate_flags_unc_and_device(bad):
    assert is_unc_or_device_path(bad) is True


@pytest.mark.parametrize("ok", [".", "myproject", r"D:\projects\foo", "/home/x/proj"])
def test_predicate_allows_ordinary_paths(ok):
    assert is_unc_or_device_path(ok) is False


# --------------------------------------------------------------------------- #
#  (1) write-time: POST / PUT /api/jobs                                       #
# --------------------------------------------------------------------------- #
#
# Early rejection only: neither route calls Path.is_dir()/.resolve() on cwd at
# write time (only _run_coder does, at run time), so only the status code
# changes here. Sections (3) and (4) below cover syscall avoidance.

def test_post_jobs_rejects_unc_cwd(home, monkeypatch):
    """400 at creation, and nothing persisted."""
    from localm.plugins.builtin.jobs.store import JobStore
    with _jobs_client(home, monkeypatch) as c:
        r = c.post("/api/jobs", json={"name": "evil", "task_kind": "coder",
                                      "prompt": "do x", "cwd": UNC,
                                      "schedule_kind": "interval", "schedule": 60})
        assert r.status_code == 400, r.text
        assert "UNC" in r.text or "device" in r.text
    assert JobStore().list() == [], "a rejected job must never reach disk"


def test_post_jobs_rejects_device_cwd(home, monkeypatch):
    from localm.plugins.builtin.jobs.store import JobStore
    with _jobs_client(home, monkeypatch) as c:
        r = c.post("/api/jobs", json={"name": "evil2", "task_kind": "coder",
                                      "prompt": "do x", "cwd": DEVICE,
                                      "schedule_kind": "interval", "schedule": 60})
        assert r.status_code == 400, r.text
    assert JobStore().list() == []


def test_post_jobs_accepts_ordinary_cwd(home, tmp_path, monkeypatch):
    """Fires-control: a normal local cwd is unaffected by the guard."""
    work = tmp_path / "proj"
    work.mkdir()
    with _jobs_client(home, monkeypatch) as c:
        r = c.post("/api/jobs", json={"name": "ok", "task_kind": "coder",
                                      "prompt": "do x", "cwd": str(work),
                                      "schedule_kind": "interval", "schedule": 60})
        assert r.status_code == 200, r.text


def test_put_jobs_rejects_unc_cwd(home, tmp_path, monkeypatch):
    """PUT is a second write path into the same field and needs the same gate."""
    from localm.plugins.builtin.jobs.store import JobStore
    work = tmp_path / "proj"
    work.mkdir()
    with _jobs_client(home, monkeypatch) as c:
        r = c.post("/api/jobs", json={"name": "ok2", "task_kind": "coder",
                                      "prompt": "do x", "cwd": str(work),
                                      "schedule_kind": "interval", "schedule": 60})
        assert r.status_code == 200, r.text
        jid = r.json()["id"]

        r = c.put(f"/api/jobs/{jid}", json={"cwd": UNC})
        assert r.status_code == 400, r.text

    assert JobStore().get(jid).cwd == str(work), \
        "a rejected update must not be persisted"


def test_put_jobs_accepts_ordinary_cwd_change(home, tmp_path, monkeypatch):
    """Fires-control for PUT specifically: a legitimate cwd CHANGE (not just
    "the original value survives a rejected update") must actually apply. The
    test above only proves a bad update does not overwrite the original; this
    proves a good one does."""
    from localm.plugins.builtin.jobs.store import JobStore
    original = tmp_path / "proj"
    original.mkdir()
    new = tmp_path / "proj2"
    new.mkdir()
    with _jobs_client(home, monkeypatch) as c:
        r = c.post("/api/jobs", json={"name": "ok3", "task_kind": "coder",
                                      "prompt": "do x", "cwd": str(original),
                                      "schedule_kind": "interval", "schedule": 60})
        assert r.status_code == 200, r.text
        jid = r.json()["id"]

        r = c.put(f"/api/jobs/{jid}", json={"cwd": str(new)})
        assert r.status_code == 200, r.text

    assert JobStore().get(jid).cwd == str(new), \
        "a legitimate cwd update must actually be persisted"


# --------------------------------------------------------------------------- #
#  (2) write-time: the CLI (third write path into cwd)                        #
# --------------------------------------------------------------------------- #
#
# Early rejection only: job_add never touches cwd via Path before persisting it.

def test_cli_job_add_refuses_unc_cwd(home):
    from click.testing import CliRunner
    from localm.plugins.builtin.jobs.cli import main
    from localm.plugins.builtin.jobs.store import JobStore

    res = CliRunner().invoke(main, ["add", "j", "--coder", "--prompt", "do x",
                                    "--cwd", UNC, "--every", "60"])
    assert res.exit_code != 0, res.output
    assert "UNC" in res.output or "device" in res.output
    assert JobStore().list() == [], "a refused job must not reach disk"


def test_cli_job_add_accepts_ordinary_cwd(home, tmp_path):
    """Fires-control for the CLI writer."""
    from click.testing import CliRunner
    from localm.plugins.builtin.jobs.cli import main
    from localm.plugins.builtin.jobs.store import JobStore

    work = tmp_path / "proj"
    work.mkdir()
    res = CliRunner().invoke(main, ["add", "j", "--coder", "--prompt", "do x",
                                    "--cwd", str(work), "--every", "60"])
    assert res.exit_code == 0, res.output
    assert len(JobStore().list()) == 1


# --------------------------------------------------------------------------- #
#  (3) run-time (AUTHORITATIVE): a row already on disk, via run-now           #
# --------------------------------------------------------------------------- #

def test_run_now_refuses_a_persisted_unc_cwd_without_touching_filesystem(
        home, monkeypatch, fs_spy):
    """A row poisoned straight in the store - exactly what a build predating
    the write-time gate, or a direct store write, would leave behind - must
    still be refused when run on demand, and the fs boundary must never see
    the UNC string."""
    from localm.plugins.builtin.jobs.store import JobStore
    _runner_mod, captured = _fake_agent_capture(monkeypatch)

    with _jobs_client(home, monkeypatch) as c:
        r = c.post("/api/jobs", json={"name": "j", "task_kind": "coder",
                                      "prompt": "do x", "cwd": ".",
                                      "schedule_kind": "interval", "schedule": 60})
        assert r.status_code == 200, r.text
        jid = r.json()["id"]
        # Poison it directly in the store, bypassing plug.py's PUT gate: a row that
        # predates the write-time check.
        JobStore().update(jid, cwd=UNC)

        r = c.post(f"/api/jobs/{jid}/run")
        # run_job never raises (it catches internally), so the route always
        # 200s; the failure shows up in the recorded result's own status.
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "error", body
        assert "UNC" in body["error"] or "device" in body["error"]

    assert _unc_calls(fs_spy) == [], (
        "the UNC cwd reached a filesystem call - the SMB dial (and the "
        "net-NTLMv2 leak) happens there, before any status is chosen")
    assert captured == {}, "the coder Agent must never be constructed"


def test_run_now_ordinary_cwd_still_works(home, tmp_path, monkeypatch):
    """Control: an ordinary cwd is unaffected by the guard, end to end."""
    _runner_mod, captured = _fake_agent_capture(monkeypatch)
    work = tmp_path / "proj"
    work.mkdir()

    with _jobs_client(home, monkeypatch) as c:
        r = c.post("/api/jobs", json={"name": "j", "task_kind": "coder",
                                      "prompt": "do x", "cwd": str(work),
                                      "schedule_kind": "interval", "schedule": 60})
        jid = r.json()["id"]
        r = c.post(f"/api/jobs/{jid}/run")
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "ok", r.json()
    assert captured, "the coder Agent must actually have run"


# --------------------------------------------------------------------------- #
#  (4) run-time (AUTHORITATIVE): the autonomous SCHEDULER tick                #
# --------------------------------------------------------------------------- #
#
# The scheduler fires roughly every 30s with no caller at all, so a hostile cwd
# stored once keeps dialing SMB on a timer.

@pytest.mark.parametrize("bad", [UNC, UNC_FWD, DEVICE])
def test_scheduler_tick_refuses_a_persisted_bad_cwd_without_touching_filesystem(
        home, monkeypatch, fs_spy, bad):
    """Drives the REAL scheduler tick, real runner.run_job, real _run_coder
    (only the coder Agent and backend are faked), so this exercises the actual
    autonomous path with no request or caller in sight."""
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore

    _runner_mod, captured = _fake_agent_capture(monkeypatch)

    store = JobStore()
    # Persisted directly to the store, bypassing plug.py's _check_cwd.
    job = store.add(_make_job(cwd=bad, last_run=None))
    assert store.get(job.id).cwd == bad, "precondition: the row is on disk"

    sched = JobScheduler(store)     # real runner.run_job (module default)
    ran = sched.tick(now=1000.0)

    assert ran == [job.id], "the tick must attempt the due job, not silently skip it"
    assert _unc_calls(fs_spy) == [], (
        "the UNC/device cwd reached a filesystem call during an unattended "
        "scheduler tick - the SMB dial happens there, before any result is "
        "recorded")
    results = store.list_results(job.id)
    assert results and results[0]["status"] == "error"
    assert "UNC" in results[0]["error"] or "device" in results[0]["error"]
    assert captured == {}, "the coder Agent must never be constructed"


def test_scheduler_tick_runs_ordinary_cwd_normally(home, tmp_path, monkeypatch):
    """Control: with an ordinary cwd, the scheduler tick runs the job for
    real, proving the refusal above is specific to UNC/device syntax."""
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore

    _runner_mod, captured = _fake_agent_capture(monkeypatch)
    work = tmp_path / "proj"
    work.mkdir()
    store = JobStore()
    job = store.add(_make_job(cwd=str(work), last_run=None))

    sched = JobScheduler(store)
    ran = sched.tick(now=1000.0)

    assert ran == [job.id]
    results = store.list_results(job.id)
    assert results and results[0]["status"] == "ok"
    assert captured, "the coder Agent must actually have run"
