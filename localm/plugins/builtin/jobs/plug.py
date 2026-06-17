"""Jobs plugin: scheduled recurring tasks.

Routes (mounted by the engine, auto-scoped to the ``jobs`` capability):
  GET    /api/jobs                 - list jobs
  POST   /api/jobs                 - create a job
  GET    /api/jobs/{id}            - job detail
  PUT    /api/jobs/{id}            - update a job
  DELETE /api/jobs/{id}            - delete a job (and its results)
  POST   /api/jobs/{id}/run        - run the job now
  GET    /api/jobs/{id}/results    - past run results (newest first)

A job runs a chat or coder prompt on an interval or 5-field cron schedule. The
JobScheduler wakes periodically (~30s) and runs every enabled + due job via the
runner, recording each result. Job RESULTS are explicit user data (saved in any
privacy mode, like generated images); a coder/chat RUN's own session trace still
honours ``effective_mode``.

The scheduler starts only when the plugin is loaded under a running event loop
(a live server). Under a bare-FastAPI test harness or a headless import there is
no loop, so ``start`` is a safe no-op and the routes still work for run-now.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from localm.plugins.builtin.jobs import runner as _runner
from localm.plugins.builtin.jobs.scheduler import JobScheduler
from localm.plugins.builtin.jobs.store import Job, JobStore


def _run_job(job, *, engine=None):
    """Indirection through the runner MODULE (not a bound import) so the route
    and scheduler always call the live ``runner.run_job`` - and a test can patch
    it via the canonical ``localm.plugins.builtin.jobs.runner`` path."""
    return _runner.run_job(job, engine=engine)

_router = APIRouter()

# Module-level so unregister() can stop the scheduler the live server started.
_scheduler: "JobScheduler | None" = None
_host = None


# ------------------------------------------------------------------ #
#  Request models                                                     #
# ------------------------------------------------------------------ #

class JobCreate(BaseModel):
    name: str
    task_kind: str = "chat"             # "chat" | "coder"
    prompt: str
    schedule_kind: str = "interval"     # "interval" | "cron"
    schedule: "int | str" = 3600        # seconds, or a 5-field cron string
    model: "str | None" = None
    cwd: "str | None" = None
    scope: "str | None" = None
    enabled: bool = True


class JobUpdate(BaseModel):
    name: "str | None" = None
    task_kind: "str | None" = None
    prompt: "str | None" = None
    schedule_kind: "str | None" = None
    schedule: "int | str | None" = None
    model: "str | None" = None
    cwd: "str | None" = None
    scope: "str | None" = None
    enabled: "bool | None" = None


def _store() -> JobStore:
    return JobStore()


def _engine_resolver():
    """Resolve the live inference engine from the plugin host, if any."""
    if _host is None:
        return None
    try:
        return _host.engine()
    except Exception:
        return None


def _job_dict(job: Job) -> dict:
    return job.to_dict()


# ------------------------------------------------------------------ #
#  Routes                                                             #
# ------------------------------------------------------------------ #

@_router.get("/api/jobs")
async def list_jobs():
    return {"jobs": [_job_dict(j) for j in _store().list()]}


@_router.post("/api/jobs")
async def create_job(req: JobCreate):
    try:
        job = Job(
            name=req.name.strip(),
            task_kind=req.task_kind,
            prompt=req.prompt,
            schedule_kind=req.schedule_kind,
            schedule=req.schedule,
            model=req.model,
            cwd=req.cwd,
            scope=req.scope,
            enabled=req.enabled,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    _store().add(job)
    return _job_dict(job)


@_router.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = _store().get(job_id)
    if job is None:
        raise HTTPException(404, f"No such job: {job_id}")
    return _job_dict(job)


@_router.put("/api/jobs/{job_id}")
async def update_job(job_id: str, req: JobUpdate):
    changes = {k: v for k, v in req.model_dump().items() if v is not None}
    try:
        job = _store().update(job_id, **changes)
    except KeyError:
        raise HTTPException(404, f"No such job: {job_id}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _job_dict(job)


@_router.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    if not _store().remove(job_id):
        raise HTTPException(404, f"No such job: {job_id}")
    return {"status": "deleted", "id": job_id}


@_router.post("/api/jobs/{job_id}/run")
async def run_now(job_id: str):
    store = _store()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, f"No such job: {job_id}")
    engine = _engine_resolver()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: _run_job(job, engine=engine))
    result_id = store.record_result(job_id, result)
    return {"result_id": result_id, **result}


@_router.get("/api/jobs/{job_id}/results")
async def job_results(job_id: str):
    store = _store()
    if store.get(job_id) is None:
        raise HTTPException(404, f"No such job: {job_id}")
    return {"id": job_id, "results": store.list_results(job_id)}


# ------------------------------------------------------------------ #
#  Plugin lifecycle                                                   #
# ------------------------------------------------------------------ #

def register(host) -> None:
    """Mount the jobs routes and start the scheduler when running under a live
    server. The host auto-scopes every route to the ``jobs`` capability."""
    global _scheduler, _host
    _host = host
    host.mount_router(_router)

    scheduler = JobScheduler(JobStore(),
                             run_job=_run_job,
                             engine=_engine_resolver)
    # start() is a safe no-op when there is no running event loop (tests /
    # headless import), so a missing loop never breaks loading the plugin.
    try:
        scheduler.start()
    except Exception:
        pass
    _scheduler = scheduler


def unregister() -> None:
    """Stop the scheduler on disable/uninstall."""
    global _scheduler, _host
    if _scheduler is not None:
        try:
            _scheduler.stop()
        except Exception:
            pass
        _scheduler = None
    _host = None
