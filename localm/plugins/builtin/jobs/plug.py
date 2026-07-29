# SPDX-License-Identifier: AGPL-3.0-or-later
"""Jobs plugin: scheduled recurring tasks.

Routes (mounted by the engine, auto-scoped to the ``jobs`` capability):
  GET    /api/jobs                 - list jobs
  POST   /api/jobs                 - create a job
  GET    /api/jobs/{id}            - job detail
  PUT    /api/jobs/{id}            - update a job
  DELETE /api/jobs/{id}            - delete a job (and its results)
  POST   /api/jobs/{id}/run        - run the job now
  GET    /api/jobs/{id}/results    - past run results (newest first)

A job runs a chat or coder prompt, a memory-synthesis pass, or a RAG collection
re-sync, on an interval or 5-field cron schedule. The
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

from fastapi import APIRouter, Depends, HTTPException, Request
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
    task_kind: str = "chat"             # "chat" | "coder" | "memory" | "rag"
    # memory and rag jobs are fully specified without one, so this cannot be a
    # required field; Job.validate() still rejects an empty prompt for the kinds
    # that need it, so the rule stays in ONE place.
    prompt: str = ""
    schedule_kind: str = "interval"     # "interval" | "cron"
    schedule: "int | str" = 3600        # seconds, or a 5-field cron string
    model: "str | None" = None
    cwd: "str | None" = None
    scope: "str | None" = None
    collection: "str | None" = None     # rag jobs: the collection to re-sync
    allow_shell: bool = False           # coder jobs: opt in to full shell exec
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
    collection: "str | None" = None
    allow_shell: "bool | None" = None
    enabled: "bool | None" = None


def _check_model_name(model) -> None:
    """Refuse a job whose ``model`` is not a registered model name.

    The jobs scope is NOT in scopes.PRIVILEGED_SCOPES, so a deliberately
    restricted key reaches this field, and the field used to flow verbatim into
    get_model_info -> Engine(...). That let any jobs-scoped caller name an
    arbitrary path on disk and have the server stat it, walk it, read it, and -
    for a HuggingFace directory - execute its bundled .py through transformers.
    Registry membership is checked HERE, at the write, so a poisoned row never
    reaches disk in the first place (the runner re-checks at run time for rows
    persisted by an older build)."""
    from localm.model_manager import unregistered_model_error
    err = unregistered_model_error(model)
    if err:
        raise HTTPException(400, err)


def _caller_can_allow_shell(request: Request) -> bool:
    """True if the caller may opt a coder job into the full, shell-capable coder.

    Only the OWNER (open-mode loopback, or an ADMIN key) or a key holding the
    privileged ``coder:full`` scope qualifies. ``coder:full`` is owner-only to
    mint, so a plain ``jobs`` or ``coder`` key can never schedule an
    unrestricted-shell job: it gets the safe restricted coder. Mirrors the
    coder route's ``restricted = not (is_owner or coder:full)`` policy."""
    from localm import scopes as S
    from localm.auth import any_key_configured
    from localm.inference.http_server import caller_scopes
    if not any_key_configured():
        return True                              # open mode = loopback owner
    # caller_scopes resolves both a bearer key AND a cookie session (an opaque
    # session id) to its scope snapshot; a raw verify() would fail on a session id.
    held = caller_scopes(request)
    if not held:
        return False
    return S.ADMIN in held or S.CODER_FULL in held


def _caller_is_owner_key(request: Request) -> bool:
    """True when the caller's credential IS the owner key, i.e. the one credential
    whose authority does not come from a revocable keystore entry.

    This asks a POSITIVE question - "is this principal the current owner key?" -
    and reads nothing but the owner key itself. An earlier version of this fix
    asked the negative ("absent from the keystore, therefore the owner") via
    ``not key_hash_live(h)``, and that was wrong twice over (both caught by the
    pre-merge security review of REG-509):

    - ``verify()`` rejects an EXPIRED key on the bearer path, but expiry (unlike
      ``revoke_key``) neither deletes the keystore record nor drops sessions, and
      ``_principal_from_token`` exempts an ADMIN session from ``key_hash_live``
      entirely. So an expired ADMIN-scoped keystore key over its still-live cookie
      resolved a principal that ``key_hash_live`` called dead - and the negative
      test read that as "the owner", handing a revoked-by-expiry key permanent
      shell. That is precisely the LM-DA-014 hole this whole re-check exists to
      close.
    - ``_load_keystore()`` swallows OSError/ValueError and returns ``[]``, so a
      transient unreadable/corrupt auth.json makes ``key_hash_live`` say "not
      live" for a perfectly live key. Deriving a PERMANENT, persisted privilege
      stamp from a fail-open read is exactly backwards - and note the module's own
      ``_keystore_configured()`` deliberately fails CLOSED on those same errors.

    ADMIN is deliberately not consulted: it is in PRIVILEGED_SCOPES, so the owner
    may mint an ADMIN-scoped keystore key, and that key is revocable and must stay
    subject to the re-check.

    Returns False in open mode / for a tokenless caller: ``owner`` is then None,
    which the runner already treats as needing no re-check.

    KNOWN GAP (pre-existing, not a regression - master behaves the same): a job
    created through an owner SESSION whose minting key was ALREADY rotated away
    resolves to the old key's hash, so it does not match here and is not stamped.
    Closing that needs the session to record at MINT time that the owner key
    minted it (sessions.create), which is outside this change's blast radius.
    """
    from localm.auth import _hash_key, _legacy_owner_identity, ct_equal, get_api_key
    from localm.inference.http_server import principal_id
    owner_key = get_api_key()
    if not owner_key:
        return False
    h = principal_id(request)
    if h is None:
        return False
    # ct_equal is the house idiom for every secret compare (see auth.ct_equal).
    # Both sides here are computed hexdigests, so bare compare_digest could not
    # actually raise on them - this is for uniformity, so no compare_digest-on-str
    # remains anywhere to be copied to a site where the operand IS a raw header.
    #
    # The LEGACY unsalted digest is accepted alongside the current derived one for
    # the same reason as in the runner: the owner key's identity moved to a salted
    # KDF (CodeQL 88). Without it there is a window where a COOKIE request is the
    # first thing to touch the owner key after the upgrade - principal_id() reads
    # the session's still-legacy key_hash a few lines above, while _hash_key()
    # below is what triggers the derivation and re-link - so that one request
    # would fail to stamp a job as the owner's. This is an identity comparison
    # against an already-authenticated principal, not an authentication step.
    return (ct_equal(h, _hash_key(owner_key))
            or ct_equal(h, _legacy_owner_identity(owner_key)))


def _store() -> JobStore:
    return JobStore()


def _engine_resolver():
    """Resolve the live inference engine from the plugin host, if any."""
    try:
        from localm.inference.http_server import _engine as _live_engine
        if _live_engine is not None and _live_engine.loaded:
            return _live_engine
    except Exception as e:
        # Fall through to the host resolver, but log so a persistently broken
        # lookup is traceable rather than a silent degrade (AGENTS.md rule 5).
        from localm.debuglog import logger
        logger.debug("jobs engine resolver: live-engine lookup failed: %s", e)
    if _host is None:
        return None
    try:
        return _host.engine()
    except Exception as e:
        from localm.debuglog import logger
        logger.debug("jobs engine resolver: host.engine() failed: %s", e)
        return None


def _job_dict(job: Job) -> dict:
    d = job.to_dict()
    d.pop("owner", None)        # internal principal binding; never sent to the client
    d.pop("owner_is_owner_key", None)     # ditto: how that binding is re-validated
    return d


# ------------------------------------------------------------------ #
#  Routes                                                             #
# ------------------------------------------------------------------ #

def owned_job(job_id: str, request: Request) -> Job:
    """FastAPI dependency: fetch the job named by the job_id path param and
    enforce per-principal ownership - only the creating key (or an
    admin/owner) may touch it. Depends()-injectable (``job: Job =
    Depends(owned_job)``), so a new per-job route cannot omit the ownership
    check by construction (design-audit LM-DA-020). A mismatch returns the
    SAME 404 as a missing id so a foreign jobs-scoped key cannot even confirm
    another principal's job exists. owner=None (a tokenless / open-mode
    creation) stays unrestricted. Imports http_server lazily (not at module
    level) so this plugin still imports cleanly under a headless/no-engine
    harness."""
    from localm.inference.http_server import job_owner_ok
    job = _store().get(job_id)
    if job is None or not job_owner_ok(request, getattr(job, "owner", None)):
        raise HTTPException(404, f"No such job: {job_id}")
    return job


@_router.get("/api/jobs")
async def list_jobs(request: Request):
    # Only the caller's OWN jobs (an admin/owner sees all; unowned legacy jobs are
    # visible, matching job_owner_ok), so one jobs key cannot enumerate another
    # principal's scheduled jobs.
    from localm.inference.http_server import job_owner_ok
    return {"jobs": [_job_dict(j) for j in _store().list()
                     if job_owner_ok(request, getattr(j, "owner", None))]}


@_router.post("/api/jobs")
async def create_job(req: JobCreate, request: Request):
    # Opting a job into shell execution is privileged: owner / coder:full only.
    # Reject (do not silently downgrade) so the caller knows the opt-in was denied.
    if req.allow_shell and not _caller_can_allow_shell(request):
        raise HTTPException(
            403, "allow_shell needs the owner key or a coder:full key; a scheduled "
            "coder job otherwise runs restricted (read + confined edit, no shell).")
    _check_model_name(req.model)
    from localm.inference.http_server import principal_id
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
            collection=req.collection,
            allow_shell=req.allow_shell,
            enabled=req.enabled,
            owner=principal_id(request),    # bind the job to its creator
            # Capture WHAT KIND of credential that creator was, while it is still
            # resolvable: a rotated-away owner key is indistinguishable from a
            # revoked scoped key at run time (REG-509). Stamped for every job, not
            # just allow_shell ones, so a later PATCH that enables shell on an
            # owner-created job inherits the right answer.
            owner_is_owner_key=_caller_is_owner_key(request),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    _store().add(job)
    return _job_dict(job)


@_router.get("/api/jobs/{job_id}")
async def get_job(job: Job = Depends(owned_job)):
    return _job_dict(job)


@_router.put("/api/jobs/{job_id}")
async def update_job(job_id: str, req: JobUpdate, request: Request,
                     job: Job = Depends(owned_job)):
    store = _store()
    changes = {k: v for k, v in req.model_dump().items() if v is not None}
    can_shell = _caller_can_allow_shell(request)
    # Escalating an existing job to shell execution is privileged (same gate as
    # create); de-escalating (allow_shell -> False) is always allowed.
    if changes.get("allow_shell") and not can_shell:
        raise HTTPException(
            403, "allow_shell needs the owner key or a coder:full key.")
    # EDITING an ALREADY shell-enabled job is privileged too. Without this a
    # non-privileged caller could poison the prompt/cwd of an unowned (legacy/CLI/
    # open-mode, owner=None) allow_shell job - which the run_now re-check protects,
    # but the AUTONOMOUS SCHEDULER would then run with full shell on its next tick.
    # Gate the edit itself so the scheduler can only ever run an owner-authored prompt.
    if getattr(job, "allow_shell", False) and not can_shell:
        raise HTTPException(
            403, "editing a shell-enabled job needs the owner key or a coder:full key.")
    # PUT is the second write path into `model` and needs the same gate as POST,
    # or the create-time check is simply routed around with an update.
    if "model" in changes:
        _check_model_name(changes["model"])
    try:
        job = store.update(job_id, **changes)
    except KeyError:
        raise HTTPException(404, f"No such job: {job_id}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _job_dict(job)


@_router.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str, job: Job = Depends(owned_job)):
    store = _store()
    if not store.remove(job_id):
        raise HTTPException(404, f"No such job: {job_id}")
    return {"status": "deleted", "id": job_id}


@_router.post("/api/jobs/{job_id}/run")
async def run_now(job_id: str, request: Request, job: Job = Depends(owned_job)):
    store = _store()
    # Defense in depth (crown-jewel invariant): an on-demand run of a SHELL-enabled
    # job must re-check the CALLER, not just the stored flag - so an unowned/legacy
    # allow_shell job (owner=None) can never be triggered into run_shell by a plain
    # jobs key. The autonomous SCHEDULER path is unaffected (the owner set the flag).
    if getattr(job, "allow_shell", False) and not _caller_can_allow_shell(request):
        raise HTTPException(
            403, "running a shell-enabled job on demand needs the owner key or a "
            "coder:full key.")
    engine = _engine_resolver()
    loop = asyncio.get_running_loop()

    # Overlap guard (U-4): if a scheduled tick or another run-now is still in flight,
    # do NOT load a second model on top of it (that stacks VRAM and OOMs the GPU) -
    # tell the caller it is busy. Acquired and released inside the worker thread.
    def _run_guarded():
        from localm.plugins.builtin.jobs.runguard import run_slot
        with run_slot() as got_slot:
            if not got_slot:
                return None
            return _run_job(job, engine=engine)

    result = await loop.run_in_executor(None, _run_guarded)
    if result is None:
        raise HTTPException(
            409, "A job run is already in progress; try again once it finishes.")
    result_id = store.record_result(job_id, result)
    return {"result_id": result_id, **result}


@_router.get("/api/jobs/{job_id}/results")
async def job_results(job_id: str, limit: int = 100, offset: int = 0,
                      job: Job = Depends(owned_job)):
    store = _store()
    # Page the results so a high-frequency job's history cannot load every result
    # file into memory and OOM the API (CHK-JOBS-RESULTS-PAGE). Default + hard cap.
    limit = max(1, min(int(limit), 1000))
    offset = max(0, int(offset))
    return {"id": job_id, "limit": limit, "offset": offset,
            "results": store.list_results(job_id, limit=limit, offset=offset)}


# ------------------------------------------------------------------ #
#  Plugin lifecycle                                                   #
# ------------------------------------------------------------------ #

def register(host) -> None:
    """Mount the jobs routes and start the scheduler when running under a live
    server. The host auto-scopes every route to the ``jobs`` capability."""
    global _scheduler, _host
    _host = host
    host.mount_router(_router)
    # Serve the client_entry (static/jobs.js) at /plugins/jobs/ so the GUI's
    # loadClientPlugins() can import() it; without this the Jobs view 404s and
    # silently never loads (same pattern as the tts plugin).
    host.mount_static("static")

    scheduler = JobScheduler(JobStore(),
                             run_job=_run_job,
                             engine=_engine_resolver)
    _scheduler = scheduler
    # Start via the host's startup hook: on a stock server, register() runs
    # BEFORE uvicorn creates the event loop, so a direct start() no-opped and
    # NO scheduled job ever fired (memory-audit 2026-07-02, critical C2). The
    # hook runs the start once the loop exists (app lifespan), or immediately
    # when the plugin is enabled at runtime with the loop already up.
    on_startup = getattr(host, "on_startup", None)
    if callable(on_startup):
        def _on_loop():
            _publish_self_url(host)      # so scheduled coder jobs reach US
            scheduler.start()
        on_startup(_on_loop)
    else:
        # Minimal host (older embedder / unit-test fakes): keep the old
        # best-effort direct start, which works when a loop is running.
        try:
            scheduler.start()
        except Exception:
            pass


def _publish_self_url(host) -> None:
    """Publish the live server's OWN /v1 URL into LOCALM_SELF_URL so a scheduled
    coder job talks to THIS server (the actual, possibly auto-bumped, port),
    not a wrong hardcoded default. Only set when unset, so an explicit
    LOCALM_SELF_URL from the environment always wins. Best-effort: a missing
    app.state just leaves the runner's configured-port fallback in place."""
    import os
    if os.environ.get("LOCALM_SELF_URL"):
        return
    app = getattr(host, "_app", None)
    state = getattr(app, "state", None)
    port = getattr(state, "instance_port", None)
    if not port:
        return
    scheme = getattr(state, "instance_scheme", "http")
    os.environ["LOCALM_SELF_URL"] = f"{scheme}://127.0.0.1:{port}/v1"


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
