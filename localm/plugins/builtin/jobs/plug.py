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
from localm.plugins.builtin.jobs.store import Job, JobStore, cwd_unc_error


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
    restricted key reaches this field, and an unchecked field flows verbatim into
    get_model_info -> Engine(...): any jobs-scoped caller could name an arbitrary
    path on disk and have the server stat it, walk it, read it, and - for a
    HuggingFace directory - execute its bundled .py through transformers.
    Registry membership is checked HERE, at the write, so a poisoned row never
    reaches disk in the first place (the runner re-checks at run time for rows
    persisted by an older build)."""
    from localm.model_manager import unregistered_model_error
    err = unregistered_model_error(model)
    if err:
        raise HTTPException(400, err)


def _effective_cwd(cwd, request: Request):
    """The working directory a job may actually be stored with: *cwd* as asked when
    the caller is allowed to choose one, else the instance's project root.

    ``_check_cwd`` below asks what SHAPE the path is; this asks WHO is choosing it.
    The coder route draws exactly this line - a restricted caller is forced into
    the project root "so a scoped key cannot point the (confined) file tools at
    arbitrary paths" - and without the same rule here a plain ``jobs``-scoped key
    gets read plus confined-write anywhere on the server just by scheduling a job
    there.

    CONFINED rather than REJECTED, the opposite of how ``allow_shell`` is handled
    two routes down: ``allow_shell`` is an optional opt-in, so refusing it still
    leaves a working job. ``cwd`` is MANDATORY for a coder job (Job.validate), so
    refusing it would remove the feature rather than restrict it - a jobs-scoped
    key could no longer schedule a coder job at all.

    Not silent either: the created or updated job is returned to the caller with
    its effective ``cwd``, so the value that will actually be used is visible in
    the response. The runner ALSO confines at run time (``_cwd_trusted``), because
    this route is not the only writer - the CLI and rows persisted by older builds
    reach the scheduler without passing here - and it reports the confinement in
    the job's own output when it bites."""
    if cwd is None or not str(cwd).strip():
        return cwd                    # nothing chosen: let validate() have its say
    if _caller_can_allow_shell(request):
        return cwd
    from localm.instances import resolve_root_dir
    return resolve_root_dir()


def _check_cwd(cwd) -> None:
    """Refuse a coder job's ``cwd`` when it is UNC or device syntax.

    Same shape as _check_model_name above, sharing store.cwd_unc_error's wording
    the same way _check_model_name shares unregistered_model_error's: checked
    HERE, at the write, so a poisoned row never persists via this route. This is
    early rejection only, not the authoritative guard - a row written before this
    check existed (or added straight to the store) is still on disk, and the
    autonomous scheduler tick has no caller to gate here. _run_coder's own check
    is authoritative because it runs immediately before the filesystem call, for
    every job regardless of when or how its cwd was set (mirroring the model-name
    re-check in runner.py's _load_engine)."""
    err = cwd_unc_error(cwd)
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

    A POSITIVE question - "is this principal the current owner key?" - reading
    nothing but the owner key itself. The negative form ("absent from the
    keystore, therefore the owner", via ``not key_hash_live(h)``) is wrong twice
    over:

    - ``verify()`` rejects an EXPIRED key on the bearer path, but expiry (unlike
      ``revoke_key``) neither deletes the keystore record nor drops sessions, and
      ``_principal_from_token`` exempts an ADMIN session from ``key_hash_live``
      entirely. So an expired ADMIN-scoped keystore key over its still-live cookie
      resolves a principal that ``key_hash_live`` calls dead, and the negative
      test would read that as "the owner", handing a revoked-by-expiry key
      permanent shell.
    - ``_load_keystore()`` swallows OSError/ValueError and returns ``[]``, so a
      transient unreadable or corrupt auth.json makes ``key_hash_live`` say "not
      live" for a perfectly live key. A PERMANENT, persisted privilege stamp must
      not be derived from a fail-open read - the module's own
      ``_keystore_configured()`` fails CLOSED on those same errors.

    ADMIN is not consulted: it is in PRIVILEGED_SCOPES, so the owner may mint an
    ADMIN-scoped keystore key, and that key is revocable and stays subject to the
    re-check.

    Returns False in open mode / for a tokenless caller: ``owner`` is then None,
    which the runner already treats as needing no re-check.

    TWO WAYS TO PROVE IT, because one of them cannot survive a roll:

    1. The SESSION's own mint-time record (``caller_minted_by_owner_key``). This
       is the authoritative answer for a cookie caller. An owner key ROLL leaves
       browser sessions alive (cli/keys.py: a GUI roll must not log the browser
       out), so a job created afterwards through that still-valid owner session
       resolves the OLD key's hash - matching neither the new owner key nor any
       keystore entry, i.e. indistinguishable from a REVOKED scoped key, which
       would strip shell from the owner's own automation.
    2. The key VALUE comparison below. It is the ONLY answer for a BEARER caller
       (no session exists to have recorded anything), and it covers a cookie
       session minted before the flag existed.

    Both are positive proofs, and (1) adds no authority: the session it reads is
    already ADMIN-scoped and already exempt from ``key_hash_live``, so the same
    caller can already create this job and drive the coder route with shell
    interactively. The flag is an attribute of the session, so it dies with it:
    logout, ``revoke_all`` (key clear / sign out everywhere) and expiry all revoke
    it, leaving the containment path for a leaked owner key unchanged.
    """
    from localm.auth import _hash_key, _legacy_owner_identity, ct_equal, get_api_key
    from localm.inference.http_server import caller_minted_by_owner_key, principal_id
    owner_key = get_api_key()
    if not owner_key:
        return False
    # Checked first because it is the one answer a rotation cannot invalidate. It
    # is gated on an owner key EXISTING (above) so open mode still answers False,
    # and it reads the session through the same re-validation every other cookie
    # consumer uses - never a bare sessions.lookup().
    if caller_minted_by_owner_key(request):
        return True
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


def _needs_owner_key_restamp(job: Job, request: Request) -> bool:
    """True when THIS request is provable proof that its caller now owns *job*
    the way ``owner_is_owner_key`` records, and the stamp has not been set yet.

    Repairs a shell-enabled job whose stamp is False because it was created over
    an owner COOKIE session, or by a build with no such field at all, AND that
    never ran while the owner key still had its creation-time value (disabled,
    long-cron, or created shortly before a roll), so ``_remember_owner_key_job``
    (runner.py) never got a run to prove it during. After a roll the runner alone
    cannot tell that job apart from a genuinely revoked scoped key (both hash to
    nothing live). An authenticated REQUEST is a second, independent proof:
    ``_caller_is_owner_key`` reads the session's own mint-time record
    (``caller_minted_by_owner_key``), which survives a roll, precisely when the
    runner's key-VALUE compare stops matching.

    The ``job.owner == principal_id(request)`` conjunction is load-bearing and is
    NOT redundant with ``_caller_is_owner_key`` alone: ``owned_job`` (via
    ``job_owner_ok``) lets an ADMIN caller reach ANOTHER principal's job, and that
    caller must never be able to upgrade THAT job to permanent owner-key shell
    just by running or editing it - only the job's own creator, proven as the
    owner key, may. ``principal_id`` returns the SAME frozen identity a cookie
    session was minted with (never re-derived from the current key value), so for
    the exact pre-roll session that created *job* the equality still holds after
    the roll."""
    if getattr(job, "owner_is_owner_key", False) or job.owner is None:
        return False
    if not _caller_is_owner_key(request):
        return False
    from localm.inference.http_server import principal_id
    return principal_id(request) == job.owner


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
        # lookup is traceable rather than a silent degrade.
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
    check by construction. A mismatch returns the SAME 404 as a missing id so a
    foreign jobs-scoped key cannot even confirm another principal's job exists.
    owner=None (a tokenless / open-mode creation) stays unrestricted. Imports
    http_server lazily (not at module level) so this plugin still imports
    cleanly under a headless/no-engine harness."""
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
                     if job_owner_ok(request, getattr(j, "owner", None))],
            "scheduler_warning": _scheduler.tick_error() if _scheduler else None}


@_router.post("/api/jobs")
async def create_job(req: JobCreate, request: Request):
    # Opting a job into shell execution is privileged: owner / coder:full only.
    # Reject (do not silently downgrade) so the caller knows the opt-in was denied.
    if req.allow_shell and not _caller_can_allow_shell(request):
        raise HTTPException(
            403, "allow_shell needs the owner key or a coder:full key; a scheduled "
            "coder job otherwise runs restricted (read + confined edit, no shell).")
    _check_model_name(req.model)
    _check_cwd(req.cwd)
    # Shape checked above; AUTHORITY checked here. A caller not allowed to choose a
    # directory gets the project root instead, and sees it in the response.
    req_cwd = _effective_cwd(req.cwd, request)
    from localm.inference.http_server import principal_id
    try:
        job = Job(
            name=req.name.strip(),
            task_kind=req.task_kind,
            prompt=req.prompt,
            schedule_kind=req.schedule_kind,
            schedule=req.schedule,
            model=req.model,
            cwd=req_cwd,
            scope=req.scope,
            collection=req.collection,
            allow_shell=req.allow_shell,
            enabled=req.enabled,
            owner=principal_id(request),    # bind the job to its creator
            # Capture WHAT KIND of credential that creator was, while it is still
            # resolvable: a rotated-away owner key is indistinguishable from a
            # revoked scoped key at run time. Stamped for every job, not just
            # allow_shell ones, so a later PATCH that enables shell on an
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
    # PUT is the second write path into `model` (and `cwd`) and needs the same
    # gates as POST, or the create-time checks are simply routed around with an
    # update.
    if "model" in changes:
        _check_model_name(changes["model"])
    if "cwd" in changes:
        _check_cwd(changes["cwd"])
        # Same authority gate as POST: PUT is the second write path into cwd, and
        # without this the create-time confinement is simply routed around with an
        # update.
        changes["cwd"] = _effective_cwd(changes["cwd"], request)
    # Fold the restamp into this SAME write rather than a second one - see
    # _needs_owner_key_restamp. "changes" has no such key from the request body
    # (JobUpdate carries no such field), so this can never be attacker-supplied.
    if _needs_owner_key_restamp(job, request):
        changes["owner_is_owner_key"] = True
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
    # A live, provable owner acting on their OWN job is a second proof the runner
    # cannot make on its own after a key roll (see _needs_owner_key_restamp). Done
    # BEFORE the run so _shell_still_authorized sees the repaired stamp on this
    # very run, not just the next one.
    if _needs_owner_key_restamp(job, request):
        try:
            job = store.update(job_id, owner_is_owner_key=True)
        except (KeyError, ValueError) as e:
            from localm.debuglog import logger
            logger.debug(
                "jobs: could not persist the owner-key re-stamp for job %s "
                "(%s); it will be re-derived on the next provable-owner "
                "request", job_id, e)
            job.owner_is_owner_key = True
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
    # file into memory and OOM the API. Default plus a hard cap.
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
    # BEFORE uvicorn creates the event loop, so a direct start() no-ops and NO
    # scheduled job ever fires. The hook runs the start once the loop exists
    # (app lifespan), or immediately when the plugin is enabled at runtime with
    # the loop already up.
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
    from localm.bindhost import self_connect_host, url_host
    host_part = url_host(self_connect_host(getattr(state, "bind_host", None)))
    os.environ["LOCALM_SELF_URL"] = f"{scheme}://{host_part}:{port}/v1"


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
