# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI diagnostics routes: run localm's ACTIVE self-checks and read the result.

  POST /api/doctor/run  -> start the checks as a background JOB; {"job_id"}
  GET  /api/doctor      -> the last report, plus whether a run is in flight

COVERS only the five ACTIVE probes from ``localm.diagnostics``: the ones that go
and try something rather than read a version string. The rest of what ``localm
doctor`` prints - VRAM, the GPU list, the installed backend, plugin pip extras,
the ComfyUI hint, the Python version, package versions - has its own GUI form
(``/api/stats``, ``/api/gpus``, ``/api/backend``, the Plugins page,
``/api/comfy/managed-status``, ``POST /api/bug-report``).

The aggregate verdict is therefore worded as a statement about THESE CHECKS,
never about the machine.

A JOB, NOT A BLOCKING CALL: a real run is ~20s on a healthy box and its worst
case is minutes (the ABI probe alone allows 120s, the venv probe 60s plus 30s
for pip). The child reports which check it has started, so the card can name the
one currently taking the time.

THE WORK RUNS IN A CHILD INTERPRETER: the HF-backend check imports torch and
transformers, and once llama.cpp's native runtime is loaded in a process - which
it is in any server that has served a GGUF model - that import is the
known-doomed DLL-identity conflict. See ``diagnostics.run_report_isolated``.

SCOPES. Reading the last report is ``config:read``. STARTING a run is
``config:write``: it spawns processes and builds a throwaway venv, the same
scope other privileged non-config actions use (``/api/logs/export``,
``/api/comfyui/create-launcher``).
"""

from __future__ import annotations

import time

from fastapi import Depends, FastAPI, HTTPException, Request

from localm import diagnostics, scopes
from localm.inference.http_server import principal_id, require_scope

# The job kind, which is also what ``has_running`` is keyed on. One run at a
# time.
_KIND = "diagnostics"


def _blank_state() -> dict:
    """The state a server starts with: nothing has been run.

    NOT persisted across a restart: a diagnostic is a statement about this
    machine at the moment it ran."""
    return {"job_id": None, "started_at": None, "finished_at": None,
            "progress": None, "report": None}


def _snapshot(app: FastAPI, jobs) -> dict:
    """The GET body. ``running`` comes from the job manager rather than from a
    flag this module maintains, so a worker thread that died without finishing
    cannot leave the card spinning forever.

    The worker WRITES this dict from a thread while the event loop READS it here.
    Nothing is lost (one run at a time, so one writer), but the three progress
    fields have to be read as a set: fetched one by one they could come from two
    different updates and report a phase against the wrong count. They are stored
    as ONE object and swapped in a single assignment, so a reader gets a
    consistent triple or the previous one, never a mixture."""
    st = getattr(app.state, "diagnostics_run", None) or _blank_state()
    running = jobs.has_running(_KIND)
    return {
        "running": running,
        "job_id": st["job_id"],
        "started_at": st["started_at"],
        "finished_at": st["finished_at"],
        # Only while something is in flight: a phase left over from a finished
        # run reads as a check still going.
        "progress": st["progress"] if running else None,
        "report": st["report"],
        # What this endpoint checks, so a client can name the five before the
        # first run.
        "covers": [{"key": k, "label": diagnostics.CHECK_LABELS[k]}
                   for k in diagnostics.CHECK_KEYS],
    }


def register(app: FastAPI, ctx) -> None:
    jobs = ctx.jobs
    app.state.diagnostics_run = _blank_state()

    @app.get("/api/doctor",
             dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def doctor_report_ep(request: Request):
        """The last diagnostics report, or nulls when nothing has been run.

        Read-only and cheap: it runs no probe and starts no process, so a card
        may poll it while a run is in flight."""
        return _snapshot(request.app, jobs)

    @app.post("/api/doctor/run",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def doctor_run_ep(request: Request):
        """Run the five active self-checks in a child interpreter, as a job.

        Refuses (409) while one is already running. Returns the job id so a
        client can stream it; the structured result is read back from
        ``GET /api/doctor``, never parsed out of the job log."""
        if jobs.has_running(_KIND):
            raise HTTPException(409, "A diagnostics run is already in progress.")

        # Published BEFORE start_fn, which starts its thread immediately: if the
        # reset happened after, a fast run could have its result overwritten by
        # this module's own initialisation. The worker only ever fills fields
        # in; the one late write is job_id, which nothing depends on.
        state = _blank_state()
        state["started_at"] = time.time()
        request.app.state.diagnostics_run = state

        def _work(job) -> bool:
            def _progress(key, label, done, total):
                # Both surfaces of the same fact: job.progress feeds
                # /api/activity and the job stream, the dict feeds this module's
                # own GET. ONE assignment, so a concurrent reader cannot see a
                # phase from this update beside a count from the last.
                job.progress(phase=label, done=done, total=total, unit="checks")
                state["progress"] = {"phase": label, "done": done, "total": total}

            report = diagnostics.run_report_isolated(on_progress=_progress)
            # finished_at BEFORE report, so a reader that can see the report can
            # always see when it landed. The reverse order leaves a window where
            # a finished report looks like it never finished.
            state["finished_at"] = time.time()
            state["report"] = report.as_dict()
            job.push({"type": "line", "text": _job_line(report)})
            # The JOB failed only when the RUN could not be completed. A
            # report that ran and found a real fault is a successful run with a
            # failing verdict.
            return report.verdict != diagnostics.ERROR

        job = jobs.start_fn(_KIND, _work, owner=principal_id(request),
                            label="System diagnostics")
        state["job_id"] = job.id
        return {"job_id": job.id}


def _job_line(report) -> str:
    """One line for the job log and the host console.

    Names the failing checks, not a count."""
    if report.verdict == diagnostics.ERROR:
        return f"diagnostics could not run: {report.error}"
    bad = [c.label for c in report.checks
           if c.status in (diagnostics.FAIL, diagnostics.WARN)]
    if not bad:
        return (f"diagnostics: all {len(report.checks)} active checks passed "
                "(this covers the active probes only, not the whole system)")
    return "diagnostics: " + ", ".join(bad) + " need attention"
