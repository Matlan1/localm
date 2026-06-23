# SPDX-License-Identifier: AGPL-3.0-or-later
"""Execute a single job and return a result record.

``run_job(job, *, engine=None)`` runs the job's prompt and returns a dict:
    {status: "ok"|"error", output: str, error: str|None,
     started: float, finished: float}

It NEVER raises - any failure is caught and reported as an ``error`` result, so
a scheduler tick can safely run many jobs in a row.

task_kind "chat":  the prompt is run against the inference engine. A passed-in
    ``engine`` is reused; otherwise one is loaded via the model manager from the
    job's ``model`` (or the active/first registered model).
task_kind "coder": a coder Agent runs the prompt in the job's ``cwd`` with the
    job's ``scope`` and the current privacy mode. The coder path is best-effort:
    a full agentic run needs the coder extra installed and a working backend; it
    is unit-tested with the agent/backend mocked.

Results are explicit user data (the store saves them in every privacy mode), but
any session TRACE a run would leave (audit JSONL, transcripts) still honours
``effective_mode`` - the coder Agent is constructed with the resolved mode, and
the chat path writes no trace of its own.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from localm.debuglog import logger
from localm.plugins.builtin.jobs.store import Job


def run_job(job: Job, *, engine=None) -> dict:
    """Run *job* and return a result record. Dispatches on task_kind and never
    raises.

    When no *engine* is passed for a chat/memory job, the runner loads one itself and
    UNLOADS it again afterwards, so a sequence of headless runs (a scheduler tick with
    no host model, or a CLI run) does not stack model loads in VRAM (U-4). An engine
    passed in by the live server (the host's shared model) is never unloaded here."""
    started = time.time()
    owned_engine = None
    try:
        eng = engine
        if job.task_kind in ("chat", "memory") and eng is None:
            eng = _load_engine(job.model)   # may raise (model not found) -> caught below
            owned_engine = eng              # we loaded it, so we unload it after the run
        if job.task_kind == "chat":
            output = _run_chat(job, engine=eng)
        elif job.task_kind == "coder":
            output = _run_coder(job, engine=engine)
        elif job.task_kind == "memory":
            output = _run_memory(job, engine=eng)
        else:
            raise ValueError(f"unknown task_kind: {job.task_kind!r}")
        return {
            "status": "ok",
            "output": output,
            "error": None,
            "prompt": job.prompt,
            "task_kind": job.task_kind,
            "model": job.model,
            "started": started,
            "finished": time.time(),
        }
    except Exception as e:
        return {
            "status": "error",
            "output": "",
            "error": f"{type(e).__name__}: {e}",
            "prompt": job.prompt,
            "task_kind": job.task_kind,
            "model": job.model,
            "started": started,
            "finished": time.time(),
        }
    finally:
        if owned_engine is not None:
            _unload_engine(owned_engine)


def _unload_engine(eng) -> None:
    """Release a model the runner loaded itself, freeing VRAM for the next run.
    Best-effort, but a failure is surfaced (not silenced): a leaked model would
    accumulate across scheduled runs while the job still reported success."""
    unload = getattr(eng, "unload", None)
    if not callable(unload):
        return
    try:
        unload()
    except Exception as e:
        logger.warning("jobs: failed to unload the run's own engine: %s", e)


# --------------------------------------------------------------------------- #
#  chat                                                                        #
# --------------------------------------------------------------------------- #

def _run_chat(job: Job, *, engine=None) -> str:
    """Run the prompt through the inference engine and return the reply text.

    Scheduled chat jobs get the same web-search tool the interactive chat has, so a
    web-lookup job no longer answers "I have no real-time access" (U-3); the bounded
    tool loop and the net_mode gating live in :mod:`webtool`. The chat path leaves no
    session trace of its own (no audit/transcript writes here), so it is privacy-safe
    regardless of mode; the explicit result is saved by the store like any other
    generated artifact."""
    eng = engine
    if eng is None:
        raise RuntimeError(
            "no inference engine available (pass one, or register a model)")
    from localm.plugins.builtin.jobs import webtool
    return webtool.run_chat_with_web(eng, job.prompt)


def _load_engine(model: Optional[str]):
    """Load an inference Engine for *model* (or the active/first registered
    model) via the model manager. Returns a loaded Engine, or None when no model
    can be resolved."""
    from localm.config import load_config, load_registry
    from localm.inference.engine import Engine
    from localm.model_manager import get_model_info

    name = model
    if not name:
        cfg = load_config()
        name = cfg.get("default_model") or cfg.get("model")
    if not name:
        reg = load_registry()
        if reg:
            name = sorted(reg)[0]
    if not name:
        return None
    info = get_model_info(name)
    if info is None:
        raise RuntimeError(f"model not found: {name}")
    model_path, display_hint = info
    eng = Engine(str(model_path), display_name=(name if model else display_hint))
    eng.load()
    return eng


# --------------------------------------------------------------------------- #
#  memory (A2 auto-synthesis)                                                  #
# --------------------------------------------------------------------------- #

def _run_memory(job: Job, *, engine=None) -> str:
    """Distil durable user facts from recent sessions into the assistant memory
    file, using the model. The privacy gate lives inside synthesize_memory (it
    skips with a clear status in privacy mode, never a silent success). Returns a
    human-readable summary saved as the job result."""
    from localm.plugins.builtin.chat.plug import synthesize_memory
    eng = engine
    if eng is None:
        raise RuntimeError(
            "no inference engine available (pass one, or register a model)")

    def complete(prompt: str) -> str:
        return "".join(
            eng.chat_stream([{"role": "user", "content": prompt}])).strip()

    result = synthesize_memory(complete)
    if result.get("status") == "skipped":
        return f"memory synthesis skipped ({result.get('reason')})"
    facts = result.get("facts") or []
    if not facts:
        return "memory synthesis: no new durable facts found"
    return ("memory synthesis: added %d fact(s):\n" % result["added"]) + \
           "\n".join(f"- {f}" for f in facts)


# --------------------------------------------------------------------------- #
#  coder (best-effort)                                                         #
# --------------------------------------------------------------------------- #

def _run_coder(job: Job, *, engine=None) -> str:
    """Run a coder Agent for the prompt in the job's cwd. Best-effort: requires
    the coder plugin and a reachable backend. Honours the current privacy mode
    for any session trace the agent would write.

    The agent always talks to an OpenAI-compatible HTTP endpoint, so a job run
    needs a localm server (``self_url``) reachable; without one this raises and
    the run is recorded as an error (never crashing the tick)."""
    from localm.audit import effective_mode

    cwd = Path(job.cwd or ".").expanduser()
    if not cwd.is_dir():
        raise RuntimeError(f"coder cwd is not a directory: {job.cwd}")

    backend = _coder_backend(job)
    mode = effective_mode("coder")

    from localm.plugins.coder.agent import Agent
    agent = Agent(
        backend,
        cwd.resolve(),
        auto_approve=True,        # unattended scheduled run: no interactive prompts
        mode=mode,
        scope=job.scope,
    )
    try:
        return (agent.run_task(job.prompt) or "").strip()
    finally:
        close = getattr(agent, "close", None)
        if callable(close):
            try:
                close()
            except Exception as e:
                # Surface (not silence) a failed cleanup: a hung subprocess or
                # leaked handle would otherwise accumulate across scheduled runs
                # while the job still reports success. Stay best-effort: the job
                # result is unaffected.
                logger.warning("coder agent cleanup failed: %s", e)


def _coder_backend(job: Job):
    """Build the HTTP backend the coder Agent talks to. Points at this machine's
    localm server (LOCALM_SELF_URL or the default local address)."""
    import os

    from localm.plugins.coder.backends.http import HTTPBackend

    self_url = (os.environ.get("LOCALM_SELF_URL")
                or "http://127.0.0.1:8080/v1")
    api_key = os.environ.get("LOCALM_API_KEY") or "localm"
    return HTTPBackend(self_url, model=job.model or "localm", api_key=api_key)
