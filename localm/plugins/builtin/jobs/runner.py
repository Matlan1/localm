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

    try:
        from localm.inference.http_server import _engine as _live
        if _live is not None and _live.loaded:
            if not name or _live.display_name == name:
                return _live          # reuse - no VRAM cost, no load time
            
            # VRAM gate: unload the live engine if VRAM is tight. Uses
            # vram_capacity() (combined free across a configured multi-GPU
            # split, else the same single-GPU vram_info() number) - the same
            # "will this fit" ceiling switch_engine's eviction gate uses, so a
            # split-configured machine doesn't needlessly evict the live chat
            # engine when the combined capacity already covers the media job.
            from localm import vram as _vram
            from localm.discover import vram_capacity
            free = vram_capacity().get("free")
            est = _vram.media_estimate_bytes("chat")
            if _vram.should_swap_for_media(free, est):
                _live.unload()
                _vram.wait_for_vram_release(
                    lambda: vram_capacity().get("free"),
                    before_bytes=free)
    except Exception as e:
        # Best-effort live-engine reuse + VRAM gate. If anything here fails
        # (http_server not importable in this context, vram_info unavailable on
        # this platform, etc.) we fall through to loading a fresh engine below -
        # a degraded but correct path, not a hard failure. Surface the cause
        # (AGENTS.md rule 5: log, do not silently swallow) rather than muting it.
        from localm.debuglog import logger as _dbg
        _dbg.debug("jobs: live-engine reuse / VRAM gate skipped (%s); "
                   "loading a fresh engine instead", e)

    if not name:
        cfg = load_config()
        name = cfg.get("default_model") or cfg.get("model")
    if not name:
        from localm.model_manager import is_auto_chat_eligible
        reg = load_registry()
        # Auto-pick the first chat-eligible model; skip a type='unknown' model so a
        # background chat/memory job never silently loads one (it stays runnable when
        # a job explicitly configures default_model/model above).
        name = next((n for n in sorted(reg) if is_auto_chat_eligible(reg[n])), None)
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
    # Import the memory plugin's synthesizer directly (memory is its own plugin
    # now). This resolves the bundled-store source even when the memory plugin is
    # not installed/enabled - the privacy + write gates inside synthesize_memory
    # still apply. Safe ONLY while synthesize_memory stays module-level-stateless
    # (fresh store per call; shared state lives on disk/config) - keep it that way
    # (LM-DA-005). Guarded so a memory module that cannot import degrades to a
    # clear job result instead of crashing the runner.
    try:
        from localm.plugins.builtin.memory.plug import synthesize_memory
    except Exception as e:
        return f"Memory is unavailable, so nothing was consolidated: {e}"
    eng = engine
    if eng is None:
        raise RuntimeError(
            "no inference engine available (pass one, or register a model)")

    from localm.inference.textnorm import strip_think

    # Track when a reply was ALL reasoning (empty after the strip): "no facts
    # found" then needs a caveat, or a truncated thinking model reads as a
    # clean no-op (rule 5: a degraded run must not report unqualified success).
    state = {"empty_replies": 0}

    def complete(prompt: str) -> str:
        raw = "".join(
            eng.chat_stream([{"role": "user", "content": prompt}])).strip()
        # strip_think: memory must never ingest the reasoning channel (audit C1).
        text = strip_think(raw).strip()
        if raw and not text:
            state["empty_replies"] += 1
        return text

    result = synthesize_memory(complete)
    if result.get("status") == "skipped":
        return f"memory synthesis skipped ({result.get('reason')})"
    facts = result.get("facts") or []
    if not facts:
        if state["empty_replies"]:
            return ("memory synthesis: no facts extracted - the model produced "
                    "only reasoning output (%d reply/replies were empty after "
                    "removing the think channel; likely truncated by the "
                    "completion limit)" % state["empty_replies"])
        return "memory synthesis: no new durable facts found"
    return ("memory synthesis: added %d fact(s):\n" % result["added"]) + \
           "\n".join(f"- {f}" for f in facts)


# --------------------------------------------------------------------------- #
#  coder (best-effort)                                                         #
# --------------------------------------------------------------------------- #

def _shell_still_authorized(job: Job) -> bool:
    """Re-validate a shell-opt-in job's authorization at RUN time, so a revoked or
    expired key cannot keep an unattended scheduled job running with shell access
    forever (LM-DA-014: the runner used to just trust the stored ``allow_shell``
    flag - its own comment said so).

    ``job.owner`` is the sha256 key hash ``principal_id()`` stamps at creation
    (store.py) - the same value ``auth.key_hash_live()`` checks for a cookie
    session. Two cases need no re-check: a job with no owner (``allow_shell``
    needed no privileged key when ``any_key_configured()`` was False at creation,
    so there is no key whose liveness matters), and a job owned by the OWNER key
    itself (the owner key is not a keystore entry and is not revocable/expirable
    the way a scoped key is - mirrors ``key_hash_live``'s own "owner sessions are
    not gated on this" contract). Any other owner hash must still resolve to a
    live (unrevoked, unexpired) keystore key, or the run is downgraded to
    restricted rather than trusting a stale grant."""
    if job.owner is None:
        return True
    from localm.auth import _hash_key, get_api_key, key_hash_live
    owner_key = get_api_key()
    if owner_key and _hash_key(owner_key) == job.owner:
        return True
    return key_hash_live(job.owner)


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

    # Safe-by-default: an unattended scheduled run has nobody to approve a
    # destructive tool, so it runs RESTRICTED (read + confined edit, no run_shell,
    # no network, no sub-agents) unless the owner explicitly opted this job into
    # the full shell-capable coder. Restricted hard-refuses run_shell/fetch_url at
    # dispatch (agent.py), which closes both the indirect-injection -> run_shell
    # vector (the AutoJack analogue) and the jobs-scope -> shell privilege
    # escalation. The allow_shell opt-in is gated to owner / coder:full at the
    # creation route - but a stored True is not trusted FOREVER: the autonomous
    # scheduler tick has no request/caller to re-check (unlike run-now, which
    # re-validates the CALLER), so the runner re-validates the OWNING key's live
    # state on every run instead (LM-DA-014).
    restricted = not (getattr(job, "allow_shell", False) and _shell_still_authorized(job))

    from localm.plugins.coder.agent import Agent
    agent = Agent(
        backend,
        cwd.resolve(),
        auto_approve=True,        # unattended scheduled run: no interactive prompts
        mode=mode,
        scope=job.scope,
        restricted=restricted,
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
    localm server.

    URL resolution, most-authoritative first: LOCALM_SELF_URL (the live server
    publishes its OWN bind coordinates here at scheduler start, so an
    auto-bumped port is honoured), else the configured port. The old hardcoded
    :8080 was simply wrong - the default server binds 8642 - so a shipped coder
    job never reached the server on a stock install (memory-audit 2026-07-02).
    In open mode any api_key is accepted; in keyed mode the launcher injects
    LOCALM_API_KEY, so that is preferred over the open-mode placeholder."""
    import os

    from localm.plugins.coder.backends.http import HTTPBackend

    self_url = os.environ.get("LOCALM_SELF_URL")
    if not self_url:
        from localm.config import load_config
        port = load_config().get("port", 8642)
        self_url = f"http://127.0.0.1:{port}/v1"
    api_key = os.environ.get("LOCALM_API_KEY") or "localm"
    # self-connection: grammar sampling available
    return HTTPBackend(self_url, model=job.model or "localm", api_key=api_key,
                       localm_server=True)
