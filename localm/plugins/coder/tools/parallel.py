# SPDX-License-Identifier: AGPL-3.0-or-later
"""``dispatch_parallel``: run up to two child agents at once, each in its own git
worktree, and return their diffs for EXPLICIT human review.

RELATIONSHIP TO ``spawn_agent``
-------------------------------
``spawn_agent`` hands the child the parent's own ``cwd``, is marked destructive,
and runs strictly serially. This tool is the way to get concurrency: each child
gets its own checkout, so two children can edit the SAME file without
interfering, and the parent's working tree is never touched. Concurrency is
available only through the isolation, never without it.

WHAT IS AND IS NOT ISOLATED
---------------------------
Isolated: every file tool. They all resolve paths through ``_confine(cwd, path)``
(tools/base.py), which rejects anything outside ``cwd``. With ``cwd`` set to the
child's own worktree, file reads and writes cannot escape it.

NOT isolated: ``run_shell`` and ``run_tests``. Both are in
``_INTENTIONALLY_UNSCOPED`` (agent/constants.py), because a path-argument check
cannot confine arbitrary code. A child can still shell its way out of its
worktree. So worktree isolation is REAL for file tools and BEST-EFFORT for
shell; a dispatched child is not sandboxed.

NEVER AUTO-MERGE
----------------
Each child's work is committed to its own branch and the diff is returned.
Nothing is ever merged; the human decides.

The BRANCH is the durable artifact; the WORKTREE is transient. After a child
finishes its work is committed, its diff captured, and its worktree removed.
Committing is not merging: the parent's tree is untouched either way.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
from pathlib import Path
from typing import Any, Optional

from .. import child_limit
from .. import delegated as _delegated
from ..confirm import invoke_confirm
from .base import ToolResult, _truncate
from .git import (
    WORKTREE_PREFIX,
    WORKTREE_SUBDIR,
    _git,
    git_commit_all_in,
    git_list_child_worktrees,
    git_prune_child_worktrees,
    git_repo_root,
    git_worktree_add,
    git_worktree_remove,
)

# The cap on how many children one dispatch runs at once.
MAX_PARALLEL_TASKS = 2

# Wall-clock budget for one child.
DEFAULT_CHILD_TIMEOUT_S = 600

# How long a child waits for the confirmation channel to free up before giving up.
# Giving up denies the call (fail closed) and says so.
_CONFIRM_WAIT_S = 120

# Serialises human confirmation across concurrently running children: only one
# child may hold the channel at a time.
_CONFIRM_LOCK = threading.RLock()


def _serialised_confirm_handler(parent: Any, child_label: str):
    """The confirmation channel for one concurrent child.

    Wraps the parent's real channel so that (a) only one child can prompt at a
    time, and (b) the human is told WHICH child is asking before the prompt - on
    the terminal by the announcement line below, and in the GUI by the ``agent``
    keyword this relays to the parent's handler (../confirm.py).

    Fails closed: when the parent has no channel at all (a genuinely unattended
    run), this returns None, and the caller's ``needs_confirm`` branch denies the
    tool rather than self-approving. It never invents an approval - it only queues
    and labels real ones.
    """
    from .agents import _inherited_confirm_handler
    base = _inherited_confirm_handler(parent)
    if base is None:
        # Unattended: no channel to serialise. Returning None keeps the child on
        # execution.py's fail-closed branch.
        return None

    def handler(call, agent: Optional[str] = None) -> bool:
        # *agent* is the optional attribution keyword (see ../confirm.py). The
        # child's Agent supplies its own name there, so this normally relays it; the
        # fallback to child_label covers a caller that invokes the wrapper directly.
        agent = agent or child_label
        acquired = _CONFIRM_LOCK.acquire(timeout=_CONFIRM_WAIT_S)
        if not acquired:
            # Another child held the channel past the budget: deny, and say why.
            try:
                from ..display import console
                console.print(
                    f"    [yellow]sub-agent '{child_label}': confirmation channel "
                    f"busy for {_CONFIRM_WAIT_S}s - denying "
                    f"{getattr(call, 'name', 'tool')}[/yellow]"
                )
            except Exception:
                pass
            return False
        try:
            _announce_asker(agent, call)
            return invoke_confirm(base, call, agent=agent)
        finally:
            _CONFIRM_LOCK.release()

    return handler


def _announce_asker(child_label: str, call) -> None:
    """Name the child that is about to prompt.

    The prompt itself is rendered by the parent's own handler (terminal or GUI) and
    the ToolCall is not rewritten, so this prints an attribution line immediately
    before the prompt instead. This is the TERMINAL half of the attribution, and is
    what a REPL user sees even when a child borrows the parent's ``_confirm_tool``
    (a plain one-argument handler that cannot be told a label). The GUI half
    travels the confirm chain as the optional ``agent`` keyword (../confirm.py) and
    reaches the browser on the confirm_request event.
    """
    try:
        from ..display import console
        console.print(
            f"    [cyan]sub-agent '{child_label}' is asking to run "
            f"{getattr(call, 'name', 'a tool')}[/cyan]"
        )
    except Exception:
        # Attribution is a display aid; never let it break a confirmation flow.
        pass


def _normalise_tasks(tasks: Any) -> tuple[list[dict], Optional[str]]:
    """Accept the shapes a model actually emits and return [{name, task, model}].

    Tolerates a list of strings, a list of dicts, or a single string. Returns
    (tasks, error).
    """
    if tasks is None:
        return [], "dispatch_parallel needs a 'tasks' list"
    if isinstance(tasks, (str, bytes)):
        tasks = [tasks]
    if isinstance(tasks, dict):
        tasks = [tasks]
    if not isinstance(tasks, (list, tuple)):
        return [], f"'tasks' must be a list, got {type(tasks).__name__}"

    out: list[dict] = []
    for i, entry in enumerate(tasks):
        if isinstance(entry, str):
            out.append({"name": f"child{i + 1}", "task": entry, "model": None})
        elif isinstance(entry, dict):
            text = entry.get("task") or entry.get("prompt") or entry.get("description")
            if not text:
                return [], f"task {i + 1} has no 'task' text"
            raw_name = str(entry.get("name") or f"child{i + 1}")
            model = entry.get("model")
            if model is not None and not isinstance(model, str):
                # An unhashable model value is rejected here, at the edge, before
                # it reaches the set comprehension that builds the requested-model
                # set.
                return [], (f"task {i + 1}'s 'model' must be a model name (a "
                            f"string), got {type(model).__name__}")
            out.append({
                "name": _safe_label(raw_name) or f"child{i + 1}",
                "task": str(text),
                "model": model,
            })
        else:
            return [], f"task {i + 1} must be a string or an object, got {type(entry).__name__}"
    return out, None


def _absorb_child_errors(parent: Any, errors: list) -> None:
    """Fold a finished child's error trace into the parent, and nothing else.

    ERRORS ONLY. A child ran in its own worktree, so its ``_changed_files`` keys
    are relative to a tree this parent does not have and would resolve against the
    PARENT's cwd. Errors carry no path, so that half stays valid.

    Must be called on the parent's OWN thread once every worker has been waited
    on, never from a worker.
    """
    if not errors:
        return
    from ..agent.constants import _MAX_ERROR_TRACE
    trace = getattr(parent, "_error_trace", None)
    if trace is None:
        return
    trace.extend(errors)
    if len(trace) > _MAX_ERROR_TRACE:
        parent._error_trace = trace[-_MAX_ERROR_TRACE:]


def _safe_label(raw: str) -> str:
    """A filesystem- and git-safe slug for a branch/worktree name."""
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in raw.strip().lower()]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:32]


class _ChildOutcome:
    """What happened to one dispatched child, for the review report.

    SHARED BETWEEN TWO THREADS, so every write is synchronised. A child that
    outlives the batch deadline is ABANDONED, not stopped - a Python thread cannot
    be killed - so its worker is still running, and still holds a reference to this
    object, while the parent reports on it and tears the batch down. Two rules
    govern that:

    * The PARENT's terminal verdict is authoritative and one-way. ``seal()``
      records it and locks the object; no later worker write can move the status
      off it.
    * A WORKER publishes through ``publish()`` only, which writes every field under
      one lock acquisition, so a parent never sees ``status="ok"`` next to the
      ``turns=0`` of a child that had not finished.

    A refused publish is REPORTED in ``late_note``, never silently dropped: the
    child did finish, too late to be used, and the files it wrote are uncommitted
    in a worktree that was left in place.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        # ok | error | timeout | not_started | rejected. "timeout" means the child
        # was RUNNING when the batch deadline expired and was abandoned;
        # "not_started" means it never ran at all (see the wait loop).
        self.status = "pending"
        self.detail = ""
        self.errors: list = []   # the child's own _error_trace, folded into the
                                 # parent on the parent's thread after the wait
        self.branch = ""
        self.worktree = ""       # the worktree root (what we create and remove)
        self.child_cwd = None    # where the child actually runs inside it
        self.diff = ""
        self.turns = 0
        self.cleanup_warning = ""
        self.model = ""          # the model this child ACTUALLY ran on
        self.file_count = 0      # files this child changed, for the /diff footer
        # What an abandoned worker tried to report after the parent had already
        # given up on it. Rendered in the report.
        self.late_note = ""
        self._lock = threading.Lock()
        self._sealed = False
        self._sealed_at = 0.0

    def seal(self, status: str, detail: str) -> None:
        """Record the PARENT's terminal verdict and lock out later worker writes.

        Parent thread only. After this the status is final: a worker that finishes
        afterwards can no longer change what was reported.
        """
        with self._lock:
            self._sealed = True
            self._sealed_at = time.monotonic()
            self.status = status
            self.detail = detail

    def publish(self, **fields) -> bool:
        """Publish a WORKER's fields atomically. Worker thread only.

        Returns False - having written NOTHING - when the parent has already
        sealed a verdict for this child, and records the refusal for the report.
        """
        with self._lock:
            if self._sealed:
                self.late_note = self._describe_late_locked(fields)
                return False
            for key, value in fields.items():
                setattr(self, key, value)
            return True

    def _describe_late_locked(self, fields: dict) -> str:
        """Word a refused publish. Caller holds the lock."""
        late = time.monotonic() - self._sealed_at
        verdict = fields.get("status") or "a result"
        return (
            f"this child kept running and finished {late:.1f}s AFTER the batch was "
            f"given up on, reporting '{verdict}'. That result arrived too late to "
            f"be used, so the '{self.status}' above stands. Anything it wrote is "
            "uncommitted in its worktree; re-run the task if you need the result."
        )


def _run_one_child(parent: Any, spec: dict, child_cwd: Path, branch: str,
                   max_turns: int, outcome: _ChildOutcome) -> None:
    """Run one child agent to completion inside its own worktree.

    *child_cwd* is where the child runs, which mirrors the parent's position within
    the repo and so may be a subdirectory of the worktree root rather than the root
    itself. ``outcome.worktree`` remains the root, which is what gets committed,
    diffed, and removed.

    Runs on a pool worker, so every write to *outcome* MUST go through
    ``publish()``, never a bare attribute assignment: this thread can outlive the
    parent's wait. ``detail`` is accumulated in a LOCAL string rather than read
    back off the shared object, so there is no read-modify-write across the two
    threads either.
    """
    from ..agent import Agent
    from .agents import inherited_child_kwargs

    detail = ""
    backend = parent.backend
    model = spec.get("model")
    if model and model != getattr(backend, "model_id", None):
        # Falls back to the parent's backend when the requested model cannot be
        # loaded, recording why in detail.
        from ..backends.http import make_localm_backend
        raw_url = getattr(backend, "_base_url", "http://127.0.0.1:8642/v1")
        try:
            port = int(raw_url.split(":")[-1].split("/")[0])
        except Exception:
            port = 8642
        try:
            backend = make_localm_backend(model, port=port)
        except Exception as exc:
            detail = (f"requested model '{model}' unavailable ({exc}); "
                      "ran on the parent's model instead")
            backend = parent.backend

    # Published before the run so an abandoned child still reports which model it
    # was on. publish() refuses once the parent has sealed the verdict.
    outcome.publish(model=getattr(backend, "model_id", "") or "", detail=detail)

    # One shared constructor for every child agent, used by this path and by
    # spawn_agent. scope is INHERITED, not set to the worktree: cwd is what
    # confines the file tools here (each resolves through _confine). An inherited
    # glob like src/** stays meaningful because the child sits at the same
    # relative position inside its worktree as the parent occupies in the repo.
    child = Agent(**inherited_child_kwargs(
        parent,
        backend=backend,
        cwd=child_cwd,
        name=spec["name"],
        max_turns=max_turns,
        # The serialising wrapper, so concurrent children cannot talk over each
        # other on the parent's single confirmation channel.
        confirm_handler=_serialised_confirm_handler(parent, spec["name"]),
    ))
    # Record the child's location, so a caller reporting on a finished child knows
    # which worktree and branch its diff belongs to.
    child.worktree_path = outcome.worktree or str(child_cwd)
    child.worktree_branch = branch

    result_text = child.run_task(spec["task"])
    if result_text:
        from ..provenance import neutralise
        summary = neutralise(str(result_text))
        detail = (detail + "\n" if detail else "") + summary
    # ASK THE CHILD whether it succeeded: run_task RETURNS its failure message
    # rather than raising, so reaching here without an exception is not success.
    #
    # ONE publish for the whole verdict, so the parent can never read a status
    # next to the turns/errors/detail of a child that had not finished yet.
    # Refused outright if the parent already gave up on this child. The error
    # trace is stashed here and folded in on the parent's own thread once the wait
    # is over; the changed-file keys are not carried up at all.
    outcome.publish(
        turns=getattr(child, "turns", 0),
        status="ok" if getattr(child, "last_run_ok", True) else "error",
        errors=list(getattr(child, "_error_trace", None) or []),
        detail=detail,
    )


def tool_dispatch_parallel(
    cwd: Path,
    tasks: Any = None,
    max_turns: int = 10,
    timeout_s: int = DEFAULT_CHILD_TIMEOUT_S,
    _parent_agent: Optional[Any] = None,
) -> ToolResult:
    """
    Run up to 2 sub-tasks CONCURRENTLY, each in its own git worktree.

    Each child gets an isolated checkout on its own branch, so two children may edit
    the same file without interfering and your working tree is never modified. Each
    child's work is committed to its branch and its diff returned FOR REVIEW.
    Nothing is merged - you decide what to keep.

    Parameters
    ----------
    tasks:
        1 or 2 sub-tasks. Either plain strings, or objects with ``task`` (required),
        ``name``, and ``model``.
    max_turns:
        Per-child iteration cap (default 10).
    timeout_s:
        Wall-clock budget for the WHOLE batch (default 600), not per child: one
        shared deadline, because waiting the full budget on each child in turn
        would let two hung children block the parent for twice what the caller
        asked for. A child still running when it expires is abandoned and
        reported; a child that never got a slot is reported as never started.

    Notes
    -----
    This tool is registered ``destructive=True``, so under ``--dry-run`` the whole
    dispatch is skipped (no worktrees, no child turns burned), and an unattended
    parent with ``auto_approve=False`` and no confirm handler FAILS CLOSED on it
    instead of dispatching unconfirmed.
    """
    if _parent_agent is None:
        return ToolResult.error("dispatch_parallel requires a running parent agent")

    specs, err = _normalise_tasks(tasks)
    if err:
        return ToolResult.error(err)
    if not specs:
        return ToolResult.error("dispatch_parallel needs at least one task")
    if len(specs) > MAX_PARALLEL_TASKS:
        return ToolResult.error(
            f"dispatch_parallel takes at most {MAX_PARALLEL_TASKS} tasks, got "
            f"{len(specs)}. The cap matches the practical limit of about two "
            "resident models on one GPU; it is a deliberate design constraint, not "
            "a quota to retry around. Run the remaining work in a second dispatch."
        )

    repo = git_repo_root(cwd)
    if repo is None:
        return ToolResult.error(
            f"dispatch_parallel needs a git repository (worktree isolation is how "
            f"the children are kept apart), but '{cwd}' is not inside one."
        )

    # Who was already holding the budget BEFORE we take any of it: once we acquire,
    # describe_holders() would also list our own children.
    prior_holders = child_limit.describe_holders()

    # Everything that can hold a slot happens inside the try below, so the finally
    # is the single place the budget is returned. The acquire has to stay inside it:
    # child_limit's gate is a process-wide singleton with no timeout and no reaper,
    # so a slot skipped by an exception is never given back.
    tokens: list[Any] = []
    # Every future we submitted, read from the finally to tell which children are
    # still alive. Bound here so an early return cannot leave the name undefined.
    futures_started: list[Any] = []
    degraded = ""
    residency_note = ""
    outcomes = [_ChildOutcome(s["name"]) for s in specs]
    created: list[tuple[Path, str, _ChildOutcome]] = []
    run_id = uuid.uuid4().hex[:8]
    # Where the parent sits inside the repo, so each child is placed at the same
    # relative spot in its worktree rather than at the wider worktree root.
    try:
        _rel = cwd.resolve().relative_to(repo.resolve())
        rel_cwd = None if str(_rel) == "." else _rel
    except ValueError:
        rel_cwd = None

    try:
        # Take the concurrency budget before creating anything on disk, and inside
        # the guarded block so every exit path returns it.
        for spec in specs:
            tok = child_limit.try_acquire("parallel", spec["name"])
            if tok is None:
                break
            tokens.append(tok)

        if not tokens:
            return ToolResult.error(
                "no child-agent slots are free: " + prior_holders +
                f". The limit is {child_limit.MAX_CONCURRENT_CHILDREN} concurrent "
                "children in total across all dispatch mechanisms."
            )

        if len(tokens) < len(specs):
            degraded = (
                f"NOTE: only {len(tokens)} of {len(specs)} child slots were free "
                f"({prior_holders}), so the tasks ran with reduced "
                "parallelism rather than being rejected. Results are unaffected.\n\n"
            )

        # Two distinct models run genuinely side by side only when the server can
        # hold both resident; otherwise it evicts the idle LRU peer and the two
        # children take turns. That is not predicted here and the caller's model
        # choice is not rewritten: the note below states the condition and each
        # child reports the model it actually ran on.
        requested = {s.get("model") for s in specs if s.get("model")}
        if len(requested) > 1:
            residency_note = (
                "NOTE: two different models were requested. They run truly "
                "concurrently only if the server can hold both resident; if VRAM "
                "does not allow it, it will evict one to load the other and the "
                "children take turns instead of running in parallel. Each child's "
                "actual model is reported below. Omit 'model' to run both on the "
                "already-resident one.\n\n"
            )

        base_sha, ok = _git(repo, "rev-parse", "HEAD")
        if not ok:
            # An empty repo has no commit to branch a worktree from.
            return ToolResult.error(
                f"'{repo}' has no commits yet, so there is nothing to base an "
                "isolated worktree on. Make an initial commit first."
            )
        base_sha = base_sha.splitlines()[0].strip()

        # 1. Create one worktree per child.
        for spec, outcome in zip(specs, outcomes):
            label = _safe_label(spec["name"]) or "child"
            wt_name = f"{WORKTREE_PREFIX}{label}-{run_id}"
            worktree = repo / WORKTREE_SUBDIR / wt_name
            branch = f"coder/{label}-{run_id}"
            out, ok = git_worktree_add(repo, worktree, branch, base_sha)
            if not ok:
                # Sealed like every other parent verdict. No worker exists for this
                # child: it is dropped from runnable below and never submitted.
                outcome.seal("error",
                             f"could not create an isolated worktree: {out}")
                continue
            outcome.branch = branch
            outcome.worktree = str(worktree)
            # The child's cwd mirrors the parent's position WITHIN the repo. Handing
            # the child the worktree root would widen its reach to files the parent
            # itself could not touch, since cwd is what confines the file tools.
            outcome.child_cwd = worktree / rel_cwd if rel_cwd else worktree
            created.append((worktree, branch, outcome))

        runnable = [(s, o) for s, o in zip(specs, outcomes) if o.status == "pending"]
        if not runnable:
            return ToolResult.error(
                "dispatch_parallel could not create any isolated worktree:\n" +
                "\n".join(f"  {o.name}: {o.detail}" for o in outcomes)
            )

        # 2. Run them. max_workers is the number of slots we actually hold, so the
        #    global child budget is respected even when it forces sequential runs.
        pool = ThreadPoolExecutor(max_workers=max(1, len(tokens)))
        futures = {}
        try:
            for spec, outcome in runnable:
                wt = outcome.child_cwd or Path(outcome.worktree)
                fut = pool.submit(_run_one_child, _parent_agent, spec, wt,
                                  outcome.branch, max_turns, outcome)
                futures[fut] = outcome
                futures_started.append(fut)
            # ONE shared deadline for the whole batch, not a fresh timeout per
            # future, so two hung children cannot block the parent for twice the
            # budget the caller asked for.
            deadline = time.monotonic() + timeout_s
            for fut, outcome in futures.items():
                try:
                    fut.result(timeout=max(0.0, deadline - time.monotonic()))
                except _FuturesTimeout:
                    # cancel() succeeds only on a future that never started. With
                    # fewer slots than tasks the pool runs them in turn, so a child
                    # still QUEUED when the batch deadline expires is reported as
                    # not_started rather than as a timeout: it ran no turns and holds
                    # no worktree, so the teardown must not skip its worktree.
                    if fut.cancel():
                        outcome.seal("not_started", (
                            f"never started: the batch's {timeout_s}s budget was "
                            "already spent by the other child(ren) before a slot "
                            "freed up. It ran no turns and changed nothing. Re-run "
                            "it on its own, or raise timeout_s."
                        ))
                    else:
                        # Running, and a thread cannot be killed, only abandoned. It
                        # writes into its OWN worktree.
                        #
                        # SEALED, not assigned: cancel() returning False means the
                        # worker is still alive and still holding this object.
                        outcome.seal("timeout", (
                            f"exceeded the {timeout_s}s batch budget and was "
                            "abandoned. Its worktree is still held by the running "
                            "thread, so it was left in place rather than "
                            "force-removed."
                        ))
                except Exception as exc:
                    # The future raised, so its worker has stopped; seal anyway, so
                    # the parent's verdict is the one reported.
                    outcome.seal("error", f"{type(exc).__name__}: {exc}")
        finally:
            pool.shutdown(wait=False)

        # Fold each child's error trace into the parent, on the PARENT's thread now
        # that every worker has been waited on or abandoned. Errors only, never the
        # child's changed-file keys: those are relative to a tree this parent does
        # not have.
        for outcome in outcomes:
            if outcome.errors:
                _absorb_child_errors(_parent_agent, outcome.errors)

        # 3. Commit each child's work and capture its diff BEFORE teardown. A child
        #    that never started has nothing to commit and no diff to read.
        for worktree, branch, outcome in created:
            if outcome.status in ("timeout", "not_started"):
                continue
            msg = f"coder child '{outcome.name}' ({run_id})"
            out, ok = git_commit_all_in(worktree, msg)
            if not ok:
                outcome.cleanup_warning = f"could not commit child work: {out}"
                continue
            diff, dok = _git(worktree, "diff", base_sha, "HEAD", "--stat", "-p",
                             timeout=30)
            if dok:
                outcome.diff, _ = _truncate(diff)
                names, nok = _git(worktree, "diff", "--name-only", base_sha, "HEAD",
                                  timeout=30)
                outcome.file_count = (
                    len([ln for ln in names.splitlines() if ln.strip()])
                    if nok and names != "(no output)" else 0
                )
            else:
                outcome.cleanup_warning = f"could not read the child's diff: {diff}"

        # Record each change-set on the parent so /diff and /changes can POINT at
        # it. A pointer, never merged into the parent's own diff: session_diff()
        # feeds the self-reviewer and episodic memory.
        for _wt, branch, outcome in created:
            if not branch or outcome.status == "not_started":
                # Nothing ran, so there is no change set to point at: the branch is
                # still exactly base_sha and its worktree has been removed.
                continue
            _delegated.record(_parent_agent, _delegated.DelegatedChangeSet(
                label=outcome.name,
                branch=branch,
                file_count=outcome.file_count,
                source="parallel",
                status=outcome.status,
                base=base_sha,
                diff=outcome.diff,
            ))

    finally:
        # 4. Tear down every worktree we created, on success AND on failure, and
        #    release the concurrency budget no matter what happened.
        for worktree, branch, outcome in created:
            if outcome.status == "timeout":
                # A live thread still holds it; removing would fail "busy" anyway.
                outcome.cleanup_warning = (
                    f"worktree left in place at {worktree} (a live process still "
                    f"holds it); branch '{branch}' keeps any committed work"
                )
                continue
            out, ok = git_worktree_remove(repo, worktree)
            if not ok:
                # A failed removal is real (dirty tree, or a process still holding
                # it): report it rather than forcing past it.
                outcome.cleanup_warning = out
                continue
            if outcome.status == "not_started" and branch:
                # Nothing ever ran on it, so the branch is still exactly base_sha and
                # there is no work to preserve. -d, never -D: if it does carry a
                # commit, git refuses and the branch is kept.
                _git(repo, "branch", "-d", branch)

        # Scoped prune, with its outcome reported. A bare git worktree prune is
        # repo-wide and takes no pathspec, so it would also drop the record of a
        # user's worktree that lives on a drive that is not mounted right now. Only
        # runs when we actually created something.
        if created:
            pruned, pok = git_prune_child_worktrees(repo)
            if not pok and outcomes:
                # Append to whatever warning is already there, so a prune failure is
                # not dropped when every child already carries one.
                target = outcomes[0]
                target.cleanup_warning = (
                    f"{target.cleanup_warning}; {pruned}"
                    if target.cleanup_warning else pruned)

        # Return the budget only for children that have actually TERMINATED: an
        # abandoned child is still running, so its slot goes back when its thread
        # ends, not when this call returns. add_done_callback fires immediately if
        # the future finished in the meantime, so there is no lost-release race.
        #
        # Each token is paired with the child it was TAKEN FOR wherever the two
        # line up, since child_limit's holder list is what a later rejection names.
        # Where they cannot line up (fewer slots than tasks, so child2 occupies the
        # slot child1 finished with) a spare token is held back instead; the COUNT
        # is what must be right, the label is only the diagnostic.
        keep: list = []
        spare: list = []
        unpaired: list = []
        for i in range(max(len(tokens), len(futures_started))):
            tok = tokens[i] if i < len(tokens) else None
            fut = futures_started[i] if i < len(futures_started) else None
            live = fut is not None and not fut.done()
            if tok is not None and live:
                keep.append((fut, tok))
            elif tok is not None:
                spare.append(tok)
            elif live:
                unpaired.append(fut)
        for fut in unpaired:
            if not spare:
                break
            keep.append((fut, spare.pop(0)))
        for tok in spare:
            child_limit.release(tok)
        for fut, tok in keep:
            fut.add_done_callback(
                lambda _f, _tok=tok: child_limit.release(_tok))

    return ToolResult(
        ok=any(o.status == "ok" for o in outcomes),
        output=degraded + residency_note + _render_report(outcomes, repo),
        summary=_render_summary(outcomes),
    )


def _render_summary(outcomes: list[_ChildOutcome]) -> str:
    done = sum(1 for o in outcomes if o.status == "ok")
    return (f"dispatch_parallel: {done}/{len(outcomes)} child(ren) finished; "
            "diffs returned for review, nothing merged")


def _render_report(outcomes: list[_ChildOutcome], repo: Path) -> str:
    lines: list[str] = [
        "Parallel dispatch finished. NOTHING HAS BEEN MERGED - each child's work is "
        "committed on its own branch for you to review and merge (or delete).",
        "",
    ]
    for o in outcomes:
        lines.append(f"=== {o.name} [{o.status}] ===")
        if o.branch:
            lines.append(f"branch:   {o.branch}")
        if o.worktree:
            lines.append(f"worktree: {o.worktree}")
        if o.model:
            lines.append(f"model:    {o.model}")
        if o.turns:
            lines.append(f"turns:    {o.turns}")
        if o.detail:
            lines.append(o.detail.strip())
        if o.cleanup_warning:
            lines.append(f"WARNING: {o.cleanup_warning}")
        # A worker that finished after the parent gave up. Its result was refused,
        # the sealed verdict stands, and this note says the work exists and where.
        if o.late_note:
            lines.append(f"NOTE: {o.late_note}")
        if o.diff:
            lines.append("")
            lines.append(o.diff.strip())
        elif o.status == "ok":
            lines.append("(no file changes)")
        lines.append("")

    # Surface anything of ours still on disk. After a clean run this is empty; a
    # non-empty list means an earlier dispatch was hard-killed between creating a
    # worktree and removing it.
    live = [p for p in git_list_child_worktrees(repo)
            if str(p) not in {o.worktree for o in outcomes if o.status == "timeout"}]
    if live:
        lines.append("Worktrees from this or an earlier dispatch are still on disk "
                     "(a hard kill can orphan one):")
        for p in live:
            lines.append(f"    {p}")
        lines.append("    Remove a finished one with: git worktree remove <path>")
        lines.append("")

    merged = [o.branch for o in outcomes if o.branch and o.status == "ok"]
    if merged:
        lines.append("To take one of these, review the diff then merge its branch, "
                     "e.g.:")
        lines.append(f"    git -C {repo} merge --no-ff {merged[0]}")
    return "\n".join(lines)
