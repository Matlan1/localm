# SPDX-License-Identifier: AGPL-3.0-or-later
"""``dispatch_parallel``: run up to two child agents at once, each in its own git
worktree, and return their diffs for EXPLICIT human review.

WHY THIS EXISTS SEPARATELY FROM ``spawn_agent``
-----------------------------------------------
``spawn_agent`` hands the child the parent's own ``cwd``. Two of those running at
once would edit the same files in the same directory with no isolation at all, so
``spawn_agent`` is marked destructive and therefore runs strictly serially. This
tool is the safe way to get real concurrency: each child gets its own checkout, so
two children can edit the SAME file without interfering, and the parent's working
tree is never touched.

``spawn_agent`` staying serial is the permanent design, not a stopgap this tool
lifts. Concurrency is available only through the isolation, never without it.

WHAT IS AND IS NOT ISOLATED (do not overstate this)
---------------------------------------------------
Isolated: every file tool. They all resolve paths through ``_confine(cwd, path)``
(tools/base.py:84), which rejects anything outside ``cwd``. With ``cwd`` set to the
child's own worktree, file reads and writes genuinely cannot escape it.

NOT isolated: ``run_shell`` and ``run_tests``. Both are in
``_INTENTIONALLY_UNSCOPED`` (agent/constants.py:36-38) because a path-argument check
cannot confine arbitrary code. A child can still shell its way out of its worktree.
So worktree isolation is REAL for file tools and BEST-EFFORT for shell. Describe it
that way; do not call a dispatched child sandboxed.

NEVER AUTO-MERGE
----------------
Each child's work is committed to its own branch and the diff is returned. Nothing
is ever merged. A local model resolving a merge conflict unsupervised is the worst
version of silently papering over a problem, so the human always decides.

The BRANCH is the durable artifact; the WORKTREE is transient. After a child
finishes we commit its work, capture the diff, and remove the worktree. That way
the review artifact survives while nothing is left lying around on disk. Committing
is not merging: the parent's tree is untouched either way.
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
from .base import ToolResult, _truncate
from .git import (
    WORKTREE_PREFIX,
    WORKTREE_SUBDIR,
    _git,
    git_commit_all_in,
    git_list_child_worktrees,
    git_repo_root,
    git_worktree_add,
    git_worktree_prune,
    git_worktree_remove,
)

# The cap. Deliberately a design constraint (the practical resident-model ceiling on
# one consumer GPU), not a limit to engineer around. This is not a scheduler.
MAX_PARALLEL_TASKS = 2

# Wall-clock budget for one child. A hung child must be reported, never allowed to
# hang the parent or its sibling.
DEFAULT_CHILD_TIMEOUT_S = 600

# How long a child will wait for the confirmation channel to free up before giving
# up. Bounded on purpose: an unbounded wait would let one unanswered prompt hang the
# whole dispatch, which is exactly the failure this tool must not have. Giving up
# DENIES (fail closed) and says so.
_CONFIRM_WAIT_S = 120

# Serialises human confirmation across concurrently running children. Two children
# prompting on one terminal at the same time would interleave their questions and
# leave the human unable to tell which answer applied to which child - a correctness
# problem, not just a cosmetic one. Only one child may hold the channel at a time.
_CONFIRM_LOCK = threading.RLock()


def _serialised_confirm_handler(parent: Any, child_label: str):
    """The confirmation channel for one concurrent child.

    Wraps the parent's real channel so that (a) only one child can prompt at a
    time, and (b) the human is told WHICH child is asking before the prompt.

    The fail-closed property is preserved exactly: when the parent has no channel
    at all (a genuinely unattended run), this returns None, and the caller's
    ``needs_confirm`` branch denies the tool rather than self-approving. This never
    invents an approval - it only queues and labels real ones.
    """
    from .agents import _inherited_confirm_handler
    base = _inherited_confirm_handler(parent)
    if base is None:
        # Unattended: no channel to serialise. Returning None keeps the child on
        # execution.py's fail-CLOSED branch. Do NOT substitute a permissive default.
        return None

    def handler(call) -> bool:
        acquired = _CONFIRM_LOCK.acquire(timeout=_CONFIRM_WAIT_S)
        if not acquired:
            # Another child has held the human's attention past the budget. Deny,
            # loudly. Denying is the safe direction, and saying why keeps this from
            # looking like a mysterious rejection.
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
            _announce_asker(child_label, call)
            return base(call)
        finally:
            _CONFIRM_LOCK.release()

    return handler


def _announce_asker(child_label: str, call) -> None:
    """Name the child that is about to prompt.

    With several children alive, an unattributed "run_shell? [y/N]" is ambiguous.
    The prompt itself is rendered by the parent's own handler (terminal or GUI) and
    we deliberately do not rewrite the ToolCall to smuggle a label into it, so this
    prints an attribution line immediately before the prompt instead.

    KNOWN LIMIT, stated rather than hidden: for a GUI/web confirm_handler this line
    goes to the server console, not the browser, so a GUI user sees a serialised but
    unlabelled prompt. The serialisation (the correctness half) holds in both cases;
    labelling in the GUI needs a handler-signature change that belongs with the GUI.
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

    Tolerates a list of strings, a list of dicts, or a single string, because a
    local model will produce all three. Returns (tasks, error).
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
            out.append({
                "name": _safe_label(raw_name) or f"child{i + 1}",
                "task": str(text),
                "model": entry.get("model"),
            })
        else:
            return [], f"task {i + 1} must be a string or an object, got {type(entry).__name__}"
    return out, None


def _safe_label(raw: str) -> str:
    """A filesystem- and git-safe slug for a branch/worktree name."""
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in raw.strip().lower()]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:32]


class _ChildOutcome:
    """What happened to one dispatched child, for the review report."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.status = "pending"      # ok | error | timeout | rejected
        self.detail = ""
        self.branch = ""
        self.worktree = ""       # the worktree root (what we create and remove)
        self.child_cwd = None    # where the child actually runs inside it
        self.diff = ""
        self.turns = 0
        self.cleanup_warning = ""
        self.model = ""          # the model this child ACTUALLY ran on
        self.file_count = 0      # files this child changed, for the /diff footer


def _run_one_child(parent: Any, spec: dict, child_cwd: Path, branch: str,
                   max_turns: int, outcome: _ChildOutcome) -> None:
    """Run one child agent to completion inside its own worktree.

    *child_cwd* is where the child runs, which mirrors the parent's position within
    the repo and so may be a subdirectory of the worktree root rather than the root
    itself. ``outcome.worktree`` remains the root, which is what gets committed,
    diffed, and removed.
    """
    from ..agent import Agent
    from ..audit import SessionMode as _SessionMode

    backend = parent.backend
    model = spec.get("model")
    if model and model != getattr(backend, "model_id", None):
        # A second model is only a WIN when the server can hold both resident. When
        # it cannot, falling back to the parent's backend keeps the dispatch working
        # at reduced benefit rather than failing outright - and we say so in detail.
        from ..backends.http import make_localm_backend
        raw_url = getattr(backend, "_base_url", "http://127.0.0.1:8642/v1")
        try:
            port = int(raw_url.split(":")[-1].split("/")[0])
        except Exception:
            port = 8642
        try:
            backend = make_localm_backend(model, port=port)
        except Exception as exc:
            outcome.detail = (f"requested model '{model}' unavailable ({exc}); "
                              "ran on the parent's model instead")
            backend = parent.backend

    outcome.model = getattr(backend, "model_id", "") or ""

    child = Agent(
        backend=backend,
        cwd=child_cwd,
        name=spec["name"],
        max_turns=max_turns,
        verbose=False,
        # Same posture inheritance spawn_agent uses: a child is never LESS confirmed
        # than its parent, and never MORE capable. The confirm handler is the
        # serialising wrapper so concurrent children cannot talk over each other.
        auto_approve=getattr(parent, "auto_approve", True),
        dry_run=getattr(parent, "dry_run", False),
        always_confirm=getattr(parent, "always_confirm", None),
        confirm_handler=_serialised_confirm_handler(parent, spec["name"]),
        parent=parent,
        mode=getattr(parent, "mode", _SessionMode.PRIVACY),
        restricted=getattr(parent, "restricted", False),
        disabled_tools=getattr(parent, "disabled_tools", frozenset()),
        # DELIBERATE: inherit the parent's scope rather than overwriting it with the
        # worktree path. Two reasons, and this is a choice, not an oversight.
        #
        # 1. cwd is what actually confines the file tools - every one of them
        #    resolves through _confine(cwd, path) - so the worktree isolation is
        #    already enforced without touching scope. Setting scope to the worktree
        #    would add nothing and would DISCARD a narrowing the parent had.
        # 2. A scope is a glob relative to the agent's own cwd. Because the child is
        #    placed at the SAME relative position inside its worktree as the parent
        #    occupies in the repo, an inherited glob like "src/**" still designates
        #    the corresponding files. That equivalence is exactly why the mirroring
        #    above matters; without it an inherited scope would silently point
        #    somewhere else.
        #
        # Note this OVERRIDES an inherited value rather than filling an empty one:
        # spawn_agent also passes the parent's scope down, so the attribute is
        # normally already set. Passing it explicitly keeps the two paths agreeing.
        scope=getattr(parent, "scope", None),
    )
    # Make the child's location introspectable: a caller that reports on a finished
    # child needs to know which worktree and branch its diff belongs to.
    child.worktree_path = outcome.worktree or str(child_cwd)
    child.worktree_branch = branch

    result_text = child.run_task(spec["task"])
    outcome.turns = getattr(child, "turns", 0)
    outcome.status = "ok"
    if result_text:
        from ..provenance import neutralise
        summary = neutralise(str(result_text))
        outcome.detail = (outcome.detail + "\n" if outcome.detail else "") + summary


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
        Per-child wall-clock budget (default 600). A child that exceeds it is
        abandoned and reported, never left to hang the dispatch.

    Notes
    -----
    This tool is registered ``destructive=True``, which has two consequences worth
    knowing rather than discovering: under ``--dry-run`` the whole dispatch is
    skipped (no worktrees, no child turns burned), and an unattended parent with
    ``auto_approve=False`` and no confirm handler FAILS CLOSED on it instead of
    dispatching unconfirmed. Both are intended - dispatching two children that
    write files is exactly the kind of action that should need a human.
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

    # Who was already holding the budget BEFORE we take any of it. Captured up
    # front because once we acquire, describe_holders() would also list OUR OWN
    # children and report them to the user as if they were the blockers.
    prior_holders = child_limit.describe_holders()

    # Take the concurrency budget BEFORE creating anything on disk.
    tokens: list[Any] = []
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

    degraded = ""
    if len(tokens) < len(specs):
        degraded = (
            f"NOTE: only {len(tokens)} of {len(specs)} child slots were free "
            f"({prior_holders}), so the tasks ran with reduced "
            "parallelism rather than being rejected. Results are unaffected.\n\n"
        )

    # Two DISTINCT models only run genuinely side by side when the server can hold
    # both resident: it loads alongside with no eviction when free VRAM is measured
    # and fits, and otherwise evicts the idle LRU peer (http_server.py). Requesting
    # two models that do not both fit therefore means the two children take turns
    # evicting each other rather than running in parallel.
    #
    # We do NOT predict that here. Deciding it needs the per-model VRAM sizing the
    # server already does, and re-deriving it in a coder tool would be duplicated,
    # brittle, and a second place to get wrong; probing GPUs from a hot path is also
    # what once hung the server. Nor do we silently rewrite the caller's explicit
    # model choice. So the server stays the authority, the SAFE case is the default
    # (omit `model` and both children share the parent's already-resident engine),
    # and when two distinct models ARE asked for we say what the condition is and
    # report which model each child actually ran on.
    requested = {s.get("model") for s in specs if s.get("model")}
    residency_note = ""
    if len(requested) > 1:
        residency_note = (
            "NOTE: two different models were requested. They run truly "
            "concurrently only if the server can hold both resident; if VRAM does "
            "not allow it, it will evict one to load the other and the children "
            "take turns instead of running in parallel. Each child's actual model "
            "is reported below. Omit 'model' to run both on the already-resident "
            "one.\n\n"
        )

    outcomes = [_ChildOutcome(s["name"]) for s in specs]
    created: list[tuple[Path, str, _ChildOutcome]] = []
    run_id = uuid.uuid4().hex[:8]
    # Where the parent sits inside the repo, so each child can be placed at the
    # SAME relative spot in its worktree rather than at the (wider) worktree root.
    try:
        _rel = cwd.resolve().relative_to(repo.resolve())
        rel_cwd = None if str(_rel) == "." else _rel
    except ValueError:
        rel_cwd = None

    try:
        base_sha, ok = _git(repo, "rev-parse", "HEAD")
        if not ok:
            # An empty repo has no commit to branch a worktree from. Say that
            # plainly instead of letting `worktree add` fail with git's wording.
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
                outcome.status = "error"
                outcome.detail = f"could not create an isolated worktree: {out}"
                continue
            outcome.branch = branch
            outcome.worktree = str(worktree)
            # The child's cwd mirrors the parent's position WITHIN the repo. If the
            # parent is working in repo/sub, handing the child the worktree ROOT
            # would widen its reach to files the parent itself could not touch
            # (cwd is what confines the file tools), so re-apply the same subpath.
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
            # ONE shared deadline for the whole batch, not a fresh timeout per
            # future: waiting `timeout_s` on each in turn would let two hung
            # children block the parent for twice the budget the caller asked for.
            deadline = time.monotonic() + timeout_s
            for fut, outcome in futures.items():
                try:
                    fut.result(timeout=max(0.0, deadline - time.monotonic()))
                except _FuturesTimeout:
                    # The thread cannot be killed, only abandoned. It is writing
                    # into its OWN worktree, so an abandoned child corrupts only its
                    # own checkout - which is exactly what the isolation buys us.
                    outcome.status = "timeout"
                    outcome.detail = (
                        f"exceeded its {timeout_s}s budget and was abandoned. Its "
                        "worktree is still held by the running thread, so it was "
                        "left in place rather than force-removed."
                    )
                    fut.cancel()
                except Exception as exc:
                    outcome.status = "error"
                    outcome.detail = f"{type(exc).__name__}: {exc}"
        finally:
            pool.shutdown(wait=False)

        # 3. Commit each child's work and capture its diff BEFORE teardown.
        for worktree, branch, outcome in created:
            if outcome.status == "timeout":
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
        # it. Deliberately a pointer, never merged into the parent's own diff:
        # session_diff() feeds the self-reviewer and episodic memory, so foreign
        # content there would corrupt two model-facing loops, not just a view.
        for _wt, branch, outcome in created:
            if not branch:
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
                # A failed removal is REAL (dirty tree, or a process still holding
                # it). Report it so the operator can reap it; never --force past it.
                outcome.cleanup_warning = out
        try:
            git_worktree_prune(repo)
        except Exception:
            pass
        for tok in tokens:
            child_limit.release(tok)

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
        if o.diff:
            lines.append("")
            lines.append(o.diff.strip())
        elif o.status == "ok":
            lines.append("(no file changes)")
        lines.append("")

    # Surface anything of ours still on disk. After a clean run this is empty; a
    # non-empty list means an earlier dispatch was hard-killed between creating a
    # worktree and removing it. That cannot be prevented, only made discoverable,
    # so we say it rather than let stale worktrees pile up unnoticed.
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
