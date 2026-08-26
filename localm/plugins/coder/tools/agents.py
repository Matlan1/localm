# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ``spawn_agent`` tool: launch a focused child Agent for a sub-task.

The ``Agent`` class is imported lazily inside the call, so the tools package has
no import-time dependency on the agent module and a monkeypatched
``localm.plugins.coder.agent.Agent`` is honoured."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .base import ToolResult
from .files import tool_read_file

# Sentinel meaning no explicit scope was given, so inherit the parent's. None
# cannot serve as one: it is itself a meaningful scope value (no confinement).
_INHERIT = object()


def _inherited_confirm_handler(parent: Any):
    """The channel the CHILD asks for approval on: the parent's REAL one.

    Precedence:
    - the parent's own handler when it has one (GUI/web routes to the browser);
    - else the parent's terminal prompt when the parent is genuinely in an
      interactive loop. spawn_agent runs synchronously inside the parent's own
      tool call, so that terminal is free and the answer returns up the same
      stack;
    - else None: a genuinely unattended parent (a scheduled job, a
      non-interactive run_task) has nobody to ask, so the child keeps failing
      CLOSED. This never self-approves; it only ever reaches a real human.

    A child always runs ``run_task`` -> ``_loop(interactive=False)``, so it can
    never use execution.py's own interactive prompt branch.
    """
    handler = getattr(parent, "confirm_handler", None)
    if handler is not None:
        return handler
    if getattr(parent, "_interactive", False):
        return getattr(parent, "_confirm_tool", None)
    return None


def inherited_child_kwargs(
    parent: Any,
    *,
    backend: Any,
    cwd: Path,
    name: str,
    max_turns: int,
    confirm_handler: Any,
    scope: Any = _INHERIT,
    role: Any = None,
) -> dict:
    """The kwargs every child Agent is constructed with, in one place.

    Shared by ``spawn_agent`` and by worktree-isolated parallel dispatch, so the
    two cannot drift.

    A child must be no LESS confirmed than its parent: the parent's confirmation
    posture is inherited rather than hardcoding auto_approve=True.
    confirm_handler is a synchronous callback, so passing it through works even
    though the child runs non-interactively (run_task ->
    _loop(interactive=False)): the child calls it in the same call stack the
    parent's tool call is already on.

    A child must also be no MORE capable than its parent, along BOTH axes the
    parent is confined on:
     - restricted / disabled_tools: a restricted session cannot spawn a child
       that re-enables run_shell and the like. (spawn_agent is itself disabled
       for a restricted session, so that half is belt-and-suspenders.)
     - scope: scope and restricted are independent request fields, and
       spawn_agent is only disabled for a RESTRICTED session, so a scoped,
       non-restricted session would otherwise spawn a child with no path
       confinement at all.

    Two KINDS of argument: everything derived from *parent* is INHERITED, read
    with getattr AND a default, because *parent* is whatever the caller passed
    and a duck-typed or partial parent must fall back rather than raise.
    Everything else - backend, cwd, name, max_turns, confirm_handler, role - is
    PER-SPAWN and belongs to the individual call. *role* is an argument of
    spawn_agent, not a property of the parent, so it has no getattr form.

    This helper only ASSEMBLES kwargs and does no toolset resolution: a role's
    narrowing is applied inside ``Agent`` after ``_apply_restricted_toolset``
    and after MCP/plugin/skill registration. Pre-computing a toolset here would
    move the narrowing BEFORE dynamic registration and let MCP/plugin tools
    through.

    *scope* defaults to inheriting the parent's. A caller may pass an explicit
    value; overriding it with something broader would discard a narrowing the
    parent had. Worktree-isolated dispatch still inherits, because cwd is what
    confines the file tools there.
    """
    from ..audit import SessionMode as _SessionMode
    return dict(
        backend=backend,
        cwd=cwd,
        name=name,
        max_turns=max_turns,
        verbose=False,
        auto_approve=getattr(parent, "auto_approve", True),
        dry_run=getattr(parent, "dry_run", False),
        always_confirm=getattr(parent, "always_confirm", None),
        confirm_handler=confirm_handler,
        parent=parent,
        mode=getattr(parent, "mode", _SessionMode.PRIVACY),
        restricted=getattr(parent, "restricted", False),
        disabled_tools=getattr(parent, "disabled_tools", frozenset()),
        # A live SKILL.md allowed-tools restriction on the parent narrows the child
        # too. Passed as the ALLOWLIST so the child applies it after its own
        # MCP/plugin/skill registration, not before.
        inherited_skill_tools=(parent.active_skill_tools()
                               if hasattr(parent, "active_skill_tools") else None),
        # The parent's value AND a flag saying it was inherited. The copy is the
        # confinement floor and survives the parent reference going away; the flag
        # lets Agent.scope follow a scope TIGHTENED after this child was spawned.
        scope=(getattr(parent, "scope", None) if scope is _INHERIT else scope),
        scope_inherited=(scope is _INHERIT),
        # The role's allowlist is subtracted from the live registry on top of the
        # inherited disabled set, never in place of it, so it can only take
        # capability away from the child.
        role=role,
    )


def _prepare_child(
    cwd: Path,
    task: str,
    name: str,
    files: Optional[list],
    model: Optional[str],
    max_turns: int,
    role: Optional[str],
    parent: Any,
    *,
    confirm_handler: Any,
    tool: str,
):
    """Build the child Agent and its full task, shared by BOTH spawn paths.

    Returns ``(child, full_task)`` on success or ``(None, ToolResult)`` on a
    failure the caller should return verbatim.

    ONE construction path: every safety property a child needs - scope, role,
    restricted/disabled_tools, the confirmation posture - flows through
    ``inherited_child_kwargs``. The background variant reuses this rather than
    assembling its own kwargs.
    """
    from ..roles import resolve_role
    try:
        resolve_role(role)
    except ValueError as exc:
        return None, ToolResult.error(f"{tool}: {exc}")

    from ..agent import Agent

    backend = parent.backend
    if model and model != backend.model_id:
        from ..backends.http import make_localm_backend
        raw_url = getattr(backend, "_base_url", "http://127.0.0.1:8642/v1")
        try:
            port = int(raw_url.split(":")[-1].split("/")[0])
        except Exception:
            port = 8642
        try:
            backend = make_localm_backend(model, port=port)
        except Exception:
            backend = parent.backend

    preload_text = ""
    if files:
        failed: list[str] = []
        for fp in files:
            r = tool_read_file(cwd, fp)
            if not r.ok:
                failed.append(f"{fp}: {r.output}")
                continue
            preload_text += f"\n{r.output}\n"
        if failed:
            # Fail BEFORE spawning, rather than handing the read error to the child
            # as file content.
            return None, ToolResult.error(
                f"{tool}: could not pre-load file(s):\n  " + "\n  ".join(failed))

    full_task = task
    if preload_text:
        full_task = f"Context files:\n{preload_text}\n\nTask:\n{task}"

    child = Agent(**inherited_child_kwargs(
        parent, backend=backend, cwd=cwd, name=name, max_turns=max_turns,
        confirm_handler=confirm_handler, role=role,
    ))
    return child, full_task


def tool_spawn_agent(
    cwd: Path,
    task: str,
    name: str = "subagent",
    files: Optional[list] = None,
    model: Optional[str] = None,
    max_turns: int = 10,
    role: Optional[str] = None,
    _parent_agent: Optional[Any] = None,
) -> ToolResult:
    """
    Spawn a child Agent with a focused task.

    The child inherits the parent's backend (or uses ``model`` if given),
    gets ``files`` pre-loaded into its first user message, and runs until
    it produces a final answer or ``max_turns`` is reached.

    ``role`` selects a preset (reviewer / researcher / test-writer) that gives the
    child a focused mission AND narrows its toolset to that role's allowlist. The
    narrowing is strictly subtractive on top of everything the parent already
    forbids, so a role can only ever remove capability (see roles.py). Without a
    role the child keeps the parent's full toolset.

    Returns the child agent's final response as a string.
    """
    if _parent_agent is None:
        return ToolResult.error("spawn_agent requires a running parent agent")

    # Validation (an unknown role is an error, not a quiet full-capability child)
    # and construction both live in the shared helper, so this path and the
    # background one cannot drift.
    child, prepared = _prepare_child(
        cwd, task, name, files, model, max_turns, role, _parent_agent,
        confirm_handler=_inherited_confirm_handler(_parent_agent),
        tool="spawn_agent",
    )
    if child is None:
        return prepared
    full_task = prepared

    result_text = child.run_task(full_task)
    turns_used  = child.turns

    # Fold the child's changed files and failure trace into the parent: the child is
    # never close()d and shares this cwd. Best-effort - episodic bookkeeping never
    # breaks the tool.
    try:
        _parent_agent._absorb_child_state(child)
    except Exception:
        pass

    # Defang frame markers and chat-template control tokens in the child's summary,
    # which may quote untrusted web or MCP content verbatim, so the child cannot
    # forge a role or frame boundary in the parent. Not wrapped in the untrusted
    # fence, so the parent can still act on a legitimate delegated result; the child
    # runs its own fence internally.
    from ..provenance import neutralise
    # ASK THE CHILD whether it succeeded: run_task RETURNS its failure message
    # rather than raising, so arriving here is not success, and ToolResult.success
    # additionally CLEARS the parent's per-tool failure streak. The ToolResult stays
    # a success (the delegation ran, and the child's own text states it)
    # while the summary reports the child's verdict.
    child_ok = getattr(child, "last_run_ok", True)
    verdict = ("finished in" if child_ok else
               "DID NOT COMPLETE its task, stopping after")
    return ToolResult.success(
        neutralise(result_text),
        summary=f"sub-agent '{name}' {verdict} {turns_used} turn(s)",
    )


def _isolate_child(cwd: Path, name: str):
    """Create a git worktree for a background child, mirroring dispatch_parallel.

    Returns ``(child_cwd, info, error)``. *info* carries what the teardown and the
    parent's change-set record need; *error* is a ToolResult when isolation is
    impossible and the spawn must be refused.

    A background child MUST be isolated: it writes into the working tree while the
    parent keeps editing, which is a data race on the user's files.
    """
    import uuid

    from ..child_limit import MAX_CONCURRENT_CHILDREN  # noqa: F401  (doc anchor)
    from .git import WORKTREE_PREFIX, WORKTREE_SUBDIR, _git, git_repo_root, git_worktree_add
    from .parallel import _safe_label

    repo = git_repo_root(cwd)
    if repo is None:
        return None, None, ToolResult.error(
            "spawn_agent_background needs a git repository: the child runs in its "
            "own worktree so it cannot race your files. Use spawn_agent "
            "(synchronous) outside a repo.")

    base_sha, ok = _git(repo, "rev-parse", "HEAD")
    if not ok:
        return None, None, ToolResult.error(
            f"'{repo}' has no commits yet, so there is nothing to base an "
            "isolated worktree on. Make an initial commit first.")
    base_sha = base_sha.splitlines()[0].strip()

    run_id = uuid.uuid4().hex[:8]
    label = _safe_label(name) or "child"
    worktree = repo / WORKTREE_SUBDIR / f"{WORKTREE_PREFIX}{label}-{run_id}"
    branch = f"coder/{label}-{run_id}"
    out, ok = git_worktree_add(repo, worktree, branch, base_sha)
    if not ok:
        return None, None, ToolResult.error(
            f"spawn_agent_background: could not create an isolated worktree: {out}")

    # Mirror the parent's position WITHIN the repo, so an inherited scope glob like
    # src/** still means the same thing and the child's reach is not widened to the
    # worktree root.
    try:
        rel = cwd.resolve().relative_to(repo.resolve())
        rel_cwd = None if str(rel) == "." else rel
    except ValueError:
        rel_cwd = None
    child_cwd = worktree / rel_cwd if rel_cwd else worktree

    return child_cwd, {"repo": repo, "worktree": worktree, "branch": branch,
                       "base": base_sha, "run_id": run_id, "name": name}, None


def _finalize_isolated_child(info: dict):
    """Teardown closure: commit the child's work, capture its diff, remove its
    worktree. Runs on the JOB's worker thread; touches no parent state.

    Branch durable, worktree transient - the same contract dispatch_parallel uses.
    """
    def _finalize(child) -> dict:
        from .base import _truncate
        from .git import (
            _git, git_commit_all_in, git_prune_child_worktrees, git_worktree_remove,
        )

        worktree, repo, base = info["worktree"], info["repo"], info["base"]
        out: dict = {"branch": info["branch"], "base": base,
                     "worktree": str(worktree), "file_count": 0, "diff": ""}

        msg = f"coder background child '{info['name']}' ({info['run_id']})"
        committed, ok = git_commit_all_in(worktree, msg)
        if not ok:
            out["cleanup_warning"] = f"could not commit child work: {committed}"
        else:
            diff, dok = _git(worktree, "diff", base, "HEAD", "--stat", "-p",
                             timeout=30)
            if dok:
                out["diff"], _ = _truncate(diff)
                names, nok = _git(worktree, "diff", "--name-only", base, "HEAD",
                                  timeout=30)
                out["file_count"] = (
                    len([ln for ln in names.splitlines() if ln.strip()])
                    if nok and names != "(no output)" else 0)
            else:
                out["cleanup_warning"] = f"could not read the child's diff: {diff}"

        removed, rok = git_worktree_remove(repo, worktree)
        if not rok:
            # A failed removal is real (a dirty tree, or a process still holding
            # it): report it rather than forcing past it.
            prior = out.get("cleanup_warning")
            out["cleanup_warning"] = f"{prior}; {removed}" if prior else removed
        # Scoped prune, with its outcome reported rather than discarded. A bare git
        # worktree prune is repo-wide and would drop the record of a user's worktree
        # that merely sits on an unmounted drive.
        pruned, pok = git_prune_child_worktrees(repo)
        if not pok:
            prior = out.get("cleanup_warning")
            out["cleanup_warning"] = f"{prior}; {pruned}" if prior else pruned
        return out

    return _finalize


def tool_spawn_agent_background(
    cwd: Path,
    task: str,
    name: str = "subagent",
    files: Optional[list] = None,
    model: Optional[str] = None,
    max_turns: int = 10,
    role: Optional[str] = None,
    _parent_agent: Optional[Any] = None,
) -> ToolResult:
    """
    Start a sub-agent in the BACKGROUND and get a job id back immediately.

    Same child as ``spawn_agent`` - same inherited scope, role, restrictions and
    confirmation posture - but the parent does not wait for it. Poll with
    ``check_agent_job``; a finished job is also folded into the parent
    automatically at the start of its next turn.

    CONFIRMATION HAPPENS ONCE, AT DISPATCH. This tool is destructive, so starting
    a background sub-agent goes through the confirmation gate exactly like
    ``spawn_agent``. A backgrounded child cannot come back and ask: the single
    approval given here covers EVERYTHING the child later does - every write,
    shell command and git operation, for its whole run. When no confirmation
    channel exists at all the dispatch is refused rather than self-approved.

    A background sub-agent CANNOT be stopped once running (there is no
    cooperative cancellation in ``Agent.run_task``), so there is no kill tool for
    it.
    """
    if _parent_agent is None:
        return ToolResult.error(
            "spawn_agent_background requires a running parent agent")

    # REFUSE TO LAUNCH when there is no channel to ask on. A background child must
    # not inherit the terminal prompt (_confirm_tool): the parent may be mid-turn or
    # streaming to a browser. An explicit confirm_handler (GUI or web) is a real
    # async channel and is accepted, and auto_approve means the user has
    # pre-approved everything.
    explicit_handler = getattr(_parent_agent, "confirm_handler", None)
    if explicit_handler is None and not getattr(_parent_agent, "auto_approve", True):
        return ToolResult.error(
            "spawn_agent_background: this session confirms tools at the terminal, "
            "and a background sub-agent cannot prompt there while you are still "
            "working. Use spawn_agent (synchronous) instead, or start the session "
            "with --yes to pre-approve.")

    # The AUTHORITATIVE ceiling: one shared gate across every kind of concurrent
    # child, so background jobs and worktree-isolated parallel dispatch cannot each
    # admit their own quota. Taken BEFORE the worktree, and released when the job
    # FINISHES (see AgentJob._finish), not when it is submitted.
    from ..child_limit import describe_holders, release, try_acquire
    token = try_acquire("background", name)
    if token is None:
        return ToolResult.error(
            "spawn_agent_background: the concurrent sub-agent limit is full "
            f"({describe_holders()}). Wait for one to finish, or run this task "
            "with spawn_agent instead.")

    from ..background import AgentJob, get_registry
    from .git import git_worktree_remove
    child_cwd = info = None
    try:
        child_cwd, info, iso_error = _isolate_child(cwd, name)
        if iso_error is not None:
            release(token)
            return iso_error

        child, prepared = _prepare_child(
            child_cwd, task, name, files, model, max_turns, role, _parent_agent,
            confirm_handler=explicit_handler, tool="spawn_agent_background",
        )
        if child is None:
            git_worktree_remove(info["repo"], info["worktree"])
            release(token)
            return prepared
        full_task = prepared
        # Make the child's location introspectable, as dispatch_parallel does.
        child.worktree_path = str(info["worktree"])
        child.worktree_branch = info["branch"]

        job = get_registry().submit(
            lambda: AgentJob(child, full_task, label=name, token=token,
                             finalize=_finalize_isolated_child(info),
                             # Attributed to the PARENT's session, not the
                             # child's own agent, so the session that asked for
                             # the work is the one that can list it. The child
                             # inherits the same id anyway via Agent.__init__'s
                             # parent lookup.
                             owner=getattr(_parent_agent, "job_owner", None)),
            kind="agent")
    except Exception as exc:                          # noqa: BLE001 - reported
        # Never leak the slot OR the worktree when the submit is refused or the
        # factory raises.
        if info is not None:
            try:
                git_worktree_remove(info["repo"], info["worktree"])
            except Exception:
                pass
        release(token)
        return ToolResult.error(f"spawn_agent_background: {exc}")

    return ToolResult.success(
        f"Started sub-agent '{name}' in the background as {job.id}, isolated in "
        f"its own worktree on branch '{info['branch']}'.\n"
        f"Check it with check_agent_job('{job.id}'); its result is also folded "
        "in automatically at the start of a later turn. Its changes land on that "
        "branch and are NEVER merged into your working tree automatically.",
        summary=f"sub-agent '{name}' started in background ({job.id})",
    )


def tool_check_agent_job(cwd: Path, job_id: str) -> ToolResult:
    """
    Check a background sub-agent started by ``spawn_agent_background``.

    Safe to call repeatedly, and a pure read - it never confirms, never mutates
    the job, and never removes it, so polling after the result has already been
    folded in still works.
    """
    from ..background import get_registry

    registry = get_registry()
    job = registry.get(job_id)
    if job is None or job.kind != "agent":
        known = registry.ids(kind="agent")
        hint = ("; running or recent: " + ", ".join(known)) if known else \
               "; no background sub-agents have been started this session"
        return ToolResult.error(f"no background sub-agent '{job_id}'{hint}")

    st = job.status()
    state = st["state"]
    if state == "running":
        return ToolResult.success(
            f"sub-agent '{st['label']}' ({job_id}) is still running "
            f"({job.elapsed():.0f}s so far).",
            summary=f"{job_id}: running")

    result = st.get("result") or {}
    if st.get("error"):
        return ToolResult.success(
            f"sub-agent '{st['label']}' ({job_id}) FAILED: {st['error']}",
            summary=f"{job_id}: {state}")

    body = result.get("summary") or "(no output)"
    warn = ("\nwarnings: " + "; ".join(st["warnings"])) if st.get("warnings") else ""
    # A job reaches a terminal state whenever its thread returned, and run_task
    # RETURNS its failure message rather than raising, so a job that ended is not a
    # sub-agent that succeeded. Report the child's OWN verdict. The ToolResult stays
    # a success (the POLL worked; it is the child that did not), so a poll never
    # counts toward the tool-failure breaker.
    child_ok = result.get("ok", True)
    verdict = ("finished in" if child_ok else
               "DID NOT COMPLETE its task, stopping after")
    return ToolResult.success(
        f"sub-agent '{st['label']}' ({job_id}) {verdict} "
        f"{result.get('turns', 0)} turn(s):\n\n{body}{warn}",
        summary=f"{job_id}: {state}" if child_ok else f"{job_id}: {state} (failed)")
