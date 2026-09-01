# SPDX-License-Identifier: AGPL-3.0-or-later
"""Single tool-call execution: the security-critical dispatch path (disabled-tool
gate, scope check, dry-run, network policy, fail-closed confirmation, snapshot,
hidden-arg injection, the failure breaker), plus patch-mode capture, the
confirm prompt, scope resolution, and the per-write map refresh. Mixed into Agent."""

from __future__ import annotations

import os
import re
import shlex
import time
from pathlib import Path
from typing import Optional

import localm.plugins.coder.agent as _agent
from ..display import (
    console, print_tool_call, print_tool_error, print_tool_result,
)
from ..diffutil import (
    compute_multifile_diff, compute_search_replace_diff, compute_tool_diff,
    read_old_content, read_old_content_checked, resolve_new_content,
)
from ..confirm import invoke_confirm
from .. import shell_guard
from ..parser import ToolCall
from ..tools import ToolResult
from ..audit import SessionMode
from .constants import (
    _CODE_EXTS, _GLOBAL_ERROR_ABORT, _MAX_SHELL_SCOPE_FLAGS,
    _MCP_SCOPE_PATH_ARGS, _MUTATING_TOOLS, _NETWORK_TOOLS, _PARENT_AGENT_TOOLS,
    _PATCH_MODE_ELIGIBLE_TOOLS, _PROJECT_MAP_TOOLS, _SCOPE_PATH_ARGS,
    _SCOPED_TOOLS, _SHELL_COMMAND_ARGS, _SHELL_DECLARED_PATH_ARGS,
    _SHELL_EXEC_TOOLS, _SKILL_STATE_TOOLS, _SHELL_UNSCOPED_TOOLS,
    _TEST_COMMAND_MARKERS, _TODO_TOOLS, _UNDOABLE_TOOLS, _call_target_paths,
)
from .scope import _scope_pattern


def _looks_like_drive_path(tok: str) -> bool:
    """True for a Windows drive-qualified token: ``C:``, ``C:\\x``, ``d:/x``.

    The drive letter must be a single ASCII letter followed by a separator or the
    end of the token, so ``5:30`` and ``4:3`` (an ffmpeg offset, an aspect ratio)
    and ``s:old:new:`` (a sed delimiter) are not paths. ``C:foo``, drive-relative
    with no separator, is not recognised: it cannot be told apart from an
    ordinary key:value argument.

    A real drive-qualified path carries exactly ONE colon, so requiring that also
    rejects ``s:/usr/local:/opt:g``, the common sed form.
    """
    if len(tok) < 2 or tok[1] != ":" or tok.count(":") != 1:
        return False
    if not (tok[0].isascii() and tok[0].isalpha()):
        return False
    return len(tok) == 2 or tok[2] in "/\\"


def _is_path_like(tok: str) -> bool:
    """True when a shell token is a path by SYNTAX alone, with no filesystem look.

    Absolute, drive-qualified (``C:\\x``, bare ``E:``), explicitly relative
    (``./``, ``../``, ``~/``), or carrying a ``/../`` segment. Everything else is
    treated as not-a-path, and this never falls back to asking the filesystem -
    see ``_shell_paths_outside_scope``.
    """
    norm = tok.replace("\\", "/")
    return (norm.startswith(("./", "../", "~/", "/"))
            or "/../" in norm
            or _looks_like_drive_path(tok))


class _ExecutionMixin:
    def _scope_rel(self, value: str) -> Optional[str]:
        """
        Resolve a path/glob arg to a cwd-relative POSIX string for scope
        matching, or return None if it escapes cwd.

        Relative paths are joined onto cwd; absolute paths are accepted only
        when they live inside cwd, so an in-cwd absolute path that matches the
        scope passes. Glob metacharacters in *value* (e.g. ``**/*.py`` for
        grep/search_replace) survive resolution: they are kept verbatim in the
        relative string and matched against the scope as-is.

        The VALUE is never touched on disk, not even to refuse it. A stat is an
        access, and at the access point a legitimate gate-check, a command gone
        wrong and a live injection attempt are indistinguishable, so the gate
        does not have the capability at all. See :meth:`_scope_rel_lexical`.

        The cwd ANCHOR does resolve: it is not a model-supplied value but the
        owner's own working directory, which this process is already running in.

        With R = ``self.cwd.resolve()``, a path is allowed only when R prefixes
        it, so an absolute path that is lexically OUTSIDE cwd but reaches INSIDE
        through a symlink reads as outside. The escape direction (a path
        lexically INSIDE cwd that symlinks OUT) satisfies the ``relative_to``
        here and is caught by ``tools/base.py::_confine``, which resolves at
        actual execution time. Every Agent construction site passes an
        already-resolved cwd, so cwd itself is never symlinked here.
        """
        raw = str(value).replace("\\", "/")
        p = Path(raw)
        cwd = self.cwd.resolve()
        if p.is_absolute():
            try:
                # No resolve(): the path may contain glob chars or not exist, and
                # resolving it would STAT what this gate exists to refuse.
                rel = Path(raw).relative_to(cwd)
            except ValueError:
                return None   # outside cwd, and NOT re-checked via resolve()
            return rel.as_posix()
        # Relative: collapse any leading ./ and reject cwd escapes (../).
        rel_posix = (Path(".") / raw).as_posix()
        if rel_posix.startswith("./"):
            rel_posix = rel_posix[2:]
        parts = [seg for seg in rel_posix.split("/") if seg not in ("", ".")]
        if ".." in parts:
            return None   # escapes cwd
        return "/".join(parts)

    def _scope_allows(self, value: str) -> bool:
        """True if *value* (a path or glob arg) is within the active scope."""
        rel = self._scope_rel(value)
        if rel is None:
            return False
        return _scope_pattern(self.scope).match(rel) is not None

    def _scope_rel_lexical(self, value: str) -> Optional[str]:
        """Filesystem-free twin of :meth:`_scope_rel`, for the shell WARNING only.

        Same contract (cwd-relative POSIX string, or None if it escapes cwd).
        Neither this nor :meth:`_scope_rel` resolves the model-supplied VALUE.
        The one difference is the cwd ANCHOR.

        ``os.path.abspath`` rather than ``resolve()`` for cwd here: it normalises
        and anchors without a symlink lookup, so no part of this call stats
        ANYTHING at all, which is what a warning needs - it runs over a command
        the model has merely PROPOSED, before any confirmation and before
        anything executes. :meth:`_scope_rel` keeps ``resolve()`` for its anchor,
        so its refusal set does not shift in the cwd-is-symlinked case.

        A path that reaches cwd only through a symlink reads as outside, so this
        can over-report a link and never misses a real escape.
        """
        raw = str(value).replace("\\", "/")
        p = Path(raw)
        cwd = Path(os.path.abspath(self.cwd))
        if p.is_absolute():
            try:
                return p.relative_to(cwd).as_posix()
            except ValueError:
                return None   # outside cwd, and NOT re-checked via resolve()
        # Relative: collapse any leading ./ and reject cwd escapes (../).
        rel_posix = (Path(".") / raw).as_posix()
        if rel_posix.startswith("./"):
            rel_posix = rel_posix[2:]
        parts = [seg for seg in rel_posix.split("/") if seg not in ("", ".")]
        if ".." in parts:
            return None   # escapes cwd
        return "/".join(parts)

    def _scope_allows_lexical(self, value: str) -> bool:
        """True if *value* is within the active scope, decided without any
        filesystem access. See :meth:`_scope_rel_lexical`."""
        rel = self._scope_rel_lexical(value)
        if rel is None:
            return False
        return _scope_pattern(self.scope).match(rel) is not None

    def _scope_violation(self, call: ToolCall) -> Optional[str]:
        """
        Return the first in-scope-checked arg value that falls outside the
        active scope, or None if the call is allowed.

        Defaults to checking the ``path`` arg; ``_SCOPE_PATH_ARGS`` overrides
        this for tools whose primary target is a ``glob`` or ``output_path``
        arg (and may add ``path`` alongside it).
        """
        # MCP and PLUGIN tools are registered dynamically with unknown arg schemas,
        # so an owner's --scope confines their file ops via a broad set of common
        # path-arg names. Best-effort: an unusual path-arg name is not caught.
        if call.name.startswith(("mcp_", "plugin_")):
            arg_names = _MCP_SCOPE_PATH_ARGS
        else:
            # A nested-path tool (edit_files) has NO top-level `path` arg, so
            # checking arg names alone would find nothing and let the call
            # through. Check its real targets first.
            for value in _call_target_paths(call.name, call.args):
                if not self._scope_allows(value):
                    return value
            arg_names = _SCOPE_PATH_ARGS.get(call.name, ("path",))
        for name in arg_names:
            value = call.args.get(name)
            if value:
                if not self._scope_allows(str(value)):
                    return str(value)
        return None

    def _shell_paths_outside_scope(self, call: ToolCall) -> list[str]:
        """Best-effort: the path-like tokens of a shell call that fall outside the
        active scope. Empty when nothing suspicious was found OR when the check
        simply could not tell - it is a heuristic, never a gate.

        PURELY LEXICAL: a model-supplied token is never stat-ed, resolved, or
        ``.exists()``-ed. This runs over a command the model has merely PROPOSED,
        before any confirmation and before anything executes, so a filesystem
        probe here would reach out of the workspace on the model's say so alone.
        A legitimate probe, a command gone wrong, and a live injection attempt
        are indistinguishable at the access point.

        A token counts as path-like only by SYNTAX (``_is_path_like``). An arg
        the tool's own schema DECLARES to be a path (run_tests' ``path``) needs
        no heuristic and is checked whole, spaces and all.

        A bare-relative path that merely exists (``cat secrets.txt``, ``git -C
        docs``) is therefore not flagged. What is flagged is the case with no
        false positives: a command reaching OUT of the workspace. This warning
        blocks nothing; hard confinement (the disabled-tool gate, and
        ``_scope_violation`` for file tools) is enforced elsewhere.
        """
        flagged: list[str] = []
        for arg in _SHELL_COMMAND_ARGS.get(call.name, ()):
            raw = call.args.get(arg)
            if not raw:
                continue
            text = str(raw)
            declared = arg in _SHELL_DECLARED_PATH_ARGS.get(call.name, ())
            if declared:
                # A path by contract: checked whole rather than tokenised, since a
                # path may hold spaces.
                tokens = [text]
            else:
                try:
                    # posix=False keeps Windows backslashes intact; quotes survive
                    # as part of the token and are stripped below.
                    #
                    # Not tools/shell.py:_split_command, which is what actually
                    # EXECUTES the command. posix=False honours a quote only where
                    # one opens a token, so a quoted path containing spaces arrives
                    # here as fragments. The warning still fires on it - every
                    # fragment is checked - it can just name a fragment. This is a
                    # lexical best-effort warning that blocks nothing.
                    tokens = shlex.split(text, posix=False)
                except ValueError:
                    # Unbalanced quotes: fall back to whitespace splitting rather
                    # than skipping the check entirely.
                    tokens = text.split()
            for tok in tokens:
                if not declared:
                    tok = tok.strip("'\"")
                if not tok or tok in flagged:
                    continue
                if not declared:
                    if tok.startswith("-"):
                        continue       # a flag, never a path
                    if not _is_path_like(tok):
                        continue       # not a path by syntax, and we do not ask disk
                try:
                    if not self._scope_allows_lexical(tok):
                        flagged.append(tok)
                except Exception:
                    continue       # best-effort: an unparseable token is not a finding
        return flagged

    def _warn_shell_outside_scope(self, call: ToolCall) -> None:
        """Warn (never block) when a shell command references paths outside the
        active scope. The scope glob cannot confine a process, so this is the only
        signal the user gets that the command they are about to see run is not
        bounded by the scope they set."""
        print_warning = _agent.print_warning  # live: honour a patched agent.print_warning
        flagged = self._shell_paths_outside_scope(call)
        if not flagged:
            return
        shown = flagged[:_MAX_SHELL_SCOPE_FLAGS]
        more = len(flagged) - len(shown)
        tail = f" (and {more} more)" if more > 0 else ""
        msg = (
            f"{call.name}: this command references {', '.join(shown)}{tail}, "
            f"outside the active scope '{self.scope}'. Shell execution is NOT "
            "confined by the scope, so it will run anyway. Best-effort check."
        )
        print_warning(msg)
        self._emit("info", text=msg)
        self._audit.notice("scope_shell_path", msg)

    def _shell_guard_verdict(self, call: ToolCall):
        """Classify a shell call against the reject-list in shell_guard.

        Returns ``(refusal, unchecked)``: the first matching refusal, and
        whether the check FAILED to run. A check that raises returns
        ``(None, True)``, and the caller must then require confirmation instead
        of executing, so a command the gate never inspected is never
        auto-approved. The failure is warned, emitted and audited, never
        silenced. See test_a_classifier_failure_denies_rather_than_allows.
        """
        print_warning = _agent.print_warning
        for arg_name in _SHELL_COMMAND_ARGS.get(call.name, ("command",)):
            command = str(call.args.get(arg_name) or "")
            if not command.strip():
                continue
            try:
                refusal = shell_guard.classify(command, self.cwd)
            except Exception as exc:
                msg = (f"{call.name}: the shell safety check could not run "
                       f"({type(exc).__name__}: {exc}), so this command now "
                       "requires confirmation before it can run.")
                print_warning(msg)
                self._emit("info", text=msg)
                self._audit.notice("shell_guard_error", msg)
                return None, True
            if refusal is not None:
                return refusal, False
        return None, False

    def _execute_tool(self, call: ToolCall, interactive: bool) -> ToolResult:
        TOOL_REGISTRY = _agent.TOOL_REGISTRY  # live: honour a patched agent.TOOL_REGISTRY
        # Hard gate: a tool disabled for this session (e.g. run_shell for a
        # restricted, shareable coder key) can never execute, whatever the model
        # emits. This is the security boundary; the prompt/parse exclusions below
        # only stop the model wasting turns.
        if call.name in self.disabled_tools:
            result = ToolResult.error(
                f"'{call.name}' is disabled for this session and was not run.")
            if interactive:
                print_tool_error(call.name, result.output)
            return result

        # Second hard gate: a skill loaded with use_skill restricts this turn to
        # its declared allowed-tools (core._skill_gate_denial). Sited AFTER
        # disabled_tools and never instead of it, so the two compose as an
        # INTERSECTION - a skill can only subtract, never hand back a tool the
        # operator disabled. No "tool_result" event is emitted here: no
        # "tool_call" event has been emitted yet (that happens after the registry
        # lookup), so a result would have no call to attach to. The audit notice
        # carries the refusal instead.
        skill_denial = self._skill_gate_denial(call.name)
        if skill_denial is not None:
            result = ToolResult.error(skill_denial)
            self._audit.notice("skill_tool_denied", skill_denial)
            if interactive:
                print_tool_error(call.name, result.output)
            return result

        tool_def = TOOL_REGISTRY.get(call.name)

        if tool_def is None:
            result = ToolResult.error(
                f"Unknown tool '{call.name}'. "
                f"Available: {', '.join(TOOL_REGISTRY)}"
            )
            if interactive:
                print_tool_error(call.name, result.output)
            return result

        self._audit.tool_call(call.name, call.args)
        if interactive:
            print_tool_call(call.name, call.args)
        self._emit("tool_call", tool=call.name, args=call.args)

        # Patch-mode: intercept write tools, accumulate diffs, do not touch disk.
        # A write tool the interceptor cannot express as a diff must NOT fall
        # through to a real disk write. search_replace is eligible too
        # (_PATCH_MODE_ELIGIBLE_TOOLS), via its own dry_run rather than the
        # pre-call snapshot _UNDOABLE_TOOLS uses.
        if self.patch_mode and call.name in _PATCH_MODE_ELIGIBLE_TOOLS:
            t_start = time.monotonic()
            chunk = self._patch_mode_intercept(call)
            duration_s = time.monotonic() - t_start
            if chunk is not None:
                self._patch_chunks.append(chunk)
                # Name the files the DIFF covers, not every path the call
                # mentioned: a file whose edit produced no change is not captured.
                # search_replace has no pre-known targets (its files are discovered
                # by the sweep, not named in the call args), so they are read off
                # the diff's own "+++ b/<path>" headers.
                if call.name == "search_replace":
                    covered = re.findall(r"^\+\+\+ b/(.+)$", chunk, re.MULTILINE)
                else:
                    targets = list(dict.fromkeys(_call_target_paths(call.name, call.args)))
                    covered = [t for t in targets if f"b/{t}" in chunk] or targets
                label = ", ".join(covered) if covered else "?"
                result = ToolResult.success(
                    f"[patch-mode] diff captured for {label}",
                    summary=f"[patch-mode] {call.name}",
                )
                if interactive:
                    console.print("    [dim cyan][patch-mode] diff captured[/dim cyan]")
            else:
                result = ToolResult.error(
                    f"[patch-mode] {call.name} cannot be captured as a diff "
                    "(no change, or unsupported operation) - skipped. Use "
                    "write_file/edit_file/patch_file in patch mode."
                )
                if interactive:
                    console.print("    [dim yellow][patch-mode] skipped[/dim yellow]")
            # A real duration: unlike the never-ran paths below,
            # _patch_mode_intercept does the diffing work, so the elapsed time is
            # a fact about this call.
            self._emit("tool_result", tool=call.name, ok=result.ok,
                       summary=result.summary, output=result.output[:4000],
                       duration_s=duration_s)
            return result

        # Scope check - reject file operations that fall outside the active glob.
        # MCP (mcp_*) AND plugin (plugin_*) tools are included so an owner's --scope
        # confines those dynamically-registered file tools too: built-in file tools
        # are default-denied at authoring time (_SCOPED_TOOLS), while the dynamic
        # families are not in the registry then and are gated here by prefix.
        if self.scope and (call.name in _SCOPED_TOOLS
                           or call.name.startswith(("mcp_", "plugin_"))):
            offending = self._scope_violation(call)
            if offending is not None:
                result = ToolResult.error(
                    f"'{offending}' is outside the active scope '{self.scope}'. "
                    "Only files matching this glob pattern can be accessed."
                )
                if interactive:
                    print_tool_error(call.name, result.output)
                # No duration_s: never reached tool_def.fn.
                self._emit("tool_result", tool=call.name, ok=False,
                           summary=f"outside scope '{self.scope}'")
                return result

        # The shell tools are NOT in _SCOPED_TOOLS: a path-arg check cannot confine
        # a process. Flag the gap at runtime instead - a warning, never a block.
        if self.scope and call.name in _SHELL_UNSCOPED_TOOLS:
            self._warn_shell_outside_scope(call)

        # Shell reject-list. Runs for every shell call, ahead of dry-run,
        # network policy and confirmation, and is not affected by auto_approve,
        # always_confirm, lenient or the confirm handler. An unchecked call
        # falls through to needs_confirm below.
        # See test_a_dangerous_command_is_blocked_under_auto_approve.
        shell_unchecked = False
        if call.name in _SHELL_EXEC_TOOLS:
            refusal, shell_unchecked = self._shell_guard_verdict(call)
            if refusal is not None:
                result = ToolResult.error(refusal.message())
                self._audit.notice("shell_guard_blocked", refusal.message())
                if interactive:
                    print_tool_error(call.name, result.output)
                # No duration_s: the command never ran.
                self._emit("tool_result", tool=call.name, ok=False,
                           summary=f"blocked by the shell safety gate "
                                   f"({refusal.rule})")
                return result

        # Dry-run: show destructive calls but don't execute them
        if self.dry_run and tool_def.destructive:
            result = ToolResult.success(
                f"[dry-run] {call.name} - skipped",
                summary=f"[dry-run] {call.name}",
            )
            if interactive:
                console.print("    [dim yellow][dry-run] skipped[/dim yellow]")
            # No duration_s: skipped, never reached tool_def.fn.
            self._emit("tool_result", tool=call.name, ok=True,
                       summary=result.summary)
            return result

        # Network policy: model-initiated network tools are governed by
        # net_mode (off = fail fast, ask = approval flow, allow = run).
        net_mode = None
        if call.name in _NETWORK_TOOLS:
            from localm.netpolicy import network_mode
            net_mode = network_mode()
            if net_mode == "off":
                result = ToolResult.error(
                    "Network access is disabled (net_mode=off). The user can "
                    "enable it with: localm config net_mode ask"
                )
                if interactive:
                    print_tool_error(call.name, result.output)
                # No duration_s: the tool never ran, which is a different fact from
                # "ran and took 0.0s". The client renders nothing for an absent
                # field and a real "0.0s" for a present zero.
                self._emit("tool_result", tool=call.name, ok=False,
                           summary="blocked by network policy (net_mode=off)")
                return result

        # Confirmation for destructive tools (diff preview for write_file), for
        # network tools when net_mode is "ask", and for a non-mutating tool that
        # opts in via ask_by_default (see ToolDef.ask_by_default, registry.py).
        #
        # call.lenient forces confirmation too, same as always_confirm: a call
        # recovered only via parser.py's name-gated fallback (a bare JSON object
        # or an unlabelled fence) never engaged the lazy-grammar trigger, so
        # nothing constrained its content. auto_approve does not skip the human
        # look for such a call on a destructive/network/ask-by-default tool.
        # See ToolCall.lenient.
        #
        # isinstance-gated rather than a bare call.lenient or getattr(call,
        # "lenient", False): a ToolCall-shaped MagicMock auto-vivifies any
        # attribute access as a fresh, TRUTHY MagicMock, which would demand
        # confirmation for every such double. Only a genuine ToolCall's lenient
        # field (dataclass default False) is trusted; anything else counts as
        # non-lenient.
        needs_confirm = shell_unchecked or (
            (tool_def.destructive or net_mode == "ask" or tool_def.ask_by_default) and (
                not self.auto_approve or call.name in self.always_confirm
                or (isinstance(call, ToolCall) and call.lenient)
            )
        )
        if needs_confirm:
            if self.confirm_handler is not None:
                approved = invoke_confirm(self.confirm_handler, call,
                                          agent=self._confirm_agent_label())
            elif interactive:
                approved = self._confirm_tool(call)
            else:
                # Fail CLOSED: this tool requires confirmation and there is no way
                # to obtain it (non-interactive run, no approval handler), so deny
                # it rather than execute. A safety gate that cannot run is not
                # treated as passed, so a configured always_confirm or
                # auto_approve=off is honoured and an unattended run cannot be
                # steered into an unconfirmed destructive or network action.
                result = ToolResult.error(
                    f"{call.name} requires confirmation, but this run is "
                    "non-interactive with no approval handler - denied. Run "
                    "interactively, or use the restricted coder for unattended runs.")
                self._emit("tool_result", tool=call.name, ok=False,
                           summary="denied: confirmation required, none available")
                return result
            if not approved:
                result = ToolResult.error("Rejected by user.")
                if interactive:
                    print_tool_result(call.name, result, verbose=False)
                self._emit("tool_result", tool=call.name, ok=False,
                           summary="rejected by user")
                return result

        # Snapshot file content before undoable writes so /undo can restore it and
        # the changed-files tracker can diff against the original. A list, because
        # edit_files targets several files in one call and needs one undo entry
        # per file.
        snapshots: dict[str, bytes | None] = {}
        if call.name in _UNDOABLE_TOOLS:
            # One id per CALL: a multi-file call pushes several entries and undo()
            # reverts them together (see persistence.undo), so /undo never leaves a
            # batch half-restored.
            self._undo_seq = getattr(self, "_undo_seq", 0) + 1
            undo_call_id = self._undo_seq
            for path_arg in dict.fromkeys(_call_target_paths(call.name, call.args)):
                abs_path = (self.cwd / path_arg).resolve()
                try:
                    old_bytes = abs_path.read_bytes() if abs_path.is_file() else None
                except Exception:
                    old_bytes = None
                snapshots[path_arg] = old_bytes
                self._undo_stack.append({
                    "path": abs_path,
                    "old_content": old_bytes,
                    "tool": call.name,
                    "call_id": undo_call_id,
                })

        # Episodic change-detection baseline: snapshot the git work-tree state just
        # before the FIRST run_shell, pre-execution, so a mutating shell command's
        # writes (git apply, a formatter, codegen) can be attributed to this session
        # at close - the write-tool tracker never sees them. Captured before the
        # shell runs so the shell's own changes are the delta rather than part of
        # the baseline. Episodic sessions only.
        if (call.name in _SHELL_EXEC_TOOLS and self._episodic
                and not self._shell_baseline_captured):
            self._shell_baseline_captured = True
            self._git_baseline = self._git_status_paths()

        # Inject hidden runtime args into specific tools, AFTER the copy so a
        # model-supplied "_parent_agent" cannot win.
        args = dict(call.args)
        if call.name in _PARENT_AGENT_TOOLS:
            args["_parent_agent"] = self
        # The task-list tools operate on THIS session's state (tools/tasks.py),
        # use_skill arms this session's active-skill restriction (skills.py), and
        # find_references reads this session's live ProjectMap (tools/references.py).
        # Injected after the copy, so a model-supplied "_session" cannot win and
        # choose its own restriction.
        if call.name in _TODO_TOOLS or call.name in _SKILL_STATE_TOOLS \
                or call.name in _PROJECT_MAP_TOOLS:
            args["_session"] = self
        if call.name in (*_SHELL_EXEC_TOOLS, "fetch_url", "web_search", "generate_image") \
                and self.mode == SessionMode.PRIVACY:
            args["_privacy"] = True
        # Which session a background job belongs to. Injected after the copy, so a
        # model-supplied "_owner" cannot attribute a job to another session. The
        # sibling spawn_agent_background needs no injection - it already receives
        # _parent_agent and reads job_owner off it.
        if call.name == "run_shell_background":
            args["_owner"] = self.job_owner

        # Timed around the invocation ONLY, not the bookkeeping below, so this is
        # how long the tool took - the number the GUI shows next to the card.
        t_start = time.monotonic()
        try:
            result = tool_def.fn(self.cwd, **args)
        except TypeError as e:
            result = ToolResult.error(f"Bad arguments for {call.name}: {e}")
        except Exception as e:
            result = ToolResult.error(f"Tool error: {e}")
        duration_s = time.monotonic() - t_start

        result = self._track_tool_failure(call, result)

        self._audit.tool_result(call.name, result.ok, result.summary)
        if interactive:
            print_tool_result(call.name, result, verbose=self.verbose)
        self._emit("tool_result", tool=call.name, ok=result.ok,
                   summary=result.summary, output=result.output[:4000],
                   duration_s=duration_s)

        # Incremental map refresh after file-mutating tools
        if result.ok and call.name in _MUTATING_TOOLS:
            self._refresh_map_for_tool(call, result)

        self._post_tool_success(call, result, snapshots)

        return result

    def _track_tool_failure(self, call: ToolCall, result: ToolResult) -> ToolResult:
        """Update the per-tool + global failure streaks and arm the circuit
        breakers; on a failure, fold escalating recovery hints into the result.
        Returns the (possibly augmented) result. Split out of _execute_tool; the
        breaker flags it sets are checked back in _loop."""
        # Track consecutive failures and inject escalating recovery hints;
        # at 4 identical failures the circuit breaker stops the task after
        # this batch (checked in _loop).
        if not result.ok:
            # Record the ORIGINAL error, before the hint augmentation below, into
            # the bounded session error trace that feeds the close-time episode
            # reflection's what_failed.
            self._record_error(call.name, result.output)
            streak = self._consecutive_errors.get(call.name, 0) + 1
            self._consecutive_errors[call.name] = streak
            if streak == 2:
                result = ToolResult.error(
                    result.output
                    + "\n\n[Hint: this tool has failed twice in a row. "
                    "Try a different approach - check paths, arguments, or preconditions.]"
                )
            elif streak >= 3:
                result = ToolResult.error(
                    result.output
                    + f"\n\n[Warning: {call.name} has failed {streak} times consecutively. "
                    "Step back and reconsider your strategy. "
                    "Consider reading the relevant files first, "
                    "or breaking the task into smaller steps.]"
                )
            if streak >= 4:
                self._abort_streak_tool = call.name
            # Global no-progress breaker: count failures across ANY tool so a model
            # spinning on varied failing calls cannot burn the whole budget.
            self._global_error_streak += 1
            if self._global_error_streak >= _GLOBAL_ERROR_ABORT:
                self._abort_no_progress = True
        else:
            self._consecutive_errors.pop(call.name, None)
            self._global_error_streak = 0
        return result

    def _post_tool_success(self, call: ToolCall, result: ToolResult,
                           snapshots: "dict[str, bytes | None]") -> None:
        """Post-success bookkeeping split out of _execute_tool: record changed
        code files for the changed-files tracker and clear the unverified-writes
        set when the agent runs the test suite (or a test command).

        *snapshots* maps each path the call targeted to its pre-call bytes (a
        multi-file tool such as edit_files supplies several), so every file it
        wrote is tracked, not just the first."""
        # Self-verification bookkeeping: remember code files changed on disk,
        # forget them once the agent runs the test suite (or a test command)
        if result.ok and not self.dry_run and not self.patch_mode:
            if call.name in _UNDOABLE_TOOLS:
                for path_arg, snapshot_old in snapshots.items():
                    self._record_changed_file(path_arg, snapshot_old, call.name)
                    if Path(path_arg).suffix.lower() in _CODE_EXTS:
                        self._unverified_writes.add(path_arg)
            elif call.name == "run_tests":
                self._unverified_writes.clear()
            elif call.name == "run_shell":
                cmd = str(call.args.get("command", "")).lower()
                if any(marker in cmd for marker in _TEST_COMMAND_MARKERS):
                    self._unverified_writes.clear()

        # search_replace: tracked from its OWN result.changes (populated by the
        # tool once it knows which files the sweep touched), not the result.ok-gated
        # branch above. A PARTIAL apply reports ok=False, but the files written
        # before the failure are real mutations on disk and are still tracked and
        # undoable - result.changes carries exactly those, whether the call
        # succeeded or partially failed. Guarded on the TOOL'S OWN dry_run arg
        # (distinct from self.dry_run / self.patch_mode, already excluded above and
        # by the early-return interceptors), because a model-requested preview
        # populates the same field with data that was never written.
        if (not self.dry_run and not self.patch_mode
                and call.name == "search_replace"
                and not call.args.get("dry_run") and result.changes):
            self._undo_seq = getattr(self, "_undo_seq", 0) + 1
            undo_call_id = self._undo_seq
            for rel, old_bytes, _new_text in result.changes:
                self._record_changed_file(rel, old_bytes, call.name)
                self._undo_stack.append({
                    "path": (self.cwd / rel).resolve(),
                    "old_content": old_bytes,
                    "tool": call.name,
                    "call_id": undo_call_id,
                })
                if Path(rel).suffix.lower() in _CODE_EXTS:
                    self._unverified_writes.add(rel)

    def _patch_mode_intercept(self, call: ToolCall) -> Optional[str]:
        """
        Compute a unified diff for a write/edit/patch call without touching disk.

        Returns the diff string, or None if the diff cannot be computed.
        """
        # edit_files spans several files, so it has no single old_content: without
        # this branch compute_tool_diff returns None and the intercept reports
        # "cannot be captured".
        if call.name == "edit_files":
            return compute_multifile_diff(self.cwd, call.args.get("edits"))
        # search_replace's targets are a glob + regex sweep, not a `path` arg, so
        # there is no old_content to read ahead of time. It gets its own helper,
        # which runs the real sweep via dry_run rather than touching disk.
        if call.name == "search_replace":
            return compute_search_replace_diff(
                self.cwd, call.args.get("pattern", ""),
                call.args.get("replacement", ""),
                call.args.get("glob", "**/*"),
            )
        path_arg = call.args.get("path", "")
        old_text = read_old_content(self.cwd, path_arg)
        return compute_tool_diff(call.name, call.args, old_text)

    def current_patch(self) -> str:
        """The accumulated unified diff so far, WITHOUT clearing the buffer.

        For readers that only want to LOOK: a GUI "show me the patch" request, a
        status poll, a preview. :meth:`flush_patch` clears the buffer.
        """
        return "\n".join(c for c in self._patch_chunks if c)

    def has_patch(self) -> bool:
        """Is there anything in the patch buffer? Cheap, and exactly equivalent
        to ``bool(self.current_patch())`` without joining the diff.

        ``CoderSession.info()`` needs only the boolean and is called once per
        session by the session-list route.
        """
        return any(c for c in self._patch_chunks)

    def flush_patch(self, output_path: Optional[Path] = None) -> str:
        """
        Return the accumulated unified diff (and optionally write it to a file).

        Clears the internal patch buffer.
        """
        content = self.current_patch()
        self._patch_chunks.clear()
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
        return content

    def _confirm_agent_label(self) -> Optional[str]:
        """Who is asking, for a confirmation prompt: this sub-agent's name, or None.

        ``parent`` is set only when this Agent IS a sub-agent, so it - not the name,
        which every Agent has - is what distinguishes "a child is asking on the
        human's shared confirmation channel" from "the session the human started is
        asking for itself". The top-level agent gets None.

        This is the ONE place a child's identity enters the confirm chain, so every
        delegation path (worktree-isolated parallel dispatch, spawn_agent, background
        sub-agents) is attributed by construction.

        A child with a falsy name still gets a label: ``spawn_agent``'s ``name``
        comes straight from the model's tool-call arguments, and an empty one would
        collapse to "no label", making a delegated request look like the human's own.
        """
        if self.parent is None:
            return None
        return self.name or "sub-agent"

    def _confirm_tool(self, call: ToolCall) -> bool:
        """
        Ask the user to approve a destructive tool call.

        For *write_file*, shows a coloured unified diff of the proposed change
        before the prompt so the user can see exactly what will happen.
        For all other destructive tools, falls back to a plain y/N prompt.
        """
        # Live attribute lookup, so a patched agent.confirm / confirm_diff /
        # print_diff_preview is honoured.
        confirm = _agent.confirm
        confirm_diff = _agent.confirm_diff
        print_diff_preview = _agent.print_diff_preview
        if call.name in ("write_file", "edit_file"):
            path_arg = call.args.get("path", "")
            old_content, readable = read_old_content_checked(self.cwd, path_arg)
            if not readable:
                # A consent surface: old_content is "" because the file could not be
                # READ, not because it is new, and print_diff_preview renders "" as
                # "does not exist yet", showing the whole write as an addition and
                # nothing as deleted. Say so, so the user is not approving an
                # overwrite whose destructive half is invisible.
                from ..display import console as _con
                _con.print(
                    f"[yellow]![/yellow] Could not read the current contents of "
                    f"{path_arg or 'the file'}. The diff below CANNOT show what "
                    f"would be replaced - treat it as an overwrite of unknown "
                    f"content.")
            new_content = resolve_new_content(call.name, call.args, old_content)
            print_diff_preview(old_content, new_content, path_label=path_arg)
            return confirm_diff(path_arg or "file")

        # Show the same kind of diff for a multi-file edit, concatenated over
        # every file the call would touch.
        if call.name == "edit_files":
            targets = dict.fromkeys(_call_target_paths(call.name, call.args))
            label = ", ".join(targets) if targets else "files"
            diff = compute_multifile_diff(self.cwd, call.args.get("edits"))
            if diff:
                from ..display import console as _con
                from rich.syntax import Syntax
                _con.print()
                _con.print(Syntax(diff, "diff", theme="monokai", line_numbers=False))
            return confirm_diff(label)

        if call.name == "patch_file":
            path_arg = call.args.get("path", "")
            patch    = call.args.get("diff", "")
            # The patch is already a unified diff - display it directly
            from ..display import console as _con
            from rich.syntax import Syntax
            _con.print()
            _con.print(Syntax(patch, "diff", theme="monokai", line_numbers=False))
            return confirm_diff(path_arg or "file")

        return confirm(f"  Allow {call.name}?")

    def _refresh_map_for_tool(self, call: ToolCall,
                              result: "ToolResult | None" = None) -> None:
        """Update the project map for files touched by a write/edit tool call.

        *result* is optional (existing callers - and the test that drives
        this directly - omit it) and consulted only for a tool whose targets
        are not knowable from the call args alone: search_replace's paths
        come from its own glob+regex sweep, reported post-call via
        ToolResult.changes, not from a `path`-shaped arg _call_target_paths()
        could resolve ahead of time.

        run_shell is a step further: it has no `path`-shaped arg AT ALL (only
        a free-form `command` string), so there is no path to resolve even
        post-call. It marks the whole map dirty and returns, without calling
        _rebuild_system_prompt() as every other branch below does. That rebuild
        is what reconciles the map (see ProjectMap._rescan_if_dirty via
        context._build_messages, called once per turn), so it runs once before
        the map is next read rather than once per run_shell call.
        """
        if call.name == "run_shell":
            self._project_map.mark_dirty()
            return
        paths = dict.fromkeys(_call_target_paths(call.name, call.args))
        if result is not None and result.changes:
            for rel, _old, _new in result.changes:
                paths.setdefault(rel, None)
        if paths:
            for path_arg in paths:
                abs_path = (self.cwd / path_arg).resolve()
                self._project_map.refresh_file(abs_path)
            # Regenerate the system prompt with the updated map (combined mcp+
            # plugin+skill tool docs preserved - see _rebuild_system_prompt).
            # Once, after ALL the call's files, not once per file.
            self._rebuild_system_prompt()
