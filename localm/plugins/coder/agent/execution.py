# SPDX-License-Identifier: AGPL-3.0-or-later
"""Single tool-call execution: the security-critical dispatch path (disabled-tool
gate, scope check, dry-run, network policy, fail-closed confirmation, snapshot,
hidden-arg injection, the failure breaker), plus patch-mode capture, the
confirm prompt, scope resolution, and the per-write map refresh. Mixed into Agent."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Optional

import localm.plugins.coder.agent as _agent
from ..display import (
    console, print_tool_call, print_tool_error, print_tool_result,
)
from ..diffutil import (
    compute_multifile_diff, compute_tool_diff, read_old_content,
    resolve_new_content,
)
from ..confirm import invoke_confirm
from ..parser import ToolCall
from ..tools import ToolResult
from ..audit import SessionMode
from .constants import (
    _CODE_EXTS, _GLOBAL_ERROR_ABORT, _MAX_SHELL_SCOPE_FLAGS,
    _MCP_SCOPE_PATH_ARGS, _MUTATING_TOOLS, _NETWORK_TOOLS, _SCOPE_PATH_ARGS,
    _SCOPED_TOOLS, _SHELL_COMMAND_ARGS, _SHELL_EXEC_TOOLS,
    _SHELL_NESTED_RUNNER_WORDS, _SHELL_PATH_VALUE_FLAGS, _SHELL_SEPARATORS,
    _SHELL_SUBCOMMAND_RUNNERS, _SHELL_TRANSPARENT_PREFIXES,
    _SHELL_UNSCOPED_TOOLS,
    _TEST_COMMAND_MARKERS, _TODO_TOOLS, _UNDOABLE_TOOLS, _call_target_paths,
)
from .scope import _scope_pattern


def _looks_like_drive_path(tok: str) -> bool:
    """True for a Windows drive-qualified token: ``C:``, ``C:\\x``, ``d:/x``.

    The drive letter must be a single ASCII letter followed by a separator or the
    end of the token. Testing only for a colon in position 1 made every ``a:b``
    token a path, so ``5:30`` and ``4:3`` (an ffmpeg offset, an aspect ratio) and
    ``s:old:new:`` (a sed delimiter) were all reported as out-of-scope paths.
    ``C:foo``, drive-relative with no separator, is given up deliberately: it is
    rare and cannot be told apart from an ordinary key:value argument.

    A real drive-qualified path carries exactly ONE colon, so requiring that
    also rejects ``s:/usr/local:/opt:g``. That is the COMMON sed form (a colon
    delimiter is chosen precisely because the pattern contains slashes), and the
    separator rule alone would still have read it as a drive path.
    """
    if len(tok) < 2 or tok[1] != ":" or tok.count(":") != 1:
        return False
    if not (tok[0].isascii() and tok[0].isalpha()):
        return False
    return len(tok) == 2 or tok[2] in "/\\"


def _is_command_prefix(tok: str) -> bool:
    """True for a token that precedes the real program word without being it.

    An environment assignment (``CI=1 npm test``) or a transparent wrapper
    (``sudo``/``env``/``time``). Without this the prefix takes the program slot,
    the real runner is never recognised, and ``CI=1 npm test`` goes on warning
    about a ``test/`` directory - the very false positive this check exists to
    stop.
    """
    if tok.lower() in _SHELL_TRANSPARENT_PREFIXES:
        return True
    name, sep, _ = tok.partition("=")
    return bool(sep) and name.isidentifier()


class _ExecutionMixin:
    def _scope_rel(self, value: str) -> Optional[str]:
        """
        Resolve a path/glob arg to a cwd-relative POSIX string for scope
        matching, or return None if it escapes cwd.

        Relative paths are joined onto cwd; absolute paths are accepted only
        when they live inside cwd (an in-cwd absolute path that matches the
        scope must pass - BUG-6). Glob metacharacters in *value* (e.g.
        ``**/*.py`` for grep/search_replace) survive resolution: they are kept
        verbatim in the relative string and matched against the scope as-is.
        """
        raw = str(value).replace("\\", "/")
        p = Path(raw)
        cwd = self.cwd.resolve()
        if p.is_absolute():
            try:
                # No resolve(): the path may contain glob chars or not exist.
                rel = Path(raw).relative_to(cwd)
            except ValueError:
                # Try once more against the resolved abs form for symlinks etc.
                try:
                    rel = Path(raw).resolve().relative_to(cwd)
                except ValueError:
                    return None   # outside cwd
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
        # path-arg names (CHK-MCP-SCOPE / CHK-SCOPE-PLUGIN). Best-effort: an unusual
        # path-arg name is not caught (the tool's author is the owner's own config).
        if call.name.startswith(("mcp_", "plugin_")):
            arg_names = _MCP_SCOPE_PATH_ARGS
        else:
            # A nested-path tool (edit_files) has NO top-level `path` arg, so
            # checking arg names alone would find nothing and let the call
            # through - fail-OPEN. Check its real targets first.
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

        A token counts as path-like only when it is unambiguously one: absolute,
        explicitly relative (``./``, ``../``, ``~/``), or an existing file/dir
        under cwd. That deliberately misses things like ``sed s/a/b/`` (which is
        not a path) at the cost of missing a path that does not exist yet - the
        right trade for a warning the user must be able to trust.

        For the same reason the exists-under-cwd guess is skipped in the two
        positions of a command line that hold a verb rather than a path: the
        program word, and the word after a known runner. Otherwise ``npm test``
        in a repo with a ``test/`` directory, ``make docs``, and ``cargo build``
        all warned, which are ordinary layouts and ordinary commands. A path
        written out explicitly (``./build/run.sh``) is still flagged there, and
        ``run_tests`` is unaffected because its ``path`` arg is a path, not a
        command line.
        """
        flagged: list[str] = []
        for arg in _SHELL_COMMAND_ARGS.get(call.name, ()):
            raw = call.args.get(arg)
            if not raw:
                continue
            text = str(raw)
            try:
                # posix=False keeps Windows backslashes intact; quotes survive as
                # part of the token and are stripped below.
                tokens = shlex.split(text, posix=False)
            except ValueError:
                # Unbalanced quotes: fall back to whitespace splitting rather than
                # skipping the check entirely.
                tokens = text.split()
            # Only a real command line has a program word to skip. run_tests
            # passes a target path in `path`, which must stay fully checked.
            cmdline = call.name in _SHELL_EXEC_TOOLS and arg == "command"
            at_program = True
            after_runner = False
            path_flag = False
            for tok in tokens:
                tok = tok.strip("'\"")
                if not tok:
                    continue
                if cmdline and tok in _SHELL_SEPARATORS:
                    at_program, after_runner, path_flag = True, False, False
                    continue
                if tok.startswith("-"):
                    # A flag never takes the program/subcommand slot, but one that
                    # takes a PATH value means the NEXT token is that path.
                    path_flag = tok.split("=", 1)[0] in _SHELL_PATH_VALUE_FLAGS
                    continue
                if path_flag:
                    # "git -C docs status": the value is a path to check, and the
                    # subcommand slot is still waiting for `status`.
                    path_flag = False
                    command_word = False
                elif cmdline and at_program and _is_command_prefix(tok):
                    continue    # "CI=1 npm test": the real program is still to come
                else:
                    command_word = cmdline and (at_program or after_runner)
                    if at_program:
                        prog = tok.replace("\\", "/").rsplit("/", 1)[-1].lower()
                        after_runner = prog in _SHELL_SUBCOMMAND_RUNNERS
                        at_program = False
                    else:
                        # "npm run build": a dispatch subcommand hands off to one
                        # more name, which is a script, not a path.
                        after_runner = (after_runner
                                        and tok.lower() in _SHELL_NESTED_RUNNER_WORDS)
                if tok in flagged:
                    continue
                norm = tok.replace("\\", "/")
                explicit = (norm.startswith(("./", "../", "~/", "/"))
                            or "/../" in norm
                            or _looks_like_drive_path(tok))
                if not explicit:
                    # A command word is a bare VERB. A token carrying a path
                    # separator never is, even in the program slot, so it stays
                    # checked ("build/run.sh", "uv run tools/gen.py").
                    if command_word and "/" not in norm:
                        continue   # a verb ("npm test"), not a path
                    try:
                        if not (self.cwd / tok).exists():
                            continue
                    except OSError:
                        continue   # not a usable path (too long, bad chars)
                try:
                    if not self._scope_allows(tok):
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

    def _execute_tool(self, call: ToolCall, interactive: bool) -> ToolResult:
        TOOL_REGISTRY = _agent.TOOL_REGISTRY  # live: honour a patched agent.TOOL_REGISTRY
        # Hard gate: a tool disabled for this session (e.g. run_shell for a
        # restricted, shareable coder key) can never execute, whatever the model
        # emits. This is the security boundary - the prompt/parse exclusions below
        # are only there so the model does not waste turns trying.
        if call.name in self.disabled_tools:
            result = ToolResult.error(
                f"'{call.name}' is disabled for this session and was not run.")
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

        # Patch-mode: intercept write tools, accumulate diffs, don't touch disk.
        # A write tool the interceptor can't express as a diff must NOT fall
        # through to a real disk write - patch-mode promises no changes.
        if self.patch_mode and call.name in _UNDOABLE_TOOLS:
            chunk = self._patch_mode_intercept(call)
            if chunk is not None:
                self._patch_chunks.append(chunk)
                # Name the files the DIFF actually covers, not every path the
                # call mentioned: a file whose edit produced no change is not
                # "captured", and saying it was overstates what patch mode holds.
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
            return result

        # Scope check - reject file operations that fall outside the active glob.
        # MCP (mcp_*) AND plugin (plugin_*) tools are included so an owner's --scope
        # confines those dynamically-registered file tools too - built-in file tools
        # are default-denied at authoring time (_SCOPED_TOOLS), but the dynamic
        # families are not in the registry then, so they are gated here by prefix.
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
                return result

        # The shell tools are deliberately NOT in _SCOPED_TOOLS: a path-arg check
        # cannot confine a process, so pretending otherwise would be worse than
        # honest. But that trade-off used to be invisible at runtime, leaving a
        # user who set --scope believing the session was confined when it was not.
        # Flag it instead of hiding it - a warning, never a block.
        if self.scope and call.name in _SHELL_UNSCOPED_TOOLS:
            self._warn_shell_outside_scope(call)

        # Dry-run: show destructive calls but don't execute them
        if self.dry_run and tool_def.destructive:
            result = ToolResult.success(
                f"[dry-run] {call.name} - skipped",
                summary=f"[dry-run] {call.name}",
            )
            if interactive:
                console.print("    [dim yellow][dry-run] skipped[/dim yellow]")
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
                self._emit("tool_result", tool=call.name, ok=False,
                           summary="blocked by network policy (net_mode=off)")
                return result

        # Confirmation for destructive tools (diff preview for write_file)
        # and for network tools when net_mode is "ask"
        needs_confirm = (
            (tool_def.destructive or net_mode == "ask") and (
                not self.auto_approve or call.name in self.always_confirm
            )
        )
        if needs_confirm:
            if self.confirm_handler is not None:
                approved = invoke_confirm(self.confirm_handler, call,
                                          agent=self._confirm_agent_label())
            elif interactive:
                approved = self._confirm_tool(call)
            else:
                # Fail CLOSED: this tool requires confirmation, but there is no way
                # to obtain it (non-interactive run, no approval handler). Deny it
                # rather than silently executing - so a configured always_confirm or
                # auto_approve=off is honoured, and an unattended run can never be
                # steered into an unconfirmed destructive/network action. (RULE 5: a
                # safety gate that cannot run must not be treated as passed. The safe
                # default for an unattended coder is the restricted, no-shell agent.)
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

        # Snapshot file content before undoable writes so /undo can restore
        # it and the changed-files tracker can diff against the original
        # (edit_files targets several files in one call, so this is a list: one
        # undo entry per file, or /undo would restore only the first of them.)
        snapshots: dict[str, bytes | None] = {}
        if call.name in _UNDOABLE_TOOLS:
            # One id per CALL: a multi-file call pushes several entries, and
            # undo() reverts them together (see persistence.undo) so /undo can
            # never leave a batch half-restored.
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

        # Episodic change-detection baseline: snapshot the git work-tree state
        # just before the FIRST run_shell, PRE-execution, so a mutating shell
        # command's writes (git apply, a formatter, codegen) can be attributed to
        # THIS session at close - the write-tool tracker never sees them (audit
        # cluster 11). Captured before the shell runs so the shell's own changes
        # are the delta, not part of the baseline; only for episodic sessions.
        if (call.name in _SHELL_EXEC_TOOLS and self._episodic
                and not self._shell_baseline_captured):
            self._shell_baseline_captured = True
            self._git_baseline = self._git_status_paths()

        # Inject hidden runtime args into specific tools
        args = dict(call.args)
        if call.name in ("spawn_agent", "spawn_agent_background"):
            args["_parent_agent"] = self
        # The task-list tools operate on THIS session's state (tools/tasks.py).
        # Injected after the copy, so a model-supplied "_session" cannot win.
        if call.name in _TODO_TOOLS:
            args["_session"] = self
        if call.name in (*_SHELL_EXEC_TOOLS, "fetch_url", "web_search", "generate_image") \
                and self.mode == SessionMode.PRIVACY:
            args["_privacy"] = True

        try:
            result = tool_def.fn(self.cwd, **args)
        except TypeError as e:
            result = ToolResult.error(f"Bad arguments for {call.name}: {e}")
        except Exception as e:
            result = ToolResult.error(f"Tool error: {e}")

        result = self._track_tool_failure(call, result)

        self._audit.tool_result(call.name, result.ok, result.summary)
        if interactive:
            print_tool_result(call.name, result, verbose=self.verbose)
        self._emit("tool_result", tool=call.name, ok=result.ok,
                   summary=result.summary, output=result.output[:4000])

        # Incremental map refresh after file-mutating tools
        if result.ok and call.name in _MUTATING_TOOLS:
            self._refresh_map_for_tool(call)

        self._post_tool_success(call, result, snapshots)

        return result

    def _track_tool_failure(self, call: ToolCall, result: ToolResult) -> ToolResult:
        """Update the per-tool + global failure streaks and arm the circuit
        breakers; on a failure, fold escalating recovery hints into the result.
        Returns the (possibly augmented) result. Split out of _execute_tool; the
        breaker flags it sets are checked back in _loop."""
        # Track consecutive failures and inject escalating recovery hints;
        # at 4 identical failures the circuit breaker stops the task after
        # this batch (checked in _loop) instead of burning the turn budget.
        if not result.ok:
            # Record the ORIGINAL error (before the hint augmentation below) into
            # the bounded session error trace, so the close-time episode reflection
            # has real evidence for what_failed (audit cluster 13).
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

    def _patch_mode_intercept(self, call: ToolCall) -> Optional[str]:
        """
        Compute a unified diff for a write/edit/patch call without touching disk.

        Returns the diff string, or None if the diff cannot be computed.
        """
        # edit_files spans several files, so it has no single old_content -
        # without this branch compute_tool_diff returns None, the intercept
        # reports "cannot be captured", and patch mode silently loses the edit.
        if call.name == "edit_files":
            return compute_multifile_diff(self.cwd, call.args.get("edits"))
        path_arg = call.args.get("path", "")
        old_text = read_old_content(self.cwd, path_arg)
        return compute_tool_diff(call.name, call.args, old_text)

    def flush_patch(self, output_path: Optional[Path] = None) -> str:
        """
        Return the accumulated unified diff (and optionally write it to a file).

        Clears the internal patch buffer.
        """
        content = "\n".join(c for c in self._patch_chunks if c)
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
        asking for itself". The top-level agent deliberately gets None so its own
        prompts are worded exactly as they always were.

        This is the ONE place a child's identity enters the confirm chain, so every
        delegation path (worktree-isolated parallel dispatch, spawn_agent, background
        sub-agents) is attributed by construction rather than each re-implementing it.

        A child with a falsy name still gets a label. ``spawn_agent``'s ``name`` comes
        straight from the model's tool-call arguments, and an empty one would collapse
        to "no label" - making a delegated request look exactly like the human's own,
        which is the confusion this whole path exists to prevent. Being unable to say
        WHICH child is asking is tolerable; letting a child's prompt pass for the
        user's own is not.
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
        # Live-attribute access so tests patching agent.confirm / confirm_diff /
        # print_diff_preview are honoured (the names moved into this submodule).
        confirm = _agent.confirm
        confirm_diff = _agent.confirm_diff
        print_diff_preview = _agent.print_diff_preview
        if call.name in ("write_file", "edit_file"):
            path_arg = call.args.get("path", "")
            old_content = read_old_content(self.cwd, path_arg)
            new_content = resolve_new_content(call.name, call.args, old_content)
            print_diff_preview(old_content, new_content, path_label=path_arg)
            return confirm_diff(path_arg or "file")

        # A multi-file edit is the LAST thing to approve blind, so show the same
        # kind of diff, concatenated over every file the call would touch.
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

    def _refresh_map_for_tool(self, call: ToolCall) -> None:
        """Update the project map for files touched by a write/edit tool call."""
        paths = dict.fromkeys(_call_target_paths(call.name, call.args))
        if paths:
            for path_arg in paths:
                abs_path = (self.cwd / path_arg).resolve()
                self._project_map.refresh_file(abs_path)
            # Regenerate the system prompt with the updated map (combined mcp+
            # plugin+skill tool docs preserved - see _rebuild_system_prompt).
            # Once, after ALL the call's files, not once per file.
            self._rebuild_system_prompt()
