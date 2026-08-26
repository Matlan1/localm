# SPDX-License-Identifier: AGPL-3.0-or-later
"""Session lifecycle: history reset, system-prompt (re)build, cwd/reindex/memory
refresh, history save, and session close (audit close + episodic reflection +
the Markdown transcript). Mixed into Agent."""

from __future__ import annotations

import threading
from pathlib import Path

import localm.plugins.coder.agent as _agent
from ..display import print_info
from ..memory import cap_user_instructions, forget, remember
from ..parser import strip_tool_calls
from ..prompts import build_subagent_system_prompt, build_system_prompt
from ..audit import SessionMode

# Upper bound on how long the CLI's SYNCHRONOUS close-time reflection (see
# _maybe_store_episode) may hold the process open. An expired deadline is turned
# by _reflect_into_episode's own try/except into the same "reflection skipped"
# outcome as any other backend failure. GUI/web sessions background the call
# (on_event is not None below), so this cap does not apply to them.
_CLI_REFLECTION_DEADLINE_S = 30.0


class _SessionMixin:
    def reset(self) -> None:
        """Clear conversation history."""
        self._messages = []
        self._turns = 0
        self._total_tokens = 0
        self._last_turn_tokens = 0
        self._compact_warned = False
        self._consecutive_errors.clear()
        self._global_error_streak = 0
        self._abort_no_progress = False
        self._last_response_fp = ""
        self._repeat_response_count = 0
        # Cleared with them: the similarity breaker compares against this history,
        # so a new session must not trip on responses the user can no longer see.
        self._recent_finals = []
        self._last_run_ok = True
        # Cleared with it: a verdict about a run whose conversation has just been
        # dropped is not a verdict about anything. _loop re-arms this at the start
        # of every run as well.
        self._last_verify_state = None
        # Clearing the conversation also drops the evidence a failure lesson would
        # be built from (history, error trace), so the session-level failure marker
        # goes with it and /clear leaves no close-time reflection armed.
        self._had_any_failure = False
        self._unverified_writes.clear()
        self._review_task = ""
        self._error_trace.clear()
        self._shell_baseline_captured = False
        self._git_baseline = None
        # A cleared conversation is a NEW session, not a continuation of whatever
        # this agent had resumed, so it gets its own checkpoint identity and title.
        # Without that, a later interruption would overwrite the checkpoint of the
        # session /clear just discarded.
        import uuid
        self._checkpoint_id = uuid.uuid4().hex[:12]
        self._session_title = ""

    def _rebuild_system_prompt(self) -> None:
        """Single source of truth for (re)building the system prompt.

        Every build and rebuild site goes through here so the kwargs cannot
        drift - notably the COMBINED external tool docs (mcp + plugin + skill)
        and the provenance flag. Rebuilding with only ``_mcp_docs`` would drop
        plugin tools and agent skills from the prompt after a reindex, memory
        reload or per-write map refresh.
        """
        self._system_prompt = build_system_prompt(
            self.cwd,
            agent_name=self.name,
            project_map=self._project_map,
            memory=self._memory,
            model_name=getattr(self, "_family_id", self._model_name),  # family id, not the alias
            extra_tool_docs="\n\n".join(
                d for d in (self._mcp_docs, self._plugin_docs, self._skill_docs) if d
            ),
            disabled_tools=self.disabled_tools,
            untrusted_provenance=self._untrusted_provenance,
            custom_instructions=self._custom_instructions,
            role_brief=self._role_brief(),
        )

    def _role_brief(self) -> str:
        """The role section for a spawned sub-agent; empty for a main agent.

        Built from the CURRENT disabled_tools so the advertised tool line tracks
        the narrowing that was actually applied, rather than the preset's ideal.
        """
        preset = getattr(self, "_role_preset", None)
        if preset is None:
            return ""
        return build_subagent_system_prompt(
            self.cwd,
            preset.name,
            model_name=getattr(self, "_family_id", self._model_name),
            disabled_tools=self.disabled_tools,
            mission=preset.mission,
        )

    def set_cwd(self, cwd: Path) -> None:
        """Point this session at another project directory (the REPL's /cd, and
        the GUI's cwd route).

        THE CHECKPOINT MOVES WITH THE SESSION. A checkpoint is filed under
        ``<digest(cwd)>/<checkpoint_id>.json``, so changing the cwd changes
        where the NEXT save lands. The old file is migrated, keeping exactly one
        copy, where the session actually is; leaving it behind would give one
        conversation two resumable entries, the abandoned one frozen at the
        moment of the cd.

        The migration is best-effort: the cwd change has already been decided by
        the caller and does not fail because a stale file could not be moved. A
        failure is logged, and the old checkpoint stays where it was.
        """
        self._migrate_checkpoint(self.cwd, cwd)
        load_memory = _agent.load_memory  # live: honour a patched agent.load_memory
        load_custom_instructions = _agent.load_custom_instructions
        self.cwd = cwd
        self._project_map = self._build_project_map(cwd)
        self._memory = load_memory(cwd)
        # An explicit --system override persists across a cwd change; otherwise
        # re-read the new cwd's .localcoder/system.md.
        self._custom_instructions = (
            cap_user_instructions(self._system_override)
            if self._system_override is not None
            else load_custom_instructions(cwd))
        self._rebuild_system_prompt()
        self._warn_injected_file_limits()

    def _migrate_checkpoint(self, old_cwd: Path, new_cwd: Path) -> None:
        """Move this session's own checkpoint file from *old_cwd*'s project
        directory to *new_cwd*'s. See set_cwd for why.

        Only ever touches THIS agent's own file (keyed on ``_checkpoint_id``),
        never the whole project directory, so a sibling session's checkpoint in
        the same project is left alone - the same scope clear_checkpoint keeps.
        """
        from .checkpoint import _checkpoint_path_for
        try:
            if Path(old_cwd).resolve() == Path(new_cwd).resolve():
                return
            src = _checkpoint_path_for(old_cwd, self._checkpoint_id)
            if not src.is_file():
                return                       # nothing saved yet: nothing to move
            dst = _checkpoint_path_for(new_cwd, self._checkpoint_id)
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dst)   # atomic within a filesystem; overwrites a stale dst
        except Exception as e:                                 # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(
                "coder: could not move this session's checkpoint from %s to %s "
                "(%s); the old copy stays where it is and this session's next "
                "save will start a new one under the new directory",
                old_cwd, new_cwd, e)

    def reindex(self) -> int:
        """Rebuild the full project map and regenerate the system prompt."""
        self._project_map = self._build_project_map(self.cwd)
        self._rebuild_system_prompt()
        return self._project_map.file_count()

    def _warn_injected_file_limits(self) -> None:
        """Tell the user when project memory or user instructions did NOT go into
        the system prompt whole (over the injection budget, or unreadable).

        Both surfaces are used, because they have different channels:
        ``print_warning`` reaches the CLI (which registers no event sink, so
        ``_emit`` is a no-op there), and an ``info`` event reaches the GUI (which
        renders ``info`` and would drop an unknown ``warning`` type).

        Called from every site that (re)loads those files - at session start, on
        /remember and /forget, and on a cwd change.
        """
        print_warning = _agent.print_warning  # live: honour a patched print_warning
        for text in (_agent.memory_warning(self.cwd),
                     _agent.custom_instructions_warning(self.cwd,
                                                        self._system_override)):
            if text:
                print_warning(text)
                self._emit("info", text=text)

    def reload_memory(self) -> str:
        """Re-read the memory file from disk and rebuild the system prompt."""
        load_memory = _agent.load_memory  # live: honour a patched agent.load_memory
        self._memory = load_memory(self.cwd)
        self._rebuild_system_prompt()
        self._warn_injected_file_limits()
        return self._memory

    def remember(self, text: str) -> Path:
        """Append a bullet to the memory file and refresh the system prompt."""
        p = remember(self.cwd, text)
        self.reload_memory()
        return p

    def forget(self, pattern: str) -> tuple:
        """Remove matching bullets from the memory file and refresh the system prompt."""
        p, n = forget(self.cwd, pattern)
        if n:
            self.reload_memory()
        return p, n

    def save_history(self, path: Path) -> None:
        import json as _json
        path.write_text(
            _json.dumps(self._messages, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def close(self) -> Path | None:
        """
        Finalise the session.

        - Closes the audit log (``log`` and ``full`` modes).
        - Writes a Markdown transcript to ``.localcoder/sessions/`` in
          ``full`` mode.

        Returns the path of the Markdown file, or None.
        Called automatically by the CLI's ``finally`` block.
        """
        self._maybe_store_episode()
        self._audit.close()
        if self.mode == SessionMode.FULL:
            return self._write_session_markdown()
        return None

    def _maybe_store_episode(self) -> None:
        """Distil this finished session into one episodic-memory record.

        Gated on the privacy contract: skipped in privacy mode and for restricted
        sessions, so no trace is written that the mode forbids. Fires when the
        session changed files (via the write-tool tracker, OR via run_shell detected
        against a git baseline), or when it FAILED (incomplete, or repeated tool /
        command errors) even with no file change. A clean read-only / no-op
        session adds nothing. GUI/web sessions (event sink,
        still-running server) run the reflection in a background thread so the model
        call never blocks the close path / event loop; CLI runs reflect synchronously
        because the process is about to exit and a daemon thread might not finish.
        """
        if not self._episodic or self._episode_store is None:
            return
        if self.mode == SessionMode.PRIVACY or self.restricted:
            return
        changed = self.changed_files()
        diff_override = None
        if not changed:
            # run_shell writes (git apply, formatters, codegen) are invisible to
            # the write-tool tracker; recover them from git so a shell-driven
            # session still reflects.
            changed, git_diff = self._detect_shell_changes()
            if changed:
                diff_override = git_diff
        # A failure with no file change (an investigation-only or
        # failed-before-first-write session) still stores an episode, but the bar
        # is a run that ACTUALLY failed to finish. Two things are not failures:
        #  - a bare tool-error count: incidental failures (a missing read_file, a
        #    failed grep) carry no lesson, and a genuinely broken run trips
        #    max_turns or a circuit breaker (_GLOBAL_ERROR_ABORT /
        #    _REPEAT_RESPONSE_ABORT / the per-tool streak), each of which already
        #    clears _last_run_ok.
        #  - a USER-initiated stop.
        # The question is "did ANY run this session fail", not just the last one:
        # _last_run_ok is per-run (re-armed at the start of every _loop), while
        # _had_any_failure is _loop's session-level record. The second term keeps
        # the answer right when _last_run_ok is set outside _loop, where a not-ok
        # current state is itself the failure.
        # The user-stop guard scopes to that SECOND term ONLY: _user_stopped is
        # per-run exactly like _last_run_ok, so the pair describes THIS run, while
        # _had_any_failure already excludes stops at the point it is written
        # (_loop's finally). Spread over the whole disjunction it would discard
        # every earlier failure whenever the last run was stopped.
        had_failure = self._had_any_failure or (
            not self._last_run_ok and not self._user_stopped)
        if not changed and not had_failure:
            return
        if self.on_event is not None:
            threading.Thread(
                target=self._reflect_into_episode, args=(changed,),
                kwargs={"diff_override": diff_override}, daemon=True).start()
        else:
            # CLI: the process is about to exit, so this runs SYNCHRONOUSLY - a
            # daemon thread might never be scheduled again before exit. The wait is
            # announced and capped rather than silent and unbounded.
            print_info("Reflecting on this session before exiting...")
            self._reflect_into_episode(
                changed, diff_override=diff_override,
                deadline=_CLI_REFLECTION_DEADLINE_S)

    def _reflect_into_episode(self, changed: list, diff_override=None,
                              deadline: "float | None" = None) -> None:
        """Build and store one episode for this session (best-effort).

        *diff_override* supplies the work-log diff when the changes were detected
        outside the write-tool tracker (e.g. run_shell writes via git); None means
        use the tracker's cumulative session_diff(). *deadline* bounds the model
        call itself (seconds); None means unbounded, which is correct for the
        GUI/web path, since it already runs this off a daemon thread with nobody
        waiting on it. The CLI caller always passes one."""
        print_warning = _agent.print_warning  # live: honour a patched agent.print_warning
        try:
            import time as _time

            from ..episodes import reflect_and_store
            files = [c.get("path") for c in changed if c.get("path")]
            task = self._episode_task or next(
                (m.get("content", "") for m in self._messages
                 if m.get("role") == "user"), "")
            # The per-run flag, not the session-level one used by the trigger
            # above: this records how the session ENDED ("did not complete" vs
            # "completed with errors"). A session that failed and then recovered
            # completed; its failure is carried by what_failed.
            outcome = "ok" if self._last_run_ok else "incomplete"
            diff = diff_override if diff_override is not None else self.session_diff()
            # Real evidence of what went wrong, so the reflection can fill
            # what_failed instead of restating the diff.
            errors = "\n".join(self._error_trace)

            def _complete(prompt: str) -> str:
                # 1024 tokens: a thinking model spends the first hundreds of
                # tokens on its reasoning channel, and a lower cap leaves no room
                # for the JSON. strip_think keeps the scratchpad out of the stored
                # lesson.
                from localm.textnorm import strip_think

                def _call() -> str:
                    return self.backend.chat(
                        [{"role": "user", "content": prompt}],
                        max_tokens=1024) or ""

                if deadline is None:
                    return strip_think(_call())
                # Bounded: run the blocking HTTP/inference call off-thread and cap
                # the wait, so a slow or wedged backend cannot hold the CLI open.
                # reflect_and_store treats an exception from _complete as "no
                # usable reply" and, when the session had real tool/command errors,
                # still stores a thin failure episode from those (episodes.py), so
                # a timeout costs the model's prose, not the evidence.
                # daemon=True: an expired deadline abandons this thread rather than
                # joining it, and it must never block interpreter exit.
                # Named "_holder": the enclosing _reflect_into_episode already
                # binds "outcome" to the "ok"/"incomplete" episode-outcome
                # string.
                _holder: dict = {}

                def _target() -> None:
                    try:
                        _holder["text"] = _call()
                    except Exception as e:
                        _holder["error"] = e

                t = threading.Thread(target=_target, daemon=True)
                t.start()
                t.join(deadline)
                if t.is_alive():
                    raise TimeoutError(
                        f"reflection did not finish within {deadline:.0f}s")
                if "error" in _holder:
                    raise _holder["error"]
                return strip_think(_holder.get("text", ""))

            reflect_and_store(
                self._episode_store, task=task, diff=diff,
                outcome=outcome, files=files, turns=self.turns,
                errors=errors, complete=_complete, ts=_time.time())
        except Exception as e:
            print_warning("episodic memory: reflection skipped (%s)" % e)

    def _write_session_markdown(self) -> Path:
        """
        Write a human-readable Markdown transcript of the session to
        ``.localcoder/sessions/<YYYY-MM-DD_HHMMSS>.md`` inside the project
        working directory.

        Tool-result messages (which are large XML blobs) are skipped.
        Tool calls embedded in assistant messages are extracted and listed
        as bullet points.
        """
        import time as _time

        ts_label = _time.strftime("%Y-%m-%d_%H%M%S")
        ts_human = _time.strftime("%Y-%m-%d %H:%M:%S")

        out_dir = self.cwd / ".localcoder" / "sessions"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ts_label}.md"

        tokens_line = (
            f"**Tokens (billed est.)**: ~{self._total_tokens:,}  "
            if self._total_tokens
            else ""
        )

        lines: list[str] = [
            f"# localcoder Session - {ts_human}",
            "",
            f"**Model**: {self._model_name or 'unknown'}  ",
            f"**Working directory**: {self.cwd}  ",
            f"**Turns**: {self._turns}  ",
        ]
        if tokens_line:
            lines.append(tokens_line)
        lines += ["", "---", ""]

        # tool_names mirrors the live agentic loop's own gate (loop.py's _loop
        # passes the same expression to parse_tool_calls), so the name-gated
        # json/bare-JSON call shapes are recognised here as they were when the
        # model called them. Computed once, not per message: neither operand
        # changes while this method runs.
        tool_names = set(_agent.TOOL_REGISTRY) - self.disabled_tools

        # The transcript strips every recognised tool-call shape with the
        # PARSER's own strip_tool_calls, a strict superset of an
        # strip_xml_tool_calls-only splitter: it also recognises the fenced
        # json/tool_call and bare top-level JSON shapes parse_tool_calls does, so
        # a call written in one of those is summarised rather than surviving as
        # raw fence markers or a raw JSON blob.
        for msg in self._messages:
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if not isinstance(content, str):
                # multipart - join text parts
                content = " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict)
                )

            # Skip tool-result feed-backs (huge XML blobs)
            if content.lstrip().startswith("<tool_result"):
                continue

            if role == "user":
                lines.append(f"**You**: {content[:2000]}")
                lines.append("")

            elif role == "assistant":
                # Strip tool_call blocks and extract summaries
                calls, clean, malformed = strip_tool_calls(content, tool_names=tool_names)
                clean = clean.strip()

                if clean:
                    lines.append(f"**{self.name}**: {clean[:2000]}")
                elif calls or malformed:
                    lines.append(f"**{self.name}**:")

                for call in calls:
                    args = call.args
                    # Show path/command arg if present, else first arg value
                    hint = (
                        args.get("path")
                        or args.get("command")
                        or args.get("url")
                        or (next(iter(args.values()), None) if args else None)
                    )
                    hint_str = f" `{str(hint)[:60]}`" if hint else ""
                    lines.append(f"  - `{call.name}`{hint_str}")
                # A block that LOOKED like a tool call but whose JSON body never
                # parsed (loop.py's tool-call repair path persists the raw attempt
                # to history before the repair succeeds) gets a generic
                # placeholder, counted by strip_tool_calls. Rendered after `calls`
                # in a separate pass, so a message mixing a validly parsed call
                # with a separate malformed block can list its bullets out of
                # strict textual order.
                for _ in range(malformed):
                    lines.append("  - (tool call)")

                lines.append("")

            lines.append("---")
            lines.append("")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path
