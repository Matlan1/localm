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
from ..audit import SessionMode, unregister_coder_session_mode

# Upper bound on how long the CLI's SYNCHRONOUS close-time reflection (see
# _maybe_store_episode) may hold the process open. A no-file-change session
# that ended via max_turns, a circuit breaker, or a failed verify oracle still
# pays for one 1024-token model call before the CLI can exit (REG-594) - this
# caps that wait instead of leaving it unbounded, and _reflect_into_episode's
# own try/except (episodic memory is best-effort) turns an expired deadline
# into the same "reflection skipped" outcome as any other backend failure.
# GUI/web sessions never see this: they already background the call
# (on_event is not None below), so nothing there is waiting on it.
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
        # Goes with them: the similarity breaker compares against this history,
        # so carrying it past a cleared conversation would let a new session
        # trip on responses the user can no longer see.
        self._recent_finals = []
        self._last_run_ok = True
        # Goes with it: a verdict about a run whose conversation has just been
        # dropped is not a verdict about anything. _loop re-arms this too, so
        # nothing reads a stale value today - but leaving it set here is the
        # exact one-way-flag shape that made a failed turn poison every later
        # one (#792), and the field is new enough to close it before it bites.
        self._last_verify_state = None
        # Clearing the conversation also drops the evidence a failure lesson would
        # be built from (history, error trace), so the session-level failure marker
        # goes with it - otherwise /clear would leave a close-time reflection armed
        # for a run whose context no longer exists.
        self._had_any_failure = False
        self._unverified_writes.clear()
        self._review_task = ""
        self._error_trace.clear()
        self._shell_baseline_captured = False
        self._git_baseline = None
        # A cleared conversation is a NEW session, not a continuation of
        # whatever this agent had resumed - it must get its own checkpoint
        # identity and title (NEW-CODER-RESUME-DESTROYS-SESSIONS), or a later
        # interruption would silently overwrite the checkpoint of the session
        # /clear just discarded, one file collision reintroduced right where
        # this whole fix started.
        import uuid
        self._checkpoint_id = uuid.uuid4().hex[:12]
        self._session_title = ""

    def _rebuild_system_prompt(self) -> None:
        """Single source of truth for (re)building the system prompt.

        Every build and rebuild site goes through here so the kwargs cannot drift -
        notably the COMBINED external tool docs (mcp + plugin + skill) and the
        provenance flag. A prior bug rebuilt with only ``_mcp_docs``, so plugin
        tools and agent skills silently vanished from the prompt after a reindex /
        memory reload / per-write map refresh, and the model "forgot" they existed.
        """
        self._system_prompt = build_system_prompt(
            self.cwd,
            agent_name=self.name,
            project_map=self._project_map,
            memory=self._memory,
            model_name=getattr(self, "_family_id", self._model_name),  # REC-CODER-FAMILY
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
        where the NEXT save lands. Leaving the old file behind would give one
        conversation two resumable entries - and the abandoned one is frozen
        forever at the moment of the cd while describing work that has since
        moved elsewhere, so "continue last session" in the old project would
        offer a phantom that can never catch up. Migrating keeps exactly one
        copy, where the session actually is.

        Best-effort, deliberately: the cwd change itself has already been
        decided by the caller and must not fail because a stale file could not
        be moved. A failure is logged rather than swallowed, and it is not
        silent data loss - the old checkpoint simply stays where it was.
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
        never the whole project directory - a sibling session's checkpoint in
        the same project is none of this session's business, exactly as
        clear_checkpoint is careful not to reach one.
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

        Both surfaces, because they have different channels: ``print_warning``
        reaches the CLI (which registers no event sink, so ``_emit`` is a no-op
        there), and an ``info`` event reaches the GUI (which renders ``info`` and
        would drop an unknown ``warning`` type). Capping without this would be a
        silent truncation, which AGENTS.md rule 5 forbids.

        Called from every site that (re)loads those files, so the user hears about
        it right when they cause it - at session start, on /remember and /forget,
        and on a cwd change - rather than never.
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
        if self.parent is None:
            unregister_coder_session_mode(self.mode)
        if self.mode == SessionMode.FULL:
            return self._write_session_markdown()
        return None

    def _maybe_store_episode(self) -> None:
        """Distil this finished session into one episodic-memory record.

        Gated on the privacy contract: skipped in privacy mode and for restricted
        sessions, so no trace is written that the mode forbids. Fires when the
        session changed files (via the write-tool tracker, OR via run_shell detected
        against a git baseline), or when it FAILED (incomplete, or repeated tool /
        command errors) even with no file change - failure lessons are the most
        valuable and were systematically absent before (audit cluster 11). A clean
        read-only / no-op session still adds nothing. GUI/web sessions (event sink,
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
            # session still reflects (audit cluster 11).
            changed, git_diff = self._detect_shell_changes()
            if changed:
                diff_override = git_diff
        # A failure with no file change (an investigation-only or
        # failed-before-first-write session) still carries a lesson - but the bar
        # is a run that ACTUALLY failed to finish, and this stays deliberately
        # narrow because it costs a full model reflection that the CLI pays for
        # SYNCHRONOUSLY as the user quits. Two things are not failures (REG-594):
        #  - a bare tool-error COUNT: `len(self._error_trace) >= 2` fired on any
        #    routine read-only session that hit a couple of incidental failures (a
        #    missing read_file, a failed grep) - extremely common, no lesson. It was
        #    redundant too: a genuinely broken run trips max_turns or a circuit
        #    breaker (_GLOBAL_ERROR_ABORT / _REPEAT_RESPONSE_ABORT / the per-tool
        #    streak), and every one of those already clears _last_run_ok.
        #  - a USER-initiated stop: the user asked to leave, which is the worst
        #    moment to start a 1024-token inference, and their own Ctrl-C is not a
        #    lesson to learn.
        # "Did ANY run this session fail", not just the last one: _last_run_ok is
        # per-run (re-armed at the start of every _loop), so a session whose first
        # turn failed and whose second turn succeeded would otherwise lose its
        # lesson. _had_any_failure is _loop's session-level record; the second term
        # keeps the answer right when _last_run_ok is set outside _loop, where a
        # not-ok current state is itself the failure.
        # The user-stop guard scopes to that SECOND term ONLY. _user_stopped is
        # per-run exactly like _last_run_ok, so the pair describes THIS run, while
        # _had_any_failure already excludes stops at the point it is written
        # (_loop's finally). Spread over the whole disjunction the guard read "the
        # last run was stopped, so forget every earlier failure too", and a genuine
        # max_turns or circuit-breaker failure followed by a stopped run lost its
        # lesson - the same #594 loss this trigger exists to prevent.
        had_failure = self._had_any_failure or (
            not self._last_run_ok and not self._user_stopped)
        if not changed and not had_failure:
            return
        if self.on_event is not None:
            threading.Thread(
                target=self._reflect_into_episode, args=(changed,),
                kwargs={"diff_override": diff_override}, daemon=True).start()
        else:
            # CLI: the process is about to exit, so this must still run
            # SYNCHRONOUSLY (a daemon thread might never get scheduled again
            # before exit) - but it must not be a SILENT, UNBOUNDED wait
            # (REG-594): say so, and cap it.
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
        GUI/web path - it already runs this off a daemon thread with nobody
        waiting on it (REG-594's caller, the CLI, always passes one)."""
        print_warning = _agent.print_warning  # live: honour a patched agent.print_warning
        try:
            import time as _time

            from ..episodes import reflect_and_store
            files = [c.get("path") for c in changed if c.get("path")]
            task = self._episode_task or next(
                (m.get("content", "") for m in self._messages
                 if m.get("role") == "user"), "")
            # The per-run flag is the right one HERE (unlike the trigger above):
            # this records how the session ENDED, and the thin-episode fallback
            # spells the distinction out - "did not complete" vs "completed with
            # errors". A session that failed and then recovered completed; its
            # failure is carried by what_failed, not by the outcome label.
            outcome = "ok" if self._last_run_ok else "incomplete"
            diff = diff_override if diff_override is not None else self.session_diff()
            # Real evidence of what went wrong, so the reflection can actually fill
            # what_failed instead of restating the diff (audit cluster 13).
            errors = "\n".join(self._error_trace)

            def _complete(prompt: str) -> str:
                # 1024 tokens, not 400: thinking models spend the first
                # hundreds of tokens on their reasoning channel, and with the
                # old cap the JSON never arrived, so NO episode was ever
                # stored on those models (memory-audit 2026-07-02, live).
                # strip_think keeps the scratchpad out of the stored lesson.
                from localm.textnorm import strip_think

                def _call() -> str:
                    return self.backend.chat(
                        [{"role": "user", "content": prompt}],
                        max_tokens=1024) or ""

                if deadline is None:
                    return strip_think(_call())
                # Bounded: run the blocking HTTP/inference call off-thread and
                # cap the wait, so a slow or wedged backend cannot hold the CLI
                # open indefinitely (REG-594). reflect_and_store already treats
                # an exception from _complete as "no usable reply" and, when
                # the session had real tool/command errors, still stores a
                # thin failure episode from those (episodes.py) - so a timeout
                # costs the model's PROSE, not the evidence that led here.
                # daemon=True: if the deadline expires this thread is
                # abandoned, not joined again; it must never block interpreter
                # exit on its own. Named "_holder", not "outcome" - the
                # enclosing _reflect_into_episode already binds "outcome" to
                # the "ok"/"incomplete" episode-outcome string, and shadowing
                # it here would be one accidental read away from a real bug.
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
        # passes the exact same expression to parse_tool_calls) so the name-gated
        # ```json/bare-JSON call shapes are recognised here exactly as they were
        # when the model actually called them, not a hardcoded or stale subset.
        # Computed once, not per message: neither operand changes while this
        # method runs.
        tool_names = set(_agent.TOOL_REGISTRY) - self.disabled_tools

        # The transcript strips every recognised tool-call shape with the PARSER's
        # own strip_tool_calls rather than a private copy of the regex. The copy
        # that used to live here was the same ambiguous ``>\s*(.*?)\s*</tool_call>``
        # shape (cubic: measured 0.11 / 0.88 / 5.49s on a bare <tool_call> followed
        # by 500 / 1000 / 2000 spaces, min of 3 on a quiet box) and it ran over
        # EVERY stored assistant message at teardown, so a single poisoned response
        # re-hung the export on every later close. strip_tool_calls is a strict
        # superset of the old strip_xml_tool_calls-only splitter (it also
        # recognises the fenced ```json/```tool_call and bare top-level JSON
        # shapes parse_tool_calls does), so a call written in one of THOSE shapes
        # is now summarised here too instead of surviving as raw fence markers or
        # a raw JSON blob.
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
                # to history before the repair succeeds, a real and not rare
                # local-model failure mode) - the same generic placeholder this
                # path has always shown, now counted by strip_tool_calls instead
                # of a bare json.loads try/except here. Rendered after `calls` in
                # a separate pass, so if one message somehow mixed a validly
                # parsed call with a SEPARATE malformed block, bullet order could
                # diverge from strict textual order - an accepted, purely
                # cosmetic quirk for a doubly-rare combination, not worth the
                # extra span-tracking it would take to solve precisely.
                for _ in range(malformed):
                    lines.append("  - (tool call)")

                lines.append("")

            lines.append("---")
            lines.append("")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path
