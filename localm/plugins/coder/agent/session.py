# SPDX-License-Identifier: AGPL-3.0-or-later
"""Session lifecycle: history reset, system-prompt (re)build, cwd/reindex/memory
refresh, history save, and session close (audit close + episodic reflection +
the Markdown transcript). Mixed into Agent."""

from __future__ import annotations

import threading
from pathlib import Path

import localm.plugins.coder.agent as _agent
from ..memory import cap_user_instructions, forget, remember
from ..parser import strip_xml_tool_calls
from ..prompts import build_subagent_system_prompt, build_system_prompt
from ..audit import SessionMode


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
        had_failure = (
            (self._had_any_failure or not self._last_run_ok)
            and not self._user_stopped)
        if not changed and not had_failure:
            return
        if self.on_event is not None:
            threading.Thread(
                target=self._reflect_into_episode, args=(changed,),
                kwargs={"diff_override": diff_override}, daemon=True).start()
        else:
            self._reflect_into_episode(changed, diff_override=diff_override)

    def _reflect_into_episode(self, changed: list, diff_override=None) -> None:
        """Build and store one episode for this session (best-effort).

        *diff_override* supplies the work-log diff when the changes were detected
        outside the write-tool tracker (e.g. run_shell writes via git); None means
        use the tracker's cumulative session_diff()."""
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
                return strip_think(self.backend.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=1024) or "")

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

        # The transcript strips tool_call blocks with the PARSER's splitter rather
        # than a private copy of the regex. The copy that used to live here was the
        # same ambiguous ``>\s*(.*?)\s*</tool_call>`` shape (cubic: measured
        # 0.11 / 0.88 / 5.49s on a bare <tool_call> followed by 500 / 1000 / 2000
        # spaces, min of 3 on a quiet box) and it runs over EVERY
        # stored assistant message at teardown, so a single poisoned response
        # re-hung the export on every later close. Two copies also meant two things
        # to fix and a drift risk; the parser's is a strict superset (it also
        # matches the ``name=`` attribute form and is case-insensitive), so the
        # transcript now renders mangled calls it previously dumped as raw XML.
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
                calls, clean = strip_xml_tool_calls(content)
                clean = clean.strip()

                if clean:
                    lines.append(f"**{self.name}**: {clean[:2000]}")
                elif calls:
                    lines.append(f"**{self.name}**:")

                for name_attr, raw_json in calls:
                    try:
                        import json as _json
                        obj  = _json.loads(raw_json)
                        # Two body shapes, same split _try_parse_body makes: a
                        # full {"name":..., "args":...}, or an args-ONLY body
                        # whose tool name lives on the name= attribute. Those
                        # tagged blocks used to survive into the transcript as
                        # raw XML that at least named the tool, so without the
                        # name_attr fallback stripping them renders a bare "?".
                        if "name" in obj:
                            tool = obj.get("name") or "?"
                            args = obj.get("args")
                            if args is None:
                                args = obj.get("arguments", {})
                        else:
                            tool = name_attr or "?"
                            args = obj
                        if not isinstance(args, dict):
                            args = {}
                        # Show path/command arg if present, else first arg value
                        hint = (
                            args.get("path")
                            or args.get("command")
                            or args.get("url")
                            or (next(iter(args.values()), None) if args else None)
                        )
                        hint_str = f" `{str(hint)[:60]}`" if hint else ""
                        lines.append(f"  - `{tool}`{hint_str}")
                    except Exception:
                        lines.append("  - (tool call)")

                lines.append("")

            lines.append("---")
            lines.append("")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path
