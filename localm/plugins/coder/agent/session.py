# SPDX-License-Identifier: AGPL-3.0-or-later
"""Session lifecycle: history reset, system-prompt (re)build, cwd/reindex/memory
refresh, history save, and session close (audit close + episodic reflection +
the Markdown transcript). Mixed into Agent."""

from __future__ import annotations

import threading
from pathlib import Path

import localm.plugins.coder.agent as _agent
from ..memory import forget, remember
from ..prompts import build_system_prompt
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
        self._unverified_writes.clear()
        self._review_task = ""

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
            self._system_override if self._system_override is not None
            else load_custom_instructions(cwd))
        self._rebuild_system_prompt()

    def reindex(self) -> int:
        """Rebuild the full project map and regenerate the system prompt."""
        self._project_map = self._build_project_map(self.cwd)
        self._rebuild_system_prompt()
        return self._project_map.file_count()

    def reload_memory(self) -> str:
        """Re-read the memory file from disk and rebuild the system prompt."""
        load_memory = _agent.load_memory  # live: honour a patched agent.load_memory
        self._memory = load_memory(self.cwd)
        self._rebuild_system_prompt()
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
        sessions, so no trace is written that the mode forbids. Only fires when the
        session actually changed files, so a read-only or no-op session adds
        nothing. GUI/web sessions (which have an event sink and a still-running
        server) run the reflection in a background thread so the model call never
        blocks the close path / event loop; CLI runs reflect synchronously because
        the process is about to exit and a daemon thread might not finish.
        """
        if not self._episodic or self._episode_store is None:
            return
        if self.mode == SessionMode.PRIVACY or self.restricted:
            return
        changed = self.changed_files()
        if not changed:
            return
        if self.on_event is not None:
            threading.Thread(target=self._reflect_into_episode,
                             args=(changed,), daemon=True).start()
        else:
            self._reflect_into_episode(changed)

    def _reflect_into_episode(self, changed: list) -> None:
        """Build and store one episode for this session (best-effort)."""
        print_warning = _agent.print_warning  # live: honour a patched agent.print_warning
        try:
            import time as _time

            from ..episodes import reflect_and_store
            files = [c.get("path") for c in changed if c.get("path")]
            task = self._episode_task or next(
                (m.get("content", "") for m in self._messages
                 if m.get("role") == "user"), "")
            outcome = "ok" if self._last_run_ok else "incomplete"

            def _complete(prompt: str) -> str:
                return self.backend.chat(
                    [{"role": "user", "content": prompt}], max_tokens=400) or ""

            reflect_and_store(
                self._episode_store, task=task, diff=self.session_diff(),
                outcome=outcome, files=files, turns=self.turns,
                complete=_complete, ts=_time.time())
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
        import re as _re
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

        _TC_RE = _re.compile(
            r"<tool_call>\s*(.*?)\s*</tool_call>", _re.DOTALL
        )

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
                call_matches = _TC_RE.findall(content)
                clean = _TC_RE.sub("", content).strip()

                if clean:
                    lines.append(f"**{self.name}**: {clean[:2000]}")
                elif call_matches:
                    lines.append(f"**{self.name}**:")

                for raw_json in call_matches:
                    try:
                        import json as _json
                        obj  = _json.loads(raw_json)
                        tool = obj.get("name", "?")
                        args = obj.get("args", {})
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
