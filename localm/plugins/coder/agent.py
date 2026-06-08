"""
Agent — the core agentic loop.

Flow per turn:
    1. Call the LLM with the current message history
    2. Parse the response for <tool_call> blocks
    3. If no tool calls → final answer, break
    4. For each tool call:
       a. Display it
       b. Optionally confirm (destructive tools + auto_approve=False)
       c. Execute
       d. Append result to messages as a user turn
    5. Repeat

The Agent class is used by:
    - CLI in interactive chat mode
    - CLI in single-task mode (run_task)
    - spawn_agent tool (child agents)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .backends.base import BaseLLMBackend
from .indexer import ProjectMap
from .memory import load_memory, remember, forget
from .parser import ToolCall, parse_tool_calls, split_response
from .tools import TOOL_REGISTRY, ToolResult
from .display import (
    confirm,
    confirm_diff,
    console,
    print_assistant_label,
    print_assistant_response,
    print_diff_preview,
    print_info,
    print_streaming_done,
    print_streaming_token,
    print_thinking,
    print_tool_call,
    print_tool_error,
    print_tool_result,
    print_turn_divider,
    print_warning,
)
from .audit import AuditLog, AuditLogT, NullAuditLog, SessionMode, make_audit_log
from .prompts import build_system_prompt

# Tools that mutate files — trigger a project map refresh after they run
_MUTATING_TOOLS: frozenset[str] = frozenset({"write_file", "edit_file", "run_shell"})

# Fraction of estimated context window at which compaction is triggered
_COMPACT_WARN_RATIO  = 0.70   # warn user in interactive mode
_COMPACT_AUTO_RATIO  = 0.90   # silently compact in non-interactive mode
_DEFAULT_CTX_TOKENS  = 4096   # fallback when n_ctx is unknown


# ---------------------------------------------------------------------------
#  Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    Stateful agentic loop.

    Parameters
    ----------
    backend:
        LLM backend (local or remote).
    cwd:
        Working directory for all file/shell operations.
    name:
        Display name (shown in the terminal and sub-agent logs).
    max_turns:
        Hard cap on LLM calls per task to prevent infinite loops.
    verbose:
        Print full tool outputs (not just summaries).
    auto_approve:
        Skip confirmation prompts for destructive tools.
    parent:
        Parent Agent when this instance is a sub-agent.
    gen_kwargs:
        Extra kwargs forwarded to every LLM call (temperature, max_tokens, …).
    """

    def __init__(
        self,
        backend: BaseLLMBackend,
        cwd: Path,
        name: str = "localcoder",
        max_turns: int = 40,
        verbose: bool = False,
        auto_approve: bool = True,
        parent: Optional["Agent"] = None,
        mode: SessionMode = SessionMode.PRIVACY,
        **gen_kwargs,
    ) -> None:
        self.backend      = backend
        self.cwd          = cwd
        self.name         = name
        self.max_turns    = max_turns
        self.verbose      = verbose
        self.auto_approve = auto_approve
        self.parent       = parent
        self.mode         = mode
        self.gen_kwargs   = gen_kwargs

        self._messages: list[dict] = []
        self._turns: int = 0
        self._total_tokens: int = 0
        self._compact_warned: bool = False
        self._model_name: str = getattr(backend, "model_id", "")
        self._audit: AuditLogT = make_audit_log(mode, label=name)
        self._project_map: ProjectMap = ProjectMap.build(cwd)
        self._memory: str = load_memory(cwd)
        self._system_prompt: str = build_system_prompt(
            cwd,
            agent_name=name,
            project_map=self._project_map,
            memory=self._memory,
            model_name=self._model_name,
        )

    @property
    def turns(self) -> int:
        return self._turns

    @property
    def total_tokens(self) -> int:
        """Cumulative token count across all LLM calls in this session (server estimate)."""
        return self._total_tokens

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Clear conversation history."""
        self._messages = []
        self._turns = 0
        self._total_tokens = 0
        self._compact_warned = False

    def set_cwd(self, cwd: Path) -> None:
        self.cwd = cwd
        self._project_map = ProjectMap.build(cwd)
        self._memory = load_memory(cwd)
        self._system_prompt = build_system_prompt(
            cwd,
            agent_name=self.name,
            project_map=self._project_map,
            memory=self._memory,
            model_name=self._model_name,
        )

    def reindex(self) -> int:
        """Rebuild the full project map and regenerate the system prompt."""
        self._project_map = ProjectMap.build(self.cwd)
        self._system_prompt = build_system_prompt(
            self.cwd,
            agent_name=self.name,
            project_map=self._project_map,
            memory=self._memory,
            model_name=self._model_name,
        )
        return self._project_map.file_count()

    def reload_memory(self) -> str:
        """Re-read the memory file from disk and rebuild the system prompt."""
        self._memory = load_memory(self.cwd)
        self._system_prompt = build_system_prompt(
            self.cwd,
            agent_name=self.name,
            project_map=self._project_map,
            memory=self._memory,
            model_name=self._model_name,
        )
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

    def run_task(self, task: str) -> str:
        """
        Run a single task to completion (non-interactive).

        Returns the agent's final text response.
        Used by spawn_agent and the CLI `run` command.
        """
        self._add_user(task)
        return self._loop(interactive=False)

    def chat(self, user_input: str) -> str:
        """
        Send one user message in an ongoing interactive session.

        History is preserved between calls.
        Returns the agent's final text response for this turn.
        """
        self._add_user(user_input)
        return self._loop(interactive=True)

    # ------------------------------------------------------------------ #
    #  Core loop
    # ------------------------------------------------------------------ #

    def _loop(self, interactive: bool) -> str:
        """
        Agentic loop: call LLM → parse tool calls → execute → repeat.
        Returns the final response text.
        """
        final_response = ""

        while self._turns < self.max_turns:
            self._turns += 1

            if interactive:
                print_turn_divider(self._turns, self._total_tokens)

            self._audit.set_turn(self._turns)

            # ---- context-budget check --------------------------------
            self._maybe_compact(interactive=interactive)

            # ---- call LLM -------------------------------------------
            messages = self._build_messages()
            response = self._call_llm(messages, interactive=interactive)

            # ---- parse tool calls ------------------------------------
            calls = parse_tool_calls(response)

            if not calls:
                # No tool calls → this is the final answer
                if interactive:
                    # response was already streamed; just ensure newline
                    pass
                else:
                    print_assistant_response(response, name=self.name)
                final_response = response
                self._add_assistant(response)
                break

            # ---- there are tool calls --------------------------------
            # Show the non-tool-call text parts first
            if interactive:
                segments = split_response(response, calls)
                for seg in segments:
                    if isinstance(seg, str) and seg.strip():
                        console.print(seg.strip())

            self._add_assistant(response)

            # Execute each tool and collect results
            result_blocks: list[str] = []
            for call in calls:
                result = self._execute_tool(call, interactive=interactive)
                result_blocks.append(result.to_xml(call.name))

            # Feed all results back as a user message
            combined = "\n\n".join(result_blocks)
            self._add_user(combined)

        else:
            msg = f"[max_turns={self.max_turns} reached]"
            print_warning(msg)
            final_response = msg

        return final_response

    # ------------------------------------------------------------------ #
    #  Compaction
    # ------------------------------------------------------------------ #

    def _ctx_window_tokens(self) -> int:
        """Estimated context window size in tokens."""
        try:
            from localm.config import load_config
            return load_config().get("n_ctx", _DEFAULT_CTX_TOKENS)
        except Exception:
            return _DEFAULT_CTX_TOKENS

    def _fill_ratio(self) -> float:
        """Fraction of estimated context window currently consumed (0.0 – 1.0+)."""
        estimated = self.context_chars() // 4
        return estimated / max(1, self._ctx_window_tokens())

    def compact(self) -> bool:
        """
        Summarise old conversation history into a single condensed exchange.

        Keeps the 4 most recent messages verbatim (= last 2 full turns) so
        the agent retains immediate context.  Everything older is replaced by
        a summary produced by a direct backend call (no tools, no loop).

        Returns True if compaction happened, False if there was nothing to compact.
        """
        return self._compact_history()

    def _compact_history(self) -> bool:
        keep_n = 4   # last 4 messages kept verbatim (~2 turns)
        if len(self._messages) <= keep_n:
            return False   # not enough history to compact

        older  = self._messages[:-keep_n]
        recent = self._messages[-keep_n:]

        # Build a concise conversation excerpt for the summariser
        excerpt_parts = []
        for m in older:
            role    = m["role"].upper()
            content = m.get("content", "")
            if isinstance(content, list):          # multipart messages
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            excerpt_parts.append(f"{role}: {content[:600]}")
        excerpt = "\n\n".join(excerpt_parts)

        summary_prompt = (
            "Produce a concise summary (≤300 words) of the following coding session. "
            "Focus on: decisions made, files created or edited, errors and fixes, "
            "and any open problems or next steps.\n\n"
            f"{excerpt}"
        )
        try:
            summary = self.backend.chat(
                [{"role": "user", "content": summary_prompt}],
                max_tokens=400,
            )
        except Exception:
            return False   # best-effort; don't crash on summary failure

        self._messages = [
            {"role": "user",      "content": f"[Session summary]\n{summary}"},
            {"role": "assistant", "content": "Understood. Continuing from this context."},
            *recent,
        ]
        return True

    def _maybe_compact(self, interactive: bool) -> None:
        """Check fill ratio and warn or auto-compact as appropriate."""
        ratio = self._fill_ratio()
        if interactive:
            if ratio >= _COMPACT_WARN_RATIO and not getattr(self, "_compact_warned", False):
                print_warning(
                    f"Context is ~{ratio:.0%} full. "
                    "Use [bold]/compact[/bold] to summarise old turns."
                )
                self._compact_warned = True   # warn once per session
        else:
            # Non-interactive (run_task): auto-compact silently at 90%
            if ratio >= _COMPACT_AUTO_RATIO:
                self._compact_history()

    # ------------------------------------------------------------------ #
    #  LLM call
    # ------------------------------------------------------------------ #

    def _call_llm(self, messages: list[dict], interactive: bool) -> str:
        if interactive:
            print_thinking()
            print_assistant_label(self.name)
            full = ""
            try:
                for piece in self.backend.chat_stream(messages, **self.gen_kwargs):
                    print_streaming_token(piece)
                    full += piece
                print_streaming_done()
            except KeyboardInterrupt:
                print_streaming_done()
                print_info("(interrupted)")
            self._accumulate_usage()
            self._audit.llm(full, tokens=self._total_tokens)
            return full
        else:
            # Silent call — used by sub-agents and non-interactive mode
            result = self.backend.chat(messages, **self.gen_kwargs)
            self._accumulate_usage()
            self._audit.llm(result, tokens=self._total_tokens)
            return result

    def _accumulate_usage(self) -> None:
        """Pull token counts from the backend's last call and add to the session total."""
        usage = getattr(self.backend, "last_usage", {})
        if usage.get("total_tokens"):
            self._total_tokens += usage["total_tokens"]

    # ------------------------------------------------------------------ #
    #  Tool execution
    # ------------------------------------------------------------------ #

    def _execute_tool(self, call: ToolCall, interactive: bool) -> ToolResult:
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

        # Confirmation for destructive tools — with diff preview for write_file
        if tool_def.destructive and not self.auto_approve and interactive:
            approved = self._confirm_tool(call)
            if not approved:
                result = ToolResult.error("Rejected by user.")
                print_tool_result(call.name, result, verbose=False)
                return result

        # Inject parent agent reference for spawn_agent
        args = dict(call.args)
        if call.name == "spawn_agent":
            args["_parent_agent"] = self

        try:
            result = tool_def.fn(self.cwd, **args)
        except TypeError as e:
            result = ToolResult.error(f"Bad arguments for {call.name}: {e}")
        except Exception as e:
            result = ToolResult.error(f"Tool error: {e}")

        self._audit.tool_result(call.name, result.ok, result.summary)
        if interactive:
            print_tool_result(call.name, result, verbose=self.verbose)

        # Incremental map refresh after file-mutating tools
        if result.ok and call.name in _MUTATING_TOOLS:
            self._refresh_map_for_tool(call)

        return result

    def _confirm_tool(self, call: ToolCall) -> bool:
        """
        Ask the user to approve a destructive tool call.

        For *write_file*, shows a coloured unified diff of the proposed change
        before the prompt so the user can see exactly what will happen.
        For all other destructive tools, falls back to a plain y/N prompt.
        """
        if call.name == "write_file":
            path_arg = call.args.get("path", "")
            new_content = call.args.get("content", "")
            abs_path = (self.cwd / path_arg).resolve() if path_arg else None

            # Read current content (empty string if file doesn't exist)
            old_content = ""
            if abs_path and abs_path.is_file():
                try:
                    old_content = abs_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    pass

            print_diff_preview(old_content, new_content, path_label=path_arg)
            return confirm_diff(path_arg or "file")

        return confirm(f"  Allow {call.name}?")

    def _refresh_map_for_tool(self, call: ToolCall) -> None:
        """Update the project map for files touched by a write/edit tool call."""
        path_arg = call.args.get("path")
        if path_arg:
            abs_path = (self.cwd / path_arg).resolve()
            self._project_map.refresh_file(abs_path)
            # Regenerate system prompt with updated map
            self._system_prompt = build_system_prompt(
                self.cwd,
                agent_name=self.name,
                project_map=self._project_map,
                memory=self._memory,
                model_name=self._model_name,
            )

    # ------------------------------------------------------------------ #
    #  Message management
    # ------------------------------------------------------------------ #

    def _build_messages(self) -> list[dict]:
        """Build the full message list with system prompt prepended."""
        return [
            {"role": "system", "content": self._system_prompt},
            *self._messages,
        ]

    def _add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})
        # Only log human-originating messages (skip tool results, which are very long)
        if not content.startswith("<tool_result"):
            self._audit.user(content)

    def _add_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    # ------------------------------------------------------------------ #
    #  Context stats
    # ------------------------------------------------------------------ #

    def context_chars(self) -> int:
        """Rough estimate of total characters in the current context."""
        total = len(self._system_prompt)
        for m in self._messages:
            total += len(m.get("content", ""))
        return total

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
        self._audit.close()
        if self.mode == SessionMode.FULL:
            return self._write_session_markdown()
        return None

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
            f"# localcoder Session — {ts_human}",
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
                # multipart — join text parts
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
                        lines.append(f"  - (tool call)")

                lines.append("")

            lines.append("---")
            lines.append("")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path
