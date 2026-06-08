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
    console,
    confirm,
    print_assistant_label,
    print_assistant_response,
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
from .prompts import build_system_prompt

# Tools that mutate files — trigger a project map refresh after they run
_MUTATING_TOOLS: frozenset[str] = frozenset({"write_file", "edit_file", "run_shell"})


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
        **gen_kwargs,
    ) -> None:
        self.backend      = backend
        self.cwd          = cwd
        self.name         = name
        self.max_turns    = max_turns
        self.verbose      = verbose
        self.auto_approve = auto_approve
        self.parent       = parent
        self.gen_kwargs   = gen_kwargs

        self._messages: list[dict] = []
        self._turns: int = 0
        self._total_tokens: int = 0
        self._project_map: ProjectMap = ProjectMap.build(cwd)
        self._memory: str = load_memory(cwd)
        self._system_prompt: str = build_system_prompt(
            cwd, agent_name=name, project_map=self._project_map, memory=self._memory
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

    def set_cwd(self, cwd: Path) -> None:
        self.cwd = cwd
        self._project_map = ProjectMap.build(cwd)
        self._memory = load_memory(cwd)
        self._system_prompt = build_system_prompt(
            cwd, agent_name=self.name, project_map=self._project_map, memory=self._memory
        )

    def reindex(self) -> int:
        """Rebuild the full project map and regenerate the system prompt."""
        self._project_map = ProjectMap.build(self.cwd)
        self._system_prompt = build_system_prompt(
            self.cwd, agent_name=self.name, project_map=self._project_map, memory=self._memory
        )
        return self._project_map.file_count()

    def reload_memory(self) -> str:
        """Re-read the memory file from disk and rebuild the system prompt."""
        self._memory = load_memory(self.cwd)
        self._system_prompt = build_system_prompt(
            self.cwd, agent_name=self.name, project_map=self._project_map, memory=self._memory
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
            return full
        else:
            # Silent call — used by sub-agents and non-interactive mode
            result = self.backend.chat(messages, **self.gen_kwargs)
            self._accumulate_usage()
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

        if interactive:
            print_tool_call(call.name, call.args)

        # Confirmation for destructive tools
        if tool_def.destructive and not self.auto_approve and interactive:
            if not confirm(f"  Allow {call.name}?"):
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

        if interactive:
            print_tool_result(call.name, result, verbose=self.verbose)

        # Incremental map refresh after file-mutating tools
        if result.ok and call.name in _MUTATING_TOOLS:
            self._refresh_map_for_tool(call)

        return result

    def _refresh_map_for_tool(self, call: ToolCall) -> None:
        """Update the project map for files touched by a write/edit tool call."""
        path_arg = call.args.get("path")
        if path_arg:
            abs_path = (self.cwd / path_arg).resolve()
            self._project_map.refresh_file(abs_path)
            # Regenerate system prompt with updated map
            self._system_prompt = build_system_prompt(
                self.cwd, agent_name=self.name, project_map=self._project_map
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
