# SPDX-License-Identifier: AGPL-3.0-or-later
"""The agentic loop: the public run/chat entry points, the turn loop itself, and
parallel tool dispatch. Mixed into Agent (see core.py)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import localm.plugins.coder.agent as _agent
from ..display import (
    console, print_assistant_response, print_info, print_tool_call,
    print_tool_error, print_tool_result, print_turn_divider,
)
from ..parser import looks_like_tool_attempt, split_response
from ..tools import ToolResult
from ..audit import SessionMode
from .constants import _MAX_TOOL_REPAIRS, _REPEAT_RESPONSE_ABORT


class _LoopMixin:
    def run_task(self, task: str) -> str:
        """
        Run a single task to completion (non-interactive).

        Returns the agent's final text response.
        Used by spawn_agent and the CLI `run` command.
        """
        if self._episodic and not self._episode_task:
            self._episode_task = task
        if not self._review_task:
            self._review_task = task   # remembered for the pre-done diff review
        self._add_user(self._with_episodes(task))
        return self._loop(interactive=False)

    def continue_task(self, message: str) -> str:
        """
        Continue the current session with another non-interactive instruction,
        preserving history.

        Unlike :meth:`run_task` it does not re-inject episodic recall (already
        injected on the first task). Used by the CLI goal loop to feed a
        verification failure back to the agent for another fix attempt.
        """
        self._add_user(message)
        return self._loop(interactive=False)

    def chat(self, user_input: str) -> str:
        """
        Send one user message in an ongoing interactive session.

        History is preserved between calls.
        Returns the agent's final text response for this turn.
        """
        # Recall relevant past lessons on the first turn of a session (the turn
        # that sets the session's task); later turns keep the same context.
        if self._episodic and not self._episode_task:
            self._episode_task = user_input
            user_input = self._with_episodes(user_input)
        if not self._review_task:
            self._review_task = user_input
        self._add_user(user_input)
        return self._loop(interactive=True)

    def _with_episodes(self, task: str) -> str:
        """Prepend relevant past lessons (episodic memory) to *task*, if any.
        Best-effort: a retrieval failure just returns the task unchanged."""
        if not self._episodic or self._episode_store is None:
            return task
        try:
            from ..episodes import render_for_prompt
            block = render_for_prompt(self._episode_store.search(task))
        except Exception:
            return task
        if not block:
            return task
        return block + "\n\n## Task\n" + task

    def _loop(self, interactive: bool) -> str:
        """
        Agentic loop: call LLM → parse tool calls → execute → repeat.
        Returns the final response text.
        """
        # Record whether this session owns a terminal it can prompt on. A spawned
        # child always runs _loop(interactive=False), so without this a child had
        # no way to tell an unattended run (nobody to ask -> fail closed) from a
        # parent sitting in the REPL (a user who can answer). REG-507.
        self._interactive = interactive
        # Live-attribute access so tests patching agent.parse_tool_calls /
        # confirm / print_warning / TOOL_REGISTRY are honoured (the names moved
        # into this submodule when agent.py became a package).
        parse_tool_calls = _agent.parse_tool_calls
        confirm = _agent.confirm
        print_warning = _agent.print_warning
        TOOL_REGISTRY = _agent.TOOL_REGISTRY
        final_response = ""
        self._stop_requested = False       # a stale stop must not kill a new task
        self._user_stopped = False         # per-run: a stop in an EARLIER run must
                                           # not mute this run's failure lesson
        start_turns = self._turns          # turns used by *this* task only
        budget_escalated = False           # uncertainty escalation fires at most once per task
        # Per-task one-shot flags for the no-tool-calls handler (split out below):
        # self-verification + pre-done review fire once each; repair re-prompts are
        # capped. Held on a namespace so the helper can persist them across turns.
        st = SimpleNamespace(verify_nudged=False, review_done=False, repair_count=0)

        try:
            while self._turns < self.max_turns:
                if self._stop_requested:
                    self._stop_requested = False
                    final_response = "[stopped by user]"
                    self._last_run_ok = False
                    self._user_stopped = True
                    break

                # Mid-task steering: deliver queued user messages before the
                # next LLM call so the agent reads them this turn.
                for queued in self._drain_queued():
                    self._add_user(
                        "[user steering note - read this before continuing, it "
                        f"overrides earlier instructions where they conflict]\n{queued}"
                    )
                    self._emit("info", text="steering note delivered to the agent")
                    if interactive:
                        print_info("(steering note delivered)")

                self._turns += 1
                prev_turn_tokens = self._last_turn_tokens
                self._last_turn_tokens = 0   # reset counter for this turn

                ctx_ratio = self._fill_ratio()
                if interactive:
                    print_turn_divider(self._turns, self._total_tokens,
                                       prev_turn_tokens, ctx_ratio=ctx_ratio)
                self._emit("turn", turn=self._turns,
                           total_tokens=self._total_tokens,
                           ctx_ratio=round(min(ctx_ratio, 1.0), 3))

                self._audit.set_turn(self._turns)

                # ---- uncertainty escalation ------------------------------
                # When the task exceeds its turn budget, stop guessing:
                # interactively ask the user whether to keep going; in
                # non-interactive mode tell the model to surface blockers.
                task_turns = self._turns - start_turns
                if not budget_escalated and task_turns > self.turn_budget:
                    budget_escalated = True
                    if interactive:
                        print_warning(
                            f"This task has used {task_turns} turns "
                            f"(budget: {self.turn_budget})."
                        )
                        if not confirm("  Keep going?"):
                            final_response = (
                                f"[stopped by user after {task_turns} turns - "
                                "task exceeded its turn budget]"
                            )
                            self._last_run_ok = False
                            self._user_stopped = True
                            break
                    else:
                        self._emit(
                            "info",
                            text=f"Turn budget exceeded ({task_turns}/{self.turn_budget}) - "
                                 "asking the agent to surface blockers instead of guessing.",
                        )
                        self._add_user(
                            f"[turn budget] You have used {task_turns} turns on this "
                            f"task (budget: {self.turn_budget}). If you are stuck or "
                            "uncertain, STOP guessing: summarise what you tried, state "
                            "exactly what is blocking you, and ask for guidance instead "
                            "of continuing to experiment. If you are genuinely close to "
                            "done, finish with the minimal remaining steps."
                        )

                # ---- context-budget check --------------------------------
                self._maybe_compact(interactive=interactive)

                # ---- call LLM -------------------------------------------
                messages = self._build_messages()
                response = self._call_llm(messages, interactive=interactive)

                if self._stop_requested:
                    # Stopped mid-generation: keep the partial text, run nothing
                    self._stop_requested = False
                    self._add_assistant(response)
                    final_response = response or "[stopped by user]"
                    self._last_run_ok = False
                    self._user_stopped = True
                    break

                # ---- repeated-scaffold breaker (REC-CODER-LOOPBREAK) ------
                # A stuck model can emit the SAME response over and over (the
                # "Message 1..4 / I will now wait" narration) making no progress.
                # The error-streak breakers only catch FAILED tool calls; this
                # catches identical NON-failing repetition. Abort after N in a row.
                fp = (response or "").strip()
                if fp and fp == self._last_response_fp:
                    self._repeat_response_count += 1
                else:
                    self._repeat_response_count = 0
                    self._last_response_fp = fp
                if self._repeat_response_count >= _REPEAT_RESPONSE_ABORT - 1:
                    final_response = (
                        "[circuit breaker: the model repeated the same response "
                        f"{self._repeat_response_count + 1} times with no progress - "
                        "stopping so you can adjust the approach instead of burning "
                        "more turns. The conversation is intact.]")
                    print_warning(final_response)
                    self._emit("info", text=final_response)
                    self._add_assistant(response)
                    self._last_run_ok = False
                    break

                # ---- parse tool calls ------------------------------------
                # Pass the known tool names so the lenient, name-gated formats
                # (bare JSON and ```json / bare fences) are recognised without
                # mistaking a JSON example in prose for a call.
                calls = parse_tool_calls(
                    response, tool_names=set(TOOL_REGISTRY) - self.disabled_tools)

                if not calls:
                    should_break, fr = self._handle_no_tool_calls(
                        response, interactive, st)
                    if should_break:
                        final_response = fr
                        break
                    continue

                # ---- there are tool calls --------------------------------
                # Show the non-tool-call text parts first
                if interactive:
                    segments = split_response(response, calls)
                    for seg in segments:
                        if isinstance(seg, str) and seg.strip():
                            console.print(seg.strip())

                self._add_assistant(response)

                # Execute tools - run non-destructive batches in parallel
                result_blocks = self._execute_tools(calls, interactive=interactive)

                # Feed all results back as a user message, compressing large
                # outputs when the context is filling up
                result_blocks = self._compress_results(result_blocks)
                combined = "\n\n".join(result_blocks)
                self._add_user(combined)

                breaker_msg = self._check_post_batch_breakers()
                if breaker_msg is not None:
                    final_response = breaker_msg
                    break

            else:
                msg = f"[max_turns={self.max_turns} reached]"
                print_warning(msg)
                final_response = msg
                self._last_run_ok = False

        except KeyboardInterrupt:
            if interactive and self._messages:
                if self.mode == SessionMode.PRIVACY:
                    print_info(
                        "(interrupted - privacy mode, no checkpoint saved; "
                        "/resume is unavailable.)"
                    )
                else:
                    self.save_checkpoint()
                    print_info(
                        "(interrupted - progress saved. "
                        "Type /resume to continue or start a new task.)"
                    )
            raise

        else:
            # Clean finish - discard any stale checkpoint
            self.clear_checkpoint()

        return final_response

    def _handle_no_tool_calls(self, response, interactive, st) -> "tuple[bool, str]":
        """Handle a turn that produced no tool calls (split out of _loop).

        Runs, in order: the one-shot self-verification nudge, the tool-call repair
        re-prompt (capped), the give-up surface for an unparseable attempt, the
        one-shot pre-done reviewer pass, and finally accepting the response as the
        final answer. ``st`` carries the per-task one-shot flags (verify_nudged,
        repair_count, review_done) so they persist across turns.

        Returns ``(should_break, final_response)``: ``(False, "")`` to continue the
        loop, ``(True, text)`` to end it with that final response."""
        # Self-verification: don't accept a final answer while code
        # changes sit unverified - nudge the agent to check its work.
        # Fires at most once per task to avoid infinite loops.
        if (
            self.self_verify
            and not st.verify_nudged
            and self._unverified_writes
            and self._turns < self.max_turns
        ):
            st.verify_nudged = True
            files = ", ".join(sorted(self._unverified_writes))
            self._add_assistant(response)
            self._add_user(
                f"[self-verification] You changed code files ({files}) "
                "but have not verified them. Before giving your final "
                "answer: run run_tests if this project has a test "
                "suite, otherwise re-read the changed files to check "
                "for mistakes. Then give your final answer."
            )
            if interactive:
                print_info(
                    "(self-verification: asking agent to verify "
                    f"changes to {files})"
                )
            return (False, "")

        # Repair turn: the response looks like a tool call that
        # failed to parse (a marker/fence, or a name+args object the
        # lenient parser still could not recover - malformed JSON,
        # an unknown tool name, or Python-style tool_code). Re-prompt
        # once with the exact format instead of printing the broken
        # call as the final answer.
        if (
            st.repair_count < _MAX_TOOL_REPAIRS
            and self._turns < self.max_turns
            and looks_like_tool_attempt(response)
        ):
            st.repair_count += 1
            self._add_assistant(response)
            self._add_user(
                "[tool-call format] That looked like a tool call, "
                "but I could not parse it. Re-emit it in EXACTLY "
                "this format and nothing else:\n"
                "<tool_call>\n"
                '{"name": "TOOL_NAME", "args": {"PARAM": "VALUE"}}\n'
                "</tool_call>\n"
                "The args must be valid JSON: use a single double-quoted "
                'string with \\n escapes for multi-line "content" - NOT a '
                "Python triple-quoted string. Use one of the available "
                "tools by its exact name. If you did NOT mean to call a "
                "tool, ignore this and give your final answer as plain text."
            )
            if interactive:
                print_info(
                    "(re-prompting: tool call could not be parsed)"
                )
            return (False, "")

        # Give-up case: it STILL looks like a tool call we could not parse
        # after the repair attempts. SURFACE the raw attempt instead of
        # finalising a hidden <tool_call> block - which the streaming
        # display hides, leaving an empty bubble + "task finished" + no
        # file (a silent no-op the user gets zero feedback on).
        if looks_like_tool_attempt(response):
            self._emit("info", text=(
                "the model tried to call a tool but emitted invalid output "
                f"{st.repair_count + 1} times - surfacing its raw attempt "
                "(nothing was run or written)"))
            notice = (
                "[I tried to call a tool but could not produce valid "
                "tool-call JSON after several attempts, so nothing was run "
                "or written. My raw attempt was:]\n\n" + response
            )
            if not interactive and self.on_event is None:
                print_assistant_response(notice, name=self.name)
            self._add_assistant(response)
            return (True, notice)

        # Pre-done review: before accepting the final answer, let a
        # reviewer model check the cumulative diff and feed any blocking
        # issues back for one more fix pass. Fires at most once per loop,
        # only when there is a real diff and turns remain. Fail-open: a
        # reviewer error never blocks the answer (review_feedback="").
        if (
            self._reviewer is not None
            and not st.review_done
            and self._turns < self.max_turns
        ):
            st.review_done = True
            diff = self.session_diff()
            if diff.strip():
                feedback = self._reviewer.review_feedback(
                    diff, self._review_task)
                if feedback:
                    self._add_assistant(response)
                    self._add_user(feedback)
                    self._emit("info", text=(
                        "self-review: the reviewer flagged issues - "
                        "asking the agent to address them"))
                    if interactive:
                        print_info(
                            "(self-review: reviewer flagged issues - "
                            "feeding them back)")
                    return (False, "")

        # No tool calls → this is the final answer
        if not interactive and self.on_event is None:
            print_assistant_response(response, name=self.name)
        self._add_assistant(response)
        return (True, response)

    def _check_post_batch_breakers(self) -> "str | None":
        """After a tool batch, return a circuit-breaker message (and mark the run
        not-ok + emit it) if a breaker tripped, else None. Split out of _loop."""
        print_warning = _agent.print_warning  # live: honour a patched agent.print_warning
        # Circuit breaker: a tool that keeps failing identically wastes
        # the whole turn budget - stop and hand control back instead.
        if self._abort_streak_tool:
            tool = self._abort_streak_tool
            self._abort_streak_tool = None
            streak = self._consecutive_errors.get(tool, 0)
            final_response = (
                f"[circuit breaker: {tool} failed {streak} times in a "
                "row - stopping so you can take a look instead of "
                "burning more turns. The conversation is intact; "
                "adjust the approach and continue.]"
            )
            print_warning(final_response)
            self._emit("info", text=final_response)
            self._last_run_ok = False
            return final_response

        # No-progress breaker: many tool calls failed in a row across ANY
        # tools (a weak model spinning on varied junk calls). Stop instead
        # of burning the whole budget.
        if self._abort_no_progress:
            self._abort_no_progress = False
            self._global_error_streak = 0
            final_response = (
                "[circuit breaker: many tool calls failed in a row with no "
                "progress - stopping so you can take a look instead of burning "
                "more turns. The conversation is intact; adjust and continue.]"
            )
            print_warning(final_response)
            self._emit("info", text=final_response)
            self._last_run_ok = False
            return final_response

        return None

    def _result_block(self, call, result) -> str:
        """The <tool_result> XML for a finished tool call, provenance-tagged.

        Results from untrusted (network / MCP) tools are re-framed as
        data-not-instructions with a hardened boundary (provenance.py); trusted
        tools keep the plain frame. The outer <tool_result> tag is preserved
        either way so the rest of the agent (audit / transcript skips) is
        unaffected. When the feature is off, every result uses the plain frame.
        """
        TOOL_REGISTRY = _agent.TOOL_REGISTRY  # live: honour a patched agent.TOOL_REGISTRY
        if not getattr(self, "_untrusted_provenance", True):
            return result.to_xml(call.name)
        from ..provenance import build_result_block, is_untrusted_tool
        untrusted = is_untrusted_tool(call.name, TOOL_REGISTRY.get(call.name))
        return build_result_block(call.name, result, untrusted)

    def _execute_tools(self, calls: list, interactive: bool) -> list[str]:
        """
        Execute a list of tool calls and return their XML result blocks.

        Groups consecutive non-destructive calls and runs each group in parallel
        with a ``ThreadPoolExecutor``.  Destructive calls are always run alone,
        in order, to avoid unintended interactions (file corruption, overlapping
        shell commands, etc.).

        The grouping strategy preserves the original ordering of destructive
        calls relative to non-destructive ones: given [read, read, write, read],
        the two leading reads run in parallel, then the write runs alone, then
        the final read runs alone.  This is conservative but safe.
        """
        TOOL_REGISTRY = _agent.TOOL_REGISTRY  # live: honour a patched agent.TOOL_REGISTRY
        result_blocks: list[str] = []

        # Split into segments: each segment is (is_destructive, [calls])
        segments: list[tuple[bool, list]] = []
        for call in calls:
            td = TOOL_REGISTRY.get(call.name)
            destructive = td.destructive if td else True
            if segments and segments[-1][0] == destructive:
                segments[-1][1].append(call)
            else:
                segments.append((destructive, [call]))

        for destructive, group in segments:
            if destructive or len(group) == 1:
                # Serial execution
                for call in group:
                    result = self._execute_tool(call, interactive=interactive)
                    result_blocks.append(self._result_block(call, result))
            else:
                # Parallel execution for non-destructive batch. The pool is
                # shut down without waiting so one hung tool (network fetch,
                # slow disk) cannot block the whole batch past the deadline -
                # the stuck thread is abandoned and reported as a timeout.
                ordered: dict[int, str] = {}
                pool = ThreadPoolExecutor(max_workers=min(len(group), 8))
                futures = {
                    pool.submit(self._execute_tool, call, False): (i, call)
                    for i, call in enumerate(group)
                }
                try:
                    for fut in as_completed(futures,
                                            timeout=self._PARALLEL_BATCH_TIMEOUT_S):
                        i, call = futures[fut]
                        try:
                            result = fut.result()
                        except Exception as exc:
                            result = ToolResult.error(f"Parallel execution error: {exc}")
                        # Print results in original order when interactive
                        ordered[i] = self._result_block(call, result)
                        if interactive:
                            print_tool_call(call.name, call.args)
                            print_tool_result(call.name, result, verbose=self.verbose)
                except TimeoutError:
                    for fut, (i, call) in futures.items():
                        if i not in ordered:
                            fut.cancel()
                            result = ToolResult.error(
                                f"{call.name} did not finish within "
                                f"{self._PARALLEL_BATCH_TIMEOUT_S}s (parallel "
                                "batch timeout) - try a narrower target."
                            )
                            ordered[i] = self._result_block(call, result)
                            if interactive:
                                print_tool_error(call.name, result.output)
                finally:
                    pool.shutdown(wait=False)
                for i in range(len(group)):
                    result_blocks.append(ordered[i])

        return result_blocks

    # Wall-clock deadline for one parallel batch of non-destructive tools
    _PARALLEL_BATCH_TIMEOUT_S = 120
