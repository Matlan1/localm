# SPDX-License-Identifier: AGPL-3.0-or-later
"""The agentic loop: the public run/chat entry points, the turn loop itself, and
parallel tool dispatch. Mixed into Agent (see core.py)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import localm.plugins.coder.agent as _agent
from ..display import (
    console, print_assistant_response, print_info, print_success,
    print_tool_call, print_tool_error, print_tool_result, print_turn_divider,
)
from ..parser import looks_like_tool_attempt, split_response
from ..tools import ToolResult
from ..audit import SessionMode
from .constants import (
    _ACTION_VERBS, _MAX_NOCALL_ESCALATIONS, _MAX_TOOL_REPAIRS,
    _REPEAT_HISTORY_MAX, _REPEAT_RESPONSE_ABORT, _REPEAT_SIMILARITY,
    _SKILL_STATE_TOOLS, _WORKSPACE_HINT,
)

_RE_WORKSPACE = None      # compiled on first use


def implies_action(text: str) -> bool:
    """True when *text* asks for something that needs a TOOL, not an
    explanation - the precondition for escalating a turn that produced no tool
    call.

    Two independent signals, either of which is enough: an imperative action
    verb (``_ACTION_VERBS``), or a reference to this workspace - a path, a
    filename with an extension, or a project noun (``_WORKSPACE_HINT``). Read
    verbs count: "show me what is in config.py" needs read_file exactly as much
    as "write config.py" needs write_file.

    THE BAR IS LOW, and leans toward firing. A false POSITIVE costs one extra
    turn whose re-prompt states that a plain answer is acceptable if no tool is
    needed, so the model can decline and the loop finishes normally. A false
    NEGATIVE silently answers the request with prose and does nothing.

    Pure and module-level, so it can be tested directly on request strings
    without constructing an Agent."""
    global _RE_WORKSPACE
    if not text:
        return False
    lowered = text.lower()
    import re
    if _RE_WORKSPACE is None:
        _RE_WORKSPACE = re.compile(_WORKSPACE_HINT, re.IGNORECASE)
    for word in re.findall(r"[a-z]+", lowered):
        if word in _ACTION_VERBS:
            return True
    return bool(_RE_WORKSPACE.search(text))


def response_similarity(a: str, b: str) -> float:
    """difflib ratio of two responses, whitespace-normalised. 1.0 is identical."""
    import difflib
    return difflib.SequenceMatcher(None, " ".join(a.split()),
                                   " ".join(b.split())).ratio()


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
        if not self._session_title:
            self._session_title = task   # resume-listing display title
        # The raw request, before _with_episodes can prepend a lessons preamble.
        self._last_user_request = task
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
        self._last_user_request = message
        self._add_user(message)
        return self._loop(interactive=False)

    def chat(self, user_input: str) -> str:
        """
        Send one user message in an ongoing interactive session.

        History is preserved between calls.
        Returns the agent's final text response for this turn.
        """
        if not self._session_title:
            # The raw input, before _with_episodes can prepend a lessons preamble.
            self._session_title = user_input
        # Recall relevant past lessons on the first turn of a session (the turn
        # that sets the session's task); later turns keep the same context.
        self._last_user_request = user_input   # raw, see run_task
        if self._episodic and not self._episode_task:
            self._episode_task = user_input
            user_input = self._with_episodes(user_input)
        if not self._review_task:
            self._review_task = user_input
        self._add_user(user_input)
        return self._loop(interactive=True)

    def _with_episodes(self, task: str) -> str:
        """Prepend relevant past lessons (episodic memory) to *task*, if any.
        Best-effort: a retrieval failure just returns the task unchanged.

        Also RECORDS which lessons were injected (id + text), on the agent, on the
        event stream, and in the audit trail, so a lesson that steered a run badly
        is identifiable afterwards and can be forgotten by id."""
        if not self._episodic or self._episode_store is None:
            return task
        try:
            from ..episodes import render_for_prompt
            episodes = self._episode_store.search(task)
            block = render_for_prompt(episodes)
        except Exception as e:
            # A failed recall is recorded rather than silently ignored; the task
            # still runs.
            self._record_episodes_used([], reason="recall failed: %s" % e)
            return task
        if not block:
            self._record_episodes_used([], reason="no relevant past lesson")
            return task
        self._record_episodes_used(episodes)
        return block + "\n\n## Task\n" + task

    def _record_episodes_used(self, episodes: list, reason: str = "") -> None:
        """Stash + surface the recalled lessons for this run.

        Mirrors the chat plugin's ``_stash_memory_used``: metadata plus the lesson
        text that was ALREADY injected into the prompt, so this adds no disclosure
        beyond what the prompt itself carries (and the audit log already records
        the prompt). Best-effort and side-effect free - a surfacing failure must
        never break the run."""
        try:
            used = [{"id": e.id, "outcome": e.outcome,
                     "lesson": (e.lesson or e.summary or "")[:200]}
                    for e in episodes]
            self._episodes_used = used
            self._episodes_degrade_reason = reason
            from localm.debuglog import logger
            logger.debug("episodic recall: injected %d lesson(s)%s", len(used),
                         (" (%s)" % reason) if reason else "")
            if used:
                self._audit.episodes_recalled(used)
                self._emit("episodes_recalled", episodes=used)
        except Exception as e:
            from localm.debuglog import logger
            logger.debug("episodic recall surfacing skipped: %s", e)

    def _loop(self, interactive: bool) -> str:
        """
        Agentic loop: call LLM → parse tool calls → execute → repeat.
        Returns the final response text.
        """
        # Record whether this session owns a terminal it can prompt on. A spawned
        # child always runs _loop(interactive=False).
        self._interactive = interactive
        # Live attribute lookup, so a patched agent.parse_tool_calls / confirm /
        # print_warning / TOOL_REGISTRY is honoured.
        parse_tool_calls = _agent.parse_tool_calls
        confirm = _agent.confirm
        print_warning = _agent.print_warning
        TOOL_REGISTRY = _agent.TOOL_REGISTRY
        final_response = ""
        self._stop_requested = False       # a stale stop must not kill a new task
        self._user_stopped = False         # per-run: an earlier run's stop does not
                                           # mute this run's failure lesson
        self._last_run_ok = True           # per-run: an earlier run's failure does not
                                           # label this run failed. _had_any_failure is
                                           # the session-wide flag, set in the finally.
        self._last_verify_state = None     # per-run: this run is unverified until
                                           # its own gate says otherwise
        start_turns = self._turns          # turns used by *this* task only
        budget_escalated = False           # uncertainty escalation fires at most once per task
        # Per-task one-shot flags for the no-tool-calls handler: self-verification
        # and pre-done review fire once each, repair re-prompts are capped. Held on
        # a namespace so the helper can persist them across turns. The verify_*
        # fields drive the exit-code oracle gate (_run_verify_gate);
        # verify_checked_at is the write count the check last passed at, so the
        # gate fires only once this task has written something since.
        st = SimpleNamespace(verify_nudged=False, review_done=False, repair_count=0,
                             verify_retries=0, verify_settled=False,
                             verify_checked_at=self._write_total(),
                             partial_notice_count=0, partial_notice_cap_announced=False,
                             # Zero-tool-call escalation: how many rungs of the
                             # ladder this task has used, and whether the model
                             # has produced any call yet.
                             nocall_escalation=0, tool_calls_made=0,
                             writes_at_start=self._write_total())

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

                # Absorb background sub-agents that finished since the last turn,
                # on this thread only, at the same point as steering notes so the
                # model reads the result before its next call.
                for note in self._drain_background_agents():
                    self._add_user(f"[background sub-agent result]\n{note}")
                    self._emit("info", text="background sub-agent result absorbed")
                    if interactive:
                        print_info("(background sub-agent finished)")

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
                # Past the turn budget: interactively ask the user whether to keep
                # going; non-interactively tell the model to surface blockers.
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
                # One-shot: the forcing grammar applies only to the turn the
                # ladder armed it for. Cleared here, next to the single call that
                # consumes it, so no later path inherits a constrained sampler.
                self._force_tool_grammar = False

                if self._stop_requested:
                    # Stopped mid-generation: keep the partial text, run nothing
                    self._stop_requested = False
                    self._add_assistant(response)
                    final_response = response or "[stopped by user]"
                    self._last_run_ok = False
                    self._user_stopped = True
                    break

                # ---- parse tool calls ------------------------------------
                # Pass the known tool names so the name-gated formats (bare JSON
                # and json / bare fences) are recognised.
                tool_names = set(TOOL_REGISTRY) - self.disabled_tools
                calls = parse_tool_calls(response, tool_names=tool_names)

                # ---- repeated-scaffold breaker ---------------------------
                # Counts a turn whose response is SIMILAR (not identical) to any
                # turn in a bounded history AND whose tool-call signature matches
                # that turn's. Both conditions must hold. Matching against the
                # whole history, not just the previous turn, also catches an
                # A-B-A-B alternation.
                fp = (response or "").strip()
                sig = " | ".join(
                    f"{c.name}({sorted((c.args or {}).items())!r})" for c in calls)
                if fp:
                    history = getattr(self, "_recent_finals", None)
                    if history is None:
                        history = self._recent_finals = []
                    match = max((response_similarity(fp, prev_fp)
                                 for prev_fp, prev_sig in history
                                 if prev_sig == sig), default=0.0)
                    if match >= _REPEAT_SIMILARITY:
                        self._repeat_response_count += 1
                    else:
                        self._repeat_response_count = 0
                    history.append((fp, sig))
                    del history[:-_REPEAT_HISTORY_MAX]
                    self._last_response_fp = fp
                if self._repeat_response_count >= _REPEAT_RESPONSE_ABORT - 1:
                    final_response = (
                        "[circuit breaker: the model repeated essentially the "
                        f"same response {self._repeat_response_count + 1} times "
                        "with no progress - stopping so you can adjust the "
                        "approach instead of burning more turns. The "
                        "conversation is intact.]")
                    print_warning(final_response)
                    self._emit("info", text=final_response)
                    self._add_assistant(response)
                    self._last_run_ok = False
                    break

                if not calls:
                    should_break, fr = self._handle_no_tool_calls(
                        response, interactive, st)
                    if should_break:
                        final_response = fr
                        break
                    continue

                # ---- there are tool calls --------------------------------
                # Recorded before execution: the escalation ladder asks whether
                # the model can produce a CALL at all, not whether it succeeded.
                st.tool_calls_made += len(calls)

                # Show the non-tool-call text parts first
                segments = split_response(response, calls)
                if interactive:
                    for seg in segments:
                        if isinstance(seg, str) and seg.strip():
                            console.print(seg.strip())

                # The event-sink (GUI) surface streamed the RAW response live,
                # before parse_tool_calls could know which spans were real calls,
                # and its live hider (context.py) only recognises the
                # <tool_call>/<|tool_call> wrappers. `calls`/`segments` above are
                # the source of truth for what was consumed as a call, so re-send
                # the real leftover text and the GUI replaces what it streamed,
                # for every call shape. A no-op when nothing leaked.
                self._emit("assistant_text",
                          text="".join(seg for seg in segments if isinstance(seg, str)))

                self._add_assistant(response)

                # Execute tools - run non-destructive batches in parallel
                result_blocks = self._execute_tools(calls, interactive=interactive)

                # parse_tool_calls silently drops a tool-call-shaped block whose
                # body could not be recovered, and the only downstream check
                # (_handle_no_tool_calls) fires solely when NO call parsed.
                # `segments` has the successfully-parsed call spans excised, so
                # anything tool-call-shaped left in it is what failed to parse:
                # tell the model that part was NOT run.
                #
                # Capped at _MAX_TOOL_REPAIRS: the notice's own example text is
                # itself tool-call-shaped, so a model echoing it back would
                # re-trigger the notice every turn.
                leftover = "".join(seg for seg in segments if isinstance(seg, str))
                if looks_like_tool_attempt(leftover, tool_names):
                    if st.partial_notice_count < _MAX_TOOL_REPAIRS:
                        st.partial_notice_count += 1
                        result_blocks.append(
                            "[tool-call format] Part of this response looked "
                            "like another tool call, but it could not be "
                            "parsed (the JSON body was likely malformed) - "
                            "it was NOT run, unlike the call(s) above. If you "
                            "still need it, re-emit just that one call in "
                            'the exact <tool_call>\n{"name": "TOOL_NAME", '
                            '"args": {...}}\n</tool_call> format.'
                        )
                    elif not st.partial_notice_cap_announced:
                        # The cap bounds repetition, not visibility: one final
                        # notice naming the change, then a debug trace for every
                        # further occurrence.
                        st.partial_notice_cap_announced = True
                        result_blocks.append(
                            "[tool-call format] Another part of this response "
                            "looked like an unparseable tool call, same as "
                            "the earlier notice(s) this task - further "
                            "occurrences will not be reported individually "
                            "from here on, but each one is still a call that "
                            "did NOT run."
                        )
                        from localm.debuglog import logger
                        logger.debug(
                            "partial tool-call parse failures capped at %d "
                            "notices this task; further drops logged only",
                            _MAX_TOOL_REPAIRS)
                    else:
                        from localm.debuglog import logger
                        logger.debug(
                            "tool-call parse likely dropped a call (leftover "
                            "text still looks like an attempt); notice cap "
                            "already reached, not reported to the model "
                            "this turn")

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

        finally:
            # Fold this run's outcome into the session-level flag before the next
            # run re-arms the per-run one, so every failure path (max_turns,
            # either circuit breaker) is captured in one place. A user stop is
            # excluded: _last_run_ok and _user_stopped are per-run and re-armed
            # above, while _had_any_failure is session-level and cleared only by
            # reset().
            if not self._last_run_ok and not self._user_stopped:
                self._had_any_failure = True

        return final_response

    def _handle_no_tool_calls(self, response, interactive, st) -> "tuple[bool, str]":
        """Handle a turn that produced no tool calls (split out of _loop).

        Runs, in order: the harness-run exit-code oracle (when a verify command is
        configured), the one-shot self-verification nudge, the tool-call repair
        re-prompt (capped), the give-up surface for an unparseable attempt, the
        one-shot pre-done reviewer pass, and finally accepting the response as the
        final answer. ``st`` carries the per-task one-shot flags (verify_nudged,
        repair_count, review_done, verify_retries/verify_settled) so they persist
        across turns.

        Returns ``(should_break, final_response)``: ``(False, "")`` to continue the
        loop, ``(True, text)`` to end it with that final response."""
        # Live attribute lookup, so a patched agent.TOOL_REGISTRY is honoured;
        # disabled_tools can differ per agent.
        tool_names = set(_agent.TOOL_REGISTRY) - self.disabled_tools

        # The exit-code oracle runs FIRST among the finish gates, before the
        # self-graded nudge and the reviewer, so a failing check goes straight
        # back as a fix request. Skipped for a response that only looks like a
        # broken tool call - the repair turn below handles that.
        if not looks_like_tool_attempt(response, tool_names):
            gated = self._run_verify_gate(response, interactive, st)
            if gated is not None:
                return gated

        # Self-verification: nudge the agent to check its work while code changes
        # sit unverified. Fires at most once per task.
        if (
            self.self_verify
            and not st.verify_nudged
            and self._unverified_writes
            and self._turns < self.max_turns
        ):
            st.verify_nudged = True
            files = ", ".join(sorted(self._unverified_writes))
            self._add_assistant(response)
            # Name the project's real check when one is known, so the nudge asks
            # for a command run rather than a re-read.
            if self.verify_cmd is not None:
                from ..verify import command_text
                how = (f"run `{command_text(self.verify_cmd)}` (via run_shell, "
                       "or run_tests if it is the project's test command)")
            else:
                how = ("run run_tests if this project has a test suite, "
                       "otherwise re-read the changed files to check for mistakes")
            self._add_user(
                f"[self-verification] You changed code files ({files}) "
                "but have not verified them. Before giving your final "
                f"answer: {how}. Then give your final answer."
            )
            if interactive:
                print_info(
                    "(self-verification: asking agent to verify "
                    f"changes to {files})"
                )
            return (False, "")

        # Repair turn: the response looks like a tool call that failed to parse
        # (a marker/fence, or a name+args object the lenient parser could not
        # recover). Re-prompt once with the exact format.
        if (
            st.repair_count < _MAX_TOOL_REPAIRS
            and self._turns < self.max_turns
            and looks_like_tool_attempt(response, tool_names)
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

        # Still looks like an unparseable tool call after the repair attempts:
        # surface the raw attempt instead of finalising a hidden <tool_call>
        # block, which the streaming display hides.
        if looks_like_tool_attempt(response, tool_names):
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

        # Zero-attempt escalation: the model produced nothing tool-shaped on a
        # request that needs a tool. Every branch above is reached only via
        # looks_like_tool_attempt().
        escalated = self._escalate_no_tool_attempt(response, interactive, st)
        if escalated is not None:
            return escalated

        # Pre-done review: a reviewer model checks the cumulative diff and any
        # blocking issues go back for one more fix pass. Fires at most once per
        # loop, only when there is a real diff and turns remain. Fail-open: a
        # reviewer error never blocks the answer.
        if (
            self._reviewer is not None
            and not st.review_done
            and self._turns < self.max_turns
        ):
            st.review_done = True
            diff = self.session_diff()
            if diff.strip():
                feedback = self._run_pre_done_review(diff)
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

        # No tool calls, so this is the final answer; _grounding_footer grounds it
        # in what the session actually recorded.
        # Rung 3, and only here: the ladder ran and the model still never produced
        # a call. Reached when the rungs are exhausted, when forcing is
        # unavailable, or when turns ran out mid-ladder.
        enforcement = ""
        if st.nocall_escalation and not self._used_tools_this_task(st):
            why = ("" if self.can_force_tool_calls() else
                   " Constrained sampling, which would have forced one, is not "
                   "available here (this backend cannot enforce a grammar, or "
                   "coder_tool_grammar is off in config).")
            enforcement = (
                "\n\n[tool use not achieved: this model was asked "
                f"{st.nocall_escalation} more time(s) to call a tool and did not, "
                f"so nothing was run or written.{why}]")
            self._audit.notice(
                "no_tool_call",
                f"escalation exhausted after {st.nocall_escalation} attempt(s); "
                "no tool call was produced")
            self._emit("info", text=(
                "the model did not call any tool despite being asked again "
                "- nothing was run or written"))

        footer = self._grounding_footer()
        final_text = response + enforcement + (footer or "")
        if not interactive and self.on_event is None:
            print_assistant_response(final_text, name=self.name)
        self._add_assistant(response)
        return (True, final_text)

    def _used_tools_this_task(self, st) -> bool:
        """Has the model demonstrated, THIS TASK, that it can drive a tool?

        Three independent pieces of evidence, any one of which settles it. Only
        the first is about parsing; the other two are about what the harness
        actually recorded happening, which is the same grounding rule
        _grounding_footer follows - never the model's own account of itself.

        Both artifact checks are needed, and neither subsumes the other: a
        write is recorded in _unverified_writes the moment it lands, while
        _write_total() is the cumulative counter, snapshotted at task start so
        an EARLIER task's writes in a long REPL session cannot be mistaken for
        this one's. A task that has written something is self-evidently not a
        task in which the model refuses to act, and escalating at it would
        nag a model that is working."""
        return bool(st.tool_calls_made
                    or self._unverified_writes
                    or self._write_total() > st.writes_at_start)

    def _escalate_no_tool_attempt(self, response, interactive, st):
        """Escalate a turn that produced NO tool call and no attempt at one, on a
        request that needs a tool.

        Returns None to fall through to the remaining gates, or the same
        ``(should_break, final_response)`` pair the caller propagates.

        THE POINT IS TO MAKE THE CALL HAPPEN, NOT TO REPORT THAT IT DID NOT.
        Reporting already existed (_grounding_footer's "no files changed") and
        it left the user with a model that never touched a tool across six
        turns. The rungs, in order:

          1. Re-prompt with the exact format block - the same treatment the
             MALFORMED case has always had, triggered by ABSENCE instead. Some
             models simply never saw the wrapper in a prompt this long and
             produce a correct call the moment it is shown to them again.
          2. Re-run the turn with the tool-call grammar bound from the FIRST
             token (see context.can_force_tool_calls). At this point the
             sampler cannot emit anything except an optional reasoning block
             and a structurally valid call, so the model's willingness stops
             being the deciding factor.
          3. Only once forcing has actually been tried and still failed - or is
             genuinely unavailable on this backend - tell the user. That
             message is a report of FAILED ENFORCEMENT, never a suggestion to
             pick a different model: which model to run is the user's choice
             and this code's job is to make their choice work.

        Deliberately NOT gated on the response's wording. Every phrasing-based
        check inherits the unreliability of the self-report it is reading (see
        _grounding_footer); "did the harness parse a call" is an observable
        fact about this turn and "does the request need one" is a fact about
        the user's own text, so neither can be talked past."""
        if self._used_tools_this_task(st):
            return None                      # this model calls tools fine
        if not implies_action(getattr(self, "_last_user_request", "") or ""):
            return None                      # a question, not an action
        if self._turns >= self.max_turns:
            return None                      # no turns left to escalate into
        if st.nocall_escalation >= _MAX_NOCALL_ESCALATIONS:
            return None                      # ladder exhausted; caller reports below

        st.nocall_escalation += 1
        rung = st.nocall_escalation

        if rung == 1:
            self._audit.notice(
                "no_tool_call",
                "model produced no tool call and no attempt on an action request "
                "- re-prompting with the tool-call format")
            self._add_assistant(response)
            self._add_user(
                "[no tool call] That request needs you to USE a tool - I run the "
                "tools on this machine for you, and nothing was run or written, "
                "so the task is not done. Do not describe the steps, do not hand "
                "back a script for me to run: emit the call itself, in EXACTLY "
                "this format:\n"
                "<tool_call>\n"
                '{"name": "TOOL_NAME", "args": {"PARAM": "VALUE"}}\n'
                "</tool_call>\n"
                "Use one of the available tools by its exact name, with valid "
                "JSON args. Emit one call now and I will run it and give you the "
                "result. If this genuinely needs no tool at all, say so in one "
                "sentence and I will accept that as your answer."
            )
            if interactive:
                print_info("(no tool call: re-prompting with the tool-call format)")
            self._emit("info", text=(
                "the model answered without using a tool - asking it again "
                "with the tool-call format"))
            return (False, "")

        # Rung 2: force it at the sampler.
        if not self.can_force_tool_calls():
            self._audit.notice(
                "no_tool_call",
                "re-prompt did not produce a tool call and grammar forcing is "
                "unavailable on this backend/config")
            return None                      # caller's own branch reports it
        self._force_tool_grammar = True
        self._audit.notice(
            "no_tool_call",
            "re-prompt did not produce a tool call - re-running the turn with "
            "the tool-call grammar bound from the first token")
        self._add_assistant(response)
        self._add_user(
            "[no tool call] Still nothing was run. This turn is constrained: "
            "the only output accepted is a tool call. Think first if you need "
            "to, then emit the call."
        )
        if interactive:
            print_info("(no tool call: forcing a tool call via constrained sampling)")
        self._emit("info", text=(
            "the model still did not use a tool - forcing a tool call via "
            "constrained sampling"))
        return (False, "")

    def _grounding_footer(self) -> str:
        """A factual line grounding the final answer in the session's own
        record, appended UNCONDITIONALLY - never gated on what the response
        text itself claims.

        Built from the facts loop.py already tracks for the self-verify nudge
        and the exit-code oracle (changed_files(), _last_verify_state), never
        from the response text, so it cannot be gamed by phrasing: it reports
        only what the harness recorded.
        """
        changed = self.changed_files()
        if changed:
            names = ", ".join(sorted(f["path"] for f in changed))
            parts = [f"{len(changed)} file(s) changed: {names}"]
        else:
            parts = ["no files changed"]
        if self.verify_cmd is not None and self._last_verify_state:
            parts.append(f"verify: {self._last_verify_state}")
        return "\n\n[session record: " + "; ".join(parts) + "]"

    def _run_pre_done_review(self, diff: str) -> str:
        """Run the pre-done review over *diff* and return the feedback to feed
        back (``""`` when the answer stands).

        Fail-OPEN: a reviewer that crashes or emits garbage never blocks the
        agent. It is not silent either - ``ReviewResult.ok=False`` is surfaced as
        a warning plus an audit entry, distinct from an approval, so the user
        knows the diff went out unchecked. Visibility only; control flow is the
        same in both cases.
        """
        print_warning = _agent.print_warning  # live: honour a patched agent.print_warning
        result = self._reviewer.review(diff, self._review_task)
        warning = self._reviewer.failure_warning(result)
        if warning:
            print_warning(warning)
            self._emit("info", text=warning)
            self._audit.notice("review_failed", warning)
        return self._reviewer.feedback_for(result)
    def _write_total(self) -> int:
        """Total file writes recorded this session. Compared against a per-task
        snapshot so the verify gate can tell "this task changed something" from
        "an earlier turn in this REPL session did"."""
        return sum(int(f.get("writes", 0)) for f in self._changed_files.values())

    def _run_verify_gate(self, response, interactive, st):
        """The harness-run exit-code oracle at the pre-done boundary.

        This is the same un-gameable check goal mode runs (``cli/goal.py``'s
        ``--until``), reaching the interactive REPL and the GUI, where the only
        finish gates were otherwise self-graded (the verify nudge) or advisory
        (the reviewer's diff opinion). The HARNESS runs the command and reads its
        exit code; the model's own claim of success is not consulted, so it
        cannot declare a premature one.

        Returns None to fall through to the remaining gates, or the
        ``(should_break, final_response)`` pair ``_handle_no_tool_calls`` returns.

        ``verify_cmd`` is set by the CLI's outer ``--until`` loop, an
        interactive/GUI session, or (tools/agents.py's ``_isolated_verify_cmd``)
        a worktree-isolated child - never more than one of these for the same
        Agent instance's own run, so the command is never executed twice per
        iteration. A parent that dispatches an isolated child and ALSO has its
        own verify_cmd runs two independent gates on two independent trees,
        which is by design, not a double-count of one."""
        if self.verify_cmd is None or st.verify_settled:
            return None
        # Nothing written since the last passing check means nothing to verify.
        if self._write_total() <= st.verify_checked_at:
            return None

        from .. import verify as _verify
        cmd = self.verify_cmd
        label = _verify.command_text(cmd)
        self._emit("info", text=f"verification: running `{label}`")
        if interactive:
            print_info(f"(verification: running `{label}`)")
        outcome = _verify.run_verify(cmd, self.cwd)
        code, output = outcome
        # Whether the command STARTED is carried on the outcome, never inferred
        # from the exit code.
        did_not_start = _verify.launch_failed(outcome)

        if code == 0:
            # Passing is not terminal: anything written after this point is
            # checked again. Only the inconclusive and exhausted cases settle.
            st.verify_checked_at = self._write_total()
            # A real check just passed, so the self-verification nudge has
            # nothing left to ask for and its writes count as verified.
            st.verify_nudged = True
            self._unverified_writes.clear()
            self._last_verify_state = "passed"
            self._emit("info", text=f"verification passed: `{label}` exited 0")
            if interactive:
                print_success(f"Verification passed: `{label}` exited 0.")
            return None

        if _verify.is_inconclusive(cmd, code, did_not_start):
            # Not a pass and not a fixable failure: the check either could not
            # start or collected nothing, so settle instead of retrying, state
            # that nothing was verified, and record the third state.
            st.verify_settled = True
            self._last_verify_state = "inconclusive"
            msg = (f"verification inconclusive: `{label}` "
                   f"{_verify.inconclusive_reason(cmd, code, did_not_start)} "
                   f"(exit {code}) - nothing was actually verified")
            self._emit("info", text=msg)
            _agent.print_warning(msg)
            return None

        if st.verify_retries < self.verify_max_retries and self._turns < self.max_turns:
            st.verify_retries += 1
            self._add_assistant(response)
            self._add_user(_verify.verify_feedback(cmd, code, output))
            msg = (f"verification failed (exit {code}); asking for a fix "
                   f"({st.verify_retries}/{self.verify_max_retries})")
            self._emit("info", text=msg)
            if interactive:
                _agent.print_warning(f"({msg})")
            return (False, "")

        # Retries exhausted and the check still fails: surface the answer but
        # mark it not-ok and state the failure.
        st.verify_settled = True
        self._last_run_ok = False
        self._last_verify_state = "failed"
        notice = (
            f"\n\n[verification FAILED] `{label}` still exits {code} after "
            f"{self.verify_max_retries} fix attempt(s). This task is NOT verified."
        )
        self._emit("info", text=(
            f"verification FAILED: `{label}` still exits {code} after "
            f"{self.verify_max_retries} fix attempt(s) - reporting failure, "
            "not a false success"))
        _agent.print_warning(notice.strip())
        self._add_assistant(response)
        # The grounding footer is unconditional. _last_verify_state is already
        # "failed" here, so the same call includes it.
        return (True, response + notice + self._grounding_footer())

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

        # No-progress breaker: many tool calls failed in a row across ANY tools.
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

        A call that ARMS a dispatch-time restriction runs alone too, whatever its
        destructive flag says.  See the segmentation below for why.
        """
        TOOL_REGISTRY = _agent.TOOL_REGISTRY  # live: honour a patched agent.TOOL_REGISTRY
        result_blocks: list[str] = []

        # Split into segments: each segment is (is_destructive, [calls]).
        #
        # A call that ARMS this session's active-skill restriction (use_skill,
        # _SKILL_STATE_TOOLS) is a segment boundary on BOTH sides, so it runs
        # alone despite being non-destructive.
        #
        # Ordering invariant any regrouping must preserve: everything emitted
        # BEFORE the arming call finishes before it runs, and nothing emitted
        # AFTER it starts until the restriction is armed. Narrowing therefore
        # applies FORWARD ONLY.
        #
        # A singleton segment runs serially and so has no batch deadline.
        segments: list[tuple[bool, list]] = []
        extendable = False              # may the last segment take another call?
        for call in calls:
            td = TOOL_REGISTRY.get(call.name)
            destructive = td.destructive if td else True
            solo = call.name in _SKILL_STATE_TOOLS
            if (extendable and not solo
                    and segments and segments[-1][0] == destructive):
                segments[-1][1].append(call)
            else:
                segments.append((destructive, [call]))
            extendable = not solo

        # Non-destructive peers abandoned at a batch deadline, as (future, tool
        # name). The timeout path cancels (a no-op on a running future) and shuts
        # the pool down without joining, so an abandoned thread can still be
        # executing when the next segment starts.
        #
        # Stored on the agent, not in this frame, so it outlives the turn. Ones
        # that have since finished are dropped so the list cannot grow.
        self._abandoned_peers = [(f, n) for f, n in self._abandoned_peers
                                 if not f.done()]
        abandoned = self._abandoned_peers

        for destructive, group in segments:
            still_live: list[tuple] = []
            if destructive and abandoned:
                abandoned = self._await_abandoned_peers(abandoned)
                self._abandoned_peers = abandoned
                still_live = abandoned
            if destructive or len(group) == 1:
                # Serial execution
                for call in group:
                    if still_live:
                        # Refuse rather than run alongside it. The model can call
                        # it again next turn.
                        peers = ", ".join(sorted({n for _f, n in still_live}))
                        result = ToolResult.error(
                            f"{call.name} was not run: it is a destructive tool, so "
                            f"it runs alone, and {peers} is still running after an "
                            f"extra {self._ABANDONED_PEER_GRACE_S}s. Running them "
                            "together would stack more concurrency than this "
                            f"machine can serve. WAIT for {peers} to finish - "
                            "retrying immediately, this turn or the next, is "
                            "refused again for the same reason."
                        )
                        if interactive:
                            print_tool_error(call.name, result.output)
                    else:
                        result = self._execute_tool(call, interactive=interactive)
                    result_blocks.append(self._result_block(call, result))
            else:
                # Parallel execution for a non-destructive batch. The pool is
                # shut down without waiting, so one hung tool cannot block the
                # batch past the deadline; the stuck thread is abandoned and
                # reported as a timeout.
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
                            if not fut.done():
                                # cancel() cannot stop a RUNNING future, so this
                                # one is still executing. Remember it: a later
                                # destructive segment must not start alongside it.
                                abandoned.append((fut, call.name))
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

    def _await_abandoned_peers(self, abandoned: list) -> list:
        """Wait a bounded grace for abandoned non-destructive peers to end.

        Returns the ones STILL running, so the caller can refuse to start a
        destructive tool beside them. The wait is bounded: an unbounded join here
        would defeat the batch deadline.
        """
        deadline = time.monotonic() + self._ABANDONED_PEER_GRACE_S
        for fut, _name in abandoned:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                fut.result(timeout=remaining)
            except Exception:
                # Only THAT it ended matters. Its outcome was already turned into
                # a result block in the segment that abandoned it.
                pass
        return [(f, n) for f, n in abandoned if not f.done()]

    # Wall-clock deadline for one parallel batch of non-destructive tools
    _PARALLEL_BATCH_TIMEOUT_S = 120

    # Extra grace a destructive tool gives an abandoned non-destructive peer from
    # the same batch to finish before it refuses to run beside it.
    _ABANDONED_PEER_GRACE_S = 30
