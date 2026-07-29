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
        Best-effort: a retrieval failure just returns the task unchanged.

        Also RECORDS which lessons were injected (id + text), on the agent, on the
        event stream, and in the audit trail. Retrieval used to render the lessons
        and throw the Episode objects away, so a lesson that steered a run badly
        was invisible after the fact and there was no handle to forget it by."""
        if not self._episodic or self._episode_store is None:
            return task
        try:
            from ..episodes import render_for_prompt
            episodes = self._episode_store.search(task)
            block = render_for_prompt(episodes)
        except Exception as e:
            # Rule 5: a failed recall is a real event, not a no-op. It stays
            # best-effort (the task still runs), but it leaves a trace.
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
                # Privacy/restricted sessions never reach here (recall is off), and
                # in privacy-recall opt-in the audit log is a NullAuditLog, so the
                # trail stays fail-safe by construction.
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
        self._last_run_ok = True           # per-run: an EARLIER run's failure must
                                           # not label THIS run failed. Without this
                                           # the flag only ever went False, so after
                                           # one bad turn every later turn of a REPL
                                           # or GUI session was reported as failed
                                           # too. The session-wide "did anything
                                           # fail" answer lives in _had_any_failure,
                                           # recorded in the finally below.
        self._last_verify_state = None     # per-run, same reasoning: this run has
                                           # not been verified until its own gate says so
        start_turns = self._turns          # turns used by *this* task only
        budget_escalated = False           # uncertainty escalation fires at most once per task
        # Per-task one-shot flags for the no-tool-calls handler (split out below):
        # self-verification + pre-done review fire once each; repair re-prompts are
        # capped. Held on a namespace so the helper can persist them across turns.
        # The verify_* fields drive the exit-code oracle gate (_run_verify_gate).
        # verify_checked_at is the write count the check last passed at, seeded
        # with the count on entry: the gate fires only when THIS task has written
        # something since, so a follow-up question in a REPL session whose EARLIER
        # turn edited files does not trigger the suite - and a fix made after a
        # pass (the reviewer can prompt one) is re-checked rather than riding in
        # on the earlier green.
        st = SimpleNamespace(verify_nudged=False, review_done=False, repair_count=0,
                             verify_retries=0, verify_settled=False,
                             verify_checked_at=self._write_total(),
                             partial_notice_count=0, partial_notice_cap_announced=False)

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

                # Background sub-agents that finished since the last turn are
                # absorbed HERE, on this thread, at a defined point - never from
                # the worker thread that ran them (see _drain_background_agents).
                # Same place as steering notes so the model reads the result
                # before its next call.
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
                segments = split_response(response, calls)
                if interactive:
                    for seg in segments:
                        if isinstance(seg, str) and seg.strip():
                            console.print(seg.strip())

                self._add_assistant(response)

                # Execute tools - run non-destructive batches in parallel
                result_blocks = self._execute_tools(calls, interactive=interactive)

                # A PARTIAL parse failure is otherwise invisible: parse_tool_calls
                # silently drops any tool-call-shaped block whose body could not
                # be recovered (malformed JSON that exhausts every lenient-parse
                # fallback), and the only downstream check for a parse problem is
                # "calls came back empty" (_handle_no_tool_calls) - which never
                # fires when a SIBLING call in the same response parsed fine.
                # Measured live: an edit_file on an existing file (large, more
                # escaping-prone content) silently vanished while a sibling
                # write_file and run_shell in the same turn executed for real -
                # the model never found out and, without this, neither would the
                # session record. `segments` already has the successfully-parsed
                # call spans excised (split_response), so anything tool-call-
                # shaped left in it is exactly what failed to parse.
                #
                # Capped at _MAX_TOOL_REPAIRS (matching the repair-turn's own
                # cap, same reasoning): the notice's own example text is itself
                # tool-call-shaped (a literal <tool_call> block with "name"/
                # "args" keys), so a model that echoes it back verbatim as
                # commentary - confirmed live: constructing exactly that echo
                # alongside one real call reproduces looks_like_tool_attempt()
                # returning True on the leftover - would otherwise re-trigger
                # this notice indefinitely, once per turn, forever.
                leftover = "".join(seg for seg in segments if isinstance(seg, str))
                if looks_like_tool_attempt(leftover):
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
                        # Rule 5 (AGENTS.md "we do not hide problems"): the cap
                        # bounds REPETITION, not VISIBILITY. Going fully silent
                        # here would trade the loud bug #920 fixed - a dropped
                        # call nobody hears about - for a quiet one, in exactly
                        # the sessions hitting this the most. One final notice
                        # naming the change, then a durable debug trace (below)
                        # for every further occurrence so the drop stays
                        # discoverable even once the model stops being told.
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
            # Fold this run's outcome into the session-level flag before the
            # per-run one is re-armed by the next run. One place, so every
            # failure path (max_turns, either circuit breaker, a stop) is
            # captured and a future one cannot forget to - the close-time
            # episodic reflection depends on this not being lost.
            if not self._last_run_ok:
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
        # The exit-code oracle goes FIRST among the finish gates: it is the only
        # un-gameable one (the harness runs a command and reads its exit code, so
        # the model cannot talk its way past it), and it costs no tokens. Running
        # it before the self-graded nudge and the reviewer means a failing check
        # goes straight back as a fix request instead of spending an LLM review
        # pass on code that does not pass. Skipped for a response that only LOOKS
        # like a broken tool call - that model is not finished, it is mid-call,
        # and the repair turn below handles it.
        if not looks_like_tool_attempt(response):
            gated = self._run_verify_gate(response, interactive, st)
            if gated is not None:
                return gated

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
            # Name the project's REAL check when one is known, so "verify your
            # work" means running it rather than re-reading the file it just
            # wrote (which is the model grading its own homework from memory).
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
        # reviewer error never blocks the answer (see _run_pre_done_review).
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

        # No tool calls → this is the final answer
        if not interactive and self.on_event is None:
            print_assistant_response(response, name=self.name)
        self._add_assistant(response)
        return (True, response)

    def _run_pre_done_review(self, diff: str) -> str:
        """Run the pre-done review over *diff* and return the feedback to feed
        back (``""`` when the answer stands).

        Fail-OPEN is kept exactly as it was: a reviewer that crashes or emits
        garbage never blocks the agent. What changes is that such a review is no
        longer SILENT. ReviewResult.ok=False used to have zero readers anywhere,
        so ``review_feedback()`` handed back the same "" for "approved" and for
        "the reviewer threw an exception" - a verification step that failed
        reported as success (AGENTS.md rule 5). It is now surfaced as a warning +
        an audit entry, distinct from an approval, so the user knows the diff went
        out unchecked. Visibility only: control flow is unchanged.
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

        Only the CLI's outer ``--until`` loop or an interactive/GUI session sets
        ``verify_cmd``, and never both for the same run, so the command is never
        executed twice per iteration."""
        if self.verify_cmd is None or st.verify_settled:
            return None
        # Nothing written since the last passing check -> nothing to verify. A
        # question in an ongoing REPL session must not trigger the project's test
        # suite just because an earlier turn edited a file.
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
        # Whether the command STARTED is the runner's own knowledge, carried on
        # the outcome; inferring it from the exit code would misread a genuine
        # 127 from a check that ran as "nothing was verified".
        did_not_start = _verify.launch_failed(outcome)

        if code == 0:
            # Passing is not terminal: anything written AFTER this point (a fix
            # the reviewer asks for) must be checked again rather than inheriting
            # this green. Only the inconclusive and exhausted cases settle.
            st.verify_checked_at = self._write_total()
            # A REAL check just passed, so the self-verification nudge - a
            # self-graded proxy for exactly this - has nothing left to ask for,
            # and the writes it guards are genuinely verified.
            st.verify_nudged = True
            self._unverified_writes.clear()
            self._last_verify_state = "passed"
            self._emit("info", text=f"verification passed: `{label}` exited 0")
            if interactive:
                print_success(f"Verification passed: `{label}` exited 0.")
            return None

        if _verify.is_inconclusive(cmd, code, did_not_start):
            # Not a pass and not a fixable failure: the check either could not
            # start or collected nothing. Retrying would burn every attempt on
            # something no code change can affect, so stop - but say plainly that
            # nothing was verified rather than letting an exit code the model
            # never saw look like success, and record the third state so a
            # programmatic consumer is not left reading an unqualified ok.
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

        # Retries exhausted and the check still fails. Report that honestly: the
        # answer is surfaced (the work may still be useful) but it is marked
        # not-ok and the failure is stated, never papered over as success.
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
        return (True, response + notice)

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

        # Non-destructive peers abandoned at a batch deadline, as (future, tool
        # name). A destructive tool is marked destructive precisely so it runs
        # ALONE; the timeout path cancels (a no-op on a running future) and shuts
        # the pool down without joining, so before this the next segment started
        # while the abandoned thread was still executing. run_tests is
        # non-destructive and a real suite comfortably outlives the 120s deadline,
        # so [run_tests, check_shell_job, dispatch_parallel] could put a whole test
        # run and two freshly dispatched child agents on the box at once - the exact
        # stacked concurrency the flag exists to prevent.
        #
        # Kept on the AGENT, not in this frame: _execute_tools runs once per turn,
        # and a tool that outlived a 120s deadline is very likely still running on
        # the next one. A per-call list would be empty by then, so "call it again
        # next turn" - which is what the refusal below tells the model - would walk
        # straight through an ungated path into the peer this exists to avoid.
        # Drop the ones that have since finished so the list cannot grow.
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
                        # REFUSE rather than run alongside it. The model can call it
                        # again next turn, by which time the peer has usually ended;
                        # quietly running it anyway would break the guarantee the
                        # destructive flag is supposed to give.
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
                            if not fut.done():
                                # cancel() cannot stop a RUNNING future, so this one
                                # is still executing. Remember it: a destructive
                                # segment later in this batch must not start
                                # alongside it.
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
        destructive tool beside them. Bounded, because an unbounded join here would
        reintroduce the very hang the batch deadline exists to avoid.
        """
        deadline = time.monotonic() + self._ABANDONED_PEER_GRACE_S
        for fut, _name in abandoned:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                fut.result(timeout=remaining)
            except Exception:
                # We only care THAT it ended, not how. Its outcome was already
                # turned into a result block for the model in the segment that
                # abandoned it, so re-reporting it here would duplicate it.
                pass
        return [(f, n) for f, n in abandoned if not f.done()]

    # Wall-clock deadline for one parallel batch of non-destructive tools
    _PARALLEL_BATCH_TIMEOUT_S = 120

    # Extra grace a destructive tool gives an abandoned non-destructive peer from
    # the same batch to finish before it refuses to run beside it.
    _ABANDONED_PEER_GRACE_S = 30
