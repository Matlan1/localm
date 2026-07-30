# SPDX-License-Identifier: AGPL-3.0-or-later
"""
GBNF grammar strings for constrained sampling.

These grammars are passed to ``llama_sampler_init_grammar`` so the sampler
masks any token that would violate the grammar at the current parse position.
The result is structurally valid output without post-hoc repair.

Usage example::

    from localm.inference.gbnf import JSON_OBJECT
    tokens = engine.chat_stream(messages, grammar=JSON_OBJECT)

All grammars use ``root`` as the entry rule, which is what
``llama_sampler_init_grammar`` expects.
"""

from __future__ import annotations

import re

#  JSON grammars

# Any valid JSON value at the root
JSON_VALUE = r"""
root   ::= ws value ws
value  ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (string ws ":" ws value ws ("," ws string ws ":" ws value ws)*)? "}"
array  ::= "[" ws (value ws ("," ws value ws)*)? "]"
string ::= "\"" ([^\"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws     ::= ([ \t\n\r])*
""".strip()

# Root must be a JSON object (dict)
JSON_OBJECT = r"""
root   ::= ws object ws
object ::= "{" ws (member ws ("," ws member ws)*)? "}"
member ::= string ws ":" ws value
value  ::= object | array | string | number | "true" | "false" | "null"
array  ::= "[" ws (value ws ("," ws value ws)*)? "]"
string ::= "\"" ([^\"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws     ::= ([ \t\n\r])*
""".strip()

# Root must be a JSON array
JSON_ARRAY = r"""
root   ::= ws array ws
array  ::= "[" ws (value ws ("," ws value ws)*)? "]"
value  ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (member ws ("," ws member ws)*)? "}"
member ::= string ws ":" ws value
string ::= "\"" ([^\"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws     ::= ([ \t\n\r])*
""".strip()


#  Tool-call grammar (coder XML format)

# Constrains the whole response to one or more <tool_call>...</tool_call> blocks
# separated by optional whitespace.
#
# STRICT mode (grammar=TOOL_CALLS_ONLY alone) forces tool-only output from the
# first token (no thinking/prose). Live-tested 2026-07-02: a thinking model
# masked off its <think> opener stalls into the leading `ws` (whitespace-only
# reply), so strict suits only a "must call a tool NOW" routing step.
#
# LAZY mode (grammar_lazy=True + TOOL_CALL_TRIGGER) is the general-turn form:
# free text and <think> flow unconstrained; the grammar engages only when the
# model starts a <tool_call>, from where the call must be valid JSON.
TOOL_CALLS_ONLY = r"""
root       ::= opt-ws tool-block+ opt-ws
tool-block ::= "<tool_call>" opt-ws json-obj opt-ws "</tool_call>" opt-ws
json-obj   ::= "{" ws "\"name\"" ws ":" ws string ws "," ws "\"args\"" ws ":" ws object ws "}"
object     ::= "{" ws (member ws ("," ws member ws)*)? "}"
member     ::= string ws ":" ws value
value      ::= object | array | string | number | "true" | "false" | "null"
array      ::= "[" ws (value ws ("," ws value ws)*)? "]"
string     ::= "\"" ([^\"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number     ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws         ::= ([ \t\n\r])*
opt-ws     ::= [ \t\n\r]? [ \t\n\r]? [ \t\n\r]?
""".strip()

# Lazy-grammar trigger for TOOL_CALLS_ONLY: full-match-with-capture-group form
# per llama.cpp's trigger_patterns contract (the grammar is fed from capture
# group 1, so enforcement starts exactly at the tag). Verified live 2026-07-02
# on the bundled runtime: prose before the tag flows free; a started
# <tool_call> is forced to a valid call.
#
# NO LEADING `[\s\S]*?`. It looks harmless and it is catastrophic. llama.cpp
# appends every generated token to `grammar.trigger_buffer` with NO SIZE CAP and
# re-runs this WHOLE pattern over the WHOLE buffer on EVERY token while no
# trigger has matched yet (src/llama-grammar.cpp, llama_grammar_accept_impl),
# so the per-token cost is whatever this pattern costs on the accumulated text,
# and the generation-total cost is that again for every token. With the leading
# lazy wildcard in front of a literal and a trailing greedy wildcard, that
# per-token cost is quadratic in the buffer on a backtracking engine, measured
# on the repetitive output that provoked it (rows of a markdown table):
#
#     buffer          old pattern        this pattern
#      5,700 chars         243 ms            0.061 ms
#     11,400 chars         975 ms            0.008 ms
#     22,800 chars       4,100 ms            0.007 ms
#    102,600 chars      66,344 ms            0.000 ms
#
# On the MSVC STL that cost eventually hits an internal complexity limit and
# throws `regex_error`, which crosses the ctypes boundary as WinError
# 0xe06d7363 and kills the worker mid-generation (GitHub #928, #833). On a
# build whose stdlib has no such limit it simply burns cores until the request
# is abandoned. Both were misattributed to NVIDIA Blackwell for five days; the
# mechanism is CPU-side and platform-independent.
#
# The leading wildcard was never needed: llama.cpp matches with a SEARCH (every
# start position is tried already), so it only ever added backtracking. Keep
# the tag INSIDE group 1 - the grammar's first literal is "<tool_call>", so a
# group that starts after the tag makes enforcement fail to match immediately
# and tool calling stops working entirely. Guarded by
# tests/test_redos_bounds.py.
TOOL_CALL_TRIGGER = r"(<tool_call>[\s\S]*)"


#  Structural pre-validation (LM-FZ-001)

# llama.cpp's GBNF parser is a hand-written recursive-descent parser with no
# bound on nesting depth, alternation count, or repeat-count size. A grammar
# string built of thousands of unmatched "(" drives its recursion far enough
# to overflow the native call stack - observed live as both a caught
# "exception: stack overflow" and a hard STATUS_ACCESS_VIOLATION process
# crash for the identical input, depending on heap layout at the moment of
# overflow. These bounds are deliberately generous for any real grammar (the
# largest production grammar in this file, TOOL_CALLS_ONLY, nests 3 levels
# deep) while being far below anything that risks the native stack.
MAX_GRAMMAR_BYTES = 65536
MAX_GRAMMAR_NESTING_DEPTH = 128
MAX_GRAMMAR_REPEAT_COUNT = 10000

_REPEAT_COUNT_RE = re.compile(r"\{(\d+)(?:,(\d+))?\}")


def check_grammar_structure(grammar: str) -> None:
    """Reject a grammar whose size or structural complexity could drive the
    native GBNF parser into stack overflow, BEFORE any of it reaches that
    parser. Pure Python, no native call - safe to run unconditionally on
    every grammar-bearing request, including the path where up-front
    validation would otherwise be deferred (RunnerBusy) straight into a
    generation-time native call.

    Raises :class:`InvalidGrammarError` (the same typed error the native
    validation path raises for a malformed grammar) so callers can treat both
    as one clean 400, never a native fault."""
    from localm.inference.backends.base import InvalidGrammarError

    if len(grammar) > MAX_GRAMMAR_BYTES:
        raise InvalidGrammarError(
            f"grammar is {len(grammar)} bytes, over the {MAX_GRAMMAR_BYTES}-byte limit")

    depth = 0
    max_depth = 0
    for ch in grammar:
        if ch in "([":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch in ")]":
            depth = max(0, depth - 1)
    if max_depth > MAX_GRAMMAR_NESTING_DEPTH:
        raise InvalidGrammarError(
            f"grammar nests {max_depth} levels deep, over the "
            f"{MAX_GRAMMAR_NESTING_DEPTH}-level limit (a common shape behind a "
            "native parser stack overflow)")

    for m in _REPEAT_COUNT_RE.finditer(grammar):
        for group in m.groups():
            if group is not None and int(group) > MAX_GRAMMAR_REPEAT_COUNT:
                raise InvalidGrammarError(
                    f"grammar repeat count {{{m.group(0)}}} exceeds the "
                    f"{MAX_GRAMMAR_REPEAT_COUNT} limit")


#  Trigger-pattern validation (GitHub #928, #833)

# GitHub #928/#833: localm's OWN hardcoded TOOL_CALL_TRIGGER (above) turned out
# to be catastrophically backtracking-prone - fixed in PR #933. But
# grammar_triggers is caller-supplied on the public API (protocol.py,
# documented at docs/server-api.md), and nothing validated those patterns
# before they reached the identical native std::regex path. Removing the
# feature was considered and explicitly rejected (maintainer: "removing a
# feature IS NOT fixing a problem") - this validates instead.
#
# check_grammar_structure above solves the adjacent problem for the *grammar*
# string with pure-Python structural bounds (size/nesting/repeat-count),
# because that native failure mode (a GBNF parser stack overflow) has a
# structural cause a static check can catch reliably. A trigger PATTERN's
# danger is different in kind: catastrophic regex backtracking is not fully
# decidable from the pattern's shape alone (whether an NFA has exponential
# worst-case behavior over some input is not, in general, a solved problem) -
# so this validates EMPIRICALLY instead of structurally: actually run the
# candidate pattern against fixed adversarial probe strings and see whether it
# completes, exactly the technique used to confirm PR #933's bug in the first
# place.
#
# That means an ordinary in-process call is not safe FOR THE VALIDATOR
# ITSELF - a dangerous pattern would hang the very code trying to reject it.
# See inference/_trigger_probe.py's docstring for the full mechanism: a
# long-lived daemon subprocess runs the actual probe, and the caller's own
# timeout on the round-trip - not anything inside the daemon - is what
# detects a hang, because nothing inside a single thread can interrupt a
# stuck C-level regex match. Same shape as
# backends/llamacpp/_loader.py's gpu_memory_isolated()/_probe_roundtrip for
# the VRAM-probe daemon; deliberately not reused directly (a different
# protocol, a different failure domain) but following the identical
# spawn/lock/timeout/kill-and-respawn architecture rather than inventing a
# new one.

import threading as _threading

# No caller-supplied trigger pattern is bounded in size before this point -
# found in cross-session review after the fix below first landed (GitHub
# #928/#833's own PR only bounded the GRAMMAR string, via MAX_GRAMMAR_BYTES
# above; the TRIGGER pattern had no equivalent). An uncapped pattern still
# costs real, unbounded work before any adversarial-shape logic runs:
# _static_shape_rejection's own paren-depth scan below is O(len(pattern));
# _pattern_derived_probes in _trigger_probe.py (the isolated daemon) scans
# every character to build a frequency table, also O(len(pattern)), for
# whatever survives the static filter; and the pattern string is itself the
# cache key in _VALIDATED_TRIGGER_PATTERNS, so an oversized pattern is
# retained in memory up to _MAX_CACHED_TRIGGER_PATTERNS times over. None of
# this is catastrophic-backtracking-class cost (compiling and scanning a
# long but simple string is linear, not exponential), but it is real,
# unbounded cost for zero legitimate reason: real trigger patterns are tiny
# (TOOL_CALL_TRIGGER is ~30 characters). Checked FIRST in
# _static_shape_rejection, before any other scan, so an oversized pattern
# never reaches even that function's own O(n) work, let alone the daemon.
MAX_TRIGGER_PATTERN_BYTES = 4096

_TRIGGER_PROBE_PROC = None
_TRIGGER_PROBE_LOCK = _threading.Lock()
_PREWARM_THREAD = None

# validate_trigger_patterns's per-process cache - see that function's
# docstring for why this is an efficiency measure, not the safety mechanism.
_VALIDATED_TRIGGER_PATTERNS: "dict[str, str | None]" = {}
_MAX_CACHED_TRIGGER_PATTERNS = 1024

# Per-check timeout against a running daemon. Generous relative to a SAFE
# pattern (measured near-instant - a fixed TOOL_CALL_TRIGGER-shaped pattern
# against the daemon's probes is a few milliseconds), tight relative to how
# long a legitimate request should ever wait on validation before generation
# even starts.
_TRIGGER_PROBE_TIMEOUT = 2.0
# First check after a (re)spawn pays Python interpreter + module-import
# cold-start cost, not measured separately here but bounded generously above
# any plausible cold-start on this codebase's own precedent (_loader.py's
# VRAM-probe daemon measured 1.9-7.9s for a load that also loads a native
# DLL and a large binding surface; this daemon imports only re/json/sys, so
# its cold start should be far cheaper - the wider bound is deliberate
# headroom, not a measurement of THIS daemon specifically).
_TRIGGER_PROBE_SPAWN_TIMEOUT = 10.0
# A pattern that hangs a probe means the daemon never replies at all - the
# caller has no way to distinguish "still checking" from "will never
# reply" except by how long it has already waited, so this bounds it too
# (same value as the steady-state timeout; there is no cold-start
# distinction once a hang is suspected, unlike a normal query).


def _spawn_trigger_probe_daemon():
    """Start the long-lived trigger-pattern-probe daemon. Caller holds
    _TRIGGER_PROBE_LOCK. Line-buffered text pipes (bufsize=1) so one
    print()/readline() on either side is exactly one protocol message - see
    _trigger_probe.py for the request/response contract. stderr discarded:
    this daemon has no diagnostic output a caller needs (a probe hanging is
    the caller's own timeout to observe, not something the daemon can log
    about itself)."""
    import subprocess

    from localm._mp_spawn import interpreter_for_localm_children
    return subprocess.Popen(
        [interpreter_for_localm_children(), "-u", "-m", "localm.inference._trigger_probe"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1)


def _readline_with_timeout(stream, timeout: float):
    """stream.readline() with a timeout (Windows pipes offer no select()-style
    wait, so the read runs in a daemon thread and this waits on it via a
    queue - same pattern as backends/llamacpp/_loader.py's helper of the
    same name, reimplemented here rather than imported across a private
    backend-internal module boundary). On timeout the reader thread is
    abandoned, not joined, so this function itself never blocks past
    *timeout*; the thread exits on its own once the read unblocks or the
    pipe closes (killing the daemon process closes its stdout, which is
    exactly what happens on a detected timeout below). Returns None on
    timeout, EOF, or any read error."""
    import queue

    q: "queue.Queue" = queue.Queue(maxsize=1)

    def _reader():
        try:
            q.put(stream.readline())
        except Exception:
            q.put(None)

    t = _threading.Thread(target=_reader, daemon=True)
    t.start()
    try:
        line = q.get(timeout=timeout)
    except queue.Empty:
        return None
    if not line:
        return None
    return line


def _kill_and_clear_trigger_probe() -> None:
    """Caller holds _TRIGGER_PROBE_LOCK. Best-effort kill of a dead/hung
    daemon, then hands off to a BACKGROUND thread to pre-spawn its
    replacement, so the respawn cost usually lands on nobody's request
    instead of landing on whichever caller happens to arrive next.

    Why this exists (measured live): every timeout kills the daemon
    (nothing else can be trusted alive after a stuck native regex match),
    so without this, the VERY NEXT validate_trigger_patterns() call - from
    an attacker sending the next of many distinct dangerous patterns, or
    from a wholly unrelated legitimate caller who just happened to arrive
    after one - pays the SPAWN timeout (multiple seconds) synchronously,
    not the fast steady-state one. N distinct attack requests were measured
    compounding at close to the spawn cost EACH as a direct result.

    ASYMMETRY AND FIX: Previously, the background pre-warm thread spawned
    outside _TRIGGER_PROBE_LOCK and acquired the lock only to install its
    finished result, while the main thread's synchronous daemon spawn ran
    INSIDE _TRIGGER_PROBE_LOCK, holding it for the full multi-second spawn.
    This caused the pre-warm thread to block behind the main thread's spawn,
    rendering pre-warming ineffective under back-to-back requests. The main
    thread now spawns outside the lock as well and joins an in-flight
    pre-warm thread rather than starting a duplicate spawn, removing lock
    contention during process creation."""
    global _TRIGGER_PROBE_PROC, _PREWARM_THREAD
    if _TRIGGER_PROBE_PROC is not None:
        try:
            _TRIGGER_PROBE_PROC.kill()
        except Exception:
            pass
    _TRIGGER_PROBE_PROC = None

    def _prewarm_replacement() -> None:
        global _TRIGGER_PROBE_PROC
        try:
            replacement = _spawn_trigger_probe_daemon()
        except Exception:
            return   # best-effort: the next caller's own synchronous spawn is the fallback
        with _TRIGGER_PROBE_LOCK:
            if _TRIGGER_PROBE_PROC is None or _TRIGGER_PROBE_PROC.poll() is not None:
                _TRIGGER_PROBE_PROC = replacement
            else:
                # Someone else already spawned one while we were working -
                # do not leak our own spare, it was never handed to anyone.
                try:
                    replacement.kill()
                except Exception:
                    pass

    t = _threading.Thread(target=_prewarm_replacement, daemon=True,
                          name="localm-trigger-probe-prewarm")
    _PREWARM_THREAD = t
    t.start()


def _static_shape_rejection(pattern: str) -> "str | None":
    """Cheap, in-process, microsecond-cost rejection of KNOWN catastrophic-
    backtracking shapes, run BEFORE the expensive daemon probe. Returns a
    rejection reason string, or None if the pattern passes - which does NOT
    mean the pattern is safe, only that it does not match a shape this
    function recognizes; see validate_trigger_patterns for how this
    composes with the probe that actually decides.

    Why this exists in addition to the probe, not instead of it: rejecting
    a pattern via the probe costs the FULL probe timeout by construction
    (that is how a hang is detected), and every rejection also kills and
    respawns the daemon (its process cannot be trusted alive after a
    timeout), so the NEXT check - attacker or legitimate - pays the SLOW
    spawn timeout too, not the steady-state one. MEASURED live: ~10s per
    rejected pattern even with a 1s steady-state timeout configured,
    because every one of N distinct attack patterns forces a respawn for
    the (N+1)th check. N distinct attack patterns therefore serialize
    behind one lock at close to the SPAWN timeout each, turning the
    validator itself into the chokepoint it exists to prevent. Catching
    the shapes recognized here removes them from that queue entirely -
    zero lock contention, zero subprocess cost, zero effect on legitimate
    traffic - which is the only way to make repeated attack traffic cheap
    to reject rather than merely SAFE to reject.

    A length check runs first (see MAX_TRIGGER_PATTERN_BYTES above), before
    either shape check below even scans the pattern once - real trigger
    patterns are tiny, so this costs nothing for legitimate traffic while
    bounding the two shape checks' own O(len(pattern)) scan cost, and the
    daemon-side probe-derivation cost of whatever survives this function.

    Two shapes, chosen because they are provably useless-or-dangerous
    rather than merely "look risky" (avoiding the false-positive trap that
    would violate "keep the feature working" as much as removing the
    feature would have):

    1. A leading unanchored variable-length wildcard (``.*``, ``.*?``,
       ``.+``, ``.+?``, or a character-class equivalent like ``[\\s\\S]*``)
       at the START of an unanchored pattern - the EXACT shape of the
       historical #928/#833 defect. Not merely risky: REDUNDANT.
       llama_grammar_trigger_pattern::find() (llama.cpp) uses
       std::regex_search (search-anywhere) for any pattern that does not
       begin with ``^`` and end with ``$`` - already trying every start
       position - so an explicit leading wildcard can never change what
       matches, only add backtracking cost trying to. A pattern that
       genuinely needs leading-wildcard semantics would anchor with ``^``,
       which this check does not touch.
    2. A classic nested-quantifier shape: a group containing its own
       unescaped, top-level quantifier (``*``/``+``/``{m,n}``), itself
       immediately followed by another quantifier - e.g. ``(a+)+``,
       ``(a*)*``, ``(x{2,})+``. The textbook catastrophic-backtracking
       construct, independent of this codebase's own history. Detected
       with a linear paren-depth scan (same style as
       check_grammar_structure above), not a full regex parser: it can
       miss an exotic construction of the same danger (ReDoS-safety is not
       fully decidable from shape alone - ONLY the daemon probe below
       actually proves anything) and, rarely, could flag an unusual but
       benign pattern using literal unescaped braces in an unexpected way.
       Documented as a heuristic, not a proof, same honesty as this whole
       mechanism's module docstring."""
    if len(pattern) > MAX_TRIGGER_PATTERN_BYTES:
        return (
            f"pattern is {len(pattern)} bytes, over the "
            f"{MAX_TRIGGER_PATTERN_BYTES}-byte limit for a trigger pattern")

    if re.match(r"^(?!\^)(\.\*\??|\.\+\??|\[[^\]]*\][*+]\??)", pattern):
        return ("leading unanchored wildcard (.*, .+, or a character-class "
                 "equivalent) - redundant under search() and the exact shape "
                 "of the historical #928 defect; anchor with ^ if literal "
                 "leading-wildcard semantics are genuinely needed")

    depth = 0
    in_class = False
    group_has_quantifier_at: "dict[int, bool]" = {}
    i, n = 0, len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "\\":
            i += 2
            continue
        if in_class:
            if ch == "]":
                in_class = False
            i += 1
            continue
        if ch == "[":
            in_class = True
            i += 1
            continue
        if ch == "(":
            depth += 1
            group_has_quantifier_at[depth] = False
            i += 1
            continue
        if ch == ")":
            had_quantifier = group_has_quantifier_at.pop(depth, False)
            depth -= 1
            if had_quantifier and i + 1 < n and pattern[i + 1] in "*+{":
                return (
                    f"nested quantifier at position {i} - a group containing "
                    "its own top-level */+/{n,m} immediately followed by "
                    "another quantifier is the textbook catastrophic-"
                    "backtracking shape")
            i += 1
            continue
        if ch in "*+{" and depth in group_has_quantifier_at:
            group_has_quantifier_at[depth] = True
        i += 1
    return None


def _probe_pattern_is_safe(pattern: str) -> "tuple[bool, str]":
    """(is_safe, reason). is_safe=False covers three distinct cases, all
    treated identically by the caller (reject): the daemon determined the
    pattern is invalid or dangerous and said so ("BAD ..."), the daemon
    could not be reached at all (spawn failure), or - the case this whole
    mechanism exists for - the daemon hung trying to check the pattern and
    never replied within the timeout. Never raises."""
    global _TRIGGER_PROBE_PROC, _PREWARM_THREAD
    first_spawn = False

    with _TRIGGER_PROBE_LOCK:
        if _TRIGGER_PROBE_PROC is not None and _TRIGGER_PROBE_PROC.poll() is None:
            proc = _TRIGGER_PROBE_PROC
        else:
            if _TRIGGER_PROBE_PROC is not None:
                try:
                    _TRIGGER_PROBE_PROC.kill()
                except Exception:
                    pass
                _TRIGGER_PROBE_PROC = None
            proc = None

    if proc is None:
        prewarm = _PREWARM_THREAD
        if prewarm is not None and prewarm.is_alive():
            prewarm.join(timeout=_TRIGGER_PROBE_SPAWN_TIMEOUT)

        with _TRIGGER_PROBE_LOCK:
            if _TRIGGER_PROBE_PROC is not None and _TRIGGER_PROBE_PROC.poll() is None:
                proc = _TRIGGER_PROBE_PROC
                first_spawn = True

        if proc is None:
            try:
                new_proc = _spawn_trigger_probe_daemon()
            except Exception as e:
                return False, f"trigger-pattern probe could not be started ({e})"

            with _TRIGGER_PROBE_LOCK:
                if _TRIGGER_PROBE_PROC is None or _TRIGGER_PROBE_PROC.poll() is not None:
                    _TRIGGER_PROBE_PROC = new_proc
                    proc = new_proc
                    first_spawn = True
                else:
                    try:
                        new_proc.kill()
                    except Exception:
                        pass
                    proc = _TRIGGER_PROBE_PROC

    with _TRIGGER_PROBE_LOCK:
        if proc is not _TRIGGER_PROBE_PROC or proc.poll() is not None:
            return False, "trigger-pattern probe process is no longer valid"
        try:
            import json
            proc.stdin.write(json.dumps({"pattern": pattern}) + "\n")
            proc.stdin.flush()
        except Exception as e:
            _kill_and_clear_trigger_probe()
            return False, f"trigger-pattern probe could not be reached ({e})"
        timeout = _TRIGGER_PROBE_SPAWN_TIMEOUT if first_spawn else _TRIGGER_PROBE_TIMEOUT
        line = _readline_with_timeout(proc.stdout, timeout)
        if line is None:
            # No reply within the timeout: either the daemon crashed/exited
            # (EOF), or - the case this mechanism exists to catch - it is
            # still alive and stuck inside re.search() on an adversarial
            # probe. Either way the pattern is not safe to use, and the
            # daemon (alive or not) is killed so a fresh one handles the
            # NEXT request rather than staying wedged for it too.
            _kill_and_clear_trigger_probe()
            return False, (
                f"pattern did not pass the safety probe within {timeout:.1f}s - "
                "likely catastrophic regex backtracking")
        line = line.strip()
        if line == "OK":
            return True, ""
        return False, line[4:] if line.startswith("BAD ") else line


def validate_trigger_patterns(patterns: "list[str]") -> None:
    """Reject a caller-supplied lazy-grammar trigger pattern list that is
    invalid or unsafe to run, BEFORE any of it reaches
    llama_sampler_init_grammar_lazy_patterns - matching
    check_grammar_structure's contract for the grammar string itself
    (raises the same typed InvalidGrammarError so a caller can treat both as
    one clean 400).

    Two layers, cheapest first: _static_shape_rejection (in-process,
    microseconds, catches KNOWN catastrophic shapes - see its own docstring
    for why this is not optional ceremony but load-bearing: without it, N
    distinct attacker-supplied patterns serialize behind the probe's single
    lock at close to its SPAWN timeout each - MEASURED live - turning this
    validator into the same kind of chokepoint it exists to remove), then
    _probe_pattern_is_safe (the daemon round-trip - see that function and
    _trigger_probe.py) for whatever survives the cheap check. Composing
    both catches more than either alone: the static layer is free but
    provably incomplete (ReDoS-safety is not fully decidable from a
    pattern's shape), the probe is empirical and catches anything that
    times out but costs real time to reject. Documented HONESTLY, not
    oversold as complete: a NOVEL catastrophic pattern that matches
    neither known static shape AND is not slow against
    _trigger_probe.py's specific adversarial corpus would pass both layers
    and still reach the native sampler - see that module's docstring for
    exactly what the corpus does and does not cover.

    This function BLOCKS for up to a few seconds when a pattern survives
    the static filter but is genuinely dangerous (it then waits out the
    probe timeout - see _probe_pattern_is_safe). Callers on an asyncio
    route MUST run this via loop.run_in_executor, never call it directly
    from an async handler - see routes/chat.py's call sites for why (a
    direct call would freeze that whole event loop, and every OTHER
    concurrent request on it, for the duration of one caller's bad
    pattern). Residual, honestly-noted limitation of that same executor
    hand-off: it shares Python's default thread pool with other blocking
    work this server offloads (engine.load, embedding, token counting), so
    a coordinated flood of MANY concurrent requests each submitting a
    distinct uncached pattern that survives the static filter could still
    exhaust that pool for a few seconds. Lower severity than the original
    defect (temporary degradation under a sustained multi-connection
    attack, not a crash or an indefinite hang, and it self-resolves once
    patterns time out) - not solved here; general request-concurrency
    bounding is a broader problem than this one fix.

    Each unique pattern's final verdict (from either layer) is cached for
    this process's lifetime, keyed on the exact pattern string - most real
    traffic reuses a small set of patterns (localm's own TOOL_CALL_TRIGGER,
    or one integration's fixed pattern), so after the first check of any
    given pattern this is an in-memory dict lookup, not repeated validation
    work. The cache is intentionally NOT a safety mechanism by itself (a
    determined caller sending a fresh pattern on every request bypasses it
    trivially) - it exists purely so legitimate, repeated use of the same
    pattern does not pay validation cost on every request; the actual
    safety guarantee is the static filter plus the probe's per-call daemon
    round-trip, run unconditionally on every pattern this process has not
    already validated."""
    from localm.inference.backends.base import InvalidGrammarError

    for pattern in patterns:
        if pattern in _VALIDATED_TRIGGER_PATTERNS:
            if _VALIDATED_TRIGGER_PATTERNS[pattern] is not None:
                raise InvalidGrammarError(
                    f"grammar trigger pattern rejected: {_VALIDATED_TRIGGER_PATTERNS[pattern]}")
            continue
        static_reason = _static_shape_rejection(pattern)
        if static_reason is not None:
            is_safe, reason = False, static_reason
        else:
            is_safe, reason = _probe_pattern_is_safe(pattern)
        # Bounded so a flood of distinct junk patterns cannot grow this
        # dict without limit; legitimate use reuses a handful of patterns,
        # well under this ceiling in practice.
        if len(_VALIDATED_TRIGGER_PATTERNS) >= _MAX_CACHED_TRIGGER_PATTERNS:
            _VALIDATED_TRIGGER_PATTERNS.clear()
        _VALIDATED_TRIGGER_PATTERNS[pattern] = None if is_safe else reason
        if not is_safe:
            raise InvalidGrammarError(f"grammar trigger pattern rejected: {reason}")

