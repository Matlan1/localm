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
# Used alone (grammar=TOOL_CALLS_ONLY) it forces tool-only output from the first
# token, with no thinking or prose.
#
# Used with grammar_lazy=True and TOOL_CALL_TRIGGER, free text and <think> flow
# unconstrained and the grammar engages only once the model starts a
# <tool_call>, from where the call must be valid JSON.
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

# Used alone (no trigger, grammar_lazy off). Requires at least one structurally
# valid tool call, with a bounded reasoning prelude allowed in front of it.
#
# The opening "<think>" is required when the prelude is used: a model that opens
# one may think and must then call; a model that does not open one is held to a
# tool call from its first token.
#
# think-char matches any character except '<', or '<' not followed by '/', or
# '</' not followed by 't', so only the byte sequence "</t" is withheld and
# reasoning text may contain '<' freely. A literal "</td>" inside reasoning ends
# the block early.
#
# The {0,1900} bound caps the prelude: after 1900 prelude characters the only
# legal continuation is "</think>", so the model must reach the tool call. The
# count must stay under llama.cpp's own repeat-count ceiling, and moves together
# with MAX_GRAMMAR_REPEAT_COUNT below.
TOOL_CALLS_AFTER_THINK = r"""
root        ::= opt-think opt-ws tool-block+ opt-ws
opt-think   ::= ("<think>" think-body "</think>")?
think-body  ::= think-char{0,1900}
think-char  ::= [^<] | "<" [^/] | "</" [^t]
tool-block  ::= "<tool_call>" opt-ws json-obj opt-ws "</tool_call>" opt-ws
json-obj    ::= "{" ws "\"name\"" ws ":" ws string ws "," ws "\"args\"" ws ":" ws object ws "}"
object      ::= "{" ws (member ws ("," ws member ws)*)? "}"
member      ::= string ws ":" ws value
value       ::= object | array | string | number | "true" | "false" | "null"
array       ::= "[" ws (value ws ("," ws value ws)*)? "]"
string      ::= "\"" ([^\"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number     ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws          ::= ([ \t\n\r])*
opt-ws      ::= [ \t\n\r]? [ \t\n\r]? [ \t\n\r]?
""".strip()

# Lazy-grammar trigger for TOOL_CALLS_ONLY, in the full-match-with-capture-group
# form llama.cpp's trigger_patterns contract expects. The grammar is fed from
# capture group 1, so enforcement starts exactly at the tag; the tag must stay
# INSIDE group 1 or enforcement never matches and tool calling stops entirely.
#
# The pattern must not begin with a lazy wildcard. llama.cpp re-runs the whole
# pattern over the whole accumulated trigger buffer on every generated token, so
# a leading wildcard in front of a literal makes the per-token cost quadratic in
# the buffer. It matches with a search, so a leading wildcard adds nothing.
TOOL_CALL_TRIGGER = r"(<tool_call>[\s\S]*)"


#  Structural pre-validation

# Bounds on a grammar string, checked before any of it reaches llama.cpp's GBNF
# parser, which is recursive-descent with no bound on nesting depth, alternation
# count or repeat-count size and overflows the native stack on deep input.
MAX_GRAMMAR_BYTES = 65536
MAX_GRAMMAR_NESTING_DEPTH = 128

# Ceiling on a {n,m} repeat count in any grammar. Must stay below llama.cpp's
# own native repeat-count ceiling, or a grammar clears this check and then fails
# at the native parse instead. Shares its value with TOOL_CALLS_AFTER_THINK's
# think-body bound above.
MAX_GRAMMAR_REPEAT_COUNT = 1900

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
                    f"{MAX_GRAMMAR_REPEAT_COUNT} limit (llama.cpp's native "
                    "GBNF parser rejects repeat counts above roughly 2000 "
                    "as unreasonable; reduce this repeat count)")


#  Trigger-pattern validation

# grammar_triggers is caller-supplied on the public API and its patterns reach
# llama.cpp's native std::regex path, so every one is validated here first.
#
# Validation is empirical rather than structural: the candidate pattern is run
# against fixed adversarial probe strings and the check is whether it completes.
#
# The probe runs in a long-lived daemon subprocess (inference/_trigger_probe.py)
# and the caller's own timeout on the round trip, not anything inside the
# daemon, is what detects a hang: nothing inside a single thread can interrupt a
# stuck C-level regex match.

import queue
import threading as _threading

# Size cap on a caller-supplied trigger pattern. Checked FIRST in
# _static_shape_rejection, before any other scan, so an oversized pattern never
# reaches that function's own O(n) work or the daemon.
MAX_TRIGGER_PATTERN_BYTES = 4096

# How many probes may run at once, and how long a caller may wait for its turn.
#
# _PROBE_SLOTS_FREE below hands out one independent daemon per concurrent
# caller, so N patterns cost ceil(N / _TRIGGER_PROBE_POOL_SIZE) rounds.
#
# _TRIGGER_PROBE_SLOT_WAIT bounds how long a caller queues for a slot before
# being refused outright, which bounds per-caller latency. It does not bound how
# many callers may be in that wait at once; _TRIGGER_PROBE_MAX_WAITERS below
# does that.
#
# Slots are spawned lazily and handed out LIFO, so a server serving one caller
# at a time keeps exactly one daemon alive.
_TRIGGER_PROBE_POOL_SIZE = 4
_TRIGGER_PROBE_SLOT_WAIT = 5.0

# How many callers may queue for a slot at once, a different bound from how long
# each one queues.
#
# Both call sites run this on the asyncio loop's default executor, which is
# shared with engine.load, engine.embed, count_tokens, the GPU/VRAM probes and
# the isolated-runner RPCs, so an unbounded number of waiters would be an
# unbounded number of default-executor threads parked for up to
# _TRIGGER_PROBE_SLOT_WAIT each.
#
# A caller served from a free slot never takes a waiter permit (see
# _probe_pattern_is_safe's get_nowait fast path), so this bounds the QUEUE and
# never the throughput.
#
# Must stay >= _TRIGGER_PROBE_POOL_SIZE, or a burst the pool is about to absorb
# gets refused while slots are freeing up. The ceiling on default-executor
# threads held by this function is _TRIGGER_PROBE_POOL_SIZE +
# _TRIGGER_PROBE_MAX_WAITERS.
_TRIGGER_PROBE_MAX_WAITERS = 2 * _TRIGGER_PROBE_POOL_SIZE


class _ProbeSlot:
    """One probe daemon plus the state needed to replace it.

    ``lock`` guards ``proc`` against the background pre-warm thread ONLY, and is
    held for microseconds to install or claim a process - NEVER across a spawn or
    a probe round trip. That is the whole point of this class: exclusive use of
    the daemon comes from having CHECKED THE SLOT OUT of _PROBE_SLOTS_FREE, not
    from holding a lock, so one caller's 2.0s timeout no longer serialises every
    other caller behind it.
    """

    __slots__ = ("lock", "proc", "prewarm")

    def __init__(self) -> None:
        self.lock = _threading.Lock()
        self.proc = None
        self.prewarm = None


# Free slots, LIFO: the most-recently-returned (and so already-spawned) slot is
# re-handed to the next caller, so a server whose real concurrency is 1 only
# ever spawns one daemon and the other slots stay empty.
_PROBE_SLOTS_FREE: "queue.LifoQueue" = queue.LifoQueue()
for _ in range(_TRIGGER_PROBE_POOL_SIZE):
    _PROBE_SLOTS_FREE.put(_ProbeSlot())


class _WaiterGate:
    """How many callers are QUEUEING for a probe slot right now, and a
    non-blocking way to join them.

    Deliberately NOT a threading.Semaphore, for one reason worth the extra
    dozen lines: a semaphore's remaining permits are private
    (``_value``), so the leak that this class's ``waiting`` count makes
    directly assertable would only be observable by reading a private
    attribute. A waiter permit that is taken and never released is a
    permanent, silent capacity loss - the same failure mode as a leaked pool
    slot, and it has the same tell - so being able to assert it returned to
    zero is worth more here than reusing the stdlib primitive.

    The limit is passed to try_enter rather than stored, so a test can
    monkeypatch _TRIGGER_PROBE_MAX_WAITERS the way it already monkeypatches
    _TRIGGER_PROBE_SLOT_WAIT and _TRIGGER_PROBE_POOL_SIZE. Storing it at
    construction would silently ignore that patch, which is a fixture that
    cannot express the case it exists to test.
    """

    __slots__ = ("_lock", "_waiting")

    def __init__(self) -> None:
        self._lock = _threading.Lock()
        self._waiting = 0

    @property
    def waiting(self) -> int:
        with self._lock:
            return self._waiting

    def try_enter(self, limit: int) -> bool:
        """Take a waiter permit, or return False immediately. Never blocks:
        the whole point is that a caller which cannot queue is refused now
        rather than queued anyway."""
        with self._lock:
            if self._waiting >= limit:
                return False
            self._waiting += 1
            return True

    def leave(self) -> None:
        with self._lock:
            self._waiting -= 1


_PROBE_WAITER_GATE = _WaiterGate()

# The three outcomes of a probe. SAFE and UNSAFE describe the PATTERN;
# UNDETERMINED describes the VALIDATOR and is never cached, because the cache
# lives for the whole process. UNSAFE and UNDETERMINED both reject the request,
# and differ only in what is remembered and in whether the caller or this side
# is reported at fault (see TriggerValidatorUnavailableError).
_PROBE_SAFE = "safe"
_PROBE_UNSAFE = "unsafe"
_PROBE_UNDETERMINED = "undetermined"

# validate_trigger_patterns's per-process cache. An efficiency measure, not the
# safety mechanism.
_VALIDATED_TRIGGER_PATTERNS: "dict[str, str | None]" = {}
_MAX_CACHED_TRIGGER_PATTERNS = 1024

# Per-check timeout against an already-running daemon.
_TRIGGER_PROBE_TIMEOUT = 2.0
# Timeout for the first check after a (re)spawn, which also pays Python
# interpreter and module-import cold-start cost.
_TRIGGER_PROBE_SPAWN_TIMEOUT = 10.0
# A pattern that hangs the probe produces no reply at all, so the wait for a
# reply is what bounds it; the steady-state timeout applies, with no cold-start
# distinction once a hang is suspected.


def _spawn_trigger_probe_daemon():
    """Start the long-lived trigger-pattern-probe daemon. Caller must NOT hold
    any slot lock: process creation takes multiple seconds, and holding a lock
    across it is exactly the contention PR #943 removed (see
    _slot_kill_and_prewarm). Line-buffered text pipes (bufsize=1) so one
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


def _slot_kill_and_prewarm(slot: "_ProbeSlot") -> None:
    """Best-effort kill of *slot*'s dead/hung daemon, then hand off to a
    BACKGROUND thread to pre-spawn its replacement, so the respawn cost usually
    lands on nobody's request instead of landing on whichever caller happens to
    check this slot out next.

    Caller OWNS *slot* (it checked it out of _PROBE_SLOTS_FREE) and must not
    hold slot.lock.

    Why this exists (measured live): every timeout kills the daemon (nothing
    else can be trusted alive after a stuck native regex match), so without
    this, the VERY NEXT caller to take this slot - an attacker sending the next
    of many distinct dangerous patterns, or a wholly unrelated legitimate
    caller who just happened to arrive after one - pays the SPAWN timeout
    (multiple seconds) synchronously, not the fast steady-state one. N distinct
    attack requests were measured compounding at close to the spawn cost EACH
    as a direct result.

    ASYMMETRY AND FIX (PR #943, preserved here per-slot): the background
    pre-warm thread spawns OUTSIDE any lock and acquires slot.lock only to
    install its finished result. The main thread also spawns outside the lock
    and JOINS an in-flight pre-warm rather than starting a duplicate. An
    earlier version held the lock across the main thread's multi-second spawn,
    which blocked the pre-warm thread behind it and made pre-warming
    ineffective under back-to-back requests."""
    with slot.lock:
        proc, slot.proc = slot.proc, None
    if proc is not None:
        try:
            proc.kill()
        except Exception:
            pass

    def _prewarm_replacement() -> None:
        try:
            replacement = _spawn_trigger_probe_daemon()
        except Exception:
            return   # best-effort: the next caller's own synchronous spawn is the fallback
        with slot.lock:
            if slot.proc is None or slot.proc.poll() is not None:
                slot.proc = replacement
                return
        # Another thread already spawned one; kill the unused spare. Killed
        # OUTSIDE the lock: no lock is held while a process teardown runs.
        try:
            replacement.kill()
        except Exception:
            pass

    t = _threading.Thread(target=_prewarm_replacement, daemon=True,
                          name="localm-trigger-probe-prewarm")
    slot.prewarm = t
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
    the (N+1)th check. That measurement was taken when a SINGLE daemon sat
    behind a SINGLE lock held across the whole round trip, so N patterns
    serialized at close to the SPAWN timeout each; the pool and the
    admission wait above have since bounded that (see
    validate_trigger_patterns). This filter is not thereby optional -
    a bound is not an absence, and it is what keeps the KNOWN shapes from
    consuming a bounded resource at all: a pattern recognized here never
    takes a probe slot, never spawns a subprocess, and therefore never
    contributes to the saturation that would refuse a legitimate caller.
    Catching them here is still the only way to make repeated attack
    traffic cheap to reject rather than merely SAFE to reject.

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


def _probe_pattern_is_safe(pattern: str) -> "tuple[str, str]":
    """(verdict, reason) where verdict is _PROBE_SAFE / _PROBE_UNSAFE /
    _PROBE_UNDETERMINED. Never raises.

    ADMISSION CONTROL LIVES HERE, in TWO bounds that answer different questions
    and are both needed. Checking a slot out of _PROBE_SLOTS_FREE is what grants
    the right to run a probe. _TRIGGER_PROBE_SLOT_WAIT then bounds HOW LONG one
    caller may queue for a slot: a caller that cannot get one in that time is
    refused as _PROBE_UNDETERMINED rather than waiting behind the whole queue,
    because the residual this closes is per-caller latency and making the last
    caller in a flood wait 36s "but correctly" would not have fixed anything.
    _TRIGGER_PROBE_MAX_WAITERS bounds HOW MANY may be in that queue at once, and
    a bounded wait does not imply a bounded queue: without it an arbitrary number
    of callers could each be correctly parked for 5s at the same time, on the
    shared default executor other work needs. Refusing at the gate costs
    microseconds and is the only one of the two that a caller cannot make
    expensive.

    _PROBE_UNDETERMINED is NOT a verdict about the pattern and must never be
    cached as one. It says the validator could not answer: the pool was
    saturated, the queue for it was full, or the daemon could not be spawned or
    reached. The request is
    still REJECTED (an unproven pattern never reaches the native sampler), but
    an identical retry a second later can legitimately succeed."""
    # Bind the pool reference ONCE and return the slot to THAT queue, never to
    # whatever _PROBE_SLOTS_FREE names by the time this finishes.
    pool = _PROBE_SLOTS_FREE
    gate = _PROBE_WAITER_GATE
    try:
        # Fast path: a caller the pool can serve right now never becomes a
        # waiter, so it takes no permit and cannot be refused for lack of one.
        # Only a caller that would otherwise block below is counted.
        slot = pool.get_nowait()
    except queue.Empty:
        if not gate.try_enter(_TRIGGER_PROBE_MAX_WAITERS):
            # Refusal because the queue for the pool is full, answered in
            # microseconds. Distinct from the timeout refusal below, which
            # means this caller did queue and its wait ran out.
            return _PROBE_UNDETERMINED, (
                f"the trigger-pattern validator is busy (all "
                f"{_TRIGGER_PROBE_POOL_SIZE} probe slots in use and the "
                f"{_TRIGGER_PROBE_MAX_WAITERS}-caller queue for them is "
                "full); this pattern was not checked, so it was not accepted "
                "- retry shortly")
        try:
            slot = pool.get(timeout=_TRIGGER_PROBE_SLOT_WAIT)
        except queue.Empty:
            return _PROBE_UNDETERMINED, (
                f"the trigger-pattern validator is busy (all "
                f"{_TRIGGER_PROBE_POOL_SIZE} probe slots in use for more than "
                f"{_TRIGGER_PROBE_SLOT_WAIT:.1f}s); this pattern was not checked, "
                "so it was not accepted - retry shortly")
        finally:
            # The permit covers the WAIT and nothing else, so it is released as
            # soon as this caller stops waiting, including on the refusal path.
            gate.leave()
    try:
        return _probe_on_slot(slot, pattern)
    finally:
        # Unconditional: a slot that is not returned is leaked from the pool
        # permanently.
        pool.put(slot)


def _probe_on_slot(slot: "_ProbeSlot", pattern: str) -> "tuple[str, str]":
    """One probe round trip on a slot the caller OWNS. Never raises.

    No lock is held across the round trip, and that is the fix: exclusive access
    to this daemon comes from having checked the slot out of _PROBE_SLOTS_FREE,
    so the up-to-_TRIGGER_PROBE_TIMEOUT wait that REJECTING a dangerous pattern
    costs by construction no longer blocks any other caller. slot.lock is taken
    only for the microseconds needed to claim or install a process, never around
    a spawn and never around _readline_with_timeout."""
    first_spawn = False

    with slot.lock:
        if slot.proc is not None and slot.proc.poll() is None:
            proc = slot.proc
        else:
            if slot.proc is not None:
                try:
                    slot.proc.kill()
                except Exception:
                    pass
                slot.proc = None
            proc = None

    if proc is None:
        # slot.prewarm is read outside slot.lock: thread references are atomic
        # object assignments, and joining a finished thread is a no-op.
        prewarm = slot.prewarm
        if prewarm is not None and prewarm.is_alive():
            prewarm.join(timeout=_TRIGGER_PROBE_SPAWN_TIMEOUT)

        with slot.lock:
            if slot.proc is not None and slot.proc.poll() is None:
                proc = slot.proc
                first_spawn = True

        if proc is None:
            try:
                new_proc = _spawn_trigger_probe_daemon()
            except Exception as e:
                return (_PROBE_UNDETERMINED,
                        f"trigger-pattern probe could not be started ({e})")

            with slot.lock:
                if slot.proc is None or slot.proc.poll() is not None:
                    slot.proc = new_proc
                    proc = new_proc
                    first_spawn = True
                else:
                    try:
                        new_proc.kill()
                    except Exception:
                        pass
                    proc = slot.proc

    if proc.poll() is not None:
        return (_PROBE_UNDETERMINED,
                "trigger-pattern probe process is no longer valid")
    try:
        import json
        proc.stdin.write(json.dumps({"pattern": pattern}) + "\n")
        proc.stdin.flush()
    except Exception as e:
        _slot_kill_and_prewarm(slot)
        return (_PROBE_UNDETERMINED,
                f"trigger-pattern probe could not be reached ({e})")
    timeout = _TRIGGER_PROBE_SPAWN_TIMEOUT if first_spawn else _TRIGGER_PROBE_TIMEOUT
    line = _readline_with_timeout(proc.stdout, timeout)
    if line is None:
        # No reply within the timeout: the daemon either crashed or exited
        # (EOF), or is still alive and stuck inside re.search() on an adversarial
        # probe. The pattern is treated as unsafe either way, and the daemon is
        # killed so a fresh one serves the next caller of this slot.
        #
        # _PROBE_UNSAFE rather than _PROBE_UNDETERMINED, so the verdict is
        # cached: a timeout is the only evidence of danger this check has.
        _slot_kill_and_prewarm(slot)
        return (_PROBE_UNSAFE,
                f"pattern did not pass the safety probe within {timeout:.1f}s - "
                "likely catastrophic regex backtracking")
    line = line.strip()
    if line == "OK":
        return _PROBE_SAFE, ""
    return _PROBE_UNSAFE, (line[4:] if line.startswith("BAD ") else line)


def validate_trigger_patterns(patterns: "list[str]") -> None:
    """Reject a caller-supplied lazy-grammar trigger pattern list that is
    invalid or unsafe to run, BEFORE any of it reaches
    llama_sampler_init_grammar_lazy_patterns - matching
    check_grammar_structure's contract for the grammar string itself
    (raises the same typed InvalidGrammarError so a caller can treat both as
    one clean 400).

    Two layers, cheapest first: _static_shape_rejection (in-process,
    microseconds, catches KNOWN catastrophic shapes - see its own docstring
    for why this is not optional ceremony but load-bearing: a pattern it
    recognizes never occupies a probe slot at all, so known attack traffic
    costs the pool nothing, whereas anything reaching the probe holds a slot
    for as long as it takes to time out), then
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
    pattern).

    CONCURRENT CALLERS DO NOT QUEUE BEHIND EACH OTHER, which is the 2026-07-30
    ruling's residual and is why the pool above exists rather than a single
    daemon behind a single lock. Up to _TRIGGER_PROBE_POOL_SIZE probes run at
    once, a caller that cannot get a slot within _TRIGGER_PROBE_SLOT_WAIT is
    refused outright instead of waiting behind the whole queue, and no more than
    _TRIGGER_PROBE_MAX_WAITERS may be in that wait at any moment. So the cost of
    a flood of N distinct dangerous patterns is bounded per caller rather than
    growing with N, and the number of threads it can occupy is bounded too rather
    than growing with N. MEASURED on this box, 18 concurrent adversarial patterns
    that each hang their probe, at the production 2.0s timeout, every arm in
    steady state (a cold slot legitimately gets _TRIGGER_PROBE_SPAWN_TIMEOUT for
    its FIRST query, so a cold pool measures cold start and is not comparable):

        pool=1, no refusals   36.13s for the last caller, 18 answered
        pool=4, no refusals   10.02s for the last caller, 18 answered
        pool=4, wait=5.0s      6.02s for the last caller, 12 answered + 6
                               refused in milliseconds

    pool=1 is not an approximation of the old code, it IS the old behaviour for
    this property, and 36.13s reproduces the ruling's own 18 x 2.0s = 36s. The
    third arm is what ships. NOTHING WAS WRONGLY ACCEPTED IN ANY ARM: all 18
    patterns were still judged dangerous or refused outright, never passed as
    safe - the latency was bought from serialization, not from the check.

    NOTHING OUTSIDE THIS FUNCTION BOUNDS IT, so the numbers here are the only
    ones there are. Checked rather than assumed: this runs on the asyncio loop's
    TRUE DEFAULT executor (`run_in_executor(None, ...)`), which is NOT one of the
    ~21 sites covered by _threadpool_timeout.run_in_threadpool_bounded, that
    module covers the separate anyio pool; _executor_health.py states plainly
    that the default pool "has no timeout/cancellation mechanism as of this
    writing"; and uvicorn is started with no request-level timeout. So the
    worst case one caller can see is the sum of the waits below, and it used to
    have no ceiling at all:

        _TRIGGER_PROBE_SLOT_WAIT        5.0s   waiting for a slot
      + _TRIGGER_PROBE_SPAWN_TIMEOUT   10.0s   joining an in-flight pre-warm
                                               that never finishes
      + _TRIGGER_PROBE_SPAWN_TIMEOUT   10.0s   first query on the daemon it
                                               then spawns itself
                                     = 25.0s   absolute worst case, per caller

    That path is pre-existing (PR #943's join-rather-than-duplicate-spawn) and
    needs all three worst cases at once; the ordinary bad-pattern cost is
    _TRIGGER_PROBE_TIMEOUT, and the ordinary flood cost is measured above. It is
    quoted anyway because "bounded" is worth nothing without the number, and 25s
    is a long time to hold an HTTP request even if it beats the unbounded queue
    it replaces.

    HOW MANY THREADS THIS CAN HOLD AT ONCE, which is a different question from
    how long each one holds one and used to have no answer: at most
    _TRIGGER_PROBE_POOL_SIZE running plus _TRIGGER_PROBE_MAX_WAITERS queueing,
    i.e. 12 of that shared default executor's threads. A caller arriving on a
    full queue is refused in microseconds as _PROBE_UNDETERMINED rather than
    parked, so a flood wider than 12 costs this server nothing per extra caller.
    Before that cap, "bounded" described only each caller's own latency while an
    arbitrary number of them could be parked at once, which is the shape that
    starves the OTHER work sharing that pool (engine.load, embedding, token
    counting) rather than starving grammar validation.

    WHAT IS STILL NOT SOLVED, stated because a bound is not an absence: 12
    threads is a bound, not zero, and it is a bound on THIS function only. The
    same default pool carries that other blocking work with no equivalent cap of
    its own, so general request-concurrency bounding remains a broader problem
    than this one fix. What changed is that grammar validation can no longer be
    the unbounded contributor to it.

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
    from localm.inference.backends.base import (
        InvalidGrammarError,
        TriggerValidatorUnavailableError,
    )

    for pattern in patterns:
        if pattern in _VALIDATED_TRIGGER_PATTERNS:
            if _VALIDATED_TRIGGER_PATTERNS[pattern] is not None:
                raise InvalidGrammarError(
                    f"grammar trigger pattern rejected: {_VALIDATED_TRIGGER_PATTERNS[pattern]}")
            continue
        static_reason = _static_shape_rejection(pattern)
        if static_reason is not None:
            verdict, reason = _PROBE_UNSAFE, static_reason
        else:
            verdict, reason = _probe_pattern_is_safe(pattern)

        if verdict == _PROBE_UNDETERMINED:
            # Not cached: the cache is keyed on the pattern and lives for the
            # whole process, so an undetermined verdict would reject that
            # pattern for the rest of the process. Still a rejection, and a
            # distinct type so the route answers 503 rather than 400.
            raise TriggerValidatorUnavailableError(
                f"grammar trigger pattern could not be validated: {reason}")

        # Bounded so a flood of distinct patterns cannot grow this dict without
        # limit.
        if len(_VALIDATED_TRIGGER_PATTERNS) >= _MAX_CACHED_TRIGGER_PATTERNS:
            _VALIDATED_TRIGGER_PATTERNS.clear()
        _VALIDATED_TRIGGER_PATTERNS[pattern] = None if verdict == _PROBE_SAFE else reason
        if verdict != _PROBE_SAFE:
            raise InvalidGrammarError(f"grammar trigger pattern rejected: {reason}")

