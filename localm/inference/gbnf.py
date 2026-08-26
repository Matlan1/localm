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
# separated by optional whitespace. Alone it forces tool-only output from the
# first token; with grammar_lazy=True and TOOL_CALL_TRIGGER, free text and
# <think> flow unconstrained until the model starts a <tool_call>, from where
# the call must be valid JSON.
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
# valid tool call, with a bounded reasoning prelude allowed in front of it. The
# opening "<think>" is required when that prelude is used; a model that does not
# open one is held to a tool call from its first token.
#
# think-char withholds only the byte sequence "</t", so reasoning text may
# contain '<' freely. A literal "</td>" inside reasoning ends the block early.
#
# The {0,1900} bound caps the prelude: past 1900 prelude characters the only
# legal continuation is "</think>". The count must stay under llama.cpp's own
# repeat-count ceiling, and moves together with MAX_GRAMMAR_REPEAT_COUNT below.
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
# capture group 1, so the tag must stay INSIDE group 1 or enforcement never
# matches. The pattern must not begin with a lazy wildcard.
TOOL_CALL_TRIGGER = r"(<tool_call>[\s\S]*)"


#  Structural pre-validation

# Bounds on a grammar string, checked before any of it reaches llama.cpp's
# recursive-descent GBNF parser.
MAX_GRAMMAR_BYTES = 65536
MAX_GRAMMAR_NESTING_DEPTH = 128

# Ceiling on a {n,m} repeat count in any grammar. Must stay below llama.cpp's
# own native repeat-count ceiling. Shares its value with
# TOOL_CALLS_AFTER_THINK's think-body bound above.
MAX_GRAMMAR_REPEAT_COUNT = 1900

_REPEAT_COUNT_RE = re.compile(r"\{(\d+)(?:,(\d+))?\}")


def check_grammar_structure(grammar: str) -> None:
    """Reject a grammar whose size or structural complexity could drive the
    native GBNF parser into stack overflow, BEFORE any of it reaches that
    parser. Pure Python, no native call, so it is safe to run unconditionally
    on every grammar-bearing request.

    Raises :class:`InvalidGrammarError`, the same typed error the native
    validation path raises for a malformed grammar."""
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

# Every caller-supplied grammar_triggers pattern is validated here before it
# reaches llama.cpp's native std::regex path. Validation is empirical: the
# candidate is run against fixed adversarial probe strings in a long-lived
# daemon subprocess (inference/_trigger_probe.py), and the caller's own timeout
# on the round trip is what detects a hang.

import queue
import threading as _threading

# Size cap on a caller-supplied trigger pattern, checked FIRST in
# _static_shape_rejection, before any other scan.
MAX_TRIGGER_PATTERN_BYTES = 4096

# How many probes may run at once, and how long a caller may wait for its turn.
# Slots are spawned lazily and handed out LIFO, so a server serving one caller
# at a time keeps exactly one daemon alive.
_TRIGGER_PROBE_POOL_SIZE = 4
_TRIGGER_PROBE_SLOT_WAIT = 5.0

# How many callers may queue for a slot at once, a different bound from how long
# each one queues. A caller served from a free slot never takes a waiter permit,
# so this bounds the QUEUE and never the throughput. Must stay >=
# _TRIGGER_PROBE_POOL_SIZE. The ceiling on default-executor threads held by this
# module is _TRIGGER_PROBE_POOL_SIZE + _TRIGGER_PROBE_MAX_WAITERS.
_TRIGGER_PROBE_MAX_WAITERS = 2 * _TRIGGER_PROBE_POOL_SIZE


class _ProbeSlot:
    """One probe daemon plus the state needed to replace it.

    ``lock`` guards ``proc`` against the background pre-warm thread ONLY, and is
    held just long enough to install or claim a process - NEVER across a spawn
    or a probe round trip. Exclusive use of the daemon comes from having CHECKED
    THE SLOT OUT of _PROBE_SLOTS_FREE, not from holding the lock.
    """

    __slots__ = ("lock", "proc", "prewarm")

    def __init__(self) -> None:
        self.lock = _threading.Lock()
        self.proc = None
        self.prewarm = None


# Free slots, LIFO: the most-recently-returned (and so already-spawned) slot is
# re-handed to the next caller.
_PROBE_SLOTS_FREE: "queue.LifoQueue" = queue.LifoQueue()
for _ in range(_TRIGGER_PROBE_POOL_SIZE):
    _PROBE_SLOTS_FREE.put(_ProbeSlot())


class _WaiterGate:
    """How many callers are QUEUEING for a probe slot right now, and a
    non-blocking way to join them. ``waiting`` reports the current count.

    The limit is passed to try_enter rather than stored at construction, so a
    changed _TRIGGER_PROBE_MAX_WAITERS takes effect immediately.
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
        """Take a waiter permit, or return False immediately. Never blocks."""
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
# UNDETERMINED describes the VALIDATOR and is never cached. UNSAFE and
# UNDETERMINED both reject the request.
_PROBE_SAFE = "safe"
_PROBE_UNSAFE = "unsafe"
_PROBE_UNDETERMINED = "undetermined"

# validate_trigger_patterns's per-process cache.
_VALIDATED_TRIGGER_PATTERNS: "dict[str, str | None]" = {}
_MAX_CACHED_TRIGGER_PATTERNS = 1024

# Per-check timeout against an already-running daemon.
_TRIGGER_PROBE_TIMEOUT = 2.0
# Timeout for the first check after a (re)spawn, which also pays Python
# interpreter and module-import cold-start cost.
_TRIGGER_PROBE_SPAWN_TIMEOUT = 10.0


def _spawn_trigger_probe_daemon():
    """Start the long-lived trigger-pattern-probe daemon. Caller must NOT hold
    any slot lock. Line-buffered text pipes (bufsize=1), so one
    print()/readline() on either side is exactly one protocol message; see
    _trigger_probe.py for the request/response contract. stderr is discarded."""
    import subprocess

    from localm._mp_spawn import interpreter_for_localm_children
    return subprocess.Popen(
        [interpreter_for_localm_children(), "-u", "-m", "localm.inference._trigger_probe"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1)


def _readline_with_timeout(stream, timeout: float):
    """stream.readline() with a timeout: the read runs in a daemon thread and
    this waits on it via a queue, since Windows pipes offer no select()-style
    wait. On timeout the reader thread is abandoned, not joined, so this never
    blocks past *timeout*; the thread exits once the read unblocks or the pipe
    closes. Returns None on timeout, EOF, or any read error."""
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
    lands on nobody's request instead of on whichever caller happens to check
    this slot out next.

    Caller OWNS *slot* (it checked it out of _PROBE_SLOTS_FREE) and must not
    hold slot.lock.

    The background pre-warm thread spawns OUTSIDE any lock and acquires
    slot.lock only to install its finished result. The main thread also spawns
    outside the lock and JOINS an in-flight pre-warm rather than starting a
    duplicate."""
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
            return   # best-effort; the next caller spawns one synchronously
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
    """Cheap, in-process rejection of KNOWN catastrophic-backtracking shapes,
    run BEFORE the daemon probe. Returns a rejection reason string, or None if
    the pattern passes - which does NOT mean the pattern is safe, only that it
    does not match a shape this function recognizes.

    A pattern recognized here never takes a probe slot and never spawns a
    subprocess.

    A length check runs first (see MAX_TRIGGER_PATTERN_BYTES above), before
    either shape check below scans the pattern.

    Two shapes are recognized:

    1. A leading unanchored variable-length wildcard (``.*``, ``.*?``,
       ``.+``, ``.+?``, or a character-class equivalent like ``[\\s\\S]*``)
       at the START of an unanchored pattern. llama.cpp matches any pattern
       that does not begin with ``^`` and end with ``$`` using
       std::regex_search, which already tries every start position, so such a
       wildcard cannot change what matches. A pattern anchored with ``^`` is
       not touched.
    2. A group containing its own unescaped, top-level quantifier
       (``*``/``+``/``{m,n}``), itself immediately followed by another
       quantifier - e.g. ``(a+)+``, ``(a*)*``, ``(x{2,})+``. Detected with a
       linear paren-depth scan, not a full regex parser, so it can miss an
       exotic construction of the same danger and can rarely flag a benign
       pattern using literal unescaped braces. A heuristic, not a proof."""
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

    Admission control lives here. Checking a slot out of _PROBE_SLOTS_FREE
    grants the right to run a probe. _TRIGGER_PROBE_SLOT_WAIT bounds how long
    one caller may queue for a slot; a caller that cannot get one in that time
    is refused as _PROBE_UNDETERMINED. _TRIGGER_PROBE_MAX_WAITERS bounds how
    many may be in that queue at once; a caller arriving on a full queue is
    refused in microseconds.

    _PROBE_UNDETERMINED is NOT a verdict about the pattern and is never cached
    as one. It says the validator could not answer: the pool was saturated, the
    queue for it was full, or the daemon could not be spawned or reached. The
    request is still REJECTED, but an identical retry a second later can
    succeed."""
    # Bind the pool reference ONCE and return the slot to THAT queue, never to
    # whatever _PROBE_SLOTS_FREE names by the time this finishes.
    pool = _PROBE_SLOTS_FREE
    gate = _PROBE_WAITER_GATE
    try:
        # Fast path: a caller the pool can serve right now never becomes a
        # waiter and takes no permit.
        slot = pool.get_nowait()
    except queue.Empty:
        if not gate.try_enter(_TRIGGER_PROBE_MAX_WAITERS):
            # Refusal because the queue for the pool is full, distinct from the
            # timeout refusal below.
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
        # Unconditional: an unreturned slot is leaked from the pool permanently.
        pool.put(slot)


def _probe_on_slot(slot: "_ProbeSlot", pattern: str) -> "tuple[str, str]":
    """One probe round trip on a slot the caller OWNS. Never raises.

    No lock is held across the round trip: exclusive access to this daemon comes
    from having checked the slot out of _PROBE_SLOTS_FREE. slot.lock is taken
    only to claim or install a process, never around a spawn and never around
    _readline_with_timeout."""
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
        # slot.prewarm is read outside slot.lock; joining a finished thread is
        # a no-op.
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
        # No reply within the timeout: the pattern is treated as unsafe, and the
        # daemon is killed so a fresh one serves the next caller of this slot.
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
    llama_sampler_init_grammar_lazy_patterns.

    Raises the same typed InvalidGrammarError as check_grammar_structure for a
    rejected pattern, and TriggerValidatorUnavailableError when the validator
    itself could not answer.

    Two layers, cheapest first: _static_shape_rejection (in-process,
    microseconds, catches KNOWN catastrophic shapes; a pattern it recognizes
    never occupies a probe slot at all), then _probe_pattern_is_safe (the daemon
    round trip) for whatever survives. Neither layer is complete: a NOVEL
    catastrophic pattern that matches no known static shape AND is not slow
    against _trigger_probe.py's adversarial corpus passes both layers and
    reaches the native sampler.

    This function BLOCKS for up to a few seconds when a pattern survives the
    static filter but is genuinely dangerous. Callers on an asyncio route MUST
    run it via loop.run_in_executor, never directly from an async handler.

    Concurrent callers do not queue behind each other: up to
    _TRIGGER_PROBE_POOL_SIZE probes run at once, a caller that cannot get a slot
    within _TRIGGER_PROBE_SLOT_WAIT is refused outright, and no more than
    _TRIGGER_PROBE_MAX_WAITERS may be in that wait at any moment. The ordinary
    bad-pattern cost is _TRIGGER_PROBE_TIMEOUT; the per-caller worst case is
    _TRIGGER_PROBE_SLOT_WAIT plus two _TRIGGER_PROBE_SPAWN_TIMEOUTs (25.0s),
    which needs all three worst cases at once. At most
    _TRIGGER_PROBE_POOL_SIZE + _TRIGGER_PROBE_MAX_WAITERS threads of the shared
    default executor are held by this function at any moment.

    Each unique pattern's final verdict (from either layer) is cached for this
    process's lifetime, keyed on the exact pattern string. The cache is not
    itself a safety mechanism: a caller sending a fresh pattern on every request
    bypasses it, and the static filter plus the probe still run on every pattern
    this process has not already validated."""
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
            # Not cached. Still a rejection, and a distinct type so the route
            # answers 503 rather than 400.
            raise TriggerValidatorUnavailableError(
                f"grammar trigger pattern could not be validated: {reason}")

        # Bounded so a flood of distinct patterns cannot grow this dict without
        # limit.
        if len(_VALIDATED_TRIGGER_PATTERNS) >= _MAX_CACHED_TRIGGER_PATTERNS:
            _VALIDATED_TRIGGER_PATTERNS.clear()
        _VALIDATED_TRIGGER_PATTERNS[pattern] = None if verdict == _PROBE_SAFE else reason
        if verdict != _PROBE_SAFE:
            raise InvalidGrammarError(f"grammar trigger pattern rejected: {reason}")

