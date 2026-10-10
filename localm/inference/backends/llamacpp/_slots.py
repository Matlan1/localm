# SPDX-License-Identifier: AGPL-3.0-or-later
"""Continuous batching for one loaded GGUF model: several text replies decode
together, one KV sequence each, in a single ``llama_decode`` per step.

:class:`SlotScheduler` owns ``n_slots`` sequences of one context whose KV cache
is unified (``kv_unified``), so all sequences draw on the same ``n_ctx`` cells.
One scheduler thread makes every native call. A caller :meth:`~SlotScheduler.submit`\\ s
a prompt with its own sampler and reads token ids from the returned
:class:`SlotStream`.

Each step builds one batch: the pending token of every decoding slot, then
prompt chunks of prefilling slots until ``n_batch`` tokens. After the decode,
every slot whose last row was computed samples its next token with its own
sampler (so a grammar applies to its own sequence only).

Admission reserves ``prompt + budget + RESERVE_PAD`` cells per request
(``prompt + UNLIMITED_STEP + RESERVE_PAD`` for a reply with no budget, extended
as it runs), never more than the context ceiling. A request that does not fit
first takes the cells idle slots keep for prefix reuse (least recently used
first), then a bigger context, and otherwise waits, FIFO, while the other
replies run. Growing the context with replies in flight re-decodes their
cached tokens into the new context, and only happens when the VRAM check says
the bigger cache fits on the GPU.

Lock order: the model's ``LlamaCpp._gen_lock`` (``ops.lock()``) is taken
before ``SlotScheduler._cond``, never after. See
test_exclusive_section_waits_for_active_slots_without_deadlock.
"""
from __future__ import annotations

import collections
import contextlib
import ctypes
import itertools
import queue
import threading
import time
from typing import Any, Callable, Iterator, Optional

from localm.inference.backends.base import WAITING_FOR_MODEL_STATUS

GENERATING_STATUS = "Generating response..."
UNLOADED_MESSAGE = "Model was unloaded during generation - request aborted."

RESERVE_PAD = 64          # cells reserved past prompt + budget
UNLIMITED_STEP = 512      # cells a reply with no budget reserves at a time
WAIT_HEARTBEAT_S = 5.0    # a waiting request receives its status this often
POLL_S = 0.25             # how often a reader blocked on its stream polls stop
BATCH_CAP = 2048          # most tokens one decode carries

# (token, position, sequence id, wants logits)
Entry = tuple[int, int, int, bool]


class _Status:
    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


class _End:
    __slots__ = ("reason", "error")

    def __init__(self, reason: str, error: Optional[BaseException] = None) -> None:
        self.reason = reason
        self.error = error


class SlotStream:
    """One submitted reply, iterated for its token ids.

    ``finish_reason`` is "stop" (end of generation, a cancel, or a stop the
    caller requested) or "length" (the budget or the context ran out) once
    iteration ends. Iteration raises the exception that ended the reply when
    one did. Status texts reach *on_status* on the iterating thread; the
    waiting status repeats every ``WAIT_HEARTBEAT_S`` while the request waits.

    :meth:`close` cancels the reply; it is idempotent and safe from any
    thread. While blocked waiting for an item, the iterator polls
    *stop_requested* every ``POLL_S`` seconds and closes itself once it
    returns True."""

    def __init__(self, scheduler: SlotScheduler, prompt: list[int], budget: int,
                 sampler: Any, reserve: int,
                 on_status: Optional[Callable[[str], None]],
                 stop_requested: Optional[Callable[[], bool]]) -> None:
        self.prompt = prompt
        self.budget = budget
        self.sampler = sampler
        self.reserve = reserve
        self.finish_reason = "stop"
        self.cancelled = threading.Event()
        self.waiting_heard_at: Optional[float] = None
        self._scheduler = scheduler
        self._items: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._on_status = on_status
        self._stop_requested = stop_requested
        self._finished = False

    def _put(self, item: Any) -> None:
        self._items.put(item)

    def __iter__(self) -> SlotStream:
        return self

    def __next__(self) -> int:
        if self._finished:
            raise StopIteration
        while True:
            try:
                item = self._items.get(timeout=POLL_S)
            except queue.Empty:
                if (not self.cancelled.is_set() and self._stop_requested is not None
                        and self._stop_requested()):
                    self.close()
                continue
            if isinstance(item, int):
                return item
            if isinstance(item, _Status):
                if self._on_status is not None:
                    self._on_status(item.text)
                continue
            self._finished = True
            self.finish_reason = item.reason
            if item.error is not None:
                raise item.error
            raise StopIteration

    def close(self) -> None:
        if not self.cancelled.is_set():
            self.cancelled.set()
            self._scheduler.wake()


class _Slot:
    """One KV sequence and the reply it is serving, if any."""

    def __init__(self, seq: int) -> None:
        self.seq = seq
        self.kv: list[int] = []           # tokens in the KV cache at positions 0..len-1
        self.stream: Optional[SlotStream] = None
        self.todo: list[int] = []         # prompt tokens still to prefill
        self.pending: Optional[int] = None  # emitted token not yet decoded
        self.generated = 0
        self.reserve = 0
        self.first = False                # the next sample is the reply's first
        self.order = 0                    # admission order
        self.last_used = 0.0


_WAIT = object()


class SlotScheduler:
    """Decodes the replies of up to ``n_slots`` requests together.

    *ops* is the native seam (:class:`LlamaSlotOps` for a loaded model):
    ``lock()``, ``stopped()``, ``capacity()``, ``max_capacity()``,
    ``target_ctx(n)``, ``vram_fit(n)``, ``recreate(n, offload_kqv)``,
    ``n_batch()``, ``decode(entries)``, ``sample(sampler, row)``,
    ``is_eog(tok)``, ``seq_rm(seq, p0, p1)``, ``clear_memory()``,
    ``free_sampler(sampler)``, ``stderr_scope()``.

    Not reentrant from the scheduler thread. Thread-safe otherwise."""

    def __init__(self, ops: Any, n_slots: int, *,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if n_slots < 1:
            raise ValueError("n_slots must be at least 1")
        self._ops = ops
        self._slots = [_Slot(i) for i in range(n_slots)]
        self._queue: collections.deque[SlotStream] = collections.deque()
        # Lock order: ops.lock() (LlamaCpp._gen_lock) before _cond, never after.
        self._cond = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self._exclusive_waiting = 0
        self._exclusive_held = False
        self._dirty = True      # the KV cache may hold tokens no slot records
        self._touched = False   # a slot decoded since the last exclusive section
        self._clock = clock
        self._admissions = itertools.count()

    @property
    def n_slots(self) -> int:
        return len(self._slots)

    # ------------------------------------------------------------------ #
    #  Caller side                                                         #
    # ------------------------------------------------------------------ #

    def submit(self, prompt: list[int], budget: int, sampler: Any, *,
               on_status: Optional[Callable[[str], None]] = None,
               stop_requested: Optional[Callable[[], bool]] = None) -> SlotStream:
        """Queue a reply to *prompt* of at most *budget* tokens (<= 0 for no
        budget), sampled with *sampler*. The scheduler owns *sampler* once this
        returns and frees it when the reply ends; when this raises, the caller
        still owns it. Raises RuntimeError once the scheduler is closed."""
        if not prompt:
            raise ValueError("prompt must not be empty")
        reserve = len(prompt) + (budget if budget > 0 else UNLIMITED_STEP) + RESERVE_PAD
        ceiling = self._ops.max_capacity()
        if ceiling:
            reserve = min(reserve, ceiling)
        stream = SlotStream(self, list(prompt), budget, sampler, reserve,
                            on_status, stop_requested)
        with self._cond:
            if self._closed:
                raise RuntimeError(UNLOADED_MESSAGE)
            self._queue.append(stream)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="localm-gguf-slots", daemon=True)
                self._thread.start()
            self._cond.notify_all()
        return stream

    def wake(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def counts(self) -> tuple[int, int]:
        """(replies decoding or prefilling, replies waiting)."""
        with self._cond:
            return (sum(1 for s in self._slots if s.stream is not None),
                    len(self._queue))

    @contextlib.contextmanager
    def exclusive(self, on_wait: Optional[Callable[[], None]] = None) -> Iterator[None]:
        """Hold the model alone: admits nothing new, waits until every active
        reply has ended, empties the KV cache when a slot used it since the last
        exclusive section, and on exit marks the cache as holding tokens no slot
        records. While it waits, *on_wait* is called at once and then every
        ``WAIT_HEARTBEAT_S`` seconds. Raises RuntimeError once the scheduler is
        closed."""
        with self._cond:
            if self._closed:
                raise RuntimeError(UNLOADED_MESSAGE)
            self._exclusive_waiting += 1
        try:
            while True:
                with self._cond:
                    if self._closed:
                        self._cond.notify_all()
                        raise RuntimeError(UNLOADED_MESSAGE)
                    if not (self._exclusive_held or self._busy()):
                        self._exclusive_held = True
                        break
                if on_wait is not None:
                    on_wait()
                with self._cond:
                    if self._exclusive_held or self._busy():
                        self._cond.wait(WAIT_HEARTBEAT_S)
        finally:
            with self._cond:
                self._exclusive_waiting -= 1
                self._cond.notify_all()
        try:
            with self._ops.lock():
                if self._touched:
                    self._ops.clear_memory()
                    self._touched = False
                for slot in self._slots:
                    slot.kv = []
            yield
        finally:
            with self._cond:
                self._exclusive_held = False
                self._dirty = True
                self._cond.notify_all()

    def close(self, timeout: float = 10.0) -> None:
        """Stop the scheduler: every queued and active reply ends with a
        RuntimeError. Waits up to *timeout* seconds for the scheduler thread,
        which frees the samplers it owns."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        if thread is None or not thread.is_alive():
            self._abort(RuntimeError(UNLOADED_MESSAGE))

    # ------------------------------------------------------------------ #
    #  Scheduler thread                                                    #
    # ------------------------------------------------------------------ #

    def _busy(self) -> bool:
        return any(s.stream is not None for s in self._slots)

    def _admitting(self) -> bool:
        return not (self._exclusive_waiting or self._exclusive_held)

    def _has_work(self) -> bool:
        return self._busy() or (bool(self._queue) and self._admitting())

    def _run(self) -> None:
        from localm.debuglog import logger
        try:
            running = True
            while running:
                with self._cond:
                    while not self._closed and not self._has_work():
                        self._reap_queued()
                        self._heartbeat_queued()
                        self._cond.wait(WAIT_HEARTBEAT_S if self._queue else None)
                    if self._closed:
                        break
                with self._ops.stderr_scope():
                    while True:
                        with self._cond:
                            if self._closed or not self._has_work():
                                break
                        if not self._step():
                            running = False
                            break
        except BaseException as exc:
            logger.exception("gguf slots: scheduler failed")
            self._abort(exc)
            return
        self._abort(RuntimeError(UNLOADED_MESSAGE))

    def _abort(self, error: BaseException) -> None:
        """End every active and queued reply with *error*, freeing their
        samplers."""
        with self._cond:
            self._closed = True
            queued = list(self._queue)
            self._queue.clear()
            active = [s for s in self._slots if s.stream is not None]
            self._cond.notify_all()
        for slot in active:
            self._end(slot, "error", error, drop=False)
        for stream in queued:
            self._release(stream, _End("error", error))

    def _step(self) -> bool:
        """One scheduling step. False when the scheduler must stop."""
        with self._ops.lock():
            if self._ops.stopped():
                return False
            self._reap()
            with self._cond:
                admitting = self._admitting()
            if admitting:
                self._admit()
            plan = self._compose()
            if not plan:
                return True
            self._touched = True
            entries = [e for part in plan.values() for e in part[0]]
            rc = self._ops.decode(entries)
            if rc == 0:
                offset = 0
                rows = {}
                for seq, (part, row) in plan.items():
                    if row is not None:
                        rows[seq] = offset + row
                    offset += len(part)
                self._commit(plan)
                self._sample(rows)
            else:
                self._isolate(plan, rc)
        return True

    # -- admission ------------------------------------------------------- #

    def _reap(self) -> None:
        for slot in self._slots:
            if slot.stream is not None and slot.stream.cancelled.is_set():
                self._end(slot, "stop")
        with self._cond:
            self._reap_queued()

    def _reap_queued(self) -> None:
        """End the queued replies that were cancelled. Caller holds _cond."""
        gone = [s for s in self._queue if s.cancelled.is_set()]
        for stream in gone:
            self._queue.remove(stream)
            self._release(stream, _End("stop"))

    def _heartbeat_queued(self) -> None:
        now = self._clock()
        for stream in self._queue:
            heard = stream.waiting_heard_at
            if heard is None or now - heard >= WAIT_HEARTBEAT_S:
                stream.waiting_heard_at = now
                stream._put(_Status(WAITING_FOR_MODEL_STATUS))

    def _admit(self) -> None:
        if self._dirty:
            self._ops.clear_memory()
            for slot in self._slots:
                slot.kv = []
            self._dirty = False
            self._touched = True
        while True:
            with self._cond:
                if not self._queue:
                    return
                stream = self._queue[0]
            free = [s for s in self._slots if s.stream is None]
            if not free:
                with self._cond:
                    self._heartbeat_queued()
                return
            slot = self._pick(free, stream.prompt)
            fit = self._fit(stream.reserve, slot)
            if fit is _WAIT:
                with self._cond:
                    self._heartbeat_queued()
                return
            with self._cond:
                self._queue.popleft()
            if isinstance(fit, BaseException):
                self._release(stream, _End("error", fit))
                continue
            self._assign(slot, stream)

    def _pick(self, free: list[_Slot], prompt: list[int]) -> _Slot:
        """The free slot whose cache shares the longest prefix with *prompt*;
        among equals an empty slot, then the least recently used."""
        best = None
        best_key = None
        for slot in free:
            prefix = _common_prefix_len(slot.kv, prompt)
            key = (prefix, 1 if not slot.kv else 0, -slot.last_used)
            if best_key is None or key > best_key:
                best, best_key = slot, key
        assert best is not None
        return best

    def _committed(self) -> int:
        return sum(s.reserve for s in self._slots if s.stream is not None)

    def _fit(self, need: int, slot: _Slot) -> Any:
        """True when *need* cells fit beside the active replies (after dropping
        idle caches or growing the context), ``_WAIT`` when the request has to
        wait for active replies, or the exception a failed growth raised."""
        cap = self._ops.capacity()
        committed = self._committed()
        idle = [s for s in self._slots
                if s.stream is None and s is not slot and s.kv]
        if committed + need + sum(len(s.kv) for s in idle) <= cap:
            return True
        if committed + need <= cap:
            for other in sorted(idle, key=lambda s: s.last_used):
                self._drop_cache(other)
                if committed + need + sum(len(s.kv) for s in idle) <= cap:
                    break
            return True
        total = committed + need
        target = self._ops.target_ctx(total)
        if target < total:
            return _WAIT
        busy = committed > 0
        decision = self._ops.vram_fit(target)
        if busy and decision is False:
            return _WAIT
        try:
            self._grow(target, decision is not False)
        except Exception as exc:
            return exc
        return True

    def _grow(self, target: int, offload_kqv: bool) -> None:
        """Recreate the context with *target* cells and re-decode the cached
        tokens of every active reply into it. Idle caches are dropped. A failed
        recreate ends every active reply with its exception and re-raises."""
        from localm.debuglog import logger
        active = [s for s in self._slots if s.stream is not None]
        logger.info("gguf slots: growing the context to %d tokens (%d active)",
                    target, len(active))
        try:
            self._ops.recreate(target, offload_kqv)
        except Exception as exc:
            for slot in self._slots:
                slot.kv = []
            for slot in active:
                self._end(slot, "error", exc, drop=False)
            raise
        self._touched = True
        for slot in self._slots:
            if slot.stream is None:
                slot.kv = []
        for slot in active:
            tokens, slot.kv = slot.kv, []
            step = max(1, self._ops.n_batch())
            for i in range(0, len(tokens), step):
                chunk = tokens[i:i + step]
                entries = [(tok, i + j, slot.seq, j == len(chunk) - 1)
                           for j, tok in enumerate(chunk)]
                rc = self._ops.decode(entries)
                if rc != 0:
                    self._end(slot, "error", RuntimeError(
                        f"llama_decode failed while rebuilding the cache (code {rc})"))
                    break
                slot.kv.extend(chunk)

    def _assign(self, slot: _Slot, stream: SlotStream) -> None:
        from localm.debuglog import logger
        keep = min(_common_prefix_len(slot.kv, stream.prompt), len(stream.prompt) - 1)
        if keep <= 0:
            keep = 0
            if slot.kv:
                self._drop_cache(slot)
        elif keep < len(slot.kv):
            if not self._ops.seq_rm(slot.seq, keep, -1):
                self._drop_cache(slot)
                keep = 0
        slot.kv = slot.kv[:keep]
        slot.todo = list(stream.prompt[keep:])
        slot.pending = None
        slot.generated = 0
        slot.first = True
        slot.reserve = stream.reserve
        slot.order = next(self._admissions)
        with self._cond:
            slot.stream = stream
        logger.info("gguf slots: slot %d prefill starting, %d prompt token(s), %d reused",
                    slot.seq, len(stream.prompt), keep)

    def _drop_cache(self, slot: _Slot) -> None:
        if slot.kv and not self._ops.seq_rm(slot.seq, 0, -1):
            from localm.debuglog import logger
            logger.warning("gguf slots: sequence %d could not be removed from the "
                           "cache; the next request clears the whole cache", slot.seq)
            self._dirty = True
        slot.kv = []

    # -- one decode ------------------------------------------------------ #

    def _active(self) -> list[_Slot]:
        return sorted((s for s in self._slots if s.stream is not None),
                      key=lambda s: s.order)

    def _compose(self) -> dict[int, tuple[list[Entry], Optional[int]]]:
        """Per sequence: its batch entries and the index of its sampled row
        within them (None while its prompt is still being prefilled)."""
        plan: dict[int, tuple[list[Entry], Optional[int]]] = {}
        active = self._active()
        for slot in list(active):
            if slot.pending is None and not slot.todo:
                self._end(slot, "error", RuntimeError(
                    "the reply had nothing left to decode"))
                active.remove(slot)
        for slot in active:
            if slot.pending is not None:
                plan[slot.seq] = ([(slot.pending, len(slot.kv), slot.seq, True)], 0)
        room = min(self._ops.n_batch(), BATCH_CAP) - len(plan)
        for slot in active:
            if slot.pending is not None or not slot.todo or room <= 0:
                continue
            n = min(room, len(slot.todo))
            done = n == len(slot.todo)
            base = len(slot.kv)
            part = [(tok, base + i, slot.seq, done and i == n - 1)
                    for i, tok in enumerate(slot.todo[:n])]
            plan[slot.seq] = (part, n - 1 if done else None)
            room -= n
        return plan

    def _commit(self, plan: dict[int, tuple[list[Entry], Optional[int]]]) -> None:
        for seq, (part, _row) in plan.items():
            slot = self._slots[seq]
            if slot.pending is not None:
                slot.kv.append(slot.pending)
                slot.pending = None
            else:
                slot.kv.extend(slot.todo[:len(part)])
                slot.todo = slot.todo[len(part):]

    def _sample(self, rows: dict[int, int]) -> None:
        for seq, row in rows.items():
            slot = self._slots[seq]
            stream = slot.stream
            if stream is None:
                continue
            try:
                token = self._ops.sample(stream.sampler, row)
            except OSError as exc:
                self._end(slot, "error", exc)
                continue
            if slot.first:
                slot.first = False
                stream._put(_Status(GENERATING_STATUS))
            if self._ops.is_eog(token):
                self._end(slot, "stop")
                continue
            stream._put(token)
            slot.generated += 1
            if stream.budget > 0 and slot.generated >= stream.budget:
                self._end(slot, "length")
                continue
            slot.pending = token
            if len(slot.kv) + 1 > slot.reserve and not self._extend(slot):
                self._end(slot, "length")

    def _extend(self, slot: _Slot) -> bool:
        """Reserve ``UNLIMITED_STEP`` more cells for a reply with no budget.
        False when neither the context nor its ceiling has room."""
        if slot.stream is None or slot.stream.budget > 0:
            return False
        ceiling = self._ops.max_capacity()
        want = slot.reserve + UNLIMITED_STEP
        if ceiling:
            want = min(want, ceiling)
        if want <= slot.reserve:
            return False
        extra = want - slot.reserve
        cap = self._ops.capacity()
        committed = self._committed()
        idle = [s for s in self._slots if s.stream is None and s.kv]
        if committed + extra > cap:
            for other in sorted(idle, key=lambda s: s.last_used):
                self._drop_cache(other)
            total = committed + extra
            target = self._ops.target_ctx(total)
            if target < total:
                return False
            others = sum(1 for s in self._slots if s.stream is not None) > 1
            decision = self._ops.vram_fit(target)
            if others and decision is False:
                return False
            try:
                self._grow(target, decision is not False)
            except Exception:
                return False
            if slot.stream is None:
                return False
        elif committed + extra + sum(len(s.kv) for s in idle) > cap:
            for other in sorted(idle, key=lambda s: s.last_used):
                self._drop_cache(other)
                if committed + extra + sum(len(s.kv) for s in idle) <= cap:
                    break
        slot.reserve = want
        return True

    def _isolate(self, plan: dict[int, tuple[list[Entry], Optional[int]]], rc: int) -> None:
        """The combined decode failed with *rc*: decode each sequence's part on
        its own so one failing sequence does not end the others."""
        from localm.debuglog import logger
        logger.warning("gguf slots: a decode of %d sequence(s) failed (code %d); "
                       "decoding them one at a time", len(plan), rc)
        for seq, (part, row) in plan.items():
            slot = self._slots[seq]
            if slot.stream is None:
                continue
            code = self._ops.decode(part)
            if code == 0:
                self._commit({seq: (part, row)})
                if row is not None:
                    self._sample({seq: row})
                continue
            if slot.pending is not None:
                logger.info("gguf slots: slot %d reply cut short (decode code %d)",
                            seq, code)
                self._end(slot, "length", drop=True)
            else:
                self._end(slot, "error", RuntimeError(
                    f"llama_decode failed during prefill (code {code})"))

    def _end(self, slot: _Slot, reason: str, error: Optional[BaseException] = None,
             *, drop: Optional[bool] = None) -> None:
        """Finish *slot*'s reply with *reason* (and *error*). Its cache is kept
        for prefix reuse unless *drop* (default: when there is an error)."""
        stream = slot.stream
        if stream is None:
            return
        if drop is None:
            drop = error is not None
        if drop:
            self._drop_cache(slot)
        slot.pending = None
        slot.todo = []
        slot.first = False
        slot.reserve = 0
        slot.last_used = self._clock()
        with self._cond:
            slot.stream = None
            self._cond.notify_all()
        self._release(stream, _End(reason, error))

    def _release(self, stream: SlotStream, end: _End) -> None:
        sampler, stream.sampler = stream.sampler, None
        if sampler is not None:
            try:
                self._ops.free_sampler(sampler)
            except Exception:
                from localm.debuglog import logger
                logger.debug("gguf slots: freeing a sampler raised", exc_info=True)
        stream._put(end)


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


class LlamaSlotOps:
    """The native seam of :class:`SlotScheduler` over a loaded ``LlamaCpp``."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def lock(self):
        return self._llm._gen_lock

    def stopped(self) -> bool:
        return self._llm._stop.is_set() or not self._llm._model_ptr

    def capacity(self) -> int:
        llm = self._llm
        return int(llm._ctx_capacity) if llm._ctx_ptr else 0

    def max_capacity(self) -> Optional[int]:
        return self._llm._n_ctx_max or None

    def target_ctx(self, needed: int) -> int:
        return self._llm._target_ctx(needed)

    def vram_fit(self, target: int) -> Optional[bool]:
        check = getattr(self._llm, "_vram_check", None)
        if check is None:
            return None
        return check(target, self.capacity())

    def recreate(self, target: int, offload_kqv: bool) -> None:
        self._llm._recreate_slots_context(target, offload_kqv)

    def n_batch(self) -> int:
        return min(self.capacity() or BATCH_CAP, BATCH_CAP)

    def decode(self, entries: list[Entry]) -> int:
        from . import _api as api
        from ._structs import llama_token
        llm = self._llm
        if not llm._ctx_ptr:
            return -1
        n = len(entries)
        batch = api.llama_batch_init(n, 0, 1)
        try:
            batch.n_tokens = n
            tok = ctypes.cast(batch.token, ctypes.POINTER(llama_token))
            pos = ctypes.cast(batch.pos, ctypes.POINTER(ctypes.c_int32))
            n_seq = ctypes.cast(batch.n_seq_id, ctypes.POINTER(ctypes.c_int32))
            seq = ctypes.cast(batch.seq_id, ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)))
            logits = ctypes.cast(batch.logits, ctypes.POINTER(ctypes.c_int8))
            for i, (t, p, s, want) in enumerate(entries):
                tok[i] = t
                pos[i] = p
                n_seq[i] = 1
                seq[i][0] = s
                logits[i] = 1 if want else 0
            return int(api.llama_decode(llm._ctx_ptr, batch))
        finally:
            api.llama_batch_free(batch)

    def sample(self, sampler: Any, row: int) -> int:
        from . import _api as api
        return int(api.llama_sampler_sample(sampler, self._llm._ctx_ptr, row))

    def is_eog(self, token: int) -> bool:
        return bool(self._llm._tokenizer.is_eog(token))

    def seq_rm(self, seq: int, p0: int, p1: int) -> bool:
        from . import _api as api
        llm = self._llm
        if not llm._ctx_ptr:
            return True
        return bool(api.llama_memory_seq_rm(api.llama_get_memory(llm._ctx_ptr), seq, p0, p1))

    def clear_memory(self) -> None:
        from . import _api as api
        llm = self._llm
        llm._cached_tokens = []
        if llm._ctx_ptr:
            api.llama_memory_clear(api.llama_get_memory(llm._ctx_ptr), True)

    def free_sampler(self, sampler: Any) -> None:
        from . import _api as api
        api.llama_sampler_free(sampler)

    def stderr_scope(self):
        from .llama import _stderr_ctx_for_generate
        return _stderr_ctx_for_generate(self._llm._verbose)()
