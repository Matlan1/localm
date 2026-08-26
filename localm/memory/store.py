# SPDX-License-Identifier: AGPL-3.0-or-later
"""
The per-namespace memory store: persistence + retrieval + forgetting.

One store == one ``(principal, agent, scope_key)`` namespace, backed by a single
JSONL file (one record per line) under ``<home>/memory/<agent>/<ns>.jsonl`` with
an OPTIONAL aligned vector sidecar ``<ns>.vec.json`` (``{"dim", "vectors": {id:
vec}}``). Keying vectors by record id (not position) keeps them correct across
edits and deletes.

Design mirrors ``localm/rag/store.py``: home-scale JSON, atomic tmp+replace, BM25
always-available, embeddings OPTIONAL, and a per-namespace lock so concurrent
writers cannot silently clobber each other. It stays SMALL: consolidation + decay
+ the ``N_MAX`` cap keep a namespace to a few hundred distilled records, never a
transcript, so whole-file rewrites are cheap.

Retrieval blends the Generative-Agents signals (Park et al. 2023): relevance
(lexical BM25, optionally 50/50 with embedding cosine when an embedder is present),
recency (exponential decay since last use), and importance (write-time salience).
This store imports no session/audit state; the privacy gate lives with the caller
(see ``gating.writes_allowed`` and the ``reinforce`` flag on ``recall``).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from localm.jsonl import split_jsonl
from localm.rag import BM25
from localm.rag.collection_lock import CollectionLockedError
from localm.rag.bm25 import tokenize as _tokenize
from localm.rag.store import _cosine
from localm.storekit import NamespaceLockRegistry, atomic_write as _storekit_atomic_write

from .corrections import PendingCorrection
from .record import MemoryRecord

EmbedFn = Callable[[list[str]], list[list[float]]]

# ---- on-disk format -------------------------------------------------------- #
# Stamped on every JSONL line at save ("v"). Load tolerates lines without it
# (files written before versioning are format 1). Bump this on a breaking
# MemoryRecord schema change and branch on the stored value in _load(): _save()
# rewrites the whole file, so an unrecognized line that is merely skipped is
# erased by the next write.
#
# Also reused, unchanged, as the "v" stamp for the three correction sidecars
# (.corrections.jsonl, .forgotten.jsonl, .corrections-dismissed.json) - a SHARED
# counter across four independent schemas, not a per-schema one. A bump for a
# MemoryRecord-only change also stamps a new value onto the sidecars even though
# their own schemas did not change; any version-gated migration written against
# "v" must branch per-schema (which file it came from), not assume the raw number
# alone identifies a MemoryRecord revision.
FORMAT_VERSION = 1

# ---- bounds (DoS / poisoning / bloat) ------------------------------------- #
MAX_TEXT_LEN = 500          # per-record text cap (truncate on store)
N_MAX = 256                 # records per namespace; prune evicts the weakest
K_CAP = 32                  # hard ceiling on a recall k

# ---- retrieval scoring ---------------------------------------------------- #
W_REL, W_REC, W_IMP = 0.5, 0.3, 0.2     # blend weights (sum 1.0)
# Within the relevance signal, semantic cosine is weighted ABOVE lexical BM25 when
# an embedder is present: paraphrased factual queries share few tokens with the
# stored fact, so BM25 is near-zero and noisy. Some BM25 is kept so exact-term
# queries (ids, filenames) still hit.
REL_LEX_SHARE = 0.20       # BM25 share of relevance when vectors present (rest = cosine)
TAU_DAYS = 30.0             # recency e-folding time: 30d -> 0.37, 90d -> 0.05
FLOOR = 0.05               # a recalled memory must beat this normalized score
TINY_CORPUS = 8            # below this, BM25 idf is noisy -> rank by rec+imp only
VEC_COVERAGE = 0.8         # blend cosine only when >= this fraction have vectors

# ---- absolute relevance gate (recall precision) --------------------------- #
# Recall injects a memory ONLY when it actually relates to the query, mirroring the
# coder's episode gate (plugins/coder/episodes.py), so an OFF-TOPIC turn injects
# NOTHING. Gates are ABSOLUTE (not max-normalised) so silence-when-irrelevant
# holds:
#   lexical: the query shares a CONTENT word (stopwords removed) with the record, OR
#   semantic: the raw cosine to the record clears REL_COS_MIN (when vectors usable).
# The blended score below still RANKS the eligible records; the gate only decides
# eligibility. The pinned user/import recency keeps a relevant durable fact
# prominent among the eligible records; an off-topic user fact fails the gate
# regardless.
# Absolute cosine floor for the semantic gate, matching the coder episode gate.
REL_COS_MIN = 0.55
# Stopwords stripped from the LEXICAL gate: a query and a fact sharing only "the"
# must NOT clear it. Mirrors the coder episode store's _STOPWORDS.
_STOPWORDS = frozenset(
    "a an and are as at be been but by can could did do does done for from had has "
    "have he her him his i if in into is it its me my no not of on only or our over "
    "own same she should so some such than that the their them then there these they "
    "this to too under up us very was we were what when where which who will with "
    "would you your".split()
)


def _content_tokens(text: str) -> set:
    """Lowercased CONTENT-word token set (stopwords removed) for the lexical relevance
    gate. Reuses the shared rag tokenizer (unicode-aware) so CJK/accented queries work;
    empty when *text* is all stopwords/punctuation."""
    return {t for t in _tokenize(text or "") if t not in _STOPWORDS}


# ---- self-reference discriminator ------------------------------------------ #
# First-person pronouns, read from the RAW query BEFORE _STOPWORDS is applied -
# every one of these IS a stopword (see _STOPWORDS above). TRUSTED_SOURCES records
# are PROFILE facts, so they are on-topic exactly when the user asks about
# themselves.
#
# A HEURISTIC, not relevance: "is the metric system better?" is about the user's
# stored preference but carries no pronoun, so it still misses.
_SELF_REF = frozenset({"i", "me", "my", "mine", "myself"})

# "my" does NOT mean "about me" when it introduces ANOTHER PERSON: a possessive
# naming a RELATIONSHIP is about that other person, so it must not open the
# profile-fact fallback. "my name", "my birthday", "my preference" are untouched
# and still self-referential.
_OTHER_PERSON = frozenset({
    "friend", "friends", "colleague", "colleagues", "coworker", "coworkers",
    "boss", "manager", "neighbour", "neighbor", "neighbours", "neighbors",
    "wife", "husband", "partner", "spouse", "girlfriend", "boyfriend",
    "mother", "mum", "mom", "father", "dad", "parents", "brother", "sister",
    "sibling", "siblings", "son", "daughter", "child", "children", "kids",
    "cousin", "uncle", "aunt", "nephew", "niece", "grandma", "grandmother",
    "grandpa", "grandfather", "family", "mate", "buddy", "pal", "team",
    "guest", "visitor", "client", "customer", "student", "teacher", "doctor",
})
_POSSESSIVE = frozenset({"my", "mine"})


def _is_self_referential(text: str) -> bool:
    """True when *text* refers to the asker in the first person, i.e. the query is
    plausibly ABOUT the user rather than about the world.

    A possessive immediately followed by a word naming ANOTHER PERSON ("my friend",
    "my boss") does not count: that query is about them, not about the asker, so it
    must not pull the asker's profile facts in behind it."""
    toks = _tokenize(text or "")
    hits = _SELF_REF & set(toks)
    if not hits:
        return False
    # Every first-person token is a possessive introducing someone else -> not about
    # the asker. A bare "I"/"me"/"myself" anywhere still counts.
    for i, t in enumerate(toks):
        if t not in _SELF_REF:
            continue
        if t in _POSSESSIVE and i + 1 < len(toks) and toks[i + 1] in _OTHER_PERSON:
            continue                      # "my friend ..." - about them
        return True
    return False

# ---- forgetting ----------------------------------------------------------- #
PRUNE_FLOOR = 0.02         # decayed(importance*recency) below this is forgettable
_FORGOTTEN_MAX = 1000      # cap on the recoverable-forgotten archive sidecar
_CORRECTIONS_MAX = 200     # cap on pending, un-reviewed supersede proposals
_DAY = 86400.0

# Trusted (non-synth) sources: a distilled synth candidate may never silently
# rewrite these (see corrections.py / consolidate.py). Grouped identically by
# recall (recency pin) and prune (decay exemption).
TRUSTED_SOURCES = ("user", "import")
# How many TRUSTED_SOURCES facts may be promoted past a LEXICAL MISS when the
# semantic signal is degraded (no_embedder / no_vectors / low_coverage /
# dim_mismatch).
TRUST_FALLBACK_K = 2

_AGENTS = ("chat", "coder")


def _memory_root(root: Optional[Path] = None) -> Path:
    if root is not None:
        return Path(root)
    from localm.config import home_dir
    return home_dir() / "memory"


def namespace_hash(principal: str, agent: str, scope_key: str) -> str:
    """Stable 16-hex namespace id. UTF-8 encoded before hashing so a crafted
    unicode principal cannot alias another; agent + scope_key are joined with a
    delimiter that record ids never contain."""
    raw = "|".join((principal or "owner", agent, scope_key or "")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def namespace_file(principal: str, agent: str, scope_key: str,
                   root: Optional[Path] = None) -> Path:
    if agent not in _AGENTS:
        raise ValueError(f"unknown memory agent {agent!r}; expected one of {_AGENTS}")
    base = _memory_root(root).resolve()
    path = (base / agent / f"{namespace_hash(principal, agent, scope_key)}.jsonl")
    # Path-safety: the resolved file must stay under <home>/memory. agent is
    # allow-listed and the ns is a hex hash.
    #
    # Two DIFFERENT namespaces of the same agent (distinct scope_key -> distinct
    # ns_hash -> distinct per-namespace lock, so they do NOT serialise against
    # each other) can race to be the first-ever write under a shared agent
    # directory (e.g. <home>/memory/chat/) that does not exist yet: one thread's
    # _save() may create that directory mid-flight while another thread is inside
    # path.resolve() for a sibling file under it. On Windows this can make
    # path.resolve() return a \\?\-prefixed extended-length form (ntpath.realpath's
    # prefix-stripping re-validation, done via a second internal resolve, can
    # itself be caught by the same race and leave the prefix in place) while
    # base.resolve() - resolved moments earlier against a stable,
    # already-existing directory - has no prefix. Both denote the identical
    # location, so compare prefix-stripped forms; a real escape attempt is
    # unaffected since the prefix carries no path-safety information (it only
    # marks how the OS chose to resolve).
    rp = _strip_extended_prefix(path.resolve())
    base = _strip_extended_prefix(base)
    if not (rp == base or _within(rp, base)):
        raise ValueError("refusing a memory path outside the memory directory")
    return path


_EXTENDED_PREFIX = "\\\\?\\"
_EXTENDED_UNC_PREFIX = "\\\\?\\UNC\\"


def _strip_extended_prefix(path: Path) -> Path:
    """Strip Windows' \\?\\ (or \\?\\UNC\\) extended-length-path prefix, if
    present, so two resolutions of the identical location compare equal
    regardless of which one the OS chose. A no-op on POSIX and on any path that
    never had the prefix."""
    s = str(path)
    if s.startswith(_EXTENDED_UNC_PREFIX):
        return Path("\\\\" + s[len(_EXTENDED_UNC_PREFIX):])
    if s.startswith(_EXTENDED_PREFIX):
        return Path(s[len(_EXTENDED_PREFIX):])
    return path


def _within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _maxnorm(scores: list[float]) -> list[float]:
    top = max(scores) if scores else 0.0
    return [s / top for s in scores] if top > 0 else [0.0 for _ in scores]


# Per-namespace-hash locks. A fresh MemoryStore is constructed per call site -
# every HTTP request/plugin call builds its own instance (see plug.py's
# _chat_store) - so a per-INSTANCE lock cannot serialise them: two instances each
# _load() the same on-disk state, mutate their own in-memory copy, and _save(),
# and the losing write is silently lost. Keyed by namespace_hash, so the map is
# bounded by the number of active namespaces, not per-call. RLock so prune()
# calling replace(), or a caller batching several save=False mutations under
# store.lock(), does not deadlock.
_NAMESPACE_LOCKS = NamespaceLockRegistry()


def _namespace_lock(ns_hash: str):
    return _NAMESPACE_LOCKS.get(ns_hash)


# The registry above serialises writers INSIDE one process only. It cannot
# serialise `localm memory add|forget|restore|accept|clear` - its own OS process,
# its own registry - against a running server's consolidation pass: both _load()
# the same state, mutate their copy and _save(), and the CLI reports success for a
# write that is then gone. `localm setup-embeddings` writes every memory namespace
# from its own process too (cli/maintenance.py).
#
# The cross-process lock is rag's heartbeat-based collection_write_lock, NOT
# config._cross_process_lock: that one reclaims any holder older than 30s as
# abandoned, and store.add()/replace() resolve the embedder INSIDE the lock, which
# can trigger a VRAM swap lasting minutes - a wall-clock reclaim would reap a live
# holder and let both write. collection_lock has no wall-clock limit and keys
# staleness on a heartbeat instead.

# How long recall's reinforcement waits for a busy namespace before skipping the
# bump. It runs on the chat turn.
_REINFORCE_LOCK_WAIT = 2.0

# Gates the FILE lock so only ONE thread per process ever contends for it: a
# sibling thread waits on this mutex instead of burning its whole budget on
# collection_write_lock's O_EXCL create against a holder in its own process.
_FILE_GATES = NamespaceLockRegistry()

_XPROC_DEPTH = threading.local()


def _namespace_lockfile(store_file: Path) -> Path:
    """The cross-process lock file for a namespace: a sibling of the store file.

    ``<ns>.jsonl.lock``, so it can never be mistaken for a namespace by
    backfill._namespaces (which globs ``*/*.jsonl``) nor for any sidecar."""
    return store_file.with_name(store_file.name + ".lock")


@contextlib.contextmanager
def _namespace_write_lock(ns_hash: str, store_file: Path, op: str,
                          timeout: Optional[float] = None):
    """The in-process lock AND the cross-process one, for a WRITE.

    Reentrant on both halves. The in-process half is an RLock already; the
    cross-process half is NOT (collection_write_lock turns a nested acquisition
    into an error), and nesting here is normal - prune() calls replace(), and
    store.lock() is public so a caller can batch several save=False mutations. So
    the file lock is taken by the OUTERMOST acquisition only, tracked per thread
    and per namespace.

    READS do NOT take the FILE lock. _save() writes through storekit.atomic_write
    (tmp + os.replace), so a concurrent reader sees the old file or the new one,
    never a mix. Reads DO take the namespace RLock (MemoryStore.__init__ does, to
    _load()), so the ordering below matters: hold that RLock across the file-lock
    wait and every read in this process waits with you.

    Never returns without the lock: a refusal raises CollectionLockedError rather
    than proceeding unprotected. *timeout* bounds the wait for callers that must
    not block (recall's reinforcement); the default budget is collection_lock's."""
    from localm.rag.collection_lock import collection_write_lock
    depth = getattr(_XPROC_DEPTH, "depth", None)
    if depth is None:
        depth = _XPROC_DEPTH.depth = {}
    if depth.get(ns_hash):
        # Nested (prune -> replace, or a store.lock() batch). This thread already
        # owns the gate and the file lock; only the RLock re-enters.
        depth[ns_hash] += 1
        try:
            with _namespace_lock(ns_hash):
                yield
        finally:
            depth[ns_hash] -= 1
        return
    # ORDER IS LOAD-BEARING: gate, then FILE lock, then the namespace RLock - and
    # the RLock is taken only AFTER the file lock is held. MemoryStore.__init__
    # takes that same RLock to _load(), so waiting for the file lock inside it
    # blocks every READ in this process for the writer's whole budget. Waiting on
    # the file lock OUTSIDE the RLock keeps a reader's block down to the short
    # load/mutate/save section.
    #
    # No deadlock: every path takes them in this one order, and the gate means a
    # sibling thread waits on a cheap mutex rather than racing for the file.
    #
    # THE GATE IS BOUNDED BY THE SAME BUDGET as the file lock: an unbounded gate
    # moves the starvation onto itself, because a writer blocked for its full
    # budget holds the gate throughout and a caller that passes a short timeout
    # (corrections(), reinforcement) queues behind it for that whole budget
    # instead of its own.
    from localm.rag.collection_lock import wait_budget
    budget = wait_budget() if timeout is None else timeout
    gate = _FILE_GATES.get(ns_hash)
    started = time.monotonic()
    if not gate.acquire(timeout=budget):
        raise CollectionLockedError(ns_hash, None, budget, same_process=True,
                                    kind="Memory namespace")
    try:
        # Whatever the gate cost comes out of the same budget, so a bounded caller
        # stays bounded overall rather than paying budget twice.
        left = max(0.5, budget - (time.monotonic() - started))
        with collection_write_lock(_namespace_lockfile(store_file),
                                   collection=ns_hash, op=op, timeout=left,
                                   kind="Memory namespace"):
            depth[ns_hash] = 1
            try:
                with _namespace_lock(ns_hash):
                    yield
            finally:
                depth[ns_hash] = 0
    finally:
        gate.release()


class MemoryStore:
    """JSONL-backed store for one ``(principal, agent, scope_key)`` namespace."""

    def __init__(self, principal: str, agent: str, scope_key: str = "", *,
                 root: Optional[Path] = None) -> None:
        self.principal = principal or "owner"
        self.agent = agent
        self.scope_key = scope_key or ""
        # The key every mutating method locks on (see _namespace_lock). Computed
        # from the (principal, agent, scope_key) strings alone (no I/O), so it is
        # safe to compute before acquiring the lock below.
        self._ns_hash = namespace_hash(self.principal, agent, self.scope_key)
        self._records: list[MemoryRecord] = []
        self._vectors: dict = {}        # id -> vector (present only when embedded)
        self._dim: Optional[int] = None
        self._bm25: Optional[BM25] = None
        # User/import records evicted by the most recent prune (size cap), so a
        # caller can surface an otherwise-silent user-fact loss. See prune().
        self.last_evicted_user: list[MemoryRecord] = []
        # namespace_file() and _load() must BOTH hold the lock, not just _load().
        # Every real call site constructs a fresh instance per call (see plug.py's
        # _chat_store), so an unlocked read here can overlap ANOTHER thread's
        # locked _atomic_write -> tmp.replace(path). This is not only _load()'s
        # read_text(): namespace_file()'s own Path.resolve() calls can open a
        # handle to an EXISTING target file to resolve it, which can likewise
        # collide with an in-flight tmp.replace() and raise PermissionError on
        # Windows.
        with _namespace_lock(self._ns_hash):
            self._file = namespace_file(self.principal, agent, self.scope_key, root=root)
            self._load()

    # ----------------------------------------------------------------- IO -- #
    @classmethod
    def open_file(cls, path: Path) -> "MemoryStore":
        """Open an EXISTING namespace file directly, without knowing the
        (principal, agent, scope_key) that produced it.

        A namespace file is named for its hash, and that hash is exactly the key
        every mutating method locks on - so a store opened this way locks
        identically to one opened the normal way, and cannot race a concurrent
        writer of the same namespace. Needed by the vector backfill, which walks
        the memory root and must reach EVERY namespace including key-scoped ones,
        whose principal is a bearer-key hash that cannot be reconstructed from
        disk.

        Not a general constructor: it does no path-safety derivation, because it
        takes an already-resolved file the caller enumerated from the memory root
        itself.
        """
        obj = cls.__new__(cls)
        obj.principal = ""
        obj.agent = path.parent.name
        obj.scope_key = ""
        obj._ns_hash = path.stem
        obj._records = []
        obj._vectors = {}
        obj._dim = None
        obj._bm25 = None
        obj.last_evicted_user = []
        obj._file = path
        with _namespace_lock(obj._ns_hash):
            obj._load()
        return obj

    def vectorless_count(self) -> int:
        """How many records still have no vector. The honest denominator for a
        backfill that reports what it did NOT finish."""
        return sum(1 for r in self._records if r.id not in self._vectors)

    @property
    def path(self) -> Path:
        return self._file

    def lock(self):
        """This namespace's RLock. Every mutating method below acquires it
        internally for its own single call; exposed so a caller that needs to
        batch several ``save=False`` mutations under ONE reload + save (e.g.
        plug.py's ``_migrate_legacy``) can hold it across the whole batch.

        This is the WRITE lock: every documented use of it is a batch of
        MUTATIONS, so it takes the cross-process lock too. Returns a FRESH context
        manager per call - do not stash one and reuse it."""
        return self._wlock("a batch")

    def _wlock(self, op: str, timeout: Optional[float] = None):
        """This namespace's write lock: in-process AND cross-process, reentrant."""
        return _namespace_write_lock(self._ns_hash, self._file, op,
                                     timeout=timeout)

    def _vec_file(self) -> Path:
        return self._file.with_suffix(".vec.json")

    def _load(self) -> None:
        self._records = []
        skipped = 0
        if self._file.is_file():
            # split_jsonl, not splitlines(): splitlines() also breaks on
            # U+0085/U+2028/U+2029, which json.dumps(ensure_ascii=False) writes
            # RAW, so a memory whose text contains one is torn in half and
            # silently dropped on load.
            for line in split_jsonl(self._file.read_text(encoding="utf-8")):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        raise ValueError("memory record line is not a JSON object")
                    self._records.append(MemoryRecord.from_dict(data))
                except (json.JSONDecodeError, TypeError, ValueError):
                    # A partial/corrupt line must not break recall, but it may
                    # not vanish silently either: _save() rewrites the whole
                    # file, so every skipped line is erased by the next write
                    # (warned about below).
                    skipped += 1
        if skipped:
            from localm.debuglog import logger as _dbg
            _dbg.warning(
                "memory store %s: skipped %d unparseable line(s) on load; "
                "the next save will drop them permanently", self._file, skipped)
        self._vectors = {}
        self._dim = None
        vf = self._vec_file()
        if vf.is_file():
            try:
                data = json.loads(vf.read_text(encoding="utf-8"))
                # A non-object top level (a bare list/number) has no .get; treat
                # it as corrupt, like the records path treats a non-object line
                # above.
                vecs = data.get("vectors", {}) if isinstance(data, dict) else None
                if not isinstance(vecs, dict):
                    raise ValueError("vector sidecar has an unexpected structure")
                ids = {r.id for r in self._records}
                self._vectors = {k: v for k, v in vecs.items() if k in ids and v}
                self._dim = data.get("dim") or _first_dim(self._vectors)
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                # The sidecar EXISTS but is corrupt/unreadable: degrade to no
                # vectors (recall falls back to lexical BM25) and warn - unlike an
                # absent sidecar (a normal cold start, handled by the is_file()
                # gate), a present-yet-unparseable one is an unexpected fault the
                # operator should see. The next _save() rewrites a clean file.
                self._vectors = {}
                self._dim = None
                from localm.debuglog import logger as _dbg
                _dbg.warning(
                    "memory store %s: vector sidecar corrupt/unreadable (%s); "
                    "recall degrades to lexical until the next save rewrites it",
                    vf, exc)
        self._bm25 = None

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        # Each line carries the format version so a future breaking schema
        # change can migrate instead of silently skipping (see FORMAT_VERSION).
        body = "\n".join(json.dumps({"v": FORMAT_VERSION, **r.to_dict()},
                                    ensure_ascii=False)
                         for r in self._records)
        self._atomic_write(self._file, body + ("\n" if body else ""))
        vf = self._vec_file()
        if self._vectors:
            self._atomic_write(
                vf, json.dumps({"dim": self._dim, "vectors": self._vectors}))
        else:
            vf.unlink(missing_ok=True)
        self._bm25 = None

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        # storekit.atomic_write: a unique per-writer temp name, so tmp+replace
        # stays crash-safe even for a caller that writes a sidecar file outside
        # the namespace lock, PLUS a Windows PermissionError retry (an AV
        # real-time scan / Search Indexer can transiently hold a handle to the
        # target, which would otherwise fail a good write).
        _storekit_atomic_write(path, content)

    # ------------------------------------------------------------- basics -- #
    def all(self) -> list[MemoryRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def get(self, mem_id: str) -> Optional[MemoryRecord]:
        return next((r for r in self._records if r.id == mem_id), None)

    def _embed_one(self, text: str, embed_fn: Optional[EmbedFn]) -> Optional[list]:
        """Embed *text*, honouring the single-dimensionality invariant. A vector
        of a different dim than the store's (a switched embedding model) is
        dropped, not stored, so cosine never mixes dims (best-effort, never
        raises - memory writes must not crash on an embedder hiccup, but a
        real failure is still surfaced at debug level, not swallowed silently).

        The failure log is CONTENT-GATED: *text* is a memory record (chat-derived),
        so the snippet is only written when debug_content_enabled() allows it. The
        failure itself is always logged, in every mode."""
        if embed_fn is None:
            return None
        try:
            vec = embed_fn([text])[0]
        except Exception as e:
            from localm.debuglog import debug_content_enabled, logger as _dbg
            # The FAILURE is always reported. *text* is a memory RECORD, i.e.
            # chat-derived CONTENT, so the snippet is gated on
            # debug_content_enabled(); that returns False in privacy mode even
            # when the debug log is ON, so privacy mode never persists memory
            # content to the log file. Only the length is kept when the gate is
            # closed, so an over-long record is still diagnosable.
            if debug_content_enabled():
                _dbg.debug("memory embed_one failed for %r: %s", text[:80], e)
            else:
                _dbg.debug("memory embed_one failed (content withheld: privacy "
                           "mode, %d chars): %s", len(text), e)
            return None
        if not vec:
            return None
        if self._dim is None:
            self._dim = len(vec)
        elif len(vec) != self._dim:
            return None
        return list(vec)

    def add(self, record: MemoryRecord, *, embed_fn: Optional[EmbedFn] = None,
            save: bool = True) -> MemoryRecord:
        # Re-sync with the latest committed state under the lock before mutating,
        # so a concurrent add() elsewhere that finished first is not
        # read-stale-then-overwritten. save=False (a caller batching several
        # mutations, e.g. plug.py's _migrate_legacy) skips the reload - the caller
        # already loaded fresh once via store.lock() and must not have that
        # in-progress batch wiped.
        with self._wlock('an add'):
            if save:
                self._load()
            record.text = record.text[:MAX_TEXT_LEN]
            self._records.append(record)
            vec = self._embed_one(record.text, embed_fn)
            if vec is not None:
                self._vectors[record.id] = vec
            if save:
                self._save()
            return record

    def update(self, mem_id: str, *, embed_fn: Optional[EmbedFn] = None,
               save: bool = True, **fields) -> Optional[MemoryRecord]:
        with self._wlock('an update'):
            if save:
                self._load()
            rec = self.get(mem_id)
            if rec is None:
                return None
            text_changed = False
            for key, val in fields.items():
                if key == "text" and isinstance(val, str):
                    val = val.strip()[:MAX_TEXT_LEN]
                    text_changed = val != rec.text
                    rec.text = val
                elif key == "importance":
                    from .record import _clamp01
                    rec.importance = _clamp01(val)
                elif key in ("last_used", "created", "uses", "kind", "source", "meta"):
                    setattr(rec, key, val)
            rec.updated = time.time()
            if text_changed and embed_fn is not None:
                vec = self._embed_one(rec.text, embed_fn)
                if vec is not None:
                    self._vectors[rec.id] = vec
                else:
                    self._vectors.pop(rec.id, None)
            if save:
                self._save()
            return rec

    def delete(self, mem_id: str, *, save: bool = True) -> bool:
        with self._wlock('a delete'):
            if save:
                self._load()
            before = len(self._records)
            self._records = [r for r in self._records if r.id != mem_id]
            self._vectors.pop(mem_id, None)
            removed = len(self._records) != before
            if removed and save:
                self._save()
            return removed

    def clear(self, *, include_forgotten: bool = False) -> None:
        """Erase this namespace's live records.

        The forgotten sidecar is a SEPARATE file and a plain clear() does not
        touch it, so after one, every record that prune eviction or an accepted
        correction ever archived is still readable on disk.

        ``include_forgotten`` takes the archive too, which is what any user-facing
        "erase what you remember about me" must pass: leaving the text in a sidecar
        while reporting the memory cleared is a privacy claim that is not true. It
        also erases the pending-corrections sidecar (target_text/proposed_text) and
        the corrections-dismissed sidecar (a rejected proposal's text, casefolded,
        inside its dedup key) - both hold the same verbatim user text as the
        forgotten archive.

        The default is False; the CLI passes True.
        """
        with self._wlock('a clear'):
            self._records = []
            self._vectors = {}
            self._dim = None
            self._save()
            if include_forgotten:
                self._forgotten_file().unlink(missing_ok=True)
                self._corrections_file().unlink(missing_ok=True)
                self._dismissed_file().unlink(missing_ok=True)

    def remnants(self) -> list[str]:
        """Names of the parts of this namespace that still hold something on
        disk: any of "live records", "forgotten archive", "pending
        corrections", "dismissed corrections". Empty means genuinely nothing
        remains.

        Checked by FILE PRESENCE for the three sidecars rather than through
        corrections()'s live-target pruning, so a pending correction whose
        target record was separately hard-deleted (forget, not archived)
        still counts as something left to erase - `corrections()` would
        report that same entry as stale and skip it.
        """
        out = []
        if self._records:
            out.append("live records")
        if self._forgotten_file().is_file():
            out.append("forgotten archive")
        if self._corrections_file().is_file():
            out.append("pending corrections")
        if self._dismissed_file().is_file():
            out.append("dismissed corrections")
        return out

    def invalidate_vectors(self, ids) -> None:
        """Drop the cached vectors of *ids* so the next save/replace re-embeds
        them. Needed when record TEXT is mutated outside :meth:`update` (the
        consolidation batch): ``replace`` only embeds ids WITHOUT a vector, so
        a text change would otherwise keep serving the old text's vector
        forever. No save here; the caller's replace/save persists the result."""
        for mem_id in ids:
            self._vectors.pop(mem_id, None)

    def semantic_nearest(self, text: str, records: list,
                         embed_fn: Optional[EmbedFn]) -> tuple:
        """(index into *records*, cosine) of the record most semantically similar
        to *text*, or (-1, 0.0) when no embedder, no stored vectors, or a dim
        mismatch. Used by consolidation to catch PARAPHRASED contradictions that
        share few tokens ('lives in Berlin' vs 'moved to Munich'), which the
        lexical matcher misses, so they reach the ADD/UPDATE/DELETE decision
        instead of blind-accumulating. Compares against THIS store's cached
        vectors keyed by record id."""
        if embed_fn is None or not self._vectors:
            return -1, 0.0
        try:
            qvec = embed_fn([text])[0]
        except Exception as e:
            from localm.debuglog import logger as _dbg
            _dbg.debug("memory semantic_nearest query embed failed: %s", e)
            return -1, 0.0
        if not qvec:
            return -1, 0.0
        qdim = len(qvec)
        best_i, best_s = -1, 0.0
        for i, r in enumerate(records):
            vec = self._vectors.get(r.id)
            if not vec or len(vec) != qdim:
                continue
            s = _cosine(qvec, vec)
            if s > best_s:
                best_i, best_s = i, s
        return best_i, best_s

    def backfill_vectors(self, embed_fn: EmbedFn, *, limit: int = 64) -> int:
        """Embed records that have no vector yet, up to *limit* per call, and
        save. Returns the number embedded.

        This is how semantic recall turns on RETROACTIVELY: a user who chats
        before installing an embedding model has memories with no vectors, and
        nothing else re-embeds them.

        BOUNDED per call, so a large store never stalls one caller. That means a
        single call does NOT get coverage up on its own - drive it to completion
        with ``memory.backfill.backfill_all``, which is what setup-embeddings
        uses. Best-effort: an embed failure for one record is skipped, not fatal.

        Locked and re-loaded like the other mutating methods, so a backfill pass
        started from a stale snapshot cannot silently clobber a concurrent
        add/update/delete's save."""
        if embed_fn is None:
            return 0
        with self._wlock('a vector backfill'):
            self._load()
            done = 0
            for r in self._records:
                if done >= limit:
                    break
                if r.id in self._vectors:
                    continue
                vec = self._embed_one(r.text, embed_fn)
                if vec is not None:
                    self._vectors[r.id] = vec
                    done += 1
            if done:
                self._save()
            return done

    def replace(self, records: list[MemoryRecord], *,
                embed_fn: Optional[EmbedFn] = None,
                invalidate_ids=None) -> None:
        """Overwrite the whole namespace in ONE atomic save (used by the
        consolidation batch and prune, so a crash leaves the pre-change store
        intact - never a half-consolidated state).

        Locked AND re-loaded like every other mutating method, so a
        standalone caller (e.g. the PUT /api/memory bulk-edit route) cannot
        silently clobber a concurrent add/update/delete's already-persisted
        vector. *records* (the caller's list) always overwrites ``self._records``
        regardless - that is the point of a full replace - but reloading first
        means the ``keep_ids`` filter below preserves a FRESH on-disk vector for
        any surviving id instead of a stale in-memory one.

        *invalidate_ids*: ids whose cached vector must be dropped so the
        embed-if-missing loop below re-embeds them with new text (consolidation's
        UPDATE decisions). This must be applied AFTER the reload above, or a
        reload would silently restore the stale vector straight from disk and
        undo the caller's invalidation - so it is a parameter here, not a
        separate ``invalidate_vectors()`` call the caller makes beforehand."""
        with self._wlock('a replace'):
            self._load()
            if invalidate_ids:
                self.invalidate_vectors(invalidate_ids)
            keep_ids = {r.id for r in records}
            self._vectors = {k: v for k, v in self._vectors.items() if k in keep_ids}
            self._records = []
            for r in records:
                r.text = (r.text or "").strip()[:MAX_TEXT_LEN]
                self._records.append(r)
                if embed_fn is not None and r.id not in self._vectors:
                    vec = self._embed_one(r.text, embed_fn)
                    if vec is not None:
                        self._vectors[r.id] = vec
            self._save()

    # --------------------------------------------------------- retrieval -- #
    def _vector_status(self, embed_fn: Optional[EmbedFn]) -> tuple[bool, Optional[str]]:
        """Whether the semantic (cosine) signal is usable for recall right now,
        and, when it is not, a short reason (surfaced to callers as the recall
        DEGRADE reason, mirroring RAG's lexical-only fallback). Single source of
        truth for ``_vector_relevance`` and ``recall``'s diagnostics, so the
        surfaced reason can never drift from the branch actually taken.

        Reasons: ``no_embedder`` (no embedding model resolved), ``no_vectors``
        (records not embedded yet, e.g. before ``setup-embeddings``),
        ``low_coverage`` (< VEC_COVERAGE of records carry a vector), ``dim_mismatch``
        (mixed vector dimensions in the sidecar)."""
        if embed_fn is None:
            return False, "no_embedder"
        if not self._vectors:
            return False, "no_vectors"
        n = len(self._records)
        if n == 0 or len(self._vectors) / n < VEC_COVERAGE:
            return False, "low_coverage"
        if len({len(v) for v in self._vectors.values()}) != 1:
            return False, "dim_mismatch"
        return True, None

    def _relevance(self, query: str, embed_fn: Optional[EmbedFn],
                   diagnostics: Optional[dict] = None) -> list[float]:
        n = len(self._records)
        vec_rel = self._vector_relevance(query, embed_fn, diagnostics=diagnostics)
        if n < TINY_CORPUS:
            # BM25 idf is unstable on a handful of records, so the LEXICAL signal
            # is skipped here - but the SEMANTIC (cosine) signal is not noisy on a
            # tiny corpus, so use it when an embedder is present. No vectors ->
            # no relevance signal, rank by recency+importance (transparent to
            # callers).
            return vec_rel if vec_rel is not None else [0.0] * n
        if self._bm25 is None:
            self._bm25 = BM25([r.text for r in self._records])
        rel = _maxnorm(self._bm25.scores(query))
        if vec_rel is not None:
            rel = [REL_LEX_SHARE * a + (1.0 - REL_LEX_SHARE) * b
                   for a, b in zip(rel, vec_rel)]
        return rel

    def _vector_relevance(self, query: str, embed_fn: Optional[EmbedFn],
                          diagnostics: Optional[dict] = None) -> Optional[list[float]]:
        usable, reason = self._vector_status(embed_fn)
        if not usable:
            if diagnostics is not None:
                diagnostics["degrade_reason"] = reason
            return None
        # _vector_status verified a single shared dimension across all vectors.
        stored_dim = len(next(iter(self._vectors.values())))
        try:
            qvec = embed_fn([query])[0]
        except Exception:
            if diagnostics is not None:
                diagnostics["degrade_reason"] = "query_embed_failed"
            return None
        if not qvec or len(qvec) != stored_dim:
            if diagnostics is not None:
                diagnostics["degrade_reason"] = "query_embed_failed"
            return None
        out = [
            _cosine(qvec, self._vectors[r.id]) if r.id in self._vectors else 0.0
            for r in self._records
        ]
        if diagnostics is not None:
            diagnostics["degrade_reason"] = None       # semantic signal active
        return _maxnorm(out)

    def _eligible(self, query: str, embed_fn: Optional[EmbedFn]) -> list[bool]:
        """Per-record ABSOLUTE relevance eligibility for the recall precision
        gate: a record is eligible for injection only when the query shares a
        CONTENT word with it (lexical) OR its raw cosine to the query clears
        REL_COS_MIN (semantic, when vectors are usable). A record failing BOTH is
        dropped, so recall stays SILENT when nothing is relevant. The query is
        embedded once here - a single short-string embed against the cached embedder
        singleton, negligible next to the turn's own inference."""
        q_tokens = _content_tokens(query)
        usable, _reason = self._vector_status(embed_fn)
        cos = None
        if usable:
            try:
                qv = embed_fn([query])[0]
            except Exception:
                # The SAME embed failure on the SAME embedder is surfaced as
                # diagnostics["degrade_reason"]="query_embed_failed" by
                # _vector_relevance, which recall() runs (via _relevance) on this
                # same call before _eligible. Here it just drops the semantic
                # eligibility signal; lexical eligibility still applies, so recall
                # degrades to lexical rather than erroring.
                qv = None
            stored_dim = len(next(iter(self._vectors.values())))
            if qv and len(qv) == stored_dim:
                cos = [(_cosine(qv, self._vectors[r.id]) if r.id in self._vectors
                        else 0.0) for r in self._records]
        lex_hits = [bool(q_tokens & _content_tokens(r.text)) for r in self._records]
        sem_hits = [cos is not None and cos[i] >= REL_COS_MIN
                    for i in range(len(self._records))]
        # A LEXICAL HIT RAISES THE BAR FOR EVERYTHING ELSE. When the query shares a
        # content word with some record, that word says what the query is about, so
        # a cosine-only record riding in behind it must be strongly similar, not
        # merely over the floor. A cosine-only record is still eligible on a
        # strong paraphrase, so a semantic-only record can out-rank a
        # lexical-only one.
        REL_COS_COMPANION = 0.75
        if any(lex_hits) and cos is not None:
            return [lex_hits[i] or cos[i] >= REL_COS_COMPANION
                    for i in range(len(self._records))]
        return [lex_hits[i] or sem_hits[i] for i in range(len(self._records))]

    def recall(self, query: str, *, k: int = 6, embed_fn: Optional[EmbedFn] = None,
               reinforce: bool = False, now: Optional[float] = None,
               diagnostics: Optional[dict] = None) -> list[MemoryRecord]:
        """Top-*k* records for *query* by relevance+recency+importance.

        ``reinforce=True`` bumps last_used/uses on the returned records (a WRITE):
        the caller passes ``reinforce=gating.writes_allowed(surface)`` so privacy
        mode recalls WITHOUT any side effect. Deterministic (stable tie-break).

        ``diagnostics`` (optional): when a dict is passed it is filled with the
        recall's observability (``degrade_reason`` - why the semantic/cosine signal
        was not used, or None when it was; ``n_records``/``n_vectors``/
        ``n_recalled``) so a caller can surface "used N memories" + the degrade
        reason. Default None keeps the call side-effect-free."""
        if not (query or "").strip() or not self._records:
            if diagnostics is not None:
                diagnostics.update({"degrade_reason": None,
                                    "n_records": len(self._records),
                                    "n_vectors": len(self._vectors),
                                    "n_recalled": 0})
            return []
        k = max(1, min(int(k), K_CAP))
        now = time.time() if now is None else now
        rel = self._relevance(query, embed_fn, diagnostics=diagnostics)
        # Recency is RAW exponential decay (already in [0,1]), NOT max-normalised:
        # max-normalising would make the newest record always score 1.0, so recall
        # could never fall silent on a store of only stale/irrelevant memories.
        # Raw decay (Generative-Agents style) lets an old, unimportant, off-topic
        # memory score below the floor and be dropped.
        # Stable, user-authored facts (source user/import) do NOT decay in relevance:
        # a durable preference stated long ago must not be buried under recent
        # chatter. Recency decay still applies to synth memories. Mirrors prune(),
        # which already exempts user/import from decay-based eviction.
        rec = [
            1.0 if r.source in ("user", "import")
            else math.exp(-((now - r.last_used) / _DAY) / TAU_DAYS)
            for r in self._records
        ]
        scored = []
        for i, r in enumerate(self._records):
            score = W_REL * rel[i] + W_REC * rec[i] + W_IMP * r.importance
            scored.append((score, i, r))
        # Sort by score desc; ties keep insertion order (index asc) for determinism.
        scored.sort(key=lambda t: (-t[0], t[1]))
        # Absolute relevance gate: only inject records that actually relate to the
        # query (lexical content-word hit or cosine over REL_COS_MIN); an off-topic
        # turn injects nothing rather than the top-k.
        eligible = self._eligible(query, embed_fn)
        # With the semantic signal degraded the gate above is LEXICAL-ONLY, and
        # exact content-word overlap misses every paraphrase ("what is my name" vs
        # "User is called Sam" share nothing). Rather than falling back to the
        # ungated top-k, promote at most TRUST_FALLBACK_K TRUSTED_SOURCES facts, in
        # score order, and only when the semantic signal is NOT available (the
        # healthy path keeps the strict gate untouched).
        #
        # This is a BOUNDED TRUSTED-FACT FALLBACK, not paraphrase recall. Degraded
        # means there is no relevance signal at all - rel is 0 for a zero-overlap
        # record and rec is pinned to 1.0 for TRUSTED_SOURCES - so this promotion
        # is effectively IMPORTANCE-ordered and can surface 2 unrelated user facts
        # while still missing the one asked about. It also applies with an embedder
        # installed: low_coverage alone degrades any store whose backfill has not
        # caught up yet.
        hits = [(s, r) for s, i, r in scored if s > FLOOR and eligible[i]][:k]
        usable, _reason = self._vector_status(embed_fn)
        promoted = 0
        # ONLY when the recall would otherwise be SILENT: an on-topic query that
        # already found a lexical hit must not have unrelated trusted facts dragged
        # in behind it. And only for a SELF-REFERENTIAL query - without that the
        # fallback cannot tell "what is my name" from "recommend a pasta recipe"
        # (identical zero-overlap inputs) and would re-inject unrelated profile
        # facts on every off-topic turn.
        if not usable and not hits and _is_self_referential(query):
            for _s, i, r in scored:                    # score order: best trusted first
                if promoted >= TRUST_FALLBACK_K:
                    break
                if eligible[i] or _s <= FLOOR:
                    continue                            # already in, or below the floor
                if r.source in TRUSTED_SOURCES:
                    eligible[i] = True
                    promoted += 1
            if promoted:
                hits = [(s, r) for s, i, r in scored if s > FLOOR and eligible[i]][:k]
        results = [r for _s, r in hits]
        if reinforce and results:
            # recall() is called with reinforce=True on every chat turn
            # (_memory_inlet), making this the highest-frequency write path - a
            # plain self._save() here would persist THIS instance's whole
            # self._records, so an unlocked save could silently revert a concurrent
            # add/update/delete/replace (e.g. resurrect a record another thread
            # just deleted), not merely lose a last_used/uses bump. Lock + reload
            # fresh state, then re-apply reinforcement by id against the RELOADED
            # records (not `results`, whose objects would otherwise be orphaned by
            # the reload) before saving. The cross-process lock is taken too: an
            # unlocked save here can resurrect a record `localm memory forget` just
            # deleted in ANOTHER process.
            #
            # BOUNDED, and skipped rather than waited out, because this runs on the
            # chat turn. Reinforcement is a usage-counter bump - losing one costs a
            # little recall ordering and nothing durable - so when the namespace is
            # busy the WRITE is skipped and the records are still returned. The
            # skip is logged and reported in diagnostics, never silent.
            try:
                with self._wlock("reinforcement", timeout=_REINFORCE_LOCK_WAIT):
                    self._load()
                    by_id = {r.id: r for r in self._records}
                    reinforced = []
                    for r in results:
                        fresh = by_id.get(r.id)
                        if fresh is not None:
                            fresh.last_used = now
                            fresh.uses += 1
                            reinforced.append(fresh)
                        else:
                            # Concurrently deleted elsewhere: nothing to persist for
                            # it, but still return it (best-effort) so the caller's
                            # result count/content is not silently changed mid-call.
                            reinforced.append(r)
                    self._save()
                    results = reinforced
            except CollectionLockedError as e:
                from localm.debuglog import logger
                logger.debug(
                    "memory recall: skipped reinforcing %d record(s), the "
                    "namespace is being written elsewhere (%s)", len(results), e)
                if diagnostics is not None:
                    diagnostics["reinforce_skipped"] = True
        if diagnostics is not None:
            # degrade_reason was set by _vector_relevance during _relevance above.
            diagnostics.setdefault("degrade_reason", None)
            diagnostics["n_records"] = len(self._records)
            diagnostics["n_vectors"] = len(self._vectors)
            diagnostics["n_recalled"] = len(results)
            # How many facts only made it in through the trust fallback, i.e.
            # because the semantic signal was unavailable. The caller already
            # surfaces degrade_reason ("keyword-only matching") alongside it.
            diagnostics["trust_fallback"] = promoted
        return results

    # --------------------------------------------------------- forgetting - #
    def _decayed(self, rec: MemoryRecord, now: float) -> float:
        recency = math.exp(-((now - rec.last_used) / _DAY) / TAU_DAYS)
        # Reinforcement lifts effective importance a little (frequently-used
        # memories resist decay), capped at 1.0.
        eff_imp = min(1.0, rec.importance + 0.05 * rec.uses)
        return eff_imp * recency

    def _forgotten_file(self) -> Path:
        return self._file.with_suffix(".forgotten.jsonl")

    def _archive_forgotten(self, records: list[MemoryRecord]) -> bool:
        """Append evicted records to a ``.forgotten.jsonl`` sidecar so forgetting
        is RECOVERABLE, not a silent hard delete. Returns True when the archive is
        persisted (or there was nothing to archive), False when it failed. prune()
        treats archival as best-effort (it logs and proceeds), but an interactive
        accept of a supersession must NOT destroy the trusted record when this
        returns False. The archive is capped so it cannot grow unbounded."""
        if not records:
            return True
        try:
            ff = self._forgotten_file()
            prior = []
            if ff.is_file():
                prior = [ln for ln in split_jsonl(ff.read_text(encoding="utf-8"))
                         if ln.strip()]
            # "v": FORMAT_VERSION mirrors the main record store's stamp - same
            # forward-tolerant from_dict pattern, so a future schema change has a
            # migration hook here too, not just on the primary store.
            new = [json.dumps({"v": FORMAT_VERSION, **r.to_dict(),
                               "forgotten_at": time.time()}, ensure_ascii=False)
                   for r in records]
            lines = (prior + new)[-_FORGOTTEN_MAX:]
            self._atomic_write(ff, "\n".join(lines) + "\n")
            return True
        except OSError as e:
            from localm.debuglog import logger
            logger.warning("memory: could not archive %d forgotten record(s): %s",
                           len(records), e)
            return False

    def _load_forgotten(self) -> list[dict]:
        """Read the ``.forgotten.jsonl`` archive sidecar as raw dicts (record fields
        plus ``forgotten_at``, and an optional ``v`` stamp tolerated like every other
        sidecar - see FORMAT_VERSION). A corrupt/partial line is skipped and warned
        about, like ``_load``/``_load_corrections`` (including a line that is not valid
        UTF-8); an absent file is simply empty; a present-but-unreadable file warns and
        reports empty (see below)."""
        ff = self._forgotten_file()
        if not ff.is_file():
            return []
        out: list[dict] = []
        skipped = 0
        try:
            raw = ff.read_bytes()
        except OSError as e:
            # Exists but unreadable (transient lock). Unlike the corrections sidecar
            # this read is NON-destructive - forgotten()/restore_forgotten only READ
            # the archive - so keep the empty return, but WARN rather than silently
            # reporting "nothing to restore" when the archive merely could not be
            # read - an absent archive and an unreadable one are not the same thing.
            # read_bytes (not read_text) so invalid-UTF-8 CONTENT is a per-line skip
            # below, not an uncaught UnicodeDecodeError that would 500 the restore route.
            from localm.debuglog import logger as _dbg
            _dbg.warning(
                "memory forgotten archive %s: unreadable, reporting no recoverable "
                "records for now: %s", ff, e)
            return []
        for raw_line in raw.split(b"\n"):
            try:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError("forgotten line is not a JSON object")
                out.append(data)
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
                skipped += 1
        if skipped:
            from localm.debuglog import logger as _dbg
            _dbg.warning(
                "memory forgotten archive %s: skipped %d unparseable line(s) on load",
                ff, skipped)
        return out

    def _save_forgotten(self, entries: list[dict]) -> None:
        ff = self._forgotten_file()
        if not entries:
            ff.unlink(missing_ok=True)
            return
        body = "\n".join(json.dumps(e, ensure_ascii=False) for e in entries)
        self._atomic_write(ff, body + "\n")

    def forgotten(self) -> list[dict]:
        """Archived (forgotten) records for THIS namespace, newest-forgotten-first,
        each an on-disk snapshot (record fields + ``forgotten_at``). The read half
        of ``_archive_forgotten``'s recoverable-not-deleted contract, used by the
        recovery route to show what can be restored."""
        with _namespace_lock(self._ns_hash):
            return list(reversed(self._load_forgotten()))

    def restore_forgotten(self, mem_id: str, *,
                          embed_fn: Optional[EmbedFn] = None) -> Optional[MemoryRecord]:
        """Recover one archived snapshot for *mem_id* back into the live store.
        Two archive shapes exist, both handled here:

          EVICTED - no live record with this id (prune's size cap, or an accepted
          DELETE correction fully removed it): the snapshot is re-added as a live
          record.

          SUPERSEDED - a live record with this id already exists: an accepted
          UPDATE correction (see ``resolve_correction``) archives the PRE-CHANGE
          snapshot under the SAME id as the record it then mutates in place, so
          the id never actually frees up. This case REVERTS the live record's text
          to the archived snapshot in place (undoing whatever changed it),
          matching exactly what an accepted UPDATE correction could have altered.

        When a record has more than one archive entry (forgotten/superseded more
        than once), the MOST RECENT one is applied, so repeated restores step back
        through history one snapshot at a time - reverting a record that was
        corrected Berlin -> Munich -> Ghent first undoes to Munich, then Berlin.
        Returns None when no archive entry matches *mem_id* at all.

        Locked and reloaded like every other mutating method, so a concurrent
        add/delete/restore cannot race the read-decide-write sequence below. The
        applied entry is removed from the archive on success (the archive is a
        recovery queue, not an immutable audit log)."""
        with self._wlock('a restore'):
            self._load()
            entries = self._load_forgotten()
            matches = [e for e in entries if e.get("id") == mem_id]
            if not matches:
                return None
            entry = matches[-1]                  # most recent (archive is append-ordered)
            snapshot = MemoryRecord.from_dict(entry)
            live = self.get(mem_id)
            if live is None:
                self._records.append(snapshot)    # EVICTED: re-add as a live record
                record = live = snapshot
            else:
                # SUPERSEDED: revert in place. Only text (+ the timestamps a
                # correction-accept itself would touch) - mirrors exactly what
                # resolve_correction's UPDATE branch can change, nothing more.
                live.text = snapshot.text
                live.updated = time.time()
                live.last_used = live.updated
                record = live
            self._vectors.pop(record.id, None)
            vec = self._embed_one(record.text, embed_fn)
            if vec is not None:
                self._vectors[record.id] = vec
            self._save()
            remaining = [e for e in entries if e is not entry]
            try:
                self._save_forgotten(remaining)
            except OSError as e:
                # The restore/revert above already succeeded and is persisted; only
                # the archive-trim cleanup failed, so it must NOT surface as a
                # restore failure. Logged so the stale archive entry is
                # discoverable. A later restore of the same id at worst re-applies
                # this same (already-applied) snapshot, which is idempotent.
                from localm.debuglog import logger
                logger.warning(
                    "memory: restored %s but could not trim the forgotten archive: %s",
                    mem_id, e)
            return record

    def prune(self, *, now: Optional[float] = None, n_max: int = N_MAX) -> int:
        """Forget decayed, low-value memories and enforce the size cap. User- and
        import-sourced records are never auto-dropped by decay (only the size cap
        may evict them, weakest first); synth memories below ``PRUNE_FLOOR`` are
        forgotten. Evicted records are archived to a ``.forgotten.jsonl`` sidecar
        (recoverable, not a silent hard delete) and the user-sourced evictions are
        exposed on ``last_evicted_user`` so a caller can surface them. Returns the
        number removed."""
        # Locked and re-loaded like every other mutating method (the RLock lets the
        # nested self.replace() call below re-acquire without deadlocking), so
        # eviction is computed against the latest committed state,
        # not a stale in-memory snapshot that could re-evict an already-saved
        # record or miss one a concurrent writer just added.
        with self._wlock('a prune'):
            self._load()
            now = time.time() if now is None else now
            kept = [
                r for r in self._records
                if r.source in ("user", "import") or self._decayed(r, now) >= PRUNE_FLOOR
            ]
            if len(kept) > n_max:
                kept.sort(key=lambda r: self._decayed(r, now), reverse=True)
                kept = kept[:n_max]
            kept_ids = {r.id for r in kept}
            evicted = [r for r in self._records if r.id not in kept_ids]
            # Exposed for callers (consolidation surfaces user-fact evictions in
            # its result); reset every prune so it reflects THIS run only.
            self.last_evicted_user = [
                r for r in evicted if r.source in ("user", "import")]
            if evicted:
                self._archive_forgotten(evicted)
                self.replace(kept)
            return len(evicted)

    # ------------------------------------------------- pending corrections - #
    # A synth candidate may never silently rewrite a trusted (user/import) fact.
    # Consolidation records a PENDING CORRECTION here and the memory modal
    # surfaces it for the user to accept (apply, old text archived + recoverable)
    # or reject (keep + reset staleness).
    # Kept in a separate <ns>.corrections.jsonl sidecar so recall/prune/replace
    # never touch it.
    def _corrections_file(self) -> Path:
        return self._file.with_suffix(".corrections.jsonl")

    def _load_corrections(self) -> list[PendingCorrection]:
        """Read the pending-corrections sidecar. A corrupt/partial LINE is skipped
        (best-effort, like the record loader) - including a line that is not valid
        UTF-8, so a torn multibyte write corrupts only that line, not the whole file;
        an ABSENT file is simply empty; a present-but-UNREADABLE file (I/O error)
        RAISES - missing is not unreadable. Collapsing an unreadable file to []
        would let propose_corrections rewrite the sidecar with only the freshly
        proposed entries and permanently wipe every pending correction, while
        telling the caller it succeeded. The save-bearing callers here
        (propose_corrections / corrections / resolve_correction) catch the OSError,
        warn, and abort the save rather than crash."""
        cf = self._corrections_file()
        if not cf.is_file():
            return []
        out: list[PendingCorrection] = []
        skipped = 0
        # read_bytes so ONLY a real I/O failure (OSError) is "unreadable" and raises
        # for the callers to abort on; decode + parse each LINE inside the try, so a
        # bad-JSON OR invalid-UTF-8 line is skipped as corrupt content and never
        # escapes as an uncaught UnicodeDecodeError (a ValueError, NOT OSError) that
        # would 500 the memory routes. read_text() would instead decode the whole file
        # up front and raise past the callers' OSError guard on a single bad byte.
        for raw_line in cf.read_bytes().split(b"\n"):
            try:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError("correction line is not a JSON object")
                out.append(PendingCorrection.from_dict(data))
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
                skipped += 1
        if skipped:
            from localm.debuglog import logger as _dbg
            _dbg.warning(
                "memory corrections %s: skipped %d unparseable line(s) on load",
                cf, skipped)
        return out

    def _save_corrections(self, corrections: list[PendingCorrection]) -> None:
        cf = self._corrections_file()
        if not corrections:
            cf.unlink(missing_ok=True)
            return
        # "v": FORMAT_VERSION mirrors the main record store's stamp.
        # PendingCorrection.from_dict already filters to known dataclass fields, so
        # an unstamped line loads unchanged - no read-side change needed for
        # tolerance, same as the main store's "v" stamp.
        body = "\n".join(json.dumps({"v": FORMAT_VERSION, **c.to_dict()},
                                    ensure_ascii=False)
                         for c in corrections)
        self._atomic_write(cf, body + "\n")

    def _dismissed_file(self) -> Path:
        return self._file.with_suffix(".corrections-dismissed.json")

    def _load_dismissed(self) -> set:
        """The set of correction dedup keys the user REJECTED. Consolidation skips
        re-proposing these, so a dismissed supersession does not reappear every pass
        while the contradicting session is still in the recent window. A
        corrupt/absent file is treated as empty.

        Current files are ``{"v": FORMAT_VERSION, "keys": [...]}``; a pre-stamp
        file (a bare JSON array) still loads unchanged, same forward-tolerant
        approach as the main store's "v" stamp.

        Known asymmetry: this is the one stamped sidecar whose top-level SHAPE is
        an object rather than a dict-shaped line. A build older than the stamp
        reading a file written by this version sees a dict, falls through its own
        ``isinstance(data, list)`` check, and silently treats it as an empty
        dismissed set - so a downgrade or auto-rollback (see updater.py) after a
        dismissal was saved re-surfaces corrections the user already rejected. Not
        data loss, and it self-heals on the next dismissal."""
        df = self._dismissed_file()
        if not df.is_file():
            return set()
        # A present-but-UNREADABLE file (transient lock) RAISES via read_bytes:
        # collapsing it to an empty set would let resolve_correction's reject
        # branch rewrite the file with only the new key and wipe every prior
        # dismissal. That caller catches the OSError and skips the dismissed save.
        # Corrupt CONTENT is a different case that self-heals on the next write, so it
        # stays a warned empty set. Read the raw BYTES first (only an I/O failure,
        # OSError, means "unreadable"), then decode + parse INSIDE the try so a
        # bad-JSON OR invalid-UTF-8 body is treated as corrupt content. Decoding via
        # read_text() instead would raise UnicodeDecodeError (a ValueError, NOT an
        # OSError) straight past the callers' OSError guard and 500 the reject route.
        raw_bytes = df.read_bytes()
        try:
            data = json.loads(raw_bytes.decode("utf-8"))
            if isinstance(data, dict):
                raw = data.get("keys", [])
            elif isinstance(data, list):
                raw = data                       # unstamped bare array
            else:
                return set()
            return {tuple(k) for k in raw if isinstance(k, (list, tuple))}
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
            from localm.debuglog import logger as _dbg
            _dbg.warning(
                "memory corrections-dismissed %s: unparseable, treating as an empty "
                "dismissed set (self-heals on the next dismissal)", df)
            return set()

    def _save_dismissed(self, keys: set) -> None:
        df = self._dismissed_file()
        trimmed = list(keys)[-_CORRECTIONS_MAX:]        # bounded like the sidecar
        if not trimmed:
            df.unlink(missing_ok=True)
            return
        self._atomic_write(df, json.dumps(
            {"v": FORMAT_VERSION, "keys": [list(k) for k in trimmed]}))

    def propose_corrections(self, proposals: list[PendingCorrection]) -> int:
        """Append *proposals* to the pending-corrections sidecar, skipping any that
        duplicate an already-pending one OR one the user already REJECTED (same
        target/action/proposed text), so the same contradiction distilled run after
        run does not stack or re-nag. Newest are kept when the cap is exceeded.
        Returns the number newly recorded.

        Locked and re-read like the record methods so a proposal from a
        consolidation pass cannot clobber a concurrent accept/reject."""
        if not proposals:
            return 0
        with self._wlock('a correction proposal'):
            try:
                existing = self._load_corrections()
                dismissed = self._load_dismissed()
            except OSError as e:
                # A sidecar exists but could not be read (transient lock). Do NOT
                # proceed: _save_corrections below would rewrite the file with only
                # the freshly proposed entries and wipe the pending ones. Skip this
                # pass (nothing added, nothing lost), warn, and let the
                # consolidation caller keep going.
                from localm.debuglog import logger as _dbg
                _dbg.warning(
                    "memory corrections %s: sidecar unreadable, skipping this "
                    "propose pass to avoid wiping pending corrections: %s",
                    self._corrections_file(), e)
                return 0
            seen = {c.dedup_key() for c in existing}
            added = 0
            for p in proposals:
                key = p.dedup_key()
                if key in seen or key in dismissed:
                    continue                          # already pending, or rejected
                existing.append(p)
                seen.add(key)
                added += 1
            if added:
                self._save_corrections(existing[-_CORRECTIONS_MAX:])
            return added

    def corrections(self) -> list[PendingCorrection]:
        """Pending corrections whose target record still exists. A proposal whose
        target was deleted/evicted meanwhile is stale and dropped (and pruned from
        the sidecar) so the modal never shows an un-actionable suggestion.

        This READS but may also prune, so it wants the write lock - yet it backs
        GET /api/memory and `localm memory corrections`, and a read must not start
        failing because someone else holds the namespace. So the lock is bounded
        and OPTIONAL: on contention the answer is still returned, just without the
        opportunistic cleanup (which the next caller redoes), and the skip is
        logged."""
        try:
            with self._wlock("a stale-correction prune",
                             timeout=_REINFORCE_LOCK_WAIT):
                return self._corrections_locked(prune=True)
        except CollectionLockedError as e:
            from localm.debuglog import logger as _dbg
            _dbg.debug("memory corrections: listing without the stale-prune, the "
                       "namespace is being written elsewhere (%s)", e)
            with _namespace_lock(self._ns_hash):
                return self._corrections_locked(prune=False)

    def _corrections_locked(self, *, prune: bool) -> list[PendingCorrection]:
        """corrections()'s body, with the namespace lock already held."""
        self._load()
        try:
            corrs = self._load_corrections()
        except OSError as e:
            # Unreadable sidecar: show nothing this call rather than crash, and do
            # NOT run the stale-prune save below over a phantom-empty list (which
            # would wipe the file). Warned, not silently reported as "none".
            from localm.debuglog import logger as _dbg
            _dbg.warning(
                "memory corrections %s: unreadable, showing none for now: %s",
                self._corrections_file(), e)
            return []
        ids = {r.id for r in self._records}
        live = [c for c in corrs if c.target_id in ids]
        if prune and len(live) != len(corrs):
            self._save_corrections(live)          # drop stale entries
        return live

    def resolve_correction(self, correction_id: str, accept: bool, *,
                           embed_fn: Optional[EmbedFn] = None,
                           now: Optional[float] = None) -> Optional[dict]:
        """Apply or dismiss a pending correction. Returns a small status dict, or
        None when *correction_id* is unknown (a 404 for the route).

        accept: archive the target record to ``.forgotten.jsonl`` (recoverable),
        then apply the proposed change (replace the text, re-embedding, or delete
        the record). reject: keep the record, bump its ``updated`` so its
        last-confirmed staleness resets, and remember the dismissed suggestion so
        consolidation does not re-propose it (see ``_load_dismissed``). In BOTH
        cases the pending entry is removed. Atomic. If the target vanished
        meanwhile, the entry is simply dropped (nothing to apply)."""
        with self._wlock('a correction'):
            self._load()
            try:
                corrs = self._load_corrections()
            except OSError as e:
                # Cannot read the sidecar (transient lock): abort rather than crash or
                # act on a phantom-empty list. Non-destructive - the pending entry,
                # the record, and the dismissals are all left intact, so the user can
                # retry once the lock clears. Warned, not a silent 404-or-wipe.
                from localm.debuglog import logger as _dbg
                _dbg.warning(
                    "memory corrections %s: unreadable, cannot resolve %s now: %s",
                    self._corrections_file(), correction_id, e)
                return None
            corr = next((c for c in corrs if c.id == correction_id), None)
            if corr is None:
                return None
            now = time.time() if now is None else now
            target = self.get(corr.target_id)
            outcome = "target_gone"
            if target is not None and accept:
                # Archive the pre-change record so an accepted supersession is
                # recoverable, never a silent hard delete. If the archive FAILS,
                # abort: do NOT destroy the trusted record and do NOT report
                # success - leave the record and the pending correction intact so
                # the user can retry.
                if not self._archive_forgotten([target]):
                    return {"status": "archive_failed", "id": correction_id,
                            "target_id": corr.target_id, "action": corr.action}
                if corr.action == "delete":
                    self._records = [r for r in self._records if r.id != target.id]
                    self._vectors.pop(target.id, None)
                    outcome = "deleted"
                else:
                    target.text = (corr.proposed_text or "").strip()[:MAX_TEXT_LEN]
                    target.updated = now
                    target.last_used = now
                    # Re-embed so semantic recall tracks the new text; if no embedder
                    # is available, invalidate the stale vector (backfill re-embeds).
                    self._vectors.pop(target.id, None)
                    vec = self._embed_one(target.text, embed_fn)
                    if vec is not None:
                        self._vectors[target.id] = vec
                    outcome = "updated"
                self._save()
            elif target is not None and not accept:
                # Rejected: the user confirms the existing fact. Bump updated so the
                # staleness affordance resets, and REMEMBER this exact suggestion as
                # dismissed so consolidation does not re-propose it while the
                # contradicting session is still in the recent window - the propose
                # path never reads `updated`, so bumping it alone does nothing.
                target.updated = now
                self._save()
                try:
                    dismissed = self._load_dismissed()
                    dismissed.add(corr.dedup_key())
                    self._save_dismissed(dismissed)
                except OSError as e:
                    # Dismissed file unreadable: do NOT rewrite it (that would wipe
                    # every prior dismissal). The record is still confirmed and the
                    # pending entry is still cleared below; the only cost is this one
                    # dismissal is not recorded, so consolidation MAY re-propose it
                    # later - a bounded re-nag, not data loss. Warned rather than
                    # saved over.
                    from localm.debuglog import logger as _dbg
                    _dbg.warning(
                        "memory corrections-dismissed %s: unreadable, not recording "
                        "this dismissal to avoid wiping prior ones: %s",
                        self._dismissed_file(), e)
                outcome = "rejected"
            self._save_corrections([c for c in corrs if c.id != correction_id])
            return {"status": outcome, "id": correction_id,
                    "target_id": corr.target_id, "action": corr.action}


def _first_dim(vectors: dict) -> Optional[int]:
    for v in vectors.values():
        if v:
            return len(v)
    return None
