# SPDX-License-Identifier: AGPL-3.0-or-later
"""
On-device text embeddings via a DEDICATED embedding GGUF, on the bundled runtime.

This module loads a small, dedicated embedding model (bge / nomic, ~25-90 MB) into
its OWN llama.cpp model + context in EMBEDDINGS mode, independent of whatever chat
model is loaded, so semantic retrieval (RAG hybrid search, agent memory) works on
the default GGUF runtime with consistent quality and without disturbing the chat
model.

Why a dedicated model rather than the chat model, in order of how binding each
reason is:

1. CAPABILITY: the bundled GGUF chat backend cannot embed at all - the ctypes
   binding exposes no create_embedding (``backends/gguf.py``, ``can_embed=False``).
2. COST: loading a multi-GB chat model purely to embed would be wasteful, and
   under VRAM pressure would evict the resident one.
3. QUALITY: a chat model is a decoder LLM trained to predict the next token, not
   to place related texts near each other, so its pooled hidden states make poor
   embeddings. Measured 2026-07-15 through this very module, CPU-only, against
   memory's REL_COS_MIN=0.55 gate: bge-small separated related from unrelated
   pairs by +0.29 (min related 0.6498 vs max unrelated 0.3585, 4/4 gate), while
   Qwen2.5-0.5B's max UNRELATED cosine (0.7523) EXCEEDED its min RELATED cosine
   (0.7518) - NO threshold splits them. That is decoder anisotropy (no contrastive
   objective), not a pooling artifact: LAST-token pooling was measured too and
   scored worse (-0.17). The failure is silent - the vectors are non-zero,
   normalised and plausible - which is precisely why this path is not optional.

It is a process-wide, lazily-loaded singleton (``get_embedder`` / ``embed_texts``):
one small model is loaded once and shared by every caller (chat memory, coder
memory, /v1/embeddings). Loading serialises on the engine's process-global load lock
so it never races a chat-model load onto the GPU.

Provisioning (``resolve_embedding_model_path``): the ``embedding_model`` config key
is either a filesystem path, a registered model name, or a known key (default
``bge-small-en-v1.5``). A known model missing from ``<home>/models/embeddings/`` is
downloaded on demand, gated by the network policy (never behind ``net_mode=off``;
auto only under ``net_mode=allow`` - otherwise the user runs ``localm setup-embeddings``).
When no embedding model can be resolved, callers degrade to lexical-only retrieval;
the reason is recorded for ``last_error()`` (what the GUI's RAG-embedding status
reads) on every resolve attempt, not only when a load is actually attempted - a
registered model can name something real (localm pulled it, it is on disk) and
still not be usable HERE, because this embedder loads a single GGUF file only. The
common way that happens is a HuggingFace-format pull (a directory of shards, not a
GGUF): that is a real, different embedding path - see ``inference/backends/hf.py``'s
``HFBackend``, used when such a checkpoint is loaded as the primary model instead.

The native load (and every ``embed()`` call - it hits ``llama_decode``, the
same abort-prone native call class) runs inside an ISOLATED CHILD PROCESS
(``_embedder_runner.py``), not in this process - mirroring
``backends/gguf.py``'s containment for the identical native-abort risk (PR
#606: a native CUDA/HIP driver failure can hard-``abort()`` the whole process
in C, uncatchable from Python). ``GGUFEmbedder`` below is the raw, unguarded
native loader (constructed only inside that child); ``IsolatedEmbedder`` is
the parent-side handle ``get_embedder()`` actually returns.
"""

from __future__ import annotations

import atexit
import ctypes
import math
import re
import threading
from pathlib import Path
from typing import Callable, List, Optional

from localm import pathscrub
from localm.debuglog import dedup_native_stderr, logger
from localm.inference.backends.llamacpp._sizing import VramSizingMixin

# Known small embedding models, keyed by a friendly name -> (hf_repo, filename).
# Chosen for size + quality: bge-small is 24 MB Q4 and scores well; nomic is the
# common alternative. A user can also point embedding_model at any GGUF path or a
# registered model name.
KNOWN_EMBEDDING_MODELS = {
    "bge-small-en-v1.5": (
        "CompendiumLabs/bge-small-en-v1.5-gguf", "bge-small-en-v1.5-q4_k_m.gguf"),
    "nomic-embed-text-v1.5": (
        "nomic-ai/nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.Q4_K_M.gguf"),
}
DEFAULT_EMBEDDING_MODEL = "bge-small-en-v1.5"

# Same rationale as pull.py's _HF_ENDPOINT: HF_ENDPOINT/HF_HUB_ENDPOINT are
# ambient env vars localm never exposes as a setting, so the endpoint is
# pinned explicitly rather than left to whatever the shell happens to set.
_HF_ENDPOINT = "https://huggingface.co"

# llama.cpp LLAMA_POOLING_TYPE_* values. UNSPECIFIED (-1) is llama.cpp's own
# default and means "the model's declared pooling decides" (see llamacpp/_abi.py,
# which treats that -1 as an ABI keystone).
_POOLING_UNSPECIFIED = -1
_POOLING_NONE = 0
_POOLING_MEAN = 1
_POOLING_CLS = 2
_POOLING_LAST = 3

# Pooling settings a user may choose (config ``embedding_pooling``). "auto" is not
# a llama.cpp value: it is resolved per-model against what the GGUF declares.
POOLING_AUTO = "auto"
_POOLING_BY_NAME = {"none": _POOLING_NONE, "mean": _POOLING_MEAN,
                    "cls": _POOLING_CLS, "last": _POOLING_LAST}
_POOLING_NAMES = {v: k for k, v in _POOLING_BY_NAME.items()}
POOLING_CHOICES = [POOLING_AUTO, *_POOLING_BY_NAME]

# Why MEAN is the DEFAULT rather than unconditionally "whatever the model
# declares" (measured 2026-07-15 against the real GGUFs, via the bundled
# llama.dll):
#
#   model                 declares                 forced MEAN today
#   bge-small (default)   bert.pooling_type=2 CLS  works: +0.29 related/unrelated margin
#   nomic (known key)     nomic-bert...=1 MEAN     already exactly right
#   gte-Qwen2-1.5B        (no pooling key)         MEAN is the RESCUE (-1 -> NONE -> no output)
#   Qwen3-Embedding-0.6B  qwen3.pooling_type=3 LAST  MIS-POOLED - the real defect
#
# So MEAN is right for three of the four and is what every existing index was
# built with; switching the default to unconditionally follow the declared
# type would silently flip bge from MEAN to CLS at the SAME 384 dims - no dim
# guard fires (rag/store.py's mixed-dim check, memory's dim check), so every
# already-embedded collection would quietly stop matching new queries. But the
# fourth row - a model DECLARING last-token pooling - is a confirmed, narrow
# exception the table itself names "the real defect": MEAN measurably degrades
# a decoder-based embedder trained for LAST pooling, and there is no
# already-working index of THIS shape to protect (nothing was ever correctly
# built with mean for a LAST-declaring model). So the untouched default
# (``embedding_pooling`` never configured - _POOLING_UNSET) resolves LAST
# specifically when the model declares LAST, and MEAN for every other
# declaration (CLS, MEAN, unspecified) exactly as before - zero regression risk
# for bge/nomic/gte-Qwen2, and a decoder-based embedder like Qwen3-Embedding
# now genuinely works out of the box with no manual setting to discover. An
# EXPLICIT choice (including an explicit "mean") always wins as before - this
# only changes what happens when nothing was configured at all.
_POOLING_DEFAULT = _POOLING_MEAN
# Distinct from both an explicit numeric override and POOLING_AUTO (which
# unconditionally follows whatever is declared, including CLS - measured WORSE
# than the MEAN override for bge). Internal only: never a user-facing choice
# in POOLING_CHOICES: the user cannot "select" absence-of-configuration.
_POOLING_UNSET = "unset"
# The isolated worker sizes the embedding window to the loaded model's OWN
# native training context (llama_model_n_ctx_train, read once the model
# handle exists - see GGUFEmbedder.__init__) instead of a flat guess:
# different embedding models declare wildly different windows - bge-small-
# en-v1.5's (the bundled default) native window really is 512, but
# nomic-embed-text/bge-m3 declare 8192 and Qwen3-Embedding/gte-Qwen2 declare
# far more - and treating every model as if it matched the smallest bundled
# default silently truncated, and degraded, every embedding for any larger
# model with no signal to the user beyond a debug log line.
#
# _EMBED_CTX_CEILING bounds how large that auto-sized window is allowed to
# get: requesting a model's FULL native window (up to 32768 on some) for
# EVERY embed call, including a one-sentence memory fact, sizes the batch/
# scratch buffers to match on every single call, not just the rare long one -
# a real VRAM and per-call latency cost paid every time. 2048 tokens
# comfortably covers this project's own RAG chunk target (rag/chunk.py's
# CHUNK_CHARS=1200, observed in practice at 300-650+ tokens depending on text
# density) with wide headroom, while an explicit n_ctx argument (no in-tree
# caller passes one today, but the option stays for one that wants a specific
# window) still wins outright over the auto-sized value.
_EMBED_CTX_CEILING = 2048
# Fallback ONLY when a model's native window cannot be read at all (the API
# call fails, or returns an implausible <= 0 value) - the ORIGINAL flat
# default, so that narrow case still gets exactly the behavior that shipped
# before this fix, not a broken or arbitrary one.
_EMBED_CTX_FALLBACK = 512


def _resolve_embed_ctx(native_ctx_train: int) -> int:
    """The auto-sizing decision itself, isolated as a pure function so it is
    directly unit-testable without mocking the whole native load path: cap
    the model's own declared training window at _EMBED_CTX_CEILING, or fall
    back to _EMBED_CTX_FALLBACK when the model does not usefully declare one
    (<= 0 - a build too old for llama_model_n_ctx_train, or genuinely absent
    metadata)."""
    return min(native_ctx_train, _EMBED_CTX_CEILING) if native_ctx_train > 0 else _EMBED_CTX_FALLBACK


# The largest batch (in TEXTS, not tokens) this embedder will try to pack into
# one native multi-sequence llama_decode call - see _choose_n_seq_max, which
# is the reason this needs to be a search rather than a flat constant.
_EMBED_BATCH_TARGET = 32


def configure_embed_context(cp, n_ctx: int, n_seq_max: int, pooling_type: int):
    """Fill a llama_context_params for EMBEDDING and return it.

    A plain function taking the params object rather than code inline in
    __init__, so the settings below are testable WITHOUT a native runtime. The
    first version of the kv_unified test drove the whole GGUFEmbedder and passed
    on a dev box while failing on CI: with no llama.cpp provisioned, __init__
    raised long before it reached context creation, the pytest.raises() around
    it swallowed that, and the assertion then read an unset value. A test for a
    parameter should not depend on whether a GPU runtime exists.

    kv_unified: ONE shared KV cache across the sequences of a batch, instead of
    llama.cpp's default of carving n_ctx into n_seq_max private slices. See
    _pack_groups, which bounds a group by its SUMMED token count against n_ctx -
    the correct budget for a shared cache and far too generous for a sliced one.
    """
    cp.n_ctx = n_ctx
    cp.n_batch = n_ctx
    cp.n_ubatch = n_ctx      # non-causal encode needs ubatch >= seq len
    cp.n_seq_max = n_seq_max
    cp.kv_unified = True
    cp.embeddings = True
    cp.pooling_type = pooling_type
    return cp


def _choose_n_seq_max(n_ubatch: int, target_max: int = _EMBED_BATCH_TARGET) -> int:
    """How many sequences a multi-sequence embed batch may use, for a context
    whose n_ubatch is *n_ubatch*.

    ROOT CAUSE (measured 2026-08-05 via subprocess-isolated bisection across
    two models of different dim, three n_ctx values, and both CPU and GPU -
    see dev-notes/FINDING-embedder-serial-batching-2026-08-04.md): a
    multi-sequence llama_context whose ``n_seq_max`` does NOT evenly divide
    its ``n_ubatch`` hits a hard, uncatchable native
    ``GGML_ASSERT(ggml_can_mul_mat(a, b))`` abort() *during context creation*
    (llama_init_from_model's own internal graph-reserve warmup pass) - before
    any real batch is ever submitted. n_ubatch=512 with n_seq_max=8 works;
    n_seq_max=12 hard-crashes the process. This is NOT a property of the
    model (confirmed identical across dim=384 and dim=768, and across CPU and
    GPU) - it is a property of the (n_seq_max, n_ubatch) PAIR, so a fixed
    hardcoded n_seq_max would still be a latent crash for any future model
    whose resolved n_ctx (== n_ubatch here) does not happen to be a multiple
    of it.

    Searches powers of two descending from *target_max*, returning the first
    that evenly divides n_ubatch. Falls back to 1 - today's exact
    single-sequence behavior, proven safe by every model this embedder has
    ever loaded - if nothing larger divides evenly; 1 always divides
    everything, so this can never return an unsafe value regardless of what
    n_ctx a future model resolves to. In practice this lands on
    _EMBED_BATCH_TARGET for every currently-known model (2048, the
    _EMBED_CTX_CEILING, and 512, the _EMBED_CTX_FALLBACK, are both powers of
    two, and real GGUF native_ctx_train metadata is essentially always a
    power of two or otherwise highly composite) - but the search means an
    unusual future value degrades safely instead of crashing."""
    n = target_max
    while n > 1:
        if n_ubatch % n == 0:
            return n
        n //= 2
    return 1


def resolve_pooling_setting(spec: object) -> object:
    """Map the ``embedding_pooling`` config value to a llama.cpp pooling int,
    POOLING_AUTO for per-model resolution, or _POOLING_UNSET when nothing was
    configured (see _effective_pooling for what that resolves to). An
    unrecognised value falls back to the same per-model-aware default rather
    than failing the load, but says so (never silently)."""
    if spec is None:
        return _POOLING_UNSET
    text = str(spec).strip().lower()
    if not text:
        return _POOLING_UNSET
    if text == POOLING_AUTO:
        return POOLING_AUTO
    if text in _POOLING_BY_NAME:
        return _POOLING_BY_NAME[text]
    logger.warning(
        "embedding_pooling=%r is not one of %s; using the default",
        spec, ", ".join(POOLING_CHOICES))
    return _POOLING_UNSET


def declared_pooling_type(model, api) -> Optional[int]:
    """The pooling type the GGUF itself DECLARES (``<arch>.pooling_type``), or
    None when it declares none. Read from model metadata, so it needs no context.

    Verified 2026-07-15: bge-small -> 2 (CLS), nomic -> 1 (MEAN),
    Qwen3-Embedding-0.6B -> 3 (LAST), gte-Qwen2 / Qwen2.5-chat -> not declared.
    Best-effort by design: a model that declares nothing is the NORMAL case, and
    an unreadable value must not fail an otherwise fine load - the caller then
    just keeps its configured pooling (and this is debug-logged, not silent)."""
    try:
        if not api.has_model_meta_api():
            logger.debug("llama.dll exports no metadata reader; cannot read the "
                         "model's declared pooling type")
            return None
        arch = api.llama_model_meta_val_str(model, "general.architecture")
        if not arch:
            return None
        raw = api.llama_model_meta_val_str(model, f"{arch}.pooling_type")
        if raw is None:
            return None
        value = int(str(raw).strip())
        # A model that explicitly declares UNSPECIFIED has declared nothing
        # usable; report it the same as an absent key so "auto" falls back to
        # MEAN rather than asking llama.cpp to pool by an unspecified rule.
        return None if value == _POOLING_UNSPECIFIED else value
    except Exception as e:
        logger.debug("could not read the declared pooling type (%s: %s)",
                     type(e).__name__, e)
        return None


def _effective_pooling(requested: object, declared: Optional[int]) -> int:
    """Resolve the pooling actually used:
    - UNSET (nothing configured) applies the measured-safe default: MEAN,
      EXCEPT when the model declares LAST-token pooling specifically - the one
      case the module docstring's table names "the real defect", where there is
      no already-working mean-built index of this shape to protect. Every
      other declaration (CLS, MEAN, unspecified) keeps the exact MEAN default
      it always had.
    - an explicit choice (a real pooling int, including an explicit "mean")
      always wins as-is, never overridden - the user's call, not ours;
    - AUTO honours what the model declares, falling back to MEAN when it
      declares nothing usable (a decoder declaring NONE would otherwise
      produce no pooled output at all - see _embed_one's null-embedding
      guard)."""
    if requested == _POOLING_UNSET:
        return _POOLING_LAST if declared == _POOLING_LAST else _POOLING_DEFAULT
    if requested != POOLING_AUTO:
        return int(requested)
    if declared in (_POOLING_MEAN, _POOLING_CLS, _POOLING_LAST):
        return declared
    return _POOLING_DEFAULT


def pooling_name(value: Optional[int]) -> str:
    if value is None:
        return "not declared"
    return _POOLING_NAMES.get(value, f"type {value}")


def _embeddings_dir() -> Path:
    from localm.config import home_dir
    return home_dir() / "models" / "embeddings"


_URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]+://")


def _current_spec() -> str:
    """The ``embedding_model`` config value, defaulted and stripped - the
    identity ``resolve_embedding_model_path`` resolves from, and the same
    identity ``_LOAD_FAILED_SPEC``/``_set_resolve_outcome``/``get_embedder``
    compare against to tell "still the same broken spec" from "the config
    changed". Factored out so every one of those derives it identically -
    two independent copies of this read+default+strip is exactly the kind of
    derivation that can silently drift the day only one of them changes."""
    from localm.config import load_config
    return str(load_config().get("embedding_model") or DEFAULT_EMBEDDING_MODEL).strip()


def _nonlocal_spec_reason(spec: str) -> Optional[str]:
    """Why *spec* is not something this module may hand to the filesystem, or
    None if it is fine. Purely LEXICAL - no syscall - so it can run first.

    ``embedding_model`` accepts exactly three shapes: a KNOWN_EMBEDDING_MODELS
    key, a registered model name, or a LOCAL GGUF path. A UNC/device path and a
    URL are none of those, so refusing them here loses no documented behaviour
    while keeping the string away from the SMB redirector.

    The UNC/device half is pathsafe's shared predicate - CALLED, never
    reimplemented, so a fix there reaches this call site. It covers the mixed
    separator spellings (``\\/host\\share``) too, which Windows resolves as UNC;
    a test here asserts that delegation so this policy cannot drift into a fork.

    The URL-scheme half is deliberately LOCAL, and belongs here rather than in
    pathsafe: "is this string a locator rather than a path at all" is a question
    about what this SETTING accepts, not about path syntax, and a scheme rule
    inside a path-confinement primitive is a second concern that the next caller
    would bend or fork. The regex requires TWO or more scheme characters so a
    Windows drive letter can never match - a drive is exactly one letter, so
    ``C://models/x.gguf`` stays a path while ``http://``, ``file://`` and
    ``smb://`` are caught."""
    from localm.pathsafe import is_unc_or_device_path
    if is_unc_or_device_path(spec):
        return ("it is a UNC or device path; only a local filesystem path, a "
                "known key, or a registered model name is allowed")
    if _URL_SCHEME_RE.match(spec):
        return "it is a URL, not a local file path"
    return None


def _set_resolve_outcome(spec: str, reason: Optional[str]) -> None:
    """The ONE place any RESOLVE-side code may touch ``_LAST_ERROR`` - never
    assigned directly anywhere else in this module below this point. Every
    resolve-side writer (``_record_resolve_failure``, ``_record_resolve_success``,
    and ``_download_known``'s two policy-decline branches) routes through this
    single choke point, specifically so the guard below cannot be silently
    reopened by a future write site that forgets to carry it - the exact way
    this bug happened the first time: ``_record_resolve_success`` got the
    guard, ``_record_resolve_failure`` and ``_download_known`` (added in the
    same change) did not, and a status poll silently destroyed a genuine load
    failure the instant it re-resolved a not-yet-downloaded known key (#996's
    own regression, root-caused live against a real corrupted-then-deleted
    model file).

    *spec* is the ``embedding_model`` config value the CALLER is currently
    resolving - required, not optional, because a bare "is anything latched"
    check (the previous ``_LOAD_FAILED: bool``) cannot tell a stale failure
    for an abandoned spec apart from a live one for the spec actually
    configured now. That conflation was measured to be a real regression:
    once ANY load failed, a completely unrelated, ACTIVELY REFUSED spec
    (e.g. a UNC path just typed into ``embedding_model``) had its own live
    resolve failure silently swallowed here, and ``last_error()`` kept
    reporting the old, unrelated reason instead - defeating the whole point
    of the step-0 UNC/URL refusal that recorded it in the first place.

    Suppression applies ONLY when *spec* matches the latched
    ``_LOAD_FAILED_SPEC`` exactly:
      * A successful resolve for the SAME spec proves a file exists - already
        true before that load attempt failed too, so re-confirming it proves
        nothing new.
      * A resolve FAILURE for the SAME spec CAN happen once ``_LOAD_FAILED_SPEC``
        matches it (the file can be removed, or otherwise stop resolving,
        between the load attempt and the next poll - verified live, this is
        not a hypothetical) but is still suppressed: the latched load failure
        is the more specific, harder-won fact, and a caller who already knows
        loading this exact spec failed learns nothing new from "and now it
        cannot even be found."
      * A resolve outcome for ANY OTHER spec is new, live information about a
        setting the user has since changed (or a poll racing a config edit)
        and is never suppressed, regardless of what is latched.
    Only an ACTUAL new load attempt for a MATCHING spec (``get_embedder()``'s
    own except/success branches), a load attempt for a DIFFERENT spec (which
    clears the stale latch itself - see ``get_embedder()``), or an explicit
    ``reset_embedder()`` may otherwise change what is latched."""
    global _LAST_ERROR
    if _LOAD_FAILED_SPEC is not None and _LOAD_FAILED_SPEC == spec:
        return
    _LAST_ERROR = reason


def _record_resolve_failure(spec: str, reason: str) -> None:
    """Record *reason* as the current resolve failure for *spec* (subject to
    ``_set_resolve_outcome``'s guard), and log it - WARNING the first time
    this exact reason is seen, DEBUG on repeats of the same one.

    The dedup matters because ``resolve_embedding_model_path`` can run on every
    single ``embed_texts()`` call while no embedder is loaded: ``get_embedder()``
    deliberately never caches a missing-model result (so a mid-session
    ``localm setup-embeddings`` is picked up without a restart - see its
    docstring), which means an UNCHANGED misconfiguration re-hits this function on
    every embed, not once. Logging WARNING every time would flood the log for the
    life of the process (the exact failure mode named in this project's own
    diff-review notes: a warning on somebody else's timer). ``last_error()`` still
    carries the reason on every single call regardless of the log level, so the
    GUI is never blind to it even once the log goes quiet."""
    global _LAST_RESOLVE_WARNED
    _set_resolve_outcome(spec, reason)
    if _LAST_RESOLVE_WARNED != reason:
        logger.warning(reason)
        _LAST_RESOLVE_WARNED = reason
    else:
        logger.debug(reason)


def _record_resolve_success(spec: str) -> None:
    """Clear a prior RESOLVE failure once *spec* resolves (subject to
    ``_set_resolve_outcome``'s guard) - a fixed config (or one that never
    needed fixing) must not keep reporting a stale reason via ``last_error()``,
    and a later, DIFFERENT misconfiguration must warn again rather than
    staying silenced by a dedup entry from an unrelated failure."""
    global _LAST_RESOLVE_WARNED
    _set_resolve_outcome(spec, None)
    _LAST_RESOLVE_WARNED = None


def resolve_embedding_model_path(*, allow_download: Optional[bool] = None) -> Optional[str]:
    """Resolve the configured embedding model to a GGUF path, or None.

    Order: an explicit filesystem path -> a registered model name -> a known key
    (downloaded into <home>/models/embeddings if missing and the net policy allows).
    ``allow_download`` overrides the policy (used by ``localm setup-embeddings`` to
    force the fetch); default follows net_mode (auto only under 'allow').

    Every failure to resolve is recorded via ``last_error()`` (see
    ``_record_resolve_failure``/``_record_resolve_success``) - including a
    registered model that names something real but is not a single GGUF file
    (almost always a HuggingFace-format pull: a directory of safetensors shards,
    not a file), which used to be indistinguishable from "not found at all" and
    surfaced only as a DEBUG log line nobody but the operator could see (#949)."""
    spec = _current_spec()
    if not spec:
        return None

    # 0. Refuse a non-local spec BEFORE any filesystem call. Ordering is the
    #    whole point: a single is_file() on a UNC path hands the string to the
    #    Windows SMB redirector, which blocks for minutes on an unroutable host
    #    and auto-authenticates against a reachable one, leaking the host's
    #    net-NTLMv2 credential outbound. That happens before any of the three
    #    lookups below could reject the value, so the check cannot go after them.
    #    Returning None rather than raising keeps the documented contract ("not
    #    resolvable -> None -> lexical fallback"), but the reason is recorded and
    #    logged, never swallowed: a silently ignored setting otherwise looks
    #    identical to a model that is merely not downloaded yet (rule 5).
    bad = _nonlocal_spec_reason(spec)
    if bad:
        _record_resolve_failure(
            spec,
            # NOT {spec!r}: repr() doubles backslashes in a Windows path, so
            # pathscrub's literal-prefix match (and its regex backstop, which
            # requires exactly one separator after the drive letter) never
            # recognises the escaped form - the home directory, and with it
            # the OS account name, survived scrubbing this way (measured
            # 2026-08-04). Quoting without repr() keeps the readability and
            # lets the scrub see the real separators.
            f"ignoring embedding_model '{spec}': {bad}. Use a known key "
            f"{tuple(KNOWN_EMBEDDING_MODELS)}, a registered model name, or a "
            "local GGUF path.")
        return None

    # 1. An explicit path to a GGUF.
    p = Path(spec).expanduser()
    if p.is_file():
        _record_resolve_success(spec)
        return str(p)

    # 2. A registered model name. A hit that does not resolve to a single FILE is
    #    a REAL, DIFFERENT outcome from "not registered at all" - the spec named
    #    something that genuinely exists (get_model_info found it), it is just not
    #    something this GGUF-only embedder can load. localm's own model pull can
    #    produce exactly this: a HuggingFace-format model lands as a DIRECTORY
    #    (shards, config.json, tokenizer files), and get_model_info's own contract
    #    is "exists" (file or directory), not "is a loadable GGUF" - collapsing
    #    the two into the same "not found" message (as this used to) hid exactly
    #    the case in #949, where localm itself downloaded the model successfully
    #    and then denied it existed.
    registered_not_gguf = None   # None: no registry hit; else bool, is it a directory
    try:
        from localm.model_manager.registry import get_model_info
        info = get_model_info(spec)
        if info:
            path = info[0] if isinstance(info, tuple) else info
            # Same ordering rule as step 0's spec check: refuse a UNC/device path
            # BEFORE any stat, never after. A registered entry's stored path is
            # normally trustworthy (it was written by a separately-authorized
            # `localm pull`/`add`, not derived live from this request), but this
            # function has no way to tell "the registry" from "a hand-edited
            # registry.json" apart, and the cost of being wrong is the same
            # outbound SMB/NTLM auto-auth step 0 exists to prevent. Treating a
            # UNC hit as "nothing usable here" (fall through, same as an absent
            # entry) costs nothing: no legitimate registration produces one.
            from localm.pathsafe import is_unc_or_device_path
            if path and not is_unc_or_device_path(str(path)):
                p2 = Path(path)
                if p2.is_file():
                    _record_resolve_success(spec)
                    return str(path)
                try:
                    registered_not_gguf = p2.is_dir()
                except OSError:
                    # A stat that fails differently from the is_file() above (a
                    # permission error, a race with something deleting the path)
                    # - still a real registry hit, just of unknown shape. Report
                    # it generically rather than letting the exception escape:
                    # embed_texts() has no try/except around get_embedder(), and
                    # /v1/embeddings only catches RuntimeError, so an uncaught
                    # OSError here would reach the caller as an opaque 500
                    # instead of the informative 422 this function exists to
                    # produce.
                    registered_not_gguf = False
    except Exception:
        pass

    # 3. A known embedding-model key.
    known = KNOWN_EMBEDDING_MODELS.get(spec)
    if not known:
        # Neither branch uses {spec!r} - see the why-comment on the identical
        # choice at this function's step-0 refusal above.
        if registered_not_gguf is not None:
            kind = ("a directory (a HuggingFace-format model)"
                    if registered_not_gguf else "not a single file")
            reason = (
                f"embedding_model '{spec}' is registered but resolves to {kind}, "
                "not a GGUF file. This dedicated embedder only loads a GGUF "
                f"embedding model: pick a known key {tuple(KNOWN_EMBEDDING_MODELS)}, "
                "a registered GGUF model, or a local .gguf path. A HuggingFace "
                "embedding checkpoint is used directly when loaded as the primary "
                "model instead, not via embedding_model.")
        else:
            reason = (
                f"embedding_model '{spec}' is not a path, a registered model, or "
                f"a known key {tuple(KNOWN_EMBEDDING_MODELS)}.")
        _record_resolve_failure(spec, reason)
        return None
    repo, filename = known
    dest = _embeddings_dir() / filename
    if dest.is_file():
        _record_resolve_success(spec)
        return str(dest)
    result = _download_known(spec, repo, filename, dest, allow_download)
    if result:
        _record_resolve_success(spec)
    return result


def _download_known(name: str, repo: str, filename: str, dest: Path,
                    allow_download: Optional[bool]) -> Optional[str]:
    """Fetch a known embedding GGUF, gated by the network policy.

    Every failure path also records into ``last_error()`` (like
    ``resolve_embedding_model_path``'s own failure branches - both policy
    branches below route through the SAME ``_set_resolve_outcome`` choke
    point, so its spec-aware guard applies here too without being
    re-implemented; *name* is that spec, since this is only ever called with
    the ``spec`` a caller in ``resolve_embedding_model_path`` was already
    resolving), so GET /api/rag/embedding's ``error`` field explains a
    policy-gated or failed download too. The log LEVEL for the two
    policy-gated branches stays INFO regardless of whether the write is
    skipped - an unset net_mode or a deliberately offline box is an expected
    state, not a defect worth a WARNING - only the download-failure branch
    below (a real fault) warns, deduped the same way as
    ``resolve_embedding_model_path``'s own WARNING-worthy failures."""
    from localm.netpolicy import network_mode
    if allow_download is None:
        allow_download = network_mode() == "allow"
    if not allow_download:
        reason = (
            f"embedding model {name!r} not present and not auto-downloading "
            f"(net_mode={network_mode()}); run 'localm setup-embeddings' or set "
            "net_mode=allow to enable semantic search (memory/RAG use lexical "
            "BM25 until then)")
        _set_resolve_outcome(name, reason)
        logger.info(reason)
        return None
    if network_mode() == "off":
        reason = f"embedding model {name!r} missing and network is off; lexical-only"
        _set_resolve_outcome(name, reason)
        logger.info(reason)
        return None
    try:
        from huggingface_hub import hf_hub_download
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("downloading embedding model %s/%s (one-time)...", repo, filename)
        got = hf_hub_download(repo, filename, local_dir=str(dest.parent), endpoint=_HF_ENDPOINT)
        # hf may nest under the repo dir; normalise to dest.
        got_p = Path(got)
        if got_p.resolve() != dest.resolve() and got_p.is_file():
            return str(got_p)
        return str(dest) if dest.is_file() else (str(got_p) if got_p.is_file() else None)
    except Exception as e:
        _record_resolve_failure(
            name, f"embedding model {name!r} download failed ({e}); lexical-only")
        return None


def _warn_if_context_config_drifted(api, ctx, requested_n_ctx: int) -> None:
    """Best-effort, log-only comparison of what this embedder REQUESTED for its
    context (n_ctx, and kv_unified=True - see configure_embed_context) against
    what the loaded native runtime actually reports, via llama_n_ctx /
    llama_n_ctx_seq - the closest observable proxy for kv_unified, since
    llama.cpp exposes no direct getter for that flag once a context exists.

    With kv_unified honoured, n_ctx_seq is close to n_ctx; a runtime that
    silently falls back to slicing n_ctx across n_seq_max private KV slots
    instead reports a much smaller n_ctx_seq - exactly the field-reported shape
    (a large embedding batch failing native decode). This can only OBSERVE and
    log the drift, not explain it (see _abi.py's own kv_unified diagnostic for
    the complementary load-time struct-offset check) - the root cause needs a
    live reproduction on the affected hardware.

    Never raises: a probe failure here must not take down an otherwise-working
    embedder, and this never changes what was actually configured."""
    try:
        actual_n_ctx = int(api.llama_n_ctx(ctx))
        actual_n_ctx_seq = api.llama_n_ctx_seq(ctx)
    except Exception as e:
        logger.debug("embedder context drift check failed: %s", e)
        return
    if actual_n_ctx != requested_n_ctx:
        logger.warning(
            "embedder: requested n_ctx=%d but the loaded context reports "
            "n_ctx=%d - the runtime may not honour the requested context size",
            requested_n_ctx, actual_n_ctx)
    if not isinstance(actual_n_ctx_seq, int):
        return                    # no accessor (older build), or a non-int
                                   # probe result - nothing usable to compare
    # A generous margin, not exact equality: kv_unified honoured can still
    # leave n_ctx_seq a hair under n_ctx on some builds. Below half is only
    # reachable via the sliced-KV fallback (n_ctx / n_seq_max, and n_seq_max
    # is always >= 2 whenever this matters - see _choose_n_seq_max), so it is
    # a safe, non-noisy threshold for "kv_unified was silently not honoured".
    if actual_n_ctx_seq < requested_n_ctx // 2:
        logger.warning(
            "embedder: requested kv_unified=True (n_ctx=%d) but the loaded "
            "context reports n_ctx_seq=%d, far smaller than requested - "
            "kv_unified was likely not honoured by this runtime build, and a "
            "large embedding batch may fail to decode",
            requested_n_ctx, actual_n_ctx_seq)


class GGUFEmbedder:
    """A dedicated embedding GGUF loaded in embeddings mode via the native llama.dll."""

    def __init__(self, model_path: str, *, n_gpu_layers: int = 99,
                 n_ctx: Optional[int] = None,
                 pooling_type: object = _POOLING_DEFAULT,
                 gpu_split_ratios: Optional[list] = None) -> None:
        from localm.inference.backends.llamacpp import _api as api
        from localm.inference.backends.llamacpp._structs import (
            llama_pos, llama_seq_id, llama_token)
        self._api = api
        self._llama_token = llama_token
        self._llama_pos = llama_pos
        self._llama_seq_id = llama_seq_id
        self.model_path = model_path
        self._lock = threading.RLock()
        # Resolved once the model handle exists, below - None here means "not
        # yet sized"; nothing that reads self.n_ctx is reachable before
        # __init__ finishes.
        self.n_ctx = n_ctx
        self._model = None
        self._ctx = None
        self._vocab = None
        self._mem = None
        self.dim = 0
        # What the GGUF declares vs what we actually pool with. Reported up to
        # IsolatedEmbedder (via the runner's load meta) so the PARENT can warn a
        # user whose model is being mis-pooled - this class runs in an isolated
        # child process, so it reports facts rather than owning that decision.
        self.declared_pooling: Optional[int] = None
        self.pooling_type: int = _POOLING_DEFAULT

        if not api.has_embeddings_api():
            raise RuntimeError(
                "this llama.dll build does not expose the embeddings API")
        api.llama_backend_init()
        mp = api.llama_model_default_params()
        mp.n_gpu_layers = n_gpu_layers
        if n_gpu_layers >= 99:
            # See llamacpp/llama.py: newer builds have no use_mmap field, and
            # assigning it would silently write into check_tensors instead.
            from localm.inference.backends.llamacpp._structs import set_use_mmap
            set_use_mmap(mp, False)
        # Multi-GPU: honour the configured main_gpu_index / gpu_split_indices,
        # same as the chat backend (see discover.apply_main_gpu/apply_gpu_split
        # and llamacpp/llama.py). The returned buffer must stay alive through
        # llama_load_model_from_file below - it is read once at load time, not
        # held as a live pointer. VRAM preflight (both the single-GPU
        # VramSizingMixin._check_vram()-style check and the multi-GPU
        # gpu_split_shortfall() check) now lives in the PARENT
        # (IsolatedEmbedder, below) instead of here, mirroring how
        # GgufWorker.load() carries no preflight of its own (only
        # GgufBackend.load() does) - this class is the raw native loader from
        # here on, isolated inside a child process by _embedder_runner.py.
        from localm.discover import apply_gpu_split, apply_main_gpu
        apply_main_gpu(mp)
        # gpu_split_ratios: the PARENT's already-resolved effective ratios
        # (auto free-VRAM-proportional distribution) - this isolated child
        # must not probe for them itself (discover.resolve_auto_split_ratios).
        _tensor_split_keepalive = apply_gpu_split(
            mp, ratios_override=gpu_split_ratios)
        # This isolated child (see the class docstring) never suppressed or
        # grouped its native stderr at all - unlike every load path in
        # llamacpp/llama.py, which wraps its own equivalent calls in
        # _quiet_stderr/dedup_native_stderr. llama_load_model_from_file's
        # per-tensor "create_tensor"/"load_tensors" lines and
        # llama_init_from_model's "sched_reserve"/"graph_reserve" lines went
        # straight to whatever this child's inherited fd 2 pointed at - raw,
        # ungrouped, and (via multiprocessing 'spawn' fd inheritance) landing
        # directly in the shared debug log file with no timestamp/level
        # prefix, exactly like the file always gets for the chat backend too
        # (see debuglog.py's dedup_native_stderr docstring: the raw record is
        # never grouped) - but with no LIVE view at all, unlike the chat
        # backend, and no repeat-count collapsing of a chatty line (#951).
        # One contiguous scope covers both native calls below (never
        # re-entered per call - a model load is one-shot, not a per-token
        # loop, so dedup_native_stderr's background-thread cost is paid once).
        #
        # Does NOT violate "an isolated child must never console.print" (see
        # HFWorker's UnicodeEncodeError incident): that invariant is about
        # rich.Console().print() specifically - a child's stdout does not
        # inherit the parent's console encoding, and rich's default crashed on
        # a literal unicode glyph. dedup_native_stderr never touches rich; its
        # console stream is a plain file object opened by _stable_console_stream()
        # with encoding="utf-8", errors="backslashreplace" (debuglog.py) -
        # built specifically so an unencodable byte degrades to an escaped
        # sequence instead of raising. It also does not use stdout/stdin, so it
        # cannot corrupt this child's IPC (multiprocessing Queues, never stdio -
        # see _embedder_runner.py's protocol docstring). This exact mechanism
        # already runs inside GgufWorker's own isolated child today (llama.py's
        # _generate(), PR #443) - reusing it here follows established
        # precedent rather than introducing a new console-writing path.
        with dedup_native_stderr():
            self._model = api.llama_load_model_from_file(model_path, mp)
            if not self._model:
                raise RuntimeError(f"failed to load embedding model: {model_path}")
            self._vocab = api.llama_model_get_vocab(self._model)
            self.dim = int(api.llama_model_n_embd(self._model))
            # Read what the model declares BEFORE creating the context (a metadata
            # read needs no context), so "auto" can honour it and so a mis-pool is
            # reportable either way.
            self.declared_pooling = declared_pooling_type(self._model, api)
            self.pooling_type = _effective_pooling(pooling_type, self.declared_pooling)
            logger.debug("embedder %s: declared pooling %s, using %s",
                         Path(model_path).name, pooling_name(self.declared_pooling),
                         pooling_name(self.pooling_type))
            if self.n_ctx is None:
                native_ctx = int(api.llama_model_n_ctx_train(self._model))
                self.n_ctx = _resolve_embed_ctx(native_ctx)
                logger.debug(
                    "embedder %s: native training window %d token(s), using %d",
                    Path(model_path).name, native_ctx, self.n_ctx)
            cp = api.llama_context_default_params()
            # How many texts embed() may pack into one native decode call -
            # see _choose_n_seq_max's own docstring for why this cannot be a
            # flat constant (a hard native crash, not a graceful error, for
            # the wrong (n_seq_max, n_ubatch) pairing).
            self._n_seq_max = _choose_n_seq_max(self.n_ctx)
            configure_embed_context(cp, self.n_ctx, self._n_seq_max,
                                    self.pooling_type)
            self._ctx = api.llama_init_from_model(self._model, cp)
            if not self._ctx:
                api.llama_free_model(self._model)
                self._model = None
                raise RuntimeError("failed to create embedding context")
        self._mem = api.llama_get_memory(self._ctx) if api.has_memory_api() else None
        _warn_if_context_config_drifted(api, self._ctx, self.n_ctx)

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize *text*, truncated to fit n_ctx. Returns ``[]`` only for
        the degenerate retokenize-failure case below - a real success always
        yields at least the BERT CLS/SEP special tokens, so an empty list is
        an unambiguous "could not tokenize this at all" signal to callers,
        never a legitimate zero-token text."""
        api = self._api
        raw = (text or " ").encode("utf-8")
        buf = (self._llama_token * self.n_ctx)()
        n = api.llama_tokenize(self._vocab, raw, len(raw), buf, self.n_ctx,
                               True, True)   # add_special (BERT CLS/SEP), parse_special
        if n < 0:
            # Over-long input: llama_tokenize returns -(tokens needed) and, on
            # this DLL, writes NOTHING into the short buffer (probe-verified),
            # so treating the zero-filled buffer as tokens embedded EVERY
            # over-long text to one identical garbage vector (all token 0;
            # memory-audit 2026-07-02, high). Retokenize into a right-sized
            # buffer and truncate explicitly to the context window.
            needed = -n
            full = (self._llama_token * needed)()
            n2 = api.llama_tokenize(self._vocab, raw, len(raw), full, needed,
                                    True, True)
            if n2 <= 0:                      # should not happen; fail visibly
                logger.warning(
                    "embedder: retokenize of an over-long text failed (%d)", n2)
                return []
            # Keep the first n_ctx tokens but preserve the FINAL token of the
            # full sequence: with add_special=True on the BERT-family models
            # this embedder serves (bge/nomic), that is the [SEP] the pooled
            # encoding expects; dropping it degrades the embedding.
            for i in range(self.n_ctx):
                buf[i] = full[i]
            buf[self.n_ctx - 1] = full[n2 - 1]
            n = self.n_ctx
            logger.debug(
                "embedder: input of %d tokens truncated to the %d-token window",
                n2, self.n_ctx)
        if n <= 0:
            return []
        return list(buf[:n])

    def _decode_single(self, tokens: List[int]) -> List[float]:
        """One sequence via ``llama_batch_get_one`` - the pre-existing,
        proven-fast path for a lone text. Deliberately kept as its OWN path
        rather than folded into the multi-sequence one below (a "batch of
        one"): measured, a real multi-sequence batch is SLOWER for a single
        sequence (0.60x on GPU - the batch-allocation/seq-id-array setup
        costs more than it saves when there is only one sequence to pack),
        and a lone embed (a chat memory query, most /v1/embeddings calls in
        practice) is the common case that must not regress."""
        api = self._api
        if self._mem is not None:
            api.llama_memory_clear(self._mem, True)
        if not tokens:
            return [0.0] * self.dim
        arr = (self._llama_token * len(tokens))(*tokens)
        batch = api.llama_batch_get_one(arr, len(tokens))
        ret = api.llama_decode(self._ctx, batch)
        if ret != 0:
            raise RuntimeError(f"embedding decode failed (code {ret})")
        ptr = api.llama_get_embeddings_seq(self._ctx, 0)
        if not ptr:
            raise RuntimeError("null embedding (pooling produced no output)")
        v = [float(ptr[i]) for i in range(self.dim)]
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else v

    def _decode_batch(self, token_lists: List[List[int]]) -> List[List[float]]:
        """Decode 2+ texts in ONE native ``llama_decode`` call, each as its
        own sequence (`self._n_seq_max` computed at load time - see
        ``_choose_n_seq_max``). Proven correct against the pre-existing
        serial path (cosine similarity 0.9999-1.0, measurement unit
        2026-08-05) before ever being trusted for its timing.

        FAILURE GRANULARITY (AGENTS.md rule 5 - never let a partial failure
        report as success): a nonzero decode return, OR a null/non-finite
        readout for ANY ONE sequence in the group, raises and discards the
        WHOLE group's result - never a partial list, and never a fabricated
        placeholder vector for the sequence that failed. This does not
        change embed()'s external contract: the pre-existing serial loop
        ALREADY had all-or-nothing semantics (one _embed_one raising
        propagated out of the whole call, per Python's ordinary list-
        comprehension exception behavior - no caller anywhere in this
        codebase ever received or handled a partial result). Batching moves
        the failure UNIT from "one text" to "one group"; it does not
        introduce a new way for a caller to see wrong data as success."""
        api = self._api
        if self._mem is not None:
            api.llama_memory_clear(self._mem, True)
        n_seq = len(token_lists)
        total_tokens = sum(len(t) for t in token_lists)
        batch = api.llama_batch_init(total_tokens, 0, n_seq)
        try:
            token_arr = ctypes.cast(batch.token, ctypes.POINTER(self._llama_token))
            pos_arr = ctypes.cast(batch.pos, ctypes.POINTER(self._llama_pos))
            n_seq_id_arr = ctypes.cast(batch.n_seq_id, ctypes.POINTER(ctypes.c_int32))
            seq_id_arr = ctypes.cast(
                batch.seq_id, ctypes.POINTER(ctypes.POINTER(self._llama_seq_id)))
            logits_arr = ctypes.cast(batch.logits, ctypes.POINTER(ctypes.c_int8))
            idx = 0
            for seq, toks in enumerate(token_lists):
                for pos, tok in enumerate(toks):
                    token_arr[idx] = tok
                    pos_arr[idx] = pos
                    n_seq_id_arr[idx] = 1
                    seq_id_arr[idx][0] = seq
                    # Pooled/non-causal embedding needs output for every
                    # token in the sequence, not just the last one (unlike
                    # causal next-token generation) - matches llama.cpp's own
                    # embedding.cpp example's batch-building convention.
                    logits_arr[idx] = 1
                    idx += 1
            batch.n_tokens = total_tokens

            ret = api.llama_decode(self._ctx, batch)
            if ret != 0:
                raise RuntimeError(f"batched embedding decode failed (code {ret})")

            out = []
            for seq in range(n_seq):
                ptr = api.llama_get_embeddings_seq(self._ctx, seq)
                if not ptr:
                    raise RuntimeError(
                        f"null embedding for sequence {seq} of {n_seq} in a "
                        "batched decode (pooling produced no output)")
                v = [float(ptr[i]) for i in range(self.dim)]
                norm = math.sqrt(sum(x * x for x in v))
                if not math.isfinite(norm):
                    raise RuntimeError(
                        f"non-finite embedding for sequence {seq} of {n_seq} "
                        "in a batched decode")
                out.append([x / norm for x in v] if norm else v)
            return out
        finally:
            api.llama_batch_free(batch)

    def _pack_groups(self, token_lists: List[List[int]]) -> List[List[int]]:
        """Group token-list INDICES for one embed() call into batches of up
        to ``self._n_seq_max`` texts, never letting a group's SUMMED token
        count exceed ``self.n_ctx`` (== n_ubatch here) - llama.cpp's own hard
        constraint for this non-causal 'encoder' architecture ("encoder
        requires n_ubatch >= n_tokens": unlike causal generation, it cannot
        split one micro-batch across multiple internal passes). A single
        text is already truncated to fit n_ctx by _tokenize, so it always
        fits alone; only MULTIPLE texts sharing a group can overflow, and
        this shrinks the group automatically for longer texts rather than
        ever submitting a batch that would violate the constraint."""
        groups: List[List[int]] = []
        current: List[int] = []
        current_tokens = 0
        for i, toks in enumerate(token_lists):
            n = len(toks)
            if current and (len(current) >= self._n_seq_max
                            or current_tokens + n > self.n_ctx):
                groups.append(current)
                current = []
                current_tokens = 0
            current.append(i)
            current_tokens += n
        if current:
            groups.append(current)
        return groups

    def embed(self, texts: List[str]) -> List[List[float]]:
        """L2-normalised embedding per text (aligned 1:1 with *texts*).

        Multiple texts sharing one call are packed into groups of up to
        ``self._n_seq_max`` and decoded with ONE native ``llama_decode`` per
        group (measured: a real, growing speedup on GPU - 1.6x/2.4x/4.5x at
        N=2/4/8, still climbing at the largest N tested; ~1.0x, no
        regression, on CPU - see dev-notes/FINDING-embedder-serial-batching-
        2026-08-04.md). A lone text still uses the cheaper single-sequence
        path (measured slower when routed through the batched machinery for
        just one sequence).

        Every ``llama_decode`` call here, single or batched, writes native
        lines like ``decode: cannot decode batches with this context
        (calling encode() instead)`` to raw stderr (#963 adversarial
        follow-up: #993 only wrapped the model LOAD). Deliberately NOT
        wrapped in ``dedup_native_stderr()`` at this level - measured live
        that grouping buys nothing here, since a typical call (one text, or
        one group) feeds the grouper exactly one line and flushes it raw at
        scope exit either way; the real repetition is ACROSS separate
        embed() calls (20 single-text RAG/memory calls in a row emit the
        SAME line 20 times), which only a scope spanning multiple calls can
        collapse. See ``_embedder_runner.py``'s dispatch loop, which wraps
        the whole isolated child's run of "embed" commands in one scope for
        exactly this reason."""
        with self._lock:
            if self._ctx is None:
                raise RuntimeError("embedder is closed")
            if not texts:
                return []
            token_lists = [self._tokenize(t) for t in texts]
            out: List[Optional[List[float]]] = [None] * len(texts)
            # A text that failed to tokenize at all (see _tokenize's own
            # docstring) needs no native call - short-circuit it exactly
            # like the pre-existing single-item path did, and keep it OUT of
            # the packing math below (it has no tokens to pack).
            pending_idx: List[int] = []
            pending_toks: List[List[int]] = []
            for i, toks in enumerate(token_lists):
                if toks:
                    pending_idx.append(i)
                    pending_toks.append(toks)
                else:
                    out[i] = [0.0] * self.dim
            groups = self._pack_groups(pending_toks) if pending_toks else []
            for group in groups:
                group_idx = [pending_idx[j] for j in group]
                group_toks = [pending_toks[j] for j in group]
                vecs = ([self._decode_single(group_toks[0])] if len(group_toks) == 1
                        else self._decode_batch(group_toks))
                for gi, v in zip(group_idx, vecs):
                    out[gi] = v
            return out

    def close(self) -> None:
        with self._lock:
            api = self._api
            if self._ctx is not None:
                try:
                    api.llama_free(self._ctx)
                finally:
                    self._ctx = None
            if self._model is not None:
                try:
                    api.llama_free_model(self._model)
                finally:
                    self._model = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class IsolatedEmbedder(VramSizingMixin):
    """Parent-side handle to a ``GGUFEmbedder`` running inside an isolated
    child process (see ``_embedder_runner.py``) - the same native-abort
    containment PR #606 gave the chat backend, applied to the embedder's
    separate, previously-unisolated GGUF/llama.cpp load path (honesty-audit
    follow-up, 2026-07-14).

    Preflight VRAM sizing (inherited from VramSizingMixin, shared with
    GgufBackend) runs HERE, before a child is even spawned - a load that can
    never fit still fails fast without paying a process-spawn cost, exactly
    mirroring GgufBackend/GgufWorker's split of responsibilities."""

    def __init__(self, model_path: str, *, n_gpu_layers: int = 99,
                 n_ctx: Optional[int] = None,
                 pooling_type: object = _POOLING_DEFAULT,
                 gpu_fallback_reason: Optional[str] = None) -> None:
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.effective_gpu_layers = None    # no auto-sizing for the embedder
        # None means "auto-size to the model's own native training window"
        # (see GGUFEmbedder.__init__ / _EMBED_CTX_CEILING) - but ONLY the
        # child can determine that, since only it ever loads the model. This
        # parent's own VRAM preflight (_check_vram, inherited from
        # VramSizingMixin - see _preflight_vram below) needs a concrete
        # number BEFORE any child spawns, so self.n_ctx starts at the
        # ceiling: a safe, never-under-estimating placeholder (the child's
        # real auto-sized value can never exceed it, by construction), later
        # overwritten with the real figure the child reports back at load
        # (_reload, mirroring how dim/declared_pooling already flow back).
        # The ORIGINAL request (None for auto, or an explicit override) is
        # kept separately so _reload always forwards the user's actual intent
        # to the child, not this placeholder.
        self._requested_n_ctx = n_ctx
        self.n_ctx = n_ctx if n_ctx is not None else _EMBED_CTX_CEILING
        self._pooling_type = pooling_type
        self.dim = 0
        # Reported by the child at load (see _reload): what the GGUF declares and
        # what is actually pooled with.
        self.declared_pooling: Optional[int] = None
        self.effective_pooling: Optional[int] = None
        # Set once a GPU-offloaded embed() crashes the worker and this embedder
        # falls back to CPU (see embed()'s crash-recovery branch), or SEEDED at
        # construction when automatic placement already chose CPU over a full
        # card (get_embedder -> _choose_embedder_gpu_layers) - None while on
        # GPU. Seeding it here also makes _reload() pass cpu_only to the child
        # from the very first spawn (n_gpu_layers=0 alone does not guarantee
        # zero GPU-backend involvement - see _reload's docstring). Exposed via
        # gpu_fallback_reason() so the GUI can show it, not just a log line.
        self.gpu_fallback_reason: Optional[str] = gpu_fallback_reason
        self._runner = None
        # Serializes embed() below. The worker protocol has NO request-id
        # correlation (one req_q/resp_q pair, see _embedder_runner.py's module
        # docstring: "one command processed at a time"), so two overlapping
        # RPCs are two threads blocked in the same resp_q.get() and the queue
        # hands each whichever response arrives first - a caller can silently
        # receive vectors belonging to a DIFFERENT text. This restores the
        # serialization the pre-#643 in-process GGUFEmbedder.embed() had via
        # its own RLock (still above), which the move to an isolated worker
        # dropped. Costs no throughput: the single child drains req_q FIFO, so
        # concurrent embeds were never actually parallel.
        self._rpc_lock = threading.RLock()
        # In-flight embed() calls, mirroring Engine.active_requests - checked by
        # http_server.py's unload/shutdown/restart paths (via active_requests()
        # below) before releasing this embedder, exactly like a pinned chat
        # Engine is skipped by unload_all_models/unload_one_model (AUDIT-CRIT-1).
        # Plain int, incremented/decremented lock-free in embed() below: the
        # same best-effort precision Engine's own _pin/_unpin already accept.
        self.active_requests = 0
        self._reload()

    def _preflight_vram(self) -> None:
        """Refuse a load that cannot fit BEFORE spawning a child. The
        multi-GPU split case needs its own per-device check distinct from
        VramSizingMixin._check_vram() (which budgets the split's COMBINED
        capacity - see _split_free_total_bytes - so one device's
        proportional share can be individually short while its aggregate
        passes) - see gpu_split_shortfall's docstring (discover.py) for the
        full rationale. The single-GPU case (the common one) used to have
        NO real check at all; _check_vram() closes that gap."""
        from localm.config import load_config
        from localm.discover import applied_split_device_count, gpu_split_shortfall
        cfg = load_config()
        # applied_split_device_count (loader truth), NOT split_device_count: on the
        # vulkan build list_gpus() is blind to the real split devices, so the
        # DETECTED count collapses a live 2-way split to < 2 and would wrongly send
        # us down the single-GPU _check_vram() branch on exactly the split that is
        # active (GPU-SPLIT-VKINDEX). On vulkan the gpu_split_shortfall() call below
        # SELF-SKIPS the per-device check (honest-unknown: it cannot name the right
        # card in torch's index space, so it logs the skip at INFO and the
        # subprocess-isolated loader backstops an oversized load) - so this branch
        # reads as "run the split-aware preflight" while being honest that the
        # per-device step is deferred there on that one backend.
        if applied_split_device_count(cfg) >= 2:
            try:
                file_size = Path(self.model_path).stat().st_size
            except OSError:
                file_size = 0
            if file_size:
                shortfall = gpu_split_shortfall(int(file_size * 1.2), cfg)
                if shortfall:
                    detail = "; ".join(
                        f"GPU {d['index']} needs ~{d['needed'] // 1024 ** 2} MB, "
                        f"{d['free'] // 1024 ** 2} MB free" for d in shortfall)
                    raise RuntimeError(
                        f"Not enough VRAM on the configured split device(s) "
                        f"to load the embedding model ({detail}).")
        else:
            # The common single-GPU case: the same preflight GgufBackend.load()
            # runs before every chat-model load.
            self._check_vram()

    def _reload(self) -> None:
        """(Re)run preflight and spawn a fresh child that loads the model.
        Used both by __init__ and by embed()'s auto-respawn after a crash -
        VRAM may have changed since the last load, so preflight re-runs too.

        cpu_only is passed once gpu_fallback_reason is set (a prior crash
        already forced n_gpu_layers to 0 - see embed()): n_gpu_layers=0 alone
        does not guarantee no GPU backend involvement (see
        _embedder_runner.py's cpu_only handling), so every reload from that
        point on re-asserts the guarantee, not just the first one."""
        self._preflight_vram()
        from ._embedder_runner import EmbedderRunner
        # Same parent-pins-worker-consumes contract as GgufBackend._load_native:
        # the effective split distribution (auto free-VRAM-proportional when
        # gpu_split_ratios is unset) is resolved HERE and carried into the
        # child, which must not probe for it (see resolve_auto_split_ratios).
        # Skipped when this embedder is CPU-bound anyway (cpu_only, or zero
        # GPU layers): no layer will be offloaded, so there is nothing to
        # distribute and the probe would be pure cost.
        from localm.discover import resolve_auto_split_ratios
        cpu_only = self.gpu_fallback_reason is not None
        auto_ratios = None
        if not cpu_only and self.n_gpu_layers != 0:
            # wait_for_inflight: off-loop here (loads run in an executor/CLI
            # thread); a heartbeat-probe collision joins instead of declining
            # auto into the equal fallback (same as GgufBackend._load_native).
            auto_ratios = resolve_auto_split_ratios(wait_for_inflight=True)
        params = dict(model_path=self.model_path, n_gpu_layers=self.n_gpu_layers,
                      n_ctx=self._requested_n_ctx, pooling_type=self._pooling_type,
                      cpu_only=cpu_only, gpu_split_ratios=auto_ratios)
        self._runner = EmbedderRunner()
        meta = self._runner.spawn_and_load(params)
        self.dim = meta["dim"]
        self.declared_pooling = meta.get("declared_pooling")
        self.effective_pooling = meta.get("effective_pooling")
        # The child is the only place that ever loads the model, so it is the
        # only place that can resolve "auto" against the model's real native
        # window - overwrite the preflight-time ceiling placeholder (see
        # __init__) with the actual figure it used.
        self.n_ctx = meta["n_ctx"]
        self._warn_if_mispooled()

    def _warn_if_mispooled(self) -> None:
        """Say so when this model is being pooled against its own training.

        A model declaring LAST-token pooling is a DECODER-based embedder
        (Qwen3-Embedding, verified 2026-07-15: qwen3.pooling_type=3). Its
        embeddings are trained to be read off the final token, so pooling it any
        other way degrades them - silently, because it still returns healthy,
        normalised, plausible vectors. That silence is the defect; the pooling
        choice itself stays the user's (embedding_pooling), we just stop hiding
        the consequence (AGENTS.md rule 5).

        With the untouched default now resolving LAST correctly for a
        LAST-declaring model (see _effective_pooling), this only fires when the
        user has EXPLICITLY set embedding_pooling to something else (mean/cls/
        none) for such a model - their call, but still worth surfacing, not
        silencing. Only LAST is worth a warning at all: bge declares CLS and is
        pooled MEAN by default, which measures fine (+0.29 related/unrelated
        margin) and is what every existing index was built with - warning about
        that would be noise, not signal, so it stays at debug level in
        GGUFEmbedder.
        """
        declared, effective = self.declared_pooling, self.effective_pooling
        if declared != _POOLING_LAST or effective == _POOLING_LAST:
            return
        logger.warning(
            "embedding model %s declares %s-token pooling (it is a decoder-based "
            "embedder) but embedding_pooling is explicitly set to %s, which "
            "degrades its embeddings. Unset embedding_pooling (or set it to "
            "'last'/'auto') to use the model's own pooling; existing RAG "
            "collections and memory vectors were built with %s and need "
            "re-indexing after the change.",
            Path(self.model_path).name, pooling_name(declared),
            pooling_name(effective), pooling_name(effective))

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed *texts* via the isolated worker, transparently respawning it
        first if a PRIOR call's crash left it dead - so one transient native
        fault does not permanently disable embeddings for the rest of the
        process's life (mirrors Engine.chat_stream's auto-reload after a
        chat-backend crash). A crash DURING this call is still raised to the
        caller (rule 5: never silently swallowed) - only the NEXT call
        recovers automatically, EXCEPT for the GPU-crash-once case below,
        which retries inline so a batch already in progress does not fall
        back to lexical-only for the rest of its run while a perfectly usable
        CPU path sits unused.

        Pinned via ``active_requests`` for the whole call (including a
        respawn), not just the RPC itself - a ``reset_embedder()`` arriving
        mid-respawn would free the very runner this call is about to use.

        Serialized on ``_rpc_lock`` (see __init__): the respawn check and the
        RPC together, so concurrent callers can neither swap responses on the
        correlation-free queue pair nor both respawn and orphan a worker. The
        pin is taken BEFORE the lock, so a caller merely queued behind another
        still counts as in-flight and keeps this embedder from being freed."""
        self.active_requests += 1
        try:
            with self._rpc_lock:
                if self._runner is None or not self._runner.is_alive():
                    logger.warning("embedder worker is not running; reloading %s",
                                   self.model_path)
                    self._reload()
                try:
                    return self._runner.embed(list(texts))
                except RuntimeError:
                    # Discard the worker ONLY if it is actually gone. The child's
                    # dispatch loop answers an ordinary embed failure with a
                    # clean ("error", msg) envelope and keeps serving - embedding
                    # is stateless per call, so one bad request is not supposed to
                    # take down a worker that can still work (see
                    # _embedder_runner's module docstring, and
                    # test_embed_dispatch_catches_ordinary_exceptions_without_
                    # crashing, which pins that the child stays alive). The parent
                    # sees the SAME RuntimeError for that as for a crash, so
                    # dropping the reference unconditionally orphaned a LIVE child:
                    # EmbedderRunner has no __del__ and GC never terminates an
                    # mp.Process, so it sat blocked on req_q.get() holding the
                    # model in VRAM while the next call spawned a SECOND worker
                    # beside it - one leak per clean embed error, invisible to
                    # close()/reset_embedder()/release_for_exit(), which all only
                    # reach the CURRENT runner.
                    if self._runner is not None and self._runner.is_alive():
                        logger.exception(
                            "embedding failed; the worker is healthy and will "
                            "serve the next call")
                        raise
                    logger.exception(
                        "embedding worker fault; it will reload on the next call")
                    runner, self._runner = self._runner, None
                    if runner is not None:
                        # Release its queues/handles. Safe and idempotent when the
                        # child is already dead (ModelRunner.shutdown's contract),
                        # and _wait's own timeout path has already done it.
                        runner.shutdown(grace=0)
                    # A crash while GPU-offloaded is a SYSTEMIC case this project
                    # has hit for real, not a hypothetical: an incomplete ROCm
                    # runtime (rocBLAS/hipBLASLt shipped without their Tensile
                    # kernel data - see setup_llama.py's _copy_blas_library_dirs,
                    # added for exactly this) makes every GPU embed call abort
                    # the native process the same way, every time. Retrying with
                    # the SAME config on the caller's next call would just crash
                    # again forever, silently disabling embeddings for the rest
                    # of the process's life. An already-provisioned runtime
                    # (installed before that fix, or not yet reprovisioned)
                    # stays broken regardless of the fix, so this bridge matters
                    # now, not just for some future fault. Fall back to CPU
                    # ONCE per embedder instance and retry THIS call inline - a
                    # small dedicated embedder (~25-90MB) is cheap enough on CPU
                    # that this is a genuine recovery, not a degraded stub. Say
                    # so loudly (rule 5): this is not a silent override of the
                    # user's n_gpu_layers setting (which still governs the chat
                    # model), it is a narrow, discoverable, in-memory-only
                    # response to a proven native failure.
                    if self.n_gpu_layers > 0 and self.gpu_fallback_reason is None:
                        self.gpu_fallback_reason = (
                            "GPU-offloaded embedding crashed natively (worker "
                            "exit); retrying on CPU. Chat-model GPU settings "
                            "are unaffected. Run 'localm setup-llama --force' "
                            "to reprovision the GPU runtime and restore "
                            "GPU-accelerated embedding.")
                        logger.warning("embedder %s: %s",
                                       Path(self.model_path).name,
                                       self.gpu_fallback_reason)
                        self.n_gpu_layers = 0
                        self._reload()
                        return self._runner.embed(list(texts))
                    raise
        finally:
            self.active_requests = max(0, self.active_requests - 1)

    def close(self) -> None:
        if self._runner is not None:
            self._runner.shutdown()
            self._runner = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
#  Process-wide singleton                                                      #
# --------------------------------------------------------------------------- #

# LOCK ORDER: engine._LOAD_LOCK (outer) -> _LOCK (inner), never the reverse.
# The engine's load path holds _LOAD_LOCK for a whole model load and calls
# loaded_path() here from its ctx sizing (#767), so acquiring _LOAD_LOCK -
# directly or by starting a load - while holding _LOCK is an ABBA deadlock
# against every concurrent model load. That exact inversion wedged the whole
# server permanently on 2026-08-18 (see get_embedder, the one site that
# needs both locks, which takes them in the legal order).
#
# HOLD TIME: microseconds normally, UP TO 300s during a load. get_embedder holds
# it across an entire IsolatedEmbedder construction (process spawn + native model
# load, _embedder_runner.LOAD_TIMEOUT_DEFAULT). So the cheap-looking readers below
# - loaded_dim, loaded_path, last_error, gpu_fallback_reason - are cheap in WORK
# and unbounded in WAITING, and each one's "Does NOT trigger a load" line means
# only the first of those. NONE OF THEM MAY BE CALLED FROM AN `async def` ROUTE
# HANDLER: on the event loop that wait is the whole server, every client and every
# route, not one request. That is not hypothetical - it shipped, and localm's own
# hang alarm caught it at 47s and climbing on POST /api/embedding/warmup.
# `scripts/check_event_loop_offload.py` now fails on a new instance.
_LOCK = threading.RLock()

# Two budgets for offloading one of those readers, e.g.
# `run_in_threadpool_bounded(loaded_dim, timeout=PEEK_TIMEOUT_S)`. Which one a
# caller wants depends on whether it has something useful to do with a timeout,
# and _LOCK's own shape is what makes that a real choice: it is either free (the
# read costs microseconds) or held by a load (it costs minutes), with nothing in
# between. So a budget expiring is not noise, it is the fact "a load is running
# right now" arriving by another route.
#
#   PEEK   for a caller that can ACT on that fact. POST /api/embedding/warmup
#          reads loaded_dim only to skip work it is about to do anyway, so a
#          timeout means "not already warm", which is the same answer the
#          blocking call would have given, minutes later.
#   READ   for a caller whose ANSWER is the value. Waiting is then correct and
#          the offload is the whole fix: one request waiting is fine, the event
#          loop waiting is the defect. Sized past a legitimate cold load
#          (_embedder_runner.LOAD_TIMEOUT_DEFAULT, 300s) plus the queue behind
#          engine._LOAD_LOCK it may sit in first, so it only ever fires for a
#          genuinely wedged lock - see run_in_threadpool_bounded's docstring on
#          picking a budget generously above the legitimate worst case.
PEEK_TIMEOUT_S = 2.0
READ_TIMEOUT_S = 660.0
_EMBEDDER: Optional[IsolatedEmbedder] = None
# A model that is present on disk but fails to LOAD (corrupt / OOM) is cached as
# failed so the expensive load is not retried on every call. A *missing* model is
# deliberately NOT cached: `localm setup-embeddings` can install one into a RUNNING
# server, and get_embedder re-checks the filesystem each call so it is picked up
# without a restart. (Regression fixed: the old single `_TRIED` latch cached the
# "no model" result for the whole process lifetime, so embeddings stayed dead -
# 422 - until a restart even right after `setup-embeddings`.)
#
# The ``embedding_model`` config value (stripped) that was current when a LOAD
# last failed, or None when nothing is latched. NOT a bare bool (that was the
# bug: a bool has no identity, so it could not tell "still the same broken
# spec" from "the user already fixed this" - once ANY load failed,
# get_embedder() stayed permanently dead, even after embedding_model was
# changed to a perfectly good model, until an explicit reset_embedder(). See
# get_embedder()'s own use of this for how a changed spec clears the stale
# latch, and _set_resolve_outcome for the matching resolve-side guard.
_LOAD_FAILED_SPEC: Optional[str] = None
_TRIED_DOWNLOAD = False          # one-time auto-download attempt (only net_mode=allow)
# Why the last LOAD or RESOLVE failed (for the GUI picker) - see last_error(),
# _record_resolve_failure/_record_resolve_success above resolve_embedding_model_path.
_LAST_ERROR: Optional[str] = None
# Dedup key for _record_resolve_failure's WARNING-once-then-DEBUG: the reason last
# logged at WARNING, so an unchanged misconfiguration does not re-warn on every
# embed_texts() call (resolve_embedding_model_path re-runs on every one while no
# embedder is loaded - see the _EMBEDDER caching comment above).
_LAST_RESOLVE_WARNED: Optional[str] = None


def _explicit_embedder_gpu_layers(cfg: dict) -> Optional[int]:
    """The user's EXPLICIT GPU-layer choice for the embedder, or None when the
    automatic placement below should decide.

    ``embedding_gpu_layers`` (the dedicated key) wins outright. Failing that,
    a global ``n_gpu_layers`` moved off its "everything" default (99) is
    inherited exactly as get_embedder() always did - someone who set -g 24 or
    -g 0 globally expressed an offload preference this load keeps honoring
    (never silently override an explicit selection). A bool sneaking in via a
    hand-edited config is not an int choice; malformed values fall through to
    auto rather than crashing the load."""
    raw = cfg.get("embedding_gpu_layers")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return int(raw)
    try:
        chat = int(cfg.get("n_gpu_layers", 99))
    except (TypeError, ValueError):
        return None
    if chat != 99:
        return chat
    return None


def _choose_embedder_gpu_layers(path: str, cfg: dict, *,
                                read_free=None) -> "tuple[int, Optional[str]]":
    """Pick the embedder's ``n_gpu_layers``: ``(layers, reason)`` where
    *reason* is a user-facing explanation only when automatic placement chose
    CPU over a GPU that cannot hold the model.

    Why this exists: the pre-load eviction (``_maybe_swap_for_embedder``)
    cannot free room when the chat engine is pinned by an in-flight request -
    which is the NORMAL case, since per-turn memory recall fires during one.
    The embedder then force-loaded all layers onto a full card anyway, and
    Windows/WDDM paged the overcommit to system RAM, thrashing the resident
    chat model (measured live 2026-07-18: 34 -> 5.0 tok/s; the same load
    placed on CPU ran at 21 and left chat's VRAM untouched). Degrading THIS
    load to CPU is the right trade: embeds get slower, the chat model the
    user is actively talking to does not.

    An explicit user choice (see _explicit_embedder_gpu_layers) is honored
    verbatim, tight VRAM or not. Unmeasurable free VRAM (including a probe
    that did not complete fresh this call - see the freshness note below) or
    an unreadable model file keep the historical full-offload attempt - the
    load preflight still warns there, so nothing is hidden, and a guess of
    CPU on a healthy box would silently slow every embed for no reason.

    Freshness, not scope, is what read_free()'s default checks (AGENTS.md
    rule 5): a GPU_PROBE_TIMEOUT/BUSY reading is a frozen last-known-good
    value, not this call's own measurement, so it is treated the same as
    "unmeasurable" above. Deliberately does NOT gate on free_scope: a
    PROCESS-scoped reading over-states free the same way
    discover.gpu_split_shortfall's own docstring explains, so the ONLY
    place that matters here is the branch below that already fires when
    the (possibly inflated) free reads as INSUFFICIENT - if even an
    over-stated free comes up short, the real free is shorter still, so
    that verdict only ever under-states the problem, never invents one.
    Gating this on scope instead would make free read as permanently
    unmeasurable on every Windows + AMD ROCm/HIP box, silently reverting to
    the risky full-GPU-offload default this function exists to avoid on
    exactly that platform."""
    explicit = _explicit_embedder_gpu_layers(cfg)
    if explicit is not None:
        return explicit, None
    try:
        size = Path(path).stat().st_size
    except OSError:
        size = 0
    if not size:
        return 99, None
    if read_free is None:
        def read_free() -> Optional[int]:
            from localm.discover import GPU_PROBE_OK, vram_capacity
            info, status = vram_capacity(return_status=True)
            return info.get("free") if status == GPU_PROBE_OK else None
    try:
        free = read_free()
    except Exception:
        free = None
    if free is None:
        return 99, None
    # Same loaded-footprint approximation the swap decision and the split
    # preflight already use (weights + ~20% KV/compute slop).
    estimate = int(size * 1.2)
    if free >= estimate:
        return 99, None
    reason = (f"insufficient free VRAM for the embedding model "
              f"(~{estimate // 1024 ** 2} MB needed, "
              f"{int(free) // 1024 ** 2} MB free); the embedder runs on CPU "
              f"so the loaded chat model is not slowed by VRAM "
              f"oversubscription - set embedding_gpu_layers to override")
    return 0, reason


def _maybe_swap_for_embedder(path: str, n_gpu_layers: int) -> None:
    """Before the embedder's native load, evict a resident chat model when it
    would not otherwise fit - the SAME VRAM-aware swap the image/music/video
    plugins run before their own model load (see ``localm.vram.decide_media_swap``),
    generalized here via ``decide_embedder_swap``/``evict_chat_for_embedder``.

    This is complementary to, not redundant with, IsolatedEmbedder's own
    ``_preflight_vram()`` (below): that check only REFUSES a load that will
    not fit (fail fast, no child spawned). It never makes room. This function
    is what actually frees VRAM by evicting the resident chat model first -
    the same swap-before-load the image/music/video plugins already do -
    so the preflight then has a real chance of succeeding instead of just
    failing faster.

    A CPU-only embedder load (``n_gpu_layers <= 0``) never contends for VRAM,
    so it is skipped. Best-effort: any failure to read the file size or decide
    the swap leaves the load path exactly as before this check existed (never
    blocks a legitimate load - the swap is an optimization, not a
    correctness requirement of the load itself)."""
    if n_gpu_layers <= 0:
        return
    try:
        file_size = Path(path).stat().st_size
    except OSError:
        return
    if not file_size:
        return
    from localm.config import load_config
    from localm.vram import decide_embedder_swap, evict_chat_for_embedder, resolve_swap_policy
    cfg = load_config()
    policy = resolve_swap_policy({}, cfg)
    # Same loaded-footprint approximation IsolatedEmbedder._preflight_vram()
    # already uses for its own gpu_split_shortfall check (weights + ~20%
    # KV-cache/compute slop).
    estimate = int(file_size * 1.2)
    if not decide_embedder_swap(estimate, policy=policy):
        return
    evict_chat_for_embedder()


def _emit_stage(on_progress: Optional[Callable[[str], None]], msg: str) -> None:
    """Best-effort coarse stage announcement for get_embedder()'s caller (ADR-0004
    Unit B: the "warm up the embedder" job). Mirrors managed_comfy_provision._emit
    - a raising sink must never abort a load. Parent-side only: this never touches
    the isolated embedder child's own load/embed IPC protocol."""
    if on_progress is None:
        return
    try:
        on_progress(msg)
    except Exception:
        logger.debug("embedder progress sink raised (ignored)", exc_info=True)


def get_embedder(*, on_progress: Optional[Callable[[str], None]] = None
                  ) -> Optional[IsolatedEmbedder]:
    """The shared embedder, loading the configured model on first use. Returns None
    when no embedding model can be resolved - callers then fall back to lexical
    retrieval. A missing model is re-checked on every call (so a mid-session
    ``localm setup-embeddings`` is picked up without a restart); only a genuine
    load FAILURE is cached. Loading holds the engine's process-global load lock so
    it cannot race a chat-model load onto the GPU.

    *on_progress*, if given, receives coarse human-readable stage announcements
    (resolving/downloading/evicting/loading/ready) - purely additive, every
    existing caller is unaffected. Added for the explicit "warm up the embedder"
    job (ADR-0004 Unit B) so a cold server's first load - up to two 300s timeout
    windows (VRAM eviction wait + child spawn/native-load) - is not silent.

    The pre-load swap check (``_maybe_swap_for_embedder``) deliberately runs
    OUTSIDE ``_LOCK`` (double-checked locking below), not inside a single held
    lock the way the rest of this function is structured. It can evict a
    resident chat model via ``vram.evict_chat_for_embedder``, which submits
    ``http_server.unload_all_models()`` onto the SERVER'S event-loop thread and
    blocks THIS thread waiting for it - and that coroutine calls
    ``loaded_dim()``, which itself needs ``_LOCK``. Since ``_LOCK`` is a plain
    ``threading.RLock`` (thread-owned, not reentrant across threads), holding
    it here while blocking on that coroutine is a genuine cross-thread
    deadlock: this thread waits on the coroutine, the coroutine (on the event
    loop) waits on this thread's lock. Confirmed via live reproduction
    (2026-07-14 review) - the event loop stayed blocked for the full
    eviction timeout before either side could proceed."""
    global _EMBEDDER, _LOAD_FAILED_SPEC, _TRIED_DOWNLOAD, _LAST_ERROR
    cur_spec = None   # lazily computed below, only once _EMBEDDER is confirmed unset -
                       # the hot "already loaded" path stays a free dict lookup, no I/O.
    with _LOCK:
        if _EMBEDDER is not None:
            return _EMBEDDER
        if _LOAD_FAILED_SPEC is not None:
            # The identity a latched load failure is compared against, so a
            # caller that changed embedding_model since the failure was
            # recorded is recognised as asking about something new, not stuck
            # behind a stale bare-bool "something once failed" flag (the
            # permanent-breakage bug this replaces - measured live:
            # get_embedder() returned None forever, never even calling
            # resolve_embedding_model_path() again, regardless of later
            # config changes, until an explicit reset_embedder()). Reused
            # below (not re-derived) for the second, re-check latch test.
            cur_spec = _current_spec()
            if _LOAD_FAILED_SPEC == cur_spec:
                return None
            # The config has moved on since this was latched - it no longer
            # describes what we are about to try, so it must not keep
            # blocking a real attempt at the (different) spec configured now.
            _LOAD_FAILED_SPEC = None
            _LAST_ERROR = None
        _emit_stage(on_progress, "Resolving the embedding model...")
        # Cheap filesystem-only re-check every call (NO download): finds a model a
        # user just installed into this running server.
        path = resolve_embedding_model_path(allow_download=False)
        if not path and not _TRIED_DOWNLOAD:
            # First miss: one auto-download attempt, gated by net policy inside
            # (only actually fetches under net_mode=allow). Latched so a batch of
            # embed calls does not re-attempt the download on every chunk.
            _TRIED_DOWNLOAD = True
            _emit_stage(on_progress, "Not found locally - attempting a download...")
            path = resolve_embedding_model_path()
        if not path:
            _emit_stage(on_progress, "No embedding model is configured or available.")
            return None
        from localm.config import load_config
        _cfg = load_config()
        pooling = resolve_pooling_setting(_cfg.get("embedding_pooling"))
        # GPU INTENT for the pre-load eviction check below: an explicit user
        # choice, else full offload. The FINAL placement is chosen after the
        # swap runs, against post-eviction free VRAM - so a successful
        # eviction still yields a GPU load, and only a card that stays full
        # (chat pinned by the very request that triggered this embed) demotes
        # the embedder to CPU (see _choose_embedder_gpu_layers).
        explicit = _explicit_embedder_gpu_layers(_cfg)
        intent_ngl = explicit if explicit is not None else 99

    # OUTSIDE _LOCK (see docstring): safe to block here on the cross-thread
    # eviction round trip. Another thread may concurrently reach this same
    # window and run its own swap check too - harmless, since
    # evict_chat_for_embedder()/decide_embedder_swap() are idempotent (a
    # second call simply finds nothing left to evict, or enough VRAM already
    # free) and the actual load below re-checks the singleton state.
    _emit_stage(on_progress, "Checking VRAM (may evict a resident chat model)...")
    _maybe_swap_for_embedder(path, intent_ngl)
    ngl, placement_reason = _choose_embedder_gpu_layers(path, _cfg)
    if placement_reason is not None:
        logger.warning("embedder placement: %s", placement_reason)

    # LOCK ORDER (fixes the 2026-08-18 whole-server deadlock): the engine's
    # process-global load lock is acquired FIRST, strictly OUTSIDE _LOCK.
    # Engine.load() holds _LOAD_LOCK across an entire model load, and its ctx
    # sizing calls loaded_path() - which takes _LOCK - via
    # _sizing.embedder_ctx_reservation_bytes (#767). This block used to take
    # _LOAD_LOCK INSIDE _LOCK (#313), the opposite order, so a chat-model
    # load racing a first embed deadlocked both threads permanently, and
    # every later _LOCK caller (status probes, the GUI's models/stats
    # fetches) wedged behind them until the whole server was unusable. The
    # only permitted nesting is _LOAD_LOCK (outer) -> _LOCK (inner) - see
    # both lock definitions. Side benefit: waiting here for a running chat
    # load no longer holds _LOCK, so status probes stay responsive while
    # this thread queues for its turn to load.
    from localm.inference.engine import _LOAD_LOCK
    with _LOAD_LOCK:
        with _LOCK:
            # Re-check: another thread may have completed (or failed) the load
            # while this thread was outside the lock running the swap check -
            # including latching a NEW failure this thread has not seen yet, for
            # whatever spec is current now (re-derived, not the possibly-unset
            # local above: another thread could have raced a config change too).
            if _EMBEDDER is not None:
                return _EMBEDDER
            if _LOAD_FAILED_SPEC is not None:
                if cur_spec is None:
                    cur_spec = _current_spec()
                if _LOAD_FAILED_SPEC == cur_spec:
                    return None
                _LOAD_FAILED_SPEC = None
                _LAST_ERROR = None
            try:
                # The stated bound is the deadline this stage ACTUALLY runs under:
                # _embedder_runner.LOAD_TIMEOUT_DEFAULT (300s), which applies because
                # _reload's spawn_and_load() call passes no override. It said "up to a
                # minute", understating its own ceiling 5x, so a user watching a slow
                # first load had every reason to think it had hung. The OTHER 300s
                # window this function can spend (vram.evict_chat_for_embedder) is
                # already past by here, which is why this says five and not ten.
                # test_get_embedder_progress_states_the_real_load_bound pins the two
                # together, so changing the constant without the copy fails.
                _emit_stage(on_progress,
                           "Loading into memory (this can take up to five minutes)...")
                _EMBEDDER = IsolatedEmbedder(
                    path, n_gpu_layers=ngl, pooling_type=pooling,
                    gpu_fallback_reason=placement_reason)
                # getattr, not attribute access: this status line sits INSIDE the
                # try below, so anything it raises would be caught as a LOAD failure
                # and silently drop embeddings to lexical-only for the rest of the
                # process. A line that merely describes the load must never be able
                # to fail it.
                logger.info("embedding model ready: %s (dim=%d, pooling=%s)", path,
                            _EMBEDDER.dim,
                            pooling_name(getattr(_EMBEDDER, "effective_pooling", None)))
                _LAST_ERROR = None
                _emit_stage(on_progress, f"Ready ({_EMBEDDER.dim}-dim).")
                return _EMBEDDER
            except Exception as e:
                # Latch by SPEC, not a bare bool: what just failed to load is
                # whatever embedding_model names right now, and that identity is
                # what a later call must compare against to tell "still this
                # exact broken model" from "the user already changed it" - see
                # this global's own docstring and _set_resolve_outcome.
                _LOAD_FAILED_SPEC = cur_spec if cur_spec is not None else _current_spec()
                _LAST_ERROR = str(e)
                logger.warning("could not load embedding model %s (%s); lexical-only",
                               path, e)
                _emit_stage(on_progress, f"Load failed: {e}")
                return None


def embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    """Embed *texts* with the shared embedder, or None when unavailable."""
    emb = get_embedder()
    if emb is None:
        return None
    return emb.embed(list(texts))


def loaded_dim() -> Optional[int]:
    """Dimension of the currently-loaded embedder, or None if none is loaded.
    Does NOT trigger a load - safe for a cheap status probe (GUI picker)."""
    with _LOCK:
        return _EMBEDDER.dim if _EMBEDDER is not None else None


def gpu_fallback_reason() -> Optional[str]:
    """Why the currently-loaded embedder dropped from GPU to CPU after a native
    crash (see IsolatedEmbedder.embed), or None when it hasn't (still on GPU,
    or configured for CPU from the start). Does NOT trigger a load - for the
    GUI picker to surface this instead of leaving it as a log-only fact.

    Path-scrubbed for the same reason as ``last_error`` below."""
    with _LOCK:
        raw = _EMBEDDER.gpu_fallback_reason if _EMBEDDER is not None else None
    return pathscrub.scrub_paths(raw) if raw else raw


def loaded_path() -> Optional[str]:
    """Filesystem path of the currently-loaded embedder, or None if none is
    loaded. Does NOT trigger a load. Lets a caller outside this module (the
    Models page, the targeted unload route) recognise a registered model
    entry as "this is the resident embedder" - it never appears in
    ``http_server._engines``, so that is otherwise invisible - by comparing
    against the entry's own resolved path. Pairs with ``loaded_dim()``."""
    with _LOCK:
        return _EMBEDDER.model_path if _EMBEDDER is not None else None


def active_requests() -> int:
    """In-flight embed() calls on the currently-loaded embedder, or 0 if none is
    loaded. Mirrors ``Engine.active_requests`` - checked by http_server.py's
    unload-all/targeted-unload/shutdown/restart paths before releasing this
    embedder, exactly like a pinned chat Engine is skipped by
    unload_all_models/unload_one_model (AUDIT-CRIT-1): without this, any of
    those paths could free the embedder - and the isolated worker process a
    request is waiting on - out from under an in-flight embed() call."""
    with _LOCK:
        return _EMBEDDER.active_requests if _EMBEDDER is not None else 0


def last_error() -> Optional[str]:
    """Why the last embedding-model LOAD or RESOLVE failed (e.g. the model is not
    an embedding model, or the configured spec resolves to a HuggingFace-format
    directory rather than a GGUF file), or None. For the GUI picker to tell the
    user what went wrong - resolve_embedding_model_path records a failure here on
    every call, not only when a load is actually attempted (see
    _record_resolve_failure), so GET /api/rag/embedding can explain a broken
    embedding_model without anyone having to trigger an embed first.

    PATH-SCRUBBED on the way out. The stored message is the raw exception text,
    which for a load failure reads "failed to load embedding model:
    <absolute path>" and crosses the isolated-child boundary intact - so it
    carries the data dir, and with it the OS account name. Every caller of this
    is an HTTP response (``GET /api/rag/embedding``, the 422 in routes/chat.py,
    the rag job log), i.e. potentially a rag-scoped or non-owner reader, and
    ``rag`` is deliberately NOT a privileged scope.

    Scrubbed on READ, not at the assignment: the WARNING logged at the failure
    site keeps the full path, so the local operator diagnosing this in the debug
    log loses nothing. Per AGENTS.md rule 5 the CAUSE is preserved verbatim -
    only the directory prefix is replaced - because muting the reason would
    leave the user with a broken embedder and no way to find out why."""
    with _LOCK:
        raw = _LAST_ERROR
    return pathscrub.scrub_paths(raw) if raw else raw


def reset_embedder(*, force: bool = True) -> bool:
    """Drop the cached embedder and its negative caches (tests / a model
    change). Returns True if an embedder was actually cleared, False if none
    was loaded or (``force=False``) one was loaded but pinned.

    ``force=False`` checks ``active_requests() == 0`` and clears the embedder
    in the SAME ``_LOCK`` acquisition, atomically. Every caller that means to
    skip a BUSY embedder (unload_all_models, _unload_embedder_if_matches,
    switch_engine's own VRAM-shortfall eviction) used to do that check via a
    SEPARATE, unlocked ``active_requests()`` call before calling this
    function unconditionally - a real TOCTOU window, since
    ``IsolatedEmbedder.embed()`` pins ``active_requests`` without taking
    ``_LOCK``: a concurrent embed() could start in that window and have its
    worker torn out from under it by the ``close()`` below. Folding the check
    into this one locked critical section (no ``await`` between the read and
    the close - both run synchronously inside a single executor hop) closes
    that window instead of narrowing it per call site.

    ``force=True`` (the default) keeps the original unconditional-clear
    behavior, for callers that must tear down regardless of a pin (e.g. an
    explicit model-selection change) and for the many existing bare
    ``reset_embedder()`` call sites (tests, the RAG embedding-model-setup
    route) that predate the pin-aware option and rightly don't need it.
    ``release_for_exit()`` deliberately does NOT go through this function at
    all - see its own docstring (taking ``_LOCK`` there risks hanging a
    stop/restart behind an in-progress load).

    A pinned (``force=False``, busy) embedder is a full no-op, including the
    negative caches: nothing actually changed, so nothing is cleared.
    Otherwise ``_LOAD_FAILED_SPEC``/``_TRIED_DOWNLOAD``/``_LAST_ERROR``/
    ``_LAST_RESOLVE_WARNED`` are always cleared alongside ``_EMBEDDER`` - even
    when no embedder was loaded at all (only a cached load FAILURE), matching
    the original unconditional behavior: a caller resetting a failed-load state
    expects the next ``get_embedder()`` to retry fresh, not keep returning the
    stale cached failure (or a suppressed re-warn of it)."""
    global _EMBEDDER, _LOAD_FAILED_SPEC, _TRIED_DOWNLOAD, _LAST_ERROR, _LAST_RESOLVE_WARNED
    with _LOCK:
        if not force and _EMBEDDER is not None and _EMBEDDER.active_requests > 0:
            return False
        cleared = _EMBEDDER is not None
        if cleared:
            _EMBEDDER.close()
        _EMBEDDER = None
        _LOAD_FAILED_SPEC = None
        _TRIED_DOWNLOAD = False
        _LAST_ERROR = None
        _LAST_RESOLVE_WARNED = None
        return cleared


def release_for_exit() -> bool:
    """Release the isolated embedder worker for a caller that is about to
    ``os._exit()`` / ``os.execv()``. Returns True if a worker was released.

    Both of those bypass ``atexit`` - and multiprocessing's daemon-child
    reclamation IS an atexit hook (``multiprocessing.util._exit_function``) - so
    a worker left running does NOT die with the parent: it survives as an orphan
    still holding its model in VRAM until killed by hand, and the restarted
    server spawns a second one next to it. Verified live (2026-07-15): a daemon
    child is reclaimed on a normal interpreter exit, but survives BOTH os._exit
    and os.execv.

    This is the WHOLE exit-path decision on purpose, because every other way to
    make it takes ``_LOCK``: ``active_requests()`` does, and ``reset_embedder()``
    does. ``get_embedder()`` holds ``_LOCK`` for the FULL duration of an
    embedding-model load (up to its load timeout), so a stop or restart issued
    during a load would block on the guard itself and hang the very action the
    user asked for - never reaching the teardown, and leaving exactly the orphan
    this function exists to prevent. Nothing here touches ``_LOCK``: the
    singleton is snapshotted directly, which is a best-effort read the exiting
    caller can afford (a hard exit racing a load that has not yet published its
    embedder is inherent to any hard exit).

    A BUSY worker (a request is mid-``embed()``) is terminated WITHOUT waiting:
    the pinned request cannot be answered either way once the process exits, so
    there is nothing left to protect, and the polite "shutdown" command would
    only queue behind the in-flight embed and stall the exit for the grace
    period. An IDLE worker gets the polite command first, so the child frees the
    model cleanly before it goes.

    The module singleton is deliberately NOT cleared (that is reset_embedder()'s
    job, and it needs ``_LOCK``): the process is going away regardless."""
    emb = _EMBEDDER
    if emb is None:
        return False
    runner = getattr(emb, "_runner", None)
    if runner is None:
        return False
    busy = getattr(emb, "active_requests", 0) > 0
    runner.shutdown(grace=0 if busy else 5.0)
    return True


# A daemon worker already dies with the parent, but tear it down explicitly at
# interpreter exit so its queue feeder threads never delay shutdown - mirrors
# voice.py's atexit registration for the identical process-wide-singleton shape
# (the embedder, unlike GgufBackend/ModelRunner, is not torn down by an owning
# caller on every unload - it outlives any single request).
atexit.register(reset_embedder)
