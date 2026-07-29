# SPDX-License-Identifier: AGPL-3.0-or-later
"""GGUF file-format helpers: split-part detection, the gguf magic check,
safe model filenames, and SHA256 hashing. Leaf utilities, no registry/pull deps."""

import localm.model_manager as _mm  # read package-patchable names at call time

import hashlib
import os
import re
import struct
import threading
from pathlib import Path
from typing import Callable
from typing import List
from typing import Optional
from rich.progress import BarColumn
from rich.progress import DownloadColumn
from rich.progress import Progress
from rich.progress import TextColumn
from rich.progress import TimeRemainingColumn
from ..debuglog import logger
from ._shared import console




# ------------------------------------------------------------------ #
#  Split GGUF (multi-part *-00001-of-00003.gguf files)                 #
# ------------------------------------------------------------------ #

# llama.cpp split naming convention: <stem>-00001-of-00003.gguf
_SPLIT_GGUF_RE = re.compile(
    r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE
)




def split_gguf_parts(filename: str) -> Optional[List[str]]:
    """
    If *filename* follows the llama.cpp split convention
    (``model-00001-of-00003.gguf``), return the full ordered list of part
    filenames. Returns None for regular single-file GGUFs.
    """
    m = _SPLIT_GGUF_RE.match(Path(filename).name)
    if not m:
        return None
    total = int(m.group("total"))
    if total < 2:
        return None
    stem = m.group("stem")
    return [f"{stem}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]




def first_split_part(filename: str) -> str:
    """Return the first-part filename for a split GGUF (llama.cpp wants this one)."""
    parts = split_gguf_parts(filename)
    return parts[0] if parts else filename




def missing_split_parts(first_part: Path) -> List[Path]:
    """
    Given the path of any part of a split GGUF, return sibling part paths
    that are missing on disk. Empty list means all parts present (or the
    file is not a split GGUF at all).
    """
    parts = split_gguf_parts(first_part.name)
    if not parts:
        return []
    return [first_part.parent / p for p in parts
            if not (first_part.parent / p).is_file()]




def _sha256_file(
    path: Path,
    progress: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Return the hex SHA256 digest of a file.

    When *progress* is given it is called ``progress(bytes_done, total_bytes)``
    after each block so a caller can drive a progress bar; *total_bytes* is the
    file size (0 if it cannot be stat'd). The callback must not raise."""
    h = hashlib.sha256()
    total = 0
    if progress is not None:
        try:
            total = path.stat().st_size
        except OSError:
            total = 0
    done = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
            if progress is not None:
                done += len(block)
                progress(done, total)
    return h.hexdigest()




def _sha256_file_bytes(data: bytes) -> str:
    """Return the hex SHA256 digest of an in-memory byte string."""
    return hashlib.sha256(data).hexdigest()




def _safe_models_filename(filename: str, base_dir: Optional[Path] = None) -> Optional[str]:
    """Return a single-component filename confined to *base_dir* (``MODELS_DIR``
    by default).

    A model download must never write outside its destination folder.
    ``filename`` is derived from untrusted input (a URL path or an
    ``owner/repo:file`` spec), so a value like ``../../evil.gguf`` or
    ``sub/dir/evil.gguf`` must be rejected rather than used as a destination.
    Returns the bare filename when it is a single, non-traversing path
    component, else ``None`` (GAP-CLI-2). *base_dir* lets a caller routing a
    download to a non-default destination (e.g. a ComfyUI models subfolder)
    validate against the REAL destination instead of always against
    ``MODELS_DIR``.
    """
    base_dir = base_dir if base_dir is not None else _mm.MODELS_DIR
    if not filename:
        return None
    # An embedded NUL is not a legal filename on any OS and makes Path.resolve() /
    # os.stat raise ValueError, which would ESCAPE the OSError-only guard below and
    # crash the caller (an uncaught traceback) instead of being rejected. Fail
    # closed: unsafe input must return None, exactly like a '/'/'..'/drive spec.
    if "\x00" in filename:
        return None
    # Reject anything that is not a single path component (no separators, no
    # drive/absolute prefixes, no '.'/'..').
    name = Path(filename).name
    if name != filename or name in ("", ".", ".."):
        return None
    if "/" in filename or "\\" in filename or os.sep in filename:
        return None
    try:
        dest = (base_dir / name).resolve()
        if dest.parent != base_dir.resolve():
            return None
    except (OSError, ValueError):
        # ValueError also covers a null/odd path that slipped past the check above
        # (belt and suspenders); either way, an unresolvable name is not safe.
        return None
    return name




# Files at or below this size hash inline (silently); larger ones get the
# off-main-thread progress bar - the threading/UI overhead isn't worth it for
# small files. ~0.5 GB, matching the prior notice threshold.
_HASH_PROGRESS_MIN_BYTES = 512 * 1024 * 1024




def _hash_with_progress(path: Path) -> Optional[str]:
    """SHA256 a model file, showing live progress for large files.

    For files larger than ``_HASH_PROGRESS_MIN_BYTES`` the hash runs in a
    background worker thread so the main thread stays responsive and can render a
    progress bar (a one-time cost stored in the registry for duplicate
    detection); smaller files hash inline without any UI. Returns None for
    directories (HF models are identified by path only)."""
    if not path.is_file():
        return None
    try:
        size = path.stat().st_size
    except OSError:
        size = 0

    if size <= _mm._HASH_PROGRESS_MIN_BYTES:
        return _mm._sha256_file(path)

    from rich.progress import (SpinnerColumn)

    result: dict = {}

    def _worker(report):
        try:
            result["digest"] = _mm._sha256_file(path, progress=report)
        except Exception as exc:                      # surface on the main thread
            result["error"] = exc

    with Progress(
        SpinnerColumn(),
        TextColumn("[dim]Hashing {task.description} for duplicate detection[/dim]"),
        BarColumn(),
        DownloadColumn(),
        TimeRemainingColumn(),
        transient=True,
        console=console,
    ) as prog:
        task = prog.add_task(path.name, total=size)
        worker = threading.Thread(
            target=_worker,
            args=(lambda done, total: prog.update(task, completed=done),),
            daemon=True,
        )
        worker.start()
        worker.join()

    if "error" in result:
        raise result["error"]
    return result.get("digest")




# Conservative size floor for a plausible GGUF. The fixed header alone (magic +
# version + tensor count + KV count) is 24 bytes, and every real model carries a
# metadata KV block (general.architecture, tokenizer, ...) plus tensor infos well
# past 1 KiB before the first weight byte; even the tiniest test GGUFs are tens of
# KB. Kept deliberately low so no legitimate model can ever be rejected.
_GGUF_MIN_BYTES = 1024


def _has_gguf_magic(path: Path) -> bool:
    """True when *path* begins with the GGUF magic ``b"GGUF"`` and is at least
    ``_GGUF_MIN_BYTES`` long.

    Auto-registration (sync_models_dir) keys on the ``.gguf`` extension alone, so
    a foreign file renamed ``.gguf``, a 0-byte placeholder, or a partial copy that
    never received its header would otherwise be registered and then crash a later
    load - in the worst case wedging the app if it became the active model (R45,
    "copying a file into models/ broke the whole app"). A real GGUF always starts
    with this 4-byte magic; an unreadable file is treated as not-a-GGUF and
    skipped. The size floor additionally rejects a header-only truncated copy or
    placeholder that got just the magic, which would pass the magic check and then
    fail a later load with an opaque ggml error. (A mid-copy of a *valid* GGUF that
    already passed the floor stays a best-effort gap.)"""
    floor = _mm._GGUF_MIN_BYTES
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != b"GGUF":
                return False
        size = path.stat().st_size
    except OSError:
        return False
    if size < floor:
        logger.debug(
            "skipping %s: GGUF magic but only %d bytes (< %d) - truncated or "
            "placeholder, not a usable model",
            path.name, size, floor,
        )
        return False
    return True




def _gguf_first_parts(d: Path, max_depth: int = 3) -> List[Path]:
    """First-part GGUF files inside *d*, scanning up to *max_depth* levels deep.

    *max_depth* counts the filename as level 1 (depth 1 = files directly in *d*,
    depth 3 = up to two subfolders down), so batch imports of models organised in
    subdirectories are picked up. Split GGUFs (``model-00001-of-00003.gguf``)
    contribute only their first part - llama.cpp finds the siblings on its own;
    loose single-file GGUFs contribute themselves. Mirrors the first-part filter
    in sync_models_dir.
    """
    out: List[Path] = []
    for f in sorted(d.rglob("*.gguf")):
        try:
            if not f.is_file():
                continue
            depth = len(f.relative_to(d).parts)
        except (OSError, ValueError):
            continue
        if depth > max_depth:
            continue
        parts = split_gguf_parts(f.name)
        if parts and f.name != parts[0]:
            continue   # non-first split part -> registered via its first part
        out.append(f)
    return out




# ------------------------------------------------------------------ #
#  GGUF embedding-model detection (hard metadata, not a filename guess) #
# ------------------------------------------------------------------ #

# llama.cpp's own LLM_ARCH_NAMES entries for encoder/embedding-only
# architectures (verified against src/llama-arch.cpp) - a GGUF whose
# general.architecture is one of these is unambiguously an embedding/encoder
# model, never a causal-chat LLM.
_GGUF_EMBEDDING_ARCHITECTURES = frozenset({
    "bert", "modern-bert", "nomic-bert", "nomic-bert-moe", "neo-bert",
    "jina-bert-v2", "jina-bert-v3", "eurobert", "gemma-embedding",
    "t5encoder", "pangu-embedded",
})

# GGUF metadata value types (ggml gguf.h `enum gguf_type`). STRING(8) and
# ARRAY(9) are variable-length and handled specially; every other type here
# maps to its fixed byte width.
_GGUF_FIXED_TYPE_SIZES = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
}
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9

# struct formats for the same fixed-width types, keyed identically to
# _GGUF_FIXED_TYPE_SIZES so the two tables cannot drift apart.
_GGUF_SCALAR_FORMATS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}

# The "<architecture>."-prefixed keys describing the attention shape, i.e. what
# the KV cache actually costs per token. Stored as suffixes because llama.cpp
# namespaces them under general.architecture ("llama.block_count",
# "qwen3moe.attention.head_count_kv", ...). Deliberately does NOT include any
# expert/MoE key: expert weights cost VRAM but contribute nothing to KV, and
# conflating the two is the bug this exists to fix.
_GGUF_KV_SHAPE_SUFFIXES = (
    ".block_count",
    ".embedding_length",
    ".attention.head_count",
    ".attention.head_count_kv",
    ".attention.key_length",
    ".attention.value_length",
)

# Bounded read for the metadata probe: real GGUF writers put general.* and
# <arch>.pooling_type keys before the (often large) tokenizer vocab arrays and
# all tensor data, so a few MB is always enough; this guarantees the probe
# never reads a multi-GB model file just to classify it.
_GGUF_META_PROBE_BYTES = 4 * 1024 * 1024


def _gguf_read_string(buf: bytes, off: int):
    """Read a GGUF length-prefixed string at *off*; returns (value, new_offset).
    Raises struct.error/UnicodeDecodeError/IndexError on malformed input -
    callers catch and treat that as 'no signal', never crash."""
    (n,) = struct.unpack_from("<Q", buf, off)
    off += 8
    if n > len(buf) - off or n > 1_000_000:
        # No real GGUF key or architecture NAME is anywhere near 1 MB; a huge
        # length here means we're mis-parsing (or past our bounded read), not
        # that this is a legitimately giant string - bail out cleanly.
        raise struct.error("gguf string length implausible or out of bounds")
    return buf[off:off + n].decode("utf-8"), off + n


def _gguf_skip_value(buf: bytes, off: int, vtype: int) -> int:
    """Advance past one GGUF metadata VALUE of type *vtype* at *off*, returning
    the new offset. Raises on an unsupported/malformed type (caught by the
    caller) rather than silently mis-parsing the rest of the file."""
    if vtype == _GGUF_TYPE_STRING:
        _, off = _gguf_read_string(buf, off)
        return off
    if vtype == _GGUF_TYPE_ARRAY:
        (elem_type,) = struct.unpack_from("<I", buf, off)
        off += 4
        (count,) = struct.unpack_from("<Q", buf, off)
        off += 8
        if elem_type == _GGUF_TYPE_STRING:
            for _ in range(count):
                _, off = _gguf_read_string(buf, off)
            return off
        size = _GGUF_FIXED_TYPE_SIZES.get(elem_type)
        if size is None:
            raise struct.error(f"unsupported gguf array element type {elem_type}")
        return off + size * count
    size = _GGUF_FIXED_TYPE_SIZES.get(vtype)
    if size is None:
        raise struct.error(f"unsupported gguf value type {vtype}")
    return off + size


def _gguf_read_scalar(buf: bytes, off: int, vtype: int):
    """Read one FIXED-WIDTH GGUF scalar at *off*; returns (value, new_offset).

    Raises struct.error for a string/array/unknown type rather than guessing, so
    a key whose value is not a plain number (e.g. a per-layer head_count_kv
    ARRAY) can never be silently read as one. Callers catch and treat that as
    'no signal'."""
    fmt = _GGUF_SCALAR_FORMATS.get(vtype)
    if fmt is None:
        raise struct.error(f"gguf value type {vtype} is not a fixed-width scalar")
    (val,) = struct.unpack_from(fmt, buf, off)
    return val, off + _GGUF_FIXED_TYPE_SIZES[vtype]


def gguf_kv_bytes_per_token(path: Path) -> int:
    """f16 KV-cache bytes per token, computed from *path*'s own GGUF header.

    Same formula as ``LlamaCpp._read_kv_bytes_per_token``, but read from the
    FILE rather than from a loaded model. That is the entire point: the offload
    decision (how many layers fit in VRAM) has to be made BEFORE the model is
    loaded, so until now it charged KV from the file's SIZE - which is wrong in
    both directions. A sparse MoE inflates the file with expert weights that
    cost no KV at all, so it was over-charged; a wide-KV dense model was
    under-charged (~2.6x low on a 12B) and could be judged to fit when its KV
    cache actually overflows VRAM.

    K and V cache = n_layers * n_head_kv * head_dim, times 2 (K and V) and times
    2 bytes/element (llama.cpp's default f16 type_k/type_v). head_dim comes from
    the explicit ``attention.key_length``/``value_length`` keys when present
    (several architectures set a head_dim that is NOT n_embd/n_head) and falls
    back to n_embd // n_head otherwise.

    Returns 0 - never raises - when the file is not a readable GGUF, or the
    shape keys are absent, non-scalar, or non-positive. 0 means 'no signal', and
    the caller keeps its previous heuristic; a wrong number here would silently
    mis-size every load, so refusing to answer is the safe failure."""
    try:
        with open(path, "rb") as f:
            buf = f.read(_GGUF_META_PROBE_BYTES)
    except OSError:
        return 0

    architecture = None
    vals: dict = {}
    try:
        if buf[:4] != b"GGUF":
            return 0
        (version,) = struct.unpack_from("<I", buf, 4)
        if version < 2:
            return 0            # v1's 32-bit counts predate every arch we size
        _tensor_count, kv_count = struct.unpack_from("<QQ", buf, 8)
        off = 24
        for _ in range(kv_count):
            key, off = _gguf_read_string(buf, off)
            (vtype,) = struct.unpack_from("<I", buf, off)
            off += 4
            if key == "general.architecture" and vtype == _GGUF_TYPE_STRING:
                architecture, off = _gguf_read_string(buf, off)
                continue
            # Collect by FULL key and resolve against the architecture at the
            # end, so key order does not matter and an mmproj's parallel
            # "clip.*" attention block can never be mistaken for the LLM's.
            if any(key.endswith(s) for s in _GGUF_KV_SHAPE_SUFFIXES):
                try:
                    vals[key], off = _gguf_read_scalar(buf, off, vtype)
                    continue
                except struct.error:
                    pass        # not a scalar (array/string) - skip it normally
            off = _gguf_skip_value(buf, off, vtype)
    except (struct.error, IndexError, UnicodeDecodeError):
        # Truncated inside the bounded read, or a malformed layout. Fall through
        # and answer from whatever resolved cleanly; if that is not enough the
        # arithmetic below returns 0 and the caller falls back.
        pass

    if not architecture:
        return 0

    def _get(suffix: str) -> int:
        v = vals.get(f"{architecture}{suffix}")
        return int(v) if isinstance(v, int) and v > 0 else 0

    n_layers = _get(".block_count")
    n_head_kv = _get(".attention.head_count_kv")
    if not n_layers or not n_head_kv:
        return 0
    k_len = _get(".attention.key_length")
    v_len = _get(".attention.value_length")
    if k_len and v_len:
        return n_layers * n_head_kv * (k_len + v_len) * 2
    n_embd = _get(".embedding_length")
    n_head = _get(".attention.head_count")
    if not n_embd or not n_head:
        return 0
    head_dim = n_embd // n_head
    if head_dim <= 0:
        return 0
    return n_layers * n_head_kv * head_dim * 2 * 2


def gguf_expert_count(path: Path) -> int:
    """Number of EXPERTS in *path*, or 0 when it is not a Mixture-of-Experts model.

    Read from the header's ``<arch>.expert_count`` before the model is loaded, so
    a placement decision that only makes sense for an MoE can tell the difference
    rather than silently doing nothing on a dense model.

    Deliberately separate from ``gguf_kv_bytes_per_token``: expert weights cost
    VRAM but contribute NOTHING to the KV cache, and conflating the two is the bug
    that function exists to fix. Returns 0 - never raises - on an unreadable or
    non-GGUF file, same 'no signal' contract as the other probes here."""
    try:
        with open(path, "rb") as f:
            buf = f.read(_GGUF_META_PROBE_BYTES)
    except OSError:
        return 0

    architecture = None
    counts: dict = {}
    try:
        if buf[:4] != b"GGUF":
            return 0
        (version,) = struct.unpack_from("<I", buf, 4)
        if version < 2:
            return 0
        _tensor_count, kv_count = struct.unpack_from("<QQ", buf, 8)
        off = 24
        for _ in range(kv_count):
            key, off = _gguf_read_string(buf, off)
            (vtype,) = struct.unpack_from("<I", buf, off)
            off += 4
            if key == "general.architecture" and vtype == _GGUF_TYPE_STRING:
                architecture, off = _gguf_read_string(buf, off)
                continue
            if key.endswith(".expert_count"):
                try:
                    counts[key], off = _gguf_read_scalar(buf, off, vtype)
                    continue
                except struct.error:
                    pass
            off = _gguf_skip_value(buf, off, vtype)
    except (struct.error, IndexError, UnicodeDecodeError):
        pass

    if not architecture:
        return 0
    value = counts.get(architecture + ".expert_count")
    return int(value) if isinstance(value, int) and value > 0 else 0


def _gguf_metadata_probe(path: Path) -> dict:
    """Best-effort read of the GGUF header metadata needed for embedding-model
    detection: ``general.architecture`` and whether any ``*.pooling_type`` key
    is present. Reads only a bounded prefix of the file (see
    ``_GGUF_META_PROBE_BYTES`` - real metadata always precedes the large
    tokenizer vocab arrays and tensor data), never the whole model. Returns
    ``{}`` on any parse failure or truncation within that bound - this must
    NEVER raise or crash a caller; an unreadable/unexpected header just means
    'no signal', exactly like ``_has_gguf_magic``'s existing defensive style,
    and the caller falls back to its pre-existing classification. A truncation
    or malformed key AFTER a definitive signal was already resolved (e.g. a
    huge multilingual tokenizer vocab array following an early
    general.architecture="bert" key) returns that already-resolved signal
    rather than discarding it - the early-exit below stops as soon as either
    signal is definitively known, so this is the normal case, not just a
    fallback."""
    try:
        with open(path, "rb") as f:
            buf = f.read(_GGUF_META_PROBE_BYTES)
    except OSError:
        return {}
    architecture = None
    has_pooling_type = False
    try:
        if buf[:4] != b"GGUF":
            return {}
        (version,) = struct.unpack_from("<I", buf, 4)
        if version < 2:
            # v1 used 32-bit tensor/kv counts and predates every architecture
            # this detector cares about; no signal rather than mis-parsing it
            # with the v2+ 64-bit layout.
            return {}
        tensor_count, kv_count = struct.unpack_from("<QQ", buf, 8)
        off = 24
        for _ in range(kv_count):
            key, off = _gguf_read_string(buf, off)
            (vtype,) = struct.unpack_from("<I", buf, off)
            off += 4
            if key == "general.architecture" and vtype == _GGUF_TYPE_STRING:
                architecture, off = _gguf_read_string(buf, off)
            else:
                if key.endswith(".pooling_type"):
                    has_pooling_type = True
                off = _gguf_skip_value(buf, off, vtype)
            # Stop as soon as the final answer is already decided: a definitive
            # embedding architecture, or any pooling_type key at all, makes the
            # rest of the KV block irrelevant to classification. Requiring only
            # ONE of the two (not both) means a truncation past this point can
            # never discard an already-confirmed signal.
            if has_pooling_type or architecture in _GGUF_EMBEDDING_ARCHITECTURES:
                break
    except (struct.error, IndexError, UnicodeDecodeError):
        # Truncated within our bounded read, or a malformed/unexpected layout -
        # fall through and report whatever was already resolved before the
        # failure, rather than discarding a signal found earlier in the walk.
        pass
    return {"architecture": architecture, "has_pooling_type": has_pooling_type}


def gguf_embedding_signal(path: Path) -> bool:
    """True when *path*'s own GGUF metadata marks it as an embedding/pooling
    model rather than a causal-chat LLM: either its ``general.architecture`` is
    one of llama.cpp's dedicated encoder/embedding architectures
    (``_GGUF_EMBEDDING_ARCHITECTURES``), or it carries a
    ``"<architecture>.pooling_type"`` key at all - llama.cpp's converter only
    writes that key for a pooling-configured export (e.g. Qwen3-Embedding,
    gte-Qwen2, e5-mistral all reuse a decoder architecture whose
    general.architecture is unchanged from the chat variant, so the pooling-type
    key is the only signal that catches them). Both are hard metadata baked
    into the file itself - never a filename guess. Used by
    ``_detect_local_model_type`` (local add + folder auto-sync) and by
    ``pull.py`` (a freshly-downloaded remote GGUF)."""
    meta = _gguf_metadata_probe(path)
    if meta.get("architecture") in _GGUF_EMBEDDING_ARCHITECTURES:
        return True
    return bool(meta.get("has_pooling_type"))

