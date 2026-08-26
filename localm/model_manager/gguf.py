# SPDX-License-Identifier: AGPL-3.0-or-later
"""GGUF file-format helpers: split-part detection, the gguf magic check,
safe model filenames, and SHA256 hashing. Leaf utilities, no registry/pull deps."""

import localm.model_manager as _mm  # read package-patchable names at call time

import hashlib
import queue
import re
import struct
import threading
import time
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




# Block size for the hashing read. A reader thread overlaps the read with the
# hash.
_HASH_BLOCK_BYTES = 4 * 1024 * 1024

# How many blocks the reader may run ahead. Bounds peak memory at roughly
# _HASH_BLOCK_BYTES * (_HASH_READAHEAD_BLOCKS + 1).
_HASH_READAHEAD_BLOCKS = 4

# Below this size, hash inline on the calling thread instead of spawning the
# reader.
_HASH_THREAD_MIN_BYTES = 32 * 1024 * 1024


def _iter_file_blocks(path: Path):
    """Yield *path*'s bytes in ``_HASH_BLOCK_BYTES`` chunks.

    Large files are read by a background thread one block ahead of the consumer
    so the read and whatever the consumer does with the block overlap; small
    ones are read inline. An error in the reader is re-raised in the CONSUMER's
    thread."""
    # Read through _mm so the thresholds stay patchable at call time.
    block_bytes = _mm._HASH_BLOCK_BYTES
    try:
        size = path.stat().st_size
    except OSError:
        size = 0

    if size < _mm._HASH_THREAD_MIN_BYTES:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(block_bytes), b""):
                yield block
        return

    q: "queue.Queue" = queue.Queue(maxsize=_mm._HASH_READAHEAD_BLOCKS)
    eof = object()

    def _reader() -> None:
        try:
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(block_bytes), b""):
                    q.put(block)
        except BaseException as exc:   # handed to the consumer, never swallowed
            q.put(exc)
        else:
            q.put(eof)

    t = threading.Thread(target=_reader, daemon=True,
                         name="localm-sha256-reader")
    t.start()
    try:
        while True:
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                if not t.is_alive():
                    # The reader exited without posting eof OR an exception.
                    raise RuntimeError(
                        f"the reader thread for '{path.name}' exited without "
                        "delivering data or an error; the digest would be "
                        "incomplete, so it is not returned")
                continue
            if item is eof:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        # If the consumer abandons the generator, the reader can be parked on a
        # full queue. Drain until it exits so the thread and its buffered blocks
        # are released. On the normal path the reader has already finished.
        while t.is_alive():
            try:
                q.get(timeout=0.05)
            except queue.Empty:
                pass
        t.join()


def _sha256_file(
    path: Path,
    progress: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Return the hex SHA256 digest of a file.

    When *progress* is given it is called ``progress(bytes_done, total_bytes)``
    after each block so a caller can drive a progress bar; *total_bytes* is the
    file size (0 if it cannot be stat'd). The callback must not raise.

    *progress* is invoked on the CALLER's own thread, not on the reader
    thread."""
    h = hashlib.sha256()
    total = 0
    if progress is not None:
        try:
            total = path.stat().st_size
        except OSError:
            total = 0
    done = 0
    for block in _iter_file_blocks(path):
        h.update(block)
        if progress is not None:
            done += len(block)
            progress(done, total)
    return h.hexdigest()




def _sha256_file_bytes(data: bytes) -> str:
    """Return the hex SHA256 digest of an in-memory byte string."""
    return hashlib.sha256(data).hexdigest()




# Every character Windows refuses to let a real filename contain, plus the C0
# control range. ':' is included: it opens an NTFS Alternate Data Stream instead
# of failing, so 'somefile.exe:mmproj.gguf' stays inside base_dir while writing
# its content into a stream hidden behind a zero-byte 'somefile.exe'.
_WINDOWS_RESERVED_CHARS = frozenset('<>:"/\\|?*') | frozenset(chr(c) for c in range(32))

# Windows reserved DEVICE names, matched against the part of a filename before
# its FIRST '.', case-insensitively, so 'con.mmproj.gguf' is reserved as well as
# a bare 'CON'.
_WINDOWS_RESERVED_STEMS = frozenset({
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
})


def _safe_models_filename(filename: str, base_dir: Optional[Path] = None) -> Optional[str]:
    """Return a single-component filename confined to *base_dir* (``MODELS_DIR``
    by default).

    A model download must never write outside its destination folder.
    ``filename`` is derived from untrusted input (a URL path, an
    ``owner/repo:file`` spec, or a remote HF repo's own file listing). Returns
    the bare filename when it is a single, non-traversing, non-hazardous path
    component, else ``None`` - so ``../../evil.gguf`` and ``sub/dir/evil.gguf``
    are both rejected. *base_dir* lets a caller routing a download to a
    non-default destination (e.g. a ComfyUI models subfolder) validate against
    the REAL destination instead of always against ``MODELS_DIR``.

    Beyond directory escape, this also rejects the Windows filename-CONFUSION
    shapes: any reserved character (notably ':', which names an NTFS Alternate
    Data Stream - it stays confined to *base_dir* and so passes a pure escape
    check, while hiding its content behind an apparently-empty sibling file -
    see ``_WINDOWS_RESERVED_CHARS``), a reserved device stem
    (``_WINDOWS_RESERVED_STEMS``), and a trailing '.' or ' ' (Windows strips
    these when resolving a path, so ``evil.gguf.`` and ``evil.gguf`` name the
    SAME file while a directory listing only ever shows the stripped form).
    """
    base_dir = base_dir if base_dir is not None else _mm.MODELS_DIR
    if not filename:
        return None
    if any(c in _WINDOWS_RESERVED_CHARS for c in filename):
        return None
    if filename[-1] in (".", " "):
        return None
    stem = filename.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED_STEMS:
        return None
    # Reject anything that is not a single path component: no separators, no
    # drive or absolute prefixes, no lone '.' or '..'. Path separators are
    # already excluded by the reserved-character check above.
    name = Path(filename).name
    if name != filename or name in ("", ".", ".."):
        return None
    try:
        dest = (base_dir / name).resolve()
        if dest.parent != base_dir.resolve():
            return None
        if dest.exists() and dest.name.lower() != name.lower():
            # The resolved destination is an OS-level alias for a
            # differently-named existing file (an NTFS 8.3 short name such as
            # 'LONGMO~1.GGU'). Compared case-insensitively, since Windows path
            # resolution returns the on-disk casing: requesting an existing
            # 'model.gguf' as 'MODEL.GGUF' is the same file and stays accepted.
            return None
    except (OSError, ValueError):
        # ValueError also covers a null/odd path that slipped past the check
        # above; either way, an unresolvable name is not safe.
        return None
    return name




# Files at or below this size hash inline and silently; larger ones get the
# off-main-thread progress bar.
_HASH_PROGRESS_MIN_BYTES = 512 * 1024 * 1024




def _hash_with_progress(path: Path,
                        *, purpose: str = "for duplicate detection") -> Optional[str]:
    """SHA256 a model file, showing live progress for large files.

    For files larger than ``_HASH_PROGRESS_MIN_BYTES`` the hash runs in a
    background worker thread so the main thread stays responsive and can render a
    progress bar; smaller files hash inline without any UI. Returns None for
    directories (HF models are identified by path only).

    *purpose* completes the bar's label ("Hashing <file> <purpose>"), naming
    which wait this is: duplicate detection, or verifying a download against
    --sha256."""
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
        # {task.description} is rich's own placeholder and must survive the
        # f-string, hence the doubled braces; only *purpose* is interpolated.
        TextColumn(f"[dim]Hashing {{task.description}} {purpose}[/dim]"),
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




# Size floor for a plausible GGUF. The fixed header alone (magic, version,
# tensor count, KV count) is 24 bytes; a real model carries a metadata KV block
# plus tensor infos well past 1 KiB before the first weight byte.
_GGUF_MIN_BYTES = 1024


def _has_gguf_magic(path: Path) -> bool:
    """True when *path* begins with the GGUF magic ``b"GGUF"`` and is at least
    ``_GGUF_MIN_BYTES`` long.

    Auto-registration (sync_models_dir) keys on the ``.gguf`` extension alone,
    so this is what keeps a foreign file renamed ``.gguf``, a 0-byte
    placeholder, or a partial copy that never received its header out of the
    registry. A real GGUF always starts with this 4-byte magic; an unreadable
    file is treated as not-a-GGUF and skipped. The size floor additionally
    rejects a header-only truncated copy that got just the magic. A mid-copy of
    a *valid* GGUF that already passed the floor remains a best-effort gap."""
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


# How long a file's mtime must be untouched before auto-registration treats it as
# finished rather than still being written. A best-effort quiet period, not a
# lock: a copy that itself stalls for longer than this window reads as settled.
# A file that fails this check is picked up on a later sync_models_dir call.
_GGUF_SETTLE_SECONDS = 5.0


def _gguf_recently_written(path: Path) -> bool:
    """True when *path*'s mtime is within ``_GGUF_SETTLE_SECONDS`` of now, i.e.
    it may still be mid-copy. An unreadable file is treated as NOT recently
    written so it falls through to whatever check runs next (fails safe: this
    function only ever defers registration, never blocks it outright)."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return False
    return (time.time() - mtime) < _mm._GGUF_SETTLE_SECONDS


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

# llama.cpp's LLM_ARCH_NAMES entries for encoder/embedding-only architectures. A
# GGUF whose general.architecture is one of these is an embedding/encoder model,
# never a causal-chat LLM.
#
# Also read by localm.discover.classify_hf_metadata, which badges a HuggingFace
# search result from HF's server-side gguf.architecture expand field, so an edit
# here changes what the search page reports as well as local detection.
_GGUF_EMBEDDING_ARCHITECTURES = frozenset({
    "bert", "modern-bert", "nomic-bert", "nomic-bert-moe", "neo-bert",
    "jina-bert-v2", "jina-bert-v3", "eurobert", "gemma-embedding",
    "t5encoder", "pangu-embedded",
})

# GGUF metadata value types (ggml gguf.h `enum gguf_type`). STRING(8) and
# ARRAY(9) are variable-length and handled separately; every other type here maps
# to its fixed byte width.
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

# The '<architecture>.'-prefixed keys describing the attention shape, i.e. what
# the KV cache costs per token. Stored as suffixes because llama.cpp namespaces
# them under general.architecture ('llama.block_count',
# 'qwen3moe.attention.head_count_kv', ...). No expert/MoE key belongs here:
# expert weights contribute nothing to the KV cache.
_GGUF_KV_SHAPE_SUFFIXES = (
    ".block_count",
    ".embedding_length",
    ".attention.head_count",
    ".attention.head_count_kv",
    ".attention.key_length",
    ".attention.value_length",
)

# The one shape key a hybrid architecture states PER LAYER rather than once for
# the whole stack, so it needs an array read the others do not.
_GGUF_KV_HEADS_SUFFIX = ".attention.head_count_kv"

# Integer element types a per-layer array may use, keyed identically to
# _GGUF_FIXED_TYPE_SIZES. Real files write int32 (type 5); the smaller widths are
# accepted too. Floats, bools and strings are absent, so an array of those is
# never read as a head count.
_GGUF_INT_ARRAY_FORMATS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 10: "<Q", 11: "<q",
}

# Upper bound on a per-layer array, as a mis-parse guard. llama.cpp's
# LLAMA_MAX_LAYERS is 512, so a longer array means a wrong offset, and no list is
# allocated from such a count.
_GGUF_MAX_LAYER_ARRAY = 4096

# Mis-parse guards for the tensor-info walk, generous against any real model.
_GGUF_MAX_TENSOR_COUNT = 1_000_000
_GGUF_MAX_TENSOR_DIMS = 8

# Key families that mark an architecture as keeping a FIXED-size recurrent state
# (state-space, linear attention, short convolution) in place of a growing KV
# cache on some layers. Matched as '<arch>' plus infix, so an mmproj's or another
# tower's keys cannot vote, like the shape keys above. A marker rather than an
# architecture-name table; an unmarked hybrid keeps today's behaviour. '.ssm.'
# alone is not sufficient: lfm2 is hybrid via short convolution and declares no
# ssm.* key at all.
_GGUF_RECURRENT_KEY_INFIXES = (".ssm.", ".shortconv.")

# Bounded read for the metadata probe. Real GGUF writers put general.* and
# <arch>.pooling_type keys before the tokenizer vocab arrays and all tensor data,
# so the probe never reads a multi-GB model file just to classify it.
_GGUF_META_PROBE_BYTES = 4 * 1024 * 1024

# A second, larger bound used only by the tensor-name pass below: the tensor list
# sits after the whole metadata block, past the budget above. Paid only by a
# hybrid that states a single head count, and still a bounded prefix.
_GGUF_TENSOR_PROBE_BYTES = 32 * 1024 * 1024


def _gguf_read_string(buf: bytes, off: int):
    """Read a GGUF length-prefixed string at *off*; returns (value, new_offset).
    Raises struct.error/UnicodeDecodeError/IndexError on malformed input -
    callers catch and treat that as 'no signal', never crash."""
    (n,) = struct.unpack_from("<Q", buf, off)
    off += 8
    if n > len(buf) - off or n > 1_000_000:
        # No real GGUF key or architecture name approaches 1 MB, so a length this
        # large means a mis-parse or a read past the bounded prefix.
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


def _gguf_read_int_array(buf: bytes, off: int):
    """Read a GGUF array of FIXED-WIDTH INTEGERS at *off*; returns (list, new_offset).

    A hybrid architecture states ``attention.head_count_kv`` as one entry PER
    LAYER, and those entries are the exact per-layer truth - a 0 marks a layer
    that keeps a fixed-size recurrent state and holds no KV cache at all.
    ``_gguf_read_scalar`` REFUSES an array, so a per-layer value can never be
    read as a whole-stack one; this is the separate reader for that case.

    Raises struct.error for a non-integer element type, an implausible length, or
    a count running past the bounded read. Callers catch and treat that as 'no
    signal'."""
    (elem_type,) = struct.unpack_from("<I", buf, off)
    (count,) = struct.unpack_from("<Q", buf, off + 4)
    off += 12
    fmt = _GGUF_INT_ARRAY_FORMATS.get(elem_type)
    if fmt is None:
        raise struct.error(f"gguf array element type {elem_type} is not an integer")
    size = _GGUF_FIXED_TYPE_SIZES[elem_type]
    # Bounds BEFORE building the list, so a bogus count can never allocate.
    if count > _GGUF_MAX_LAYER_ARRAY or size * count > len(buf) - off:
        raise struct.error("gguf array implausibly long or out of bounds")
    return ([struct.unpack_from(fmt, buf, off + i * size)[0] for i in range(count)],
            off + size * count)


def _gguf_attending_layer_count(path: Path, n_layers: int) -> int:
    """How many of *n_layers* blocks actually hold a KV cache, read from the
    file's TENSOR NAMES. Returns 0 - never raises - when that cannot be
    determined, which callers treat as 'no signal'.

    A hybrid that states ONE head_count_kv for a stack whose layers differ does
    not record which layers attend anywhere in its metadata. The tensor list does
    record it, unambiguously and with NO architecture table: an attending layer
    carries blk.<i>.attn_k / attn_v weights, and a linear-attention, state-space
    or short-convolution layer does not.

    Keyed on attn_k/attn_v SPECIFICALLY, never on a bare "attn": a hybrid carries
    blk.<i>.attn_norm.weight on every layer whether it attends or not, so
    matching the norm tensor would count the whole stack."""
    try:
        with open(path, "rb") as f:
            buf = f.read(_GGUF_TENSOR_PROBE_BYTES)
    except OSError:
        return 0
    try:
        if buf[:4] != b"GGUF":
            return 0
        tensor_count, kv_count = struct.unpack_from("<QQ", buf, 8)
        if tensor_count > _GGUF_MAX_TENSOR_COUNT:
            return 0            # a mis-parse, not a real model
        off = 24
        for _ in range(kv_count):           # skip the whole metadata block
            _key, off = _gguf_read_string(buf, off)
            (vtype,) = struct.unpack_from("<I", buf, off)
            off = _gguf_skip_value(buf, off + 4, vtype)
        attending = set()
        for _ in range(tensor_count):
            name, off = _gguf_read_string(buf, off)
            (n_dims,) = struct.unpack_from("<I", buf, off)
            if n_dims > _GGUF_MAX_TENSOR_DIMS:
                return 0
            # n_dims (u32) + dims (u64 each) + ggml type (u32) + offset (u64)
            off += 4 + 8 * n_dims + 4 + 8
            if not name.startswith("blk."):
                continue
            if ".attn_k." not in name and ".attn_v." not in name:
                continue
            try:
                attending.add(int(name.split(".")[1]))
            except ValueError:
                return 0        # an unexpected layout - refuse rather than guess
    except (struct.error, IndexError, UnicodeDecodeError):
        # Ran past the bounded read, or a malformed layout. A truncated tensor
        # list under-counts attending layers and would under-charge the KV cache,
        # so this refuses rather than answering from a partial result.
        return 0
    # More attending layers than the stack has means the two disagree about what
    # they describe, so neither can be trusted.
    return len(attending) if 0 < len(attending) <= n_layers else 0


def gguf_kv_bytes_per_token(path: Path) -> int:
    """f16 KV-cache bytes per token, computed from *path*'s own GGUF header.

    Same formula as ``LlamaCpp._read_kv_bytes_per_token``, but read from the
    FILE rather than from a loaded model, so the offload decision (how many
    layers fit in VRAM) can be made BEFORE the model is loaded.

    K and V cache = (total KV heads across all layers) * head_dim, times 2 (K and
    V) and times 2 bytes/element (llama.cpp's default f16 type_k/type_v).

    "Across all layers" is not n_layers * n_head_kv on a HYBRID architecture
    (Qwen3-Next, Granite 4 H, LFM2, Jamba, Falcon-H1 ...), where most layers use
    linear attention / a state-space model / a short convolution and keep a
    FIXED-size recurrent state instead of a KV cache that grows with the context.
    Those layers cost no per-token KV at all.

    Such a file states head_count_kv one entry PER LAYER, and that array is
    summed exactly, zeros included. When a hybrid instead states a single scalar
    (Qwen3-Next does) the file does not record which layers attend, so this
    returns 0.

    head_dim comes from the explicit ``attention.key_length``/``value_length``
    keys when present (several architectures set a head_dim that is NOT
    n_embd/n_head) and falls back to n_embd // n_head otherwise.

    Returns 0 - never raises - when the file is not a readable GGUF, or the
    shape keys are absent, non-scalar, or non-positive. 0 means 'no signal', and
    the caller keeps its previous heuristic."""
    try:
        with open(path, "rb") as f:
            buf = f.read(_GGUF_META_PROBE_BYTES)
    except OSError:
        return 0

    architecture = None
    vals: dict = {}
    per_layer_kv: dict = {}
    recurrent_keys: set = set()
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
            # Collect by FULL key and resolve against the architecture at the end,
            # so key order does not matter and an mmproj's parallel 'clip.*'
            # attention block is not read as the LLM's. A hybrid stack states
            # head_count_kv once PER LAYER, so that array is read here.
            if key.endswith(_GGUF_KV_HEADS_SUFFIX) and vtype == _GGUF_TYPE_ARRAY:
                try:
                    per_layer_kv[key], off = _gguf_read_int_array(buf, off)
                    continue
                except struct.error:
                    pass        # not an integer array - skip it normally
            if any(key.endswith(s) for s in _GGUF_KV_SHAPE_SUFFIXES):
                try:
                    vals[key], off = _gguf_read_scalar(buf, off, vtype)
                    continue
                except struct.error:
                    pass        # not a scalar (array/string) - skip it normally
            # Note by NAME only, with no value read, that this file declares a
            # recurrent-state layer family; resolved against the architecture at
            # the end like everything else here.
            if any(infix in key for infix in _GGUF_RECURRENT_KEY_INFIXES):
                recurrent_keys.add(key)
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

    # Total KV heads summed over the whole stack. Every layer contributes on a
    # uniform architecture; on a hybrid only the attending layers do.
    per_layer = per_layer_kv.get(f"{architecture}{_GGUF_KV_HEADS_SUFFIX}")
    if per_layer is not None:
        # The array is the per-layer truth, so this is exact. It must describe the
        # same stack block_count does; a length mismatch returns 0.
        if not n_layers or len(per_layer) != n_layers:
            return 0
        total_kv_heads = sum(v for v in per_layer if v > 0)
    else:
        n_head_kv = _get(_GGUF_KV_HEADS_SUFFIX)
        if not n_layers or not n_head_kv:
            return 0
        if any(k.startswith(f"{architecture}{infix}")
               for infix in _GGUF_RECURRENT_KEY_INFIXES for k in recurrent_keys):
            # Hybrid stated as ONE number for a stack whose layers differ, so
            # n_layers is the wrong multiplier and the metadata block does not
            # carry the right one. The tensor names give it exactly.
            attending = _gguf_attending_layer_count(path, n_layers)
            if not attending:
                return 0        # could not tell - no signal, caller falls back
            total_kv_heads = attending * n_head_kv
        else:
            total_kv_heads = n_layers * n_head_kv
    if total_kv_heads <= 0:
        return 0                # e.g. a fully recurrent stack: no KV cache

    k_len = _get(".attention.key_length")
    v_len = _get(".attention.value_length")
    if k_len and v_len:
        return total_kv_heads * (k_len + v_len) * 2
    n_embd = _get(".embedding_length")
    n_head = _get(".attention.head_count")
    if not n_embd or not n_head:
        return 0
    head_dim = n_embd // n_head
    if head_dim <= 0:
        return 0
    return total_kv_heads * head_dim * 2 * 2


def gguf_expert_count(path: Path) -> int:
    """Number of EXPERTS in *path*, or 0 when it is not a Mixture-of-Experts model.

    Read from the header's ``<arch>.expert_count`` before the model is loaded, so
    a placement decision that only makes sense for an MoE can tell the difference
    from a dense model.

    Separate from ``gguf_kv_bytes_per_token``: expert weights cost VRAM but
    contribute NOTHING to the KV cache. Returns 0 - never raises - on an
    unreadable or non-GGUF file, the same 'no signal' contract as the other
    probes here."""
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


# The FUSED per-layer expert weight tensors, as llama.cpp's converters name them:
# blk.<i>.ffn_gate_exps / ffn_down_exps / ffn_up_exps - one tensor PER PROJECTION
# with every expert fused into it, not one tensor per expert. The router
# (ffn_gate_inp) and any SHARED expert are excluded; llama.cpp never moves them.
#
# llamacpp/llama.py's _apply_cpu_moe imports these same constants to build its
# native tensor_buft_overrides regex, and gguf_moe_pinned_expert_bytes below sums
# exactly the tensors this pattern matches. Built by concatenation, so the
# literal stays a plain regex with no interpolation.
_MOE_TENSOR_PREFIX = r"blk\."
_MOE_TENSOR_SUFFIX = r"\.ffn_(gate|down|up)_exps"
# General matcher (not tied to one layer index) with the index as a capture
# group, derived from the same prefix and suffix _apply_cpu_moe builds per-layer.
_MOE_EXPERT_TENSOR_RE = re.compile(_MOE_TENSOR_PREFIX + r"(\d+)" + _MOE_TENSOR_SUFFIX)


def _gguf_read_string_stream(f) -> str:
    """Read a length-prefixed GGUF string from an open, positioned file
    handle, advancing past it. Streaming counterpart to ``_gguf_read_string``
    (which reads from an in-memory buffer): tensor-info parsing (see
    ``gguf_moe_pinned_expert_bytes``) must read PAST the entire metadata KV
    block, including whatever tokenizer vocabulary array it contains (routinely
    several MB for a 100k+-token vocab), so it cannot use the bounded 4 MB
    slurp ``gguf_kv_bytes_per_token``/``gguf_expert_count`` use."""
    (n,) = struct.unpack("<Q", f.read(8))
    if n > 10_000_000:
        # No real GGUF tensor or key name approaches 10 MB, so a length this large
        # means the stream is misaligned or corrupt.
        raise struct.error("gguf string length implausible")
    data = f.read(n)
    if len(data) != n:
        raise struct.error("gguf string truncated")
    return data.decode("utf-8")


def _gguf_skip_value_stream(f, vtype: int) -> None:
    """Streaming counterpart to ``_gguf_skip_value`` - advance *f* past one
    GGUF metadata VALUE of type *vtype* by seeking, not by reading array/blob
    bytes into memory that nothing here needs."""
    if vtype == _GGUF_TYPE_STRING:
        _gguf_read_string_stream(f)
        return
    if vtype == _GGUF_TYPE_ARRAY:
        (elem_type,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if elem_type == _GGUF_TYPE_STRING:
            for _ in range(count):
                _gguf_read_string_stream(f)
            return
        size = _GGUF_FIXED_TYPE_SIZES.get(elem_type)
        if size is None:
            raise struct.error(f"unsupported gguf array element type {elem_type}")
        f.seek(size * count, 1)
        return
    size = _GGUF_FIXED_TYPE_SIZES.get(vtype)
    if size is None:
        raise struct.error(f"unsupported gguf value type {vtype}")
    f.seek(size, 1)


# GGUF pads the tensor-info section to this many bytes before tensor DATA begins.
# A file may override it via a general.alignment KV key, which
# gguf_moe_pinned_expert_bytes does not read.
_GGUF_DEFAULT_ALIGNMENT = 32

# Sanity ceiling on a single tensor's dimension count, generous against
# GGML_MAX_DIMS (4 in every real ggml build).
_GGUF_MAX_TENSOR_DIMS = 8


def gguf_moe_pinned_expert_bytes(path: Path, n_pinned_layers: int) -> Optional[int]:
    """Bytes occupied by the routed-expert weight tensors of the FIRST
    *n_pinned_layers* transformer layers - the exact tensors ``_apply_cpu_moe``
    (llamacpp/llama.py) pins to system RAM for an ``n_cpu_moe=N`` load (see
    ``_MOE_EXPERT_TENSOR_RE`` above for which tensors).

    Computed from each matching tensor's OFFSET DELTA in the file's own
    tensor-info section (the next tensor's offset minus this one's, sorted by
    offset; the last tensor's size comes from the file's total size instead)
    rather than decoding ggml's per-quantization-type block format. It needs
    no per-type size table and is EXACT regardless of quantization scheme, as
    long as tensors are laid out contiguously in offset order - true for every
    llama.cpp-produced GGUF.

    Returns ``None`` - never raises - when the file cannot be parsed as a
    GGUF, or *n_pinned_layers* is <= 0; the caller then charges the whole
    file. Returns ``0`` (a real answer, not a failure) when parsing succeeds
    but nothing in the pinned layer range matches - e.g. a dense model, where
    ``_apply_cpu_moe`` already treats ``n_cpu_moe`` as a no-op via its own
    ``gguf_expert_count() == 0`` guard."""
    if n_pinned_layers <= 0:
        return None
    try:
        file_size = path.stat().st_size
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            (version,) = struct.unpack("<I", f.read(4))
            if version < 2:
                return None
            tensor_count, kv_count = struct.unpack("<QQ", f.read(16))
            for _ in range(kv_count):
                _gguf_read_string_stream(f)          # key (value unused here)
                (vtype,) = struct.unpack("<I", f.read(4))
                _gguf_skip_value_stream(f, vtype)
            entries = []
            for _ in range(tensor_count):
                name = _gguf_read_string_stream(f)
                (n_dims,) = struct.unpack("<I", f.read(4))
                if n_dims > _GGUF_MAX_TENSOR_DIMS:
                    raise struct.error(f"implausible tensor n_dims {n_dims}")
                f.seek(8 * n_dims, 1)   # dims[] - unneeded for offset-delta sizing
                f.seek(4, 1)            # ggml_type - unneeded too
                (offset,) = struct.unpack("<Q", f.read(8))
                entries.append((name, offset))
            data_start = f.tell()
    except (OSError, struct.error, IndexError, UnicodeDecodeError) as exc:
        logger.debug("gguf MoE expert-byte probe: could not parse %s (%s)",
                     path.name, type(exc).__name__)
        return None

    if not entries:
        return None
    remainder = data_start % _GGUF_DEFAULT_ALIGNMENT
    if remainder:
        # Only the LAST tensor's size below depends on this: every other tensor's
        # size comes from the delta to the NEXT tensor's offset, which cancels the
        # alignment padding.
        data_start += _GGUF_DEFAULT_ALIGNMENT - remainder

    entries.sort(key=lambda e: e[1])
    total = 0
    for idx, (name, offset) in enumerate(entries):
        m = _MOE_EXPERT_TENSOR_RE.search(name)
        if not m or int(m.group(1)) >= n_pinned_layers:
            continue
        nxt = entries[idx + 1][1] if idx + 1 < len(entries) else (file_size - data_start)
        size = nxt - offset
        if size > 0:
            total += size
    return total


def _gguf_metadata_probe(path: Path) -> dict:
    """Best-effort read of the GGUF header metadata needed for embedding-model
    detection: ``general.architecture`` and whether any ``*.pooling_type`` key
    is present. Reads only a bounded prefix of the file (see
    ``_GGUF_META_PROBE_BYTES`` - real metadata always precedes the large
    tokenizer vocab arrays and tensor data), never the whole model. Returns
    ``{}`` on any parse failure or truncation within that bound - this NEVER
    raises; an unreadable/unexpected header means 'no signal' and the caller
    falls back to its own classification. A truncation or malformed key AFTER
    a definitive signal was already resolved (e.g. a huge multilingual
    tokenizer vocab array following an early general.architecture="bert" key)
    returns that already-resolved signal: the early-exit below stops as soon
    as either signal is definitively known."""
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
            # v1 used 32-bit tensor/kv counts, so it reports no signal rather than
            # being parsed with the v2+ 64-bit layout.
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
            # Stop as soon as the answer is decided: a definitive embedding
            # architecture, or any pooling_type key. Either one alone is enough,
            # so a truncation past this point cannot discard a confirmed signal.
            if has_pooling_type or architecture in _GGUF_EMBEDDING_ARCHITECTURES:
                break
    except (struct.error, IndexError, UnicodeDecodeError):
        # Truncated within the bounded read, or a malformed layout - fall through
        # and report whatever resolved before the failure.
        pass
    return {"architecture": architecture, "has_pooling_type": has_pooling_type}


def gguf_embedding_signal(path: Path, meta: Optional[dict] = None) -> bool:
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
    ``pull.py`` (a freshly-downloaded remote GGUF).

    *meta*, when given, is an already-computed ``_gguf_metadata_probe(path)``
    result, so a caller that already has one (``_detect_local_model_type``,
    which also needs the architecture string) does not pay for a second read
    of the same bytes. Defaults to None (reads *path* itself)."""
    if meta is None:
        meta = _gguf_metadata_probe(path)
    if meta.get("architecture") in _GGUF_EMBEDDING_ARCHITECTURES:
        return True
    return bool(meta.get("has_pooling_type"))


# llama.cpp's clip.cpp writes this general.architecture value for every
# vision-projector (mmproj) GGUF it exports; projector variants are distinguished
# by a 'clip.projector_type' key instead. Such a file also carries
# general.type='clip-vision' and a 'clip.vision.*'-prefixed metadata block, never
# a '<arch>.*'-prefixed one, so it cannot collide with gguf_embedding_signal's
# architecture check above.
_GGUF_MMPROJ_ARCHITECTURE = "clip"


def gguf_is_mmproj(path: Path, meta: Optional[dict] = None) -> bool:
    """True when *path*'s own GGUF metadata marks it as a vision projector
    (mmproj) rather than a standalone text LLM: its ``general.architecture`` is
    ``"clip"``. Hard metadata baked into the file itself by every llama.cpp
    mmproj export - never a filename guess (contrast ``find_sibling_mmproj``
    in registry.py, which only pairs a model with a co-located projector file
    and is a filename heuristic by necessity). Used by
    ``_detect_local_model_type`` (local add + folder auto-sync) and by
    ``pull.py`` (a freshly-downloaded remote GGUF), the same two call sites as
    ``gguf_embedding_signal``.

    *meta*, when given, is an already-computed ``_gguf_metadata_probe(path)``
    result - see ``gguf_embedding_signal``. Defaults to None (reads *path*
    itself)."""
    if meta is None:
        meta = _gguf_metadata_probe(path)
    return meta.get("architecture") == _GGUF_MMPROJ_ARCHITECTURE


def gguf_registry_metadata(path: Path, meta: Optional[dict] = None) -> dict:
    """Architecture family and MoE expert count for a GGUF file, to persist on
    its registry entry at registration time.

    Returns ``{"architecture": Optional[str], "expert_count": Optional[int]}``.
    Both are None together when the header could not be read/parsed at all
    (unreadable file, bad magic, a v1 GGUF, or a truncated read that never
    reached ``general.architecture``) - genuinely UNKNOWN, never coerced to a
    false "0 experts". ``expert_count`` is 0 (not None) only once
    ``architecture`` was actually resolved, which is the one condition under
    which "no expert_count key" is a confirmed answer.

    *meta*, when given, is an already-computed ``_gguf_metadata_probe(path)``
    result (see ``gguf_embedding_signal``), avoiding a THIRD read of the same
    header when the caller already ran mmproj/embedding detection.
    ``gguf_expert_count`` still does its own separate, equally-bounded read
    regardless: it is a different key scan, not captured by the shared probe."""
    if meta is None:
        meta = _gguf_metadata_probe(path)
    architecture = meta.get("architecture")
    expert_count = gguf_expert_count(path) if architecture is not None else None
    return {"architecture": architecture, "expert_count": expert_count}

