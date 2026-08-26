# SPDX-License-Identifier: AGPL-3.0-or-later
"""`_sha256_file` reads large files through a background reader thread one
block ahead of the hasher (see `_iter_file_blocks`). The digest is what
verifies a downloaded model, so the properties that matter are that the
threaded path returns a BYTE-IDENTICAL digest to the inline one, that a read
error is re-raised in the caller's thread rather than swallowed, and that an
abandoned consumer does not leak the reader thread.

Every test here shrinks `_HASH_THREAD_MIN_BYTES` / `_HASH_BLOCK_BYTES` so the
threaded path runs on a few KB instead of needing a 32MB fixture. Without that
shrink these tests would silently exercise the INLINE path and assert nothing
about the threading at all - which is the failure mode they exist to rule out,
so `test_the_threshold_patch_actually_selects_the_threaded_path` pins that the
shrink works before the rest lean on it.
"""

from __future__ import annotations

import hashlib
import threading

import pytest

import localm.model_manager as mm


@pytest.fixture
def threaded(monkeypatch):
    """Force the threaded path with a 256-byte block on any non-empty file."""
    monkeypatch.setattr(mm, "_HASH_THREAD_MIN_BYTES", 1)
    monkeypatch.setattr(mm, "_HASH_BLOCK_BYTES", 256)
    monkeypatch.setattr(mm, "_HASH_READAHEAD_BLOCKS", 2)


@pytest.fixture
def inline(monkeypatch):
    """Force the inline path, whatever the file size."""
    monkeypatch.setattr(mm, "_HASH_THREAD_MIN_BYTES", 1 << 40)
    monkeypatch.setattr(mm, "_HASH_BLOCK_BYTES", 256)


def _write(tmp_path, name, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_the_threshold_patch_actually_selects_the_threaded_path(tmp_path, threaded):
    """The other tests are only about threading if the patch really routes
    there. Pin it by observing that a reader thread exists while blocks are
    being produced - not by trusting the constant."""
    data = b"x" * 4096
    p = _write(tmp_path, "m.gguf", data)
    seen_reader = []
    for _ in mm._iter_file_blocks(p):
        if any(t.name == "localm-sha256-reader" for t in threading.enumerate()):
            seen_reader.append(True)
            break
    assert seen_reader, "no localm-sha256-reader thread: the inline path ran"


def test_inline_and_threaded_digests_are_identical(tmp_path, monkeypatch):
    # Switching paths does not change one byte of the digest.
    data = bytes(range(256)) * 97          # 24832 bytes, not a block multiple
    p = _write(tmp_path, "m.gguf", data)
    expected = hashlib.sha256(data).hexdigest()

    monkeypatch.setattr(mm, "_HASH_THREAD_MIN_BYTES", 1 << 40)
    monkeypatch.setattr(mm, "_HASH_BLOCK_BYTES", 256)
    inline_digest = mm._sha256_file(p)

    monkeypatch.setattr(mm, "_HASH_THREAD_MIN_BYTES", 1)
    monkeypatch.setattr(mm, "_HASH_READAHEAD_BLOCKS", 2)
    threaded_digest = mm._sha256_file(p)

    assert inline_digest == expected
    assert threaded_digest == expected


@pytest.mark.parametrize("size", [0, 1, 255, 256, 257, 512, 4095, 4096])
def test_digest_correct_across_block_boundaries(tmp_path, threaded, size):
    """Off-by-one at a block edge is the classic chunked-read bug, so walk the
    boundary: one under, exactly on, and one over a 256-byte block."""
    data = bytes((i * 7) % 256 for i in range(size))
    p = _write(tmp_path, f"m{size}.gguf", data)
    assert mm._sha256_file(p) == hashlib.sha256(data).hexdigest()


def test_read_error_reaches_the_caller(tmp_path, threaded, monkeypatch):
    """A failure inside the reader thread must surface, never vanish: a digest
    computed over a partial read would report a model as verified when it was
    never fully read."""
    p = _write(tmp_path, "m.gguf", b"y" * 4096)
    real_open = open
    calls = {"n": 0}

    def exploding_open(*a, **kw):
        calls["n"] += 1
        f = real_open(*a, **kw)
        real_read = f.read

        def read(*ra, **rkw):
            if calls["n"] and f.tell() >= 512:
                raise OSError(5, "simulated device error")
            return real_read(*ra, **rkw)

        f.read = read
        return f

    monkeypatch.setattr("builtins.open", exploding_open)
    with pytest.raises(OSError) as ei:
        mm._sha256_file(p)
    assert "simulated device error" in str(ei.value)


def test_abandoning_the_generator_does_not_leak_the_reader(tmp_path, threaded):
    """A consumer that stops early (its own exception, or a progress callback
    that raised despite its contract) leaves the reader parked on a full queue.
    The generator's finally must drain it, or a long-running server accumulates
    a stuck thread holding megabytes per abandoned hash."""
    p = _write(tmp_path, "m.gguf", b"z" * 65536)      # 256 blocks, queue holds 2
    before = {t.ident for t in threading.enumerate()}

    gen = mm._iter_file_blocks(p)
    next(gen)                                          # start it, then walk away
    gen.close()

    deadline = threading.Event()
    deadline.wait(0.5)
    leaked = [t for t in threading.enumerate()
              if t.ident not in before and t.name == "localm-sha256-reader"]
    assert leaked == [], f"reader thread leaked: {leaked}"


def test_a_reader_that_dies_silently_raises_instead_of_hanging(tmp_path, threaded,
                                                               monkeypatch):
    """If the reader exits without posting eof or an exception, the consumer
    must FAIL, not block forever: a silent forever-hang inside the function that
    verifies a downloaded model is worse than a loud error.

    The assertion names the EXACT exception type. `pytest.raises(Exception)` would
    pass on a `queue.Empty` leaking out, and this test is specifically about which
    failure occurs."""
    p = _write(tmp_path, "m.gguf", b"v" * 4096)

    class _NeverRuns:
        """A thread that reports itself dead and never runs its target."""
        def __init__(self, *a, **kw):
            self.name = kw.get("name")

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(mm.gguf.threading, "Thread", _NeverRuns)

    with pytest.raises(RuntimeError) as ei:
        mm._sha256_file(p)
    assert "without delivering data or an error" in str(ei.value)
    assert "m.gguf" in str(ei.value)


def test_progress_is_monotonic_and_ends_at_total(tmp_path, threaded):
    data = b"q" * 4096
    p = _write(tmp_path, "m.gguf", data)
    seen: list[tuple[int, int]] = []
    digest = mm._sha256_file(p, progress=lambda d, t: seen.append((d, t)))

    assert digest == hashlib.sha256(data).hexdigest()
    assert seen, "progress was never called"
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)
    assert seen[-1] == (len(data), len(data))
    assert all(t == len(data) for _, t in seen)


def test_progress_runs_on_the_callers_thread_not_the_reader(tmp_path, threaded):
    """_hash_with_progress calls _sha256_file from a worker thread and drives a
    rich progress bar from this callback. If the callback were invoked on the
    reader thread instead, that contract would change silently."""
    p = _write(tmp_path, "m.gguf", b"w" * 4096)
    caller = threading.current_thread().ident
    threads: set[int] = set()
    mm._sha256_file(p, progress=lambda d, t: threads.add(
        threading.current_thread().ident))
    assert threads == {caller}
