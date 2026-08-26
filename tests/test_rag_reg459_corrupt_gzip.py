# SPDX-License-Identifier: AGPL-3.0-or-later
"""A corrupt or truncated gzip raises ExtractError rather than escaping raw.

extract_bytes routes any tar-family suffix / gzip-sniffed input to
_extract_tar_or_stream. tarfile's gzopen reads the first tar block through a
GzipFile; on a TRUNCATED stream GzipFile raises EOFError, and gzopen converts
only OSError to ReadError - EOFError is not an OSError. An unguarded EOFError
(or zlib.error for a mid-block corruption) would propagate past every
`except ExtractError` guard in the callers:

  - store.py's folder index records a failing file in failed[] and CONTINUES;
    an unhandled EOFError instead breaks the per-file loop and aborts the whole
    index run.
  - plug.py's /api/rag/extract upload would answer 500 rather than 422.

ExtractError means "this ONE file could not be read", so a bad member does not
take down the batch around it.
"""

from __future__ import annotations

import bz2
import gzip
import lzma

import pytest

from localm.rag.extract import ExtractError, extract_bytes


def _truncated_gzip() -> bytes:
    """A stream with the gzip magic whose deflate body stops early - e.g. a
    partially-downloaded notes.txt.gz."""
    full = gzip.compress(b"hello world " * 100)
    return full[: len(full) // 2]


class TestCorruptCompressedStreamsRaiseExtractError:
    def test_truncated_gzip_raises_extracterror(self):
        with pytest.raises(ExtractError):
            extract_bytes(_truncated_gzip(), "notes.txt.gz")

    def test_truncated_gzip_named_as_a_tar_raises_extracterror(self):
        # The .tar.gz suffix routes to the same tar-family handler.
        with pytest.raises(ExtractError):
            extract_bytes(_truncated_gzip(), "archive.tar.gz")

    def test_truncated_tgz_raises_extracterror(self):
        with pytest.raises(ExtractError):
            extract_bytes(_truncated_gzip(), "archive.tgz")

    def test_corrupt_gzip_body_raises_extracterror(self):
        # Valid header + magic, garbage deflate body -> zlib.error rather than
        # EOFError. Also not an OSError, so also not converted by gzopen.
        full = gzip.compress(b"hello world " * 100)
        corrupt = full[:12] + b"\x00\xff\x00\xff" * 8 + full[-8:]
        with pytest.raises(ExtractError):
            extract_bytes(corrupt, "notes.txt.gz")

    def test_empty_gzip_magic_only_raises_extracterror(self):
        with pytest.raises(ExtractError):
            extract_bytes(b"\x1f\x8b", "notes.txt.gz")

    @pytest.mark.parametrize("name", ["notes.txt.bz2", "notes.txt.xz"])
    def test_sibling_compression_formats_stay_extracterror(self, name):
        """Control: the .bz2/.xz openers convert their own EOF to ReadError, so
        these also surface as ExtractError."""
        raw = b"hello world " * 100
        full = bz2.compress(raw) if name.endswith(".bz2") else lzma.compress(raw)
        with pytest.raises(ExtractError):
            extract_bytes(full[: len(full) // 2], name)


class TestValidArchivesStillExtract:
    """NEGATIVE CASE: a well-formed stream still extracts, so the broadened
    error guard has not turned into 'every compressed input fails'."""

    def test_valid_single_gzip_still_extracts_its_text(self):
        out = extract_bytes(gzip.compress(b"hello gzip world"), "notes.txt.gz")
        assert "hello gzip world" in out

    def test_valid_tar_gz_still_extracts_its_member_text(self):
        import io
        import tarfile

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            payload = b"hello tar member"
            info = tarfile.TarInfo("inner.txt")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
        out = extract_bytes(buf.getvalue(), "archive.tar.gz")
        assert "hello tar member" in out
