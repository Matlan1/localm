# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fuzz the RAG document extractors and chunker.

A document reaches ``extract_bytes`` from a chat attachment or an indexed
folder, so its bytes and its filename are attacker-controlled. The contract:
the only exception that escapes is ``ExtractError``, the call returns promptly,
and it never allocates in proportion to a size the file declares."""
from __future__ import annotations

import bz2
import gzip
import io
import lzma
import tarfile
import zipfile

import pytest

pytest.importorskip("hypothesis")

from hypothesis import example, given, strategies as st  # noqa: E402

from localm.rag import chunk as rag_chunk  # noqa: E402
from localm.rag import extract  # noqa: E402
from tests.fuzz import _bounds  # noqa: E402

_NAMES = st.one_of(
    st.sampled_from(["a.txt", "a.md", "a.zip", "a.tar", "a.tgz", "a.tar.gz", "a.gz",
                     "a.bz2", "a.xz", "a.docx", "a.ipynb", "a.pdf", "a.html", "a.json",
                     "a.csv", "a.png", "a.bin", "noext", ".hidden", "a.zip.gz"]),
    st.text(max_size=20),
)

_member_names = st.one_of(
    st.sampled_from(["a.txt", "dir/b.md", "../evil.txt", "/abs.txt", ".git/x.txt",
                     "node_modules/x.js", "word/document.xml", "n.ipynb", "x.pdf",
                     "nested.zip", "nested.tar.gz"]),
    st.text(max_size=16),
)

_member_bodies = st.one_of(
    st.binary(max_size=200),
    st.text(max_size=200).map(lambda s: s.encode("utf-8")),
    st.just(b'{"cells": [{"source": ["x"], "cell_type": "code"}]}'),
    st.just(b"<html><body><p>hi</p></body></html>"),
)


@st.composite
def zip_bytes(draw):
    buf = io.BytesIO()
    method = draw(st.sampled_from([zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED]))
    members = draw(st.dictionaries(_member_names, _member_bodies, max_size=5))
    with zipfile.ZipFile(buf, "w", method) as zf:
        for member_name, body in members.items():
            zf.writestr(member_name, body)
    return buf.getvalue()


@st.composite
def tar_bytes(draw):
    buf = io.BytesIO()
    mode = draw(st.sampled_from(["w", "w:gz", "w:bz2", "w:xz"]))
    with tarfile.open(fileobj=buf, mode=mode) as tf:
        for _ in range(draw(st.integers(0, 5))):
            body = draw(_member_bodies)
            info = tarfile.TarInfo(draw(_member_names))
            info.size = len(body)
            info.type = draw(st.sampled_from([tarfile.REGTYPE] * 4 + [tarfile.SYMTYPE,
                                                                       tarfile.DIRTYPE]))
            info.linkname = draw(st.sampled_from(["", "../x", "/etc/passwd"]))
            tf.addfile(info, io.BytesIO(body) if info.type == tarfile.REGTYPE else None)
    return buf.getvalue()


@st.composite
def stream_bytes(draw):
    body = draw(st.one_of(_member_bodies, tar_bytes(), zip_bytes()))
    codec = draw(st.sampled_from([gzip.compress, bz2.compress, lzma.compress]))
    return codec(body)


@st.composite
def docx_bytes(draw):
    buf = io.BytesIO()
    xml = draw(st.one_of(
        st.just('<w:p><w:r><w:t>hello</w:t></w:r></w:p>'),
        st.text(alphabet="<>/wpt:r =\"&;#x0123456789abc", max_size=120),
    ))
    with zipfile.ZipFile(buf, "w", draw(st.sampled_from([zipfile.ZIP_STORED,
                                                       zipfile.ZIP_DEFLATED]))) as zf:
        zf.writestr("word/document.xml", xml)
    data = bytearray(buf.getvalue())
    for _ in range(draw(st.integers(0, 3))):
        local, central = data.find(b"PK\x03\x04"), data.find(b"PK\x01\x02")
        field = draw(st.sampled_from([(local + 6, central + 8), (local + 8, central + 10),
                                      (local + 14, central + 16)]))
        value = draw(st.integers(0, 255))
        for at in field[:draw(st.integers(1, 2))]:
            if 0 <= at < len(data):
                data[at] = value
    if draw(st.integers(0, 3)) == 0 and data:
        data[draw(st.integers(0, len(data) - 1))] = draw(st.integers(0, 255))
    return bytes(data)


_PDF_TEMPLATE = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R"
    b"/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
    b"4 0 obj<</Length 44>>stream\nBT /F1 12 Tf 20 100 Td (Hello fuzz) Tj ET\nendstream endobj\n"
    b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"trailer<</Root 1 0 R/Size 6>>\nstartxref\n0\n%%EOF\n"
)


@st.composite
def pdf_bytes(draw):
    data = bytearray(_PDF_TEMPLATE)
    for _ in range(draw(st.integers(0, 6))):
        kind = draw(st.integers(0, 4))
        if kind == 0 and data:
            data[draw(st.integers(0, len(data) - 1))] = draw(st.integers(0, 255))
        elif kind == 1 and data:
            del data[draw(st.integers(0, len(data) - 1)):]
        elif kind == 2:
            at = draw(st.integers(0, len(data)))
            data[at:at] = draw(st.binary(max_size=24))
        elif kind == 3:
            data = bytearray(bytes(data).replace(
                b"/Count 1", b"/Count " + str(draw(st.sampled_from(
                    [0, 2, 10**9, 2**63, 10**30]))).encode()))
        else:
            data = bytearray(bytes(data).replace(b"3 0 R]", b"2 0 R 2 0 R]"))
    return bytes(data)


@st.composite
def ipynb_bytes(draw):
    return draw(st.one_of(
        st.just(b"[" * 5000 + b"]" * 5000),
        st.just(b'{"cells": ' + b"[" * 3000 + b"]" * 3000 + b"}"),
        st.text(max_size=80).map(lambda s: s.encode("utf-8")),
        st.recursive(
            st.none() | st.booleans() | st.integers() | st.text(max_size=8),
            lambda inner: st.lists(inner, max_size=4)
            | st.dictionaries(st.sampled_from(["cells", "source", "cell_type"]), inner,
                              max_size=3),
            max_leaves=15).map(lambda v: __import__("json").dumps(v).encode()),
    ))


_documents = st.tuples(
    st.one_of(st.binary(max_size=300), zip_bytes(), tar_bytes(), stream_bytes(),
              docx_bytes(), pdf_bytes(), ipynb_bytes()),
    _NAMES,
)


@given(doc=_documents)
@example(doc=(b"[" * 5000 + b"]" * 5000, "a.ipynb"))
def test_extract_bytes_raises_only_extract_error(doc):
    data, name = doc
    try:
        out = _bounds.returns_within(extract.extract_bytes, data, name, seconds=20)
    except extract.ExtractError:
        return
    assert isinstance(out, str)
    assert len(out) <= extract.MAX_TEXT_CHARS


@given(data=st.one_of(pdf_bytes(), zip_bytes(), stream_bytes()))
def test_extract_allocation_is_not_proportional_to_declared_sizes(data):
    result, peak = _bounds.peak_allocation(extract.extract_bytes, data, "x.pdf")
    assert peak < 200 * 1024 * 1024, f"allocated {peak} bytes for {len(data)} input bytes"
    assert not isinstance(result, BaseException) or isinstance(result, extract.ExtractError)


@given(text=st.text(max_size=600), name=_NAMES)
def test_classify_and_sniff_never_raise(text, name):
    label = extract.classify_format(text, name)
    assert isinstance(label, str) and label
    extract.sniff_text_format(text)


@given(data=st.binary(max_size=300), name=_NAMES)
def test_sniff_format_never_raises(data, name):
    extract.sniff_format(data, name)


@given(
    text=st.text(alphabet=st.sampled_from(["a", "b", " ", "\n", "\r", "é"]),
                 max_size=2000),
    chunk_chars=st.integers(-3, 80),
    overlap=st.integers(-3, 200),
)
def test_chunk_text_terminates_with_bounded_chunks(text, chunk_chars, overlap):
    try:
        chunks = _bounds.returns_within(
            rag_chunk.chunk_text, text, chunk_chars=chunk_chars, overlap=overlap)
    except ValueError:
        assert chunk_chars < 1 and text.strip()
        return
    assert len(chunks) <= max(1, len(text))
    for c in chunks:
        assert isinstance(c["text"], str) and c["pos"] >= 1
