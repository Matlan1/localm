# SPDX-License-Identifier: AGPL-3.0-or-later
"""Paragraph-aware text chunking for retrieval."""

from __future__ import annotations

# Targets chosen for small local models: ~1200 chars ≈ 300 tokens per chunk,
# so 4 retrieved chunks cost ~1200 tokens of a 4096-token default window.
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150


def chunk_text(text: str, *, chunk_chars: int = CHUNK_CHARS,
               overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split *text* into chunks of roughly *chunk_chars* characters.

    Paragraph boundaries (blank lines) are preferred split points; a paragraph
    longer than the target is split at line breaks, then hard-wrapped. Each
    chunk dict carries ``text`` and ``pos`` (1-based line number of its start
    in the original document) so citations can point somewhere useful.
    """
    if not text.strip():
        return []
    # A non-positive chunk size makes the wrap loop below (len(line) >
    # chunk_chars -> line = line[chunk_chars - overlap:]) never shrink `line`.
    if chunk_chars < 1:
        raise ValueError("chunk_chars must be >= 1")
    overlap = max(0, min(overlap, chunk_chars // 2))

    # Build (line_no, paragraph) pairs
    paragraphs: list[tuple[int, str]] = []
    line_no = 1
    for raw in text.split("\n\n"):
        # An ODD number of blank lines between paragraphs leaves a leading '\n'
        # on this split element; those newlines precede this paragraph's text, so
        # pos advances past them.
        lead = len(raw) - len(raw.lstrip("\n"))
        stripped = raw.strip("\n")
        if stripped.strip():
            paragraphs.append((line_no + lead, stripped))
        line_no += raw.count("\n") + 2   # the paragraph's lines + blank line

    # Oversized paragraphs are pre-split so the packer below stays simple
    pieces: list[tuple[int, str]] = []
    for ln, para in paragraphs:
        if len(para) <= chunk_chars:
            pieces.append((ln, para))
            continue
        offset = ln
        current = ""
        for line in para.split("\n"):
            while len(line) > chunk_chars:          # pathological single line
                pieces.append((offset, line[:chunk_chars]))
                line = line[chunk_chars - overlap:]
            if current and len(current) + len(line) + 1 > chunk_chars:
                pieces.append((offset, current))
                offset += current.count("\n") + 1
                current = line
            else:
                current = f"{current}\n{line}" if current else line
        if current:
            pieces.append((offset, current))

    # Pack pieces into chunks, carrying a tail overlap between neighbours
    chunks: list[dict] = []
    buf: list[tuple[int, str]] = []
    size = 0
    for ln, piece in pieces:
        if buf and size + len(piece) + 2 > chunk_chars:
            chunks.append({"pos": buf[0][0],
                           "text": "\n\n".join(p for _, p in buf)})
            tail = buf[-1]
            buf = [tail] if len(tail[1]) <= overlap else []
            size = sum(len(p) for _, p in buf)
        buf.append((ln, piece))
        size += len(piece) + 2
    if buf:
        chunks.append({"pos": buf[0][0],
                       "text": "\n\n".join(p for _, p in buf)})
    return chunks
