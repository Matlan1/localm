# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded multipart/form-data reading for routes that accept a file upload.

Parses entirely in memory with no third-party dependency. Nothing is written to
disk and the client-supplied file name is never used to build a path: it is
returned as text for the caller to ignore or to display.

``read_form`` reads the request body as a stream and refuses it as soon as it
exceeds the byte limit, so an oversized upload is never fully buffered. Every
refusal raises ``MultipartError`` carrying the HTTP status to answer with.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from email.message import Message
from email.utils import collapse_rfc2231_value
from typing import Optional

MAX_FIELDS = 32
MAX_FIELD_BYTES = 64 * 1024
MAX_HEADER_BYTES = 8 * 1024
MAX_FILENAME_CHARS = 255

_BOUNDARY_RE = re.compile(r"^[0-9A-Za-z'()+_,\-./:=? ]{1,70}$")


class MultipartError(Exception):
    """A refused upload. ``status`` is the HTTP status code to answer with."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class UploadedFile:
    """One file part. ``filename`` has directory components and control
    characters removed and is capped at ``MAX_FILENAME_CHARS``."""
    filename: str
    content_type: str
    data: bytes


@dataclass
class Form:
    """A parsed form: ``fields`` maps a field name to every value sent under it
    (a name may repeat), ``files`` maps a field name to its uploaded files."""
    fields: dict[str, list[str]] = field(default_factory=dict)
    files: dict[str, list[UploadedFile]] = field(default_factory=dict)

    def first(self, name: str) -> Optional[str]:
        """The first value of field ``name``, or None when it was not sent."""
        values = self.fields.get(name)
        return values[0] if values else None


def boundary_of(content_type: str) -> bytes:
    """The boundary token of a ``multipart/form-data`` Content-Type header.

    Raises ``MultipartError`` (415) when the type is not multipart/form-data and
    (400) when the boundary is missing or not a legal RFC 2046 boundary."""
    msg = Message()
    msg["content-type"] = content_type or ""
    if msg.get_content_type() != "multipart/form-data":
        raise MultipartError(
            415, "Expected a multipart/form-data request body "
                 "(Content-Type: multipart/form-data; boundary=...).")
    raw = msg.get_param("boundary")
    boundary = collapse_rfc2231_value(raw) if raw else ""
    if not isinstance(boundary, str) or not _BOUNDARY_RE.match(boundary) \
            or boundary.endswith(" "):
        raise MultipartError(
            400, "The multipart Content-Type has a missing or invalid boundary.")
    return boundary.encode("ascii")


async def read_limited_body(request, limit: int,
                            too_large: Optional[str] = None) -> bytes:
    """Read the request body, raising ``MultipartError`` (413) the moment it
    exceeds ``limit`` bytes, by declared Content-Length or by bytes received.
    ``too_large`` replaces the default refusal text."""
    message = too_large or _too_large(limit)
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise MultipartError(413, message)
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise MultipartError(413, message)
        chunks.append(chunk)
    return b"".join(chunks)


def _too_large(limit: int) -> str:
    return f"Request body too large (max {limit // (1024 * 1024)} MB)."


async def read_form(request, *, max_bytes: int,
                    too_large: Optional[str] = None) -> Form:
    """Read and parse a multipart/form-data request body of at most
    ``max_bytes`` bytes (``too_large`` replaces the 413 text). Raises
    ``MultipartError``."""
    boundary = boundary_of(request.headers.get("content-type", ""))
    body = await read_limited_body(request, max_bytes, too_large)
    return parse_multipart(body, boundary)


def _delimiter_pattern(delim: bytes) -> re.Pattern[bytes]:
    """A pattern for a real delimiter line: ``CRLF + delim`` followed by CRLF,
    ``--`` or transport padding. One C-level scan finds it, so the cost stays
    linear however many false delimiter prefixes the body holds."""
    return re.compile(re.escape(b"\r\n" + delim) + rb"(?=\r\n|--|[ \t])")


def _find_delimiter(body: bytes, pattern: re.Pattern[bytes], start: int) -> int:
    """Index of the next real delimiter line at or after ``start``, or -1."""
    found = pattern.search(body, start)
    return found.start() if found else -1


def _parse_headers(block: bytes) -> Message:
    if len(block) > MAX_HEADER_BYTES:
        raise MultipartError(400, "A multipart part header is too large.")
    msg = Message()
    for line in block.split(b"\r\n"):
        if not line:
            continue
        name, sep, value = line.partition(b":")
        if not sep or not name.strip():
            raise MultipartError(400, "A multipart part header is malformed.")
        msg[name.strip().decode("latin-1").lower()] = (
            value.strip().decode("utf-8", "replace"))
    return msg


def _clean_filename(raw: str) -> str:
    base = re.split(r"[\\/]", raw)[-1]
    base = "".join(ch for ch in base if ch >= " " and ch != "\x7f")
    return base[:MAX_FILENAME_CHARS]


def parse_multipart(body: bytes, boundary: bytes) -> Form:
    """Parse a complete multipart/form-data body.

    Part data is taken byte-exact: only the single CRLF that precedes the next
    delimiter is removed, so binary content ending in CR or LF is preserved.
    Raises ``MultipartError`` (400) for a body that is not well-formed multipart
    and (413) for a non-file field larger than ``MAX_FIELD_BYTES`` or a form with
    more than ``MAX_FIELDS`` parts."""
    delim = b"--" + boundary
    delimiter = _delimiter_pattern(delim)
    if body.startswith(delim):
        pos = 0
    else:
        i = body.find(b"\r\n" + delim)
        if i == -1:
            raise MultipartError(400, "The multipart body has no boundary delimiter.")
        pos = i + 2
    form = Form()
    parts = 0
    while True:
        after = pos + len(delim)
        if body[after:after + 2] == b"--":
            return form
        eol = body.find(b"\r\n", after)
        if eol == -1 or body[after:eol].strip(b" \t"):
            raise MultipartError(400, "The multipart body is malformed.")
        head_start = eol + 2
        if body[head_start:head_start + 2] == b"\r\n":
            header_block, data_start = b"", head_start + 2
        else:
            head_end = body.find(b"\r\n\r\n", head_start)
            if head_end == -1:
                raise MultipartError(400, "The multipart body is malformed.")
            header_block, data_start = body[head_start:head_end], head_end + 4
        nxt = _find_delimiter(body, delimiter, data_start)
        if nxt == -1:
            raise MultipartError(
                400, "The multipart body is truncated (no closing boundary).")
        parts += 1
        if parts > MAX_FIELDS:
            raise MultipartError(413, f"Too many form fields (max {MAX_FIELDS}).")
        _store_part(form, _parse_headers(header_block), body[data_start:nxt])
        pos = nxt + 2


def _store_part(form: Form, headers: Message, data: bytes) -> None:
    disposition = headers.get("content-disposition", "")
    if headers.get_content_disposition() != "form-data" or not disposition:
        raise MultipartError(
            400, "A multipart part is missing its Content-Disposition: form-data header.")
    raw_name = headers.get_param("name", header="content-disposition")
    name = collapse_rfc2231_value(raw_name) if raw_name else ""
    if not isinstance(name, str) or not name:
        raise MultipartError(400, "A multipart part has no field name.")
    filename = headers.get_filename()
    if filename is not None:
        form.files.setdefault(name, []).append(UploadedFile(
            filename=_clean_filename(filename),
            content_type=headers.get_content_type()
            if headers.get("content-type") else "application/octet-stream",
            data=data))
        return
    if len(data) > MAX_FIELD_BYTES:
        raise MultipartError(
            413, f"Form field {name!r} is too large (max {MAX_FIELD_BYTES // 1024} KB).")
    form.fields.setdefault(name, []).append(data.decode("utf-8", "replace"))
