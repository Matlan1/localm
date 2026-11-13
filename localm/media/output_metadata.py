# SPDX-License-Identifier: AGPL-3.0-or-later
"""
In-place metadata removal for generated audio and video files.

ComfyUI's save nodes embed the full submitted workflow (the prompt, and for
ACE-Step the lyrics) as container metadata unless it was launched with
``--disable-metadata``: a FLAC VORBIS_COMMENT block, an MP3 ID3v2 tag, or an
MP4 ``udta``/``meta`` box under ``moov``/``trak``. ``strip_audio_metadata`` and
``strip_video_metadata`` overwrite those regions with same-size padding
structures. The file length and the position of every other byte are
unchanged: MP4 ``stco``/``co64`` chunk offsets and FLAC seek points are
absolute and must stay valid.

PNG images are handled the same way: ``tEXt``, ``zTXt``, ``iTXt`` and ``eXIf``
chunks are dropped, and so are any bytes after the ``IEND`` chunk.

Pure Python, no dependencies. ``strip_output`` is the typed entry point: it
returns a ``StripResult`` that says whether metadata was stripped, there was
nothing to strip, or the strip failed (with the reason). The per-kind
``strip_*_metadata`` functions return "" unless the strip failed, and a WARNING
string when the container is not one they recognise or the rewrite failed.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable

# FLAC metadata block types (RFC 9639 section 8.1).
_FLAC_MAGIC = b"fLaC"
_FLAC_PADDING = 1
_FLAC_APPLICATION = 2
_FLAC_VORBIS_COMMENT = 4
_FLAC_PICTURE = 6
_FLAC_STRIP_TYPES = frozenset({_FLAC_APPLICATION, _FLAC_VORBIS_COMMENT, _FLAC_PICTURE})

# ISO BMFF boxes that carry user metadata (ISO/IEC 14496-12 sections 8.10 and
# 8.11) and the containers they are nested in.
_MP4_STRIP_BOXES = frozenset({b"udta", b"meta"})
_MP4_CONTAINER_BOXES = frozenset({b"moov", b"trak"})
_MP4_MAX_DEPTH = 8
# HEIF-family major brands (ISO/IEC 23008-12): still images whose file-level
# ``meta`` box is the item table, not user metadata.
_HEIF_BRANDS = frozenset({
    b"avif", b"avis", b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx",
    b"hevm", b"hevs", b"mif1", b"msf1", b"miaf",
})

_ID3V2_FOOTER_FLAG = 0x10
_ID3V1_TAG_SIZE = 128

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_PNG_METADATA_CHUNKS = frozenset({b"tEXt", b"zTXt", b"iTXt", b"eXIf"})

_CONTENTS = {"image": "prompt/EXIF data", "audio": "prompt and lyrics",
             "video": "prompt"}


class StripStatus(enum.Enum):
    """Outcome of stripping one generated file."""
    STRIPPED = "stripped"
    NOTHING_TO_STRIP = "nothing_to_strip"
    FAILED = "failed"


@dataclass(frozen=True)
class StripResult:
    """The typed outcome of ``strip_output``: *status*, plus *reason* (the
    WARNING text) when the status is FAILED, else ""."""
    status: StripStatus
    reason: str = ""

    @property
    def failed(self) -> bool:
        return self.status is StripStatus.FAILED

    @property
    def warning(self) -> str:
        """The WARNING text for a failed strip, else ""."""
        return self.reason if self.failed else ""


def _zero(f: BinaryIO, count: int) -> None:
    """Write *count* zero bytes at the current position."""
    chunk = b"\x00" * min(count, 1 << 16)
    while count > 0:
        n = min(count, len(chunk))
        f.write(chunk[:n])
        count -= n


def _syncsafe(raw: bytes) -> int:
    """Decode a 4-byte ID3v2 syncsafe integer (7 bits per byte)."""
    return (raw[0] << 21) | (raw[1] << 14) | (raw[2] << 7) | raw[3]


def _neutralise_flac(f: BinaryIO, size: int) -> int:
    """Rewrite every APPLICATION, VORBIS_COMMENT and PICTURE block as a zeroed
    PADDING block of the same length. Returns the number of blocks rewritten.
    Raises ValueError on a malformed block chain (never writes past the end of
    the file)."""
    pos = len(_FLAC_MAGIC)
    count = 0
    while True:
        f.seek(pos)
        header = f.read(4)
        if len(header) < 4:
            raise ValueError("truncated FLAC metadata block header")
        is_last = header[0] & 0x80
        block_type = header[0] & 0x7F
        length = int.from_bytes(header[1:4], "big")
        if pos + 4 + length > size:
            raise ValueError("FLAC metadata block runs past the end of the file")
        if block_type in _FLAC_STRIP_TYPES:
            f.seek(pos)
            f.write(bytes([is_last | _FLAC_PADDING]))
            f.seek(pos + 4)
            _zero(f, length)
            count += 1
        pos += 4 + length
        if is_last:
            return count


def _neutralise_mp3(f: BinaryIO, size: int) -> int:
    """Zero the body of a leading ID3v2 tag (keeping its header, so the tag
    becomes pure padding) and the fields of a trailing ID3v1 tag. Returns the
    number of tags that held data. Raises ValueError when the ID3v2 header declares a
    tag longer than the file."""
    count = 0
    f.seek(0)
    header = f.read(10)
    if len(header) == 10 and header[:3] == b"ID3":
        tag_size = _syncsafe(header[6:10])
        if 10 + tag_size > size:
            raise ValueError("ID3v2 tag runs past the end of the file")
        # Clear every flag except the footer flag.
        f.seek(5)
        f.write(bytes([header[5] & _ID3V2_FOOTER_FLAG]))
        f.seek(10)
        had_content = any(f.read(tag_size))
        f.seek(10)
        _zero(f, tag_size)
        count += 1 if had_content else 0
    if size >= _ID3V1_TAG_SIZE:
        f.seek(size - _ID3V1_TAG_SIZE)
        if f.read(3) == b"TAG":
            had_content = any(f.read(_ID3V1_TAG_SIZE - 3))
            f.seek(size - _ID3V1_TAG_SIZE + 3)
            _zero(f, _ID3V1_TAG_SIZE - 3)
            count += 1 if had_content else 0
    return count


def _neutralise_mp4_boxes(f: BinaryIO, start: int, end: int, depth: int) -> int:
    """Walk the boxes in [start, end), turning every ``udta``/``meta`` box nested
    inside ``moov``/``trak`` (``depth`` >= 1) into a zeroed ``free`` box of the
    same size. File-level boxes (``depth`` 0) are only descended into, never
    rewritten. Returns the number of boxes rewritten. Raises ValueError on a box
    that overruns its parent or on nesting deeper than ``_MP4_MAX_DEPTH``."""
    if depth > _MP4_MAX_DEPTH:
        raise ValueError(f"MP4 boxes nested deeper than {_MP4_MAX_DEPTH} levels")
    pos = start
    count = 0
    while pos + 8 <= end:
        f.seek(pos)
        header = f.read(8)
        if len(header) < 8:
            raise ValueError("truncated MP4 box header")
        box_size = int.from_bytes(header[:4], "big")
        box_type = header[4:8]
        header_len = 8
        if box_size == 1:
            large = f.read(8)
            if len(large) < 8:
                raise ValueError("truncated MP4 large box header")
            box_size = int.from_bytes(large, "big")
            header_len = 16
        elif box_size == 0:
            box_size = end - pos
        if box_size < header_len or pos + box_size > end:
            raise ValueError(f"malformed MP4 box size for {box_type!r}")
        if depth > 0 and box_type in _MP4_STRIP_BOXES:
            f.seek(pos + 4)
            f.write(b"free")
            f.seek(pos + header_len)
            _zero(f, box_size - header_len)
            count += 1
        elif box_type in _MP4_CONTAINER_BOXES:
            count += _neutralise_mp4_boxes(f, pos + header_len, pos + box_size,
                                           depth + 1)
        pos += box_size
    return count


def _neutralise_mp4(f: BinaryIO, size: int) -> int:
    return _neutralise_mp4_boxes(f, 0, size, 0)


def _is_flac(head: bytes) -> bool:
    return head.startswith(_FLAC_MAGIC)


def _is_mp3(head: bytes) -> bool:
    return head.startswith(b"ID3") or (
        len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0)


def _is_mp4(head: bytes) -> bool:
    return head[4:8] == b"ftyp" and head[8:12] not in _HEIF_BRANDS


_Neutraliser = Callable[[BinaryIO, int], int]

_AUDIO_FORMATS: tuple[tuple[str, Callable[[bytes], bool], _Neutraliser], ...] = (
    ("FLAC", _is_flac, _neutralise_flac),
    ("MP3", _is_mp3, _neutralise_mp3),
)
_VIDEO_FORMATS: tuple[tuple[str, Callable[[bytes], bool], _Neutraliser], ...] = (
    ("MP4", _is_mp4, _neutralise_mp4),
)


def _strip(output_path: Path, formats, kind: str, contents: str) -> StripResult:
    try:
        with open(output_path, "r+b") as f:
            head = f.read(12)
            f.seek(0, 2)
            size = f.tell()
            for _name, sniff, neutralise in formats:
                if sniff(head):
                    count = neutralise(f, size)
                    return StripResult(StripStatus.STRIPPED if count
                                       else StripStatus.NOTHING_TO_STRIP)
    except Exception as e:  # noqa: BLE001
        return StripResult(StripStatus.FAILED, (
            f"WARNING: could not strip {kind} metadata ({e}); the file may "
            f"still contain the {contents}."))
    names = " or ".join(name for name, _sniff, _neutralise in formats)
    return StripResult(StripStatus.FAILED, (
        f"WARNING: generated file is not {names}; {kind} metadata could not "
        f"be stripped and may still contain the {contents}."))


def _neutralise_png(f: BinaryIO, size: int) -> int:
    """Drop every text and EXIF chunk from the PNG in *f*, and any bytes after
    its IEND chunk, in place. Returns the number of chunks dropped, counting
    trailing bytes as one. Raises ValueError, writing nothing, when a chunk runs
    past the end of the file or the file has no IEND chunk."""
    f.seek(0)
    data = f.read()
    clean = bytearray(_PNG_MAGIC)
    pos, dropped = len(_PNG_MAGIC), 0
    while True:
        if pos + 12 > len(data):
            raise ValueError("PNG ends without an IEND chunk")
        length = int.from_bytes(data[pos:pos + 4], "big")
        total = 12 + length
        if pos + total > len(data):
            raise ValueError("PNG chunk runs past the end of the file")
        ctype = data[pos + 4:pos + 8]
        if ctype in _PNG_METADATA_CHUNKS:
            dropped += 1
        else:
            clean.extend(data[pos:pos + total])
        pos += total
        if ctype == b"IEND":
            break
    if pos < len(data):
        dropped += 1
    if dropped:
        f.seek(0)
        f.write(bytes(clean))
        f.truncate()
    return dropped


def _is_png(head: bytes) -> bool:
    return head.startswith(_PNG_MAGIC)


_IMAGE_FORMATS: tuple[tuple[str, Callable[[bytes], bool], _Neutraliser], ...] = (
    ("PNG", _is_png, _neutralise_png),
)

_KIND_FORMATS = {"image": _IMAGE_FORMATS, "audio": _AUDIO_FORMATS,
                 "video": _VIDEO_FORMATS}


def strip_output(kind: str, output_path: Path) -> StripResult:
    """Strip the metadata ComfyUI embedded in a generated file, in place.

    *kind* is ``"image"`` (PNG), ``"audio"`` (FLAC or MP3) or ``"video"`` (MP4).
    Returns a ``StripResult``: STRIPPED when metadata was neutralised,
    NOTHING_TO_STRIP when the container is recognised and carried none, FAILED
    (with the reason) when the file is a container this kind does not handle,
    is unreadable or malformed (a block, box or chunk that runs past the end of
    the file, or a PNG without an IEND chunk), or the rewrite failed. Raises
    KeyError for an unknown *kind*.
    """
    return _strip(output_path, _KIND_FORMATS[kind], kind, _CONTENTS[kind])


def strip_image_metadata(output_path: Path) -> str:
    """Strip PNG text and EXIF chunks from a generated image, in place.

    Returns "" unless the strip failed, else a WARNING string: the file is not a
    PNG, is malformed, or the rewrite failed."""
    return strip_output("image", output_path).warning


def strip_audio_metadata(output_path: Path) -> str:
    """Strip container metadata from a generated FLAC or MP3 file, in place.

    Returns "" unless the strip failed, else a WARNING string: the file is
    neither FLAC nor MP3, or the rewrite failed."""
    return strip_output("audio", output_path).warning


def strip_video_metadata(output_path: Path) -> str:
    """Strip container metadata from a generated MP4 file, in place.

    Returns "" unless the strip failed, else a WARNING string: the file is not
    an MP4, or the rewrite failed."""
    return strip_output("video", output_path).warning
