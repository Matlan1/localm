# SPDX-License-Identifier: AGPL-3.0-or-later
"""
In-place metadata removal for generated audio and video files.

ComfyUI's save nodes embed the full submitted workflow (the prompt, and for
ACE-Step the lyrics) as container metadata unless it was launched with
``--disable-metadata``: a FLAC VORBIS_COMMENT block, an MP3 ID3v2 tag, or an
MP4 ``udta``/``meta`` box. ``strip_audio_metadata`` and ``strip_video_metadata``
overwrite those regions with same-size padding structures. The file length and
the position of every other byte are unchanged: MP4 ``stco``/``co64`` chunk
offsets and FLAC seek points are absolute and must stay valid.

Pure Python, no dependencies. Both entry points return "" on a clean strip and
a WARNING string when the container is not one they recognise or the rewrite
failed, mirroring ``localm.image_gen.comfy._strip_png_metadata``.
"""

from __future__ import annotations

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

_ID3V2_FOOTER_FLAG = 0x10
_ID3V1_TAG_SIZE = 128


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


def _neutralise_flac(f: BinaryIO, size: int) -> None:
    """Rewrite every APPLICATION, VORBIS_COMMENT and PICTURE block as a zeroed
    PADDING block of the same length. Raises ValueError on a malformed block
    chain (never writes past the end of the file)."""
    pos = len(_FLAC_MAGIC)
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
        pos += 4 + length
        if is_last:
            return


def _neutralise_mp3(f: BinaryIO, size: int) -> None:
    """Zero the body of a leading ID3v2 tag (keeping its header, so the tag
    becomes pure padding) and the fields of a trailing ID3v1 tag. Raises
    ValueError when the ID3v2 header declares a tag longer than the file."""
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
        _zero(f, tag_size)
    if size >= _ID3V1_TAG_SIZE:
        f.seek(size - _ID3V1_TAG_SIZE)
        if f.read(3) == b"TAG":
            _zero(f, _ID3V1_TAG_SIZE - 3)


def _neutralise_mp4_boxes(f: BinaryIO, start: int, end: int) -> None:
    """Walk the boxes in [start, end), turning every ``udta``/``meta`` box into
    a zeroed ``free`` box of the same size and descending into ``moov`` and
    ``trak``. Raises ValueError on a box that overruns its parent."""
    pos = start
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
        if box_type in _MP4_STRIP_BOXES:
            f.seek(pos + 4)
            f.write(b"free")
            f.seek(pos + header_len)
            _zero(f, box_size - header_len)
        elif box_type in _MP4_CONTAINER_BOXES:
            _neutralise_mp4_boxes(f, pos + header_len, pos + box_size)
        pos += box_size


def _neutralise_mp4(f: BinaryIO, size: int) -> None:
    _neutralise_mp4_boxes(f, 0, size)


def _is_flac(head: bytes) -> bool:
    return head.startswith(_FLAC_MAGIC)


def _is_mp3(head: bytes) -> bool:
    return head.startswith(b"ID3") or (
        len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0)


def _is_mp4(head: bytes) -> bool:
    return head[4:8] == b"ftyp"


_Neutraliser = Callable[[BinaryIO, int], None]

_AUDIO_FORMATS: tuple[tuple[str, Callable[[bytes], bool], _Neutraliser], ...] = (
    ("FLAC", _is_flac, _neutralise_flac),
    ("MP3", _is_mp3, _neutralise_mp3),
)
_VIDEO_FORMATS: tuple[tuple[str, Callable[[bytes], bool], _Neutraliser], ...] = (
    ("MP4", _is_mp4, _neutralise_mp4),
)


def _strip(output_path: Path, formats, kind: str, contents: str) -> str:
    try:
        with open(output_path, "r+b") as f:
            head = f.read(12)
            f.seek(0, 2)
            size = f.tell()
            for _name, sniff, neutralise in formats:
                if sniff(head):
                    neutralise(f, size)
                    return ""
    except (OSError, ValueError) as e:
        return (f"WARNING: could not strip {kind} metadata ({e}); the file may "
                f"still contain the {contents}.")
    names = " or ".join(name for name, _sniff, _neutralise in formats)
    return (f"WARNING: generated file is not {names}; {kind} metadata could not "
            f"be stripped and may still contain the {contents}.")


def strip_audio_metadata(output_path: Path) -> str:
    """Strip container metadata from a generated FLAC or MP3 file, in place.

    Returns "" when the metadata was neutralised, else a WARNING string: the
    file is neither FLAC nor MP3, or the rewrite failed."""
    return _strip(output_path, _AUDIO_FORMATS, "audio", "prompt and lyrics")


def strip_video_metadata(output_path: Path) -> str:
    """Strip container metadata from a generated MP4 file, in place.

    Returns "" when the metadata was neutralised, else a WARNING string: the
    file is not an MP4, or the rewrite failed."""
    return _strip(output_path, _VIDEO_FORMATS, "video", "prompt")
