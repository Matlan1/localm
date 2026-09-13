# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.media.output_metadata: in-place removal of the prompt/workflow
metadata ComfyUI embeds in generated FLAC, MP3 and MP4 files.

The fixtures under tests/fixtures/media were written with PyAV the way
ComfyUI's save nodes write them (see gen_prompt_fixtures.py there) and carry
MARKER inside the embedded workflow JSON, so "the marker is gone" is the same
claim as "the lyrics/prompt are gone"."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from localm.media.output_metadata import strip_audio_metadata, strip_video_metadata

FIXTURES = Path(__file__).parent / "fixtures" / "media"
MARKER = b"SECRET-LYRICS-MARKER-7Q4M"


def _copy(tmp_path: Path, name: str, dest: str | None = None) -> Path:
    dst = tmp_path / (dest or name)
    shutil.copy(FIXTURES / name, dst)
    return dst


# --------------------------------------------------------------------------- #
#  Container walkers (test-side, independent of the implementation)           #
# --------------------------------------------------------------------------- #

def _flac_blocks(data: bytes) -> list[tuple[int, int, int, int]]:
    """[(offset, type, length, is_last), ...] for every metadata block."""
    assert data[:4] == b"fLaC"
    blocks, pos = [], 4
    while True:
        header = data[pos]
        length = int.from_bytes(data[pos + 1:pos + 4], "big")
        blocks.append((pos, header & 0x7F, length, header >> 7))
        pos += 4 + length
        if header >> 7:
            return blocks


def _flac_frames_start(data: bytes) -> int:
    last = _flac_blocks(data)[-1]
    return last[0] + 4 + last[2]


def _mp4_boxes(data: bytes, start: int = 0, end: int | None = None,
               path: str = "") -> list[tuple[str, int, int]]:
    """[(path, offset, size), ...] walking moov and trak like the stripper."""
    end = len(data) if end is None else end
    found, pos = [], start
    while pos + 8 <= end:
        size = int.from_bytes(data[pos:pos + 4], "big")
        typ = data[pos + 4:pos + 8].decode("latin1")
        header = 8
        if size == 1:
            size = int.from_bytes(data[pos + 8:pos + 16], "big")
            header = 16
        elif size == 0:
            size = end - pos
        name = f"{path}/{typ}"
        found.append((name, pos, size))
        if typ in ("moov", "trak"):
            found.extend(_mp4_boxes(data, pos + header, pos + size, name))
        pos += size
    return found


def _box(typ: bytes, payload: bytes, large: bool = False) -> bytes:
    if large:
        return (1).to_bytes(4, "big") + typ + (16 + len(payload)).to_bytes(8, "big") + payload
    return (8 + len(payload)).to_bytes(4, "big") + typ + payload


def _synthetic_mp4(*, top_meta: bool = True, trak_udta: bool = True,
                   large_udta: bool = False, brand: bytes = b"isom") -> bytes:
    """ftyp + moov(mvhd, trak(tkhd, udta?), udta) + meta? + mdat, every
    metadata box carrying MARKER."""
    udta = _box(b"udta", _box(b"meta", b"\x00\x00\x00\x00" + MARKER), large=large_udta)
    trak = _box(b"trak", _box(b"tkhd", b"T" * 20)
                + (_box(b"udta", _box(b"name", MARKER)) if trak_udta else b""))
    moov = _box(b"moov", _box(b"mvhd", b"M" * 24) + trak + udta)
    top = _box(b"meta", b"\x00\x00\x00\x00" + _box(b"ilst", MARKER)) if top_meta else b""
    return (_box(b"ftyp", brand + b"\x00\x00\x02\x00isomiso2mp41") + moov + top
            + _box(b"mdat", b"D" * 40))


# --------------------------------------------------------------------------- #
#  Positive control: the fixtures can fail the tests below                     #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["prompt.flac", "prompt.mp3", "prompt.mp4", "prompt.opus"])
def test_fixture_carries_the_embedded_prompt(name):
    data = (FIXTURES / name).read_bytes()
    assert data.count(MARKER) >= 1
    assert b"prompt" in data


# --------------------------------------------------------------------------- #
#  FLAC                                                                       #
# --------------------------------------------------------------------------- #

def test_flac_vorbis_comment_becomes_zeroed_padding_and_frames_are_untouched(tmp_path):
    path = _copy(tmp_path, "prompt.flac")
    before = path.read_bytes()
    blocks_before = _flac_blocks(before)
    assert any(t == 4 for _, t, _, _ in blocks_before)      # VORBIS_COMMENT present

    assert strip_audio_metadata(path) == ""

    after = path.read_bytes()
    assert MARKER not in after
    assert b"prompt" not in after
    assert len(after) == len(before)
    blocks_after = _flac_blocks(after)
    # Same block layout (offsets, lengths, is_last), only the types changed.
    assert [(o, ln, last) for o, _, ln, last in blocks_after] == \
           [(o, ln, last) for o, _, ln, last in blocks_before]
    for (off, t_before, length, _), (_, t_after, _, _) in zip(blocks_before, blocks_after):
        if t_before == 4:
            assert t_after == 1                                  # PADDING
            assert after[off + 4:off + 4 + length] == b"\x00" * length
        else:
            assert t_after == t_before
            assert after[off:off + 4 + length] == before[off:off + 4 + length]
    start = _flac_frames_start(before)
    assert after[start:] == before[start:]                        # audio frames identical
    assert after[start:start + 2] == b"\xff\xf8"                  # frame sync still there


def test_flac_application_and_picture_blocks_are_neutralised_too(tmp_path):
    data = bytearray((FIXTURES / "prompt.flac").read_bytes())
    blocks = _flac_blocks(bytes(data))
    off, _, length, _ = blocks[1]
    assert blocks[1][1] == 4
    # Retag the vorbis block as APPLICATION, and the padding block as PICTURE
    # carrying the marker.
    data[off] = (data[off] & 0x80) | 2
    p_off, p_type, p_len, p_last = blocks[2]
    assert p_type == 1
    data[p_off] = (data[p_off] & 0x80) | 6
    data[p_off + 4:p_off + 4 + len(MARKER)] = MARKER
    path = tmp_path / "retagged.flac"
    path.write_bytes(bytes(data))

    assert strip_audio_metadata(path) == ""
    after = path.read_bytes()
    assert MARKER not in after
    assert [t for _, t, _, _ in _flac_blocks(after)] == [0, 1, 1]


def test_flac_truncated_block_warns_and_writes_nothing(tmp_path):
    full = (FIXTURES / "prompt.flac").read_bytes()
    off, _, length, _ = _flac_blocks(full)[1]
    truncated = full[:off + 4 + length // 2]
    path = tmp_path / "cut.flac"
    path.write_bytes(truncated)

    warning = strip_audio_metadata(path)

    assert warning.startswith("WARNING: could not strip audio metadata")
    assert "prompt and lyrics" in warning
    assert path.read_bytes() == truncated


# --------------------------------------------------------------------------- #
#  MP3                                                                        #
# --------------------------------------------------------------------------- #

def test_mp3_id3v2_body_and_id3v1_trailer_are_zeroed(tmp_path):
    original = (FIXTURES / "prompt.mp3").read_bytes()
    assert original[:3] == b"ID3"
    trailer = b"TAG" + (b"title " + MARKER).ljust(125, b"\x00")
    path = tmp_path / "track.mp3"
    path.write_bytes(original + trailer)
    before = path.read_bytes()
    tag_size = ((before[6] << 21) | (before[7] << 14) | (before[8] << 7) | before[9])

    assert strip_audio_metadata(path) == ""

    after = path.read_bytes()
    assert MARKER not in after
    assert len(after) == len(before)
    assert after[:5] == b"ID3\x04\x00"                     # header kept
    assert after[10:10 + tag_size] == b"\x00" * tag_size   # tag body is padding
    assert after[10 + tag_size:-128] == before[10 + tag_size:-128]   # audio frames
    assert after[-128:-125] == b"TAG" and after[-125:] == b"\x00" * 125


def test_mp3_without_any_tag_is_left_byte_identical(tmp_path):
    original = (FIXTURES / "prompt.mp3").read_bytes()
    tag_size = ((original[6] << 21) | (original[7] << 14) | (original[8] << 7) | original[9])
    bare = original[10 + tag_size:]
    assert bare[0] == 0xFF and bare[1] & 0xE0 == 0xE0        # starts at frame sync
    path = tmp_path / "bare.mp3"
    path.write_bytes(bare)

    assert strip_audio_metadata(path) == ""
    assert path.read_bytes() == bare


def test_mp3_id3v2_size_past_eof_warns_and_writes_nothing(tmp_path):
    data = bytearray((FIXTURES / "prompt.mp3").read_bytes())
    data[6:10] = bytes([0x7F, 0x7F, 0x7F, 0x7F])
    path = tmp_path / "bad.mp3"
    path.write_bytes(bytes(data))

    warning = strip_audio_metadata(path)

    assert warning.startswith("WARNING: could not strip audio metadata")
    assert path.read_bytes() == bytes(data)


# --------------------------------------------------------------------------- #
#  MP4                                                                        #
# --------------------------------------------------------------------------- #

def test_mp4_udta_becomes_free_and_every_offset_survives(tmp_path):
    path = _copy(tmp_path, "prompt.mp4")
    before = path.read_bytes()
    boxes_before = _mp4_boxes(before)
    names_before = [n for n, _, _ in boxes_before]
    assert "/moov/udta" in names_before
    assert names_before.index("/moov") < names_before.index("/mdat")   # faststart layout

    assert strip_video_metadata(path) == ""

    after = path.read_bytes()
    assert MARKER not in after
    assert b"prompt" not in after
    assert len(after) == len(before)
    boxes_after = _mp4_boxes(after)
    assert [(o, s) for _, o, s in boxes_after] == [(o, s) for _, o, s in boxes_before]
    assert [n.replace("/moov/udta", "/moov/free") for n in names_before] == \
           [n for n, _, _ in boxes_after]
    udta_off, udta_size = next((o, s) for n, o, s in boxes_before if n == "/moov/udta")
    assert after[udta_off + 8:udta_off + udta_size] == b"\x00" * (udta_size - 8)
    for name, off, size in boxes_before:
        if name != "/moov/udta" and not name.startswith("/moov/udta/"):
            if name == "/moov":
                continue                                      # contains the rewritten box
            assert after[off:off + size] == before[off:off + size], name


def test_mp4_track_udta_is_neutralised_and_file_level_meta_is_left_alone(tmp_path):
    data = _synthetic_mp4()
    assert data.count(MARKER) == 3
    path = tmp_path / "clip.mp4"
    path.write_bytes(data)

    assert strip_video_metadata(path) == ""

    after = path.read_bytes()
    assert len(after) == len(data)
    boxes_before = _mp4_boxes(data)
    names = [n for n, _, _ in _mp4_boxes(after)]
    assert names == ["/ftyp", "/moov", "/moov/mvhd", "/moov/trak", "/moov/trak/tkhd",
                     "/moov/trak/free", "/moov/free", "/meta", "/mdat"]
    # Both nested metadata boxes are gone; the file-level meta box (the item
    # table of a HEIF-family file) is byte-identical.
    meta_off, meta_size = next((o, sz) for n, o, sz in boxes_before if n == "/meta")
    assert after[meta_off:meta_off + meta_size] == data[meta_off:meta_off + meta_size]
    assert after.count(MARKER) == 1
    assert after.find(MARKER) > meta_off
    mdat_off = next(o for n, o, _ in boxes_before if n == "/mdat")
    assert after[mdat_off:] == data[mdat_off:]


def test_mp4_heif_family_brand_is_not_treated_as_video(tmp_path):
    data = _synthetic_mp4(brand=b"avif")
    path = tmp_path / "picture.mp4"
    path.write_bytes(data)

    warning = strip_video_metadata(path)

    assert warning == ("WARNING: generated file is not MP4; video metadata could not "
                       "be stripped and may still contain the prompt.")
    assert path.read_bytes() == data


def test_mp4_nesting_deeper_than_the_guard_warns_and_writes_nothing(tmp_path):
    inner = _box(b"udta", MARKER)
    for _ in range(12):
        inner = _box(b"trak", inner)
    data = (_box(b"ftyp", b"isom\x00\x00\x02\x00isom") + _box(b"moov", inner)
            + _box(b"mdat", b"D" * 8))
    path = tmp_path / "deep.mp4"
    path.write_bytes(data)

    warning = strip_video_metadata(path)

    assert warning.startswith("WARNING: could not strip video metadata")
    assert "nested deeper" in warning
    assert path.read_bytes() == data


def test_mp4_64bit_size_udta_keeps_its_large_header(tmp_path):
    data = _synthetic_mp4(top_meta=False, trak_udta=False, large_udta=True)
    path = tmp_path / "large.mp4"
    path.write_bytes(data)
    udta_off, udta_size = next((o, s) for n, o, s in _mp4_boxes(data) if n == "/moov/udta")
    assert data[udta_off:udta_off + 4] == (1).to_bytes(4, "big")

    assert strip_video_metadata(path) == ""

    after = path.read_bytes()
    assert MARKER not in after
    assert after[udta_off:udta_off + 4] == (1).to_bytes(4, "big")
    assert after[udta_off + 4:udta_off + 8] == b"free"
    assert after[udta_off + 8:udta_off + 16] == data[udta_off + 8:udta_off + 16]
    assert after[udta_off + 16:udta_off + udta_size] == b"\x00" * (udta_size - 16)


def test_mp4_box_overrunning_its_parent_warns_and_writes_nothing(tmp_path):
    bad_udta = (500).to_bytes(4, "big") + b"udta" + MARKER
    data = (_box(b"ftyp", b"isom") + _box(b"moov", _box(b"mvhd", b"M" * 8) + bad_udta)
            + _box(b"mdat", b"D" * 8))
    path = tmp_path / "bad.mp4"
    path.write_bytes(data)

    warning = strip_video_metadata(path)

    assert warning.startswith("WARNING: could not strip video metadata")
    assert "prompt" in warning
    assert path.read_bytes() == data


# --------------------------------------------------------------------------- #
#  Unrecognised containers and I/O failures: warn, never touch the bytes       #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,data", [
    ("placeholder.flac", b"FAKEMEDIADATA"),
    ("prompt.opus", None),
])
def test_audio_strip_warns_on_a_container_it_does_not_handle(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data if data is not None else (FIXTURES / name).read_bytes())
    before = path.read_bytes()

    warning = strip_audio_metadata(path)

    assert warning == ("WARNING: generated file is not FLAC or MP3; audio metadata "
                       "could not be stripped and may still contain the prompt and lyrics.")
    assert path.read_bytes() == before


def test_video_strip_warns_on_a_non_mp4(tmp_path):
    path = _copy(tmp_path, "prompt.flac", "clip.mp4")
    before = path.read_bytes()

    warning = strip_video_metadata(path)

    assert warning == ("WARNING: generated file is not MP4; video metadata could not "
                       "be stripped and may still contain the prompt.")
    assert path.read_bytes() == before


def test_missing_file_warns_instead_of_raising(tmp_path):
    missing = tmp_path / "never-written.flac"
    assert strip_audio_metadata(missing).startswith("WARNING: could not strip audio metadata")
    assert strip_video_metadata(missing).startswith("WARNING: could not strip video metadata")


# --------------------------------------------------------------------------- #
#  The stripped files still decode (needs PyAV, the library ComfyUI writes    #
#  with; present on a box with the faster-whisper extra, absent on CI)         #
# --------------------------------------------------------------------------- #

def _decode(path: Path) -> tuple[int, dict]:
    av = pytest.importorskip("av")
    with av.open(str(path)) as container:
        tags = dict(container.metadata)
        frames = sum(1 for stream in container.streams for _ in container.decode(stream))
    return frames, tags


@pytest.mark.parametrize("name,strip", [
    ("prompt.flac", strip_audio_metadata),
    ("prompt.mp3", strip_audio_metadata),
    ("prompt.mp4", strip_video_metadata),
])
def test_stripped_file_still_decodes_with_the_same_frame_count(tmp_path, name, strip):
    frames_before, tags_before = _decode(FIXTURES / name)
    assert "prompt" in tags_before and MARKER.decode() in tags_before["prompt"]
    path = _copy(tmp_path, name)

    assert strip(path) == ""

    frames_after, tags_after = _decode(path)
    assert frames_after == frames_before
    assert "prompt" not in tags_after and "workflow" not in tags_after
