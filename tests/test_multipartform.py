# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.multipartform: bounded, dependency-free multipart/form-data parsing."""

import asyncio

import httpx
import pytest

from localm.multipartform import (
    MAX_FIELD_BYTES, MAX_FIELDS, MAX_FILENAME_CHARS, MultipartError,
    boundary_of, parse_multipart, read_form, read_limited_body,
)


def _encode(files=None, data=None):
    """The exact bytes httpx (what the openai SDK sends with) emits."""
    req = httpx.Request("POST", "http://x/", files=files, data=data)
    return req.read(), req.headers["content-type"]


def _parse(files=None, data=None):
    body, ctype = _encode(files, data)
    return parse_multipart(body, boundary_of(ctype))


def _manual(parts: list[bytes], boundary=b"BOUND") -> bytes:
    out = b""
    for p in parts:
        out += b"--" + boundary + b"\r\n" + p + b"\r\n"
    return out + b"--" + boundary + b"--\r\n"


class _FakeRequest:
    def __init__(self, chunks, headers=None):
        self._chunks = chunks
        self.headers = headers or {}
        self.consumed = 0

    async def stream(self):
        for c in self._chunks:
            self.consumed += 1
            yield c


class TestRoundTrip:
    def test_fields_and_file(self):
        form = _parse(files={"file": ("a.wav", b"RIFFdata", "audio/wav")},
                      data={"model": "whisper-1", "language": "en"})
        assert form.first("model") == "whisper-1"
        assert form.first("language") == "en"
        up = form.files["file"][0]
        assert (up.filename, up.content_type, up.data) == (
            "a.wav", "audio/wav", b"RIFFdata")

    def test_repeated_field_keeps_every_value_in_order(self):
        form = _parse(files={"file": ("a.wav", b"x", "audio/wav")},
                      data={"timestamp_granularities[]": ["word", "segment"]})
        assert form.fields["timestamp_granularities[]"] == ["word", "segment"]

    def test_binary_data_is_byte_exact_including_trailing_crlf_and_boundary_text(self):
        payload = b"\r\n\x00\xff--x\r\n--BOUND not a delimiter\r\n"
        form = _parse(files={"file": ("a.bin", payload, "application/octet-stream")})
        assert form.files["file"][0].data == payload

    def test_empty_file_part_is_kept_empty(self):
        form = _parse(files={"file": ("a.wav", b"", "audio/wav")})
        assert form.files["file"][0].data == b""

    def test_utf8_field_value_decodes(self):
        form = _parse(files={"file": ("a.wav", b"x", "audio/wav")},
                      data={"prompt": "h\u00e9llo \u4e16\u754c"})
        assert form.first("prompt") == "h\u00e9llo \u4e16\u754c"

    def test_missing_field_is_none(self):
        form = _parse(files={"file": ("a.wav", b"x", "audio/wav")}, data={"a": "1"})
        assert form.first("b") is None

    def test_preamble_before_first_delimiter_is_ignored(self):
        body = (b"this is a preamble\r\n"
                + _manual([b'Content-Disposition: form-data; name="a"\r\n\r\n1']))
        assert parse_multipart(body, b"BOUND").first("a") == "1"

    def test_part_without_headers_block_end_is_refused(self):
        body = b'--BOUND\r\nContent-Disposition: form-data; name="a"'
        with pytest.raises(MultipartError) as e:
            parse_multipart(body, b"BOUND")
        assert e.value.status == 400


class TestFilenames:
    @pytest.mark.parametrize("raw, expected", [
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\x\\evil.wav", "evil.wav"),
        ("a/b\\c.wav", "c.wav"),
        ("nul\x00byte.wav", "nulbyte.wav"),
        ("tab\tname.wav", "tabname.wav"),
    ])
    def test_directories_and_control_characters_are_removed(self, raw, expected):
        body = _manual([
            b'Content-Disposition: form-data; name="file"; filename="'
            + raw.encode("utf-8") + b'"\r\n\r\ndata'])
        up = parse_multipart(body, b"BOUND").files["file"][0]
        assert up.filename == expected
        assert "/" not in up.filename and "\\" not in up.filename

    def test_long_filename_is_capped(self):
        body = _manual([
            b'Content-Disposition: form-data; name="file"; filename="'
            + b"a" * 1000 + b'.wav"\r\n\r\ndata'])
        up = parse_multipart(body, b"BOUND").files["file"][0]
        assert len(up.filename) == MAX_FILENAME_CHARS

    def test_rfc2231_filename_star_is_decoded(self):
        body = _manual([
            b"Content-Disposition: form-data; name=\"file\"; "
            b"filename*=UTF-8''caf%C3%A9.wav\r\n\r\ndata"])
        assert parse_multipart(body, b"BOUND").files["file"][0].filename == "caf\u00e9.wav"

    def test_missing_content_type_defaults_to_octet_stream(self):
        body = _manual([
            b'Content-Disposition: form-data; name="file"; filename="a"\r\n\r\nd'])
        assert (parse_multipart(body, b"BOUND").files["file"][0].content_type
                == "application/octet-stream")


class TestBoundary:
    def test_quoted_boundary(self):
        assert boundary_of('multipart/form-data; boundary="a b-c"') == b"a b-c"

    def test_unquoted_boundary(self):
        assert boundary_of("multipart/form-data; boundary=xyz123") == b"xyz123"

    def test_case_insensitive_type(self):
        assert boundary_of("Multipart/Form-Data; Boundary=abc") == b"abc"

    @pytest.mark.parametrize("ctype", [
        "", "application/json", "text/plain", "multipart/mixed; boundary=a"])
    def test_wrong_type_is_415(self, ctype):
        with pytest.raises(MultipartError) as e:
            boundary_of(ctype)
        assert e.value.status == 415

    @pytest.mark.parametrize("ctype", [
        "multipart/form-data", "multipart/form-data; boundary=",
        "multipart/form-data; boundary=" + "a" * 71,
        'multipart/form-data; boundary="bad\\nchar"',
        'multipart/form-data; boundary="trailing "'])
    def test_missing_or_illegal_boundary_is_400(self, ctype):
        with pytest.raises(MultipartError) as e:
            boundary_of(ctype)
        assert e.value.status == 400


class TestDelimiterScan:
    def _part(self, data: bytes) -> bytes:
        return _manual([b'Content-Disposition: form-data; name="file"; '
                        b'filename="a"\r\n\r\n' + data])

    def test_a_delimiter_prefix_inside_the_data_does_not_end_the_part(self):
        data = b"x\r\n--BOUNDARYX more\r\n--BOUNDx\r\n--BOUND-\r\nend"
        form = parse_multipart(self._part(data), b"BOUND")
        assert form.files["file"][0].data == data

    def test_a_delimiter_followed_by_transport_padding_ends_the_part(self):
        body = (b'--BOUND\r\nContent-Disposition: form-data; name="a"\r\n\r\n1'
                b"\r\n--BOUND \t\r\n"
                b'Content-Disposition: form-data; name="b"\r\n\r\n2'
                b"\r\n--BOUND--\r\n")
        form = parse_multipart(body, b"BOUND")
        assert (form.first("a"), form.first("b")) == ("1", "2")

    def test_a_body_made_of_false_delimiters_is_refused_quickly(self):
        import time
        head = b'--a\r\nContent-Disposition: form-data; name="x"\r\n\r\n'
        body = head + b"\r\n--ab" * (26 * 1024 * 1024 // 6)
        started = time.monotonic()
        with pytest.raises(MultipartError) as e:
            parse_multipart(body, b"a")
        assert e.value.status == 400
        assert time.monotonic() - started < 1.5


class TestMalformedBodies:
    @pytest.mark.parametrize("body", [b"", b"garbage", b"--BOUND", b"\r\n"])
    def test_no_delimiter_or_nothing_after_it_is_400(self, body):
        with pytest.raises(MultipartError) as e:
            parse_multipart(body, b"BOUND")
        assert e.value.status == 400

    def test_truncated_body_is_400(self):
        body, ctype = _encode(files={"file": ("a.wav", b"x" * 100, "audio/wav")})
        with pytest.raises(MultipartError) as e:
            parse_multipart(body[:-20], boundary_of(ctype))
        assert e.value.status == 400

    def test_part_without_disposition_is_400(self):
        body = _manual([b"Content-Type: text/plain\r\n\r\nx"])
        with pytest.raises(MultipartError) as e:
            parse_multipart(body, b"BOUND")
        assert e.value.status == 400

    def test_part_without_name_is_400(self):
        body = _manual([b"Content-Disposition: form-data\r\n\r\nx"])
        with pytest.raises(MultipartError) as e:
            parse_multipart(body, b"BOUND")
        assert e.value.status == 400

    def test_malformed_header_line_is_400(self):
        body = _manual([b"no colon here\r\n\r\nx"])
        with pytest.raises(MultipartError) as e:
            parse_multipart(body, b"BOUND")
        assert e.value.status == 400

    def test_oversized_part_header_is_400(self):
        body = _manual([b'Content-Disposition: form-data; name="a"\r\nX-Pad: '
                        + b"p" * 9000 + b"\r\n\r\nx"])
        with pytest.raises(MultipartError) as e:
            parse_multipart(body, b"BOUND")
        assert e.value.status == 400

    def test_oversized_text_field_is_413(self):
        body = _manual([b'Content-Disposition: form-data; name="a"\r\n\r\n'
                        + b"x" * (MAX_FIELD_BYTES + 1)])
        with pytest.raises(MultipartError) as e:
            parse_multipart(body, b"BOUND")
        assert e.value.status == 413

    def test_text_field_at_the_limit_is_accepted(self):
        body = _manual([b'Content-Disposition: form-data; name="a"\r\n\r\n'
                        + b"x" * MAX_FIELD_BYTES])
        assert len(parse_multipart(body, b"BOUND").first("a")) == MAX_FIELD_BYTES

    def test_too_many_parts_is_413(self):
        part = b'Content-Disposition: form-data; name="a"\r\n\r\nx'
        with pytest.raises(MultipartError) as e:
            parse_multipart(_manual([part] * (MAX_FIELDS + 1)), b"BOUND")
        assert e.value.status == 413

    def test_exactly_max_parts_is_accepted(self):
        part = b'Content-Disposition: form-data; name="a"\r\n\r\nx'
        form = parse_multipart(_manual([part] * MAX_FIELDS), b"BOUND")
        assert len(form.fields["a"]) == MAX_FIELDS


class TestReadLimitedBody:
    def test_declared_length_over_limit_is_refused_before_reading(self):
        req = _FakeRequest([b"x" * 10], {"content-length": "5000"})
        with pytest.raises(MultipartError) as e:
            asyncio.run(read_limited_body(req, 1000))
        assert e.value.status == 413
        assert req.consumed == 0

    def test_streamed_bytes_over_limit_stop_the_read_early(self):
        chunks = [b"x" * 400] * 100
        req = _FakeRequest(chunks)
        with pytest.raises(MultipartError) as e:
            asyncio.run(read_limited_body(req, 1000))
        assert e.value.status == 413
        assert req.consumed == 3

    def test_body_at_the_limit_is_returned(self):
        req = _FakeRequest([b"a" * 500, b"b" * 500], {"content-length": "1000"})
        assert asyncio.run(read_limited_body(req, 1000)) == b"a" * 500 + b"b" * 500

    def test_lying_content_length_cannot_exceed_the_limit(self):
        req = _FakeRequest([b"x" * 800, b"x" * 800], {"content-length": "10"})
        with pytest.raises(MultipartError) as e:
            asyncio.run(read_limited_body(req, 1000))
        assert e.value.status == 413

    def test_read_form_parses_a_streamed_upload(self):
        body, ctype = _encode(files={"file": ("a.wav", b"abc", "audio/wav")},
                              data={"model": "m"})
        req = _FakeRequest([body[:30], body[30:]], {"content-type": ctype})
        form = asyncio.run(read_form(req, max_bytes=10_000))
        assert form.first("model") == "m"
        assert form.files["file"][0].data == b"abc"

    def test_read_form_refuses_non_multipart_without_reading_the_body(self):
        req = _FakeRequest([b"{}"], {"content-type": "application/json"})
        with pytest.raises(MultipartError) as e:
            asyncio.run(read_form(req, max_bytes=10_000))
        assert e.value.status == 415
        assert req.consumed == 0
