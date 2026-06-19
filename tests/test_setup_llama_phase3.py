# SPDX-License-Identifier: AGPL-3.0-or-later
"""SEC-8 regression: the native-binary download had no integrity check.

``localm setup-llama`` downloaded a prebuilt release zip and extracted +
pip-installed it verbatim. A truncated transfer, a MITM'd payload, or an HTML
error page served with a 200 would all be unpacked and installed, with the only
guard being a post-hoc BadZipFile catch on extraction (too late, and no defense
against a zip that is the wrong artifact).

The fix adds artifact validation BEFORE extraction/install:

  1. size: reject a trivially small / empty download (likely an error page),
  2. shape: reject anything that is not a valid zip (zipfile.is_zipfile),
  3. provenance (opt-in): when an expected sha256 is provided, verify it and
     refuse on mismatch.

The size + valid-zip checks are always on; the sha256 is opt-in (we do not
hardcode a brittle hash for the live URL). These tests pin each adversarial case
(non-zip body, too-small file, sha256 mismatch) is rejected with no extraction
or install, and that a valid zip with a matching (or absent) hash proceeds.
"""

from __future__ import annotations

import hashlib
import io
import os
import zipfile
from pathlib import Path

import pytest

from localm import setup_llama

# Comfortably above setup_llama._MIN_ARTIFACT_BYTES so a "valid" fixture zip is
# never mistaken for a too-small body. Stored uncompressed (random bytes) so the
# archive size is deterministic regardless of the floor's exact value.
_BIG = setup_llama._MIN_ARTIFACT_BYTES * 4


def _make_zip(payload_size: int = _BIG) -> bytes:
    """A real, valid zip with one member large enough to clear the size floor.
    The member is incompressible random data stored uncompressed, so the on-disk
    archive size tracks *payload_size* rather than collapsing under DEFLATE."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("llama.dll", os.urandom(payload_size))
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# _validate_archive: the always-on guard (size + valid-zip) and opt-in sha256. #
# --------------------------------------------------------------------------- #


def test_validate_rejects_non_zip(tmp_path):
    """An HTML error page (HTTP 200 served instead of the zip) is not a zip and
    must be rejected before anyone tries to extract it. Sized above the floor so
    this targets the valid-zip branch, not the size branch."""
    p = tmp_path / "dl.zip"
    body = b"<!DOCTYPE html><html><body>404 Not Found</body></html>\n" * 10000
    assert len(body) > setup_llama._MIN_ARTIFACT_BYTES
    p.write_bytes(body)
    with pytest.raises(setup_llama.ArtifactError, match="not a valid zip"):
        setup_llama._validate_archive(p)


def test_validate_rejects_truncated_zip(tmp_path):
    """A zip truncated mid-transfer fails the is_zipfile structural check."""
    good = _make_zip()
    p = tmp_path / "dl.zip"
    # Keep it large enough to clear the size floor, but lop off the trailing
    # central directory so it is no longer a structurally valid zip.
    p.write_bytes(good[: len(good) // 2])
    with pytest.raises(setup_llama.ArtifactError, match="not a valid zip"):
        setup_llama._validate_archive(p)


def test_validate_rejects_too_small(tmp_path):
    """A tiny body (an empty file, or a one-line error) is rejected by the size
    floor before the zip check even runs."""
    p = tmp_path / "dl.zip"
    p.write_bytes(b"nope")
    with pytest.raises(setup_llama.ArtifactError, match="too small"):
        setup_llama._validate_archive(p)


def test_validate_rejects_empty(tmp_path):
    """A zero-byte download (connection dropped immediately) is rejected."""
    p = tmp_path / "dl.zip"
    p.write_bytes(b"")
    with pytest.raises(setup_llama.ArtifactError, match="too small"):
        setup_llama._validate_archive(p)


def test_validate_rejects_sha256_mismatch(tmp_path):
    """A valid zip whose sha256 does not match the pin is refused: this is the
    MITM / wrong-artifact case the opt-in pin defends against."""
    data = _make_zip()
    p = tmp_path / "dl.zip"
    p.write_bytes(data)
    wrong = "0" * 64
    with pytest.raises(setup_llama.ArtifactError, match="sha256"):
        setup_llama._validate_archive(p, expected_sha256=wrong)


def test_validate_accepts_valid_zip_no_hash(tmp_path):
    """The happy path with no pin: a real, non-trivial zip validates."""
    p = tmp_path / "dl.zip"
    p.write_bytes(_make_zip())
    setup_llama._validate_archive(p)   # no raise


def test_validate_accepts_matching_sha256(tmp_path):
    """A valid zip whose sha256 matches the pin validates."""
    data = _make_zip()
    p = tmp_path / "dl.zip"
    p.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    setup_llama._validate_archive(p, expected_sha256=digest)   # no raise


def test_validate_sha256_case_insensitive(tmp_path):
    """A pin given in uppercase (or with surrounding whitespace) still matches:
    operators paste hashes from many sources."""
    data = _make_zip()
    p = tmp_path / "dl.zip"
    p.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest().upper()
    setup_llama._validate_archive(p, expected_sha256="  " + digest + "  ")   # no raise


# --------------------------------------------------------------------------- #
# main(): the validation gate runs BEFORE extraction/install on the download   #
# path. We mock the network and the extract/install side effects.              #
# --------------------------------------------------------------------------- #


class _Trace:
    def __init__(self):
        self.extracted = False
        self.installed = False


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    """Point the runtime-lib target at a temp dir and stub out the heavy /
    side-effecting operations so we can assert whether they ran."""
    target = tmp_path / "lib"
    target.mkdir()
    monkeypatch.setattr(setup_llama, "_repo_runtime_lib", lambda: target)
    monkeypatch.setattr(setup_llama, "_runtime_pkg_dir", lambda: tmp_path / "pkg")
    monkeypatch.setattr(setup_llama, "_verify", lambda: None)
    monkeypatch.setattr(setup_llama, "_ensure_importable", lambda: None)
    # Provisioning now load-tests the native library in a subprocess; against a
    # stub llama.dll that would always fail (and trigger a fallback re-download),
    # so we treat the load as successful here. The fallback path itself is
    # covered explicitly in test_cuda_setup.py.
    monkeypatch.setattr(setup_llama, "_native_loads_ok", lambda: (True, ""))
    # Pretend we are on Windows so the default-URL download path is taken even
    # when the test host is not win32.
    monkeypatch.setattr(setup_llama.sys, "platform", "win32")

    trace = _Trace()

    def fake_copy(src_dir, tgt):
        # Simulate a successful extraction: drop the expected lib in place.
        trace.extracted = True
        (tgt / "llama.dll").write_bytes(b"copied")
        return 1

    def fake_install(pkg_dir):
        trace.installed = True
        return True

    monkeypatch.setattr(setup_llama, "_copy_binaries", fake_copy)
    monkeypatch.setattr(setup_llama, "_install_runtime_wheel", fake_install)
    return target, trace


def _stub_download(monkeypatch, payload: bytes):
    """Make _download write *payload* to dest instead of hitting the network."""
    def fake_download(url, dest):
        Path(dest).write_bytes(payload)
    monkeypatch.setattr(setup_llama, "_download", fake_download)


def _run(args):
    from click.testing import CliRunner
    return CliRunner().invoke(setup_llama.main, args)


def test_main_rejects_non_zip_download_no_extract(wired, monkeypatch):
    """A 200-with-an-error-page download must NOT be extracted or installed.
    Sized above the floor so this exercises the valid-zip branch specifically,
    not the size branch (covered separately below)."""
    target, trace = wired
    bogus = b"<!DOCTYPE html><html><body>upstream error</body></html>\n" * 10000
    assert len(bogus) > setup_llama._MIN_ARTIFACT_BYTES
    _stub_download(monkeypatch, bogus)
    result = _run([])
    assert result.exit_code != 0
    assert not trace.extracted
    assert not trace.installed
    assert not (target / "llama.dll").exists()


def test_main_rejects_too_small_download(wired, monkeypatch):
    """A tiny truncated body is rejected before extraction."""
    target, trace = wired
    _stub_download(monkeypatch, b"x")
    result = _run([])
    assert result.exit_code != 0
    assert not trace.extracted
    assert not trace.installed


def test_main_rejects_mismatched_sha256(wired, monkeypatch):
    """A valid zip whose body does not match the --sha256 pin is refused, and
    nothing is extracted or installed."""
    target, trace = wired
    _stub_download(monkeypatch, _make_zip())
    result = _run(["--sha256", "0" * 64])
    assert result.exit_code != 0
    assert not trace.extracted
    assert not trace.installed


def test_main_accepts_valid_zip(wired, monkeypatch):
    """The happy path: a valid, non-trivial zip is extracted and installed."""
    target, trace = wired
    _stub_download(monkeypatch, _make_zip())
    result = _run([])
    assert result.exit_code == 0, result.output
    assert trace.extracted
    assert trace.installed
    assert (target / "llama.dll").exists()


def test_main_accepts_matching_sha256(wired, monkeypatch):
    """A valid zip whose body matches the --sha256 pin proceeds normally."""
    target, trace = wired
    data = _make_zip()
    _stub_download(monkeypatch, data)
    digest = hashlib.sha256(data).hexdigest()
    result = _run(["--sha256", digest])
    assert result.exit_code == 0, result.output
    assert trace.extracted
    assert trace.installed


# --------------------------------------------------------------------------- #
# --from / --url: a user-pinned build that provisions but does NOT load must    #
# exit non-zero (never report success on a broken runtime), not warn + exit 0.  #
# --------------------------------------------------------------------------- #


def test_from_load_failure_exits_nonzero(wired, monkeypatch, tmp_path):
    target, trace = wired
    monkeypatch.setattr(setup_llama, "_native_loads_ok", lambda: (False, "wrong arch"))
    src = tmp_path / "src"
    src.mkdir()
    result = _run(["--from", str(src)])
    assert result.exit_code != 0
    assert "did not load" in result.output


def test_from_load_success_reports_ok(wired, monkeypatch, tmp_path):
    target, trace = wired   # fixture stubs _native_loads_ok -> (True, "")
    src = tmp_path / "src"
    src.mkdir()
    result = _run(["--from", str(src)])
    assert result.exit_code == 0, result.output
    assert "loads on this machine" in result.output


def test_url_load_failure_exits_nonzero(wired, monkeypatch):
    target, trace = wired
    _stub_download(monkeypatch, _make_zip())
    monkeypatch.setattr(setup_llama, "_native_loads_ok", lambda: (False, "missing dep"))
    result = _run(["--url", "https://example.test/llama.zip"])
    assert result.exit_code != 0
    assert "did not load" in result.output
