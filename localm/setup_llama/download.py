# SPDX-License-Identifier: AGPL-3.0-or-later
"""Downloading a prebuilt archive, validating it (size, archive shape, optional
sha256) and extracting it without letting a member escape the destination.
"""

from __future__ import annotations

import hashlib
import socket
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from localm.http_ssl import RedirectDowngradeRefused
from localm.setup_llama._common import console
import localm.setup_llama as _sl

# A prebuilt llama runtime archive is many megabytes. Anything below this
# floor is almost certainly an error page, a redirect stub, or a truncated
# transfer, never the real artifact. This is the always-on lower bound; the
# valid-archive structural check below is the second always-on guard. A pinned
# sha256 is the opt-in third guard (we do not hardcode a brittle hash for the
# live URLs, which move with every upstream release).
_MIN_ARTIFACT_BYTES = 256 * 1024   # 256 KiB


# Per-read socket timeout for the archive download. The chunked urlopen read loop
# honours the default socket timeout as an idle (between-reads) deadline, NOT a
# total-transfer cap, so a large-but-progressing download is never killed; only a
# genuinely stalled connection (no bytes for this many seconds) trips it. This
# turns an indefinite hang on a dropped/throttled transfer into a clear, loud
# error the caller reports, instead of a frozen progress line with no diagnostic.
_DOWNLOAD_STALL_TIMEOUT = 60   # seconds


@dataclass
class _DownloadResult:
    """What actually happened on the wire, captured for diagnosis - never
    guessed after the fact. ``content_length`` is 0 when the server sent none
    (completeness could not be checked structurally); ``final_url`` is the URL
    after following redirects (a proxy that redirects to its own block page
    shows up here even when the request otherwise looked normal)."""
    bytes_received: int
    content_length: int
    content_type: str
    final_url: str


class ArtifactError(Exception):
    """A downloaded artifact failed integrity validation (size, archive shape,
    or a provided sha256 pin) and must NOT be extracted or installed."""


# --------------------------------------------------------------------------- #
#  Download / validate / extract                                              #
# --------------------------------------------------------------------------- #

def _download(url: str, dest: Path) -> _DownloadResult:
    """Stream *url* to *dest*, capturing what actually happened on the wire (not
    just whether it succeeded) so a caller can diagnose a bad result from real
    evidence instead of a guess. Distinguishes three distinct failure shapes,
    each reported with its own specific cause:

    * a STALL (no bytes for ``_DOWNLOAD_STALL_TIMEOUT``s) - the connection is
      alive but frozen;
    * a transport-level drop mid-transfer (connection reset, broken pipe, ...) -
      the connection died outright, with however many bytes had arrived so far;
    * a CLEAN completion that is nonetheless short of what was promised - the
      server (or something between it and us) considers the response finished,
      it is just not the archive. This third case is NOT an error here - it is
      returned normally and diagnosed by the caller once the file is on disk,
      because "too short" alone does not yet say WHY (see
      :func:`_diagnose_bad_artifact`)."""
    console.print(f"[dim]Downloading {url}[/dim]")
    last = [-1]

    def _report(nread: int, total: int) -> None:
        if total <= 0:
            return
        pct = min(100, nread * 100 // total)
        if pct != last[0] and pct % 5 == 0:
            last[0] = pct
            mb = total / 1024 ** 2
            console.print(f"[dim]  {pct:3d}%  ({mb:.0f} MB)[/dim]", end="\r")

    prev_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(_DOWNLOAD_STALL_TIMEOUT)
    total = 0
    nread = 0
    try:
        # verified_urlopen (see localm/http_ssl.py) follows the GitHub -> release-CDN
        # 302 and verifies both hops. "Over HTTPS" is ENFORCED, not assumed: its
        # default HttpsOnlyRedirect refuses a redirect off https and raises
        # RedirectDowngradeRefused, handled below. Until that guard existed this
        # comment asserted a property nothing checked, on the one download whose
        # bytes become a loaded native DLL - and _validate_archive's digest check
        # is opt-in (its expected_sha256, i.e. --sha256), so an unpinned archive
        # has no cryptographic check on its content either. Stream in chunks so a
        # multi-hundred-MB archive is never held in memory; the default socket
        # timeout is the between-reads stall deadline (not a total cap).
        req = urllib.request.Request(url, headers={"User-Agent": "localm-setup-llama"})
        with _sl.verified_urlopen(req, timeout=_DOWNLOAD_STALL_TIMEOUT) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            content_type = r.headers.get("Content-Type") or ""
            # geturl() is standard on every real urllib response, but stay
            # defensive for the rare test double that does not implement it -
            # the final URL is a nice-to-have diagnostic, not load-bearing.
            try:
                final_url = r.geturl() or url
            except Exception:
                final_url = url
            while True:
                chunk = r.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                nread += len(chunk)
                _report(nread, total)
    except RedirectDowngradeRefused as e:
        # BEFORE the OSError clause below, which it would otherwise hit
        # (RedirectDowngradeRefused is a URLError, and URLError is an OSError):
        # that clause tells the user this "looks like a dropped or flaky
        # connection" and to retry. Retrying an attempt to hand us a native DLL
        # over cleartext is the opposite of the right advice, so the two are
        # never collapsed into one message.
        raise ArtifactError(
            f"refused to follow this download off HTTPS ({e}) - the archive "
            "would have arrived in cleartext, where anything on the network "
            "path can replace it, and its bytes are loaded as a native "
            "library. This is not a transient network fault: check the URL, "
            "or provision from a local build with 'localm setup-llama --from "
            "<build-dir>'."
        ) from e
    except (socket.timeout, TimeoutError) as e:
        raise ArtifactError(
            f"download stalled (no data for {_DOWNLOAD_STALL_TIMEOUT}s, after "
            f"{nread} of {total or 'an unknown number of'} bytes) - the "
            "connection was interrupted or throttled. Retry on a stable network, "
            "or provision from a local build with 'localm setup-llama --from "
            "<build-dir>' / '--url <archive-url>'."
        ) from e
    except OSError as e:
        # A live transport failure mid-stream (connection reset, broken pipe, a
        # proxy dropping the connection outright) - distinct from a download
        # that completes normally but turns out short (that is not an
        # exception at all; see the docstring). Report the partial state
        # honestly instead of a generic "download failed".
        raise ArtifactError(
            f"the connection was interrupted after {nread} of "
            f"{total or 'an unknown number of'} bytes ({e}) - this looks like a "
            "dropped or flaky connection, not a blocked download. Retry, or "
            "provision from a local build with 'localm setup-llama --from "
            "<build-dir>' / '--url <archive-url>'."
        ) from e
    finally:
        socket.setdefaulttimeout(prev_timeout)
    console.print()
    return _DownloadResult(bytes_received=nread, content_length=total,
                           content_type=content_type, final_url=final_url)


def _sha256_file(path: Path) -> str:
    """Stream the file through sha256 so a multi-hundred-MB artifact is not
    read into memory at once."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_supported_archive(path: Path) -> bool:
    return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)


def _sniff_content_kind(path: Path, peek: int = 4096) -> str:
    """Classify what the file's own bytes actually look like, independent of
    what it was supposed to be - the one honest way to tell a substituted
    HTML/JSON response apart from a genuinely truncated archive. The content
    itself is authoritative here: a header or URL can be wrong, spoofed, or
    just uninformative, but a real llama.cpp archive's opening bytes never
    decode as text.

    Returns one of: 'empty', 'zip_truncated', 'gzip_truncated', 'html', 'xml',
    'json', 'text', 'binary'. The two '..._truncated' results specifically mean
    the file STARTS with a real archive's magic bytes but is not (yet, or ever
    going to be) a complete one - a different cause than a substituted page."""
    try:
        with open(path, "rb") as f:
            head = f.read(peek)
    except OSError:
        return "binary"
    if not head:
        return "empty"
    if head[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
        return "zip_truncated"
    if head[:2] == b"\x1f\x8b":
        return "gzip_truncated"
    # Structural markers are pure ASCII and sit at/near the very start of a
    # real error/block page regardless of the page's OVERALL encoding, so look
    # for them with a lossy decode first (never raises, so it still finds
    # HTML/JSON/XML served as e.g. windows-1252, not just UTF-8). A strict
    # decode is only needed for the weaker 'text vs binary' distinction below.
    lossy = head.decode("ascii", errors="replace").lstrip()
    lower = lossy[:200].lower()
    if lower.startswith("<!doctype html") or lower.startswith("<html") or "<html" in lower:
        return "html"
    if lower.startswith("<?xml") or "<error>" in lower:
        return "xml"
    if lossy[:1] in ("{", "["):
        return "json"
    try:
        head.decode("utf-8")
        return "text"
    except UnicodeDecodeError:
        return "binary"


def _diagnose_bad_artifact(path: Path, dl: Optional["_DownloadResult"]) -> str:
    """Turn what the bytes on disk actually look like - plus, when available,
    what the response claimed (:func:`_download`'s result for this same file) -
    into ONE specific, evidence-backed explanation. Never states a cause the
    evidence does not actually support: the fallback case says 'not clear'
    plainly instead of picking the most likely-sounding story."""
    kind = _sniff_content_kind(path)
    try:
        size = path.stat().st_size
    except OSError:
        size = 0

    if kind == "empty":
        cause = "nothing was received at all"
    elif kind == "html":
        cause = ("the response is an HTML page, not the archive - almost always "
                 "a network that blocks or filters this download (a corporate "
                 "proxy or security product), not a problem with the release itself")
    elif kind in ("json", "xml"):
        cause = (f"the response is {kind.upper()}, not the archive - most likely "
                 "an error response from a proxy or the CDN standing in for the "
                 "real file (again typically a corporate network filter)")
    elif kind in ("zip_truncated", "gzip_truncated"):
        cause = ("the response starts like the real archive but cuts off "
                 "partway through - this looks like a genuinely interrupted "
                 "transfer (a dropped or throttled connection), not a deliberate "
                 "block")
    else:
        cause = ("the content does not clearly indicate the cause - it is "
                 "neither a recognisable webpage nor a valid archive")

    detail_bits = [f"{size} bytes received"]
    if dl is not None:
        detail_bits.append(f"{dl.content_length} expected (Content-Length)"
                           if dl.content_length else "no Content-Length given")
        if dl.content_type:
            detail_bits.append(f"Content-Type: {dl.content_type}")
        if dl.final_url:
            detail_bits.append(f"final URL: {dl.final_url}")
    return f"{cause} ({'; '.join(detail_bits)})."


def _validate_archive(
    path: Path,
    expected_sha256: Optional[str] = None,
    min_size: int = _MIN_ARTIFACT_BYTES,
    dl: Optional[_DownloadResult] = None,
) -> None:
    """Validate a downloaded artifact BEFORE it is extracted or installed.
    Raises :class:`ArtifactError` on any failure.

    Three checks, in cheapest-first order:
      1. size: a real prebuilt runtime archive is many MB; a tiny/empty body is
         an error page, a redirect stub, or a truncated transfer (always on).
      2. shape: it must be a structurally valid zip OR tar archive, so a
         200-with-HTML or a half-transferred file is rejected before we hand it
         to extraction (always on).
      3. provenance: when *expected_sha256* is given, the file's digest must
         match it (opt-in; refuses on mismatch). Comparison is whitespace- and
         case-insensitive so a pasted hash from any source works.

    *dl*, when given (the :func:`_download` result for this same file), lets
    checks 1 and 2 explain WHY from real evidence - what the bytes actually
    look like, plus what the response claimed - instead of a generic hedge
    that names every possible cause without saying which one actually
    happened (see :func:`_diagnose_bad_artifact`).
    """
    try:
        size = path.stat().st_size
    except OSError as e:
        raise ArtifactError(f"could not stat downloaded file: {e}") from e
    if size < min_size:
        raise ArtifactError(
            f"download is too small ({size} bytes < {min_size} minimum): "
            f"{_diagnose_bad_artifact(path, dl)}"
        )
    if not _is_supported_archive(path):
        raise ArtifactError(
            f"download is not a valid zip or tar archive: "
            f"{_diagnose_bad_artifact(path, dl)}"
        )
    if expected_sha256:
        want = expected_sha256.strip().lower()
        got = _sha256_file(path)
        if got != want:
            raise ArtifactError(
                "download sha256 does not match the expected pin "
                f"(expected {want}, got {got}). Refusing to install a "
                "possibly tampered or wrong artifact."
            )


def _safe_extractall_tar(tf: tarfile.TarFile, dest: Path) -> None:
    """Path-traversal-safe tar extraction for Python < 3.12, which has no
    extraction ``filter`` keyword. Backports the 'data' filter's core guarantee:
    every member, and every symlink/hardlink TARGET, must resolve INSIDE *dest*.
    Absolute, drive-letter, ``..`` and escaping-link members are refused. On
    Python 3.12+ ``filter="data"`` is used directly (see _extract_archive), so
    this is only the older-interpreter path - but it must be just as safe."""
    dest_resolved = dest.resolve()

    def _contained(p: Path) -> bool:
        rp = p.resolve()
        return rp == dest_resolved or dest_resolved in rp.parents

    for m in tf.getmembers():
        name = m.name
        if name.startswith(("/", "\\")) or ".." in Path(name).parts \
                or (len(name) > 1 and name[1] == ":"):
            raise ArtifactError(f"unsafe path in archive: {name!r}")
        if not _contained(dest / name):
            raise ArtifactError(f"unsafe path in archive: {name!r}")
        if m.issym() or m.islnk():
            link = m.linkname
            if link.startswith(("/", "\\")) or (len(link) > 1 and link[1] == ":") \
                    or not _contained(dest / Path(name).parent / link):
                raise ArtifactError(
                    f"unsafe link target in archive: {name!r} -> {link!r}")
    tf.extractall(dest)


def _extract_archive(path: Path, dest: Path) -> None:
    """Extract a validated zip or tar.gz into *dest*, refusing any member that
    would escape *dest* (an absolute path, a drive letter, or a ``..`` segment).
    Tar uses the 'data' filter (Python 3.12+), or a hand-rolled equivalent
    (_safe_extractall_tar) on older interpreters, for the same guarantee."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            for n in zf.namelist():
                if n.startswith(("/", "\\")) or ".." in Path(n).parts \
                        or (len(n) > 1 and n[1] == ":"):
                    raise ArtifactError(f"unsafe path in archive: {n!r}")
            zf.extractall(dest)
        return
    with tarfile.open(path) as tf:
        try:
            tf.extractall(dest, filter="data")     # py3.12+: path-traversal safe
        except TypeError:
            _safe_extractall_tar(tf, dest)         # py<3.12: same guarantee, by hand


def _human_mb(nbytes) -> str:
    try:
        return f"{int(nbytes) / 1024 ** 2:.0f} MB"
    except Exception:
        return "?"


def _fetch_and_place(url: str, target: Path, sha256: Optional[str] = None) -> int:
    """Download -> validate -> extract -> copy one prebuilt archive into
    *target*. Returns the number of binary files copied. Raises on a download or
    validation failure (the caller decides fatal-vs-fallback)."""
    with tempfile.TemporaryDirectory() as tmp:
        suffix = ".zip" if url.lower().endswith(".zip") else ".tar.gz"
        arc = Path(tmp) / f"llama-prebuilt{suffix}"
        dl = _sl._download(url, arc)
        _validate_archive(arc, expected_sha256=sha256, dl=dl)   # validation gate, pre-extract
        ex = Path(tmp) / "x"
        _extract_archive(arc, ex)
        return _sl._copy_binaries(ex, target)


def _fetch_verified(url: str, target: Path, sha: Optional[str], what: str = "release asset") -> None:
    """Fetch + place an archive, WARNING honestly when no checksum is available
    to verify it. A GitHub asset can publish no `digest`, and the offline hash
    table only covers the tags pins.py pins, so the provenance check can be
    skipped - and when it is, it is never skipped in silence. Size +
    archive-shape checks still apply either way.

    Which path is exposed changed with the pin: _PINNED_TAG's own assets ARE in
    that table, so the DEFAULT install stays verified even when the release
    listing is unavailable. It is the `--tag latest` and arbitrary `--tag <x>`
    paths that can land here with nothing to check against."""
    if not sha:
        console.print(
            f"[yellow]Warning: this {what} publishes no checksum, so the download's "
            "integrity is not cryptographically verified (its size and archive "
            "shape are still checked). Pass --sha256 <hex> to pin one.[/yellow]")
    _sl._fetch_and_place(url, target, sha)
