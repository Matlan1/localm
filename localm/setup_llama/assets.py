# SPDX-License-Identifier: AGPL-3.0-or-later
"""Picking a backend for this machine, and resolving a backend to a release asset
(download URL, sha256, tag).
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from typing import Optional

import click

from localm import hwdetect
from localm.debuglog import logger
from localm.setup_llama._common import console
from localm.setup_llama.pins import (_AMD_ROCM_ASSET_TAG, _ASSET_MATCH, _CUDA_LINUX_PINNED_SHA256,
                                     _CUDA_LINUX_REPO, _PINNED_FALLBACK_SHA256, _ROCM_TAG,
                                     _UPSTREAM_REPO, DEFAULT_URL, DEFAULT_URL_SHA256)
from localm.setup_llama.versions import _tag_for
from localm.setup_llama.cuda import _CUDA_LINE
import localm.setup_llama as _sl

# --------------------------------------------------------------------------- #
#  Backend resolution                                                          #
# --------------------------------------------------------------------------- #

def _auto_backend() -> str:
    """Pick the broadest WORKING backend for this machine - via the SAME policy the
    installers use (``hwdetect.recommended_install_backend``), so bare
    ``setup-llama`` and setup.bat / setup.sh can never drift:

      NVIDIA, any OS -> cuda (self-contained build + runtime fetch on both
      Windows and Linux, peak performance); AMD on Windows (RX 6000 / unknown)
      -> the self-contained ROCm build; AMD elsewhere with a system ROCm/HIP
      toolkit detected -> hip; Intel on Windows -> sycl (self-contained);
      Apple Silicon -> metal; every other GPU (Intel on Linux, AMD with no
      toolkit detected) -> vulkan; no GPU -> cpu."""
    try:
        from localm import hwdetect
        det = hwdetect.detect()
    except Exception as e:
        # Surface the skipped GPU setup so a detection failure is visible and the
        # user knows how to force a GPU backend, rather than a silent CPU default.
        console.print(f"[yellow]GPU detection failed ({e}); defaulting to CPU - "
                      "override with --backend.[/yellow]")
        return "cpu"
    return hwdetect.recommended_install_backend(det)


def _resolve_backend_asset(backend: str, cuda_line: Optional[str] = None,
                           tag: Optional[str] = None
                           ) -> tuple[str, Optional[str], Optional[str]]:
    """Resolve a backend name to a (url, sha256_digest, tag) triple.

    If the release listing is available, resolves it dynamically and gets the
    sha256 from the digest field. If offline, falls back to the templated guess
    and queries the local pinned checksum dictionary.

    *cuda_line* selects which asset-name substrings to match for the 'cuda'
    backend on Windows AND Linux (see NvidiaInfo.cuda_line) - ignored for
    every other backend/platform, which have a single, non-line-specific
    matcher list. Defaults to _CUDA_LINE when None.

    *tag* lets a caller that has ALREADY resolved one (the Windows CUDA branch,
    which needs it to pair the build with its cudart bundle) pass it in rather
    than resolve a second time. When omitted this resolves its own through
    _tag_for - pin, else upstream's newest.

    THE THIRD ELEMENT IS THE TAG: this is the function that decides it, and the
    marker has to record it, so returning it lets the caller record the
    installed build with NO extra lookup. Re-resolving in the caller doubles the
    network call, and parsing the tag back out of the returned URL is a second
    derivation that diverges on the templated-guess path below, where the URL is
    CONSTRUCTED rather than read from the release listing.

    It is None for amd-rocm, whose build comes from lemonade-sdk's own release
    numbering (_ROCM_TAG) rather than an upstream tag; the caller supplies that
    constant itself."""
    cuda_line = cuda_line or _CUDA_LINE
    if backend == "amd-rocm":
        return _resolve_amd_rocm_asset()
    plat = _sl._platform_key()
    arch = _sl._machine_arch()
    _require_prebuilt(backend, plat, arch)
    if backend == "cuda" and plat == "linux":
        return _resolve_linux_cuda_asset(backend, cuda_line, tag)

    entry = _ASSET_MATCH.get(plat, {}).get(backend)
    # The 'cuda' entry on win32 is keyed by cuda_line (a dict), not a flat
    # list, since the right asset depends on the GPU's architecture, not just
    # the platform - see _ASSET_MATCH's comment.
    matchers = entry.get(cuda_line) if isinstance(entry, dict) else entry
    if not matchers:
        raise click.ClickException(
            f"backend {backend!r} is not available on this platform "
            f"({plat}). Available: {_available_here(plat, arch) or 'none'}.")
    matchers = [m.format(arch=arch) for m in matchers]

    tag = tag or _tag_for(backend)
    assets = _sl._release_assets(tag)
    for a in assets:
        name = str(a.get("name", "")).lower()
        if (any(m in name for m in matchers) and "cudart" not in name
                and _asset_arch_ok(name, arch)
                and a.get("browser_download_url")):
            url = a["browser_download_url"]
            digest = a.get("digest")
            sha = digest.split("sha256:")[-1].strip() if digest and "sha256:" in digest else None
            if not sha:
                sha = _PINNED_FALLBACK_SHA256.get(a.get("name", ""))
            return url, sha, tag

    # Fallback: the release listing was unavailable, so the asset name has to
    # come from somewhere else.
    #
    # PREFER A REAL NAME WE ALREADY KNOW over a constructed one. For a tag
    # pins.py pins, _PINNED_FALLBACK_SHA256 IS that release's asset list, so the
    # exact filename is in hand and needs no guessing. Matchers are tried in
    # their declared order, because that order encodes a preference the names
    # alone do not: linux sycl lists the fp16 build first and a bare
    # "bin-ubuntu-sycl" would also match fp32.
    #
    # This is not a tidy-up. The template below builds `llama-<tag>-<matcher>`,
    # which silently stops matching whenever upstream renames an asset - and it
    # had: b10356 renamed the Windows ROCm asset from `bin-win-hip-radeon-x64` to
    # `bin-win-rocm-<version>-x64`, and the Linux ROCm asset at the pinned tag is
    # `bin-ubuntu-rocm-7.14-x64.ZIP`, an extension the template cannot even
    # express (it assumes tar.gz off win32). Both produced a confident 404 offline.
    # Reading the name from the table fixes every such rename at once, and keeps
    # fixing them: bump the pin and its digests, and this follows.
    if assets:
        raise _no_build_in_release(tag, backend, plat, arch)
    fname = ""
    for m in matchers:
        hits = sorted(n for n in _PINNED_FALLBACK_SHA256
                      if n.startswith(f"llama-{tag}-") and m in n.lower()
                      and "cudart" not in n and _asset_arch_ok(n.lower(), arch))
        if hits:
            fname = hits[0]
            break
    if not fname and any(n.startswith(f"llama-{tag}-") for n in _PINNED_FALLBACK_SHA256):
        raise _no_build_in_release(tag, backend, plat, arch)
    if not fname:
        # An unpinned tag (--tag <something>), so there is nothing to read and a
        # constructed name is the only option left. Still worth attempting: it is
        # right whenever upstream's naming has not drifted.
        ext = "zip" if plat == "win32" else "tar.gz"
        fname = f"llama-{tag}-{matchers[0]}.{ext}"
    guess = f"https://github.com/{_UPSTREAM_REPO}/releases/download/{tag}/{fname}"
    sha = _PINNED_FALLBACK_SHA256.get(fname)
    console.print(f"[yellow]Could not verify release asset list; using unverified URL: {guess}[/yellow]\n"
                  "[yellow]If download fails, pass --from <build dir> or --url <archive>.[/yellow]")
    return guess, sha, tag


def _resolve_amd_rocm_asset() -> tuple[str, Optional[str], None]:
    """The (url, sha256, None) triple for the self-contained amd-rocm build of
    this machine's AMD GPU family, from lemonade-sdk's release listing, else the
    pinned URL and checksum for that family. Raises ``click.ClickException``
    off Windows, and on a CPU architecture the build is not published for."""
    if sys.platform != "win32":
        raise click.ClickException(
            "the self-contained 'amd-rocm' build is Windows-only; on Linux "
            "use --backend hip (needs ROCm) or build with --from.")
    _require_prebuilt("amd-rocm", "win32", _sl._machine_arch())
    # Try to resolve dynamically first
    tag = _ROCM_TAG
    try:
        from localm import hwdetect
        fam = hwdetect.amd_gfx_family(hwdetect.detect().gpu_names)
    except Exception as e:
        # Best-effort like _auto_backend's own detection call: a hiccup
        # here must not block an explicit --backend amd-rocm request. ""
        # is the same conservative default amd_gfx_family() itself returns
        # for an unrecognised card, and it resolves to gfx103X below.
        logger.debug("AMD gfx-family detection failed (%s); defaulting to gfx103X", e)
        fam = ""
    asset_tag = _AMD_ROCM_ASSET_TAG.get(fam, "gfx103X")
    if asset_tag == "gfx103X":
        fallback_url, fallback_sha = DEFAULT_URL, DEFAULT_URL_SHA256
    else:
        asset_name = f"llama-{tag}-windows-rocm-{asset_tag}-x64.zip"
        fallback_url = ("https://github.com/lemonade-sdk/llamacpp-rocm/releases/"
                        f"download/{tag}/{asset_name}")
        fallback_sha = _PINNED_FALLBACK_SHA256.get(asset_name)
    assets = _sl._release_assets(tag, repo="lemonade-sdk/llamacpp-rocm")
    for a in assets:
        if f"windows-rocm-{asset_tag}" in a.get("name", ""):
            url = a.get("browser_download_url") or fallback_url
            digest = a.get("digest")
            sha = digest.split("sha256:")[-1].strip() if digest and "sha256:" in digest else None
            if not sha:
                sha = fallback_sha
            return url, sha, None
    # Surface the fallback so the user knows the build may not be current
    # (the lemonade-sdk release lookup was unreachable, or this release is
    # missing the expected asset for this GPU family); mirrors the
    # visible-fallback warning in _resolve_backend_asset's general (non-ROCm) path
    # instead of silently handing back a possibly-stale pinned URL/checksum.
    console.print("[yellow]Could not find a lemonade-sdk/llamacpp-rocm release asset "
                  f"for {tag} ({asset_tag}); using pinned amd-rocm build - rerun later "
                  "for the latest.[/yellow]")
    return fallback_url, fallback_sha, None


def _resolve_linux_cuda_asset(backend: str, cuda_line: str,
                              tag: Optional[str]) -> tuple[str, Optional[str], str]:
    """The (url, sha256, tag) triple for the Linux CUDA build of *cuda_line*
    from hybridgroup/llama-cpp-builder's release listing. When the listing is
    unavailable (offline, rate-limited) and *tag* is the pinned build, the pinned
    download URL and checksum are returned instead. Raises
    ``click.ClickException`` when neither resolves."""
    # Upstream (ggml-org/llama.cpp) publishes no bare Linux CUDA binary at
    # all, so the generic _ASSET_MATCH path in _resolve_backend_asset would only
    # ever construct a guessed URL that 404s. Resolve against
    # hybridgroup/llama-cpp-builder instead, the same shape as the amd-rocm ->
    # lemonade-sdk resolution in _resolve_amd_rocm_asset: they track upstream's
    # own tag numbering 1:1 and publish upstream's own asset-name convention, so
    # the same tag every other Linux backend uses applies here too.
    #
    # cuda_line-aware, like the win32 cuda matchers in _ASSET_MATCH: hybridgroup
    # publishes both a cuda-12 asset ("...-cuda-x64.tar.gz") and a
    # cuda-13 one ("...-cuda-13-x64.tar.gz").
    suffix = "-cuda-13-x64.tar.gz" if cuda_line == "cuda-13" else "-cuda-x64.tar.gz"
    tag = tag or _tag_for(backend)
    assets = _sl._release_assets(tag, repo=_CUDA_LINUX_REPO)
    for a in assets:
        name = str(a.get("name", "")).lower()
        if name.endswith(suffix) and a.get("browser_download_url"):
            url = a["browser_download_url"]
            digest = a.get("digest")
            sha = digest.split("sha256:")[-1].strip() if digest and "sha256:" in digest else None
            return url, sha, tag
    pinned_name = f"llama-{tag}-bin-ubuntu{suffix}"
    pinned_sha = _CUDA_LINUX_PINNED_SHA256.get(pinned_name)
    if not assets and pinned_sha:
        console.print(f"[yellow]Could not read the {_CUDA_LINUX_REPO} release listing for "
                      f"{tag}; using its pinned {cuda_line} build and checksum.[/yellow]")
        return (f"https://github.com/{_CUDA_LINUX_REPO}/releases/download/{tag}/{pinned_name}",
                pinned_sha, tag)
    # Genuinely unresolvable (hybridgroup has not built that exact
    # upstream tag yet): raise click.ClickException, which
    # _provision_with_fallback's caller already catches and turns into the
    # same offer/force-vulkan-fallback path every other provisioning
    # failure in this package uses. Never construct a guessed URL here the way
    # _resolve_backend_asset's generic path does: a guessed URL against a third
    # party's repo is even less trustworthy than one against upstream itself.
    raise click.ClickException(
        f"no Linux CUDA build found for llama.cpp tag {tag!r} on "
        f"{_CUDA_LINUX_REPO} - falling back to vulkan.")


def _resolve_backend_url(backend: str, cuda_line: Optional[str] = None) -> str:
    """Resolve a backend name to a downloadable archive URL.

    ``amd-rocm`` is the self-contained lemonade build (special-cased). Every
    other backend maps to an upstream llama.cpp release asset for this platform.
    *cuda_line* is passed straight through to _resolve_backend_asset (see its
    docstring); no production code calls this function (main() resolves via
    _provision_backend -> _resolve_backend_asset directly), but it is kept
    line-aware so it cannot silently drift back to a hardcoded cuda-12 default
    if something starts calling it again.
    Raises ``click.ClickException`` if the backend is not available here."""
    url, _sha, _tag = _sl._resolve_backend_asset(backend, cuda_line)
    return url


def _release_assets(tag: str, repo: str = _UPSTREAM_REPO) -> list:
    """The REAL uploaded asset list for a release tag, or [] if the API is
    unavailable or the release has none (yet).

    Does NOT fall back to scraping download links out of the release body: those
    links describe files upstream's CI intends to upload, not files that
    necessarily exist yet (see ``_latest_tag``), so a match built from them 404s
    instead of surfacing as a caught "no assets" case. ``_latest_tag`` already
    skips a release in that state; a tag passed in explicitly by the caller
    (--url, --force, etc.) gets an empty list rather than a guess."""
    api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    try:
        req = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "localm-setup-llama"})
        with _sl.verified_urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
            return data.get("assets", [])
    except Exception as e:
        # Best-effort probe: every caller has a pinned fallback for exactly
        # this case (offline, rate-limited, API down), so this must not raise -
        # but the cause stays discoverable instead of vanishing.
        logger.debug("release asset lookup failed for %s (%s)", api, e)
        return []


def _pick_asset(assets: list, *needles: str, exclude: tuple = (),
                arch: Optional[str] = None) -> Optional[dict]:
    """First asset whose (lowercased) name contains ALL *needles* and NONE of
    *exclude*, and, when *arch* is given, names no CPU architecture other than
    *arch* (see ``_asset_arch_ok``)."""
    for a in assets:
        name = str(a.get("name", "")).lower()
        if (all(n in name for n in needles)
                and not any(x in name for x in exclude)
                and (arch is None or _asset_arch_ok(name, arch))
                and a.get("browser_download_url")):
            return a
    return None


# A CPU-architecture token in a release asset name, delimited so "x64" never
# matches inside a longer word.
_ASSET_ARCH_RE = re.compile(r"(?<![a-z0-9])(x86_64|amd64|x64|aarch64|arm64|s390x)(?![a-z0-9])")
_ASSET_ARCH_ALIASES = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64"}


def _asset_arch_ok(name: str, arch: str) -> bool:
    """False when the lowercased asset *name* names a CPU architecture and none
    of the ones it names is *arch*; True for a name that names none."""
    found = {_ASSET_ARCH_ALIASES.get(t, t) for t in _ASSET_ARCH_RE.findall(name)}
    return not found or arch in found


def _require_prebuilt(backend: str, plat: str, arch: str) -> None:
    """Raise ``click.ClickException`` naming what IS available when *backend*
    has no prebuilt build for *plat* on *arch* (``hwdetect.PREBUILT_ARCHES``).
    A backend the table does not list for *plat* passes, so the caller's own
    "not available on this platform" message still applies to it."""
    if backend not in hwdetect.PREBUILT_ARCHES.get(plat, {}):
        return
    if hwdetect.has_prebuilt(backend, plat, arch):
        return
    raise click.ClickException(
        f"backend {backend!r} has no prebuilt build for {plat} on {arch}. "
        f"Available here: {_available_here(plat, arch) or 'none'}. Or build "
        "llama.cpp yourself and pass --from <build dir>.")


def _available_here(plat: str, arch: str) -> str:
    """The upstream backends with a prebuilt build for *plat* on *arch*, sorted
    and comma-separated ("" when there are none)."""
    return ", ".join(sorted(b for b in _ASSET_MATCH.get(plat, {})
                            if hwdetect.has_prebuilt(b, plat, arch)))


def _no_build_in_release(tag: str, backend: str, plat: str, arch: str) -> click.ClickException:
    """The error for a release whose asset list has no *backend* build for
    *plat* on *arch*."""
    return click.ClickException(
        f"llama.cpp release {tag} has no {backend} build for {plat} on {arch}. "
        "Pick another backend, or build llama.cpp yourself and pass --from <build dir>.")


def _resolve_cuda_pair(tag: str, line: str = _CUDA_LINE) -> tuple:
    """(build_asset, cudart_asset) for the Windows CUDA *line* ('cuda-12' or
    'cuda-13' - see NvidiaInfo.cuda_line). Either may be None when the release
    listing is unavailable or lacks it.

    The build and the cudart runtime share the "...bin-win-cuda-X.Y..." name
    fragment (the runtime is e.g. cudart-llama-bin-win-cuda-12.4-x64.zip), and
    the runtime is often listed FIRST, so the build matcher MUST exclude
    "cudart" - otherwise build resolves to the runtime-only zip (CUDA DLLs, no
    llama.dll) and provisioning aborts with "the archive did not contain
    llama.dll"."""
    assets = _sl._release_assets(tag)
    arch = _sl._machine_arch()
    build = _pick_asset(assets, "bin-win-" + line, exclude=("cudart",), arch=arch)
    cudart = _pick_asset(assets, "cudart", "win-" + line, arch=arch)
    return build, cudart
