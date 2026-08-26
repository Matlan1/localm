# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model download/transport: pull routing, the HF + URL + Ollama backends,
resumable downloads, hashing-on-the-wire, and GUI progress streaming."""

import localm.model_manager as _mm  # read package-patchable names at call time

import contextlib
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import List
from typing import Optional
from rich.progress import BarColumn
from rich.progress import DownloadColumn
from rich.progress import Progress
from rich.progress import TextColumn
from rich.progress import TimeRemainingColumn
from rich.progress import TransferSpeedColumn
from ..debuglog import logger
from ._shared import _emit_progress
from ._shared import _verify_digest
from ._shared import console
from .gguf import _safe_models_filename
from .gguf import split_gguf_parts
from .registry import _detect_local_model_type, _sanitize_name
from .registry import alias_model
from .registry import find_aliases_by_path

# HF_ENDPOINT / HF_HUB_ENDPOINT are ambient env vars localm never exposes as a
# setting. This endpoint is pinned explicitly at every huggingface_hub call site
# below, so an env var in the user's shell cannot redirect a model pull.
_HF_ENDPOINT = "https://huggingface.co"




def _progress_file_info(target_parts: List[Path]) -> "tuple[str | None, int, int]":
    """(current-file name, 1-based index, count) for a multi-part download, derived
    from which parts have already landed at their final path - the first one not yet
    present is the file currently downloading. Cheap existence checks only. Returns
    (None, 0, 0) for a single-file download (nothing to disambiguate)."""
    n = len(target_parts)
    if n <= 1:
        return (None, 0, 0)
    done = sum(1 for p in target_parts if p.exists())
    cur = next((p for p in target_parts if not p.exists()), target_parts[-1])
    return (cur.name, min(done + 1, n), n)




class _ProgressOutcome:
    """Explicit success signal for the progress context managers.

    The body must SAY it finished by calling ``ok()``; silence means it did not.
    The absence of an exception is not enough: ``_pull_gguf_file`` reports a
    failed part with ``return False`` from INSIDE its ``with`` block, which
    unwinds perfectly cleanly.
    """

    __slots__ = ("succeeded",)

    def __init__(self) -> None:
        self.succeeded = False

    def ok(self) -> None:
        self.succeeded = True


def _incomplete_prefixes(base_dir: Path, rel_parts: List[str]) -> "set[str] | None":
    """Filename prefixes of the ``.incomplete`` temp files for *rel_parts*.

    huggingface_hub names a local-dir temp file
    ``<short_hash(<name>.metadata)>.<etag>.incomplete`` under
    ``<local_dir>/.cache/huggingface/download/<subpath>/``. The etag is not
    knowable in advance, but the hash prefix is, and it is what separates OUR
    parts from a concurrent pull's.

    Returns None when the layout cannot be computed. The caller then falls back
    to an unfiltered scan of this destination, which is coarser but never counts
    another DESTINATION's bytes.
    """
    try:
        from huggingface_hub._local_folder import _short_hash
        from huggingface_hub._local_folder import get_local_download_paths
        out = set()
        for rel in rel_parts:
            paths = get_local_download_paths(Path(base_dir), rel)
            out.add(_short_hash(paths.metadata_path.name))
        return out or None
    except Exception as e:
        # Non-fatal: progress simply gets coarser. Logged at debug so a
        # huggingface_hub layout change stays discoverable.
        logger.debug("cannot compute .incomplete prefixes (progress will be "
                     "coarser, not wrong): %s", e)
        return None


@contextlib.contextmanager
def _download_progress(target_parts: List[Path], total_size: int, *,
                       base_dir: "Path | None" = None,
                       rel_parts: "List[str] | None" = None):
    """Stream JSON download progress while files land under *base_dir*.

    Active in GUI mode (LOCALM_PROGRESS_JSON=1). A total of 0 means "we could
    not size this": progress still streams with ``pct: null`` so the GUI shows a
    busy bar with a running byte count, matching _snapshot_progress.

    Yields a _ProgressOutcome; call ``.ok()`` on the success path or the closing
    event reports the measured partial instead of 100%.
    """
    outcome = _ProgressOutcome()
    if os.environ.get("LOCALM_PROGRESS_JSON") != "1":
        yield outcome
        return

    stop = threading.Event()
    cache_root = (Path(base_dir) if base_dir is not None else _mm.MODELS_DIR) / ".cache"
    prefixes = _incomplete_prefixes(Path(base_dir), rel_parts) \
        if base_dir is not None and rel_parts else None

    def _downloaded_bytes() -> int:
        done = 0
        for p in target_parts:
            try:
                done += p.stat().st_size
            except OSError:
                pass  # not finished yet
        active = 0
        if cache_root.is_dir():
            try:
                for f in cache_root.rglob("*.incomplete"):
                    # Count only the .incomplete files belonging to THIS job, so a
                    # concurrent pull's temp file is not added to this numerator.
                    if prefixes is not None and f.name.split(".")[0] not in prefixes:
                        continue
                    try:
                        active += f.stat().st_size
                    except OSError:
                        pass
            except OSError:
                pass
        total_now = done + active
        # Only clamp against a total we actually have. min(x, 0) is 0, which
        # would pin an indeterminate download's byte count at zero forever.
        return min(total_now, total_size) if total_size else total_now

    def _poll() -> None:
        last = -1
        while not stop.is_set():
            dl = _downloaded_bytes()
            if dl != last:
                last = dl
                fn, fi, fc = _progress_file_info(target_parts)
                _emit_progress(dl, total_size, name=fn, index=fi, count=fc,
                               zero_is_unknown=True)
            stop.wait(0.7)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    fn0, fi0, fc0 = _progress_file_info(target_parts)
    # Seed from the measurement rather than a literal 0: on a resume, parts already
    # on disk make 0 wrong. zero_is_unknown renders the fresh case, where the
    # measurement is genuinely 0, as unknown rather than a confident 0%.
    _emit_progress(_downloaded_bytes(), total_size, name=fn0, index=fi0, count=fc0,
                   zero_is_unknown=True)
    try:
        yield outcome
    finally:
        stop.set()
        t.join(timeout=2)
        fn1, fi1, fc1 = _progress_file_info(target_parts)
        # 100% is a claim about the outcome, so only the success path may make
        # it. Otherwise report what is actually on disk.
        final = total_size if (outcome.succeeded and total_size) else _downloaded_bytes()
        _emit_progress(final, total_size, name=fn1, index=fi1, count=fc1)




@contextlib.contextmanager
def _snapshot_progress(disk_bytes_fn, total_size: int):
    """Like _download_progress but for snapshot_download (many files): byte
    count comes from a caller-supplied directory-size function. Indeterminate
    (total_size == 0) still streams a 'downloading' phase so the GUI can show
    a busy bar; no-op outside GUI mode.

    Yields a _ProgressOutcome; call ``.ok()`` on the success path or the closing
    event reports the measured partial instead of 100%.
    """
    outcome = _ProgressOutcome()
    if os.environ.get("LOCALM_PROGRESS_JSON") != "1":
        yield outcome
        return

    stop = threading.Event()

    def _measured() -> int:
        dl = disk_bytes_fn()
        return min(dl, total_size) if total_size else dl

    def _poll() -> None:
        last = -1
        while not stop.is_set():
            dl = _measured()
            if dl != last:
                last = dl
                _emit_progress(dl, total_size, zero_is_unknown=True)
            stop.wait(0.7)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    # Seed from the measurement (see _download_progress): a resumed snapshot or
    # .part file already has bytes on disk. zero_is_unknown renders a measured 0
    # as unknown rather than a confident 0%.
    _emit_progress(_measured(), total_size, zero_is_unknown=True)
    try:
        yield outcome
    finally:
        stop.set()
        t.join(timeout=2)
        # Only a successful run may claim the total. A failure reports what
        # actually landed.
        if outcome.succeeded and total_size:
            _emit_progress(total_size, total_size)
        else:
            _emit_progress(_measured(), total_size)




def _report_success(rich_msg: str, plain_msg: str) -> None:
    """Announce a completed pull. A DISPLAY failure never reads as an OPERATION
    failure: every exception raised by the announcement is caught here.

    Precondition: at every call site the download, checksum verification and
    registry write are already fully done, so there is no remaining work this
    call could be masking. A call moved earlier, before that work completes,
    breaks that guarantee.

    The rich message is printed first; on any exception a warning is logged and
    the plain-ASCII message is printed instead. A failure of that fallback is
    logged too and never propagates.
    """
    try:
        console.print(rich_msg)
    except Exception as e:
        logger.warning("could not render the pull success message (%s); "
                        "falling back to a plain-ASCII version", e)
        try:
            console.print(plain_msg)
        except Exception as e2:
            logger.warning("plain-text fallback also failed to print: %s", e2)



def pull_model(
    model_spec: str,
    name: Optional[str] = None,
    expected_sha256: Optional[str] = None,
    redownload: bool = False,
    mmproj_spec: Optional[str] = None,
    model_type: str = "auto",
    store: Optional[str] = None,
    dest_dir: Optional[Path] = None,
    register: bool = True,
) -> bool:
    """Download a model from HuggingFace or a URL.

    Returns True on success or a benign no-op (already present / aliased /
    user-skipped), False on a real error, so callers can set a non-zero exit
    code and the GUI can mark the job failed instead of reporting "finished".

    *store* ("copy" / "move" / None) only applies to the local-path branch
    below - a remote HF/URL download already lands in MODELS_DIR on its own.

    *dest_dir*, when given, routes the download to that directory instead of
    MODELS_DIR (e.g. a ComfyUI models subfolder) and skips localm's own
    registry when *register* is False. Only supported for a single-file HF
    spec (``owner/repo:file`` or ``owner/repo/file.gguf``) - a bare-repo
    snapshot or a direct URL pull with *dest_dir* set is refused rather than
    silently downloading to MODELS_DIR anyway.
    """
    spec = _mm.resolve_spec(model_spec)
    type_is_auto = (model_type == "auto")

    # A local filesystem path is registered in place rather than parsed as an
    # owner/repo:file spec, and this runs BEFORE any network auto-detect so a local
    # path is never probed against huggingface.co. Only an absolute path or an
    # existing file counts, so a bare HF owner/repo is not shadowed by a same-named
    # local directory. add_local does the validation and dedup.
    try:
        local = Path(model_spec).expanduser()
        is_local_path = local.exists() and (local.is_absolute() or local.is_file())
    except OSError:
        is_local_path = False
    if is_local_path:
        # A user-supplied --sha256 is verified against the real bytes for a local
        # file and refused on mismatch. A full HF repo refuses the flag outright.
        if expected_sha256:
            if local.is_dir():
                console.print(
                    "[red]--sha256 is not supported for a local directory[/red] "
                    f"({local}): a folder has many files and no single digest to "
                    "verify. Drop --sha256, or point at a single .gguf file.")
                return False
            want = expected_sha256.strip().lower()
            actual = _verify_digest(local).lower()
            if actual != want:
                console.print(
                    f"[red]SHA256 mismatch![/red] {local} is {actual[:16]}…, not "
                    f"--sha256 {want[:16]}…. Refusing to register.")
                return False
            _report_success(f"[green]✓[/green] SHA256 verified: {actual[:16]}…",
                            f"[green]OK[/green] SHA256 verified: {actual[:16]}…")
        # A local file gets no remote type probe: honour an explicit --type, else
        # let add_local detect it (GGUF -> llm, HF dir -> config.json, otherwise
        # 'unknown').
        local_type = None if model_type == "auto" else model_type
        return _mm.add_local(str(local), name=name, model_type=local_type, store=store)

    # net_mode=off stops a remote pull, including the type auto-detect probe below.
    from localm.netpolicy import network_mode
    if network_mode() == "off":
        console.print(
            "[red]Network access is disabled (net_mode=off).[/red] A model pull "
            "needs the network; enable it with: localm config net_mode ask")
        return False

    # Remote spec: resolve the model type (a network probe against HF for a bare
    # owner/repo). Reached only for confirmed-remote specs, after the local-path and
    # net_mode gates above.
    detected_type = "llm"
    if model_type == "auto":
        if "/" in spec and not (spec.startswith("http://") or spec.startswith("https://")):
            repo_id = spec.split(":")[0] if ":" in spec else spec
            # A named .gguf file is typed 'llm' from the container format alone,
            # without falling back to the repo's HF pipeline_tag. Matches both file
            # forms the dispatch below accepts (owner/repo:file.gguf and
            # owner/repo/file.gguf) via the last path segment; the pipeline tag is
            # probed only for a bare repo.
            if spec.rsplit("/", 1)[-1].lower().endswith(".gguf"):
                detected_type = "llm"
            else:
                detected_type = _hf_pipeline_tag_to_type(repo_id)
                logger.info("Auto-detected model type for %s: %s", repo_id, detected_type)
                # A bare owner/repo pull is a full snapshot, and _pull_hf_snapshot
                # re-checks the type against the config.json it downloads and
                # announces the outcome itself. Every other spec in this branch
                # registers the probe's answer as-is, so it is announced here.
                if detected_type == "unknown" and ":" in spec:
                    # An 'unknown' model is not auto-loaded for chat, but it stays
                    # runnable by name.
                    console.print(
                        "[yellow]Could not determine this model's type[/yellow] - "
                        "registering it as 'unknown'. Run it by name, or set its type "
                        "later: [bold]localm set-type <name> <type>[/bold]")
        else:
            detected_type = "llm"
    else:
        detected_type = model_type

    is_url_spec = spec.startswith("http://") or spec.startswith("https://")
    is_single_file_spec = not is_url_spec and "/" in spec and (
        ":" in spec or spec.rsplit("/", 1)[-1].endswith(".gguf"))
    if dest_dir is not None and not is_single_file_spec:
        # dest_dir routing is wired only through _pull_gguf_file, so any other spec
        # shape is refused rather than downloaded to MODELS_DIR.
        console.print(
            "[red]dest_dir is only supported for a single-file spec[/red] "
            "(owner/repo:file or owner/repo/file.gguf) - "
            f"{model_spec!r} would pull a full snapshot or direct URL instead."
        )
        return False

    if spec.startswith("http://") or spec.startswith("https://"):
        res = _pull_url(spec, _sanitize_name(name or _stem_from_url(spec)),
                         expected_sha256=expected_sha256, redownload=redownload,
                         model_type=detected_type)
    elif "/" in spec:
        if ":" in spec or spec.rsplit("/", 1)[-1].endswith(".gguf"):
            # owner/repo:file.gguf  or  owner/repo/file.gguf  -> single GGUF file
            res = _mm._pull_gguf_file(spec, name, expected_sha256=expected_sha256,
                                   redownload=redownload, model_type=detected_type,
                                   dest_dir=dest_dir, register=register,
                                   type_is_auto=type_is_auto, mmproj_spec=mmproj_spec)
        else:
            # owner/repo  (no filename) -> full HuggingFace snapshot
            res = _mm._pull_hf_snapshot(spec, name, expected_sha256=expected_sha256,
                                     redownload=redownload, model_type=detected_type)
    else:
        console.print(f"[red]Unknown spec:[/red] {model_spec}")
        console.print("Formats:")
        console.print("  [bold]owner/repo[/bold]              full HF model directory")
        console.print("  [bold]owner/repo:file.gguf[/bold]   single GGUF file")
        console.print("  [bold]https://...[/bold]             direct URL")
        console.print("Run [bold]localm models[/bold] for GGUF shortcuts.")
        res = False

    # A single-file GGUF spec already threaded mmproj_spec into _pull_gguf_file
    # above, which fetched it and recorded it on the registry entry. This tail
    # covers the other dispatch shapes (a direct URL, a full HF snapshot), where the
    # projector is only downloaded, not attached.
    if res and mmproj_spec and not is_single_file_spec:
        console.print(f"Pulling mmproj: {mmproj_spec}")
        # A '/' must be present (an owner/repo), the same precondition
        # _fetch_explicit_mmproj enforces: a bare 'file.gguf' would otherwise reach
        # _pull_gguf_file's split and raise IndexError.
        if "/" in mmproj_spec and (
                ":" in mmproj_spec or mmproj_spec.rsplit("/", 1)[-1].endswith(".gguf")):
            _mm._pull_gguf_file(mmproj_spec, name=None, register=False)
        else:
            console.print("[red]mmproj spec must be a specific file (owner/repo:file.gguf)[/red]")

    return res




def _stem_from_url(url: str) -> str:
    return url.split("/")[-1].split("?")[0].removesuffix(".gguf")




def _check_disk_space(dest_dir: Path, required_bytes: int) -> bool:
    """Verify there is at least *required_bytes* of free space on the volume that
    holds *dest_dir*. Prints a warning and returns False when space is
    insufficient; returns True when fine or when the check is skipped
    (e.g. ``required_bytes == 0``).

    An unmeasurable free-space check (offline models dir, permission denied,
    etc.) is treated as OK: a WARNING is logged and the download proceeds.
    """
    if not required_bytes:
        return True
    try:
        usage = shutil.disk_usage(dest_dir)
        if usage.free < required_bytes:
            need_gb  = required_bytes / 1024**3
            free_gb  = usage.free / 1024**3
            console.print(
                f"[red]Not enough disk space.[/red] "
                f"Need {need_gb:.1f} GB, have {free_gb:.1f} GB free on {dest_dir}"
            )
            return False
    except Exception as e:
        # Surface an unmeasurable check (do not silence) but stay non-fatal:
        # an unknown free-space value is treated as OK so the download proceeds.
        logger.warning("could not check free space on %s (%s); proceeding", dest_dir, e)
    return True




def _hf_file_sha256(repo_id: str, filename: str) -> Optional[str]:
    """
    Ask the HuggingFace API for a file's LFS sha256 without downloading it.
    Returns None when offline, on any API error, or for non-LFS files.
    """
    try:
        from huggingface_hub import HfApi
        info = HfApi(endpoint=_HF_ENDPOINT).get_paths_info(repo_id, [filename])
        if info:
            lfs = getattr(info[0], "lfs", None)
            digest = getattr(lfs, "sha256", None) if lfs else None
            return digest.lower() if digest else None
    except Exception:
        pass
    return None




def _pick_best_of_same_repo_mmprojs(cands: List[str]) -> str:
    """Deterministic pick among several mmproj filenames found in the SAME repo,
    once stem-matching (``_pick_mmproj_candidate``) could not narrow them to one.
    Every candidate here comes from the ONE repo the caller is pulling from, so
    they are quantised variants of the same projector for the same model.
    Prefers the conventional highest-precision f16 build; falls back to a
    sorted-first pick for determinism."""
    f16 = [c for c in cands if "f16" in c.lower()]
    return f16[0] if f16 else sorted(cands)[0]


def _hf_repo_files(repo_id: str) -> Optional[List[str]]:
    """*repo_id*'s file listing, or None when it could not be fetched at all
    (offline, API error, rate limit) - kept distinct from "fetched, and it lists
    none". A listing FAILURE must never be read as "this repo has no projector",
    which would print a false "no vision projector found" note on a transient HF
    API hiccup."""
    try:
        from huggingface_hub import HfApi
        return HfApi(endpoint=_HF_ENDPOINT).list_repo_files(repo_id)
    except Exception as e:
        logger.debug("could not list files for %s to look for an mmproj "
                     "sibling: %s", repo_id, e)
        return None


def _pick_mmproj_from_listing(
    files: List[str], model_filename: str, base_dir: Path,
) -> Optional[str]:
    """The mmproj (vision projector) filename among *files* (a repo's file
    listing) that pairs with *model_filename*, or None when none qualify.

    *files* comes from a REMOTE HF repo listing, so every candidate is confined
    through ``_safe_models_filename`` - the same guard an explicit --mmproj
    filename gets - before it is considered for picking, not merely rejected
    after being chosen. A single-path-component check alone (e.g. "no '/'") is
    not enough: on Windows a value with no forward slash can still be a
    drive-qualified or backslash-relative path. ``_safe_models_filename`` also
    confines the result to land inside *base_dir*."""
    cands = [f for f in files
             if f != model_filename and "mmproj" in f.lower()
             and f.lower().endswith(".gguf")
             and _mm._safe_models_filename(f, base_dir) is not None]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    picked = _mm._pick_mmproj_candidate(Path(model_filename).stem, cands)
    if picked:
        return picked
    # _pick_mmproj_candidate gives up in two cases it cannot distinguish: no
    # candidate shares the model's leading token, or several do. Only the second is
    # resolved here, so _pick_best_of_same_repo_mmprojs only ever chooses among
    # candidates already known to relate to this model.
    stem = Path(model_filename).stem.lower().replace("mmproj", "").split("-")[0].split(".")[0]
    stem_matches = [c for c in cands if stem and stem in c.lower()]
    if len(stem_matches) >= 2:
        return _pick_best_of_same_repo_mmprojs(stem_matches)
    return None


def _hf_repo_mmproj_filename(
    repo_id: str, model_filename: str, base_dir: Path,
) -> Optional[str]:
    """The mmproj (vision projector) filename in *repo_id*'s OWN file listing
    that pairs with *model_filename*, or None when the repo ships none (or its
    listing could not be fetched at all). A free HuggingFace metadata call (repo
    file listing, no download), so a GUI/MCP pull can attach a vision model's
    projector without the CLI --mmproj flag."""
    files = _hf_repo_files(repo_id)
    if files is None:
        return None
    return _pick_mmproj_from_listing(files, model_filename, base_dir)


def _maybe_fetch_repo_mmproj(repo_id: str, filename: str, base_dir: Path) -> Optional[Path]:
    """Auto-attach companion: look for a vision projector shipped in the SAME HF
    repo as *filename* and fetch it too.

    Returns the local Path of a verified projector to record on the model's
    registry entry, or None. When the listing could not be fetched at all, stays
    silent (see ``_hf_repo_files``). Only when the repo listing was genuinely
    read and *filename* looks like a vision-language release (by name) with no
    usable projector among it does this print an informational note, so the gap
    is visible at pull time rather than at first image; registry.py's
    ``vision_input_guidance`` is the analogous message for the chat-time case.
    """
    files = _hf_repo_files(repo_id)
    if files is None:
        return None
    candidate = _pick_mmproj_from_listing(files, filename, base_dir)
    if candidate is None:
        if _mm._looks_like_vision_gguf_name(repo_id, filename):
            console.print(
                "[yellow]Note:[/yellow] this looks like a vision-language "
                f"model, but no vision projector (mmproj) file was found in "
                f"{repo_id}. It may not be able to see images - if the "
                "projector lives in a different repo, pull it explicitly "
                "with [bold]--mmproj <repo>:<file>[/bold]."
            )
        return None

    dest = base_dir / candidate
    if not dest.exists():
        console.print(f"Pulling vision projector: {candidate}")
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(repo_id=repo_id, filename=candidate,
                                     local_dir=str(base_dir), endpoint=_HF_ENDPOINT)
            if Path(local) != dest:
                shutil.move(local, dest)
        except Exception as e:
            console.print(
                f"[yellow]Found a vision projector ({candidate}) in {repo_id} "
                f"but could not download it: {e}. Vision may not work for "
                "this model.[/yellow]"
            )
            return None

    if not _mm.gguf_is_mmproj(dest):
        # Hard-verify the fetched file against real GGUF metadata: the filename
        # match is only a heuristic, and a file failing this check is reported as
        # 'none found' rather than attached.
        console.print(
            f"[yellow]{candidate} does not look like a valid vision "
            "projector (GGUF metadata check failed) - not attaching it.[/yellow]"
        )
        return None
    return dest


def mmproj_backfill_candidate(entry: dict, path: Path) -> bool:
    """True when *entry* (a registry entry whose file is *path*) is a plausible
    target for the mmproj backfill: pulled from an HF repo, a plain LLM
    registration (never a projector needing its own projector), and not already
    carrying a recorded ``mmproj``. Pure and does no I/O - the network decision
    lives in ``backfill_mmproj_for_entry`` below, so a caller can filter cheaply
    first."""
    source = str(entry.get("source", ""))
    if not source.startswith("hf:"):
        return False
    if entry.get("model_type") != "llm":
        return False
    if entry.get("mmproj"):
        return False
    if "mmproj" in path.name.lower():
        return False
    return True


def backfill_mmproj_for_entry(entry: dict, path: Path) -> Optional[Path]:
    """Attach a vision projector to an LLM registry entry that has none, using
    the same-repo lookup a fresh pull does (``_maybe_fetch_repo_mmproj``) and the
    ``source`` this entry already recorded (``hf:<repo_id>``). The model itself
    is not re-downloaded and no user action is needed; the fetched projector gets
    the same auto-attach plus hard-verify treatment a fresh pull gives it.

    Returns the fetched/verified projector Path (the caller records it), or None
    when not a candidate, blocked by policy, or nothing was found. Never raises,
    mirroring ``_maybe_fetch_repo_mmproj``'s own contract, so a sync pass is not
    taken down by one bad entry.

    Network policy: gated on ``network_mode() != "off"``, the SAME bar
    ``_pull_gguf_file``'s own net_mode gate uses for this identical
    HF-listing-plus-download operation on an explicit pull, not the stricter
    "== allow" bar ``embedder.py`` uses for its background fetch. Only the "off"
    kill-switch is a hard stop."""
    if not mmproj_backfill_candidate(entry, path):
        return None
    from localm.netpolicy import network_mode
    if network_mode() == "off":
        return None
    repo_id = str(entry["source"])[len("hf:"):]
    if not repo_id:
        return None
    return _maybe_fetch_repo_mmproj(repo_id, path.name, path.parent)


def _fetch_explicit_mmproj(mmproj_spec: str, base_dir: Path) -> Optional[Path]:
    """Download the user-named --mmproj file (owner/repo:file.gguf) into
    *base_dir* and return its local path, verified as a real vision projector -
    or None on a bad spec, failed download, or failed verification (always
    printed, never silent). An explicit --mmproj always wins over the same-repo
    auto-detection in ``_maybe_fetch_repo_mmproj``; the caller only reaches here
    when the user named one."""
    # A '/' must be present (an owner/repo) as well as one of the two file markers,
    # matching pull_model's own is_single_file_spec check. Without the '/' a bare
    # 'file.gguf' reaches the else branch below, where rsplit on a '/'-free string
    # returns a ONE-element list and parts[1] raises IndexError.
    if not ("/" in mmproj_spec
            and (":" in mmproj_spec or mmproj_spec.rsplit("/", 1)[-1].endswith(".gguf"))):
        console.print("[red]mmproj spec must be a specific file (owner/repo:file.gguf)[/red]")
        return None
    if ":" in mmproj_spec:
        m_repo, m_file = mmproj_spec.rsplit(":", 1)
    else:
        parts = mmproj_spec.rsplit("/", 1)
        m_repo, m_file = parts[0], parts[1]

    # Same traversal guard as the main file: m_file comes from a spec, which may be
    # user- or client-supplied over MCP.
    safe = _mm._safe_models_filename(m_file, base_dir)
    if safe is None:
        console.print(f"[red]Unsafe mmproj filename:[/red] {m_file}")
        return None

    dest = base_dir / safe
    if not dest.exists():
        console.print(f"Pulling mmproj: {mmproj_spec}")
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(repo_id=m_repo, filename=safe, local_dir=str(base_dir),
                                     endpoint=_HF_ENDPOINT)
            if Path(local) != dest:
                shutil.move(local, dest)
        except Exception as e:
            console.print(f"[red]mmproj download failed:[/red] {e}")
            return None

    if not _mm.gguf_is_mmproj(dest):
        console.print(
            f"[yellow]{safe} does not look like a valid vision projector "
            "(GGUF metadata check failed) - not attaching it.[/yellow]"
        )
        return None
    return dest


def _mmproj_for_registration(
    reg_type: str,
    repo_id: str,
    filename: str,
    base_dir: Path,
    dest_dir: Optional[Path],
    mmproj_spec: Optional[str],
) -> Optional[Path]:
    """The vision-projector Path to record on this pull's registry entry, or
    None. An explicit --mmproj wins when given, else the same-repo listing is
    auto-checked. Skipped entirely for a foreign destination (ComfyUI's
    dest_dir), for anything that is not a plain 'llm' registration, and for a
    *filename* that already looks like a projector by its own name."""
    if dest_dir is not None or reg_type != "llm" or "mmproj" in filename.lower():
        return None
    if mmproj_spec:
        return _fetch_explicit_mmproj(mmproj_spec, base_dir)
    return _maybe_fetch_repo_mmproj(repo_id, filename, base_dir)


def _pull_gguf_file(
    spec: str,
    name: Optional[str],
    expected_sha256: Optional[str] = None,
    redownload: bool = False,
    register: bool = True,
    model_type: str = "llm",
    dest_dir: Optional[Path] = None,
    type_is_auto: bool = False,
    mmproj_spec: Optional[str] = None,
) -> bool:
    """Download a single file from a HuggingFace repo (despite the name, not
    restricted to .gguf - any single-file ``owner/repo:filename`` spec dispatches
    here, see ``pull_model``'s docstring).

    ``expected_sha256`` is the user-supplied ``--sha256`` digest. When given it
    is reconciled with HuggingFace's own LFS metadata up front, and the
    downloaded first part is verified against it before the model is registered.

    ``dest_dir``, when given, routes the download to that directory instead of
    ``MODELS_DIR`` (e.g. a ComfyUI models subfolder) and is created via
    ``_mkdir_or_explain`` instead of ``ensure_dirs()``. ``register`` still
    controls whether the download is added to localm's own model registry -
    a file routed elsewhere should normally pass ``register=False``.

    ``mmproj_spec``, when given, is the user's explicit ``--mmproj
    owner/repo:file.gguf`` choice and always wins over the automatic same-repo
    projector lookup below. When it is None and the pulled file registers as a
    plain 'llm', the HF repo's own file listing is checked for a vision-projector
    (mmproj) sibling and, if found, fetched and recorded on the registry entry.
    """
    try:
        from huggingface_hub import hf_hub_download, hf_hub_url
    except ImportError:
        console.print("[red]Missing:[/red] huggingface-hub  (run: uv pip install huggingface-hub)")
        return False

    base_dir = dest_dir if dest_dir is not None else _mm.MODELS_DIR

    if ":" in spec:
        repo_id, filename = spec.rsplit(":", 1)
    else:
        parts = spec.rsplit("/", 1)
        repo_id, filename = parts[0], parts[1]

    # Split GGUF: normalise to the full ordered part list. The first part is what
    # gets registered, since llama.cpp loads the model from it. A non-split,
    # non-gguf file is a one-element list.
    all_parts = split_gguf_parts(filename) or [filename]
    filename  = all_parts[0]

    # The filename comes from an untrusted spec (owner/repo:../../evil.gguf), so
    # every part is confined to base_dir before it is used as a destination. Any
    # unsafe part rejects the whole pull.
    for part in all_parts:
        if _safe_models_filename(part, base_dir) is None:
            console.print(
                f"[red]Unsafe model filename:[/red] {part}\n"
                "A model filename must be a single name inside the models folder "
                "(no '/', '\\', or '..')."
            )
            return False

    model_name = _sanitize_name(name or filename.removesuffix(".gguf"))
    dest = base_dir / filename

    # Expected digest from HF metadata - free, no download needed.
    # (Only identifies the first part of a split GGUF, which is enough.)
    expected = _mm._hf_file_sha256(repo_id, filename)

    # Honour a user-supplied --sha256: when HF's own metadata digest is known and
    # disagrees with it, refuse before downloading.
    want = expected_sha256.lower() if expected_sha256 else None
    if want and expected and want != expected.lower():
        console.print(
            f"[red]SHA256 mismatch (before download):[/red] --sha256 "
            f"{want[:16]}… does not match HuggingFace's metadata for "
            f"{filename} ({expected[:16]}…). Refusing to download."
        )
        return False
    # The digest we will verify against / store: prefer HF metadata, else the
    # user-supplied value.
    verify_digest = expected or want

    missing = [p for p in all_parts if not (base_dir / p).exists()]
    if not missing:
        console.print(f"[yellow]Already downloaded:[/yellow] {filename}")
        # If the user asserted a hash, verify the file actually on disk before
        # treating it as the requested model.
        if want:
            on_disk = _verify_digest(dest, purpose="to check the file already here")
            if on_disk.lower() != want:
                console.print(
                    f"[red]SHA256 mismatch![/red] The file already at {filename} "
                    f"({on_disk[:16]}…) does not match --sha256 ({want[:16]}…)."
                )
                return False
        if register:
            reg_type = model_type
            # One shared header probe backs both the mmproj/embedding refinement
            # below and the persisted architecture/expert_count, captured
            # regardless of type_is_auto.
            gguf_meta = _mm.gguf_registry_metadata(dest)
            if type_is_auto and reg_type == "llm":
                if _mm.gguf_is_mmproj(dest):
                    console.print("[dim]Detected as a vision projector (GGUF metadata).[/dim]")
                    reg_type = "mmproj"
                elif _mm.gguf_embedding_signal(dest):
                    console.print("[dim]Detected as an embedding model (GGUF metadata).[/dim]")
                    reg_type = "embedding"
            mmproj_path = _mmproj_for_registration(
                reg_type, repo_id, filename, base_dir, dest_dir, mmproj_spec)
            _mm._register_with_dedup(model_name, dest, f"hf:{repo_id}",
                                 digest=verify_digest, model_type=reg_type,
                                 mmproj=mmproj_path, architecture=gguf_meta.get("architecture"),
                                 expert_count=gguf_meta.get("expert_count"))
        return True

    # Pre-download duplicate check: are the same bytes already registered? Applied
    # only when the destination IS localm's own models dir, since find_by_sha256
    # answers 'is it in the registry', not 'is it at the destination'. A pull with
    # an explicit dest_dir is register=False, so neither skipping nor aliasing
    # applies.
    if verify_digest and not redownload and dest_dir is None:
        dups = _mm.find_by_sha256(verify_digest)
        if dups:
            action = _mm._prompt_predownload_dup(dups, model_name)
            if action == "skip":
                return True
            if action == "alias":
                alias_model(dups[0], model_name)
                return True
            # "download" falls through

    if dest_dir is not None:
        from ..config import _mkdir_or_explain
        _mkdir_or_explain(base_dir, is_home=False)
    else:
        _mm.ensure_dirs()

    # Disk space pre-flight - HEAD each missing part's CDN URL for Content-Length
    try:
        import requests as _req
        total_size = 0
        for part in missing:
            cdn_url = hf_hub_url(repo_id, part, endpoint=_HF_ENDPOINT)
            head    = _req.head(cdn_url, allow_redirects=True, timeout=10)
            total_size += int(head.headers.get("content-length", 0))
    except Exception:
        total_size = 0

    if not _mm._check_disk_space(base_dir, total_size):
        return False

    if len(all_parts) > 1:
        console.print(
            f"Pulling [bold cyan]{repo_id}[/bold cyan] / [bold]{filename}[/bold] "
            f"[dim](split GGUF, {len(all_parts)} parts, "
            f"{len(missing)} to download)[/dim]"
        )
    else:
        console.print(f"Pulling [bold cyan]{repo_id}[/bold cyan] / [bold]{filename}[/bold]")

    with _download_progress([base_dir / p for p in missing], total_size,
                            base_dir=base_dir, rel_parts=list(missing)) as _prog:
        for part in missing:
            try:
                local = hf_hub_download(
                    repo_id=repo_id,
                    filename=part,
                    local_dir=str(base_dir),
                    endpoint=_HF_ENDPOINT,
                )
                final = base_dir / part
                if Path(local) != final:
                    shutil.move(local, final)
            except Exception as e:
                console.print(f"[red]Download failed[/red] ({part}): {e}")
                # Returns without _prog.ok(), so the context manager reports what is
                # actually on disk instead of 100%.
                return False
        _prog.ok()

    # Verify the downloaded first part against the user's --sha256. On mismatch,
    # delete the part(s) and fail.
    if want:
        actual = _verify_digest(dest).lower()
        if actual != want:
            console.print(
                f"[red]SHA256 mismatch![/red] Expected {want[:16]}…, got "
                f"{actual[:16]}… - deleting downloaded file(s)"
            )
            for part in all_parts:
                p = base_dir / part
                if p.exists():
                    p.unlink()
            return False
        _report_success(f"[green]✓[/green] SHA256 verified: {actual[:16]}…",
                        f"[green]OK[/green] SHA256 verified: {actual[:16]}…")

    if register:
        reg_type = model_type
        # One shared header probe, as in the 'already downloaded' branch above.
        gguf_meta = _mm.gguf_registry_metadata(base_dir / filename)
        if type_is_auto and reg_type == "llm":
            if _mm.gguf_is_mmproj(base_dir / filename):
                console.print("[dim]Detected as a vision projector (GGUF metadata).[/dim]")
                reg_type = "mmproj"
            elif _mm.gguf_embedding_signal(base_dir / filename):
                console.print("[dim]Detected as an embedding model (GGUF metadata).[/dim]")
                reg_type = "embedding"
        mmproj_path = _mmproj_for_registration(
            reg_type, repo_id, filename, base_dir, dest_dir, mmproj_spec)
        _mm._register(model_name, base_dir / filename, f"hf:{repo_id}",
                  sha256=verify_digest, model_type=reg_type, mmproj=mmproj_path,
                  architecture=gguf_meta.get("architecture"),
                  expert_count=gguf_meta.get("expert_count"))
        _report_success(f"[green]✓[/green] [bold]{model_name}[/bold] is ready",
                        f"[green]OK[/green] [bold]{model_name}[/bold] is ready")
    else:
        _report_success(f"[green]✓[/green] [bold]{filename}[/bold] downloaded",
                        f"[green]OK[/green] [bold]{filename}[/bold] downloaded")
    return True




def _snapshot_bytes_on_disk(dest: Path) -> int:
    """Bytes of *dest* that a snapshot resume would not re-fetch. Excludes
    huggingface_hub's own .cache scratch, which is not part of the model."""
    try:
        return sum(f.stat().st_size for f in dest.rglob("*")
                   if f.is_file() and ".cache" not in f.parts)
    except OSError:
        return 0


def _warn_if_repo_ships_code(dest: Path, repo_id: str) -> None:
    """Say plainly when a downloaded repo contains Python.

    ``snapshot_download`` fetches the WHOLE repo, so a HuggingFace repo's own .py
    lands on disk like any other file.

    The file is inert: ``hf_trust_remote_code`` defaults to False and a model
    that needs custom code is refused with an explanation instead of being run
    (see inference/backends/hf.py). Everything is fetched rather than filtered
    through an allow_patterns allowlist, which would silently produce a broken,
    half-downloaded model whenever it missed a needed file (chat templates
    (.jinja), merges.txt, shard index files, per-component subdirectories for
    multimodal repos).

    So: fetch everything, and make the presence of code VISIBLE at the moment it
    arrives.
    """
    try:
        py = sorted(p for p in dest.rglob("*.py") if p.is_file())
    except OSError as e:
        # A failed scan does not fail the download, and is reported as not checked
        # rather than as 'no code found'.
        console.print(f"[yellow]Could not check {repo_id} for bundled code: {e}[/yellow]")
        return
    if not py:
        return
    # escape(): these names come from a remote repo and are interpolated into a
    # Rich markup string. Unescaped, a name like '[/b]evil.py' raises MarkupError
    # and one like '[red]x.py' is parsed as a style tag and disappears from the
    # notice.
    from rich.markup import escape
    shown = ", ".join(escape(p.name) for p in py[:5])
    if len(py) > 5:
        shown += f", and {len(py) - 5} more"
    console.print(
        f"[yellow]Note:[/yellow] {escape(str(repo_id))} ships {len(py)} "
        f"Python file(s) ({shown}).")
    console.print(
        "[dim]  localm will NOT run them: model-bundled custom code is disabled "
        "by default. A model that requires it is refused with an explanation "
        "unless you enable 'hf_trust_remote_code'.[/dim]")


def _resolve_snapshot_type(dest: Path, model_type: str) -> str:
    """The type to register a downloaded HF snapshot under.

    The pipeline_tag probe runs BEFORE the download and answers 'unknown' for any
    repo without an exact tag (common for base and older repos). The files now on
    disk are a HARDER signal than that API record, so when the probe could not
    resolve, the real config.json is classified with the same deterministic
    reader ``add_local`` uses.

    A probe that DID resolve (lora/vae/embedding/...) is authoritative and is
    never overridden here, and an unresolvable config.json stays 'unknown'.
    """
    if model_type != "unknown":
        return model_type
    detected, _gmeta = _detect_local_model_type(dest, is_gguf=False, is_hf=True)
    logger.info("Snapshot %s typed %r from its downloaded config.json (the HF "
                "pipeline_tag probe could not resolve it)", dest.name, detected)
    if detected != "unknown":
        console.print(f"[green]Determined model type from config.json:[/green] "
                      f"[bold]{detected}[/bold]")
    else:
        # Still no hard signal even from the real files: say so. It stays runnable
        # by name, it just is not auto-loaded for chat.
        console.print(
            "[yellow]Could not determine this model's type[/yellow] - "
            "registering it as 'unknown'. Run it by name, or set its type "
            "later: [bold]localm set-type <name> <type>[/bold]")
    return detected


def _snapshot_is_complete(dest: Path, repo_siblings, repo_id: str) -> bool:
    """True when every file the remote repo listing names is present under *dest*
    at its stated size.

    A disk-full mid-download can leave config.json - usually one of the smallest,
    earliest files - on disk while weight shards are still missing, so an
    existence check on config.json alone would register a broken snapshot as a
    ready model on the very next retry. *repo_siblings* is None when the listing
    could not be fetched (offline / API error), which degrades to exactly that
    weaker check.

    Module-level rather than a closure, so the confinement below is directly
    testable."""
    # Imported inside the function so the CLI download path does not import fastapi
    # (pulled in by pathsafe for confined_name's HTTPException) just to validate a
    # filename.
    from localm import pathsafe
    if not (dest / "config.json").exists():
        return False
    if repo_siblings is None:
        return True
    for sib in repo_siblings:
        # rfilename comes from the remote model_info response and is not a trusted
        # path component: an absolute or drive-qualified value would replace dest
        # entirely, and a '..' walks out of it. Confine before the stat, never
        # after.
        try:
            fp = pathsafe.confined_under(dest, str(sib.rfilename))
        except ValueError as e:
            # An out-of-bounds name is logged and reports the snapshot INCOMPLETE,
            # so it re-downloads rather than registering a half-present tree.
            logger.warning("repo %s lists an out-of-bounds filename (%s); "
                           "treating the local snapshot as incomplete", repo_id, e)
            return False
        if not fp.is_file():
            return False
        if sib.size is not None and fp.stat().st_size != sib.size:
            return False
    return True


def _pull_hf_snapshot(
    repo_id: str,
    name: Optional[str],
    expected_sha256: Optional[str] = None,
    redownload: bool = False,
    model_type: str = "llm",
) -> bool:
    """Download a complete HuggingFace model repo (for transformers/HF format models)."""
    # A full-repo snapshot is many files with no single digest, so --sha256 is
    # refused with a message rather than ignored.
    if expected_sha256:
        console.print(
            "[red]--sha256 is not supported for a full HuggingFace repo[/red] "
            f"({repo_id}): a snapshot has many files and no single digest to "
            "verify. Drop --sha256, or pull a single file with "
            "[bold]owner/repo:file.gguf --sha256 <hash>[/bold]."
        )
        return False

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        console.print("[red]Missing:[/red] huggingface-hub  (run: uv pip install huggingface-hub)")
        return False

    model_name = _sanitize_name(name or repo_id.split("/")[-1])
    dest = _mm.MODELS_DIR / model_name

    # Fetch the repo's file listing once: it backs both the completeness check
    # (every file present at its stated size, not just config.json) and the
    # disk-space preflight / progress total below.
    repo_siblings = None
    total_size = 0
    try:
        from huggingface_hub import HfApi
        info = HfApi(endpoint=_HF_ENDPOINT).model_info(repo_id, files_metadata=True)
        repo_siblings = info.siblings
        total_size = sum(getattr(s, "size", None) or 0 for s in repo_siblings)
    except Exception as e:
        # Offline / API error: fall back to a config.json-only completeness check
        # below and an indeterminate (0) progress total.
        logger.debug("could not fetch file listing for %s (%s); falling back "
                     "to a config.json-only completeness check", repo_id, e)

    if dest.exists() and _snapshot_is_complete(dest, repo_siblings, repo_id):
        console.print(f"[yellow]Already downloaded:[/yellow] {model_name}")
        _mm._register_with_dedup(model_name, dest, f"hf:{repo_id}",
                                 model_type=_resolve_snapshot_type(dest, model_type))
        return True

    # Same repo already pulled under a different name?
    if not redownload:
        reg = _mm.load_registry()

        def _is_same_repo(info) -> bool:
            # Skip a malformed sibling entry (non-dict, or a null / non-string /
            # empty path) by routing it through _entry_path, so one corrupt entry
            # cannot crash the pull-dedup scan.
            epath = _mm._entry_path(info)
            if epath is None:
                return False
            return info.get("source") == f"hf:{repo_id}" and Path(epath).is_dir()

        same_source = sorted(n for n, info in reg.items() if _is_same_repo(info))
        if same_source:
            console.print(
                f"[yellow]This repo is already downloaded - registered as "
                f"{', '.join(repr(n) for n in same_source)}[/yellow]"
            )
            if not sys.stdin.isatty():
                console.print("[dim]Non-interactive session - skipping. "
                              "Use --redownload to force.[/dim]")
                return True
            import click
            choice = click.prompt(
                f"  [a]lias as '{model_name}'  [d]ownload anyway  [s]kip",
                type=click.Choice(["a", "d", "s"], case_sensitive=False),
                default="a", show_choices=False,
            )
            if choice.lower() == "s":
                return True
            if choice.lower() == "a":
                alias_model(same_source[0], model_name)
                return True

    # Collision check: model_name comes from _sanitize_name, a lossy coercion, and
    # is used as both the MODELS_DIR subdirectory name and the registry key with no
    # uniqueness check upstream, so two different repos - or a --name reused across
    # two pulls - can compute the same dest. snapshot_download MERGES into
    # local_dir instead of clearing it first, so pulling into an occupied folder
    # mixes the two repos' files together.
    #
    # The test: is there ALREADY a registry entry pointing at this exact dest for a
    # DIFFERENT source? A resumable partial of the SAME repo has no registry entry
    # yet, and a redownload of the model already registered here has a matching
    # source, so neither is refused.
    if dest.exists():
        reg_now = _mm.load_registry()
        foreign = [n for n in find_aliases_by_path(dest, reg_now)
                  if reg_now[n].get("source") != f"hf:{repo_id}"]
        if foreign:
            console.print(
                f"[red]Refusing to pull {repo_id} into {dest}:[/red] this "
                f"folder already holds a DIFFERENT model, registered as "
                f"{', '.join(repr(n) for n in foreign)}. Pulling here would "
                f"silently mix the two repos' files together. Pull with "
                f"[bold]-n <a-different-name>[/bold] to give this repo its "
                "own directory."
            )
            return False

    def _disk_bytes() -> int:
        return _snapshot_bytes_on_disk(dest)

    # Only the bytes still MISSING need room: snapshot_download resumes on top of
    # whatever already landed in dest. _pull_gguf_file sums only the `missing` parts
    # and _pull_url uses (total - already_have).
    if not _mm._check_disk_space(_mm.MODELS_DIR,
                                 max(0, total_size - _disk_bytes())):
        return False

    _mm.ensure_dirs()
    console.print(
        f"Downloading full model [bold cyan]{repo_id}[/bold cyan] "
        f"-> [bold]{dest}[/bold]"
    )
    console.print("[dim]This may take a while for large models...[/dim]")

    try:
        with _snapshot_progress(_disk_bytes, total_size) as _prog:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(dest),
                endpoint=_HF_ENDPOINT,
            )
            _prog.ok()
    except Exception as e:
        console.print(f"[red]Download failed:[/red] {e}")
        return False

    _warn_if_repo_ships_code(dest, repo_id)
    # Registration goes through _register_with_dedup.
    #
    # Its bool return is checked, not assumed True: the dest-collision block above
    # only guards a foreign occupant already at THIS path, and model_name can still
    # be taken by a DIFFERENT path, which _register_with_dedup declines
    # non-interactively. Real bytes are already on disk by then, so the outcome is
    # reported rather than assumed.
    registered = _mm._register_with_dedup(
        model_name, dest, f"hf:{repo_id}",
        model_type=_resolve_snapshot_type(dest, model_type))
    if not registered:
        console.print(
            f"[yellow]{repo_id} was downloaded to {dest}, but could not be "
            f"registered as '{model_name}'[/yellow] (see message above) - "
            "the files are on disk. Retry with a different -n name, or "
            "'localm alias' it in."
        )
        return False
    _report_success(f"[green]✓[/green] [bold]{model_name}[/bold] downloaded to {dest}",
                    f"[green]OK[/green] [bold]{model_name}[/bold] downloaded to {dest}")
    return True




def _ssrf_resolve_final_url(url: str) -> str:
    """Follow the redirect chain HEAD-only, re-validating EVERY hop against the
    netpolicy SSRF guard, and return the final URL. Model pulls legitimately
    redirect (HuggingFace -> CDN), so redirects are followed, but each hop is
    checked rather than trusting requests' automatic, UNCHECKED redirect
    following, which a public URL could use to bounce the download into
    127.0.0.1 / 169.254.169.254 / an RFC1918 service. Each HEAD is IP-pinned to
    the validated address so the connect cannot rebind off the checked host.
    Raises NetworkPolicyError if any hop resolves to a non-public host or cannot
    be resolved to a validated address."""
    import urllib.parse

    from localm import netpolicy
    current = url
    for _ in range(6):
        netpolicy.check_url(current)
        try:
            resp = netpolicy.pinned_request(
                "HEAD", current, allow_redirects=False, timeout=10)
        except netpolicy.NetworkPolicyError:
            raise                       # a policy refusal / rebind must NOT be swallowed into a silent GET
        except Exception:
            break                       # unreachable HEAD -> let the GET report it
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("Location")
            if not loc:
                break
            current = urllib.parse.urljoin(current, loc)
            continue
        break
    netpolicy.check_url(current)        # final target, revalidated
    return current


def _pull_url(
    url: str,
    name: str,
    expected_sha256: Optional[str] = None,
    redownload: bool = False,
    model_type: str = "llm",
) -> bool:
    """Download a model from a direct URL with resumable .part file support."""
    import requests
    from localm.netpolicy import NetworkPolicyError

    stem = _stem_from_url(url)
    if not stem:
        console.print(
            f"[red]Invalid URL - no file name in the path:[/red] {url}\n"
            "A direct URL must point at a file, e.g. https://host/model.gguf"
        )
        return False

    filename = stem + ".gguf"

    # Traversal guard: the filename is derived from an untrusted URL path segment,
    # so confine it to MODELS_DIR before using it as a dest.
    safe = _safe_models_filename(filename)
    if safe is None:
        console.print(
            f"[red]Unsafe model filename derived from URL:[/red] {filename}\n"
            "The download destination must be a single name inside the models "
            "folder (no '/', '\\', or '..')."
        )
        return False
    filename = safe

    dest      = _mm.MODELS_DIR / filename
    part_file = _mm.MODELS_DIR / (filename + ".part")

    if dest.exists():
        # A file with this derived name is already here. It counts as the requested
        # model only when the caller's --sha256, if given, matches its bytes; a new
        # name is never aliased onto unrelated existing bytes.
        if expected_sha256:
            on_disk = _verify_digest(dest, purpose="to check the file already here")
            if on_disk.lower() != expected_sha256.lower():
                console.print(
                    f"[red]SHA256 mismatch![/red] A different file already "
                    f"occupies {filename} ({on_disk[:16]}…); it does not match "
                    f"--sha256 ({expected_sha256.lower()[:16]}…). Refusing to "
                    "alias onto unrelated bytes - use --redownload or a "
                    "different -n name."
                )
                return False
            console.print(
                f"[yellow]Already downloaded:[/yellow] {filename} "
                f"[dim](sha256 verified)[/dim]"
            )
        else:
            console.print(f"[yellow]Already downloaded:[/yellow] {filename}")
        _mm._register_with_dedup(name, dest, url, model_type=model_type)
        return True

    # Pre-download check by user-supplied hash (URL servers can't tell us one)
    if expected_sha256 and not redownload:
        dups = _mm.find_by_sha256(expected_sha256)
        if dups:
            action = _mm._prompt_predownload_dup(dups, name)
            if action == "skip":
                return True
            if action == "alias":
                alias_model(dups[0], name)
                return True

    _mm.ensure_dirs()

    # Determine how much we already have (from a prior interrupted download)
    already_have = part_file.stat().st_size if part_file.exists() else 0

    # Resolve the redirect chain with each hop validated, then use the final CHECKED
    # URL for both the size HEAD and the streaming GET with redirects OFF, so no
    # unchecked hop can bounce the download into an internal host.
    from localm import netpolicy
    try:
        dl_url = _ssrf_resolve_final_url(url)
    except NetworkPolicyError as e:
        console.print(f"[red]Refused by network policy:[/red] {e}")
        return False

    # HEAD the final URL for the total file size used by the disk-space check.
    # Pinned to the validated IP like the GET below; a connect error here is
    # non-fatal, while the GET fails closed regardless.
    try:
        head  = netpolicy.pinned_request("HEAD", dl_url, allow_redirects=False, timeout=10)
        total = int(head.headers.get("content-length", 0))
    except NetworkPolicyError as e:
        # A policy refusal fails closed, like the GET below, instead of collapsing
        # into total=0.
        console.print(f"[red]Refused by network policy:[/red] {e}")
        return False
    except Exception as e:
        # Benign: a HEAD connect error only costs us the disk-space pre-check; the
        # GET still fails closed. Log at debug so the failure stays discoverable.
        logger.debug("size HEAD failed for %s (non-fatal, size unknown): %s", dl_url, e)
        total = 0

    remaining = max(0, total - already_have)
    if not _mm._check_disk_space(_mm.MODELS_DIR, remaining):
        return False

    # Build request - try to resume from where we left off
    headers: dict = {}
    if already_have:
        headers["Range"] = f"bytes={already_have}-"
        console.print(
            f"Resuming [bold cyan]{url}[/bold cyan] "
            f"[dim](skipping first {already_have / 1024**2:.1f} MB)[/dim]"
        )
    else:
        console.print(f"Downloading [bold cyan]{url}[/bold cyan]")

    try:
        netpolicy.check_url(dl_url)     # revalidate immediately before the connect
        r = netpolicy.pinned_request("GET", dl_url, headers=headers, stream=True,
                                     timeout=30, allow_redirects=False)
        if r.status_code in (301, 302, 303, 307, 308):
            console.print(f"[red]Refused:[/red] unexpected redirect from {dl_url}")
            return False
        r.raise_for_status()
    except NetworkPolicyError as e:
        console.print(f"[red]Refused by network policy:[/red] {e}")
        return False
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", "?")
        console.print(f"[red]Download failed[/red] (HTTP {code}): {url}")
        return False
    except requests.RequestException as e:
        console.print(f"[red]Could not reach[/red] {url}: {e}")
        return False

    # Server may ignore the Range header - detect and reset if needed
    if already_have and r.status_code == 200:
        # Server returned the full file despite Range request
        already_have = 0

    content_length = int(r.headers.get("content-length", 0))
    # Without a content-length the total stays unknown, even when bytes are already
    # on disk: adding already_have to a zero content_length would make the total
    # equal already_have and report a stuck 100% for the whole resumed transfer.
    total_display = (already_have + content_length) if content_length else None

    def _write_chunks(on_chunk=None):
        mode = "ab" if already_have else "wb"
        with open(part_file, mode) as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
                if on_chunk is not None:
                    on_chunk(len(chunk))

    if os.environ.get("LOCALM_PROGRESS_JSON") == "1":
        # GUI mode: stream JSON progress polled from the .part file on disk, the
        # same mechanism the HuggingFace path uses (_download_progress). The Rich
        # bar is skipped here; there is no terminal, and its ANSI would clutter the
        # stdout the GUI parses.
        def _part_bytes() -> int:
            try:
                return part_file.stat().st_size
            except OSError:
                return 0
        with _snapshot_progress(_part_bytes, total_display or 0) as _prog:
            _write_chunks()
            _prog.ok()
    else:
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as prog:
            task = prog.add_task(filename, total=total_display, completed=already_have)
            _write_chunks(lambda n: prog.update(task, advance=n))

    # Atomically rename on successful completion
    part_file.rename(dest)

    # SHA256 verification
    actual = _verify_digest(dest)
    if expected_sha256:
        if actual.lower() == expected_sha256.lower():
            _report_success(f"[green]✓[/green] SHA256 verified: {actual[:16]}…",
                            f"[green]OK[/green] SHA256 verified: {actual[:16]}…")
        else:
            console.print(
                f"[red]SHA256 mismatch![/red] Expected {expected_sha256[:16]}…, "
                f"got {actual[:16]}… - deleting corrupted file"
            )
            dest.unlink()
            return False
    else:
        console.print(f"[dim]SHA256: {actual}[/dim]")

    # Post-download identity check: is the downloaded file byte-identical to
    # something already registered? A URL download has no hash up front, so this is
    # the first point it can be detected.
    dups = [n for n in _mm.find_by_sha256(actual) if n != name]
    if dups and not redownload:
        names = ", ".join(f"'{n}'" for n in dups)
        console.print(
            f"[yellow]Downloaded file is byte-identical to {names}[/yellow]"
        )
        if sys.stdin.isatty():
            import click
            choice = click.prompt(
                f"  [a]lias as '{name}' and delete the duplicate file  "
                "[k]eep both copies",
                type=click.Choice(["a", "k"], case_sensitive=False),
                default="a", show_choices=False,
            )
            if choice.lower() == "a":
                # Route through _entry_path like every other registry consumer, so a
                # null/int path and a '..' component are both rejected before the
                # value decides an unlink().
                epath = _mm._entry_path(_mm.load_registry().get(dups[0]))
                if epath is None:
                    # An unreadable sibling entry keeps both copies and is reported;
                    # it never licenses deleting a file.
                    console.print(
                        f"[yellow]Registry entry for {dups[0]!r} is malformed - "
                        "keeping both copies rather than deleting a file on the "
                        "strength of an unreadable path.[/yellow]"
                    )
                else:
                    existing_path = Path(epath)
                    if dest.resolve() != existing_path.resolve():
                        dest.unlink()
                    alias_model(dups[0], name)
                    return True

    _mm._register(name, dest, url, sha256=actual, model_type=model_type)
    _report_success(f"[green]✓[/green] [bold]{name}[/bold] is ready",
                    f"[green]OK[/green] [bold]{name}[/bold] is ready")
    return True


def _hf_pipeline_tag_to_type(repo_id: str) -> str:
    """Classify a HuggingFace repo's model type from HARD metadata. The real
    implementation lives in localm.discover (shared with search-result
    classification there); the import is lazy, so localm.discover is not imported
    at module scope."""
    from localm.discover import _hf_pipeline_tag_to_type as _classify
    return _classify(repo_id)

