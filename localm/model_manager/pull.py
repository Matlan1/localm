# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model download/transport: pull routing, the HF + URL + Ollama backends,
resumable downloads, hashing-on-the-wire, and GUI progress streaming."""

import localm.model_manager as _mm  # read package-patchable names at call time

import contextlib
import json
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
from ._shared import PROGRESS_SENTINEL
from ._shared import console
from .gguf import _safe_models_filename
from .gguf import split_gguf_parts
from .registry import _sanitize_name
from .registry import alias_model




def _emit_progress(downloaded: int, total: int, *, phase: str = "download",
                   name: "str | None" = None, index: int = 0, count: int = 0) -> None:
    pct = round(downloaded * 100 / total, 1) if total else None
    payload = {"phase": phase, "downloaded": downloaded, "total": total, "pct": pct}
    # R06: for a multi-file download (a split GGUF), tell the GUI which file is in
    # flight so it can show "file 2 of 3: <name>". Omitted for a single file so the
    # single-file progress UX is unchanged.
    if count > 1:
        payload["count"] = count
        payload["index"] = index
        if name:
            payload["name"] = name
    sys.stdout.write(PROGRESS_SENTINEL + json.dumps(payload) + "\n")
    sys.stdout.flush()




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




@contextlib.contextmanager
def _download_progress(target_parts: List[Path], total_size: int):
    """Stream JSON download progress while files land under MODELS_DIR.

    Active only in GUI mode (LOCALM_PROGRESS_JSON=1) with a known total - a
    no-op otherwise, so the CLI keeps huggingface_hub's tqdm bars. Progress is
    measured from bytes on disk (completed parts + the growing ``.incomplete``
    temp file), which is robust across huggingface_hub versions.
    """
    if os.environ.get("LOCALM_PROGRESS_JSON") != "1" or not total_size:
        yield
        return

    stop = threading.Event()

    def _downloaded_bytes() -> int:
        done = 0
        for p in target_parts:
            try:
                done += p.stat().st_size
            except OSError:
                pass  # not finished yet
        active = 0
        cache = _mm.MODELS_DIR / ".cache"
        if cache.is_dir():
            try:
                for f in cache.rglob("*.incomplete"):
                    try:
                        active += f.stat().st_size
                    except OSError:
                        pass
            except OSError:
                pass
        return min(done + active, total_size)

    def _poll() -> None:
        last = -1
        while not stop.is_set():
            dl = _downloaded_bytes()
            if dl != last:
                last = dl
                fn, fi, fc = _progress_file_info(target_parts)
                _emit_progress(dl, total_size, name=fn, index=fi, count=fc)
            stop.wait(0.7)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    fn0, fi0, fc0 = _progress_file_info(target_parts)
    _emit_progress(0, total_size, name=fn0, index=fi0, count=fc0)
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=2)
        fn1, fi1, fc1 = _progress_file_info(target_parts)
        _emit_progress(total_size, total_size, name=fn1, index=fi1, count=fc1)




@contextlib.contextmanager
def _snapshot_progress(disk_bytes_fn, total_size: int):
    """Like _download_progress but for snapshot_download (many files): byte
    count comes from a caller-supplied directory-size function. Indeterminate
    (total_size == 0) still streams a 'downloading' phase so the GUI can show
    a busy bar; no-op outside GUI mode."""
    if os.environ.get("LOCALM_PROGRESS_JSON") != "1":
        yield
        return

    stop = threading.Event()

    def _poll() -> None:
        last = -1
        while not stop.is_set():
            dl = disk_bytes_fn()
            if total_size:
                dl = min(dl, total_size)
            if dl != last:
                last = dl
                _emit_progress(dl, total_size)
            stop.wait(0.7)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    _emit_progress(0, total_size)
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=2)
        if total_size:
            _emit_progress(total_size, total_size)




def pull_model(
    model_spec: str,
    name: Optional[str] = None,
    expected_sha256: Optional[str] = None,
    redownload: bool = False,
    mmproj_spec: Optional[str] = None,
    model_type: str = "auto",
    store: Optional[str] = None,
) -> bool:
    """Download a model from HuggingFace or a URL.

    Returns True on success or a benign no-op (already present / aliased /
    user-skipped), False on a real error, so callers can set a non-zero exit
    code and the GUI can mark the job failed instead of reporting "finished".

    *store* ("copy" / "move" / None) only applies to the local-path branch
    below - a remote HF/URL download already lands in MODELS_DIR on its own.
    """
    spec = _mm.resolve_spec(model_spec)

    # A local filesystem path is not a remote spec: register it in place rather
    # than mis-parsing a Windows drive-colon as an owner/repo:file spec, or
    # rejecting it as "Unknown spec" (H1). This is checked BEFORE any network
    # auto-detect so registering a local file never leaks its path (and model
    # filename) to huggingface.co - a POSIX absolute path or a forward-slash
    # relative path both contain "/" and would otherwise be probed against HF
    # (AUDIT-HIGH-6). Only an absolute path or an existing file counts, so a bare
    # HF "owner/repo" is never shadowed by a same-named local directory. add_local
    # does the validation + dedup.
    try:
        local = Path(model_spec).expanduser()
        is_local_path = local.exists() and (local.is_absolute() or local.is_file())
    except OSError:
        is_local_path = False
    if is_local_path:
        # FAC-5 / AGENTS.md rule 5: a user-supplied --sha256 is a SAFETY assertion.
        # For a local file we can and MUST actually verify it, never register the
        # file and report success while silently ignoring the hash (a false-success
        # that tells the user integrity held when it was never checked). A full HF
        # repo refuses --sha256 outright; a single local file we verify against the
        # real bytes and refuse on mismatch, mirroring the URL/GGUF download paths.
        if expected_sha256:
            if local.is_dir():
                console.print(
                    "[red]--sha256 is not supported for a local directory[/red] "
                    f"({local}): a folder has many files and no single digest to "
                    "verify. Drop --sha256, or point at a single .gguf file.")
                return False
            want = expected_sha256.strip().lower()
            actual = _mm._sha256_file(local).lower()
            if actual != want:
                console.print(
                    f"[red]SHA256 mismatch![/red] {local} is {actual[:16]}…, not "
                    f"--sha256 {want[:16]}…. Refusing to register.")
                return False
            console.print(f"[green]✓[/green] SHA256 verified: {actual[:16]}…")
        # A local file gets no remote type probe: honour an explicit --type, else let
        # add_local deterministically detect it (GGUF -> llm, HF dir -> config.json,
        # otherwise the 'unknown' sentinel rather than a silent 'llm').
        local_type = None if model_type == "auto" else model_type
        return _mm.add_local(str(local), name=name, model_type=local_type, store=store)

    # SSRF-PULL: honour the net_mode kill switch for a REMOTE pull. net_mode=off
    # means "no network at all", so it must stop a model download - and the type
    # auto-detect probe below - too.
    from localm.netpolicy import network_mode
    if network_mode() == "off":
        console.print(
            "[red]Network access is disabled (net_mode=off).[/red] A model pull "
            "needs the network; enable it with: localm config net_mode ask")
        return False

    # Remote spec: resolve the model type (a network probe against HF for a bare
    # owner/repo). Only reached for confirmed-remote specs, after the local-path
    # and net_mode gates above.
    detected_type = "llm"
    if model_type == "auto":
        if "/" in spec and not (spec.startswith("http://") or spec.startswith("https://")):
            repo_id = spec.split(":")[0] if ":" in spec else spec
            # A named .gguf file is a hard LLM signal (the container format itself),
            # exactly like a local `add` of a .gguf - so type it 'llm' and do NOT
            # fall back to the repo's HF pipeline_tag, which a GGUF-quant repo often
            # lacks, mislabeling the model 'unknown'. An 'unknown' type then hides
            # the model from the desktop launcher and blocks auto-chat selection, so
            # a plain `localm pull owner/repo:model.gguf` would vanish from the UI.
            # Catch both file forms the dispatch below accepts (owner/repo:file.gguf
            # AND owner/repo/file.gguf) via the last path segment; only probe the
            # pipeline tag for a bare repo (no specific .gguf file).
            if spec.rsplit("/", 1)[-1].lower().endswith(".gguf"):
                detected_type = "llm"
            else:
                detected_type = _hf_pipeline_tag_to_type(repo_id)
                logger.info("Auto-detected model type for %s: %s", repo_id, detected_type)
                if detected_type == "unknown":
                    # Surface the honest result (AGENTS.md rule 5): it won't be
                    # auto-loaded for chat, but it stays runnable by name.
                    console.print(
                        "[yellow]Could not determine this model's type[/yellow] - "
                        "registering it as 'unknown'. Run it by name, or set its type "
                        "later: [bold]localm set-type <name> <type>[/bold]")
        else:
            detected_type = "llm"
    else:
        detected_type = model_type

    if spec.startswith("http://") or spec.startswith("https://"):
        res = _pull_url(spec, _sanitize_name(name or _stem_from_url(spec)),
                         expected_sha256=expected_sha256, redownload=redownload,
                         model_type=detected_type)
    elif "/" in spec:
        if ":" in spec or spec.rsplit("/", 1)[-1].endswith(".gguf"):
            # owner/repo:file.gguf  or  owner/repo/file.gguf  -> single GGUF file
            res = _mm._pull_gguf_file(spec, name, expected_sha256=expected_sha256,
                                   redownload=redownload, model_type=detected_type)
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

    if res and mmproj_spec:
        console.print(f"Pulling mmproj: {mmproj_spec}")
        if ":" in mmproj_spec or mmproj_spec.rsplit("/", 1)[-1].endswith(".gguf"):
            _mm._pull_gguf_file(mmproj_spec, name=None, register=False)
        else:
            console.print("[red]mmproj spec must be a specific file (owner/repo:file.gguf)[/red]")

    return res




def _stem_from_url(url: str) -> str:
    return url.split("/")[-1].split("?")[0].removesuffix(".gguf")




def _check_disk_space(dest_dir: Path, required_bytes: int) -> bool:
    """
    Verify there is at least *required_bytes* of free space on the volume that
    holds *dest_dir*.  Prints a warning and returns False when space is
    insufficient; returns True when fine or when the check is skipped
    (e.g. ``required_bytes == 0``).

    If the free-space check itself cannot be measured (offline models dir,
    permission denied, etc.) it is treated as OK by design: a WARNING is
    logged and the download proceeds rather than blocking a working setup.
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
        info = HfApi().get_paths_info(repo_id, [filename])
        if info:
            lfs = getattr(info[0], "lfs", None)
            digest = getattr(lfs, "sha256", None) if lfs else None
            return digest.lower() if digest else None
    except Exception:
        pass
    return None




def _pull_gguf_file(
    spec: str,
    name: Optional[str],
    expected_sha256: Optional[str] = None,
    redownload: bool = False,
    register: bool = True,
    model_type: str = "llm",
) -> bool:
    """Download a single .gguf file from a HuggingFace repo.

    ``expected_sha256`` is the user-supplied ``--sha256`` digest. It is NOT a
    facade here (FAC-5): when given it is reconciled with HuggingFace's own LFS
    metadata up front, and the downloaded first part is verified against it
    before the model is registered.
    """
    try:
        from huggingface_hub import hf_hub_download, hf_hub_url
    except ImportError:
        console.print("[red]Missing:[/red] huggingface-hub  (run: uv pip install huggingface-hub)")
        return False

    if ":" in spec:
        repo_id, filename = spec.rsplit(":", 1)
    else:
        parts = spec.rsplit("/", 1)
        repo_id, filename = parts[0], parts[1]

    # Split GGUF: normalise to the full ordered part list. llama.cpp loads
    # the model from the first part, so that's what gets registered.
    all_parts = split_gguf_parts(filename) or [filename]
    filename  = all_parts[0]

    # Traversal guard (GAP-CLI-2): the filename comes from an untrusted spec
    # (owner/repo:../../evil.gguf), so confine every part to MODELS_DIR before
    # it is used as a destination. Reject the whole pull on any unsafe part.
    for part in all_parts:
        if _safe_models_filename(part) is None:
            console.print(
                f"[red]Unsafe model filename:[/red] {part}\n"
                "A GGUF filename must be a single name inside the models folder "
                "(no '/', '\\', or '..')."
            )
            return False

    model_name = _sanitize_name(name or filename.removesuffix(".gguf"))
    dest = _mm.MODELS_DIR / filename

    # Expected digest from HF metadata - free, no download needed.
    # (Only identifies the first part of a split GGUF, which is enough.)
    expected = _mm._hf_file_sha256(repo_id, filename)

    # FAC-5: honour a user-supplied --sha256. If HF's own metadata digest is
    # known and disagrees with it, the bytes can never match - refuse up front
    # rather than spending a download to discover the mismatch.
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

    missing = [p for p in all_parts if not (_mm.MODELS_DIR / p).exists()]
    if not missing:
        console.print(f"[yellow]Already downloaded:[/yellow] {filename}")
        # If the user asserted a hash, verify the file actually on disk before
        # treating it as the requested model.
        if want:
            on_disk = _mm._sha256_file(dest)
            if on_disk.lower() != want:
                console.print(
                    f"[red]SHA256 mismatch![/red] The file already at {filename} "
                    f"({on_disk[:16]}…) does not match --sha256 ({want[:16]}…)."
                )
                return False
        if register:
            _mm._register_with_dedup(model_name, dest, f"hf:{repo_id}",
                                 digest=verify_digest, model_type=model_type)
        return True

    # Pre-download duplicate check: same bytes already on disk elsewhere?
    if verify_digest and not redownload:
        dups = _mm.find_by_sha256(verify_digest)
        if dups:
            action = _mm._prompt_predownload_dup(dups, model_name)
            if action == "skip":
                return True
            if action == "alias":
                alias_model(dups[0], model_name)
                return True
            # "download" falls through

    _mm.ensure_dirs()

    # Disk space pre-flight - HEAD each missing part's CDN URL for Content-Length
    try:
        import requests as _req
        total_size = 0
        for part in missing:
            cdn_url = hf_hub_url(repo_id, part)
            head    = _req.head(cdn_url, allow_redirects=True, timeout=10)
            total_size += int(head.headers.get("content-length", 0))
    except Exception:
        total_size = 0

    if not _mm._check_disk_space(_mm.MODELS_DIR, total_size):
        return False

    if len(all_parts) > 1:
        console.print(
            f"Pulling [bold cyan]{repo_id}[/bold cyan] / [bold]{filename}[/bold] "
            f"[dim](split GGUF, {len(all_parts)} parts, "
            f"{len(missing)} to download)[/dim]"
        )
    else:
        console.print(f"Pulling [bold cyan]{repo_id}[/bold cyan] / [bold]{filename}[/bold]")

    with _download_progress([_mm.MODELS_DIR / p for p in missing], total_size):
        for part in missing:
            try:
                local = hf_hub_download(
                    repo_id=repo_id,
                    filename=part,
                    local_dir=str(_mm.MODELS_DIR),
                )
                final = _mm.MODELS_DIR / part
                if Path(local) != final:
                    shutil.move(local, final)
            except Exception as e:
                console.print(f"[red]Download failed[/red] ({part}): {e}")
                return False

    # FAC-5: verify the downloaded first part against the user's --sha256.
    # (HF metadata is already trusted; we only need to confirm a user assertion
    # against the real bytes.) On mismatch, delete the part(s) and fail.
    if want:
        actual = _mm._sha256_file(dest).lower()
        if actual != want:
            console.print(
                f"[red]SHA256 mismatch![/red] Expected {want[:16]}…, got "
                f"{actual[:16]}… - deleting downloaded file(s)"
            )
            for part in all_parts:
                p = _mm.MODELS_DIR / part
                if p.exists():
                    p.unlink()
            return False
        console.print(f"[green]✓[/green] SHA256 verified: {actual[:16]}…")

    if register:
        _mm._register(model_name, _mm.MODELS_DIR / filename, f"hf:{repo_id}",
                  sha256=verify_digest, model_type=model_type)
        console.print(f"[green]✓[/green] [bold]{model_name}[/bold] is ready")
    else:
        console.print(f"[green]✓[/green] [bold]{filename}[/bold] downloaded")
    return True




def _pull_hf_snapshot(
    repo_id: str,
    name: Optional[str],
    expected_sha256: Optional[str] = None,
    redownload: bool = False,
    model_type: str = "llm",
) -> bool:
    """Download a complete HuggingFace model repo (for transformers/HF format models)."""
    # FAC-5: a full-repo snapshot is many files; there is no single digest to
    # check --sha256 against. Refuse the flag with a clear message rather than
    # silently ignoring it (which would give a false sense of verification).
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

    # Fetch the repo's file listing once - used both to verify an existing
    # download is genuinely complete (every file present with a matching size,
    # not just config.json) and to size the disk-space preflight / progress
    # display below. A disk-full mid-download can leave config.json - usually
    # one of the smallest, earliest files - on disk while weight shards are
    # still missing; checking only config.json's existence would then register
    # that broken snapshot as a ready model on the very next retry.
    repo_siblings = None
    total_size = 0
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id, files_metadata=True)
        repo_siblings = info.siblings
        total_size = sum(getattr(s, "size", None) or 0 for s in repo_siblings)
    except Exception as e:
        # Offline / API error: fall back to a config.json-only completeness
        # check below and an indeterminate (0) progress total - best effort,
        # matching how _pull_gguf_file/_pull_url degrade when a size HEAD fails.
        logger.debug("could not fetch file listing for %s (%s); falling back "
                     "to a config.json-only completeness check", repo_id, e)

    def _snapshot_is_complete() -> bool:
        if not (dest / "config.json").exists():
            return False
        if repo_siblings is None:
            return True
        for sib in repo_siblings:
            fp = dest / sib.rfilename
            if not fp.is_file():
                return False
            if sib.size is not None and fp.stat().st_size != sib.size:
                return False
        return True

    if dest.exists() and _snapshot_is_complete():
        console.print(f"[yellow]Already downloaded:[/yellow] {model_name}")
        _mm._register_with_dedup(model_name, dest, f"hf:{repo_id}", model_type=model_type)
        return True

    # Same repo already pulled under a different name?
    if not redownload:
        reg = _mm.load_registry()

        def _is_same_repo(info) -> bool:
            # Skip a malformed sibling entry (non-dict, or a null / non-string /
            # empty path): a single corrupt entry must not crash the pull-dedup
            # scan (a str entry's .get / Path(None) would). Mirrors #562's registry
            # consumers - route every entry through _entry_path.
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

    if not _mm._check_disk_space(_mm.MODELS_DIR, total_size):
        return False

    _mm.ensure_dirs()
    console.print(
        f"Downloading full model [bold cyan]{repo_id}[/bold cyan] "
        f"-> [bold]{dest}[/bold]"
    )
    console.print("[dim]This may take a while for large models...[/dim]")

    def _disk_bytes() -> int:
        try:
            return sum(f.stat().st_size for f in dest.rglob("*")
                       if f.is_file() and ".cache" not in f.parts)
        except OSError:
            return 0

    try:
        with _snapshot_progress(_disk_bytes, total_size):
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(dest),
            )
    except Exception as e:
        console.print(f"[red]Download failed:[/red] {e}")
        return False

    _mm._register(model_name, dest, f"hf:{repo_id}", model_type=model_type)
    console.print(f"[green]✓[/green] [bold]{model_name}[/bold] downloaded to {dest}")
    return True




def _ssrf_resolve_final_url(url: str) -> str:
    """Follow the redirect chain HEAD-only, re-validating EVERY hop against the
    netpolicy SSRF guard, and return the final URL. Model pulls legitimately
    redirect (HuggingFace -> CDN), so we follow - but check each hop instead of
    trusting requests' automatic, UNCHECKED redirect following, which a public
    URL could otherwise use to bounce the download into 127.0.0.1 /
    169.254.169.254 / an RFC1918 service (SSRF-PULL). Each HEAD is IP-pinned to the
    validated address so the connect cannot rebind off the checked host
    (SSRF-REBIND). Raises NetworkPolicyError if any hop resolves to a non-public
    host or cannot be resolved to a validated address."""
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

    # Traversal guard (GAP-CLI-2): the filename is derived from an untrusted URL
    # path segment, so confine it to MODELS_DIR before using it as a dest.
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
        # A file with this derived name is already here. Only treat it as the
        # requested model if the caller's --sha256 (when given) matches its
        # bytes - never alias a new name onto unrelated existing bytes
        # (GAP-CLI-2).
        if expected_sha256:
            on_disk = _mm._sha256_file(dest)
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

    # SSRF-PULL: resolve the redirect chain with each hop validated, then use the
    # final CHECKED URL for both the size HEAD and the streaming GET with redirects
    # OFF - so no unchecked hop can bounce the download into an internal host.
    from localm import netpolicy
    try:
        dl_url = _ssrf_resolve_final_url(url)
    except NetworkPolicyError as e:
        console.print(f"[red]Refused by network policy:[/red] {e}")
        return False

    # HEAD the final URL to get total file size for the disk space check. Pinned to
    # the validated IP like the GET below; a connect error here is non-fatal (the
    # size is only for the disk-space check), but the GET fails closed regardless.
    try:
        head  = netpolicy.pinned_request("HEAD", dl_url, allow_redirects=False, timeout=10)
        total = int(head.headers.get("content-length", 0))
    except NetworkPolicyError as e:
        # A policy refusal is NOT the benign case: surface it and fail closed
        # (like the GET below), rather than collapsing it into total=0 - do not
        # let a rebind/deny slip through the size probe (AGENTS.md rule 5).
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
    total_display  = (already_have + content_length) or None

    def _write_chunks(on_chunk=None):
        mode = "ab" if already_have else "wb"
        with open(part_file, mode) as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
                if on_chunk is not None:
                    on_chunk(len(chunk))

    if os.environ.get("LOCALM_PROGRESS_JSON") == "1":
        # GUI mode: stream JSON progress polled from the .part file on disk - the
        # same mechanism the HuggingFace path uses (_download_progress). Direct-URL
        # pulls used to emit only a Rich bar, which the GUI cannot render, so a URL
        # download looked frozen until it finished (G1). Skip the Rich bar here:
        # there is no terminal, and its ANSI would only clutter the captured stdout
        # the GUI parses.
        def _part_bytes() -> int:
            try:
                return part_file.stat().st_size
            except OSError:
                return 0
        with _snapshot_progress(_part_bytes, total_display or 0):
            _write_chunks()
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
    actual = _mm._sha256_file(dest)
    if expected_sha256:
        if actual.lower() == expected_sha256.lower():
            console.print(f"[green]✓[/green] SHA256 verified: {actual[:16]}…")
        else:
            console.print(
                f"[red]SHA256 mismatch![/red] Expected {expected_sha256[:16]}…, "
                f"got {actual[:16]}… - deleting corrupted file"
            )
            dest.unlink()
            return False
    else:
        console.print(f"[dim]SHA256: {actual}[/dim]")

    # Post-download identity check: did we just download a byte-identical
    # copy of something already registered? (URL downloads can't know the
    # hash up front, so this is the earliest possible detection point.)
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
                existing_path = Path(_mm.load_registry()[dups[0]]["path"])
                if dest.resolve() != existing_path.resolve():
                    dest.unlink()
                alias_model(dups[0], name)
                return True

    _mm._register(name, dest, url, sha256=actual, model_type=model_type)
    console.print(f"[green]✓[/green] [bold]{name}[/bold] is ready")
    return True


def _hf_pipeline_tag_to_type(repo_id: str) -> str:
    """Classify a HuggingFace repo's model type from HARD metadata (pipeline_tag,
    library_name, and exact tag tokens).

    Matching is EXACT, never substring: a tag that merely CONTAINS 'vae' / 'lora' /
    'clip' (e.g. 'exploration' contains 'lora') must NOT be misclassified (MED-15).
    Returns the 'unknown' sentinel - not a silent 'llm' - when no hard signal
    resolves (including an offline/failed query), so an ambiguous pull is registered
    honestly and is not auto-loaded as the chat model. Embedding models are
    provisioned via `setup-embeddings`, not pulled into the chat registry, so there
    is no 'embedding' type here.
    """
    from localm.discover import _get, HF_API
    try:
        data = _get(f"{HF_API}/api/models/{repo_id}", {"full": "false"})
        if isinstance(data, dict):
            tag = data.get("pipeline_tag")
            library = (data.get("library_name") or "").strip().lower()
            # Exact, lowercased tag tokens - a set so membership is equality, not
            # substring containment.
            tags = {str(t).strip().lower() for t in data.get("tags", []) if isinstance(t, str)}

            # Media / auxiliary types (exact pipeline_tag or exact tag token). These
            # are checked before the generic text-generation LLM signal because a
            # LoRA/VAE repo can also carry a text-generation pipeline tag.
            if tag in ("text-to-image", "image-to-image", "text-to-audio", "audio-to-audio"):
                return "diffusion-unet"
            if "vae" in tags:
                return "vae"
            if "lora" in tags or library == "peft":
                return "lora"
            if {"text-encoder", "clip"} & tags:
                return "text-encoder"
            # Text generation / chat model.
            if tag in ("text-generation", "text2text-generation", "conversational"):
                return "llm"
    except Exception as e:
        logger.debug("HF pipeline tag query failed for %s: %s", repo_id, e)
        return "unknown"
    return "unknown"

