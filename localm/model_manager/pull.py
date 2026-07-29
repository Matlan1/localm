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
from .registry import _detect_local_model_type, _sanitize_name
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
                # A bare owner/repo pull is a full snapshot, and _pull_hf_snapshot
                # gets a second, HARDER look at the config.json it downloads, so it
                # announces the outcome itself. Saying "registering it as 'unknown'"
                # here would be a false statement whenever that resolves the type
                # (REG-477). Every other spec in this branch registers the probe's
                # answer as-is, so it is announced now.
                if detected_type == "unknown" and ":" in spec:
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

    is_url_spec = spec.startswith("http://") or spec.startswith("https://")
    is_single_file_spec = not is_url_spec and "/" in spec and (
        ":" in spec or spec.rsplit("/", 1)[-1].endswith(".gguf"))
    if dest_dir is not None and not is_single_file_spec:
        # dest_dir routing is only wired through _pull_gguf_file - refuse rather
        # than silently downloading to MODELS_DIR while the caller believed it
        # went to dest_dir (AGENTS.md rule 5: no silent wrong-destination writes).
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
                                   type_is_auto=type_is_auto)
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
    dest_dir: Optional[Path] = None,
    type_is_auto: bool = False,
) -> bool:
    """Download a single file from a HuggingFace repo (despite the name, not
    restricted to .gguf - any single-file ``owner/repo:filename`` spec dispatches
    here, see ``pull_model``'s docstring).

    ``expected_sha256`` is the user-supplied ``--sha256`` digest. It is NOT a
    facade here (FAC-5): when given it is reconciled with HuggingFace's own LFS
    metadata up front, and the downloaded first part is verified against it
    before the model is registered.

    ``dest_dir``, when given, routes the download to that directory instead of
    ``MODELS_DIR`` (e.g. a ComfyUI models subfolder) and is created via
    ``_mkdir_or_explain`` instead of ``ensure_dirs()``. ``register`` still
    controls whether the download is added to localm's own model registry -
    a file routed elsewhere (e.g. for ComfyUI, not for localm's own chat-model
    catalog) should normally pass ``register=False``.
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

    # Split GGUF: normalise to the full ordered part list. llama.cpp loads
    # the model from the first part, so that's what gets registered. A
    # non-split, non-gguf file (e.g. a .safetensors) is just a one-element list.
    all_parts = split_gguf_parts(filename) or [filename]
    filename  = all_parts[0]

    # Traversal guard (GAP-CLI-2): the filename comes from an untrusted spec
    # (owner/repo:../../evil.gguf), so confine every part to base_dir before
    # it is used as a destination. Reject the whole pull on any unsafe part.
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

    missing = [p for p in all_parts if not (base_dir / p).exists()]
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
            reg_type = model_type
            if type_is_auto and reg_type == "llm" and _mm.gguf_embedding_signal(dest):
                console.print("[dim]Detected as an embedding model (GGUF metadata).[/dim]")
                reg_type = "embedding"
            _mm._register_with_dedup(model_name, dest, f"hf:{repo_id}",
                                 digest=verify_digest, model_type=reg_type)
        return True

    # Pre-download duplicate check: same bytes already on disk elsewhere?
    # Only meaningful when the destination IS localm's own models dir, because
    # find_by_sha256 answers "is it in localm's REGISTRY", not "is it at the
    # destination". With an explicit dest_dir (a ComfyUI models folder) the file
    # is wanted THERE: skipping the download because a copy is registered
    # elsewhere reported success while nothing reached the ComfyUI folder, and the
    # next generate still failed with the model missing - success claimed for work
    # not done (REG-641, AGENTS.md rule 5). Aliasing is equally wrong here: a
    # dest_dir pull is register=False by design, so it must not touch the registry.
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
            cdn_url = hf_hub_url(repo_id, part)
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

    with _download_progress([base_dir / p for p in missing], total_size):
        for part in missing:
            try:
                local = hf_hub_download(
                    repo_id=repo_id,
                    filename=part,
                    local_dir=str(base_dir),
                )
                final = base_dir / part
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
                p = base_dir / part
                if p.exists():
                    p.unlink()
            return False
        console.print(f"[green]✓[/green] SHA256 verified: {actual[:16]}…")

    if register:
        reg_type = model_type
        if type_is_auto and reg_type == "llm" and _mm.gguf_embedding_signal(base_dir / filename):
            console.print("[dim]Detected as an embedding model (GGUF metadata).[/dim]")
            reg_type = "embedding"
        _mm._register(model_name, base_dir / filename, f"hf:{repo_id}",
                  sha256=verify_digest, model_type=reg_type)
        console.print(f"[green]✓[/green] [bold]{model_name}[/bold] is ready")
    else:
        console.print(f"[green]✓[/green] [bold]{filename}[/bold] downloaded")
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
    lands on disk like any other file. Historically the HF backend then loaded a
    model directory with transformers' remote-code flag hard-coded on, so that
    Python was imported and executed on the next load (CodeQL alert 49).

    Two reasons this is a warning and not an allow_patterns allowlist:

    1. Execution is already off. ``hf_trust_remote_code`` defaults to False and a
       model that needs custom code is refused with an explanation instead of
       being run (see inference/backends/hf.py), so the file on disk is inert.
    2. An allowlist is the riskier change. A model needs more than weights plus a
       tokenizer - chat templates (.jinja), merges.txt, shard index files,
       per-component subdirectories for multimodal repos - and a pattern list that
       misses one silently produces a broken, half-downloaded model. Refusing to
       download a file we might need, to protect against code we already refuse to
       run, trades a real breakage for no extra safety.

    So: fetch everything, and make the presence of code VISIBLE at the moment it
    arrives, rather than leaving the user to discover it later or not at all.
    """
    try:
        py = sorted(p for p in dest.rglob("*.py") if p.is_file())
    except OSError as e:
        # A failed scan must not fail an otherwise-good download, but it must not
        # read as "no code found" either (AGENTS.md rule 5): say it was not checked.
        console.print(f"[yellow]Could not check {repo_id} for bundled code: {e}[/yellow]")
        return
    if not py:
        return
    # escape(): these names come from a REMOTE repo and are interpolated into a
    # Rich markup string. Unescaped, a file named '[/b]evil.py' raises MarkupError
    # (which would abort the pull between the download and the _register call,
    # leaving the model on disk and unregistered), and one named '[red]x.py' is
    # parsed as a style tag and VANISHES - a security notice that reports "ships 1
    # Python file(s) ()" and names nothing. A repo must not be able to edit, blank,
    # or weaponise the warning that is about it.
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
    resolve, classify the real config.json with the same deterministic reader
    ``add_local`` uses. Registering a plainly-LlamaForCausalLM repo as 'unknown'
    hid it from GUI auto-select, the MCP EngineCache and the jobs runner even
    though it downloaded fine (REG-477).

    A probe that DID resolve (lora/vae/embedding/...) is authoritative and is
    never overridden here, and an unresolvable config.json stays 'unknown' - this
    fills in a gap, it does not restore the old silent 'llm' fallback.
    """
    if model_type != "unknown":
        return model_type
    detected = _detect_local_model_type(dest, is_gguf=False, is_hf=True)
    logger.info("Snapshot %s typed %r from its downloaded config.json (the HF "
                "pipeline_tag probe could not resolve it)", dest.name, detected)
    if detected != "unknown":
        console.print(f"[green]Determined model type from config.json:[/green] "
                      f"[bold]{detected}[/bold]")
    else:
        # Still no hard signal even from the real files: say so (AGENTS.md rule
        # 5). It stays runnable by name, it just is not auto-loaded for chat.
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

    Module-level rather than a closure so the confinement below is directly
    testable (tests/test_registry_confinement.py)."""
    # Imported inside the function, not at module scope, so the CLI download
    # path does not pull fastapi (which pathsafe imports for confined_name's
    # HTTPException) just to validate a filename. By the time this runs we are
    # mid-pull, where a web-stack import is free next to the download itself.
    from localm import pathsafe
    if not (dest / "config.json").exists():
        return False
    if repo_siblings is None:
        return True
    for sib in repo_siblings:
        # rfilename comes from the remote model_info response, so it is not a
        # trusted path component: pathlib lets an absolute or drive-qualified
        # value REPLACE dest entirely (any Windows path naming a drive does), and
        # a '..' walks out of it. Unconfined, this check would stat files anywhere
        # on disk - and a repo listing names that happen to exist off-tree could
        # make an EMPTY download look complete and get registered as ready.
        # Confine before the stat, never after.
        try:
            fp = pathsafe.confined_under(dest, str(sib.rfilename))
        except ValueError as e:
            # Rule 5: an out-of-bounds name is a real signal about the repo, so
            # say so. Reporting the snapshot INCOMPLETE is the safe direction - it
            # re-downloads rather than registering a half-present tree.
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

    def _disk_bytes() -> int:
        return _snapshot_bytes_on_disk(dest)

    # Only the bytes still MISSING need room. snapshot_download resumes on top of
    # whatever already landed in dest, so charging the retry for the FULL repo
    # size made recovery from a disk-full impossible: free < total refused even
    # with ample room for the remainder, defeating the resume this preflight
    # exists to enable (REG-514). _pull_gguf_file sums only the `missing` parts
    # and _pull_url uses (total - already_have); match them.
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
        with _snapshot_progress(_disk_bytes, total_size):
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(dest),
            )
    except Exception as e:
        console.print(f"[red]Download failed:[/red] {e}")
        return False

    _warn_if_repo_ships_code(dest, repo_id)
    _mm._register(model_name, dest, f"hf:{repo_id}",
                  model_type=_resolve_snapshot_type(dest, model_type))
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
                # Route through _entry_path like every other registry consumer
                # (the invariant documented on it). The raw ``["path"]`` read this
                # replaces raised TypeError on a null/int path - the exact crash
                # the choke point exists to prevent - and skipped its ``..``
                # rejection. That matters more here than at a read-only consumer:
                # the value decides an unlink().
                epath = _mm._entry_path(_mm.load_registry().get(dups[0]))
                if epath is None:
                    # Say so rather than silently keeping both: an unreadable
                    # sibling entry is a real registry problem the user should
                    # see, and it must never license deleting a file.
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
    console.print(f"[green]✓[/green] [bold]{name}[/bold] is ready")
    return True


def _hf_pipeline_tag_to_type(repo_id: str) -> str:
    """Classify a HuggingFace repo's model type from HARD metadata. The real
    implementation lives in localm.discover (shared with search-result
    classification there); lazy import matches this module's existing
    convention of not importing localm.discover at module scope."""
    from localm.discover import _hf_pipeline_tag_to_type as _classify
    return _classify(repo_id)

