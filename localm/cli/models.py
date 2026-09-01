# SPDX-License-Identifier: AGPL-3.0-or-later
import sys
from pathlib import Path

import click

from ..config import HOME_DIR, find_binary_dir, load_config, update_config
from ..model_manager import (
    get_model_info, list_models, MODEL_TYPES, pull_model, relocate_model,
    remove_model, set_model_type, show_shortcuts, sync_models_dir,
)
from ..console import show_url
from ._core import (console, main, _complete_model_name,
                    no_server_message, report_server_failure,
                    running_server, server_call)
from ..bindhost import self_connect_host, url_host
from ..selfclient import read_activity


# ------------------------------------------------------------------ #
#  benchmark                                                           #
# ------------------------------------------------------------------ #

@main.command()
@click.argument("model", shell_complete=_complete_model_name)
@click.option("-n", "--gen-tokens", default=128, show_default=True,
              help="Tokens to generate per run.")
@click.option("--prompts", default="64,512,2048", show_default=True,
              help="Comma-separated approximate prompt sizes in tokens.")
@click.option("-c", "--ctx",        default=None, type=int)
@click.option("-g", "--gpu-layers", default=None, type=click.IntRange(0, 1000))
def benchmark(model, gen_tokens, prompts, ctx, gpu_layers):
    """Measure TTFT and generation throughput at increasing prompt lengths.

    Runs a fixed prompt padded to each requested size, streams GEN_TOKENS
    tokens, and reports time to first token, tokens per second, and total
    time. Results depend on your hardware, quantisation, and context size.
    """
    import time as _time

    from rich.markup import escape

    # allow_direct_path: operator-typed on the command line (`localm bench <path>`).
    info = get_model_info(model, allow_direct_path=True)
    if info is None:
        console.print(f"[red]Model not found:[/red] {escape(model)}")
        sys.exit(1)
    model_path, _hint = info

    try:
        sizes = [int(s) for s in prompts.split(",") if s.strip()]
    except ValueError:
        console.print(f"[red]Invalid --prompts:[/red] {escape(prompts)}")
        sys.exit(1)

    from ..inference.engine import Engine
    engine = Engine(str(model_path), n_ctx=ctx, n_gpu_layers=gpu_layers,
                    display_name=model)
    console.print(f"Loading [cyan]{escape(model)}[/cyan]…")
    engine.load()

    pad_block = (
        "The quick brown fox jumps over the lazy dog while the river keeps "
        "flowing through the quiet valley under a pale morning sky. "
    )
    question = "\n\nReply with a short story continuing the text above."

    rows = []
    try:
        for target in sizes:
            # Build a prompt of roughly `target` tokens using the real tokenizer
            padding = pad_block
            while engine.count_tokens(padding) < target:
                padding += pad_block
            prompt = padding + question
            prompt_tokens = engine.count_tokens(prompt)

            console.print(f"  prompt ≈ {prompt_tokens} tok … ", end="")
            start = _time.perf_counter()
            first_at = None
            generated = 0
            for token in engine.chat_stream(
                [{"role": "user", "content": prompt}],
                max_tokens=gen_tokens, temperature=0.0,
            ):
                if first_at is None:
                    first_at = _time.perf_counter()
                generated += 1
            elapsed = _time.perf_counter() - start

            ttft_ms = (first_at - start) * 1000 if first_at else float("nan")
            gen_time = elapsed - (first_at - start) if first_at else elapsed
            tps = generated / gen_time if gen_time > 0 and generated else 0.0
            console.print("done")
            rows.append((prompt_tokens, generated, ttft_ms, tps, elapsed))
    finally:
        engine.unload()

    from rich.table import Table
    table = Table(title=f"benchmark - {escape(model)}")
    table.add_column("prompt tok", justify="right")
    table.add_column("gen tok", justify="right")
    table.add_column("TTFT ms", justify="right")
    table.add_column("tok/s", justify="right")
    table.add_column("total s", justify="right")
    for prompt_tokens, generated, ttft_ms, tps, elapsed in rows:
        table.add_row(
            str(prompt_tokens), str(generated),
            f"{ttft_ms:.0f}", f"{tps:.1f}", f"{elapsed:.1f}",
        )
    console.print(table)
    console.print("[dim]TTFT includes prompt prefill; tok/s measures pure "
                  "generation after the first token.[/dim]")




# ------------------------------------------------------------------ #
#  Model management                                                    #
# ------------------------------------------------------------------ #

@main.command()
@click.argument("model_spec")
@click.option("-n", "--name", default=None, help="Alias for the downloaded model.")
@click.option("--sha256", default=None, metavar="HASH",
              help="Expected SHA256 hex digest (direct-URL and HF single-GGUF pulls; "
                   "refused for full-repo snapshots). Download is deleted on mismatch.")
@click.option("--redownload", is_flag=True,
              help="Download even when an identical model is already registered.")
@click.option("--mmproj", default=None, metavar="FILE",
              help="Download an associated mmproj file alongside the main model.")
@click.option("--type", default="auto",
              type=click.Choice(["auto", *sorted(MODEL_TYPES)], case_sensitive=False),
              help="Model type/role.")
@click.option("--store", default=None,
              type=click.Choice(["copy", "move"], case_sensitive=False),
              help="For a local PATH (not a HF/URL spec): bring the file/dir into "
                   "<data dir>/models before registering it, instead of registering "
                   "it in place at its original location.")
@click.option("--comfy-dest-dir", default=None, hidden=True,
              type=click.Path(file_okay=False, path_type=Path),
              help="Internal: route a single-file HF pull to this directory "
                   "instead of <data dir>/models (used by the GUI's ComfyUI "
                   "missing-model download flow).")
@click.option("--register/--no-register", default=True, hidden=True,
              help="Internal: whether to add the download to localm's own "
                   "model registry (off for a file routed to --comfy-dest-dir, "
                   "which belongs to ComfyUI, not localm's chat-model catalog).")
def pull(model_spec, name, sha256, redownload, mmproj, type, store,
         comfy_dest_dir, register):
    """Download a model from HuggingFace, CivitAI, or a URL.

    \b
    Full HF model (transformers format, for multimodal / HF-native models):
      localm pull google/gemma-3-4b-it

    \b
    Single GGUF file (quantized, lighter, works with GGUF backend):
      localm pull llama3.2-3b
      localm pull bartowski/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct-Q4_K_M.gguf

    \b
    CivitAI model VERSION id (checkpoints/LoRAs/VAEs for image generation;
    find one with `localm search --source civitai`). Lands in the active
    ComfyUI's models/<type> folder, not <data dir>/models:
      localm pull civitai:135867
      localm pull civitai:135867:99264

    \b
    Direct URL:
      localm pull https://example.com/model.gguf

    \b
    A local path already on disk (--store brings it into <data dir>/models):
      localm pull D:\\models\\mymodel.gguf --store copy

    Models are stored in <data dir>/models/ and registered automatically.
    """
    if not pull_model(model_spec, name, expected_sha256=sha256, redownload=redownload,
                       mmproj_spec=mmproj, model_type=type, store=store,
                       dest_dir=comfy_dest_dir, register=register):
        sys.exit(1)




@main.command("search")
@click.argument("query", nargs=-1)
@click.option("-n", "--limit", default=10, show_default=True, help="Max repos/models.")
@click.option("--files", "list_files", is_flag=True,
              help="Treat QUERY as one repo id (--source hf) or one model "
                   "VERSION id (--source civitai) and list its downloadable "
                   "files.")
@click.option("--source", default="hf", show_default=True,
              type=click.Choice(["hf", "civitai"], case_sensitive=False),
              help="Which model source to search.")
@click.option("--nsfw", is_flag=True,
              help="(--source civitai only) include NSFW results. Off by "
                   "default; a minor-flagged result is excluded either way.")
@click.option("--legacy-formats", "legacy_formats", is_flag=True,
              help="(--source civitai, with --files) also list legacy "
                   "PickleTensor/unknown-format files, hidden by default.")
def search_cmd(query, limit, list_files, source, nsfw, legacy_formats):
    """Search HuggingFace or CivitAI for models (empty query = most downloaded).

    \b
    Examples:
      localm search qwen2.5 7b instruct
      localm search bartowski/Qwen2.5-7B-Instruct-GGUF --files
      localm search --source civitai detail tweaker
      localm search --source civitai --files 135867
    """
    from rich.console import Console
    from rich.markup import escape

    console = Console()
    text = " ".join(query).strip()
    if source.lower() == "civitai":
        _search_civitai(console, text, limit, list_files, nsfw, legacy_formats)
        return

    from ..discover import (DiscoverError, fit_label, hf_gguf_files,
                           hf_search, vram_capacity)
    try:
        if list_files:
            if not text:
                console.print("[red]--files needs a repo id, e.g. "
                              "owner/repo[/red]")
                sys.exit(1)
            total = vram_capacity().get("total")
            if total:
                # vram_capacity() sums across a configured 2+ GPU split;
                # otherwise it is the single main GPU's ceiling, so a one-GPU
                # number is not labelled a machine total. The label is gated on
                # the DETECTED split device count, not the raw config length: a
                # stale index or a GGUF-only box leaves a 2-entry split
                # resolving to one device.
                from ..discover import split_device_count
                n_split = split_device_count()
                basis = (f"{total / 1024**3:.0f} GB combined across your "
                         f"{n_split}-GPU split"
                         if n_split >= 2
                         else f"your main GPU's {total / 1024**3:.0f} GB")
                console.print(f"[dim]fit vs {basis} "
                              "(weights + ~1.5 GB overhead)[/dim]")
            for f in hf_gguf_files(text):
                fit = fit_label(f["size_bytes"], total)
                badge = {"fits": "[green]fits[/green]",
                         "tight": "[yellow]tight[/yellow]",
                         "too-big": "[red]too big[/red]"}.get(fit, "")
                parts = f" ({f['n_parts']} parts)" if f["n_parts"] > 1 else ""
                # quant/file are parsed from a remote repo's own filenames
                # (hf_gguf_files), so they are escaped.
                console.print(
                    f"  [cyan]{escape(f['quant'] or '?'):10}[/cyan] "
                    f"{f['size_bytes'] / 1024**3:6.1f} GB{parts}  {badge}  "
                    f"[dim]{escape(f['file'])}[/dim]")
            console.print(f"\n[dim]pull one:  localm pull {escape(text)}:<file>[/dim]")
        else:
            results = hf_search(text, limit=limit)
            if not results:
                console.print("[dim](no GGUF repos found)[/dim]")
                return
            for r in results:
                # r['id']: a remote HF repo id, not restricted to a safe charset.
                console.print(f"[cyan]{escape(r['id'])}[/cyan]  "
                              f"[dim]⬇ {r['downloads']:,}  ♥ {r['likes']:,}[/dim]")
            console.print("\n[dim]list quants:  localm search <repo> "
                          "--files[/dim]")
    except DiscoverError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        sys.exit(1)


def _search_civitai(console, text, limit, list_files, nsfw, legacy_formats):
    """The --source civitai half of search_cmd, split out to keep that
    command's HF-shaped body unchanged."""
    from rich.markup import escape

    from ..model_manager.sources import (
        ModelSourceError, civitai_list_files, civitai_search)
    try:
        if list_files:
            if not text:
                console.print("[red]--files needs a CivitAI model VERSION id, "
                              "e.g. localm search --source civitai --files "
                              "135867[/red]")
                sys.exit(1)
            files = civitai_list_files(text, include_legacy_formats=legacy_formats)
            if not files:
                console.print("[dim](no downloadable files - retry with "
                              "--legacy-formats if you expect one of the "
                              "riskier formats)[/dim]")
                return
            for f in files:
                fmt = (f.get("metadata") or {}).get("format") or "?"
                size_kb = f.get("sizeKB") or 0
                scan = f.get("virusScanResult") or "?"
                # name/fmt/scan come from CivitAI's own file metadata.
                console.print(
                    f"  [cyan]{escape(str(f.get('id')))}[/cyan]  "
                    f"{size_kb / 1024**2:6.2f} GB  [dim]{escape(fmt)}[/dim]  "
                    f"scan:{escape(scan)}  {escape(f.get('name', '?'))}")
            console.print(f"\n[dim]pull one:  localm pull civitai:{escape(text)}"
                          f"[/dim]")
        else:
            result = civitai_search(text, limit=limit, nsfw=nsfw)
            items = result["items"]
            if not items:
                console.print("[dim](no CivitAI models found)[/dim]")
                return
            for it in items:
                versions = it.get("modelVersions") or []
                latest = versions[0].get("id") if versions else None
                stats = it.get("stats") or {}
                # name/type/creator come from CivitAI's own model metadata.
                console.print(
                    f"[cyan]{escape(str(it.get('name', '?')))}[/cyan] "
                    f"[dim]({escape(str(it.get('type', '?')))})[/dim]  "
                    f"⬇ {stats.get('downloadCount', 0):,}  "
                    f"[dim]version:{escape(str(latest))}[/dim]")
            console.print("\n[dim]list files:  localm search --source civitai "
                          "--files <version>[/dim]")
    except ModelSourceError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        sys.exit(1)




@main.command("list")
@click.option("--type", default=None,
              type=click.Choice(sorted(MODEL_TYPES), case_sensitive=False),
              help="Filter models by type.")
def list_cmd(type):
    """List registered models (auto-detecting changes in the models folder)."""
    result = sync_models_dir()
    if result.changed:
        bits = []
        if result.added:
            bits.append(f"{result.added} new")
        if result.flagged:
            bits.append(f"{result.flagged} missing")
        if result.restored:
            bits.append(f"{result.restored} restored")
        if result.pruned:
            bits.append(f"{result.pruned} pruned")
        console.print(f"[dim]Models folder synced: {', '.join(bits)}.[/dim]")
    if result.note:
        # note is built from registry model names plus fixed counts and
        # literal text, and is escaped.
        from rich.markup import escape
        console.print(f"[yellow]{escape(result.note)}[/yellow]")
    list_models(type_filter=type)


@main.command("relocate")
@click.argument("model", shell_complete=_complete_model_name)
@click.argument("new_path", type=click.Path())
def relocate_cmd(model, new_path):
    """Re-point a registered MODEL to NEW_PATH after you MOVED its file.

    An externally-referenced model (one whose file lives outside <data dir>/models)
    shows as 'missing' when you move it. Instead of re-adding it under a new name,
    relocate re-points the existing registry entry - keeping its name, source, and
    any aliases. NEW_PATH must be a real .gguf file or a HuggingFace model dir.
    """
    import sys
    if not relocate_model(model, str(new_path)):
        sys.exit(1)


@main.command("set-type")
@click.argument("model", shell_complete=_complete_model_name)
@click.argument("model_type",
                type=click.Choice(sorted(MODEL_TYPES), case_sensitive=False))
def set_type_cmd(model, model_type):
    """Change a registered MODEL's type to MODEL_TYPE.

    Type is not frozen at registration: an ambiguous import (registered as
    'unknown') or a mis-detected model is corrected here. A model set to 'unknown'
    stays runnable by name but is not auto-loaded as the default chat model.

    \b
    Example:
      localm set-type my-model llm
    """
    if not set_model_type(model, model_type.lower()):
        sys.exit(1)




@main.command()
@click.argument("model", shell_complete=_complete_model_name)
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation.")
def rm(model, yes):
    """Remove a model from the registry (and delete the file if it's in <data dir>).

    The confirmation prompt describes exactly what will happen (delete vs
    unregister-only). Disable it permanently with:
    localm config confirm_remove false

    Refuses when a running localm server still has the model's file loaded,
    whether or not --yes was passed - this one-shot process shares no memory
    with a server, so it has nothing of its own to check and asks over HTTP.
    """
    from rich.markup import escape

    from ..config import load_registry
    from ..model_manager import (
        _entry_path, find_aliases_by_path, is_owned_model_path)
    from ..selfclient import remote_hold_reason

    reg = load_registry()
    if model in reg:
        reason = remote_hold_reason(model)
        if reason is not None:
            console.print(
                f"[red]Refusing to remove '{escape(model)}':[/red] "
                f"{escape(reason)}. Removing it could delete the model "
                f"file while it is in use. Unload it there (or stop that "
                f"server), then try again.")
            sys.exit(1)

    cfg = load_config()
    if not yes and cfg.get("confirm_remove", True):
        if model in reg:
            epath = _entry_path(reg[model])
            if epath is None:
                # A malformed / corrupt entry: no valid file to describe, and
                # reg[model]["path"] would itself raise. Confirm a plain drop;
                # remove_model then clears the corrupt name.
                click.confirm(
                    f"Remove corrupt registry entry '{model}'?", abort=True)
            else:
                path = Path(epath)
                others = [a for a in find_aliases_by_path(path, reg) if a != model]
                if others:
                    detail = (f"unregisters the name only - file kept, "
                              f"still registered as: {', '.join(others)}")
                # Same predicate as remove_model's delete gate, via the one
                # shared helper, so this prompt cannot promise a deletion the
                # gate will not perform.
                elif is_owned_model_path(path):
                    if path.exists():
                        size = path.stat().st_size / 1024**3 if path.is_file() else None
                        size_s = f" ({size:.1f} GB)" if size else ""
                        detail = f"PERMANENTLY deletes {path}{size_s}"
                    else:
                        # Owned but already gone: remove_model's `owned and
                        # path.exists()` gate skips the delete, so this is a
                        # name-only drop.
                        detail = "unregisters the name only (the file is already missing)"
                else:
                    detail = "unregisters the name only (file is outside <data dir>/models)"
                click.confirm(f"Remove '{model}'? This {detail}. Continue?", abort=True)
        else:
            click.confirm(f"Remove '{model}'?", abort=True)
    remove_model(model)




@main.command()
@click.argument("path")
@click.option("-n", "--name", default=None, help="Name to register the model as.")
@click.option("--no-hash", is_flag=True,
              help="Skip SHA256 computation (disables content-level duplicate detection).")
@click.option("--fast", is_flag=True,
              help="Skip the full-file SHA256 and dedup on file size only. "
                   "Faster for known-unique bulk imports; size matching is a "
                   "weaker (stat-only) duplicate check than the content hash.")
@click.option("--on-duplicate", default="ask",
              type=click.Choice(["ask", "alias", "copy", "move", "register", "skip"]),
              help="What to do when the model is already registered (default: ask).")
@click.option("--type", default="auto",
              type=click.Choice(["auto", *sorted(MODEL_TYPES)], case_sensitive=False),
              help="Model type/role. Default 'auto' detects it from hard metadata "
                   "(GGUF architecture/pooling signal, HF config.json architectures).")
@click.option("--store", default=None,
              type=click.Choice(["copy", "move"], case_sensitive=False),
              help="Bring PATH into <data dir>/models before registering it, "
                   "instead of registering it in place at its original location. "
                   "'copy' leaves the original untouched; 'move' relocates it.")
def add(path, name, no_hash, fast, on_duplicate, type, store):
    """Register a local model file or HuggingFace directory.

    Duplicate detection is two-tier: the resolved path is checked first,
    then the file's SHA256 against digests stored in the registry. For large
    files that hash runs off the main thread with a live progress bar; pass
    --fast to skip it and dedup on size alone.

    \b
    Examples:
      localm add D:\\models\\mymodel.gguf
      localm add D:\\models\\gemma.gguf --name gemma4-12b
      localm add D:\\models\\gemma.gguf -n g2 --on-duplicate alias
      localm add D:\\models\\bulk-dir --fast
      localm add D:\\models\\mymodel.gguf --store copy   # copy into <data dir>/models
      localm add D:\\models\\mymodel.gguf --store move   # move into <data dir>/models
    """
    # Resolve add_local from the package at call time so tests that monkeypatch
    # localm.cli.add_local affect this call site.
    from localm import cli as _cli
    add_local = _cli.add_local
    model_type = None if type == "auto" else type
    if not add_local(path, name, on_duplicate=on_duplicate,
                     no_hash=no_hash, fast=fast, model_type=model_type, store=store):
        sys.exit(1)




@main.command()
@click.argument("existing", shell_complete=_complete_model_name)
@click.argument("new_name")
def alias(existing, new_name):
    """Register NEW_NAME as another name for EXISTING (same file, no copy).

    Works like 'ollama cp': both names point at the same model file.
    Removing one name keeps the file as long as another name references it.

    \b
    Example:
      localm alias gemma3-12b daily-driver
    """
    from ..model_manager import alias_model

    if not alias_model(existing, new_name):
        sys.exit(1)


def _rename_on_running_server(old_name: str, new_name: str):
    """Ask the localm server serving this directory to perform the rename, so
    the registry move and the live engine's re-key happen in ONE process.

    Returns True on success, False when the server refused the rename ITSELF
    (already reported to the user), or None when the caller should go ahead and
    rename locally.

    A rename done in THIS process would move the registry entry while a running
    server kept its loaded engine keyed under the OLD name, so the server would
    serve a model the registry no longer lists.
    """
    import os

    import requests
    from rich.markup import escape

    from .. import instances, tls
    from ..auth import resolve_bearer_headers

    url = os.environ.get("LOCALM_URL", "").rstrip("/")
    entry = None
    if not url:
        entry = instances.find_attachable(HOME_DIR, instances.resolve_root_dir())
        if entry is None:
            return None                      # nothing running: rename locally
        scheme = entry.get("scheme", "http")
        url = f"{scheme}://{url_host(self_connect_host(entry.get('host')))}:{entry.get('port')}"

    headers = resolve_bearer_headers(entry.get("token") if entry is not None else None)
    try:
        resp = requests.post(f"{url}/v1/models/rename", headers=headers,
                             params={"model": old_name, "new_name": new_name},
                             timeout=60, verify=tls.requests_verify(url))
    except requests.ConnectionError as e:
        # The connection was never established, so the server did not act:
        # report it and fall back to the local rename. ConnectTimeout
        # subclasses this and belongs here - no connection means no request.
        # show_url() already escapes its own argument; only `e` needs it here.
        console.print(f"[dim]No reachable server at {show_url(url)} ({escape(str(e))}) - "
                      f"renaming locally.[/dim]")
        return None
    except requests.RequestException as e:
        # The request WAS sent and the reply did not arrive, so whether the
        # rename was applied is unknown. The registry is the ground truth, so
        # it is re-read rather than guessed at.
        from ..config import load_registry
        from ..model_manager import _sanitize_name
        safe = _sanitize_name(new_name)
        reg = load_registry()
        if safe in reg and old_name not in reg:
            console.print(f"[green]✓[/green] Renamed [bold]{escape(old_name)}[/bold] -> "
                          f"[bold]{escape(safe)}[/bold]")
            console.print(f"[dim](no reply from the server: {escape(str(e))} - but the rename "
                          f"itself completed)[/dim]")
            return True
        console.print(f"[yellow]No reply from {show_url(url)} ({escape(str(e))}); the rename does not "
                      f"appear to have been applied - doing it locally.[/yellow]")
        return None

    if resp.ok:
        data = resp.json()
        console.print(f"[green]✓[/green] Renamed [bold]{escape(old_name)}[/bold] -> "
                      f"[bold]{escape(str(data.get('new_name', new_name)))}[/bold]")
        for note in data.get("notes") or []:
            console.print(f"[dim]{escape(str(note))}[/dim]")
        return True

    detail = ""
    try:
        detail = resp.json().get("detail", "")
    except Exception:
        pass
    # Match the two verdicts the rename route itself raises, NOT the status
    # code alone: a server too old to have this route also answers 404, with
    # FastAPI's bare "Not Found". An unmatched 404 falls through to the local
    # rename.
    _verdict = detail.lower()
    if resp.status_code in (404, 409) and (
            "not registered" in _verdict or "already taken" in _verdict):
        # The server's own verdict on the rename itself: report it and stop.
        console.print(f"[red]{escape(detail)}[/red]")
        return False
    # Anything else (401 without a key, a server too old to have the route, a
    # 5xx) leaves the rename undone, so it is done locally and the message
    # states what the running server still believes.
    console.print(f"[yellow]The running server declined the rename "
                  f"({resp.status_code}{': ' + escape(detail) if detail else ''}).[/yellow]")
    console.print("[yellow]Renaming locally; that server will keep the model "
                  "loaded under its old name until it is restarted or the "
                  "model is unloaded ([bold]localm unload[/bold]).[/yellow]")
    return None


@main.command()
@click.argument("old_name", shell_complete=_complete_model_name)
@click.argument("new_name")
def rename(old_name, new_name):
    """Rename a registered model from OLD_NAME to NEW_NAME.

    Unlike 'localm alias', OLD_NAME stops working: this MOVES the
    registration, and best-effort updates every config/job/RAG reference
    inside <data dir> that named OLD_NAME. A per-project
    .localcoder/config.toml 'model' setting (if any) lives outside <data dir>
    and is not touched - update it by hand in any project that pinned this
    model.

    If a localm server is serving this directory it performs the rename, so a
    model that is loaded right now keeps serving under its new name instead of
    being stranded under the old one. Set LOCALM_URL to target a different
    instance, and LOCALM_API_KEY if it requires one.

    \b
    Example:
      localm rename gemma3-12b daily-driver
    """
    from ..model_manager import rename_model

    done = _rename_on_running_server(old_name, new_name)
    if done is not None:
        sys.exit(0 if done else 1)
    if not rename_model(old_name, new_name):
        sys.exit(1)




@main.command()
def models():
    """Show popular GGUF model shortcuts."""
    show_shortcuts()




# ------------------------------------------------------------------ #
#  Config / info                                                       #
# ------------------------------------------------------------------ #

@main.command()
def info():
    """Show paths and current configuration."""
    from rich.markup import escape

    cfg = load_config()
    binary_dir = find_binary_dir()

    # HOME_DIR is wherever LOCALM_HOME/config points and a config VALUE is
    # whatever the user or a hand-edited config.json set, so both are escaped.
    console.print(f"  [bold]models dir[/bold]   {escape(str(HOME_DIR / 'models'))}")
    console.print(f"  [bold]registry   [/bold]   {escape(str(HOME_DIR / 'registry.json'))}")
    console.print(f"  [bold]config     [/bold]   {escape(str(HOME_DIR / 'config.json'))}")
    binaries_s = escape(str(binary_dir)) if binary_dir else "[dim]not found[/dim]"
    console.print(f"  [bold]binaries   [/bold]   {binaries_s}")
    console.print()
    for k, v in sorted(cfg.items()):
        # `cfg` is DEFAULT_CONFIG overlaid with the raw contents of
        # config.json, so BOTH the key and the value can be arbitrary text from
        # a hand-edited or newer-version file.
        if v is None:
            v_s = "[dim](auto)[/dim]"
        else:
            v_s = escape(str(v))
        console.print(f"  {escape(str(k)):<22} {v_s}")




@main.command("ps")
def ps_cmd():
    """List running localm servers (per-directory instances).

    Each `localm gui`/`localm serve` advertises itself so other launches attach
    instead of double-loading the model. This shows them: which directory each
    serves, its address and surface (api = bare OpenAI API, full = API + GUI),
    and whether it answers a liveness probe."""
    from rich.markup import escape
    from rich.table import Table

    from localm import instances
    from localm.config import home_dir
    rows = instances.snapshot(home_dir())
    if not rows:
        console.print("[dim]No running localm instances.[/dim]")
        return
    table = Table(title="localm instances")
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("Surface")
    table.add_column("Address")
    table.add_column("PID", justify="right")
    table.add_column("Directory")
    for r in rows:
        alive = r.get("alive")
        status = "[green]live[/green]" if alive else "[yellow]no answer[/yellow]"
        scheme = r.get("scheme", "http")
        # The BIND address (0.0.0.0 for a wildcard bind), bracketed so an IPv6
        # literal is legible. NOT self_connect_host: this column answers what
        # the instance bound. show_url() escapes its own argument.
        addr = show_url(f"{scheme}://"
                        f"{url_host(r.get('host') or '127.0.0.1')}"
                        f":{r.get('port', '?')}")
        # Table cell strings go through the same markup parsing as
        # console.print(), and instance_id, mode and root_dir all come from the
        # on-disk instance registry, so all three are escaped.
        table.add_row(
            escape(str(r.get("instance_id", ""))[:8]), status,
            escape(str(r.get("mode", "?"))),
            addr, str(r.get("pid", "?")), escape(str(r.get("root_dir", ""))))
    console.print(table)
    console.print("[dim]Stop one with[/dim] localm stop <id>  [dim]or[/dim]  "
                  "localm stop --all")




def _fmt_age(seconds) -> str:
    """A coarse human age: seconds, minutes, or hours and minutes. Empty for
    None or a negative value."""
    if seconds is None or seconds < 0:
        return ""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _print_activity(scheme: str, port, instance_token=None,
                    bind_host=None) -> None:
    """Render what the server is doing, or why that could not be determined."""
    from rich.markup import escape

    state, payload = read_activity(scheme, port, instance_token, bind_host)
    console.print()
    if state == "unreachable":
        console.print("[yellow]![/yellow]  Could not ask this server what it is "
                      f"doing ({payload}). It may have just stopped.")
        return
    if state == "unauthorized":
        # Same "could not ask" framing as the unreachable branch above. A busy
        # and an idle server print this identical message, so it never implies
        # "nothing is running" - it means this client cannot tell.
        console.print("[yellow]![/yellow]  Could not ask this server what it "
                      "is doing: it requires an API key this client does not "
                      "have.")
        console.print("[dim]Set LOCALM_API_KEY to the server's key, or run "
                      "from the install whose auth.key it uses, to see what "
                      "it is doing.[/dim]")
        return
    if state == "unsupported":
        console.print("[dim]This server does not report activity (it predates "
                      "this feature).[/dim]")
        return
    if state == "http":
        console.print(f"[yellow]![/yellow]  Could not read activity (HTTP {payload}).")
        return

    ops = (payload or {}).get("operations") or []
    if not ops:
        console.print("[dim]Nothing running right now.[/dim]")
        return
    now = (payload or {}).get("now")
    console.print("[bold]activity[/bold]")
    cancellable = False
    for op in ops:
        label = op.get("label") or op.get("kind") or "operation"
        pct = op.get("pct")
        # Absent, not zero: an operation that has not reported progress is at
        # an UNKNOWN percentage, so nothing is printed.
        pct_s = f"  {pct:.0f}%" if isinstance(pct, (int, float)) else ""
        # Age against the SERVER's clock, never this machine's: the two can
        # disagree.
        age = ""
        if isinstance(now, (int, float)) and isinstance(op.get("created_at"), (int, float)):
            age = _fmt_age(now - op["created_at"])
        age_s = f"  [dim]{age}[/dim]" if age else ""
        status = op.get("status") or "?"
        # `colour` comes from a fixed whitelist and forms the tag name, so it
        # is not escaped; `status` is shown as text below and is not guaranteed
        # to be one of the four known values, so it is escaped there.
        colour = {"running": "cyan", "done": "green",
                  "failed": "red", "cancelled": "yellow"}.get(status, "white")
        # Job ids are 12 hex chars, short enough to print whole and type.
        op_id = op.get("id")
        id_s = f"  [dim]{escape(str(op_id))}[/dim]" if op_id else ""
        if op.get("cancellable") and op_id:
            cancellable = True
        status_s = escape(str(status))
        console.print(f"  [{colour}]{status_s:<9}[/{colour}] {escape(str(label))}{pct_s}{age_s}{id_s}")
    if cancellable:
        console.print("[dim]Cancel one with[/dim] localm cancel <id>")


@main.command("status")
@click.option("--project", default=None, type=click.Path(file_okay=False),
              help="Check this directory instead of the current one.")
def status_cmd(project):
    """Show the localm server serving this directory, and what it is doing."""
    from rich.markup import escape

    from localm import instances
    from localm.config import home_dir
    root = instances.resolve_root_dir(override=project)
    entry = instances.find_attachable(home_dir(), root)
    if entry is None:
        console.print(f"[dim]No localm server is serving[/dim] {escape(root)}")
        console.print("[dim]Start one with[/dim] localm gui  [dim]or[/dim]"
                      "  localm serve <model>")
        return
    scheme = entry.get("scheme", "http")
    console.print(f"  [bold]directory[/bold]  {escape(root)}")
    # show_url() already escapes its own argument.
    console.print("  [bold]address  [/bold]  "
                  + show_url(f"{scheme}://"
                             f"{url_host(entry.get('host') or '127.0.0.1')}"
                             f":{entry.get('port')}"))
    console.print(f"  [bold]surface  [/bold]  {escape(str(entry.get('mode')))}")
    console.print(f"  [bold]pid      [/bold]  {entry.get('pid')}")
    console.print(f"  [bold]version  [/bold]  {escape(str(entry.get('version')))}")
    # The one place a terminal can learn what a running server is actually
    # doing; everything above is read from the on-disk instance registry and is
    # fixed at process start. Printed last so the identity block still renders
    # when the server cannot be reached.
    _print_activity(scheme, entry.get("port"), entry.get("token"),
                    entry.get("host"))
    console.print()
    console.print("[dim]Stop it with[/dim] localm stop")




def _match_operation(ops, wanted: str):
    """The one operation whose id is *wanted* or starts with it.

    Returns ``(op, error)``; exactly one is None. Prefix matching mirrors
    ``localm stop``, which takes an instance-id prefix. An AMBIGUOUS prefix is
    an error, never a silent pick of the first match.
    """
    exact = [o for o in ops if str(o.get("id") or "") == wanted]
    if exact:
        return exact[0], None
    hits = [o for o in ops if str(o.get("id") or "").startswith(wanted)]
    if not hits:
        return None, "none"
    if len(hits) > 1:
        return None, "ambiguous"
    return hits[0], None


@main.command("cancel")
@click.argument("operation_id")
def cancel_cmd(operation_id):
    """Cancel one operation a running localm server is doing.

    Pass an id from `localm status` (a unique prefix is enough). This is the
    same cancel the GUI's activity list offers: a model pull's subprocess is
    ended, and an in-process job (a RAG re-embed, a media generation) is asked
    to stop at its next checkpoint.

    Scheduled jobs are a different thing under a different id space - use
    `localm job` for those.
    """
    from rich.markup import escape

    server = running_server()
    if server is None:
        no_server_message("cancelling an operation")
        sys.exit(1)
    url, headers = server

    # Resolve the id against what the server says it is doing, BEFORE posting:
    # that is what lets a prefix work, and what tells "already finished" apart
    # from "cancelling". POSTing blind to a finished job returns the same
    # {"status": "cancelling"} as a real cancellation.
    state, payload = server_call(url, headers, "GET", "/api/activity", timeout=10)
    if state != "ok":
        report_server_failure(state, payload, "ask the server what it is doing")
        sys.exit(1)
    ops = (payload or {}).get("operations") or []
    op, err = _match_operation(ops, operation_id)
    if err == "ambiguous":
        console.print(f"[red]{escape(repr(operation_id))} matches more than one operation "
                      "- be more specific:[/red]")
        for o in ops:
            if str(o.get("id") or "").startswith(operation_id):
                console.print(f"  {escape(str(o.get('id')))}  "
                              f"{escape(str(o.get('label') or o.get('kind')))}")
        sys.exit(1)
    if err == "none":
        console.print(f"[red]No operation matches[/red] {escape(repr(operation_id))}")
        console.print("[dim]See[/dim] localm status [dim]for what this server is "
                      "doing. Finished operations are forgotten after a while.[/dim]")
        sys.exit(1)
    if not op.get("cancellable"):
        console.print(f"[dim]{escape(str(op.get('id')))} is not running (status: "
                      f"{escape(str(op.get('status') or '?'))}) - nothing to cancel.[/dim]")
        return

    # not_found="missing": a 404 HERE means the job is gone (finished and
    # evicted between the read above and this POST), not that the server has no
    # cancel route.
    state, payload = server_call(url, headers, "POST",
                                 f"/api/jobs/{op['id']}/cancel", timeout=30,
                                 not_found="missing")
    if state != "ok":
        report_server_failure(state, payload, "cancel that operation")
        sys.exit(1)
    label = op.get("label") or op.get("kind") or "operation"
    # "cancelling", not "cancelled": the server sets a cooperative flag and
    # terminates any subprocess, and an in-process job stops at its next
    # checkpoint.
    console.print(f"[green]Cancelling[/green] {escape(str(label))} "
                  f"[dim]({escape(str(op['id']))})[/dim]")
    console.print("[dim]Confirm with[/dim] localm status")


@main.command("config")
@click.argument("key")
@click.argument("value")
def config_cmd(key, value):
    """Set a persistent configuration value.

    \b
    Examples:
      localm config n_gpu_layers 99
      localm config n_ctx 8192
      localm config temperature 0.7
      localm config main_gpu_index 1
      localm config hf_token hf_xxx
    """
    from rich.markup import escape

    from localm.model_source_credentials import CREDENTIAL_KEYS, set_credentials
    from localm.settings_schema import validate_update
    # hf_token / civitai_api_key are not config.json keys (see
    # settings_schema.py's comment above their SettingField entries): route them
    # to their own owner-only file instead of validate_update/update_config, and
    # never echo the value back.
    if key in CREDENTIAL_KEYS:
        try:
            set_credentials({key: value})
        except ValueError as e:
            raise click.ClickException(str(e))
        action = "cleared" if not value.strip() else "updated"
        console.print(f"[green]✓[/green] {escape(str(key))} {action}")
        return
    try:
        validated = validate_update({key: value})
    except ValueError as e:
        raise click.ClickException(str(e))
    # update_config() is the atomic read-modify-write helper; a bare
    # load_config()/save_config() pair has an unlocked window where a
    # concurrent config write can be silently lost.
    update_config(lambda cfg: cfg.update(validated))
    # `key` only reaches here after validate_update() proved it is one of
    # DEFAULT_CONFIG's own keys, and is escaped anyway; `value` (a free-text
    # setting like mdns_name) has no such guarantee.
    console.print(f"[green]✓[/green] {escape(str(key))} = {escape(str(validated[key]))}")


@main.command("gpus")
def gpus_cmd():
    """List detected GPUs, the configured main GPU index, and any GPU split.

    On a multi-GPU system, localm loads models onto device 0 unless you set
    main_gpu_index. Pick one from this list, then:

    \b
      localm config main_gpu_index <index>

    A model too large for one card can instead be split across 2+ GPUs:

    \b
      localm config gpu_split_indices 0,1
    """
    from rich.markup import escape

    from .. import discover, gpu_usage

    cfg = load_config()
    # main_gpu_index/gpu_split_indices are schema-validated to an int / list of
    # ints, so `configured`/`split` below are numeric and printed unescaped: an
    # int or a list of ints cannot start a "[...]" span with the letter, #, @ or
    # backslash Rich's tag parser requires.
    configured = cfg.get("main_gpu_index")
    split = cfg.get("gpu_split_indices") or []
    # A one-shot CLI waits out a slow COLD GPU driver init (the first
    # torch.cuda / HIP call) rather than reporting it as "no GPU". The
    # cold-init-tolerant deadline is passed explicitly so it survives a change
    # to the default. The status tells a genuine empty result apart from a
    # timeout, so the message never blames "no torch" for a probe that had not
    # finished.
    gpus, status = discover.list_gpus(
        deadline=discover._GPU_PROBE_CLI_DEADLINE, return_status=True)
    if not gpus:
        if status == discover.GPU_PROBE_OK:
            console.print("[dim]No GPUs detected (or VRAM detection is unavailable - "
                          "no torch, no nvidia-smi).[/dim]")
        else:
            console.print(
                "[yellow]GPU detection timed out[/yellow] before the driver "
                "responded (a cold GPU driver can take several seconds to "
                "initialize on first use). Run [cyan]localm gpus[/cyan] again - "
                "the driver is usually warm on a second try.")
        return
    for g in gpus:
        markers = []
        if g["index"] == configured:
            markers.append("[green](configured)[/green]")
        if g["index"] in split:
            markers.append("[cyan](split)[/cyan]")
        marker = "  " + " ".join(markers) if markers else ""
        free = g.get("free")
        # Trustworthy only when THIS reading is both fresh and device-global,
        # the same two-part definition sysstats._vram_reading_trusted() applies:
        #   * fresh - status == GPU_PROBE_OK. list_gpus() can return a non-empty
        #     `gpus` that is really a served LAST-KNOWN-GOOD value from an
        #     EARLIER successful probe when THIS call timed out, was busy or was
        #     inconclusive.
        #   * device-global - not PROCESS-scoped, and not an untagged reading on
        #     a platform blind to other processes' VRAM (Windows + AMD
        #     ROCm/HIP), where a process-scoped free counts only this process's
        #     own allocations.
        untrusted = isinstance(free, int) and (
            status != discover.GPU_PROBE_OK
            or g.get("free_scope") == discover.FREE_SCOPE_PROCESS
            or (g.get("free_scope") != discover.FREE_SCOPE_DEVICE
                and gpu_usage.raw_reading_is_process_scoped()))
        # An untrusted free figure is not printed at all: the line shows the
        # total only, with a note in place of the figure.
        if isinstance(free, int) and not untrusted:
            free_s = f"{free / 1024**3:.1f} GB free / "
            note = ""
        else:
            free_s = ""
            note = ("  [yellow](free VRAM reading unavailable on this "
                    "platform)[/yellow]" if untrusted else "")
        # g['name'] is a driver/OS-reported device description, so it is
        # escaped.
        name_s = escape(str(g.get("name") or "?"))
        console.print(
            f"  [cyan]{g['index']}[/cyan]  {name_s:<30} "
            f"{free_s}{g['total'] / 1024**3:.1f} GB total{marker}{note}")
    if configured is not None and not any(g["index"] == configured for g in gpus):
        console.print(
            f"[yellow]![/yellow]  configured main_gpu_index={configured} does not "
            "match any GPU detected right now; loads fall back to device 0.")
    if split and not all(any(g["index"] == idx for g in gpus) for idx in split):
        console.print(
            f"[yellow]![/yellow]  configured gpu_split_indices={split} references a "
            "GPU not detected right now; that device is dropped from the split.")
    console.print("\n[dim]Set the main GPU:  localm config main_gpu_index <index>[/dim]")
    console.print("[dim]Split across GPUs:  localm config gpu_split_indices 0,1[/dim]")


@main.command("unload")
@click.argument("model", required=False, shell_complete=_complete_model_name)
def unload_cmd(model):
    """Unload the currently loaded model(s) from a running localm server.

    With no MODEL, unloads everything - frees the GPU for another task, the
    same as the GUI's "Unload all". With MODEL, unloads only that one,
    leaving any other loaded models running.

    Talks to the server already serving this directory (see `localm status`);
    set LOCALM_URL to target a different instance, and LOCALM_API_KEY if it
    requires one.
    """
    import os

    import requests
    from rich.markup import escape

    from .. import instances, tls
    from ..auth import resolve_bearer_headers

    url = os.environ.get("LOCALM_URL", "").rstrip("/")
    entry = None
    if not url:
        entry = instances.find_attachable(HOME_DIR, instances.resolve_root_dir())
        if entry is None:
            console.print("[red]No running localm server found for this directory.[/red]")
            console.print("[dim]Start one with[/dim] localm gui  [dim]or[/dim]  "
                          "localm serve <model>")
            console.print("[dim]Or set LOCALM_URL to target a different instance.[/dim]")
            sys.exit(1)
        scheme = entry.get("scheme", "http")
        url = f"{scheme}://{url_host(self_connect_host(entry.get('host')))}:{entry.get('port')}"

    # Owner key (env, else persisted auth.key) wins; else the discovered
    # instance's own attach token (0600 per-instance registry file), which the
    # open-mode management gate accepts in place of a key. The token is only
    # available when the server was found via discovery; an explicit LOCALM_URL
    # override has no registry entry to read one from.
    headers = resolve_bearer_headers(entry.get("token") if entry is not None else None)

    try:
        resp = requests.post(f"{url}/v1/models/unload", headers=headers,
                             params={"model": model} if model else None,
                             timeout=180, verify=tls.requests_verify(url))
    except requests.RequestException as e:
        # show_url() already escapes its own argument; only `e` needs it here.
        console.print(f"[red]Could not reach {show_url(url)}:[/red] {escape(str(e))}")
        sys.exit(1)

    if resp.status_code == 401:
        console.print("[red]Unauthorized.[/red] Set LOCALM_API_KEY to the server's key.")
        sys.exit(1)
    if not resp.ok:
        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            pass
        console.print(f"[red]Unload failed ({resp.status_code}):[/red] "
                      f"{escape(str(detail or resp.text))}")
        sys.exit(1)

    data = resp.json()
    if data.get("status") == "already_unloaded":
        # A targeted unload on a registered-but-not-loaded model is an
        # idempotent no-op success, reported as such rather than as an unload.
        console.print(f"[dim]'{escape(str(data.get('model', model)))}' was not "
                      "loaded - nothing to do.[/dim]")
        return
    unloaded = data.get("unloaded_models")
    if unloaded is None:
        unloaded = [data["model"]] if data.get("model") not in (None, "none") else []
    else:
        # unload-all's unloaded_models lists only chat engines; the shared
        # embedding model has its own lifecycle and is reported via
        # embedder_unloaded, so it is named here too.
        unloaded = list(unloaded)
        if data.get("embedder_unloaded"):
            unloaded.append("embedding model")
    if not unloaded:
        console.print("[dim]Nothing was loaded.[/dim]")
    else:
        # unloaded holds server-reported model names plus the literal
        # "embedding model", so they are escaped.
        console.print(f"[green]✓[/green] unloaded: "
                      f"{', '.join(escape(str(u)) for u in unloaded)}")


@main.command("stop")
@click.argument("instance_id", required=False)
@click.option("--all", "stop_all", is_flag=True,
              help="Stop every running localm instance.")
@click.option("--timeout", default=10.0, show_default=True, type=float,
              help="Seconds to wait for a clean shutdown before forcing the "
                   "process to end.")
def stop_cmd(instance_id, stop_all, timeout):
    """Stop a running localm server.

    With no argument, stops the instance serving the current directory (see
    `localm status`). Pass an ID or ID prefix (as shown by `localm ps`) to
    stop a specific instance, or --all to stop every running one.

    Requests the same clean shutdown the GUI's Settings page uses (model
    unloaded, then the process exits). Without LOCALM_API_KEY set, a plain
    local instance declines the unauthenticated request (by design), and if
    the server does not confirm within --timeout for any reason, `stop` ends
    the process directly instead.
    """
    import time

    import requests
    from rich.markup import escape

    from .. import instances, tls
    from ..auth import resolve_bearer_headers
    from ..config import home_dir

    if instance_id and stop_all:
        console.print("[red]Pass either an instance id or --all, not both.[/red]")
        sys.exit(1)

    home = home_dir()
    instances.reap_stale(home)
    entries = instances.list_entries(home)

    if stop_all:
        targets = entries
        if not targets:
            console.print("[dim]No running localm instances.[/dim]")
            return
    elif instance_id:
        targets = [e for e in entries
                   if str(e.get("instance_id", "")).startswith(instance_id)]
        if not targets:
            console.print(f"[red]No running instance matches[/red] {escape(repr(instance_id))}")
            console.print("[dim]See[/dim] localm ps [dim]for the running instances.[/dim]")
            sys.exit(1)
        if len(targets) > 1:
            console.print(f"[red]{escape(repr(instance_id))} matches {len(targets)} instances "
                          f"- be more specific:[/red]")
            for e in targets:
                console.print(f"  {escape(str(e.get('instance_id', ''))[:8])}  "
                              f"{escape(str(e.get('root_dir', '')))}")
            sys.exit(1)
    else:
        root = instances.resolve_root_dir()
        entry = instances.find_attachable(home, root)
        if entry is None:
            console.print(f"[dim]No localm server is serving[/dim] {escape(root)}")
            console.print("[dim]Pass an id (see[/dim] localm ps[dim]) or --all.[/dim]")
            sys.exit(1)
        targets = [entry]

    any_failed = False
    for entry in targets:
        iid = str(entry.get("instance_id", ""))[:8]
        root = entry.get("root_dir", "")
        pid = entry.get("pid")
        scheme = entry.get("scheme", "http")
        url = f"{scheme}://{url_host(self_connect_host(entry.get('host')))}:{entry.get('port')}"

        # --all / an id prefix can span several instances, each with its OWN
        # attach token, so the open-mode fallback is resolved inside the loop.
        # The owner key, if any, is process-wide and resolve_bearer_headers
        # re-reads it identically each time.
        headers = resolve_bearer_headers(entry.get("token"))

        stopped = False
        graceful_denied = False
        try:
            resp = requests.post(f"{url}/v1/server/shutdown", headers=headers,
                                 timeout=5, verify=tls.requests_verify(url))
            if resp.status_code in (401, 403):
                # The open-mode management gate refuses an unauthenticated POST
                # from a bare local client with no shell/API-key credential,
                # which is the DEFAULT case for a plain `localm
                # run`/`gui`/`serve` with no LOCALM_API_KEY configured. Fall
                # back to a direct kill rather than hard-failing.
                graceful_denied = True
            elif resp.status_code == 200:
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline and not stopped:
                    if not instances.pid_alive(int(pid or -1)):
                        stopped = True
                    else:
                        time.sleep(0.25)
        except requests.RequestException:
            pass  # server unreachable (hung / already gone) - fall through to a direct kill

        if not stopped:
            if graceful_denied:
                console.print(f"[dim]{escape(iid)}:[/dim] server declined an "
                              f"unauthenticated shutdown request (set "
                              f"LOCALM_API_KEY for a clean model-unload "
                              f"shutdown) - ending the process directly.")
            stopped = instances.kill_pid(int(pid or -1), timeout=timeout)
            if stopped:
                # A direct kill bypasses the server's own clean-shutdown path
                # (_do_shutdown), which normally clears the crash marker, so it
                # is cleared here instead. Only after confirming the process is
                # actually gone: a still-running process must keep its marker.
                try:
                    from localm import bugreport
                    bugreport.disarm_crash_guard(
                        home, instance_id=entry.get("instance_id"))
                except Exception:
                    pass

        path = entry.get("_path")
        if path:
            instances.unregister_instance(path)

        if stopped:
            console.print(f"[green]stopped[/green]  {escape(iid)}  {escape(root)}")
        else:
            console.print(f"[red]{escape(iid)}:[/red] could not confirm it "
                          f"stopped (pid {pid})")
            any_failed = True

    if any_failed:
        sys.exit(1)
