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

    # allow_direct_path: operator-typed on the command line (`localm bench <path>`).
    info = get_model_info(model, allow_direct_path=True)
    if info is None:
        console.print(f"[red]Model not found:[/red] {model}")
        sys.exit(1)
    model_path, _hint = info

    try:
        sizes = [int(s) for s in prompts.split(",") if s.strip()]
    except ValueError:
        console.print(f"[red]Invalid --prompts:[/red] {prompts}")
        sys.exit(1)

    from ..inference.engine import Engine
    engine = Engine(str(model_path), n_ctx=ctx, n_gpu_layers=gpu_layers,
                    display_name=model)
    console.print(f"Loading [cyan]{model}[/cyan]…")
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
    table = Table(title=f"benchmark - {model}")
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
    """Download a model from HuggingFace or a URL.

    \b
    Full HF model (transformers format, for multimodal / HF-native models):
      localm pull google/gemma-3-4b-it

    \b
    Single GGUF file (quantized, lighter, works with GGUF backend):
      localm pull llama3.2-3b
      localm pull bartowski/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct-Q4_K_M.gguf

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
@click.option("-n", "--limit", default=10, show_default=True, help="Max repos.")
@click.option("--files", "list_files", is_flag=True,
              help="Treat QUERY as one repo id and list its GGUF files "
                   "with per-quant VRAM fit.")
def search_cmd(query, limit, list_files):
    """Search HuggingFace for GGUF models (empty query = most downloaded).

    \b
    Examples:
      localm search qwen2.5 7b instruct
      localm search bartowski/Qwen2.5-7B-Instruct-GGUF --files
    """
    from rich.console import Console
    from ..discover import (DiscoverError, fit_label, hf_gguf_files,
                           hf_search, vram_capacity)
    console = Console()
    text = " ".join(query).strip()
    try:
        if list_files:
            if not text:
                console.print("[red]--files needs a repo id, e.g. "
                              "owner/repo[/red]")
                sys.exit(1)
            total = vram_capacity().get("total")
            if total:
                # Name what the number is: vram_capacity() sums across a
                # configured 2+ GPU split, otherwise it is the single main GPU's
                # ceiling. The wording is gated on the DETECTED split device count
                # (split_device_count, the signal vram_capacity() itself used), not
                # the raw config length: a stale index or a GGUF-only box leaves a
                # 2-entry split resolving to one device.
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
                console.print(
                    f"  [cyan]{f['quant'] or '?':10}[/cyan] "
                    f"{f['size_bytes'] / 1024**3:6.1f} GB{parts}  {badge}  "
                    f"[dim]{f['file']}[/dim]")
            console.print(f"\n[dim]pull one:  localm pull {text}:<file>[/dim]")
        else:
            results = hf_search(text, limit=limit)
            if not results:
                console.print("[dim](no GGUF repos found)[/dim]")
                return
            for r in results:
                console.print(f"[cyan]{r['id']}[/cyan]  "
                              f"[dim]⬇ {r['downloads']:,}  ♥ {r['likes']:,}[/dim]")
            console.print("\n[dim]list quants:  localm search <repo> "
                          "--files[/dim]")
    except DiscoverError as e:
        console.print(f"[red]{e}[/red]")
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
        console.print(f"[yellow]{result.note}[/yellow]")
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
    """
    from ..config import load_registry
    from ..model_manager import (
        _entry_path, find_aliases_by_path, is_owned_model_path)

    cfg = load_config()
    if not yes and cfg.get("confirm_remove", True):
        reg = load_registry()
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
                # Same predicate as remove_model's delete gate, through the one
                # shared helper, so this prompt never promises a deletion the gate
                # will not perform.
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
    # Resolve add_local from the package at call time, so a rebound
    # localm.cli.add_local is used here.
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
    (already reported to the user, and renaming locally would fail the same
    way), or None when the caller should go ahead and rename locally.
    """
    import os

    import requests

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
        # The connection was never established, so the server did not act; fall
        # back to the local rename. ConnectTimeout subclasses this, and belongs
        # here: no connection means no request.
        console.print(f"[dim]No reachable server at {show_url(url)} ({e}) - "
                      f"renaming locally.[/dim]")
        return None
    except requests.RequestException as e:
        # The request WAS sent and no reply arrived, so whether the rename was
        # applied is unknown. Read the registry, which is the ground truth, and
        # report from that rather than guessing.
        from ..config import load_registry
        from ..model_manager import _sanitize_name
        safe = _sanitize_name(new_name)
        reg = load_registry()
        if safe in reg and old_name not in reg:
            console.print(f"[green]✓[/green] Renamed [bold]{old_name}[/bold] -> "
                          f"[bold]{safe}[/bold]")
            console.print(f"[dim](no reply from the server: {e} - but the rename "
                          f"itself completed)[/dim]")
            return True
        console.print(f"[yellow]No reply from {show_url(url)} ({e}); the rename does not "
                      f"appear to have been applied - doing it locally.[/yellow]")
        return None

    if resp.ok:
        data = resp.json()
        console.print(f"[green]✓[/green] Renamed [bold]{old_name}[/bold] -> "
                      f"[bold]{data.get('new_name', new_name)}[/bold]")
        for note in data.get("notes") or []:
            console.print(f"[dim]{note}[/dim]")
        return True

    detail = ""
    try:
        detail = resp.json().get("detail", "")
    except Exception:
        pass
    # Match the two verdicts the rename route itself raises, not the status code
    # alone: a server with no such route answers 404 too, with FastAPI's bare
    # "Not Found". Anything unmatched falls through to the local rename.
    _verdict = detail.lower()
    if resp.status_code in (404, 409) and (
            "not registered" in _verdict or "already taken" in _verdict):
        # The server's own verdict on the rename itself. Report it and stop; a
        # local rename would fail identically.
        console.print(f"[red]{detail}[/red]")
        return False
    # Anything else (401 without a key, a server with no such route, a 5xx)
    # leaves the rename undone: do it locally and state what the running server
    # still believes.
    console.print(f"[yellow]The running server declined the rename "
                  f"({resp.status_code}{': ' + detail if detail else ''}).[/yellow]")
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
    cfg = load_config()
    binary_dir = find_binary_dir()

    console.print(f"  [bold]models dir[/bold]   {HOME_DIR / 'models'}")
    console.print(f"  [bold]registry   [/bold]   {HOME_DIR / 'registry.json'}")
    console.print(f"  [bold]config     [/bold]   {HOME_DIR / 'config.json'}")
    console.print(f"  [bold]binaries   [/bold]   {binary_dir or '[dim]not found[/dim]'}")
    console.print()
    for k, v in sorted(cfg.items()):
        if v is None:
            v = "[dim](auto)[/dim]"
        console.print(f"  {k:<22} {v}")




@main.command("ps")
def ps_cmd():
    """List running localm servers (per-directory instances).

    Each `localm gui`/`localm serve` advertises itself so other launches attach
    instead of double-loading the model. This shows them: which directory each
    serves, its address and surface (api = bare OpenAI API, full = API + GUI),
    and whether it answers a liveness probe."""
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
        # literal is legible. Not self_connect_host: this column answers what the
        # instance bound.
        addr = show_url(f"{scheme}://"
                        f"{url_host(r.get('host') or '127.0.0.1')}"
                        f":{r.get('port', '?')}")
        table.add_row(
            str(r.get("instance_id", ""))[:8], status, str(r.get("mode", "?")),
            addr, str(r.get("pid", "?")), str(r.get("root_dir", "")))
    console.print(table)
    console.print("[dim]Stop one with[/dim] localm stop <id>  [dim]or[/dim]  "
                  "localm stop --all")




def _fmt_age(seconds) -> str:
    """A coarse human age ("45s", "12m", "3h07m") for how long something has
    been going. Returns "" for None or a negative *seconds*."""
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
    state, payload = read_activity(scheme, port, instance_token, bind_host)
    console.print()
    if state == "unreachable":
        console.print("[yellow]![/yellow]  Could not ask this server what it is "
                      f"doing ({payload}). It may have just stopped.")
        return
    if state == "unauthorized":
        # Lead with the same "could not ask" framing as the unreachable branch
        # above. A busy and an idle server print this identical message, so it
        # must never imply nothing is running: it means this client cannot tell.
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
        # An operation that has not reported progress is at an UNKNOWN
        # percentage: print nothing rather than 0%.
        pct_s = f"  {pct:.0f}%" if isinstance(pct, (int, float)) else ""
        # Age against the SERVER's clock, never this machine's; the two can
        # disagree.
        age = ""
        if isinstance(now, (int, float)) and isinstance(op.get("created_at"), (int, float)):
            age = _fmt_age(now - op["created_at"])
        age_s = f"  [dim]{age}[/dim]" if age else ""
        status = op.get("status") or "?"
        colour = {"running": "cyan", "done": "green",
                  "failed": "red", "cancelled": "yellow"}.get(status, "white")
        # Job ids are 12 hex chars, printed whole so they can be typed back into
        # `localm cancel`.
        op_id = op.get("id")
        id_s = f"  [dim]{op_id}[/dim]" if op_id else ""
        if op.get("cancellable") and op_id:
            cancellable = True
        console.print(f"  [{colour}]{status:<9}[/{colour}] {label}{pct_s}{age_s}{id_s}")
    if cancellable:
        console.print("[dim]Cancel one with[/dim] localm cancel <id>")


@main.command("status")
@click.option("--project", default=None, type=click.Path(file_okay=False),
              help="Check this directory instead of the current one.")
def status_cmd(project):
    """Show the localm server serving this directory, and what it is doing."""
    from localm import instances
    from localm.config import home_dir
    root = instances.resolve_root_dir(override=project)
    entry = instances.find_attachable(home_dir(), root)
    if entry is None:
        console.print(f"[dim]No localm server is serving[/dim] {root}")
        console.print("[dim]Start one with[/dim] localm gui  [dim]or[/dim]"
                      "  localm serve <model>")
        return
    scheme = entry.get("scheme", "http")
    console.print(f"  [bold]directory[/bold]  {root}")
    console.print("  [bold]address  [/bold]  "
                  + show_url(f"{scheme}://"
                             f"{url_host(entry.get('host') or '127.0.0.1')}"
                             f":{entry.get('port')}"))
    console.print(f"  [bold]surface  [/bold]  {entry.get('mode')}")
    console.print(f"  [bold]pid      [/bold]  {entry.get('pid')}")
    console.print(f"  [bold]version  [/bold]  {entry.get('version')}")
    # Everything above is read from the on-disk instance registry and is fixed at
    # process start; this asks the server what it is doing now. Printed last, so
    # the identity block still renders when the server cannot be reached.
    _print_activity(scheme, entry.get("port"), entry.get("token"),
                    entry.get("host"))
    console.print()
    console.print("[dim]Stop it with[/dim] localm stop")




def _match_operation(ops, wanted: str):
    """The one operation whose id is *wanted* or starts with it.

    Returns ``(op, error)``; exactly one is None. *error* is ``"none"`` when
    nothing matches, and ``"ambiguous"`` when a prefix matches more than one
    operation - never resolved by picking the first match.
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
    server = running_server()
    if server is None:
        no_server_message("cancelling an operation")
        sys.exit(1)
    url, headers = server

    # Resolve the id against what the server says it is doing BEFORE posting:
    # that is what makes a prefix work, and what tells "already finished" apart
    # from "cancelling", since POSTing to a finished job returns the same
    # {"status": "cancelling"} as a real cancellation.
    state, payload = server_call(url, headers, "GET", "/api/activity", timeout=10)
    if state != "ok":
        report_server_failure(state, payload, "ask the server what it is doing")
        sys.exit(1)
    ops = (payload or {}).get("operations") or []
    op, err = _match_operation(ops, operation_id)
    if err == "ambiguous":
        console.print(f"[red]{operation_id!r} matches more than one operation "
                      "- be more specific:[/red]")
        for o in ops:
            if str(o.get("id") or "").startswith(operation_id):
                console.print(f"  {o.get('id')}  {o.get('label') or o.get('kind')}")
        sys.exit(1)
    if err == "none":
        console.print(f"[red]No operation matches[/red] {operation_id!r}")
        console.print("[dim]See[/dim] localm status [dim]for what this server is "
                      "doing. Finished operations are forgotten after a while.[/dim]")
        sys.exit(1)
    if not op.get("cancellable"):
        console.print(f"[dim]{op.get('id')} is not running (status: "
                      f"{op.get('status') or '?'}) - nothing to cancel.[/dim]")
        return

    # not_found="missing": a 404 HERE means the job is gone (finished and
    # evicted between the read above and this POST), not that the server has
    # no cancel route.
    state, payload = server_call(url, headers, "POST",
                                 f"/api/jobs/{op['id']}/cancel", timeout=30,
                                 not_found="missing")
    if state != "ok":
        report_server_failure(state, payload, "cancel that operation")
        sys.exit(1)
    label = op.get("label") or op.get("kind") or "operation"
    # "cancelling", not "cancelled": the server sets a cooperative flag and
    # terminates any subprocess, and an in-process job stops at its next
    # checkpoint, which this command does not wait for.
    console.print(f"[green]Cancelling[/green] {label} [dim]({op['id']})[/dim]")
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
    """
    from localm.settings_schema import validate_update
    try:
        validated = validate_update({key: value})
    except ValueError as e:
        raise click.ClickException(str(e))
    # update_config() is the atomic read-modify-write helper; a bare
    # load_config()/save_config() pair has an unlocked window in which a
    # concurrent config write is lost.
    update_config(lambda cfg: cfg.update(validated))
    console.print(f"[green]✓[/green] {key} = {validated[key]}")


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
    from .. import discover, gpu_usage

    cfg = load_config()
    configured = cfg.get("main_gpu_index")
    split = cfg.get("gpu_split_indices") or []
    # The deadline is passed explicitly, pinning a cold-init-tolerant value
    # against any change to the probe's default: a one-shot CLI has to wait out a
    # cold GPU driver init (the first torch.cuda / HIP call) instead of reporting
    # "no GPU". The returned status separates a genuine empty result from a
    # timeout, so the message below never blames "no torch" for a probe that had
    # not finished.
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
        # A free reading counts as trustworthy only when it is both fresh and
        # device-global, the same two-part test sysstats._vram_reading_trusted()
        # applies:
        #   * fresh - status == GPU_PROBE_OK. A non-empty `gpus` can otherwise be
        #     a served last-known-good value from an earlier probe when this call
        #     timed out, was busy or was inconclusive.
        #   * device-global - not PROCESS-scoped, and not an untagged reading on a
        #     platform blind to other processes' VRAM (Windows + AMD ROCm/HIP). A
        #     process-scoped free counts only this process's own allocations and
        #     overstates what is available.
        untrusted = isinstance(free, int) and (
            status != discover.GPU_PROBE_OK
            or g.get("free_scope") == discover.FREE_SCOPE_PROCESS
            or (g.get("free_scope") != discover.FREE_SCOPE_DEVICE
                and gpu_usage.raw_reading_is_process_scoped()))
        # An untrusted free reading is omitted, not printed with a caveat: the
        # row shows the total only, as sysstats._vram() does for the GUI/MCP
        # surface.
        if isinstance(free, int) and not untrusted:
            free_s = f"{free / 1024**3:.1f} GB free / "
            note = ""
        else:
            free_s = ""
            note = ("  [yellow](free VRAM reading unavailable on this "
                    "platform)[/yellow]" if untrusted else "")
        console.print(
            f"  [cyan]{g['index']}[/cyan]  {g.get('name') or '?':<30} "
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

    # The owner key (env, else persisted auth.key) wins; otherwise the discovered
    # instance's own attach token, which _origin_guard's open-mode management gate
    # accepts in place of a key. A token exists only when the server was found via
    # discovery: an explicit LOCALM_URL override has no registry entry to read one
    # from.
    headers = resolve_bearer_headers(entry.get("token") if entry is not None else None)

    try:
        resp = requests.post(f"{url}/v1/models/unload", headers=headers,
                             params={"model": model} if model else None,
                             timeout=180, verify=tls.requests_verify(url))
    except requests.RequestException as e:
        console.print(f"[red]Could not reach {show_url(url)}:[/red] {e}")
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
        console.print(f"[red]Unload failed ({resp.status_code}):[/red] {detail or resp.text}")
        sys.exit(1)

    data = resp.json()
    if data.get("status") == "already_unloaded":
        # A targeted unload on a registered-but-not-loaded model is an idempotent
        # no-op success, reported as such rather than as an unload.
        console.print(f"[dim]'{data.get('model', model)}' was not loaded - nothing to do.[/dim]")
        return
    unloaded = data.get("unloaded_models")
    if unloaded is None:
        unloaded = [data["model"]] if data.get("model") not in (None, "none") else []
    else:
        # unload-all's unloaded_models lists chat engines only. The shared
        # embedding model has its own lifecycle and is reported separately via
        # embedder_unloaded, so it is named here too.
        unloaded = list(unloaded)
        if data.get("embedder_unloaded"):
            unloaded.append("embedding model")
    if not unloaded:
        console.print("[dim]Nothing was loaded.[/dim]")
    else:
        console.print(f"[green]✓[/green] unloaded: {', '.join(unloaded)}")


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
            console.print(f"[red]No running instance matches[/red] {instance_id!r}")
            console.print("[dim]See[/dim] localm ps [dim]for the running instances.[/dim]")
            sys.exit(1)
        if len(targets) > 1:
            console.print(f"[red]{instance_id!r} matches {len(targets)} instances "
                          f"- be more specific:[/red]")
            for e in targets:
                console.print(f"  {str(e.get('instance_id', ''))[:8]}  "
                              f"{e.get('root_dir', '')}")
            sys.exit(1)
    else:
        root = instances.resolve_root_dir()
        entry = instances.find_attachable(home, root)
        if entry is None:
            console.print(f"[dim]No localm server is serving[/dim] {root}")
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

        # Resolved per target: --all or an id prefix can span several instances,
        # each with its OWN attach token. The owner key, if any, is process-wide
        # and resolve_bearer_headers re-reads it identically each time.
        headers = resolve_bearer_headers(entry.get("token"))

        stopped = False
        graceful_denied = False
        try:
            resp = requests.post(f"{url}/v1/server/shutdown", headers=headers,
                                 timeout=5, verify=tls.requests_verify(url))
            if resp.status_code in (401, 403):
                # The open-mode management gate refuses an unauthenticated POST
                # from a local client with no shell or API-key credential, which
                # is the default for a plain `localm run`/`gui`/`serve` with no
                # LOCALM_API_KEY. Fall back to a direct kill.
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
                console.print(f"[dim]{iid}:[/dim] server declined an "
                              f"unauthenticated shutdown request (set "
                              f"LOCALM_API_KEY for a clean model-unload "
                              f"shutdown) - ending the process directly.")
            stopped = instances.kill_pid(int(pid or -1), timeout=timeout)
            if stopped:
                # A direct kill bypasses the server's own clean-shutdown path
                # (_do_shutdown), which normally clears the crash marker, so clear
                # it here or the next `localm` start reports this stop as a crash.
                # Runs only after the process is confirmed gone, never before the
                # kill is attempted: a still-running process must keep its marker.
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
            console.print(f"[green]stopped[/green]  {iid}  {root}")
        else:
            console.print(f"[red]{iid}:[/red] could not confirm it stopped (pid {pid})")
            any_failed = True

    if any_failed:
        sys.exit(1)
