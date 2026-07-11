# SPDX-License-Identifier: AGPL-3.0-or-later
import sys
from pathlib import Path

import click

from ..config import HOME_DIR, find_binary_dir, load_config, save_config
from ..model_manager import (
    get_model_info, list_models, MODEL_TYPES, pull_model, relocate_model,
    remove_model, set_model_type, show_shortcuts, sync_models_dir,
)
from ._core import console, main, _complete_model_name


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

    info = get_model_info(model)
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
              type=click.Choice(["auto", "llm", "mmproj", "diffusion-unet", "text-encoder", "vae", "lora", "unknown"], case_sensitive=False),
              help="Model type/role.")
@click.option("--store", default=None,
              type=click.Choice(["copy", "move"], case_sensitive=False),
              help="For a local PATH (not a HF/URL spec): bring the file/dir into "
                   "~/.localm/models before registering it, instead of registering "
                   "it in place at its original location.")
def pull(model_spec, name, sha256, redownload, mmproj, type, store):
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
    A local path already on disk (--store brings it into ~/.localm/models):
      localm pull D:\\models\\mymodel.gguf --store copy

    Models are stored in ~/.localm/models/ and registered automatically.
    """
    if not pull_model(model_spec, name, expected_sha256=sha256, redownload=redownload,
                       mmproj_spec=mmproj, model_type=type, store=store):
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
                # Name what the number is: vram_capacity() sums across a configured
                # 2+ GPU split, otherwise it is the single main GPU's ceiling - so
                # do not call a one-GPU number the machine "total" (a 2x16 GB box
                # with no split has a 16 GB main-GPU ceiling, not 32). Gate on the
                # DETECTED split device count (split_device_count), the same signal
                # vram_capacity() used, not the raw config length - a stale index or
                # a GGUF-only box leaves a 2-entry split resolving to one device.
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
              type=click.Choice(["llm", "mmproj", "diffusion-unet", "text-encoder", "vae", "lora", "unknown"], case_sensitive=False),
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

    An externally-referenced model (one whose file lives outside ~/.localm/models)
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
    """Remove a model from the registry (and delete the file if it's in ~/.localm).

    The confirmation prompt describes exactly what will happen (delete vs
    unregister-only). Disable it permanently with:
    localm config confirm_remove false
    """
    from ..config import MODELS_DIR, load_registry
    from ..model_manager import _entry_path, find_aliases_by_path

    cfg = load_config()
    if not yes and cfg.get("confirm_remove", True):
        reg = load_registry()
        if model in reg:
            epath = _entry_path(reg[model])
            if epath is None:
                # A malformed / corrupt entry: no valid file to describe, and
                # reg[model]["path"] would itself raise. Confirm a plain drop -
                # remove_model then clears the corrupt name (CLI recovery path).
                click.confirm(
                    f"Remove corrupt registry entry '{model}'?", abort=True)
            else:
                path = Path(epath)
                others = [a for a in find_aliases_by_path(path, reg) if a != model]
                if others:
                    detail = (f"unregisters the name only - file kept, "
                              f"still registered as: {', '.join(others)}")
                elif str(path).startswith(str(MODELS_DIR)) and path.exists():
                    size = path.stat().st_size / 1024**3 if path.is_file() else None
                    size_s = f" ({size:.1f} GB)" if size else ""
                    detail = f"PERMANENTLY deletes {path}{size_s}"
                else:
                    detail = "unregisters the name only (file is outside ~/.localm/models)"
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
@click.option("--store", default=None,
              type=click.Choice(["copy", "move"], case_sensitive=False),
              help="Bring PATH into ~/.localm/models before registering it, "
                   "instead of registering it in place at its original location. "
                   "'copy' leaves the original untouched; 'move' relocates it.")
def add(path, name, no_hash, fast, on_duplicate, store):
    """Register a local model file or HuggingFace directory.

    Duplicate detection is two-tier: the resolved path is checked first,
    then the file's SHA256 against digests stored in the registry. For large
    files that hash runs off the main thread with a live progress bar; pass
    --fast to skip it and dedup on size alone.

    \b
    Examples:
      localm add C:\\models\\mymodel.gguf
      localm add D:\\models\\gemma.gguf --name gemma4-12b
      localm add D:\\models\\gemma.gguf -n g2 --on-duplicate alias
      localm add D:\\models\\bulk-dir --fast
      localm add D:\\models\\mymodel.gguf --store copy   # copy into ~/.localm/models
      localm add D:\\models\\mymodel.gguf --store move   # move into ~/.localm/models
    """
    # Resolve add_local from the package at call time so tests that monkeypatch
    # localm.cli.add_local affect this call site.
    from localm import cli as _cli
    add_local = _cli.add_local
    if not add_local(path, name, on_duplicate=on_duplicate,
                     no_hash=no_hash, fast=fast, store=store):
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
        addr = f"{scheme}://{r.get('host', '127.0.0.1')}:{r.get('port', '?')}"
        table.add_row(
            str(r.get("instance_id", ""))[:8], status, str(r.get("mode", "?")),
            addr, str(r.get("pid", "?")), str(r.get("root_dir", "")))
    console.print(table)




@main.command("status")
@click.option("--project", default=None, type=click.Path(file_okay=False),
              help="Check this directory instead of the current one.")
def status_cmd(project):
    """Show the localm server serving this directory, if any."""
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
    console.print(f"  [bold]address  [/bold]  "
                  f"{scheme}://{entry.get('host', '127.0.0.1')}:{entry.get('port')}")
    console.print(f"  [bold]surface  [/bold]  {entry.get('mode')}")
    console.print(f"  [bold]pid      [/bold]  {entry.get('pid')}")
    console.print(f"  [bold]version  [/bold]  {entry.get('version')}")




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
    cfg = load_config()
    cfg.update(validated)
    save_config(cfg)
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
    from .. import discover

    cfg = load_config()
    configured = cfg.get("main_gpu_index")
    split = cfg.get("gpu_split_indices") or []
    # A one-shot CLI is NOT on the server event loop, so it can wait out a slow
    # COLD GPU driver init (the first torch.cuda / HIP call, measured ~6.5s)
    # instead of the 4s server cap misreporting it as "no GPU". The status tells a
    # genuine empty result apart from a timeout so the message never blames "no
    # torch" for a probe that simply had not finished (AGENTS.md rule 5).
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
        free_s = f"{free / 1024**3:.1f} GB free / " if isinstance(free, int) else ""
        console.print(
            f"  [cyan]{g['index']}[/cyan]  {g.get('name') or '?':<30} "
            f"{free_s}{g['total'] / 1024**3:.1f} GB total{marker}")
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

    url = os.environ.get("LOCALM_URL", "").rstrip("/")
    if not url:
        entry = instances.find_attachable(HOME_DIR, instances.resolve_root_dir())
        if entry is None:
            console.print("[red]No running localm server found for this directory.[/red]")
            console.print("[dim]Start one with[/dim] localm gui  [dim]or[/dim]  "
                          "localm serve <model>")
            console.print("[dim]Or set LOCALM_URL to target a different instance.[/dim]")
            sys.exit(1)
        scheme = entry.get("scheme", "http")
        url = f"{scheme}://{entry.get('host', '127.0.0.1')}:{entry.get('port')}"

    headers = {}
    key = os.environ.get("LOCALM_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        resp = requests.post(f"{url}/v1/models/unload", headers=headers,
                             params={"model": model} if model else None,
                             timeout=180, verify=tls.requests_verify(url))
    except requests.RequestException as e:
        console.print(f"[red]Could not reach {url}:[/red] {e}")
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
        # A targeted unload on a registered-but-not-loaded model is a no-op
        # success (idempotent, matches the unload-everything "nothing to do"
        # case) - say so plainly rather than claiming something was unloaded.
        console.print(f"[dim]'{data.get('model', model)}' was not loaded - nothing to do.[/dim]")
        return
    unloaded = data.get("unloaded_models")
    if unloaded is None:
        unloaded = [data["model"]] if data.get("model") not in (None, "none") else []
    if not unloaded:
        console.print("[dim]Nothing was loaded.[/dim]")
    else:
        console.print(f"[green]✓[/green] unloaded: {', '.join(unloaded)}")
