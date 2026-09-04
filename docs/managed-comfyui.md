# localm-managed ComfyUI (optional)

localm's image, music, and video plugins drive a local **ComfyUI** server. By
default that is *your* ComfyUI: you install it, point localm at it with
`comfy_workdir` (see [flux-setup.md](flux-setup.md) and [video.md](video.md)),
and localm never touches it beyond driving it over HTTP.

localm can also run its **own** ComfyUI instead. This is opt-in, off by default,
and lives entirely under the localm data folder. When you turn it on, localm owns
that stack: it pins a known-good ComfyUI version, carries a small set of its own
fixes, and shares your existing models without copying them. Your own ComfyUI is
never modified either way.

## Why this exists

ComfyUI is a fast-moving upstream, and a core regression can break generation
through no fault of localm. One real example is the ACE-Step audio crash
(`'function' object has no attribute '__func__'`, Comfy-Org/ComfyUI #12116): a
change to `make_locked_method_func` in ComfyUI core assumed every node's function
is a bound method, which a native audio node is not. When localm depends on your
ComfyUI, an upstream break like that is out of localm's hands.

A localm-managed ComfyUI fixes that trade-off. Because localm owns the checkout,
it can pin a version it has tested and apply its own patches on top, so the stack
stays working until localm deliberately advances it. localm still never edits your
own ComfyUI: the managed instance is a separate install it fully controls.

## Set it up

```bash
localm comfy setup                 # provision localm's own ComfyUI - media routes to
                                    # it right away (comfy_target defaults to "own")
```

`localm comfy setup` picks one of two paths automatically:

- **Copy your stack.** If you already have a working ComfyUI (`comfy_workdir` set,
  with a venv under it), localm replicates it: it reads your ComfyUI repo commit
  and a freeze of your ComfyUI venv, clones that same commit into the data folder,
  makes a **fresh** localm venv, and installs the same package versions. It does
  not byte-copy your venv (venvs are not portable). You are asked before any of
  your custom nodes are copied, since a clean ComfyUI is a legitimate choice; pass
  `--copy-custom-nodes` or `--no-custom-nodes` to decide up front.

  This only works when every package in your venv is resolvable from PyPI or the
  derived PyTorch wheel index. A package that came from a private index or a local
  wheel on your machine (e.g. some AMD driver-bundled Python packages on Windows)
  cannot be replicated; setup fails and names the package. Clear your ComfyUI
  folder setting (Settings -> Media, or `comfy_workdir` in config) and run `localm
  comfy setup` again to install the fresh, hardware-matched path below instead.
- **Fresh, hardware-matched install.** If you have no ComfyUI to copy, localm
  installs one: it clones a pinned ComfyUI version, makes a localm venv, installs
  the PyTorch build for your GPU (the same hardware detection that picks your
  llama.cpp backend selects the torch variant: CUDA for NVIDIA, ROCm for AMD, XPU
  for Intel, CPU otherwise), and adds only the custom nodes its shipped workflows
  need (today just city96's GGUF loader, for GGUF image models).

Either path can download several GB and take a while; localm tells you which path
it is running before it starts. A fresh install always starts with clean custom
nodes.

You can also do this from the GUI: **Settings -> Media** has a "localm's own
ComfyUI" panel with a **Set up localm's own ComfyUI** button (it shows **Remove**
once one is installed). The GUI setup always starts with clean custom nodes; use
`localm comfy setup --copy-custom-nodes` on the CLI if you want to bring yours
over.

## Turn it on (and off)

One setting decides which ComfyUI localm targets:

| Setting | Default | Effect |
|---|---|---|
| `comfy_target` | `own` | `own` = use the managed instance once one is installed; `user` = always use your own ComfyUI. |

localm targets the managed instance **only** when `comfy_target` is `own` AND a
managed instance is actually installed - so `comfy_target` defaults to `own` and
is inert until you run `localm comfy setup`; nothing changes for you until then.
To go back to your own ComfyUI without removing the managed one, set
`comfy_target` to `user`. Either way, your own ComfyUI's settings (`comfy_workdir`
and friends) and the managed instance are independent - switching between them
never loses either one's configuration. Also in the GUI under **Settings ->
Media**, and takes effect on the next server start.

The managed instance runs on its own loopback port (`http://127.0.0.1:8189`), one
above ComfyUI's default `8188`, so it and your own ComfyUI can run at the same
time.

## Models are shared, never copied

localm does not duplicate your models. On setup it writes an
`extra_model_paths.yaml` into the managed ComfyUI that points at both a
localm-managed models folder (`<data dir>/comfyui-models`) and, when localm knows
your ComfyUI folder, your existing `<comfy_workdir>/models`. This is ComfyUI's
native model-sharing mechanism: no copy, no symlink, no re-download, no admin
rights. The generated file has a header showing how to add more model sources by
hand.

A CivitAI pull (`localm pull civitai:<versionId>`, or the Models page's
CivitAI search) lands directly in one of these same folders -
`<data dir>/comfyui-models/<type>` when the managed instance is active,
`<comfy_workdir>/models/<type>` otherwise - so it is picked up the same way,
with no extra copy or scan step. See
[cli.md](cli.md#search-and-pull-from-civitai).

### Registering ComfyUI's models in localm

To make ComfyUI's own models show up in localm's model browser, the GUI Models
page's "Add a model" card has a **Re-scan ComfyUI folder** action. It walks the
`models/` folders under your `comfy_workdir` and registers what it finds, mapping
each subfolder to a localm model type (`unet` to `diffusion-unet`, `vae` to `vae`,
and so on), leaving anything it cannot map as `unknown`. If a scan finds nothing it
tells you why (no `comfy_workdir` configured, or no `models` folder under it)
instead of a bare "Added 0", so a misconfiguration is visible rather than silent.
This is separate from `extra_model_paths.yaml`: sharing lets the managed ComfyUI
*use* your models, scanning lets *localm* list them.

The scan runs as a background job (same shape as a model pull) and reports live
progress - "registering model N of M" with the file name - instead of a static
"Scanning..." message, so a large models folder does not look stuck. The dry-run
preview step (shown before you confirm the scan) has no per-item total, so it is
unchanged.

The neighboring **Import from ComfyUI…** action covers a different case: previewing
and importing from a ComfyUI folder other than your configured `comfy_workdir` -
a one-off install, or localm's own managed ComfyUI - without changing that setting.

## Check status and remove

```bash
localm comfy status                # is one installed, where, and which is targeted now
localm comfy remove                # delete the managed ComfyUI (asks first)
localm comfy remove --models       # also delete the managed models folder
localm comfy remove -y             # skip the confirmation
```

`localm comfy status` reports the `comfy_target` setting, whether an instance is
installed and where, and which ComfyUI media calls target right now. By default it
also pings the targeted ComfyUI (whether it is actually running, and, when a localm
server is up, whether localm itself launched it); pass `--no-ping` to keep the
command instant and offline. `localm comfy remove` deletes only
`<data dir>/comfyui`; your own ComfyUI is never a target. Managed models are kept
by default (they are expensive to re-download) unless you pass `--models`. The
whole feature is self-contained under the data folder, so removing it leaves no
trace elsewhere.

`localm doctor` shows a one-line discovery hint when no managed instance exists
(`localm can manage its own ComfyUI (isolated, patched, pinned): run 'localm comfy
setup'`) and confirms the install path once one is set up. The hint never installs
anything.

## Controlling the ComfyUI process from the CLI

```bash
localm comfy start                 # start ComfyUI (no generation, just bring it up)
localm comfy stop                  # abort the in-flight render, clear the queue, free VRAM
localm comfy restart               # stop then start again
```

These act on whichever ComfyUI `localm comfy status` reports as the current target
(managed or your own) and need a running localm server (`localm gui` or `localm
serve`) to reach it. `start` uses the same launch path a real generation would
(`comfy_launch_cmd`/`comfy_workdir`, or the managed instance's own), so a cold
ROCm/ZLUDA start that compiles GPU kernels gets the same generous timeout a real
generation waits on. If ComfyUI is already running, `start` says so and changes
nothing.

`stop` and `restart` behave differently depending on who launched ComfyUI: if
localm started it, its process is ended too; if you started it yourself, localm
only aborts the render and frees VRAM, leaving the process running (`restart` then
finds it already up). Either way localm never kills a process it did not start.

You do not need any of this to just generate: `localm image`/`music`/`video`
already auto-launch ComfyUI when `comfy_launch_cmd` is set (or always, for the
managed instance), and pressing Ctrl-C during a CLI generation now tells ComfyUI to
abort the render and free its VRAM instead of leaving it running.

## Managing uploaded workflows from the CLI

`localm comfy workflow` manages each media plugin's uploaded ComfyUI workflows
(image, music, video) - the same store the GUI's Workflow card writes to. It works
fully offline; no running server needed.

```bash
localm comfy workflow list image                    # uploaded workflows + which is active
localm comfy workflow add image my_flux.json --use   # upload and select it
localm comfy workflow use image my_flux.json         # select an uploaded workflow
localm comfy workflow use image --clear              # fall back to the built-in default
localm comfy workflow rm image my_flux.json          # delete one (refuses on the active one)
```

`add` validates the file is ComfyUI's API-format JSON (Save > API format in
ComfyUI) before storing it, so a bad upload fails immediately instead of at the
next generation. Whatever is selected here governs `localm image`/`music`/`video`
and the GUI's own picker on the matching page - they all resolve the same active
workflow.

## Keeping it current

localm ships a pinned, known-good ComfyUI version and advances it only
deliberately, never automatically. To move to the version localm currently ships
and re-apply localm's patches:

```bash
localm comfy update                          # advance to the shipped pin, re-apply patches
localm comfy update --reinstall-requirements # also refresh ComfyUI's Python deps
localm comfy update --commit <sha>           # advanced: test a specific commit first
```

Update is safe: on any failure it rolls the managed ComfyUI back to its previous
version with git. It only touches the managed instance. By default it stays within
that git rollback and does not reinstall ComfyUI's Python requirements (a partial
dependency upgrade cannot be rolled back exactly); pass `--reinstall-requirements`
when a new pin changed them.

The same update is available from **Settings -> Media**: the managed-ComfyUI panel
shows the installed version next to the version localm ships, an **Update**
button, an "Also reinstall ComfyUI's dependencies" checkbox for
`--reinstall-requirements`, and an advanced "update to a specific commit" field for
`--commit` - the same knobs as the CLI, streamed to a log as the job runs. If a
setup attempt was interrupted (a crash, a closed browser tab mid-install) the panel
shows a **Repair** action instead, which clears the incomplete install and re-runs
setup in one step.

## The patches localm carries

localm carries a small, versioned set of its own fixes on top of the pinned
ComfyUI, applied after provisioning and re-applied by `localm comfy update`.
Because this is localm's own checkout, a patch is a direct edit to a ComfyUI core
file. Today there is exactly one: the `make_locked_method_func` `__func__`
tolerance that fixes the ACE-Step audio crash above. Every patch is guarded (it
only rewrites the exact known-fragile code and leaves an already-fixed file
alone), idempotent, and fail-safe (a rewrite that would not parse is never
written, and the write is atomic), so it can never corrupt the checkout or fight
an upstream fix.

## Relationship to the reactive shim

There is a lighter-weight, install-free option for the same ACE-Step crash on a
ComfyUI localm does *not* own: the `comfy_func_shim` setting (off by default).
When on, and only for a ComfyUI that **localm itself starts**, localm adds an
in-memory compatibility patch via a `PYTHONPATH` entry. It writes nothing into
your ComfyUI install, never patches a ComfyUI it did not start, and self-expires
once ComfyUI ships its own fix. It exists for users who have not opted into a
managed ComfyUI; once you run a managed instance, its patch set covers the same
crash persistently and you do not need the shim.

## Limits worth knowing

- The fresh-install path fetches a GPU-matched PyTorch build. It rides the same
  hardware detection as the llama.cpp runtime; the non-AMD torch variants are
  less exercised than the AMD path the project develops on. If a fresh install
  picks the wrong build for an unusual setup, `localm comfy remove` and a retry
  (or your own ComfyUI via `comfy_target user`) are always available.
- Provisioning downloads several GB and, on a fresh install, compiles nothing but
  can still take many minutes.
- The managed instance shares your VRAM handoff with the rest of media generation
  unchanged: the chat model is unloaded before a generation and reloaded after.
