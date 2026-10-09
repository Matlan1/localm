# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ``localm setup-llama`` click command.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from localm.debuglog import logger
from localm.setup_llama._common import console
from localm.setup_llama.pins import _ROCM_BUILD, _ROCM_TAG
from localm.setup_llama.download import ArtifactError
from localm.setup_llama.runtime_dir import (_exit_provisioning_busy, _exit_runtime_in_use,
                                            _record_provisioned_backend,
                                            ProvisioningBusyError, RuntimeInUseError)
from localm.setup_llama.versions import (_apply_version_request, _pin_note_for_backend,
                                         _record_runtime_history)
from localm.setup_llama.cuda import _CUDA_LINE
from localm.setup_llama.provision import _provision_with_fallback
import localm.setup_llama as _sl

# The complete set of values --backend accepts, and the ONE place that decides
# it. Public and module-level rather than inline in the click.Choice below,
# because a second surface now offers the same choice: the GUI's runtime route
# validates a caller-supplied backend against this, so a name the CLI accepts
# and the route rejects (or the reverse) is not representable. "auto" is a real
# member, not a sentinel - it is what a bare `setup-llama` resolves through
# _auto_backend, and it is the right default for a first provision.
BACKENDS: tuple[str, ...] = ("auto", "vulkan", "cuda", "sycl", "hip", "cpu",
                               "metal", "amd-rocm")


@click.command("setup-llama", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--from", "from_dir", default=None, type=click.Path(exists=True, file_okay=False),
              help="Copy binaries from a local llama.cpp build directory instead of downloading.")
@click.option("--backend", default="auto",
              type=click.Choice(list(BACKENDS), case_sensitive=False),
              help="Which prebuilt to fetch. 'auto' detects your GPU and picks "
                   "the best-performing backend it can run out of the box: cuda "
                   "for NVIDIA on both Windows and Linux (self-contained, falls "
                   "back to vulkan if the driver is too old); the self-contained "
                   "ROCm build for AMD RX 6000 on Windows; hip for AMD elsewhere "
                   "when a system ROCm/HIP toolkit is detected; sycl for Intel on "
                   "Windows (self-contained); vulkan for Intel on Linux and for "
                   "AMD with no toolkit detected; cpu if no GPU.")
@click.option("--url", default=None, help="Override with an explicit prebuilt archive URL.")
@click.option("--sha256", "sha256", default=None,
              help="Expected sha256 of the downloaded archive. When given, the "
                   "download is refused unless its digest matches (opt-in "
                   "integrity pin).")
@click.option("--force", is_flag=True, help="Re-provision even if binaries are already present.")
@click.option("--tag", "tag", default=None, metavar="TAG",
              help="Install a specific llama.cpp release (e.g. 'b10355') and PIN "
                   "it, so later setup-llama runs and 'localm update' keep that "
                   "exact build. Two words are special: 'latest' opts in to "
                   "upstream's newest release, which localm has not tested, and "
                   "'default' returns to the build localm ships and confirmed.")
@click.option("--rollback", is_flag=True,
              help="Go back to the previous llama.cpp build recorded for this "
                   "backend and pin it. For when an upstream release turns out to "
                   "be broken on your hardware. See 'localm doctor' for what is "
                   "installed now.")
@click.option("--cuda-line", "cuda_line", default=None,
              type=click.Choice(["cuda-12", "cuda-13"], case_sensitive=False),
              help="With --backend cuda on Linux: fetch the CUDA build and runtime "
                   "libraries of this line without an NVIDIA GPU present, for building "
                   "container images. cuda-12 covers every architecture before "
                   "Blackwell, cuda-13 is for Blackwell. The driver check and the load "
                   "test are skipped, the runtime is recorded as not load-tested, and "
                   "the container's start check tests it on the GPU host.")
@click.option("--yes", "-y", "assume_yes", is_flag=True,
              help="Non-interactive: accept the recommended action at every prompt "
                   "(e.g. fetch the self-contained CUDA runtime). Used by the "
                   "one-click installer and for scripted setups.")
def main(from_dir: Optional[str], backend: str, url: Optional[str],
         sha256: Optional[str], force: bool, tag: Optional[str],
         rollback: bool, assume_yes: bool, cuda_line: Optional[str] = None) -> None:
    """Download or copy the native llama.cpp binaries into localm's own venv.

    The chosen backend is load-tested after provisioning. If it cannot load on
    this machine, your pick is NOT changed silently: for a vendor backend
    (cuda/hip/sycl/amd-rocm, e.g. CUDA without a new-enough driver) setup
    explains why and (interactively) offers the universal Vulkan build
    instead, or - in a non-interactive install - falls back with a loud
    warning and tells you how to retry your backend once the cause is fixed.
    vulkan and cpu are themselves the universal builds, so a failure there
    instead names the missing piece and offers a retry, then reports the
    cause and stops rather than silently degrading.

    By default the llama.cpp build this localm release confirmed is installed.
    --tag <tag> pins another exact release, --tag latest tracks upstream's
    newest (untested here), --tag default returns to the confirmed build, and
    --rollback returns to the previous one. The choice sticks across
    'localm update'.

    \b
      localm setup-llama                        # auto-detect GPU, fetch the right prebuilt
      localm setup-llama --backend vulkan       # universal GPU build (any vendor)
      localm setup-llama --backend cuda         # NVIDIA: checks the driver, fetches a
                                                #   self-contained CUDA runtime (no Toolkit)
      localm setup-llama --backend cpu          # no GPU
      localm setup-llama --backend cuda --cuda-line cuda-12   # image build, no GPU
      localm setup-llama --from /path/to/llama.cpp/build/bin
      localm setup-llama --url https://.../llama-...zip
      localm setup-llama --sha256 <hex>         # pin the expected archive digest
      localm setup-llama --tag b10355           # install exactly b10355 and keep it
      localm setup-llama --tag latest           # track upstream's newest (untested here)
      localm setup-llama --tag default          # back to the build localm confirmed
      localm setup-llama --rollback             # back to the previous build
    """
    lib_name = _sl._lib_name()
    target = _sl._repo_runtime_lib()
    if cuda_line:
        cuda_line = cuda_line.lower()
        _require_staging_context(backend, cuda_line, from_dir, url, rollback)
    _apply_version_request(tag, rollback, backend, from_dir, url)
    # A version request is inherently a re-provision: the guard below compares
    # BACKENDS, and the whole point here is to change the BUILD while the
    # backend stays the same. Without this an explicit --tag/--rollback on an
    # already-provisioned box would print "Already provisioned" and change
    # nothing, having just moved the pin - the worst outcome available, because
    # the config and the disk would then disagree with no sign of it.
    if tag is not None or rollback:
        force = True
    target.mkdir(parents=True, exist_ok=True)

    if _keeps_existing_install(target, lib_name, backend, force, assume_yes):
        return

    # Everything below actually MUTATES target (clear + refill), so it is guarded
    # by the cross-process provisioning lock: two processes provisioning the same
    # target must not interleave. Nothing above this point (the "already
    # provisioned" read and its short-circuit) touches disk, so it runs unlocked.
    try:
        with _sl._provisioning_lock(target):
            if cuda_line:
                _provision_staged_cuda(cuda_line, target, lib_name, sha256)
                _refresh_install_record(target)
                return
            if from_dir:
                _provision_from_dir(from_dir, target, lib_name)
            elif url:
                _provision_from_url(url, target, lib_name, sha256)
            else:
                _provision_release(backend, target, sha256, assume_yes)

            _refresh_install_record(target)
            _sl._verify()
    except ProvisioningBusyError as e:
        _exit_provisioning_busy(e)


def _require_staging_context(backend: str, cuda_line: str, from_dir: Optional[str],
                             url: Optional[str], rollback: bool) -> None:
    """Exit non-zero unless --cuda-line is used with --backend cuda on Linux and
    without --from, --url or --rollback."""
    problem = None
    if backend.lower() != "cuda":
        problem = "--cuda-line needs --backend cuda"
    elif sys.platform != "linux":
        problem = "--cuda-line is only available on Linux"
    elif from_dir or url or rollback:
        problem = "--cuda-line cannot be combined with --from, --url or --rollback"
    if problem:
        console.print(f"[red]{problem}.[/red]")
        sys.exit(2)


def _provision_staged_cuda(cuda_line: str, target: Path, lib_name: str,
                           sha256: Optional[str]) -> None:
    """Fetch the *cuda_line* CUDA build and runtime libraries into *target*
    without testing that they load, and record that they were not tested.

    Exits non-zero when the fetch fails or the archive holds no library; it never
    falls back to another backend."""
    console.print(f"[bold yellow]Staging the {cuda_line} CUDA runtime without a GPU.[/bold yellow] "
                  "The NVIDIA driver check and the load test are skipped.")
    _pin_note_for_backend("cuda")
    used_tag: Optional[str] = None
    try:
        _sl._clear_target_or_refuse(target)
        used_tag = _sl._provision_backend("cuda", target, sha256, True, cuda_line)
    except RuntimeInUseError as e:
        _exit_runtime_in_use(e)
    except click.ClickException as e:
        console.print(f"[red]Staging {cuda_line} failed:[/red] {e.message}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Staging {cuda_line} failed:[/red] {e}")
        sys.exit(1)
    if not (target / lib_name).exists():
        console.print(f"[red]The {cuda_line} archive did not contain {lib_name}.[/red]")
        sys.exit(1)
    _sl._bundle_missing_native_deps(target)
    _sl._install_runtime_wheel(_sl._runtime_pkg_dir())
    _record_provisioned_backend(target, "cuda", build=used_tag)
    _record_runtime_history("cuda", used_tag)
    try:
        _sl.record_staged_cuda(target, cuda_line)
    except OSError as e:
        console.print(f"[red]Could not record the staged runtime in {target}:[/red] {e}")
        sys.exit(1)
    console.print(f"[yellow]Staged the {cuda_line} CUDA runtime in {target}. It has NOT "
                  "been load-tested: a container started from this image tests it on "
                  "the GPU host before serving.[/yellow]")


def _keeps_existing_install(target: Path, lib_name: str, backend: str, force: bool,
                            assume_yes: bool) -> bool:
    """Whether the runtime already in *target* is kept, which ends the command.
    False means provisioning goes ahead; what it replaces has then been
    printed."""
    already = (target / lib_name).exists()
    if already and not force:
        # Backend-aware guard: 'auto' means "give me something that works",
        # and something already does, so do not re-download. An EXPLICIT backend
        # is honoured - short-circuit only when we can confirm THAT backend is the
        # one on disk; otherwise (a different recorded backend, or none recorded)
        # fall through and provision what the user asked for. This is what lets
        # `setup-llama --backend cuda` on a box that already has a vulkan/cpu build
        # actually fetch CUDA, instead of keeping the old runtime silently.
        want = backend.lower()
        have = _sl._provisioned_backend(target)
        if want == "auto" or (have is not None and have == want):
            # Name the BUILD as well as the backend when it is recorded, so
            # "which llama.cpp is on this box" is answerable without inspecting
            # library filenames.
            build = _sl._provisioned_build(target) if have else None
            label = f" ({have} {build})" if build else (f" ({have})" if have else "")
            console.print(f"[green]Already provisioned[/green]{label} at {target}")
            if not assume_yes and sys.stdin and sys.stdin.isatty() and click.confirm("Do you want to re-download/replace them?", default=False):
                force = True
                console.print("[yellow]Replacing existing build...[/yellow]")
            else:
                _sl._ensure_importable()
                return True
        # Four distinct situations. Two of them need their own wording: the
        # same backend is a RE-DOWNLOAD, not a replacement, and "auto" is not a
        # backend but how one gets chosen.
        if not have:
            console.print(f"[yellow]Replacing unrecorded build with {want}.[/yellow]")
        elif want == "auto":
            console.print(f"[yellow]Replacing {have} build with the "
                          f"auto-detected backend.[/yellow]")
        elif have == want:
            # Three genuinely different events, distinguished rather than all
            # printed as "Re-downloading", which reads as a no-op even when it
            # is an upgrade. The marker carries the installed build tag when it
            # is known (see _provisioned_build), so a real tag-to-tag upgrade
            # says so.
            #
            # The INSTALLED build is recorded for every tag-based backend, not
            # only amd-rocm, so naming it here needs no network
            # call - it is read from the marker. What still cannot be named for
            # free is the build we are about to install: only amd-rocm knows
            # that without a lookup (_ROCM_TAG is a constant), so only amd-rocm
            # gets the "X -> Y" arrow. The others say which build is being
            # replaced and stop there, which is honest rather than guessing.
            #
            # A marker written before tag recording existed still reads back
            # None, and that case keeps its original wording.
            have_build = _sl._provisioned_build(target)
            if want == "amd-rocm" and have_build and have_build != _ROCM_BUILD:
                console.print(f"[yellow]Upgrading the {have} build: "
                              f"{have_build} -> {_ROCM_BUILD}.[/yellow]")
            elif have_build:
                console.print(f"[yellow]Re-downloading the {have} build "
                              f"({have_build}).[/yellow]")
            else:
                # NOT named `tag`: that is this command's --tag parameter, and
                # rebinding it here would silently shadow the user's request.
                tag_label = f" ({_ROCM_TAG})" if want == "amd-rocm" else ""
                console.print(
                    f"[yellow]Re-downloading the {have} build{tag_label}.[/yellow]")
        else:
            console.print(f"[yellow]Replacing {have} build with {want}.[/yellow]")
    return False


def _provision_from_dir(from_dir: str, target: Path, lib_name: str) -> None:
    """Install the libraries of a local build directory (``--from``)."""
    src = Path(from_dir)
    console.print(f"Copying binaries from [bold]{src}[/bold] ...")
    try:
        _sl._clear_target_or_refuse(target)
    except RuntimeInUseError as e:
        _exit_runtime_in_use(e)
    n = _sl._copy_binaries(src, target)
    if not (target / lib_name).exists():
        console.print(f"[red]No {lib_name} found in the source directory.[/red] "
                      f"Point --from at the build output containing {lib_name}.")
        sys.exit(1)
    console.print(f"[green]Copied {n} file(s)[/green] into {target}")
    _sl._install_runtime_wheel(_sl._runtime_pkg_dir())
    loaded, detail = _sl._native_loads_ok()
    if loaded:
        console.print("[green]OK - the provided build loads on this machine.[/green]")
        _record_provisioned_backend(target, "custom")
    else:
        # The user pinned this build, so we do NOT fall back - but we must not
        # report success on a library that will not load. Exit non-zero with a
        # clear reason rather than leaving a broken runtime behind.
        console.print(f"[red]Copied, but the library did not load[/red] "
                      f"({detail}) - is it built for this OS/GPU? "
                      "See docs/gpu-setup.md.")
        sys.exit(1)


def _provision_from_url(url: str, target: Path, lib_name: str,
                        sha256: Optional[str]) -> None:
    """Install a prebuilt archive from an explicit URL (``--url``)."""
    if not sha256:
        console.print("[yellow]Warning: Custom URL download is unverified (no --sha256 provided).[/yellow]")
    console.print(f"[dim]Fetching:[/dim] {url}")
    try:
        _sl._clear_target_or_refuse(target)
        _sl._fetch_and_place(url, target, sha256)
    except RuntimeInUseError as e:
        # Ahead of the broad handlers below, which would otherwise report a
        # locked file as "Download failed" - a cause the user would go and
        # investigate instead of the one that is true.
        _exit_runtime_in_use(e)
    except ArtifactError as e:
        console.print(f"[red]Refusing to install:[/red] {e}")
        console.print("Provide a local build with --from instead, or a different "
                      "--url (and --sha256 if you pin one).")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Download failed:[/red] {e}")
        console.print("Provide a local build with --from instead, or a different --url.")
        sys.exit(1)
    if not (target / lib_name).exists():
        console.print(f"[red]The archive did not contain {lib_name}.[/red] "
                      "Try a different --url or use --from.")
        sys.exit(1)
    _sl._install_runtime_wheel(_sl._runtime_pkg_dir())
    loaded, detail = _sl._native_loads_ok()
    if loaded:
        console.print("[green]OK - the fetched build loads on this machine.[/green]")
        _record_provisioned_backend(target, "custom")
    else:
        # A user-pinned --url: do not fall back, but do not claim success on a
        # library that will not load. Exit non-zero with a clear reason.
        console.print(f"[red]Placed, but the library did not load[/red] "
                      f"({detail}). Is this build right for your OS/GPU? "
                      "See docs/gpu-setup.md.")
        sys.exit(1)


def _provision_release(backend: str, target: Path, sha256: Optional[str],
                       assume_yes: bool) -> None:
    """Install *backend* from its release asset, with the CUDA dialogue, the
    load test and its fallbacks, then record the backend and build."""
    chosen = _sl._auto_backend() if backend == "auto" else backend
    # warn-once-then-comply: an explicit off-profile choice is the user's to
    # make, but flag a vendor mismatch a single time so a misclick is visible.
    # Also captures the SAME detection for the CUDA dialogue below, so a
    # "no NVIDIA found" fallback can name the vendor that IS actually
    # present and recommend the real match for it, rather than compute it
    # a second time (or not have it at all).
    det = _sl._warn_off_profile(chosen) if backend != "auto" else None
    # CUDA is the visible peak-NVIDIA option: detect the driver, then offer to
    # fetch a self-contained runtime (no Toolkit) or fall back cleanly.
    with_cudart = False
    cuda_line = _CUDA_LINE
    # NOT platform-gated to win32: nvidia_preflight() and _cuda_setup_dialogue()
    # are both fully platform-neutral (nvidia-smi runs on Linux too, and the
    # dialogue's text/branches reference no OS). Restricting this to win32 was
    # an accident of when Linux CUDA support was added (_provision_backend's
    # Linux cudart branch, _resolve_backend_asset's Linux cuda_line-aware
    # matcher, and _fetch_cuda_runtime_libs already handle cuda_line correctly
    # for Linux and are unit-tested for it - see test_linux_cuda_runtime_
    # provisioning.py) without this call site being revisited, so a real
    # Blackwell (or any cuda-13-line) GPU on Linux silently got the cuda-12
    # line - a build with no kernels for it - and no PyPI cudart runtime
    # fetch, producing a runtime that LOADS (passes the ABI check)
    # but registers zero usable GPU devices ('GPU: none in the
    # loaded runtime (cuda)').
    # Only darwin is excluded - CUDA is not a real path on Apple Silicon.
    if chosen == "cuda" and sys.platform != "darwin":
        # Preflight ONCE and reuse it for both the dialogue and the asset
        # line - a second nvidia-smi call could (rarely) see different
        # hardware and pick a line the dialogue never actually displayed.
        info = _sl.nvidia_preflight()
        cuda_line = info.cuda_line
        chosen, with_cudart = _sl._cuda_setup_dialogue(info, assume_yes, det)
    _pin_note_for_backend(chosen)
    result, used_tag = _provision_with_fallback(chosen, target, sha256,
                                                with_cudart, assume_yes,
                                                cuda_line)
    # Record the build tag for EVERY backend now, not only amd-rocm. The old
    # restriction rested on "the upstream backends resolve theirs through
    # _latest_tag(), a NETWORK CALL, and recording a version is not worth
    # making one". The premise was that the tag was not in hand; it was -
    # the fetch had already resolved it and simply discarded it.
    # _provision_with_fallback now returns the tag of the attempt that
    # SUCCEEDED, so this costs no additional lookup and, on a fallback,
    # records the build actually installed rather than the one that failed.
    #
    # amd-rocm still supplies its build from the constants, because its build
    # is not resolved from an upstream tag at all (used_tag is None for it):
    # _ROCM_BUILD with the SIMD CPU backend installed over it, else _ROCM_TAG.
    build = _sl.rocm_build(target) if result == "amd-rocm" else used_tag
    _record_provisioned_backend(target, result, build=build)
    _record_runtime_history(result, build)


def _refresh_install_record(target: Path) -> None:
    """Re-snapshot the provisioned runtime files into the clone's install
    record (``.localm-install.json``), when the clone has one. A failure is
    logged at debug level; uninstall lists unrecorded runtime files anyway."""
    try:
        from localm import install_manifest
        install_manifest.refresh_lib(target)
    except Exception as e:
        logger.debug("could not update the install record for %s: %s", target, e)


def _ensure_importable() -> None:
    try:
        import localm_llama_runtime  # noqa: F401
    except Exception:
        if _sl._install_runtime_wheel(_sl._runtime_pkg_dir()):
            console.print("[green]OK[/green] localm-llama-runtime installed.")
        else:
            # Surface, do not swallow: the runtime is neither importable nor
            # installable, so a later `localm run` will fail. Say so now.
            console.print("[yellow]Warning:[/yellow] localm-llama-runtime is not "
                          "importable and could not be installed. Re-run "
                          "[bold]localm setup-llama[/bold] or check the network/log; "
                          "[bold]localm doctor[/bold] will show what is missing.")


def _verify() -> None:
    try:
        from localm.inference.backends.llamacpp._loader import runtime_binary_dir
        d = runtime_binary_dir()
        if d:
            console.print(f"[bold green]Native runtime ready[/bold green] -> {d}")
            console.print("Try it:  [bold]localm run <model>[/bold]")
        else:
            console.print("[yellow]Binaries placed but not yet resolvable - "
                          "restart your shell so the new package is importable.[/yellow]")
    except Exception as e:
        # Surface a verify failure instead of exiting silently after "setup done":
        # a swallowed error here is exactly the "looks fine, actually broken" trap.
        console.print(f"[yellow]Warning:[/yellow] could not verify the native runtime "
                      f"({e}); it may not load. Run [bold]localm doctor[/bold] to check.")
