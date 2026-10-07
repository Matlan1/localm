# SPDX-License-Identifier: AGPL-3.0-or-later
"""Provisioning a chosen backend and proving it loads: the ABI floor at the
pinned tag, and the fallback to the universal builds.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from localm.setup_llama._common import console
from localm.setup_llama.pins import _PINNED_FALLBACK_SHA256, _PINNED_TAG
from localm.setup_llama.download import _human_mb, ArtifactError
from localm.setup_llama.runtime_dir import _exit_runtime_in_use, RuntimeInUseError
from localm.setup_llama.versions import _tag_for, tracks_latest
from localm.setup_llama.cuda import _CUDA_LINE
from localm.setup_llama.load_probe import _is_abi_rejection, _name_missing_shared_lib
import localm.setup_llama as _sl

def _provision_backend(chosen: str, target: Path, sha256: Optional[str],
                       with_cudart: bool, cuda_line: str = _CUDA_LINE,
                       tag: Optional[str] = None) -> Optional[str]:
    """Resolve + fetch the prebuilt(s) for *chosen* into *target*. For CUDA with
    *with_cudart* it also fetches the matching cudart runtime bundle so the
    build is self-contained (no CUDA Toolkit needed). *cuda_line* picks which
    upstream CUDA asset line to fetch ('cuda-12' or 'cuda-13' - see
    NvidiaInfo.cuda_line); it is ignored for every other backend. Raises on a
    fatal error.

    RETURNS the release tag this provision used, so the caller can record which
    build is now on disk, or None when there is no upstream tag to report
    (amd-rocm ships from lemonade-sdk's own numbering, which the caller already
    has as _ROCM_TAG).

    Every branch already resolved a tag and threw it away, which is the whole
    reason the marker could only ever record a version for amd-rocm. Each now
    keeps it, so recording the build costs NO extra network call - and the
    lookup stays exactly where it always was rather than being hoisted to the
    top of this function, which would make it run for backends that never
    needed it and would move it out from behind _resolve_backend_asset, the
    seam every caller and test already isolates.

    *tag* pins this ONE provision to a specific upstream release, overriding
    the pin/newest resolution. It exists for the ABI walk-back, which retries
    the SAME backend against an older release; it does NOT touch the stored
    pin, so a walk-back is a recovery rather than a silent change
    to what the user asked for. Ignored for amd-rocm, which has no upstream
    tag."""
    if chosen == "cuda" and with_cudart and sys.platform == "win32":
        # Resolved here, not in _resolve_backend_asset, because this branch
        # needs the tag to PAIR the build with its matching cudart bundle - a
        # cudart from a different release is exactly the mismatch this pairing
        # exists to prevent - and only reaches _resolve_backend_asset in the
        # no-assets fallback below, to which it then hands the same tag.
        tag = tag or _tag_for(chosen)
        build, cudart = _sl._resolve_cuda_pair(tag, cuda_line)
        if build is None:
            # Asset listing unavailable: fall back to the templated build URL and
            # warn that the runtime bundle could not be resolved automatically.
            console.print("[yellow]Could not resolve CUDA assets; fetching build only.[/yellow]\n"
                          "[yellow]If it fails to load, use --backend vulkan or install CUDA Toolkit.[/yellow]")
            url, fallback_sha, _t = _sl._resolve_backend_asset("cuda", cuda_line, tag=tag)
            _sl._fetch_verified(url, target, sha256 or fallback_sha, "CUDA build asset")
            return tag

        # Resolve build sha256
        build_digest = build.get("digest")
        build_sha = build_digest.split("sha256:")[-1].strip() if build_digest and "sha256:" in build_digest else None
        if not build_sha:
            build_sha = _PINNED_FALLBACK_SHA256.get(build["name"])
        
        console.print(f"[dim]CUDA build:[/dim] {build['name']} ({_human_mb(build.get('size'))})")
        _sl._fetch_verified(build["browser_download_url"], target, sha256 or build_sha, "CUDA build asset")
        if cudart is not None:
            if sha256:
                # The pin is a single hash; it can only cover the build. Be honest
                # that the cudart bundle is validated by size + archive shape, not
                # by the pinned digest (upstream publishes no per-asset hash here).
                console.print("[yellow]Note:[/yellow] --sha256 pins the CUDA build only.")
            
            # Resolve cudart sha256
            cudart_digest = cudart.get("digest")
            cudart_sha = cudart_digest.split("sha256:")[-1].strip() if cudart_digest and "sha256:" in cudart_digest else None
            if not cudart_sha:
                cudart_sha = _PINNED_FALLBACK_SHA256.get(cudart["name"])
            
            console.print(f"[dim]CUDA runtime:[/dim] {cudart['name']} "
                          f"({_human_mb(cudart.get('size'))}) - no Toolkit install needed")
            _sl._fetch_and_place(cudart["browser_download_url"], target, cudart_sha)
        else:
            console.print("[yellow]No cudart bundle found; CUDA Toolkit may be required.[/yellow]")
        return tag
    if chosen == "cuda" and with_cudart and sys.platform not in ("win32", "darwin"):
        # sys.platform, not _platform_key(): matches this function's OWN
        # existing style two lines up (the win32 cudart branch), rather than
        # mixing the two equivalent-in-production-but-differently-mockable
        # spellings within one function - caught by a test that mocked
        # _platform_key alone and still hit the win32 branch on a real
        # Windows test box, since sys.platform itself never moved.
        #
        # Self-contained Linux CUDA: the binary comes from a third-party
        # prebuilt, hybridgroup/llama-cpp-builder (_resolve_linux_cuda_asset in
        # assets.py, NOT a localm-built binary), and the
        # runtime libraries (cudart/cublas) from PyPI wheels
        # (_fetch_cuda_runtime_libs) - never from scanning anything already on
        # the user's machine. If _resolve_backend_asset raises (no
        # matching build exists yet for this exact upstream tag on
        # hybridgroup's repo), that propagates to _provision_with_fallback's
        # caller exactly like every other provisioning failure, which
        # offers/forces the vulkan fallback - nothing new to handle here.
        url, fallback_sha, tag = _sl._resolve_backend_asset("cuda", cuda_line, tag=tag)
        _sl._fetch_verified(url, target, sha256 or fallback_sha, "CUDA build asset")
        if sha256:
            console.print("[yellow]Note:[/yellow] --sha256 pins the CUDA build only; "
                          "the PyPI runtime libraries are verified by their own "
                          "published checksums instead.")
        n = _sl._fetch_cuda_runtime_libs(cuda_line, target)
        console.print(f"[dim]CUDA runtime:[/dim] {n} librar{'y' if n == 1 else 'ies'} "
                      "fetched from PyPI - no CUDA Toolkit install needed")
        return tag
    # Every other backend is a single archive resolved from the chosen name.
    # Also reached for chosen == "cuda" with with_cudart False (no current
    # caller produces that combination - see _cuda_setup_dialogue - but
    # forwarding cuda_line here means it never silently reverts to the
    # cuda-12 default if one ever does).
    url, fallback_sha, tag = _sl._resolve_backend_asset(chosen, cuda_line, tag=tag)
    _sl._fetch_verified(url, target, sha256 or fallback_sha, "release asset")
    return tag


def _warn_off_profile(chosen: str):
    """One-line heads-up when a vendor-specific backend was chosen for a vendor
    we did NOT detect. We respect the user's choice - no block, no nag, no
    re-prompt - just flag it once so a misclick is visible.

    Returns the ``hwdetect.Detection`` used for the check (or ``None`` if it
    was never computed, or detection failed), so a caller that needs the SAME
    vendor info downstream - the CUDA dialogue, to name what IS actually
    present instead of a generic "not found" hedge - does not need a second,
    redundant ``hwdetect.detect()`` call, and the two can never see different
    hardware if detection is non-deterministic (e.g. a flaky WMI query)."""
    vendor_specific = {"cuda": "nvidia", "amd-rocm": "amd", "hip": "amd",
                       "sycl": "intel", "metal": "apple"}
    owner = vendor_specific.get(chosen)
    if not owner:
        return None
    try:
        from localm import hwdetect
        det = hwdetect.detect()
    except Exception:
        return None
    vendors = det.vendors or []
    if vendors and owner not in vendors:
        seen = ", ".join(vendors)
        console.print(f"[yellow]Heads up:[/yellow] Picked [bold]{chosen}[/bold] but detected [bold]{seen}[/bold].\n"
                      "[yellow]Proceeding. Hardware must be present.[/yellow]")
    return det


# WAS a bounded WALK over the last few upstream releases, picking whichever one
# happened to load. It is now a FLOOR at _PINNED_TAG, and the difference is the
# whole point rather than a tidy-up.
#
# A walk SELECTS A VERSION WHILE SETUP IS RUNNING, which is exactly what the pin
# exists to stop; and it selects it on the ABI gate alone, so its destination is
# "an older build that LOADS" - a build nobody has ever generated a token with.
# Under a pin it also inverts: landing the user on an older, less-tested release
# is a worse outcome than the one it was rescuing them from, and it looks like a
# success.
#
# A floor has exactly ONE destination and it is a constant: the build we
# confirmed. It cannot go anywhere a human did not decide, and it can only ever
# move the user TOWARDS the tested build, never away from it.
_FLOOR_TAG_DESCRIPTION = "the confirmed build localm ships"


def _floor_at_pinned_tag(chosen: str, with_cudart: bool, rejected_tag: str,
                         try_fn, detail: str) -> tuple:
    """After our own ABI gate refused *rejected_tag*, fall back to _PINNED_TAG -
    the one build we confirmed - and only from an install that had opted OUT of
    it. Returns ``(ok, tag)``.

    LOUD IN EVERY BRANCH, including the ones that do nothing. That is the point:
    three of the four outcomes here install nothing at all, and a recovery (or a
    refusal) the user cannot see is the failure mode this area keeps producing.
    Each branch names the build involved, the ABI reason, and the --tag command
    that changes it.

    THE FOUR CASES, which need genuinely different answers:

      * an exact USER PIN - not moved, ever. Moving off it is the override this
        project forbids, and the user is told which commands change it. The check
        lives here rather than at the call site so a future second caller cannot
        bypass it.
      * TRACKING upstream (--tag latest) - the case the floor exists for. The
        user asked for a build nobody had confirmed, got one our gate refuses,
        and the confirmed pin is the honest destination. Exactly one attempt: the
        destination is a constant, so there is nothing to iterate over.
      * ALREADY ON THE PIN - nothing to fall back TO; the floor IS what was just
        refused. This means localm shipped a pin its own binding rejects, which
        is a localm bug rather than an upstream one, and saying so is the whole
        value of this branch. Silently installing some older release here would
        replace a loud localm bug with a quiet unconfirmed runtime.
      * the pin itself then fails to load - reported with both causes."""
    pin = _sl.pinned_tag()
    if pin:
        console.print(
            f"[red]The pinned llama.cpp build {pin} does not load on this "
            f"machine:[/red] {detail}")
        console.print("[dim]Your pin is kept, not changed. Move it with: "
                      "localm setup-llama --rollback  (previous build), "
                      "localm setup-llama --tag default  (the build localm "
                      "ships and confirmed), or localm setup-llama --tag latest "
                      "(track upstream).[/dim]")
        return False, None

    console.print(f"[yellow]llama.cpp {rejected_tag} was rejected by localm's own "
                  f"ABI check:[/yellow] {detail}")
    console.print("[dim]That is a mismatch between the release and this build of "
                  "localm, not a fault of your machine.[/dim]")

    if rejected_tag == _PINNED_TAG or not tracks_latest():
        # No floor below the floor. Do not go hunting for some older release that
        # happens to load: that build is one nobody confirmed, and installing it
        # would turn a localm bug we can fix into a runtime we cannot vouch for.
        console.print(
            f"[red]{_PINNED_TAG} is {_FLOOR_TAG_DESCRIPTION}, so there is no "
            "more-tested build to fall back to.[/red]")
        console.print("[dim]This means this localm and its own pinned llama.cpp "
                      "build disagree, which is a bug in localm rather than in "
                      "the release. Please report it. To try another build "
                      "meanwhile: localm setup-llama --tag <release>  (for "
                      "example --tag b10361).[/dim]")
        return False, None

    console.print(f"[yellow]Falling back to {_PINNED_TAG}, {_FLOOR_TAG_DESCRIPTION}"
                  f".[/yellow]")
    try:
        try_fn(chosen, with_cudart, _PINNED_TAG)
    except Exception as e:
        console.print(f"[red]{_PINNED_TAG} could not be provisioned:[/red] {e}")
        return False, None
    ok, why = _sl._native_loads_ok()
    if not ok:
        console.print(f"[red]{_PINNED_TAG} did not load either:[/red] "
                      f"{why or 'unknown'}")
        return False, None
    # State the outcome AND why it is not what was asked for: a user who is not
    # told will report "localm installed an old runtime" as a bug.
    console.print(f"[green]OK - llama.cpp {_PINNED_TAG} loads on this machine."
                  "[/green]")
    console.print(f"[dim]Installed {_PINNED_TAG} rather than {rejected_tag}: you "
                  "asked to track upstream's newest (--tag latest) and that "
                  "release does not match this build of localm. Update localm "
                  "and re-run 'localm setup-llama --force' to move forward "
                  "again.[/dim]")
    return True, _PINNED_TAG


def _sycl_backend_note() -> str:
    """Describe the SYCL build's runtime dependency for the current OS.

    Windows and Linux ship different SYCL archives. Confirmed by inspecting
    both b10375 assets: the Windows zip bundles the whole oneAPI
    DPC++ runtime alongside ggml-sycl.dll (sycl8.dll, mkl_*.dll,
    ur_adapter_level_zero*.dll, ur_adapter_opencl.dll, tbb12.dll,
    libiomp5md.dll, dnnl.dll, sycl-ls.exe, ...), while the Linux tarball
    ships only libggml-sycl.so with none of that - a separate oneAPI
    install is still required there."""
    if sys.platform == "win32":
        return "Intel oneAPI build + self-contained oneAPI runtime"
    return "Intel oneAPI build (needs the oneAPI runtime present)"


def _provision_with_fallback(chosen: str, target: Path, sha256: Optional[str],
                             with_cudart: bool, assume_yes: bool = False,
                             cuda_line: str = _CUDA_LINE) -> tuple[str, Optional[str]]:
    """Provision *chosen* and prove it loads. If it does not load, NEVER swap the
    user's pick silently (the never-override rule): inform WHY, then OFFER the
    universal Vulkan build when interactive (or fall back with a LOUD warning when
    *assume_yes* / no tty), and always say how to retry the chosen backend with
    --force. Exits non-zero if the user declines the fallback, or if NOTHING
    loads (a genuine environment fault).

    Returns ``(backend, tag)``: the backend that ended up working AND the release
    tag it was provisioned from (None when that backend has no upstream tag).
    The tag belongs to the attempt that SUCCEEDED, so it is returned from here
    rather than recomputed by the caller: on a cuda-to-vulkan fallback the
    installed build is vulkan's, and a caller re-deriving it would record the
    tag of the backend that failed.

    *cuda_line* is the CUDA asset line to fetch when *chosen* is 'cuda' (see
    NvidiaInfo.cuda_line); irrelevant otherwise.

    vulkan and cpu are self-contained and treated as terminal: if the user
    explicitly chose one and it does not load, that is an environment problem we
    report rather than paper over with a different backend."""
    lib_name = _sl._lib_name()

    # The tag of the attempt currently in flight. _try writes it; the success
    # paths below read it. A list rather than a rebound local because _try is a
    # closure and Python would otherwise need a `nonlocal` declaration that is
    # easy to forget when a new branch is added.
    used_tag: list = [None]

    def _try(backend: str, cudart: bool, tag: Optional[str] = None) -> None:
        _sl._clear_target_or_refuse(target)
        # Cleared FIRST, so a failed attempt can never leave the previous
        # attempt's tag standing to be recorded against this backend.
        used_tag[0] = None
        used_tag[0] = _sl._provision_backend(
            backend, target, sha256 if backend == chosen else None,
            cudart, cuda_line, tag=tag)
        if not (target / lib_name).exists():
            raise ArtifactError(f"the archive did not contain {lib_name}")
        _sl._bundle_missing_native_deps(target)
        _sl._install_runtime_wheel(_sl._runtime_pkg_dir())

    notes = {
        "vulkan": "universal GPU build (AMD/NVIDIA/Intel via the display driver)",
        "amd-rocm": "self-contained AMD ROCm build (gfx103X / RX 6000)",
        "cuda": "NVIDIA CUDA build + self-contained runtime",
        "sycl": _sycl_backend_note(),
        "hip": "AMD ROCm build (needs the ROCm/HIP runtime present)",
        "cpu": "CPU-only build (no GPU)",
        "metal": "Apple Silicon (Metal) build",
    }
    console.print(f"[dim]Backend:[/dim] [bold]{chosen}[/bold]  ({notes.get(chosen, chosen)})")

    provisioned = True
    try:
        _try(chosen, with_cudart)
    except RuntimeInUseError as e:
        # MUST precede the handlers below, and must NOT fall through to the
        # Vulkan fallback. Falling back is right when the CHOSEN BUILD cannot
        # run here; it is wrong when the chosen build is fine and a process is
        # merely holding a file. Swapping the user's backend for that reason
        # would answer a question nobody asked, and the honest fix (close it and
        # retry) is one the user can actually act on.
        _exit_runtime_in_use(e)
    except click.ClickException as e:
        console.print(f"[red]{e.message}[/red]")
        provisioned = False
    except (ArtifactError, OSError) as e:
        console.print(f"[red]Provisioning {chosen} failed:[/red] {e}")
        provisioned = False
    except Exception as e:
        console.print(f"[red]Provisioning {chosen} failed:[/red] {e}")
        provisioned = False

    loaded, detail = (_sl._native_loads_ok() if provisioned else (False, "not provisioned"))
    if loaded:
        console.print(f"[green]OK - {chosen} runtime loads on this machine.[/green]")
        if chosen == "amd-rocm":
            _sl.install_rocm_simd_cpu(target)
        return chosen, used_tag[0]

    # ---- The installer must never hand the user a runtime our OWN gate rejects.
    #
    # This runs BEFORE the backend fallback below, and the order is the whole
    # point. An ABI rejection means the BUILD is wrong for this code, so EVERY
    # backend from that release fails identically - field issue 1208 reports
    # cuda, vulkan AND cpu all AbiMismatch together, and the structural reason is
    # that one shared llama library carries the struct (see _PINNED_TAG).
    # Falling back by backend first therefore cannot help, burns the whole chain,
    # and ends in "no backend could be provisioned" having also moved the user
    # off the backend they asked for. Addressing the RELEASE addresses the cause.
    #
    # Gated on an ABI rejection SPECIFICALLY, never on any load failure: "cuda
    # will not load, the driver is too old" is about this MACHINE and a different
    # release cannot fix it, so that case must still reach the vulkan fallback.
    #
    # NOT written as "avoid a known-bad tag": the property is "never ship a
    # runtime the gate rejects", whichever tag and whatever the cause, and a
    # hard-coded bad tag would be wrong the moment the binding is fixed. With
    # the pin in place this is unreachable on a default install, since the
    # pinned build is one that loaded and generated; if it fires there anyway,
    # _floor_at_pinned_tag says so rather than quietly installing something
    # else.
    if _is_abi_rejection(detail) and used_tag[0]:
        floored, floor_tag = _floor_at_pinned_tag(chosen, with_cudart, used_tag[0],
                                                  _try, detail)
        if floored:
            return chosen, floor_tag
        # Fall through: the confirmed build did not load either (or there was
        # none to fall back to), so this is not release drift after all and the
        # backend fallback is next.

    # An explicit --sha256 pin means "exactly this artifact" - never silently
    # swap to a different (unpinned) build, even to recover. Report and stop.
    if sha256:
        why = "failed validation" if not provisioned else "provisioned but did not load"
        console.print(f"[red]The pinned artifact {why}.[/red] Not falling back "
                      "(an explicit --sha256 was set).")
        sys.exit(1)

    # A self-contained backend the user pinned: do not silently swap to another.
    if chosen in ("vulkan", "cpu"):
        return _universal_backend_failed(chosen, with_cudart, assume_yes, provisioned,
                                         detail, _try, used_tag)
    return _fall_back_to_universal(chosen, detail, assume_yes, _try, used_tag)


def _universal_backend_failed(chosen: str, with_cudart: bool, assume_yes: bool,
                              provisioned: bool, detail: str, try_fn,
                              used_tag: list) -> tuple[str, Optional[str]]:
    """Report a vulkan or cpu provision that failed or did not load, and exit
    non-zero. Interactively, first offer to retry the same build; returns
    ``(chosen, tag)`` only when a retry loads."""
    if not provisioned:
        # The exception handler in _provision_with_fallback already printed
        # the SPECIFIC cause (from _diagnose_bad_artifact, when it was a
        # download/validation failure) - this adds the escape hatches, which
        # that message does not otherwise mention, so a genuinely blocked
        # network is not a dead end.
        console.print(
            f"[dim]If your network blocks or filters this download (common on "
            f"a corporate network), download the archive yourself through a "
            f"browser and use --from <extracted-dir>, or point --url at a "
            f"mirror your network allows. Retry the same command once the "
            f"cause is fixed: localm setup-llama --backend {chosen}[/dim]")
        sys.exit(1)
    # Provisioned but would not load: an environment fault, not a bad pick -
    # vulkan and cpu ARE the universal builds, so there is no different
    # backend to fall back to. Name the missing piece, offer a retry once
    # the user has had a chance to fix it, and only then give up.
    missing = _name_missing_shared_lib(detail)
    interactive = (not assume_yes) and sys.stdin.isatty()
    if interactive:
        while True:
            console.print(f"[red]You picked '{chosen}' and it was provisioned, "
                          "but the native library did not load.[/red]")
            if missing:
                console.print(f"[yellow]Missing OS library:[/yellow] {missing}")
            else:
                console.print(f"[yellow]Cause:[/yellow] {detail}")
            _sl._flush_stdin()
            if not click.confirm(
                    f"  Retry the same '{chosen}' build now (after fixing the "
                    "cause above)?", default=bool(missing)):
                break
            try:
                try_fn(chosen, with_cudart)
            except Exception as e:
                console.print(f"[red]Provisioning {chosen} failed:[/red] {e}")
                break
            loaded, detail = _sl._native_loads_ok()
            if loaded:
                console.print(f"[green]OK - {chosen} runtime loads on this machine.[/green]")
                return chosen, used_tag[0]
            missing = _name_missing_shared_lib(detail)
    # A plain print + exit, matching the "not provisioned" sibling above
    # rather than raising LocalmError: this is the one recovery path a
    # caller like setup.sh already wraps with its own report offer (see
    # handle_provision_failure), and every OTHER failure exit in
    # _provision_with_fallback is unreportable-by-the-CLI the same way - raising here
    # would make the CLI's own crash handler offer a report AND the
    # caller's wrapper offer a second one for the identical failure.
    console.print(f"[red]'{chosen}' was provisioned but the native library "
                  f"did not load.[/red]")
    console.print(f"[yellow]{'Missing OS library' if missing else 'Cause'}:[/yellow] "
                  f"{missing or detail}")
    console.print(
        f"[dim]Fix the cause and retry with: localm setup-llama --backend "
        f"{chosen} --force  -  or provide your own build with --from "
        "<build dir>. See docs/gpu-setup.md.[/dim]")
    sys.exit(1)


def _fall_back_to_universal(chosen: str, detail: str, assume_yes: bool, try_fn,
                            used_tag: list) -> tuple[str, Optional[str]]:
    """After *chosen* did not load, offer (interactive) or perform
    (non-interactive) the vulkan then cpu fallback. Returns ``(backend, tag)``
    of the first one that loads; exits non-zero when the user declines; raises
    ``LocalmError`` when none loads."""
    # chosen needs a runtime and did not load HERE. Honour the user's pick: never
    # swap it silently. INFORM why, then OFFER the universal build (interactive)
    # or fall back with a LOUD warning (non-interactive), and always say how to
    # retry the real pick once the cause is fixed.
    why = detail
    # Every attempt's own cause, chosen backend first - NOT just the last one
    # tried. The final LocalmError's *reason* is the only thing that survives
    # into the saved bug-report file and the "Sorry - X because Y" console
    # line (report_failure/build_report render summary+reason only; the
    # console.print calls below are not threaded into that context). A user
    # who explicitly picked cuda and only ever sees the final message needs to
    # know THAT failed too, not only whatever the last fallback's problem was.
    attempts = [(chosen, why)]
    console.print(f"[yellow]'{chosen}' backend provisioned but failed to load: {why}[/yellow]")
    console.print(f"[dim]To retry later: localm setup-llama --backend {chosen} --force[/dim]")
    interactive = (not assume_yes) and sys.stdin.isatty()
    if interactive:
        _sl._flush_stdin()
        if not click.confirm(
                f"  Install the universal Vulkan build now so you have a working "
                f"setup? (your '{chosen}' pick is kept, not changed; decline to stop "
                f"and fix it yourself)", default=True):
            console.print(f"[yellow]Keeping your '{chosen}' choice and stopping.[/yellow] "
                          "It does not load here yet. Fix the cause, then re-run: "
                          f"localm setup-llama --backend {chosen} --force")
            sys.exit(1)
    else:
        console.print("[yellow][!] Non-interactive: falling back to universal build.[/yellow]")
    for fb in ("vulkan", "cpu"):
        console.print(f"[yellow]Trying {fb}...[/yellow]")
        try:
            try_fn(fb, False)
        except Exception as e:
            console.print(f"[red]{fb} provisioning failed:[/red] {e}")
            attempts.append((fb, str(e)))
            continue
        ok, fb_detail = _sl._native_loads_ok()
        if ok:
            console.print(f"[green]OK - {fb} runtime loads.[/green]")
            return fb, used_tag[0]
        console.print(f"[red]{fb} provisioned but failed to load:[/red] {fb_detail or 'unknown'}")
        attempts.append((fb, fb_detail or "unknown"))
    # Nothing loaded - the one genuinely stuck case. Raise a typed, reportable
    # error and let the CLI's single graceful handler say sorry + offer a bug
    # report. setup-llama describes the failure; it does not own reporting.
    from localm.bugreport import LocalmError
    tried = "; ".join(f"{b}: {d}" for b, d in attempts)
    raise LocalmError(
        "no llama.cpp backend could be provisioned and loaded",
        reason=(f"none of {len(attempts)} backends loaded on this machine - {tried}. "
                "You can provide a local build "
                "with: localm setup-llama --from <build dir>, or see docs/gpu-setup.md."),
        context={"operation": "setup-llama", "requested_backend": chosen})
