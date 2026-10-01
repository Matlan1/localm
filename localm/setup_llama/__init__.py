# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm setup-llama`` - provision the native llama.cpp binaries locally.

Makes localm self-contained: the native inference runtime (the llama shared
library + its ggml deps, plus a matched GPU runtime when the prebuilt ships one)
is placed inside the project's own ``localm-llama-runtime`` wheel rather than
depending on a folder elsewhere on disk.

Backends (``--backend``), so any machine has a working out-of-the-box path:
  * ``auto`` (default) - detect the GPU and pick the fastest backend that works
    with no user-installed toolkit: NVIDIA, any OS -> ``cuda`` (self-contained
    build + runtime fetch on both Windows and Linux, see below); AMD on Windows
    (RX 6000 / unknown) -> the self-contained ROCm build; AMD elsewhere with a
    system ROCm/HIP toolkit detected present -> ``hip``; Intel on Windows ->
    ``sycl`` (self-contained); Intel on Linux, and AMD with no toolkit detected
    -> ``vulkan`` (runs on NVIDIA/Intel/AMD through the normal display driver,
    no vendor toolkit); Apple Silicon -> ``metal``; no GPU -> ``cpu``. See
    ``hwdetect.recommended_install_backend`` for the full policy.
  * ``vulkan`` - universal GPU build from upstream llama.cpp (a no-toolkit
    fallback for any vendor; the default for Intel on Linux, and for AMD with
    no ROCm/HIP toolkit detected).
  * ``cuda`` - NVIDIA peak performance, self-contained on BOTH Windows and Linux:
    the matching ``cudart`` runtime bundle (Windows) or CUDA runtime libraries
    (Linux, fetched from PyPI) are fetched alongside the build, so NO CUDA
    Toolkit is needed on either OS; a driver preflight + load-test fall back to
    ``vulkan`` if the driver is too old. The CUDA asset LINE is also chosen from
    the detected GPU architecture on both platforms (Blackwell - sm_100/sm_120 -
    automatically gets the newer 13.x line; every older architecture stays on
    the broad-compatibility 12.x line).
  * ``hip`` - AMD peak performance via an already-installed system ROCm/HIP
    toolkit (a real downloadable prebuilt binary on both Windows and Linux;
    needs that toolkit present to load - see ``_rocm_toolkit_present`` in
    hwdetect.py).
  * ``sycl`` / ``cpu`` - upstream llama.cpp prebuilts. ``sycl`` delivers peak
    Intel performance; the Windows build bundles the whole oneAPI DPC++
    runtime and is self-contained (the auto default on Windows), while the
    Linux build does not and needs oneAPI installed separately, so it stays
    opt-in there; ``cpu`` is self-contained.
  * ``amd-rocm`` - the self-contained gfx103X (RDNA2) ROCm build (bundles its
    own ROCm runtime; the current default for AMD RX 6000 on Windows, since it
    needs no system toolkit at all).

Sources, in order of preference:
  * ``--from <dir>``  - copy from a local llama.cpp build output (any backend).
  * ``--url <url>``   - an explicit prebuilt archive URL.
  * ``--backend ...`` - resolve the matching asset of the PINNED llama.cpp
    release (``ggml-org/llama.cpp``); see below.

Which BUILD, as distinct from which backend:
  * localm installs ``_PINNED_TAG``: one upstream release we confirmed loads AND
    generates, decided in pins.py. No version is ever computed while setup is
    running, so a release published by a third party cannot change what an
    install gets. See _PINNED_TAG's own comment for what "confirmed" covers per
    backend, and scripts/confirm_llama_runtime.py for the check that earns it.
  * The installed release tag is recorded in the runtime dir's marker alongside
    the backend, so ``localm doctor`` and a bug report can name the build
    instead of inferring it from library filenames.
  * ``--tag <tag>`` installs one exact release and PINS it, so later runs and
    ``localm update``'s re-provision keep it. ``--tag latest`` opts IN to
    upstream's newest release, which localm has not confirmed; ``--tag default``
    returns to the shipped pin. All three live in one config key
    (``llama_runtime_pin``) read by ``_tag_for``, so the updater inherits the
    choice without knowing it exists.
  * ``--rollback`` returns to the previous build recorded for this backend.
    This exists because upstream can ship a release that is broken on a given
    machine - the ``llama_context_params`` ABI shift, an asset rename, a
    backend-specific fault - and until now there was no way to step off it.

After placing the files it installs the runtime wheel editable so the loader can
import it.

This package module is the public surface; the implementation lives in its
submodules. Inside the package, every name that callers replace on
``localm.setup_llama`` is read through it at call time (``_sl.NAME``), so a
replacement here reaches every call site. See
test_live_names_are_never_read_by_bare_name.
"""

from __future__ import annotations

import subprocess  # noqa: F401
import sys  # noqa: F401

import click  # noqa: F401

from localm import config  # noqa: F401
from localm.http_ssl import verified_urlopen
from localm.setup_llama._common import (
    console, _flush_stdin,
)
from localm.setup_llama.pins import (
    DEFAULT_URL, DEFAULT_URL_SHA256, _ROCM_TAG, _AMD_ROCM_ASSET_TAG, _UPSTREAM_REPO,
    _PINNED_TAG, _PIN_CONFIRMATION, _TRACK_LATEST, _TRACK_DEFAULT, _CUDA_LINUX_REPO,
    _PINNED_FALLBACK_SHA256, _ASSET_MATCH, _UPSTREAM_BACKENDS,
)
from localm.setup_llama.download import (
    _MIN_ARTIFACT_BYTES, _DOWNLOAD_STALL_TIMEOUT, _DownloadResult, ArtifactError,
    _download, _sha256_file, _is_supported_archive, _sniff_content_kind,
    _diagnose_bad_artifact, _validate_archive, _safe_extractall_tar, _extract_archive,
    _human_mb, _fetch_and_place, _fetch_verified,
)
from localm.setup_llama.library_files import (
    _is_wanted, _BLAS_LIBRARY_DIRS, _BLAS_DIRS_REQUIRING_KERNELS, _has_vendor_library,
    blas_kernel_problems, _copy_blas_library_dirs, _LLAMA_CPP_MIT_NOTICE,
    _LICENSE_NAME_PREFIXES, _copy_license_files, _safe_is_file, _copy_binaries,
)
from localm.setup_llama.runtime_dir import (
    _platform_key, _lib_name, _BACKEND_MARKER, _record_provisioned_backend,
    _read_marker, _provisioned_backend, _provisioned_build, installed_backend,
    installed_build, installed_runtime_identity, _repo_runtime_lib, _runtime_pkg_dir,
    _install_runtime_wheel, _PRESERVED_TARGET_FILES, RuntimeInUseError, _clearable_files, _files_in_use,
    _clear_target, _clear_target_or_refuse, _exit_runtime_in_use, _PROVISION_LOCK_OWNER,
    ProvisioningBusyError, _provision_lock_path, _provision_lock_holder_pid,
    _provisioning_lock, _exit_provisioning_busy,
)
from localm.setup_llama.versions import (
    _RUNTIME_HISTORY_MAX, _TAG_SAFE_RE, TAG_HELP, is_safe_tag, tracks_latest,
    pinned_tag, set_pinned_tag, _record_runtime_history, runtime_history, previous_tag,
    check_runtime_update, _tag_for, _pin_note_for_backend, _latest_tag, _RELEASE_TAG_RE,
    _recent_tags, _validated_tag, _apply_version_request,
)
from localm.setup_llama.assets import (
    _auto_backend, _resolve_backend_asset, _resolve_backend_url, _release_assets,
    _pick_asset, _resolve_cuda_pair,
)
from localm.setup_llama.cuda import (
    _CUDA_LINE, _BLACKWELL_MIN_CAP, _MIN_DRIVER_CUDA, _ver_tuple, _ver_at_least,
    NvidiaInfo, _nvidia_smi, nvidia_preflight, _CUDA_RUNTIME_PYPI_PACKAGES,
    _pypi_wheel_url_and_sha, _fetch_pypi_runtime_lib, _fetch_cuda_runtime_libs,
    _cuda_setup_dialogue,
)
from localm.setup_llama.native_deps import (
    _LIBGOMP_SONAME, _LIBGOMP_DEB_URL, _LIBGOMP_DEB_SHA256, _LIBGOMP_DEB_MIN_BYTES,
    _LIBGOMP_LICENSE_NOTICE, _read_ar_archive, _extract_libgomp_from_deb,
    _bundle_missing_native_deps,
)
from localm.setup_llama.load_probe import (
    _EXC_HEADER_RE, _informative_error_line, _KNOWN_SHARED_LIB_PACKAGES, _MISSING_SO_RE,
    _name_missing_shared_lib, _PROBE_NO_BACKENDS, _PROBE_ABI_MISMATCH,
    _ABI_REJECT_PREFIX, _LOAD_PROBE_CODE, _is_abi_rejection, _native_loads_ok,
)
from localm.setup_llama.provision import (
    _provision_backend, _warn_off_profile, _FLOOR_TAG_DESCRIPTION, _floor_at_pinned_tag,
    _sycl_backend_note, _provision_with_fallback,
)
from localm.setup_llama.cli import (
    BACKENDS, main, _refresh_install_record, _ensure_importable, _verify,
)

__all__ = [
    "_ABI_REJECT_PREFIX", "_AMD_ROCM_ASSET_TAG", "_apply_version_request",
    "ArtifactError", "_ASSET_MATCH", "_auto_backend", "_BACKEND_MARKER", "BACKENDS",
    "_BLACKWELL_MIN_CAP", "_BLAS_DIRS_REQUIRING_KERNELS", "blas_kernel_problems",
    "_BLAS_LIBRARY_DIRS", "_bundle_missing_native_deps", "check_runtime_update",
    "_clear_target", "_clear_target_or_refuse", "_clearable_files", "console",
    "_copy_binaries", "_copy_blas_library_dirs", "_copy_license_files", "_CUDA_LINE",
    "_CUDA_LINUX_REPO", "_CUDA_RUNTIME_PYPI_PACKAGES", "_cuda_setup_dialogue",
    "DEFAULT_URL", "DEFAULT_URL_SHA256", "_diagnose_bad_artifact", "_download",
    "_DOWNLOAD_STALL_TIMEOUT", "_DownloadResult", "_ensure_importable",
    "_EXC_HEADER_RE", "_exit_provisioning_busy", "_exit_runtime_in_use",
    "_extract_archive", "_extract_libgomp_from_deb", "_fetch_and_place",
    "_fetch_cuda_runtime_libs", "_fetch_pypi_runtime_lib", "_fetch_verified",
    "_files_in_use", "_floor_at_pinned_tag", "_FLOOR_TAG_DESCRIPTION", "_flush_stdin",
    "_has_vendor_library", "_human_mb", "_informative_error_line",
    "_install_runtime_wheel", "installed_backend", "installed_build",
    "installed_runtime_identity",
    "_is_abi_rejection", "is_safe_tag", "_is_supported_archive", "_is_wanted",
    "_KNOWN_SHARED_LIB_PACKAGES", "_latest_tag", "_lib_name", "_LIBGOMP_DEB_MIN_BYTES",
    "_LIBGOMP_DEB_SHA256", "_LIBGOMP_DEB_URL", "_LIBGOMP_LICENSE_NOTICE",
    "_LIBGOMP_SONAME", "_LICENSE_NAME_PREFIXES", "_LLAMA_CPP_MIT_NOTICE",
    "_LOAD_PROBE_CODE", "main", "_MIN_ARTIFACT_BYTES", "_MIN_DRIVER_CUDA",
    "_MISSING_SO_RE", "_name_missing_shared_lib", "_native_loads_ok",
    "nvidia_preflight", "_nvidia_smi", "NvidiaInfo", "_pick_asset", "_PIN_CONFIRMATION",
    "_pin_note_for_backend", "_PINNED_FALLBACK_SHA256", "_PINNED_TAG", "pinned_tag",
    "_platform_key", "_PRESERVED_TARGET_FILES", "previous_tag", "_PROBE_ABI_MISMATCH",
    "_PROBE_NO_BACKENDS", "_provision_backend", "_provision_lock_holder_pid",
    "_PROVISION_LOCK_OWNER", "_provision_lock_path", "_provision_with_fallback",
    "_provisioned_backend", "_provisioned_build", "_provisioning_lock",
    "ProvisioningBusyError", "_pypi_wheel_url_and_sha", "_read_ar_archive",
    "_read_marker", "_recent_tags", "_record_provisioned_backend",
    "_record_runtime_history", "_refresh_install_record", "_release_assets",
    "_RELEASE_TAG_RE", "_repo_runtime_lib", "_resolve_backend_asset",
    "_resolve_backend_url", "_resolve_cuda_pair", "_ROCM_TAG", "runtime_history",
    "_RUNTIME_HISTORY_MAX", "_runtime_pkg_dir", "RuntimeInUseError",
    "_safe_extractall_tar", "_safe_is_file", "set_pinned_tag", "_sha256_file",
    "_sniff_content_kind", "_sycl_backend_note", "_tag_for", "TAG_HELP", "_TAG_SAFE_RE",
    "_TRACK_DEFAULT", "_TRACK_LATEST", "tracks_latest", "_UPSTREAM_BACKENDS",
    "_UPSTREAM_REPO", "_validate_archive", "_validated_tag", "_ver_at_least",
    "_ver_tuple", "verified_urlopen", "_verify", "_warn_off_profile",
]
