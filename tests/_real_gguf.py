# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared preconditions for the real-GGUF integration tests.

Every real_gguf test needs the native llama runtime and a model file. Both are
resources a host may genuinely lack, so their ABSENCE is a skip. Their PRESENCE
turns every later failure into a real one: a runtime that is on disk but does
not load is an ABI, loader or packaging defect, and a model that is on disk but
does not load is a backend defect. Neither is an environment gap, and neither
may be reported as a skip.

These helpers are the one place that decides what counts as absence. A test
file calls them instead of wrapping ``load_lib()``, ``hf_hub_download()`` or a
model load in ``except Exception: pytest.skip(...)``; tests/test_real_gguf_gate.py
pins that contract and fails on any file that wraps those calls itself.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pytest


def native_runtime_lib_path() -> Optional[Path]:
    """The provisioned native llama library file, or None when no candidate
    directory holds it. Looks in exactly the places ``load_lib()`` looks
    (``runtime_binary_dir()``, then the parent of an explicit ``LLAMA_CPP_LIB``)
    and checks only that the file exists. Loads nothing."""
    from localm.inference.backends.llamacpp._loader import (
        lib_filename, runtime_binary_dir)
    name = lib_filename()
    binary_dir = runtime_binary_dir()
    if binary_dir is None:
        explicit = os.environ.get("LLAMA_CPP_LIB")
        if not explicit:
            return None
        binary_dir = Path(explicit).parent
    lib_path = binary_dir / name
    try:
        present = lib_path.is_file()
    except OSError:
        present = False
    return lib_path if present else None


def require_native_runtime(setup_hint: str = "localm setup-llama") -> None:
    """Skip when the native llama runtime is not provisioned; otherwise load
    it. A runtime that is on disk but fails to load raises, and the test that
    called this errors with ``load_lib()``'s own message."""
    if native_runtime_lib_path() is None:
        pytest.skip(f"native llama runtime not provisioned (run '{setup_hint}')")
    from localm.inference.backends.llamacpp._loader import load_lib
    load_lib()


def _fetch_skip_types() -> tuple:
    import httpx
    from huggingface_hub.errors import (
        HfHubHTTPError, LocalEntryNotFoundError, XetDownloadError)
    return (LocalEntryNotFoundError, HfHubHTTPError, XetDownloadError,
            httpx.HTTPError)


def fetch_gguf(repo_id: str, filename: str, **kwargs) -> str:
    """Download (or resolve from the local cache) one file from the Hub and
    return its path. Skips only when the fetch fails for a network, offline or
    Hub-side reason: the Hub unreachable or in offline mode with no cached
    copy (LocalEntryNotFoundError), an HTTP error from the Hub or the CDN
    (HfHubHTTPError, httpx.HTTPError), or a transport failure of the download
    itself (httpx transport errors, XetDownloadError). Any other exception
    propagates.

    The download runs over plain HTTP for the duration of the call:
    ``huggingface_hub.constants.HF_HUB_DISABLE_XET`` is set and then restored
    to its previous value, so a transfer failure arrives as one of the typed
    httpx errors above. See test_fetch_disables_xet_for_the_call."""
    from huggingface_hub import constants, hf_hub_download
    saved = constants.HF_HUB_DISABLE_XET
    constants.HF_HUB_DISABLE_XET = True
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
    except _fetch_skip_types() as e:
        pytest.skip(f"could not fetch {repo_id}/{filename}: {e}")
    finally:
        constants.HF_HUB_DISABLE_XET = saved
