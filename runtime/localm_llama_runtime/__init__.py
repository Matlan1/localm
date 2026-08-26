# SPDX-License-Identifier: AGPL-3.0-or-later
"""Project-local home for localm's native llama.cpp inference binaries.

Installed (editable) into localm's venv, so the project carries its own
inference runtime. The binaries live in ``lib/`` next to this file and are
provisioned by ``localm setup-llama``; they are never committed.

On a GPU build the bundled set is the matched llama.cpp + ROCm/CUDA runtime
(e.g. the lemonade-sdk gfx1030 prebuilt: llama.dll, ggml-*.dll, amdhip64_7.dll,
rocm_kpack.dll, rocblas.dll, ...). A minimal from-source build may ship only
llama.dll + ggml-*.dll and lean on the ROCm runtime that the rocm-sdk wheels
already place in the same venv.
"""

from __future__ import annotations

from pathlib import Path

LIB_DIR = Path(__file__).resolve().parent / "lib"

# The shared library a caller actually loads, per platform.
_LOADER_LIBS = ("llama.dll", "libllama.so", "libllama.dylib")


def lib_dir() -> Path | None:
    """Return the bundled-binary directory if it has been provisioned with a
    loadable llama library, else None (not set up yet)."""
    if any((LIB_DIR / name).exists() for name in _LOADER_LIBS):
        return LIB_DIR
    return None


def is_provisioned() -> bool:
    return lib_dir() is not None
