# SPDX-License-Identifier: AGPL-3.0-or-later
"""The upstream reference for LoRA token comparisons.

The provisioned llama runtime ships upstream's own ``llama-completion`` program
as a library (``llama-completion-impl``) exporting ``int llama_completion(int
argc, char **argv)``. ``upstream_completion_text`` runs it, in a child process,
with ``--lora``, so a test can compare localm's output to the output of the
code upstream's command line runs. ``None`` from ``upstream_completion_dll``
means the runtime has no such library (a different packaging), which is an
absent oracle, not a failure.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

_IMPL_NAMES = ("llama-completion-impl.dll",)
_ENTRY = "?llama_completion@@YAHHPEAPEAD@Z"

_CHILD = r"""
import ctypes, json, os, sys
dll_path = sys.argv[1]
args = [a.encode("utf-8") for a in json.loads(sys.argv[2])]
entry = sys.argv[3]
lib_dir = os.path.dirname(dll_path)
os.add_dll_directory(lib_dir)
os.environ["PATH"] = lib_dir + os.pathsep + os.environ.get("PATH", "")
lib = ctypes.CDLL(dll_path)
fn = getattr(lib, entry)
fn.restype = ctypes.c_int
fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
argv = (ctypes.c_char_p * (len(args) + 1))(*args, None)
sys.stdout.flush()
rc = fn(len(args), argv)
sys.stdout.flush()
os._exit(rc)
"""


def upstream_completion_dll() -> Optional[Path]:
    """The provisioned runtime's upstream completion library, or None."""
    from tests._real_gguf import native_runtime_lib_path
    lib = native_runtime_lib_path()
    if lib is None:
        return None
    for name in _IMPL_NAMES:
        candidate = lib.parent / name
        if candidate.is_file():
            return candidate
    return None


def upstream_completion_text(model: str, prompt: str, n_predict: int, *,
                             adapter: Optional[str] = None,
                             seed: int = 1, timeout: float = 600.0) -> str:
    """The text upstream's completion command generates after *prompt*: *n_predict*
    greedy tokens with the weights on the CPU, with the LoRA *adapter* (at scale
    1.0) when given. Raises RuntimeError with upstream's stderr when it exits
    non-zero."""
    dll = upstream_completion_dll()
    if dll is None:
        raise RuntimeError("the runtime has no upstream completion library")
    args = ["llama-completion", "-m", model, "-p", prompt, "-n", str(n_predict),
            "--temp", "0", "--top-k", "1", "--repeat-penalty", "1.0",
            "-s", str(seed), "-ngl", "0", "-c", "512", "-t", "2", "-b", "512",
            "-no-cnv", "--no-display-prompt", "--no-warmup", "--simple-io"]
    if adapter is not None:
        args += ["--lora", adapter]
    done = subprocess.run(
        [sys.executable, "-I", "-c", _CHILD, str(dll), json.dumps(args), _ENTRY],
        capture_output=True, timeout=timeout)
    if done.returncode != 0:
        raise RuntimeError(
            f"upstream llama_completion exited {done.returncode}:\n"
            + done.stderr.decode("utf-8", "replace")[-2000:])
    return done.stdout.decode("utf-8", "replace")
