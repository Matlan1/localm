# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prints the architecture name (``gfx1030``) of every GPU the HIP runtime in a
folder reports, one per line. Run in a child interpreter by
:func:`localm.setup_llama.hip_libraries.gpu_archs`:

    python -I _hip_probe.py <folder holding amdhip64_*.dll>

Exit 0 with the names (none when the runtime finds no GPU), 2 when the runtime
does not load or the device count fails, with the reason on stderr. Imports
nothing of localm, so it runs with ``-I``.
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
from pathlib import Path
from typing import Optional

# hipGetDeviceProperties writes a hipDeviceProp_t, whose first field is the
# 256-byte marketing name and which carries gcnArchName ("gfx1030", optionally
# with feature flags such as "gfx90a:sramecc+:xnack-") further in. The buffer is
# larger than any revision of the struct.
_PROPS_BYTES = 16384
_NAME_BYTES = 256
_ARCH_RE = re.compile(rb"(gfx[0-9a-f]+)(?=[:\x00])")

# hipErrorNoDevice: the runtime loaded and found no GPU.
_HIP_ERROR_NO_DEVICE = 100


def arch_from_properties(props: bytes) -> Optional[str]:
    """The architecture name in the hipDeviceProp_t bytes *props*: the first
    ``gfx<hex>`` string after the name field, without feature flags. None when
    there is none."""
    m = _ARCH_RE.search(props, _NAME_BYTES)
    return m.group(1).decode("ascii") if m else None


def _runtime(folder: Path) -> Optional[Path]:
    """The HIP runtime DLL in *folder* (the highest-numbered ``amdhip64_*.dll``)."""
    found = sorted(folder.glob("amdhip64_*.dll"))
    return found[-1] if found else None


def main(argv: list) -> int:
    folder = Path(argv[1])
    runtime = _runtime(folder)
    if runtime is None:
        sys.stderr.write(f"no amdhip64_*.dll in {folder}")
        return 2
    try:
        os.add_dll_directory(str(folder))
        hip = ctypes.CDLL(str(runtime))
    except OSError as e:
        sys.stderr.write(f"{runtime.name} did not load: {e}")
        return 2
    count = ctypes.c_int(0)
    rc = hip.hipGetDeviceCount(ctypes.byref(count))
    if rc == _HIP_ERROR_NO_DEVICE:
        return 0
    if rc != 0:
        sys.stderr.write(f"hipGetDeviceCount returned error {rc}")
        return 2
    get = getattr(hip, "hipGetDevicePropertiesR0600", None) or hip.hipGetDeviceProperties
    for i in range(count.value):
        props = ctypes.create_string_buffer(_PROPS_BYTES)
        rc = get(props, ctypes.c_int(i))
        arch = arch_from_properties(props.raw) if rc == 0 else None
        if arch is None:
            sys.stderr.write(f"device {i}: no architecture name (error {rc})\n")
            continue
        sys.stdout.write(arch + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
