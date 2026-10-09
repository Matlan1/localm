# SPDX-License-Identifier: AGPL-3.0-or-later
"""Load-testing a provisioned runtime in a child interpreter, and turning a
failed load into a one-line cause.
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Optional

_EXC_HEADER_RE = re.compile(
    r"^(?:[\w.]+\.)?\w*(?:Error|Exception|Warning|Interrupt|Exit)(?::|\s|\Z)")


def _informative_error_line(text: str) -> str:
    """Pull the line that actually explains a failed load from a subprocess's
    captured output.

    A Python traceback ends with the exception, but when that exception carries a
    MULTI-LINE message the literal last line is not the cause. ``load_lib()``
    raises a ``RuntimeError`` whose first line is the real dlopen error (e.g.
    ``libgomp.so.1: cannot open shared object file``) followed by four
    re-provision hint lines; a blind ``splitlines()[-1]`` returns the last hint
    (``localm setup-llama --backend amd-rocm --force  (AMD RX 6000)``) and throws
    the actual cause away, so setup reports a nonsensical "still failed to load
    (<a command>)" reason.

    Prefer the exception HEADER line (``SomeError: <cause>``), which carries the
    real error even when the message spans several lines; fall back to the last
    non-empty line when the output is not a recognisable traceback."""
    lines = [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "library failed to load"
    for ln in reversed(lines):
        if _EXC_HEADER_RE.match(ln.lstrip()):
            return ln.strip()
    return lines[-1].strip()


# Sonames known to reach the dlopen error a load failure can carry, mapped to
# the Debian/Ubuntu package that provides them - used only to phrase an
# actionable message. libgomp.so.1 is bundled automatically on Linux (see
# _bundle_missing_native_deps), so this fires for it only when that bundling
# did not run or did not help; a vendor library like libvulkan genuinely has
# to come from the system either way.
_KNOWN_SHARED_LIB_PACKAGES = {
    "libgomp.so.1": "libgomp1",
    "libvulkan.so.1": "libvulkan1 (or your GPU vendor's Vulkan ICD/driver package)",
}


_MISSING_SO_RE = re.compile(r"([\w.+-]+\.so(?:\.[\w.]+)?): cannot open shared object file")


def _name_missing_shared_lib(detail: str) -> Optional[str]:
    """A plain-words description of the OS shared library a dlopen failure in
    *detail* named, or None when *detail* is not that shape.

    *detail* is already the trimmed exception-header line
    _informative_error_line produces (e.g. ``RuntimeError: Failed to load
    libllama.so from ...: libgomp.so.1: cannot open shared object file: No
    such file or directory``); this pulls the soname back out of it rather
    than re-running the load probe."""
    m = _MISSING_SO_RE.search(detail or "")
    if not m:
        return None
    soname = m.group(1)
    package = _KNOWN_SHARED_LIB_PACKAGES.get(soname)
    if package:
        return f"{soname} - on Debian/Ubuntu: sudo apt install {package}"
    return f"{soname} - install the OS package that provides it, then retry"


# Exit codes the load probe uses to tell its outcomes apart STRUCTURALLY rather
# than by matching text in a traceback. 88 predates this. 89 exists so the tag
# walk-back can fire on an ABI rejection SPECIFICALLY and not on, say, a CUDA
# build refusing to load because the driver is too old - those need opposite
# responses (walk back a release vs fall back to another backend), and telling
# them apart by grepping an exception message would depend on wording that is
# upstream's to change, not ours.
_PROBE_NO_BACKENDS = 88


_PROBE_ABI_MISMATCH = 89


# The prefix _native_loads_ok puts on an ABI rejection. A string WE own on both
# ends - written here, matched by _is_abi_rejection - so it cannot drift with
# anyone else's message. Not a substring search over a traceback.
_ABI_REJECT_PREFIX = "the runtime does not match this build's struct layout"


# load_lib() runs verify_abi and RE-RAISES (see _loader.py: `except Exception:
# _loaded_lib = None; raise`), so AbiMismatch propagates out uncaught and can be
# caught here. Verified by reading that call site, not assumed.
_LOAD_PROBE_CODE = f"""\
import sys
from localm.inference.backends.llamacpp import _loader
from localm.inference.backends.llamacpp._abi import AbiMismatch
try:
    _loader.load_lib()
except AbiMismatch as e:
    sys.stderr.write(str(e))
    sys.exit({_PROBE_ABI_MISMATCH})
sys.exit(0 if _loader.compute_backends_available() else {_PROBE_NO_BACKENDS})
"""


def _is_abi_rejection(detail: Optional[str]) -> bool:
    """Whether *detail* is _native_loads_ok reporting OUR OWN ABI gate refusing
    the runtime, as opposed to any other load failure.

    The discriminator for the tag walk-back: an ABI rejection means the BUILD is
    wrong for this code, which a different release can fix; every other load
    failure is about this machine, which a different release cannot."""
    return str(detail or "").startswith(_ABI_REJECT_PREFIX)


def _native_loads_ok() -> tuple:
    """Load-test the provisioned native library in a FRESH interpreter, exactly
    as ``localm run`` will, AND confirm it registered a compute backend. A build
    can load cleanly yet register ZERO backends ("no backends are loaded"), which
    must count as a FAILED provision, not a silent success - otherwise
    _provision_with_fallback's "prove it loads" guarantee holds only for
    self-registering builds and a broken runtime slips through, failing only at
    the first model load with the real cause already lost. A subprocess keeps the
    setup process clean (the loader mutates the DLL/lib search path) and matches
    the real run environment. Returns (ok, last_error_line)."""
    try:
        r = subprocess.run([sys.executable, "-c", _LOAD_PROBE_CODE],
                           capture_output=True, text=True, timeout=120)
    except Exception as e:
        return False, str(e)
    if r.returncode == 0:
        return True, ""
    if r.returncode == _PROBE_NO_BACKENDS:
        return False, ('runtime loaded but registered no compute backends '
                       '("no backends are loaded") - this build does not fit this machine')
    if r.returncode == _PROBE_ABI_MISMATCH:
        # Kept behind its own prefix so callers can recognise this specific
        # outcome without re-parsing upstream's wording - see _is_abi_rejection.
        why = _informative_error_line((r.stderr or "").strip()) or "layout drift"
        return False, f"{_ABI_REJECT_PREFIX}: {why}"
    detail = (r.stderr or r.stdout or "").strip()
    return False, _informative_error_line(detail)
