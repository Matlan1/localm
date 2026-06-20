#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-check localm's ctypes struct layouts against an upstream ``llama.h``.

This is a VERIFIER, not a generator. It parses the by-value structs localm passes
across the FFI boundary (``llama_model_params``, ``llama_context_params``) plus
``llama_batch`` out of a real ``llama.h``, computes each field's byte offset under
natural 64-bit alignment, and diffs those offsets against
``localm.inference.backends.llamacpp._structs``. It is the tripwire to run when
bumping the bundled llama.cpp build: a mid-struct REORDER or insertion (the
dangerous, memory-corrupting kind of drift) makes a named field's offset move and
this exits non-zero; a purely trailing field ADDITION is reported as a note (it
is absorbed safely by the structs' reserved pad and the default_params round-trip,
but is worth naming).

It complements the runtime guard ``_abi.verify_abi`` (which validates whatever DLL
is actually loaded on a user's machine) and the CI load test: this one catches
header-level drift before a build is even shipped.

Usage:
    python scripts/check_llama_abi.py                 # fetch the pinned ref's header
    python scripts/check_llama_abi.py --ref b9740     # a specific tag/commit/branch
    python scripts/check_llama_abi.py --header path/to/llama.h   # a local header

Stdlib only (urllib + re + ctypes), so it runs anywhere without extra installs.
A fuller cross-check with clang2py/ctypeslib2 is possible but needs libclang; for
these few POD structs the lightweight parse here is sufficient and dependency-free.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

# The upstream ref whose llama.h localm's layouts are expected to match. localm's
# structs include ctx_other (added after b1288), matching b9682+/master. Bump
# this whenever you bump the bundled build, then reconcile any reported drift.
LLAMA_ABI_REF = "b9740"
_REPO = "ggml-org/llama.cpp"

# The structs we actually care about (passed by value / read field-by-field).
_STRUCTS = ("llama_model_params", "llama_context_params", "llama_batch")


# --------------------------------------------------------------------------- #
#  Header acquisition
# --------------------------------------------------------------------------- #

def _resolve_ref(ref: str) -> str:
    """Resolve ``latest`` to the newest upstream release tag (for CI drift checks);
    any other value is used as-is. Falls back to the pin if the lookup fails."""
    if ref != "latest":
        return ref
    import json
    api = f"https://api.github.com/repos/{_REPO}/releases/latest"
    try:
        req = urllib.request.Request(
            api, headers={"User-Agent": "localm-abi-check",
                          "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            tag = json.loads(r.read().decode("utf-8")).get("tag_name")
        if tag:
            return tag
    except Exception as e:  # noqa: BLE001
        print(f"  (could not resolve latest tag, using {LLAMA_ABI_REF}: {e})",
              file=sys.stderr)
    return LLAMA_ABI_REF


def _fetch_header(ref: str) -> str:
    """Fetch llama.h for *ref*, trying the modern include/ path then the old root."""
    last_err = None
    for path in ("include/llama.h", "llama.h"):
        url = f"https://raw.githubusercontent.com/{_REPO}/{ref}/{path}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "localm-abi-check"})
            with urllib.request.urlopen(req, timeout=20) as r:
                if r.status == 200:
                    return r.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise SystemExit(f"could not fetch llama.h for ref {ref!r}: {last_err}")


# --------------------------------------------------------------------------- #
#  Minimal C struct parsing (enough for these POD structs)
# --------------------------------------------------------------------------- #

def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _extract_struct_body(header: str, name: str) -> str:
    """Return the brace body of ``struct <name> { ... }`` from *header*."""
    m = re.search(r"struct\s+" + re.escape(name) + r"\s*\{", header)
    if not m:
        raise SystemExit(f"struct {name} not found in header")
    i = m.end()
    depth = 1
    start = i
    while i < len(header) and depth:
        if header[i] == "{":
            depth += 1
        elif header[i] == "}":
            depth -= 1
        i += 1
    return header[start:i - 1]


def _field_size(decl: str) -> int:
    """Size in bytes of a field whose declaration (type up to the name) is *decl*."""
    d = decl.strip()
    if "*" in d:
        return 8                                  # any pointer
    # Function-pointer typedefs are declared without a '*' but are 8 bytes.
    if "callback" in d:
        return 8
    if re.search(r"\b(int64_t|uint64_t|size_t)\b", d):
        return 8
    if re.search(r"\b(bool|int8_t|uint8_t|char)\b", d):
        return 1
    if re.search(r"\b(int16_t|uint16_t)\b", d):
        return 2
    # int32_t / uint32_t / float / int / unsigned / any enum -> 4
    if re.search(r"\b(int32_t|uint32_t|float|int|unsigned|enum)\b", d):
        return 4
    # Unknown scalar type: assume pointer-width so we over-report rather than
    # silently misalign. Flag it so a human checks.
    print(f"  ! unknown type, assuming 8 bytes: {d!r}", file=sys.stderr)
    return 8


def _parse_fields(body: str):
    """[(name, size)] for each ';'-terminated field declaration in *body*."""
    fields = []
    for raw in _strip_comments(body).split(";"):
        decl = raw.strip()
        if not decl:
            continue
        names = re.findall(r"[A-Za-z_]\w*", decl)
        if not names:
            continue
        name = names[-1]
        type_part = decl[: decl.rfind(name)]
        fields.append((name, _field_size(type_part)))
    return fields


def _layout(fields):
    """Compute (name, offset, size) per field under natural alignment, + total."""
    out = []
    off = 0
    max_align = 1
    for name, size in fields:
        align = size if size in (1, 2, 4, 8) else 8
        max_align = max(max_align, align)
        if off % align:
            off += align - (off % align)
        out.append((name, off, size))
        off += size
    if off % max_align:
        off += max_align - (off % max_align)
    return out, off


# --------------------------------------------------------------------------- #
#  localm side (ctypes introspection)
# --------------------------------------------------------------------------- #

def _localm_layout(struct_name: str):
    """name -> (offset, size) for localm's ctypes struct of the C *struct_name*."""
    import ctypes

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from localm.inference.backends.llamacpp import _structs as S

    cls = {
        "llama_model_params": S.LlamaModelParams,
        "llama_context_params": S.LlamaContextParams,
        "llama_batch": S.LlamaBatch,
    }[struct_name]
    out = {}
    for fname, ftype in cls._fields_:
        out[fname] = (getattr(cls, fname).offset, ctypes.sizeof(ftype))
    return out, ctypes.sizeof(cls)


# --------------------------------------------------------------------------- #
#  Diff
# --------------------------------------------------------------------------- #

def _check(struct_name: str, header: str) -> int:
    body = _extract_struct_body(header, struct_name)
    upstream, up_size = _layout(_parse_fields(body))
    localm, lm_size = _localm_layout(struct_name)

    print(f"\n=== {struct_name} ===")
    print(f"  upstream native size: {up_size}   localm allocated size: {lm_size}")
    problems = 0
    last_named = max((o + s for _, o, s in upstream), default=0)

    for name, off, size in upstream:
        if name not in localm:
            if off >= _localm_last_named(localm):
                print(f"  NOTE  upstream field {name!r} at offset {off} is trailing and "
                      "not named in localm (absorbed by the reserved pad; consider naming it)")
            else:
                print(f"  FAIL  upstream field {name!r} at offset {off} missing in localm "
                      "(mid-struct - shifts the offsets of later fields)")
                problems += 1
            continue
        lm_off, lm_size_f = localm[name]
        if lm_off != off:
            print(f"  FAIL  {name}: upstream offset {off} != localm offset {lm_off}")
            problems += 1
        elif lm_size_f != size:
            print(f"  FAIL  {name}: upstream size {size} != localm size {lm_size_f} "
                  f"(offset {off})")
            problems += 1
        else:
            print(f"  ok    {name:30s} offset {off:3d} size {size}")

    if lm_size < last_named:
        print(f"  FAIL  localm struct ({lm_size}B) is SMALLER than upstream's named "
              f"fields ({last_named}B) - would under-allocate and read past the buffer")
        problems += 1
    return problems


def _localm_last_named(localm) -> int:
    """End offset of localm's last NON-reserved/pad named field."""
    end = 0
    for name, (off, size) in localm.items():
        if name.startswith("_pad") or name == "_reserved":
            continue
        end = max(end, off + size)
    return end


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default=LLAMA_ABI_REF,
                    help=f"upstream tag/commit/branch, or 'latest' to resolve the "
                         f"newest release (default {LLAMA_ABI_REF})")
    ap.add_argument("--header", default=None,
                    help="path to a local llama.h instead of fetching")
    args = ap.parse_args()

    if args.header:
        header = Path(args.header).read_text(encoding="utf-8", errors="replace")
        print(f"Checking localm structs against header: {args.header}")
    else:
        ref = _resolve_ref(args.ref)
        print(f"Checking localm structs against {_REPO}@{ref}")
        header = _fetch_header(ref)

    total = 0
    for s in _STRUCTS:
        total += _check(s, header)

    print()
    if total:
        print(f"ABI CHECK FAILED: {total} mismatch(es). A mid-struct reorder/insert "
              "corrupts memory - update _structs.py (and _abi anchors) to match, then "
              "re-probe a real build. See issues/abi-verification-design-2026-06-20.md.")
        return 1
    print("ABI check passed: localm's named fields match upstream offsets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
