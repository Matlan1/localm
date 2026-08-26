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

IT ALSO CHECKS ENUM DOMAINS, WHICH IS A SEPARATE QUESTION FROM LAYOUT. An
offset check reads WHERE a field sits and is blind to WHICH VALUES are legal in
it: a new enum member can become a function's default return without moving any
field by a byte.

The two enum outcomes are DISTINCT:

  * a NEW member appearing upstream is ADDITIVE. Reported loudly, exit code
    unchanged. It only becomes dangerous when it becomes the DEFAULT, which a
    header cannot show - see ``_check_enum``.
  * a CHANGED VALUE for a member localm already binds is the corrupting class.
    Hard fail.

It complements the runtime guard ``_abi.verify_abi`` (which validates whatever DLL
is actually loaded on a user's machine) and the CI load test: this one catches
header-level drift before a build is even shipped.

Usage:
    python scripts/check_llama_abi.py                 # fetch the pinned ref's header
    python scripts/check_llama_abi.py --ref b9740     # a specific tag/commit/branch
    python scripts/check_llama_abi.py --header path/to/llama.h   # a local header

Stdlib only (urllib + re + ctypes), so it runs anywhere without extra installs.
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
import urllib.request
from pathlib import Path
from typing import NamedTuple

# One upstream ref per llama_model_params layout localm binds. The default run
# checks both.
LLAMA_ABI_REFS = {
    "v1": "b9870",    # pre-reorder: use_mmap/use_direct_io/use_mlock, main_gpu@24
    "v2": "b10360",   # post-reorder: load_mode@24, main_gpu@28, load_mtp
}
LLAMA_ABI_REF = LLAMA_ABI_REFS["v2"]
_REPO = "ggml-org/llama.cpp"

# Structs passed by value or read field-by-field.
_STRUCTS = ("llama_model_params", "llama_context_params", "llama_batch")


class _EnumBinding(NamedTuple):
    """One upstream enum whose DOMAIN localm binds, and where it binds it.

    The member VALUES are never restated here - they are read live out of the
    localm module named below. This registry records only WHERE localm's copy
    lives, so the verifier reads the same constants the runtime does.
    """
    c_enum: str      # the enum type's name in llama.h
    module: str      # the localm module holding the bound constants
    prefix: str      # localm-side constant prefix; the suffix is the member name
    c_prefix: str    # upstream member prefix, prepended to that same suffix
    members: tuple   # member suffixes localm binds, named rather than guessed
    gate: tuple      # (struct, field) that must exist for this enum to be expected


# Modules listed here must import cleanly with no side effect. Member suffixes
# are named explicitly rather than scanned by prefix; the `others` cross-check in
# _check_enum catches a binding this list omits.
_ENUM_BINDINGS = (
    _EnumBinding(
        c_enum="llama_load_mode",
        module="localm.inference.backends.llamacpp._structs",
        prefix="LLAMA_LOAD_MODE_",
        c_prefix="LLAMA_LOAD_MODE_",
        members=("AUTO", "NONE", "MMAP", "MLOCK", "MMAP_MLOCK", "DIRECT_IO"),
        gate=("llama_model_params", "load_mode"),
    ),
    _EnumBinding(
        c_enum="llama_pooling_type",
        module="localm.inference.embedder",
        prefix="_POOLING_",
        c_prefix="LLAMA_POOLING_TYPE_",
        members=("UNSPECIFIED", "NONE", "MEAN", "CLS", "LAST"),
        gate=("llama_context_params", "pooling_type"),
    ),
)


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


def _extract_block_body(header: str, keyword: str, name: str):
    """Brace body of ``<keyword> <name> { ... }``, or None if there is no such block.

    Requiring a ``{`` is what keeps this off the many non-definition mentions of
    the same name - ``enum llama_load_mode load_mode;`` inside a struct, and
    ``llama_load_mode_from_str(...)`` in a prototype, both match the name and
    neither is a definition.
    """
    m = re.search(keyword + r"\s+" + re.escape(name) + r"\s*\{", header)
    if not m:
        return None
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


def _extract_struct_body(header: str, name: str) -> str:
    """Return the brace body of ``struct <name> { ... }`` from *header*."""
    body = _extract_block_body(header, "struct", name)
    if body is None:
        raise SystemExit(f"struct {name} not found in header")
    return body


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
    # Unknown scalar type: assume pointer-width and flag it.
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

def _header_model_params_layout(header: str) -> str:
    """Which llama_model_params layout a header carries: 'v1' or 'v2'.

    Keyed on the field that defines the split rather than on a version number,
    so an arbitrary --ref or --header is classified by what it actually says."""
    body = _strip_comments(_extract_struct_body(header, "llama_model_params"))
    return "v2" if re.search(r"\bload_mode\b", body) else "v1"


def _header_context_params_layout(header: str) -> str:
    """Which llama_context_params layout a header carries: 'v1' or 'v2'.

    Independent axis from the model_params split above (see _structs' module
    docstring) - keyed on n_outputs_max_per_seq, the field upstream inserted
    directly before n_threads sometime between lemonade b1307 and b10360."""
    body = _strip_comments(_extract_struct_body(header, "llama_context_params"))
    return "v2" if re.search(r"\bn_outputs_max_per_seq\b", body) else "v1"


def _ensure_localm_importable() -> None:
    """Put the repo root on sys.path so ``localm`` imports from THIS checkout."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _localm_layout(struct_name: str, layout: str):
    """name -> (offset, size) for localm's ctypes struct of the C *struct_name*."""
    import ctypes

    _ensure_localm_importable()
    from localm.inference.backends.llamacpp import _structs as S

    cls = {
        "llama_model_params": (S.LlamaModelParamsV2 if layout == "v2"
                               else S.LlamaModelParamsV1),
        "llama_context_params": (S.LlamaContextParamsV2 if layout == "v2"
                                 else S.LlamaContextParamsV1),
        "llama_batch": S.LlamaBatch,
    }[struct_name]
    out = {}
    for fname, ftype in cls._fields_:
        out[fname] = (getattr(cls, fname).offset, ctypes.sizeof(ftype))
    return out, ctypes.sizeof(cls)


# --------------------------------------------------------------------------- #
#  Enum domain (WHICH VALUES are legal, as opposed to WHERE the field sits)
# --------------------------------------------------------------------------- #

def _parse_enum_members(body: str):
    """({member: value}, {members whose value could not be read}) for an enum body.

    C enumerators may omit ``= N`` and take previous+1. An unreadable value
    therefore poisons every implicit member after it, so the running counter is
    dropped rather than guessed - a guessed value would be indistinguishable
    from a read one and could manufacture a false FAIL or a false ok.
    """
    members = {}
    unreadable = set()
    nxt = 0
    for raw in _strip_comments(body).split(","):
        item = raw.strip()
        if not item:
            continue
        name, eq, val = item.partition("=")
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            print(f"  ! unparseable enumerator, skipping: {item!r}", file=sys.stderr)
            unreadable.add(name)
            nxt = None
            continue
        if eq:
            try:
                nxt = int(val.strip(), 0)
            except ValueError:
                print(f"  ! non-literal enumerator value, cannot verify: {item!r}",
                      file=sys.stderr)
                unreadable.add(name)
                nxt = None
                continue
        elif nxt is None:
            unreadable.add(name)
            continue
        members[name] = nxt
        nxt += 1
    return members, unreadable


def _localm_enum_binding(binding: _EnumBinding):
    """({member suffix: value}, {suffixes the module names but this registry does not}).

    Values are read live out of the localm module, never restated in the
    registry, so this verifier compares against the same constants the runtime
    uses rather than against a copy that can drift from them.
    """
    _ensure_localm_importable()
    mod = importlib.import_module(binding.module)
    bound = {}
    for suffix in binding.members:
        val = getattr(mod, binding.prefix + suffix, None)
        # bool is an int subclass and never an enumerator; None means the named
        # constant is gone. Neither goes into `bound`.
        if isinstance(val, bool) or not isinstance(val, int):
            continue
        bound[suffix] = val
    missing = [s for s in binding.members if s not in bound]

    # Everything else under the same prefix.
    others = {}
    for attr in dir(mod):
        if not attr.startswith(binding.prefix):
            continue
        suffix = attr[len(binding.prefix):]
        if suffix in bound or suffix in binding.members:
            continue
        val = getattr(mod, attr)
        if isinstance(val, bool) or not isinstance(val, int):
            continue
        others[suffix] = val
    return bound, missing, others


def _check_enum(binding: _EnumBinding, header: str, additive: list) -> int:
    """Diff one bound enum's DOMAIN against *header*; return the problem count.

    Additive findings are APPENDED to *additive* instead of counted: a new
    upstream enumerator is safe on its own. A changed value for a name localm
    already binds IS counted - localm would be passing or accepting a number
    that no longer carries the meaning it binds.
    """
    gate_struct, gate_field = binding.gate
    struct_body = _extract_block_body(header, "struct", gate_struct)
    gated_in = struct_body is not None and re.search(
        r"\b" + re.escape(gate_field) + r"\b", _strip_comments(struct_body))
    body = _extract_block_body(header, "enum", binding.c_enum)

    if not gated_in:
        print(f"  skip  {binding.c_enum}: this header has no "
              f"{gate_struct}.{gate_field}, so it predates the enum")
        return 0
    if body is None:
        print(f"  FAIL  enum {binding.c_enum} not found, yet {gate_struct}."
              f"{gate_field} IS present - renamed upstream, or this parse broke. "
              "Either way the domain is UNVERIFIED, which is not the same as ok.")
        return 1

    upstream, unreadable = _parse_enum_members(body)
    if not upstream and not unreadable:
        print(f"  FAIL  enum {binding.c_enum} parsed to ZERO members - an empty "
              "parse reads exactly like a clean check, so it fails instead.")
        return 1
    bound, missing, others = _localm_enum_binding(binding)
    problems = 0
    if missing:
        print(f"  FAIL  {binding.module} no longer defines "
              f"{', '.join(binding.prefix + s for s in missing)} - the constants "
              "this registry names have moved or been renamed, so those members "
              "were NOT compared against anything.")
        problems += len(missing)
    if not bound:
        return problems or 1

    bound_c_names = set()
    for suffix, lm_val in sorted(bound.items(), key=lambda kv: kv[1]):
        cname = binding.c_prefix + suffix
        bound_c_names.add(cname)
        if cname in unreadable:
            print(f"  FAIL  {cname}: localm binds {lm_val}, upstream's value could "
                  "not be read from the header (see the ! line above)")
            problems += 1
        elif cname not in upstream:
            print(f"  NOTE  {cname}: localm binds {lm_val}, absent from this header "
                  "(it predates the member, or upstream removed it)")
        elif upstream[cname] != lm_val:
            print(f"  FAIL  {cname}: localm binds {lm_val}, upstream says "
                  f"{upstream[cname]} - a value localm passes or accepts has "
                  "changed meaning")
            problems += 1
        else:
            print(f"  ok    {cname:34s} = {lm_val}")

    # A constant localm defines under the same prefix that upstream also defines
    # as a member of this enum is a binding this registry omits.
    for suffix, lm_val in sorted(others.items(), key=lambda kv: kv[1]):
        cname = binding.c_prefix + suffix
        if cname in upstream or cname in unreadable:
            print(f"  FAIL  {binding.prefix}{suffix} = {lm_val} is a real "
                  f"{binding.c_enum} member upstream but is missing from this "
                  "script's _ENUM_BINDINGS, so its value went unchecked.")
            problems += 1

    for cname, up_val in sorted(upstream.items(), key=lambda kv: kv[1]):
        if cname not in bound_c_names:
            print(f"  NEW   {cname} = {up_val} (upstream defines it, localm does not "
                  "bind it)")
            additive.append(f"{binding.c_enum}.{cname} = {up_val}")
    return problems


# --------------------------------------------------------------------------- #
#  Diff
# --------------------------------------------------------------------------- #

def _check(struct_name: str, header: str, model_layout: str, context_layout: str) -> int:
    body = _extract_struct_body(header, struct_name)
    upstream, up_size = _layout(_parse_fields(body))
    # _localm_layout ignores the layout argument for llama_batch.
    layout = context_layout if struct_name == "llama_context_params" else model_layout
    localm, lm_size = _localm_layout(struct_name, layout)

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
    ap.add_argument("--ref", default=None,
                    help="upstream tag/commit/branch, or 'latest' to resolve the "
                         "newest release. Default: check BOTH pinned refs, "
                         f"{LLAMA_ABI_REFS['v1']} (pre-reorder) and "
                         f"{LLAMA_ABI_REFS['v2']} (post-reorder), since localm "
                         "binds both llama_model_params layouts.")
    ap.add_argument("--header", default=None,
                    help="path to a local llama.h instead of fetching")
    args = ap.parse_args()

    headers = []   # (label, header text)
    if args.header:
        headers.append((args.header,
                        Path(args.header).read_text(encoding="utf-8", errors="replace")))
    elif args.ref:
        ref = _resolve_ref(args.ref)
        headers.append((f"{_REPO}@{ref}", _fetch_header(ref)))
    else:
        for want, ref in LLAMA_ABI_REFS.items():
            headers.append((f"{_REPO}@{ref} (expect {want})", _fetch_header(ref)))

    total = 0
    seen_layouts = set()
    seen_context_layouts = set()
    additive = []
    for label, header in headers:
        layout = _header_model_params_layout(header)
        context_layout = _header_context_params_layout(header)
        seen_layouts.add(layout)
        seen_context_layouts.add(context_layout)
        print(f"\n############ Checking localm structs against {label} "
              f"[llama_model_params layout: {layout}, "
              f"llama_context_params layout: {context_layout}] ############")
        for s in _STRUCTS:
            total += _check(s, header, layout, context_layout)
        print("\n=== enum domains ===")
        for binding in _ENUM_BINDINGS:
            total += _check_enum(binding, header, additive)

    # Both layout axes must have been exercised.
    if not args.header and not args.ref and seen_layouts != {"v1", "v2"}:
        print(f"\nFAIL: expected to check both model_params layouts, only saw "
              f"{sorted(seen_layouts)} - the pinned refs in LLAMA_ABI_REFS no "
              "longer straddle the model_params reorder.")
        total += 1
    if not args.header and not args.ref and seen_context_layouts != {"v1", "v2"}:
        print(f"\nFAIL: expected to check both context_params layouts, only saw "
              f"{sorted(seen_context_layouts)} - the pinned refs in LLAMA_ABI_REFS "
              "no longer straddle the context_params reorder.")
        total += 1

    if additive:
        print(f"\nNEW UPSTREAM ENUM MEMBER(S) localm does not bind ({len(additive)}):")
        for item in sorted(set(additive)):
            print(f"  - {item}")
        print("  Additive on its own, so this does NOT fail the check. It turns "
              "dangerous when upstream makes the new member the DEFAULT, and a "
              "header cannot show that:\n"
              "  b10373 added LLAMA_LOAD_MODE_AUTO = -1 AND made "
              "llama_model_default_params() return it, so every build from then on "
              "was refused.\n"
              "  So on seeing this: bind the member, then re-probe a real build's "
              "llama_*_default_params() rather than reading the header alone.")

    print()
    if total:
        print(f"ABI CHECK FAILED: {total} mismatch(es). A mid-struct reorder/insert "
              "corrupts memory, and a changed enum value silently redefines what a "
              "field means - update _structs.py (and _abi anchors) to match, then "
              "re-probe a real build. See issues/abi-verification-design-2026-06-20.md.")
        return 1
    print(f"ABI check passed: localm's named fields match upstream offsets "
          f"for layout(s) {sorted(seen_layouts)}, and every enum member localm "
          f"binds still has its upstream value.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
