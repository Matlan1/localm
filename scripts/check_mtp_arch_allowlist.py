# SPDX-License-Identifier: AGPL-3.0-or-later
"""Keep MTP detection a CAPABILITY test rather than a METADATA test.

A GGUF carrying ``<arch>.nextn_predict_layers`` is DECLARING multi-token
prediction heads. It is not promising that the runtime can BUILD an MTP draft
graph for that architecture, and those two diverge: glm4moe (GLM-4.5 / 4.5-Air /
4.6) ships the key and the NextN tensors while upstream's ``build_arch_graph``
ignores the MTP graph type and returns the ordinary decoder. Asking for an MTP
context there yields a SECOND FULL DECODER with its own VRAM-pinned KV cache,
created after the caller's memory preflight and charged to nobody.

Detection therefore consults ``MTP_GRAPH_ARCHITECTURES`` in
``localm/inference/backends/llamacpp/_api.py``: the architectures whose
llama.cpp model class declares a nested ``struct graph_mtp`` AND dispatches to
it from ``build_arch_graph`` on ``LLM_GRAPH_TYPE_DECODER_MTP``. That set is a
property of ONE upstream release, so it goes stale the moment the runtime pin
moves - at b10375 twelve of one hundred and forty-one model classes qualify and
glm4moe is not one of them, while on upstream's later default branch it is.

WHAT THIS GATE CHECKS, all offline
----------------------------------
  1. ``MTP_ARCH_SOURCE_TAG`` still equals ``localm.setup_llama._PINNED_TAG``. A
     pin bump without a re-derivation is how this set rots silently.
  2. The detector still reads the allowlist. Deleting that one condition
     restores the original defect and nothing else in the tree would notice.
  3. Every site that asks for an MTP context is gated by the detector. A second
     call site added later is the other way the defect returns.
  4. The allowlist is non-empty and every entry looks like a GGUF arch string.

The re-derivation itself needs the network and is deliberately NOT part of the
gate: run ``--refresh`` to fetch upstream at the pinned tag, parse the model
class table, and print the set the allowlist should hold.

USAGE
-----
    python scripts/check_mtp_arch_allowlist.py            # full report
    python scripts/check_mtp_arch_allowlist.py --gate     # exit 1 on a failure
    python scripts/check_mtp_arch_allowlist.py --refresh  # re-derive from upstream

``tests/test_mtp_arch_allowlist.py`` pins the gate and, more importantly, proves
this file FIRES: a check that has never been red proves nothing.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent

API_PATH = REPO / "localm" / "inference" / "backends" / "llamacpp" / "_api.py"
SETUP_PATH = REPO / "localm" / "setup_llama.py"

# The detector every MTP decision must go through, and the constant it must read.
DETECTOR = "llama_model_mtp_support"
ALLOWLIST = "MTP_GRAPH_ARCHITECTURES"
TAG_CONST = "MTP_ARCH_SOURCE_TAG"

# Setting this on a context params struct is what asks llama.cpp for an MTP
# context, so every assignment of it has to sit downstream of the detector.
MTP_CTX_CONST = "LLAMA_CONTEXT_TYPE_MTP"

UPSTREAM_RAW = "https://raw.githubusercontent.com/ggml-org/llama.cpp/%s/%s"

_ARCH_RE = re.compile(r"^[a-z][a-z0-9._-]*$")


def _literal(node):
    """The value of a str constant, or a set/frozenset of str constants.

    Deliberately narrow: this gate reads two module-level constants out of a
    source file it must not import, because importing ``_api`` loads the native
    library. Anything else in those files returns None and is skipped.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "frozenset" and len(node.args) == 1):
        return _literal(node.args[0])
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        items = [e.value for e in node.elts
                 if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if len(items) == len(node.elts):
            return set(items)
    return None


def _module_constants(path: Path) -> dict:
    """Read the module-level constants this gate needs, without importing."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = _literal(node.value)
        if value is not None:
            out[target.id] = value
    return out


def _detector_reads_allowlist() -> bool:
    """True when the detector function body still mentions the allowlist."""
    tree = ast.parse(API_PATH.read_text(encoding="utf-8"), filename=str(API_PATH))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == DETECTOR:
            return any(isinstance(n, ast.Name) and n.id == ALLOWLIST
                       for n in ast.walk(node))
    return False


def _ungated_mtp_context_sites(root: Path) -> list:
    """Functions that set an MTP context type without consulting the detector.

    Scoped to the package that owns the native binding, so a test may name the
    constant freely.
    """
    findings = []
    pkg = root / "localm"
    for path in sorted(pkg.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            findings.append((str(path.relative_to(root)), "<unparseable>", 0))
            continue
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names = {n.id for n in ast.walk(func) if isinstance(n, ast.Name)}
            attrs = {n.attr for n in ast.walk(func) if isinstance(n, ast.Attribute)}
            if MTP_CTX_CONST not in names:
                continue
            if DETECTOR in names or DETECTOR in attrs:
                continue
            findings.append((str(path.relative_to(root)), func.name, func.lineno))
    return findings


def _pinned_tag() -> Optional[str]:
    value = _module_constants(SETUP_PATH).get("_PINNED_TAG")
    return value if isinstance(value, str) else None


def _fetch(tag: str, path: str) -> str:
    """Fetch one upstream file, refusing anything that is not a real 200 body.

    A grep over an error page returns the same zero as a grep over a real file
    with no match, so the fetch is checked before the parse.
    """
    from localm.http_ssl import verified_urlopen

    url = UPSTREAM_RAW % (tag, path)
    with verified_urlopen(url, timeout=30) as resp:
        status = getattr(resp, "status", None) or resp.getcode()
        body = resp.read().decode("utf-8", "replace")
    if status != 200:
        raise RuntimeError("%s -> HTTP %s" % (url, status))
    if len(body) < 1024:
        raise RuntimeError("%s -> implausibly short body (%d bytes)" % (url, len(body)))
    return body


def refresh(tag: str) -> set:
    """Re-derive the MTP-capable architecture set from upstream at *tag*.

    Three files, joined on the C++ class name: ``models.h`` says which classes
    declare a nested ``struct graph_mtp``, ``llama-model.cpp`` maps each
    ``LLM_ARCH_*`` enum to the class it instantiates, and ``llama-arch.cpp``
    maps the enum to the ``general.architecture`` string a GGUF carries.
    """
    models_h = _fetch(tag, "src/models/models.h")
    model_cpp = _fetch(tag, "src/llama-model.cpp")
    arch_cpp = _fetch(tag, "src/llama-arch.cpp")

    owners, current = set(), None
    for line in models_h.splitlines():
        header = re.match(r"^struct\s+(llama_model_[A-Za-z0-9_]+)\s*[:{]", line)
        if header:
            current = header.group(1)
        if current and re.search(r"\bstruct\s+graph_mtp\b", line):
            owners.add(current)
    if not owners:
        raise RuntimeError("no graph_mtp declarations found - upstream layout changed")

    class_to_enums = {}
    pending = []
    for line in model_cpp.splitlines():
        case = re.search(r"case\s+(LLM_ARCH_[A-Z0-9_]+)\s*:", line)
        if case:
            pending.append(case.group(1))
            continue
        ret = re.search(r"return\s+new\s+(llama_model_[A-Za-z0-9_]+)\s*\(", line)
        if ret:
            class_to_enums.setdefault(ret.group(1), []).extend(pending)
            pending = []
    if not class_to_enums:
        raise RuntimeError("model factory table not found - upstream layout changed")

    enum_to_name = dict(re.findall(
        r'\{\s*(LLM_ARCH_[A-Z0-9_]+)\s*,\s*"([^"]+)"\s*\}', arch_cpp))
    if not enum_to_name:
        raise RuntimeError("LLM_ARCH_NAMES table not found - upstream layout changed")

    derived = set()
    for cls in sorted(owners):
        for enum in class_to_enums.get(cls, []):
            name = enum_to_name.get(enum)
            if name:
                derived.add(name)
    return derived


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gate", action="store_true", help="exit 1 on a failure")
    ap.add_argument("--refresh", action="store_true",
                    help="re-derive the allowlist from upstream (needs network)")
    args = ap.parse_args(argv)

    consts = _module_constants(API_PATH)
    allowlist = set(consts.get(ALLOWLIST) or ())
    source_tag = consts.get(TAG_CONST)
    pinned = _pinned_tag()

    failures = []

    print("MTP capability gate")
    print("  %-19s : %d" % ("allowlist entries", len(allowlist)))
    print("  %-19s : %s" % (TAG_CONST, source_tag))
    print("  %-19s : %s" % ("_PINNED_TAG", pinned))
    print()

    if not allowlist:
        failures.append(
            "%s is empty, so MTP can never engage. If that is intended, say so at "
            "the constant rather than leaving an empty set." % ALLOWLIST)
    bad = sorted(a for a in allowlist if not _ARCH_RE.match(a))
    if bad:
        failures.append(
            "not GGUF architecture strings (those are lower-case, as written in "
            "general.architecture): %s" % ", ".join(bad))

    if not isinstance(source_tag, str) or pinned is None:
        failures.append(
            "could not read %s from %s and/or _PINNED_TAG from %s"
            % (TAG_CONST, API_PATH.name, SETUP_PATH.name))
    elif source_tag != pinned:
        failures.append(
            "the runtime pin moved to %s while the MTP allowlist was derived from "
            "%s. Which architectures build an MTP graph is a property of one "
            "release, so re-derive with --refresh, update %s and %s, then re-run."
            % (pinned, source_tag, ALLOWLIST, TAG_CONST))

    if not _detector_reads_allowlist():
        failures.append(
            "%s() no longer reads %s. Without it, detection is back to trusting "
            "GGUF metadata alone and engages MTP on architectures that build no "
            "MTP graph." % (DETECTOR, ALLOWLIST))

    ungated = _ungated_mtp_context_sites(REPO)
    if ungated:
        for rel, func, line in ungated:
            print("  UNGATED %s:%d in %s()" % (rel, line, func))
        print()
        failures.append(
            "%d function(s) request an MTP context without consulting %s()."
            % (len(ungated), DETECTOR))

    if args.refresh:
        tag = pinned or source_tag
        print("re-deriving from upstream at %s ..." % tag)
        try:
            derived = refresh(tag)
        except Exception as exc:
            print("  refresh FAILED: %s: %s" % (type(exc).__name__, exc))
            return 2
        print("  derived: %s" % ", ".join(sorted(derived)))
        missing = sorted(derived - allowlist)
        extra = sorted(allowlist - derived)
        if missing:
            print("  MISSING from %s: %s" % (ALLOWLIST, ", ".join(missing)))
        if extra:
            print("  EXTRA in %s:   %s" % (ALLOWLIST, ", ".join(extra)))
        if not missing and not extra:
            print("  allowlist matches upstream at %s" % tag)
        else:
            failures.append("the allowlist does not match upstream at %s" % tag)

    if failures:
        print("FAILED:")
        for item in failures:
            print("  - %s" % item)
        return 1 if args.gate else 0

    print("MTP capability gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
