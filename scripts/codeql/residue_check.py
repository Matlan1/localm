#!/usr/bin/env python
"""Residue checker: classify CodeQL alerts by EVIDENCE, not by count.

Why this exists
---------------
Counting open alerts is broken in BOTH directions on this codebase:

  * It FAILS correct work. CodeQL does not model this repo's confinement
    patterns as barriers - proven from the SARIF: taint was traced straight
    through a re.sub() rewrite, through resolve() + is_relative_to() WITH a
    raise, and through a parent-equality re-check WITH a raise. So a correct
    fix routinely leaves its alert open.
  * It FAILS on churn. A fix that shifts line numbers makes CodeQL retire an
    alert id and raise a new one at the new line. Measured on this repo:
    a raw (rule, file, line) diff reported 79 gone / 85 new between two
    analyses, while the fingerprint diff reported 10 gone / 16 new. About 69
    "changes" were pure line-shift noise.
  * It can read fixes as REGRESSIONS. A shared helper introduced to centralise
    confinement becomes a new sink itself, so a good fix ADDS alerts.

So the criterion is per-alert: **is the sink confined and the flow broken?**
This tool mechanises the parts of that which can be mechanised and refuses to
guess the rest.

Identity
--------
Alerts are keyed on (ruleId, file, primaryLocationLineHash) - GitHub's own
fingerprint, which is what lets an alert survive a line move. Never on line
numbers.

Usage
-----
  python residue_check.py --baseline <sarif> --current <sarif> [--barriers b.json]

`barriers.json` is OPTIONAL and is how condition 3 ("name the barrier") is
supplied. It is a list of {"name":..., "file":..., "function":...}. The tool
reports whether a surviving alert's flow passes THROUGH a declared barrier.
It never invents one: an undeclared barrier is reported as NOT VERIFIED.

What makes a barrier verifiable at all
--------------------------------------
Three shapes CANNOT satisfy condition 2, and none of them means the guard is
absent or wrong. Condition 2 asks whether a flow passes THROUGH a barrier, so it
is answerable only when there is a traversal to observe:

  1. INLINE GUARD - the barrier is the sink's own function. No traversal exists.
  2. CHOKE POINT - the barrier cannot barrier itself (the shared helper that
     centralises confinement becomes a sink in its own right).
  3. NON-TRANSFORMING - the barrier is EXTRACTED but returns no value. It is
     called for its side effect (raising), so the tainted value goes INTO it
     while the value used downstream is the ORIGINAL. Nothing flows THROUGH it.

So the design rule is sharper than "extract the guard", which is necessary but
NOT sufficient: **the sink must consume the barrier's RETURN VALUE.**
`confined_under(base, name) -> Path` is verifiable. `reject(raw) -> None` is not,
however correct it is. Transform, do not merely validate - the same root as the
finding that CodeQL walks through validate-and-raise, arriving from the barrier
side instead of the query side.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["runs"][0]["results"]


def fingerprint(r):
    """Line-move-immune identity. Falls back to the line only if absent."""
    loc = r["locations"][0]["physicalLocation"]
    fp = r.get("partialFingerprints", {}).get("primaryLocationLineHash")
    return (r["ruleId"], loc["artifactLocation"]["uri"],
            fp or f"line:{loc['region']['startLine']}")


def where(r):
    loc = r["locations"][0]["physicalLocation"]
    return f"{loc['artifactLocation']['uri']}:{loc['region']['startLine']}"


def flow_steps(r):
    """Every (file, line) the taint is traced through, per codeFlow."""
    out = []
    for cf in r.get("codeFlows", []):
        steps = []
        for loc in cf["threadFlows"][0]["locations"]:
            p = loc["location"]["physicalLocation"]
            steps.append((p["artifactLocation"]["uri"], p["region"]["startLine"]))
        out.append(steps)
    return out


def sources(r):
    return sorted({s[0] + ":" + str(s[1]) for f in flow_steps(r) for s in f[:1]})


def passes_barrier(r, barriers, traversed_barriers=frozenset()):
    """True only if EVERY codeFlow passes through a declared barrier file.

    Every flow, not any: an alert with four sources where one flow is barriered
    and three are not is NOT confined. That asymmetry is the whole point - it is
    the mistake that booked a fix as closing alerts it could not close.
    """
    if not barriers:
        return None                      # nothing declared: cannot verify
    flows = flow_steps(r)
    if not flows:
        return None
    loc = r["locations"][0]["physicalLocation"]
    sink_file = loc["artifactLocation"]["uri"].replace("\\", "/")
    sink_line = loc["region"]["startLine"]

    def hit(step):
        f, ln = step[0].replace("\\", "/"), step[1]
        for bf, lo, hi, _name in barriers:
            if f == bf and lo <= ln <= hi:
                # The barrier must be TRAVERSED on the way to the sink, not BE
                # the sink. Without this, an alert whose sink sits inside a
                # declared barrier passes vacuously.
                if f == sink_file and lo <= sink_line <= hi:
                    continue
                return True
        return False

    if all(any(hit(s) for s in flow) for flow in flows):
        return True

    # Condition 2 is NOT APPLICABLE (structural), not merely unmet, when the
    # declared barrier CONTAINS the sink - there is no traversal to observe.
    # Two distinct structural cases, and they are distinguished by EVIDENCE
    # rather than by a label: a genuinely EXTRACTED barrier is one that other
    # alerts' flows actually traverse. If nothing traverses it, the "barrier" is
    # an inline guard in the handler itself.
    for bf, lo, hi, name in barriers:
        if bf == sink_file and lo <= sink_line <= hi:
            return ("NA_CHOKEPOINT" if name in traversed_barriers
                    else "NA_INLINE")
    return False


def _returns_a_value(fn_node) -> bool:
    """True if the function returns a VALUE somewhere (not just bare `return`).

    Nested functions are excluded: a returning closure does not make its
    enclosing guard transforming."""
    import ast
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Return) and node.value is not None:
            owner = None
            for cand in ast.walk(fn_node):
                if isinstance(cand, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and cand is not fn_node and node in list(ast.walk(cand)):
                    owner = cand
                    break
            if owner is None:
                return True
    return False


def _unsafe_declared_path(rel: str) -> str:
    """Reason a declared barrier path is unusable, or "" if it is fine.

    LEXICAL ONLY, and it runs BEFORE any filesystem call. That ordering is the
    point, not a style choice: `Path(...).resolve()` performs real I/O, so
    resolving an unvalidated component is itself the hazard. A declared file of
    `\\\\host\\share` makes resolve() dial SMB - measured elsewhere in this repo
    at 271 seconds against a non-routable host before it gave up, and on Windows
    an auto-authenticated dial leaks the host credential. Validate the STRING,
    then touch the disk."""
    s = (rel or "").strip()
    if not s:
        return "is empty"
    n = s.replace("\\", "/")
    if n.startswith("//"):
        return "is a UNC path"
    if n.startswith("/"):
        return "is absolute"
    if len(n) >= 2 and n[1] == ":":
        return "is drive-qualified"
    if any(part == ".." for part in n.split("/")):
        return "contains a '..' component"
    return ""


def barrier_spans(declared, repo_root="."):
    """Resolve each declared barrier to (file, first_line, last_line, name) by AST.

    Matching on the FILE alone is vacuous, and that was a real defect in this
    tool: a declared barrier file is frequently also a SINK file, so every alert
    located in it trivially "passes through a barrier" while nothing is verified.
    Measured on PR 845: 10 of 14 apparent passes were vacuous for exactly that
    reason. The barrier is the declared FUNCTION, not its file."""
    import ast
    spans = []
    root = Path(repo_root).resolve()
    for b in declared:
        # b["file"] is a path component read from the barriers JSON, i.e. from a
        # DATA FILE rather than from this script's argv. A barriers.json
        # containing {"file": "../../../etc/passwd"} would otherwise read outside
        # the repo. Confine it, and SAY SO when it escapes rather than silently
        # skipping - a barrier that was declared but never resolved must never
        # look the same as one that verified nothing.
        why = _unsafe_declared_path(b["file"])
        if why:
            print(f"  WARNING: barrier {b['name']!r} declares file {b['file']!r} "
                  f"which {why} - REFUSED, and it will not verify anything")
            continue
        try:
            path = (root / b["file"]).resolve()
            if root != path and root not in path.parents:
                raise ValueError("resolves outside the repo root")
        except (OSError, ValueError) as e:
            print(f"  WARNING: barrier {b['name']!r} declares file {b['file']!r} "
                  f"which {e if isinstance(e, ValueError) else 'cannot be resolved'}"
                  f" - REFUSED, and it will not verify anything")
            continue
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except (OSError, SyntaxError):
            print(f"  WARNING: cannot parse {b['file']} - barrier "
                  f"{b['name']!r} NOT resolved and will not verify anything")
            continue
        found = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == b["function"]:
                spans.append((b["file"].replace("\\", "/"), node.lineno,
                              node.end_lineno or node.lineno, b["name"]))
                found = True
                # A barrier that returns NO VALUE is called for its side effect
                # (raising). The tainted value goes INTO it but the value used
                # downstream is the ORIGINAL, so there is no data flow THROUGH it
                # for the analyser to observe. Extraction alone does NOT make a
                # barrier verifiable - the sink must consume its RETURN VALUE.
                # Flagged here, at declaration time, because such a barrier can
                # never appear in any flow and so can never verify ANY alert.
                if not _returns_a_value(node):
                    print(f"  NOTE: barrier {b['name']!r} "
                          f"({b['file']}::{b['function']}) returns NO VALUE. It is "
                          f"extracted but NON-TRANSFORMING, so no flow can pass "
                          f"THROUGH it and it can never satisfy condition 2 for any "
                          f"alert. That is a property of its SHAPE, not evidence "
                          f"the guard is absent or wrong.")
        if not found:
            print(f"  WARNING: function {b['function']!r} not found in "
                  f"{b['file']} - barrier {b['name']!r} NOT resolved")
    return spans


def traversed(results, barriers):
    """Declared barriers that some alert's flow actually PASSES THROUGH.

    This is what separates a genuinely EXTRACTED barrier from an inline guard,
    using evidence instead of the declaration's wording: if no flow anywhere
    traverses it, it is not a barrier the analyser can ever observe."""
    seen = set()
    for r in results:
        loc = r["locations"][0]["physicalLocation"]
        sf = loc["artifactLocation"]["uri"].replace("\\", "/")
        sl = loc["region"]["startLine"]
        for flow in flow_steps(r):
            for f, ln in flow:
                f = f.replace("\\", "/")
                for bf, lo, hi, name in barriers:
                    if f == bf and lo <= ln <= hi:
                        if not (bf == sf and lo <= sl <= hi):
                            seen.add(name)
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--barriers")
    ap.add_argument("--repo-root", dest="repo_root", default=".")
    ap.add_argument("--json", help="write the machine-readable residue here")
    args = ap.parse_args()

    base = {fingerprint(r): r for r in load(args.baseline)}
    cur = {fingerprint(r): r for r in load(args.current)}
    declared = json.load(open(args.barriers, encoding="utf-8")) if args.barriers else []
    barriers = barrier_spans(declared, args.repo_root) if declared else []

    tb = traversed(list(cur.values()), barriers)
    resolved = sorted(base.keys() - cur.keys())
    survived = sorted(base.keys() & cur.keys())
    added = sorted(cur.keys() - base.keys())

    # FILE MOVES. The fingerprint key includes the FILE, so code moved between
    # files (e.g. a refactor collapsing one module into another) makes the same
    # alert read as RESOLVED in the old file AND ADDED in the new one - a
    # spurious pair in both directions, inflating both counts and reading as
    # churn or regression. Detect it by re-matching on (ruleId, lineHash) ALONE
    # across the two sets: identical rule and identical line CONTENT in a
    # different file is a move, not a fix plus a new finding.
    def loose(k):
        return (k[0], k[2])
    base_loose = {loose(k): k for k in resolved}
    cur_loose = {loose(k): k for k in added}
    moved = sorted(set(base_loose) & set(cur_loose))
    if moved:
        resolved = [k for k in resolved if loose(k) not in dict(zip(moved, moved))]
        added = [k for k in added if loose(k) not in dict(zip(moved, moved))]

    print(f"baseline results : {len(base)}")
    print(f"current  results : {len(cur)}")
    print(f"  RESOLVED       : {len(resolved)}   (gone by fingerprint, not by line)")
    print(f"  MOVED          : {len(moved)}   (same rule + same line CONTENT, different file)")
    print(f"  SURVIVED       : {len(survived)}")
    print(f"  ADDED          : {len(added)}")
    print()

    if moved:
        print("== MOVED (a refactor relocated the code; NOT resolved, NOT new) ==")
        for lk in moved:
            print(f"  {lk[0]:26s} {where(base[base_loose[lk]])}"
                  f"  ->  {where(cur[cur_loose[lk]])}")
        print("  These are excluded from RESOLVED and ADDED. Counting them in either")
        print("  would report a refactor as a fix and as a regression simultaneously.")
        print()

    if resolved:
        print("== RESOLVED (these are genuinely gone) ==")
        for k in resolved:
            print(f"  {k[0]:28s} {where(base[k])}")
        print()

    if added:
        print("== ADDED (each needs an explanation, NOT assumed to be regression) ==")
        by_file = defaultdict(list)
        for k in added:
            by_file[k[1]].append(where(cur[k]))
        for f, locs in sorted(by_file.items()):
            print(f"  {f}")
            for loc in sorted(locs):
                print(f"      {loc}")
        print("  NOTE: a new alert AT A CHOKE POINT a fix created is EXPECTED and is")
        print("  evidence the centralisation worked. It must be NAMED in the PR body")
        print("  with that reasoning, or a later auditor reads it as a new vulnerability.")
        print()

    verified = notverified = 0
    if survived:
        print("== SURVIVED: barrier verification (condition 2) ==")
        unadjudicable = 0
        for k in survived:
            r = cur[k]
            ok = passes_barrier(r, barriers, tb)
            if ok is True:
                verified += 1
                continue
            if ok in ("NA_INLINE", "NA_CHOKEPOINT"):
                unadjudicable += 1
                tag = ("N/A inline guard" if ok == "NA_INLINE"
                       else "N/A choke point")
            else:
                notverified += 1
                tag = "NOT VERIFIED"
            print(f"  [{tag:16s}] {k[0]:24s} {where(r)}  sources={len(sources(r))}")
        print()
        print(f"  BARRIERED     (every flow through a declared barrier): {verified}")
        print(f"  NOT APPLICABLE (structural: no traversal to observe) : {unadjudicable}")
        print(f"  NOT VERIFIED  (no declared barrier on the flow)      : {notverified}")
        if unadjudicable:
            print()
            print("  NOT APPLICABLE is a STRUCTURAL result, NOT a failed fix, and NOT")
            print("  a pass either. Condition 2 asks whether a flow passes THROUGH a")
            print("  barrier; where the barrier CONTAINS the sink there is no traversal")
            print("  to observe, so the question cannot be asked at all.")
            print("    N/A inline guard  - the declared barrier is the handler itself")
            print("                        (nothing anywhere traverses it), so the guard")
            print("                        is in-line before the sink. Legitimate design.")
            print("    N/A choke point   - the declared barrier IS traversed by other")
            print("                        flows, so it is a real extracted barrier; this")
            print("                        alert is the helper becoming a sink itself.")
            print("  Both route to conditions 1 and 3 plus an INDEPENDENT human read of")
            print("  the diff, which cannot be the author's. Neither verdict asserts a")
            print("  working guard: N/A only means the tool cannot see one.")
        print()

    print("== VERDICT ==")
    print("  An alert counts FIXED-BUT-NOT-CLOSED only with ALL THREE of:")
    print("    1. a demonstrated fix (a test that goes red pre-fix), and")
    print("    2. every codeFlow shown to pass through the barrier (above), and")
    print("    3. the barrier NAMED with why CodeQL does not model it.")
    print("  This tool can only mechanise (2). Conditions 1 and 3 are human")
    print("  judgements and are NOT satisfied by running this.")
    if not barriers:
        print()
        print("  NO BARRIERS DECLARED: condition 2 is unverified for every surviving")
        print("  alert. Supply --barriers to check it. Absence of a declaration is")
        print("  NOT evidence of absence of a barrier, and must not be read as one.")

    if args.json:
        out = {
            "resolved": [{"rule": k[0], "at": where(base[k])} for k in resolved],
            "added": [{"rule": k[0], "at": where(cur[k])} for k in added],
            "survived_not_verified": [
                {"rule": k[0], "at": where(cur[k]), "sources": sources(cur[k])}
                for k in survived if passes_barrier(cur[k], barriers, tb) is not True],
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()
