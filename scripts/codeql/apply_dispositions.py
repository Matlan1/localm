#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dismiss adjudicated code-scanning alerts from `dispositions.json`.

    python apply_dispositions.py            # report only, changes nothing
    python apply_dispositions.py --apply    # post the dismissals

An alert is matched by (file, ENCLOSING FUNCTION), parsed with `ast` from the
COMMIT THE ALERT WAS RAISED AGAINST (`git show <sha>:<path>`), never from the
working tree. Line numbers and alert numbers both move whenever a fix edits a
sink line, so keying on either is how a dismissal silently detaches from the
thing it was about; a function name survives that, but only if it is read from
the revision the line number belongs to.

What is POSTED is the group's `short` text (GitHub caps a dismissal comment at
280 characters). The full reasoning stays in `justifications` in this repo,
under version control where it can be reviewed and corrected; the short text
names its group so an alert can be tied back to it.

THE IMPORTANT PROPERTY: an alert whose function is not in the table is NEVER
dismissed. It is listed as UNADJUDICATED and the exit code is non-zero. A tool
that closed whatever it did not recognise would be the exact failure this whole
directory exists to prevent - the point is that every closed alert had someone
look at it.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

REPO = "Matlan1/localm"
# GitHub's hard cap on dismissed_comment; over it the PATCH 422s.
COMMENT_LIMIT = 280
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def gh_json(path: str, method: str = "GET", fields: dict | None = None):
    cmd = ["gh", "api", "-X", method, path]
    for k, v in (fields or {}).items():
        cmd += ["-f", f"{k}={v}"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"gh api {method} {path} failed: {out.stderr.strip()}")
    return json.loads(out.stdout) if out.stdout.strip() else None


def open_alerts() -> list[dict]:
    """Every OPEN alert. Paginated by hand because `gh --paginate` emits one JSON
    array per page, which json.load cannot read as a single document."""
    alerts, page = [], 1
    while True:
        batch = gh_json(
            f"repos/{REPO}/code-scanning/alerts?state=open&per_page=100&page={page}")
        if not batch:
            break
        alerts += batch
        page += 1
    return alerts


_spans: dict[tuple[str, str], list] = {}


def _source_at(commit: str, relpath: str) -> str | None:
    """The file as it was in *commit*, or None if that object is not available."""
    out = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{relpath}"],
                         capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


def enclosing_function(commit: str, relpath: str, line: int) -> str:
    """Innermost def containing *line* IN THE ANALYSED COMMIT, or '<module>'.

    Read from the commit CodeQL actually analysed, never from the working tree.
    An alert's line number is only meaningful against the revision that produced
    it, and this repo is worked in worktrees that are routinely several commits
    off master, so resolving against whatever happens to be checked out silently
    attributes an alert to the wrong function and hands it the wrong
    justification. That is not hypothetical: it misfiled 8 of 125 on the first
    run of this script, all of them plausibly.

    Innermost def, because a nested helper's alert must not be attributed to the
    outer function that merely wraps it.
    """
    key = (commit, relpath)
    if key not in _spans:
        src = _source_at(commit, relpath)
        if src is None:
            _spans[key] = None          # distinct from "parsed, found nothing"
        else:
            try:
                tree = ast.parse(src)
            except SyntaxError:
                _spans[key] = None
            else:
                _spans[key] = sorted(
                    ((n.lineno, n.end_lineno, n.name) for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
                    key=lambda s: s[1] - s[0])
    spans = _spans[key]
    if spans is None:
        # Unavailable is NOT '<module>'. Collapsing the two would let a commit we
        # cannot read fall through to a table lookup that happens to match.
        return "<unavailable>"
    for start, end, name in spans:
        if start <= line <= end:
            return name
    return "<module>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually post the dismissals (default is a dry run)")
    ap.add_argument("--limit", type=int, default=0, metavar="N",
                    help="with --apply, post at most N dismissals. For validating "
                         "the path end to end on one alert before committing to "
                         "the whole set.")
    args = ap.parse_args()

    table = json.loads((HERE / "dispositions.json").read_text(encoding="utf-8"))
    dispositions = table["dispositions"]
    justifications = table["justifications"]
    reasons = table["reasons"]
    short = table["short"]

    missing = sorted(set(dispositions.values())
                     - (set(justifications) & set(reasons) & set(short)))
    if missing:
        print(f"BROKEN TABLE: no justification/reason/short for {missing}",
              file=sys.stderr)
        return 2
    # GitHub rejects a dismissed_comment over 280 characters with a 422 (measured,
    # not read off the docs). Checked for EVERY group up front rather than
    # discovering it per alert mid-run: a partial sweep that dismissed some alerts
    # and 422'd the rest is the state that is hardest to reason about afterwards.
    too_long = {g: len(t) for g, t in short.items() if len(t) > COMMENT_LIMIT}
    if too_long:
        print(f"BROKEN TABLE: short text over {COMMENT_LIMIT} chars: {too_long}",
              file=sys.stderr)
        return 2
    # GitHub accepts only these three. A typo would otherwise surface as a 422
    # per alert, mid-run, after some had already been dismissed.
    bad = {r for r in reasons.values()
           if r not in ("false positive", "won't fix", "used in tests")}
    if bad:
        print(f"BROKEN TABLE: not a GitHub dismissal reason: {sorted(bad)}",
              file=sys.stderr)
        return 2

    alerts = open_alerts()
    print(f"open alerts: {len(alerts)}")

    planned: list[tuple[dict, str, str]] = []
    unadjudicated: list[dict] = []
    for a in alerts:
        inst = a["most_recent_instance"]
        loc = inst["location"]
        fn = enclosing_function(inst["commit_sha"], loc["path"], loc["start_line"])
        key = f"{loc['path']}::{fn}"
        group = dispositions.get(key)
        if group is None:
            unadjudicated.append(a)
        else:
            planned.append((a, group, short[group], reasons[group]))

    by_group: dict[str, int] = {}
    for _a, g, _c, _r in planned:
        by_group[g] = by_group.get(g, 0) + 1
    print(f"\nadjudicated: {len(planned)}")
    for g, n in sorted(by_group.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {g}")

    if unadjudicated:
        print(f"\nUNADJUDICATED (left OPEN on purpose): {len(unadjudicated)}")
        for a in unadjudicated:
            inst = a["most_recent_instance"]
            loc = inst["location"]
            fn = enclosing_function(inst["commit_sha"], loc["path"], loc["start_line"])
            print(f"  #{a['number']:<5} {a['rule']['id']:<28} "
                  f"{loc['path']}:{loc['start_line']} ({fn})")

    if not args.apply:
        print("\nDRY RUN. Re-run with --apply to post these dismissals.")
        return 1 if unadjudicated else 0

    if args.limit:
        # Say what was held back. A capped run that printed only its successes
        # would read as a completed sweep.
        print(f"\n--limit {args.limit}: posting {min(args.limit, len(planned))} "
              f"of {len(planned)}; the rest are LEFT OPEN.")
        planned = planned[:args.limit]

    failures = 0
    for a, group, comment, reason in planned:
        try:
            gh_json(f"repos/{REPO}/code-scanning/alerts/{a['number']}",
                    method="PATCH",
                    fields={"state": "dismissed",
                            "dismissed_reason": reason,
                            "dismissed_comment": comment})
        except RuntimeError as e:
            # Report and keep going: one refused alert must not strand the rest,
            # and a silent partial run is what makes a later count unexplainable.
            print(f"  FAILED #{a['number']}: {e}", file=sys.stderr)
            failures += 1
    print(f"\ndismissed {len(planned) - failures} / {len(planned)}")
    return 1 if (failures or unadjudicated) else 0


if __name__ == "__main__":
    raise SystemExit(main())
