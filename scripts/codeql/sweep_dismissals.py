#!/usr/bin/env python
"""Post-merge dismissal sweep: detect and restore dismissals that EVAPORATED.

The failure this exists for
---------------------------
Dismissing a CodeQL alert attaches the dismissal to an alert NUMBER. A later
merge that edits the alert's SINK LINE changes that line's content, so its
`primaryLocationLineHash` changes, CodeQL retires the old alert and raises a NEW
one at the new position - **carrying no dismissal**.

That is a silent security-record regression with no reporter: no test fails, no
check flags it, CI is green, and the alert queue simply grows with no
explanation. A false positive sitting open is indistinguishable from unfixed
work, which is exactly the state dismissing it was meant to eliminate.

**It is not hypothetical.** On 2026-07-29, #842 edited `cli/models.py:333` and
silently un-dismissed alerts #139/#140/#141, which reappeared as #231/#232/#233.
A merged security fix can un-dismiss a false positive it had nothing to do with.

Why the record must exist BEFORE the event
------------------------------------------
After a renumbering the link between "the alert I dismissed" and "this new alert"
is NOT recoverable from the API: the old number is retired and the new one
carries no reference back. **The window to make this detectable closes the moment
a merge lands.** So the identities must be captured in advance - that is what
`dismissed_fingerprints.json` is, and why this script re-captures on every run.

Usage
-----
  python sweep_dismissals.py                 # check only, report evaporations
  python sweep_dismissals.py --restore       # re-dismiss what evaporated
  python sweep_dismissals.py --capture-only  # refresh the record, no checking

Run it after EVERY merge. It is a merge-gate obligation, not a habit: a merge is
not landed until this has run clean.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = "Matlan1/localm"
RECORD = Path(__file__).with_name("dismissed_fingerprints.json")


def gh_json(url):
    out = subprocess.run(["gh", "api", url, "--paginate"],
                         capture_output=True, text=True, check=True).stdout
    dec, i, items = json.JSONDecoder(), 0, []
    while i < len(out):
        while i < len(out) and out[i].isspace():
            i += 1
        if i >= len(out):
            break
        obj, i = dec.raw_decode(out, i)
        items.extend(obj)
    return items


def latest_sarif():
    """Line hashes for current master, keyed by (rule, file, line)."""
    an = gh_json(f"repos/{REPO}/code-scanning/analyses"
                 "?per_page=20&ref=refs/heads/master")
    py = [a for a in an if a.get("category") == "/language:python"]
    if not py:
        return {}
    raw = subprocess.run(
        ["gh", "api", f"repos/{REPO}/code-scanning/analyses/{py[0]['id']}",
         "-H", "Accept: application/sarif+json"],
        capture_output=True, text=True, check=True).stdout
    idx = {}
    for r in json.loads(raw)["runs"][0]["results"]:
        p = r["locations"][0]["physicalLocation"]
        idx[(r["ruleId"], p["artifactLocation"]["uri"], p["region"]["startLine"])] = \
            r.get("partialFingerprints", {}).get("primaryLocationLineHash")
    return idx


def capture(dismissed, idx):
    """Record (rule, lineHash, justification) per dismissed alert.

    The justification is stored so a RESTORE carries the ORIGINAL reasoning
    rather than something re-improvised later - a restored dismissal whose
    comment differs from the one it replaces is a quiet rewrite of the record."""
    rec, missed = {}, []
    for a in dismissed:
        loc = a["most_recent_instance"]["location"]
        h = idx.get((a["rule"]["id"], loc["path"], loc["start_line"]))
        if h:
            rec[str(a["number"])] = {
                "rule": a["rule"]["id"], "file": loc["path"],
                "line": loc["start_line"], "lineHash": h,
                "comment": a.get("dismissed_comment") or "",
            }
        else:
            missed.append((a["number"], a["rule"]["id"], loc["path"]))
    return rec, missed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--capture-only", action="store_true")
    args = ap.parse_args()

    dismissed = gh_json(f"repos/{REPO}/code-scanning/alerts?state=dismissed&per_page=100")
    open_now = gh_json(f"repos/{REPO}/code-scanning/alerts?state=open&per_page=100")
    idx = latest_sarif()
    print(f"dismissed now: {len(dismissed)}   open now: {len(open_now)}")

    prior = json.loads(RECORD.read_text()) if RECORD.exists() else {}
    live = {str(a["number"]) for a in dismissed}

    # An alert we recorded as dismissed that is no longer in the dismissed set
    # has been RETIRED. Look for its content reappearing under a new number.
    evaporated = []
    if not args.capture_only:
        open_by_hash = {}
        for a in open_now:
            loc = a["most_recent_instance"]["location"]
            h = idx.get((a["rule"]["id"], loc["path"], loc["start_line"]))
            if h:
                open_by_hash.setdefault((a["rule"]["id"], h), []).append(a)
        for num, r in prior.items():
            if num in live:
                continue                      # still dismissed, fine
            for a in open_by_hash.get((r["rule"], r["lineHash"]), []):
                evaporated.append((num, a, r))

        if evaporated:
            print(f"\n!! {len(evaporated)} EVAPORATED DISMISSAL(S) - open again "
                  f"under a new number:")
            for old, a, r in evaporated:
                loc = a["most_recent_instance"]["location"]
                print(f"   #{old} ({r['file']}:{r['line']}) -> now #{a['number']} "
                      f"at {loc['path']}:{loc['start_line']}")
        else:
            print("\nno evaporated dismissals detected")

        if evaporated and args.restore:
            for old, a, r in evaporated:
                c = r.get("comment") or ""
                c = (c[:200] + f" [restored from #{old}]")[:280]
                ok = subprocess.run(
                    ["gh", "api", "--method", "PATCH",
                     f"repos/{REPO}/code-scanning/alerts/{a['number']}",
                     "-f", "state=dismissed",
                     "-f", "dismissed_reason=false positive",
                     "-f", f"dismissed_comment={c}"],
                    capture_output=True, text=True).returncode == 0
                print(f"   {'restored' if ok else 'FAILED  '} #{a['number']}")
            dismissed = gh_json(
                f"repos/{REPO}/code-scanning/alerts?state=dismissed&per_page=100")
        elif evaporated:
            print("   (re-run with --restore to re-dismiss these)")

    rec, missed = capture(dismissed, idx)
    RECORD.write_text(json.dumps(rec, indent=1))
    print(f"\ncaptured {len(rec)} dismissal fingerprints -> {RECORD.name}")
    if missed:
        print(f"  {len(missed)} dismissed alert(s) NOT in the current SARIF - "
              f"already retired, so their identity is unrecoverable:")
        for m in missed:
            print(f"    #{m[0]}  {m[1]}  {m[2]}")
        print("  A retired-and-not-reappearing alert is usually benign (the code")
        print("  changed). Treat it as unexplained until you have read the diff.")
    return 1 if (evaporated and not args.restore) else 0


if __name__ == "__main__":
    sys.exit(main())
