#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Non-blocking live canary: run one real web search through each of
localm's own search services and report whether each result layout still
parses (ADR-0022 Phase 5 quality gates, item 25).

``localm.web_retrieval.providers`` scrapes DuckDuckGo's HTML and lite pages
and Brave Search's result page (or a configured SearXNG instance) by
hand-written parsing - see ``localm/web_retrieval/providers.py``. None of
them has a stable public contract: a service can change its result markup or
rate-limit a datacenter IP at any time, and a self-hosted SearXNG's shape
depends on its own version. Each built-in service is searched on its own, so
a layout break on one is reported even while another still answers. Every OTHER test of this pipeline
(``tests/test_web_retrieval_*.py``) runs against a recorded fixture double, by
design - offline PR CI must never depend on the internet, see
``tests/_web_retrieval_fixtures.py``. Nothing else in this repo makes a REAL
network call to a live search backend, so a live layout break would otherwise
go unnoticed until a user hit it.

This script closes that gap as a MAINTENANCE SIGNAL, never a gate: it makes
one real search per service and reports whether each returned parseable
results,
and it ALWAYS exits 0 - the same shape as scripts/check_comfyui_pin.py's
default (non-``--gate``) mode. It is wired into .github/workflows/ci.yml on
``schedule`` and ``workflow_dispatch`` only - never ``pull_request`` and never
a plain ``push`` - so offline PR CI never depends on the internet and a rate
limit or a markup change can never turn a PR red.

Usage:
    python scripts/check_web_search_canary.py
    python scripts/check_web_search_canary.py --query "a different query"

Needs localm installed (imports ``localm.web_retrieval`` / ``localm.netpolicy``),
unlike the stdlib-only pin-currency scripts this mirrors the shape of - it
exercises localm's OWN scraping code, not a GitHub API response.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import ci_runner_files  # noqa: E402

DEFAULT_QUERY = "localm local llm"
_RESULTS_TO_REQUEST = 5


def _annotate(level: str, message: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level}::{message}")


def _summarise(lines: list[str]) -> None:
    try:
        ci_runner_files.append(ci_runner_files.STEP_SUMMARY, "\n".join(lines) + "\n")
    except OSError as e:
        print(f"(could not write the step summary: {e})")


def run_canary(query: str = DEFAULT_QUERY) -> list[dict]:
    """One real search per service localm's search uses: each service of the
    built-in chain separately, or the configured SearXNG instance
    (``localm.web_retrieval.providers.provider_from_config``). Returns one
    dict per service: ``{"provider", "ok": True, "count"}``, or
    ``{"provider", "ok": False, "error", "bot_check"}``. Never raises."""
    from localm.web_retrieval import providers

    try:
        provider = providers.provider_from_config()
    except Exception as e:
        return [{"provider": "search", "ok": False, "bot_check": False,
                 "error": f"{type(e).__name__}: {e}"}]
    out = []
    for service in getattr(provider, "routes", None) or [provider]:
        try:
            results = service.search(query, _RESULTS_TO_REQUEST)
        except Exception as e:
            out.append({"provider": service.name, "ok": False,
                        "bot_check": isinstance(e, providers.BotCheckError),
                        "error": f"{type(e).__name__}: {e}"})
            continue
        if results:
            out.append({"provider": service.name, "ok": True,
                        "count": len(results)})
        else:
            out.append({"provider": service.name, "ok": False,
                        "bot_check": False, "error": "returned no results"})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", default=DEFAULT_QUERY,
                    help=f"search query to run (default: {DEFAULT_QUERY!r})")
    args = ap.parse_args(argv)

    services = run_canary(args.query)
    lines = []
    for r in services:
        if r["ok"]:
            line = f"{r['provider']}: OK, {r['count']} result(s)"
        elif r.get("bot_check"):
            line = (f"{r['provider']}: answered with a bot check (a rate "
                    f"limit, not a layout break): {r['error']}")
        else:
            line = f"{r['provider']}: DEGRADED - {r['error']}"
        lines.append(line)
    degraded = [r for r in services if not r["ok"] and not r.get("bot_check")]

    if not degraded:
        print(f"WEB SEARCH CANARY: OK for {args.query!r}")
        for line in lines:
            print(f"  {line}")
        _annotate("notice", "web search canary: " + "; ".join(lines))
        _summarise(["## Web search canary: OK", ""]
                   + [f"- {line}" for line in lines])
        return 0

    print(f"WEB SEARCH CANARY: DEGRADED for {args.query!r}")
    for line in lines:
        print(f"  {line}")
    print("  This is a maintenance signal, not a failure of this build: it "
          "may be a rate limit, a layout change, or a transient network "
          "issue. See localm/web_retrieval/providers.py.")
    for r in degraded:
        _annotate("warning", f"web search canary: {r['provider']} looks "
                             f"degraded ({r['error']}); this job never fails "
                             "the build")
    _summarise(["## Web search canary: DEGRADED", ""]
               + [f"- {line}" for line in lines]
               + ["", "Non-blocking - investigate if this persists across runs."])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
