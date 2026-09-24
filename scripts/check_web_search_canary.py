#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Non-blocking live canary: run ONE real web search through localm's own
search provider and report whether the result layout still parses (ADR-0022
Phase 5 quality gates, item 25).

``localm.web_retrieval.providers`` scrapes DuckDuckGo's no-key HTML endpoint
(or a configured SearXNG instance's JSON API) by hand-written parsing - see
``localm/web_retrieval/providers.py``'s ``_DDGParser``. Neither backend has a
stable public contract: DuckDuckGo can change its result markup or rate-limit
a datacenter IP at any time, and a self-hosted SearXNG's shape depends on its
own version. Every OTHER test of this pipeline
(``tests/test_web_retrieval_*.py``) runs against a recorded fixture double, by
design - offline PR CI must never depend on the internet, see
``tests/_web_retrieval_fixtures.py``. Nothing else in this repo makes a REAL
network call to a live search backend, so a live layout break would otherwise
go unnoticed until a user hit it.

This script closes that gap as a MAINTENANCE SIGNAL, never a gate: it makes
exactly one real search and reports whether it returned parseable results,
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


def _summarise(lines: "list[str]") -> None:
    try:
        ci_runner_files.append(ci_runner_files.STEP_SUMMARY, "\n".join(lines) + "\n")
    except OSError as e:
        print(f"(could not write the step summary: {e})")


def run_canary(query: str = DEFAULT_QUERY) -> dict:
    """One real search through localm's own provider selection - DuckDuckGo by
    default, or a configured SearXNG (``localm.web_retrieval.providers.
    provider_from_config``). Returns a result dict; never raises - every
    failure (a policy refusal, a transport error, an unparseable response) is
    captured in the dict so main() has exactly one path to report from."""
    from localm.web_retrieval import providers

    provider = providers.provider_from_config()
    try:
        results = provider.search(query, _RESULTS_TO_REQUEST)
    except Exception as e:
        return {"provider": provider.name, "ok": False,
                "error": f"{type(e).__name__}: {e}"}
    return {"provider": provider.name, "ok": bool(results),
            "count": len(results)}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", default=DEFAULT_QUERY,
                    help=f"search query to run (default: {DEFAULT_QUERY!r})")
    args = ap.parse_args(argv)

    result = run_canary(args.query)
    provider = result["provider"]

    if result["ok"]:
        print(f"WEB SEARCH CANARY: OK - {provider} returned "
              f"{result['count']} result(s) for {args.query!r}.")
        _annotate("notice", f"web search canary: {provider} is working "
                            f"({result['count']} result(s))")
        _summarise(["## Web search canary: OK",
                    f"`{provider}` returned {result['count']} result(s) for "
                    f"`{args.query}`."])
        return 0

    reason = result.get("error", "returned no parseable results")
    print(f"WEB SEARCH CANARY: DEGRADED - {provider}: {reason}")
    print("  This is a maintenance signal, not a failure of this build: it "
          "may be a rate limit, a layout change, or a transient network "
          "issue. See localm/web_retrieval/providers.py.")
    _annotate("warning", f"web search canary: {provider} looks degraded "
                        f"({reason}); this job never fails the build")
    _summarise(["## Web search canary: DEGRADED",
                f"`{provider}`: {reason}", "",
                "Non-blocking - investigate if this persists across runs."])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
