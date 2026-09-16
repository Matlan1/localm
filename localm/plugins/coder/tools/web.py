# SPDX-License-Identifier: AGPL-3.0-or-later
"""Network tools: ``fetch_url`` (one page through localm.netpolicy) and
``web_search`` (the shared localm.web_retrieval controller: search, read the
top result pages, select evidence). Both import lazily inside the call."""

from __future__ import annotations

from pathlib import Path

from .base import ToolResult, _truncate

def tool_fetch_url(
    cwd: Path,
    url: str,
    max_chars: int = 8000,
    _privacy: bool = False,
) -> ToolResult:
    """
    Fetch a URL and return its plain-text content (HTML tags stripped).

    Useful for documentation pages, GitHub raw files, Stack Overflow answers,
    and package changelogs.  Content is truncated to ``max_chars`` to avoid
    flooding the context window.

    Routed through localm.netpolicy: net_mode/net_allow/net_deny apply, every
    redirect hop is re-validated, and private/loopback targets are refused
    unless net_allow_private is set.

    In privacy mode (``_privacy=True``) a one-line network audit message is
    emitted to stderr before the request so the user can see outbound URLs.
    """
    from localm.netpolicy import NetworkPolicyError, fetch_text

    if _privacy:
        import sys as _sys
        print(f"[localm privacy] fetch_url: {url}", file=_sys.stderr, flush=True)

    try:
        final_url, text = fetch_text(url)
    except NetworkPolicyError as e:
        return ToolResult.error(str(e))
    except Exception as e:
        return ToolResult.error(f"Could not fetch {url}: {e}")

    output, trunc = _truncate(text, max_chars)
    return ToolResult(
        ok=True,
        output=f"<url>{final_url}</url>\n<content>\n{output}\n</content>",
        summary=f"fetched {url[:60]} ({len(text):,} chars{', truncated' if trunc else ''})",
        truncated=trunc,
    )


def tool_web_search(
    cwd: Path,
    query: str,
    max_results: int = 5,
    _privacy: bool = False,
) -> ToolResult:
    """
    Search the web, read the top result pages and return an evidence bundle:
    a grounding label, the sources labelled S1, S2, ... (title, URL, what
    backs each) and evidence excerpts from the pages that could be read.

    ``max_results`` is the number of search candidates (1..10; the top three
    are read). Use fetch_url to read a page the evidence did not cover.
    Every request goes through localm.netpolicy like fetch_url. A provider
    failure or an empty search is a tool error. In privacy mode
    (``_privacy=True``) the query and then every attempted page read are
    echoed to stderr as network audit lines.

    The evidence text is remote-controlled: the output is built with
    ``untrusted_span``, which neutralises it and records it as an untrusted
    range; the grounding label stays trusted.
    """
    from localm.netpolicy import NetworkPolicyError
    from localm.textguard import compose, neutralise, untrusted_span
    from localm.web_retrieval import retrieve
    import sys as _sys

    if _privacy:
        print(f"[localm privacy] web_search: {query}", file=_sys.stderr, flush=True)

    try:
        bundle = retrieve(query, search_candidates=max_results)
    except NetworkPolicyError as e:
        return ToolResult.error(str(e))
    except Exception as e:
        return ToolResult.error(f"Web search failed: {e}")

    if _privacy:
        # Every page read the retrieval attempted is an outbound request too.
        for src in bundle.sources:
            if src.retrieval_status != "skipped":
                print(f"[localm privacy] web_search read: {src.url}",
                      file=_sys.stderr, flush=True)

    if bundle.search_status != "ok":
        return ToolResult.error(
            "Web search failed: "
            + neutralise(bundle.search_error or bundle.search_status))

    return ToolResult.success(
        compose(f"[{bundle.grounding_summary()}]\n",
                untrusted_span(bundle.to_prompt_text())),
        summary=(f"web_search '{query[:50]}' ({len(bundle.sources)} sources, "
                 f"{bundle.pages_read} pages read, {bundle.grounding})"),
    )
