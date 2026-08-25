# SPDX-License-Identifier: AGPL-3.0-or-later
"""Network tools: ``fetch_url`` and ``web_search``, both routed through localm.netpolicy (imported lazily inside each call)."""

from __future__ import annotations

from pathlib import Path

from .base import ToolResult, _truncate

def tool_fetch_url(
    cwd: Path,
    url: str,
    max_chars: int = 8000,
    _privacy: bool = False,
) -> ToolResult:
    """Fetch a URL and return its plain-text content (HTML tags stripped)."""
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
    """Search the web and return numbered results (title, URL, snippet)."""
    from localm.netpolicy import NetworkPolicyError, format_results, web_search

    if _privacy:
        import sys as _sys
        print(f"[localm privacy] web_search: {query}", file=_sys.stderr, flush=True)

    try:
        results = web_search(query, max_results=max_results)
    except NetworkPolicyError as e:
        return ToolResult.error(str(e))
    except Exception as e:
        return ToolResult.error(f"Web search failed: {e}")

    return ToolResult.success(
        format_results(results),
        summary=f"web_search '{query[:50]}' ({len(results)} results)",
    )
