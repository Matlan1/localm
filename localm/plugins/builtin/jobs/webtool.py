# SPDX-License-Identifier: AGPL-3.0-or-later
"""Give a scheduled CHAT job the same web-search tool the interactive chat has.

A scheduled chat job runs its prompt straight against the engine, with no tools, so
a web-lookup job ("look up the weather") answers "I have no real-time access" every
run (U-3). This module runs a small, BOUNDED server-side ReAct loop that mirrors the
GUI's web tool: a system prompt teaches the model the
``<tool_call>{"name":"web_search","args":{"query":"..."}}</tool_call>`` protocol
(plus ``fetch_url``); when the model emits a call we run it through the same
policy-enforced ``localm.netpolicy`` search/fetch the GUI uses, inject the result as
the next message, and let the model answer and cite it.

Policy: web access is offered only when ``localm.netpolicy.network_mode() != "off"``.
A scheduled job the user created and enabled is a pre-authorised standing action
(netpolicy treats explicit user actions as consent), so there is no separate per-run
"ask" prompt - there is nobody to ask in an unattended run. ``net_mode=off`` still
kills it, exactly like every other model-initiated request. When web is off the model
is given an offline-honesty floor so it says it cannot verify rather than inventing.
The loop is capped so a job can never spin on the web forever.
"""

from __future__ import annotations

import json
import re

from localm.debuglog import logger

# How many search/fetch rounds a single job run may take before it must answer.
_MAX_ROUNDS = 4

_WEB_TOOLS = {"web_search", "fetch_url"}


# --------------------------------------------------------------------------- #
#  System prompts (mirror the GUI's WEB_TOOL_PROMPT / NO_WEB_PROMPT floors)    #
# --------------------------------------------------------------------------- #

WEB_TOOL_SYSTEM = (
    "You are running an automated, scheduled task. You can access the internet "
    "through tools. When the answer depends on current, real-time, or external "
    "information you cannot be certain of (news, weather, prices, software versions, "
    "documentation, anything after your training cutoff), get it from the web instead "
    "of guessing. Reply with ONLY a tool call block and nothing else:\n"
    '<tool_call>{"name": "web_search", "args": {"query": "..."}}</tool_call>\n'
    "To read a specific page:\n"
    '<tool_call>{"name": "fetch_url", "args": {"url": "https://..."}}</tool_call>\n'
    "The results arrive in the next message; then answer and cite the source URLs you "
    "used. HONESTY: never invent search results, URLs, or page contents, and never say "
    "you searched or read a page unless you actually emitted a tool call and received "
    "its result. If a search fails or finds nothing useful, say so plainly."
)

OFFLINE_SYSTEM = (
    "You are running an automated, scheduled task with NO internet access. Do not "
    "present guessed or invented information as verified fact: current events, news, "
    "weather, prices, live data, or anything you cannot confirm. Never claim you looked "
    "something up, searched the web, or read a page, because you cannot. If the task "
    "needs current or external information, say plainly that you cannot verify it "
    "offline (web access is off; net_mode=off). Saying you do not know is better than "
    "stating something false."
)


# --------------------------------------------------------------------------- #
#  Tool-call parsing (a server-side port of the GUI's parseWebCall)           #
# --------------------------------------------------------------------------- #

_THINK_RE = re.compile(r"<(think|reasoning|r)\b[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_WRAP_RE = re.compile(
    r"<\|?/?tool_call\|?>\s*(?:call:(\w+)\s*)?(.*?)\s*<\|?/?tool_call\|?>",
    re.DOTALL)
_FENCE_RE = re.compile(r"```[ \t]*[A-Za-z_]*[ \t]*\r?\n(.*?)\r?\n[ \t]*```", re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "")


def _lenient_json(body: str):
    """Parse JSON, tolerating the mangles local finetunes emit (single-quoted keys,
    trailing commas). Returns a dict or None."""
    fixes = (
        lambda s: s,
        lambda s: re.sub(r"'([^']+)'\s*:", r'"\1":', s),
        lambda s: re.sub(r",(\s*[}\]])", r"\1", s),
        lambda s: re.sub(r",(\s*[}\]])", r"\1",
                         re.sub(r"'([^']+)'\s*:", r'"\1":', s)),
    )
    for fix in fixes:
        try:
            obj = json.loads(fix(body))
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _as_web_call(obj):
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if name not in _WEB_TOOLS:
        return None
    args = obj.get("args") if isinstance(obj.get("args"), dict) else None
    if args is None:
        args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {}
    return {"name": name, "args": args}


def _top_level_objects(text: str):
    """Yield each brace-balanced top-level ``{...}`` region (string-aware)."""
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, in_str, esc = 0, False, False
        j = i
        matched = False
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[i:j + 1]
                    i = j + 1
                    matched = True
                    break
            j += 1
        if not matched:
            i += 1


def parse_web_call(text: str):
    """First web tool call in *text*, or None. Tolerates the ``<tool_call>`` wrapper,
    code fences, and bare JSON, plus the JSON mangles local models emit - mirroring
    the GUI so a real attempt is not silently dropped."""
    clean = _strip_think(text or "")
    candidates = []
    for m in _WRAP_RE.finditer(clean):
        candidates.append((m.group(1), m.group(2)))
    for m in _FENCE_RE.finditer(clean):
        candidates.append((None, m.group(1)))
    for prefix, body in candidates:
        obj = _lenient_json(body.strip())
        call = _as_web_call(obj)
        if not call and obj is not None and prefix in _WEB_TOOLS:
            call = {"name": prefix, "args": obj}
        if call:
            return call
    for chunk in _top_level_objects(clean):
        call = _as_web_call(_lenient_json(chunk))
        if call:
            return call
    return None


# --------------------------------------------------------------------------- #
#  Running a tool call through the network policy                             #
# --------------------------------------------------------------------------- #

def web_enabled() -> bool:
    """True when scheduled jobs may use the web (net_mode is not "off")."""
    try:
        from localm.netpolicy import network_mode
        return network_mode() != "off"
    except Exception:
        return False


def run_web_call(call: dict) -> str:
    """Execute one web tool call through ``localm.netpolicy`` and return the text to
    feed back to the model (results, or a failure note it can adapt to)."""
    from localm import netpolicy

    name = call.get("name")
    args = call.get("args") or {}
    try:
        if name == "web_search":
            query = str(args.get("query", "")).strip()
            results = netpolicy.web_search(query, max_results=5)
            return (f'[Results of web_search "{query}"]\n'
                    + netpolicy.format_results(results))
        if name == "fetch_url":
            final_url, text = netpolicy.fetch_text(str(args.get("url", "")))
            return f"[Content of {final_url}]\n{text[:6000]}"
        return f"[Unknown web tool: {name}] Answer without it."
    except netpolicy.NetworkPolicyError as e:
        return (f"[Web request refused by policy: {e}] Answer without the web and say "
                "web access was refused.")
    except Exception as e:
        return (f"[Web request failed: {e}] Answer without the web, and say that web "
                "access did not work.")


def _complete(engine, messages: list) -> str:
    return "".join(engine.chat_stream(messages)).strip()


def run_chat_with_web(engine, prompt: str, *, max_rounds: int = _MAX_ROUNDS) -> str:
    """Run a scheduled chat *prompt* against *engine*, giving it the web tool when
    network access is on. Returns the model's final answer.

    Mirrors the GUI: a system prompt teaches the tool protocol, the model emits a
    tool call, we run it and feed the result back, and the model answers. Capped at
    *max_rounds* search rounds so a job can never spin on the web forever (U-3 / R36)."""
    if not web_enabled():
        messages = [{"role": "system", "content": OFFLINE_SYSTEM},
                    {"role": "user", "content": prompt}]
        return _complete(engine, messages)

    messages = [{"role": "system", "content": WEB_TOOL_SYSTEM},
                {"role": "user", "content": prompt}]
    for _ in range(max_rounds):
        reply = _complete(engine, messages)
        call = parse_web_call(reply)
        if call is None:
            return reply                       # the model answered
        logger.debug("jobs web tool: %s %s", call.get("name"), call.get("args"))
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": run_web_call(call)})

    # Round cap reached: force an answer from what was gathered, no more searching.
    messages.append({"role": "user", "content":
                     "Use the information already gathered to answer now. "
                     "Do not search again."})
    final = _complete(engine, messages)
    if parse_web_call(final) is not None:
        # The model kept trying to call a tool instead of answering - don't return a
        # raw tool-call block as the job's "answer".
        return ("(Could not complete the web lookup within the round limit; the "
                "search results gathered are above in the run log.)")
    return final
