# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rendering a report as markdown: its sections, the error detail, the browser
context, the footer, and the title.
"""

from __future__ import annotations

import traceback
from typing import Optional

from localm.bugreport._common import MAINTAINER_EMAIL
from localm.bugreport.diagnostics import _ring_activity, collect_diagnostics
import localm.bugreport as _br


def _format_error(error: Optional[BaseException]) -> str:
    if error is None:
        return ""
    tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    # Full scrub, not just home paths: an exception message can embed a
    # user-configured credentialed URL (a comfy_api_url / net_search_url /
    # remote server base with user:pass@) or a pasted token, and this traceback
    # ships in the uploaded "Error detail" body. _scrub_secrets adds the
    # url-creds and bearer/apikey strips the sibling log-tail already applies.
    tb = _br._scrub_secrets(tb)
    # Keep the tail: the last frames are the useful ones.
    lines = tb.strip().splitlines()
    if len(lines) > 25:
        lines = ["... (earlier frames trimmed) ..."] + lines[-24:]
    return "\n".join(lines)


def _kv_lines(d: dict) -> list:
    """Render a flat dict as ``- key: value`` markdown lines (stable order)."""
    lines = []
    for k, v in d.items():
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v) if v else "(none)"
        elif isinstance(v, bool):
            v = "yes" if v else "no"
        lines.append(f"- {k}: {v}")
    return lines


def _app_state_lines(diag: dict) -> list:
    """The "what the app was doing" half of the report: loaded model, effective
    backend, session mode, debug state, and enabled plugins."""
    lines = []
    lm = diag.get("loaded_model") or {}
    if lm:
        loaded = "loaded" if lm.get("loaded") else "NOT loaded"
        line = f"- Model: {lm.get('model', '(unnamed)')} ({loaded})"
        if lm.get("backend"):
            line += f", backend {lm['backend']}"
        if lm.get("effective_ctx_max"):
            line += f", ctx<={lm['effective_ctx_max']}"
        lines.append(line)
    else:
        lines.append("- Model: (no engine loaded in this process)")
    if "session_mode" in diag:
        lines.append(f"- Session mode: {diag['session_mode']}")
    if "debug_mode" in diag:
        lines.append(f"- Debug logging: {'on' if diag['debug_mode'] else 'off'}")
    plugins = diag.get("enabled_plugins")
    if plugins is not None:
        lines.append(f"- Enabled plugins: {', '.join(plugins) if plugins else '(none)'}")
    return lines


def _environment_lines(diag: dict) -> list:
    """The "## Environment" lines: each collected diagnostics field under its
    label, in a fixed order, a list value joined with commas."""
    env_lines = []
    label = {
        "localm_version": "localm", "python": "Python", "platform": "OS",
        "machine": "Arch", "gpu_vendors": "GPU vendor(s)",
        "recommended_backend": "Recommended backend", "detect_source": "Detected via",
        "nvidia_gpu": "NVIDIA GPU", "nvidia_driver": "NVIDIA driver",
        "nvidia_cuda_capability": "Driver CUDA capability",
        "nvidia_compute_capability": "GPU compute capability",
        "nvidia_cuda_line": "Selected CUDA line",
        "nvidia_gpus": "NVIDIA GPUs (nvidia-smi order)",
        "gpus": "GPUs",
        "native_runtime_provisioned": "Native runtime provisioned",
        "native_runtime_backend": "Native runtime backend",
        "native_runtime_build": "Native runtime build",
        "native_runtime_pin": "Native runtime pinned to",
        "native_libs": "Native libraries", "operation": "Operation",
        "backend": "Backend (effective)", "requested_backend": "Backend (requested)",
        "with_cudart": "Fetched CUDA runtime bundle",
    }
    for key, name in label.items():
        if key in diag:
            val = diag[key]
            if isinstance(val, list):
                val = ", ".join(str(x) for x in val) if val else "(none)"
            env_lines.append(f"- {name}: {val}")
    return env_lines


def _context_diagnostic_sections(context: Optional[dict]) -> list:
    """The report sections built from the crash diagnostics passed in
    *context*: the native fault trace, the process exit code, the run's log
    tail (or why it was not collected) and the server hang trace, each scrubbed."""
    parts = []
    # Crash diagnostics passed via context (e.g. recovered-crash reports, which
    # have no Python error object): a faulthandler native trace when one was
    # captured, and the crashed run's own log tail (the actionable bit when a
    # window-close/OS-kill left no trace).
    ctx = context or {}
    native = ctx.get("native_trace")
    if native:
        parts += ["", "## Native fault trace", "```",
                  trim_trace_for_report(_br._scrub_secrets(str(native))), "```"]
    prior_exit = ctx.get("prior_exit")
    if isinstance(prior_exit, dict) and isinstance(prior_exit.get("exit_code"), int):
        from localm._mp_spawn import describe_exit_code
        parts += ["", "## Process exit",
                  f"- Exit code: {describe_exit_code(prior_exit['exit_code'], posix=False)}"]
        if isinstance(prior_exit.get("watched_for_s"), (int, float)):
            parts.append(f"- Watched for: {prior_exit['watched_for_s']}s")
    tail = ctx.get("recent_log_tail")
    if tail:
        parts += ["", "## Recent log (tail)", "```",
                  _br._scrub_secrets(str(tail))[:4000], "```"]
    elif ctx.get("log_unavailable"):
        # Say the log could not be COLLECTED rather than rendering nothing: an
        # omitted section is indistinguishable from a clean run. The reason is
        # built path-free at its source - see _log_failure_reason - and scrubbed
        # again here like every other context-supplied string.
        parts += ["", "## Recent log (tail)",
                  f"(not collected: {_br._scrub_secrets(str(ctx['log_unavailable']))[:200]})"]
    hang = ctx.get("hang_traces")
    if hang:
        parts += ["", "## Server hang trace (event-loop stall)",
                  "The hang watchdog captured every thread's stack when the server "
                  "froze (the top of the main thread is the blocking call):",
                  "```", _br._scrub_secrets(str(hang))[:8000], "```"]
    return parts


def build_report(summary: str, reason: str = "",
                 error: Optional[BaseException] = None,
                 context: Optional[dict] = None) -> str:
    """Render an editable markdown bug report. The user owns it before sending."""
    # The user-typed summary/reason are the only report fields not otherwise
    # scrubbed; a home path (username) or pasted credential in them would ship
    # in the uploaded body - and, via the derived title, into a PUBLIC issue.
    # Scrubbed at this choke point so every caller (report_failure,
    # save_user_report) is covered. _scrub_secrets is idempotent and no-ops on
    # empty text.
    summary = _br._scrub_secrets(summary)
    reason = _br._scrub_secrets(reason)
    ctx = context or {}
    # A user-composed report (save_user_report/the GUI form) can supply these
    # three DISTINCT fields via context, the same threading pattern used for
    # native_trace/recent_log_tail/hang_traces (_context_diagnostic_sections)
    # and client below. Absent for every
    # automatic (crash/LocalmError) report, whose "What happened" renders as
    # summary+reason. Scrubbed here, at the same choke point as summary/reason
    # above.
    what_i_did = _br._scrub_secrets((ctx.get("what_i_did") or "").strip())
    what_i_expected = _br._scrub_secrets((ctx.get("what_i_expected") or "").strip())
    what_happened = _br._scrub_secrets((ctx.get("what_happened") or "").strip())
    diag = collect_diagnostics(context)
    err = _format_error(error)

    env_lines = _environment_lines(diag)

    parts = [
        f"# localm bug report: {summary}",
        "",
        "## What I was doing",
        what_i_did or "<!-- Please describe what you ran and what you expected. -->",
        "",
    ]
    if what_i_expected:
        parts += ["## What I expected", what_i_expected, ""]
    if what_happened:
        happened_body = what_happened
    elif what_i_did:
        # A user-composed report (what_i_did present) that answered "what I was
        # doing" but skipped "what happened". summary is DERIVED from
        # what_happened-or-description upstream (save_user_report), so falling
        # back to it here would repeat what_i_did's own first line as a second
        # section; say plainly that this section was left blank instead.
        happened_body = "(not stated)"
    else:
        # No user-composed fields at all: an automatic crash/LocalmError
        # report, where there was never a separate "what happened" prompt -
        # summary/reason ARE the description of what happened, and this is the
        # report's ONLY account of it.
        happened_body = summary + ((f"\n\nReason: {reason}") if reason else "")
    parts += [
        "## What happened",
        happened_body,
        "",
        "## App state",
        "\n".join(_app_state_lines(diag)),
        "",
        "## Environment",
        "\n".join(env_lines) if env_lines else "(could not collect environment)",
    ]

    config_subset = diag.get("config_subset") or {}
    if config_subset:
        section = ["", "## Configuration (safe subset)"]
        if diag.get("config_unreadable"):
            # Say the file could not be READ rather than silently showing
            # defaults: without this line a corrupt config.json renders a
            # section byte-identical to a user with no config at all.
            section.append("(config.json exists but could not be read; the "
                            "values below are DEFAULTS, not the user's settings)")
        section.append("\n".join(_kv_lines(config_subset)))
        parts += section
    deps = diag.get("dependencies") or {}
    if deps:
        parts += ["", "## Dependencies", "\n".join(_kv_lines(deps))]

    if err:
        parts += ["", "## Error detail", "```", err, "```"]
    parts += _context_diagnostic_sections(context)

    # Always-on in-memory breadcrumbs (INFO+; no chat content) so even a non-debug
    # report shows what the app was doing right before the problem.
    ring = _ring_activity()
    if ring:
        parts += ["", "## Recent activity (in-memory log)", "```",
                  _br._scrub_secrets("\n".join(ring[-80:]))[-4000:], "```"]

    # Browser/client context for a GUI-filed report (user agent, page, viewport,
    # and recent JS console errors). Already a plain dict from the GUI; scrubbed
    # for home paths defensively.
    client = ctx.get("client")
    if isinstance(client, dict) and client:
        parts += ["", "## Browser / client"] + _client_lines(client)

    parts += [
        "",
        "---",
        _report_footer(),
        "",
    ]
    return "\n".join(parts)


def _report_footer() -> str:
    """The in-app disclaimer build_report() appends: true for a human reading
    the saved file or a downloaded copy (they genuinely can still edit it),
    false the instant the SAME text is what actually got sent - see
    _strip_report_footer(), the single place that fact is acted on."""
    return (f"Sent to the localm maintainer ({MAINTAINER_EMAIL}). "
            "You can edit anything above before sending.")


def _strip_report_footer(text: str) -> str:
    """Remove build_report()'s trailing edit-disclaimer block, if present.

    Called ONLY at the upload boundary (upload_report(), the single choke
    point every uploaded body flows through - the local saved file and the
    GUI's download-for-manual-send copy are UNCHANGED and keep the footer,
    since a human reading either of those genuinely can still edit before
    sending it themselves). Once a body is what actually got sent, "you can
    edit this before sending" is no longer true, and it re-publishes
    MAINTAINER_EMAIL into whatever the upload becomes (a PUBLIC GitHub issue).

    Exact-suffix match built from the SAME _report_footer() text
    build_report() appends, so the append site and the strip site cannot
    silently drift apart. Tolerates a missing trailing newline (a
    manually re-saved file); a body that never had the footer, or had it
    edited away, is returned unchanged - never raises."""
    footer = _report_footer()
    for suffix in (f"\n\n---\n{footer}\n", f"\n\n---\n{footer}"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _client_lines(client: dict) -> list:
    """Render the GUI-supplied browser context (user agent, page, viewport, recent
    console errors) for the report. All values are treated as untrusted text."""
    lines = []
    for field, label_ in (("userAgent", "User agent"), ("page", "Page"),
                          ("viewport", "Viewport"), ("appVersion", "GUI build")):
        val = client.get(field)
        if val:
            lines.append(f"- {label_}: {_br._scrub_secrets(str(val))[:300]}")
    console_errs = client.get("console")
    if isinstance(console_errs, list) and console_errs:
        rendered = "\n".join(_br._scrub_secrets(str(e)) for e in console_errs[-40:])
        lines += ["", "Recent browser console errors:", "```", rendered[-3000:], "```"]
    return lines


def report_title(summary: str, what_happened: str, description: str) -> str:
    """The final report/issue title: *summary* verbatim if the caller gave one
    explicitly, else derived from *what_happened* (a more useful issue title
    than "what I was doing"), falling back to *description* - first line
    only, truncated to 120 chars, never empty.

    A standalone function (not inlined in save_user_report) so the ``localm
    bug-report`` CLI can compute the EXACT SAME title after the report is
    built, to hand to upload_report/offer_to_send, rather than re-deriving it
    with logic that could silently disagree with what the report itself
    says."""
    if not summary:
        title_source = what_happened or description
        summary = title_source.splitlines()[0] if title_source else ""
    return summary.strip()[:120] or "user-reported issue"


_TRACE_REPORT_LIMIT = 4000


def trim_trace_for_report(text: str) -> str:
    """*text* cut to the report limit by dropping the MIDDLE, so both the first
    fault header and the end of the file (where the process stopped writing)
    survive. Debug mode keeps the whole trace."""
    from localm.debuglog import debug_enabled
    if debug_enabled() or len(text) <= _TRACE_REPORT_LIMIT:
        return text
    head = _TRACE_REPORT_LIMIT // 3
    tail = _TRACE_REPORT_LIMIT - head
    dropped = len(text) - head - tail
    return "\n".join((text[:head], f"... ({dropped} characters omitted) ...",
                      text[-tail:]))
