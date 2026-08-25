# SPDX-License-Identifier: AGPL-3.0-or-later
"""File-system tools: read/write/edit/patch, directory listing/tree, notebook cell edits, and content/glob search-and-replace."""

from __future__ import annotations

import difflib
import glob as _glob
import json
import os
import os.path as _os_path
import re
import stat as _stat
import textwrap
from collections import deque
from pathlib import Path
from typing import Optional

from .base import ToolResult, _confine, _truncate
# The indexer's own skip/file-type tables, so grep and the project map agree on
# what counts as noise and what counts as source (indexer imports nothing from
# this package, so this is not a cycle).
from ..indexer import _SKIP_DIRS, _SYMBOL_LANGS, _TEXT_EXTS

def _render_notebook(nb: dict) -> str:
    """Convert a parsed .ipynb dict to a human-readable text representation."""
    cells = nb.get("cells", [])
    parts: list[str] = []
    for i, cell in enumerate(cells):
        ctype  = cell.get("cell_type", "code")
        source = "".join(cell.get("source", []))
        header = f"[Cell {i} | {ctype}]"
        parts.append(header)
        parts.append(source)
        if ctype == "code":
            outputs = cell.get("outputs", [])
            if outputs:
                out_lines: list[str] = []
                for out in outputs:
                    otype = out.get("output_type", "")
                    if otype in ("stream",):
                        out_lines.extend(out.get("text", []))
                    elif otype in ("execute_result", "display_data"):
                        data = out.get("data", {})
                        if "text/plain" in data:
                            out_lines.extend(data["text/plain"])
                    elif otype == "error":
                        out_lines.append(f"{out.get('ename')}: {out.get('evalue')}")
                if out_lines:
                    parts.append("--- output ---")
                    parts.append("".join(out_lines).rstrip())
        parts.append("")
    return "\n".join(parts)


def _line_count(text: str) -> int:
    return text.count("\n") + 1


def _closest_snippet(text: str, old: str, min_score: float = 0.55) -> str:
    """Find the file region most similar to a failed `old` string and return a short line-numbered snippet of it, so the model can see exactly how the real text differs (usually whitespace or a changed identifier)."""
    text_lines = text.splitlines()
    old_lines = [l for l in old.splitlines() if l.strip()]
    if not text_lines or not old_lines:
        return ""
    probe = old_lines[0].strip()
    best_i, best_score = -1, 0.0
    for i, line in enumerate(text_lines):
        score = difflib.SequenceMatcher(None, probe, line.strip()).ratio()
        if score > best_score:
            best_i, best_score = i, score
    if best_i < 0 or best_score < min_score:
        return ""
    start = max(0, best_i - 1)
    end = min(len(text_lines), best_i + len(old_lines) + 1)
    return "\n".join(f"{n + 1:4d}: {text_lines[n]}" for n in range(start, end))


def _verify_syntax(path: Path, content: str) -> Optional[str]:
    """Quick offline syntax check for common file types."""
    suffix = path.suffix.lower()
    if suffix == ".py":
        # compile() builtin only: it parses to a code object in memory and never
        # touches disk. py_compile.compile() (the previous approach) writes the
        # source to a temp .py file to compile it, and CPython's import-cache
        # write leaves a matching .pyc behind in the system temp dir's
        # __pycache__/ that nothing here ever unlinked - i.e. the file's own
        # content, compiled, sitting in shared temp storage regardless of
        # session mode. compile() has no such side effect.
        try:
            compile(content, str(path), "exec")
        except SyntaxError as e:
            return f"Python syntax error: {e}"
    elif suffix == ".json":
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            return f"JSON syntax error: {e}"
    elif suffix == ".toml":
        try:
            import tomllib  # type: ignore[import]
            tomllib.loads(content)
        except Exception as e:
            return f"TOML syntax error: {e}"
    return None


def tool_read_file(cwd: Path, path: str, offset: int = 0, limit: int = 0) -> ToolResult:
    """Read a file. *offset* (1-based start line) and *limit* (max lines) slice big files so a truncated first read can be followed by targeted reads of the middle instead of re-fetching the whole file."""
    try:
        p = _confine(cwd, path)
    except PermissionError as e:
        return ToolResult.error(str(e))
    if not p.exists():
        return ToolResult.error(f"File not found: {p}")
    if not p.is_file():
        return ToolResult.error(f"Not a file: {p}")
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return ToolResult.error(str(e))

    rel = p.relative_to(cwd) if p.is_relative_to(cwd) else p

    # Render Jupyter notebooks as readable text rather than raw JSON
    # (offset/limit are ignored - cell structure beats line numbers there)
    if p.suffix == ".ipynb":
        try:
            nb   = json.loads(raw)
            text = _render_notebook(nb)
            n    = len(nb.get("cells", []))
            output, trunc = _truncate(text)
            return ToolResult(
                ok=True,
                output=f"<path>{rel}</path>\n<cells>{n}</cells>\n<content>\n{output}\n</content>",
                summary=f"{rel} - {n} cells{' (truncated)' if trunc else ''}",
                truncated=trunc,
            )
        except Exception:
            pass  # fall through to plain-text read if JSON is malformed

    total_lines = _line_count(raw)
    if offset or limit:
        start = max(1, int(offset) or 1)
        if start > total_lines:
            return ToolResult.error(
                f"offset {start} is past the end of {rel} ({total_lines} lines)")
        count = int(limit) if limit else total_lines
        # An empty file is 1 (empty) line per _line_count, but splitlines()
        # gives []; without the fallback the slice is empty and the range
        # label comes out backwards ("1-0 of 1").
        all_lines = raw.splitlines(keepends=True) or [""]
        sliced = all_lines[start - 1:start - 1 + count]
        end = start + len(sliced) - 1
        output, trunc = _truncate("".join(sliced))
        range_label = f"{start}-{end} of {total_lines}"
        return ToolResult(
            ok=True,
            output=f"<path>{rel}</path>\n<lines>{range_label}</lines>\n<content>\n{output}\n</content>",
            summary=f"{rel} - lines {range_label}{' (truncated)' if trunc else ''}",
            truncated=trunc,
        )

    output, trunc = _truncate(raw)
    if trunc:
        output += ("\n[file truncated - re-read specific parts with "
                   "read_file(path, offset=<start line>, limit=<lines>)]")
    return ToolResult(
        ok=True,
        output=f"<path>{rel}</path>\n<lines>{total_lines}</lines>\n<content>\n{output}\n</content>",
        summary=f"{rel} - {total_lines} lines{' (truncated)' if trunc else ''}",
        truncated=trunc,
    )


def tool_write_file(cwd: Path, path: str, content: str) -> ToolResult:
    try:
        p = _confine(cwd, path)
    except PermissionError as e:
        return ToolResult.error(str(e))
    existed = p.exists()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except Exception as e:
        return ToolResult.error(str(e))
    rel  = p.relative_to(cwd) if p.is_relative_to(cwd) else p
    verb = "updated" if existed else "created"
    lines = _line_count(content)
    result = ToolResult.success(
        f"{verb} {rel} ({lines} lines)",
        summary=f"{verb} {rel} ({lines} lines)",
    )
    # Soft syntax check - surface obvious errors immediately
    warn = _verify_syntax(p, content)
    if warn:
        result = ToolResult.success(
            result.output + f"\n\n[syntax check] {warn}",
            summary=result.summary + " ⚠ syntax error",
        )
    return result


# Bounds for the whitespace-tolerant edit fallback below. Generous on purpose:
# they are meant to exclude only inputs whose cost is already pathological, not
# to police ordinary edits. A 400-token `old` is a ~40-line snippet and a 512 KB
# file is far past what a model edits in one call.
_WS_FALLBACK_MAX_TOKENS = 400
_WS_FALLBACK_MAX_TEXT = 512 * 1024


# --------------------------------------------------------------------------- #
#  Model-supplied regex: the one place the PATTERN is attacker-controlled       #
# --------------------------------------------------------------------------- #
#
# Everywhere else in this codebase the pattern is ours and the TEXT is hostile,
# which is fixed by de-ambiguating the pattern. `grep` and `search_replace` are
# the inverse: the model hands us the regex. You cannot de-ambiguate a pattern
# you did not write, so this is bounded instead - and it needs BOTH halves,
# because neither alone is sufficient:
#
#   THE TIMEOUT is the only thing that can stop a match already running.
#   Measured, end to end through tool_search_replace with dry_run=True, one file
#   of n spaces and pattern `(\s*)*x`: 0.008 / 0.117 / 1.72s at n = 16 / 20 / 24.
#   That is ~4x per two characters - EXPONENTIAL, extrapolating to ~31 hours at
#   n=40, against a repo that really contains lines with 70 characters of
#   leading whitespace. stdlib `re` cannot be interrupted once inside a match at
#   any price, which is why the engine has to change.
#
#   THE CAPS keep the common path from ever reaching the timeout, and they are
#   not redundant: a timeout still lets an attacker burn the full budget on
#   every file in a glob.
#
# Switching engines is NOT itself the fix, and assuming it was would have moved
# the vulnerability rather than closed it. The two engines have DIFFERENT
# catastrophic sets, neither containing the other (measured):
#     (\s*)*x   on 26 spaces   stdlib 7.0044s   regex 0.0001s
#     (a|a)*$   on 30 a's      stdlib 2.3561s   regex 6.4100s   (both exponential)
# `regex` is immune to the witness above and ~2.7x WORSE on the textbook
# alternation shape - which a model writing a search for alternatives produces
# by accident. So no engine choice is safe on a pattern we did not write, and
# the timeout is load-bearing rather than belt-and-braces.
_MODEL_REGEX_TIMEOUT = 2.0          # seconds per individual match operation
_MODEL_REGEX_MAX_LINE = 64 * 1024   # a single line longer than this is not searched


def _compile_model_pattern(pattern: str, flags):
    """Compile an attacker-supplied pattern on the interruptible engine."""
    try:
        import regex
    except ImportError as exc:      # pragma: no cover - declared dependency
        raise _ModelRegexUnavailable(
            "the 'regex' package is required to run a model-supplied pattern "
            "safely (it is what makes a runaway match interruptible) and it is "
            f"not installed: {exc}. Install localm's dependencies; this tool "
            "will not fall back to an unbounded engine."
        ) from exc
    try:
        return regex.compile(pattern, flags)
    except Exception as exc:
        raise _ModelRegexInvalid(str(exc)) from exc


class _ModelRegexUnavailable(RuntimeError):
    """The interruptible engine is missing, so the pattern will not be run."""


class _ModelRegexInvalid(ValueError):
    """The model supplied a pattern the engine could not compile."""


class _ModelRegexTooSlow(RuntimeError):
    """A model-supplied pattern exceeded its per-match time budget."""


class _ModelRegexTooExpensive(RuntimeError):
    """A model-supplied pattern exhausted memory while matching."""


def _model_regex_flags(ignore_case: bool = False):
    """Translate our flag intent into the `regex` engine's flags."""
    import regex
    flags = regex.MULTILINE
    if ignore_case:
        flags |= regex.IGNORECASE
    return flags


def _run_model_regex(op, *args, **kwargs):
    """Run one match operation under the time budget, or raise _ModelRegexTooSlow / _ModelRegexTooExpensive."""
    import regex
    try:
        return op(*args, timeout=_MODEL_REGEX_TIMEOUT, **kwargs)
    except regex.error as exc:
        raise _ModelRegexInvalid(str(exc)) from exc
    except TimeoutError as exc:
        raise _ModelRegexTooSlow(
            f"the search pattern took longer than {_MODEL_REGEX_TIMEOUT:g}s on a "
            "single file and was stopped. Patterns with a nested quantifier "
            r"(for example `(\s*)*` or `(a|a)*`) can cost time that doubles with "
            "every extra character of input. Simplify the pattern - a plain "
            "substring or a single quantifier is usually what was meant."
        ) from exc
    except MemoryError as exc:
        # A DIFFERENT fact from _ModelRegexTooSlow, so it gets its own message:
        # this is memory exhaustion, not a time-budget overrun, and saying
        # "took longer than Xs" here would be false. It is still the same CLASS
        # of fact as a timeout though - about the PATTERN, not the file - so it
        # is reported the same way (see the two call sites that catch both).
        raise _ModelRegexTooExpensive(
            "the search pattern exhausted memory while matching a single file "
            "and was stopped. Patterns with nested or recursive quantifiers "
            r"(for example `(?:a(?R)?b){e<=1}` or `(\s*)*`) can allocate "
            "unbounded backtracking state as well as unbounded time. Simplify "
            "the pattern - a plain substring or a single quantifier is usually "
            "what was meant."
        ) from exc


def _resolve_edit(text: str, old: str):
    """Find the single region of `text` an edit should replace."""
    idx = text.find(old)
    if idx != -1:
        return idx, idx + len(old), text.count(old), False
    # Whitespace-tolerant fallback. Tokens (the non-whitespace runs) must still
    # match literally and in order; only the whitespace between them is relaxed.
    stripped = old.strip()
    if not stripped:
        return None
    tokens = re.split(r"\s+", stripped)
    # Both `text` and `old` are model-supplied, and this search is O(len(text) x
    # len(pattern)): every start position can match a long prefix before failing.
    # Measured 0.646s for a 16 KB file against an 8 KB `old`, which squares - a
    # 160 KB file would be about a minute of pinned CPU on ONE edit_file call.
    # The exact-match path above (str.find, linear) is unaffected; this is only
    # the whitespace-tolerant FALLBACK, so declining it on a pathological input
    # loses a convenience, not a capability - the caller gets the same "no unique
    # match" answer it already gets whenever the fallback finds none or many.
    if len(tokens) > _WS_FALLBACK_MAX_TOKENS or len(text) > _WS_FALLBACK_MAX_TEXT:
        return None
    pattern = r"\s+".join(re.escape(tok) for tok in tokens)
    matches = list(re.finditer(pattern, text))
    if len(matches) == 1:
        return matches[0].start(), matches[0].end(), 1, True
    return None


def tool_edit_file(cwd: Path, path: str, old: str, new: str) -> ToolResult:
    """Replace the first occurrence of `old` with `new` in `path`."""
    try:
        p = _confine(cwd, path)
    except PermissionError as e:
        return ToolResult.error(str(e))
    if not p.exists():
        return ToolResult.error(f"File not found: {p}")
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return ToolResult.error(str(e))

    # '' is "in" every string, so an empty `old` would silently prepend `new`
    # to the file and report a bogus occurrence count - reject it instead.
    if not old:
        return ToolResult.error(
            "`old` is empty - pass the exact text to replace (read the file "
            "first and copy the snippet). To replace the whole file, use "
            "write_file."
        )

    located = _resolve_edit(text, old)
    if located is None:
        return ToolResult.error(_edit_miss_message(path, old, text))
    start, end, count, tolerant = located
    new_text = text[:start] + new + text[end:]
    try:
        p.write_text(new_text, encoding="utf-8")
    except Exception as e:
        return ToolResult.error(str(e))

    rel = p.relative_to(cwd) if p.is_relative_to(cwd) else p
    note = f" ({count - 1} more occurrence(s) unchanged)" if count > 1 else ""
    tnote = " (matched ignoring whitespace differences)" if tolerant else ""
    result = ToolResult.success(
        f"Replaced 1 occurrence in {rel}{note}{tnote}",
        summary=f"edited {rel}{note}",
    )
    warn = _verify_syntax(p, new_text)
    if warn:
        result = ToolResult.success(
            result.output + f"\n\n[syntax check] {warn}",
            summary=result.summary + " ⚠ syntax error",
        )
    return result


def _edit_miss_message(path: str, old: str, text: str) -> str:
    """The 'string not found' error for an exact-string edit, including the closest-region hint."""
    wanted = textwrap.shorten(repr(old[:120]), width=120)
    nearest = _closest_snippet(text, old)
    hint = f"Closest match in the file:\n{nearest}\n" if nearest else ""
    return (
        f"String not found in {path}.\n"
        f"Looking for: {wanted}\n"
        f"{hint}"
        "Hint: `old` must match the file exactly (whitespace and "
        "indentation included) - read the file first and copy the text."
    )


def _restore_snapshots(snapshots: dict, written: list) -> list[str]:
    """Restore each already-written file from its pre-edit bytes."""
    failures: list[str] = []
    for p in written:
        original = snapshots.get(p)
        if original is None:
            continue
        try:
            p.write_bytes(original)
        except Exception as e:
            failures.append(f"{p} ({e})")
    return failures


def tool_edit_files(cwd: Path, edits: list) -> ToolResult:
    """Apply the same class of exact-string edit as ``edit_file`` across several files in ONE call, all-or-nothing."""
    if not isinstance(edits, list) or not edits:
        return ToolResult.error(
            "`edits` must be a non-empty list of {path, old, new} objects. "
            "For a single file, edit_file is simpler."
        )

    # ---- Phase 1: validate every edit and snapshot every target, no writes ----
    # Validation is complete BEFORE the first byte is written, so the common
    # failure (a typo'd `old`) costs no disk churn and no rollback at all.
    planned: list[tuple[Path, str]] = []   # (abs path, text after this edit)
    snapshots: dict[Path, bytes] = {}      # abs path -> original bytes on disk
    current: dict[Path, str] = {}          # abs path -> text as edits so far left it
    for i, edit in enumerate(edits):
        label = f"edits[{i}]"
        if not isinstance(edit, dict):
            return ToolResult.error(
                f"{label} is not an object - each edit must be "
                '{"path": ..., "old": ..., "new": ...}.')
        path = edit.get("path") or ""
        old  = edit.get("old")
        new  = edit.get("new")
        if not path:
            return ToolResult.error(f"{label} has no `path`.")
        if old is None or new is None:
            return ToolResult.error(
                f"{label} ({path}) needs both `old` and `new`.")
        old, new = str(old), str(new)

        # '' is "in" every string, so an empty `old` would silently prepend
        # `new` to the file - reject it exactly as edit_file does.
        if not old:
            return ToolResult.error(
                f"{label} ({path}): `old` is empty - pass the exact text to "
                "replace (read the file first and copy the snippet). To "
                "replace the whole file, use write_file."
            )
        try:
            p = _confine(cwd, str(path))
        except PermissionError as e:
            return ToolResult.error(f"{label}: {e}")
        if not p.exists():
            return ToolResult.error(f"{label}: File not found: {p}")
        if not p.is_file():
            return ToolResult.error(f"{label}: Not a file: {p}")

        # Later edits must see the text as EARLIER edits in this batch left it,
        # so two edits to the same file compose instead of the second silently
        # missing (or clobbering the first). The SNAPSHOT stays the original
        # on-disk bytes, so a rollback still undoes the whole batch.
        if p in current:
            text = current[p]
        else:
            try:
                snapshots[p] = p.read_bytes()
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                return ToolResult.error(f"{label} ({path}): {e}")

        located = _resolve_edit(text, old)
        if located is None:
            return ToolResult.error(
                f"{label}: " + _edit_miss_message(str(path), old, text))
        start, end, _count, _tolerant = located
        current[p] = text[:start] + new + text[end:]
        planned.append((p, current[p]))

    # ---- Phase 2: write, rolling every file back if any write fails ----
    # One write per FILE (not per edit): several edits to the same file were
    # already composed in phase 1, so the final text is written once.
    targets: list[Path] = []
    for p, _text in planned:
        if p not in targets:
            targets.append(p)
    written: list[Path] = []
    for p in targets:
        try:
            p.write_text(current[p], encoding="utf-8")
            written.append(p)
        except Exception as e:
            # The FAILING file is restored too, not just the ones before it:
            # write_text opens with "w", which truncates before writing, so a
            # failure partway through leaves that file empty or half-written.
            # Restoring it is a no-op when the write never started.
            rollback_errors = _restore_snapshots(snapshots, written + [p])
            failed_rel = p.relative_to(cwd) if p.is_relative_to(cwd) else p
            msg = (f"Failed to write {failed_rel}: {e}\n"
                   f"Rolled back {len(written) + 1} file(s) - no file was left "
                   "partially edited.")
            if rollback_errors:
                # RULE 5: a rollback that itself failed must NEVER be reported as
                # a clean all-or-nothing. Say exactly which files are now suspect.
                msg += ("\nWARNING: the rollback ITSELF failed for: "
                        + "; ".join(rollback_errors)
                        + "\nThese files may hold a partial edit - inspect them "
                        "before continuing.")
            return ToolResult.error(msg)

    # ---- Phase 3: post-write syntax check (same soft check as edit_file) ----
    warnings: list[str] = []
    for p in targets:
        warn = _verify_syntax(p, current[p])
        if warn:
            rel = p.relative_to(cwd) if p.is_relative_to(cwd) else p
            warnings.append(f"{rel}: {warn}")

    touched = len(targets)
    lines = [f"Applied {len(planned)} edit(s) across {touched} file(s):"]
    for p in targets:
        rel = p.relative_to(cwd) if p.is_relative_to(cwd) else p
        lines.append(f"  - {rel}")
    summary = f"edited {touched} file(s) ({len(planned)} edit(s))"
    output  = "\n".join(lines)
    if warnings:
        output += "\n\n[syntax check] " + "\n[syntax check] ".join(warnings)
        summary += " ⚠ syntax error"
    return ToolResult.success(output, summary=summary)


def tool_patch_file(cwd: Path, path: str, diff: str) -> ToolResult:
    """Apply a unified diff to a file."""
    from .._patch import apply_diff, PatchError

    try:
        p = _confine(cwd, path)
    except PermissionError as e:
        return ToolResult.error(str(e))
    if not p.exists():
        return ToolResult.error(f"File not found: {p}")
    if not p.is_file():
        return ToolResult.error(f"Not a file: {p}")

    try:
        original = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return ToolResult.error(str(e))

    try:
        patched = apply_diff(original, diff)
    except PatchError as e:
        return ToolResult.error(str(e))
    except Exception as e:
        return ToolResult.error(f"Unexpected patch error: {e}")

    try:
        p.write_text(patched, encoding="utf-8")
    except Exception as e:
        return ToolResult.error(str(e))

    rel = p.relative_to(cwd) if p.is_relative_to(cwd) else p
    # Count changed lines for summary
    orig_lines  = set(original.splitlines())
    patch_lines = set(patched.splitlines())
    added   = len(patch_lines - orig_lines)
    removed = len(orig_lines - patch_lines)
    result = ToolResult.success(
        f"Patched {rel} (+{added} / -{removed} lines)",
        summary=f"patched {rel} (+{added} / -{removed})",
    )
    warn = _verify_syntax(p, patched)
    if warn:
        result = ToolResult.success(
            result.output + f"\n\n[syntax check] {warn}",
            summary=result.summary + " ⚠ syntax error",
        )
    return result


def tool_list_dir(cwd: Path, path: str = ".") -> ToolResult:
    try:
        p = _confine(cwd, path)
    except PermissionError as e:
        return ToolResult.error(str(e))
    if not p.exists():
        return ToolResult.error(f"Path not found: {p}")
    if not p.is_dir():
        return ToolResult.error(f"Not a directory: {p}")

    entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    lines = []
    for e in entries[:200]:
        if e.is_dir():
            lines.append(f"  {e.name}/")
        else:
            size = e.stat().st_size
            sz   = f"{size/1e6:.1f}M" if size >= 1e6 else f"{size/1e3:.0f}k" if size >= 1e3 else f"{size}B"
            lines.append(f"  {e.name}  [{sz}]")

    rel = p.relative_to(cwd) if p.is_relative_to(cwd) else p
    output = f"{rel}/\n" + "\n".join(lines)
    if len(entries) > 200:
        output += f"\n  ... ({len(entries) - 200} more entries)"
    return ToolResult.success(output, summary=f"{rel}/ - {len(entries)} entries")


def tool_tree(
    cwd: Path,
    path: str = ".",
    max_depth: int = 3,
    max_files: int = 300,
) -> ToolResult:
    """Recursive directory tree with file sizes."""
    try:
        root = _confine(cwd, path)
    except PermissionError as e:
        return ToolResult.error(str(e))
    if not root.exists():
        return ToolResult.error(f"Path not found: {root}")
    if not root.is_dir():
        return ToolResult.error(f"Not a directory: {root}")

    _IGNORE = {
        ".git", "__pycache__", ".mypy_cache", ".pytest_cache",
        "node_modules", ".venv", "venv", ".tox", "dist", "build",
        "*.egg-info", ".localcoder",
    }

    lines: list[str] = []
    total = 0

    def _fmt_size(n: int) -> str:
        if n >= 1_000_000:
            return f"{n/1e6:.1f}M"
        if n >= 1_000:
            return f"{n/1e3:.0f}k"
        return f"{n}B"

    def _walk(dirpath: Path, prefix: str, depth: int) -> None:
        nonlocal total
        if depth > max_depth or total >= max_files:
            return
        try:
            entries = sorted(dirpath.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        except PermissionError:
            return
        entries = [e for e in entries if e.name not in _IGNORE and not e.name.endswith(".egg-info")]
        for i, entry in enumerate(entries):
            if total >= max_files:
                lines.append(
                    f"{prefix}    ... (file limit {max_files} reached - entries "
                    "omitted; raise max_files or point tree at a subdirectory)"
                )
                return
            connector = "└── " if i == len(entries) - 1 else "├── "
            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                extension = "    " if i == len(entries) - 1 else "│   "
                _walk(entry, prefix + extension, depth + 1)
            else:
                total += 1
                try:
                    sz = _fmt_size(entry.stat().st_size)
                except OSError:
                    sz = "?"
                lines.append(f"{prefix}{connector}{entry.name}  [{sz}]")

    rel = root.relative_to(cwd) if root.is_relative_to(cwd) else root
    lines.append(f"{rel}/")
    _walk(root, "", 1)
    return ToolResult.success(
        "\n".join(lines),
        summary=f"tree {rel}/ ({total} files)",
    )


def tool_edit_notebook_cell(
    cwd: Path,
    path: str,
    cell_index: int,
    source: str,
    cell_type: Optional[str] = None,
) -> ToolResult:
    """Replace the source of a single notebook cell."""
    try:
        p = _confine(cwd, path)
    except PermissionError as e:
        return ToolResult.error(str(e))
    if not p.exists():
        return ToolResult.error(f"File not found: {p}")
    if p.suffix != ".ipynb":
        return ToolResult.error(f"Not a notebook: {p}")
    try:
        nb = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return ToolResult.error(f"Failed to parse notebook: {e}")

    cells = nb.get("cells", [])
    if not (0 <= cell_index < len(cells)):
        return ToolResult.error(
            f"Cell index {cell_index} out of range (notebook has {len(cells)} cells)"
        )

    cell = cells[cell_index]
    # source is stored as a list of lines in the format
    cell["source"] = [line if line.endswith("\n") else line + "\n"
                      for line in source.splitlines()]
    # strip trailing newline from the last line (notebook convention)
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")
    if cell_type is not None:
        if cell_type not in ("code", "markdown", "raw"):
            return ToolResult.error(f"Invalid cell_type: {cell_type!r}  (use code/markdown/raw)")
        cell["cell_type"] = cell_type
        if cell_type == "code":
            cell.setdefault("outputs", [])
            cell.setdefault("execution_count", None)

    try:
        p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        return ToolResult.error(f"Failed to write notebook: {e}")

    rel = p.relative_to(cwd) if p.is_relative_to(cwd) else p
    return ToolResult.success(
        f"Updated cell {cell_index} in {rel}",
        summary=f"edited cell {cell_index} in {rel}",
    )


def tool_search_files(cwd: Path, pattern: str, path: str = ".") -> ToolResult:
    try:
        base = _confine(cwd, path)
    except PermissionError as e:
        return ToolResult.error(str(e))
    full_pattern = str(base / pattern) if not Path(pattern).is_absolute() else pattern
    try:
        matches = set(_glob.glob(full_pattern, recursive=True))
        # Bare filename patterns ("*.py") only match the top level - agents
        # almost always mean "anywhere in the project", so search subdirs too
        if not Path(pattern).is_absolute() and "/" not in pattern \
                and "\\" not in pattern and "**" not in pattern:
            matches |= set(_glob.glob(str(base / "**" / pattern), recursive=True))
        matches = sorted(matches)
    except Exception as e:
        return ToolResult.error(str(e))

    # Filter results that escaped cwd via pattern traversal (e.g. ../../etc/*)
    cwd_resolved = cwd.resolve()
    matches = [m for m in matches if Path(m).resolve().is_relative_to(cwd_resolved)]

    if not matches:
        return ToolResult.success("No files matched.", summary="0 matches")

    rel_matches = []
    for m in matches[:200]:
        try:
            rel_matches.append(str(Path(m).relative_to(cwd)))
        except ValueError:
            rel_matches.append(m)

    output = "\n".join(rel_matches)
    trunc  = len(matches) > 200
    if trunc:
        output += f"\n... ({len(matches) - 200} more)"
    return ToolResult(
        ok=True,
        output=output,
        summary=f"{len(matches)} file(s) matched '{pattern}'",
        truncated=trunc,
    )


# ---------------------------------------------------------------------------
#  grep: caps, skip rules, and the streaming matcher
# ---------------------------------------------------------------------------

# Defaults for grep's three caps. Each is overridable PER CALL (max_per_file,
# max_output_lines, max_file_bytes) and persistently via the matching
# coder_grep_* config key; these are only the fallbacks. Every cap that actually
# bites is reported in the output - a cap the caller cannot see is a silent lie
# about coverage (AGENTS.md rule 5), which is why none of them is quiet.
_GREP_MAX_PER_FILE     = 20          # hits shown per file (the rest are COUNTED)
_GREP_MAX_OUTPUT_LINES = 300         # output lines before the sweep stops
_GREP_MAX_FILE_BYTES   = 4 * 1024 * 1024   # per-file size cap

# Bytes sniffed to tell text from binary. A NUL in the first chunk is the same
# heuristic git uses; it needs no new extension table (the known-text extensions
# below skip the sniff entirely).
_GREP_SNIFF_BYTES = 4096


def _grep_config() -> dict:
    """The config dict for grep's cap settings, or {} if config is unreadable."""
    try:
        from localm.config import load_config
        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _grep_cap(arg_value, cfg: dict, config_key: str, fallback: int) -> int:
    """Resolve one grep cap: explicit call arg > config key > module default."""
    if arg_value is not None:
        try:
            return max(0, int(arg_value))
        except (TypeError, ValueError):
            pass
    raw = cfg.get(config_key)
    if raw is not None:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            pass
    return fallback


def _is_probably_binary(fp: Path) -> bool:
    """True if *fp* looks like a binary file."""
    ext = fp.suffix.lower()
    if ext in _SYMBOL_LANGS or ext in _TEXT_EXTS:
        return False
    try:
        with fp.open("rb") as fh:
            return b"\x00" in fh.read(_GREP_SNIFF_BYTES)
    except Exception:
        return False   # unreadable is reported separately, not as "binary"


def _grep_file_hits(fp: Path, rx, context: int, cap: int) -> tuple[list, int]:
    """Stream *fp* one line at a time and collect matches."""
    before: deque = deque(maxlen=context)
    pending: list = []   # [lineno, window, hit_offset, wanted_len] - awaiting trailing context
    hits:    list = []
    total = 0
    with fp.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, 1):
            # Text mode is universal-newline: \r\n and a lone \r are already
            # translated to \n on read, so stripping \n is the whole job here
            # (verified against io.TextIOWrapper, not assumed).
            line = raw.rstrip("\n")
            if pending:
                still = []
                for entry in pending:
                    entry[1].append(line)
                    if len(entry[1]) >= entry[3]:
                        hits.append((entry[0], entry[1], entry[2]))
                    else:
                        still.append(entry)
                pending = still
            # A single absurdly long line is skipped rather than searched: the
            # per-match budget below would stop a runaway, but only after paying
            # it, and paying it once per line of a large file is the accumulation
            # the caps exist to prevent.
            if len(line) > _MODEL_REGEX_MAX_LINE:
                continue
            if _run_model_regex(rx.search, line):
                total += 1
                if cap <= 0 or len(hits) + len(pending) < cap:
                    window = list(before) + [line]
                    hit_offset = len(before)
                    wanted = len(window) + context
                    if len(window) >= wanted:
                        # Already complete (context=0: the hit line IS the whole
                        # window). Closing it here rather than on the next line
                        # is what keeps a context=0 hit a ONE-line window.
                        hits.append((lineno, window, hit_offset))
                    else:
                        pending.append([lineno, window, hit_offset, wanted])
            if context > 0:
                before.append(line)
    # Hits still short of their trailing context at EOF (the old whole-file
    # implementation clamped the slice at the last line - same result).
    hits.extend((entry[0], entry[1], entry[2]) for entry in pending)
    return hits, total


def tool_grep(cwd: Path, pattern: str, path: str = ".", glob: str = "",
              context: int = 2, max_per_file: Optional[int] = None,
              max_output_lines: Optional[int] = None,
              max_file_bytes: Optional[int] = None) -> ToolResult:
    """Search file contents with a regex pattern (pure Python, no external tools)."""
    try:
        base = _confine(cwd, path)
    except PermissionError as e:
        return ToolResult.error(str(e))

    cfg = _grep_config()
    per_file_cap = _grep_cap(max_per_file, cfg, "coder_grep_max_per_file",
                             _GREP_MAX_PER_FILE)
    output_cap   = _grep_cap(max_output_lines, cfg, "coder_grep_max_output_lines",
                             _GREP_MAX_OUTPUT_LINES)
    size_cap     = _grep_cap(max_file_bytes, cfg, "coder_grep_max_file_bytes",
                             _GREP_MAX_FILE_BYTES)
    context = max(0, int(context))

    file_glob = glob or "**/*"
    candidates = sorted(base.glob(file_glob)) if base.is_dir() else [base]

    try:
        rx = _compile_model_pattern(pattern, _model_regex_flags(ignore_case=True))
    except _ModelRegexInvalid as e:
        return ToolResult.error(f"Invalid regex: {e}")
    except _ModelRegexUnavailable as e:
        return ToolResult.error(str(e))

    # ---- Pre-filter: decide what to read BEFORE reading anything ----
    # Ordered cheapest-check-first, because on a real repo most candidates are
    # rejected: name-only checks (free), then ONE stat (regular-file + size),
    # then the cwd-confinement resolve(), then the binary sniff (the only step
    # that opens the file). Enumeration itself is cheap (measured ~0.08s for
    # 4.5k entries); resolve() and stat() are what cost, so they run on
    # survivors, not on every .git object.
    #
    # Confinement still gates every file that is READ: a traversal glob like
    # '../*' makes base.glob() climb above the project root, so the resolved
    # path is checked back inside cwd (the same guard tool_search_files
    # applies - _confine() only protects the `path` arg, not the `glob` arg).
    # It sits BEFORE the sniff so an out-of-cwd file is never even opened.
    # normcase both sides so the prefix test is case-correct on Windows, matching
    # WindowsPath.is_relative_to (which compares case-insensitively); realpath is
    # exactly what Path.resolve() calls, minus the Path object churn - measured
    # ~1.9x cheaper, and this runs once per candidate file.
    cwd_prefix = _os_path.normcase(str(cwd.resolve()))
    cwd_prefix_sep = cwd_prefix + os.sep
    base_depth = len(base.parts)
    # A noise directory the CALLER named in glob= is not noise - they asked for
    # it. Without this, glob="build/**/*.py" returns nothing (build/, dist/,
    # target/, venv/ are all in _SKIP_DIRS), which is a silent coverage loss
    # dressed up as "no matches".
    requested = {seg for seg in file_glob.replace("\\", "/").split("/")
                 if seg in _SKIP_DIRS}
    prune_dirs = _SKIP_DIRS - requested
    files: list[Path] = []
    skipped_noise: set = set()
    skipped_binary = skipped_large = 0
    for fp in candidates:
        # Noise directories are pruned by NAME and only BELOW the search root,
        # so pointing path= AT one (grep inside node_modules) still works.
        noise_dir = next((part for part in fp.parts[base_depth:-1]
                          if part in prune_dirs), None)
        if noise_dir is not None:
            # Count the DIRECTORY, not each entry under it: glob yields
            # directories as well as files, so counting entries reported
            # thousands of "files" not searched when a handful were.
            skipped_noise.add(noise_dir)
            continue
        try:
            st = fp.stat()
        except OSError:
            continue          # vanished or unstattable: nothing to search
        if not _stat.S_ISREG(st.st_mode):
            continue
        if size_cap and st.st_size > size_cap:
            skipped_large += 1
            continue
        real = _os_path.normcase(_os_path.realpath(fp))
        if real != cwd_prefix and not real.startswith(cwd_prefix_sep):
            continue
        if _is_probably_binary(fp):
            skipped_binary += 1
            continue
        files.append(fp)

    results = []
    total_hits = 0
    capped_note = ""
    unreadable = []  # files we could not read, surfaced below so coverage stays honest
    for file_idx, fp in enumerate(files):
        try:
            hits, hit_total = _grep_file_hits(fp, rx, context, per_file_cap)
        except (_ModelRegexTooSlow, _ModelRegexTooExpensive) as e:
            # BEFORE the generic handler on purpose. Falling through to it would
            # file a runaway pattern under "could not read this file" (MemoryError
            # IS an Exception subclass, so it would fall straight into the generic
            # handler below and be reported that way), which is both wrong and
            # unactionable - the user would go looking at the file. A pattern too
            # slow or too memory-hungry to run is a fact about the PATTERN, so it
            # aborts the call and says so rather than degrading into a partial
            # result.
            return ToolResult.error(str(e))
        except Exception:
            # Record (do not silence) the skip so an incomplete match set is not reported as complete.
            try:
                unreadable.append(str(fp.relative_to(cwd)))
            except ValueError:
                unreadable.append(str(fp))
            continue

        if hits:
            try:
                rel = fp.relative_to(cwd)
            except ValueError:
                rel = fp
            results.append(f"## {rel}")
            for lineno, ctx_lines, hit_offset in hits:
                for j, ctx_line in enumerate(ctx_lines):
                    marker = "→ " if j == hit_offset else "  "
                    results.append(f"{marker}{lineno - hit_offset + j:4d}: {ctx_line}")
                results.append("")
            if hit_total > len(hits):
                results.append(f"[... {hit_total - len(hits)} more match(es) in {rel} not shown]")
                results.append("")
            total_hits += hit_total
            if output_cap and len(results) > output_cap:
                remaining = len(files) - file_idx - 1
                if remaining:
                    capped_note = (
                        f"\n[output cap reached - {remaining} more file(s) were NOT "
                        "searched; narrow the search with glob= or path= to cover them]"
                    )
                break

    # Note unreadable files so an incomplete search is not mistaken for complete.
    unreadable_note = ""
    if unreadable:
        shown = ", ".join(unreadable[:20])
        if len(unreadable) > 20:
            shown += f", ... (+{len(unreadable) - 20} more)"
        unreadable_note = (
            f"\n[{len(unreadable)} file(s) could not be read and were not searched: {shown}]"
        )

    # Note the pre-filter skips for the same reason: a faster search that quietly
    # covers less is exactly the "hidden problem" rule 5 forbids.
    skipped_note = ""
    reasons = []
    if skipped_noise:
        # Name the ACTUAL directories skipped, and the way to search them
        # anyway. "(.git, node_modules, ...)" told the caller neither which dir
        # was dropped nor how to get it back.
        named = ", ".join(sorted(skipped_noise)[:6])
        if len(skipped_noise) > 6:
            named += f", +{len(skipped_noise) - 6} more"
        reasons.append(f"noise dir(s) {named} (search one anyway by naming it "
                       "in path= or glob=)")
    if skipped_binary:
        reasons.append(f"{skipped_binary} binary file(s)")
    if skipped_large:
        reasons.append(f"{skipped_large} file(s) over the "
                       f"{size_cap / (1024 * 1024):.1f} MB size cap "
                       "(raise with max_file_bytes=)")
    if reasons:
        skipped_note = ("\n[not searched: " + "; ".join(reasons) + "]")

    if not results:
        msg = f"No matches for '{pattern}'"
        if skipped_note:
            msg += skipped_note
        if unreadable_note:
            msg += unreadable_note
        return ToolResult.success(msg, summary="0 matches")
    if capped_note:
        results.append(capped_note)
    if skipped_note:
        results.append(skipped_note)
    if unreadable_note:
        results.append(unreadable_note)

    output, trunc = _truncate("\n".join(results))
    return ToolResult(
        ok=True,
        output=output,
        summary=f"{total_hits} match(es) for '{pattern}'",
        truncated=trunc,
    )


def tool_search_replace(
    cwd: Path,
    pattern: str,
    replacement: str,
    glob: str = "**/*",
    dry_run: bool = False,
) -> ToolResult:
    """Search for *pattern* across files and replace all matches."""
    try:
        rx = _compile_model_pattern(pattern, _model_regex_flags())
    except _ModelRegexInvalid as e:
        return ToolResult.error(f"Invalid regex: {e}")
    except _ModelRegexUnavailable as e:
        return ToolResult.error(str(e))

    # The same per-file size cap `grep` already honours. Its absence here was a
    # bug on its own account, independent of the timeout: search_replace read
    # every matched file whole with NO ceiling and ran the pattern over the full
    # text, so it was strictly the worse of the two places to hand a hostile
    # pattern. Reusing grep's configured value rather than inventing a second
    # knob, so one setting still means one thing.
    size_cap = _grep_cap(None, _grep_config(), "coder_grep_max_file_bytes",
                         _GREP_MAX_FILE_BYTES)

    # Confine glob results to cwd: a traversal glob like '../*' makes
    # cwd.glob() climb above the project root and would rewrite files outside
    # it. Filter matches back inside cwd before touching anything on disk.
    cwd_resolved = cwd.resolve()
    candidates = sorted(
        p for p in cwd.glob(glob)
        if p.is_file() and p.resolve().is_relative_to(cwd_resolved)
    )
    changes: list[tuple[Path, Path, str, int, bytes]] = []  # (abs, rel, new_text, count, old_bytes)
    unreadable: list[str] = []  # files we could not read; a replacement may be left partial

    oversized: list[str] = []   # skipped by the size cap, reported not silenced
    for fp in candidates:
        try:
            if size_cap > 0 and fp.stat().st_size > size_cap:
                try:
                    oversized.append(str(fp.relative_to(cwd)))
                except ValueError:
                    oversized.append(str(fp))
                continue
            # TWO separate reads, deliberately not one decoded from the other:
            #
            # old_bytes (read_bytes, no translation) is the pre-change
            # snapshot the changed-files/undo tracker needs (ToolResult.
            # changes) - matching execution.py's existing snapshot convention
            # for write_file/edit_file (`abs_path.read_bytes()`), which is
            # also untranslated raw bytes.
            #
            # text (read_text, universal-newline translation ON) is what the
            # regex substitution runs against. Decoding old_bytes directly
            # instead of using read_text() here was tried and is WRONG: on
            # Windows, read_text() normalises CRLF -> LF on the way in and
            # write_text() re-normalises LF -> CRLF on the way out, so the
            # round trip is symmetric and a CRLF file stays CRLF with no
            # extra bytes. Skipping the read-side normalisation left literal
            # \r\n in `text`; write_text()'s own \n -> \r\n translation then
            # ran on top of that untouched \r, inserting an EXTRA \r before
            # every line ending post-substitution ("x = 2\r\n" -> disk bytes
            # "x = 2\r\r\n", which read_text() then reports as "x = 2\n\n" -
            # a spurious blank line on every line the sweep touched).
            #
            # ONE root cause, TWO symptoms in different subsystems: the exact
            # same raw-vs-normalised mismatch, left unhandled, would also make
            # diffutil.compute_search_replace_diff's patch-mode preview show
            # spurious full-line noise on a CRLF file (it diffs this old_bytes
            # against new_text, which is always LF-only) - see the matching
            # normalise step there. Collapsing this back to one read will look
            # correct on Linux (no CRLF to expose it) and silently reopen both.
            old_bytes = fp.read_bytes()
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            # Record (do not silence) the skip so a partial mutation is not reported as complete.
            try:
                unreadable.append(str(fp.relative_to(cwd)))
            except ValueError:
                unreadable.append(str(fp))
            continue
        try:
            matches = _run_model_regex(rx.findall, text)
            if not matches:
                continue
            new_text = _run_model_regex(rx.sub, replacement, text)
        except (_ModelRegexTooSlow, _ModelRegexTooExpensive) as e:
            # Abort before writing anything. This tool MUTATES, so a partial
            # sweep is worse than none: half the glob rewritten and the rest not
            # is a state nobody asked for and the caller cannot tell from success.
            # `changes` is applied after this loop, so returning here leaves the
            # working tree untouched. Before this except also named
            # _ModelRegexTooExpensive, a MemoryError here escaped the try block
            # entirely (no handler matched it) and reached execution.py's
            # generic `except Exception as e: ToolResult.error(f"Tool error: {e}")`
            # - and str(MemoryError()) is '', so it surfaced as an empty
            # "Tool error: " with no indication anything about the pattern.
            return ToolResult.error(str(e))
        try:
            rel = fp.relative_to(cwd)
        except ValueError:
            rel = fp
        changes.append((fp, rel, new_text, len(matches), old_bytes))

    # Warn about unreadable files so the user knows the replacement may be partial
    # (matches in these files were skipped, not applied) - surface, do not silence.
    unreadable_note = ""
    if unreadable:
        shown = ", ".join(unreadable[:20])
        if len(unreadable) > 20:
            shown += f", ... (+{len(unreadable) - 20} more)"
        unreadable_note = (
            f"\n[WARNING: {len(unreadable)} file(s) could not be read and were "
            f"skipped; any matches there were NOT replaced: {shown}]"
        )

    if not changes:
        return ToolResult.success(
            f"No matches for pattern '{pattern}'.{unreadable_note}",
            summary="search_replace - 0 matches",
        )

    total = sum(n for _, _, _, n, _ in changes)
    summary_lines = [
        f"  {rel}  ({n} match{'es' if n != 1 else ''})"
        for _, rel, _, n, _ in changes
    ]
    report = "\n".join(summary_lines)

    if dry_run:
        # Same (path, old_bytes, new_text) shape as the real-apply return
        # below, populated from the SAME matching pass above rather than a
        # second implementation - so patch mode's preview (which calls this
        # tool with dry_run=True to compute a diff without touching disk)
        # can never see a different set of files than a real apply would.
        return ToolResult.success(
            f"[dry-run] Would replace {total} match(es) in {len(changes)} file(s):\n{report}{unreadable_note}",
            summary=f"[dry-run] {total} replacement(s) in {len(changes)} file(s)",
            changes=[(str(rel), old_bytes, new_text)
                    for _, rel, new_text, _, old_bytes in changes],
        )

    written: list[str] = []
    applied: list[tuple[str, bytes, str]] = []  # files actually written, for changes=
    for fp, rel, new_text, _, old_bytes in changes:
        try:
            fp.write_text(new_text, encoding="utf-8")
        except Exception as e:
            # Honesty on a mid-loop write failure: the files written before this
            # one ARE modified on disk. A bare error would read as "nothing
            # changed" - say exactly what was and was not applied. `changes`
            # carries only the files that succeeded, so the caller (the
            # changed-files/undo tracker) can still record and undo the real,
            # partial mutation instead of it going untracked because the call
            # overall reports failure.
            pending = [str(r) for _, r, _, _, _ in changes
                       if str(r) not in written]
            return ToolResult.error(
                f"Failed to write {rel}: {e}\n"
                f"[PARTIAL APPLY: {len(written)} of {len(changes)} file(s) were "
                f"already modified before this failure]\n"
                + (f"Modified: {', '.join(written)}\n" if written else "")
                + f"NOT modified: {', '.join(pending)}",
                changes=applied or None,
            )
        written.append(str(rel))
        applied.append((str(rel), old_bytes, new_text))

    return ToolResult.success(
        f"Replaced {total} match(es) in {len(changes)} file(s):\n{report}{unreadable_note}",
        summary=f"search_replace: {total} replacement(s) in {len(changes)} file(s)",
        changes=applied,
    )
