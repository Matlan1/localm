# SPDX-License-Identifier: AGPL-3.0-or-later
"""File-system tools: read/write/edit/patch, directory listing/tree, notebook cell
edits, and content/glob search-and-replace. All paths are confined to cwd."""

from __future__ import annotations

import difflib
import glob as _glob
import json
import re
import textwrap
from pathlib import Path
from typing import Optional

from .base import ToolResult, _confine, _truncate

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
    """
    Find the file region most similar to a failed `old` string and return a
    short line-numbered snippet of it, so the model can see exactly how the
    real text differs (usually whitespace or a changed identifier).
    Returns "" when nothing is similar enough to help.
    """
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
    """
    Quick offline syntax check for common file types.

    Returns a short warning string on failure, or None if everything looks fine.
    Does not raise - always safe to call after a write.
    """
    suffix = path.suffix.lower()
    if suffix == ".py":
        import py_compile
        import tempfile
        import os as _os
        try:
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w",
                                             encoding="utf-8") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            py_compile.compile(tmp_path, doraise=True)
        except py_compile.PyCompileError as e:
            msg = str(e).replace(tmp_path, str(path))
            return f"Python syntax error: {msg}"
        finally:
            try:
                _os.unlink(tmp_path)
            except Exception:
                pass
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
    """Read a file. *offset* (1-based start line) and *limit* (max lines)
    slice big files so a truncated first read can be followed by targeted
    reads of the middle instead of re-fetching the whole file."""
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

    if old not in text:
        wanted = textwrap.shorten(repr(old[:120]), width=120)
        nearest = _closest_snippet(text, old)
        hint = (f"Closest match in the file:\n{nearest}\n"
                if nearest else "")
        return ToolResult.error(
            f"String not found in {path}.\n"
            f"Looking for: {wanted}\n"
            f"{hint}"
            "Hint: `old` must match the file exactly (whitespace and "
            "indentation included) - read the file first and copy the text."
        )

    count = text.count(old)
    new_text = text.replace(old, new, 1)
    try:
        p.write_text(new_text, encoding="utf-8")
    except Exception as e:
        return ToolResult.error(str(e))

    rel = p.relative_to(cwd) if p.is_relative_to(cwd) else p
    note = f" ({count - 1} more occurrence(s) unchanged)" if count > 1 else ""
    result = ToolResult.success(
        f"Replaced 1 occurrence in {rel}{note}",
        summary=f"edited {rel}{note}",
    )
    warn = _verify_syntax(p, new_text)
    if warn:
        result = ToolResult.success(
            result.output + f"\n\n[syntax check] {warn}",
            summary=result.summary + " ⚠ syntax error",
        )
    return result


def tool_patch_file(cwd: Path, path: str, diff: str) -> ToolResult:
    """
    Apply a unified diff to a file.

    The diff must be in standard ``patch -u`` format::

        --- a/path/to/file.py
        +++ b/path/to/file.py
        @@ -10,4 +10,5 @@
         context line
        -old line
        +new line
        +added line

    File-header lines (``---``/``+++``) are optional but recommended.
    Line numbers in ``@@`` headers are used as hints only - minor off-by-one
    errors are tolerated.  Always read the file before generating the diff.
    """
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


def tool_grep(cwd: Path, pattern: str, path: str = ".", glob: str = "", context: int = 2) -> ToolResult:
    """Search file contents with a regex pattern (pure Python, no external tools)."""
    try:
        base = _confine(cwd, path)
    except PermissionError as e:
        return ToolResult.error(str(e))
    file_glob = glob or "**/*"
    files = sorted(base.glob(file_glob)) if base.is_dir() else [base]

    # Confine glob results to cwd: a traversal glob like '../*' makes
    # base.glob() climb above the project root, so filter the matches back
    # inside cwd (the same guard tool_search_files applies). _confine() only
    # protects the `path` arg, not the `glob` arg.
    cwd_resolved = cwd.resolve()
    files = [f for f in files if f.resolve().is_relative_to(cwd_resolved)]

    try:
        rx = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    except re.error as e:
        return ToolResult.error(f"Invalid regex: {e}")

    results = []
    total_hits = 0
    capped_note = ""
    unreadable = []  # files we could not read, surfaced below so coverage stays honest
    for file_idx, fp in enumerate(files):
        if not fp.is_file():
            continue
        try:
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            # Record (do not silence) the skip so an incomplete match set is not reported as complete.
            try:
                unreadable.append(str(fp.relative_to(cwd)))
            except ValueError:
                unreadable.append(str(fp))
            continue

        hits = []
        for i, line in enumerate(lines):
            if rx.search(line):
                start = max(0, i - context)
                end   = min(len(lines), i + context + 1)
                hits.append((i + 1, lines[start:end], i - start))

        if hits:
            try:
                rel = fp.relative_to(cwd)
            except ValueError:
                rel = fp
            results.append(f"## {rel}")
            for lineno, ctx_lines, hit_offset in hits[:20]:
                for j, ctx_line in enumerate(ctx_lines):
                    marker = "→ " if j == hit_offset else "  "
                    results.append(f"{marker}{lineno - hit_offset + j:4d}: {ctx_line}")
                results.append("")
            if len(hits) > 20:
                results.append(f"[... {len(hits) - 20} more match(es) in {rel} not shown]")
                results.append("")
            total_hits += len(hits)
            if len(results) > 300:
                remaining = sum(1 for f in files[file_idx + 1:] if f.is_file())
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

    if not results:
        msg = f"No matches for '{pattern}'"
        if unreadable_note:
            msg += unreadable_note
        return ToolResult.success(msg, summary="0 matches")
    if capped_note:
        results.append(capped_note)
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
    """
    Search for *pattern* across files and replace all matches.

    Parameters
    ----------
    pattern:
        Python regex.  Applied with ``re.MULTILINE``.
    replacement:
        Replacement string (supports ``\\1`` back-references).
    glob:
        File filter applied relative to *cwd* (default: all files).
    dry_run:
        When True, report what would change without modifying anything.
    """
    try:
        rx = re.compile(pattern, re.MULTILINE)
    except re.error as e:
        return ToolResult.error(f"Invalid regex: {e}")

    # Confine glob results to cwd: a traversal glob like '../*' makes
    # cwd.glob() climb above the project root and would rewrite files outside
    # it. Filter matches back inside cwd before touching anything on disk.
    cwd_resolved = cwd.resolve()
    candidates = sorted(
        p for p in cwd.glob(glob)
        if p.is_file() and p.resolve().is_relative_to(cwd_resolved)
    )
    changes: list[tuple[Path, Path, str, int]] = []  # (abs, rel, new_text, count)
    unreadable: list[str] = []  # files we could not read; a replacement may be left partial

    for fp in candidates:
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            # Record (do not silence) the skip so a partial mutation is not reported as complete.
            try:
                unreadable.append(str(fp.relative_to(cwd)))
            except ValueError:
                unreadable.append(str(fp))
            continue
        matches = rx.findall(text)
        if not matches:
            continue
        new_text = rx.sub(replacement, text)
        try:
            rel = fp.relative_to(cwd)
        except ValueError:
            rel = fp
        changes.append((fp, rel, new_text, len(matches)))

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

    total = sum(n for _, _, _, n in changes)
    summary_lines = [
        f"  {rel}  ({n} match{'es' if n != 1 else ''})"
        for _, rel, _, n in changes
    ]
    report = "\n".join(summary_lines)

    if dry_run:
        return ToolResult.success(
            f"[dry-run] Would replace {total} match(es) in {len(changes)} file(s):\n{report}{unreadable_note}",
            summary=f"[dry-run] {total} replacement(s) in {len(changes)} file(s)",
        )

    written: list[str] = []
    for fp, rel, new_text, _ in changes:
        try:
            fp.write_text(new_text, encoding="utf-8")
        except Exception as e:
            # Honesty on a mid-loop write failure: the files written before this
            # one ARE modified on disk. A bare error would read as "nothing
            # changed" - say exactly what was and was not applied.
            pending = [str(r) for _, r, _, _ in changes
                       if str(r) not in written]
            return ToolResult.error(
                f"Failed to write {rel}: {e}\n"
                f"[PARTIAL APPLY: {len(written)} of {len(changes)} file(s) were "
                f"already modified before this failure]\n"
                + (f"Modified: {', '.join(written)}\n" if written else "")
                + f"NOT modified: {', '.join(pending)}"
            )
        written.append(str(rel))

    return ToolResult.success(
        f"Replaced {total} match(es) in {len(changes)} file(s):\n{report}{unreadable_note}",
        summary=f"search_replace: {total} replacement(s) in {len(changes)} file(s)",
    )
