"""
Tool implementations for the coding agent.

Each tool is a plain function that takes a ``cwd`` (working directory Path)
plus keyword arguments matching the tool's parameter schema, and returns
a ``ToolResult``.

Tools are registered in the TOOL_REGISTRY dict at the bottom of this file.
"""

from __future__ import annotations

import glob as _glob
import json
import os
import re
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from .agent import Agent


# ---------------------------------------------------------------------------
#  ToolResult
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    ok:      bool
    output:  str
    summary: str = ""       # one-line display shown in the console
    truncated: bool = False

    @classmethod
    def success(cls, output: str, summary: str = "") -> "ToolResult":
        return cls(ok=True, output=output, summary=summary)

    @classmethod
    def error(cls, message: str) -> "ToolResult":
        return cls(ok=False, output=message, summary=f"ERROR: {message}")

    def to_xml(self, tool_name: str) -> str:
        status = "ok" if self.ok else "error"
        trunc  = ' truncated="true"' if self.truncated else ""
        return (
            f'<tool_result name="{tool_name}" status="{status}"{trunc}>\n'
            f"{self.output}\n"
            f"</tool_result>"
        )


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

_MAX_OUTPUT = 8_000   # chars — truncate large outputs to spare context

def _truncate(text: str, max_chars: int = _MAX_OUTPUT) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    half = max_chars // 2
    return (
        text[:half] + f"\n\n... [{len(text) - max_chars} chars truncated] ...\n\n" + text[-half:],
        True,
    )


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


def _resolve(cwd: Path, path: str) -> Path:
    """Resolve a possibly-relative path against cwd."""
    p = Path(path)
    return p if p.is_absolute() else cwd / p


def _confine(cwd: Path, path: str) -> Path:
    """
    Resolve *path* against *cwd* and verify it stays inside *cwd*.

    Raises ``PermissionError`` with a clear message if the resolved path
    escapes the working directory (path traversal attempt or accidental
    absolute path outside the project root).
    """
    resolved = (cwd / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    cwd_resolved = cwd.resolve()
    if not resolved.is_relative_to(cwd_resolved):
        raise PermissionError(
            f"'{path}' resolves outside the working directory '{cwd_resolved}'. "
            "All file operations must stay within the project root."
        )
    return resolved


def _line_count(text: str) -> int:
    return text.count("\n") + 1


# ---------------------------------------------------------------------------
#  Tool functions
# ---------------------------------------------------------------------------

def tool_read_file(cwd: Path, path: str) -> ToolResult:
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
    if p.suffix == ".ipynb":
        try:
            nb   = json.loads(raw)
            text = _render_notebook(nb)
            n    = len(nb.get("cells", []))
            output, trunc = _truncate(text)
            return ToolResult(
                ok=True,
                output=f"<path>{rel}</path>\n<cells>{n}</cells>\n<content>\n{output}\n</content>",
                summary=f"{rel} — {n} cells{' (truncated)' if trunc else ''}",
                truncated=trunc,
            )
        except Exception:
            pass  # fall through to plain-text read if JSON is malformed

    lines = _line_count(raw)
    output, trunc = _truncate(raw)
    return ToolResult(
        ok=True,
        output=f"<path>{rel}</path>\n<lines>{lines}</lines>\n<content>\n{output}\n</content>",
        summary=f"{rel} — {lines} lines{' (truncated)' if trunc else ''}",
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
    # Soft syntax check — surface obvious errors immediately
    warn = _verify_syntax(p, content)
    if warn:
        result = ToolResult.success(
            result.output + f"\n\n[syntax check] {warn}",
            summary=result.summary + " ⚠ syntax error",
        )
    return result


def _verify_syntax(path: Path, content: str) -> Optional[str]:
    """
    Quick offline syntax check for common file types.

    Returns a short warning string on failure, or None if everything looks fine.
    Does not raise — always safe to call after a write.
    """
    suffix = path.suffix.lower()
    if suffix == ".py":
        import py_compile, tempfile, os as _os
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

    if old not in text:
        snippet = textwrap.shorten(repr(old[:120]), width=120)
        return ToolResult.error(
            f"String not found in {path}.\n"
            f"Looking for: {snippet}\n"
            f"Hint: read the file first to get the exact text."
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
    Line numbers in ``@@`` headers are used as hints only — minor off-by-one
    errors are tolerated.  Always read the file before generating the diff.
    """
    from ._patch import apply_diff, PatchError

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


def _needs_shell(command: str) -> bool:
    """Return True when the command uses shell operators that require a real shell."""
    # Characters that only mean something inside a shell
    shell_chars = {"&", "|", ";", "<", ">", "$", "`", "~", "(", ")", "{", "}", "!", "\\", "*", "?"}
    in_single = False
    in_double = False
    for ch in command:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double and ch in shell_chars:
            return True
    return False


def tool_run_shell(
    cwd: Path,
    command: str,
    timeout: int = 30,
    _privacy: bool = False,
) -> ToolResult:
    """
    Execute a shell command.

    When the command contains no shell operators (pipes, redirects, globs,
    variable expansion, etc.) it is parsed with ``shlex.split`` and run as
    a plain argument list — no shell injection possible.  Otherwise it falls
    back to the system shell (cmd /C on Windows, /bin/sh -c elsewhere).

    In privacy mode (``_privacy=True``) the subprocess environment has
    shell-history variables zeroed.
    """
    import shlex

    shell_cmd: list[str]
    if _needs_shell(command):
        # Complex command — must go through a shell
        if sys.platform == "win32":
            shell_cmd = ["cmd", "/C", command]
        else:
            shell_cmd = ["/bin/sh", "-c", command]
    else:
        try:
            shell_cmd = shlex.split(command, posix=(sys.platform != "win32"))
        except ValueError:
            # Malformed quoting — fall back to shell
            if sys.platform == "win32":
                shell_cmd = ["cmd", "/C", command]
            else:
                shell_cmd = ["/bin/sh", "-c", command]
        else:
            # Shell builtins (echo, dir, type, …) have no executable on disk —
            # argument-list mode would fail with "file not found". Detect via
            # PATH lookup and route those through the shell instead.
            import shutil as _shutil
            if not shell_cmd or _shutil.which(shell_cmd[0]) is None:
                if sys.platform == "win32":
                    shell_cmd = ["cmd", "/C", command]
                else:
                    shell_cmd = ["/bin/sh", "-c", command]

    env: dict | None = None
    if _privacy:
        from .privacy import subprocess_privacy_env
        env = subprocess_privacy_env()

    try:
        proc = subprocess.run(
            shell_cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except subprocess.TimeoutExpired:
        return ToolResult.error(f"Command timed out after {timeout}s")
    except Exception as e:
        return ToolResult.error(str(e))

    combined = ""
    if proc.stdout:
        combined += proc.stdout
    if proc.stderr:
        combined += ("\n" if combined else "") + "STDERR:\n" + proc.stderr

    combined = combined.strip() or "(no output)"
    output, trunc = _truncate(combined)
    rc = proc.returncode
    status = "ok" if rc == 0 else f"exit {rc}"

    return ToolResult(
        ok=(rc == 0),
        output=f"<exit_code>{rc}</exit_code>\n<output>\n{output}\n</output>",
        summary=f"$ {command[:60]}  [{status}]",
        truncated=trunc,
    )


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
    return ToolResult.success(output, summary=f"{rel}/ — {len(entries)} entries")


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
                lines.append(f"{prefix}    ... (limit reached)")
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
        # Bare filename patterns ("*.py") only match the top level — agents
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

    try:
        rx = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    except re.error as e:
        return ToolResult.error(f"Invalid regex: {e}")

    results = []
    total_hits = 0
    for fp in files:
        if not fp.is_file():
            continue
        try:
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
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
            total_hits += len(hits)
            if len(results) > 300:
                break

    if not results:
        return ToolResult.success(f"No matches for '{pattern}'", summary="0 matches")

    output, trunc = _truncate("\n".join(results))
    return ToolResult(
        ok=True,
        output=output,
        summary=f"{total_hits} match(es) for '{pattern}'",
        truncated=trunc,
    )


# ---------------------------------------------------------------------------
#  spawn_agent — sub-agent tool
# ---------------------------------------------------------------------------

def tool_spawn_agent(
    cwd: Path,
    task: str,
    name: str = "subagent",
    files: Optional[list] = None,
    model: Optional[str] = None,
    max_turns: int = 10,
    _parent_agent: Optional[Any] = None,
) -> ToolResult:
    """
    Spawn a child Agent with a focused task.

    The child inherits the parent's backend (or uses ``model`` if given),
    gets ``files`` pre-loaded into its first user message, and runs until
    it produces a final answer or ``max_turns`` is reached.

    Returns the child agent's final response as a string.
    """
    if _parent_agent is None:
        return ToolResult.error("spawn_agent requires a running parent agent")

    from .agent import Agent

    backend = _parent_agent.backend
    if model and model != backend.model_id:
        from .backends.http import make_localm_backend
        raw_url = getattr(backend, "_base_url", "http://127.0.0.1:8642/v1")
        try:
            port = int(raw_url.split(":")[-1].split("/")[0])
        except Exception:
            port = 8642
        try:
            backend = make_localm_backend(model, port=port)
        except Exception:
            backend = _parent_agent.backend

    preload_text = ""
    if files:
        for fp in files:
            r = tool_read_file(cwd, fp)
            preload_text += f"\n{r.output}\n"

    full_task = task
    if preload_text:
        full_task = f"Context files:\n{preload_text}\n\nTask:\n{task}"

    from .audit import SessionMode as _SessionMode
    inherited_mode = getattr(_parent_agent, "mode", _SessionMode.PRIVACY)

    child = Agent(
        backend=backend,
        cwd=cwd,
        name=name,
        max_turns=max_turns,
        verbose=False,
        auto_approve=True,
        parent=_parent_agent,
        mode=inherited_mode,
    )
    result_text = child.run_task(full_task)
    turns_used  = child.turns

    return ToolResult.success(
        result_text,
        summary=f"sub-agent '{name}' finished in {turns_used} turn(s)",
    )


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

    In privacy mode (``_privacy=True``) a one-line network audit message is
    emitted to stderr before the request so the user can see outbound URLs.
    """
    import html.parser
    import urllib.error
    import urllib.request

    if _privacy:
        import sys as _sys
        print(f"[localm privacy] fetch_url: {url}", file=_sys.stderr, flush=True)

    class _HTMLStripper(html.parser.HTMLParser):
        _SKIP = {"script", "style", "head", "meta", "link", "noscript"}

        def __init__(self):
            super().__init__(convert_charrefs=True)
            self._buf: list[str] = []
            self._skip = 0

        def handle_starttag(self, tag, attrs):
            if tag.lower() in self._SKIP:
                self._skip += 1

        def handle_endtag(self, tag):
            if tag.lower() in self._SKIP and self._skip:
                self._skip -= 1

        def handle_data(self, data):
            if not self._skip:
                self._buf.append(data)

        def get_text(self) -> str:
            import re
            raw = "".join(self._buf)
            # Collapse excessive blank lines
            return re.sub(r"\n{3,}", "\n\n", raw).strip()

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "localm/0.1 (fetch_url tool)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(1_000_000).decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        return ToolResult.error(f"Could not fetch {url}: {e}")
    except Exception as e:
        return ToolResult.error(str(e))

    if "html" in content_type.lower():
        stripper = _HTMLStripper()
        stripper.feed(raw)
        text = stripper.get_text()
    else:
        text = raw.strip()

    output, trunc = _truncate(text, max_chars)
    return ToolResult(
        ok=True,
        output=f"<url>{url}</url>\n<content>\n{output}\n</content>",
        summary=f"fetched {url[:60]} ({len(text):,} chars{', truncated' if trunc else ''})",
        truncated=trunc,
    )


def _git(cwd: Path, *args: str, timeout: int = 10) -> tuple[str, bool]:
    """Run a git command and return (output, ok)."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
        out = (proc.stdout + proc.stderr).strip() or "(no output)"
        return out, proc.returncode == 0
    except FileNotFoundError:
        return "git not found in PATH", False
    except subprocess.TimeoutExpired:
        return f"git {args[0]} timed out", False
    except Exception as e:
        return str(e), False


def tool_git_status(cwd: Path) -> ToolResult:
    """Return the output of `git status --short` in the working directory."""
    out, ok = _git(cwd, "status", "--short", "--branch")
    return ToolResult(ok=ok, output=out, summary=f"git status ({len(out.splitlines())} lines)")


def tool_git_diff(cwd: Path, path: str = "", staged: bool = False) -> ToolResult:
    """
    Return `git diff` output.

    Parameters
    ----------
    path   : limit diff to this file or directory (optional)
    staged : if True, show staged changes (`git diff --cached`)
    """
    args = ["diff", "--stat", "-p"]
    if staged:
        args.append("--cached")
    if path:
        args += ["--", path]
    out, ok = _git(cwd, *args, timeout=15)
    out, trunc = _truncate(out)
    return ToolResult(ok=ok, output=out, summary="git diff" + (" --cached" if staged else ""), truncated=trunc)


def tool_git_log(cwd: Path, n: int = 10, path: str = "") -> ToolResult:
    """Return the last n commits as a compact log."""
    args = ["log", f"--max-count={n}", "--oneline", "--decorate"]
    if path:
        args += ["--", path]
    out, ok = _git(cwd, *args)
    return ToolResult(ok=ok, output=out, summary=f"git log -{n}")


def tool_generate_image(
    cwd: Path,
    prompt: str,
    output_path: str = "output.png",
    guidance: Optional[float] = None,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    lora_name: Optional[str] = None,
    lora_strength_model: float = 1.0,
    lora_strength_clip: float = 0.5,
    input_image: Optional[str] = None,
    denoise: Optional[float] = None,
) -> ToolResult:
    """Thin wrapper — delegates to localm.image_gen.comfy.generate_image."""
    import os
    from localm.image_gen.comfy import generate_image

    try:
        out_p = _confine(cwd, output_path)
        input_p = _confine(cwd, input_image) if input_image else None
    except PermissionError as e:
        return ToolResult.error(str(e))
    api_url = os.environ.get("FLUX_API_URL", "http://127.0.0.1:8188")
    ok, message = generate_image(
        prompt, out_p,
        api_url=api_url,
        guidance=guidance,
        negative_prompt=negative_prompt,
        seed=seed,
        lora_name=lora_name,
        lora_strength_model=lora_strength_model,
        lora_strength_clip=lora_strength_clip,
        input_image=input_p,
        denoise=denoise,
    )
    if ok:
        rel = out_p.relative_to(cwd) if out_p.is_relative_to(cwd) else out_p
        return ToolResult.success(message, summary=f"generated {rel}")
    return ToolResult.error(message)


# ---------------------------------------------------------------------------
#  New tools: test runner, git write ops, search-replace
# ---------------------------------------------------------------------------

def _detect_test_runner(cwd: Path) -> list[str]:
    """Return the command list for the most appropriate test runner in *cwd*."""
    if (cwd / "Cargo.toml").exists():
        return ["cargo", "test", "--color=never"]
    if (cwd / "go.mod").exists():
        return ["go", "test", "./..."]
    if (cwd / "package.json").exists():
        lock = "yarn" if (cwd / "yarn.lock").exists() else "npm"
        return [lock, "test", "--passWithNoTests"]
    # Python — prefer pytest; fall back to unittest
    return ["python", "-m", "pytest", "--tb=short", "-q", "--no-header"]


def tool_run_tests(
    cwd: Path,
    runner: str = "auto",
    path: str = ".",
    extra_args: str = "",
) -> ToolResult:
    """
    Run the project's test suite and return the result.

    Parameters
    ----------
    runner:
        ``auto`` (default) detects from project files; or specify
        ``pytest``, ``cargo``, ``go``, ``npm``, ``yarn`` explicitly.
    path:
        Subdirectory or file to limit the test run (default: whole project).
    extra_args:
        Additional arguments appended verbatim to the test command.
    """
    if runner == "auto":
        cmd = _detect_test_runner(cwd)
    elif runner == "pytest":
        cmd = ["python", "-m", "pytest", "--tb=short", "-q", "--no-header"]
    elif runner == "cargo":
        cmd = ["cargo", "test", "--color=never"]
    elif runner == "go":
        cmd = ["go", "test", "./..."]
    elif runner in ("npm", "yarn"):
        cmd = [runner, "test", "--passWithNoTests"]
    else:
        return ToolResult.error(
            f"Unknown runner '{runner}'. Use: auto, pytest, cargo, go, npm, yarn"
        )

    if path and path != ".":
        cmd.append(path)
    if extra_args:
        cmd.extend(extra_args.split())

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return ToolResult.error(
            f"Test runner not found: {cmd[0]}. "
            "Make sure it is installed and on PATH."
        )
    except subprocess.TimeoutExpired:
        return ToolResult.error("Test run timed out after 120s")
    except Exception as e:
        return ToolResult.error(str(e))

    combined = ""
    if proc.stdout:
        combined += proc.stdout
    if proc.stderr:
        combined += ("\n" if combined else "") + proc.stderr
    combined = combined.strip() or "(no output)"
    output, trunc = _truncate(combined)

    ok = proc.returncode == 0
    if ok:
        status = "passed"
    elif proc.returncode == 5 and "pytest" in " ".join(cmd):
        # pytest exit 5 = no tests collected — not a failure, but worth
        # distinguishing so the agent doesn't "fix" passing code
        ok = True
        status = "no tests found"
    else:
        status = f"failed (exit {proc.returncode})"
    return ToolResult(
        ok=ok,
        output=f"<runner>{' '.join(cmd[:2])}</runner>\n"
               f"<status>{status}</status>\n"
               f"<output>\n{output}\n</output>",
        summary=f"tests {status}",
        truncated=trunc,
    )


def tool_git_commit(
    cwd: Path,
    message: str,
    files: Optional[list] = None,
) -> ToolResult:
    """
    Stage files and create a git commit.

    Parameters
    ----------
    message:
        Commit message.
    files:
        List of file paths to stage.  When omitted, stages all tracked
        modifications (``git add -A``).
    """
    # Stage
    if files:
        for f in files:
            out, ok = _git(cwd, "add", "--", f)
            if not ok:
                return ToolResult.error(f"Failed to stage '{f}': {out}")
    else:
        out, ok = _git(cwd, "add", "-A")
        if not ok:
            return ToolResult.error(f"Failed to stage changes: {out}")

    # Commit
    out, ok = _git(cwd, "commit", "-m", message, timeout=30)
    if not ok:
        if "nothing to commit" in out.lower():
            return ToolResult.success(
                "Nothing to commit, working tree clean.",
                summary="git commit — nothing to commit",
            )
        return ToolResult.error(f"git commit failed: {out}")

    return ToolResult.success(out, summary=f"git commit: {message[:60]}")


def tool_git_push(
    cwd: Path,
    remote: str = "origin",
    branch: str = "",
) -> ToolResult:
    """Push the current branch to a remote."""
    args = ["push", remote]
    if branch:
        args.append(branch)
    out, ok = _git(cwd, *args, timeout=60)
    if not ok:
        return ToolResult.error(f"git push failed: {out}")
    return ToolResult.success(out, summary=f"git push {remote}")


def tool_git_create_branch(
    cwd: Path,
    name: str,
    checkout: bool = True,
) -> ToolResult:
    """
    Create a new git branch.

    Parameters
    ----------
    name:
        Branch name, e.g. ``feat/my-feature``.
    checkout:
        Switch to the new branch after creating it (default: True).
    """
    if checkout:
        out, ok = _git(cwd, "checkout", "-b", name)
    else:
        out, ok = _git(cwd, "branch", name)
    if not ok:
        return ToolResult.error(f"git branch failed: {out}")
    action = "created and checked out" if checkout else "created"
    return ToolResult.success(out or f"Branch '{name}' {action}.", summary=f"git branch {name}")


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

    candidates = sorted(p for p in cwd.glob(glob) if p.is_file())
    changes: list[tuple[Path, Path, str, int]] = []  # (abs, rel, new_text, count)

    for fp in candidates:
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
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

    if not changes:
        return ToolResult.success(
            f"No matches for pattern '{pattern}'.",
            summary="search_replace — 0 matches",
        )

    total = sum(n for _, _, _, n in changes)
    summary_lines = [
        f"  {rel}  ({n} match{'es' if n != 1 else ''})"
        for _, rel, _, n in changes
    ]
    report = "\n".join(summary_lines)

    if dry_run:
        return ToolResult.success(
            f"[dry-run] Would replace {total} match(es) in {len(changes)} file(s):\n{report}",
            summary=f"[dry-run] {total} replacement(s) in {len(changes)} file(s)",
        )

    for fp, rel, new_text, _ in changes:
        try:
            fp.write_text(new_text, encoding="utf-8")
        except Exception as e:
            return ToolResult.error(f"Failed to write {rel}: {e}")

    return ToolResult.success(
        f"Replaced {total} match(es) in {len(changes)} file(s):\n{report}",
        summary=f"search_replace: {total} replacement(s) in {len(changes)} file(s)",
    )


# ---------------------------------------------------------------------------
#  Tool registry
# ---------------------------------------------------------------------------

@dataclass
class ToolDef:
    name:        str
    fn:          Callable
    description: str
    params:      dict
    destructive: bool = False


TOOL_REGISTRY: dict[str, ToolDef] = {
    "read_file": ToolDef(
        name="read_file",
        fn=tool_read_file,
        description="Read the contents of a file.",
        params={"path": {"type": "string", "description": "File path (relative to cwd)", "required": True}},
    ),
    "write_file": ToolDef(
        name="write_file",
        fn=tool_write_file,
        description="Write or overwrite a file with new content.",
        params={
            "path":    {"type": "string", "description": "File path",         "required": True},
            "content": {"type": "string", "description": "Full file content", "required": True},
        },
        destructive=True,
    ),
    "edit_file": ToolDef(
        name="edit_file",
        fn=tool_edit_file,
        description="Replace the first occurrence of `old` text with `new` text in a file.",
        params={
            "path": {"type": "string", "description": "File path",            "required": True},
            "old":  {"type": "string", "description": "Exact text to replace","required": True},
            "new":  {"type": "string", "description": "Replacement text",     "required": True},
        },
        destructive=True,
    ),
    "patch_file": ToolDef(
        name="patch_file",
        fn=tool_patch_file,
        description=(
            "Apply a unified diff (patch -u format) to a file. "
            "More reliable than edit_file for multi-hunk or large changes. "
            "Always read_file first so line numbers are accurate."
        ),
        params={
            "path": {"type": "string", "description": "File path (relative to cwd)",              "required": True},
            "diff": {"type": "string", "description": "Unified diff string (patch -u format)",    "required": True},
        },
        destructive=True,
    ),
    "run_shell": ToolDef(
        name="run_shell",
        fn=tool_run_shell,
        description="Execute a shell command in the working directory.",
        params={
            "command": {"type": "string", "description": "Shell command",      "required": True},
            "timeout": {"type": "int",    "description": "Timeout in seconds", "required": False},
        },
        destructive=True,
    ),
    "list_dir": ToolDef(
        name="list_dir",
        fn=tool_list_dir,
        description="List the contents of a directory.",
        params={"path": {"type": "string", "description": "Directory path (default: .)", "required": False}},
    ),
    "tree": ToolDef(
        name="tree",
        fn=tool_tree,
        description="Recursive directory tree with file sizes. Skips common noise dirs (.git, __pycache__, node_modules, etc.).",
        params={
            "path":      {"type": "string", "description": "Root directory (default: .)", "required": False},
            "max_depth": {"type": "int",    "description": "How many levels deep to recurse (default: 3)", "required": False},
            "max_files": {"type": "int",    "description": "Stop after this many files (default: 300)", "required": False},
        },
    ),
    "edit_notebook_cell": ToolDef(
        name="edit_notebook_cell",
        fn=tool_edit_notebook_cell,
        description="Replace the source of a single cell in a Jupyter notebook (.ipynb). Use read_file first to see cell indices.",
        params={
            "path":       {"type": "string", "description": "Path to the .ipynb file.", "required": True},
            "cell_index": {"type": "int",    "description": "Zero-based index of the cell to edit.", "required": True},
            "source":     {"type": "string", "description": "New source code or markdown text for the cell.", "required": True},
            "cell_type":  {"type": "string", "description": "Override cell type: code, markdown, or raw (optional).", "required": False},
        },
        destructive=True,
    ),
    "search_files": ToolDef(
        name="search_files",
        fn=tool_search_files,
        description="Find files matching a glob pattern.",
        params={
            "pattern": {"type": "string", "description": "Glob pattern, e.g. **/*.py", "required": True},
            "path":    {"type": "string", "description": "Root directory to search",    "required": False},
        },
    ),
    "grep": ToolDef(
        name="grep",
        fn=tool_grep,
        description="Search file contents with a regex pattern.",
        params={
            "pattern": {"type": "string", "description": "Regex pattern",                "required": True},
            "path":    {"type": "string", "description": "File or directory to search",   "required": False},
            "glob":    {"type": "string", "description": "File filter, e.g. **/*.py",     "required": False},
            "context": {"type": "int",    "description": "Lines of context (default 2)",  "required": False},
        },
    ),
    "git_status": ToolDef(
        name="git_status",
        fn=tool_git_status,
        description="Show working-tree status (git status --short --branch).",
        params={},
    ),
    "git_diff": ToolDef(
        name="git_diff",
        fn=tool_git_diff,
        description="Show git diff (unstaged by default; pass staged=true for staged changes).",
        params={
            "path":   {"type": "string", "description": "Limit to this file/dir",    "required": False},
            "staged": {"type": "bool",   "description": "Show staged diff (default false)", "required": False},
        },
    ),
    "git_log": ToolDef(
        name="git_log",
        fn=tool_git_log,
        description="Show recent commits (oneline format).",
        params={
            "n":    {"type": "int",    "description": "Number of commits (default 10)", "required": False},
            "path": {"type": "string", "description": "Limit to this file/dir",         "required": False},
        },
    ),
    "fetch_url": ToolDef(
        name="fetch_url",
        fn=tool_fetch_url,
        description="Fetch a URL and return its plain-text content (HTML stripped).",
        params={
            "url":       {"type": "string", "description": "Full URL to fetch",               "required": True},
            "max_chars": {"type": "int",    "description": "Truncate output (default 8000)",  "required": False},
        },
    ),
    "spawn_agent": ToolDef(
        name="spawn_agent",
        fn=tool_spawn_agent,
        description="Spawn a focused sub-agent to handle a specific sub-task and return its result.",
        params={
            "task":      {"type": "string", "description": "What the sub-agent should do",     "required": True},
            "name":      {"type": "string", "description": "Short name for this sub-agent",     "required": False},
            "files":     {"type": "array",  "description": "Files to pre-load into sub-agent", "required": False},
            "model":     {"type": "string", "description": "Override model for sub-agent",      "required": False},
            "max_turns": {"type": "int",    "description": "Max iterations (default 10)",       "required": False},
        },
    ),
    "generate_image": ToolDef(
        name="generate_image",
        fn=tool_generate_image,
        description=(
            "Generate or refine an image using the local FLUX model. "
            "Without input_image: generates from scratch (txt2img). "
            "With input_image: refines an existing image guided by the prompt (img2img)."
        ),
        params={
            "prompt":       {"type": "string", "description": "What to generate. For img2img, describe what to change rather than the full scene.", "required": True},
            "output_path":  {"type": "string", "description": "Path to save the result (default: output.png)", "required": False},
            "input_image":  {"type": "string", "description": "Path to an existing image to use as the starting point (img2img mode).", "required": False},
            "denoise":      {"type": "float",  "description": "img2img only — how much to change the input (0.0=no change, 1.0=completely new). Default 0.75.", "required": False},
            "seed":         {"type": "int",    "description": "Noise seed for reproducible output. Each result reports its seed; pass it back to reproduce or tweak.", "required": False},
            "guidance":        {"type": "float",  "description": "Guidance scale (default: 3.5). Lower values (2.5-3.0) improve photorealism.", "required": False},
            "negative_prompt": {"type": "string", "description": "Things to steer away from, e.g. 'old, mature, middle-aged'. Applied via ConditioningConcat.", "required": False},
            "lora_name":          {"type": "string", "description": "LoRA filename to load (optional).", "required": False},
            "lora_strength_model":{"type": "float",  "description": "LoRA strength on the UNet (default: 1.0). Main lever for unlock/style LoRAs.", "required": False},
            "lora_strength_clip": {"type": "float",  "description": "LoRA strength on the text encoder (default: 0.5).", "required": False},
        },
    ),
    "run_tests": ToolDef(
        name="run_tests",
        fn=tool_run_tests,
        description=(
            "Run the project test suite. Auto-detects pytest, cargo test, go test, or npm/yarn test "
            "from project files. Returns pass/fail counts and failure output."
        ),
        params={
            "runner":     {"type": "string", "description": "Test runner: auto (default), pytest, cargo, go, npm, yarn", "required": False},
            "path":       {"type": "string", "description": "Limit run to a file or directory (default: whole project)", "required": False},
            "extra_args": {"type": "string", "description": "Extra arguments appended to the test command", "required": False},
        },
    ),
    "git_commit": ToolDef(
        name="git_commit",
        fn=tool_git_commit,
        description="Stage files and create a git commit. Stages all tracked changes when files is omitted.",
        params={
            "message": {"type": "string", "description": "Commit message", "required": True},
            "files":   {"type": "array",  "description": "Specific files to stage (omit to stage all changes)", "required": False},
        },
        destructive=True,
    ),
    "git_push": ToolDef(
        name="git_push",
        fn=tool_git_push,
        description="Push the current branch to a remote (default: origin).",
        params={
            "remote": {"type": "string", "description": "Remote name (default: origin)", "required": False},
            "branch": {"type": "string", "description": "Branch name (default: current branch)", "required": False},
        },
        destructive=True,
    ),
    "git_create_branch": ToolDef(
        name="git_create_branch",
        fn=tool_git_create_branch,
        description="Create a new git branch and optionally check it out.",
        params={
            "name":     {"type": "string", "description": "Branch name, e.g. feat/my-feature", "required": True},
            "checkout": {"type": "bool",   "description": "Switch to the new branch (default: true)", "required": False},
        },
        destructive=True,
    ),
    "search_replace": ToolDef(
        name="search_replace",
        fn=tool_search_replace,
        description=(
            "Search for a regex pattern across multiple files and replace all matches atomically. "
            "Use dry_run=true to preview changes before committing."
        ),
        params={
            "pattern":     {"type": "string", "description": "Python regex pattern (re.MULTILINE)", "required": True},
            "replacement": {"type": "string", "description": "Replacement string (supports \\1 back-references)", "required": True},
            "glob":        {"type": "string", "description": "File filter relative to cwd, e.g. **/*.py (default: all files)", "required": False},
            "dry_run":     {"type": "bool",   "description": "Preview without modifying (default: false)", "required": False},
        },
        destructive=True,
    ),
}

