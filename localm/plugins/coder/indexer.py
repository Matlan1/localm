# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Project map / codebase index.

Scans the working directory at agent startup and produces a compact,
human+LLM-readable summary of the codebase.  This is injected into the
agent's context so it knows what files exist and what they contain -
without having to ``list_dir`` or ``read_file`` every turn.

The map includes:
  - A file tree (git-ignored paths skipped)
  - Per-file: path, language, line count
  - Per-source-file: top-level symbol names (functions, classes, exports)

The whole thing is kept under ~3 000 characters so it doesn't dominate
context.  When the agent writes/edits a file, the map is refreshed for
that file only (cheap incremental update).
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

# Directories that are never interesting
_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache",
    "node_modules", ".next", ".nuxt", "dist", "build", ".dist",
    "venv", ".venv", "env", ".env",
    ".tox", ".eggs", "*.egg-info",
    "target",          # Rust / Maven
    ".idea", ".vscode",
})

# File extensions we can extract symbols from
_SYMBOL_LANGS: dict[str, str] = {
    ".py":   "python",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".jsx":  "javascript",
    ".tsx":  "typescript",
    ".go":   "go",
    ".rs":   "rust",
    ".java": "java",
    ".cs":   "csharp",
    ".cpp":  "cpp",
    ".c":    "c",
    ".rb":   "ruby",
    ".php":  "php",
    ".swift":"swift",
    ".kt":   "kotlin",
    ".lua":  "lua",
    ".sh":   "shell",
    ".bash": "shell",
}

# Extensions that are source/text but we don't extract symbols from
_TEXT_EXTS: frozenset[str] = frozenset({
    ".md", ".txt", ".rst", ".toml", ".yaml", ".yml", ".json", ".xml",
    ".html", ".css", ".scss", ".sass", ".less", ".sql", ".graphql",
    ".proto", ".tf", ".hcl", ".dockerfile", ".gitignore", ".env.example",
    ".lock", ".cfg", ".ini", ".conf",
})

_MAX_MAP_CHARS  = 3_000   # soft cap on the whole map string
_MAX_FILE_COUNT = 300     # stop scanning after this many files
_MAX_SYMBOLS    = 12      # max symbol names shown per file

# Wall-clock cap (seconds) on the startup scan so a session pointed at a huge
# root (e.g. C:\) cannot appear to hang. Generous - a pruned, bounded walk of a
# normal repo finishes in well under a second; only a pathological tree hits
# this, and when it does the map is marked truncated (surfaced, not hidden).
# Overridable per call (and via the coder_index_timeout config key); pass
# deadline_s=None to disable.
_BUILD_DEADLINE_S = 20.0


# ---------------------------------------------------------------------------
#  Symbol extractors (regex-based, best-effort)
# ---------------------------------------------------------------------------

def _extract_symbols_python(text: str) -> list[str]:
    """Top-level def and class names in Python source."""
    return re.findall(r"^(?:def|class|async def)\s+(\w+)", text, re.MULTILINE)


def _extract_symbols_js(text: str) -> list[str]:
    """Top-level function / class / exported const names."""
    patterns = [
        r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)",
        r"^(?:export\s+)?class\s+(\w+)",
        r"^export\s+(?:const|let|var)\s+(\w+)",
        r"^(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(",
    ]
    syms: list[str] = []
    for p in patterns:
        syms.extend(re.findall(p, text, re.MULTILINE))
    return syms


def _extract_symbols_go(text: str) -> list[str]:
    return re.findall(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)", text, re.MULTILINE)


def _extract_symbols_rust(text: str) -> list[str]:
    return re.findall(r"^pub\s+(?:async\s+)?fn\s+(\w+)|^(?:async\s+)?fn\s+(\w+)", text, re.MULTILINE)


def _extract_symbols_java_cs(text: str) -> list[str]:
    return re.findall(
        r"(?:public|private|protected|static|void|int|String|bool)\s+(\w+)\s*\(", text
    )


def _extract_symbols_generic(text: str) -> list[str]:
    """Fallback: grab anything that looks like a top-level identifier."""
    return re.findall(r"^(?:def|fn|func|function|class|sub|proc)\s+(\w+)", text, re.MULTILINE)


def _extract_symbols(text: str, lang: str) -> list[str]:
    extractors = {
        "python":     _extract_symbols_python,
        "javascript": _extract_symbols_js,
        "typescript": _extract_symbols_js,
        "go":         _extract_symbols_go,
        "rust":       _extract_symbols_rust,
        "java":       _extract_symbols_java_cs,
        "csharp":     _extract_symbols_java_cs,
    }
    fn = extractors.get(lang, _extract_symbols_generic)
    syms = fn(text)
    # deduplicate + filter empty strings (Rust alternation groups produce "")
    seen: set[str] = set()
    result: list[str] = []
    for s in syms:
        if s and s not in seen:
            seen.add(s)
            result.append(s)
    return result[:_MAX_SYMBOLS]


# ---------------------------------------------------------------------------
#  FileSummary
# ---------------------------------------------------------------------------

@dataclass
class FileSummary:
    path:    Path          # relative to project root
    lang:    str           # "python", "markdown", "unknown", …
    lines:   int
    symbols: list[str] = field(default_factory=list)

    def one_line(self) -> str:
        """Compact one-liner for the map."""
        lang_tag = f"[{self.lang}]" if self.lang not in ("unknown", "text") else ""
        sym_str  = ""
        if self.symbols:
            syms = ", ".join(self.symbols[:8])
            if len(self.symbols) > 8:
                syms += f", +{len(self.symbols) - 8}"
            sym_str = f"  - {syms}"
        lines_str = f" ({self.lines}L)" if self.lines else ""
        return f"  {self.path}{lines_str} {lang_tag}{sym_str}"


# ---------------------------------------------------------------------------
#  .gitignore parser (minimal)
# ---------------------------------------------------------------------------

def _load_gitignore_patterns(root: Path) -> list[str]:
    gi = root / ".gitignore"
    if not gi.exists():
        return []
    patterns: list[str] = []
    for line in gi.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(line.rstrip("/"))
    return patterns


def _is_ignored(rel: Path, patterns: list[str]) -> bool:
    """Very simple gitignore check - handles exact names and * globs."""
    name = rel.name
    parts = rel.parts
    for pat in patterns:
        # exact name match anywhere in the path
        if pat == name or pat in parts:
            return True
        # simple wildcard suffix: *.pyc
        if pat.startswith("*"):
            if name.endswith(pat[1:]):
                return True
    return False


# ---------------------------------------------------------------------------
#  ProjectMap
# ---------------------------------------------------------------------------

@dataclass
class ProjectMap:
    root:   Path
    files:  list[FileSummary] = field(default_factory=list)
    truncated: bool = False   # True if we hit _MAX_FILE_COUNT or _MAX_MAP_CHARS

    # ------------------------------------------------------------------
    #  Build
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, root: Path, max_files: int = _MAX_FILE_COUNT, *,
              deadline_s: float | None = _BUILD_DEADLINE_S,
              on_progress=None) -> "ProjectMap":
        """Scan *root* and summarise up to *max_files* files.

        Walks with ``os.walk`` and PRUNES uninteresting directories in place
        (hidden, ``_SKIP_DIRS``, gitignored) so it never descends into
        ``node_modules`` / ``.git`` or - the bug this fixes (CODER-1) - a huge
        root like ``C:\\``. Candidates are collected with bounded headroom and
        ONLY that bounded subset is sorted, so a giant tree cannot make startup
        hang on a full materialise-and-sort. A wall-clock *deadline_s* (None to
        disable) caps even a pathological walk; *on_progress*, if given, is
        called with the running candidate count. Either limit sets
        ``truncated`` so the map (and the model) shows the index is partial.
        """
        pm = cls(root=root)
        gi_patterns = _load_gitignore_patterns(root)
        start = time.monotonic()
        # Collect a bounded set of candidate files, then sort only those. Headroom
        # over max_files because some candidates are dropped below (binary / unreadable).
        candidate_cap = max(max_files * 4, max_files + 50)
        candidates: list[Path] = []
        hit_cap = False

        for dirpath, dirnames, filenames in os.walk(root):
            if deadline_s is not None and (time.monotonic() - start) > deadline_s:
                pm.truncated = True
                break
            # Prune in place so os.walk never DESCENDS into these dirs (the real
            # fix - this is what stops a C:\ scan dead). Sorted for a deterministic
            # traversal, so the sorted candidate subset is stable.
            dirnames[:] = sorted(
                d for d in dirnames
                if not d.startswith(".")
                and d not in _SKIP_DIRS
                and not _is_ignored((Path(dirpath) / d).relative_to(root), gi_patterns)
            )
            for name in sorted(filenames):
                if name.startswith("."):           # skip hidden files
                    continue
                abs_path = Path(dirpath) / name
                try:
                    rel = abs_path.relative_to(root)
                except ValueError:
                    continue
                if _is_ignored(rel, gi_patterns):
                    continue
                candidates.append(abs_path)
                if len(candidates) >= candidate_cap:
                    pm.truncated = True
                    hit_cap = True
                    break
            if on_progress is not None:
                try:
                    on_progress(len(candidates))
                except Exception:
                    pass
            if hit_cap:
                break

        count = 0
        for abs_path in sorted(candidates):
            if count >= max_files:
                pm.truncated = True
                break
            try:
                rel = abs_path.relative_to(root)
            except ValueError:
                continue
            if not abs_path.is_file():
                continue

            ext  = abs_path.suffix.lower()
            lang = _SYMBOL_LANGS.get(ext) or ("text" if ext in _TEXT_EXTS else "unknown")

            # Skip truly binary files fast
            if lang == "unknown" and ext not in ("", ".lock"):
                continue

            try:
                text  = abs_path.read_text(encoding="utf-8", errors="ignore")
                lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
            except OSError:
                continue

            syms: list[str] = []
            if lang in _SYMBOL_LANGS.values():
                syms = _extract_symbols(text, lang)

            pm.files.append(FileSummary(path=rel, lang=lang, lines=lines, symbols=syms))
            count += 1

        return pm

    # ------------------------------------------------------------------
    #  Incremental update for a single file
    # ------------------------------------------------------------------

    def refresh_file(self, abs_path: Path) -> None:
        """Re-index one file after a write or edit."""
        try:
            rel  = abs_path.relative_to(self.root)
        except ValueError:
            return

        ext  = abs_path.suffix.lower()
        lang = _SYMBOL_LANGS.get(ext) or ("text" if ext in _TEXT_EXTS else "unknown")

        self.files = [f for f in self.files if f.path != rel]

        if not abs_path.exists():
            return   # file was deleted

        try:
            text  = abs_path.read_text(encoding="utf-8", errors="ignore")
            lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        except OSError:
            return

        syms: list[str] = []
        if lang in _SYMBOL_LANGS.values():
            syms = _extract_symbols(text, lang)

        self.files.append(FileSummary(path=rel, lang=lang, lines=lines, symbols=syms))
        self.files.sort(key=lambda f: f.path)

    # ------------------------------------------------------------------
    #  Render
    # ------------------------------------------------------------------

    def to_context_string(self) -> str:
        """
        Produce the block injected into the agent's system prompt.
        Capped at _MAX_MAP_CHARS to avoid dominating context.
        """
        # HOME-ANCHORED, for the same reason as the prompt's identity line: this
        # block lands in the SAME system prompt, so printing the raw root here
        # handed back the absolute machine path and OS username that
        # _display_cwd had just stripped two lines above (REC-CODER-GUI-PATH,
        # AGENTS.md rule 2). Imported inside the method because prompts.py
        # imports this module for ProjectMap.
        from .prompts import _display_cwd
        shown_root = _display_cwd(self.root)
        if not self.files:
            return f"Working directory: {shown_root}\n(empty or no source files found)"

        lines: list[str] = [f"## Codebase map  ({shown_root})"]
        chars_used = len(lines[0])

        for f in self.files:
            line = f.one_line()
            if chars_used + len(line) + 1 > _MAX_MAP_CHARS:
                lines.append("  … (truncated - use list_dir / search_files to explore further)")
                self.truncated = True
                break
            lines.append(line)
            chars_used += len(line) + 1

        if self.truncated:
            lines.append("  (index truncated - some files not shown)")

        return "\n".join(lines)

    def file_count(self) -> int:
        return len(self.files)
