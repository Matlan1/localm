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

run_shell can write anything a `command` string cannot resolve to a `path`
arg ahead of time, so it cannot get the same eager per-file refresh - see
mark_dirty(). Instead it flags the whole map dirty, and the next read
(to_context_string / file_count) reconciles it against the filesystem with a
bounded stat-diff (see _rescan_if_dirty): every currently-tracked file is
stat()-ed for a moved mtime/size, and - unless build() already left files
uncounted on a repo bigger than max_files (files_capped) - every directory
that already has a tracked file is listdir()-ed once for names not yet
tracked there. No git, no subprocess, no full re-walk - and no cost at all
when nothing is dirty.
"""

from __future__ import annotations

import json
import os
import re
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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

# Wall-clock cap (seconds) on the startup scan. When it is hit, the map is marked
# truncated. Overridable per call and via the coder_index_timeout config key;
# pass deadline_s=None to disable.
_BUILD_DEADLINE_S = 20.0

# How old a cached map (see load_cached_and_reconcile/save_cache) may be before a
# session ignores it and does a full build() instead. On a CAPPED project,
# reconciling a cached map catches edits and deletions to already-tracked files
# exactly, but a brand-new file stays invisible until the next FULL build; this
# age cap bounds how long that lasts.
_MAX_CACHE_AGE_S = 7 * 24 * 3600


def _lang_for_ext(ext: str) -> str:
    """Map a lowercased suffix to its language tag, or "text" / "unknown".

    Shared by build(), refresh_file() and _rescan_if_dirty() so the three
    per-file passes can never drift out of sync with each other.
    """
    return _SYMBOL_LANGS.get(ext) or ("text" if ext in _TEXT_EXTS else "unknown")


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
    # mtime/size captured from the same stat() call build()/refresh_file() already
    # make: the staleness baseline _rescan_if_dirty() compares against after a
    # run_shell command. Defaulted, so callers that predate the field still work.
    mtime:   float = 0.0
    size:    int = 0

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
    # True ONLY when build() itself left matching files on disk uncounted (the
    # candidate cap, max_files, or the deadline cut the WALK short). Narrower than
    # `truncated`, which to_context_string() also sets for a too-long rendered
    # STRING even when every file WAS indexed. _rescan_if_dirty() needs the narrow
    # signal, because a listdir sweep cannot tell a never-a-candidate file apart
    # from one run_shell just created.
    files_capped: bool = False
    # Set by mark_dirty() after a tool that can write files with no resolvable
    # `path` arg (run_shell). Consumed by _rescan_if_dirty(), which every read
    # (to_context_string / file_count) calls first, so N shell calls before the
    # next read cost one rescan, not N.
    dirty: bool = False
    # Set by build() when *cache_path* was given AND a usable cache was reconciled
    # instead of a full walk. Never serialised by to_cache_dict(): like `dirty` it
    # describes how THIS instance was obtained, not a property of the project.
    # persistence.py reads it to log whether the map came from cache.
    from_cache: bool = False

    # ------------------------------------------------------------------
    #  Build
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, root: Path, max_files: int = _MAX_FILE_COUNT, *,
              deadline_s: float | None = _BUILD_DEADLINE_S,
              on_progress=None,
              cache_path: Path | None = None) -> "ProjectMap":
        """Scan *root* and summarise up to *max_files* files.

        Walks with ``os.walk`` and PRUNES uninteresting directories in place
        (hidden, ``_SKIP_DIRS``, gitignored) so it never descends into
        ``node_modules`` / ``.git`` or a huge root like ``C:\\``. Candidates are
        collected with bounded headroom and
        ONLY that bounded subset is sorted, so a giant tree cannot make startup
        hang on a full materialise-and-sort. A wall-clock *deadline_s* (None to
        disable) caps even a pathological walk; *on_progress*, if given, is
        called with the running candidate count. Either limit sets
        ``truncated`` so the map (and the model) shows the index is partial.

        When *cache_path* is given, this is ALSO the entry point for the
        cross-session cache: a usable cache is reconciled and returned directly,
        skipping the walk below entirely, and either way (cache hit or full
        walk) the result is saved back to *cache_path* before returning, ready
        for the next call to reconcile from. build() therefore stays the ONE
        call production code makes into ProjectMap.
        """
        if cache_path is not None:
            cached = cls.load_cached_and_reconcile(root, cache_path)
            if cached is not None:
                cached.from_cache = True
                cached.save_cache(cache_path)
                return cached

        pm = cls(root=root)
        gi_patterns = _load_gitignore_patterns(root)
        start = time.monotonic()
        # Collect a bounded set of candidate files, then sort only those. Headroom
        # over max_files because some candidates are dropped below (binary or
        # unreadable).
        candidate_cap = max(max_files * 4, max_files + 50)
        candidates: list[Path] = []
        hit_cap = False

        for dirpath, dirnames, filenames in os.walk(root):
            if deadline_s is not None and (time.monotonic() - start) > deadline_s:
                pm.truncated = pm.files_capped = True
                break
            # Prune in place so os.walk never DESCENDS into these dirs. Sorted for
            # a deterministic traversal, so the candidate subset is stable.
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
                    pm.truncated = pm.files_capped = True
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
                pm.truncated = pm.files_capped = True
                break
            try:
                rel = abs_path.relative_to(root)
            except ValueError:
                continue

            # One stat() call covers the is_file() check AND the mtime/size
            # _rescan_if_dirty() later compares against - not a second syscall.
            try:
                st = abs_path.stat()
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue

            ext  = abs_path.suffix.lower()
            lang = _lang_for_ext(ext)

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

            pm.files.append(FileSummary(path=rel, lang=lang, lines=lines, symbols=syms,
                                         mtime=st.st_mtime, size=st.st_size))
            count += 1

        if cache_path is not None:
            pm.save_cache(cache_path)
        return pm

    # ------------------------------------------------------------------
    #  Cross-session cache: persist the built map, reconcile it against the
    #  filesystem on load instead of re-walking the whole tree from scratch.
    # ------------------------------------------------------------------

    def to_cache_dict(self) -> dict:
        """Serialisable snapshot for save_cache(). ``root`` is recorded (and
        checked on load) as a cheap belt-and-suspenders guard against a
        project-digest collision handing this map to the wrong project - see
        checkpoint.py's _project_digest, whose 16-hex-char sha256 truncation
        makes an actual collision astronomically unlikely, but the check
        costs one string compare either way."""
        return {
            "version": 1,
            "root": str(self.root),
            "cached_at": time.time(),
            "truncated": self.truncated,
            "files_capped": self.files_capped,
            "files": [
                {"path": str(f.path), "lang": f.lang, "lines": f.lines,
                 "symbols": f.symbols, "mtime": f.mtime, "size": f.size}
                for f in self.files
            ],
        }

    @classmethod
    def _from_cache_dict(cls, root: Path, data: dict) -> Optional["ProjectMap"]:
        """Reconstruct a ProjectMap from to_cache_dict()'s shape, or None for
        anything that does not look right - a corrupt/foreign/future-version
        cache file must fall back to a full build(), never raise and never be
        silently trusted."""
        if not isinstance(data, dict) or data.get("version") != 1:
            return None
        if data.get("root") != str(root):
            return None
        try:
            files = [
                FileSummary(path=Path(f["path"]), lang=f["lang"], lines=f["lines"],
                            symbols=list(f.get("symbols") or []),
                            mtime=f["mtime"], size=f["size"])
                for f in data["files"]
            ]
        except (KeyError, TypeError):
            return None
        return cls(root=root, files=files,
                   truncated=bool(data.get("truncated", False)),
                   files_capped=bool(data.get("files_capped", False)))

    @classmethod
    def load_cached_and_reconcile(cls, root: Path, cache_path: Path,
                                  max_age_s: float = _MAX_CACHE_AGE_S,
                                  ) -> Optional["ProjectMap"]:
        """Load a map cached by save_cache() and bring it up to date against
        the CURRENT filesystem via the existing _rescan_if_dirty() machinery -
        NOT a raw cache hit. Returns None (never raises) when there is no
        usable cache to reconcile: absent, unreadable, wrong shape/version,
        for a different root, or older than *max_age_s* (see that constant's
        docstring for why a capped project needs this backstop). The caller's
        job on None is the same as if caching did not exist: call build()."""
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        cached_at = raw.get("cached_at") if isinstance(raw, dict) else None
        if not isinstance(cached_at, (int, float)):
            return None
        if (time.time() - cached_at) > max_age_s:
            return None
        pm = cls._from_cache_dict(root, raw)
        if pm is None:
            return None
        # Force exactly one reconciliation pass regardless of the loaded dirty
        # value: a cache is a snapshot from an earlier point in time, so it is
        # always treated as dirty on load.
        pm.dirty = True
        pm._rescan_if_dirty()
        return pm

    def save_cache(self, cache_path: Path) -> None:
        """Best-effort persist for the NEXT session's load_cached_and_reconcile()
        call. Never raises: a caching failure must not break indexing for the
        session that hit it. Written atomically (tmp file + os.replace), so a
        crash or a second concurrent coder session in the same project can never
        leave a torn/corrupt file for the next reader - the project-map cache is
        shared across every session in a project, unlike per-session
        checkpoints."""
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(self.to_cache_dict(), ensure_ascii=False),
                encoding="utf-8")
            os.replace(tmp_path, cache_path)
        except OSError:
            pass

    # ------------------------------------------------------------------
    #  Incremental update for a single file
    # ------------------------------------------------------------------

    def refresh_file(self, abs_path: Path) -> None:
        """Re-index one file after a write or edit (or, via _rescan_if_dirty,
        a run_shell command). Also the removal path: a path that no longer
        exists is dropped from the map and nothing is re-added; a path not
        previously tracked is simply added (the filter below is a no-op)."""
        try:
            rel  = abs_path.relative_to(self.root)
        except ValueError:
            return

        ext  = abs_path.suffix.lower()
        lang = _lang_for_ext(ext)

        self.files = [f for f in self.files if f.path != rel]

        try:
            st = abs_path.stat()
        except OSError:
            return   # file was deleted (or otherwise inaccessible)

        try:
            text  = abs_path.read_text(encoding="utf-8", errors="ignore")
            lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        except OSError:
            return

        syms: list[str] = []
        if lang in _SYMBOL_LANGS.values():
            syms = _extract_symbols(text, lang)

        self.files.append(FileSummary(path=rel, lang=lang, lines=lines, symbols=syms,
                                       mtime=st.st_mtime, size=st.st_size))
        self.files.sort(key=lambda f: f.path)

    # ------------------------------------------------------------------
    #  Deferred refresh after a run_shell command
    # ------------------------------------------------------------------

    def mark_dirty(self) -> None:
        """Flag that files may have changed outside the write/edit tools.

        run_shell can write anything - unlike write_file/edit_file it has no
        `path`-shaped arg a caller can resolve ahead of time - so instead of a
        per-file refresh_file() call, the whole map is marked dirty. Cheap: one
        bool, no stat, no subprocess. The actual reconciliation is deferred to
        the next read (see _rescan_if_dirty), so however many run_shell calls
        happen before that next read, they cost one rescan, not one each.
        """
        self.dirty = True

    def _rescan_if_dirty(self) -> None:
        """Reconcile the map against the filesystem if mark_dirty() was called
        since the last read. Two bounded passes, no subprocess and no full
        re-walk of the tree:

          1. Stat every currently-tracked file. refresh_file() whatever is
             gone (deleted) or whose mtime/size moved (edited) - a changed
             stat is treated as "moved" without re-reading the content first,
             since reading it IS the refresh. Always runs, regardless of
             files_capped: a tracked file's own stat is meaningful either way.
          2. listdir() every directory that has at least one tracked file, for
             names not already tracked there (created) - exact, unlike a
             coarser dir-mtime heuristic. SKIPPED when files_capped is True:
             on a repo bigger than max_files, a known directory can hold many
             files that were never candidates at all, and listdir cannot tell
             those apart from one a shell command genuinely just created, so
             treating every one as "new" would grow the map past max_files on
             every dirty read. A file inside a brand-new directory (no
             previously-tracked file at all) is not discovered this way either,
             capped or not; only a full reindex() picks up either case.

        Runs once per dirty period regardless of how many files actually
        changed: bounded by O(tracked files [+ tracked dirs, when pass 2
        runs]), never O(whole tree).
        """
        if not self.dirty:
            return
        self.dirty = False

        known_dirs: dict[Path, set[str]] = {}
        stale: list[Path] = []
        for f in self.files:
            abs_path = self.root / f.path
            known_dirs.setdefault(abs_path.parent, set()).add(f.path.name)
            try:
                st = abs_path.stat()
            except OSError:
                stale.append(abs_path)                      # deleted
                continue
            if st.st_mtime != f.mtime or st.st_size != f.size:
                stale.append(abs_path)                       # edited

        # Gate: when build() left files uncounted, a known dir can hold many that
        # were never candidates at all, and listdir cannot tell one of those apart
        # from a file run_shell just created, so new-file discovery is skipped.
        if not self.files_capped:
            gi_patterns = _load_gitignore_patterns(self.root)
            for abs_dir, known_names in known_dirs.items():
                try:
                    entries = os.listdir(abs_dir)
                except OSError:
                    continue                                 # directory itself is gone
                for name in entries:
                    if name in known_names or name.startswith("."):
                        continue
                    abs_path = abs_dir / name
                    try:
                        rel = abs_path.relative_to(self.root)
                    except ValueError:
                        continue
                    if _is_ignored(rel, gi_patterns):
                        continue
                    ext  = abs_path.suffix.lower()
                    lang = _lang_for_ext(ext)
                    if lang == "unknown" and ext not in ("", ".lock"):
                        continue                              # binary - build() skips these too
                    stale.append(abs_path)

        for abs_path in stale:
            self.refresh_file(abs_path)

    # ------------------------------------------------------------------
    #  Render
    # ------------------------------------------------------------------

    def to_context_string(self) -> str:
        """
        Produce the block injected into the agent's system prompt.
        Capped at _MAX_MAP_CHARS to avoid dominating context.
        """
        self._rescan_if_dirty()
        # Home-anchored: this block lands in the same system prompt as the
        # identity line, so the raw root is never printed. Imported inside the
        # method because prompts.py imports this module for ProjectMap.
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
        self._rescan_if_dirty()
        return len(self.files)
