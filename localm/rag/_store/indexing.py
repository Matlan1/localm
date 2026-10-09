# SPDX-License-Identifier: AGPL-3.0-or-later
"""Indexing documents into a collection: the folder walk, files named
on disk, and uploaded documents."""

from __future__ import annotations

import hashlib
import os
import stat as _stat
import time
from pathlib import Path
from typing import Optional

from localm.debuglog import logger as _log
from localm.rag import store as _st

from ..chunk import chunk_text
from ..extract import (BLACKLISTED_SUFFIXES, UNINDEXABLE_SUFFIXES,
                       ExtractError, classify_format, extract_text,
                       is_secret_index_name)
from .types import ClassifyFn, DescribeImageFn, EmbedFn, ProgressFn
from .vectors import _first_dim, _vectors_finite


# Directories never worth indexing when a folder is added
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__",
              ".pytest_cache", ".mypy_cache", "dist", "build", ".idea",
              ".vscode"}


# Cap the folder-walk recursion depth, bounding a pathological directory cycle.
_MAX_WALK_DEPTH = 50


def _walk_files(root: Path, *, max_depth: int = _MAX_WALK_DEPTH):
    """Yield files under *root* without following linked DIRECTORIES, bounded by
    depth and a visited-realpath set.

    Never descends into a linked directory (junction OR bind-mount OR symlink)
    and refuses to revisit a resolved directory, so no directory cycle can hang
    indexing. ``_SKIP_DIRS`` are pruned during descent.

    A linked FILE **is** yielded. Confinement (a link escaping an allowed root)
    is enforced by ``_expand``'s confine loop, not here."""
    reparse_flag = getattr(_stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    seen: set = set()
    stack: list = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            real = os.path.realpath(d)
        except OSError:
            continue
        if real in seen:
            continue
        seen.add(real)
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        for e in entries:
            try:
                attrs = getattr(e.stat(follow_symlinks=False), "st_file_attributes", 0)
                if e.is_symlink() or (attrs & reparse_flag):
                    # Branch on the RESOLVED type: a linked DIRECTORY is not
                    # followed, a linked FILE is yielded.
                    try:
                        if e.is_dir(follow_symlinks=True):
                            _log.debug("rag: not following linked directory during "
                                       "index walk: %s", e.path)
                            continue
                        if e.is_file(follow_symlinks=True):
                            yield Path(e.path)
                            continue
                        # Neither: a dangling or unresolvable link.
                        _log.debug("rag: skipping unresolvable link during index "
                                   "walk: %s", e.path)
                    except OSError as exc:
                        _log.debug("rag: could not resolve link during index walk: "
                                   "%s (%s)", e.path, exc)
                    continue
                if e.is_dir(follow_symlinks=False):
                    if e.name in _SKIP_DIRS:
                        continue                   # prune .git/node_modules/etc.
                    stack.append((Path(e.path), depth + 1))
                elif e.is_file(follow_symlinks=False):
                    yield Path(e.path)
            except OSError:
                continue


class _CollectionIndexing:
    """Mixin of ``Collection``: indexing files, folders and uploads, chunking and
    embedding each document."""

    @staticmethod
    def _expand(paths: list,
                policy: Optional[dict] = None) -> list[Path]:
        """Resolve files + recursive folder contents to indexable files.

        When *policy* is given, files that fail confinement (system paths,
        credential dirs, denied roots, symlinks escaping an allowed folder, or a
        model-weight / binary / credential FILE) are dropped by the confine loop
        below. With no policy (the CLI) explicit picks are unfiltered."""
        out: list[Path] = []
        for p in paths:
            p = Path(p).expanduser()
            if p.is_file():
                out.append(p.resolve())
            elif p.is_dir():
                # _walk_files (NOT rglob): it bounds directory-link loops, branches
                # on the resolved type of a link, and prunes _SKIP_DIRS during
                # descent.
                for f in sorted(_walk_files(p)):
                    if (f.suffix.lower() not in BLACKLISTED_SUFFIXES
                            and not is_secret_index_name(f.name)
                            and not any(part in _SKIP_DIRS for part in f.parts)):
                        out.append(f.resolve())
        # de-dup, keep order
        seen: set = set()
        deduped = [p for p in out if not (p in seen or seen.add(p))]
        if policy is None:
            return deduped
        kept: list[Path] = []
        for p in deduped:
            try:
                _st.confine_index_path(p, policy)
            except ValueError:
                continue   # nested escape (symlink / credential / denied) -> skip
            kept.append(p)
        return kept

    def _record_roots(self, paths: list) -> bool:
        """Persist the FOLDER roots among *paths*. Returns True if anything new
        was recorded.

        Only directories are recorded. An individually added FILE is already
        tracked by its own ``docs`` entry, which ``resync`` re-checks directly.

        Called from ``_add_paths_locked`` AFTER the confinement check.
        """
        roots = self._meta.setdefault("roots", {})
        if not isinstance(roots, dict):
            # An externally written meta.json could hold anything here. The bad
            # value is replaced and the collection is flagged corrupt.
            _log.warning("RAG collection %r: meta.json 'roots' was not an object "
                         "(%s); starting a fresh roots map", self.name,
                         type(roots).__name__)
            roots = {}
            self._meta["roots"] = roots
            self.corrupt = True
        changed = False
        for p in paths:
            try:
                rp = Path(p).expanduser()
                if not rp.is_dir():
                    continue
                key = str(rp.resolve())
            except (OSError, ValueError) as e:
                # Already skipped by _expand; logged rather than dropped silently.
                _log.debug("rag: could not record %s as an index root: %s", p, e)
                continue
            if key not in roots:
                roots[key] = {"added": time.time()}
                changed = True
        return changed

    def _add_paths_locked(self, paths: list, *, embed_fn: Optional[EmbedFn] = None,
                          classify_fn: Optional[ClassifyFn] = None,
                          describe_image_fn: Optional[DescribeImageFn] = None,
                          on_progress: Optional[ProgressFn] = None,
                          policy: Optional[dict] = None,
                          force: bool = False,
                          model_name: Optional[str] = None) -> dict:
        """The add_paths read-modify-write body. MUST run under
        _collection_lock(self.name) after a fresh _load() (see add_paths)."""
        say = on_progress or (lambda _t: None)
        if policy is not None:
            # confine_index_path returns the RESOLVED path it validated, and
            # that is what the walk below starts from, rather than the caller's
            # original string.
            paths = [_st.confine_index_path(p, policy) for p in paths]  # raises ValueError
        # Persist the FOLDER roots now that confinement has accepted them, and
        # before the expand, so an add that finds no indexable file still records
        # the folder.
        roots_changed = self._record_roots(paths)
        files = self._expand(paths, policy)
        if not files:
            if roots_changed:
                self._save_meta()   # metadata only: this add indexed nothing
            return {"added": 0, "updated": 0, "skipped": 0, "failed": [],
                    "chunks": len(self._chunks)}

        added = updated = skipped = 0
        failed: list = []
        embed_broken = embed_fn is None

        for index, f in enumerate(files, 1):
            key = str(f)
            # An EXPLICITLY-NAMED non-secret binary; a folder walk already filters
            # these out in _expand, so only a direct pick reaches here. Reported as
            # an ordinary per-file failure, the same shape an ExtractError below
            # produces, and BEFORE stat/read_bytes.
            if f.suffix.lower() in UNINDEXABLE_SUFFIXES:
                msg = (f"{f.name}: no extractable text (binary, media, or model "
                       f"weights)")
                failed.append({"path": key, "error": msg})
                say(f"skip {msg}")
                continue
            try:
                stat = f.stat()
            except OSError as e:
                failed.append({"path": key, "error": str(e)})
                continue
            known = self._meta["docs"].get(key)
            # Content hash as well as (mtime, size), so a same-size edit whose
            # mtime is unchanged is still re-indexed. Legacy entries lacking
            # "hash" compare unequal and self-heal on the next add.
            try:
                digest = hashlib.sha256(f.read_bytes()).hexdigest()
            except OSError as e:
                failed.append({"path": key, "error": str(e)})
                continue
            if not force and known \
                    and known.get("mtime") == stat.st_mtime \
                    and known.get("size") == stat.st_size \
                    and known.get("hash") == digest:
                skipped += 1
                continue
            say(f"[{index}/{len(files)}] reading {f.name}...")
            try:
                text = extract_text(f, describe_image_fn=describe_image_fn)
            except ExtractError as e:
                failed.append({"path": key, "error": str(e)})
                say(f"skip {f.name}: {e}")
                continue
            new_chunks = chunk_text(text)
            # Heuristic-first format label; the LLM tie-break is consulted only
            # for an unknown extension whose structure is unclear AND a chat model
            # loaded (classify_fn short-circuits otherwise).
            fmt = classify_format(text, f.name, classify_fn=classify_fn)
            for c in new_chunks:
                c["source"] = key
                c["format"] = fmt

            vectors, embed_broken = self._embed_doc(
                new_chunks, f"[{index}/{len(files)}]", f.name, embed_fn=embed_fn,
                embed_broken=embed_broken, model_name=model_name, say=say)

            # Replace any previous chunks (and vectors) for this document
            if known:
                updated += 1
            else:
                added += 1
            self._put_doc_chunks(key, bool(known), new_chunks, vectors)
            self._meta["docs"][key] = {
                "mtime": stat.st_mtime, "size": stat.st_size,
                "hash": digest,
                "chunks": len(new_chunks),
            }
            say(f"indexed {f.name} ({len(new_chunks)} chunks)")

        if added or updated:
            self._save()
        elif roots_changed or self.corrupt:
            # Nothing was indexed (every file skipped as unchanged, or every one
            # failed), so chunks and vectors are exactly as _load() read them.
            # Persist metadata only, and only when there is something to persist:
            # a newly recorded root, or a meta.json that _load() flagged corrupt
            # and rebuilt a docs map for. Mirrors the no-indexable-files early
            # return above.
            self._save_meta()
        return {"added": added, "updated": updated, "skipped": skipped,
                "failed": failed, "chunks": len(self._chunks)}

    def _add_uploads_locked(self, uploads: list, *,
                            embed_fn: Optional[EmbedFn] = None,
                            classify_fn: Optional[ClassifyFn] = None,
                            describe_image_fn: Optional[DescribeImageFn] = None,
                            on_progress: Optional[ProgressFn] = None,
                            force: bool = False,
                            model_name: Optional[str] = None) -> dict:
        """The add_uploads body. MUST run under _collection_lock after _load().

        Mirrors the per-document body of _add_paths_locked (chunk -> embed ->
        replace-prior-chunks -> record meta), but sourced from in-memory bytes with
        a hash-only dedup (no fs mtime/size)."""
        # The no-op accepts **_ because _finished below passes the structured
        # keywords ProgressFn carries.
        say = on_progress or (lambda _t, **_: None)
        added = updated = skipped = 0
        failed: list = []
        embed_broken = embed_fn is None
        n_total = len(uploads)

        def _finished(n: int, text: str) -> None:
            """Report one item DONE, whatever its outcome.

            Called on every exit from the loop body, including the two that
            `continue`, so the done-SEQUENCE is exactly 1..n_total. Progress is
            about how far the LOOP got, not how many succeeded.
            """
            say(text, phase="indexing uploads", done=n, total=n_total,
                unit="files")

        for n_done, up in enumerate(uploads, start=1):
            filename = (str(up.get("filename") or "").strip() or "upload")
            data = up.get("data") or b""
            key = f"upload:{filename}"
            digest = hashlib.sha256(data).hexdigest()
            known = self._meta["docs"].get(key)
            if not force and known and known.get("hash") == digest:
                skipped += 1
                _finished(n_done, f"skip {filename} (unchanged)")
                continue
            say(f"[{n_done}/{n_total}] reading {filename}...")
            try:
                text = _st.extract_bytes(data, filename, describe_image_fn=describe_image_fn)
            except ExtractError as e:
                failed.append({"path": key, "error": str(e)})
                _finished(n_done, f"skip {filename}: {e}")
                continue
            new_chunks = chunk_text(text)
            # Same heuristic-first labeling as add_paths (see there).
            fmt = classify_format(text, filename, classify_fn=classify_fn)
            for c in new_chunks:
                c["source"] = key
                c["format"] = fmt

            vectors, embed_broken = self._embed_doc(
                new_chunks, f"[{n_done}/{n_total}]", filename, embed_fn=embed_fn,
                embed_broken=embed_broken, model_name=model_name, say=say)

            if known:
                updated += 1
            else:
                added += 1
            self._put_doc_chunks(key, bool(known), new_chunks, vectors)
            self._meta["docs"][key] = {
                "size": len(data), "hash": digest,
                "chunks": len(new_chunks), "uploaded": True,
            }
            _finished(n_done, f"indexed {filename} ({len(new_chunks)} chunks)")

        self._save()
        return {"added": added, "updated": updated, "skipped": skipped,
                "failed": failed, "chunks": len(self._chunks)}

    def _embed_doc(self, new_chunks: list, position: str, label: str, *,
                   embed_fn: Optional[EmbedFn], embed_broken: bool,
                   model_name: Optional[str], say: ProgressFn) -> "tuple[list, bool]":
        """Vectors for one document's *new_chunks*, all None when embedding is
        unavailable, fails, or yields non-finite values, and whether embedding is
        now broken for the rest of the batch. *position* (``[i/n]``) and *label*
        (the document's name) go into the progress messages.

        Raises ``ValueError``, after saving the documents this call already
        finished, when the vectors' dimensionality differs from the
        collection's."""
        vectors: list = [None] * len(new_chunks)
        if not embed_broken and new_chunks:
            say(f"{position} embedding {label} "
                f"({len(new_chunks)} chunks)...")
            try:
                vecs = embed_fn([c["text"] for c in new_chunks])
            except Exception as e:
                embed_broken = True
                say(f"embeddings unavailable ({e}) - indexing lexical-only")
            else:
                if len(vecs) == len(new_chunks):
                    if not _vectors_finite(vecs):
                        # A NaN/inf component: no vectors are stored for this
                        # doc (lexical-only) and the degrade is reported.
                        say(f"embeddings had non-finite (NaN/inf) values for "
                            f"{label} - indexing it lexical-only")
                    else:
                        new_dim = _first_dim(vecs)
                        # A different embedding dimensionality means a
                        # different model: refused rather than stored as
                        # mixed-dim vectors.
                        if (self._vec_dim is not None and new_dim is not None
                                and new_dim != self._vec_dim):
                            # Persist every file this call already finished
                            # before raising: the exception HALTS the batch,
                            # and add_paths()/add_uploads() otherwise _save()
                            # only once at the very end.
                            self._save()
                            raise ValueError(self._dim_mismatch_message(new_dim))
                        vectors = vecs
                        if self._vec_dim is None and new_dim is not None:
                            self._vec_dim = new_dim
                            # Record which model built this index, the same
                            # key reembed() writes.
                            if model_name:
                                self._meta["embedding_model"] = str(model_name)
                        elif (new_dim is not None and model_name
                                and model_name != self._meta.get("embedding_model")):
                            # Same dimension as before but a different named
                            # model: the recorded label above still names
                            # only the first model, so mark the vectors as
                            # spanning more than one.
                            self._meta["embedding_model_mixed"] = True
        return vectors, embed_broken

    def _put_doc_chunks(self, key: str, replace: bool, new_chunks: list,
                        vectors: list) -> None:
        """Append document *key*'s *new_chunks* and their *vectors*, first
        dropping the chunks (and vectors) it already has when *replace*."""
        if replace:
            keep = [i for i, c in enumerate(self._chunks)
                    if c.get("source") != key]
            self._chunks = [self._chunks[i] for i in keep]
            if self._vectors is not None:
                self._vectors = [self._vectors[i] for i in keep]
        if self._vectors is None:
            self._vectors = [None] * len(self._chunks)
        self._chunks.extend(new_chunks)
        self._vectors.extend(vectors)

    def _dim_mismatch_message(self, new_dim: int) -> str:
        """The refusal a user sees when they change embedding model.

        Names the collection, both models where known, and the command that
        re-embeds in place without touching the source files.
        """
        was = self.embedding_model()
        built = f" with {was}" if was else ""
        return (
            f"Embedding dimension changed ({self._vec_dim} -> {new_dim}): "
            f"collection {self.name!r} was built{built} and its stored vectors "
            f"cannot be mixed with a different model's. Re-embed it in place from "
            f"the text already stored (no source files needed, nothing deleted):\n"
            f"    localm rag reembed {self.name}\n"
            f"or use 'Re-embed' on the Knowledge page. To keep the existing index "
            f"instead, switch the embedding model back{built}.")
