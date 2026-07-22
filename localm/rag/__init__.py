# SPDX-License-Identifier: AGPL-3.0-or-later
"""
localm.rag - chat with your documents, fully offline.

Design constraints that shaped this package:

- The built-in ctypes GGUF binding does not support embeddings, so retrieval
  is **lexical-first**: BM25 over chunked documents always works, with zero
  extra dependencies. When the running backend does support embeddings
  (HF models, llama-cpp-python), vectors are stored too and the query score
  becomes a blend of both. Embeddings are an enhancement, never a requirement.
- Collections are explicit user data (like generated images): creating and
  indexing one writes to ``<data dir>/rag/<name>/`` in every session mode.
  Transient document *attachments* in chat are extracted in memory and write
  nothing - they stay privacy-clean.
- Extraction is stdlib wherever possible: txt/md/code directly, .docx via
  zipfile+xml, .html via the netpolicy stripper, .ipynb via json. Only PDF
  needs a third-party parser (pypdf, the ``[rag]`` extra).
"""

from .bm25 import BM25
from .chunk import chunk_text
from .extract import (EXTRACTABLE_SUFFIXES, ExtractError, classify_format,
                      extract_bytes, extract_text, sniff_text_format)
from .store import (Collection, check_collection_name, collection_names,
                    delete_collection, rag_dir)

__all__ = [
    "BM25",
    "Collection",
    "check_collection_name",
    "ExtractError",
    "EXTRACTABLE_SUFFIXES",
    "chunk_text",
    "classify_format",
    "collection_names",
    "delete_collection",
    "extract_bytes",
    "extract_text",
    "rag_dir",
    "sniff_text_format",
]
