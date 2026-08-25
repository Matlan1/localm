# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.rag - chat with your documents, fully offline."""

from .bm25 import BM25
from .chunk import chunk_text
from .collection_lock import CollectionLockedError
from .extract import (EXTRACTABLE_SUFFIXES, ExtractError, classify_format,
                      extract_bytes, extract_text, sniff_text_format)
from .store import (Collection, check_collection_name, collection_names,
                    collection_provenance_note, collection_provenance_report,
                    delete_collection, rag_dir)

__all__ = [
    "BM25",
    "Collection",
    "CollectionLockedError",
    "check_collection_name",
    "ExtractError",
    "EXTRACTABLE_SUFFIXES",
    "chunk_text",
    "classify_format",
    "collection_names",
    "collection_provenance_note",
    "collection_provenance_report",
    "delete_collection",
    "extract_bytes",
    "extract_text",
    "rag_dir",
    "sniff_text_format",
]
