# SPDX-License-Identifier: AGPL-3.0-or-later
"""The chat and coder attach pickers offer every document suffix the server
extracts text from (localm.rag.extract), so the file dialog never hides a file
the attachment route reads."""
from __future__ import annotations

import re
from pathlib import Path

from localm.rag import extract

INDEX_HTML = (Path(__file__).resolve().parent.parent / "localm" / "plugins" / "gui"
              / "static" / "index.html").read_text(encoding="utf-8")

DOCUMENT_SUFFIXES = extract.EXTRACTABLE_SUFFIXES - extract._IMAGE_SUFFIXES


def _accept(input_id: str) -> set:
    m = re.search(r'<input[^>]*\bid="' + re.escape(input_id) + r'"[^>]*\baccept="([^"]*)"',
                  INDEX_HTML)
    assert m, f"#{input_id} has no accept attribute"
    return {part.strip().lower() for part in m.group(1).split(",") if part.strip()}


def test_chat_picker_offers_every_extractable_document():
    missing = sorted(DOCUMENT_SUFFIXES - _accept("chat-file"))
    assert not missing, f"#chat-file hides suffixes the extractor reads: {missing}"


def test_coder_picker_offers_every_extractable_document():
    missing = sorted(DOCUMENT_SUFFIXES - _accept("coder-file"))
    assert not missing, f"#coder-file hides suffixes the extractor reads: {missing}"


def test_the_document_set_is_not_empty():
    assert {".pdf", ".ini", ".zip"} <= DOCUMENT_SUFFIXES
