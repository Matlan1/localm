# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Knowledge page refuses an upload at the same per-file size the server
refuses it at, so a file is never accepted by the page and then refused by the
route."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_JS = (ROOT / "localm" / "plugins" / "gui" / "static" / "pages" / "knowledge.js").read_text(encoding="utf-8")
RAG_PLUG = (ROOT / "localm" / "plugins" / "builtin" / "rag" / "plug.py").read_text(encoding="utf-8")


def _int(pattern: str, src: str) -> int:
    m = re.search(pattern, src)
    assert m, pattern
    return int(m.group(1).replace("_", ""))


def test_kb_upload_cap_matches_the_server():
    page = _int(r"export const KB_UPLOAD_MAX_BYTES = ([\d_]+);", KNOWLEDGE_JS)
    b64_per_file = _int(r"_B64_PER_FILE = ([\d_]+)", RAG_PLUG)
    assert page == b64_per_file * 3 // 4, (page, b64_per_file)


def test_the_page_uses_the_cap_for_its_check():
    assert "const MAX = KB_UPLOAD_MAX_BYTES;" in KNOWLEDGE_JS
