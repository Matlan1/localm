# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contract check: tests-js/web-fixtures.mjs's bundleOf() is a hand-maintained
JS mirror of the /api/web/retrieve response shape (its own docstring says so).
web.test.mjs proves the real client JS (settings-perf.js) builds the correct
wire format against that mirror, but never against the real server, so a field
renamed/added/removed on either side would go unnoticed there. These tests pin
bundleOf()'s field names, at every nesting level, against the field names the
real production function the endpoint calls (_neutralise_bundle) actually
returns. Complements test_jobs_web_search.py's TestGrammarMirroredInGuiSurface,
which pins a different JS/Python pair the same way.
"""

from __future__ import annotations

import re
from pathlib import Path

from localm.plugins.builtin.web.plug import _neutralise_bundle
from localm.web_retrieval.contracts import EvidenceBundle, EvidenceChunk, Source

_JS_FIXTURES = (Path(__file__).resolve().parents[1] / "tests-js" / "web-fixtures.mjs")

_KEY_RE = re.compile(r'[{,]\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*(?=[:,}])')


def _object_literal_keys(src: str, start_marker: str) -> set[str]:
    """The property-key names of the first, brace-balanced object literal
    that appears after *start_marker* in *src*."""
    start = src.index(start_marker) + len(start_marker)
    open_brace = src.index("{", start)
    depth = 0
    close_brace = None
    for i in range(open_brace, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                close_brace = i
                break
    assert close_brace is not None, f"unbalanced braces after {start_marker!r}"
    body = src[open_brace:close_brace + 1]
    return set(_KEY_RE.findall(body))


def _real_bundle_dict() -> dict:
    """The real /api/web/retrieve response: one source and one chunk, run
    through the actual production function web_retrieve_endpoint calls."""
    bundle = EvidenceBundle(
        query="q", provider="stub", search_status="ok", search_error=None,
        sources=[Source(id="S1", url="https://x", canonical_url="https://x",
                         title="t", snippet="s", provider_rank=1,
                         retrieval_status="fetched", grounding="page-backed",
                         final_url="https://x", error=None, region="main",
                         text_chars=5)],
        chunks=[EvidenceChunk(source_id="S1", text="hello", score=1.0,
                               offset=0, kind="page")],
    )
    return _neutralise_bundle(bundle)


class TestBundleOfMirrorsTheRealResponseShape:
    """Bound to the REAL shipped tests-js/web-fixtures.mjs and the REAL
    _neutralise_bundle, at every nesting level, so a field drift on either
    side is caught here instead of only staying invisible inside JS tests
    that run against a mock which quietly stopped matching reality."""

    def test_bundle_level_fields_match(self):
        js = _JS_FIXTURES.read_text(encoding="utf-8")
        js_keys = _object_literal_keys(js, "\n  return")
        real_keys = set(_real_bundle_dict().keys())
        assert js_keys == real_keys, (
            "tests-js/web-fixtures.mjs's bundleOf() top-level fields have "
            f"drifted from web/plug.py's real response.\n"
            f"  only in bundleOf(): {sorted(js_keys - real_keys)}\n"
            f"  only in the real response: {sorted(real_keys - js_keys)}\n"
            "Update bundleOf() to match web/plug.py's _neutralise_bundle().")

    def test_source_level_fields_match(self):
        js = _JS_FIXTURES.read_text(encoding="utf-8")
        js_keys = _object_literal_keys(
            js, "const sources = results.map((r, i) => (")
        real_keys = set(_real_bundle_dict()["sources"][0].keys())
        assert js_keys == real_keys, (
            "tests-js/web-fixtures.mjs's bundleOf() per-source fields have "
            f"drifted from web/plug.py's real response.\n"
            f"  only in bundleOf(): {sorted(js_keys - real_keys)}\n"
            f"  only in the real response: {sorted(real_keys - js_keys)}\n"
            "Update bundleOf() to match web_retrieval.contracts.Source.")

    def test_chunk_level_fields_match(self):
        js = _JS_FIXTURES.read_text(encoding="utf-8")
        js_keys = _object_literal_keys(
            js, "const chunks = results.filter((r) => r.page || "
                "r.snippet).map((r, i) => (")
        real_keys = set(_real_bundle_dict()["chunks"][0].keys())
        assert js_keys == real_keys, (
            "tests-js/web-fixtures.mjs's bundleOf() per-chunk fields have "
            f"drifted from web/plug.py's real response.\n"
            f"  only in bundleOf(): {sorted(js_keys - real_keys)}\n"
            f"  only in the real response: {sorted(real_keys - js_keys)}\n"
            "Update bundleOf() to match web_retrieval.contracts.EvidenceChunk.")
