# SPDX-License-Identifier: AGPL-3.0-or-later
"""RAG format labeling contract:

  * a document's format is labeled heuristic-FIRST (free, deterministic) and the
    label is carried into chunk metadata;
  * the LLM tie-break runs ONLY when the heuristic is unsure AND a chat model is
    loaded, so an embedding-only index fires no chat call.
"""

import requests

from localm.rag import Collection
from localm.plugins.builtin.rag import plug


# --------------------------------------------------------------------------- #
#  Store-level oracle: format reaches chunk metadata, zero chat calls          #
# --------------------------------------------------------------------------- #

def test_odd_ext_json_labeled_via_heuristic_no_chat(tmp_path):
    """An odd-extension file whose content is JSON gets chunk metadata
    format="json" from the free structural heuristic, with ZERO classify calls."""
    coll = Collection("kb", base=tmp_path)
    coll.create()

    calls = []
    def spy_classify(snippet):
        calls.append(snippet)
        return "must-not-be-used"

    payload = b'{"name": "localm", "tags": ["a", "b"], "n": 3, "ok": true}'
    result = coll.add_uploads(
        [{"filename": "data.weirdext", "data": payload}],
        classify_fn=spy_classify)

    assert result["added"] == 1
    # (a) the label reaches EVERY chunk's metadata instead of being discarded
    assert coll._chunks, "expected at least one indexed chunk"
    assert all(c.get("format") == "json" for c in coll._chunks)
    # (b) a confident heuristic means no chat/classify call at all
    assert calls == []
    # and the label is persisted, not just in-memory
    reloaded = Collection("kb", base=tmp_path)
    assert reloaded._chunks
    assert all(c.get("format") == "json" for c in reloaded._chunks)


def test_embedding_only_index_does_not_call_chat_path(tmp_path, monkeypatch):
    """With no chat model loaded, indexing an ambiguous odd-extension file does
    not fire the (10s-timeout) chat endpoint and labels the chunks "text"."""
    posted = []
    def fake_request(*args, **kwargs):
        posted.append((args, kwargs))
        raise RuntimeError("chat endpoint must not be called with no model loaded")
    # self_request (localm.selfclient) issues requests.request(method, ...), not
    # requests.post directly, so this patches the mechanism actually used.
    monkeypatch.setattr(requests, "request", fake_request)

    # active_model() == "" is exactly what the server reports when no engine is
    # resident (see http_server.active_model).
    classify = plug._make_self_classify("https://127.0.0.1:65535/v1", lambda: "")

    coll = Collection("kb2", base=tmp_path)
    coll.create()
    # Content the structural heuristic cannot pin down (mid-line braces, no CSV /
    # YAML / markdown shape) -> the tie-break is *attempted*, and shorts out.
    payload = b"function add(a, b) { return a + b }\nconst x = 1\n"
    result = coll.add_uploads(
        [{"filename": "snippet.zzz", "data": payload}],
        classify_fn=classify)

    assert result["added"] == 1
    assert posted == []                       # no chat HTTP call -> no 10s stall
    assert coll._chunks
    assert all(c.get("format") == "text" for c in coll._chunks)


def test_known_extension_labels_from_suffix(tmp_path):
    """A known text extension is labeled from its suffix."""
    coll = Collection("kb3", base=tmp_path)
    coll.create()
    coll.add_uploads([{"filename": "readme.md", "data": b"# Title\n\nbody text"}])
    coll.add_uploads([{"filename": "app.py", "data": b"import os\nx = 1\n"}])
    fmts = {c["source"]: c.get("format") for c in coll._chunks}
    assert fmts["upload:readme.md"] == "markdown"
    assert fmts["upload:app.py"] == "python"


# --------------------------------------------------------------------------- #
#  _make_self_classify availability gate (plug.py)                             #
# --------------------------------------------------------------------------- #

def test_make_self_classify_no_model_short_circuits(monkeypatch):
    """No chat model loaded -> classify returns None WITHOUT any HTTP call."""
    called = []
    monkeypatch.setattr(requests, "request", lambda *a, **k: called.append(1))
    classify = plug._make_self_classify("https://127.0.0.1:65535/v1", lambda: "")
    assert classify("some ambiguous text") is None
    assert called == []


def test_make_self_classify_with_model_calls_endpoint(monkeypatch):
    """A loaded chat model lets the tie-break through and cleans the reply."""
    class FakeResp:
        ok = True
        def json(self):
            return {"choices": [{"message": {"content": "YAML."}}]}
    called = []
    def fake_request(method, url, **k):
        called.append((method, url))
        return FakeResp()
    # _make_self_classify goes through localm.selfclient.self_request, which
    # issues requests.request(method, ...), not requests.post directly.
    monkeypatch.setattr(requests, "request", fake_request)
    classify = plug._make_self_classify("https://127.0.0.1:65535/v1", lambda: "mymodel")
    assert classify("key: value") == "yaml"
    assert called == [("POST", "https://127.0.0.1:65535/v1/chat/completions")]


# --------------------------------------------------------------------------- #
#  Pure heuristic / classify_format unit tests (lazy import)                   #
# --------------------------------------------------------------------------- #

def test_sniff_text_format_structural_shapes():
    from localm.rag.extract import sniff_text_format
    assert sniff_text_format('{"a": 1, "b": [1,2]}') == "json"
    assert sniff_text_format('[1, 2, 3]') == "json"
    assert sniff_text_format('a,b,c\n1,2,3\n4,5,6') == "csv"
    assert sniff_text_format('# Title\n\nprose\n\n## Section\n\n- item') == "markdown"
    assert sniff_text_format('<!DOCTYPE html>\n<html><body>hi</body></html>') == "html"
    assert sniff_text_format('<?xml version="1.0"?>\n<root><a/></root>') == "xml"
    assert sniff_text_format('name: localm\nversion: 1\nport: 8080') == "yaml"
    assert sniff_text_format('[server]\nhost = 0.0.0.0\nport = 80') == "ini"
    # unsure -> None so the caller may tie-break / fall back to "text"
    assert sniff_text_format('just some plain english words here') is None
    # a lone '#' comment must NOT be mislabeled markdown (no corroboration)
    assert sniff_text_format('# a lone shell/python comment\nx=1') is None


def test_classify_format_prefers_known_extension():
    from localm.rag.extract import classify_format
    # extension is authoritative even when the structural sniff is inconclusive
    assert classify_format("print('hi')\nx = 1\n", "script.py") == "python"
    assert classify_format("anything at all", "notes.md") == "markdown"
    assert classify_format('{"a":1}', "data.json") == "json"


def test_classify_format_heuristic_for_unknown_extension():
    from localm.rag.extract import classify_format
    assert classify_format('{"a": 1}', "weird.blah") == "json"
    assert classify_format('x,y,z\n1,2,3\n4,5,6', "weird.blah") == "csv"


def test_classify_format_defaults_to_text_without_model():
    from localm.rag.extract import classify_format
    # unknown ext + ambiguous content + no classify_fn -> "text", no model call
    assert classify_format("plain words with no structure at all", "f.zzz") == "text"


def test_classify_format_tiebreak_gated_and_cached():
    from localm.rag.extract import classify_format, _EXT_CLASSIFICATION_CACHE
    _EXT_CLASSIFICATION_CACHE.clear()
    calls = []
    def fake(snippet):
        calls.append(snippet)
        return "javascript"
    assert classify_format("function add(a,b){return a+b}",
                           "a.custom_code", classify_fn=fake) == "javascript"
    assert len(calls) == 1
    # same unknown extension -> cached, classify_fn not called again
    assert classify_format("function sub(a,b){return a-b}",
                           "b.custom_code", classify_fn=fake) == "javascript"
    assert len(calls) == 1
    assert _EXT_CLASSIFICATION_CACHE[".custom_code"] == "javascript"


def test_classify_format_config_disables_tiebreak(monkeypatch):
    from localm.rag.extract import classify_format, _EXT_CLASSIFICATION_CACHE
    import localm.config as cfg
    _EXT_CLASSIFICATION_CACHE.clear()
    monkeypatch.setattr(cfg, "load_config",
                        lambda: {"rag_classify_unknown_files": False})
    calls = []
    def fake(snippet):
        calls.append(snippet)
        return "javascript"
    # config off -> tie-break suppressed, label "text", classify_fn never called
    assert classify_format("function add(a,b){}",
                           "x.custom_code", classify_fn=fake) == "text"
    assert calls == []


def test_classify_format_tiebreak_attempted_at_most_once_per_ext(monkeypatch):
    # Simulate a resident NON-chat engine: classify_fn is actually invoked (not
    # short-circuited) but returns None because it cannot classify. That outcome
    # is cached, so a SECOND same-extension unclear file does not re-invoke it.
    from localm.rag.extract import classify_format, _EXT_CLASSIFICATION_CACHE
    import localm.config as cfg
    _EXT_CLASSIFICATION_CACHE.clear()
    monkeypatch.setattr(cfg, "load_config",
                        lambda: {"rag_classify_unknown_files": True})
    calls = []
    def failing_classify(snippet):
        calls.append(snippet)
        return None
    assert classify_format("weird stuff no structure one",
                           "one.zzq", classify_fn=failing_classify) == "text"
    assert classify_format("weird stuff no structure two",
                           "two.zzq", classify_fn=failing_classify) == "text"
    assert len(calls) == 1                       # attempted once, then cached
    assert _EXT_CLASSIFICATION_CACHE[".zzq"] == "text"


def test_sniff_rejects_comma_prose_and_annotated_code():
    from localm.rag.extract import sniff_text_format
    # comma-bearing prose / call-style code must NOT be labeled csv
    assert sniff_text_format("Hello, everyone\nGoodbye, everyone") is None
    assert sniff_text_format("foo(a, b);\nbar(c, d);") is None
    # colon-annotated code must NOT be labeled yaml (assignment '=' present)
    assert sniff_text_format('x: int = 1\ny: int = 2') is None
    # a genuine csv (>=3 rows, uniform comma count) is still detected
    assert sniff_text_format("id,name\n1,ann\n2,bob") == "csv"


def test_sniff_html_fragment_vs_xml():
    from localm.rag.extract import sniff_text_format
    # a headerless HTML fragment is html, not xml
    assert sniff_text_format('<div class="card"><p>Hi</p></div>') == "html"
    # an explicit XML declaration stays xml even with an <a>-like element
    assert sniff_text_format('<?xml version="1.0"?>\n<root><a/></root>') == "xml"
    # a generic tag with no HTML cues is xml
    assert sniff_text_format('<config><item value="1"/></config>') == "xml"


def test_sniff_large_json_uses_bracket_match(monkeypatch):
    import localm.rag.extract as ex
    # Force the "too big to parse" branch: confirm structurally by the closer.
    monkeypatch.setattr(ex, "_JSON_PARSE_MAX", 8)
    assert ex.sniff_text_format('{"a": 1, "b": 2}') == "json"
    # no closing bracket -> unsure, not a false "json"
    assert ex.sniff_text_format('{"a": 1, "b": 2   ') is None
