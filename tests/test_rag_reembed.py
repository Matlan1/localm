"""Changing the embedding model must not be a dead end, and a chunk whose text
contains an exotic line separator must not vanish.

The separator defect is the root cause of the first's whole symptom cluster, so
each test here asserts the BEHAVIOUR (records survive, vectors get rebuilt), never
the wording of a warning.
"""
import json
import os

import pytest

from localm.jsonl import UNSAFE_SEPARATORS, dumps_line, split_jsonl
from localm.rag.store import Collection


def _embedder(dim, fail_after=-1):
    seen = {"n": 0}

    def embed(texts):
        out = []
        for t in texts:
            seen["n"] += 1
            if 0 <= fail_after < seen["n"]:
                raise RuntimeError("embedder died")
            h = sum(ord(c) for c in t[:64])
            out.append([(h % 97) / 97.0] * dim)
        return out

    embed.seen = seen
    return embed


def _collection(tmp_path, texts, dim=8):
    c = Collection("kb", base=tmp_path).create()
    c._chunks = [{"source": f"doc{i}.txt", "pos": i, "text": t}
                 for i, t in enumerate(texts)]
    c._vectors = _embedder(dim)(texts)
    c._save()
    return c


# --------------------------------------------------------------------------- #
#  The round-trip: every separator str.splitlines() splits on                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sep", sorted(UNSAFE_SEPARATORS))
def test_chunk_text_containing_a_line_separator_survives_save_and_reload(tmp_path, sep):
    """json.dumps(ensure_ascii=False) emits U+0085/U+2028/U+2029 raw, and
    str.splitlines() splits on them - which tore one record into two unparseable
    fragments. Measured in the wild: 36 raw U+0085 cost 26 chunks per load/save."""
    texts = ["plain one", f"before{sep}after", "plain two"]
    _collection(tmp_path, texts)

    reloaded = Collection("kb", base=tmp_path)
    assert len(reloaded._chunks) == 3, (
        f"a chunk containing {sep!r} was dropped on reload")
    assert reloaded._chunks[1]["text"] == f"before{sep}after"
    assert not reloaded.corrupt, "healthy data reported as corrupt"


@pytest.mark.parametrize("sep", sorted(UNSAFE_SEPARATORS))
def test_load_save_load_never_deletes_records(tmp_path, sep):
    """The damaging half: the reader dropped records, then _save() rewrote the
    file from the survivors, permanently deleting the user's data."""
    texts = ["a", f"x{sep}y", "b", f"p{sep}q", "c"]
    _collection(tmp_path, texts)
    path = tmp_path / "kb" / "chunks.jsonl"
    before = len([ln for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()])

    Collection("kb", base=tmp_path)._save()          # ordinary persist cycle

    after = len([ln for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()])
    assert after == before == 5, f"load->save changed the record count {before} -> {after}"


def test_writer_emits_no_raw_separator(tmp_path):
    """Belt and braces: the file must be safe for ANY line-oriented reader,
    including a JS JSON.parse consumer that U+2028/U+2029 would break."""
    text = "".join(f"seg{i}{s}" for i, s in enumerate(sorted(UNSAFE_SEPARATORS)))
    _collection(tmp_path, [text])
    raw = (tmp_path / "kb" / "chunks.jsonl").read_text(encoding="utf-8")
    for sep in UNSAFE_SEPARATORS:
        assert sep not in raw, f"{sep!r} written raw into chunks.jsonl"
    assert len(raw.splitlines()) == 1, "even splitlines() must see one record now"


def test_split_jsonl_only_splits_on_line_feed():
    rec = dumps_line({"text": "a" + "" + "b" + " " + "c"})
    assert len(split_jsonl(rec)) == 1
    assert json.loads(rec)["text"] == "ab c"
    # CRLF files still yield clean records
    assert split_jsonl('{"a":1}\r\n{"b":2}') == ['{"a":1}', '{"b":2}']


# --------------------------------------------------------------------------- #
#  Re-embedding: the model change must be recoverable, from stored text alone   #
# --------------------------------------------------------------------------- #
def test_reembed_rebuilds_every_vector_at_the_new_dimension(tmp_path):
    c = _collection(tmp_path, ["alpha", "beta", "gamma"], dim=8)
    assert c._vec_dim == 8

    res = c.reembed(embed_fn=_embedder(32), model_name="new-model")

    assert res == {"chunks": 3, "dim": 32, "model": "new-model"}
    on_disk = json.loads((tmp_path / "kb" / "vectors.json").read_text(encoding="utf-8"))
    assert on_disk["dim"] == 32 and len(on_disk["vectors"]) == 3
    reloaded = Collection("kb", base=tmp_path)
    assert reloaded.vector_degrade_reason is None
    assert reloaded.embedding_model() == "new-model"


def test_reembed_needs_no_source_files(tmp_path):
    """The whole point: `rag repair --embed` re-indexes from the ORIGINAL FILES
    and cannot help when they are gone. Re-embed works from stored text."""
    c = _collection(tmp_path, ["alpha", "beta"], dim=8)
    assert not any(p.exists() for p in [tmp_path / "doc0.txt", tmp_path / "doc1.txt"])
    assert c.reembed(embed_fn=_embedder(16))["dim"] == 16


def test_a_failed_reembed_leaves_the_previous_index_intact(tmp_path):
    """A half-dimension index would mis-score every query, so a dying embedder
    must change nothing."""
    c = _collection(tmp_path, ["a", "b", "c", "d"], dim=8)
    vf = tmp_path / "kb" / "vectors.json"
    before = vf.read_text(encoding="utf-8")

    with pytest.raises(Exception):
        c.reembed(embed_fn=_embedder(64, fail_after=2))

    assert vf.read_text(encoding="utf-8") == before, "index changed after a failed re-embed"
    assert Collection("kb", base=tmp_path).vector_degrade_reason is None


def test_reembed_refuses_a_short_result_rather_than_saving_it(tmp_path):
    c = _collection(tmp_path, ["a", "b", "c"], dim=8)
    with pytest.raises(RuntimeError, match="vectors for 3 chunks"):
        c.reembed(embed_fn=lambda texts: [[0.5] * 16])      # one vector, three chunks
    assert Collection("kb", base=tmp_path)._vec_dim == 8


def test_reembed_refuses_ragged_dimensions(tmp_path):
    c = _collection(tmp_path, ["a", "b"], dim=8)
    with pytest.raises(RuntimeError, match="inconsistent vector sizes"):
        c.reembed(embed_fn=lambda texts: [[0.5] * 16, [0.5] * 32])
    assert Collection("kb", base=tmp_path)._vec_dim == 8


def test_dimension_mismatch_error_names_the_collection_and_a_working_command(tmp_path):
    """The old text said "Rebuild it (delete and re-add)" - destructive, and it
    named neither the collection nor a command that worked."""
    c = _collection(tmp_path, ["alpha"], dim=8)
    doc = tmp_path / "extra.txt"
    doc.write_text("some new content", encoding="utf-8")

    with pytest.raises(ValueError) as ei:
        c.add_paths([str(doc)], embed_fn=_embedder(32))

    msg = str(ei.value)
    assert "'kb'" in msg, "the error must name the collection"
    assert "localm rag reembed kb" in msg, "the error must give the command that works"
    assert "delete and re-add" not in msg, "must no longer tell the user to delete their data"


def test_degrade_warning_fires_once_per_process_not_once_per_instance(tmp_path, caplog):
    """_load() runs from __init__ and every request builds a FRESH Collection, so
    the old instance-level guard re-armed constantly: 25+ identical WARNING lines
    in one real session, several twice inside a single request."""
    from localm.rag import store as store_mod

    _collection(tmp_path, ["a", "b", "c"], dim=8)
    (tmp_path / "kb" / "vectors.json").write_text(
        json.dumps({"dim": 8, "vectors": [[0.1] * 8]}), encoding="utf-8")  # 1 vector, 3 chunks
    store_mod._WARNED_DEGRADES.clear()

    with caplog.at_level("WARNING"):
        instances = [Collection("kb", base=tmp_path) for _ in range(6)]

    degrade_lines = [r for r in caplog.records if "vectors" in r.getMessage()]
    assert len(degrade_lines) == 1, (
        f"expected ONE warning across 6 loads, got {len(degrade_lines)}")
    # ...but every instance must still KNOW it is degraded, so stats()/the GUI
    # badge stay accurate.
    assert all(i.vector_degrade_reason for i in instances), (
        "deduping the log also hid the state from stats()")


def test_a_new_degrade_reason_still_warns(tmp_path, caplog):
    """Dedup must be per (collection, reason) - a fault that changes shape is a
    new fault and must not hide behind an earlier one."""
    from localm.rag import store as store_mod

    _collection(tmp_path, ["a", "b"], dim=8)
    store_mod._WARNED_DEGRADES.clear()
    kb = tmp_path / "kb"

    with caplog.at_level("WARNING"):
        (kb / "vectors.json").write_text(json.dumps({"dim": 8, "vectors": [[0.1] * 8]}),
                                         encoding="utf-8")
        Collection("kb", base=tmp_path)
        first = len([r for r in caplog.records if "vectors" in r.getMessage()])
        (kb / "vectors.json").write_text("{not json", encoding="utf-8")
        Collection("kb", base=tmp_path)
        second = len([r for r in caplog.records if "vectors" in r.getMessage()])

    assert first == 1, f"first fault should warn once, got {first}"
    assert second > first, "a DIFFERENT fault must still warn, not be deduped away"


def test_set_aside_vector_indexes_are_pruned_to_the_newest_few(tmp_path):
    """Each set-aside file is a full copy of the index. A real install had three
    totalling 88 MB and the allocator would have gone on to twenty."""
    from localm.rag import store as store_mod

    c = _collection(tmp_path, ["a", "b"], dim=8)
    kb = tmp_path / "kb"
    names = ["vectors.json.rejected"] + [f"vectors.json.rejected.{n}" for n in range(2, 8)]
    for i, n in enumerate(names):
        p = kb / n
        p.write_text("x" * 1000, encoding="utf-8")
        os.utime(p, (1_000_000 + i, 1_000_000 + i))     # oldest first
    assert len(c._rejected_vector_files()) == 7

    c._prune_rejected_vectors()

    left = sorted(p.name for p in c._rejected_vector_files())
    assert len(left) == store_mod._MAX_REJECTED_KEPT, f"kept {left}"
    # the NEWEST must survive: pruning orders by mtime, never by name (a name sort
    # puts .20 before .3 and would delete the wrong ones past nine rejections)
    assert "vectors.json.rejected.7" in left
    assert "vectors.json.rejected" not in left, "oldest should have been pruned"


def test_pruning_never_touches_the_live_index(tmp_path):
    c = _collection(tmp_path, ["a", "b"], dim=8)
    kb = tmp_path / "kb"
    live = (kb / "vectors.json").read_text(encoding="utf-8")
    for n in ["vectors.json.rejected"] + [f"vectors.json.rejected.{i}" for i in range(2, 8)]:
        (kb / n).write_text("x", encoding="utf-8")

    c._prune_rejected_vectors()

    assert (kb / "vectors.json").read_text(encoding="utf-8") == live
    assert Collection("kb", base=tmp_path).vector_degrade_reason is None


#: numpy is bound at MODULE level (store._numpy) to close the concurrent-first-
#: import race, so these tests must patch THAT, never sys.modules["numpy"].
def _unusable_numpy():
    """A numpy that IMPORTS but has no asarray - the observed CI shape."""
    import types
    return types.ModuleType("numpy")


@pytest.mark.parametrize("fn,args", [
    ("_cosine", ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])),
    ("_vectors_finite", ([[1.0, 2.0]],)),
])
def test_numpy_present_but_unusable_falls_back_at_every_site(monkeypatch, fn, args):
    """numpy can be PARTIALLY INITIALISED, so `import numpy` SUCCEEDS while
    np.asarray is absent. Catching only ImportError is not enough at either site.

    _vectors_finite is the more consequential of the two: it returns the boolean
    deciding whether the vector store is TRUSTED, so an escaping AttributeError
    errors the whole query instead of degrading to BM25 with a surfaced
    reason."""
    from localm.rag import store as store_mod

    monkeypatch.setattr(store_mod, "_numpy", _unusable_numpy())
    store_mod._NUMPY_DEGRADE_LOGGED.clear()

    result = getattr(store_mod, fn)(*args)          # must not raise

    assert result == pytest.approx(1.0) if fn == "_cosine" else result is True


def test_the_unusable_numpy_degrade_is_announced_once_not_per_chunk(monkeypatch, caplog):
    from localm.rag import store as store_mod

    monkeypatch.setattr(store_mod, "_numpy", _unusable_numpy())
    store_mod._NUMPY_DEGRADE_LOGGED.clear()

    with caplog.at_level("WARNING"):
        assert store_mod._cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
        store_mod._cosine([1.0, 1.0], [1.0, 1.0])
        store_mod._vectors_finite([[1.0, 2.0]])

    warned = [r for r in caplog.records if "numpy is present but unusable" in r.getMessage()]
    assert len(warned) == 1, f"expected exactly one warning across 3 calls, got {len(warned)}"


def test_a_namespace_stub_numpy_is_reported_as_a_broken_environment(monkeypatch, caplog):
    """A bare directory named `numpy` on sys.path resolves as a PEP 420 implicit
    namespace package: `import numpy` SUCCEEDS, the module has no attributes, and
    `__file__` is None. That is NOT "numpy is missing" - something has put a fake
    one on the path, which breaks anything else here that imports numpy.

    It must not be collapsed into a routine "numpy unavailable" message."""
    import types

    from localm.rag import store as store_mod

    stub = types.ModuleType("numpy")
    stub.__path__ = ["D:/stray/numpy"]          # namespace packages carry __path__
    monkeypatch.setattr(store_mod, "_numpy", stub)
    monkeypatch.setattr(store_mod, "_NUMPY_IS_STUB", True)
    store_mod._NUMPY_DEGRADE_LOGGED.clear()

    with caplog.at_level("WARNING"):
        assert store_mod._cosine([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)

    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "NAMESPACE PACKAGE" in msg, "the broken-environment case must be named as such"
    assert "D:/stray/numpy" in msg, "the message must name the artefact to delete"


def test_an_ordinary_unusable_numpy_is_not_called_a_broken_environment(monkeypatch, caplog):
    """The discriminator must cut both ways, or it is just a louder message."""
    import types

    from localm.rag import store as store_mod

    monkeypatch.setattr(store_mod, "_numpy", types.ModuleType("numpy"))
    monkeypatch.setattr(store_mod, "_NUMPY_IS_STUB", False)
    store_mod._NUMPY_DEGRADE_LOGGED.clear()

    with caplog.at_level("WARNING"):
        store_mod._cosine([1.0, 2.0], [1.0, 2.0])

    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "present but unusable" in msg
    assert "NAMESPACE PACKAGE" not in msg, "a non-stub was mislabelled as a broken environment"


def test_numpy_absent_entirely_still_works(monkeypatch):
    """numpy is optional and arrives transitively - None must be a clean path."""
    from localm.rag import store as store_mod

    monkeypatch.setattr(store_mod, "_numpy", None)
    assert store_mod._cosine([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)
    assert store_mod._vectors_finite([[1.0, 2.0]]) is True
    assert store_mod._vectors_finite([[float("nan")]]) is False


def test_a_usable_numpy_failing_for_a_real_reason_still_propagates(monkeypatch):
    """The widened except must not become a bare catch-all: a broken install that
    fails numerically has to surface, not silently halve retrieval quality."""
    import types

    from localm.rag import store as store_mod

    fake = types.ModuleType("numpy")

    def _boom(*a, **k):
        raise ValueError("genuine numerical failure")

    fake.asarray = _boom
    monkeypatch.setattr(store_mod, "_numpy", fake)

    with pytest.raises(ValueError, match="genuine numerical failure"):
        store_mod._cosine([1.0, 2.0], [1.0, 2.0])


def test_numpy_is_never_imported_lazily_inside_a_function():
    """Both numpy users run on the shared plugin ThreadPoolExecutor. CPython's
    import lock is PER-MODULE, so a thread finding numpy already in sys.modules
    does NOT wait for it to finish initialising - it gets the module as it stands,
    possibly with no asarray. numpy must therefore stay a module-level binding.

    An edit re-introducing `import numpy` inside one of these functions restores
    that race while every other test here still passes, because the
    state-handling fallback masks it. This pins the absence."""
    import ast
    import inspect
    import textwrap

    from localm.rag import store as store_mod

    # Parsed, not string-matched: these functions legitimately DISCUSS `import
    # numpy` in their comments, and a substring check would flag that as a
    # violation.
    for fn in (store_mod._cosine, store_mod._vectors_finite):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert "numpy" not in imported, (
            f"{fn.__name__} imports numpy lazily again - that reopens the "
            "concurrent-first-import race the module-level binding closed")


def _api():
    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app
    return TestClient(create_app(None), raise_server_exceptions=False)


def _post(client, body: str):
    return client.post("/v1/chat/completions", content=body,
                       headers={"content-type": "application/json"})


_MSGS = '{"model":"localm","messages":[{"role":"user","content":"hi"}],'


def test_max_tokens_zero_explains_itself_instead_of_dumping_pydantic():
    """Reported verbatim by the maintainer as
    `422: {"detail":[{"type":"greater_than_equal","loc":["body","max_tokens"],...}]}`.
    0 looks like a legitimate "no limit"; the real reason it is refused (it
    collides with the engine's internal unlimited sentinel) is not guessable."""
    r = _post(_api(), _MSGS + '"max_tokens":0}')
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert isinstance(detail, str), "detail must be a readable sentence, not a raw error list"
    assert "max_tokens" in detail and "no limit" in detail
    assert "greater_than_equal" not in detail, "pydantic's machine type leaked into the message"


def test_validation_errors_keep_the_structured_form_under_errors():
    """Readability must not cost machine-readability."""
    r = _post(_api(), _MSGS + '"max_tokens":-5}')
    body = r.json()
    assert isinstance(body["detail"], str)
    assert isinstance(body["errors"], list) and body["errors"], "structured form was dropped"
    assert body["errors"][0]["loc"][-1] == "max_tokens"


def test_a_non_finite_number_still_renders_a_422_not_a_500():
    """JSONResponse serializes with allow_nan=False, so a NaN reaching the error
    detail crashes the handler into a 500. Reformatting the message must not
    let one through."""
    r = _post(_api(), _MSGS + '"temperature":NaN}')
    assert r.status_code == 422, f"got {r.status_code}, the 422 handler crashed"
    assert "temperature" in r.json()["detail"]


def test_a_missing_field_does_not_echo_the_whole_request_body():
    r = _post(_api(), '{"model":"localm"}')
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "messages" in detail
    assert "localm" not in detail, "the whole body was echoed back as 'got ...'"


def test_reembed_after_a_mismatch_lets_the_add_succeed(tmp_path):
    """End to end: hit the refusal, follow the advice, and the add now works -
    which is the whole user journey that was a dead end."""
    c = _collection(tmp_path, ["alpha"], dim=8)
    doc = tmp_path / "extra.txt"
    doc.write_text("some new content", encoding="utf-8")

    with pytest.raises(ValueError):
        c.add_paths([str(doc)], embed_fn=_embedder(32))

    Collection("kb", base=tmp_path).reembed(embed_fn=_embedder(32), model_name="m32")
    Collection("kb", base=tmp_path).add_paths([str(doc)], embed_fn=_embedder(32))

    final = Collection("kb", base=tmp_path)
    assert final._vec_dim == 32
    assert len(final._chunks) >= 2
    assert final.vector_degrade_reason is None


# --------------------------------------------------------------------------- #
#  The REAL ingest path, end to end                                           #
# --------------------------------------------------------------------------- #
#  Everything above builds its collection by assigning _chunks/_vectors directly
#  via _collection(). These drive a collection produced by the real upload path,
#  whose source bytes were never on disk at all, through re-embed as far as a
#  correct QUERY.
_TOPICS = ["zebra", "quasar", "tugboat", "marzipan"]


def _topic_embedder(dim, fail_after=-1):
    """An embedder that genuinely DISCRIMINATES: one axis per topic word.

    _embedder() above hashes the text into a single repeated value, so every
    vector is a scalar multiple of every other and cosine cannot tell any two
    chunks apart. That is harmless where only the DIMENSION is under test, but a
    query assertion written against it would pass no matter which chunk came
    back. Here the retrieved chunk is the property, so the embedding has to be
    able to be wrong.
    """
    seen = {"n": 0}

    def embed(texts):
        out = []
        for t in texts:
            seen["n"] += 1
            if 0 <= fail_after < seen["n"]:
                raise RuntimeError("embedder died")
            v = [0.0] * dim
            for i, w in enumerate(_TOPICS):
                if w in t.lower():
                    v[i % dim] = 1.0
            if not any(v):
                v[dim - 1] = 1.0
            out.append(v)
        return out

    return embed


def _uploads_only(tmp_path, dim=384, model="bge-small-en-v1.5"):
    """A collection built ENTIRELY from uploads: no source file ever exists."""
    coll = Collection("eng", base=tmp_path).create()
    coll.add_uploads(
        [{"filename": "a.txt", "data": b"the zebra grazes on the open plain"},
         {"filename": "b.txt", "data": b"a quasar is an extremely luminous galactic nucleus"},
         {"filename": "c.txt", "data": b"the tugboat hauled the barge upriver"}],
        embed_fn=_topic_embedder(dim), model_name=model)
    return Collection("eng", base=tmp_path)


def test_an_uploads_only_collection_survives_a_model_switch_end_to_end(tmp_path):
    """After switching the embedding model, a collection built entirely from
    uploads whose source files never existed can be re-embedded in place and
    QUERIED CORRECTLY, without the user deleting anything."""
    coll = _uploads_only(tmp_path)
    assert coll._vec_dim == 384
    n_chunks = len(coll._chunks)
    assert not list(tmp_path.rglob("*.txt")), "no source file may exist on disk"

    # The refusal is correct (mixed dimensions mis-score every query) and stays
    # actionable rather than telling anyone to delete their data.
    with pytest.raises(ValueError) as ei:
        coll.add_uploads([{"filename": "d.txt", "data": b"marzipan is made of almonds"}],
                         embed_fn=_topic_embedder(1024))
    assert "localm rag reembed eng" in str(ei.value)
    assert len(Collection("eng", base=tmp_path)._chunks) == n_chunks, "the refusal lost chunks"

    Collection("eng", base=tmp_path).reembed(
        embed_fn=_topic_embedder(1024), model_name="Qwen3-Embedding-0.6B")

    after = Collection("eng", base=tmp_path)
    # Assert on the DATA before the dimension: a re-embed that loses content is
    # the finding, and must be reported as such rather than as a number.
    assert len(after._chunks) == n_chunks, "re-embed lost chunks"
    assert len(after.documents()) == 3, "re-embed lost documents"
    assert after._vec_dim == 1024
    assert after.vector_degrade_reason is None
    assert after.embedding_model() == "Qwen3-Embedding-0.6B"

    # Queried CORRECTLY: a loadable but WRONG index fails only here.
    hits = after.query("quasar", k=1, embed_fn=_topic_embedder(1024))
    assert hits and "quasar" in hits[0]["text"].lower(), f"wrong chunk retrieved: {hits}"

    # And the add that was refused now goes through.
    after.add_uploads([{"filename": "d.txt", "data": b"marzipan is made of almonds"}],
                      embed_fn=_topic_embedder(1024), model_name="Qwen3-Embedding-0.6B")
    final = Collection("eng", base=tmp_path)
    assert final.vector_degrade_reason is None
    hits = final.query("marzipan", k=1, embed_fn=_topic_embedder(1024))
    assert hits and "marzipan" in hits[0]["text"].lower()


def test_a_failed_reembed_leaves_the_index_intact_across_MULTIPLE_batches(tmp_path):
    """The multi-batch case, which the single-batch test above cannot express.

    test_a_failed_reembed_leaves_the_previous_index_intact embeds 4 chunks at the
    default batch of 32, so the whole run is ONE embed_fn call: the embedder
    raises INSIDE it, before any result is ever appended, and a write-as-you-go
    implementation never reaches a write there. Only a failure in a LATER batch,
    after an earlier one has already succeeded, can tell the two apart, and that
    is the shape that leaves a half-written mixed-dimension index behind.
    """
    coll = _uploads_only(tmp_path)
    vf = tmp_path / "eng" / "vectors.json"
    before = vf.read_text(encoding="utf-8")

    with pytest.raises(RuntimeError):
        coll.reembed(embed_fn=_topic_embedder(1024, fail_after=1), batch=1,
                     model_name="Qwen3-Embedding-0.6B")

    assert vf.read_text(encoding="utf-8") == before, "a failed re-embed rewrote the index"
    reloaded = Collection("eng", base=tmp_path)
    assert reloaded._vec_dim == 384, "the collection was left at a half-written dimension"
    assert reloaded.vector_degrade_reason is None
