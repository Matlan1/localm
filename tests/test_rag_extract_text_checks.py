# SPDX-License-Identifier: AGPL-3.0-or-later
"""A file named as text but holding binary data is refused instead of read as
cp1252 mojibake; real text in any of the encodings the extractor reads still
passes. A format tie-break that gave no label is not remembered for the extension.
The embedder's warm-up stage names the real reason a configured model is
unavailable."""

from __future__ import annotations

import huggingface_hub
import pytest

from localm.rag import extract
from localm.rag.extract import ExtractError, extract_bytes
from tests._net_config import set_config

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + bytes(range(256)) * 4


@pytest.mark.parametrize("name,data,what", [
    ("photo.txt", _PNG, "a PNG image"),
    ("notes.md", b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj", "a PDF document"),
    ("data.csv", bytes(range(1, 32)) * 40, "binary data"),
])
def test_binary_named_as_text_is_refused(name, data, what):
    with pytest.raises(ExtractError) as exc:
        extract_bytes(data, name)
    assert what in str(exc.value)
    assert name in str(exc.value)


@pytest.mark.parametrize("data,expected", [
    ("Grüße aus Graz\n".encode(), "Grüße aus Graz"),
    ("Grüße aus Graz\n".encode("cp1252"), "Grüße aus Graz"),
    ("Grüße aus Graz\n".encode("utf-16-le"), "Grüße aus Graz"),
    (b"\xff\xfe" + "Grüße\n".encode("utf-16-le"), "Grüße"),
    (b"", ""),
])
def test_text_in_every_supported_encoding_still_reads(data, expected):
    assert extract_bytes(data, "a.txt").strip() == expected


def test_a_tie_break_without_a_label_is_not_cached(monkeypatch):
    import localm.config as cfg
    monkeypatch.setattr(cfg, "load_config", lambda: {"rag_classify_unknown_files": True})
    extract._EXT_CLASSIFICATION_CACHE.clear()
    answers = [None, "", "yaml"]
    calls = []

    def flaky(snippet):
        calls.append(snippet)
        return answers[len(calls) - 1]

    text = "plain words with no structure at all"
    assert extract.classify_format(text, "a.weird", classify_fn=flaky) == "text"
    assert extract.classify_format(text, "b.weird", classify_fn=flaky) == "text"
    assert ".weird" not in extract._EXT_CLASSIFICATION_CACHE
    assert extract.classify_format(text, "c.weird", classify_fn=flaky) == "yaml"
    assert extract.classify_format(text, "d.weird", classify_fn=flaky) == "yaml"
    assert len(calls) == 3
    extract._EXT_CLASSIFICATION_CACHE.clear()


def test_the_warm_up_stage_names_why_a_configured_model_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda *a, **k: pytest.fail("no download under net_mode off"))
    from localm.inference import embedder as emb
    set_config(monkeypatch, lambda: {"net_mode": "off"})
    emb.reset_embedder()
    stages = []
    try:
        assert emb.get_embedder(on_progress=stages.append) is None
        assert stages, "the warm-up reports its stages"
        assert "net_mode=off" in stages[-1], stages
        assert stages[-1] != "No embedding model is configured or available."
    finally:
        emb.reset_embedder()
