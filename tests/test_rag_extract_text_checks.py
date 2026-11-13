# SPDX-License-Identifier: AGPL-3.0-or-later
"""A file named as text but holding binary data is refused instead of read as
cp1252 or UTF-16 mojibake; real text in any of the encodings the extractor reads
still passes, a coloured log included. A format tie-break that gave no label is not remembered for the extension.
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
    ("table.txt", b"".join(i.to_bytes(4, "little") for i in range(400)), "binary data"),
    ("table.log", b"".join(i.to_bytes(4, "big") for i in range(400)), "binary data"),
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
])
def test_text_in_every_supported_encoding_still_reads(data, expected):
    assert extract_bytes(data, "a.txt").strip() == expected


_LEGACY_TEXT = {
    "shift_jis": "これは日本語のテキストです。東京は大きな都市です。",
    "gbk": "这是一个中文文本文件。我们在这里测试编码。",
    "big5": "這是一個中文文本文件。我們在這裡測試編碼。",
    "euc-kr": "이것은 한국어 텍스트 파일입니다. 우리는 여기서 인코딩을 테스트합니다.",
}


@pytest.mark.parametrize("encoding", sorted(_LEGACY_TEXT))
def test_legacy_multibyte_text_reads_in_its_own_script(encoding):
    text = _LEGACY_TEXT[encoding]
    data = ((text + "\n") * 5).encode(encoding)

    out = extract_bytes(data, "notes.txt")

    assert out.splitlines()[0] == text


def test_an_ansi_coloured_log_is_read_without_its_colour_codes():
    data = "".join(
        f"\x1b[32m2026-11-13 10:00:{i:02d}\x1b[0m \x1b[1;34mINFO\x1b[0m worker {i} ready\n"
        "\x1b]0;build\x07"
        for i in range(50)).encode()

    out = extract_bytes(data, "build.log")

    assert out.splitlines()[0] == "2026-11-13 10:00:00 INFO worker 0 ready"
    assert "\x1b" not in out and "\x07" not in out
    assert extract.sniff_format(data, "build.weird") == ".txt"


def test_a_tie_break_without_a_label_is_tried_again_later(monkeypatch):
    import localm.config as cfg
    monkeypatch.setattr(cfg, "load_config", lambda: {"rag_classify_unknown_files": True})
    extract._EXT_CLASSIFICATION_CACHE.clear()
    extract._EXT_CLASSIFICATION_RETRY_AT.clear()
    clock = {"t": 1000.0}
    monkeypatch.setattr(extract.time, "monotonic", lambda: clock["t"])
    answers = [None, "yaml"]
    calls = []

    def flaky(snippet):
        calls.append(snippet)
        return answers[len(calls) - 1]

    text = "plain words with no structure at all"
    assert extract.classify_format(text, "a.weird", classify_fn=flaky) == "text"
    assert extract.classify_format(text, "b.weird", classify_fn=flaky) == "text"
    assert len(calls) == 1, "within the retry window the failure is not re-asked"
    clock["t"] += extract._FAILED_TIEBREAK_RETRY_S
    assert extract.classify_format(text, "c.weird", classify_fn=flaky) == "yaml"
    clock["t"] += 10 * extract._FAILED_TIEBREAK_RETRY_S
    assert extract.classify_format(text, "d.weird", classify_fn=flaky) == "yaml"
    assert len(calls) == 2, "a real label is kept"
    extract._EXT_CLASSIFICATION_CACHE.clear()
    extract._EXT_CLASSIFICATION_RETRY_AT.clear()


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
