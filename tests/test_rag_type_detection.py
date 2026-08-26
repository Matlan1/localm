# SPDX-License-Identifier: AGPL-3.0-or-later
import zipfile
import io
import pytest
from localm.rag import extract_text, ExtractError
from localm.rag.extract import sniff_format, _EXT_CLASSIFICATION_CACHE
from tests.test_rag import _tiny_pdf

def _docx_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", "xml")
    return buf.getvalue()

def _zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("notes.txt", "some notes")
    return buf.getvalue()

@pytest.mark.parametrize("data,filename,expected", [
    (_tiny_pdf("hello"), "file.bin", ".pdf"),
    (_docx_bytes(), "file.dat", ".docx"),
    (_zip_bytes(), "file.custom", ".zip"),
    (b"Hello, this is plain text!", "file.xyz", ".txt"),
    (b"\x00\x01\x02\x03\x04\xff", "file.xyz", None),
], ids=["pdf", "docx", "zip", "plain_text", "binary"])
def test_sniff_format(data, filename, expected):
    assert sniff_format(data, filename) == expected

def test_extract_custom_extension_text(tmp_path):
    f = tmp_path / "notes.custom_ext"
    f.write_text("my custom content", encoding="utf-8")
    assert extract_text(f) == "my custom content"

def test_extract_extensionless_pdf(tmp_path):
    f = tmp_path / "mypdf"
    f.write_bytes(_tiny_pdf("grounded text"))
    try:
        text = extract_text(f)
        assert "grounded text" in text
    except ExtractError as e:
        if "pypdf" in str(e):
            pytest.skip("pypdf not installed, skipping pdf extraction check")
        else:
            raise

def test_extract_zip_archive(tmp_path):
    f = tmp_path / "archive.zip"
    with zipfile.ZipFile(f, "w") as zf:
        zf.writestr("doc1.txt", "content of doc 1")
        zf.writestr("doc2.md", "content of doc 2")
        zf.writestr("binary.exe", b"\x00\x01\x02") # should be skipped
        zf.writestr(".hidden/doc.txt", "hidden doc") # should be skipped
    
    text = extract_text(f)
    assert "[file: doc1.txt]" in text
    assert "content of doc 1" in text
    assert "[file: doc2.md]" in text
    assert "content of doc 2" in text
    assert "binary.exe" not in text
    assert "hidden doc" not in text

def test_cached_llm_classification(monkeypatch):
    # The format label is derived by classify_format (heuristic-first); the LLM
    # tie-break only fires for an unknown extension whose structure is unclear,
    # and its guess is cached per extension so a same-extension corpus classifies
    # at most once.
    from localm.rag.extract import classify_format
    import localm.config as cfg
    monkeypatch.setattr(cfg, "load_config",
                        lambda: {"rag_classify_unknown_files": True})
    _EXT_CLASSIFICATION_CACHE.clear()

    calls = []
    def dummy_classify(snippet):
        calls.append(snippet)
        return "javascript"

    fmt1 = classify_format("function add(a, b) { return a + b; }",
                           "file1.custom_code", classify_fn=dummy_classify)
    assert fmt1 == "javascript"
    assert len(calls) == 1

    # Second file, same extension -> served from cache, classifier not re-invoked.
    fmt2 = classify_format("function sub(a, b) { return a - b; }",
                           "file2.custom_code", classify_fn=dummy_classify)
    assert fmt2 == "javascript"
    assert len(calls) == 1
    assert _EXT_CLASSIFICATION_CACHE[".custom_code"] == "javascript"


def test_sniff_format_images_and_tar():
    assert sniff_format(b"\x89PNG\r\n\x1a\n", "file.png") == ".png"
    assert sniff_format(b"\xff\xd8\xff", "file.jpg") == ".jpeg"
    assert sniff_format(b"GIF89a", "file.gif") == ".gif"
    assert sniff_format(b"RIFF\x00\x00\x00\x00WEBP", "file.webp") == ".webp"
    
    # Tar signatures
    ustar_header = b"\x00" * 257 + b"ustar"
    assert sniff_format(ustar_header, "file.tar") == ".tar"
    assert sniff_format(b"\x1f\x8b\x08", "file.tar.gz") == ".tar"
    assert sniff_format(b"BZh9", "file.tar.bz2") == ".tar"
    assert sniff_format(b"\xfd7zXZ\x00\x00", "file.tar.xz") == ".tar"


def test_extract_tar_archive(tmp_path):
    import tarfile
    f = tmp_path / "archive.tar"
    with tarfile.open(f, "w") as tf:
        # doc1
        t1 = tarfile.TarInfo("doc1.txt")
        data1 = b"tar text one"
        t1.size = len(data1)
        tf.addfile(t1, io.BytesIO(data1))
        # doc2
        t2 = tarfile.TarInfo("doc2.md")
        data2 = b"tar text two"
        t2.size = len(data2)
        tf.addfile(t2, io.BytesIO(data2))
        # binary
        t3 = tarfile.TarInfo("binary.dll")
        data3 = b"\x00\x01\x02"
        t3.size = len(data3)
        tf.addfile(t3, io.BytesIO(data3))
        
    text = extract_text(f)
    assert "[file: doc1.txt]" in text
    assert "tar text one" in text
    assert "[file: doc2.md]" in text
    assert "tar text two" in text
    assert "binary.dll" not in text


def test_image_ocr_description(tmp_path):
    f = tmp_path / "photo.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\nimage_payload_here")
    
    # Without describe_image_fn, it should raise ExtractError
    with pytest.raises(ExtractError, match="To index images, load a vision-capable"):
        extract_text(f)
        
    # With describe_image_fn
    calls = []
    def dummy_describe(data, mime):
        calls.append((data, mime))
        return "A beautiful sunset over the sea."
        
    text = extract_text(f, describe_image_fn=dummy_describe)
    assert text == "A beautiful sunset over the sea."
    assert len(calls) == 1
    assert calls[0][1] == "image/png"
    assert b"image_payload_here" in calls[0][0]
