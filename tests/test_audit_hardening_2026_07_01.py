# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for a batch of audit-hardening fixes. Each test pins one
concrete property:

  - a .docx zip bomb is refused, never decompressed into RAM
  - malformed docx XML does not trigger a quadratic paragraph regex
  - malformed .ipynb shapes raise ExtractError, never an unhandled 500
  - chunk_text(chunk_chars<=0) fails fast instead of looping forever
  - the bug-report home scrub covers forward-slash and case variants
  - bug-report client/log fields are token/credential scrubbed
  - config._read_json falls back instead of crashing
  - cors_origins:"*" does not waive the open-mode shell-token gate
  - doctor does not read click.__version__ (DeprecationWarning)
  - GET /v1/comfy/status does not 500 on a stale import
"""

import json
import sys
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from localm.rag import ExtractError, extract_text
from localm.rag import extract as _extract
from localm.rag.chunk import chunk_text


# --------------------------------------------------------------------------- #
#  Document extraction hardening
# --------------------------------------------------------------------------- #

def _write_docx(path, document_xml: str) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", document_xml)


def test_docx_decompression_bomb_rejected(tmp_path, monkeypatch):
    # A small compressed upload whose document.xml decompresses past the cap is
    # refused rather than read into RAM. The cap is shrunk so the test payload
    # stays tiny (the real cap is 80 MB).
    monkeypatch.setattr(_extract, "MAX_ARCHIVE_MEMBER_BYTES", 2000)
    f = tmp_path / "bomb.docx"
    big = ("<w:document><w:body>"
           + "<w:p><w:r><w:t>x</w:t></w:r></w:p>" * 3000
           + "</w:body></w:document>")           # ~100 KB decompressed >> 2000
    _write_docx(f, big)
    with pytest.raises(ExtractError, match="zip bomb|decompressed-size|limit"):
        extract_text(f)


def test_docx_normal_still_extracts(tmp_path):
    # A well-formed docx still extracts.
    f = tmp_path / "ok.docx"
    _write_docx(f,
                '<?xml version="1.0"?><w:document xmlns:w="ns"><w:body>'
                "<w:p><w:r><w:t>First paragraph.</w:t></w:r></w:p>"
                "<w:p><w:r><w:t>Second </w:t></w:r><w:r><w:t>part &amp; more.</w:t></w:r></w:p>"
                "</w:body></w:document>")
    text = extract_text(f)
    assert "First paragraph." in text
    assert "Second part & more." in text
    assert text.index("First") < text.index("Second")


def test_docx_redos_pathological_completes_fast(tmp_path):
    # 40k unmatched <w:p openers; the linear str.split finishes near-instantly.
    f = tmp_path / "evil.docx"
    doc = ("<w:document><w:body>"
           + "<w:p><w:r><w:t>a</w:t></w:r>" * 40000       # NOTE: no </w:p> closers
           + "</w:body></w:document>")
    _write_docx(f, doc)
    start = time.perf_counter()
    text = extract_text(f)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"extraction took {elapsed:.1f}s - regex backtracking?"
    assert "a" in text


# --------------------------------------------------------------------------- #
#  Malformed notebook shapes must not 500
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("payload", [
    {"cells": "not-a-list"},
    {"cells": [123, "str", None]},
    {"cells": [{"cell_type": "code", "source": 42}]},
    {"cells": [{"source": [1, 2, 3]}]},
    ["not", "a", "dict"],
    {"no_cells_key": True},
])
def test_ipynb_malformed_shapes_no_unhandled_crash(tmp_path, payload):
    f = tmp_path / "bad.ipynb"
    f.write_text(json.dumps(payload), encoding="utf-8")
    # The only acceptable failure is a clean ExtractError (-> 422); a raw
    # TypeError/AttributeError would surface as a generic 500.
    try:
        extract_text(f)
    except ExtractError:
        pass
    except Exception as e:  # noqa: BLE001 - the whole point is "no other exception"
        pytest.fail(f"malformed ipynb raised {type(e).__name__}, not ExtractError: {e}")


# --------------------------------------------------------------------------- #
#  A non-positive chunk size must fail fast, not spin forever
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [0, -1, -1200])
def test_chunk_text_nonpositive_raises(bad):
    with pytest.raises(ValueError):
        chunk_text("some text that would otherwise be chunked", chunk_chars=bad)


def test_chunk_text_default_still_works():
    chunks = chunk_text("para one\n\npara two\n\npara three")
    assert chunks and all("text" in c for c in chunks)


# --------------------------------------------------------------------------- #
#  Bug-report scrubbing
# --------------------------------------------------------------------------- #

def test_scrub_home_redacts_user_segment_both_separators():
    from localm.bugreport import _scrub_home
    # The always-on backstop strips the username from a home-rooted path in both
    # separator forms and the posix roots.
    assert "Alice" not in _scrub_home(r"see Z:\Users\Alice\app\config.json")
    assert "Alice" not in _scrub_home("see Z:/Users/Alice/app/config.json")
    assert "bob" not in _scrub_home("path /home/bob/.localm/models")
    assert "carol" not in _scrub_home("path /Users/carol/Library/x")


@pytest.mark.skipif(sys.platform != "win32", reason="win32 case-insensitive paths")
def test_scrub_home_case_insensitive_on_windows():
    from localm.bugreport import _scrub_home
    # A lowercased drive + 'users' is still redacted on Windows, where paths are
    # case-insensitive.
    out = _scrub_home(r"opened z:\users\alice\secret.txt").lower()
    assert "alice" not in out


def test_client_lines_scrub_tokens_and_url_creds():
    from localm.bugreport import _client_lines
    client = {
        "userAgent": "Mozilla/5.0",
        "console": [
            "GET failed with Authorization: Bearer sk-FAKE-1234-not-a-real-key-xx",
            "cannot reach https://joe:hunter2pass@searx.example/search",
        ],
    }
    out = "\n".join(_client_lines(client))
    assert "sk-FAKE-1234-not-a-real-key-xx" not in out
    assert "hunter2pass" not in out
    assert "<redacted>" in out


# --------------------------------------------------------------------------- #
#  A damaged config/registry falls back, never crashes
# --------------------------------------------------------------------------- #

def test_read_json_falls_back_on_non_utf8(tmp_path):
    from localm import config as cfg
    bad = tmp_path / "config.json"
    bad.write_bytes(b"\xff\xfe garbage \x80\x81 not utf-8")   # UnicodeDecodeError
    assert cfg._read_json(bad, {"fallback": True}) == {"fallback": True}


def test_read_json_falls_back_on_huge_integer(tmp_path):
    from localm import config as cfg
    huge = tmp_path / "registry.json"
    huge.write_text("1" * 5000, encoding="utf-8")            # ValueError (int limit)
    assert cfg._read_json(huge, {"ok": 1}) == {"ok": 1}


# --------------------------------------------------------------------------- #
#  Server: helpers for a keyless open-mode app
# --------------------------------------------------------------------------- #

def _keyless_app(tmp_path, monkeypatch, config=None):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    # config.py freezes these at import, so the autouse LOCALM_HOME env alone does
    # not redirect load_config/ensure_dirs; pin them all, MODELS_DIR included.
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    if config:
        from localm.config import save_config
        save_config(config)
    from localm.inference.http_server import create_app
    return create_app(None)


# --------------------------------------------------------------------------- #
#  cors_origins:"*" must NOT waive the shell-token gate
# --------------------------------------------------------------------------- #

def test_cors_wildcard_still_requires_shell_token(tmp_path, monkeypatch):
    app = _keyless_app(tmp_path, monkeypatch, config={"cors_origins": "*"})
    client = TestClient(app)
    # "*" opts out of same-origin only; a keyless state change with no shell token
    # is still refused.
    refused = client.patch("/v1/config", json={"n_ctx": 8192},
                           headers={"Origin": "https://evil.example"})
    assert refused.status_code == 403
    ok = client.patch("/v1/config", json={"n_ctx": 8192},
                      headers={"Authorization": f"Bearer {app.state.shell_token}"})
    assert ok.status_code == 200


# --------------------------------------------------------------------------- #
#  The comfy status route must not ImportError -> 500
# --------------------------------------------------------------------------- #

def test_comfy_status_import_symbols_exist():
    # The route imports these.
    from localm.image_gen.comfy import _comfy_alive, default_api_url
    assert callable(_comfy_alive) and callable(default_api_url)


def test_comfy_status_route_returns_200(tmp_path, monkeypatch):
    app = _keyless_app(tmp_path, monkeypatch)
    client = TestClient(app)
    r = client.get("/v1/comfy/status", headers={"Authorization": f"Bearer {app.state.shell_token}"})
    assert r.status_code == 200, r.text
    assert "alive" in r.json()


# --------------------------------------------------------------------------- #
#  doctor must not read click.__version__ (DeprecationWarning)
# --------------------------------------------------------------------------- #

def test_doctor_no_version_deprecation_warning(recwarn):
    from localm.cli.doctor import _check_packages
    _check_packages()
    offenders = [str(w.message) for w in recwarn.list
                 if "__version__" in str(w.message)]
    assert not offenders, offenders
