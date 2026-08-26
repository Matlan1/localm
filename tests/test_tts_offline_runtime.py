# SPDX-License-Identifier: AGPL-3.0-or-later
"""The onnxruntime-web runtime is VENDORED, so neural TTS needs no CDN.

The Kokoro bundle otherwise fetches its ONNX backend from cdn.jsdelivr.net on
every cold load, so text-to-speech would not work offline or behind a filtering
proxy, and the GUI's Content-Security-Policy would have to grant that origin in
script-src (a dynamic import() is a module script). The runtime ships under
``static/vendor/onnxruntime/`` and the CSP grants nothing.

These tests guard the three things that would silently put the CDN back: the
shipped default losing its value, the two places that state that default
drifting apart, and the vendored file set no longer matching what the bundle
actually asks for. The CSP half is asserted separately.
"""

import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import localm.plugins.builtin.tts.settings as tts_settings

_PLUGIN = Path(tts_settings.__file__).resolve().parent
_STATIC = _PLUGIN / "static"
_VENDOR = _STATIC / "vendor"

# The two artefacts @huggingface/transformers ships in dist/, and the only two
# the bundle names. Read back out of kokoro.min.js rather than trusted.
_RUNTIME_MJS = "ort-wasm-simd-threaded.jsep.mjs"
_RUNTIME_WASM = "ort-wasm-simd-threaded.jsep.wasm"


def _shipped_wasm_paths() -> str:
    return tts_settings.defaults()["wasm_paths"]


def test_the_shipped_default_points_at_the_vendored_runtime():
    """A blank default is how this regresses, so assert the value AND the files.

    Blank is not neutral any more: the bundle's own fallback is the CDN origin,
    which the CSP now refuses, so shipping "" would mean no neural voice at all
    rather than "use the upstream default".
    """
    value = _shipped_wasm_paths()
    assert value, (
        "tts.example.json ships a blank wasm_paths. Blank falls through to the "
        "bundle's cdn.jsdelivr.net default, which the CSP no longer admits."
    )
    assert value.endswith("/"), (
        f"wasm_paths={value!r} has no trailing slash. onnxruntime concatenates "
        "the filename straight onto this prefix."
    )
    root = (_STATIC / value).resolve()
    assert root.is_dir(), f"{value!r} is not a directory under {_STATIC}"
    assert _STATIC.resolve() in root.parents, f"{value!r} escapes the static tree"
    for name in (_RUNTIME_MJS, _RUNTIME_WASM):
        f = root / name
        assert f.is_file(), f"vendored runtime file missing: {f}"
        assert f.stat().st_size > 0, f"vendored runtime file is empty: {f}"


def test_the_vendored_wasm_is_a_real_webassembly_module():
    """Not "the file exists": a truncated or placeholder drop passes that.

    Every WebAssembly binary opens with the 4-byte magic 0x00 'a' 's' 'm'
    followed by a little-endian version word, so this is the cheapest check that
    the bytes are the artefact rather than a stub.
    """
    blob = (_STATIC / _shipped_wasm_paths() / _RUNTIME_WASM).read_bytes()
    assert blob[:4] == b"\x00asm", (
        f"vendored runtime carries no WebAssembly magic, got {blob[:8]!r}")
    assert blob[4:8] == b"\x01\x00\x00\x00", f"unexpected wasm version {blob[4:8]!r}"
    assert len(blob) > 1_000_000, (
        f"vendored runtime is only {len(blob)} bytes; the real onnxruntime jsep "
        "build is roughly 21 MB, so this looks truncated")


def test_the_js_fallback_and_the_shipped_default_agree():
    """Two places state this default. They must not drift.

    tts.js carries its own fallback deliberately (the config fetch swallows a
    failure into an empty cfg, and an old install may still hold a saved blank
    override), so the SAME string is written twice. Drift would be invisible:
    each site is individually correct-looking, and only the combination decides
    what an offline browser actually loads.
    """
    src = (_STATIC / "tts.js").read_text(encoding="utf-8")
    value = _shipped_wasm_paths()
    assert f'cfg.wasm_paths || "{value}"' in src, (
        f"tts.js does not fall back to the shipped default {value!r}. Either the "
        "template or the fallback moved without the other.")


def test_every_runtime_file_the_bundle_asks_for_is_vendored():
    """The bundle decides the filenames; vendoring a different set just 404s.

    Reads the real kokoro.min.js rather than a fixture: the artefact that ships
    is the only thing that can say which runtime files it will request, and a
    rebuilt bundle (a newer transformers.js, a different ort build) is exactly
    the change that would move them without touching any Python.
    """
    bundle = (_VENDOR / "kokoro.min.js").read_text(encoding="utf-8", errors="replace")
    asked = {m for m in re.findall(r"ort-wasm[A-Za-z0-9._-]*", bundle)
             if m.endswith(".mjs") or m.endswith(".wasm")}
    assert asked, "no ort runtime filename found in the bundle at all"
    root = _STATIC / _shipped_wasm_paths()
    missing = sorted(n for n in asked if not (root / n).is_file())
    assert not missing, (
        f"the bundle requests {missing} but they are not vendored under {root}. "
        "Re-vendor from the matching @huggingface/transformers release; see "
        "vendor/NOTICE.md.")


def test_the_shipped_default_is_a_value_the_settings_validator_accepts():
    """A default the write surface would reject is a trap for the next editor.

    /v1/tts/config validates wasm_paths against the plugin's asset root, so a
    shipped default that failed that check would 400 for anyone who opened the
    setting and saved it back unchanged.
    """
    from localm.settings_schema import validate_tts_block
    value = _shipped_wasm_paths()
    assert validate_tts_block({"wasm_paths": value}) == {"wasm_paths": value}


def test_a_traversing_wasm_paths_is_still_rejected():
    """Fires-control for the test above: the validator must be able to say no.

    Without this, the acceptance check alone would pass just as happily against
    a validator that had been loosened to accept anything.
    """
    from localm.settings_schema import validate_tts_block
    with pytest.raises(ValueError):
        validate_tts_block({"wasm_paths": "../../etc/"})


def test_the_runtime_is_served_over_http_with_the_right_content_types(
        tmp_path, monkeypatch):
    """The browser has to actually GET these, and the MIME type is load-bearing.

    An ES module import() refuses anything that is not a JS MIME type, and
    WebAssembly.instantiateStreaming refuses anything that is not
    application/wasm. Both are decided by the process mimetypes table, which on
    Windows is seeded from the registry, so a machine's file associations could
    otherwise break neural TTS with nothing in the code to show for it.
    """
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")

    from localm.plugins.engine import attach_engine
    app = FastAPI()
    attach_engine(app)
    base = "/plugins/tts/" + _shipped_wasm_paths()
    with TestClient(app) as c:
        assert c.post("/api/plugins/tts/install").status_code == 200
        mjs = c.get(base + _RUNTIME_MJS)
        assert mjs.status_code == 200, mjs.status_code
        assert mjs.headers["content-type"].split(";")[0] == "text/javascript", \
            mjs.headers["content-type"]
        wasm = c.get(base + _RUNTIME_WASM)
        assert wasm.status_code == 200, wasm.status_code
        assert wasm.headers["content-type"].split(";")[0] == "application/wasm", \
            wasm.headers["content-type"]
        assert wasm.content[:4] == b"\x00asm"
