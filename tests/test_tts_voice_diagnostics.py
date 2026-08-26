# SPDX-License-Identifier: AGPL-3.0-or-later
"""TTS/voice diagnostics honesty gaps.

- tts/plug.py must not silently return {} for a corrupt or unreadable SHIPPED
  template, nor silently drop the user's overrides on a config-layer failure -
  both leave a log trace (behaviour unchanged: TTS keeps serving).
- voice/plug.py must not pick the HTTP status (501 vs 422) by substring-matching
  "faster-whisper" in the human error MESSAGE; it branches on the structured
  VoiceError.code, so rewording a message cannot flip the status."""

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import localm.plugins.builtin.tts.plug as tts_plug
import localm.plugins.builtin.tts.settings as tts_settings
from localm.voice import VoiceError


class TestTtsTemplateDiagnostics:
    @pytest.mark.parametrize(
        "filename, write_corrupt",
        [("tts.example.json", True), ("absent.json", False)],
        ids=["corrupt", "missing"],
    )
    def test_broken_template_logs_and_falls_back(
        self, filename, write_corrupt, tmp_path, monkeypatch, caplog
    ):
        path = tmp_path / filename
        if write_corrupt:
            path.write_text("{ not json", encoding="utf-8")
        # Patch the module that OWNS the global, not the one that re-exports the
        # function: plug.py does `from ...settings import defaults as _defaults`, so
        # _defaults resolves _TEMPLATE in settings' namespace.
        monkeypatch.setattr(tts_settings, "_TEMPLATE", path)
        with caplog.at_level(logging.WARNING, logger="localm"):
            assert tts_plug._defaults() == {}
        assert any(filename in r.message for r in caplog.records), \
            "the broken shipped template leaves a warning naming the file"

    def test_healthy_template_stays_silent(self, caplog):
        # The real shipped template parses; no warning on the happy path.
        with caplog.at_level(logging.WARNING, logger="localm"):
            cfg = tts_plug._defaults()
        assert cfg, "the real shipped template parses to a non-empty dict"
        assert not any("tts" in r.message for r in caplog.records)

    def test_config_layer_failure_logs_and_keeps_defaults(self, monkeypatch, caplog):
        class _Host:
            def plugin_config(self, name):
                raise RuntimeError("config layer hiccup")
        monkeypatch.setattr(tts_plug, "_host", _Host())
        with caplog.at_level(logging.DEBUG, logger="localm"):
            cfg = tts_plug._resolved()
        assert cfg == tts_plug._defaults(), "falls back to the template defaults"
        assert any("plugin_config('tts') failed" in r.message for r in caplog.records)


class TestVoiceStatusByCode:
    def _client(self, exc: VoiceError, monkeypatch):
        import localm.plugins.builtin.voice.plug as voice_plug
        import localm.voice as voice_mod

        def _boom(data, language=None):
            raise exc
        monkeypatch.setattr(voice_mod, "transcribe_bytes", _boom)
        app = FastAPI()
        app.include_router(voice_plug._router)
        return TestClient(app)

    def test_missing_dependency_is_501_via_code(self, monkeypatch):
        # The message does NOT contain "faster-whisper", so a substring match
        # would mis-pick 422 for this failure class.
        c = self._client(VoiceError("speech engine not installed",
                                    code="needs-faster-whisper"), monkeypatch)
        r = c.post("/api/voice/transcribe", json={"audio_b64": "aGk="})
        assert r.status_code == 501

    def test_decode_error_is_422_even_if_message_mentions_the_package(self, monkeypatch):
        # The class tag wins over the wording: "faster-whisper" in a DECODE
        # message must not flip the status to 501.
        c = self._client(VoiceError("could not decode (hint: faster-whisper "
                                    "supports wav/webm)", code="decode"), monkeypatch)
        r = c.post("/api/voice/transcribe", json={"audio_b64": "aGk="})
        assert r.status_code == 422

    def test_codeless_error_defaults_to_422(self, monkeypatch):
        c = self._client(VoiceError("some other failure"), monkeypatch)
        r = c.post("/api/voice/transcribe", json={"audio_b64": "aGk="})
        assert r.status_code == 422


class TestVoiceErrorCodes:
    def test_pure_python_guards_carry_codes(self):
        from localm.voice import transcribe_bytes
        with pytest.raises(VoiceError) as ei:
            transcribe_bytes(b"")
        assert ei.value.code == "empty"

    def test_default_code_is_empty_string(self):
        assert VoiceError("plain").code == ""
