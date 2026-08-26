# SPDX-License-Identifier: AGPL-3.0-or-later
"""A registry entry with NO model_type is a third state, and the API has to say so.

Both model endpoints read ``entry.get("model_type", "llm")``. That default is
LOAD-BEARING: the chat-model picker asks ``/api/models?type=llm``, so a legacy
entry from before the field existed stays selectable for chat. It is not a
recorded fact, though, so ``model_type_recorded: false`` rides alongside,
emitted ONLY when nothing is recorded. Every other model's payload is
unchanged.

The filter assertions sit in the same tests as the flag ones, because reporting
the absence and keeping the permissive default are two halves of one behaviour:
dropping the default would satisfy every flag assertion while removing untagged
models from the chat picker.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from localm.model_manager import has_recorded_model_type


TYPED = {"path": "Z:/nonexistent/chat.gguf", "source": "local", "model_type": "llm"}
UNTAGGED = {"path": "Z:/nonexistent/legacy.gguf", "source": "local"}
EMPTY_TYPE = {"path": "Z:/nonexistent/blank.gguf", "source": "local", "model_type": ""}
NULL_TYPE = {"path": "Z:/nonexistent/null.gguf", "source": "local", "model_type": None}
UNKNOWN = {"path": "Z:/nonexistent/mystery.gguf", "source": "local",
           "model_type": "unknown"}


class TestPredicate:
    def test_a_recorded_type_is_recorded(self):
        assert has_recorded_model_type(TYPED) is True
        # 'unknown' is a CHOICE somebody made ("we looked, it is none of these"),
        # not an absence. It must not be lumped in with never-classified.
        assert has_recorded_model_type(UNKNOWN) is True

    def test_absent_empty_or_non_string_is_not_recorded(self):
        assert has_recorded_model_type(UNTAGGED) is False
        assert has_recorded_model_type(EMPTY_TYPE) is False
        assert has_recorded_model_type(NULL_TYPE) is False
        assert has_recorded_model_type({"model_type": "   "}) is False
        assert has_recorded_model_type({"model_type": 123}) is False

    def test_a_non_dict_entry_does_not_raise(self):
        # Registry entries reach this from a file on disk; a corrupt one must
        # not 500 the models list.
        assert has_recorded_model_type("not-a-dict") is False
        assert has_recorded_model_type(None) is False


class TestModelsListRoute:
    def _models(self, gui_app, registry):
        app, _ = gui_app
        with patch("localm.config.load_registry", return_value=registry):
            with TestClient(app) as client:
                r = client.get("/api/models")
        assert r.status_code == 200
        return {m["name"]: m for m in r.json()["models"]}

    def test_untagged_entry_is_flagged_and_typed_entry_is_not(self, gui_app):
        rows = self._models(gui_app, {"typed": TYPED, "untagged": UNTAGGED})
        assert rows["untagged"]["model_type_recorded"] is False
        # ABSENT, not True: the key is omitted for an already-typed model.
        assert "model_type_recorded" not in rows["typed"]

    def test_the_llm_default_still_rides_along_for_the_untagged_row(self, gui_app):
        # The flag says the type is a guess; it does not remove the guess, so
        # `model_type` is still present.
        rows = self._models(gui_app, {"untagged": UNTAGGED})
        assert rows["untagged"]["model_type"] == "llm"

    def test_recorded_unknown_is_not_flagged(self, gui_app):
        # 'unknown' was chosen; "nothing recorded" was not.
        rows = self._models(gui_app, {"mystery": UNKNOWN})
        assert "model_type_recorded" not in rows["mystery"]
        assert rows["mystery"]["model_type"] == "unknown"

    def test_empty_and_null_types_are_flagged_too(self, gui_app):
        rows = self._models(gui_app, {"blank": EMPTY_TYPE, "nulled": NULL_TYPE})
        assert rows["blank"]["model_type_recorded"] is False
        assert rows["nulled"]["model_type_recorded"] is False

    def test_type_llm_filter_STILL_returns_the_untagged_entry(self, gui_app):
        """models-sidebar.js fetches /api/models?type=llm for the chat picker,
        so a legacy untagged model stays in that list."""
        app, _ = gui_app
        with patch("localm.config.load_registry",
                   return_value={"typed": TYPED, "untagged": UNTAGGED, "mystery": UNKNOWN}):
            with TestClient(app) as client:
                r = client.get("/api/models?type=llm")
        assert r.status_code == 200
        assert {m["name"] for m in r.json()["models"]} == {"typed", "untagged"}


class TestModelDetailRoute:
    def _detail(self, monkeypatch, name, registry):
        from localm.inference.http_server import create_app
        monkeypatch.setattr("localm.config.load_registry", lambda: registry)
        app = create_app(None)
        with TestClient(app) as client:
            r = client.get(f"/v1/models/{name}")
        assert r.status_code == 200, r.text
        return r.json()

    def test_detail_flags_the_untagged_entry(self, auth, monkeypatch):
        body = self._detail(monkeypatch, "untagged", {"untagged": UNTAGGED})
        assert body["model_type_recorded"] is False
        assert body["model_type"] == "llm", "the default still rides along"

    def test_detail_omits_the_flag_for_a_recorded_type(self, auth, monkeypatch):
        body = self._detail(monkeypatch, "typed", {"typed": TYPED})
        assert "model_type_recorded" not in body
        assert body["model_type"] == "llm"


@pytest.fixture
def auth(tmp_path, monkeypatch):
    """Throwaway data dir + clean auth environment, matching tests/test_auth.py.

    config.py freezes its paths at import, so LOCALM_HOME alone does not
    redirect load_config/save_config.
    """
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    import localm.auth as a
    return a


@pytest.fixture
def gui_app(tmp_path, monkeypatch):
    """The GUI router on a bare app, mirroring tests/test_gui.py's fixture of the
    same name. NOT create_app(), which mounts the full auth stack and would 403
    every request here."""
    from fastapi import FastAPI
    from localm.plugins.gui.web import attach_gui
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    app = FastAPI()
    attach_gui(
        app,
        self_url="http://127.0.0.1:9/v1",   # never dialled in these tests
        switch_model=lambda name: {"status": "loaded", "model": name},
        active_model=lambda: None,
    )
    return app, tmp_path
