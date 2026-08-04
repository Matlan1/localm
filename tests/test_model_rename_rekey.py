# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.inference.http_server.rekey_loaded_model: re-keys every in-memory
record of a loaded model's identity after its registry entry is renamed, so a
still-loaded/serving engine is not orphaned under its old name. See
tests/test_driving_engine.py for the sibling http_server module-state test
pattern this reuses.
"""

from __future__ import annotations

from localm.inference import http_server as hs


class _FakeEngine:
    def __init__(self, name):
        self.display_name = name
        self.loaded = True


def _reset():
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._last_activity_per_model.clear()
    hs._active_model_name = None
    hs._engine = None


def test_rekey_moves_engine_and_updates_display_name():
    _reset()
    eng = _FakeEngine("old")
    hs._engines["old"] = eng
    hs._engines_lru.append("old")
    hs._active_model_name = "old"
    hs._inference_sems["old"] = object()
    hs._last_activity_per_model["old"] = 123.0

    assert hs.rekey_loaded_model("old", "new") is True

    assert "old" not in hs._engines
    assert hs._engines["new"] is eng
    assert eng.display_name == "new", "active_model() reads engine.display_name directly"
    assert hs._engines_lru == ["new"]
    assert hs._active_model_name == "new"
    assert "old" not in hs._inference_sems
    assert "new" in hs._inference_sems
    assert hs._last_activity_per_model.get("new") == 123.0
    assert "old" not in hs._last_activity_per_model
    _reset()


def test_rekey_is_a_noop_when_the_old_name_is_not_loaded():
    _reset()
    assert hs.rekey_loaded_model("nowhere", "elsewhere") is False
    assert hs._engines == {}
    assert hs._active_model_name is None
    _reset()


def test_rekey_does_not_touch_active_model_name_for_a_background_model():
    # Renaming a BACKGROUND-loaded (not active) model must not steal the
    # active-model pointer or disturb the actually-active engine's entry.
    _reset()
    active_eng = _FakeEngine("active-one")
    bg_eng = _FakeEngine("bg-one")
    hs._engines["active-one"] = active_eng
    hs._engines["bg-one"] = bg_eng
    hs._engines_lru.extend(["active-one", "bg-one"])
    hs._active_model_name = "active-one"

    assert hs.rekey_loaded_model("bg-one", "bg-renamed") is True

    assert hs._active_model_name == "active-one"
    assert hs._engines["bg-renamed"] is bg_eng
    assert bg_eng.display_name == "bg-renamed"
    assert hs._engines["active-one"] is active_eng
    _reset()


def test_rekey_fixes_the_stale_active_model_guard_hazard():
    """The concrete hazard this function exists to close: active_model()
    reads _engine.display_name, and the GUI's remove-model guard is exactly
    `req.model == active_model()`. Without the rekey, renaming the active
    model would leave that comparison checking the NEW registry name against
    the engine's stale OLD display_name, so it would never match - and the
    GUI could delete the file out from under the model still serving
    requests. Reproduces active_model()'s own read directly (it is a closure
    built per-app, but it always reduces to exactly this)."""
    _reset()
    eng = _FakeEngine("old")
    hs._engine = eng
    hs._engines["old"] = eng
    hs._engines_lru.append("old")
    hs._active_model_name = "old"

    assert hs.rekey_loaded_model("old", "new") is True

    assert hs._engine.display_name == "new"
    _reset()
