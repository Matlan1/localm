# SPDX-License-Identifier: AGPL-3.0-or-later
"""registry.model_vision_capability: the TRI-STATE behind the Models page's vision pill."""
import json
from pathlib import Path

import pytest

from localm.model_manager import registry as reg_mod
from localm.model_manager.registry import (
    model_vision_capability,
    vision_capable_models,
)


@pytest.fixture
def models(tmp_path):
    """A registry covering every branch, on disk."""
    vis = tmp_path / "vis"
    vis.mkdir()
    (vis / "gemma.gguf").write_bytes(b"x")
    (vis / "mmproj-gemma-f16.gguf").write_bytes(b"x")

    txt = tmp_path / "txt"
    txt.mkdir()
    (txt / "plain.gguf").write_bytes(b"x")

    hf_vis = tmp_path / "hfvis"
    hf_vis.mkdir()
    (hf_vis / "config.json").write_text(json.dumps({"vision_config": {}}),
                                        encoding="utf-8")

    hf_txt = tmp_path / "hftxt"
    hf_txt.mkdir()
    (hf_txt / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"]}), encoding="utf-8")

    return {
        "gguf-vision": {"path": str(vis / "gemma.gguf"), "source": "local",
                        "model_type": "llm"},
        "gguf-text": {"path": str(txt / "plain.gguf"), "source": "local",
                      "model_type": "llm"},
        "hf-vision": {"path": str(hf_vis), "source": "local", "model_type": "llm"},
        "hf-text": {"path": str(hf_txt), "source": "local", "model_type": "llm"},
        # A standalone projector is not a model to route an image TO. We know
        # its type without touching disk, so this is a confirmed False.
        "the-projector": {"path": str(vis / "mmproj-gemma-f16.gguf"),
                          "source": "local", "model_type": "mmproj"},
        # The cases the tri-state exists for.
        "gone-drive": {"path": "Z:/nonexistent/gone.gguf", "source": "local",
                       "model_type": "llm"},
        "malformed": "this entry is a string, not a dict",
        "null-path": {"path": None, "source": "local", "model_type": "llm"},
    }


@pytest.mark.parametrize("name,expected", [
    ("gguf-vision", True),
    ("hf-vision", True),
    ("gguf-text", False),
    ("hf-text", False),
    ("the-projector", False),
    ("gone-drive", None),
    ("malformed", None),
    ("null-path", None),
])
def test_vision_capability_is_a_real_tristate(models, name, expected):
    assert model_vision_capability(name, reg=models) is expected


def test_unknown_is_not_false(models):
    """The single assertion this module exists for, stated on its own so a future edit that collapses the two cannot pass by tweaking a parametrize row."""
    unreachable = model_vision_capability("gone-drive", reg=models)
    text_only = model_vision_capability("gguf-text", reg=models)
    assert text_only is False
    assert unreachable is None
    assert unreachable is not text_only


def test_a_name_absent_from_the_registry_is_unknown(models):
    assert model_vision_capability("never-registered", reg=models) is None


def test_recorded_projector_wins_over_an_unreachable_model_path(tmp_path, models):
    """ORDERING, and it is load-bearing rather than incidental."""
    proj = tmp_path / "vis" / "mmproj-gemma-f16.gguf"
    entry = {"path": "Z:/nonexistent/gone2.gguf", "source": "local",
             "model_type": "llm", "mmproj": str(proj)}
    assert model_vision_capability("recorded", reg={**models, "recorded": entry}) is True


@pytest.fixture
def raising_stat(monkeypatch):
    """Inject the OSError a dead UNC share produces, DETERMINISTICALLY."""
    fired = []
    real_stat = reg_mod.os.stat

    def boom(path, *a, **kw):
        if "unreadable" in str(path):
            fired.append(str(path))
            raise OSError(64, "The specified network name is no longer available")
        return real_stat(path, *a, **kw)

    monkeypatch.setattr(reg_mod.os, "stat", boom)
    # Prove the injector is live BEFORE any assertion leans on it.
    with pytest.raises(OSError):
        reg_mod.os.stat("Z:/unreadable/probe.gguf")
    assert fired, "the injected stat never fired - the rest of this test is theatre"
    fired.clear()
    return fired


def test_an_oserror_during_the_probe_is_unknown_not_false(models, raising_stat):
    """pathlib swallows only the 'not found'-ish subset of OSError, so a path probe can RAISE rather than return False."""
    entry = {"path": "Z:/unreadable/x.gguf", "source": "local", "model_type": "llm"}
    got = model_vision_capability("boom", reg={**models, "boom": entry})
    assert raising_stat, "the probe never reached the injected stat"
    assert got is None, "an OSError mid-probe is unknown, never a confirmed 'not vision'"


def test_an_oserror_does_not_take_down_the_whole_listing(models, raising_stat):
    entry = {"path": "Z:/unreadable/x.gguf", "source": "local", "model_type": "llm"}
    listed = vision_capable_models_for({**models, "boom": entry})
    assert raising_stat, "the probe never reached the injected stat"
    assert listed == ["gguf-vision", "hf-vision"], \
        "one unreadable entry must not raise out of, or empty, the listing"


def test_an_unreachable_entry_skips_the_projector_probes(models, monkeypatch):
    """The cost guard, asserted from OUTSIDE the call with a counting mock rather than a raising side_effect - defensive code catches an AssertionError as an input and the test then passes both ways."""
    from unittest.mock import MagicMock
    spy = MagicMock(return_value=None)
    monkeypatch.setattr(reg_mod, "get_model_mmproj", spy)

    assert model_vision_capability("gone-drive", reg=models) is None
    spy.assert_not_called()

    # ...but an entry that DOES record one still gets asked, or the ordering
    # this file's other test pins would be unreachable.
    spy.reset_mock()
    entry = {"path": "Z:/nonexistent/gone2.gguf", "source": "local",
             "model_type": "llm", "mmproj": "Z:/nonexistent/mmproj.gguf"}
    model_vision_capability("recorded", reg={**models, "recorded": entry})
    spy.assert_called_once()


def vision_capable_models_for(registry):
    """vision_capable_models() with a supplied registry."""
    import localm.model_manager as mm
    original = mm.load_registry
    mm.load_registry = lambda: registry
    try:
        return vision_capable_models()
    finally:
        mm.load_registry = original


def test_vision_capable_models_true_set_is_unchanged(models):
    """The positive-membership contract this refactor must not move."""
    assert vision_capable_models_for(models) == ["gguf-vision", "hf-vision"]


def test_vision_capable_models_reads_the_registry_once(models, monkeypatch):
    """One snapshot per call, threaded down."""
    import localm.model_manager as mm
    calls = []

    def counting():
        calls.append(1)
        return models

    monkeypatch.setattr(mm, "load_registry", counting)
    vision_capable_models()
    assert len(calls) == 1, f"registry.json read {len(calls)} times for one listing"


@pytest.fixture
def one_folder(tmp_path):
    """Two models and one projector sharing ONE folder - the normal shape of a models directory, and the shape the dir_cache exists for."""
    d = tmp_path / "models"
    d.mkdir()
    (d / "gemma.gguf").write_bytes(b"x")
    (d / "mmproj-gemma-f16.gguf").write_bytes(b"x")
    (d / "llama.gguf").write_bytes(b"x")
    return d, {
        "gemma": {"path": str(d / "gemma.gguf"), "source": "local", "model_type": "llm"},
        "llama": {"path": str(d / "llama.gguf"), "source": "local", "model_type": "llm"},
        # Registered as an llm on purpose, so it does NOT short-circuit on the
        # model_type gate and actually reaches the sibling scan. That is the
        # branch where caching the folder listing could let a projector resolve
        # ITSELF as its own projector.
        "the-projector": {"path": str(d / "mmproj-gemma-f16.gguf"), "source": "local",
                          "model_type": "llm"},
    }


def test_dir_cache_changes_the_cost_not_the_answer(one_folder):
    """The cache is an optimisation, so the ONLY thing that may differ is time."""
    _d, reg = one_folder
    uncached = {n: model_vision_capability(n, reg=reg) for n in reg}
    cache: dict = {}
    cached = {n: model_vision_capability(n, reg=reg, dir_cache=cache) for n in reg}
    assert cached == uncached
    # And pin what those answers actually ARE, so "identical" cannot be
    # identical-and-both-wrong: only gemma matches the projector by stem.
    assert uncached["gemma"] is True
    assert uncached["the-projector"] is False, \
        "a projector must never resolve ITSELF as its own projector"


def test_dir_cache_lists_each_folder_once(one_folder, monkeypatch):
    """Asserted from OUTSIDE with a counting spy, not by timing: a wall-clock assertion on a shared box measures the box."""
    _d, reg = one_folder
    globbed = []
    real_glob = Path.glob

    def counting(self, pattern, *a, **kw):
        globbed.append(str(self))
        return real_glob(self, pattern, *a, **kw)

    monkeypatch.setattr(Path, "glob", counting)

    for n in reg:
        model_vision_capability(n, reg=reg, dir_cache={})
    per_call = len(globbed)
    assert per_call == len(reg), \
        "control: a fresh cache per call must glob once per model"

    globbed.clear()
    shared: dict = {}
    for n in reg:
        model_vision_capability(n, reg=reg, dir_cache=shared)
    assert len(globbed) == 1, \
        f"one shared folder must be listed once, not {len(globbed)} times"
    assert len(set(globbed)) == 1


def test_probe_failure_is_logged_not_silent(models, raising_stat, caplog):
    """AGENTS.md rule 5: the tri-state is the surfaced signal, and the CAUSE is still recorded rather than swallowed."""
    import logging
    entry = {"path": "Z:/unreadable/x.gguf", "source": "local", "model_type": "llm"}
    with caplog.at_level(logging.DEBUG, logger=reg_mod.logger.name):
        model_vision_capability("boom", reg={**models, "boom": entry})
    assert raising_stat, "the probe never reached the injected stat"
    assert any("vision capability probe failed" in r.message or
               "vision capability probe failed" in r.getMessage()
               for r in caplog.records), \
        "an OSError during the probe must leave a debug trace of why"
