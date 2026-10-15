# SPDX-License-Identifier: AGPL-3.0-or-later
"""A reload of an unloaded Engine builds its backend again, so engine settings
read from the config take their current values while explicit arguments stay."""

from __future__ import annotations

import threading

import localm.inference.engine as engine_mod


class _Backend:
    def __init__(self, n_ctx, kwargs):
        self.n_ctx = n_ctx
        self.kwargs = kwargs
        self.loaded = False
        self.cancel = None
        self.adapters = None

    def set_load_cancel(self, event):
        self.cancel = event

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False


def _factory(monkeypatch, cfg):
    built = []

    def create_backend(model_path, **kwargs):
        backend = _Backend(kwargs.get("n_ctx") or cfg["n_ctx"], kwargs)
        built.append(backend)
        return backend

    monkeypatch.setattr(engine_mod, "create_backend", create_backend)
    monkeypatch.setattr(engine_mod.Engine, "_attach_registered_adapters", lambda self: None)
    return built


def test_a_reload_after_unload_reads_the_current_config(monkeypatch):
    cfg = {"n_ctx": 4096}
    built = _factory(monkeypatch, cfg)
    eng = engine_mod.Engine("m.gguf", display_name="m")
    eng.load()
    assert eng._backend.n_ctx == 4096 and len(built) == 1

    eng.unload()
    cfg["n_ctx"] = 2048
    cancel = threading.Event()
    eng.set_load_cancel(cancel)
    eng.load()

    assert len(built) == 2
    assert eng._backend is built[1] and eng._backend.loaded
    assert eng._backend.n_ctx == 2048
    assert eng._backend.cancel is cancel


def test_an_explicit_argument_survives_the_reload(monkeypatch):
    cfg = {"n_ctx": 4096}
    _factory(monkeypatch, cfg)
    eng = engine_mod.Engine("m.gguf", display_name="m", n_ctx=8192, spec_source="ngram")
    eng.load()
    eng.unload()
    cfg["n_ctx"] = 2048
    eng.load()
    assert eng._backend.n_ctx == 8192
    assert eng._backend.kwargs["spec_source"] == "ngram"


def test_a_load_of_a_loaded_engine_builds_nothing(monkeypatch):
    built = _factory(monkeypatch, {"n_ctx": 4096})
    eng = engine_mod.Engine("m.gguf", display_name="m")
    eng.load()
    eng.load()
    assert len(built) == 1
