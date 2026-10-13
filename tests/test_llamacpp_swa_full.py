# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every llama.cpp context localm creates keeps the full-size sliding-window
(SWA) cache, so KV prefix reuse can cut the cache at any earlier position."""

from __future__ import annotations

import pytest

from localm.inference.backends.llamacpp import _api as api
from localm.inference.backends.llamacpp import _structs


@pytest.mark.parametrize("layout", [
    _structs.LlamaContextParamsV1,
    _structs.LlamaContextParamsV2,
    _structs.LlamaContextParamsV3,
])
def test_every_context_is_created_with_the_full_swa_cache(monkeypatch, layout):
    seen = []

    def fake_init(model, params):
        seen.append(params.swa_full)
        return 1234

    monkeypatch.setattr(api, "_context_params_class", lambda: layout)
    monkeypatch.setattr(api, "_bind", lambda name, *sig: fake_init)
    params = layout()
    params.swa_full = False
    assert api.llama_init_from_model(object(), params) == 1234
    assert seen == [True]
    assert params.swa_full is True


def test_a_foreign_layout_is_still_refused_before_any_change(monkeypatch):
    monkeypatch.setattr(api, "_context_params_class", lambda: _structs.LlamaContextParamsV3)
    monkeypatch.setattr(api, "_bind", lambda *a: pytest.fail("native call reached"))
    params = _structs.LlamaContextParamsV1()
    params.swa_full = False
    with pytest.raises(TypeError):
        api.llama_init_from_model(object(), params)
    assert params.swa_full is False
