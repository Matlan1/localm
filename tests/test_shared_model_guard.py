# SPDX-License-Identifier: AGPL-3.0-or-later
"""The fixtures that keep a test from leaving a shared model resident in its worker."""

from __future__ import annotations

import pytest

from localm.inference import embedder, reranker, speech
from tests._shared_model_guard import (
    fail_if_shared_model_left_resident,
    no_embedder_download,  # noqa: F401
)


class _FakeModel:
    active_requests = 0

    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


@pytest.mark.parametrize(
    ("module", "holder", "name"),
    [(embedder, "_EMBEDDER", "embedder"),
     (reranker, "_RERANKER", "reranker"),
     (speech, "_ENGINE", "speech model")],
)
def test_a_resident_model_fails_naming_the_test_and_is_released(monkeypatch, module, holder, name):
    model = _FakeModel()
    monkeypatch.setattr(module, holder, model)

    with pytest.raises(pytest.fail.Exception) as failure:
        fail_if_shared_model_left_resident("tests/test_x.py::test_leaker")

    assert model.close_calls == 1
    assert getattr(module, holder) is None
    assert "tests/test_x.py::test_leaker" in str(failure.value)
    assert name in str(failure.value)


def test_nothing_resident_passes():
    fail_if_shared_model_left_resident("tests/test_x.py::test_clean")


def test_no_embedder_download_keeps_the_first_use_download_from_running(monkeypatch):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(embedder, "_download_pinned",
                        lambda *a, **kw: calls.append("pinned"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda *a, **kw: calls.append("hub"))

    assert embedder.get_embedder() is None
    assert calls == []
    assert embedder._TRIED_DOWNLOAD is False
