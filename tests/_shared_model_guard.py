# SPDX-License-Identifier: AGPL-3.0-or-later
"""Autouse fixtures for test modules whose tests reach the shared embedder.

Importing a fixture into a test module's namespace arms it for every test there::

    from tests._shared_model_guard import (  # noqa: F401
        no_embedder_download,
        no_shared_model_left_resident,
    )
"""

from __future__ import annotations

import sys

import pytest

_SHARED_MODELS = (
    ("embedder", "localm.inference.embedder", "_EMBEDDER", "reset_embedder"),
    ("reranker", "localm.inference.reranker", "_RERANKER", "reset_reranker"),
    ("speech model", "localm.inference.speech", "_ENGINE", "reset_speech"),
)


def release_resident_shared_models() -> list[str]:
    """Release every shared model loaded in this process and return their names.
    A module that was never imported holds nothing and is not imported here."""
    released = []
    for name, module_name, holder, reset in _SHARED_MODELS:
        module = sys.modules.get(module_name)
        if module is not None and getattr(module, holder) is not None:
            getattr(module, reset)(force=True)
            released.append(name)
    return released


def fail_if_shared_model_left_resident(nodeid: str) -> None:
    """Fail naming *nodeid* when a shared model is loaded, after releasing it so
    the next test in this worker starts with nothing loaded."""
    left = release_resident_shared_models()
    if left:
        pytest.fail(f"{nodeid} left the shared {', '.join(left)} resident in this worker",
                    pytrace=False)


@pytest.fixture(autouse=True)
def no_shared_model_left_resident(request):
    """Fail the test, by name, when it finishes with a shared model still loaded.
    Runs before the test's ``monkeypatch`` is undone, so a test in an armed module
    must not patch a model into the module's singleton."""
    yield
    fail_if_shared_model_left_resident(request.node.nodeid)


@pytest.fixture(autouse=True)
def no_embedder_download(monkeypatch):
    """Run the test under ``LOCALM_NET_MODE=off``: the embedder's first-use download
    does not run, so with no model on disk ``get_embedder()`` returns None and
    memory recall stays lexical. A test may set its own value."""
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
