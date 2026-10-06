# SPDX-License-Identifier: AGPL-3.0-or-later
"""The image, music and video request routes refuse a non-finite number in any
optional float tuning field with a 422 before a job is started; the same
fields still accept a finite value."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.inference.app_assembly.errors import register_exception_handlers

_REQUIRED = {
    "image": ("/api/imagine", '"prompt": "x"'),
    "music": ("/api/music", '"tags": "x"'),
    "video": ("/api/video", '"prompt": "x"'),
}

_FIELDS = [
    ("image", "guidance"),
    ("image", "cfg"),
    ("image", "denoise"),
    ("image", "lora_strength_model"),
    ("image", "lora_strength_clip"),
    ("music", "cfg"),
    ("music", "lyrics_strength"),
    ("music", "shift"),
    ("video", "cfg"),
]


class _Jobs:
    def __init__(self):
        self.started = []

    def start_fn(self, kind, fn, *, result_path=None, owner=None, label=None):
        self.started.append(kind)
        return MagicMock(id=f"job-{kind}")


def _client(tmp_path, plugin):
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    register_exception_handlers(app)
    manager = PluginManager(app, external_root=tmp_path / "plugins")
    manager.install(plugin)
    app.state.jobs = _Jobs()
    app.state.self_url = "http://127.0.0.1:9/v1"
    return app, TestClient(app, raise_server_exceptions=False)


def _post(client, plugin, field, token):
    route, required = _REQUIRED[plugin]
    body = "{" + required + f', "{field}": {token}' + "}"
    return client.post(route, content=body.encode(),
                       headers={"content-type": "application/json"})


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("plugin,field", _FIELDS,
                         ids=[f"{p}-{f}" for p, f in _FIELDS])
def test_non_finite_number_is_refused_before_a_job_starts(
        tmp_path, plugin, field, token):
    app, client = _client(tmp_path, plugin)
    response = _post(client, plugin, field, token)
    assert app.state.jobs.started == [], \
        f"{plugin}.{field}={token} started a job"
    assert response.status_code == 422, response.text[:300]


@pytest.mark.parametrize("plugin", sorted(_REQUIRED))
def test_a_finite_number_still_starts_a_job(tmp_path, plugin):
    field = next(f for p, f in _FIELDS if p == plugin)
    app, client = _client(tmp_path, plugin)
    response = _post(client, plugin, field, "1.5")
    assert response.status_code == 200, response.text[:300]
    assert app.state.jobs.started == [_REQUIRED[plugin][0].rsplit("/", 1)[-1]]
