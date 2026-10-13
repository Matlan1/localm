# SPDX-License-Identifier: AGPL-3.0-or-later
"""The bug report's configuration section scrubs every string in an
allowlisted value, whatever the value's shape, with the same scrub as the rest
of the report."""

from __future__ import annotations

import pytest

from localm.bugreport import diagnostics

SECRET = "CFGCANARY9Z3K"


def _subset(monkeypatch, cfg: dict) -> dict:
    import localm.config as config
    monkeypatch.setattr(config, "load_config", lambda: dict(cfg))
    return diagnostics._safe_config_subset()


def test_dict_and_nested_list_values_are_scrubbed(monkeypatch):
    out = _subset(monkeypatch, {
        "cors_origins": [["https://u:" + SECRET + "@evil.example.org"]],
        "comfy_api_url": {"base": "https://u:pw@comfy.example.org/api?api_key=" + SECRET,
                          "C:\\Users\\zqalice\\key": ("t", "token=" + SECRET)},
    })
    rendered = repr(out)
    assert SECRET not in rendered
    assert "zqalice" not in rendered
    assert out["cors_origins"] == [["https://<redacted>@evil.example.org"]]
    assert out["comfy_api_url"]["base"] == (
        "https://<redacted>@comfy.example.org/api?api_key=<redacted>")


@pytest.mark.parametrize("value,expected", [
    ("python main.py --api-key sk-proj-" + SECRET,
     "python main.py --api-key <redacted>"),
    ("run.bat --header Bearer " + SECRET, "run.bat --header Bearer <redacted>"),
    ('{"api_key": "' + SECRET + '"}', '{"api_key": <redacted>}'),
])
def test_bearer_api_key_and_json_shapes_are_scrubbed(monkeypatch, value, expected):
    assert _subset(monkeypatch, {"comfy_launch_cmd": value}) == {"comfy_launch_cmd": expected}


def test_scalars_keep_their_type(monkeypatch):
    cfg = {"port": 8642, "n_gpu_layers": -1, "mode": "privacy"}
    out = _subset(monkeypatch, cfg)
    for key, val in out.items():
        assert val == cfg[key]
        assert type(val) is type(cfg[key])


def test_an_unexpected_type_is_rendered_and_scrubbed(monkeypatch):
    class _Odd:
        def __str__(self):
            return "https://u:" + SECRET + "@h.example.org"

    out = _subset(monkeypatch, {"comfy_api_url": _Odd()})
    assert out == {"comfy_api_url": "https://<redacted>@h.example.org"}
