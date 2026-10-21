# SPDX-License-Identifier: AGPL-3.0-or-later
"""GET /v1/models carries the same vision / audio_input fields as GET /v1/models/{id},
in the same true / false / key-absent shape, from the model's own projector header."""

import json
import struct

import pytest
from fastapi.testclient import TestClient

import localm.config as config
import localm.model_manager as model_manager
from localm.inference.http_server import create_app

_STR, _BOOL, _U32 = 8, 7, 4


def _kv(key, vtype, value):
    k = key.encode()
    out = struct.pack("<Q", len(k)) + k + struct.pack("<I", vtype)
    if vtype == _STR:
        v = value.encode()
        return out + struct.pack("<Q", len(v)) + v
    if vtype == _BOOL:
        return out + struct.pack("<?", value)
    return out + struct.pack("<I", value)


def _projector(path, *, vision=None, audio=None):
    kvs = [("general.architecture", _STR, "clip"), ("general.type", _STR, "mmproj")]
    if vision is not None:
        kvs.append(("clip.has_vision_encoder", _BOOL, vision))
    if audio is not None:
        kvs.append(("clip.has_audio_encoder", _BOOL, audio))
    kvs.append(("clip.audio.num_mel_bins", _U32, 128))
    body = b"".join(_kv(*kv) for kv in kvs)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + body)


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    (h / "models").mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(model_manager, "MODELS_DIR", h / "models")
    monkeypatch.setattr(config, "HOME_DIR", h)
    monkeypatch.setattr(config, "MODELS_DIR", h / "models")
    monkeypatch.setattr(config, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(config, "REGISTRY_FILE", h / "registry.json")
    return h


def _model(home, name, **projector):
    d = home / "models" / name
    d.mkdir()
    (d / f"{name}.gguf").write_bytes(b"x")
    if projector:
        _projector(d / f"mmproj-{name}.gguf", **projector)
    return {"path": str(d / f"{name}.gguf"), "source": "local", "model_type": "llm"}


def test_the_list_carries_the_capability_fields_the_detail_route_has(home):
    registry = {
        "asr": _model(home, "asr", audio=True),
        "vlm": _model(home, "vlm", vision=True),
        "text": _model(home, "text"),
        "gone": {"path": "Z:/nonexistent/gone.gguf", "source": "local", "model_type": "llm"},
    }
    (home / "registry.json").write_text(json.dumps(registry), encoding="utf-8")

    with TestClient(create_app(None)) as client:
        listed = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
        detail = {n: client.get(f"/v1/models/{n}").json() for n in registry}

    assert (listed["asr"]["vision"], listed["asr"]["audio_input"]) == (False, True)
    assert (listed["vlm"]["vision"], listed["vlm"]["audio_input"]) == (True, False)
    assert (listed["text"]["vision"], listed["text"]["audio_input"]) == (False, False)
    assert "vision" not in listed["gone"] and "audio_input" not in listed["gone"]
    for name in registry:
        for key in ("vision", "audio_input"):
            assert listed[name].get(key) == detail[name].get(key), (name, key)
