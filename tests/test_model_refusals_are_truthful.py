# SPDX-License-Identifier: AGPL-3.0-or-later
"""A model localm will not load is reported as a refusal, with the real reason:
an expected refusal is not "an unexpected error", a damaged GGUF is named as a
damaged file (not a broken runtime), and a vision projector is not blamed on
the runtime's age."""

from __future__ import annotations

import struct

import click
import pytest
from click.testing import CliRunner

from localm.cli import _GracefulGroup
from localm.inference.backends.base import (
    AdapterLoadError, ModelFileDamagedError, ModelFileUnreadableError, ModelRefusedError,
    PretokenizerUnusableModelError, UnsupportedModelRoleError)
from localm.inference.backends.gguf import (GgufBackend, _damaged_file_reason,
                                            _unopenable_file_reason)

_T_STRING = 8


def _gguf(path, arch: str):
    raw = arch.encode()
    key = b"general.architecture"
    body = (struct.pack("<Q", len(key)) + key + struct.pack("<I", _T_STRING)
            + struct.pack("<Q", len(raw)) + raw)
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                     + struct.pack("<Q", 1) + body + b"\0" * 4096)
    return path


_ALIGNMENT_FAILURE = (
    "Failed to load model: {path}\n"
    "gguf_init_from_reader: alignment 3 is not a power of 2\n"
    "llama_model_load: error loading model: failed to open GGUF file")


@pytest.mark.parametrize("exc_type", [
    UnsupportedModelRoleError, AdapterLoadError, PretokenizerUnusableModelError,
    ModelFileDamagedError])
def test_every_typed_load_refusal_is_a_model_refusal(exc_type):
    assert issubclass(exc_type, ModelRefusedError)
    assert issubclass(exc_type, RuntimeError)


def _group(exc):
    @click.group(cls=_GracefulGroup)
    def g():
        pass

    @g.command()
    def load():
        raise exc

    return g


def test_a_model_refusal_prints_as_a_refusal_with_no_bug_report(tmp_path, monkeypatch):
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)
    monkeypatch.setenv("LOCALM_KEEP_DIAGNOSTICS", "1")
    res = CliRunner().invoke(
        _group(UnsupportedModelRoleError("This is an importance matrix, not a model.")),
        ["load"])
    assert res.exit_code == 1
    assert "This is an importance matrix, not a model." in res.output
    assert "Sorry -" not in res.output
    assert "unexpected error" not in res.output
    assert not (tmp_path / "bug-reports").exists()


def test_a_plain_runtime_error_is_still_an_unexpected_error(tmp_path, monkeypatch):
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)
    monkeypatch.setenv("LOCALM_KEEP_DIAGNOSTICS", "1")
    res = CliRunner().invoke(_group(RuntimeError("kaboom")), ["load"])
    assert res.exit_code == 1
    assert "Sorry -" in res.output


def _backend_failing_with(tmp_path, monkeypatch, text):
    model = _gguf(tmp_path / "bad-model.gguf", "llama")
    backend = GgufBackend(str(model))
    monkeypatch.setattr(backend, "_effective_gpu_layers", lambda: 0)
    monkeypatch.setattr(backend, "_check_vram", lambda: None)

    def _fail():
        raise RuntimeError(text)
    monkeypatch.setattr(backend, "_load_native", _fail)
    return backend


def test_a_gguf_llama_cpp_cannot_read_is_named_as_damaged(tmp_path, monkeypatch):
    backend = _backend_failing_with(
        tmp_path, monkeypatch, _ALIGNMENT_FAILURE.format(path="bad-model.gguf"))
    with pytest.raises(ModelFileDamagedError) as caught:
        backend.load()
    text = str(caught.value)
    assert "bad-model.gguf" in text
    assert "alignment 3 is not a power of 2" in text
    assert "setup-llama" not in text and "runtime" not in text.lower()


_OPEN_FAILURE = (
    "Failed to load model: {path}\n"
    "gguf_init_from_file: failed to open GGUF file '{path}' (Permission denied)\n"
    "llama_model_load: error loading model: llama_model_loader: failed to load model "
    "from {path}")


def test_a_gguf_llama_cpp_cannot_open_is_not_called_damaged(tmp_path, monkeypatch):
    path = str(tmp_path / "bad-model.gguf")
    backend = _backend_failing_with(tmp_path, monkeypatch, _OPEN_FAILURE.format(path=path))
    with pytest.raises(ModelFileUnreadableError) as caught:
        backend.load()
    text = str(caught.value)
    assert not isinstance(caught.value, ModelFileDamagedError)
    assert text.startswith("bad-model.gguf could not be opened")
    assert '"Permission denied"' in text
    assert "damaged" not in text and "Download" not in text
    assert "setup-llama" not in text


def test_an_unreadable_gguf_prints_as_a_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path)
    monkeypatch.setenv("LOCALM_KEEP_DIAGNOSTICS", "1")
    res = CliRunner().invoke(
        _group(ModelFileUnreadableError("m.gguf could not be opened")), ["load"])
    assert res.exit_code == 1
    assert "m.gguf could not be opened" in res.output and "Sorry -" not in res.output


@pytest.mark.parametrize("text,expected", [
    (_OPEN_FAILURE.format(path="/models/m.gguf"), "Permission denied"),
    ("gguf_init_from_file: failed to open GGUF file '/it's (here)/m.gguf' "
     "(Permission denied)\nnext", "Permission denied"),
    ("gguf_init_from_file: failed to open GGUF file '/m.gguf' (No such file or directory)",
     "No such file or directory"),
    ("gguf_init_from_reader: alignment 3 is not a power of 2", None),
    ("could not load the native library", None),
])
def test_unopenable_file_reason_reads_the_system_reason(text, expected):
    assert _unopenable_file_reason(RuntimeError(text)) == expected


def test_a_runtime_failure_without_a_file_reason_keeps_the_runtime_advice(
        tmp_path, monkeypatch):
    backend = _backend_failing_with(
        tmp_path, monkeypatch, "could not load the native library")
    with pytest.raises(RuntimeError) as caught:
        backend.load()
    assert not isinstance(caught.value, ModelRefusedError)
    assert "setup-llama" in str(caught.value)


@pytest.mark.parametrize("text,expected", [
    ("gguf_init_from_reader: alignment 3 is not a power of 2\nnext line",
     "gguf_init_from_reader: alignment 3 is not a power of 2"),
    ("llama_model_load: error loading model: tensor 'x' data is not within the "
     "file bounds, model is corrupted or incomplete",
     "llama_model_load: error loading model: tensor 'x' data is not within the "
     "file bounds, model is corrupted or incomplete"),
    ("Failed to load model: m.gguf\ngguf_init_from_reader: tensor 'blk.0.ffn_down.weight' "
     "has invalid ggml type 41 (NONE)", None),
    ("Failed to create llama context", None),
    ("unknown model architecture: 'clip'", None),
    ("could not load the native library", None),
    (_OPEN_FAILURE.format(path="m.gguf"), None),
])
def test_damaged_file_reason_matches_only_file_reader_lines(text, expected):
    assert _damaged_file_reason(RuntimeError(text)) == expected


def test_a_vision_projector_is_refused_as_a_projector_not_blamed_on_the_runtime(
        tmp_path):
    projector = _gguf(tmp_path / "mmproj-model-f16.gguf", "clip")
    backend = GgufBackend(str(projector))
    with pytest.raises(UnsupportedModelRoleError) as caught:
        backend.load()
    text = str(caught.value)
    assert "mmproj-model-f16.gguf" in text and "vision projector" in text
    assert "newer than the runtime" not in text and "setup-llama" not in text


def test_list_names_a_safetensors_file_by_its_format(tmp_path, monkeypatch):
    import io

    from rich.console import Console

    import localm.model_manager as mm
    from localm.model_manager import registry

    weights = tmp_path / "vae.safetensors"
    weights.write_bytes(b"\0" * 64)
    model = _gguf(tmp_path / "chat.gguf", "llama")
    entries = {
        "my-vae": {"path": str(weights), "source": "local", "model_type": "vae"},
        "my-chat": {"path": str(model), "source": "local", "model_type": "llm"},
    }
    monkeypatch.setattr(mm, "load_registry", lambda: entries)
    buf = io.StringIO()
    monkeypatch.setattr(registry, "console",
                        Console(file=buf, force_terminal=False, width=300))

    registry.list_models()

    rows = {name: line for line in buf.getvalue().splitlines()
            for name in entries if name in line}
    assert "safetensors" in rows["my-vae"] and "gguf" not in rows["my-vae"]
    assert "gguf" in rows["my-chat"]
