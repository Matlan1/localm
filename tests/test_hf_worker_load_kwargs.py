# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for HF worker load kwargs, dtype deprecation avoidance, and docstring leak suppression."""

import io
import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from localm.inference.backends._hf_worker import (
    _build_load_kwargs,
    _filter_docstring_leak,
    _silence_upstream_docstring_leak,
    _suppress_offload_buffer_advisory,
)

BF16 = object()
F32 = object()

_ADVISORY_LOGGER = "transformers.integrations.accelerate"
_LEAK_LINE = ("[ERROR] `dummy_param` is part of DummyKwargs, but not "
              "documented. Make sure to add it to the docstring.")


def _tr(version):
    return SimpleNamespace(__version__=version)


def test_build_load_kwargs_modern_transformers():
    """Modern transformers uses dtype and offload_buffers=True."""
    device_map_kwargs = {"device_map": "auto", "max_memory": {0: 10000, "cpu": 50000}}
    kwargs = _build_load_kwargs(
        _tr("5.15.1"),
        device_map_kwargs=device_map_kwargs,
        dtype=BF16,
        trust_remote_code=False,
    )

    assert kwargs["offload_buffers"] is True
    assert kwargs["trust_remote_code"] is False
    assert kwargs["device_map"] == "auto"
    assert kwargs["max_memory"] == {0: 10000, "cpu": 50000}
    assert kwargs["dtype"] is BF16
    assert "torch_dtype" not in kwargs


def test_build_load_kwargs_legacy_transformers():
    """Old transformers falls back to torch_dtype."""
    kwargs = _build_load_kwargs(
        _tr("4.40.0"),
        device_map_kwargs={"device_map": "cpu"},
        dtype=F32,
        trust_remote_code=True,
    )

    assert kwargs["offload_buffers"] is True
    assert kwargs["trust_remote_code"] is True
    assert kwargs["device_map"] == "cpu"
    assert kwargs["torch_dtype"] is F32
    assert "dtype" not in kwargs


@pytest.mark.parametrize("version", ["4.49.0", "4.55.4"])
def test_build_load_kwargs_below_dtype_support_uses_torch_dtype(version):
    kwargs = _build_load_kwargs(
        _tr(version), device_map_kwargs={"device_map": "cpu"},
        dtype=F32, trust_remote_code=False)
    assert kwargs["torch_dtype"] is F32
    assert "dtype" not in kwargs


@pytest.mark.parametrize("version", ["4.56.0", "4.57.1", "5.15.1"])
def test_build_load_kwargs_dtype_supported_uses_dtype(version):
    kwargs = _build_load_kwargs(
        _tr(version), device_map_kwargs={"device_map": "cpu"},
        dtype=BF16, trust_remote_code=False)
    assert kwargs["dtype"] is BF16
    assert "torch_dtype" not in kwargs


def test_filter_docstring_leak_drops_upstream_errors():
    """Verify _filter_docstring_leak context manager drops auto_docstring lint error lines."""
    captured = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = captured
        with _filter_docstring_leak():
            # Upstream transformers auto_docstring lint lines should be suppressed
            print("[ERROR] `min_frames` is part of Qwen3VLVideoProcessorInitKwargs, but not documented. Make sure to add it to the docstring of the function in foo.py.")
            print("[ERROR] `max_frames` is part of Qwen3VLVideoProcessorInitKwargs, but not documented. Make sure to add it to the docstring of the function in foo.py.")
            # Normal output should pass through
            print("Normal diagnostic message")
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()
    assert "Normal diagnostic message" in output
    assert "min_frames" not in output
    assert "max_frames" not in output
    assert "not documented" not in output


def test_silence_upstream_docstring_leak_intercepts_print():
    """Verify _silence_upstream_docstring_leak silences prints from auto_docstring."""
    mock_tr = MagicMock()

    def fake_auto_docstring(cls):
        print("[ERROR] `dummy_param` is part of DummyKwargs, but not documented.")
        return cls

    mock_tr.utils.auto_docstring = fake_auto_docstring

    _silence_upstream_docstring_leak(mock_tr)

    captured = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = captured
        # Call the wrapped auto_docstring
        mock_tr.utils.auto_docstring(object)
    finally:
        sys.stdout = old_stdout

    # The stdout print should have been intercepted
    assert "not documented" not in captured.getvalue()


def test_silence_upstream_docstring_leak_intercepts_parameterized_decorator():
    """The decorator returned by auto_docstring(custom_intro=...) is silenced too."""
    def fake_auto_docstring(obj=None, *, custom_intro=None):
        def decorator(cls):
            print(_LEAK_LINE)
            return cls
        if obj:
            return decorator(obj)
        return decorator

    tr = SimpleNamespace(utils=SimpleNamespace(auto_docstring=fake_auto_docstring))
    _silence_upstream_docstring_leak(tr)

    captured = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = captured
        decorated = tr.utils.auto_docstring(custom_intro="intro")(object)
    finally:
        sys.stdout = old_stdout

    assert decorated is object
    assert captured.getvalue() == ""


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _spill_buffers_through_infer_auto_device_map():
    """Run transformers' real infer_auto_device_map over a model whose buffers
    overflow the GPU budget, and return the messages logged by its logger."""
    torch = pytest.importorskip("torch", reason="needs torch to build the tiny model")
    pytest.importorskip("accelerate", reason="infer_auto_device_map needs accelerate")
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.integrations.accelerate import infer_auto_device_map

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("w", torch.zeros(1024, 1024, dtype=torch.uint8))

    class _Config(PretrainedConfig):
        model_type = "localm_tiny_buffer_model"

    class _Model(PreTrainedModel):
        config_class = _Config
        _no_split_modules = ["_Layer"]

        def __init__(self, config):
            super().__init__(config)
            self.layers = torch.nn.ModuleList(_Layer() for _ in range(4))
            self.post_init()

    handler = _Capture()
    lg = logging.getLogger(_ADVISORY_LOGGER)
    lg.addHandler(handler)
    try:
        infer_auto_device_map(
            _Model(_Config()),
            max_memory={0: int(2.5 * 1024 * 1024), "cpu": 10 ** 9},
            no_split_module_classes=["_Layer"])
    finally:
        lg.removeHandler(handler)
    return handler.messages


def test_infer_auto_device_map_advisory_fires_without_suppression():
    """Control: the real transformers path does log the advisory."""
    messages = _spill_buffers_through_infer_auto_device_map()
    assert any("offload_buffers=True" in m for m in messages)


def test_infer_auto_device_map_advisory_is_suppressed_during_load():
    with _suppress_offload_buffer_advisory():
        messages = _spill_buffers_through_infer_auto_device_map()
    assert not any("offload_buffers=True" in m for m in messages)


def test_suppress_offload_buffer_advisory_keeps_other_records():
    handler = _Capture()
    lg = logging.getLogger(_ADVISORY_LOGGER)
    lg.addHandler(handler)
    try:
        with _suppress_offload_buffer_advisory():
            lg.warning("something unrelated went wrong")
    finally:
        lg.removeHandler(handler)
    assert handler.messages == ["something unrelated went wrong"]


def test_suppress_offload_buffer_advisory_removes_its_filter():
    lg = logging.getLogger(_ADVISORY_LOGGER)
    before = list(lg.filters)
    with _suppress_offload_buffer_advisory():
        assert len(lg.filters) == len(before) + 1
    assert lg.filters == before


class _LoadRecorder:
    """Fake transformers namespace that records what HFWorker.load() passes."""

    __version__ = "5.15.1"

    def __init__(self):
        self.model_kwargs = None
        self.advisory = []
        recorder = self

        class _Processor:
            tokenizer = object()
            image_processor = object()

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                print(_LEAK_LINE)
                return cls()

        class _Model:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                print(_LEAK_LINE)
                recorder.model_kwargs = kwargs
                recorder.advisory = _spill_buffers_through_infer_auto_device_map()
                model = MagicMock()
                model.config = SimpleNamespace(max_position_embeddings=128)
                return model

        self.AutoProcessor = _Processor
        self.AutoModelForImageTextToText = _Model


def test_worker_load_passes_modern_kwargs_and_stays_quiet(
        tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch", reason="HFWorker.load() resolves a torch dtype")
    from localm.inference.backends import _hf_worker

    fake = _LoadRecorder()
    monkeypatch.setattr(_hf_worker, "_require_transformers", lambda: fake)
    monkeypatch.setattr(_hf_worker, "_trust_remote_code_enabled", lambda: False)

    worker = _hf_worker.HFWorker(str(tmp_path), device="cpu")
    worker.load()

    kwargs = fake.model_kwargs
    assert kwargs["offload_buffers"] is True
    assert kwargs["trust_remote_code"] is False
    assert kwargs["device_map"] == "cpu"
    assert "dtype" in kwargs
    assert "torch_dtype" not in kwargs
    assert "not documented" not in capsys.readouterr().out
    assert not any("offload_buffers=True" in m for m in fake.advisory)
