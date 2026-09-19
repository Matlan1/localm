# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for HF worker load kwargs, dtype deprecation avoidance, and docstring leak suppression."""

import io
import sys
from unittest.mock import MagicMock
import pytest

torch = pytest.importorskip("torch")

from localm.inference.backends._hf_worker import (
    _build_load_kwargs,
    _silence_upstream_docstring_leak,
    _filter_docstring_leak,
)


def test_build_load_kwargs_modern_transformers():
    """Verify modern transformers (>=4.49.0) uses dtype and offload_buffers=True."""
    mock_tr = MagicMock()
    mock_tr.__version__ = "5.15.1"

    device_map_kwargs = {"device_map": "auto", "max_memory": {0: 10000, "cpu": 50000}}
    kwargs = _build_load_kwargs(
        mock_tr,
        device_map_kwargs=device_map_kwargs,
        dtype=torch.bfloat16,
        trust_remote_code=False,
    )

    assert kwargs["offload_buffers"] is True
    assert kwargs["trust_remote_code"] is False
    assert kwargs["device_map"] == "auto"
    assert kwargs["max_memory"] == {0: 10000, "cpu": 50000}
    # Modern transformers must use dtype, not deprecated torch_dtype
    assert kwargs["dtype"] == torch.bfloat16
    assert "torch_dtype" not in kwargs


def test_build_load_kwargs_legacy_transformers():
    """Verify legacy transformers (<4.49.0) falls back to torch_dtype for compatibility."""
    mock_tr = MagicMock()
    mock_tr.__version__ = "4.40.0"

    device_map_kwargs = {"device_map": "cpu"}
    kwargs = _build_load_kwargs(
        mock_tr,
        device_map_kwargs=device_map_kwargs,
        dtype=torch.float32,
        trust_remote_code=True,
    )

    assert kwargs["offload_buffers"] is True
    assert kwargs["trust_remote_code"] is True
    assert kwargs["device_map"] == "cpu"
    # Legacy transformers must use torch_dtype
    assert kwargs["torch_dtype"] == torch.float32
    assert "dtype" not in kwargs


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
