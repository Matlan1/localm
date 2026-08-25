# SPDX-License-Identifier: AGPL-3.0-or-later
"""A missing Pillow must not be reported as a native inference fault."""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

from localm.inference.backends.base import (
    ImageDecodeUnavailable,
    UnsupportedInputError,
)
from localm.inference.media import decode_image_url

# A 1x1 PNG. Never decoded in these tests - the import fails first - but a
# well-formed URI keeps the test honest about WHICH failure it observed.
_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_missing_pillow_raises_a_named_error_not_a_bare_importerror(monkeypatch):
    """`sys.modules['PIL'] = None` makes `from PIL import Image` raise ModuleNotFoundError, which is the exact shape a genuinely absent Pillow produces."""
    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.delitem(sys.modules, "PIL.Image", raising=False)

    with pytest.raises(ImageDecodeUnavailable) as exc:
        decode_image_url(_PNG_DATA_URI)

    msg = str(exc.value)
    assert "Pillow" in msg, msg
    # It must name the fix, not merely the symptom.
    assert "pip install pillow" in msg.lower(), msg
    # And it must NOT imply a native fault, which is the wrong-diagnosis half.
    assert "native" not in msg.lower(), msg


def test_the_error_is_recoverable_by_the_gguf_worker(monkeypatch):
    """The worker survives ONLY exceptions it catches by type. `_runner.py` catches `UnsupportedInputError`; anything else escapes and kills the process, evicting the model."""
    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.delitem(sys.modules, "PIL.Image", raising=False)

    with pytest.raises(UnsupportedInputError):
        decode_image_url(_PNG_DATA_URI)

    assert issubclass(ImageDecodeUnavailable, UnsupportedInputError)


def test_decoding_an_image_does_not_require_numpy(monkeypatch):
    """The image path needs Pillow and nothing else."""
    import importlib

    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.delitem(sys.modules, "localm.inference.media", raising=False)

    media = importlib.import_module("localm.inference.media")
    img = media.decode_image_url(_PNG_DATA_URI)
    assert img.size == (1, 1)


def test_pillow_is_a_core_dependency_not_an_extra():
    """The packaging half."""
    root = Path(__file__).resolve().parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    core = data["project"]["dependencies"]
    assert any("pillow" in spec.lower() for spec in core), (
        "pillow must be a core dependency: the GGUF vision path imports it and "
        f"the torch stack is optional. core deps were: {core}"
    )

    gpu = data["project"].get("optional-dependencies", {}).get("gpu", [])
    assert not any("pillow" in spec.lower() for spec in gpu), (
        "pillow is declared in BOTH core and the gpu extra; drop the duplicate so "
        "there is one place that decides the version"
    )


def test_cli_reports_the_missing_library_not_vision_guidance(monkeypatch, capsys):
    """The CLI's `except UnsupportedInputError` arm DISCARDS the exception and prints vision-capability guidance in its place ('pick or download a vision model')."""
    from localm.cli import chat as chat_mod

    class _Engine:
        def chat_stream(self, *a, **k):
            raise ImageDecodeUnavailable(
                "Cannot decode the attached image: the Pillow imaging library is "
                "not installed in this localm environment. Install it into the "
                "same environment (uv pip install pillow) and try again."
            )

    # Fail loudly if the guidance path is reached at all, rather than inferring
    # it from absent output: an assertion on missing text passes when the whole
    # call silently no-ops.
    def _boom(*a, **k):
        raise AssertionError("vision_input_guidance must not run for this error")

    import localm.model_manager as mm
    monkeypatch.setattr(mm, "vision_input_guidance", _boom, raising=False)

    out = chat_mod._stream_once(_Engine(), [{"role": "user", "content": "hi"}])
    text = capsys.readouterr().out
    assert out == ""
    assert "Pillow" in text, text
    assert "vision model" not in text.lower(), text
