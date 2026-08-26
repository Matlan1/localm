# SPDX-License-Identifier: AGPL-3.0-or-later
"""A missing Pillow must not be reported as a native inference fault.

Pillow shipping only in the `gpu` extra leaves any install that skipped the torch
stack - the CPU backend, or a torch install that failed, both of which print
"GGUF chat still works" - with no image decoder at all.

Asking a GGUF vision model about an image then produces:

    [inference error: Native inference fault (worker exit 1). The model has been
     unloaded and will reload on the next request. See the debug log for the
     native stack trace.]

Every clause of that is false: the debug log holds a plain `ModuleNotFoundError:
No module named 'PIL'`, there is no native fault and no native stack trace, and
the model is fine. The GGUF worker's dispatch loop lets an escaping exception
kill the process by design (it means a native fault left the model in an unknown
state), so an unguarded ImportError inside the worker inherits that treatment.

Two independent guards, because either alone leaves a real hole:
  1. Pillow is a CORE dependency, so the situation should not arise.
  2. If it arises anyway (a broken or hand-trimmed environment), the failure names
     its real cause and is recoverable rather than a fake native fault.
"""
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

# A well-formed 1x1 PNG. Never decoded in these tests: the import fails first.
_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_missing_pillow_raises_a_named_error_not_a_bare_importerror(monkeypatch):
    """`sys.modules["PIL"] = None` makes `from PIL import Image` raise
    ModuleNotFoundError, which is the exact shape a genuinely absent Pillow
    produces. Patching the LEAF dependency rather than decode_image_url itself
    keeps every other line of that function running for real."""
    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.delitem(sys.modules, "PIL.Image", raising=False)

    with pytest.raises(ImageDecodeUnavailable) as exc:
        decode_image_url(_PNG_DATA_URI)

    msg = str(exc.value)
    assert "Pillow" in msg, msg
    # The message names the fix, not merely the symptom.
    assert "pip install pillow" in msg.lower(), msg
    # The message does not imply a native fault.
    assert "native" not in msg.lower(), msg


def test_the_error_is_recoverable_by_the_gguf_worker(monkeypatch):
    """The worker survives ONLY exceptions it catches by type. `_runner.py`
    catches `UnsupportedInputError`; anything else escapes and kills the process,
    evicting the model. So the inheritance IS the recovery contract, and asserting
    it here is what stops a later refactor from silently reinstating the crash."""
    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.delitem(sys.modules, "PIL.Image", raising=False)

    with pytest.raises(UnsupportedInputError):
        decode_image_url(_PNG_DATA_URI)

    assert issubclass(ImageDecodeUnavailable, UnsupportedInputError)


def test_decoding_an_image_does_not_require_numpy(monkeypatch):
    """The image path needs Pillow and nothing else.

    numpy is NOT a core dependency - it arrives transitively via the voice extra,
    so a base install can easily lack it. A module-scope `import numpy` in
    `localm.inference.media` for a single return annotation under
    `from __future__ import annotations` is never evaluated at runtime and buys
    nothing, while making the whole module, decode_image_url included,
    unimportable without numpy.

    Blocking the LEAF module and re-importing is what makes this real: asserting
    on the source text would pass against any refactor that reintroduced the
    import somewhere else.
    """
    import importlib

    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.delitem(sys.modules, "localm.inference.media", raising=False)

    media = importlib.import_module("localm.inference.media")
    img = media.decode_image_url(_PNG_DATA_URI)
    assert img.size == (1, 1)


def test_pillow_is_a_core_dependency_not_an_extra():
    """The packaging half. Guard 2 above turns a crash into a clear message, but
    a clear message is still a broken feature: the GGUF vision path needs Pillow
    and needs no torch, so Pillow may not live behind the `gpu` extra."""
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
    """The CLI's `except UnsupportedInputError` arm DISCARDS the exception and
    prints vision-capability guidance in its place ("pick or download a vision
    model"). That is correct for its own case and wrong for this one: the model
    is vision-capable and the image is fine.

    A new subclass silently inherits into every existing handler of its parent,
    so this asserts the CLI distinguishes them rather than collapsing both into
    one message.
    """
    from localm.cli import chat as chat_mod

    class _Engine:
        def chat_stream(self, *a, **k):
            raise ImageDecodeUnavailable(
                "Cannot decode the attached image: the Pillow imaging library is "
                "not installed in this localm environment. Install it into the "
                "same environment (uv pip install pillow) and try again."
            )

    # Fail loudly if the guidance path is reached at all, rather than asserting
    # on absent output, which would pass when the whole call silently no-ops.
    def _boom(*a, **k):
        raise AssertionError("vision_input_guidance must not run for this error")

    import localm.model_manager as mm
    monkeypatch.setattr(mm, "vision_input_guidance", _boom, raising=False)

    out = chat_mod._stream_once(_Engine(), [{"role": "user", "content": "hi"}])
    text = capsys.readouterr().out
    assert out == ""
    assert "Pillow" in text, text
    assert "vision model" not in text.lower(), text
