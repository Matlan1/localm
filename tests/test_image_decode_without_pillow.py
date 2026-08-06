# SPDX-License-Identifier: AGPL-3.0-or-later
"""A missing Pillow must not be reported as a native inference fault.

Found during 0.1.4 release verification on a cold install with the base extras
(`coder,voice,monitor` - exactly what setup.bat/setup.sh install before the
OPTIONAL torch step). Pillow shipped only in the `gpu` extra, so any install that
skipped the torch stack - the CPU backend, or a torch install that failed, both
of which print "GGUF chat still works" - had no image decoder at all.

Asking a GGUF vision model about an image then produced:

    [inference error: Native inference fault (worker exit 1). The model has been
     unloaded and will reload on the next request. See the debug log for the
     native stack trace.]

Every clause of that was false. The debug log held a plain `ModuleNotFoundError:
No module named 'PIL'`; there was no native fault and no native stack trace, and
the model was fine. The GGUF worker's dispatch loop lets an escaping exception
kill the process on purpose (it means a native fault left the model in an unknown
state), so an unguarded ImportError inside the worker inherited that treatment.

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

# A 1x1 PNG. Never decoded in these tests - the import fails first - but a
# well-formed URI keeps the test honest about WHICH failure it observed.
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
    # It must name the fix, not merely the symptom.
    assert "pip install pillow" in msg.lower(), msg
    # And it must NOT imply a native fault, which is the wrong-diagnosis half.
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
