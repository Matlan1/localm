# SPDX-License-Identifier: AGPL-3.0-or-later
"""tokenizers must be pinned to ONE version, so it cannot flip between setup
runs. It is a transitive dependency of transformers, and there is no lockfile.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_PIN = re.compile(r"tokenizers==(\d+\.\d+\.\d+)")


def _versions(name):
    return set(_PIN.findall((ROOT / name).read_text(encoding="utf-8")))


def test_pyproject_pins_tokenizers():
    assert len(_versions("pyproject.toml")) == 1


def test_installers_carry_no_inline_transformers_pin():
    """setup.sh, setup.bat and installer/gui.py must resolve the HF stack
    from pyproject's [hf] extra, never from an inline "transformers["
    specifier of their own."""
    for name in ("setup.sh", "setup.bat", "installer/gui.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        hits = [line.strip() for line in text.splitlines() if "transformers[" in line]
        assert not hits, f"{name} carries its own transformers[ specifier: {hits}"


def test_single_tokenizers_version_across_files():
    vs = set()
    for name in ("pyproject.toml", "setup.sh", "setup.bat", "installer/gui.py"):
        vs |= _versions(name)
    assert len(vs) == 1, f"expected one tokenizers version, found {vs}"
