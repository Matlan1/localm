# SPDX-License-Identifier: AGPL-3.0-or-later
"""U5: tokenizers must be pinned to ONE version so it stops flipping between
setup runs (the reported 0.23.1 <-> 0.22.2 churn). tokenizers is a transitive
dep of the unpinned transformers~=5.12, with no lockfile, so it floated.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_PIN = re.compile(r"tokenizers==(\d+\.\d+\.\d+)")


def _versions(name):
    return set(_PIN.findall((ROOT / name).read_text(encoding="utf-8")))


def test_pyproject_pins_tokenizers():
    assert len(_versions("pyproject.toml")) == 1


def test_inline_transformers_installs_also_pin_tokenizers():
    for name in ("setup.sh", "setup.bat"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for line in text.splitlines():
            if "transformers[kernels]" in line and "uv pip install" in line:
                assert "tokenizers==" in line, f"{name}: {line.strip()}"


def test_single_tokenizers_version_across_files():
    vs = set()
    for name in ("pyproject.toml", "setup.sh", "setup.bat"):
        vs |= _versions(name)
    assert len(vs) == 1, f"expected one tokenizers version, found {vs}"
