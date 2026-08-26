# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm is AGPL-3.0-or-later only. The README must not advertise a separate
commercial license.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_states_agpl():
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "agpl-3.0-or-later" in text


def test_readme_has_no_commercial_license_offer():
    # The README must not offer a separate commercial license.
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "commercial license" not in text
