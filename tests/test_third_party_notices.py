# SPDX-License-Identifier: AGPL-3.0-or-later
"""THIRD-PARTY-NOTICES.md agrees with the licence files vendored beside the code.

The root notices file is the public statement of what the repository bundles.
Each vendor directory carries its own licence texts (and the TTS bundle a
NOTICE.md table). These tests read both sides and compare:

  * every vendored ``LICENSE.*`` file has a root-table row for its component
    whose licence is the one the file's own text declares;
  * every package in the TTS NOTICE.md table has a root-table row with the same
    licence;
  * the root file makes no blanket "permissive" claim while a copyleft component
    is listed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTICES = ROOT / "THIRD-PARTY-NOTICES.md"
TTS_VENDOR = ROOT / "localm" / "plugins" / "builtin" / "tts" / "static" / "vendor"
GUI_VENDOR = ROOT / "localm" / "plugins" / "gui" / "static" / "vendor"

# LICENSE file suffix -> a fragment that must appear in the root row's Component cell.
COMPONENT_KEY = {
    "LICENSE.kokoro-js": "kokoro-js",
    "LICENSE.transformers-js": "transformers.js",
    "LICENSE.onnxruntime-web": "onnxruntime-web",
    "LICENSE.espeak-ng": "espeak ng",
    "LICENSE.jsqr": "jsqr",
    "inter/OFL.txt": "inter",
}

_FAMILIES = (
    ("GPL-3.0", re.compile(r"GNU GENERAL PUBLIC LICENSE\s+Version 3", re.IGNORECASE)),
    ("Apache-2.0", re.compile(r"Apache License\s+Version 2\.0", re.IGNORECASE)),
    ("OFL-1.1", re.compile(r"SIL OPEN FONT LICENSE", re.IGNORECASE)),
    ("MIT", re.compile(r"^\s*MIT License", re.IGNORECASE)),
)


def licence_family(text: str) -> str:
    """Name the licence a licence text declares, from its own opening wording."""
    for family, pattern in _FAMILIES:
        if pattern.search(text[:2000]):
            return family
    raise AssertionError("licence text not recognised")


def normalise(licence: str) -> str:
    """Reduce a table's licence cell to a comparable family name."""
    cell = re.sub(r"\(c\).*", "", licence, flags=re.IGNORECASE).strip()
    cell = cell.replace("`", "")
    if cell.upper().startswith("GPL-3.0"):
        return "GPL-3.0"
    if cell.upper().startswith("SIL OFL"):
        return "OFL-1.1"
    return cell


def _rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        if line.startswith("|") and not re.match(r"^\|[-| ]+\|$", line):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


def root_vendored_rows() -> dict[str, str]:
    """Component cell (lowercased) -> normalised licence, for the browser-asset tables."""
    text = NOTICES.read_text(encoding="utf-8")
    section = text.split("## Vendored in this repository", 1)[1].split("\n## ", 1)[0]
    out = {}
    for cells in _rows(section):
        if len(cells) >= 4 and cells[0] != "Component":
            out[cells[0].lower()] = normalise(cells[2])
    return out


def tts_notice_rows() -> dict[str, str]:
    """Package name -> normalised licence, from the TTS NOTICE.md package table."""
    text = (TTS_VENDOR / "NOTICE.md").read_text(encoding="utf-8")
    out = {}
    for cells in _rows(text.split("\n## ", 1)[0]):
        if len(cells) == 4 and cells[0] != "Package":
            name = re.sub(r"\(.*?\)", "", cells[0]).replace("`", "").strip()
            out[name.lower()] = normalise(cells[2])
    return out


def _find(rows: dict[str, str], key: str) -> str | None:
    key = key.lower()
    return next((lic for comp, lic in rows.items() if key in comp), None)


def test_the_parsers_read_real_content():
    assert len(root_vendored_rows()) >= 10
    assert {"kokoro-js", "phonemizer"} <= set(tts_notice_rows())


@pytest.mark.parametrize("suffix", sorted(COMPONENT_KEY))
def test_every_vendored_licence_file_has_a_matching_root_row(suffix):
    base = GUI_VENDOR if suffix in ("LICENSE.jsqr", "inter/OFL.txt") else TTS_VENDOR
    declared = licence_family((base / suffix).read_text(encoding="utf-8"))
    listed = _find(root_vendored_rows(), COMPONENT_KEY[suffix])
    assert listed is not None, f"no root row for {suffix}"
    assert listed == declared


def test_every_licence_file_in_a_vendor_directory_is_covered_by_the_mapping():
    on_disk = {f"LICENSE.{p.name.split('.', 1)[1]}" for p in (*TTS_VENDOR.glob("LICENSE.*"),
                                                                *GUI_VENDOR.glob("LICENSE.*"))}
    assert on_disk <= set(COMPONENT_KEY)


# TTS NOTICE.md package name -> a fragment of the root row's Component cell.
PACKAGE_KEY = {
    "kokoro-js": "kokoro-js",
    "@huggingface/transformers": "transformers.js",
    "onnxruntime-web": "onnxruntime-web",
    "phonemizer": "phonemizer",
    "espeak ng": "espeak ng",
}


def test_the_package_mapping_covers_every_package_the_tts_notice_lists():
    assert set(tts_notice_rows()) == set(PACKAGE_KEY)


@pytest.mark.parametrize("package", sorted(PACKAGE_KEY))
def test_every_package_in_the_tts_notice_has_a_root_row_with_the_same_licence(package):
    listed = _find(root_vendored_rows(), PACKAGE_KEY[package])
    assert listed is not None, f"no root row for {package}"
    assert listed == tts_notice_rows()[package]


def test_the_root_file_does_not_call_a_copyleft_bundle_permissive():
    text = NOTICES.read_text(encoding="utf-8")
    assert "GPL-3.0" in set(root_vendored_rows().values())
    assert "All are permissive" not in text
