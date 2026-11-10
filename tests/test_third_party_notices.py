# SPDX-License-Identifier: AGPL-3.0-or-later
"""THIRD-PARTY-NOTICES.md agrees with the licence files vendored beside the code.

The root notices file is the public statement of what the repository bundles.
Each vendor directory carries its own licence texts (and the TTS bundle a
NOTICE.md table). These tests read both sides and compare:

  * every vendored ``LICENSE.*`` file has a root-table row for its component
    whose licence is the one the file's own text declares;
  * every component row of the GUI table has its licence text beside the files,
    one text per licence the row offers;
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
    "LICENSE.katex": "katex",
    "LICENSE.marked": "marked",
    "LICENSE.highlightjs": "highlight.js",
    "LICENSE.dompurify": "dompurify",
    "LICENSE.dompurify-mpl": "dompurify",
    "inter/OFL.txt": "inter",
}

# Licence files that live in the GUI vendor directory (the rest are the TTS bundle's).
GUI_FILES = frozenset({
    "LICENSE.jsqr", "LICENSE.katex", "LICENSE.marked", "LICENSE.highlightjs",
    "LICENSE.dompurify", "LICENSE.dompurify-mpl", "inter/OFL.txt",
})

_FAMILIES = (
    ("GPL-3.0", re.compile(r"GNU GENERAL PUBLIC LICENSE\s+Version 3", re.IGNORECASE)),
    ("Apache-2.0", re.compile(r"Apache License\s+Version 2\.0", re.IGNORECASE)),
    ("OFL-1.1", re.compile(r"SIL OPEN FONT LICENSE", re.IGNORECASE)),
    ("MPL-2.0", re.compile(r"^\s*Mozilla Public License Version 2\.0", re.IGNORECASE)),
    ("BSD-3-Clause", re.compile(r"^\s*BSD 3-Clause License", re.IGNORECASE)),
    ("MIT", re.compile(r"^\s*(?:The )?MIT License"
                       r"|Permission is hereby granted, free of charge, to any person "
                       r"obtaining a copy\s+of this software", re.IGNORECASE)),
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


def offered(listed: str) -> set[str]:
    """The licences a table cell offers: one, or the alternatives of an ``A OR B`` cell."""
    return {part.strip() for part in listed.split(" OR ")}


def gui_vendored_rows() -> dict[str, str]:
    """Component cell (lowercased) -> normalised licence, for the GUI table only."""
    text = NOTICES.read_text(encoding="utf-8")
    section = text.split("## Vendored in this repository", 1)[1].split("\n## ", 1)[0]
    table = section.split("The TTS plugin vendors", 1)[0]
    return {cells[0].lower(): normalise(cells[2]) for cells in _rows(table)
            if len(cells) >= 4 and cells[0] != "Component"}


@pytest.mark.parametrize("text, family", [
    ("MIT License\n\nCopyright (c) x", "MIT"),
    ("The MIT License (MIT)\n\nCopyright (c) x", "MIT"),
    ("# License information\n\nPermission is hereby granted, free of charge, to any "
     "person obtaining a copy\nof this software", "MIT"),
    ("BSD 3-Clause License\n\nCopyright (c) x", "BSD-3-Clause"),
    ("Mozilla Public License Version 2.0\n====", "MPL-2.0"),
    ("\n      Apache License\n   Version 2.0, January 2004", "Apache-2.0"),
])
def test_licence_family_names_each_text_by_its_own_wording(text, family):
    assert licence_family(text) == family


def test_licence_family_refuses_a_text_it_does_not_know():
    with pytest.raises(AssertionError):
        licence_family("Redistribution permitted under terms of my own.")


def test_the_parsers_read_real_content():
    assert len(root_vendored_rows()) >= 10
    assert {"kokoro-js", "phonemizer"} <= set(tts_notice_rows())


@pytest.mark.parametrize("suffix", sorted(COMPONENT_KEY))
def test_every_vendored_licence_file_has_a_matching_root_row(suffix):
    base = GUI_VENDOR if suffix in GUI_FILES else TTS_VENDOR
    declared = licence_family((base / suffix).read_text(encoding="utf-8"))
    listed = _find(root_vendored_rows(), COMPONENT_KEY[suffix])
    assert listed is not None, f"no root row for {suffix}"
    assert declared in offered(listed)


def test_the_gui_table_is_read_without_the_tts_rows():
    rows = gui_vendored_rows()
    assert len(rows) >= 6
    assert not any("kokoro" in comp or "espeak" in comp for comp in rows)


@pytest.mark.parametrize("component", sorted(gui_vendored_rows()))
def test_every_gui_component_has_a_licence_text_for_each_licence_its_row_offers(component):
    files = [f for f in GUI_FILES if COMPONENT_KEY[f] in component]
    assert files, f"no licence text beside the vendored files for {component!r}"
    declared = {licence_family((GUI_VENDOR / f).read_text(encoding="utf-8")) for f in files}
    assert declared == offered(gui_vendored_rows()[component])


def test_the_root_file_says_where_the_gui_licence_texts_are():
    text = NOTICES.read_text(encoding="utf-8")
    section = text.split("## Vendored in this repository", 1)[1].split("The TTS plugin vendors", 1)[0]
    for name in sorted(f for f in GUI_FILES if f.startswith("LICENSE.")):
        assert f"`{name}`" in section, name


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


def test_the_llama_cpp_rows_carry_the_copyright_line_of_the_notice_setup_llama_writes():
    from localm.setup_llama.library_files import _LLAMA_CPP_MIT_NOTICE
    line = next(ln for ln in _LLAMA_CPP_MIT_NOTICE.splitlines() if ln.startswith("Copyright"))
    holder = line.removeprefix("Copyright ")
    rows = [cells for cells in _rows(NOTICES.read_text(encoding="utf-8"))
            if cells[0] in ("llama.cpp", "ggml")]
    assert len(rows) == 2
    assert all(cells[1] == f"MIT {holder}" for cells in rows)


def test_the_root_file_does_not_call_a_copyleft_bundle_permissive():
    text = NOTICES.read_text(encoding="utf-8")
    assert "GPL-3.0" in set(root_vendored_rows().values())
    assert "All are permissive" not in text
