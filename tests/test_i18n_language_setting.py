# SPDX-License-Identifier: AGPL-3.0-or-later
"""The `language` config key and the GUI language catalogs.

The web GUI ships English inside app/i18n-en.js and fetches
static/i18n/<id>.json for every other language. The set of ids is declared
twice on purpose (Python validates PATCH /v1/config, the GUI builds the picker
and the catalog URL from it), so these tests pin the two together and check
that every declared language actually has a catalog on disk.
"""
import json
import re
from pathlib import Path

import pytest

from localm import settings_schema as ss
from localm.config import DEFAULT_CONFIG

STATIC = Path(ss.__file__).resolve().parent / "plugins" / "gui" / "static"
I18N_JS = STATIC / "app" / "i18n.js"
CATALOGS = STATIC / "i18n"


def _gui_language_ids() -> list:
    """The ids in the GUI's own LANGUAGES registry, in picker order."""
    src = I18N_JS.read_text(encoding="utf-8")
    m = re.search(r"export const LANGUAGES = \[(.*?)\];", src, re.S)
    assert m, "LANGUAGES registry not found in app/i18n.js"
    return re.findall(r'id:\s*"([^"]+)"', m.group(1))


def test_language_ids_match_the_gui_registry():
    assert _gui_language_ids() == ss.LANGUAGE_IDS


def test_english_is_the_first_language_and_the_default():
    assert ss.LANGUAGE_IDS[0] == "en"
    assert DEFAULT_CONFIG["language"] == "en"


def test_every_declared_language_has_a_catalog_that_parses():
    for lang in ss.LANGUAGE_IDS:
        path = CATALOGS / f"{lang}.json"
        if lang == "en":
            # English ships inside the app, not as a fetched catalog.
            assert not path.exists()
            continue
        assert path.is_file(), f"{lang} is offered but {path.name} does not exist"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict) and data, f"{path.name} is not a non-empty object"
        assert all(isinstance(v, str) for v in data.values())


def test_a_catalog_on_disk_is_offered_in_the_picker():
    on_disk = sorted(p.stem for p in CATALOGS.glob("*.json"))
    assert on_disk == sorted(set(ss.LANGUAGE_IDS) - {"en"}), (
        "a catalog that is not in LANGUAGE_IDS can never be selected, and an id "
        "with no catalog cannot be loaded")


@pytest.mark.parametrize("lang", ["en", "de"])
def test_a_known_language_is_accepted(lang):
    assert ss.validate_update({"language": lang}) == {"language": lang}


@pytest.mark.parametrize("bad", [
    "fr",                    # not shipped
    "../../etc/passwd",      # the value becomes part of a fetch path in the GUI
    "..%2fde",
    "de/../../secret",
    "",
    "EN",
])
def test_an_unknown_language_is_refused(bad):
    with pytest.raises(ValueError, match="language"):
        ss.validate_update({"language": bad})


def test_the_language_field_is_hidden_from_the_generic_settings_form():
    field = {f.key: f for f in ss.CORE_FIELDS}["language"]
    assert field.widget == ss.Widget.HIDDEN, (
        "a generic SELECT would render its own label in English, which is the one "
        "language the reader of this control may not have")
    assert not field.admin_only
