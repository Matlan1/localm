# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every "Settings > Page > Place" a text names exists on the GUI Settings page.

Scans the string literals and comments of every tracked ``localm/**/*.py`` file,
the paragraphs of README.md and every tracked docs/*.md, the English catalog
(app/i18n-en.js) and the text of index.html for
``Settings > ...`` (also ``->``, the arrow sign, ``&gt;``) and ``(Settings, ...``.
The first segment must be a nav page label from ``SETTINGS_GROUPS`` in
pages/settings.js. Every later segment must be something that page shows: the
label of a field rendered there (placed by ``CORE_GROUP_TO_TOP``, labelled as the
catalog labels it), a section heading, a static card heading or sub-heading from
index.html, or a media plugin box and its fields. All of it is read from the
shipped files when the test runs.
"""
from __future__ import annotations

import ast
import html
import io
import json
import re
import subprocess
import tokenize
from pathlib import Path

from localm.settings_schema import (CORE_FIELDS, MEDIA_PLUGIN_FIELDS,
                                    MEDIA_PLUGIN_LABELS, MEDIA_PLUGINS, Widget,
                                    media_fields_for)

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "localm" / "plugins" / "gui" / "static"
SETTINGS_JS = (STATIC / "pages" / "settings.js").read_text(encoding="utf-8")
INDEX_HTML = (STATIC / "index.html").read_text(encoding="utf-8")
I18N_EN_JS = (STATIC / "app" / "i18n-en.js").read_text(encoding="utf-8")

SEP = re.compile(r"\s*(?:->|>|→)\s*")
START = re.compile(r"\bSettings\s*(?:->|>|→)\s*|\(Settings,\s*")

# Settings places of another app that the certificate how-to texts walk through:
# (app the text names, the exact place). A match is skipped only when the text
# names that app and the place starts there, word for word, written with any of
# the separators the check reads.
OTHER_APP_PLACES = (
    ("Firefox", "Settings → Privacy & Security → Certificates → "
                "View Certificates → Authorities → Import"),
    ("iOS", "Settings → General → VPN & Device Management"),
    ("iOS", "Settings → General → About → Certificate Trust Settings"),
)


# --------------------------------------------------------------------------- #
#  JavaScript reading                                                          #
# --------------------------------------------------------------------------- #

def _js_tokens(src: str) -> list:
    """Tokens of a JS source as (kind, value): kind "str" (decoded string
    literal), or "op" (any other single non-space character). Comments are
    dropped; identifiers come out one character at a time."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif c in "\"'":
            j = i + 1
            while src[j] != c:
                j += 2 if src[j] == "\\" else 1
            raw = src[i + 1:j]
            if c == "'":
                raw = raw.replace("\\'", "'").replace('"', '\\"')
            out.append(("str", json.loads('"' + raw + '"')))
            i = j + 1
        elif c.isspace():
            i += 1
        else:
            out.append(("op", c))
            i += 1
    return out


def _catalog_entries(src: str) -> list:
    """(key, value) for every ``"key": "text" + "text"`` entry of a catalog."""
    toks = _js_tokens(src)
    out = []
    i = 0
    while i < len(toks) - 2:
        if toks[i][0] == "str" and toks[i + 1] == ("op", ":") and toks[i + 2][0] == "str":
            key, parts = toks[i][1], [toks[i + 2][1]]
            i += 3
            while i < len(toks) - 1 and toks[i] == ("op", "+") and toks[i + 1][0] == "str":
                parts.append(toks[i + 1][1])
                i += 2
            out.append((key, "".join(parts)))
        else:
            i += 1
    return out


EN = dict(_catalog_entries(I18N_EN_JS))


def _js_object(name: str) -> dict:
    m = re.search(r"const " + name + r" = \{(.*?)\n\};", SETTINGS_JS, re.S)
    assert m, f"{name} not found in pages/settings.js"
    body = re.sub(r"//[^\n]*", "", m.group(1))
    return {k or q: v for k, q, v in re.findall(
        r'(?:([A-Za-z]\w*)|"([^"]+)")\s*:\s*"([^"]+)"', body)}


def _settings_groups() -> list:
    m = re.search(r"const SETTINGS_GROUPS = \[(.*?)\n\];", SETTINGS_JS, re.S)
    assert m, "SETTINGS_GROUPS not found in pages/settings.js"
    groups = re.findall(r'\{\s*id:\s*"([^"]+)",\s*label:\s*"([^"]+)"\s*\}', m.group(1))
    assert groups, "SETTINGS_GROUPS has no entries"
    return groups


# --------------------------------------------------------------------------- #
#  What each Settings page shows                                               #
# --------------------------------------------------------------------------- #

def _camel(key: str) -> str:
    return re.sub(r"_([a-z0-9])", lambda m: m.group(1).upper(), key)


def _field_label(key: str, fallback: str, scopes=None) -> str:
    """The English label the GUI renders for a field (fieldCatalogText)."""
    ck = _camel(key)
    keys = ([f"settings.field.{s}.{ck}.label" for s in scopes] if scopes
            else [f"settings.field.{ck}.label"])
    for k in keys:
        if k in EN:
            return EN[k]
    return fallback


def _static_cards() -> list:
    """(page id, [heading, sub-heading, label, button text, ...]) per settings
    card in index.html, each in the English the page shows."""
    out = []
    for m in re.finditer(r'<section class="card settings-section"[^>]*'
                         r'data-group="([^"]+)"[^>]*>(.*?)</section>', INDEX_HTML, re.S):
        names = []
        for h in re.finditer(r"<(h3|h4|label|button)\b([^>]*)>(.*?)</\1>", m.group(2), re.S):
            key = re.search(r'\sdata-i18n="([^"]+)"', h.group(2))
            text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", h.group(3)))).strip()
            names.append(EN.get(key.group(1), text) if key else text)
        out.append((m.group(1), names))
    return out


def settings_places() -> dict:
    """{page label: set of place names shown on that page}."""
    groups = _settings_groups()
    label_of = dict(groups)
    group_to_top = _js_object("CORE_GROUP_TO_TOP")
    heading_key = _js_object("CORE_SECTION_HEADING_KEY")
    names = {gid: set() for gid, _ in groups}

    media_mapped = {m.global_key for m in MEDIA_PLUGIN_FIELDS}
    for f in CORE_FIELDS:
        if f.widget == Widget.HIDDEN:
            continue
        if f.group == "Media" and f.key in media_mapped:
            continue
        names[group_to_top[f.group]].add(_field_label(f.key, f.label))
    for group, key in heading_key.items():
        names[group_to_top[group]].add(EN[key])
    for key, gid in re.findall(r'settingsSectionHead\(t\("([^"]+)"\),\s*"([^"]+)"\)',
                               SETTINGS_JS):
        names[gid].add(EN[key])
    for gid, heads in _static_cards():
        names[gid].update(heads)
    for plugin in MEDIA_PLUGINS:
        names["media"].add(MEDIA_PLUGIN_LABELS[plugin])
        for f in media_fields_for(plugin):
            if f.widget != Widget.HIDDEN:
                names["media"].add(_field_label(f.key, f.label, [plugin, "media"]))
    return {label_of[gid]: {n for n in v if n} for gid, v in names.items()}


# --------------------------------------------------------------------------- #
#  Texts                                                                       #
# --------------------------------------------------------------------------- #

def _tracked_py() -> list:
    try:
        r = subprocess.run(["git", "ls-files", "localm"], cwd=ROOT,
                           capture_output=True, text=True, check=True)
        files = [ROOT / p for p in r.stdout.splitlines() if p.endswith(".py")]
    except (OSError, subprocess.CalledProcessError):
        files = sorted((ROOT / "localm").rglob("*.py"))
    return [p for p in files if p.is_file()]


def _py_texts(path: Path) -> list:
    """(line, text) for each string literal (an f-string's fields read as
    ``{}``, implicit concatenation joined) and each run of comment lines."""
    src = path.read_text(encoding="utf-8")
    out = []
    tree = ast.parse(src)
    inner = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            inner.update(id(v) for v in node.values)
            text = "".join(v.value if isinstance(v, ast.Constant) else "{}"
                           for v in node.values)
            out.append((node.lineno, text))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in inner):
            out.append((node.lineno, node.value))
    block, start, last = [], 0, -2
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type != tokenize.COMMENT:
            continue
        line = tok.start[0]
        if line != last + 1 and block:
            out.append((start, " ".join(block)))
            block = []
        if not block:
            start = line
        block.append(tok.string.lstrip("#").strip())
        last = line
    if block:
        out.append((start, " ".join(block)))
    return out


def _catalog_texts() -> list:
    return [(key, html.unescape(value)) for key, value in _catalog_entries(I18N_EN_JS)]


def _html_texts() -> list:
    src = re.sub(r"<!--.*?-->", lambda m: "\n" * m.group(0).count("\n"), INDEX_HTML,
                 flags=re.S)
    src = re.sub(r"<br\s*/?>", " ", src)
    src = re.sub(r"</?(?:b|strong|code|em|i|a|span|kbd)\b[^>]*>", "", src)
    out = []
    for m in re.finditer(r">([^<]+)<", src):
        out.append((src.count("\n", 0, m.start()) + 1, html.unescape(m.group(1))))
    for m in re.finditer(r'\s(?:title|placeholder|aria-label)="([^"]*)"', src):
        out.append((src.count("\n", 0, m.start()) + 1, html.unescape(m.group(1))))
    return out


def _tracked_docs() -> list:
    """README.md and every tracked docs/*.md."""
    try:
        r = subprocess.run(["git", "ls-files", "README.md", "docs"], cwd=ROOT,
                           capture_output=True, text=True, check=True)
        files = [ROOT / p for p in r.stdout.splitlines() if p.endswith(".md")]
    except (OSError, subprocess.CalledProcessError):
        files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    return [p for p in files if p.is_file()]


def _md_texts(path: Path) -> list:
    """(line, text) for each paragraph of a Markdown file, with links reduced
    to their text and emphasis and code marks removed."""
    out, block, start = [], [], 0
    lines = path.read_text(encoding="utf-8").splitlines() + [""]
    for n, line in enumerate(lines, 1):
        if line.strip():
            if not block:
                start = n
            block.append(line)
            continue
        if block:
            text = " ".join(block)
            text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
            text = re.sub(r"\*\*|__|`|(?<!\w)\*(?!\s)|(?<!\s)\*(?!\w)", "", text)
            out.append((start, html.unescape(text)))
            block = []
    return out


def all_texts() -> list:
    """(where, text) for every text the check reads."""
    out = []
    for path in _tracked_py():
        rel = path.relative_to(ROOT).as_posix()
        out += [(f"{rel}:{line}", text) for line, text in _py_texts(path)]
    for path in _tracked_docs():
        rel = path.relative_to(ROOT).as_posix()
        out += [(f"{rel}:{line}", text) for line, text in _md_texts(path)]
    rel = (STATIC / "app" / "i18n-en.js").relative_to(ROOT).as_posix()
    out += [(f"{rel} {key}", text) for key, text in _catalog_texts()]
    rel = (STATIC / "index.html").relative_to(ROOT).as_posix()
    out += [(f"{rel}:{line}", text) for line, text in _html_texts()]
    return out


# --------------------------------------------------------------------------- #
#  The check                                                                   #
# --------------------------------------------------------------------------- #

def _longest_prefix(text: str, pos: int, names) -> str | None:
    best = None
    for name in names:
        end = pos + len(name)
        if text.startswith(name, pos) and (end == len(text) or not text[end].isalnum()):
            if best is None or len(name) > len(best):
                best = name
    return best


def check_text(text: str, places: dict) -> list:
    """A message per ``Settings > ...`` in *text* that names a missing place."""
    text = re.sub(r"\s+", " ", text)
    bad = []
    for m in START.finditer(text):
        rest = SEP.sub(" > ", text[m.start():])
        if any(app in text and rest.startswith(SEP.sub(" > ", place))
               for app, place in OTHER_APP_PLACES):
            continue
        pos = m.end()
        page = _longest_prefix(text, pos, places)
        if page is None:
            bad.append(f"no Settings page {text[pos:pos + 40]!r}")
            continue
        pos += len(page)
        while True:
            sep = SEP.match(text, pos)
            if not sep:
                break
            pos = sep.end()
            name = _longest_prefix(text, pos, places[page])
            if name is None:
                bad.append(f"nothing on Settings > {page} called {text[pos:pos + 40]!r}")
                break
            pos += len(name)
    return bad


def test_every_settings_place_named_in_a_text_exists():
    places = settings_places()
    bad = []
    for where, text in all_texts():
        bad += [f"{where}: {msg}" for msg in check_text(text, places)]
    assert not bad, "texts name Settings places that do not exist:\n" + "\n".join(bad)


def test_page_labels_match_the_catalog():
    for gid, label in _settings_groups():
        assert EN.get(f"settings.group.{gid}") == label, gid


def test_the_check_reads_every_kind_of_text():
    wheres = [w for w, _ in all_texts()]
    assert any(w.startswith("localm/settings_schema.py:") for w in wheres)
    assert any(w.startswith("localm/plugins/gui/static/app/i18n-en.js ") for w in wheres)
    assert any(w.startswith("localm/plugins/gui/static/index.html:") for w in wheres)
    assert any(w.startswith("README.md:") for w in wheres)
    assert any(w.startswith("docs/phone.md:") for w in wheres)


def test_a_markdown_mention_is_read_through_its_emphasis_and_links(tmp_path):
    page = tmp_path / "page.md"
    page.write_text("Open **Settings > Server > Bind address**\nand go on.\n\n"
                    "See [Settings > Media > Image > Native model](x.md).\n", encoding="utf-8")
    places = settings_places()
    texts = _md_texts(page)
    assert [line for line, _ in texts] == [1, 4]
    assert check_text(texts[0][1], places)
    assert not check_text(texts[1][1], places)


def test_the_check_rejects_missing_places_and_accepts_real_ones():
    places = settings_places()
    assert check_text("set it in Settings > Images.", places)
    assert check_text("see (Settings, Images)", places)
    assert check_text("Settings > Server > Bind address", places)
    assert check_text("Settings -> Media -> Nope", places)
    assert check_text("Settings > Media > Image > Nope", places)
    assert not check_text("set it in Settings > Media > Image > Native model.", places)
    assert not check_text("(Settings > Media > Music > Native diffusion model)", places)
    assert not check_text("Settings > Server & network failed: x", places)
    assert not check_text("Settings > System > Updates > Inference runtime", places)
    assert not check_text("in Settings > Privacy & data.", places)
