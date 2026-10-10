# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/bump_vendored_js.py: replacing a vendored GUI library with a verified
upstream npm release.

Everything runs offline against a copy of the real vendored files and the real
tests-js pin files, with an in-memory fake of the npm registry. The tests cover:

  * a dry run changes nothing; --write replaces exactly the library files and the
    pins that name them, byte for byte;
  * a refusal edits nothing: same or older version, malformed version, a registry
    that answers for another package, a deprecated release, a tampered tarball, a
    missing or duplicated tarball member, an inconsistent tree;
  * a major boundary (a 0.x minor counts) is called out in the checklist;
  * the regions the script edits match exactly once on the real tree.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import re
import shutil
import tarfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "bump_vendored_js.py"


def _load():
    spec = importlib.util.spec_from_file_location("bump_vendored_js", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bump():
    return _load()


# --------------------------------------------------------------------------- #
#  A copy of the real tree and a fake registry                                #
# --------------------------------------------------------------------------- #

@pytest.fixture
def tree(bump, tmp_path, monkeypatch):
    """The real vendor directory, the real vendor pin files and the real notices,
    copied under tmp_path, with the script pointed at the copy."""
    shutil.copytree(_ROOT / bump.VENDOR_REL, tmp_path / bump.VENDOR_REL)
    (tmp_path / bump.TESTS_REL).mkdir()
    for name in ("marked", "dompurify", "highlightjs", "katex"):
        src = _ROOT / bump.TESTS_REL / f"vendor-{name}.test.mjs"
        shutil.copy(src, tmp_path / bump.TESTS_REL / src.name)
    shutil.copy(_ROOT / "THIRD-PARTY-NOTICES.md", tmp_path / "THIRD-PARTY-NOTICES.md")
    monkeypatch.setattr(bump, "REPO", tmp_path)

    def no_network(req, timeout):
        raise AssertionError("a test reached the network")
    monkeypatch.setattr(bump, "_default_open", no_network)
    return tmp_path


def snapshot(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def make_tarball(files: dict, extra=None) -> bytes:
    """A gzip tar of {member name: bytes}; *extra* is a list of ready TarInfo +
    payload pairs for the odd members (links, duplicates)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        for info, payload in extra or []:
            tf.addfile(info, io.BytesIO(payload) if payload is not None else None)
    return buf.getvalue()


class Registry:
    """The npm registry answer and tarball download for one package version."""

    def __init__(self, bump, npm, version, tarball, **body_overrides):
        self.npm, self.version, self.tarball = npm, version, tarball
        self.url = f"https://registry.npmjs.org/{npm}/-/{npm.split('/')[-1]}-{version}.tgz"
        self.body = {
            "name": npm, "version": version,
            "dist": {
                "integrity": "sha512-" + base64.b64encode(
                    hashlib.sha512(tarball).digest()).decode(),
                "shasum": hashlib.sha1(tarball).hexdigest(),
                "tarball": self.url,
            },
        }
        self.body.update(body_overrides)
        self.json_urls, self.byte_urls = [], []

    def fetch_json(self, url):
        self.json_urls.append(url)
        return self.body

    def fetch_bytes(self, url):
        self.byte_urls.append(url)
        return self.tarball


def run(bump, argv, registry):
    return bump.main(argv, fetch_json_fn=registry.fetch_json,
                     fetch_bytes_fn=registry.fetch_bytes)


def vendored(tree, bump, name) -> bytes:
    return (tree / bump.VENDOR_REL / name).read_bytes().replace(b"\r\n", b"\n")


def release_files(tree, bump, key, new, css_changed=False, font_changes=None):
    """{tar member: bytes} for a pretend upstream release of *key* at *new*:
    the vendored bytes (LF, as npm carries them) with the version moved."""
    old = bump.read_state(bump.LIBS[key], tree)[3]
    out = {}
    for member, dest in bump.LIBS[key].files + bump.LIBS[key].extra_files:
        if not (tree / bump.VENDOR_REL / dest).exists():
            continue
        data = vendored(tree, bump, dest).replace(old.encode(), new.encode())
        if dest == "katex.min.css" and css_changed:
            data += b"\n.katex-extra{color:red}\n"
        out[member] = data
    if bump.LIBS[key].fonts:
        for path in sorted((tree / bump.VENDOR_REL / "fonts").glob("KaTeX_*.woff2")):
            out[f"package/dist/fonts/{path.name}"] = path.read_bytes()
        for name, data in (font_changes or {}).items():
            if data is None:
                out.pop(f"package/dist/fonts/{name}", None)
            else:
                out[f"package/dist/fonts/{name}"] = data
    return out


def registry_for(bump, tree, key, new, **kw):
    files = release_files(tree, bump, key, new, **{k: kw.pop(k) for k in
                          ("css_changed", "font_changes") if k in kw})
    return Registry(bump, bump.LIBS[key].npm, new, make_tarball(files), **kw)


def n64(data: bytes) -> str:
    return base64.b64encode(hashlib.sha256(data.replace(b"\r\n", b"\n")).digest()).decode()


# --------------------------------------------------------------------------- #
#  Versions                                                                   #
# --------------------------------------------------------------------------- #

def test_crosses_major_counts_a_zero_x_minor(bump):
    assert bump.crosses_major((12, 0, 2), (13, 0, 0))
    assert not bump.crosses_major((3, 4, 13), (3, 5, 0))
    assert bump.crosses_major((0, 18, 4), (0, 19, 0))
    assert not bump.crosses_major((0, 18, 4), (0, 18, 5))


def test_the_content_hash_ignores_line_ending_conversion_only(bump):
    assert bump.normalised_sha256_b64(b"a\r\nb\r\n") == bump.normalised_sha256_b64(b"a\nb\n")
    assert bump.normalised_sha256_b64(b"a\nb\n") != bump.normalised_sha256_b64(b"a\nc\n")
    assert bump.normalised_sha256_b64(b"") == "47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU="


@pytest.mark.parametrize("bad", ["v3.4.16", "3.4", "3.4.16-beta.1", "latest", "", "3.4.16+x",
                                 "03.4.x", " 3.4.16", "3.4.16\n3.4.17"])
def test_only_a_plain_release_version_parses(bump, bad):
    with pytest.raises(bump.Refused):
        bump.parse_version(bad)


# --------------------------------------------------------------------------- #
#  The real tree: every edited region matches exactly once                    #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("key", ["marked", "DOMPurify", "highlight.js", "KaTeX"])
def test_the_real_tree_is_in_the_shape_the_script_edits(bump, key):
    lib = bump.LIBS[key]
    files, test_text, _, version = bump.read_state(lib, _ROOT)
    bump.parse_version(version)
    if lib.pinned:
        for dest, recorded in bump._pin_hashes(lib, test_text).items():
            assert recorded == n64(files[dest]), f"{dest} no longer matches its recorded pin"
        dummy = {dest: "X" * 44 for _, dest in lib.files}
        new_text = bump.rewrite_test(lib, test_text, version, "999.9.9", dummy, "a" * 40)
        assert new_text != test_text
        assert '"999.9.9"' in new_text
        assert "X" * 44 in new_text
        assert re.search(rf"npm pack {re.escape(lib.npm)}@999\.9\.9", new_text)


def test_the_dompurify_guard_has_no_version_pin_to_edit(bump):
    lib = bump.LIBS["DOMPurify"]
    _, test_text, _, _ = bump.read_state(lib, _ROOT)
    assert bump.rewrite_test(lib, test_text, "3.4.13", "3.4.14", {}, "a" * 40) == test_text


# --------------------------------------------------------------------------- #
#  DOMPurify: one file, no pins                                               #
# --------------------------------------------------------------------------- #

def test_dry_run_changes_nothing_and_shows_the_plan(bump, tree, capsys):
    before = snapshot(tree)
    reg = registry_for(bump, tree, "DOMPurify", "3.4.99")
    assert run(bump, ["--lib", "DOMPurify", "--tag", "3.4.99"], reg) == 0
    out = capsys.readouterr().out
    assert "replace localm/plugins/gui/static/vendor/purify.min.js" in out
    assert "dry run: nothing written" in out
    assert "REMAINING STEPS" in out and "MAJOR BOUNDARY" not in out
    assert snapshot(tree) == before


def test_write_replaces_the_library_byte_for_byte_and_nothing_else(bump, tree, capsys):
    before = snapshot(tree)
    reg = registry_for(bump, tree, "DOMPurify", "3.4.99")
    assert run(bump, ["--lib", "DOMPurify", "--tag", "3.4.99", "--write"], reg) == 0
    after = snapshot(tree)
    changed = {k for k in after if before.get(k) != after[k]}
    assert changed == {f"{bump.VENDOR_REL}/purify.min.js"}
    new_file = after[f"{bump.VENDOR_REL}/purify.min.js"]
    assert new_file == reg_files(reg)["package/dist/purify.min.js"]
    assert b"\r" not in new_file
    assert b"@license DOMPurify 3.4.99" in new_file

    capsys.readouterr()
    assert run(bump, ["--lib", "DOMPurify", "--tag", "3.4.99", "--write"], reg) == 1
    assert "not newer than the vendored 3.4.99" in capsys.readouterr().out


def reg_files(reg) -> dict:
    with tarfile.open(fileobj=io.BytesIO(reg.tarball), mode="r:gz") as tf:
        return {m.name: tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}


def test_a_crlf_only_difference_is_not_a_change(bump, tree):
    css = tree / bump.VENDOR_REL / "github-dark.min.css"
    css.write_bytes(css.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    files = release_files(tree, bump, "highlight.js", "11.13.0")
    assert b"\n" in files["package/styles/github-dark.min.css"]
    plan = bump.build_plan(bump.LIBS["highlight.js"], "11.13.0", files, {}, "a" * 40)
    assert not [c for c in plan["changes"] if c.rel.endswith("github-dark.min.css")]


# --------------------------------------------------------------------------- #
#  marked and highlight.js: version, hash and provenance pins                 #
# --------------------------------------------------------------------------- #

def test_marked_bump_moves_every_pin_and_only_the_pins(bump, tree, capsys):
    test_path = tree / bump.TESTS_REL / "vendor-marked.test.mjs"
    old_text = test_path.read_bytes().decode("utf-8")
    reg = registry_for(bump, tree, "marked", "13.0.0")
    assert run(bump, ["--lib", "marked", "--tag", "13.0.0", "--write"], reg) == 0
    out = capsys.readouterr().out
    assert "MAJOR BOUNDARY: marked 12.0.2 -> 13.0.0" in out

    new_text = test_path.read_bytes().decode("utf-8")
    new_file = reg_files(reg)["package/marked.min.js"]
    assert 'const VENDORED_VERSION = "13.0.0";' in new_text
    assert f'const PINNED_HASH = "{n64(new_file)}";' in new_text
    assert "npm pack marked@13.0.0  ->  package/marked.min.js" in new_text
    assert f"dist.shasum ({reg.body['dist']['shasum']})" in new_text
    assert 'test("the banner comment says 13.0.0"' in new_text
    old_lines, new_lines = old_text.splitlines(), new_text.splitlines()
    assert len(old_lines) == len(new_lines)
    differing = [i for i, (a, b) in enumerate(zip(old_lines, new_lines, strict=True)) if a != b]
    assert len(differing) == 5, "version, hash, npm pack, shasum and title only"
    assert (tree / bump.VENDOR_REL / "marked.min.js").read_bytes() == new_file


def test_the_test_file_keeps_its_own_line_endings(bump, tree):
    test_path = tree / bump.TESTS_REL / "vendor-marked.test.mjs"
    for newline in (b"\n", b"\r\n"):
        text = test_path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", newline)
        test_path.write_bytes(text)
        plan = bump.build_plan(bump.LIBS["marked"], "12.0.3",
                               release_files(tree, bump, "marked", "12.0.3"), {}, "b" * 40)
        change = [c for c in plan["changes"] if c.text][0]
        bump._write_change(change)
        data = test_path.read_bytes()
        assert data.count(newline) == data.count(b"\n")
        assert (b"\r" in data) == (newline == b"\r\n")
        test_path.write_bytes(text)


def test_highlightjs_bump_pins_the_new_hash_and_leaves_an_identical_theme_alone(
        bump, tree, capsys):
    reg = registry_for(bump, tree, "highlight.js", "11.13.0")
    assert run(bump, ["--lib", "highlight.js", "--tag", "11.13.0", "--write"], reg) == 0
    out = capsys.readouterr().out
    assert "replace localm/plugins/gui/static/vendor/github-dark.min.css" not in out
    text = (tree / bump.TESTS_REL / "vendor-highlightjs.test.mjs").read_bytes().decode("utf-8")
    new_file = reg_files(reg)["package/highlight.min.js"]
    assert 'const VENDORED_VERSION = "11.13.0";' in text
    assert f'const PINNED_HASH = "{n64(new_file)}";' in text
    assert "npm pack @highlightjs/cdn-assets@11.13.0" in text
    assert "and both say 11.13.0" in text
    assert "(11.12.0)" in text, "the MIN_SAFE floor wording is not a pin"
    assert reg.json_urls == ["https://registry.npmjs.org/@highlightjs%2Fcdn-assets/11.13.0"]


def test_highlightjs_theme_that_differs_is_replaced_too(bump, tree):
    files = release_files(tree, bump, "highlight.js", "11.13.0")
    files["package/styles/github-dark.min.css"] += b".hljs-new{color:red}\n"
    reg = Registry(bump, "@highlightjs/cdn-assets", "11.13.0", make_tarball(files))
    assert run(bump, ["--lib", "highlight.js", "--tag", "11.13.0", "--write"], reg) == 0
    assert (tree / bump.VENDOR_REL / "github-dark.min.css").read_bytes().endswith(
        b".hljs-new{color:red}\n")


def test_a_theme_missing_from_the_release_is_not_an_error(bump, tree):
    files = release_files(tree, bump, "highlight.js", "11.13.0")
    del files["package/styles/github-dark.min.css"]
    reg = Registry(bump, "@highlightjs/cdn-assets", "11.13.0", make_tarball(files))
    assert run(bump, ["--lib", "highlight.js", "--tag", "11.13.0"], reg) == 0


# --------------------------------------------------------------------------- #
#  KaTeX: the matched set                                                     #
# --------------------------------------------------------------------------- #

def test_katex_bump_moves_the_set_pins_and_fonts_together(bump, tree, capsys):
    fonts = tree / bump.VENDOR_REL / "fonts"
    changed_font = b"new font bytes"
    reg = registry_for(bump, tree, "KaTeX", "0.19.0", css_changed=True, font_changes={
        "KaTeX_Main-Regular.woff2": changed_font,
        "KaTeX_Zzz-Regular.woff2": b"added font",
        "KaTeX_AMS-Regular.woff2": None})
    css_refs = reg_files(reg)["package/dist/katex.min.css"].decode()
    assert "KaTeX_AMS-Regular.woff2" in css_refs, "the fake release still references the removed font"
    assert run(bump, ["--lib", "KaTeX", "--tag", "0.19.0", "--write"], reg) == 1
    assert "references woff2 font(s) the release does not ship: KaTeX_AMS-Regular.woff2" \
        in capsys.readouterr().out
    assert (fonts / "KaTeX_AMS-Regular.woff2").exists(), "a refusal edits nothing"

    reg = registry_for(bump, tree, "KaTeX", "0.19.0", css_changed=True, font_changes={
        "KaTeX_Main-Regular.woff2": changed_font, "KaTeX_Zzz-Regular.woff2": b"added font"})
    capsys.readouterr()
    assert run(bump, ["--lib", "KaTeX", "--tag", "0.19.0", "--write"], reg) == 0
    out = capsys.readouterr().out
    assert "MAJOR BOUNDARY: KaTeX 0.18.4 -> 0.19.0" in out
    assert "fonts: 19 unchanged, 1 replaced, 1 added, 0 removed" in out
    assert (fonts / "KaTeX_Main-Regular.woff2").read_bytes() == changed_font
    assert (fonts / "KaTeX_Zzz-Regular.woff2").read_bytes() == b"added font"

    files = reg_files(reg)
    text = (tree / bump.TESTS_REL / "vendor-katex.test.mjs").read_bytes().decode("utf-8")
    assert 'const VENDORED_VERSION = "0.19.0";' in text
    for member, name in (("package/dist/katex.min.js", "katex.min.js"),
                         ("package/dist/katex.min.css", "katex.min.css"),
                         ("package/dist/contrib/auto-render.min.js", "auto-render.min.js")):
        assert f'"{name}": "{n64(files[member])}",' in text
        assert (tree / bump.VENDOR_REL / name).read_bytes() == files[member]
    assert "npm pack katex@0.19.0" in text


def test_a_katex_release_with_a_removed_font_the_css_no_longer_uses_deletes_it(bump, tree, capsys):
    css = vendored(tree, bump, "katex.min.css")
    css = css.replace(b"url(fonts/KaTeX_AMS-Regular.woff2)", b"url(fonts/KaTeX_AMS-Regular.woff)")
    files = release_files(tree, bump, "KaTeX", "0.19.0", font_changes={"KaTeX_AMS-Regular.woff2": None})
    files["package/dist/katex.min.css"] = css
    reg = Registry(bump, "katex", "0.19.0", make_tarball(files))
    assert run(bump, ["--lib", "KaTeX", "--tag", "0.19.0", "--write"], reg) == 0
    assert "0 added, 1 removed" in capsys.readouterr().out
    assert not (tree / bump.VENDOR_REL / "fonts" / "KaTeX_AMS-Regular.woff2").exists()


def test_a_katex_patch_release_is_not_a_major_boundary(bump, tree, capsys):
    reg = registry_for(bump, tree, "KaTeX", "0.18.5")
    assert run(bump, ["--lib", "KaTeX", "--tag", "0.18.5"], reg) == 0
    assert "MAJOR BOUNDARY" not in capsys.readouterr().out


# --------------------------------------------------------------------------- #
#  Refusals: nothing is edited                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tag, fragment", [
    ("3.4.13", "not newer than the vendored 3.4.13"),
    ("3.4.12", "not newer than the vendored 3.4.13"),
    ("2.9.9", "not newer than the vendored 3.4.13"),
    ("3.4.13-rc.1", "not a plain MAJOR.MINOR.PATCH"),
    ("v3.4.14", "not a plain MAJOR.MINOR.PATCH"),
    ("latest", "not a plain MAJOR.MINOR.PATCH"),
])
def test_a_same_older_or_malformed_version_is_refused_before_any_request(
        bump, tree, capsys, tag, fragment):
    before = snapshot(tree)
    reg = Registry(bump, "dompurify", "3.4.14", b"unused")
    assert run(bump, ["--lib", "DOMPurify", "--tag", tag, "--write"], reg) == 1
    assert fragment in capsys.readouterr().out
    assert reg.json_urls == [] and reg.byte_urls == []
    assert snapshot(tree) == before


@pytest.mark.parametrize("name, mutate, fragment", [
    ("another package", lambda b: b.update(name="dompurify-evil"), "not dompurify@3.4.99"),
    ("another version", lambda b: b.update(version="3.4.98"), "not dompurify@3.4.99"),
    ("deprecated", lambda b: b.update(deprecated="use something else"), "is deprecated"),
    ("no dist", lambda b: b.pop("dist"), "no dist block"),
    ("no integrity", lambda b: b["dist"].pop("integrity"), "no sha512 dist.integrity"),
    ("sha1-only integrity", lambda b: b["dist"].update(integrity="sha1-AAAA"),
     "no sha512 dist.integrity"),
    ("no shasum", lambda b: b["dist"].pop("shasum"), "no usable dist.shasum"),
    ("tarball elsewhere", lambda b: b["dist"].update(tarball="https://evil.example/x.tgz"),
     "is not on https://registry.npmjs.org"),
    ("tarball over http", lambda b: b["dist"].update(
        tarball="http://registry.npmjs.org/dompurify/-/dompurify-3.4.99.tgz"),
     "is not on https://registry.npmjs.org"),
])
def test_a_registry_answer_that_does_not_check_out_is_refused(
        bump, tree, capsys, name, mutate, fragment):
    before = snapshot(tree)
    reg = registry_for(bump, tree, "DOMPurify", "3.4.99")
    mutate(reg.body)
    assert run(bump, ["--lib", "DOMPurify", "--tag", "3.4.99", "--write"], reg) == 1
    assert fragment in capsys.readouterr().out
    assert snapshot(tree) == before


def test_a_registry_answer_that_is_not_an_object_is_refused(bump, tree, capsys):
    reg = registry_for(bump, tree, "DOMPurify", "3.4.99")
    reg.body = ["not", "an", "object"]
    assert run(bump, ["--lib", "DOMPurify", "--tag", "3.4.99", "--write"], reg) == 1
    assert "is not an object" in capsys.readouterr().out


def test_a_tampered_tarball_is_refused_and_nothing_is_edited(bump, tree, capsys):
    before = snapshot(tree)
    reg = registry_for(bump, tree, "DOMPurify", "3.4.99")
    good = reg.tarball
    reg.tarball = make_tarball({"package/dist/purify.min.js": b"alert(1)"})
    assert reg.tarball != good
    assert run(bump, ["--lib", "DOMPurify", "--tag", "3.4.99", "--write"], reg) == 1
    assert "does not match its registry integrity" in capsys.readouterr().out
    assert snapshot(tree) == before


def test_a_tarball_whose_shasum_differs_is_refused(bump, tree, capsys):
    reg = registry_for(bump, tree, "DOMPurify", "3.4.99")
    reg.body["dist"]["shasum"] = "0" * 40
    assert run(bump, ["--lib", "DOMPurify", "--tag", "3.4.99", "--write"], reg) == 1
    assert "does not match its registry shasum" in capsys.readouterr().out


def test_a_tarball_that_is_not_a_tarball_is_refused(bump, tree, capsys):
    reg = Registry(bump, "dompurify", "3.4.99", b"this is not gzip")
    assert run(bump, ["--lib", "DOMPurify", "--tag", "3.4.99", "--write"], reg) == 1
    assert "cannot be opened" in capsys.readouterr().out


def test_a_missing_member_names_what_the_tarball_does_carry(bump, tree, capsys):
    files = {"package/lib/marked.umd.js": b"umd", "package/README.md": b"x"}
    reg = Registry(bump, "marked", "13.0.0", make_tarball(files))
    assert run(bump, ["--lib", "marked", "--tag", "13.0.0", "--write"], reg) == 1
    out = capsys.readouterr().out
    assert "no package/marked.min.js" in out and "package/lib/marked.umd.js" in out


def test_a_duplicated_member_is_refused(bump, tree, capsys):
    info = tarfile.TarInfo("package/dist/purify.min.js")
    info.size = 3
    files = {"package/dist/purify.min.js": b"one"}
    reg = Registry(bump, "dompurify", "3.4.99", make_tarball(files, extra=[(info, b"two")]))
    assert run(bump, ["--lib", "DOMPurify", "--tag", "3.4.99", "--write"], reg) == 1
    assert "appears more than once" in capsys.readouterr().out


def test_a_link_in_place_of_a_file_is_refused(bump, tree, capsys):
    info = tarfile.TarInfo("package/dist/purify.min.js")
    info.type = tarfile.SYMTYPE
    info.linkname = "../../../../etc/passwd"
    reg = Registry(bump, "dompurify", "3.4.99", make_tarball({}, extra=[(info, None)]))
    assert run(bump, ["--lib", "DOMPurify", "--tag", "3.4.99", "--write"], reg) == 1
    assert "is not a regular file" in capsys.readouterr().out


def test_a_font_name_with_a_path_in_it_is_refused(bump, tree, capsys):
    files = release_files(tree, bump, "KaTeX", "0.19.0")
    files["package/dist/fonts/../../evil.woff2"] = b"x"
    reg = Registry(bump, "katex", "0.19.0", make_tarball(files))
    assert run(bump, ["--lib", "KaTeX", "--tag", "0.19.0", "--write"], reg) == 1
    assert "unexpected font member" in capsys.readouterr().out


def test_an_oversized_member_is_refused(bump, tree, capsys, monkeypatch):
    monkeypatch.setattr(bump, "MAX_MEMBER_BYTES", 10)
    reg = registry_for(bump, tree, "DOMPurify", "3.4.99")
    assert run(bump, ["--lib", "DOMPurify", "--tag", "3.4.99", "--write"], reg) == 1
    assert "is larger than 10 bytes" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
#  Refusals: an inconsistent tree                                             #
# --------------------------------------------------------------------------- #

def test_a_pin_that_disagrees_with_the_banner_is_refused(bump, tree, capsys):
    reg = registry_for(bump, tree, "marked", "13.0.0")
    path = tree / bump.TESTS_REL / "vendor-marked.test.mjs"
    path.write_bytes(path.read_bytes().replace(b'VENDORED_VERSION = "12.0.2"',
                                               b'VENDORED_VERSION = "12.0.1"'))
    assert run(bump, ["--lib", "marked", "--tag", "13.0.0", "--write"], reg) == 1
    assert "the tree is inconsistent" in capsys.readouterr().out


def test_vendored_bytes_that_no_longer_match_their_pin_are_refused(bump, tree, capsys):
    before = snapshot(tree)
    reg = registry_for(bump, tree, "marked", "13.0.0")
    path = tree / bump.VENDOR_REL / "marked.min.js"
    path.write_bytes(path.read_bytes() + b"// hand edit\n")
    assert run(bump, ["--lib", "marked", "--tag", "13.0.0", "--write"], reg) == 1
    assert "does not match the hash recorded" in capsys.readouterr().out
    assert snapshot(tree)[f"{bump.TESTS_REL}/vendor-marked.test.mjs"] == \
        before[f"{bump.TESTS_REL}/vendor-marked.test.mjs"]


@pytest.mark.parametrize("lib, test, needle", [
    ("marked", "vendor-marked.test.mjs", b'const PINNED_HASH = "'),
    ("highlight.js", "vendor-highlightjs.test.mjs", b"npm pack @highlightjs/cdn-assets@"),
    ("marked", "vendor-marked.test.mjs", b"dist.shasum ("),
    ("KaTeX", "vendor-katex.test.mjs", b'"auto-render.min.js": "'),
    ("marked", "vendor-marked.test.mjs", b"the banner comment says "),
])
def test_a_region_found_twice_or_not_at_all_refuses_the_edit(bump, tree, capsys, lib, test, needle):
    path = tree / bump.TESTS_REL / test
    original = path.read_bytes()
    new = {"marked": "13.0.0", "highlight.js": "11.13.0", "KaTeX": "0.19.0"}[lib]
    for mutated in (original.replace(needle, b"// gone " + needle[:12] + b"x ", 1),
                    original + b"\n" + next(
                        line for line in original.split(b"\n") if needle in line) + b"\n"):
        path.write_bytes(mutated)
        reg = registry_for(bump, tree, lib, new)
        capsys.readouterr()
        assert run(bump, ["--lib", lib, "--tag", new, "--write"], reg) == 1
        assert "REFUSED" in capsys.readouterr().out
        path.write_bytes(original)


# --------------------------------------------------------------------------- #
#  The checklist                                                              #
# --------------------------------------------------------------------------- #

def test_the_checklist_lists_prose_that_still_names_the_old_version(bump, tree, capsys):
    reg = registry_for(bump, tree, "marked", "13.0.0")
    assert run(bump, ["--lib", "marked", "--tag", "13.0.0"], reg) == 0
    out = capsys.readouterr().out
    assert "lines still naming 12.0.2" in out
    assert "vendor/README.md" in out
    assert "npm ci && npm test" in out
    assert "MAJOR BOUNDARY" in out


# --------------------------------------------------------------------------- #
#  The upstream seam                                                          #
# --------------------------------------------------------------------------- #

class _Resp:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        return self.data if n < 0 else self.data[:n]


def test_fetch_json_and_bytes_refuse_unreadable_and_oversized_answers(bump, monkeypatch):
    with pytest.raises(bump.Refused, match="could not read"):
        bump.fetch_json("https://x/y", opener=lambda req, t: _Resp(b"{not json"))
    def boom(req, t):
        raise OSError("down")
    with pytest.raises(bump.Refused, match="could not download"):
        bump.fetch_bytes("https://x/y", opener=boom)
    monkeypatch.setattr(bump, "MAX_TARBALL_BYTES", 4)
    with pytest.raises(bump.Refused, match="larger than"):
        bump.fetch_bytes("https://x/y", opener=lambda req, t: _Resp(b"123456789"))
    monkeypatch.setattr(bump, "MAX_JSON_BYTES", 4)
    with pytest.raises(bump.Refused, match="could not read"):
        bump.fetch_json("https://x/y", opener=lambda req, t: _Resp(b'{"a": 1234567}'))
    assert bump.fetch_json("https://x/y", opener=lambda req, t: _Resp(b"[1]")) == [1]
