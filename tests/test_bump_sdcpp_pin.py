# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/bump_sdcpp_pin.py: the mechanical half of advancing the stable-diffusion.cpp pin.

Covers the properties that make a scripted bump safe:

  * tag, commit and the whole archive table move together, and nothing else in
    pins.py changes; regenerating today's tables reproduces the shipped file;
  * a write needs a receipt that is a PASS for exactly this tag, not a --current run,
    with every required check PASS and the mandatory cpu checks present;
  * the release API is read through an injected opener (no network here) and a
    malformed or changed answer refuses;
  * every edited region is found exactly once, and no other tracked file may still
    name the old release.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_BUMP = _ROOT / "scripts" / "bump_sdcpp_pin.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bump():
    return _load(_BUMP, "bump_sdcpp_pin")


OLD_SHORT = "aaaaaaa"
OLD_TAG = f"master-100-{OLD_SHORT}"
OLD_COMMIT = OLD_SHORT + "0" * 33
NEW_SHORT = "bbbbbbb"
NEW_TAG = f"master-105-{NEW_SHORT}"
NEW_COMMIT = NEW_SHORT + "1" * 33


def _sha(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def _names(short: str, rocm="7.14.0", mac="26.6.2", cuda="12") -> dict:
    """The archive names a release for *short* publishes, keyed like the pins table."""
    return {
        ("windows", "cpu"): f"sd-master-{short}-bin-win-cpu-x64.zip",
        ("windows", "vulkan"): f"sd-master-{short}-bin-win-vulkan-x64.zip",
        ("windows", "cuda"): f"sd-master-{short}-bin-win-cuda{cuda}-x64.zip",
        ("windows", "rocm"): f"sd-master-{short}-bin-win-rocm-{rocm}-x64.zip",
        ("linux", "cpu"): f"sd-master-{short}-bin-Linux-Ubuntu-24.04-x86_64.zip",
        ("linux", "vulkan"): f"sd-master-{short}-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip",
        ("linux", "rocm"): f"sd-master-{short}-bin-Linux-Ubuntu-24.04-x86_64-rocm-{rocm}.zip",
        ("macos-arm64", "metal"): f"sd-master-{short}-bin-Darwin-macOS-{mac}-arm64.zip",
    }


EXTRA_NAME = "cudart-sd-bin-win-cu12-x64.zip"


def _pins_text(short=OLD_SHORT, tag=OLD_TAG, commit=OLD_COMMIT) -> str:
    lines = ['"""Fixture pins."""', "", "from __future__ import annotations", "",
             'REPO = "o/r"', "", f'TAG = "{tag}"', "", f'COMMIT = "{commit}"', "",
             '_BASE_URL = f"https://example.invalid/{TAG}/"', "",
             "# (platform, backend) -> (asset name, sha256).",
             "ASSETS: dict[tuple[str, str], tuple[str, str]] = {"]
    for (plat, backend), name in _names(short).items():
        lines += [f'    ("{plat}", "{backend}"): (', f'        "{name}",',
                  f'        "{_sha(name)}"),']
    lines += ["}", "", "# Extra archives.",
              "EXTRA_ASSETS: dict[tuple[str, str], list[tuple[str, str]]] = {",
              '    ("windows", "cuda"): [(', f'        "{EXTRA_NAME}",',
              f'        "{_sha(EXTRA_NAME)}")],', "}", "", "",
              "def asset_url(name: str) -> str:", "    return _BASE_URL + name", ""]
    return "\n".join(lines)


def _release(tag=NEW_TAG, short=NEW_SHORT, commit=NEW_COMMIT, **name_kw) -> dict:
    """The API answers for a release: {url: body}."""
    from_names = list(_names(short, **name_kw).values()) + [EXTRA_NAME]
    assets = [{"name": n, "size": 1000 + i, "digest": "sha256:" + _sha(n + tag)}
              for i, n in enumerate(from_names)]
    return {"release": {"tag_name": tag, "target_commitish": commit, "assets": assets},
            "commit": {"sha": commit}}


class _Resp:
    def __init__(self, body):
        self._b = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._b


def _opener(bump, answers: dict, tag=NEW_TAG):
    routes = {bump.RELEASE_URL % (bump.UPSTREAM_REPO, tag): answers["release"],
              bump.COMMIT_URL % (bump.UPSTREAM_REPO, tag): answers["commit"]}

    def opener(req, timeout=30):
        body = routes[req.full_url]
        if isinstance(body, Exception):
            raise body
        return _Resp(body)
    return opener


def _release_view(answers: dict) -> dict:
    return {"tag": answers["release"]["tag_name"], "commit": answers["commit"]["sha"],
            "assets": {a["name"]: {"size": a["size"], "sha256": a["digest"].split(":")[1]}
                       for a in answers["release"]["assets"]}}


MANDATORY = ("isolation", "release_assets", "header_layout", "download_cpu", "abi_cpu",
             "device_cpu", "generate_cpu")


def _receipt(bump, tmp_path: Path, answers=None, tag=NEW_TAG, mutate=None) -> Path:
    answers = answers or _release()
    view = _release_view(answers)
    receipt = {
        "schema": 1, "component": "sdcpp", "tag": tag, "current": False, "verdict": "PASS",
        "why": "every required check passed", "written_at": "2026-10-10T12:00:00Z",
        "hardware": {"platform": "windows", "gpu": True, "backends": ["cpu", "vulkan"]},
        "candidate": {"tag": tag, "commit": view["commit"],
                      "assets": {n: dict(v) for n, v in view["assets"].items()}},
        "checks": {n: {"status": "PASS", "required": True, "detail": "ok"}
                   for n in MANDATORY + ("download_vulkan", "generate_vulkan")},
    }
    if mutate:
        mutate(receipt)
    p = tmp_path / "receipt.json"
    p.write_text(json.dumps(receipt), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
#  Tags                                                                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tag", ["", "b1234", "master-12", "master-x-abcdef0", "master-12-ABCDEF0",
                                 "master-12-abc", "v1.2.3", "master-12-abcdef0-extra"])
def test_a_tag_that_is_not_an_sdcpp_release_tag_is_refused(bump, tag):
    with pytest.raises(bump.Refused, match="not a stable-diffusion.cpp release tag"):
        bump.parse_tag(tag)


def test_parse_tag_returns_number_and_short_commit(bump):
    assert bump.parse_tag("master-123-0a1b2c3") == (123, "0a1b2c3")


@pytest.mark.parametrize("target", ["master-100-bbbbbbb", "master-99-bbbbbbb",
                                    "master-100-aaaaaaa"])
def test_the_pin_only_moves_forward(bump, target):
    with pytest.raises(bump.Refused, match="not newer"):
        bump.forward_only(OLD_TAG, target)


def test_a_higher_build_number_is_forward(bump):
    bump.forward_only(OLD_TAG, "master-101-bbbbbbb")


# --------------------------------------------------------------------------- #
#  Naming rules                                                                #
# --------------------------------------------------------------------------- #

def test_the_real_pins_table_matches_its_own_name_patterns(bump):
    import re
    from localm.media.sdcpp import pins
    short = pins.TAG.rsplit("-", 1)[1]
    assert set(pins.ASSETS) <= set(bump.ASSET_PATTERNS), "a pins key has no name pattern"
    for key, (name, _sha256) in pins.ASSETS.items():
        rx = re.compile(bump.ASSET_PATTERNS[key].format(s=short))
        assert rx.fullmatch(name), f"{name} does not match the pattern for {key}"
    assert set(pins.EXTRA_ASSETS) <= set(bump.EXTRA_PATTERNS)
    for key, entries in pins.EXTRA_ASSETS.items():
        for (name, _s), pat in zip(entries, bump.EXTRA_PATTERNS[key], strict=True):
            assert re.fullmatch(pat, name), f"{name} does not match {pat}"


def test_toolkit_versions_in_names_do_not_change_the_classification(bump):
    names = list(_names(NEW_SHORT, rocm="7.15.1", mac="27.0", cuda="13").values())
    out = bump.classify_assets(NEW_SHORT, names, list(bump.ASSET_PATTERNS), [])
    assert out["missing"] == [] and out["ambiguous"] == []
    assert out["assets"][("windows", "rocm")].endswith("rocm-7.15.1-x64.zip")
    assert out["assets"][("windows", "cuda")].endswith("cuda13-x64.zip")
    assert out["assets"][("macos-arm64", "metal")].endswith("macOS-27.0-arm64.zip")


def test_cpu_and_vulkan_archives_of_one_platform_are_told_apart(bump):
    names = list(_names(NEW_SHORT).values())
    out = bump.classify_assets(NEW_SHORT, names, [("linux", "cpu"), ("linux", "vulkan")], [])
    assert out["assets"][("linux", "cpu")].endswith("x86_64.zip")
    assert out["assets"][("linux", "vulkan")].endswith("x86_64-vulkan.zip")


def test_a_companion_file_with_a_longer_name_is_not_an_archive(bump):
    names = list(_names(NEW_SHORT).values())
    names += [n + ".sig" for n in names] + [n + ".sha256" for n in names]
    out = bump.classify_assets(NEW_SHORT, names, list(bump.ASSET_PATTERNS), [])
    assert out["ambiguous"] == [] and out["missing"] == []
    assert not any(n.endswith((".sig", ".sha256")) for n in out["assets"].values())


def test_two_archives_for_one_key_are_ambiguous(bump):
    names = list(_names(NEW_SHORT).values()) + [f"sd-master-{NEW_SHORT}-bin-win-rocm-7.99.0-x64.zip"]
    out = bump.classify_assets(NEW_SHORT, names, [("windows", "rocm")], [])
    assert out["ambiguous"] and "windows/rocm" in out["ambiguous"][0]


def test_an_archive_of_another_commit_is_not_matched(bump):
    names = list(_names("ccccccc").values())
    out = bump.classify_assets(NEW_SHORT, names, [("windows", "cpu")], [])
    assert out["missing"] == ["windows/cpu"]


def test_a_key_without_a_pattern_refuses(bump):
    with pytest.raises(bump.Refused, match="no name pattern"):
        bump.classify_assets(NEW_SHORT, [], [("plan9", "cpu")], [])
    with pytest.raises(bump.Refused, match="no name pattern"):
        bump.classify_assets(NEW_SHORT, [], [], [("plan9", "cpu")])


# --------------------------------------------------------------------------- #
#  The release API                                                             #
# --------------------------------------------------------------------------- #

def test_fetch_release_returns_commit_sizes_and_digests(bump):
    answers = _release()
    rel = bump.fetch_release(NEW_TAG, _opener(bump, answers))
    assert rel["commit"] == NEW_COMMIT
    name = _names(NEW_SHORT)[("windows", "cpu")]
    assert rel["assets"][name] == {"size": 1000, "sha256": _sha(name + NEW_TAG)}


def _broken(answers, edit):
    edit(answers)
    return answers


@pytest.mark.parametrize("label,edit,match", [
    ("wrong tag", lambda a: a["release"].update(tag_name="master-1-ccccccc"), "not for"),
    ("no assets", lambda a: a["release"].update(assets=[]), "no asset list"),
    ("assets not a list", lambda a: a["release"].update(assets="x"), "no asset list"),
    ("no digest", lambda a: a["release"]["assets"][0].pop("digest"), "no sha256 digest"),
    ("md5 digest", lambda a: a["release"]["assets"][0].update(digest="md5:abc"),
     "no sha256 digest"),
    ("short digest", lambda a: a["release"]["assets"][0].update(digest="sha256:abcd"),
     "malformed digest"),
    ("no size", lambda a: a["release"]["assets"][0].pop("size"), "no usable size"),
    ("zero size", lambda a: a["release"]["assets"][0].update(size=0), "no usable size"),
    ("bool size", lambda a: a["release"]["assets"][0].update(size=True), "no usable size"),
    ("duplicate asset",
     lambda a: a["release"]["assets"].append(dict(a["release"]["assets"][0])), "listed twice"),
    ("commit not hex", lambda a: a["commit"].update(sha="zz"), "40-hex"),
    ("commit not for the tag", lambda a: a["commit"].update(sha="c" * 40),
     "does not start with"),
    ("target differs", lambda a: a["release"].update(target_commitish=NEW_SHORT + "9" * 33),
     "release targets"),
])
def test_a_malformed_release_answer_refuses(bump, label, edit, match):
    answers = _broken(_release(), edit)
    with pytest.raises(bump.Refused, match=match):
        bump.fetch_release(NEW_TAG, _opener(bump, answers))


def test_a_body_that_is_not_an_object_refuses(bump):
    answers = {"release": ["not", "an", "object"], "commit": {"sha": NEW_COMMIT}}
    with pytest.raises(bump.Refused, match="not for"):
        bump.fetch_release(NEW_TAG, _opener(bump, answers))


def test_an_unreachable_api_is_reported_as_unreadable(bump):
    answers = {"release": OSError("rate limited"), "commit": {"sha": NEW_COMMIT}}
    with pytest.raises(bump.UpstreamUnreadable, match="rate limited"):
        bump.fetch_release(NEW_TAG, _opener(bump, answers))


def test_a_branch_name_as_target_commitish_is_tolerated(bump):
    answers = _release()
    answers["release"]["target_commitish"] = "master"
    assert bump.fetch_release(NEW_TAG, _opener(bump, answers))["commit"] == NEW_COMMIT


def test_build_tables_keeps_keys_and_order_and_takes_digests_from_the_listing(bump):
    cur = bump.read_pins(_pins_text())
    rel = _release_view(_release())
    assets, extra = bump.build_tables(rel, cur["assets"], cur["extra"])
    assert list(assets) == list(cur["assets"])
    name = _names(NEW_SHORT)[("linux", "vulkan")]
    assert assets[("linux", "vulkan")] == (name, _sha(name + NEW_TAG))
    assert extra == {("windows", "cuda"): [(EXTRA_NAME, _sha(EXTRA_NAME + NEW_TAG))]}


def test_a_release_that_drops_an_archive_is_incomplete(bump):
    answers = _release()
    answers["release"]["assets"] = [a for a in answers["release"]["assets"]
                                    if "rocm" not in a["name"]]
    cur = bump.read_pins(_pins_text())
    with pytest.raises(bump.IncompleteRelease, match="windows/rocm"):
        bump.build_tables(_release_view(answers), cur["assets"], cur["extra"])


def test_a_release_without_the_extra_archive_is_incomplete(bump):
    answers = _release()
    answers["release"]["assets"] = [a for a in answers["release"]["assets"]
                                    if not a["name"].startswith("cudart")]
    cur = bump.read_pins(_pins_text())
    with pytest.raises(bump.IncompleteRelease, match="extra"):
        bump.build_tables(_release_view(answers), cur["assets"], cur["extra"])


def test_an_ambiguous_release_refuses_to_guess(bump):
    answers = _release()
    answers["release"]["assets"].append(
        {"name": f"sd-master-{NEW_SHORT}-bin-win-rocm-9.9.9-x64.zip", "size": 5,
         "digest": "sha256:" + "d" * 64})
    cur = bump.read_pins(_pins_text())
    with pytest.raises(bump.Refused, match="more than one archive"):
        bump.build_tables(_release_view(answers), cur["assets"], cur["extra"])


# --------------------------------------------------------------------------- #
#  The rewrite                                                                 #
# --------------------------------------------------------------------------- #

def _new_tables(bump):
    cur = bump.read_pins(_pins_text())
    return bump.build_tables(_release_view(_release()), cur["assets"], cur["extra"])


def test_rewrite_moves_tag_commit_and_every_archive_and_nothing_else(bump):
    text = _pins_text()
    assets, extra = _new_tables(bump)
    new = bump.rewrite(text, NEW_TAG, NEW_COMMIT, assets, extra)
    parsed = bump.read_pins(new)
    assert parsed["tag"] == NEW_TAG and parsed["commit"] == NEW_COMMIT
    assert parsed["assets"] == assets and parsed["extra"] == extra
    assert OLD_SHORT not in new and OLD_COMMIT not in new
    old_lines, new_lines = text.split("\n"), new.split("\n")
    assert len(old_lines) == len(new_lines)
    changed = [o for o, n in zip(old_lines, new_lines, strict=True) if o != n]
    assert all(("TAG" in o or "COMMIT" in o or o.strip().startswith('"')) for o in changed)
    assert '_BASE_URL = f"https://example.invalid/{TAG}/"' in new
    assert "# (platform, backend) -> (asset name, sha256)." in new
    assert "# Extra archives." in new
    assert new.endswith("return _BASE_URL + name\n")


def test_the_real_pins_file_is_reproduced_byte_for_byte_by_the_formatter(bump):
    text = (_ROOT / "localm" / "media" / "sdcpp" / "pins.py").read_bytes().decode("utf-8")
    text = text.replace("\r\n", "\n")
    cur = bump.read_pins(text)
    assert bump.rewrite(text, cur["tag"], cur["commit"], cur["assets"], cur["extra"]) == text


def test_several_extra_archives_for_one_key_render_and_parse(bump):
    extra = {("windows", "cuda"): [("a-cu12.zip", "1" * 64), ("b-cu12.zip", "2" * 64)],
             ("linux", "cuda"): [("c.zip", "3" * 64)]}
    import ast
    assert ast.literal_eval(bump.render_extra(extra).split("= ", 1)[1]) == extra


def test_a_crlf_file_keeps_its_line_endings(bump, tmp_path):
    path = tmp_path / "pins.py"
    path.write_bytes(_pins_text().replace("\n", "\r\n").encode())
    text, newline = bump._read(path)
    assert newline == "\r\n" and "\r" not in text
    assets, extra = _new_tables(bump)
    bump._write(path, bump.rewrite(text, NEW_TAG, NEW_COMMIT, assets, extra), newline)
    raw = path.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n") and NEW_TAG.encode() in raw


@pytest.mark.parametrize("label,edit", [
    ("TAG missing", lambda t: t.replace(f'TAG = "{OLD_TAG}"', "")),
    ("TAG twice", lambda t: t + f'\nTAG = "{OLD_TAG}"\n'),
    ("COMMIT missing", lambda t: t.replace(f'COMMIT = "{OLD_COMMIT}"', "")),
    ("ASSETS missing", lambda t: t.replace("ASSETS: dict[tuple[str, str], tuple[str, str]] =",
                                           "ASSETZ: dict[tuple[str, str], tuple[str, str]] =")),
    ("EXTRA_ASSETS twice", lambda t: t + "\nEXTRA_ASSETS = {}\n"),
    ("TAG not a plain string", lambda t: t.replace(f'TAG = "{OLD_TAG}"', 'TAG = "a" + "b"')),
])
def test_an_edited_region_that_is_not_found_exactly_once_refuses(bump, label, edit):
    assets, extra = _new_tables(bump)
    with pytest.raises(bump.Refused):
        bump.rewrite(edit(_pins_text()), NEW_TAG, NEW_COMMIT, assets, extra)


def test_a_commit_that_is_not_40_hex_refuses(bump):
    assets, extra = _new_tables(bump)
    with pytest.raises(bump.Refused, match="40-character"):
        bump.rewrite(_pins_text(), NEW_TAG, "abc", assets, extra)


def test_read_pins_refuses_a_file_that_does_not_parse(bump):
    with pytest.raises(bump.Refused, match="does not parse"):
        bump.read_pins("TAG = (")


# --------------------------------------------------------------------------- #
#  The receipt                                                                 #
# --------------------------------------------------------------------------- #

def test_a_good_receipt_loads(bump, tmp_path):
    rec = bump.load_receipt(_receipt(bump, tmp_path), NEW_TAG)
    assert rec["verdict"] == "PASS"


def _set(**changes):
    return lambda r: r.update(changes)


def _check(name, **changes):
    return lambda r: r["checks"][name].update(changes)


def _drop(name):
    return lambda r: r["checks"].pop(name)


@pytest.mark.parametrize("label,mutate,match", [
    ("wrong tag", _set(tag="master-104-bbbbbbb"), "is for"),
    ("a --current receipt", _set(current=True), "--current run"),
    ("current missing", lambda r: r.pop("current"), "--current run"),
    ("verdict FAIL", _set(verdict="FAIL"), "not PASS"),
    ("verdict INCONCLUSIVE", _set(verdict="INCONCLUSIVE"), "not PASS"),
    ("wrong component", _set(component="llama"), "not a schema"),
    ("wrong schema", _set(schema=2), "not a schema"),
    ("no timestamp", lambda r: r.pop("written_at"), "written_at"),
    ("bad timestamp", _set(written_at="yesterday"), "written_at"),
    ("no checks", _set(checks=None), "no checks"),
    ("required check FAILED but verdict says PASS", _check("abi_cpu", status="FAIL"),
     "abi_cpu"),
    ("generation SKIP (not measured)", _check("generate_cpu", status="SKIP"),
     "generate_cpu: SKIP"),
    ("mandatory check missing", _drop("header_layout"), "header_layout: not in the receipt"),
    ("mandatory check demoted to optional", _check("device_cpu", required=False),
     "device_cpu"),
    ("a non-mandatory required check FAILED", _check("generate_vulkan", status="FAIL"),
     "generate_vulkan"),
    ("cpu not among the backends", lambda r: r["hardware"].update(backends=["vulkan"]),
     "cpu among"),
    ("a GPU box that ran only cpu", lambda r: r["hardware"].update(backends=["cpu"]),
     "GPU backend was not measured"),
    ("no candidate", lambda r: r.pop("candidate"), "no candidate commit"),
    ("candidate commit malformed", lambda r: r["candidate"].update(commit="abc"),
     "no candidate commit"),
])
def test_a_receipt_that_does_not_prove_the_candidate_refuses(bump, tmp_path, label, mutate,
                                                             match):
    path = _receipt(bump, tmp_path, mutate=mutate)
    with pytest.raises(bump.Refused, match=match):
        bump.load_receipt(path, NEW_TAG)


def test_a_receipt_must_report_pass_for_a_check_named_by_require(bump, tmp_path):
    path = _receipt(bump, tmp_path)
    bump.load_receipt(path, NEW_TAG, ("generate_vulkan",))
    with pytest.raises(bump.Refused, match="download_rocm: not in the receipt"):
        bump.load_receipt(path, NEW_TAG, ("download_rocm",))


def test_an_unreadable_or_non_object_receipt_refuses(bump, tmp_path):
    with pytest.raises(bump.Refused, match="could not read the receipt"):
        bump.load_receipt(tmp_path / "missing.json", NEW_TAG)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(bump.Refused, match="could not read the receipt"):
        bump.load_receipt(bad, NEW_TAG)
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(bump.Refused, match="not a JSON object"):
        bump.load_receipt(bad, NEW_TAG)


def test_the_receipt_must_equal_the_release_as_it_is_now(bump, tmp_path):
    rec = bump.load_receipt(_receipt(bump, tmp_path), NEW_TAG)
    rel = _release_view(_release())
    bump.compare_with_receipt(rec, rel)

    moved = dict(rel, commit="b" * 7 + "2" * 33)
    with pytest.raises(bump.Refused, match="now resolves to"):
        bump.compare_with_receipt(rec, moved)

    name = next(iter(rel["assets"]))
    reuploaded = dict(rel, assets={**rel["assets"], name: {**rel["assets"][name],
                                                            "sha256": "e" * 64}})
    with pytest.raises(bump.Refused, match="changed after it was confirmed"):
        bump.compare_with_receipt(rec, reuploaded)

    resized = dict(rel, assets={**rel["assets"], name: {**rel["assets"][name], "size": 1}})
    with pytest.raises(bump.Refused, match="changed after it was confirmed"):
        bump.compare_with_receipt(rec, resized)

    gone = dict(rel, assets={k: v for k, v in rel["assets"].items() if k != name})
    with pytest.raises(bump.Refused, match="gone from the release"):
        bump.compare_with_receipt(rec, gone)


# --------------------------------------------------------------------------- #
#  Nothing else may still name the old release                                 #
# --------------------------------------------------------------------------- #

def _tree(tmp_path: Path, extra_files: dict | None = None) -> Path:
    root = tmp_path / "repo"
    pins = root / "localm" / "media" / "sdcpp"
    pins.mkdir(parents=True)
    (pins / "pins.py").write_text(_pins_text(), encoding="utf-8")
    for rel, body in (extra_files or {}).items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body, encoding="utf-8")
    return root


def test_stale_references_to_the_tag_commit_and_short_commit_are_found(bump, tmp_path):
    root = _tree(tmp_path, {
        "tests/test_a.py": f'X = "{OLD_TAG}"\n',
        "tests/test_b.py": f'Y = "{OLD_COMMIT}"\n',
        "docs/c.md": f"built from {OLD_SHORT.upper()} in october\nsecond line\n",
        "tests/test_clean.py": "Z = 1\n",
    })
    hits = bump.find_stale_references(root, OLD_TAG, OLD_COMMIT)
    assert hits == ["docs/c.md:1", "tests/test_a.py:1", "tests/test_b.py:1"]


def test_pins_py_the_changelogs_and_a_longer_hex_run_are_not_stale_references(bump, tmp_path):
    root = _tree(tmp_path, {
        "CHANGELOG.md": f"- moved to {OLD_TAG}\n",
        "dev/CHANGELOG-FULL.md": f"- {OLD_TAG}\n",
        "tests/test_digest.py": f'D = "{"1" * 5}{OLD_SHORT}{"2" * 5}"\n',
        "tests/blob.bin": f"{OLD_TAG}\n",
    })
    assert bump.find_stale_references(root, OLD_TAG, OLD_COMMIT) == []


def test_the_skip_list_exempts_a_named_file(bump, tmp_path):
    root = _tree(tmp_path, {"tests/test_a.py": f'X = "{OLD_TAG}"\n'})
    assert bump.find_stale_references(root, OLD_TAG, OLD_COMMIT,
                                      skip=("tests/test_a.py",)) == []


def test_no_tracked_file_other_than_pins_names_the_pinned_release(bump):
    from localm.media.sdcpp import pins
    assert bump.find_stale_references(_ROOT, pins.TAG, pins.COMMIT) == []


# --------------------------------------------------------------------------- #
#  main()                                                                      #
# --------------------------------------------------------------------------- #

@pytest.fixture
def tree(bump, tmp_path, monkeypatch):
    root = _tree(tmp_path)
    monkeypatch.setattr(bump, "REPO", root)
    monkeypatch.setattr(bump, "PINS_PATH", root / "localm" / "media" / "sdcpp" / "pins.py")
    return root


def _run(bump, capsys, argv, answers=None, tag=NEW_TAG):
    opener = _opener(bump, answers or _release(), tag)
    rc = bump.main(argv, opener=opener)
    return rc, capsys.readouterr().out


def test_a_dry_run_prints_the_diff_and_changes_nothing(bump, tree, tmp_path, capsys):
    before = bump.PINS_PATH.read_bytes()
    receipt = _receipt(bump, tmp_path)
    rc, out = _run(bump, capsys, ["--tag", NEW_TAG, "--receipt", str(receipt)])
    assert rc == 0
    assert f'-TAG = "{OLD_TAG}"' in out and f'+TAG = "{NEW_TAG}"' in out
    assert "dry run: nothing written" in out and "REMAINING STEPS" in out
    assert bump.PINS_PATH.read_bytes() == before


def test_write_applies_the_edit(bump, tree, tmp_path, capsys):
    receipt = _receipt(bump, tmp_path)
    rc, out = _run(bump, capsys, ["--tag", NEW_TAG, "--receipt", str(receipt), "--write"])
    assert rc == 0 and "wrote localm/media/sdcpp/pins.py" in out
    parsed = bump.read_pins(bump.PINS_PATH.read_text(encoding="utf-8"))
    assert parsed["tag"] == NEW_TAG and parsed["commit"] == NEW_COMMIT
    name = _names(NEW_SHORT)[("windows", "vulkan")]
    assert parsed["assets"][("windows", "vulkan")] == (name, _sha(name + NEW_TAG))


def _refused(bump, tree, capsys, argv, answers=None, match=""):
    before = bump.PINS_PATH.read_bytes()
    rc, out = _run(bump, capsys, argv, answers)
    assert rc == 1 and "REFUSED:" in out
    assert match in out, out
    assert bump.PINS_PATH.read_bytes() == before, "a refusal must not touch the file"


def test_write_without_a_receipt_refuses(bump, tree, capsys):
    _refused(bump, tree, capsys, ["--tag", NEW_TAG, "--write"], match="--write needs --receipt")


def test_a_same_or_older_tag_refuses_before_any_network(bump, tree, tmp_path, capsys):
    def no_network(req, timeout=30):
        raise AssertionError("the API must not be read for a non-forward tag")
    rc = bump.main(["--tag", OLD_TAG], opener=no_network)
    out = capsys.readouterr().out
    assert rc == 1 and "not newer" in out


def test_a_receipt_for_another_tag_refuses(bump, tree, tmp_path, capsys):
    receipt = _receipt(bump, tmp_path, tag="master-104-bbbbbbb")
    _refused(bump, tree, capsys, ["--tag", NEW_TAG, "--receipt", str(receipt), "--write"],
             match="is for")


def test_a_release_that_changed_after_the_confirm_refuses(bump, tree, tmp_path, capsys):
    receipt = _receipt(bump, tmp_path)
    changed = _release()
    changed["release"]["assets"][0]["digest"] = "sha256:" + "f" * 64
    _refused(bump, tree, capsys, ["--tag", NEW_TAG, "--receipt", str(receipt), "--write"],
             answers=changed, match="changed after it was confirmed")


def test_a_release_with_a_missing_archive_refuses(bump, tree, tmp_path, capsys):
    answers = _release()
    answers["release"]["assets"] = [a for a in answers["release"]["assets"]
                                    if "Darwin" not in a["name"]]
    receipt = _receipt(bump, tmp_path, answers=answers)
    _refused(bump, tree, capsys, ["--tag", NEW_TAG, "--receipt", str(receipt), "--write"],
             answers=answers, match="macos-arm64/metal")


def test_an_api_failure_refuses(bump, tree, tmp_path, capsys):
    receipt = _receipt(bump, tmp_path)
    answers = {"release": OSError("403 rate limit"), "commit": {"sha": NEW_COMMIT}}
    _refused(bump, tree, capsys, ["--tag", NEW_TAG, "--receipt", str(receipt), "--write"],
             answers=answers, match="403 rate limit")


def test_a_file_that_still_names_the_old_release_blocks_the_write_but_not_the_dry_run(
        bump, tree, tmp_path, capsys):
    (tree / "tests").mkdir()
    (tree / "tests" / "test_old.py").write_text(f'X = "{OLD_TAG}"\n', encoding="utf-8")
    receipt = _receipt(bump, tmp_path)
    rc, out = _run(bump, capsys, ["--tag", NEW_TAG, "--receipt", str(receipt)])
    assert rc == 0 and "WARNING" in out and "tests/test_old.py:1" in out
    _refused(bump, tree, capsys, ["--tag", NEW_TAG, "--receipt", str(receipt), "--write"],
             match="tests/test_old.py:1")


def test_a_reshaped_pins_file_refuses(bump, tree, tmp_path, capsys):
    path = bump.PINS_PATH
    path.write_text(path.read_text(encoding="utf-8").replace("COMMIT = ", "COMMITS = "),
                    encoding="utf-8")
    receipt = _receipt(bump, tmp_path)
    _refused(bump, tree, capsys, ["--tag", NEW_TAG, "--receipt", str(receipt), "--write"],
             match="COMMIT")
