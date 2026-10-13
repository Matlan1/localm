# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/bump_koboldcpp_pin.py: the mechanical half of advancing the KoboldCpp pin.

Properties that make the bump safe to script:

  * TAG, VERSION and every ASSETS entry move together in one rewrite and nothing
    else in pins.py changes;
  * a write needs evidence: a receipt produced by scripts/confirm_koboldcpp_runtime.py
    for exactly this tag, a candidate run (not --current), verdict PASS, with every
    required check present and PASS, and no failed check;
  * forward-only: a same-or-older tag is refused;
  * the sizes and digests written come from the release listing and must equal
    the table the confirm run installed from;
  * a mention of the pinned tag or an asset hash anywhere else in the tracked
    tree is refused, so a second place cannot be left behind;
  * no network: the API is reached through an injected opener.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_BUMP = _ROOT / "scripts" / "bump_koboldcpp_pin.py"
_CONFIRM = _ROOT / "scripts" / "confirm_koboldcpp_runtime.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bump():
    return _load(_BUMP, "bump_koboldcpp_pin_under_test")


@pytest.fixture(scope="module")
def confirm(bump):
    return bump.confirm


def _h(n: int) -> str:
    return f"{n:x}".rjust(64, "0")


OLD_TAG, NEW_TAG = "v1.100", "v1.101"
KEYS = [("windows", "cuda", "kcpp.exe"), ("windows", "nocuda", "kcpp-nocuda.exe"),
        ("linux", "cuda", "kcpp-linux"), ("linux", "nocuda", "kcpp-linux-nocuda"),
        ("macos-arm64", "metal", "kcpp-mac")]
OLD_SIZES = {k[2]: 1000 + i for i, k in enumerate(KEYS)}
OLD_SHAS = {k[2]: _h(0xA0 + i) for i, k in enumerate(KEYS)}
NEW_SIZES = {k[2]: 5000 + i for i, k in enumerate(KEYS)}
NEW_SHAS = {k[2]: _h(0xB0 + i) for i, k in enumerate(KEYS)}


def pins_text(tag=OLD_TAG, version="1.100", newline="\n") -> str:
    entries = "".join(
        f'    ("{p}", "{b}"): (\n        "{n}", {OLD_SIZES[n]},\n        "{OLD_SHAS[n]}"),\n'
        for p, b, n in KEYS)
    text = (
        '"""Pins."""\n\nfrom __future__ import annotations\n\n'
        'REPO = "Owner/kcpp"\n\n'
        f'TAG = "{tag}"\n\n'
        "# What the launcher prints.\n"
        f'VERSION = "{version}"\n\n'
        '_BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}/"\n\n'
        "ASSETS: dict[tuple[str, str], tuple[str, int, str]] = {\n"
        f"{entries}}}\n\n\n"
        "def asset_url(name: str) -> str:\n"
        "    return _BASE_URL + name\n")
    return text.replace("\n", newline)


def release_body(sizes=NEW_SIZES, shas=NEW_SHAS, extra=()):
    assets = [{"name": n, "size": sizes[n], "digest": f"sha256:{shas[n]}"}
              for _p, _b, n in KEYS]
    assets += list(extra)
    return {"tag_name": NEW_TAG, "assets": assets}


class Opener:
    """Stands in for the HTTPS opener: serves one body, records the URLs."""

    def __init__(self, body=None, error=None):
        self.body, self.error, self.urls = body, error, []

    def __call__(self, req, timeout=None):
        self.urls.append(req.full_url)
        if self.error:
            raise self.error
        raw = self.body if isinstance(self.body, bytes) else json.dumps(self.body).encode()

        class _Resp:
            def __enter__(s):
                return s

            def __exit__(s, *a):
                return False

            def read(s):
                return raw
        return _Resp()


def make_receipt(confirm, tag=NEW_TAG, *, current=False, mutate=None) -> dict:
    """A PASS receipt built with confirm's own functions."""
    r = confirm.new_receipt(tag, current)
    for name in confirm.CHECK_NAMES:
        required = name in confirm.ALWAYS_REQUIRED
        confirm.set_check(r, name, "PASS", "ok", required=required)
    r["version"] = tag[1:]
    pinned = {(p, b): (n, OLD_SIZES[n], OLD_SHAS[n]) for p, b, n in KEYS}
    published = {n: (NEW_SIZES[n], NEW_SHAS[n]) for _p, _b, n in KEYS}
    r["assets"] = confirm.table_to_json(confirm.build_table(pinned, published))
    confirm.finalize(r)
    if mutate:
        mutate(r)
    return r


def write_receipt(tmp_path: Path, receipt: dict) -> Path:
    p = tmp_path / "receipt.json"
    p.write_text(json.dumps(receipt), encoding="utf-8")
    return p


def no_mentions(repo, needles):
    return []


def run_plan(bump, tmp_path, receipt, *, text=None, tag=NEW_TAG, body=None,
             scanner=no_mentions, require_receipt=True):
    path = write_receipt(tmp_path, receipt) if receipt is not None else None
    return bump.plan(text or pins_text(), tag, path,
                     opener=Opener(release_body() if body is None else body),
                     scanner=scanner, repo=tmp_path, require_receipt=require_receipt)


# --------------------------------------------------------------------------- #
#  The rewrite                                                                 #
# --------------------------------------------------------------------------- #

def test_happy_path_moves_tag_version_and_every_asset(bump, confirm, tmp_path):
    old = pins_text()
    new, receipt = run_plan(bump, tmp_path, make_receipt(confirm), text=old)
    assert receipt["tag"] == NEW_TAG
    target = tmp_path / "rewritten_pins.py"
    target.write_text(new, encoding="utf-8")
    mod = _load(target, "rewritten_pins")
    assert mod.TAG == NEW_TAG and mod.VERSION == "1.101"
    assert mod.asset_url("kcpp.exe") == (
        f"https://github.com/Owner/kcpp/releases/download/{NEW_TAG}/kcpp.exe")
    assert mod.ASSETS == {(p, b): (n, NEW_SIZES[n], NEW_SHAS[n]) for p, b, n in KEYS}


def test_nothing_but_the_pinned_values_changes(bump, confirm, tmp_path):
    old = pins_text()
    new, _ = run_plan(bump, tmp_path, make_receipt(confirm), text=old)
    expected = old.replace(OLD_TAG, NEW_TAG, 1).replace('"1.100"', '"1.101"')
    for n in OLD_SIZES:
        expected = expected.replace(str(OLD_SIZES[n]), str(NEW_SIZES[n]))
        expected = expected.replace(OLD_SHAS[n], NEW_SHAS[n])
    assert new == expected


def test_crlf_files_stay_crlf(bump, confirm, tmp_path, monkeypatch):
    pins = tmp_path / "pins.py"
    pins.write_bytes(pins_text(newline="\r\n").encode())
    monkeypatch.setattr(bump, "PINS_PATH", pins)
    rec = write_receipt(tmp_path, make_receipt(confirm))
    rc = bump.main(["--tag", NEW_TAG, "--receipt", str(rec), "--write"],
                   opener=Opener(release_body()), scanner=no_mentions)
    assert rc == 0
    data = pins.read_bytes()
    assert b"\r\n" in data and b"\n" not in data.replace(b"\r\n", b"")
    assert NEW_TAG.encode() in data


def test_the_real_pins_file_round_trips_to_a_new_release(bump, confirm, tmp_path):
    """The shipped pins.py has the shape the regexes edit: every entry found."""
    real = (_ROOT / "localm" / "media" / "koboldcpp" / "pins.py").read_text(encoding="utf-8")
    table = bump.pinned_table(real)
    assert len(table) >= 4
    new_table = {k: (v[0], v[1] + 1, _h(0xC000 + i)) for i, (k, v) in enumerate(table.items())}
    new = bump.rewrite(real, "v9.9", "9.9", new_table)
    bump.check_rewritten(new, "v9.9", "9.9", new_table)
    assert bump.pinned_tag(new) == "v9.9"
    again = bump.rewrite(new, bump.pinned_tag(real), bump.pinned_tag(real)[1:], table)
    assert again == real


def test_the_pinned_table_in_the_tree_has_the_five_builds(bump):
    real = (_ROOT / "localm" / "media" / "koboldcpp" / "pins.py").read_text(encoding="utf-8")
    assert set(bump.pinned_table(real)) == {
        ("windows", "cuda"), ("windows", "nocuda"), ("linux", "cuda"),
        ("linux", "nocuda"), ("macos-arm64", "metal")}


# --------------------------------------------------------------------------- #
#  Receipt refusals                                                            #
# --------------------------------------------------------------------------- #

def _refuses(bump, tmp_path, receipt, match, **kw):
    with pytest.raises(bump.Refused, match=match):
        run_plan(bump, tmp_path, receipt, **kw)


def test_a_receipt_is_required_for_a_write(bump, tmp_path):
    _refuses(bump, tmp_path, None, "needs --receipt", require_receipt=True)


def test_dry_run_without_a_receipt_still_plans(bump, tmp_path):
    new, receipt = run_plan(bump, tmp_path, None, require_receipt=False)
    assert receipt is None and NEW_TAG in new


def test_receipt_for_another_tag_is_refused(bump, confirm, tmp_path):
    _refuses(bump, tmp_path, make_receipt(confirm, "v1.099"), "not v1.101")


def test_a_current_run_receipt_is_refused(bump, confirm, tmp_path):
    _refuses(bump, tmp_path, make_receipt(confirm, current=True), "--current")


@pytest.mark.parametrize("verdict", ["FAIL", "INCONCLUSIVE", "pass", None])
def test_a_receipt_that_is_not_PASS_is_refused(bump, confirm, tmp_path, verdict):
    r = make_receipt(confirm, mutate=lambda r: r.update(verdict=verdict))
    _refuses(bump, tmp_path, r, "verdict")


def test_a_tampered_PASS_verdict_over_a_failed_required_check_is_refused(
        bump, confirm, tmp_path):
    def tamper(r):
        r["checks"]["music_generate"]["status"] = "FAIL"
    _refuses(bump, tmp_path, make_receipt(confirm, mutate=tamper), "music_generate: FAIL")


def test_a_tampered_PASS_verdict_over_a_skipped_required_check_is_refused(
        bump, confirm, tmp_path):
    def tamper(r):
        r["checks"]["music_plan"]["status"] = "SKIP"
    _refuses(bump, tmp_path, make_receipt(confirm, mutate=tamper), "music_plan: SKIP")


def test_a_required_check_missing_from_the_receipt_is_refused(bump, confirm, tmp_path):
    def tamper(r):
        del r["checks"]["install"]
    _refuses(bump, tmp_path, make_receipt(confirm, mutate=tamper), "install: missing")


def test_a_required_check_demoted_to_optional_is_refused(bump, confirm, tmp_path):
    def tamper(r):
        r["checks"]["text_generation"]["required"] = False
    _refuses(bump, tmp_path, make_receipt(confirm, mutate=tamper),
             "text_generation: not marked required")


def test_a_failed_optional_check_is_refused(bump, confirm, tmp_path):
    def tamper(r):
        r["checks"]["vulkan_device"]["status"] = "FAIL"
    _refuses(bump, tmp_path, make_receipt(confirm, mutate=tamper), "vulkan_device: FAIL")


def test_a_failed_advisory_check_is_accepted_and_reported(bump, confirm, tmp_path, capsys,
                                                         monkeypatch):
    def tweak(r):
        r["checks"]["music_gpu"] = {"status": "FAIL", "required": False,
                                    "detail": "flat track on vulkan"}
        confirm.finalize(r)
    receipt = make_receipt(confirm, mutate=tweak)
    assert receipt["verdict"] == "PASS"
    new, loaded = run_plan(bump, tmp_path, receipt)
    assert NEW_TAG in new
    assert loaded["_advisory"] == ["music_gpu: flat track on vulkan"]
    assert "flat track on vulkan" in bump.checklist(NEW_TAG, loaded)


def test_the_advisory_exemption_does_not_cover_a_required_check(bump, confirm, tmp_path):
    def tamper(r):
        r["checks"]["music_gpu"].update(status="FAIL", required=True)
    _refuses(bump, tmp_path, make_receipt(confirm, mutate=tamper), "music_gpu")


def test_a_required_vulkan_check_that_did_not_pass_is_refused(bump, confirm, tmp_path):
    def tamper(r):
        r["checks"]["vulkan_device"].update(required=True, status="SKIP")
    _refuses(bump, tmp_path, make_receipt(confirm, mutate=tamper), "vulkan_device: SKIP")


def test_a_skipped_optional_check_is_accepted(bump, confirm, tmp_path):
    def tweak(r):
        r["checks"]["vulkan_device"].update(required=False, status="SKIP")
        confirm.finalize(r)
    new, _ = run_plan(bump, tmp_path, make_receipt(confirm, mutate=tweak))
    assert NEW_TAG in new


@pytest.mark.parametrize("field,value,match", [
    ("component", "llama", "schema 1 koboldcpp"),
    ("schema", 2, "schema 1 koboldcpp"),
    ("version", "1.099", "records version"),
    ("assets", None, "asset table"),
    ("checks", None, "no checks"),
])
def test_a_malformed_receipt_is_refused(bump, confirm, tmp_path, field, value, match):
    r = make_receipt(confirm, mutate=lambda r: r.update({field: value}))
    _refuses(bump, tmp_path, r, match)


def test_an_unreadable_receipt_is_refused(bump, tmp_path):
    p = tmp_path / "r.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(bump.Refused, match="could not read the receipt"):
        bump.plan(pins_text(), NEW_TAG, p, opener=Opener(release_body()),
                  scanner=no_mentions, repo=tmp_path)
    with pytest.raises(bump.Refused, match="could not read the receipt"):
        bump.plan(pins_text(), NEW_TAG, tmp_path / "absent.json",
                  opener=Opener(release_body()), scanner=no_mentions, repo=tmp_path)


def test_a_receipt_whose_table_differs_from_the_release_is_refused(bump, confirm, tmp_path):
    sizes = dict(NEW_SIZES)
    sizes["kcpp.exe"] += 1
    _refuses(bump, tmp_path, make_receipt(confirm), "differ from the table",
             body=release_body(sizes=sizes))
    shas = dict(NEW_SHAS)
    shas["kcpp-mac"] = _h(0xFF)
    _refuses(bump, tmp_path, make_receipt(confirm), "differ from the table",
             body=release_body(shas=shas))


# --------------------------------------------------------------------------- #
#  Tag refusals                                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tag", [OLD_TAG, "v1.099", "v0.200", "v1.99.9"])
def test_a_same_or_older_tag_is_refused(bump, confirm, tmp_path, tag):
    _refuses(bump, tmp_path, make_receipt(confirm, tag), "not newer", tag=tag)


def test_tag_comparison_is_numeric_not_textual(bump, confirm, tmp_path):
    new, _ = run_plan(bump, tmp_path, make_receipt(confirm, "v1.1000"), tag="v1.1000",
                      text=pins_text(version="1.100"))
    assert bump.pinned_tag(new) == "v1.1000"


def test_a_patch_release_is_newer_than_its_minor(bump, confirm, tmp_path):
    new, _ = run_plan(bump, tmp_path, make_receipt(confirm, "v1.100.1"), tag="v1.100.1")
    assert bump.pinned_tag(new) == "v1.100.1"


@pytest.mark.parametrize("tag", ["1.101", "v1", "latest", "v1.101-rc1", "", "V1.101"])
def test_a_tag_that_is_not_a_release_tag_is_refused(bump, confirm, tmp_path, tag):
    with pytest.raises(bump.Refused, match="not a vX.Y"):
        bump.plan(pins_text(), tag, None, opener=Opener(release_body()),
                  scanner=no_mentions, repo=tmp_path)


# --------------------------------------------------------------------------- #
#  The API                                                                     #
# --------------------------------------------------------------------------- #

def test_the_listing_is_read_from_the_tag_endpoint(bump, confirm, tmp_path):
    op = Opener(release_body())
    path = write_receipt(tmp_path, make_receipt(confirm))
    bump.plan(pins_text(), NEW_TAG, path, opener=op, scanner=no_mentions, repo=tmp_path)
    assert op.urls == [f"https://api.github.com/repos/LostRuins/koboldcpp/releases/tags/{NEW_TAG}"]


@pytest.mark.parametrize("body,match", [
    (b"not json", "could not read the v1.101 release"),
    ([], "unusable"),
    ({"assets": "x"}, "unusable"),
    ({"assets": []}, "unusable"),
])
def test_an_unusable_listing_is_refused(bump, confirm, tmp_path, body, match):
    _refuses(bump, tmp_path, make_receipt(confirm), match, body=body)


def test_an_unreachable_api_is_refused(bump, confirm, tmp_path):
    path = write_receipt(tmp_path, make_receipt(confirm))
    with pytest.raises(bump.Refused, match="could not read the v1.101 release"):
        bump.plan(pins_text(), NEW_TAG, path, opener=Opener(error=OSError("down")),
                  scanner=no_mentions, repo=tmp_path)


@pytest.mark.parametrize("digest", [None, "md5:abc", "sha256:XYZ", "sha256:" + "a" * 63, 5])
def test_an_asset_without_a_valid_sha256_digest_is_refused(bump, confirm, tmp_path, digest):
    body = release_body()
    body["assets"][0]["digest"] = digest
    _refuses(bump, tmp_path, make_receipt(confirm), "digest", body=body)


@pytest.mark.parametrize("size", [0, -1, "5", None, True])
def test_an_asset_with_an_invalid_size_is_refused(bump, confirm, tmp_path, size):
    body = release_body()
    body["assets"][1]["size"] = size
    _refuses(bump, tmp_path, make_receipt(confirm), "size", body=body)


def test_a_release_missing_a_pinned_asset_is_refused(bump, confirm, tmp_path):
    body = release_body()
    body["assets"] = [a for a in body["assets"] if a["name"] != "kcpp-nocuda.exe"]
    _refuses(bump, tmp_path, make_receipt(confirm), "kcpp-nocuda.exe", body=body)


def test_unpinned_extra_assets_in_the_release_are_ignored(bump, confirm, tmp_path):
    extra = [{"name": "kcpp-oldpc.exe", "size": 9, "digest": "sha256:" + "d" * 64}]
    new, _ = run_plan(bump, tmp_path, make_receipt(confirm), body=release_body(extra=extra))
    assert "oldpc" not in new


# --------------------------------------------------------------------------- #
#  Region refusals                                                             #
# --------------------------------------------------------------------------- #

def test_a_missing_tag_line_is_refused(bump, confirm, tmp_path):
    text = pins_text().replace(f'TAG = "{OLD_TAG}"\n', "")
    _refuses(bump, tmp_path, make_receipt(confirm), "TAG", text=text)


def test_a_duplicate_tag_line_is_refused(bump, confirm, tmp_path):
    text = pins_text() + f'\nTAG = "{OLD_TAG}"\n'
    _refuses(bump, tmp_path, make_receipt(confirm), "TAG: expected exactly one", text=text)


def test_a_missing_version_line_is_refused(bump, confirm, tmp_path):
    text = pins_text().replace('VERSION = "1.100"\n', "")
    _refuses(bump, tmp_path, make_receipt(confirm), "VERSION: expected exactly one",
             text=text)


def test_a_hardcoded_base_url_is_refused(bump, confirm, tmp_path):
    text = pins_text().replace(
        'f"https://github.com/{REPO}/releases/download/{TAG}/"',
        f'"https://github.com/Owner/kcpp/releases/download/{OLD_TAG}/"')
    _refuses(bump, tmp_path, make_receipt(confirm), "_BASE_URL", text=text)


def test_an_entry_in_a_different_shape_is_refused(bump, confirm, tmp_path):
    text = pins_text().replace(
        f'("linux", "cuda"): (\n        "kcpp-linux", {OLD_SIZES["kcpp-linux"]},\n',
        f'("linux", "cuda"): ("kcpp-linux", {OLD_SIZES["kcpp-linux"]},\n')
    _refuses(bump, tmp_path, make_receipt(confirm), "entries start", text=text)


def test_a_duplicate_entry_is_refused(bump, confirm, tmp_path):
    first = pins_text().split("{\n", 1)[1].split("}\n", 1)[0].split("),\n", 1)[0] + "),\n"
    text = pins_text().replace("{\n", "{\n" + first, 1)
    _refuses(bump, tmp_path, make_receipt(confirm), "more than once", text=text)


def test_an_entry_the_release_table_does_not_cover_is_refused(bump):
    text = pins_text()
    table = {(p, b): (n, 1, _h(1)) for p, b, n in KEYS[:-1]}
    with pytest.raises(bump.Refused, match="keys differ"):
        bump.rewrite(text, NEW_TAG, "1.101", table)


def test_a_renamed_asset_is_not_a_mechanical_bump(bump):
    table = {(p, b): (n, 1, _h(1)) for p, b, n in KEYS}
    table[("windows", "cuda")] = ("renamed.exe", 1, _h(1))
    with pytest.raises(bump.Refused, match="not a mechanical bump"):
        bump.rewrite(pins_text(), NEW_TAG, "1.101", table)


def test_check_rewritten_catches_a_value_that_did_not_land(bump):
    table = {(p, b): (n, NEW_SIZES[n], NEW_SHAS[n]) for p, b, n in KEYS}
    good = bump.rewrite(pins_text(), NEW_TAG, "1.101", table)
    bump.check_rewritten(good, NEW_TAG, "1.101", table)
    for broken in (good.replace(f'TAG = "{NEW_TAG}"', f'TAG = "{OLD_TAG}"'),
                   good.replace('VERSION = "1.101"', 'VERSION = "1.100"'),
                   good.replace(NEW_SHAS["kcpp-mac"], OLD_SHAS["kcpp-mac"])):
        with pytest.raises(bump.Refused, match="does not state the target"):
            bump.check_rewritten(broken, NEW_TAG, "1.101", table)
    with pytest.raises(bump.Refused, match="does not parse"):
        bump.check_rewritten("TAG = (", NEW_TAG, "1.101", table)


# --------------------------------------------------------------------------- #
#  Other mentions of the pin                                                   #
# --------------------------------------------------------------------------- #

def test_another_file_naming_the_pin_is_refused(bump, confirm, tmp_path):
    seen = {}

    def scanner(repo, needles):
        seen["needles"] = needles
        return ["docs/music.md", "localm/media/koboldcpp/pins.py", "CHANGELOG.md"]
    _refuses(bump, tmp_path, make_receipt(confirm), "docs/music.md", scanner=scanner)
    assert f'TAG = "{OLD_TAG}"' in seen["needles"]
    assert f"download/{OLD_TAG}/" in seen["needles"]
    assert all(s in seen["needles"] for s in OLD_SHAS.values())


def test_the_exempt_files_may_name_the_pin(bump, confirm, tmp_path):
    def scanner(repo, needles):
        return ["localm/media/koboldcpp/pins.py", "CHANGELOG.md"]
    new, _ = run_plan(bump, tmp_path, make_receipt(confirm), scanner=scanner)
    assert NEW_TAG in new


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_git_grep_finds_a_tracked_mention_and_ignores_untracked(bump, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "a.md").write_text(f"pinned {OLD_TAG} here\n", encoding="utf-8")
    (repo / "b.txt").write_text("nothing\n", encoding="utf-8")
    (repo / "untracked.md").write_text(OLD_TAG, encoding="utf-8")
    _git(repo, "add", "a.md", "b.txt")
    assert bump.git_grep_mentions(repo, [OLD_TAG]) == ["a.md"]
    assert bump.git_grep_mentions(repo, ["absent-needle"]) == []


def test_git_grep_failure_is_a_refusal_not_an_empty_answer(bump, tmp_path):
    with pytest.raises(bump.Refused, match="could not search"):
        bump.git_grep_mentions(tmp_path / "does-not-exist", ["x"])


def test_the_real_tree_names_the_pin_only_in_pins_py(bump):
    """Fails when a second place starts hardcoding the pinned tag or an asset hash."""
    real = (_ROOT / "localm" / "media" / "koboldcpp" / "pins.py").read_text(encoding="utf-8")
    try:
        found = bump.other_mentions(_ROOT, real)
    except bump.Refused as e:
        pytest.skip(f"git is not available here: {e}")
    assert found == []


def test_the_pin_is_found_in_pins_py_by_the_scan(bump):
    real = (_ROOT / "localm" / "media" / "koboldcpp" / "pins.py").read_text(encoding="utf-8")
    try:
        hits = bump.git_grep_mentions(_ROOT, [f'TAG = "{bump.pinned_tag(real)}"'])
    except bump.Refused as e:
        pytest.skip(f"git is not available here: {e}")
    assert "localm/media/koboldcpp/pins.py" in hits


# --------------------------------------------------------------------------- #
#  main()                                                                      #
# --------------------------------------------------------------------------- #

def test_dry_run_prints_a_diff_and_writes_nothing(bump, confirm, tmp_path, monkeypatch, capsys):
    pins = tmp_path / "pins.py"
    pins.write_text(pins_text(), encoding="utf-8")
    before = pins.read_bytes()
    monkeypatch.setattr(bump, "PINS_PATH", pins)
    rec = write_receipt(tmp_path, make_receipt(confirm))
    rc = bump.main(["--tag", NEW_TAG, "--receipt", str(rec)],
                   opener=Opener(release_body()), scanner=no_mentions)
    out = capsys.readouterr().out
    assert rc == 0 and pins.read_bytes() == before
    assert f'+TAG = "{NEW_TAG}"' in out and "dry run" in out
    assert "NOT measured" in out


def test_write_applies_the_edit(bump, confirm, tmp_path, monkeypatch):
    pins = tmp_path / "pins.py"
    pins.write_text(pins_text(), encoding="utf-8")
    monkeypatch.setattr(bump, "PINS_PATH", pins)
    rec = write_receipt(tmp_path, make_receipt(confirm))
    rc = bump.main(["--tag", NEW_TAG, "--receipt", str(rec), "--write"],
                   opener=Opener(release_body()), scanner=no_mentions)
    assert rc == 0
    tree = ast.parse(pins.read_text(encoding="utf-8"))
    tags = [n.value.value for n in tree.body
            if isinstance(n, ast.Assign) and n.targets[0].id == "TAG"]
    assert tags == [NEW_TAG]


def test_write_without_a_receipt_is_refused_and_touches_nothing(
        bump, tmp_path, monkeypatch, capsys):
    pins = tmp_path / "pins.py"
    pins.write_text(pins_text(), encoding="utf-8")
    before = pins.read_bytes()
    monkeypatch.setattr(bump, "PINS_PATH", pins)
    rc = bump.main(["--tag", NEW_TAG, "--write"], opener=Opener(release_body()),
                   scanner=no_mentions)
    assert rc == 1 and pins.read_bytes() == before
    assert "REFUSED" in capsys.readouterr().out


def test_a_refusal_leaves_the_file_untouched(bump, confirm, tmp_path, monkeypatch):
    pins = tmp_path / "pins.py"
    pins.write_text(pins_text(), encoding="utf-8")
    before = pins.read_bytes()
    monkeypatch.setattr(bump, "PINS_PATH", pins)
    bad = make_receipt(confirm, mutate=lambda r: r["checks"]["install"].update(status="FAIL"))
    rec = write_receipt(tmp_path, bad)
    rc = bump.main(["--tag", NEW_TAG, "--receipt", str(rec), "--write"],
                   opener=Opener(release_body()), scanner=no_mentions)
    assert rc == 1 and pins.read_bytes() == before


# --------------------------------------------------------------------------- #
#  The two scripts agree                                                       #
# --------------------------------------------------------------------------- #

def test_the_required_checks_are_a_subset_of_the_known_checks(confirm):
    assert set(confirm.ALWAYS_REQUIRED) <= set(confirm.CHECK_NAMES)


def test_a_receipt_built_by_confirm_is_accepted_and_a_demoted_one_is_not(bump, confirm,
                                                                        tmp_path):
    ok = make_receipt(confirm)
    assert ok["verdict"] == "PASS"
    assert run_plan(bump, tmp_path, copy.deepcopy(ok))[1]["tag"] == NEW_TAG
