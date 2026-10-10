# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/bump_rocm_pin.py: the mechanical half of advancing the AMD ROCm llama.cpp pin.

Properties that make a scripted bump safe:

  * the values that must move together (the lemonade tag, its gfx103X URL and
    digest, the paired upstream CPU tag, both checksum blocks) move in one
    rewrite of the real pins.py and nothing else in the file changes;
  * the CPU tag is found from the llama.cpp commit the lemonade release names,
    and anything ambiguous or unpairable is refused;
  * a write needs a receipt that is a genuine PASS for exactly this tag, checked
    check by check rather than by its verdict field;
  * every edited region is located exactly once, on the real tree as well as on
    broken fixtures.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

bump = importlib.import_module("bump_rocm_pin")
confirm = importlib.import_module("confirm_rocm_runtime")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REAL_PINS = bump.read_lf(bump.PINS_PATH)
REAL_VIEW = bump.read_pins(REAL_PINS)

TAG = "b9990"
COMMIT = "abcde"
FULL = COMMIT + "f" * 35
CPU_TAG = "b99000"
OTHER_FULL = "1234567" + "9" * 33
FAMILIES = ("gfx103X", "gfx110X", "gfx1150", "gfx1151", "gfx120X", "gfx908", "gfx90a")


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _asset(name: str, sha: str | None) -> dict:
    a = {"name": name, "size": 1}
    if sha is not None:
        a["digest"] = f"sha256:{sha}"
    return a


def lemonade_body(tag: str = TAG, *, commit: str = COMMIT, drop=(), skip_digest=(), families=FAMILIES) -> dict:
    assets = []
    for plat in ("windows", "ubuntu"):
        for i, fam in enumerate(families):
            name = f"llama-{tag}-{plat}-rocm-{fam}-x64.zip"
            if name in drop:
                continue
            assets.append(_asset(name, None if name in skip_digest else _sha(f"{plat[0]}{i}")))
    assets.append(_asset("not-an-archive.txt", None))
    return {"tag_name": tag, "draft": False, "prerelease": False,
            "published_at": "2026-10-20T10:00:00Z",
            "body": ("**Build Number**: " + tag + "\n**Operating System(s)**: windows,ubuntu\n"
                     "**ROCm Version**: 10.3.0a20261020\n**Llama.cpp Commit Hash**: " + commit + "\n"
                     "**Build Date**: 2026-10-20 10:00:00 UTC\n"),
            "assets": assets}


def upstream_release(tag: str, commit: str, published: str, *, cpu_sha: str | None = "c" * 64) -> dict:
    asset = _asset(f"llama-{tag}-bin-win-cpu-x64.zip", cpu_sha)
    return {"tag_name": tag, "target_commitish": commit, "published_at": published,
            "assets": [asset, _asset(f"llama-{tag}-bin-win-vulkan-x64.zip", "d" * 64)]}


class FakeApi:
    """The GitHub API as a path -> body table; counts requests."""

    def __init__(self, tag: str = TAG, **kw):
        self.calls: list[str] = []
        self.routes: dict[str, object] = {
            f"repos/{bump.LEMONADE_REPO}/releases/tags/{tag}": lemonade_body(tag, **kw),
            f"repos/{bump.UPSTREAM_REPO}/releases?per_page=100&page=1": [
                upstream_release("b99002", OTHER_FULL, "2026-10-19T12:00:00Z"),
                upstream_release(CPU_TAG, FULL, "2026-10-19T08:00:00Z"),
                upstream_release("b98999", "9" * 40, "2026-10-18T08:00:00Z"),
            ],
            f"repos/{bump.UPSTREAM_REPO}/releases?per_page=100&page=2": [],
            f"repos/{bump.UPSTREAM_REPO}/git/ref/tags/{CPU_TAG}": {"object": {"type": "commit", "sha": FULL}},
        }

    def __call__(self, path: str):
        self.calls.append(path)
        if path not in self.routes:
            raise bump.UpstreamUnreadable(f"HTTP 404 for {path}")
        return copy.deepcopy(self.routes[path])


@pytest.fixture
def api():
    return FakeApi()


@pytest.fixture
def cand(api):
    return bump.fetch_candidate(TAG, api)


# --------------------------------------------------------------------------- #
#  Finding the CPU tag from the lemonade release                               #
# --------------------------------------------------------------------------- #

def test_candidate_pairs_the_lemonade_commit_with_its_upstream_release(api):
    c = bump.fetch_candidate(TAG, api)
    assert (c.tag, c.lemonade_commit, c.rocm_version) == (TAG, COMMIT, "10.3.0a20261020")
    assert (c.cpu_tag, c.cpu_commit, c.cpu_asset) == (CPU_TAG, FULL, f"llama-{CPU_TAG}-bin-win-cpu-x64.zip")
    assert c.cpu_sha256 == "c" * 64
    assert len(c.rocm_assets) == 14 and "not-an-archive.txt" not in c.rocm_assets
    assert c.gfx103x_sha256 == _sha("w0")


@pytest.mark.parametrize("tag", ["1342", "b", "b13x2", "v1342", "", "b1342-rc1"])
def test_candidate_refuses_a_tag_that_is_not_a_lemonade_build_tag(tag, api):
    with pytest.raises(bump.Refused):
        bump.fetch_candidate(tag, api)
    assert api.calls == []


@pytest.mark.parametrize("flag", ["draft", "prerelease"])
def test_candidate_refuses_a_draft_or_prerelease(api, flag):
    api.routes[f"repos/{bump.LEMONADE_REPO}/releases/tags/{TAG}"][flag] = True
    with pytest.raises(bump.Refused, match="draft or prerelease"):
        bump.fetch_candidate(TAG, api)


@pytest.mark.parametrize("body", [
    "no commit line at all",
    "**Llama.cpp Commit Hash**: abcde\n**Llama.cpp Commit Hash**: 12345",
    "**Llama.cpp Commit Hash**: abc",
])
def test_candidate_refuses_release_notes_without_exactly_one_usable_commit(api, body):
    api.routes[f"repos/{bump.LEMONADE_REPO}/releases/tags/{TAG}"]["body"] = body
    with pytest.raises(bump.Refused, match="commit"):
        bump.fetch_candidate(TAG, api)


@pytest.mark.parametrize("body", [None, [], "text", {"tag_name": "b1"}])
def test_candidate_calls_an_unusable_release_body_unreadable_not_bad(body):
    api = FakeApi()
    api.routes[f"repos/{bump.LEMONADE_REPO}/releases/tags/{TAG}"] = body
    with pytest.raises(bump.UpstreamUnreadable):
        bump.fetch_candidate(TAG, api)


def test_candidate_without_an_asset_list_is_unreadable(api):
    del api.routes[f"repos/{bump.LEMONADE_REPO}/releases/tags/{TAG}"]["assets"]
    with pytest.raises(bump.UpstreamUnreadable):
        bump.fetch_candidate(TAG, api)


def test_candidate_refuses_a_release_without_the_gfx103x_windows_asset():
    api = FakeApi(drop=(f"llama-{TAG}-windows-rocm-gfx103X-x64.zip",))
    with pytest.raises(bump.Refused, match="gfx103X"):
        bump.fetch_candidate(TAG, api)


@pytest.mark.parametrize("digest", [None, "md5:abc", "sha256:xyz", "sha256:" + "A" * 10])
def test_candidate_refuses_an_asset_with_no_usable_sha256(digest):
    api = FakeApi()
    body = api.routes[f"repos/{bump.LEMONADE_REPO}/releases/tags/{TAG}"]
    body["assets"][3] = {"name": body["assets"][3]["name"], **({"digest": digest} if digest else {})}
    with pytest.raises(bump.Refused, match="digest"):
        bump.fetch_candidate(TAG, api)


def test_candidate_refuses_when_no_upstream_release_was_built_from_the_commit(api):
    api.routes[f"repos/{bump.LEMONADE_REPO}/releases/tags/{TAG}"]["body"] = "**Llama.cpp Commit Hash**: 00000"
    with pytest.raises(bump.Refused, match="no upstream"):
        bump.fetch_candidate(TAG, api)


def test_candidate_refuses_an_abbreviated_hash_that_matches_two_upstream_commits(api):
    pages = api.routes[f"repos/{bump.UPSTREAM_REPO}/releases?per_page=100&page=1"]
    pages.append(upstream_release("b98000", COMMIT + "0" * 35, "2026-10-17T08:00:00Z"))
    with pytest.raises(bump.Refused, match="more than one"):
        bump.fetch_candidate(TAG, api)


def test_candidate_refuses_two_upstream_releases_of_the_same_commit(api):
    pages = api.routes[f"repos/{bump.UPSTREAM_REPO}/releases?per_page=100&page=1"]
    pages.append(upstream_release("b98000", FULL, "2026-10-17T08:00:00Z"))
    with pytest.raises(bump.Refused, match="more than one"):
        bump.fetch_candidate(TAG, api)


def test_candidate_ignores_upstream_releases_whose_target_is_not_a_commit(api):
    pages = api.routes[f"repos/{bump.UPSTREAM_REPO}/releases?per_page=100&page=1"]
    pages.append(upstream_release("b98000", "master", "2026-10-17T08:00:00Z"))
    pages.append(upstream_release("not-a-build-tag", FULL, "2026-10-17T08:00:00Z"))
    assert bump.fetch_candidate(TAG, api).cpu_tag == CPU_TAG


def test_candidate_refuses_when_the_tag_ref_points_at_another_commit(api):
    api.routes[f"repos/{bump.UPSTREAM_REPO}/git/ref/tags/{CPU_TAG}"] = {
        "object": {"type": "commit", "sha": OTHER_FULL}}
    with pytest.raises(bump.Refused, match="points at"):
        bump.fetch_candidate(TAG, api)


def test_candidate_follows_an_annotated_tag_to_its_commit(api):
    api.routes[f"repos/{bump.UPSTREAM_REPO}/git/ref/tags/{CPU_TAG}"] = {
        "object": {"type": "tag", "sha": "t" * 40, "url": "https://api.github.com/repos/x/git/tags/t"}}
    api.routes["https://api.github.com/repos/x/git/tags/t"] = {"object": {"type": "commit", "sha": FULL}}
    assert bump.fetch_candidate(TAG, api).cpu_commit == FULL
    api.routes["https://api.github.com/repos/x/git/tags/t"] = {"object": {"type": "commit", "sha": OTHER_FULL}}
    with pytest.raises(bump.Refused, match="points at"):
        bump.fetch_candidate(TAG, api)


def test_candidate_refuses_an_upstream_release_without_the_windows_cpu_archive(api):
    rel = api.routes[f"repos/{bump.UPSTREAM_REPO}/releases?per_page=100&page=1"][1]
    rel["assets"] = [a for a in rel["assets"] if "win-cpu" not in a["name"]]
    with pytest.raises(bump.Refused, match="win-cpu"):
        bump.fetch_candidate(TAG, api)


def test_candidate_refuses_a_cpu_archive_without_a_digest():
    api = FakeApi()
    api.routes[f"repos/{bump.UPSTREAM_REPO}/releases?per_page=100&page=1"][1] = upstream_release(
        CPU_TAG, FULL, "2026-10-19T08:00:00Z", cpu_sha=None)
    with pytest.raises(bump.Refused, match="digest"):
        bump.fetch_candidate(TAG, api)


@pytest.mark.parametrize("page", [None, {}, "x"])
def test_candidate_calls_a_release_list_that_is_not_a_list_unreadable(api, page):
    api.routes[f"repos/{bump.UPSTREAM_REPO}/releases?per_page=100&page=1"] = page
    with pytest.raises(bump.UpstreamUnreadable):
        bump.fetch_candidate(TAG, api)


def test_an_unreachable_api_is_unreadable_not_a_bad_build():
    def down(path):
        raise bump.UpstreamUnreadable("HTTP 403 from the GitHub API")
    with pytest.raises(bump.UpstreamUnreadable, match="403"):
        bump.fetch_candidate(TAG, down)


def test_upstream_scan_stops_once_releases_are_older_than_the_window(api):
    old = (bump.parse_date("2026-10-20T10:00:00Z") - bump.UPSTREAM_WINDOW - bump._dt.timedelta(days=1))
    page1 = api.routes[f"repos/{bump.UPSTREAM_REPO}/releases?per_page=100&page=1"]
    page1.append(upstream_release("b50000", "5" * 40, old.strftime("%Y-%m-%dT%H:%M:%SZ")))
    bump.fetch_candidate(TAG, api)
    assert not any(c.endswith("page=2") for c in api.calls)


def test_upstream_scan_continues_to_the_next_page_inside_the_window(api):
    api.routes[f"repos/{bump.UPSTREAM_REPO}/releases?per_page=100&page=2"] = [
        upstream_release("b98000", "8" * 40, "2026-10-17T08:00:00Z")]
    api.routes[f"repos/{bump.UPSTREAM_REPO}/releases?per_page=100&page=3"] = []
    bump.fetch_candidate(TAG, api)
    assert any(c.endswith("page=3") for c in api.calls)


def test_github_get_refuses_a_url_outside_the_api():
    with pytest.raises(bump.UpstreamUnreadable, match="outside"):
        bump.github_get("https://example.invalid/repos/x")


# --------------------------------------------------------------------------- #
#  Rewriting pins.py                                                           #
# --------------------------------------------------------------------------- #

def _changed_lines(old: str, new: str) -> tuple[list[str], list[str]]:
    removed, added = [], []
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0):
        if line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    return removed, added


def test_rewrite_moves_exactly_the_rocm_pin_regions_of_the_real_pins_file(cand):
    new = bump.rewrite_pins(REAL_PINS, cand)
    removed, added = _changed_lines(REAL_PINS, new)
    assert all(
        any(k in line for k in (REAL_VIEW.tag, REAL_VIEW.cpu_tag, REAL_VIEW.url_sha256,
                                "rocm-gfx", "win-cpu-x64"))
        for line in removed), removed
    assert f'_ROCM_TAG = "{TAG}"' in added and f'_ROCM_CPU_TAG = "{CPU_TAG}"' in added
    assert f'    "{TAG}/llama-{TAG}-windows-rocm-gfx103X-x64.zip"' in added
    assert f'    "{_sha("w0")}"' in added
    assert f"    # tag {TAG} ROCm assets (llama.cpp {FULL[:12]}, ROCm 10.3.0a20261020)" in added
    assert f'    "llama-{CPU_TAG}-bin-win-cpu-x64.zip": "{"c" * 64}",' in added
    assert len(removed) == len(added), "one line replaces one line: nothing was added or dropped"
    view = bump.read_pins(new)
    assert (view.tag, view.cpu_tag, view.url_tag, view.url_file_tag) == (TAG, CPU_TAG, TAG, TAG)
    assert bump.pins_problems(view, cand) == []


def test_rewrite_leaves_everything_outside_the_rocm_regions_byte_identical(cand):
    new = bump.rewrite_pins(REAL_PINS, cand)
    start = REAL_PINS.index("# tag b11118 upstream assets")
    end = REAL_PINS.index("    # tag " + REAL_VIEW.tag + " ROCm assets")
    assert REAL_PINS[start:end] in new, "the ggml-org block, which bump_llama_pin.py owns, is untouched"
    assert REAL_PINS[:REAL_PINS.index("DEFAULT_URL")] in new
    assert REAL_PINS[REAL_PINS.index("_ASSET_MATCH = {"):] in new


def test_rewriting_to_the_pinned_values_changes_nothing():
    view = REAL_VIEW
    cand = bump.Candidate(
        tag=view.tag, lemonade_commit="71ad0", rocm_version="10.2.0a20261008",
        published_at=bump.parse_date("2026-10-08T20:46:47Z"), rocm_assets=dict(view.rocm_table),
        cpu_tag=view.cpu_tag, cpu_commit="71ad0590f4808b6202f9213d166913858c73b1bc",
        cpu_asset=f"llama-{view.cpu_tag}-bin-win-cpu-x64.zip",
        cpu_sha256=next(iter(view.cpu_table.values())))
    assert bump.pins_problems(view, cand) == []
    assert bump.rewrite_pins(REAL_PINS, cand) == REAL_PINS


def test_rewrite_orders_the_asset_block_windows_first_then_by_name(cand):
    names = list(bump.read_pins(bump.rewrite_pins(REAL_PINS, cand)).rocm_table)
    assert names == sorted(names, key=lambda n: (0 if "-windows-" in n else 1, n))
    assert names[0].startswith(f"llama-{TAG}-windows-") and names[-1].startswith(f"llama-{TAG}-ubuntu-")


def test_rewrite_without_a_rocm_version_in_the_notes_labels_the_block_by_commit_only():
    api = FakeApi()
    body = api.routes[f"repos/{bump.LEMONADE_REPO}/releases/tags/{TAG}"]
    body["body"] = body["body"].replace("**ROCm Version**: 10.3.0a20261020\n", "")
    cand = bump.fetch_candidate(TAG, api)
    assert cand.rocm_version == ""
    assert f"# tag {TAG} ROCm assets (llama.cpp {FULL[:12]})\n" in bump.rewrite_pins(REAL_PINS, cand)


@pytest.mark.parametrize("name, mutate", [
    ("_ROCM_TAG missing", lambda t: t.replace("_ROCM_TAG = ", "_ROCM_TAG_X = ")),
    ("_ROCM_TAG twice", lambda t: t + '\n_ROCM_TAG = "b1"\n'),
    ("_ROCM_CPU_TAG missing", lambda t: t.replace("_ROCM_CPU_TAG = ", "_CPU_TAG = ")),
    ("_ROCM_CPU_TAG twice", lambda t: t + '\n_ROCM_CPU_TAG = "b2"\n'),
    ("DEFAULT_URL reshaped", lambda t: t.replace("DEFAULT_URL = (", "DEFAULT_URL = str(")),
    ("DEFAULT_URL_SHA256 reshaped", lambda t: t.replace("DEFAULT_URL_SHA256 = (", "DEFAULT_URL_SHA256 = f(")),
    ("ROCm block label changed", lambda t: t.replace(" ROCm assets (", " ROCm builds (")),
    ("ROCm block twice", lambda t: t.replace("    # tag " + REAL_VIEW.tag + " ROCm assets (",
                                             "    # tag b1 ROCm assets (x)\n"
                                             f'    "llama-b1-windows-rocm-gfx103X-x64.zip": "{"1" * 64}",\n'
                                             "    # tag " + REAL_VIEW.tag + " ROCm assets (")),
    ("CPU entry label changed", lambda t: t.replace("upstream Windows CPU archive", "upstream CPU zip")),
    ("CPU entry gone", lambda t: t.replace(f'"llama-{REAL_VIEW.cpu_tag}-bin-win-cpu-x64.zip"', '"other.zip"')),
])
def test_rewrite_refuses_when_a_region_is_not_found_exactly_once(name, mutate, cand):
    broken = mutate(REAL_PINS)
    assert broken != REAL_PINS
    with pytest.raises(bump.Refused, match="exactly one|expected exactly"):
        bump.rewrite_pins(broken, cand)


def test_the_llama_bump_still_finds_its_own_block_after_a_rocm_rewrite(cand):
    llama = _load("bump_llama_pin")
    new = bump.rewrite_pins(REAL_PINS, cand)
    old_block = llama._SHA_BLOCK_RE.search(REAL_PINS)
    new_block = llama._SHA_BLOCK_RE.search(new)
    assert len(list(llama._SHA_BLOCK_RE.finditer(new))) == 1
    assert old_block.group("entries") == new_block.group("entries")
    assert llama._PIN_RE.search(new).group(2) == llama._PIN_RE.search(REAL_PINS).group(2)


def test_the_currency_gate_reads_the_rewritten_tag(cand, tmp_path):
    gate = _load("check_llama_rocm_pin")
    p = tmp_path / "pins.py"
    p.write_text(bump.rewrite_pins(REAL_PINS, cand), encoding="utf-8")
    assert gate.pinned_tag(p) == TAG


# --------------------------------------------------------------------------- #
#  Asset set                                                                   #
# --------------------------------------------------------------------------- #

def test_asset_set_check_accepts_the_same_families_and_extra_ones():
    api = FakeApi(families=FAMILIES + ("gfx1200",))
    bump.check_asset_set(REAL_VIEW, bump.fetch_candidate(TAG, api))


def test_asset_set_check_refuses_a_release_that_dropped_a_family():
    api = FakeApi(drop=(f"llama-{TAG}-ubuntu-rocm-gfx908-x64.zip",))
    with pytest.raises(bump.Refused, match="ubuntu-rocm-gfx908"):
        bump.check_asset_set(REAL_VIEW, bump.fetch_candidate(TAG, api))


# --------------------------------------------------------------------------- #
#  Receipts                                                                    #
# --------------------------------------------------------------------------- #

def _pass_receipt(cand, tag=TAG) -> dict:
    r = confirm.new_receipt(tag, False)
    for name in bump.REQUIRED_CHECKS:
        confirm.set_check(r, name, confirm.PASS, "measured")
    r["candidate"] = cand.to_receipt()
    confirm.finalize(r)
    assert r["verdict"] == "PASS"
    return r


def _write(tmp_path, receipt) -> Path:
    p = tmp_path / "receipt.json"
    p.write_text(json.dumps(receipt), encoding="utf-8")
    return p


def test_the_confirm_script_and_the_bump_agree_on_the_check_names():
    assert tuple(confirm.CHECK_NAMES) == tuple(bump.REQUIRED_CHECKS)
    assert confirm.bump.COMPONENT == bump.COMPONENT == "rocm"


def test_a_genuine_pass_receipt_is_accepted(cand, tmp_path):
    receipt = bump.load_receipt(_write(tmp_path, _pass_receipt(cand)), TAG)
    assert receipt["candidate"]["cpu_tag"] == CPU_TAG


def _tamper(fn):
    def apply(r):
        fn(r)
        return r
    return apply


@pytest.mark.parametrize("why, tamper", [
    ("another tag", _tamper(lambda r: r.update(tag="b9991"))),
    ("--current receipt", _tamper(lambda r: r.update(current=True))),
    ("current missing", _tamper(lambda r: r.pop("current"))),
    ("verdict FAIL", _tamper(lambda r: r.update(verdict="FAIL"))),
    ("verdict INCONCLUSIVE", _tamper(lambda r: r.update(verdict="INCONCLUSIVE"))),
    ("verdict missing", _tamper(lambda r: r.pop("verdict"))),
    ("other component", _tamper(lambda r: r.update(component="llama"))),
    ("other schema", _tamper(lambda r: r.update(schema=2))),
    ("checks missing", _tamper(lambda r: r.pop("checks"))),
    ("checks not an object", _tamper(lambda r: r.update(checks=[]))),
    ("a required check is gone", _tamper(lambda r: r["checks"].pop("gpu_generate"))),
    ("a required check is SKIP under a PASS verdict", _tamper(lambda r: r["checks"]["gpu_generate"].update(status="SKIP"))),
    ("a required check is FAIL under a PASS verdict", _tamper(lambda r: r["checks"]["abi"].update(status="FAIL"))),
    ("a required check is demoted to optional", _tamper(lambda r: r["checks"]["abi"].update(required=False))),
    ("an extra required check is not PASS", _tamper(lambda r: r["checks"].update(
        extra={"status": "SKIP", "required": True, "detail": "x"}))),
    ("an extra optional check FAILED", _tamper(lambda r: r["checks"].update(
        extra={"status": "FAIL", "required": False, "detail": "x"}))),
    ("a check is not an object", _tamper(lambda r: r["checks"].update(abi="PASS"))),
    ("no candidate record", _tamper(lambda r: r.pop("candidate"))),
])
def test_a_receipt_that_is_not_a_genuine_pass_for_this_tag_is_refused(why, tamper, cand, tmp_path):
    with pytest.raises(bump.Refused):
        bump.load_receipt(_write(tmp_path, tamper(_pass_receipt(cand))), TAG)


def test_an_extra_optional_check_that_did_not_run_does_not_block(cand, tmp_path):
    r = _pass_receipt(cand)
    r["checks"]["info"] = {"status": "SKIP", "required": False, "detail": "x"}
    assert bump.load_receipt(_write(tmp_path, r), TAG)


@pytest.mark.parametrize("content", ["", "not json", "[]", "null", '"PASS"'])
def test_an_unreadable_receipt_is_refused(content, tmp_path):
    p = tmp_path / "r.json"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(bump.Refused):
        bump.load_receipt(p, TAG)


def test_a_missing_receipt_file_is_refused(tmp_path):
    with pytest.raises(bump.Refused, match="could not read"):
        bump.load_receipt(tmp_path / "absent.json", TAG)


@pytest.mark.parametrize("key, value", [
    ("tag", "b9999"), ("lemonade_commit", "fffff"), ("gfx103X_sha256", "0" * 64),
    ("cpu_tag", "b1"), ("cpu_commit", "e" * 40), ("cpu_sha256", "0" * 64),
    ("rocm_assets", {}),
])
def test_a_release_that_changed_after_it_was_confirmed_is_detected(key, value, cand):
    recorded = cand.to_receipt()
    assert bump.receipt_mismatches(recorded, cand) == []
    recorded[key] = value
    assert [m for m in bump.receipt_mismatches(recorded, cand) if m.startswith(key)]


# --------------------------------------------------------------------------- #
#  The command                                                                 #
# --------------------------------------------------------------------------- #

@pytest.fixture
def tree(tmp_path):
    pins = tmp_path / "pins.py"
    pins.write_bytes(REAL_PINS.encode("utf-8"))
    return pins


def _run(tree, tmp_path, cand, capsys, *extra, tag=TAG, receipt=None, api=None):
    rp = _write(tmp_path, receipt if receipt is not None else _pass_receipt(cand, tag))
    argv = ["--tag", tag, "--receipt", str(rp), *extra]
    code = bump.main(argv, fetch=api or FakeApi(), pins_path=tree)
    return code, capsys.readouterr().out


def test_a_dry_run_prints_the_diff_and_changes_nothing(tree, tmp_path, cand, capsys):
    before = tree.read_bytes()
    code, out = _run(tree, tmp_path, cand, capsys)
    assert code == 0
    assert tree.read_bytes() == before
    assert f'+_ROCM_TAG = "{TAG}"' in out and "(dry run" in out
    assert "check_llama_abi.py --ref " + CPU_TAG in out


def test_write_applies_the_diff_and_the_result_reads_back(tree, tmp_path, cand, capsys):
    code, out = _run(tree, tmp_path, cand, capsys, "--write")
    assert code == 0 and "wrote" in out
    view = bump.read_pins(bump.read_lf(tree))
    assert (view.tag, view.cpu_tag) == (TAG, CPU_TAG)
    assert bump.pins_problems(view, cand) == []


def test_write_keeps_the_files_crlf_newlines(tree, tmp_path, cand, capsys):
    tree.write_bytes(REAL_PINS.replace("\n", "\r\n").encode("utf-8"))
    code, _ = _run(tree, tmp_path, cand, capsys, "--write")
    data = tree.read_bytes()
    assert code == 0 and data.count(b"\r\n") == data.count(b"\n") > 100


def test_a_second_write_of_the_same_tag_is_refused_as_not_newer(tree, tmp_path, cand, capsys):
    assert _run(tree, tmp_path, cand, capsys, "--write")[0] == 0
    after_first = tree.read_bytes()
    code, out = _run(tree, tmp_path, cand, capsys, "--write")
    assert code == 1 and "not newer" in out
    assert tree.read_bytes() == after_first


@pytest.mark.parametrize("tag", [REAL_VIEW.tag, "b1307", "b1"])
def test_the_pin_never_moves_backwards_or_stays(tag, tree, tmp_path, cand, capsys):
    before = tree.read_bytes()
    code, out = _run(tree, tmp_path, cand, capsys, "--write", tag=tag, receipt=_pass_receipt(cand, tag))
    assert code == 1 and "not newer" in out
    assert tree.read_bytes() == before


@pytest.mark.parametrize("tag", ["1350", "bx", "v1350"])
def test_a_malformed_tag_is_refused(tag, tree, tmp_path, cand, capsys):
    code, out = _run(tree, tmp_path, cand, capsys, "--write", tag=tag)
    assert code == 1 and "REFUSED" in out


def test_a_write_without_a_receipt_is_refused(tree, capsys):
    before = tree.read_bytes()
    assert bump.main(["--tag", TAG, "--write"], fetch=FakeApi(), pins_path=tree) == 1
    assert "--receipt is required" in capsys.readouterr().out
    assert tree.read_bytes() == before


def test_a_dry_run_without_a_receipt_is_refused_too(tree, capsys):
    assert bump.main(["--tag", TAG], fetch=FakeApi(), pins_path=tree) == 1
    assert "--receipt is required" in capsys.readouterr().out


def test_a_tampered_receipt_leaves_the_tree_untouched(tree, tmp_path, cand, capsys):
    bad = _pass_receipt(cand)
    bad["checks"]["gpu_generate"]["status"] = "SKIP"
    before = tree.read_bytes()
    code, out = _run(tree, tmp_path, cand, capsys, "--write", receipt=bad)
    assert code == 1 and "gpu_generate" in out
    assert tree.read_bytes() == before


def test_a_release_republished_after_confirmation_is_refused(tree, tmp_path, cand, capsys):
    api = FakeApi()
    body = api.routes[f"repos/{bump.LEMONADE_REPO}/releases/tags/{TAG}"]
    for a in body["assets"]:
        if a.get("name") == f"llama-{TAG}-windows-rocm-gfx103X-x64.zip":
            a["digest"] = "sha256:" + "9" * 64
    before = tree.read_bytes()
    code, out = _run(tree, tmp_path, cand, capsys, "--write", api=api)
    assert code == 1 and "changed after it was confirmed" in out
    assert tree.read_bytes() == before


def test_an_unreachable_api_refuses_without_writing(tree, tmp_path, cand, capsys):
    api = FakeApi()
    api.routes.clear()
    before = tree.read_bytes()
    code, out = _run(tree, tmp_path, cand, capsys, "--write", api=api)
    assert code == 1 and "REFUSED" in out
    assert tree.read_bytes() == before


def test_a_release_that_dropped_a_family_is_refused_without_writing(tree, tmp_path, capsys):
    api = FakeApi(drop=(f"llama-{TAG}-windows-rocm-gfx90a-x64.zip",))
    receipt = _pass_receipt(bump.fetch_candidate(TAG, api))
    before = tree.read_bytes()
    code, out = _run(tree, tmp_path, None, capsys, "--write", api=api, receipt=receipt)
    assert code == 1 and "no longer publishes" in out
    assert tree.read_bytes() == before


def test_a_reshaped_pins_file_is_refused_without_writing(tree, tmp_path, cand, capsys):
    tree.write_bytes(REAL_PINS.replace("_ROCM_CPU_TAG = ", "_CPU = ").encode("utf-8"))
    before = tree.read_bytes()
    code, out = _run(tree, tmp_path, cand, capsys, "--write")
    assert code == 1 and "REFUSED" in out
    assert tree.read_bytes() == before


def test_the_real_tree_parses_and_agrees_with_itself():
    assert REAL_VIEW.tag == REAL_VIEW.url_tag == REAL_VIEW.url_file_tag == REAL_VIEW.rocm_block_tag
    assert REAL_VIEW.cpu_tag == REAL_VIEW.cpu_block_tag
    assert REAL_VIEW.url_sha256 == REAL_VIEW.rocm_table[f"llama-{REAL_VIEW.tag}-{bump._GFX103X}"]
    assert len(REAL_VIEW.rocm_table) == 14
