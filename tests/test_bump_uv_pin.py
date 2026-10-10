# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/bump_uv_pin.py: the mechanical half of advancing the pinned uv.

Covers the properties that make a scripted bump safe:

  * all five pinned places (setup.sh, setup-gui.sh, setup.bat, setup-gui.bat,
    docker/Dockerfile) move together, each with the digest of the asset it
    verifies, and nothing else in those files changes;
  * a missed place fails the bump rather than shipping a disagreeing pin, and a
    sixth file that starts pinning uv fails a test;
  * a write needs a receipt that confirms exactly this tag, and the digests it
    recorded must still equal what the release API publishes;
  * the bump only moves forward, and refuses when an edited region is not found
    exactly once.

No network: the GitHub opener is injected. The five files are copied from the
real tree, so the edited regions are the shipped ones.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_BUMP = _ROOT / "scripts" / "bump_uv_pin.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bump():
    return _load(_BUMP, "bump_uv_pin")


NEW = "9.9.9"
SH, PS1, LINUX = "11" * 32, "22" * 32, "33" * 32
DIGESTS = {"uv-installer.sh": SH, "uv-installer.ps1": PS1,
           "uv-x86_64-unknown-linux-gnu.tar.gz": LINUX}
SITE_PATHS = ("setup.sh", "setup-gui.sh", "setup.bat", "setup-gui.bat", "docker/Dockerfile")


def _release_body(tag: str = NEW, digests: dict | None = None) -> dict:
    digests = DIGESTS if digests is None else digests
    assets = [{"name": n, "digest": f"sha256:{d}"} for n, d in digests.items()]
    assets.append({"name": "uv-aarch64-apple-darwin.tar.gz", "digest": "sha256:" + "ee" * 32})
    assets.append({"name": "sha256.sum", "digest": None})
    return {"tag_name": tag, "assets": assets}


class _Resp:
    def __init__(self, payload):
        self._data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _opener(payload):
    calls = []

    def opener(req, timeout=None):
        calls.append(req.full_url)
        if isinstance(payload, Exception):
            raise payload
        return _Resp(payload)

    opener.calls = calls
    return opener


def _copy_tree(dest: Path) -> dict:
    """Copy the five real files into *dest*; returns {path: original bytes}."""
    originals = {}
    for rel in SITE_PATHS:
        data = (_ROOT / rel).read_bytes()
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        originals[rel] = data
    return originals


def _texts(root: Path) -> dict:
    return {rel: (root / rel).read_bytes().decode("utf-8").replace("\r\n", "\n")
            for rel in SITE_PATHS}


def _good_receipt(tag: str = NEW) -> dict:
    names = ("release_listing", "installer_digests", "isolation", "installer_run",
             "containment", "version", "venv_python", "pip_install", "lock_check")
    return {
        "schema": 1, "component": "uv", "tag": tag, "current": False, "verdict": "PASS",
        "why": "every required check passed", "written_at": "2026-01-01T00:00:00Z",
        "hardware": {},
        "checks": {n: {"status": "PASS", "required": True, "detail": "ok"} for n in names},
        "assets": dict(DIGESTS),
    }


def _write_receipt(tmp_path: Path, receipt) -> Path:
    path = tmp_path / "receipt.json"
    path.write_text(receipt if isinstance(receipt, str) else json.dumps(receipt),
                    encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
#  The rewrite                                                                 #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("rel", SITE_PATHS)
def test_each_pinned_place_is_rewritten_with_its_own_digest(bump, rel):
    texts = {r: (_ROOT / r).read_bytes().decode("utf-8").replace("\r\n", "\n")
             for r in SITE_PATHS}
    new = bump.rewrite(texts, NEW, DIGESTS)
    site = next(s for s in bump.SITES if s.path == rel)
    assert bump.read_site(site, new[rel]) == (NEW, DIGESTS[site.asset])


@pytest.mark.parametrize("rel", SITE_PATHS)
def test_only_the_version_and_sha_lines_change(bump, rel):
    texts = _texts(_ROOT)
    new = bump.rewrite(texts, NEW, DIGESTS)
    old_lines, new_lines = texts[rel].splitlines(), new[rel].splitlines()
    assert len(old_lines) == len(new_lines)
    changed = [(a, b) for a, b in zip(old_lines, new_lines, strict=True) if a != b]
    assert len(changed) == 2
    assert all(NEW in b or any(d in b for d in DIGESTS.values()) for _, b in changed)


def test_the_five_places_are_the_five_files_named_in_the_contract(bump):
    assert tuple(s.path for s in bump.SITES) == SITE_PATHS
    assert {s.asset for s in bump.SITES} == set(DIGESTS)


@pytest.mark.parametrize("missed", SITE_PATHS)
def test_a_bump_that_misses_one_place_is_refused(bump, monkeypatch, missed):
    real = bump.rewrite_site

    def skipping(site, text, tag, digests):
        return text if site.path == missed else real(site, text, tag, digests)

    monkeypatch.setattr(bump, "rewrite_site", skipping)
    with pytest.raises(bump.Refused, match=re.escape(missed)):
        bump.rewrite(_texts(_ROOT), NEW, DIGESTS)


@pytest.mark.parametrize("missed", SITE_PATHS)
def test_a_bump_that_leaves_one_digest_stale_is_refused(bump, missed):
    texts = _texts(_ROOT)
    new = bump.rewrite(texts, NEW, DIGESTS)
    site = next(s for s in bump.SITES if s.path == missed)
    new[missed] = bump._replace(site.sha_re, new[missed], "0" * 64, "sha")
    with pytest.raises(bump.Refused, match=re.escape(missed)):
        bump.verify(new, NEW, DIGESTS)


def test_no_file_outside_the_five_pins_uv_from_a_release_download(bump):
    git = subprocess.run(["git", "-C", str(_ROOT), "ls-files", "-z"],
                         capture_output=True, text=False, timeout=60)
    if git.returncode != 0:
        pytest.skip("not a git checkout")
    pinning = []
    for rel in git.stdout.decode("utf-8").split("\0"):
        path = _ROOT / rel
        if not rel or rel.startswith(("tests/", "scripts/")) or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "astral-sh/uv/releases/download" in text:
            pinning.append(rel)
    assert sorted(pinning) == sorted(SITE_PATHS)


# --------------------------------------------------------------------------- #
#  Forward only                                                                #
# --------------------------------------------------------------------------- #

def test_the_shipped_pins_can_be_bumped_to_a_newer_release(bump):
    bump.rewrite(_texts(_ROOT), "999.0.0", DIGESTS)


@pytest.mark.parametrize("tag", ["0.0.1", "0.12.9", "0.12.23"])
def test_an_older_tag_is_refused(bump, tag):
    with pytest.raises(bump.Refused, match="older"):
        bump.rewrite(_texts(_ROOT), tag, DIGESTS)


def test_the_same_tag_is_refused_when_every_place_already_pins_it(bump):
    texts = _texts(_ROOT)
    aligned = bump.rewrite(texts, NEW, DIGESTS)
    with pytest.raises(bump.Refused, match="nothing to bump"):
        bump.rewrite(aligned, NEW, DIGESTS)


def test_a_tag_older_than_one_place_is_refused_even_if_newer_than_the_rest(bump):
    aligned = bump.rewrite(_texts(_ROOT), "5.0.0", DIGESTS)
    site = next(s for s in bump.SITES if s.path == "docker/Dockerfile")
    aligned["docker/Dockerfile"] = bump._replace(site.version_re, aligned["docker/Dockerfile"],
                                                 "7.0.0", "v")
    with pytest.raises(bump.Refused, match="older"):
        bump.rewrite(aligned, "6.0.0", DIGESTS)


def test_the_newest_pinned_version_realigns_a_lagging_place(bump):
    aligned = bump.rewrite(_texts(_ROOT), "5.0.0", DIGESTS)
    site = next(s for s in bump.SITES if s.path == "docker/Dockerfile")
    aligned["docker/Dockerfile"] = bump._replace(site.version_re, aligned["docker/Dockerfile"],
                                                 "4.9.0", "v")
    new = bump.rewrite(aligned, "5.0.0", DIGESTS)
    assert bump.read_site(site, new["docker/Dockerfile"])[0] == "5.0.0"


@pytest.mark.parametrize("tag", ["v0.14.0", "0.14", "0.14.0.1", "0.14.0-rc1", "", "latest", "0.14.x"])
def test_a_tag_that_is_not_a_plain_release_version_is_refused(bump, tag):
    with pytest.raises(bump.Refused):
        bump.parse_semver(tag)


def test_versions_compare_numerically_not_as_text(bump):
    texts = _texts(_ROOT)
    aligned = bump.rewrite(texts, "99.9.0", DIGESTS)
    bump.rewrite(aligned, "99.10.0", DIGESTS)
    with pytest.raises(bump.Refused, match="older"):
        bump.rewrite(bump.rewrite(texts, "99.10.0", DIGESTS), "99.9.0", DIGESTS)


# --------------------------------------------------------------------------- #
#  Regions found exactly once                                                  #
# --------------------------------------------------------------------------- #

def _region_line(bump, rel: str, which: str) -> str:
    site = next(s for s in bump.SITES if s.path == rel)
    text = _texts(_ROOT)[rel]
    m = (site.version_re if which == "version" else site.sha_re).search(text)
    return text[text.rfind("\n", 0, m.start()) + 1:text.find("\n", m.end())]


@pytest.mark.parametrize("which", ["version", "sha"])
@pytest.mark.parametrize("rel", SITE_PATHS)
def test_a_missing_region_is_refused(bump, rel, which):
    texts = _texts(_ROOT)
    line = _region_line(bump, rel, which)
    texts[rel] = texts[rel].replace(line + "\n", "", 1)
    with pytest.raises(bump.Refused, match="exactly one match, found 0"):
        bump.rewrite(texts, "999.0.0", DIGESTS)


@pytest.mark.parametrize("which", ["version", "sha"])
@pytest.mark.parametrize("rel", SITE_PATHS)
def test_a_duplicated_region_is_refused(bump, rel, which):
    texts = _texts(_ROOT)
    line = _region_line(bump, rel, which)
    texts[rel] = texts[rel].replace(line + "\n", line + "\n" + line + "\n", 1)
    with pytest.raises(bump.Refused, match="exactly one match, found 2"):
        bump.rewrite(texts, "999.0.0", DIGESTS)


# --------------------------------------------------------------------------- #
#  The API listing                                                             #
# --------------------------------------------------------------------------- #

def test_digests_are_read_from_the_asset_digest_field(bump):
    opener = _opener(_release_body())
    assert bump.fetch_digests(NEW, opener) == DIGESTS
    assert opener.calls == [f"https://api.github.com/repos/astral-sh/uv/releases/tags/{NEW}"]


def _without(name):
    body = _release_body()
    body["assets"] = [a for a in body["assets"] if a["name"] != name]
    return body


def _with_digest(name, digest):
    body = _release_body()
    for a in body["assets"]:
        if a["name"] == name:
            a["digest"] = digest
    return body


@pytest.mark.parametrize("payload, why", [
    (b"not json", "could not read"),
    (RuntimeError("boom"), "could not read"),
    ([], "is for None"),
    ({"tag_name": "0.0.1", "assets": []}, "is for '0.0.1'"),
    ({"tag_name": NEW}, "no asset list"),
    ({"tag_name": NEW, "assets": "x"}, "no asset list"),
    (_without("uv-installer.ps1"), "no asset named uv-installer.ps1"),
    (_without("uv-x86_64-unknown-linux-gnu.tar.gz"), "no asset named uv-x86_64"),
    (_with_digest("uv-installer.sh", None), "no sha256 digest"),
    (_with_digest("uv-installer.sh", "md5:" + "a" * 32), "no sha256 digest"),
    (_with_digest("uv-installer.sh", "sha256:abc"), "malformed digest"),
    (_with_digest("uv-installer.sh", "sha256:" + "A" * 64), "malformed digest"),
])
def test_a_bad_release_listing_is_refused(bump, payload, why):
    with pytest.raises(bump.Refused, match=why):
        bump.fetch_digests(NEW, _opener(payload))


# --------------------------------------------------------------------------- #
#  The receipt                                                                 #
# --------------------------------------------------------------------------- #

def test_a_good_receipt_yields_its_recorded_digests(bump, tmp_path):
    assert bump.load_receipt(_write_receipt(tmp_path, _good_receipt()), NEW) == DIGESTS


def _mut(fn):
    def apply(r):
        fn(r)
        return r
    return apply


@pytest.mark.parametrize("mutate, why", [
    (_mut(lambda r: r.update(tag="9.9.8")), "is for '9.9.8'"),
    (_mut(lambda r: r.update(current=True)), "--current"),
    (_mut(lambda r: r.update(current="no")), "--current"),
    (_mut(lambda r: r.pop("current")), "--current"),
    (_mut(lambda r: r.update(verdict="FAIL")), "not PASS"),
    (_mut(lambda r: r.update(verdict="INCONCLUSIVE")), "not PASS"),
    (_mut(lambda r: r.update(schema=2)), "schema 1"),
    (_mut(lambda r: r.update(component="llama")), "schema 1"),
    (_mut(lambda r: r["checks"]["lock_check"].update(status="FAIL")), "lock_check: FAIL"),
    (_mut(lambda r: r["checks"]["installer_run"].update(status="SKIP")), "installer_run: SKIP"),
    (_mut(lambda r: r["checks"].pop("containment")), "containment: absent"),
    (_mut(lambda r: r["checks"]["version"].update(required=False)), "version: not marked required"),
    (_mut(lambda r: r["checks"].update(extra={"status": "FAIL", "required": True})), "extra: FAIL"),
    (_mut(lambda r: r.pop("checks")), "no checks"),
    (_mut(lambda r: r.pop("assets")), "does not record a sha256"),
    (_mut(lambda r: r["assets"].pop("uv-installer.ps1")), "does not record a sha256"),
    (_mut(lambda r: r["assets"].update({"uv-installer.sh": "xyz"})), "does not record a sha256"),
])
def test_a_receipt_that_does_not_confirm_this_tag_is_refused(bump, tmp_path, mutate, why):
    receipt = mutate(copy.deepcopy(_good_receipt()))
    with pytest.raises(bump.Refused, match=re.escape(why)):
        bump.load_receipt(_write_receipt(tmp_path, receipt), NEW)


@pytest.mark.parametrize("text", ["", "not json", "[]", "null", '"PASS"'])
def test_an_unreadable_receipt_is_refused(bump, tmp_path, text):
    with pytest.raises(bump.Refused):
        bump.load_receipt(_write_receipt(tmp_path, text), NEW)


def test_a_missing_receipt_file_is_refused(bump, tmp_path):
    with pytest.raises(bump.Refused, match="could not read the receipt"):
        bump.load_receipt(tmp_path / "absent.json", NEW)


# --------------------------------------------------------------------------- #
#  The command line                                                            #
# --------------------------------------------------------------------------- #

def _run(bump, tmp_path, *args, payload=None, receipt=None):
    root = tmp_path / "tree"
    originals = _copy_tree(root)
    argv = ["--tag", NEW, "--repo-root", str(root), *args]
    if receipt is not None:
        argv += ["--receipt", str(_write_receipt(tmp_path, receipt))]
    rc = bump.main(argv, opener=_opener(_release_body() if payload is None else payload))
    return rc, root, originals


def _bytes(root):
    return {rel: (root / rel).read_bytes() for rel in SITE_PATHS}


def test_a_dry_run_prints_a_diff_and_changes_nothing(bump, tmp_path, capsys):
    rc, root, originals = _run(bump, tmp_path, receipt=_good_receipt())
    out = capsys.readouterr().out
    assert rc == 0
    assert _bytes(root) == originals
    for rel in SITE_PATHS:
        assert f"+++ b/{rel}" in out
    assert "dry run: nothing written" in out


def test_a_dry_run_without_a_receipt_says_nothing_is_confirmed(bump, tmp_path, capsys):
    rc, root, originals = _run(bump, tmp_path)
    assert rc == 0
    assert "nothing is confirmed" in capsys.readouterr().out
    assert _bytes(root) == originals


def test_write_without_a_receipt_is_refused_and_changes_nothing(bump, tmp_path, capsys):
    rc, root, originals = _run(bump, tmp_path, "--write")
    assert rc == 1
    assert "--write needs --receipt" in capsys.readouterr().out
    assert _bytes(root) == originals


def test_write_with_a_good_receipt_moves_all_five_places(bump, tmp_path):
    rc, root, originals = _run(bump, tmp_path, "--write", receipt=_good_receipt())
    assert rc == 0
    after = _texts(root)
    for s in bump.SITES:
        assert bump.read_site(s, after[s.path]) == (NEW, DIGESTS[s.asset])
    assert bump.verify(after, NEW, DIGESTS) is None


def test_write_keeps_each_files_newline_style(bump, tmp_path):
    rc, root, originals = _run(bump, tmp_path, "--write", receipt=_good_receipt())
    assert rc == 0
    for rel, before in originals.items():
        after = (root / rel).read_bytes()
        assert (b"\r\n" in before) == (b"\r\n" in after)
        assert after.count(b"\n") == before.count(b"\n")
        if b"\r\n" in after:
            assert after.count(b"\r\n") == after.count(b"\n")


def test_a_receipt_for_another_tag_is_refused_and_changes_nothing(bump, tmp_path, capsys):
    rc, root, originals = _run(bump, tmp_path, "--write", receipt=_good_receipt("9.9.8"))
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().out
    assert _bytes(root) == originals


def test_a_receipt_whose_digests_differ_from_the_api_is_refused(bump, tmp_path, capsys):
    receipt = _good_receipt()
    receipt["assets"]["uv-installer.sh"] = "44" * 32
    rc, root, originals = _run(bump, tmp_path, "--write", receipt=receipt)
    out = capsys.readouterr().out
    assert rc == 1
    assert "differ from the ones the API publishes now for: uv-installer.sh" in out
    assert _bytes(root) == originals


def test_an_unreadable_release_is_refused_and_changes_nothing(bump, tmp_path, capsys):
    rc, root, originals = _run(bump, tmp_path, "--write", receipt=_good_receipt(),
                               payload=RuntimeError("offline"))
    assert rc == 1
    assert "could not read the 9.9.9 release" in capsys.readouterr().out
    assert _bytes(root) == originals


def test_an_older_tag_is_refused_on_the_command_line(bump, tmp_path, capsys):
    root = tmp_path / "tree"
    originals = _copy_tree(root)
    receipt = _write_receipt(tmp_path, _good_receipt("0.0.1"))
    rc = bump.main(["--tag", "0.0.1", "--repo-root", str(root), "--receipt", str(receipt),
                    "--write"], opener=_opener(_release_body("0.0.1")))
    assert rc == 1
    assert "older" in capsys.readouterr().out
    assert _bytes(root) == originals


def test_a_failed_write_restores_every_file(bump, tmp_path, monkeypatch, capsys):
    root = tmp_path / "tree"
    originals = _copy_tree(root)
    real = bump._write
    state = {"n": 0}

    def flaky(path, text, newline):
        state["n"] += 1
        if state["n"] == 3:
            raise OSError("disk full")
        real(path, text, newline)

    monkeypatch.setattr(bump, "_write", flaky)
    receipt = _write_receipt(tmp_path, _good_receipt())
    rc = bump.main(["--tag", NEW, "--repo-root", str(root), "--receipt", str(receipt),
                    "--write"], opener=_opener(_release_body()))
    assert rc == 1
    assert "every file was restored" in capsys.readouterr().out
    assert _bytes(root) == originals


def test_an_edited_file_that_lost_a_region_is_refused_on_the_command_line(bump, tmp_path, capsys):
    root = tmp_path / "tree"
    originals = _copy_tree(root)
    path = root / "docker/Dockerfile"
    path.write_bytes(path.read_bytes().replace(b"ARG UV_SHA256=", b"ARG UV_SHA_=", 1))
    originals["docker/Dockerfile"] = path.read_bytes()
    receipt = _write_receipt(tmp_path, _good_receipt())
    rc = bump.main(["--tag", NEW, "--repo-root", str(root), "--receipt", str(receipt),
                    "--write"], opener=_opener(_release_body()))
    assert rc == 1
    assert "exactly one match, found 0" in capsys.readouterr().out
    assert _bytes(root) == originals


def test_a_missing_pin_file_is_refused_not_skipped(bump, tmp_path, capsys):
    root = tmp_path / "tree"
    _copy_tree(root)
    (root / "setup-gui.bat").unlink()
    rc = bump.main(["--tag", NEW, "--repo-root", str(root)], opener=_opener(_release_body()))
    assert rc == 1
    assert "could not read setup-gui.bat" in capsys.readouterr().out
