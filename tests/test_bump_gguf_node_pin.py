# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/bump_gguf_node_pin.py: advancing the pinned ComfyUI-GGUF commit.

Offline: the GitHub API is replaced by an in-memory fake that answers in the real
JSON shapes (repository, compare), and source archives are served by
tests/_source_archives.py. The tests cover:

  * a dry run changes nothing; --write moves the CustomNodePin commit and no
    other byte of the file, and replaces the old commit's ARCHIVE_SHA256 entry
    with the sha256 of the new commit's archive, so the shipped-pins test in
    test_source_archive.py passes on the bumped tree;
  * a refusal edits nothing: a malformed or unchanged sha, a commit GitHub does
    not know, one that is behind, diverged or not a descendant, one that is not
    reachable from the default branch, an answer with another tip or merge base;
  * the pin is found exactly once and only the commit changes;
  * the pin on the real tree parses.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import urllib.error
from pathlib import Path

import pytest

from tests._source_archives import FakeGitHub as FakeArchives
from tests._source_archives import assert_shipped_pins_carry_a_sha256

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "bump_gguf_node_pin.py"

OLD = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
MID = "b" * 40
NEW = "c" * 40

PIN_FIXTURE = f'''COMFYUI_PINNED_VERSION = "v0.31.1"


class CustomNodePin:
    pass


# The ONLY non-core node any shipped workflow uses today.
_GGUF_NODE = CustomNodePin(
    name="ComfyUI-GGUF",
    repo="https://github.com/city96/ComfyUI-GGUF.git",
    commit="{OLD}")

CLASS_TYPE_TO_NODE: dict = {{
    "UnetLoaderGGUFAdvanced": _GGUF_NODE,
}}
'''


COMFY_COMMIT = "426f5eea52a7ece1dd599fff4cb6cd1f352ddd24"
ARCHIVE_URL = f"https://codeload.github.com/city96/ComfyUI-GGUF/zip/{NEW}"

ARCHIVE_FIXTURE = f'''"""Source trees."""

ARCHIVE_SHA256 = {{
    "{COMFY_COMMIT}":
        "{"1" * 64}",
    "{OLD}":
        "{"2" * 64}",
}}
'''


def _load():
    spec = importlib.util.spec_from_file_location("bump_gguf_node_pin", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bump():
    return _load()


@pytest.fixture
def archives(tmp_path, monkeypatch):
    """Offline GitHub archives; NEW of ComfyUI-GGUF is served."""
    fake = FakeArchives(tmp_path / "archives").install(monkeypatch)
    fake.add("city96", "ComfyUI-GGUF", {"nodes.py": "NODES = {}\n"}, commit=NEW)
    return fake


def _served_sha256(archives):
    return hashlib.sha256(archives.archives[ARCHIVE_URL].read_bytes()).hexdigest()


@pytest.fixture
def archive_file(tmp_path):
    path = tmp_path / "localm" / "media" / "source_archive.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ARCHIVE_FIXTURE, encoding="utf-8", newline="\n")
    return path


@pytest.fixture
def tree(bump, tmp_path, monkeypatch, archives, archive_file):
    pin = tmp_path / bump.PIN_REL
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text(PIN_FIXTURE, encoding="utf-8", newline="\n")
    monkeypatch.setattr(bump, "REPO", tmp_path)

    def no_network(req, timeout):
        raise AssertionError("a test reached the network")
    monkeypatch.setattr(bump, "_default_open", no_network)
    return pin


class FakeGitHub:
    """Answers the three API reads the script makes."""

    def __init__(self, slug="city96/ComfyUI-GGUF", branch="main", ahead=2, **overrides):
        self.slug, self.requested = slug, []
        commits = [{"sha": MID, "commit": {"message": "Refactor loader\n\nbody"}},
                   {"sha": NEW, "commit": {"message": "Support a new tensor type"}}][-ahead:]
        self.docs = {
            f"repos/{slug}": {"default_branch": branch},
            f"repos/{slug}/compare/{OLD}...{NEW}": {
                "status": "ahead", "ahead_by": ahead, "behind_by": 0,
                "merge_base_commit": {"sha": OLD}, "commits": commits,
                "files": [{"filename": "loader.py", "status": "modified"},
                          {"filename": "nodes.py", "status": "modified"}]},
            f"repos/{slug}/compare/{NEW}...{branch}": {
                "status": "ahead", "ahead_by": 3, "behind_by": 0},
        }
        self.docs.update(overrides)

    def __call__(self, url):
        self.requested.append(url)
        key = url.removeprefix("https://api.github.com/")
        return self.docs[key]


def run(bump, argv, fake):
    return bump.main(argv, fetch_json_fn=fake)


# --------------------------------------------------------------------------- #
#  The edit                                                                   #
# --------------------------------------------------------------------------- #

def test_dry_run_changes_nothing_and_shows_what_changed_upstream(bump, tree, archive_file,
                                                                archives, capsys):
    before = tree.read_bytes()
    fake = FakeGitHub()
    assert run(bump, ["--tag", NEW], fake) == 0
    out = capsys.readouterr().out
    assert f'-    commit="{OLD}")' in out and f'+    commit="{NEW}")' in out
    assert f'-    "{OLD}":' in out and f'+    "{NEW}":' in out
    assert f'+        "{_served_sha256(archives)}",' in out
    assert archive_file.read_text(encoding="utf-8") == ARCHIVE_FIXTURE
    assert "2 commit(s), 2 file(s) changed" in out
    assert "Support a new tensor type" in out and "modified  loader.py" in out
    assert "descendant of the pin, reachable from main" in out
    assert "dry run: nothing written" in out and "REMAINING STEPS" in out
    assert "pinned ComfyUI (v0.31.1)" in out
    assert tree.read_bytes() == before
    assert fake.requested == [
        "https://api.github.com/repos/city96/ComfyUI-GGUF",
        f"https://api.github.com/repos/city96/ComfyUI-GGUF/compare/{OLD}...{NEW}",
        f"https://api.github.com/repos/city96/ComfyUI-GGUF/compare/{NEW}...main"]


def test_write_moves_only_the_commit(bump, tree, capsys):
    assert run(bump, ["--tag", NEW.upper(), "--write"], FakeGitHub()) == 0
    assert tree.read_text(encoding="utf-8") == PIN_FIXTURE.replace(OLD, NEW)
    assert "wrote localm/media/managed_comfy_fresh.py" in capsys.readouterr().out
    assert run(bump, ["--tag", NEW, "--write"], FakeGitHub()) == 1
    assert "is the commit already pinned" in capsys.readouterr().out


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_the_file_keeps_its_own_line_endings(bump, tree, newline):
    tree.write_bytes(PIN_FIXTURE.replace("\n", newline).encode("utf-8"))
    assert run(bump, ["--tag", NEW, "--write"], FakeGitHub()) == 0
    data = tree.read_bytes()
    assert data.count(newline.encode()) == data.count(b"\n")
    assert (b"\r" in data) == (newline == "\r\n")
    assert NEW.encode() in data and OLD.encode() not in data


def test_write_replaces_the_old_commits_archive_sha256_in_place(bump, tree, archive_file,
                                                                archives, capsys):
    assert run(bump, ["--tag", NEW, "--write"], FakeGitHub()) == 0
    assert "wrote localm/media/source_archive.py" in capsys.readouterr().out
    expected = ARCHIVE_FIXTURE.replace(f'"{OLD}":\n        "{"2" * 64}"',
                                       f'"{NEW}":\n        "{_served_sha256(archives)}"')
    assert expected != ARCHIVE_FIXTURE
    assert archive_file.read_text(encoding="utf-8") == expected
    assert archives.requested == [ARCHIVE_URL]


@pytest.mark.parametrize("case", ["another commit", "missing"])
def test_an_unusable_archive_refuses_and_edits_nothing(bump, tree, archive_file, archives,
                                                       capsys, case):
    if case == "another commit":
        archives.add("city96", "ComfyUI-GGUF", {"nodes.py": "x\n"}, commit=NEW, comment=MID)
        fragment = f"is of commit '{MID}'"
    else:
        del archives.archives[ARCHIVE_URL]
        fragment = "could not be fetched"
    before = tree.read_bytes()
    assert run(bump, ["--tag", NEW, "--write"], FakeGitHub()) == 1
    out = capsys.readouterr().out
    assert "REFUSED" in out and fragment in out
    assert tree.read_bytes() == before
    assert archive_file.read_text(encoding="utf-8") == ARCHIVE_FIXTURE


def test_a_simulated_bump_of_the_real_tree_passes_the_shipped_pins_test(bump, tmp_path,
                                                                        archives, monkeypatch):
    media = tmp_path / "localm" / "media"
    media.mkdir(parents=True)
    for name in ("managed_comfy_fresh.py", "source_archive.py"):
        shutil.copyfile(_ROOT / "localm" / "media" / name, media / name)
    monkeypatch.setattr(bump, "REPO", tmp_path)
    pin = media / "managed_comfy_fresh.py"
    original = pin.read_bytes()
    assert bump.read_pin(original.decode("utf-8"))[1] == OLD, "FakeGitHub answers for OLD"

    pin.write_text(bump.rewrite(original.decode("utf-8"), NEW), encoding="utf-8")
    with pytest.raises(AssertionError, match=ARCHIVE_URL):
        assert_shipped_pins_carry_a_sha256(tmp_path)
    pin.write_bytes(original)

    assert run(bump, ["--tag", NEW, "--write"], FakeGitHub()) == 0
    assert_shipped_pins_carry_a_sha256(tmp_path)
    text = (media / "source_archive.py").read_text(encoding="utf-8")
    assert f'"{NEW}":\n        "{_served_sha256(archives)}",' in text


def test_old_commit_mentions_elsewhere_are_listed_not_edited(bump, tree, capsys):
    root = tree.parents[2]
    other = root / "localm" / "media" / "comfy_client.py"
    other.write_text(f"# custom_nodes/ComfyUI-GGUF, pin {OLD[:7]}).\nx = 1\n", encoding="utf-8")
    note = root / "tests" / "t.py"
    note.parent.mkdir()
    note.write_text(f'PIN = "{OLD}"\nunrelated = "6ea2650"\n', encoding="utf-8")
    before = (other.read_bytes(), note.read_bytes())
    assert run(bump, ["--tag", NEW, "--write"], FakeGitHub()) == 0
    out = capsys.readouterr().out
    assert "localm/media/comfy_client.py:1:" in out and "tests/t.py:1:" in out
    assert "tests/t.py:2:" not in out
    assert "localm/media/source_archive.py:" not in out, "the archive table is edited, not listed"
    assert (other.read_bytes(), note.read_bytes()) == before


# --------------------------------------------------------------------------- #
#  Refusals                                                                   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tag", ["", "abc1234", "main", NEW[:39], NEW + "0", "g" * 40,
                                 f"{NEW}\n{NEW}", "v1.0"])
def test_a_malformed_sha_is_refused_before_any_request(bump, tree, archives, capsys, tag):
    before = tree.read_bytes()
    fake = FakeGitHub()
    assert run(bump, ["--tag", tag, "--write"], fake) == 1
    assert "full 40-character hex commit sha" in capsys.readouterr().out
    assert fake.requested == [] and tree.read_bytes() == before
    assert archives.requested == []


def test_the_pinned_commit_itself_is_refused_before_any_request(bump, tree, capsys):
    fake = FakeGitHub()
    assert run(bump, ["--tag", OLD, "--write"], fake) == 1
    assert "is the commit already pinned" in capsys.readouterr().out
    assert fake.requested == []


@pytest.mark.parametrize("name, key_fmt, doc, fragment", [
    ("identical", "compare/{old}...{new}", {"status": "identical", "ahead_by": 0, "behind_by": 0},
     "is the commit already pinned"),
    ("behind", "compare/{old}...{new}", {"status": "behind", "ahead_by": 0, "behind_by": 4,
                                         "merge_base_commit": {"sha": NEW}}, "not a descendant"),
    ("diverged", "compare/{old}...{new}", {"status": "diverged", "ahead_by": 2, "behind_by": 1,
                                           "merge_base_commit": {"sha": "d" * 40}}, "not a descendant"),
    ("ahead but with a commit behind", "compare/{old}...{new}",
     {"status": "ahead", "ahead_by": 2, "behind_by": 1, "merge_base_commit": {"sha": OLD}},
     "not a descendant"),
    ("ahead by zero", "compare/{old}...{new}",
     {"status": "ahead", "ahead_by": 0, "behind_by": 0, "merge_base_commit": {"sha": OLD}},
     "not a descendant"),
    ("another merge base", "compare/{old}...{new}",
     {"status": "ahead", "ahead_by": 1, "behind_by": 0, "merge_base_commit": {"sha": MID},
      "commits": [{"sha": NEW}]}, "merge base of the compare"),
    ("another tip", "compare/{old}...{new}",
     {"status": "ahead", "ahead_by": 1, "behind_by": 0, "merge_base_commit": {"sha": OLD},
      "commits": [{"sha": MID}]}, "lists a different tip"),
    ("not an object", "compare/{old}...{new}", ["x"], "is not an object"),
    ("fork-only commit", "compare/{new}...main", {"status": "diverged", "ahead_by": 5, "behind_by": 1},
     "not reachable from"),
    ("branch behind the commit", "compare/{new}...main", {"status": "behind", "ahead_by": 0,
                                                          "behind_by": 2}, "not reachable from"),
    ("no default branch", "", {"default_branch": None}, "reports no default branch"),
])
def test_a_commit_that_is_not_a_clean_descendant_is_refused(
        bump, tree, archives, capsys, name, key_fmt, doc, fragment):
    before = tree.read_bytes()
    key = "repos/city96/ComfyUI-GGUF" + (("/" + key_fmt.format(old=OLD, new=NEW)) if key_fmt else "")
    fake = FakeGitHub(**{key: doc})
    assert run(bump, ["--tag", NEW, "--write"], fake) == 1
    assert fragment in capsys.readouterr().out
    assert tree.read_bytes() == before
    assert archives.requested == [], "a refused commit is never downloaded"


def test_a_commit_github_does_not_have_is_refused_with_a_clear_message(bump):
    def opener(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
    with pytest.raises(bump.Refused, match="no such object"):
        bump.fetch_json("https://api.github.com/repos/o/r/compare/a...b", opener=opener)


def test_other_http_errors_and_garbage_are_refused(bump):
    def forbidden(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 403, "rate limited", {}, None)
    with pytest.raises(bump.Refused, match="HTTP 403"):
        bump.fetch_json("https://api.github.com/x", opener=forbidden)

    class R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=-1):
            return b"<html>"
    with pytest.raises(bump.Refused, match="could not read"):
        bump.fetch_json("https://api.github.com/x", opener=lambda req, t: R())


def test_the_token_is_sent_only_as_a_bearer_header(bump, monkeypatch):
    seen = {}

    class R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=-1):
            return b"{}"

    def opener(req, timeout):
        seen.update(req.headers)
        seen["url"] = req.full_url
        return R()
    monkeypatch.setenv("GITHUB_TOKEN", "t0ken")
    bump.fetch_json("https://api.github.com/x", opener=opener)
    assert seen["Authorization"] == "Bearer t0ken" and "t0ken" not in seen["url"]


# --------------------------------------------------------------------------- #
#  The pin region                                                             #
# --------------------------------------------------------------------------- #

def test_the_pin_must_exist_exactly_once(bump):
    with pytest.raises(bump.Refused, match="found 0"):
        bump.read_pin("x = 1\n")
    with pytest.raises(bump.Refused, match="found 2"):
        bump.read_pin(PIN_FIXTURE + PIN_FIXTURE)


def test_a_pin_on_another_host_is_refused(bump):
    with pytest.raises(bump.Refused, match="github.com repository"):
        bump.read_pin(PIN_FIXTURE.replace("https://github.com/", "https://gitlab.com/"))


def test_only_the_commit_inside_the_pin_changes(bump):
    text = PIN_FIXTURE + f'\nother = CustomNodePin(commit="{OLD}")\n'
    new = bump.rewrite(text, NEW)
    assert new.count(NEW) == 1 and new.count(OLD) == 1
    assert 'name="ComfyUI-GGUF"' in new and "city96/ComfyUI-GGUF.git" in new


def test_the_pin_on_the_real_tree_parses(bump):
    text, _ = bump._read(_ROOT / bump.PIN_REL)
    slug, commit = bump.read_pin(text)
    assert slug == "city96/ComfyUI-GGUF"
    assert len(commit) == 40
    assert bump.rewrite(text, NEW).count(NEW) == 1
