# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/bump_cuda_runtime_pin.py: advancing the pinned Linux CUDA runtime wheels.

Offline: PyPI is replaced by an in-memory fake that answers in PyPI's real JSON
shape. The tests cover:

  * a dry run changes nothing; --write moves exactly the named packages' version
    and wheel sha256 and no other byte of cuda.py;
  * a refusal edits nothing: a same, older or cross-major version, an unpinned or
    repeated package, a PyPI answer for another package or version, a yanked
    release, zero or several Linux x86_64 wheels, a missing digest;
  * the pin block and every entry must be found exactly once;
  * the block this script edits parses on the real tree.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "bump_cuda_runtime_pin.py"

SHA_A = "25bba2dfb01d48a9b59ca474a1ac43c6ebf7011f1b0b8cc44f54eb6ac48a96c3"
SHA_B = "e4f53a8ca8c5d6e8c492d0d0a3d565ecb59a751b19cfdaa4f6da0ab2104c1702"
SHA_C = "9641f797da20ce1dd8e779b6e96d08cf9ba564cec8e8225458811ee26423f3a5"
SHA_D = "c11a27fd4379510e5b1f84b367a2514d1e52fe5cc13442117a0e0a1addee3cf2"
NEW1, NEW2, NEW3, NEW4 = ("1" * 64, "2" * 64, "3" * 64, "4" * 64)

CUDA_FIXTURE = f'''import json

_CUDA_RUNTIME_PYPI_PACKAGES = {{
    "cuda-12": ("nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"),
    "cuda-13": ("nvidia-cuda-runtime", "nvidia-cublas"),
}}

# package -> (version, sha256 of its Linux x86_64 wheel).
_CUDA_RUNTIME_PIN = {{
    "nvidia-cuda-runtime-cu12": ("12.9.79",
        "{SHA_A}"),
    "nvidia-cublas-cu12": ("12.9.2.10",
        "{SHA_B}"),
    "nvidia-cuda-runtime": ("13.4.92",
        "{SHA_C}"),
    "nvidia-cublas": ("13.8.1.7",
        "{SHA_D}"),
}}


def after():
    return 12.9
'''

TEST_FIXTURE = f'''def test_pin():
    assert sl.x == "{SHA_A}"
    fixture = "libcudart.so.12.9.79"
'''


def _load():
    spec = importlib.util.spec_from_file_location("bump_cuda_runtime_pin", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bump():
    return _load()


@pytest.fixture
def tree(bump, tmp_path, monkeypatch):
    cuda = tmp_path / bump.CUDA_REL
    test = tmp_path / bump.TEST_REL
    cuda.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    cuda.write_text(CUDA_FIXTURE, encoding="utf-8", newline="\n")
    test.write_text(TEST_FIXTURE, encoding="utf-8", newline="\n")
    monkeypatch.setattr(bump, "REPO", tmp_path)

    def no_network(req, timeout):
        raise AssertionError("a test reached the network")
    monkeypatch.setattr(bump, "_default_open", no_network)
    return cuda, test


def wheel_names(pkg, ver):
    stem = pkg.replace("-", "_")
    return [f"{stem}-{ver}-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl",
            f"{stem}-{ver}-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
            f"{stem}-{ver}-py3-none-win_amd64.whl"]


def release_doc(pkg, ver, sha, **info):
    names = wheel_names(pkg, ver)
    urls = [{"filename": names[0], "digests": {"sha256": "a" * 64}, "yanked": False},
            {"filename": names[1], "digests": {"sha256": sha}, "yanked": False},
            {"filename": names[2], "digests": {"sha256": "b" * 64}, "yanked": False},
            {"filename": f"{pkg.replace('-', '_')}-{ver}.tar.gz",
             "digests": {"sha256": "c" * 64}, "yanked": False}]
    return {"info": {"name": pkg, "version": ver, "yanked": False, **info}, "urls": urls}


class FakePyPI:
    def __init__(self, docs):
        self.docs, self.requested = docs, []

    def __call__(self, url):
        self.requested.append(url)
        m = re.fullmatch(r"https://pypi\.org/pypi/([^/]+)/([^/]+)/json", url)
        assert m, url
        return self.docs[(m.group(1), m.group(2))]


def run(bump, tag, fake, *extra):
    return bump.main(["--tag", tag, *extra], fetch_json_fn=fake)


ALL_NEW = ("nvidia-cuda-runtime-cu12==12.9.80,nvidia-cublas-cu12==12.9.3.1,"
           "nvidia-cuda-runtime==13.4.93,nvidia-cublas==13.8.2.0")


def all_docs():
    return {("nvidia-cuda-runtime-cu12", "12.9.80"): release_doc("nvidia-cuda-runtime-cu12", "12.9.80", NEW1),
            ("nvidia-cublas-cu12", "12.9.3.1"): release_doc("nvidia-cublas-cu12", "12.9.3.1", NEW2),
            ("nvidia-cuda-runtime", "13.4.93"): release_doc("nvidia-cuda-runtime", "13.4.93", NEW3),
            ("nvidia-cublas", "13.8.2.0"): release_doc("nvidia-cublas", "13.8.2.0", NEW4)}


# --------------------------------------------------------------------------- #
#  Versions                                                                   #
# --------------------------------------------------------------------------- #

def test_versions_compare_numerically_across_component_counts(bump):
    assert bump.is_newer("12.10.0", "12.9.79")
    assert bump.is_newer("12.9.2.11", "12.9.2.10")
    assert bump.is_newer("12.9.3", "12.9.2.10")
    assert bump.is_newer("12.9.2.10.1", "12.9.2.10")
    assert not bump.is_newer("12.9.2.10", "12.9.2.10")
    assert not bump.is_newer("12.9.2", "12.9.2.0"), "trailing zeros are the same version"
    assert not bump.is_newer("12.9.9", "12.9.10")


@pytest.mark.parametrize("tag", ["", "  ", "nvidia-cublas", "nvidia-cublas==", "==1.0",
                                 "nvidia-cublas==1.0,", "nvidia-cublas>=1.0", "a==1==2",
                                 "nvidia-cublas==1.0rc1", "nvidia-cublas==v1.0",
                                 "nvidia-cublas==1.0,nvidia-cublas==1.1", "pk g==1.0",
                                 "nvidia-cublas==1"])
def test_a_malformed_request_is_refused(bump, tag):
    with pytest.raises(bump.Refused):
        bump.parse_versions(tag)


def test_a_well_formed_request_parses(bump):
    assert bump.parse_versions(" nvidia-cublas==13.8.2.0 , nvidia-cublas-cu12==12.9.3.1 ") == {
        "nvidia-cublas": "13.8.2.0", "nvidia-cublas-cu12": "12.9.3.1"}


# --------------------------------------------------------------------------- #
#  The edit                                                                   #
# --------------------------------------------------------------------------- #

def test_dry_run_changes_nothing_and_prints_the_diff(bump, tree, capsys):
    cuda, test = tree
    before = (cuda.read_bytes(), test.read_bytes())
    fake = FakePyPI(all_docs())
    assert run(bump, ALL_NEW, fake) == 0
    out = capsys.readouterr().out
    assert '+    "nvidia-cublas": ("13.8.2.0",' in out and f'+        "{NEW4}"),' in out
    assert "dry run: nothing written" in out and "REMAINING STEPS" in out
    assert (cuda.read_bytes(), test.read_bytes()) == before
    assert fake.requested == [
        "https://pypi.org/pypi/nvidia-cuda-runtime-cu12/12.9.80/json",
        "https://pypi.org/pypi/nvidia-cublas-cu12/12.9.3.1/json",
        "https://pypi.org/pypi/nvidia-cuda-runtime/13.4.93/json",
        "https://pypi.org/pypi/nvidia-cublas/13.8.2.0/json"]


def test_write_moves_every_named_pin_and_nothing_else(bump, tree, capsys):
    cuda, test = tree
    assert run(bump, ALL_NEW, FakePyPI(all_docs()), "--write") == 0
    expected = CUDA_FIXTURE
    for old, new in (("12.9.79", "12.9.80"), (SHA_A, NEW1), ("12.9.2.10", "12.9.3.1"), (SHA_B, NEW2),
                     ("13.4.92", "13.4.93"), (SHA_C, NEW3), ("13.8.1.7", "13.8.2.0"), (SHA_D, NEW4)):
        assert expected.count(old) == 1
        expected = expected.replace(old, new)
    assert cuda.read_text(encoding="utf-8") == expected
    assert test.read_text(encoding="utf-8") == TEST_FIXTURE.replace(SHA_A, NEW1)

    capsys.readouterr()
    assert run(bump, ALL_NEW, FakePyPI(all_docs()), "--write") == 1
    assert "nothing to bump" in capsys.readouterr().out


def test_the_full_list_with_unchanged_packages_moves_only_the_changed_ones(bump, tree, capsys):
    cuda, _ = tree
    docs = {k: v for k, v in all_docs().items() if k[0] == "nvidia-cublas"}
    full = ("nvidia-cuda-runtime-cu12==12.9.79,nvidia-cublas-cu12==12.9.2.10,"
            "nvidia-cuda-runtime==13.4.92,nvidia-cublas==13.8.2.0")
    fake = FakePyPI(docs)
    assert run(bump, full, fake, "--write") == 0
    assert fake.requested == ["https://pypi.org/pypi/nvidia-cublas/13.8.2.0/json"]
    assert cuda.read_text(encoding="utf-8") == CUDA_FIXTURE.replace("13.8.1.7", "13.8.2.0").replace(SHA_D, NEW4)
    assert "kept  nvidia-cuda-runtime-cu12==12.9.79" in capsys.readouterr().out


def test_an_older_package_in_the_full_list_refuses_the_whole_request(bump, tree, capsys):
    cuda, _ = tree
    before = cuda.read_bytes()
    fake = FakePyPI(all_docs())
    assert run(bump, "nvidia-cuda-runtime-cu12==12.9.1,nvidia-cublas==13.8.2.0", fake, "--write") == 1
    assert "not newer than the pinned 12.9.79" in capsys.readouterr().out
    assert fake.requested == [] and cuda.read_bytes() == before


def test_a_subset_moves_only_those_packages(bump, tree, capsys):
    cuda, _ = tree
    docs = {k: v for k, v in all_docs().items() if k[0] == "nvidia-cublas-cu12"}
    assert run(bump, "nvidia-cublas-cu12==12.9.3.1", FakePyPI(docs), "--write") == 0
    text = cuda.read_text(encoding="utf-8")
    assert text == CUDA_FIXTURE.replace("12.9.2.10", "12.9.3.1").replace(SHA_B, NEW2)
    out = capsys.readouterr().out
    assert "kept  nvidia-cuda-runtime-cu12==12.9.79" in out
    assert "moved nvidia-cublas-cu12==12.9.3.1" in out


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_the_file_keeps_its_own_line_endings(bump, tree, newline):
    cuda, _ = tree
    cuda.write_bytes(CUDA_FIXTURE.replace("\n", newline).encode("utf-8"))
    docs = {k: v for k, v in all_docs().items() if k[0] == "nvidia-cublas"}
    assert run(bump, "nvidia-cublas==13.8.2.0", FakePyPI(docs), "--write") == 0
    data = cuda.read_bytes()
    assert data.count(newline.encode()) == data.count(b"\n")
    assert (b"\r" in data) == (newline == "\r\n")
    assert b"13.8.2.0" in data


def test_a_missing_test_file_is_not_an_error(bump, tree):
    _, test = tree
    test.unlink()
    docs = {k: v for k, v in all_docs().items() if k[0] == "nvidia-cublas"}
    assert run(bump, "nvidia-cublas==13.8.2.0", FakePyPI(docs), "--write") == 0


def test_the_checklist_lists_test_lines_that_name_an_old_version(bump, tree, capsys):
    docs = {k: v for k, v in all_docs().items() if k[0] == "nvidia-cuda-runtime-cu12"}
    assert run(bump, "nvidia-cuda-runtime-cu12==12.9.80", FakePyPI(docs)) == 0
    out = capsys.readouterr().out
    assert 'fixture = "libcudart.so.12.9.79"' in out
    assert "moved nvidia-cuda-runtime-cu12==12.9.80" in out


# --------------------------------------------------------------------------- #
#  Refusals                                                                   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tag, fragment", [
    ("nvidia-cuda-runtime-cu12==12.9.79", "nothing to bump"),
    ("nvidia-cuda-runtime-cu12==12.9.79,nvidia-cublas==13.8.1.7", "nothing to bump"),
    ("nvidia-cuda-runtime-cu12==12.9.78", "not newer than the pinned 12.9.79"),
    ("nvidia-cublas-cu12==12.9.2.9", "not newer than the pinned 12.9.2.10"),
    ("nvidia-cuda-runtime-cu12==13.0.0", "leaves the 12.x line"),
    ("nvidia-cublas-cu12==13.8.2.0", "leaves the 12.x line"),
    ("nvidia-cuda-runtime==14.0.0", "leaves the 13.x line"),
    ("nvidia-cublas==12.9.9", "leaves the 13.x line"),
    ("nvidia-cublas-cu11==11.1.0", "is not pinned in _CUDA_RUNTIME_PIN"),
    ("nvidia-nccl-cu12==2.0.0", "is not pinned in _CUDA_RUNTIME_PIN"),
    ("nvidia-cublas==13.9.0,nvidia-cublas==13.9.1", "appears more than once"),
    ("nvidia-cublas==13.8", "not newer than the pinned 13.8.1.7"),
])
def test_a_bad_move_is_refused_before_any_request_and_edits_nothing(
        bump, tree, capsys, tag, fragment):
    cuda, test = tree
    before = (cuda.read_bytes(), test.read_bytes())
    fake = FakePyPI({})
    assert run(bump, tag, fake, "--write") == 1
    assert fragment in capsys.readouterr().out
    assert fake.requested == []
    assert (cuda.read_bytes(), test.read_bytes()) == before


@pytest.mark.parametrize("mutate, fragment", [
    (lambda d: d["info"].update(name="nvidia-cublas-cu12"), "PyPI answered with"),
    (lambda d: d["info"].update(version="13.8.1.9"), "PyPI answered with"),
    (lambda d: d["info"].update(yanked=True), "is yanked on PyPI"),
    (lambda d: d["urls"].__setitem__(1, {**d["urls"][1], "yanked": True}), "wheel is yanked"),
    (lambda d: d["urls"].__delitem__(1), "found 0"),
    (lambda d: d["urls"].append({**d["urls"][1], "filename": "x-1-py3-none-manylinux_2_28_x86_64.whl"}),
     "found 2"),
    (lambda d: d["urls"][1].pop("digests"), "has no sha256 digest"),
    (lambda d: d["urls"][1].update(digests={"sha256": "XYZ"}), "has no sha256 digest"),
    (lambda d: d["urls"][1].update(digests={"sha256": "A" * 64}), "has no sha256 digest"),
    (lambda d: d.pop("urls"), "lists no files"),
    (lambda d: d.pop("info"), "not a release document"),
])
def test_a_pypi_answer_that_does_not_check_out_is_refused(bump, tree, capsys, mutate, fragment):
    cuda, test = tree
    before = (cuda.read_bytes(), test.read_bytes())
    docs = {("nvidia-cublas", "13.8.2.0"): release_doc("nvidia-cublas", "13.8.2.0", NEW4)}
    mutate(docs[("nvidia-cublas", "13.8.2.0")])
    assert run(bump, "nvidia-cublas==13.8.2.0", FakePyPI(docs), "--write") == 1
    assert fragment in capsys.readouterr().out
    assert (cuda.read_bytes(), test.read_bytes()) == before


def test_a_pypi_answer_that_is_not_an_object_is_refused(bump, tree, capsys):
    assert run(bump, "nvidia-cublas==13.8.2.0",
               FakePyPI({("nvidia-cublas", "13.8.2.0"): ["x"]})) == 1
    assert "not a release document" in capsys.readouterr().out


def test_an_unreachable_pypi_is_refused(bump, tree, capsys):
    def down(url):
        raise bump.Refused("could not read the url: URLError: down")
    assert run(bump, "nvidia-cublas==13.8.2.0", down, "--write") == 1
    assert "could not read" in capsys.readouterr().out


def test_the_underscore_spelling_pypi_normalises_to_is_accepted(bump):
    doc = release_doc("nvidia-cublas", "13.8.2.0", NEW4, name="nvidia_cublas")
    assert bump.fetch_wheel_sha("nvidia-cublas", "13.8.2.0", lambda url: doc) == NEW4


# --------------------------------------------------------------------------- #
#  Regions found exactly once                                                 #
# --------------------------------------------------------------------------- #

def test_the_pin_block_must_exist_exactly_once(bump):
    with pytest.raises(bump.Refused, match="found 0"):
        bump.read_pin("x = 1\n")
    with pytest.raises(bump.Refused, match="found 2"):
        bump.read_pin(CUDA_FIXTURE + CUDA_FIXTURE)


def test_an_entry_listed_twice_is_refused(bump):
    doubled = CUDA_FIXTURE.replace(
        '    "nvidia-cublas": ("13.8.1.7",',
        f'    "nvidia-cublas-cu12": ("12.9.2.10",\n        "{SHA_B}"),\n    "nvidia-cublas": ("13.8.1.7",')
    with pytest.raises(bump.Refused, match="more than once"):
        bump.read_pin(doubled)


def test_an_entry_outside_the_block_is_not_edited(bump):
    outside = CUDA_FIXTURE + f'\n_OTHER = {{\n    "nvidia-cublas": ("13.8.1.7",\n        "{SHA_D}"),\n}}\n'
    with pytest.raises(bump.Refused, match="expected exactly one pin entry"):
        bump.rewrite_pin(outside, "nvidia-cublas", "13.9.0.1", NEW4)


def test_a_digest_repeated_in_the_test_file_is_refused(bump):
    with pytest.raises(bump.Refused, match="carries the old digest 2 times"):
        bump.rewrite_test_sha(f'"{SHA_A}"\n"{SHA_A}"\n', SHA_A, NEW1)


def test_the_pin_entries_may_sit_on_one_line(bump):
    one = f'_CUDA_RUNTIME_PIN = {{\n    "p": ("1.0", "{SHA_A}"),\n}}\n'
    assert bump.read_pin(one) == {"p": ("1.0", SHA_A)}
    assert bump.rewrite_pin(one, "p", "1.1", NEW1) == f'_CUDA_RUNTIME_PIN = {{\n    "p": ("1.1", "{NEW1}"),\n}}\n'


# --------------------------------------------------------------------------- #
#  The real tree                                                              #
# --------------------------------------------------------------------------- #

def test_the_pin_block_the_script_edits_parses_on_the_real_tree(bump):
    text, _ = bump._read(_ROOT / bump.CUDA_REL)
    pin = bump.read_pin(text)
    names = {p for pkgs in re.findall(r'\(("[^)]*")\)',
                                      re.search(r"_CUDA_RUNTIME_PYPI_PACKAGES = \{.*?\n\}", text, re.S).group(0))
             for p in re.findall(r'"([^"]+)"', pkgs)}
    assert set(pin) == names
    for package, (version, sha) in pin.items():
        assert re.fullmatch(r"\d+(\.\d+)+", version), package
        assert re.fullmatch(r"[0-9a-f]{64}", sha), package
        assert bump.rewrite_pin(text, package, version + ".1", "f" * 64) != text
