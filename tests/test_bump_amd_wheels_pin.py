# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/bump_amd_wheels_pin.py: advancing the AMD ROCm torch and rocm-sdk pins.

Offline: AMD's wheel index is an in-memory fake that answers in the real PEP 503
shape, and `uv lock` is replaced by a stand-in that rewrites the locked versions
the way uv does. One test runs the real `uv` on a trivial project to prove the
invocation shape and exit codes. The tests cover:

  * a dry run changes nothing; --write moves the four pyproject regions and
    replaces uv.lock, keeping each file's line endings;
  * a refusal edits nothing: a malformed or partial request, a package moving
    backwards, nothing moving, torch and torchvision on different ROCm releases,
    a wheel AMD does not list, a uv failure, a lock that does not record the
    requested versions, a lock that fails `uv lock --check`;
  * the regions the script edits are found exactly once, also on the real tree.
"""

import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "bump_amd_wheels_pin.py"

BASE = "https://repo.amd.com/rocm/whl/gfx103X-all"

PYPROJECT = f'''[project]
name = "localm"
dependencies = [
    "click>=8.5.0",
]

[project.optional-dependencies]
gpu = [
    "torch==2.10.0+rocm7.13.0; sys_platform == 'win32'",
    "torchvision==0.25.0+rocm7.13.0; sys_platform == 'win32'",
    "rocm; sys_platform == 'win32'",
    "rocm-sdk-core; sys_platform == 'win32'",
    "rocm-sdk-libraries-gfx103x-all; sys_platform == 'win32'",
]

[tool.uv.sources]
torch = [
  {{ url = "{BASE}/torch-2.10.0%2Brocm7.13.0-cp312-cp312-win_amd64.whl", marker = "sys_platform == 'win32' and python_version == '3.12'" }},
]
torchvision = [
  {{ url = "{BASE}/torchvision-0.25.0%2Brocm7.13.0-cp312-cp312-win_amd64.whl", marker = "sys_platform == 'win32' and python_version == '3.12'" }},
]
rocm-sdk-core = [
  {{ index = "rocm-gfx1030", marker = "sys_platform == 'win32'" }},
]
'''

LOCK = f'''version = 1
requires-python = ">=3.12, <3.13"

[[package]]
name = "rocm"
version = "7.13.0"
source = {{ url = "{BASE}/rocm-7.13.0.tar.gz" }}

[[package]]
name = "rocm-sdk-core"
version = "7.13.0"
source = {{ registry = "{BASE}/" }}

[[package]]
name = "rocm-sdk-libraries-gfx103x-all"
version = "7.13.0"
source = {{ registry = "{BASE}/" }}

[[package]]
name = "sympy"
version = "1.14.0"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "torch"
version = "2.13.0"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "torch"
version = "2.10.0+rocm7.13.0"
source = {{ url = "{BASE}/torch-2.10.0%2Brocm7.13.0-cp312-cp312-win_amd64.whl" }}

[[package]]
name = "torchvision"
version = "0.25.0+rocm7.13.0"
source = {{ url = "{BASE}/torchvision-0.25.0%2Brocm7.13.0-cp312-cp312-win_amd64.whl" }}
'''

NEW = {"torch": "2.11.0+rocm7.13.0", "torchvision": "0.26.0+rocm7.13.0",
       "rocm-sdk-core": "7.13.0", "rocm-sdk-libraries-gfx103x-all": "7.13.0"}


def tag(**overrides) -> str:
    return ",".join(f"{p}=={v}" for p, v in {**NEW, **overrides}.items())


def _load():
    spec = importlib.util.spec_from_file_location("bump_amd_wheels_pin", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bump():
    return _load()


@pytest.fixture
def tree(bump, tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8", newline="\n")
    (tmp_path / "uv.lock").write_text(LOCK, encoding="utf-8", newline="\n")
    monkeypatch.setattr(bump, "REPO", tmp_path)

    def no_network(req, timeout):
        raise AssertionError("a test reached the network")
    monkeypatch.setattr(bump, "_default_open", no_network)
    return tmp_path


def index_page(names):
    links = "".join(f'<a href="{BASE}/{n.replace("+", "%2B")}#sha256=00">{n}</a><br/>'
                    for n in names)
    return f"<html><body>{links}</body></html>"


class FakeIndex:
    """AMD's index pages: every wheel the NEW request names, plus older torch builds."""

    def __init__(self, drop=(), extra=()):
        from_new = {
            "torch": ["torch-2.10.0+rocm7.13.0-cp312-cp312-win_amd64.whl",
                      "torch-2.11.0+rocm7.13.0-cp312-cp312-win_amd64.whl"],
            "torchvision": ["torchvision-0.25.0+rocm7.13.0-cp312-cp312-win_amd64.whl",
                            "torchvision-0.26.0+rocm7.13.0-cp312-cp312-win_amd64.whl"],
            "rocm-sdk-core": ["rocm_sdk_core-7.13.0-py3-none-win_amd64.whl",
                              "rocm_sdk_core-7.14.0-py3-none-win_amd64.whl"],
            "rocm-sdk-libraries-gfx103x-all": [
                "rocm_sdk_libraries_gfx103x_all-7.13.0-py3-none-win_amd64.whl",
                "rocm_sdk_libraries_gfx103x_all-7.14.0-py3-none-win_amd64.whl"],
            "rocm": ["rocm-7.13.0.tar.gz", "rocm-7.14.0.tar.gz"],
        }
        self.pages = {pkg: [n for n in names if n not in drop] + list(extra)
                      for pkg, names in from_new.items()}
        self.requested = []

    def __call__(self, url):
        self.requested.append(url)
        m = re.fullmatch(re.escape(BASE) + r"/([^/]+)/", url)
        assert m, url
        return index_page(self.pages[m.group(1)])


class FakeUv:
    """Stands in for `uv lock` / `uv lock --check`: rewrites the locked versions from the
    scratch pyproject.toml and --upgrade-package arguments, as uv does."""

    def __init__(self, check_code=0, lock_code=0, drift=None):
        self.calls, self.check_code, self.lock_code, self.drift = [], check_code, lock_code, drift

    def __call__(self, args, directory):
        self.calls.append((list(args), sorted(p.name for p in Path(directory).iterdir())))
        if args == ["lock", "--check"]:
            return self.check_code, "" if self.check_code == 0 else "lockfile out of date"
        if self.lock_code:
            return self.lock_code, "No solution found when resolving dependencies"
        py = (Path(directory) / "pyproject.toml").read_text(encoding="utf-8")
        lock = (Path(directory) / "uv.lock").read_text(encoding="utf-8")
        torch = re.search(r'"torch==([^";]+);', py).group(1)
        vision = re.search(r'"torchvision==([^";]+);', py).group(1)
        lock = re.sub(r'(name = "torch"\nversion = ")[^"]*\+rocm[^"]*(")',
                      lambda m: m.group(1) + torch + m.group(2), lock)
        lock = re.sub(r'(name = "torchvision"\nversion = ")[^"]*(")',
                      lambda m: m.group(1) + vision + m.group(2), lock)
        for i, arg in enumerate(args):
            if arg == "--upgrade-package":
                pkg, _, ver = args[i + 1].partition("==")
                lock = re.sub(rf'(name = "{re.escape(pkg)}"\nversion = ")[^"]*(")',
                              lambda m, v=ver: m.group(1) + v + m.group(2), lock)
        if self.drift:
            lock = lock.replace(*self.drift)
        (Path(directory) / "uv.lock").write_text(lock, encoding="utf-8", newline="\n")
        return 0, "Resolved 159 packages"


def run(bump, argv, index=None, uv=None):
    return bump.main(argv, fetch_text_fn=index or FakeIndex(), run_uv_fn=uv or FakeUv())


# --------------------------------------------------------------------------- #
#  The request                                                                #
# --------------------------------------------------------------------------- #

def test_a_complete_request_parses(bump):
    assert bump.parse_request(tag()) == NEW


@pytest.mark.parametrize("text", [
    "", "torch==2.11.0+rocm7.13.0",
    tag() + ",torch==2.12.0+rocm7.13.0",
    tag() + ",rocm==7.13.0",
    tag(torch="2.11.0"), tag(torch="2.11.0+cu126"), tag(torch="v2.11.0+rocm7.13.0"),
    tag(**{"rocm-sdk-core": "7.13.0+rocm7.13.0"}), tag(**{"rocm-sdk-core": "latest"}),
    tag() + ",", tag().replace("==", "=", 1), "torch>=2.11.0",
])
def test_a_partial_or_malformed_request_is_refused(bump, text):
    with pytest.raises(bump.Refused):
        bump.parse_request(text)


def test_versions_order_by_base_then_rocm_release(bump):
    key = bump.sort_key
    assert key("torch", "2.11.0+rocm7.13.0") > key("torch", "2.10.0+rocm7.13.0")
    assert key("torch", "2.10.0+rocm7.14.0") > key("torch", "2.10.0+rocm7.13.0")
    assert key("torch", "2.10.0+rocm7.13") == key("torch", "2.10.0+rocm7.13.0")
    assert key("torch", "2.9.1+rocm7.14.0") < key("torch", "2.10.0+rocm7.13.0")
    assert key("rocm-sdk-core", "7.13.1") > key("rocm-sdk-core", "7.13.0")
    assert key("rocm-sdk-core", "7.13") == key("rocm-sdk-core", "7.13.0")


def test_plan_moves_reports_only_the_packages_that_move(bump):
    current = dict(NEW, torch="2.10.0+rocm7.13.0", torchvision="0.25.0+rocm7.13.0")
    assert bump.plan_moves(current, NEW) == {"torch": NEW["torch"],
                                             "torchvision": NEW["torchvision"]}
    assert bump.plan_moves(dict(NEW, torch="2.10.0+rocm7.13.0"), NEW) == {"torch": NEW["torch"]}


@pytest.mark.parametrize("current_override, wanted_override, fragment", [
    ({}, {}, "nothing to bump"),
    ({"torch": "2.12.0+rocm7.13.0"}, {}, "older than the pinned"),
    ({"rocm-sdk-core": "7.14.0", "rocm-sdk-libraries-gfx103x-all": "7.14.0"}, {"torch": "2.12.0+rocm7.13.0",
                                                                                "torchvision": "0.27.0+rocm7.13.0"},
     "older than the pinned"),
    ({"torch": "2.10.0+rocm7.13.0"}, {"torchvision": "0.26.0+rocm7.14.0"},
     "different ROCm releases"),
    ({"torch": "2.10.0+rocm7.13.0"}, {"rocm-sdk-core": "7.14.0"}, "share one version"),
])
def test_plan_moves_refuses_a_backward_stalled_or_split_request(
        bump, current_override, wanted_override, fragment):
    current = {**NEW, **current_override}
    with pytest.raises(bump.Refused, match=fragment):
        bump.plan_moves(current, {**NEW, **wanted_override})


# --------------------------------------------------------------------------- #
#  The tree                                                                   #
# --------------------------------------------------------------------------- #

def test_the_current_pins_are_read_from_pyproject_and_the_lock(bump):
    assert bump.read_current(PYPROJECT, LOCK) == {
        "torch": "2.10.0+rocm7.13.0", "torchvision": "0.25.0+rocm7.13.0",
        "rocm-sdk-core": "7.13.0", "rocm-sdk-libraries-gfx103x-all": "7.13.0"}


@pytest.mark.parametrize("mutate, fragment", [
    (lambda py, lock: (py.replace('"torch==2.10.0+rocm7.13.0;', '"torch==2.9.1+rocm7.13.0;'), lock),
     "the tree is inconsistent"),
    (lambda py, lock: (py.replace('"torchvision==0.25.0+rocm7.13.0; sys_platform == \'win32\'",\n', ""), lock),
     "torchvision == pin: expected exactly one match, found 0"),
    (lambda py, lock: (py + '\n"torch==2.10.0+rocm7.13.0; sys_platform == \'win32\'",\n', lock),
     "torch == pin: expected exactly one match, found 2"),
    (lambda py, lock: (py.replace("torch-2.10.0%2B", "torch-2.9.1%2B"), lock),
     "the tree is inconsistent"),
    (lambda py, lock: (py.replace("-cp312-cp312-win_amd64.whl\", marker", ".whl\", marker", 1), lock),
     "torch wheel URL: expected exactly one match, found 0"),
    (lambda py, lock: (py, lock + '\n[[package]]\nname = "rocm-sdk-core"\nversion = "7.12.0"\n'),
     "records 2 versions"),
    (lambda py, lock: (py, lock.replace('name = "rocm-sdk-core"', 'name = "rocm-sdk-corex"')),
     "records 0 versions"),
])
def test_a_tree_that_is_not_in_the_edited_shape_is_refused(bump, mutate, fragment):
    py, lock = mutate(PYPROJECT, LOCK)
    with pytest.raises(bump.Refused, match=fragment):
        bump.read_current(py, lock)


def test_the_rewrite_moves_the_four_regions_and_nothing_else(bump):
    new = bump.rewrite_pyproject(PYPROJECT, NEW)
    old_lines, new_lines = PYPROJECT.split("\n"), new.split("\n")
    assert len(old_lines) == len(new_lines)
    changed = [(a, b) for a, b in zip(old_lines, new_lines, strict=True) if a != b]
    assert len(changed) == 4
    assert '"torch==2.11.0+rocm7.13.0; sys_platform == \'win32\'",' in new
    assert '"torchvision==0.26.0+rocm7.13.0; sys_platform == \'win32\'",' in new
    assert f"{BASE}/torch-2.11.0%2Brocm7.13.0-cp312-cp312-win_amd64.whl" in new
    assert f"{BASE}/torchvision-0.26.0%2Brocm7.13.0-cp312-cp312-win_amd64.whl" in new
    assert "2.10.0" not in new and "0.25.0" not in new


def test_changed_packages_lists_every_locked_difference(bump):
    new = LOCK.replace("2.10.0+rocm7.13.0", "2.11.0+rocm7.13.0").replace("sympy", "sympyx")
    lines = bump.changed_packages(LOCK, new)
    assert "torch: 2.13.0, 2.10.0+rocm7.13.0 -> 2.13.0, 2.11.0+rocm7.13.0" in lines
    assert "sympy: 1.14.0 -> (removed)" in lines and "sympyx: (absent) -> 1.14.0" in lines


# --------------------------------------------------------------------------- #
#  The index                                                                  #
# --------------------------------------------------------------------------- #

def test_the_index_check_reads_percent_encoded_names_and_fragments(bump):
    names = bump.index_filenames("torch", FakeIndex())
    assert "torch-2.11.0+rocm7.13.0-cp312-cp312-win_amd64.whl" in names


@pytest.mark.parametrize("drop, fragment", [
    ("torch-2.11.0+rocm7.13.0-cp312-cp312-win_amd64.whl", "lists no torch-2.11.0"),
    ("torchvision-0.26.0+rocm7.13.0-cp312-cp312-win_amd64.whl", "lists no torchvision-0.26.0"),
    ("rocm_sdk_core-7.13.0-py3-none-win_amd64.whl", "lists no rocm_sdk_core-7.13.0"),
])
def test_a_wheel_the_index_does_not_list_is_refused(bump, drop, fragment):
    with pytest.raises(bump.Refused, match=fragment):
        bump.verify_on_index(NEW, {"torch": NEW["torch"]}, FakeIndex(drop=(drop,)))


def test_a_cp313_only_torch_is_not_a_cp312_wheel(bump):
    index = FakeIndex(drop=("torch-2.11.0+rocm7.13.0-cp312-cp312-win_amd64.whl",),
                      extra=("torch-2.11.0+rocm7.13.0-cp313-cp313-win_amd64.whl",))
    with pytest.raises(bump.Refused, match="lists no torch-2.11.0"):
        bump.verify_on_index(NEW, {"torch": NEW["torch"]}, index)


def test_the_rocm_sdist_is_checked_only_when_the_sdk_moves(bump):
    no_sdist = FakeIndex(drop=("rocm-7.14.0.tar.gz",))
    wanted = dict(NEW, **{"rocm-sdk-core": "7.14.0", "rocm-sdk-libraries-gfx103x-all": "7.14.0"})
    with pytest.raises(bump.Refused, match="rocm-7.14.0.tar.gz"):
        bump.verify_on_index(wanted, {"rocm-sdk-core": "7.14.0"}, no_sdist)
    bump.verify_on_index(NEW, {"torch": NEW["torch"]}, FakeIndex(drop=("rocm-7.13.0.tar.gz",)))


def test_an_unreadable_index_is_refused(bump):
    def down(url):
        raise bump.Refused("could not read the url: URLError: down")
    with pytest.raises(bump.Refused, match="could not read"):
        bump.verify_on_index(NEW, {"torch": NEW["torch"]}, down)


# --------------------------------------------------------------------------- #
#  The lock                                                                   #
# --------------------------------------------------------------------------- #

def test_the_lock_is_regenerated_in_a_scratch_directory_holding_two_files(bump):
    uv = FakeUv()
    py = bump.rewrite_pyproject(PYPROJECT, NEW)
    new_lock, _ = bump.regenerate_lock(py, LOCK, NEW, {"torch": NEW["torch"]}, uv)
    assert [c[0] for c in uv.calls] == [["lock"], ["lock", "--check"]]
    assert all(c[1] == ["pyproject.toml", "uv.lock"] for c in uv.calls)
    assert 'version = "2.11.0+rocm7.13.0"' in new_lock


def test_an_sdk_move_adds_the_upgrade_arguments_for_all_three_packages(bump):
    wanted = dict(NEW, **{"rocm-sdk-core": "7.14.0", "rocm-sdk-libraries-gfx103x-all": "7.14.0"})
    args = bump.upgrade_args(wanted, {"rocm-sdk-core": "7.14.0"})
    assert args == ["--upgrade-package", "rocm-sdk-core==7.14.0",
                    "--upgrade-package", "rocm-sdk-libraries-gfx103x-all==7.14.0",
                    "--upgrade-package", "rocm==7.14.0"]
    assert bump.upgrade_args(NEW, {"torch": NEW["torch"]}) == []


def test_uv_failure_is_refused_with_its_output(bump):
    with pytest.raises(bump.Refused, match="uv lock failed .exit 2.: No solution found"):
        bump.regenerate_lock(PYPROJECT, LOCK, NEW, {"torch": NEW["torch"]}, FakeUv(lock_code=2))


def test_a_lock_that_does_not_record_the_requested_versions_is_refused(bump):
    py = bump.rewrite_pyproject(PYPROJECT, NEW)
    drift = FakeUv(drift=('version = "0.26.0+rocm7.13.0"', 'version = "0.25.0+rocm7.13.0"'))
    with pytest.raises(bump.Refused, match="holds torchvision"):
        bump.regenerate_lock(py, LOCK, NEW, {"torch": NEW["torch"]}, drift)
    sdk = FakeUv(drift=('name = "rocm-sdk-core"\nversion = "7.13.0"',
                        'name = "rocm-sdk-core"\nversion = "7.12.0"'))
    with pytest.raises(bump.Refused, match="holds rocm-sdk-core"):
        bump.regenerate_lock(py, LOCK, NEW, {"torch": NEW["torch"]}, sdk)
    meta = FakeUv(drift=('name = "rocm"\nversion = "7.13.0"', 'name = "rocm"\nversion = "7.12.0"'))
    with pytest.raises(bump.Refused, match="holds rocm "):
        bump.regenerate_lock(py, LOCK, NEW, {"torch": NEW["torch"]}, meta)


def test_a_lock_that_fails_the_check_is_refused(bump):
    py = bump.rewrite_pyproject(PYPROJECT, NEW)
    with pytest.raises(bump.Refused, match="--check rejects"):
        bump.regenerate_lock(py, LOCK, NEW, {"torch": NEW["torch"]}, FakeUv(check_code=1))


def test_a_missing_uv_is_refused(bump, monkeypatch):
    monkeypatch.setattr(bump.shutil, "which", lambda name: None)
    with pytest.raises(bump.Refused, match="uv is not on PATH"):
        bump.run_uv(["lock"], Path("."))


# --------------------------------------------------------------------------- #
#  main                                                                       #
# --------------------------------------------------------------------------- #

def snapshot(root: Path) -> dict:
    return {p.name: p.read_bytes() for p in sorted(root.iterdir()) if p.is_file()}


def test_dry_run_changes_nothing_and_shows_both_diffs(bump, tree, capsys):
    before = snapshot(tree)
    assert run(bump, ["--tag", tag()]) == 0
    out = capsys.readouterr().out
    assert "moving: torch 2.10.0+rocm7.13.0 -> 2.11.0+rocm7.13.0" in out
    assert '+    "torch==2.11.0+rocm7.13.0; sys_platform == \'win32\'",' in out
    assert "--- a/uv.lock" in out and "dry run: nothing written" in out
    assert "REMAINING STEPS" in out and "_AMD_GFX103X_TORCH" in out
    assert snapshot(tree) == before


def test_write_moves_the_pyproject_regions_and_replaces_the_lock(bump, tree, capsys):
    assert run(bump, ["--tag", tag(), "--write"]) == 0
    assert "wrote pyproject.toml" in capsys.readouterr().out
    assert (tree / "pyproject.toml").read_text(encoding="utf-8") == bump.rewrite_pyproject(PYPROJECT, NEW)
    lock = (tree / "uv.lock").read_text(encoding="utf-8")
    assert lock == LOCK.replace("2.10.0+rocm7.13.0", "2.11.0+rocm7.13.0").replace(
        "0.25.0+rocm7.13.0", "0.26.0+rocm7.13.0")
    assert run(bump, ["--tag", tag(), "--write"]) == 1


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_each_file_keeps_its_own_line_endings(bump, tree, newline):
    for name, text in (("pyproject.toml", PYPROJECT), ("uv.lock", LOCK)):
        (tree / name).write_bytes(text.replace("\n", newline).encode("utf-8"))
    assert run(bump, ["--tag", tag(), "--write"]) == 0
    for name in ("pyproject.toml", "uv.lock"):
        data = (tree / name).read_bytes()
        assert data.count(newline.encode()) == data.count(b"\n")
        assert (b"\r" in data) == (newline == "\r\n")
        assert b"2.11.0" in data or b"0.26.0" in data


@pytest.mark.parametrize("uv", [FakeUv(lock_code=1), FakeUv(check_code=1),
                                FakeUv(drift=('version = "0.26.0+rocm7.13.0"',
                                              'version = "0.25.0+rocm7.13.0"'))],
                         ids=["uv-fails", "check-fails", "lock-drifts"])
def test_a_regeneration_that_does_not_verify_leaves_the_tree_untouched(bump, tree, capsys, uv):
    before = snapshot(tree)
    assert run(bump, ["--tag", tag(), "--write"], uv=uv) == 1
    assert "REFUSED" in capsys.readouterr().out
    assert snapshot(tree) == before


def test_a_wheel_missing_from_the_index_never_reaches_uv(bump, tree, capsys):
    uv = FakeUv()
    index = FakeIndex(drop=("torch-2.11.0+rocm7.13.0-cp312-cp312-win_amd64.whl",))
    assert run(bump, ["--tag", tag(), "--write"], index=index, uv=uv) == 1
    assert uv.calls == []
    assert "lists no torch-2.11.0" in capsys.readouterr().out


def test_a_backward_request_is_refused_before_any_request(bump, tree, capsys):
    index, uv = FakeIndex(), FakeUv()
    assert run(bump, ["--tag", tag(torch="2.9.1+rocm7.13.0"), "--write"], index=index, uv=uv) == 1
    assert "older than the pinned" in capsys.readouterr().out
    assert index.requested == [] and uv.calls == []


def test_an_sdk_move_passes_the_upgrade_arguments_to_uv(bump, tree):
    uv = FakeUv()
    sdk = {"rocm-sdk-core": "7.14.0", "rocm-sdk-libraries-gfx103x-all": "7.14.0"}
    assert run(bump, ["--tag", tag(**sdk), "--write"], uv=uv) == 0
    assert uv.calls[0][0] == ["lock", "--upgrade-package", "rocm-sdk-core==7.14.0",
                              "--upgrade-package", "rocm-sdk-libraries-gfx103x-all==7.14.0",
                              "--upgrade-package", "rocm==7.14.0"]
    lock = (tree / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "rocm-sdk-core"\nversion = "7.14.0"' in lock


# --------------------------------------------------------------------------- #
#  The real tree and the real uv                                              #
# --------------------------------------------------------------------------- #

def test_the_regions_the_script_edits_exist_once_on_the_real_tree(bump):
    py, _ = bump._read(_ROOT / "pyproject.toml")
    lock, _ = bump._read(_ROOT / "uv.lock")
    current = bump.read_current(py, lock)
    assert set(current) == set(bump.PACKAGES)
    new_wanted = dict(current, torch=current["torch"].replace("+rocm", ".1+rocm"),
                      torchvision=current["torchvision"].replace("+rocm", ".1+rocm"))
    new_py = bump.rewrite_pyproject(py, new_wanted)
    assert new_py != py
    assert new_py.count(new_wanted["torch"]) == 1 and new_py.count(new_wanted["torchvision"]) == 1
    assert bump.read_current(new_py, lock)["torch"] == new_wanted["torch"]


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_the_real_uv_locks_and_checks_a_trivial_project(bump, tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "probe"\nversion = "0.1.0"\nrequires-python = ">=3.8"\n'
        "dependencies = []\n", encoding="utf-8")
    code, out = bump.run_uv(["lock"], tmp_path)
    assert code == 0, out
    assert (tmp_path / "uv.lock").is_file()
    assert bump.run_uv(["lock", "--check"], tmp_path)[0] == 0
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "probe"\nversion = "0.2.0"\nrequires-python = ">=3.8"\n'
        'description = "changed"\ndependencies = []\n', encoding="utf-8")
    stale = subprocess.run(["uv", "--directory", str(tmp_path), "lock", "--check"],
                           capture_output=True, text=True)
    assert stale.returncode != 0
    assert bump.run_uv(["lock", "--check"], tmp_path)[0] != 0
