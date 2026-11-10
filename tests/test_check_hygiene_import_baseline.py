# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/check_hygiene.py checks 12 and 13, compared with
.github/hygiene-import-baseline.json.

Check 12 counts function-local imports that go up docs/layering.toml or between
peers, per unit pair. Check 13 counts imports of localm.plugins.builtin.<name>
from outside that plugin's folder, per file. A count above the baseline fails,
a count below it fails until the baseline is lowered, and the real tree matches
the committed baseline.
"""

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

_LAYERING = (
    '[[tier]]\nname = "top"\nrole = "top"\nunits = ["app"]\n\n'
    '[[tier]]\nname = "mid"\nrole = "mid"\nunits = ["plugins", "svc", "svc2"]\n\n'
    '[[tier]]\nname = "low"\nrole = "low"\nunits = ["util"]\n'
)
_BASE_FILES = {
    "app.py": "", "svc.py": "", "svc2.py": "", "util.py": "",
    "plugins/__init__.py": "",
    "plugins/builtin/__init__.py": "",
    "plugins/builtin/jobs/__init__.py": "",
    "plugins/builtin/jobs/store.py": "def load():\n    return 1\n",
    "plugins/builtin/memory/__init__.py": "",
    "plugins/builtin/memory/synth.py": "",
}


def _load_check_hygiene():
    spec = importlib.util.spec_from_file_location(
        "check_hygiene", REPO_ROOT / "scripts" / "check_hygiene.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _empty_baseline() -> dict:
    return {"deferred_up_imports": {}, "builtin_plugin_imports": {}}


def _run(tmp_path, monkeypatch, files, baseline=None, raw_baseline=None):
    """Build a throwaway checkout and return (module, problems)."""
    ch = _load_check_hygiene()
    root = tmp_path / "localm"
    root.mkdir(exist_ok=True)
    (root / "__init__.py").write_text("", encoding="utf-8")
    for rel, src in {**_BASE_FILES, **files}.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "layering.toml").write_text(_LAYERING, encoding="utf-8")
    gh = tmp_path / ".github"
    gh.mkdir(exist_ok=True)
    if raw_baseline is not None:
        (gh / "hygiene-import-baseline.json").write_text(raw_baseline, encoding="utf-8")
    elif baseline is not False:
        (gh / "hygiene-import-baseline.json").write_text(
            json.dumps(baseline or _empty_baseline()), encoding="utf-8")
    monkeypatch.setattr(ch, "REPO", tmp_path)
    monkeypatch.setattr(ch, "_tracked_localm_files", lambda _root: None)
    return ch, ch._import_baseline_violations()


_UP = "def f():\n    from localm.app import run\n    return run\n"


# --------------------------------------------------------------------------- #
#  Check 12: function-local imports up the layering or between peers           #
# --------------------------------------------------------------------------- #

def test_function_local_upward_import_above_baseline_fires(tmp_path, monkeypatch):
    _, problems = _run(tmp_path, monkeypatch, {"util.py": _UP})
    assert len(problems) == 1, problems
    assert "function-local import goes UP" in problems[0]
    assert "util -> app (1, baseline 0" in problems[0]


def test_function_local_peer_import_fires(tmp_path, monkeypatch):
    _, problems = _run(tmp_path, monkeypatch, {
        "svc.py": "def f():\n    import localm.svc2\n"})
    assert len(problems) == 1, problems
    assert "svc -> svc2 (1, baseline 0" in problems[0]


def test_method_and_nested_function_imports_are_counted(tmp_path, monkeypatch):
    _, problems = _run(tmp_path, monkeypatch, {"util.py": (
        "class C:\n    def m(self):\n        def g():\n"
        "            from localm import app\n        return g\n")})
    assert len(problems) == 1, problems
    assert "util -> app (1, baseline 0" in problems[0]


def test_upward_import_matching_the_baseline_is_clean(tmp_path, monkeypatch):
    base = _empty_baseline()
    base["deferred_up_imports"] = {"util -> app": 2}
    _, problems = _run(tmp_path, monkeypatch,
                       {"util.py": _UP + "def g():\n    import localm.app\n"},
                       baseline=base)
    assert problems == []


def test_count_below_the_baseline_fails_until_lowered(tmp_path, monkeypatch):
    base = _empty_baseline()
    base["deferred_up_imports"] = {"util -> app": 3}
    _, problems = _run(tmp_path, monkeypatch, {"util.py": _UP}, baseline=base)
    assert len(problems) == 1, problems
    assert "is stale" in problems[0] and "'util -> app' is 1, baseline 3" in problems[0]


def test_downward_function_local_import_is_not_counted(tmp_path, monkeypatch):
    _, problems = _run(tmp_path, monkeypatch, {
        "app.py": "def f():\n    from localm.util import x\n"})
    assert problems == []


def test_module_level_upward_import_is_left_to_check_9(tmp_path, monkeypatch):
    _, problems = _run(tmp_path, monkeypatch, {"util.py": "import localm.app\n"})
    assert problems == []


def test_type_checking_guarded_function_import_is_not_counted(tmp_path, monkeypatch):
    _, problems = _run(tmp_path, monkeypatch, {"util.py": (
        "from typing import TYPE_CHECKING\n"
        "def f():\n    if TYPE_CHECKING:\n        from localm.app import run\n")})
    assert problems == []


# --------------------------------------------------------------------------- #
#  Check 13: imports of a first-party plugin from outside its folder          #
# --------------------------------------------------------------------------- #

def test_core_importing_a_builtin_plugin_fires(tmp_path, monkeypatch):
    _, problems = _run(tmp_path, monkeypatch, {
        "app.py": "def f():\n    from localm.plugins.builtin.jobs.store import load\n"})
    assert len(problems) == 1, problems
    assert "first-party plugin from outside its own folder" in problems[0]
    assert "localm/app.py -> jobs (1, baseline 0" in problems[0]


def test_from_builtin_import_name_shape_fires(tmp_path, monkeypatch):
    _, problems = _run(tmp_path, monkeypatch, {
        "app.py": "from localm.plugins.builtin import jobs\n"})
    assert any("localm/app.py -> jobs (1, baseline 0" in p for p in problems), problems


def test_another_plugin_importing_relatively_fires(tmp_path, monkeypatch):
    _, problems = _run(tmp_path, monkeypatch, {
        "plugins/builtin/memory/synth.py": "from ..jobs.store import load\n"})
    assert len(problems) == 1, problems
    assert "localm/plugins/builtin/memory/synth.py -> jobs (1" in problems[0]


def test_a_plugin_importing_its_own_modules_is_clean(tmp_path, monkeypatch):
    _, problems = _run(tmp_path, monkeypatch, {
        "plugins/builtin/jobs/runner.py": (
            "from localm.plugins.builtin.jobs.store import load\n"
            "from .store import load as again\n")})
    assert problems == []


def test_plugin_import_matching_the_baseline_is_clean(tmp_path, monkeypatch):
    base = _empty_baseline()
    base["builtin_plugin_imports"] = {"localm/app.py -> jobs": 1}
    _, problems = _run(tmp_path, monkeypatch, {
        "app.py": "import localm.plugins.builtin.jobs.store\n"}, baseline=base)
    assert problems == []


# --------------------------------------------------------------------------- #
#  The baseline file itself                                                    #
# --------------------------------------------------------------------------- #

def test_missing_baseline_fails_loud(tmp_path, monkeypatch):
    _, problems = _run(tmp_path, monkeypatch, {}, baseline=False)
    assert len(problems) == 1 and "could not be read" in problems[0], problems
    assert "did not run" in problems[0]


def test_invalid_json_baseline_fails_loud(tmp_path, monkeypatch):
    _, problems = _run(tmp_path, monkeypatch, {}, raw_baseline="{not json")
    assert len(problems) == 1 and "not valid JSON" in problems[0], problems


def test_baseline_with_an_extra_key_fails_loud(tmp_path, monkeypatch):
    raw = json.dumps({**_empty_baseline(), "allow": {}})
    _, problems = _run(tmp_path, monkeypatch, {}, raw_baseline=raw)
    assert len(problems) == 1 and "exactly the keys" in problems[0], problems


def test_baseline_with_a_non_positive_count_fails_loud(tmp_path, monkeypatch):
    raw = json.dumps({"deferred_up_imports": {"util -> app": 0},
                      "builtin_plugin_imports": {}})
    _, problems = _run(tmp_path, monkeypatch, {}, raw_baseline=raw)
    assert len(problems) == 1 and "positive integer" in problems[0], problems


def test_write_baseline_round_trips(tmp_path, monkeypatch):
    ch, problems = _run(tmp_path, monkeypatch, {
        "util.py": _UP,
        "app.py": "def f():\n    import localm.plugins.builtin.memory.synth\n"})
    assert len(problems) == 2, problems
    assert ch._write_import_baseline() == 0
    written = json.loads((tmp_path / ".github" / "hygiene-import-baseline.json")
                         .read_text(encoding="utf-8"))
    assert written == {"deferred_up_imports": {"util -> app": 1},
                       "builtin_plugin_imports": {"localm/app.py -> memory": 1}}
    assert ch._import_baseline_violations() == []


# --------------------------------------------------------------------------- #
#  The real tree                                                               #
# --------------------------------------------------------------------------- #

def test_real_tree_matches_the_committed_baseline():
    ch = _load_check_hygiene()
    assert ch._import_baseline_violations() == []


def test_real_tree_counts_are_non_trivial():
    ch = _load_check_hygiene()
    counts, problems = ch._import_counts()
    assert problems == []
    assert sum(counts["deferred_up_imports"].values()) > 0
    assert sum(counts["builtin_plugin_imports"].values()) > 0
