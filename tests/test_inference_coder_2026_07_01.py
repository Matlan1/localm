# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the 2026-07-01 inference/coder-behavior backlog cluster.

  AUD-SPLITTHINK     - split_think is linear (no O(n^2) buffer re-slice)
  REC-CODER-GUI-PATH - the coder system prompt home-anchors the cwd (no username leak)
  REC-CODER-CWD-LEAK - launcher.json persists a home-anchored coder_dir, not absolute
"""

import time
from pathlib import Path

import pytest


# --------------------------------------------------------------------------- #
#  AUD-SPLITTHINK
# --------------------------------------------------------------------------- #

def test_split_think_basic_cases():
    from localm.inference.textnorm import split_think
    assert split_think("hello") == ("hello", "")
    assert split_think("a<think>b</think>c") == ("ac", "b")
    assert split_think("<think>only</think>") == ("", "only")
    assert split_think("x<think>unclosed") == ("x", "unclosed")
    c, r = split_think("a<think>1</think>b<think>2</think>c")
    assert c == "abc" and r == "12"


def test_split_think_linear_on_pathological_input():
    from localm.inference.textnorm import split_think
    # 50k unmatched <think> openers: the old ThinkSplitter re-sliced its buffer
    # per tag (O(n^2)) and would take seconds+; the linear scan is near-instant.
    text = "<think>" * 50000 + "answer"
    start = time.perf_counter()
    content, reasoning = split_think(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"split_think took {elapsed:.2f}s - not linear?"
    assert content == ""                       # all openers, nothing closed
    assert "answer" in reasoning


# --------------------------------------------------------------------------- #
#  REC-CODER-GUI-PATH
# --------------------------------------------------------------------------- #

def test_display_cwd_home_anchors():
    from localm.plugins.coder.prompts import _display_cwd
    shown = _display_cwd(Path.home() / "projects" / "secretapp")
    assert shown.startswith("~/")
    assert "secretapp" in shown


def test_coder_prompt_does_not_leak_absolute_home_path():
    from localm.plugins.coder.prompts import build_system_prompt
    proj = Path.home() / "projects" / "secretapp"
    prompt = build_system_prompt(proj, model_name="generic")
    assert str(proj) not in prompt, "the absolute cwd (with username) leaked into the prompt"
    assert "~/projects/secretapp" in prompt


# --------------------------------------------------------------------------- #
#  REC-CODER-CWD-LEAK
# --------------------------------------------------------------------------- #

def _load_launcher():
    import importlib.machinery
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    path = str(root / "launcher.pyw")
    # .pyw is a recognized Python source suffix only on Windows; on POSIX
    # spec_from_file_location cannot infer a loader and returns None, so pass an
    # explicit SourceFileLoader (mirrors tests/test_launcher_phase3.py) to load it
    # on every platform.
    loader = importlib.machinery.SourceFileLoader("launcher_pyw", path)
    spec = importlib.util.spec_from_file_location("launcher_pyw", path, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # tkinter unavailable in a headless env, etc.
        pytest.skip(f"launcher.pyw not importable here: {e}")
    return mod


def test_launcher_coder_dir_home_anchored_roundtrip():
    launcher = _load_launcher()
    abs_path = str(Path.home() / "projects" / "myapp")
    anchored = launcher._anchor_home(abs_path)
    assert anchored.startswith("~/"), anchored
    assert str(Path.home()) not in anchored, "the OS username must not persist"
    assert Path(launcher._expand_home(anchored)) == Path(abs_path)


def test_launcher_anchor_home_leaves_non_home_paths(tmp_path):
    launcher = _load_launcher()
    # A path outside home has no username to hide; keep it usable as-is.
    outside = str(tmp_path / "work")
    assert Path(launcher._expand_home(launcher._anchor_home(outside))) == Path(outside)
