# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/check_hygiene.py check 8: no console.print(...) reaching a module
that executes inside an isolated child process. A child reports facts as data
and never prints directly, so a native crash there stays a clean, catchable
error in the parent.

These tests pin what makes the check worth having: it FIRES on a real
violation, respects the allowlist for the one confirmed parent-side exception,
and does not false-positive on an unrelated ``.print(`` call.
"""

import ast
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_check_hygiene():
    spec = importlib.util.spec_from_file_location(
        "check_hygiene", REPO_ROOT / "scripts" / "check_hygiene.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pkg(tmp_path, files: dict) -> Path:
    """Build a throwaway localm/ package tree from {relative path: source}."""
    root = tmp_path / "localm"
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
    return root


# --------------------------------------------------------------------------- #
#  NEGATIVE: it must actually fire                                            #
# --------------------------------------------------------------------------- #

def test_console_print_in_a_child_module_is_detected(tmp_path, monkeypatch):
    ch = _load_check_hygiene()
    _pkg(tmp_path, {
        "inference/backends/llamacpp/llama.py": (
            "from localm.console import console\n"
            "def _apply_cpu_moe():\n"
            "    console.print('[yellow]MoE skipped[/yellow]')\n"
        ),
    })
    monkeypatch.setattr(ch, "REPO", tmp_path)
    monkeypatch.setattr(ch, "_CHILD_PROCESS_MODULES",
                         ("localm/inference/backends/llamacpp/llama.py",))
    monkeypatch.setattr(ch, "_CHILD_PROCESS_CONSOLE_PRINT_ALLOWLIST", {})
    problems = ch._child_process_console_print_violations()
    assert problems, "a real console.print in a child module must be reported"
    joined = "\n".join(problems)
    assert "llama.py:3" in joined, joined
    assert "_apply_cpu_moe" in joined, joined


def test_dotted_qualname_reports_class_and_method(tmp_path, monkeypatch):
    """The enclosing-scope tracker must see through a class boundary, not just
    a function boundary - the real _sizing.py case (VramSizingMixin's own
    methods) is exactly this shape."""
    ch = _load_check_hygiene()
    _pkg(tmp_path, {
        "inference/backends/llamacpp/_sizing.py": (
            "from localm.console import console\n"
            "class VramSizingMixin:\n"
            "    def _new_method(self):\n"
            "        console.print('oops')\n"
        ),
    })
    monkeypatch.setattr(ch, "REPO", tmp_path)
    monkeypatch.setattr(ch, "_CHILD_PROCESS_MODULES",
                         ("localm/inference/backends/llamacpp/_sizing.py",))
    monkeypatch.setattr(ch, "_CHILD_PROCESS_CONSOLE_PRINT_ALLOWLIST", {})
    problems = ch._child_process_console_print_violations()
    assert problems
    assert "VramSizingMixin._new_method" in "\n".join(problems), problems


# --------------------------------------------------------------------------- #
#  POSITIVE: the allowlist is honoured, and only for the named site           #
# --------------------------------------------------------------------------- #

def test_allowlisted_call_site_is_not_reported(tmp_path, monkeypatch):
    ch = _load_check_hygiene()
    _pkg(tmp_path, {
        "inference/backends/llamacpp/_sizing.py": (
            "from localm.console import console\n"
            "class VramSizingMixin:\n"
            "    def _check_vram(self):\n"
            "        console.print('[yellow]Low VRAM[/yellow]')\n"
        ),
    })
    monkeypatch.setattr(ch, "REPO", tmp_path)
    monkeypatch.setattr(ch, "_CHILD_PROCESS_MODULES",
                         ("localm/inference/backends/llamacpp/_sizing.py",))
    monkeypatch.setattr(ch, "_CHILD_PROCESS_CONSOLE_PRINT_ALLOWLIST", {
        "localm/inference/backends/llamacpp/_sizing.py":
            frozenset({"VramSizingMixin._check_vram"}),
    })
    assert ch._child_process_console_print_violations() == []


def test_allowlist_is_scoped_to_the_named_method_only(tmp_path, monkeypatch):
    """Allow-listing one method in a file must not blanket-exempt the whole
    file - a second, unrelated console.print in the SAME module still fires.
    Precision is the entire point: the allowlist is meant to force the
    parent/child boundary into executable form, not paper over it."""
    ch = _load_check_hygiene()
    _pkg(tmp_path, {
        "inference/backends/llamacpp/_sizing.py": (
            "from localm.console import console\n"
            "class VramSizingMixin:\n"
            "    def _check_vram(self):\n"
            "        console.print('[yellow]Low VRAM[/yellow]')\n"
            "    def _check_context_fit(self):\n"
            "        console.print('this one is NOT allow-listed')\n"
        ),
    })
    monkeypatch.setattr(ch, "REPO", tmp_path)
    monkeypatch.setattr(ch, "_CHILD_PROCESS_MODULES",
                         ("localm/inference/backends/llamacpp/_sizing.py",))
    monkeypatch.setattr(ch, "_CHILD_PROCESS_CONSOLE_PRINT_ALLOWLIST", {
        "localm/inference/backends/llamacpp/_sizing.py":
            frozenset({"VramSizingMixin._check_vram"}),
    })
    problems = ch._child_process_console_print_violations()
    assert len(problems) == 1, problems
    assert "_check_context_fit" in problems[0], problems


# --------------------------------------------------------------------------- #
#  Precision: must not false-positive on a lookalike call                     #
# --------------------------------------------------------------------------- #

def test_unrelated_print_calls_are_ignored(tmp_path, monkeypatch):
    """A bare builtin print(), a logger call, and a DIFFERENT object's own
    .print() method (not named "console") must all be left alone - only the
    exact ``console.print(...)`` shape this codebase's one rich Console
    convention uses (see localm/console.py) is in scope."""
    ch = _load_check_hygiene()
    _pkg(tmp_path, {
        "inference/backends/llamacpp/llama.py": (
            "def f():\n"
            "    print('plain builtin print, not rich')\n"
            "    logger.info('not console.print either')\n"
            "    progress.print('a Progress object, not named console')\n"
            "    foo.console.print('attribute chain, not a bare console name')\n"
        ),
    })
    monkeypatch.setattr(ch, "REPO", tmp_path)
    monkeypatch.setattr(ch, "_CHILD_PROCESS_MODULES",
                         ("localm/inference/backends/llamacpp/llama.py",))
    monkeypatch.setattr(ch, "_CHILD_PROCESS_CONSOLE_PRINT_ALLOWLIST", {})
    assert ch._child_process_console_print_violations() == []


# --------------------------------------------------------------------------- #
#  Robustness                                                                  #
# --------------------------------------------------------------------------- #

def test_unparseable_child_module_is_reported_not_skipped(tmp_path, monkeypatch):
    """A file this gate cannot read is one it cannot vouch for: treating it as
    print-free would let a violation hide behind a syntax error."""
    ch = _load_check_hygiene()
    _pkg(tmp_path, {"inference/backends/llamacpp/llama.py": "def (\n"})
    monkeypatch.setattr(ch, "REPO", tmp_path)
    monkeypatch.setattr(ch, "_CHILD_PROCESS_MODULES",
                         ("localm/inference/backends/llamacpp/llama.py",))
    monkeypatch.setattr(ch, "_CHILD_PROCESS_CONSOLE_PRINT_ALLOWLIST", {})
    problems = ch._child_process_console_print_violations()
    assert any("could not parse" in p for p in problems), problems


def test_a_module_in_the_list_that_does_not_exist_is_skipped_quietly(tmp_path, monkeypatch):
    """Not a localm checkout, or the module moved: this check stays quiet and
    lets other gates (or the list itself going stale) surface that - mirrors
    check 7's "absent localm package" contract."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    monkeypatch.setattr(ch, "_CHILD_PROCESS_MODULES",
                         ("localm/inference/backends/llamacpp/llama.py",))
    monkeypatch.setattr(ch, "_CHILD_PROCESS_CONSOLE_PRINT_ALLOWLIST", {})
    assert ch._child_process_console_print_violations() == []


# --------------------------------------------------------------------------- #
#  The shipped tree                                                           #
# --------------------------------------------------------------------------- #

def test_the_shipped_tree_has_no_unlisted_child_console_prints():
    """The tree is clean under the real, hand-verified module list and the real
    allowlist. If this goes red, the fix is to stop the child from printing
    (report the fact as return data instead), NOT to grow the allowlist, unless
    the new site is verifiably parent-side (see check 8's block comment)."""
    ch = _load_check_hygiene()
    assert ch._child_process_console_print_violations() == []


def test_the_real_sizing_allowlist_entries_actually_exist_in_the_file(tmp_path):
    """A stale allowlist entry (the method was renamed or removed) would make
    this check silently weaker without ever going red - _check_vram etc. must
    still be real methods of VramSizingMixin in the real file, not just
    strings nobody re-verifies."""
    ch = _load_check_hygiene()
    real_path = REPO_ROOT / "localm/inference/backends/llamacpp/_sizing.py"
    tree = ast.parse(real_path.read_text(encoding="utf-8"))
    finder = ch._ConsolePrintFinder()
    finder.visit(tree)
    found_qualnames = {q for _, q in finder.hits}
    allowed = ch._CHILD_PROCESS_CONSOLE_PRINT_ALLOWLIST[
        "localm/inference/backends/llamacpp/_sizing.py"]
    for qualname in allowed:
        assert qualname in found_qualnames, (
            f"{qualname} is allow-listed but no console.print(...) call was "
            "found under that qualname in the real file - stale entry")
