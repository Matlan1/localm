# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every setup variant offers the same install.

The variants are setup.bat, setup.sh, setup-gui.bat and setup-gui.sh with
installer/gui.py. localm/setup_spec.py lists every question with its choices and
default; each prompt in a variant carries a ``SETUP-Q: <id>`` marker. These
tests read the five files and check them against the spec: each question is
asked by every variant it applies to, no prompt goes unmarked, a console
prompt shows the spec's default and lists exactly the spec's choices, the
window starts on the spec's defaults, and no variant carries its own copy of a
list the spec owns.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from tests._tk_root import build_tk_root, collect_tk_garbage  # noqa: F401

from localm import setup_spec as spec

ROOT = Path(__file__).resolve().parents[1]
TAG = re.compile(r"SETUP-Q:\s*([a-z0-9-]+)")

# (kind, platform group) -> the files that make up that variant.
VARIANTS = {
    (spec.CONSOLE, "win32"): ("setup.bat",),
    (spec.CONSOLE, "posix"): ("setup.sh",),
    (spec.GUI, "win32"): ("setup-gui.bat", "installer/gui.py"),
    (spec.GUI, "posix"): ("setup-gui.sh", "installer/gui.py"),
}
GROUP_PLATFORMS = {"win32": ("win32",), "posix": spec.POSIX}
SHARED_FILES = ("installer/gui.py",)
SCRIPTS = ("setup.bat", "setup.sh", "setup-gui.bat", "setup-gui.sh")


def _lines(name: str) -> list:
    return (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")


def _tags(names) -> dict:
    """{id: [(file, line index)]} for every marker in *names*."""
    found: dict = {}
    for name in names:
        for i, line in enumerate(_lines(name)):
            for qid in TAG.findall(line):
                found.setdefault(qid, []).append((name, i))
    return found


def _applies(q, kind: str, group: str) -> bool:
    return kind in q.variants and bool(set(q.platforms) & set(GROUP_PLATFORMS[group]))


def _is_prompt(name: str, line: str) -> bool:
    s = line.strip()
    if s.startswith(("rem ", "#", "::")) or not s:
        return False
    if name.endswith(".bat"):
        return bool(re.search(r"\bset /p\b", s, re.I))
    if s.startswith(("ask()", "ask() {")):
        return False
    if 'read -r -p "$prompt"' in s:
        return False
    return bool(re.search(r'\bask "', s)) or "read -r -p" in s


def _prompts(name: str) -> list:
    """(line index, text) of every prompt line in a script."""
    return [(i, line) for i, line in enumerate(_lines(name)) if _is_prompt(name, line)]


def _tag_above(lines: list, i: int) -> str:
    """The id of the marker directly above line *i* (only a `call :flush`, or
    another marker, may sit between), or empty."""
    j = i - 1
    while j >= 0:
        s = lines[j].strip()
        m = TAG.search(s)
        if m:
            return m.group(1)
        if s.lower() == "call :flush":
            j -= 1
            continue
        return ""
    return ""


def _next_prompt(name: str, i: int) -> str:
    """The first prompt line within 6 lines after the marker at *i*, or empty."""
    lines = _lines(name)
    for j in range(i + 1, min(i + 7, len(lines))):
        if TAG.search(lines[j]):
            continue
        if _is_prompt(name, lines[j]):
            return lines[j]
    return ""


def _shown_default(prompt: str) -> str:
    """The default a console prompt shows: the last [N], [Y/n] or [y/N]."""
    shown = re.findall(r"\[(\d|Y/n|y/N)\]", prompt)
    if not shown:
        return ""
    last = shown[-1]
    return {"Y/n": "y", "y/N": "n"}.get(last, last)


def _sh_ask_default(prompt: str):
    """The default argument of a setup.sh ask() call, lowercased; None when the
    line is not an ask() call."""
    m = re.search(r'\bask "(?:[^"\\]|\\.)*"\s+("?)([^"\s)]*)\1\s*\)', prompt)
    return None if m is None else m.group(2).lower()


def _menu_numbers(name: str, i: int) -> list:
    """The [N] menu entries printed just above the prompt at line *i*."""
    lines = _lines(name)
    nums: list = []
    j = i - 1
    while j >= 0 and i - j <= 14:
        s = lines[j].strip()
        if (TAG.search(s) or s.lower() == "call :flush" or s.startswith("set \"")
                or s.lower().startswith("rem ") or s.startswith("#")
                or re.match(r"^[A-Za-z_]+=\S*$", s) or not s):
            j -= 1
            continue
        if s.lower().startswith(("echo", "say")):
            nums = re.findall(r"^(?:echo|say)\s+\"?\s*\[(\d)\]", s, re.I) + nums
            j -= 1
            continue
        break
    return nums


# --------------------------------------------------------------------------- #
#  Every question is asked where it applies, and every prompt is marked       #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("variant", sorted(VARIANTS), ids=lambda v: f"{v[0]}-{v[1]}")
def test_every_question_is_asked_by_every_variant_it_applies_to(variant):
    kind, group = variant
    found = _tags(VARIANTS[variant])
    missing = [q.id for q in spec.QUESTIONS
               if _applies(q, kind, group) and q.id not in found]
    assert not missing, f"{kind}/{group} does not ask: {missing}"


@pytest.mark.parametrize("variant", sorted(VARIANTS), ids=lambda v: f"{v[0]}-{v[1]}")
def test_no_variant_asks_a_question_the_spec_does_not_list_for_it(variant):
    kind, group = variant
    ids = {q.id: q for q in spec.QUESTIONS}
    extra = []
    for qid, places in _tags(VARIANTS[variant]).items():
        q = ids.get(qid)
        # installer/gui.py serves every platform; it checks the platform itself.
        shared = all(f in SHARED_FILES for f, _ in places)
        if q is None or not (kind in q.variants if shared else _applies(q, kind, group)):
            extra.append((qid, places[0]))
    assert not extra, f"{kind}/{group} asks questions the spec does not give it: {extra}"


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_prompt_in_a_script_is_marked(name):
    lines = _lines(name)
    unmarked = [(i + 1, line.strip()) for i, line in _prompts(name)
                if not _tag_above(lines, i)]
    assert not unmarked, f"{name}: prompts with no SETUP-Q marker: {unmarked}"


# --------------------------------------------------------------------------- #
#  Defaults and choices                                                        #
# --------------------------------------------------------------------------- #

def _tagged_prompts():
    for name in SCRIPTS:
        lines = _lines(name)
        for i, line in _prompts(name):
            qid = _tag_above(lines, i)
            if qid:
                yield name, i, qid, line


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_console_prompt_shows_the_spec_default(name):
    wrong = []
    for fname, i, qid, line in _tagged_prompts():
        if fname != name or qid not in {q.id for q in spec.QUESTIONS}:
            continue
        q = spec.question(qid)
        shown = _shown_default(line)
        if shown and shown != q.default:
            wrong.append((i + 1, qid, shown, q.default))
        if name.endswith(".sh"):
            arg = _sh_ask_default(line)
            if arg is not None and arg and shown and arg != shown:
                wrong.append((i + 1, qid, f"ask default {arg}", shown))
    assert not wrong, f"{name}: (line, question, shown, spec): {wrong}"


def _effective_default(name: str, i: int, prompt: str):
    """The default the script applies after the prompt at line *i*: the value of
    ``if not defined VAR set "VAR=..."`` (.bat) or ``VAR="${VAR:-...}"`` (read in
    .sh), lowercased first character; "" when the prompt sets none; None when the
    prompt does not name a variable (setup.sh ask(), checked by its argument)."""
    lines = _lines(name)
    if name.endswith(".bat"):
        m = re.search(r'set /p "([A-Za-z_]+)=', prompt, re.I)
        if not m:
            return None
        var = m.group(1)
        pat = re.compile(rf'if not defined {var} set "{var}=([^"]*)"', re.I)
    else:
        m = re.search(r'read -r -p "[^"]*"\s+([A-Za-z_]+)', prompt)
        if not m:
            return None
        var = m.group(1)
        pat = re.compile(rf'\$\{{{var}:-([^}}]*)\}}')
    for j in range(i, min(i + 5, len(lines))):
        found = pat.search(lines[j])
        if found:
            return found.group(1)[:1].lower()
    return ""


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_console_prompt_applies_the_spec_default(name):
    """What Enter actually gives, not only what the prompt text shows."""
    wrong = []
    for fname, i, qid, line in _tagged_prompts():
        if fname != name:
            continue
        applied = _effective_default(name, i, line)
        if applied is None:
            continue
        shown = _shown_default(line)
        expected = spec.question(qid).default[:1] if shown else ""
        if applied != expected:
            wrong.append((i + 1, qid, applied, expected))
    assert not wrong, f"{name}: (line, question, applied, spec): {wrong}"


def test_each_question_shows_its_default_somewhere_it_is_asked():
    """A question asked with a visible default shows the spec's default in at
    least one place, so a marker on an unrelated line cannot hide a change."""
    shown: dict = {}
    for _name, _i, qid, line in _tagged_prompts():
        d = _shown_default(line)
        if d:
            shown.setdefault(qid, set()).add(d)
    for qid, values in shown.items():
        assert values == {spec.question(qid).default}, (qid, values)


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_numbered_menu_lists_exactly_the_spec_choices(name):
    wrong = []
    for fname, i, qid, _line in _tagged_prompts():
        if fname != name:
            continue
        q = spec.question(qid)
        if not q.choices:
            continue
        numbers = _menu_numbers(name, i)
        if numbers and numbers != [a for a, _ in q.choices]:
            wrong.append((i + 1, qid, numbers, [a for a, _ in q.choices]))
    assert not wrong, f"{name}: (line, question, menu, spec): {wrong}"


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_menu_question_prints_its_menu(name):
    """At least one prompt of each numbered-menu question prints its [N] menu
    (a follow-up text entry under the same question prints none)."""
    printed: dict = {}
    for fname, i, qid, _line in _tagged_prompts():
        if fname == name and spec.question(qid).choices:
            printed[qid] = printed.get(qid, False) or bool(_menu_numbers(name, i))
    assert printed, f"{name} asks no numbered-menu question"
    assert all(printed.values()), f"{name}: menus not printed: {printed}"


def test_the_launchers_ask_the_tooling_question_exactly_as_the_consoles_do():
    def block(name, qid):
        lines = _lines(name)
        for fname, i, tag, line in _tagged_prompts():
            if fname == name and tag == qid:
                start = i
                while start > 0 and not re.search(r"Keep localm's Python tooling",
                                                  lines[start]):
                    start -= 1
                body = [re.sub(r"^\s*(echo|say)\s*", "", ln.strip()).strip('"')
                        for ln in lines[start:i] if re.match(r"\s*(echo|say)\b", ln)]
                body = [b.replace("^(", "(").replace("^)", ")") for b in body]
                return body, _shown_default(line)
        raise AssertionError(f"{name} does not ask {qid}")
    for console, launcher in (("setup.bat", "setup-gui.bat"), ("setup.sh", "setup-gui.sh")):
        assert block(launcher, "tooling-store") == block(console, "tooling-store")


# --------------------------------------------------------------------------- #
#  The window starts on the spec's defaults                                    #
# --------------------------------------------------------------------------- #

def _gui_module(tmp_path, monkeypatch):
    path = ROOT / "installer" / "gui.py"
    s = importlib.util.spec_from_file_location("localm_installer_gui_spec", path)
    mod = importlib.util.module_from_spec(s)
    sys.modules["localm_installer_gui_spec"] = mod
    s.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    for name in (mod.STORE_ENV, mod.UV_ENV, mod.UV_INSTALLED_ENV):
        monkeypatch.delenv(name, raising=False)
    return mod


def test_the_window_starts_on_the_spec_defaults(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    from tkinter import filedialog, ttk
    gui = _gui_module(tmp_path, monkeypatch)
    root = build_tk_root(tk.Tk)
    root.withdraw()
    try:
        w = gui.Wizard(root, tk, ttk, filedialog)
        plan = w.current_plan()
    finally:
        root.destroy()
    assert plan.backend == w.recommended
    assert plan.app_window is (spec.default_choice("window-mode") == "app-window")
    assert plan.portable_data is (spec.default_choice("data-folder") == "portable")
    assert plan.shortcut == spec.default_choice("shortcut")
    assert plan.add_to_path is spec.default_yes("global-command")
    assert plan.plugin_deps is spec.default_yes("plugin-deps")
    assert plan.replace_venv is spec.default_yes("replace-venv")
    assert plan.own_build_dir == spec.question("own-build-dir").default
    assert plan.portable_store is (spec.default_choice("tooling-store") == "portable")
    offered = [n for n, _ in spec.setup_plugins()]
    assert plan.plugins == tuple(n for n in offered if n in spec.RECOMMENDED_PLUGINS)
    assert set(plan.plugins) == set(spec.question("plugins").default.split(","))
    assert w.choice_var.get() == spec.default_choice("existing-install")
    assert w.purge_var.get() is spec.default_yes("uninstall-data")


# The Wizard variable that answers each spec question the window asks.
WINDOW_CONTROLS = {
    "existing-install": ("choice_var",),
    "uninstall-data": ("purge_var",),
    "replace-venv": ("replace_venv_var",),
    "data-folder": ("portable_var", "path_var"),
    "window-mode": ("appwin_var",),
    "backend": ("backend_var",),
    "own-build-dir": ("own_dir_var",),
    "shortcut": ("shortcut_var",),
    "global-command": ("path_cmd_var",),
    "plugins": ("plugin_vars",),
    "plugin-deps": ("deps_var",),
}
# Window questions asked by a dialog or a step, not by a control on a page.
WINDOW_DIALOGS = {"uninstall-confirm", "venv-lockers", "venv-retry", "runtime-failed",
                  "keep-data", "data-offline"}
# Questions the launcher asks before the window opens.
LAUNCHER_QUESTIONS = {"tooling-store", "uv-install"}


def test_every_window_control_answers_a_spec_question(tmp_path, monkeypatch):
    """Walk every page of a repair-mode window (so the .venv question shows) and
    match each control's bound variable against the spec's questions: an extra
    control or a lost one fails."""
    tk = pytest.importorskip("tkinter")
    from tkinter import filedialog, ttk
    gui = _gui_module(tmp_path, monkeypatch)
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(gui.sys, "platform", "win32")
    root = build_tk_root(tk.Tk)
    root.withdraw()
    try:
        w = gui.Wizard(root, tk, ttk, filedialog, existing=True)
        names = {}
        for attr, value in vars(w).items():
            if isinstance(value, tk.Variable):
                names[str(value)] = attr
            elif attr == "plugin_vars":
                for v in value.values():
                    names[str(v)] = "plugin_vars"
        bound = set()
        stack = [w.container]
        while stack:
            widget = stack.pop()
            stack.extend(widget.winfo_children())
            for option in ("variable", "textvariable"):
                try:
                    v = str(widget.cget(option))
                except Exception:
                    continue
                if v:
                    bound.add(names.get(v, f"<unknown {v}>"))
    finally:
        root.destroy()
    expected = {a for attrs in WINDOW_CONTROLS.values() for a in attrs}
    assert bound == expected, (sorted(bound - expected), sorted(expected - bound))


def test_the_window_question_table_covers_the_spec():
    asked = {q.id for q in spec.QUESTIONS if spec.GUI in q.variants}
    covered = set(WINDOW_CONTROLS) | WINDOW_DIALOGS | LAUNCHER_QUESTIONS
    assert asked == covered, (sorted(asked - covered), sorted(covered - asked))


# --------------------------------------------------------------------------- #
#  No variant keeps its own copy of a list the spec owns                       #
# --------------------------------------------------------------------------- #

BACKEND_NAMES = [k for k, _ in spec.BACKENDS]


@pytest.mark.parametrize("name", ("setup.bat", "setup.sh", "installer/gui.py"))
def test_no_variant_hardcodes_the_extras(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    assert spec.BASE_EXTRAS not in text


@pytest.mark.parametrize("name", ("setup.bat", "setup.sh", "installer/gui.py"))
def test_no_variant_hardcodes_a_backend_list(name):
    """A line naming three or more backends is a menu or a fallback list, which
    belongs in the spec."""
    lists = []
    for i, line in enumerate(_lines(name)):
        named = {b for b in BACKEND_NAMES
                 if re.search(rf"(?<![\w-]){re.escape(b)}(?![\w-])", line)}
        if len(named) >= 3:
            lists.append((i + 1, line.strip()))
    assert not lists, f"{name} carries its own backend list: {lists}"


@pytest.mark.parametrize("name", ("setup.bat", "setup.sh"))
def test_the_console_backend_menu_comes_from_the_spec(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    assert "localm.setup_spec backend-menu" in text
    assert "localm.setup_spec backend-hint" in text
    assert "localm.setup_spec extras" in text


@pytest.mark.parametrize("name", ("setup.bat", "setup.sh", "installer/gui.py"))
def test_every_variant_records_the_uv_it_used(name):
    """Shared installs record the uv they ran with, so the updater finds it."""
    text = (ROOT / name).read_text(encoding="utf-8")
    assert "--uv-exe" in text


def test_no_variant_hardcodes_the_plugin_set():
    text = (ROOT / "installer" / "gui.py").read_text(encoding="utf-8")
    assert '"coder", "rag"' not in text and "'coder', 'rag'" not in text
    from localm.cli import plugins as cli_plugins
    assert cli_plugins._SETUP_DEFAULTS is spec.RECOMMENDED_PLUGINS


def test_the_plugin_list_is_the_catalog_filtered_by_offered(tmp_path, monkeypatch):
    from localm.plugins import catalog
    expected = [n for n in catalog.names()
                if catalog.offered(n) and n not in catalog.preinstalled()]
    assert [n for n, _ in spec.setup_plugins()] == expected
    gui = _gui_module(tmp_path, monkeypatch)
    assert [n for n, _ in gui.plugin_choices()] == expected


@pytest.mark.parametrize("rec", [k for k in BACKEND_NAMES] + [""])
def test_the_window_backend_menu_is_the_console_menu(tmp_path, monkeypatch, rec):
    """The window lists the console menu's entries once each, recommendation first."""
    gui = _gui_module(tmp_path, monkeypatch)
    console = []
    for _, k, _ in spec.backend_menu(rec):
        if k not in console:
            console.append(k)
    if rec and rec not in spec.offered_backends():
        console.remove(rec)
        console.insert(0, rec)
    assert [k for k, _ in gui.backend_choices(rec or None)] == [k for k in console if k]


# --------------------------------------------------------------------------- #
#  The spec itself                                                             #
# --------------------------------------------------------------------------- #

def test_question_ids_are_unique_and_defaults_are_valid():
    ids = [q.id for q in spec.QUESTIONS]
    assert len(ids) == len(set(ids))
    for q in spec.QUESTIONS:
        assert q.variants and set(q.variants) <= {spec.CONSOLE, spec.GUI}, q.id
        assert q.platforms and set(q.platforms) <= set(spec.ALL_PLATFORMS), q.id
        if q.choices:
            assert q.default in dict(q.choices), q.id
            assert [a for a, _ in q.choices] == [str(n) for n in range(1, len(q.choices) + 1)]


@pytest.mark.parametrize("plat,arch", [("win32", "x64"), ("win32", "arm64"),
                                       ("linux", "x64"), ("linux", "arm64"),
                                       ("darwin", "arm64"), ("darwin", "x64")])
def test_the_backend_menu_offers_only_prebuilt_backends(plat, arch):
    from localm import hwdetect, setup_llama
    rows = spec.backend_menu("cpu", plat, arch)
    assert [n for n, _, _ in rows] == [str(i) for i in range(1, len(rows) + 1)]
    keys = [k for _, k, _ in rows[1:-1]]
    assert keys == [k for k in BACKEND_NAMES if hwdetect.has_prebuilt(k, plat, arch)]
    assert rows[-1][1] == spec.OWN
    for k in keys:
        assert k in setup_llama.BACKENDS
    assert sum(spec.SAME_SUFFIX in label for _, _, label in rows) == 1
