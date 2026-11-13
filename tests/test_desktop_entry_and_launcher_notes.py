# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Linux menu entry every setup variant writes (localm.setup_spec), the
launcher make-launcher builds, and what make-launcher reports.

The Exec= tests parse the written value back by the Desktop Entry
Specification's own rules (string unescape first, then the quoting rule) and
compare the argv with the one that went in, so a path with a space, a quote, a
dollar sign or a backslash must survive the round trip."""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

import pytest

from localm import applaunch, setup_spec


def _string_unescape(value: str) -> str:
    out, i = [], 0
    while i < len(value):
        c = value[i]
        if c == "\\" and i + 1 < len(value):
            nxt = value[i + 1]
            out.append({"s": " ", "n": "\n", "t": "\t", "r": "\r", "\\": "\\"}.get(nxt, "\\" + nxt))
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _parse_exec(value: str) -> list:
    """argv from an Exec= value, per the Desktop Entry Specification."""
    text = _string_unescape(value)
    argv, cur, i, quoted, have = [], [], 0, False, False
    while i < len(text):
        c = text[i]
        if quoted:
            if c == "\\" and i + 1 < len(text) and text[i + 1] in '"`$\\':
                cur.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                quoted = False
                i += 1
                continue
            cur.append(c)
        elif c == '"':
            quoted, have = True, True
        elif c == " ":
            if cur or have:
                argv.append("".join(cur))
            cur, have = [], False
        else:
            assert c not in setup_spec._EXEC_RESERVED, f"unquoted reserved {c!r} in {value!r}"
            cur.append(c)
        i += 1
    assert not quoted, value
    if cur or have:
        argv.append("".join(cur))
    return [a.replace("%%", "%") for a in argv]


@pytest.mark.parametrize("argv", [
    ["/opt/localm/.venv/bin/LocaLM", "-m", "localm", "gui"],
    ["/home/a b/my apps/localm/localm-launcher.sh"],
    ['/home/o"brien/x', "-m", "localm", "gui"],
    ["/srv/$HOME/`cmd`/x"],
    ["/weird\\dir/x", "50%"],
    ["/tab\there/x", "(paren)", "semi;colon", "amp&", "pipe|", "hash#", "star*", "q?", "<>~"],
    ["/new\nline/x"],
    [""],
])
def test_exec_round_trips_through_the_spec_parser(argv):
    assert _parse_exec(setup_spec.desktop_exec(argv)) == argv


def test_a_plain_path_is_left_unquoted():
    assert setup_spec.desktop_exec(["/opt/x/LocaLM", "-m", "localm", "gui"]) == \
        "/opt/x/LocaLM -m localm gui"


def test_a_path_with_a_space_is_quoted():
    assert setup_spec.desktop_exec(["/home/a b/x"]) == '"/home/a b/x"'


def _entry_lines(text):
    return dict(line.split("=", 1) for line in text.splitlines()[1:])


def test_menu_entry_launcher_for_a_folder_with_a_space(tmp_path):
    root = tmp_path / "my apps" / "localm"
    (root / "assets").mkdir(parents=True)
    (root / "assets" / "localm.svg").write_text("<svg/>", encoding="utf-8")
    (root / "assets" / "localm.ico").write_bytes(b"\0")
    text = setup_spec.menu_entry("launcher", root)
    assert text.startswith("[Desktop Entry]\n")
    keys = _entry_lines(text)
    assert _parse_exec(keys["Exec"]) == [(root / "localm-launcher.sh").as_posix()]
    assert _string_unescape(keys["Path"]) == root.as_posix()
    assert _string_unescape(keys["Icon"]) == (root / "assets" / "localm.svg").as_posix()
    assert keys["Terminal"] == "false"
    assert keys["Name"] == "LocaLM"


def test_menu_entry_gui_prefers_the_branded_binary(tmp_path, monkeypatch):
    bindir = tmp_path / ".venv" / "bin"
    bindir.mkdir(parents=True)
    (bindir / "LocaLM").write_bytes(b"")
    monkeypatch.setattr(setup_spec.os, "access", lambda p, mode: Path(p).name == "LocaLM")
    keys = _entry_lines(setup_spec.menu_entry("gui", tmp_path))
    assert _parse_exec(keys["Exec"]) == [(bindir / "LocaLM").as_posix(), "-m", "localm", "gui"]
    monkeypatch.setattr(setup_spec.os, "access", lambda p, mode: False)
    keys = _entry_lines(setup_spec.menu_entry("gui", tmp_path))
    assert _parse_exec(keys["Exec"]) == [(bindir / "localm").as_posix(), "gui"]


def test_menu_entry_without_icons_has_no_icon_line(tmp_path):
    assert "Icon=" not in setup_spec.menu_entry("launcher", tmp_path)


def test_menu_entry_refuses_another_kind(tmp_path):
    with pytest.raises(ValueError):
        setup_spec.menu_entry("browser", tmp_path)


def test_menu_entry_cli(tmp_path, capsys):
    assert setup_spec.main(["menu-entry", "launcher", str(tmp_path)]) == 0
    assert capsys.readouterr().out == setup_spec.menu_entry("launcher", tmp_path)
    assert setup_spec.main(["menu-entry", "nope", str(tmp_path)]) == 2


def test_applaunch_desktop_file_uses_the_same_quoting():
    text = applaunch._desktop_entry_text(
        exec_path=PurePosixPath("/home/a b/.venv/bin/LocaLM"),
        workdir=PurePosixPath("/home/a b"), icon=None)
    keys = _entry_lines(text)
    assert _parse_exec(keys["Exec"]) == ["/home/a b/.venv/bin/LocaLM", "-m", "localm", "gui"]


# --------------------------------------------------------------------------- #
#  make-launcher notes                                                         #
# --------------------------------------------------------------------------- #

def test_the_copy_note_does_not_suggest_a_second_menu_entry(tmp_path, monkeypatch):
    repo = tmp_path / "clone"
    repo.mkdir()
    home = tmp_path / "home"
    apps = home / ".local" / "share" / "applications"
    apps.mkdir(parents=True)
    monkeypatch.setattr(applaunch, "_repo_root", lambda: repo)
    monkeypatch.setattr(applaunch.Path, "home", classmethod(lambda cls: home))
    assert applaunch._menu_entry_installed() is None
    (apps / "localm.desktop").write_text(setup_spec.menu_entry("launcher", repo),
                                         encoding="utf-8")
    assert applaunch._menu_entry_installed() == apps / "localm.desktop"
    other = tmp_path / "other"
    (apps / "localm.desktop").write_text(setup_spec.menu_entry("launcher", other),
                                         encoding="utf-8")
    assert applaunch._menu_entry_installed() is None


def test_the_menu_entry_is_found_through_another_spelling_of_the_folder(tmp_path,
                                                                        monkeypatch):
    repo = tmp_path / "clone"
    repo.mkdir()
    home = tmp_path / "home"
    apps = home / ".local" / "share" / "applications"
    apps.mkdir(parents=True)
    monkeypatch.setattr(applaunch, "_repo_root", lambda: repo.resolve())
    monkeypatch.setattr(applaunch.Path, "home", classmethod(lambda cls: home))
    spelled = tmp_path / "clone" / ".." / "clone"
    text = setup_spec.menu_entry("launcher", spelled).replace(
        "Path=" + setup_spec.desktop_string(spelled.as_posix()),
        "Path=" + setup_spec.desktop_string(str(spelled)))
    (apps / "localm.desktop").write_text(text, encoding="utf-8")
    assert applaunch._menu_entry_installed() == apps / "localm.desktop"


def test_desktop_string_round_trips():
    for value in ["/plain", "/a b/c", "/back\\slash", "/new\nline", "/tab\there"]:
        assert setup_spec.desktop_unstring(setup_spec.desktop_string(value)) == value


def test_unsupported_platform_reports_a_problem(monkeypatch):
    monkeypatch.setattr(applaunch.sys, "platform", "sunos5")
    res = applaunch.make_launcher()
    assert res.ok is False
    assert res.problems == res.notes and res.problems


def test_quiet_make_launcher_prints_only_problems(monkeypatch):
    from click.testing import CliRunner

    from localm.cli.maintenance import make_launcher_cmd
    res = applaunch.LauncherResult(ok=True, path=Path("x"),
                                   notes=["built LocaLM.exe from python.exe",
                                          "could not stamp the exe icon"],
                                   problems=["could not stamp the exe icon"])
    monkeypatch.setattr(applaunch, "make_launcher", lambda **_: res)
    quiet = CliRunner().invoke(make_launcher_cmd, ["--quiet"])
    loud = CliRunner().invoke(make_launcher_cmd, [])
    assert "could not stamp" in quiet.output
    assert "built LocaLM.exe" not in quiet.output
    assert "built LocaLM.exe" in loud.output


def test_a_stale_old_copy_is_removed_by_the_next_plain_copy(tmp_path):
    src = tmp_path / "python.exe"
    src.write_bytes(b"MZnew")
    dst = tmp_path / "LocaLM.exe"
    dst.write_bytes(b"MZold")
    stale = tmp_path / "LocaLM.exe.old"
    stale.write_bytes(b"MZolder")
    assert applaunch._copy_replacing_possibly_running_exe(src, dst) is False
    assert dst.read_bytes() == b"MZnew"
    assert not stale.exists()


# --------------------------------------------------------------------------- #
#  LocaLM.exe started with no arguments                                        #
# --------------------------------------------------------------------------- #

def _run_bare_line(monkeypatch, *, orig_argv, executable):
    called = []
    monkeypatch.setattr(sys, "orig_argv", orig_argv)
    monkeypatch.setattr(sys, "executable", executable)
    monkeypatch.setattr(applaunch, "start_gui_from_bare_exe", lambda: called.append(1))
    exec(compile(applaunch.BARE_START_LINE, applaunch.BARE_START_PTH, "exec"), {})  # noqa: S102
    return called


def test_bare_localm_exe_starts_the_gui(monkeypatch):
    assert _run_bare_line(monkeypatch, orig_argv=["C:\\i\\.venv\\localm-app\\LocaLM.exe"],
                          executable="C:\\i\\.venv\\localm-app\\LocaLM.exe") == [1]


def test_localm_exe_with_arguments_is_left_alone(monkeypatch):
    assert _run_bare_line(monkeypatch, orig_argv=["LocaLM.exe", "-m", "localm", "gui"],
                          executable="C:\\i\\LocaLM.exe") == []


def test_another_interpreter_without_arguments_is_left_alone(monkeypatch):
    assert _run_bare_line(monkeypatch, orig_argv=["python.exe"],
                          executable="C:\\i\\.venv\\Scripts\\python.exe") == []


def test_the_bare_start_runs_localm_gui_owning_its_console(monkeypatch):
    monkeypatch.delenv("LOCALM_DEBUG", raising=False)
    monkeypatch.delenv("LOCALM_OWN_CONSOLE", raising=False)
    monkeypatch.setattr(applaunch, "_owns_console", lambda: True)
    ran, exits = [], []
    applaunch.start_gui_from_bare_exe(run=lambda argv, env: ran.append((argv, env)) or 7,
                                      exit_now=exits.append)
    argv, env = ran[0]
    assert argv == [sys.executable, "-m", "localm", "gui"]
    assert env["LOCALM_OWN_CONSOLE"] == "1"
    assert exits == [7]


def test_the_bare_start_in_debug_keeps_the_console(monkeypatch):
    monkeypatch.setenv("LOCALM_DEBUG", "1")
    monkeypatch.delenv("LOCALM_OWN_CONSOLE", raising=False)
    monkeypatch.setattr(applaunch, "_owns_console", lambda: True)
    _, env = applaunch.bare_start_command()
    assert "LOCALM_OWN_CONSOLE" not in env


def test_the_bare_start_never_hides_a_terminal_the_person_also_uses(monkeypatch):
    monkeypatch.delenv("LOCALM_DEBUG", raising=False)
    monkeypatch.delenv("LOCALM_OWN_CONSOLE", raising=False)
    monkeypatch.setattr(applaunch, "_owns_console", lambda: False)
    _, env = applaunch.bare_start_command()
    assert "LOCALM_OWN_CONSOLE" not in env


@pytest.mark.parametrize("rc, code", [(0, 0), (3, 3), (0xC0000005, -1073741819),
                                      (0xC000013A, -1073741510), (-1073741819, -1073741819)])
def test_a_windows_exit_code_reaches_exit_as_a_c_int(monkeypatch, rc, code):
    monkeypatch.setattr(applaunch, "_owns_console", lambda: False)
    exits = []
    applaunch.start_gui_from_bare_exe(run=lambda argv, env: rc, exit_now=exits.append)
    assert exits == [code]


def test_ctrl_c_while_waiting_exits_instead_of_reaching_site(monkeypatch):
    monkeypatch.setattr(applaunch, "_owns_console", lambda: False)

    def interrupted(argv, env):
        raise KeyboardInterrupt
    exits = []
    applaunch.start_gui_from_bare_exe(run=interrupted, exit_now=exits.append)
    assert exits == [130]


def test_a_bare_start_that_cannot_run_says_so(monkeypatch, capsys):
    monkeypatch.setattr(applaunch, "_owns_console", lambda: False)
    def boom(argv, env):
        raise OSError("no such file")
    exits = []
    applaunch.start_gui_from_bare_exe(run=boom, exit_now=exits.append)
    assert exits == [1]
    assert "could not start the GUI: no such file" in capsys.readouterr().err


def test_the_pth_is_written_once_into_site_packages(tmp_path, monkeypatch):
    site = tmp_path / "Lib" / "site-packages"
    site.mkdir(parents=True)
    monkeypatch.setattr(applaunch, "_venv_root", lambda: tmp_path)
    notes, problems = [], []
    applaunch._ensure_bare_start(notes, problems)
    assert (site / applaunch.BARE_START_PTH).read_text(encoding="utf-8") == \
        applaunch.BARE_START_LINE
    assert problems == [] and len(notes) == 1
    applaunch._ensure_bare_start(notes, problems)
    assert len(notes) == 1


def test_no_site_packages_is_a_problem(tmp_path, monkeypatch):
    monkeypatch.setattr(applaunch, "_venv_root", lambda: tmp_path)
    notes, problems = [], []
    applaunch._ensure_bare_start(notes, problems)
    assert problems and "Python prompt" in problems[0]
