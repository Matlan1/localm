# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm plugin setup` - the installer's plugin-selection step.

Out of the box only chat is active; setup lets the user (or the installer, or a
script via --plugins/--all/--defaults) turn on the first-party plugins they want.
Driven through Click's CliRunner against the real bundled store, with a throwaway
LOCALM_HOME so installs land in tmp.

EVERY setup invocation here passes --no-deps (or with_deps=False when calling the
callback directly), and that is load-bearing, not tidiness.

A throwaway LOCALM_HOME isolates the DATA dir. It does not isolate the
INTERPRETER, and a plugin's pip extras go to the interpreter: `--all` includes
voice, whose extra is faster-whisper, so this file used to run a real
`uv pip install --python <the venv running the suite> faster-whisper>=1.2.1`.
That is the isolation blind spot - the author isolated the obvious thing and the
install went somewhere else entirely.

The damage was not "a slow test". numpy arrives as a transitive dependency of
faster-whisper, and WHILE THE WHEEL UNPACKS, site-packages/numpy exists with no
__init__.py yet - a PEP 420 namespace package, so `import numpy` SUCCEEDS and
hands back a module with no attributes. Any test in any other xdist worker that
touched numpy inside that window died with
`AttributeError: module 'numpy' has no attribute 'asarray'` from rag/store.py,
hundreds of lines from the cause, and the install then completed and erased the
evidence. It reddened windows CI at roughly 1 in 3 across unrelated branches for
a day. Run locally, the same line installs into the developer's shared venv.

The deps orchestration itself is covered in tests/test_plugin_deps_cli.py with
_run_pip mocked, so nothing is lost by disabling it here. tests/conftest.py now
BLOCKS an install aimed at the running interpreter, so a regression fails
immediately, in the test that caused it, instead of somewhere else an hour later.
"""

import pytest
from click.testing import CliRunner


@pytest.fixture
def home_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _installed():
    from localm.plugins.engine import PluginManager
    return {p["name"] for p in PluginManager(None).api_state()["plugins"] if p.get("installed")}


def test_setup_installs_named_plugins(home_env):
    from localm.cli import main
    r = CliRunner().invoke(main, ["plugin", "setup", "--no-deps", "--plugins", "web,rag"])
    assert r.exit_code == 0, r.output
    inst = _installed()
    assert "web" in inst and "rag" in inst
    assert "image" not in inst        # only the selected ones


def test_setup_defaults_installs_recommended_set(home_env):
    from localm.cli import main
    r = CliRunner().invoke(main, ["plugin", "setup", "--no-deps", "--defaults"])
    assert r.exit_code == 0, r.output
    inst = _installed()
    for n in ("coder", "rag", "web", "tts"):
        assert n in inst, f"{n} should be in the default set"
    assert "video" not in inst        # heavy/opt-in plugins are not defaulted


def test_setup_all_installs_every_first_party_plugin(home_env):
    from localm.cli import main
    r = CliRunner().invoke(main, ["plugin", "setup", "--no-deps", "--all"])
    assert r.exit_code == 0, r.output
    inst = _installed()
    for n in ("coder", "image", "music", "video", "rag", "web", "voice", "tts", "mcp"):
        assert n in inst, f"--all should install {n}"


def test_setup_non_interactive_shell_skips_without_flags(home_env):
    # CliRunner's stdin is not a tty: with no flags, setup must skip (not hang,
    # not install anything) so a scripted/CI install does not block.
    from localm.cli import main
    r = CliRunner().invoke(main, ["plugin", "setup"])
    assert r.exit_code == 0, r.output
    assert _installed() == set() or "chat" in _installed()  # nothing newly installed
    assert "skipping" in r.output.lower()


def test_setup_unknown_name_is_skipped_not_fatal(home_env):
    from localm.cli import main
    r = CliRunner().invoke(main, ["plugin", "setup", "--no-deps", "--plugins", "web,bogusplugin"])
    assert r.exit_code == 0, r.output
    inst = _installed()
    assert "web" in inst
    assert "bogusplugin" not in inst


# --------------------------------------------------------------------------- #
# R21 / SETUP-3: the interactive prompt parses ranges, de-dups, and re-asks on  #
# an all-junk entry (never silently leaves zero plugins). Blank still skips.    #
# --------------------------------------------------------------------------- #

def _available():
    from localm.plugins import catalog
    return [e for e in catalog.CATALOG if not e.preinstalled]


def test_parse_plugin_selection_expands_range_and_dedups():
    from localm.cli import _parse_plugin_selection
    avail = _available()
    names = [e.name for e in avail]
    assert _parse_plugin_selection("1-3", avail) == names[:3]
    # an explicit number overlapping a range is de-duplicated, order preserved
    assert _parse_plugin_selection("1,1-2", avail) == names[:2]
    # mixed names + range
    assert _parse_plugin_selection(f"{names[0]},2-3", avail) == names[:3]


def test_parse_plugin_selection_junk_yields_empty():
    """Pure junk (e.g. 'ewew') and an out-of-range range both parse to nothing,
    which is what makes the interactive caller re-prompt instead of installing
    zero plugins."""
    from localm.cli import _parse_plugin_selection
    avail = _available()
    assert _parse_plugin_selection("ewew", avail) == []
    assert _parse_plugin_selection("99-100", avail) == []
    assert _parse_plugin_selection("5-2", avail) == []   # reversed = empty


def test_parse_plugin_selection_exotic_unicode_digit_no_crash():
    """A pasted exotic 'digit' (str.isdigit() True but int() rejects, e.g. the
    superscripts in 'b2-3') must be flagged and skipped, not crash the range
    branch. Guards the isdecimal (not isdigit) gate."""
    from localm.cli import _parse_plugin_selection
    avail = _available()
    assert _parse_plugin_selection("²-³", avail) == []   # superscript 2-3


def test_interactive_reprompts_on_junk_then_installs(home_env, monkeypatch):
    """SETUP-3: typing 'ewew' must NOT silently leave zero plugins - the prompt
    re-asks, and a valid follow-up entry installs. Driven by calling the command
    callback directly with a forced-tty + scripted prompt answers."""
    from localm import cli
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    answers = iter(["ewew", "1"])   # junk, then the first plugin by index
    monkeypatch.setattr(cli.click, "prompt", lambda *a, **k: next(answers))
    cli.plugin_setup.callback(None, False, False, with_deps=False)
    first = _available()[0].name
    assert first in _installed()


def test_interactive_blank_skips_without_reprompt(home_env, monkeypatch):
    """A blank entry is a deliberate skip: it must NOT re-prompt and must install
    nothing (only one prompt is consumed)."""
    from localm import cli
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    answers = iter([""])            # a single blank; a re-prompt would StopIteration
    monkeypatch.setattr(cli.click, "prompt", lambda *a, **k: next(answers))
    cli.plugin_setup.callback(None, False, False, with_deps=False)
    assert _installed() in (set(), {"chat"})
