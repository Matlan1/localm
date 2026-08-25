# SPDX-License-Identifier: AGPL-3.0-or-later
"""A plugin name, scope, version, description, or dependency list shown by the
`plugin` CLI must survive verbatim - Rich's ``Console.print()`` parses ``[...]``
in ANY interpolated string as markup, not just inside a command's own literal
``[style]`` tags. Reproduced directly against this venv's rich (same repro as
test_rag_cli_markup_escaping.py):

    Console().print('widget[draft]')       -> prints "widget"
    Console().print('widget[bold red]')    -> prints "widget"

The bracketed span is either dropped outright or consumed as a (bogus) style
directive, in both cases silently.

Unlike a collection name (rag.py), a plugin's ``name`` field IS restricted to
a safe character class before it ever reaches a print site
(``name.replace("-", "_").isidentifier()`` in ``engine.parse_spec`` - no
brackets possible), so this file cannot construct a NAME that exercises the
bug and does not try to. ``scope``, ``version``, ``description``, ``requires``
and ``requires_extras`` are read from the SAME manifest with NO such
restriction (plain ``str()``/``list()`` casts in ``parse_spec``), so those are
the fields exercised here with real, on-disk third-party plugin manifests -
the same convention test_cli_plugin.py's ``cli_env`` fixture uses (a
synthetic store/installed pair, ``PluginManager`` pointed at them).

A CLI-typed argument (an unknown plugin NAME/TARGET/KEY passed to install/
enable/disable/uninstall/config) is a second, independent risk: that text is
never manifest-validated at all, so TestUnknownPluginArgumentMarkupEscaping
exercises it directly. TestInstallErrorMarkupEscaping and
TestSetupSkipErrorMarkupEscaping force a real ``ValueError`` the same way
test_rag_cli_markup_escaping.py's TestLockMessageEscaping does, since the
exact wording of a collision/validation error is not something a fixture can
reliably reproduce on disk.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

# One value Rich DROPS outright, one it consumes as a (bogus) style tag - the
# two distinct failure shapes described in the module docstring above.
BRACKET_DROP = "[draft]"
BRACKET_STYLE = "[bold red]"


def _write_plugin(root, name, *, scope=None, version=None, description=None,
                   requires=None, requires_extras=None):
    """A minimal on-disk 'engine' plugin (register=, not entry=) under *root*,
    matching test_cli_plugin.py's own synthetic-plugin shape."""
    d = root / name
    d.mkdir(parents=True)
    lines = ["[plugin]", f'name = "{name}"']
    if scope is not None:
        lines.append(f'scope = "{scope}"')
    if version is not None:
        lines.append(f'version = "{version}"')
    if description is not None:
        lines.append(f'description = "{description}"')
    lines.append('register = "plug"')
    if requires:
        rq = ", ".join(f'"{r}"' for r in requires)
        lines.append(f"requires = [{rq}]")
    if requires_extras:
        rx = ", ".join(f'"{r}"' for r in requires_extras)
        lines.append(f"requires_extras = [{rx}]")
    (d / "plugin.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "plug.py").write_text(
        "def register(host):\n    pass\n\ndef unregister():\n    pass\n",
        encoding="utf-8")
    return d


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Own copy of test_cli_plugin.py's fixture (module-private there), with
    an EMPTY store/installed pair this file populates per-test via
    _write_plugin - the fixed 'needy'/'dep1' pair does not carry the manifest
    fields (scope/version/description/requires) these tests need to control."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")

    store = tmp_path / "store"
    installed = tmp_path / "installed"
    store.mkdir()
    installed.mkdir()

    import localm.cli as climod
    from localm.plugins.engine import PluginManager
    monkeypatch.setattr(
        climod, "_engine_manager",
        lambda: PluginManager(None, store_root=store, installed_root=installed))
    from types import SimpleNamespace
    return SimpleNamespace(main=climod.main, store=store, installed=installed)


@pytest.fixture
def runner():
    return CliRunner()


# --------------------------------------------------------------------------- #
#  Third-party install: scope, version - genuinely unvalidated manifest text  #
# --------------------------------------------------------------------------- #

class TestInstallFromDirectoryMarkupEscaping:
    def test_scope_bracket_drop_survives_verbatim(self, runner, cli_env, tmp_path):
        ext = tmp_path / "thirdparty"
        _write_plugin(ext.parent, "thirdparty", scope=f"widget{BRACKET_DROP}")
        r = runner.invoke(cli_env.main, ["plugin", "install", str(ext)])
        assert r.exit_code == 0, r.output
        assert f"widget{BRACKET_DROP}" in r.output, (
            f"the 'Granted scope' line must show the manifest's scope "
            f"verbatim, not silently drop the bracketed span: {r.output!r}")

    def test_scope_bracket_style_survives_verbatim(self, runner, cli_env, tmp_path):
        ext = tmp_path / "thirdparty"
        _write_plugin(ext.parent, "thirdparty", scope=f"widget{BRACKET_STYLE}")
        r = runner.invoke(cli_env.main, ["plugin", "install", str(ext)])
        assert r.exit_code == 0, r.output
        assert f"widget{BRACKET_STYLE}" in r.output, (
            f"a scope segment that looks like a style tag must be shown as "
            f"literal text, not consumed as Rich styling: {r.output!r}")

    def test_version_survives_verbatim(self, runner, cli_env, tmp_path):
        ext = tmp_path / "thirdparty"
        _write_plugin(ext.parent, "thirdparty", version=f"1.0{BRACKET_DROP}")
        r = runner.invoke(cli_env.main, ["plugin", "install", str(ext)])
        assert r.exit_code == 0, r.output
        assert f"1.0{BRACKET_DROP}" in r.output, (
            f"the install success line must show the manifest version "
            f"verbatim: {r.output!r}")

    def test_requires_extras_survives_verbatim(self, runner, cli_env, tmp_path):
        ext = tmp_path / "thirdparty"
        _write_plugin(ext.parent, "thirdparty",
                       requires_extras=[f"extra{BRACKET_STYLE}"])
        r = runner.invoke(cli_env.main, ["plugin", "install", str(ext)])
        assert r.exit_code == 0, r.output
        assert f"extra{BRACKET_STYLE}" in r.output, (
            f"the 'declares extra dependencies' note must name the real "
            f"extra verbatim: {r.output!r}")


class TestWarnMissingRequiresMarkupEscaping:
    def test_missing_dependency_name_survives_verbatim(
            self, runner, cli_env, tmp_path):
        """A declared 'requires' entry that names a plugin NOT installed - the
        entry is free text in THIS plugin's own manifest, not the dependency's
        own (validated) name field, so it carries no isidentifier() guarantee."""
        ext = tmp_path / "thirdparty"
        _write_plugin(ext.parent, "thirdparty", requires=[f"dep{BRACKET_DROP}"])
        r = runner.invoke(cli_env.main, ["plugin", "install", str(ext)])
        assert r.exit_code == 0, r.output
        assert f"dep{BRACKET_DROP}" in r.output, (
            f"the missing-dependency note must name the real dependency "
            f"verbatim, twice (the note text and the install command it "
            f"suggests): {r.output!r}")
        assert f"localm plugin install dep{BRACKET_DROP}" in " ".join(
            r.output.split()), (
            f"the suggested install command must carry the real name, not a "
            f"mangled one a user would copy-paste and get wrong: {r.output!r}")


# --------------------------------------------------------------------------- #
#  `plugin status`: an INSTALLED plugin's own description (its own manifest)  #
# --------------------------------------------------------------------------- #

class TestPluginStatusMarkupEscaping:
    def test_installed_description_survives_verbatim(
            self, runner, cli_env, tmp_path):
        ext = tmp_path / "thirdparty"
        _write_plugin(ext.parent, "thirdparty",
                      description=f"does things{BRACKET_STYLE}")
        add = runner.invoke(cli_env.main, ["plugin", "install", str(ext)])
        assert add.exit_code == 0, add.output

        r = runner.invoke(cli_env.main, ["plugin", "status"])
        assert r.exit_code == 0, r.output
        assert f"does things{BRACKET_STYLE}" in r.output, (
            f"an installed plugin's own description must survive verbatim in "
            f"'plugin status', even though it comes from a THIRD PARTY's own "
            f"plugin.toml: {r.output!r}")


# --------------------------------------------------------------------------- #
#  CLI-typed arguments: never manifest-validated at all                       #
# --------------------------------------------------------------------------- #

class TestUnknownPluginArgumentMarkupEscaping:
    """install/uninstall/enable/disable all build ``missing_msg=f"No such
    plugin: {target}"`` and hand it to the SHARED ``run_or_die`` (errors.py,
    out of scope for this file) - escaped here, at the point the value is
    interpolated, so the message is already markup-safe by the time
    run_or_die prints it, regardless of that file's own state."""

    def test_uninstall_unknown_name_survives_verbatim(self, runner, cli_env):
        """NOT 'plugin install': that path resolves the target through
        _installed_dir()/_check_plugin_name FIRST (path-traversal defence,
        engine.py:1119), which raises its OWN ValueError for any non-identifier
        name before this file's missing_msg is ever built - a different,
        pre-existing, already-unescaped site in engine.py/errors.py, out of
        scope here. 'plugin uninstall' checks membership in a dict/set first
        (no shape validation), so it reaches missing_msg via a clean KeyError."""
        bad = f"ghost{BRACKET_DROP}"
        r = runner.invoke(cli_env.main, ["plugin", "uninstall", bad])
        assert r.exit_code == 1, r.output
        assert bad in r.output, (
            f"'No such plugin' must echo the exact typed name verbatim, so "
            f"the user can see what they actually typed: {r.output!r}")

    def test_enable_unknown_name_survives_verbatim(self, runner, cli_env):
        bad = f"ghost{BRACKET_STYLE}"
        r = runner.invoke(cli_env.main, ["plugin", "enable", bad])
        assert r.exit_code == 1, r.output
        assert bad in r.output, (
            f"'plugin enable' on an unknown name must echo it verbatim: "
            f"{r.output!r}")

    def test_config_unknown_key_survives_verbatim(self, runner, cli_env):
        """image/music/video/tts have a static, developer-defined key schema,
        so the KEY here is never validated against the manifest - it is the
        raw CLI argument, exactly like the plugin names above."""
        bad_key = f"workdir{BRACKET_DROP}"
        r = runner.invoke(cli_env.main, ["plugin", "config", "image", bad_key])
        assert r.exit_code == 1, r.output
        assert bad_key in r.output, (
            f"'Unknown setting for image' must echo the mistyped key "
            f"verbatim: {r.output!r}")


# --------------------------------------------------------------------------- #
#  A local (media) plugin setting's own VALUE - round-tripped through         #
#  _report_set (write) and _fmt_field_value (read), the two shared helpers    #
#  every plugin-config display path funnels through.                         #
# --------------------------------------------------------------------------- #

class TestPluginConfigValueMarkupEscaping:
    def test_set_then_read_back_survives_verbatim(self, runner, cli_env):
        value = f"/srv/comfy{BRACKET_STYLE}"
        w = runner.invoke(cli_env.main,
                          ["plugin", "config", "image", "workdir", value])
        assert w.exit_code == 0, w.output
        assert value in w.output, (
            f"the '... = <value>' confirmation on WRITE must show the stored "
            f"value verbatim (_report_set): {w.output!r}")

        r = runner.invoke(cli_env.main, ["plugin", "config", "image", "workdir"])
        assert r.exit_code == 0, r.output
        assert value in r.output, (
            f"reading the field back must show the same value verbatim "
            f"(_fmt_field_value): {r.output!r}")


# --------------------------------------------------------------------------- #
#  A real domain ValueError, forced via monkeypatch (test_rag_cli_markup_     #
#  escaping.py's TestLockMessageEscaping does the same for its own exception) #
# --------------------------------------------------------------------------- #

class TestInstallErrorMarkupEscaping:
    def test_directory_install_failure_message_survives_verbatim(
            self, runner, cli_env, tmp_path, monkeypatch):
        from localm.plugins.engine import PluginManager
        bad = f"scope 'x{BRACKET_DROP}' is already taken"

        def _boom(self, source, *, force=False, enable=True):
            raise ValueError(bad)
        monkeypatch.setattr(PluginManager, "set_installed_from_dir", _boom)

        ext = tmp_path / "thirdparty"
        _write_plugin(ext.parent, "thirdparty")
        r = runner.invoke(cli_env.main, ["plugin", "install", str(ext)])
        assert r.exit_code == 1, r.output
        assert bad in r.output, (
            f"a ValueError from set_installed_from_dir must be shown "
            f"verbatim, not mangled by markup parsing: {r.output!r}")


class TestSetupSkipErrorMarkupEscaping:
    def test_setup_skip_message_survives_verbatim(
            self, runner, cli_env, tmp_path, monkeypatch):
        """'plugin setup --plugins <name>' hits a DIFFERENT except (KeyError,
        ValueError) site than 'plugin install' above (line ~304, 'Skipped
        NAME: ERROR'), reached only through the catalog-driven setup path."""
        from localm.plugins import catalog
        from localm.plugins.engine import PluginManager

        entry = catalog.CatalogEntry("widget", "A widget plugin")
        monkeypatch.setattr(catalog, "CATALOG", (entry,))

        bad = f"conflicts with x{BRACKET_STYLE}"

        def _boom(self, name, on, *, enable=True):
            raise ValueError(bad)
        monkeypatch.setattr(PluginManager, "set_installed_state", _boom)

        r = runner.invoke(cli_env.main,
                          ["plugin", "setup", "--plugins", "widget"])
        assert r.exit_code == 0, r.output   # a per-plugin skip, not a hard exit
        assert bad in r.output, (
            f"'Skipped <name>: <error>' must show the real ValueError text "
            f"verbatim: {r.output!r}")
