# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rich markup escaping in localm/cli/comfy.py.

`_report_action`, `comfy_setup` and `comfy_update`'s `result.message` prints
already wrap with `rich.markup.escape()` and are not covered here. This file
covers workflow names, the LOCALM_HOME-derived install paths shown by
`comfy status`/`comfy remove`, the config-driven `comfy_target`/`comfy_api_url`
values, and the `--commit` preview line in `comfy update`.

`rich.console.Console.print()`, and `rich.table.Table.add_row()` too, parse
any '[...]' in an interpolated string as markup:

    Console().print('flux[draft].json')        -> prints "flux.json"
    Console().print('flux[bold red].json')      -> prints "flux.json"

The bracketed span is either dropped outright or consumed as a style
directive, in both cases silently.

`media_workflows.save_workflow`'s only validation is path-traversal safety
(`pathsafe.confined_name`, which blocks `<>:"|?*` and control characters but
NOT `[`/`]`), so `localm comfy workflow add image "my[flux].json"` is an
accepted filename. LOCALM_HOME and comfy_api_url are user-configurable;
comfy_target is SELECT-validated through the setters, but config.json can be
hand-edited.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

import localm.cli.comfy as comfy_cli

# One name Rich drops outright, one it consumes as a style tag.
BRACKET_DROP_NAME = "flux[draft].json"
BRACKET_STYLE_NAME = "flux[bold red].json"

_WF = json.dumps({"3": {"class_type": "KSampler", "inputs": {}},
                  "4": {"class_type": "SaveImage", "inputs": {}}}).encode()


def _write_workflow_file(tmp_path, name, content=_WF):
    p = tmp_path / name
    p.write_bytes(content)
    return p


@pytest.fixture
def bracket_home_runner(tmp_path, monkeypatch):
    """A CliRunner whose LOCALM_HOME basename itself contains a bracket.

    Used by the comfy_status 'Installed: yes, at <path>' assertions, which run
    against a real, unmocked `managed_comfy_paths()` derived from home_dir().
    """
    import localm.config as cfg
    home = tmp_path / "home[legacy]" / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    # rich.console.Console() reads COLUMNS at construction time; without it the
    # non-tty default of 80 hard-wraps mid-word inside a long basetemp path.
    from tests.conftest import make_console_wide_and_plain
    make_console_wide_and_plain(monkeypatch, width="300")
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return CliRunner()


# ---------------------------------------------------------------------------
#  workflow add
# ---------------------------------------------------------------------------

class TestWorkflowAddMarkupEscaping:
    def test_saved_message_shows_bracket_drop_name_verbatim(self, cli_runner, tmp_path):
        f = _write_workflow_file(tmp_path, BRACKET_DROP_NAME)
        result = cli_runner.invoke(comfy_cli.workflow_add, ["image", str(f)])
        assert result.exit_code == 0, result.output
        saved_line = next(l for l in result.output.splitlines() if "Saved" in l)
        assert BRACKET_DROP_NAME in saved_line, (
            f"the 'Saved ... workflow NAME' line must show the real stored "
            f"filename verbatim, not silently mangled by Rich markup "
            f"parsing: {saved_line!r}")

    def test_select_hint_shows_bracket_style_name_verbatim(self, cli_runner, tmp_path):
        f = _write_workflow_file(tmp_path, BRACKET_STYLE_NAME)
        result = cli_runner.invoke(comfy_cli.workflow_add, ["image", str(f)])
        assert result.exit_code == 0, result.output
        hint_line = next(l for l in result.output.splitlines()
                         if "Select it with" in l or "workflow use" in l)
        assert BRACKET_STYLE_NAME in hint_line, (
            f"a filename segment shaped like a style tag must be shown as "
            f"literal text in the 'workflow use' hint, not consumed as Rich "
            f"styling: {hint_line!r}")


# ---------------------------------------------------------------------------
#  workflow list
# ---------------------------------------------------------------------------

class TestWorkflowListMarkupEscaping:
    def test_active_header_shows_bracket_drop_name_verbatim(self, cli_runner, tmp_path):
        f = _write_workflow_file(tmp_path, BRACKET_DROP_NAME)
        cli_runner.invoke(comfy_cli.workflow_add, ["image", str(f), "--use"])

        result = cli_runner.invoke(comfy_cli.workflow_list, ["image"])
        assert result.exit_code == 0, result.output
        header = result.output.splitlines()[0]
        assert BRACKET_DROP_NAME in header, (
            f"the '<media> workflow: <active>' header must show the active "
            f"workflow's real name verbatim: {header!r}")

    def test_table_row_shows_bracket_style_name_verbatim(self, cli_runner, tmp_path):
        """Table.add_row() cell strings go through the same markup parsing as
        console.print f-strings, not plain text."""
        f = _write_workflow_file(tmp_path, BRACKET_STYLE_NAME)
        cli_runner.invoke(comfy_cli.workflow_add, ["image", str(f)])   # not active

        result = cli_runner.invoke(comfy_cli.workflow_list, ["image"])
        assert result.exit_code == 0, result.output
        assert BRACKET_STYLE_NAME in result.output, (
            f"the workflow table's Name column must show the real filename "
            f"verbatim, not have '[bold red]' consumed as styling: "
            f"{result.output!r}")


# ---------------------------------------------------------------------------
#  workflow use
# ---------------------------------------------------------------------------

class TestWorkflowUseMarkupEscaping:
    def test_now_uses_message_shows_bracket_name_verbatim(self, cli_runner, tmp_path):
        f = _write_workflow_file(tmp_path, BRACKET_DROP_NAME)
        cli_runner.invoke(comfy_cli.workflow_add, ["image", str(f)])   # not active yet

        result = cli_runner.invoke(comfy_cli.workflow_use, ["image", BRACKET_DROP_NAME])
        assert result.exit_code == 0, result.output
        assert BRACKET_DROP_NAME in result.output, (
            f"the '<media> now uses NAME' message must show the selected "
            f"workflow's real name verbatim: {result.output!r}")


# ---------------------------------------------------------------------------
#  workflow rm
# ---------------------------------------------------------------------------

class TestWorkflowRmMarkupEscaping:
    def test_deleted_message_shows_bracket_style_name_verbatim(self, cli_runner, tmp_path):
        f = _write_workflow_file(tmp_path, BRACKET_STYLE_NAME)
        cli_runner.invoke(comfy_cli.workflow_add, ["image", str(f)])

        result = cli_runner.invoke(
            comfy_cli.workflow_rm, ["image", BRACKET_STYLE_NAME, "-y"])
        assert result.exit_code == 0, result.output
        assert BRACKET_STYLE_NAME in result.output, (
            f"the 'Deleted ... workflow NAME' message must show the real "
            f"name verbatim: {result.output!r}")


# ---------------------------------------------------------------------------
#  comfy status (defense-in-depth: config-driven and LOCALM_HOME-derived)
# ---------------------------------------------------------------------------

class TestComfyStatusMarkupEscaping:
    def test_preferred_target_config_value_survives_verbatim(self, cli_runner):
        """comfy_target is SELECT-validated (own/user) through the GUI/CLI
        setters, but config.json can be hand-edited to anything, and this
        print does not rely on that validation."""
        from localm.config import update_config
        bracket_value = "user[legacy]"
        update_config(lambda cfg: cfg.__setitem__("comfy_target", bracket_value))

        result = cli_runner.invoke(comfy_cli.comfy_status, ["--no-ping"])
        assert result.exit_code == 0, result.output
        target_line = next(l for l in result.output.splitlines()
                           if "Preferred target" in l)
        assert bracket_value in target_line, (
            f"the 'Preferred target' line must show the raw config value "
            f"verbatim: {target_line!r}")

    def test_target_api_url_survives_verbatim(self, cli_runner):
        """target.api_url is comfy_api_url, admin-set free text, whenever no
        managed instance is active. The managed case instead uses
        MANAGED_COMFY_API_URL, a fixed loopback constant, which is not
        escaped."""
        from localm.config import update_config
        bracket_url = "http://127.0.0.1:9999/comfy[legacy]"
        update_config(lambda cfg: cfg.__setitem__("comfy_api_url", bracket_url))

        result = cli_runner.invoke(comfy_cli.comfy_status, ["--no-ping"])
        assert result.exit_code == 0, result.output
        target_now_line = next(l for l in result.output.splitlines()
                               if "Target now" in l)
        assert bracket_url in target_now_line, (
            f"the 'Target now' line must show the configured comfy_api_url "
            f"verbatim: {target_now_line!r}")

    def test_installed_paths_survive_verbatim(self, bracket_home_runner, monkeypatch):
        """paths.root/paths.models_dir are <LOCALM_HOME>/comfyui(-models), and
        LOCALM_HOME is user-configurable through the environment. Uses the
        real managed_comfy_paths() derived from bracket_home_runner's home
        dir; only is_managed_comfy_installed is mocked."""
        from localm.media import managed_comfy as mc_mod
        monkeypatch.setattr(mc_mod, "is_managed_comfy_installed", lambda: True)

        result = bracket_home_runner.invoke(comfy_cli.comfy_status, ["--no-ping"])
        assert result.exit_code == 0, result.output
        installed_line = next(l for l in result.output.splitlines()
                              if "Installed" in l and "yes" in l)
        models_line = next(l for l in result.output.splitlines()
                           if "Managed models" in l)
        assert "home[legacy]" in installed_line, (
            f"the 'Installed: yes, at <path>' line must show the real "
            f"LOCALM_HOME-derived path verbatim: {installed_line!r}")
        assert "home[legacy]" in models_line, (
            f"the 'Managed models' line must show the real path verbatim: "
            f"{models_line!r}")


# ---------------------------------------------------------------------------
#  comfy remove
# ---------------------------------------------------------------------------

class TestComfyRemoveMarkupEscaping:
    def test_delete_listing_shows_bracket_path_verbatim(self, cli_runner, monkeypatch):
        from pathlib import Path

        from localm.media import managed_comfy as mc_mod
        target = Path("/fake/home[legacy]/comfyui")
        monkeypatch.setattr(mc_mod, "managed_comfy_remove_targets",
                            lambda with_models: [target])
        monkeypatch.setattr(mc_mod, "remove_managed_comfy",
                            lambda with_models: ([target], []))

        result = cli_runner.invoke(comfy_cli.comfy_remove, ["-y"])
        assert result.exit_code == 0, result.output
        listing_line = next(l for l in result.output.splitlines()
                            if "home[legacy]" in l)
        assert str(target) in listing_line, (
            f"the 'This will delete:' listing must show the real target "
            f"path verbatim: {listing_line!r}")

    def test_could_not_remove_shows_bracket_failure_verbatim(self, cli_runner, monkeypatch):
        from pathlib import Path

        from localm.media import managed_comfy as mc_mod
        target = Path("/fake/home[legacy]/comfyui")
        failure_text = f"{target} (PermissionError: in use)"
        monkeypatch.setattr(mc_mod, "managed_comfy_remove_targets",
                            lambda with_models: [target])
        monkeypatch.setattr(mc_mod, "remove_managed_comfy",
                            lambda with_models: ([], [failure_text]))

        result = cli_runner.invoke(comfy_cli.comfy_remove, ["-y"])
        assert result.exit_code == 1, result.output
        # "home[legacy]" also appears on the earlier "This will delete:"
        # listing line, so key on the failure-specific text to pick the
        # "Could not remove:" line.
        failed_line = next(l for l in result.output.splitlines()
                           if "PermissionError" in l)
        assert failure_text in failed_line, (
            f"the 'Could not remove:' line must show the real failure text "
            f"verbatim, including the path and the OS error: {failed_line!r}")


# ---------------------------------------------------------------------------
#  comfy update
# ---------------------------------------------------------------------------

class TestComfyUpdateMarkupEscaping:
    def test_advanced_commit_preview_survives_verbatim(self, cli_runner, monkeypatch):
        """The --commit preview line, printed before update_managed_comfy runs
        and distinct from result.message. --commit is raw user input; the
        sibling branch prints COMFYUI_PINNED_COMMIT, a hardcoded module
        constant, unescaped."""
        from localm.media import managed_comfy as mc_mod
        from localm.media import managed_comfy_update as upd_mod
        from localm.media.managed_comfy_provision import ProvisionResult

        monkeypatch.setattr(mc_mod, "is_managed_comfy_installed", lambda: True)
        monkeypatch.setattr(
            upd_mod, "update_managed_comfy",
            lambda *a, **k: ProvisionResult(ok=True, status="updated",
                                            message="done", installed_packages=0,
                                            custom_nodes_copied=0))

        # The complete bracket pair must fall WITHIN the first 12 characters:
        # commit[:12] truncates before interpolation, and an unclosed "["
        # fragment is printed literally whether escaped or not.
        bracket_commit = "ab[cd]ef1234567890"
        result = cli_runner.invoke(comfy_cli.comfy_update, ["--commit", bracket_commit])
        assert result.exit_code == 0, result.output
        preview_line = next(l for l in result.output.splitlines()
                            if "Updating to ComfyUI" in l)
        # Only the first 12 chars are shown (commit[:12]).
        assert bracket_commit[:12] in preview_line, (
            f"the '--commit' advanced-override preview must show the "
            f"truncated commit verbatim, not have '[bold red]' consumed as "
            f"styling: {preview_line!r}")
