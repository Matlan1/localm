# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm comfy workflow list|add|use|rm` - the CLI verbs for per-plugin
ComfyUI workflow management (image/music/video).

They wrap `save_workflow` / `list_workflows` / `select_workflow` /
`delete_workflow` (localm.media_workflows), the same functions the GUI's HTTP
routes call. Those functions are framework-free (no server needed), so the
commands call them directly, the same way `localm config` / `localm plugin
config` work locally against config.json.
"""

from __future__ import annotations

import json

import localm.cli.comfy as comfy_cli

_WF = json.dumps({"3": {"class_type": "KSampler", "inputs": {}},
                  "4": {"class_type": "SaveImage", "inputs": {}}}).encode()


def _write_workflow_file(tmp_path, name="my_flux.json", content=_WF):
    p = tmp_path / name
    p.write_bytes(content)
    return p


# --------------------------------------------------------------------------
#  list
# --------------------------------------------------------------------------

def test_list_on_a_fresh_media_shows_the_built_in_default_and_no_uploads(cli_runner):
    result = cli_runner.invoke(comfy_cli.workflow_list, ["image"])
    assert result.exit_code == 0, result.output
    assert "built-in default" in result.output
    assert "No uploaded workflows" in result.output


def test_list_rejects_an_unknown_media_type(cli_runner):
    result = cli_runner.invoke(comfy_cli.workflow_list, ["not-a-media"])
    assert result.exit_code == 2, result.output   # click usage error


def test_list_shows_uploaded_files_and_marks_the_active_one(cli_runner, tmp_path):
    f = _write_workflow_file(tmp_path)
    cli_runner.invoke(comfy_cli.workflow_add, ["image", str(f)])
    cli_runner.invoke(comfy_cli.workflow_use, ["image", "my_flux.json"])

    result = cli_runner.invoke(comfy_cli.workflow_list, ["image"])
    assert result.exit_code == 0, result.output
    assert "my_flux.json" in result.output
    assert "built-in default" not in result.output.split("\n")[0]


# --------------------------------------------------------------------------
#  add (upload)
# --------------------------------------------------------------------------

def test_add_saves_the_file_under_the_media_workflows_dir(cli_runner, tmp_path):
    from localm import media_workflows as mw

    f = _write_workflow_file(tmp_path)
    result = cli_runner.invoke(comfy_cli.workflow_add, ["image", str(f)])
    assert result.exit_code == 0, result.output
    assert "Saved" in result.output
    assert (mw.workflows_dir("image") / "my_flux.json").is_file()
    # Not selected by default: add and use are separate steps.
    assert mw.selected_name("image") is None


def test_add_with_use_flag_also_selects_it(cli_runner, tmp_path):
    from localm import media_workflows as mw

    f = _write_workflow_file(tmp_path)
    result = cli_runner.invoke(comfy_cli.workflow_add, ["image", str(f), "--use"])
    assert result.exit_code == 0, result.output
    assert "selected" in result.output.lower()
    assert mw.selected_name("image") == "my_flux.json"


def test_add_honours_a_custom_stored_name(cli_runner, tmp_path):
    from localm import media_workflows as mw

    f = _write_workflow_file(tmp_path, name="source.json")
    result = cli_runner.invoke(
        comfy_cli.workflow_add, ["music", str(f), "--name", "renamed.json"])
    assert result.exit_code == 0, result.output
    assert (mw.workflows_dir("music") / "renamed.json").is_file()
    assert not (mw.workflows_dir("music") / "source.json").exists()


def test_add_rejects_a_file_that_is_not_a_comfyui_workflow(cli_runner, tmp_path):
    f = tmp_path / "not_a_workflow.json"
    f.write_bytes(json.dumps({"hello": "world"}).encode())
    result = cli_runner.invoke(comfy_cli.workflow_add, ["image", str(f)])
    assert result.exit_code != 0, result.output
    assert "not a ComfyUI API-format workflow" in result.output


def test_add_rejects_non_json_content(cli_runner, tmp_path):
    f = tmp_path / "garbage.json"
    f.write_bytes(b"not json{{")
    result = cli_runner.invoke(comfy_cli.workflow_add, ["image", str(f)])
    assert result.exit_code != 0, result.output
    assert "not valid JSON" in result.output


def test_add_rejects_a_missing_source_file(cli_runner, tmp_path):
    result = cli_runner.invoke(
        comfy_cli.workflow_add, ["image", str(tmp_path / "ghost.json")])
    assert result.exit_code != 0, result.output


def test_add_with_a_traversal_name_is_a_clean_error_not_a_crash(cli_runner, tmp_path):
    """media_workflows._confined() normalizes to ValueError: its underlying
    path-safety check, confined_name(), raises fastapi.HTTPException, which
    pathsafe.py documents as being for HTTP call sites only and which no
    `except ValueError` in this CLI path would catch."""
    f = _write_workflow_file(tmp_path)
    result = cli_runner.invoke(
        comfy_cli.workflow_add, ["image", str(f), "--name", "../escape.json"])
    assert result.exit_code == 1, result.output
    # Click's own ClickException format.
    assert result.output.startswith("Error:"), result.output
    assert "Invalid file name" in result.output


# --------------------------------------------------------------------------
#  use (select / clear)
# --------------------------------------------------------------------------

def test_use_selects_an_uploaded_workflow(cli_runner, tmp_path):
    from localm import media_workflows as mw

    f = _write_workflow_file(tmp_path)
    cli_runner.invoke(comfy_cli.workflow_add, ["image", str(f)])
    result = cli_runner.invoke(comfy_cli.workflow_use, ["image", "my_flux.json"])
    assert result.exit_code == 0, result.output
    assert "now uses" in result.output
    assert mw.selected_name("image") == "my_flux.json"


def test_use_clear_falls_back_to_the_built_in_default(cli_runner, tmp_path):
    from localm import media_workflows as mw

    f = _write_workflow_file(tmp_path)
    cli_runner.invoke(comfy_cli.workflow_add, ["image", str(f), "--use"])
    assert mw.selected_name("image") == "my_flux.json"

    result = cli_runner.invoke(comfy_cli.workflow_use, ["image", "--clear"])
    assert result.exit_code == 0, result.output
    assert "cleared" in result.output.lower()
    assert mw.selected_name("image") is None


def test_use_an_unknown_name_is_a_clean_error(cli_runner):
    result = cli_runner.invoke(comfy_cli.workflow_use, ["image", "ghost.json"])
    assert result.exit_code == 1, result.output
    assert "no such workflow" in result.output.lower()


def test_use_requires_either_name_or_clear(cli_runner):
    result = cli_runner.invoke(comfy_cli.workflow_use, ["image"])
    assert result.exit_code == 2, result.output


def test_use_rejects_name_and_clear_together(cli_runner, tmp_path):
    f = _write_workflow_file(tmp_path)
    cli_runner.invoke(comfy_cli.workflow_add, ["image", str(f)])
    result = cli_runner.invoke(
        comfy_cli.workflow_use, ["image", "my_flux.json", "--clear"])
    assert result.exit_code == 2, result.output


# --------------------------------------------------------------------------
#  rm (delete)
# --------------------------------------------------------------------------

def test_rm_with_yes_flag_deletes_without_prompting(cli_runner, tmp_path):
    from localm import media_workflows as mw

    f = _write_workflow_file(tmp_path)
    cli_runner.invoke(comfy_cli.workflow_add, ["image", str(f)])
    result = cli_runner.invoke(
        comfy_cli.workflow_rm, ["image", "my_flux.json", "-y"])
    assert result.exit_code == 0, result.output
    assert "Deleted" in result.output
    assert not (mw.workflows_dir("image") / "my_flux.json").exists()


def test_rm_explicit_yes_at_the_prompt_deletes(cli_runner, tmp_path):
    from localm import media_workflows as mw

    f = _write_workflow_file(tmp_path)
    cli_runner.invoke(comfy_cli.workflow_add, ["image", str(f)])
    result = cli_runner.invoke(
        comfy_cli.workflow_rm, ["image", "my_flux.json"], input="y\n")
    assert result.exit_code == 0, result.output
    assert not (mw.workflows_dir("image") / "my_flux.json").exists()


def test_rm_explicit_no_at_the_prompt_aborts_without_deleting(cli_runner, tmp_path):
    from localm import media_workflows as mw

    f = _write_workflow_file(tmp_path)
    cli_runner.invoke(comfy_cli.workflow_add, ["image", str(f)])
    result = cli_runner.invoke(
        comfy_cli.workflow_rm, ["image", "my_flux.json"], input="n\n")
    assert result.exit_code != 0, result.output
    assert (mw.workflows_dir("image") / "my_flux.json").is_file()


def test_rm_with_no_stdin_aborts_without_deleting(cli_runner, tmp_path):
    """No non-interactive branch: `rm` refuses to delete without an explicit
    yes, matching `rag rm` / `comfy remove`, which also use
    click.confirm(..., abort=True) with no override."""
    from localm import media_workflows as mw

    f = _write_workflow_file(tmp_path)
    cli_runner.invoke(comfy_cli.workflow_add, ["image", str(f)])
    result = cli_runner.invoke(
        comfy_cli.workflow_rm, ["image", "my_flux.json"], input="")
    assert result.exit_code != 0, result.output
    assert (mw.workflows_dir("image") / "my_flux.json").is_file()


def test_rm_refuses_to_delete_the_active_workflow(cli_runner, tmp_path):
    from localm import media_workflows as mw

    f = _write_workflow_file(tmp_path)
    cli_runner.invoke(comfy_cli.workflow_add, ["image", str(f), "--use"])
    result = cli_runner.invoke(
        comfy_cli.workflow_rm, ["image", "my_flux.json", "-y"])
    assert result.exit_code == 1, result.output
    assert "active" in result.output.lower()
    assert (mw.workflows_dir("image") / "my_flux.json").is_file()


def test_rm_an_unknown_name_is_a_clean_error(cli_runner):
    result = cli_runner.invoke(comfy_cli.workflow_rm, ["image", "ghost.json", "-y"])
    assert result.exit_code == 1, result.output
    assert "no such workflow" in result.output.lower()


# --------------------------------------------------------------------------
#  media types stay independent (image/music/video do not share uploads)
# --------------------------------------------------------------------------

def test_workflows_are_scoped_per_media(cli_runner, tmp_path):
    from localm import media_workflows as mw

    f = _write_workflow_file(tmp_path)
    cli_runner.invoke(comfy_cli.workflow_add, ["image", str(f), "--use"])
    assert mw.selected_name("music") is None
    assert mw.list_workflows("music") == []
