# SPDX-License-Identifier: AGPL-3.0-or-later
"""A model id, instance id, operation id/label, config value, GPU name, or
exception/HTTP-error string shown by the `models`-family CLI commands
(localm/cli/models.py) must survive verbatim - Rich's ``Console.print()``
parses ``[...]`` in ANY interpolated string as markup, not just inside a
command's own literal ``[style]`` tags, and the same parsing applies to
``Table`` titles and cell values added via ``.add_row()``. Rich renders
these as:

    Console().print('report[draft].txt')       -> prints "report.txt"
    Console().print('notes[bold red].md')       -> prints "notes.md"

The bracketed span is either dropped outright (an unrecognised tag) or
consumed as a (bogus) style directive (a recognised one), in both cases
silently.

Most of these commands need a running localm server, a live instance
registry entry, or a real HuggingFace API response - none of which a unit
test can produce for real. Each test therefore drives the REAL CLI command
via ``CliRunner`` and forces the exact code path by monkeypatching the one
external boundary that path depends on (``requests.post``, ``server_call``,
``discover.list_gpus``, an instance registry lookup, ...) with realistic,
bracketed data.

Two bracket shapes are used throughout, matching the reference file's own
constants and the two distinct Rich failure modes:
``BRACKET_DROP`` (an unrecognised tag, silently dropped) and
``BRACKET_STYLE`` (a real style name, silently consumed as styling).
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from localm.cli import main

BRACKET_DROP = "alpha[draft]beta"
BRACKET_STYLE = "alpha[bold red]beta"


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Widen the console for every test in this module.

    Only 3 of this file's ~22 tests ever called this inline; the rest
    (TestBenchmarkMarkupEscaping, TestSearchFilesMarkupEscaping, and most
    others) had no width protection at all."""
    from tests.conftest import make_console_wide_and_plain
    make_console_wide_and_plain(monkeypatch, width="300")


# ------------------------------------------------------------------ #
#  benchmark                                                          #
# ------------------------------------------------------------------ #

class TestBenchmarkMarkupEscaping:
    def test_model_not_found_shows_bracketed_name_verbatim(self, cli_runner):
        res = CliRunner().invoke(main, ["benchmark", BRACKET_DROP])
        assert res.exit_code == 1, res.output
        assert BRACKET_DROP in res.output, (
            f"the model name must survive verbatim in the 'Model not found' "
            f"message: {res.output!r}")

    def test_model_not_found_shows_bracket_style_name_verbatim(self, cli_runner):
        res = CliRunner().invoke(main, ["benchmark", BRACKET_STYLE])
        assert res.exit_code == 1, res.output
        assert BRACKET_STYLE in res.output, (
            f"a model name segment that looks like a style tag must be shown "
            f"as literal text: {res.output!r}")

    def test_invalid_prompts_shows_bracketed_value_verbatim(self, cli_runner, monkeypatch):
        # get_model_info must succeed first (--prompts is only parsed after),
        # so force it rather than needing a real registered/loadable model.
        import localm.cli.models as modelscli
        monkeypatch.setattr(modelscli, "get_model_info",
                            lambda *a, **k: ("/some/model.gguf", None))
        bad_prompts = "64,not-a-number[bold red],2048"
        res = CliRunner().invoke(
            main, ["benchmark", "some-model", "--prompts", bad_prompts])
        assert res.exit_code == 1, res.output
        assert bad_prompts in res.output, (
            f"the invalid --prompts value must survive verbatim: {res.output!r}")


# ------------------------------------------------------------------ #
#  search                                                              #
# ------------------------------------------------------------------ #

class TestSearchFilesMarkupEscaping:
    def test_bracket_drop_filename_and_quant_survive_verbatim(self, cli_runner):
        files = [{"file": f"model-{BRACKET_DROP}.gguf", "quant": f"Q4{BRACKET_DROP}",
                 "size_bytes": 2_000_000_000, "n_parts": 1}]
        from unittest.mock import patch
        with patch("localm.discover.hf_gguf_files", return_value=files), \
             patch("localm.discover.vram_info", return_value={"total": None}):
            res = CliRunner().invoke(main, ["search", "owner/repo", "--files"])
        assert res.exit_code == 0, res.output
        assert f"model-{BRACKET_DROP}.gguf" in res.output, (
            f"a remote repo's own filename must survive verbatim: {res.output!r}")
        assert f"Q4{BRACKET_DROP}" in res.output

    def test_bracket_style_filename_survives_verbatim(self, cli_runner):
        files = [{"file": f"model-{BRACKET_STYLE}.gguf", "quant": "Q4_K_M",
                 "size_bytes": 2_000_000_000, "n_parts": 1}]
        from unittest.mock import patch
        with patch("localm.discover.hf_gguf_files", return_value=files), \
             patch("localm.discover.vram_info", return_value={"total": None}):
            res = CliRunner().invoke(main, ["search", "owner/repo", "--files"])
        assert res.exit_code == 0, res.output
        assert f"model-{BRACKET_STYLE}.gguf" in res.output, (
            f"a filename segment that looks like a style tag must be shown "
            f"as literal text, not consumed as Rich styling: {res.output!r}")


class TestSearchPlainMarkupEscaping:
    def test_repo_id_survives_verbatim(self, cli_runner):
        results = [{"id": f"owner/{BRACKET_DROP}", "downloads": 5, "likes": 1}]
        from unittest.mock import patch
        with patch("localm.discover.hf_search", return_value=results):
            res = CliRunner().invoke(main, ["search", "some query"])
        assert res.exit_code == 0, res.output
        assert f"owner/{BRACKET_DROP}" in res.output, (
            f"a remote HF repo id must survive verbatim: {res.output!r}")

    def test_discover_error_message_survives_verbatim(self, cli_runner):
        from unittest.mock import patch
        from localm.discover import DiscoverError
        msg = f"Not a HuggingFace repo id: {BRACKET_STYLE}"

        def _raise(*a, **k):
            raise DiscoverError(msg)
        with patch("localm.discover.hf_gguf_files", side_effect=_raise):
            res = CliRunner().invoke(main, ["search", "owner/repo", "--files"])
        assert res.exit_code == 1, res.output
        assert msg in res.output, (
            f"a DiscoverError's message must survive verbatim: {res.output!r}")


# ------------------------------------------------------------------ #
#  info / config                                                      #
# ------------------------------------------------------------------ #

class TestInfoMarkupEscaping:
    def test_home_dir_path_with_bracket_survives_verbatim(self, cli_runner, tmp_path,
                                                           monkeypatch):
        # Console() reads COLUMNS dynamically (not just at construction), and
        # without it CliRunner's non-tty default (80) hard-wraps mid-word
        # inside a long path - see test_rag_cli_markup_escaping.py's identical
        # fixture note.
        import localm.cli.models as modelscli
        bracket_home = tmp_path / BRACKET_DROP
        bracket_home.mkdir()
        modelscli_home_before = modelscli.HOME_DIR
        try:
            modelscli.HOME_DIR = bracket_home
            res = CliRunner().invoke(main, ["info"])
        finally:
            modelscli.HOME_DIR = modelscli_home_before
        assert res.exit_code == 0, res.output
        assert str(bracket_home / "models") in res.output, (
            f"a bracketed data-dir path must survive verbatim: {res.output!r}")


class TestConfigCmdMarkupEscaping:
    def test_config_value_with_bracket_survives_verbatim(self, cli_runner):
        value = f"You are a helpful assistant. {BRACKET_STYLE}"
        res = CliRunner().invoke(main, ["config", "chat_system_prompt", value])
        assert res.exit_code == 0, res.output
        assert value in res.output, (
            f"a free-text config VALUE must echo back verbatim: {res.output!r}")


# ------------------------------------------------------------------ #
#  stop                                                                #
# ------------------------------------------------------------------ #

class TestStopCmdMarkupEscaping:
    def test_no_instance_matches_shows_bracketed_id_verbatim(self, cli_runner, monkeypatch):
        from localm import instances
        monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
        monkeypatch.setattr(instances, "list_entries", lambda *a, **k: [])
        res = CliRunner().invoke(main, ["stop", BRACKET_DROP])
        assert res.exit_code == 1, res.output
        assert BRACKET_DROP in res.output, (
            f"the instance id argument must survive verbatim (via repr()): "
            f"{res.output!r}")

    def test_ambiguous_match_shows_bracketed_prefix_and_root_dirs_verbatim(
            self, cli_runner, monkeypatch):
        # The per-displayed-id instance_id column IS truncated to 8 chars (by
        # design, unrelated to markup), so a bracket placed early enough in
        # `wanted` survives both there and in the un-truncated matching
        # prefix - this covers the PREFIX's own escape(repr(...)) call, and
        # root_dir (never truncated) covers the loop's second escape() call.
        from localm import instances
        wanted = f"x{BRACKET_DROP}"
        entries = [
            {"instance_id": f"{wanted}1", "pid": 1, "root_dir": f"/proj/a{BRACKET_DROP}",
             "scheme": "http", "host": "127.0.0.1", "port": 1, "_path": "/x/a.json"},
            {"instance_id": f"{wanted}2", "pid": 2, "root_dir": f"/proj/b{BRACKET_STYLE}",
             "scheme": "http", "host": "127.0.0.1", "port": 2, "_path": "/x/b.json"},
        ]
        monkeypatch.setattr(instances, "reap_stale", lambda *a, **k: [])
        monkeypatch.setattr(instances, "list_entries", lambda *a, **k: entries)
        res = CliRunner().invoke(main, ["stop", wanted])
        assert res.exit_code == 1, res.output
        assert f"'{wanted}'" in res.output, (
            f"the ambiguous-match prefix (shown via repr()) must survive "
            f"verbatim: {res.output!r}")
        assert f"/proj/a{BRACKET_DROP}" in res.output and \
            f"/proj/b{BRACKET_STYLE}" in res.output, (
            f"each candidate's root_dir must survive verbatim: {res.output!r}")


# ------------------------------------------------------------------ #
#  unload                                                              #
# ------------------------------------------------------------------ #

class TestUnloadCmdMarkupEscaping:
    def test_unreachable_server_shows_bracketed_exception_verbatim(
            self, cli_runner, monkeypatch):
        import requests
        monkeypatch.setenv("LOCALM_URL", "http://127.0.0.1:19999")
        msg = f"Connection refused talking to {BRACKET_STYLE}"

        def _raise(*a, **k):
            raise requests.exceptions.ConnectionError(msg)
        monkeypatch.setattr(requests, "post", _raise)
        res = CliRunner().invoke(main, ["unload"])
        assert res.exit_code == 1, res.output
        assert msg in res.output, (
            f"the requests exception text must survive verbatim: {res.output!r}")


# ------------------------------------------------------------------ #
#  cancel                                                              #
# ------------------------------------------------------------------ #

def _cancel_server(monkeypatch, ops):
    """Force `cancel_cmd`'s server round trip without a real server: the GET
    /api/activity read returns *ops*, and a POST cancel (if reached)
    succeeds."""
    import localm.cli.models as modelscli

    def _running_server(**kw):
        return "http://127.0.0.1:1", {}

    def _server_call(url, headers, method, path, timeout=None, **kw):
        if method == "GET" and path == "/api/activity":
            return "ok", {"operations": ops}
        if method == "POST" and path.startswith("/api/jobs/"):
            return "ok", {"status": "cancelling"}
        raise AssertionError(f"unexpected server_call {method} {path}")

    monkeypatch.setattr(modelscli, "running_server", _running_server)
    monkeypatch.setattr(modelscli, "server_call", _server_call)


class TestCancelCmdMarkupEscaping:
    def test_no_operation_matches_shows_bracketed_id_verbatim(
            self, cli_runner, monkeypatch):
        _cancel_server(monkeypatch, [])
        res = CliRunner().invoke(main, ["cancel", BRACKET_STYLE])
        assert res.exit_code == 1, res.output
        assert BRACKET_STYLE in res.output, (
            f"the operation id argument (shown via repr()) must survive "
            f"verbatim: {res.output!r}")

    def test_not_running_shows_bracketed_id_and_status_verbatim(
            self, cli_runner, monkeypatch):
        ops = [{"id": f"op-{BRACKET_DROP}", "kind": "pull", "label": "p",
                "status": f"done{BRACKET_DROP}", "cancellable": False}]
        _cancel_server(monkeypatch, ops)
        res = CliRunner().invoke(main, ["cancel", f"op-{BRACKET_DROP}"])
        assert res.exit_code == 0, res.output
        assert f"op-{BRACKET_DROP}" in res.output
        assert f"done{BRACKET_DROP}" in res.output, (
            f"the operation's status must survive verbatim: {res.output!r}")

    def test_cancelling_shows_bracketed_label_and_id_verbatim(
            self, cli_runner, monkeypatch):
        ops = [{"id": f"op-{BRACKET_STYLE}", "kind": "pull",
                "label": f"pull owner/{BRACKET_STYLE}", "status": "running",
                "cancellable": True}]
        _cancel_server(monkeypatch, ops)
        res = CliRunner().invoke(main, ["cancel", f"op-{BRACKET_STYLE}"])
        assert res.exit_code == 0, res.output
        assert f"pull owner/{BRACKET_STYLE}" in res.output, (
            f"the operation label must survive verbatim: {res.output!r}")
        assert f"op-{BRACKET_STYLE}" in res.output


# ------------------------------------------------------------------ #
#  gpus                                                                #
# ------------------------------------------------------------------ #

class TestGpusCmdMarkupEscaping:
    def test_gpu_name_with_bracket_survives_verbatim(self, cli_runner, monkeypatch):
        from localm import discover
        name = f"Radeon {BRACKET_STYLE} 6900 XT"
        gpus = [{"index": 0, "name": name, "total": 16 * 1024 ** 3}]
        monkeypatch.setattr(discover, "list_gpus", lambda **k: (gpus, discover.GPU_PROBE_OK))
        res = CliRunner().invoke(main, ["gpus"])
        assert res.exit_code == 0, res.output
        assert name in res.output, (
            f"a driver-reported GPU name must survive verbatim: {res.output!r}")


# ------------------------------------------------------------------ #
#  ps / status                                                        #
# ------------------------------------------------------------------ #

class TestPsCmdMarkupEscaping:
    def test_instance_row_bracketed_fields_survive_verbatim(self, cli_runner, monkeypatch):
        from localm import instances
        root = f"/proj/{BRACKET_DROP}"
        monkeypatch.setattr(instances, "snapshot", lambda *a, **k: [
            {"instance_id": f"{BRACKET_STYLE}-id", "alive": True, "mode": "full",
             "scheme": "http", "host": "127.0.0.1", "port": 8642, "pid": 4321,
             "root_dir": root},
        ])
        res = CliRunner().invoke(main, ["ps"])
        assert res.exit_code == 0, res.output
        assert root in res.output, (
            f"a bracketed instance root_dir must survive verbatim in the "
            f"table: {res.output!r}")


class TestStatusCmdMarkupEscaping:
    def test_surface_and_version_bracketed_fields_survive_verbatim(
            self, cli_runner, monkeypatch):
        from localm import instances
        mode = f"full{BRACKET_DROP}"
        version = f"0.1.0{BRACKET_STYLE}"
        monkeypatch.setattr(instances, "find_attachable", lambda *a, **k: {
            "scheme": "http", "host": "127.0.0.1", "port": 8642, "mode": mode,
            "pid": 4321, "version": version})
        monkeypatch.setattr(
            "localm.cli.models.read_activity",
            lambda *a, **k: ("ok", {"now": 1.0, "operations": []}))
        res = CliRunner().invoke(main, ["status"])
        assert res.exit_code == 0, res.output
        assert mode in res.output, f"'surface' must survive verbatim: {res.output!r}"
        assert version in res.output, f"'version' must survive verbatim: {res.output!r}"

    def test_activity_label_and_id_survive_verbatim(self, cli_runner, monkeypatch):
        """The `_print_activity` operation loop, exercised through `status`."""
        from localm import instances
        label = f"embedding {BRACKET_STYLE} docs"
        op_id = f"job{BRACKET_DROP}"
        monkeypatch.setattr(instances, "find_attachable", lambda *a, **k: {
            "scheme": "http", "host": "127.0.0.1", "port": 8642, "mode": "full",
            "pid": 4321, "version": "0.1.0"})
        monkeypatch.setattr(
            "localm.cli.models.read_activity",
            lambda *a, **k: ("ok", {"now": 1.0, "operations": [
                {"id": op_id, "kind": "reembed", "label": label,
                 "status": "running", "created_at": 0.0, "cancellable": True},
            ]}))
        res = CliRunner().invoke(main, ["status"])
        assert res.exit_code == 0, res.output
        assert label in res.output, f"operation label must survive verbatim: {res.output!r}"
        assert op_id in res.output, f"operation id must survive verbatim: {res.output!r}"


# ------------------------------------------------------------------ #
#  list (sync note)                                                    #
# ------------------------------------------------------------------ #

class TestListCmdMarkupEscaping:
    def test_sync_note_with_bracket_survives_verbatim(self, cli_runner, monkeypatch):
        import localm.cli.models as modelscli
        from localm.model_manager.registry import ModelSyncResult
        note = f"Skipped autoprune: {BRACKET_STYLE} appear missing."
        monkeypatch.setattr(modelscli, "sync_models_dir",
                            lambda *a, **k: ModelSyncResult(note=note))
        monkeypatch.setattr(modelscli, "list_models", lambda *a, **k: None)
        res = CliRunner().invoke(main, ["list"])
        assert res.exit_code == 0, res.output
        assert note in res.output, (
            f"sync_models_dir's note must survive verbatim: {res.output!r}")
