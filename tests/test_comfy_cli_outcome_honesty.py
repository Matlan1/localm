# SPDX-License-Identifier: AGPL-3.0-or-later
"""Board item #27: cli/comfy.py's comfy_setup / comfy_update are the two call sites that go through jobs.py's start_cli with no post-success-print guard at all - pull.py already has #1111's _report_success, and remove has no risky post-success print to begin with (see dev-notes/ROOTCAUSE-pull-success-r..."""

from __future__ import annotations

import json

from localm.cli import comfy as comfy_cli
from localm.media import managed_comfy as mc_mod
from localm.media import managed_comfy_fresh as fresh_mod
from localm.media import managed_comfy_provision as prov_mod
from localm.media import managed_comfy_update as upd_mod
from localm.media.managed_comfy_provision import ProvisionResult
from localm.model_manager._shared import PROGRESS_SENTINEL


def _ok_result(message="all good", status="fresh"):
    return ProvisionResult(ok=True, status=status, message=message,
                           installed_packages=3, custom_nodes_copied=1)


def _fail_result(message="it broke"):
    return ProvisionResult(ok=False, status="error", message=message)


def _outcome_frames(output: str) -> list:
    """Every {'type':'outcome',...} sentinel frame in *output*, in order."""
    frames = []
    for line in output.splitlines():
        if PROGRESS_SENTINEL in line:
            _, _, payload = line.partition(PROGRESS_SENTINEL)
            data = json.loads(payload)
            if data.get("type") == "outcome":
                frames.append(data)
    return frames


class TestComfySetupEmitsAnOutcomeFrame:
    def test_no_frame_outside_gui_mode(self, cli_runner, monkeypatch):
        """Interactive terminal use (no LOCALM_PROGRESS_JSON) must never print raw sentinel JSON - matches _emit_progress's own gating."""
        monkeypatch.delenv("LOCALM_PROGRESS_JSON", raising=False)
        monkeypatch.setattr(prov_mod, "discover_user_comfy", lambda cfg: None)
        monkeypatch.setattr(fresh_mod, "setup_managed_comfy",
                            lambda *a, **k: _ok_result())
        result = cli_runner.invoke(comfy_cli.comfy_setup, ["--no-custom-nodes"])
        assert result.exit_code == 0, result.output
        assert PROGRESS_SENTINEL not in result.output

    def test_done_frame_on_success(self, cli_runner, monkeypatch):
        monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")
        monkeypatch.setattr(prov_mod, "discover_user_comfy", lambda cfg: None)
        monkeypatch.setattr(fresh_mod, "setup_managed_comfy",
                            lambda *a, **k: _ok_result())
        result = cli_runner.invoke(comfy_cli.comfy_setup, ["--no-custom-nodes"])
        assert result.exit_code == 0, result.output
        assert _outcome_frames(result.output) == [{"type": "outcome", "status": "done"}]

    def test_done_frame_reaches_stdout_even_though_the_success_print_then_crashes(
            self, cli_runner, monkeypatch):
        """The exact scenario the whole mechanism exists for: real work is done (setup_managed_comfy returned ok=True), the frame is written, and ONLY THEN does the trailing status print raise."""
        monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")
        monkeypatch.setattr(prov_mod, "discover_user_comfy", lambda cfg: None)
        monkeypatch.setattr(fresh_mod, "setup_managed_comfy",
                            lambda *a, **k: _ok_result())

        def _raise_on_success_line(msg, **kw):
            if isinstance(msg, str) and msg.startswith("[green]"):
                raise ModuleNotFoundError("simulated rich crash")
        monkeypatch.setattr(comfy_cli.console, "print", _raise_on_success_line)

        result = cli_runner.invoke(comfy_cli.comfy_setup, ["--no-custom-nodes"])

        assert result.exit_code != 0, (
            "the crash is real and still propagates at the CLI level")
        assert _outcome_frames(result.output) == [{"type": "outcome", "status": "done"}], (
            "the done frame must already be on stdout before the crash")

    def test_failed_frame_on_failure(self, cli_runner, monkeypatch):
        monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")
        monkeypatch.setattr(prov_mod, "discover_user_comfy", lambda cfg: None)
        monkeypatch.setattr(fresh_mod, "setup_managed_comfy",
                            lambda *a, **k: _fail_result())
        result = cli_runner.invoke(comfy_cli.comfy_setup, ["--no-custom-nodes"])
        assert result.exit_code != 0
        assert _outcome_frames(result.output) == [{"type": "outcome", "status": "failed"}]


class TestComfyUpdateEmitsAnOutcomeFrame:
    def test_done_frame_on_success(self, cli_runner, monkeypatch):
        monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")
        monkeypatch.setattr(mc_mod, "is_managed_comfy_installed", lambda: True)
        monkeypatch.setattr(upd_mod, "update_managed_comfy",
                            lambda *a, **k: _ok_result(status="updated"))
        result = cli_runner.invoke(comfy_cli.comfy_update, [])
        assert result.exit_code == 0, result.output
        assert _outcome_frames(result.output) == [{"type": "outcome", "status": "done"}]

    def test_failed_frame_on_failure(self, cli_runner, monkeypatch):
        monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")
        monkeypatch.setattr(mc_mod, "is_managed_comfy_installed", lambda: True)
        monkeypatch.setattr(upd_mod, "update_managed_comfy",
                            lambda *a, **k: _fail_result())
        result = cli_runner.invoke(comfy_cli.comfy_update, [])
        assert result.exit_code != 0
        assert _outcome_frames(result.output) == [{"type": "outcome", "status": "failed"}]


class TestMessageMarkupIsEscaped:
    def test_setup_success_message_is_escaped(self, cli_runner, monkeypatch):
        monkeypatch.setattr(prov_mod, "discover_user_comfy", lambda cfg: None)
        monkeypatch.setattr(
            fresh_mod, "setup_managed_comfy",
            lambda *a, **k: _ok_result(message="ready: handler(x: List[int]) ok"))
        calls = []
        monkeypatch.setattr(comfy_cli.console, "print", calls.append)

        result = cli_runner.invoke(comfy_cli.comfy_setup, ["--no-custom-nodes"])

        assert result.exit_code == 0, result.output
        success_lines = [c for c in calls if isinstance(c, str) and "[green]" in c]
        assert success_lines, f"no success line printed: {calls}"
        # escape() turns a literal '[' into '\[', which rich renders back to a
        # literal '[' - an UNESCAPED '[' surviving here means the raw message
        # was interpolated straight into the markup string, exactly the bug.
        assert "\\[int]" in success_lines[0], (
            f"result.message must be escaped before interpolation: "
            f"{success_lines[0]!r}")

    def test_setup_failure_message_is_escaped(self, cli_runner, monkeypatch):
        monkeypatch.setattr(prov_mod, "discover_user_comfy", lambda cfg: None)
        monkeypatch.setattr(
            fresh_mod, "setup_managed_comfy",
            lambda *a, **k: _fail_result(message="pip failed: Optional[str] mismatch"))
        calls = []
        monkeypatch.setattr(comfy_cli.console, "print", calls.append)

        result = cli_runner.invoke(comfy_cli.comfy_setup, ["--no-custom-nodes"])

        assert result.exit_code != 0
        fail_lines = [c for c in calls if isinstance(c, str) and "[red]" in c]
        assert fail_lines, f"no failure line printed: {calls}"
        assert "\\[str]" in fail_lines[0]

    def test_update_success_message_is_escaped(self, cli_runner, monkeypatch):
        monkeypatch.setattr(mc_mod, "is_managed_comfy_installed", lambda: True)
        monkeypatch.setattr(
            upd_mod, "update_managed_comfy",
            lambda *a, **k: _ok_result(message="updated: Dict[str, int] fine",
                                       status="updated"))
        calls = []
        monkeypatch.setattr(comfy_cli.console, "print", calls.append)

        result = cli_runner.invoke(comfy_cli.comfy_update, [])

        assert result.exit_code == 0, result.output
        success_lines = [c for c in calls if isinstance(c, str) and "[green]" in c]
        assert success_lines and "\\[str, int]" in success_lines[0]

    def test_an_unmatched_close_tag_does_not_crash_the_command(self, cli_runner, monkeypatch):
        """The sharper case, EMPIRICALLY VERIFIED against this exact rich version before writing this test (an ordinary stray '[' with no close-tag shape, e.g. a bare Windows path fragment, does NOT raise - it is merely silently swallowed, which the escaped-content assertions above already cover)."""
        monkeypatch.setattr(prov_mod, "discover_user_comfy", lambda cfg: None)
        monkeypatch.setattr(
            fresh_mod, "setup_managed_comfy",
            lambda *a, **k: _ok_result(message="done, wrote [/b]evil"))

        result = cli_runner.invoke(comfy_cli.comfy_setup, ["--no-custom-nodes"])

        assert result.exit_code == 0, (
            f"an unmatched close-tag-shaped fragment in the message must not "
            f"crash the command: {result.exception}\n{result.output}")
