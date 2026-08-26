# SPDX-License-Identifier: AGPL-3.0-or-later
"""cli/comfy.py's comfy_setup / comfy_update are the two call sites that go
through jobs.py's start_cli with no post-success-print guard of their own -
pull.py already has _report_success, and remove has no risky post-success print
to begin with.

Proves the PRODUCER side: the CLI emits the {"type":"outcome"} sentinel frame at
the right moment and gated correctly. The consumer side - the part of jobs.py
that actually decides job.status - is tested separately against a fake
subprocess.

Also proves the independent markup-injection fix: result.message is partly built
from raw subprocess tail() output (managed_comfy_fresh.py /
managed_comfy_update.py's f"...{_tail(out)}"), so it can contain '[' / ']' -
e.g. a Python traceback embedding a type hint like List[int], or a Windows path.
Unescaped, that either raises rich.errors.MarkupError (turning a real failure
message into a WORSE, less informative crash) or gets silently parsed as a style
tag and vanishes.
"""

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
    """Every {"type":"outcome",...} sentinel frame in *output*, in order."""
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
        """Interactive terminal use (no LOCALM_PROGRESS_JSON) must never
        print raw sentinel JSON - matches _emit_progress's own gating."""
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
        """The scenario the whole mechanism exists for: real work is done
        (setup_managed_comfy returned ok=True), the frame is written, and
        ONLY THEN does the trailing status print raise. Proves the frame is
        not lost to the crash. The CLI's own exit code is still allowed to go
        non-zero here; what the frame buys is that jobs.py can see the truth
        despite that, not that this print becomes crash-proof - see the
        comfy_setup docstring."""
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
        # escape() turns a literal '[' into an escaped one, which rich renders
        # back to a literal '['. An UNESCAPED '[' surviving here means the raw
        # message was interpolated straight into the markup string.
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
        """The sharper case: an ordinary stray '[' with no close-tag shape
        (e.g. a bare Windows path fragment) does NOT raise - it is merely
        silently swallowed, which the escaped-content assertions above
        already cover. A bracket pair shaped like a CLOSE tag with no
        matching open ('[/b]', matching _warn_if_repo_ships_code's own
        documented crash in pull.py) is a genuine rich markup SYNTAX error:

            >>> Console().print("[green]done [/b]evil[/green]")
            rich.errors.MarkupError: closing tag '[/b]' at position 12
            doesn't match any open tag

        A subprocess tail() of a Python traceback easily produces this shape
        (e.g. a raised exception whose message itself contains bracketed
        text). Confirms the fix prevents a REAL crash, not merely that some
        escaping call was made against a pattern that was never going to
        raise regardless."""
        monkeypatch.setattr(prov_mod, "discover_user_comfy", lambda cfg: None)
        monkeypatch.setattr(
            fresh_mod, "setup_managed_comfy",
            lambda *a, **k: _ok_result(message="done, wrote [/b]evil"))

        result = cli_runner.invoke(comfy_cli.comfy_setup, ["--no-custom-nodes"])

        assert result.exit_code == 0, (
            f"an unmatched close-tag-shaped fragment in the message must not "
            f"crash the command: {result.exception}\n{result.output}")
