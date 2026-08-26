# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm gui` must not silently discard explicit server-config flags by attaching.

These tests drive the real `gui` command (CliRunner), stubbing only the
instance-discovery seam, and assert that a conflicting explicit flag is refused
(with the --new way out) while a compatible invocation still attaches.
"""

import os

import pytest
from click.testing import CliRunner

from localm.plugins.gui import cli as guicli


@pytest.fixture(autouse=True)
def _clean_mode_env():
    # `gui --mode M` sets LOCALM_MODE in-process; keep it from leaking between tests.
    os.environ.pop("LOCALM_MODE", None)
    yield
    os.environ.pop("LOCALM_MODE", None)


@pytest.fixture
def running(monkeypatch):
    """Pretend a full-mode localm is already serving this directory on port 8793.
    Returns a setter for the running instance's active model (for the probe)."""
    entry = {
        "pid": 26164, "port": 8793, "host": "127.0.0.1", "scheme": "http",
        "mode": "full", "token": "tok", "root_dir": "/proj",
    }
    monkeypatch.setattr("localm.winconsole.disable_quickedit", lambda: None)
    monkeypatch.setattr("localm.instances.resolve_root_dir", lambda *a, **k: "/proj")
    monkeypatch.setattr("localm.instances.find_attachable", lambda *a, **k: entry)

    def set_active_model(model_id):
        monkeypatch.setattr("localm.inference.http_engine.remote_model_status",
                            lambda *a, **k: ("loaded", model_id))

    # Default: model probe returns "unknown" (no model named in most tests anyway).
    set_active_model(None)
    monkeypatch.setattr("localm.inference.http_engine.remote_model_status",
                        lambda *a, **k: ("unknown", None))
    return set_active_model


def _flat(output: str) -> str:
    return " ".join(output.split()).lower()


def test_gui_different_port_is_refused_not_silently_dropped(running):
    result = CliRunner().invoke(guicli.main, ["--port", "8794", "--no-browser"])
    flat = _flat(result.output)
    assert result.exit_code == 1, result.output
    assert "--port 8794" in flat
    assert "8793" in flat                 # names the running server's port
    assert "--new" in flat                # offers the way out
    assert "attaching" not in flat        # did NOT silently attach


def test_gui_explicit_mode_is_refused(running):
    result = CliRunner().invoke(guicli.main, ["--mode", "full", "--no-browser"])
    flat = _flat(result.output)
    assert result.exit_code == 1, result.output
    assert "--mode" in flat
    assert "--new" in flat
    assert "attaching" not in flat


def test_gui_same_port_still_attaches(running):
    # Re-passing the port the server is already on is harmless: attach, do not refuse.
    result = CliRunner().invoke(guicli.main, ["--port", "8793", "--no-browser"])
    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert "attaching" in flat
    assert "cannot apply" not in flat


def test_gui_no_conflicting_flags_still_attaches(running):
    result = CliRunner().invoke(guicli.main, ["--no-browser"])
    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert "attaching" in flat


def test_gui_no_model_flag_is_not_a_conflict(running):
    # --no-model only picks a STARTUP model; on an attach nothing starts, so it
    # is moot rather than a conflict: attach, do not refuse.
    result = CliRunner().invoke(guicli.main, ["--no-model", "--no-browser"])
    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert "attaching" in flat
    assert "cannot apply" not in flat


def test_gui_new_flag_bypasses_the_conflict_and_starts_fresh(running, monkeypatch):
    # --new skips the attach path entirely and starts a separate server with the
    # requested settings. The serve seam is stubbed so no real server runs.
    ran = {}
    monkeypatch.setattr("localm.portmux.run_server",
                        lambda *a, **k: ran.setdefault("ok", True))
    result = CliRunner().invoke(
        guicli.main, ["--new", "--no-model", "--port", "8794", "--no-browser"])
    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert "cannot apply" not in flat     # not refused
    assert "attaching" not in flat        # not attached
    assert ran.get("ok"), "should start a fresh server"


def test_gui_different_model_is_refused(running):
    running("qwen2.5-0.5b")   # the running server serves Y
    result = CliRunner().invoke(guicli.main, ["SmolLM2-135M", "--no-browser"])
    flat = _flat(result.output)
    assert result.exit_code == 1, result.output
    assert "smollm2-135m" in flat
    assert "qwen2.5-0.5b" in flat
    assert "--new" in flat
    assert "attaching" not in flat


def test_gui_same_model_still_attaches(running):
    running("SmolLM2-135M")   # the running server already serves X
    result = CliRunner().invoke(guicli.main, ["SmolLM2-135M", "--no-browser"])
    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert "attaching" in flat
    assert "cannot apply" not in flat
