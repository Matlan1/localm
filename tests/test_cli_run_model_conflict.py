# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm run X` must NEVER answer with a different model.

When a background server for this directory serves model Y (or none),
`localm run X` switches that server to X and chats with X, or loads X in this
process when the server is not switched. These tests drive the real `run`
command (CliRunner), stubbing only the network seam (attach discovery, the
/v1/models probe, the switch request and the HTTP engine), so the real run()
control flow decides what happens. The switch request itself is exercised
against a loopback server in test_cli_run_switches_server_model.py.
"""

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from localm.audit import SessionMode
from localm.cli.chat import run


@pytest.fixture
def patched(monkeypatch):
    """Stub the attach seam and hand the test control of the probe result, the
    switch result and a spy HttpEngine, so it can assert whether an
    attach/stream to the server ever happened and with which model."""
    fake_target = {"base_url": "http://127.0.0.1:8642/v1", "token": "tok"}
    engine_spy = MagicMock(name="HttpEngine")
    switches = []
    outcome = {"result": None}

    monkeypatch.setattr("localm.instances.attach_target",
                        lambda *a, **k: fake_target)
    monkeypatch.setattr("localm.instances.resolve_root_dir", lambda *a, **k: ".")
    monkeypatch.setattr("localm.inference.http_engine.HttpEngine", engine_spy)
    monkeypatch.setattr("localm.audit.effective_mode",
                        lambda *a, **k: SessionMode.LOG)

    def fake_switch(base_url, token, model):
        switches.append((base_url, model))
        return outcome["result"] or (model, None)

    monkeypatch.setattr("localm.cli.chat._switch_server_model", fake_switch)

    def set_status(state, active):
        monkeypatch.setattr("localm.inference.http_engine.remote_model_status",
                            lambda *a, **k: (state, active))

    return engine_spy, set_status, switches, outcome


@pytest.fixture
def in_process(monkeypatch):
    """Record that run() reached its in-process load, stopping it there (the
    model is reported missing) so no real model is loaded, and fail if a
    background server would be started."""
    lookups = []

    def lookup(name, *a, **k):
        lookups.append(name)
        return None

    monkeypatch.setattr("localm.cli.chat.get_operator_model_info", lookup)
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("server started")))
    return lookups


def _flat(output: str) -> str:
    """Collapse rich line-wrapping so phrase assertions are wrap-proof."""
    return " ".join(output.split()).lower()


def test_run_switches_a_server_serving_a_different_model(patched):
    engine_spy, set_status, switches, _ = patched
    set_status("loaded", "qwen2.5-0.5b")   # server serves Y

    result = CliRunner().invoke(
        run, ["SmolLM2-135M", "-p", "What color is the sky?"])

    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert switches == [("http://127.0.0.1:8642/v1", "SmolLM2-135M")]
    assert engine_spy.call_args.kwargs["model"] == "SmolLM2-135M"
    assert engine_spy.call_args.kwargs["display_name"] == "SmolLM2-135M"
    assert "serves qwen2.5-0.5b; switching it to smollm2-135m" in flat
    assert "not the requested" not in flat
    assert "--no-server" not in flat


def test_run_attaches_when_server_serves_the_requested_model(patched):
    engine_spy, set_status, switches, _ = patched
    set_status("loaded", "SmolLM2-135M")   # server already serves X

    result = CliRunner().invoke(run, ["SmolLM2-135M", "-p", "hi"])

    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert switches == [], "the model it already serves is not switched"
    assert engine_spy.call_args.kwargs["model"] == "SmolLM2-135M"
    assert "switching" not in flat
    assert "--no-server" not in flat


def test_run_switches_a_server_with_no_model_loaded(patched):
    engine_spy, set_status, switches, _ = patched
    set_status("empty", None)   # server up, nothing loaded

    result = CliRunner().invoke(run, ["SmolLM2-135M", "-p", "hi"])

    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert switches == [("http://127.0.0.1:8642/v1", "SmolLM2-135M")]
    assert engine_spy.call_args.kwargs["model"] == "SmolLM2-135M"
    assert "has no model loaded; switching it to smollm2-135m" in flat
    assert "--no-server" not in flat


def test_run_addresses_the_model_by_the_name_the_switch_reports(patched):
    engine_spy, set_status, switches, outcome = patched
    set_status("loaded", "qwen2.5-0.5b")
    outcome["result"] = ("SmolLM2-135M-Instruct", None)

    result = CliRunner().invoke(run, ["SmolLM2-135M", "-p", "hi"])

    assert result.exit_code == 0, result.output
    assert engine_spy.call_args.kwargs["model"] == "SmolLM2-135M-Instruct"


def test_run_loads_in_process_when_the_switch_fails(patched, in_process):
    engine_spy, set_status, switches, outcome = patched
    set_status("loaded", "qwen2.5-0.5b")
    outcome["result"] = (None, "HTTP 503: not enough VRAM")

    result = CliRunner().invoke(run, ["SmolLM2-135M", "-p", "hi"])

    flat = _flat(result.output)
    assert switches == [("http://127.0.0.1:8642/v1", "SmolLM2-135M")]
    assert not engine_spy.called, "a server that did not switch is not chatted with"
    assert in_process == ["SmolLM2-135M"], "the model loads in this process"
    assert ("could not switch to smollm2-135m (http 503: not enough vram); "
            "loading it in this process") in flat
    assert "starting one in the background" not in flat
    assert "--no-server" not in flat


def test_run_with_load_options_loads_in_process_instead_of_switching(
        patched, in_process):
    engine_spy, set_status, switches, _ = patched
    set_status("loaded", "qwen2.5-0.5b")

    result = CliRunner().invoke(
        run, ["SmolLM2-135M", "-p", "hi", "--ctx", "4096", "-g", "0"])

    flat = _flat(result.output)
    assert switches == [], "the server would not apply --ctx/--gpu-layers"
    assert not engine_spy.called
    assert in_process == ["SmolLM2-135M"]
    assert "serves qwen2.5-0.5b" in flat
    assert "--ctx, --gpu-layers" in flat
    assert "loaded in this process" in flat


def test_run_attaches_quietly_when_model_is_unknown(patched):
    # A chat-scoped attach token cannot read /v1/models -> ("unknown", None),
    # which attaches quietly with the requested model.
    engine_spy, set_status, switches, _ = patched
    set_status("unknown", None)

    result = CliRunner().invoke(run, ["SmolLM2-135M", "-p", "hi"])

    assert result.exit_code == 0, result.output
    assert switches == []
    assert engine_spy.call_args.kwargs["model"] == "SmolLM2-135M"


def test_run_no_server_does_not_consult_the_server(patched, monkeypatch):
    # --no-server opts out of attaching entirely, so a running server serving Y
    # is irrelevant: it is neither probed nor switched.
    engine_spy, set_status, switches, _ = patched
    set_status("loaded", "qwen2.5-0.5b")
    # attach_target is not consulted under --no-server; this stub raises if it
    # is, so the opt-out has to short-circuit before the probe.
    monkeypatch.setattr(
        "localm.instances.attach_target",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("consulted under --no-server")))

    result = CliRunner().invoke(
        run, ["SmolLM2-135M", "-p", "hi", "--no-server"])

    flat = _flat(result.output)
    assert switches == []
    assert "qwen2.5-0.5b" not in flat, "must not mention the server's model"
    assert "the localm server here" not in flat
