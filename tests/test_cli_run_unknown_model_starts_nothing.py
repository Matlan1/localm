# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm run <unknown name>` says "Model not found" without starting a
background server first."""

import subprocess

from localm import instances
from localm.cli import main


def test_an_unknown_model_starts_no_server(cli_runner, monkeypatch):
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: started.append(a) or None)
    monkeypatch.setattr(instances, "attach_target", lambda *a, **k: None)

    res = cli_runner.invoke(main, ["run", "no-such-model-anywhere", "-p", "hi"])

    assert res.exit_code == 1, res.output
    assert "Model not found" in res.output
    assert "Starting one in the background" not in res.output
    assert started == []
