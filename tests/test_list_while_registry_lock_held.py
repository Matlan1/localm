# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm list` is a read: while another localm process holds the registry's
cross-process lock it lists without syncing the models folder, says so in one
line, and does not wait out the lock budget or report an unexpected error."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_HOLDER = """
import sys, time
from pathlib import Path
import localm.config as c
c.REGISTRY_FILE = Path(sys.argv[1])
with c._cross_process_lock(c.REGISTRY_FILE):
    Path(sys.argv[2]).write_text("held", encoding="utf-8")
    time.sleep(float(sys.argv[3]))
"""


@pytest.fixture
def wired(cli_runner, monkeypatch):
    import localm.cli.models as models_cli
    calls = {"sync": 0, "list": 0}

    class _Result:
        changed = False
        note = ""

    def fake_sync(*a, **k):
        calls["sync"] += 1
        return _Result()

    def fake_list(*a, **k):
        calls["list"] += 1

    monkeypatch.setattr(models_cli, "sync_models_dir", fake_sync)
    monkeypatch.setattr(models_cli, "list_models", fake_list)
    return cli_runner, calls


def _hold_registry_lock(tmp_path, seconds):
    import localm.config as cfg
    ready = tmp_path / "ready.txt"
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(cfg.REGISTRY_FILE), str(ready), str(seconds)],
        cwd=str(REPO_ROOT), env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)})
    deadline = time.time() + 30
    while not ready.exists():
        assert proc.poll() is None, "lock holder exited early"
        assert time.time() < deadline, "lock holder never took the lock"
        time.sleep(0.05)
    return proc


def test_list_without_sync_when_another_process_holds_the_registry_lock(wired, tmp_path):
    from localm.cli.models import list_cmd
    runner, calls = wired
    proc = _hold_registry_lock(tmp_path, 60)
    try:
        started = time.time()
        result = runner.invoke(list_cmd, [])
        elapsed = time.time() - started
    finally:
        proc.kill()
        proc.wait()

    assert result.exit_code == 0, result.output
    assert "listing without syncing" in " ".join(result.output.split())
    assert "unexpected error" not in result.output
    assert calls["list"] == 1
    assert calls["sync"] == 0
    assert elapsed < 5


def test_list_still_syncs_when_the_registry_lock_is_free(wired):
    from localm.cli.models import list_cmd
    runner, calls = wired

    result = runner.invoke(list_cmd, [])

    assert result.exit_code == 0, result.output
    assert "listing without syncing" not in " ".join(result.output.split())
    assert calls["sync"] == 1
    assert calls["list"] == 1


def test_list_falls_back_when_the_lock_is_taken_after_the_probe(wired, monkeypatch):
    import localm.cli.models as models_cli
    from localm.config import ConfigLockTimeout
    runner, calls = wired

    def busy_sync(*a, **k):
        calls["sync"] += 1
        raise ConfigLockTimeout("timed out after 10s waiting for registry.json.lock")

    monkeypatch.setattr(models_cli, "sync_models_dir", busy_sync)

    result = runner.invoke(models_cli.list_cmd, [])

    assert result.exit_code == 0, result.output
    assert "listing without syncing" in " ".join(result.output.split())
    assert calls["sync"] == 1
    assert calls["list"] == 1
