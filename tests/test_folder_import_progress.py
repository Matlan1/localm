# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adding a folder of models says what it is doing: a scan line before the first
registration, per-file hashing progress (bytes) on the GUI job channel, and a
final summary. The GUI job runs the real CLI child with a NUL stdin, so the
progress has to reach the job stream without a terminal."""

from __future__ import annotations

import os
import struct
import sys
import time
from pathlib import Path

import pytest

from localm import model_manager as mm
from localm.model_manager import _shared
from localm.plugins.gui import jobs as gj

_REPO = Path(__file__).resolve().parent.parent


def _gguf(path: Path, payload: bytes = b"") -> Path:
    key, val = b"general.architecture", b"llama"
    body = (struct.pack("<Q", len(key)) + key + struct.pack("<I", 8)
            + struct.pack("<Q", len(val)) + val)
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(head + body + b"\0" * 4096 + payload)
    old = time.time() - 3600
    os.utime(path, (old, old))
    return path


def _folder(root: Path) -> Path:
    for i, name in enumerate(("alpha", "beta", "gamma")):
        _gguf(root / f"{name}.gguf", payload=bytes([i]) * 2048)
    return root


@pytest.fixture
def home(tmp_path, monkeypatch):
    import localm.config as cfg
    h = tmp_path / ".localm"
    (h / "models").mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    monkeypatch.setattr(mm, "MODELS_DIR", h / "models")
    monkeypatch.setattr(mm, "REGISTRY_FILE", h / "registry.json")
    return h


@pytest.fixture
def printed(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(_shared.console, "print",
                        lambda *a, **k: lines.append(" ".join(str(x) for x in a)))
    return lines


def _index(lines, needle):
    for i, line in enumerate(lines):
        if needle in line:
            return i
    raise AssertionError(f"{needle!r} not in {lines}")


class TestTerminalLines:
    def test_scan_and_found_lines_come_before_the_first_registration(
            self, home, tmp_path, printed):
        assert mm.add_local(str(_folder(tmp_path / "pile"))) is True
        assert _index(printed, "Scanning") < _index(printed, "Found 3 GGUF")
        assert _index(printed, "Found 3 GGUF") < _index(printed, "Registered")

    def test_each_file_is_announced_with_its_place_in_the_batch(
            self, home, tmp_path, printed):
        mm.add_local(str(_folder(tmp_path / "pile")))
        for k, name in enumerate(("alpha", "beta", "gamma"), 1):
            assert any(f"Hashing {name}.gguf for duplicate detection ({k} of 3" in line
                       for line in printed), printed

    def test_final_summary_counts_registered_files(self, home, tmp_path, printed):
        mm.add_local(str(_folder(tmp_path / "pile")))
        assert "Done: 3 registered, 0 already registered, 0 skipped or refused, " \
               "0 not usable" in printed[-1]

    def test_unusable_file_is_counted_in_the_summary(self, home, tmp_path, printed):
        folder = _folder(tmp_path / "pile")
        bad = folder / "old.gguf"
        bad.write_bytes(b"GGUF" + struct.pack("<I", 1) + b"\0" * 4096)
        os.utime(bad, (time.time() - 3600,) * 2)
        mm.add_local(str(folder))
        assert "3 registered" in printed[-1] and "1 not usable" in printed[-1]

    def test_a_duplicate_is_skipped_by_name_and_counted(self, home, tmp_path, printed):
        mm.add_local(str(_folder(tmp_path / "one")))
        printed.clear()
        _gguf(tmp_path / "two" / "alpha-copy.gguf", payload=b"\x00" * 2048)
        mm.add_local(str(tmp_path / "two"), on_duplicate="skip")
        assert any("Skipped alpha-copy: byte-identical content, already registered as "
                   in line for line in printed), printed
        assert "0 registered" in printed[-1] and "1 skipped or refused" in printed[-1]


class TestGuiJobChannel:
    """The real job runner around the real CLI child, stdin NUL."""

    def _run(self, tmp_path, folder):
        env = {
            "LOCALM_PROGRESS_JSON": "1",
            "LOCALM_HOME": str(tmp_path / "jobhome"),
            "PYTHONPATH": str(_REPO),
            "PYTHONIOENCODING": "utf-8",
        }
        job = gj.JobManager().start_cli(
            "pull", ["pull", "--", str(folder)], extra_env=env)
        deadline = time.monotonic() + 120
        while job.status == "running" and time.monotonic() < deadline:
            time.sleep(0.1)
        assert job.status == "done", [e for e in job._history if e.get("type") == "line"]
        return job

    def test_phase_hashing_and_summary_reach_the_job_stream(self, tmp_path):
        job = self._run(tmp_path, _folder(tmp_path / "pile"))
        events = list(job._history)
        details = [e["detail"] for e in events
                   if e.get("type") == "progress" and e.get("phase") == "import"]
        assert details[0].startswith("Scanning"), details
        assert any(d.startswith("Found 3 GGUF") for d in details)

        hashing = [e for e in events if e.get("type") == "progress"
                   and e.get("phase") == "import"
                   and e.get("detail", "").startswith("Hashing alpha.gguf")]
        assert hashing, details
        assert hashing[-1]["total"] > 0 and hashing[-1]["pct"] == 100.0

        lines = [e["text"] for e in events if e.get("type") == "line"]
        first_registered = _index(lines, "Registered")
        assert _index(lines, "Scanning") < first_registered
        assert "Done: 3 registered" in lines[-1]
        assert job.summary()["detail"].startswith("Done: 3 registered")
        assert job.summary()["phase"] == "import"


class TestManifestProbeSkipsModelFiles:
    def test_a_file_too_big_to_be_a_manifest_is_never_read(self, tmp_path, monkeypatch):
        from localm.model_manager import registry as reg
        folder = tmp_path / "pile"
        folder.mkdir()
        big = folder / "big.gguf"
        big.write_bytes(b"\0" * ((1 << 20) + 1))
        read: list[str] = []
        real = Path.read_text
        monkeypatch.setattr(
            Path, "read_text",
            lambda self, *a, **k: (read.append(self.name), real(self, *a, **k))[1])
        assert reg._resolve_ollama_manifest(folder) is None
        assert read == []
