# SPDX-License-Identifier: AGPL-3.0-or-later
"""Download progress and speed come from the bytes received, measured against
the known total, on every surface that shows a download: the GUI job runner's
frames, a terminal bar, and the MCP server's progress notifications. The
partial data of a stopped pull can be deleted, and Ctrl-C leaves it resumable.
"""

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from localm import model_manager as mm
from localm.model_manager import pull as pull_mod


class FakeMeter:
    """Stands in for the process-wide byte meter with fixed readings."""

    def __init__(self, transferred=0, rate=None):
        self.transferred = transferred
        self._rate = rate

    def reset(self):
        pass

    def rate(self):
        return self._rate


def _events(capsys):
    out = capsys.readouterr().out
    return [json.loads(line.split(mm.PROGRESS_SENTINEL, 1)[1])
            for line in out.splitlines() if mm.PROGRESS_SENTINEL in line]


class TestByteTracker:
    def test_the_figure_keeps_moving_while_the_file_on_disk_is_flat(self, monkeypatch):
        """A Xet download writes its file in bursts: bytes arrive while the file
        stays the same size."""
        monkeypatch.setattr(pull_mod, "METER", FakeMeter(transferred=300_000, rate=50_000.0))
        tracker = pull_mod._ByteTracker(total=1_000_000, base=0)
        downloaded, rate, eta = tracker.sample(on_disk=50_000)
        assert downloaded == 300_000
        assert rate == 50_000.0
        assert eta == 14

    def test_a_resumed_download_counts_the_bytes_it_started_with(self, monkeypatch):
        monkeypatch.setattr(pull_mod, "METER", FakeMeter(transferred=100, rate=10.0))
        tracker = pull_mod._ByteTracker(total=1000, base=400)
        assert tracker.sample(on_disk=400)[0] == 500

    def test_the_file_on_disk_wins_when_it_is_ahead(self, monkeypatch):
        monkeypatch.setattr(pull_mod, "METER", FakeMeter(transferred=10, rate=10.0))
        tracker = pull_mod._ByteTracker(total=1000, base=0)
        assert tracker.sample(on_disk=600)[0] == 600

    def test_the_figure_never_passes_the_total(self, monkeypatch):
        monkeypatch.setattr(pull_mod, "METER", FakeMeter(transferred=5000, rate=10.0))
        tracker = pull_mod._ByteTracker(total=1000, base=0)
        assert tracker.sample(on_disk=0)[0] == 1000

    def test_an_unknown_total_gives_no_eta(self, monkeypatch):
        monkeypatch.setattr(pull_mod, "METER", FakeMeter(transferred=500, rate=10.0))
        tracker = pull_mod._ByteTracker(total=0, base=0)
        downloaded, rate, eta = tracker.sample(on_disk=0)
        assert (downloaded, eta) == (500, None)

    def test_a_download_with_no_network_counts_gets_the_rate_its_file_grows_at(
            self, monkeypatch):
        monkeypatch.setattr(pull_mod, "METER", FakeMeter(transferred=0, rate=None))
        tracker = pull_mod._ByteTracker(total=10_000, base=0)
        tracker._disk_rate.add(0, now=0.0)
        tracker._disk_rate.add(1000, now=1.0)
        monkeypatch.setattr("localm.model_manager._transfer.time.monotonic", lambda: 1.0)
        downloaded, rate, eta = tracker.sample(on_disk=2000)
        assert downloaded == 2000
        assert rate is not None and rate > 0
        assert eta is not None


@pytest.fixture()
def gui(monkeypatch):
    monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")


class TestGuiFrames:
    def test_frames_carry_the_rate_and_time_left(self, gui, capsys, monkeypatch):
        monkeypatch.setattr(pull_mod, "METER", FakeMeter(transferred=400, rate=100.0))
        run = pull_mod._ProgressRun(lambda: 0, 1000)
        run.start()
        run.finish(succeeded=False)
        first = _events(capsys)[0]
        assert first["downloaded"] == 400
        assert first["rate"] == 100
        assert first["eta"] == 6

    def test_a_frame_without_a_measured_rate_omits_it(self, gui, capsys, monkeypatch):
        monkeypatch.setattr(pull_mod, "METER", FakeMeter(transferred=0, rate=None))
        run = pull_mod._ProgressRun(lambda: 0, 1000)
        run.start()
        run.finish(succeeded=False)
        first = _events(capsys)[0]
        assert "rate" not in first and "eta" not in first

    def test_the_closing_frame_reports_the_total_only_on_success(self, gui, capsys, monkeypatch):
        monkeypatch.setattr(pull_mod, "METER", FakeMeter())
        run = pull_mod._ProgressRun(lambda: 30, 100)
        run.start()
        run.finish(succeeded=True)
        assert _events(capsys)[-1]["downloaded"] == 100

        run = pull_mod._ProgressRun(lambda: 30, 100)
        run.start()
        run.finish(succeeded=False)
        assert _events(capsys)[-1]["downloaded"] == 30

    def test_a_snapshot_download_streams_the_rate_too(self, gui, capsys, monkeypatch):
        monkeypatch.setattr(pull_mod, "METER", FakeMeter(transferred=250, rate=25.0))
        with mm._snapshot_progress(lambda: 0, 1000, label="model.gguf") as prog:
            prog.ok()
        evs = _events(capsys)
        assert any(e.get("rate") == 25 for e in evs)


class TestTerminalBar:
    def test_a_terminal_shows_a_bar_with_size_speed_and_time_left(self, monkeypatch):
        from rich.console import Console
        buf = io.StringIO()
        monkeypatch.setattr(pull_mod, "console",
                            Console(file=buf, force_terminal=True, width=110))
        monkeypatch.delenv("LOCALM_PROGRESS_JSON", raising=False)
        monkeypatch.setattr(pull_mod, "METER",
                            FakeMeter(transferred=500_000, rate=250_000.0))
        run = pull_mod._ProgressRun(lambda: 0, 1_000_000, label="m.gguf")
        assert run.active
        run.start()
        run.finish(succeeded=False)
        text = buf.getvalue()
        assert "m.gguf" in text
        assert "500.0 KB" in text and "1.0 MB" in text
        assert "250.0 KB/s" in text
        assert "ETA 2s" in text

    def test_a_pipe_shows_nothing(self, monkeypatch, capsys):
        monkeypatch.delenv("LOCALM_PROGRESS_JSON", raising=False)
        run = pull_mod._ProgressRun(lambda: 0, 1000)
        assert not run.active
        run.start()
        run.finish(succeeded=True)
        assert capsys.readouterr().out == ""


class TestListener:
    def test_a_listener_gets_a_readable_line_and_is_throttled(self, monkeypatch):
        monkeypatch.delenv("LOCALM_PROGRESS_JSON", raising=False)
        monkeypatch.setattr(pull_mod, "METER", FakeMeter(transferred=250, rate=25.0))
        lines = []
        with pull_mod.progress_listener(lines.append):
            run = pull_mod._ProgressRun(lambda: 0, 1000, label="m.gguf")
            run.start()
            run._tick()
            run.finish(succeeded=False)
        assert lines == ["m.gguf: 25% (250 B / 1.0 KB) at 25 B/s, 30s left"]

    def test_the_listener_is_removed_afterwards(self):
        with pull_mod.progress_listener(lambda s: None):
            assert pull_mod._PROGRESS_LISTENER is not None
        assert pull_mod._PROGRESS_LISTENER is None

    def test_the_mcp_pull_tool_passes_progress_to_its_client(self, monkeypatch):
        from localm.plugins.mcpserver.server import (EngineCache, MCPStdioServer,
                                                     build_tools)
        seen = {}

        def fake_pull(spec, name=None, **kw):
            seen["listener"] = pull_mod._PROGRESS_LISTENER
            return True

        engines = EngineCache(default_model="m", engine_factory=lambda n: MagicMock(
            display_name=n, active_requests=0, unloading=False))
        server = MCPStdioServer(build_tools(engines, enable_images=False))
        with patch("localm.model_manager.pull.pull_model", fake_pull):
            server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": "pull_model", "arguments": {
                               "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
                               "name": "newmodel", "load": False}}})
        from localm.plugins.mcpserver import server as srv
        assert seen["listener"] is srv.report_progress


class TestSnapshotBytes:
    def test_temp_files_of_files_still_downloading_are_counted(self, tmp_path):
        dest = tmp_path / "snap"
        (dest / ".cache" / "huggingface" / "download").mkdir(parents=True)
        (dest / "config.json").write_bytes(b"x" * 10)
        (dest / ".cache" / "huggingface" / "download" / "abc.etag.incomplete").write_bytes(
            b"y" * 100)
        (dest / ".cache" / "huggingface" / "download" / "abc.metadata").write_bytes(b"z" * 7)
        assert pull_mod._snapshot_bytes_on_disk(dest) == 110


class TestDiscardPartial:
    @pytest.fixture()
    def models(self, tmp_path, monkeypatch):
        models = tmp_path / "models"
        models.mkdir()
        monkeypatch.setattr(mm, "MODELS_DIR", models)
        return models

    @staticmethod
    def _temp(models, rel, etag, size, owner=True, id_="aaaa1111"):
        from huggingface_hub._local_folder import get_local_download_paths
        inc = get_local_download_paths(models, rel).incomplete_path(etag)
        inc.parent.mkdir(parents=True, exist_ok=True)
        f = inc.with_name(f"{inc.stem}.{id_}.incomplete")
        f.write_bytes(b"x" * size)
        if owner:
            pull_mod._partial_owner_path(f).write_text(
                json.dumps({"pid": 2 ** 22 + 12345, "space": None}), encoding="utf-8")
        return f

    def test_a_single_file_pull_loses_its_temp_files_and_owner_records(self, models):
        mine = self._temp(models, "m.gguf", "e1", 40)
        stock = self._temp(models, "m.gguf", "e2", 60, owner=False, id_="bbbb2222")
        other = self._temp(models, "other.gguf", "e1", 25)
        freed = pull_mod.discard_partial_download("owner/repo:m.gguf")
        assert freed == 100
        assert not mine.exists() and not pull_mod._partial_owner_path(mine).exists()
        assert not stock.exists()
        assert other.exists(), "another file's temp data was deleted"

    def test_every_part_of_a_split_file_is_discarded(self, models):
        a = self._temp(models, "m-00001-of-00002.gguf", "e", 5)
        b = self._temp(models, "m-00002-of-00002.gguf", "e", 7)
        assert pull_mod.discard_partial_download(
            "owner/repo:m-00001-of-00002.gguf") == 12
        assert not a.exists() and not b.exists()

    def test_a_finished_file_is_never_deleted(self, models):
        done = models / "m.gguf"
        done.write_bytes(b"x" * 9)
        self._temp(models, "m.gguf", "e", 3)
        pull_mod.discard_partial_download("owner/repo:m.gguf")
        assert done.read_bytes() == b"x" * 9

    def test_a_running_download_is_left_alone(self, models):
        from filelock import FileLock
        from huggingface_hub._local_folder import get_local_download_paths
        temp = self._temp(models, "m.gguf", "e", 30)
        holder = FileLock(str(get_local_download_paths(models, "m.gguf").lock_path))
        holder.acquire()
        try:
            with pytest.raises(pull_mod.PullInFlight):
                pull_mod.discard_partial_download("owner/repo:m.gguf")
        finally:
            holder.release()
        assert temp.exists()

    def test_a_direct_url_loses_its_part_file_and_record(self, models):
        part = models / "model.gguf.part"
        part.write_bytes(b"x" * 21)
        pull_mod._part_record_path(part).write_text("{}", encoding="utf-8")
        assert pull_mod.discard_partial_download("https://example.com/model.gguf") == 21
        assert not part.exists() and not pull_mod._part_record_path(part).exists()

    def test_a_full_repo_download_loses_only_its_temp_files(self, models):
        folder = models / "snap" / ".cache" / "huggingface" / "download"
        folder.mkdir(parents=True)
        temp = folder / "abc.e.incomplete"
        temp.write_bytes(b"x" * 50)
        kept = models / "snap" / "config.json"
        kept.write_text("{}", encoding="utf-8")
        pull_mod._partial_owner_path(temp).write_text(
            json.dumps({"pid": 2 ** 22 + 12345, "space": None}), encoding="utf-8")
        assert pull_mod.discard_partial_download("owner/snap", "snap") == 50
        assert kept.exists() and not temp.exists()

    def test_an_unsafe_spec_deletes_nothing(self, models):
        assert pull_mod.discard_partial_download("owner/repo:../../evil.gguf") == 0
        assert pull_mod.discard_partial_download("not a spec") == 0


class TestCliInterrupt:
    def test_ctrl_c_stops_with_a_resumable_message_and_exit_130(self):
        from click.testing import CliRunner

        from localm.cli import main
        with patch("localm.cli.models.pull_model", side_effect=KeyboardInterrupt):
            result = CliRunner().invoke(main, ["pull", "--", "owner/repo:m.gguf"])
        assert result.exit_code == 130
        assert "run the same command again" in result.output

    def test_a_failed_pull_still_exits_1(self):
        from click.testing import CliRunner

        from localm.cli import main
        with patch("localm.cli.models.pull_model", return_value=False):
            result = CliRunner().invoke(main, ["pull", "--", "owner/repo:m.gguf"])
        assert result.exit_code == 1


class TestReviewedEdges:
    """Edges found in review: subfolders, the file's own lock, a restart from zero."""

    @pytest.fixture()
    def models(self, tmp_path, monkeypatch):
        models = tmp_path / "models"
        models.mkdir()
        monkeypatch.setattr(mm, "MODELS_DIR", models)
        return models

    @staticmethod
    def _temp(base, rel, size):
        from huggingface_hub._local_folder import get_local_download_paths
        inc = get_local_download_paths(base, rel).incomplete_path("e1")
        inc.parent.mkdir(parents=True, exist_ok=True)
        f = inc.with_name(f"{inc.stem}.aaaa1111.incomplete")
        f.write_bytes(b"x" * size)
        return f

    def test_a_file_inside_a_repo_folder_loses_its_temp_file(self, models):
        staging = models / pull_mod._REPO_FOLDER_STAGING
        temp = self._temp(staging, "sub/dir/m.gguf", 70)
        assert pull_mod.discard_partial_download("owner/repo:sub/dir/m.gguf") == 70
        assert not temp.exists()

    def test_a_full_repo_download_loses_the_temp_files_in_its_subfolders(self, models):
        dest = models / "snap"
        top = self._temp(dest, "model.safetensors", 10)
        nested = self._temp(dest, "unet/diffusion.safetensors", 20)
        for f in (top, nested):
            pull_mod._partial_owner_path(f).write_text(
                json.dumps({"pid": 2 ** 22 + 12345, "space": None}), encoding="utf-8")
        assert pull_mod.discard_partial_download("owner/snap", "snap") == 30
        assert not top.exists() and not nested.exists()

    def test_the_download_lock_is_free_again_after_a_discard(self, models):
        from huggingface_hub._local_folder import get_local_download_paths
        self._temp(models, "m.gguf", 5)
        pull_mod.discard_partial_download("owner/repo:m.gguf")
        assert pull_mod._lock_file_is_free(
            get_local_download_paths(models, "m.gguf").lock_path)

    def test_a_restart_from_zero_lowers_the_starting_figure(self, monkeypatch):
        monkeypatch.setattr(pull_mod, "METER", FakeMeter(transferred=100, rate=10.0))
        tracker = pull_mod._ByteTracker(total=1000, base=500)
        assert tracker.sample(on_disk=0)[0] == 100

    def test_a_failing_first_report_does_not_abort_the_download(self, monkeypatch):
        monkeypatch.delenv("LOCALM_PROGRESS_JSON", raising=False)
        monkeypatch.setattr(pull_mod, "METER", FakeMeter(transferred=10, rate=1.0))

        def boom(text):
            raise RuntimeError("sink gone")

        with pull_mod.progress_listener(boom):
            run = pull_mod._ProgressRun(lambda: 0, 100)
            run.start()
            run.finish(succeeded=False)

    def test_the_snapshot_aggregate_bars_can_read_the_bar_they_are_given(self):
        from huggingface_hub.utils._xet_progress_reporting import _set_aggregate_rate_postfix

        from localm.model_manager._transfer import MeterBar
        _set_aggregate_rate_postfix(MeterBar(desc="Downloading bytes", unit="B"))
