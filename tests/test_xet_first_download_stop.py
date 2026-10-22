# SPDX-License-Identifier: AGPL-3.0-or-later
"""A stop during the first download of a file stored with Xet keeps the bytes
received, the rerun continues from them, and ``localm pull`` says how much is
saved.

The download runs through the real ``_resumable_download_to_tmp_and_move`` and
huggingface_hub's real ``http_get`` against a local HTTP server that streams
the file and honours Range requests."""

from __future__ import annotations

import http.server
import os
import threading
from pathlib import Path

import pytest
from click.testing import CliRunner

from localm.model_manager import pull as pull_mod
from localm.model_manager.pull import _resumable_download_to_tmp_and_move

SIZE = 4 * 1024 * 1024
STOP_AT = 1024 * 1024
BODY = bytes(i % 251 for i in range(SIZE))


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        start = 0
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start = int(rng[len("bytes="):].split("-")[0])
        self.server.ranges.append(start)
        body = BODY[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        if start:
            self.send_header("Content-Range", f"bytes {start}-{SIZE - 1}/{SIZE}")
        self.end_headers()
        try:
            for i in range(0, len(body), 64 * 1024):
                self.wfile.write(body[i:i + 64 * 1024])
        except (ConnectionError, OSError):
            pass


@pytest.fixture(autouse=True)
def small_chunks(monkeypatch):
    monkeypatch.setattr("huggingface_hub.constants.DOWNLOAD_CHUNK_SIZE", 64 * 1024)


@pytest.fixture()
def server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    httpd.ranges = []
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


def _stop_after(limit: int):
    """A tqdm_class whose bar raises KeyboardInterrupt on the first update after
    *limit* bytes were reported, as a Ctrl+C between two chunks does."""
    from tqdm.auto import tqdm

    class _Bar(tqdm):
        def __init__(self, *a, **kw):
            kw["disable"] = True
            super().__init__(*a, **kw)
            self.seen = 0

        def update(self, n=1):
            if self.seen >= limit:
                raise KeyboardInterrupt
            self.seen += n or 0
            return super().update(n)

    return _Bar


def _download(server, inc_path: Path, dest: Path, tqdm_class=None):
    _resumable_download_to_tmp_and_move(
        incomplete_path=inc_path,
        destination_path=dest,
        url_to_download=f"http://127.0.0.1:{server.server_address[1]}/file.gguf",
        headers={},
        expected_size=SIZE,
        filename="file.gguf",
        force_download=False,
        etag="etag",
        xet_file_data=object(),
        tqdm_class=tqdm_class,
    )


@pytest.fixture()
def xet_buffers_in_memory(monkeypatch):
    """hf_xet is available, and a Xet transfer holds what it received in memory
    until the end: a stop leaves nothing in its file."""
    calls = []

    def xet_get(*, incomplete_path, **kw):
        calls.append(incomplete_path)
        Path(incomplete_path).write_bytes(b"")
        raise KeyboardInterrupt

    monkeypatch.setattr("huggingface_hub.file_download.is_xet_available", lambda: True)
    monkeypatch.setattr("huggingface_hub.file_download.xet_get", xet_get)
    return calls


def _partials(cache: Path) -> "list[Path]":
    return [p for p in cache.iterdir() if p.name.endswith(".incomplete")]


def test_a_stopped_first_download_of_a_xet_file_keeps_the_bytes_received(
        tmp_path, server, xet_buffers_in_memory):
    cache = tmp_path / "cache"
    cache.mkdir()
    inc_path = cache / "file.etag.incomplete"
    dest = tmp_path / "file.gguf"

    with pytest.raises(KeyboardInterrupt):
        _download(server, inc_path, dest, tqdm_class=_stop_after(STOP_AT))

    parts = _partials(cache)
    assert len(parts) == 1
    kept = parts[0].stat().st_size
    assert STOP_AT <= kept < SIZE, f"{kept} bytes kept after {STOP_AT} were received"
    assert parts[0].read_bytes() == BODY[:kept]
    assert not dest.exists()
    assert xet_buffers_in_memory == []


def test_the_rerun_continues_from_the_bytes_kept(tmp_path, server, xet_buffers_in_memory):
    cache = tmp_path / "cache"
    cache.mkdir()
    inc_path = cache / "file.etag.incomplete"
    dest = tmp_path / "file.gguf"
    with pytest.raises(KeyboardInterrupt):
        _download(server, inc_path, dest, tqdm_class=_stop_after(STOP_AT))
    kept = _partials(cache)[0].stat().st_size

    _download(server, inc_path, dest)

    assert dest.read_bytes() == BODY
    assert server.ranges == [0, kept]
    assert _partials(cache) == []


def _invoke_pull_stopped(monkeypatch, measure, total):
    """Run ``localm pull`` with a download that Ctrl+C stops inside the
    download progress scope, *measure* being the bytes on disk."""
    from localm.cli import models as cli_models

    def fake_pull_model(*a, **kw):
        with pull_mod._snapshot_progress(measure, total):
            raise KeyboardInterrupt

    monkeypatch.setattr(cli_models, "pull_model", fake_pull_model)
    return CliRunner().invoke(cli_models.pull, ["owner/repo:model.gguf"])


def test_the_stop_message_states_the_bytes_saved(monkeypatch):
    result = _invoke_pull_stopped(monkeypatch, lambda: 755_300_000, 2_489_900_000)

    assert result.exit_code == 130
    assert "755.3 MB of 2.5 GB is saved to disk" in result.output
    assert "run the same command again to continue from there" in result.output


def test_the_stop_message_never_claims_bytes_kept_when_none_were_saved(monkeypatch):
    result = _invoke_pull_stopped(monkeypatch, lambda: 0, 2_489_900_000)

    assert result.exit_code == 130
    assert "before anything was saved to disk" in result.output
    assert "kept" not in result.output
    assert "is saved" not in result.output


def test_the_stop_record_measures_the_partial_left_by_the_stopped_download(
        tmp_path, server, xet_buffers_in_memory):
    base = tmp_path / "models"
    cache = base / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    inc_path = cache / "file.etag.incomplete"

    with pytest.raises(KeyboardInterrupt):
        with pull_mod._download_progress([base / "file.gguf"], SIZE, base_dir=base):
            _download(server, inc_path, base / "file.gguf",
                      tqdm_class=_stop_after(STOP_AT))

    kept = _partials(cache)[0].stat().st_size
    assert pull_mod.stopped_download() == (kept, SIZE)
    assert kept >= STOP_AT


def test_a_stop_outside_a_download_leaves_no_stop_record(monkeypatch):
    from localm.cli import models as cli_models
    with pull_mod._snapshot_progress(lambda: 5, 10):
        pass

    def fake_pull_model(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_models, "pull_model", fake_pull_model)
    result = CliRunner().invoke(cli_models.pull, ["owner/repo:model.gguf"])

    assert result.exit_code == 130
    assert "Download stopped. Run the same command again to continue." in result.output
    assert pull_mod.stopped_download() is None
