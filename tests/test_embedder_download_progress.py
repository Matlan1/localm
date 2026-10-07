# SPDX-License-Identifier: AGPL-3.0-or-later
"""A one-time embedding-model download reports its progress (bytes received of
the total) to the caller instead of drawing a bar on the server console."""

from __future__ import annotations

import logging

import pytest

from localm.inference import embedder

MB = 1024 * 1024


def _fake_hf_download(chunks, total):
    """An ``hf_hub_download`` stand-in that drives ``tqdm_class`` through
    huggingface_hub's own progress-bar factory, as its HTTP download does."""
    def _download(repo, filename, local_dir=None, endpoint=None, tqdm_class=None):
        from pathlib import Path
        from huggingface_hub.utils.tqdm import _create_progress_bar, tqdm
        bar = _create_progress_bar(cls=tqdm_class or tqdm, log_level=logging.INFO,
                                   name="huggingface_hub.http_get", unit="B",
                                   unit_scale=True, total=total, initial=0,
                                   desc=filename)
        with bar:
            for n in chunks:
                bar.update(n)
        out = Path(local_dir) / filename
        out.write_bytes(b"gguf")
        return str(out)
    return _download


class TestProgressLine:
    def test_with_total(self):
        assert (embedder.download_progress_line("the embedding model x", 13 * MB, 130 * MB)
                == "Downloading the embedding model x: 13 of 130 MB (10%)...")

    def test_without_total(self):
        assert (embedder.download_progress_line("m", 5 * MB, None)
                == "Downloading m: 5 MB...")


class TestReporter:
    def test_each_update_is_counted_and_reported_when_due(self):
        lines = []
        cls = embedder._download_progress_class("m", lines.append, every=0.0)
        bar = cls(total=4 * MB, initial=MB, desc="f")
        bar.update(MB)
        bar.update(2 * MB)
        bar.close()
        assert lines == ["Downloading m: 2 of 4 MB (50%)...",
                         "Downloading m: 4 of 4 MB (100%)..."]

    def test_a_raising_callback_does_not_break_the_download(self):
        def _boom(text):
            raise RuntimeError("sink gone")
        cls = embedder._download_progress_class("m", _boom, every=0.0)
        bar = cls(total=MB)
        bar.update(MB)
        bar.close()


class TestDownloadKnown:
    @pytest.fixture
    def allow(self, monkeypatch):
        monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "allow")

    def test_progress_reaches_the_caller(self, tmp_path, monkeypatch, allow):
        monkeypatch.setattr("huggingface_hub.hf_hub_download",
                            _fake_hf_download([MB, MB, MB], 3 * MB))
        lines = []
        dest = tmp_path / "model.gguf"
        got = embedder._download_known("bge", "org/repo", "model.gguf", dest, True,
                                       on_progress=lines.append)
        assert got and dest.is_file()
        assert lines and lines[0].startswith("Downloading the embedding model bge: ")
        assert "of 3 MB" in lines[0]

    def test_without_a_callback_no_reporter_is_used(self, tmp_path, monkeypatch, allow):
        seen = {}

        def _download(repo, filename, local_dir=None, endpoint=None, **kw):
            seen.update(kw)
            from pathlib import Path
            out = Path(local_dir) / filename
            out.write_bytes(b"gguf")
            return str(out)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)
        embedder._download_known("bge", "org/repo", "model.gguf",
                                 tmp_path / "model.gguf", True)
        assert "tqdm_class" not in seen


class TestWillDownloadOnFirstUse:
    @pytest.fixture
    def fresh(self, monkeypatch, tmp_path):
        monkeypatch.setattr(embedder, "_EMBEDDER", None)
        monkeypatch.setattr(embedder, "_TRIED_DOWNLOAD", False)
        monkeypatch.setattr(embedder, "_embeddings_dir", lambda: tmp_path)
        monkeypatch.setattr(embedder, "_current_spec", lambda: "bge-small-en-v1.5")
        monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "allow")
        return tmp_path

    def test_a_missing_known_model_under_allow_will_download(self, fresh):
        assert embedder.will_download_on_first_use() is True

    def test_a_model_on_disk_will_not(self, fresh):
        _repo, filename = embedder.KNOWN_EMBEDDING_MODELS["bge-small-en-v1.5"]
        (fresh / filename).write_bytes(b"gguf")
        assert embedder.will_download_on_first_use() is False

    @pytest.mark.parametrize("change", ["loaded", "tried", "ask", "path"])
    def test_any_other_state_will_not(self, fresh, monkeypatch, change):
        if change == "loaded":
            monkeypatch.setattr(embedder, "_EMBEDDER", object())
        elif change == "tried":
            monkeypatch.setattr(embedder, "_TRIED_DOWNLOAD", True)
        elif change == "ask":
            monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "ask")
        else:
            monkeypatch.setattr(embedder, "_current_spec", lambda: "/path/to/x.gguf")
        assert embedder.will_download_on_first_use() is False
