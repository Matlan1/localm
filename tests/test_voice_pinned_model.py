# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Whisper speech models are fetched at a pinned commit and their model.bin
is checked against a pinned sha256 before it is used; a cache filled before the
pin existed keeps loading offline. The hub and the model class are fakes; the
pin lookup, the hashing, the deletion and the revision choice are the real code."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import huggingface_hub
import pytest

from localm import voice
from tests.conftest import set_config

BODY = b"whisper-model-bytes" * 100
REPO = "Systran/faster-whisper-base"
PIN_COMMIT = voice._STT_PINS[REPO][0]
OLD_COMMIT = "1" * 40


@pytest.fixture(autouse=True)
def _home(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    set_config(monkeypatch, lambda: {"voice_stt_model": "base", "net_mode": "allow"})


@pytest.fixture()
def pinned_to_body(monkeypatch):
    digest = hashlib.sha256(BODY).hexdigest()
    monkeypatch.setattr(voice, "_STT_PINS", {
        repo: (commit, digest) for repo, (commit, _) in voice._STT_PINS.items()})


def _lay(commit: str, body: bytes = BODY, name: str = "base", refs_main: bool = False) -> Path:
    repo_dir = voice.stt_cache_dir() / f"models--Systran--faster-whisper-{name}"
    snap = repo_dir / "snapshots" / commit
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "model.bin").write_bytes(body)
    (snap / "config.json").write_text("{}", encoding="utf-8")
    if refs_main:
        (repo_dir / "refs").mkdir(exist_ok=True)
        (repo_dir / "refs" / "main").write_text(commit, encoding="utf-8")
    return repo_dir


def test_the_pins_cover_the_four_whisper_sizes_and_are_well_formed():
    assert set(voice._STT_PINS) == {voice._stt_repo_for(n)
                                    for n in ("tiny", "base", "small", "medium")}
    for commit, digest in voice._STT_PINS.values():
        assert re.fullmatch(r"[0-9a-f]{40}", commit) and re.fullmatch(r"[0-9a-f]{64}", digest)
    assert voice._stt_pin("base") == voice._STT_PINS[REPO]
    assert voice._stt_pin("someone/else") is None


class TestRevisionChoice:
    def test_a_download_is_made_at_the_pinned_commit(self, tmp_path):
        assert voice._stt_revision("base", tmp_path, local_files_only=False) == PIN_COMMIT

    def test_a_cached_pinned_snapshot_loads_at_the_pinned_commit(self, pinned_to_body):
        _lay(PIN_COMMIT)
        assert voice._stt_revision("base", voice.stt_cache_dir(), True) == PIN_COMMIT

    def test_a_cache_with_only_an_older_snapshot_loads_from_its_own_default(self, pinned_to_body):
        _lay(OLD_COMMIT, refs_main=True)
        assert voice.stt_model_cached() == (True, "base")
        assert voice._stt_revision("base", voice.stt_cache_dir(), True) is None

    def test_a_model_that_is_not_pinned_has_no_revision(self, tmp_path):
        assert voice._stt_revision("someone/else", tmp_path, False) is None
        assert voice._stt_revision("someone/else", tmp_path, True) is None


class TestPrefetch:
    @pytest.fixture()
    def hub(self, monkeypatch, pinned_to_body):
        calls = []
        state = {"body": BODY}

        def snapshot_download(repo_id, **kw):
            calls.append((repo_id, kw))
            _lay(kw.get("revision") or "0" * 40, state["body"])
            return "unused"

        monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
        return SimpleHub(calls, state)

    def test_the_fetch_is_made_at_the_pinned_commit_from_the_public_hub(self, hub):
        assert voice.prefetch_stt_model(allow_download=True) == (True, "")
        (repo, kw), = hub.calls
        assert repo == REPO and kw["revision"] == PIN_COMMIT
        assert kw["endpoint"] == "https://huggingface.co"
        assert voice.stt_model_cached() == (True, "base")

    def test_a_model_bin_that_is_not_the_pinned_file_is_deleted_and_refused(self, hub):
        hub.state["body"] = b"another upload" * 100
        ok, reason = voice.prefetch_stt_model(allow_download=True)
        assert ok is False and "integrity check" in reason
        assert not (voice.stt_cache_dir() / "models--Systran--faster-whisper-base").exists()
        assert voice.stt_model_cached() == (False, "base")

    def test_an_unpinned_model_is_fetched_without_a_revision(self, hub, monkeypatch):
        set_config(monkeypatch, lambda: {"voice_stt_model": "Systran/faster-whisper-large-v3",
                                     "net_mode": "allow"})

        def snapshot_download(repo_id, **kw):
            hub.calls.append((repo_id, kw))
            d = (voice.stt_cache_dir() / "models--Systran--faster-whisper-large-v3"
                 / "snapshots" / ("2" * 40))
            d.mkdir(parents=True)
            (d / "model.bin").write_bytes(b"x")
            return "unused"

        monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
        assert voice.prefetch_stt_model(allow_download=True) == (True, "")
        assert "revision" not in hub.calls[0][1]


class SimpleHub:
    def __init__(self, calls, state):
        self.calls, self.state = calls, state


class FakeModel:
    instances: list = []

    def __init__(self, source, **kw):
        self.source, self.kw = source, kw
        FakeModel.instances.append(self)


@pytest.fixture(autouse=True)
def _reset_fake_model():
    FakeModel.instances = []


class TestLoad:
    def test_a_cached_pinned_snapshot_loads_offline_at_the_pinned_commit(self, pinned_to_body):
        _lay(PIN_COMMIT)
        voice._load_whisper(FakeModel, "base", voice.stt_cache_dir(), True,
                            fetch=lambda *a: pytest.fail("a cached load must not fetch"))
        kw = FakeModel.instances[0].kw
        assert kw["local_files_only"] is True and kw["revision"] == PIN_COMMIT

    def test_a_cache_with_only_an_older_snapshot_still_loads_offline(self, pinned_to_body):
        _lay(OLD_COMMIT, refs_main=True)
        voice._load_whisper(FakeModel, "base", voice.stt_cache_dir(), True,
                            fetch=lambda *a: pytest.fail("a cached load must not fetch"))
        kw = FakeModel.instances[0].kw
        assert kw["local_files_only"] is True and "revision" not in kw

    def test_a_download_is_fetched_at_the_pinned_commit_checked_and_loaded_from_disk(
            self, pinned_to_body):
        seen = []

        def fetch(name, root, revision):
            seen.append((name, revision))
            return str(_lay(revision) / "snapshots" / revision)

        voice._load_whisper(FakeModel, "base", voice.stt_cache_dir(), False, fetch=fetch)
        assert seen == [("base", PIN_COMMIT)]
        assert FakeModel.instances[0].source.endswith(PIN_COMMIT)

    def test_a_download_that_is_not_the_pinned_file_is_deleted_and_never_loaded(
            self, pinned_to_body):
        def fetch(name, root, revision):
            return str(_lay(revision, b"another upload" * 100) / "snapshots" / revision)

        with pytest.raises(RuntimeError, match="integrity check"):
            voice._load_whisper(FakeModel, "base", voice.stt_cache_dir(), False, fetch=fetch)
        assert FakeModel.instances == []
        assert not (voice.stt_cache_dir() / "models--Systran--faster-whisper-base").exists()

    def test_a_model_that_is_not_pinned_loads_as_before(self, tmp_path):
        voice._load_whisper(FakeModel, "someone/else", tmp_path, False,
                            fetch=lambda *a: pytest.fail("an unpinned model is not fetched here"))
        kw = FakeModel.instances[0].kw
        assert kw["local_files_only"] is False and "revision" not in kw


def test_faster_whisper_finds_a_cache_filled_before_the_pin_only_without_a_revision(
        pinned_to_body):
    """The behaviour the offline fallback relies on, measured against the real
    library: a snapshot under another commit is found through the cache's own
    default reference, and asking for the pinned commit instead finds nothing."""
    utils = pytest.importorskip("faster_whisper.utils")
    root = voice.stt_cache_dir()
    _lay(OLD_COMMIT, refs_main=True)
    for filename in ("tokenizer.json", "vocabulary.txt"):
        (root / "models--Systran--faster-whisper-base" / "snapshots" / OLD_COMMIT
         / filename).write_text("{}", encoding="utf-8")
    found = utils.download_model("base", cache_dir=str(root), local_files_only=True, revision=None)
    assert Path(found).name == OLD_COMMIT
    with pytest.raises(Exception):  # noqa: B017 - whichever offline-miss error the hub client raises
        utils.download_model("base", cache_dir=str(root), local_files_only=True,
                             revision=PIN_COMMIT)
