# SPDX-License-Identifier: AGPL-3.0-or-later
"""A Hugging Face pull checks every file it keeps against the digest the hub
publishes, reads that digest and the bytes at one commit, and says so when no
digest could be read. The hub is replaced by fakes; the pull, the hashing and
the deletion on a mismatch are the real code."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from localm import model_manager as mm
from localm.model_manager import pull as pull_mod

COMMIT = "0123456789abcdef0123456789abcdef01234567"
BODY = b"model-bytes-" * 64


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture()
def hub(tmp_path, monkeypatch):
    """A fake hub: ``files`` maps a repo file name to the bytes the CDN serves,
    ``digests`` to the digest the metadata API publishes (absent = not LFS).
    Records every revision the pull asked for."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models_dir)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    state = SimpleNamespace(files={}, digests={}, commit=None, download_revisions=[],
                            models_dir=models_dir)

    def fake_sha(repo, filename, token=None):
        digest = state.digests.get(filename)
        return pull_mod._PublishedDigest(digest, state.commit) if digest else None

    def fake_download(repo_id, filename, local_dir, revision=None, **kw):
        state.download_revisions.append(revision)
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(state.files[filename])
        return str(p)

    def fake_head(url, allow_redirects=None, timeout=None):
        h = MagicMock()
        h.headers = {"content-length": "1"}
        return h

    import huggingface_hub
    monkeypatch.setattr(mm, "_hf_file_sha256", fake_sha)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr("requests.head", fake_head)
    return state


def _pull(hub, spec, **kw):
    dest = hub.models_dir.parent / "out"
    return mm._pull_gguf_file(spec, None, dest_dir=dest, register=False, **kw), dest


class TestNoPublishedDigest:
    def test_a_download_without_a_digest_says_it_was_not_checked(self, hub, capsys, caplog):
        hub.files["a.safetensors"] = BODY
        with caplog.at_level(logging.WARNING):
            ok, dest = _pull(hub, "o/r:a.safetensors")
        assert ok is True and (dest / "a.safetensors").read_bytes() == BODY
        assert "no sha256 could be read for a.safetensors" in " ".join(
            capsys.readouterr().out.split())
        assert any("not integrity-checked" in r.getMessage() for r in caplog.records)

    def test_a_download_with_a_digest_prints_no_such_note(self, hub, capsys):
        hub.files["a.safetensors"] = BODY
        hub.digests["a.safetensors"] = _sha(BODY)
        ok, _dest = _pull(hub, "o/r:a.safetensors")
        assert ok is True
        assert "no sha256 could be read" not in capsys.readouterr().out

    def test_the_user_supplied_digest_counts_as_one(self, hub, capsys):
        hub.files["a.safetensors"] = BODY
        ok, _dest = _pull(hub, "o/r:a.safetensors", expected_sha256=_sha(BODY))
        assert ok is True
        assert "no sha256 could be read" not in capsys.readouterr().out

    def test_the_metadata_failure_is_logged_not_swallowed(self, monkeypatch):
        import huggingface_hub

        class Failing:
            def __init__(self, *a, **k):
                pass

            def get_paths_info(self, *a, **k):
                raise OSError("rate limited")

        seen = []
        monkeypatch.setattr(huggingface_hub, "HfApi", Failing)
        monkeypatch.setattr(pull_mod, "logger",
                            SimpleNamespace(debug=lambda *a, **k: seen.append(a)))
        assert pull_mod._hf_file_sha256("o/r", "f.gguf") is None
        assert seen and "rate limited" in str(seen[0][-1])


class TestOneCommit:
    def test_each_file_is_downloaded_at_the_commit_its_digest_was_read_at(self, hub):
        hub.commit = COMMIT
        hub.files["a.safetensors"] = BODY
        hub.digests["a.safetensors"] = _sha(BODY)
        ok, _dest = _pull(hub, "o/r:a.safetensors")
        assert ok is True
        assert hub.download_revisions == [COMMIT]

    def test_without_a_known_commit_nothing_is_pinned(self, hub):
        hub.files["a.safetensors"] = BODY
        hub.digests["a.safetensors"] = _sha(BODY)
        ok, _dest = _pull(hub, "o/r:a.safetensors")
        assert ok is True
        assert hub.download_revisions == [None]

    def test_the_metadata_call_reports_the_files_last_commit(self, monkeypatch):
        import huggingface_hub
        asked = {}

        class Api:
            def __init__(self, *a, **k):
                pass

            def get_paths_info(self, repo_id, paths, expand=False, token=None, **kw):
                asked["expand"] = expand
                return [SimpleNamespace(lfs=SimpleNamespace(sha256="AB" * 32),
                                        last_commit=SimpleNamespace(oid=COMMIT))]

        monkeypatch.setattr(huggingface_hub, "HfApi", Api)
        digest = pull_mod._hf_file_sha256("o/r", "f.gguf")
        assert digest == "ab" * 32 and digest.commit == COMMIT
        assert asked == {"expand": True}

    @pytest.mark.parametrize("oid", ["main", None, "A" * 40, "abc"])
    def test_a_commit_that_is_not_a_full_commit_id_is_dropped(self, oid):
        digest = pull_mod._PublishedDigest("ab" * 32, oid)
        assert digest == "ab" * 32 and digest.commit is None
        assert pull_mod._at_commit(digest) == {}

    def test_a_plain_string_digest_pins_nothing(self):
        assert pull_mod._at_commit("ab" * 32) == {}
        assert pull_mod._at_commit(None) == {}


class TestSplitGguf:
    PARTS = ("m-00001-of-00002.gguf", "m-00002-of-00002.gguf")

    def _split(self, hub, second=b"second-part" * 50):
        first = b"first-part" * 50
        hub.files.update({self.PARTS[0]: first, self.PARTS[1]: second})
        hub.digests.update({self.PARTS[0]: _sha(first),
                            self.PARTS[1]: _sha(b"second-part" * 50)})

    def test_every_part_matching_is_accepted(self, hub):
        self._split(hub)
        ok, dest = _pull(hub, f"o/r:{self.PARTS[0]}")
        assert ok is True and all((dest / p).is_file() for p in self.PARTS)

    def test_a_damaged_later_part_fails_the_pull_and_deletes_every_part(self, hub, capsys):
        self._split(hub, second=b"damaged-part" * 50)
        ok, dest = _pull(hub, f"o/r:{self.PARTS[0]}")
        assert ok is False
        assert not any((dest / p).exists() for p in self.PARTS)
        out = capsys.readouterr().out
        assert "SHA256 mismatch" in out and self.PARTS[1] in out


class TestAlreadyDownloaded:
    def test_a_file_that_does_not_match_the_published_digest_is_refused(self, hub, capsys):
        dest = hub.models_dir.parent / "out"
        dest.mkdir()
        (dest / "a.safetensors").write_bytes(b"truncated")
        hub.files["a.safetensors"] = BODY
        hub.digests["a.safetensors"] = _sha(BODY)
        ok, _ = _pull(hub, "o/r:a.safetensors")
        assert ok is False
        assert "does not match the digest Hugging Face publishes" in " ".join(
            capsys.readouterr().out.split())
        assert (dest / "a.safetensors").read_bytes() == b"truncated"

    def test_a_file_that_matches_is_accepted_without_a_download(self, hub):
        dest = hub.models_dir.parent / "out"
        dest.mkdir()
        (dest / "a.safetensors").write_bytes(BODY)
        hub.digests["a.safetensors"] = _sha(BODY)
        ok, _ = _pull(hub, "o/r:a.safetensors")
        assert ok is True and hub.download_revisions == []

    def test_without_a_published_digest_the_file_is_accepted_and_the_gap_is_said(
            self, hub, capsys):
        dest = hub.models_dir.parent / "out"
        dest.mkdir()
        (dest / "a.safetensors").write_bytes(b"whatever")
        ok, _ = _pull(hub, "o/r:a.safetensors")
        assert ok is True
        assert "no sha256 could be read" in capsys.readouterr().out


class TestVisionProjector:
    def _download(self, hub, tmp_path, name="mmproj-f16.gguf"):
        base = tmp_path / "proj"
        base.mkdir()
        dest = base / name
        pull_mod._download_mmproj_file("o/r", name, dest, base)
        return dest

    def test_a_projector_that_matches_is_kept(self, hub, tmp_path):
        hub.files["mmproj-f16.gguf"] = BODY
        hub.digests["mmproj-f16.gguf"] = _sha(BODY)
        assert self._download(hub, tmp_path).read_bytes() == BODY

    def test_a_projector_that_does_not_match_is_deleted_and_reported(self, hub, tmp_path):
        hub.files["mmproj-f16.gguf"] = b"damaged" * 20
        hub.digests["mmproj-f16.gguf"] = _sha(BODY)
        with pytest.raises(RuntimeError, match="sha256 mismatch for mmproj-f16.gguf"):
            self._download(hub, tmp_path)
        assert not (tmp_path / "proj" / "mmproj-f16.gguf").exists()

    def test_a_projector_without_a_digest_is_kept_with_a_note(self, hub, tmp_path, capsys):
        hub.files["mmproj-f16.gguf"] = BODY
        assert self._download(hub, tmp_path).read_bytes() == BODY
        assert "no sha256 could be read for mmproj-f16.gguf" in " ".join(
            capsys.readouterr().out.split())

    def test_the_projector_is_fetched_at_the_commit(self, hub, tmp_path):
        hub.commit = COMMIT
        hub.files["mmproj-f16.gguf"] = BODY
        hub.digests["mmproj-f16.gguf"] = _sha(BODY)
        self._download(hub, tmp_path)
        assert hub.download_revisions == [COMMIT]


class TestSnapshot:
    FILES = {"config.json": b'{"model_type": "bert"}', "model.safetensors": BODY}

    @pytest.fixture()
    def snap(self, tmp_path, monkeypatch):
        import huggingface_hub
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        monkeypatch.setattr(mm, "MODELS_DIR", models_dir)
        monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
        monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
        monkeypatch.setattr(mm, "load_registry", lambda: {})
        registered = []
        monkeypatch.setattr(mm, "_register_with_dedup",
                            lambda *a, **k: registered.append(a[0]) or True)
        state = SimpleNamespace(served=dict(self.FILES), revisions=[], registered=registered,
                                listing=True, sha="main", models_dir=models_dir)

        def siblings():
            return [SimpleNamespace(rfilename=n, size=len(b),
                                    lfs=SimpleNamespace(sha256=_sha(b)) if n.endswith(".safetensors")
                                    else None)
                    for n, b in self.FILES.items()]

        class Api:
            def __init__(self, *a, **k):
                pass

            def model_info(self, repo_id, files_metadata=False, token=None):
                if not state.listing:
                    raise OSError("offline")
                return SimpleNamespace(siblings=siblings(), sha=state.sha)

        def snapshot_download(repo_id, local_dir, endpoint=None, revision=None):
            state.revisions.append(revision)
            for n, b in state.served.items():
                p = Path(local_dir) / n
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b)
            return local_dir

        monkeypatch.setattr(huggingface_hub, "HfApi", Api)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
        monkeypatch.setattr(pull_mod, "_resolve_snapshot_type", lambda *a, **k: "llm")
        return state

    def test_files_that_match_are_registered_and_fetched_at_the_listed_commit(self, snap):
        snap.sha = COMMIT
        assert mm._pull_hf_snapshot("o/bert", None) is True
        assert snap.registered == ["bert"] and snap.revisions == [COMMIT]

    def test_a_damaged_weight_file_fails_the_pull_and_is_deleted(self, snap, capsys):
        snap.served["model.safetensors"] = b"damaged" * 20
        assert mm._pull_hf_snapshot("o/bert", None) is False
        out = capsys.readouterr().out
        assert "SHA256 mismatch" in out and "model.safetensors" in out
        assert snap.registered == []
        assert not (snap.models_dir / "bert" / "model.safetensors").exists()

    def test_a_missing_weight_file_fails_the_pull(self, snap):
        del snap.served["model.safetensors"]
        assert mm._pull_hf_snapshot("o/bert", None) is False and snap.registered == []

    def test_without_a_listing_the_pull_goes_ahead_and_says_it_was_not_checked(
            self, snap, capsys):
        snap.listing = False
        assert mm._pull_hf_snapshot("o/bert", None) is True
        assert "not checked against them" in capsys.readouterr().out
        assert snap.revisions == [None]
