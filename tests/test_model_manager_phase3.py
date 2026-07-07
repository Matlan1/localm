# SPDX-License-Identifier: AGPL-3.0-or-later
"""Phase 3 hardening tests for localm.model_manager.

Covers three previously-unguarded behaviours:

FAC-5      --sha256 was a facade for HuggingFace pulls: only _pull_url consumed
           it, so a mismatching --sha256 on an HF GGUF/snapshot pull was silently
           ignored. It must now be honoured (verify against the downloaded HF
           file) or refused with a clear error - never quietly dropped.

GAP-CLI-1  A user-supplied -n name went into the registry raw (add_local used
           'name or p.stem'), so '../evil' or 'a/b' became a registry key
           unchanged. It must run through the same sanitizer sync_models_dir
           uses for auto-discovered names.

GAP-CLI-2  The GGUF/URL filename was used as a dest path with no traversal guard
           (o/r:../../evil.gguf or a URL ending in ../../evil.gguf), and a URL
           stem collision short-circuited to "Already downloaded" and aliased a
           new name onto whatever bytes already existed - even when the caller
           supplied a --sha256 that did not match those bytes.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from localm import model_manager as mm


@pytest.fixture()
def fake_registry(tmp_path, monkeypatch):
    """In-memory registry + temp MODELS_DIR wired into model_manager."""
    store: dict = {}
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models_dir)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    monkeypatch.setattr(mm, "load_registry", lambda: dict(store))

    def _save(reg):
        store.clear()
        store.update(reg)

    monkeypatch.setattr(mm, "save_registry", _save)

    def _update(mutator):
        reg = dict(store)
        mutator(reg)
        store.clear()
        store.update(reg)
        return dict(store)

    monkeypatch.setattr(mm, "update_registry", _update)
    return store, models_dir


def _fake_resp(body: bytes, status=200, content_length=None):
    r = MagicMock()
    r.status_code = status
    r.raise_for_status = MagicMock()
    cl = len(body) if content_length is None else content_length
    r.headers = {"content-length": str(cl)}

    def _iter(chunk_size):
        for i in range(0, len(body), chunk_size):
            yield body[i:i + chunk_size]

    r.iter_content = _iter
    return r


def _wire_http(monkeypatch, body: bytes):
    def fake_head(url, allow_redirects=None, timeout=None):
        h = MagicMock()
        h.headers = {"content-length": str(len(body))}
        return h

    def fake_get(url, headers=None, stream=None, timeout=None):
        return _fake_resp(body)

    monkeypatch.setattr("requests.head", fake_head)
    monkeypatch.setattr("requests.get", fake_get)


# ---------------------------------------------------------------------------
# FAC-5: --sha256 is honoured (not silently ignored) on HF pulls
# ---------------------------------------------------------------------------

class TestHfShaIsNotAFacade:
    def test_gguf_pull_rejects_on_sha256_mismatch(
            self, fake_registry, tmp_path, monkeypatch):
        """A wrong --sha256 on an HF GGUF pull must fail and not register."""
        store, models_dir = fake_registry
        # HF has no LFS metadata for this file (offline / non-LFS).
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda r, fn: None)
        monkeypatch.setattr(mm, "find_by_sha256", lambda *a, **k: [])

        def _fake_download(repo_id, filename, local_dir, **kw):
            p = Path(local_dir) / filename
            p.write_bytes(b"actual-bytes")
            return str(p)

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)

        wrong = "0" * 64
        ok = mm._pull_gguf_file("o/r:new.gguf", None, expected_sha256=wrong)

        assert ok is False
        assert "new" not in store
        # The corrupted/unexpected file must not be left lying around.
        assert not (models_dir / "new.gguf").exists()

    def test_gguf_pull_accepts_on_sha256_match(
            self, fake_registry, tmp_path, monkeypatch):
        store, models_dir = fake_registry
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda r, fn: None)
        monkeypatch.setattr(mm, "find_by_sha256", lambda *a, **k: [])

        body = b"actual-bytes"

        def _fake_download(repo_id, filename, local_dir, **kw):
            p = Path(local_dir) / filename
            p.write_bytes(body)
            return str(p)

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)

        good = mm._sha256_file_bytes(body)
        ok = mm._pull_gguf_file("o/r:new.gguf", None, expected_sha256=good)

        assert ok is True
        assert store["new"]["sha256"] == good

    def test_gguf_pull_rejects_sha256_conflicting_with_hf_metadata(
            self, fake_registry, tmp_path, monkeypatch):
        """If HF's own metadata digest disagrees with --sha256, refuse up front
        (no point downloading bytes we already know will not match)."""
        store, models_dir = fake_registry
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda r, fn: "a" * 64)
        monkeypatch.setattr(mm, "find_by_sha256", lambda *a, **k: [])

        downloaded = []
        import huggingface_hub
        monkeypatch.setattr(
            huggingface_hub, "hf_hub_download",
            lambda **kw: downloaded.append(1))

        ok = mm._pull_gguf_file("o/r:new.gguf", None, expected_sha256="b" * 64)

        assert ok is False
        assert downloaded == []          # refused before any network I/O
        assert "new" not in store

    def test_snapshot_pull_refuses_sha256(
            self, fake_registry, tmp_path, monkeypatch, capsys):
        """A full-repo snapshot has no single file digest, so --sha256 cannot be
        verified - it must be refused, not silently ignored."""
        store, _ = fake_registry

        downloaded = []
        import huggingface_hub
        monkeypatch.setattr(
            huggingface_hub, "snapshot_download",
            lambda **kw: downloaded.append(1))

        ok = mm._pull_hf_snapshot("owner/repo", None, expected_sha256="a" * 64)

        assert ok is False
        assert downloaded == []
        out = capsys.readouterr().out.lower()
        assert "sha256" in out

    def test_pull_model_threads_sha256_into_gguf(
            self, fake_registry, monkeypatch):
        """pull_model must forward --sha256 to the HF GGUF path (not drop it)."""
        captured = {}

        def _fake_gguf(spec, name, expected_sha256=None, redownload=False, **kw):
            captured["sha256"] = expected_sha256
            return True

        monkeypatch.setattr(mm, "resolve_spec", lambda s: s)
        monkeypatch.setattr(mm, "_pull_gguf_file", _fake_gguf)
        mm.pull_model("owner/repo:file.gguf", expected_sha256="dead" * 16)
        assert captured["sha256"] == "dead" * 16

    def test_pull_model_threads_sha256_into_snapshot(
            self, fake_registry, monkeypatch):
        captured = {}

        def _fake_snap(repo_id, name, expected_sha256=None, redownload=False, **kw):
            captured["sha256"] = expected_sha256
            return True

        monkeypatch.setattr(mm, "resolve_spec", lambda s: s)
        monkeypatch.setattr(mm, "_pull_hf_snapshot", _fake_snap)
        mm.pull_model("owner/repo", expected_sha256="beef" * 16)
        assert captured["sha256"] == "beef" * 16


# ---------------------------------------------------------------------------
# GAP-CLI-1: user -n name is sanitized before becoming a registry key
# ---------------------------------------------------------------------------

class TestUserNameSanitized:
    def test_traversal_name_is_sanitized(self, fake_registry, tmp_path):
        store, _ = fake_registry
        f = tmp_path / "m.gguf"
        f.write_bytes(b"bytes")
        mm.add_local(str(f), "../../evil", on_duplicate="register")
        # No traversal sequence survives into a registry key.
        assert "../../evil" not in store
        assert not any(("/" in k or "\\" in k or ".." in k) for k in store)
        # Something sane got registered.
        assert len(store) == 1

    def test_slash_name_is_sanitized(self, fake_registry, tmp_path):
        store, _ = fake_registry
        f = tmp_path / "m.gguf"
        f.write_bytes(b"bytes")
        mm.add_local(str(f), "a/b/c", on_duplicate="register")
        assert "a/b/c" not in store
        key = next(iter(store))
        assert "/" not in key and "\\" not in key

    def test_gguf_pull_name_is_sanitized(self, fake_registry, tmp_path, monkeypatch):
        # the #83 fix sanitized add's -n but NOT pull's -n (re-audit residual)
        store, models_dir = fake_registry
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda r, fn: None)
        monkeypatch.setattr(mm, "find_by_sha256", lambda *a, **k: [])

        def _fake_download(repo_id, filename, local_dir, **kw):
            p = Path(local_dir) / filename
            p.write_bytes(b"data")
            return str(p)

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
        ok = mm._pull_gguf_file("o/r:good.gguf", "../../evil")
        assert ok is True
        assert "../../evil" not in store
        assert not any((".." in k or "/" in k or "\\" in k) for k in store)

    def test_hf_snapshot_pull_name_is_sanitized(self, fake_registry, tmp_path, monkeypatch):
        # model_name is both the registry key AND the dest dir (MODELS_DIR/name),
        # so an unsanitized -n escaped the models folder too.
        store, models_dir = fake_registry

        def _fake_snap(repo_id, local_dir, **kw):
            d = Path(local_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text("{}", encoding="utf-8")
            return str(d)

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snap)
        ok = mm._pull_hf_snapshot("owner/repo", "../../evil")
        assert ok is True
        assert "../../evil" not in store
        assert not any((".." in k or "/" in k or "\\" in k) for k in store)
        # the dest dir stayed inside MODELS_DIR (no traversal escape)
        assert not list(models_dir.parent.glob("evil"))


# ---------------------------------------------------------------------------
# GAP-CLI-2: dest filename confined to MODELS_DIR; collision is hash-checked
# ---------------------------------------------------------------------------

class TestTraversalGuards:
    def test_url_traversal_filename_rejected(
            self, fake_registry, tmp_path, monkeypatch):
        """A URL whose last path segment contains separators/traversal (a real
        Windows escape vector once the stem is joined to MODELS_DIR) must be
        refused without writing anything outside the models folder."""
        store, models_dir = fake_registry
        get_spy = MagicMock()
        monkeypatch.setattr("requests.get", get_spy)
        monkeypatch.setattr("requests.head", MagicMock())
        monkeypatch.setattr(mm, "find_by_sha256", lambda *a, **k: [])

        # The final '/'-segment itself carries backslash traversal, which
        # _stem_from_url keeps verbatim.
        ok = mm._pull_url(r"https://host/sub\..\..\evil.gguf", "m")

        assert ok is False
        get_spy.assert_not_called()
        # Nothing escaped upward.
        assert not (tmp_path / "evil.gguf").exists()
        assert not (models_dir.parent / "evil.gguf").exists()

    def test_gguf_traversal_filename_rejected(
            self, fake_registry, tmp_path, monkeypatch):
        """o/r:../../evil.gguf must not let the HF file land outside MODELS_DIR."""
        store, models_dir = fake_registry
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda r, fn: None)
        monkeypatch.setattr(mm, "find_by_sha256", lambda *a, **k: [])

        downloaded = []
        import huggingface_hub
        monkeypatch.setattr(
            huggingface_hub, "hf_hub_download",
            lambda **kw: downloaded.append(1))

        ok = mm._pull_gguf_file("o/r:../../evil.gguf", None)

        assert ok is False
        assert downloaded == []
        assert not (tmp_path / "evil.gguf").exists()
        assert not (models_dir.parent / "evil.gguf").exists()

    def test_url_collision_with_mismatched_sha256_does_not_alias(
            self, fake_registry, tmp_path, monkeypatch):
        """An existing file with the same derived name must NOT be aliased onto
        a new registry name when the caller's --sha256 does not match its
        bytes."""
        store, models_dir = fake_registry
        existing = models_dir / "model.gguf"
        existing.write_bytes(b"unrelated-existing-bytes")
        monkeypatch.setattr(mm, "find_by_sha256", lambda *a, **k: [])

        get_spy = MagicMock()
        monkeypatch.setattr("requests.get", get_spy)
        monkeypatch.setattr("requests.head", MagicMock())

        wrong = "f" * 64
        ok = mm._pull_url(
            "https://host/model.gguf", "newname", expected_sha256=wrong)

        assert ok is False
        # The new name must not have been registered against the stale bytes.
        assert "newname" not in store
        get_spy.assert_not_called()

    def test_url_collision_with_matching_sha256_aliases(
            self, fake_registry, tmp_path, monkeypatch):
        """When --sha256 matches the existing file's bytes, treating it as the
        same file (register/alias) is correct."""
        store, models_dir = fake_registry
        body = b"existing-bytes"
        existing = models_dir / "model.gguf"
        existing.write_bytes(body)
        monkeypatch.setattr(mm, "find_by_sha256", lambda *a, **k: [])

        get_spy = MagicMock()
        monkeypatch.setattr("requests.get", get_spy)
        monkeypatch.setattr("requests.head", MagicMock())

        good = mm._sha256_file(existing)
        ok = mm._pull_url(
            "https://host/model.gguf", "newname", expected_sha256=good)

        assert ok is True
        assert "newname" in store
        get_spy.assert_not_called()
