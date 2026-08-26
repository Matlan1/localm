# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hardening tests for localm.model_manager.

Covers three behaviours:

--sha256 on a HuggingFace pull. Only _pull_url consumed it, so a mismatching
    --sha256 on an HF GGUF or snapshot pull was silently ignored. It must be
    honoured (verified against the downloaded HF file) or refused with a clear
    error - never quietly dropped.

A user-supplied -n name. It went into the registry raw (add_local used
    'name or p.stem'), so '../evil' or 'a/b' became a registry key unchanged. It
    must run through the same sanitizer sync_models_dir uses for auto-discovered
    names.

The GGUF/URL filename as a dest path. It had no traversal guard
    (o/r:../../evil.gguf or a URL ending in ../../evil.gguf), and a URL stem
    collision short-circuited to "Already downloaded" and aliased a new name onto
    whatever bytes already existed - even when the caller supplied a --sha256
    that did not match those bytes.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from localm import model_manager as mm
from localm.model_manager import pull as pull_mod


def _flat(text: str) -> str:
    """Lowercase captured console output with all whitespace collapsed.

    These messages are rendered through rich, which WRAPS them, and the wrap point
    moves with the length of values interpolated into the message - notably the
    dest PATH. So a long --basetemp (an xdist popen-gwN directory, say) can split a
    phrase like "DIFFERENT model" across a line break and defeat a plain substring
    match.

    The negative assertions are the dangerous half: "different model" NOT in out
    passes trivially once the wrap splits it, so the test goes green without
    testing anything. Collapsing whitespace makes every assertion here about the
    MESSAGE rather than about where the terminal happened to break the line.
    """
    return " ".join(text.lower().split())


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
# --sha256 is honoured (not silently ignored) on HF pulls
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
        out = _flat(capsys.readouterr().out)
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
# An HF snapshot pull preflights disk space (like _pull_gguf_file / _pull_url
# do) and does not register an interrupted snapshot as complete just because
# config.json - usually one of the smallest, earliest files - is present while
# a weight shard is still missing.
# ---------------------------------------------------------------------------

def _fake_hf_api(siblings):
    class _FakeInfo:
        pass
    info = _FakeInfo()
    info.siblings = siblings

    class _FakeHfApi:
        def __init__(self, *a, **kw):
            pass

        def model_info(self, repo_id, files_metadata=True):
            return info
    return _FakeHfApi


class TestSnapshotDiskSpaceAndCompleteness:
    def test_snapshot_pull_refuses_on_disk_space(
            self, fake_registry, monkeypatch, capsys):
        """A large HF snapshot must preflight-check free disk space and fail
        cleanly - the same 'Not enough disk space' message _pull_gguf_file and
        _pull_url produce - instead of running until the OS hits ENOSPC."""
        store, models_dir = fake_registry

        siblings = [
            SimpleNamespace(rfilename="config.json", size=100),
            SimpleNamespace(rfilename="model.safetensors", size=5_000_000_000),
        ]
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", _fake_hf_api(siblings))

        # Undo fake_registry's blanket `_check_disk_space -> True` stub so the
        # REAL check runs (and prints its real message) against a tiny free-space.
        # `mm.shutil` is the actual shutil module (re-exported for patching), so
        # patching .disk_usage on it also affects pull.py's own `shutil` binding.
        monkeypatch.setattr(mm, "_check_disk_space", pull_mod._check_disk_space)
        monkeypatch.setattr(mm.shutil, "disk_usage",
                             lambda p: SimpleNamespace(free=1_000_000))

        downloaded = []
        monkeypatch.setattr(
            huggingface_hub, "snapshot_download",
            lambda **kw: downloaded.append(1))

        ok = mm._pull_hf_snapshot("owner/repo", "big-repo")

        assert ok is False
        assert downloaded == []             # refused before any transfer started
        assert "big-repo" not in store
        out = _flat(capsys.readouterr().out)
        assert "not enough disk space" in out

    def test_snapshot_pull_retry_does_not_register_incomplete(
            self, fake_registry, monkeypatch, capsys):
        """Simulate a disk-full mid-download: config.json landed, the weight
        shard did not. A retry must detect the missing shard and re-run the
        download - never silently register the partial directory as ready."""
        store, models_dir = fake_registry

        dest = models_dir / "partial-repo"
        dest.mkdir(parents=True)
        (dest / "config.json").write_text("{}", encoding="utf-8")
        # model.safetensors intentionally absent - the interrupted shard.

        siblings = [
            SimpleNamespace(rfilename="config.json", size=2),
            SimpleNamespace(rfilename="model.safetensors", size=1000),
        ]
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", _fake_hf_api(siblings))

        downloaded = []

        def _fake_snapshot_download(repo_id, local_dir, **kw):
            d = Path(local_dir)
            (d / "model.safetensors").write_bytes(b"x" * 1000)
            downloaded.append(repo_id)
            return str(d)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download)

        ok = mm._pull_hf_snapshot("owner/repo", "partial-repo")

        assert ok is True
        assert downloaded == ["owner/repo"]   # retried instead of short-circuiting
        assert (dest / "model.safetensors").exists()
        out = _flat(capsys.readouterr().out)
        assert "already downloaded" not in out

    def test_snapshot_pull_recognizes_genuinely_complete_download(
            self, fake_registry, monkeypatch, capsys):
        """A snapshot where every listed file is present with a matching size
        IS treated as already downloaded (no unnecessary re-download)."""
        store, models_dir = fake_registry

        dest = models_dir / "full-repo"
        dest.mkdir(parents=True)
        (dest / "config.json").write_text("{}", encoding="utf-8")
        (dest / "model.safetensors").write_bytes(b"x" * 1000)

        siblings = [
            SimpleNamespace(rfilename="config.json", size=2),
            SimpleNamespace(rfilename="model.safetensors", size=1000),
        ]
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", _fake_hf_api(siblings))

        downloaded = []
        monkeypatch.setattr(
            huggingface_hub, "snapshot_download",
            lambda **kw: downloaded.append(1))

        ok = mm._pull_hf_snapshot("owner/repo", "full-repo")

        assert ok is True
        assert downloaded == []              # nothing re-downloaded
        assert store["full-repo"]["source"] == "hf:owner/repo"
        out = _flat(capsys.readouterr().out)
        assert "already downloaded" in out


# ---------------------------------------------------------------------------
# Two different repos (or one repo pulled twice under a reused --name, no
# special characters required) can compute the exact same dest: model_name
# comes from _sanitize_name, a LOSSY coercion used directly as both the
# MODELS_DIR subdirectory and the registry key. snapshot_download merges into
# whatever is already at dest rather than clearing it, so a collision has to be
# refused before it registers.
# ---------------------------------------------------------------------------

def _fake_hf_api_by_repo(files_by_repo: dict):
    """Like _fake_hf_api, but a DIFFERENT sibling list per repo_id - needed
    here because the whole point is two DIFFERENT repos."""
    class _FakeInfo:
        def __init__(self, siblings):
            self.siblings = siblings

    class _FakeHfApi:
        def __init__(self, *a, **kw):
            pass

        def model_info(self, repo_id, files_metadata=True):
            return _FakeInfo([SimpleNamespace(rfilename=name, size=len(content))
                              for name, content in files_by_repo[repo_id].items()])
    return _FakeHfApi


def _fake_snapshot_download_writing(files_by_repo: dict, downloaded: list):
    def _download(repo_id, local_dir, **kw):
        d = Path(local_dir)
        d.mkdir(parents=True, exist_ok=True)
        for name, content in files_by_repo[repo_id].items():
            (d / name).write_bytes(content)
        downloaded.append(repo_id)
        return str(d)
    return _download


class TestSnapshotDestCollision:
    REPO_A = "TheBloke/Llama-2-7B-Chat-GGUF-HF"
    REPO_B = "NousResearch/CompletelyUnrelatedModel"
    FILES = {
        REPO_A: {"config.json": b"A-CONFIG", "pytorch_model.bin": b"A-WEIGHTS"},
        REPO_B: {"config.json": b"B-CONFIG-DIFFERENT-LENGTH",
                 "pytorch_model.bin": b"B-WEIGHTS-ALSO-DIFFERENT"},
    }

    def _wire(self, monkeypatch):
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi",
                            _fake_hf_api_by_repo(self.FILES))
        downloaded: list = []
        monkeypatch.setattr(
            huggingface_hub, "snapshot_download",
            _fake_snapshot_download_writing(self.FILES, downloaded))
        return downloaded

    def test_plain_name_reuse_across_two_repos_is_refused(
            self, fake_registry, monkeypatch, capsys):
        """An ordinary user reusing a --name for a different repo weeks later -
        no special characters, no attacker, just a name they liked - must be
        refused, not silently merged."""
        store, models_dir = fake_registry
        downloaded = self._wire(monkeypatch)

        ok_a = mm._pull_hf_snapshot(self.REPO_A, "mymodel")
        assert ok_a is True
        capsys.readouterr()

        ok_b = mm._pull_hf_snapshot(self.REPO_B, "mymodel")

        assert ok_b is False
        assert downloaded == [self.REPO_A]        # B's download never ran
        assert store["mymodel"]["source"] == f"hf:{self.REPO_A}"   # A survives
        dest = models_dir / "mymodel"
        assert (dest / "pytorch_model.bin").read_bytes() == b"A-WEIGHTS"
        out = _flat(capsys.readouterr().out)
        assert "different model" in out
        assert "mymodel" in out

    def test_sanitized_default_name_collision_is_also_refused(
            self, fake_registry, monkeypatch, capsys):
        """The same defect via the OTHER trigger: two default (un-named)
        pulls whose repo-tail collapses to the same sanitized name."""
        store, models_dir = fake_registry
        repo_a, repo_b = "owner1/Llama_3(8B)", "owner2/Llama_3-8B"
        files = {
            repo_a: {"config.json": b"A", "w-a.bin": b"AAAA"},
            repo_b: {"config.json": b"BB", "w-b.bin": b"BBBB"},
        }
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", _fake_hf_api_by_repo(files))
        downloaded: list = []
        monkeypatch.setattr(huggingface_hub, "snapshot_download",
                            _fake_snapshot_download_writing(files, downloaded))

        from localm.model_manager.registry import _sanitize_name
        assert _sanitize_name(repo_a.split("/")[-1]) == \
               _sanitize_name(repo_b.split("/")[-1]), \
               "test setup broken: these must actually collide"

        assert mm._pull_hf_snapshot(repo_a, None) is True
        assert mm._pull_hf_snapshot(repo_b, None) is False
        assert downloaded == [repo_a]

    def test_redownload_of_the_same_repo_still_works(
            self, fake_registry, monkeypatch, capsys):
        """The collision check must not fire on a legitimate --redownload of
        the repo already registered at this exact dest (matching source)."""
        store, models_dir = fake_registry
        self._wire(monkeypatch)

        assert mm._pull_hf_snapshot(self.REPO_A, "mymodel") is True
        capsys.readouterr()

        ok = mm._pull_hf_snapshot(self.REPO_A, "mymodel", redownload=True)

        assert ok is True
        # redownload=True skips the "already downloaded" fast path's dedup prompt
        # branch, but the snapshot IS already complete, so it still short-circuits
        # there rather than re-downloading. What this checks is only that the
        # collision check does not refuse it.
        assert store["mymodel"]["source"] == f"hf:{self.REPO_A}"
        out = _flat(capsys.readouterr().out)
        assert "different model" not in out

    def test_resumable_partial_of_the_same_repo_still_works(
            self, fake_registry, monkeypatch):
        """An interrupted download of THIS repo (config.json landed, the
        weight file did not, never reached registration) must still resume -
        the collision check must not mistake 'nothing registered here yet'
        for a foreign occupant."""
        store, models_dir = fake_registry
        # An UNRELATED prior registration exists elsewhere, to prove the
        # check is keyed on THIS dest, not merely "is the registry non-empty".
        store["someone-else"] = {"path": str(models_dir / "someone-else"),
                                 "source": "hf:other/repo", "model_type": "llm"}
        (models_dir / "someone-else").mkdir(parents=True)

        dest = models_dir / "partial-repo"
        dest.mkdir(parents=True)
        (dest / "config.json").write_text("{}", encoding="utf-8")

        siblings = [SimpleNamespace(rfilename="config.json", size=2),
                   SimpleNamespace(rfilename="model.safetensors", size=1000)]
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", _fake_hf_api(siblings))
        downloaded: list = []

        def _resume(repo_id, local_dir, **kw):
            (Path(local_dir) / "model.safetensors").write_bytes(b"x" * 1000)
            downloaded.append(repo_id)
            return str(local_dir)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", _resume)

        ok = mm._pull_hf_snapshot("owner/repo", "partial-repo")

        assert ok is True
        assert downloaded == ["owner/repo"]
        assert store["partial-repo"]["source"] == "hf:owner/repo"

    def test_final_registration_routes_through_dedup(
            self, fake_registry, monkeypatch):
        """_pull_hf_snapshot's OWN call site for the post-download registration
        must be _register_with_dedup, not the plain _register - the same
        asymmetry _pull_gguf_file already closed for its own path.
        (_register_with_dedup legitimately calls the plain _register internally
        as its own final step once its dedup checks pass; this is about which
        function THIS CALL SITE reaches for directly, so both are replaced with
        bare recording stubs rather than letting either run for real.)"""
        store, models_dir = fake_registry
        self._wire(monkeypatch)

        called_dedup = []
        monkeypatch.setattr(mm, "_register_with_dedup",
                            lambda name, *a, **k: called_dedup.append(name) or True)
        called_plain = []
        monkeypatch.setattr(mm, "_register",
                            lambda name, *a, **k: called_plain.append(name))

        ok = mm._pull_hf_snapshot(self.REPO_A, "mymodel")

        assert ok is True
        assert called_dedup == ["mymodel"]
        assert called_plain == []

    def test_name_collision_at_a_different_path_is_reported_honestly(
            self, fake_registry, monkeypatch, capsys):
        """_register_with_dedup returns False for a REAL name conflict that
        is orthogonal to the dest-collision check above: 'mymodel' is already
        registered pointing at some OTHER path, and this dest ('mymodel')
        does not exist yet, so the new pre-download collision block never
        fires. The download still runs and writes real bytes to disk - but
        registration is then correctly refused (non-interactive, different
        file under the same name). The call site must not paper over that:
        it must not print a green checkmark / return True for a download
        that was never actually registered under the requested name."""
        store, models_dir = fake_registry
        downloaded = self._wire(monkeypatch)

        elsewhere = models_dir / "elsewhere"
        elsewhere.mkdir(parents=True)
        store["mymodel"] = {"path": str(elsewhere), "source": "hf:owner/old",
                            "model_type": "llm"}

        ok = mm._pull_hf_snapshot(self.REPO_A, "mymodel")

        assert downloaded == [self.REPO_A]          # the download DID run
        assert ok is False                          # but registration was refused
        assert store["mymodel"]["source"] == "hf:owner/old"   # untouched
        out = _flat(capsys.readouterr().out)
        assert "could not be registered" in out
        assert "✓" not in out                  # no false success checkmark


# ---------------------------------------------------------------------------
# user -n name is sanitized before becoming a registry key
# ---------------------------------------------------------------------------

class TestUserNameSanitized:
    @pytest.mark.parametrize("bad_name", ["../../evil", "a/b/c"])
    def test_add_local_name_is_sanitized(self, fake_registry, tmp_path, bad_name):
        store, _ = fake_registry
        f = tmp_path / "m.gguf"
        f.write_bytes(b"bytes")
        mm.add_local(str(f), bad_name, on_duplicate="register")
        # No traversal or path-separator sequence survives into a registry key.
        assert bad_name not in store
        assert not any(("/" in k or "\\" in k or ".." in k) for k in store)
        # Something sane got registered.
        assert len(store) == 1

    def test_gguf_pull_name_is_sanitized(self, fake_registry, tmp_path, monkeypatch):
        # pull's -n is sanitized as well as add's
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
        # so an unsanitized -n would escape the models folder too.
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
# dest filename confined to MODELS_DIR; collision is hash-checked
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


# ---------------------------------------------------------------------------
# The full-snapshot pull's model_type resolution and its disk preflight both
# account for what is actually on disk.
# ---------------------------------------------------------------------------

_LLAMA_CONFIG = '{"architectures": ["LlamaForCausalLM"], "model_type": "llama"}'


def _default_writer(d: Path):
    (d / "config.json").write_text(_LLAMA_CONFIG, encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"x" * 1000)


def _snapshot_env(monkeypatch, siblings, writer=None):
    """Fake HF API + a snapshot_download that writes REAL files into dest, so the
    type fallback below reads a real config.json rather than a mock of it."""
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", _fake_hf_api(siblings))

    def _fake_download(repo_id, local_dir, **kw):
        d = Path(local_dir)
        d.mkdir(parents=True, exist_ok=True)
        (writer or _default_writer)(d)
        return str(d)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_download)


_LLM_SIBLINGS = [
    SimpleNamespace(rfilename="config.json", size=len(_LLAMA_CONFIG)),
    SimpleNamespace(rfilename="model.safetensors", size=1000),
]


class TestSnapshotTypeResolution:
    def test_tagless_transformers_llm_registers_as_llm_not_unknown(
            self, fake_registry, monkeypatch):
        """Many base and older HF repos carry no exact pipeline_tag, so the
        pre-download probe returns 'unknown'. The snapshot must then read the
        config.json it just downloaded rather than registering 'unknown', or a
        clearly-LlamaForCausalLM model vanishes from GUI auto-select, the MCP
        EngineCache and the jobs runner."""
        store, _ = fake_registry
        _snapshot_env(monkeypatch, _LLM_SIBLINGS)

        assert mm._pull_hf_snapshot("owner/repo", "tagless", model_type="unknown")

        assert store["tagless"]["model_type"] == "llm", (
            "a downloaded LlamaForCausalLM repo must register as an LLM")

    def test_config_without_a_causal_arch_stays_unknown(
            self, fake_registry, monkeypatch):
        """Negative case: the fallback reads HARD metadata only. A repo whose
        config.json shows no causal-LM architecture stays 'unknown', never a
        silent 'llm'."""
        store, _ = fake_registry

        def _writer(d: Path):
            (d / "config.json").write_text(
                '{"architectures": ["BertModel"]}', encoding="utf-8")
            (d / "model.safetensors").write_bytes(b"x" * 1000)

        _snapshot_env(monkeypatch, [
            SimpleNamespace(rfilename="config.json", size=None),
            SimpleNamespace(rfilename="model.safetensors", size=1000),
        ], writer=_writer)

        assert mm._pull_hf_snapshot("owner/bert", "berty", model_type="unknown")
        assert store["berty"]["model_type"] == "unknown"

    def test_a_resolved_probe_type_is_never_overridden(
            self, fake_registry, monkeypatch):
        """Negative case: when the HF probe DID resolve a type from hard metadata
        (lora/vae/embedding/...) that stays authoritative - the config.json
        fallback only answers what the probe could not."""
        store, _ = fake_registry
        _snapshot_env(monkeypatch, _LLM_SIBLINGS)      # config.json says causal LM

        assert mm._pull_hf_snapshot("owner/adapter", "adapt", model_type="lora")
        assert store["adapt"]["model_type"] == "lora"

    def test_already_downloaded_snapshot_also_resolves_its_type(
            self, fake_registry, monkeypatch):
        """The early 'Already downloaded' branch registers too, so it needs the
        same resolution - otherwise a re-pull re-stamps 'unknown'."""
        store, models_dir = fake_registry
        dest = models_dir / "tagless"
        dest.mkdir(parents=True)
        _default_writer(dest)
        _snapshot_env(monkeypatch, _LLM_SIBLINGS)

        assert mm._pull_hf_snapshot("owner/repo", "tagless", model_type="unknown")
        assert store["tagless"]["model_type"] == "llm"


class TestSnapshotResumePreflight:
    def test_retry_counts_only_the_bytes_still_missing(
            self, fake_registry, monkeypatch, capsys):
        """A disk-full pull leaves a partial dest. Once the user frees enough for
        the REMAINING bytes the retry must proceed: a preflight that demands the
        FULL repo size, ignoring what is already on disk, could never resume.
        _pull_gguf_file sums only the missing parts and _pull_url uses
        (total - already_have); this path must do the same."""
        store, models_dir = fake_registry
        dest = models_dir / "partial"
        dest.mkdir(parents=True)
        # 6000 of ~10000 bytes already on disk; ~4000 still to fetch.
        (dest / "config.json").write_text(_LLAMA_CONFIG, encoding="utf-8")
        (dest / "shard-a.bin").write_bytes(b"x" * 6000)

        siblings = [
            SimpleNamespace(rfilename="config.json", size=len(_LLAMA_CONFIG)),
            SimpleNamespace(rfilename="shard-a.bin", size=6000),
            SimpleNamespace(rfilename="shard-b.bin", size=4000),      # missing
        ]
        _snapshot_env(monkeypatch, siblings,
                      writer=lambda d: (d / "shard-b.bin").write_bytes(b"x" * 4000))
        # The REAL check, against 5000 free: less than the 10000 total, more than
        # the ~4000 actually still needed.
        monkeypatch.setattr(mm, "_check_disk_space", pull_mod._check_disk_space)
        monkeypatch.setattr(mm.shutil, "disk_usage",
                            lambda p: SimpleNamespace(free=5000))

        ok = mm._pull_hf_snapshot("owner/repo", "partial", model_type="unknown")

        assert ok is True, ("the retry must resume: there is room for the "
                            "remaining bytes\n" + capsys.readouterr().out)
        assert (dest / "shard-b.bin").exists()

    def test_a_genuinely_full_disk_still_refuses(
            self, fake_registry, monkeypatch, capsys):
        """Negative case: subtracting the on-disk bytes must not disable the
        check. 100 bytes free and ~4000 still needed must still refuse."""
        store, models_dir = fake_registry
        dest = models_dir / "partial"
        dest.mkdir(parents=True)
        (dest / "config.json").write_text(_LLAMA_CONFIG, encoding="utf-8")
        (dest / "shard-a.bin").write_bytes(b"x" * 6000)

        siblings = [
            SimpleNamespace(rfilename="config.json", size=len(_LLAMA_CONFIG)),
            SimpleNamespace(rfilename="shard-a.bin", size=6000),
            SimpleNamespace(rfilename="shard-b.bin", size=4000),
        ]
        downloaded = []
        _snapshot_env(monkeypatch, siblings, writer=lambda d: downloaded.append(1))
        monkeypatch.setattr(mm, "_check_disk_space", pull_mod._check_disk_space)
        monkeypatch.setattr(mm.shutil, "disk_usage",
                            lambda p: SimpleNamespace(free=100))

        ok = mm._pull_hf_snapshot("owner/repo", "partial", model_type="unknown")

        assert ok is False
        assert downloaded == [], "nothing may transfer when the disk is really full"
        assert "not enough disk space" in _flat(capsys.readouterr().out)
