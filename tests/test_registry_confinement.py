# SPDX-License-Identifier: AGPL-3.0-or-later
"""Registry poisoning: registration authorisation + registry path integrity.

registry.json's stored paths are read back by ~40 stat/glob sites across the GUI,
the /v1 API and the MCP server. Those reads are only safe if the file can only be
written by a principal that already holds host filesystem reach. Two doors used to
bypass require_fs_host, and three registry.py defects rode along:

  (a) POST /api/models/scan gated on `if workdir:`, so a BODYLESS post scanned
      get_comfy_workdir() with no fs_access check at all - and comfy_workdir is
      settable by any config:write key.
  (b) POST /api/models/pull was MODELS_WRITE-only and forwarded `spec` verbatim;
      pull.py registers an existing local path IN PLACE via add_local(store=None).
  (c) _resolve_ollama_manifest joined a remote-authored manifest's `digest`
      straight into a path.
  (d) remove_model's delete gate was LEXICAL (is_relative_to does not resolve
      '..') with a `startswith` fallback that prefix-matched a sibling directory,
      immediately in front of shutil.rmtree / unlink.
  (e) _snapshot_is_complete joined a remote HF listing's `rfilename` onto dest.

WHY EVERY TEST HERE MINTS A KEY: effective_fs_access() returns "host" for EVERY
caller when no key is configured (open/dev mode is the trusted loopback owner).
A fixture that skips create_key would therefore pass VACUOUSLY - every assertion
would hold for the wrong reason. `keyed_app` mints one and asserts the resulting
level really is "none" before any test runs on it.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import scopes as S


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway LOCALM_HOME with config/registry redirected into it."""
    h = tmp_path / ".localm"
    (h / "models").mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", h)
    monkeypatch.setattr(_cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", h / "registry.json")
    return h


@pytest.fixture
def app(home):
    from localm.plugins.engine import attach_engine
    from localm.plugins.gui.web import attach_gui
    a = FastAPI()
    attach_engine(a)
    attach_gui(a, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda name: None,
               active_model=lambda: None)
    return a


@pytest.fixture
def restricted_key(home):
    """The exact principal the triage names: everything it needs to drive both
    routes, and NO host filesystem reach. This is the configuration
    require_fs_host exists to constrain."""
    from localm import auth
    created = auth.create_key(
        "restricted", [S.CONFIG_WRITE, S.MODELS_WRITE, S.MODELS_READ],
        fs_access="none")
    assert created["fs_access"] == "none", "fixture must not grant host fs reach"
    return created["key"]


def _hdr(key):
    return {"Authorization": f"Bearer {key}"}


def _registry_bytes(home):
    f = home / "registry.json"
    return f.read_bytes() if f.exists() else b""


@pytest.fixture
def captured_pull(monkeypatch):
    """Stub JobManager.start_cli so an ALLOWED pull captures its argv instead of
    spawning a real `localm pull` subprocess that would hit the network."""
    captured = {}

    class _FakeJob:
        id = "job-test"

    def fake_start_cli(self, kind, cli_args, **kw):
        captured["args"] = list(cli_args)
        return _FakeJob()

    monkeypatch.setattr("localm.plugins.gui.jobs.JobManager.start_cli", fake_start_cli)
    return captured


# --------------------------------------------------------------------------- #
#  (a) + (b): the two registration doors                                        #
# --------------------------------------------------------------------------- #

class TestRegistrationRequiresHostFsAccess:
    def test_the_fixture_key_really_has_no_host_fs_reach(self, app, tmp_path,
                                                          restricted_key):
        """Guards the vacuous-pass trap named in the module docstring: prove this
        key is refused by the CANONICAL require_fs_host route, so every 403 below
        is the fs gate doing its job rather than a scope error or a typo."""
        with TestClient(app) as c:
            r = c.get("/api/fs/dirs", params={"path": str(tmp_path)},
                      headers=_hdr(restricted_key))
        assert r.status_code == 403, r.text

    def test_an_unkeyed_client_would_pass_vacuously(self, app, tmp_path):
        """The control for the test above: with NO key configured every caller is
        the trusted loopback owner, so an unkeyed fixture would make every
        assertion in this file hold for the wrong reason. Fails loudly if that
        stops being true, which is what would silently gut this file."""
        with TestClient(app) as c:
            r = c.get("/api/fs/dirs", params={"path": str(tmp_path)})
        assert r.status_code == 200, r.text

    def test_bodyless_scan_is_403_and_leaves_the_registry_untouched(
            self, app, home, restricted_key):
        before = _registry_bytes(home)
        with TestClient(app) as c:
            r = c.post("/api/models/scan", headers=_hdr(restricted_key))
        assert r.status_code == 403, r.text
        assert _registry_bytes(home) == before

    def test_pull_of_a_local_path_is_403_and_leaves_the_registry_untouched(
            self, app, home, tmp_path, restricted_key):
        victim = tmp_path / "x.gguf"
        before = _registry_bytes(home)
        with TestClient(app) as c:
            r = c.post("/api/models/pull", headers=_hdr(restricted_key),
                       json={"spec": str(victim)})
        assert r.status_code == 403, r.text
        assert _registry_bytes(home) == before

    def test_pull_of_an_EXISTING_local_path_is_403(self, app, tmp_path, restricted_key):
        """The exploitable shape: a real file, which pull.py's is_local_path
        branch would register in place."""
        victim = tmp_path / "real.gguf"
        victim.write_bytes(b"GGUF")
        with TestClient(app) as c:
            r = c.post("/api/models/pull", headers=_hdr(restricted_key),
                       json={"spec": str(victim)})
        assert r.status_code == 403, r.text

    def test_pull_of_a_unc_path_is_403_without_ever_stat_ing_it(
            self, app, restricted_key):
        """A UNC spec must be classified TEXTUALLY. Any stat/resolve on it would
        block in the Windows SMB redirector (minutes, on an unroutable host) and
        draw an outbound authentication attempt from the server process, so the
        classification has to happen before the filesystem is touched. The client
        is built OUTSIDE the patch so only the request is instrumented."""
        with TestClient(app) as c:
            with patch.object(Path, "is_file",
                              side_effect=AssertionError("stat'ed a UNC spec")):
                r = c.post("/api/models/pull", headers=_hdr(restricted_key),
                           json={"spec": r"\\192.0.2.1\share\evil.gguf"})
        assert r.status_code == 403, r.text

    def test_a_remote_spec_is_still_allowed_for_the_same_key(
            self, app, restricted_key, captured_pull):
        """The gate must not become 'models:write can no longer pull'. An ordinary
        HuggingFace spec still starts a job for this exact fs_access='none' key."""
        with TestClient(app) as c:
            r = c.post("/api/models/pull", headers=_hdr(restricted_key),
                       json={"spec": "owner/repo"})
        assert r.status_code == 200, r.text
        assert captured_pull["args"] == ["pull", "--", "owner/repo"]

    def test_a_host_fs_key_may_still_pull_a_local_path(
            self, app, tmp_path, home, captured_pull):
        """The capability is re-gated, not removed."""
        from localm import auth
        key = auth.create_key("hostwriter", [S.MODELS_WRITE], fs_access="host")["key"]
        victim = tmp_path / "real.gguf"
        victim.write_bytes(b"GGUF")
        with TestClient(app) as c:
            r = c.post("/api/models/pull", headers=_hdr(key),
                       json={"spec": str(victim)})
        assert r.status_code == 200, r.text
        assert captured_pull["args"] == ["pull", "--", str(victim)]


class TestMediaWorkdirIsAdminGated:
    """get_comfy_workdir() prefers the per-plugin block over the global key, so
    the per-plugin write surface is the same door by another name."""

    def test_non_owner_config_write_cannot_set_a_media_workdir(
            self, app, tmp_path, restricted_key):
        with TestClient(app) as c:
            r = c.post("/v1/media/config/image", headers=_hdr(restricted_key),
                       json={"workdir": str(tmp_path)})
        assert r.status_code == 403, r.text

    def test_the_owner_key_still_can(self, app, tmp_path, home, monkeypatch):
        monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
        with TestClient(app) as c:
            r = c.post("/v1/media/config/image", headers=_hdr("ownersecret"),
                       json={"workdir": str(tmp_path)})
        assert r.status_code == 200, r.text


# --------------------------------------------------------------------------- #
#  (c) the Ollama manifest digest                                              #
# --------------------------------------------------------------------------- #

def _ollama_tree(root: Path, digest: str) -> Path:
    """<root>/manifests/r/o/m/<tag> holding one model layer, plus <root>/blobs."""
    manifest_dir = root / "manifests" / "reg" / "owner" / "model"
    manifest_dir.mkdir(parents=True)
    (root / "blobs").mkdir()
    (manifest_dir / "Q8_0").write_text(json.dumps({
        "layers": [{"mediaType": "application/vnd.ollama.image.model",
                    "digest": digest}]
    }))
    return manifest_dir


class TestOllamaManifestDigestIsConfined:
    def test_a_traversing_digest_returns_none_and_stays_inside_blobs(self, tmp_path):
        from localm.model_manager import registry as reg
        root = tmp_path / "ollama"
        manifest_dir = _ollama_tree(root, "../../../../etc/passwd")
        # Plant the file the traversal would find, so a successful escape is
        # DETECTABLE: without the fix the walk-up loop finds it and returns it.
        outside = tmp_path / "etc"
        outside.mkdir()
        (outside / "passwd").write_bytes(b"root:x:0:0")

        touched = []
        real_exists = Path.exists

        def _spy(self):
            touched.append(Path(self))
            return real_exists(self)

        with patch.object(Path, "exists", _spy):
            assert reg._resolve_ollama_manifest(manifest_dir) is None

        blobs = (root / "blobs").resolve()
        for p in touched:
            try:
                resolved = p.resolve()
            except OSError:
                continue
            assert blobs not in resolved.parents or resolved.name.startswith("sha256-"), \
                f"probed a non-blob path under blobs: {p}"
            assert "passwd" not in str(resolved), f"escaped to {p}"

    @pytest.mark.parametrize("digest", [
        "sha256:../../evil",
        "../../evil",
        "sha256-NOTHEX" + "0" * 58,
        "sha256-" + "a" * 63,          # one hex digit short
        "sha256-" + "a" * 65,          # one too many
        "",
    ])
    def test_malformed_digests_are_all_rejected(self, tmp_path, digest):
        from localm.model_manager import registry as reg
        root = tmp_path / "ollama"
        manifest_dir = _ollama_tree(root, digest)
        assert reg._resolve_ollama_manifest(manifest_dir) is None

    @pytest.mark.parametrize("layers", [
        "notalist", 123, None, [None], ["str"], [{"mediaType": "application/vnd.ollama.image.model"}],
        [{"mediaType": "application/vnd.ollama.image.model", "digest": 7}],
    ])
    def test_malformed_remote_json_never_raises(self, tmp_path, layers):
        """A manifest is remote-authored: no shape may be assumed. Each of these
        used to be a TypeError / AttributeError / KeyError escaping the caller."""
        from localm.model_manager import registry as reg
        manifest_dir = tmp_path / "manifests" / "reg" / "owner" / "model"
        manifest_dir.mkdir(parents=True)
        (tmp_path / "blobs").mkdir()
        (manifest_dir / "Q8_0").write_text(json.dumps({"layers": layers}))
        assert reg._resolve_ollama_manifest(manifest_dir) is None

    def test_a_well_formed_digest_still_resolves(self, tmp_path):
        """The confinement must not break the feature it guards."""
        from localm.model_manager import registry as reg
        root = tmp_path / "ollama"
        digest = "sha256:" + "ab" * 32
        manifest_dir = _ollama_tree(root, digest)
        blob = root / "blobs" / ("sha256-" + "ab" * 32)
        blob.write_bytes(b"GGUF")
        got = reg._resolve_ollama_manifest(manifest_dir)
        assert got is not None
        assert got[0] == blob


# --------------------------------------------------------------------------- #
#  (d) the remove_model delete gate                                            #
# --------------------------------------------------------------------------- #

class TestRemoveModelDeleteGate:
    def test_a_traversing_entry_does_not_delete_outside_models_dir(self, home, monkeypatch):
        """`<models>/../../victim` passed the OLD lexical is_relative_to test and
        was rmtree'd. Two independent layers now stop it - _entry_path reads a
        '..' entry as malformed, and the delete gate resolves before comparing -
        so this asserts the OUTCOME rather than which layer fired. The sibling
        test below covers the gate on its own, with a path holding no '..' at
        all."""
        import localm.model_manager as _mm
        from localm.model_manager import registry as reg
        monkeypatch.setattr(_mm, "HOME_DIR", home)
        monkeypatch.setattr(_mm, "MODELS_DIR", home / "models")

        victim = home.parent / "victim"
        victim.mkdir()
        (victim / "keepme.txt").write_text("do not delete")
        poisoned = str(home / "models" / ".." / ".." / "victim")

        _mm.save_registry({"evil": {"path": poisoned, "source": "local",
                                    "model_type": "llm"}})
        reg.remove_model("evil")

        assert victim.exists(), "remove_model deleted outside <data dir>/models"
        assert (victim / "keepme.txt").read_text() == "do not delete"
        assert "evil" not in _mm.load_registry(), "the name should still be dropped"

    def test_a_sibling_prefix_directory_is_not_deleted(self, home, monkeypatch):
        """`<data dir>/models-old` string-prefix-matched `<data dir>/models` in the
        old startswith fallback."""
        import localm.model_manager as _mm
        from localm.model_manager import registry as reg
        monkeypatch.setattr(_mm, "HOME_DIR", home)
        monkeypatch.setattr(_mm, "MODELS_DIR", home / "models")

        sibling = home / "models-old"
        sibling.mkdir()
        target = sibling / "m.gguf"
        target.write_bytes(b"GGUF")

        _mm.save_registry({"old": {"path": str(target), "source": "local",
                                   "model_type": "llm"}})
        reg.remove_model("old")

        assert target.exists(), "deleted a file in a sibling prefix directory"
        assert "old" not in _mm.load_registry()

    def test_the_models_dir_itself_is_never_deleted(self, home, monkeypatch):
        import localm.model_manager as _mm
        from localm.model_manager import registry as reg
        monkeypatch.setattr(_mm, "HOME_DIR", home)
        monkeypatch.setattr(_mm, "MODELS_DIR", home / "models")
        (home / "models" / "sentinel").write_bytes(b"x")

        _mm.save_registry({"root": {"path": str(home / "models"), "source": "local",
                                    "model_type": "llm"}})
        reg.remove_model("root")

        assert (home / "models" / "sentinel").exists()

    def test_a_genuinely_owned_file_is_still_deleted(self, home, monkeypatch):
        """The gate must not turn `localm rm` into a no-op for real models."""
        import localm.model_manager as _mm
        from localm.model_manager import registry as reg
        monkeypatch.setattr(_mm, "HOME_DIR", home)
        monkeypatch.setattr(_mm, "MODELS_DIR", home / "models")

        owned = home / "models" / "mine.gguf"
        owned.write_bytes(b"GGUF")
        _mm.save_registry({"mine": {"path": str(owned), "source": "local",
                                    "model_type": "llm"}})
        reg.remove_model("mine")

        assert not owned.exists(), "an owned model under <data dir>/models must be deleted"
        assert "mine" not in _mm.load_registry()


# --------------------------------------------------------------------------- #
#  (e) _entry_path and the remote HF file listing                              #
# --------------------------------------------------------------------------- #

class TestEntryPathRejectsTraversal:
    @pytest.mark.parametrize("stored", [
        "/models/../../etc/passwd",
        r"C:\models\..\..\Windows\win.ini",
        "..",
        "a/../b",
        r"a\..\b",
    ])
    def test_a_dotdot_path_reads_as_malformed(self, stored):
        from localm.model_manager import _entry_path
        assert _entry_path({"path": stored}) is None

    @pytest.mark.parametrize("stored", [
        "/home/u/.localm/models/m.gguf",
        r"C:\Users\u\.localm\models\m.gguf",
        "m.gguf",
        "..hidden/m.gguf",       # NOT a '..' component
        "a..b/m.gguf",
    ])
    def test_ordinary_paths_are_unaffected(self, stored):
        from localm.model_manager import _entry_path
        assert _entry_path({"path": stored}) == stored


class TestSnapshotCompletenessConfinesRemoteFilenames:
    @pytest.mark.parametrize("rfilename", [
        "C:/Windows/win.ini",
        "/etc/passwd",
        "../../../etc/passwd",
        r"\\192.0.2.1\share\x",
        "C:evil",
    ])
    def test_an_escaping_rfilename_is_rejected_before_the_stat(self, tmp_path, rfilename):
        from localm import pathsafe
        with pytest.raises(ValueError):
            pathsafe.confined_under(tmp_path, rfilename)

    def test_nested_subpaths_are_still_permitted(self, tmp_path):
        """A real HF listing uses them, so confined_name's flat-only rule is wrong
        here and confined_under must allow this."""
        from localm import pathsafe
        got = pathsafe.confined_under(tmp_path, "subdir/model-00001-of-2.safetensors")
        assert got == tmp_path / "subdir" / "model-00001-of-2.safetensors"

    def test_snapshot_is_incomplete_rather_than_stat_ing_outside_dest(self, tmp_path):
        """End to end through pull.py: a repo whose listing escapes must make the
        snapshot read INCOMPLETE (re-download), never register a half-present tree
        by matching a file that lives elsewhere on disk."""
        from localm.model_manager import pull as _pull

        dest = tmp_path / "snap"
        dest.mkdir()
        (dest / "config.json").write_text("{}")
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"1234")

        class _Sib:
            def __init__(self, rfilename, size):
                self.rfilename, self.size = rfilename, size

        # The escaping name points at a file that DOES exist with a matching size,
        # so an unconfined check would report "complete".
        siblings = [_Sib("config.json", 2), _Sib(f"../{outside.name}", 4)]
        assert _pull._snapshot_is_complete(dest, siblings, "owner/repo") is False

    def test_a_well_formed_listing_still_reports_complete(self, tmp_path):
        """The confinement must not break the completeness check it guards."""
        from localm.model_manager import pull as _pull

        dest = tmp_path / "snap"
        (dest / "sub").mkdir(parents=True)
        (dest / "config.json").write_text("{}")
        (dest / "sub" / "w.safetensors").write_bytes(b"12345")

        class _Sib:
            def __init__(self, rfilename, size):
                self.rfilename, self.size = rfilename, size

        siblings = [_Sib("config.json", 2), _Sib("sub/w.safetensors", 5)]
        assert _pull._snapshot_is_complete(dest, siblings, "owner/repo") is True
