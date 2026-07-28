# SPDX-License-Identifier: AGPL-3.0-or-later
"""Registry poisoning: registration authorisation + registry path integrity.

registry.json's stored paths are read back by ~40 stat/glob sites across the GUI,
the /v1 API and the MCP server. Those reads are only safe if the file can only be
written by a principal that already holds host filesystem reach. Two doors used to
bypass require_fs_host, and three path-integrity defects rode along:

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
would hold for the wrong reason. `restricted_key` mints one and asserts its
recorded level really is "none", and the first two tests in the file prove it end
to end: the key is refused by the canonical require_fs_host route, and an UNKEYED
client on the same route is not.
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
    # allow_privileged=True is REQUIRED, not incidental: config:write is in
    # scopes.PRIVILEGED_SCOPES, so create_key raises PermissionError without it
    # (auth.py:452-459) and every test requesting this fixture would ERROR at
    # setup rather than run. That is what an owner does when deliberately minting
    # such a key, and minting it is the whole point here - the threat model IS a
    # key that legitimately holds config:write and models:write while holding no
    # filesystem reach at all.
    created = auth.create_key(
        "restricted", [S.CONFIG_WRITE, S.MODELS_WRITE, S.MODELS_READ],
        allow_privileged=True, fs_access="none")
    # Assert the key really came out the way the threat model needs. Without
    # these, a future change to create_key's defaults could silently hand back a
    # key with different reach and every 403 below would still pass, for the
    # wrong reason.
    assert created["fs_access"] == "none", "fixture must not grant host fs reach"
    assert S.CONFIG_WRITE in created["scopes"], created["scopes"]
    assert S.MODELS_WRITE in created["scopes"], created["scopes"]
    assert S.ADMIN not in created["scopes"], "fixture must not be owner-equivalent"
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


# NOT TESTED HERE ON PURPOSE: the per-plugin `comfy.workdir` admin gate in
# POST /v1/media/config/{name}. get_comfy_workdir() prefers the per-plugin block
# over the global comfy_workdir, so that IS the same door by another name - but
# localm/inference/routes/config.py has a single designated owner for this
# program (three lanes were adding admin_only-style flags to it at once, and a
# rebase taking one side wholesale silently drops another lane's flag with no
# conflict marker and no failing test). The one-line gate change was handed to
# that owner and lands in their diff, with the end-to-end non-ADMIN -> 403
# assertion alongside it. The scan-route fix above closes the registry-poisoning
# door independently of it.


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

class TestPublicSurfaceSurvivesARebase:
    """localm/model_manager/__init__.py is a pure re-export surface, and two
    concurrent branches are inserting names into the SAME two blocks. A bad
    conflict resolution in an export list drops a name silently - there is no
    conflict marker left behind and nothing fails until an unrelated module
    raises ImportError at runtime. Assert the name is reachable BOTH ways, so a
    dropped line fails here instead."""

    def test_is_owned_model_path_is_exported(self):
        import localm.model_manager as mm
        from localm.model_manager import is_owned_model_path  # noqa: F401
        assert "is_owned_model_path" in mm.__all__
        assert callable(mm.is_owned_model_path)

    def test_a_registry_key_cannot_be_path_shaped(self):
        """Guards the assumption that makes the poisoned-row fallthrough in
        get_model_info harmless. When an entry is malformed, get_model_info falls
        through to Path(<registry KEY>); that is safe only while keys are model
        NAMES and never path strings. _sanitize_name is the single chokepoint
        enforcing it for every registration route, so pin its behavior here - if
        it ever stops stripping separators, the fallthrough becomes a live
        resolve of an attacker-chosen path for a CLI caller."""
        from localm.model_manager import _sanitize_name
        for hostile in ("D:/evil/x.gguf", r"C:\evil\x.gguf", "../../evil",
                        "/etc/passwd", r"\\server\share\x", ".."):
            out = _sanitize_name(hostile)
            assert "/" not in out and "\\" not in out and ":" not in out, out
            assert ".." not in out, out
            assert out not in (".", "..", ""), out


class TestIsOwnedModelPathIsTheSingleDefinition:
    """Three hand-rolled variants of "is this file localm's to delete" had drifted
    apart in the repo, two of them wrong in a way that reaches shutil.rmtree. The
    predicate now lives in ONE place; these tests pin its behavior directly, so a
    caller that stops using it cannot quietly reintroduce a weaker copy."""

    def test_it_rejects_what_the_lexical_variant_accepted(self, home, monkeypatch):
        """`is_relative_to` does not normalise '..' - the traversal below tested
        True under it while resolving outside the models folder."""
        import localm.model_manager as _mm
        from localm.model_manager import is_owned_model_path
        monkeypatch.setattr(_mm, "MODELS_DIR", home / "models")
        evil = home / "models" / ".." / "models-old" / "x.gguf"
        assert evil.is_relative_to(home / "models"), \
            "precondition: the OLD lexical test must accept this, else the case is moot"
        assert is_owned_model_path(evil) is False

    def test_it_rejects_what_the_startswith_variant_accepted(self, home, monkeypatch):
        """A raw string prefix matches a SIBLING directory."""
        import localm.model_manager as _mm
        from localm.model_manager import is_owned_model_path
        monkeypatch.setattr(_mm, "MODELS_DIR", home / "models")
        sibling = home / "models-old" / "x.gguf"
        assert str(sibling).startswith(str(home / "models")), \
            "precondition: the OLD startswith test must accept this"
        assert is_owned_model_path(sibling) is False

    def test_the_models_root_itself_is_not_owned(self, home, monkeypatch):
        import localm.model_manager as _mm
        from localm.model_manager import is_owned_model_path
        monkeypatch.setattr(_mm, "MODELS_DIR", home / "models")
        assert is_owned_model_path(home / "models") is False

    def test_a_real_managed_model_is_owned(self, home, monkeypatch):
        """The fires-control: the predicate must still say YES to the thing it
        exists to permit, or every assertion above passes for free."""
        import localm.model_manager as _mm
        from localm.model_manager import is_owned_model_path
        monkeypatch.setattr(_mm, "MODELS_DIR", home / "models")
        mine = home / "models" / "mine.gguf"
        mine.write_bytes(b"GGUF")
        assert is_owned_model_path(mine) is True
        assert is_owned_model_path(home / "models" / "sub" / "deep.gguf") is True

    def test_remove_model_routes_through_it(self, home, monkeypatch):
        """Pins the WIRING, not just the predicate: if remove_model stops calling
        the shared definition, this fails even though the predicate is still
        correct on its own."""
        import localm.model_manager as _mm
        from localm.model_manager import registry as reg
        monkeypatch.setattr(_mm, "MODELS_DIR", home / "models")
        seen = []
        real = reg.is_owned_model_path
        monkeypatch.setattr(reg, "is_owned_model_path",
                            lambda p, root=None: seen.append(Path(p)) or real(p, root))
        owned = home / "models" / "mine.gguf"
        owned.write_bytes(b"GGUF")
        _mm.save_registry({"mine": {"path": str(owned), "source": "local",
                                    "model_type": "llm"}})
        reg.remove_model("mine")
        assert owned in seen, "remove_model must decide ownership via the shared helper"


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
        # A drive letter on a NESTED component. pathlib only parses a drive at
        # position 0, so this survives an is_absolute/.drive check and a
        # containment check both - joinpath treats it as drive-relative, and when
        # it matches the base's drive the "C:" is silently DROPPED, handing back a
        # path that names a different file than was requested. Measured, and the
        # reason the rejection is per-component rather than per-string.
        "a/C:evil",
        "sub/dir/C:evil",
    ])
    def test_an_escaping_rfilename_is_rejected_before_the_stat(self, tmp_path, rfilename):
        from localm import _pathcheck
        with pytest.raises(ValueError):
            _pathcheck.confined_under(tmp_path, rfilename)

    def test_nested_subpaths_are_still_permitted(self, tmp_path):
        """A real HF listing uses them, so confined_name's flat-only rule is wrong
        here and confined_under must allow this."""
        from localm import _pathcheck
        got = _pathcheck.confined_under(tmp_path, "subdir/model-00001-of-2.safetensors")
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
