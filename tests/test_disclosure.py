# SPDX-License-Identifier: AGPL-3.0-or-later
"""WS9: what a lower-privileged or anonymous reader is allowed to learn.

Four disclosure defects, each with its own class below:

* ``/debug/stacks`` (CodeQL 97) is fs-host gated, but in DEFAULT KEYLESS mode
  ``effective_fs_access`` returns "host" for everyone, so that gate is a
  TAUTOLOGY; and the open-mode shell-token gate only covers ``/api/`` and
  ``/v1/`` prefixes, so the route was one of only two fully unauthenticated
  reads. It emits ``traceback.format_stack`` (absolute paths -> the OS username
  and the install dir, plus source lines) and thread names naming the installed
  plugins.
* ``GET /api/rag/embedding`` (CodeQL 98) returned ``last_error()`` /
  ``gpu_fallback_reason()`` verbatim, and those carry
  "failed to load embedding model: <absolute path>" across the isolated-child
  boundary - the data dir, hence the OS username, to a rag-scoped caller.
* ``POST /api/rag/collections/{name}/add`` (CodeQL 59) ran the ``p.exists()``
  sweep BEFORE ``confine_index_path`` and echoed the expanded path in the 400,
  a path-existence oracle for any absolute path.
* ``sessions.json`` / ``jobs.json`` (CodeQL 88) store the SHA-256 digest of API
  keys, while only ``auth.key`` (the PLAINTEXT) got the Windows icacls
  treatment - so the digest was readable where the plaintext was not.

Rule 5 runs through all of it: every assertion below that checks a path is gone
is paired with one checking the CAUSE survived. Redact the path, keep the
reason; a scrub must never become a mute.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.inference.http_server import create_app

# A username that cannot collide with the real one running the suite, so
# "was it scrubbed" is decidable instead of accidentally true.
FAKE_USER = "ws9victimuser"


def _engine():
    e = MagicMock()
    e.display_name = "test-model"
    e.count_tokens.return_value = 5
    return e


def _strings(payload) -> str:
    """Every string leaf of a decoded JSON body, joined.

    Deliberately NOT ``response.text``: JSON escapes ``\\`` as ``\\\\``, so a
    Windows path assertion against the raw text can NEVER match and the test
    passes whether or not the leak is real. Two of these assertions were
    false-greens against pre-fix source until this walked the DECODED values.
    """
    out: list[str] = []

    def walk(node):
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(payload)
    return "\n".join(out)


def _assert_absent(haystack: str, needle: str) -> None:
    """Absent in BOTH separator forms, case-insensitively on Windows (a path
    can come back lowercased or slash-flipped and still name the account)."""
    for form in {needle, needle.replace("\\", "/")}:
        hay = haystack.lower() if os.name == "nt" else haystack
        want = form.lower() if os.name == "nt" else form
        assert want not in hay, f"still discloses {form!r}"


# --------------------------------------------------------------------------- #
#  /debug/stacks (alert 97)                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture
def open_app(monkeypatch):
    """Default KEYLESS loopback mode - the configuration in which the fs-host
    gate on /debug/stacks resolves to "host" for every caller."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    app = create_app(_engine())
    with TestClient(app) as c:
        yield c, app


def _shell(app):
    return {"Authorization": f"Bearer {app.state.shell_token}"}


class TestDebugStacksGate:
    def test_keyless_mode_really_is_the_tautology_case(self, open_app):
        """Guard the PREMISE: if this ever stops being true the gate below is
        testing nothing, and the test would keep passing anyway."""
        from localm.auth import any_key_configured
        assert not any_key_configured()

    def test_no_credential_refused(self, open_app):
        # NEGATIVE: pre-fix this returned 200 with every thread's stack.
        c, _ = open_app
        r = c.get("/debug/stacks")
        assert r.status_code in (401, 403), r.text

    def test_shell_token_allowed(self, open_app):
        c, app = open_app
        r = c.get("/debug/stacks", headers=_shell(app))
        assert r.status_code == 200, r.text
        body = r.json()
        assert "threads" in body and body["threads"]

    def test_frames_do_not_leak_install_or_home_prefix(self, open_app):
        """The stacks stay useful (file + line + function) but stop carrying the
        absolute prefixes that name the OS account and the install location."""
        from localm.config import home_dir
        c, app = open_app
        r = c.get("/debug/stacks", headers=_shell(app))
        assert r.status_code == 200
        text = _strings(r.json())
        install_root = str(Path(sys.modules["localm"].__file__).resolve().parent.parent)
        # Guard the PREMISE: an unredacted frame really would carry this prefix,
        # so a green here means redaction happened, not that the string was
        # never going to appear.
        assert install_root in str(Path(create_app.__code__.co_filename).resolve())
        for leaked in (install_root, str(home_dir()), str(Path.home())):
            _assert_absent(text, leaked)
        # Rule 5: redacted, NOT muted - the debugging value must survive.
        assert "http_server.py" in text
        assert "line " in text

    def test_still_404s_off_loopback(self, monkeypatch):
        """The pre-existing bind_host gate is untouched by the new token gate."""
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
        app = create_app(_engine())
        with TestClient(app) as c:
            app.state.bind_host = "0.0.0.0"
            r = c.get("/debug/stacks", headers=_shell(app))
            assert r.status_code == 404, r.text

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.50", "10.0.0.7"])
    def test_off_loopback_no_credential_still_404s_not_403(self, monkeypatch, host):
        """The token gate must NOT answer 403 off loopback.

        /debug/stacks is deliberately hidden (404) on a network bind, and an
        unknown path under /debug/ also 404s - so a 403 here would tell an
        unauthenticated network caller that this endpoint exists, opening a new
        existence oracle while closing a disclosure. It must stay
        indistinguishable from a path that is not there."""
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
        app = create_app(_engine())
        with TestClient(app) as c:
            app.state.bind_host = host
            assert c.get("/debug/stacks").status_code == 404
            # The control: a path that genuinely does not exist.
            assert c.get("/debug/no-such-endpoint").status_code == 404


# --------------------------------------------------------------------------- #
#  Embedding status strings (alert 98)                                         #
# --------------------------------------------------------------------------- #

@pytest.fixture
def rag_app(tmp_path, monkeypatch):
    """Minimal app with the builtin rag plugin mounted (mirrors the fixture in
    tests/test_rag_confinement.py)."""
    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.web import attach_gui
    home = tmp_path
    data = home / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(data))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", data)
    monkeypatch.setattr(cfg, "MODELS_DIR", data / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", data / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", data / "registry.json")

    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("rag")

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    return app, home


class TestEmbeddingStatusScrub:
    """The load-failure reason must reach the caller WITHOUT the absolute path."""

    def _poison(self, monkeypatch, home):
        """Plant a failure message shaped exactly like the real one at
        embedder.py:997 - a genuine cause plus a machine-identifying path."""
        from localm.config import home_dir
        import localm.inference.embedder as emb
        secret_path = str(home_dir() / "models" / "embeddings" / "bge.gguf")
        user_path = str(Path(f"C:/Users/{FAKE_USER}/models/bge.gguf"))
        monkeypatch.setattr(
            emb, "_LAST_ERROR",
            f"failed to load embedding model: {secret_path} "
            f"(also tried {user_path}): not an embedding model", raising=False)
        fake = types.SimpleNamespace(
            gpu_fallback_reason=f"native GPU crash loading {secret_path}; using CPU",
            dim=None, model_path=None, active_requests=0)
        monkeypatch.setattr(emb, "_EMBEDDER", fake, raising=False)
        return secret_path, user_path

    def test_accessors_scrub_but_keep_the_cause(self, rag_app, monkeypatch):
        _, home = rag_app
        secret_path, user_path = self._poison(monkeypatch, home)
        from localm.inference.embedder import gpu_fallback_reason, last_error
        err = last_error()
        assert secret_path not in err and user_path not in err
        assert FAKE_USER not in err
        # Rule 5: the reason survives.
        assert "not an embedding model" in err
        gfr = gpu_fallback_reason()
        assert secret_path not in gfr and "native GPU crash" in gfr

    def test_route_body_carries_no_absolute_path(self, rag_app, monkeypatch):
        app, home = rag_app
        secret_path, user_path = self._poison(monkeypatch, home)
        from localm.config import home_dir
        with TestClient(app) as c:
            r = c.get("/api/rag/embedding")
            assert r.status_code == 200, r.text
            body = r.json()
            blob = _strings(body)
            _assert_absent(blob, str(home_dir()))
            _assert_absent(blob, secret_path)
            _assert_absent(blob, user_path)
            assert FAKE_USER not in blob
            assert body["error"] and "not an embedding model" in body["error"]
            assert body["gpu_fallback_reason"]
            assert "native GPU crash" in body["gpu_fallback_reason"]

    def test_scrubs_a_data_dir_that_is_not_under_the_home_dir(self, tmp_path,
                                                              monkeypatch):
        """The two tests above cannot prove _machine_prefixes does anything.

        Their fixture puts the data dir INSIDE the patched Path.home(), so
        scrub_user_paths' home -> "~" replacement alone already removes the
        asserted string - stubbing _machine_prefixes() to [] leaves them green.
        A portable install puts LOCALM_HOME nowhere near the home dir, and that
        is the case that needs the data-dir prefix."""
        elsewhere = tmp_path / "portable" / "localm-data"
        elsewhere.mkdir(parents=True)
        monkeypatch.setenv("LOCALM_HOME", str(elsewhere))
        import localm.config as cfg
        monkeypatch.setattr(cfg, "HOME_DIR", elsewhere)
        # Home is somewhere else entirely, so the username rule cannot cover this.
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "hom"))
        from localm.pathscrub import scrub_paths
        text = f"failed to load embedding model: {elsewhere}/m.gguf: bad model"
        out = scrub_paths(text)
        _assert_absent(out, str(elsewhere))
        assert "bad model" in out

    def test_scrubs_the_interpreter_prefix_as_shipped(self):
        """sys.prefix / sys.base_prefix must be scrubbed in the form the TEXT
        carries, not only in resolved form.

        localm provisions its interpreter with uv, whose directory is a
        version-less alias that Path.resolve() follows to the versioned real
        path. Registering only the resolved form matched nothing, so every
        stdlib frame in /debug/stacks came back with a full absolute path while
        the scrubber reported success."""
        from localm.pathscrub import scrub_paths
        for prefix in {sys.prefix, sys.base_prefix}:
            frame = f'  File "{prefix}{os.sep}Lib{os.sep}threading.py", line 1, in x'
            out = scrub_paths(frame)
            _assert_absent(out, prefix)
            assert "threading.py" in out and "line 1" in out

    def test_none_stays_none(self, rag_app, monkeypatch):
        """A scrubber that turns None into "" would break the GUI's
        "is there an error at all" check."""
        import localm.inference.embedder as emb
        monkeypatch.setattr(emb, "_LAST_ERROR", None, raising=False)
        monkeypatch.setattr(emb, "_EMBEDDER", None, raising=False)
        from localm.inference.embedder import gpu_fallback_reason, last_error
        assert last_error() is None
        assert gpu_fallback_reason() is None


# --------------------------------------------------------------------------- #
#  /api/rag/collections/{name}/add existence oracle (alert 59)                 #
# --------------------------------------------------------------------------- #

class TestRagAddExistenceOracle:
    """An out-of-confinement path must get the SAME answer whether or not it
    exists, so the response cannot be used to probe the server's disk."""

    def _post(self, client, path):
        return client.post("/api/rag/collections/kb/add",
                           json={"paths": [str(path)], "embed": False})

    def test_exists_and_missing_are_indistinguishable(self, rag_app,
                                                      tmp_path_factory):
        app, _ = rag_app
        outside = tmp_path_factory.mktemp("ws9_outside")
        real = outside / "present.txt"
        real.write_text("x", encoding="utf-8")
        ghost = outside / "definitely-absent.txt"
        with TestClient(app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            r_real = self._post(c, real)
            r_ghost = self._post(c, ghost)
        # NEGATIVE: pre-fix the missing one 400'd "Not found: <path>" while the
        # existing one 409'd needs_consent - a clean existence oracle.
        assert r_real.status_code == r_ghost.status_code, (
            f"existence leaked: {r_real.status_code} vs {r_ghost.status_code}")
        assert "not found" not in r_ghost.text.lower()

    def test_secret_named_path_does_not_leak_existence_either(
            self, rag_app, tmp_path_factory):
        """The secret-material refusal used to consult is_file(), so an EXISTING
        .pem answered 400 secret_file while a missing one fell through to the
        whitelist branch (409/403) - the oracle survived for exactly the
        interesting targets. Both must now answer the same."""
        app, _ = rag_app
        outside = tmp_path_factory.mktemp("ws9_secret")
        real = outside / "server.key"
        real.write_text("PRIVATE", encoding="utf-8")
        ghost = outside / "absent.key"
        with TestClient(app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            r_real = self._post(c, real)
            r_ghost = self._post(c, ghost)
        assert r_real.status_code == r_ghost.status_code, (
            f"existence leaked: {r_real.status_code} vs {r_ghost.status_code}")

    def test_secret_named_directory_is_still_walkable(self, rag_app):
        """Not over-blocked: the is_file() guard existed so a real directory
        NAMED like a secret stays indexable. Replacing it with is_dir() must
        keep that true."""
        app, home = rag_app
        d = home / "credentials"
        d.mkdir()
        (d / "notes.md").write_text("just notes", encoding="utf-8")
        with TestClient(app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            r = self._post(c, d)
        assert r.status_code == 200, r.text

    def test_confinement_still_refuses_a_credential_dir(self, rag_app,
                                                        tmp_path_factory):
        """Moving the confinement earlier must not weaken it."""
        app, _ = rag_app
        outside = tmp_path_factory.mktemp("ws9_cred")
        ssh = outside / ".ssh"
        ssh.mkdir()
        (ssh / "id_rsa").write_text("KEY", encoding="utf-8")
        with TestClient(app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            r = self._post(c, ssh / "id_rsa")
        assert r.status_code == 400, r.text

    def test_in_policy_missing_path_still_explains_itself(self, rag_app):
        """Rule 5: inside the allowed roots the caller is entitled to the real
        reason, so a genuinely absent file must still say so."""
        app, home = rag_app
        with TestClient(app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            r = self._post(c, home / "nope.txt")
        assert r.status_code == 400
        assert "not found" in r.text.lower()

    def test_happy_path_unchanged(self, rag_app):
        app, home = rag_app
        docs = home / "kdocs"
        docs.mkdir()
        (docs / "a.md").write_text("rocm gfx1030 dll", encoding="utf-8")
        with TestClient(app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            r = self._post(c, docs)
        assert r.status_code == 200, r.text


# --------------------------------------------------------------------------- #
#  Key-digest file permissions (alert 88)                                      #
# --------------------------------------------------------------------------- #

def _icacls(path: Path) -> str:
    r = subprocess.run(["icacls", str(path)], capture_output=True,
                       text=True, check=False)
    # A tool that failed to run would make every "X not in output" assertion
    # below pass VACUOUSLY, which is how a permissions test reports a fix that
    # never happened.
    assert r.returncode == 0, f"icacls failed: {r.returncode} {r.stderr}"
    assert r.stdout.strip(), "icacls produced no output"
    return r.stdout


def _has_explicit_user_ace(out: str) -> bool:
    """True when icacls shows an ACE for the current user that is NOT inherited.

    This, not the absence of ``BUILTIN\\Users``, is the discriminator. Inside
    pytest's basetemp the ambient ACL ALREADY lacks ``BUILTIN\\Users``, so an
    absence check passes whether or not the fix ran (it did, against pre-fix
    source, until this replaced it). ``icacls /inheritance:r /grant:r user:F``
    leaves an explicit ``USER:(F)``; an inherited grant would read ``USER:(I)(F)``.
    """
    user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    if not user:
        return False
    return f"{user}:(F)" in out


@pytest.mark.skipif(os.name != "nt",
                    reason="the ACL asymmetry this covers is Windows-only; "
                           "sessions.py already chmod 0600s on POSIX")
class TestKeyDigestFilePermsWindows:
    """sessions.json / jobs.json hold the SAME sha256 the keystore stores
    (``http_server.principal_id`` returns ``key_hash``), so they need auth.key's
    ACL. Pre-fix only auth.key - the PLAINTEXT - got the icacls treatment, so
    the digest was readable where the plaintext was not."""

    def test_the_check_can_actually_fire(self, tmp_path):
        """Fires-control. A file written into the same directory WITHOUT the
        restriction must NOT show the explicit ACE, or the two tests below prove
        nothing about their subject."""
        control = tmp_path / "control.json"
        control.write_text("{}", encoding="utf-8")
        assert not _has_explicit_user_ace(_icacls(control)), (
            "an unrestricted file already shows an explicit user ACE here, so "
            "this assertion cannot distinguish restricted from unrestricted")

    def test_sessions_file_is_owner_only(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "h"))
        import localm.config as cfg
        monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "h")
        import localm.sessions as sessions
        sessions.create(scopes=["chat"], key_hash="deadbeef" * 8)
        f = sessions.sessions_file()
        assert f.is_file()
        out = _icacls(f)
        # NEGATIVE: pre-fix sessions._restrict_perms was POSIX-only, so no
        # explicit ACE was ever written on Windows.
        assert _has_explicit_user_ace(out), out
        assert "BUILTIN\\Users" not in out, out
        assert "Everyone" not in out, out

    def test_jobs_defs_file_is_owner_only(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "h"))
        import localm.config as cfg
        monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "h")
        from localm.plugins.builtin.jobs.store import Job, JobStore
        store = JobStore(root=tmp_path / "h" / "jobs")
        store.add(Job(name="j", task_kind="chat", prompt="hi",
                      owner="deadbeef" * 8))
        f = store._defs_file
        assert f.is_file()
        out = _icacls(f)
        assert _has_explicit_user_ace(out), out
        assert "BUILTIN\\Users" not in out, out
        assert "Everyone" not in out, out

    def test_corrupt_backup_is_owner_only(self, tmp_path, monkeypatch):
        """The quarantine copy is a VERBATIM copy of jobs.json, digests and all,
        and nothing ever deletes it - so restricting only the live file would
        leave the exposure behind permanently on the first corrupt read."""
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "h"))
        import localm.config as cfg
        monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "h")
        from localm.plugins.builtin.jobs.store import Job, JobStore
        store = JobStore(root=tmp_path / "h" / "jobs")
        store.add(Job(name="j", task_kind="chat", prompt="hi",
                      owner="deadbeef" * 8))
        # Corrupt the defs file WITHOUT losing its contents - appending garbage
        # keeps the digests in the file the quarantine copies. (Overwriting it
        # outright made the backup carry only the garbage, which the premise
        # assertion below correctly refused to accept as coverage.)
        store._defs_file.write_text(
            store._defs_file.read_text(encoding="utf-8") + "\n{trailing garbage",
            encoding="utf-8")
        store._read_all()
        backups = list(store._defs_file.parent.glob("jobs.json.corrupt-*"))
        assert backups, "no quarantine copy was written"
        for b in backups:
            assert "deadbeef" in b.read_text(encoding="utf-8"), (
                "premise check: the backup really does carry the digest")
            out = _icacls(b)
            assert _has_explicit_user_ace(out), out
            assert "BUILTIN\\Users" not in out, out


class TestKeyDigestFilePermsPosix:
    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
    def test_sessions_file_mode(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "h"))
        import localm.config as cfg
        monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "h")
        import localm.sessions as sessions
        sessions.create(scopes=["chat"], key_hash="deadbeef" * 8)
        assert (sessions.sessions_file().stat().st_mode & 0o077) == 0

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
    def test_jobs_defs_file_mode(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "h"))
        import localm.config as cfg
        monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "h")
        from localm.plugins.builtin.jobs.store import Job, JobStore
        store = JobStore(root=tmp_path / "h" / "jobs")
        store.add(Job(name="j", task_kind="chat", prompt="hi",
                      owner="deadbeef" * 8))
        assert (store._defs_file.stat().st_mode & 0o077) == 0

    # DELIBERATELY NO "the hash stays SHA-256" TEST HERE.
    #
    # An earlier revision of this file had one, on the premise that every key
    # reaching auth._hash_key is a generated secrets.token_urlsafe(32) so a KDF
    # would be pure latency cost. That premise is FALSE for the owner key:
    # `localm key set KEY` takes a key the user provides, and the LOCALM_API_KEY
    # env var and a hand-edited auth.key bypass set_api_key's length/charset
    # guards entirely (its own docstring says so). A user-chosen, unvalidated
    # secret therefore reaches a single unsalted SHA-256, which makes CodeQL
    # alert 88 a true positive on that path.
    #
    # A test pinning the algorithm would have turned the CORRECT fix red and
    # made it look like the regression. The ACL coverage above is the part of
    # alert 88 this unit actually closes; the hashing half is an open design
    # decision (move the expensive verify off the per-request path) tracked
    # outside this file. See the comment at auth._hash_key.
