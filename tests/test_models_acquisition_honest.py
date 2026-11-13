# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model acquisition does what it says.

``--redownload`` fetches the files of the pull again and replaces the copy in
the models folder on every pull path (Hugging Face file, split GGUF, full repo,
direct URL), only after the new bytes match their digest, and never over a file
a model from another source uses. ``localm search owner/repo@rev --files`` lists
the files of that revision. Registering a local path from the GUI runs as a
"Model add" job, not a download.

The pulls run through the real ``pull_model``, registry and staging code; only
the Hugging Face client calls and the HTTP transport are doubled.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm import model_manager as mm
from localm.model_manager import pull as pull_mod
from tests._net_config import set_config

BODY = b"weights" * 200
BAD = b"damaged" * 200
CONFIG = b'{"a": 1}'


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _plain(capsys) -> str:
    return " ".join(capsys.readouterr().out.split())


# ------------------------------------------------------------------ #
#  Hugging Face                                                        #
# ------------------------------------------------------------------ #

@pytest.fixture()
def hub(tmp_path, monkeypatch):
    """A fake Hugging Face hub and a real registry in this test's LOCALM_HOME.
    ``files`` maps a repo file to its published bytes (its digest is published
    when it is listed in ``lfs``); ``served`` overrides the bytes a download
    gets."""
    set_config(monkeypatch, {"net_mode": "ask"})
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.setattr("socket.getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    monkeypatch.setattr(mm, "_hf_repo_mmproj_filename", lambda *a, **k: None)
    monkeypatch.setattr(mm, "gguf_unusable_reason", lambda *a, **k: None)
    monkeypatch.setattr(mm, "gguf_registry_metadata", lambda *a, **k: {})
    state = SimpleNamespace(models=models, files={}, lfs=set(), served={},
                            downloads=[], snapshots=[])

    def published(name):
        return _sha(state.files[name]) if name in state.lfs else None

    def fake_sha(repo, filename, token=None, revision=None):
        d = published(filename)
        return pull_mod._PublishedDigest(d) if d else None

    def served(name):
        return state.served.get(name, state.files[name])

    def fake_download(repo_id, filename, local_dir, revision=None, **kw):
        state.downloads.append((filename, Path(local_dir)))
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(served(filename))
        return str(p)

    def fake_snapshot(repo_id, local_dir, revision=None, **kw):
        state.snapshots.append(Path(local_dir))
        for name in state.files:
            p = Path(local_dir) / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(served(name))
        return local_dir

    class Api:
        def model_info(self, repo_id, revision=None, files_metadata=False, **kw):
            sibs = [SimpleNamespace(
                rfilename=n, size=len(b), blob_id=None,
                lfs={"sha256": _sha(b), "size": len(b)} if n in state.lfs else None)
                for n, b in state.files.items()]
            return SimpleNamespace(sha="c" * 40, siblings=sibs)

    def fake_head(url, allow_redirects=None, timeout=None):
        h = MagicMock()
        h.headers = {"content-length": "1"}
        return h

    import huggingface_hub
    monkeypatch.setattr(pull_mod, "_hf_api", lambda: Api())
    monkeypatch.setattr(mm, "_hf_file_sha256", fake_sha)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(pull_mod, "_policy_head", fake_head)
    return state


class TestHfFileRedownload:
    def _pulled(self, hub):
        hub.files = {"w.safetensors": BODY}
        hub.lfs = {"w.safetensors"}
        assert mm.pull_model("o/r:w.safetensors", name="m", model_type="llm") is True
        hub.downloads.clear()
        return hub.models / "w.safetensors"

    def test_a_damaged_file_is_fetched_again_and_replaced(self, hub, capsys):
        dest = self._pulled(hub)
        dest.write_bytes(BAD)
        capsys.readouterr()
        assert mm.pull_model("o/r:w.safetensors", name="m", model_type="llm") is False
        assert "Delete it and pull again" in _plain(capsys)
        assert dest.read_bytes() == BAD and hub.downloads == []

        assert mm.pull_model("o/r:w.safetensors", name="m", model_type="llm",
                             redownload=True) is True
        assert dest.read_bytes() == BODY
        [(name, landed)] = hub.downloads
        assert landed == hub.models / pull_mod._REPO_FOLDER_STAGING
        assert not (landed / "w.safetensors").exists()
        entry = mm.load_registry()["m"]
        assert entry["source"] == "hf:o/r" and entry["sha256"] == _sha(BODY)

    def test_a_file_already_there_is_downloaded_again(self, hub, capsys):
        dest = self._pulled(hub)
        capsys.readouterr()
        assert mm.pull_model("o/r:w.safetensors", name="m", model_type="llm") is True
        assert "Already downloaded" in _plain(capsys) and hub.downloads == []
        assert mm.pull_model("o/r:w.safetensors", name="m", model_type="llm",
                             redownload=True) is True
        assert [n for n, _ in hub.downloads] == ["w.safetensors"]
        assert "Already downloaded" not in _plain(capsys)
        assert dest.read_bytes() == BODY

    def test_new_bytes_failing_their_digest_leave_the_old_file(self, hub, capsys):
        dest = self._pulled(hub)
        hub.served = {"w.safetensors": BAD}
        capsys.readouterr()
        assert mm.pull_model("o/r:w.safetensors", name="m", model_type="llm",
                             redownload=True) is False
        out = _plain(capsys)
        assert "SHA256 mismatch" in out and "left as they were" in out
        assert dest.read_bytes() == BODY
        staged = hub.models / pull_mod._REPO_FOLDER_STAGING / "w.safetensors"
        assert not staged.exists()

    def test_a_file_another_source_uses_is_never_replaced(self, hub, capsys):
        hub.files = {"w.safetensors": BODY}
        hub.lfs = {"w.safetensors"}
        dest = hub.models / "w.safetensors"
        dest.write_bytes(BAD)
        mm._register("theirs", dest, "hf:x/y", model_type="llm")
        assert mm.pull_model("o/r:w.safetensors", name="m", model_type="llm",
                             redownload=True) is False
        out = _plain(capsys)
        assert "Refusing to replace" in out and "'theirs' (hf:x/y)" in out
        assert "Nothing was changed" in out
        assert hub.downloads == [] and dest.read_bytes() == BAD
        assert "m" not in mm.load_registry()

    def test_every_part_of_a_split_gguf_is_replaced(self, hub):
        p1, p2 = "m-00001-of-00002.gguf", "m-00002-of-00002.gguf"
        hub.files = {p1: BODY, p2: BODY + b"2"}
        hub.lfs = {p1, p2}
        assert mm.pull_model(f"o/r:{p1}", name="m", model_type="llm") is True
        (hub.models / p2).write_bytes(BAD)
        hub.downloads.clear()
        assert mm.pull_model(f"o/r:{p1}", name="m", model_type="llm",
                             redownload=True) is True
        assert sorted(n for n, _ in hub.downloads) == [p1, p2]
        assert (hub.models / p1).read_bytes() == BODY
        assert (hub.models / p2).read_bytes() == BODY + b"2"

    def test_one_failing_part_replaces_none_of_them(self, hub):
        p1, p2 = "m-00001-of-00002.gguf", "m-00002-of-00002.gguf"
        hub.files = {p1: BODY, p2: BODY + b"2"}
        hub.lfs = {p1, p2}
        assert mm.pull_model(f"o/r:{p1}", name="m", model_type="llm") is True
        (hub.models / p1).write_bytes(BAD)
        hub.served = {p2: BAD}
        assert mm.pull_model(f"o/r:{p1}", name="m", model_type="llm",
                             redownload=True) is False
        assert (hub.models / p1).read_bytes() == BAD
        assert (hub.models / p2).read_bytes() == BODY + b"2"


class TestRefuseReplacing:
    """The one rule every pull with redownload asks."""

    def test_an_entry_of_another_revision_is_another_source(self, hub, capsys):
        dest = hub.models / "w.safetensors"
        dest.write_bytes(BODY)
        mm._register("v1", dest, "hf:o/r", model_type="llm", revision="v1",
                     commit="a" * 40)
        assert pull_mod._refuse_replacing(dest, "hf:o/r") is True
        assert "'v1' (hf:o/r@v1)" in _plain(capsys)
        assert pull_mod._refuse_replacing(dest, "hf:o/r@v1") is False

    def test_a_file_nobody_registered_may_be_replaced(self, hub):
        dest = hub.models / "w.safetensors"
        dest.write_bytes(BODY)
        elsewhere = hub.models / "other.safetensors"
        elsewhere.write_bytes(BAD)
        mm._register("theirs", elsewhere, "hf:x/y", model_type="llm")
        assert pull_mod._refuse_replacing(dest, "hf:o/r") is False

    def test_an_unreadable_registry_refuses(self, hub, monkeypatch, capsys):
        dest = hub.models / "w.safetensors"
        dest.write_bytes(BODY)
        monkeypatch.setattr(pull_mod, "_civitai_registry_unreadable", lambda: True)
        assert pull_mod._refuse_replacing(dest, "hf:o/r") is True
        assert "could not be read" in _plain(capsys)


class TestHfRepoRedownload:
    def _pulled(self, hub):
        hub.files = {"config.json": CONFIG, "w.safetensors": BODY}
        hub.lfs = {"w.safetensors"}
        assert mm.pull_model("o/r", name="snap", model_type="llm") is True
        hub.snapshots.clear()
        return hub.models / "snap"

    def test_a_file_no_check_can_see_is_replaced(self, hub, capsys):
        dest = self._pulled(hub)
        (dest / "config.json").write_bytes(b'{"a": 2}')
        capsys.readouterr()
        assert mm.pull_model("o/r", name="snap", model_type="llm") is True
        assert "Already downloaded" in _plain(capsys) and hub.snapshots == []
        assert (dest / "config.json").read_bytes() == b'{"a": 2}'

        assert mm.pull_model("o/r", name="snap", model_type="llm",
                             redownload=True) is True
        assert hub.snapshots == [dest / pull_mod._SNAPSHOT_REDOWNLOAD_STAGING]
        assert (dest / "config.json").read_bytes() == CONFIG
        assert (dest / "w.safetensors").read_bytes() == BODY
        assert not (dest / pull_mod._SNAPSHOT_REDOWNLOAD_STAGING).exists()
        assert mm.load_registry()["snap"]["source"] == "hf:o/r"

    def test_new_bytes_failing_their_digest_leave_the_folder(self, hub, capsys):
        dest = self._pulled(hub)
        hub.served = {"w.safetensors": BAD, "config.json": b'{"a": 3}'}
        capsys.readouterr()
        assert mm.pull_model("o/r", name="snap", model_type="llm",
                             redownload=True) is False
        out = _plain(capsys)
        assert "SHA256 mismatch" in out and "were left as they were" in out
        assert (dest / "w.safetensors").read_bytes() == BODY
        assert (dest / "config.json").read_bytes() == CONFIG

    def test_a_folder_another_source_uses_is_never_replaced(self, hub, capsys):
        dest = self._pulled(hub)
        mm._register("theirs", dest, "hf:x/y", model_type="llm")
        capsys.readouterr()
        assert mm.pull_model("o/r", name="snap", model_type="llm",
                             redownload=True) is False
        assert "Refusing to replace" in _plain(capsys)
        assert hub.snapshots == []

    def test_unreadable_digests_refuse_instead_of_replacing_unchecked(
            self, hub, monkeypatch, capsys):
        dest = self._pulled(hub)
        hub.served = {"config.json": b'{"a": 3}'}

        class Down:
            def model_info(self, *a, **k):
                raise OSError("offline")

            def repo_info(self, *a, **k):
                return SimpleNamespace(sha="c" * 40)

            def list_repo_files(self, *a, **k):
                return list(hub.files)

        monkeypatch.setattr(pull_mod, "_hf_api", lambda: Down())
        capsys.readouterr()
        assert mm.pull_model("o/r", name="snap", model_type="llm",
                             redownload=True) is False
        assert "could not be read" in _plain(capsys)
        assert hub.snapshots == [] and (dest / "config.json").read_bytes() == CONFIG

    def test_an_interrupted_redownload_leaves_temp_files_discard_can_find(self, hub):
        dest = self._pulled(hub)
        pending = (dest / pull_mod._SNAPSHOT_REDOWNLOAD_STAGING / ".cache"
                   / "huggingface" / "download")
        pending.mkdir(parents=True)
        (pending / "w.safetensors.abc.incomplete").write_bytes(b"x" * 10)
        assert pull_mod.discard_partial_download("o/r", "snap") == 10
        assert not (pending / "w.safetensors.abc.incomplete").exists()
        assert (dest / "w.safetensors").read_bytes() == BODY


# ------------------------------------------------------------------ #
#  Direct URL                                                          #
# ------------------------------------------------------------------ #

URL = "http://host1.example/tiny.gguf"


def _resp(status, body: bytes):
    r = MagicMock()
    r.status_code = status
    r.raise_for_status = MagicMock()
    r.headers = {"content-length": str(len(body))}
    r.iter_content = lambda n: (body[i:i + n] for i in range(0, len(body), n))
    return r


class _Server:
    """A fake ``netpolicy.pinned_request`` serving *body*."""

    def __init__(self, body: bytes):
        self.body = body
        self.gets = 0

    def __call__(self, method, url, **kw):
        if method == "HEAD":
            h = MagicMock()
            h.status_code = 200
            h.headers = {"content-length": str(len(self.body))}
            return h
        self.gets += 1
        return _resp(200, self.body)


@pytest.fixture()
def web(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    monkeypatch.setattr(mm, "gguf_unusable_reason", lambda *a, **k: None)
    monkeypatch.setattr(
        "socket.getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    monkeypatch.setenv("LOCALM_NET_MODE", "allow")
    server = _Server(BODY)
    monkeypatch.setattr("localm.netpolicy.pinned_request", server)
    return SimpleNamespace(models=models, server=server, dest=models / "tiny.gguf")


class TestUrlRedownload:
    def test_a_damaged_file_is_fetched_again_and_replaced(self, web, capsys):
        assert mm.pull_model(URL, name="tiny", model_type="llm") is True
        web.dest.write_bytes(BAD)
        web.server.gets = 0
        capsys.readouterr()
        assert mm.pull_model(URL, name="tiny", model_type="llm") is True
        assert "Already downloaded" in _plain(capsys)
        assert web.server.gets == 0 and web.dest.read_bytes() == BAD

        assert mm.pull_model(URL, name="tiny", model_type="llm", redownload=True) is True
        assert web.server.gets == 1 and web.dest.read_bytes() == BODY
        assert not (web.models / "tiny.gguf.part").exists()
        assert mm.load_registry()["tiny"]["sha256"] == _sha(BODY)

    def test_new_bytes_failing_sha256_leave_the_old_file(self, web, capsys):
        web.dest.write_bytes(BAD)
        mm._register("tiny", web.dest, URL, model_type="llm")
        capsys.readouterr()
        assert mm.pull_model(URL, name="tiny", model_type="llm", redownload=True,
                             expected_sha256=_sha(b"something else")) is False
        assert "SHA256 mismatch" in _plain(capsys)
        assert web.dest.read_bytes() == BAD
        assert not (web.models / "tiny.gguf.part").exists()

    def test_a_file_another_source_uses_is_never_replaced(self, web, capsys):
        web.dest.write_bytes(BAD)
        mm._register("theirs", web.dest, "hf:x/y", model_type="llm")
        assert mm.pull_model(URL, name="tiny", model_type="llm", redownload=True) is False
        out = _plain(capsys)
        assert "Refusing to replace" in out and "'theirs' (hf:x/y)" in out
        assert web.server.gets == 0 and web.dest.read_bytes() == BAD

    def test_a_sha256_mismatch_names_a_step_that_works(self, web, capsys):
        web.dest.write_bytes(BAD)
        assert mm.pull_model(URL, name="tiny", model_type="llm",
                             expected_sha256=_sha(BODY)) is False
        out = _plain(capsys)
        assert "Pull again with --redownload to replace it" in out
        assert "-n name" not in out


# ------------------------------------------------------------------ #
#  localm search owner/repo@revision --files                           #
# ------------------------------------------------------------------ #

class TestSearchFilesAtARevision:
    TREE = [{"path": "m-Q4_K_M.gguf", "size": 100}, {"path": "README.md", "size": 1}]

    @pytest.fixture()
    def urls(self, monkeypatch):
        monkeypatch.setenv("LOCALM_NET_MODE", "ask")
        seen: list[str] = []

        def fake(url, **kw):
            seen.append(url)
            return url, "application/json", json.dumps(self.TREE).encode("utf-8")
        monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", fake)
        return seen

    @pytest.mark.parametrize("repo, tree", [
        ("owner/repo", "/api/models/owner/repo/tree/main?"),
        ("owner/repo@v1.0", "/api/models/owner/repo/tree/v1.0?"),
        ("owner/repo@refs/pr/3", "/api/models/owner/repo/tree/refs%2Fpr%2F3?"),
    ])
    def test_the_files_of_the_revision_are_listed(self, urls, repo, tree):
        from localm.discover import hf_gguf_files
        files = hf_gguf_files(repo)
        assert [f["file"] for f in files] == ["m-Q4_K_M.gguf"]
        assert tree in urls[0]

    @pytest.mark.parametrize("repo", ["owner/repo@", "owner/repo@a..b",
                                      "owner/repo@v1:m.gguf"])
    def test_a_malformed_spec_is_refused_without_a_request(self, urls, repo):
        from localm.discover import DiscoverError, hf_gguf_files
        with pytest.raises(DiscoverError):
            hf_gguf_files(repo)
        assert urls == []

    def test_the_cli_lists_them_and_names_the_pull_spec(self, urls, monkeypatch, cli_runner):
        from localm.cli import main
        monkeypatch.setattr("localm.discover.vram_capacity", lambda: {})
        r = CliRunner().invoke(main, ["search", "owner/repo@v1.0", "--files"])
        assert r.exit_code == 0, r.output
        assert "m-Q4_K_M.gguf" in r.output
        assert "localm pull owner/repo@v1.0:<file>" in " ".join(r.output.split())


# ------------------------------------------------------------------ #
#  GUI: registering a local path is a "Model add" job                  #
# ------------------------------------------------------------------ #

class _Jobs:
    def __init__(self):
        self.cli_calls = []
        self.running = []

    def start_cli(self, kind, cli_args, **kw):
        self.cli_calls.append((kind, list(cli_args), kw))
        return SimpleNamespace(id=f"job-{len(self.cli_calls)}")

    def snapshot(self, visible=None):
        return list(self.running)


@pytest.fixture()
def gui(monkeypatch):
    from localm.plugins.gui.routes import models as models_routes
    monkeypatch.setattr("localm.plugins.lookup.is_available", lambda name: True)
    app = FastAPI()
    jobs = _Jobs()
    models_routes.register(app, SimpleNamespace(active_model=lambda: "",
                                                switch_model=None, jobs=jobs))
    return SimpleNamespace(app=app, jobs=jobs)


class TestLocalAddJob:
    def test_a_local_file_is_an_add_job_with_its_own_label(self, gui, tmp_path):
        f = tmp_path / "mine.gguf"
        f.write_bytes(b"GGUF")
        with TestClient(gui.app) as c:
            r = c.post("/api/models/pull", json={"spec": str(f)})
        assert r.status_code == 200 and r.json()["kind"] == "add"
        kind, args, kw = gui.jobs.cli_calls[0]
        assert kind == "add" and kw["host_label"] == f"Model add {f}"
        assert args[-1] == str(f)

    def test_a_remote_spec_is_still_a_pull_job(self, gui):
        with TestClient(gui.app) as c:
            r = c.post("/api/models/pull", json={"spec": "o/r:m.gguf"})
        assert r.json()["kind"] == "pull"
        kind, _args, kw = gui.jobs.cli_calls[0]
        assert kind == "pull" and kw["host_label"] == "Model pull o/r:m.gguf"

    def test_a_running_add_of_the_same_path_is_409_in_add_words(self, gui, tmp_path):
        f = tmp_path / "mine.gguf"
        f.write_bytes(b"GGUF")
        gui.jobs.running = [{"kind": "add", "status": "running",
                             "label": f"Model add {f}"}]
        with TestClient(gui.app) as c:
            r = c.post("/api/models/pull", json={"spec": str(f)})
        assert r.status_code == 409
        assert r.json()["detail"].startswith(f"Already adding {f}")
        assert "download" not in r.json()["detail"]
        assert gui.jobs.cli_calls == []

    def test_a_running_pull_does_not_block_an_add_of_a_path(self, gui, tmp_path):
        f = tmp_path / "mine.gguf"
        f.write_bytes(b"GGUF")
        gui.jobs.running = [{"kind": "pull", "status": "running",
                             "label": f"Model pull {f}"}]
        with TestClient(gui.app) as c:
            assert c.post("/api/models/pull", json={"spec": str(f)}).status_code == 200

    def test_a_caller_without_host_access_never_adds(self, gui, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "rel.gguf").write_bytes(b"GGUF")
        monkeypatch.setattr(
            "localm.plugins.gui.routes.models.acquisition.effective_fs_access",
            lambda request: "none")
        with TestClient(gui.app) as c:
            r = c.post("/api/models/pull", json={"spec": "rel.gguf"})
        kind, args, _kw = gui.jobs.cli_calls[0]
        assert r.json()["kind"] == "pull" and kind == "pull"
        assert "--remote-only" in args

    def test_a_relative_file_is_an_add_for_a_host_caller(self, gui, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "rel.gguf").write_bytes(b"GGUF")
        with TestClient(gui.app) as c:
            r = c.post("/api/models/pull", json={"spec": "rel.gguf"})
        assert r.json()["kind"] == "add"
        assert gui.jobs.cli_calls[0][2]["host_label"] == "Model add rel.gguf"


class TestLocalSpecPathIsTheOneRule:
    def test_pull_model_and_the_route_use_the_same_decision(self, tmp_path):
        f = tmp_path / "mine.gguf"
        f.write_bytes(b"GGUF")
        assert pull_mod.local_spec_path(str(f)) == f
        assert pull_mod.local_spec_path(str(f), allow_local=False) is None
        assert pull_mod.local_spec_path("o/r:m.gguf") is None
        assert pull_mod.local_spec_path(str(tmp_path / "missing.gguf")) is None
