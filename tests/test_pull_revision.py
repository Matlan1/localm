# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm pull owner/repo@<revision>[:file]`` pulls a branch, tag or commit of a
Hugging Face repo: the spec is parsed without disturbing URLs, paths or file
names holding an ``@``, a revision that does not exist is refused before any
download (and kept apart from a network or permission failure), every hub call
of the pull is pinned to the revision's commit, and the registry entry records
it. The hub is replaced by fakes; the parser, the pull, the registry and
``localm list`` are the real code."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import hashlib

import httpx
import pytest

from localm import model_manager as mm
from localm.model_manager import pull as pull_mod
from tests._net_config import set_config

COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40
BODY = b"weights" * 100


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/api"))


class TestSplitHfRevision:
    @pytest.mark.parametrize("spec, expected", [
        ("owner/repo", ("owner/repo", None)),
        ("owner/repo:m.gguf", ("owner/repo:m.gguf", None)),
        ("owner/repo/m.gguf", ("owner/repo/m.gguf", None)),
        ("owner/repo@4.0bpw", ("owner/repo", "4.0bpw")),
        ("owner/repo@v1.0:m.gguf", ("owner/repo:m.gguf", "v1.0")),
        ("owner/repo@refs/pr/3", ("owner/repo", "refs/pr/3")),
        ("owner/repo@refs/pr/3:sub/dir/m.safetensors",
         ("owner/repo:sub/dir/m.safetensors", "refs/pr/3")),
        ("owner/repo@" + COMMIT, ("owner/repo", COMMIT)),
    ])
    def test_a_revision_is_split_off_the_repo(self, spec, expected):
        assert pull_mod.split_hf_revision(spec) == expected

    @pytest.mark.parametrize("spec", [
        "https://user@host.example/x.gguf",
        "http://user:pw@host.example/a@b/x.gguf",
        "D:/a@b/m.gguf",
        "D:\\models\\a@b.gguf",
        "/abs/dir@v2/m.gguf",
        "\x7e/dir@v2/m.gguf",
        "./dir/repo@v2",
        "../dir/repo@v2",
        "owner/repo:m@2.gguf",
        "owner/repo/m@2.gguf",
        "owner/repo/sub@x/m.gguf",
        "civitai:123@4",
        "user@host",
        "name@rev",
        "",
    ])
    def test_a_spec_that_is_not_owner_repo_at_rev_is_returned_untouched(self, spec):
        assert pull_mod.split_hf_revision(spec) == (spec, None)

    @pytest.mark.parametrize("spec, fragment", [
        ("owner/repo@", "empty revision"),
        ("owner/repo@:m.gguf", "empty revision"),
        ("owner/repo@rev/m.gguf", "owner/repo@<revision>:<file>"),
        ("owner/repo@a b", "not a valid"),
        ("owner/repo@..", "not a valid"),
        ("owner/repo@a..b", "not a valid"),
        ("owner/repo@-x", "not a valid"),
        ("owner/repo@a//b", "not a valid"),
        ("owner/repo@a/", "not a valid"),
        ("owner/repo@x" + "y" * 300, "not a valid"),
        ("owner/repo@rev:", "no file name"),
    ])
    def test_a_malformed_revision_is_refused_with_a_message(self, spec, fragment):
        with pytest.raises(ValueError) as e:
            pull_mod.split_hf_revision(spec)
        assert fragment in str(e.value)
        assert len(str(e.value)) < 300


class _FakeApi:
    """Stands in for ``HfApi``: ``model_info`` answers from ``state``."""

    def __init__(self, state):
        self.state = state

    def model_info(self, repo_id, revision=None, files_metadata=False, **kw):
        self.state.info_calls.append((repo_id, revision, files_metadata))
        if self.state.info_error is not None:
            raise self.state.info_error
        if not revision:
            sha = self.state.default_commit
        else:
            sha = self.state.revisions.get(revision) or (
                revision if revision in self.state.repo_files else None)
        if sha is None:
            raise self.state.missing_error(revision)
        files = [SimpleNamespace(rfilename=n, size=len(b), lfs=None, blob_id=None)
                 for n, b in self.state.repo_files.get(sha, {}).items()]
        return SimpleNamespace(sha=sha, siblings=files)


@pytest.fixture()
def hub(tmp_path, monkeypatch):
    """A fake hub plus a real registry in a throwaway LOCALM_HOME. ``revisions``
    maps a revision to its commit; ``repo_files`` maps a commit to its files."""
    from huggingface_hub.errors import RevisionNotFoundError

    set_config(monkeypatch, {"net_mode": "ask"})
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.setattr("socket.getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models_dir)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    monkeypatch.setattr(mm, "_hf_repo_mmproj_filename", lambda *a, **k: None)
    state = SimpleNamespace(
        models_dir=models_dir, revisions={"4.0bpw": COMMIT, "2.0bpw": OTHER_COMMIT},
        default_commit="c" * 40, info_error=None, info_calls=[], sha_calls=[],
        downloads=[], snapshots=[], digests={}, snapshot_error=None,
        repo_files={COMMIT: {"config.json": b"{}", "w.safetensors": BODY},
                    OTHER_COMMIT: {"config.json": b'{"bits": 2}', "w.safetensors": BODY + b"x"},
                    "c" * 40: {"README.md": b"readme"}},
        missing_error=lambda rev: RevisionNotFoundError(f"no {rev}", response=_response(404)))

    def fake_sha(repo, filename, token=None, revision=None):
        state.sha_calls.append((repo, filename, revision))
        digest = state.digests.get(filename)
        return pull_mod._PublishedDigest(digest) if digest else None

    def fake_download(repo_id, filename, local_dir, revision=None, **kw):
        state.downloads.append((repo_id, filename, revision))
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(BODY)
        return str(p)

    def fake_snapshot(repo_id, local_dir, revision=None, **kw):
        state.snapshots.append((repo_id, revision))
        for name, data in state.repo_files[revision or state.default_commit].items():
            p = Path(local_dir) / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            if state.snapshot_error is not None:
                raise state.snapshot_error
        return local_dir

    def fake_head(url, allow_redirects=None, timeout=None):
        h = MagicMock()
        h.headers = {"content-length": "1"}
        return h

    import huggingface_hub
    monkeypatch.setattr(pull_mod, "_hf_api", lambda: _FakeApi(state))
    monkeypatch.setattr(mm, "_hf_file_sha256", fake_sha)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(pull_mod, "_policy_head", fake_head)
    monkeypatch.setattr(pull_mod, "_hf_pipeline_tag_to_type", lambda repo: "llm")
    return state


class TestResolveHfRevision:
    def test_a_branch_resolves_to_its_commit(self, hub):
        rev = pull_mod.resolve_hf_revision("o/r", "4.0bpw")
        assert rev == pull_mod.HfRevision("4.0bpw", COMMIT)
        assert hub.info_calls == [("o/r", "4.0bpw", False)]

    def test_a_missing_revision_is_reported_as_missing(self, hub):
        with pytest.raises(pull_mod.HfRevisionError) as e:
            pull_mod.resolve_hf_revision("o/r", "nope")
        assert e.value.kind == "missing"
        assert "o/r" in str(e.value) and "nope" in str(e.value)

    @pytest.mark.parametrize("make, kind", [
        (lambda: __import__("huggingface_hub.errors", fromlist=["x"]).RepositoryNotFoundError(
            "gone", response=_response(404)), "repo"),
        (lambda: __import__("huggingface_hub.errors", fromlist=["x"]).GatedRepoError(
            "gated", response=_response(403)), "denied"),
        (lambda: __import__("huggingface_hub.errors", fromlist=["x"]).HfHubHTTPError(
            "denied", response=_response(401)), "denied"),
        (lambda: __import__("huggingface_hub.errors", fromlist=["x"]).HfHubHTTPError(
            "boom", response=_response(503)), "unreachable"),
        (lambda: httpx.ConnectError("no route"), "unreachable"),
        (lambda: OSError("offline"), "unreachable"),
    ])
    def test_a_failure_that_is_not_a_missing_revision_is_never_called_one(
            self, hub, make, kind):
        hub.info_error = make()
        with pytest.raises(pull_mod.HfRevisionError) as e:
            pull_mod.resolve_hf_revision("o/r", "4.0bpw")
        assert e.value.kind == kind
        assert "has no branch, tag or commit" not in str(e.value)


class TestPullAtARevision:
    def test_a_single_file_is_read_and_downloaded_at_the_revisions_commit(self, hub):
        assert mm.pull_model("o/r@4.0bpw:w.safetensors", name="m", model_type="llm") is True
        assert hub.info_calls[0] == ("o/r", "4.0bpw", False)
        assert hub.sha_calls == [("o/r", "w.safetensors", COMMIT)]
        assert hub.downloads == [("o/r", "w.safetensors", COMMIT)]
        assert (hub.models_dir / "w.safetensors").read_bytes() == BODY

    def test_the_registry_entry_records_the_revision_and_the_commit(self, hub):
        assert mm.pull_model("o/r@4.0bpw:w.safetensors", name="m", model_type="llm") is True
        entry = mm.load_registry()["m"]
        assert entry["source"] == "hf:o/r"
        assert entry["revision"] == "4.0bpw" and entry["commit"] == COMMIT

    def test_a_pull_of_the_default_branch_records_no_revision(self, hub):
        assert mm.pull_model("o/r:w.safetensors", name="m", model_type="llm") is True
        entry = mm.load_registry()["m"]
        assert "revision" not in entry and "commit" not in entry
        assert hub.info_calls == []
        assert hub.downloads == [("o/r", "w.safetensors", None)]

    def test_list_shows_the_revision_beside_the_source(self, hub, capsys):
        mm.pull_model("o/r@4.0bpw:w.safetensors", name="m", model_type="llm")
        capsys.readouterr()
        mm.list_models()
        out = " ".join(capsys.readouterr().out.split())
        assert "hf:o/r@4.0bpw" in out

    def test_a_file_without_a_published_digest_is_still_read_at_the_revision(self, hub):
        mm.pull_model("o/r@4.0bpw:w.safetensors", name="m", model_type="llm")
        assert all(rev == COMMIT for _repo, _name, rev in hub.downloads)

    def test_a_missing_revision_is_refused_before_any_download(self, hub, capsys):
        assert mm.pull_model("o/r@nope:w.safetensors", name="m", model_type="llm") is False
        out = " ".join(capsys.readouterr().out.split())
        assert "o/r@nope" in out and "no branch, tag or commit named 'nope'" in out
        assert hub.downloads == [] and hub.sha_calls == [] and hub.snapshots == []
        assert "m" not in mm.load_registry()

    def test_a_full_repo_at_a_missing_revision_is_refused_before_any_download(
            self, hub, capsys):
        assert mm.pull_model("o/r@nope", model_type="llm") is False
        assert "no branch, tag or commit named 'nope'" in " ".join(capsys.readouterr().out.split())
        assert hub.snapshots == [] and hub.downloads == []

    def test_an_unreachable_hub_is_not_reported_as_a_missing_revision(self, hub, capsys):
        hub.info_error = httpx.ConnectError("no route")
        assert mm.pull_model("o/r@4.0bpw:w.safetensors", name="m", model_type="llm") is False
        out = " ".join(capsys.readouterr().out.split())
        assert "could not reach Hugging Face" in out and "no branch, tag or commit" not in out
        assert hub.downloads == []

    def test_a_malformed_revision_is_refused_without_touching_the_hub(self, hub, capsys):
        assert mm.pull_model("o/r@rev/w.gguf", model_type="llm") is False
        assert "Invalid model spec" in capsys.readouterr().out
        assert hub.info_calls == [] and hub.downloads == []

    def test_a_revision_in_mmproj_is_refused(self, hub, capsys):
        assert mm.pull_model("o/r@4.0bpw:w.safetensors", name="m", model_type="llm",
                             mmproj_spec="o/p@x:proj.gguf") is False
        assert "--mmproj does not take a revision" in capsys.readouterr().out
        assert hub.downloads == []


class TestFullRepoAtARevision:
    def test_the_listing_and_the_download_use_the_commit(self, hub):
        assert mm.pull_model("o/r@4.0bpw", model_type="llm") is True
        assert hub.snapshots == [("o/r", COMMIT)]
        listing = [c for c in hub.info_calls if c[2]]
        assert listing and all(rev == COMMIT for _r, rev, _f in listing)

    def test_the_default_folder_name_carries_the_revision(self, hub):
        mm.pull_model("o/r@4.0bpw", model_type="llm")
        assert (hub.models_dir / "r@4.0bpw".replace("@", "-")).is_dir()
        assert (hub.models_dir / "r-4.0bpw" / "w.safetensors").read_bytes() == BODY

    def test_the_registry_entry_records_the_revision(self, hub):
        mm.pull_model("o/r@4.0bpw", name="exl", model_type="llm")
        entry = mm.load_registry()["exl"]
        assert (entry["source"], entry["revision"], entry["commit"]) == (
            "hf:o/r", "4.0bpw", COMMIT)

    def test_two_revisions_of_one_repo_are_two_models(self, hub):
        assert mm.pull_model("o/r@4.0bpw", model_type="llm") is True
        assert mm.pull_model("o/r@2.0bpw", model_type="llm") is True
        reg = mm.load_registry()
        assert {n: e["revision"] for n, e in reg.items()} == {
            "r-4.0bpw": "4.0bpw", "r-2.0bpw": "2.0bpw"}
        assert (hub.models_dir / "r-2.0bpw" / "w.safetensors").read_bytes() == BODY + b"x"
        assert (hub.models_dir / "r-4.0bpw" / "w.safetensors").read_bytes() == BODY

    def test_a_second_revision_is_not_called_already_downloaded(self, hub, capsys):
        mm.pull_model("o/r@4.0bpw", model_type="llm")
        capsys.readouterr()
        mm.pull_model("o/r@2.0bpw", model_type="llm")
        out = capsys.readouterr().out
        assert "already downloaded" not in out.lower() and "Already downloaded" not in out

    def test_the_default_branch_and_a_revision_do_not_share_a_folder(self, hub):
        hub.repo_files["c" * 40] = {"config.json": b"{}", "main.bin": b"m"}
        assert mm.pull_model("o/r", model_type="llm") is True
        assert mm.pull_model("o/r@4.0bpw", model_type="llm") is True
        assert not (hub.models_dir / "r" / "w.safetensors").exists()
        assert (hub.models_dir / "r-4.0bpw" / "w.safetensors").exists()

    def test_a_repull_of_the_same_revision_is_a_no_op_that_keeps_the_record(
            self, hub, capsys):
        mm.pull_model("o/r@4.0bpw", name="exl", model_type="llm")
        hub.snapshots.clear()
        capsys.readouterr()
        assert mm.pull_model("o/r@4.0bpw", name="exl", model_type="llm") is True
        assert "Already downloaded" in capsys.readouterr().out
        assert hub.snapshots == []
        assert mm.load_registry()["exl"]["revision"] == "4.0bpw"

    def test_redownload_fetches_the_same_revision_again(self, hub):
        mm.pull_model("o/r@4.0bpw", name="exl", model_type="llm")
        hub.snapshots.clear()
        mm.pull_model("o/r@4.0bpw", name="exl", model_type="llm", redownload=True)
        entry = mm.load_registry()["exl"]
        assert entry["revision"] == "4.0bpw" and entry["commit"] == COMMIT

    def test_a_listing_failure_does_not_pass_another_revisions_files_off_as_complete(
            self, hub, capsys, monkeypatch):
        assert mm.pull_model("o/r@4.0bpw", name="exl", model_type="llm") is True
        capsys.readouterr()
        real = _FakeApi.model_info

        def listing_fails(self, repo_id, revision=None, files_metadata=False, **kw):
            if files_metadata:
                raise httpx.ConnectError("dropped")
            return real(self, repo_id, revision, files_metadata, **kw)

        monkeypatch.setattr(_FakeApi, "model_info", listing_fails)
        hub.snapshots.clear()
        assert mm.pull_model("o/r@2.0bpw", name="exl", model_type="llm") is False
        assert "Already downloaded" not in capsys.readouterr().out
        assert hub.snapshots == []
        assert mm.load_registry()["exl"]["revision"] == "4.0bpw"

    def test_one_name_cannot_hold_two_revisions(self, hub, capsys):
        assert mm.pull_model("o/r@4.0bpw", name="exl", model_type="llm") is True
        capsys.readouterr()
        assert mm.pull_model("o/r@2.0bpw", name="exl", model_type="llm") is False
        assert "already holds a DIFFERENT model" in " ".join(capsys.readouterr().out.split())
        assert mm.load_registry()["exl"]["revision"] == "4.0bpw"


class TestSingleFileRecords:
    def _digest_for_body(self, hub):
        hub.digests["w.safetensors"] = hashlib.sha256(BODY).hexdigest()

    def test_the_same_file_pulled_again_at_a_revision_takes_the_revision_record(self, hub):
        self._digest_for_body(hub)
        assert mm.pull_model("o/r:w.safetensors", name="m", model_type="llm") is True
        assert "revision" not in mm.load_registry()["m"]
        assert mm.pull_model("o/r@4.0bpw:w.safetensors", name="m", model_type="llm") is True
        entry = mm.load_registry()["m"]
        assert (entry["revision"], entry["commit"]) == ("4.0bpw", COMMIT)

    def test_the_same_file_pulled_again_without_a_revision_drops_the_record(self, hub):
        self._digest_for_body(hub)
        assert mm.pull_model("o/r@4.0bpw:w.safetensors", name="m", model_type="llm") is True
        assert mm.pull_model("o/r:w.safetensors", name="m", model_type="llm") is True
        assert "revision" not in mm.load_registry()["m"]

    def test_a_file_this_revision_already_owns_is_fetched_again_without_a_digest(self, hub):
        assert mm.pull_model("o/r@4.0bpw:w.safetensors", name="a", model_type="llm") is True
        hub.downloads.clear()
        assert mm.pull_model("o/r@4.0bpw:w.safetensors", name="a", model_type="llm") is True
        assert hub.downloads == [("o/r", "w.safetensors", COMMIT)]

    def test_an_unregistered_same_named_file_is_fetched_again_at_the_revision(self, hub):
        (hub.models_dir / "w.safetensors").write_bytes(b"stray")
        assert mm.pull_model("o/r@4.0bpw:w.safetensors", name="b", model_type="llm") is True
        assert hub.downloads == [("o/r", "w.safetensors", COMMIT)]
        assert (hub.models_dir / "w.safetensors").read_bytes() == BODY


    def test_a_starter_pin_is_not_applied_to_a_revision(self, hub, monkeypatch):
        asked = []
        monkeypatch.setattr(mm, "starter_model_sha256", lambda spec: asked.append(spec) or "0" * 64)
        assert mm.pull_model("o/r@4.0bpw:w.safetensors", name="m", model_type="llm") is True
        assert asked == []

    def test_an_entry_pulled_at_a_revision_is_not_offered_a_projector_fetch(self, hub):
        from localm.model_manager import registry
        assert mm.pull_model("o/r:w.safetensors", name="plain", model_type="llm") is True
        path = str(hub.models_dir / "w.safetensors")
        assert registry._active_model_missing_mmproj(path) == ("plain", "o/r")
        assert mm.pull_model("o/r@4.0bpw:w.safetensors", name="plain", model_type="llm") is True
        assert registry._active_model_missing_mmproj(path) is None


class TestASameNamedFileIsNeverOverwrittenByAnotherPull:
    OTHER = b"another model's weights"

    def _own(self, hub, name, source, **record):
        path = hub.models_dir / "w.safetensors"
        path.write_bytes(self.OTHER)
        mm._register(name, path, source, **record)
        return path

    def _refused(self, hub, capsys, spec, name="b"):
        capsys.readouterr()
        assert mm.pull_model(spec, name=name, model_type="llm") is False
        out = " ".join(capsys.readouterr().out.split())
        assert "File name already in use" in out
        assert hub.downloads == []
        assert (hub.models_dir / "w.safetensors").read_bytes() == self.OTHER
        assert name not in mm.load_registry()
        return out

    def test_a_file_registered_to_another_repo_is_not_replaced(self, hub, capsys):
        self._own(hub, "a", "hf:x/y")
        self._refused(hub, capsys, "o/r@4.0bpw:w.safetensors")
        assert mm.load_registry()["a"]["source"] == "hf:x/y"

    def test_a_file_registered_from_the_default_branch_is_not_replaced(self, hub, capsys):
        self._own(hub, "a", "hf:o/r")
        self._refused(hub, capsys, "o/r@4.0bpw:w.safetensors")

    def test_a_file_registered_at_another_revision_is_not_replaced(self, hub, capsys):
        self._own(hub, "a", "hf:o/r", revision="2.0bpw", commit=OTHER_COMMIT)
        self._refused(hub, capsys, "o/r@4.0bpw:w.safetensors")
        assert mm.load_registry()["a"]["revision"] == "2.0bpw"

    def test_the_refusal_names_the_owner_and_how_to_free_the_name(self, hub, capsys):
        self._own(hub, "a", "hf:o/r", revision="2.0bpw", commit=OTHER_COMMIT)
        out = self._refused(hub, capsys, "o/r@4.0bpw:w.safetensors")
        assert "'a' (hf:o/r@2.0bpw)" in out and "localm rm a" in out

    def test_the_same_name_is_refused_as_well(self, hub, capsys):
        self._own(hub, "a", "hf:x/y")
        capsys.readouterr()
        assert mm.pull_model("o/r@4.0bpw:w.safetensors", name="a", model_type="llm") is False
        assert "File name already in use" in " ".join(capsys.readouterr().out.split())
        assert hub.downloads == []
        assert mm.load_registry()["a"]["source"] == "hf:x/y"

    def test_the_default_branch_does_not_relabel_a_revisions_file(self, hub, capsys):
        self._own(hub, "a", "hf:o/r", revision="2.0bpw", commit=OTHER_COMMIT)
        self._refused(hub, capsys, "o/r:w.safetensors")
        assert mm.load_registry()["a"]["revision"] == "2.0bpw"

    def test_the_default_branch_refetches_a_revisions_file_it_replaces_by_name(self, hub):
        self._own(hub, "a", "hf:o/r", revision="2.0bpw", commit=OTHER_COMMIT)
        assert mm.pull_model("o/r:w.safetensors", name="a", model_type="llm") is True
        assert hub.downloads == [("o/r", "w.safetensors", None)]
        assert "revision" not in mm.load_registry()["a"]

    def test_every_undigested_part_of_a_split_file_is_fetched_again_with_the_owned_one(self, hub):
        for n in (1, 2):
            (hub.models_dir / f"m-0000{n}-of-00002.gguf").write_bytes(self.OTHER)
        mm._register("a", hub.models_dir / "m-00001-of-00002.gguf", "hf:o/r",
                     revision="2.0bpw", commit=OTHER_COMMIT)
        mm.pull_model("o/r:m-00001-of-00002.gguf", name="a", model_type="llm")
        assert sorted(f for _r, f, _v in hub.downloads) == [
            "m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]

    def test_a_split_file_whose_first_part_is_another_models_is_not_torn_down(self, hub, capsys):
        first = hub.models_dir / "m-00001-of-00002.gguf"
        first.write_bytes(self.OTHER)
        mm._register("a", first, "hf:x/y")
        hub.digests["m-00001-of-00002.gguf"] = hashlib.sha256(BODY).hexdigest()
        capsys.readouterr()
        assert mm.pull_model("o/r:m-00001-of-00002.gguf", name="b", model_type="llm") is False
        assert "File name already in use" in " ".join(capsys.readouterr().out.split())
        assert hub.downloads == []
        assert first.read_bytes() == self.OTHER
        assert mm.load_registry()["a"]["source"] == "hf:x/y"

    def test_a_revision_replaces_its_own_name_from_the_default_branch(self, hub):
        self._own(hub, "a", "hf:o/r")
        assert mm.pull_model("o/r@4.0bpw:w.safetensors", name="a", model_type="llm") is True
        assert hub.downloads == [("o/r", "w.safetensors", COMMIT)]
        assert mm.load_registry()["a"]["revision"] == "4.0bpw"

    def test_a_default_branch_file_without_a_digest_is_still_trusted(self, hub):
        self._own(hub, "a", "hf:o/r")
        assert mm.pull_model("o/r:w.safetensors", name="a", model_type="llm") is True
        assert hub.downloads == []


class _InterruptedAfterTheFirstFile:
    """An unfinished snapshot: the first file of the listing is on disk."""

    def stop(self, hub, spec, name="q"):
        hub.snapshot_error = OSError("disk full")
        assert mm.pull_model(spec, name=name, model_type="llm") is False
        hub.snapshot_error = None
        hub.snapshots.clear()


class TestAnUnfinishedRevisionIsNotMergedWithAnother(_InterruptedAfterTheFirstFile):
    def test_another_revision_into_the_same_folder_is_refused(self, hub, capsys):
        self.stop(hub, "o/r@4.0bpw")
        capsys.readouterr()
        assert mm.pull_model("o/r@2.0bpw", name="q", model_type="llm") is False
        assert "unfinished download of another revision" in " ".join(
            capsys.readouterr().out.split())
        assert hub.snapshots == [] and "q" not in mm.load_registry()

    def test_the_first_revisions_file_is_left_untouched_by_the_refusal(self, hub):
        self.stop(hub, "o/r@4.0bpw")
        mm.pull_model("o/r@2.0bpw", name="q", model_type="llm")
        assert (hub.models_dir / "q" / "config.json").read_bytes() == b"{}"

    def test_the_default_branch_into_that_folder_is_refused_too(self, hub, capsys):
        self.stop(hub, "o/r@4.0bpw")
        hub.repo_files["c" * 40] = {"config.json": b'{"main": true}'}
        capsys.readouterr()
        assert mm.pull_model("o/r", name="q", model_type="llm") is False
        assert "unfinished download of another revision" in " ".join(
            capsys.readouterr().out.split())
        assert hub.snapshots == []

    def test_a_revision_into_an_unfinished_default_branch_folder_is_refused(self, hub, capsys):
        hub.repo_files["c" * 40] = {"README.md": b"readme", "main.bin": b"m"}
        self.stop(hub, "o/r")
        capsys.readouterr()
        assert mm.pull_model("o/r@4.0bpw", name="q", model_type="llm") is False
        assert "unfinished download" in " ".join(capsys.readouterr().out.split())
        assert hub.snapshots == [] and "q" not in mm.load_registry()

    def test_another_repos_unfinished_download_is_not_merged_either(self, hub, capsys):
        self.stop(hub, "o/r@4.0bpw")
        capsys.readouterr()
        assert mm.pull_model("p/s@4.0bpw", name="q", model_type="llm") is False
        assert "unfinished download" in " ".join(capsys.readouterr().out.split())
        assert hub.snapshots == []

    def test_the_same_revision_resumes_in_that_folder(self, hub):
        self.stop(hub, "o/r@4.0bpw")
        assert mm.pull_model("o/r@4.0bpw", name="q", model_type="llm") is True
        assert mm.load_registry()["q"]["revision"] == "4.0bpw"

    def test_the_same_revision_repairs_a_damaged_file_in_that_folder(self, hub):
        self.stop(hub, "o/r@4.0bpw")
        (hub.models_dir / "q" / "config.json").write_bytes(b"torn")
        assert mm.pull_model("o/r@4.0bpw", name="q", model_type="llm") is True
        assert (hub.models_dir / "q" / "config.json").read_bytes() == b"{}"

    def test_an_unreadable_marker_is_reported_and_counts_as_no_claim(self, hub, caplog):
        self.stop(hub, "o/r@4.0bpw")
        (hub.models_dir / "q" / ".localm-revision.json").write_text("{not json", encoding="utf-8")
        with caplog.at_level("WARNING"):
            assert mm.pull_model("o/r@2.0bpw", name="q", model_type="llm") is True
        assert ".localm-revision.json" in caplog.text

    def _unfinished_without_a_marker(self, hub):
        hub.repo_files["c" * 40] = {"README.md": b"readme", "main.bin": b"m"}
        self.stop(hub, "o/r")
        (hub.models_dir / "q" / ".localm-revision.json").unlink()

    def test_a_revision_into_an_earlier_localms_unfinished_folder_is_refused(self, hub, capsys):
        self._unfinished_without_a_marker(hub)
        capsys.readouterr()
        assert mm.pull_model("o/r@4.0bpw", name="q", model_type="llm") is False
        out = " ".join(capsys.readouterr().out.split())
        assert "unfinished download of an earlier localm" in out
        assert hub.snapshots == [] and "q" not in mm.load_registry()

    def test_the_default_branch_still_resumes_an_earlier_localms_folder(self, hub):
        self._unfinished_without_a_marker(hub)
        assert mm.pull_model("o/r", name="q", model_type="llm") is True
        assert "q" in mm.load_registry()

    def test_an_empty_folder_without_a_marker_is_free_for_a_revision(self, hub):
        (hub.models_dir / "q" / ".cache").mkdir(parents=True)
        assert mm.pull_model("o/r@4.0bpw", name="q", model_type="llm") is True

    def test_the_same_revision_resumes_with_the_repo_in_another_letter_case(self, hub):
        self.stop(hub, "o/r@4.0bpw")
        assert mm.pull_model("O/R@4.0bpw", name="q", model_type="llm") is True

    def test_another_revision_in_another_letter_case_is_still_refused(self, hub, capsys):
        self.stop(hub, "o/r@4.0bpw")
        capsys.readouterr()
        assert mm.pull_model("O/R@2.0bpw", name="q", model_type="llm") is False
        assert "unfinished download of another revision" in " ".join(
            capsys.readouterr().out.split())

    def test_a_folder_that_was_deleted_is_free_for_another_revision(self, hub):
        import shutil
        self.stop(hub, "o/r@4.0bpw")
        shutil.rmtree(hub.models_dir / "q")
        assert mm.pull_model("o/r@2.0bpw", name="q", model_type="llm") is True
        assert mm.load_registry()["q"]["revision"] == "2.0bpw"


class TestTheClaimMarkerEndsWithTheRegistration(_InterruptedAfterTheFirstFile):
    MARKER = ".localm-revision.json"

    def test_it_is_gone_once_the_revision_is_registered(self, hub):
        assert mm.pull_model("o/r@4.0bpw", name="q", model_type="llm") is True
        assert not (hub.models_dir / "q" / self.MARKER).exists()

    def test_it_is_gone_once_the_default_branch_is_registered(self, hub):
        assert mm.pull_model("o/r", name="q", model_type="llm") is True
        assert not (hub.models_dir / "q" / self.MARKER).exists()

    def test_it_stays_while_the_download_is_unfinished(self, hub):
        self.stop(hub, "o/r@4.0bpw")
        assert (hub.models_dir / "q" / self.MARKER).is_file()

    def test_it_is_removed_when_the_unfinished_download_is_finished(self, hub):
        self.stop(hub, "o/r@4.0bpw")
        assert mm.pull_model("o/r@4.0bpw", name="q", model_type="llm") is True
        assert not (hub.models_dir / "q" / self.MARKER).exists()

    def test_a_stale_marker_on_a_registered_folder_is_cleared_by_a_repull(self, hub, capsys):
        hub.repo_files["c" * 40] = {"config.json": b"{}"}
        assert mm.pull_model("o/r", name="q", model_type="llm") is True
        (hub.models_dir / "q" / self.MARKER).write_text(
            '{"repo": "o/r", "revision": "4.0bpw"}', encoding="utf-8")
        hub.snapshots.clear()
        capsys.readouterr()
        assert mm.pull_model("o/r", name="q", model_type="llm") is True
        assert "Already downloaded" in capsys.readouterr().out
        assert hub.snapshots == []
        assert not (hub.models_dir / "q" / self.MARKER).exists()

    def test_a_repull_of_a_revision_does_not_download_again(self, hub, capsys):
        assert mm.pull_model("o/r@4.0bpw", name="q", model_type="llm") is True
        hub.snapshots.clear()
        capsys.readouterr()
        assert mm.pull_model("o/r@4.0bpw", name="q", model_type="llm") is True
        assert "Already downloaded" in capsys.readouterr().out
        assert hub.snapshots == []


class TestSpecsWithAnAtSignAreNotRevisions:
    def test_a_url_with_userinfo_reaches_the_url_pull_whole(self, hub, monkeypatch):
        seen = []
        monkeypatch.setattr(pull_mod, "_pull_url", lambda url, *a, **k: seen.append(url) or True)
        url = "https://user@host.example/dir@v2/model.gguf"
        assert mm.pull_model(url) is True
        assert seen == [url]
        assert hub.info_calls == []

    def test_an_existing_local_file_with_an_at_sign_is_registered_in_place(
            self, hub, tmp_path, monkeypatch):
        folder = tmp_path / "dir@v2"
        folder.mkdir()
        gguf = folder / "m@2.gguf"
        gguf.write_bytes(b"GGUF")
        seen = []
        monkeypatch.setattr(mm, "add_local", lambda path, **k: seen.append(path) or True)
        assert mm.pull_model(str(gguf)) is True
        assert seen == [str(gguf)]
        assert hub.info_calls == []

    def test_a_file_name_with_an_at_sign_is_downloaded_as_named(self, hub):
        assert mm.pull_model("o/r:m@2.safetensors", name="m", model_type="llm") is True
        assert hub.downloads == [("o/r", "m@2.safetensors", None)]
        assert hub.info_calls == []


class TestPullPassesNothingNewForAnUnrevisionedSpec:
    def test_the_file_path_is_called_without_a_revision(self, hub, monkeypatch):
        seen = {}
        monkeypatch.setattr(mm, "_pull_gguf_file", lambda spec, name, **k: seen.update(k) or True)
        assert mm.pull_model("o/r:m.gguf") is True
        assert "revision" not in seen

    def test_the_snapshot_path_is_called_without_a_revision(self, hub, monkeypatch):
        seen = {}
        monkeypatch.setattr(mm, "_pull_hf_snapshot", lambda spec, name, **k: seen.update(k) or True)
        assert mm.pull_model("o/r") is True
        assert "revision" not in seen


class TestOtherConsumersOfASpec:
    def test_discarding_a_partial_download_ignores_the_revision(self, hub, monkeypatch):
        seen = []
        monkeypatch.setattr(pull_mod, "_delete_hf_temp_files",
                            lambda base, rels: seen.append((base, rels)) or 5)
        assert pull_mod.discard_partial_download("o/r@4.0bpw:dir/w.safetensors") == 5
        assert seen[0][1] == ["dir/w.safetensors"]

    def test_discarding_a_full_repo_at_a_revision_targets_the_revision_folder(
            self, hub, monkeypatch):
        seen = []
        monkeypatch.setattr(pull_mod, "_delete_hf_temp_files",
                            lambda base, rels: seen.append((base, rels)) or 0)
        pull_mod.discard_partial_download("o/r@4.0bpw")
        assert seen == [(hub.models_dir / "r-4.0bpw", None)]

    def test_a_full_repo_at_a_revision_is_still_a_repo_pull_for_the_discard_notice(self):
        assert pull_mod.partial_download_kept("o/r@4.0bpw") == "repo"
        assert pull_mod.partial_download_kept("o/r@4.0bpw:f.gguf") is None

    def test_an_entry_pulled_at_a_revision_is_not_a_projector_backfill_candidate(self):
        entry = {"source": "hf:o/r", "model_type": "llm"}
        assert pull_mod.mmproj_backfill_candidate(entry, Path("m.gguf")) is True
        assert pull_mod.mmproj_backfill_candidate(
            {**entry, "revision": "4.0bpw"}, Path("m.gguf")) is False

    def test_a_pull_at_a_revision_attaches_only_an_explicit_projector(self, monkeypatch):
        calls = []
        monkeypatch.setattr(pull_mod, "_maybe_fetch_repo_mmproj",
                            lambda *a, **k: calls.append(a) or Path("p.gguf"))
        monkeypatch.setattr(pull_mod, "_fetch_explicit_mmproj",
                            lambda *a, **k: Path("explicit.gguf"))
        args = ("llm", "o/r", "m.gguf", Path("d"), None)
        assert pull_mod._mmproj_for_registration(*args, None, pinned_revision=True) is None
        assert calls == []
        assert pull_mod._mmproj_for_registration(
            *args, "o/p:x.gguf", pinned_revision=True) == Path("explicit.gguf")
        assert pull_mod._mmproj_for_registration(*args, None) == Path("p.gguf")


class TestMcpPullStaysOnTheDefaultBranch:
    def _handler(self):
        from localm.plugins.mcpserver.tools import models as tools
        return tools.build(SimpleNamespace(get_loaded=lambda n: None))["pull_model"]["handler"]

    def test_a_registered_repo_is_not_pullable_at_a_revision_over_mcp(self, hub, monkeypatch):
        mm.pull_model("o/r:w.safetensors", name="first", model_type="llm")
        pulled = []
        monkeypatch.setattr(pull_mod, "pull_model", lambda spec, **k: pulled.append(spec) or True)
        result = self._handler()({"repo": "o/r@refs/pr/3", "file": "w.safetensors",
                                  "name": "second", "load": False})
        assert result["isError"] is True and "Refusing to pull" in str(result)
        assert pulled == []

    def test_a_registered_repo_is_still_pullable_over_mcp(self, hub, monkeypatch):
        mm.pull_model("o/r:w.safetensors", name="first", model_type="llm")
        pulled = []
        monkeypatch.setattr(pull_mod, "pull_model", lambda spec, **k: pulled.append(spec) or True)
        result = self._handler()({"repo": "o/r", "file": "w.safetensors",
                                  "name": "second", "load": False})
        assert not result["isError"], result
        assert pulled == ["o/r:w.safetensors"]
