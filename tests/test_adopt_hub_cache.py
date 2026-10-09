# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm add` over a Hugging Face hub cache registers one model per repository
from the snapshot folder, never from ``blobs/``, in both the symlink layout
(POSIX, Windows with symlinks) and the plain-copy layout (Windows without)."""

import hashlib
import os
import time
from pathlib import Path

import pytest

from localm.config import load_registry
from localm.model_manager import add_local
from localm.model_manager.gguf import (
    hub_repo_id,
    hub_snapshot_repo_id,
    logical_model_path,
    scan_hub_cache,
)

_REV_A = "a" * 40
_REV_B = "b" * 40


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    import localm.model_manager as mm
    (home / "models").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mm, "MODELS_DIR", home / "models")
    monkeypatch.setattr(mm, "ensure_dirs", lambda: (home / "models").mkdir(parents=True, exist_ok=True))
    return home


@pytest.fixture(params=["links", "copies"])
def layout(request):
    if request.param == "links":
        probe_dir = Path(os.environ.get("TMP", ".")) / f"symprobe-{os.getpid()}"
        probe_dir.mkdir(exist_ok=True)
        try:
            (probe_dir / "t").write_bytes(b"x")
            os.symlink(probe_dir / "t", probe_dir / "l")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are not available here")
        finally:
            for f in ("l", "t"):
                try:
                    (probe_dir / f).unlink()
                except OSError:
                    pass
            try:
                probe_dir.rmdir()
            except OSError:
                pass
    return request.param


def _gguf_bytes(tag: str) -> bytes:
    return b"GGUF\x00\x00\x00\x00" + tag.encode()


def _repo(root: Path, layout: str, repo_id: str, revision: str, files: dict,
          *, main=True, mtime=None) -> Path:
    """Write one revision of a hub-cache repo: *files* maps a snapshot-relative
    path to bytes. In the link layout each file is a relative symlink into
    ``blobs/<sha>``; in the copy layout it is a plain file and ``blobs`` is empty."""
    repo = root / ("models--" + repo_id.replace("/", "--"))
    (repo / "blobs").mkdir(parents=True, exist_ok=True)
    (repo / "refs").mkdir(exist_ok=True)
    snap = repo / "snapshots" / revision
    for rel, data in files.items():
        target = snap / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if layout == "links":
            blob = repo / "blobs" / hashlib.sha256(data).hexdigest()
            blob.write_bytes(data)
            os.symlink(os.path.relpath(blob, target.parent), target)
        else:
            target.write_bytes(data)
    if main:
        (repo / "refs" / "main").write_text(revision)
    if mtime is not None:
        os.utime(snap, (mtime, mtime))
    return snap


_HF_FILES = {
    "config.json": b'{"model_type": "llama"}',
    "tokenizer.json": b"{}",
    "model.safetensors": b"\0" * 64,
}


def _cache(tmp_path, layout) -> Path:
    root = tmp_path / "hub"
    root.mkdir()
    _repo(root, layout, "org/alpha-GGUF", _REV_A, {"alpha.Q4_K_M.gguf": _gguf_bytes("alpha")})
    _repo(root, layout, "org/beta", _REV_A, _HF_FILES)
    return root


class TestHubNames:
    def test_repo_id_from_folder_name(self):
        assert hub_repo_id("models--org--name") == "org/name"
        assert hub_repo_id("models--gpt2") == "gpt2"
        assert hub_repo_id("models--") is None
        assert hub_repo_id("snapshots") is None
        assert hub_repo_id("modelsXorgXname") is None

    def test_snapshot_repo_id(self, tmp_path):
        assert hub_snapshot_repo_id(tmp_path / "models--o--n" / "snapshots" / _REV_A) == "o/n"
        assert hub_snapshot_repo_id(tmp_path / "plain" / "snapshots" / _REV_A) is None


class TestHubCacheRoot:
    def test_two_repos_register_from_snapshots(self, tmp_path, isolated_home, layout):
        root = _cache(tmp_path, layout)
        assert add_local(str(root)) is True
        reg = load_registry()
        assert set(reg) == {"org-alpha-GGUF", "org-beta"}
        gguf = Path(reg["org-alpha-GGUF"]["path"])
        assert gguf == (root / "models--org--alpha-GGUF" / "snapshots"
                        / _REV_A).resolve() / "alpha.Q4_K_M.gguf"
        assert gguf.suffix == ".gguf"
        hf = Path(reg["org-beta"]["path"])
        assert hf == (root / "models--org--beta" / "snapshots" / _REV_A).resolve()
        assert all("blobs" not in Path(e["path"]).parts for e in reg.values())

    def test_hf_home_folder_resolves_to_its_hub(self, tmp_path, isolated_home, layout):
        hf_home = tmp_path / "hf"
        hf_home.mkdir()
        root = hf_home / "hub"
        root.mkdir()
        _repo(root, layout, "org/alpha-GGUF", _REV_A, {"a.gguf": _gguf_bytes("a")})
        assert add_local(str(hf_home)) is True
        assert set(load_registry()) == {"org-alpha-GGUF"}

    def test_single_repo_folder_and_name_override(self, tmp_path, isolated_home, layout):
        root = _cache(tmp_path, layout)
        assert add_local(str(root / "models--org--alpha-GGUF"), name="mine") is True
        assert set(load_registry()) == {"mine"}

    def test_name_is_not_applied_to_many_repos(self, tmp_path, isolated_home, layout):
        root = _cache(tmp_path, layout)
        assert add_local(str(root), name="mine") is True
        assert set(load_registry()) == {"org-alpha-GGUF", "org-beta"}

    def test_several_ggufs_in_one_repo_keep_their_stems(self, tmp_path, isolated_home, layout):
        root = tmp_path / "hub"
        root.mkdir()
        _repo(root, layout, "org/multi", _REV_A,
              {"multi-Q4.gguf": _gguf_bytes("q4"), "multi-Q8.gguf": _gguf_bytes("q8")})
        assert add_local(str(root)) is True
        assert set(load_registry()) == {"multi-Q4", "multi-Q8"}

    def test_registered_twice_is_idempotent(self, tmp_path, isolated_home, layout):
        root = _cache(tmp_path, layout)
        assert add_local(str(root), on_duplicate="skip") is True
        assert add_local(str(root), on_duplicate="skip") is True
        assert set(load_registry()) == {"org-alpha-GGUF", "org-beta"}


class TestRevisions:
    def test_refs_main_picks_the_revision_even_when_older(self, tmp_path, isolated_home, layout):
        root = tmp_path / "hub"
        root.mkdir()
        now = time.time()
        _repo(root, layout, "org/m", _REV_A, {"m-old.gguf": _gguf_bytes("old")}, mtime=now - 500)
        _repo(root, layout, "org/m", _REV_B, {"m-new.gguf": _gguf_bytes("new")},
              main=False, mtime=now)
        (root / "models--org--m" / "refs" / "main").write_text(_REV_A)
        assert add_local(str(root)) is True
        reg = load_registry()
        assert len(reg) == 1
        assert Path(next(iter(reg.values()))["path"]).name == "m-old.gguf"

    def test_newest_snapshot_when_refs_main_is_absent(self, tmp_path, isolated_home, layout):
        root = tmp_path / "hub"
        root.mkdir()
        now = time.time()
        _repo(root, layout, "org/m", _REV_A, {"m-old.gguf": _gguf_bytes("old")},
              main=False, mtime=now - 500)
        _repo(root, layout, "org/m", _REV_B, {"m-new.gguf": _gguf_bytes("new")},
              main=False, mtime=now)
        assert add_local(str(root)) is True
        reg = load_registry()
        assert len(reg) == 1
        assert Path(next(iter(reg.values()))["path"]).name == "m-new.gguf"

    def test_dangling_refs_main_falls_back_to_newest(self, tmp_path, isolated_home, layout):
        root = tmp_path / "hub"
        root.mkdir()
        _repo(root, layout, "org/m", _REV_A, {"m.gguf": _gguf_bytes("m")}, main=False)
        (root / "models--org--m" / "refs" / "main").write_text("c" * 40)
        assert add_local(str(root)) is True
        assert set(load_registry()) == {"org-m"}


class TestNeverRegisters:
    def test_incomplete_blob_only_repo_registers_nothing(self, tmp_path, isolated_home):
        root = tmp_path / "hub"
        repo = root / "models--org--partial"
        (repo / "blobs").mkdir(parents=True)
        (repo / "refs").mkdir()
        (repo / "blobs" / ("d" * 64 + ".incomplete")).write_bytes(_gguf_bytes("half"))
        (repo / "refs" / "main").write_text(_REV_A)
        assert add_local(str(root)) is False
        assert load_registry() == {}

    def test_incomplete_blob_beside_a_good_repo_is_ignored(self, tmp_path, isolated_home, layout):
        root = _cache(tmp_path, layout)
        partial = root / "models--org--partial"
        (partial / "blobs").mkdir(parents=True)
        (partial / "blobs" / ("d" * 64 + ".incomplete")).write_bytes(_gguf_bytes("half"))
        # A good repo's own half-finished download sits next to its finished blob.
        (root / "models--org--alpha-GGUF" / "blobs" / ("e" * 64 + ".incomplete")
         ).write_bytes(_gguf_bytes("half"))
        (root / "models--org--alpha-GGUF" / ".no_exist" / _REV_A).mkdir(parents=True)
        (root / "models--org--alpha-GGUF" / ".no_exist" / _REV_A / "x.gguf").write_bytes(b"")
        assert add_local(str(root)) is True
        reg = load_registry()
        assert set(reg) == {"org-alpha-GGUF", "org-beta"}
        for entry in reg.values():
            parts = Path(entry["path"]).parts
            assert "blobs" not in parts and ".no_exist" not in parts
            assert not entry["path"].endswith(".incomplete")

    def test_refs_only_repo_is_skipped_not_fatal(self, tmp_path, isolated_home, layout):
        root = _cache(tmp_path, layout)
        (root / "models--org--meta" / "refs").mkdir(parents=True)
        (root / "models--org--meta" / "refs" / "main").write_text(_REV_A)
        assert add_local(str(root)) is True
        assert set(load_registry()) == {"org-alpha-GGUF", "org-beta"}

    def test_dangling_link_is_not_a_model(self, tmp_path, isolated_home, layout):
        if layout != "links":
            pytest.skip("only the link layout has links")
        root = tmp_path / "hub"
        root.mkdir()
        snap = _repo(root, layout, "org/m", _REV_A, {"m.gguf": _gguf_bytes("m")})
        for blob in (root / "models--org--m" / "blobs").iterdir():
            blob.unlink()
        assert (snap / "m.gguf").is_symlink() and not (snap / "m.gguf").exists()
        assert add_local(str(root)) is False
        assert load_registry() == {}


class TestOrdinaryFoldersAreNotCaches:
    def test_folder_named_snapshots_is_walked_normally(self, tmp_path, isolated_home):
        d = tmp_path / "plain"
        (d / "snapshots" / _REV_A).mkdir(parents=True)
        (d / "snapshots" / _REV_A / "loose.gguf").write_bytes(_gguf_bytes("loose"))
        assert scan_hub_cache(d) is None
        assert add_local(str(d)) is True
        assert set(load_registry()) == {"loose"}

    def test_deep_gguf_under_plain_folders_stays_beyond_the_depth(self, tmp_path, isolated_home):
        d = tmp_path / "plain"
        deep = d / "a" / "snapshots" / _REV_A / "sub"
        deep.mkdir(parents=True)
        (deep / "deep.gguf").write_bytes(_gguf_bytes("deep"))
        assert add_local(str(d)) is False
        assert load_registry() == {}

    def test_models_prefix_without_cache_folders_is_not_a_repo(self, tmp_path, isolated_home):
        d = tmp_path / "plain"
        (d / "models--org--x").mkdir(parents=True)
        (d / "models--org--x" / "z.gguf").write_bytes(_gguf_bytes("z"))
        assert scan_hub_cache(d) is None
        assert add_local(str(d)) is True
        assert set(load_registry()) == {"z"}


class TestDirectPaths:
    def test_linked_gguf_added_directly_keeps_its_name(self, tmp_path, isolated_home, layout):
        root = tmp_path / "hub"
        root.mkdir()
        snap = _repo(root, layout, "org/m", _REV_A, {"m.Q4.gguf": _gguf_bytes("m")})
        assert add_local(str(snap / "m.Q4.gguf")) is True
        reg = load_registry()
        assert set(reg) == {"m.Q4"}
        assert reg["m.Q4"]["path"].endswith("m.Q4.gguf")
        assert "blobs" not in Path(reg["m.Q4"]["path"]).parts

    def test_snapshot_folder_added_directly_is_named_for_the_repo(self, tmp_path, isolated_home, layout):
        root = tmp_path / "hub"
        root.mkdir()
        snap = _repo(root, layout, "org/beta", _REV_A, _HF_FILES)
        assert add_local(str(snap)) is True
        assert set(load_registry()) == {"org-beta"}

    def test_logical_path_keeps_only_a_hub_link(self, tmp_path, layout):
        if layout != "links":
            pytest.skip("only the link layout has links")
        root = tmp_path / "hub"
        root.mkdir()
        snap = _repo(root, layout, "org/m", _REV_A, {"m.gguf": _gguf_bytes("m")})
        assert logical_model_path(snap / "m.gguf").name == "m.gguf"
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        (outside / "real.bin").write_bytes(b"x")
        os.symlink(outside / "real.bin", outside / "alias.gguf")
        assert logical_model_path(outside / "alias.gguf").name == "real.bin"


class TestStore:
    def test_move_out_of_the_cache_is_refused(self, tmp_path, isolated_home, layout):
        root = _cache(tmp_path, layout)
        before = sorted(p.name for p in (root / "models--org--alpha-GGUF" / "blobs").iterdir())
        assert add_local(str(root), store="move") is False
        assert load_registry() == {}
        after = sorted(p.name for p in (root / "models--org--alpha-GGUF" / "blobs").iterdir())
        assert before == after
        assert (root / "models--org--alpha-GGUF" / "snapshots" / _REV_A
                / "alpha.Q4_K_M.gguf").exists()

    def test_copy_lands_in_the_models_folder_under_real_names(self, tmp_path, isolated_home, layout):
        root = _cache(tmp_path, layout)
        assert add_local(str(root), store="copy") is True
        reg = load_registry()
        assert set(reg) == {"org-alpha-GGUF", "org-beta"}
        models = isolated_home / "models"
        gguf = Path(reg["org-alpha-GGUF"]["path"])
        assert gguf.parent == models.resolve() and gguf.name == "alpha.Q4_K_M.gguf"
        assert gguf.read_bytes() == _gguf_bytes("alpha")
        hf = Path(reg["org-beta"]["path"])
        assert hf.parent == models.resolve() and hf.name == "org-beta"
        assert (hf / "config.json").is_file()


@pytest.mark.integration
class TestRealHubCache:
    """A cache written by huggingface_hub itself, into a throwaway folder."""

    def test_downloaded_repos_register_from_their_snapshots(self, tmp_path, isolated_home):
        hf = pytest.importorskip("huggingface_hub")
        hub = tmp_path / "hub"
        hf.snapshot_download("hf-internal-testing/tiny-random-BertModel", cache_dir=str(hub))
        hf.hf_hub_download("mradermacher/tiny-random-granite-moe-GGUF",
                           "tiny-random-granite-moe.Q8_0.gguf", cache_dir=str(hub))
        assert add_local(str(hub)) is True
        reg = load_registry()
        assert set(reg) == {"hf-internal-testing-tiny-random-BertModel",
                            "mradermacher-tiny-random-granite-moe-GGUF"}
        gguf = Path(reg["mradermacher-tiny-random-granite-moe-GGUF"]["path"])
        assert gguf.name == "tiny-random-granite-moe.Q8_0.gguf" and gguf.is_file()
        assert "snapshots" in gguf.parts and "blobs" not in gguf.parts
        assert reg["mradermacher-tiny-random-granite-moe-GGUF"]["model_type"] == "llm"
        snap = Path(reg["hf-internal-testing-tiny-random-BertModel"]["path"])
        assert (snap / "config.json").is_file() and "snapshots" in snap.parts
