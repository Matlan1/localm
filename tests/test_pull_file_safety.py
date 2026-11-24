# SPDX-License-Identifier: AGPL-3.0-or-later
"""A pull gives a file its own name in the models folder only once it is complete
and verified, never takes a file for a download it is not shown to be, and
replaces a file under ``--redownload`` only with a verified one. The hub and the
HTTP server are replaced by fakes; the pull, the hashing and the registry are the
real code."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import huggingface_hub
import pytest

from localm import model_manager as mm
from localm.model_manager import pending_pulls
from localm.model_manager import pull as pull_mod
from localm.model_manager import registry as registry_mod
from tests import test_folder_import_progress as _progress
from tests.test_cached_download_reverify import _snapshot_hub, CONFIG, DAMAGED_WEIGHTS, GOOD_WEIGHTS
from tests.test_mmproj_discovery import _real_mmproj_gguf, _real_text_model_gguf
from tests.test_url_pull_resume import _RangeServer, _resp

home, _gguf = _progress.home, _progress._gguf

NAME = "weights.safetensors"
GOOD = b"good-bytes-" * 400
OTHER = b"other-bytes-" * 300
SPEC = f"o/r:{NAME}"
URL = "http://example.com/model.gguf"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _text(capsys) -> str:
    return " ".join(capsys.readouterr().out.split())


@pytest.fixture()
def hub(home, monkeypatch):
    """A fake hub: ``files`` maps a repo file name to its bytes; a name in
    ``undigested`` has no published digest. ``served`` overrides the bytes sent."""
    state = SimpleNamespace(files={}, served={}, undigested=set(), fail=None, local_dirs=[],
                            models=home / "models", calls=0)

    def fake_sha(repo, filename, token=None):
        data = state.files.get(filename)
        return (pull_mod._PublishedDigest(_sha(data), None)
                if data is not None and filename not in state.undigested else None)

    def fake_download(repo_id, filename, local_dir, **kw):
        state.calls += 1
        state.local_dirs.append(Path(local_dir))
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        if state.fail is not None:
            p.write_bytes(b"half")
            raise state.fail
        p.write_bytes(state.served.get(filename, state.files[filename]))
        return str(p)

    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    monkeypatch.setattr(mm, "_hf_file_sha256", fake_sha)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(pull_mod, "_policy_head",
                        lambda *a, **k: MagicMock(headers={"content-length": "1"}))
    monkeypatch.setattr(pull_mod, "_maybe_fetch_repo_mmproj", lambda *a, **k: None)
    monkeypatch.setattr(mm, "_hf_repo_mmproj_filename", lambda *a, **k: None)
    return state


def _pull(spec=SPEC, name=None, **kw):
    kw.setdefault("model_type", "vae")
    return mm._pull_gguf_file(spec, name, **kw)


def _visible(models: Path) -> list[str]:
    return sorted(p.name for p in models.iterdir() if not p.name.startswith("."))


def _held(models: Path) -> list[Path]:
    return sorted((models / ".cache").rglob("*.incomplete"))


class TestAHuggingFaceFileIsPlacedOnlyVerified:
    def test_stopping_the_check_leaves_partial_data_and_no_model(self, hub, monkeypatch):
        hub.files[NAME] = GOOD
        real = pull_mod._verify_digest

        def _stop(path, **kw):
            raise KeyboardInterrupt
        monkeypatch.setattr(pull_mod, "_verify_digest", _stop)
        with pytest.raises(KeyboardInterrupt):
            _pull()
        assert _visible(hub.models) == [] and [p.read_bytes() for p in _held(hub.models)] == [GOOD]
        assert mm.sync_models_dir(backfill_mmproj=False).added == 0 and mm.load_registry() == {}
        monkeypatch.setattr(pull_mod, "_verify_digest", real)
        assert _pull() is True
        assert (hub.models / NAME).read_bytes() == GOOD
        assert mm.load_registry()["weights.safetensors"]["source"] == "hf:o/r"

    def test_a_stopped_check_is_listed_as_an_interrupted_download(self, hub, monkeypatch):
        hub.files[NAME] = GOOD

        def _stop(path, **kw):
            raise KeyboardInterrupt
        monkeypatch.setattr(pull_mod, "_verify_digest", _stop)
        with pytest.raises(KeyboardInterrupt):
            mm.pull_model(SPEC, model_type="vae")
        (row,) = pending_pulls.list_interrupted()
        assert row["spec"] == SPEC and row["bytes_on_disk"] == len(GOOD)
        assert pending_pulls.delete_interrupted(row["id"]) == len(GOOD)
        assert _held(hub.models) == []

    def test_the_parts_of_a_split_model_arrive_together_first_part_last(self, hub, monkeypatch):
        parts = [f"m-0000{n}-of-00002.gguf" for n in (1, 2)]
        for p in parts:
            hub.files[p] = _gguf(hub.models.parent / "x" / p).read_bytes() + p.encode()
        seen, order = [], []
        real = pull_mod._verify_digest
        monkeypatch.setattr(pull_mod, "_verify_digest",
                            lambda path, **kw: seen.append(_visible(hub.models)) or real(path, **kw))
        real_replace = os.replace

        def _replace(src, dst):
            if Path(dst).parent == hub.models:
                order.append(Path(dst).name)
            return real_replace(src, dst)
        monkeypatch.setattr(pull_mod.os, "replace", _replace)
        assert _pull("o/r:" + parts[0], model_type="llm") is True
        assert seen == [[], []] and order == parts[::-1]

    def test_the_second_file_of_the_same_pull_is_named_in_the_progress(self):
        parts = [Path("a-1.gguf"), Path("a-2.gguf")]
        assert pull_mod._progress_file_info(parts, 0) == ("a-1.gguf", 1, 2)
        assert pull_mod._progress_file_info(parts, 1) == ("a-2.gguf", 2, 2)
        assert pull_mod._progress_file_info(parts, 2) == ("a-2.gguf", 2, 2)

    def test_partial_data_of_an_earlier_localm_is_taken_over(self, hub):
        from huggingface_hub._local_folder import get_local_download_paths
        old = get_local_download_paths(hub.models, NAME).incomplete_path("etag1")
        old.parent.mkdir(parents=True, exist_ok=True)
        old.write_bytes(b"saved")
        stage = hub.models / pull_mod._REPO_FOLDER_STAGING
        pull_mod._take_over_partials(hub.models, stage, NAME)
        new = get_local_download_paths(stage, NAME).incomplete_path("etag1")
        assert not old.exists() and new.read_bytes() == b"saved"

    def test_stopping_the_projector_download_leaves_the_model_registered(self, hub, monkeypatch):
        hub.files["m.gguf"] = _gguf(hub.models.parent / "m.gguf").read_bytes()

        def _stop(*a, **k):
            raise KeyboardInterrupt
        monkeypatch.setattr(pull_mod, "_maybe_fetch_repo_mmproj", _stop)
        with pytest.raises(KeyboardInterrupt):
            _pull("o/r:m.gguf", model_type="llm")
        entry = mm.load_registry()["m"]
        assert entry["source"] == "hf:o/r" and entry["sha256"] == _sha(hub.files["m.gguf"])

    def test_two_pulls_of_one_file_name_do_not_run_at_once(self, hub, capsys):
        hub.files[NAME] = GOOD
        with pull_mod._part_lock(NAME):
            assert _pull() is False
            with pytest.raises(pull_mod.PullInFlight):
                pull_mod.discard_partial_download(SPEC)
        assert hub.calls == 0 and "Download already in progress" in _text(capsys)
        assert _pull() is True

    def test_a_file_that_appeared_meanwhile_is_not_overwritten(self, hub, capsys, monkeypatch):
        hub.files[NAME] = GOOD
        real = pull_mod._verify_digest

        def _appear(path, **kw):
            (hub.models / NAME).write_bytes(OTHER)
            return real(path, **kw)
        monkeypatch.setattr(pull_mod, "_verify_digest", _appear)
        assert _pull() is False
        assert (hub.models / NAME).read_bytes() == OTHER and mm.load_registry() == {}
        assert "left as it is" in _text(capsys)


class TestAHuggingFaceFileThatIsAlreadyThere:
    def test_without_any_digest_an_unregistered_file_is_not_taken_for_it(self, hub, capsys):
        hub.files[NAME] = GOOD
        hub.undigested.add(NAME)
        (hub.models / NAME).write_bytes(OTHER)
        assert _pull() is False
        assert mm.load_registry() == {} and (hub.models / NAME).read_bytes() == OTHER
        out = _text(capsys)
        assert "nothing shows that it came from" in out and "--redownload" in out

    @pytest.mark.parametrize("source", ["local", "hf:someone/else"])
    def test_without_any_digest_one_registered_from_elsewhere_is_not_taken_for_it(
            self, hub, capsys, source):
        hub.files[NAME] = GOOD
        hub.undigested.add(NAME)
        (hub.models / NAME).write_bytes(OTHER)
        mm._register("mine", hub.models / NAME, source, model_type="vae")
        before = mm.load_registry()
        assert _pull(name="other-name") is False
        assert mm.load_registry() == before and (hub.models / NAME).read_bytes() == OTHER
        assert "localm rm mine" in _text(capsys)

    def test_without_any_digest_one_registered_from_there_is_it(self, hub):
        hub.files[NAME] = GOOD
        hub.undigested.add(NAME)
        assert _pull() is True and _pull() is True and hub.calls == 1

    def test_one_the_scan_registered_takes_the_source_it_is_verified_for(self, hub):
        hub.files[NAME] = GOOD
        (hub.models / NAME).write_bytes(GOOD)
        mm._register("weights.safetensors", hub.models / NAME, model_type="vae")
        assert _pull() is True
        entry = mm.load_registry()["weights.safetensors"]
        assert entry["source"] == "hf:o/r" and entry["sha256"] == _sha(GOOD)
        assert hub.calls == 0

    def test_a_part_that_stays_must_match_before_the_others_are_fetched(self, hub, capsys):
        parts = [f"m-0000{n}-of-00002.gguf" for n in (1, 2)]
        for p in parts:
            hub.files[p] = p.encode() * 99
        (hub.models / parts[0]).write_bytes(OTHER)
        assert _pull("o/r:" + parts[0], model_type="llm") is False
        assert hub.calls == 0 and (hub.models / parts[0]).read_bytes() == OTHER
        assert "--redownload" in _text(capsys)

    @pytest.mark.parametrize("there", [0, 1])
    def test_a_checksum_failure_of_one_part_leaves_the_parts_that_were_there(
            self, hub, tmp_path, there):
        parts = [f"m-0000{n}-of-00002.gguf" for n in (1, 2)]
        for p in parts:
            hub.files[p] = _gguf(tmp_path / p).read_bytes() + p.encode()
        (hub.models / parts[there]).write_bytes(hub.files[parts[there]])
        hub.served[parts[1 - there]] = b"damaged" * 50
        assert _pull("o/r:" + parts[0], model_type="llm") is False
        assert hub.calls == 1 and _visible(hub.models) == [parts[there]]
        assert (hub.models / parts[there]).read_bytes() == hub.files[parts[there]]
        assert _held(hub.models) == [] and mm.load_registry() == {}


class TestRedownloadOfAHuggingFaceFile:
    @pytest.mark.parametrize("sha", [None, _sha(GOOD)])
    def test_the_refusal_names_a_way_out_and_it_works(self, hub, capsys, sha):
        hub.files[NAME] = GOOD
        (hub.models / NAME).write_bytes(OTHER)
        assert _pull(expected_sha256=sha) is False
        assert (hub.models / NAME).read_bytes() == OTHER
        assert "Pull again with --redownload to replace it" in _text(capsys)
        assert _pull(expected_sha256=sha, redownload=True) is True
        assert (hub.models / NAME).read_bytes() == GOOD
        assert mm.load_registry()["weights.safetensors"]["source"] == "hf:o/r"

    def test_it_fetches_again_even_when_the_file_is_right(self, hub):
        hub.files[NAME] = GOOD
        assert _pull() is True and _pull() is True and hub.calls == 1
        assert _pull(redownload=True) is True and hub.calls == 2

    @pytest.mark.parametrize("error", [OSError("connection reset"), KeyboardInterrupt()])
    def test_a_download_that_fails_leaves_the_old_file(self, hub, error):
        hub.files[NAME] = GOOD
        assert _pull() is True
        (hub.models / NAME).write_bytes(OTHER)
        hub.fail = error
        if isinstance(error, KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                _pull(redownload=True)
        else:
            assert _pull(redownload=True) is False
        assert (hub.models / NAME).read_bytes() == OTHER and "weights.safetensors" in mm.load_registry()

    def test_a_download_with_the_wrong_digest_leaves_the_old_file(self, hub, capsys):
        hub.files[NAME] = GOOD
        assert _pull() is True
        hub.served[NAME] = b"damaged" * 50
        assert _pull(redownload=True) is False
        assert (hub.models / NAME).read_bytes() == GOOD and _held(hub.models) == []
        assert "SHA256 mismatch" in _text(capsys)

    def test_one_bad_part_replaces_none_of_them(self, hub, tmp_path):
        parts = [f"m-0000{n}-of-00002.gguf" for n in (1, 2)]
        for p in parts:
            hub.files[p] = _gguf(tmp_path / p).read_bytes() + p.encode()
        assert _pull("o/r:" + parts[0], model_type="llm") is True
        (hub.models / parts[0]).write_bytes(OTHER)
        hub.served[parts[1]] = b"damaged" * 50
        assert _pull("o/r:" + parts[0], model_type="llm", redownload=True) is False
        assert (hub.models / parts[0]).read_bytes() == OTHER
        assert (hub.models / parts[1]).read_bytes() == hub.files[parts[1]]
        assert _held(hub.models) == []

    @pytest.mark.parametrize("source", ["local", "hf:someone/else"])
    def test_a_file_registered_from_elsewhere_is_not_replaced(self, hub, capsys, source):
        hub.files[NAME] = GOOD
        (hub.models / NAME).write_bytes(OTHER)
        mm._register("mine", hub.models / NAME, source, model_type="vae")
        before = mm.load_registry()
        assert _pull(redownload=True) is False
        assert (hub.models / NAME).read_bytes() == OTHER and mm.load_registry() == before
        assert "localm rm mine" in _text(capsys) and hub.calls == 0

    def test_an_unreadable_registry_blocks_the_replacement(self, hub, home, capsys):
        hub.files[NAME] = GOOD
        (hub.models / NAME).write_bytes(OTHER)
        (home / "registry.json").write_text("{ not json", encoding="utf-8")
        assert _pull(redownload=True) is False
        assert (hub.models / NAME).read_bytes() == OTHER
        assert "could not be read" in _text(capsys)

    def test_every_name_of_the_file_takes_the_new_digest_and_keeps_its_keys(self, hub):
        hub.files[NAME] = GOOD
        assert _pull() is True
        assert mm.alias_model("weights.safetensors", "other-name") is True
        mm.update_registry(lambda r: r["weights.safetensors"].__setitem__("note", "kept"))
        hub.files[NAME] = OTHER
        assert _pull(redownload=True) is True
        reg = mm.load_registry()
        assert reg["weights.safetensors"]["sha256"] == reg["other-name"]["sha256"] == _sha(OTHER)
        assert reg["weights.safetensors"]["note"] == "kept"

    def test_a_file_in_use_is_kept_and_so_is_the_download(self, hub, capsys, monkeypatch):
        hub.files[NAME] = GOOD
        assert _pull() is True
        hub.files[NAME] = OTHER
        real = os.replace

        def _replace(src, dst):
            if Path(dst) == hub.models / NAME:
                raise PermissionError(13, "in use")
            return real(src, dst)
        monkeypatch.setattr(pull_mod.os, "replace", _replace)
        assert _pull(redownload=True) is False
        assert (hub.models / NAME).read_bytes() == GOOD
        assert [p.read_bytes() for p in _held(hub.models)] == [OTHER]
        assert "unload the model and pull again" in _text(capsys)

    def test_replacing_several_parts_never_leaves_a_name_without_a_file(self, tmp_path, monkeypatch):
        base = tmp_path / "models"
        base.mkdir()
        held = {}
        for n in ("a", "b"):
            (base / n).write_bytes(f"old {n}".encode())
            held[n] = tmp_path / f"new-{n}"
            held[n].write_bytes(f"new {n}".encode())
        real, seen = os.replace, []

        def _replace(src, dst):
            seen.append(Path(dst).read_text() if Path(dst).exists() else None)
            return real(src, dst)
        monkeypatch.setattr(pull_mod.os, "replace", _replace)
        pull_mod._place_downloads(held, base)
        assert seen == ["old b", "old a"]
        assert {p.name: p.read_text() for p in base.iterdir()} == {"a": "new a", "b": "new b"}

    @pytest.mark.parametrize("links", [True, False])
    def test_when_one_part_cannot_be_replaced_the_others_go_back(self, tmp_path, monkeypatch, links):
        if not links:
            def _link(src, dst, **kw):
                raise OSError(errno.EPERM, "links are not supported here")
            monkeypatch.setattr(pull_mod.os, "link", _link)
        base = tmp_path / "models"
        base.mkdir()
        held = {}
        for n in ("a", "b"):
            (base / n).write_bytes(f"old {n}".encode())
            held[n] = tmp_path / f"new-{n}"
            held[n].write_bytes(f"new {n}".encode())
        real = os.replace

        def _replace(src, dst):
            if Path(src) == held["a"]:
                raise PermissionError(13, "in use")
            return real(src, dst)
        monkeypatch.setattr(pull_mod.os, "replace", _replace)
        with pytest.raises(PermissionError):
            pull_mod._place_downloads(held, base)
        assert {p.name: p.read_text() for p in base.iterdir()} == {"a": "old a", "b": "old b"}
        assert held["a"].read_text() == "new a" and held["b"].read_text() == "new b"


# ------------------------------------------------------------------ direct URLs

@pytest.fixture()
def url(home, monkeypatch):
    body = _gguf(home / "body.gguf").read_bytes() + b"tail"
    server = _RangeServer(body)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    monkeypatch.setattr("socket.getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    monkeypatch.setattr("localm.netpolicy.pinned_request", server)
    return SimpleNamespace(body=body, server=server, dest=home / "models" / "model.gguf",
                           part=home / "models" / "model.gguf.part")


class TestADirectUrlWhoseFileNameIsTaken:
    def test_an_unregistered_file_is_not_taken_for_it(self, url, capsys):
        _gguf(url.dest, payload=b"mine")
        before = url.dest.read_bytes()
        assert mm._pull_url(URL, "m") is False
        assert mm.load_registry() == {} and url.dest.read_bytes() == before
        assert url.server.gets == []
        out = _text(capsys)
        assert "nothing shows that it came from" in out
        for way in ("--redownload", "--sha256", "move it out"):
            assert way in out

    def test_a_file_registered_from_elsewhere_is_not_taken_for_it(self, url, capsys):
        _gguf(url.dest, payload=b"mine")
        mm._register("mine", url.dest)
        assert mm._pull_url(URL, "m") is False
        assert set(mm.load_registry()) == {"mine"}
        assert "localm rm mine" in _text(capsys)

    def test_the_file_the_registry_records_from_this_url_is_it(self, url):
        assert mm._pull_url(URL, "m") is True
        gets = len(url.server.gets)
        assert mm._pull_url(URL, "m") is True and len(url.server.gets) == gets

    def test_the_right_digest_proves_it(self, url):
        url.dest.write_bytes(url.body)
        assert mm._pull_url(URL, "m", expected_sha256=_sha(url.body)) is True
        assert url.server.gets == [] and mm.load_registry()["m"]["sha256"] == _sha(url.body)

    def test_redownload_replaces_it(self, url):
        _gguf(url.dest, payload=b"mine")
        assert mm._pull_url(URL, "m", redownload=True) is True
        assert url.dest.read_bytes() == url.body
        assert mm.load_registry()["m"]["source"] == URL

    def test_redownload_replaces_a_damaged_file_of_this_url(self, url):
        assert mm._pull_url(URL, "m") is True
        _gguf(url.dest, payload=b"damaged")
        assert mm._pull_url(URL, "m", redownload=True) is True
        assert url.dest.read_bytes() == url.body
        assert mm.load_registry()["m"]["sha256"] == _sha(url.body)

    def test_redownload_refuses_a_file_registered_from_elsewhere(self, url, capsys):
        _gguf(url.dest, payload=b"mine")
        before = url.dest.read_bytes()
        mm._register("mine", url.dest)
        assert mm._pull_url(URL, "m", redownload=True) is False
        assert url.dest.read_bytes() == before and url.server.gets == []
        assert "localm rm mine" in _text(capsys)

    def test_a_redownload_with_the_wrong_digest_leaves_the_old_file(self, url, capsys):
        _gguf(url.dest, payload=b"mine")
        before = url.dest.read_bytes()
        assert mm._pull_url(URL, "m", redownload=True, expected_sha256=_sha(b"x")) is False
        assert url.dest.read_bytes() == before and not url.part.exists()
        assert "SHA256 mismatch" in _text(capsys)

    def test_a_file_in_use_is_kept_and_so_is_the_download(self, url, capsys, monkeypatch):
        _gguf(url.dest, payload=b"mine")
        before = url.dest.read_bytes()
        real = os.replace

        def _replace(src, dst):
            if Path(dst) == url.dest:
                raise PermissionError(13, "in use")
            return real(src, dst)
        monkeypatch.setattr(pull_mod.os, "replace", _replace)
        assert mm._pull_url(URL, "m", redownload=True) is False
        assert url.dest.read_bytes() == before and url.part.read_bytes() == url.body
        assert "unload the model and pull again" in _text(capsys)
        monkeypatch.setattr(pull_mod.os, "replace", real)
        gets = len(url.server.gets)
        assert mm._pull_url(URL, "m", redownload=True) is True
        assert url.dest.read_bytes() == url.body and len(url.server.gets) == gets


class TestADirectUrlIsPlacedOnlyVerified:
    @pytest.mark.parametrize("head_length", [True, False])
    def test_stopping_the_check_leaves_the_partial_and_no_model(self, url, monkeypatch, head_length):
        url.server.head_length = head_length
        real_call = url.server.__call__

        def _call(method, u, **kw):
            r = real_call(method, u, **kw)
            if method == "GET" and r.status_code == 416:
                r.headers = {"content-range": f"bytes */{len(url.body)}"}
            return r
        monkeypatch.setattr("localm.netpolicy.pinned_request", _call)
        real = pull_mod._verify_digest

        def _stop(path, **kw):
            raise KeyboardInterrupt
        monkeypatch.setattr(pull_mod, "_verify_digest", _stop)
        with pytest.raises(KeyboardInterrupt):
            mm._pull_url(URL, "m")
        assert not url.dest.exists() and url.part.read_bytes() == url.body
        assert mm.sync_models_dir(backfill_mmproj=False).added == 0
        monkeypatch.setattr(pull_mod, "_verify_digest", real)
        assert mm._pull_url(URL, "m") is True
        assert url.dest.read_bytes() == url.body and mm.load_registry()["m"]["source"] == URL
        assert [g for g in url.server.gets if "Range" not in g] == [{}]
        assert not url.part.exists() and not url.part.with_name(url.part.name + ".json").exists()

    @pytest.mark.parametrize("scanned", [False, True])
    def test_one_killed_before_registering_is_finished_by_the_next_pull(
            self, url, monkeypatch, scanned):
        def _killed(*a, **k):
            raise KeyboardInterrupt
        monkeypatch.setattr(pull_mod, "_register_download", _killed)
        with pytest.raises(KeyboardInterrupt):
            mm._pull_url(URL, "model")
        monkeypatch.undo()
        self._again(url, monkeypatch, scanned)

    def _again(self, url, monkeypatch, scanned):
        models = url.dest.parent
        monkeypatch.setenv("LOCALM_HOME", str(models.parent))
        import localm.config as cfg
        for mod in (cfg, mm):
            monkeypatch.setattr(mod, "HOME_DIR", models.parent)
            monkeypatch.setattr(mod, "MODELS_DIR", models)
            monkeypatch.setattr(mod, "REGISTRY_FILE", models.parent / "registry.json")
        monkeypatch.setattr(cfg, "CONFIG_FILE", models.parent / "config.json")
        if scanned:
            monkeypatch.setattr(mm, "_GGUF_SETTLE_SECONDS", 0)
            assert mm.sync_models_dir(backfill_mmproj=False).added == 1
        assert mm._pull_url(URL, "model") is True
        entry = mm.load_registry()["model"]
        assert entry["source"] == URL and entry["sha256"] == _sha(url.body)
        assert not url.part.with_name(url.part.name + ".json").exists()

    def test_a_recorded_digest_the_file_no_longer_has_proves_nothing(self, url):
        _gguf(url.dest, payload=b"mine")
        url.part.with_name(url.part.name + ".json").write_text(json.dumps({
            "source": "url", "url_sha256": _sha(URL.encode()), "finished_sha256": _sha(url.body)}))
        assert mm._pull_url(URL, "m") is False and mm.load_registry() == {}

    def test_its_type_is_read_from_the_file(self, url, tmp_path, monkeypatch):
        model = _real_text_model_gguf(tmp_path / "bert.gguf", "bert", 384).read_bytes()
        url.server.body = model
        seen = []
        monkeypatch.setattr(mm, "gguf_embedding_signal", lambda p: seen.append(Path(p)) or True)
        assert mm.pull_model(URL, "m") is True
        assert mm.load_registry()["m"]["model_type"] == "embedding" and seen

    def test_a_civitai_download_that_was_stopped_while_checked_is_not_fetched_again(
            self, home, monkeypatch):
        dest = home / "models" / "lora.safetensors"
        part = dest.with_name(dest.name + ".part")
        body = b"lora" * 50
        resolved = SimpleNamespace(url="https://civitai.example/d", size_bytes=None, file_id="1",
                                   source_tag="civitai:1:1", filename=dest.name)
        ident = pull_mod._civitai_part_identity("1", resolved, _sha(body))
        pull_mod._start_part(part, ident).close()
        part.write_bytes(body)
        answer = _resp(416, b"")
        answer.headers = {"content-range": f"bytes */{len(body)}"}
        asked = []
        monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
        monkeypatch.setattr(pull_mod, "_ssrf_resolve_final_url", lambda u: u)
        monkeypatch.setattr("localm.netpolicy.check_url", lambda *a, **k: None)
        monkeypatch.setattr("localm.netpolicy.pinned_request",
                            lambda method, u, **kw: asked.append(kw["headers"]) or answer)
        assert pull_mod._pull_civitai_file_locked(
            "1", resolved, dest, verify_digest=_sha(body), register=False, model_name="lora",
            reg_type="lora") is True
        assert dest.read_bytes() == body and asked == [{"Range": f"bytes={len(body)}-"}]


# ------------------------------------------------------------------ projectors

class TestAVisionProjector:
    MM = "mmproj-f16.gguf"

    def _fetch(self, hub):
        pull_mod._download_mmproj_file("o/r", self.MM, hub.models / self.MM, hub.models)

    def test_stopping_the_check_leaves_no_projector(self, hub, monkeypatch):
        hub.files[self.MM] = GOOD

        def _stop(path, **kw):
            raise KeyboardInterrupt
        monkeypatch.setattr(pull_mod, "_verify_digest", _stop)
        with pytest.raises(KeyboardInterrupt):
            self._fetch(hub)
        assert _visible(hub.models) == []

    def test_a_damaged_one_stays_until_the_new_one_is_verified(self, hub, tmp_path, monkeypatch):
        right = _real_mmproj_gguf(tmp_path / "p.gguf", 64).read_bytes()
        hub.files[self.MM] = right
        self._fetch(hub)
        damaged = right + b"damaged"
        (hub.models / self.MM).write_bytes(damaged)
        hub.fail = OSError("offline")
        dest, skip = pull_mod._mmproj_download_dest(hub.models, self.MM, "o/r", None)
        assert (dest, skip) == (hub.models / self.MM, False)
        with pytest.raises(OSError):
            self._fetch(hub)
        assert (hub.models / self.MM).read_bytes() == damaged
        hub.fail = None
        self._fetch(hub)
        assert (hub.models / self.MM).read_bytes() == right

    def test_a_second_pull_waits_and_uses_the_first_ones_file(self, hub, monkeypatch):
        hub.files[self.MM] = GOOD
        monkeypatch.setattr(pull_mod, "_LOCK_WAIT_POLL_SECONDS", 0.02)
        started, go = threading.Event(), threading.Event()
        real = huggingface_hub.hf_hub_download

        def _slow(**kw):
            started.set()
            assert go.wait(30)
            return real(**kw)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _slow)
        first = threading.Thread(target=self._fetch, args=(hub,))
        first.start()
        assert started.wait(30)
        errors = []

        def _second():
            try:
                self._fetch(hub)
            except BaseException as e:
                errors.append(e)
        second = threading.Thread(target=_second)
        second.start()
        second.join(0.3)
        assert second.is_alive()
        go.set()
        first.join(30)
        second.join(30)
        assert errors == [] and hub.calls == 1
        assert (hub.models / self.MM).read_bytes() == GOOD


# ------------------------------------------------------------------ full repos

class TestAFullRepository:
    @pytest.fixture()
    def snap(self, home, monkeypatch):
        monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
        monkeypatch.setattr(pull_mod, "_resolve_snapshot_type", lambda *a, **k: "llm")
        dest = home / "models" / "snap"
        dest.mkdir()
        (dest / "config.json").write_bytes(CONFIG)
        downloaded: list = []
        _snapshot_hub(monkeypatch, downloaded)
        return SimpleNamespace(dest=dest, downloaded=downloaded, weights=dest / "model.safetensors")

    def _lay(self, snap, weights):
        snap.weights.write_bytes(weights)
        mm._register("snap", snap.dest, "hf:o/r")

    @pytest.mark.parametrize("error", [OSError("offline"), KeyboardInterrupt()])
    def test_a_damaged_file_stays_until_the_new_one_is_verified(self, snap, capsys, monkeypatch, error):
        self._lay(snap, DAMAGED_WEIGHTS)

        def _fail(**kw):
            raise error
        real = huggingface_hub.snapshot_download
        monkeypatch.setattr(huggingface_hub, "snapshot_download", _fail)
        if isinstance(error, KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                mm._pull_hf_snapshot("o/r", "snap")
        else:
            assert mm._pull_hf_snapshot("o/r", "snap") is False
            assert "are unchanged" in _text(capsys)
        assert snap.weights.read_bytes() == DAMAGED_WEIGHTS
        monkeypatch.setattr(huggingface_hub, "snapshot_download", real)
        assert mm._pull_hf_snapshot("o/r", "snap") is True
        assert snap.weights.read_bytes() == GOOD_WEIGHTS
        assert not (snap.dest / pull_mod._SNAPSHOT_STAGING).exists()

    def test_a_refetch_that_arrives_wrong_replaces_nothing(self, snap, capsys, monkeypatch):
        self._lay(snap, DAMAGED_WEIGHTS)
        real = huggingface_hub.snapshot_download

        def _wrong(**kw):
            out = real(**kw)
            (Path(out) / "model.safetensors").write_bytes(b"wrong")
            return out
        monkeypatch.setattr(huggingface_hub, "snapshot_download", _wrong)
        assert mm._pull_hf_snapshot("o/r", "snap") is False
        assert snap.weights.read_bytes() == DAMAGED_WEIGHTS
        assert "Nothing was replaced" in _text(capsys)

    def test_redownload_fetches_a_complete_model_again(self, snap):
        self._lay(snap, GOOD_WEIGHTS)
        assert mm._pull_hf_snapshot("o/r", "snap") is True and snap.downloaded == []
        assert mm._pull_hf_snapshot("o/r", "snap", redownload=True) is True
        assert snap.downloaded == ["o/r"]
        assert snap.weights.read_bytes() == GOOD_WEIGHTS

    def test_redownload_replaces_a_file_no_check_can_see(self, snap):
        self._lay(snap, GOOD_WEIGHTS)
        other = b"x" * len(CONFIG)
        (snap.dest / "config.json").write_bytes(other)
        assert mm._pull_hf_snapshot("o/r", "snap") is True
        assert (snap.dest / "config.json").read_bytes() == other
        assert mm._pull_hf_snapshot("o/r", "snap", redownload=True) is True
        assert (snap.dest / "config.json").read_bytes() == CONFIG

    def test_redownload_refuses_a_folder_registered_from_elsewhere(self, snap, capsys):
        self._lay(snap, GOOD_WEIGHTS)
        mm._register("theirs", snap.dest, "hf:someone/else")
        assert mm._pull_hf_snapshot("o/r", "snap", redownload=True) is False
        assert snap.downloaded == [] and "theirs" in _text(capsys)

    def test_redownload_without_a_file_list_replaces_nothing(self, snap, capsys, monkeypatch):
        self._lay(snap, GOOD_WEIGHTS)

        class _Down:
            def __init__(self, *a, **kw):
                pass

            def model_info(self, *a, **kw):
                raise OSError("offline")

            def list_repo_files(self, *a, **kw):
                return ["config.json", "model.safetensors"]
        monkeypatch.setattr(huggingface_hub, "HfApi", _Down)
        assert mm._pull_hf_snapshot("o/r", "snap", redownload=True) is False
        assert snap.downloaded == [] and "could not be read" in _text(capsys)

    def test_only_the_files_to_replace_are_asked_for(self, snap, monkeypatch):
        self._lay(snap, DAMAGED_WEIGHTS)
        asked = []
        real = huggingface_hub.snapshot_download
        monkeypatch.setattr(huggingface_hub, "snapshot_download",
                            lambda **kw: asked.append((kw["local_dir"], kw.get("allow_patterns")))
                            or real(**kw))
        assert mm._pull_hf_snapshot("o/r", "snap") is True
        assert asked == [(str(snap.dest / pull_mod._SNAPSHOT_STAGING), ["model.safetensors"])]

    def test_a_folder_the_scan_registered_takes_the_source_of_its_own_pull(self, snap):
        snap.weights.write_bytes(GOOD_WEIGHTS)
        mm._register("snap", snap.dest)
        (snap.dest / registry_mod.SNAPSHOT_CLAIM).write_text(
            json.dumps({"repo": "o/r", "revision": None}))
        assert mm._pull_hf_snapshot("o/r", "snap") is True
        assert mm.load_registry()["snap"]["source"] == "hf:o/r" and snap.downloaded == []

    def test_discarding_a_refetch_frees_it_and_keeps_the_model(self, snap):
        self._lay(snap, DAMAGED_WEIGHTS)
        stage = snap.dest / pull_mod._SNAPSHOT_STAGING
        stage.mkdir(parents=True)
        (stage / "model.safetensors").write_bytes(b"n" * 10)
        assert pull_mod.discard_partial_download("o/r", "snap") == 10
        assert not stage.exists() and snap.weights.read_bytes() == DAMAGED_WEIGHTS


def test_a_failed_placement_errno_is_reported_like_any_other(hub, capsys, monkeypatch):
    hub.files[NAME] = GOOD

    def _full(held, base):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(pull_mod, "_place_downloads", _full)
    assert _pull() is False
    assert mm.load_registry() == {} and [p.read_bytes() for p in _held(hub.models)] == [GOOD]
    assert "No space left on device" in _text(capsys)
