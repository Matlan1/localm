# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for resumable HuggingFace model downloads: a per-process temp file
with an owner record, adoption of a partial only when its owner is proven
gone, an etag-scoped disk-space preflight, resume over HTTP Range for
Xet-backed files, and verification against HuggingFace's own digest.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from localm import model_manager as mm
from localm.model_manager.pull import (
    _ensure_hf_resumable_download,
    _partial_owner_path,
    _resumable_download_to_tmp_and_move,
)
from tests._process_identity import (
    a_forward_step_past_boot,
    spawn_on_this_tree,
    start_identity_of,
    started_an_hour_earlier,
    step_the_clock,
)


def _owner(path: Path, pid: int, start) -> None:
    """Write the owner record for *path* naming *pid* / start identity *start*."""
    _partial_owner_path(path).write_text(
        json.dumps({"pid": pid, "start": start}), encoding="utf-8")


@pytest.fixture()
def live_child():
    """A real child process that stays alive for the test, with its start
    identity."""
    from localm.model_manager.pull import _process_start_identity
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield p.pid, _process_start_identity(p.pid)
    finally:
        p.kill()
        p.wait()


@pytest.fixture()
def dead_pid():
    """The pid and start identity of a child that has already exited."""
    from localm.model_manager.pull import _process_start_identity
    p = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
                         stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    start = _process_start_identity(p.pid)
    p.stdin.close()
    p.wait()
    return p.pid, start


OWN_PARTIAL = '''
    import sys
    from pathlib import Path
    from localm.model_manager.pull import _write_partial_owner
    p = Path(sys.argv[1])
    p.write_bytes(b"LIVE-PROCESS-BYTES")
    _write_partial_owner(p)
    print("OWNED", flush=True)
    sys.stdin.read()
'''


def _call(inc_path: Path, dest: Path, *, expected_size, force=False,
          xet_file_data=None):
    _resumable_download_to_tmp_and_move(
        incomplete_path=inc_path,
        destination_path=dest,
        url_to_download="https://example.com/file",
        headers={},
        expected_size=expected_size,
        filename="dest.gguf",
        force_download=force,
        etag="etag",
        xet_file_data=xet_file_data,
    )


def _wire_http(monkeypatch, tail: bytes):
    """Fake http_get that records resume_size and appends *tail*."""
    captured = {}

    def fake_http_get(url, f, resume_size=0, headers=None, expected_size=None, **kw):
        captured["resume_size"] = resume_size
        captured["expected_size"] = expected_size
        f.write(tail)

    monkeypatch.setattr("huggingface_hub.file_download.http_get", fake_http_get)
    monkeypatch.setattr("huggingface_hub.file_download._check_disk_space", lambda *a: None)
    return captured


def _partials(cache: Path):
    return sorted(p.name for p in cache.iterdir() if p.name.endswith(".incomplete"))


@pytest.fixture()
def cache(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    return d


class TestPartialOwnership:
    def test_live_owned_partial_is_left_alone_and_orphan_is_adopted(
            self, cache, tmp_path, monkeypatch, live_child, dead_pid):
        inc_path = cache / "file.etag.incomplete"
        live = cache / "file.etag.aaaa1111.incomplete"
        live.write_bytes(b"LIVE-PROCESS-BYTES")
        _owner(live, *live_child)
        orphan = cache / "file.etag.bbbb2222.incomplete"
        orphan.write_bytes(b"HELLO")
        _owner(orphan, *dead_pid)
        dest = tmp_path / "dest.gguf"
        captured = _wire_http(monkeypatch, b" WORLD")

        _call(inc_path, dest, expected_size=11)

        assert live.read_bytes() == b"LIVE-PROCESS-BYTES", (
            "another process's in-flight temp file was touched")
        assert _partial_owner_path(live).exists()
        assert dest.read_bytes() == b"HELLO WORLD"
        assert captured["resume_size"] == 5
        assert not orphan.exists() and not _partial_owner_path(orphan).exists()
        assert _partials(cache) == [live.name], _partials(cache)

    def test_a_live_owners_partial_survives_a_clock_step(
            self, cache, tmp_path, monkeypatch):
        """A clock step (NTP after sleep, a VM or WSL guest resyncing) does not
        make a live download's temp file look orphaned."""
        inc_path = cache / "file.etag.incomplete"
        live = cache / "file.etag.aaaa1111.incomplete"
        owner = spawn_on_this_tree(OWN_PARTIAL, tmp_path / ".localm", live,
                                   stdin=subprocess.PIPE)
        try:
            ready = owner.stdout.readline().strip()
            if ready != "OWNED":
                owner.stdin.close()
                owner.wait(timeout=60)
                pytest.fail(f"the owner did not start: {ready!r} "
                            f"{owner.stderr.read()}")
            # The injection took: a live process recorded itself as the owner.
            assert _partial_owner_path(live).exists()
            dest = tmp_path / "dest.gguf"
            captured = _wire_http(monkeypatch, b"HELLO")

            with monkeypatch.context() as m:
                step_the_clock(m, a_forward_step_past_boot())
                _call(inc_path, dest, expected_size=5)

            assert (live.read_bytes() if live.exists() else None) == (
                b"LIVE-PROCESS-BYTES"), (
                "another process's in-flight temp file was touched after a "
                "clock step")
            assert _partial_owner_path(live).exists()
            assert captured["resume_size"] == 0
            assert dest.read_bytes() == b"HELLO"
        finally:
            owner.stdin.close()
            owner.wait(timeout=60)

    def test_a_partial_whose_pid_now_names_another_process_is_adopted(
            self, cache, tmp_path, monkeypatch, live_child):
        """The recorded pid is alive but its start identity is not the
        recorded one, so the partial's owner is gone and the partial is
        resumed from."""
        pid, _ = live_child
        inc_path = cache / "file.etag.incomplete"
        orphan = cache / "file.etag.bbbb2222.incomplete"
        orphan.write_bytes(b"HELLO")
        _owner(orphan, pid, started_an_hour_earlier(start_identity_of(pid)))
        dest = tmp_path / "dest.gguf"
        captured = _wire_http(monkeypatch, b" WORLD")

        _call(inc_path, dest, expected_size=11)

        assert dest.read_bytes() == b"HELLO WORLD"
        assert captured["resume_size"] == 5
        assert not orphan.exists() and not _partial_owner_path(orphan).exists()

    def test_partial_without_owner_record_is_neither_adopted_nor_deleted(
            self, cache, tmp_path, monkeypatch):
        inc_path = cache / "file.etag.incomplete"
        unknown = cache / "file.etag.cccc3333.incomplete"
        unknown.write_bytes(b"WHO-OWNS-THIS")
        legacy = cache / "file.etag.incomplete"
        legacy.write_bytes(b"OLD-SHARED")
        dest = tmp_path / "dest.gguf"
        captured = _wire_http(monkeypatch, b"FRESH")

        _call(inc_path, dest, expected_size=5)

        assert unknown.read_bytes() == b"WHO-OWNS-THIS"
        assert legacy.read_bytes() == b"OLD-SHARED"
        assert captured["resume_size"] == 0
        assert dest.read_bytes() == b"FRESH"

    def test_smaller_proven_orphans_are_removed_after_adoption(
            self, cache, tmp_path, monkeypatch, dead_pid):
        inc_path = cache / "file.etag.incomplete"
        big = cache / "file.etag.aaaa1111.incomplete"
        big.write_bytes(b"HELLO")
        _owner(big, *dead_pid)
        small = cache / "file.etag.bbbb2222.incomplete"
        small.write_bytes(b"HE")
        _owner(small, *dead_pid)
        dest = tmp_path / "dest.gguf"
        captured = _wire_http(monkeypatch, b" WORLD")

        _call(inc_path, dest, expected_size=11)

        assert captured["resume_size"] == 5
        assert dest.read_bytes() == b"HELLO WORLD"
        assert _partials(cache) == []
        assert not _partial_owner_path(small).exists()

    def test_own_temp_and_owner_record_survive_a_failed_transfer(
            self, cache, tmp_path, monkeypatch):
        inc_path = cache / "file.etag.incomplete"
        dest = tmp_path / "dest.gguf"

        def failing_http_get(url, f, resume_size=0, **kw):
            f.write(b"HELLO")
            raise ConnectionError("link dropped")

        monkeypatch.setattr("huggingface_hub.file_download.http_get", failing_http_get)
        monkeypatch.setattr("huggingface_hub.file_download._check_disk_space", lambda *a: None)
        with pytest.raises(ConnectionError):
            _call(inc_path, dest, expected_size=11)

        left = _partials(cache)
        assert len(left) == 1 and left[0].startswith("file.etag.") , left
        partial = cache / left[0]
        rec = json.loads(_partial_owner_path(partial).read_text(encoding="utf-8"))
        assert rec["pid"] == os.getpid()

        captured = _wire_http(monkeypatch, b" WORLD")
        _call(inc_path, dest, expected_size=11)
        assert captured["resume_size"] == 5
        assert dest.read_bytes() == b"HELLO WORLD"
        assert _partials(cache) == []
        assert not _partial_owner_path(partial).exists()

    def test_failed_size_check_deletes_own_temp(self, cache, tmp_path, monkeypatch):
        inc_path = cache / "file.etag.incomplete"
        dest = tmp_path / "dest.gguf"

        def bad_http_get(url, f, resume_size=0, **kw):
            f.write(b"SHORT")
            raise OSError("Consistency check failed: file should be of size 11 but has size 5")

        monkeypatch.setattr("huggingface_hub.file_download.http_get", bad_http_get)
        monkeypatch.setattr("huggingface_hub.file_download._check_disk_space", lambda *a: None)
        with pytest.raises(OSError):
            _call(inc_path, dest, expected_size=11)

        assert sorted(p.name for p in cache.iterdir()) == []

    def test_oversized_orphan_is_discarded(self, cache, tmp_path, monkeypatch, dead_pid):
        inc_path = cache / "file.etag.incomplete"
        big = cache / "file.etag.aaaa1111.incomplete"
        big.write_bytes(b"WAY_TOO_LONG_DATA_HERE")
        _owner(big, *dead_pid)
        dest = tmp_path / "dest.gguf"
        captured = _wire_http(monkeypatch, b"CORRECT")

        _call(inc_path, dest, expected_size=7)

        assert captured["resume_size"] == 0
        assert dest.read_bytes() == b"CORRECT"
        assert not big.exists()

    def test_force_download_discards_a_reusable_orphan_and_keeps_a_live_one(
            self, cache, tmp_path, monkeypatch, dead_pid, live_child):
        inc_path = cache / "file.etag.incomplete"
        orphan = cache / "file.etag.aaaa1111.incomplete"
        orphan.write_bytes(b"OLD")
        _owner(orphan, *dead_pid)
        live = cache / "file.etag.bbbb2222.incomplete"
        live.write_bytes(b"LIVE")
        _owner(live, *live_child)
        dest = tmp_path / "dest.gguf"
        captured = _wire_http(monkeypatch, b"FRESH")

        _call(inc_path, dest, expected_size=5, force=True)

        assert captured["resume_size"] == 0
        assert dest.read_bytes() == b"FRESH"
        assert not orphan.exists() and not _partial_owner_path(orphan).exists()
        assert live.read_bytes() == b"LIVE"
        assert _partials(cache) == [live.name]

    def test_importing_pull_does_not_patch_huggingface_hub(self):
        code = (
            "import localm.model_manager.pull as p\n"
            "import huggingface_hub.file_download as fd\n"
            "print(getattr(fd, '_localm_resumable_patched', False))\n"
            "print(fd._download_to_tmp_and_move.__module__)\n"
            "p._ensure_hf_resumable_download()\n"
            "print(fd._download_to_tmp_and_move.__module__)\n"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, timeout=120, check=True).stdout.split()
        assert out == ["False", "huggingface_hub.file_download",
                       "localm.model_manager.pull"], out


class TestStaleEtagPartials:
    def test_stale_etag_orphan_is_reaped_and_live_owned_kept(
            self, cache, tmp_path, monkeypatch, live_child, dead_pid):
        inc_path = cache / "file.etag.incomplete"
        stale_dead = cache / "file.oldetag.dddd4444.incomplete"
        stale_dead.write_bytes(b"0" * 6)
        _owner(stale_dead, *dead_pid)
        stale_live = cache / "file.oldetag.eeee5555.incomplete"
        stale_live.write_bytes(b"1" * 6)
        _owner(stale_live, *live_child)
        other_file = cache / "other.oldetag.ffff6666.incomplete"
        other_file.write_bytes(b"2" * 6)
        _owner(other_file, *dead_pid)
        dest = tmp_path / "dest.gguf"
        captured = _wire_http(monkeypatch, b"0123456789")

        _call(inc_path, dest, expected_size=10)

        assert captured["resume_size"] == 0, "a stale etag's bytes were reused"
        assert not stale_dead.exists() and not _partial_owner_path(stale_dead).exists()
        assert stale_live.read_bytes() == b"1" * 6
        assert other_file.exists(), "a different file's partial was reaped"
        assert dest.read_bytes() == b"0123456789"


class TestXetResume:
    def test_xet_partial_is_resumed_over_http_range(
            self, cache, tmp_path, monkeypatch, dead_pid):
        inc_path = cache / "file.etag.incomplete"
        orphan = cache / "file.etag.aaaa1111.incomplete"
        orphan.write_bytes(b"HELLO")
        _owner(orphan, *dead_pid)
        dest = tmp_path / "dest.gguf"
        captured = _wire_http(monkeypatch, b" WORLD")
        xet_calls = []
        monkeypatch.setattr("huggingface_hub.file_download.is_xet_available", lambda: True)
        monkeypatch.setattr("huggingface_hub.file_download.xet_get",
                            lambda **kw: xet_calls.append(kw))

        _call(inc_path, dest, expected_size=11, xet_file_data=object())

        assert xet_calls == [], "Xet rewrote the file from byte 0 on a resume"
        assert captured["resume_size"] == 5
        assert dest.read_bytes() == b"HELLO WORLD"

    def test_fresh_download_uses_xet_into_an_owned_temp(
            self, cache, tmp_path, monkeypatch):
        inc_path = cache / "file.etag.incomplete"
        dest = tmp_path / "dest.gguf"
        http_calls = []
        monkeypatch.setattr("huggingface_hub.file_download.http_get",
                            lambda *a, **kw: http_calls.append(1))
        monkeypatch.setattr("huggingface_hub.file_download._check_disk_space", lambda *a: None)
        monkeypatch.setattr("huggingface_hub.file_download.is_xet_available", lambda: True)
        seen = {}

        def fake_xet_get(*, incomplete_path, **kw):
            seen["path"] = Path(incomplete_path)
            seen["owner_present"] = _partial_owner_path(Path(incomplete_path)).exists()
            Path(incomplete_path).write_bytes(b"XET-BYTES")

        monkeypatch.setattr("huggingface_hub.file_download.xet_get", fake_xet_get)

        _call(inc_path, dest, expected_size=9, xet_file_data=object())

        assert http_calls == []
        assert seen["path"].name.startswith("file.etag.") and seen["path"] != inc_path
        assert seen["owner_present"]
        assert dest.read_bytes() == b"XET-BYTES"
        assert sorted(p.name for p in cache.iterdir()) == []


@pytest.fixture()
def hf_env(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    monkeypatch.setattr(mm, "find_by_sha256", lambda *a, **k: [])
    reg_spy = MagicMock()
    monkeypatch.setattr(mm, "_register", reg_spy)
    monkeypatch.setattr(mm, "_register_with_dedup", MagicMock(return_value=True))
    monkeypatch.setattr("huggingface_hub.hf_hub_url",
                        lambda *a, **kw: "https://example.com/test.gguf")
    _ensure_hf_resumable_download()
    return models, reg_spy


def _fake_head(monkeypatch, size: int, etag: "str | None"):
    first = MagicMock()
    first.headers = {"X-Linked-Etag": f'"{etag}"'} if etag else {}
    head = MagicMock(status_code=200, headers={"content-length": str(size)})
    head.history = [first]
    monkeypatch.setattr("requests.head", lambda *a, **kw: head)


def _fake_download(monkeypatch, body: bytes):
    def fake_hf_download(repo_id, filename, local_dir, **kw):
        out = Path(local_dir) / filename
        out.write_bytes(body)
        return str(out)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_download)


def _hf_incomplete(models: Path, rel: str, etag: str) -> Path:
    from huggingface_hub._local_folder import get_local_download_paths
    p = get_local_download_paths(models, rel).incomplete_path(etag)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class TestGgufPullPreflight:
    def test_a_current_etag_orphan_is_charged_and_announced(
            self, hf_env, monkeypatch, capsys, dead_pid):
        models, _ = hf_env
        inc = _hf_incomplete(models, "test.gguf", "newetag")
        partial = inc.with_name(f"{inc.stem}.abcd1234.incomplete")
        partial.write_bytes(b"123456")
        _owner(partial, *dead_pid)
        checked = []
        monkeypatch.setattr(mm, "_check_disk_space",
                            lambda path, size: checked.append(size) or True)
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda *a: None)
        _fake_head(monkeypatch, 10, "newetag")
        _fake_download(monkeypatch, b"0" * 10)

        assert mm._pull_gguf_file("owner/repo:test.gguf", name="test") is True
        assert checked == [4]
        assert "Resuming" in capsys.readouterr().out

    def test_a_stale_etag_partial_is_not_charged_and_not_announced(
            self, hf_env, monkeypatch, capsys, dead_pid):
        models, _ = hf_env
        inc = _hf_incomplete(models, "test.gguf", "oldetag")
        partial = inc.with_name(f"{inc.stem}.abcd1234.incomplete")
        partial.write_bytes(b"123456")
        _owner(partial, *dead_pid)
        checked = []
        monkeypatch.setattr(mm, "_check_disk_space",
                            lambda path, size: checked.append(size) or True)
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda *a: None)
        _fake_head(monkeypatch, 10, "newetag")
        _fake_download(monkeypatch, b"0" * 10)

        assert mm._pull_gguf_file("owner/repo:test.gguf", name="test") is True
        assert checked == [10], "a stale etag's bytes were subtracted from the preflight"
        assert "Resuming" not in capsys.readouterr().out

    def test_a_live_owned_partial_is_not_charged(
            self, hf_env, monkeypatch, capsys, live_child):
        models, _ = hf_env
        inc = _hf_incomplete(models, "test.gguf", "newetag")
        partial = inc.with_name(f"{inc.stem}.abcd1234.incomplete")
        partial.write_bytes(b"123456")
        _owner(partial, *live_child)
        checked = []
        monkeypatch.setattr(mm, "_check_disk_space",
                            lambda path, size: checked.append(size) or True)
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda *a: None)
        _fake_head(monkeypatch, 10, "newetag")
        _fake_download(monkeypatch, b"0" * 10)

        assert mm._pull_gguf_file("owner/repo:test.gguf", name="test") is True
        assert checked == [10]
        assert "Resuming" not in capsys.readouterr().out


class TestGgufPullVerifiesHfDigest:
    def test_digest_mismatch_deletes_the_file_and_registers_nothing(
            self, hf_env, monkeypatch, capsys):
        models, reg_spy = hf_env
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda *a: "ab" * 32)
        _fake_head(monkeypatch, 10, "newetag")
        _fake_download(monkeypatch, b"0" * 10)

        res = mm._pull_gguf_file("owner/repo:test.gguf", name="test")

        assert not (models / "test.gguf").exists(), (
            "a file whose bytes do not match HuggingFace's digest was kept")
        assert not reg_spy.called, "a file failing verification was registered"
        assert res is False
        assert "SHA256 mismatch" in capsys.readouterr().out

    def test_digest_match_registers_under_the_verified_digest(
            self, hf_env, monkeypatch, capsys):
        models, reg_spy = hf_env
        body = b"0" * 10
        digest = hashlib.sha256(body).hexdigest()
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda *a: digest)
        _fake_head(monkeypatch, 10, "newetag")
        _fake_download(monkeypatch, body)

        res = mm._pull_gguf_file("owner/repo:test.gguf", name="test")

        assert res is True
        assert (models / "test.gguf").read_bytes() == body
        assert reg_spy.call_args.kwargs["sha256"] == digest
        assert "SHA256 verified" in capsys.readouterr().out
