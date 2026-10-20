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
    this_pid_space,
)


def _owner(path: Path, pid: int, start, space: "str | None" = "this") -> None:
    """Write the owner record for *path* naming *pid* / start identity *start*,
    in this process's pid space unless *space* names another one."""
    if space == "this":
        space = this_pid_space()
    _partial_owner_path(path).write_text(
        json.dumps({"pid": pid, "space": space, "start": start}),
        encoding="utf-8")


@pytest.fixture()
def live_child():
    """A real child process that stays alive for the test, with its start
    identity."""
    from localm.instances import process_start_identity
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield p.pid, process_start_identity(p.pid)
    finally:
        p.kill()
        p.wait()


@pytest.fixture()
def dead_pid():
    """The pid and start identity of a child that has already exited."""
    from localm.instances import process_start_identity
    p = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
                         stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    start = process_start_identity(p.pid)
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

    @pytest.mark.parametrize("space", ["0123456789abcdef", None])
    def test_a_live_pid_from_another_or_no_pid_space_keeps_its_partial(
            self, cache, tmp_path, monkeypatch, live_child, space):
        """A record from another pid space, or one naming none, is not
        evidence that its live pid is a different process: the partial is
        neither adopted nor deleted."""
        pid, _ = live_child
        inc_path = cache / "file.etag.incomplete"
        other = cache / "file.etag.bbbb2222.incomplete"
        other.write_bytes(b"OTHER-PID-SPACE")
        _owner(other, pid, started_an_hour_earlier(start_identity_of(pid)),
               space=space)
        dest = tmp_path / "dest.gguf"
        captured = _wire_http(monkeypatch, b"FRESH")

        _call(inc_path, dest, expected_size=5)

        assert (other.read_bytes() if other.exists() else None) == (
            b"OTHER-PID-SPACE"), "a partial of another pid space was touched"
        assert _partial_owner_path(other).exists()
        assert captured["resume_size"] == 0
        assert dest.read_bytes() == b"FRESH"

    def test_a_live_partial_of_a_cloned_machine_reaching_the_folder_remotely(
            self, cache, tmp_path, monkeypatch, live_child):
        """A cloned machine shares this machine's host id but has another host
        name. Its record says the cache folder was not on its own disk, so
        its partial is neither adopted nor deleted although the folder is
        local here and the pid here names another process."""
        import platform

        from localm import instances
        host = instances.host_id()
        if host is None:
            pytest.skip("no machine identifier is readable here")
        if not instances.path_is_local(cache):
            pytest.skip("the test folder is not on a local disk here")
        with monkeypatch.context() as m:
            m.setattr(platform, "node", lambda: "cloned-vm-other-name")
            m.setattr(instances, "_PID_SPACE", None)
            clone_space = instances.pid_space_id()
        assert clone_space != this_pid_space(), "the injected name took no effect"
        pid, _ = live_child
        recorded = started_an_hour_earlier(start_identity_of(pid))
        assert instances.start_identity_differs(
            recorded, instances.process_start_identity(pid))
        inc_path = cache / "file.etag.incomplete"
        other = cache / "file.etag.bbbb2222.incomplete"
        other.write_bytes(b"CLONE-IN-FLIGHT")
        _partial_owner_path(other).write_text(json.dumps(
            {"pid": pid, "space": clone_space, "host": host, "local": False,
             "start": recorded}), encoding="utf-8")
        dest = tmp_path / "dest.gguf"
        captured = _wire_http(monkeypatch, b"FRESH")

        _call(inc_path, dest, expected_size=5)

        assert (other.read_bytes() if other.exists() else None) == (
            b"CLONE-IN-FLIGHT"), "a cloned machine's live partial was touched"
        assert _partial_owner_path(other).exists()
        assert captured["resume_size"] == 0
        assert dest.read_bytes() == b"FRESH"

    @pytest.mark.parametrize("whose", ["dead", "this-process"])
    def test_a_partial_from_another_pid_space_is_left_alone_whatever_its_pid(
            self, cache, tmp_path, monkeypatch, dead_pid, whose):
        """A record from another pid space names a pid this process cannot
        look up, whether that number is dead here or equal to this process's
        own: the partial is neither adopted nor deleted."""
        pid = dead_pid[0] if whose == "dead" else os.getpid()
        inc_path = cache / "file.etag.incomplete"
        other = cache / "file.etag.bbbb2222.incomplete"
        other.write_bytes(b"OTHER-PID-SPACE")
        _owner(other, pid, None, space="0123456789abcdef")
        dest = tmp_path / "dest.gguf"
        captured = _wire_http(monkeypatch, b"FRESH")

        _call(inc_path, dest, expected_size=5)

        assert (other.read_bytes() if other.exists() else None) == (
            b"OTHER-PID-SPACE"), "a partial of another pid space was touched"
        assert _partial_owner_path(other).exists()
        assert captured["resume_size"] == 0
        assert dest.read_bytes() == b"FRESH"

    @pytest.mark.parametrize("local,writer_local", [
        (True, True), (False, True), (True, False), (True, None)])
    def test_a_dead_owners_partial_from_before_a_host_rename(
            self, cache, tmp_path, monkeypatch, dead_pid, local, writer_local):
        """A record written before the host was renamed carries another pid
        space id but this machine's host id: its dead owner's partial is
        adopted when the record says its writer saw the cache folder on a
        local disk and the folder is on a local disk here, and left alone
        otherwise, including when the record has no ``local`` field (a cloned
        machine with another host name that reaches the folder over a network
        share writes ``local: false``)."""
        adopted = local and writer_local is True
        import platform

        from localm import instances
        host = instances.host_id()
        if host is None:
            pytest.skip("no machine identifier is readable here")
        with monkeypatch.context() as m:
            m.setattr(platform, "node", lambda: "renamed-host-before")
            m.setattr(instances, "_PID_SPACE", None)
            old_space = instances.pid_space_id()
        assert old_space != this_pid_space(), "the injected rename took no effect"
        if not local:
            monkeypatch.setattr(instances, "path_is_local", lambda p: False)
        elif not instances.path_is_local(cache):
            pytest.skip("the test folder is not on a local disk here")
        inc_path = cache / "file.etag.incomplete"
        orphan = cache / "file.etag.dddd4444.incomplete"
        orphan.write_bytes(b"HELLO")
        rec = {"pid": dead_pid[0], "space": old_space, "host": host,
               "start": dead_pid[1]}
        if writer_local is not None:
            rec["local"] = writer_local
        _partial_owner_path(orphan).write_text(json.dumps(rec),
                                               encoding="utf-8")
        dest = tmp_path / "dest.gguf"
        captured = _wire_http(monkeypatch, b" WORLD" if adopted else b"FRESH")

        _call(inc_path, dest, expected_size=11 if adopted else 5)

        if adopted:
            assert captured["resume_size"] == 5
            assert dest.read_bytes() == b"HELLO WORLD"
            assert not orphan.exists() and not _partial_owner_path(orphan).exists()
        else:
            assert captured["resume_size"] == 0
            assert orphan.read_bytes() == b"HELLO"
            assert _partial_owner_path(orphan).exists()

    def test_the_owner_record_names_this_process_its_pid_space_and_start(
            self, cache):
        from localm.instances import host_id, path_is_local
        from localm.instances import process_start_identity
        from localm.model_manager.pull import _pid_space_id, _write_partial_owner
        p = cache / "file.etag.cccc3333.incomplete"
        p.write_bytes(b"")
        _write_partial_owner(p)
        rec = json.loads(_partial_owner_path(p).read_text(encoding="utf-8"))
        assert rec == {"pid": os.getpid(), "space": _pid_space_id(),
                       "host": host_id(), "local": path_is_local(cache),
                       "start": process_start_identity(os.getpid())}

    def test_partials_without_owner_record_are_resumed_under_the_download_lock(
            self, cache, tmp_path, monkeypatch):
        inc_path = cache / "file.etag.incomplete"
        unknown = cache / "file.etag.cccc3333.incomplete"
        unknown.write_bytes(b"HELLO")
        legacy = cache / "file.etag.incomplete"
        legacy.write_bytes(b"HE")
        dest = tmp_path / "dest.gguf"
        captured = _wire_http(monkeypatch, b" WORLD")

        _call(inc_path, dest, expected_size=11)

        assert captured["resume_size"] == 5
        assert dest.read_bytes() == b"HELLO WORLD"
        assert _partials(cache) == []

    def test_a_partial_without_owner_record_is_not_counted_while_the_lock_is_held(
            self, cache):
        inc_path = cache / "file.etag.incomplete"
        stock = cache / "file.etag.incomplete"
        stock.write_bytes(b"HELLO")

        from localm.model_manager.pull import _reusable_partial_bytes
        assert _reusable_partial_bytes(inc_path, 11, lock_free=False) == 0
        assert _reusable_partial_bytes(inc_path, 11, lock_free=True) == 5

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


    def test_a_xet_failure_before_the_temp_exists_leaves_no_owner_record(
            self, cache, tmp_path, monkeypatch):
        inc_path = cache / "file.etag.incomplete"
        dest = tmp_path / "dest.gguf"
        monkeypatch.setattr("huggingface_hub.file_download._check_disk_space", lambda *a: None)
        monkeypatch.setattr("huggingface_hub.file_download.is_xet_available", lambda: True)
        seen = {}

        def failing_xet_get(*, incomplete_path, **kw):
            seen["owner_present"] = _partial_owner_path(Path(incomplete_path)).exists()
            raise ConnectionError("token refresh failed")

        monkeypatch.setattr("huggingface_hub.file_download.xet_get", failing_xet_get)

        with pytest.raises(ConnectionError):
            _call(inc_path, dest, expected_size=9, xet_file_data=object())

        assert seen["owner_present"], "the transfer started without an owner record"
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
    monkeypatch.setattr("localm.model_manager.pull._policy_head", lambda *a, **kw: head)


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


def _spy_progress_total(monkeypatch) -> list:
    """Record the total each model download's progress is measured against."""
    from localm.model_manager import pull
    real = pull._download_progress
    totals = []

    def spy(parts, total_size, **kw):
        totals.append(total_size)
        return real(parts, total_size, **kw)

    monkeypatch.setattr(pull, "_download_progress", spy)
    return totals


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

    def test_charges_for_a_same_repo_projector_it_will_auto_attach(
            self, hf_env, monkeypatch):
        import huggingface_hub

        class _FakeHfApi:
            def __init__(self, *a, **kw):
                pass

            def list_repo_files(self, repo_id):
                return ["test.gguf", "mmproj-test-f16.gguf"]

        monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
        checked = []
        monkeypatch.setattr(mm, "_check_disk_space",
                            lambda path, size: checked.append(size) or True)
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda *a: None)
        totals = _spy_progress_total(monkeypatch)
        _fake_head(monkeypatch, 10, "newetag")
        _fake_download(monkeypatch, b"0" * 10)

        assert mm._pull_gguf_file("owner/repo:test.gguf", name="test") is True
        assert checked == [20], (
            "the preflight must charge for the model AND the same-repo "
            "projector it is about to auto-attach")
        assert totals == [10], (
            "the model's download progress counted the projector's bytes")

    @pytest.mark.parametrize("on_disk, charged", [(10, 10), (7, 20)],
                             ids=["same-size", "other-size"])
    def test_a_projector_name_already_in_the_models_folder(
            self, hf_env, monkeypatch, on_disk, charged):
        """A file of the projector's name and published size is not charged
        again; one of another size is a different file, so the projector is
        downloaded beside it and charged."""
        import huggingface_hub
        models, _ = hf_env
        (models / "mmproj-test-f16.gguf").write_bytes(b"P" * on_disk)

        class _FakeHfApi:
            def __init__(self, *a, **kw):
                pass

            def list_repo_files(self, repo_id):
                return ["test.gguf", "mmproj-test-f16.gguf"]

        monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
        checked = []
        monkeypatch.setattr(mm, "_check_disk_space",
                            lambda path, size: checked.append(size) or True)
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda *a: None)
        totals = _spy_progress_total(monkeypatch)
        _fake_head(monkeypatch, 10, "newetag")
        _fake_download(monkeypatch, b"0" * 10)

        assert mm._pull_gguf_file("owner/repo:test.gguf", name="test") is True
        assert checked == [charged]
        assert totals == [10]

    def test_does_not_charge_a_projector_when_the_repo_has_none(
            self, hf_env, monkeypatch):
        import huggingface_hub

        class _FakeHfApi:
            def __init__(self, *a, **kw):
                pass

            def list_repo_files(self, repo_id):
                return ["test.gguf"]

        monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
        checked = []
        monkeypatch.setattr(mm, "_check_disk_space",
                            lambda path, size: checked.append(size) or True)
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda *a: None)
        _fake_head(monkeypatch, 10, "newetag")
        _fake_download(monkeypatch, b"0" * 10)

        assert mm._pull_gguf_file("owner/repo:test.gguf", name="test") is True
        assert checked == [10]

    def test_does_not_charge_a_projector_for_an_explicit_mmproj_spec(
            self, hf_env, monkeypatch):
        """An explicit --mmproj is a separate, user-named download path
        (_fetch_explicit_mmproj) that this preflight does not probe, so the
        same-repo listing must not even be consulted."""
        import huggingface_hub

        class _FailingHfApi:
            def __init__(self, *a, **kw):
                pass

            def list_repo_files(self, repo_id):
                raise AssertionError("same-repo listing must not be probed")

        monkeypatch.setattr(huggingface_hub, "HfApi", _FailingHfApi)
        checked = []
        monkeypatch.setattr(mm, "_check_disk_space",
                            lambda path, size: checked.append(size) or True)
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda *a: None)
        _fake_head(monkeypatch, 10, "newetag")
        _fake_download(monkeypatch, b"0" * 10)

        assert mm._pull_gguf_file(
            "owner/repo:test.gguf", name="test",
            mmproj_spec="owner/repo:mmproj-test-f16.gguf") is True
        assert checked == [10]


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


class TestOwnerlessPartialResume:
    """A temp file written by huggingface_hub itself, or by a localm that wrote no
    owner record, resumes whenever no download of that file holds its lock."""

    @staticmethod
    def _preflight(hf_env, monkeypatch, partial_bytes: bytes):
        models, _ = hf_env
        inc = _hf_incomplete(models, "test.gguf", "newetag")
        inc.write_bytes(partial_bytes)
        checked = []
        monkeypatch.setattr(mm, "_check_disk_space",
                            lambda path, size: checked.append(size) or True)
        monkeypatch.setattr(mm, "_hf_file_sha256", lambda *a: None)
        _fake_head(monkeypatch, 10, "newetag")
        _fake_download(monkeypatch, b"0" * 10)
        return models, checked

    def test_preflight_charges_and_announces_a_stock_partial(
            self, hf_env, monkeypatch, capsys):
        _, checked = self._preflight(hf_env, monkeypatch, b"123456")

        assert mm._pull_gguf_file("owner/repo:test.gguf", name="test") is True
        assert checked == [4]
        assert "Resuming" in capsys.readouterr().out

    def test_preflight_ignores_it_while_another_download_holds_the_file_lock(
            self, hf_env, monkeypatch, capsys):
        from filelock import FileLock
        from huggingface_hub._local_folder import get_local_download_paths
        models, checked = self._preflight(hf_env, monkeypatch, b"123456")
        holder = FileLock(str(get_local_download_paths(models, "test.gguf").lock_path))
        holder.acquire()
        try:
            assert mm._pull_gguf_file("owner/repo:test.gguf", name="test") is True
        finally:
            holder.release()
        assert checked == [10], "a partial a live download holds was charged as reusable"
        assert "Resuming" not in capsys.readouterr().out

    def test_the_lock_probe_sees_a_held_lock_and_a_free_one(self, tmp_path):
        from filelock import FileLock
        from localm.model_manager.pull import _lock_file_is_free
        lock_path = tmp_path / "f.lock"
        assert _lock_file_is_free(lock_path) is True
        holder = FileLock(str(lock_path))
        holder.acquire()
        try:
            assert _lock_file_is_free(lock_path) is False
        finally:
            holder.release()
        assert _lock_file_is_free(lock_path) is True

    def test_a_stale_etag_partial_with_no_owner_record_is_reaped_in_a_local_dir_folder(
            self, tmp_path, monkeypatch):
        folder = tmp_path / ".cache" / "huggingface" / "download"
        folder.mkdir(parents=True)
        inc_path = folder / "file.etag.incomplete"
        stale = folder / "file.oldetag.incomplete"
        stale.write_bytes(b"0" * 6)
        dest = tmp_path / "dest.gguf"
        captured = _wire_http(monkeypatch, b"0123456789")

        _call(inc_path, dest, expected_size=10)

        assert not stale.exists()
        assert captured["resume_size"] == 0
        assert dest.read_bytes() == b"0123456789"

    def test_a_stale_etag_partial_with_no_owner_record_is_kept_elsewhere(
            self, cache, tmp_path, monkeypatch):
        inc_path = cache / "file.etag.incomplete"
        stale = cache / "file.oldetag.incomplete"
        stale.write_bytes(b"0" * 6)
        dest = tmp_path / "dest.gguf"
        _wire_http(monkeypatch, b"0123456789")

        _call(inc_path, dest, expected_size=10)

        assert stale.read_bytes() == b"0" * 6

    def test_progress_counts_a_stock_partial_only_while_the_file_lock_is_free(
            self, tmp_path, monkeypatch, capsys):
        import json as _json
        from filelock import FileLock
        from huggingface_hub._local_folder import get_local_download_paths
        monkeypatch.setenv("LOCALM_PROGRESS_JSON", "1")
        base = tmp_path / "models"
        base.mkdir()
        paths = get_local_download_paths(base, "m.gguf")
        paths.incomplete_path("deadbeef").write_bytes(b"z" * 123)

        def seen() -> list:
            out = capsys.readouterr().out
            return [_json.loads(line.split(mm.PROGRESS_SENTINEL, 1)[1])["downloaded"]
                    for line in out.splitlines() if mm.PROGRESS_SENTINEL in line]

        with mm._download_progress([base / "m.gguf"], 1000, base_dir=base,
                                   rel_parts=["m.gguf"]) as prog:
            prog.ok()
        assert 123 in seen()[:-1]

        holder = FileLock(str(paths.lock_path))
        holder.acquire()
        try:
            with mm._download_progress([base / "m.gguf"], 1000, base_dir=base,
                                       rel_parts=["m.gguf"]) as prog:
                prog.ok()
            assert 123 not in seen()[:-1]
        finally:
            holder.release()
