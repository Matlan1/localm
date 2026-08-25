# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for three issues found by the unowned-surface audit:."""
from __future__ import annotations

import io
import json
import logging
import os
import tarfile

import pytest

from localm import instances
from localm.setup_llama import (
    ArtifactError,
    _repo_runtime_lib,
    _safe_extractall_tar,
)

# _safe_extractall_tar is the Python<3.12 path (no extraction filter). Calling it
# directly on a 3.12+ test interpreter trips 3.12's "use the filter argument"
# deprecation on the post-validation extractall - but members are already
# path-validated above it, so it is safe here. Silence ONLY that one message; in
# production this helper is reached only on <3.12, where the warning never exists.
pytestmark = pytest.mark.filterwarnings("ignore:Python 3.14 will:DeprecationWarning")


# --------------------------------------------------------------------------- #
#  1. tar path-traversal fallback (HIGH)
# --------------------------------------------------------------------------- #

def _write_tar(path, *, files=(), symlinks=()):
    with tarfile.open(path, "w") as tf:
        for name, data in files:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        for name, target in symlinks:
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tf.addfile(info)


def _extract(tar_path, dest):
    with tarfile.open(tar_path) as tf:
        _safe_extractall_tar(tf, dest)


def test_clean_tar_extracts(tmp_path):
    tar = tmp_path / "ok.tar"
    _write_tar(tar, files=[("lib/a.so", b"x"), ("b.txt", b"y")])
    dest = tmp_path / "out"
    dest.mkdir()
    _extract(tar, dest)
    assert (dest / "lib" / "a.so").read_bytes() == b"x"
    assert (dest / "b.txt").read_bytes() == b"y"


@pytest.mark.parametrize("evil_name", [
    "../escape.txt",
    "../../etc/passwd",
    "/abs/evil.txt",
    "sub/../../escape.txt",
])
def test_traversal_member_is_refused(tmp_path, evil_name):
    tar = tmp_path / "evil.tar"
    _write_tar(tar, files=[(evil_name, b"pwn")])
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(ArtifactError):
        _extract(tar, dest)
    # nothing escaped
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "etc").exists()


def test_drive_letter_member_is_refused(tmp_path):
    tar = tmp_path / "evil.tar"
    _write_tar(tar, files=[("Z:\\windows\\evil.dll", b"pwn")])
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(ArtifactError):
        _extract(tar, dest)


def test_escaping_symlink_is_refused(tmp_path):
    tar = tmp_path / "evil.tar"
    _write_tar(tar, symlinks=[("link", "../../../../etc/passwd")])
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(ArtifactError):
        _extract(tar, dest)


def test_absolute_symlink_target_is_refused(tmp_path):
    tar = tmp_path / "evil.tar"
    _write_tar(tar, symlinks=[("link", "/etc/passwd")])
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(ArtifactError):
        _extract(tar, dest)


def test_contained_symlink_is_allowed(tmp_path):
    # A symlink whose target stays inside dest is fine (matches the data filter).
    tar = tmp_path / "ok.tar"
    _write_tar(tar, files=[("real.txt", b"hi")], symlinks=[("link.txt", "real.txt")])
    dest = tmp_path / "out"
    dest.mkdir()
    _extract(tar, dest)   # must not raise
    assert (dest / "real.txt").read_bytes() == b"hi"


# --------------------------------------------------------------------------- #
#  2. runtime-wheel import fallback surfaces, does not hard-fail (rule 5)
# --------------------------------------------------------------------------- #

def test_repo_runtime_lib_falls_back_without_raising():
    # The wheel is not installed in the test env, so the import fails and the
    # repo-relative fallback path is returned (never raised).
    p = _repo_runtime_lib()
    assert p.parts[-2:] == ("localm_llama_runtime", "lib")


def test_repo_runtime_lib_logs_the_swallowed_import(monkeypatch, caplog):
    # A BROKEN install (import raises something other than not-found) must be
    # surfaced at debug level, not silently swallowed.
    import builtins
    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name == "localm_llama_runtime":
            raise RuntimeError("simulated broken wheel")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    with caplog.at_level(logging.DEBUG, logger="localm"):
        p = _repo_runtime_lib()
    assert p.parts[-2:] == ("localm_llama_runtime", "lib")
    assert any("localm_llama_runtime import failed" in r.getMessage()
               for r in caplog.records), "the swallowed import must be logged"


# --------------------------------------------------------------------------- #
#  3. reap_stale reaps a null-pid (malformed) entry instead of keeping it
# --------------------------------------------------------------------------- #

def _raw_entry(home, iid, pid):
    d = instances.run_dir(home)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{iid}.json").write_text(json.dumps({
        "instance_id": iid, "pid": pid, "port": 1, "host": "127.0.0.1",
        "root_dir": str(home), "mode": "normal",
    }), encoding="utf-8")


def test_reap_stale_reaps_null_pid_entry(tmp_path):
    _raw_entry(tmp_path, "nullpid", None)          # malformed: pid is JSON null
    _raw_entry(tmp_path, "alive", os.getpid())     # control: this process is alive
    removed = instances.reap_stale(tmp_path)        # DEFAULT probe (the fixed lambda)
    assert "nullpid" in removed, "a null-pid entry must be reaped, not kept forever"
    assert "alive" not in removed
    assert not instances.registry_path(tmp_path, "nullpid").exists()
    assert instances.registry_path(tmp_path, "alive").exists()
