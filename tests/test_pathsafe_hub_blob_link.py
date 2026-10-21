# SPDX-License-Identifier: AGPL-3.0-or-later
"""``confined_model_file`` accepts a hub-cache link into the repo's own
``blobs`` folder, and refuses one when ``blobs`` is itself a link out of the
repo."""

import os
from pathlib import Path

import pytest

from localm.pathsafe import confined_model_file


def _link(link: Path, target: Path, directory: bool = False) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(target, link, target_is_directory=directory)
    except OSError as e:
        pytest.skip(f"symlink creation not permitted here: {e}")


def _snapshot(tmp_path: Path) -> Path:
    snap = tmp_path / "models--org--m" / "snapshots" / ("a" * 40)
    snap.mkdir(parents=True)
    return snap


def test_a_link_into_the_repos_own_blobs_folder_is_accepted(tmp_path):
    snap = _snapshot(tmp_path)
    blobs = snap.parent.parent / "blobs"
    blobs.mkdir()
    (blobs / "abc").write_bytes(b"{}")
    _link(snap / "config.json", Path("..") / ".." / "blobs" / "abc")
    got = confined_model_file(snap, "config.json")
    assert got == snap.resolve() / "config.json"


def test_a_blobs_folder_that_is_a_link_out_of_the_repo_is_refused(tmp_path):
    snap = _snapshot(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_bytes(b"secret")
    _link(snap.parent.parent / "blobs", outside, directory=True)
    _link(snap / "config.json", Path("..") / ".." / "blobs" / "secret")
    with pytest.raises(ValueError):
        confined_model_file(snap, "config.json")
