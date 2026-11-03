# SPDX-License-Identifier: AGPL-3.0-or-later
"""A managed-ComfyUI source fetch must not fail with Windows' legacy 260-char
MAX_PATH limit.

A custom node under a sufficiently nested LOCALM_HOME puts some of its files past
260 characters. The archive is unpacked and applied through extended-length
paths, so that works whatever the OS-level LongPathsEnabled policy is.
"""

from __future__ import annotations

import os

from localm.media import managed_comfy_fresh as fresh
from localm.media import source_archive
from tests._source_archives import FakeGitHub


def test_a_tree_deeper_than_max_path_is_fetched_and_replaced(monkeypatch, tmp_path):
    deep_rel = "/".join(["a_rather_long_folder_name_for_testing"] * 6) + "/leaf.py"
    hub = FakeGitHub(tmp_path / "hub").install(monkeypatch)
    repo, commit = hub.add("o", "Node", {"__init__.py": "x\n", deep_rel: "deep\n"})
    dest = tmp_path / ("n" * 40) / "custom_nodes" / "Node"
    assert len(str(dest / deep_rel)) > 260
    ok, out = fresh._fetch_at_commit(repo, dest, commit, on_progress=None)
    assert ok, out
    assert os.path.isfile(source_archive._long(dest / deep_rel))

    repo2, commit2 = hub.add("o", "Node", {"__init__.py": "y\n", deep_rel: "deeper\n"})
    new = tmp_path / ("m" * 40) / "new"
    source_archive.fetch_tree(repo2, commit2, new)
    change = source_archive.apply_tree(new, dest, source_archive.tree_files(dest),
                                       tmp_path / ("b" * 40) / "backup")
    with open(source_archive._long(dest / deep_rel), encoding="utf-8") as f:
        assert f.read() == "deeper\n"
    assert change.rollback() == []
    with open(source_archive._long(dest / deep_rel), encoding="utf-8") as f:
        assert f.read() == "deep\n"
