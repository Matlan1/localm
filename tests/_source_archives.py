# SPDX-License-Identifier: AGPL-3.0-or-later
"""GitHub-style source archives for tests, served without a network.

``FakeGitHub.add`` builds a zip laid out like GitHub's archive of a commit (one
top folder ``<repo>-<commit>``, the commit id as the zip comment) and
``FakeGitHub.install`` points ``localm.media.source_archive.download`` at those
files, recording every URL asked for.
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path


def fake_commit(seed: str) -> str:
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()


def make_archive(dest: Path, repo_name: str, commit: str, files: dict,
                 *, comment: str | None = None, executable: tuple = ()) -> Path:
    """Write a GitHub-style zip of *files* (``{relative path: text}``) to *dest*."""
    top = f"{repo_name}-{commit}"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zipfile.ZipInfo(top + "/"), b"")
        for rel, text in files.items():
            info = zipfile.ZipInfo(f"{top}/{rel}")
            info.external_attr = (0o100755 if rel in executable else 0o100644) << 16
            zf.writestr(info, text.encode("utf-8") if isinstance(text, str) else text)
        zf.comment = (commit if comment is None else comment).encode("ascii")
    return dest


class FakeGitHub:
    """Archives keyed by ``(owner, repo, commit)``, served by URL."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.archives: dict = {}
        self.requested: list = []

    def add(self, owner: str, repo: str, files: dict, *, commit: str | None = None,
            comment: str | None = None) -> tuple:
        """Add an archive; returns ``(repo URL, commit)``."""
        commit = commit or fake_commit(f"{owner}/{repo}/{sorted(files.items())!r}")
        zpath = self.root / f"{owner}-{repo}-{commit}.zip"
        make_archive(zpath, repo, commit, files, comment=comment)
        url = f"https://codeload.github.com/{owner}/{repo}/zip/{commit}"
        self.archives[url] = zpath
        return f"https://github.com/{owner}/{repo}.git", commit

    def install(self, monkeypatch) -> FakeGitHub:
        from localm.media import source_archive

        def _download(url, dest):
            self.requested.append(url)
            if url not in self.archives:
                raise source_archive.ArchiveError(f"downloading {url} failed: HTTP 404")
            shutil.copyfile(self.archives[url], dest)

        monkeypatch.setattr(source_archive, "download", _download)
        return self


def assert_shipped_pins_carry_a_sha256(root: Path) -> None:
    """Run ``test_the_shipped_pins_carry_a_sha256`` against the
    ``managed_comfy_fresh.py`` and ``source_archive.py`` under
    ``root/localm/media`` in place of the installed ones. Raises its
    AssertionError when a pin there has no sha256. Writes no ``.pyc`` there;
    a ``.pyc`` already under *root* could be loaded in place of an edited file
    of the same size and modification second."""
    import importlib.util
    import sys

    import pytest

    from localm.media import managed_comfy_fresh as fresh
    from localm.media import source_archive
    from tests.test_source_archive import test_the_shipped_pins_carry_a_sha256

    with pytest.MonkeyPatch.context() as mp:
        # No .pyc is written under *root*, so every call compiles the files as they
        # are on disk. See test_a_simulated_bump_of_the_real_tree_passes_the_shipped_pins_test.
        mp.setattr(sys, "dont_write_bytecode", True)
        loaded = {}
        for name in ("managed_comfy_fresh", "source_archive"):
            mod_name = f"_bumped_{name}"
            spec = importlib.util.spec_from_file_location(
                mod_name, root / "localm" / "media" / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            mp.setitem(sys.modules, mod_name, mod)
            spec.loader.exec_module(mod)
            loaded[name] = mod
        mp.setattr(source_archive, "ARCHIVE_SHA256", loaded["source_archive"].ARCHIVE_SHA256)
        mp.setattr(fresh, "COMFYUI_PINNED_COMMIT",
                   loaded["managed_comfy_fresh"].COMFYUI_PINNED_COMMIT)
        mp.setattr(fresh, "CLASS_TYPE_TO_NODE", loaded["managed_comfy_fresh"].CLASS_TYPE_TO_NODE)
        test_the_shipped_pins_carry_a_sha256()
