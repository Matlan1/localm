# SPDX-License-Identifier: AGPL-3.0-or-later
"""A mapped network drive keeps its drive letter through RAG confinement.

``Path.resolve`` turns ``Y:\\docs`` into the share's UNC path, so checking the
drive after resolving never saw a drive. The drive-type and resolve calls are
faked here (no drive is mapped on the test box and none may be created): ``Y:``
is a mapped drive whose root is ``\\\\fakehost\\share``."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="drive letters are Windows-only")

SHARE = "\\\\fakehost\\share"


@pytest.fixture
def mapped_y(monkeypatch):
    real_resolve = Path.resolve
    real_drive_type = ctypes.windll.kernel32.GetDriveTypeW

    def fake_resolve(self, strict=False):
        text = str(self)
        if text[:2].upper() == "Y:":
            rest = text[2:].lstrip("\\")
            return Path(SHARE + ("\\" + rest if rest else ""))
        return real_resolve(self, strict)

    def fake_drive_type(root):
        return 4 if str(root)[:2].upper() == "Y:" else real_drive_type(root)

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW", fake_drive_type)


def test_resolving_a_mapped_drive_path_keeps_the_drive_letter(mapped_y):
    from localm.pathsafe import resolve_keeping_mapped_drive

    assert resolve_keeping_mapped_drive("Y:\\docs\\a.txt") == Path("Y:\\docs\\a.txt")
    assert resolve_keeping_mapped_drive("Y:\\") == Path("Y:\\")


def test_resolving_a_local_path_is_plain_resolve(mapped_y, tmp_path):
    from localm.pathsafe import resolve_keeping_mapped_drive

    assert resolve_keeping_mapped_drive(tmp_path) == tmp_path.resolve()


def test_a_mapped_drive_is_refused_when_network_drives_are_off(mapped_y):
    from localm.rag._store.confine import ConfinementError, hard_floor

    with pytest.raises(ConfinementError) as ei:
        hard_floor("Y:\\docs", allow_net=False)

    assert ei.value.reason == "network_drive_denied"


def test_a_mapped_drive_is_accepted_as_its_letter_path_when_allowed(mapped_y):
    from localm.rag._store.confine import hard_floor

    first = hard_floor("Y:\\docs", allow_net=True)

    assert first == Path("Y:\\docs")
    assert hard_floor(first, allow_net=True) == first


def test_confinement_returns_a_path_that_confines_again(mapped_y):
    from localm.rag.store import confine_index_path, indexing_policy

    policy = {**indexing_policy(cfg={}), "mode": "whitelist",
              "allowed": [Path("Y:\\")], "allow_network_drives": True}

    once = confine_index_path("Y:\\docs\\a.txt", policy)
    twice = confine_index_path(once, policy)

    assert once == twice == Path("Y:\\docs\\a.txt")


def test_confinement_with_network_drives_off_refuses_a_mapped_drive_in_both_modes(mapped_y):
    from localm.rag.store import ConfinementError, confine_index_path, indexing_policy

    for mode in ("whitelist", "blacklist"):
        policy = {**indexing_policy(cfg={}), "mode": mode, "allowed": [Path("Y:\\")],
                  "allow_network_drives": False}
        with pytest.raises(ConfinementError) as ei:
            confine_index_path("Y:\\docs", policy)
        assert ei.value.reason == "network_drive_denied"


def test_a_credential_folder_on_a_mapped_drive_is_still_refused(mapped_y):
    from localm.rag._store.confine import ConfinementError, hard_floor

    with pytest.raises(ConfinementError) as ei:
        hard_floor("Y:\\home\\.ssh", allow_net=True)

    assert ei.value.reason == "credential"


def test_a_path_on_a_mapped_drive_that_resolves_off_the_share_is_refused(mapped_y, monkeypatch):
    from localm.rag._store.confine import ConfinementError, hard_floor
    escaped = Path("\\\\elsewhere\\other\\x")
    inner = Path.resolve

    def resolve(self, strict=False):
        return escaped if str(self).upper().startswith("Y:\\LINK") else inner(self, strict)

    monkeypatch.setattr(Path, "resolve", resolve)

    with pytest.raises(ConfinementError) as ei:
        hard_floor("Y:\\link", allow_net=True)

    assert ei.value.reason == "unc_or_device"
