# SPDX-License-Identifier: AGPL-3.0-or-later
"""The project config's ``memory_file`` setting takes effect, confined to the
project, only for a trusted project, and is write-protected for a restricted
session."""

import os
import subprocess
import sys

import pytest

from localm.plugins.coder import hook_trust
from localm.plugins.coder.memory import (
    default_memory_file,
    find_injected_memory_file,
    find_memory_file,
    load_memory,
    memory_warning,
    remember,
)
from localm.plugins.coder.tools.base import (
    RESTRICTED_CONFIGURED_MEMORY_MESSAGE,
    _confine,
    restricted_write_refusal,
)


def _project(tmp_path, value, *, toml_value=None):
    root = tmp_path / "proj"
    (root / ".localcoder").mkdir(parents=True)
    rendered = toml_value if toml_value is not None else '"%s"' % value
    (root / ".localcoder" / "config.toml").write_text(
        f"memory_file = {rendered}\n", encoding="utf-8")
    return root


def _trust(monkeypatch, value=True):
    monkeypatch.setattr(hook_trust, "project_trusted", lambda cwd: value)


def test_setting_changes_which_file_is_injected(tmp_path, monkeypatch):
    _trust(monkeypatch)
    root = _project(tmp_path, "docs/notes.md")
    (root / "docs").mkdir()
    (root / "docs" / "notes.md").write_text("configured fact", encoding="utf-8")
    (root / "LOCALCODER.md").write_text("default fact", encoding="utf-8")
    assert find_memory_file(root) == (root / "docs" / "notes.md").resolve()
    assert "configured fact" in load_memory(root)
    assert "default fact" not in load_memory(root)


def test_missing_configured_file_is_not_replaced_by_default_order(tmp_path, monkeypatch):
    _trust(monkeypatch)
    root = _project(tmp_path, "docs/notes.md")
    (root / "LOCALCODER.md").write_text("default fact", encoding="utf-8")
    assert find_memory_file(root) is None
    assert default_memory_file(root) == (root / "docs" / "notes.md").resolve()
    written = remember(root, "new fact")
    assert written == (root / "docs" / "notes.md").resolve()
    assert "new fact" in written.read_text(encoding="utf-8")


def test_localcoder_folder_file_still_needs_trust(tmp_path, monkeypatch):
    _trust(monkeypatch, True)
    root = _project(tmp_path, ".localcoder/other.md")
    (root / ".localcoder" / "other.md").write_text("x", encoding="utf-8")
    assert find_injected_memory_file(root) is not None
    _trust(monkeypatch, False)
    assert find_injected_memory_file(root) is None


@pytest.mark.parametrize("value", [
    "../outside.md",
    "sub/../../outside.md",
    "/etc/outside.md",
    "C:/outside.md",
    "C:outside.md",
    "\\\\server\\share\\x.md",
    "..\\outside.md",
])
def test_escaping_path_is_refused_and_default_order_used(tmp_path, monkeypatch, value):
    _trust(monkeypatch)
    (tmp_path / "outside.md").write_text("SECRET OUTSIDE", encoding="utf-8")
    root = _project(tmp_path, value.replace("\\", "\\\\"))
    (root / "LOCALCODER.md").write_text("default fact", encoding="utf-8")
    assert find_memory_file(root) == root / "LOCALCODER.md"
    assert "SECRET OUTSIDE" not in load_memory(root)
    assert "is not inside the project" in memory_warning(root)
    assert default_memory_file(root) == root / "LOCALCODER.md"


def test_hostile_untrusted_project_injects_nothing_it_configured(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("SECRET OUTSIDE", encoding="utf-8")
    root = _project(tmp_path, "../outside.md")
    assert hook_trust.project_trusted(root) is False
    assert find_memory_file(root) is None
    assert load_memory(root) == ""


def test_untrusted_project_does_not_use_inside_file_and_warns(tmp_path):
    root = _project(tmp_path, "notes.md")
    (root / "notes.md").write_text("inside fact", encoding="utf-8")
    assert hook_trust.project_trusted(root) is False
    assert find_memory_file(root) is None
    assert "not trusted" in memory_warning(root)


def test_changing_memory_file_changes_the_trust_digest(tmp_path):
    a = _project(tmp_path, "a.md")
    raw_a = hook_trust.trust_subject(a)[1]
    digest_a = hook_trust.project_digest(raw_a, [])
    b = _project(tmp_path / "other", "b.md")
    digest_b = hook_trust.project_digest(hook_trust.trust_subject(b)[1], [])
    assert digest_a != digest_b
    assert any("memory_file" in line for line in hook_trust.gated_items(raw_a, []))


def test_non_string_value_is_refused(tmp_path, monkeypatch):
    _trust(monkeypatch)
    root = _project(tmp_path, "", toml_value="42")
    (root / "LOCALCODER.md").write_text("default fact", encoding="utf-8")
    assert find_memory_file(root) == root / "LOCALCODER.md"
    assert "non-empty path string" in memory_warning(root)


def test_restricted_write_refuses_the_configured_file(tmp_path, monkeypatch):
    _trust(monkeypatch)
    root = _project(tmp_path, "docs/notes.md")
    (root / "docs").mkdir()
    assert (restricted_write_refusal(root, "docs/notes.md")
            == RESTRICTED_CONFIGURED_MEMORY_MESSAGE)
    assert (restricted_write_refusal(root, "./docs/../docs/notes.md")
            == RESTRICTED_CONFIGURED_MEMORY_MESSAGE)
    assert restricted_write_refusal(root, "docs/other.md") is None


def test_restricted_write_refuses_configured_file_even_when_untrusted(tmp_path):
    root = _project(tmp_path, "docs/notes.md")
    assert (restricted_write_refusal(root, "docs/notes.md")
            == RESTRICTED_CONFIGURED_MEMORY_MESSAGE)


def test_symlink_alias_of_configured_file_is_refused_by_the_write_tool_itself(tmp_path):
    root = _project(tmp_path, "notes.md")
    (root / "notes.md").write_text("x", encoding="utf-8")
    try:
        os.symlink(root / "notes.md", root / "alias.md")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(PermissionError):
        _confine(root, "alias.md")
    assert restricted_write_refusal(root, "alias.md") is None


def test_symlink_to_outside_is_refused(tmp_path, monkeypatch):
    _trust(monkeypatch)
    outside = tmp_path / "outside.md"
    outside.write_text("SECRET OUTSIDE", encoding="utf-8")
    root = _project(tmp_path, "link.md")
    try:
        os.symlink(outside, root / "link.md")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert find_memory_file(root) is None
    assert "SECRET OUTSIDE" not in load_memory(root)
    assert "is not inside the project" in memory_warning(root)


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS junctions")
def test_junction_to_outside_is_refused(tmp_path, monkeypatch):
    _trust(monkeypatch)
    outside = tmp_path / "outdir"
    outside.mkdir()
    (outside / "m.md").write_text("SECRET OUTSIDE", encoding="utf-8")
    root = _project(tmp_path, "jn/m.md")
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(root / "jn"), str(outside)],
                       capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if r.returncode != 0:
        pytest.skip("junction creation failed")
    assert find_memory_file(root) is None
    assert "SECRET OUTSIDE" not in load_memory(root)


@pytest.mark.parametrize("value", [
    ".git/config",
    ".git/hooks/pre-commit",
    ".localcoder/config.toml",
    ".localcoder/hooks.toml",
    ".localcoder/sub/memory.md",
    "a/.GIT/config",
])
def test_repository_and_coder_config_files_are_refused(tmp_path, monkeypatch, value):
    _trust(monkeypatch)
    root = _project(tmp_path, value)
    target = root / value
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("[core]", encoding="utf-8")
    assert find_memory_file(root) is None
    assert load_memory(root) == ""
    assert "configuration file" in memory_warning(root)
    assert default_memory_file(root) == root / "LOCALCODER.md"


def test_editing_the_configured_file_changes_the_trust_digest(tmp_path):
    root = _project(tmp_path, "notes.md")
    (root / "notes.md").write_text("one", encoding="utf-8")

    def digest():
        _, raw, _, content = hook_trust.trust_subject(root)
        return hook_trust.project_digest(raw, content)

    before = digest()
    (root / "notes.md").write_text("two", encoding="utf-8")
    assert digest() != before


def test_trusted_digest_does_not_survive_a_rewrite_of_the_file(tmp_path, monkeypatch):
    root = _project(tmp_path, "notes.md")
    (root / "notes.md").write_text("one", encoding="utf-8")
    _, raw, trust_path, content = hook_trust.trust_subject(root)
    hook_trust.remember_trust(trust_path, hook_trust.project_digest(raw, content))
    assert hook_trust.project_trusted(root) is True
    assert find_memory_file(root) is not None
    (root / "notes.md").write_text("two", encoding="utf-8")
    assert hook_trust.project_trusted(root) is False
    assert find_memory_file(root) is None
