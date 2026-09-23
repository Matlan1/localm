# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/bump_comfyui_pin.py: the mechanical half of advancing the ComfyUI pin.

Direct structural sibling of test_bump_llama_pin.py, adapted for a git-clone
pin (one file, two constants, no sha256 table, no MTP-style derived set).

Covers the three properties that make this bump safe to script:

  * COMFYUI_PINNED_COMMIT and COMFYUI_PINNED_VERSION move together in one
    rewrite, and nothing else in the file changes (COMFYUI_REPO,
    COMFYUI_PLACEMENT_MIN_VERSION untouched);
  * a write needs evidence: a receipt from confirm_comfyui_runtime.py naming
    BOTH the target tag AND commit with PASS on every required check;
  * forward-only - a same-or-older tag is refused, never silently applied or
    silently treated as a no-op;
  * unlike bump_llama_pin.py, this script makes NO network calls at all (the
    receipt already carries the cross-checked commit) - proven directly, not
    just claimed in the docstring.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_BUMP = _ROOT / "scripts" / "bump_comfyui_pin.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bump():
    return _load(_BUMP, "bump_comfyui_pin")


OLD_COMMIT = "fe4195f7f4275f2626cbafc703acc3ddde1e5490"
NEW_COMMIT = "1234567890abcdef1234567890abcdef12345678"

CONSTANTS_FIXTURE = f'''# header
COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
COMFYUI_PINNED_COMMIT = "{OLD_COMMIT}"
COMFYUI_PINNED_VERSION = "v0.31.1"

COMFYUI_PLACEMENT_MIN_VERSION = "v0.23.0"

OTHER = 1
'''


def _receipt(tmp_path: Path, tag: str, commit: str, checks: dict, *,
            requirements_changed: bool = False) -> Path:
    p = tmp_path / "receipt.json"
    p.write_text(json.dumps({
        "schema": 1, "tag": tag, "commit": commit, "exit_code": 0,
        "checks": {c: {"verdict": v, "why": f"{c} says {v}"} for c, v in checks.items()},
        "baseline": {"requirements_changed": requirements_changed},
    }), encoding="utf-8")
    return p


def _all_pass(tag=None, commit=None) -> dict:
    return dict.fromkeys(("isolation", "provision", "checkout", "custom_nodes",
                          "localm_patches", "torch_device", "identity",
                          "nodes_registered", "shipped_workflows", "gpu_roundtrip"), "PASS")


# --------------------------------------------------------------------------- #
#  The rewrite                                                                 #
# --------------------------------------------------------------------------- #

def test_rewrite_moves_commit_and_version_together(bump):
    new_text = bump.rewrite(CONSTANTS_FIXTURE, "v0.32.0", NEW_COMMIT)
    assert f'COMFYUI_PINNED_COMMIT = "{NEW_COMMIT}"' in new_text
    assert OLD_COMMIT not in new_text
    assert 'COMFYUI_PINNED_VERSION = "v0.32.0"' in new_text
    assert 'COMFYUI_PINNED_VERSION = "v0.31.1"' not in new_text
    assert 'COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"' in new_text, (
        "the repo URL is never touched by this script")
    assert 'COMFYUI_PLACEMENT_MIN_VERSION = "v0.23.0"' in new_text, (
        "the placement threshold is a different constant and stays untouched")
    assert new_text.endswith("OTHER = 1\n")


def test_rewrite_is_idempotent_and_a_no_op_at_the_current_values(bump):
    """A property of rewrite() itself, as a pure function - main() can never
    actually exercise this at the current-pin values, since _forward_only
    refuses any tag that is not strictly newer before rewrite() is ever
    called (see test_write_applies_once_and_a_second_identical_write_refuses)."""
    once = bump.rewrite(CONSTANTS_FIXTURE, "v0.32.0", NEW_COMMIT)
    twice = bump.rewrite(once, "v0.32.0", NEW_COMMIT)
    assert twice == once

    same = bump.rewrite(CONSTANTS_FIXTURE, "v0.31.1", OLD_COMMIT)
    assert same == CONSTANTS_FIXTURE


def test_each_region_must_be_found_exactly_once(bump):
    with pytest.raises(bump.Refused, match="COMFYUI_PINNED_COMMIT"):
        bump.set_commit(CONSTANTS_FIXTURE + f'\nCOMFYUI_PINNED_COMMIT = "{NEW_COMMIT}"\n',
                        NEW_COMMIT)
    with pytest.raises(bump.Refused, match="COMFYUI_PINNED_VERSION"):
        bump.set_version(CONSTANTS_FIXTURE + '\nCOMFYUI_PINNED_VERSION = "v0.9.9"\n', "v0.32.0")


def test_the_regions_match_exactly_once_on_the_real_tree(bump):
    """Bound to the shipped file: the shapes this script edits exist there,
    once each, so a dry run against the tree is a real dry run."""
    text = bump.CONSTANTS_PATH.read_text(encoding="utf-8")
    assert len(bump._PIN_COMMIT_RE.findall(text)) == 1
    assert len(bump._PIN_VERSION_RE.findall(text)) == 1

    from localm.media import managed_comfy_fresh as mcf
    assert bump._current_pin_version(text) == mcf.COMFYUI_PINNED_VERSION


# --------------------------------------------------------------------------- #
#  Evidence                                                                    #
# --------------------------------------------------------------------------- #

def test_receipt_must_name_the_tag_and_commit_and_pass_every_required_check(bump, tmp_path):
    with pytest.raises(bump.Refused, match="not v0.32.0"):
        bump.load_receipt(_receipt(tmp_path, "v0.30.0", NEW_COMMIT, _all_pass()),
                          "v0.32.0", NEW_COMMIT, ("isolation",))

    with pytest.raises(bump.Refused, match="not " + NEW_COMMIT):
        bump.load_receipt(_receipt(tmp_path, "v0.32.0", OLD_COMMIT, _all_pass()),
                          "v0.32.0", NEW_COMMIT, ("isolation",))

    checks = dict(_all_pass())
    checks["gpu_roundtrip"] = "INCONCLUSIVE"
    p = _receipt(tmp_path, "v0.32.0", NEW_COMMIT, checks)
    with pytest.raises(bump.Refused) as e:
        bump.load_receipt(p, "v0.32.0", NEW_COMMIT, ("isolation", "gpu_roundtrip"))
    assert "gpu_roundtrip" in str(e.value) and "INCONCLUSIVE" in str(e.value)
    assert "gpu_roundtrip says INCONCLUSIVE" in str(e.value), "the receipt's own reason is quoted"
    receipt = bump.load_receipt(p, "v0.32.0", NEW_COMMIT, ("isolation",))
    assert receipt["tag"] == "v0.32.0"

    with pytest.raises(bump.Refused, match="not run"):
        bump.load_receipt(_receipt(tmp_path, "v0.32.0", NEW_COMMIT, {"isolation": "PASS"}),
                          "v0.32.0", NEW_COMMIT, ("isolation", "gpu_roundtrip"))
    with pytest.raises(bump.Refused, match="could not read"):
        bump.load_receipt(tmp_path / "absent.json", "v0.32.0", NEW_COMMIT, ("isolation",))


def test_forward_only_refuses_a_same_or_older_tag(bump):
    with pytest.raises(bump.Refused, match="is not newer than the currently pinned v0.31.1"):
        bump._forward_only("v0.31.1", "v0.31.1")
    with pytest.raises(bump.Refused, match="is not newer than the currently pinned v0.31.1"):
        bump._forward_only("v0.31.1", "v0.9.9")
    bump._forward_only("v0.31.1", "v0.32.0")  # does not raise


def test_a_malformed_tag_is_refused_before_anything_is_read(bump, tmp_path, capsys):
    setup = tmp_path / "managed_comfy_fresh.py"
    setup.write_text(CONSTANTS_FIXTURE, encoding="utf-8")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bump, "CONSTANTS_PATH", setup)
        assert bump.main(["--tag", "latest", "--commit", NEW_COMMIT]) == 1
    assert "not an upstream release tag" in capsys.readouterr().out
    assert setup.read_text(encoding="utf-8") == CONSTANTS_FIXTURE, "a refusal edits nothing"


def test_a_malformed_commit_is_refused_before_anything_is_read(bump, tmp_path, capsys):
    setup = tmp_path / "managed_comfy_fresh.py"
    setup.write_text(CONSTANTS_FIXTURE, encoding="utf-8")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bump, "CONSTANTS_PATH", setup)
        assert bump.main(["--tag", "v0.32.0", "--commit", "not-hex"]) == 1
    assert "not a 40-character hex commit sha" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
#  main(): dry run versus --write                                              #
# --------------------------------------------------------------------------- #

@pytest.fixture
def tree(bump, tmp_path, monkeypatch):
    constants = tmp_path / "managed_comfy_fresh.py"
    constants.write_text(CONSTANTS_FIXTURE, encoding="utf-8")
    monkeypatch.setattr(bump, "CONSTANTS_PATH", constants)
    monkeypatch.setattr(bump, "REPO", tmp_path)
    return constants


def test_dry_run_writes_nothing_and_write_needs_a_receipt(bump, tree, capsys):
    assert bump.main(["--tag", "v0.32.0", "--commit", NEW_COMMIT]) == 0
    out = capsys.readouterr().out
    assert "no receipt given" in out
    assert f'+COMFYUI_PINNED_COMMIT = "{NEW_COMMIT}"' in out
    assert '+COMFYUI_PINNED_VERSION = "v0.32.0"' in out
    assert "dry run: nothing written" in out
    assert "REMAINING STEPS" in out
    assert tree.read_text(encoding="utf-8") == CONSTANTS_FIXTURE

    assert bump.main(["--tag", "v0.32.0", "--commit", NEW_COMMIT, "--write"]) == 1
    assert "REFUSED" in capsys.readouterr().out
    assert tree.read_text(encoding="utf-8") == CONSTANTS_FIXTURE, "a refusal edits nothing"


def test_write_applies_once_and_a_second_identical_write_refuses(bump, tree, tmp_path, capsys):
    """Unlike bump_llama_pin.py (idempotent - re-running with the same tag is
    a safe no-op), this script's forward-only guard refuses a SAME-OR-older
    tag by design (see _forward_only) - a second attempt at the tag the tree
    is already pinned to is a REFUSAL, not a silent no-op, which is what
    actually stops a stale retry from masquerading as a fresh advance."""
    receipt = _receipt(tmp_path, "v0.32.0", NEW_COMMIT, _all_pass())
    assert bump.main(["--tag", "v0.32.0", "--commit", NEW_COMMIT,
                      "--receipt", str(receipt), "--write"]) == 0
    out = capsys.readouterr().out
    assert "wrote managed_comfy_fresh.py" in out
    text = tree.read_text(encoding="utf-8")
    assert f'COMFYUI_PINNED_COMMIT = "{NEW_COMMIT}"' in text
    assert 'COMFYUI_PINNED_VERSION = "v0.32.0"' in text

    assert bump.main(["--tag", "v0.32.0", "--commit", NEW_COMMIT,
                      "--receipt", str(receipt), "--write"]) == 1
    out2 = capsys.readouterr().out
    assert "REFUSED" in out2 and "is not newer than the currently pinned v0.32.0" in out2
    assert tree.read_text(encoding="utf-8") == text, "the refused re-attempt changes nothing"


def test_a_receipt_missing_a_required_check_refuses_by_default(bump, tree, tmp_path, capsys):
    """Default --require is every check this script knows about: a receipt
    missing even one of them is not enough to write."""
    partial = dict(_all_pass())
    del partial["gpu_roundtrip"]
    receipt = _receipt(tmp_path, "v0.32.0", NEW_COMMIT, partial)
    assert bump.main(["--tag", "v0.32.0", "--commit", NEW_COMMIT,
                      "--receipt", str(receipt), "--write"]) == 1
    out = capsys.readouterr().out
    assert "does not confirm" in out
    assert "gpu_roundtrip" in out
    assert tree.read_text(encoding="utf-8") == CONSTANTS_FIXTURE, "a refusal edits nothing"


def test_require_narrows_which_checks_must_pass(bump, tree, tmp_path, capsys):
    receipt = _receipt(tmp_path, "v0.32.0", NEW_COMMIT, {"isolation": "PASS", "provision": "PASS"})
    assert bump.main(["--tag", "v0.32.0", "--commit", NEW_COMMIT, "--receipt", str(receipt),
                      "--require", "isolation", "--require", "provision", "--write"]) == 0
    assert "wrote managed_comfy_fresh.py" in capsys.readouterr().out


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"], ids=["lf", "crlf"])
def test_write_keeps_the_files_own_line_endings(bump, tree, tmp_path, newline):
    """A --write must change only the two edited lines: a file that is LF-only
    stays LF-only and a CRLF file stays CRLF, byte for byte outside the edit."""
    tree.write_bytes(CONSTANTS_FIXTURE.encode("utf-8").replace(b"\n", newline))
    before = tree.read_bytes().split(newline)
    receipt = _receipt(tmp_path, "v0.32.0", NEW_COMMIT, _all_pass())
    assert bump.main(["--tag", "v0.32.0", "--commit", NEW_COMMIT,
                      "--receipt", str(receipt), "--write"]) == 0
    data = tree.read_bytes()
    assert data.count(newline) == data.count(b"\n"), "every line keeps the original ending"
    if newline == b"\n":
        assert b"\r" not in data
    lines = data.split(newline)
    unchanged = [line for line in lines if line in before]
    assert len(lines) - len(unchanged) <= 2, "only the commit and version lines moved"


def test_checklist_mentions_reinstall_requirements_only_when_the_receipt_says_so(bump):
    with_change = bump.checklist("v0.32.0", True)
    without_change = bump.checklist("v0.32.0", False)
    assert "--reinstall-requirements" in with_change
    assert "--reinstall-requirements" not in without_change
    assert "v0.32.0" in with_change and "v0.32.0" in without_change


# --------------------------------------------------------------------------- #
#  No network calls, anywhere - the property that distinguishes this from     #
#  bump_llama_pin.py (which still needs the GitHub API for the sha256 table)  #
# --------------------------------------------------------------------------- #

def test_never_touches_the_network(bump, tree, tmp_path, monkeypatch):
    """urllib.request.urlopen is patched to explode if called at all - both a
    dry run and a --write must complete without ever reaching it, since the
    receipt already carries the cross-checked commit."""
    import urllib.request

    def _boom(*a, **k):
        raise AssertionError("bump_comfyui_pin.py must never open a network connection")
    monkeypatch.setattr(urllib.request, "urlopen", _boom)

    assert bump.main(["--tag", "v0.32.0", "--commit", NEW_COMMIT]) == 0
    receipt = _receipt(tmp_path, "v0.32.0", NEW_COMMIT, _all_pass())
    assert bump.main(["--tag", "v0.32.0", "--commit", NEW_COMMIT,
                      "--receipt", str(receipt), "--write"]) == 0
