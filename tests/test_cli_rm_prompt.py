# SPDX-License-Identifier: AGPL-3.0-or-later
"""The `localm rm` confirmation prompt must describe what the deletion actually
does.

The prompt and the real delete gate in remove_model() share ONE predicate:
path.is_relative_to(MODELS_DIR). A str(path).startswith(str(MODELS_DIR)) test
also matches a SIBLING directory, so an entry at <data dir>/models-old/x.gguf
would be announced as "PERMANENTLY deletes" when remove_model would in fact
only unregister the name.

These tests run the REAL CLI command against a REAL registry file and a REAL
models dir, then check the file's actual fate on disk - so they assert the
agreement itself, not that two copies of a predicate happen to match.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

import localm.config as config
import localm.model_manager as model_manager
from localm.cli.models import rm
from localm.model_manager import is_owned_model_path

_DELETE_TEXT = "PERMANENTLY deletes"
_KEEP_TEXT = "unregisters the name only"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An isolated data dir whose models root BOTH call sites resolve through.

    model_manager.MODELS_DIR is what the shared helper reads, so patching it
    moves the prompt and the delete gate together.

    config.MODELS_DIR is pinned to the SAME directory even though no code under
    test reads it any more. Without that pin, a prompt reading MODELS_DIR from
    config compares against the real session home, answers "not owned" for every
    path, and these tests pass VACUOUSLY. Do not drop this line.

    REGISTRY_FILE is read at call time by load/save/update_registry, so one patch
    redirects the whole real path.
    """
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(model_manager, "MODELS_DIR", models)
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(config, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _register(home, name: str, path) -> None:
    (home / "registry.json").write_text(
        json.dumps({name: {"path": str(path), "source": "local"}}), encoding="utf-8")


def _run_rm(name: str):
    """Run the real `localm rm <name>`, confirming the prompt. Returns its output."""
    result = CliRunner().invoke(rm, [name], input="y\n")
    assert result.exit_code == 0, result.output
    return result.output


# --------------------------- the shared predicate -------------------------- #

def test_is_owned_model_path_rejects_a_sibling_directory(home):
    """<root>/models-old shares the <root>/models prefix as a STRING but is
    not inside it as a PATH."""
    assert is_owned_model_path(home / "models" / "a.gguf") is True
    assert is_owned_model_path(home / "models" / "sub" / "a.gguf") is True
    assert is_owned_model_path(home / "models-old" / "x.gguf") is False
    assert is_owned_model_path(home / "elsewhere" / "y.gguf") is False
    # A plain str.startswith would say True for the sibling.
    assert str(home / "models-old" / "x.gguf").startswith(str(home / "models"))


# ------------------- prompt text agrees with the real delete ---------------- #

@pytest.mark.parametrize("subdir, announces_delete", [
    ("models", True),        # genuinely owned -> really deleted
    ("models-old", False),   # sibling: prefix-matches as a string, NOT owned
    ("elsewhere", False),    # plainly outside
])
def test_prompt_matches_what_remove_model_actually_does(home, subdir, announces_delete):
    target = home / subdir / "x.gguf"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"GGUF" + b"\0" * 64)
    _register(home, "victim", target)

    out = _run_rm("victim")

    # 1. The prompt says what the caller expects.
    assert (_DELETE_TEXT in out) is announces_delete, out
    if not announces_delete:
        assert _KEEP_TEXT in out, out

    # 2. The announcement matched the real outcome on disk.
    really_deleted = not target.exists()
    assert really_deleted is announces_delete, (
        f"prompt said {_DELETE_TEXT!r}={announces_delete} but the file was "
        f"{'deleted' if really_deleted else 'kept'}\n{out}")

    # 3. Either way the name is gone from the registry.
    assert "victim" not in config.load_registry()


def test_sibling_models_dir_file_is_never_announced_as_a_deletion(home):
    """A file in a sibling models directory is never announced as a
    deletion."""
    target = home / "models-old" / "x.gguf"
    target.parent.mkdir()
    target.write_bytes(b"GGUF")
    _register(home, "sib", target)

    out = _run_rm("sib")

    assert _DELETE_TEXT not in out, out
    assert _KEEP_TEXT in out, out
    assert target.exists(), "remove_model must not delete outside <data dir>/models"


def test_owned_file_is_announced_as_a_deletion_and_is_deleted(home):
    target = home / "models" / "real.gguf"
    target.write_bytes(b"GGUF" + b"\0" * 128)
    _register(home, "real", target)

    out = _run_rm("real")

    assert _DELETE_TEXT in out, out
    assert str(target) in out, out
    assert not target.exists()


def test_owned_but_already_missing_file_is_not_called_outside_models(home):
    """An owned path whose file is gone is a name-only drop, but saying it is
    'outside <data dir>/models' would be a false statement about the user's
    own data dir."""
    target = home / "models" / "ghost.gguf"       # never created
    _register(home, "ghost", target)

    out = _run_rm("ghost")

    assert _DELETE_TEXT not in out, out
    assert _KEEP_TEXT in out, out
    assert "outside <data dir>/models" not in out, out
    assert "already missing" in out, out


def test_alias_still_wins_over_the_owned_branch(home):
    """A second name pointing at the same owned file is reported as
    unregister-only, and the file survives."""
    target = home / "models" / "shared.gguf"
    target.write_bytes(b"GGUF")
    (home / "registry.json").write_text(json.dumps({
        "one": {"path": str(target), "source": "local"},
        "two": {"path": str(target), "source": "local"},
    }), encoding="utf-8")

    out = _run_rm("one")

    assert _DELETE_TEXT not in out, out
    assert "still registered as" in out, out
    assert target.exists()
