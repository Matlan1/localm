# SPDX-License-Identifier: AGPL-3.0-or-later
"""The autouse LOCALM_HOME isolation must reach config's FROZEN constants,
not just the env var.

``config.HOME_DIR``/``MODELS_DIR``/``REGISTRY_FILE``/``CONFIG_FILE`` are
plain module attributes computed once at ``config`` import time
(``HOME_DIR = _detect_home()``), before any test's ``LOCALM_HOME`` override
ever runs. ``config.home_dir()`` is the dynamic counterpart that re-resolves
on every call and DOES track the env var. A write path that calls
``ensure_dirs()`` (which creates ``HOME_DIR``) and then computes its actual
target via ``home_dir()`` (e.g. ``auth.create_key`` /
``model_source_credentials.save_credentials``) silently creates the WRONG
directory when only the env var is isolated - the failure surfaces as
``FileNotFoundError`` on a ``*.lock`` file whose parent was never made,
since a lock file's own creation (``os.O_CREAT | os.O_EXCL``) does not
create parent directories.
"""

from __future__ import annotations

from localm import config as cfg


def test_frozen_home_constants_match_the_dynamic_resolver(tmp_path):
    """The property the autouse fixture must hold for every test: the frozen
    constants agree with home_dir()'s current resolution, not a stale
    import-time snapshot."""
    assert cfg.HOME_DIR == cfg.home_dir()
    assert cfg.MODELS_DIR == cfg.HOME_DIR / "models"
    assert cfg.REGISTRY_FILE == cfg.HOME_DIR / "registry.json"
    assert cfg.CONFIG_FILE == cfg.HOME_DIR / "config.json"
    # And it is genuinely THIS test's tmp_path, not merely internally
    # consistent with itself.
    assert cfg.HOME_DIR == tmp_path / ".localm"


def test_ensure_dirs_creates_the_directory_a_dynamic_caller_actually_uses(tmp_path):
    """The concrete failure this fixes: a write path that calls ensure_dirs()
    then locks a sibling of a home_dir()-derived path must find that
    directory already there - not a stale one from import time."""
    assert not cfg.HOME_DIR.exists()
    cfg.ensure_dirs()
    target = cfg.home_dir() / "some_store.json"
    lockpath = target.with_name(target.name + ".lock")
    import os
    fd = os.open(str(lockpath), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    lockpath.unlink()
