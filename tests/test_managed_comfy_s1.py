# SPDX-License-Identifier: AGPL-3.0-or-later
"""STAGE S1 (scaffolding) for the localm-managed ComfyUI feature.

Covers the OFF-by-default no-op contract, the coexistence resolver, the
extra_model_paths.yaml generator, and the `localm comfy status/remove` CLI -
all WITHOUT provisioning anything (S1 is scaffolding; S2/S3 provision). The
cardinal rule tested here: with the flag off and nothing installed, ComfyUI
targeting is byte-identical to an install that has never heard of this feature,
and no managed directory is ever created.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import localm.config as cfg
from localm.media import comfy_client
from localm.media import managed_comfy as mc


# --------------------------------------------------------------------------- #
#  Isolation: a throwaway LOCALM_HOME wired through both the lazy home_dir()   #
#  AND the import-frozen config.CONFIG_FILE, so load_config/save_config and    #
#  managed_comfy path resolution agree on the same tmp dir. FLUX_API_URL is    #
#  cleared so the off baseline is deterministic.                               #
# --------------------------------------------------------------------------- #
@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    h.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    return h


def _install_managed(home_dir: Path) -> mc.ManagedComfyPaths:
    """Create the minimal on-disk layout that makes is_managed_comfy_installed()
    true, using the module's OWN path accessors so the test is platform-agnostic
    (venv interpreter path differs on Windows vs POSIX). Includes the completion
    marker: main.py + venv alone means "install still running", not "installed"
    - see is_managed_comfy_installed()'s docstring."""
    from localm.media.managed_comfy_provision import MARKER_FILENAME
    paths = mc.managed_comfy_paths()
    paths.main_py.parent.mkdir(parents=True, exist_ok=True)
    paths.main_py.write_text("# stand-in for ComfyUI main.py\n", encoding="utf-8")
    paths.venv_python.parent.mkdir(parents=True, exist_ok=True)
    paths.venv_python.write_text("", encoding="utf-8")
    (paths.root / MARKER_FILENAME).write_text("{}", encoding="utf-8")
    assert mc.is_managed_comfy_installed()
    return paths


# --------------------------------------------------------------------------- #
#  OFF by default: no-op, byte-identical to today, no dirs created.            #
# --------------------------------------------------------------------------- #

def test_defaults_are_off(home):
    c = cfg.load_config()
    assert c["comfy_target"] == "own"          # prefer own when one exists


def test_off_by_default_is_not_installed(home):
    assert mc.is_managed_comfy_installed() is False
    assert mc.managed_comfy_active() is False


# --------------------------------------------------------------------------- #
#  A still-installing instance must NOT read as installed.                    #
# --------------------------------------------------------------------------- #

def test_main_py_and_venv_alone_are_not_installed(home):
    """The on-disk state right after `git clone` + `python -m venv` (steps 1-2
    of a 7-8 step pipeline) - main.py and the venv interpreter exist, but
    torch/requirements/custom-nodes/the completion marker have not run yet -
    must read as NOT installed. A real install takes minutes past this point,
    during which the Settings pill, `localm comfy status`, and the actual
    Generate-button routing (managed_comfy_active() calls this) would otherwise
    claim the instance is ready and route to it."""
    paths = mc.managed_comfy_paths()
    paths.main_py.parent.mkdir(parents=True, exist_ok=True)
    paths.main_py.write_text("# stand-in for ComfyUI main.py\n", encoding="utf-8")
    paths.venv_python.parent.mkdir(parents=True, exist_ok=True)
    paths.venv_python.write_text("", encoding="utf-8")

    assert mc.is_managed_comfy_installed() is False
    assert mc.managed_comfy_active() is False


def test_installed_flips_true_only_once_the_marker_is_written(home):
    """The other half of the same regression: once the marker DOES land (the
    real pipeline's step 7/8, right after everything else has succeeded),
    installed correctly flips true - the fix does not just make the check
    permanently stricter, it makes it accurate."""
    from localm.media.managed_comfy_provision import MARKER_FILENAME
    paths = mc.managed_comfy_paths()
    paths.main_py.parent.mkdir(parents=True, exist_ok=True)
    paths.main_py.write_text("# stand-in for ComfyUI main.py\n", encoding="utf-8")
    paths.venv_python.parent.mkdir(parents=True, exist_ok=True)
    paths.venv_python.write_text("", encoding="utf-8")
    assert mc.is_managed_comfy_installed() is False

    (paths.root / MARKER_FILENAME).write_text("{}", encoding="utf-8")

    assert mc.is_managed_comfy_installed() is True


def test_off_resolver_returns_users_comfy_exactly(home):
    """With the flag off and nothing installed, the resolver must return the
    user's ComfyUI - identical to default_api_url() today - and NOT the managed
    target."""
    target = mc.resolve_comfy_target()
    assert target.managed is False
    assert target.api_url == comfy_client.default_api_url()
    assert target.api_url == "http://127.0.0.1:8188"     # today's loopback default
    assert target.workdir == cfg.load_config().get("comfy_workdir")  # None here


def test_off_honours_user_config_url(home):
    """The user's comfy_api_url still flows through unchanged when off."""
    cfg.save_config({**cfg.load_config(), "comfy_api_url": "http://127.0.0.1:9191"})
    assert comfy_client.default_api_url() == "http://127.0.0.1:9191"
    assert mc.resolve_comfy_target().api_url == "http://127.0.0.1:9191"
    assert mc.resolve_comfy_target().managed is False


def test_off_does_not_create_managed_dirs(home):
    """A normal target resolution must not materialize any managed directory."""
    _ = mc.resolve_comfy_target()
    _ = comfy_client.default_api_url()
    paths = mc.managed_comfy_paths()
    assert not paths.root.exists()
    assert not paths.models_dir.exists()


# --------------------------------------------------------------------------- #
#  Coexistence routing.                                                       #
# --------------------------------------------------------------------------- #

def test_installed_and_own_targets_managed(home):
    paths = _install_managed(home)
    cfg.save_config({**cfg.load_config(), "comfy_target": "own"})
    assert mc.managed_comfy_active() is True
    t = mc.resolve_comfy_target()
    assert t.managed is True
    assert t.api_url == mc.managed_comfy_api_url()
    assert t.api_url == mc.MANAGED_COMFY_API_URL
    assert t.workdir == str(paths.root)
    # Every caller of default_api_url() hits the managed instance without
    # touching the launch/spawn path.
    assert comfy_client.default_api_url() == mc.MANAGED_COMFY_API_URL


def test_installed_but_target_user_targets_user(home):
    _install_managed(home)
    cfg.save_config({**cfg.load_config(), "comfy_target": "user"})
    assert mc.managed_comfy_active() is False
    t = mc.resolve_comfy_target()
    assert t.managed is False
    assert t.api_url == "http://127.0.0.1:8188"
    assert comfy_client.default_api_url() == "http://127.0.0.1:8188"


def test_managed_port_differs_from_user_default(home):
    """Coexistence requires the managed instance on its OWN port, never the
    user's ComfyUI default (8188), or both could not run at once."""
    assert mc.managed_comfy_api_url() != "http://127.0.0.1:8188"


def test_flux_env_still_overrides_when_managed_off(home):
    """FLUX_API_URL stays the top override when managed is inactive (today's
    contract preserved)."""
    import os
    os.environ["FLUX_API_URL"] = "http://127.0.0.1:7777"
    try:
        assert comfy_client.default_api_url() == "http://127.0.0.1:7777"
    finally:
        del os.environ["FLUX_API_URL"]


# --------------------------------------------------------------------------- #
#  extra_model_paths.yaml generator.                                          #
# --------------------------------------------------------------------------- #

def test_extra_model_paths_includes_both_dirs(home, tmp_path):
    """With comfy_workdir known, the generated yaml points at BOTH the user's
    existing ComfyUI models dir and the localm-managed models dir."""
    import yaml   # guaranteed present transitively via huggingface-hub (core dep)

    user_comfy = tmp_path / "user-comfy"
    (user_comfy / "models").mkdir(parents=True)
    cfg.save_config({**cfg.load_config(), "comfy_workdir": str(user_comfy)})

    paths = mc.managed_comfy_paths()
    paths.root.mkdir(parents=True, exist_ok=True)   # generator writes into the install dir
    out = mc.write_extra_model_paths()
    assert out == paths.extra_model_paths
    assert out.is_file()

    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    base_paths = {entry["base_path"] for entry in data.values()
                  if isinstance(entry, dict) and "base_path" in entry}
    assert str(paths.models_dir) in base_paths                 # managed models dir
    assert str(user_comfy / "models") in base_paths            # user's models dir


def test_extra_model_paths_managed_only_without_workdir(home):
    """No comfy_workdir -> only the managed models dir is referenced (fresh path)."""
    import yaml

    paths = mc.managed_comfy_paths()
    paths.root.mkdir(parents=True, exist_ok=True)
    out = mc.write_extra_model_paths()
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    base_paths = {entry["base_path"] for entry in data.values()
                  if isinstance(entry, dict) and "base_path" in entry}
    assert str(paths.models_dir) in base_paths
    # the only base_path is the managed one (no user entry)
    assert base_paths == {str(paths.models_dir)}


# --------------------------------------------------------------------------- #
#  CLI: localm comfy status / remove                                          #
# --------------------------------------------------------------------------- #

def test_cli_status_not_set_up(cli_runner):
    from localm.cli import main
    res = cli_runner.invoke(main, ["comfy", "status"])
    assert res.exit_code == 0, res.output
    low = res.output.lower()
    assert "not set up" in low or "not installed" in low
    assert "target" in low            # reports the current target


def test_cli_status_reports_managed_when_installed(cli_runner, monkeypatch):
    import localm.config as cfg2
    from localm.cli import main
    _install_managed(cfg2.home_dir())
    cfg2.save_config({**cfg2.load_config(), "comfy_target": "own"})
    res = cli_runner.invoke(main, ["comfy", "status"])
    assert res.exit_code == 0, res.output
    assert mc.MANAGED_COMFY_API_URL in res.output


def test_cli_remove_deletes_only_managed_dir(cli_runner):
    """remove deletes <HOME>/comfyui and NEVER touches the user's comfy_workdir."""
    import localm.config as cfg2
    from localm.cli import main

    # A user ComfyUI folder that must be left completely alone.
    user_comfy = cfg2.home_dir().parent / "user-comfy"
    (user_comfy / "models").mkdir(parents=True)
    (user_comfy / "main.py").write_text("# user's own ComfyUI\n", encoding="utf-8")
    cfg2.save_config({**cfg2.load_config(), "comfy_workdir": str(user_comfy)})

    paths = _install_managed(cfg2.home_dir())
    assert paths.root.exists()

    res = cli_runner.invoke(main, ["comfy", "remove", "--yes"])
    assert res.exit_code == 0, res.output
    assert not paths.root.exists()                 # managed install gone
    assert user_comfy.exists()                     # user's ComfyUI untouched
    assert (user_comfy / "main.py").is_file()


def test_cli_remove_nothing_installed(cli_runner):
    from localm.cli import main
    res = cli_runner.invoke(main, ["comfy", "remove", "--yes"])
    assert res.exit_code == 0, res.output
    assert "nothing" in res.output.lower() or "not" in res.output.lower()


def test_cli_setup_is_honest_on_failure(cli_runner, monkeypatch):
    """`localm comfy setup` is a REAL feature (S2 copy + S3 fresh), not a facade.
    It is also HONEST about failure: when provisioning fails it exits non-zero and
    leaves nothing installed. The heavy provisioning is mocked so the test stays
    inert (no multi-GB clone / torch download)."""
    from localm.cli import main
    from localm.media import managed_comfy_fresh as fresh
    from localm.media import managed_comfy_provision as prov
    # No user ComfyUI (throwaway home) -> the fresh path; mock it to a clean failure.
    monkeypatch.setattr(fresh, "provision_fresh",
                        lambda **kw: prov.ProvisionResult(
                            ok=False, status="error", message="clone failed (mocked)"))
    res = cli_runner.invoke(main, ["comfy", "setup"])
    assert res.exit_code != 0
    assert "failed" in res.output.lower()
    assert not mc.is_managed_comfy_installed()
