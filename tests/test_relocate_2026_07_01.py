# SPDX-License-Identifier: AGPL-3.0-or-later
"""REC-EXTPATH-RELOCATE - mark external models + re-point a moved one."""

from pathlib import Path

import pytest


@pytest.fixture
def env(tmp_path, monkeypatch):
    import localm.config as cfg
    import localm.model_manager as mm
    home = tmp_path / ".localm"
    models = home / "models"
    models.mkdir(parents=True)
    reg_file = home / "registry.json"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", models)
    monkeypatch.setattr(cfg, "REGISTRY_FILE", reg_file)
    monkeypatch.setattr(mm, "MODELS_DIR", models, raising=False)
    monkeypatch.setattr(mm, "REGISTRY_FILE", reg_file, raising=False)
    return tmp_path, models


def _gguf(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"GGUF" + b"\x00" * 2048)      # valid GGUF magic + past the size floor
    return p


def test_is_external_path(env):
    import localm.model_manager as mm
    _, models = env
    assert mm.is_external_path(str(models / "downloaded.gguf")) is False
    assert mm.is_external_path(str(Path("/somewhere/else/ext.gguf"))) is True


def test_relocate_repoints_a_moved_external_model(env):
    import localm.model_manager as mm
    tmp, _ = env
    old = _gguf(tmp / "ext" / "model.gguf")       # external location
    mm.save_registry({"mymodel": {"path": str(old), "source": "local", "missing": True}})
    assert mm.model_is_external("mymodel") is True

    new = _gguf(tmp / "moved" / "model.gguf")      # the user moved the file
    assert mm.relocate_model("mymodel", str(new)) is True

    reg = mm.load_registry()
    assert Path(reg["mymodel"]["path"]) == new.resolve()
    assert "missing" not in reg["mymodel"], "the missing flag must be cleared"


def test_relocate_rejects_bad_inputs(env):
    import localm.model_manager as mm
    tmp, _ = env
    mm.save_registry({"m": {"path": "/x/old.gguf", "source": "local"}})
    assert mm.relocate_model("m", str(tmp / "nope.gguf")) is False        # does not exist
    bad = tmp / "not-really.gguf"
    bad.write_bytes(b"NOPE not a gguf")                                    # wrong magic
    assert mm.relocate_model("m", str(bad)) is False
    good = _gguf(tmp / "ok.gguf")
    assert mm.relocate_model("unknown-model", str(good)) is False         # no such model


def test_relocate_cli_command(env, monkeypatch):
    from click.testing import CliRunner
    from localm.cli import models as models_cli
    import localm.model_manager as mm
    tmp, _ = env
    mm.save_registry({"m": {"path": "/gone/old.gguf", "source": "local", "missing": True}})
    new = _gguf(tmp / "here" / "m.gguf")
    r = CliRunner().invoke(models_cli.relocate_cmd, ["m", str(new)])
    assert r.exit_code == 0, r.output
    assert Path(mm.load_registry()["m"]["path"]) == new.resolve()
