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


def test_relocate_target_shared_validator(env):
    """relocate_target is the validator relocate_model AND the GUI's POST
    /api/models/relocate route both call - so it needs its own direct coverage,
    not just indirect coverage through relocate_model's bool return."""
    import localm.model_manager as mm
    tmp, _ = env
    good = _gguf(tmp / "ok.gguf")
    p, reason = mm.relocate_target(str(good))
    assert reason is None
    assert p == good.resolve()

    p, reason = mm.relocate_target(str(tmp / "nope.gguf"))
    assert p is None
    assert "does not exist" in reason

    bad = tmp / "not-really.gguf"
    bad.write_bytes(b"NOPE not a gguf")
    p, reason = mm.relocate_target(str(bad))
    assert p is None
    assert "Not a GGUF" in reason

    not_hf_dir = tmp / "empty-dir"
    not_hf_dir.mkdir()
    p, reason = mm.relocate_target(str(not_hf_dir))
    assert p is None
    assert "HuggingFace" in reason


def test_relocate_target_truncated_gguf_gets_its_own_reason(env):
    """A truncated GGUF has real magic bytes - it IS a GGUF file, just an
    incomplete one. Telling the user "Not a GGUF model file" for it is false
    and sends them looking for the wrong problem (their file is fine, it is
    just early). The declared-size gap in _has_gguf_magic must not collapse
    into the same generic message the wrong-magic case gets."""
    import struct

    import localm.model_manager as mm
    from localm.model_manager.gguf import _GGUF_MIN_BYTES

    tmp, _ = env

    def s(text):
        raw = text.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    # weight.0 declares 4096 bytes, so weight.1 (never reached) starts at
    # offset 4096 - the file below has only a handful of bytes after the
    # header, nowhere near even weight.0's own declared size.
    header = b"".join([
        b"GGUF", struct.pack("<I", 3), struct.pack("<QQ", 2, 0),
        s("weight.0"), struct.pack("<I", 1), struct.pack("<Q", 1),
        struct.pack("<I", 0), struct.pack("<Q", 0),
        s("weight.1"), struct.pack("<I", 1), struct.pack("<Q", 1),
        struct.pack("<I", 0), struct.pack("<Q", 4096),
    ])
    truncated = tmp / "truncated.gguf"
    truncated.parent.mkdir(parents=True, exist_ok=True)
    truncated.write_bytes(header.ljust(_GGUF_MIN_BYTES + 4, b"\0"))

    p, reason = mm.relocate_target(str(truncated))

    assert p is None
    assert "Not a GGUF" not in reason
    assert "has not finished copying" in reason
    assert str(truncated) in reason

    # Regression guard: a genuinely foreign file still gets the original,
    # generic message - only the truncated-GGUF case gets the new one.
    bad = tmp / "not-really-2.gguf"
    bad.write_bytes(b"NOPE not a gguf")
    p2, reason2 = mm.relocate_target(str(bad))
    assert p2 is None
    assert "Not a GGUF" in reason2


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
