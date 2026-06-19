"""H1: `localm pull <local path>` must register the file in place instead of
mis-parsing a Windows drive-colon as owner/repo:file or rejecting "Unknown spec".
"""

import pytest

import localm.model_manager as mm
from localm.config import load_registry
from localm.model_manager import add_local, pull_model


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    # load_registry/save_registry read the module-level REGISTRY_FILE frozen at
    # import, so the autouse LOCALM_HOME env alone does not isolate them. Redirect
    # the config paths to a throwaway dir (mirrors conftest.cli_runner) so these
    # tests never touch the real registry or collide with each other.
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return home


def _gguf(tmp_path, name="m.gguf"):
    # Unique content per name -> unique sha256, so two fixtures never collide on
    # the content-dedup path.
    p = tmp_path / name
    p.write_bytes(b"GGUF\x00\x00\x00\x00" + name.encode())
    return p


class TestAddLocalReturnsBool:
    def test_gguf_returns_true_and_registers(self, tmp_path, isolated_home):
        assert add_local(str(_gguf(tmp_path))) is True
        assert "m" in load_registry()

    def test_missing_returns_false(self, tmp_path, isolated_home):
        assert add_local(str(tmp_path / "nope.gguf")) is False

    def test_non_model_returns_false(self, tmp_path, isolated_home):
        bad = tmp_path / "notes.txt"
        bad.write_text("hello")
        assert add_local(str(bad)) is False


class TestPullByLocalPath:
    def test_pull_registers_local_gguf(self, tmp_path, isolated_home):
        # FAILS pre-fix: pull_model rejected an absolute path ("Unknown spec" /
        # "Unsafe model filename").
        p = _gguf(tmp_path, "mymodel.gguf")
        assert pull_model(str(p)) is True
        assert "mymodel" in load_registry()

    def test_pull_non_model_path_returns_false(self, tmp_path, isolated_home):
        bad = tmp_path / "data.txt"
        bad.write_text("x")
        assert pull_model(str(bad)) is False

    def test_pull_hf_spec_not_treated_as_local(self, monkeypatch):
        # a bare owner/repo must still route to HF, never to add_local.
        called = {"add_local": False, "hf": False}
        monkeypatch.setattr(mm, "add_local",
                            lambda *a, **k: called.__setitem__("add_local", True) or True)
        monkeypatch.setattr(mm, "_pull_hf_snapshot",
                            lambda *a, **k: called.__setitem__("hf", True) or True)
        pull_model("owner/repo")
        assert called["hf"] is True
        assert called["add_local"] is False
