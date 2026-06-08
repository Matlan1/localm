"""
Tests for localm.plugins.coder.project_config
"""

import pytest
from pathlib import Path

from localm.plugins.coder.project_config import (
    find_project_config,
    load_project_config,
)


# ---------------------------------------------------------------------------
#  find_project_config
# ---------------------------------------------------------------------------

class TestFindProjectConfig:
    def test_finds_config_in_cwd(self, tmp_path):
        cfg = tmp_path / ".localcoder" / "config.toml"
        cfg.parent.mkdir()
        cfg.write_text("[placeholder]\n")
        assert find_project_config(tmp_path) == cfg

    def test_finds_config_in_parent(self, tmp_path):
        subdir = tmp_path / "src" / "sub"
        subdir.mkdir(parents=True)
        cfg = tmp_path / ".localcoder" / "config.toml"
        cfg.parent.mkdir()
        cfg.write_text("[placeholder]\n")
        assert find_project_config(subdir) == cfg

    def test_returns_none_when_absent(self, tmp_path):
        assert find_project_config(tmp_path) is None

    def test_stops_at_filesystem_root(self, tmp_path):
        # Should not raise even when walking all the way up
        result = find_project_config(tmp_path)
        assert result is None or isinstance(result, Path)


# ---------------------------------------------------------------------------
#  load_project_config
# ---------------------------------------------------------------------------

class TestLoadProjectConfig:
    def _write_cfg(self, tmp_path: Path, content: str) -> Path:
        cfg_dir = tmp_path / ".localcoder"
        cfg_dir.mkdir(exist_ok=True)
        cfg = cfg_dir / "config.toml"
        cfg.write_text(content, encoding="utf-8")
        return cfg

    def test_empty_when_no_file(self, tmp_path):
        assert load_project_config(tmp_path) == {}

    def test_loads_model(self, tmp_path):
        self._write_cfg(tmp_path, 'model = "gemma4-4b"\n')
        cfg = load_project_config(tmp_path)
        assert cfg["model"] == "gemma4-4b"

    def test_loads_max_turns(self, tmp_path):
        self._write_cfg(tmp_path, "max_turns = 25\n")
        cfg = load_project_config(tmp_path)
        assert cfg["max_turns"] == 25

    def test_loads_auto_approve(self, tmp_path):
        self._write_cfg(tmp_path, "auto_approve = true\n")
        cfg = load_project_config(tmp_path)
        assert cfg["auto_approve"] is True

    def test_loads_max_tokens(self, tmp_path):
        self._write_cfg(tmp_path, "max_tokens = 4096\n")
        cfg = load_project_config(tmp_path)
        assert cfg["max_tokens"] == 4096

    def test_loads_temperature(self, tmp_path):
        self._write_cfg(tmp_path, "temperature = 0.5\n")
        cfg = load_project_config(tmp_path)
        assert cfg["temperature"] == pytest.approx(0.5)

    def test_loads_memory_file(self, tmp_path):
        self._write_cfg(tmp_path, 'memory_file = ".localcoder/notes.md"\n')
        cfg = load_project_config(tmp_path)
        assert cfg["memory_file"] == ".localcoder/notes.md"

    def test_ignores_unknown_keys(self, tmp_path):
        self._write_cfg(tmp_path, 'unknown_key = "hello"\nmodel = "phi4"\n')
        cfg = load_project_config(tmp_path)
        assert "unknown_key" not in cfg
        assert cfg["model"] == "phi4"

    def test_loads_multiple_keys(self, tmp_path):
        self._write_cfg(tmp_path, (
            'model = "deepseek-r1"\n'
            "max_turns = 15\n"
            "auto_approve = false\n"
            "max_tokens = 1024\n"
            "temperature = 0.7\n"
        ))
        cfg = load_project_config(tmp_path)
        assert cfg["model"] == "deepseek-r1"
        assert cfg["max_turns"] == 15
        assert cfg["auto_approve"] is False
        assert cfg["max_tokens"] == 1024
        assert cfg["temperature"] == pytest.approx(0.7)

    def test_returns_empty_on_invalid_toml(self, tmp_path):
        cfg_dir = tmp_path / ".localcoder"
        cfg_dir.mkdir()
        (cfg_dir / "config.toml").write_text("this is NOT [valid toml [\n")
        result = load_project_config(tmp_path)
        assert result == {}

    def test_finds_config_in_parent_dir(self, tmp_path):
        subdir = tmp_path / "deep" / "nested"
        subdir.mkdir(parents=True)
        self._write_cfg(tmp_path, 'model = "phi4-mini"\n')
        cfg = load_project_config(subdir)
        assert cfg["model"] == "phi4-mini"
