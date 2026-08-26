# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for localm.plugins.coder.project_config
"""

import pytest
from pathlib import Path

from localm.plugins.coder.project_config import (
    ProjectConfigUnreadable,
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

    @pytest.mark.parametrize("toml_line,key,expected", [
        ('model = "gemma4-4b"\n', "model", "gemma4-4b"),
        ("max_turns = 25\n", "max_turns", 25),
        ("auto_approve = true\n", "auto_approve", True),
        ("max_tokens = 4096\n", "max_tokens", 4096),
        ("temperature = 0.5\n", "temperature", pytest.approx(0.5)),
        ('memory_file = ".localcoder/notes.md"\n', "memory_file", ".localcoder/notes.md"),
    ])
    def test_loads_single_key(self, tmp_path, toml_line, key, expected):
        self._write_cfg(tmp_path, toml_line)
        cfg = load_project_config(tmp_path)
        assert cfg[key] == expected

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

    def test_raises_on_invalid_toml_rather_than_reading_as_absent(self, tmp_path):
        """A file that EXISTS but does not parse must not answer ``{}``.

        ``{}`` is byte-identical to "no project config here", and the two keys
        it would silently drop are SAFETY settings: ``always_confirm`` (prompt
        before a shell command even under --yes) empties in cli/_main.py, and
        ``mode = "privacy"`` is dropped there and in audit.py, so a session the
        user marked private would fall through to the global coder_mode and
        write a transcript.
        """
        cfg_dir = tmp_path / ".localcoder"
        cfg_dir.mkdir()
        (cfg_dir / "config.toml").write_text("this is NOT [valid toml [\n")
        with pytest.raises(ProjectConfigUnreadable) as ei:
            load_project_config(tmp_path)
        # The message names the file.
        assert "config.toml" in str(ei.value)

    def test_absent_file_is_still_an_empty_config(self, tmp_path):
        """The control. "No project config" must keep meaning ``{}`` and must
        not raise, or the refusal above would be unfalsifiable: a loader that
        raised on everything would satisfy the test above."""
        assert load_project_config(tmp_path) == {}

    def test_finds_config_in_parent_dir(self, tmp_path):
        subdir = tmp_path / "deep" / "nested"
        subdir.mkdir(parents=True)
        self._write_cfg(tmp_path, 'model = "phi4-mini"\n')
        cfg = load_project_config(subdir)
        assert cfg["model"] == "phi4-mini"


class TestCliRefusesAnUnreadableProjectConfig:
    """The coder CLI must REFUSE to start rather than run with the project
    file's settings silently dropped.

    `always_confirm` is what keeps shell tools prompting under --yes, and
    `mode = "privacy"` is what keeps a transcript off disk, so starting anyway
    would run the session without protections the user configured.
    """

    def _corrupt(self, tmp_path):
        cfg_dir = tmp_path / ".localcoder"
        cfg_dir.mkdir()
        (cfg_dir / "config.toml").write_text(
            'always_confirm = ["run_shell"]\nmode = "privacy"\n'
            'this is NOT [valid toml [\n', encoding="utf-8")
        return cfg_dir / "config.toml"

    def test_cli_refuses_and_says_why(self, tmp_path, monkeypatch):
        from click.testing import CliRunner

        import localm.plugins.coder.cli as ccli

        # The CLI gates on the coder being installed first, so bypass that gate
        # to reach the config refusal.
        monkeypatch.setattr("localm.plugins.engine.PluginManager.is_active",
                            lambda self, name: True)
        self._corrupt(tmp_path)
        res = CliRunner().invoke(
            ccli.main, ["--cwd", str(tmp_path), "--model", "m", "hi"])

        # Assert the message first: it is the only assertion that separates a
        # config refusal from a later, unrelated non-zero exit.
        assert "could not be read" in res.output.lower(), (
            "the coder did not refuse: it started with always_confirm and "
            f"mode silently dropped. output was: {res.output!r}")
        # The message names the file.
        assert "config.toml" in res.output
        assert res.exit_code != 0

    def test_cli_does_not_refuse_without_a_project_config(self, tmp_path,
                                                          monkeypatch):
        """The control: an absent project config must NOT trip the refusal, or
        the test above would pass on a CLI that refused to start at all."""
        from click.testing import CliRunner

        import localm.plugins.coder.cli as ccli

        monkeypatch.setattr("localm.plugins.engine.PluginManager.is_active",
                            lambda self, name: True)
        res = CliRunner().invoke(
            ccli.main, ["--cwd", str(tmp_path), "--model", "m", "hi"])
        assert "could not be read" not in res.output.lower()
