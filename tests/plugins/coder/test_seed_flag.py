# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm coder --seed N`: the seed reaches the request body.

Whether a fixed seed actually reproduces output is a property of the runtime,
not of this wiring, and is not asserted here. What these tests pin is that the
flag survives the CLI, the project config, the Agent's gen kwargs and the
backend body build, and that the one provider without a seed parameter says so
instead of dropping it silently.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from localm.plugins.coder.backends.http import HTTPBackend


class _Backend:
    model_id = "test-model"
    native_tools = False
    supports_grammar = False
    last_usage: dict = {}
    last_reasoning = ""

    def chat_stream(self, messages, on_reasoning=None, **kwargs):
        yield "done"

    def set_tools(self, tool_defs):
        pass

    def context_capacity(self):
        return None


def _make_agent(tmp_path: Path, **kwargs):
    from localm.plugins.coder.agent import Agent
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        return Agent(backend=_Backend(), cwd=tmp_path, **kwargs)


class TestBackendBody:
    def test_seed_lands_in_the_openai_body(self):
        backend = HTTPBackend("http://x/v1", "m", verify=False)
        body = backend._body([{"role": "user", "content": "hi"}],
                             stream=False, seed=1234)
        assert body["seed"] == 1234

    def test_no_seed_means_no_field(self):
        backend = HTTPBackend("http://x/v1", "m", verify=False)
        body = backend._body([{"role": "user", "content": "hi"}], stream=False)
        assert "seed" not in body


class TestAgentGenKwargs:
    def test_seed_is_forwarded_to_every_llm_call(self, tmp_path):
        agent = _make_agent(tmp_path, seed=99)
        assert agent._llm_kwargs()["seed"] == 99

    def test_absent_by_default(self, tmp_path):
        assert "seed" not in _make_agent(tmp_path)._llm_kwargs()


class TestCliResolution:
    def _resolve(self, tmp_path, seed=None, provider=None):
        from localm.plugins.coder.cli._main import _resolve_session_config
        # mode is log, not privacy: the privacy branch defuses this process's
        # readline history as a side effect.
        return _resolve_session_config(
            tmp_path, "m", 40, None, None, seed, True, False, "log",
            False, provider)[5]

    def test_flag_reaches_gen_kwargs(self, tmp_path):
        assert self._resolve(tmp_path, seed=7)["seed"] == 7

    def test_absent_without_the_flag(self, tmp_path):
        assert "seed" not in self._resolve(tmp_path)

    def test_project_config_supplies_a_seed(self, tmp_path):
        (tmp_path / ".localcoder").mkdir()
        (tmp_path / ".localcoder" / "config.toml").write_text("seed = 4242\n")
        assert self._resolve(tmp_path)["seed"] == 4242

    def test_flag_overrides_project_config(self, tmp_path):
        (tmp_path / ".localcoder").mkdir()
        (tmp_path / ".localcoder" / "config.toml").write_text("seed = 4242\n")
        assert self._resolve(tmp_path, seed=7)["seed"] == 7

    def test_anthropic_drops_the_seed_and_says_so(self, tmp_path):
        """The Messages API has no seed parameter. Silently passing it on would
        make the run look pinned when nothing pinned it."""
        with patch("localm.plugins.coder.cli._main.print_warning") as warn:
            gen_kw = self._resolve(tmp_path, seed=7, provider="anthropic")
        assert "seed" not in gen_kw
        said = " ".join(str(c) for c in warn.call_args_list)
        assert "not reproducible" in said
