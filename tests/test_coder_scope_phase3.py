# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Three properties of the coder plugin:

- Scope globbing is path-aware. ``*`` must not cross ``/`` (so ``src/*.py``
  rejects ``src/a/b/c.py``), ``**`` may cross ``/``, and an absolute path that
  lives inside cwd and matches the scope must be allowed.

- ``--scope`` enforcement covers grep, search_files, search_replace, read_env,
  edit_notebook_cell and generate_image. ``run_shell`` stays unscoped. The
  check keys on the ``path`` arg and, for tools whose primary target is
  ``glob`` or ``output_path``, on that arg too.

- ``make_anthropic_backend`` speaks the Anthropic Messages API: it posts to
  ``/v1/messages`` with an ``x-api-key`` header and an ``anthropic-version``
  header, NOT a Bearer ``Authorization`` header against ``/chat/completions``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.plugins.coder.agent import Agent, _SCOPED_TOOLS
from localm.plugins.coder.backends.http import (
    make_anthropic_backend,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _make_agent(tmp_path: Path, scope: str | None = None) -> Agent:
    backend = MagicMock()
    backend.model_id = "test-model"
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        agent = Agent(backend=backend, cwd=tmp_path, scope=scope)
    return agent


def _call(name: str, **args):
    c = MagicMock()
    c.name = name
    c.args = args
    return c


def _rejected(result) -> bool:
    return (not result.ok) and ("outside the active scope" in result.output)


# ---------------------------------------------------------------------------
#  Path-aware glob boundaries
# ---------------------------------------------------------------------------

class TestScopeGlobBoundary:
    def test_single_star_does_not_cross_slash(self, tmp_path):
        """src/*.py must NOT match a nested file src/a/b/c.py (adversarial)."""
        agent = _make_agent(tmp_path, scope="src/*.py")
        nested = tmp_path / "src" / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "c.py").write_text("pass\n")

        result = agent._execute_tool(
            _call("read_file", path="src/a/b/c.py"), interactive=False
        )
        assert _rejected(result), (
            "src/*.py wrongly allowed a nested path - '*' crossed a '/'"
        )

    def test_single_star_matches_direct_child(self, tmp_path):
        """src/*.py still matches a direct child src/main.py."""
        agent = _make_agent(tmp_path, scope="src/*.py")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("pass\n")

        result = agent._execute_tool(
            _call("read_file", path="src/main.py"), interactive=False
        )
        assert "outside the active scope" not in result.output

    def test_double_star_crosses_slash(self, tmp_path):
        """src/**/*.py matches a nested file."""
        agent = _make_agent(tmp_path, scope="src/**/*.py")
        nested = tmp_path / "src" / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "c.py").write_text("pass\n")

        result = agent._execute_tool(
            _call("read_file", path="src/a/b/c.py"), interactive=False
        )
        assert "outside the active scope" not in result.output

    def test_absolute_path_inside_cwd_matching_scope_is_allowed(self, tmp_path):
        """An absolute path inside cwd that matches the scope must pass."""
        agent = _make_agent(tmp_path, scope="src/*.py")
        (tmp_path / "src").mkdir()
        f = tmp_path / "src" / "main.py"
        f.write_text("pass\n")

        # Pass the resolved absolute path as the tool arg.
        result = agent._execute_tool(
            _call("read_file", path=str(f.resolve())), interactive=False
        )
        assert "outside the active scope" not in result.output, (
            "an in-cwd absolute path matching the scope was wrongly rejected"
        )

    def test_absolute_path_outside_cwd_is_rejected(self, tmp_path):
        """An absolute path that escapes cwd is rejected regardless of glob."""
        agent = _make_agent(tmp_path, scope="**/*.py")
        outside = tmp_path.parent / "elsewhere.py"

        result = agent._execute_tool(
            _call("read_file", path=str(outside)), interactive=False
        )
        assert _rejected(result)


# ---------------------------------------------------------------------------
#  Scoped-tool coverage
# ---------------------------------------------------------------------------

class TestScopedToolCoverage:
    @pytest.mark.parametrize("tool", [
        "grep", "search_files", "search_replace",
        "read_env", "edit_notebook_cell", "generate_image",
    ])
    def test_new_tool_in_scoped_set(self, tool):
        assert tool in _SCOPED_TOOLS

    def test_run_shell_stays_unscoped(self):
        assert "run_shell" not in _SCOPED_TOOLS

    @pytest.mark.parametrize(
        "scope,tool_name,kwargs",
        [
            pytest.param(
                "src/*.py", "grep",
                {"pattern": "TODO", "path": "docs"},
                id="grep_path",
            ),
            # grep keys on the glob arg too, not just path
            pytest.param(
                "src/*.py", "grep",
                {"pattern": "TODO", "glob": "docs/*.md"},
                id="grep_glob",
            ),
            # search_replace's primary scope target is the glob arg
            pytest.param(
                "src/*.py", "search_replace",
                {"pattern": "a", "replacement": "b", "glob": "tests/**/*.py"},
                id="search_replace_glob",
            ),
            # generate_image's primary scope target is output_path
            pytest.param(
                "art/*.png", "generate_image",
                {"prompt": "a cat", "output_path": "out/x.png"},
                id="generate_image_output_path",
            ),
            # img2img keys on input_image as well
            pytest.param(
                "art/*.png", "generate_image",
                {"prompt": "a cat", "output_path": "art/ok.png",
                 "input_image": "secret/private.png"},
                id="generate_image_input_image",
            ),
            pytest.param(
                "src/*.py", "edit_notebook_cell",
                {"path": "nb/x.ipynb", "cell_index": 0, "source": "x=1"},
                id="edit_notebook_cell",
            ),
            pytest.param(
                "src/*.py", "read_env",
                {"path": "secrets/.env"},
                id="read_env",
            ),
        ],
    )
    def test_outside_scope_rejected(self, tmp_path, scope, tool_name, kwargs):
        agent = _make_agent(tmp_path, scope=scope)
        result = agent._execute_tool(
            _call(tool_name, **kwargs), interactive=False
        )
        assert _rejected(result)

    def test_search_replace_glob_inside_scope_allowed(self, tmp_path):
        agent = _make_agent(tmp_path, scope="src/*.py")
        result = agent._execute_tool(
            _call("search_replace", pattern="a", replacement="b",
                  glob="src/main.py", dry_run=True),
            interactive=False,
        )
        assert "outside the active scope" not in result.output

    def test_run_shell_not_scope_filtered(self, tmp_path):
        """run_shell is intentionally unscoped - never scope-rejected."""
        agent = _make_agent(tmp_path, scope="src/*.py")
        result = agent._execute_tool(
            _call("run_shell", command="echo hi"), interactive=False
        )
        assert "outside the active scope" not in result.output


# ---------------------------------------------------------------------------
#  Anthropic Messages API shape
# ---------------------------------------------------------------------------

class TestAnthropicBackend:
    def test_constructor_targets_messages_endpoint(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        backend = make_anthropic_backend(model="claude-opus-4-5")

        captured = {}

        def fake_post(url, *, headers, json_body, timeout, stream=False, verify=True, retry_503=True):
            captured["url"] = url
            captured["headers"] = headers
            captured["json_body"] = json_body
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.json.return_value = {
                "content": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
            return resp

        with patch("localm.plugins.coder.backends.http._post_with_retry",
                   side_effect=fake_post):
            out = backend.chat([{"role": "user", "content": "hello"}])

        assert out == "hi"
        # URL must hit the Anthropic Messages endpoint, NOT /chat/completions
        assert captured["url"].endswith("/v1/messages"), captured["url"]
        assert "/chat/completions" not in captured["url"]
        # Headers: x-api-key + anthropic-version, NOT a Bearer Authorization
        assert captured["headers"].get("x-api-key") == "sk-ant-test"
        assert "anthropic-version" in captured["headers"]
        assert "Authorization" not in captured["headers"]

    def test_base_url_is_anthropic(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        backend = make_anthropic_backend(model="claude-opus-4-5")
        # The grammar guard must still treat anthropic as an external provider.
        assert backend.supports_grammar is False

    def test_system_message_is_lifted_out(self, monkeypatch):
        """Anthropic takes a top-level `system`, not a system role message."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        backend = make_anthropic_backend(model="claude-opus-4-5")

        captured = {}

        def fake_post(url, *, headers, json_body, timeout, stream=False, verify=True, retry_503=True):
            captured["json_body"] = json_body
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.json.return_value = {"content": [{"type": "text", "text": "ok"}]}
            return resp

        with patch("localm.plugins.coder.backends.http._post_with_retry",
                   side_effect=fake_post):
            backend.chat([
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hello"},
            ])

        body = captured["json_body"]
        assert body.get("system") == "be terse"
        assert all(m["role"] != "system" for m in body["messages"])
        assert "max_tokens" in body  # Anthropic requires max_tokens
