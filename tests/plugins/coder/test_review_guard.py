# SPDX-License-Identifier: AGPL-3.0-or-later
"""Review guard: classify the changed files a passing check cannot vouch for
(rewritten tests / weakened CI config) so the CLI can surface them for review.
"""

from __future__ import annotations

from localm.plugins.coder.review_guard import (
    classify_sensitive_changes,
    render_reviewer_note,
    render_warning,
)


def _cf(*paths):
    return [{"path": p} for p in paths]


def test_classifies_python_test_files():
    flags = classify_sensitive_changes(_cf(
        "tests/test_foo.py", "src/app.py", "foo_test.py", "pkg/conftest.py"))
    assert set(flags["tests"]) == {"tests/test_foo.py", "foo_test.py",
                                   "pkg/conftest.py"}
    assert "src/app.py" not in flags["tests"]


def test_classifies_js_ts_test_files():
    flags = classify_sensitive_changes(_cf(
        "ui/Button.test.tsx", "ui/util.spec.js", "ui/util.ts"))
    assert set(flags["tests"]) == {"ui/Button.test.tsx", "ui/util.spec.js"}


def test_classifies_ci_config():
    flags = classify_sensitive_changes(_cf(
        ".github/workflows/ci.yml", ".pre-commit-config.yaml", "ruff.toml",
        ".flake8", "README.md"))
    assert set(flags["ci_config"]) == {".github/workflows/ci.yml",
                                       ".pre-commit-config.yaml", "ruff.toml",
                                       ".flake8"}
    assert "README.md" not in flags["ci_config"]
    assert "README.md" not in flags["tests"]


def test_plain_source_is_not_flagged():
    flags = classify_sensitive_changes(_cf(
        "src/main.py", "lib/helpers.py", "docs/guide.md", "pyproject.toml"))
    assert flags == {"tests": [], "ci_config": []}     # incl. pyproject


def test_windows_paths_are_normalised():
    flags = classify_sensitive_changes([
        {"path": "tests\\test_x.py"},
        {"path": ".github\\workflows\\ci.yml"},
    ])
    assert "tests/test_x.py" in flags["tests"]
    assert ".github/workflows/ci.yml" in flags["ci_config"]


def test_a_file_is_at_most_one_category():
    # conftest under tests/ is a test, never double-counted as config.
    flags = classify_sensitive_changes(_cf("tests/conftest.py"))
    assert flags["tests"] == ["tests/conftest.py"]
    assert flags["ci_config"] == []


def test_accepts_plain_path_strings_and_dedupes():
    flags = classify_sensitive_changes(["tests/test_a.py", "tests/test_a.py"])
    assert flags["tests"] == ["tests/test_a.py"]


def test_render_warning_empty_when_nothing_sensitive():
    assert render_warning({"tests": [], "ci_config": []}) == ""
    assert render_warning(classify_sensitive_changes(_cf("src/app.py"))) == ""


def test_render_warning_lists_files_and_says_review():
    msg = render_warning({"tests": ["tests/test_x.py"], "ci_config": [".flake8"]})
    assert "tests/test_x.py" in msg
    assert ".flake8" in msg
    assert "review" in msg.lower()


# --------------------------------------------------------------------------- #
#  render_reviewer_note: the same fact, addressed to the REVIEWER MODEL        #
# --------------------------------------------------------------------------- #

def test_render_reviewer_note_empty_when_nothing_sensitive():
    assert render_reviewer_note({"tests": [], "ci_config": []}) == ""
    assert render_reviewer_note(classify_sensitive_changes(_cf("src/app.py"))) == ""


def test_render_reviewer_note_lists_files_and_instructs_scrutiny():
    note = render_reviewer_note({"tests": ["tests/test_x.py"], "ci_config": [".flake8"]})
    assert "tests/test_x.py" in note
    assert ".flake8" in note
    assert "scrutinize" in note.lower()


def test_cli_warn_helper_surfaces_edits(capsys):
    import localm.plugins.coder.cli as cli

    class _Agent:
        def changed_files(self):
            return [{"path": "tests/test_x.py"}, {"path": "src/ok.py"}]

    cli._warn_sensitive_changes(_Agent())
    captured = capsys.readouterr()
    assert "tests/test_x.py" in (captured.out + captured.err)


def test_cli_warn_helper_silent_when_nothing_sensitive(capsys):
    import localm.plugins.coder.cli as cli

    class _Agent:
        def changed_files(self):
            return [{"path": "src/app.py"}]

    cli._warn_sensitive_changes(_Agent())
    captured = capsys.readouterr()
    assert "review" not in (captured.out + captured.err).lower()


def test_cli_warn_helper_never_raises():
    import localm.plugins.coder.cli as cli

    class _Boom:
        def changed_files(self):
            raise RuntimeError("boom")

    cli._warn_sensitive_changes(_Boom())     # best-effort: must not propagate


# --------------------------------------------------------------------------- #
#  The notice reaches the FINAL ANSWER TEXT, not only a console print.        #
#                                                                              #
# _warn_sensitive_changes above is console-only, reached via finish_agent -   #
# which a GUI session's CoderSession.close() never calls, and which an MCP    #
# run_coder_task caller can never see (its stdout is swallowed and its        #
# TaskResult.response was already captured before finish_agent ever runs).    #
# loop.py's _sensitive_changes_notice folds the same fact into the final      #
# answer text itself, so every surface gets it by construction.               #
# --------------------------------------------------------------------------- #

def _make_agent(tmp_path, **kwargs):
    from unittest.mock import MagicMock, patch
    from localm.plugins.coder.agent import Agent
    backend = MagicMock()
    backend.model_id = "test-model"
    backend.native_tools = False
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        return Agent(backend=backend, cwd=tmp_path, **kwargs)


def test_final_answer_carries_the_sensitive_changes_notice(tmp_path):
    from unittest.mock import patch
    agent = _make_agent(tmp_path, auto_approve=True, self_verify=False)
    responses = iter([
        '<tool_call>\n{"name": "write_file", "args": '
        '{"path": "tests/test_x.py", "content": "def test_x(): pass\\n"}}'
        '\n</tool_call>\n',
        "Done.",
    ])
    with patch.object(agent, "_call_llm", side_effect=lambda *a, **k: next(responses)):
        result = agent.run_task("add a test")
    assert "tests/test_x.py" in result
    assert "review these by hand" in result.lower()


def test_final_answer_stays_clean_when_nothing_sensitive_changed(tmp_path):
    """Control: an ordinary source file must not draw the notice, or it is
    noise on every task rather than signal on the rare sensitive one."""
    from unittest.mock import patch
    agent = _make_agent(tmp_path, auto_approve=True, self_verify=False)
    responses = iter([
        '<tool_call>\n{"name": "write_file", "args": '
        '{"path": "app.py", "content": "x = 1\\n"}}\n</tool_call>\n',
        "Done.",
    ])
    with patch.object(agent, "_call_llm", side_effect=lambda *a, **k: next(responses)):
        result = agent.run_task("write app.py")
    assert "review these by hand" not in result.lower()


def test_sensitive_changes_notice_is_emitted_as_an_event_too(tmp_path):
    """A GUI/MCP caller may read the event stream instead of (or as well as)
    the final text; on_event is the channel both rely on (loop.py's _emit)."""
    from unittest.mock import patch
    events = []
    agent = _make_agent(tmp_path, auto_approve=True, self_verify=False,
                        on_event=events.append)
    responses = iter([
        '<tool_call>\n{"name": "write_file", "args": '
        '{"path": "tests/test_x.py", "content": "def test_x(): pass\\n"}}'
        '\n</tool_call>\n',
        "Done.",
    ])
    with patch.object(agent, "_call_llm", side_effect=lambda *a, **k: next(responses)):
        agent.run_task("add a test")
    texts = [str(e.get("text", "")) for e in events if e.get("type") == "info"]
    assert any("tests/test_x.py" in t for t in texts), texts


def test_sensitive_changes_notice_survives_a_review_guard_import_failure(tmp_path):
    """Best-effort like its console-only predecessor: a broken import must
    never cost the user their answer."""
    from unittest.mock import patch
    agent = _make_agent(tmp_path, auto_approve=True, self_verify=False)
    responses = iter([
        '<tool_call>\n{"name": "write_file", "args": '
        '{"path": "tests/test_x.py", "content": "def test_x(): pass\\n"}}'
        '\n</tool_call>\n',
        "Done.",
    ])
    with patch.object(agent, "_call_llm", side_effect=lambda *a, **k: next(responses)), \
         patch("localm.plugins.coder.review_guard.classify_sensitive_changes",
               side_effect=RuntimeError("boom")):
        result = agent.run_task("add a test")
    assert result.startswith("Done.")
