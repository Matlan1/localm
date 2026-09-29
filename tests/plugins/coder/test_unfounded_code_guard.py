# SPDX-License-Identifier: AGPL-3.0-or-later
"""A final answer that shows a code block for a workspace file, defining names
that file does not contain, is invented file content, not the file.

Checked against the file on disk, never against the reply's own account of
what it read: the model announcing "I'll check X" and then printing functions X
does not have is exactly the case its wording cannot be trusted for.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from localm.plugins.coder.agent.loop import unfounded_code

API_PY = (
    "def llama_batch_get_one(tokens):\n"
    "    return tokens\n"
    "\n"
    "\n"
    "class LlamaBatch:\n"
    "    pass\n"
)

INVENTED = (
    "I'll check the `pkg\\llamacpp\\_api.py` file for tokenization-related functions.\n"
    "\n"
    "```python\n"
    "def llama_batch_get_token(L, idx):\n"
    "    return fn(L, idx)\n"
    "\n"
    "\n"
    "def llama_batch_free_token(token):\n"
    "    fn(token)\n"
    "```\n"
    "\n"
    "This file contains low-level C API bindings.")


def _workspace(tmp_path: Path) -> Path:
    (tmp_path / "pkg" / "llamacpp").mkdir(parents=True)
    (tmp_path / "pkg" / "llamacpp" / "_api.py").write_text(API_PY, encoding="utf-8")
    return tmp_path


def _make_agent(tmp_path: Path, **kwargs):
    from localm.plugins.coder.agent import Agent
    backend = MagicMock()
    backend.model_id = "test-model"
    backend.native_tools = False
    with patch("localm.plugins.coder.agent.ProjectMap") as MockPM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        MockPM.build.return_value.file_count.return_value = 0
        agent = Agent(backend=backend, cwd=tmp_path, **kwargs)
    return agent


def _notes(agent) -> list:
    return [m["content"] for m in agent._messages
            if m.get("role") == "user"
            and str(m.get("content", "")).startswith("[unverified code]")]


class TestUnfoundedCode:
    def test_invented_functions_for_a_real_file_are_reported(self, tmp_path):
        ws = _workspace(tmp_path)
        assert unfounded_code(INVENTED, ws) == [
            ("pkg/llamacpp/_api.py",
             ["llama_batch_get_token", "llama_batch_free_token"])]

    def test_code_that_matches_the_file_is_not_reported(self, tmp_path):
        ws = _workspace(tmp_path)
        text = ("In `pkg/llamacpp/_api.py`:\n\n```python\n"
                "def llama_batch_get_one(tokens):\n    ...\n\n"
                "class LlamaBatch:\n    ...\n```\n")
        assert unfounded_code(text, ws) == []

    def test_only_names_the_block_defines_are_checked(self, tmp_path):
        ws = _workspace(tmp_path)
        text = ("`pkg/llamacpp/_api.py` uses it like this:\n\n```python\n"
                "batch = llama_batch_get_one(tokens)\nmade_up_helper(batch)\n```\n")
        assert unfounded_code(text, ws) == []

    def test_a_block_is_checked_against_the_path_named_just_before_it(self, tmp_path):
        ws = _workspace(tmp_path)
        (ws / "pkg" / "other.py").write_text("def llama_batch_get_token():\n    pass\n",
                                             encoding="utf-8")
        text = ("`pkg/other.py` has it:\n\n```python\ndef llama_batch_get_token():\n"
                "    pass\n```\n\nand `pkg/llamacpp/_api.py` has:\n\n```python\n"
                "def llama_batch_free_token():\n    pass\n```\n")
        assert unfounded_code(text, ws) == [
            ("pkg/llamacpp/_api.py", ["llama_batch_free_token"])]

    def test_a_new_file_is_not_checked(self, tmp_path):
        ws = _workspace(tmp_path)
        text = "Create `pkg/new_module.py`:\n\n```python\ndef helper():\n    pass\n```\n"
        assert unfounded_code(text, ws) == []

    def test_code_with_no_named_file_is_not_checked(self, tmp_path):
        ws = _workspace(tmp_path)
        assert unfounded_code("```python\ndef anything():\n    pass\n```", ws) == []

    def test_prose_without_a_code_block_is_not_checked(self, tmp_path):
        ws = _workspace(tmp_path)
        text = "`pkg/llamacpp/_api.py` defines `llama_batch_get_token`."
        assert unfounded_code(text, ws) == []

    def test_a_file_outside_the_workspace_is_never_read(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (tmp_path / "outside.py").write_text("x = 1\n", encoding="utf-8")
        text = "In `../outside.py`:\n\n```python\ndef not_there():\n    pass\n```\n"
        assert unfounded_code(text, ws) == []

    def test_other_languages_definitions_are_checked(self, tmp_path):
        ws = tmp_path
        (ws / "app.js").write_text("function realOne() {}\n", encoding="utf-8")
        text = ("From `app.js`:\n\n```js\nfunction realOne() {}\n"
                "async function inventedTwo() {}\n```\n")
        assert unfounded_code(text, ws) == [("app.js", ["inventedTwo"])]


class TestUnfoundedCodeInTheLoop:
    def test_invented_file_content_gets_one_repair_turn(self, tmp_path):
        agent = _make_agent(_workspace(tmp_path), auto_approve=True)
        responses = iter([
            INVENTED,
            '<tool_call>\n{"name": "read_file", "args": '
            '{"path": "pkg/llamacpp/_api.py"}}\n</tool_call>\n',
            "It defines `llama_batch_get_one` and the `LlamaBatch` class.",
        ])
        with patch.object(agent, "_call_llm",
                          side_effect=lambda *a, **k: next(responses)) as llm:
            result = agent.run_task("what tokenization functions do we have?")
        assert llm.call_count == 3
        notes = _notes(agent)
        assert len(notes) == 1
        assert "llama_batch_get_token" in notes[0]
        assert "pkg/llamacpp/_api.py" in notes[0]
        assert result.startswith("It defines `llama_batch_get_one`")
        assert "[unverified code" not in result

    def test_matching_code_is_accepted_without_a_repair_turn(self, tmp_path):
        agent = _make_agent(_workspace(tmp_path))
        answer = ("`pkg/llamacpp/_api.py` has:\n\n```python\n"
                  "def llama_batch_get_one(tokens):\n    return tokens\n```")
        with patch.object(agent, "_call_llm", return_value=answer) as llm:
            result = agent.run_task("what tokenization functions do we have?")
        assert llm.call_count == 1
        assert _notes(agent) == []
        assert result.startswith(answer)

    def test_the_repair_is_once_per_task_and_the_answer_is_flagged(self, tmp_path):
        agent = _make_agent(_workspace(tmp_path))
        events = []
        agent.on_event = events.append
        with patch.object(agent, "_call_llm", return_value=INVENTED) as llm:
            result = agent.run_task("what tokenization functions do we have?")
        assert llm.call_count == 2
        assert len(_notes(agent)) == 1
        assert ("[unverified code: llama_batch_get_token, llama_batch_free_token "
                "not found in pkg/llamacpp/_api.py]") in result
        infos = [e.get("text", "") for e in events if e.get("type") == "info"]
        assert any(i.startswith("unverified code: llama_batch_get_token") for i in infos), (
            "the GUI renders info events, not the final text, so the flag is emitted too")
