"""Tests for tolerant tool-call parsing (mangled finetune dialects).

The "logged" cases are verbatim model outputs from a real session where
zero tool calls parsed — the finetune wraps valid JSON in broken markers.
"""

from localm.plugins.coder.parser import parse_tool_calls


class TestCanonicalStillWorks:
    def test_xml_wrapper(self):
        text = '<tool_call>\n{"name": "read_file", "args": {"path": "a.py"}}\n</tool_call>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].args == {"path": "a.py"}

    def test_fenced_block(self):
        text = '```tool_call\n{"name": "tree", "args": {}}\n```'
        assert parse_tool_calls(text)[0].name == "tree"


class TestMangledVariants:
    def test_logged_run_shell_call(self):
        # Verbatim from the audit log: <|tool_call>call:tool_call\n{json}\n<tool_call|>
        text = ('<|channel>thought\n<channel|><|tool_call>call:tool_call\n'
                '{"name": "run_shell", "args": {"command": "npm init -y"}}\n'
                '<tool_call|>')
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "run_shell"
        assert calls[0].args == {"command": "npm init -y"}

    def test_logged_read_file_call(self):
        text = ('<|tool_call>call:tool_call\n'
                '{"name": "read_file", "args": {"path": "package.json"}}\n'
                '<tool_call|>')
        calls = parse_tool_calls(text)
        assert calls[0].name == "read_file"
        assert calls[0].args == {"path": "package.json"}

    def test_logged_git_status_call(self):
        text = ('<|tool_call>call:tool_call\n'
                '{"name": "git_status", "args": {}}\n<tool_call|>')
        calls = parse_tool_calls(text)
        assert calls[0].name == "git_status"
        assert calls[0].args == {}

    def test_gemma_native_with_real_name_prefix(self):
        # Older Gemma dialect: name in the prefix, args-only JSON body
        text = '<|tool_call>call:read_file{"path": "utils.py"}<tool_call|>'
        calls = parse_tool_calls(text)
        assert calls[0].name == "read_file"
        assert calls[0].args == {"path": "utils.py"}

    def test_gemma_quote_tokens(self):
        text = '<|tool_call>call:read_file{path:<|"|>utils.py<|"|>}<tool_call|>'
        calls = parse_tool_calls(text)
        assert calls[0].name == "read_file"
        assert calls[0].args == {"path": "utils.py"}

    def test_pipe_both_sides_markers(self):
        text = ('<|tool_call|>\n{"name": "list_dir", "args": {"path": "."}}\n'
                '<|tool_call|>')
        calls = parse_tool_calls(text)
        assert calls[0].name == "list_dir"

    def test_nested_json_args_survive(self):
        text = ('<|tool_call>call:tool_call\n'
                '{"name": "write_file", "args": {"path": "a.json", '
                '"content": "{\\"k\\": {\\"n\\": 1}}"}}\n<tool_call|>')
        calls = parse_tool_calls(text)
        assert calls[0].name == "write_file"
        assert "k" in calls[0].args["content"]

    def test_multiple_calls_in_one_response(self):
        text = ('<|tool_call>call:tool_call\n{"name": "git_status", "args": {}}\n'
                '<tool_call|>\nsome text\n'
                '<|tool_call>call:tool_call\n{"name": "git_diff", "args": {}}\n'
                '<tool_call|>')
        names = [c.name for c in parse_tool_calls(text)]
        assert names == ["git_status", "git_diff"]

    def test_garbage_without_json_ignored(self):
        assert parse_tool_calls("<|tool_call>call:tool_call\nnot json\n<tool_call|>") == []

    def test_plain_text_not_matched(self):
        assert parse_tool_calls("just a normal answer with {braces} in it") == []


class TestStreamHiding:
    def _collect(self, pieces):
        from localm.plugins.coder.agent import Agent
        shown, hidden = [], []
        for token, is_hidden in Agent._stream_hiding_tool_calls(iter(pieces)):
            (hidden if is_hidden else shown).append(token)
        return "".join(shown), "".join(hidden)

    def test_canonical_block_hidden(self):
        shown, hidden = self._collect(
            ["before ", "<tool_call>{\"name\": \"x\", \"args\": {}}</tool_call>", " after"])
        assert shown == "before  after"
        assert "name" in hidden

    def test_mangled_block_hidden(self):
        shown, hidden = self._collect(
            ["text ", '<|tool_call>call:tool_call\n{"name": "run_shell"}\n<tool_call|>'])
        assert shown == "text "
        assert "run_shell" in hidden

    def test_split_across_pieces(self):
        shown, hidden = self._collect(
            ["a <|tool_", 'call>{"name": "t"}', "<tool_call|> b"])
        assert shown == "a  b"
        assert '{"name": "t"}' in hidden
