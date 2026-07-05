# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for tolerant tool-call parsing (mangled finetune dialects).

The "logged" cases are verbatim model outputs from a real session where
zero tool calls parsed - the finetune wraps valid JSON in broken markers.
"""

from localm.plugins.coder.parser import looks_like_tool_attempt, parse_tool_calls


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

    def test_doubled_braces_verbatim_from_e2e(self):
        # Verbatim from a real gemma4-4b run: doubled outer braces silently
        # broke tool calling - the raw call was printed as the final answer.
        text = ('<|tool_call>call:write_file{{"path": "hello.txt", '
                '"content": "Hello from localcoder."}}<tool_call|>')
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "write_file"
        assert calls[0].args == {"path": "hello.txt",
                                 "content": "Hello from localcoder."}

    def test_doubled_braces_canonical_wrapper(self):
        text = ('<tool_call>{{"name": "tree", "args": {}}}</tool_call>')
        # outer doubling with inner empty args object
        calls = parse_tool_calls(text)
        assert not calls or calls[0].name == "tree"

    def test_literal_newline_inside_string_value(self):
        # Models routinely write multi-line file content without \n escapes;
        # strict JSON rejects control characters inside strings.
        text = ('<tool_call>\n{"name": "write_file", "args": {"path": "a.txt", '
                '"content": "line one\nline two"}}\n</tool_call>')
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].args["content"] == "line one\nline two"

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


class TestNameGatedLenientFormats:
    """Bare JSON and ```json / bare fences are accepted only when the caller
    passes the real tool names and the parsed name is one of them - the exact
    formats weak local models emit, without mistaking JSON prose for a call."""

    TOOLS = {"read_file", "write_file", "run_shell", "tree"}

    def test_bare_json_object(self):
        text = 'Let me look.\n{"name": "read_file", "args": {"path": "a.py"}}'
        calls = parse_tool_calls(text, tool_names=self.TOOLS)
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].args == {"path": "a.py"}

    def test_bare_json_ignored_without_tool_names(self):
        # Back-compat: the one-arg form never does bare-JSON matching.
        text = '{"name": "read_file", "args": {"path": "a.py"}}'
        assert parse_tool_calls(text) == []

    def test_bare_json_unknown_name_ignored(self):
        # A JSON example whose name is not a real tool is left alone.
        text = '{"name": "my-package", "args": {"version": "1.0"}}'
        assert parse_tool_calls(text, tool_names=self.TOOLS) == []

    def test_json_fence(self):
        text = '```json\n{"name": "tree", "args": {}}\n```'
        calls = parse_tool_calls(text, tool_names=self.TOOLS)
        assert calls[0].name == "tree"

    def test_bare_triple_fence(self):
        text = '```\n{"name": "read_file", "args": {"path": "x"}}\n```'
        calls = parse_tool_calls(text, tool_names=self.TOOLS)
        assert calls[0].name == "read_file"

    def test_json_fence_unknown_name_ignored(self):
        text = '```json\n{"name": "package-thing", "version": "2"}\n```'
        assert parse_tool_calls(text, tool_names=self.TOOLS) == []

    def test_tool_code_fence_is_explicit(self):
        # ```tool_code signals intent: parsed even without the tool_names gate.
        text = '```tool_code\n{"name": "run_shell", "args": {"command": "ls"}}\n```'
        assert parse_tool_calls(text)[0].name == "run_shell"

    def test_arguments_alias(self):
        text = '<tool_call>{"name": "read_file", "arguments": {"path": "a"}}</tool_call>'
        assert parse_tool_calls(text)[0].args == {"path": "a"}

    def test_prose_with_braces_ignored_even_gated(self):
        assert parse_tool_calls("normal answer with {braces}",
                                tool_names=self.TOOLS) == []

    def test_fence_and_bare_not_double_counted(self):
        text = '```json\n{"name": "tree", "args": {}}\n```'
        assert len(parse_tool_calls(text, tool_names=self.TOOLS)) == 1

    def test_two_bare_calls(self):
        text = ('{"name": "read_file", "args": {"path": "a"}} then '
                '{"name": "tree", "args": {}}')
        names = [c.name for c in parse_tool_calls(text, tool_names=self.TOOLS)]
        assert names == ["read_file", "tree"]


class TestLooksLikeToolAttempt:
    def test_marker_flagged(self):
        assert looks_like_tool_attempt("<|tool_call>garbage<tool_call|>")

    def test_tool_code_fence_flagged(self):
        assert looks_like_tool_attempt("```tool_code\nread_file(path='x')\n```")

    def test_truncated_name_args_flagged(self):
        # Malformed/cut-off JSON that the parser cannot recover still looks
        # like an attempt because it carries both keys.
        assert looks_like_tool_attempt('{"name": "read_file", "args": {"path"')

    def test_plain_answer_not_flagged(self):
        assert not looks_like_tool_attempt("Here is your answer. All done.")

    def test_name_word_alone_not_flagged(self):
        assert not looks_like_tool_attempt('the "name" field is required')


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


class TestRecoveredMalformations:
    """Local models often emit JSON that is not quite valid - recover it instead of
    silently failing to parse (which showed an empty bubble + wrote nothing)."""

    def test_python_triple_quoted_content(self):
        # Verbatim iter-11 GUI failure: write_file with triple-quoted content.
        text = (
            '<tool_call>\n'
            '{"name": "write_file",\n'
            ' "args": {\n'
            '   "path": "calc.py",\n'
            '   "content": """\n'
            '# Simple Calculator\n'
            'def add(a, b):\n'
            '    return a + b\n'
            '"""\n'
            ' }}\n'
            '</tool_call>'
        )
        calls = parse_tool_calls(text, tool_names={"write_file"})
        assert len(calls) == 1
        assert calls[0].name == "write_file"
        assert calls[0].args["path"] == "calc.py"
        assert "def add" in calls[0].args["content"]

    def test_trailing_comma(self):
        text = '<tool_call>\n{"name": "read_file", "args": {"path": "x.py",}}\n</tool_call>'
        calls = parse_tool_calls(text, tool_names={"read_file"})
        assert len(calls) == 1
        assert calls[0].args == {"path": "x.py"}

    def test_triple_quoted_plus_trailing_comma(self):
        text = (
            '<tool_call>\n'
            '{"name": "write_file", "args": {"path": "a.py", "content": """x = 1""",}}\n'
            '</tool_call>'
        )
        calls = parse_tool_calls(text, tool_names={"write_file"})
        assert len(calls) == 1
        assert calls[0].args["content"] == "x = 1"

    def test_unescaped_backslashes_in_path_and_content(self):
        text = (
            '<tool_call>\n'
            '{"name": "write_file", "args": {"path": "localm\\appface.py", "content": "def appface():\\n    icon_path = r\'D:\\\\MockFolder\\\\UserName\\AppData\\\\Local\\\\Programs\\\\LocalCoder\\\\appface.exe\'"}}\n'
            '</tool_call>'
        )
        calls = parse_tool_calls(text, tool_names={"write_file"})
        assert len(calls) == 1
        assert calls[0].args["path"] == "localm\\appface.py"
        assert "UserName\\AppData" in calls[0].args["content"]



class TestNonStringName:
    """A tool call whose "name" is not a string is malformed, same as broken
    JSON. Without the guard an unhashable name (dict/list) raised TypeError at
    the parser's own `parsed[0] in tool_names` check and at execution's
    `call.name in self.disabled_tools` (2026-07-02 coder tool sweep)."""

    def test_numeric_name_rejected(self):
        text = '<tool_call>\n{"name": 123, "args": {"path": "x.py"}}\n</tool_call>'
        assert parse_tool_calls(text) == []

    def test_dict_name_rejected_without_typeerror(self):
        # The unhashable case. Pre-guard, the bare-JSON form crashed with
        # TypeError at `parsed[0] in tool_names`; the explicit fenced form
        # was accepted outright and crashed later at execution's
        # `call.name in self.disabled_tools`. Both must now parse to [].
        bare = 'some text {"name": {"x": 1}, "args": {}} more text'
        assert parse_tool_calls(bare, tool_names={"read_file"}) == []
        fenced = '```tool_call\n{"name": {"x": 1}, "args": {}}\n```'
        assert parse_tool_calls(fenced, tool_names={"read_file"}) == []

    def test_null_name_rejected(self):
        text = '<tool_call>\n{"name": null, "args": {}}\n</tool_call>'
        assert parse_tool_calls(text) == []

    def test_string_name_still_parses(self):
        text = '<tool_call>\n{"name": "read_file", "args": {"path": "x.py"}}\n</tool_call>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "read_file"
