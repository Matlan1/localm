# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for tolerant tool-call parsing (mangled finetune dialects).

The "logged" cases are verbatim model outputs from a real session where
zero tool calls parsed - the finetune wraps valid JSON in broken markers.
"""

import pytest

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
        # The <|tool_call>call:tool_call ... <tool_call|> dialect.
        text = ('<|channel>thought\n<channel|><|tool_call>call:tool_call\n'
                '{"name": "run_shell", "args": {"command": "npm init -y"}}\n'
                '<tool_call|>')
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "run_shell"
        assert calls[0].args == {"command": "npm init -y"}

    @pytest.mark.parametrize(
        "json_body, expected_name, expected_args",
        [
            ('{"name": "read_file", "args": {"path": "package.json"}}',
             "read_file", {"path": "package.json"}),
            ('{"name": "git_status", "args": {}}',
             "git_status", {}),
        ],
    )
    def test_logged_calls(self, json_body, expected_name, expected_args):
        text = f'<|tool_call>call:tool_call\n{json_body}\n<tool_call|>'
        calls = parse_tool_calls(text)
        assert calls[0].name == expected_name
        assert calls[0].args == expected_args

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
        # Doubled outer braces, as emitted by a real gemma4-4b run.
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


class TestLenientFlag:
    """ToolCall.lenient marks a call recovered ONLY because its JSON shape
    happened to match a real tool name, with no marker of its own signalling
    the model intended to call a tool at all (a bare top-level JSON object,
    or a ```json/bare ``` fence). Every OTHER recognised shape carries such a
    marker, however mangled, and must stay unflagged: execution.py keys a
    confirmation requirement on this flag."""

    TOOLS = {"read_file", "write_file", "edit_files", "run_shell", "tree"}

    def test_bare_json_object_is_lenient(self):
        text = '{"name": "read_file", "args": {"path": "a.py"}}'
        calls = parse_tool_calls(text, tool_names=self.TOOLS)
        assert calls[0].lenient is True

    def test_bare_triple_fence_is_lenient(self):
        text = '```\n{"name": "read_file", "args": {"path": "x"}}\n```'
        calls = parse_tool_calls(text, tool_names=self.TOOLS)
        assert calls[0].lenient is True

    def test_json_fence_is_lenient(self):
        text = '```json\n{"name": "tree", "args": {}}\n```'
        calls = parse_tool_calls(text, tool_names=self.TOOLS)
        assert calls[0].lenient is True

    def test_canonical_xml_wrapper_not_lenient(self):
        text = '<tool_call>\n{"name": "read_file", "args": {"path": "a.py"}}\n</tool_call>'
        calls = parse_tool_calls(text, tool_names=self.TOOLS)
        assert calls[0].lenient is False

    def test_explicit_tool_call_fence_not_lenient(self):
        text = '```tool_call\n{"name": "tree", "args": {}}\n```'
        calls = parse_tool_calls(text, tool_names=self.TOOLS)
        assert calls[0].lenient is False

    def test_explicit_tool_code_fence_not_lenient(self):
        text = '```tool_code\n{"name": "run_shell", "args": {"command": "ls"}}\n```'
        calls = parse_tool_calls(text, tool_names=self.TOOLS)
        assert calls[0].lenient is False

    def test_marker_variant_not_lenient(self):
        # Mangled <|tool_call> dialect - malformed wrapper, but a wrapper.
        text = ('<|tool_call>call:tool_call\n'
                '{"name": "read_file", "args": {"path": "package.json"}}\n'
                '<tool_call|>')
        calls = parse_tool_calls(text, tool_names=self.TOOLS)
        assert calls[0].lenient is False

    def test_hallucinated_lorem_ipsum_edit_files_call_is_lenient(self):
        # A model with no <tool_call> training free-runs past an unfired grammar
        # trigger, opens its own ## toolname heading instead of the real wrapper,
        # and the JSON body is still recovered by the bare-object fallback. It is
        # flagged so execution.py can require a human look at it.
        text = (
            '## edit_files\n'
            '{"name": "edit_files", "args": {"edits": ['
            '{"path": "./my_new_folder/file1.txt", "old": "", '
            '"new": "Lorem ipsum dolor sit amet."}]}}'
        )
        calls = parse_tool_calls(text, tool_names=self.TOOLS)
        assert len(calls) == 1
        assert calls[0].name == "edit_files"
        assert calls[0].lenient is True


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

    # ---- tool_names-gated XML-tag hallucination detection ----
    # A model can hallucinate an XML convention (<edit_file>/<read_file path=...>)
    # built from this project's REAL tool names instead of its own <tool_call>
    # wrapper. Such a response matches none of the checks above.
    _REAL_TOOL_NAMES = {"read_file", "write_file", "edit_file", "run_shell", "run_tests"}

    def test_hallucinated_xml_tag_flagged_when_tool_names_given(self):
        text = (
            '<edit_file>\n{"path": "sample.py", "old": "def add(a, b): pass", '
            '"new": "..."}\n\nLet me verify the file exists:\n\n'
            '<read_file path="sample.py">\n\nPlease confirm.'
        )
        assert looks_like_tool_attempt(text, self._REAL_TOOL_NAMES)

    def test_hallucinated_xml_tag_not_flagged_without_tool_names(self):
        # No tool_names passed -> falls back to the narrower checks only.
        text = '<edit_file>\n{"path": "sample.py"}\n</edit_file>'
        assert not looks_like_tool_attempt(text)

    def test_unregistered_tag_name_not_flagged(self):
        # A tag that merely LOOKS like a tool call, but names something not in
        # the registry, must not be treated as an attempt - only a REAL tool's
        # exact name is a reliable signal of intent.
        text = "<some_random_tag>not a tool</some_random_tag>"
        assert not looks_like_tool_attempt(text, self._REAL_TOOL_NAMES)

    def test_legitimate_prose_answer_not_flagged_even_with_tool_names(self):
        # A genuinely tool-free answer (no tag of any kind) stays unflagged
        # regardless of what tool names are known.
        text = (
            "Idempotence in HTTP: a request method is idempotent if making the "
            "same request multiple times has the same effect as making it once. "
            "GET, HEAD, OPTIONS and TRACE are idempotent; POST is not."
        )
        assert not looks_like_tool_attempt(text, self._REAL_TOOL_NAMES)

    def test_hallucinated_tag_flagged_case_insensitively(self):
        text = '<Read_File path="sample.py">'
        assert looks_like_tool_attempt(text, self._REAL_TOOL_NAMES)


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
    """Local models often emit JSON that is not quite valid - recover it instead
    of silently failing to parse."""

    def test_python_triple_quoted_content(self):
        # write_file with triple-quoted content.
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
    JSON. Without the guard an unhashable name (dict/list) raises TypeError at
    the parser's own `parsed[0] in tool_names` check and at execution's
    `call.name in self.disabled_tools`."""

    @pytest.mark.parametrize("name_literal", ["123", "null"])
    def test_scalar_name_rejected(self, name_literal):
        text = (
            f'<tool_call>\n{{"name": {name_literal}, "args": {{"path": "x.py"}}}}\n'
            '</tool_call>'
        )
        assert parse_tool_calls(text) == []

    def test_dict_name_rejected_without_typeerror(self):
        # The unhashable case: both the bare-JSON form and the explicit fenced
        # form must parse to [].
        bare = 'some text {"name": {"x": 1}, "args": {}} more text'
        assert parse_tool_calls(bare, tool_names={"read_file"}) == []
        fenced = '```tool_call\n{"name": {"x": 1}, "args": {}}\n```'
        assert parse_tool_calls(fenced, tool_names={"read_file"}) == []

    def test_string_name_still_parses(self):
        text = '<tool_call>\n{"name": "read_file", "args": {"path": "x.py"}}\n</tool_call>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "read_file"
