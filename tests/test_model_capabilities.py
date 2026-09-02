# SPDX-License-Identifier: AGPL-3.0-or-later
"""Capability detection: tool use, context length, reasoning, vision.

Every answer here is a TRI-STATE and the tests exist mainly to hold the third
state open: ``None`` means NOT INSPECTED and must never collapse into ``False``,
because a caller that renders "cannot do X" from an unread file is making a claim
about a model nobody opened.

The GGUF cases build REAL GGUF containers (real magic, real KV framing, walked by
the real parser) carrying chat-template text taken VERBATIM from two real
published models, so the positive and the negative are genuine artifacts rather
than strings chosen to match the detector:

  tool-calling   Qwen2.5-Instruct - the ``{%- if tools %}`` branch of its own
                 baked-in template, through the ``</tool_call>`` formatting.
  plain chat     SmolLM2-135M-Instruct - its complete template, which has no
                 tools branch at all.
"""

import struct
import unittest
from pathlib import Path

import pytest

from localm.model_manager import capabilities as caps
from localm.model_manager.gguf import (
    _gguf_capability_probe,
    chat_template_tool_signal,
    gguf_capability_metadata,
    gguf_context_length,
    gguf_tool_use_signal,
)

# Verbatim from Qwen2.5-Instruct's own tokenizer.chat_template.
REAL_TOOL_TEMPLATE = r"""{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}"""

# Verbatim from SmolLM2-135M-Instruct's own tokenizer.chat_template, in full.
REAL_PLAIN_TEMPLATE = (
    "{% for message in messages %}{% if loop.first and messages[0]['role'] != "
    "'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named "
    "SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' "
    "+ message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

_T_UINT32 = 4
_T_STRING = 8
_T_ARRAY = 9


def _kv_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<I", _T_STRING) + struct.pack("<Q", len(raw)) + raw


def _kv_uint32(value: int) -> bytes:
    return struct.pack("<I", _T_UINT32) + struct.pack("<I", value)


def _kv_string_array(values) -> bytes:
    """A string ARRAY, the shape a tokenizer vocabulary takes. Present in these
    fixtures on purpose: it is what pushes the chat template past a bounded read
    in a real file, so a test that omitted it could not exercise the truncation
    path at all."""
    out = struct.pack("<I", _T_ARRAY) + struct.pack("<I", _T_STRING)
    out += struct.pack("<Q", len(values))
    for v in values:
        raw = v.encode("utf-8")
        out += struct.pack("<Q", len(raw)) + raw
    return out


def write_gguf(path: Path, kv, *, version: int = 3, tensor_count: int = 0,
               kv_count=None) -> Path:
    """Write a real GGUF container holding *kv*, a list of (key, encoded value).

    *kv_count* overrides the declared entry count, so a test can declare more
    entries than it writes and produce a genuinely truncated header."""
    body = b""
    for key, encoded in kv:
        kb = key.encode("utf-8")
        body += struct.pack("<Q", len(kb)) + kb + encoded
    header = (b"GGUF" + struct.pack("<I", version)
              + struct.pack("<Q", tensor_count)
              + struct.pack("<Q", len(kv) if kv_count is None else kv_count))
    path.write_bytes(header + body)
    return path


def _model_gguf(path: Path, *, arch="qwen2", template=None, ctx=None,
                vocab=0, **kw) -> Path:
    kv = [("general.architecture", _kv_string(arch))]
    if ctx is not None:
        kv.append((f"{arch}.context_length", _kv_uint32(ctx)))
    if vocab:
        kv.append(("tokenizer.ggml.tokens",
                   _kv_string_array([f"tok{i:05d}" for i in range(vocab)])))
    if template is not None:
        kv.append(("tokenizer.chat_template", _kv_string(template)))
    return write_gguf(path, kv, **kw)


# --------------------------------------------------------------------------- #
#  The template signal itself                                                  #
# --------------------------------------------------------------------------- #

class TestChatTemplateToolSignal:
    def test_real_tool_calling_template_is_true(self):
        assert chat_template_tool_signal(REAL_TOOL_TEMPLATE) is True

    def test_real_plain_chat_template_is_false(self):
        assert chat_template_tool_signal(REAL_PLAIN_TEMPLATE) is False

    def test_absent_template_is_unknown_not_false(self):
        assert chat_template_tool_signal(None) is None

    def test_prose_about_tools_does_not_match(self):
        """A system prompt that merely mentions tools is not a tool template.

        The markers are Jinja control constructs and OpenAI message fields, so
        ordinary English about tools must not read as tool-call support."""
        prose = ("{% for message in messages %}You are a helpful assistant with "
                 "many tools at your disposal, including tools for search."
                 "{% endfor %}")
        assert chat_template_tool_signal(prose) is False


# --------------------------------------------------------------------------- #
#  GGUF header reads                                                           #
# --------------------------------------------------------------------------- #

class TestGgufToolUseSignal:
    def test_real_tool_model_header(self, tmp_path):
        p = _model_gguf(tmp_path / "tools.gguf", template=REAL_TOOL_TEMPLATE,
                        ctx=32768, vocab=64)
        assert gguf_tool_use_signal(p) is True

    def test_real_plain_model_header(self, tmp_path):
        p = _model_gguf(tmp_path / "plain.gguf", arch="llama",
                        template=REAL_PLAIN_TEMPLATE, ctx=8192, vocab=64)
        assert gguf_tool_use_signal(p) is False

    def test_walked_in_full_with_no_template_is_false(self, tmp_path):
        """No chat-template key, but every declared entry was read: a real
        answer (the file declares no template), not a failed read."""
        p = _model_gguf(tmp_path / "notmpl.gguf", ctx=4096, vocab=8)
        meta = _gguf_capability_probe(p)
        assert meta["complete"] is True
        assert meta["chat_template"] is None
        assert gguf_tool_use_signal(p) is False

    def test_truncated_read_is_unknown_not_false(self, tmp_path, monkeypatch):
        """The tri-state's whole point: a template that sits past the read bound
        answers UNKNOWN.

        Reported as False, every model with a big vocabulary would be a model
        confirmed not to support tools, on no evidence at all. The bound is
        shrunk rather than the file grown so the case is exercised without
        writing a 16 MiB fixture."""
        p = _model_gguf(tmp_path / "big.gguf", template=REAL_TOOL_TEMPLATE,
                        ctx=32768, vocab=400)
        assert gguf_tool_use_signal(p) is True          # reachable by default

        import localm.model_manager.gguf as g
        monkeypatch.setattr(g, "_GGUF_CAPABILITY_PROBE_BYTES", 512)
        meta = _gguf_capability_probe(p)
        assert meta["complete"] is False
        assert gguf_tool_use_signal(p) is None

    def test_unreadable_path_is_unknown(self, tmp_path):
        assert gguf_tool_use_signal(tmp_path / "nope.gguf") is None

    def test_not_a_gguf_is_unknown(self, tmp_path):
        p = tmp_path / "bad.gguf"
        p.write_bytes(b"NOTGGUF" + b"\x00" * 64)
        assert gguf_tool_use_signal(p) is None

    def test_gguf_v1_is_unknown(self, tmp_path):
        """v1 used 32-bit counts, so the v2+ layout would mis-parse it."""
        p = _model_gguf(tmp_path / "v1.gguf", template=REAL_TOOL_TEMPLATE,
                        version=1)
        assert gguf_tool_use_signal(p) is None


class TestGgufContextLength:
    @pytest.mark.parametrize("arch,ctx", [("qwen2", 32768), ("llama", 8192),
                                          ("bert", 512)])
    def test_reads_the_architecture_prefixed_key(self, tmp_path, arch, ctx):
        """The three architectures this was verified against on real files."""
        p = _model_gguf(tmp_path / f"{arch}.gguf", arch=arch, ctx=ctx)
        assert gguf_context_length(p) == ctx

    def test_another_architectures_key_does_not_win(self, tmp_path):
        """A projector block's parallel key must never answer for the model.

        Keys are collected by full name and resolved against
        general.architecture, so clip.context_length is ignored here even though
        it also ends in .context_length."""
        p = write_gguf(tmp_path / "m.gguf", [
            ("clip.context_length", _kv_uint32(77)),
            ("general.architecture", _kv_string("qwen2")),
            ("qwen2.context_length", _kv_uint32(32768)),
        ])
        assert gguf_context_length(p) == 32768

    def test_absent_key_is_none(self, tmp_path):
        p = _model_gguf(tmp_path / "noctx.gguf")
        assert gguf_context_length(p) is None

    def test_zero_is_not_a_context_length(self, tmp_path):
        """A declared 0 is not a usable window, so it reads as unknown rather
        than as a model with a zero-token context."""
        p = _model_gguf(tmp_path / "zero.gguf", ctx=0)
        assert gguf_context_length(p) is None

    def test_unreadable_path_is_none(self, tmp_path):
        assert gguf_context_length(tmp_path / "gone.gguf") is None


class TestGgufCapabilityMetadata:
    def test_both_together_from_one_read(self, tmp_path):
        p = _model_gguf(tmp_path / "m.gguf", template=REAL_TOOL_TEMPLATE,
                        ctx=32768, vocab=16)
        assert gguf_capability_metadata(p) == {"tool_use": True,
                                               "context_length": 32768}

    def test_unreadable_file_is_unknown_on_both(self, tmp_path):
        assert gguf_capability_metadata(tmp_path / "x.gguf") == {
            "tool_use": None, "context_length": None}


# --------------------------------------------------------------------------- #
#  Registry-level capability API                                               #
# --------------------------------------------------------------------------- #

def _reg(**entries):
    return dict(entries)


class TestRegistryToolUse:
    def test_stored_true(self, tmp_path):
        reg = _reg(m={"path": str(tmp_path / "m.gguf"), "tool_use": True})
        assert caps.model_tool_use_capability("m", reg=reg) is True

    def test_stored_false_stays_false(self, tmp_path):
        """A confirmed negative must not degrade into unknown: the model's own
        template WAS read and renders no tool calls."""
        reg = _reg(m={"path": str(tmp_path / "m.gguf"), "tool_use": False})
        assert caps.model_tool_use_capability("m", reg=reg) is False

    def test_key_absent_is_unknown_not_false(self, tmp_path):
        """An entry registered before the field existed. Nobody has looked, so
        the answer is unknown - the case the whole tri-state exists for."""
        reg = _reg(m={"path": str(tmp_path / "m.gguf")})
        assert caps.model_tool_use_capability("m", reg=reg) is None

    def test_unknown_model_is_unknown(self):
        assert caps.model_tool_use_capability("absent", reg={}) is None

    def test_malformed_entry_is_unknown(self):
        """A bare string entry from a hand-edited registry.json."""
        reg = {"m": "Z:/models/m.gguf"}
        assert caps.model_tool_use_capability("m", reg=reg) is None

    def test_hf_directory_template_is_read_live(self, tmp_path):
        """A HuggingFace directory has no registration-time capture, so its
        template is read from tokenizer_config.json on demand."""
        import json as _json
        d = tmp_path / "hf"
        d.mkdir()
        (d / "tokenizer_config.json").write_text(
            _json.dumps({"chat_template": REAL_TOOL_TEMPLATE}), encoding="utf-8")
        reg = _reg(m={"path": str(d)})
        assert caps.model_tool_use_capability("m", reg=reg) is True

    def test_hf_directory_without_template_is_unknown(self, tmp_path):
        d = tmp_path / "hf"
        d.mkdir()
        (d / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        reg = _reg(m={"path": str(d)})
        assert caps.model_tool_use_capability("m", reg=reg) is None


class TestRegistryContextLength:
    def test_stored_value(self, tmp_path):
        reg = _reg(m={"path": str(tmp_path / "m.gguf"), "context_length": 32768})
        assert caps.model_context_length("m", reg=reg) == 32768

    def test_absent_is_none(self, tmp_path):
        reg = _reg(m={"path": str(tmp_path / "m.gguf")})
        assert caps.model_context_length("m", reg=reg) is None

    def test_hf_directory_config_is_read_live(self, tmp_path):
        import json as _json
        d = tmp_path / "hf"
        d.mkdir()
        (d / "config.json").write_text(
            _json.dumps({"max_position_embeddings": 4096}), encoding="utf-8")
        reg = _reg(m={"path": str(d)})
        assert caps.model_context_length("m", reg=reg) == 4096


class TestReasoning:
    def test_marker_in_the_name_is_true(self, tmp_path):
        reg = _reg(**{"DeepSeek-R1-Distill": {"path": str(tmp_path / "m.gguf")}})
        assert caps.model_reasoning_capability("DeepSeek-R1-Distill",
                                               reg=reg) is True

    def test_no_marker_is_unknown_NEVER_false(self, tmp_path):
        """A name heuristic can evidence a reasoning model; it can never
        evidence the absence of one. An opaque alias is exactly what a reasoning
        model with a plain name looks like, so False would assert what no signal
        supports."""
        reg = _reg(m8={"path": str(tmp_path / "m.gguf")})
        got = caps.model_reasoning_capability("m8", reg=reg)
        assert got is None
        assert got is not False

    def test_unknown_model_is_unknown(self):
        assert caps.model_reasoning_capability("qwq-nonexistent", reg={}) is None

    def test_is_thinking_model_contract_is_unchanged(self):
        """The two existing call sites (the coder's per-family prompt tuning and
        chat's <think> inlet) keep a plain bool: the capability layer wraps this
        rather than replacing it."""
        from localm.inference.model_family import is_thinking_model
        assert is_thinking_model("DeepSeek-R1-Distill") is True
        assert is_thinking_model("m8") is False
        assert is_thinking_model("") is False


class TestCapabilityDispatch:
    def test_unknown_capability_name_raises(self):
        """A typo must not answer None, which is indistinguishable from a real
        "not inspected" and would silently mean "nothing qualifies" forever."""
        with pytest.raises(ValueError):
            caps.model_capability("m", "tool-use", reg={})

    def test_every_boolean_capability_is_dispatchable(self, tmp_path):
        reg = _reg(m={"path": str(tmp_path / "m.gguf"), "model_type": "llm"})
        for cap in caps.BOOLEAN_CAPABILITIES:
            caps.model_capability("m", cap, reg=reg)

    def test_capabilities_report_always_carries_every_key(self, tmp_path):
        reg = _reg(m={"path": str(tmp_path / "m.gguf")})
        got = caps.model_capabilities("m", reg=reg)
        assert set(got) == set(caps.BOOLEAN_CAPABILITIES) | {caps.CONTEXT_LENGTH}


class TestPositiveMembership:
    def test_only_confirmed_models_are_listed(self, tmp_path):
        reg = _reg(
            yes={"path": str(tmp_path / "a.gguf"), "tool_use": True},
            no={"path": str(tmp_path / "b.gguf"), "tool_use": False},
            unknown={"path": str(tmp_path / "c.gguf")},
        )
        assert caps.models_with_capability(caps.TOOL_USE, reg=reg) == ["yes"]

    def test_context_listing_is_roomiest_first(self, tmp_path):
        reg = _reg(
            small={"path": str(tmp_path / "a.gguf"), "context_length": 4096},
            big={"path": str(tmp_path / "b.gguf"), "context_length": 131072},
            mid={"path": str(tmp_path / "c.gguf"), "context_length": 32768},
            unknown={"path": str(tmp_path / "d.gguf")},
        )
        assert caps.models_with_context_at_least(4096, reg=reg) == [
            "big", "mid", "small"]

    def test_unknown_context_is_never_listed(self, tmp_path):
        reg = _reg(unknown={"path": str(tmp_path / "a.gguf")})
        assert caps.models_with_context_at_least(1, reg=reg) == []


class TestVisionStillDelegates(unittest.TestCase):
    def test_capability_layer_calls_the_shipped_probe(self):
        """Vision is not reimplemented here. It stays a LIVE probe because it
        depends on a sibling projector file that moves independently of the
        model, so a stored answer would go stale without the model changing."""
        from unittest.mock import patch
        with patch("localm.model_manager.registry.model_vision_capability",
                   return_value=None) as probe:
            got = caps.model_capability("m", caps.VISION, reg={})
        probe.assert_called_once()
        self.assertIsNone(got)
