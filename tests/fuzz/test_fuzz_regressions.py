# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plain regression tests for the inputs the fuzz suite found.

Each one is the minimal hostile input of a real crash, kept as an ordinary
test so it runs on every PR whatever the Hypothesis profile or seed."""
from __future__ import annotations

import io
import json
import struct
import zipfile

import pytest

from localm import install_manifest, media_workflows, model_meta
from localm.inference import gbnf
from localm.inference.backends.base import InvalidGrammarError
from localm.inference import hf_shard_index_safety, hf_tokenizer_safety
from localm.model_manager import capabilities, gguf, registry
from localm.plugins import engine, loader
from localm.plugins.mcpserver.server import MCPStdioServer
from localm.rag import extract
from tests.fuzz import _bounds


class TestGrammarRepeatCount:
    def test_a_repeat_count_wider_than_the_int_conversion_limit_is_an_invalid_grammar(self):
        grammar = 'root ::= "a"{' + "9" * 5000 + "}"
        with pytest.raises(InvalidGrammarError) as exc:
            gbnf.check_grammar_structure(grammar)
        assert len(str(exc.value)) < 400

    def test_a_range_with_a_huge_upper_bound_is_an_invalid_grammar(self):
        with pytest.raises(InvalidGrammarError):
            gbnf.check_grammar_structure('root ::= "a"{1,' + "9" * 5000 + "}")

    def test_leading_zeros_do_not_count_toward_the_limit(self):
        gbnf.check_grammar_structure('root ::= "a"{' + "0" * 5000 + "5}")

    @pytest.mark.parametrize("count,rejected", [(1900, False), (1901, True), (0, False)])
    def test_the_limit_is_unchanged(self, count, rejected):
        grammar = f'root ::= "a"{{{count}}}'
        if rejected:
            with pytest.raises(InvalidGrammarError):
                gbnf.check_grammar_structure(grammar)
        else:
            gbnf.check_grammar_structure(grammar)


def _plugin(tmp_path, name, manifest: bytes):
    d = tmp_path / name
    d.mkdir()
    (d / "plugin.toml").write_bytes(manifest)
    return d


_HOSTILE_MANIFESTS = {
    "not_utf8": b"\xff\xfe[plugin]\nname='x'\n",
    "entry_is_an_int": b"[plugin]\nname='x'\nentry=5\n",
    "entry_is_a_table": b"[plugin]\nname='x'\nentry={':'=1}\n",
    "nested_too_deep": b"[plugin]\nname='x'\nentry='m:f'\na=" + b"[" * 5000 + b"]" * 5000,
}


class TestLegacyManifest:
    @pytest.mark.parametrize("case", sorted(_HOSTILE_MANIFESTS))
    def test_a_hostile_manifest_raises_plugin_error(self, tmp_path, case):
        d = _plugin(tmp_path, "p", _HOSTILE_MANIFESTS[case])
        with pytest.raises(loader.PluginError):
            loader.parse_manifest(d)

    def test_one_hostile_plugin_does_not_hide_the_good_ones(self, tmp_path):
        _plugin(tmp_path, "a_bad", _HOSTILE_MANIFESTS["not_utf8"])
        _plugin(tmp_path, "b_bad", _HOSTILE_MANIFESTS["entry_is_an_int"])
        _plugin(tmp_path, "c_good", b"[plugin]\nname='good'\nentry='m:f'\n")
        found = [m.name for m in loader.discover_plugins(tmp_path)]
        assert found == ["good"]
        errors = loader.discover_errors(tmp_path)
        assert len(errors) == 2


class TestEngineSpec:
    @pytest.mark.parametrize("manifest", [
        pytest.param(_HOSTILE_MANIFESTS["not_utf8"], id="not_utf8"),
        pytest.param(_HOSTILE_MANIFESTS["nested_too_deep"], id="nested_too_deep"),
        pytest.param(b"[plugin]\nname='x'\napi_version=inf\n", id="api_version_inf"),
        pytest.param(b"[plugin]\nname='x'\napi_version='one'\n", id="api_version_text"),
        pytest.param(b"[plugin]\nname='x'\nrequires_extras=5\n", id="requires_extras_int"),
        pytest.param(b"[plugin]\nname='x'\nrequires='chat'\n", id="requires_string"),
        pytest.param(b"[plugin]\nname='x'\ncapabilities=[1, 2]\n", id="capabilities_ints"),
    ])
    def test_a_hostile_manifest_raises_value_error(self, tmp_path, manifest):
        with pytest.raises(ValueError):
            engine.parse_spec(_plugin(tmp_path, "p", manifest))

    def test_a_well_formed_manifest_still_parses(self, tmp_path):
        spec = engine.parse_spec(_plugin(
            tmp_path, "p",
            b"[plugin]\nname='x'\nrequires=['chat']\nrequires_extras=['rag']\n"
            b"capabilities=['a']\n"))
        assert (spec.requires, spec.requires_extras, spec.capabilities) == (
            ["chat"], ["rag"], ["a"])


_DEEP_JSON = b"[" * 100_000


class TestDeeplyNestedJson:
    """JSON nested past the recursion limit is a corrupt file like any other."""

    def test_install_manifest_load_reads_it_as_absent(self, tmp_path):
        install_manifest.manifest_path(tmp_path).write_bytes(_DEEP_JSON)
        assert install_manifest.load(tmp_path) is None

    def test_data_marker_reads_it_as_absent(self, tmp_path):
        (tmp_path / install_manifest.DATA_MARKER).write_bytes(_DEEP_JSON)
        assert install_manifest.read_marker(tmp_path) is None

    def test_model_meta_cache_reads_it_as_empty(self):
        model_meta._meta_path().parent.mkdir(parents=True, exist_ok=True)
        model_meta._meta_path().write_bytes(_DEEP_JSON)
        assert model_meta._load_all() == {}

    def test_plugin_provenance_marker_reads_it_as_absent(self, tmp_path):
        (tmp_path / engine._PLUGIN_MARKER).write_bytes(_DEEP_JSON)
        assert engine._read_marker(tmp_path) is None

    def test_an_uploaded_workflow_is_rejected_with_value_error(self):
        with pytest.raises(ValueError, match="nested too deeply"):
            media_workflows.save_workflow("image", "w.json", _DEEP_JSON)


def _header_with_alignment(alignment: int) -> bytes:
    kvs = (_lstr("general.alignment") + struct.pack("<II", 4, alignment)
           + _lstr("general.architecture") + struct.pack("<I", 8) + _lstr("clip"))
    return b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + kvs + b"\x00" * 100


class TestGgufAlignment:
    @pytest.mark.parametrize("alignment", [2 ** 20 + 1, 2 ** 24, 2 ** 26])
    def test_a_rewrite_refuses_an_implausible_alignment_without_allocating_for_it(
            self, tmp_path, alignment):
        src, dst = tmp_path / "src.gguf", tmp_path / "dst.gguf"
        src.write_bytes(_header_with_alignment(alignment))
        outcome, peak = _peak(gguf.write_gguf_with_string_kv, src, dst, "k.new", "v")
        assert isinstance(outcome, ValueError)
        assert peak < 4 * 1024 * 1024
        assert not dst.exists()

    @pytest.mark.parametrize("alignment", [32, 64, 4096, 2 ** 20])
    def test_a_plausible_alignment_still_rewrites(self, tmp_path, alignment):
        src, dst = tmp_path / "src.gguf", tmp_path / "dst.gguf"
        src.write_bytes(_header_with_alignment(alignment))
        gguf.write_gguf_with_string_kv(src, dst, "k.new", "v")
        assert b"k.new" in dst.read_bytes()
        assert len(dst.read_bytes()) >= alignment


def _peak(fn, *args):
    return _bounds.peak_allocation(fn, *args)


class TestMcpStdioServer:
    @staticmethod
    def _server():
        return MCPStdioServer({"echo": {"description": "d", "inputSchema": {},
                                        "handler": lambda args: {"content": [],
                                                                 "isError": False}}})

    @pytest.mark.parametrize("params", [[1], "x", 5, {"name": ["echo"]}, {"name": {"a": 1}}])
    def test_a_tools_call_with_unusable_params_is_answered_with_an_error(self, params):
        reply = self._server().handle(
            {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": params})
        assert reply["id"] == 7 and reply["error"]["code"] == -32602

    @pytest.mark.parametrize("line", [
        pytest.param("9" * 5000, id="integer_past_the_conversion_limit"),
        pytest.param("[" * 100_000, id="deeply_nested")])
    def test_a_hostile_line_is_skipped_and_the_next_line_is_served(self, line):
        out = io.StringIO()
        self._server().run_stdio(
            io.StringIO(line + '\n{"jsonrpc": "2.0", "id": 1, "method": "ping"}\n'), out)
        assert json.loads(out.getvalue()) == {"jsonrpc": "2.0", "id": 1, "result": {}}


class TestHostileHuggingFaceModelDirectory:
    """A pulled model directory is the repo owner's files, verbatim."""

    @staticmethod
    def _model_dir(tmp_path, name, body):
        (tmp_path / name).write_bytes(body)
        return tmp_path

    def test_a_deeply_nested_tokenizer_config_declares_no_template(self, tmp_path):
        d = self._model_dir(tmp_path, "tokenizer_config.json", _DEEP_JSON)
        assert capabilities._hf_dir_chat_template(d) is None

    def test_a_deeply_nested_config_declares_no_context_length(self, tmp_path):
        d = self._model_dir(tmp_path, "config.json", _DEEP_JSON)
        assert capabilities._hf_dir_context_length(d) is None

    @pytest.mark.parametrize("body", [
        pytest.param(_DEEP_JSON, id="deeply_nested"), b"5", b'"vision"', b"[1]",
        b'{"architectures": 5}', b'{"architectures": [1, null]}',
        b'{"architectures": {"a": 1}}'])
    def test_a_config_of_the_wrong_shape_is_not_vision(self, tmp_path, body):
        d = self._model_dir(tmp_path, "config.json", body)
        assert registry._hf_is_vision(d) is False

    def test_a_config_naming_a_vision_architecture_is_still_vision(self, tmp_path):
        d = self._model_dir(tmp_path, "config.json",
                            b'{"architectures": ["Qwen2VLForConditionalGeneration", 7]}')
        assert registry._hf_is_vision(d) is True

    @pytest.mark.parametrize("body", [
        pytest.param(_DEEP_JSON, id="deeply_nested"),
        pytest.param(b"9" * 100_000, id="integer_past_the_conversion_limit"),
        pytest.param(b'{"a": ' + b"9" * 5000 + b"}", id="member_past_the_conversion_limit")])
    def test_an_unreadable_tokenizer_json_is_refused_with_runtime_error(self, tmp_path, body):
        d = self._model_dir(tmp_path, "tokenizer.json", body)
        with pytest.raises(RuntimeError, match="cannot be verified safe"):
            hf_tokenizer_safety._extract_unique_patterns(str(d))

    def test_a_deeply_nested_shard_index_is_left_to_transformers(self, tmp_path):
        d = self._model_dir(tmp_path, "model.safetensors.index.json", _DEEP_JSON)
        hf_shard_index_safety.validate_shard_index(str(d))


def _docx(*, encrypted: bool = False, method: int | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", "<w:p><w:r><w:t>hi</w:t></w:r></w:p>")
    data = bytearray(buf.getvalue())
    local, central = data.find(b"PK\x03\x04"), data.find(b"PK\x01\x02")
    if encrypted:
        data[local + 6] |= 1
        data[central + 8] |= 1
    if method is not None:
        struct.pack_into("<H", data, local + 8, method)
        struct.pack_into("<H", data, central + 10, method)
    return bytes(data)


class TestDocxExtraction:
    def test_a_well_formed_docx_still_extracts(self):
        assert extract.extract_bytes(_docx(), "a.docx") == "hi"

    @pytest.mark.parametrize("data", [_docx(encrypted=True), _docx(method=99)],
                             ids=["encrypted_member", "unsupported_compression"])
    def test_a_member_the_zip_reader_refuses_is_an_extract_error(self, data):
        with pytest.raises(extract.ExtractError):
            extract.extract_bytes(data, "a.docx")


def _lstr(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _header_with_lying_array(count: int, *, elem_type: int = 0) -> bytes:
    """A GGUF v3 header whose first KV is a fixed-width array declaring *count*
    elements it does not contain, followed by a second KV."""
    first = _lstr("llama.block_count") + struct.pack("<IIQ", 9, elem_type, count)
    second = _lstr("general.architecture") + struct.pack("<I", 8) + _lstr("llama")
    return b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + first + second


class TestGgufArrayCountPastTheBuffer:
    @pytest.mark.parametrize("count", [2 ** 63, 2 ** 64 - 1, 2 ** 40])
    def test_skip_value_refuses_an_array_longer_than_the_buffer(self, count):
        buf = _header_with_lying_array(count)
        off = 24 + len(_lstr("llama.block_count")) + 4
        with pytest.raises(struct.error):
            gguf._gguf_skip_value(buf, off, gguf._GGUF_TYPE_ARRAY)

    def test_skip_value_still_skips_an_array_that_fits(self):
        buf = struct.pack("<IQ", 0, 4) + b"abcd" + b"tail"
        assert gguf._gguf_skip_value(buf, 0, gguf._GGUF_TYPE_ARRAY) == 16

    @pytest.mark.parametrize("count", [2 ** 63, 2 ** 64 - 1])
    def test_the_readers_report_no_signal_instead_of_raising(self, tmp_path, count):
        path = tmp_path / "lie.gguf"
        path.write_bytes(_header_with_lying_array(count))
        probe = gguf._gguf_capability_probe(path)
        assert probe["complete"] is False
        assert gguf.gguf_capability_metadata(path) is not None
        assert gguf.gguf_architecture(path) is None
        assert gguf.gguf_kv_bytes_per_token(path) == 0
        assert gguf.gguf_expert_count(path) == 0
