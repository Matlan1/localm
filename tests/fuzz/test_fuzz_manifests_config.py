# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fuzz the plugin manifest, config and registry loaders.

``plugin.toml`` comes from a third-party plugin directory; ``config.json``,
``registry.json`` and ``model_meta.json`` are user-editable files that a torn
write or a hand edit can corrupt; settings updates arrive as arbitrary JSON.
The contract: a corrupt file degrades to the documented fallback and a rejected
value raises the documented validation error, never anything else."""
from __future__ import annotations

import json
import itertools

import pytest

pytest.importorskip("hypothesis")

from hypothesis import example, given, strategies as st  # noqa: E402

from localm import config  # noqa: E402
from localm import install_manifest, media_workflows, model_meta  # noqa: E402
from localm import settings_schema  # noqa: E402
from localm.inference import hf_shard_index_safety, hf_tokenizer_safety  # noqa: E402
from localm.model_manager import capabilities, registry  # noqa: E402
from localm.plugins import engine, loader  # noqa: E402
from tests.fuzz import _bounds  # noqa: E402

_counter = itertools.count()

_TOML_VALUES = [
    '"x"', '""', '"a:b"', '"mod:attr"', '"it"', "1", "0", "-1", "3.5", "inf", "-inf",
    "nan", "1e999", "0x10", "99999999999999999999999999999", "true", "false",
    "[]", '["a", "b"]', "[1, 2]", "[[1], [2]]", '["a", 1]', "{}", '{a = 1}',
    "1979-05-27", "07:32:00", "1979-05-27T07:32:00Z", '"\\u0000"', '"' + "x" * 5000 + '"',
    "[" * 200 + "]" * 200, "[" * 5000 + "]" * 5000, "{a={b={c=1}}}",
]
_TOML_KEYS = ["name", "version", "description", "entry", "api_version", "scope",
              "requires_extras", "requires", "capabilities", "data_subdir", "protected",
              "default_enabled", "cli", "register", "exports", "tab_id", "label", "icon",
              "assets_dir", "client_entry", "settings_group", "group", "weird"]


@st.composite
def plugin_toml(draw):
    lines = []
    for table in draw(st.lists(st.sampled_from(["plugin", "surface", "tools", "plugin.x",
                                                "[plugin]", "other"]), max_size=5)):
        header = table if table.startswith("[") else f"[{table}]"
        lines.append(header)
        for _ in range(draw(st.integers(0, 8))):
            lines.append(f"{draw(st.sampled_from(_TOML_KEYS))} = {draw(st.sampled_from(_TOML_VALUES))}")
    if draw(st.booleans()):
        lines.insert(0, "[plugin]")
        lines.insert(1, f'name = {draw(st.sampled_from(["\"good\"", "\"my-plug\"", "\"a b\"", "5", "\"coder\""]))}')
        lines.insert(2, f'entry = {draw(st.sampled_from(["\"m:main\"", "\"m\"", "5", "[]", "\"\""]))}')
    text = "\n".join(lines)
    if draw(st.integers(0, 6)) == 0:
        text += draw(st.text(max_size=40))
    return text


def _plugin_dir(tmp_path, manifest: bytes):
    d = tmp_path / f"p{next(_counter)}"
    d.mkdir()
    (d / "plugin.toml").write_bytes(manifest)
    return d


_manifest_bytes = st.one_of(
    plugin_toml().map(lambda s: s.encode("utf-8")),
    st.binary(max_size=200),
    st.text(max_size=120).map(lambda s: s.encode("utf-8", errors="replace")),
)


@given(manifest=_manifest_bytes)
@example(manifest=b"\xff\xfe[plugin]")
@example(manifest=b'[plugin]\nname="x"\nentry=5\n')
@example(manifest=b"[plugin]\nname='x'\nentry='m:f'\n" + b"a=" + b"[" * 5000 + b"]" * 5000)
def test_parse_manifest_raises_only_plugin_error(manifest, tmp_path):
    d = _plugin_dir(tmp_path, manifest)
    try:
        _bounds.returns_within(loader.parse_manifest, d, warnings=[])
    except loader.PluginError:
        pass


@given(manifest=_manifest_bytes)
@example(manifest=b"\xff\xfe[plugin]")
@example(manifest=b'[plugin]\nname="x"\napi_version=inf\n')
@example(manifest=b'[plugin]\nname="x"\nrequires_extras=5\n')
def test_parse_spec_raises_only_value_error(manifest, tmp_path):
    d = _plugin_dir(tmp_path, manifest)
    try:
        _bounds.returns_within(engine.parse_spec, d, warnings=[])
    except ValueError:
        pass


@given(manifest=_manifest_bytes)
@example(manifest=b"\xff\xfe[plugin]")
def test_discovery_survives_any_manifest(manifest, tmp_path):
    root = tmp_path / f"root{next(_counter)}"
    root.mkdir()
    d = root / "plug"
    d.mkdir()
    (d / "plugin.toml").write_bytes(manifest)
    _bounds.returns_within(loader.discover_plugins, root)
    _bounds.returns_within(loader.discover_errors, root)
    _bounds.returns_within(loader.discover_warnings, root)


_json_leaf = st.none() | st.booleans() | st.integers() | st.floats() | st.text(max_size=12)
_json_value = st.recursive(
    _json_leaf,
    lambda inner: st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=8), inner,
                                                                  max_size=4),
    max_leaves=20)

_json_file_bytes = st.one_of(
    _json_value.map(lambda v: json.dumps(v).encode("utf-8")),
    st.binary(max_size=120),
    st.just(b"[" * 100000),
    st.just(b'{"a":' * 100000),
    st.just(b"9" * 100000),
    st.just(b"\xff\xfe{}"),
)


@given(blob=_json_file_bytes)
def test_load_config_never_raises(blob):
    config.ensure_dirs()
    config.CONFIG_FILE.write_bytes(blob)
    out = _bounds.returns_within(config.load_config)
    assert isinstance(out, dict)


@given(blob=_json_file_bytes)
def test_load_registry_never_raises(blob):
    config.ensure_dirs()
    config.REGISTRY_FILE.write_bytes(blob)
    out = _bounds.returns_within(config.load_registry)
    assert isinstance(out, dict)


@given(blob=_json_file_bytes)
def test_model_meta_cache_never_raises(blob):
    config.ensure_dirs()
    model_meta._meta_path().write_bytes(blob)
    assert isinstance(_bounds.returns_within(model_meta._load_all), dict)
    assert model_meta.cached_n_layers("/no/such/model.gguf") is None


@given(blob=_json_file_bytes)
def test_install_records_never_raise(blob, tmp_path):
    root = tmp_path / f"i{next(_counter)}"
    root.mkdir()
    install_manifest.manifest_path(root).write_bytes(blob)
    (root / install_manifest.DATA_MARKER).write_bytes(blob)
    out = _bounds.returns_within(install_manifest.load, root)
    assert out is None or isinstance(out, dict)
    marker = _bounds.returns_within(install_manifest.read_marker, root)
    assert marker is None or isinstance(marker, dict)


@given(blob=_json_file_bytes | _manifest_bytes)
def test_uploaded_workflow_is_stored_or_rejected_with_value_error(blob):
    try:
        _bounds.returns_within(media_workflows.save_workflow, "image", "fuzz.json", blob)
    except ValueError:
        pass


@given(blob=_json_file_bytes | st.just(b"5") | st.just(b'{"architectures": [1, null]}')
       | st.just(b'{"architectures": 5}') | st.just(b'{"chat_template": [5, {"template": 1}]}'))
def test_hf_model_directory_readers_never_raise_unexpectedly(blob, tmp_path):
    d = tmp_path / f"hf{next(_counter)}"
    d.mkdir()
    for name in ("config.json", "tokenizer_config.json", "preprocessor_config.json",
                 "tokenizer.json", "model.safetensors.index.json"):
        (d / name).write_bytes(blob)
    _bounds.returns_within(capabilities._hf_dir_chat_template, d)
    _bounds.returns_within(capabilities._hf_dir_context_length, d)
    assert isinstance(_bounds.returns_within(registry._hf_is_vision, d), bool)
    try:
        _bounds.returns_within(hf_shard_index_safety.validate_shard_index, str(d))
    except RuntimeError:
        pass
    try:
        _bounds.returns_within(hf_tokenizer_safety._extract_unique_patterns, str(d))
    except RuntimeError:
        pass


_update_values = st.one_of(
    _json_value,
    st.sampled_from(["", "  ", "0", "-1", "1e999", "nan", "inf", "true", "false", "on",
                     "off", "x" * 10000, "9" * 5000, "\x00", "../../etc/passwd"]),
    st.integers(min_value=-(2 ** 70), max_value=2 ** 70),
)


@given(key=st.sampled_from(sorted(config.DEFAULT_CONFIG)), value=_update_values)
@example(key="port", value="9" * 5000)
def test_validate_update_raises_only_value_error(key, value):
    try:
        out = _bounds.returns_within(settings_schema.validate_update, {key: value})
    except ValueError:
        return
    assert set(out) == {key}


@given(name=st.sampled_from(sorted(settings_schema.MEDIA_PLUGINS)),
       data=st.data())
def test_validate_media_block_raises_only_value_error(name, data):
    keys = [f.key for f in settings_schema.media_fields_for(name)] + ["bogus"]
    updates = {data.draw(st.sampled_from(keys)): data.draw(_update_values)
               for _ in range(data.draw(st.integers(0, 3)))}
    try:
        _bounds.returns_within(settings_schema.validate_media_block, name, updates)
    except ValueError:
        pass
