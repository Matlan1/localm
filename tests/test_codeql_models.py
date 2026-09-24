# SPDX-License-Identifier: AGPL-3.0-or-later
"""The CodeQL models-as-data rows in .github/codeql/extensions/ name real localm
functions, and the facts each barrier declares hold."""

from __future__ import annotations

import importlib
import inspect
import itertools
import json
import ntpath
import posixpath
from pathlib import Path

import pytest

MODEL_FILE = (Path(__file__).resolve().parents[1] / ".github" / "codeql" / "extensions"
              / "localm-python-models" / "models" / "localm.model.yml")


def _rows(extensible: str) -> list[list[str]]:
    """The data rows listed under *extensible*; each row is one JSON list."""
    rows, current = [], None
    for line in MODEL_FILE.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("extensible:"):
            current = s.split(":", 1)[1].strip()
        elif s.startswith("- [") and current == extensible:
            rows.append(json.loads(s[2:]))
    return rows


def _resolve(type_: str, path: str):
    """``(object, remaining access-path tokens)`` for a row's type and path."""
    obj = importlib.import_module(type_)
    tokens = path.split(".")
    while tokens and tokens[0].startswith("Member["):
        name = tokens.pop(0)[len("Member["):-1]
        if hasattr(obj, "__path__") and not hasattr(obj, name):
            obj = importlib.import_module(f"{obj.__name__}.{name}")
        else:
            obj = getattr(obj, name)
    return obj, tokens


def _first_param(fn) -> str:
    return next(iter(inspect.signature(fn).parameters))


def test_model_file_has_rows():
    assert _rows("barrierModel") and _rows("barrierGuardModel")


@pytest.mark.parametrize("row", _rows("barrierModel"), ids=lambda r: r[1])
def test_barrier_row_names_a_real_function(row):
    type_, path, kind = row
    assert kind == "path-injection"
    fn, rest = _resolve(type_, path)
    assert callable(fn), path
    if rest == ["ReturnValue"]:
        return
    arg, param = rest
    assert param == "Parameter[0]", path
    position, keyword = arg[len("Argument["):-1].split(",")
    assert position == "0" and _first_param(fn) == keyword.rstrip(":"), path


@pytest.mark.parametrize("row", _rows("barrierGuardModel"), ids=lambda r: r[1])
def test_barrier_guard_row_names_a_real_function(row):
    type_, path, accepting, kind = row
    assert (accepting, kind) == ("true", "path-injection")
    fn, rest = _resolve(type_, path)
    assert callable(fn) and len(rest) == 1, path
    position, keyword = rest[0][len("Argument["):-1].split(",")
    assert position == "0" and _first_param(fn) == keyword.rstrip(":"), path


def test_resolver_rejects_a_missing_function():
    with pytest.raises(AttributeError):
        _resolve("localm", "Member[config].Member[no_such_accessor].ReturnValue")


def test_registry_and_installed_plugins_share_the_data_dir():
    """registry.json and the installed-plugins root, whose modules localm
    imports and runs, are both directly under the data directory."""
    import localm.config as cfg
    from localm.plugins.loader import plugins_dir
    assert Path(cfg.REGISTRY_FILE).parent == plugins_dir().parent


_ID_ALPHABET = ("a", "-", "_", ".", "/", "\\", ":", " ")


def _candidate_ids():
    for n in range(1, 4):
        for chars in itertools.product(_ID_ALPHABET, repeat=n):
            yield "".join(chars)
    yield from ("chat", "my-plugin", "..", "../x", "..\\x", "Q:x", "Q:\\x",
                "\\\\host\\share", "//host/share", "/root", "con", "a.b")


def test_accepted_plugin_id_is_one_component_under_any_root():
    """Every id _is_valid_plugin_name accepts, joined under a Windows or POSIX
    root and normalised, is exactly one direct child with the same name."""
    from localm.plugins.ids import _is_valid_plugin_name
    accepted = [n for n in _candidate_ids() if _is_valid_plugin_name(n)]
    assert {"a", "a-a", "_a", "chat", "my-plugin"} <= set(accepted)
    for name in accepted:
        for mod, root in ((ntpath, "Q:\\root"), (posixpath, "/root")):
            joined = mod.normpath(mod.join(root, name))
            assert (mod.dirname(joined), mod.basename(joined)) == (root, name), (name, root)
