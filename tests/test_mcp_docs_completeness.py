# SPDX-License-Identifier: AGPL-3.0-or-later
"""docs/mcp.md's "Exposed tools" table claims to be the full list, and its
Options block claims to show every CLI flag. Bind both claims to the real
build_tools() output and the real localm.plugins.mcpserver.cli.main Click
command, so a tool or flag added without a matching doc row fails here.

Reads the REAL docs/mcp.md, not a fixture string.
"""

import re
from pathlib import Path

from localm.plugins.mcpserver.cli import main as mcp_cli
from localm.plugins.mcpserver.server import EngineCache, build_tools

_DOC = Path(__file__).resolve().parents[1] / "docs" / "mcp.md"


def _doc_text() -> str:
    return _DOC.read_text(encoding="utf-8")


def _all_tools(monkeypatch) -> dict:
    """Every tool build_tools() can EVER return, by forcing every conditional
    gate on. Mirrors tests/test_mcp_memory_tools.py's monkeypatch of
    _memory_available and tests/test_config_admin_gating.py's build_tools
    call shape."""
    import localm.plugins.mcpserver.server as srv
    monkeypatch.setattr(srv, "_backend_can_embed", lambda *a, **k: True)
    monkeypatch.setattr(srv, "_coder_available", lambda: True)
    monkeypatch.setattr(srv, "_memory_available", lambda: True)
    engines = EngineCache(default_model=None, engine_factory=lambda *a, **k: None)
    return build_tools(engines, enable_images=True, enable_coder=True,
                       enable_memory=True, enable_memory_write=True)


def _documents_tool(doc: str, name: str) -> bool:
    """True if `doc` shows `name` as a table row's tool name, i.e. inside
    backtick-code immediately after a leading `|`, not merely as a substring
    (which `memory` would match inside `memory_recall` and vice versa)."""
    return re.search(r"\|\s*`" + re.escape(name) + r"`\s*\|", doc) is not None


def test_matcher_does_not_accept_an_absent_tool():
    assert not _documents_tool(_doc_text(), "totally_not_a_real_mcp_tool")


def test_every_buildable_tool_is_documented_in_exposed_tools_table(monkeypatch):
    doc = _doc_text()
    tools = _all_tools(monkeypatch)
    missing = sorted(name for name in tools if not _documents_tool(doc, name))
    assert not missing, f"tools missing from docs/mcp.md's Exposed tools table: {missing}"


def _long_options(command) -> set:
    """Every `--long-option` flag the command actually accepts, skipping
    Click's built-in --help."""
    names = set()
    for param in command.params:
        for opt in param.opts + list(getattr(param, "secondary_opts", [])):
            if opt.startswith("--") and opt != "--help":
                names.add(opt)
    return names


def test_matcher_does_not_accept_an_absent_option():
    assert "--totally-not-a-real-flag" not in _doc_text()


def test_every_mcp_cli_option_is_documented(monkeypatch):
    doc = _doc_text()
    missing = sorted(opt for opt in _long_options(mcp_cli) if opt not in doc)
    assert not missing, f"options missing from docs/mcp.md: {missing}"
