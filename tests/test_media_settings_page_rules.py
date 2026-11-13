"""The Settings Media page's per-field rules follow the schema.

pages/settings.js hides a media box's ComfyUI-only fields while its backend is
native (MEDIA_COMFY_ONLY_KEYS) and never copies a backend or native field
between boxes (mediaFieldCopyable). Both are decided by a field's block path in
settings_schema.MEDIA_PLUGIN_FIELDS; these tests fail when a field is added or
moved so that the page's key rules no longer match that path.
"""
from __future__ import annotations

import re
from pathlib import Path

from localm.settings_schema import MEDIA_PLUGIN_FIELDS

SETTINGS_JS = (Path(__file__).resolve().parent.parent / "localm" / "plugins" / "gui"
               / "static" / "pages" / "settings.js").read_text(encoding="utf-8")


def _comfy_only_keys_in_js() -> set[str]:
    m = re.search(r"export const MEDIA_COMFY_ONLY_KEYS = new Set\(\[(.*?)\]\);",
                  SETTINGS_JS, re.S)
    assert m, "settings.js no longer defines MEDIA_COMFY_ONLY_KEYS"
    return set(re.findall(r'"([a-z_]+)"', m.group(1)))


def _js_copyable(key: str) -> bool:
    return key != "backend" and not key.startswith("native_")


def test_comfy_only_keys_are_the_fields_stored_under_comfy():
    want = {f.key for f in MEDIA_PLUGIN_FIELDS if f.block_path[0] == "comfy"}
    assert _comfy_only_keys_in_js() == want


def test_copy_from_skips_exactly_the_backend_and_native_fields():
    assert "export function mediaFieldCopyable(key)" in SETTINGS_JS
    assert 'return key !== "backend" && !key.startsWith("native_");' in SETTINGS_JS
    wrong = sorted({f.key for f in MEDIA_PLUGIN_FIELDS
                    if _js_copyable(f.key) == (f.block_path[0] in ("backend", "native"))})
    assert not wrong, f"Copy from would copy or skip the wrong fields: {wrong}"
