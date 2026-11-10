# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where every settings schema field shows up on the GUI Settings page.

The page builds one card per schema ``group``. ``CORE_GROUP_TO_TOP`` in
pages/settings.js puts the card in a nav group and ``CORE_SECTION_HEADING_KEY``
gives it a heading from the catalog. These tests read those maps from the shipped
files and pin the resulting nav group and heading for every real schema field, so
a field moved to another group, a group remapped, or a heading renamed fails here
by key. Hidden fields must be rendered by a named GUI file or be listed as not
editable in the GUI. tests-js/settings-section-placement.test.mjs renders the same
mapping in the page.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from localm.settings_schema import CORE_FIELDS, Widget

STATIC = Path(__file__).resolve().parent.parent / "localm" / "plugins" / "gui" / "static"
SETTINGS_JS = (STATIC / "pages" / "settings.js").read_text(encoding="utf-8")
INDEX_HTML = (STATIC / "index.html").read_text(encoding="utf-8")


def _js_object(name: str) -> dict:
    m = re.search(r"const " + name + r" = \{(.*?)\n\};", SETTINGS_JS, re.S)
    assert m, f"{name} not found in pages/settings.js"
    body = re.sub(r"//[^\n]*", "", m.group(1))
    return {k or q: v for k, q, v in re.findall(
        r'(?:([A-Za-z]\w*)|"([^"]+)")\s*:\s*"([^"]+)"', body)}


def _catalog_en() -> dict:
    src = (STATIC / "app" / "i18n-en.js").read_text(encoding="utf-8")
    return {m.group(1): json.loads(m.group(2)) for m in re.finditer(
        r'^\s*"([^"]+)":\s*("(?:[^"\\]|\\.)*")\s*,?\s*$', src, re.M)}


GROUP_TO_TOP = _js_object("CORE_GROUP_TO_TOP")
HEADING_KEY = _js_object("CORE_SECTION_HEADING_KEY")
EN = _catalog_en()
DE = json.loads((STATIC / "i18n" / "de.json").read_text(encoding="utf-8"))

# Schema group -> (nav group id, English card heading).
GROUP_PLACEMENT = {
    "Engine": ("model", "Runtime & GPU"),
    "Timeouts": ("model", "Timeouts & limits"),
    "Chat": ("model", "Generation defaults"),
    "Models": ("model", "Model Library & Sources"),
    "Embeddings": ("model", "Embeddings"),
    "Server": ("server", "Server"),
    "Network": ("server", "Outbound access"),
    "Security": ("security", "Access"),
    "Plugins": ("plugins", "Plugin management"),
    "Coder": ("plugins", "Coder"),
    "Knowledge": ("plugins", "Knowledge (RAG)"),
    "Voice": ("plugins", "Voice"),
    "Privacy": ("privacy", "Session persistence"),
    "Memory": ("privacy", "Memory"),
    "Diagnostics": ("privacy", "Crash diagnostics"),
    "Updates": ("system", "Updates"),
    "Bug reports": ("system", "Bug reports"),
    "General": ("system", "General"),
    "Desktop": ("system", "Desktop app"),
}

# Every field the generic form renders -> the schema group whose card it sits in.
VISIBLE_FIELD_GROUP = {
    "binary_dir": "Engine", "n_ctx": "Engine", "n_ctx_max": "Engine",
    "n_ctx_grow": "Engine", "ctx_auto": "Engine", "n_gpu_layers": "Engine",
    "ctx_reserve": "Engine", "unload_for_room": "Engine",
    "n_cpu_moe": "Engine", "moe_cache_mib": "Engine", "use_mmap": "Engine",
    "kv_cache_type": "Engine", "n_gpu_layers_auto": "Engine", "mtp_enabled": "Engine",
    "mtp_draft_tokens": "Engine", "spec_source": "Engine", "spec_draft_tokens": "Engine",
    "spec_draft_model": "Engine", "parallel_slots": "Engine", "diffusion_steps": "Engine",
    "diffusion_max_tokens": "Engine", "vram_overhead_mb": "Engine",
    "idle_unload_seconds": "Engine", "hf_trust_remote_code": "Engine",
    "gguf_load_timeout_s": "Timeouts", "gguf_first_token_timeout_s": "Timeouts",
    "hf_load_timeout_s": "Timeouts", "hf_first_token_timeout_s": "Timeouts",
    "hf_embed_timeout_s": "Timeouts", "hf_embed_max_texts": "Timeouts",
    "hf_embed_max_chars": "Timeouts",
    "model_autoswitch": "Models", "import_max_depth": "Models",
    "confirm_remove": "Models", "autoprune_missing_models": "Models",
    "hf_token": "Models", "civitai_api_key": "Models",
    "chat_system_prompt": "Chat", "temperature": "Chat", "top_p": "Chat",
    "top_k": "Chat", "repeat_penalty": "Chat", "max_tokens": "Chat",
    "chat_tool_grammar": "Chat", "chat_web_max_lookups": "Chat", "chat_style_note": "Chat",
    "gui_preview_enabled": "Chat", "gui_preview_owner_only": "Chat",
    "port": "Server", "bind_host": "Server", "tls_enabled": "Server",
    "tls_cert": "Server", "tls_key": "Server", "cors_origins": "Server",
    "allowed_hosts": "Server", "mdns_name": "Server", "metrics_enabled": "Server", "mdns_enabled": "Server",
    "require_auth": "Security", "allow_network_drives": "Security",
    "coder_remember_projects": "Coder", "coder_projects_remembered": "Coder",
    "coder_rail_side": "Coder", "coder_confirm_timeout": "Coder",
    "coder_index_timeout": "Coder", "coder_grep_max_per_file": "Coder",
    "coder_grep_max_output_lines": "Coder", "coder_grep_max_file_bytes": "Coder",
    "coder_tool_grammar": "Coder", "coder_episodic_memory": "Coder",
    "coder_untrusted_provenance": "Coder", "coder_review": "Coder",
    "coder_reviewer": "Coder", "coder_reviewer_model": "Coder",
    "desktop_window_mode": "Desktop", "desktop_window_quit_on_close": "Desktop",
    "temp_location": "General",
    "mode": "Privacy", "chat_mode": "Privacy", "coder_mode": "Privacy",
    "memory_enabled": "Memory", "memory_auto_consolidate": "Memory",
    "memory_recall_in_privacy": "Memory", "memory_recall_in_privacy_chat": "Memory",
    "memory_recall_in_privacy_coder": "Memory",
    "keep_diagnostics": "Diagnostics",
    "embedding_model": "Embeddings", "embedding_pooling": "Embeddings",
    "embedding_gpu_layers": "Embeddings",
    "suggest_plugins": "Plugins", "auto_install_plugin_deps": "Plugins",
    "update_allow_prerelease": "Updates", "update_ignore_net_policy": "Updates",
    "rag_indexing_mode": "Knowledge", "rag_allowed_roots": "Knowledge",
    "rag_denied_roots": "Knowledge", "rag_classify_unknown_files": "Knowledge",
    "rag_rerank": "Knowledge", "rag_rerank_model": "Knowledge",
    "rag_rerank_candidates": "Knowledge", "rag_rerank_min_score": "Knowledge",
    "net_mode": "Network", "net_allow": "Network", "net_deny": "Network",
    "net_allow_private": "Network", "net_allow_model_downloads": "Network",
    "net_search_url": "Network", "gui_proxy_remote_images": "Network",
    "browser_enabled": "Network", "browser_engine": "Network",
    "browser_headless": "Network", "browser_custom_domain_rules": "Network",
    "browser_allow": "Network", "browser_deny": "Network",
    "browser_inline_live_view": "Coder",
    "voice_stt_model": "Voice", "voice_stt_language": "Voice",
}

# Hidden fields -> the GUI file that edits them with its own control, or None when
# the GUI does not edit them (CLI, launcher or a dedicated page owns the value).
HIDDEN_RENDERED_BY = {
    "llama_runtime_pin": None, "llama_runtime_history": None,
    "main_gpu_index": "app/settings-perf.js", "gpu_split_indices": "app/settings-perf.js",
    "gpu_split_ratios": "app/settings-perf.js", "max_resident_models": "app/settings-perf.js",
    "pinned_models": "app/settings-perf.js",
    "user_avatar": "pages/settings.js", "user_name": "pages/settings.js",
    "model_avatar_default": "pages/settings.js", "model_avatar_overrides": "pages/settings.js",
    "logo_style": "app/logo.js", "language": "app/i18n.js",
    "chat_background": "pages/settings.js", "key_presets": "pages/settings.js",
    "plugins_enabled": None, "plugins": None,
    "bugreport_upload_url": None, "bugreport_upload_token": None,
    "update_url": None, "update_token": None,
}

MEDIA_GROUP = "Media"


def _visible_generic():
    return {f.key: f.group for f in CORE_FIELDS
            if f.widget != Widget.HIDDEN and f.group != MEDIA_GROUP}


def test_group_to_top_map_is_total_and_matches_the_pinned_placement():
    groups = {f.group for f in CORE_FIELDS}
    missing = groups - set(GROUP_TO_TOP)
    assert not missing, f"schema groups with no CORE_GROUP_TO_TOP entry: {sorted(missing)}"
    for group, (top, _) in GROUP_PLACEMENT.items():
        assert GROUP_TO_TOP.get(group) == top, (group, GROUP_TO_TOP.get(group), top)
    assert GROUP_TO_TOP.get(MEDIA_GROUP) == "media"


@pytest.mark.parametrize("group", sorted(GROUP_PLACEMENT))
def test_each_group_heading_is_pinned_and_translated(group):
    key = HEADING_KEY.get(group)
    assert key, f"{group} has no CORE_SECTION_HEADING_KEY entry"
    assert EN.get(key) == GROUP_PLACEMENT[group][1], (group, EN.get(key))
    assert DE.get(key), f"{key} is missing from de.json"


def test_every_visible_field_is_pinned_to_its_card():
    actual = _visible_generic()
    assert set(actual) == set(VISIBLE_FIELD_GROUP), (
        "fields rendered but not pinned: " + str(sorted(set(actual) - set(VISIBLE_FIELD_GROUP)))
        + "; pinned but not rendered: " + str(sorted(set(VISIBLE_FIELD_GROUP) - set(actual))))
    moved = {k: (actual[k], g) for k, g in VISIBLE_FIELD_GROUP.items() if actual[k] != g}
    assert not moved, f"fields whose card changed (schema group, pinned group): {moved}"


def test_every_visible_group_has_a_placement():
    for group in set(_visible_generic().values()):
        assert group in GROUP_PLACEMENT, group


def test_hidden_fields_are_rendered_by_their_own_control_or_listed_as_not_editable():
    hidden = {f.key for f in CORE_FIELDS if f.widget == Widget.HIDDEN}
    assert hidden == set(HIDDEN_RENDERED_BY), (
        sorted(hidden ^ set(HIDDEN_RENDERED_BY)))
    for key, rel in HIDDEN_RENDERED_BY.items():
        if rel is None:
            continue
        src = (STATIC / rel).read_text(encoding="utf-8")
        assert re.search(r"\b" + re.escape(key) + r"\b", src), f"{key} not found in {rel}"


def _static_card_headings():
    out = []
    for m in re.finditer(r'<section class="card settings-section" id="([^"]+)" data-group="([^"]+)"'
                         r'[^>]*>(.*?)</section>', INDEX_HTML, re.S):
        h = re.search(r'<h3 data-i18n="([^"]+)"', m.group(3))
        if h:
            out.append((m.group(1), m.group(2), EN.get(h.group(1))))
    return out


def test_updates_fields_render_inside_the_static_updates_card():
    assert 'sec.id === "core-Updates"' in SETTINGS_JS
    assert 'id="update-toggles-block"' in INDEX_HTML


def test_no_two_settings_cards_share_a_heading():
    # The search box shows matching cards from every nav group at once, so a heading
    # must be unique on the whole page. Updates renders into the static Updates card;
    # Bug reports has no visible field and renders no card.
    card_groups = set(_visible_generic().values()) - {"Updates"}
    cards = [("settings-sec-core-" + g, GROUP_TO_TOP[g], EN.get(HEADING_KEY.get(g, "")))
             for g in sorted(card_groups)]
    static = _static_card_headings()
    assert len(static) > 10, "static cards were not found in index.html"
    seen = {}
    for card_id, _top, head in cards + static:
        assert head, f"{card_id} has no English heading"
        assert head not in seen, f"{card_id} and {seen.get(head)} are both headed {head!r}"
        seen[head] = card_id
