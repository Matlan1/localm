# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the capability/scope taxonomy (localm/scopes.py)."""

from localm import scopes


def test_kernel_scopes_documented():
    for s in (scopes.MODELS_READ, scopes.MODELS_WRITE, scopes.CONFIG_READ,
              scopes.CONFIG_WRITE, scopes.PLUGINS_READ, scopes.PLUGINS_ADMIN,
              scopes.KEYS_ADMIN, scopes.ADMIN):
        assert s in scopes.KERNEL_SCOPES


def test_builtin_plugin_scopes_documented():
    for s in (scopes.CHAT, scopes.CODER, scopes.IMAGE, scopes.MUSIC,
              scopes.VIDEO, scopes.RAG, scopes.WEB, scopes.VOICE, scopes.MCP):
        assert s in scopes.BUILTIN_PLUGIN_SCOPES


def test_every_builtin_plugin_scope_is_reserved():
    import tomllib
    from pathlib import Path

    import localm.plugins.builtin as builtin
    root = Path(next(iter(builtin.__path__)))
    manifests = sorted(root.glob("*/plugin.toml"))
    assert manifests, "no built-in plugin manifests found"
    for toml_path in manifests:
        data = tomllib.loads(toml_path.read_text(encoding="utf-8"))["plugin"]
        name = data["name"]
        scope = data.get("scope", name)
        assert scope == name, f"{name}: scope {scope!r} differs from its name"
        assert scope in scopes.BUILTIN_PLUGIN_SCOPES, (
            f"{name}: scope {scope!r} is not reserved in BUILTIN_PLUGIN_SCOPES")


def test_all_scope_strings_unique():
    vals = list(scopes.KERNEL_SCOPES) + list(scopes.BUILTIN_PLUGIN_SCOPES)
    assert len(vals) == len(set(vals))


def test_is_valid_scope():
    assert scopes.is_valid_scope(scopes.CODER)
    assert scopes.is_valid_scope(scopes.MODELS_WRITE)     # colon kernel scope
    assert scopes.is_valid_scope("my_third_party")        # plugin-name scope
    assert scopes.is_valid_scope("x", extra={"x"})        # registered third-party
    assert not scopes.is_valid_scope("")
    assert not scopes.is_valid_scope("bad name")          # space
    assert not scopes.is_valid_scope(None)


def test_a_registered_scope_need_not_be_an_identifier():
    assert scopes.is_valid_scope("vendor.tool:read", extra={"vendor.tool:read"})
    assert not scopes.is_valid_scope("vendor.tool:read", extra={"other"})
    assert not scopes.is_valid_scope("bad name", extra={"x"})


def test_a_dashed_plugin_name_is_a_valid_scope():
    assert scopes.is_valid_scope("my-plugin")
    assert scopes.is_valid_scope("-leading-dash")
    assert not scopes.is_valid_scope("my-plugin!")


def test_a_non_string_scope_is_refused_without_raising():
    assert scopes.is_valid_scope(123) is False
    assert scopes.is_valid_scope(["admin"]) is False
    assert scopes.is_valid_scope(b"chat") is False


def test_privileged_scopes_are_known():
    # Every privileged scope must be a real, known scope: a kernel scope or an
    # EXTRA capability variant like coder:full.
    for s in scopes.PRIVILEGED_SCOPES:
        assert s in scopes.all_known_scopes()


def test_grants_admin_implies_all():
    assert scopes.grants({scopes.ADMIN}, scopes.CODER)
    assert scopes.grants({scopes.CODER}, scopes.CODER)
    assert not scopes.grants({scopes.CHAT}, scopes.CODER)


def test_coder_full_implies_only_the_base_coder_scope():
    assert scopes.grants({scopes.CODER_FULL}, scopes.CODER)
    assert scopes.grants({scopes.CODER_FULL}, scopes.CODER_FULL)
    assert not scopes.grants({scopes.CODER_FULL}, scopes.CHAT)
    assert not scopes.grants({scopes.CODER_FULL}, scopes.MODELS_WRITE)
    assert not scopes.grants({scopes.CODER}, scopes.CODER_FULL)
    assert not scopes.grants(set(), scopes.CODER)


_TAIL = " A key can hold localm's own scopes and those of installed plugins."


def test_unknown_scope_refusal_names_every_unknown_scope_and_the_closest_one():
    assert scopes.unknown_scope_refusal(["chat", "chatt", "not a scope!"]) == (
        "Unknown scope(s): chatt (did you mean 'chat'?), not a scope!." + _TAIL)


def test_unknown_scope_refusal_has_no_hint_without_a_close_known_scope():
    assert scopes.unknown_scope_refusal(["zqxwvy"]) == (
        "Unknown scope(s): zqxwvy." + _TAIL)


def test_unknown_scope_refusal_passes_localm_and_installed_plugin_scopes():
    assert scopes.unknown_scope_refusal(sorted(scopes.all_known_scopes())) is None
    assert scopes.unknown_scope_refusal([]) is None
    assert scopes.unknown_scope_refusal(None) is None
    assert scopes.unknown_scope_refusal(["qzxjvk"], installed=["qzxjvk"]) is None
    assert scopes.unknown_scope_refusal(["qzxjvk"], installed=None) == (
        "Unknown scope(s): qzxjvk." + _TAIL)


def test_normalize():
    assert scopes.normalize([" coder ", "coder", "", "chat", None]) == ["chat", "coder"]
