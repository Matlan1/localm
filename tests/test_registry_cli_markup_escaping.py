# SPDX-License-Identifier: AGPL-3.0-or-later
"""A model name, registry key, filesystem path, exception message, or
Ollama-manifest digest shown by `localm/model_manager/registry.py`'s
console.print()/Table calls must survive verbatim - Rich's Console.print()
parses "[...]" in ANY interpolated string as markup, not just inside a
call's own literal [style] tags, and Table.add_row() cells parse markup the
same way. Rich renders these as:

    Console().print('report[draft].txt')       -> prints "report.txt"
    Console().print('notes[bold red].md')      -> prints "notes.md"

The bracketed span is either dropped outright (an unrecognised tag) or
consumed as a (bogus) style directive (a recognised one), in both cases
silently.

registry.py backs several localm/cli/models.py commands one layer down, and
several of its functions (alias_model, set_model_type, relocate_model) are
called DIRECTLY IN-PROCESS from GUI HTTP routes with raw request-body values
(localm/plugins/gui/routes/models.py) - no subprocess boundary at all.
remove_model is additionally reachable via a spawned `localm rm <model>
--yes` subprocess whose stdout is re-pushed into the GUI job/activity log
verbatim (POST /api/models/remove).

These tests call registry.py's functions directly (not through CliRunner):
none of them need a running server or a real HuggingFace API response, and a
direct call exercises the exact code under test without click/CliRunner
overhead.

Two bracket shapes are used throughout, matching the reference files' own
constants and the two distinct Rich failure modes: BRACKET_DROP (an
unrecognised tag, silently dropped) and BRACKET_STYLE (a real style name,
silently consumed as styling).

Several escape() calls in registry.py wrap a value that is PROVABLY
restricted to a safe charset at that exact call site (a name that has just
passed through `_sanitize_name()`, or a `new_type` that has just passed the
MODEL_TYPES membership check) - defense-in-depth, matching this sweep's own
precedent for rag.py's collection names. A bracket can never reach those
specific interpolations, so no test below claims to fires-control them; the
surrounding non-bracketed value in the same message is still covered.
"""

from __future__ import annotations

import sys

import pytest

from localm import model_manager as mm

BRACKET_DROP = "alpha[draft]beta"
BRACKET_STYLE = "alpha[bold red]beta"


@pytest.fixture()
def fake_registry(tmp_path, monkeypatch):
    """In-memory registry + temp MODELS_DIR wired into model_manager, mirroring
    test_model_rename.py's fixture of the same name (registry.py reads every
    one of these through `_mm.<name>`, so patching the package attribute is
    picked up at call time regardless of which registry.py function runs)."""
    store: dict = {}
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models_dir)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "load_registry", lambda: dict(store))

    def _save(reg):
        store.clear()
        store.update(reg)
    monkeypatch.setattr(mm, "save_registry", _save)

    def _update(mutator):
        reg = dict(store)
        mutator(reg)
        store.clear()
        store.update(reg)
        return dict(store)
    monkeypatch.setattr(mm, "update_registry", _update)
    return store, models_dir


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    # Console() reads COLUMNS dynamically; without it CliRunner-less capsys
    # capture defaults to 80 columns and hard-wraps the longer paths/messages
    # used below mid-word, breaking a plain substring assertion. Same fixture
    # note as test_rag_cli_markup_escaping.py / test_models_cli_markup_escaping.py.
    monkeypatch.setenv("COLUMNS", "300")


_GGUF_BYTES = b"GGUF" + b"\x00" * 1024   # real magic + past _GGUF_MIN_BYTES (1024)


def _mkfile(models_dir, name: str, content: bytes = _GGUF_BYTES) -> "object":
    p = models_dir / name
    p.write_bytes(content)
    return p


# ------------------------------------------------------------------ #
#  list_models                                                        #
# ------------------------------------------------------------------ #

class TestListModelsMarkupEscaping:
    def test_corrupt_entry_name_survives_verbatim(self, fake_registry, capsys):
        store, _ = fake_registry
        store[BRACKET_DROP] = "not-a-dict"          # malformed -> _entry_path is None
        mm.list_models()
        out = capsys.readouterr().out
        assert BRACKET_DROP in out, f"a corrupt entry's registry KEY must survive verbatim: {out!r}"

    def test_present_entry_name_survives_verbatim(self, fake_registry, capsys):
        store, models_dir = fake_registry
        _mkfile(models_dir, "m.gguf")
        store[f"model-{BRACKET_STYLE}"] = {"path": str(models_dir / "m.gguf")}
        mm.list_models()
        out = capsys.readouterr().out
        assert f"model-{BRACKET_STYLE}" in out, (
            f"a registry key that looks like a style tag must be shown as "
            f"literal text: {out!r}")

    def test_missing_entry_name_survives_verbatim(self, fake_registry, capsys):
        store, models_dir = fake_registry
        store[f"gone-{BRACKET_DROP}"] = {"path": str(models_dir / "does-not-exist.gguf")}
        mm.list_models()
        out = capsys.readouterr().out
        assert f"gone-{BRACKET_DROP}" in out, (
            f"a missing-file entry's name must survive verbatim: {out!r}")

    def test_source_role_and_path_survive_verbatim(self, fake_registry, capsys):
        store, models_dir = fake_registry
        gguf = _mkfile(models_dir, f"path-with-{BRACKET_STYLE}.gguf")
        store["m"] = {
            "path": str(gguf),
            # source/model_type: a hand-edited registry.json is never
            # re-validated against a known set at display time.
            "source": f"src-{BRACKET_DROP}",
            "model_type": f"type-{BRACKET_STYLE}",
        }
        mm.list_models()
        out = capsys.readouterr().out
        assert f"src-{BRACKET_DROP}" in out, f"source must survive verbatim: {out!r}"
        assert f"type-{BRACKET_STYLE}" in out, f"model_type must survive verbatim: {out!r}"
        assert str(gguf) in out, f"the path column must survive verbatim: {out!r}"


# ------------------------------------------------------------------ #
#  relocate_model / relocate_target                                   #
# ------------------------------------------------------------------ #

class TestRelocateModelMarkupEscaping:
    def test_unknown_name_shows_bracketed_name_verbatim(self, fake_registry, capsys):
        ok = mm.relocate_model(BRACKET_DROP, "/whatever")
        assert ok is False
        out = capsys.readouterr().out
        assert BRACKET_DROP in out, f"the unknown name must survive verbatim: {out!r}"

    def test_invalid_new_path_reason_survives_verbatim(self, fake_registry, tmp_path, capsys):
        store, models_dir = fake_registry
        gguf = _mkfile(models_dir, "m.gguf")
        store["m"] = {"path": str(gguf)}
        # relocate_target's "Not a GGUF model file: {p}" reason is built from
        # the caller's own new_path - a bracket in the TARGET path (not one
        # localm chose) reaches relocate_model's own console.print via `reason`.
        bad = tmp_path / f"notgguf{BRACKET_STYLE}.txt"
        bad.write_text("nope")
        ok = mm.relocate_model("m", str(bad))
        assert ok is False
        out = capsys.readouterr().out
        assert f"notgguf{BRACKET_STYLE}.txt" in out, (
            f"relocate_target's reason must survive verbatim: {out!r}")

    def test_corrupt_entry_shows_bracketed_name_verbatim(self, fake_registry, tmp_path, capsys):
        store, models_dir = fake_registry
        gguf = _mkfile(models_dir, "new.gguf")
        store[f"corrupt-{BRACKET_DROP}"] = "not-a-dict"
        ok = mm.relocate_model(f"corrupt-{BRACKET_DROP}", str(gguf))
        assert ok is False
        out = capsys.readouterr().out
        assert f"corrupt-{BRACKET_DROP}" in out, f"the name must survive verbatim: {out!r}"

    def test_success_shows_bracketed_name_and_path_verbatim(self, fake_registry, tmp_path, capsys):
        store, models_dir = fake_registry
        old = _mkfile(models_dir, "old.gguf")
        store[f"m-{BRACKET_STYLE}"] = {"path": str(old), "missing": True}
        new = tmp_path / f"moved-{BRACKET_DROP}.gguf"
        new.write_bytes(old.read_bytes())
        ok = mm.relocate_model(f"m-{BRACKET_STYLE}", str(new))
        assert ok is True
        out = capsys.readouterr().out
        assert f"m-{BRACKET_STYLE}" in out, f"the name must survive verbatim: {out!r}"
        assert f"moved-{BRACKET_DROP}.gguf" in out, (
            f"the resolved target path must survive verbatim: {out!r}")


# ------------------------------------------------------------------ #
#  set_model_type                                                     #
# ------------------------------------------------------------------ #

class TestSetModelTypeMarkupEscaping:
    def test_unknown_name_shows_bracketed_name_verbatim(self, fake_registry, capsys):
        ok = mm.set_model_type(BRACKET_STYLE, "llm")
        assert ok is False
        out = capsys.readouterr().out
        assert BRACKET_STYLE in out, f"the unknown name must survive verbatim: {out!r}"

    def test_invalid_type_shows_bracketed_value_verbatim(self, fake_registry, capsys):
        # new_type is not passed through !r, which does NOT protect against Rich
        # markup: repr() escapes quotes and backslashes only, and "[" and "]"
        # pass through.
        store, models_dir = fake_registry
        store["m"] = {"path": str(_mkfile(models_dir, "m.gguf"))}
        ok = mm.set_model_type("m", BRACKET_DROP)
        assert ok is False
        out = capsys.readouterr().out
        assert BRACKET_DROP in out, f"the invalid type must survive verbatim: {out!r}"

    def test_invalid_type_bracket_style_survives_verbatim(self, fake_registry, capsys):
        store, models_dir = fake_registry
        store["m"] = {"path": str(_mkfile(models_dir, "m.gguf"))}
        ok = mm.set_model_type("m", BRACKET_STYLE)
        assert ok is False
        out = capsys.readouterr().out
        assert BRACKET_STYLE in out, (
            f"an invalid type that looks like a style tag must be shown as "
            f"literal text, not consumed as styling: {out!r}")

    def test_success_shows_bracketed_name_verbatim(self, fake_registry, capsys):
        store, models_dir = fake_registry
        store[f"m-{BRACKET_DROP}"] = {"path": str(_mkfile(models_dir, "m.gguf"))}
        ok = mm.set_model_type(f"m-{BRACKET_DROP}", "lora")
        assert ok is True
        out = capsys.readouterr().out
        assert f"m-{BRACKET_DROP}" in out, f"the name must survive verbatim: {out!r}"


# ------------------------------------------------------------------ #
#  alias_model                                                        #
# ------------------------------------------------------------------ #

class TestAliasModelMarkupEscaping:
    def test_unknown_existing_shows_bracketed_name_verbatim(self, fake_registry, capsys):
        ok = mm.alias_model(BRACKET_DROP, "newname")
        assert ok is False
        out = capsys.readouterr().out
        assert BRACKET_DROP in out, f"the unknown 'existing' name must survive verbatim: {out!r}"

    def test_success_shows_bracketed_existing_name_verbatim(self, fake_registry, capsys):
        store, models_dir = fake_registry
        store[f"real-{BRACKET_STYLE}"] = {"path": str(_mkfile(models_dir, "m.gguf"))}
        ok = mm.alias_model(f"real-{BRACKET_STYLE}", "alias1")
        assert ok is True
        out = capsys.readouterr().out
        assert f"real-{BRACKET_STYLE}" in out, (
            f"the existing model's name must survive verbatim: {out!r}")


# ------------------------------------------------------------------ #
#  rename_model / rename_model_with_notes                             #
# ------------------------------------------------------------------ #

class TestRenameModelMarkupEscaping:
    def test_unknown_old_name_shows_bracketed_name_verbatim(self, fake_registry, capsys):
        ok, notes = mm.rename_model_with_notes(BRACKET_STYLE, "newname")
        assert ok is False
        out = capsys.readouterr().out
        assert BRACKET_STYLE in out, f"the unknown old_name must survive verbatim: {out!r}"

    def test_self_rename_noop_shows_bracketed_name_verbatim(self, fake_registry, capsys):
        store, models_dir = fake_registry
        store[f"m-{BRACKET_DROP}"] = {"path": str(_mkfile(models_dir, "m.gguf"))}
        ok, notes = mm.rename_model_with_notes(f"m-{BRACKET_DROP}", f"m-{BRACKET_DROP}")
        assert ok is True
        out = capsys.readouterr().out
        assert f"m-{BRACKET_DROP}" in out, f"the name must survive verbatim: {out!r}"

    def test_success_shows_bracketed_old_name_verbatim(self, fake_registry, capsys):
        store, models_dir = fake_registry
        store[f"old-{BRACKET_STYLE}"] = {"path": str(_mkfile(models_dir, "m.gguf"))}
        ok, notes = mm.rename_model_with_notes(f"old-{BRACKET_STYLE}", "newname")
        assert ok is True
        out = capsys.readouterr().out
        assert f"old-{BRACKET_STYLE}" in out, f"old_name must survive verbatim: {out!r}"

    def test_migration_note_exception_text_survives_verbatim(
            self, fake_registry, monkeypatch, capsys):
        store, models_dir = fake_registry
        store["old"] = {"path": str(_mkfile(models_dir, "m.gguf"))}

        def _raise(*a, **k):
            raise RuntimeError(f"config write failed {BRACKET_DROP}")
        # _migrate_model_references calls update_config directly (a module-
        # level name in registry.py, not routed through `_mm.`).
        monkeypatch.setattr("localm.model_manager.registry.update_config", _raise)
        ok, notes = mm.rename_model_with_notes("old", "new")
        assert ok is True
        out = capsys.readouterr().out
        assert f"config write failed {BRACKET_DROP}" in out, (
            f"a migration-failure note carrying an exception message must "
            f"survive verbatim: {out!r}")


# ------------------------------------------------------------------ #
#  remove_model                                                       #
# ------------------------------------------------------------------ #

class TestRemoveModelMarkupEscaping:
    def test_unknown_name_shows_bracketed_name_verbatim(self, fake_registry, capsys):
        mm.remove_model(BRACKET_STYLE)
        out = capsys.readouterr().out
        assert BRACKET_STYLE in out, f"the unknown name must survive verbatim: {out!r}"

    def test_corrupt_entry_shows_bracketed_name_verbatim(self, fake_registry, capsys):
        store, _ = fake_registry
        store[f"corrupt-{BRACKET_DROP}"] = "not-a-dict"
        mm.remove_model(f"corrupt-{BRACKET_DROP}")
        out = capsys.readouterr().out
        assert f"corrupt-{BRACKET_DROP}" in out, f"the name must survive verbatim: {out!r}"

    def test_kept_with_other_aliases_shows_bracketed_names_verbatim(self, fake_registry, capsys):
        store, models_dir = fake_registry
        gguf = _mkfile(models_dir, "shared.gguf")
        store[f"a-{BRACKET_DROP}"] = {"path": str(gguf)}
        store[f"b-{BRACKET_STYLE}"] = {"path": str(gguf)}
        mm.remove_model(f"a-{BRACKET_DROP}")
        out = capsys.readouterr().out
        assert f"a-{BRACKET_DROP}" in out, f"the removed name must survive verbatim: {out!r}"
        assert f"b-{BRACKET_STYLE}" in out, (
            f"the surviving sibling alias must survive verbatim: {out!r}")

    def test_deleted_file_message_shows_bracketed_path_verbatim(self, fake_registry, capsys):
        store, models_dir = fake_registry
        gguf = _mkfile(models_dir, f"del-{BRACKET_DROP}.gguf")
        store["m"] = {"path": str(gguf)}
        mm.remove_model("m")
        out = capsys.readouterr().out
        assert f"del-{BRACKET_DROP}.gguf" in out, (
            f"the deleted file's path must survive verbatim: {out!r}")

    def test_final_removed_message_shows_bracketed_name_verbatim(self, fake_registry, capsys):
        store, models_dir = fake_registry
        store[f"m-{BRACKET_STYLE}"] = {"path": str(_mkfile(models_dir, "m.gguf"))}
        mm.remove_model(f"m-{BRACKET_STYLE}")
        out = capsys.readouterr().out
        assert f"m-{BRACKET_STYLE}" in out, f"the name must survive verbatim: {out!r}"


# ------------------------------------------------------------------ #
#  _store_into_models_dir                                             #
# ------------------------------------------------------------------ #

class TestStoreIntoModelsDirMarkupEscaping:
    def test_dir_copy_progress_shows_bracketed_paths_verbatim(
            self, fake_registry, tmp_path, capsys):
        _, models_dir = fake_registry
        src_dir = tmp_path / f"srcdir-{BRACKET_DROP}"
        src_dir.mkdir()
        (src_dir / "x.bin").write_bytes(b"x")
        mm._store_into_models_dir(src_dir, "copy")
        out = capsys.readouterr().out
        assert f"srcdir-{BRACKET_DROP}" in out, (
            f"the source directory path must survive verbatim: {out!r}")

    def test_file_move_progress_shows_bracketed_filename_verbatim(
            self, fake_registry, tmp_path, capsys):
        _, models_dir = fake_registry
        src = tmp_path / f"file-{BRACKET_STYLE}.gguf"
        src.write_bytes(b"x")
        mm._store_into_models_dir(src, "move")
        out = capsys.readouterr().out
        assert f"file-{BRACKET_STYLE}.gguf" in out, (
            f"the filename must survive verbatim, not consumed as styling: {out!r}")


# ------------------------------------------------------------------ #
#  _store_loose_gguf_dir                                               #
# ------------------------------------------------------------------ #

class TestStoreLooseGgufDirMarkupEscaping:
    def test_runtime_error_shows_bracketed_path_verbatim(self, fake_registry, tmp_path, capsys):
        _, models_dir = fake_registry
        # A name collision forces _store_into_models_dir to raise, whose
        # message this function re-prints - the same RuntimeError-catch
        # shape as add_local's and _register_with_dedup's own copy/move
        # paths (already independently fires-controlled), exercised here
        # for _store_loose_gguf_dir's own escape() call specifically.
        conflict = models_dir / f"m-{BRACKET_STYLE}.gguf"
        conflict.write_bytes(b"different content")
        external = tmp_path / "ext" / f"m-{BRACKET_STYLE}.gguf"
        external.parent.mkdir()
        external.write_bytes(_GGUF_BYTES)
        result = mm._store_loose_gguf_dir([external], "copy")
        assert result is None
        out = capsys.readouterr().out
        assert f"m-{BRACKET_STYLE}.gguf" in out, (
            f"the conflicting destination path must survive verbatim: {out!r}")


# ------------------------------------------------------------------ #
#  _register_with_dedup                                               #
# ------------------------------------------------------------------ #

class TestRegisterWithDedupMarkupEscaping:
    def test_already_registered_same_file_shows_bracketed_name_and_path_verbatim(
            self, fake_registry, capsys):
        store, models_dir = fake_registry
        gguf = _mkfile(models_dir, f"same-{BRACKET_DROP}.gguf")
        store[f"m-{BRACKET_STYLE}"] = {"path": str(gguf)}
        mm._register_with_dedup(f"m-{BRACKET_STYLE}", gguf, "local")
        out = capsys.readouterr().out
        assert f"m-{BRACKET_STYLE}" in out, f"model_name must survive verbatim: {out!r}"
        assert f"same-{BRACKET_DROP}.gguf" in out, f"the path must survive verbatim: {out!r}"

    def test_also_registered_as_shows_bracketed_alias_verbatim(self, fake_registry, capsys):
        store, models_dir = fake_registry
        gguf = _mkfile(models_dir, "shared.gguf")
        # model_name itself must ALREADY be a registered alias of this file
        # for the code to reach the "Also registered as" join at all - it
        # lives inside the `model_name in aliases` branch (the true no-op
        # case), not the dedup-prompt path a bare fresh model_name reaches.
        store["primary"] = {"path": str(gguf)}
        store[f"other-{BRACKET_STYLE}"] = {"path": str(gguf)}
        mm._register_with_dedup("primary", gguf, "local")
        out = capsys.readouterr().out
        assert f"other-{BRACKET_STYLE}" in out, (
            f"the sibling alias name must survive verbatim: {out!r}")

    def test_name_collision_shows_bracketed_name_and_old_path_verbatim(
            self, fake_registry, tmp_path, capsys):
        store, models_dir = fake_registry
        old = _mkfile(models_dir, f"old-{BRACKET_DROP}.gguf")
        store[f"m-{BRACKET_STYLE}"] = {"path": str(old)}
        new = tmp_path / "different.gguf"
        new.write_bytes(b"y")
        mm._register_with_dedup(f"m-{BRACKET_STYLE}", new, "local", on_duplicate="skip")
        out = capsys.readouterr().out
        assert f"m-{BRACKET_STYLE}" in out, f"model_name must survive verbatim: {out!r}"
        assert f"old-{BRACKET_DROP}.gguf" in out, (
            f"the conflicting old path must survive verbatim: {out!r}")

    def test_register_success_shows_bracketed_name_verbatim(self, fake_registry, capsys):
        _, models_dir = fake_registry
        gguf = _mkfile(models_dir, "fresh.gguf")
        mm._register_with_dedup(f"m-{BRACKET_DROP}", gguf, "local")
        out = capsys.readouterr().out
        assert f"m-{BRACKET_DROP}" in out, f"model_name must survive verbatim: {out!r}"


# ------------------------------------------------------------------ #
#  _resolve_ollama_manifest                                           #
# ------------------------------------------------------------------ #

class TestResolveOllamaManifestMarkupEscaping:
    def _manifest_dir(self, tmp_path, digest: str):
        import json
        manifest_dir = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "m" / "latest"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "tagfile").write_text(json.dumps({
            "layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": digest}],
        }))
        return manifest_dir

    def test_malformed_digest_survives_verbatim(self, tmp_path, capsys):
        # digest is REMOTE-authored (the whole point of this branch is that it
        # failed the safe-charset check), so it must be escaped.
        digest = f"sha256:not-hex-{BRACKET_DROP}"
        manifest_dir = self._manifest_dir(tmp_path, digest)
        result = mm._resolve_ollama_manifest(manifest_dir)
        assert result is None
        out = capsys.readouterr().out
        assert BRACKET_DROP in out, f"the malformed digest must survive verbatim: {out!r}"

    def test_malformed_digest_bracket_style_survives_verbatim(self, tmp_path, capsys):
        digest = f"sha256:not-hex-{BRACKET_STYLE}"
        manifest_dir = self._manifest_dir(tmp_path, digest)
        result = mm._resolve_ollama_manifest(manifest_dir)
        assert result is None
        out = capsys.readouterr().out
        assert BRACKET_STYLE in out, (
            f"a malformed digest that looks like a style tag must be shown "
            f"as literal text: {out!r}")

    def test_blob_missing_shows_bracketed_manifest_dir_verbatim(self, tmp_path, capsys):
        # A well-formed (64 hex char) digest whose blob file does not exist -
        # exercises the "blob missing" branch, printing `p` (the caller's own
        # manifest directory path).
        digest = "sha256:" + "a" * 64
        base = tmp_path / f"ollama-{BRACKET_STYLE}"
        manifest_dir = self._manifest_dir(base, digest)
        result = mm._resolve_ollama_manifest(manifest_dir)
        assert result is None
        out = capsys.readouterr().out
        assert f"ollama-{BRACKET_STYLE}" in out, (
            f"the manifest directory path must survive verbatim: {out!r}")


# ------------------------------------------------------------------ #
#  _prompt_predownload_dup / _prompt_duplicate_action                 #
# ------------------------------------------------------------------ #

class TestPromptPredownloadDupMarkupEscaping:
    def test_dup_names_survive_verbatim(self, monkeypatch, capsys):
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        choice = mm._prompt_predownload_dup(
            [f"a-{BRACKET_DROP}", f"b-{BRACKET_STYLE}"], "model_name")
        assert choice == "skip"
        out = capsys.readouterr().out
        assert f"a-{BRACKET_DROP}" in out, f"dup_names must survive verbatim: {out!r}"
        assert f"b-{BRACKET_STYLE}" in out, f"dup_names must survive verbatim: {out!r}"


class TestPromptDuplicateActionMarkupEscaping:
    def test_existing_names_survive_verbatim(self, monkeypatch, capsys):
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        choice = mm._prompt_duplicate_action(
            [f"a-{BRACKET_STYLE}", f"b-{BRACKET_DROP}"], "same file")
        assert choice == "skip"
        out = capsys.readouterr().out
        assert f"a-{BRACKET_STYLE}" in out, f"existing_names must survive verbatim: {out!r}"
        assert f"b-{BRACKET_DROP}" in out, f"existing_names must survive verbatim: {out!r}"


# ------------------------------------------------------------------ #
#  add_local                                                          #
# ------------------------------------------------------------------ #

class TestAddLocalMarkupEscaping:
    def test_not_found_shows_bracketed_path_verbatim(self, fake_registry, tmp_path, capsys):
        bad_path = str(tmp_path / f"missing-{BRACKET_DROP}")
        ok = mm.add_local(bad_path)
        assert ok is False
        out = capsys.readouterr().out
        assert f"missing-{BRACKET_DROP}" in out, f"path_str must survive verbatim: {out!r}"

    def test_localm_data_folder_refused_shows_bracketed_path_verbatim(
            self, fake_registry, monkeypatch, capsys):
        store, models_dir = fake_registry
        home = models_dir.parent / f"home-{BRACKET_STYLE}"
        home.mkdir()
        monkeypatch.setattr(mm, "HOME_DIR", home)
        ok = mm.add_local(str(home))
        assert ok is False
        out = capsys.readouterr().out
        assert f"home-{BRACKET_STYLE}" in out, (
            f"the refused data-dir path must survive verbatim: {out!r}")

    def test_incomplete_safetensors_shows_bracketed_path_verbatim(
            self, fake_registry, tmp_path, capsys):
        d = tmp_path / f"lone-{BRACKET_DROP}"
        d.mkdir()
        (d / "model.safetensors").write_bytes(b"x")   # no config.json alongside
        ok = mm.add_local(str(d / "model.safetensors"))
        assert ok is False
        out = capsys.readouterr().out
        assert f"lone-{BRACKET_DROP}" in out, f"the path must survive verbatim: {out!r}"

    def test_not_a_model_shows_bracketed_path_verbatim(self, fake_registry, tmp_path, capsys):
        d = tmp_path / f"empty-{BRACKET_STYLE}"
        d.mkdir()
        (d / "readme.txt").write_text("nothing here")
        ok = mm.add_local(str(d))
        assert ok is False
        out = capsys.readouterr().out
        assert f"empty-{BRACKET_STYLE}" in out, f"the path must survive verbatim: {out!r}"

    def test_name_collision_shows_bracketed_name_and_conflict_verbatim(
            self, fake_registry, tmp_path, monkeypatch, capsys):
        store, models_dir = fake_registry
        old = tmp_path / f"old-{BRACKET_DROP}.gguf"
        old.write_bytes(b"x")
        # No -n: model_name derives from p.stem (the caller's own filename),
        # which is NOT sanitized (unlike a -n value), so a bracket here
        # reaches _name_collision's message unmodified.
        model_name = f"m-{BRACKET_STYLE}"
        store[model_name] = {"path": str(old)}
        new = tmp_path / "external" / f"{model_name}.gguf"
        new.parent.mkdir()
        new.write_bytes(b"y")
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        ok = mm.add_local(str(new), store="copy")
        assert ok is False
        out = capsys.readouterr().out
        assert model_name in out, f"model_name must survive verbatim: {out!r}"
        assert f"old-{BRACKET_DROP}.gguf" in out, (
            f"the conflicting old path must survive verbatim: {out!r}")

    def test_store_copy_registered_but_failed_message_verbatim(
            self, fake_registry, tmp_path, monkeypatch, capsys):
        # A --store copy that lands in MODELS_DIR under a fresh name, but
        # whose CONTENT path-collides with an already-registered entry
        # (pre-registered here pointing at the not-yet-existing destination),
        # is declined by the (forced non-interactive) dedup prompt - the
        # honest "moved/copied but not registered" outcome, not a claimed
        # success.
        monkeypatch.setenv("COLUMNS", "1000")   # keep the long final message on one line
        store, models_dir = fake_registry
        external = tmp_path / "external" / f"new-{BRACKET_STYLE}.gguf"
        external.parent.mkdir()
        external.write_bytes(_GGUF_BYTES)
        dest = models_dir / external.name
        store[f"already-{BRACKET_DROP}"] = {"path": str(dest.resolve())}
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        ok = mm.add_local(str(external), name="newname", store="copy", no_hash=True)
        assert ok is False
        out = capsys.readouterr().out
        # _store_into_models_dir's own (separately tested, in
        # TestStoreIntoModelsDirMarkupEscaping) "Copying ..." progress print
        # ALSO shows this bracketed filename, earlier in `out` - isolate the
        # assertion to add_local's OWN final message (the distinct escape()
        # call this test exists to cover), so this test cannot pass on the
        # strength of a DIFFERENT function's escaping alone.
        final_line = next(
            line for line in out.splitlines() if "not registered as" in line)
        assert f"new-{BRACKET_STYLE}.gguf" in final_line, (
            f"the stored path must survive verbatim in add_local's own "
            f"message: {final_line!r}")
