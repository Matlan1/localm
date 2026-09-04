# SPDX-License-Identifier: AGPL-3.0-or-later
"""An alias is a second name for one model file, not a second model.

`localm run X` refuses to attach to a running server that is serving a
DIFFERENT model, so it can never answer with something other than what was
asked for. That guard compared the two NAMES, so asking for an alias of the
model already loaded was read as asking for a different model: the run was
refused, and the feature whose whole promise is "both names point at the same
model file" did not work whenever a server was up - which is the normal state.

The same comparison guarded the gui attach path, where the code's own comment
says "same model -> attach".
"""

from __future__ import annotations

from localm.model_manager import names_same_model


def _reg(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"")
    other = tmp_path / "other.gguf"
    other.write_bytes(b"")
    return {
        "full-name": {"path": str(f)},
        "alias": {"path": str(f)},
        "different": {"path": str(other)},
    }, f


class TestNamesSameModel:
    def test_an_alias_is_the_same_model(self, tmp_path):
        reg, _ = _reg(tmp_path)
        assert names_same_model("full-name", "alias", reg) is True
        assert names_same_model("alias", "full-name", reg) is True

    def test_a_different_file_is_a_different_model(self, tmp_path):
        reg, _ = _reg(tmp_path)
        assert names_same_model("full-name", "different", reg) is False

    def test_the_same_name_needs_no_registry(self):
        assert names_same_model("x", "x", reg={}) is True

    def test_an_unregistered_name_is_not_assumed_the_same(self, tmp_path):
        """Not knowing a name is not evidence that it is the model loaded; the
        guard must still refuse."""
        reg, _ = _reg(tmp_path)
        assert names_same_model("full-name", "never-registered", reg) is False
        assert names_same_model("never-registered", "full-name", reg) is False

    def test_empty_names_are_not_the_same(self, tmp_path):
        reg, _ = _reg(tmp_path)
        assert names_same_model("", "full-name", reg) is False
        assert names_same_model("full-name", "", reg) is False
        assert names_same_model("", "", reg) is False

    def test_a_malformed_entry_is_not_assumed_the_same(self, tmp_path):
        """registry.json is hand-editable; a junk entry must not read as a
        match, which would let a different model answer."""
        reg, f = _reg(tmp_path)
        for junk in ("not-a-dict", {}, {"path": None}, {"path": ""},
                     {"path": 17}, {"path": "../escape.gguf"}):
            reg["junk"] = junk
            assert names_same_model("full-name", "junk", reg) is False, junk

    def test_paths_are_compared_resolved_not_as_strings(self, tmp_path):
        """The same file spelled two ways is one model. A path with a '..'
        component is NOT one of those ways: the registry refuses it outright as
        a traversal guard on a hand-editable file, which the malformed-entry
        case above pins."""
        f = tmp_path / "m.gguf"
        f.write_bytes(b"")
        reg = {
            "a": {"path": str(f)},
            "b": {"path": str(f).replace("\\", "/")},
        }
        assert names_same_model("a", "b", reg) is True


class TestBothGuardsUseIt:
    """The refusal exists in two places and both must resolve aliases, or the
    bug simply moves to the other entry point."""

    def test_the_run_guard_resolves_aliases(self):
        from pathlib import Path
        src = Path(__file__).resolve().parents[1] / "localm" / "cli" / "chat.py"
        t = src.read_text(encoding="utf-8")
        assert "names_same_model(active, model)" in t, (
            "localm run compares the names, so an alias of the loaded model "
            "reads as a different model")
        assert "if active and active != model:" not in t

    def test_the_gui_attach_guard_resolves_aliases(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "localm" / "plugins"
               / "gui" / "cli.py")
        t = src.read_text(encoding="utf-8")
        assert "names_same_model(active, model)" in t
        assert "if active and active != model:" not in t
