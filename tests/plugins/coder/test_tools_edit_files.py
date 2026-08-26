# SPDX-License-Identifier: AGPL-3.0-or-later
"""edit_files - the multi-file exact-string batch edit.

The contract worth testing is ATOMICITY: a batch either applies completely or
leaves every target byte-identical to how it started. Two distinct failure
points have to hold that line:

- a REJECTED edit (bad path, missing file, `old` that does not match). Caught in
  the validate-first phase, before any write, so earlier files are never touched.
- a FAILED WRITE partway through the batch. The already-written files must be
  restored from the pre-batch snapshots.

Both are tested against real files on disk and asserted on BYTES, never on the
tool's own report.
"""

from unittest.mock import patch

import pytest

from localm.plugins.coder.tools import tool_edit_files


@pytest.fixture
def project(tmp_path):
    (tmp_path / "a.py").write_text("import old\nvalue = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("import old\nvalue = 2\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("import old\nvalue = 3\n", encoding="utf-8")
    return tmp_path


def _swap(path):
    return {"path": path, "old": "import old", "new": "import new"}


class TestBatchApply:
    def test_applies_every_edit(self, project):
        r = tool_edit_files(project, [_swap("a.py"), _swap("b.py"), _swap("c.py")])
        assert r.ok is True
        for name in ("a.py", "b.py", "c.py"):
            assert (project / name).read_text(encoding="utf-8").startswith("import new")
        assert "3 edit(s) across 3 file(s)" in r.output

    def test_replaces_only_the_first_occurrence_per_edit(self, project):
        p = project / "a.py"
        p.write_text("dup\ndup\n", encoding="utf-8")
        r = tool_edit_files(project, [{"path": "a.py", "old": "dup", "new": "one"}])
        assert r.ok is True
        assert p.read_text(encoding="utf-8") == "one\ndup\n"

    def test_two_edits_to_the_same_file_compose(self, project):
        r = tool_edit_files(project, [
            {"path": "a.py", "old": "import old", "new": "import mid"},
            {"path": "a.py", "old": "value = 1", "new": "value = 99"},
        ])
        assert r.ok is True
        assert (project / "a.py").read_text(encoding="utf-8") == "import mid\nvalue = 99\n"

    def test_second_edit_sees_the_first_edits_text(self, project):
        # Chained on the SAME text: edits compose in memory.
        r = tool_edit_files(project, [
            {"path": "a.py", "old": "import old", "new": "import mid"},
            {"path": "a.py", "old": "import mid", "new": "import new"},
        ])
        assert r.ok is True
        assert (project / "a.py").read_text(encoding="utf-8").startswith("import new")

    def test_syntax_error_is_surfaced_not_swallowed(self, project):
        r = tool_edit_files(project, [
            {"path": "a.py", "old": "value = 1", "new": "def broken("},
        ])
        assert r.ok is True                     # the write happened
        assert "syntax check" in r.output       # and the breakage is reported
        assert "syntax error" in r.summary


class TestAllOrNothing:
    """The required guarantee: edit 2 fails -> file 1 is untouched on disk."""

    def test_rejected_edit_leaves_earlier_file_byte_identical(self, project):
        before = (project / "a.py").read_bytes()
        r = tool_edit_files(project, [
            _swap("a.py"),
            {"path": "b.py", "old": "NOT IN THE FILE", "new": "x"},
        ])
        assert r.ok is False
        assert (project / "a.py").read_bytes() == before
        assert "edits[1]" in r.output

    def test_failed_write_rolls_earlier_file_back_to_original_bytes(self, project):
        """The rollback path proper: a.py IS written, then b.py's write fails."""
        before_a = (project / "a.py").read_bytes()
        before_b = (project / "b.py").read_bytes()
        real_write = type(project).write_text

        def failing_write(self, *args, **kwargs):
            if self.name == "b.py":
                raise OSError("disk full")
            return real_write(self, *args, **kwargs)

        with patch.object(type(project), "write_text", failing_write):
            r = tool_edit_files(project, [_swap("a.py"), _swap("b.py")])

        assert r.ok is False
        assert (project / "a.py").read_bytes() == before_a, \
            "a.py was written and must be restored to its ORIGINAL bytes"
        assert (project / "b.py").read_bytes() == before_b
        assert "Rolled back" in r.output

    def test_a_half_written_file_is_restored_too(self, project):
        """write_text opens with "w", so a failure PARTWAY through a write leaves
        that file truncated. It is not in the already-written list, so it has to
        be restored explicitly or the 'nothing was left partial' claim is false."""
        before_b = (project / "b.py").read_bytes()
        real_write = type(project).write_text

        def truncate_then_fail(self, *args, **kwargs):
            if self.name == "b.py":
                real_write(self, "", encoding="utf-8")   # truncated, as "w" does
                raise OSError("disk full mid-write")
            return real_write(self, *args, **kwargs)

        with patch.object(type(project), "write_text", truncate_then_fail):
            r = tool_edit_files(project, [_swap("a.py"), _swap("b.py")])

        assert r.ok is False
        assert (project / "b.py").read_bytes() == before_b, \
            "the half-written file must be restored, not left truncated"

    def test_rollback_restores_all_earlier_files(self, project):
        originals = {n: (project / n).read_bytes() for n in ("a.py", "b.py", "c.py")}
        real_write = type(project).write_text

        def failing_write(self, *args, **kwargs):
            if self.name == "c.py":
                raise OSError("disk full")
            return real_write(self, *args, **kwargs)

        with patch.object(type(project), "write_text", failing_write):
            r = tool_edit_files(project, [_swap("a.py"), _swap("b.py"), _swap("c.py")])

        assert r.ok is False
        for name, original in originals.items():
            assert (project / name).read_bytes() == original, f"{name} not restored"

    def test_a_failed_rollback_is_reported_never_claimed_clean(self, project):
        """If the restore itself fails, the tool must say so: a caller told
        'rolled back' while a file holds a partial edit is the worst case."""
        real_write_text = type(project).write_text

        def failing_write_text(self, *args, **kwargs):
            if self.name == "b.py":
                raise OSError("disk full")
            return real_write_text(self, *args, **kwargs)

        def failing_write_bytes(self, *args, **kwargs):
            raise OSError("still full")         # the rollback write also fails

        with patch.object(type(project), "write_text", failing_write_text), \
             patch.object(type(project), "write_bytes", failing_write_bytes):
            r = tool_edit_files(project, [_swap("a.py"), _swap("b.py")])

        assert r.ok is False
        assert "rollback ITSELF failed" in r.output
        assert "a.py" in r.output
        # The file is left changed.
        assert (project / "a.py").read_text(encoding="utf-8").startswith("import new")

    def test_missing_file_rejects_the_whole_batch(self, project):
        before = (project / "a.py").read_bytes()
        r = tool_edit_files(project, [_swap("a.py"), _swap("nope.py")])
        assert r.ok is False
        assert "File not found" in r.output
        assert (project / "a.py").read_bytes() == before

    def test_directory_target_rejects_the_whole_batch(self, project):
        (project / "sub").mkdir()
        before = (project / "a.py").read_bytes()
        r = tool_edit_files(project, [_swap("a.py"), _swap("sub")])
        assert r.ok is False
        assert "Not a file" in r.output
        assert (project / "a.py").read_bytes() == before


class TestValidation:
    def test_empty_old_rejected(self, project):
        before = (project / "a.py").read_bytes()
        r = tool_edit_files(project, [{"path": "a.py", "old": "", "new": "x"}])
        assert r.ok is False
        assert "`old` is empty" in r.output
        assert (project / "a.py").read_bytes() == before

    def test_miss_shows_the_closest_snippet_hint(self, project):
        # A miss (a token that is not in the file) still shows the closest-snippet
        # hint. A whitespace-only variant is not a miss, so this is a real token
        # difference rather than different spacing.
        r = tool_edit_files(project, [
            {"path": "a.py", "old": "import  older", "new": "import new"},
        ])
        assert r.ok is False
        assert "Closest match in the file" in r.output
        assert "import old" in r.output

    def test_whitespace_only_variant_matches(self, project):
        # `old` differs from the file ("import old") only in spacing; edit_files
        # tolerates that as a unique match.
        r = tool_edit_files(project, [
            {"path": "a.py", "old": "import  old", "new": "import new"},   # 2 spaces
        ])
        assert r.ok is True
        assert (project / "a.py").read_text(encoding="utf-8") == "import new\nvalue = 1\n"

    def test_path_traversal_rejected(self, project):
        r = tool_edit_files(project, [
            {"path": "../escape.txt", "old": "a", "new": "b"},
        ])
        assert r.ok is False
        assert "outside the working directory" in r.output

    def test_traversal_in_a_later_edit_leaves_earlier_file_untouched(self, project):
        before = (project / "a.py").read_bytes()
        r = tool_edit_files(project, [
            _swap("a.py"),
            {"path": "../escape.txt", "old": "a", "new": "b"},
        ])
        assert r.ok is False
        assert (project / "a.py").read_bytes() == before

    @pytest.mark.parametrize("edits", [[], "not a list", None, {}])
    def test_non_list_or_empty_edits_rejected(self, project, edits):
        r = tool_edit_files(project, edits)
        assert r.ok is False
        assert "non-empty list" in r.output

    def test_malformed_item_rejected(self, project):
        r = tool_edit_files(project, ["a.py"])
        assert r.ok is False
        assert "not an object" in r.output

    @pytest.mark.parametrize("edit", [
        {"path": "a.py", "old": "import old"},          # no new
        {"path": "a.py", "new": "import new"},          # no old
        {"old": "import old", "new": "import new"},     # no path
    ])
    def test_incomplete_item_rejected(self, project, edit):
        before = (project / "a.py").read_bytes()
        r = tool_edit_files(project, [edit])
        assert r.ok is False
        assert (project / "a.py").read_bytes() == before


class TestRegistration:
    def test_registered_as_destructive(self):
        from localm.plugins.coder.tools import TOOL_REGISTRY
        assert TOOL_REGISTRY["edit_files"].destructive is True

    def test_native_tool_schema_declares_object_items(self):
        """An array of {path, old, new} must not be advertised as an array of
        strings, or a native tool-calling backend sends the wrong shape."""
        from localm.plugins.coder.agent.tooldefs import _build_openai_tool_defs
        defs = {d["function"]["name"]: d for d in _build_openai_tool_defs()}
        items = defs["edit_files"]["function"]["parameters"]["properties"]["edits"]["items"]
        assert items["type"] == "object"
        assert set(items["properties"]) == {"path", "old", "new"}

    def test_other_array_params_still_default_to_string_items(self):
        from localm.plugins.coder.agent.tooldefs import _build_openai_tool_defs
        defs = {d["function"]["name"]: d for d in _build_openai_tool_defs()}
        items = defs["spawn_agent"]["function"]["parameters"]["properties"]["files"]["items"]
        assert items == {"type": "string"}
