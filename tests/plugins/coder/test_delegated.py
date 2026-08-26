# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the shared delegated-work presentation (localm.plugins.coder.delegated).

session_diff() is an INPUT to the self-reviewer (agent/loop.py) and to episodic
memory (agent/session.py), so delegated work is POINTED AT there and never
merged in.
"""

from __future__ import annotations

from localm.plugins.coder import delegated as d


def _cs(**kw):
    base = dict(label="child1", branch="coder/child1-ab12ef34", file_count=3,
                source="parallel", status="ok", base="e141f3cf")
    base.update(kw)
    return d.DelegatedChangeSet(**base)


def test_empty_footer_is_empty_string():
    """Every display site appends this unconditionally, so the empty case must be
    a true no-op - otherwise wiring it in would change output for sessions that
    never delegated anything."""
    assert d.render_footer([]) == ""


def test_changeset_without_a_branch_is_not_advertised():
    """A child whose worktree could not be created has nothing to point at."""
    assert d.render_footer([_cs(branch="")]) == ""


def test_footer_names_branch_file_count_and_view_command():
    out = d.render_footer([_cs()])
    # The heading says the changes are not in the user's working tree.
    assert "Delegated work (NOT in your working tree)" in out
    assert "child1" in out
    assert "3 file(s)" in out
    assert "coder/child1-ab12ef34" in out
    # A runnable command, since the worktree may already be gone.
    assert "git diff" in out


def test_inlined_diff_is_labelled_as_not_in_this_tree():
    """The hunks ARE inlined (discoverability), so the labelling is what stops a
    user reading them as changes already applied to their working tree."""
    out = d.render_footer([_cs(diff="diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n")])
    assert "NOT in your working tree" in out
    assert "have not been merged" in out
    # The hunks are present and the branch that holds them is named.
    assert "@@" in out
    assert "coder/child1-ab12ef34" in out


def test_inlined_diff_is_capped_and_says_so():
    """/diff is invoked repeatedly; two unbounded child diffs would drown the
    parent's own changes. The cap must announce itself and stay reachable."""
    huge = "\n".join(f"+line {i}" for i in range(4000))
    out = d.render_footer([_cs(diff=huge)])
    assert len(out) < 6000, "an unbounded child diff was inlined"
    assert "truncated" in out
    assert "git diff" in out, "the full diff must stay reachable after truncation"


def test_footer_is_never_wired_into_a_model_facing_site():
    """STRUCTURAL GUARD. session_diff() feeds the self-reviewer (loop.py) and
    episodic memory (session.py). Appending the delegated section AT THOSE CALL
    SITES corrupts those loops just as merging foreign keys would, even though
    session_diff() itself is untouched. Pin it in the source so a future
    well-meaning edit cannot quietly reintroduce it.
    """
    from pathlib import Path
    import localm.plugins.coder as pkg

    root = Path(pkg.__file__).parent
    for module in ("agent/loop.py", "agent/session.py"):
        src = (root / module).read_text(encoding="utf-8", errors="replace")
        for banned in ("footer_for", "render_footer", "delegated_report"):
            assert banned not in src, (
                f"{module} references {banned}: the delegated section must never "
                "reach the self-reviewer or the episode"
            )


def test_recording_delegated_work_does_not_change_what_the_reviewer_would_see():
    """The reviewer is handed session_diff()'s return. Recording delegated work
    must leave that byte-identical."""
    from localm.plugins.coder.agent.persistence import _PersistenceMixin

    class Agent(_PersistenceMixin):
        def __init__(self):
            self.cwd = __import__("pathlib").Path(".")
            self._changed_files = {}

    a = Agent()
    before = a.session_diff()
    d.record(a, _cs(diff="diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n"))
    assert a.session_diff() == before, \
        "delegated work altered what the self-reviewer and episode receive"


def test_failed_and_timed_out_children_are_flagged():
    out = d.render_footer([_cs(status="timeout")])
    assert "timeout" in out
    ok = d.render_footer([_cs(status="ok")])
    assert "[ok]" not in ok, "the healthy case should not be noisy"


def test_zero_file_change_is_stated_not_blank():
    out = d.render_footer([_cs(file_count=0)])
    assert "no file changes" in out


def test_record_and_footer_for_round_trip():
    class Agent:
        pass

    a = Agent()
    assert d.footer_for(a) == "", "a fresh agent has delegated nothing"
    d.record(a, _cs())
    d.record(a, _cs(label="child2", branch="coder/child2-ff99", file_count=1))
    out = d.footer_for(a)
    assert "child1" in out and "child2" in out
    assert "3 file(s)" in out and "1 file(s)" in out


def test_record_does_not_touch_the_parents_changed_files():
    """THE INVARIANT. Delegated work must never enter the parent's own tree map,
    which feeds session_diff() -> the self-reviewer and episodic memory."""
    class Agent:
        def __init__(self):
            self._changed_files = {"real.py": {"writes": 1}}

    a = Agent()
    d.record(a, _cs())
    assert a._changed_files == {"real.py": {"writes": 1}}, \
        "delegated work leaked into the parent's changed-files map"


def test_view_command_uses_the_branch_not_the_worktree_path():
    """The worktree is transient and may already be removed; the branch is durable."""
    cmd = _cs().view_command()
    assert "coder/child1-ab12ef34" in cmd
    assert "worktree" not in cmd.lower()
