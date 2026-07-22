# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the shared delegated-work presentation (localm.plugins.coder.delegated).

The invariant under test is not cosmetic: session_diff() is an INPUT to the
self-reviewer (agent/loop.py:381) and to episodic memory (agent/session.py:190),
so foreign content there would corrupt two model-facing loops. These tests pin
that delegated work is POINTED AT and never merged in.
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
    assert "Delegated changes (not in this tree)" in out
    assert "child1" in out
    assert "3 file(s)" in out
    assert "coder/child1-ab12ef34" in out
    # A runnable command, since the worktree may already be gone.
    assert "git diff" in out


def test_footer_contains_no_raw_diff_hunks():
    """The footer must not inline foreign hunks: /diff renders its diff through
    Syntax(..., "diff"), so inlined hunks would read as directly applicable."""
    out = d.render_footer([_cs(), _cs(label="child2", branch="coder/child2-ff99")])
    for marker in ("@@", "+++ ", "--- ", "diff --git"):
        assert marker not in out, f"footer leaked a diff hunk marker: {marker}"


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
