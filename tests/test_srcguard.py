# SPDX-License-Identifier: AGPL-3.0-or-later
"""The wrong-source-launch guard: localm/_srcguard.py.

The defect being guarded is that a console script, or a script run by path, sets
``sys.path[0]`` to the SCRIPT's directory and never to cwd, so an editable
install silently serves the checkout it was installed from. A caller edits one
tree, exercises another, and records an observation about code they did not
change.

The load-bearing tests here are the SUBPROCESS ones. They run the real entry
points, in a real child, against a real second checkout on disk, because the
property under test is entirely about how a real interpreter resolves a real
import. A test that monkeypatched the resolution would be asserting against its
own mock.

HOW THE MISMATCH IS BUILT, since it is not obvious: the child runs from a
SUBDIRECTORY of the fake checkout. A subdirectory holds no ``localm`` package, so
cwd being first on sys.path resolves nothing, and PYTHONPATH (the real tree)
wins. Standing root then walks up to the fake checkout, and the two differ. Put
the child directly in the fake checkout root instead and it would import the fake
package, testing nothing. This is also a real shape: running from ``<checkout>/
docs`` does exactly that.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from localm import _srcguard

REPO_ROOT = Path(__file__).resolve().parents[1]

# Long enough for a cold interpreter start plus the CLI import on a loaded box,
# short enough that a hang fails the suite instead of stalling it.
_TIMEOUT = 120


def _make_checkout(root: Path) -> Path:
    """A directory with both markers of a localm source checkout."""
    (root / "localm").mkdir(parents=True, exist_ok=True)
    (root / "localm" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text('[project]\nname = "localm"\n', encoding="utf-8")
    return root


def _run(args: list[str], cwd: Path, extra_env: dict[str, str] | None = None):
    env = dict(os.environ)
    # The real tree under test, reached the same way a caller would have to reach
    # it. Anything inherited would decide the result instead of the fixture.
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.pop("LOCALM_ALLOW_FOREIGN_SRC", None)
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
    )


class TestCheckoutDetection:
    """The predicate that decides whether the guard can speak at all."""

    def test_both_markers_required(self, tmp_path):
        both = _make_checkout(tmp_path / "both")
        assert _srcguard._is_checkout_root(str(both))

    def test_a_plain_python_project_is_not_a_localm_checkout(self, tmp_path):
        # pyproject.toml alone matches any project the caller happens to stand in.
        other = tmp_path / "other"
        other.mkdir()
        (other / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        assert not _srcguard._is_checkout_root(str(other))

    def test_an_installed_package_dir_is_not_a_checkout(self, tmp_path):
        # This is the shape of site-packages, and it is THE reason the guard is
        # inert for every ordinary user: an installed localm has no pyproject.
        site = tmp_path / "site-packages"
        (site / "localm").mkdir(parents=True)
        (site / "localm" / "__init__.py").write_text("", encoding="utf-8")
        assert not _srcguard._is_checkout_root(str(site))

    def test_standing_root_found_from_a_subdirectory(self, tmp_path):
        root = _make_checkout(tmp_path / "co")
        deep = root / "docs" / "guides"
        deep.mkdir(parents=True)
        assert _srcguard._standing_root(str(deep)) == str(root)

    def test_standing_root_is_none_outside_any_checkout(self, tmp_path):
        loose = tmp_path / "nowhere"
        loose.mkdir()
        assert _srcguard._standing_root(str(loose)) is None


class TestForeignSourceDecision:
    def test_same_checkout_is_silent(self):
        # The real tree, standing in the real tree: the normal case, and the one
        # that must never fire or the guard is unusable.
        assert _srcguard.foreign_source(cwd=str(REPO_ROOT)) is None

    def test_outside_any_checkout_is_silent(self, tmp_path):
        # A pytest basetemp, a home directory, a service working directory.
        assert _srcguard.foreign_source(cwd=str(tmp_path)) is None

    def test_different_checkout_is_reported(self, tmp_path):
        other = _make_checkout(tmp_path / "elsewhere")
        found = _srcguard.foreign_source(cwd=str(other))
        assert found is not None, "standing in a different checkout must be reported"
        running, standing = found
        assert Path(running).resolve() == REPO_ROOT
        assert Path(standing).resolve() == other.resolve()

    @pytest.mark.skipif(os.name != "nt", reason="junctions are a Windows shape")
    def test_same_checkout_reached_through_a_junction_is_silent(self, tmp_path):
        """One checkout, two spellings, via a directory junction.

        This is the ONLY case that exercises the realpath comparison: the cheap
        string compare says these differ, and only resolving them shows they are
        one tree. Worktrees do sit behind junctions on this platform, and reading
        that as a second checkout would refuse a perfectly correct launch.
        """
        link = tmp_path / "linked"
        made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(REPO_ROOT)],
                              capture_output=True, text=True, timeout=60)
        if made.returncode != 0:
            pytest.skip("could not create a junction: %s" % made.stderr.strip())
        # Guard the guard: if realpath does not actually collapse the junction,
        # this test would pass for the wrong reason on a future Python.
        assert os.path.realpath(str(link)).lower() == str(REPO_ROOT).lower(), (
            "realpath did not resolve the junction, so this test proves nothing")
        assert _srcguard.foreign_source(cwd=str(link)) is None

    def test_escape_hatch_silences_it(self, tmp_path, monkeypatch):
        other = _make_checkout(tmp_path / "elsewhere")
        assert _srcguard.foreign_source(cwd=str(other)) is not None
        monkeypatch.setenv(_srcguard.ENV_ALLOW, "1")
        assert _srcguard.foreign_source(cwd=str(other)) is None

    def test_report_names_both_trees_and_the_fix(self, tmp_path):
        other = _make_checkout(tmp_path / "elsewhere")
        running, standing = _srcguard.foreign_source(cwd=str(other))
        text = _srcguard._report(running, standing)
        # Both paths, so the reader can tell which is which without re-deriving.
        assert str(REPO_ROOT).replace("\\", "/") in text
        assert str(other).replace("\\", "/") in text
        # And the correction, ready to re-run.
        assert "PYTHONPATH=" in text


class TestRealEntryPoints:
    """A real child process, a real second checkout, the real entry points."""

    @pytest.fixture()
    def foreign_cwd(self, tmp_path) -> Path:
        sub = _make_checkout(tmp_path / "otherco") / "docs"
        sub.mkdir()
        return sub

    def test_import_warns_on_stderr(self, foreign_cwd):
        proc = _run(["-c", "import localm"], cwd=foreign_cwd)
        # "WARNING:" and not "Error:": at import this must REPORT, never refuse.
        assert "WARNING:" in proc.stderr, proc.stderr
        assert "DIFFERENT source checkout" in proc.stderr, proc.stderr
        assert "Error:" not in proc.stderr, proc.stderr
        # Never stdout: localm has children whose stdout is a machine-read
        # protocol, and a stray line there corrupts it.
        assert proc.stdout == ""
        # A warning must not take the process down.
        assert proc.returncode == 0

    def test_import_is_silent_in_its_own_checkout(self):
        proc = _run(["-c", "import localm"], cwd=REPO_ROOT)
        assert proc.returncode == 0
        assert "DIFFERENT source checkout" not in proc.stderr, proc.stderr

    def test_import_is_silent_outside_any_checkout(self, tmp_path):
        proc = _run(["-c", "import localm"], cwd=tmp_path)
        assert proc.returncode == 0
        assert "DIFFERENT source checkout" not in proc.stderr, proc.stderr

    def test_import_is_silent_with_the_escape_hatch(self, foreign_cwd):
        proc = _run(["-c", "import localm"], cwd=foreign_cwd,
                    extra_env={"LOCALM_ALLOW_FOREIGN_SRC": "1"})
        assert proc.returncode == 0
        assert "DIFFERENT source checkout" not in proc.stderr, proc.stderr

    def test_module_entry_point_refuses(self, foreign_cwd):
        proc = _run(["-m", "localm", "--version"], cwd=foreign_cwd)
        # Non-zero so a shell script or CI step cannot carry on past it.
        assert proc.returncode != 0, (proc.stdout, proc.stderr)
        # "Error:" is the REFUSAL, which only require_own_source emits. Matching
        # the shared phrase alone would also be satisfied by the import-time
        # WARNING, so this test would pass with the entry-point wiring deleted.
        assert "Error:" in proc.stderr, proc.stderr
        assert "DIFFERENT source checkout" in proc.stderr, proc.stderr
        # It must REFUSE, not merely complain and then run: a version banner here
        # would mean the wrong tree answered anyway.
        assert "localm, version" not in proc.stdout

    def test_module_entry_point_runs_in_its_own_checkout(self):
        proc = _run(["-m", "localm", "--version"], cwd=REPO_ROOT)
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        assert "version" in proc.stdout.lower()

    def test_console_entry_point_refuses(self, foreign_cwd):
        # console_main is what pyproject wires the `localm` script to. Driving it
        # here covers the exe without depending on an exe being installed.
        proc = _run(
            ["-c", "from localm.cli._core import console_main; console_main()"],
            cwd=foreign_cwd,
        )
        assert proc.returncode != 0, (proc.stdout, proc.stderr)
        assert "Error:" in proc.stderr, proc.stderr
        assert "DIFFERENT source checkout" in proc.stderr, proc.stderr

    def test_coder_console_entry_point_refuses(self, foreign_cwd):
        proc = _run(
            ["-c",
             "from localm.plugins.coder.cli._main import console_main; console_main()"],
            cwd=foreign_cwd,
        )
        # returncode is NOT discriminating here and must not be leaned on: with
        # the guard removed this entry point still exits 1, because the coder
        # plugin is not installed in a bare environment. Measured while
        # fires-controlling this file. The refusal text is the real assertion.
        assert "Error:" in proc.stderr, proc.stderr
        assert "DIFFERENT source checkout" in proc.stderr, proc.stderr
        # And it must refuse BEFORE reaching the plugin check, or the guard is
        # sitting behind something that answers first.
        assert "plugin is not active" not in proc.stderr, proc.stderr
