# SPDX-License-Identifier: AGPL-3.0-or-later
"""The pip-install path: a wheel install must be able to provision and stop.

A pip install has no git checkout, so the runtime package ships inside the
wheel and its lib/ is provisioned in place. These pin the two failures that
made `pip install localm` unusable.
"""
from __future__ import annotations

import builtins
import sys
import tomllib
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


class TestRuntimePackageShips:
    def test_wheel_target_includes_the_runtime_package(self):
        """Without this the wheel carries no runtime package, so a pip install
        has nothing to provision into and setup-llama cannot succeed."""
        cfg = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        packages = cfg["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
        assert "localm" in packages
        assert "runtime/localm_llama_runtime" in packages

    def test_runtime_package_has_an_importable_init_and_lib_dir(self):
        pkg = REPO / "runtime" / "localm_llama_runtime"
        assert (pkg / "__init__.py").is_file()
        assert (pkg / "lib").is_dir()

    def test_pypi_readme_is_the_declared_readme_and_exists(self):
        """PyPI renders `readme`; the GitHub README tells the reader to clone,
        which is the wrong instruction on a PyPI page."""
        cfg = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        readme = cfg["project"]["readme"]
        assert readme == "docs/pypi-readme.md"
        body = (REPO / readme).read_text(encoding="utf-8")
        assert "pip install localm" in body
        assert "localm setup-llama" in body

    def test_pypi_readme_does_not_overclaim_nothing_leaves_your_machine(self):
        """The unqualified claim was false: an update check leaves the machine
        by default (config.DEFAULT_CONFIG["bugreport_upload_url"] - the
        updater's fallback endpoint - is non-empty out of the box), so any
        surviving "nothing leaving your machine" phrasing must be qualified,
        and the page must say so."""
        from localm.config import DEFAULT_CONFIG

        cfg = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        readme = cfg["project"]["readme"]
        body = (REPO / readme).read_text(encoding="utf-8")
        assert "nothing leaving your machine" not in body
        assert bool(DEFAULT_CONFIG.get("bugreport_upload_url")), (
            "this test's premise (an update endpoint is configured by default) "
            "no longer holds - re-check whether the page still needs the "
            "update-check caveat at all")
        assert "update check" in body.lower()


class TestKillPidWithoutPsutil:
    def test_kill_pid_without_psutil_does_not_raise(self, monkeypatch):
        """psutil is an optional extra. Importing it INSIDE the try left the name
        unbound while `except psutil.NoSuchProcess` still had to evaluate it,
        raising UnboundLocalError out of a never-raises function."""
        from localm import instances

        monkeypatch.setattr(instances, "pid_alive", lambda pid: True)
        real_import = builtins.__import__

        def no_psutil(name, *a, **kw):
            if name == "psutil":
                raise ImportError("No module named 'psutil'")
            return real_import(name, *a, **kw)

        monkeypatch.delitem(sys.modules, "psutil", raising=False)
        monkeypatch.setattr(builtins, "__import__", no_psutil)

        # Must return a bool, not raise. pid_alive is patched True, so the
        # honest answer is "not confirmed gone".
        assert instances.kill_pid(999999) is False
