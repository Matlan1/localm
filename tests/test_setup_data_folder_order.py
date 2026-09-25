# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every installer settles where data lives before it runs any LocaLM code.

`localm setup-llama` records the builds it installs in the data folder's
config.json. If it ran before the data folder was chosen, it wrote into
`<clone>/home` even when the user then picked another folder: the build
history was lost to that install and uninstall left the folder behind.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _localm_runs(lines, *, comment, runner):
    """Indexes of lines that run LocaLM code other than install_manifest: the
    console script invoked as a command, `python -m localm <command>` or
    `python -m localm.<module>`."""
    hits = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or re.match(comment, s):
            continue
        if re.search(runner, s) or re.search(
                r"python[\w.]*\s+-m\s+localm(?:\s|\.(?!install_manifest))", s):
            hits.append(i)
    return hits


@pytest.mark.parametrize("script,comment,runner,dispatch", [
    ("setup.bat", r"(?i)(rem\b|echo\b|::)",
     r"(^|[(&|]\s*)\.venv\\Scripts\\localm\s+[a-z]", r"call :portable_home$"),
    ("setup.sh", r"(#|say\b|echo\b)",
     r"(^|\$\(\s*|[;&|]\s*|\b(then|do|else)\s+)\.venv/bin/localm\s+[a-z]", r"^portable_home$"),
])
def test_data_folder_is_chosen_before_the_runtime_is_provisioned(script, comment,
                                                                 runner, dispatch):
    lines = (ROOT / script).read_text(encoding="utf-8").splitlines()
    chosen = [i for i, ln in enumerate(lines) if re.search(dispatch, ln.strip())]
    assert chosen, f"{script}: the data-folder step moved"
    runs = _localm_runs(lines, comment=comment, runner=runner)
    provision = [i for i in runs if "setup-llama --backend" in lines[i]]
    assert provision, f"{script}: the provisioning command moved"
    early = [lines[i].strip() for i in runs if i < chosen[-1]]
    assert early == [], f"{script} runs LocaLM code before the data folder is chosen"
    assert chosen[-1] < min(provision)


def _load_gui(tmp_path, monkeypatch):
    name = "localm_installer_gui_order"
    spec = importlib.util.spec_from_file_location(name, ROOT / "installer" / "gui.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    return mod


def test_the_window_prepares_the_data_folder_before_provisioning(tmp_path, monkeypatch):
    gui = _load_gui(tmp_path, monkeypatch)
    data = tmp_path / "elsewhere" / "data"
    seen = {}

    def fake_run(cmd, emit, plan, **kw):
        flat = [str(c) for c in cmd]
        if "setup-llama" in flat:
            cfg = tmp_path / "localm-home.cfg"
            seen["cfg"] = cfg.read_text(encoding="utf-8").strip() if cfg.is_file() else None
            seen["data_exists"] = data.is_dir()
        return 0

    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(gui, "_run", fake_run)
    monkeypatch.setattr(gui, "find_uv", lambda root: "uv")
    monkeypatch.setattr(gui, "make_shortcut", lambda plan, emit: "")
    monkeypatch.setattr(gui, "torch_spec_for", lambda b: ("", ""))
    monkeypatch.setattr(gui, "_query", lambda args: "")
    plan = gui.Plan(backend="cpu", portable_data=False, data_path=str(data),
                    shortcut="none")
    for step in gui.build_steps(plan):
        step.run(lambda _s: None)
    assert seen == {"cfg": str(data), "data_exists": True}
    assert not (tmp_path / "home").exists()
