# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every ``pip install`` in a workflow installs from a hash-locked set.

``_unpinned`` applies the rule OpenSSF Scorecard's Pinned-Dependencies check uses
for pip: a command passes with ``--require-hashes``, or as an editable install of
local source with ``--no-deps``; ``python -m pip install --upgrade pip`` and a
bare package name do not. The CPU torch set (optional-stacks) and the mutmut set
(mutation jobs) are committed, hash-locked files.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
TORCH_CPU = ROOT / ".github" / "requirements" / "torch-cpu.txt"
MUTMUT = ROOT / ".github" / "requirements" / "mutmut.txt"

_FLAG = re.compile(r"^(\-\-?\w+)+$")
_REMOTE = re.compile(r"^(git|svn|hg|bzr).+$")
_GIT_SHA = re.compile(r"^git(\+(https?|ssh|git))?\:\/\/.*(.git)?@[a-fA-F0-9]{40}(#egg=.*)?$")
_HASHED_PIN = re.compile(
    r"^([A-Za-z0-9._-]+)==(\S+) \\\n((?:    --hash=sha256:[0-9a-f]{64}(?: \\)?\n)+)", re.MULTILINE)


def _is_pip_install(cmd: list[str]) -> bool:
    return len(cmd) >= 2 and Path(cmd[0]).name.lower() in ("pip", "pip3") and cmd[1].lower() == "install"


def _unpinned(cmd: list[str]) -> bool:
    no_deps = editable = extra = wheel = False
    pinned_editable = True
    for tok in cmd[2:]:
        if tok.lower() == "--no-deps":
            no_deps = True
        elif tok in ("-e", "--editable"):
            editable = True
        elif tok.lower() == "--require-hashes":
            return editable and (not no_deps or not pinned_editable)
        elif _FLAG.match(tok):
            continue
        elif tok.endswith(".whl"):
            wheel = True
        elif editable:
            if _REMOTE.match(tok) and not _GIT_SHA.match(tok):
                pinned_editable = False
        else:
            extra = True
    if editable:
        return not no_deps or not pinned_editable
    return extra or not wheel


def _unpinned_commands(run: str) -> list[str]:
    found = []
    for line in run.splitlines():
        line = re.sub(r"\$\{\{.*?\}\}", "X", line.strip())
        if not line or line.startswith("#"):
            continue
        try:
            cmd = shlex.split(line)
        except ValueError:
            continue
        for i in range(len(cmd) - 1):
            if cmd[i].lower() == "-m" and cmd[i + 1].lower() == "pip":
                cmd = cmd[i + 1:]
                break
        if _is_pip_install(cmd) and _unpinned(cmd):
            found.append(line)
    return found


def _offenders(workflow: Path) -> list[str]:
    doc = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    out = []
    for job_name, job in (doc.get("jobs") or {}).items():
        for step in job.get("steps", []):
            for line in _unpinned_commands(step.get("run", "")):
                out.append(f"{workflow.name} / {job_name}: {line}")
    return out


@pytest.mark.parametrize("command, unpinned", [
    ("pip install -e .", True),
    ('pip install -e ".[dev]"', True),
    ("pip install --no-deps -e .", False),
    ("pip install mutmut==3.7.0", True),
    ("python -m pip install --upgrade pip", True),
    ("pip install --require-hashes --no-deps -r requirements.txt", False),
    ("pip install --require-hashes -r requirements.txt", False),
    ("pip install -r requirements.txt", True),
    ("pip install ./some-1.0-py3-none-any.whl", False),
])
def test_the_rule_classifies_known_commands(command, unpinned):
    assert bool(_unpinned_commands(command)) is unpinned


# publish-pypi.yml waits for its own change: its workflow_dispatch publishes.
@pytest.mark.parametrize("workflow", sorted(
    p for p in WORKFLOWS.glob("*.yml") if p.name != "publish-pypi.yml"), ids=lambda p: p.name)
def test_every_pip_install_in_the_workflow_is_hash_pinned(workflow):
    assert _offenders(workflow) == []


def _locked_pins(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    entries = _HASHED_PIN.findall(text)
    declared = re.findall(r"^[A-Za-z0-9._-]+==", text, re.MULTILINE)
    assert len(entries) == len(declared), f"{path.name} has a requirement without a sha256 hash"
    return {name.lower(): version for name, version, _ in entries}


def _runs(job: str) -> str:
    doc = yaml.safe_load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
    return "\n".join(s.get("run", "") for s in doc["jobs"][job]["steps"])


def test_the_torch_cpu_set_is_hash_locked_and_pins_the_cpu_build():
    pins = _locked_pins(TORCH_CPU)
    assert pins.get("torch") == "2.11.0+cpu"
    assert "setuptools" not in pins


def test_the_mutmut_set_is_hash_locked_and_pins_mutmut():
    assert _locked_pins(MUTMUT).get("mutmut") == "3.7.0"


def test_the_optional_stacks_job_installs_the_torch_cpu_set_with_hashes():
    runs = _runs("optional-stacks")
    assert "pip install --require-hashes --no-deps -r .github/requirements/torch-cpu.txt" in runs
    assert "--prune torch" in runs


@pytest.mark.parametrize("job", ["mutation-run", "mutation-test"])
def test_the_mutation_jobs_install_the_mutmut_set_with_hashes(job):
    assert "pip install --require-hashes --no-deps -r .github/requirements/mutmut.txt" in _runs(job)
