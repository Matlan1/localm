# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.sh's localm-entry-point install/retry block, and localm.sh's own
diagnosis of the same failure mode.

Regression for a filed residual: setup.sh:342-360 retries the `uv pip install`
once if `.venv/bin/localm` did not land (a real, reproduced WSL2/DrvFs quirk
where uv reports success but drops exactly one file), then warns loudly and
CONTINUES if it is still missing afterwards - by design, since the rest of
setup does not depend on this entry point. But the retry call itself was a
bare command under `set -euo pipefail`, the one unguarded command in that
block: a retry that ERRORS (not just "ran but didn't create the file") kills
setup right there, so the loud "STILL missing" warning a few lines below never
prints - contradicting the block's own comment.

No test covered any of the three already-verified-live branches (present /
recovered-by-retry / still-missing-after-retry), let alone the retry-errors
case this file adds coverage for. Extracts and runs ONLY the install/retry
block (not the whole script) against a stub `uv`, in the extract-a-source-
slice style of tests/test_setup_gpu_detect.py.

Also covers localm.sh's own misdiagnosis of this exact failure mode: it used
to report "No .venv found" even when the venv was fine and only the console
script was missing.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP_SH = ROOT / "setup.sh"
LOCALM_SH = ROOT / "localm.sh"


def _bash() -> str | None:
    return shutil.which("bash")


pytestmark = pytest.mark.skipif(_bash() is None, reason="no bash on PATH")


def _make_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


# ---------------------------------------------------------------------------
# setup.sh: install + retry block
# ---------------------------------------------------------------------------


def _install_block() -> str:
    src = SETUP_SH.read_text(encoding="utf-8")
    start = src.index("# ---- install localm (editable) ----")
    end = src.index("# ---- native llama.cpp runtime wheel", start)
    return src[start:end]


def _make_uv_stub(bin_dir: Path) -> None:
    # Records its own call count in a file RELATIVE to cwd (the test sets cwd
    # to tmp_path for the whole run, so both this stub and the extracted
    # setup.sh block agree on where ".venv/bin/localm" and the counter live -
    # no absolute path needs to survive translation into the bash script
    # text, which is what makes this portable across a Windows tmp_path).
    # Call N's exit code / whether it creates .venv/bin/localm are read from
    # STUB_RC_<N> / STUB_CREATE_<N> env vars, set per-test.
    _make_executable(
        bin_dir / "uv",
        "#!/bin/sh\n"
        'COUNT_FILE=".stub-uv-calls"\n'
        'N=$(cat "$COUNT_FILE" 2>/dev/null || echo 0)\n'
        'N=$((N + 1))\n'
        'echo "$N" > "$COUNT_FILE"\n'
        'eval "RC=\\${STUB_RC_$N:-0}"\n'
        'eval "CREATE=\\${STUB_CREATE_$N:-0}"\n'
        'if [ "$CREATE" = "1" ]; then\n'
        '  mkdir -p .venv/bin\n'
        "  printf '#!/bin/sh\\nexit 0\\n' > .venv/bin/localm\n"
        '  chmod +x .venv/bin/localm\n'
        'fi\n'
        'exit "$RC"\n',
    )


def _run_install_block(tmp_path: Path, *, call_specs: dict[int, tuple[int, bool]]
                        ) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_uv_stub(bin_dir)
    script = (
        "set -euo pipefail\n"
        'say() { printf "%s\\n" "$*"; }\n'
        + _install_block()
        + '\nprintf "COMPLETED\\n"\n'
    )
    env = dict(os.environ)
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    for n, (rc, creates) in call_specs.items():
        env[f"STUB_RC_{n}"] = str(rc)
        env[f"STUB_CREATE_{n}"] = "1" if creates else "0"
    return subprocess.run([_bash(), "-c", script], capture_output=True, text=True,
                          env=env, cwd=str(tmp_path), timeout=15)


def test_present_no_retry_needed(tmp_path):
    result = _run_install_block(tmp_path, call_specs={1: (0, True)})
    assert result.returncode == 0, result.stderr
    assert "COMPLETED" in result.stdout
    assert "retrying once" not in result.stdout
    assert "STILL missing" not in result.stdout
    assert (tmp_path / ".venv" / "bin" / "localm").exists()
    assert (tmp_path / ".stub-uv-calls").read_text().strip() == "1"


def test_recovered_by_retry(tmp_path):
    result = _run_install_block(tmp_path, call_specs={1: (0, False), 2: (0, True)})
    assert result.returncode == 0, result.stderr
    assert "COMPLETED" in result.stdout
    assert "retrying once" in result.stdout
    assert "STILL missing" not in result.stdout
    assert (tmp_path / ".venv" / "bin" / "localm").exists()


def test_still_missing_after_retry_warns_and_continues(tmp_path):
    result = _run_install_block(tmp_path, call_specs={1: (0, False), 2: (0, False)})
    assert result.returncode == 0, result.stderr
    assert "COMPLETED" in result.stdout
    assert "retrying once" in result.stdout
    assert "STILL missing" in result.stdout
    assert not (tmp_path / ".venv" / "bin" / "localm").exists()


def test_retry_itself_erroring_still_reaches_the_still_missing_warning(tmp_path):
    # THE FIX under test. Before it, setup.sh:344's retry call was a bare
    # command under `set -euo pipefail`: a non-zero exit here killed the
    # extracted block right at this line, "COMPLETED" and the STILL-missing
    # warning never printed, and the process exit code was the retry's own
    # (7 here) instead of 0. Revert the `|| true` guard to see this go red.
    result = _run_install_block(tmp_path, call_specs={1: (0, False), 2: (7, False)})
    assert result.returncode == 0, result.stderr
    assert "COMPLETED" in result.stdout
    assert "retrying once" in result.stdout
    assert "STILL missing" in result.stdout


def test_retry_install_is_guarded_in_source():
    """Static backstop that holds even without bash on PATH: the retry
    install must not be a bare command under `set -euo pipefail`."""
    block = _install_block()
    first = block.index('uv pip install -p .venv -e ".[coder,voice,monitor]"')
    second = block.index('uv pip install -p .venv -e ".[coder,voice,monitor]"', first + 1)
    line_end = block.index("\n", second)
    retry_line = block[second:line_end]
    assert "||" in retry_line, "the retry install must be guarded (e.g. `|| true`)"


# ---------------------------------------------------------------------------
# localm.sh: must distinguish "no venv" from "venv present, entry point missing"
# ---------------------------------------------------------------------------


def _run_localm_sh(tmp_path: Path, *, venv_python: bool, venv_localm: bool,
                    localm_args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    if venv_python:
        _make_executable(venv_bin / "python", "#!/bin/sh\nexit 0\n")
    if venv_localm:
        _make_executable(venv_bin / "localm", "#!/bin/sh\nprintf 'ok %s\\n' \"$*\"\n")
    script_copy = tmp_path / "localm.sh"
    script_copy.write_text(LOCALM_SH.read_text(encoding="utf-8"), encoding="utf-8")
    script_copy.chmod(script_copy.stat().st_mode | stat.S_IEXEC)
    args = [_bash(), str(script_copy), *(localm_args or [])]
    return subprocess.run(args, capture_output=True, text=True, cwd=str(tmp_path), timeout=15)


def test_localm_sh_reports_no_venv_when_venv_truly_absent(tmp_path):
    result = _run_localm_sh(tmp_path, venv_python=False, venv_localm=False)
    assert result.returncode == 1
    assert "No .venv found" in result.stderr


def test_localm_sh_distinguishes_missing_entrypoint_from_missing_venv(tmp_path):
    # THE FIX under test: the venv is fine (python present), only the console
    # script is missing - must NOT be misreported as "No .venv found".
    result = _run_localm_sh(tmp_path, venv_python=True, venv_localm=False)
    assert result.returncode == 1
    assert "No .venv found" not in result.stderr
    assert "entry point is missing" in result.stderr


def test_localm_sh_execs_localm_when_both_present(tmp_path):
    result = _run_localm_sh(tmp_path, venv_python=True, venv_localm=True,
                            localm_args=["gui", "--no-model"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok gui --no-model"
