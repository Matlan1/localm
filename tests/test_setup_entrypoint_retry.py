# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.sh's localm-entry-point install/retry block, and localm.sh's own
diagnosis of the same failure mode.

setup.sh retries the `uv pip install` once if `.venv/bin/localm` did not land
(a WSL2/DrvFs quirk where uv reports success but drops exactly one file), then
warns loudly and CONTINUES if it is still missing afterwards, since the rest of
setup does not depend on this entry point. The retry call must be GUARDED: as a
bare command under `set -euo pipefail`, a retry that ERRORS (not just "ran but
did not create the file") kills setup right there, so the loud "STILL missing"
warning a few lines below never prints.

Four branches are covered: present, recovered-by-retry, still-missing-after-
retry, and retry-errors. Extracts and runs ONLY the install/retry block (not the
whole script) against a stub `uv`.

Also covers localm.sh's own diagnosis of this failure mode: it must not report
"No .venv found" when the venv is fine and only the console script is missing.
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


def _heartbeat_functions() -> str:
    # The install block calls heartbeat_start/heartbeat_stop, defined earlier
    # in the real script (near ask()/offer_report()). Extract them the same
    # way _install_block() extracts its own slice, so the synthetic script
    # below has them.
    src = SETUP_SH.read_text(encoding="utf-8")
    start = src.index("HB_SEQ=0")
    stop_def = src.index("heartbeat_stop() {", start)
    end = src.index("\n}\n", stop_def) + len("\n}\n")
    return src[start:end]


def _make_uv_stub(bin_dir: Path) -> None:
    # Records its own call count in a file RELATIVE to cwd (the test sets cwd
    # to tmp_path for the whole run, so both this stub and the extracted
    # setup.sh block agree on where ".venv/bin/localm" and the counter live).
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
        # The real script sets EXTRAS earlier (the browser/app-window prompt,
        # further up the file), which the extracted block references via
        # "${EXTRAS}" without setting it itself - under `set -u` above that is
        # otherwise an unbound-variable error before uv is even reached.
        'EXTRAS="coder,voice,monitor"\n'
        + _heartbeat_functions()
        + _install_block()
        # LOCALM_BIN_OK is set at the tail of the extracted block (right before
        # the "native llama.cpp runtime wheel" marker _install_block() cuts at)
        # - echo it so tests can assert it tracks .venv/bin/localm's real
        # presence, not just that the retry/warning text looks right.
        + '\nprintf "COMPLETED LOCALM_BIN_OK=%s\\n" "$LOCALM_BIN_OK"\n'
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
    assert "LOCALM_BIN_OK=1" in result.stdout


def test_recovered_by_retry(tmp_path):
    result = _run_install_block(tmp_path, call_specs={1: (0, False), 2: (0, True)})
    assert result.returncode == 0, result.stderr
    assert "COMPLETED" in result.stdout
    assert "retrying once" in result.stdout
    assert "STILL missing" not in result.stdout
    assert (tmp_path / ".venv" / "bin" / "localm").exists()
    assert "LOCALM_BIN_OK=1" in result.stdout


def test_still_missing_after_retry_warns_and_continues(tmp_path):
    result = _run_install_block(tmp_path, call_specs={1: (0, False), 2: (0, False)})
    assert result.returncode == 0, result.stderr
    assert "COMPLETED" in result.stdout
    assert "retrying once" in result.stdout
    assert "STILL missing" in result.stdout
    assert not (tmp_path / ".venv" / "bin" / "localm").exists()
    # THE FIX under test: LOCALM_BIN_OK must reflect the real, still-missing
    # state, because setup-llama/make-launcher/plugin-setup downstream branch
    # on it instead of blindly invoking a binary that was just diagnosed absent.
    assert "LOCALM_BIN_OK=0" in result.stdout


def test_retry_itself_erroring_still_reaches_the_still_missing_warning(tmp_path):
    # The retry call is guarded with `|| true`, so a non-zero exit does not
    # kill the extracted block under `set -euo pipefail`: "COMPLETED" and the
    # still-missing warning both print, and the process exit code is 0.
    result = _run_install_block(tmp_path, call_specs={1: (0, False), 2: (7, False)})
    assert result.returncode == 0, result.stderr
    assert "COMPLETED" in result.stdout
    assert "retrying once" in result.stdout
    assert "STILL missing" in result.stdout
    assert "LOCALM_BIN_OK=0" in result.stdout


def test_retry_install_is_guarded_in_source():
    """Static backstop that holds even without bash on PATH: the retry
    install must not be a bare command under `set -euo pipefail`.

    setup.sh builds EXTRAS from a variable (coder,voice,monitor, optionally
    plus desktop) rather than a hardcoded literal, so the source text always
    reads `-e ".[${EXTRAS}]"` at both call sites, never the resolved value.
    """
    block = _install_block()
    needle = 'uv pip install -p .venv -e ".[${EXTRAS}]"'
    first = block.index(needle)
    second = block.index(needle, first + 1)
    line_end = block.index("\n", second)
    retry_line = block[second:line_end]
    assert "||" in retry_line, "the retry install must be guarded (e.g. `|| true`)"


# ---------------------------------------------------------------------------
# setup.sh: the backend-provision (setup-llama) block must not attempt to run
# a .venv/bin/localm that was already diagnosed missing above.
#
# Regression for a second-order break the entry-point fix left in place: once
# .venv/bin/localm is confirmed STILL missing, the script used to continue
# straight into `.venv/bin/localm setup-llama --backend ...` anyway (BACKEND
# is anything but "own" by default, since REC auto-picks a real backend).
# Bash fails that call with "No such file or directory", the surrounding
# `|| { ...; exit 1; }` catches it, and setup.sh exits 1 on a MISLEADING
# "setup-llama failed" message - contradicting the retry block's own comment
# that "the rest of this script does not depend on the entry point". Proven
# live: bash -c './missing-binary setup-llama --backend x || { exit 1; };
# echo unreachable' never prints "unreachable".
# ---------------------------------------------------------------------------


def _provision_block() -> str:
    src = SETUP_SH.read_text(encoding="utf-8")
    start = src.index('if [ "$LOCALM_BIN_OK" != 1 ]; then\n')
    end = src.index("\nfi\n\n# ---- PyTorch + transformers", start)
    return src[start : end + len("\nfi")]


def _make_localm_setup_llama_stub(bin_dir: Path, *, rc: int) -> None:
    # Records each invocation's args (one per line) into a counter file
    # relative to cwd, exactly like _make_uv_stub, so a test can assert the
    # binary was never called at all (the skip path) or called exactly once
    # with the expected args (the normal path).
    _make_executable(
        bin_dir / "localm",
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> .stub-localm-calls\n'
        f"exit {rc}\n",
    )


def _run_provision_block(tmp_path: Path, *, localm_bin_ok: bool, backend: str,
                          stub_rc: int = 0) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    _make_localm_setup_llama_stub(venv_bin, rc=stub_rc)
    script = (
        "set -euo pipefail\n"
        'say() { printf "%s\\n" "$*"; }\n'
        'ask() { printf "%s" "$2"; }\n'  # always answers the prompt's default
        'offer_report() { printf "OFFERED_REPORT\\n"; }\n'
        f'LOCALM_BIN_OK={1 if localm_bin_ok else 0}\n'
        f'BACKEND="{backend}"\n'
        + _provision_block()
        + '\nprintf "COMPLETED\\n"\n'
    )
    env = dict(os.environ)
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    return subprocess.run([_bash(), "-c", script], capture_output=True, text=True,
                          env=env, cwd=str(tmp_path), timeout=15)


def test_provision_skips_without_invoking_localm_when_bin_missing(tmp_path):
    result = _run_provision_block(tmp_path, localm_bin_ok=False, backend="vulkan")
    assert result.returncode == 0, result.stderr
    assert "COMPLETED" in result.stdout
    assert "Skipped - .venv/bin/localm is missing" in result.stdout
    assert "setup-llama failed" not in result.stdout
    # The decisive assertion: the stub must NEVER have been invoked. Before
    # the fix this branch didn't exist and the call below fell straight
    # through to `.venv/bin/localm setup-llama --backend vulkan`.
    assert not (tmp_path / ".stub-localm-calls").exists()


def test_provision_skips_without_invoking_localm_when_bin_missing_own_backend(tmp_path):
    # BACKEND=own used to be asked its own buildpath question first; confirm
    # the missing-binary skip pre-empts that path too, not just the `else`.
    result = _run_provision_block(tmp_path, localm_bin_ok=False, backend="own")
    assert result.returncode == 0, result.stderr
    assert "COMPLETED" in result.stdout
    assert "Skipped - .venv/bin/localm is missing" in result.stdout
    assert not (tmp_path / ".stub-localm-calls").exists()


def test_provision_runs_normally_when_bin_present(tmp_path):
    # Regression protection: the pre-existing, working case must be unchanged.
    result = _run_provision_block(tmp_path, localm_bin_ok=True, backend="vulkan", stub_rc=0)
    assert result.returncode == 0, result.stderr
    assert "COMPLETED" in result.stdout
    assert "Skipped - .venv/bin/localm is missing" not in result.stdout
    calls = (tmp_path / ".stub-localm-calls").read_text().strip()
    assert calls == "setup-llama --backend vulkan"


def test_provision_still_hard_fails_on_a_real_setup_llama_failure(tmp_path):
    # A genuine setup-llama failure (binary present, provisioning itself
    # fails) must still abort setup loudly - only the missing-BINARY case is
    # now soft. Distinguishes "skip" from "the pre-existing hard-fail path".
    result = _run_provision_block(tmp_path, localm_bin_ok=True, backend="vulkan", stub_rc=3)
    assert result.returncode == 1
    assert "setup-llama failed" in result.stdout
    assert "OFFERED_REPORT" in result.stdout
    assert "COMPLETED" not in result.stdout


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
    # The venv is fine (python present), only the console script is missing -
    # it must NOT be reported as "No .venv found".
    result = _run_localm_sh(tmp_path, venv_python=True, venv_localm=False)
    assert result.returncode == 1
    assert "No .venv found" not in result.stderr
    assert "entry point is missing" in result.stderr


def test_localm_sh_execs_localm_when_both_present(tmp_path):
    result = _run_localm_sh(tmp_path, venv_python=True, venv_localm=True,
                            localm_args=["gui", "--no-model"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok gui --no-model"
