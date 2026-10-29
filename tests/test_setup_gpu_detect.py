# SPDX-License-Identifier: AGPL-3.0-or-later
"""setup.sh's GPU recommendation, GPU question and backend menu.

The recommendation comes from `python -m localm.hwdetect` once the venv exists;
detect_gpu() (NVIDIA first, so leftover ROCm tooling on an NVIDIA box never
reports "rocm") is used only when that gives no backend this machine has a build
of. The GPU question follows the recommendation, as in setup.bat. The menu rows
come from `python -m localm.setup_spec backend-menu`.

Extracts and runs only the needed blocks of setup.sh (not the whole script,
which has side effects and prompts) with a controlled PATH holding stub
commands and a stub .venv/bin/python - no real GPU hardware needed either way.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP_SH = ROOT / "setup.sh"


def _bash() -> str | None:
    """Git for Windows ships TWO bash.exe: <root>/bin/bash.exe (a thin shim)
    and <root>/usr/bin/bash.exe (the real MSYS bash). The shim bootstraps its
    own MSYS root's /usr/bin ahead of whatever PATH this test passes via
    subprocess's env=, so a stub placed on an isolated PATH silently loses to
    the real system uname/etc - the real bash does not do this. Which one
    shutil.which("bash") finds first depends on this machine's own PATH
    ordering, so prefer the real bash's sibling path when it exists rather
    than trusting whichever one comes first."""
    found = shutil.which("bash")
    if found is None:
        return None
    shim = Path(found)
    real = shim.parent.parent / "usr" / "bin" / shim.name
    return str(real) if real.is_file() else found


def _function_body(name: str) -> str:
    src = SETUP_SH.read_text(encoding="utf-8")
    start = src.index(f"{name}() {{")
    end = src.index("\n}\n", start) + len("\n}\n")
    return src[start:end]


def _mark_executable(path: Path) -> None:
    """os.chmod()/Path.chmod() cannot set a real POSIX execute bit on NTFS -
    it silently no-ops the S_IEXEC/S_IXGRP/S_IXOTH bits, which some Git-Bash/
    MSYS builds require before they will invoke a shebang script via command
    substitution (others are lenient about it, which is why this was never
    caught locally). Route through bash's own chmod, which does set a bit
    MSYS itself recognizes."""
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    subprocess.run(["chmod", "+x", str(path)], check=True)


def _make_stub(bin_dir: Path, name: str) -> None:
    stub = bin_dir / name
    stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    _mark_executable(stub)


def _run_detect_gpu(tmp_path: Path, *, stub_names: list[str]) -> str:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in stub_names:
        _make_stub(bin_dir, name)
    script = _function_body("detect_gpu") + "\ndetect_gpu\n"
    env = dict(os.environ)
    env["PATH"] = str(bin_dir)   # isolate: no real nvidia-smi/rocminfo must leak in
    result = subprocess.run([_bash(), "-c", script], capture_output=True,
                            text=True, env=env, timeout=15)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


pytestmark = pytest.mark.skipif(_bash() is None, reason="no bash on PATH")


def test_nvidia_wins_over_leftover_rocm_tooling(tmp_path):
    # rocminfo present ALONGSIDE nvidia-smi must still detect nvidia, not rocm.
    assert _run_detect_gpu(tmp_path, stub_names=["nvidia-smi", "rocminfo"]) == "cuda"


def test_nvidia_wins_over_rocm_smi_too(tmp_path):
    assert _run_detect_gpu(tmp_path, stub_names=["nvidia-smi", "rocm-smi"]) == "cuda"


def test_rocm_alone_is_still_detected(tmp_path):
    assert _run_detect_gpu(tmp_path, stub_names=["rocminfo"]) == "rocm"


def test_nvidia_alone_is_still_detected(tmp_path):
    assert _run_detect_gpu(tmp_path, stub_names=["nvidia-smi"]) == "cuda"


def test_neither_is_cpu(tmp_path):
    assert _run_detect_gpu(tmp_path, stub_names=[]) == "cpu"


def test_nvidia_checked_before_rocm_in_source():
    """Static backstop that holds even without bash on PATH: the nvidia-smi
    check must appear (and therefore short-circuit) before the rocm probes."""
    body = _function_body("detect_gpu")
    assert body.index("nvidia-smi") < body.index("rocminfo"), (
        "detect_gpu() must check nvidia-smi before rocminfo/rocm-smi/opt-rocm")


# ---------------------------------------------------------------------------
# Before the venv exists setup.sh asks nothing about the GPU and sets no $GPU:
# the GPU question waits for `python -m localm.hwdetect`, as in setup.bat.
# ---------------------------------------------------------------------------


def _pre_venv_gpu_block() -> str:
    src = SETUP_SH.read_text(encoding="utf-8")
    start = src.index("# ---- GPU probe fallback and platform flags")
    end = src.index("# ---- create the venv", start)
    return src[start:end]


def test_the_pre_venv_block_asks_nothing_even_with_a_gpu_tool_present(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_stub(bin_dir, "nvidia-smi")
    script = (
        'say() { printf "%s\\n" "$*"; }\n'
        'ask() { echo "ASKED $1"; echo Y; }\n'
        "YES=0\nGPU=unset\n"
        + _pre_venv_gpu_block()
        + '\nprintf "GPU=%s\\n" "$GPU"\n'
    )
    env = dict(os.environ)
    env["PATH"] = str(bin_dir)
    result = subprocess.run([_bash(), "-c", script], capture_output=True,
                            text=True, env=env, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "ASKED" not in result.stdout
    assert "GPU=unset" in result.stdout
    assert "Detected acceleration" not in result.stdout


def test_authoritative_recommendation_still_sourced_from_hwdetect():
    """Static lock-in: the one line users should trust as a verdict must stay
    driven by python -m localm.hwdetect, not by detect_gpu()."""
    src = SETUP_SH.read_text(encoding="utf-8")
    assert 'Recommended for your hardware: $REC' in src
    assert "localm.hwdetect" in src


def test_detect_gpu_guess_not_printed_as_its_own_verdict_line_in_source():
    """Static backstop that holds even without bash on PATH."""
    src = SETUP_SH.read_text(encoding="utf-8")
    assert 'say "  Detected acceleration: $GPU"' not in src


def test_no_cpu_override_from_the_shell_probe_remains():
    src = SETUP_SH.read_text(encoding="utf-8")
    assert 'if [ "$GPU" = cpu ]; then REC=cpu; fi' not in src
    assert 'GPU="$(detect_gpu)"' in src
    assert src.index('HWLINE="$(.venv/bin/python -m localm.hwdetect') < src.index(
        'GPU="$(detect_gpu)"')


# ---------------------------------------------------------------------------
# The recommendation and the GPU question (G2). REC comes from
# `python -m localm.hwdetect` ("<vendor> <backend>"); the shell probe
# detect_gpu() is used only when that gives no backend this machine has a
# build of. The GPU question is asked when REC is not cpu; "n" makes it cpu.
# ---------------------------------------------------------------------------


def _write_python_stub(root: Path, *, hw: str | None, hint: str | None,
                       menu: str | None = None) -> None:
    """.venv/bin/python answering hwdetect, setup_spec backend-hint and
    backend-menu with canned text (shell builtins only), recording the
    backend-menu argument in menu_args.txt."""
    bin_dir = root / ".venv" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    lines = ["#!/bin/sh"]
    if hw is not None:
        lines.append(f'if [ "$2" = localm.hwdetect ] && [ -z "$3" ]; then printf "%s\\n" "{hw}"; fi')
    if hint is not None:
        lines.append(f'if [ "$3" = backend-hint ]; then printf "%s\\n" "{hint}"; fi')
    lines.append('if [ "$3" = backend-menu ]; then printf "%s\\n" "$4" >> menu_args.txt; fi')
    if menu is not None:
        (root / "menu.txt").write_text(menu, encoding="utf-8")
        lines.append('if [ "$3" = backend-menu ]; then while IFS= read -r l; do '
                     'printf "%s\\n" "$l"; done < menu.txt; fi')
    lines.append("exit 0")
    stub = bin_dir / "python"
    stub.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _mark_executable(stub)


def _rec_block() -> str:
    src = SETUP_SH.read_text(encoding="utf-8")
    start = src.index('BACKEND_HINT="$(.venv/bin/python -m localm.setup_spec backend-hint')
    end = src.index('# The "provision later" hint.', start)
    return src[start:end]


def _run_rec(tmp_path: Path, *, hw: str | None, hint: str | None = "vulkan|cuda|hip|sycl|cpu",
             answer: str = "Y", yes: bool = False, stub_names: tuple = (),
             os_name: str | None = None, arch: str = "x86_64", is_macos: int = 0) -> str:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in stub_names:
        _make_stub(bin_dir, name)
    if os_name:
        _make_uname_stub(bin_dir, os_name=os_name, arch=arch)
    _write_python_stub(tmp_path, hw=hw, hint=hint)
    script = (
        'say() { printf "%s\\n" "$*"; }\n'
        f'ask() {{ echo "ASKED $1" >&2; echo {answer}; }}\n'
        f'YES={1 if yes else 0}\nIS_MACOS={is_macos}\n'
        + _function_body("detect_gpu")
        + _rec_block()
        + '\nprintf "REC=%s\\n" "$REC"\n'
    )
    env = dict(os.environ)
    env["PATH"] = str(bin_dir)   # isolate: no real nvidia-smi/rocminfo must leak in
    result = subprocess.run([_bash(), "-c", script], capture_output=True,
                            text=True, cwd=tmp_path, env=env, timeout=15)
    assert result.returncode == 0, result.stderr
    return result.stdout + result.stderr


def _rec(out: str) -> str:
    return [ln for ln in out.splitlines() if ln.startswith("REC=")][-1][4:]


def test_hwdetect_gpu_wins_over_a_shell_probe_that_found_none(tmp_path):
    out = _run_rec(tmp_path, hw="amd vulkan", stub_names=())
    assert "ASKED   GPU acceleration looks available (amd). Use it? [Y/n] (n = CPU only): " in out
    assert _rec(out) == "vulkan"


def test_answering_no_to_the_gpu_question_recommends_cpu(tmp_path):
    out = _run_rec(tmp_path, hw="amd vulkan", answer="n")
    assert "ASKED" in out
    assert _rec(out) == "cpu"


def test_yes_mode_asks_nothing_and_keeps_the_gpu_backend(tmp_path):
    out = _run_rec(tmp_path, hw="nvidia cuda", yes=True)
    assert "ASKED" not in out
    assert _rec(out) == "cuda"


def test_a_cpu_recommendation_asks_nothing(tmp_path):
    out = _run_rec(tmp_path, hw="none cpu", stub_names=("nvidia-smi",))
    assert "ASKED" not in out
    assert _rec(out) == "cpu"


def test_no_hwdetect_answer_falls_back_to_the_shell_probe(tmp_path):
    out = _run_rec(tmp_path, hw=None, stub_names=("nvidia-smi",))
    assert "GPU acceleration looks available (cuda)" in out
    assert _rec(out) == "vulkan"


def test_a_backend_with_no_build_here_falls_back_to_the_shell_probe(tmp_path):
    out = _run_rec(tmp_path, hw="nvidia cuda", hint="cpu", stub_names=())
    assert "ASKED" not in out
    assert _rec(out) == "cpu"


def test_no_spec_answer_falls_back_to_the_shell_probe(tmp_path):
    out = _run_rec(tmp_path, hw="amd vulkan", hint=None, stub_names=())
    assert _rec(out) == "cpu"


def test_probe_failed_fallback_metal_gpu_stays_metal(tmp_path):
    out = _run_rec(tmp_path, hw=None, os_name="Darwin", arch="arm64", is_macos=1)
    assert "GPU acceleration looks available (metal)" in out
    assert _rec(out) == "metal"


def test_probe_failed_fallback_on_macos_is_cpu_not_vulkan(tmp_path):
    out = _run_rec(tmp_path, hw=None, stub_names=("rocminfo",), is_macos=1)
    assert _rec(out) == "cpu"


def test_probe_succeeded_metal_is_trusted_directly(tmp_path):
    out = _run_rec(tmp_path, hw="apple metal", hint="metal|cpu", stub_names=())
    assert _rec(out) == "metal"


def test_the_shell_probe_fallback_keeps_metal_before_the_generic_vulkan_case():
    """Static backstop that holds even without bash on PATH."""
    body = _rec_block()
    assert body.index("metal) REC=metal") < body.index("REC=vulkan")


# ---------------------------------------------------------------------------
# The backend menu (G3): rows from `python -m localm.setup_spec backend-menu
# REC`, printed as "    [N] label"; the prompt is "Pick 1-<last> [1]: "; a
# number with no row prints a line and uses REC; no rows: a [!] line and REC.
# ---------------------------------------------------------------------------

ROWS = (
    "1\tvulkan\tvulkan - any GPU, no vendor toolkit   (recommended for your hardware)\n"
    "2\tvulkan\tvulkan - any GPU, no vendor toolkit   (same as [1])\n"
    "3\tcuda\tcuda - NVIDIA, peak performance\n"
    "4\tcpu\tcpu - no GPU\n"
    "5\town\tI will build / provide my own (skip the download)\n"
)


def _backend_menu_block() -> str:
    src = SETUP_SH.read_text(encoding="utf-8")
    start = src.index("# The menu: `python -m localm.setup_spec backend-menu REC`")
    end = src.index('\nif [ "$LOCALM_BIN_OK" != 1 ]; then\n', start)
    return src[start:end]


def _run_menu(tmp_path: Path, *, rec: str, bpick: str, rows: str | None = ROWS,
              env: dict | None = None) -> subprocess.CompletedProcess:
    if rows is not None or env is None:
        _write_python_stub(tmp_path, hw=None, hint=None, menu=rows)
    script = (
        'set -euo pipefail\n'
        'say() { printf "%s\\n" "$*"; }\n'
        f'ask() {{ printf "%s" "$1" >&2; echo "{bpick}"; }}\n'
        f'REC="{rec}"\n'
        + _backend_menu_block()
        + '\nprintf "BACKEND=%s\\n" "$BACKEND"\n'
    )
    result = subprocess.run([_bash(), "-c", script], capture_output=True,
                            text=True, cwd=tmp_path, timeout=60, env=env)
    assert result.returncode == 0, result.stderr
    return result


def _backend(result) -> str:
    return result.stdout.strip().splitlines()[-1].split("=", 1)[1]


def test_every_spec_row_is_listed_with_its_label(tmp_path):
    r = _run_menu(tmp_path, rec="vulkan", bpick="1")
    for line in ROWS.splitlines():
        n, _key, label = line.split("\t")
        assert f"    [{n}] {label}" in r.stdout
    assert "Pick 1-5 [1]: " in r.stderr
    assert "Recommended for your hardware: vulkan" in r.stdout
    assert _backend(r) == "vulkan"


def test_the_recommendation_is_passed_to_the_spec(tmp_path):
    _run_menu(tmp_path, rec="cuda", bpick="1")
    assert (tmp_path / "menu_args.txt").read_text(encoding="utf-8").split() == ["cuda"]


@pytest.mark.parametrize("bpick, backend", [("2", "vulkan"), ("3", "cuda"), ("4", "cpu"),
                                            ("5", "own")])
def test_a_listed_number_picks_its_backend(tmp_path, bpick, backend):
    r = _run_menu(tmp_path, rec="vulkan", bpick=bpick)
    assert _backend(r) == backend
    assert "is not offered" not in r.stdout


@pytest.mark.parametrize("bpick", ["6", "8", "11"])
def test_an_unlisted_number_says_what_it_used_instead(tmp_path, bpick):
    r = _run_menu(tmp_path, rec="cuda", bpick=bpick)
    assert f"  [{bpick}] is not offered on this machine; using cuda." in r.stdout
    assert _backend(r) == "cuda"


def test_a_junk_answer_takes_the_recommendation_silently(tmp_path):
    r = _run_menu(tmp_path, rec="cuda", bpick="zz")
    assert "is not offered" not in r.stdout
    assert _backend(r) == "cuda"


def test_no_rows_from_the_spec_says_so_and_uses_the_recommendation(tmp_path):
    r = _run_menu(tmp_path, rec="metal", bpick="3", rows=None)
    assert "  [!] Could not list the llama.cpp backends for this machine; using metal." in r.stdout
    assert "Pick" not in r.stderr
    assert _backend(r) == "metal"


def test_the_load_tested_note_is_kept(tmp_path):
    r = _run_menu(tmp_path, rec="vulkan", bpick="1")
    assert "(your pick is load-tested; on failure you can retry after fixing the" in r.stdout


def test_the_real_spec_menu_parses(tmp_path):
    """The real interpreter and the real localm.setup_spec behind
    .venv/bin/python: every row is listed, and the last number picks "own"."""
    from localm import setup_spec

    bin_dir = tmp_path / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    exe = Path(sys.executable).as_posix()
    stub = bin_dir / "python"
    stub.write_text(f'#!/bin/sh\nexec "{exe}" "$@"\n', encoding="utf-8")
    _mark_executable(stub)
    rows = setup_spec.backend_menu("cpu")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = _run_menu(tmp_path, rec="cpu", bpick=rows[-1][0], rows=None, env=env)
    for n, _key, label in rows:
        assert f"    [{n}] {label}" in r.stdout, r.stdout
    assert f"Pick 1-{rows[-1][0]} [1]: " in r.stderr
    assert _backend(r) == "own"


def test_no_backend_list_is_hardcoded_in_the_menu():
    src = SETUP_SH.read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    for gone in ("hwdetect backends", "_m2", "_m8", "for _b in", "vulkan|cuda|hip",
                 'AVAIL="vulkan', "if offered "):
        assert gone not in code, gone
    assert "localm.setup_spec backend-menu" in code


# ---------------------------------------------------------------------------
# Apple Silicon: detect_gpu() must recognize Darwin/arm64 so an Apple Silicon
# machine is actually offered the metal backend through the normal
# automatic/interactive path, instead of always falling through to "cpu"
# (detect_gpu() had no Darwin branch at all - every real Mac silently landed
# on the CPU-only build, with the correct hwdetect.py "metal" recommendation
# clobbered by the "$GPU = cpu -> REC=cpu" guard further down in setup.sh).
# ---------------------------------------------------------------------------


def _make_uname_stub(bin_dir: Path, *, os_name: str, arch: str) -> None:
    stub = bin_dir / "uname"
    stub.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "-s" ]; then echo {os_name}; fi\n'
        f'if [ "$1" = "-m" ]; then echo {arch}; fi\n',
        encoding="utf-8",
    )
    _mark_executable(stub)


def _run_detect_gpu_with_uname(tmp_path: Path, *, os_name: str, arch: str,
                                stub_names: list[str] | None = None) -> str:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in stub_names or []:
        _make_stub(bin_dir, name)
    _make_uname_stub(bin_dir, os_name=os_name, arch=arch)
    script = _function_body("detect_gpu") + "\ndetect_gpu\n"
    env = dict(os.environ)
    env["PATH"] = str(bin_dir)   # isolate: no real nvidia-smi/rocminfo/uname must leak in
    result = subprocess.run([_bash(), "-c", script], capture_output=True,
                            text=True, env=env, timeout=15)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_apple_silicon_is_metal(tmp_path):
    assert _run_detect_gpu_with_uname(tmp_path, os_name="Darwin", arch="arm64") == "metal"


def test_intel_mac_is_cpu_not_metal(tmp_path):
    # Matches hwdetect.py's own policy: Intel Macs get "cpu", not metal - the
    # official llama.cpp macOS Metal build targets Apple Silicon.
    assert _run_detect_gpu_with_uname(tmp_path, os_name="Darwin", arch="x86_64") == "cpu"


def test_linux_uname_never_mistaken_for_darwin(tmp_path):
    assert _run_detect_gpu_with_uname(tmp_path, os_name="Linux", arch="x86_64") == "cpu"


def test_nvidia_still_wins_over_a_darwin_arm64_uname(tmp_path):
    # Priority order: a real vendor tool detected earlier in detect_gpu() must
    # still win even if uname would otherwise match the Darwin/arm64 branch
    # (not reachable on real hardware - Apple Silicon has no nvidia-smi - but
    # this pins that the Darwin check is an elif appended AFTER the vendor
    # checks, not a check that could ever shadow them).
    assert _run_detect_gpu_with_uname(
        tmp_path, os_name="Darwin", arch="arm64", stub_names=["nvidia-smi"]
    ) == "cuda"


def test_darwin_branch_checked_after_vendor_tools_in_source():
    """Static backstop that holds even without bash on PATH."""
    body = _function_body("detect_gpu")
    assert body.index("nvidia-smi") < body.index('"Darwin"')
    assert body.index("rocminfo") < body.index('"Darwin"')
