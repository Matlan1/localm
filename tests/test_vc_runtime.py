# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Microsoft Visual C++ runtime on Windows (localm.vc_runtime): the check of
the system folder, the offer to run Microsoft's installer and every path on
which it does not run, the pinned and signed download, setup-llama's one
placement path, the exit code setup reads, and the load error that names a
missing DLL.

The DLLs are real PE images in their real format, built here."""

from __future__ import annotations

import ast
import hashlib
import re
import struct
import sys
from pathlib import Path

import pytest

from localm import pe_deps, setup_spec, vc_runtime
from localm import setup_llama as sl
from localm.setup_llama import load_probe

VERSION = vc_runtime.REDIST_VERSION
_REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
#  Builders                                                                   #
# --------------------------------------------------------------------------- #

def _pe(imports: list[str], version: tuple | None = None, machine: int = 0x8664) -> bytes:
    """A minimal PE32+ image importing *imports*, with a VS_FIXEDFILEINFO of
    *version* after its import section when given."""
    opt_size, dirs_at, va, raw = 240, 112, 0x1000, 0x400
    names, name_rva = b"", {}
    strings_at = 20 * (len(imports) + 1)
    for n in imports:
        name_rva[n] = va + strings_at + len(names)
        names += n.encode() + b"\0"
    body = b"".join(struct.pack("<IIIII", 0, 0, 0, name_rva[n], 0) for n in imports)
    body += b"\0" * 20 + names
    opt = bytearray(opt_size)
    struct.pack_into("<H", opt, 0, 0x20B)
    struct.pack_into("<I", opt, dirs_at - 4, 16)
    struct.pack_into("<II", opt, dirs_at + 8, va, 20 * (len(imports) + 1))
    section = b".idata\0\0" + struct.pack("<IIII", len(body), va, len(body), raw) + b"\0" * 16
    head = bytearray(0x40)
    head[:2] = b"MZ"
    struct.pack_into("<I", head, 0x3C, 0x40)
    image = bytes(head) + b"PE\0\0" + struct.pack("<HHIIIHH", machine, 1, 0, 0, 0, opt_size, 0)
    image += bytes(opt) + section
    image = image + b"\0" * (raw - len(image)) + body
    if version is not None:
        a, b, c, d = version
        image += b"\0" * 8 + b"\xbd\x04\xef\xfe" + struct.pack(
            "<IIII", 0x10000, (a << 16) | b, (c << 16) | d, 0)
    return image


def _runtime_folder(tmp_path: Path, imports: list[str]) -> Path:
    folder = tmp_path / "lib"
    folder.mkdir()
    (folder / "llama.dll").write_bytes(_pe(imports))
    (folder / "ggml-base.dll").write_bytes(_pe(["MSVCP140.dll", "KERNEL32.dll"]))
    return folder


_PACKAGE = b"MZ the installer stand-in" * 100


class _Rig:
    """A runtime folder, a stand-in system folder, and recorders for every step
    :func:`vc_runtime.ensure` can take."""

    def __init__(self, tmp_path: Path, monkeypatch):
        self.lib = _runtime_folder(tmp_path, ["MSVCP140.dll", "VCRUNTIME140_1.dll"])
        self.system = tmp_path / "system32"
        self.system.mkdir()
        self.downloads: list[str] = []
        self.runs: list[tuple] = []
        self.questions: list[str] = []
        self.said: list[str] = []
        self.answer = True
        self.exit_code = 0
        self.installs = True
        self.package = _PACKAGE
        self.refused = None
        url, _sha, _size = vc_runtime._REDIST["amd64"]
        monkeypatch.setitem(vc_runtime._REDIST, "amd64", (
            url, hashlib.sha256(_PACKAGE).hexdigest(), len(_PACKAGE)))
        monkeypatch.setattr(vc_runtime, "authenticode_signer", lambda p: vc_runtime.SIGNER)
        monkeypatch.setattr(vc_runtime, "run_installer", self._run)
        monkeypatch.setattr(vc_runtime, "_hold_open", lambda p: None)
        monkeypatch.setitem(vc_runtime._STATE, "declined", False)

    def download(self, url: str, dest: Path) -> None:
        self.downloads.append(url)
        Path(dest).write_bytes(self.package)

    def ask(self, text: str) -> bool:
        self.questions.append(text)
        return self.answer

    def _run(self, path, args=vc_runtime.INSTALL_ARGS):
        self.runs.append((Path(path).name, tuple(args), Path(path).read_bytes() == _PACKAGE))
        if self.installs:
            for name in ("msvcp140.dll", "vcruntime140_1.dll"):
                (self.system / name).write_bytes(_pe(["KERNEL32.dll"], VERSION))
        return self.exit_code

    def ensure(self, *, ask="default") -> bool:
        return vc_runtime.ensure(self.lib, download=self.download,
                                 refusal=lambda url: self.refused,
                                 ask=self.ask if ask == "default" else ask,
                                 say=self.said.append, system_dir=self.system,
                                 machine="AMD64")


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    return _Rig(tmp_path, monkeypatch)


# --------------------------------------------------------------------------- #
#  Detection                                                                  #
# --------------------------------------------------------------------------- #

def test_missing_from_system_counts_absent_and_older_copies(tmp_path):
    folder = _runtime_folder(tmp_path, ["MSVCP140.dll", "VCRUNTIME140.dll", "KERNEL32.dll"])
    system = tmp_path / "system"
    system.mkdir()
    assert vc_runtime.missing_from_system(folder, system) == ["msvcp140.dll",
                                                              "vcruntime140.dll"]
    (system / "msvcp140.dll").write_bytes(_pe(["KERNEL32.dll"], (14, 29, 30133, 0)))
    (system / "vcruntime140.dll").write_bytes(_pe(["KERNEL32.dll"], VERSION))
    assert vc_runtime.missing_from_system(folder, system) == ["msvcp140.dll"]
    (system / "msvcp140.dll").write_bytes(_pe(["KERNEL32.dll"], (14, 60, 1, 0)))
    assert vc_runtime.missing_from_system(folder, system) == []


def test_absent_from_system_counts_only_missing_files(tmp_path):
    folder = _runtime_folder(tmp_path, ["MSVCP140.dll", "VCRUNTIME140.dll"])
    system = tmp_path / "system"
    system.mkdir()
    (system / "msvcp140.dll").write_bytes(_pe(["KERNEL32.dll"], (14, 29, 30133, 0)))
    assert vc_runtime.absent_from_system(folder, system) == ["vcruntime140.dll"]
    assert vc_runtime.missing_from_system(folder, system) == ["msvcp140.dll", "vcruntime140.dll"]


def test_nothing_is_missing_when_nothing_imports_the_runtime(tmp_path):
    folder = tmp_path / "lib"
    folder.mkdir()
    (folder / "llama.dll").write_bytes(_pe(["KERNEL32.dll", "vulkan-1.dll"]))
    assert vc_runtime.missing_from_system(folder, None) == []


def test_off_windows_nothing_is_missing_and_nothing_is_offered(tmp_path, monkeypatch, rig):
    monkeypatch.setattr(vc_runtime.sys, "platform", "linux")
    assert vc_runtime.missing_from_system(rig.lib) == []
    assert vc_runtime.ensure(rig.lib, download=rig.download, ask=rig.ask,
                             refusal=lambda url: None, say=rig.said.append) is True
    assert rig.questions == rig.downloads == rig.runs == rig.said == []


@pytest.mark.parametrize("machine,key", [("AMD64", "amd64"), ("x86_64", "amd64"),
                                         ("ARM64", "arm64"), ("x86", None)])
def test_redist_machine(machine, key):
    assert vc_runtime.redist_machine(machine) == key


def test_every_pin_is_microsofts_versioned_download():
    for url, sha, size in vc_runtime._REDIST.values():
        assert url.startswith(f"https://{vc_runtime.REDIST_HOST}/download/pr/")
        assert sha.upper() in url and len(sha) == 64 and size > 1_000_000


def test_the_installer_runs_quietly_without_restarting_windows():
    assert vc_runtime.INSTALL_ARGS == ("/install", "/quiet", "/norestart")


# --------------------------------------------------------------------------- #
#  The offer                                                                  #
# --------------------------------------------------------------------------- #

def test_a_present_runtime_asks_and_downloads_nothing(rig):
    for name in ("msvcp140.dll", "vcruntime140_1.dll"):
        (rig.system / name).write_bytes(_pe(["KERNEL32.dll"], VERSION))
    assert rig.ensure() is True
    assert rig.questions == rig.downloads == rig.runs == rig.said == []


def test_yes_downloads_checks_runs_and_checks_again(rig):
    assert rig.ensure() is True
    assert len(rig.questions) == 1
    q = rig.questions[0]
    assert "no model can load without it" in q and "one administrator prompt" in q
    assert "msvcp140.dll, vcruntime140_1.dll" in q and vc_runtime.REDIST_HOST in q
    assert rig.downloads == [vc_runtime._REDIST["amd64"][0]]
    assert rig.runs == [("VC_redist.x64.exe", vc_runtime.INSTALL_ARGS, True)]
    assert rig.said[-1] == "The Microsoft Visual C++ Redistributable is installed."
    assert sorted(p.name for p in rig.lib.iterdir()) == ["ggml-base.dll", "llama.dll"]


def test_an_older_runtime_is_named_as_older(rig):
    (rig.system / "msvcp140.dll").write_bytes(_pe(["KERNEL32.dll"], (14, 29, 30133, 0)))
    rig.answer = False
    assert rig.ensure() is False
    assert "has an older version of" in rig.questions[0]
    assert "may fail to load" in rig.questions[0]
    assert rig.said[-1].startswith("Models may fail to load or stop with an error:")
    assert "has an older version of" in rig.said[-1]


def _assert_fallback(text: str, reason: str) -> None:
    assert text.startswith("No model can load yet:")
    assert f"({reason}" in text
    assert vc_runtime.REDIST_PAGE in text and vc_runtime.redist_url() in text
    assert f"'{vc_runtime.RERUN_HINT}'" in text


def test_declined_downloads_nothing_and_is_not_asked_again(rig):
    rig.answer = False
    assert rig.ensure() is False
    assert rig.downloads == rig.runs == []
    _assert_fallback(rig.said[-1], "you declined it")
    assert rig.ensure() is False
    assert len(rig.questions) == 1
    _assert_fallback(rig.said[-1], "you declined it earlier in this run")


def test_an_unattended_run_never_downloads_or_runs_the_installer(rig):
    assert rig.ensure(ask=None) is False
    assert rig.questions == rig.downloads == rig.runs == []
    _assert_fallback(rig.said[-1], "this run is unattended")


def test_network_off_refuses_before_asking(rig):
    rig.refused = "Network access is off."
    assert rig.ensure() is False
    assert rig.questions == rig.downloads == rig.runs == []
    _assert_fallback(rig.said[-1], "Network access is off")
    assert "off.)" not in rig.said[-1]


def test_setup_llama_refuses_the_download_with_network_access_off(monkeypatch):
    nd = sl.native_deps
    monkeypatch.setattr(nd, "_net_allowed", lambda: False)
    refusal = nd._vc_refusal(vc_runtime._REDIST["amd64"][0]) or ""
    assert "Visual C++ Redistributable was not downloaded" in refusal
    assert "net_allow_model_downloads" in refusal
    monkeypatch.setattr(nd, "_net_allowed", lambda: True)
    assert nd._vc_refusal(vc_runtime._REDIST["amd64"][0]) is None

def test_a_corrupted_download_is_refused_and_never_runs(rig):
    rig.package = _PACKAGE[:-1] + b"!"
    assert rig.ensure() is False
    assert rig.runs == []
    _assert_fallback(rig.said[-1], "the download's sha256 is")


def test_a_download_of_the_wrong_size_is_refused(rig):
    rig.package = _PACKAGE + b"x"
    assert rig.ensure() is False
    assert rig.runs == []
    assert "bytes, expected" in rig.said[-1]


def test_a_package_not_signed_by_microsoft_never_runs(rig, monkeypatch):
    monkeypatch.setattr(vc_runtime, "authenticode_signer", lambda p: "Someone Else")
    assert rig.ensure() is False
    assert rig.runs == []
    assert "signed by 'Someone Else'" in rig.said[-1]


def test_an_invalid_signature_never_runs(rig, monkeypatch):
    def invalid(path):
        raise vc_runtime.RedistError("the installer's signature is not valid "
                                     "(WinVerifyTrust 0x80096010)")
    monkeypatch.setattr(vc_runtime, "authenticode_signer", invalid)
    assert rig.ensure() is False
    assert rig.runs == []
    assert "0x80096010" in rig.said[-1]


def test_a_failed_download_ends_in_the_message(rig):
    def broken(url, dest):
        raise OSError("connection reset")
    rig.download = broken
    assert rig.ensure() is False
    assert rig.runs == []
    _assert_fallback(rig.said[-1], "connection reset")


@pytest.mark.parametrize("code,reason", [
    (vc_runtime.ERROR_CANCELLED, "the administrator prompt was declined"),
    (vc_runtime.ERROR_INSTALL_USEREXIT, "the installation was cancelled"),
    (vc_runtime.ERROR_INSTALL_FAILURE, "the installer failed (code 1603"),
    (vc_runtime.ERROR_INSTALL_ALREADY_RUNNING, "another installation is in progress"),
    (vc_runtime.WAIT_TIMEOUT, "the installer was still running after 20 minutes"),
    (1714, "the installer exited with code 1714"),
    (vc_runtime.ERROR_SUCCESS_REBOOT_REQUIRED, "Windows needs a restart"),
    (vc_runtime.ERROR_SUCCESS, "the installer finished but the system folder still lacks it"),
])
def test_every_install_that_leaves_the_runtime_missing_says_why(rig, code, reason):
    rig.exit_code, rig.installs = code, False
    assert rig.ensure() is False
    assert len(rig.runs) == 1
    _assert_fallback(rig.said[-1], reason)


def test_a_restart_request_with_the_runtime_in_place_counts_as_installed(rig):
    rig.exit_code = vc_runtime.ERROR_SUCCESS_REBOOT_REQUIRED
    assert rig.ensure() is True
    assert "restart" in rig.said[-1] and "models can load now" in rig.said[-1]


def test_a_newer_version_already_installed_counts_when_the_system_has_it(rig):
    rig.exit_code = vc_runtime.ERROR_PRODUCT_VERSION
    assert rig.ensure() is True


def test_no_pinned_installer_for_the_machine_downloads_nothing(rig):
    assert vc_runtime.ensure(rig.lib, download=rig.download, ask=rig.ask,
                             refusal=lambda url: None, say=rig.said.append,
                             system_dir=rig.system, machine="x86") is False
    assert rig.questions == rig.downloads == []
    assert "no installer is pinned for x86" in rig.said[-1]


def test_installer_helpers_refuse_off_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(vc_runtime.sys, "platform", "linux")
    with pytest.raises(vc_runtime.RedistError):
        vc_runtime.authenticode_signer(tmp_path / "x.exe")
    with pytest.raises(vc_runtime.RedistError):
        vc_runtime.run_installer(tmp_path / "x.exe")


@pytest.mark.skipif(sys.platform != "win32", reason="WinVerifyTrust is Windows-only")
def test_an_unsigned_file_has_no_valid_signature(tmp_path):
    exe = tmp_path / "unsigned.exe"
    exe.write_bytes(_pe(["KERNEL32.dll"]))
    with pytest.raises(vc_runtime.RedistError, match="signature is not valid"):
        vc_runtime.authenticode_signer(exe)


# --------------------------------------------------------------------------- #
#  setup-llama: one placement path, the keep path and the exit code           #
# --------------------------------------------------------------------------- #

def _setup_llama_rig(rig, monkeypatch, *, interactive=True, answer=True):
    monkeypatch.setattr(sl.sys, "platform", "win32")
    monkeypatch.setattr(vc_runtime.sys, "platform", "win32")
    monkeypatch.setattr(vc_runtime, "_system_dir", lambda: rig.system)
    monkeypatch.setattr(vc_runtime, "redist_machine", lambda machine=None: "amd64")
    monkeypatch.setattr(sl, "_download", rig.download)
    monkeypatch.setattr(sl, "_interactive", lambda: interactive)
    monkeypatch.setattr(sl, "_flush_stdin", lambda: None)
    asked: list[str] = []
    monkeypatch.setattr(sl, "_confirm", lambda text, default: asked.append(text) or answer)
    monkeypatch.setattr(sl.native_deps, "_VC_UNATTENDED", {"on": False})
    return asked


def test_bundling_on_windows_offers_through_setup_llamas_download(rig, monkeypatch):
    asked = _setup_llama_rig(rig, monkeypatch)
    sl._bundle_missing_native_deps(rig.lib)
    assert len(asked) == 1 and rig.downloads and len(rig.runs) == 1
    assert sl.vc_runtime_missing(rig.lib) is False


def test_yes_flag_never_offers(rig, monkeypatch):
    asked = _setup_llama_rig(rig, monkeypatch)
    sl.set_vc_unattended(True)
    assert sl.offer_vc_runtime(rig.lib) is False
    assert asked == [] and rig.downloads == [] and rig.runs == []
    assert sl.vc_runtime_missing(rig.lib) is True


def test_no_terminal_never_offers(rig, monkeypatch):
    asked = _setup_llama_rig(rig, monkeypatch, interactive=False)
    assert sl.offer_vc_runtime(rig.lib) is False
    assert asked == [] and rig.runs == []


def test_bundling_on_windows_never_raises(rig, monkeypatch, caplog):
    _setup_llama_rig(rig, monkeypatch)

    def boom(*a, **k):
        raise ValueError("unexpected")
    monkeypatch.setattr(vc_runtime, "ensure", boom)
    sl._bundle_missing_native_deps(rig.lib)
    assert any("Visual C++ Redistributable check failed" in r.message
               for r in caplog.records)


def test_replace_target_bundles_into_staging_before_the_swap(tmp_path, monkeypatch):
    target = tmp_path / "lib"
    seen = []
    monkeypatch.setattr(sl, "_bundle_missing_native_deps",
                        lambda d: seen.append((d.name, sorted(p.name for p in d.iterdir()))))

    def fill(staging):
        (staging / "llama.dll").write_bytes(b"x")
        return "tag"
    assert sl._replace_target(target, fill) == "tag"
    assert seen == [(sl._staging_path(target).name, ["llama.dll"])]
    assert (target / "llama.dll").is_file()


def test_only_replace_target_calls_the_bundler():
    """Every way of placing a runtime goes through _replace_target, so the
    bundler is called there and nowhere else in the package."""
    root = Path(sl.__file__).parent
    callers = []
    for f in sorted(root.glob("*.py")):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_bundle_missing_native_deps"):
                callers.append(f.name)
    assert callers == ["runtime_dir.py"]


def test_vc_runtime_missing_skips_the_backend_fallback(tmp_path, monkeypatch):
    from localm.setup_llama import provision
    target = tmp_path / "lib"
    target.mkdir()
    fell_back: list[str] = []
    monkeypatch.setattr(sl, "_replace_target", lambda t, fill: "b9999")
    monkeypatch.setattr(sl, "_install_runtime_wheel", lambda d: True)
    monkeypatch.setattr(sl, "_runtime_pkg_dir", lambda: tmp_path / "pkg")
    monkeypatch.setattr(provision, "_chosen_backend_loads", lambda c: (False, "no load"))
    monkeypatch.setattr(sl, "vc_runtime_missing", lambda t: True)
    monkeypatch.setattr(provision, "_fall_back_to_universal",
                        lambda *a, **k: fell_back.append("universal") or ("vulkan", None))
    monkeypatch.setattr(provision, "_universal_backend_failed",
                        lambda *a, **k: fell_back.append("terminal") or ("cpu", None))
    assert provision._provision_with_fallback("cuda", target, None, False) == ("cuda", "b9999")
    assert provision._provision_with_fallback("cpu", target, None, False) == ("cpu", "b9999")
    assert fell_back == []
    monkeypatch.setattr(sl, "vc_runtime_missing", lambda t: False)
    provision._provision_with_fallback("cuda", target, None, False)
    assert fell_back == ["universal"]


def _run_main(tmp_path, monkeypatch, *, vc_missing: bool, loads: bool = False,
              args=("--backend", "cpu")):
    from click.testing import CliRunner
    from localm.setup_llama import cli
    target = tmp_path / "runtime" / "lib"
    verified: list[bool] = []
    unattended: list[bool] = []
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: target)
    monkeypatch.setattr(sl, "set_vc_unattended", unattended.append)
    monkeypatch.setattr(cli, "_require_backend_here", lambda b: None)
    monkeypatch.setattr(cli, "_refuse_when_net_off", lambda: None)
    monkeypatch.setattr(cli, "_apply_version_request", lambda *a: None)
    monkeypatch.setattr(cli, "_stored_pin", lambda: None)
    monkeypatch.setattr(cli, "_keeps_existing_install", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_provision_release", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_refresh_install_record", lambda t: None)
    monkeypatch.setattr(sl, "vc_runtime_missing", lambda t: vc_missing)
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: (loads, "" if loads else "no load"))
    monkeypatch.setattr(sl, "_verify", lambda: verified.append(True))
    result = CliRunner().invoke(sl.main, list(args))
    return result, verified, unattended


def test_setup_llama_exits_3_without_success_when_the_runtime_stays_missing(
        tmp_path, monkeypatch):
    result, verified, unattended = _run_main(tmp_path, monkeypatch, vc_missing=True,
                                             args=("--backend", "cpu", "--yes"))
    assert result.exit_code == setup_spec.VC_RUNTIME_ABSENT_EXIT == 3
    assert verified == []
    assert unattended == [True]
    assert "Native runtime ready" not in result.output


def test_setup_llama_succeeds_with_an_older_runtime_that_loads(tmp_path, monkeypatch):
    result, verified, _ = _run_main(tmp_path, monkeypatch, vc_missing=True, loads=True)
    assert result.exit_code == 0, result.output
    assert verified == [True]


def test_setup_llama_verifies_when_the_runtime_is_present(tmp_path, monkeypatch):
    result, verified, unattended = _run_main(tmp_path, monkeypatch, vc_missing=False)
    assert result.exit_code == 0, result.output
    assert verified == [True]
    assert unattended == [False]


def test_a_kept_install_offers_again_and_load_tests(tmp_path, monkeypatch):
    from localm.setup_llama import cli
    offered: list[Path] = []
    state = {"missing": True}
    monkeypatch.setattr(sl, "vc_runtime_missing", lambda t: state["missing"])

    def offer(t):
        offered.append(t)
        state["missing"] = False
        return True
    monkeypatch.setattr(sl, "offer_vc_runtime", offer)
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: (True, ""))
    cli._vc_runtime_for_kept_install(tmp_path, "cpu")
    assert offered == [tmp_path]


def test_a_kept_install_without_the_runtime_exits_3(tmp_path, monkeypatch):
    from localm.setup_llama import cli
    monkeypatch.setattr(sl, "vc_runtime_missing", lambda t: True)
    monkeypatch.setattr(sl, "offer_vc_runtime", lambda t: False)
    loads: list[bool] = []
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: loads.append(True) or (False, "x"))
    with pytest.raises(SystemExit) as e:
        cli._vc_runtime_for_kept_install(tmp_path, "cpu")
    assert e.value.code == setup_spec.VC_RUNTIME_ABSENT_EXIT
    assert loads == [True]


def test_a_kept_install_with_an_older_runtime_that_loads_is_kept(tmp_path, monkeypatch):
    from localm.setup_llama import cli
    monkeypatch.setattr(sl, "vc_runtime_missing", lambda t: True)
    monkeypatch.setattr(sl, "offer_vc_runtime", lambda t: False)
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: (True, ""))
    cli._vc_runtime_for_kept_install(tmp_path, "cpu")


def test_a_kept_install_with_the_runtime_present_offers_nothing(tmp_path, monkeypatch):
    from localm.setup_llama import cli
    monkeypatch.setattr(sl, "vc_runtime_missing", lambda t: False)
    monkeypatch.setattr(sl, "offer_vc_runtime",
                        lambda t: pytest.fail("offered with the runtime present"))
    cli._vc_runtime_for_kept_install(tmp_path, "cpu")


def test_the_keep_path_calls_the_offer():
    from localm.setup_llama import cli
    src = Path(cli.__file__).read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_keeps_existing_install")
    calls = [n.func.id for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "_vc_runtime_for_kept_install" in calls


# --------------------------------------------------------------------------- #
#  Every setup variant reads exit 3 as "runtime absent", not as a failure      #
# --------------------------------------------------------------------------- #

def test_setup_bat_reads_exit_3_after_every_setup_llama_run():
    lines = (_REPO / "setup.bat").read_text(encoding="utf-8").splitlines()
    runs = [i for i, line in enumerate(lines)
            if re.match(r"\s*\.venv\\Scripts\\localm setup-llama ", line)]
    assert len(runs) == 2
    for i in runs:
        assert lines[i + 1].strip() == 'if "!errorlevel!"=="3" ('
        block = "\n".join(lines[i + 2:i + 6])
        assert 'set "VCMISSING=1"' in block and 'set "RUNTIME_OK=0"' in block
        assert "provision_failed" not in block


def test_the_gui_installer_reads_exit_3_after_every_setup_llama_run():
    src = (_REPO / "installer" / "gui.py").read_text(encoding="utf-8")
    runs = [m.start() for m in re.finditer(r'_run\(\[localm_exe, "setup-llama"', src)]
    assert len(runs) == 2
    for at in runs:
        after = src[at:at + 400]
        assert "if code == spec.VC_RUNTIME_ABSENT_EXIT:" in after
        assert after.index("VC_RUNTIME_ABSENT_EXIT") < after.index("runtime_failed(")


def test_the_gui_installer_step_names_the_remedy_without_a_report():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "localm_installer_gui_vc", _REPO / "installer" / "gui.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
        with pytest.raises(mod.StepFailed) as e:
            mod.vc_runtime_absent("localm")
    finally:
        sys.modules.pop(spec.name, None)
    text = str(e.value)
    assert "Visual C++ Redistributable" in text and '"localm" setup-llama' in text
    assert vc_runtime.REDIST_PAGE.replace("https://", "") in text


# --------------------------------------------------------------------------- #
#  stable-diffusion.cpp                                                       #
# --------------------------------------------------------------------------- #

def test_sdcpp_install_without_the_runtime_raises_and_tries_no_other_backend(monkeypatch):
    from localm.media.sdcpp import runtime as sd
    tried: list[str] = []

    def install(b, **kw):
        tried.append(b)
        raise sd.VcRuntimeMissingError("needs the Microsoft Visual C++ Redistributable")
    monkeypatch.setattr(sd, "install", install)
    monkeypatch.setattr(sd, "platform_key", lambda: "windows")
    monkeypatch.setattr(sd, "recommended_backend", lambda: "vulkan")
    monkeypatch.setattr(sd, "installed", lambda b: None)
    monkeypatch.setattr(sd, "load_test_failed", lambda b: None)
    with pytest.raises(sd.VcRuntimeMissingError):
        sd.provision("auto", force=True)
    assert tried == ["vulkan"]


def test_sdcpp_requires_the_runtime_and_passes_its_question(tmp_path, monkeypatch):
    from localm.media.sdcpp import runtime as sd
    seen = {}

    def ensure(folder, *, download, refusal, ask, say):
        seen["ask"], seen["refusal"] = ask, refusal
        say("No model can load yet: ...")
        return False
    monkeypatch.setattr(vc_runtime, "ensure", ensure)
    said: list[str] = []

    def ask(text):
        return False
    monkeypatch.setattr(vc_runtime, "absent_from_system", lambda folder: ["msvcp140.dll"])
    with pytest.raises(sd.VcRuntimeMissingError) as e:
        sd._require_vc_runtime(tmp_path, said.append, ask)
    from localm import netpolicy
    assert seen["ask"] is ask and said
    assert seen["refusal"] is netpolicy.download_refusal
    assert vc_runtime.RERUN_HINT in str(e.value)
    monkeypatch.setattr(vc_runtime, "absent_from_system", lambda folder: [])
    sd._require_vc_runtime(tmp_path, said.append, None)
    monkeypatch.setattr(vc_runtime, "ensure", lambda folder, **k: True)
    monkeypatch.setattr(vc_runtime, "absent_from_system", lambda folder: ["msvcp140.dll"])
    sd._require_vc_runtime(tmp_path, said.append, None)


# --------------------------------------------------------------------------- #
#  Naming a missing DLL                                                       #
# --------------------------------------------------------------------------- #

def test_unresolved_imports_follows_dependencies_in_the_search_dirs(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "llama.dll").write_bytes(_pe(["ggml.dll", "MSVCP140.dll", "api-ms-win-crt-runtime-l1-1-0.dll"]))
    (lib / "ggml.dll").write_bytes(_pe(["vulkan-1.dll", "KERNEL32.dll"]))
    system = tmp_path / "sys"
    system.mkdir()
    (system / "kernel32.dll").write_bytes(b"")
    missing = pe_deps.unresolved_imports([lib / "llama.dll"], [lib, system],
                                         loaded=lambda n: False)
    assert missing == {"msvcp140.dll": ["llama.dll"], "vulkan-1.dll": ["ggml.dll"]}
    assert pe_deps.unresolved_imports([lib / "llama.dll"], [lib, system],
                                      loaded=lambda n: True) == {}


def test_the_clause_names_the_runtime_and_setup_reads_it_back():
    clause = vc_runtime.describe_missing(
        {"msvcp140.dll": ["llama.dll", "ggml.dll"], "vulkan-1.dll": ["ggml-vulkan.dll"]})
    assert clause == ("msvcp140.dll is missing (the Microsoft Visual C++ runtime; needed by "
                      "ggml.dll, llama.dll); vulkan-1.dll is missing (needed by ggml-vulkan.dll)")
    line = f"RuntimeError: Failed to load llama.dll from X: {clause}. [WinError 126] ..."
    named = load_probe._name_missing_shared_lib(line) or ""
    assert named.startswith("msvcp140.dll (the Microsoft Visual C++ runtime)")
    assert vc_runtime.REDIST_PAGE in named and vc_runtime.RERUN_HINT in named
    other = load_probe._name_missing_shared_lib(
        "RuntimeError: Failed to load llama.dll from X: vulkan-1.dll is missing "
        "(needed by ggml-vulkan.dll). [WinError 126]") or ""
    assert other.startswith("vulkan-1.dll (needed by ggml-vulkan.dll)")
    assert vc_runtime.describe_missing({}) is None


def test_loader_names_the_missing_runtime_dll(tmp_path, monkeypatch):
    from localm.inference.backends.llamacpp import _loader
    lib = _runtime_folder(tmp_path, ["MSVCP140.dll", "ggml-base.dll"])
    monkeypatch.setattr(_loader.sys, "platform", "win32")
    monkeypatch.setattr(_loader, "rocm_runtime_dirs", lambda: [])
    monkeypatch.setattr(vc_runtime, "_system_dir", lambda: None)
    monkeypatch.setattr(pe_deps, "_is_loaded", lambda name: name.lower() == "kernel32.dll")
    monkeypatch.setattr(_loader.sys, "base_prefix", str(tmp_path / "py"))
    monkeypatch.setattr(_loader.sys, "executable", str(tmp_path / "py" / "python.exe"))
    clause = _loader.missing_dll_clause(lib, [lib / "llama.dll"])
    assert clause == ("msvcp140.dll is missing (the Microsoft Visual C++ runtime; needed by "
                      "ggml-base.dll, llama.dll)")
    (lib / "msvcp140.dll").write_bytes(_pe(["VCRUNTIME140.dll", "KERNEL32.dll"]))
    assert _loader.missing_dll_clause(lib, [lib / "llama.dll"]) == (
        "vcruntime140.dll is missing (the Microsoft Visual C++ runtime; needed by "
        "msvcp140.dll)")
    (lib / "vcruntime140.dll").write_bytes(_pe(["KERNEL32.dll"]))
    assert _loader.missing_dll_clause(lib, [lib / "llama.dll"]) is None


def test_loader_names_an_older_system_runtime(tmp_path, monkeypatch):
    from localm.inference.backends.llamacpp import _loader
    lib = _runtime_folder(tmp_path, ["MSVCP140.dll"])
    system = tmp_path / "system32"
    system.mkdir()
    (system / "msvcp140.dll").write_bytes(_pe(["KERNEL32.dll"], (14, 29, 30133, 0)))
    monkeypatch.setattr(_loader.sys, "platform", "win32")
    monkeypatch.setattr(vc_runtime.sys, "platform", "win32")
    monkeypatch.setattr(vc_runtime, "_system_dir", lambda: system)
    assert _loader._vc_runtime_short(lib) is True
    (system / "msvcp140.dll").write_bytes(_pe(["KERNEL32.dll"], VERSION))
    assert _loader._vc_runtime_short(lib) is False


def test_sdcpp_load_error_names_the_missing_runtime_dll(tmp_path, monkeypatch):
    from localm.media.sdcpp import _binding, _child
    rt = tmp_path / "sd"
    rt.mkdir()
    monkeypatch.setattr(vc_runtime.sys, "platform", "win32")
    (rt / _binding.lib_filename()).write_bytes(_pe(["MSVCP140.dll", "KERNEL32.dll"]))
    monkeypatch.setattr(vc_runtime, "_system_dir", lambda: None)
    monkeypatch.setattr(pe_deps, "_is_loaded", lambda name: name.lower() == "kernel32.dll")
    monkeypatch.setattr(vc_runtime.sys, "base_prefix", str(tmp_path / "py"))
    monkeypatch.setattr(vc_runtime.sys, "executable", str(tmp_path / "py" / "python.exe"))
    monkeypatch.setattr(_child, "_prepare_environment", lambda d: None)

    def refuse(runtime_dir, extra):
        raise OSError("[WinError 126] The specified module could not be found")
    monkeypatch.setattr(_binding, "load_library", refuse)
    with pytest.raises(RuntimeError) as e:
        _child._load_lib({"runtime_dir": str(rt), "extra_dirs": []})
    message = str(e.value)
    assert "msvcp140.dll is missing (the Microsoft Visual C++ runtime" in message
    assert vc_runtime.install_hint() in message


def test_nothing_places_microsoft_files_any_more():
    for name in ("place_app_local", "copy_placed", "cabinet_files", "runtime_files",
                 "LICENSE_NAME"):
        assert not hasattr(vc_runtime, name), name
    tree = [*(_REPO / "localm" / "setup_llama").glob("*.py"),
            *(_REPO / "localm" / "media" / "sdcpp").glob("*.py"),
            _REPO / "localm" / "vc_runtime.py"]
    assert not [p for p in tree if "msvc-runtime" in p.read_text(encoding="utf-8")]
