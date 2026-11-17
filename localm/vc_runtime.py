# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Microsoft Visual C++ runtime that the Windows builds of llama.cpp and
stable-diffusion.cpp need.

Those builds are built with MSVC and import its runtime (msvcp140.dll,
vcruntime140.dll, vcomp140.dll, ...), which their release archives do not
include and a clean Windows does not have. :func:`ensure` checks whether the
system folder holds every runtime DLL a runtime folder imports, at
:data:`REDIST_VERSION` or newer. When it does not and the caller can ask, it
offers to run Microsoft's Visual C++ Redistributable installer: downloaded from
Microsoft at a pinned version, checked against a pinned size and sha256 and a
valid Authenticode signature by Microsoft, and started elevated, so Windows
shows one administrator prompt. localm places no Microsoft file itself.
Whenever the runtime is still absent afterwards, :func:`absent_message` names
it, Microsoft's download page, and how to be offered the install again.

:func:`missing_clause` names the DLLs a failed load lacks."""

from __future__ import annotations

import hashlib
import platform
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Optional

from localm import pe_deps

# Every DLL of the Visual C++ v14 runtime package, lower case.
RUNTIME_DLLS = frozenset({
    "concrt140.dll", "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll", "vcamp140.dll",
    "vccorlib140.dll", "vcomp140.dll", "vcruntime140.dll", "vcruntime140_1.dll",
    "vcruntime140_threads.dll",
})

REDIST_VERSION = (14, 52, 37029, 0)

# (machine as platform.machine() reports it, lower case) -> (url, sha256, size).
# The URLs are Microsoft's versioned download locations behind
# https://aka.ms/vs/18/release/14.52.37029/.
_REDIST = {
    "amd64": (
        "https://download.visualstudio.microsoft.com/download/pr/"
        "78d061c5-9ef3-42ab-98ed-3309d1095dc4/"
        "9C02C9ED120F671B900E6B9471D890E3B2414117A7E53F4AE2E855466FA2B7C9/"
        "VC_redist.x64.exe",
        "9c02c9ed120f671b900e6b9471d890e3b2414117a7e53f4ae2e855466fa2b7c9",
        18863624),
    "arm64": (
        "https://download.visualstudio.microsoft.com/download/pr/"
        "78d061c5-9ef3-42ab-98ed-3309d1095dc4/"
        "3C81BCF5B06156858362400D7066F65ECE38E8ACCB5E687FC65BEF5735F31FFC/"
        "VC_redist.arm64.exe",
        "3c81bcf5b06156858362400d7066f65ece38e8accb5e687fc65bef5735f31ffc",
        11967280),
}
_MACHINE_ALIASES = {"x86_64": "amd64", "x64": "amd64", "aarch64": "arm64"}

#: Microsoft's page that lists the current Visual C++ Redistributable downloads.
REDIST_PAGE = "https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist"

#: The host the installer is downloaded from.
REDIST_HOST = "download.visualstudio.microsoft.com"

#: The installer's command line: install, no user interface, never restart
#: Windows. Microsoft's "Command-line options for the redistributable packages".
INSTALL_ARGS = ("/install", "/quiet", "/norestart")

#: The signer an installer must carry a valid Authenticode signature from.
SIGNER = "Microsoft Corporation"

#: The command that offers the install again.
RERUN_HINT = "localm setup-llama"

#: How long :func:`run_installer` waits for the installer, in milliseconds.
INSTALL_TIMEOUT_MS = 20 * 60 * 1000

# Exit codes the installer returns (Windows Installer and system error codes).
ERROR_SUCCESS = 0
ERROR_INSTALL_USEREXIT = 1602
ERROR_INSTALL_FAILURE = 1603
ERROR_INSTALL_ALREADY_RUNNING = 1618
ERROR_PRODUCT_VERSION = 1638
ERROR_SUCCESS_REBOOT_INITIATED = 1641
ERROR_SUCCESS_REBOOT_REQUIRED = 3010
#: Not the installer's: ShellExecuteEx fails with it when the administrator
#: prompt is declined, and :func:`run_installer` returns it then.
ERROR_CANCELLED = 1223
#: Not the installer's: :func:`run_installer` returns it when the installer is
#: still running after :data:`INSTALL_TIMEOUT_MS`.
WAIT_TIMEOUT = 258

_RESTART_CODES = frozenset({ERROR_SUCCESS_REBOOT_INITIATED, ERROR_SUCCESS_REBOOT_REQUIRED})

_EXIT_MEANINGS = {
    ERROR_CANCELLED: "the administrator prompt was declined",
    ERROR_INSTALL_USEREXIT: "the installation was cancelled",
    ERROR_INSTALL_FAILURE: "the installer failed (code 1603; Microsoft's log is in %TEMP%, "
                           "named dd_vcredist_*)",
    ERROR_INSTALL_ALREADY_RUNNING: "another installation is in progress (code 1618); "
                                   "let it finish first",
    ERROR_PRODUCT_VERSION: "the installer reports another version is already installed "
                           "(code 1638)",
    WAIT_TIMEOUT: "the installer was still running after 20 minutes",
}

# Whether the person declined the install in this process: they are not asked
# again until the next run.
_STATE = {"declined": False}


class RedistError(Exception):
    """The installer could not be downloaded, verified or started."""


def redist_machine(machine: Optional[str] = None) -> Optional[str]:
    """The key of :data:`_REDIST` for *machine* (default: this machine), or None
    when no package is pinned for it."""
    m = (machine if machine is not None else platform.machine()).lower()
    m = _MACHINE_ALIASES.get(m, m)
    return m if m in _REDIST else None


def redist_url(machine: Optional[str] = None) -> str:
    """Microsoft's direct download of the current Visual C++ Redistributable for
    *machine* (default: this machine)."""
    m = redist_machine(machine)
    return f"https://aka.ms/vc14/vc_redist.{'arm64' if m == 'arm64' else 'x64'}.exe"


def needed(folder: Path) -> set[str]:
    """The runtime DLL names (lower case) that a ``.dll`` or ``.exe`` directly
    in *folder* imports and *folder* does not hold. Empty when *folder* cannot
    be listed."""
    have: set[str] = set()
    images: list[Path] = []
    try:
        for f in folder.iterdir():
            have.add(f.name.lower())
            if f.suffix.lower() in (".dll", ".exe") and f.is_file():
                images.append(f)
    except OSError:
        return set()
    wanted: set[str] = set()
    for image in images:
        wanted.update(n.lower() for n in pe_deps.imported_dlls(image))
    return (wanted & RUNTIME_DLLS) - have


def _system_dir() -> Optional[Path]:
    """The Windows system folder (System32), or None off Windows or when it
    cannot be read."""
    if sys.platform != "win32":
        return None
    import ctypes
    buf = ctypes.create_unicode_buffer(512)
    try:
        n = ctypes.windll.kernel32.GetSystemDirectoryW(buf, len(buf))
    except (AttributeError, OSError):
        return None
    return Path(buf.value) if n else None


def provided_by_system(names: Iterable[str], system_dir: Optional[Path]) -> set[str]:
    """The *names* the system folder has at :data:`REDIST_VERSION` or newer."""
    out: set[str] = set()
    if system_dir is None:
        return out
    for name in names:
        version = pe_deps.file_version(system_dir / name)
        if version is not None and version >= REDIST_VERSION:
            out.add(name)
    return out


_SYSTEM = object()


def _resolve_system(system_dir: object) -> Optional[Path]:
    if system_dir is _SYSTEM:
        return _system_dir()
    return system_dir if isinstance(system_dir, Path) else None


def missing_from_system(folder: Path, system_dir: object = _SYSTEM) -> list[str]:
    """The runtime DLLs the binaries in *folder* import that neither *folder*
    nor the system folder provides at :data:`REDIST_VERSION` or newer, sorted.
    *system_dir* defaults to the Windows system folder; None means there is
    none. Empty off Windows when *system_dir* is left at its default."""
    if system_dir is _SYSTEM and sys.platform != "win32":
        return []
    want = needed(folder)
    if not want:
        return []
    return sorted(want - provided_by_system(want, _resolve_system(system_dir)))


def _older_on_system(names: Iterable[str], system_dir: Optional[Path]) -> bool:
    """Whether the system folder holds any of *names*, at a version older than
    :data:`REDIST_VERSION` (the caller passes only names it lacks at that
    version)."""
    if system_dir is None:
        return False
    return any(pe_deps.file_version(system_dir / n) is not None for n in names)


def _what(names: list[str], older: bool) -> str:
    state = "has an older version of" if older else "does not have"
    return (f"this Windows {state} the Microsoft Visual C++ Redistributable "
            f"({', '.join(names)}) that the llama.cpp and image/video runtimes need")


def question(names: list[str], older: bool = False, machine: Optional[str] = None) -> str:
    """The yes/no question that offers the install of *names*."""
    key = redist_machine(machine) or "amd64"
    size_mb = round(_REDIST[key][2] / 1_000_000)
    what = _what(names, older)
    return (f"{what[0].upper()}{what[1:]}; no model can "
            f"load without it. Download Microsoft's installer (about {size_mb} MB from "
            f"{REDIST_HOST}) and install it now? Windows will show one administrator "
            "prompt.")


def absent_message(names: list[str], reason: str, older: bool = False) -> str:
    """The message for every path on which the runtime stays absent: what is
    missing, *reason* (why it was not installed), Microsoft's download page, and
    the command that offers the install again."""
    return (f"No model can load yet: {_what(names, older)}, and it was not installed "
            f"({reason.rstrip('. ')}). Install the Visual C++ Redistributable from Microsoft: "
            f"{REDIST_PAGE} (direct download for this PC: {redist_url()}). Or run "
            f"'{RERUN_HINT}' (or setup again), which offers to install it.")


def net_off_refusal() -> str:
    """The text of a Network access off refusal of the installer download, with
    the remedy."""
    from localm.config import net_off_download_refusal
    return net_off_download_refusal(
        "the Microsoft Visual C++ Redistributable was not downloaded")


# --------------------------------------------------------------------------- #
#  Checking and running the installer                                          #
# --------------------------------------------------------------------------- #

def _hold_open(path: Path):
    """A handle on *path* that lets others read and run it but not write,
    rename or delete it, so the file checked is the file started. None off
    Windows. Raises RedistError when it cannot be opened."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                wintypes.HANDLE]
    generic_read, file_share_read, open_existing = 0x80000000, 0x1, 3
    h = k32.CreateFileW(str(path), generic_read, file_share_read, None, open_existing, 0, None)
    if h in (None, wintypes.HANDLE(-1).value):
        raise RedistError(f"could not open the downloaded installer "
                          f"(error {ctypes.get_last_error()})")
    return h


def _close(handle) -> None:
    """Close a handle from :func:`_hold_open`; None is ignored."""
    if handle is None:
        return
    import ctypes
    from ctypes import wintypes
    close = ctypes.windll.kernel32.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close(handle)


def check_package(path: Path, key: str) -> None:
    """Raise RedistError unless *path* is the pinned installer for *key*: its
    size, its sha256, and a valid Authenticode signature by :data:`SIGNER`."""
    _url, sha256, size = _REDIST[key]
    got_size = path.stat().st_size
    if got_size != size:
        raise RedistError(f"the download is {got_size} bytes, expected {size}")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        raise RedistError(f"the download's sha256 is {digest.hexdigest()}, expected "
                          f"{sha256}: refusing a possibly tampered or wrong file")
    signer = authenticode_signer(path)
    if signer != SIGNER:
        raise RedistError(f"the installer is signed by {signer!r}, not {SIGNER!r}")


def authenticode_signer(path: Path) -> str:
    """The display name of the certificate that signed *path*, after Windows
    verified its Authenticode signature and certificate chain (WinVerifyTrust,
    no revocation check, no network fetch). Raises RedistError when the
    signature is missing or not valid, or off Windows."""
    if sys.platform != "win32":
        raise RedistError("Authenticode signatures can only be checked on Windows")
    import ctypes
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

    class WINTRUST_FILE_INFO(ctypes.Structure):
        _fields_ = [("cbStruct", wintypes.DWORD), ("pcwszFilePath", wintypes.LPCWSTR),
                    ("hFile", wintypes.HANDLE), ("pgKnownSubject", ctypes.POINTER(GUID))]

    class WINTRUST_DATA(ctypes.Structure):
        _fields_ = [("cbStruct", wintypes.DWORD), ("pPolicyCallbackData", ctypes.c_void_p),
                    ("pSIPClientData", ctypes.c_void_p), ("dwUIChoice", wintypes.DWORD),
                    ("fdwRevocationChecks", wintypes.DWORD),
                    ("dwUnionChoice", wintypes.DWORD),
                    ("pFile", ctypes.POINTER(WINTRUST_FILE_INFO)),
                    ("dwStateAction", wintypes.DWORD), ("hWVTStateData", wintypes.HANDLE),
                    ("pwszURLReference", wintypes.LPCWSTR), ("dwProvFlags", wintypes.DWORD),
                    ("dwUIContext", wintypes.DWORD), ("pSignatureSettings", ctypes.c_void_p)]

    class CRYPT_PROVIDER_CERT(ctypes.Structure):
        _fields_ = [("cbStruct", wintypes.DWORD), ("pCert", ctypes.c_void_p)]

    class CRYPT_PROVIDER_SGNR(ctypes.Structure):
        _fields_ = [("cbStruct", wintypes.DWORD), ("sftVerifyAsOf", wintypes.FILETIME),
                    ("csCertChain", wintypes.DWORD),
                    ("pasCertChain", ctypes.POINTER(CRYPT_PROVIDER_CERT))]

    # WINTRUST_ACTION_GENERIC_VERIFY_V2 {00AAC56B-CD44-11d0-8CC2-00C04FC295EE}
    action = GUID(0x00AAC56B, 0xCD44, 0x11D0,
                  (ctypes.c_ubyte * 8)(0x8C, 0xC2, 0x00, 0xC0, 0x4F, 0xC2, 0x95, 0xEE))
    wtd_ui_none, wtd_revoke_none, wtd_choice_file = 2, 0, 1
    wtd_stateaction_verify, wtd_stateaction_close = 1, 2
    wtd_cache_only_url_retrieval = 0x1000
    cert_name_simple_display_type = 4

    wintrust = ctypes.WinDLL("wintrust")
    crypt32 = ctypes.WinDLL("crypt32")
    wintrust.WinVerifyTrust.restype = ctypes.c_long
    wintrust.WinVerifyTrust.argtypes = [wintypes.HWND, ctypes.POINTER(GUID),
                                        ctypes.POINTER(WINTRUST_DATA)]
    wintrust.WTHelperProvDataFromStateData.restype = ctypes.c_void_p
    wintrust.WTHelperProvDataFromStateData.argtypes = [wintypes.HANDLE]
    wintrust.WTHelperGetProvSignerFromChain.restype = ctypes.POINTER(CRYPT_PROVIDER_SGNR)
    wintrust.WTHelperGetProvSignerFromChain.argtypes = [ctypes.c_void_p, wintypes.DWORD,
                                                        wintypes.BOOL, wintypes.DWORD]
    crypt32.CertGetNameStringW.restype = wintypes.DWORD
    crypt32.CertGetNameStringW.argtypes = [ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                           ctypes.c_void_p, wintypes.LPWSTR, wintypes.DWORD]

    info = WINTRUST_FILE_INFO(ctypes.sizeof(WINTRUST_FILE_INFO), str(path), None, None)
    data = WINTRUST_DATA()
    data.cbStruct = ctypes.sizeof(WINTRUST_DATA)
    data.dwUIChoice = wtd_ui_none
    data.fdwRevocationChecks = wtd_revoke_none
    data.dwUnionChoice = wtd_choice_file
    data.pFile = ctypes.pointer(info)
    data.dwStateAction = wtd_stateaction_verify
    data.dwProvFlags = wtd_cache_only_url_retrieval
    status = wintrust.WinVerifyTrust(None, ctypes.byref(action), ctypes.byref(data))
    try:
        if status != 0:
            raise RedistError(f"the installer's signature is not valid "
                              f"(WinVerifyTrust {status & 0xFFFFFFFF:#010x})")
        prov = wintrust.WTHelperProvDataFromStateData(data.hWVTStateData)
        signer = wintrust.WTHelperGetProvSignerFromChain(prov, 0, False, 0) if prov else None
        if not signer or signer.contents.csCertChain < 1:
            raise RedistError("the installer's signer could not be read")
        cert = signer.contents.pasCertChain[0].pCert
        buf = ctypes.create_unicode_buffer(256)
        if crypt32.CertGetNameStringW(cert, cert_name_simple_display_type, 0, None,
                                      buf, len(buf)) <= 1:
            raise RedistError("the installer's signer has no name")
        return buf.value
    finally:
        data.dwStateAction = wtd_stateaction_close
        wintrust.WinVerifyTrust(None, ctypes.byref(action), ctypes.byref(data))


def run_installer(path: Path, args: Iterable[str] = INSTALL_ARGS) -> int:
    """Start *path* with *args* elevated (the "runas" verb, so Windows shows its
    administrator prompt unless this process is elevated already), wait for it
    and return its exit code. Returns :data:`ERROR_CANCELLED` when the prompt is
    declined and :data:`WAIT_TIMEOUT` when it is still running after
    :data:`INSTALL_TIMEOUT_MS` (it is left running). Raises RedistError when it
    cannot be started, or off Windows."""
    if sys.platform != "win32":
        raise RedistError("the installer runs only on Windows")
    import ctypes
    from ctypes import wintypes
    import subprocess

    class SHELLEXECUTEINFOW(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("fMask", ctypes.c_ulong),
                    ("hwnd", wintypes.HWND), ("lpVerb", wintypes.LPCWSTR),
                    ("lpFile", wintypes.LPCWSTR), ("lpParameters", wintypes.LPCWSTR),
                    ("lpDirectory", wintypes.LPCWSTR), ("nShow", ctypes.c_int),
                    ("hInstApp", wintypes.HINSTANCE), ("lpIDList", ctypes.c_void_p),
                    ("lpClass", wintypes.LPCWSTR), ("hkeyClass", wintypes.HKEY),
                    ("dwHotKey", wintypes.DWORD), ("hIconOrMonitor", wintypes.HANDLE),
                    ("hProcess", wintypes.HANDLE)]

    see_mask_nocloseprocess, see_mask_noasync, see_mask_flag_no_ui = 0x40, 0x100, 0x400
    sw_hide = 0
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ole32 = ctypes.WinDLL("ole32")
    shell32.ShellExecuteExW.restype = wintypes.BOOL
    shell32.ShellExecuteExW.argtypes = [ctypes.POINTER(SHELLEXECUTEINFOW)]
    k32.WaitForSingleObject.restype = wintypes.DWORD
    k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.GetExitCodeProcess.restype = wintypes.BOOL
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]

    coinit = ole32.CoInitializeEx(None, 0x2 | 0x4)  # apartment threaded, no OLE1 DDE
    try:
        info = SHELLEXECUTEINFOW()
        info.cbSize = ctypes.sizeof(SHELLEXECUTEINFOW)
        info.fMask = see_mask_nocloseprocess | see_mask_noasync | see_mask_flag_no_ui
        info.lpVerb = "runas"
        info.lpFile = str(path)
        info.lpParameters = subprocess.list2cmdline(list(args))
        info.lpDirectory = str(path.parent)
        info.nShow = sw_hide
        if not shell32.ShellExecuteExW(ctypes.byref(info)):
            err = ctypes.get_last_error()
            if err == ERROR_CANCELLED:
                return ERROR_CANCELLED
            raise RedistError(f"the installer could not be started (error {err}: "
                              f"{ctypes.FormatError(err).strip()})")
        if not info.hProcess:
            raise RedistError("the installer started but its process cannot be waited on")
        try:
            if k32.WaitForSingleObject(info.hProcess, INSTALL_TIMEOUT_MS) == WAIT_TIMEOUT:
                return WAIT_TIMEOUT
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(info.hProcess, ctypes.byref(code)):
                raise RedistError(f"the installer's exit code cannot be read "
                                  f"(error {ctypes.get_last_error()})")
            return int(code.value)
        finally:
            k32.CloseHandle(info.hProcess)
    finally:
        if coinit in (0, 1):
            ole32.CoUninitialize()


def _exit_meaning(code: int) -> str:
    return _EXIT_MEANINGS.get(code, f"the installer exited with code {code}")


def ensure(folder: Path, *, download: Callable[[str, Path], object],
           refusal: Callable[[str], Optional[str]],
           ask: Optional[Callable[[str], bool]], say: Callable[[str], None],
           system_dir: object = _SYSTEM, machine: Optional[str] = None) -> bool:
    """Whether the Visual C++ runtime the binaries in *folder* import is present
    at :data:`REDIST_VERSION` or newer, offering to install it when it is not.

    Off Windows (with *system_dir* left at its default), or when nothing in
    *folder* imports the runtime, or the system folder already has it: returns
    True and does nothing else. Otherwise, when *ask* is None (an unattended
    run), the person declined earlier in this process, no installer is pinned
    for this machine, or *refusal(url)* (the caller's network policy check of
    the download) returns a reason: *say* gets :func:`absent_message` and it
    returns False without asking or downloading anything.
    Otherwise *ask(question)* is asked; on yes the installer is downloaded with
    *download(url, dest)* into a temporary folder inside *folder*, checked
    (:func:`check_package`), run (:func:`run_installer`) and removed, and the
    system folder is checked again: True when the runtime is now present (with
    a restart note when the installer asked for one), else *say* gets
    :func:`absent_message` with the reason and it returns False.

    A failed or refused download, a failed check and a failed installer end in
    :func:`absent_message`, never an exception; an exception raised by *ask* or
    *say* propagates."""
    sysdir = _resolve_system(system_dir)
    if system_dir is _SYSTEM and sys.platform != "win32":
        return True
    names = missing_from_system(folder, sysdir)
    if not names:
        return True
    older = _older_on_system(names, sysdir)
    key = redist_machine(machine)
    if key is None:
        say(absent_message(names, f"no installer is pinned for "
                                  f"{machine or platform.machine()}", older))
        return False
    if ask is None:
        say(absent_message(names, "this run is unattended, so it was not offered", older))
        return False
    if _STATE["declined"]:
        say(absent_message(names, "you declined it earlier in this run", older))
        return False
    url = _REDIST[key][0]
    refused = refusal(url)
    if refused:
        say(absent_message(names, refused, older))
        return False
    if not ask(question(names, older, machine)):
        _STATE["declined"] = True
        say(absent_message(names, "you declined it", older))
        return False
    code: Optional[int] = None
    try:
        with tempfile.TemporaryDirectory(prefix=".vcredist-", dir=folder) as tmp:
            exe = Path(tmp) / url.rsplit("/", 1)[-1]
            download(url, exe)
            if not exe.is_file():
                raise RedistError("the download produced no file")
            handle = _hold_open(exe)
            try:
                check_package(exe, key)
                say("Installing the Microsoft Visual C++ Redistributable: approve the "
                    "administrator prompt Windows shows.")
                code = run_installer(exe)
            finally:
                _close(handle)
    except Exception as e:  # noqa: BLE001 - every failure ends in absent_message
        from localm.debuglog import logger
        logger.warning("Visual C++ Redistributable install did not happen: %s", e)
        say(absent_message(names, f"{e}", older))
        return False
    still = missing_from_system(folder, sysdir)
    if not still:
        note = (" Windows asks for a restart to finish it; models can load now."
                if code in _RESTART_CODES else "")
        say(f"The Microsoft Visual C++ Redistributable is installed.{note}")
        return True
    if code in _RESTART_CODES:
        reason = "Windows needs a restart to finish the install"
    elif code == ERROR_SUCCESS:
        reason = "the installer finished but the system folder still lacks it"
    else:
        reason = _exit_meaning(code if code is not None else -1)
    say(absent_message(still, reason, _older_on_system(still, sysdir)))
    return False


# --------------------------------------------------------------------------- #
#  Naming what is missing after a failed load                                  #
# --------------------------------------------------------------------------- #

def missing_clause(images: Iterable[Path], dirs: Iterable[Path]) -> Optional[str]:
    """Windows, after a library in *images* failed to load: the
    :func:`describe_missing` clause for the DLLs that neither *dirs*, Python's
    own folder, nor the system folder holds and that this process has not
    loaded. None off Windows, when every import is found, or when the scan
    itself fails (logged)."""
    if sys.platform != "win32":
        return None
    own = [Path(d) for d in dirs]
    search = own + [Path(sys.base_prefix), Path(sys.executable).parent]
    system = _system_dir()
    if system is not None:
        search.append(system)
    try:
        missing = pe_deps.unresolved_imports([Path(p) for p in images], search, follow=own)
    except Exception:
        from localm.debuglog import logger
        logger.warning("could not work out which DLL is missing", exc_info=True)
        return None
    return describe_missing(missing)


def describe_missing(missing: dict[str, list[str]]) -> Optional[str]:
    """One clause naming the DLLs in *missing* (as
    :func:`pe_deps.unresolved_imports` returns it), saying which belong to the
    Visual C++ runtime. None when *missing* is empty. The text starts with the
    DLL names and "is missing" or "are missing", which
    :func:`localm.setup_llama.load_probe._name_missing_shared_lib` reads back."""
    if not missing:
        return None
    vc = sorted(n for n in missing if n in RUNTIME_DLLS)
    other = sorted(n for n in missing if n not in RUNTIME_DLLS)
    parts = []
    for names, what in ((vc, "the Microsoft Visual C++ runtime"), (other, None)):
        if not names:
            continue
        users = sorted({u for n in names for u in missing[n]})
        verb = "is" if len(names) == 1 else "are"
        needed_by = ", ".join(users[:3]) + (", ..." if len(users) > 3 else "")
        label = f"{what}; " if what else ""
        parts.append(f"{', '.join(names)} {verb} missing ({label}needed by {needed_by})")
    return "; ".join(parts)


def install_hint() -> str:
    """The remedy a load error names when a Visual C++ runtime DLL is missing."""
    return (f"Run '{RERUN_HINT}', which offers to install the Microsoft Visual C++ "
            f"Redistributable, or install it from Microsoft: {REDIST_PAGE}")
