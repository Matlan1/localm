# SPDX-License-Identifier: AGPL-3.0-or-later
"""Install / uninstall an OPTIONAL global ``localm`` command - without ever
breaking anything already on the user's machine.

Rule zero of this project: never damage the user's existing setup. So:

- ``setx PATH ...`` is BANNED here. It truncates the per-user PATH at 1024
  characters and SILENTLY corrupts it (it prints "SUCCESS" while eating the
  rest); it has wiped real users' PATHs and broke the GitHub Desktop installer.
  On Windows we edit the per-user PATH through the registry (``winreg``, no
  truncation), append our ONE directory only if absent (idempotent), and
  broadcast the change so new shells pick it up.
- We never put the venv's ``Scripts``/``bin`` directory on PATH: it carries
  ``python``/``pip`` and would SHADOW the user's own tools. Only a directory
  holding a single ``localm`` shim goes on PATH.
- On Linux we drop a symlink in ``~/.local/bin`` (already on PATH by
  convention; pip / pipx / uv use it) and touch a shell rc only if that dir is
  not yet on PATH.
- If a DIFFERENT ``localm`` already resolves, we do not clobber it - we append
  ours (lowest precedence) and report the conflict so the caller can tell the
  user. Every change is recorded (by install_manifest) for an exact, reversible
  uninstall.

Windows puts the shim in ``<clone>/bin`` (inside the clone, so it travels with
the install); Linux uses the conventional ``~/.local/bin`` symlink. Both are the
idiomatic per-OS choice and both are fully reversible.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
#  Deterministic locations (the caller records these; no output capture needed) #
# --------------------------------------------------------------------------- #

def bin_dir(clone_root) -> Path:
    """The single directory that goes on PATH for the global command.

    Windows: ``<clone>/bin`` (in the clone). Linux/macOS: ``~/.local/bin`` (the
    conventional per-user bin dir). It holds ONLY the localm shim, never the venv
    scripts dir."""
    if sys.platform == "win32":
        return Path(clone_root) / "bin"
    return Path.home() / ".local" / "bin"


def shim_path(clone_root) -> Path:
    """Full path of the ``localm`` shim/symlink we create."""
    d = bin_dir(clone_root)
    return d / ("localm.cmd" if sys.platform == "win32" else "localm")


def _venv_localm(clone_root) -> Path:
    """The real localm entry point inside this clone's venv."""
    root = Path(clone_root)
    if sys.platform == "win32":
        return root / ".venv" / "Scripts" / "localm.exe"
    return root / ".venv" / "bin" / "localm"


def existing_localm(our_shim: Path) -> Optional[str]:
    """If a DIFFERENT ``localm`` already resolves on PATH, return its path so the
    caller can warn the user. Returns None when there is none, or when the only
    match is our own shim (so re-running setup is not reported as a conflict)."""
    found = shutil.which("localm")
    if not found:
        return None
    try:
        if Path(found).resolve() == Path(our_shim).resolve():
            return None
    except OSError:
        pass
    return found


# --------------------------------------------------------------------------- #
#  Windows PATH via the registry (NEVER setx)                                   #
# --------------------------------------------------------------------------- #

def _win_broadcast_env_change() -> None:
    """Tell running processes the environment changed so a NEW shell sees the new
    PATH without a logout. Best-effort: a failure just means "open a new
    terminal", never a broken PATH, so it is safe to swallow."""
    try:
        import ctypes

        HWND_BROADCAST = 0xFFFF
        WM_SETTINGCHANGE = 0x1A
        SMTO_ABORTIFHUNG = 0x0002
        ctypes.windll.user32.SendMessageTimeoutW(
            HWND_BROADCAST, WM_SETTINGCHANGE, 0, "Environment",
            SMTO_ABORTIFHUNG, 5000, None)
    except Exception:
        pass


def _win_read_user_path():
    """Return ``(value, regtype)`` of ``HKCU\\Environment\\Path`` (the RAW per-user
    PATH only - never the merged system+user PATH, so a write-back can never fold
    the large system PATH into the user one). ``("", REG_EXPAND_SZ)`` if unset."""
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        try:
            value, regtype = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            return "", winreg.REG_EXPAND_SZ
    return (value or ""), regtype


def _win_write_user_path(value: str, regtype) -> None:
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                        winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, "Path", 0, regtype, value)
    _win_broadcast_env_change()


def _split_path(value: str) -> list:
    return [p for p in value.split(os.pathsep) if p]


def _same_dir(a: str, b: str) -> bool:
    try:
        return (os.path.normcase(os.path.normpath(a))
                == os.path.normcase(os.path.normpath(b)))
    except Exception:
        return a == b


def _win_path_add(dir_str: str) -> bool:
    """Append *dir_str* to the USER PATH iff absent (idempotent). Returns True if
    PATH changed. Registry write, no ``setx``, every existing entry preserved."""
    value, regtype = _win_read_user_path()
    entries = _split_path(value)
    if any(_same_dir(e, dir_str) for e in entries):
        return False
    entries.append(dir_str)
    _win_write_user_path(os.pathsep.join(entries), regtype)
    return True


def _win_path_remove(dir_str: str) -> bool:
    """Remove ONLY *dir_str* from the USER PATH; leave every other entry exactly
    as it was. Returns True if PATH changed."""
    value, regtype = _win_read_user_path()
    entries = _split_path(value)
    kept = [e for e in entries if not _same_dir(e, dir_str)]
    if len(kept) == len(entries):
        return False
    _win_write_user_path(os.pathsep.join(kept), regtype)
    return True


def _write_win_shim(shim: Path) -> None:
    """A tiny .cmd that forwards to this clone's venv localm. ``%~dp0`` is the
    shim's OWN directory, so it resolves the venv relative to itself and keeps
    working even if the clone is later moved."""
    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text(
        "@echo off\r\n"
        '"%~dp0..\\.venv\\Scripts\\localm.exe" %*\r\n',
        encoding="utf-8")


# --------------------------------------------------------------------------- #
#  Linux / macOS: ~/.local/bin symlink + shell-rc ensure                        #
# --------------------------------------------------------------------------- #

_RC_MARK = "# added by localm setup (global `localm` command)"


def _posix_on_path(bindir: Path) -> bool:
    return any(_same_dir(p, str(bindir)) for p in _split_path(os.environ.get("PATH", "")))


def _posix_ensure_on_path(bindir: Path) -> bool:
    """If *bindir* is not already on PATH, append an export line to the user's
    shell rc (pipx ``ensurepath`` style). Returns True if an rc was edited.
    ``~/.local/bin`` is usually already on PATH, so this is usually a no-op."""
    if _posix_on_path(bindir):
        return False
    line = f'\n{_RC_MARK}\nexport PATH="$HOME/.local/bin:$PATH"\n'
    edited = False
    for name in (".bashrc", ".zshrc", ".profile"):
        rc = Path.home() / name
        try:
            if rc.exists():
                if _RC_MARK not in rc.read_text(encoding="utf-8", errors="replace"):
                    with rc.open("a", encoding="utf-8") as fh:
                        fh.write(line)
                edited = True
        except OSError:
            pass
    if not edited:
        # No rc existed: create ~/.profile so a login shell picks it up.
        try:
            (Path.home() / ".profile").write_text(line, encoding="utf-8")
            edited = True
        except OSError:
            pass
    return edited


# --------------------------------------------------------------------------- #
#  Public install / uninstall                                                   #
# --------------------------------------------------------------------------- #

def install(clone_root) -> dict:
    """Make ``localm`` available from any terminal. Non-destructive: creates our
    shim and appends our one bin dir to the user PATH; never overwrites another
    tool's command. Returns a dict the caller records in the install manifest:
    ``{path_dir, shim, path_modified, conflict}`` (conflict = a pre-existing
    ``localm`` on PATH, or None)."""
    clone_root = Path(clone_root)
    d = bin_dir(clone_root)
    shim = shim_path(clone_root)
    conflict = existing_localm(shim)

    if sys.platform == "win32":
        _write_win_shim(shim)
        changed = _win_path_add(str(d))
    else:
        d.mkdir(parents=True, exist_ok=True)
        target = _venv_localm(clone_root)
        # Replace only our OWN symlink on a re-run; never touch a real file there.
        if shim.is_symlink():
            shim.unlink()
        elif shim.exists():
            # Something that is not our symlink occupies the name - do not clobber.
            return {"path_dir": str(d), "shim": str(shim), "path_modified": False,
                    "conflict": str(shim)}
        shim.symlink_to(target)
        changed = _posix_ensure_on_path(d)

    return {"path_dir": str(d), "shim": str(shim),
            "path_modified": bool(changed), "conflict": conflict}


def uninstall_command(path_dir: str, shim: str) -> dict:
    """Reverse install(): remove our shim and take our one dir back off the user
    PATH, leaving every other PATH entry untouched. Returns
    ``{removed: [...], notes: [...]}``. Safe to call when nothing was installed."""
    report = {"removed": [], "notes": []}

    # 1) the shim file / symlink
    try:
        p = Path(shim) if shim else None
        if p and (p.exists() or p.is_symlink()):
            p.unlink()
            report["removed"].append(str(p))
    except OSError as e:
        report["notes"].append(f"could not remove shim {shim}: {e}")

    # 2) our directory off PATH
    if path_dir:
        try:
            if sys.platform == "win32":
                if _win_path_remove(path_dir):
                    report["removed"].append(f"PATH entry {path_dir}")
            else:
                # ~/.local/bin is shared with other tools, so we do NOT strip it
                # from PATH or the shell rc on uninstall (removing it could break
                # pipx/uv commands). Only our symlink is removed, above.
                report["notes"].append(
                    f"left {path_dir} on PATH (shared with other tools)")
        except Exception as e:  # never let uninstall hard-fail on a PATH edit
            report["notes"].append(f"could not update PATH for {path_dir}: {e}")

    return report


# --------------------------------------------------------------------------- #
#  CLI (invoked by setup.sh / setup.bat)                                        #
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="localm.globalcmd")
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("install", help="add a global `localm` command")
    i.add_argument("--root", default=".")

    u = sub.add_parser("uninstall", help="remove the global `localm` command")
    u.add_argument("--path-dir", default="")
    u.add_argument("--shim", default="")

    p = sub.add_parser("path-dir", help="print the bin dir that goes on PATH")
    p.add_argument("--root", default=".")
    s = sub.add_parser("shim", help="print the shim path")
    s.add_argument("--root", default=".")

    args = ap.parse_args(argv)
    if args.cmd == "path-dir":
        print(bin_dir(args.root))
        return 0
    if args.cmd == "shim":
        print(shim_path(args.root))
        return 0
    if args.cmd == "install":
        try:
            res = install(args.root)
        except Exception as e:
            # A real failure (cannot write the shim / edit PATH): report it clearly
            # and exit nonzero so the installer records NOTHING - the manifest must
            # never claim an install that did not happen (rules 6 + 11).
            print(f"  [!] Could not add the global command: {e}")
            return 1
        if res.get("conflict"):
            print(f"  [!] A 'localm' command already exists at {res['conflict']}. "
                  "This clone's command was still added (lower precedence); the "
                  "existing one takes priority until you reorder PATH.")
        if res.get("path_modified"):
            print("  Added 'localm' to your PATH. Open a NEW terminal to use it.")
            return 0
        # Installed, but PATH was already set (a re-run, or the dir was present).
        # Exit 20 tells setup to record the command WITHOUT --path-modified, so the
        # manifest never claims to have changed a PATH it did not, and uninstall
        # will not take a dir off PATH that we did not put there.
        print("  'localm' is available (its directory was already on PATH). "
              "Open a new terminal if it is not found yet.")
        return 20
    rep = uninstall_command(args.path_dir, args.shim)
    for x in rep["removed"]:
        print(f"  removed: {x}")
    for n in rep["notes"]:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
