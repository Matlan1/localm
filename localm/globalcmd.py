# SPDX-License-Identifier: AGPL-3.0-or-later
"""Install / uninstall an OPTIONAL global ``localm`` command - without ever
breaking anything already on the user's machine.

- ``setx PATH ...`` is BANNED here: it truncates the per-user PATH at 1024
  characters and SILENTLY corrupts it, printing "SUCCESS" while eating the
  rest. On Windows the per-user PATH is edited through the registry
  (``winreg``, no truncation), our ONE directory is appended only if absent
  (idempotent), and the change is broadcast so new shells pick it up.
- The venv's ``Scripts``/``bin`` directory NEVER goes on PATH: it carries
  ``python``/``pip`` and would SHADOW the user's own tools. Only a directory
  holding a single ``localm`` shim goes on PATH.
- On Linux a symlink goes in ``~/.local/bin`` (already on PATH by convention;
  pip / pipx / uv use it) and a shell rc is touched only if that dir is not yet
  on PATH.
- A DIFFERENT ``localm`` that already resolves is never clobbered: ours is
  appended (lowest precedence) and the conflict is reported so the caller can
  tell the user. Every change is recorded (by install_manifest) for an exact,
  reversible uninstall.

Windows puts the shim in ``<clone>/bin`` (inside the clone, so it travels with
the install); Linux uses the conventional ``~/.local/bin`` symlink. Both are
fully reversible.
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


def _norm(p) -> str:
    """Lexical, I/O-free normalisation for comparing two directory strings."""
    try:
        return os.path.normcase(os.path.normpath(str(p)))
    except Exception:
        return str(p)


def path_dirs(path_value: Optional[str] = None) -> list:
    """The directories ON PATH, lexically normalised, in search order.

    LEXICAL: no ``resolve()``, no stat, no filesystem I/O of any kind."""
    raw = os.environ.get("PATH", "") if path_value is None else path_value
    return [_norm(e.strip().strip('"')) for e in _split_path(raw) if e.strip()]


def existing_localm(our_shim: Path, clone_root=None) -> Optional[str]:
    """If a DIFFERENT ``localm`` already resolves FROM PATH, return its path so
    the caller can ask the user what to do. Returns None when there is none.

    Three things are NOT a conflict:

    - our own shim (so re-running setup never reports one);
    - anything inside this clone (``clone_root``);
    - a hit whose directory is not actually on PATH."""
    found = shutil.which("localm")
    if not found:
        return None
    # Lexical rejections first: no filesystem I/O.
    absolute = os.path.abspath(found)          # cwd-relative -> absolute, no I/O
    if clone_root is not None:
        root = _norm(clone_root)
        candidate = _norm(absolute)
        if candidate == root or candidate.startswith(root + os.sep):
            return None
    if _norm(os.path.dirname(absolute)) not in path_dirs():
        return None
    # Only now spend the I/O: a symlinked shim needs a real resolve to be
    # recognised as ours (the POSIX shim IS a symlink into the venv).
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


def _win_read_system_path() -> str:
    """The RAW machine PATH, read-only. ``""`` when it cannot be read.

    Windows composes a new process's PATH as SYSTEM entries first, then USER
    entries, so a ``localm`` sitting in the system PATH can NEVER be out-ordered
    by an edit to the user PATH - and only the user PATH is ever edited here."""
    try:
        import winreg

        key = (r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
            value, _ = winreg.QueryValueEx(k, "Path")
        return value or ""
    except Exception:
        return ""


def conflict_outranks_user_path(conflict: str) -> bool:
    """True when *conflict* lives in the machine PATH, so putting our directory
    first in the USER PATH still will not make our command win."""
    if sys.platform != "win32" or not conflict:
        return False
    parent = _norm(os.path.dirname(os.path.abspath(conflict)))
    return parent in path_dirs(_win_read_system_path())


def _win_path_add(dir_str: str, prepend: bool = False) -> bool:
    """Put *dir_str* on the USER PATH. Returns True if PATH changed.

    Registry write, no ``setx``, every existing entry preserved. With *prepend*,
    the directory is moved to the FRONT of the user PATH."""
    value, regtype = _win_read_user_path()
    entries = _split_path(value)
    present = [i for i, e in enumerate(entries) if _same_dir(e, dir_str)]
    if present and (not prepend or present[0] == 0):
        return False
    entries = [e for e in entries if not _same_dir(e, dir_str)]
    if prepend:
        entries.insert(0, dir_str)
    else:
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


def _posix_ensure_on_path(bindir: Path, prepend: bool = False):
    """If *bindir* is not already on PATH, append an export line to the user's
    shell rc (pipx ``ensurepath`` style). ``~/.local/bin`` is usually already on
    PATH, so this is usually a no-op.

    Returns ``(changed, note)``:
    - ``(False, None)`` - already on PATH (nothing to do).
    - ``(True, None)``  - an rc was edited (or ~/.profile created) successfully.
    - ``(False, <str>)`` - *bindir* is NOT on PATH but every shell-rc edit failed
      (unwritable / root-owned / immutable dotfiles). A caller must not report
      that as "already on PATH"; *note* names the manual step. A note, not a
      raise."""
    # With *prepend* the rc line is written even when bindir is already on PATH,
    # since that line puts it first.
    if _posix_on_path(bindir) and not prepend:
        return False, None
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
    if not edited:
        # Reached only when bindir is NOT on PATH and every rc edit AND the
        # ~/.profile fallback failed.
        return False, (f"could not add {bindir} to your PATH (could not edit your "
                       "shell startup files); add it manually")
    return True, None


# --------------------------------------------------------------------------- #
#  Public install / uninstall                                                   #
# --------------------------------------------------------------------------- #

def install(clone_root, precedence: str = "append") -> dict:
    """Make ``localm`` available from any terminal. Non-destructive: creates our
    shim and puts our one bin dir on the user PATH; never overwrites another
    tool's command. Returns a dict the caller records in the install manifest:
    ``{path_dir, shim, path_modified, conflict, precedence, path_note}``
    (conflict = a pre-existing ``localm`` on PATH, or None; path_note = a human
    note when the shim was created but its dir could NOT be put on PATH).

    *precedence* is ``"append"`` (default - behind anything already there, so
    nothing that currently works changes) or ``"prepend"`` (this install's
    command wins). It carries the user's answer to the conflict prompt in
    ``main``."""
    clone_root = Path(clone_root)
    d = bin_dir(clone_root)
    shim = shim_path(clone_root)
    conflict = existing_localm(shim, clone_root)
    path_note = None
    prepend = (precedence == "prepend")

    if sys.platform == "win32":
        _write_win_shim(shim)
        changed = _win_path_add(str(d), prepend=prepend)
    else:
        d.mkdir(parents=True, exist_ok=True)
        target = _venv_localm(clone_root)
        # Replace only our OWN symlink on a re-run; never touch a real file there.
        if shim.is_symlink():
            shim.unlink()
        elif shim.exists():
            # Something that is not our symlink occupies the name - do not clobber.
            return {"path_dir": str(d), "shim": str(shim), "path_modified": False,
                    "conflict": str(shim), "precedence": precedence,
                    "path_note": None}
        shim.symlink_to(target)
        changed, path_note = _posix_ensure_on_path(d, prepend=prepend)

    return {"path_dir": str(d), "shim": str(shim),
            "path_modified": bool(changed), "conflict": conflict,
            "precedence": precedence, "path_note": path_note}


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

def ask_conflict(conflict: str, clone_root, assume_yes: bool = False,
                 ask=None) -> str:
    """Ask which ``localm`` should run. Returns 'priority' | 'keep' | 'skip'.

    The DEFAULT is 'keep', so pressing Enter changes nothing about a command
    that already works. A non-interactive caller gets 'keep' too, and is told
    so rather than being prompted into a hang."""
    out = []
    out.append("  [!] A different 'localm' command already exists:")
    out.append(f"        {conflict}")
    if ask is None:
        try:
            interactive = sys.stdin is not None and sys.stdin.isatty()
        except (AttributeError, ValueError, OSError):
            interactive = False
        ask = input if interactive else None
    if ask is None:
        out.append("      Leaving it in charge and adding this install behind it")
        out.append("      (nothing to ask - this is not an interactive terminal).")
        for line in out:
            print(line)
        return "keep"
    if assume_yes:
        out.append("      Leaving it in charge and adding this install behind it.")
        for line in out:
            print(line)
        return "keep"
    out.append("      Which one should run when you type 'localm'?")
    out.append(f"        [1] This install  ({Path(clone_root).resolve()})")
    out.append("        [2] Keep the existing one - add this install behind it")
    out.append("        [3] Neither - do not touch my PATH")
    if conflict_outranks_user_path(conflict):
        # Never offer a priority we cannot actually deliver: Windows searches the
        # machine PATH before the user PATH, and we only ever write the user one.
        out.append("      Note: that one is in the SYSTEM PATH, which Windows searches")
        out.append("      before your user PATH - so [1] cannot outrank it without an")
        out.append("      administrator change. [1] will still add this install.")
    for line in out:
        print(line)
    try:
        answer = (ask("      Pick 1, 2 or 3 [2]: ") or "").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "keep"
    return {"1": "priority", "3": "skip"}.get(answer, "keep")


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="localm.globalcmd")
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("install", help="add a global `localm` command")
    i.add_argument("--root", default=".")
    i.add_argument("--yes", action="store_true",
                   help="do not prompt on a conflict; keep the existing command "
                        "in charge and add this install behind it")

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
        # Detect the conflict and ask before touching anything: precedence is
        # decided by the answer.
        try:
            conflict = existing_localm(shim_path(args.root), args.root)
        except Exception:
            conflict = None
        precedence = "append"
        if conflict:
            choice = ask_conflict(conflict, args.root, assume_yes=args.yes)
            if choice == "skip":
                print("      Nothing added. Your existing 'localm' is untouched, "
                      "and this install is still usable via its own folder.")
                return 30
            precedence = "prepend" if choice == "priority" else "append"
        try:
            res = install(args.root, precedence=precedence)
        except Exception as e:
            # A real failure (cannot write the shim / edit PATH): report it and
            # exit nonzero, so the installer records nothing in the manifest.
            print(f"  [!] Could not add the global command: {e}")
            return 1
        # Report what was DONE, per the answer given - never a generic claim.
        if res.get("conflict") and res.get("precedence") == "prepend":
            if conflict_outranks_user_path(res["conflict"]):
                print("  [!] This install was put first in your user PATH, but "
                      f"{res['conflict']} is in the system PATH, which Windows "
                      "searches first - so that one still runs. Removing it, or "
                      "an administrator PATH change, is the only way past that.")
            else:
                print("  This install's 'localm' now takes priority. Open a NEW "
                      "terminal to use it.")
        elif res.get("conflict"):
            print(f"  Kept {res['conflict']} in charge, as you chose. This "
                  "install was added behind it, so typing 'localm' still runs "
                  "the existing one.")
        if res.get("path_modified"):
            print("  Added 'localm' to your PATH. Open a NEW terminal to use it.")
            return 0
        note = res.get("path_note")
        if note:
            # The shim was created but its dir could NOT be put on PATH: report
            # the manual step, and still exit 20 (installed, PATH not modified)
            # so the manifest records the shim without --path-modified.
            print(f"  [!] Global command created, but {note}.")
            print("      Add that directory to your PATH, then open a new terminal.")
            return 20
        # Installed, and PATH was already set (a re-run, or the dir was present).
        # Exit 20 tells setup to record the command WITHOUT --path-modified, so
        # uninstall will not take a dir off PATH it did not add.
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
