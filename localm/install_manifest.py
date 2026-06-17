"""Install provenance ledger - so uninstall removes ONLY what install recorded.

The lesson of the famous data-loss incidents (Valve's Steam ``rm -rf
"$STEAMROOT/"*`` wiping ``/`` when the variable was empty; Pop!_OS/apt removing
the desktop as a "dependency") is the same: an uninstaller must never *derive*,
*glob*, or *guess* what to delete. It must delete only the exact paths the
installer wrote down at install time, validate each before touching it, and
refuse anything it does not have a record for.

This module is that record. ``record()`` writes ``.localm-install.json`` listing
the venv, the provisioned native binaries (by name, inside their dir), the
home.cfg, the data directory (and whether WE created it), and the shortcut.
``uninstall()`` reads it back and removes only those items, with hard guards on
the one ``rm -rf`` it ever performs (the data dir) - never a root, drive root,
$HOME, the repo, an ancestor of the repo, a symlink, or a relative/empty path.
It never touches a path that is not in the manifest.

The venv is recorded but NOT removed here: a running interpreter cannot delete
its own venv on Windows, so the installer shell removes it (marker-checked) after
this returns.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

MANIFEST_NAME = ".localm-install.json"
SCHEMA_VERSION = 1
_BIN_SUFFIXES = (".dll", ".so", ".dylib", ".exe", ".pyd")


def manifest_path(root) -> Path:
    return Path(root) / MANIFEST_NAME


def _bin_files(lib_dir: Optional[Path]) -> list:
    """Snapshot the native-binary filenames currently in *lib_dir* (bare names,
    not paths). These are exactly what setup-llama provisioned."""
    out: list = []
    if not lib_dir:
        return out
    try:
        for f in sorted(Path(lib_dir).iterdir()):
            if f.is_file() and (f.suffix.lower() in _BIN_SUFFIXES or ".so." in f.name):
                out.append(f.name)
    except OSError:
        pass
    return out


def _abs(p) -> str:
    return str(Path(p).resolve()) if p else ""


def record(root, *, venv="", lib_dir="", home_cfg="", data_dir="",
           data_created=False, shortcut="", stamp="") -> Path:
    """Write the install manifest under *root*. Paths are stored absolute; the
    binary list is snapshotted from *lib_dir* at call time."""
    data = {
        "schema": SCHEMA_VERSION,
        "stamp": stamp,
        "venv": _abs(venv),
        "lib_dir": _abs(lib_dir),
        "binaries": _bin_files(Path(lib_dir) if lib_dir else None),
        "home_cfg": _abs(home_cfg),
        "data_dir": _abs(data_dir),
        "data_created": bool(data_created),
        "shortcut": _abs(shortcut),
    }
    p = manifest_path(root)
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return p


def load(root) -> Optional[dict]:
    try:
        return json.loads(manifest_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _unsafe_data_dir(path: str, repo: Path) -> Optional[str]:
    """A reason string if *path* is too dangerous for ``rm -rf``, else None.
    Refuses empty/relative/symlink paths, the filesystem or drive root, $HOME,
    the repo, and any ancestor of the repo."""
    if not path:
        return "empty path"
    p = Path(path)
    if not p.is_absolute():
        return "not an absolute path"
    if p.is_symlink():
        return "is a symlink"
    try:
        rp = p.resolve()
    except OSError:
        return "cannot resolve path"
    if rp == rp.parent:                       # the filesystem root or a Windows drive root
        return "filesystem/drive root"
    try:
        if rp == Path.home().resolve():
            return "user home directory"
    except Exception:
        pass
    repo_r = repo.resolve()
    if rp == repo_r:
        return "repository root"
    if repo_r.is_relative_to(rp):             # rp is an ancestor of the repo
        return "an ancestor of the install directory"
    return None


def uninstall(root, *, purge_data=False, dry_run=False, force=False, log=print) -> dict:
    """Plan/execute the uninstall. Each item is classified:

      * recorded - we have a record we created it; removed.
      * warn     - we do NOT remember creating it; removed only with *force*
                   (the caller shows a clear at-your-own-risk warning first and
                   the user chooses to continue). Always reported.
      * refuse   - a categorically dangerous target (filesystem/drive root,
                   $HOME, the repo, an ancestor of it, or a symlink); NEVER
                   removed, even with force. This is the rm -rf / guard.

    With *dry_run* nothing is touched. The venv is reported, not removed."""
    root = Path(root).resolve()
    report = {"removed": [], "skipped": [], "warned": [], "refused": [],
              "venv": "", "ok": True, "no_manifest": False}
    actions = []                              # (path, kind, status, reason)

    m = load(root)
    if m is None:
        # No record. We can still offer the provisioned binaries (a fixed,
        # repo-relative location) as WARNED guesses - never the data dir.
        report["no_manifest"] = True
        default_lib = root / "runtime" / "localm_llama_runtime" / "lib"
        for name in _bin_files(default_lib):
            actions.append((str(default_lib / name), "binary", "warn",
                            "no install record - not certain we created this"))
        report["venv"] = str(root / ".venv")
    elif m.get("schema") != SCHEMA_VERSION:
        log(f"[uninstall] Unrecognised manifest schema {m.get('schema')!r} - "
            "aborting for safety. Nothing removed.")
        report["ok"] = False
        return report
    else:
        lib_dir = m.get("lib_dir", "")
        for name in m.get("binaries", []):
            if not name or "/" in name or "\\" in name or name in (".", ".."):
                report["refused"].append((name, "suspicious binary name in manifest"))
                continue
            if not lib_dir:
                report["refused"].append((name, "no lib_dir recorded"))
                continue
            actions.append((str(Path(lib_dir) / name), "binary", "recorded", ""))
        if m.get("home_cfg"):
            actions.append((m["home_cfg"], "home_cfg", "recorded", ""))
        if m.get("shortcut"):
            actions.append((m["shortcut"], "shortcut", "recorded", ""))
        data_dir = m.get("data_dir", "")
        if purge_data and data_dir:
            reason = _unsafe_data_dir(data_dir, root)
            if reason:
                actions.append((data_dir, "data", "refuse", reason))
            elif not m.get("data_created"):
                actions.append((data_dir, "data", "warn",
                                "not recorded as created by the installer"))
            else:
                actions.append((data_dir, "data", "recorded", ""))
        elif data_dir:
            report["skipped"].append(
                (data_dir, "data kept (pass --purge-data to remove)"))
        report["venv"] = m.get("venv", "")

    def _remove(path: str, kind: str) -> None:
        p = Path(path)
        if not p.exists():
            report["skipped"].append((path, "already gone"))
            return
        if dry_run:
            report["removed"].append(path)
            return
        try:
            if p.is_dir() and kind == "data":
                shutil.rmtree(path)
            elif p.is_dir():
                report["refused"].append((path, f"{kind} is unexpectedly a directory"))
                return
            else:
                p.unlink()
            report["removed"].append(path)
        except OSError as e:
            report["refused"].append((path, f"could not remove: {e}"))

    for path, kind, status, reason in actions:
        if status == "refuse":
            report["refused"].append((path, reason))
            continue
        if status == "warn":
            report["warned"].append((path, reason))
            if not force:
                continue                      # shown, but not removed without force
            if dry_run:
                continue                      # listed under 'warned', don't double-list
        _remove(path, kind)

    if not dry_run and m is not None:
        try:
            manifest_path(root).unlink()
        except OSError:
            pass
    return report


# --------------------------------------------------------------------------- #
#  CLI: invoked by setup.sh / setup.bat (python -m localm.install_manifest)    #
# --------------------------------------------------------------------------- #

def _print_report(rep: dict) -> None:
    for x in rep["removed"]:
        print(f"  remove: {x}")
    for x, why in rep.get("warned", []):
        print(f"  WARN  : {x}  (unverified - {why}; removed only if you continue)")
    for x, why in rep["skipped"]:
        print(f"  keep  : {x}  ({why})")
    for x, why in rep["refused"]:
        print(f"  REFUSE: {x}  (unsafe - {why}; never removed)")
    if rep.get("warned") or rep.get("no_manifest"):
        print("")
        if rep.get("no_manifest"):
            print("  [!] No install record (.localm-install.json) was found.")
        print("  *** Some items above are NOT in our install record - we do not")
        print("      remember creating them. Continuing removes them anyway:")
        print("      here be dragons (possible data loss). Your call. ***")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="localm.install_manifest")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="write the install manifest")
    r.add_argument("--root", default=".")
    r.add_argument("--venv", default="")
    r.add_argument("--lib-dir", default="")
    r.add_argument("--home-cfg", default="")
    r.add_argument("--data-dir", default="")
    r.add_argument("--data-created", action="store_true")
    r.add_argument("--shortcut", default="")
    r.add_argument("--stamp", default="")

    u = sub.add_parser("uninstall", help="remove what the manifest recorded")
    u.add_argument("--root", default=".")
    u.add_argument("--purge-data", action="store_true")
    u.add_argument("--dry-run", action="store_true")
    u.add_argument("--force", action="store_true",
                   help="also remove items with no install record (after the "
                        "caller has warned the user); never overrides the "
                        "catastrophic-path guard")

    args = ap.parse_args(argv)
    if args.cmd == "record":
        p = record(args.root, venv=args.venv, lib_dir=args.lib_dir,
                   home_cfg=args.home_cfg, data_dir=args.data_dir,
                   data_created=args.data_created, shortcut=args.shortcut,
                   stamp=args.stamp)
        print(f"[install] recorded manifest at {p}")
        return 0
    rep = uninstall(args.root, purge_data=args.purge_data,
                    dry_run=args.dry_run, force=args.force)
    _print_report(rep)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
