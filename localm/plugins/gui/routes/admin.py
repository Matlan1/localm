# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI local-admin routes: log export, the ComfyUI launcher writer, the native
app launcher builder, and the directory picker (browse, create folder, rename).

These are local filesystem operations gated on CONFIG_READ / CONFIG_WRITE; none
need the shared ``ctx``.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException

from localm import scopes
from localm.inference.http_server import require_fs_host, require_scope
from localm.pathsafe import confined_name, reject_unsafe_path_string
from localm.plugins.gui.web import FsMkdirRequest, FsRenameRequest, LogExportRequest

# Serializes concurrent /api/app/rebuild-launcher requests. One process, one lock.
_launcher_build_lock = threading.Lock()

# Caps a single /api/fs/dirs listing; the response reports `truncated` when the cap
# is hit. Module-level so a test can lower it.
_FS_LIST_CAP = 5000

# The filesystem root the picker lists when given an empty path on POSIX (Windows
# enumerates drive letters instead). Module-level so a test can point it at a
# disposable fixture directory.
_POSIX_FS_ROOT = "/"


def _norm_path_str(s: str) -> str:
    """Lexically normalize a path string for comparison. PURE STRING WORK - no
    filesystem access - so it is safe to run on unsanitized caller input before
    any syscall. normpath collapses separators, "." and lexical ".."; normcase
    lowercases and maps "/" to "\\" on Windows."""
    return os.path.normcase(os.path.normpath(s))


def _network_drives_allowed() -> bool:
    """Whether a mapped network drive (``Z:\\...``) may be treated as an
    ordinary local folder by the host-filesystem routes below, read from
    config's ``allow_network_drives``. Read fresh (one cheap config load)
    rather than cached, matching every other route in this file that reads
    config per-request, so a change takes effect on the next call with no
    restart."""
    from localm.config import load_config
    return bool(load_config().get("allow_network_drives", True))


def register(app: FastAPI, ctx) -> None:

    @app.post("/api/logs/export",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE)),
                            Depends(require_fs_host)])
    def export_logs(req: LogExportRequest):
        """Copy every log of this running instance into a user-chosen folder
        (picked via the GUI's /api/fs/dirs browser). Writes a timestamped
        subfolder so repeated exports never clobber each other. Logs live under
        <home>/logs; a few (e.g. comfy-launch.log) sit in the home root, so both
        are swept. Returns the counts and the destination path.

        Plain `def`, NOT `async def`: this stats a caller-supplied directory and
        then runs a whole shutil.copy2 loop, all blocking. Starlette threadpools
        a sync handler; an async one would hold the event loop for the length of
        the copy.

        require_fs_host as well as CONFIG_WRITE: `dest` is an arbitrary host
        directory that this route mkdir(parents=True)s and writes into, and the
        is_dir check before it makes the 400-vs-200 split a directory-existence
        oracle for the whole disk. It is the same dial the /api/fs/dirs picker
        that SUPPLIES `dest` requires."""
        import shutil
        import time as _time
        from localm.config import home_dir
        from localm.debuglog import logs_dir
        dest = (req.dest or "").strip()
        if not dest:
            raise HTTPException(400, "Choose a destination folder first.")
        # Runs BEFORE is_dir(): the destination is caller-supplied and reaches the
        # filesystem.
        try:
            reject_unsafe_path_string(
                dest, reject_network_drives=not _network_drives_allowed())
        except ValueError as e:
            raise HTTPException(400, f"Invalid destination: {e}")
        dest_dir = Path(dest).expanduser()
        if not dest_dir.is_dir():
            raise HTTPException(400, "That folder does not exist.")
        sources = []
        try:
            sources.append(logs_dir())
            sources.append(home_dir())               # home root for stray *.log
        except Exception:
            pass
        out = dest_dir / f"localm-logs-{_time.strftime('%Y%m%d-%H%M%S')}"
        seen: set = set()
        found = 0            # *.log candidates seen (files that DO exist)
        copied = 0           # successfully copied
        errors: list = []    # per-file copy failures, with the real reason
        try:
            out.mkdir(parents=True, exist_ok=True)
            used: set = set()
            for src in sources:
                if not src or not src.is_dir():
                    continue
                for p in src.glob("*.log"):
                    if p.resolve() in seen:
                        continue
                    seen.add(p.resolve())
                    found += 1
                    # Two logs can share a basename across home/ and home/logs/;
                    # disambiguate so the second does not clobber the first.
                    target = p.name
                    n = 1
                    while target in used:
                        target = f"{p.stem}-{n}{p.suffix}"
                        n += 1
                    used.add(target)
                    try:
                        shutil.copy2(p, out / target)
                        copied += 1
                    except OSError as ce:
                        # Record the error so the response reports the copy failure
                        # below rather than a "no files" success.
                        errors.append(f"{p.name}: {ce}")
        except OSError as e:
            raise HTTPException(500, f"Could not write to that folder: {e}")
        if found == 0:
            # Genuinely empty: there were no *.log files anywhere to export.
            return {"copied": 0, "found": 0, "dest": str(out),
                    "message": "No log files were found to export."}
        if copied == 0:
            # Files existed but every copy failed: report it with a non-200, not with
            # the empty-case message.
            raise HTTPException(
                500, f"Found {found} log file(s) but none could be exported: "
                + "; ".join(errors))
        result = {"copied": copied, "found": found, "dest": str(out)}
        if errors:
            # Partial success: some logs copied, some failed. Report the failures.
            result["warning"] = (
                f"{len(errors)} of {found} log file(s) could not be copied: "
                + "; ".join(errors))
        return result

    @app.post("/api/comfyui/create-launcher", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    def create_comfy_launcher(workdir: str):
        # Plain `def`, NOT `async def`: every branch below touches the filesystem
        # (resolve/is_dir/exists/write_text), and Starlette runs a sync handler in its
        # threadpool instead of on the event loop.
        if not workdir:
            raise HTTPException(400, "Missing workdir parameter")

        from localm.config import load_config
        cfg = load_config()

        valid_workdirs = []
        if cfg.get("comfy_workdir"):
            valid_workdirs.append(cfg["comfy_workdir"])

        plugins_cfg = cfg.get("plugins", {})
        if isinstance(plugins_cfg, dict):
            for p_cfg in plugins_cfg.values():
                if isinstance(p_cfg, dict):
                    c_cfg = p_cfg.get("comfy", {})
                    if isinstance(c_cfg, dict) and c_cfg.get("workdir"):
                        valid_workdirs.append(c_cfg["workdir"])

        if not valid_workdirs:
            raise HTTPException(400, "ComfyUI working directory is not configured")

        # Allowlist the RAW string BEFORE any filesystem call, so no syscall ever
        # touches a path that is not already configured.
        #
        # normcase+normpath is pure string work (no filesystem access): it absorbs
        # separator, trailing-separator, "." and lexical ".." spelling differences,
        # and case on Windows. It is a PREFILTER, not a replacement - the resolved
        # exact-match below still runs, so it can only make the gate stricter. A
        # symlink that merely POINTS at a configured workdir, and a relative spelling
        # of one, are refused here.
        if _norm_path_str(workdir) not in {_norm_path_str(v) for v in valid_workdirs}:
            raise HTTPException(403, "workdir must match a configured comfy_workdir")

        try:
            p = Path(workdir).resolve()

            resolved_valids = [Path(v).resolve() for v in valid_workdirs]
            if p not in resolved_valids:
                raise HTTPException(403, "workdir must match a configured comfy_workdir")

            if not p.is_dir():
                raise HTTPException(400, "workdir is not a valid directory")

            if os.name == "nt":
                script = p / "launch-comfyui.bat"
                if script.exists():
                    raise HTTPException(409, "Launcher script already exists")
                script.write_text("@echo off\r\ncd /d \"%~dp0\"\r\nif exist venv\\Scripts\\activate (call venv\\Scripts\\activate)\r\npython main.py\r\npause\r\n", encoding="utf-8")
            else:
                script = p / "launch-comfyui.sh"
                if script.exists():
                    raise HTTPException(409, "Launcher script already exists")
                script.write_text("#!/bin/bash\ncd \"$(dirname \"$0\")\"\n[ -f venv/bin/activate ] && source venv/bin/activate\npython3 main.py\n", encoding="utf-8")
                script.chmod(0o755)
        except PermissionError:
            raise HTTPException(403, "Permission denied while checking or writing the launcher script")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Failed to create launcher: {e}")

        return {"status": "ok"}

    @app.post("/api/app/rebuild-launcher",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    def rebuild_launcher(force: bool = False):
        """The GUI form of `localm make-launcher` (localm/cli/maintenance.py):
        refresh the copied interpreter, with `--force` mirroring the CLI flag
        exactly. make_launcher() already returns a structured LauncherResult and
        never raises, so this is a thin passthrough.

        Plain `def`, NOT `async def`: make_launcher() copies the interpreter
        (+ DLLs) and spawns a self-check subprocess, all blocking - same
        reasoning as export_logs/create_comfy_launcher above. Starlette
        threadpools a sync handler instead of stalling the event loop.

        _launcher_build_lock: make_launcher() has no locking of its own, and
        --force's fast path overwrites the launcher exe + DLLs, so two
        overlapping GUI requests would copy to the same destination file at
        once. This serializes IN-PROCESS only (concurrent GUI requests); it
        cannot see a concurrent terminal `localm make-launcher`, and two
        terminals can still race each other."""
        if not _launcher_build_lock.acquire(blocking=False):
            raise HTTPException(409, "A launcher rebuild is already in progress.")
        try:
            from localm import applaunch
            res = applaunch.make_launcher(force=force)
        finally:
            _launcher_build_lock.release()
        return {
            "ok": res.ok,
            "path": str(res.path) if res.path else None,
            "desktop_file": str(res.desktop_file) if res.desktop_file else None,
            "icon_stamped": res.icon_stamped,
            "notes": res.notes,
        }

    @app.get("/api/fs/dirs", dependencies=[Depends(require_fs_host)])
    def fs_dirs(path: str = "", include_files: bool = False,
                meta: bool = False, include_hidden: bool = False):
        """Directory listing for the GUI file/folder picker.

        Requires HOST filesystem access (owner / open mode / a key granted
        fs_access=host) - a merely config-reading key cannot enumerate the disk.

        An empty path lists drive roots on Windows (filesystem root
        elsewhere). Only names (and, with ``meta=true``, each child's size +
        modification time) leave the server - never file contents.

        ``include_files=true`` lists files too (folder-only pickers leave it
        off). ``meta=true`` additionally returns an ``entries`` list of
        ``{name, is_dir, size, mtime}`` so the picker can show sizes and dates;
        the flat ``dirs``/``files`` arrays stay for older callers. A listing over
        ``_FS_LIST_CAP`` entries is truncated with ``truncated: true``.

        ``include_hidden=false`` (default) skips dot-prefixed entries, same as
        a plain ``ls``. Dot-DIRECTORIES are where several tools this app's own
        users interoperate with keep their data (``~/.ollama``, ``~/.lmstudio``,
        ``~/.cache/huggingface``), so the GUI picker always passes
        ``include_hidden=true`` and offers its own "Show hidden" toggle
        client-side rather than making that data unreachable by browsing. The
        server-side default stays off for any other caller.

        Plain `def`, NOT `async def`: is_dir/resolve/iterdir/stat all block, and
        blocking inside an async handler stalls the event loop for every other
        request. Starlette threadpools a sync handler instead. The path itself
        stays UNCONSTRAINED (browsing the host disk is the entire point of the
        picker); only the UNC/device form is refused, before any syscall.
        """
        # Runs BEFORE Path()/is_dir(): a UNC path here dials SMB from the event loop.
        # No require_absolute - the picker accepts "~/..." (expanded below) and
        # relative input.
        allow_net = _network_drives_allowed()
        if path:
            try:
                reject_unsafe_path_string(path, reject_network_drives=not allow_net)
            except ValueError as e:
                raise HTTPException(400, f"Invalid path: {e}")
        if not path:
            if os.name == "nt":
                import string
                from localm.pathsafe import is_mapped_network_drive
                # Skip a mapped network drive from the root listing entirely when
                # disallowed, so it is not offered as clickable.
                roots = [f"{letter}:\\" for letter in string.ascii_uppercase
                         if Path(f"{letter}:\\").is_dir()
                         and (allow_net or not is_mapped_network_drive(f"{letter}:\\"))]
                result = {"path": "", "parent": None, "dirs": roots, "files": []}
                if meta:
                    # Drives have no meaningful size/mtime; the picker needs only the
                    # names as navigable folders.
                    result["entries"] = [
                        {"name": r, "is_dir": True, "size": None, "mtime": None}
                        for r in roots]
                return result
            path = _POSIX_FS_ROOT
        p = Path(path).expanduser()
        if not p.is_dir():
            raise HTTPException(404, f"Not a directory: {path}")
        p = p.resolve()
        dirs = []
        files = []
        entries = []
        truncated = False
        scanned = 0
        try:
            for child in sorted(p.iterdir(), key=lambda c: c.name.lower()):
                try:
                    if not include_hidden and child.name.startswith("."):
                        continue
                    # Bound the per-child stat() work by number of children EXAMINED,
                    # not just those returned.
                    scanned += 1
                    if scanned > _FS_LIST_CAP:
                        truncated = True
                        break
                    is_dir = child.is_dir()
                    if not is_dir and not (include_files and child.is_file()):
                        # A file when only dirs were requested, or a non-file
                        # non-dir (socket, device): not selectable, skip it.
                        continue
                    (dirs if is_dir else files).append(child.name)
                    if meta:
                        size = mtime = None
                        try:
                            # follow_symlinks=False: report the link's OWN size and
                            # mtime, never the target's.
                            st = child.stat(follow_symlinks=False)
                            mtime = st.st_mtime
                            if not is_dir:
                                size = st.st_size
                        except OSError:
                            # Unreadable child (permissions, broken link): still list
                            # the name; size/mtime stay null rather than faked.
                            pass
                        entries.append({"name": child.name, "is_dir": is_dir,
                                        "size": size, "mtime": mtime})
                except OSError:
                    continue   # broken junction / reparse point
        except PermissionError:
            raise HTTPException(403, f"Permission denied: {path}")
        at_root = p.parent == p
        result = {"path": str(p),
                  "parent": "" if at_root else str(p.parent),
                  "dirs": dirs,
                  "files": files,
                  "truncated": truncated}
        if meta:
            result["entries"] = entries
        return result

    @app.post("/api/fs/mkdir",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE)),
                            Depends(require_fs_host)])
    def fs_mkdir(req: FsMkdirRequest):
        """Create a new folder inside a directory the picker is currently
        browsing. Same gating as the other host-filesystem WRITE routes in this
        file (export_logs, create_comfy_launcher): CONFIG_WRITE (a state change)
        plus require_fs_host (the parent, like fs_dirs' own `path`, is an
        arbitrary caller-chosen host location, not one already on an allowlist).

        Plain `def`, NOT `async def`: is_dir/resolve/mkdir all block, same
        reasoning as fs_dirs above.
        """
        parent_raw = (req.path or "").strip()
        name = (req.name or "").strip()
        if not parent_raw:
            raise HTTPException(400, "Missing parent folder")
        if not name:
            raise HTTPException(400, "Missing folder name")
        # Runs BEFORE Path()/is_dir(): a UNC parent dials SMB.
        try:
            reject_unsafe_path_string(
                parent_raw, reject_network_drives=not _network_drives_allowed())
        except ValueError as e:
            raise HTTPException(400, f"Invalid path: {e}")
        parent = Path(parent_raw).expanduser()
        if not parent.is_dir():
            raise HTTPException(404, f"Not a directory: {parent_raw}")
        parent = parent.resolve()
        # confined_name rejects a separator / reserved-char / ".." / empty name
        # LEXICALLY before it resolves anything, then verifies the resolved child's
        # parent is exactly `parent`. `name` is proven separator-free before that
        # resolve(), so `parent / name` cannot become a UNC string.
        child = confined_name(parent, name)
        try:
            child.mkdir()
        except FileExistsError:
            # mkdir refuses an existing target on every platform; there is no
            # overwrite mode, so this branch is race-free.
            raise HTTPException(409, f"'{name}' already exists")
        except PermissionError:
            raise HTTPException(403, f"Permission denied: {name}")
        except OSError as e:
            raise HTTPException(500, f"Could not create folder: {e}")
        return {"path": str(child), "parent": str(parent), "name": child.name}

    @app.post("/api/fs/rename",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE)),
                            Depends(require_fs_host)])
    def fs_rename(req: FsRenameRequest):
        """Rename a file or folder in place (same parent directory only - this
        is not a move). Same gating as fs_mkdir above.

        NEVER overwrites an existing target. This has to be checked EXPLICITLY
        rather than left to Path.rename()'s own error handling: on POSIX,
        os.rename() SILENTLY REPLACES an existing FILE target (that is
        rename(2)'s documented behavior) while on Windows it raises
        FileExistsError - so relying on the exception alone would make this
        route safe on Windows and silently destructive on Linux/macOS. The
        pre-check closes the ordinary case; a concurrent racer landing a file
        at the new name between the check and the rename() call below is the
        same narrow, accepted window create_comfy_launcher's own
        check-then-write already has for this key's trust level (full host
        filesystem access).

        Plain `def`, NOT `async def`: same reasoning as fs_dirs/fs_mkdir above.
        """
        target_raw = (req.path or "").strip()
        new_name = (req.new_name or "").strip()
        if not target_raw:
            raise HTTPException(400, "Missing path")
        if not new_name:
            raise HTTPException(400, "Missing new name")
        # Runs BEFORE Path()/exists(): a UNC target dials SMB.
        try:
            reject_unsafe_path_string(
                target_raw, reject_network_drives=not _network_drives_allowed())
        except ValueError as e:
            raise HTTPException(400, f"Invalid path: {e}")
        p = Path(target_raw).expanduser()
        if not p.exists():
            raise HTTPException(404, f"Not found: {target_raw}")
        p = p.resolve()
        parent = p.parent
        # confined_name proves new_name separator-free before it resolves
        # anything, so `parent / new_name` cannot become a UNC string.
        new_path = confined_name(parent, new_name)
        if new_path == p:
            raise HTTPException(400, "New name is the same as the current name")
        if new_path.exists():
            raise HTTPException(409, f"'{new_name}' already exists")
        try:
            p.rename(new_path)
        except FileExistsError:
            raise HTTPException(409, f"'{new_name}' already exists")
        except PermissionError:
            raise HTTPException(403, f"Permission denied: {new_name}")
        except OSError as e:
            raise HTTPException(500, f"Could not rename: {e}")
        return {"path": str(new_path), "parent": str(parent), "name": new_path.name}

    @app.get("/api/fs/places", dependencies=[Depends(require_fs_host)])
    def fs_places():
        """Quick-access locations for the picker's Places rail: the user's home
        and its standard subfolders (only the ones that exist), plus drive roots
        on Windows (the filesystem root elsewhere). Requires HOST filesystem
        access, same as /api/fs/dirs.

        Every path is derived from ``Path.home()`` - never hardcoded - so a
        relocated profile still resolves, and a subfolder that is absent (a
        localized profile, a machine with no Downloads) is simply omitted rather
        than guessed.

        Plain `def` for the same reason as fs_dirs: probing A-Z drive roots calls
        is_dir() on each, and a mapped-but-disconnected network drive blocks. No
        caller input reaches it, so there is nothing to REJECT here - but a
        disallowed network drive is still omitted from the listing, same as
        fs_dirs' own empty-path root listing.
        """
        places = []
        try:
            home = Path.home()
        except (OSError, RuntimeError):
            home = None
        if home is not None and home.is_dir():
            places.append({"label": "Home", "path": str(home), "icon": "home"})
            # Standard English subfolder names; only the ones that actually exist are
            # added.
            for label, sub, icon in [("Desktop", "Desktop", "desktop"),
                                     ("Documents", "Documents", "documents"),
                                     ("Downloads", "Downloads", "downloads")]:
                try:
                    d = home / sub
                    if d.is_dir():
                        places.append({"label": label, "path": str(d),
                                       "icon": icon})
                except OSError:
                    continue
        drives = []
        if os.name == "nt":
            import string
            from localm.pathsafe import is_mapped_network_drive
            # Skip a disallowed mapped network drive from the Places rail too, same
            # as fs_dirs' empty-path root listing.
            allow_net = _network_drives_allowed()
            for letter in string.ascii_uppercase:
                root = f"{letter}:\\"
                try:
                    if Path(root).is_dir() and (allow_net or not is_mapped_network_drive(root)):
                        drives.append({"label": root, "path": root,
                                       "icon": "drive"})
                except OSError:
                    continue
        else:
            drives.append({"label": "/", "path": "/", "icon": "drive"})
        return {"places": places, "drives": drives}
