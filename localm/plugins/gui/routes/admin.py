# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI local-admin routes: log export, the ComfyUI launcher writer, and the
directory picker.

Extracted verbatim from attach_gui(); behavior unchanged. These are local
filesystem operations gated on CONFIG_READ / CONFIG_WRITE; none need the shared
``ctx``.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException

from localm import scopes
from localm.inference.http_server import require_fs_host, require_scope
from localm.plugins.gui.web import LogExportRequest

# Cap a single /api/fs/dirs listing so pointing the browser at a directory with an
# enormous number of entries cannot spike CPU/IO/memory (one stat() per child with
# meta=true). The picker surfaces `truncated` so the omission is visible (AGENTS
# rule 5), not silently hidden; filtering/navigating narrows it. Module-level so a
# test can lower it without creating thousands of files.
_FS_LIST_CAP = 5000


def register(app: FastAPI, ctx) -> None:

    @app.post("/api/logs/export", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def export_logs(req: LogExportRequest):
        """R30: copy every log of this running instance into a user-chosen folder
        (picked via the GUI's /api/fs/dirs browser). Writes a timestamped
        subfolder so repeated exports never clobber each other. Logs live under
        <home>/logs; a few (e.g. comfy-launch.log) sit in the home root, so we
        sweep both. Returns the counts and the destination path."""
        import shutil
        import time as _time
        from localm.config import home_dir
        from localm.debuglog import logs_dir
        dest = (req.dest or "").strip()
        if not dest:
            raise HTTPException(400, "Choose a destination folder first.")
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
                        # AGENTS rule 5: do NOT swallow a real copy failure into a
                        # false "no files" success. Record the reason so the
                        # response can report it truthfully below.
                        errors.append(f"{p.name}: {ce}")
        except OSError as e:
            raise HTTPException(500, f"Could not write to that folder: {e}")
        if found == 0:
            # Genuinely empty: there were no *.log files anywhere to export.
            return {"copied": 0, "found": 0, "dest": str(out),
                    "message": "No log files were found to export."}
        if copied == 0:
            # Files existed but every copy failed: surface the real failure with a
            # non-200. Never report the empty-case reason here (that would hide a
            # write failure behind a false success - AGENTS rule 5).
            raise HTTPException(
                500, f"Found {found} log file(s) but none could be exported: "
                + "; ".join(errors))
        result = {"copied": copied, "found": found, "dest": str(out)}
        if errors:
            # Partial success: some logs copied, some failed. Report the failures
            # rather than silently dropping them.
            result["warning"] = (
                f"{len(errors)} of {found} log file(s) could not be copied: "
                + "; ".join(errors))
        return result

    @app.post("/api/comfyui/create-launcher", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def create_comfy_launcher(workdir: str):
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

        import os
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

    @app.get("/api/fs/dirs", dependencies=[Depends(require_fs_host)])
    async def fs_dirs(path: str = "", include_files: bool = False,
                      meta: bool = False):
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
        """
        if not path:
            if os.name == "nt":
                import string
                roots = [f"{letter}:\\" for letter in string.ascii_uppercase
                         if Path(f"{letter}:\\").is_dir()]
                result = {"path": "", "parent": None, "dirs": roots, "files": []}
                if meta:
                    # Drives have no meaningful size/mtime; the picker just needs
                    # the names as navigable folders.
                    result["entries"] = [
                        {"name": r, "is_dir": True, "size": None, "mtime": None}
                        for r in roots]
                return result
            path = "/"
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
                    if child.name.startswith("."):
                        continue
                    # Bound the per-child stat() work by number of children
                    # EXAMINED, not just those returned - else a folder of a
                    # million non-indexable files would still stat each one.
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
                            # follow_symlinks=False: report the link's OWN size/
                            # mtime, never the target's (a symlink can point
                            # outside this dir - do not leak target metadata).
                            st = child.stat(follow_symlinks=False)
                            mtime = st.st_mtime
                            if not is_dir:
                                size = st.st_size
                        except OSError:
                            # Unreadable child (permissions, broken link): still
                            # list the name so it can be navigated/reported;
                            # size/mtime stay null rather than faked.
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

    @app.get("/api/fs/places", dependencies=[Depends(require_fs_host)])
    async def fs_places():
        """Quick-access locations for the picker's Places rail: the user's home
        and its standard subfolders (only the ones that exist), plus drive roots
        on Windows (the filesystem root elsewhere). Requires HOST filesystem
        access, same as /api/fs/dirs.

        Every path is derived from ``Path.home()`` - never hardcoded - so a
        relocated profile still resolves, and a subfolder that is absent (a
        localized profile, a machine with no Downloads) is simply omitted rather
        than guessed.
        """
        places = []
        try:
            home = Path.home()
        except (OSError, RuntimeError):
            home = None
        if home is not None and home.is_dir():
            places.append({"label": "Home", "path": str(home), "icon": "home"})
            # Standard English subfolder names. Localized profiles name these
            # differently; we add only the ones that actually exist (no guessing).
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
            for letter in string.ascii_uppercase:
                root = f"{letter}:\\"
                try:
                    if Path(root).is_dir():
                        drives.append({"label": root, "path": root,
                                       "icon": "drive"})
                except OSError:
                    continue
        else:
            drives.append({"label": "/", "path": "/", "icon": "drive"})
        return {"places": places, "drives": drives}
