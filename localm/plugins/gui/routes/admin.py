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
from localm.inference.http_server import require_scope
from localm.plugins.gui.web import LogExportRequest


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
        copied = 0
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
                    except OSError:
                        pass
        except OSError as e:
            raise HTTPException(500, f"Could not write to that folder: {e}")
        if copied == 0:
            # Be honest: nothing was exported (no logs, or all copies failed).
            return {"copied": 0, "dest": str(out),
                    "message": "No log files were found to export."}
        return {"copied": copied, "dest": str(out)}

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

    @app.get("/api/fs/dirs", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def fs_dirs(path: str = "", include_files: bool = False):
        """Subdirectories of *path*, for the coder setup directory picker.

        An empty path lists drive roots on Windows (filesystem root
        elsewhere). Only directory names leave the server - no file
        names or contents. The GUI is localhost + bearer-auth, and the
        coder agent this picker feeds can read those directories anyway.
        """
        if not path:
            if os.name == "nt":
                import string
                roots = [f"{letter}:\\" for letter in string.ascii_uppercase
                         if Path(f"{letter}:\\").is_dir()]
                return {"path": "", "parent": None, "dirs": roots, "files": []}
            path = "/"
        p = Path(path).expanduser()
        if not p.is_dir():
            raise HTTPException(404, f"Not a directory: {path}")
        p = p.resolve()
        dirs = []
        files = []
        try:
            for child in sorted(p.iterdir(), key=lambda c: c.name.lower()):
                try:
                    if not child.name.startswith("."):
                        if child.is_dir():
                            dirs.append(child.name)
                        elif include_files and child.is_file():
                            files.append(child.name)
                except OSError:
                    continue   # broken junction / reparse point
        except PermissionError:
            raise HTTPException(403, f"Permission denied: {path}")
        at_root = p.parent == p
        return {"path": str(p),
                "parent": "" if at_root else str(p.parent),
                "dirs": dirs,
                "files": files}
