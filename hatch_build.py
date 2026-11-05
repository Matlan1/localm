"""Hatchling metadata hook that supplies the project's console scripts.

The scripts come from ``console-scripts`` under ``[tool.hatch.metadata.hooks.custom]``.

On Windows, a script whose launcher in this project's ``.venv/Scripts`` cannot be
opened for writing is left out of the metadata and its line is removed from the RECORD
of the installed ``localm-*.dist-info``, so an installer neither deletes nor rewrites
that launcher, and one line on the console says so. The launcher file stays and keeps
starting ``localm``; the next rebuild with the launcher free writes and records it again.

On Windows, a rebuild of an existing install also stops that install's runtime probe
daemon, so the update step that replaces the runtime right after this build finds it free.

With no such launcher, or on any other platform, the scripts pass through unchanged and
no file is touched. Any failure here also passes the scripts through unchanged:
the installer then fails on the locked launcher, as it would without this hook, and an
update that ran it rolls back.

Temporary: updates from 0.3.3 on do not reinstall the project over a running launcher.
"""
import os
import subprocess
from pathlib import Path

from hatchling.metadata.plugin.interface import MetadataHookInterface

STOP_TIMEOUT_S = 20
# updater_runtime.REBUILD_MARKER
REBUILD_MARKER = ".localm-rebuilding"
NOTICE = ("localm is running from {names}, so that file is left as it is. "
          "The update carries on and localm keeps starting the same way.\n")


def _is_locked(path: Path) -> bool:
    """True when *path* exists and cannot be opened for writing. Never raises."""
    try:
        with open(path, "r+b"):
            return False
    except FileNotFoundError:
        return False
    except OSError:
        return True


def _drop_from_record(site_packages: Path, launcher_names: set) -> bool:
    """Remove lines naming ``Scripts/<launcher>`` from every installed localm RECORD.
    True when a RECORD changed."""
    changed = False
    for record in site_packages.glob("localm-*.dist-info/RECORD"):
        text = record.read_bytes().decode("utf-8")
        kept = []
        for line in text.splitlines(keepends=True):
            parts = line.split(",", 1)[0].replace("\\", "/").rsplit("/", 2)
            if len(parts) == 3 and parts[1] == "Scripts" and parts[2] in launcher_names:
                continue
            kept.append(line)
        new_text = "".join(kept)
        if new_text != text:
            tmp = record.with_name("RECORD.tmp")
            tmp.write_bytes(new_text.encode("utf-8"))
            os.replace(tmp, record)
            changed = True
    return changed


def _tell_console(message: str) -> None:
    """Write *message* to the console this process is attached to, which an installer
    that captures the build's output does not hide. With no console nothing is written.
    See test_tell_console_with_no_console_writes_nothing."""
    try:
        with open("CONOUT$", "w", encoding="utf-8") as console:
            console.write(message)
    except OSError:
        pass


def stop_own_probe_daemon(root) -> None:
    """On a rebuild of an existing install, leave :data:`REBUILD_MARKER` in its venv, so
    a probe daemon started from now on does not load the runtime
    (``updater_runtime.replacing_runtime_now``), and stop this install's own running
    daemon, by its ``.venv`` Python and ``updater_runtime.stop_own_probe_daemons``
    (:data:`STOP_TIMEOUT_S` at most). Never raises; a failure leaves the daemon running."""
    root = Path(root)
    python = root / ".venv" / "Scripts" / "python.exe"
    site = root / ".venv" / "Lib" / "site-packages"
    try:
        if not python.exists() or not any(site.glob("localm-*.dist-info")):
            return
        (root / ".venv" / REBUILD_MARKER).write_bytes(b"")
        code = ("from localm import updater, updater_runtime as u; r = updater.repo_root(); "
                "u.stop_own_probe_daemons(u.runtime_lib(r), r)")
        subprocess.run([str(python), "-c", code], cwd=str(root), stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=STOP_TIMEOUT_S, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception as e:
        print(f"localm: could not stop the runtime probe: {e}")


def scripts_for_this_build(root, scripts: dict) -> dict:
    """*scripts* without the ones whose launcher in ``<root>/.venv/Scripts`` is locked,
    with those launchers removed from the installed RECORD. Windows only."""
    venv = Path(root) / ".venv"
    locked = sorted(n for n in scripts if _is_locked(venv / "Scripts" / f"{n}.exe"))
    if not locked:
        return scripts
    names = {f"{n}.exe" for n in locked}
    if _drop_from_record(venv / "Lib" / "site-packages", names):
        _tell_console(NOTICE.format(names=", ".join(sorted(names))))
    return {n: v for n, v in scripts.items() if n not in locked}


class LocalmScriptsHook(MetadataHookInterface):
    PLUGIN_NAME = "custom"

    def update(self, metadata: dict) -> None:
        scripts = dict(self.config["console-scripts"])
        if os.name == "nt":
            stop_own_probe_daemon(self.root)
            try:
                scripts = scripts_for_this_build(self.root, scripts)
            except Exception as e:
                print(f"localm: leaving the launchers as they are: {e}")
        metadata["scripts"] = scripts
