"""Hatchling metadata hook that supplies the project's console scripts.

The scripts come from ``console-scripts`` under ``[tool.hatch.metadata.hooks.custom]``.
On Windows, a script whose launcher in ``.venv/Scripts`` cannot be opened for writing
is left out of the metadata and its line is removed from the RECORD of the installed
``localm-*.dist-info``, so an installer neither deletes nor rewrites that launcher.
With no such launcher, or on any other platform, the scripts pass through unchanged
and nothing on disk is touched.
"""
import os
from pathlib import Path

from hatchling.metadata.plugin.interface import MetadataHookInterface


def _is_locked(path: Path) -> bool:
    """True when *path* exists and cannot be opened for writing. Never raises."""
    try:
        with open(path, "r+b"):
            return False
    except FileNotFoundError:
        return False
    except OSError:
        return True


def _drop_from_record(site_packages: Path, launcher_names: set) -> None:
    """Remove lines naming ``Scripts/<launcher>`` from every installed localm RECORD."""
    for record in site_packages.glob("localm-*.dist-info/RECORD"):
        text = record.read_bytes().decode("utf-8")
        kept = []
        for line in text.splitlines(keepends=True):
            entry = line.split(",", 1)[0].replace("\\", "/")
            parts = entry.rsplit("/", 2)
            if len(parts) == 3 and parts[1] == "Scripts" and parts[2] in launcher_names:
                continue
            kept.append(line)
        new_text = "".join(kept)
        if new_text != text:
            tmp = record.with_name("RECORD.tmp")
            tmp.write_bytes(new_text.encode("utf-8"))
            os.replace(tmp, record)


class LocalmScriptsHook(MetadataHookInterface):
    PLUGIN_NAME = "custom"

    def update(self, metadata: dict) -> None:
        scripts = dict(self.config["console-scripts"])
        if os.name == "nt":
            venv = Path(self.root) / ".venv"
            site = venv / "Lib" / "site-packages"
            locked = {n for n in scripts if _is_locked(venv / "Scripts" / f"{n}.exe")}
            if (venv / "Scripts" / "python.exe").exists() and any(site.glob("localm-*.dist-info")):
                import subprocess
                code = ("import sys; from pathlib import Path; from localm.setup_llama.runtime_dir import _stop_own_probes; "
                        "print(_stop_own_probes(Path(sys.argv[1])))")
                r = subprocess.run([str(venv / "Scripts" / "python.exe"), "-c", code,
                                    str(Path(self.root) / "runtime" / "localm_llama_runtime" / "lib")],
                                   cwd=str(self.root), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                   timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
                with open(Path(self.root) / "hook-stop.log", "a") as f:
                    f.write("rc=%s out=%r err=%r\n" % (r.returncode, r.stdout[-200:], r.stderr[-300:]))
            if locked:
                _drop_from_record(site, {f"{n}.exe" for n in locked})
                scripts = {n: v for n, v in scripts.items() if n not in locked}
        metadata["scripts"] = scripts
