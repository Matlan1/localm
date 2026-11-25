# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Python packages an update of the managed ComfyUI installs, fetched before
anything of it changes.

pip's own dry run names what it would install; each package is then fetched, or built,
as a wheel, and pip installs these wheels into the staging folder of the update, which
``update_mechanism.packages`` moves in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from localm.update_mechanism import shared

# Seconds one pip command may take.
_TIMEOUT_S = 7200


def _pip(python: Path, *args: str) -> list:
    return [str(python), "-m", "pip", args[0], "--disable-pip-version-check", *args[1:]]


class _Set:
    """The packages to install, by canonical name: ``(name, version, url, sha256)``."""

    def __init__(self) -> None:
        self.found: dict = {}

    def add(self, report: Path) -> None:
        data = json.loads(report.read_text(encoding="utf-8"))
        for item in data.get("install") or []:
            meta, info = item["metadata"], item.get("download_info") or {}
            hashes = (info.get("archive_info") or {}).get("hashes") or {}
            name = shared.updater_components._canon(meta["name"])
            self.found[name] = (meta["name"], meta["version"], info.get("url") or "",
                                hashes.get("sha256"))


def prepare(place, python: Path, requirements: Path, *, on_progress, say: Callable,
            run: Callable) -> bool:
    """Stage what brings the managed venv of *python* to this machine's PyTorch
    (``managed_comfy_fresh.comfy_torch_spec``) and to *requirements*, for the update
    *place* (``update_mechanism.steps.Place``). Returns False when nothing has to be
    installed. *run* is ``managed_comfy_provision._run``. Raises
    ``managed_comfy_update.PrepareError`` and OSError."""
    from localm.media.managed_comfy_provision import _tail
    from localm.media.managed_comfy_update import PrepareError
    from localm.update_mechanism import disk, mover, packages, steps, steps_packages
    work = Path(disk.short_path(place.path))
    wheels = work / steps_packages.PIP_WHEELS
    wheels.mkdir(exist_ok=True)
    site = shared.updater_components.site_packages(python.parent.parent)
    if site is None:
        raise PrepareError(f"{python.parent.parent} is not a Python environment")

    def pip(what: str, *args: str) -> None:
        ok, out = run(_pip(python, *args), on_progress=on_progress, timeout=_TIMEOUT_S)
        if not ok:
            raise PrepareError(f"{what} failed: {_tail(out)}")

    def report(name: str, what: str, *args: str) -> None:
        path = work / name
        pip(what, "install", "--dry-run", "--quiet", "--report", str(path), *args)
        try:
            wanted.add(path)
        except (ValueError, KeyError, TypeError, RecursionError) as e:
            raise PrepareError(f"{what} failed: what pip would install could not be read "
                               f"({e})") from e

    def fetch(what: str) -> None:
        new = [p for p in wanted.found.values() if p[2] and not p[2].startswith("file:")
               and p[0] not in fetched]
        if not new:
            return
        lines = [f"{name} @ {url}" + (f"#sha256={sha}" if sha else "")
                 for name, _version, url, sha in new]
        listing = work / "pip-fetch.txt"
        listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
        pip(what, "wheel", "--no-deps", "--wheel-dir", str(wheels), "-r", str(listing))
        fetched.update(p[0] for p in new)

    wanted, fetched = _Set(), set()
    torch = torch_args(site)
    constraints: list = []
    if torch:
        say("Fetching the pinned PyTorch for the managed venv ...")
        report("pip-torch.json", "Installing the pinned PyTorch", *torch)
        fetch("Installing the pinned PyTorch")
        pins = work / "pip-torch.txt"
        pins.write_text("".join(f"{n}=={v}\n" for n, v, _u, _s in wanted.found.values()),
                        encoding="utf-8")
        constraints = ["-c", str(pins), "--find-links", str(wheels)]
    if requirements.is_file():
        say("Fetching what ComfyUI requires ...")
        report("pip-requirements.json", "Reinstalling ComfyUI requirements", *constraints, "-r",
               disk.short_path(requirements))
        fetch("Reinstalling ComfyUI requirements")
    if not wanted.found:
        return False
    listing = work / steps_packages.PIP_LIST
    listing.write_text("".join(f"{n}=={v}\n" for n, v, _u, _s in wanted.found.values()),
                       encoding="utf-8")
    try:
        steps.require_space(site, sum(_unpacked(w) for w in wheels.iterdir()),
                            "installing the packages ComfyUI requires")
        stage = packages.stage_folder(place.root, place.id, step=mover.VENV)
        if not disk.remove_tree(stage):
            raise PrepareError(f"{stage} cannot be emptied")
        short = place.beside(f".pip-{place.id[:8]}")
        short.mkdir()
        pip("Reinstalling ComfyUI requirements", "install", "--no-index", "--no-deps",
            "--ignore-installed", "--no-compile", "--no-input", "--find-links", str(wheels),
            "--prefix", disk.short_path(short), "-r", str(listing))
        disk.move(short, stage, disk.Wait())
        packages.note_staged(place.root, place.id, mover.VENV, wanted.found)
    except mover.Refused as e:
        raise PrepareError(str(e)) from e
    disk.remove_tree(place.path / steps_packages.PIP_WHEELS)
    return True


def _unpacked(wheel: Path) -> int:
    import zipfile
    with zipfile.ZipFile(wheel) as z:
        return sum(i.file_size for i in z.infolist())


def torch_args(site: Path) -> Optional[list]:
    """The pip arguments that install the PyTorch this machine's managed ComfyUI uses,
    None when the packages in *site* are the exact versions it pins, or it pins none."""
    from localm.media.managed_comfy_fresh import comfy_torch_install_args, comfy_torch_spec
    args = comfy_torch_install_args(comfy_torch_spec())
    pinned = [r for r in shared.updater_components._requirements_in(args) if "==" in r]
    return args if pinned and (
        shared.updater_components.unsatisfied(pinned, shared.updater_components.dists(site))) else None


