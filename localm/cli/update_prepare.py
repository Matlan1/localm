# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm's side of the steps of an update (``localm.update_mechanism.steps_update``):
what the new version prepares before anything changes, and what it records when the
update is done.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

Say = Callable[[str], None]


class PrepareFailed(Exception):
    """The update cannot be made; ``str()`` is the reason for the user."""


# ------------------------------------------------------------------ prepare

def prepare(root: Path, update_id: str, uv: str, *, say: Say) -> int:
    """Prepare the steps of the update *update_id* of the install at *root* to the tree
    this code is in, with the uv at *uv*, and write them to the update's folder
    (``steps_update.PREPARED``). Returns the exit status, 1 with ``refused`` in that
    file when the update cannot be made."""
    from localm.update_mechanism import disk, mover, steps, steps_update
    work = mover.work_folder(root, update_id)
    try:
        prepared = _prepare(Path(root), update_id, uv, say)
    except (PrepareFailed, *steps.STEP_ERRORS) as e:
        disk.write_json(work / steps_update.PREPARED, {"refused": str(e)})
        return 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        disk.write_json(work / steps_update.PREPARED,
                        {"refused": f"the new version could not prepare its parts: "
                                    f"{type(e).__name__}: {e}"})
        return 1
    disk.write_json(work / steps_update.PREPARED, prepared)
    return 0


def _prepare(root: Path, update_id: str, uv: str, say: Say) -> dict:
    from localm import updater, updater_components as uc, updater_runtime as ur
    from localm.update_mechanism import steps_update
    tree = Path(__file__).resolve().parents[2]
    out: dict = {"format": steps_update.FORMAT, "torch": [], "uv": None, "parts": [], "kept": [], "notes": []}
    plans = {p["name"]: p for p in uc.plan(tree, root, deep=True)}
    out["notes"] += [p["note"] for p in plans.values() if p["note"] and not p["update"]]
    runtime = ur.plan(tree, root)
    if runtime["note"] and not runtime["reprovision"]:
        out["notes"].append(runtime["note"])
    env = updater._deps_env(root)
    if plans["pytorch"]["update"]:
        out["torch"] = _prepare_torch(root, tree, update_id, uv, plans["pytorch"], env, say)
    if plans["uv"]["update"]:
        out["uv"] = _prepare_uv(root, tree, update_id, plans["uv"], say)
    if runtime["reprovision"]:
        part = _prepare_runtime(root, update_id, runtime, out, say)
        if part:
            out["parts"].append(part)
    for name in ("sdcpp", "koboldcpp"):
        if plans[name]["update"]:
            part = _prepare_folders(root, update_id, name, plans[name], out, say)
            if part:
                out["parts"].append(part)
    if plans["comfyui"]["update"]:
        out["parts"].append(_prepare_comfyui(root, update_id, plans["comfyui"], say))
    return out


def _located(root: Path, folder: Path) -> str:
    from localm.update_mechanism import job
    return job.part_path(root, folder)


def _prepare_torch(root: Path, tree: Path, update_id: str, uv: str, plan: dict, env: dict,
                   say: Say) -> list:
    from localm import updater_components as uc
    from localm.update_mechanism import steps_packages
    future = steps_packages.Future(root, update_id, uv, env=env, say=say, tree=tree)
    return uc.fetch_pytorch(future, plan["items"], say)


def _prepare_uv(root: Path, tree: Path, update_id: str, plan: dict, say: Say) -> str:
    from localm import updater_components as uc
    from localm.update_mechanism import mover, steps
    folder = steps.Place(root, update_id).new(mover.UV)
    problem = uc.stage_uv(plan["target"], tree, folder, say)
    if problem:
        raise PrepareFailed(f"the uv {plan['target']} for localm could not be fetched: {problem}")
    return plan["target"]


def _prepare_runtime(root: Path, update_id: str, plan: dict, out: dict, say: Say) -> Optional[dict]:
    """Fetch and test the runtime of the new tree beside the installed one. None when it
    does not work here and the installed one works with the new tree: that one is kept."""
    from localm import setup_llama as sl, updater_runtime as ur
    from localm.setup_llama.library_files import _BLAS_LIBRARY_DIRS
    from localm.setup_llama.runtime_dir import _PRESERVED_TARGET_FILES
    from localm.setup_spec import VC_RUNTIME_ABSENT_EXIT
    from localm.update_mechanism import job, steps
    live = Path(os.path.realpath(ur.runtime_lib(root)))
    folder, backend = live.parent, plan["backend"]
    why = steps.unsettled(folder, wait_s=steps.SETTLE_WAIT_S)
    if why:
        raise PrepareFailed(f"the native runtime cannot be replaced now: {why}")
    steps.sweep(folder)
    job.own_part(root, update_id, folder)
    place = steps.Place(folder, update_id)
    staged = place.beside(f"{live.name}.{update_id[:8]}")
    say(f"Fetching the {backend} runtime {plan['target']}...")
    code, last = _run_localm(["setup-llama", "--backend", backend, "--into", str(staged),
                              "--yes"], say)
    part = {"kind": "runtime", "label": f"{backend} runtime", "root": _located(root, folder),
            "name": live.name, "beside": staged.name, "backend": backend,
            "from": plan["current"] or "", "to": plan["target"], "unproven": False,
            "keep": list(_PRESERVED_TARGET_FILES), "folders": sorted(_BLAS_LIBRARY_DIRS)}
    if code == VC_RUNTIME_ABSENT_EXIT:
        out["notes"].append("no model can load until the Microsoft Visual C++ Redistributable "
                            "is installed: run 'localm setup-llama', which offers to install it")
        return {**part, "unproven": True}
    if code == 0:
        built = sl._provisioned_build(staged)
        if sl._provisioned_backend(staged) != backend or not ur._is_current(
                staged, backend, built, plan["target"], plan["pins"], after=True):
            raise PrepareFailed(f"setup-llama provided the {sl._provisioned_backend(staged)} "
                                f"runtime {built}, not {backend} {plan['target']}")
        return part
    job.discard(folder, update_id)
    if code != sl.BUILD_UNUSABLE_EXIT:
        raise PrepareFailed(f"the {backend} runtime {plan['target']} could not be fetched: {last}")
    loads, why_not = ur._runtime_loads(live, backend, sl)
    if not loads:
        raise PrepareFailed(f"the {backend} runtime {plan['target']} does not work here ({last}), "
                            f"and the installed one does not work with the new version "
                            f"({why_not})")
    out["notes"].append(f"kept the {backend} runtime {plan['current']}: this release's "
                        f"{plan['target']} did not work here ({last})")
    out["kept"].append({"backend": backend, "kept": plan["current"] or "",
                        "target": plan["target"], "why": last})
    return None


def _run_localm(args: list, say: Say) -> tuple:
    from localm.update_mechanism import steps_update
    return steps_update.run_child([sys.executable, "-m", "localm", *args], cwd=os.getcwd(),
                                  env=os.environ, say=say)


def _prepare_folders(root: Path, update_id: str, name: str, plan: dict, out: dict,
                     say: Say) -> Optional[dict]:
    """Fetch and test each build of the sd.cpp or KoboldCpp runtime the update installs.
    One that does not work on this machine is left out, with a note."""
    from localm.media.koboldcpp import runtime as kobold
    from localm.media.sdcpp import runtime as sdcpp
    from localm.update_mechanism import job, mover, steps
    runtime = sdcpp if name == "sdcpp" else kobold
    base = runtime.runtimes_root()
    from localm.media.runtime_files import settle_runtimes
    try:
        settle_runtimes(base, PrepareFailed)
        steps.sweep(base)
        job.own_part(root, update_id, base)
        into = steps.Place(base, update_id).new(mover.WHOLE)
        for item in plan["items"]:
            try:
                runtime.install(item, into=into, on_progress=say)
            except sdcpp.VcRuntimeMissingError as e:
                out["notes"].append(_left_out(name, plan, item, str(e)))
            except (sdcpp.LoadTestError, kobold.LoadTestError) as e:
                out["notes"].append(_left_out(name, plan, item, f"it does not work here: {e}"))
                if runtime is sdcpp:
                    mark = into / f"{plan['target']}-{item}"
                    mark.mkdir(exist_ok=True)
                    (mark / sdcpp.FAILED_MARKER).write_text(json.dumps(
                        {"tag": plan["target"], "backend": item, "reason": str(e)}),
                        encoding="utf-8")
    except (sdcpp.ProvisionError, kobold.ProvisionError) as e:
        raise PrepareFailed(f"the {plan['label']} {plan['target']} could not be fetched: {e}") from e
    if not steps.staged_files(into):
        job.discard(base, update_id)
        return None
    return {"kind": "folders", "label": plan["label"], "root": _located(root, base)}


def _left_out(name: str, plan: dict, item: str, why: str) -> str:
    """The note for a build of a native media runtime the update leaves out."""
    what, retry = (("Image and video generation", f"localm setup-sdcpp --backend {item} --force")
                   if name == "sdcpp" else
                   ("Music generation", "localm setup-music --force"))
    return (f"the {item} build of the {plan['label']} {plan['target']} was not installed: {why}. "
            f"The installed one ({plan['current']}) stays, and this version cannot use it: {what} "
            f"with the {item} build does not work until it is installed. To try again: {retry}")


def _prepare_comfyui(root: Path, update_id: str, plan: dict, say: Say) -> dict:
    from localm.media import managed_comfy as mc
    from localm.media import managed_comfy_update as mcu
    from localm.update_mechanism import job, steps
    folder = mc.managed_comfy_paths().root
    why = mc.why_managed_comfy_is_unsettled(steps.SETTLE_WAIT_S)
    if why:
        raise PrepareFailed(f"the managed ComfyUI cannot be updated now: {why}")
    acquired, busy = mc._acquire_update_lock("update")
    if not acquired:
        raise PrepareFailed(str(busy))
    try:
        steps.sweep(folder)
        job.own_part(root, update_id, folder)
        prepared = mcu.prepare(steps.Place(folder, update_id), folder, repo=mcu.COMFYUI_REPO,
                               target=mcu.COMFYUI_PINNED_COMMIT, version=None, reinstall=True,
                               on_progress=say, say=say)
    except mcu.PrepareError as e:
        raise PrepareFailed(f"the managed ComfyUI could not be prepared: {e}") from e
    finally:
        mc._release_update_lock()
    return {"kind": "comfyui", "label": plan["label"], "root": _located(root, folder),
            "removed": prepared.removed, "packages": prepared.packages,
            "indexes": mcu.package_indexes()}


# ------------------------------------------------------------------- record

def record(old: str, new: str, kept: list) -> list:
    """Record what the update from *old* to *new*, which is done, changed: that the
    install record of localm was kept, the installed runtime's files and build, the
    runtimes *kept* (``steps_update.Run.kept``). Returns one line per record that could not be written."""
    from localm import install_manifest, updater, updater_parts, updater_runtime as ur
    from localm import setup_llama as sl
    root = updater.repo_root()
    lib = ur.runtime_lib(root)
    problems: list = []

    def attempt(what: str, call: Callable[[], object]) -> None:
        try:
            call()
        except Exception as e:
            problems.append(f"{what}: {e}")

    attempt("the install record", lambda: updater_parts.note_kept_record(root, old, new))
    attempt("the record of the runtime's files", lambda: install_manifest.refresh_lib(lib))
    for entry in kept:
        attempt("the kept runtime", lambda e=entry: ur._record_kept(
            lib, e["backend"], e["kept"], e["target"], e["why"]))
    backend, build = sl._provisioned_backend(lib), sl._provisioned_build(lib)
    if backend and build:
        attempt("the runtime history", lambda: sl._record_runtime_history(backend, build))
    return problems
