# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which version each part of an install is at, and the repair that brings every
part to the installed localm version.

The parts are the code (``VERSION``), the install record of localm in the venv
(its ``localm-<version>.dist-info``) and the dependencies the code declares,
the native llama.cpp runtime (against the build this version pins for the
installed backend, ``updater_runtime.plan``) and the provisioned components
(against this version's pins, ``updater_components.plan``). :func:`installed_parts`
reports them; :func:`repair` reinstalls the dependencies when they disagree and
records the runtime and component updates this version needs, carried out now
or, in a server, during its restart.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable, Optional

REPAIR_SNAPSHOT = "repair-venv"
# Which localm versions run on which install record because an update kept the
# dependencies and so did not rewrite it: {version: record version}.
KEPT_RECORD_FILE = "kept-install-record.json"


def _norm_version(v: Optional[str]) -> Optional[str]:
    if not v:
        return v
    try:
        from packaging.version import Version
        return str(Version(v))
    except Exception:
        return v.strip()


def _kept_records() -> dict:
    from localm import updater
    try:
        data = json.loads((updater._updates_dir(create=False) / KEPT_RECORD_FILE)
                          .read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return {}
    return data if isinstance(data, dict) else {}


def note_kept_record(root, previous: Optional[str], version: str) -> None:
    """Record that *version* runs on the install record in the venv of the install
    at *root* because an update from *previous* kept the dependencies and did not
    rewrite it (:func:`installed_parts` does not count that pair as a problem).
    Nothing is recorded when the record was already not *previous*'s (a real
    mismatch stays visible). A failure is logged."""
    from localm import updater, updater_components as uc
    site = uc.site_packages(Path(root) / ".venv")
    record = (uc.dists(site).get("localm") or {}).get("version") if site else None
    if record is None:
        return
    data = _kept_records()
    if _norm_version(record) != _norm_version(previous) and data.get(previous) != record:
        return
    data[str(version)] = record
    try:
        (updater._updates_dir() / KEPT_RECORD_FILE).write_text(json.dumps(data),
                                                               encoding="utf-8")
    except OSError as e:
        updater._apply_warn("could not record the kept install record: %s", e)


def _deps_extras() -> list:
    """The extras the dependency step installs (``_apply_update.DEPS_EXTRAS``)."""
    from localm import _apply_update as au
    return list(au.DEPS_EXTRAS)


def _deps(root: Path) -> dict:
    """The venv half: ``{"venv", "record", "problems"}``. ``venv`` is whether
    the install has a venv, ``record`` the version of localm's install record
    in it (None when there is none), ``problems`` each dependency the venv does
    not satisfy, of the declared ones and of the extras the dependency step
    installs."""
    from localm import updater, updater_components as uc
    site = uc.site_packages(root / ".venv")
    if site is None:
        return {"venv": False, "record": None, "problems": []}
    installed = uc.dists(site)
    record = (installed.get("localm") or {}).get("version")
    data = updater._read_toml(root / "pyproject.toml")
    proj = data.get("project", {}) if isinstance(data, dict) else {}
    reqs = [str(r) for r in proj.get("dependencies", []) or []]
    extras = proj.get("optional-dependencies", {}) or {}
    for extra in _deps_extras():
        reqs += [str(r) for r in extras.get(extra, []) or []
                 if not str(r).startswith("localm")]
    return {"venv": True, "record": record, "problems": uc.unsatisfied(reqs, installed)}


def installed_parts(root=None) -> dict:
    """What each part of the install at *root* (default the running one) is
    at, against this localm version. Read-only, never raises.

    Returns ``{"app", "install_record", "install_record_matches",
    "install_record_problem", "deps_problems", "runtime", "components", "problems", "consistent",
    "last_steps"}``:
    ``app`` the code's version, ``install_record`` the venv's localm record
    version (None when there is none to compare: no venv, or a pip install;
    a git clone with a venv is compared like any install) and whether it is
    ``app`` (False also when the venv has
    no record at all). A record of another version is a problem
    (``install_record_problem``) unless an update that kept the dependencies
    left it (:func:`note_kept_record`),
    ``deps_problems`` each dependency the venv does not satisfy (a problem),
    ``runtime``
    ``{"backend", "current", "target", "matches"}`` (``matches`` None when it
    cannot be told), ``components`` ``[{"name", "label", "current",
    "target", "matches"}]`` for every installed component the shallow plan can
    compare, and ``problems`` one user-facing line per part that disagrees.
    ``consistent`` is True when ``problems`` is empty. ``last_steps`` holds
    the runtime and component update results of the last day for this version
    (:func:`_last_steps`)."""
    from localm import _version, updater
    root = Path(root) if root else updater.repo_root()
    app = _version.read_version()
    problems: list = []
    record, record_matches, deps_problems = None, None, []
    record_problem = False
    if not updater.is_site_packages_install(root):
        d = {"venv": False}
        try:
            d = _deps(root)
            record, deps_problems = d["record"], d["problems"]
        except Exception as e:
            problems.append(f"could not read the venv's packages: {e}")
        if d.get("venv") and record is None:
            record_matches, record_problem = False, True
            problems.append("the venv has no install record of localm, so localm is not "
                            "installed in it")
        elif record is not None:
            record_matches = _norm_version(record) == _norm_version(app)
            record_problem = not record_matches and _kept_records().get(app) != record
            if record_problem:
                problems.append(f"the venv's install record is localm {record}, but localm "
                                f"{app} is installed")
        if deps_problems:
            problems.append("the venv does not have the dependencies localm "
                            f"{app} needs: " + ", ".join(deps_problems[:5])
                            + (f" (and {len(deps_problems) - 5} more)"
                               if len(deps_problems) > 5 else ""))
    runtime: dict = {"backend": None, "current": None, "target": None, "matches": None}
    try:
        from localm import updater_runtime
        p = updater_runtime.plan(root, root)
        runtime.update(backend=p.get("backend"), current=p.get("current"),
                       target=p.get("target"))
        if p.get("current") is not None and p.get("target"):
            runtime["matches"] = not p.get("reprovision")
            if p.get("reprovision"):
                problems.append(f"the {p.get('backend')} runtime is {p.get('current')}, but "
                                f"localm {app} uses {p.get('target')}")
    except Exception as e:
        problems.append(f"could not compare the native runtime with this version: {e}")
    components = []
    try:
        from localm import updater_components as uc
        for p in uc.plan(root, root, deep=False):
            if not p.get("installed") or p.get("check"):
                continue
            components.append({"name": p["name"], "label": p["label"], "current": p["current"],
                               "target": p["target"], "matches": not p["update"]})
            if p["update"]:
                problems.append(f"the {p['label']} is {p['current']}, but localm {app} "
                                f"pins {p['target']}")
    except Exception as e:
        problems.append(f"could not compare the components with this version: {e}")
    return {"app": app, "install_record": record, "install_record_matches": record_matches,
            "install_record_problem": record_problem,
            "deps_problems": deps_problems, "last_steps": _last_steps(app),
            "runtime": runtime, "components": components, "problems": problems,
            "consistent": not problems}


# Seconds a runtime or component update result stays in :func:`installed_parts`.
LAST_STEPS_S = 24 * 3600.0


def _last_steps(version: Optional[str]) -> list:
    """The results of the runtime and component updates of the last
    :data:`LAST_STEPS_S` seconds: ``[{"part", "state", "note", "at"}]``. A result
    recorded under another localm version than *version* is left out; one that
    names no version (written by an older build) is kept. A note both parts
    recorded (a rollback) is listed once."""
    import time
    out = []
    try:
        from localm import updater_components, updater_runtime
        for part, st in (("runtime", updater_runtime.last_status()),
                         ("components", updater_components.last_status())):
            if not st or not st.get("note"):
                continue
            if st.get("version") and _norm_version(st["version"]) != _norm_version(version):
                continue
            try:
                at = float(st.get("at") or 0)
            except (TypeError, ValueError):
                continue
            if time.time() - at < LAST_STEPS_S and st["note"] not in [s["note"] for s in out]:
                out.append({"part": part, "state": st.get("state"), "note": st["note"],
                            "at": at})
    except Exception:
        return out
    return out


def _reinstall_deps(root: Path, *, runner: Optional[Callable], say: Callable) -> str:
    """Reinstall localm and its dependencies into the venv of *root* with the
    install's own uv from its ``uv.lock``, keeping a copy of the venv first
    and putting it back when the step fails. Returns a one-line note. Raises
    LocalmError."""
    from localm import _apply_update as au
    from localm import updater, updater_components as uc
    from localm.bugreport import LocalmError
    cmd = list(au.post_swap_command("deps", uv=updater._uv_for_deps(root)) or [])
    updater._require_runnable(cmd, "deps")
    updater._require_deps_source(root)
    snap = updater._updates_dir() / REPAIR_SNAPSHOT
    try:
        uc.snapshot([uc.DEPS], snap, root=root)
    except OSError as e:
        raise LocalmError("could not keep a copy of the venv, so nothing was repaired",
                          reason=str(e)) from None
    say("Reinstalling localm and its dependencies...")
    run = runner or (lambda c: subprocess.run(c, cwd=str(root),
                                              env=updater._deps_env(root)).returncode)
    try:
        try:
            rc = int(run(cmd))
            problem = None if rc == 0 else f"{Path(str(cmd[0])).name} exited {rc}"
        except Exception as e:
            problem = f"could not run {cmd[0]}: {e}"
        if problem is None:
            return "reinstalled localm and its dependencies"
        errors = uc.restore(snap, root=root, say=say)
        if errors:
            raise LocalmError("the dependencies could not be reinstalled AND putting the "
                              "previous ones back failed", reason=f"{problem}; "
                              + "; ".join(errors))
        raise LocalmError("the dependencies could not be reinstalled; the previous ones "
                          "were put back",
                          reason=updater.with_retry_hint(problem, root, "update --repair"))
    finally:
        uc.discard(snap)


def repair(*, installed=None, defer: bool = False, runner: Optional[Callable] = None,
           say: Optional[Callable] = None) -> dict:
    """Bring every part of the install at *installed* (default the running one)
    to the installed localm version (:func:`installed_parts`): reinstall the
    dependencies when the install record or a dependency disagrees, then
    record the runtime re-provision and the component update this version
    needs and carry them out now, or with *defer* (a server) leave them to its
    restart. Holds the update lock. Returns ``{"parts", "notes",
    "restart_needed"}``: the parts before the repair, one line per step taken,
    and whether a restart must carry out or load what was changed. Raises
    LocalmError when a step fails (the dependencies are put back as they
    were)."""
    from localm import _version, updater, updater_components as uc, updater_runtime
    from localm.bugreport import LocalmError
    say = say or (lambda _m: None)
    root = Path(installed) if installed else updater.repo_root()
    if updater.is_site_packages_install(root):
        raise LocalmError("this localm was installed with pip, so it cannot repair itself",
                          reason="run 'pip install --force-reinstall localm' instead")
    notes: list = []
    restart = False
    with updater._apply_lock("repair"):
        parts = installed_parts(root)
        version = _version.read_version()
        if parts["install_record_matches"] is False or parts["deps_problems"]:
            notes.append(_reinstall_deps(root, runner=runner, say=say))
            restart = True
        rt = updater_runtime.plan(root, root)
        if rt.get("reprovision"):
            updater_runtime.write_pending(rt, root=root, version=version, repair=True)
            restart = True
            if defer:
                notes.append(f"{rt['note']} while localm restarts")
            else:
                status = updater_runtime.finish_pending(root=root, runner=runner, say=say)
                if status is not None:
                    notes.append(status["note"])
                    if status["state"] not in ("updated", "current"):
                        raise LocalmError("the native runtime could not be repaired",
                                          reason=status["note"])
        plans = uc.plan(root, root, deep=not defer)
        if uc.needs_finish(plans) and (any(p["update"] for p in plans) or defer):
            uc.write_pending(plans, root=root, version=version, repair=True)
            restart = True
            if defer:
                notes.append("checks the components while localm restarts")
            else:
                status = uc.finish_pending(root=root, runner=runner, say=say)
                if status is not None:
                    notes.append(status["note"] or "the components are at this version's pins")
                    if status["state"] not in ("updated", "current"):
                        raise LocalmError("the components could not be repaired",
                                          reason=status["note"])
    return {"parts": parts, "notes": notes, "restart_needed": restart}
