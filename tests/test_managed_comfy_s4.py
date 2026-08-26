# SPDX-License-Identifier: AGPL-3.0-or-later
"""STAGE S4 (version-pin lifecycle + localm patch set + `localm comfy update`).

localm pins a known-good ComfyUI commit and carries a small set of localm-owned
patches on top of it (direct core edits, since this is localm's OWN ComfyUI). The
pin advances only via `localm comfy update`, which re-applies the patch set.

Three oracles, each with a built-in negative case so it fails on known-bad work:

  1. PATCH SET (localm/media/comfy_patches.py): apply_patches rewrites a fragile
     comfy_api/internal/__init__.py make_locked_method_func (the bare
     `getattr(type_obj, func).__func__`) into the plain-function-tolerant form; it is
     GUARDED (already-tolerant / wrong-shape -> skipped), IDEMPOTENT (re-apply is a
     byte no-op), and FAIL-SAFE (missing target skipped without error, an unparseable
     result is never written). The patched module, imported, calls a plain-function
     node WITHOUT the AttributeError the unpatched one raises.

  2. PROVISIONING applies the patch set: after provision_fresh against a fake ComfyUI
     that ships the fragile file, the managed core reads tolerant.

  3. `localm comfy update`: moving the pin re-checks-out the managed checkout to the
     new pin and re-applies the patch set; a mid-update failure ROLLS BACK to the
     prior managed ComfyUI (prior commit, main.py present, patch restored, installed).
"""

from __future__ import annotations

import os
import subprocess
import importlib.util
from pathlib import Path

import pytest

import localm.config as cfg
from localm.media import managed_comfy as mc


# --------------------------------------------------------------------------- #
#  Throwaway LOCALM_HOME.                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    h.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    return h


# --------------------------------------------------------------------------- #
#  Fixture source bodies.                                                     #
# --------------------------------------------------------------------------- #

# The upstream buggy body: the bare `.__func__` access breaks on a node whose
# FUNCTION resolves to a plain function (core VAEDecodeAudio / ACE-Step).
_BUGGY_SRC = '''\
import asyncio


def lock_class(cls):
    return cls


def make_locked_method_func(type_obj, func, class_clone):
    locked_class = lock_class(class_clone)
    method = getattr(type_obj, func).__func__
    if asyncio.iscoroutinefunction(method):
        async def wrapped_async_func(**inputs):
            return await method(locked_class, **inputs)
        return wrapped_async_func
    else:
        def wrapped_func(**inputs):
            return method(locked_class, **inputs)
        return wrapped_func


def _plain_execute(bound_cls, **inputs):
    return ("sync", bound_cls, inputs)


async def _plain_async_execute(bound_cls, **inputs):
    return ("async", bound_cls, inputs)


class PlainFnNode:
    execute = staticmethod(_plain_execute)


class AsyncPlainFnNode:
    execute = staticmethod(_plain_async_execute)


class BoundNode:
    @classmethod
    def execute(cls, **inputs):
        return ("bound", cls, inputs)
'''

# Already tolerant (the string form) -> the patch must no-op.
_TOLERANT_SRC = '''\
import asyncio


def lock_class(cls):
    return cls


def make_locked_method_func(type_obj, func, class_clone):
    locked_class = lock_class(class_clone)
    attr = getattr(type_obj, func)
    method = getattr(attr, "__func__", attr)
    return method
'''

# Right file, but make_locked_method_func is a different shape -> no-op.
_WRONGSHAPE_SRC = '''\
def lock_class(cls):
    return cls


def make_locked_method_func(a, b):
    return None
'''


def _write_internal(root: Path, src: str) -> Path:
    """Write a fake comfy_api/internal/__init__.py under *root*, return its path."""
    pkg = root / "comfy_api" / "internal"
    pkg.mkdir(parents=True, exist_ok=True)
    (root / "comfy_api" / "__init__.py").write_text("", encoding="utf-8")
    f = pkg / "__init__.py"
    f.write_text(src, encoding="utf-8")
    return f


def _load_src_as_module(src: str, name: str, tmp_path: Path):
    """Import *src* as a throwaway module so we can call its functions for real."""
    p = tmp_path / (name + ".py")
    p.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =========================================================================== #
#  1. PATCH SET                                                               #
# =========================================================================== #

def test_apply_patches_makes_fragile_func_tolerant_and_it_works(tmp_path):
    """The core oracle: a fragile make_locked_method_func is rewritten to the
    plain-function-tolerant form, the result parses, and (proven for REAL by
    importing it) the tolerant function runs a plain-function node instead of
    raising - sync, async, and the bound-method case all still work."""
    from localm.media import comfy_patches as cp

    # NEGATIVE: the unpatched fragile function raises on a plain-function node.
    buggy = _load_src_as_module(_BUGGY_SRC, "s4_buggy", tmp_path)
    with pytest.raises(AttributeError):
        buggy.make_locked_method_func(buggy.PlainFnNode, "execute", buggy.PlainFnNode)

    root = tmp_path / "managed"
    target = _write_internal(root, _BUGGY_SRC)

    outcomes = cp.apply_patches(root)
    assert all(o.ok for o in outcomes)
    func = next(o for o in outcomes if o.name == cp.FUNC_PATCH.name)
    assert func.status == cp.APPLIED

    patched = target.read_text(encoding="utf-8")
    # The task-specified tolerant shape.
    assert "attr = getattr(type_obj, func)" in patched
    assert 'method = attr.__func__ if hasattr(attr, "__func__") else attr' in patched
    # It must still parse.
    import ast
    ast.parse(patched)

    # REAL behavior: import the PATCHED text and call it on a plain-function node.
    fixed = _load_src_as_module(patched, "s4_fixed", tmp_path)
    w = fixed.make_locked_method_func(fixed.PlainFnNode, "execute", fixed.PlainFnNode)
    assert w(x=1) == ("sync", fixed.PlainFnNode, {"x": 1})
    aw = fixed.make_locked_method_func(
        fixed.AsyncPlainFnNode, "execute", fixed.AsyncPlainFnNode)
    import asyncio
    assert asyncio.run(aw(y=2)) == ("async", fixed.AsyncPlainFnNode, {"y": 2})
    bw = fixed.make_locked_method_func(fixed.BoundNode, "execute", fixed.BoundNode)
    assert bw(z=3) == ("bound", fixed.BoundNode, {"z": 3})


def test_apply_patches_is_idempotent(tmp_path):
    """Re-applying is a byte no-op (the second run finds nothing fragile)."""
    from localm.media import comfy_patches as cp
    root = tmp_path / "managed"
    target = _write_internal(root, _BUGGY_SRC)

    first = cp.apply_patches(root)
    assert next(o for o in first if o.name == cp.FUNC_PATCH.name).status == cp.APPLIED
    after_first = target.read_bytes()

    second = cp.apply_patches(root)
    assert next(o for o in second if o.name == cp.FUNC_PATCH.name).status == cp.SKIPPED
    assert target.read_bytes() == after_first  # unchanged


def test_apply_patches_skips_already_tolerant(tmp_path):
    """An already-tolerant file (upstream fixed / previously patched) is skipped."""
    from localm.media import comfy_patches as cp
    root = tmp_path / "managed"
    target = _write_internal(root, _TOLERANT_SRC)
    before = target.read_bytes()

    outcomes = cp.apply_patches(root)
    assert next(o for o in outcomes if o.name == cp.FUNC_PATCH.name).status == cp.SKIPPED
    assert target.read_bytes() == before


def test_apply_patches_skips_unexpected_shape(tmp_path):
    """A make_locked_method_func of a different shape is left alone (never risk
    corrupting an unknown version)."""
    from localm.media import comfy_patches as cp
    root = tmp_path / "managed"
    target = _write_internal(root, _WRONGSHAPE_SRC)
    before = target.read_bytes()

    outcomes = cp.apply_patches(root)
    assert next(o for o in outcomes if o.name == cp.FUNC_PATCH.name).status == cp.SKIPPED
    assert target.read_bytes() == before


def test_apply_patches_missing_target_is_absent_not_error(tmp_path):
    """A missing/renamed target file is skipped WITHOUT error (fail-safe), and
    nothing is created."""
    from localm.media import comfy_patches as cp
    root = tmp_path / "managed"
    root.mkdir()

    outcomes = cp.apply_patches(root)  # no comfy_api/internal/__init__.py present
    func = next(o for o in outcomes if o.name == cp.FUNC_PATCH.name)
    assert func.ok
    assert func.status == cp.ABSENT
    assert not (root / "comfy_api").exists()


def test_apply_patch_never_writes_unparseable_result(tmp_path):
    """FAIL-SAFE: a patch whose transform yields text that does not parse is
    reported failed and the file is left byte-identical (no corruption)."""
    from localm.media import comfy_patches as cp
    root = tmp_path / "managed"
    target = _write_internal(root, _BUGGY_SRC)
    before = target.read_bytes()

    broken = cp.ComfyPatch(
        name="broken-test-patch",
        description="a transform that returns invalid python",
        target="comfy_api/internal/__init__.py",
        transform=lambda text: ("def (:this is not python", "intentionally broken"))
    out = cp.apply_patch(broken, root)
    assert out.status == cp.FAILED
    assert not out.ok
    assert target.read_bytes() == before  # untouched


def test_func_patch_is_registered():
    """The __func__ tolerance patch is part of the shipped patch set."""
    from localm.media import comfy_patches as cp
    assert cp.FUNC_PATCH in cp.PATCHES
    assert cp.FUNC_PATCH.target == "comfy_api/internal/__init__.py"


# =========================================================================== #
#  helpers for the provisioning + update oracles (real git)                   #
# =========================================================================== #

def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, timeout=10)
        return True
    except Exception:
        return False


_GIT_ENV = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
_GIT_IDENT = ["-c", "user.email=s4@localhost", "-c", "user.name=s4-test",
              "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main"]


def _git(cwd: Path, *args) -> None:
    subprocess.run(["git", *_GIT_IDENT, *args], cwd=str(cwd), env=_GIT_ENV,
                   check=True, capture_output=True, text=True)


def _head(root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), env=_GIT_ENV,
                          capture_output=True, text=True, check=True).stdout.strip()


def _commit_all(workdir: Path, msg: str) -> str:
    _git(workdir, "add", "-A")
    _git(workdir, "commit", "-m", msg)
    return _head(workdir)


def _make_fake_venv(root: Path) -> None:
    """A placeholder venv interpreter + completion marker so
    is_managed_comfy_installed() reads True (S4's update never rebuilds the venv;
    it only moves the git source + patches). The marker is load-bearing: main.py
    plus a venv alone reads as "still installing", not "installed"."""
    from localm.media.managed_comfy_provision import MARKER_FILENAME
    paths = mc.managed_comfy_paths()
    paths.venv_python.parent.mkdir(parents=True, exist_ok=True)
    paths.venv_python.write_text("", encoding="utf-8")
    (paths.root / MARKER_FILENAME).write_text("{}", encoding="utf-8")


# =========================================================================== #
#  2. PROVISIONING applies the patch set (fresh path, real git + venv).       #
# =========================================================================== #

@pytest.mark.skipif(not _git_available(), reason="git not on PATH")
def test_provision_fresh_applies_localm_patches(home, tmp_path, monkeypatch):
    """A fresh provision against a fake ComfyUI that ships the fragile file leaves
    the managed core PATCHED (tolerant). Negative: the fake's own source is fragile
    until provisioning applies the patch."""
    from localm import hwdetect
    from localm.media import managed_comfy_fresh as fresh

    # provision_fresh computes comfy_torch_spec() BEFORE install_torch=False is
    # consulted (it needs the spec either way), so on a real NVIDIA box this
    # would otherwise shell out to the real nvidia-smi.
    monkeypatch.setattr(hwdetect, "_cuda_compute_capabilities", lambda: [])

    # A minimal fake ComfyUI git repo: main.py + the fragile core file, no reqs.
    src = tmp_path / "ComfyUI"
    src.mkdir()
    (src / "main.py").write_text("print('comfy')\n", encoding="utf-8")
    _write_internal(src, _BUGGY_SRC)
    (src / ".gitignore").write_text("venv/\nmodels/\ncustom_nodes/\n", encoding="utf-8")
    _git(src, "init")
    commit = _commit_all(src, "init")

    result = fresh.provision_fresh(
        comfyui_repo=str(src), comfyui_commit=commit,
        custom_nodes=[], install_torch=False)
    assert result.ok, result.message

    root = mc.managed_comfy_paths().root
    patched = (root / "comfy_api" / "internal" / "__init__.py").read_text(encoding="utf-8")
    assert 'hasattr(attr, "__func__")' in patched
    assert "getattr(type_obj, func).__func__" not in patched  # fragile access gone
    assert mc.is_managed_comfy_installed()


# =========================================================================== #
#  3. `localm comfy update`: move the pin, re-apply patches, roll back safely. #
# =========================================================================== #

def _build_update_remote(tmp_path: Path):
    """A fake ComfyUI 'remote' with a linear history, returning (path, shas):
      c1: main.py + fragile core       (the 'installed' pin)
      c2: c1 + a VERSION bump, still fragile core  (a clean advance target)
      c3: c2 with main.py DELETED       (a broken target -> update must roll back)
    """
    remote = tmp_path / "remote_comfy"
    remote.mkdir()
    (remote / "main.py").write_text("print('comfy')\n", encoding="utf-8")
    _write_internal(remote, _BUGGY_SRC)
    (remote / ".gitignore").write_text("venv/\nmodels/\n", encoding="utf-8")
    _git(remote, "init")
    c1 = _commit_all(remote, "c1")
    (remote / "VERSION").write_text("0.9.3\n", encoding="utf-8")
    c2 = _commit_all(remote, "c2 advance")
    (remote / "main.py").unlink()
    c3 = _commit_all(remote, "c3 broken (main.py gone)")
    return remote, c1, c2, c3


def _install_managed_from(remote: Path, commit: str) -> Path:
    """Clone *remote* into the managed dir, pin *commit*, add a fake venv, and apply
    the real localm patch set - i.e. a managed ComfyUI installed + patched at *commit*."""
    from localm.media.comfy_patches import apply_patches
    root = mc.managed_comfy_paths().root
    _git(Path("."), "clone", "--quiet", str(remote), str(root))
    _git(root, "checkout", "--quiet", commit)
    _make_fake_venv(root)
    outcomes = apply_patches(root)
    assert all(o.ok for o in outcomes)
    return root


@pytest.mark.skipif(not _git_available(), reason="git not on PATH")
def test_update_advances_pin_and_reapplies_patches(home, tmp_path):
    """Moving the pin re-checks-out the managed checkout to the new commit and
    re-applies the localm patch set (the freshly-checked-out fragile core is made
    tolerant again). Negative: at the new commit the tracked file is fragile until
    the patch re-applies."""
    from localm.media import managed_comfy_update as upd

    remote, c1, c2, _c3 = _build_update_remote(tmp_path)
    root = _install_managed_from(remote, c1)
    assert _head(root) == c1

    result = upd.update_managed_comfy(
        comfyui_repo=str(remote), target_commit=c2, reinstall_requirements=False)
    assert result.ok, result.message
    assert _head(root) == c2                       # advanced to the new pin
    assert (root / "VERSION").is_file()            # c2 content present
    patched = (root / "comfy_api" / "internal" / "__init__.py").read_text(encoding="utf-8")
    assert 'hasattr(attr, "__func__")' in patched  # patch re-applied at the new pin
    assert "getattr(type_obj, func).__func__" not in patched
    assert mc.is_managed_comfy_installed()


@pytest.mark.skipif(not _git_available(), reason="git not on PATH")
def test_update_rolls_back_on_failure(home, tmp_path):
    """A mid-update failure (the target commit leaves the tree not-installed) ROLLS
    BACK to the prior managed ComfyUI: prior commit, main.py restored, patch
    re-applied, reads as installed again."""
    from localm.media import managed_comfy_update as upd

    remote, c1, _c2, c3 = _build_update_remote(tmp_path)
    root = _install_managed_from(remote, c1)
    assert _head(root) == c1

    result = upd.update_managed_comfy(
        comfyui_repo=str(remote), target_commit=c3, reinstall_requirements=False)
    assert not result.ok
    # Rolled back to the prior state - nothing half-updated left behind.
    assert _head(root) == c1
    assert (root / "main.py").is_file()
    patched = (root / "comfy_api" / "internal" / "__init__.py").read_text(encoding="utf-8")
    assert 'hasattr(attr, "__func__")' in patched
    assert mc.is_managed_comfy_installed()


@pytest.mark.skipif(not _git_available(), reason="git not on PATH")
def test_update_rollback_surfaces_failed_patch_reapply(home, tmp_path, monkeypatch):
    """When a mid-update failure triggers rollback and the git source IS restored
    but the localm patch set cannot be re-applied on it, update must NOT report a
    clean rollback: the failed re-apply is surfaced in the returned message.
    Discarding the rollback's apply_patches outcomes would tell the user the prior
    install was restored 'exactly as it was' while it was left UNPATCHED (the
    __func__ fix gone, ACE-Step music broken)."""
    from localm.media import managed_comfy_update as upd
    from localm.media import comfy_patches as cp

    remote, c1, _c2, c3 = _build_update_remote(tmp_path)
    root = _install_managed_from(remote, c1)
    assert _head(root) == c1

    # The forward apply (call #1, at c3) is real and succeeds; the ROLLBACK re-apply
    # (call #2, on the restored c1 source) simulates a patch that cannot be applied.
    real_apply = cp.apply_patches
    calls = {"n": 0}

    def _fake_apply(managed_dir, patches=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_apply(managed_dir, patches)
        return [cp.PatchOutcome(cp.FUNC_PATCH.name, cp.FAILED,
                                "simulated re-apply failure")]

    monkeypatch.setattr(upd, "apply_patches", _fake_apply)

    # c3 deletes main.py -> the forward update reaches the not-installed check and rolls
    # back to c1 (the rollback trigger is NOT a patch failure, isolating this leg).
    result = upd.update_managed_comfy(
        comfyui_repo=str(remote), target_commit=c3, reinstall_requirements=False)

    assert not result.ok
    assert calls["n"] == 2                    # forward apply + rollback re-apply both ran
    assert _head(root) == c1                   # the git source WAS restored ...
    assert (root / "main.py").is_file()
    # ... but the FAILED patch re-apply is SURFACED, not hidden behind a clean rollback.
    assert "could not be re-applied on the restored install" in result.message
    assert "simulated re-apply failure" in result.message


@pytest.mark.skipif(not _git_available(), reason="git not on PATH")
def test_update_requires_git_checkout(home):
    """A managed ComfyUI with no git history (the non-git copytree fallback) cannot
    be pin-updated; update says so honestly instead of pretending."""
    from localm.media import managed_comfy_update as upd
    root = mc.managed_comfy_paths().root
    root.mkdir(parents=True)
    (root / "main.py").write_text("print('comfy')\n", encoding="utf-8")
    _make_fake_venv(root)  # installed, but NOT a git checkout

    result = upd.update_managed_comfy(target_commit="0" * 40,
                                      reinstall_requirements=False)
    assert not result.ok
    assert "git" in result.message.lower()


def test_update_requires_installed(home):
    """update with no managed ComfyUI installed is an honest failure, not a no-op
    that claims success."""
    from localm.media import managed_comfy_update as upd
    result = upd.update_managed_comfy(reinstall_requirements=False)
    assert not result.ok
    assert "setup" in result.message.lower() or "no managed" in result.message.lower()


# =========================================================================== #
#  CLI wiring for `localm comfy update`.                                       #
# =========================================================================== #

def test_cli_update_errors_when_not_installed(cli_runner):
    """`localm comfy update` with nothing installed exits non-zero and points at
    setup."""
    from localm.cli import main
    res = cli_runner.invoke(main, ["comfy", "update"])
    assert res.exit_code != 0
    assert "setup" in res.output.lower()


def test_cli_update_dispatches_to_updater(cli_runner, monkeypatch):
    """When a managed ComfyUI is installed, the command calls update_managed_comfy
    and reports its result."""
    from localm.cli import main
    from localm.media import managed_comfy as mcmod
    from localm.media import managed_comfy_update as upd
    from localm.media.managed_comfy_provision import ProvisionResult

    monkeypatch.setattr(mcmod, "is_managed_comfy_installed", lambda: True)
    seen = {}

    def _fake_update(cfg=None, **kw):
        seen["ran"] = True
        return ProvisionResult(ok=True, status="updated",
                               message="Updated localm's managed ComfyUI.")

    monkeypatch.setattr(upd, "update_managed_comfy", _fake_update)
    res = cli_runner.invoke(main, ["comfy", "update"])
    assert res.exit_code == 0, res.output
    assert seen.get("ran") is True
    assert "updated" in res.output.lower()


# --------------------------------------------------------------------------- #
#  SINGLE-FLIGHT: only one update may mutate the checkout at a time            #
# --------------------------------------------------------------------------- #
# update_managed_comfy is a long MUTATION of one working tree (fetch, checkout,
# patch, and on failure a rollback checkout + patch). The GUI Update route runs
# `python -m localm comfy update` as a CHILD PROCESS, so the contenders are
# separate interpreters and an in-process threading.Lock would guard nothing.

def _lock_dir():
    from localm.media import managed_comfy_update as upd
    return upd._update_lock_path()


def _installed_git_checkout(home):
    """An installed managed ComfyUI that reads as a git checkout, so update gets past
    its read-only refusals and reaches the locked section."""
    import json as _json
    from localm.media.managed_comfy_provision import MARKER_FILENAME
    paths = mc.managed_comfy_paths()
    paths.main_py.parent.mkdir(parents=True, exist_ok=True)
    paths.main_py.write_text("# stand-in\n", encoding="utf-8")
    paths.venv_python.parent.mkdir(parents=True, exist_ok=True)
    paths.venv_python.write_text("", encoding="utf-8")
    (paths.root / MARKER_FILENAME).write_text(_json.dumps({"commit": "old"}),
                                              encoding="utf-8")
    return paths


def test_update_refuses_while_a_LIVE_holder_owns_the_lock(home, monkeypatch):
    """The core guarantee. A second update must be REFUSED, not interleaved - two
    fetch/checkout/patch sequences on one tree corrupt it."""
    import json as _json
    from localm.media import managed_comfy_update as upd
    _installed_git_checkout(home)
    monkeypatch.setattr(upd, "_rev_parse_head", lambda root: "abc123")

    lock = _lock_dir()
    lock.mkdir(parents=True)
    (lock / upd._LOCK_OWNER).write_text(_json.dumps({"pid": os.getpid()}),
                                        encoding="utf-8")   # THIS process: provably alive

    ran = []
    monkeypatch.setattr(upd, "_run", lambda *a, **k: (ran.append(a), (True, ""))[1])
    res = upd.update_managed_comfy()

    assert res.ok is False
    assert res.status == "busy", res.status
    assert "already running" in res.message
    assert str(os.getpid()) in res.message, "name the holder so the user can check"
    # It refused BEFORE touching the tree.
    assert ran == [], "a refused update must not run a single git command"


def test_update_reclaims_a_lock_whose_owner_is_DEAD(home, monkeypatch):
    """A crashed update must not wedge every later one. Staleness is judged by PID
    LIVENESS, never elapsed time - the operation is unbounded, so any fixed timeout
    would eventually reclaim a LIVE holder's lock."""
    import json as _json
    from localm.media import managed_comfy_update as upd
    _installed_git_checkout(home)
    monkeypatch.setattr(upd, "_rev_parse_head", lambda root: "abc123")
    monkeypatch.setattr("localm.instances.pid_alive", lambda pid: False)

    lock = _lock_dir()
    lock.mkdir(parents=True)
    (lock / upd._LOCK_OWNER).write_text(_json.dumps({"pid": 999999}), encoding="utf-8")

    monkeypatch.setattr(upd, "_run", lambda *a, **k: (True, ""))
    monkeypatch.setattr(upd, "apply_patches", lambda root: [])
    monkeypatch.setattr(mc, "is_managed_comfy_installed", lambda: True)
    res = upd.update_managed_comfy(target_commit="abc123")

    assert res.status != "busy", f"a dead holder must be reclaimed, got {res.message}"


def test_update_will_not_steal_a_lock_it_cannot_identify(home, monkeypatch):
    """An unreadable owner means we cannot tell whether a live update holds it.
    Conservative: refuse and say how to clear it by hand. Stealing here is exactly how
    two updates end up interleaved."""
    from localm.media import managed_comfy_update as upd
    _installed_git_checkout(home)
    monkeypatch.setattr(upd, "_rev_parse_head", lambda root: "abc123")
    lock = _lock_dir()
    lock.mkdir(parents=True)          # no owner file at all

    ran = []
    monkeypatch.setattr(upd, "_run", lambda *a, **k: (ran.append(a), (True, ""))[1])
    res = upd.update_managed_comfy()
    assert res.ok is False and res.status == "busy"
    assert "could not be read" in res.message
    assert str(lock) in res.message, "tell the user WHICH folder to remove"
    assert ran == []


def test_update_releases_the_lock_on_success_and_on_failure(home, monkeypatch):
    """A leaked lock would wedge every later update until someone deleted a folder by
    hand, so it must be released on EVERY path - including the failure/rollback one."""
    from localm.media import managed_comfy_update as upd
    _installed_git_checkout(home)
    monkeypatch.setattr(upd, "_rev_parse_head", lambda root: "abc123")
    monkeypatch.setattr(mc, "is_managed_comfy_installed", lambda: True)
    monkeypatch.setattr(upd, "apply_patches", lambda root: [])

    monkeypatch.setattr(upd, "_run", lambda *a, **k: (True, ""))
    upd.update_managed_comfy(target_commit="abc123")
    assert not _lock_dir().exists(), "lock leaked after a SUCCESSFUL update"

    # Now a failing update: the fetch fails, so it rolls back and returns an error.
    monkeypatch.setattr(upd, "_run", lambda *a, **k: (False, "boom"))
    res = upd.update_managed_comfy(target_commit="abc123")
    assert res.ok is False
    assert not _lock_dir().exists(), "lock leaked after a FAILED update"


def test_the_lock_lives_outside_the_checkout_git_force_cannot_disturb_it(home):
    """The update's own `git checkout --force` runs inside the checkout. A lock stored
    in there would be protecting itself from the very operation it guards."""
    _installed_git_checkout(home)
    root = mc.managed_comfy_paths().root
    lock = _lock_dir()
    assert root not in lock.parents, f"lock {lock} must not live inside {root}"
    assert lock.parent == root.parent
