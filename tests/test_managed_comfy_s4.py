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
from localm.media import source_archive
from tests._max_path import max_path  # noqa: F401
from tests._source_archives import FakeGitHub


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
#  helpers for the provisioning + update oracles (source archives, no git)    #
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


def _make_fake_venv(root: Path) -> None:
    """A placeholder venv interpreter + completion marker so
    is_managed_comfy_installed() reads True (S4's update never rebuilds the venv;
    it only moves the source + patches). The marker is load-bearing: main.py
    plus a venv alone reads as "still installing", not "installed"."""
    from localm.media.managed_comfy_provision import MARKER_FILENAME
    paths = mc.managed_comfy_paths()
    paths.venv_python.parent.mkdir(parents=True, exist_ok=True)
    paths.venv_python.write_text("", encoding="utf-8")
    (paths.root / MARKER_FILENAME).write_text("{}", encoding="utf-8")


_CORE = {"comfy_api/__init__.py": "", "comfy_api/internal/__init__.py": _BUGGY_SRC}


def _tree(**changes) -> dict:
    """A fake ComfyUI tree: main.py, gone_at_c2.py, a .gitignore and the fragile
    core, with *changes* applied (a None value drops that file)."""
    files = {"main.py": "print('comfy')\n", "gone_at_c2.py": "x = 1\n",
             ".gitignore": "venv/\nmodels/\ncustom_nodes/\noutput/\n", **_CORE}
    files.update(changes)
    return {k: v for k, v in files.items() if v is not None}


# =========================================================================== #
#  2. PROVISIONING applies the patch set (fresh path, source archive + venv). #
# =========================================================================== #

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

    hub = FakeGitHub(tmp_path / "hub").install(monkeypatch)
    repo, commit = hub.add("fake", "ComfyUI", _tree())

    result = fresh.provision_fresh(
        comfyui_repo=repo, comfyui_commit=commit,
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

def _build_update_remote(tmp_path: Path, monkeypatch):
    """A fake ComfyUI on a fake GitHub with three commits, returning
    (hub, repo, c1, c2, c3):
      c1: main.py + fragile core + gone_at_c2.py  (the 'installed' pin)
      c2: c1 + a VERSION file, gone_at_c2.py removed, still fragile core
      c3: c2 with main.py DELETED       (a broken target -> update must roll back)
    """
    hub = FakeGitHub(tmp_path / "hub").install(monkeypatch)
    repo, c1 = hub.add("fake", "ComfyUI", _tree())
    _, c2 = hub.add("fake", "ComfyUI", _tree(VERSION="0.9.3\n", **{"gone_at_c2.py": None}))
    _, c3 = hub.add("fake", "ComfyUI", _tree(VERSION="0.9.3\n",
                                             **{"gone_at_c2.py": None, "main.py": None}))
    return hub, repo, c1, c2, c3


def _install_managed_from(repo: str, commit: str) -> Path:
    """Fetch *commit* into the managed dir, add a fake venv, and apply the real
    localm patch set - i.e. a managed ComfyUI installed + patched at *commit*."""
    from localm.media.comfy_patches import apply_patches
    root = mc.managed_comfy_paths().root
    source_archive.fetch_tree(repo, commit, root)
    _make_fake_venv(root)
    outcomes = apply_patches(root)
    assert all(o.ok for o in outcomes)
    return root


def test_update_advances_pin_and_reapplies_patches(home, tmp_path, monkeypatch):
    """Moving the pin replaces the managed source with the new commit's files,
    removes a file the new commit dropped, keeps files that are not source (the
    venv, outputs) and re-applies the localm patch set. No git program runs."""
    from localm.media import managed_comfy_update as upd

    _hub, repo, c1, c2, _c3 = _build_update_remote(tmp_path, monkeypatch)
    root = _install_managed_from(repo, c1)
    (root / "output").mkdir()
    (root / "output" / "mine.png").write_bytes(b"user data")
    assert source_archive.tree_commit(root) == c1

    result = upd.update_managed_comfy(
        comfyui_repo=repo, target_commit=c2, reinstall_requirements=False)
    assert result.ok, result.message
    assert source_archive.tree_commit(root) == c2  # advanced to the new pin
    assert (root / "VERSION").is_file()            # c2 content present
    assert not (root / "gone_at_c2.py").exists()   # dropped upstream, removed here
    assert (root / "output" / "mine.png").read_bytes() == b"user data"
    assert mc.managed_comfy_paths().venv_python.is_file()
    patched = (root / "comfy_api" / "internal" / "__init__.py").read_text(encoding="utf-8")
    assert 'hasattr(attr, "__func__")' in patched  # patch re-applied at the new pin
    assert "getattr(type_obj, func).__func__" not in patched
    assert mc.is_managed_comfy_installed()
    leftovers = [p.name for p in root.parent.iterdir()
                 if "-backup-" in p.name or "-download-" in p.name]
    assert leftovers == []


def test_update_rolls_back_on_failure(home, tmp_path, monkeypatch):
    """A mid-update failure (the target commit leaves the tree not-installed) ROLLS
    BACK to the prior managed ComfyUI: prior commit, main.py and the dropped file
    restored, patch re-applied, reads as installed again."""
    from localm.media import managed_comfy_update as upd

    _hub, repo, c1, _c2, c3 = _build_update_remote(tmp_path, monkeypatch)
    root = _install_managed_from(repo, c1)
    before = (root / "comfy_api" / "internal" / "__init__.py").read_text(encoding="utf-8")

    result = upd.update_managed_comfy(
        comfyui_repo=repo, target_commit=c3, reinstall_requirements=False)
    assert not result.ok
    # Rolled back to the prior state - nothing half-updated left behind.
    assert source_archive.tree_commit(root) == c1
    assert (root / "main.py").is_file()
    assert (root / "gone_at_c2.py").is_file()
    assert not (root / "VERSION").exists()
    after = (root / "comfy_api" / "internal" / "__init__.py").read_text(encoding="utf-8")
    assert after == before and 'hasattr(attr, "__func__")' in after
    assert mc.is_managed_comfy_installed()


def test_update_refuses_an_archive_of_another_commit(home, tmp_path, monkeypatch):
    """An archive whose zip comment names a different commit is never applied."""
    from localm.media import managed_comfy_update as upd

    hub, repo, c1, _c2, _c3 = _build_update_remote(tmp_path, monkeypatch)
    root = _install_managed_from(repo, c1)
    _, bad = hub.add("fake", "ComfyUI", _tree(VERSION="evil\n"), comment="0" * 40)

    result = upd.update_managed_comfy(comfyui_repo=repo, target_commit=bad,
                                      reinstall_requirements=False)
    assert not result.ok
    assert f"not {bad}" in result.message
    assert source_archive.tree_commit(root) == c1
    assert not (root / "VERSION").exists()


def test_a_patch_that_cannot_be_applied_changes_nothing(home, tmp_path, monkeypatch):
    from localm.media import comfy_patches as cp
    from localm.media import managed_comfy_update as upd

    _hub, repo, c1, c2, _c3 = _build_update_remote(tmp_path, monkeypatch)
    root = _install_managed_from(repo, c1)
    before = _tree_of(root)
    monkeypatch.setattr(upd, "apply_patches", lambda d: [
        cp.PatchOutcome(cp.FUNC_PATCH.name, cp.FAILED, "simulated failure")])

    result = upd.update_managed_comfy(
        comfyui_repo=repo, target_commit=c2, reinstall_requirements=False)

    assert not result.ok and "simulated failure" in result.message
    assert _tree_of(root) == before


def _tree_of(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and ".localm-update" not in p.parts}


def _refuse_moves(monkeypatch, refused):
    """Make every rename for which ``refused(src, dst)`` is true fail; returns the call
    that allows them again."""
    real = os.rename

    def rename(src, dst):
        if refused(str(src), str(dst)):
            raise OSError(28, "No space left on device", str(dst))
        return real(src, dst)
    monkeypatch.setattr(os, "rename", rename)
    return lambda: monkeypatch.setattr(os, "rename", real)


def test_a_file_that_cannot_be_moved_in_puts_every_file_back(home, tmp_path, monkeypatch):
    from localm.media import managed_comfy_update as upd

    _hub, repo, c1, c2, _c3 = _build_update_remote(tmp_path, monkeypatch)
    root = _install_managed_from(repo, c1)
    before = _tree_of(root)
    _refuse_moves(monkeypatch, lambda src, dst: dst.endswith("VERSION")
                  and f"{os.sep}new{os.sep}" in src)

    result = upd.update_managed_comfy(
        comfyui_repo=repo, target_commit=c2, reinstall_requirements=False)

    assert not result.ok and "No space left" in result.message
    assert _tree_of(root) == before
    assert mc.is_managed_comfy_installed()


def test_files_that_cannot_be_put_back_are_put_back_by_the_next_update(home, tmp_path,
                                                                      monkeypatch):
    from localm.media import managed_comfy_update as upd

    _hub, repo, c1, c2, _c3 = _build_update_remote(tmp_path, monkeypatch)
    root = _install_managed_from(repo, c1)
    before = _tree_of(root)
    inside = os.path.realpath(root)
    allow = _refuse_moves(monkeypatch, lambda src, dst: "VERSION" in dst or (
        ".localm-update" in src and dst.endswith("gone_at_c2.py") and inside in dst))

    result = upd.update_managed_comfy(
        comfyui_repo=repo, target_commit=c2, reinstall_requirements=False)
    assert not result.ok and "mixed state" in result.message
    assert not (root / "gone_at_c2.py").exists()

    allow()
    assert mc.why_managed_comfy_is_unsettled() is None
    assert _tree_of(root) == before
    again = upd.update_managed_comfy(
        comfyui_repo=repo, target_commit=c2, reinstall_requirements=False)
    assert again.ok, again.message
    assert source_archive.tree_commit(root) == c2


def test_an_update_whose_process_ended_is_undone_by_the_next_use(home, tmp_path, monkeypatch):
    from localm.media import managed_comfy_update as upd
    from localm.update_mechanism import job, mover

    _hub, repo, c1, c2, _c3 = _build_update_remote(tmp_path, monkeypatch)
    root = _install_managed_from(repo, c1)
    before = _tree_of(root)

    class Ended(BaseException):
        pass

    def end(*_a, **_k):
        raise Ended
    with monkeypatch.context() as ending:
        ending.setattr(job, "restore", end)
        ending.setattr(mover, "back", end)
        ending.setattr(mover, "_back", end)
        _refuse_moves(ending, lambda src, dst: "VERSION" in dst and end())
        with pytest.raises(Ended):
            upd.update_managed_comfy(
                comfyui_repo=repo, target_commit=c2, reinstall_requirements=False)
    assert _tree_of(root) != before
    assert mc.is_managed_comfy_installed()
    assert _tree_of(root) == before


def test_a_custom_node_is_prepared_outside_custom_nodes(home, tmp_path, monkeypatch):
    from localm.media import managed_comfy_fresh as fresh
    from localm.media import managed_comfy_update as upd

    hub, repo, c1, c2, _c3 = _build_update_remote(tmp_path, monkeypatch)
    root = _install_managed_from(repo, c1)
    node_repo, n1 = hub.add("o", "Node", {"__init__.py": "old\n", "dropped.py": "x\n"})
    _, n2 = hub.add("o", "Node", {"__init__.py": "new\n"})
    node = root / "custom_nodes" / "Node"
    node.parent.mkdir(exist_ok=True)
    source_archive.fetch_tree(node_repo, n1, node)
    (node / "mine.json").write_text("settings", encoding="utf-8")
    pin = fresh.CustomNodePin(name="Node", repo=node_repo, commit=n2,
                              **{k: getattr(fresh._GGUF_NODE, k)
                                 for k in fresh._GGUF_NODE.__dataclass_fields__
                                 if k not in ("name", "repo", "commit")})
    monkeypatch.setattr(fresh, "required_custom_nodes",
                        lambda *a: type("Required", (), {"nodes": [pin]})())
    from localm.media import managed_comfy_packages
    monkeypatch.setattr(managed_comfy_packages, "prepare", lambda *a, **k: False)

    result = upd.update_managed_comfy(
        comfyui_repo=repo, target_commit=c2, reinstall_requirements=True)

    assert result.ok, result.message
    assert sorted(p.name for p in node.iterdir()) == [
        source_archive.SOURCE_RECORD, "__init__.py", "mine.json"]
    assert (node / "__init__.py").read_text(encoding="utf-8") == "new\n"
    assert [p.name for p in node.parent.iterdir()] == ["Node"]


def test_a_custom_node_behind_a_link_is_updated_whole(home, tmp_path, monkeypatch):
    from localm.media import managed_comfy_fresh as fresh
    from localm.media import managed_comfy_packages
    from localm.media import managed_comfy_update as upd

    hub, repo, c1, c2, _c3 = _build_update_remote(tmp_path, monkeypatch)
    root = _install_managed_from(repo, c1)
    node_repo, n1 = hub.add("o", "Node", {"__init__.py": "old\n", "same.py": "same\n",
                                          "lost.py": "lost\n", "v": "1"})
    _, n2 = hub.add("o", "Node", {"__init__.py": "", "sub/__init__.py": "", "same.py": "same\n",
                                  "lost.py": "lost\n", "v": "2"})
    theirs = tmp_path / "theirs"
    source_archive.fetch_tree(node_repo, n1, theirs / "Node")
    (theirs / "Node" / "same.py").write_bytes(b"mine\n")
    (theirs / "Node" / "lost.py").unlink()
    _link(root / "custom_nodes", theirs)
    pin = fresh.CustomNodePin(name="Node", repo=node_repo, commit=n2,
                              **{k: getattr(fresh._GGUF_NODE, k)
                                 for k in fresh._GGUF_NODE.__dataclass_fields__
                                 if k not in ("name", "repo", "commit")})
    monkeypatch.setattr(fresh, "required_custom_nodes",
                        lambda *a: type("Required", (), {"nodes": [pin]})())
    monkeypatch.setattr(managed_comfy_packages, "prepare", lambda *a, **k: False)

    result = upd.update_managed_comfy(
        comfyui_repo=repo, target_commit=c2, reinstall_requirements=True)

    assert result.ok, result.message
    files = _files(theirs / "Node")
    files.pop(source_archive.SOURCE_RECORD)
    assert files == {"__init__.py": "", "sub/__init__.py": "", "same.py": "mine\n",
                     "lost.py": "lost\n", "v": "2"}


def test_update_and_remove_work_on_a_tree_past_the_path_limit(home, tmp_path, monkeypatch,
                                                              max_path):  # noqa: F811
    from localm.media import managed_comfy_update as upd

    deep = "/".join(["a_rather_long_folder_name_for_testing"] * 7) + "/leaf.py"
    hub = FakeGitHub(tmp_path / "hub").install(monkeypatch)
    repo, c1 = hub.add("fake", "ComfyUI", _tree(**{deep: "one\n"}))
    _, c2 = hub.add("fake", "ComfyUI", _tree(VERSION="0.9.3\n", **{deep: "two\n"}))
    root = _install_managed_from(repo, c1)
    long_file = source_archive.long_path(root / deep)
    assert len(str(root / deep)) > 260

    result = upd.update_managed_comfy(comfyui_repo=repo, target_commit=c2,
                                      reinstall_requirements=False)
    assert result.ok, result.message
    assert source_archive.tree_commit(root) == c2
    assert long_file.read_text(encoding="utf-8") == "two\n"

    removed, failed = mc.remove_managed_comfy()
    assert failed == []
    assert root in removed and not os.path.lexists(source_archive.long_path(root))


@pytest.mark.skipif(not _git_available(), reason="git builds the older-localm checkout")
def test_a_checkout_an_older_localm_cloned_updates_without_git(home, tmp_path, monkeypatch):
    """A managed ComfyUI made by ``git clone`` (older localm) is read through its
    .git folder (HEAD and index) and updated from an archive with no git program:
    a tracked file the new commit drops is removed, an untracked one is kept."""
    from localm.media import managed_comfy_update as upd

    _hub, repo, _c1, c2, _c3 = _build_update_remote(tmp_path, monkeypatch)
    root = mc.managed_comfy_paths().root
    root.mkdir(parents=True)
    for rel, text in _tree().items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    _git(root, "init")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "old")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), env=_GIT_ENV,
                          capture_output=True, text=True, check=True).stdout.strip()
    _make_fake_venv(root)
    (root / "untracked.txt").write_text("keep\n", encoding="utf-8")
    assert source_archive.tree_commit(root) == head
    assert "gone_at_c2.py" in source_archive.tree_files(root)

    def _no_git(*a, **k):
        raise AssertionError("a git program was started")

    monkeypatch.setattr(upd, "_run", _no_git)
    result = upd.update_managed_comfy(comfyui_repo=repo, target_commit=c2,
                                      reinstall_requirements=False)
    assert result.ok, result.message
    assert source_archive.tree_commit(root) == c2
    assert not (root / "gone_at_c2.py").exists()
    assert (root / "untracked.txt").is_file()


def test_update_requires_a_recorded_commit(home):
    """A managed ComfyUI with no recorded source commit (the non-git copytree
    fallback) cannot be pin-updated; update says so honestly instead of pretending."""
    from localm.media import managed_comfy_update as upd
    root = mc.managed_comfy_paths().root
    root.mkdir(parents=True)
    (root / "main.py").write_text("print('comfy')\n", encoding="utf-8")
    _make_fake_venv(root)  # installed, but no source record and not a git checkout

    result = upd.update_managed_comfy(target_commit="0" * 40,
                                      reinstall_requirements=False)
    assert not result.ok
    assert "no recorded source commit" in result.message.lower()


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
    monkeypatch.setattr(source_archive, "download", lambda *a, **k: ran.append(a))
    res = upd.update_managed_comfy()

    assert res.ok is False
    assert res.status == "busy", res.status
    assert "already running" in res.message
    assert str(os.getpid()) in res.message, "name the holder so the user can check"
    # It refused BEFORE touching the tree.
    assert ran == [], "a refused update must not fetch anything"


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
    monkeypatch.setattr(source_archive, "download", lambda *a, **k: ran.append(a))
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

    upd.update_managed_comfy(target_commit="abc123")
    assert not _lock_dir().exists(), "lock leaked after a SUCCESSFUL update"

    # Now a failing update: the fetch fails, so it rolls back and returns an error.
    def _boom(url, dest):
        raise source_archive.ArchiveError("boom")

    monkeypatch.setattr(source_archive, "download", _boom)
    res = upd.update_managed_comfy(target_commit="b" * 40)
    assert res.ok is False
    assert not _lock_dir().exists(), "lock leaked after a FAILED update"


def test_the_lock_lives_outside_the_checkout_git_force_cannot_disturb_it(home):
    """The update's own source replace runs inside the checkout. A lock stored
    in there would be protecting itself from the very operation it guards."""
    _installed_git_checkout(home)
    root = mc.managed_comfy_paths().root
    lock = _lock_dir()
    assert root not in lock.parents, f"lock {lock} must not live inside {root}"
    assert lock.parent == root.parent


# --------------------------------------------------------------------------- #
#  remove_managed_comfy shares the SAME cross-process lock as update           #
# --------------------------------------------------------------------------- #
# A remove racing an in-flight `localm comfy update` CHILD PROCESS (or the CLI
# racing the GUI) must be refused, fast and honestly, and must leave the tree
# and the holder's lock alone. An in-process threading.Lock cannot see a second
# interpreter, so the second arm below uses a REAL child process.

def test_remove_refuses_while_another_process_holds_the_lock(home, monkeypatch):
    """The lock is held by a LIVE holder (this process's own pid, recorded as an
    update): remove must not delete a single path, must leave the lock in
    place, and must say who holds it."""
    import json as _json
    from localm.media import managed_comfy_update as upd
    paths = _installed_git_checkout(home)
    lock = _lock_dir()
    lock.mkdir(parents=True)
    (lock / upd._LOCK_OWNER).write_text(
        _json.dumps({"pid": os.getpid(), "op": "update"}), encoding="utf-8")

    rm: list = []
    monkeypatch.setattr(mc, "rmtree_robust", lambda p: rm.append(p))
    exc = None
    try:
        mc.remove_managed_comfy(False)
    except mc.ManagedComfyBusy as e:
        exc = e
    assert paths.root.exists(), "the tree was deleted under a live lock holder"
    assert rm == [], f"rmtree ran under a live lock holder: {rm}"
    assert lock.exists(), "a refused remove must not disturb the holder's lock"
    assert exc is not None, "remove did not refuse"
    assert "already running" in exc.reason and str(os.getpid()) in exc.reason, exc.reason
    assert "update" in exc.reason, exc.reason


# The child prints ITS OWN pid: sys.executable can be a venv launcher that runs
# the real interpreter as a grandchild, so Popen.pid is not the lock's holder.
_HOLDER = (
    "import os, sys\n"
    "from localm.media import managed_comfy as mc\n"
    "ok, why = mc._acquire_update_lock(op='update')\n"
    "print(('acquired ' + str(os.getpid())) if ok else 'busy:' + why, flush=True)\n"
    "sys.stdin.readline()\n"          # hold the lock until the parent says so
    "mc._release_update_lock()\n"
)


def test_remove_refuses_a_lock_held_by_a_REAL_separate_process(home, monkeypatch):
    """A genuinely separate interpreter holds the lock (what an in-flight
    `localm comfy update` child is). The parent's remove is refused and the
    tree survives; once the child releases, the same remove succeeds and the
    lock directory is gone. A threading.Lock cannot pass this arm."""
    import sys
    paths = _installed_git_checkout(home)
    env = dict(os.environ)
    env["LOCALM_HOME"] = str(cfg.HOME_DIR)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    child = subprocess.Popen([sys.executable, "-c", _HOLDER], env=env,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    try:
        first = child.stdout.readline().strip()
        assert first.startswith("acquired "), f"the child could not take the lock: {first!r}"
        holder_pid = first.split()[1]
        assert _lock_dir().exists(), "the child reported a lock that is not on disk"

        exc = None
        try:
            mc.remove_managed_comfy(False)
        except mc.ManagedComfyBusy as e:
            exc = e
        assert paths.root.exists(), "the tree was deleted under another process's lock"
        assert exc is not None, "remove did not refuse a lock held by another process"
        assert holder_pid in exc.reason, exc.reason

        child.stdin.write("go\n")
        child.stdin.flush()
        out, err = child.communicate(timeout=60)
        assert child.returncode == 0, err[-2000:]
    finally:
        if child.poll() is None:
            child.kill()
    assert not _lock_dir().exists(), "the child did not release the lock"

    removed, failed = mc.remove_managed_comfy(False)
    assert failed == []
    assert paths.root in removed
    assert not paths.root.exists()
    assert not _lock_dir().exists(), "remove leaked the lock it took"


def test_remove_releases_the_lock_on_success_and_on_failure(home, monkeypatch):
    """The lock is released whether the delete succeeded or reported a failure."""
    paths = _installed_git_checkout(home)

    def _boom(p):
        raise OSError("disk says no")
    with monkeypatch.context() as m:
        m.setattr(mc, "rmtree_robust", _boom)
        removed, failed = mc.remove_managed_comfy(False)
    assert removed == [] and failed and "disk says no" in failed[0]
    assert paths.root.exists()
    assert not _lock_dir().exists(), "lock leaked after a FAILED remove"

    removed, failed = mc.remove_managed_comfy(False)
    assert failed == [] and not paths.root.exists()
    assert not _lock_dir().exists(), "lock leaked after a SUCCESSFUL remove"


def test_remove_lock_names_a_remove_holder_to_a_contending_update(home, monkeypatch):
    """A lock taken by remove records op="remove"; an update refused by it says
    a remove is running, not an update."""
    from localm.media import managed_comfy_update as upd
    _installed_git_checkout(home)
    monkeypatch.setattr(upd, "_rev_parse_head", lambda root: "abc123")
    acquired, why = mc._acquire_update_lock(op="remove")
    assert acquired, why
    try:
        ran = []
        monkeypatch.setattr(source_archive, "download", lambda *a, **k: ran.append(a))
        res = upd.update_managed_comfy()
        assert res.ok is False and res.status == "busy"
        assert "remove" in res.message and str(os.getpid()) in res.message, res.message
        assert ran == []
    finally:
        mc._release_update_lock()


# =========================================================================== #
#  4. Folders of the managed ComfyUI that are the user's own (links)           #
# =========================================================================== #

def _link(path: Path, target: Path) -> None:
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(target), str(path))
    else:
        os.symlink(target, path)


def _linked(tmp_path, monkeypatch):
    hub = FakeGitHub(tmp_path / "hub").install(monkeypatch)
    extra = {"output/_put_here": "", "models/vae/put_vae_here": "", "input/example.png": "png",
             "custom_nodes/websocket.py": "one\n", "custom_nodes/same.py": "same\n",
             "custom_nodes/dropped.py": "old\n"}
    repo, c1 = hub.add("fake", "ComfyUI", _tree(**extra))
    _, c2 = hub.add("fake", "ComfyUI", _tree(VERSION="2\n", **{
        **extra, "custom_nodes/websocket.py": "two\n", "custom_nodes/new.py": "new\n",
        "models/vae/put_new_here": "", "custom_nodes/dropped.py": None,
        "models/configs/c.yaml": "new config\n"}))
    root = _install_managed_from(repo, c1)
    theirs = tmp_path / "theirs"
    for name in ("output", "models", "input", "custom_nodes"):
        (theirs / name).mkdir(parents=True)
        if name == "custom_nodes":
            for f in (root / name).iterdir():
                (theirs / name / f.name).write_bytes(f.read_bytes())
        import shutil
        shutil.rmtree(root / name)
        _link(root / name, theirs / name)
    (theirs / "output" / "mine.png").write_bytes(b"user data")
    os.rmdir(theirs / "input")   # a link that leads nowhere
    return repo, c2, root, theirs


def _files(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): p.read_text() for p in sorted(folder.rglob("*"))
            if p.is_file()}


def test_an_update_writes_only_changed_source_through_a_folder_of_the_user(home, tmp_path,
                                                                           monkeypatch):
    from localm.media import managed_comfy_update as upd
    repo, c2, root, theirs = _linked(tmp_path, monkeypatch)
    result = upd.update_managed_comfy(comfyui_repo=repo, target_commit=c2)
    assert result.ok, result.message
    assert _files(theirs) == {
        "output/mine.png": "user data", "models/configs/c.yaml": "new config\n",
        "custom_nodes/websocket.py": "two\n", "custom_nodes/same.py": "same\n",
        "custom_nodes/new.py": "new\n"}
    for name in ("output", "models", "custom_nodes"):
        assert os.path.realpath(root / name) == os.path.realpath(theirs / name)
    assert (root / "VERSION").is_file()


def _users_own(tmp_path, monkeypatch):
    """A managed ComfyUI whose ``models`` leads to a folder of the user's, which holds
    files under names of ComfyUI's source."""
    hub = FakeGitHub(tmp_path / "hub").install(monkeypatch)
    repo, c1 = hub.add("fake", "ComfyUI", _tree(**{
        "models/changes.yaml": "one", "models/stays.yaml": "source", "models/goes.yaml": "source",
        "models/emptied.py": "code", "models/fills_up": ""}))
    _, c2 = hub.add("fake", "ComfyUI", _tree(**{
        "models/changes.yaml": "two", "models/stays.yaml": "source", "models/emptied.py": "",
        "models/fills_up": "full", "models/added.py": "", "models/put_here": ""}))
    root = _install_managed_from(repo, c1)
    theirs = tmp_path / "theirs"
    theirs.mkdir()
    for name in ("changes.yaml", "stays.yaml", "goes.yaml"):
        (theirs / name).write_text(f"mine: {name}")
    (theirs / "emptied.py").write_text("code")
    (theirs / "fills_up").write_text("")
    import shutil
    shutil.rmtree(root / "models")
    _link(root / "models", theirs)
    return hub, repo, c2, root, theirs


_PUT_IN = {"changes.yaml": "two", "emptied.py": "", "fills_up": "full", "added.py": ""}
_KEPT = {"changes.yaml": "mine: changes.yaml", "goes.yaml": "mine: goes.yaml",
         "emptied.py": "code", "fills_up": ""}


def _kept_files(root: Path) -> dict:
    (kept,) = (root / ".localm-update" / "kept").iterdir()
    return _files(kept / "tree" / "models")


def test_a_file_of_the_user_behind_a_link_is_kept_or_left_alone(home, tmp_path, monkeypatch):
    from localm.media import managed_comfy_update as upd
    _hub, repo, c2, root, theirs = _users_own(tmp_path, monkeypatch)
    result = upd.update_managed_comfy(comfyui_repo=repo, target_commit=c2)
    assert result.ok, result.message
    assert _files(theirs) == {**_PUT_IN, "stays.yaml": "mine: stays.yaml"}
    assert _kept_files(root) == _KEPT
    assert "4 file(s) that were your own" in result.message and "kept" in result.message
    assert [p.name for p in (root / ".localm-update").iterdir() if p.is_dir()] == ["kept"]


def test_a_source_that_cannot_be_compared_loses_no_file_behind_a_link(home, tmp_path,
                                                                      monkeypatch):
    from localm.media import managed_comfy_update as upd
    hub, repo, c2, root, theirs = _users_own(tmp_path, monkeypatch)
    hub.archives = {url: path for url, path in hub.archives.items() if c2 in url}
    result = upd.update_managed_comfy(comfyui_repo=repo, target_commit=c2)
    assert result.ok, result.message
    assert _files(theirs) == {**_PUT_IN, "stays.yaml": "source"}
    assert _kept_files(root) == {**_KEPT, "stays.yaml": "mine: stays.yaml"}
    assert "could not be compared" in result.log


def test_an_update_compares_with_the_old_source_only_for_the_users_folders(home, tmp_path,
                                                                           monkeypatch):
    from localm.media import managed_comfy_update as upd
    hub, repo, c1, c2, _c3 = _build_update_remote(tmp_path, monkeypatch)
    _install_managed_from(repo, c1)
    hub.requested.clear()
    result = upd.update_managed_comfy(comfyui_repo=repo, target_commit=c2)
    assert result.ok, result.message
    assert [url.rpartition("/")[2] for url in hub.requested] == [c2]


def test_an_update_that_fails_puts_the_users_folder_back(home, tmp_path, monkeypatch):
    from localm.media import managed_comfy_update as upd
    repo, c2, root, theirs = _linked(tmp_path, monkeypatch)
    before = _files(theirs)
    monkeypatch.setattr(upd, "why_unusable", lambda packages: "it does not start")
    result = upd.update_managed_comfy(comfyui_repo=repo, target_commit=c2)
    assert not result.ok and "it does not start" in result.message
    assert _files(theirs) == before and not (root / "VERSION").exists()


def test_what_cannot_be_moved_in_is_found_before_anything_changes(home, tmp_path, monkeypatch):
    from localm.media import managed_comfy_update as upd
    from localm.update_mechanism import steps
    _hub, repo, c1, c2, _c3 = _build_update_remote(tmp_path, monkeypatch)
    root = _install_managed_from(repo, c1)
    (root / "VERSION").mkdir()
    (root / "VERSION" / "mine.txt").write_text("the user's")
    monkeypatch.setattr(steps, "change", lambda *a, **k: pytest.fail("it was started"))
    result = upd.update_managed_comfy(comfyui_repo=repo, target_commit=c2)
    assert not result.ok and "in the place of VERSION" in result.message


def test_a_managed_comfyui_with_incomplete_files_is_not_started(home, tmp_path, monkeypatch):
    from localm.media import comfy_client
    from localm.update_mechanism import job
    _hub, repo, c1, _c2, _c3 = _build_update_remote(tmp_path, monkeypatch)
    root = _install_managed_from(repo, c1)
    with job.Lock(root) as lock:
        (root / ".localm-update" / ("ab" * 16) / "plans").mkdir(parents=True)
        job.start(lock, "ab" * 16, old="", new="")
        ok, why = comfy_client._launch_and_wait("http://127.0.0.1:9", "x", str(root), 1,
                                                lambda _t: None)
    assert not ok and "are being replaced" in why
