# SPDX-License-Identifier: AGPL-3.0-or-later
"""Oracle for the reactive, opt-in ComfyUI ``__func__`` regression shim.

The shim is a localm-owned ``sitecustomize.py`` that localm puts on the PYTHONPATH
of a ComfyUI process it SPAWNS ITSELF, to patch a ComfyUI core regression in
memory. The absolute invariant: localm writes NOTHING into the user's ComfyUI
install and only ever shims a ComfyUI it launched.

Four checks, each with a built-in negative case so it fails on known-bad work:
  1. DEFAULT no-op: a normal spawn never puts the shim dir on the child PYTHONPATH.
  2. DETECTION: the specific ``__func__`` signature classifies as the regression; an
     unrelated exec error does not.
  3. SHIM: applied to a fake buggy ``comfy_api.internal`` it stops the AttributeError
     and calls the node right for sync AND async; it no-ops on an already-tolerant or
     differently-shaped module. A real subprocess proves the PYTHONPATH path too.
  4. APPLY SCOPE: enabled -> ONLY the shim dir is prepended to the child PYTHONPATH
     (a pre-existing PYTHONPATH preserved); nothing is written into the install dir.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


# --- fixtures: fake ComfyUI internals written to real files so getsource works ----

# Copy of the upstream body; the `.__func__` access is the line that raises.
_BUGGY_SRC = '''\
import asyncio


def lock_class(cls):
    # Real upstream rebuilds a locked class; identity is enough to prove the wiring.
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
    # A staticmethod resolves via getattr(cls, "execute") to a PLAIN function with no
    # __func__ - exactly the VAEDecodeAudio shape that trips the regression.
    execute = staticmethod(_plain_execute)


class AsyncPlainFnNode:
    execute = staticmethod(_plain_async_execute)


class BoundNode:
    # A classmethod resolves to a bound method that DOES have __func__ (the case the
    # buggy code handled); our tolerant version must keep handling it.
    @classmethod
    def execute(cls, **inputs):
        return ("bound", cls, inputs)
'''

# Already-tolerant: uses the string form getattr(x, "__func__", x) -> shim must no-op.
_TOLERANT_SRC = '''\
import asyncio


def lock_class(cls):
    return cls


def make_locked_method_func(type_obj, func, class_clone):
    locked_class = lock_class(class_clone)
    attr = getattr(type_obj, func)
    method = getattr(attr, "__func__", attr)
    if asyncio.iscoroutinefunction(method):
        async def wrapped_async_func(**inputs):
            return await method(locked_class, **inputs)
        return wrapped_async_func
    else:
        def wrapped_func(**inputs):
            return method(locked_class, **inputs)
        return wrapped_func
'''

# Different signature -> unexpected shape -> shim must no-op.
_WRONGSHAPE_SRC = '''\
def lock_class(cls):
    return cls


def make_locked_method_func(a, b):
    return None
'''

# Right shape but no lock_class -> our replacement needs it, so no-op.
_NO_LOCKCLASS_SRC = '''\
def make_locked_method_func(type_obj, func, class_clone):
    method = getattr(type_obj, func).__func__
    return method
'''


def _import_fake(tmp_path, src, name):
    p = tmp_path / (name + ".py")
    p.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_shim_module():
    """Load the REAL shim file as a throwaway module and return it, restoring
    sys.meta_path so the shim's deferred finder does not leak into the test session."""
    from localm.media.comfy_client import comfy_shim_dir
    src_path = comfy_shim_dir() / "sitecustomize.py"
    saved = list(sys.meta_path)
    spec = importlib.util.spec_from_file_location("_localm_shim_under_test", src_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.meta_path[:] = saved
    return mod


# ------------------------------- 2. DETECTION -------------------------------------

def test_detection_matches_only_the_known_regression():
    from localm.media.comfy_client import is_known_comfy_func_regression as det
    # positive: the exact AttributeError text (any object kind), and the function name
    assert det("'function' object has no attribute '__func__' (node VAEDecodeAudio)")
    assert det("'method' object has no attribute '__func__'")
    assert det("Traceback ... in make_locked_method_func ...")
    # negative: an unrelated execution error must NOT be classified as the regression
    assert not det("CUDA out of memory (node KSampler)")
    assert not det("Missing checkpoint ace_step_v1_3.5b.safetensors")
    assert not det("")
    assert not det(None)


def test_exec_error_message_is_actionable_only_for_the_regression(monkeypatch):
    from localm.media import comfy_client as c
    monkeypatch.setattr(c, "spawned_pid", lambda *a, **k: None)
    # unrelated error keeps the plain wording (no false offer)
    generic = c.comfy_exec_error_message("CUDA out of memory", "http://127.0.0.1:8188")
    assert generic == "ComfyUI execution failed: CUDA out of memory"
    # The known shape gets a richer message: it names the upstream ref and the
    # localm-side toggle, and states that nothing is written into the install.
    rich = c.comfy_exec_error_message(
        "'function' object has no attribute '__func__'", "http://127.0.0.1:8188")
    assert rich != generic
    assert "12116" in rich
    assert "comfy_func_shim" in rich
    assert "nothing" in rich.lower() or "no file" in rich.lower()


# --------------------------------- 3. SHIM ----------------------------------------

def test_shim_patches_plain_function_node_sync_and_async(tmp_path):
    mod = _import_fake(tmp_path, _BUGGY_SRC, "buggy_mod")
    # NEGATIVE: the UNPATCHED buggy function raises on a plain-function node.
    with pytest.raises(AttributeError):
        mod.make_locked_method_func(mod.PlainFnNode, "execute", mod.PlainFnNode)

    shim = _load_shim_module()
    assert shim._patch_module(mod) is True

    # sync plain-function node now works instead of raising
    wrapper = mod.make_locked_method_func(mod.PlainFnNode, "execute", mod.PlainFnNode)
    assert wrapper(x=1) == ("sync", mod.PlainFnNode, {"x": 1})
    # async branch preserved
    awrapper = mod.make_locked_method_func(
        mod.AsyncPlainFnNode, "execute", mod.AsyncPlainFnNode)
    import asyncio
    assert asyncio.run(awrapper(y=2)) == ("async", mod.AsyncPlainFnNode, {"y": 2})
    # the bound-method case the buggy code already handled still works
    bwrapper = mod.make_locked_method_func(mod.BoundNode, "execute", mod.BoundNode)
    assert bwrapper(z=3) == ("bound", mod.BoundNode, {"z": 3})
    # idempotent: patching an already-ours module is a no-op
    assert shim._patch_module(mod) is False


@pytest.mark.parametrize("src,name", [
    (_TOLERANT_SRC, "tolerant_mod"),
    (_WRONGSHAPE_SRC, "wrongshape_mod"),
    (_NO_LOCKCLASS_SRC, "nolock_mod"),
], ids=["already_tolerant", "unexpected_shape", "no_lock_class"])
def test_shim_noops_on_unpatchable_module(tmp_path, src, name):
    mod = _import_fake(tmp_path, src, name)
    shim = _load_shim_module()
    assert shim._build_tolerant(mod) is None
    if name == "tolerant_mod":
        # only the already-tolerant source also exercises _patch_module (the other two
        # sources are not full modules _patch_module would attempt to patch).
        assert shim._patch_module(mod) is False


def _tree_snapshot(root):
    # Relative file paths under root, excluding Python's own bytecode cache,
    # which any import writes.
    out = set()
    for dirpath, _dirs, files in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for f in files:
            if f.endswith(".pyc"):
                continue
            out.add(os.path.relpath(os.path.join(dirpath, f), root))
    return out


def test_shim_applies_via_pythonpath_in_a_real_subprocess(tmp_path):
    """End-to-end at the process level: a real python with the shim dir + a fake buggy
    ComfyUI on PYTHONPATH auto-imports the shim (sitecustomize) and patches the module,
    while writing nothing into the fake install."""
    import subprocess
    from localm.media.comfy_client import comfy_shim_dir

    install = tmp_path / "comfy_install"
    pkg = install / "comfy_api" / "internal"
    pkg.mkdir(parents=True)
    (install / "comfy_api" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text(_BUGGY_SRC, encoding="utf-8")
    driver = tmp_path / "driver.py"
    driver.write_text(
        "from comfy_api.internal import make_locked_method_func, PlainFnNode\n"
        "w = make_locked_method_func(PlainFnNode, 'execute', PlainFnNode)\n"
        "print('RESULT', w(a=1))\n",
        encoding="utf-8")

    before = _tree_snapshot(install)
    shim = str(comfy_shim_dir())

    base_env = dict(os.environ)
    # Do not write .pyc: the assertion below counts files added to the install.
    base_env["PYTHONDONTWRITEBYTECODE"] = "1"

    # WITH the shim on PYTHONPATH -> patched: node runs, and the INFO line is emitted.
    env_on = dict(base_env)
    env_on["PYTHONPATH"] = shim + os.pathsep + str(install)
    r = subprocess.run([sys.executable, str(driver)], capture_output=True,
                       text=True, env=env_on, cwd=str(tmp_path))
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert "RESULT ('sync'," in r.stdout
    assert "Applied an in-memory ComfyUI __func__ compatibility shim" in r.stderr

    # WITHOUT the shim -> the fixture raises.
    env_off = dict(base_env)
    env_off["PYTHONPATH"] = str(install)
    r2 = subprocess.run([sys.executable, str(driver)], capture_output=True,
                        text=True, env=env_off, cwd=str(tmp_path))
    assert r2.returncode != 0
    assert "has no attribute '__func__'" in r2.stderr

    # Nothing was written into the fake ComfyUI install.
    assert _tree_snapshot(install) == before


# ---------------------- 1 + 4. DEFAULT no-op / APPLY SCOPE ------------------------

def _capture_spawn_env(monkeypatch, tmp_path, *, once=False, remember=False,
                       preexisting_pythonpath=None):
    """Drive ensure_comfy through a faked Popen and return the env dict it was given."""
    import subprocess

    from localm.image_gen import comfy
    from localm.media import comfy_client

    comfy_client._confirmed_alive.clear()
    comfy_client._func_shim_once = False
    if once:
        comfy_client.enable_func_shim_once()

    if preexisting_pythonpath is None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", preexisting_pythonpath)

    if not hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 512, raising=False)

    captured = {}

    def fake_popen(argv, **kw):
        captured["argv"] = argv
        captured["env"] = kw.get("env")
        captured["cwd"] = kw.get("cwd")
        proc = MagicMock()
        proc.poll.return_value = None
        return proc

    cfg = {
        "comfy_launch_cmd": '"%s" main.py' % sys.executable,
        "comfy_workdir": str(tmp_path),
        "comfy_launch_timeout": 30,
    }
    if remember:
        cfg["comfy_func_shim"] = True

    # dead, dead-under-lock (the double-checked re-check under _launch_lock_for),
    # then up after spawn: ensure_comfy() makes three _comfy_alive() calls
    # minimum before a launch is confirmed.
    alive = iter([False, False, True])
    with patch("localm.config.load_config", return_value=cfg), \
         patch("subprocess.Popen", side_effect=fake_popen), \
         patch.object(comfy_client, "_comfy_alive",
                      side_effect=lambda *a, **k: next(alive)):
        comfy.ensure_comfy("http://127.0.0.1:8188")

    comfy_client._func_shim_once = False  # do not leak process state across tests
    return captured


def test_default_run_puts_no_shim_on_pythonpath(monkeypatch, tmp_path):
    from localm.media.comfy_client import comfy_shim_dir
    cap = _capture_spawn_env(monkeypatch, tmp_path)  # neither once nor remember
    shim = str(comfy_shim_dir())
    env = cap["env"]
    pythonpath = (env or {}).get("PYTHONPATH", "")
    assert shim not in pythonpath
    # and the workdir (the stand-in install) got no files from the (faked) spawn
    assert list(tmp_path.iterdir()) == []


def test_remember_prepends_only_the_shim_dir_preserving_pythonpath(monkeypatch, tmp_path):
    from localm.media.comfy_client import comfy_shim_dir
    pre = os.path.join("some", "preexisting", "entry")
    cap = _capture_spawn_env(monkeypatch, tmp_path, remember=True,
                             preexisting_pythonpath=pre)
    shim = str(comfy_shim_dir())
    env = cap["env"]
    assert env is not None
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == shim          # the shim dir is prepended...
    assert pre in parts              # ...and the pre-existing PYTHONPATH is preserved
    assert parts.count(shim) == 1    # only the shim dir is added


def test_apply_once_enables_the_shim_for_the_next_spawn(monkeypatch, tmp_path):
    from localm.media.comfy_client import comfy_shim_dir
    cap = _capture_spawn_env(monkeypatch, tmp_path, once=True)
    shim = str(comfy_shim_dir())
    env = cap["env"]
    assert env is not None
    assert shim in env["PYTHONPATH"].split(os.pathsep)


def test_shim_dir_is_inside_localm_not_the_comfy_install():
    from localm.media.comfy_client import comfy_shim_dir
    import localm
    d = comfy_shim_dir()
    localm_root = os.path.dirname(os.path.abspath(localm.__file__))
    # the shim lives inside the localm package, never in a ComfyUI folder
    assert str(d).startswith(localm_root)
    assert (d / "sitecustomize.py").is_file()


# ------------------------------ CLI offer + retry ---------------------------------

def _regression_msg():
    return "ComfyUI execution failed: 'function' object has no attribute '__func__'"


def test_cli_offer_remember_applies_config_restarts_and_retries(monkeypatch):
    from localm.cli import media
    from localm.media import comfy_client as cc
    cc._func_shim_once = False

    monkeypatch.setattr(media, "_is_interactive", lambda: True)
    monkeypatch.setattr(media.console, "input", lambda *a, **k: "r")
    remembered = {}
    monkeypatch.setattr(media, "_remember_func_shim",
                        lambda: remembered.__setitem__("v", True))
    monkeypatch.setattr(cc, "spawned_pid", lambda *a, **k: 4321)
    restarted = {}

    def _restart(*a, **k):
        restarted["v"] = True
        return True, "ok"

    monkeypatch.setattr(cc, "restart_comfy", _restart)

    retried = {"n": 0}

    def retry():
        retried["n"] += 1
        return True, "Track saved"

    ok, out = media._maybe_apply_func_shim_and_retry(
        _regression_msg(), "http://127.0.0.1:8188", retry)
    assert ok and out == "Track saved"
    assert remembered.get("v") is True
    assert restarted.get("v") is True
    assert retried["n"] == 1
    assert cc._func_shim_once is True
    cc._func_shim_once = False


def test_cli_offer_declined_changes_nothing(monkeypatch):
    from localm.cli import media
    from localm.media import comfy_client as cc
    cc._func_shim_once = False

    monkeypatch.setattr(media, "_is_interactive", lambda: True)
    monkeypatch.setattr(media.console, "input", lambda *a, **k: "n")
    remembered = {}
    monkeypatch.setattr(media, "_remember_func_shim",
                        lambda: remembered.__setitem__("v", True))

    retried = {"n": 0}

    def retry():
        retried["n"] += 1
        return True, "should not happen"

    ok, out = media._maybe_apply_func_shim_and_retry(
        _regression_msg(), "http://127.0.0.1:8188", retry)
    assert ok is False
    assert out == _regression_msg()
    assert retried["n"] == 0
    assert "v" not in remembered
    assert cc._func_shim_once is False


def test_cli_offer_skips_unrelated_error_without_prompting(monkeypatch):
    from localm.cli import media
    monkeypatch.setattr(media, "_is_interactive", lambda: True)
    prompted = {"v": False}

    def _input(*a, **k):
        prompted["v"] = True
        return "r"

    monkeypatch.setattr(media.console, "input", _input)

    def retry():
        raise AssertionError("must not retry an unrelated error")

    ok, out = media._maybe_apply_func_shim_and_retry(
        "ComfyUI execution failed: CUDA out of memory", "http://127.0.0.1:8188", retry)
    assert ok is False
    assert out.endswith("CUDA out of memory")
    assert prompted["v"] is False
