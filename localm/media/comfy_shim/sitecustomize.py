# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm-owned in-memory compatibility shim for a ComfyUI CORE regression.

This file is a plain ``sitecustomize.py`` living in a localm-owned directory.
localm adds that directory to the PYTHONPATH of a ComfyUI process localm ITSELF
spawns, so Python auto-imports this module at interpreter startup and it patches
the regression in memory. It touches NOTHING localm did not spawn: it writes no
file into the user's ComfyUI install and does not change how the user launches
ComfyUI. If the user starts ComfyUI their own way, this file is simply not on the
path and nothing here runs.

The regression: ``comfy_api/internal/__init__.py::make_locked_method_func`` does
``getattr(type_obj, func).__func__``, assuming a node's FUNCTION is a bound method.
A node whose FUNCTION resolves to a plain function (the core audio VAEDecodeAudio,
used by native ACE-Step) has no ``.__func__`` -> ``AttributeError: 'function'
object has no attribute '__func__'``. The patch makes that one access tolerant of
a plain function; everything else is a faithful copy of the upstream body.

Hard rules for this file (do not relax):
- It MUST NOT import localm - it runs inside the ComfyUI interpreter, not localm's.
- It MUST NEVER raise: every path is wrapped so a failure becomes a silent no-op
  and can never break ComfyUI startup.
- It is conservative: it patches ONLY when make_locked_method_func exists AND has
  the exact known signature AND still does the fragile UNGUARDED ``.__func__``
  access. On any unexpected or already-fixed shape it no-ops, so it stops
  applying once upstream ships its own fix.
- It prints ONE line, only when it actually patches, so a real patch is
  discoverable (localm captures the ComfyUI launcher output to a log).
- It best-effort chains any pre-existing sitecustomize it shadowed on the path.
"""

# No `from __future__ import annotations` and no localm imports: this module is
# executed by the ComfyUI interpreter as a bare top-level sitecustomize.

_TARGET = "comfy_api.internal"


def _build_tolerant(mod):
    """Return a faithful, plain-function-tolerant ``make_locked_method_func`` bound
    to *mod*'s ``lock_class``, or None when *mod* is not the exact known-buggy shape.

    Conservative by construction: any mismatch (missing name, wrong signature,
    already tolerant, source unavailable) returns None so the caller no-ops."""
    import asyncio
    import inspect

    fn = getattr(mod, "make_locked_method_func", None)
    if fn is None or not callable(fn):
        return None
    if getattr(fn, "_localm_func_shim", False):
        return None  # already patched by us
    lock_class = getattr(mod, "lock_class", None)
    if not callable(lock_class):
        return None  # our replacement calls lock_class; refuse without it
    # The signature must match the known-buggy shape exactly, else leave it alone.
    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return None
    if params != ["type_obj", "func", "class_clone"]:
        return None
    # The fragile UNGUARDED access must be present, and no tolerance already added.
    # `).__func__` is the direct dotted access on the getattr result; a defensive
    # rewrite uses the string form `getattr(x, "__func__", ...)` / `hasattr(x,
    # "__func__")`, so the presence of the literal means "already fixed" -> no-op.
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return None
    if ").__func__" not in src:
        return None  # not the access we know how to fix -> unexpected shape
    if '"__func__"' in src or "'__func__'" in src:
        return None  # looks already tolerant -> self-expire

    def make_locked_method_func(type_obj, func, class_clone):
        # Faithful copy of the upstream body; the ONLY change is tolerating a
        # plain function (no ``.__func__``) instead of assuming a bound method.
        locked_class = lock_class(class_clone)
        attr = getattr(type_obj, func)
        method = attr.__func__ if hasattr(attr, "__func__") else attr
        if asyncio.iscoroutinefunction(method):
            async def wrapped_async_func(**inputs):
                return await method(locked_class, **inputs)
            return wrapped_async_func

        def wrapped_func(**inputs):
            return method(locked_class, **inputs)
        return wrapped_func

    make_locked_method_func._localm_func_shim = True
    try:
        make_locked_method_func.__doc__ = fn.__doc__
    except Exception:
        pass
    return make_locked_method_func


def _patch_module(mod):
    """Swap the tolerant function onto *mod*. Return True iff it patched. Never
    raises. Prints one discoverable INFO line only on a real patch."""
    try:
        tolerant = _build_tolerant(mod)
        if tolerant is None:
            return False
        mod.make_locked_method_func = tolerant
    except Exception:
        return False
    try:
        import sys
        sys.stderr.write(
            "[localm] Applied an in-memory ComfyUI __func__ compatibility shim "
            "(upstream regression, Comfy-Org/ComfyUI #12116). Nothing was written "
            "to your ComfyUI install.\n")
    except Exception:
        pass
    return True


def _install():
    """Patch comfy_api.internal now if it is already imported; otherwise arrange to
    patch it right after ComfyUI imports it, WITHOUT importing it ourselves (so we
    add no early-import cost and no side effects). Never raises."""
    import sys
    try:
        mod = sys.modules.get(_TARGET)
        if mod is not None:
            _patch_module(mod)
            return
    except Exception:
        pass
    try:
        import importlib.util

        class _ComfyFuncShimFinder:
            """A meta_path finder that wraps the real loader for exactly one module,
            runs our patch after its exec, then removes itself. Returns None for
            everything else, so it never interferes with any other import."""

            def find_spec(self, name, path=None, target=None):
                if name != _TARGET:
                    return None
                # Remove ourselves before delegating so the real machinery (and our
                # own find_spec below) does not recurse back into us.
                try:
                    sys.meta_path.remove(self)
                except ValueError:
                    pass
                try:
                    spec = importlib.util.find_spec(name)
                except Exception:
                    spec = None
                if spec is None or spec.loader is None:
                    return None
                loader = spec.loader
                orig_exec = getattr(loader, "exec_module", None)
                if orig_exec is None:
                    return None

                def exec_module(module, _orig=orig_exec):
                    _orig(module)
                    _patch_module(module)

                try:
                    loader.exec_module = exec_module
                except Exception:
                    return None
                return spec

        sys.meta_path.insert(0, _ComfyFuncShimFinder())
    except Exception:
        pass


def _chain_previous_sitecustomize():
    """Best-effort: run the next sitecustomize.py our directory shadowed on
    sys.path, so a user's own environment customization still runs. Never raises."""
    try:
        import importlib.util
        import os
        import sys
        try:
            our_dir = os.path.dirname(os.path.abspath(__file__))
        except Exception:
            return
        for entry in list(sys.path):
            try:
                if not entry or os.path.abspath(entry) == our_dir:
                    continue
                cand = os.path.join(entry, "sitecustomize.py")
                if not os.path.isfile(cand):
                    continue
                spec = importlib.util.spec_from_file_location(
                    "_localm_shadowed_sitecustomize", cand)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                break
            except Exception:
                continue
    except Exception:
        pass


# Run at import (interpreter startup). Every callee is already internally guarded,
# and the whole block is wrapped as well: a sitecustomize must never break the
# interpreter it loads into, so anything that escapes degrades to a total no-op and
# ComfyUI runs exactly as if the shim were not present. The INFO line prints ONLY
# on a real patch, so no line means not patched and the generation still fails
# with its own reason.
try:
    try:
        _install()
    finally:
        _chain_previous_sitecustomize()
except Exception:
    pass
