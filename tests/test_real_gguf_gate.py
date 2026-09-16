# SPDX-License-Identifier: AGPL-3.0-or-later
"""The real-GGUF preconditions skip on ABSENCE only.

tests/_real_gguf.py and the real_gguf gate in tests/conftest.py decide when a
real-runtime test is skipped. A missing runtime file or an unreachable Hub is
an environment gap and skips; a runtime that is on disk but fails to load, or
a fetch that fails for a non-network reason, is a defect and must reach the
report as an error, never as a skip.

The sentinel at the bottom walks every test file and fails on the shape these
helpers replace: ``load_lib()``, ``hf_hub_download()`` or a model load wrapped
in a broad ``except`` that skips.
"""
from __future__ import annotations

import ast
import types
from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import (
    HfHubHTTPError, LocalEntryNotFoundError, XetDownloadError)

from localm.inference.backends.llamacpp import _loader
from tests import _real_gguf
from tests import conftest as _conftest
from tests._real_gguf import (
    fetch_gguf, native_runtime_lib_path, require_native_runtime)

_TESTS_DIR = Path(__file__).resolve().parent


def _outcome(fn, *args, **kwargs):
    """Run *fn* and return what it raised (None when it returned), so a skip,
    an error and a clean return are three distinguishable values instead of
    a skip silently ending the calling test."""
    try:
        fn(*args, **kwargs)
    except BaseException as e:      # a Skipped outcome is a BaseException
        return e
    return None


# --------------------------------------------------------------------------- #
#  native_runtime_lib_path: presence only
# --------------------------------------------------------------------------- #

def test_lib_path_is_none_when_no_candidate_dir_and_no_override(monkeypatch):
    monkeypatch.setattr(_loader, "runtime_binary_dir", lambda: None)
    monkeypatch.delenv("LLAMA_CPP_LIB", raising=False)
    assert native_runtime_lib_path() is None


def test_lib_path_found_in_runtime_binary_dir(monkeypatch, tmp_path):
    lib = tmp_path / _loader.lib_filename()
    lib.write_bytes(b"")
    monkeypatch.setattr(_loader, "runtime_binary_dir", lambda: tmp_path)
    monkeypatch.delenv("LLAMA_CPP_LIB", raising=False)
    assert native_runtime_lib_path() == lib


def test_lib_path_is_none_when_binary_dir_lacks_the_file(monkeypatch, tmp_path):
    monkeypatch.setattr(_loader, "runtime_binary_dir", lambda: tmp_path)
    monkeypatch.delenv("LLAMA_CPP_LIB", raising=False)
    assert native_runtime_lib_path() is None


def test_lib_path_honours_explicit_override_parent(monkeypatch, tmp_path):
    """Mirrors load_lib(): with no resolvable binary dir, the parent of
    LLAMA_CPP_LIB is tried, and the platform library name is what must exist
    there."""
    lib = tmp_path / _loader.lib_filename()
    lib.write_bytes(b"")
    monkeypatch.setattr(_loader, "runtime_binary_dir", lambda: None)
    monkeypatch.setenv("LLAMA_CPP_LIB", str(tmp_path / "whatever.so"))
    assert native_runtime_lib_path() == lib
    lib.unlink()
    assert native_runtime_lib_path() is None


# --------------------------------------------------------------------------- #
#  require_native_runtime: skip on absence, raise on a broken load
# --------------------------------------------------------------------------- #

def test_require_skips_and_never_loads_when_absent(monkeypatch):
    calls = []
    monkeypatch.setattr(_real_gguf, "native_runtime_lib_path", lambda: None)
    monkeypatch.setattr(_loader, "load_lib", lambda: calls.append(1))
    out = _outcome(require_native_runtime, setup_hint="localm setup-llama --backend cpu")
    assert calls == [], "load_lib() must not run when the runtime is absent"
    assert isinstance(out, pytest.skip.Exception), f"expected a skip, got {out!r}"
    assert "localm setup-llama --backend cpu" in str(out)


def test_require_raises_when_present_but_load_fails(monkeypatch, tmp_path):
    calls = []

    def broken_load_lib():
        calls.append(1)
        raise RuntimeError("verify_abi: llama_model_params layout mismatch")

    monkeypatch.setattr(_real_gguf, "native_runtime_lib_path",
                        lambda: tmp_path / "llama.dll")
    monkeypatch.setattr(_loader, "load_lib", broken_load_lib)
    out = _outcome(require_native_runtime)
    assert calls == [1]
    assert not isinstance(out, pytest.skip.Exception), (
        "a runtime that is on disk but fails to load was reported as a skip")
    assert isinstance(out, RuntimeError) and "verify_abi" in str(out)


def test_require_loads_once_when_present(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(_real_gguf, "native_runtime_lib_path",
                        lambda: tmp_path / "llama.dll")
    monkeypatch.setattr(_loader, "load_lib", lambda: calls.append(1))
    assert _outcome(require_native_runtime) is None
    assert calls == [1]


# --------------------------------------------------------------------------- #
#  fetch_gguf: skip only on network / offline / Hub-side failures
# --------------------------------------------------------------------------- #

def _http_error(status: int) -> HfHubHTTPError:
    resp = httpx.Response(status, request=httpx.Request("GET", "https://hub.invalid/x"))
    return HfHubHTTPError(f"HTTP {status}", response=resp)


@pytest.mark.parametrize("exc", [
    LocalEntryNotFoundError("offline and not in the cache"),
    _http_error(404),
    _http_error(503),
    httpx.ConnectError("name resolution failed"),
    httpx.ReadTimeout("read timed out"),
    XetDownloadError("xet transfer failed"),
], ids=["offline", "http404", "http503", "connect", "timeout", "xet"])
def test_fetch_skips_on_network_and_hub_conditions(monkeypatch, exc):
    import huggingface_hub

    def failing(**kwargs):
        raise exc

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", failing)
    out = _outcome(fetch_gguf, "org/repo", "model.gguf")
    assert isinstance(out, pytest.skip.Exception), f"expected a skip, got {out!r}"
    assert "org/repo/model.gguf" in str(out)


@pytest.mark.parametrize("exc", [
    TypeError("hf_hub_download() got an unexpected keyword argument"),
    RuntimeError("cache corrupted"),
    OSError(28, "No space left on device"),
    ValueError("bad revision"),
], ids=["typeerror", "runtimeerror", "oserror", "valueerror"])
def test_fetch_propagates_every_other_exception(monkeypatch, exc):
    import huggingface_hub

    def failing(**kwargs):
        raise exc

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", failing)
    out = _outcome(fetch_gguf, "org/repo", "model.gguf")
    assert not isinstance(out, pytest.skip.Exception), (
        f"{type(exc).__name__} from the fetch was reported as a skip")
    assert out is exc


def test_fetch_returns_the_path_and_forwards_kwargs(monkeypatch):
    import huggingface_hub
    seen = {}

    def ok(**kwargs):
        seen.update(kwargs)
        return "/cache/model.gguf"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", ok)
    assert fetch_gguf("org/repo", "model.gguf", revision="abc") == "/cache/model.gguf"
    assert seen == {"repo_id": "org/repo", "filename": "model.gguf", "revision": "abc"}


@pytest.mark.parametrize("outcome", ["returns", "raises"], ids=["ok", "failing"])
def test_fetch_disables_xet_for_the_call(monkeypatch, outcome):
    """During the download the xet transport is off, so a failed transfer is
    one of the typed httpx errors and never hf_xet's own RuntimeError; the
    switch is restored afterwards whether the fetch returned or raised."""
    import huggingface_hub
    from huggingface_hub import constants
    from huggingface_hub.utils._runtime import is_xet_available
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", False)
    during = {}

    def download(**kwargs):
        during["flag"] = constants.HF_HUB_DISABLE_XET
        during["available"] = is_xet_available()
        if outcome == "raises":
            raise RuntimeError("hf_xet: transfer failed")
        return "/cache/model.gguf"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    out = _outcome(fetch_gguf, "org/repo", "model.gguf")
    assert during == {"flag": True, "available": False}
    assert constants.HF_HUB_DISABLE_XET is False, "the xet switch was not restored"
    if outcome == "raises":
        assert isinstance(out, RuntimeError) and not isinstance(out, pytest.skip.Exception)
    else:
        assert out is None


# --------------------------------------------------------------------------- #
#  The conftest gate: a gated test skips on absence and errors on a broken load
# --------------------------------------------------------------------------- #

def _gated_item(*markers: str):
    return types.SimpleNamespace(keywords={m: True for m in markers})


@pytest.fixture
def fresh_gate(monkeypatch):
    """Isolate the per-process memo so this test neither reads a verdict left
    by an earlier real_gguf test in this worker nor leaves one behind."""
    monkeypatch.setattr(_conftest, "_resource_available", {})
    monkeypatch.delenv("LLAMA_CPP_LIB", raising=False)
    return _conftest


def test_gate_skips_when_runtime_absent(fresh_gate, monkeypatch):
    calls = []
    monkeypatch.setattr(_real_gguf, "native_runtime_lib_path", lambda: None)
    monkeypatch.setattr(_loader, "load_lib", lambda: calls.append(1))
    out = _outcome(fresh_gate.pytest_runtest_setup, _gated_item("real_gguf"))
    assert calls == []
    assert isinstance(out, pytest.skip.Exception), f"expected a skip, got {out!r}"
    assert "real_gguf" in str(out) and "not provisioned" in str(out)
    assert fresh_gate._resource_available == {"real_gguf": False}


def test_gate_errors_and_memoizes_when_runtime_present_but_broken(fresh_gate, monkeypatch, tmp_path):
    calls = []
    cause = RuntimeError("Failed to load llama.dll from x: entry point not found")

    def broken_load_lib():
        calls.append(1)
        raise cause

    monkeypatch.setattr(_real_gguf, "native_runtime_lib_path",
                        lambda: tmp_path / "llama.dll")
    monkeypatch.setattr(_loader, "load_lib", broken_load_lib)

    first = _outcome(fresh_gate.pytest_runtest_setup, _gated_item("real_gguf"))
    second = _outcome(fresh_gate.pytest_runtest_setup, _gated_item("real_gguf"))
    for out in (first, second):
        assert not isinstance(out, pytest.skip.Exception), (
            f"a present-but-broken runtime was reported as a skip: {out!r}")
        assert isinstance(out, RuntimeError)
        assert out.__cause__ is cause
        assert "real_gguf" in str(out) and "entry point not found" in str(out)
    assert calls == [1], "the failed check must be memoized, not re-run per test"
    assert fresh_gate._resource_available == {"real_gguf": cause}


def test_gate_passes_and_memoizes_when_runtime_loads(fresh_gate, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(_real_gguf, "native_runtime_lib_path",
                        lambda: tmp_path / "llama.dll")
    monkeypatch.setattr(_loader, "load_lib", lambda: calls.append(1))
    assert _outcome(fresh_gate.pytest_runtest_setup, _gated_item("real_gguf")) is None
    assert _outcome(fresh_gate.pytest_runtest_setup, _gated_item("real_gguf")) is None
    assert calls == [1]
    assert fresh_gate._resource_available == {"real_gguf": True}


def test_gate_ignores_an_ungated_item(fresh_gate, monkeypatch):
    def must_not_run():
        raise AssertionError("presence check ran for an ungated item")

    monkeypatch.setattr(_real_gguf, "native_runtime_lib_path", must_not_run)
    assert _outcome(fresh_gate.pytest_runtest_setup, _gated_item("integration")) is None
    assert fresh_gate._resource_available == {}


# --------------------------------------------------------------------------- #
#  Sentinel: no test file may wrap these calls in a broad except that skips
# --------------------------------------------------------------------------- #

# Every call that loads the native library or a model as a side effect; the
# _loader entry points listed all reach load_lib().
_GUARDED_CALLS = {
    "load_lib", "compute_backends_available", "compute_devices",
    "cpu_buffer_type", "native_device_inventory",
    "hf_hub_download", "GGUFEmbedder", "LlamaCpp",
}
# load_lib() raises RuntimeError for a runtime that is on disk but fails to
# load, so a handler for it is as broad as a bare except here.
_BROAD_TYPES = ("Exception", "BaseException", "RuntimeError")


def _call_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def _is_guarded_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    name = _call_name(node)
    if name in _GUARDED_CALLS:
        return True
    # A zero-argument ``x.load()`` is a backend/model load; json.load(f) and
    # friends take an argument.
    return (name == "load" and isinstance(node.func, ast.Attribute)
            and not node.args and not node.keywords)


def _is_broad_handler(handler: ast.ExceptHandler) -> bool:
    t = handler.type
    if t is None:
        return True
    if isinstance(t, ast.Name):
        return t.id in _BROAD_TYPES
    if isinstance(t, ast.Tuple):
        return any(isinstance(e, ast.Name) and e.id in _BROAD_TYPES
                   for e in t.elts)
    return False


def _handler_skips(handler: ast.ExceptHandler) -> bool:
    return any(isinstance(n, ast.Call) and _call_name(n) == "skip"
               for n in ast.walk(handler))


def broad_skip_wrappers(source: str) -> list:
    """Line numbers of every ``try`` whose body makes a guarded call and whose
    broad ``except`` handler skips."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Try):
            continue
        if not any(_is_guarded_call(n) for stmt in node.body for n in ast.walk(stmt)):
            continue
        if any(_is_broad_handler(h) and _handler_skips(h) for h in node.handlers):
            hits.append(node.lineno)
    return hits


def test_sentinel_recognises_the_shape_it_guards():
    src = (
        "def f():\n"
        "    try:\n"
        "        load_lib()\n"
        "    except Exception as e:\n"
        "        pytest.skip(str(e))\n"
        "    try:\n"
        "        path = hf_hub_download(repo_id=r, filename=f)\n"
        "    except Exception as e:\n"
        "        pytest.skip(str(e))\n"
        "    try:\n"
        "        be.load()\n"
        "    except Exception as e:\n"
        "        pytest.skip(str(e))\n"
        "    try:\n"
        "        emb = GGUFEmbedder(path)\n"
        "    except:\n"
        "        pytest.skip('x')\n"
        "    try:\n"
        "        llm = LlamaCpp(path)\n"
        "    except (OSError, Exception):\n"
        "        pytest.skip('x')\n"
        "    try:\n"
        "        data = json.load(fh)\n"
        "    except Exception:\n"
        "        pytest.skip('unrelated: takes an argument')\n"
        "    try:\n"
        "        _loader.load_lib()\n"
        "    except RuntimeError:\n"
        "        pytest.skip('load_lib raises RuntimeError for a broken load')\n"
        "    try:\n"
        "        if not _loader.compute_backends_available():\n"
        "            pytest.skip('predicate skip is fine; the wrapper is not')\n"
        "    except Exception as e:\n"
        "        pytest.skip(str(e))\n"
        "    try:\n"
        "        load_lib()\n"
        "    except OSError:\n"
        "        pytest.skip('narrow handler: allowed')\n"
        "    try:\n"
        "        import playwright\n"
        "    except Exception:\n"
        "        pytest.skip('optional import: allowed')\n"
    )
    assert broad_skip_wrappers(src) == [2, 6, 10, 14, 18, 26, 30]


def test_no_test_file_wraps_a_runtime_or_model_load_in_a_broad_skip():
    offenders = []
    for path in sorted(_TESTS_DIR.rglob("*.py")):
        if path == Path(__file__).resolve():
            continue
        rel = path.relative_to(_TESTS_DIR).as_posix()
        for line in broad_skip_wrappers(path.read_text(encoding="utf-8")):
            offenders.append(f"tests/{rel}:{line}")
    assert not offenders, (
        "these try blocks turn a runtime or model load failure into a skip; "
        "call tests._real_gguf.require_native_runtime() / fetch_gguf() and let "
        "the load itself raise:\n  " + "\n  ".join(offenders))
