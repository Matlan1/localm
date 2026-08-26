#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Confirm a llama.cpp release LOADS AND GENERATES before it is pinned.

``localm/setup_llama.py`` installs ``_PINNED_TAG`` - one release we decided on -
rather than whatever upstream published most recently. This script is what earns
that word "confirmed": a build that merely LOADS is not confirmed, because a
release can load fine for upstream and still be refused by localm's own ABI gate.

WHAT IT DOES, per requested backend:

  1. resolves that backend's asset for the tag through setup_llama's OWN resolver,
     so the thing tested is the thing an install would fetch, not a hand-built URL;
  2. downloads, checksum-validates and extracts it into a THROWAWAY directory;
  3. loads it in a fresh interpreter through localm's real loader, and reports the
     ABI verdict and the compute backends it registered;
  4. loads a small real GGUF and GENERATES tokens through the product's own
     GgufBackend, in the isolated worker process a real chat uses.

WHAT IT DOES NOT DO: touch the provisioned runtime. It never calls setup-llama and
never writes into the localm-llama-runtime wheel, so running it cannot disturb the
runtime this machine (or another session) is using. The build under test is
injected with ``LLAMA_CPP_LIB``, a documented first-class override.

PROVING THE INJECTION TOOK. An override that silently failed to apply looks
exactly like a healthy run against the build under test - it would load the
ALREADY-PROVISIONED runtime, generate perfectly, and report PASS for a binary it
never opened. So:

  * the probe asserts ``runtime_binary_dir()`` resolves inside the throwaway dir,
    and reports INCONCLUSIVE rather than PASS when it does not;
  * generation runs in an isolated WORKER process, a different process from the
    probe, so the probe's own successful override proves nothing about it. The
    worker's mapped modules are read back (psutil) and the loaded llama library
    is checked to be the one under test. A worker that mapped a different llama
    library, or one whose modules cannot be read, is INCONCLUSIVE - never PASS.

Three outcomes, kept distinct: PASS (loaded and generated, and both were proven to
be the build under test), FAIL (the build is bad - it did not load, or the ABI
gate refused it, or it produced no tokens), and INCONCLUSIVE (we could not
measure: no network, no model, an override that did not apply, an unreadable
worker). Exit code 0 / 1 / 2 in that order.

WHAT ONE RUN CAN AND CANNOT COVER:

  * The ABI/struct property is BACKEND-INDEPENDENT by construction. Upstream ships
    ONE ``llama.dll`` for every backend of a given tag - the backend lives in the
    separate ``ggml-*`` plugin libraries - so a single backend's ABI verdict covers
    them all. This script hashes the library for every backend it fetched and
    prints whether they are byte-identical, so that premise is re-measured at each
    tag rather than inherited.
  * GENERATION is backend-specific and hardware-specific. A cpu run says nothing
    about whether a cuda build produces tokens on an NVIDIA card. Only the
    backends actually run here are confirmed for generation; the summary names the
    ones that were not, and ``setup_llama._PIN_CONFIRMATION`` is where that record
    lives for the shipped pin.

Usage:
    python scripts/confirm_llama_runtime.py --tag b10375
    python scripts/confirm_llama_runtime.py --tag b10375 --backend cpu --backend vulkan
    python scripts/confirm_llama_runtime.py            # confirm the current pin

Needs localm importable (unlike scripts/check_llama_pin.py, which is stdlib-only).
Nothing under localm/ imports this; it never runs from a user's install.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_MODEL_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
_MODEL_FILE = "SmolLM2-135M-Instruct-Q4_K_M.gguf"

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"


# --------------------------------------------------------------------------- #
#  The probe: runs in a fresh interpreter with LLAMA_CPP_LIB pointed at the    #
#  extracted build. Its stdout's last line is a JSON verdict.                  #
# --------------------------------------------------------------------------- #
_PROBE = r'''
import json, os, sys
from pathlib import Path

out = {"verdict": "INCONCLUSIVE", "why": "", "abi": None, "compute_backends": None,
       "lib_dir": None, "worker_lib": None, "sample": "", "devices": None,
       "gpu_layers": None}
expected = Path(os.environ["LOCALM_CONFIRM_EXPECT_DIR"]).resolve()
model = os.environ["LOCALM_CONFIRM_MODEL"]

def emit(**kw):
    out.update(kw)
    sys.stdout.write("\n@@VERDICT@@" + json.dumps(out) + "\n")
    sys.stdout.flush()
    raise SystemExit(0)

try:
    from localm.inference.backends.llamacpp import _loader
    from localm.inference.backends.llamacpp._abi import AbiMismatch
except Exception as e:
    emit(verdict="INCONCLUSIVE", why="localm is not importable here: %s" % e)

# 1. Load, and prove we loaded the build under test rather than whatever this box
#    already has provisioned. The order matters: an override that did not apply
#    would otherwise load the provisioned runtime and report a healthy PASS.
try:
    _loader.load_lib()
except AbiMismatch as e:
    out["lib_dir"] = str(_loader.runtime_binary_dir())
    emit(verdict="FAIL", why="localm's ABI gate refused this build: %s" % e)
except Exception as e:
    emit(verdict="FAIL", why="the native library did not load: %s" % e)

got = _loader.runtime_binary_dir()
out["lib_dir"] = str(got) if got else None
try:
    resolved_ok = got is not None and Path(got).resolve() == expected
except Exception:
    resolved_ok = False
if not resolved_ok:
    emit(verdict="INCONCLUSIVE",
         why="LLAMA_CPP_LIB did not take effect - loaded %r, expected %r. Nothing "
             "was measured about the build under test." % (out["lib_dir"], str(expected)))

# 2. The ABI verdict and the compute backends this build registered. A build that
#    loads but registers nothing is not usable, and must not read as a success.
try:
    from localm.inference.backends.llamacpp._abi import abi_report
    out["abi"] = abi_report().status
except Exception as e:
    out["abi"] = "unavailable: %s" % e
# compute_backends_available() returns a BOOL ("at least one ggml compute backend
# registered"), not a list. Read it as one: an earlier version of this probe did
# list(...) on it, which raised, and the resulting ERROR STRING was truthy - so
# the "did it register a backend" gate passed on a probe that had failed to run.
# An unmeasured check must never read as a passed one.
try:
    out["compute_backends"] = bool(_loader.compute_backends_available())
except Exception as e:
    emit(verdict="INCONCLUSIVE",
         why="could not tell whether any compute backend registered: %s" % e)
if out["abi"] == "mismatch":
    emit(verdict="FAIL", why="abi_report() reports a mismatch against this build")
if out["compute_backends"] is not True:
    emit(verdict="FAIL",
         why='the runtime registered no compute backends ("no backends are loaded")')

# WHICH DEVICES this build actually offers, recorded so a GPU backend's verdict
# cannot quietly become a statement about the CPU. A vulkan build that fell back
# to CPU would still load, still generate, and still read as PASS - so "generated
# on the GPU" needs its own evidence rather than being inferred from "the vulkan
# archive was the one we placed".
try:
    # (name, type), in that order - read off compute_devices' own docstring, not
    # guessed. type is a raw ggml_backend_dev_type whose members have MOVED
    # between runtimes, so it is compared against _loader's constants rather than
    # a literal.
    out["devices"] = [{"name": str(n), "gpu": int(t) == _loader.GGML_DEV_TYPE_GPU}
                      for n, t in (_loader.compute_devices() or [])]
except Exception as e:
    out["devices"] = "unavailable: %s" % e

# 3. GENERATE. This is the bar "it loads so it will be fine" failed.
try:
    from localm.inference.backends.gguf import GgufBackend
except Exception as e:
    emit(verdict="INCONCLUSIVE", why="GgufBackend is not importable: %s" % e)

be = GgufBackend(model, n_ctx=2048)
try:
    be.load()
except Exception as e:
    emit(verdict="FAIL", why="a real GGUF failed to load on this build: %s" % e)

# The worker is a SEPARATE PROCESS. The override proven above was proven for THIS
# process only, so read back which llama library the worker actually mapped.
try:
    import psutil
    pid = be._runner._proc.pid
    name = os.environ["LOCALM_CONFIRM_LIB_NAME"].lower()
    mapped = [m.path for m in psutil.Process(pid).memory_maps()
              if Path(m.path).name.lower() == name]
    out["worker_lib"] = mapped
    if not mapped:
        be.unload()
        emit(verdict="INCONCLUSIVE",
             why="could not read which %s the worker process mapped" % name)
    if not all(Path(p).resolve().parent == expected for p in mapped):
        be.unload()
        emit(verdict="INCONCLUSIVE",
             why="the worker mapped a DIFFERENT %s (%r) - the generation below "
                 "would not have been this build's" % (name, mapped))
except SystemExit:
    raise
except Exception as e:
    be.unload()
    emit(verdict="INCONCLUSIVE", why="could not inspect the worker process: %s" % e)

# Read WHILE LOADED. This is the layer count the load RESOLVED, which is a
# REQUEST rather than a measurement: a cpu-only build reports 99 here just as a
# vulkan one does, measured. So it is recorded for context and the DEVICE LIST
# above is the discriminator - a build that registered no GPU device did not use
# one, whatever this number says.
out["gpu_layers"] = getattr(be, "effective_gpu_layers", None)

try:
    text = "".join(be.chat_stream(
        [{"role": "user", "content": "In one short sentence, what color is a clear daytime sky?"}],
        max_tokens=40, temperature=0.0, seed=1)).strip()
finally:
    try:
        be.unload()
    except Exception:
        pass

out["sample"] = text[:200]
if len(text) < 10 or not any(c.isalpha() for c in text):
    emit(verdict="FAIL", why="generated no usable text: %r" % text[:200])
emit(verdict="PASS", why="loaded and generated")
'''


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_model() -> "tuple[str, str]":
    """(path, error). Never raises: no model is INCONCLUSIVE, not a failure of
    the build under test."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        return "", f"huggingface_hub is not installed ({e})"
    try:
        return hf_hub_download(repo_id=_MODEL_REPO, filename=_MODEL_FILE), ""
    except Exception as e:
        return "", f"could not fetch {_MODEL_REPO}/{_MODEL_FILE}: {e}"


def _place_build(backend: str, tag: str, dest: Path, attempts: int = 3
                 ) -> "tuple[str, str]":
    """Download and extract *backend* at *tag* into *dest*. Returns (lib_path,
    error).

    Retried, because the release CDN drops connections ("Remote end closed
    connection without response" after 0 bytes). A transient drop and a build we
    could not test are the same INCONCLUSIVE outcome to the caller, so without a
    retry the common case would be a run that measured nothing."""
    from localm import setup_llama as sl
    try:
        url, sha, _tag = sl._resolve_backend_asset(backend, tag=tag)
    except Exception as e:
        return "", f"could not resolve the {backend} asset for {tag}: {e}"
    dest.mkdir(parents=True, exist_ok=True)
    last = ""
    for attempt in range(1, attempts + 1):
        try:
            sl._fetch_and_place(url, dest, sha)
            last = ""
            break
        except Exception as e:
            last = str(e)
            if attempt < attempts:
                print(f"  download attempt {attempt}/{attempts} failed ({e}); "
                      "retrying", flush=True)
    if last:
        return "", f"could not fetch {url} after {attempts} attempts: {last}"
    lib = dest / sl._lib_name()
    if not lib.exists():
        return "", f"the {backend} archive for {tag} contained no {sl._lib_name()}"
    return str(lib), ""


def _run_probe(lib: Path, model: str) -> dict:
    env = dict(os.environ)
    # Point at the library FILE, not its directory.
    env["LLAMA_CPP_LIB"] = str(lib)
    env["LOCALM_CONFIRM_EXPECT_DIR"] = str(lib.parent.resolve())
    env["LOCALM_CONFIRM_LIB_NAME"] = lib.name
    env["LOCALM_CONFIRM_MODEL"] = model
    try:
        r = subprocess.run([sys.executable, "-c", _PROBE], env=env,
                           capture_output=True, text=True, timeout=900)
    except Exception as e:
        return {"verdict": INCONCLUSIVE, "why": f"the probe could not be run: {e}"}
    for line in reversed((r.stdout or "").splitlines()):
        if line.startswith("@@VERDICT@@"):
            try:
                return json.loads(line[len("@@VERDICT@@"):])
            except Exception:
                break
    tail = "\n".join((r.stderr or r.stdout or "").strip().splitlines()[-6:])
    return {"verdict": INCONCLUSIVE,
            "why": f"the probe emitted no verdict (exit {r.returncode}): {tail}"}


def confirm(tag: str, backends: "list[str]", workdir: Path) -> dict:
    model, model_err = _fetch_model()
    results: dict = {}
    libs: dict = {}
    for backend in backends:
        print(f"\n=== {backend} @ {tag} ===", flush=True)
        # The directory name must contain no dots.
        dest = workdir / f"build-{backend}"
        lib, err = _place_build(backend, tag, dest)
        if err:
            results[backend] = {"verdict": INCONCLUSIVE, "why": err}
            print(f"  {INCONCLUSIVE}: {err}", flush=True)
            continue
        libs[backend] = _sha256(Path(lib))
        print(f"  placed {lib}  sha256[:16]={libs[backend][:16]}", flush=True)
        if not model:
            results[backend] = {"verdict": INCONCLUSIVE, "why": model_err}
            print(f"  {INCONCLUSIVE}: {model_err}", flush=True)
            continue
        res = _run_probe(Path(lib), model)
        results[backend] = res
        print(f"  {res.get('verdict')}: {res.get('why')}", flush=True)
        if res.get("sample"):
            print(f"  generated: {res['sample']!r}", flush=True)
    return {"tag": tag, "results": results, "lib_sha256": libs}


def _report(summary: dict, backends: "list[str]") -> int:
    tag, results, libs = summary["tag"], summary["results"], summary["lib_sha256"]
    print("\n" + "=" * 72)
    print(f"CONFIRMATION SUMMARY for llama.cpp {tag}")
    print("=" * 72)
    for backend in backends:
        r = results.get(backend, {})
        print(f"  {backend:10s} {r.get('verdict', INCONCLUSIVE):13s} "
              f"abi={r.get('abi')} backends={r.get('compute_backends')}")
        devices = r.get("devices")
        if isinstance(devices, list):
            gpus = [d["name"] for d in devices if d.get("gpu")]
            print(f"             devices={[d['name'] for d in devices]} "
                  f"gpu={gpus or 'NONE'} offloaded_layers={r.get('gpu_layers')}")
        elif devices:
            print(f"             devices: {devices}")
        if r.get("verdict") != PASS:
            print(f"             {r.get('why', '')}")

    if len(libs) > 1:
        distinct = set(libs.values())
        print("\n  llama library across the fetched backends: "
              + ("BYTE-IDENTICAL" if len(distinct) == 1
                 else f"DIFFERS ({len(distinct)} distinct digests)"))
        for backend, digest in libs.items():
            print(f"    {backend:10s} {digest[:16]}")
        if len(distinct) != 1:
            print("    ONE pin constant assumes this library is the same file for "
                  "every backend of a tag. It is not, here. Re-examine that "
                  "assumption before pinning.")
    elif libs:
        print("\n  Only one backend was fetched, so the byte-identity of the "
              "llama library across backends was NOT re-measured at this tag.")

    untested = sorted(set(_ALL_BACKENDS) - set(backends))
    print("\n  NOT TESTED HERE (generation is backend- and hardware-specific): "
          + (", ".join(untested) or "none"))
    print("  Their ABI compatibility is covered by the byte-identity above; their "
          "GENERATION is not covered by this run and must not be reported as "
          "confirmed. See setup_llama._PIN_CONFIRMATION.")

    verdicts = [results.get(b, {}).get("verdict", INCONCLUSIVE) for b in backends]
    if FAIL in verdicts:
        print(f"\nRESULT: {tag} is NOT confirmed - at least one backend FAILED.")
        return 1
    if INCONCLUSIVE in verdicts:
        print(f"\nRESULT: {tag} is NOT confirmed - at least one backend could not "
              "be measured. Nothing here says the build is bad; it says we did "
              "not test it.")
        return 2
    print(f"\nRESULT: {tag} loaded and GENERATED on: {', '.join(backends)}.")
    return 0


_ALL_BACKENDS = ("cpu", "vulkan", "cuda", "sycl", "hip", "metal")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default=None,
                    help="the llama.cpp release to confirm (default: the pin in "
                         "setup_llama._PINNED_TAG)")
    ap.add_argument("--backend", action="append", dest="backends", default=None,
                    choices=list(_ALL_BACKENDS),
                    help="repeatable; default 'cpu', the only backend that needs "
                         "no GPU and so runs anywhere")
    ap.add_argument("--keep", action="store_true",
                    help="keep the downloaded builds instead of deleting them")
    ap.add_argument("--workdir", default=None,
                    help="where to place the throwaway builds (default: a temp dir)")
    args = ap.parse_args(argv)

    try:
        from localm import setup_llama as sl
    except Exception as e:
        print(f"localm is not importable ({e}) - this script needs it installed.")
        return 2
    tag = args.tag or sl._PINNED_TAG
    backends = args.backends or ["cpu"]

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(
        prefix="localm-confirm-"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"Confirming llama.cpp {tag} on: {', '.join(backends)}")
    print(f"Throwaway build dir: {workdir}")
    try:
        summary = confirm(tag, backends, workdir)
        return _report(summary, backends)
    finally:
        if args.keep:
            print(f"\nKept: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
