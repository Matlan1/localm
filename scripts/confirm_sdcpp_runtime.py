#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Confirm a stable-diffusion.cpp release works with localm before it is pinned.

``localm/media/sdcpp/pins.py`` names one release, its commit and the sha256 of every
archive localm may install. This script earns the word "confirmed" for a candidate
release (``--tag``) or for the build pinned today (``--current``).

WHAT IT CHECKS (each recorded in the receipt with status PASS / FAIL / SKIP and a
``required`` flag; every check is required except ``generate_agree``):

  isolation        every localm path resolves inside the work directory, and localm
                   was imported from this checkout
  release_assets   the GitHub release lists a size and sha256 for every archive the
                   pins table needs, and resolves to the commit its tag names
  header_layout    the C structs in include/stable-diffusion.h at the candidate
                   commit equal the structs localm's ctypes binding declares, field
                   for field. A difference is a FAIL: "binding needs a code update,
                   not an automatic bump"
  download_<b>     the archive downloads through localm's own installer
                   (runtime.install) into a throwaway runtime directory; the
                   installer verifies the sha256, this script checks the size
  abi_<b>          the library loads in localm's isolated worker and the binding's
                   own verify_abi passes for the candidate commit
  device_<b>       the runtime registers a compute device for that backend (any
                   device for cpu, a non-CPU device for a GPU backend)
  generate_<b>     one small image is generated through localm's native image
                   backend (SD-Turbo, 256x256, the product's own step count) and is
                   checked for size, contrast and saturation
  generate_agree   not required: how far the images of two backends differ

<b> is each backend run: cpu always, plus vulkan and the machine's recommended
backend when hardware detection finds a GPU.

HOW THE CANDIDATE REACHES THE PRODUCT CODE. The binding checks the library against
``pins.COMMIT`` inside a spawned worker process. The candidate's tag, commit and
archive table are written to a file and applied to ``localm.media.sdcpp.pins`` by a
``sitecustomize`` module placed first on PYTHONPATH, so the driver process and the
worker see the same values. Nothing in localm is relaxed: ``verify_abi`` runs
unchanged against the candidate commit. The driver proves the values took effect, and
the abi check reports INCONCLUSIVE if the worker reports binding a different commit.

ISOLATION. LOCALM_HOME, TEMP, TMP, TMPDIR and HF_HOME point inside ``--workdir``
before localm is imported anywhere. Each backend runs in its own child process whose
PID is recorded; the process tree is killed with ``taskkill /PID <pid> /T /F`` (POSIX:
the recorded tree) in a ``finally``. Nothing outside ``--workdir`` is deleted. The
GPU lease is the caller's: wrap this command in ``gpu_lease.py run``.

MODEL. SD-Turbo (localm's recommended native image model, 2 GB) is downloaded once
into the persistent pin cache, its sha256 is compared with the one in
``native.RECOMMENDED_MODELS``, and it is reused afterwards. Without it (offline, no
disk) generate_<b> is SKIP and, being required, the verdict is INCONCLUSIVE.

VERDICT / EXIT CODE. PASS (0) only if every required check is PASS. FAIL (1) if any
required check failed (the build is bad for localm). INCONCLUSIVE (2) if nothing
failed but a required check could not run (network, hardware, model), or if the
script itself raised.

NOT MEASURED, written into the receipt: video generation, img2img and LoRA, the
backends not run on this machine.

Usage:
    python scripts/confirm_sdcpp_runtime.py --tag master-952-abcdef0 --workdir W --receipt r.json
    python scripts/confirm_sdcpp_runtime.py --current --workdir W --receipt r.json

Needs localm importable. Nothing under localm/ imports this; it never runs from a
user's install.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUMP_PATH = REPO / "scripts" / "bump_sdcpp_pin.py"
PINS_PATH = REPO / "localm" / "media" / "sdcpp" / "pins.py"
sys.path.insert(0, str(REPO))

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
V_PASS, V_FAIL, V_INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"
SCHEMA = 1
COMPONENT = "sdcpp"

HEADER_URL = ("https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/%s/"
              "include/stable-diffusion.h")
PINS_ENV = "LOCALM_CONFIRM_SDCPP_PINS"
CHILD_TIMEOUT = 2400
PROMPT = "a red apple on a wooden table"
SEED = 1234
SIZE = 256
INSTALL_ATTEMPTS = 3

NOT_MEASURED = ["video generation", "image-to-image and LoRA", "tiled VAE and hires paths",
                "models other than SD-Turbo"]

# Installed as sitecustomize.py first on PYTHONPATH in every process the driver starts,
# the spawned sd.cpp worker included.
SHIM_SOURCE = '''\
import json, os, sys

_path = os.environ.get("LOCALM_CONFIRM_SDCPP_PINS")
if _path:
    import importlib.abc
    import importlib.machinery

    _TARGET = "localm.media.sdcpp.pins"

    def _apply(module):
        with open(_path, encoding="utf-8") as fh:
            data = json.load(fh)
        module.TAG = data["tag"]
        module.COMMIT = data["commit"]
        module.ASSETS = {(p, b): (n, s) for p, b, n, s in data["assets"]}
        module.EXTRA_ASSETS = {}
        for p, b, n, s in data["extra"]:
            module.EXTRA_ASSETS.setdefault((p, b), []).append((n, s))
        module._BASE_URL = "https://github.com/%s/releases/download/%s/" % (
            module.REPO, module.TAG)
        module.LOCALM_CONFIRM_OVERRIDE = True

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return spec
            real = spec.loader

            class _Loader(importlib.abc.Loader):
                def create_module(self, spec):
                    return real.create_module(spec)

                def exec_module(self, module):
                    real.exec_module(module)
                    _apply(module)

            spec.loader = _Loader()
            return spec

    sys.meta_path.insert(0, _Finder())
'''


class Inconclusive(Exception):
    """A measurement could not be made; the message says why."""


def _bump():
    """The sibling bump script, loaded by path (it owns the release lookup and the
    asset-name rules both scripts share)."""
    spec = importlib.util.spec_from_file_location("bump_sdcpp_pin", BUMP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
#  Receipt                                                                     #
# --------------------------------------------------------------------------- #

def set_check(checks: dict, name: str, status: str, detail: str, required: bool = True,
              **extra) -> None:
    checks[name] = {"status": status, "required": required, "detail": detail, **extra}


def decide(checks: dict, mandatory=()) -> tuple[str, str]:
    """(verdict, why) for *checks*. A mandatory check that is absent counts as not
    measured."""
    failed = [n for n, c in checks.items() if c["required"] and c["status"] == FAIL]
    if failed:
        return V_FAIL, "; ".join(f"{n}: {checks[n]['detail']}" for n in failed)
    unmeasured = [n for n, c in checks.items() if c["required"] and c["status"] != PASS]
    unmeasured += [n for n in mandatory if n not in checks]
    if unmeasured:
        return V_INCONCLUSIVE, "not measured: " + "; ".join(
            f"{n}: {checks[n]['detail']}" if n in checks else f"{n}: never ran"
            for n in unmeasured)
    return V_PASS, "every required check passed"


def exit_code(verdict: str) -> int:
    return {V_PASS: 0, V_FAIL: 1}.get(verdict, 2)


def write_receipt(path: Path, receipt: dict) -> None:
    """Write *receipt* to *path* atomically (temp file in the same directory, then
    replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
#  The C header against the ctypes binding                                     #
# --------------------------------------------------------------------------- #

_STRUCT_RE = re.compile(r"typedef\s+struct\s*\{(?P<body>[^{}]*)\}\s*(?P<name>\w+)\s*;")
_SCALARS = {"bool": "bool", "int": "int", "float": "float", "uint32_t": "u32",
            "uint64_t": "u64", "size_t": "u64", "int64_t": "i64", "uint8_t": "u8",
            "void": "void"}


def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def _c_kind(ctype: str) -> str:
    """The comparable kind of C type text *ctype* (``"const char*"`` -> ``"str"``)."""
    t = " ".join(ctype.replace("const ", " ").split())
    if t == "char*":
        return "str"
    if t.endswith("*"):
        inner = _c_kind(t[:-1].strip())
        return "voidp" if inner == "void" else f"ptr:{inner}"
    if t.startswith("enum "):
        return "int"
    return _SCALARS.get(t, f"struct:{t}")


def parse_header_structs(source: str) -> dict:
    """``{struct name: [(kind, field name), ...]}`` for every ``typedef struct {...}
    name;`` in C header text *source*. A member that does not parse as ``type name``
    is recorded as ``("unparsed", text)``."""
    out = {}
    for m in _STRUCT_RE.finditer(_strip_comments(source)):
        fields = []
        for stmt in m.group("body").split(";"):
            stmt = " ".join(stmt.split())
            if not stmt:
                continue
            nm = re.search(r"(\w+)\s*$", stmt)
            if nm is None or nm.start() == 0:
                fields.append(("unparsed", stmt))
                continue
            fields.append((_c_kind(stmt[:nm.start()].strip()), nm.group(1)))
        out[m.group("name")] = fields
    return out


def _py_kind(ct) -> str:
    import ctypes
    table = {ctypes.c_bool: "bool", ctypes.c_int: "int", ctypes.c_float: "float",
             ctypes.c_uint32: "u32", ctypes.c_uint64: "u64", ctypes.c_size_t: "u64",
             ctypes.c_int64: "i64", ctypes.c_uint8: "u8", ctypes.c_char_p: "str",
             ctypes.c_void_p: "voidp"}
    if ct in table:
        return table[ct]
    if issubclass(ct, ctypes.Structure):
        return f"struct:{ct.__name__}"
    if issubclass(ct, ctypes._Pointer):
        return f"ptr:{_py_kind(ct._type_)}"
    return f"other:{ct.__name__}"


def binding_structs(binding) -> dict:
    """``{struct name: [(kind, field name), ...]}`` for every ctypes Structure
    declared in module *binding*."""
    import ctypes
    out = {}
    for name, obj in vars(binding).items():
        if isinstance(obj, type) and issubclass(obj, ctypes.Structure) \
                and obj.__module__ == binding.__name__:
            out[name] = [(_py_kind(t), f) for f, t in obj._fields_]
    return out


def compare_layout(header: dict, bound: dict) -> list[str]:
    """What differs between the header's structs and the binding's, one line per
    difference. Empty when every bound struct matches field for field."""
    diffs = []
    for name, fields in sorted(bound.items()):
        want = header.get(name)
        if want is None:
            diffs.append(f"{name}: not declared in the header")
            continue
        if want == fields:
            continue
        have = [f for _k, f in fields]
        got = [f for _k, f in want]
        added = [f for f in got if f not in have]
        removed = [f for f in have if f not in got]
        if added or removed:
            diffs.append(f"{name}: header adds {added or 'nothing'}, drops "
                         f"{removed or 'nothing'} against the binding")
        elif have != got:
            diffs.append(f"{name}: fields are reordered")
        else:
            changed = [f"{f} ({k0} in the binding, {k1} in the header)"
                       for (k0, f), (k1, _f) in zip(fields, want, strict=True) if k0 != k1]
            diffs.append(f"{name}: field types differ: " + ", ".join(changed))
    return diffs


def fetch_header(commit: str, opener=None) -> str:
    """The text of include/stable-diffusion.h at *commit*. Raises Inconclusive."""
    if opener is None:
        from localm.http_ssl import verified_urlopen
        opener = verified_urlopen
    import urllib.request
    req = urllib.request.Request(HEADER_URL % commit,
                                 headers={"User-Agent": "localm-confirm-sdcpp"})
    try:
        with opener(req, timeout=60) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        raise Inconclusive(f"could not read the header at {commit}: "
                           f"{type(e).__name__}: {e}") from e


def header_check(commit: str, opener=None, binding=None) -> tuple[str, str]:
    """(status, detail) of the header_layout check for *commit*."""
    try:
        text = fetch_header(commit, opener)
    except Inconclusive as e:
        return SKIP, str(e)
    if binding is None:
        from localm.media.sdcpp import _binding as binding
    header = parse_header_structs(text)
    bound = binding_structs(binding)
    if not bound:
        return SKIP, "the binding declares no structs; nothing to compare"
    diffs = compare_layout(header, bound)
    if diffs:
        return FAIL, ("binding needs a code update, not an automatic bump: the C structs "
                      "at this commit differ from localm's binding: " + "; ".join(diffs))
    return PASS, (f"{len(bound)} struct(s) of the binding match include/stable-diffusion.h "
                  f"at {commit[:7]} field for field")


# --------------------------------------------------------------------------- #
#  The candidate                                                               #
# --------------------------------------------------------------------------- #

def pins_payload(tag: str, commit: str, assets: dict, extra: dict) -> dict:
    """The JSON the sitecustomize shim applies to localm.media.sdcpp.pins."""
    return {"tag": tag, "commit": commit,
            "assets": [[p, b, n, s] for (p, b), (n, s) in assets.items()],
            "extra": [[p, b, n, s] for (p, b), entries in extra.items() for n, s in entries]}


def resolve_candidate(tag: str | None, current: bool, opener=None) -> dict:
    """The release to confirm: ``{"tag", "commit", "assets": {key: (name, sha)},
    "extra": {key: [(name, sha)]}, "listing": {name: {size, sha256}}}``.

    For ``current`` the tables come from pins.py and the release listing is only used
    to cross-check them. Raises ``Inconclusive`` when the listing cannot be read or
    is incomplete and ``ValueError`` when the pinned table disagrees with it or the
    release names are ambiguous."""
    bump = _bump()
    pins = bump.read_pins(PINS_PATH.read_text(encoding="utf-8"))
    try:
        if current:
            release = bump.fetch_release(pins["tag"], opener)
            if release["commit"] != pins["commit"]:
                raise ValueError(f"pins.py says commit {pins['commit']} but "
                                 f"{pins['tag']} resolves to {release['commit']}")
            drift = []
            tables = [(k, v) for k, v in pins["assets"].items()]
            for entries in pins["extra"].values():
                tables += [(None, e) for e in entries]
            for _key, (name, sha) in tables:
                listed = release["assets"].get(name)
                if listed is None:
                    drift.append(f"{name} is not in the release")
                elif listed["sha256"] != sha:
                    drift.append(f"{name}: pinned sha256 differs from the release digest")
            if drift:
                raise ValueError("the pinned table disagrees with the release: "
                                 + "; ".join(drift))
            return {"tag": pins["tag"], "commit": pins["commit"], "assets": pins["assets"],
                    "extra": pins["extra"], "listing": release["assets"]}
        release = bump.fetch_release(tag, opener)
        assets, extra = bump.build_tables(release, pins["assets"], pins["extra"])
        return {"tag": tag, "commit": release["commit"], "assets": assets, "extra": extra,
                "listing": release["assets"]}
    except (bump.UpstreamUnreadable, bump.IncompleteRelease) as e:
        raise Inconclusive(str(e)) from e
    except bump.Refused as e:
        raise ValueError(str(e)) from e


def receipt_candidate(cand: dict) -> dict:
    """The part of the receipt that records exactly which archives were confirmed."""
    names = {n for n, _s in cand["assets"].values()}
    names |= {n for entries in cand["extra"].values() for n, _s in entries}
    return {"tag": cand["tag"], "commit": cand["commit"],
            "assets": {n: dict(cand["listing"][n]) for n in sorted(names)}}


# --------------------------------------------------------------------------- #
#  Hardware and backends                                                       #
# --------------------------------------------------------------------------- #

def choose_backends(available: list, gpu_state: str, recommended: str) -> list:
    """The backends to confirm: cpu when upstream ships it, and when a GPU was found
    vulkan and *recommended* if each is available."""
    out = [b for b in ("cpu",) if b in available]
    if gpu_state == "found":
        for b in ("vulkan", recommended):
            if b in available and b not in out:
                out.append(b)
    return out


# --------------------------------------------------------------------------- #
#  Model                                                                       #
# --------------------------------------------------------------------------- #

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def default_cache_dir() -> Path:
    """The persistent model cache: ``LOCALM_PIN_CACHE``, else ``.claude/pin-cache/sdcpp``
    beside the main checkout's parent when that ``.claude`` exists, else a directory in
    the system temp dir."""
    env = os.environ.get("LOCALM_PIN_CACHE")
    if env:
        return Path(env) / "sdcpp"
    main = REPO.parents[2] if REPO.parent.name == "worktrees" else REPO
    shared = main.parent / ".claude"
    if shared.is_dir():
        return shared / "pin-cache" / "sdcpp"
    return Path(tempfile.gettempdir()) / "localm-pin-cache" / "sdcpp"


def ensure_model(cache_dir: Path) -> dict:
    """The cached SD-Turbo file, downloaded once and verified against the sha256 in
    ``native.RECOMMENDED_MODELS``. Returns ``{"path", "name", "repo", "file", "size",
    "sha256", "downloaded"}``. Raises Inconclusive when it cannot be obtained or does
    not verify."""
    from localm.plugins.builtin.image.backends.native import RECOMMENDED_MODELS
    rec = RECOMMENDED_MODELS[0]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / rec.file
    downloaded = False
    if not (target.is_file() and target.stat().st_size == rec.size_bytes):
        try:
            from huggingface_hub import hf_hub_download
        except Exception as e:
            raise Inconclusive(f"huggingface_hub is not installed ({e})") from e
        part = cache_dir / f".partial-{os.getpid()}-{int(time.time())}"
        try:
            got = hf_hub_download(repo_id=rec.repo, filename=rec.file, local_dir=str(part))
            os.replace(got, target)
        except Exception as e:
            raise Inconclusive(f"could not fetch {rec.repo}/{rec.file}: "
                               f"{type(e).__name__}: {e}") from e
        finally:
            shutil.rmtree(part, ignore_errors=True)
        downloaded = True
    digest = _sha256(target)
    if digest != rec.sha256 or target.stat().st_size != rec.size_bytes:
        raise Inconclusive(f"{target.name} in the cache does not match the sha256 "
                           f"localm lists for {rec.name}; delete it and rerun")
    with open(target, "rb") as f:
        if f.read(4) != b"GGUF":
            raise Inconclusive(f"{target.name} is not a GGUF file")
    return {"path": str(target), "name": rec.name, "repo": rec.repo, "file": rec.file,
            "size": rec.size_bytes, "sha256": digest, "downloaded": downloaded}


# --------------------------------------------------------------------------- #
#  The child: one backend, inside the isolated environment                     #
# --------------------------------------------------------------------------- #

def image_stats(png: Path) -> dict:
    """Size and pixel statistics of the PNG at *png*."""
    from PIL import Image, ImageStat
    with Image.open(png) as im:
        rgb = im.convert("RGB")
        stat = ImageStat.Stat(rgb)
        raw = rgb.tobytes()
        px = list(zip(raw[0::3], raw[1::3], raw[2::3], strict=True))
    n = len(px)
    extreme = sum(1 for r, g, b in px if min(r, g, b) <= 2 or max(r, g, b) >= 253)
    return {"width": rgb.width, "height": rgb.height,
            "stddev": round(sum(stat.stddev) / 3, 2), "mean": round(sum(stat.mean) / 3, 2),
            "extreme_fraction": round(extreme / n, 4), "colors": len(set(px))}


def judge_image(stats: dict, size: int) -> tuple[bool, str]:
    """(ok, why) for image *stats*: the requested size, enough contrast to be more
    than a flat fill, and not mostly saturated."""
    if (stats["width"], stats["height"]) != (size, size):
        return False, f"image is {stats['width']}x{stats['height']}, expected {size}x{size}"
    if stats["stddev"] < 8.0 or stats["colors"] < 256:
        return False, (f"image is nearly flat (stddev {stats['stddev']}, "
                       f"{stats['colors']} colors)")
    if stats["extreme_fraction"] > 0.5:
        return False, f"{stats['extreme_fraction']:.0%} of pixels are saturated black or white"
    return True, (f"{stats['width']}x{stats['height']}, stddev {stats['stddev']}, "
                  f"{stats['colors']} colors, {stats['extreme_fraction']:.1%} saturated")


def _classify_download_error(message: str) -> str:
    if "sha256 does not match" in message:
        return "sha"
    if any(s in message for s in ("stalled", "failed after", "network policy")):
        return "network"
    return "archive"


def _child_work(spec: dict, res: dict) -> None:
    import localm
    from localm.config import home_dir
    from localm.media.sdcpp import pins, runtime
    backend = spec["backend"]
    home = Path(spec["home"]).resolve()
    inside = lambda p: home.parent in Path(p).resolve().parents  # noqa: E731
    problems = []
    if not inside(home_dir()):
        problems.append(f"home_dir() is {home_dir()}")
    if not inside(tempfile.gettempdir()):
        problems.append(f"the temp dir is {tempfile.gettempdir()}")
    if REPO not in Path(localm.__file__).resolve().parents:
        problems.append(f"localm was imported from {localm.__file__}")
    if pins.TAG != spec["tag"] or pins.COMMIT != spec["commit"]:
        problems.append(f"pins say {pins.TAG}/{pins.COMMIT[:7]}, expected "
                        f"{spec['tag']}/{spec['commit'][:7]}")
    if spec["override"] and not getattr(pins, "LOCALM_CONFIRM_OVERRIDE", False):
        problems.append("the candidate pins were not applied")
    res["pins"] = {"tag": pins.TAG, "commit": pins.COMMIT}
    if problems:
        res["fatal"] = "isolation: " + "; ".join(problems)
        return

    downloads = []
    original = runtime._download

    def recording_download(url, dest, on_progress, label):
        original(url, dest, on_progress, label)
        downloads.append({"name": label, "size": Path(dest).stat().st_size})

    runtime._download = recording_download
    rec = {"probe": None}

    def recording_probe(rt):
        try:
            info = runtime._probe(rt)
        except Exception as e:
            rec["probe"] = {"ok": False, "error": str(e)}
            raise
        rec["probe"] = {"ok": True, "commit": info.get("commit"),
                        "devices": [list(d) for d in info.get("devices") or []]}
        return info

    def say(text):
        print(f"  [{backend}] {text}", flush=True)

    install = {"error": None, "kind": None}
    try:
        for attempt in range(1, INSTALL_ATTEMPTS + 1):
            downloads.clear()
            rec["probe"] = None
            install = {"error": None, "kind": None}
            try:
                runtime.install(backend, force=True, on_progress=say, probe=recording_probe)
                break
            except runtime.DownloadError as e:
                install = {"error": str(e), "kind": _classify_download_error(str(e))}
                say(f"download attempt {attempt}/{INSTALL_ATTEMPTS} failed: {e}")
            except runtime.ProvisionError as e:
                install = {"error": str(e), "kind": "provision"}
                break
    finally:
        runtime._download = original
    install["downloads"] = [dict(d) for d in downloads]
    res["install"] = install
    res["probe"] = rec["probe"]
    resolved = runtime.resolve(backend)
    res["resolved"] = resolved is not None
    probe = rec["probe"] or {}
    res["has_backend_device"] = bool(
        probe.get("ok") and runtime._has_backend_device(backend, probe.get("devices") or []))

    if not (probe.get("ok") and res["has_backend_device"] and resolved is not None):
        res["generate"] = {"status": "skip", "detail": "the runtime did not install and load"}
        return
    if not spec.get("model"):
        res["generate"] = {"status": "skip", "detail": spec.get("model_skip") or "no model"}
        return
    from localm.media.sdcpp import shared
    from localm.plugins.builtin.image.backends import native
    settings = {"native": {"model": spec["model"], "runtime": backend}}
    png = Path(spec["png"])
    gen = {"status": "fail", "detail": ""}
    try:
        started = time.monotonic()
        ok, message = native.generate(settings, PROMPT, png, write_sidecar=False, seed=SEED,
                                      width=SIZE, height=SIZE, on_progress=say)
        gen["seconds"] = round(time.monotonic() - started, 1)
        gen["worker_pid"] = shared.worker_pid()
        info = getattr(shared.runner, "load_info", None) or {}
        gen["model_version"] = info.get("version")
        if not ok:
            gen["detail"] = message
        elif not png.is_file():
            gen["detail"] = "generation reported success but wrote no image"
        else:
            stats = image_stats(png)
            good, why = judge_image(stats, SIZE)
            gen.update(status="ok" if good else "fail", detail=why, stats=stats,
                       png=str(png))
    finally:
        shared.free()
    gen["worker_stopped"] = shared.worker_pid() is None
    res["generate"] = gen


def child_main(spec_path: str, out_path: str) -> int:
    """Run one backend as described by the JSON at *spec_path* and write its result
    JSON to *out_path*."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    res = {"backend": spec["backend"], "fatal": None}
    try:
        _child_work(spec, res)
    except BaseException as e:  # noqa: BLE001
        res["fatal"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-1500:]
    write_receipt(Path(out_path), res)
    return 0


# --------------------------------------------------------------------------- #
#  The parent: process handling and turning child results into checks          #
# --------------------------------------------------------------------------- #

def kill_tree(pid: int) -> None:
    """Kill process *pid* and its descendants."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        return
    try:
        import psutil
        proc = psutil.Process(pid)
        for child in proc.children(recursive=True):
            try:
                child.kill()
            except psutil.Error:
                pass
        proc.kill()
    except Exception:  # noqa: BLE001
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:  # noqa: BLE001
        return False


def run_child(spec: dict, run_dir: Path, env: dict, timeout: float = CHILD_TIMEOUT) -> dict:
    """Run the child for ``spec["backend"]`` and return its result dict. A child that
    times out is killed with its tree; one that leaves no result yields a ``fatal``
    entry with the tail of its log."""
    backend = spec["backend"]
    spec_path = run_dir / f"spec-{backend}.json"
    out_path = run_dir / f"result-{backend}.json"
    log_path = run_dir / f"child-{backend}.log"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    cmd = [sys.executable, str(Path(__file__).resolve()), "--internal-run", str(spec_path),
           str(out_path)]
    pid = None
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                cwd=str(run_dir))
        pid = proc.pid
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            if proc.poll() is None:
                kill_tree(pid)
                proc.wait(timeout=30)
    tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace")
                     .strip().splitlines()[-8:])
    if timed_out:
        return {"backend": backend, "fatal": f"timed out after {timeout:.0f}s", "timeout": True,
                "log_tail": tail}
    try:
        res = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"backend": backend, "log_tail": tail,
                "fatal": f"the child exited {proc.returncode} without a result"}
    worker = (res.get("generate") or {}).get("worker_pid")
    if worker and _pid_alive(worker):
        kill_tree(worker)
        res["worker_killed_after"] = worker
    res["log_tail"] = tail
    return res


_LAYOUT_RE = re.compile(r"struct layout does not match")
_COMMIT_RE = re.compile(r"is commit (\S+), but localm binds commit (\S+)")


def checks_for_backend(backend: str, res: dict, expected_commit: str,
                       expected_sizes: dict) -> dict:
    """The download / abi / device / generate checks for *backend* from child result
    *res*. *expected_sizes* maps asset name to the size the release lists."""
    checks: dict = {}
    d, a, v, g = (f"download_{backend}", f"abi_{backend}", f"device_{backend}",
                  f"generate_{backend}")
    if res.get("fatal"):
        why = f"the {backend} child could not measure: {res['fatal']}"
        for name in (d, a, v, g):
            set_check(checks, name, SKIP, why)
        return checks

    inst = res.get("install") or {}
    kind, err = inst.get("kind"), inst.get("error") or ""
    got = {x["name"]: x["size"] for x in inst.get("downloads") or []}
    if kind == "network":
        set_check(checks, d, SKIP, f"the download could not complete: {err}")
    elif kind in ("sha", "archive"):
        set_check(checks, d, FAIL, f"the installer refused the archive: {err}")
    elif not got:
        set_check(checks, d, SKIP, "no archive was downloaded")
    else:
        wrong = [f"{n}: {s} bytes, release lists {expected_sizes.get(n)}"
                 for n, s in got.items() if expected_sizes.get(n) != s]
        if wrong:
            set_check(checks, d, FAIL, "size differs from the release listing: "
                      + "; ".join(wrong))
        else:
            set_check(checks, d, PASS,
                      f"{len(got)} archive(s) through runtime.install, sha256 verified by the "
                      "installer against the release digest, size equals the release listing: "
                      + ", ".join(f"{n} ({s} bytes)" for n, s in got.items()))

    probe = res.get("probe") or {}
    if checks[d]["status"] != PASS:
        set_check(checks, a, SKIP, "the archive was not installed")
    elif probe.get("ok"):
        lib = (probe.get("commit") or "").strip().lower()
        if len(lib) >= 7 and expected_commit.startswith(lib):
            set_check(checks, a, PASS, f"the worker loaded the library, verify_abi passed, "
                      f"sd_commit {lib} is the candidate commit")
        else:
            set_check(checks, a, FAIL, f"the library reports commit {lib or '(none)'}, "
                      f"expected {expected_commit[:7]}")
    else:
        msg = probe.get("error") or err or "no probe result"
        m = _COMMIT_RE.search(msg)
        if _LAYOUT_RE.search(msg):
            set_check(checks, a, FAIL,
                      f"binding needs a code update, not an automatic bump: {msg}")
        elif m and not expected_commit.startswith(m.group(2)):
            set_check(checks, a, SKIP, "the worker bound a different commit than the "
                      f"candidate, so the candidate pins were not applied there: {msg}")
        elif m:
            set_check(checks, a, FAIL, f"the release archive is not the commit its tag "
                      f"names: {msg}")
        elif "timed out" in msg:
            set_check(checks, a, SKIP, f"the worker timed out: {msg}")
        else:
            set_check(checks, a, FAIL, f"the runtime did not load in the worker: {msg}")

    if checks[a]["status"] != PASS:
        set_check(checks, v, SKIP, "the library did not load")
    elif res.get("has_backend_device") and res.get("resolved"):
        names = ", ".join(f"{n} ({desc})" for n, desc in probe.get("devices") or [])
        set_check(checks, v, PASS, f"devices registered: {names}; runtime.resolve finds the "
                  f"{backend} install")
    else:
        set_check(checks, v, FAIL, f"no {backend} compute device registered "
                  f"(devices: {probe.get('devices')}; resolved: {res.get('resolved')})")

    gen = res.get("generate") or {}
    if checks[v]["status"] != PASS:
        set_check(checks, g, SKIP, "the runtime did not load with a device")
    elif gen.get("status") == "ok":
        set_check(checks, g, PASS,
                  f"{gen['detail']}; {gen.get('seconds')} s through native.generate "
                  f"(model {gen.get('model_version')})", stats=gen.get("stats"))
    elif gen.get("status") == "skip":
        set_check(checks, g, SKIP, f"generation was NOT measured: {gen.get('detail')}")
    else:
        set_check(checks, g, FAIL, f"generation failed: {gen.get('detail') or 'no detail'}")
    return checks


def image_difference(a: Path, b: Path) -> float:
    """Mean absolute per-channel difference of two equally sized PNGs, 0..1."""
    from PIL import Image, ImageChops
    with Image.open(a) as ia, Image.open(b) as ib:
        diff = ImageChops.difference(ia.convert("RGB"), ib.convert("RGB"))
        data = diff.tobytes()
        return sum(data) / (255 * len(data))


def agreement_check(pngs: dict) -> tuple[str, str]:
    """(status, detail) comparing the images of the backends in *pngs* (name -> path)
    to the first one."""
    names = list(pngs)
    if len(names) < 2:
        return SKIP, "only one backend produced an image; nothing to compare"
    base = names[0]
    parts = []
    status = PASS
    for other in names[1:]:
        d = image_difference(Path(pngs[base]), Path(pngs[other]))
        parts.append(f"{base} vs {other}: mean difference {d:.3f}")
        if d > 0.35:
            status = FAIL
    return status, "; ".join(parts)


def prepare_environment(run_dir: Path, base_env=None) -> dict:
    """The environment for children: LOCALM_HOME, TEMP, TMP, TMPDIR, HF_HOME inside
    *run_dir* and PYTHONPATH with the shim directory first, then this checkout."""
    env = dict(os.environ if base_env is None else base_env)
    home, tmp, hf, shim = (run_dir / "home", run_dir / "tmp", run_dir / "hf", run_dir / "shim")
    for d in (home, tmp, hf, shim):
        d.mkdir(parents=True, exist_ok=True)
    (shim / "sitecustomize.py").write_text(SHIM_SOURCE, encoding="utf-8")
    env.update(LOCALM_HOME=str(home), TEMP=str(tmp), TMP=str(tmp), TMPDIR=str(tmp),
               HF_HOME=str(hf), PYTHONUTF8="1")
    parts = [str(shim), str(REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env.pop(PINS_ENV, None)
    return env


def confirm(tag: str | None, current: bool, workdir: Path, *, keep: bool = False,
            backends=None, cache_dir: Path | None = None, opener=None,
            child_runner=None, model_provider=None, detector=None) -> dict:
    """Run every check and return the receipt dict (not yet written).

    *opener* serves the GitHub API and header reads, *child_runner* replaces the
    per-backend child process, *model_provider* the model cache and *detector* the
    hardware detection."""
    bump = _bump()
    run_dir = Path(workdir).resolve() / f"run-{os.getpid()}"
    checks: dict = {}
    receipt = {"schema": SCHEMA, "component": COMPONENT, "tag": tag, "current": current,
               "verdict": V_INCONCLUSIVE, "why": "the run did not finish", "written_at": _now(),
               "hardware": {}, "candidate": None, "model": None,
               "not_measured": list(NOT_MEASURED), "checks": checks}
    if current:
        receipt["tag"] = bump.read_pins(PINS_PATH.read_text(encoding="utf-8"))["tag"]
    run_dir.mkdir(parents=True, exist_ok=True)
    scoped = ("LOCALM_HOME", "TEMP", "TMP", "TMPDIR", "HF_HOME")
    saved = {k: os.environ.get(k) for k in scoped}
    try:
        env = prepare_environment(run_dir)
        os.environ.update({k: env[k] for k in scoped})
        from localm import hwdetect
        from localm.config import home_dir
        from localm.media.sdcpp import runtime
        problems = []
        if run_dir not in Path(home_dir()).resolve().parents:
            problems.append(f"home_dir() is {home_dir()}")
        import localm
        if REPO not in Path(localm.__file__).resolve().parents:
            problems.append(f"localm was imported from {localm.__file__}")
        if problems:
            set_check(checks, "isolation", SKIP, "; ".join(problems))
            return _finish(receipt)
        set_check(checks, "isolation", PASS,
                  f"LOCALM_HOME, TEMP and HF_HOME are inside {run_dir.name}; localm is "
                  "imported from this checkout")

        try:
            cand = resolve_candidate(tag, current, opener)
        except Inconclusive as e:
            set_check(checks, "release_assets", SKIP, str(e))
            return _finish(receipt)
        except ValueError as e:
            set_check(checks, "release_assets", FAIL, str(e))
            return _finish(receipt)
        receipt["tag"] = cand["tag"]
        receipt["candidate"] = receipt_candidate(cand)
        set_check(checks, "release_assets", PASS,
                  f"{cand['tag']} resolves to {cand['commit']}; "
                  f"{len(receipt['candidate']['assets'])} archive(s) with size and sha256")

        status, detail = header_check(cand["commit"], opener)
        set_check(checks, "header_layout", status, detail)

        plat = runtime.platform_key()
        det = (detector or hwdetect.detect)()
        recommended = runtime.recommended_backend(det) if plat else "cpu"
        run_backends = list(backends) if backends else choose_backends(
            runtime.available_backends(plat), det.gpu_state, recommended)
        receipt["hardware"] = {"platform": plat, "gpu": det.gpu_state == "found",
                               "gpu_state": det.gpu_state, "vendors": list(det.vendors),
                               "gpu_names": det.gpu_names, "recommended": recommended,
                               "backends": run_backends}
        if not run_backends:
            set_check(checks, "download_cpu", SKIP,
                      "upstream publishes no build for this platform")
            return _finish(receipt)

        model, model_skip = None, None
        try:
            model = (model_provider or ensure_model)(cache_dir or default_cache_dir())
            receipt["model"] = {k: v for k, v in model.items() if k != "path"}
        except Inconclusive as e:
            model_skip = f"the model is not available: {e}"

        override = not current
        payload = pins_payload(cand["tag"], cand["commit"], cand["assets"], cand["extra"])
        pins_file = run_dir / "candidate-pins.json"
        pins_file.write_text(json.dumps(payload), encoding="utf-8")
        child_env = dict(env)
        if override:
            child_env[PINS_ENV] = str(pins_file)
        sizes = {n: v["size"] for n, v in cand["listing"].items()}
        pngs = {}
        runner = child_runner or run_child
        for backend in run_backends:
            print(f"=== {backend} @ {cand['tag']} ===", flush=True)
            spec = {"backend": backend, "tag": cand["tag"], "commit": cand["commit"],
                    "override": override, "home": env["LOCALM_HOME"],
                    "model": model["path"] if model else None, "model_skip": model_skip,
                    "png": str(run_dir / f"out-{backend}.png")}
            res = runner(spec, run_dir, child_env)
            checks.update(checks_for_backend(backend, res, cand["commit"], sizes))
            for name in (f"download_{backend}", f"abi_{backend}", f"device_{backend}",
                         f"generate_{backend}"):
                print(f"  {name}: {checks[name]['status']} - {checks[name]['detail']}",
                      flush=True)
            if checks[f"generate_{backend}"]["status"] == PASS:
                pngs[backend] = spec["png"]
        if len(pngs) >= 2:
            status, detail = agreement_check(pngs)
            set_check(checks, "generate_agree", status, detail, required=False)
        return _finish(receipt)
    except Exception as e:  # noqa: BLE001
        receipt["why"] = f"the script raised {type(e).__name__}: {e}"
        receipt["traceback"] = traceback.format_exc()[-2000:]
        return receipt
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if keep:
            print(f"kept: {run_dir}", flush=True)
        else:
            shutil.rmtree(run_dir, ignore_errors=True)


def _finish(receipt: dict) -> dict:
    receipt["verdict"], receipt["why"] = decide(receipt["checks"], _bump().MANDATORY_CHECKS)
    receipt["written_at"] = _now()
    return receipt


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "--internal-run":
        return child_main(argv[1], argv[2])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--tag", help="the candidate release, e.g. master-952-abcdef0")
    which.add_argument("--current", action="store_true",
                       help="confirm the build pins.py names today")
    ap.add_argument("--workdir", required=True, help="scratch directory; everything runs in it")
    ap.add_argument("--receipt", required=True, help="write the receipt JSON here")
    ap.add_argument("--keep", action="store_true", help="keep the scratch run directory")
    ap.add_argument("--backend", action="append", default=None,
                    choices=["cpu", "vulkan", "cuda", "rocm", "metal"],
                    help="repeatable; default: cpu plus the GPU backends of this machine")
    ap.add_argument("--cache-dir", default=None, help="persistent model cache directory")
    args = ap.parse_args(argv)
    if args.tag:
        try:
            _bump().parse_tag(args.tag)
        except Exception as e:  # noqa: BLE001
            print(f"INCONCLUSIVE: {e}")
            return 2
    receipt = None
    try:
        receipt = confirm(args.tag, args.current, Path(args.workdir), keep=args.keep,
                          backends=args.backend,
                          cache_dir=Path(args.cache_dir) if args.cache_dir else None)
    except BaseException as e:  # noqa: BLE001
        receipt = {"schema": SCHEMA, "component": COMPONENT, "tag": args.tag,
                   "current": args.current, "verdict": V_INCONCLUSIVE,
                   "why": f"the script raised {type(e).__name__}: {e}", "written_at": _now(),
                   "hardware": {}, "candidate": None, "model": None,
                   "not_measured": list(NOT_MEASURED), "checks": {},
                   "traceback": traceback.format_exc()[-2000:]}
    write_receipt(Path(args.receipt), receipt)
    print(f"\n{receipt['verdict']}: {receipt['why']}")
    print(f"receipt: {args.receipt}")
    return exit_code(receipt["verdict"])


if __name__ == "__main__":
    sys.exit(main())
