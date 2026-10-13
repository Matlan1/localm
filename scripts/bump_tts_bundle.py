#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rebuild the vendored TTS bundle from pinned npm releases.

The TTS plugin ships localm/plugins/builtin/tts/static/vendor/kokoro.min.js, a
browser ESM bundle of kokoro-js, @huggingface/transformers (transformers.js),
onnxruntime-web and phonemizer, plus the onnxruntime runtime files under
vendor/onnxruntime/ that the bundle loads at run time. This script rebuilds all
of it with esbuild from exact npm versions and moves every pin that names the
old bytes. It is review-only: a person reads the diff and the printed checklist,
then merges.

HOW THE BUNDLE IS BUILT: npm installs kokoro-js, @huggingface/transformers and
phonemizer at exact versions (scripts disabled, transformers forced onto the
pinned version for kokoro-js too) with esbuild at the version NOTICE.md records.
esbuild bundles ENTRY with ESBUILD_ARGS, and one line is appended that exports
the two internal eSpeak NG handles, located by content because the minifier
renames them in every build. onnxruntime-web is whatever transformers.js pins.
The build is deterministic: the same four packages and esbuild version give the
same bytes, which ``--verify`` proves against the tree.

WHAT IT REWRITES (with ``--write``; without it, a summary and diff are printed
and nothing is touched):

  vendor/kokoro.min.js
  vendor/onnxruntime/ort-wasm-simd-threaded.jsep.{mjs,wasm}  (transformers' dist/)
  vendor/LICENSE.kokoro-js, LICENSE.transformers-js, LICENSE.onnxruntime-web
  vendor/NOTICE.md            the version table, the runtime provenance and sizes,
                              the onnxruntime licence source, the esbuild version
  tests-js/vendor-onnxruntime.test.mjs   version, both content hashes, wasm size

WHAT IT CHECKS BEFORE WRITING: the tag is a plain MAJOR.MINOR.PATCH release
strictly newer than the pinned one; the installed packages are exactly the
requested versions and onnxruntime-web is installed once; the bundle contains
the transformers.js version string, exports exactly the expected names and loads;
every onnxruntime runtime file the bundle names is in transformers' dist/ and the
set equals the vendored one; every edited region is found exactly once. Nothing
is written unless every check passes, and a write that fails part way is rolled
back.

``--verify`` rebuilds the CURRENT pins and compares them with the tree: exit 0
when every file is byte-identical, 1 when anything differs.

Exit codes: 0 applied, dry run completed or verified identical; 1 refused,
failed or not identical.

Usage:
    python scripts/bump_tts_bundle.py --verify
    python scripts/bump_tts_bundle.py --lib transformers.js --tag 3.9.0
    python scripts/bump_tts_bundle.py --lib kokoro-js --tag 1.3.0 --write
"""

import argparse
import base64
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENDOR_REL = "localm/plugins/builtin/tts/static/vendor"
NOTICE_REL = f"{VENDOR_REL}/NOTICE.md"
BUNDLE_REL = f"{VENDOR_REL}/kokoro.min.js"
RUNTIME_REL = f"{VENDOR_REL}/onnxruntime"
VOICES_REL = f"{VENDOR_REL}/voices.json"
TEST_REL = "tests-js/vendor-onnxruntime.test.mjs"
LICENSE_RELS = {
    "kokoro-js": f"{VENDOR_REL}/LICENSE.kokoro-js",
    "transformers": f"{VENDOR_REL}/LICENSE.transformers-js",
    "onnxruntime-web": f"{VENDOR_REL}/LICENSE.onnxruntime-web",
}

ENTRY = "export * from 'kokoro-js';\nexport { env } from '@huggingface/transformers';\n"
ESBUILD_ARGS = ("--bundle", "--minify", "--format=esm", "--platform=browser", "--legal-comments=eof")
BUNDLE_EXPORTS = {"KokoroTTS", "TextSplitterStream", "env", "espeakWorker", "espeakFS"}
LIBS = ("kokoro-js", "transformers.js")

ORT_REPO = "microsoft/onnxruntime"
GITHUB_API = "https://api.github.com"
GITHUB_RAW = "https://raw.githubusercontent.com"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_LICENSE_BYTES = 256 * 1024
BUILD_TIMEOUT_SECONDS = 900

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_ROW_RES = {
    "kokoro-js": re.compile(r"^(\| `kokoro-js` \| )([^ |]+)( \|)", re.M),
    "transformers": re.compile(
        r"^(\| `@huggingface/transformers` \(transformers\.js\) \| )([^ |]+)( \|)", re.M),
    "onnxruntime-web": re.compile(r"^(\| `onnxruntime-web` \| )([^ |]+)( \|)", re.M),
    "phonemizer": re.compile(r"^(\| `phonemizer` \| )([^ |]+)( \|)", re.M),
}
_ESBUILD_RE = re.compile(r"(`esbuild` )(\d+\.\d+\.\d+)( bundles)")
_COPIES_RE = re.compile(r"(Byte-exact copies of `dist/` from `@huggingface/transformers@)([^`]+)(`)")
_ENV_VERSION_RE = re.compile(r"(\(its `env\.version` string\nreads `)([^`]+)(`)")
_SIZE_RES = {
    "ort-wasm-simd-threaded.jsep.mjs": re.compile(
        r"(onnxruntime/ort-wasm-simd-threaded\.jsep\.mjs)( +)(\d+)( B)"),
    "ort-wasm-simd-threaded.jsep.wasm": re.compile(
        r"(onnxruntime/ort-wasm-simd-threaded\.jsep\.wasm)( +)(\d+)( B)"),
}
_ORT_SOURCE_RE = re.compile(
    r"microsoft/onnxruntime repository at (?:commit|tag) `[^`]+`,\n"
    r"which is the (?:exact commit the pinned dev version|release the pinned version) names\.")
_ORT_DEV_RE = re.compile(r"-dev\.\d+-([0-9a-f]{7,40})$")
_TEST_RES = {
    "version": re.compile(r'^(const VENDORED_TRANSFORMERS = ")([^"]+)(";)', re.M),
    "mjs_hash": re.compile(r'^(const PINNED_MJS_HASH = ")([^"]+)(";)', re.M),
    "wasm_hash": re.compile(r'^(const PINNED_WASM_HASH = ")([^"]+)(";)', re.M),
    "wasm_bytes": re.compile(r"^(const EXPECTED_WASM_BYTES = )(\d+)(;)", re.M),
    "pack": re.compile(r"(npm pack @huggingface/transformers@)([^ ]+)(  ->)"),
}
_WORKER_RE = re.compile(
    r"var ([\w$]+)=new Promise\(\(*([\w$]+)\)*=>\{([\w$]+)\.calledRun\?\2\(new \3\.eSpeakNGWorker\)")
_FS_RE = re.compile(r"[\w$]+\.FS_createDataFile=([\w$]+)\.createDataFile")
_RUNTIME_NAME_RE = re.compile(r"ort-wasm[A-Za-z0-9._-]*")


NEEDS_CODE = "needs a code update"


class Refused(Exception):
    """The bump cannot proceed; the message says why. A message carrying NEEDS_CODE tells
    the weekly run the release needs a change in localm's own code before it can be vendored."""


@dataclass
class Pins:
    """The versions the bundle is built from, as NOTICE.md records them."""
    kokoro: str
    transformers: str
    ort: str
    phonemizer: str
    esbuild: str


@dataclass
class Built:
    """Everything one build produced, before any of it reaches the tree."""
    bundle: bytes
    runtime: dict
    licenses: dict
    versions: dict
    kokoro_peer_range: str
    voices: set
    warnings: str = ""


@dataclass
class Change:
    rel: str
    path: Path
    old: bytes | None
    new: bytes
    text: bool
    newline: str = "\n"

    def on_disk(self) -> bytes:
        """The bytes this change writes, with the file's own newline style for text."""
        if self.text and self.newline == "\r\n":
            return self.new.replace(b"\n", b"\r\n")
        return self.new


# --------------------------------------------------------------------------- #
#  Versions                                                                   #
# --------------------------------------------------------------------------- #

def parse_version(text: str) -> tuple:
    m = _VERSION_RE.match(text)
    if not m:
        raise Refused(f"{text!r} is not a plain MAJOR.MINOR.PATCH release")
    return tuple(int(x) for x in m.groups())


def crosses_major(old: tuple, new: tuple) -> bool:
    """A new major, or for 0.x a new minor."""
    if old[0] == 0 and new[0] == 0:
        return old[1] != new[1]
    return old[0] != new[0]


# --------------------------------------------------------------------------- #
#  Reading the tree                                                           #
# --------------------------------------------------------------------------- #

def _read_text(path: Path) -> tuple:
    """(text with LF newlines, the newline sequence the file uses)."""
    data = path.read_bytes().decode("utf-8")
    return data.replace("\r\n", "\n"), ("\r\n" if "\r\n" in data else "\n")


def _lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def _replace_once(pattern: re.Pattern, text: str, repl, what: str) -> str:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise Refused(f"{what}: expected exactly one match, found {len(matches)}; "
                      "the file shape this script edits has changed")
    m = matches[0]
    return text[:m.start()] + repl(m) + text[m.end():]


def _read_once(pattern: re.Pattern, text: str, group: int, what: str) -> str:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise Refused(f"{what}: expected exactly one match, found {len(matches)}; "
                      "the file shape this script reads has changed")
    return matches[0].group(group)


def read_pins(repo: Path) -> Pins:
    notice, _ = _read_text(repo / NOTICE_REL)
    got = {k: _read_once(rx, notice, 2, f"NOTICE.md {k} row") for k, rx in _ROW_RES.items()}
    esbuild = _read_once(_ESBUILD_RE, notice, 2, "NOTICE.md esbuild version")
    return Pins(kokoro=got["kokoro-js"], transformers=got["transformers"],
                ort=got["onnxruntime-web"], phonemizer=got["phonemizer"], esbuild=esbuild)


def _sha256_b64(data: bytes) -> str:
    return base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")


# --------------------------------------------------------------------------- #
#  Upstream (injectable)                                                      #
# --------------------------------------------------------------------------- #

def _default_open(req, timeout):
    try:
        from localm.http_ssl import verified_urlopen
    except ImportError as e:
        raise Refused("localm is not importable, so there is no verified HTTPS opener; "
                      "run this with the repository on PYTHONPATH") from e
    return verified_urlopen(req, timeout=timeout)


def fetch_json(url: str, opener=None) -> object:
    opener = opener or _default_open
    headers = {"Accept": "application/json", "User-Agent": "localm-bump-tts-bundle"}
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith(GITHUB_API):
        headers["Authorization"] = f"Bearer {token}"
    try:
        with opener(urllib.request.Request(url, headers=headers), 30) as resp:
            raw = resp.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise ValueError("response is larger than the allowed size")
        return json.loads(raw.decode("utf-8"))
    except Refused:
        raise
    except Exception as e:
        raise Refused(f"could not read {url}: {type(e).__name__}: {e}") from e


def fetch_bytes(url: str, opener=None) -> bytes:
    opener = opener or _default_open
    req = urllib.request.Request(url, headers={"User-Agent": "localm-bump-tts-bundle"})
    try:
        with opener(req, 60) as resp:
            raw = resp.read(MAX_LICENSE_BYTES + 1)
    except Refused:
        raise
    except Exception as e:
        raise Refused(f"could not download {url}: {type(e).__name__}: {e}") from e
    if len(raw) > MAX_LICENSE_BYTES:
        raise Refused(f"{url} is larger than {MAX_LICENSE_BYTES} bytes")
    return raw


def ort_license_source(version: str, fetch_json_fn=fetch_json) -> tuple:
    """(kind, ref) naming the microsoft/onnxruntime revision *version* was built from:
    ("commit", full sha) for a dev build, ("tag", "vX.Y.Z") for a release."""
    m = _ORT_DEV_RE.search(version)
    if m:
        data = fetch_json_fn(f"{GITHUB_API}/repos/{ORT_REPO}/commits/{m.group(1)}")
        sha = data.get("sha") if isinstance(data, dict) else None
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise Refused(f"could not resolve onnxruntime commit {m.group(1)} to a full sha")
        return "commit", sha
    parse_version(version)
    return "tag", f"v{version}"


def fetch_ort_license(version: str, fetch_json_fn=fetch_json, fetch_bytes_fn=fetch_bytes) -> tuple:
    """(licence text bytes, kind, ref) from microsoft/onnxruntime at the revision of *version*."""
    kind, ref = ort_license_source(version, fetch_json_fn)
    return fetch_bytes_fn(f"{GITHUB_RAW}/{ORT_REPO}/{ref}/LICENSE"), kind, ref


# --------------------------------------------------------------------------- #
#  Building                                                                   #
# --------------------------------------------------------------------------- #

def _tool(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise Refused(f"{name} was not found on PATH; building the bundle needs Node.js and npm")
    return exe


def _run(cmd: list, cwd: Path, what: str) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=BUILD_TIMEOUT_SECONDS, encoding="utf-8", errors="replace")
    except (OSError, subprocess.TimeoutExpired) as e:
        raise Refused(f"{what} could not run: {type(e).__name__}: {e}") from e
    if proc.returncode != 0:
        raise Refused(f"{what} failed (exit {proc.returncode}): "
                      f"{(proc.stderr or proc.stdout).strip()[-1200:]}")
    return proc


def _installed(root: Path, package: str) -> dict:
    path = root / "node_modules" / package / "package.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"{package} is not installed where the build expects it: {e}") from e


def locate_espeak_exports(bundle: str) -> tuple:
    """(worker name, filesystem name) of the two eSpeak NG handles in a minified bundle,
    found by what they are: the promise that constructs ``eSpeakNGWorker`` and the object
    whose ``createDataFile`` the emscripten module re-exports."""
    workers = _WORKER_RE.findall(bundle)
    if len(workers) != 1:
        raise Refused(f"expected exactly one eSpeakNGWorker promise in the bundle, found "
                      f"{len(workers)}; this release {NEEDS_CODE} to the export locator")
    fs_names = set(_FS_RE.findall(bundle))
    if len(fs_names) != 1:
        raise Refused("expected exactly one eSpeak NG filesystem object in the bundle, "
                      f"found {sorted(fs_names)}; this release {NEEDS_CODE} to the export locator")
    return workers[0][0], next(iter(fs_names))


def append_exports(upstream: bytes) -> bytes:
    """The esbuild output with the line that exports the two eSpeak NG handles."""
    text = upstream.decode("utf-8")
    if b"\r" in upstream:
        raise Refused("the built bundle contains a raw carriage return, which git newline "
                      "conversion would corrupt")
    if "espeakWorker" in text or "espeakFS" in text:
        raise Refused("the built bundle already exports espeakWorker or espeakFS")
    if not text.endswith("\n"):
        raise Refused("the built bundle does not end with a newline")
    worker, fs_name = locate_espeak_exports(text)
    return (text + f"\nexport{{{worker} as espeakWorker,{fs_name} as espeakFS}};\n").encode("utf-8")


def smoke_check(node: str, bundle_path: Path) -> None:
    """Import the built bundle in Node and require exactly the expected export names."""
    script = ("import(process.argv[1]).then(m=>console.log(JSON.stringify(Object.keys(m).sort())))"
              ".catch(e=>{console.error(String(e&&e.stack||e));process.exit(3)})")
    proc = _run([node, "-e", script, bundle_path.as_uri()], bundle_path.parent, "importing the built bundle")
    try:
        names = set(json.loads(proc.stdout.strip().splitlines()[-1]))
    except (ValueError, IndexError) as e:
        raise Refused(f"importing the built bundle printed no export list: {proc.stdout!r}") from e
    if names != BUNDLE_EXPORTS:
        raise Refused(f"the built bundle exports {sorted(names)}, expected {sorted(BUNDLE_EXPORTS)}; "
                      f"this release {NEEDS_CODE}")


def build_bundle(pins: Pins, workdir: Path) -> Built:
    """Install the pinned packages into *workdir* and build the bundle there."""
    node, npm = _tool("node"), _tool("npm")
    manifest = {
        "name": "tts-bundle-build", "version": "0.0.0", "private": True, "type": "module",
        "dependencies": {"kokoro-js": pins.kokoro, "@huggingface/transformers": pins.transformers,
                         "phonemizer": pins.phonemizer},
        "overrides": {"@huggingface/transformers": pins.transformers},
        "devDependencies": {"esbuild": pins.esbuild},
    }
    (workdir / "package.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (workdir / "entry.js").write_text(ENTRY, encoding="utf-8")
    _run([npm, "install", "--ignore-scripts", "--no-audit", "--no-fund", "--loglevel=error"],
         workdir, "npm install")

    kokoro = _installed(workdir, "kokoro-js")
    versions = {"kokoro-js": kokoro.get("version"),
                "transformers": _installed(workdir, "@huggingface/transformers").get("version"),
                "phonemizer": _installed(workdir, "phonemizer").get("version"),
                "esbuild": _installed(workdir, "esbuild").get("version"),
                "onnxruntime-web": _installed(workdir, "onnxruntime-web").get("version")}
    wanted = {"kokoro-js": pins.kokoro, "transformers": pins.transformers,
              "phonemizer": pins.phonemizer, "esbuild": pins.esbuild}
    for key, want in wanted.items():
        if versions[key] != want:
            raise Refused(f"npm installed {key} {versions[key]}, not the requested {want}")
    copies = [p for p in (workdir / "node_modules").rglob("package.json")
              if p.parent.name == "onnxruntime-web" and p.parent.parent.name == "node_modules"]
    if len(copies) != 1:
        raise Refused(f"expected onnxruntime-web installed once, found {len(copies)}")

    esbuild_js = workdir / "node_modules" / "esbuild" / "bin" / "esbuild"
    proc = _run([node, str(esbuild_js), "entry.js", *ESBUILD_ARGS, "--outfile=out.js",
                 "--log-level=warning"], workdir, "esbuild")
    upstream = (workdir / "out.js").read_bytes()
    bundle = append_exports(upstream)
    if f'"{pins.transformers}"'.encode() not in bundle:
        raise Refused(f"the built bundle does not contain the version string \"{pins.transformers}\"; "
                      "it was not built from the requested transformers.js")
    final = workdir / "kokoro.min.js"
    final.write_bytes(bundle)
    smoke_check(node, final)

    dist = workdir / "node_modules" / "@huggingface" / "transformers" / "dist"
    names = sorted({m for m in _RUNTIME_NAME_RE.findall(bundle.decode("utf-8", errors="replace"))
                    if m.endswith((".mjs", ".wasm"))})
    if not names:
        raise Refused("the built bundle names no onnxruntime runtime file")
    runtime = {}
    for name in names:
        path = dist / name
        if not path.is_file():
            raise Refused(f"the bundle loads {name} but transformers {pins.transformers} does not "
                          f"ship it in dist/; this release {NEEDS_CODE} (the runtime is no longer "
                          "vendored from the package)")
        runtime[name] = path.read_bytes()

    def _license(package_dir: Path) -> bytes:
        path = package_dir / "LICENSE"
        if not path.is_file():
            raise Refused(f"{package_dir.name} ships no LICENSE file")
        return path.read_bytes()

    nm = workdir / "node_modules"
    licenses = {"kokoro-js": _license(nm / "kokoro-js"),
                "transformers": _license(nm / "@huggingface" / "transformers")}
    voices_dir = nm / "kokoro-js" / "voices"
    voices = {p.stem for p in voices_dir.glob("*.bin")} if voices_dir.is_dir() else set()
    peer = (kokoro.get("dependencies") or {}).get("@huggingface/transformers", "")
    return Built(bundle=bundle, runtime=runtime, licenses=licenses, versions=versions,
                 kokoro_peer_range=peer, voices=voices, warnings=proc.stderr.strip())


# --------------------------------------------------------------------------- #
#  Planning                                                                   #
# --------------------------------------------------------------------------- #

def _change(repo: Path, rel: str, new: bytes, *, text: bool, newline: str | None = None) -> Change | None:
    """A Change turning the file at *rel* into *new*, or None when it already is *new*."""
    path = repo / rel
    old = path.read_bytes() if path.is_file() else None
    if old is not None and (_lf(old) == _lf(new) if text else old == new):
        return None
    if newline is None:
        newline = "\r\n" if (old is not None and b"\r\n" in old and text) else "\n"
    return Change(rel=rel, path=path, old=old, new=new, text=text, newline=newline)


def rewrite_notice(text: str, old: Pins, new: Pins, built: Built, ort_source: tuple | None) -> str:
    """NOTICE.md (LF text) with every region that names the moved versions rewritten."""
    for key, value in (("kokoro-js", new.kokoro), ("transformers", new.transformers),
                       ("onnxruntime-web", new.ort), ("phonemizer", new.phonemizer)):
        text = _replace_once(_ROW_RES[key], text, lambda m, v=value: m.group(1) + v + m.group(3),
                             f"NOTICE.md {key} row")
    text = _replace_once(_ESBUILD_RE, text, lambda m: m.group(1) + new.esbuild + m.group(3),
                         "NOTICE.md esbuild version")
    text = _replace_once(_COPIES_RE, text, lambda m: m.group(1) + new.transformers + m.group(3),
                         "NOTICE.md runtime provenance")
    text = _replace_once(_ENV_VERSION_RE, text, lambda m: m.group(1) + new.transformers + m.group(3),
                         "NOTICE.md env.version")
    for name, rx in _SIZE_RES.items():
        size = str(len(built.runtime[name]))

        def fit(m, size=size):
            width = len(m.group(2)) + len(m.group(3))
            return m.group(1) + " " * max(width - len(size), 2) + size + m.group(4)
        text = _replace_once(rx, text, fit, f"NOTICE.md size of {name}")
    if ort_source is not None:
        kind, ref = ort_source
        tail = ("exact commit the pinned dev version" if kind == "commit"
                else "release the pinned version")
        sentence = (f"microsoft/onnxruntime repository at {kind} `{ref}`,\n"
                    f"which is the {tail} names.")
        text = _replace_once(_ORT_SOURCE_RE, text, lambda m: sentence, "NOTICE.md onnxruntime licence source")
    return text


def rewrite_guard_test(text: str, new: Pins, built: Built) -> str:
    """tests-js/vendor-onnxruntime.test.mjs (LF text) with the new version, hashes and size."""
    mjs, wasm = built.runtime["ort-wasm-simd-threaded.jsep.mjs"], built.runtime["ort-wasm-simd-threaded.jsep.wasm"]
    values = {"version": new.transformers, "mjs_hash": _sha256_b64(_lf(mjs)),
              "wasm_hash": _sha256_b64(wasm), "wasm_bytes": str(len(wasm)), "pack": new.transformers}
    for key, value in values.items():
        text = _replace_once(_TEST_RES[key], text, lambda m, v=value: m.group(1) + v + m.group(3),
                             f"{TEST_REL} {key}")
    return text


def build_plan(repo: Path, old: Pins, new: Pins, built: Built, *, license_fetch=None) -> dict:
    """Every file change moving the tree from the *old* pins to the *new* ones.

    *license_fetch* is called with the onnxruntime-web version and returns
    (licence bytes, kind, ref); it is only called when that version moved.
    """
    runtime_dir = repo / RUNTIME_REL
    vendored = {p.name for p in runtime_dir.iterdir() if p.is_file()} if runtime_dir.is_dir() else set()
    if set(built.runtime) != vendored:
        raise Refused(f"the bundle loads runtime files {sorted(built.runtime)} but {RUNTIME_REL} holds "
                      f"{sorted(vendored)}; tts.js, the guard test and NOTICE.md name these files, "
                      f"so this release {NEEDS_CODE}")
    if set(built.runtime) != set(_SIZE_RES):
        raise Refused(f"unexpected runtime files {sorted(built.runtime)}; the guard test pins "
                      f"{sorted(_SIZE_RES)}, so this release {NEEDS_CODE}")

    ort_source = None
    ort_license = None
    if new.ort != old.ort:
        if license_fetch is None:
            raise Refused("onnxruntime-web moved but no licence source was provided")
        ort_license, kind, ref = license_fetch(new.ort)
        ort_source = (kind, ref)

    changes = []

    def add(change):
        if change is not None:
            changes.append(change)

    add(_change(repo, BUNDLE_REL, built.bundle, text=True, newline="\n"))
    for name, data in built.runtime.items():
        add(_change(repo, f"{RUNTIME_REL}/{name}", data, text=name.endswith(".mjs")))
    add(_change(repo, LICENSE_RELS["kokoro-js"], built.licenses["kokoro-js"], text=True))
    add(_change(repo, LICENSE_RELS["transformers"], built.licenses["transformers"], text=True))
    if ort_license is not None:
        add(_change(repo, LICENSE_RELS["onnxruntime-web"], ort_license, text=True))

    notice, newline = _read_text(repo / NOTICE_REL)
    new_notice = rewrite_notice(notice, old, new, built, ort_source)
    add(_change(repo, NOTICE_REL, new_notice.encode("utf-8"), text=True, newline=newline))
    guard, newline = _read_text(repo / TEST_REL)
    add(_change(repo, TEST_REL, rewrite_guard_test(guard, new, built).encode("utf-8"),
                text=True, newline=newline))
    return {"changes": changes, "ort_source": ort_source}


def check_voices(repo: Path, built: Built) -> list:
    """Voice ids that differ between the kokoro-js package and the shipped voices.json."""
    try:
        shipped = set(json.loads((repo / VOICES_REL).read_text(encoding="utf-8")))
    except (OSError, ValueError) as e:
        return [f"{VOICES_REL} could not be read: {e}"]
    notes = []
    if built.voices - shipped:
        notes.append("voices in the kokoro-js package but not in voices.json: "
                     + ", ".join(sorted(built.voices - shipped)))
    if shipped - built.voices:
        notes.append("voices in voices.json but not in the kokoro-js package: "
                     + ", ".join(sorted(shipped - built.voices)))
    return notes


def peer_mismatch(built: Built, new: Pins) -> str | None:
    """A note when kokoro-js declares a transformers.js range the bundled version is outside."""
    m = re.fullmatch(r"\^(\d+)\.(\d+)\.(\d+)", built.kokoro_peer_range or "")
    if not m:
        return None
    have = parse_version(new.transformers)
    if int(m.group(1)) != have[0]:
        return (f"kokoro-js {new.kokoro} declares @huggingface/transformers {built.kokoro_peer_range} "
                f"but the bundle carries {new.transformers}")
    return None


# --------------------------------------------------------------------------- #
#  Writing                                                                    #
# --------------------------------------------------------------------------- #

def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".bump-tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def apply_changes(changes: list) -> None:
    """Write every change; if any write fails, restore the files already written."""
    done = []
    try:
        for change in changes:
            before = change.path.read_bytes() if change.path.is_file() else None
            _write_atomic(change.path, change.on_disk())
            done.append((change.path, before))
    except BaseException:
        for path, before in reversed(done):
            if before is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(before)
        raise


# --------------------------------------------------------------------------- #
#  Output                                                                     #
# --------------------------------------------------------------------------- #

def describe(change: Change) -> str:
    verb = "create " if change.old is None else "replace"
    size = f"{len(change.new)} bytes"
    if change.old is not None:
        size = f"{len(change.old)} -> {len(change.new)} bytes"
    return f"  {verb} {change.rel}  ({size}, sha256 {hashlib.sha256(change.new).hexdigest()[:16]})"


def checklist(lib: str | None, old: Pins, new: Pins, built: Built, notes: list, mentions: list) -> str:
    lines = ["REMAINING STEPS, not automated - each has its own check:"]
    n = 1
    if lib is not None and crosses_major(parse_version(old.kokoro if lib == "kokoro-js" else old.transformers),
                                         parse_version(new.kokoro if lib == "kokoro-js" else new.transformers)):
        lines += [f"  !! MAJOR BOUNDARY in {lib}: read upstream's changelog for every release in between",
                  "       and prove neural speech in a real browser before this merges."]
    for note in notes:
        lines.append(f"  !! {note}")
    if built.warnings:
        lines += ["  !! esbuild warnings:"] + [f"       {w}" for w in built.warnings.splitlines()[:12]]
    lines += [f"  {n}. npm ci && npm test",
              "       tts-g2p pins the phonemes for all six languages against the new bundle;",
              "       vendor-onnxruntime pins the runtime pair"]
    n += 1
    lines += [f"  {n}. start `localm gui --no-model --no-browser` in a throwaway LOCALM_HOME and speak a",
              "       sentence with neural TTS in a real browser, offline; confirm the CSP holds"]
    n += 1
    lines += [f"  {n}. re-check advisory status (OSV, GitHub advisory database) for the new versions"]
    n += 1
    if mentions:
        lines += [f"  {n}. lines in NOTICE.md still naming an old version (history; edit only if now false):"]
        lines += [f"       {m}" for m in mentions]
        n += 1
    lines += [f"  {n}. CHANGELOG.md, [Unreleased]: one user-facing bullet if the shipped bundle changed",
              f"  {n + 1}. python scripts/check_hygiene.py"]
    return "\n".join(lines)


def find_mentions(text: str, olds: set) -> list:
    out = []
    for number, line in enumerate(text.split("\n"), 1):
        if any(re.search(rf"(?<![\d.]){re.escape(o)}(?![\d])", line) for o in olds):
            out.append(f"{NOTICE_REL}:{number}: {line.strip()[:110]}")
    return out


# --------------------------------------------------------------------------- #
#  Command line                                                               #
# --------------------------------------------------------------------------- #

def _moved(old: Pins, lib: str, tag: str) -> Pins:
    new = Pins(**old.__dict__)
    if lib == "kokoro-js":
        new.kokoro = tag
    else:
        new.transformers = tag
    return new


def main(argv=None, *, build_fn=build_bundle, fetch_json_fn=fetch_json,
         fetch_bytes_fn=fetch_bytes) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lib", choices=LIBS, help="the library whose release to move")
    ap.add_argument("--tag", help="the npm release to move it to, e.g. 3.9.0")
    ap.add_argument("--verify", action="store_true",
                    help="rebuild the current pins and compare them with the tree; change nothing")
    ap.add_argument("--write", action="store_true",
                    help="apply the edit (default: print a summary and diff, change nothing)")
    args = ap.parse_args(argv)
    if args.verify == bool(args.lib or args.tag) or bool(args.lib) != bool(args.tag):
        ap.error("give either --verify, or --lib and --tag together")
    if args.verify and args.write:
        ap.error("--verify never writes")

    repo = REPO
    try:
        old = read_pins(repo)
        new = Pins(**old.__dict__)
        if not args.verify:
            current = old.kokoro if args.lib == "kokoro-js" else old.transformers
            tag = args.tag.strip()
            if parse_version(tag) <= parse_version(current):
                raise Refused(f"{args.lib} {tag} is not newer than the pinned {current}; "
                              "this script only moves a library forward")
            new = _moved(old, args.lib, tag)
        print(f"building: kokoro-js {new.kokoro}, @huggingface/transformers {new.transformers}, "
              f"phonemizer {new.phonemizer}, esbuild {new.esbuild} ...")
        with tempfile.TemporaryDirectory(prefix="tts-bundle-") as tmp:
            built = build_fn(new, Path(tmp))
        new.ort = built.versions["onnxruntime-web"]
        if args.verify and new.ort != old.ort:
            raise Refused(f"transformers {new.transformers} pins onnxruntime-web {new.ort} but "
                          f"NOTICE.md records {old.ort}")
        plan = build_plan(repo, old, new, built, license_fetch=lambda v: fetch_ort_license(
            v, fetch_json_fn, fetch_bytes_fn))
        if args.verify:
            ort_license, kind, ref = fetch_ort_license(old.ort, fetch_json_fn, fetch_bytes_fn)
            extra = _change(repo, LICENSE_RELS["onnxruntime-web"], ort_license, text=True)
            if extra is not None:
                plan["changes"].append(extra)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 1

    changes = plan["changes"]
    if args.verify:
        if changes:
            print(f"NOT IDENTICAL: rebuilding the pinned versions differs from the tree in {len(changes)} file(s):")
            for change in changes:
                print(describe(change))
            return 1
        print("IDENTICAL: rebuilding the pinned versions reproduces every vendored file byte for byte")
        return 0

    olds = {old.kokoro, old.transformers, old.ort} - {new.kokoro, new.transformers, new.ort}
    notes = check_voices(repo, built)
    mismatch = peer_mismatch(built, new)
    if mismatch:
        notes.append(mismatch)
    notice_now, _ = _read_text(repo / NOTICE_REL)
    new_notice = next((c.new.decode("utf-8") for c in changes if c.rel == NOTICE_REL), notice_now)
    mentions = find_mentions(new_notice.replace("\r\n", "\n"), olds)

    if not changes:
        print(f"nothing to change: the tree already carries {args.lib} {args.tag}")
    else:
        print(f"FILES EDITED ({len(changes)}):")
        for change in changes:
            print(describe(change))
        for change in changes:
            if change.text and change.rel != BUNDLE_REL and change.old is not None:
                sys.stdout.writelines(difflib.unified_diff(
                    _lf(change.old).decode("utf-8").splitlines(keepends=True),
                    change.new.decode("utf-8").splitlines(keepends=True),
                    fromfile=f"a/{change.rel}", tofile=f"b/{change.rel}"))
        if args.write:
            try:
                apply_changes(changes)
            except OSError as e:
                print(f"REFUSED: writing failed and the files already written were restored: {e}")
                return 1
            print(f"wrote {len(changes)} file(s)")
        else:
            print("\n(dry run: nothing written; add --write to apply)")
    print()
    print(checklist(args.lib, old, new, built, notes, mentions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
