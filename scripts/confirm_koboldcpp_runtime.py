#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Confirm a KoboldCpp release WORKS WITH LOCALM before it is pinned.

localm uses KoboldCpp for native music generation (ACE-Step 1.5). This script
installs a release through localm's own installer, starts it through localm's own
process and server wrapper, and generates real output with it. It runs every
check below and records each one in the receipt:

  isolation        every localm path resolves inside the scratch dir and the
                   localm that is imported is this checkout's.
  release_assets   the GitHub release lists every asset the pin table names,
                   each with a sha256 digest. For --current the pinned table
                   must equal the published one.
  install          runtime.ensure_for_backend() downloads the platform asset,
                   verifies its size and sha256 against the published digest,
                   unpacks it, and checks the launcher version, inside a
                   throwaway LOCALM_HOME.
  launcher_version the unpacked launcher's --version equals the candidate
                   version, and the install marker names the candidate tag and
                   asset (proves the candidate, not the pin, was installed).
  server_api       the launcher, started with the same transport and backend
                   flags server.build_argv() produces and the same process
                   wrapper, answers /api/extra/version on a free port.
  text_generation  /api/v1/generate returns text from a small causal GGUF.
  vulkan_device    the startup log reports a Vulkan device (required when the
                   machine has a GPU and the backend is vulkan).
  music_generate   music.generate_wav(plan=False) on the cpu backend, the code
                   path behind `localm setup-music --test`, returns a WAV of the
                   requested length that is not silent or constant.
  music_plan       the same with the LM planner on (plan=True), which goes
                   through /api/extra/music/prepare.
  music_gpu        the music_generate request on the machine's GPU backend. It
                   is advisory: a flat or constant track on a GPU backend is a
                   recorded upstream defect that localm answers by regenerating
                   on the next backend, so it is reported but does not decide
                   the verdict. The cpu checks above are the reference.

A check that cannot run is SKIP with its reason; a required SKIP makes the
verdict INCONCLUSIVE, never PASS. Exit 0 PASS, 1 FAIL, 2 INCONCLUSIVE.

Everything runs inside --workdir: LOCALM_HOME, TEMP, TMP and TMPDIR point there
before localm is imported. Models are fetched once into the persistent cache
(--cache-dir, else $LOCALM_PIN_CACHE/koboldcpp, else ~/.cache/localm-pin-cache/
koboldcpp) and verified against the sha256 recorded in MODELS below. Processes
started are recorded by PID and killed as a tree in a finally block.

GPU work should be wrapped by the caller in the shared GPU lease.

Usage:
    python scripts/confirm_koboldcpp_runtime.py --tag v1.123 --workdir W --receipt r.json
    python scripts/confirm_koboldcpp_runtime.py --current --workdir W --receipt r.json

Needs localm importable. Nothing under localm/ imports this script.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

COMPONENT = "koboldcpp"
UPSTREAM_REPO = "LostRuins/koboldcpp"
RELEASE_URL = "https://api.github.com/repos/%s/releases/tags/%s"

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"
SKIP = "SKIP"

CHECK_NAMES = ("isolation", "release_assets", "install", "launcher_version",
               "server_api", "text_generation", "vulkan_device",
               "music_generate", "music_plan", "music_gpu")
ALWAYS_REQUIRED = ("isolation", "release_assets", "install", "launcher_version",
                   "server_api", "text_generation", "music_generate", "music_plan")

ADVISORY_CHECKS = ("music_gpu",)

NOT_MEASURED = [
    "backends other than the one run on this machine (cuda, cpu, metal)",
    "builds other than the one run on this machine",
    "ACE-Step model variants other than localm's default set",
    "music quality beyond not-silent and not-constant",
    "music tracks longer than a few seconds",
    "text generation with models other than the small causal GGUF used here",
]

_TAG_RE = re.compile(r"^v(\d+(?:\.\d+)+)$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")

# Files fetched once into the persistent cache. Identity is (repo, revision,
# file, size, sha256); a cached file that does not match is discarded.
MODELS = {
    "chat": {
        "repo": "bartowski/SmolLM2-135M-Instruct-GGUF",
        "revision": "09816acd5d99df7be770d85ea30822623dab342c",
        "file": "SmolLM2-135M-Instruct-Q4_K_M.gguf",
        "size": 105454432,
        "sha256": "2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d",
        "architecture": "llama",
    },
    "music": {
        "repo": "Serveurperso/ACE-Step-1.5-GGUF",
        "revision": "666ac70204440867d8c01ba4b119cc79c95b370a",
        "files": {
            "acestep-5Hz-lm-0.6B-Q8_0.gguf": (
                709846656,
                "bdaf9e292d4470f31c19cafeaca1b74936a114667e3a85e5d33b65247e9908ec"),
            "Qwen3-Embedding-0.6B-Q8_0.gguf": (
                784144960,
                "972f23255e46adfe744a0eb9a0039f3c63988f65753b0968d776e8b27168c321"),
            "acestep-v15-turbo-Q8_0.gguf": (
                2549528000,
                "288f708a61cfc241013a98a62f98ba331f83fe34d0d3559acdd9b0f6a2f7cd6b"),
            "vae-BF16.gguf": (
                337420928,
                "0599862ac5d15cd308e1d2e368373aea6c02e25ebd1737ad4a4562a0901b0ef8"),
        },
    },
}

MUSIC_SECONDS = 5.0
MUSIC_TIMEOUT = 1800.0
CHAT_START_TIMEOUT = 600.0


class Problem(Exception):
    """A check ended as SKIP (could not measure) or FAIL (the build is bad)."""

    def __init__(self, status: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


# --------------------------------------------------------------------------- #
#  Tags, release listing, receipt                                              #
# --------------------------------------------------------------------------- #

def parse_tag(tag: str) -> tuple[int, ...] | None:
    """The numeric parts of a ``vX.Y[.Z...]`` tag, or None when it is not one."""
    m = _TAG_RE.match(tag or "")
    return tuple(int(p) for p in m.group(1).split(".")) if m else None


def version_of_tag(tag: str) -> str:
    """The version string ``--version`` prints for *tag* (the tag without ``v``)."""
    if parse_tag(tag) is None:
        raise ValueError(f"{tag!r} is not a vX.Y[.Z] release tag")
    return tag[1:]


def parse_release_assets(body) -> dict:
    """{asset name: (size, sha256)} from a GitHub release API body.

    Raises ValueError when the body is not a release listing, an asset has no
    sha256 digest or a malformed one, or a size is not a positive integer."""
    assets = body.get("assets") if isinstance(body, dict) else None
    if not isinstance(assets, list) or not assets:
        raise ValueError("the release listing carries no asset list")
    out: dict = {}
    for a in assets:
        name = a.get("name") if isinstance(a, dict) else None
        digest = a.get("digest") if isinstance(a, dict) else None
        size = a.get("size") if isinstance(a, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError("an asset has no name")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise ValueError(f"asset {name!r} has no sha256 digest in the API listing")
        sha = digest.split("sha256:", 1)[1].strip().lower()
        if not _SHA_RE.match(sha):
            raise ValueError(f"asset {name!r} has a malformed digest {digest!r}")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"asset {name!r} has no valid size")
        out[name] = (size, sha)
    return out


def fetch_release_body(tag: str, opener=None):
    """The decoded GitHub API body of the release tagged *tag*."""
    if opener is None:
        from localm.http_ssl import verified_urlopen
        opener = verified_urlopen
    headers = {"Accept": "application/vnd.github+json",
               "User-Agent": "localm-confirm-koboldcpp"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(RELEASE_URL % (UPSTREAM_REPO, tag), headers=headers)
    with opener(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_table(pinned: dict, published: dict) -> dict:
    """The (platform, build) -> (name, size, sha256) table for a release.

    *pinned* supplies the asset names; *published* supplies size and digest.
    Raises ValueError listing every pinned asset the release does not carry."""
    missing = sorted(f"{k[0]}/{k[1]}:{v[0]}" for k, v in pinned.items()
                     if v[0] not in published)
    if missing:
        raise ValueError("the release does not carry these assets the pin table "
                         f"names: {', '.join(missing)}")
    return {k: (v[0], published[v[0]][0], published[v[0]][1])
            for k, v in pinned.items()}


def table_to_json(table: dict) -> dict:
    return {f"{k[0]}/{k[1]}": {"name": v[0], "size": v[1], "sha256": v[2]}
            for k, v in sorted(table.items())}


def table_from_json(data) -> dict:
    """The inverse of :func:`table_to_json`; ValueError on any malformed entry."""
    if not isinstance(data, dict) or not data:
        raise ValueError("the asset table is missing or empty")
    out = {}
    for key, v in data.items():
        plat, sep, build = str(key).partition("/")
        if not sep or not isinstance(v, dict):
            raise ValueError(f"malformed asset table entry {key!r}")
        name, size, sha = v.get("name"), v.get("size"), v.get("sha256")
        if (not isinstance(name, str) or not isinstance(size, int)
                or isinstance(size, bool) or not isinstance(sha, str)
                or not _SHA_RE.match(sha)):
            raise ValueError(f"malformed asset table entry {key!r}")
        out[(plat, build)] = (name, size, sha)
    return out


def compute_verdict(checks: dict) -> tuple[str, str]:
    """(verdict, why) from the checks dict.

    FAIL when any non-advisory check failed; INCONCLUSIVE when a required check
    is not PASS and none failed; PASS only when every required check is PASS.
    A failed advisory check is reported in the text but does not change it."""
    failed = sorted(n for n, c in checks.items()
                    if c.get("status") == FAIL and n not in ADVISORY_CHECKS)
    if failed:
        return FAIL, "failed: " + "; ".join(f"{n}: {checks[n].get('detail', '')}"
                                            for n in failed)
    unmet = sorted(n for n, c in checks.items()
                   if c.get("required") and c.get("status") != PASS)
    if unmet:
        return INCONCLUSIVE, "not measured: " + "; ".join(
            f"{n}: {checks[n].get('detail', '')}" for n in unmet)
    missing = [n for n in ALWAYS_REQUIRED if n not in checks]
    if missing:
        return INCONCLUSIVE, "checks absent from the run: " + ", ".join(missing)
    advisory = sorted(n for n in ADVISORY_CHECKS
                      if checks.get(n, {}).get("status") == FAIL)
    note = ("; advisory check failed (does not decide the verdict): "
            + "; ".join(f"{n}: {checks[n].get('detail', '')}" for n in advisory)
            if advisory else "")
    return PASS, "every required check passed" + note


def new_receipt(tag: str, current: bool) -> dict:
    return {"schema": 1, "component": COMPONENT, "tag": tag, "current": current,
            "verdict": INCONCLUSIVE, "why": "run in progress", "written_at": "",
            "hardware": {}, "checks": {}, "not_measured": list(NOT_MEASURED)}


def set_check(receipt: dict, name: str, status: str, detail: str, *,
              required: bool = True) -> None:
    receipt["checks"][name] = {"status": status, "required": required,
                               "detail": detail}


def save_receipt(path: Path, receipt: dict) -> None:
    """Write *receipt* to *path* atomically (temp file, then replace)."""
    receipt["written_at"] = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def finalize(receipt: dict) -> int:
    verdict, why = compute_verdict(receipt["checks"])
    receipt["verdict"], receipt["why"] = verdict, why
    return {PASS: 0, FAIL: 1}.get(verdict, 2)


# --------------------------------------------------------------------------- #
#  Candidate pins                                                              #
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def patched_pins(pins, tag: str, version: str, table: dict):
    """Make localm's pin module describe the candidate release for the duration
    of the block, restoring the originals afterwards."""
    saved = (pins.TAG, pins.VERSION, pins._BASE_URL, dict(pins.ASSETS))
    pins.TAG, pins.VERSION = tag, version
    pins._BASE_URL = f"https://github.com/{pins.REPO}/releases/download/{tag}/"
    pins.ASSETS.clear()
    pins.ASSETS.update(table)
    try:
        yield
    finally:
        pins.TAG, pins.VERSION, pins._BASE_URL = saved[0], saved[1], saved[2]
        pins.ASSETS.clear()
        pins.ASSETS.update(saved[3])


def prove_pins_applied(pins, runtime, tag: str, build: str, name: str) -> str:
    """Raise Problem(SKIP) unless the patched pins are what the installer will
    read; returns a one-line description."""
    expected_url = f"https://github.com/{pins.REPO}/releases/download/{tag}/{name}"
    got_url = pins.asset_url(name)
    got_dir = runtime.runtime_dir(build).name
    if got_url != expected_url or got_dir != f"{tag}-{build}":
        raise Problem(SKIP, f"the candidate pins did not take effect (url {got_url!r}, "
                            f"runtime dir {got_dir!r}); nothing was measured")
    return f"installer will fetch {got_url} into {got_dir}"


# --------------------------------------------------------------------------- #
#  Isolation and process bookkeeping                                           #
# --------------------------------------------------------------------------- #

def prepare_env(workdir: Path, cache_dir: Path) -> tuple[Path, Path]:
    """Create <workdir>/home and <workdir>/tmp and point LOCALM_HOME, TEMP, TMP,
    TMPDIR and HF_HOME at them. Must run before localm is imported."""
    home, tmp = workdir / "home", workdir / "tmp"
    home.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LOCALM_HOME"] = str(home)
    for k in ("TEMP", "TMP", "TMPDIR"):
        os.environ[k] = str(tmp)
    os.environ["HF_HOME"] = str(cache_dir / "hf-home")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    return home, tmp


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def verify_isolation(workdir: Path) -> tuple[bool, str]:
    """(ok, detail): localm is this checkout's and every path it will write
    resolves under *workdir*."""
    import localm
    from localm.config import HOME_DIR, home_dir
    from localm.media.koboldcpp import runtime
    src = Path(localm.__file__)
    if not _inside(src, REPO):
        return False, f"localm was imported from {src}, not from this checkout ({REPO})"
    paths = {"HOME_DIR": Path(HOME_DIR), "home_dir()": Path(home_dir()),
             "runtimes_root()": runtime.runtimes_root(),
             "tempdir": Path(tempfile.gettempdir())}
    bad = {k: str(v) for k, v in paths.items() if not _inside(v, workdir)}
    if bad:
        return False, f"paths outside the scratch dir {workdir}: {bad}"
    return True, f"localm from {src}; every path resolves under {workdir}"


class PidRecorder:
    """Records the PID (with its start time) of every process localm's wrapper
    starts, so exactly those can be killed as trees at the end."""

    def __init__(self) -> None:
        self.entries: list = []
        self._orig = None

    def install(self, proc_module) -> None:
        self._orig = proc_module.start

        def recording_start(argv, *, cwd, env):
            mp = self._orig(argv, cwd=cwd, env=env)
            self.entries.append((mp.pid, _start_time(mp.pid)))
            return mp
        proc_module.start = recording_start

    def uninstall(self, proc_module) -> None:
        if self._orig is not None:
            proc_module.start = self._orig
            self._orig = None

    def reap(self) -> list:
        """Kill every recorded process that is still the same live process.
        Returns the PIDs killed."""
        killed = []
        for pid, started in self.entries:
            if _start_time(pid) is None or _start_time(pid) != started:
                continue
            _kill_tree(pid)
            killed.append(pid)
        return killed


def _start_time(pid: int):
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:
        return None


def _kill_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, timeout=30)
        return
    try:
        import psutil
        proc = psutil.Process(pid)
        for child in proc.children(recursive=True):
            child.kill()
        proc.kill()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
#  Model cache                                                                 #
# --------------------------------------------------------------------------- #

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def default_cache_dir() -> Path:
    env = os.environ.get("LOCALM_PIN_CACHE")
    base = Path(env) if env else Path.home() / ".cache" / "localm-pin-cache"
    return base / COMPONENT


def file_matches(path: Path, size: int, sha256: str) -> bool:
    try:
        return path.is_file() and path.stat().st_size == size and sha256_of(path) == sha256
    except OSError:
        return False


@contextlib.contextmanager
def _cache_lock(cache_dir: Path, timeout: float = 3600.0):
    """Hold a cross-process lock directory in the cache; one downloader at a time."""
    lock = cache_dir / ".download.lock"
    deadline = time.monotonic() + timeout
    while True:
        try:
            lock.mkdir()
            (lock / "pid").write_text(str(os.getpid()), encoding="utf-8")
            break
        except FileExistsError:
            holder = None
            with contextlib.suppress(OSError, ValueError):
                holder = int((lock / "pid").read_text(encoding="utf-8").strip())
            alive = False
            if holder is not None:
                with contextlib.suppress(Exception):
                    import psutil
                    alive = psutil.pid_exists(holder)
            if holder is not None and not alive:
                shutil.rmtree(lock, ignore_errors=True)
                continue
            if time.monotonic() > deadline:
                raise Problem(SKIP, "another run held the model cache lock for too long") from None
            time.sleep(2.0)
    try:
        yield
    finally:
        shutil.rmtree(lock, ignore_errors=True)


def ensure_model(repo: str, revision: str, filename: str, size: int, sha256: str,
                 cache_dir: Path, downloader=None) -> Path:
    """The cached file for (repo, revision, filename), downloaded when absent and
    verified against *size* and *sha256*. Raises Problem(SKIP) when it cannot be
    obtained or does not match."""
    dest_dir = cache_dir / "models" / repo.replace("/", "__")
    dest = dest_dir / filename
    if file_matches(dest, size, sha256):
        return dest
    with _cache_lock(cache_dir):
        if file_matches(dest, size, sha256):
            return dest
        if dest.exists():
            dest.unlink()
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            if downloader is None:
                from huggingface_hub import hf_hub_download
                downloader = hf_hub_download
            got = downloader(repo_id=repo, filename=filename, revision=revision,
                             local_dir=str(dest_dir))
        except Exception as e:
            raise Problem(SKIP, f"could not download {repo}/{filename}: "
                                f"{type(e).__name__}: {e}") from e
        got = Path(got)
        if not file_matches(got, size, sha256):
            with contextlib.suppress(OSError):
                got.unlink()
            raise Problem(SKIP, f"{repo}/{filename} at {revision[:12]} does not match "
                                f"the recorded size {size} / sha256 {sha256[:16]}")
        return got


def model_record(path: Path, repo: str, revision: str, size: int, sha256: str) -> dict:
    return {"file": path.name, "repo": repo, "revision": revision, "size": size,
            "sha256": sha256}


# --------------------------------------------------------------------------- #
#  Checks                                                                      #
# --------------------------------------------------------------------------- #

def pick_build_and_backend(runtime, det, build_arg, backend_arg) -> tuple[str, str]:
    """The (build, backend) this machine uses: the product's own choice unless
    overridden on the command line."""
    backend = backend_arg or runtime.recommended_backend(det)
    build = build_arg or runtime.build_for(backend)
    return build, backend


def check_release_assets(ctx) -> dict:
    """Fetch the release listing and build the candidate asset table. Returns
    the table; for --current it must equal the pinned one."""
    try:
        body = ctx.fetch_release(ctx.tag)
        published = parse_release_assets(body)
    except ValueError as e:
        raise Problem(FAIL, f"the {ctx.tag} release listing is unusable: {e}") from e
    except Exception as e:
        raise Problem(SKIP, f"could not read the {ctx.tag} release from the GitHub "
                            f"API: {type(e).__name__}: {e}") from e
    try:
        table = build_table(ctx.pinned_assets, published)
    except ValueError as e:
        raise Problem(FAIL, str(e)) from e
    if ctx.current and table != ctx.pinned_assets:
        diff = sorted(f"{k[0]}/{k[1]}" for k in table if table[k] != ctx.pinned_assets[k])
        raise Problem(FAIL, "the pinned asset table disagrees with the published "
                            f"digests for: {', '.join(diff)}")
    ctx.table = table
    return table


def check_install(ctx):
    """Install the build for the candidate through runtime.ensure_for_backend()
    and return the Runtime."""
    runtime = ctx.runtime
    name = ctx.table[(runtime.platform_key(), ctx.build)][0]
    where = prove_pins_applied(ctx.pins, runtime, ctx.tag, ctx.build, name)
    verify_failed = []
    real_verify = runtime.verify_file

    def recording_verify(*a, **kw):
        try:
            return real_verify(*a, **kw)
        except runtime.DownloadError:
            verify_failed.append(True)
            raise

    runtime.verify_file = recording_verify
    last = ""
    try:
        for attempt in (1, 2):
            verify_failed.clear()
            try:
                rt = ctx.installer(ctx.backend)
                if rt.build != ctx.build:
                    raise Problem(SKIP, f"the installer returned the {rt.build} build, "
                                        f"not the {ctx.build} build under test")
                return rt, (f"{where}; installed {rt.path.name} and verified its size "
                            "and sha256 against the published digest")
            except runtime.DownloadError as e:
                last = str(e)
                if verify_failed and attempt == 2:
                    raise Problem(FAIL, f"the downloaded asset does not match the "
                                        f"published digest: {e}") from e
                if not verify_failed and attempt == 2:
                    raise Problem(SKIP, f"the download did not complete: {e}") from e
            except runtime.ProvisionError as e:
                raise Problem(FAIL, f"the build did not install: {e}") from e
            except OSError as e:
                raise Problem(SKIP, f"could not install: {type(e).__name__}: {e}") from e
    finally:
        runtime.verify_file = real_verify
    raise Problem(SKIP, f"the download did not complete: {last}")


def check_launcher_version(ctx, rt) -> tuple:
    runtime = ctx.runtime
    try:
        got = runtime.launcher_version(rt.launcher)
    except runtime.ProvisionError as e:
        raise Problem(FAIL, f"the launcher would not report a version: {e}") from e
    if got != ctx.version:
        raise Problem(FAIL, f"the launcher reports {got!r}, expected {ctx.version!r}")
    meta = json.loads((rt.path / runtime.MARKER).read_text(encoding="utf-8"))
    name, _size, sha = ctx.table[(runtime.platform_key(), ctx.build)]
    expected = {"tag": ctx.tag, "version": ctx.version, "build": ctx.build,
                "asset": name, "sha256": sha}
    if meta != expected:
        raise Problem(FAIL, f"the install marker is {meta}, expected {expected}")
    return True, f"launcher --version prints {got}; install marker names {ctx.tag} / {name}"


def chat_argv_tail(server_mod, launcher: str, backend: str, port: int) -> list:
    """The transport and backend flags server.build_argv() puts after its music
    model flags, for a server started on *port* with *backend*."""
    key = server_mod.ServerKey(launcher, backend,
                               server_mod.ModelSet("-", "-", "-"))
    argv = server_mod.build_argv(key, port)
    if "--host" not in argv:
        raise Problem(SKIP, "server.build_argv() no longer has a --host flag; the "
                            "chat launch cannot be derived from it")
    return argv[argv.index("--host"):]


_VULKAN_FOUND_RE = re.compile(r"ggml_vulkan: Found (\d+) Vulkan devices?", re.I)
_VULKAN_DEV_RE = re.compile(r"ggml_vulkan: (\d+) = (.+)")


def vulkan_devices_from_log(lines) -> list:
    """Device descriptions the startup log lists after 'Found N Vulkan devices'."""
    found = False
    out = []
    for ln in lines:
        if _VULKAN_FOUND_RE.search(ln):
            found = True
            continue
        m = _VULKAN_DEV_RE.search(ln)
        if found and m:
            out.append(m.group(2).strip())
    return out


def run_chat_server(ctx, rt, model: Path, record: dict) -> None:
    """Start the launcher with the chat model the way server._Server starts it
    (same env, process wrapper, flags, identity check), generate text, stop it.
    Fills *record* with results; raises Problem."""
    from localm.media.koboldcpp import _proc, server
    work = ctx.workdir / "tmp" / "chat-server"
    work.mkdir(parents=True, exist_ok=True)
    key = server.ServerKey(str(rt.launcher), ctx.backend,
                           server.ModelSet("-", "-", "-"))
    srv = server._Server(key, work)
    srv.port = server._free_port()
    srv.password = "confirm-" + hashlib.sha256(os.urandom(16)).hexdigest()[:24]
    argv = ([str(rt.launcher), "--model", str(model)]
            + chat_argv_tail(server, str(rt.launcher), ctx.backend, srv.port))
    env = dict(os.environ)
    for k in ("TEMP", "TMP", "TMPDIR"):
        env[k] = str(work)
    env["KCPP_PASSWORD"] = srv.password
    env["PYTHONUNBUFFERED"] = "1"
    try:
        srv.mp = _proc.start(argv, cwd=str(work), env=env)
    except OSError as e:
        raise Problem(FAIL, f"the launcher could not be started: {e}") from e
    srv._reader = threading.Thread(target=srv._read, args=(srv.mp,), daemon=True)
    srv._reader.start()
    try:
        deadline = time.monotonic() + CHAT_START_TIMEOUT
        info = None
        while info is None:
            code = srv.mp.poll()
            if code is not None:
                raise Problem(FAIL, f"the launcher exited while loading the chat model "
                                    f"(exit code {code}): {srv.log_tail() or 'no output'}")
            info = srv._version()
            if info is None:
                if time.monotonic() > deadline:
                    raise Problem(FAIL, "the launcher did not answer /api/extra/version "
                                        f"in {CHAT_START_TIMEOUT:.0f} s: "
                                        f"{srv.log_tail() or 'no output'}")
                time.sleep(0.5)
        ours = srv._listener_is_ours()
        record["version_reply"] = info
        if ours is False:
            raise Problem(SKIP, f"another program is listening on port {srv.port}")
        record["server_api"] = (f"/api/extra/version answered on port {srv.port} "
                                f"(listener is ours: {ours}); reply version "
                                f"{info.get('version')!r}")
        text = _generate_text(srv)
        record["text"] = text
        record["log_lines"] = list(srv.log)
    finally:
        srv.stop()


def _generate_text(srv) -> str:
    body = {"prompt": "Question: What color is a clear daytime sky?\nAnswer:",
            "max_length": 24, "temperature": 0.0, "top_k": 1}
    req = urllib.request.Request(
        srv.base + "/api/v1/generate", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + srv.password})
    try:
        with srv._opener.open(req, timeout=300) as r:
            data = json.loads(r.read())
    except Exception as e:
        raise Problem(FAIL, f"/api/v1/generate failed: {type(e).__name__}: {e}; "
                            f"{srv.log_tail()}") from e
    results = data.get("results") if isinstance(data, dict) else None
    text = results[0].get("text") if isinstance(results, list) and results \
        and isinstance(results[0], dict) else None
    if not isinstance(text, str):
        raise Problem(FAIL, f"/api/v1/generate returned an unexpected body: {str(data)[:200]}")
    return text


def text_is_usable(text: str) -> bool:
    return sum(c.isalpha() for c in text) >= 3


def analyze_wav(data: bytes, seconds: float, tolerance: tuple[float, float]) -> tuple[bool, str]:
    """(ok, detail) for a WAV returned by the music runtime: PCM, a sane sample
    rate, a duration within *tolerance* x *seconds*, and audible signal that is
    not a constant."""
    import io
    import wave
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            nch, width, rate, n = (w.getnchannels(), w.getsampwidth(),
                                   w.getframerate(), w.getnframes())
            frames = w.readframes(n)
    except (wave.Error, EOFError) as e:
        return False, f"not a readable PCM WAV: {e}"
    if n == 0 or rate < 8000:
        return False, f"empty or implausible audio ({n} frames at {rate} Hz)"
    dur = n / rate
    lo, hi = tolerance[0] * seconds, tolerance[1] * seconds
    if not lo <= dur <= hi:
        return False, f"duration {dur:.2f} s is outside {lo:.1f}..{hi:.1f} s"
    count = n * nch
    vals = _decode_pcm(frames, width, count)
    full = float(1 << (8 * width - 1))
    peak = max(abs(v) for v in vals) / full
    rms = (sum(v * v for v in vals) / len(vals)) ** 0.5 / full
    distinct = len(set(vals[:200000]))
    if peak < 0.01 or rms < 0.002 or distinct < 16:
        return False, (f"audio looks silent or constant (peak {peak:.4f}, rms {rms:.4f}, "
                       f"{distinct} distinct sample values)")
    return True, (f"{dur:.2f} s, {rate} Hz, {nch} ch, {8 * width}-bit, peak {peak:.3f}, "
                  f"rms {rms:.3f}")


def _decode_pcm(frames: bytes, width: int, count: int) -> list:
    if width == 1:
        return [b - 128 for b in frames[:count]]
    if width == 2:
        import array
        a = array.array("h")
        a.frombytes(frames[:count * 2])
        if sys.byteorder == "big":
            a.byteswap()
        return list(a)
    return [int.from_bytes(frames[i * width:(i + 1) * width], "little", signed=True)
            for i in range(count)]


def run_music(ctx, plan: bool, cfg: dict, backend: str) -> tuple:
    """Generate a short track on *backend* through music.generate_wav() and
    validate it."""
    from localm.media.koboldcpp import music, server
    from localm.media.koboldcpp.models import ModelError
    request = {"caption": "calm acoustic guitar", "lyrics": "[Instrumental]",
               "instrumental": True, "duration": MUSIC_SECONDS, "seed": 1,
               "stereo": True}
    t0 = time.monotonic()
    try:
        data, used = music.generate_wav(cfg, backend, request, plan=plan,
                                        timeout=MUSIC_TIMEOUT,
                                        on_progress=lambda m: print(f"    {m}", flush=True))
    except (music.NativeMusicError, music.ServerError) as e:
        raise Problem(FAIL, f"music generation failed: {e}") from e
    except ModelError as e:
        raise Problem(SKIP, f"the model files were not usable: {e}") from e
    except music.ProvisionError as e:
        raise Problem(FAIL, f"the runtime could not be provisioned: {e}") from e
    finally:
        server.stop()
    secs = time.monotonic() - t0
    out_dir = ctx.workdir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / f"music-{backend}-{'plan' if plan else 'noplan'}.wav"
    out_wav.write_bytes(data)
    ok, detail = analyze_wav(data, MUSIC_SECONDS, (0.4, 1.6))
    if not ok:
        raise Problem(FAIL, f"{detail}; the returned WAV is kept at {out_wav}")
    return True, (f"{detail}; backend {used}; planner {'on' if plan else 'off'}; "
                  f"took {secs:.0f} s")


# --------------------------------------------------------------------------- #
#  Orchestration                                                               #
# --------------------------------------------------------------------------- #

class Ctx:
    """Everything the checks share. The seams (fetch_release, installer) default
    to the real thing."""

    def __init__(self, tag, current, workdir, cache_dir, build_arg, backend_arg):
        self.tag, self.current = tag, current
        self.workdir, self.cache_dir = workdir, cache_dir
        self.build_arg, self.backend_arg = build_arg, backend_arg
        self.fetch_release = fetch_release_body
        self.installer = None
        self.runtime = self.pins = None
        self.pinned_assets: dict = {}
        self.table: dict = {}
        self.version = ""
        self.build = self.backend = ""


def _record(receipt, name, fn, *, required=True):
    """Run *fn*, which returns (value, detail); store its outcome as check *name*.
    Returns the value, or None when the check did not pass."""
    try:
        value, detail = fn()
    except Problem as p:
        set_check(receipt, name, p.status, p.detail, required=required)
        print(f"  {name}: {p.status}: {p.detail}", flush=True)
        return None
    set_check(receipt, name, PASS, detail, required=required)
    print(f"  {name}: PASS: {detail}", flush=True)
    return True if value is None else value


def _skip_rest(receipt, names, why, required_map=None):
    for n in names:
        if n not in receipt["checks"]:
            set_check(receipt, n, SKIP, why,
                      required=(required_map or {}).get(n, n in ALWAYS_REQUIRED))


def run_checks(ctx, receipt: dict, recorder: PidRecorder) -> None:
    """Run every check in order, recording each. Never raises for a check."""
    from localm import hwdetect
    from localm.media.koboldcpp import _proc, models as kmodels, pins, runtime, server
    ctx.pins, ctx.runtime = pins, runtime
    if ctx.build_arg:
        ctx.installer = lambda backend: runtime.install(ctx.build)
    else:
        ctx.installer = runtime.ensure_for_backend
    det = hwdetect.detect()
    plat = runtime.platform_key()
    receipt["hardware"] = {
        "platform": sys.platform, "machine": platform.machine(),
        "python": sys.version.split()[0], "platform_key": plat,
        "gpu_state": det.gpu_state, "vendors": list(det.vendors),
        "gpu_names": det.gpu_names}

    ok, detail = verify_isolation(ctx.workdir)
    set_check(receipt, "isolation", PASS if ok else SKIP, detail)
    print(f"  isolation: {'PASS' if ok else 'SKIP'}: {detail}", flush=True)
    if not ok:
        _skip_rest(receipt, CHECK_NAMES, "the isolation check did not pass")
        return
    if plat is None:
        set_check(receipt, "release_assets", SKIP,
                  "KoboldCpp publishes no build for this platform")
        _skip_rest(receipt, CHECK_NAMES, "no build for this platform")
        return

    ctx.pinned_assets = dict(pins.ASSETS)
    if ctx.current:
        ctx.tag = pins.TAG
        receipt["tag"] = ctx.tag
    ctx.version = version_of_tag(ctx.tag)
    ctx.build, ctx.backend = pick_build_and_backend(runtime, det, ctx.build_arg,
                                                    ctx.backend_arg)
    receipt["hardware"].update({"build": ctx.build, "backend": ctx.backend})
    receipt["version"] = ctx.version
    print(f"  confirming KoboldCpp {ctx.tag}: build {ctx.build}, backend {ctx.backend}",
          flush=True)

    table = _record(receipt, "release_assets", lambda: _assets_check(ctx))
    if table is None:
        _skip_rest(receipt, CHECK_NAMES, "the release assets were not confirmed")
        return
    receipt["assets"] = table_to_json(ctx.table)

    gpu_present = det.gpu_state == "found"
    vulkan_required = gpu_present and ctx.backend == "vulkan"
    optional = {"vulkan_device": vulkan_required}

    with patched_pins(pins, ctx.tag, ctx.version, ctx.table):
        recorder.install(_proc)
        try:
            rt = _record(receipt, "install", lambda: check_install(ctx))
            if rt is None:
                _skip_rest(receipt, CHECK_NAMES[3:], "the install did not pass", optional)
                return
            _record(receipt, "launcher_version", lambda: check_launcher_version(ctx, rt))

            _chat_checks(ctx, receipt, rt, optional, kmodels)
            _music_checks(ctx, receipt, kmodels)
        finally:
            server.stop()
            recorder.uninstall(_proc)


def _assets_check(ctx) -> tuple:
    table = check_release_assets(ctx)
    return table, (f"{len(table)} pinned asset(s) found in the {ctx.tag} release with "
                   "digests" + ("; the pinned table equals the published one"
                                if ctx.current else ""))


def _chat_checks(ctx, receipt, rt, optional, kmodels) -> None:
    spec = MODELS["chat"]
    record: dict = {}

    try:
        model = ensure_model(spec["repo"], spec["revision"], spec["file"], spec["size"],
                             spec["sha256"], ctx.cache_dir)
        arch = kmodels.architecture_of(model)
        if arch != spec["architecture"]:
            raise Problem(SKIP, f"{spec['file']} has architecture {arch!r}, expected "
                                f"{spec['architecture']!r}; it is not the causal LM "
                                "this check needs")
    except Problem as p:
        _skip_rest(receipt, ("server_api", "text_generation", "vulkan_device"),
                   p.detail, optional)
        print(f"  chat model: SKIP: {p.detail}", flush=True)
        return
    receipt.setdefault("models", {})["chat"] = model_record(
        model, spec["repo"], spec["revision"], spec["size"], spec["sha256"])
    receipt["models"]["chat"]["architecture"] = spec["architecture"]
    try:
        run_chat_server(ctx, rt, model, record)
    except Problem as p:
        status = p.status
        if "server_api" not in record:
            set_check(receipt, "server_api", status, p.detail)
            set_check(receipt, "text_generation", SKIP, "the server check did not pass")
        else:
            set_check(receipt, "server_api", PASS, record["server_api"])
            set_check(receipt, "text_generation", status, p.detail)
        print(f"  chat server: {status}: {p.detail}", flush=True)
        _skip_rest(receipt, ("vulkan_device",), "the chat server did not complete", optional)
        return
    set_check(receipt, "server_api", PASS, record["server_api"])
    text = record["text"]
    if text_is_usable(text):
        set_check(receipt, "text_generation", PASS,
                  f"generated {len(text)} characters: {text.strip()[:80]!r}")
    else:
        set_check(receipt, "text_generation", FAIL,
                  f"generated no usable text: {text[:80]!r}")
    _vulkan_check(ctx, receipt, record.get("log_lines", []), optional)
    for n in ("server_api", "text_generation"):
        print(f"  {n}: {receipt['checks'][n]['status']}: {receipt['checks'][n]['detail']}",
              flush=True)


def _vulkan_check(ctx, receipt, lines, optional) -> None:
    required = optional.get("vulkan_device", False)
    if ctx.backend != "vulkan":
        set_check(receipt, "vulkan_device", SKIP,
                  f"the backend under test is {ctx.backend}, not vulkan", required=False)
        return
    devices = vulkan_devices_from_log(lines)
    if devices:
        set_check(receipt, "vulkan_device", PASS,
                  f"the startup log lists {len(devices)} Vulkan device(s): "
                  + "; ".join(devices)[:200], required=required)
    elif required:
        set_check(receipt, "vulkan_device", FAIL,
                  "the machine has a GPU and the vulkan backend was selected, but the "
                  "startup log lists no Vulkan device", required=True)
    else:
        set_check(receipt, "vulkan_device", SKIP,
                  "no Vulkan device in the startup log and no GPU detected", required=False)


def _music_checks(ctx, receipt, kmodels) -> None:
    music_spec = MODELS["music"]
    cfg: dict = {}
    problems = []
    for comp in kmodels.COMPONENTS:
        fname = kmodels.DEFAULT_FILES[comp]
        entry = music_spec["files"].get(fname)
        if kmodels.DEFAULT_REPO != music_spec["repo"] or entry is None:
            problems.append(f"{comp}: localm's default {kmodels.DEFAULT_REPO}/{fname} "
                            "is not in this script's MODELS table")
            continue
        try:
            path = ensure_model(music_spec["repo"], music_spec["revision"], fname,
                                entry[0], entry[1], ctx.cache_dir)
        except Problem as p:
            problems.append(f"{comp}: {p.detail}")
            continue
        cfg[comp] = str(path)
        receipt.setdefault("models", {})[f"music_{comp}"] = model_record(
            path, music_spec["repo"], music_spec["revision"], entry[0], entry[1])
    if problems:
        why = "the ACE-Step model files were not available: " + "; ".join(problems)
        set_check(receipt, "music_generate", SKIP, why)
        set_check(receipt, "music_plan", SKIP, why)
        set_check(receipt, "music_gpu", SKIP, why, required=False)
        print(f"  music: SKIP: {why}", flush=True)
        return
    _record(receipt, "music_generate", lambda: run_music(ctx, False, cfg, "cpu"))
    _record(receipt, "music_plan", lambda: run_music(ctx, True, cfg, "cpu"))
    if ctx.backend == "cpu":
        set_check(receipt, "music_gpu", SKIP, "the backend under test is cpu",
                  required=False)
    else:
        _record(receipt, "music_gpu",
                lambda: run_music(ctx, False, cfg, ctx.backend), required=False)


def cleanup_workdir(workdir: Path) -> list:
    """Remove the throwaway home and tmp dirs under *workdir*. Returns the
    problems encountered."""
    problems = []
    for sub in ("home", "tmp"):
        target = workdir / sub
        if not target.exists():
            continue
        for _ in range(5):
            shutil.rmtree(target, ignore_errors=True)
            if not target.exists():
                break
            time.sleep(2.0)
        if target.exists():
            problems.append(str(target))
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--tag", help="the candidate release tag, e.g. v1.123")
    which.add_argument("--current", action="store_true",
                       help="confirm the build the repo pins today")
    ap.add_argument("--workdir", required=True,
                    help="scratch directory; LOCALM_HOME and TMP live under it")
    ap.add_argument("--receipt", required=True, help="where the JSON receipt is written")
    ap.add_argument("--keep", action="store_true",
                    help="keep <workdir>/home and <workdir>/tmp afterwards")
    ap.add_argument("--cache-dir", default=None,
                    help="persistent model cache (default: $LOCALM_PIN_CACHE/koboldcpp, "
                         "else ~/.cache/localm-pin-cache/koboldcpp)")
    ap.add_argument("--build", default=None, choices=("nocuda", "cuda", "metal"),
                    help="build to install (default: the one this machine would use)")
    ap.add_argument("--backend", default=None, choices=("vulkan", "cuda", "cpu", "metal"),
                    help="backend to run (default: the one this machine would use)")
    args = ap.parse_args(argv)

    receipt_path = Path(args.receipt)
    tag = args.tag.strip() if args.tag else "(pinned)"
    receipt = new_receipt(tag, bool(args.current))
    if args.tag and parse_tag(tag) is None:
        set_check(receipt, "release_assets", SKIP, f"{tag!r} is not a vX.Y[.Z] release tag")
        rc = finalize(receipt)
        save_receipt(receipt_path, receipt)
        print(f"INCONCLUSIVE: {tag!r} is not a vX.Y[.Z] release tag")
        return rc

    workdir = Path(args.workdir).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else default_cache_dir()
    save_receipt(receipt_path, receipt)
    recorder = PidRecorder()
    killed: list = []
    try:
        prepare_env(workdir, cache_dir)
        ctx = Ctx(tag, bool(args.current), workdir, cache_dir, args.build, args.backend)
        print(f"Confirming KoboldCpp {tag} in {workdir}", flush=True)
        run_checks(ctx, receipt, recorder)
        rc = finalize(receipt)
    except Exception as e:
        traceback.print_exc()
        receipt["verdict"] = INCONCLUSIVE
        receipt["why"] = f"unexpected exception: {type(e).__name__}: {e}"
        rc = 2
    finally:
        killed = recorder.reap()
        if killed:
            receipt["killed_pids"] = killed
        if not args.keep:
            left = cleanup_workdir(workdir)
            if left:
                receipt["cleanup_problems"] = left
    save_receipt(receipt_path, receipt)
    print(f"\n{receipt['verdict']}: {receipt['why']}")
    for name in CHECK_NAMES:
        c = receipt["checks"].get(name)
        if c:
            print(f"  {name:17s} {c['status']:5s} {'(required)' if c['required'] else ''}")
    print(f"receipt written: {receipt_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
