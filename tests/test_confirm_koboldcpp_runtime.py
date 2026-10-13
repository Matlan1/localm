# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/confirm_koboldcpp_runtime.py: the checks that decide whether a KoboldCpp
release works with localm, without the network, a GPU or the real binary.

The pieces that can be exercised for real are: the release-listing parser, the
verdict arithmetic and receipt, the candidate-pin patching of localm's own pin
module, the install classification, the launcher version check against a real
launcher script, the chat server flow through localm's own process wrapper and
server class against a stand-in HTTP server, the PID bookkeeping with real
processes, the model cache, and the WAV analysis with generated audio.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import os
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_CONFIRM = _ROOT / "scripts" / "confirm_koboldcpp_runtime.py"


@pytest.fixture(scope="module")
def cf():
    spec = importlib.util.spec_from_file_location("confirm_koboldcpp_under_test", _CONFIRM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _h(n: int) -> str:
    return f"{n:x}".rjust(64, "0")


def _asset(name, size=10, sha=None):
    return {"name": name, "size": size, "digest": "sha256:" + (sha or _h(1))}


# --------------------------------------------------------------------------- #
#  Tags and release listing                                                    #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tag,parsed", [
    ("v1.122.1", (1, 122, 1)), ("v1.9", (1, 9)), ("v10.0.0.1", (10, 0, 0, 1))])
def test_parse_tag(cf, tag, parsed):
    assert cf.parse_tag(tag) == parsed
    assert cf.version_of_tag(tag) == tag[1:]


@pytest.mark.parametrize("tag", ["1.2", "v1", "v1.2-rc", "V1.2", "", "v1.2.", "vx.y"])
def test_parse_tag_rejects(cf, tag):
    assert cf.parse_tag(tag) is None
    with pytest.raises(ValueError):
        cf.version_of_tag(tag)


def test_parse_release_assets(cf):
    got = cf.parse_release_assets({"assets": [_asset("a", 5, _h(0xAB)), _asset("b", 7)]})
    assert got == {"a": (5, _h(0xAB)), "b": (7, _h(1))}


@pytest.mark.parametrize("body", [
    None, [], {}, {"assets": []}, {"assets": "x"}, {"assets": [None]},
    {"assets": [{"size": 1, "digest": "sha256:" + _h(1)}]},
    {"assets": [{"name": "a", "size": 1}]},
    {"assets": [{"name": "a", "size": 1, "digest": "sha1:" + "a" * 40}]},
    {"assets": [{"name": "a", "size": 1, "digest": "sha256:" + "g" * 64}]},
    {"assets": [{"name": "a", "size": 1, "digest": "sha256:" + "a" * 63}]},
    {"assets": [{"name": "a", "size": 0, "digest": "sha256:" + _h(1)}]},
    {"assets": [{"name": "a", "size": True, "digest": "sha256:" + _h(1)}]},
    {"assets": [{"name": "a", "size": "1", "digest": "sha256:" + _h(1)}]},
    {"assets": [_asset("a"), {"name": "b", "size": 1, "digest": None}]},
])
def test_parse_release_assets_rejects(cf, body):
    with pytest.raises(ValueError):
        cf.parse_release_assets(body)


def test_build_table_takes_names_from_the_pin_and_numbers_from_the_release(cf):
    pinned = {("windows", "nocuda"): ("w.exe", 1, _h(1)),
              ("linux", "cuda"): ("l", 2, _h(2))}
    published = {"w.exe": (11, _h(0x11)), "l": (22, _h(0x22)), "other": (3, _h(3))}
    assert cf.build_table(pinned, published) == {
        ("windows", "nocuda"): ("w.exe", 11, _h(0x11)),
        ("linux", "cuda"): ("l", 22, _h(0x22))}


def test_build_table_names_every_missing_asset(cf):
    pinned = {("windows", "nocuda"): ("w.exe", 1, _h(1)), ("linux", "cuda"): ("l", 2, _h(2))}
    with pytest.raises(ValueError, match=r"linux/cuda:l.*windows/nocuda:w\.exe"
                                         r"|windows/nocuda:w\.exe.*linux/cuda:l"):
        cf.build_table(pinned, {})


def test_table_json_round_trip_and_rejects_malformed(cf):
    table = {("windows", "nocuda"): ("w.exe", 1, _h(1)), ("macos-arm64", "metal"): ("m", 2, _h(2))}
    assert cf.table_from_json(cf.table_to_json(table)) == table
    for bad in (None, {}, [], {"windows": {}}, {"a/b": []},
                {"a/b": {"name": "x", "size": True, "sha256": _h(1)}},
                {"a/b": {"name": "x", "size": 1, "sha256": "zz"}},
                {"a/b": {"name": 3, "size": 1, "sha256": _h(1)}}):
        with pytest.raises(ValueError):
            cf.table_from_json(bad)


class _Resp:
    def __init__(self, raw):
        self.raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.raw


def test_the_release_request_carries_the_environment_token(cf, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok-env")
    seen = []

    def opener(req, timeout=None):
        seen.append(req)
        return _Resp(b'{"assets": []}')
    assert cf.fetch_release_body("v1.2", opener=opener) == {"assets": []}
    assert seen[0].get_header("Authorization") == "Bearer tok-env"
    assert seen[0].full_url.endswith("/repos/LostRuins/koboldcpp/releases/tags/v1.2")


def test_without_an_environment_token_the_gh_token_is_used(cf, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="tok-gh\n", stderr="")
    monkeypatch.setattr(cf.subprocess, "run", fake_run)
    assert cf.github_token() == "tok-gh" and calls == [["gh", "auth", "token"]]


@pytest.mark.parametrize("outcome", ["missing", "failed", "empty"])
def test_no_token_anywhere_means_an_anonymous_request(cf, monkeypatch, outcome):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def fake_run(cmd, **kw):
        if outcome == "missing":
            raise FileNotFoundError("gh")
        return subprocess.CompletedProcess(cmd, 1 if outcome == "failed" else 0,
                                           stdout="", stderr="")
    monkeypatch.setattr(cf.subprocess, "run", fake_run)
    assert cf.github_token() is None
    seen = []
    cf.fetch_release_body("v1.2", opener=lambda req, timeout=None:
                          (seen.append(req), _Resp(b"{}"))[1])
    assert seen[0].get_header("Authorization") is None


# --------------------------------------------------------------------------- #
#  Verdict and receipt                                                         #
# --------------------------------------------------------------------------- #

def _all_pass(cf, **override):
    r = cf.new_receipt("v1.2", False)
    for n in cf.CHECK_NAMES:
        cf.set_check(r, n, "PASS", "ok", required=n in cf.ALWAYS_REQUIRED)
    for n, (status, required) in override.items():
        cf.set_check(r, n, status, f"{n} is {status}", required=required)
    return r


def test_every_required_pass_is_PASS(cf):
    r = _all_pass(cf)
    assert cf.finalize(r) == 0 and r["verdict"] == "PASS"


@pytest.mark.parametrize("name", ["install", "server_api", "music_generate", "music_plan",
                                  "text_generation", "launcher_version", "isolation",
                                  "release_assets"])
def test_a_skipped_required_check_is_INCONCLUSIVE_never_PASS(cf, name):
    r = _all_pass(cf, **{name: ("SKIP", True)})
    assert cf.finalize(r) == 2
    assert r["verdict"] == "INCONCLUSIVE" and name in r["why"]


def test_a_failed_check_is_FAIL_even_beside_a_skip(cf):
    r = _all_pass(cf, install=("FAIL", True), music_plan=("SKIP", True))
    assert cf.finalize(r) == 1 and r["verdict"] == "FAIL"
    assert "install" in r["why"]


def test_a_failed_optional_check_still_fails_the_run(cf):
    r = _all_pass(cf, vulkan_device=("FAIL", False))
    assert cf.finalize(r) == 1


def test_a_failed_advisory_check_is_reported_but_does_not_decide_the_verdict(cf):
    r = _all_pass(cf, music_gpu=("FAIL", False))
    assert cf.finalize(r) == 0 and r["verdict"] == "PASS"
    assert "advisory check failed" in r["why"] and "music_gpu" in r["why"]


def test_a_failed_advisory_check_never_hides_a_real_failure(cf):
    r = _all_pass(cf, music_gpu=("FAIL", False), music_plan=("FAIL", True))
    assert cf.finalize(r) == 1 and "music_plan" in r["why"]


def test_only_music_on_the_gpu_is_advisory(cf):
    assert cf.ADVISORY_CHECKS == ("music_gpu",)
    assert not set(cf.ADVISORY_CHECKS) & set(cf.ALWAYS_REQUIRED)


def test_a_skipped_optional_check_does_not_block_PASS(cf):
    r = _all_pass(cf, vulkan_device=("SKIP", False))
    assert cf.finalize(r) == 0


def test_a_required_vulkan_check_that_skipped_blocks_PASS(cf):
    r = _all_pass(cf, vulkan_device=("SKIP", True))
    assert cf.finalize(r) == 2


def test_a_required_check_absent_from_the_run_is_INCONCLUSIVE(cf):
    r = _all_pass(cf)
    del r["checks"]["music_plan"]
    assert cf.finalize(r) == 2 and "music_plan" in r["why"]


def test_the_receipt_is_written_atomically_with_the_contract_fields(cf, tmp_path):
    r = _all_pass(cf)
    cf.finalize(r)
    path = tmp_path / "sub" / "r.json"
    cf.save_receipt(path, r)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert {"schema", "component", "tag", "current", "verdict", "why", "written_at",
            "hardware", "checks"} <= set(data)
    assert data["schema"] == 1 and data["component"] == "koboldcpp"
    assert data["written_at"].endswith("Z")
    assert data["checks"]["install"] == {"status": "PASS", "required": True, "detail": "ok"}
    assert [p.name for p in path.parent.iterdir()] == ["r.json"]


def test_a_new_receipt_reads_as_not_confirmed(cf):
    r = cf.new_receipt("v1.2", True)
    assert r["verdict"] == "INCONCLUSIVE" and r["current"] is True
    assert r["not_measured"]


# --------------------------------------------------------------------------- #
#  Candidate pins                                                              #
# --------------------------------------------------------------------------- #

@pytest.fixture
def kcpp_runtime(tmp_path, monkeypatch):
    from localm.media.koboldcpp import pins, runtime
    monkeypatch.setattr(runtime, "platform_key", lambda: "windows")
    monkeypatch.setattr(runtime, "runtimes_root", lambda: tmp_path / "runtimes")
    return pins, runtime


def test_patched_pins_make_localms_own_installer_target_the_candidate(cf, kcpp_runtime):
    pins, runtime = kcpp_runtime
    original = (pins.TAG, pins.VERSION, dict(pins.ASSETS), pins.asset_url("x"))
    table = {("windows", "nocuda"): ("kcpp-nocuda.exe", 5, _h(5))}
    with cf.patched_pins(pins, "v9.9", "9.9", table):
        assert pins.asset_url("kcpp-nocuda.exe") == (
            "https://github.com/LostRuins/koboldcpp/releases/download/v9.9/kcpp-nocuda.exe")
        assert runtime.runtime_dir("nocuda").name == "v9.9-nocuda"
        assert pins.ASSETS == table and runtime.available_builds("windows") == ["nocuda"]
        cf.prove_pins_applied(pins, runtime, "v9.9", "nocuda", "kcpp-nocuda.exe")
    assert (pins.TAG, pins.VERSION, dict(pins.ASSETS), pins.asset_url("x")) == original


def test_pins_are_restored_when_the_block_raises(cf, kcpp_runtime):
    pins, _ = kcpp_runtime
    original = (pins.TAG, dict(pins.ASSETS))
    with pytest.raises(RuntimeError):
        with cf.patched_pins(pins, "v9.9", "9.9", {("windows", "nocuda"): ("x", 1, _h(1))}):
            raise RuntimeError("boom")
    assert (pins.TAG, dict(pins.ASSETS)) == original


def test_prove_pins_applied_refuses_when_the_pins_are_not_the_candidate(cf, kcpp_runtime):
    pins, runtime = kcpp_runtime
    with pytest.raises(cf.Problem) as e:
        cf.prove_pins_applied(pins, runtime, "v9.9", "nocuda", "koboldcpp-nocuda.exe")
    assert e.value.status == "SKIP" and "did not take effect" in e.value.detail


# --------------------------------------------------------------------------- #
#  Isolation                                                                   #
# --------------------------------------------------------------------------- #

def test_prepare_env_points_localm_and_temp_at_the_scratch_dir(cf, tmp_path, monkeypatch):
    for k in ("LOCALM_HOME", "TEMP", "TMP", "TMPDIR", "HF_HOME", "HF_HUB_DISABLE_PROGRESS_BARS"):
        monkeypatch.setenv(k, "old")
    work, cache = tmp_path / "w", tmp_path / "cache"
    home, tmp = cf.prepare_env(work, cache)
    assert os.environ["LOCALM_HOME"] == str(home) == str(work / "home")
    assert {os.environ[k] for k in ("TEMP", "TMP", "TMPDIR")} == {str(tmp)}
    assert os.environ["HF_HOME"].startswith(str(cache))
    assert home.is_dir() and tmp.is_dir() and cache.is_dir()


def _patch_localm_paths(monkeypatch, root: Path, **overrides):
    import tempfile

    import localm
    import localm.config
    from localm.media.koboldcpp import runtime
    paths = {"HOME_DIR": root / "home", "home_dir": root / "home",
             "runtimes_root": root / "home" / "runtimes", "tempdir": root / "tmp"}
    paths.update(overrides)
    monkeypatch.setattr(localm.config, "HOME_DIR", paths["HOME_DIR"])
    monkeypatch.setattr(localm.config, "home_dir", lambda: paths["home_dir"])
    monkeypatch.setattr(runtime, "runtimes_root", lambda: paths["runtimes_root"])
    monkeypatch.setattr(tempfile, "tempdir", str(paths["tempdir"]))
    return localm


def test_isolation_passes_when_every_path_is_under_the_scratch_dir(cf, tmp_path, monkeypatch):
    localm = _patch_localm_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(cf, "REPO", Path(localm.__file__).resolve().parent.parent)
    ok, detail = cf.verify_isolation(tmp_path)
    assert ok, detail


@pytest.mark.parametrize("which", ["HOME_DIR", "home_dir", "runtimes_root", "tempdir"])
def test_isolation_fails_when_any_one_path_escapes(cf, tmp_path, monkeypatch, which):
    elsewhere = tmp_path.parent / (tmp_path.name + "-elsewhere")
    localm = _patch_localm_paths(monkeypatch, tmp_path, **{which: elsewhere})
    monkeypatch.setattr(cf, "REPO", Path(localm.__file__).resolve().parent.parent)
    ok, detail = cf.verify_isolation(tmp_path)
    assert not ok and which in detail


def test_isolation_fails_when_localm_is_not_this_checkouts(cf, tmp_path, monkeypatch):
    _patch_localm_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(cf, "REPO", tmp_path / "some-other-checkout")
    ok, detail = cf.verify_isolation(tmp_path)
    assert not ok and "not from this checkout" in detail


# --------------------------------------------------------------------------- #
#  Process bookkeeping, with real processes                                    #
# --------------------------------------------------------------------------- #

def _alive(pid: int) -> bool:
    import psutil
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _wait_dead(pid: int, seconds: float = 15.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not _alive(pid):
            return True
        time.sleep(0.2)
    return False


_PARENT_CODE = (
    "import subprocess, sys, time\n"
    "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
    "open(sys.argv[1], 'w').write(str(c.pid))\n"
    "time.sleep(120)\n")


def _spawn_tree(tmp_path: Path):
    marker = tmp_path / "child.pid"
    proc = subprocess.Popen([sys.executable, "-c", _PARENT_CODE, str(marker)])
    end = time.monotonic() + 15
    while not marker.exists() and time.monotonic() < end:
        time.sleep(0.1)
    time.sleep(0.2)
    return proc, int(marker.read_text())


def test_the_recorder_kills_a_recorded_process_tree(cf, tmp_path):
    procs = []

    def fake_start(argv, *, cwd, env):
        p, child = _spawn_tree(tmp_path)
        procs.append((p, child))
        return SimpleNamespace(pid=p.pid)
    fake_proc = SimpleNamespace(start=fake_start)
    rec = cf.PidRecorder()
    rec.install(fake_proc)
    try:
        fake_proc.start(["x"], cwd=str(tmp_path), env={})
        parent, child = procs[0]
        assert _alive(parent.pid) and _alive(child)
        killed = rec.reap()
        assert killed == [parent.pid]
        assert _wait_dead(parent.pid) and _wait_dead(child)
    finally:
        rec.uninstall(fake_proc)
        for p, c in procs:
            p.kill()
            p.wait()
            if _alive(c):
                import psutil
                psutil.Process(c).kill()
    assert fake_proc.start is fake_start


def test_the_recorder_never_kills_a_pid_it_did_not_start(cf, tmp_path):
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        rec = cf.PidRecorder()
        rec.entries.append((victim.pid, 12345.0))
        assert rec.reap() == []
        assert _alive(victim.pid)
        rec.entries.append((victim.pid, cf._start_time(victim.pid)))
        assert rec.reap() == [victim.pid]
        assert _wait_dead(victim.pid)
    finally:
        victim.kill()
        victim.wait()


def test_the_recorder_skips_processes_that_already_exited(cf):
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    start = cf._start_time(done.pid)
    done.wait()
    rec = cf.PidRecorder()
    rec.entries.append((done.pid, start))
    assert rec.reap() == []


def test_the_recorder_wraps_and_restores_the_real_process_wrapper(cf):
    from localm.media.koboldcpp import _proc
    original = _proc.start
    rec = cf.PidRecorder()
    rec.install(_proc)
    assert _proc.start is not original
    rec.uninstall(_proc)
    assert _proc.start is original


# --------------------------------------------------------------------------- #
#  Model cache                                                                 #
# --------------------------------------------------------------------------- #

def _fake_downloader(content: bytes, calls: list):
    def dl(*, repo_id, filename, revision, local_dir):
        calls.append((repo_id, filename, revision))
        p = Path(local_dir) / filename
        p.write_bytes(content)
        return str(p)
    return dl


def _meta(content: bytes):
    return len(content), hashlib.sha256(content).hexdigest()


def test_ensure_model_downloads_once_and_then_serves_from_the_cache(cf, tmp_path):
    body = b"gguf-ish" * 100
    size, sha = _meta(body)
    calls: list = []
    p1 = cf.ensure_model("o/r", "rev1", "m.gguf", size, sha, tmp_path,
                         downloader=_fake_downloader(body, calls))
    p2 = cf.ensure_model("o/r", "rev1", "m.gguf", size, sha, tmp_path,
                         downloader=_fake_downloader(b"WRONG", calls))
    assert p1 == p2 and p1.read_bytes() == body
    assert calls == [("o/r", "m.gguf", "rev1")]
    assert not (tmp_path / ".download.lock").exists()


def test_a_corrupt_cached_file_is_discarded_and_fetched_again(cf, tmp_path):
    body = b"good" * 50
    size, sha = _meta(body)
    cache = tmp_path / "models" / "o__r"
    cache.mkdir(parents=True)
    (cache / "m.gguf").write_bytes(b"bad" * 50)
    calls: list = []
    p = cf.ensure_model("o/r", "rev", "m.gguf", size, sha, tmp_path,
                        downloader=_fake_downloader(body, calls))
    assert p.read_bytes() == body and len(calls) == 1


def test_a_download_that_does_not_match_is_skipped_and_removed(cf, tmp_path):
    size, sha = _meta(b"expected")
    with pytest.raises(cf.Problem) as e:
        cf.ensure_model("o/r", "rev", "m.gguf", size, sha, tmp_path,
                        downloader=_fake_downloader(b"different", []))
    assert e.value.status == "SKIP" and "does not match" in e.value.detail
    assert not (tmp_path / "models" / "o__r" / "m.gguf").exists()


def test_a_failed_download_is_skipped_not_failed(cf, tmp_path):
    def broken(**kw):
        raise OSError("offline")
    with pytest.raises(cf.Problem) as e:
        cf.ensure_model("o/r", "rev", "m.gguf", 1, _h(1), tmp_path, downloader=broken)
    assert e.value.status == "SKIP" and "offline" in e.value.detail
    assert not (tmp_path / ".download.lock").exists()


def test_a_stale_cache_lock_is_taken_over(cf, tmp_path):
    lock = tmp_path / ".download.lock"
    lock.mkdir()
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    (lock / "pid").write_text(str(done.pid), encoding="utf-8")
    body = b"x" * 10
    size, sha = _meta(body)
    p = cf.ensure_model("o/r", "r", "m.gguf", size, sha, tmp_path,
                        downloader=_fake_downloader(body, []))
    assert p.is_file()


def test_the_model_table_is_complete_and_matches_localms_defaults(cf):
    from localm.media.koboldcpp import models
    music = cf.MODELS["music"]
    assert music["repo"] == models.DEFAULT_REPO
    for comp in models.COMPONENTS:
        size, sha = music["files"][models.DEFAULT_FILES[comp]]
        assert size == models.DEFAULT_SIZES[comp] and len(sha) == 64
    assert set(music["files"]) == set(models.DEFAULT_FILES.values())
    chat = cf.MODELS["chat"]
    assert len(chat["revision"]) == 40 and chat["architecture"] == "llama"


def test_the_default_cache_dir_follows_the_environment(cf, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_PIN_CACHE", str(tmp_path))
    assert cf.default_cache_dir() == tmp_path / "koboldcpp"


def test_the_default_cache_dir_finds_the_nearest_ancestor_cache(cf, tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALM_PIN_CACHE", raising=False)
    repo = tmp_path / "a" / "b" / "repo"
    repo.mkdir(parents=True)
    (tmp_path / ".claude" / "pin-cache").mkdir(parents=True)
    (tmp_path / "a" / ".claude" / "pin-cache").mkdir(parents=True)
    monkeypatch.setattr(cf, "REPO", repo)
    assert cf.default_cache_dir() == tmp_path / "a" / ".claude" / "pin-cache" / "koboldcpp"


def test_the_default_cache_dir_falls_back_to_the_home_cache(cf, tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALM_PIN_CACHE", raising=False)
    monkeypatch.setattr(cf, "REPO", tmp_path / "nowhere" / "repo")
    real_is_dir = cf.Path.is_dir
    monkeypatch.setattr(cf.Path, "is_dir",
                        lambda self, *a, **k: False if self.name == "pin-cache"
                        else real_is_dir(self, *a, **k))
    monkeypatch.setattr(cf.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    assert cf.default_cache_dir() == tmp_path / "home" / ".cache" / "localm-pin-cache" / "koboldcpp"


# --------------------------------------------------------------------------- #
#  WAV analysis                                                                #
# --------------------------------------------------------------------------- #

def make_wav(seconds=5.0, rate=44100, channels=2, width=2, amp=0.5, kind="sine") -> bytes:
    n = int(seconds * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        full = (1 << (8 * width - 1)) - 1
        frames = bytearray()
        for i in range(n):
            if kind == "sine":
                v = int(amp * full * math.sin(2 * math.pi * 440 * i / rate))
            elif kind == "dc":
                v = int(amp * full)
            else:
                v = 0
            for _ in range(channels):
                if width == 2:
                    frames += struct.pack("<h", v)
                elif width == 1:
                    frames += bytes([(v >> 0) + 128 & 0xFF])
                else:
                    frames += int(v).to_bytes(width, "little", signed=True)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def test_a_tone_of_the_requested_length_passes(cf):
    ok, detail = cf.analyze_wav(make_wav(5.0), 5.0, (0.4, 1.6))
    assert ok, detail
    assert "5.00 s" in detail and "44100 Hz" in detail


@pytest.mark.parametrize("width", [1, 3, 4])
def test_other_pcm_widths_are_read(cf, width):
    ok, detail = cf.analyze_wav(make_wav(1.0, rate=8000, width=width), 1.0, (0.4, 1.6))
    assert ok, detail


@pytest.mark.parametrize("kind", ["silence", "dc"])
def test_silence_and_a_constant_are_not_music(cf, kind):
    ok, detail = cf.analyze_wav(make_wav(2.0, kind=kind), 2.0, (0.4, 1.6))
    assert not ok and "silent or constant" in detail


def test_audio_of_the_wrong_length_fails(cf):
    assert not cf.analyze_wav(make_wav(0.5), 5.0, (0.4, 1.6))[0]
    assert not cf.analyze_wav(make_wav(12.0, rate=8000), 5.0, (0.4, 1.6))[0]


@pytest.mark.parametrize("data", [b"", b"RIFF", b"not a wav at all", b"\x00" * 100])
def test_bytes_that_are_not_a_wav_fail(cf, data):
    ok, detail = cf.analyze_wav(data, 5.0, (0.4, 1.6))
    assert not ok and "not a readable PCM WAV" in detail


def test_an_empty_track_fails(cf):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
    ok, detail = cf.analyze_wav(buf.getvalue(), 5.0, (0.4, 1.6))
    assert not ok and "empty" in detail


@pytest.mark.parametrize("text,usable", [("blue", True), ("  Blue sky. ", True), ("", False),
                                         ("12 .. 34", False), ("a b", False)])
def test_text_is_usable(cf, text, usable):
    assert cf.text_is_usable(text) is usable


# --------------------------------------------------------------------------- #
#  Vulkan log                                                                  #
# --------------------------------------------------------------------------- #

def test_vulkan_devices_are_read_from_the_startup_log(cf):
    lines = ["Welcome", "ggml_vulkan: Found 1 Vulkan devices:",
             "ggml_vulkan: 0 = AMD Radeon RX 6900 XT (AMD proprietary driver) | uma: 0 | fp16: 1",
             "other"]
    assert cf.vulkan_devices_from_log(lines) == [
        "AMD Radeon RX 6900 XT (AMD proprietary driver) | uma: 0 | fp16: 1"]


def test_device_lines_without_the_found_header_are_not_counted(cf):
    assert cf.vulkan_devices_from_log(["ggml_vulkan: 0 = Something"]) == []
    assert cf.vulkan_devices_from_log([]) == []


# --------------------------------------------------------------------------- #
#  Install classification                                                      #
# --------------------------------------------------------------------------- #

def _ctx(cf, tmp_path, runtime, pins, installer, build="nocuda"):
    ctx = cf.Ctx("v9.9", False, tmp_path, tmp_path / "cache", None, None)
    ctx.runtime, ctx.pins = runtime, pins
    ctx.build, ctx.backend, ctx.version = build, "vulkan", "9.9"
    ctx.table = {("windows", build): ("kcpp.exe", 5, _h(5))}
    ctx.installer = installer
    return ctx


def _run_install(cf, kcpp_runtime, tmp_path, installer):
    pins, runtime = kcpp_runtime
    ctx = _ctx(cf, tmp_path, runtime, pins, installer)
    with cf.patched_pins(pins, "v9.9", "9.9", ctx.table):
        return cf.check_install(ctx), ctx


def _runtime_obj(runtime, tmp_path, build="nocuda"):
    return runtime.Runtime(build=build, path=tmp_path / "rt", launcher=tmp_path / "rt" / "l")


def test_a_clean_install_passes(cf, kcpp_runtime, tmp_path):
    _, runtime = kcpp_runtime
    (rt, detail), _ = _run_install(cf, kcpp_runtime, tmp_path,
                                   lambda backend: _runtime_obj(runtime, tmp_path))
    assert rt.build == "nocuda" and "verified" in detail and "v9.9-nocuda" in detail


def test_a_build_that_will_not_unpack_or_report_its_version_fails(cf, kcpp_runtime, tmp_path):
    _, runtime = kcpp_runtime

    def installer(backend):
        raise runtime.ProvisionError("the unpacked KoboldCpp reports version 1.0, expected 9.9")
    with pytest.raises(cf.Problem) as e:
        _run_install(cf, kcpp_runtime, tmp_path, installer)
    assert e.value.status == "FAIL" and "reports version" in e.value.detail


def test_an_asset_that_never_matches_the_published_digest_fails(cf, kcpp_runtime, tmp_path):
    _, runtime = kcpp_runtime
    f = tmp_path / "asset.bin"
    f.write_bytes(b"abc")
    tries = []

    def installer(backend):
        tries.append(1)
        runtime.verify_file(f, 3, _h(9), "kcpp.exe")
    with pytest.raises(cf.Problem) as e:
        _run_install(cf, kcpp_runtime, tmp_path, installer)
    assert e.value.status == "FAIL" and "published digest" in e.value.detail
    assert len(tries) == 2


def test_a_dropped_download_is_inconclusive_not_a_failure(cf, kcpp_runtime, tmp_path):
    _, runtime = kcpp_runtime

    def installer(backend):
        raise runtime.DownloadError("the download of kcpp.exe stalled after 0 bytes")
    with pytest.raises(cf.Problem) as e:
        _run_install(cf, kcpp_runtime, tmp_path, installer)
    assert e.value.status == "SKIP" and "did not complete" in e.value.detail


def test_a_transient_drop_is_retried_once(cf, kcpp_runtime, tmp_path):
    _, runtime = kcpp_runtime
    state = {"n": 0}

    def installer(backend):
        state["n"] += 1
        if state["n"] == 1:
            raise runtime.DownloadError("connection reset")
        return _runtime_obj(runtime, tmp_path)
    (rt, _), _ = _run_install(cf, kcpp_runtime, tmp_path, installer)
    assert state["n"] == 2 and rt.build == "nocuda"


def test_a_truncated_download_that_then_verifies_passes(cf, kcpp_runtime, tmp_path):
    _, runtime = kcpp_runtime
    f = tmp_path / "asset.bin"
    f.write_bytes(b"abc")
    state = {"n": 0}

    def installer(backend):
        state["n"] += 1
        good = hashlib.sha256(b"abc").hexdigest()
        runtime.verify_file(f, 3 if state["n"] == 2 else 99,
                            good if state["n"] == 2 else _h(1), "kcpp.exe")
        return _runtime_obj(runtime, tmp_path)
    (rt, _), _ = _run_install(cf, kcpp_runtime, tmp_path, installer)
    assert state["n"] == 2


def test_an_os_error_is_inconclusive(cf, kcpp_runtime, tmp_path):
    def installer(backend):
        raise PermissionError("denied")
    with pytest.raises(cf.Problem) as e:
        _run_install(cf, kcpp_runtime, tmp_path, installer)
    assert e.value.status == "SKIP"


def test_the_wrong_build_coming_back_is_inconclusive(cf, kcpp_runtime, tmp_path):
    _, runtime = kcpp_runtime
    with pytest.raises(cf.Problem) as e:
        _run_install(cf, kcpp_runtime, tmp_path,
                     lambda backend: _runtime_obj(runtime, tmp_path, build="cuda"))
    assert e.value.status == "SKIP" and "not the nocuda build" in e.value.detail


def test_the_verify_wrapper_is_removed_afterwards(cf, kcpp_runtime, tmp_path):
    _, runtime = kcpp_runtime
    original = runtime.verify_file
    _run_install(cf, kcpp_runtime, tmp_path, lambda backend: _runtime_obj(runtime, tmp_path))
    assert runtime.verify_file is original


def test_an_install_check_does_not_run_when_the_pins_are_not_the_candidate(
        cf, kcpp_runtime, tmp_path):
    pins, runtime = kcpp_runtime
    ctx = _ctx(cf, tmp_path, runtime, pins, lambda b: pytest.fail("installer was called"))
    with pytest.raises(cf.Problem) as e:
        cf.check_install(ctx)
    assert e.value.status == "SKIP" and "did not take effect" in e.value.detail


# --------------------------------------------------------------------------- #
#  Launcher version, against a real launcher script                            #
# --------------------------------------------------------------------------- #

def make_launcher(directory: Path, name: str, script_lines: str) -> Path:
    """An executable stand-in for koboldcpp-launcher that runs a Python script."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / f"{name}.py"
    script.write_text(script_lines, encoding="utf-8")
    if sys.platform == "win32":
        launcher = directory / f"{name}.bat"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        launcher = directory / name
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n',
                            encoding="utf-8")
        launcher.chmod(0o755)
    return launcher


def _installed_rt(cf, runtime, tmp_path, printed: str, marker_overrides=None, build="nocuda"):
    rtdir = tmp_path / "rt"
    launcher = make_launcher(rtdir, "launcher", f"print('banner')\nprint({printed!r})\n")
    marker = {"tag": "v9.9", "version": "9.9", "build": build, "asset": "kcpp.exe",
              "sha256": _h(5)}
    marker.update(marker_overrides or {})
    (rtdir / runtime.MARKER).write_text(json.dumps(marker), encoding="utf-8")
    return runtime.Runtime(build=build, path=rtdir, launcher=launcher)


def test_launcher_version_and_marker_must_name_the_candidate(cf, kcpp_runtime, tmp_path):
    pins, runtime = kcpp_runtime
    ctx = _ctx(cf, tmp_path, runtime, pins, None)
    rt = _installed_rt(cf, runtime, tmp_path, "9.9")
    _, detail = cf.check_launcher_version(ctx, rt)
    assert "9.9" in detail and "v9.9" in detail


def test_a_launcher_reporting_another_version_fails(cf, kcpp_runtime, tmp_path):
    pins, runtime = kcpp_runtime
    ctx = _ctx(cf, tmp_path, runtime, pins, None)
    rt = _installed_rt(cf, runtime, tmp_path, "9.8")
    with pytest.raises(cf.Problem) as e:
        cf.check_launcher_version(ctx, rt)
    assert e.value.status == "FAIL" and "9.8" in e.value.detail


@pytest.mark.parametrize("override", [{"tag": "v9.8"}, {"asset": "other.exe"},
                                      {"sha256": _h(6)}, {"version": "9.8"}])
def test_a_marker_naming_a_different_release_fails(cf, kcpp_runtime, tmp_path, override):
    pins, runtime = kcpp_runtime
    ctx = _ctx(cf, tmp_path, runtime, pins, None)
    rt = _installed_rt(cf, runtime, tmp_path, "9.9", override)
    with pytest.raises(cf.Problem) as e:
        cf.check_launcher_version(ctx, rt)
    assert e.value.status == "FAIL" and "install marker" in e.value.detail


def test_a_launcher_that_cannot_start_fails(cf, kcpp_runtime, tmp_path):
    pins, runtime = kcpp_runtime
    ctx = _ctx(cf, tmp_path, runtime, pins, None)
    rt = _installed_rt(cf, runtime, tmp_path, "9.9")
    rt = runtime.Runtime(build="nocuda", path=rt.path, launcher=tmp_path / "absent-launcher")
    with pytest.raises(cf.Problem) as e:
        cf.check_launcher_version(ctx, rt)
    assert e.value.status == "FAIL"


# --------------------------------------------------------------------------- #
#  The chat server flow, through localm's own wrapper and server class         #
# --------------------------------------------------------------------------- #

FAKE_SERVER = r'''
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

mode = os.environ.get("FAKE_MODE", "ok")
argv_file = os.environ.get("FAKE_ARGV_FILE")
if argv_file:
    open(argv_file, "w").write(json.dumps(sys.argv[1:]))
if mode == "exit":
    print("could not load the model", flush=True)
    sys.exit(3)
port = int(sys.argv[sys.argv.index("--port") + 1])
print("ggml_vulkan: Found 1 Vulkan devices:", flush=True)
print("ggml_vulkan: 0 = Fake GPU 9000 | uma: 0", flush=True)
password = os.environ["KCPP_PASSWORD"]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/api/extra/version":
            self._send(200, {"result": "KoboldCpp", "version": "9.9", "music": False})
        else:
            self._send(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.headers.get("Authorization") != "Bearer " + password:
            return self._send(401, {"error": "unauthorized"})
        if self.path == "/api/v1/generate":
            if mode == "badgen":
                return self._send(200, {"nope": 1})
            if mode == "emptygen":
                return self._send(200, {"results": [{"text": ""}]})
            if mode == "genfail":
                return self._send(500, {"error": "boom"})
            return self._send(200, {"results": [{"text": " Blue. " + body["prompt"][:5]}]})
        self._send(404, {})


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''


@pytest.fixture
def chat_env(cf, tmp_path, monkeypatch):
    from localm.media.koboldcpp import _proc
    launcher = make_launcher(tmp_path / "fake", "kobold", FAKE_SERVER)
    model = tmp_path / "chat.gguf"
    model.write_bytes(b"GGUF")
    argv_file = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_ARGV_FILE", str(argv_file))
    ctx = cf.Ctx("v9.9", False, tmp_path, tmp_path / "cache", None, None)
    ctx.backend = "vulkan"
    rec = cf.PidRecorder()
    rec.install(_proc)
    yield SimpleNamespace(ctx=ctx, rt=SimpleNamespace(launcher=launcher), model=model,
                          argv_file=argv_file, rec=rec)
    rec.reap()
    rec.uninstall(_proc)


def test_the_chat_server_is_launched_answers_generates_and_is_stopped(cf, chat_env):
    record: dict = {}
    cf.run_chat_server(chat_env.ctx, chat_env.rt, chat_env.model, record)
    assert record["text"].startswith(" Blue.")
    assert "listener is ours: True" in record["server_api"]
    assert record["version_reply"]["version"] == "9.9"
    assert cf.vulkan_devices_from_log(record["log_lines"]) == ["Fake GPU 9000 | uma: 0"]
    argv = json.loads(chat_env.argv_file.read_text())
    assert argv[:2] == ["--model", str(chat_env.model)]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert "--skiplauncher" in argv and "--quiet" in argv and "--usevulkan" in argv
    assert "--musicdiffusion" not in argv
    assert len(chat_env.rec.entries) == 1
    pid = chat_env.rec.entries[0][0]
    assert _wait_dead(pid), "the server process outlived run_chat_server"


def test_the_launch_flags_are_the_server_wrappers_own_tail(cf):
    from localm.media.koboldcpp import server
    key = server.ServerKey("L", "vulkan", server.ModelSet("a", "b", "c", "d"))
    full = server.build_argv(key, 4321)
    tail = cf.chat_argv_tail(server, "L", "vulkan", 4321)
    assert full[-len(tail):] == tail
    assert tail[:6] == ["--host", "127.0.0.1", "--port", "4321", "--skiplauncher", "--quiet"]
    assert tail[-1] == "--usevulkan"
    for flag in ("--musicembeddings", "--musicdiffusion", "--musicvae", "--musicllm"):
        assert flag not in tail


def test_the_tail_derivation_fails_loudly_if_the_wrapper_drops_its_host_flag(
        cf, monkeypatch):
    from localm.media.koboldcpp import server
    monkeypatch.setattr(server, "build_argv", lambda key, port: ["L", "--port", str(port)])
    with pytest.raises(cf.Problem) as e:
        cf.chat_argv_tail(server, "L", "vulkan", 1)
    assert e.value.status == "SKIP" and "--host" in e.value.detail


def test_a_server_that_exits_while_loading_fails_with_its_output(cf, chat_env, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "exit")
    record: dict = {}
    with pytest.raises(cf.Problem) as e:
        cf.run_chat_server(chat_env.ctx, chat_env.rt, chat_env.model, record)
    assert e.value.status == "FAIL" and "exit code 3" in e.value.detail
    assert "could not load the model" in e.value.detail
    assert "server_api" not in record


@pytest.mark.parametrize("mode", ["badgen", "genfail"])
def test_a_generation_endpoint_that_misbehaves_fails_after_the_api_answered(
        cf, chat_env, monkeypatch, mode):
    monkeypatch.setenv("FAKE_MODE", mode)
    record: dict = {}
    with pytest.raises(cf.Problem) as e:
        cf.run_chat_server(chat_env.ctx, chat_env.rt, chat_env.model, record)
    assert e.value.status == "FAIL" and "server_api" in record
    assert _wait_dead(chat_env.rec.entries[0][0])


def test_empty_generated_text_is_returned_for_the_caller_to_judge(cf, chat_env, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "emptygen")
    record: dict = {}
    cf.run_chat_server(chat_env.ctx, chat_env.rt, chat_env.model, record)
    assert record["text"] == "" and not cf.text_is_usable(record["text"])


# --------------------------------------------------------------------------- #
#  main()                                                                      #
# --------------------------------------------------------------------------- #

@pytest.fixture
def env_guard(monkeypatch):
    for k in ("LOCALM_HOME", "TEMP", "TMP", "TMPDIR", "HF_HOME", "HF_HUB_DISABLE_PROGRESS_BARS"):
        monkeypatch.setenv(k, os.environ.get(k, "x"))


def test_a_bad_tag_writes_an_inconclusive_receipt(cf, tmp_path):
    rc = cf.main(["--tag", "latest", "--workdir", str(tmp_path / "w"),
                  "--receipt", str(tmp_path / "r.json")])
    data = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert rc == 2 and data["verdict"] == "INCONCLUSIVE" and data["current"] is False


def test_tag_and_current_are_mutually_exclusive_and_one_is_required(cf, tmp_path):
    base = ["--workdir", str(tmp_path), "--receipt", str(tmp_path / "r.json")]
    with pytest.raises(SystemExit):
        cf.main(["--tag", "v1.2", "--current", *base])
    with pytest.raises(SystemExit):
        cf.main(base)


def test_an_uncaught_exception_is_inconclusive_and_the_receipt_is_still_written(
        cf, tmp_path, env_guard, monkeypatch):
    def boom(ctx, receipt, recorder):
        receipt["hardware"]["seen"] = True
        raise RuntimeError("kaboom")
    monkeypatch.setattr(cf, "run_checks", boom)
    work = tmp_path / "w"
    rc = cf.main(["--tag", "v1.2", "--workdir", str(work), "--receipt", str(work / "r.json"),
                  "--cache-dir", str(tmp_path / "cache")])
    data = json.loads((work / "r.json").read_text(encoding="utf-8"))
    assert rc == 2 and data["verdict"] == "INCONCLUSIVE"
    assert "kaboom" in data["why"] and data["hardware"]["seen"] is True


def test_the_receipt_is_in_place_before_any_check_runs(cf, tmp_path, env_guard, monkeypatch):
    seen = {}

    def peek(ctx, receipt, recorder):
        seen["data"] = json.loads((tmp_path / "w" / "r.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(cf, "run_checks", peek)
    cf.main(["--tag", "v1.2", "--workdir", str(tmp_path / "w"),
             "--receipt", str(tmp_path / "w" / "r.json"), "--cache-dir", str(tmp_path / "c")])
    assert seen["data"]["verdict"] == "INCONCLUSIVE" and "in progress" in seen["data"]["why"]


def test_a_run_with_no_checks_recorded_is_inconclusive_not_PASS(
        cf, tmp_path, env_guard, monkeypatch):
    monkeypatch.setattr(cf, "run_checks", lambda ctx, receipt, recorder: None)
    rc = cf.main(["--tag", "v1.2", "--workdir", str(tmp_path / "w"),
                  "--receipt", str(tmp_path / "w" / "r.json"),
                  "--cache-dir", str(tmp_path / "c")])
    assert rc == 2


def test_cleanup_removes_only_the_throwaway_dirs(cf, tmp_path, env_guard, monkeypatch):
    monkeypatch.setattr(cf, "run_checks", lambda ctx, receipt, recorder: None)
    work, cache = tmp_path / "w", tmp_path / "cache"
    (cache).mkdir()
    (cache / "model.gguf").write_text("m")
    (work / "keep").mkdir(parents=True)
    cf.main(["--tag", "v1.2", "--workdir", str(work), "--receipt", str(work / "r.json"),
             "--cache-dir", str(cache)])
    assert not (work / "home").exists() and not (work / "tmp").exists()
    assert (work / "r.json").is_file() and (work / "keep").is_dir()
    assert (cache / "model.gguf").read_text() == "m"


def test_keep_leaves_the_throwaway_dirs(cf, tmp_path, env_guard, monkeypatch):
    monkeypatch.setattr(cf, "run_checks", lambda ctx, receipt, recorder: None)
    work = tmp_path / "w"
    cf.main(["--tag", "v1.2", "--workdir", str(work), "--receipt", str(work / "r.json"),
             "--cache-dir", str(tmp_path / "c"), "--keep"])
    assert (work / "home").is_dir() and (work / "tmp").is_dir()
