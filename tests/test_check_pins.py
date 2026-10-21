# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline tests for scripts/check_pins.py.

The only network function, ``_get_json``, is replaced by a router keyed on URL
substrings, so every gate runs its real parsing and comparison code. The pin
constants are read from the real shipped files (bound to the artefact), and the
failure paths point ``REPO`` at a temp tree.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_pins.py"
_spec = importlib.util.spec_from_file_location("check_pins", _PATH)
cp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cp
_spec.loader.exec_module(cp)

NOW = dt.datetime(2026, 10, 10, tzinfo=dt.UTC)


def _d(days_ago: int) -> str:
    return (NOW - dt.timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(tag, days_ago, *, prerelease=False, draft=False, assets=()):
    return {"tag_name": tag, "published_at": _d(days_ago), "prerelease": prerelease,
            "draft": draft, "assets": [{"name": a} for a in assets]}


def _router(monkeypatch, routes: dict):
    """Route _get_json by the first URL substring that matches; an Exception value
    is raised as FetchError; an unmatched URL fails the test (no silent network)."""
    seen = []

    def fake(url):
        seen.append(url)
        for needle, value in routes.items():
            if needle in url:
                if isinstance(value, cp.FetchError):
                    raise value
                if isinstance(value, Exception):
                    raise cp.FetchError(str(value))
                return value
        raise AssertionError(f"unrouted URL: {url}")
    monkeypatch.setattr(cp, "_get_json", fake)
    return seen


@pytest.fixture(autouse=True)
def _no_ambient_env(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)


# --------------------------------------------------------------------------- #
#  _assess_releases                                                           #
# --------------------------------------------------------------------------- #

def _assess(pinned, releases, tol=30, key=cp._semver_key):
    return cp._assess_releases("x", "g", "a", pinned, releases, key, tol, NOW)


def test_assess_current_when_nothing_is_newer():
    row = _assess("v1.2.0", [("v1.2.0", NOW - dt.timedelta(days=5))])
    assert row.status == cp.CURRENT and row.days == 0


def test_assess_age_is_when_the_oldest_newer_release_appeared_not_the_newest():
    rels = [("v1.3.0", NOW - dt.timedelta(days=40)), ("v1.4.0", NOW - dt.timedelta(days=2)),
            ("v1.2.0", NOW - dt.timedelta(days=90))]
    row = _assess("v1.2.0", rels)
    assert row.days == 40 and row.latest == "v1.4.0"


@pytest.mark.parametrize("age,status", [(30, cp.BEHIND), (31, cp.STALE), (0, cp.BEHIND)])
def test_assess_tolerance_boundary(age, status):
    row = _assess("v1.0.0", [("v1.1.0", NOW - dt.timedelta(days=age))], tol=30)
    assert row.status == status


def test_assess_version_ordering_is_numeric_not_lexical():
    row = _assess("v1.9.0", [("v1.10.0", NOW - dt.timedelta(days=1))])
    assert row.status == cp.BEHIND
    assert _assess("v1.10.0", [("v1.9.0", NOW)]).status == cp.CURRENT


def test_assess_unparseable_pin_and_empty_listing_are_unknown_never_current():
    assert _assess("not-a-version", [("v1.0.0", NOW)]).status == cp.UNKNOWN
    assert _assess("v1.0.0", []).status == cp.UNKNOWN
    assert _assess("v1.0.0", [("nightly", NOW)]).status == cp.UNKNOWN


def test_sdcpp_key_orders_by_build_number():
    assert cp._sdcpp_key("master-123-0a1b2c3") == (123,)
    assert cp._sdcpp_key("master-1000-abcdef0") > cp._sdcpp_key("master-123-0a1b2c3")
    assert cp._sdcpp_key("v1.0") is None


def test_github_releases_skips_drafts_prereleases_and_undated(monkeypatch):
    data = [_rel("v1", 1), _rel("v2", 1, prerelease=True), _rel("v3", 1, draft=True),
            {"tag_name": "v4", "published_at": None}, "junk"]
    _router(monkeypatch, {"api.github.com": data})
    assert [t for t, _ in cp._github_releases("r")] == ["v1"]


def test_github_releases_rejects_a_non_list_body(monkeypatch):
    _router(monkeypatch, {"api.github.com": {"message": "rate limited"}})
    with pytest.raises(cp.FetchError):
        cp._github_releases("owner/repo")


# --------------------------------------------------------------------------- #
#  The new gates, against the real shipped constants                          #
# --------------------------------------------------------------------------- #

def test_koboldcpp_current_and_stale_against_the_real_pin(monkeypatch):
    pinned = cp._read_const("localm/media/koboldcpp/pins.py", r'^TAG = "([^"]+)"')
    spec = next(s for s in cp.build_registry() if s.name == "koboldcpp")
    _router(monkeypatch, {"LostRuins/koboldcpp": [_rel(pinned, 50)]})
    assert spec.check(NOW).status == cp.CURRENT
    _router(monkeypatch, {"LostRuins/koboldcpp": [_rel(pinned, 50), _rel("v99.0.0", 31)]})
    row = spec.check(NOW)
    assert row.status == cp.STALE and row.pinned == pinned and row.latest == "v99.0.0"


def test_sdcpp_stale_when_a_newer_build_number_has_been_out_past_tolerance(monkeypatch):
    spec = next(s for s in cp.build_registry() if s.name == "stable-diffusion.cpp")
    _router(monkeypatch, {"leejet/stable-diffusion.cpp": [
        _rel("master-123-0a1b2c3", 60), _rel("master-2000-aaaaaaa", 45)]})
    assert spec.check(NOW).status == cp.STALE


def test_gate_is_unknown_never_current_when_the_api_fails(monkeypatch):
    _router(monkeypatch, {"api.github.com": OSError("rate limited")})
    spec = next(s for s in cp.build_registry() if s.name == "koboldcpp")
    row = spec.check(NOW)
    assert row.status == cp.UNKNOWN and "rate limited" in row.detail


def test_a_renamed_constant_is_inconsistent_not_unknown_and_not_current(monkeypatch, tmp_path):
    (tmp_path / "localm/media/koboldcpp").mkdir(parents=True)
    (tmp_path / "localm/media/koboldcpp/pins.py").write_text("TAGG = 'v1'\n", encoding="utf-8")
    monkeypatch.setattr(cp, "REPO", tmp_path)
    _router(monkeypatch, {"api.github.com": [_rel("v2.0.0", 1)]})
    spec = next(s for s in cp.build_registry() if s.name == "koboldcpp")
    row = spec.check(NOW)
    assert row.status == cp.INCONSISTENT and "cannot read the pin" in row.detail and "not found" in row.detail
    assert cp.exit_code([row]) == cp.EXIT_STALE


def test_gguf_node_current_when_head_is_the_pinned_commit(monkeypatch):
    pinned = cp._read_const("localm/media/managed_comfy_fresh.py",
                            r'name="ComfyUI-GGUF",\s*repo="[^"]+",\s*commit="([0-9a-f]{40})"')
    body = {"sha": pinned, "commit": {"committer": {"date": _d(100)}}}
    _router(monkeypatch, {"ComfyUI-GGUF/commits": body})
    spec = next(s for s in cp.build_registry() if s.name == "ComfyUI-GGUF node")
    assert spec.check(NOW).status == cp.CURRENT


def test_gguf_node_stale_when_main_moved_far_past_the_pin(monkeypatch):
    pinned = cp._read_const("localm/media/managed_comfy_fresh.py",
                            r'name="ComfyUI-GGUF",\s*repo="[^"]+",\s*commit="([0-9a-f]{40})"')
    head = {"sha": "b" * 40, "commit": {"committer": {"date": _d(1)}}}
    pin = {"sha": pinned, "commit": {"committer": {"date": _d(400)}}}

    def fake(url):
        return head if url.endswith("/commits/main") else pin
    monkeypatch.setattr(cp, "_get_json", fake)
    spec = next(s for s in cp.build_registry() if s.name == "ComfyUI-GGUF node")
    row = spec.check(NOW)
    assert row.status == cp.STALE and row.days == 399


def test_uv_flags_disagreeing_pins_as_inconsistent_even_when_all_are_current(monkeypatch, tmp_path):
    for rel, line in (("setup.sh", 'UV_INSTALLER_VERSION="0.13.0"\n'),
                      ("setup-gui.sh", 'UV_INSTALLER_VERSION="0.13.0"\n'),
                      ("setup.bat", 'set "UV_INSTALLER_VERSION=0.13.0"\n'),
                      ("setup-gui.bat", 'set "UV_INSTALLER_VERSION=0.13.0"\n'),
                      ("docker/Dockerfile", "ARG UV_VERSION=0.12.24\n")):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(line, encoding="utf-8")
    monkeypatch.setattr(cp, "REPO", tmp_path)
    _router(monkeypatch, {"astral-sh/uv": [_rel("0.13.0", 1)]})
    row = next(s for s in cp.build_registry() if s.name.startswith("uv")).check(NOW)
    assert row.status == cp.INCONSISTENT and "docker/Dockerfile=0.12.24" in row.detail


def test_real_uv_pins_agree_with_each_other():
    values = {rel: cp._read_const(rel, pat) for rel, pat in cp._UV_SITES}
    assert len(set(values.values())) == 1, values


def _docker_pinned_digest():
    return cp._read_const("docker/Dockerfile", r"^ARG UBUNTU_IMAGE=ubuntu:[\d.]+@(sha256:[0-9a-f]{64})$")


def _docker_spec():
    return next(s for s in cp.build_registry() if s.name == "Docker base image")


def test_docker_base_current_when_the_tag_still_points_at_the_pinned_digest(monkeypatch):
    monkeypatch.setattr(cp, "_registry_digest", lambda repo, tag: _docker_pinned_digest())
    row = _docker_spec().check(NOW)
    assert row.status == cp.CURRENT


@pytest.mark.parametrize("pushed_days_ago,status", [(30, cp.BEHIND), (31, cp.STALE)])
def test_docker_base_digest_mismatch_ages_from_the_last_republish(monkeypatch, pushed_days_ago, status):
    monkeypatch.setattr(cp, "_registry_digest", lambda repo, tag: "sha256:" + "0" * 64)
    _router(monkeypatch, {"official-images/commits": [
        {"commit": {"committer": {"date": _d(pushed_days_ago)}}}]})
    row = _docker_spec().check(NOW)
    assert row.status == status and row.days == pushed_days_ago


def test_docker_base_mismatch_without_a_republish_date_is_unknown_not_current(monkeypatch):
    monkeypatch.setattr(cp, "_registry_digest", lambda repo, tag: "sha256:" + "0" * 64)
    _router(monkeypatch, {"official-images/commits": []})
    assert _docker_spec().check(NOW).status == cp.UNKNOWN


def test_docker_base_registry_failure_is_unknown_never_current(monkeypatch):
    def boom(repo, tag):
        raise cp.FetchError("registry unreachable")
    monkeypatch.setattr(cp, "_registry_digest", boom)
    row = _docker_spec().check(NOW)
    assert row.status == cp.UNKNOWN and "registry unreachable" in row.detail


class _FakeResp:
    def __init__(self, headers, body=b''):
        self.headers = headers
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _no_sleep(monkeypatch):
    waits = []
    monkeypatch.setattr(cp.time, "sleep", lambda s: waits.append(s))
    return waits


def _http_error(code):
    return cp.urllib.error.HTTPError("https://x", code, "boom", {}, None)


def test_a_server_error_is_retried_and_then_succeeds(monkeypatch):
    waits = _no_sleep(monkeypatch)
    calls = []

    def flaky(req, timeout=None):
        calls.append(1)
        if len(calls) < 3:
            raise _http_error(500)
        return _FakeResp({}, b'{"ok": true}')
    monkeypatch.setattr(cp.urllib.request, "urlopen", flaky)
    assert cp._get_json("https://example.org/x") == {"ok": True}
    assert len(calls) == 3 and waits == list(cp._RETRY_DELAYS)


def test_a_server_error_that_never_clears_is_a_fetch_error_after_the_retries(monkeypatch):
    _no_sleep(monkeypatch)
    calls = []

    def down(req, timeout=None):
        calls.append(1)
        raise _http_error(503)
    monkeypatch.setattr(cp.urllib.request, "urlopen", down)
    with pytest.raises(cp.FetchError):
        cp._get_json("https://example.org/x")
    assert len(calls) == len(cp._RETRY_DELAYS) + 1


@pytest.mark.parametrize("code", [401, 403, 404])
def test_a_client_error_is_not_retried(monkeypatch, code):
    waits = _no_sleep(monkeypatch)
    calls = []

    def refuse(req, timeout=None):
        calls.append(1)
        raise _http_error(code)
    monkeypatch.setattr(cp.urllib.request, "urlopen", refuse)
    with pytest.raises(cp.FetchError):
        cp._get_json("https://example.org/x")
    assert len(calls) == 1 and waits == []


def test_registry_digest_retries_a_transient_registry_error(monkeypatch):
    _no_sleep(monkeypatch)
    _router(monkeypatch, {"auth.docker.io/token": {"token": "t"}})
    calls = []

    def flaky(req, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise _http_error(502)
        return _FakeResp({"Docker-Content-Digest": "sha256:" + "b" * 64})
    monkeypatch.setattr(cp.urllib.request, "urlopen", flaky)
    assert cp._registry_digest("ubuntu", "24.04") == "sha256:" + "b" * 64
    assert len(calls) == 2


def test_registry_digest_asks_the_registry_with_a_pull_token_and_reads_the_header(monkeypatch):
    _router(monkeypatch, {"auth.docker.io/token": {"token": "tok-1"}})
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(req)
        return _FakeResp({"Docker-Content-Digest": "sha256:" + "a" * 64})
    monkeypatch.setattr(cp.urllib.request, "urlopen", fake_urlopen)
    assert cp._registry_digest("ubuntu", "24.04") == "sha256:" + "a" * 64
    req = seen[0]
    assert req.full_url == "https://registry-1.docker.io/v2/library/ubuntu/manifests/24.04"
    assert req.get_method() == "HEAD" and req.get_header("Authorization") == "Bearer tok-1"
    assert "application/vnd.oci.image.index.v1+json" in req.get_header("Accept")


@pytest.mark.parametrize("headers", [{}, {"Docker-Content-Digest": "not-a-digest"}])
def test_registry_digest_without_a_valid_header_is_a_fetch_error(monkeypatch, headers):
    _router(monkeypatch, {"auth.docker.io/token": {"token": "t"}})
    monkeypatch.setattr(cp.urllib.request, "urlopen", lambda req, timeout=None: _FakeResp(headers))
    with pytest.raises(cp.FetchError):
        cp._registry_digest("ubuntu", "24.04")


def test_registry_digest_token_failure_is_a_fetch_error(monkeypatch):
    _router(monkeypatch, {"auth.docker.io/token": OSError("401")})
    with pytest.raises(cp.FetchError):
        cp._registry_digest("ubuntu", "24.04")


def test_parse_date_ignores_fractional_seconds():
    assert cp._parse_date("2026-10-04T01:04:16.566829Z") == dt.datetime(
        2026, 10, 4, 1, 4, 16, tzinfo=dt.UTC)
    assert cp._parse_date("garbage") is None and cp._parse_date(None) is None


def test_cuda_linux_source_stale_when_it_lacks_the_pinned_tag(monkeypatch):
    _router(monkeypatch, {"/releases/tags/": cp.FetchError("HTTP Error 404: Not Found", status=404)})
    row = next(s for s in cp.build_registry() if s.name == "Linux CUDA build source").check(NOW)
    assert row.status == cp.STALE and "fall back to vulkan" in row.detail


def _cuda_linux_table():
    text = cp._read_text("localm/setup_llama/pins.py")
    return dict(cp._ASSET_ENTRY_RE.findall(cp._CUDA_LINUX_TABLE_RE.search(text).group(1)))


def _cuda_source_row(monkeypatch, names, *, change=None):
    table = _cuda_linux_table()
    assets = [{"name": n, "digest": "sha256:" + ("0" * 64 if n == change else table.get(n, "1" * 64))}
              for n in names]
    _router(monkeypatch, {"llama-cpp-builder/releases/tags/": {"assets": assets}})
    return next(s for s in cp.build_registry() if s.name == "Linux CUDA build source").check(NOW)


def test_cuda_linux_source_current_when_every_pinned_asset_is_published_with_its_digest(monkeypatch):
    table = _cuda_linux_table()
    row = _cuda_source_row(monkeypatch, list(table))
    assert row.status == cp.CURRENT and f"{len(table)} pinned CUDA asset(s)" in row.detail


def test_cuda_linux_source_stale_when_only_an_unpinned_cuda_asset_is_published(monkeypatch):
    tag = cp._read_const("localm/setup_llama/pins.py", r'^_PINNED_TAG = "([^"]+)"')
    row = _cuda_source_row(monkeypatch, [f"llama-{tag}-bin-ubuntu-cuda-12-x64.tar.gz"])
    assert row.status == cp.STALE and "not in the release" in row.detail


def test_cuda_linux_source_stale_when_one_pinned_asset_was_removed(monkeypatch):
    table = _cuda_linux_table()
    victim = sorted(table)[0]
    row = _cuda_source_row(monkeypatch, [n for n in table if n != victim])
    assert row.status == cp.STALE and victim in row.detail


def test_cuda_linux_source_stale_when_a_pinned_asset_was_re_uploaded(monkeypatch):
    table = _cuda_linux_table()
    victim = sorted(table)[-1]
    row = _cuda_source_row(monkeypatch, list(table), change=victim)
    assert row.status == cp.STALE and "digest differs" in row.detail and victim in row.detail


def test_cuda_linux_source_names_an_asset_published_without_a_digest(monkeypatch):
    table = _cuda_linux_table()
    victim = sorted(table)[0]
    _router(monkeypatch, {"llama-cpp-builder/releases/tags/": {"assets": [
        {"name": n, "digest": "" if n == victim else "sha256:" + table[n]} for n in table]}})
    row = next(s for s in cp.build_registry() if s.name == "Linux CUDA build source").check(NOW)
    assert row.status == cp.STALE and f"no digest published for: {victim}" in row.detail
    assert "differs" not in row.detail


def test_cuda_linux_source_inconsistent_when_the_table_does_not_name_the_pinned_tag(monkeypatch, tmp_path):
    (tmp_path / "localm" / "setup_llama").mkdir(parents=True)
    (tmp_path / "localm" / "setup_llama" / "pins.py").write_text(
        '_PINNED_TAG = "b2"\n_CUDA_LINUX_REPO = "o/llama-cpp-builder"\n'
        '_CUDA_LINUX_PINNED_SHA256 = {\n'
        f'    "llama-b1-bin-ubuntu-cuda-x64.tar.gz": "{"a" * 64}",\n}}\n', encoding="utf-8")
    monkeypatch.setattr(cp, "REPO", tmp_path)
    _router(monkeypatch, {"llama-cpp-builder/releases/tags/b2": {"assets": [
        {"name": "llama-b1-bin-ubuntu-cuda-x64.tar.gz", "digest": "sha256:" + "a" * 64}]}})
    row = next(s for s in cp.build_registry() if s.name == "Linux CUDA build source").check(NOW)
    assert row.status == cp.INCONSISTENT
    assert "no pinned checksum for llama-b2-bin-ubuntu-cuda-13-x64.tar.gz" in row.detail
    assert "llama-b1-bin-ubuntu-cuda-x64.tar.gz" in row.detail


def test_cuda_linux_suffixes_are_the_names_the_resolver_installs(monkeypatch):
    from localm.setup_llama import assets
    tag = cp._read_const("localm/setup_llama/pins.py", r'^_PINNED_TAG = "([^"]+)"')
    monkeypatch.setattr(assets._sl, "_release_assets", lambda tag, repo=None: [])
    resolved = {assets._resolve_linux_cuda_asset("cuda", line, tag)[0].rsplit("/", 1)[-1]
                for line in ("cuda-12", "cuda-13")}
    assert resolved == {f"llama-{tag}-bin-ubuntu{s}" for s in cp._CUDA_LINUX_SUFFIXES}
    assert resolved == set(_cuda_linux_table())


def test_rocm_cpu_archive_needs_the_exact_asset(monkeypatch):
    tag = cp._read_const("localm/setup_llama/pins.py", r'^_ROCM_CPU_TAG = "([^"]+)"')
    spec = next(s for s in cp.build_registry() if s.name == "ROCm CPU archive")
    _router(monkeypatch, {"/releases/tags/": _rel(tag, 1, assets=[f"llama-{tag}-bin-win-cpu-x64.zip"])})
    assert spec.check(NOW).status == cp.CURRENT
    _router(monkeypatch, {"/releases/tags/": _rel(tag, 1, assets=["something-else.zip"])})
    assert spec.check(NOW).status == cp.STALE


def test_vendored_versions_are_all_readable_from_the_real_files():
    for name, rel, pattern, _src, _tol in cp._VENDORED:
        version = cp._read_const(rel, pattern)
        assert cp._semver_key(version) is not None, (name, version)


def test_vendored_marked_stale_against_a_much_newer_release(monkeypatch):
    spec = next(s for s in cp.build_registry() if s.name == "vendored marked")
    pinned = cp._read_const(*[(v[1], v[2]) for v in cp._VENDORED if v[0] == "marked"][0])
    _router(monkeypatch, {"markedjs/marked": [_rel(f"v{pinned}", 900), _rel("v99.0.0", 200)]})
    assert spec.check(NOW).status == cp.STALE


def test_npm_releases_drop_prereleases_and_bookkeeping_keys(monkeypatch):
    _router(monkeypatch, {"registry.npmjs.org": {"time": {
        "created": "2020-01-01T00:00:00.000Z", "modified": "2026-01-01T00:00:00.000Z",
        "1.0.0": "2021-01-01T00:00:00.000Z", "2.0.0-beta.1": "2022-01-01T00:00:00.000Z"}}})
    assert [v for v, _ in cp._npm_releases("p")] == ["1.0.0"]


def test_unversioned_vendored_files_are_reported_not_hidden():
    rows = cp.run_checks([s for s in cp.build_registry() if s.name in
                          ("vendored jsQR", "vendored Inter font")], NOW)
    assert [r.status for r in rows] == [cp.UNVERSIONED, cp.UNVERSIONED]


# --------------------------------------------------------------------------- #
#  Existing gates as subprocesses                                             #
# --------------------------------------------------------------------------- #

def _fake_proc(rc, out=""):
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=out, stderr="")


@pytest.mark.parametrize("rc,out,status", [
    (0, "OK: current", cp.CURRENT), (0, "BEHIND by 3\nwithin tolerance", cp.BEHIND),
    (1, "STALE", cp.STALE), (2, "COULD NOT CHECK", cp.UNKNOWN), (7, "traceback", cp.UNKNOWN)])
def test_legacy_gate_exit_codes_map_to_statuses(monkeypatch, rc, out, status):
    monkeypatch.setattr(cp.subprocess, "run", lambda *a, **k: _fake_proc(rc, out))
    check = cp._legacy_check("n", "g", "scripts/check_llama_pin.py", "adv")
    assert check(NOW).status == status


def test_a_legacy_gate_that_dies_with_only_stderr_still_explains_itself(monkeypatch):
    monkeypatch.setattr(cp.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="could not read pins.py: boom"))
    row = cp._legacy_check("n", "g", "scripts/check_llama_pin.py", "a")(NOW)
    assert row.status == cp.STALE and "exit code 1" in row.detail and "could not read pins.py: boom" in row.detail


def test_legacy_gate_that_cannot_launch_is_unknown(monkeypatch):
    def boom(*a, **k):
        raise OSError("no python")
    monkeypatch.setattr(cp.subprocess, "run", boom)
    assert cp._legacy_check("n", "g", "scripts/check_llama_pin.py", "a")(NOW).status == cp.UNKNOWN


def test_legacy_rows_extract_pinned_and_newest_and_a_readable_detail(monkeypatch):
    out = ("localm pins llama.cpp b11118\n  pinned release date: 2026-09-23\n"
           "upstream newest with assets: b11541 (released 2026-10-10)\n"
           "BEHIND: 423 builds\nwithin the 21-day tolerance.\n")
    monkeypatch.setattr(cp.subprocess, "run", lambda *a, **k: _fake_proc(0, out))
    spec = next(s for s in cp.build_registry() if s.name == "llama.cpp")
    row = spec.check(NOW)
    assert (row.pinned, row.latest, row.status) == ("b11118", "b11541", cp.BEHIND)
    assert row.detail.startswith("BEHIND: 423 builds") and "pinned release date" not in row.detail


def _declared_options(path):
    """Every option string the script passes to an add_argument call."""
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {a.value for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"
            for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str)}


def test_every_legacy_script_exists_and_accepts_gate():
    for _name, _group, script, _adv, _p, _l in cp._LEGACY:
        assert "--gate" in _declared_options(cp.REPO / script), script


def test_the_option_scanner_ignores_a_mention_that_is_not_a_declared_option(tmp_path):
    f = tmp_path / "s.py"
    f.write_text('"""takes --gate"""\n# "--gate"\nx = "--gate"\n'
                 'import argparse\np = argparse.ArgumentParser()\np.add_argument("--other")\n',
                 encoding="utf-8")
    assert _declared_options(f) == {"--other"}


# --------------------------------------------------------------------------- #
#  Aggregation, exit codes, output                                            #
# --------------------------------------------------------------------------- #

def _row(status):
    return cp.Row(name="n", status=status)


@pytest.mark.parametrize("statuses,code", [
    ([cp.CURRENT, cp.BEHIND, cp.UNVERSIONED], 0),
    ([cp.CURRENT, cp.UNKNOWN], 2),
    ([cp.UNKNOWN, cp.STALE], 1),
    ([cp.INCONSISTENT], 1),
    ([], 0)])
def test_exit_code_contract(statuses, code):
    assert cp.exit_code([_row(s) for s in statuses]) == code


def test_a_crashing_check_becomes_unknown_and_does_not_hide_the_others():
    def boom(now):
        raise RuntimeError("kaboom")
    specs = [cp.PinSpec("a", "g", "adv", boom),
             cp.PinSpec("b", "g", "adv", lambda now: cp.Row(name="b", status=cp.CURRENT))]
    rows = cp.run_checks(specs, NOW)
    assert [r.status for r in rows] == [cp.UNKNOWN, cp.CURRENT]
    assert "kaboom" in rows[0].detail


def _stub_registry(monkeypatch, statuses):
    def checker(status):
        return lambda now: cp.Row(name="x", status=status)
    specs = [cp.PinSpec(f"p{i}", "g", "adv", checker(s), legacy=(i == 0))
             for i, s in enumerate(statuses)]
    monkeypatch.setattr(cp, "build_registry", lambda: specs)


def test_main_gate_exit_codes(monkeypatch, capsys):
    _stub_registry(monkeypatch, [cp.CURRENT, cp.STALE])
    assert cp.main(["--gate"]) == 1
    _stub_registry(monkeypatch, [cp.CURRENT, cp.UNKNOWN])
    assert cp.main(["--gate"]) == 2
    _stub_registry(monkeypatch, [cp.CURRENT, cp.BEHIND])
    assert cp.main(["--gate"]) == 0
    _stub_registry(monkeypatch, [cp.STALE])
    assert cp.main([]) == 0


def test_main_new_only_skips_legacy_and_writes_json(monkeypatch, tmp_path, capsys):
    _stub_registry(monkeypatch, [cp.STALE, cp.CURRENT])
    out = tmp_path / "rows.json"
    assert cp.main(["--gate", "--new-only", "--json", str(out)]) == 0
    assert [r["status"] for r in json.loads(out.read_text(encoding="utf-8"))] == [cp.CURRENT]


def test_main_exclude_drops_named_pins_and_rejects_unknown_names(monkeypatch, capsys):
    _stub_registry(monkeypatch, [cp.CURRENT, cp.STALE])
    assert cp.main(["--gate", "--exclude", "p1"]) == 0
    assert cp.main(["--gate", "--exclude", "not-a-pin"]) == cp.EXIT_USAGE == 3
    assert cp.main(["--exclude", "not-a-pin"]) == 3
    assert "no known pin" in capsys.readouterr().err


def test_main_only_with_no_match_is_not_a_pass(monkeypatch, capsys):
    _stub_registry(monkeypatch, [cp.CURRENT])
    assert cp.main(["--gate", "--only", "nope"]) == 3
    assert cp.main(["--only", "nope"]) == 3


def test_annotations_and_summary_only_under_github_actions(monkeypatch, tmp_path, capsys):
    _stub_registry(monkeypatch, [cp.CURRENT, cp.STALE])
    cp.main(["--gate"])
    assert "::error::" not in capsys.readouterr().out
    summary = tmp_path / "s.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    cp.main(["--gate"])
    assert "::error::" in capsys.readouterr().out
    assert "| pin | status |" in summary.read_text(encoding="utf-8")


@pytest.mark.parametrize("url,sent", [
    ("https://api.github.com/repos/o/r/releases", True),
    ("https://auth.docker.io/token?service=registry.docker.io", False),
    ("https://registry-1.docker.io/v2/library/ubuntu/manifests/24.04", False),
    ("https://pypi.org/pypi/nvidia-cublas/json", False),
    ("https://registry.npmjs.org/marked", False),
    ("https://api.github.com.evil.example/x", False),
    ("", False)])
def test_the_github_token_is_sent_to_api_github_com_only(monkeypatch, url, sent):
    monkeypatch.setenv("GITHUB_TOKEN", "tok-1")
    headers = cp._headers(url)
    assert ("Authorization" in headers) is sent
    if sent:
        assert headers["Authorization"] == "Bearer tok-1"


def test_get_json_does_not_leak_the_github_token_to_other_hosts(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok-1")
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(req.get_header("Authorization"))
        return _FakeResp({}, b"{}")
    monkeypatch.setattr(cp.urllib.request, "urlopen", fake_urlopen)
    cp._get_json("https://auth.docker.io/token?x=1")
    cp._get_json("https://api.github.com/repos/o/r")
    assert seen == [None, "Bearer tok-1"]


def test_a_404_in_the_tag_name_is_not_a_missing_release(monkeypatch):
    """The pinned tag may contain "404"; only the HTTP status decides."""
    monkeypatch.setattr(cp, "_read_const", lambda rel, pattern: "b11404")
    _router(monkeypatch, {"/releases/tags/": cp.FetchError("https://x/b11404: timed out", status=None)})
    row = next(s for s in cp.build_registry() if s.name == "Linux CUDA build source").check(NOW)
    assert row.status == cp.UNKNOWN


def test_a_semver_key_keeps_a_fourth_component():
    assert cp._semver_key("12.9.2.10") < cp._semver_key("12.9.2.11")
    assert cp._semver_key("v1.122.1") == (1, 122, 1, 0)
    assert cp._semver_key("1.2") == (1, 2, 0, 0)


def test_the_cuda_wheel_age_comes_from_the_packages_that_are_behind_only(monkeypatch):
    pins_text = cp._read_text("localm/setup_llama/cuda.py")
    m = cp._CUDA_RUNTIME_RE.search(pins_text)
    pinned = dict(cp.re.findall(r'"([a-z0-9._-]+)":\s*\(\s*"([^"]+)"', m.group(1)))
    packages = list(pinned)
    behind = packages[0]
    old, fresh = NOW - dt.timedelta(days=900), NOW - dt.timedelta(days=2)
    monkeypatch.setattr(cp, "_pypi_latest", lambda package: (
        ("999.0.0", fresh) if package == behind else (pinned[package], old)))
    row = next(s for s in cp.build_registry() if s.name == "Linux CUDA runtime wheels").check(NOW)
    assert row.days == 2 and row.status == cp.BEHIND


def test_a_fourth_component_bump_of_a_cuda_wheel_is_seen(monkeypatch):
    pins_text = cp._read_text("localm/setup_llama/cuda.py")
    m = cp._CUDA_RUNTIME_RE.search(pins_text)
    pinned = dict(cp.re.findall(r'"([a-z0-9._-]+)":\s*\(\s*"([^"]+)"', m.group(1)))
    package = next(p for p, v in pinned.items() if v.count(".") == 3)
    major, minor, patch, build = pinned[package].split(".")
    newer = f"{major}.{minor}.{patch}.{int(build) + 1}"
    monkeypatch.setattr(cp, "_pypi_latest", lambda p: (newer if p == package else pinned[p], NOW))
    row = next(s for s in cp.build_registry() if s.name == "Linux CUDA runtime wheels").check(NOW)
    assert row.status == cp.BEHIND and newer in row.detail


def test_an_accepted_blocker_turns_a_stale_row_into_blocked_until_it_expires():
    rows = [cp.Row(name="vendored marked", status=cp.STALE, detail="56 newer release(s)")]
    cp.apply_blockers(rows, NOW)
    assert rows[0].status == cp.BLOCKED and "accepted until" in rows[0].detail and "56 newer" in rows[0].detail
    assert cp.exit_code(rows) == cp.EXIT_OK


def test_an_expired_blocker_is_stale_again():
    rows = [cp.Row(name="vendored marked", status=cp.STALE)]
    cp.apply_blockers(rows, dt.datetime(2027, 6, 1, tzinfo=dt.UTC))
    assert rows[0].status == cp.STALE and cp.exit_code(rows) == cp.EXIT_STALE


def test_a_blocker_never_hides_a_row_that_is_not_stale_or_behind():
    rows = [cp.Row(name="vendored marked", status=cp.UNKNOWN), cp.Row(name="vendored marked", status=cp.INCONSISTENT)]
    cp.apply_blockers(rows, NOW)
    assert [r.status for r in rows] == [cp.UNKNOWN, cp.INCONSISTENT]


def test_only_pins_with_a_written_reason_are_blocked():
    for name, (expiry, reason) in cp.ACCEPTED_BLOCKERS.items():
        assert name in {s.name for s in cp.build_registry()}
        assert len(reason) > 20 and dt.datetime.strptime(expiry, "%Y-%m-%d")


def test_unchecked_sources_are_listed_every_run_with_their_reason():
    rows = cp.run_checks([s for s in cp.build_registry() if s.group == "unchecked"], NOW)
    assert len(rows) == len(cp._NOT_CHECKABLE) >= 3
    assert all(r.status == cp.UNVERSIONED and len(r.detail) > 10 for r in rows)


def test_an_unchecked_source_whose_file_moved_is_inconsistent_not_silently_dropped(monkeypatch, tmp_path):
    monkeypatch.setattr(cp, "REPO", tmp_path)
    rows = cp.run_checks([s for s in cp.build_registry() if s.group == "unchecked"], NOW)
    assert all(r.status == cp.INCONSISTENT and "is missing" in r.detail for r in rows)


def _real_table():
    text = cp._read_text("localm/setup_llama/pins.py")
    block = cp._ASSET_TABLE_RE.search(text).group(1)
    return dict(cp._ASSET_ENTRY_RE.findall(block))


def _asset_router(monkeypatch, *, change=None, drop=None):
    """A GitHub that publishes exactly the digests the repo pins (optionally one changed
    or one dropped)."""
    table = _real_table()
    tag = cp._read_const("localm/setup_llama/pins.py", r'^_PINNED_TAG = "([^"]+)"')
    rocm = cp._read_const("localm/setup_llama/pins.py", r'^_ROCM_TAG = "([^"]+)"')
    cpu = cp._read_const("localm/setup_llama/pins.py", r'^_ROCM_CPU_TAG = "([^"]+)"')

    cuda_table = _cuda_linux_table()
    digests = {**table, **cuda_table}

    def published(names):
        return {"assets": [{"name": n, "digest": "sha256:" + ("0" * 64 if n == change else digests[n])}
                           for n in names if n != drop]}
    rocm_names = [n for n in table if n.startswith(f"llama-{rocm}-")]
    cpu_names = [n for n in table if n.startswith(f"llama-{cpu}-")]
    upstream_names = [n for n in table if n not in rocm_names and n not in cpu_names]
    routes = {
        f"ggml-org/llama.cpp/releases/tags/{cpu}": published(cpu_names),
        f"llamacpp-rocm/releases/tags/{rocm}": published(rocm_names),
        f"ggml-org/llama.cpp/releases/tags/{tag}": published(upstream_names),
        f"llama-cpp-builder/releases/tags/{tag}": published(list(cuda_table)),
    }
    _router(monkeypatch, routes)
    return table, tag


def _checksum_row():
    return next(s for s in cp.build_registry() if s.name == "llama.cpp asset checksums").check(NOW)


def test_llama_asset_checksums_match_what_github_publishes(monkeypatch):
    table, _ = _asset_router(monkeypatch)
    row = _checksum_row()
    assert row.status == cp.CURRENT
    assert f"{len(table) + len(_cuda_linux_table())} pinned checksum(s)" in row.detail


def test_a_changed_published_digest_is_stale(monkeypatch):
    table, tag = _asset_router(monkeypatch, change=sorted(n for n in _real_table() if "-bin-win-cpu-x64" in n)[0])
    row = _checksum_row()
    assert row.status == cp.STALE and "differ from the published digest" in row.detail


def test_a_pinned_asset_missing_from_its_release_is_inconsistent(monkeypatch):
    victim = next(n for n in _real_table() if n.endswith("-bin-win-vulkan-x64.zip"))
    _asset_router(monkeypatch, drop=victim)
    row = _checksum_row()
    assert row.status == cp.INCONSISTENT and victim in row.detail


@pytest.mark.parametrize("victim", ["cudart-llama-bin-win-cuda-12.4-x64.zip",
                                          "cudart-llama-bin-win-cuda-13.4-x64.zip"])
def test_a_tagless_cudart_asset_missing_from_the_release_is_inconsistent(monkeypatch, victim):
    assert victim in _real_table()
    _asset_router(monkeypatch, drop=victim)
    row = _checksum_row()
    assert row.status == cp.INCONSISTENT and victim in row.detail


def test_a_re_uploaded_linux_cuda_asset_is_stale_in_the_checksum_row(monkeypatch):
    victim = sorted(_cuda_linux_table())[0]
    _asset_router(monkeypatch, change=victim)
    row = _checksum_row()
    assert row.status == cp.STALE and victim in row.detail


def test_a_linux_cuda_asset_missing_from_its_release_is_inconsistent_in_the_checksum_row(monkeypatch):
    victim = sorted(_cuda_linux_table())[-1]
    _asset_router(monkeypatch, drop=victim)
    row = _checksum_row()
    assert row.status == cp.INCONSISTENT and victim in row.detail


def test_an_unreadable_release_listing_is_unknown_for_the_checksums(monkeypatch):
    _router(monkeypatch, {"releases/tags/": cp.FetchError("rate limited", status=403)})
    assert _checksum_row().status == cp.UNKNOWN


def test_an_empty_or_missing_checksum_table_is_inconsistent_not_current(monkeypatch, tmp_path):
    (tmp_path / "localm" / "setup_llama").mkdir(parents=True)
    head = "\n".join(['_PINNED_TAG = "b1"', '_ROCM_TAG = "b2"', '_ROCM_CPU_TAG = "b3"',
                      '_UPSTREAM_REPO = "o/r"', ''])
    monkeypatch.setattr(cp, "REPO", tmp_path)
    for body in ("_PINNED_FALLBACK_SHA256 = {\n}\n", "# no table here\n"):
        (tmp_path / "localm" / "setup_llama" / "pins.py").write_text(head + body, encoding="utf-8")
        row = _checksum_row()
        assert row.status == cp.INCONSISTENT and "table" in row.detail


def test_the_real_checksum_table_is_parseable_and_every_entry_is_64_hex():
    table = _real_table()
    assert len(table) > 40 and all(len(v) == 64 for v in table.values())


def _pin_constant_files():
    """Every tracked source file that assigns a pin-like constant."""
    import subprocess
    out = subprocess.run(["git", "ls-files", "localm/*.py"], cwd=cp.REPO, capture_output=True, text=True).stdout
    pattern = cp.re.compile(
        r"^_?[A-Z0-9_]*(?:PINNED|_PIN|PIN_|_SHA256|_COMMIT)[A-Z0-9_]*\s*(?::[^=]*)?="
        r"|^_?[A-Z0-9_]*_TAG\s*(?::[^=]*)?=\s*[\"']|^(?:TAG|COMMIT)\s*=\s*\"", cp.re.MULTILINE)
    return {f for f in out.split() if pattern.search((cp.REPO / f).read_text(encoding="utf-8", errors="replace"))}


# Files that assign something pin-shaped that is not a version to keep current, each with the reason.
_NOT_A_VERSION_PIN = {
    "localm/config.py": "a marker name and a cache-pin tuple for GPU caches, not an upstream version",
    "localm/inference/backends/llamacpp/_api.py": "the llama.cpp tag the MTP allowlist was derived from; the "
                                                  "llama bump rewrites it with the pin and a test requires "
                                                  "it to equal the pinned tag",
    "localm/inference/residency.py": "a lock for pinned (resident) models",
    "localm/netpolicy.py": "a limit on pinned network addresses",
    "localm/media/koboldcpp/music.py": "a duration constant",
    "localm/inference/tool_calling.py": "tag delimiters of a chat format",
    "localm/plugins/builtin/voice/audio_model.py": "a tag of a chat format",
    "localm/updater.py": "the built-in update signing key set",
    "localm/plugins/coder/agent/context.py": "a cap on the changed files a compaction block lists",
    "localm/media/native_workflows.py": "a pattern that checks a sha256 digest's shape",
}


def _files_the_registry_reads(monkeypatch):
    read = set()
    real = cp._read_const

    def spy(rel, pattern):
        read.add(rel)
        return real(rel, pattern)
    monkeypatch.setattr(cp, "_read_const", spy)
    real_text = cp._read_text

    def spy_text(rel):
        read.add(rel)
        return real_text(rel)
    monkeypatch.setattr(cp, "_read_text", spy_text)
    monkeypatch.setattr(cp, "_get_json", lambda url: (_ for _ in ()).throw(cp.FetchError("offline")))
    monkeypatch.setattr(cp, "_pypi_latest", lambda p: (_ for _ in ()).throw(cp.FetchError("offline")))
    monkeypatch.setattr(cp, "_registry_digest", lambda r, t: (_ for _ in ()).throw(cp.FetchError("offline")))
    monkeypatch.setattr(cp, "_hf_linked_sha256", lambda *a, **k: (_ for _ in ()).throw(cp.FetchError("offline")))
    monkeypatch.setattr(cp.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 2, "", ""))
    cp.run_checks([s for s in cp.build_registry() if not s.legacy], NOW)
    return read


def test_every_file_that_pins_a_version_is_read_by_the_registry_or_explained(monkeypatch):
    pinned_files = _pin_constant_files()
    assert "localm/setup_llama/pins.py" in pinned_files and "localm/media/sdcpp/pins.py" in pinned_files
    read = _files_the_registry_reads(monkeypatch)
    reported = {rel for _name, rel, _why in cp._NOT_CHECKABLE}
    unaccounted = sorted(f for f in pinned_files
                         if f not in read and f not in reported and f not in _NOT_A_VERSION_PIN)
    assert unaccounted == [], f"pins nothing watches: {unaccounted}"


def test_the_not_a_version_list_has_no_stale_entries():
    pinned_files = _pin_constant_files()
    assert sorted(f for f in _NOT_A_VERSION_PIN if f not in pinned_files) == []


def test_registry_covers_every_inventory_group():
    names = {s.name for s in cp.build_registry()}
    for required in ("llama.cpp", "ROCm llama (lemonade)", "ComfyUI", "AMD ROCm wheels", "koboldcpp",
                     "stable-diffusion.cpp", "ComfyUI-GGUF node", "uv (installers + Docker)",
                     "Docker base image", "Linux CUDA build source", "ROCm CPU archive", "Linux CUDA runtime wheels",
                     "vendored marked", "vendored DOMPurify", "vendored highlight.js",
                     "vendored KaTeX", "vendored kokoro-js", "vendored transformers.js",
                     "vendored onnxruntime-web", "vendored jsQR", "vendored Inter font"):
        assert required in names, required


# ---- pinned Hugging Face file digests ---------------------------------------


def _hf_row(monkeypatch, served):
    """Run the digest row with a hub that answers *served* (a callable
    ``(repo, path, revision) -> sha256 | None`` that may raise FetchError)."""
    monkeypatch.setattr(cp, "_hf_linked_sha256", served)
    spec = next(s for s in cp.build_registry() if s.name == cp._HF_PINS_NAME)
    return cp.run_checks([spec], NOW)[0]


def _hub_serving_the_pins():
    pins = {(repo, path): sha for _label, repo, path, _rev, sha in cp.hf_file_pins()}
    return lambda repo, path, revision="main": pins[(repo, path)]


def test_the_pinned_hf_files_cover_music_comfyui_and_starter_models_and_every_digest_is_64_hex():
    pins = cp.hf_file_pins()
    labels = [p[0] for p in pins]
    assert sum(label.startswith("music ") for label in labels) == 4
    assert sum(label.startswith("ComfyUI ") for label in labels) == 8
    assert sum(label.startswith("starter ") for label in labels) == 11
    assert sum(label.startswith("whisper ") for label in labels) == 4
    from localm import voice
    assert {p[3] for p in pins if p[0].startswith("whisper ")} == {
        commit for commit, _ in voice._STT_PINS.values()}
    assert all(cp._HEX64_RE.match(p[4]) for p in pins)
    assert len({(p[1], p[2]) for p in pins}) == len(pins)


def test_a_hub_serving_the_pinned_digests_reads_current(monkeypatch):
    row = _hf_row(monkeypatch, _hub_serving_the_pins())
    assert row.status == cp.CURRENT and f"{len(cp.hf_file_pins())} pinned digest(s)" in row.detail


def test_a_reuploaded_file_reads_stale_and_names_it(monkeypatch):
    serving = _hub_serving_the_pins()
    victim = next(p for p in cp.hf_file_pins() if p[0].startswith("ComfyUI clip_l"))

    def served(repo, path, revision="main"):
        return "0" * 64 if (repo, path) == (victim[1], victim[2]) else serving(repo, path, revision)

    row = _hf_row(monkeypatch, served)
    assert row.status == cp.STALE and "ComfyUI clip_l.safetensors" in row.detail


@pytest.mark.parametrize("status", [401, 403, 404])
def test_a_pinned_source_the_hub_refuses_reads_inconsistent(monkeypatch, status):
    serving = _hub_serving_the_pins()
    victim = next(p for p in cp.hf_file_pins() if p[0].startswith("ComfyUI ae"))

    def served(repo, path, revision="main"):
        if (repo, path) == (victim[1], victim[2]):
            raise cp.FetchError("refused", status=status)
        return serving(repo, path, revision)

    row = _hf_row(monkeypatch, served)
    assert row.status == cp.INCONSISTENT and f"HTTP {status}" in row.detail


def test_a_hub_that_cannot_be_reached_reads_unknown_never_current(monkeypatch):
    def served(repo, path, revision="main"):
        raise cp.FetchError("timed out")

    assert _hf_row(monkeypatch, served).status == cp.UNKNOWN


def test_a_pinned_file_the_hub_does_not_store_as_lfs_reads_stale(monkeypatch):
    assert _hf_row(monkeypatch, lambda repo, path, revision="main": None).status == cp.STALE


def test_a_renamed_pin_table_is_inconsistent_not_current(monkeypatch, tmp_path):
    serving = _hub_serving_the_pins()
    monkeypatch.setattr(cp, "REPO", tmp_path)
    row = _hf_row(monkeypatch, serving)
    assert row.status == cp.INCONSISTENT


def test_the_unredirected_response_is_what_carries_the_digest(monkeypatch):
    import email.message
    import urllib.error
    sha = "ab" * 32
    headers = email.message.Message()
    headers["X-Linked-ETag"] = f'"{sha}"'
    seen = []

    class Opener:
        def open(self, req, timeout=None):
            seen.append((req.get_method(), req.full_url))
            raise urllib.error.HTTPError(req.full_url, 302, "Found", headers, None)

    monkeypatch.setattr(cp.urllib.request, "build_opener", lambda *handlers: Opener())
    assert cp._hf_linked_sha256("o/r", "dir/a b.gguf", "abc") == sha
    assert seen == [("HEAD", "https://huggingface.co/o/r/resolve/abc/dir/a%20b.gguf")]
    headers.replace_header("X-Linked-ETag", '"not-a-sha"')
    assert cp._hf_linked_sha256("o/r", "f", "main") is None


def test_a_missing_file_raises_with_the_status(monkeypatch):
    import email.message
    import urllib.error

    class Opener:
        def open(self, req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", email.message.Message(), None)

    monkeypatch.setattr(cp.urllib.request, "build_opener", lambda *handlers: Opener())
    with pytest.raises(cp.FetchError) as caught:
        cp._hf_linked_sha256("o/r", "f")
    assert caught.value.status == 404


def _opener_failing(statuses, retry_after=None, ok_headers=None):
    import email.message
    import urllib.error
    calls = []

    class Opener:
        def open(self, req, timeout=None):
            calls.append(req.full_url)
            n = len(calls) - 1
            if n < len(statuses):
                headers = email.message.Message()
                if retry_after is not None:
                    headers["Retry-After"] = retry_after
                raise urllib.error.HTTPError(req.full_url, statuses[n], "x", headers, None)

            class Resp:
                headers = ok_headers or {"X-Linked-ETag": '"' + "cd" * 32 + '"'}

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False
            return Resp()
    return Opener(), calls


def test_a_rate_limited_hub_is_retried_after_the_wait_it_asks_for(monkeypatch):
    opener, calls = _opener_failing([429, 429], retry_after="7")
    waits = []
    monkeypatch.setattr(cp.urllib.request, "build_opener", lambda *handlers: opener)
    monkeypatch.setattr(cp.time, "sleep", waits.append)
    assert cp._hf_linked_sha256("o/r", "f") == "cd" * 32
    assert waits == [7.0, 7.0] and len(calls) == 3


def test_a_huge_retry_after_is_capped_and_a_missing_one_uses_the_default_waits(monkeypatch):
    opener, _ = _opener_failing([429], retry_after="9999")
    waits = []
    monkeypatch.setattr(cp.urllib.request, "build_opener", lambda *handlers: opener)
    monkeypatch.setattr(cp.time, "sleep", waits.append)
    cp._hf_linked_sha256("o/r", "f")
    assert waits == [cp._HF_RETRY_AFTER_CAP]
    opener, _ = _opener_failing([503, 503, 503])
    waits.clear()
    monkeypatch.setattr(cp.urllib.request, "build_opener", lambda *handlers: opener)
    cp._hf_linked_sha256("o/r", "f")
    assert waits == list(cp._HF_RETRY_DELAYS)


def test_a_hub_that_keeps_rate_limiting_ends_as_a_fetch_error_with_the_status(monkeypatch):
    opener, calls = _opener_failing([429] * 10)
    monkeypatch.setattr(cp.urllib.request, "build_opener", lambda *handlers: opener)
    monkeypatch.setattr(cp.time, "sleep", lambda s: None)
    with pytest.raises(cp.FetchError) as caught:
        cp._hf_linked_sha256("o/r", "f")
    assert caught.value.status == 429 and len(calls) == len(cp._HF_RETRY_DELAYS) + 1


def test_a_missing_file_is_not_retried(monkeypatch):
    opener, calls = _opener_failing([404] * 10)
    monkeypatch.setattr(cp.urllib.request, "build_opener", lambda *handlers: opener)
    monkeypatch.setattr(cp.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("slept")))
    with pytest.raises(cp.FetchError) as caught:
        cp._hf_linked_sha256("o/r", "f")
    assert caught.value.status == 404 and len(calls) == 1


# ---- the finegrained-fp8 Hub kernel pin ---------------------------------------


def _kernel_pin():
    worker = "localm/inference/backends/_hf_worker.py"
    return (cp._read_const(worker, r'^FP8_KERNEL_VERSION = (\d+)'),
            cp._read_const(worker, r'^FP8_KERNEL_REVISION = "([0-9a-f]{40})"'))


def _kernel_row(monkeypatch, routes):
    _router(monkeypatch, routes)
    spec = next(s for s in cp.build_registry() if s.name == cp._HF_KERNEL_NAME)
    return cp.run_checks([spec], NOW)[0]


def _refs(version, commit):
    return {"branches": [{"name": "main", "targetCommit": "f" * 40},
                         {"name": f"v{version}", "targetCommit": commit}]}


def test_the_kernel_pin_is_current_when_its_version_points_at_the_pinned_commit(monkeypatch):
    version, pinned = _kernel_pin()
    row = _kernel_row(monkeypatch, {"/refs": _refs(version, pinned), "/tree/": []})
    assert row.status == cp.CURRENT


def test_a_kernel_version_that_moved_on_reads_behind(monkeypatch):
    version, _pinned = _kernel_pin()
    row = _kernel_row(monkeypatch, {"/refs": _refs(version, "a" * 40), "/tree/": []})
    assert row.status == cp.BEHIND and "now points at" in row.detail


def test_a_pinned_commit_the_kernel_hub_no_longer_has_reads_inconsistent(monkeypatch):
    version, pinned = _kernel_pin()
    row = _kernel_row(monkeypatch, {"/refs": _refs(version, pinned),
                                    "/tree/": cp.FetchError("gone", status=404)})
    assert row.status == cp.INCONSISTENT and "no longer exists" in row.detail


def test_a_retired_kernel_version_reads_inconsistent(monkeypatch):
    _version, pinned = _kernel_pin()
    row = _kernel_row(monkeypatch, {"/refs": {"branches": [{"name": "main", "targetCommit": "f" * 40}]},
                                    "/tree/": []})
    assert row.status == cp.INCONSISTENT


def test_an_unreachable_kernel_hub_reads_unknown(monkeypatch):
    row = _kernel_row(monkeypatch, {"/refs": cp.FetchError("timed out")})
    assert row.status == cp.UNKNOWN


# ---- the Kokoro voice model commit ---------------------------------------------


def _kokoro_commit():
    return cp._read_const("localm/plugins/builtin/tts/static/tts-util.js",
                          r'^export const KOKORO_PINNED_REVISION = "([0-9a-f]{40})"')


def _kokoro_row(monkeypatch, head, *, files=None):
    _router(monkeypatch, {"/api/models/": {"sha": head}})
    monkeypatch.setattr(cp, "_hf_linked_sha256", files or (lambda repo, path, revision="main": None))
    spec = next(s for s in cp.build_registry() if s.name == cp._KOKORO_NAME)
    return cp.run_checks([spec], NOW)[0]


def test_kokoro_commit_is_current_when_the_repository_is_still_there(monkeypatch):
    assert _kokoro_row(monkeypatch, _kokoro_commit()).status == cp.CURRENT


def test_a_newer_kokoro_commit_reads_behind(monkeypatch):
    row = _kokoro_row(monkeypatch, "b" * 40)
    assert row.status == cp.BEHIND and "newer commit" in row.detail


def test_a_kokoro_file_the_pinned_commit_no_longer_serves_reads_inconsistent(monkeypatch):
    def files(repo, path, revision="main"):
        if path == "onnx/model.onnx":
            raise cp.FetchError("gone", status=404)
    row = _kokoro_row(monkeypatch, _kokoro_commit(), files=files)
    assert row.status == cp.INCONSISTENT and "onnx/model.onnx" in row.detail


def test_a_bundle_that_does_not_fetch_the_voices_at_the_pin_reads_inconsistent(monkeypatch, tmp_path):
    _router(monkeypatch, {"/api/models/": {"sha": _kokoro_commit()}})
    spec = next(s for s in cp.build_registry() if s.name == cp._KOKORO_NAME)
    real = cp._read_text

    def text(rel):
        return "unpatched bundle" if rel.endswith("kokoro.min.js") else real(rel)

    monkeypatch.setattr(cp, "_read_text", text)
    row = cp.run_checks([spec], NOW)[0]
    assert row.status == cp.INCONSISTENT and "voices" in row.detail


def test_an_unreachable_hub_reads_unknown_for_the_kokoro_commit(monkeypatch):
    _router(monkeypatch, {"/api/models/": cp.FetchError("timed out")})
    monkeypatch.setattr(cp, "_hf_linked_sha256", lambda repo, path, revision="main": None)
    spec = next(s for s in cp.build_registry() if s.name == cp._KOKORO_NAME)
    assert cp.run_checks([spec], NOW)[0].status == cp.UNKNOWN


# ---- pins read from the product's own tables ----------------------------------


def test_the_embedding_and_native_model_pins_are_read_from_the_product():
    labels = {p[0] for p in cp.hf_file_pins()}
    assert any(label.startswith("embedding ") for label in labels)
    assert any(label.startswith("native image ") for label in labels)
    assert any(label.startswith("native video ") and label.endswith(" vae") for label in labels)
    for _label, repo, path, revision, sha in cp.hf_file_pins():
        assert repo.count("/") == 1 and path and revision and cp._HEX64_RE.match(sha)


def test_a_changed_native_model_digest_is_stale(monkeypatch):
    victim = next(p for p in cp.hf_file_pins() if p[0].startswith("native image "))

    def served(repo, path, revision="main"):
        return "0" * 64 if (repo, path) == (victim[1], victim[2]) else next(
            p[4] for p in cp.hf_file_pins() if (p[1], p[2]) == (repo, path))
    row = _hf_row(monkeypatch, served)
    assert row.status == cp.STALE and victim[0] in row.detail


def test_a_constructor_entry_that_is_not_what_the_reader_expects_is_a_local_read_error(monkeypatch, tmp_path):
    (tmp_path / "localm" / "plugins" / "builtin" / "image" / "backends").mkdir(parents=True)
    (tmp_path / "localm" / "plugins" / "builtin" / "image" / "backends" / "native.py").write_text(
        "RECOMMENDED_MODELS = (Model(name='x'),)\n", encoding="utf-8")
    monkeypatch.setattr(cp, "REPO", tmp_path)
    with pytest.raises(cp.LocalReadError, match="lacks"):
        cp._hf_native_image_pins()


# ---- torch in uv.lock ----------------------------------------------------------


def _torch_row(monkeypatch, tmp_path, lock, latest):
    (tmp_path / "uv.lock").write_text(lock, encoding="utf-8")
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(cp, "_pypi_latest", latest)
    return cp._check_torch_lock(NOW)


def _lock(v):
    return ('[[package]]\nname = "torch"\nversion = "2.11.0+rocm7.13.0"\nsource = { url = "https://x/t.whl" }\n\n'
            f'[[package]]\nname = "torch"\nversion = "{v}"\nsource = {{ registry = "https://pypi.org/simple" }}\n')


def test_torch_in_the_lock_is_current_behind_or_stale_against_pypi(monkeypatch, tmp_path):
    row = _torch_row(monkeypatch, tmp_path, _lock("2.13.0"), lambda p: ("2.13.0", NOW))
    assert row.status == cp.CURRENT and row.pinned == "2.13.0"
    row = _torch_row(monkeypatch, tmp_path, _lock("2.13.0"), lambda p: ("2.14.1", NOW - dt.timedelta(days=13)))
    assert row.status == cp.BEHIND and row.days == 13 and row.latest == "2.14.1"
    row = _torch_row(monkeypatch, tmp_path, _lock("2.13.0"), lambda p: ("2.14.1", NOW - dt.timedelta(days=400)))
    assert row.status == cp.STALE


def test_torch_reads_the_pypi_entry_not_the_local_version_wheel(monkeypatch, tmp_path):
    row = _torch_row(monkeypatch, tmp_path, _lock("2.13.0"), lambda p: ("2.13.0", NOW))
    assert row.pinned == "2.13.0"


def test_a_torch_entry_from_another_source_with_a_plain_version_is_not_taken_for_the_pypi_one(monkeypatch, tmp_path):
    other = '[[package]]\nname = "torch"\nversion = "9.9.9"\nsource = { url = "https://example.invalid/t.whl" }\n\n'
    lock = other + _lock("2.13.0")
    row = _torch_row(monkeypatch, tmp_path, lock, lambda p: ("2.13.0", NOW))
    assert row.pinned == "2.13.0" and row.status == cp.CURRENT


def test_a_lock_without_a_pypi_torch_is_a_local_read_error_row(monkeypatch, tmp_path):
    spec = next(s for s in cp.build_registry() if s.name == "torch (uv.lock)")
    (tmp_path / "uv.lock").write_text('name = "torch"\nversion = "2.11.0+rocm7.13.0"\n', encoding="utf-8")
    monkeypatch.setattr(cp, "REPO", tmp_path)
    monkeypatch.setattr(cp, "_pypi_latest", lambda p: ("2.14.0", NOW))
    row = cp.run_checks([spec], NOW)[0]
    assert row.status == cp.INCONSISTENT and "no torch entry" in row.detail


def test_an_unreachable_pypi_reads_unknown_for_torch(monkeypatch, tmp_path):
    def down(p):
        raise cp.FetchError("offline")
    assert _torch_row(monkeypatch, tmp_path, _lock("2.13.0"), down).status == cp.UNKNOWN


def test_the_real_lock_has_a_pypi_torch_entry():
    assert cp._LOCK_TORCH_RE.search(cp._read_text("uv.lock")) is not None
