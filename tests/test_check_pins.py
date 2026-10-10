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


def test_a_renamed_constant_is_unknown_not_current(monkeypatch, tmp_path):
    (tmp_path / "localm/media/koboldcpp").mkdir(parents=True)
    (tmp_path / "localm/media/koboldcpp/pins.py").write_text("TAGG = 'v1'\n", encoding="utf-8")
    monkeypatch.setattr(cp, "REPO", tmp_path)
    _router(monkeypatch, {"api.github.com": [_rel("v2.0.0", 1)]})
    spec = next(s for s in cp.build_registry() if s.name == "koboldcpp")
    row = spec.check(NOW)
    assert row.status == cp.UNKNOWN and "not found" in row.detail


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
    _router(monkeypatch, {"/releases/tags/": cp.FetchError("HTTP Error 404: Not Found")})
    row = next(s for s in cp.build_registry() if s.name == "Linux CUDA build source").check(NOW)
    assert row.status == cp.STALE and "fall back to vulkan" in row.detail


def test_cuda_linux_source_stale_when_the_release_has_no_cuda_asset(monkeypatch):
    _router(monkeypatch, {"/releases/tags/": _rel("b1", 1, assets=["llama-b1-bin-ubuntu-x64.tar.gz"])})
    row = next(s for s in cp.build_registry() if s.name == "Linux CUDA build source").check(NOW)
    assert row.status == cp.STALE


def test_cuda_linux_source_current_with_a_cuda_asset(monkeypatch):
    _router(monkeypatch, {"/releases/tags/": _rel("b1", 1, assets=["llama-b1-bin-ubuntu-cuda-13-x64.tar.gz"])})
    row = next(s for s in cp.build_registry() if s.name == "Linux CUDA build source").check(NOW)
    assert row.status == cp.CURRENT


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


def test_every_legacy_script_exists_and_accepts_gate():
    for _name, _group, script, _adv, _p, _l in cp._LEGACY:
        text = (cp.REPO / script).read_text(encoding="utf-8")
        assert '"--gate"' in text, script


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
    assert cp.main(["--gate", "--exclude", "not-a-pin"]) == 2
    assert "no known pin" in capsys.readouterr().err


def test_main_only_with_no_match_is_not_a_pass(monkeypatch, capsys):
    _stub_registry(monkeypatch, [cp.CURRENT])
    assert cp.main(["--gate", "--only", "nope"]) == 2


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


def test_registry_covers_every_inventory_group():
    names = {s.name for s in cp.build_registry()}
    for required in ("llama.cpp", "ROCm llama (lemonade)", "ComfyUI", "AMD ROCm wheels", "koboldcpp",
                     "stable-diffusion.cpp", "ComfyUI-GGUF node", "uv (installers + Docker)",
                     "Docker base image", "Linux CUDA build source", "ROCm CPU archive", "Linux CUDA runtime wheels",
                     "vendored marked", "vendored DOMPurify", "vendored highlight.js",
                     "vendored KaTeX", "vendored kokoro-js", "vendored transformers.js",
                     "vendored onnxruntime-web", "vendored jsQR", "vendored Inter font"):
        assert required in names, required
