# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/bump_tts_bundle.py: rebuilding the vendored TTS bundle from pinned npm releases.

The npm install and the esbuild run are injected (``build_fn``), so everything here
runs offline against a copy of the real NOTICE.md, the real guard test and the real
voices list, with small stand-in bundle and runtime files. The tests cover:

  * a dry run changes nothing; --write replaces exactly the bundle, the runtime
    pair, the licence texts, NOTICE.md and the guard test;
  * a refusal edits nothing: an older, equal, prerelease or malformed tag, a
    failed build, an ambiguous or missing region, a changed runtime file set;
  * a write that fails part way restores the files already written;
  * --verify is identical on a consistent tree and not identical once any file differs;
  * the eSpeak NG export line is located by content, on the real shipped bundle too;
  * the build is wired into pin_weekly as review-only and the docs agree with the code.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "bump_tts_bundle.py"
_MJS = "ort-wasm-simd-threaded.asyncify.mjs"
_WASM = "ort-wasm-simd-threaded.asyncify.wasm"
_NOTICE = (_ROOT / "localm/plugins/builtin/tts/static/vendor/NOTICE.md").read_text(encoding="utf-8")
_DEV_ORT = re.search(r"^\| `onnxruntime-web` \| ([^ |]+) \|", _NOTICE, re.M).group(1)
_DEV_ORT_SHA = _DEV_ORT.rsplit("-", 1)[1]
_NEW_DEV_ORT = "1.99.0-dev.20991101-abcdef1234"
_TF_MAJOR = int(re.search(r"^\| `@huggingface/transformers` \(transformers\.js\) \| (\d+)\.", _NOTICE,
                          re.M).group(1))
_PEER = f"^{_TF_MAJOR}.0.0"
_NEXT_PEER = f"^{_TF_MAJOR + 1}.0.0"


def _load():
    spec = importlib.util.spec_from_file_location("bump_tts_bundle", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bump():
    return _load()


def _load_registered(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def repo(tmp_path, bump):
    """A tree holding the real NOTICE, guard test and voices list, with stand-in binaries."""
    for rel in (bump.NOTICE_REL, bump.TEST_REL, bump.VOICES_REL, *bump.LICENSE_RELS.values()):
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_ROOT / rel, dest)
    (tmp_path / bump.BUNDLE_REL).write_bytes(b"old bundle\n")
    runtime = tmp_path / bump.RUNTIME_REL
    runtime.mkdir(parents=True)
    (runtime / _MJS).write_bytes(b"old mjs\n")
    (runtime / _WASM).write_bytes(b"\0asm old wasm")
    return tmp_path


def _shipped_voices(bump) -> set:
    return set(json.loads((_ROOT / bump.VOICES_REL).read_text(encoding="utf-8")))


def _built(bump, *, ort=_DEV_ORT, runtime=None, bundle=b"new bundle\n", peer=_PEER, voices=None,
           runtime_package="onnxruntime-web"):
    voices = voices if voices is not None else _shipped_voices(bump)
    return bump.Built(
        bundle=bundle,
        runtime=runtime if runtime is not None else {_MJS: b"new mjs\n", _WASM: b"\0asm new wasm!"},
        licenses={"kokoro-js": b"kokoro licence\n", "transformers": b"transformers licence\n"},
        versions={"onnxruntime-web": ort}, kokoro_peer_range=peer,
        voices=voices, runtime_package=runtime_package)


class Fakes:
    """Build and registry seams that record how they were called."""

    def __init__(self, bump, built=None, error=None):
        self.bump = bump
        self.built = built or _built(bump)
        self.error = error
        self.builds = []
        self.urls = []
        self.license = b"ort licence\n"

    def build(self, pins, workdir):
        self.builds.append(pins)
        if self.error:
            raise self.error
        return self.built

    def fetch_json(self, url):
        self.urls.append(url)
        return {"sha": "a" * 40}

    def fetch_bytes(self, url):
        self.urls.append(url)
        return self.license

    def run(self, *argv):
        return self.bump.main(list(argv), build_fn=self.build, fetch_json_fn=self.fetch_json,
                              fetch_bytes_fn=self.fetch_bytes)


@pytest.fixture
def fakes(bump, repo, monkeypatch):
    monkeypatch.setattr(bump, "REPO", repo)
    return Fakes(bump)


def _real_pins(bump):
    return bump.read_pins(_ROOT)


def _bump_argv(bump, lib="transformers.js", bump_part=(0, 1, 0)):
    pins = _real_pins(bump)
    cur = pins.kokoro if lib == "kokoro-js" else pins.transformers
    major, minor, patch = (int(x) for x in cur.split("."))
    new = f"{major + bump_part[0]}.{minor + bump_part[1]}.{patch + bump_part[2]}"
    return ["--lib", lib, "--tag", new], new


# --------------------------------------------------------------------------- #
#  Reading and rewriting the real files                                       #
# --------------------------------------------------------------------------- #

def test_the_pins_are_read_from_the_real_notice(bump):
    pins = _real_pins(bump)
    assert [pins.kokoro.count("."), pins.transformers.count("."), pins.phonemizer.count(".")] == [2, 2, 2]
    assert pins.ort.startswith("1.") and pins.esbuild.count(".") == 2


def test_every_region_the_script_edits_matches_the_real_files_exactly_once(bump):
    notice, _ = bump._read_text(_ROOT / bump.NOTICE_REL)
    test_text, _ = bump._read_text(_ROOT / bump.TEST_REL)
    for rx in (*bump._ROW_RES.values(), bump._ESBUILD_RE, bump._PROVENANCE_RE, bump._SIZES_RE,
               bump._ORT_SOURCE_RE):
        assert len(rx.findall(notice)) == 1, rx.pattern
    for key, rx in bump._TEST_RES.items():
        assert len(rx.findall(test_text)) == 1, key


def test_rewriting_notice_with_the_pinned_versions_is_a_no_op_once_sizes_match(bump):
    pins = _real_pins(bump)
    runtime = {n: bump._lf((_ROOT / bump.RUNTIME_REL / n).read_bytes()) if n == _MJS
               else (_ROOT / bump.RUNTIME_REL / n).read_bytes() for n in (_MJS, _WASM)}
    notice, _ = bump._read_text(_ROOT / bump.NOTICE_REL)
    built = _built(bump, runtime=runtime)
    assert bump.rewrite_notice(notice, pins, pins, built, None) == notice


def test_the_stand_in_runtime_names_are_the_ones_the_tree_ships(bump):
    shipped = sorted(p.name for p in (_ROOT / bump.RUNTIME_REL).iterdir())
    assert shipped == sorted([_MJS, _WASM])


def test_the_notice_provenance_names_the_package_the_files_were_copied_from(bump):
    pins = _real_pins(bump)
    notice, _ = bump._read_text(_ROOT / bump.NOTICE_REL)
    from_ort = bump.rewrite_notice(notice, pins, pins, _built(bump), None)
    assert f"from `onnxruntime-web@{pins.ort}`" in from_ort
    assert "does not ship these files in its own `dist/`" in from_ort
    from_transformers = bump.rewrite_notice(
        notice, pins, pins, _built(bump, runtime_package="@huggingface/transformers"), None)
    assert f"from `@huggingface/transformers@{pins.transformers}`" in from_transformers
    assert "onnxruntime-web@" not in bump._PROVENANCE_RE.search(from_transformers).group(0)
    assert bump.rewrite_notice(from_transformers, pins, pins, _built(bump), None) == from_ort


def test_the_notice_size_listing_follows_the_runtime_files_and_keeps_its_columns(bump):
    pins = _real_pins(bump)
    notice, _ = bump._read_text(_ROOT / bump.NOTICE_REL)
    built = _built(bump, runtime={"ort-wasm-simd-threaded.mjs": b"x" * 5,
                                  "ort-wasm-simd-threaded.wasm": b"y" * 1234567})
    out = bump.rewrite_notice(notice, pins, pins, built, None)
    listing = [line for line in out.splitlines() if line.startswith("    onnxruntime/ort-wasm")]
    assert len(listing) == 2 and len(listing[0]) == len(listing[1])
    assert listing[0].startswith("    onnxruntime/ort-wasm-simd-threaded.mjs ") and listing[0].endswith(" 5 B")
    assert listing[1].endswith("wasm  1234567 B")
    assert f"onnxruntime/{_MJS}" not in out


def test_the_notice_documents_the_entry_and_the_flags_the_script_builds_with(bump):
    notice, _ = bump._read_text(_ROOT / bump.NOTICE_REL)
    for line in bump.ENTRY.strip().splitlines():
        assert line in notice
    for flag in bump.ESBUILD_ARGS:
        assert flag in notice


def test_the_guard_test_pins_follow_the_runtime_bytes(bump):
    pins = _real_pins(bump)
    built = _built(bump)
    text, _ = bump._read_text(_ROOT / bump.TEST_REL)
    out = bump.rewrite_guard_test(text, pins, built)
    assert bump._sha256_b64(built.runtime[_WASM]) in out
    assert bump._sha256_b64(built.runtime[_MJS]) in out
    assert f"const EXPECTED_WASM_BYTES = {len(built.runtime[_WASM])};" in out


def test_the_guard_test_follows_the_runtime_names_and_the_package_they_came_from(bump):
    pins = _real_pins(bump)
    pins.ort = _NEW_DEV_ORT
    built = _built(bump, ort=_NEW_DEV_ORT, runtime={"a.mjs": b"m\n", "b.wasm": b"\0asm"})
    text, _ = bump._read_text(_ROOT / bump.TEST_REL)
    out = bump.rewrite_guard_test(text, pins, built)
    assert 'const MJS = "a.mjs";' in out and 'const WASM = "b.wasm";' in out
    assert f'const VENDORED_ORT = "{_NEW_DEV_ORT}";' in out
    assert f"npm pack onnxruntime-web@{_NEW_DEV_ORT}  ->  package/dist/" in out
    via_transformers = _built(bump, runtime=built.runtime, runtime_package="@huggingface/transformers")
    out = bump.rewrite_guard_test(text, pins, via_transformers)
    assert f"npm pack @huggingface/transformers@{pins.transformers}  ->  package/dist/" in out


def test_a_region_found_twice_is_refused_by_the_rewrite(bump):
    notice, _ = bump._read_text(_ROOT / bump.NOTICE_REL)
    pins = _real_pins(bump)
    row = next(line for line in notice.splitlines() if line.startswith("| `phonemizer`"))
    with pytest.raises(bump.Refused, match="found 2"):
        bump.rewrite_notice(notice + "\n" + row + "\n", pins, pins, _built(bump), None)
    guard, _ = bump._read_text(_ROOT / bump.TEST_REL)
    with pytest.raises(bump.Refused, match="found 2"):
        bump.rewrite_guard_test(guard + 'const PINNED_MJS_HASH = "x";\n', pins, _built(bump))


def test_a_notice_missing_a_region_is_refused_by_name(bump):
    notice, _ = bump._read_text(_ROOT / bump.NOTICE_REL)
    pins = _real_pins(bump)
    broken = notice.replace("`esbuild` ", "`esbuild`  ")
    with pytest.raises(bump.Refused, match="esbuild"):
        bump.rewrite_notice(broken, pins, pins, _built(bump), None)


# --------------------------------------------------------------------------- #
#  The export line                                                            #
# --------------------------------------------------------------------------- #

def test_the_export_names_are_located_on_the_real_shipped_bundle(bump):
    text = (_ROOT / bump.BUNDLE_REL).read_bytes().decode("utf-8")
    appended = re.search(r"\nexport\{([\w$]+) as espeakWorker,([\w$]+) as espeakFS\};\n$", text)
    assert appended
    upstream = text[:appended.start() + 1]
    assert bump.locate_espeak_exports(upstream) == appended.groups()


_SNIPPET = ("var A1=new Promise((s=>{n.calledRun?s(new n.eSpeakNGWorker):n.onRuntimeInitialized=()=>"
            "s(new n.eSpeakNGWorker)}));n.FS_createDataFile=Q9.createDataFile,n.FS_unlink=Q9.unlink;\n")


def test_the_export_line_is_appended_with_the_names_found_in_that_build(bump):
    out = bump.append_exports(_SNIPPET.encode()).decode()
    assert out == _SNIPPET + "\nexport{A1 as espeakWorker,Q9 as espeakFS};\n"


@pytest.mark.parametrize("upstream, message", [
    (_SNIPPET.replace("eSpeakNGWorker", "other"), "eSpeakNGWorker promise"),
    (_SNIPPET + _SNIPPET, "eSpeakNGWorker promise"),
    (_SNIPPET.replace("FS_createDataFile=Q9", "FS_createDataFile=Q8") + "n.FS_createDataFile=Q9.createDataFile;\n",
     "filesystem object"),
    (_SNIPPET + "export{x as espeakWorker};\n", "already exports"),
    (_SNIPPET.rstrip("\n"), "does not end with a newline"),
    (_SNIPPET.replace("\n", "\r\n"), "carriage return"),
])
def test_an_export_line_that_cannot_be_placed_unambiguously_is_refused(bump, upstream, message):
    with pytest.raises(bump.Refused, match=message):
        bump.append_exports(upstream.encode())


def test_a_shape_change_in_the_bundle_is_reported_as_needing_a_code_update(bump):
    with pytest.raises(bump.Refused, match="needs a code update"):
        bump.append_exports(_SNIPPET.replace("eSpeakNGWorker", "other").encode())


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_smoke_check_accepts_the_exact_export_set_and_refuses_any_other(bump, tmp_path):
    node = shutil.which("node")
    handles = "export const espeakFS = { createDataFile() {} };\nexport const espeakWorker = Promise.resolve(1);\n"
    others = ", ".join(f"{n} = 1" for n in sorted(bump.BUNDLE_EXPORTS - {"espeakFS", "espeakWorker"}))
    good = tmp_path / "good.js"
    good.write_text(f"export const {others};\n{handles}")
    bump.smoke_check(node, good)
    bad = tmp_path / "bad.js"
    bad.write_text("export const KokoroTTS = 1;\n")
    with pytest.raises(bump.Refused, match="exports"):
        bump.smoke_check(node, bad)
    wrong = tmp_path / "wrong.js"
    wrong.write_text(f"export const {others};\nexport const espeakFS = {{}};\n"
                     "export const espeakWorker = Promise.resolve(1);\n")
    with pytest.raises(bump.Refused, match="not the eSpeak NG handles"):
        bump.smoke_check(node, wrong)
    notpromise = tmp_path / "notpromise.js"
    notpromise.write_text(f"export const {others};\nexport const espeakFS = {{ createDataFile() {{}} }};\n"
                          "export const espeakWorker = 1;\n")
    with pytest.raises(bump.Refused, match="not the eSpeak NG handles"):
        bump.smoke_check(node, notpromise)
    broken = tmp_path / "broken.js"
    broken.write_text("throw new Error('boom');\n")
    with pytest.raises(bump.Refused, match="boom"):
        bump.smoke_check(shutil.which("node"), broken)


# --------------------------------------------------------------------------- #
#  Dry run, write, refusals                                                   #
# --------------------------------------------------------------------------- #

def test_a_dry_run_changes_nothing(bump, repo, fakes, capsys):
    before = _snapshot(repo)
    argv, new = _bump_argv(bump)
    assert fakes.run(*argv) == 0
    assert _snapshot(repo) == before
    out = capsys.readouterr().out
    assert "dry run" in out and "FILES EDITED" in out
    assert fakes.builds[0].transformers == new


def test_write_replaces_exactly_the_bundle_runtime_licences_notice_and_guard(bump, repo, fakes):
    before = _snapshot(repo)
    argv, new = _bump_argv(bump)
    assert fakes.run(*argv, "--write") == 0
    after = _snapshot(repo)
    changed = sorted(k.replace("\\", "/") for k in after if after[k] != before.get(k))
    assert changed == sorted([
        bump.BUNDLE_REL, f"{bump.RUNTIME_REL}/{_MJS}", f"{bump.RUNTIME_REL}/{_WASM}",
        bump.LICENSE_RELS["kokoro-js"], bump.LICENSE_RELS["transformers"],
        bump.NOTICE_REL, bump.TEST_REL])
    assert (repo / bump.BUNDLE_REL).read_bytes() == b"new bundle\n"
    assert (repo / bump.RUNTIME_REL / _WASM).read_bytes() == b"\0asm new wasm!"
    assert bump.read_pins(repo).transformers == new
    guard = (repo / bump.TEST_REL).read_text(encoding="utf-8")
    assert f'const VENDORED_TRANSFORMERS = "{new}";' in guard
    assert bump._sha256_b64(b"\0asm new wasm!") in guard


def test_the_onnxruntime_licence_is_fetched_only_when_that_version_moved(bump, repo, fakes):
    argv, _ = _bump_argv(bump)
    assert fakes.run(*argv, "--write") == 0
    assert fakes.urls == []
    assert (repo / bump.LICENSE_RELS["onnxruntime-web"]).read_bytes() == (
        _ROOT / bump.LICENSE_RELS["onnxruntime-web"]).read_bytes()


def test_a_moved_dev_runtime_takes_its_licence_from_the_commit_it_names(bump, repo, fakes):
    fakes.built = _built(bump, ort=_NEW_DEV_ORT)
    argv, _ = _bump_argv(bump)
    assert fakes.run(*argv, "--write") == 0
    assert fakes.urls == [
        "https://api.github.com/repos/microsoft/onnxruntime/commits/abcdef1234",
        f"https://raw.githubusercontent.com/microsoft/onnxruntime/{'a' * 40}/LICENSE"]
    notice = (repo / bump.NOTICE_REL).read_text(encoding="utf-8")
    assert f"at commit `{'a' * 40}`" in notice and _NEW_DEV_ORT in notice
    assert bump._lf((repo / bump.LICENSE_RELS["onnxruntime-web"]).read_bytes()) == b"ort licence\n"


def test_a_moved_release_runtime_takes_its_licence_from_the_release_tag(bump, repo, fakes):
    fakes.built = _built(bump, ort="1.23.1")
    argv, _ = _bump_argv(bump)
    assert fakes.run(*argv, "--write") == 0
    assert fakes.urls == ["https://raw.githubusercontent.com/microsoft/onnxruntime/v1.23.1/LICENSE"]
    notice = (repo / bump.NOTICE_REL).read_text(encoding="utf-8")
    assert "at tag `v1.23.1`" in notice and "release the pinned version names" in notice
    assert bump.read_pins(repo).ort == "1.23.1"


@pytest.mark.parametrize("make_tag", [
    lambda cur: cur,
    lambda cur: "0.0.1",
    lambda cur: cur + "-rc.1",
    lambda cur: "v" + cur,
    lambda cur: "latest",
])
def test_a_tag_that_is_not_strictly_newer_is_refused_and_nothing_is_built(bump, repo, fakes, make_tag, capsys):
    before = _snapshot(repo)
    cur = _real_pins(bump).transformers
    assert fakes.run("--lib", "transformers.js", "--tag", make_tag(cur), "--write") == 1
    assert _snapshot(repo) == before
    assert fakes.builds == [] and "REFUSED" in capsys.readouterr().out


def test_a_failed_build_leaves_the_tree_untouched(bump, repo, monkeypatch, capsys):
    monkeypatch.setattr(bump, "REPO", repo)
    failing = Fakes(bump, error=bump.Refused("npm install failed (exit 1): E404"))
    before = _snapshot(repo)
    argv, _ = _bump_argv(bump)
    assert failing.run(*argv, "--write") == 1
    assert _snapshot(repo) == before
    assert "E404" in capsys.readouterr().out and len(failing.builds) == 1


def test_an_ambiguous_notice_region_writes_nothing(bump, repo, fakes):
    notice = repo / bump.NOTICE_REL
    text = notice.read_bytes().decode("utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| `phonemizer`"))
    notice.write_bytes((text + "\n" + row + "\n").encode("utf-8"))
    before = _snapshot(repo)
    argv, _ = _bump_argv(bump)
    assert fakes.run(*argv, "--write") == 1
    assert _snapshot(repo) == before


def test_a_changed_runtime_file_set_is_refused_as_needing_a_code_update(bump, repo, fakes, capsys):
    argv, _ = _bump_argv(bump)
    for runtime in ({_MJS: b"m", "other.mjs": b"a"}, {_MJS: b"m", _WASM: b"\0asm", "extra.wasm": b"\0asm"},
                    {_MJS: b"m"}):
        fakes.built = _built(bump, runtime=runtime)
        before = _snapshot(repo)
        assert fakes.run(*argv, "--write") == 1
        assert _snapshot(repo) == before
        assert "needs a code update" in capsys.readouterr().out


def test_a_runtime_file_missing_from_the_tree_is_created_by_the_write(bump, repo, fakes):
    (repo / bump.RUNTIME_REL / _WASM).unlink()
    argv, _ = _bump_argv(bump)
    assert fakes.run(*argv, "--write") == 0
    assert (repo / bump.RUNTIME_REL / _WASM).read_bytes() == b"\0asm new wasm!"


def test_a_renamed_runtime_replaces_the_old_files_and_a_dry_run_only_reports_it(
        bump, repo, fakes, capsys):
    old_names = ("ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm")
    for name in old_names:
        (repo / bump.RUNTIME_REL / name).write_bytes(b"stale " + name.encode())
    argv, _ = _bump_argv(bump)
    before = _snapshot(repo)
    assert fakes.run(*argv) == 0
    out = capsys.readouterr().out
    assert all(f"remove  {bump.RUNTIME_REL}/{name}" in out for name in old_names)
    assert _snapshot(repo) == before
    assert fakes.run(*argv, "--write") == 0
    assert sorted(p.name for p in (repo / bump.RUNTIME_REL).iterdir()) == sorted([_MJS, _WASM])


def test_a_write_that_fails_after_removing_a_stale_file_brings_it_back(bump, repo, fakes, monkeypatch):
    stale = repo / bump.RUNTIME_REL / "ort-wasm-simd-threaded.jsep.wasm"
    stale.write_bytes(b"\0asm stale")
    before = _snapshot(repo)
    real_unlink = Path.unlink

    def failing_unlink(self, *args, **kwargs):
        if self.name == stale.name:
            real_unlink(self, *args, **kwargs)
            raise OSError("late failure")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    argv, _ = _bump_argv(bump)
    assert fakes.run(*argv, "--write") == 1
    assert _snapshot(repo) == before


def test_a_write_that_fails_part_way_restores_the_files_already_written(bump, repo, fakes, monkeypatch):
    before = _snapshot(repo)
    calls = {"n": 0, "faults": 0}
    real_replace = bump.os.replace

    def failing_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 3:
            calls["faults"] += 1
            raise OSError("disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(bump.os, "replace", failing_replace)
    argv, _ = _bump_argv(bump)
    assert fakes.run(*argv, "--write") == 1
    assert calls["faults"] == 1 and calls["n"] > 3
    assert _snapshot(repo) == before


def test_a_second_run_with_the_same_tag_is_refused_as_not_newer(bump, repo, fakes):
    argv, _ = _bump_argv(bump)
    assert fakes.run(*argv, "--write") == 0
    assert fakes.run(*argv, "--write") == 1


def test_a_kokoro_bump_moves_the_kokoro_row_only(bump, repo, fakes):
    old = bump.read_pins(repo)
    argv, new = _bump_argv(bump, "kokoro-js")
    assert fakes.run(*argv, "--write") == 0
    now = bump.read_pins(repo)
    assert (now.kokoro, now.transformers, now.phonemizer, now.esbuild) == (
        new, old.transformers, old.phonemizer, old.esbuild)
    assert fakes.builds[0].kokoro == new and fakes.builds[0].transformers == old.transformers


def test_a_major_move_is_called_out_when_kokoro_js_accepts_it(bump, repo, fakes, capsys):
    fakes.built = _built(bump, peer=_NEXT_PEER)
    argv, _ = _bump_argv(bump, bump_part=(1, 0, 0))
    assert fakes.run(*argv) == 0
    assert "MAJOR BOUNDARY in transformers.js" in capsys.readouterr().out


def test_a_move_outside_the_range_kokoro_js_declares_is_refused_and_writes_nothing(
        bump, repo, fakes, capsys):
    before = _snapshot(repo)
    argv, _ = _bump_argv(bump, bump_part=(1, 0, 0))
    assert fakes.run(*argv, "--write") == 1
    out = capsys.readouterr().out
    assert f"declares @huggingface/transformers {_PEER}" in out and "needs a code update" in out
    assert _snapshot(repo) == before


def test_a_kokoro_js_and_transformers_major_pair_verified_in_a_browser_is_not_reported(bump):
    ((kokoro, major),) = bump.VERIFIED_BEYOND_PEER_RANGE
    built = _built(bump, peer=f"^{major - 1}.5.1")
    pins = bump.Pins(kokoro=kokoro, transformers=f"{major}.0.0", ort=_DEV_ORT, phonemizer="1.2.1",
                     esbuild="0.28.2")
    assert bump.peer_mismatch(built, pins) is None
    pins.kokoro = "9.9.9"
    assert "declares @huggingface/transformers" in bump.peer_mismatch(built, pins)
    pins.kokoro = kokoro
    pins.transformers = f"{major + 1}.0.0"
    assert "declares @huggingface/transformers" in bump.peer_mismatch(built, pins)


def test_a_voice_the_package_adds_is_listed_but_does_not_block(bump, repo, fakes, capsys):
    fakes.built = _built(bump, voices=_shipped_voices(bump) | {"zz_new"})
    argv, _ = _bump_argv(bump)
    assert fakes.run(*argv) == 0
    assert "voices in the kokoro-js package but not in voices.json: zz_new" in capsys.readouterr().out


def test_a_shipped_voice_the_package_dropped_is_refused_and_writes_nothing(bump, repo, fakes, capsys):
    fakes.built = _built(bump, voices=_shipped_voices(bump) - {"af_heart"})
    before = _snapshot(repo)
    argv, _ = _bump_argv(bump)
    assert fakes.run(*argv, "--write") == 1
    assert "af_heart" in capsys.readouterr().out
    assert _snapshot(repo) == before


def test_the_tag_and_the_library_are_given_together(bump, repo, fakes):
    for argv in (["--lib", "kokoro-js"], ["--tag", "9.9.9"], [], ["--verify", "--tag", "9.9.9"],
                 ["--verify", "--write"]):
        with pytest.raises(SystemExit) as e:
            fakes.run(*argv)
        assert e.value.code == 2


# --------------------------------------------------------------------------- #
#  --verify                                                                   #
# --------------------------------------------------------------------------- #

@pytest.fixture
def consistent(bump, repo, fakes):
    """The fixture tree rewritten so that it is exactly what the fake build produces."""
    pins = bump.read_pins(repo)
    pins.ort = _DEV_ORT
    plan = bump.build_plan(repo, pins, pins, fakes.built)
    bump.apply_changes(plan["changes"])
    (repo / bump.LICENSE_RELS["onnxruntime-web"]).write_bytes(fakes.license)
    return repo


def test_planning_the_pinned_versions_again_on_a_consistent_tree_changes_nothing(bump, consistent, fakes):
    pins = bump.read_pins(consistent)
    assert bump.build_plan(consistent, pins, pins, fakes.built)["changes"] == []


def test_verify_reports_identical_on_a_consistent_tree(bump, consistent, fakes, capsys):
    before = _snapshot(consistent)
    assert fakes.run("--verify") == 0
    assert "IDENTICAL" in capsys.readouterr().out
    assert _snapshot(consistent) == before


def test_verify_reads_the_licence_by_the_abbreviated_sha_without_the_github_api(bump, consistent, fakes):
    assert fakes.run("--verify") == 0
    assert fakes.urls == [
        f"https://raw.githubusercontent.com/microsoft/onnxruntime/{_DEV_ORT_SHA}/LICENSE"]


@pytest.mark.parametrize("victim", ["bundle", "wasm", "mjs", "licence", "ort_licence", "notice", "guard",
                                    "ort_source"])
def test_verify_reports_not_identical_when_any_one_file_differs(bump, consistent, fakes, capsys, victim):
    path = {"bundle": bump.BUNDLE_REL, "wasm": f"{bump.RUNTIME_REL}/{_WASM}",
            "mjs": f"{bump.RUNTIME_REL}/{_MJS}", "licence": bump.LICENSE_RELS["kokoro-js"],
            "ort_licence": bump.LICENSE_RELS["onnxruntime-web"], "notice": bump.NOTICE_REL,
            "guard": bump.TEST_REL, "ort_source": bump.NOTICE_REL}[victim]
    target = consistent / path
    data = target.read_bytes()
    owned = {"notice": (rb"(asyncify\.mjs +)(\d+)", rb"\g<1>1"),
             "guard": (rb"(EXPECTED_WASM_BYTES = )(\d+)", rb"\g<1>7"),
             "ort_source": (rb"(repository at commit `)([0-9a-f]{4})", rb"\g<1>zzzz")}
    tampered = re.sub(*owned[victim], data, count=1) if victim in owned else data + b"x"
    assert tampered != data
    target.write_bytes(tampered)
    assert fakes.run("--verify") == 1
    assert "NOT IDENTICAL" in capsys.readouterr().out


def test_verify_reports_not_identical_when_a_stale_runtime_file_sits_in_the_tree(
        bump, consistent, fakes, capsys):
    (consistent / bump.RUNTIME_REL / "ort-wasm-simd-threaded.jsep.wasm").write_bytes(b"\0asm stale")
    assert fakes.run("--verify") == 1
    out = capsys.readouterr().out
    assert "NOT IDENTICAL" in out and "remove  " in out


def test_verify_refuses_a_tree_whose_recorded_runtime_version_is_not_what_transformers_pins(
        bump, consistent, fakes, capsys):
    fakes.built = _built(bump, ort=_NEW_DEV_ORT, runtime=fakes.built.runtime, bundle=fakes.built.bundle)
    assert fakes.run("--verify") == 1
    assert "REFUSED" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
#  Wiring                                                                     #
# --------------------------------------------------------------------------- #

def test_the_weekly_run_advances_the_two_bundle_libraries_through_this_script_and_never_merges_them(bump):
    mod = _load_registered("pin_weekly", _ROOT / "scripts" / "pin_weekly.py")
    advs = {a.key: a for a in mod.build_advancers() if a.bump_script == "scripts/bump_tts_bundle.py"}
    assert sorted(advs) == ["vendored-kokoro-js", "vendored-transformersjs"]
    for adv in advs.values():
        assert adv.confirm_script is None
        assert adv.post_bump_cmds == (("npm", "ci"), ("npm", "test"))
        assert adv.auto_merge("1.2.3", "1.2.4") is False
        assert adv.bump_args[0] == "--lib" and adv.bump_args[1] in bump.LIBS
    assert {a.bump_args[1] for a in advs.values()} == set(bump.LIBS)


def test_the_pin_report_names_the_advancer_of_each_bundle_row():
    cp = _load_registered("check_pins", _ROOT / "scripts" / "check_pins.py")
    assert "rebuilds the TTS bundle" in cp._vendored_advancer("kokoro-js")
    assert "rebuilds the TTS bundle" in cp._vendored_advancer("transformers.js")
    assert "transformers.js" in cp._vendored_advancer("onnxruntime-web")
    for name in ("vendored transformers.js", "vendored onnxruntime-web"):
        assert name not in cp.ACCEPTED_BLOCKERS


# --------------------------------------------------------------------------- #
#  The voices pin                                                             #
# --------------------------------------------------------------------------- #

_VOICES = ("let d=`https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/${o}.bin`,m;"
           'try{m=await caches.open("kokoro-voices");let w=await m.match(d);if(w)return await w.arrayBuffer()}'
           'catch(w){console.warn("Unable to open cache",w)}let C=await fetch(d),b=await C.arrayBuffer();'
           "if(m)try{await m.put(d,new Response(b,{headers:C.headers}))}catch(w){}\n")
_REV = "1" * 40


def test_the_voices_fetch_is_pinned_and_the_cache_key_is_left_alone(bump):
    out = bump.pin_voices_fetch(_VOICES.encode(), _REV).decode()
    assert f'let C=await fetch(d.replace("/resolve/main/","/resolve/{_REV}/")),b=' in out
    assert "m.match(d)" in out and "m.put(d," in out


def test_the_voices_pin_follows_the_names_the_minifier_chose(bump):
    renamed = _VOICES.replace("d", "z").replace("C", "Q").replace("zeclare", "declare")
    assert bump.pin_voices_fetch(renamed.encode(), _REV).decode().count(f"/resolve/{_REV}/") == 1


@pytest.mark.parametrize("upstream", [_VOICES.replace("await fetch(d)", "await fetch(x)"),
                                      _VOICES + _VOICES, "no voices here\n"])
def test_a_bundle_without_exactly_one_voices_fetch_is_refused(bump, upstream):
    with pytest.raises(bump.Refused, match="exactly one voices fetch"):
        bump.pin_voices_fetch(upstream.encode(), _REV)


def test_the_shipped_bundle_is_the_unpinned_one_with_the_voices_pin_applied(bump):
    shipped = (_ROOT / bump.BUNDLE_REL).read_bytes()
    revision = bump.pinned_kokoro_revision()
    pinned_call = re.compile(
        rb'fetch\((\w+)\.replace\("/resolve/main/","/resolve/' + revision.encode() + rb'/"\)\)')
    assert len(pinned_call.findall(shipped)) == 1
    assert bump.pin_voices_fetch(pinned_call.sub(rb"fetch(\1)", shipped), revision) == shipped


def test_the_pinned_revision_is_read_from_tts_util(bump, tmp_path):
    assert bump.pinned_kokoro_revision() == bump.pinned_kokoro_revision(_ROOT)
    (tmp_path / "localm/plugins/builtin/tts/static").mkdir(parents=True)
    with pytest.raises(bump.Refused, match="cannot read"):
        bump.pinned_kokoro_revision(tmp_path)
    (tmp_path / bump.TTS_UTIL_REL).write_text("no pin\n", encoding="utf-8")
    with pytest.raises(bump.Refused, match="exactly one KOKORO_PINNED_REVISION"):
        bump.pinned_kokoro_revision(tmp_path)


# --------------------------------------------------------------------------- #
#  The real thing (npm, network)                                              #
# --------------------------------------------------------------------------- #

@pytest.mark.integration
def test_rebuilding_the_pinned_versions_reproduces_the_shipped_files(bump, tmp_path):
    """Needs node, npm and the registry: the committed tree must equal a fresh build."""
    assert shutil.which("npm") and shutil.which("node"), "node and npm are required"
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--verify"], cwd=_ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(_ROOT), "TMP": str(tmp_path), "TEMP": str(tmp_path)})
    assert proc.returncode == 0, proc.stdout + proc.stderr


# --------------------------------------------------------------------------- #
#  The install checks and the helpers around the build                        #
# --------------------------------------------------------------------------- #

_BUNDLE = (b'var a="3.8.1";const u="' + _MJS.encode() + b'",w="' + _WASM.encode() + b'";\n')


def _pins(bump):
    return bump.Pins(kokoro="1.2.1", transformers="3.8.1", ort=_DEV_ORT, phonemizer="1.2.1", esbuild="0.28.2")


def _package(root: Path, path: str, version: str, deps: dict | None = None, licence: bool = True) -> Path:
    folder = root / "node_modules" / path
    folder.mkdir(parents=True)
    (folder / "package.json").write_text(json.dumps({"name": path, "version": version,
                                                      "dependencies": deps or {}}), encoding="utf-8")
    if licence:
        (folder / "LICENSE").write_bytes(f"{path} licence\n".encode())
    return folder


@pytest.fixture
def installed(tmp_path, bump):
    """A node_modules tree shaped like a real install of the pinned versions."""
    pins = _pins(bump)
    _package(tmp_path, "kokoro-js", pins.kokoro, {"@huggingface/transformers": "^3.5.1"})
    (tmp_path / "node_modules" / "kokoro-js" / "voices").mkdir()
    (tmp_path / "node_modules" / "kokoro-js" / "voices" / "af_heart.bin").write_bytes(b"v")
    transformers = _package(tmp_path, "@huggingface/transformers", pins.transformers)
    (transformers / "dist").mkdir()
    (transformers / "dist" / _MJS).write_bytes(b"mjs bytes\n")
    (transformers / "dist" / _WASM).write_bytes(b"\0asm wasm bytes")
    _package(tmp_path, "phonemizer", pins.phonemizer)
    _package(tmp_path, "esbuild", pins.esbuild)
    _package(tmp_path, "onnxruntime-web", pins.ort)
    return tmp_path


def test_a_correct_install_reports_every_installed_version(bump, installed):
    versions, kokoro = bump.check_install(installed, _pins(bump))
    assert versions == {"kokoro-js": "1.2.1", "transformers": "3.8.1", "phonemizer": "1.2.1",
                        "esbuild": "0.28.2", "onnxruntime-web": _DEV_ORT}
    assert kokoro["dependencies"] == {"@huggingface/transformers": "^3.5.1"}


def test_an_install_at_another_version_than_requested_is_refused(bump, installed):
    pins = _pins(bump)
    pins.transformers = "3.9.0"
    with pytest.raises(bump.Refused, match="installed transformers 3.8.1, not the requested 3.9.0"):
        bump.check_install(installed, pins)


@pytest.mark.parametrize("nested, package", [
    ("kokoro-js/node_modules/phonemizer", "phonemizer"),
    ("@huggingface/transformers/node_modules/onnxruntime-web", "onnxruntime-web"),
    ("phonemizer/node_modules/@huggingface/transformers", "@huggingface/transformers"),
    ("esbuild/node_modules/kokoro-js", "kokoro-js"),
])
def test_a_second_copy_of_a_bundled_package_is_refused(bump, installed, nested, package):
    _package(installed, nested, "9.9.9")
    with pytest.raises(bump.Refused, match=rf"expected {re.escape(package)} installed once, found 2"):
        bump.check_install(installed, _pins(bump))


def test_a_package_that_is_not_installed_is_refused(bump, installed):
    shutil.rmtree(installed / "node_modules" / "phonemizer")
    with pytest.raises(bump.Refused, match="phonemizer is not installed"):
        bump.check_install(installed, _pins(bump))


def _collect(bump, installed, bundle=_BUNDLE):
    pins = _pins(bump)
    versions, kokoro = bump.check_install(installed, pins)
    return bump.collect_artifacts(installed, bundle, pins, versions, kokoro, "warn")


def test_the_artifacts_are_the_installed_files_byte_for_byte(bump, installed):
    built = _collect(bump, installed)
    assert built.runtime == {_MJS: b"mjs bytes\n", _WASM: b"\0asm wasm bytes"}
    assert built.licenses == {"kokoro-js": b"kokoro-js licence\n",
                              "transformers": b"@huggingface/transformers licence\n"}
    assert built.voices == {"af_heart"} and built.kokoro_peer_range == "^3.5.1" and built.warnings == "warn"


def test_a_bundle_without_the_transformers_version_string_is_refused(bump, installed):
    with pytest.raises(bump.Refused, match='version string "3.8.1"'):
        _collect(bump, installed, _BUNDLE.replace(b"3.8.1", b"3.8.0"))


def test_a_bundle_naming_no_runtime_file_is_refused(bump, installed):
    with pytest.raises(bump.Refused, match="names no onnxruntime runtime file"):
        _collect(bump, installed, b'var a="3.8.1";\n')


def test_a_runtime_file_the_package_does_not_ship_is_refused_as_needing_a_code_update(bump, installed):
    (installed / "node_modules" / "@huggingface" / "transformers" / "dist" / _WASM).unlink()
    with pytest.raises(bump.Refused, match="ships it in dist/; this release needs a code update"):
        _collect(bump, installed)


def _move_runtime_to_onnxruntime_web(installed):
    transformers = installed / "node_modules" / "@huggingface" / "transformers" / "dist"
    ort = installed / "node_modules" / "onnxruntime-web" / "dist"
    ort.mkdir()
    for name in (_MJS, _WASM):
        (ort / name).write_bytes((transformers / name).read_bytes() + b" from ort")
        (transformers / name).unlink()
    return ort


def test_runtime_files_the_transformers_package_lacks_come_from_onnxruntime_web(bump, installed):
    _move_runtime_to_onnxruntime_web(installed)
    built = _collect(bump, installed, _BUNDLE + _DEV_ORT.encode())
    assert built.runtime == {_MJS: b"mjs bytes\n from ort", _WASM: b"\0asm wasm bytes from ort"}
    assert built.runtime_package == "onnxruntime-web"


def test_runtime_files_present_in_transformers_dist_are_taken_from_there(bump, installed):
    assert _collect(bump, installed).runtime_package == "@huggingface/transformers"


def test_runtime_files_taken_from_onnxruntime_web_need_its_version_string_in_the_bundle(bump, installed):
    _move_runtime_to_onnxruntime_web(installed)
    with pytest.raises(bump.Refused, match="does not contain the onnxruntime-web version string"):
        _collect(bump, installed, _BUNDLE)


def test_runtime_files_split_between_two_packages_are_refused(bump, installed):
    ort = _move_runtime_to_onnxruntime_web(installed)
    (installed / "node_modules" / "@huggingface" / "transformers" / "dist" / _MJS).write_bytes(b"x")
    with pytest.raises(bump.Refused, match="more than one package"):
        _collect(bump, installed, _BUNDLE + _DEV_ORT.encode())
    assert ort.is_dir()


@pytest.mark.parametrize("package", ["kokoro-js", "@huggingface/transformers"])
def test_a_package_without_a_licence_file_is_refused(bump, installed, package):
    (installed / "node_modules" / package / "LICENSE").unlink()
    with pytest.raises(bump.Refused, match="ships no LICENSE file"):
        _collect(bump, installed)


def test_the_build_commands_run_without_the_github_token(bump, tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "secret-a")
    monkeypatch.setenv("GH_TOKEN", "secret-b")
    monkeypatch.setenv("KEEP_ME", "yes")
    code = ("import os;print(os.environ.get('GITHUB_TOKEN'), os.environ.get('GH_TOKEN'), "
            "os.environ.get('KEEP_ME'))")
    proc = bump._run([sys.executable, "-c", code], tmp_path, "probe")
    assert proc.stdout.split() == ["None", "None", "yes"]


def test_a_failing_command_is_reported_with_its_exit_code_and_output(bump, tmp_path):
    with pytest.raises(bump.Refused, match=r"probe failed \(exit 3\): boom"):
        bump._run([sys.executable, "-c", "import sys;sys.stderr.write('boom');sys.exit(3)"],
                  tmp_path, "probe")


def test_a_tool_that_is_not_installed_is_refused_by_name(bump, monkeypatch):
    monkeypatch.setattr(bump.shutil, "which", lambda name: None)
    with pytest.raises(bump.Refused, match="node was not found on PATH"):
        bump._tool("node")


def test_a_fetched_licence_with_windows_newlines_is_not_doubled_in_a_windows_tree(bump, repo, fakes):
    path = repo / bump.LICENSE_RELS["kokoro-js"]
    path.write_bytes(b"old\r\nlicence\r\n")
    fakes.built.licenses["kokoro-js"] = b"new\r\nlicence\r\n"
    argv, _ = _bump_argv(bump)
    assert fakes.run(*argv, "--write") == 0
    assert path.read_bytes() == b"new\r\nlicence\r\n"


def test_a_rollback_removes_a_file_the_failed_write_had_created(bump, tmp_path, monkeypatch):
    existing = tmp_path / "a.txt"
    existing.write_bytes(b"old")
    changes = [bump.Change("a.txt", existing, b"old", b"new", False),
               bump.Change("sub/new.txt", tmp_path / "sub" / "new.txt", None, b"created", False),
               bump.Change("c.txt", tmp_path / "c.txt", None, b"boom", False)]
    real = bump.os.replace

    def failing(src, dst):
        if str(dst).endswith("c.txt"):
            raise OSError("disk full")
        return real(src, dst)

    monkeypatch.setattr(bump.os, "replace", failing)
    with pytest.raises(OSError, match="disk full"):
        bump.apply_changes(changes)
    assert existing.read_bytes() == b"old"
    assert not (tmp_path / "sub" / "new.txt").exists() and not (tmp_path / "c.txt").exists()
    assert not list(tmp_path.rglob("*.bump-tmp"))


def test_the_licence_source_in_notice_must_agree_with_the_pinned_version(bump):
    notice, _ = bump._read_text(_ROOT / bump.NOTICE_REL)
    assert bump.ort_source_problem(notice, _real_pins(bump).ort) is None
    assert "names onnxruntime commit" in bump.ort_source_problem(
        notice, _DEV_ORT.rsplit("-", 1)[0] + "-0000000000")
    assert "names onnxruntime commit" in bump.ort_source_problem(notice, "1.23.1")
    assert "no onnxruntime licence source" in bump.ort_source_problem("nothing here", "1.23.1")
    tagged = bump._ORT_REF_RE.sub("microsoft/onnxruntime repository at tag `v1.23.1`", notice)
    assert bump.ort_source_problem(tagged, "1.23.1") is None
