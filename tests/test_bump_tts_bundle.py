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
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "bump_tts_bundle.py"
_MJS = "ort-wasm-simd-threaded.jsep.mjs"
_WASM = "ort-wasm-simd-threaded.jsep.wasm"
_DEV_ORT = "1.22.0-dev.20250409-89f8206ba4"
_NEW_DEV_ORT = "1.23.0-dev.20251101-abcdef1234"


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


def _built(bump, *, ort=_DEV_ORT, runtime=None, bundle=b"new bundle\n", peer="^3.5.1", voices=None):
    return bump.Built(
        bundle=bundle,
        runtime=runtime if runtime is not None else {_MJS: b"new mjs\n", _WASM: b"\0asm new wasm!"},
        licenses={"kokoro-js": b"kokoro licence\n", "transformers": b"transformers licence\n"},
        versions={"onnxruntime-web": ort}, kokoro_peer_range=peer,
        voices=voices if voices is not None else set())


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
    for rx in (*bump._ROW_RES.values(), bump._ESBUILD_RE, bump._COPIES_RE, bump._ENV_VERSION_RE,
               bump._ORT_SOURCE_RE, *bump._SIZE_RES.values()):
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
    assert bump.locate_espeak_exports(text) == ("Y8", "te")


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
    good = tmp_path / "good.js"
    good.write_text("export const " + ", ".join(f"{n} = 1" for n in sorted(bump.BUNDLE_EXPORTS)) + ";\n")
    bump.smoke_check(shutil.which("node"), good)
    bad = tmp_path / "bad.js"
    bad.write_text("export const KokoroTTS = 1;\n")
    with pytest.raises(bump.Refused, match="exports"):
        bump.smoke_check(shutil.which("node"), bad)
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
    fakes.built = _built(bump, runtime={_MJS: b"m", "ort-wasm-simd-threaded.asyncify.mjs": b"a"})
    before = _snapshot(repo)
    argv, _ = _bump_argv(bump)
    assert fakes.run(*argv, "--write") == 1
    assert _snapshot(repo) == before
    assert "needs a code update" in capsys.readouterr().out


def test_a_runtime_file_missing_from_the_tree_is_refused(bump, repo, fakes):
    (repo / bump.RUNTIME_REL / _WASM).unlink()
    argv, _ = _bump_argv(bump)
    assert fakes.run(*argv, "--write") == 1


def test_a_write_that_fails_part_way_restores_the_files_already_written(bump, repo, fakes, monkeypatch):
    before = _snapshot(repo)
    calls = {"n": 0}
    real_replace = bump.os.replace

    def failing_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(bump.os, "replace", failing_replace)
    argv, _ = _bump_argv(bump)
    assert fakes.run(*argv, "--write") == 1
    assert calls["n"] == 3
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


def test_a_major_move_and_a_peer_range_it_breaks_are_called_out(bump, repo, fakes, capsys):
    argv, _ = _bump_argv(bump, bump_part=(1, 0, 0))
    assert fakes.run(*argv) == 0
    out = capsys.readouterr().out
    assert "MAJOR BOUNDARY in transformers.js" in out
    assert "declares @huggingface/transformers ^3.5.1" in out


def test_voice_differences_between_the_package_and_voices_json_are_listed(bump, repo, fakes, capsys):
    fakes.built = _built(bump, voices={"af_heart", "zz_new"})
    argv, _ = _bump_argv(bump)
    assert fakes.run(*argv) == 0
    out = capsys.readouterr().out
    assert "voices in the kokoro-js package but not in voices.json: zz_new" in out
    assert "voices in voices.json but not in the kokoro-js package:" in out


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


@pytest.mark.parametrize("victim", ["bundle", "wasm", "mjs", "licence", "ort_licence", "notice", "guard"])
def test_verify_reports_not_identical_when_any_one_file_differs(bump, consistent, fakes, capsys, victim):
    path = {"bundle": bump.BUNDLE_REL, "wasm": f"{bump.RUNTIME_REL}/{_WASM}",
            "mjs": f"{bump.RUNTIME_REL}/{_MJS}", "licence": bump.LICENSE_RELS["kokoro-js"],
            "ort_licence": bump.LICENSE_RELS["onnxruntime-web"], "notice": bump.NOTICE_REL,
            "guard": bump.TEST_REL}[victim]
    target = consistent / path
    data = target.read_bytes()
    owned = {"notice": (rb"(jsep\.mjs +)(\d+)", rb"\g<1>1"),
             "guard": (rb"(EXPECTED_WASM_BYTES = )(\d+)", rb"\g<1>7")}
    tampered = re.sub(*owned[victim], data, count=1) if victim in owned else data + b"x"
    assert tampered != data
    target.write_bytes(tampered)
    assert fakes.run("--verify") == 1
    assert "NOT IDENTICAL" in capsys.readouterr().out


def test_verify_refuses_a_tree_whose_recorded_runtime_version_is_not_what_transformers_pins(
        bump, consistent, fakes, capsys):
    fakes.built = _built(bump, ort=_NEW_DEV_ORT, runtime=fakes.built.runtime, bundle=fakes.built.bundle)
    assert fakes.run("--verify") == 1
    assert "REFUSED" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
#  Wiring                                                                     #
# --------------------------------------------------------------------------- #

def test_the_weekly_run_advances_the_two_bundle_libraries_through_this_script_and_never_merges_them():
    mod = _load_registered("pin_weekly", _ROOT / "scripts" / "pin_weekly.py")
    advs = {a.key: a for a in mod.build_advancers() if a.bump_script == "scripts/bump_tts_bundle.py"}
    assert sorted(advs) == ["vendored-kokoro-js", "vendored-transformersjs"]
    for adv in advs.values():
        assert adv.confirm_script is None
        assert adv.post_bump_cmds == (("npm", "ci"), ("npm", "test"))
        assert adv.auto_merge("1.2.3", "1.2.4") is False
        assert adv.bump_args[0] == "--lib" and adv.bump_args[1] in ("kokoro-js", "transformers.js")


def test_every_lib_the_weekly_run_passes_is_one_this_script_accepts(bump):
    assert set(bump.LIBS) == {"kokoro-js", "transformers.js"}


def test_the_pin_report_names_the_advancer_of_each_bundle_row():
    cp = _load_registered("check_pins", _ROOT / "scripts" / "check_pins.py")
    assert "rebuilds the TTS bundle" in cp._vendored_advancer("kokoro-js")
    assert "rebuilds the TTS bundle" in cp._vendored_advancer("transformers.js")
    assert "transformers.js" in cp._vendored_advancer("onnxruntime-web")
    for name in ("vendored transformers.js", "vendored onnxruntime-web"):
        assert "by hand" not in cp.ACCEPTED_BLOCKERS[name][1]
        assert "no automation" not in cp.ACCEPTED_BLOCKERS[name][1]


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
