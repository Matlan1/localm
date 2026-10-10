# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/confirm_sdcpp_runtime.py: confirm a stable-diffusion.cpp release works with localm.

Offline. Covers the pieces that decide the verdict:

  * the C header read against localm's ctypes binding: an unchanged header matches, any
    added, dropped, reordered or retyped field is a FAIL that says the binding needs a
    code update;
  * the candidate's pins reaching a freshly started interpreter AND a spawned child (the
    sd.cpp worker is one), and not reaching either without the shim;
  * how a backend child's result becomes download / abi / device / generate checks, with
    SKIP never counted as PASS;
  * the receipt a whole run writes is accepted by scripts/bump_sdcpp_pin.py, and every
    way of not proving the candidate is refused there;
  * the child process plumbing: a real child, a real timeout, a real tree kill.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_CONFIRM = _ROOT / "scripts" / "confirm_sdcpp_runtime.py"
_BUMP = _ROOT / "scripts" / "bump_sdcpp_pin.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CF = _load(_CONFIRM, "confirm_sdcpp_runtime")


@pytest.fixture(scope="module")
def cf():
    return _CF


@pytest.fixture(scope="module")
def bump():
    return _load(_BUMP, "bump_sdcpp_pin")


NEW_SHORT = "bbbbbbb"
NEW_TAG = f"master-999-{NEW_SHORT}"
NEW_COMMIT = NEW_SHORT + "1" * 33


# --------------------------------------------------------------------------- #
#  A C header rendered from the binding, to compare against                    #
# --------------------------------------------------------------------------- #

def _c_type(kind: str, i: int) -> str:
    base = {"bool": "bool", "float": "float", "u32": "uint32_t", "u64": "size_t",
            "i64": "int64_t", "u8": "uint8_t", "str": "const char*", "voidp": "void*"}
    if kind == "int":
        return "int" if i % 2 else "enum some_kind_t"
    if kind in base:
        return base[kind]
    if kind.startswith("ptr:"):
        return _c_type(kind[4:], i) + "*"
    if kind.startswith("struct:"):
        return kind[7:]
    raise AssertionError(kind)


def render_header(structs: dict, *, edit=None) -> str:
    """C text declaring *structs* ({name: [(kind, field)]}), with comments sprinkled in
    the way the upstream header has them. *edit* may change the struct dict first."""
    structs = {k: list(v) for k, v in structs.items()}
    if edit:
        edit(structs)
    out = ["#ifndef X", "// leading comment", "enum some_kind_t { A, B };", ""]
    for name, fields in structs.items():
        out.append("typedef struct {")
        for i, (kind, field) in enumerate(fields):
            tail = "  // trailing note" if i % 3 == 0 else ""
            out.append(f"    {_c_type(kind, i)} {field};{tail}")
            if i % 4 == 0:
                out.append("    /* block comment; with a semicolon */")
        out.append(f"}} {name};  // photo maker")
        out.append("")
    out.append("typedef struct opaque_t opaque_t;")
    return "\n".join(out)


@pytest.fixture(scope="module")
def binding(cf):
    from localm.media.sdcpp import _binding
    return _binding


@pytest.fixture(scope="module")
def bound(cf, binding):
    return cf.binding_structs(binding)


def _opener_for(text: str, seen: list | None = None):
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return text.encode()

    def opener(req, timeout=30):
        if seen is not None:
            seen.append(req.full_url)
        return Resp()
    return opener


def test_the_binding_declares_the_structs_this_check_covers(bound):
    for name in ("sd_ctx_params_t", "sd_img_gen_params_t", "sd_vid_gen_params_t",
                 "sd_sample_params_t", "sd_image_t"):
        assert name in bound and bound[name]


def test_a_header_rendered_from_the_binding_matches_it(cf, bound):
    assert cf.compare_layout(cf.parse_header_structs(render_header(bound)), bound) == []


def test_comments_and_unrelated_declarations_do_not_disturb_the_parse(cf):
    src = textwrap.dedent("""
        typedef struct {
            int a;  // x;y
            /* const char* hidden; */
            const char* b;
            float* c;
            const sd_other_t* d;
            enum e_t e;
        } sd_one_t;
        typedef struct sd_ctx_t sd_ctx_t;
        enum not_a_struct { X, Y };
    """)
    assert cf.parse_header_structs(src) == {"sd_one_t": [
        ("int", "a"), ("str", "b"), ("ptr:float", "c"), ("ptr:struct:sd_other_t", "d"),
        ("int", "e")]}


def test_an_array_member_is_not_silently_accepted(cf, bound):
    parsed = cf.parse_header_structs("typedef struct { float x[4]; } sd_image_t;")
    assert parsed["sd_image_t"][0][0] == "unparsed"
    diffs = cf.compare_layout(parsed, {"sd_image_t": bound["sd_image_t"]})
    assert diffs


def _first_struct(structs):
    return next(n for n, f in structs.items() if len(f) > 6)


@pytest.mark.parametrize("label,edit,expect", [
    ("a field appended", lambda s: s[_first_struct(s)].append(("int", "brand_new")),
     "header adds ['brand_new']"),
    ("a field dropped", lambda s: s[_first_struct(s)].pop(), "drops"),
    ("a field renamed",
     lambda s: s[_first_struct(s)].__setitem__(2, (s[_first_struct(s)][2][0], "renamed")),
     "header adds ['renamed']"),
    ("two fields swapped",
     lambda s: s[_first_struct(s)].insert(0, s[_first_struct(s)].pop(3)), "reordered"),
    ("a field retyped",
     lambda s: s[_first_struct(s)].__setitem__(0, ("u64", s[_first_struct(s)][0][1])),
     "field types differ"),
    ("a struct removed", lambda s: s.pop("sd_image_t"), "sd_image_t: not declared"),
])
def test_any_change_to_a_bound_struct_is_a_difference(cf, bound, label, edit, expect):
    header = cf.parse_header_structs(render_header(bound, edit=edit))
    diffs = cf.compare_layout(header, bound)
    assert diffs and any(expect in d for d in diffs), diffs


def test_a_struct_the_binding_does_not_declare_is_ignored(cf, bound):
    header = cf.parse_header_structs(render_header(
        bound, edit=lambda s: s.update(sd_new_t=[("int", "x")])))
    assert cf.compare_layout(header, bound) == []


def test_header_check_passes_on_a_matching_header_and_reads_the_candidate_commit(
        cf, binding, bound):
    seen = []
    status, detail = cf.header_check(NEW_COMMIT, _opener_for(render_header(bound), seen),
                                     binding)
    assert status == cf.PASS and "field for field" in detail
    assert seen == [cf.HEADER_URL % NEW_COMMIT]


def test_header_check_fails_with_the_binding_needs_a_code_update_reason(cf, binding, bound):
    text = render_header(bound, edit=lambda s: s[_first_struct(s)].append(("int", "extra")))
    status, detail = cf.header_check(NEW_COMMIT, _opener_for(text), binding)
    assert status == cf.FAIL
    assert "binding needs a code update, not an automatic bump" in detail


def test_header_check_is_skip_when_the_header_cannot_be_read(cf, binding):
    def opener(req, timeout=30):
        raise OSError("offline")
    status, detail = cf.header_check(NEW_COMMIT, opener, binding)
    assert status == cf.SKIP and "offline" in detail


def test_header_check_is_skip_when_the_binding_has_no_structs(cf):
    status, _detail = cf.header_check(NEW_COMMIT, _opener_for("x"), SimpleNamespace(
        __name__="empty"))
    assert status == cf.SKIP


def test_the_kind_of_each_ctypes_field_type_agrees_with_the_c_spelling(cf):
    pairs = [(ctypes.c_bool, "bool"), (ctypes.c_int, "int"), (ctypes.c_float, "float"),
             (ctypes.c_uint32, "uint32_t"), (ctypes.c_int64, "int64_t"),
             (ctypes.c_uint64, "uint64_t"), (ctypes.c_size_t, "size_t"),
             (ctypes.c_uint8, "uint8_t"), (ctypes.c_char_p, "const char*"),
             (ctypes.c_void_p, "void*"), (ctypes.POINTER(ctypes.c_float), "float*"),
             (ctypes.POINTER(ctypes.c_uint8), "uint8_t*"),
             (ctypes.POINTER(ctypes.c_int), "int*"), (ctypes.c_int, "enum foo_t")]
    for ct, c in pairs:
        assert cf._py_kind(ct) == cf._c_kind(c), (ct, c)


# --------------------------------------------------------------------------- #
#  The verdict                                                                 #
# --------------------------------------------------------------------------- #

def _checks(**statuses):
    out = {}
    for name, status in statuses.items():
        cf_required = not name.startswith("opt_")
        out[name] = {"status": status, "required": cf_required, "detail": f"{name} is {status}"}
    return out


def test_pass_needs_every_required_check_to_pass(cf):
    assert cf.decide(_checks(a="PASS", b="PASS"))[0] == "PASS"


def test_a_failed_required_check_is_fail_even_beside_a_skip(cf):
    verdict, why = cf.decide(_checks(a="FAIL", b="SKIP"))
    assert verdict == "FAIL" and "a is FAIL" in why


def test_a_required_skip_is_inconclusive_never_pass(cf):
    verdict, why = cf.decide(_checks(a="PASS", b="SKIP"))
    assert verdict == "INCONCLUSIVE" and "b is SKIP" in why


def test_an_optional_check_never_decides_the_verdict(cf):
    assert cf.decide(_checks(a="PASS", opt_x="FAIL", opt_y="SKIP"))[0] == "PASS"


def test_a_mandatory_check_that_never_ran_is_inconclusive(cf):
    verdict, why = cf.decide(_checks(a="PASS"), mandatory=("a", "generate_cpu"))
    assert verdict == "INCONCLUSIVE" and "generate_cpu: never ran" in why


def test_no_checks_at_all_is_not_a_pass(cf, bump):
    assert cf.decide({}, bump.MANDATORY_CHECKS)[0] == "INCONCLUSIVE"


def test_exit_codes(cf):
    assert (cf.exit_code("PASS"), cf.exit_code("FAIL"), cf.exit_code("INCONCLUSIVE")) == (0, 1, 2)


def test_the_receipt_is_written_atomically_and_replaces_an_older_one(cf, tmp_path):
    path = tmp_path / "sub" / "r.json"
    cf.write_receipt(path, {"a": 1})
    cf.write_receipt(path, {"a": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 2}
    assert [p.name for p in path.parent.iterdir()] == ["r.json"]


# --------------------------------------------------------------------------- #
#  Backend choice                                                              #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("available,gpu,rec,expect", [
    (["cpu", "vulkan", "cuda", "rocm"], "found", "rocm", ["cpu", "vulkan", "rocm"]),
    (["cpu", "vulkan", "cuda", "rocm"], "found", "vulkan", ["cpu", "vulkan"]),
    (["cpu", "vulkan", "cuda", "rocm"], "none", "cpu", ["cpu"]),
    (["cpu", "vulkan", "cuda", "rocm"], "unknown", "vulkan", ["cpu"]),
    (["cpu", "vulkan", "rocm"], "found", "cuda", ["cpu", "vulkan"]),
    (["metal"], "found", "metal", ["metal"]),
    ([], "found", "cpu", []),
])
def test_choose_backends(cf, available, gpu, rec, expect):
    assert cf.choose_backends(available, gpu, rec) == expect


# --------------------------------------------------------------------------- #
#  The image judgement                                                         #
# --------------------------------------------------------------------------- #

def _png(tmp_path, name, make):
    from PIL import Image
    im = Image.new("RGB", (64, 64))
    im.putdata([make(x, y) for y in range(64) for x in range(64)])
    p = tmp_path / name
    im.save(p)
    return p


def test_a_flat_image_is_rejected(cf, tmp_path):
    stats = cf.image_stats(_png(tmp_path, "flat.png", lambda x, y: (120, 130, 140)))
    assert cf.judge_image(stats, 64)[0] is False


def test_an_image_of_the_wrong_size_is_rejected(cf, tmp_path):
    stats = cf.image_stats(_png(tmp_path, "g.png", lambda x, y: (x * 4, y * 4, (x + y) * 2)))
    ok, why = cf.judge_image(stats, 128)
    assert not ok and "expected 128x128" in why


def test_a_saturated_image_is_rejected(cf, tmp_path):
    stats = cf.image_stats(_png(tmp_path, "s.png", lambda x, y: (
        (255, y * 4, x * 4) if (x + y) % 5 else (10 + x, 10 + y, 50))))
    ok, why = cf.judge_image(stats, 64)
    assert not ok and "saturated" in why


def test_an_image_with_structure_is_accepted(cf, tmp_path):
    stats = cf.image_stats(_png(tmp_path, "g.png",
                                lambda x, y: (20 + x * 3, 30 + y * 3, 40 + (x * y) % 150)))
    ok, why = cf.judge_image(stats, 64)
    assert ok, why


def test_the_difference_of_identical_images_is_zero_and_of_opposite_images_is_large(
        cf, tmp_path):
    a = _png(tmp_path, "a.png", lambda x, y: (x * 4, y * 4, 50))
    b = _png(tmp_path, "b.png", lambda x, y: (255 - x * 4, 255 - y * 4, 205))
    assert cf.image_difference(a, a) == 0
    assert cf.image_difference(a, b) > 0.3
    status, detail = cf.agreement_check({"cpu": str(a), "vulkan": str(a)})
    assert status == "PASS" and "cpu vs vulkan" in detail
    assert cf.agreement_check({"cpu": str(a), "vulkan": str(b)})[0] == "FAIL"
    assert cf.agreement_check({"cpu": str(a)})[0] == "SKIP"


# --------------------------------------------------------------------------- #
#  Child results -> checks                                                     #
# --------------------------------------------------------------------------- #

SIZES = {"a.zip": 1000}


def _res(**over):
    base = {"backend": "cpu", "fatal": None,
            "install": {"error": None, "kind": None,
                        "downloads": [{"name": "a.zip", "size": 1000}]},
            "probe": {"ok": True, "commit": "bbbbbbb", "devices": [["CPU", "x86"]]},
            "resolved": True, "has_backend_device": True,
            "generate": {"status": "ok", "detail": "256x256, stddev 40", "seconds": 3.0,
                         "model_version": "SD Turbo", "stats": {"stddev": 40}}}
    base.update(over)
    return base


def _statuses(cf, backend="cpu", **over):
    checks = cf.checks_for_backend(backend, _res(**over), NEW_COMMIT, SIZES)
    return {n: c["status"] for n, c in checks.items()}, checks


def test_a_healthy_backend_passes_all_four_checks(cf):
    st, checks = _statuses(cf)
    assert st == {"download_cpu": "PASS", "abi_cpu": "PASS", "device_cpu": "PASS",
                  "generate_cpu": "PASS"}
    assert all(c["required"] for c in checks.values())


def test_a_child_that_could_not_run_skips_everything(cf):
    st, checks = _statuses(cf, fatal="isolation: home_dir() is elsewhere")
    assert set(st.values()) == {"SKIP"}
    assert "isolation" in checks["download_cpu"]["detail"]


@pytest.mark.parametrize("kind,expect", [("network", "SKIP"), ("sha", "FAIL"),
                                         ("archive", "FAIL")])
def test_download_failures_split_into_could_not_measure_and_bad_build(cf, kind, expect):
    st, _ = _statuses(cf, install={"error": "boom", "kind": kind, "downloads": []},
                      probe=None)
    assert st["download_cpu"] == expect
    assert st["abi_cpu"] == st["device_cpu"] == st["generate_cpu"] == "SKIP"


def test_a_downloaded_size_that_differs_from_the_listing_fails(cf):
    st, checks = _statuses(cf, install={"error": None, "kind": None,
                                        "downloads": [{"name": "a.zip", "size": 999}]})
    assert st["download_cpu"] == "FAIL" and "999 bytes, release lists 1000" in \
        checks["download_cpu"]["detail"]


def test_an_install_with_no_recorded_download_is_not_a_pass(cf):
    st, _ = _statuses(cf, install={"error": None, "kind": None, "downloads": []})
    assert st["download_cpu"] == "SKIP"


def test_a_layout_mismatch_names_the_binding_as_the_thing_to_update(cf):
    msg = ("stable-diffusion.cpp struct layout does not match localm's binding "
           "(ctx.n_threads=0). Reinstall it")
    st, checks = _statuses(cf, probe={"ok": False, "error": msg}, has_backend_device=False,
                           resolved=False)
    assert st["abi_cpu"] == "FAIL"
    assert "binding needs a code update, not an automatic bump" in checks["abi_cpu"]["detail"]
    assert st["device_cpu"] == st["generate_cpu"] == "SKIP"


def test_a_worker_that_bound_another_commit_means_the_override_did_not_apply(cf):
    msg = ("stable-diffusion.cpp runtime is commit bbbbbbb, but localm binds commit "
           "f89d9b1 (master-951-f89d9b1).")
    st, checks = _statuses(cf, probe={"ok": False, "error": msg})
    assert st["abi_cpu"] == "SKIP" and "not applied" in checks["abi_cpu"]["detail"]


def test_a_release_archive_that_is_not_the_commit_of_its_tag_fails(cf):
    msg = ("stable-diffusion.cpp runtime is commit ccccccc, but localm binds commit "
           f"{NEW_SHORT} ({NEW_TAG}).")
    st, checks = _statuses(cf, probe={"ok": False, "error": msg})
    assert st["abi_cpu"] == "FAIL" and "not the commit its tag names" in \
        checks["abi_cpu"]["detail"]


@pytest.mark.parametrize("msg,expect", [
    ("could not load the stable-diffusion.cpp runtime from X (WinError 126)", "FAIL"),
    ("The media worker process crashed (exit code 3) during 'probe'.", "FAIL"),
    ("The media worker 'probe' timed out after 120s and was stopped.", "SKIP"),
])
def test_other_probe_failures(cf, msg, expect):
    st, _ = _statuses(cf, probe={"ok": False, "error": msg})
    assert st["abi_cpu"] == expect


def test_a_library_that_reports_another_commit_than_the_candidate_fails(cf):
    st, _ = _statuses(cf, probe={"ok": True, "commit": "ccccccc", "devices": [["CPU", "x"]]})
    assert st["abi_cpu"] == "FAIL"


def test_a_library_that_reports_no_commit_fails(cf):
    st, _ = _statuses(cf, probe={"ok": True, "commit": "", "devices": [["CPU", "x"]]})
    assert st["abi_cpu"] == "FAIL"


def test_a_gpu_backend_with_only_the_cpu_device_fails_the_device_check(cf):
    st, checks = _statuses(cf, backend="vulkan", has_backend_device=False,
                           probe={"ok": True, "commit": "bbbbbbb", "devices": [["CPU", "x"]]})
    assert st["device_vulkan"] == "FAIL" and st["generate_vulkan"] == "SKIP"
    assert "no vulkan compute device" in checks["device_vulkan"]["detail"]


def test_a_runtime_the_product_resolver_cannot_find_fails_the_device_check(cf):
    st, _ = _statuses(cf, resolved=False)
    assert st["device_cpu"] == "FAIL"


def test_generation_that_was_not_run_says_it_was_not_measured(cf):
    st, checks = _statuses(cf, generate={"status": "skip", "detail": "the model is not "
                                                                      "available: offline"})
    assert st["generate_cpu"] == "SKIP"
    assert "NOT measured" in checks["generate_cpu"]["detail"]


def test_generation_that_failed_is_a_fail(cf):
    st, checks = _statuses(cf, generate={"status": "fail", "detail": "image is nearly flat"})
    assert st["generate_cpu"] == "FAIL" and "nearly flat" in checks["generate_cpu"]["detail"]


# --------------------------------------------------------------------------- #
#  The pins shim: the candidate reaches the driver and the spawned worker      #
# --------------------------------------------------------------------------- #

_SHIM_PROBE = textwrap.dedent('''
    import json, multiprocessing as mp

    def _view():
        from localm.media.sdcpp import pins
        return [pins.TAG, pins.COMMIT, pins.asset_url("x.zip"),
                sorted("|".join(k) for k in pins.ASSETS),
                sorted(pins.EXTRA_ASSETS and ["|".join(k) for k in pins.EXTRA_ASSETS]),
                getattr(pins, "LOCALM_CONFIRM_OVERRIDE", False)]

    def _child(q):
        q.put(_view())

    if __name__ == "__main__":
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_child, args=(q,))
        p.start()
        child = q.get(timeout=120)
        p.join()
        print("@@" + json.dumps({"parent": _view(), "child": child}))
''')


def _run_shim_probe(cf, tmp_path, with_shim: bool, with_env: bool):
    shim = tmp_path / "shim"
    shim.mkdir(exist_ok=True)
    (shim / "sitecustomize.py").write_text(cf.SHIM_SOURCE, encoding="utf-8")
    payload = cf.pins_payload(
        NEW_TAG, NEW_COMMIT,
        {("windows", "cpu"): ("sd-new-cpu.zip", "a" * 64)},
        {("windows", "cuda"): [("rt-1.zip", "b" * 64), ("rt-2.zip", "c" * 64)]})
    pins_json = tmp_path / "pins.json"
    pins_json.write_text(json.dumps(payload), encoding="utf-8")
    script = tmp_path / "probe.py"
    script.write_text(_SHIM_PROBE, encoding="utf-8")
    env = dict(os.environ)
    parts = ([str(shim)] if with_shim else []) + [str(_ROOT)]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env.pop(cf.PINS_ENV, None)
    if with_env:
        env[cf.PINS_ENV] = str(pins_json)
    r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True,
                       timeout=240, cwd=str(tmp_path))
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("@@")), None)
    assert line, f"no output: {r.stdout!r} {r.stderr[-800:]!r}"
    return json.loads(line[2:])


def test_the_candidate_pins_reach_a_fresh_interpreter_and_a_spawned_child(cf, tmp_path):
    out = _run_shim_probe(cf, tmp_path, with_shim=True, with_env=True)
    for side in ("parent", "child"):
        tag, commit, url, assets, extra, flag = out[side]
        assert (tag, commit, flag) == (NEW_TAG, NEW_COMMIT, True), side
        assert url.endswith(f"/releases/download/{NEW_TAG}/x.zip"), side
        assert assets == ["windows|cpu"] and extra == ["windows|cuda"], side


def test_without_the_environment_variable_the_shim_changes_nothing(cf, tmp_path):
    from localm.media.sdcpp import pins
    out = _run_shim_probe(cf, tmp_path, with_shim=True, with_env=False)
    for side in ("parent", "child"):
        assert out[side][0] == pins.TAG and out[side][5] is False, side


def test_without_the_shim_directory_the_variable_changes_nothing(cf, tmp_path):
    from localm.media.sdcpp import pins
    out = _run_shim_probe(cf, tmp_path, with_shim=False, with_env=True)
    for side in ("parent", "child"):
        assert out[side][0] == pins.TAG and out[side][5] is False, side


def test_both_extra_archives_of_one_key_survive_the_payload(cf):
    payload = cf.pins_payload("t", "c", {}, {("windows", "cuda"): [("a", "1"), ("b", "2")]})
    assert payload["extra"] == [["windows", "cuda", "a", "1"], ["windows", "cuda", "b", "2"]]


# --------------------------------------------------------------------------- #
#  Real child processes                                                        #
# --------------------------------------------------------------------------- #

def _alive_with(marker: str) -> list:
    import psutil
    hits = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            if any(marker in part for part in p.info["cmdline"] or []):
                hits.append(p.info["pid"])
        except psutil.Error:
            continue
    return hits


def test_a_child_that_cannot_isolate_reports_why_in_its_result(cf, tmp_path):
    env = cf.prepare_environment(tmp_path)
    spec = {"backend": "cpu", "tag": "master-1-aaaaaaa", "commit": "a" * 40,
            "override": False, "home": env["LOCALM_HOME"], "model": None,
            "model_skip": "none", "png": str(tmp_path / "o.png")}
    res = cf.run_child(spec, tmp_path, env, timeout=300)
    assert res["fatal"] and res["fatal"].startswith("isolation:"), res
    assert "pins say" in res["fatal"]
    st = {n: c["status"] for n, c in cf.checks_for_backend("cpu", res, "a" * 40, {}).items()}
    assert set(st.values()) == {"SKIP"}


def test_a_child_past_its_timeout_is_killed_with_its_tree(cf, tmp_path):
    env = cf.prepare_environment(tmp_path)
    spec = {"backend": "cpu", "tag": "t", "commit": "c" * 40, "override": False,
            "home": env["LOCALM_HOME"], "model": None, "model_skip": "", "png": "x"}
    start = time.monotonic()
    res = cf.run_child(spec, tmp_path, env, timeout=0.05)
    assert res.get("timeout") and "timed out" in res["fatal"]
    assert time.monotonic() - start < 120
    time.sleep(0.5)
    assert _alive_with(str(tmp_path)) == [], "the timed out child survived"


def test_kill_tree_ends_a_process_and_its_descendants(cf, tmp_path):
    marker = tmp_path / "marker-for-kill-tree"
    script = tmp_path / "tree.py"
    script.write_text(textwrap.dedent(f"""
        import subprocess, sys, time
        sub = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", {str(marker)!r}])
        time.sleep(600)
    """), encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script)])
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and len(_alive_with(str(marker))) < 1:
            time.sleep(0.1)
        assert _alive_with(str(marker)), "the grandchild never started"
        cf.kill_tree(proc.pid)
        proc.wait(timeout=30)
        time.sleep(0.5)
        assert _alive_with(str(marker)) == []
    finally:
        if proc.poll() is None:
            proc.kill()


def test_the_environment_for_children_stays_inside_the_run_directory(cf, tmp_path):
    env = cf.prepare_environment(tmp_path, {"PYTHONPATH": "keep-me", "PATH": os.environ["PATH"]})
    for key in ("LOCALM_HOME", "TEMP", "TMP", "TMPDIR", "HF_HOME"):
        assert Path(env[key]).resolve().parent == tmp_path.resolve(), key
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert Path(parts[0]) == tmp_path / "shim" and Path(parts[1]) == _ROOT
    assert parts[2] == "keep-me"
    assert (tmp_path / "shim" / "sitecustomize.py").read_text(encoding="utf-8") == cf.SHIM_SOURCE


# --------------------------------------------------------------------------- #
#  A whole run, with the network and the per-backend child replaced            #
# --------------------------------------------------------------------------- #

_NOGPU = SimpleNamespace(gpu_state="none", vendors=[], gpu_names="")


class _Api:
    """Serves the release, commit and header reads for NEW_TAG."""

    def __init__(self, bump, binding_structs_, *, header_edit=None, tamper=None):
        from localm.media.sdcpp import runtime
        self.bump = bump
        self.header_url = _CF.HEADER_URL
        self.plat = runtime.platform_key() or "windows"
        pins = bump.read_pins((_ROOT / "localm/media/sdcpp/pins.py").read_text(encoding="utf-8"))
        names = {}
        sha = lambda n: hashlib.sha256((n + NEW_TAG).encode()).hexdigest()  # noqa: E731
        for old, _s in pins["assets"].values():
            names[old] = old.replace(pins["commit"][:7], NEW_SHORT)
        for entries in pins["extra"].values():
            for old, _s in entries:
                names[old] = old
        self.names = names
        listing = [{"name": n, "size": 5000 + i, "digest": "sha256:" + sha(n)}
                   for i, n in enumerate(sorted(set(names.values())))]
        if tamper:
            tamper(listing)
        self.release = {"tag_name": NEW_TAG, "target_commitish": NEW_COMMIT, "assets": listing}
        self.header = render_header(binding_structs_, edit=header_edit)
        self.pins = pins

    def cpu_asset(self):
        old = self.pins["assets"][(self.plat, "cpu")][0]
        new = self.names[old]
        return new, next(a["size"] for a in self.release["assets"] if a["name"] == new)

    def opener(self):
        bump = self.bump

        class Resp:
            def __init__(self, body):
                self.b = body if isinstance(body, bytes) else json.dumps(body).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return self.b

        routes = {bump.RELEASE_URL % (bump.UPSTREAM_REPO, NEW_TAG): self.release,
                  bump.COMMIT_URL % (bump.UPSTREAM_REPO, NEW_TAG): {"sha": NEW_COMMIT},
                  self.header_url % NEW_COMMIT: self.header.encode()}

        def opener(req, timeout=30):
            return Resp(routes[req.full_url])
        return opener


def _healthy_runner(api, gen=None, calls=None):
    name, size = api.cpu_asset()

    def runner(spec, run_dir, env):
        if calls is not None:
            calls.append((spec, dict(env)))
        return {"backend": spec["backend"], "fatal": None,
                "install": {"error": None, "kind": None,
                            "downloads": [{"name": name, "size": size}]},
                "probe": {"ok": True, "commit": NEW_SHORT, "devices": [["CPU", "x86"]]},
                "resolved": True, "has_backend_device": True,
                "generate": gen or {"status": "ok", "detail": "256x256, stddev 40",
                                    "seconds": 2.0, "model_version": "SD Turbo",
                                    "stats": {"stddev": 40}}}
    return runner


_MODEL = {"path": "model.gguf", "name": "sd-turbo", "repo": "r", "file": "f", "size": 1,
          "sha256": "0" * 64, "downloaded": False}


def _confirm(cf, api, tmp_path, **kw):
    kw.setdefault("child_runner", _healthy_runner(api))
    kw.setdefault("model_provider", lambda d: dict(_MODEL))
    return cf.confirm(NEW_TAG, False, tmp_path, backends=["cpu"], opener=api.opener(),
                      detector=lambda: _NOGPU, **kw)


def test_a_healthy_candidate_gives_a_receipt_the_bump_script_accepts(cf, bump, bound, tmp_path):
    api = _Api(bump, bound)
    calls = []
    rec = _confirm(cf, api, tmp_path, child_runner=_healthy_runner(api, calls=calls))
    assert rec["verdict"] == "PASS", rec["why"]
    assert (rec["schema"], rec["component"], rec["tag"], rec["current"]) == (
        1, "sdcpp", NEW_TAG, False)
    assert rec["hardware"]["backends"] == ["cpu"] and rec["hardware"]["gpu"] is False
    assert rec["model"]["name"] == "sd-turbo" and "path" not in rec["model"]
    assert "video generation" in rec["not_measured"]
    assert set(bump.MANDATORY_CHECKS) <= set(rec["checks"])
    assert rec["candidate"]["commit"] == NEW_COMMIT
    path = tmp_path / "r.json"
    cf.write_receipt(path, rec)
    loaded = bump.load_receipt(path, NEW_TAG)
    bump.compare_with_receipt(loaded, {
        "tag": NEW_TAG, "commit": NEW_COMMIT,
        "assets": {a["name"]: {"size": a["size"], "sha256": a["digest"].split(":")[1]}
                   for a in api.release["assets"]}})
    spec, env = calls[0]
    assert spec["override"] is True and cf.PINS_ENV in env
    assert Path(env["LOCALM_HOME"]).parent.name.startswith("run-")


def test_the_run_directory_is_removed_and_the_process_environment_restored(cf, bump, bound,
                                                                           tmp_path):
    api = _Api(bump, bound)
    before = {k: os.environ.get(k) for k in ("LOCALM_HOME", "TEMP", "TMP", "TMPDIR", "HF_HOME")}
    _confirm(cf, api, tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert {k: os.environ.get(k) for k in before} == before


def test_keep_leaves_the_run_directory(cf, bump, bound, tmp_path):
    api = _Api(bump, bound)
    _confirm(cf, api, tmp_path, keep=True)
    assert [p.name for p in tmp_path.iterdir()][0].startswith("run-")


def test_a_changed_struct_makes_the_whole_run_fail_with_the_reason(cf, bump, bound, tmp_path):
    api = _Api(bump, bound, header_edit=lambda s: s[_first_struct(s)].append(("int", "new")))
    rec = _confirm(cf, api, tmp_path)
    assert rec["verdict"] == "FAIL"
    assert "binding needs a code update, not an automatic bump" in rec["why"]
    cf.write_receipt(tmp_path / "r.json", rec)
    with pytest.raises(bump.Refused, match="not PASS|header_layout"):
        bump.load_receipt(tmp_path / "r.json", NEW_TAG)


def test_generation_that_was_not_measured_makes_the_run_inconclusive_and_unbumpable(
        cf, bump, bound, tmp_path):
    api = _Api(bump, bound)

    def no_model(d):
        raise cf.Inconclusive("offline")
    skipped = {"status": "skip", "detail": "the model is not available: offline"}
    rec = _confirm(cf, api, tmp_path, model_provider=no_model,
                   child_runner=_healthy_runner(api, gen=skipped))
    assert rec["verdict"] == "INCONCLUSIVE" and "generate_cpu" in rec["why"]
    assert rec["checks"]["generate_cpu"]["status"] == "SKIP"
    assert rec["model"] is None
    cf.write_receipt(tmp_path / "r.json", rec)
    with pytest.raises(bump.Refused):
        bump.load_receipt(tmp_path / "r.json", NEW_TAG)


def test_an_unreadable_api_is_inconclusive_not_fail(cf, bump, bound, tmp_path):
    def down(req, timeout=30):
        raise OSError("network is down")
    rec = cf.confirm(NEW_TAG, False, tmp_path, backends=["cpu"], opener=down,
                     detector=lambda: _NOGPU, child_runner=lambda *a: pytest.fail("ran"),
                     model_provider=lambda d: dict(_MODEL))
    assert rec["verdict"] == "INCONCLUSIVE"
    assert rec["checks"]["release_assets"]["status"] == "SKIP"


def test_a_release_missing_an_archive_is_inconclusive(cf, bump, bound, tmp_path):
    api = _Api(bump, bound, tamper=lambda listing: listing.pop(0))
    rec = _confirm(cf, api, tmp_path)
    assert rec["verdict"] == "INCONCLUSIVE"
    assert rec["checks"]["release_assets"]["status"] == "SKIP"
    assert "no archive for" in rec["checks"]["release_assets"]["detail"]


def test_a_release_without_a_digest_is_inconclusive(cf, bump, bound, tmp_path):
    api = _Api(bump, bound, tamper=lambda listing: listing[0].pop("digest"))
    rec = _confirm(cf, api, tmp_path)
    assert rec["verdict"] == "INCONCLUSIVE"
    assert rec["checks"]["release_assets"]["status"] == "SKIP"
    assert "no sha256 digest" in rec["checks"]["release_assets"]["detail"]


def test_a_child_failure_is_a_failure_of_the_candidate(cf, bump, bound, tmp_path):
    api = _Api(bump, bound)
    name, size = api.cpu_asset()

    def runner(spec, run_dir, env):
        return {"backend": "cpu", "fatal": None,
                "install": {"error": None, "kind": None,
                            "downloads": [{"name": name, "size": size}]},
                "probe": {"ok": False, "error": "struct layout does not match (x)"},
                "resolved": False, "has_backend_device": False,
                "generate": {"status": "skip", "detail": ""}}
    rec = _confirm(cf, api, tmp_path, child_runner=runner)
    assert rec["verdict"] == "FAIL" and "abi_cpu" in rec["why"]


def test_an_exception_inside_the_run_is_inconclusive_and_the_receipt_says_so(
        cf, bump, bound, tmp_path):
    api = _Api(bump, bound)

    def boom(spec, run_dir, env):
        raise RuntimeError("kaboom")
    rec = _confirm(cf, api, tmp_path, child_runner=boom)
    assert rec["verdict"] == "INCONCLUSIVE"
    assert "RuntimeError: kaboom" in rec["why"] and "traceback" in rec


def test_two_backends_add_an_agreement_check_that_never_decides_the_verdict(
        cf, bump, bound, tmp_path):
    api = _Api(bump, bound)
    name, size = api.cpu_asset()
    a = _png(tmp_path, "a.png", lambda x, y: (x * 4, y * 4, 50))
    b = _png(tmp_path, "b.png", lambda x, y: (255 - x * 4, 255 - y * 4, 205))
    pngs = {"cpu": a, "vulkan": b}

    def runner(spec, run_dir, env):
        res = _healthy_runner(api)(spec, run_dir, env)
        res["probe"]["devices"] = [["CPU", "x"]] if spec["backend"] == "cpu" else \
            [["Vulkan0", "gpu"]]
        # the planted images stand in for what the children wrote
        Path(spec["png"]).write_bytes(pngs[spec["backend"]].read_bytes())
        return res

    class _Gpu:
        gpu_state = "found"
        vendors = ["amd"]
        gpu_names = "gpu"

    rec = cf.confirm(NEW_TAG, False, tmp_path / "w", backends=["cpu", "vulkan"],
                     opener=api.opener(), detector=lambda: _Gpu(), child_runner=runner,
                     model_provider=lambda d: dict(_MODEL))
    agree = rec["checks"]["generate_agree"]
    assert agree["required"] is False and agree["status"] == "FAIL"
    assert rec["verdict"] == "PASS", rec["why"]
    assert rec["hardware"]["gpu"] is True and rec["hardware"]["backends"] == ["cpu", "vulkan"]


# --------------------------------------------------------------------------- #
#  --current                                                                   #
# --------------------------------------------------------------------------- #

def _current_api(bump, drift=None):
    pins = bump.read_pins((_ROOT / "localm/media/sdcpp/pins.py").read_text(encoding="utf-8"))
    listing = []
    for n, s in pins["assets"].values():
        listing.append({"name": n, "size": 4000, "digest": "sha256:" + s})
    for entries in pins["extra"].values():
        for n, s in entries:
            listing.append({"name": n, "size": 4000, "digest": "sha256:" + s})
    if drift:
        drift(listing)
    routes = {bump.RELEASE_URL % (bump.UPSTREAM_REPO, pins["tag"]):
              {"tag_name": pins["tag"], "target_commitish": pins["commit"], "assets": listing},
              bump.COMMIT_URL % (bump.UPSTREAM_REPO, pins["tag"]): {"sha": pins["commit"]}}

    class Resp:
        def __init__(self, body):
            self.b = json.dumps(body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.b

    def opener(req, timeout=30):
        return Resp(routes[req.full_url])
    return pins, opener


def test_current_confirms_the_pinned_tables_against_the_release_listing(cf, bump):
    pins, opener = _current_api(bump)
    cand = cf.resolve_candidate(None, True, opener)
    assert cand["tag"] == pins["tag"] and cand["commit"] == pins["commit"]
    assert cand["assets"] == pins["assets"]


def test_current_notices_a_release_that_was_re_uploaded(cf, bump):
    _pins, opener = _current_api(
        bump, drift=lambda lst: lst[0].update(digest="sha256:" + "9" * 64))
    with pytest.raises(ValueError, match="disagrees with the release"):
        cf.resolve_candidate(None, True, opener)


def test_current_notices_an_asset_that_left_the_release(cf, bump):
    _pins, opener = _current_api(bump, drift=lambda lst: lst.pop(0))
    with pytest.raises(ValueError, match="is not in the release"):
        cf.resolve_candidate(None, True, opener)


def test_current_without_the_api_is_inconclusive(cf, bump):
    def down(req, timeout=30):
        raise OSError("offline")
    with pytest.raises(cf.Inconclusive, match="offline"):
        cf.resolve_candidate(None, True, down)


def test_main_rejects_a_tag_that_is_not_a_release_tag_without_running_anything(
        cf, tmp_path, capsys):
    rc = cf.main(["--tag", "v1.2.3", "--workdir", str(tmp_path), "--receipt",
                  str(tmp_path / "r.json")])
    assert rc == 2 and "not a stable-diffusion.cpp release tag" in capsys.readouterr().out
    assert not (tmp_path / "r.json").exists()
