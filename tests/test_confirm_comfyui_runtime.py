# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/confirm_comfyui_runtime.py: the pure-logic pieces, tested against
REAL inputs wherever one is cheap to build (a real PNG, a real git repo, a
real bound socket) - never a mock standing in for something this cheap to
make real, per this repo's own testing discipline.

NOT covered here (needs a real ComfyUI server or real hardware, per the
plan's own "live verification" checklist): the full provision/smoke phases
end to end, the calibration run against the current pin, and the GPU
roundtrip's real kernel execution. Those are proven by a real run, not a
unit test standing in for one.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import struct
import subprocess
import zlib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_CONFIRM = _ROOT / "scripts" / "confirm_comfyui_runtime.py"
_BUMP = _ROOT / "scripts" / "bump_comfyui_pin.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def confirm():
    return _load(_CONFIRM, "confirm_comfyui_runtime")


# --------------------------------------------------------------------------- #
#  The two scripts' check-name contracts must never drift apart              #
# --------------------------------------------------------------------------- #

def test_check_names_match_bump_comfyui_pins_default_require(confirm):
    bump = _load(_BUMP, "bump_comfyui_pin_for_contract_check")
    assert confirm.CHECK_NAMES == bump.CHECK_NAMES == bump.DEFAULT_REQUIRE


# --------------------------------------------------------------------------- #
#  Receipt plumbing                                                          #
# --------------------------------------------------------------------------- #

def test_new_receipt_shape(confirm):
    r = confirm._new_receipt("v0.32.0", "a" * 40)
    assert r["schema"] == 1
    assert r["tag"] == "v0.32.0" and r["commit"] == "a" * 40
    assert r["checks"] == {}
    assert r["not_covered"] == confirm.NOT_COVERED
    assert r["not_covered"] is not confirm.NOT_COVERED, "must be a copy, not aliased"


def test_set_check_and_check_passed(confirm):
    r = confirm._new_receipt("v0.32.0", "a" * 40)
    confirm._set_check(r, "isolation", confirm.PASS, "ok", extra_field=1)
    assert confirm._check_passed(r, "isolation") is True
    assert r["checks"]["isolation"] == {"verdict": "PASS", "why": "ok", "extra_field": 1}
    assert confirm._check_passed(r, "provision") is False, "an unset check never reads as passed"


def test_save_and_load_receipt_round_trips_and_rejects_a_different_candidate(confirm, tmp_path):
    path = tmp_path / "receipt.json"
    r = confirm._new_receipt("v0.32.0", "a" * 40)
    confirm._set_check(r, "provision", confirm.PASS, "ok")
    confirm._save_receipt(path, r)
    assert "written_at" in json.loads(path.read_text(encoding="utf-8"))

    loaded = confirm._load_receipt(path, "v0.32.0", "a" * 40)
    assert loaded is not None and confirm._check_passed(loaded, "provision")

    assert confirm._load_receipt(path, "v0.32.0", "b" * 40) is None, "wrong commit"
    assert confirm._load_receipt(path, "v0.30.0", "a" * 40) is None, "wrong tag"
    assert confirm._load_receipt(tmp_path / "absent.json", "v0.32.0", "a" * 40) is None
    (tmp_path / "garbage.json").write_text("not json", encoding="utf-8")
    assert confirm._load_receipt(tmp_path / "garbage.json", "v0.32.0", "a" * 40) is None


# --------------------------------------------------------------------------- #
#  Identity verification (H1's real backstop)                                #
# --------------------------------------------------------------------------- #

def test_verify_identity_accepts_argv0_inside_root(confirm, tmp_path):
    root = tmp_path / "comfyui"
    root.mkdir()
    main_py = root / "main.py"
    main_py.write_text("", encoding="utf-8")
    stats = {"system": {"argv": [str(main_py)]}}
    ok, why = confirm._verify_identity(stats, root)
    assert ok is True and str(root) in why


def test_verify_identity_refuses_argv0_outside_root(confirm, tmp_path):
    """The exact hazard this exists for: a DIFFERENT ComfyUI (e.g. one
    already running on this box under a different install path) answers
    /system_stats, and its argv[0] does not resolve inside our scratch
    root."""
    root = tmp_path / "our-scratch" / "comfyui"
    root.mkdir(parents=True)
    other = tmp_path / "someone-elses-comfyui" / "main.py"
    other.parent.mkdir(parents=True)
    other.write_text("", encoding="utf-8")
    stats = {"system": {"argv": [str(other)]}}
    ok, why = confirm._verify_identity(stats, root)
    assert ok is False
    assert "DIFFERENT" in why


def test_verify_identity_refuses_missing_or_malformed_stats(confirm, tmp_path):
    root = tmp_path / "comfyui"
    root.mkdir()
    assert confirm._verify_identity(None, root) == (
        False, "/system_stats did not answer or returned unreadable JSON")
    ok, why = confirm._verify_identity({"system": {}}, root)
    assert ok is False


# --------------------------------------------------------------------------- #
#  torch_device: does the installed torch actually match the spec           #
# --------------------------------------------------------------------------- #

class _Spec:
    def __init__(self, variant, packages=()):
        self.variant = variant
        self.packages = packages


def test_torch_variant_ok_matches_by_suffix(confirm):
    assert confirm._torch_variant_ok("2.9.1+cu124", _Spec("cuda")) is True
    assert confirm._torch_variant_ok("2.9.1+cpu", _Spec("cuda")) is False, (
        "a cuda spec satisfied by a CPU build is exactly the silent pip swap this guards")
    assert confirm._torch_variant_ok("2.9.1+rocm7.13.0", _Spec("rocm")) is True
    assert confirm._torch_variant_ok(None, _Spec("cuda")) is False


def test_torch_variant_ok_matches_an_exact_pin(confirm):
    spec = _Spec("rocm", ("torch==2.9.1+rocm7.13.0",))
    assert confirm._torch_variant_ok("2.9.1+rocm7.13.0", spec) is True
    assert confirm._torch_variant_ok("2.9.1+rocm7.14.0", spec) is False


# --------------------------------------------------------------------------- #
#  The GPU-roundtrip output verifier: a REAL PNG, not a mock                 #
# --------------------------------------------------------------------------- #

def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _filter_row(cur: bytes, prev: bytes, ftype: int, bpp: int) -> bytes:
    """The real PNG spec's per-row FORWARD filter (encoder side) - the exact
    inverse of the decoder's reconstruction, so a round-trip through this and
    _unfilter_png_rows proves the decoder handles what a real encoder emits,
    not just a filter this test author decoded correctly by luck."""
    out = bytearray(len(cur))
    for i in range(len(cur)):
        a = cur[i - bpp] if i >= bpp else 0
        b = prev[i] if prev else 0
        c = prev[i - bpp] if (prev and i >= bpp) else 0
        if ftype == 0:
            out[i] = cur[i]
        elif ftype == 1:
            out[i] = (cur[i] - a) & 0xFF
        elif ftype == 2:
            out[i] = (cur[i] - b) & 0xFF
        elif ftype == 3:
            out[i] = (cur[i] - ((a + b) // 2)) & 0xFF
        elif ftype == 4:
            out[i] = (cur[i] - _paeth(a, b, c)) & 0xFF
        else:
            raise ValueError(ftype)
    return bytes(out)


def _make_solid_png(path: Path, rgb: tuple, size: int = 64, *, filter_type: int = 0,
                    per_row_filters: "list[int] | None" = None) -> None:
    """A real truecolor PNG of a solid color, filtered with *filter_type* (or
    a distinct filter PER ROW, matching how a real adaptive encoder like
    ComfyUI's own PIL-based writer behaves) - exactly the shape
    _verify_probe_output must be able to decode without Pillow."""
    width = height = size
    channels = 3
    true_row = bytes(rgb) * width
    filters = per_row_filters or [filter_type] * height
    assert len(filters) == height
    raw = bytearray()
    prev = None
    for ftype in filters:
        filtered = _filter_row(true_row, prev, ftype, channels)
        raw.append(ftype)
        raw += filtered
        prev = true_row  # the RECONSTRUCTED row, i.e. the true pixels here
    idat = zlib.compress(bytes(raw), 9)

    def chunk(ctype: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + ctype + data
               + struct.pack(">I", zlib.crc32(ctype + data)))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat)
          + chunk(b"IEND", b""))
    path.write_bytes(png)


def test_verify_probe_output_accepts_the_expected_blurred_color(confirm, tmp_path):
    path = tmp_path / "out.png"
    _make_solid_png(path, confirm._PROBE_RGB)
    ok, why = confirm._verify_probe_output(path)
    assert ok is True and "real GPU kernel ran" in why


@pytest.mark.parametrize("ftype", [0, 1, 2, 3, 4], ids=["none", "sub", "up", "average", "paeth"])
def test_verify_probe_output_decodes_every_png_filter_type(confirm, tmp_path, ftype):
    """The real bug this fixes: the first live run against a real ComfyUI
    produced a filter-type-2 (Up) PNG and the decoder only handled type 0,
    reporting a FALSE FAIL on a candidate that had actually passed. Every
    filter type the PNG spec defines must round-trip correctly, not just the
    one this test author happened to try first."""
    path = tmp_path / "out.png"
    _make_solid_png(path, confirm._PROBE_RGB, filter_type=ftype)
    ok, why = confirm._verify_probe_output(path)
    assert ok is True, why


def test_verify_probe_output_decodes_a_realistic_adaptive_per_row_filter_mix(confirm, tmp_path):
    """A real encoder (ComfyUI's own PIL-based PNG writer, matching the
    filter-2 output actually observed) picks a filter PER ROW, not one for
    the whole image - Up/Average/Paeth on any row before the target depend on
    the row above's RECONSTRUCTED bytes, so this is the case that would have
    caught the original bug even without knowing "filter 2" in advance."""
    path = tmp_path / "out.png"
    filters = [(y % 5) for y in range(64)]  # every filter type appears, in rotation
    _make_solid_png(path, confirm._PROBE_RGB, per_row_filters=filters)
    ok, why = confirm._verify_probe_output(path)
    assert ok is True, why


def test_verify_probe_output_accepts_one_off_rounding(confirm, tmp_path):
    path = tmp_path / "out.png"
    r, g, b = confirm._PROBE_RGB
    _make_solid_png(path, (r + 1, g - 1, b))
    ok, _ = confirm._verify_probe_output(path)
    assert ok is True


def test_verify_probe_output_rejects_the_wrong_color(confirm, tmp_path):
    """The actual regression this check exists to catch: a candidate that
    executes the graph but produces a wrong result (a broken kernel, a
    color-space bug, a node that silently no-ops)."""
    path = tmp_path / "out.png"
    _make_solid_png(path, (0, 0, 0))
    ok, why = confirm._verify_probe_output(path)
    assert ok is False
    assert "expected" in why


def test_verify_probe_output_rejects_the_wrong_size(confirm, tmp_path):
    path = tmp_path / "out.png"
    _make_solid_png(path, confirm._PROBE_RGB, size=32)
    ok, why = confirm._verify_probe_output(path)
    assert ok is False and "32x32" in why


def test_verify_probe_output_rejects_garbage(confirm, tmp_path):
    path = tmp_path / "out.png"
    path.write_bytes(b"not a png at all")
    ok, why = confirm._verify_probe_output(path)
    assert ok is False and "not a PNG" in why


# --------------------------------------------------------------------------- #
#  Free-port discovery, against a REAL bound socket                          #
# --------------------------------------------------------------------------- #

def test_free_port_is_actually_free_and_port_answers_detects_a_real_listener(confirm):
    port = confirm._free_port()
    assert confirm._port_answers(port) is False, "a freshly-chosen port must not already answer"

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    bound_port = srv.getsockname()[1]
    try:
        assert confirm._port_answers(bound_port) is True
    finally:
        srv.close()


# --------------------------------------------------------------------------- #
#  requirements_changed: against a REAL git repo                             #
# --------------------------------------------------------------------------- #

def _git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def real_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q", "-b", "master"], repo)
    _git(["config", "user.email", "test@example.com"], repo)
    _git(["config", "user.name", "Test"], repo)
    (repo / "requirements.txt").write_text("torch\n", encoding="utf-8")
    _git(["add", "requirements.txt"], repo)
    _git(["commit", "-q", "-m", "old"], repo)
    old = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                         capture_output=True, text=True, check=True).stdout.strip()
    return repo, old


def test_requirements_changed_true_when_it_differs(confirm, real_repo):
    repo, old = real_repo
    (repo / "requirements.txt").write_text("torch\nnumpy\n", encoding="utf-8")
    _git(["commit", "-aq", "-m", "new"], repo)
    new = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                         capture_output=True, text=True, check=True).stdout.strip()
    assert confirm._requirements_changed(repo, old, new) is True


def test_requirements_changed_false_when_it_does_not(confirm, real_repo):
    repo, old = real_repo
    (repo / "other.txt").write_text("x\n", encoding="utf-8")
    _git(["add", "other.txt"], repo)
    _git(["commit", "-q", "-m", "unrelated"], repo)
    new = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                         capture_output=True, text=True, check=True).stdout.strip()
    assert confirm._requirements_changed(repo, old, new) is False


def test_requirements_changed_none_when_it_cannot_be_determined(confirm, tmp_path):
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    assert confirm._requirements_changed(not_a_repo, "a" * 40, "b" * 40) is None


# --------------------------------------------------------------------------- #
#  CLI validation                                                            #
# --------------------------------------------------------------------------- #

def test_a_malformed_tag_is_refused_before_anything_runs(confirm, tmp_path, capsys):
    rc = confirm.main(["--tag", "latest", "--commit", "a" * 40,
                       "--workdir", str(tmp_path / "wd"), "--receipt",
                       str(tmp_path / "r.json")])
    assert rc == 1
    assert "not an upstream release tag" in capsys.readouterr().out


def test_a_malformed_commit_is_refused_before_anything_runs(confirm, tmp_path, capsys):
    rc = confirm.main(["--tag", "v0.32.0", "--commit", "not-hex",
                       "--workdir", str(tmp_path / "wd"), "--receipt",
                       str(tmp_path / "r.json")])
    assert rc == 1
    assert "not a 40-character hex commit sha" in capsys.readouterr().out


def test_smoke_phase_without_a_provision_receipt_is_inconclusive(confirm, tmp_path, capsys,
                                                                  monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "unused-home"))
    rc = confirm.run_smoke_phase("v0.32.0", "a" * 40, tmp_path / "wd", tmp_path / "no-receipt.json")
    assert rc == 2
    assert "INCONCLUSIVE" in capsys.readouterr().out


def test_smoke_phase_refuses_when_provision_checks_did_not_all_pass(confirm, tmp_path, capsys):
    receipt_path = tmp_path / "r.json"
    r = confirm._new_receipt("v0.32.0", "a" * 40)
    confirm._set_check(r, "provision", confirm.PASS, "ok")
    confirm._set_check(r, "checkout", confirm.FAIL, "mismatch")
    confirm._set_check(r, "custom_nodes", confirm.PASS, "ok")
    confirm._set_check(r, "localm_patches", confirm.PASS, "ok")
    confirm._save_receipt(receipt_path, r)
    rc = confirm.run_smoke_phase("v0.32.0", "a" * 40, tmp_path / "wd", receipt_path)
    assert rc == 2
    assert "provision phase did not fully pass" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
#  Teardown: keeps cache/, removes everything else                          #
# --------------------------------------------------------------------------- #

def test_teardown_keeps_cache_and_removes_everything_else(confirm, tmp_path):
    workdir = tmp_path / "wd"
    home = workdir / "home"
    (home / "cache").mkdir(parents=True)
    (home / "cache" / "keep-me.whl").write_text("x", encoding="utf-8")
    (home / "comfyui").mkdir()
    (home / "comfyui" / "main.py").write_text("x", encoding="utf-8")
    (home / "stray-file.txt").write_text("x", encoding="utf-8")
    (workdir / "tmp").mkdir()
    (workdir / "tmp" / "scratch.tmp").write_text("x", encoding="utf-8")

    rc = confirm.run_teardown_phase(workdir)

    assert rc == 0
    assert (home / "cache" / "keep-me.whl").exists(), "the pip/torch cache survives"
    assert not (home / "comfyui").exists(), "the installed ComfyUI is removed"
    assert not (home / "stray-file.txt").exists()
    assert not (workdir / "tmp").exists()


def test_teardown_removes_readonly_files_a_real_git_checkout_leaves(confirm, tmp_path):
    """The actual bug found by the first live run: a managed ComfyUI is a git
    checkout, git marks its object store read-only, and a plain
    shutil.rmtree(ignore_errors=True) then silently leaves .git/custom_nodes
    behind while still reporting success. Reproduces the exact mechanism (a
    real read-only file), not just a directory shutil.rmtree could always
    have deleted anyway."""
    import os
    import stat
    workdir = tmp_path / "wd"
    comfyui = workdir / "home" / "comfyui"
    comfyui.mkdir(parents=True)
    ro_file = comfyui / "readonly.pack"
    ro_file.write_text("git object data", encoding="utf-8")
    os.chmod(ro_file, stat.S_IREAD)

    try:
        rc = confirm.run_teardown_phase(workdir)
    finally:
        # Defensive: if the test itself fails before removal, leave nothing
        # read-only behind for pytest's own tmp_path cleanup to trip over.
        if ro_file.exists():
            os.chmod(ro_file, stat.S_IWRITE)

    assert rc == 0
    assert not comfyui.exists(), "the read-only file (and its parent) must actually be removed"


def test_teardown_reports_failure_rather_than_silently_swallowing_one(confirm, tmp_path,
                                                                       monkeypatch):
    """A GENUINE removal failure (not just a read-only file rmtree_robust can
    fix) must be surfaced as a real failure, never reported as a clean
    teardown - the exact rule-5 violation the read-only-file bug above was an
    instance of."""
    workdir = tmp_path / "wd"
    (workdir / "home" / "comfyui").mkdir(parents=True)

    def _boom(path):
        raise OSError("simulated: genuinely could not remove this")
    monkeypatch.setattr(
        "localm.media.managed_comfy.rmtree_robust", _boom)

    rc = confirm.run_teardown_phase(workdir)
    assert rc == 1


# --------------------------------------------------------------------------- #
#  _check_nodes_registered / _check_shipped_workflows / _check_gpu_roundtrip #
#  - no live run exercised these directly before, only end to end on real    #
#  hardware, so their FAIL/INCONCLUSIVE paths had zero unit coverage.        #
# --------------------------------------------------------------------------- #

class _FakeHTTPError:
    """Stands in for urllib.error.HTTPError: .read() returns the body bytes
    a real ComfyUI /prompt 400 response would carry."""
    def __init__(self, code, body: dict):
        self.code = code
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body


class _FakeComfyClient:
    """A minimal stand-in for localm.media.comfy_client, driven entirely by
    the test - never a real socket."""
    SUBMIT_OK = "ok"
    SUBMIT_HTTP_ERROR = "http_error"
    POLL_FINISHED = "finished"
    POLL_TIMEOUT = "timeout"

    def __init__(self, *, submit_result, poll_result=None, output_info=None,
                fetch_writes: bytes = b""):
        self._submit_result = submit_result
        self._poll_result = poll_result
        self._output_info = output_info
        self._fetch_writes = fetch_writes
        self.interrupted = []
        self.fetched_to = None

    def comfy_object_info(self, api_url):
        return self._object_info

    def comfy_submit_prompt(self, api_url, workflow, timeout=10.0):
        return self._submit_result

    def interrupt_comfy(self, api_url):
        self.interrupted.append(api_url)

    def comfy_poll_until_done(self, api_url, prompt_id, *, max_poll_seconds, **kw):
        return self._poll_result

    def select_output_info(self, entry, output_keys):
        return self._output_info

    def comfy_fetch_output(self, api_url, info, output_path, *, timeout):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(self._fetch_writes)
        self.fetched_to = output_path


def test_check_nodes_registered_passes_when_everything_is_present(confirm):
    receipt = confirm._new_receipt("v0.32.0", "a" * 40)
    fake = _FakeComfyClient(submit_result=None)
    fake._object_info = {"EmptyImage": {}, "SaveImage": {}, "SelectModelDevice": {}}
    confirm._check_nodes_registered(
        receipt, "http://x", ("SelectModelDevice",),
        lambda: [], lambda f: {"EmptyImage", "SaveImage"}, fake)
    assert confirm._check_passed(receipt, "nodes_registered")


def test_check_nodes_registered_fails_when_a_node_is_missing(confirm):
    """The regression this exists to catch: a custom node (or a core node
    upstream renamed/removed) that no longer registers against the candidate
    commit."""
    receipt = confirm._new_receipt("v0.32.0", "a" * 40)
    fake = _FakeComfyClient(submit_result=None)
    fake._object_info = {"EmptyImage": {}}  # SaveImage missing
    confirm._check_nodes_registered(
        receipt, "http://x", (), lambda: [Path("fake_workflow.json")],
        lambda f: {"EmptyImage", "SaveImage"}, fake)
    assert receipt["checks"]["nodes_registered"]["verdict"] == confirm.FAIL
    assert "SaveImage" in receipt["checks"]["nodes_registered"]["why"]


def test_check_nodes_registered_inconclusive_when_object_info_unreachable(confirm):
    receipt = confirm._new_receipt("v0.32.0", "a" * 40)
    fake = _FakeComfyClient(submit_result=None)
    fake._object_info = None
    confirm._check_nodes_registered(receipt, "http://x", (), lambda: [], lambda f: set(), fake)
    assert receipt["checks"]["nodes_registered"]["verdict"] == confirm.INCONCLUSIVE


def test_check_shipped_workflows_accepts_a_missing_model_rejection(confirm, tmp_path):
    """With an empty scratch models dir, a value_not_in_list error whose
    received_value is a model filename is the ONE acceptable rejection - the
    workflow validated correctly and simply has no model to run against yet."""
    wf = tmp_path / "flux_workflow.json"
    wf.write_text(json.dumps({"1": {"class_type": "CheckpointLoaderSimple", "inputs": {}}}),
                  encoding="utf-8")
    body = {"node_errors": {"1": {"class_type": "CheckpointLoaderSimple", "errors": [
        {"type": "value_not_in_list", "message": "Value not in list",
         "extra_info": {"input_name": "ckpt_name", "received_value": "flux1-dev.safetensors"}}
    ]}}}
    receipt = confirm._new_receipt("v0.32.0", "a" * 40)
    fake = _FakeComfyClient(submit_result=(_FakeComfyClient.SUBMIT_HTTP_ERROR,
                                           _FakeHTTPError(400, body)))
    confirm._check_shipped_workflows(receipt, "http://x", lambda: [wf], (".safetensors",), fake)
    assert confirm._check_passed(receipt, "shipped_workflows")


def test_check_shipped_workflows_fails_on_any_other_validation_error(confirm, tmp_path):
    """The actual regression this exists to catch: a renamed input, a
    dropped enum option, a changed output slot - anything that is NOT simply
    'this model file is not on disk'."""
    wf = tmp_path / "flux_workflow.json"
    wf.write_text("{}", encoding="utf-8")
    body = {"node_errors": {"1": {"class_type": "KSampler", "errors": [
        {"type": "value_not_in_list", "message": "Value not in list",
         "extra_info": {"input_name": "sampler_name", "received_value": "euler"}}
    ]}}}
    receipt = confirm._new_receipt("v0.32.0", "a" * 40)
    fake = _FakeComfyClient(submit_result=(_FakeComfyClient.SUBMIT_HTTP_ERROR,
                                           _FakeHTTPError(400, body)))
    confirm._check_shipped_workflows(receipt, "http://x", lambda: [wf], (".safetensors",), fake)
    assert receipt["checks"]["shipped_workflows"]["verdict"] == confirm.FAIL
    assert "sampler_name" in receipt["checks"]["shipped_workflows"]["why"]
    assert "euler" in receipt["checks"]["shipped_workflows"]["why"]


def test_check_shipped_workflows_accepts_outright_submission(confirm, tmp_path):
    """A workflow needing no model at all (or one that happens to already
    resolve) is accepted outright - and the run is interrupted so it never
    actually executes during a confirm."""
    wf = tmp_path / "model_free.json"
    wf.write_text("{}", encoding="utf-8")
    receipt = confirm._new_receipt("v0.32.0", "a" * 40)
    fake = _FakeComfyClient(submit_result=(_FakeComfyClient.SUBMIT_OK, "prompt-id-1"))
    confirm._check_shipped_workflows(receipt, "http://x", lambda: [wf], (".safetensors",), fake)
    assert confirm._check_passed(receipt, "shipped_workflows")
    assert fake.interrupted == ["http://x"]


def test_check_gpu_roundtrip_passes_and_fetches_the_output(confirm, tmp_path):
    real_png = tmp_path / "src.png"
    _make_solid_png(real_png, confirm._PROBE_RGB)
    receipt = confirm._new_receipt("v0.32.0", "a" * 40)
    fake = _FakeComfyClient(
        submit_result=(_FakeComfyClient.SUBMIT_OK, "prompt-id-1"),
        poll_result=(_FakeComfyClient.POLL_FINISHED, {"outputs": {}}),
        output_info={"filename": "out.png"},
        fetch_writes=real_png.read_bytes())
    confirm._check_gpu_roundtrip(receipt, "http://x", tmp_path, fake)
    assert confirm._check_passed(receipt, "gpu_roundtrip")
    assert fake.fetched_to == tmp_path / "gpu_roundtrip_output.png"


def test_check_gpu_roundtrip_fails_when_polling_times_out(confirm, tmp_path):
    receipt = confirm._new_receipt("v0.32.0", "a" * 40)
    fake = _FakeComfyClient(
        submit_result=(_FakeComfyClient.SUBMIT_OK, "prompt-id-1"),
        poll_result=(_FakeComfyClient.POLL_TIMEOUT, None))
    confirm._check_gpu_roundtrip(receipt, "http://x", tmp_path, fake)
    assert receipt["checks"]["gpu_roundtrip"]["verdict"] == confirm.FAIL
    assert "did not finish" in receipt["checks"]["gpu_roundtrip"]["why"]
