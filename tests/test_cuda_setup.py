# SPDX-License-Identifier: AGPL-3.0-or-later
"""CUDA preflight, self-assembly, and the load-test + graceful fallback for
``localm setup-llama``.

CUDA is the visible "peak NVIDIA performance" option, so picking it has to land
or fall back cleanly. These tests pin, with no network and no real GPU:
  * the nvidia-smi banner is parsed into a driver / CUDA-capability report;
  * the CUDA build + matching cudart bundle are resolved from a release listing;
  * the dialogue picks the right action per preflight state;
  * a chosen backend that does not load falls back vulkan -> cpu;
  * an explicit --sha256 pin is NEVER bypassed by the fallback.
"""

from __future__ import annotations

import io
import json
import sys

import pytest

from localm import setup_llama as sl
from tests._fake_https import patch_https_transport


# --------------------------- preflight parsing ---------------------------- #

def test_ver_tuple_parses_and_tolerates_junk():
    assert sl._ver_tuple("12.4") == (12, 4)
    assert sl._ver_tuple("13") == (13,)
    # Unparseable version -> None ("unknown"), NOT (0, 0), so driver_ok does not
    # block an unmeasurable capability as a too-old driver.
    assert sl._ver_tuple("") is None
    assert sl._ver_tuple("not.a.version") is None


def test_nvidia_preflight_parses_banner(monkeypatch):
    def fake_smi(*args):
        joined = " ".join(str(a) for a in args)
        if "compute_cap" in joined:
            return "8.9\n"
        if "--query-gpu=name" in joined:
            return "NVIDIA GeForce RTX 4070\n"
        return ("|  NVIDIA-SMI 552.22   Driver Version: 552.22   "
                "CUDA Version: 12.4  |\n")
    monkeypatch.setattr(sl, "_nvidia_smi", fake_smi)
    info = sl.nvidia_preflight()
    assert info.present
    assert info.driver_version == "552.22"
    assert info.cuda_capability == "12.4"
    assert info.gpu_name == "NVIDIA GeForce RTX 4070"
    assert info.compute_capability == "8.9"
    assert info.cuda_line == "cuda-12"
    assert info.driver_ok is True


def test_nvidia_preflight_absent_when_no_smi(monkeypatch):
    monkeypatch.setattr(sl, "_nvidia_smi", lambda *a: "")
    info = sl.nvidia_preflight()
    assert info.present is False
    assert info.driver_ok is True   # unknown capability never blocks


def test_driver_ok_false_for_old_capability():
    assert sl.NvidiaInfo(present=True, cuda_capability="11.2").driver_ok is False
    assert sl.NvidiaInfo(present=True, cuda_capability="12.4").driver_ok is True
    assert sl.NvidiaInfo(present=True, cuda_capability="13.3").driver_ok is True


# --------------------------- asset resolution ----------------------------- #

_FAKE_ASSETS = [
    {"name": "llama-b9685-bin-win-cuda-12.4-x64.zip",
     "browser_download_url": "https://x/llama-cuda-12.4.zip", "size": 200_000_000},
    {"name": "llama-b9685-bin-win-cuda-13.3-x64.zip",
     "browser_download_url": "https://x/llama-cuda-13.3.zip", "size": 200_000_000},
    {"name": "cudart-llama-bin-win-cuda-12.4-x64.zip",
     "browser_download_url": "https://x/cudart-12.4.zip", "size": 400_000_000},
    {"name": "cudart-llama-bin-win-cuda-13.3-x64.zip",
     "browser_download_url": "https://x/cudart-13.3.zip", "size": 400_000_000},
    {"name": "llama-b9685-bin-win-vulkan-x64.zip",
     "browser_download_url": "https://x/vulkan.zip", "size": 100_000_000},
]


def test_resolve_cuda_pair_prefers_12_line(monkeypatch):
    monkeypatch.setattr(sl, "_release_assets", lambda tag: _FAKE_ASSETS)
    build, cudart = sl._resolve_cuda_pair("b9685")
    assert build["name"] == "llama-b9685-bin-win-cuda-12.4-x64.zip"
    assert cudart["name"] == "cudart-llama-bin-win-cuda-12.4-x64.zip"


def test_resolve_cuda_pair_none_when_listing_empty(monkeypatch):
    monkeypatch.setattr(sl, "_release_assets", lambda tag: [])
    build, cudart = sl._resolve_cuda_pair("b9685")
    assert build is None and cudart is None


def test_pick_asset_requires_all_needles():
    assert sl._pick_asset(_FAKE_ASSETS, "vulkan")["name"].endswith("vulkan-x64.zip")
    assert sl._pick_asset(_FAKE_ASSETS, "cudart", "win-cuda-12") is not None
    assert sl._pick_asset(_FAKE_ASSETS, "rocm") is None


# ------------------ _latest_tag skips a not-yet-uploaded release ---------- #
#
# Upstream can publish a release (tag + notes) before its CI finishes uploading
# the platform archives, so `assets` can be empty while the release body already
# lists the download URLs. _latest_tag must skip such a tag.

class _FakeHTTP:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return io.BytesIO(self._b)

    def __exit__(self, *a):
        return False


def test_latest_tag_skips_release_with_no_uploaded_assets(monkeypatch):
    releases = [
        {"tag_name": "b9871", "draft": False, "prerelease": False, "assets": [],
         "body": "[Windows x64 (CUDA 12)](https://github.com/ggml-org/llama.cpp/"
                 "releases/download/b9871/llama-b9871-bin-win-cuda-12.4-x64.zip)"},
        {"tag_name": "b9870", "draft": False, "prerelease": False,
         "assets": [{"name": "llama-b9870-bin-win-cuda-12.4-x64.zip",
                     "browser_download_url": "https://example/real-cuda.zip",
                     "size": 200_000_000}]},
    ]
    patch_https_transport(monkeypatch,
                        lambda req, timeout=None, context=None: _FakeHTTP(releases))
    assert sl._latest_tag() == "b9870"     # NOT the not-yet-uploaded b9871


def test_latest_tag_skips_draft_and_prerelease(monkeypatch):
    releases = [
        {"tag_name": "b9872", "draft": True, "prerelease": False,
         "assets": [{"name": "x", "browser_download_url": "https://x", "size": 1}]},
        {"tag_name": "b9871", "draft": False, "prerelease": True,
         "assets": [{"name": "x", "browser_download_url": "https://x", "size": 1}]},
        {"tag_name": "b9870", "draft": False, "prerelease": False,
         "assets": [{"name": "x", "browser_download_url": "https://x", "size": 1}]},
    ]
    patch_https_transport(monkeypatch,
                        lambda req, timeout=None, context=None: _FakeHTTP(releases))
    assert sl._latest_tag() == "b9870"


def test_latest_tag_falls_back_when_nothing_has_assets(monkeypatch):
    releases = [
        {"tag_name": "b9871", "draft": False, "prerelease": False, "assets": []},
        {"tag_name": "b9870", "draft": False, "prerelease": False, "assets": []},
    ]
    patch_https_transport(monkeypatch,
                        lambda req, timeout=None, context=None: _FakeHTTP(releases))
    assert sl._latest_tag() == sl._PINNED_TAG


def test_latest_tag_falls_back_on_network_error(monkeypatch):
    def boom(req, timeout=None, context=None):
        raise OSError("no network")
    patch_https_transport(monkeypatch, boom)
    assert sl._latest_tag() == sl._PINNED_TAG


def test_release_assets_does_not_scrape_body_when_assets_empty(monkeypatch):
    """The body-scrape fallback removed here was the actual 404 source: a
    release with assets=[] must yield an honest empty list, not simulated
    entries built from not-yet-live body links."""
    payload = {
        "assets": [],
        "body": "[Windows x64 (CUDA 12)](https://github.com/ggml-org/llama.cpp/"
                "releases/download/b9871/llama-b9871-bin-win-cuda-12.4-x64.zip)",
    }
    patch_https_transport(monkeypatch,
                        lambda req, timeout=None, context=None: _FakeHTTP(payload))
    assert sl._release_assets("b9871") == []


# --------------------------- the CUDA dialogue ----------------------------- #

def _confirm(monkeypatch, answer: bool):
    monkeypatch.setattr(sl.click, "confirm", lambda *a, **k: answer)


def test_dialogue_driver_ok_confirm_fetches_cuda(monkeypatch):
    _confirm(monkeypatch, True)
    info = sl.NvidiaInfo(present=True, gpu_name="RTX 4070",
                         driver_version="552.22", cuda_capability="12.4")
    assert sl._cuda_setup_dialogue(info, assume_yes=False) == ("cuda", True)


def test_dialogue_driver_ok_decline_falls_back_to_vulkan(monkeypatch):
    _confirm(monkeypatch, False)
    info = sl.NvidiaInfo(present=True, cuda_capability="12.4")
    assert sl._cuda_setup_dialogue(info, assume_yes=False) == ("vulkan", False)


def test_dialogue_old_driver_recommends_vulkan(monkeypatch):
    # No confirm should be needed; an old driver cannot be self-assembled.
    _confirm(monkeypatch, True)
    info = sl.NvidiaInfo(present=True, driver_version="460.0", cuda_capability="11.2")
    assert sl._cuda_setup_dialogue(info, assume_yes=False) == ("vulkan", False)


def test_dialogue_no_nvidia_continue_is_users_choice(monkeypatch):
    _confirm(monkeypatch, True)
    info = sl.NvidiaInfo(present=False)
    assert sl._cuda_setup_dialogue(info, assume_yes=False) == ("cuda", True)


def test_dialogue_no_nvidia_decline_uses_vulkan(monkeypatch):
    _confirm(monkeypatch, False)
    info = sl.NvidiaInfo(present=False)
    assert sl._cuda_setup_dialogue(info, assume_yes=False) == ("vulkan", False)


def test_dialogue_assume_yes_fetches_when_driver_ok():
    info = sl.NvidiaInfo(present=True, cuda_capability="12.4")
    assert sl._cuda_setup_dialogue(info, assume_yes=True) == ("cuda", True)


def test_dialogue_assume_yes_uses_vulkan_when_no_nvidia():
    info = sl.NvidiaInfo(present=False)
    assert sl._cuda_setup_dialogue(info, assume_yes=True) == ("vulkan", False)


# ------------- known-vendor case: name it, recommend the REAL match -------- #
#
# When det shows a specific other vendor: the status line names that vendor, the
# recommendation is computed via the same policy setup.bat/sh use rather than a
# hardcoded vulkan, and a three-way choice (continue / switch / quit) replaces
# the binary confirm. With det omitted, the generic path is unchanged.

def _amd_windows_det():
    from localm import hwdetect
    return hwdetect.Detection(vendors=["amd"])


def test_dialogue_no_nvidia_known_amd_names_vendor_in_status(monkeypatch, capsys):
    monkeypatch.setattr(sl.click, "prompt", lambda *a, **k: "2")
    info = sl.NvidiaInfo(present=False)
    sl._cuda_setup_dialogue(info, assume_yes=False, det=_amd_windows_det())
    out = capsys.readouterr().out
    assert "amd" in out.lower()
    assert "not on path" not in out.lower()   # the old generic hedge is gone here


def test_dialogue_no_nvidia_known_amd_offers_three_way_choice(monkeypatch, capsys):
    monkeypatch.setattr(sl.click, "prompt", lambda *a, **k: "2")
    info = sl.NvidiaInfo(present=False)
    sl._cuda_setup_dialogue(info, assume_yes=False, det=_amd_windows_det())
    out = capsys.readouterr().out
    assert "Continue with CUDA anyway" in out
    assert "Quit" in out
    assert "Switch to" in out


def test_dialogue_no_nvidia_known_amd_pick_continue(monkeypatch):
    monkeypatch.setattr(sl.click, "prompt", lambda *a, **k: "1")
    info = sl.NvidiaInfo(present=False)
    assert sl._cuda_setup_dialogue(info, False, _amd_windows_det()) == ("cuda", True)


def test_dialogue_no_nvidia_known_amd_pick_switch_uses_real_recommendation(monkeypatch):
    # The recommendation comes from hwdetect.recommended_install_backend, not a
    # hardcoded "vulkan": fake that function and confirm the result is honoured.
    from localm import hwdetect
    monkeypatch.setattr(hwdetect, "recommended_install_backend", lambda det: "amd-rocm")
    monkeypatch.setattr(sl.click, "prompt", lambda *a, **k: "2")
    info = sl.NvidiaInfo(present=False)
    assert sl._cuda_setup_dialogue(info, False, _amd_windows_det()) == ("amd-rocm", False)


def test_dialogue_no_nvidia_known_amd_pick_quit_exits(monkeypatch):
    monkeypatch.setattr(sl.click, "prompt", lambda *a, **k: "3")
    info = sl.NvidiaInfo(present=False)
    with pytest.raises(SystemExit):
        sl._cuda_setup_dialogue(info, False, _amd_windows_det())


def test_dialogue_assume_yes_known_amd_uses_recommendation_not_hardcoded_vulkan(monkeypatch):
    from localm import hwdetect
    monkeypatch.setattr(hwdetect, "recommended_install_backend", lambda det: "amd-rocm")
    info = sl.NvidiaInfo(present=False)
    assert sl._cuda_setup_dialogue(info, True, _amd_windows_det()) == ("amd-rocm", False)


def test_dialogue_det_with_only_nvidia_vendor_falls_back_to_generic(monkeypatch):
    # hwdetect reports NVIDIA present (e.g. a broken driver install) but
    # nvidia-smi itself failed: no other vendor to recommend, so this takes the
    # generic path.
    from localm import hwdetect
    monkeypatch.setattr(sl.click, "confirm", lambda *a, **k: False)
    info = sl.NvidiaInfo(present=False)
    det = hwdetect.Detection(vendors=["nvidia"])
    assert sl._cuda_setup_dialogue(info, False, det) == ("vulkan", False)


def test_warn_off_profile_returns_detection_for_reuse(monkeypatch):
    """The caller (main()) needs this SAME detection for the CUDA dialogue,
    so a second, potentially-inconsistent hwdetect.detect() call is never
    needed (or skipped) downstream."""
    _fake_vendors(monkeypatch, ["amd"])
    det = sl._warn_off_profile("cuda")
    assert det is not None
    assert det.vendors == ["amd"]


def test_warn_off_profile_returns_none_for_non_vendor_specific_backend(monkeypatch):
    # vulkan/cpu are not vendor-specific - no detection is needed or run.
    from localm import hwdetect
    monkeypatch.setattr(hwdetect, "detect",
                        lambda: pytest.fail("must not detect for a non-vendor-specific backend"))
    assert sl._warn_off_profile("vulkan") is None


def test_main_threads_detection_from_warn_off_profile_into_cuda_dialogue(monkeypatch, tmp_path):
    """End-to-end wiring check: `setup-llama --backend cuda` on a machine
    hwdetect identifies as AMD must reach the vendor-aware three-way dialogue
    (not the old generic one) using the SAME detection _warn_off_profile
    already computed - proving main() actually threads det through, not just
    that _cuda_setup_dialogue behaves correctly in isolation."""
    from localm import hwdetect
    monkeypatch.setattr(sl.sys, "platform", "win32")
    monkeypatch.setattr(hwdetect, "detect", lambda: hwdetect.Detection(vendors=["amd"]))
    monkeypatch.setattr(hwdetect, "recommended_install_backend", lambda det: "amd-rocm")
    monkeypatch.setattr(sl, "nvidia_preflight", lambda: sl.NvidiaInfo(present=False))

    target = tmp_path / "lib"
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: target)
    provisioned = []

    def fake_provision_backend(backend, tgt, sha256, with_cudart, cuda_line=sl._CUDA_LINE, tag=None):
        provisioned.append(backend)
        (tgt / sl._lib_name()).write_bytes(b"stub")

    monkeypatch.setattr(sl, "_provision_backend", fake_provision_backend)
    monkeypatch.setattr(sl, "_clear_target", lambda tgt: None)
    monkeypatch.setattr(sl, "_install_runtime_wheel", lambda pkg_dir: True)
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: (True, ""))
    monkeypatch.setattr(sl, "_verify", lambda: None)

    from click.testing import CliRunner
    runner = CliRunner()
    # "2" answers the three-way prompt: switch to the recommendation.
    result = runner.invoke(sl.main, ["--backend", "cuda"], input="2\n")
    assert result.exit_code == 0, result.output
    assert "this machine looks like" in result.output.lower()
    assert "amd" in result.output.lower()
    assert "Switch to amd-rocm" in result.output
    assert provisioned == ["amd-rocm"], (
        "the recommendation from the three-way dialogue must be what actually gets provisioned")


def test_main_threads_cuda_line_detection_on_linux(monkeypatch, tmp_path):
    """nvidia_preflight()/_cuda_setup_dialogue() must be reached on Linux, not
    only win32. Without it a cuda-13-line GPU gets the cuda-12 line - a build
    with no kernels for that architecture - and no with_cudart, skipping the
    PyPI cudart runtime fetch. That runtime still LOADS (a valid cuda-12
    binary, so the ABI check passes) and registers zero usable GPU devices.
    _resolve_backend_asset/_fetch_cuda_runtime_libs already handle cuda-13 on
    Linux (test_linux_cuda_runtime_provisioning.py); this covers main()
    detecting and passing it through."""
    monkeypatch.setattr(sl.sys, "platform", "linux")
    # A real Blackwell compute capability; NvidiaInfo.cuda_line resolves it to
    # "cuda-13".
    monkeypatch.setattr(sl, "nvidia_preflight", lambda: sl.NvidiaInfo(
        present=True, gpu_name="NVIDIA RTX PRO 4000 Blackwell",
        driver_version="580.65", cuda_capability="13.3",
        compute_capability="12.0"))

    target = tmp_path / "lib"
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: target)
    calls = []

    def fake_provision_backend(backend, tgt, sha256, with_cudart, cuda_line=sl._CUDA_LINE, tag=None):
        calls.append((backend, with_cudart, cuda_line))
        (tgt / sl._lib_name()).write_bytes(b"stub")

    monkeypatch.setattr(sl, "_provision_backend", fake_provision_backend)
    monkeypatch.setattr(sl, "_clear_target", lambda tgt: None)
    monkeypatch.setattr(sl, "_install_runtime_wheel", lambda pkg_dir: True)
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: (True, ""))
    monkeypatch.setattr(sl, "_verify", lambda: None)

    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(sl.main, ["--backend", "cuda", "--yes"])
    assert result.exit_code == 0, result.output
    assert calls == [("cuda", True, "cuda-13")], (
        "on Linux, a real Blackwell GPU must reach _provision_backend with "
        "with_cudart=True and cuda_line='cuda-13', not the module default")


# ---------------- real click.confirm: reprompt + stray-input handling ------ #
#
# These use the REAL confirm() with only sys.stdin faked: garbage input re-asks
# the same question, and a genuine answer still lands correctly.

def test_dialogue_reprompts_on_invalid_input_then_accepts_real_answer(monkeypatch, capsys):
    # Two garbage lines, then a real "y" - click.confirm must re-ask after each
    # invalid line rather than erroring out or defaulting silently.
    monkeypatch.setattr("sys.stdin", io.StringIO("asdf\nqwerty\ny\n"))
    info = sl.NvidiaInfo(present=False)   # -> the "continue with CUDA anyway?" prompt
    result = sl._cuda_setup_dialogue(info, assume_yes=False)
    assert result == ("cuda", True)       # the real "y" was honoured
    out = capsys.readouterr().out
    assert out.count("Error: invalid input") == 2   # one reprompt per garbage line
    assert out.count("Continue with CUDA anyway?") == 3   # asked again each time


def test_dialogue_reprompts_then_declines(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("nope\nn\n"))
    info = sl.NvidiaInfo(present=False)
    result = sl._cuda_setup_dialogue(info, assume_yes=False)
    assert result == ("vulkan", False)
    assert capsys.readouterr().out.count("Error: invalid input") == 1


def test_flush_stdin_noop_on_non_tty(monkeypatch):
    # A piped/redirected stdin (tests, CI, scripted installs) is not a tty -
    # _flush_stdin must not touch it or raise.
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))
    sl._flush_stdin()   # must not raise
    assert sl.click.confirm("  ok?", default=False) is True


def test_flush_stdin_drains_pending_tty_bytes_on_windows(monkeypatch):
    """Windows path: a stray buffered Enter (or any keystroke) sitting in the
    console input queue from BEFORE the prompt appeared must be discarded, not
    silently consumed as the answer to the next click.confirm()."""
    monkeypatch.setattr(sl.sys, "platform", "win32")
    monkeypatch.setattr(sl.sys.stdin, "isatty", lambda: True, raising=False)
    calls = {"kbhit": [True, True, False], "getch": 0}

    def fake_kbhit():
        return calls["kbhit"].pop(0) if calls["kbhit"] else False

    def fake_getch():
        calls["getch"] += 1
        return b"\r"

    fake_msvcrt = type("_FakeMsvcrt", (), {"kbhit": staticmethod(fake_kbhit),
                                           "getch": staticmethod(fake_getch)})()
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    sl._flush_stdin()
    assert calls["getch"] == 2   # drained exactly the two buffered keystrokes


# --------------------------- off-profile warning -------------------------- #

def _fake_vendors(monkeypatch, vendors):
    from localm import hwdetect
    monkeypatch.setattr(hwdetect, "detect",
                        lambda: hwdetect.Detection(vendors=list(vendors)))


@pytest.mark.parametrize(
    "vendors, backend, expect_warning",
    [
        (["amd"], "cuda", True),      # cuda on an AMD-only box
        (["nvidia"], "cuda", False),  # vendor matches the backend -> quiet
        (["amd"], "vulkan", False),   # vulkan is not vendor-specific -> quiet
    ],
)
def test_warn_off_profile(monkeypatch, capsys, vendors, backend, expect_warning):
    _fake_vendors(monkeypatch, vendors)
    sl._warn_off_profile(backend)
    out = capsys.readouterr().out
    if expect_warning:
        assert "Heads up" in out
    else:
        assert "Heads up" not in out


# --------------------------- load-test + fallback ------------------------- #

def _stub_provision(monkeypatch):
    """_provision_backend writes the expected lib name into target; wheel install
    is a no-op. Lets us drive the fallback purely via _native_loads_ok."""
    lib = sl._lib_name()

    def fake_provision(backend, target, sha256, with_cudart, cuda_line=sl._CUDA_LINE, tag=None):
        (target / lib).write_bytes(b"x")
    monkeypatch.setattr(sl, "_provision_backend", fake_provision)
    monkeypatch.setattr(sl, "_install_runtime_wheel", lambda pkg: True)
    # Default to NON-interactive so the auto-fallback path is exercised
    # regardless of the test host's stdin; the inform+offer tests below override
    # this to True to drive the interactive prompt.
    monkeypatch.setattr(sl.sys.stdin, "isatty", lambda: False)


@pytest.mark.parametrize(
    "results, expected_backend",
    [
        ([(True, "")], "cuda"),
        ([(False, "cuda failed to load"), (True, "")], "vulkan"),
        ([(False, "cuda no"), (False, "vulkan no"), (True, "")], "cpu"),
    ],
)
def test_fallback_returns_first_backend_that_loads(monkeypatch, tmp_path, results, expected_backend):
    _stub_provision(monkeypatch)
    seq = iter(results)
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: next(seq))
    # (backend, tag): _provision_with_fallback reports which build the successful
    # attempt installed, so a fallback records the fallback's tag.
    assert sl._provision_with_fallback("cuda", tmp_path, None, True)[0] == expected_backend


def test_nothing_loads_raises_reportable_error(monkeypatch, tmp_path):
    _stub_provision(monkeypatch)
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: (False, "nope"))
    from localm.bugreport import LocalmError
    with pytest.raises(LocalmError) as ei:
        sl._provision_with_fallback("cuda", tmp_path, None, True)
    assert "no llama.cpp backend" in ei.value.summary


def test_nothing_loads_reports_every_attempt_not_just_the_last(monkeypatch, tmp_path):
    """The final report is the ONLY thing that survives into the saved bug
    report / the "Sorry - X because Y" console line (report_failure renders
    summary+reason only; the per-attempt console.print calls made along the
    way are not threaded into that context). So it must name EVERY backend
    tried and ITS OWN cause, chosen backend included - not just whichever one
    happened to be tried last. A fixture that returns the same detail string
    for every call cannot tell "only the last" from "all of them" apart -
    each call here returns a DISTINCT string so collapsing to one is provable."""
    _stub_provision(monkeypatch)
    seq = iter([(False, "cuda: driver too old"),
                (False, "vulkan: no ICD loader found"),
                (False, "cpu: illegal instruction")])
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: next(seq))
    from localm.bugreport import LocalmError
    with pytest.raises(LocalmError) as ei:
        sl._provision_with_fallback("cuda", tmp_path, None, True)
    reason = ei.value.reason
    assert "cuda: driver too old" in reason
    assert "vulkan: no ICD loader found" in reason
    assert "cpu: illegal instruction" in reason
    # The causes are listed in the order the backends were attempted.
    assert (reason.index("cuda: driver too old")
            < reason.index("vulkan: no ICD loader found")
            < reason.index("cpu: illegal instruction"))


def test_pinned_sha256_never_falls_back(monkeypatch, tmp_path):
    """A --sha256 pin means 'exactly this artifact' - a validation failure must
    stop, not silently swap to an unpinned vulkan build."""
    def boom(*a, **k):
        raise sl.ArtifactError("sha mismatch")
    monkeypatch.setattr(sl, "_provision_backend", boom)
    monkeypatch.setattr(sl, "_install_runtime_wheel", lambda pkg: True)
    # If a fallback were attempted it would call _native_loads_ok; make that loud.
    monkeypatch.setattr(sl, "_native_loads_ok",
                        lambda: pytest.fail("must not load-test after a pin failure"))
    with pytest.raises(SystemExit):
        sl._provision_with_fallback("cuda", tmp_path, "deadbeef", True)


def test_selfcontained_provisioned_but_unloadable_is_reportable(monkeypatch, tmp_path):
    """vulkan/cpu are the universal fallbacks; if one provisions but will not
    load, that is an unexpected fault - raise (report-worthy), not exit-0."""
    _stub_provision(monkeypatch)
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: (False, "broken binary"))
    from localm.bugreport import LocalmError
    with pytest.raises(LocalmError) as ei:
        sl._provision_with_fallback("vulkan", tmp_path, None, False)
    assert "did not load" in ei.value.summary


# --------------- inform + offer on load failure (never silent) ------------- #
# A chosen backend that does not load is NOT swapped silently. Interactive ->    #
# OFFER (accept falls back, decline stops); non-interactive -> fall back BUT     #
# warn loudly and say how to retry the pick.                                     #

def test_fallback_offer_accepted_provisions_vulkan(monkeypatch, tmp_path):
    _stub_provision(monkeypatch)
    seq = iter([(False, "cuda failed"), (True, "")])
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: next(seq))
    monkeypatch.setattr(sl.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sl.click, "confirm", lambda *a, **k: True)
    assert sl._provision_with_fallback("cuda", tmp_path, None, True)[0] == "vulkan"


def test_fallback_offer_declined_exits_nonzero(monkeypatch, tmp_path):
    """Declining the offer stops (exit non-zero) rather than swapping the pick."""
    _stub_provision(monkeypatch)
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: (False, "cuda failed"))
    monkeypatch.setattr(sl.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sl.click, "confirm", lambda *a, **k: False)
    with pytest.raises(SystemExit):
        sl._provision_with_fallback("cuda", tmp_path, None, True)


def test_fallback_noninteractive_warns_and_falls_back(monkeypatch, tmp_path, capsys):
    """assume_yes: no prompt, falls back, but warns loudly (never silent) and
    names how to retry the chosen backend with --force."""
    _stub_provision(monkeypatch)
    seq = iter([(False, "cuda failed"), (True, "")])
    monkeypatch.setattr(sl, "_native_loads_ok", lambda: next(seq))
    monkeypatch.setattr(sl.click, "confirm",
                        lambda *a, **k: pytest.fail("must not prompt with assume_yes"))
    out, _tag = sl._provision_with_fallback("cuda", tmp_path, None, True, assume_yes=True)
    assert out == "vulkan"
    printed = capsys.readouterr().out
    assert "Non-interactive" in printed
    assert "--force" in printed


# --------------------------- _native_loads_ok ----------------------------- #

def test_native_loads_ok_handles_subprocess_error(monkeypatch):
    import subprocess
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="python", timeout=120)
    monkeypatch.setattr(subprocess, "run", boom)
    ok, detail = sl._native_loads_ok()
    assert ok is False
    assert detail            # a non-empty, human-readable reason


# --------------------------- _clear_target -------------------------------- #

def test_clear_target_removes_files_keeps_subdirs(tmp_path):
    (tmp_path / "a.dll").write_bytes(b"x")
    (tmp_path / "b.so").write_bytes(b"y")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "keep.txt").write_bytes(b"z")
    sl._clear_target(tmp_path)
    assert not (tmp_path / "a.dll").exists()
    assert not (tmp_path / "b.so").exists()
    assert (sub / "keep.txt").exists()           # subdirectories are left alone


def test_clear_target_missing_dir_does_not_raise(tmp_path):
    sl._clear_target(tmp_path / "nope")          # must not raise


def test_clear_target_preserves_git_sentinels(tmp_path):
    """runtime/localm_llama_runtime/lib/ is a real directory in the git
    checkout, tracked only via its .gitignore ("*" / "!.gitkeep" /
    "!.gitignore"), which keeps setup-llama's downloaded native binaries out of
    version control. _clear_target must leave both sentinel files in place;
    deleting them empties the .gitignore and lets a later `git add -A` stage
    hundreds of MB of DLLs."""
    gitignore = tmp_path / ".gitignore"
    gitkeep = tmp_path / ".gitkeep"
    gitignore.write_text("*\n!.gitkeep\n!.gitignore\n", encoding="utf-8")
    gitkeep.write_text("", encoding="utf-8")
    stale_dll = tmp_path / "llama.dll"
    stale_dll.write_bytes(b"stale build")
    blas_dir = tmp_path / "rocblas"
    (blas_dir / "library").mkdir(parents=True)
    (blas_dir / "library" / "kernel.co").write_bytes(b"kernel")

    sl._clear_target(tmp_path)

    assert gitignore.read_text(encoding="utf-8") == "*\n!.gitkeep\n!.gitignore\n"
    assert gitkeep.exists()
    assert not stale_dll.exists()
    assert not blas_dir.exists()


# --------------------------- _provision_backend (cuda) -------------------- #

def test_provision_backend_cuda_fetches_build_and_cudart(monkeypatch, tmp_path):
    monkeypatch.setattr(sl.sys, "platform", "win32")
    monkeypatch.setattr(sl, "_latest_tag", lambda: "b9999")
    build = {"name": "llama-cuda-12.4.zip", "browser_download_url": "https://b", "size": 1}
    cudart = {"name": "cudart-12.4.zip", "browser_download_url": "https://c", "size": 1}
    monkeypatch.setattr(sl, "_resolve_cuda_pair", lambda tag, line=sl._CUDA_LINE: (build, cudart))
    fetched = []
    monkeypatch.setattr(sl, "_fetch_and_place",
                        lambda url, target, sha256=None: fetched.append(url))
    sl._provision_backend("cuda", tmp_path, None, with_cudart=True)
    assert fetched == ["https://b", "https://c"]


def test_provision_backend_cuda_passes_selected_line(monkeypatch, tmp_path):
    """The GPU-architecture-selected line must reach _resolve_cuda_pair, not
    just the module default - proves the plumbing from cuda_line through to
    the actual asset lookup, not merely that the default path still works."""
    monkeypatch.setattr(sl.sys, "platform", "win32")
    monkeypatch.setattr(sl, "_latest_tag", lambda: "b9999")
    build = {"name": "llama-cuda-13.3.zip", "browser_download_url": "https://b13", "size": 1}
    cudart = {"name": "cudart-13.3.zip", "browser_download_url": "https://c13", "size": 1}
    seen_lines = []

    def fake_resolve(tag, line=sl._CUDA_LINE):
        seen_lines.append(line)
        return build, cudart

    monkeypatch.setattr(sl, "_resolve_cuda_pair", fake_resolve)
    fetched = []
    monkeypatch.setattr(sl, "_fetch_and_place",
                        lambda url, target, sha256=None: fetched.append(url))
    sl._provision_backend("cuda", tmp_path, None, with_cudart=True, cuda_line="cuda-13")
    assert seen_lines == ["cuda-13"]
    assert fetched == ["https://b13", "https://c13"]


def test_provision_backend_cuda_no_assets_uses_templated_url(monkeypatch, tmp_path):
    monkeypatch.setattr(sl.sys, "platform", "win32")
    monkeypatch.setattr(sl, "_latest_tag", lambda: "b9999")
    monkeypatch.setattr(sl, "_resolve_cuda_pair", lambda tag, line=sl._CUDA_LINE: (None, None))
    monkeypatch.setattr(sl, "_resolve_backend_asset",
                        lambda backend, cuda_line=sl._CUDA_LINE, tag=None: ("https://templated/cuda.zip", "FALLBACK_SHA", "bTEST"))
    calls = []
    monkeypatch.setattr(sl, "_fetch_and_place",
                        lambda url, target, sha256=None: calls.append((url, sha256)))
    sl._provision_backend("cuda", tmp_path, "PIN", with_cudart=True)
    assert calls == [("https://templated/cuda.zip", "PIN")]   # build only, no cudart


# --------------------------- _warn_off_profile robustness ----------------- #

def test_warn_off_profile_survives_hwdetect_failure(monkeypatch):
    from localm import hwdetect
    def boom():
        raise RuntimeError("detect failed")
    monkeypatch.setattr(hwdetect, "detect", boom)
    sl._warn_off_profile("cuda")                 # must not raise
