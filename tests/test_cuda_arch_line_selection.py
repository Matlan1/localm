# SPDX-License-Identifier: AGPL-3.0-or-later
"""The CUDA runtime-LINE selection is driven by the GPU's own ARCHITECTURE
(compute capability), not just the platform.

NVIDIA Blackwell (datacenter sm_100 / consumer+workstation sm_120, e.g. RTX
50-series) is not supported by the 12.4-toolkit build's fatbin - upstream only
added Blackwell kernels starting CUDA 12.8 - so a Blackwell card handed the 12.x
build fails at inference time even though the DLL loads cleanly. Upstream ships
a 13.3-toolkit build that does support it.

These tests are entirely offline (no network, no real GPU) and drive the
selection logic with a faked ``NvidiaInfo`` across a card-architecture x
driver-capability matrix, including the unparseable and absent cases. NOT
covered here: that an sm_120 card loads a model end to end, which needs real
Blackwell hardware.
"""

from __future__ import annotations

import pytest

from localm import setup_llama as sl


# --------------------------------------------------------------------------- #
# NvidiaInfo.cuda_line: architecture -> asset line, independent of the driver. #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "compute_capability, expected_line",
    [
        # Pre-Blackwell architectures stay on the broad-compatibility line.
        ("5.0", "cuda-12"),    # Maxwell
        ("6.1", "cuda-12"),    # Pascal
        ("7.0", "cuda-12"),    # Volta
        ("7.5", "cuda-12"),    # Turing (CUDA 13's own minimum supported arch)
        ("8.6", "cuda-12"),    # Ampere (consumer)
        ("8.9", "cuda-12"),    # Ada Lovelace
        ("9.0", "cuda-12"),    # Hopper
        # Blackwell and newer need the 13.x line - both the datacenter (sm_100)
        # and consumer/workstation (sm_120) variants.
        ("10.0", "cuda-13"),   # Blackwell datacenter (B100/B200/GB100)
        ("12.0", "cuda-13"),   # Blackwell consumer/workstation (RTX 5090..5060)
        # A hypothetical later architecture also lands on the newer line.
        ("13.0", "cuda-13"),
    ],
)
def test_cuda_line_selected_from_architecture(compute_capability, expected_line):
    info = sl.NvidiaInfo(present=True, compute_capability=compute_capability)
    assert info.cuda_line == expected_line


@pytest.mark.parametrize(
    "compute_capability, expected_line",
    [
        # A BARE major version (no minor component). _ver_tuple parses "10" to
        # the 1-element tuple (10,), and plain tuple comparison treats a shorter
        # tuple that is an exact prefix as SMALLER: (10,) >= (10, 0) is False.
        # _ver_at_least pads before comparing.
        ("10", "cuda-13"),
        ("13", "cuda-13"),   # major alone already exceeds the threshold
        ("9", "cuda-12"),    # bare pre-Blackwell major stays on cuda-12
    ],
)
def test_cuda_line_bare_major_version_boundary(compute_capability, expected_line):
    info = sl.NvidiaInfo(present=True, compute_capability=compute_capability)
    assert info.cuda_line == expected_line


@pytest.mark.parametrize(
    "compute_capability",
    [
        "",                 # absent - preflight never populated it
        "N/A",              # nvidia-smi's own "not applicable" placeholder
        "not-a-number",     # genuinely unparseable garbage
        "unknown",
    ],
)
def test_unknown_or_unparseable_capability_degrades_to_safe_line(compute_capability):
    """An unmeasurable architecture must NOT be treated as evidence it needs
    the newer, narrower-compatibility line - it stays on cuda-12, the same
    "unknown != too old" treatment driver_ok gives the driver version."""
    info = sl.NvidiaInfo(present=True, compute_capability=compute_capability)
    assert info.cuda_line == "cuda-12"


def test_no_gpu_present_defaults_to_safe_line():
    info = sl.NvidiaInfo(present=False)
    assert info.cuda_line == "cuda-12"


# --------------------------------------------------------------------------- #
# driver_ok is now PER-LINE: the same driver can be "ok" for cuda-12 and      #
# "not ok" for cuda-13 - the critical safety property for Blackwell cards.    #
# --------------------------------------------------------------------------- #

def test_blackwell_card_with_only_12x_driver_is_not_driver_ok():
    """The exact failure this feature must prevent: a Blackwell GPU (needs
    cuda-13) whose driver only supports the 12.x line must be flagged as
    driver-not-ok for the line it actually needs, NOT waved through because
    12.4 happens to clear the (wrong) cuda-12 threshold."""
    info = sl.NvidiaInfo(present=True, gpu_name="RTX 5090", driver_version="552.22",
                         cuda_capability="12.4", compute_capability="12.0")
    assert info.cuda_line == "cuda-13"
    assert info.driver_ok is False


def test_blackwell_card_with_13x_driver_is_driver_ok():
    info = sl.NvidiaInfo(present=True, gpu_name="RTX 5090", driver_version="571.96",
                         cuda_capability="13.3", compute_capability="12.0")
    assert info.cuda_line == "cuda-13"
    assert info.driver_ok is True


def test_blackwell_card_with_13x_driver_below_pinned_patch_is_not_yet_ok():
    """The 13.3 minimum follows the 12.4 convention: match the PINNED asset's
    own version, not just its major. A driver reporting 13.0 is therefore
    treated as not-yet-ok and falls back to Vulkan."""
    info = sl.NvidiaInfo(present=True, compute_capability="12.0", cuda_capability="13.0")
    assert info.driver_ok is False


def test_driver_ok_bare_major_driver_version_padded_not_prefix_compared():
    """The same bare-major-version boundary as cuda_line's, but for the
    driver side: _ver_tuple treats a bare major as its own short tuple ("13"
    -> (13,)), and the comparison against a (major, minor) threshold pads
    rather than letting Python's tuple ordering treat the shorter tuple as
    smaller regardless of value. Padding with a trailing 0 reads a bare "13"
    as the earliest possible 13.x, so it fails a minimum that needs a specific
    minor (13.3), while a bare major numerically higher than the whole
    threshold still passes."""
    # Bare "13" against the cuda-13 line's (13,3) minimum: 13.0 does not
    # clear 13.3 - correctly not-ok, not waved through by a padding bug.
    assert sl.NvidiaInfo(present=True, compute_capability="12.0",
                         cuda_capability="13").driver_ok is False
    # No compute_capability set -> cuda-12 line, (12,4) minimum: bare "12"
    # (== "12.0") does not clear 12.4 - correctly not-ok.
    assert sl.NvidiaInfo(present=True, cuda_capability="12").driver_ok is False
    # Bare "13" against the cuda-12 line's (12,4) minimum: 13 > 12 on the
    # major alone, so it clears regardless of the missing minor.
    assert sl.NvidiaInfo(present=True, cuda_capability="13").driver_ok is True


def test_pre_blackwell_card_driver_thresholds_unchanged():
    """Sanity: the ORIGINAL cuda-12 threshold (12.4) must be completely
    unaffected by adding the per-line dict - these are the pre-existing
    driver_ok assertions, still true after the refactor."""
    assert sl.NvidiaInfo(present=True, cuda_capability="11.2").driver_ok is False
    assert sl.NvidiaInfo(present=True, cuda_capability="12.4").driver_ok is True
    assert sl.NvidiaInfo(present=True, cuda_capability="13.3").driver_ok is True


def test_unknown_driver_capability_never_blocks_even_on_blackwell():
    info = sl.NvidiaInfo(present=True, compute_capability="12.0", cuda_capability="")
    assert info.driver_ok is True


# --------------------------------------------------------------------------- #
# End-to-end at the dialogue layer: a 12.x-only driver is never offered a     #
# 13.x asset - it falls back to vulkan.                                       #
# --------------------------------------------------------------------------- #

def test_dialogue_blackwell_old_driver_falls_back_to_vulkan_not_cuda13(monkeypatch):
    # No confirm should be reachable - an old driver cannot be self-assembled.
    monkeypatch.setattr(sl.click, "confirm",
                        lambda *a, **k: pytest.fail("must not prompt - too-old-driver is automatic"))
    info = sl.NvidiaInfo(present=True, gpu_name="RTX 5090", driver_version="552.22",
                         cuda_capability="12.4", compute_capability="12.0")
    assert sl._cuda_setup_dialogue(info, assume_yes=False) == ("vulkan", False)


def test_dialogue_blackwell_new_driver_offers_cuda(monkeypatch):
    info = sl.NvidiaInfo(present=True, gpu_name="RTX 5090", driver_version="571.96",
                         cuda_capability="13.3", compute_capability="12.0")
    assert sl._cuda_setup_dialogue(info, assume_yes=True) == ("cuda", True)


def test_dialogue_prints_detected_architecture_and_line(monkeypatch, capsys):
    info = sl.NvidiaInfo(present=True, gpu_name="RTX 5090", driver_version="571.96",
                         cuda_capability="13.3", compute_capability="12.0")
    sl._cuda_setup_dialogue(info, assume_yes=True)
    out = capsys.readouterr().out
    assert "12.0" in out
    assert "cuda-13" in out


# --------------------------------------------------------------------------- #
# _ASSET_MATCH: the win32 'cuda' entry is now keyed by line, not a flat list. #
# --------------------------------------------------------------------------- #

def test_asset_match_cuda_entry_is_keyed_by_line():
    entry = sl._ASSET_MATCH["win32"]["cuda"]
    assert isinstance(entry, dict)
    assert set(entry) == {"cuda-12", "cuda-13"}
    assert any("12" in m for m in entry["cuda-12"])
    assert any("13" in m for m in entry["cuda-13"])
    # Every other backend/platform is untouched by this restructuring.
    assert isinstance(sl._ASSET_MATCH["win32"]["vulkan"], list)
    assert isinstance(sl._ASSET_MATCH["linux"]["cuda"], list)


# --------------------------------------------------------------------------- #
# _resolve_cuda_pair / _resolve_backend_asset actually consume the line.      #
# --------------------------------------------------------------------------- #

# A release listing carrying BOTH lines, as the real b9870 release does (see
# _PINNED_FALLBACK_SHA256), so the matcher must pick the line-appropriate asset
# out of a mixed listing.
_BOTH_LINES_ASSETS = [
    {"name": "llama-b9870-bin-win-cuda-12.4-x64.zip",
     "browser_download_url": "https://x/llama-12.4.zip", "size": 50_000_000},
    {"name": "cudart-llama-bin-win-cuda-12.4-x64.zip",
     "browser_download_url": "https://x/cudart-12.4.zip", "size": 400_000_000},
    {"name": "llama-b9870-bin-win-cuda-13.3-x64.zip",
     "browser_download_url": "https://x/llama-13.3.zip", "size": 55_000_000},
    {"name": "cudart-llama-bin-win-cuda-13.3-x64.zip",
     "browser_download_url": "https://x/cudart-13.3.zip", "size": 410_000_000},
]


@pytest.mark.parametrize(
    "line, expected_build_url, expected_cudart_url",
    [
        ("cuda-12", "https://x/llama-12.4.zip", "https://x/cudart-12.4.zip"),
        ("cuda-13", "https://x/llama-13.3.zip", "https://x/cudart-13.3.zip"),
    ],
)
def test_resolve_cuda_pair_picks_the_requested_line_from_mixed_listing(
        monkeypatch, line, expected_build_url, expected_cudart_url):
    monkeypatch.setattr(sl, "_release_assets", lambda tag: list(_BOTH_LINES_ASSETS))
    build, cudart = sl._resolve_cuda_pair("b9870", line)
    assert build["browser_download_url"] == expected_build_url
    assert "cudart" not in build["name"]
    assert cudart["browser_download_url"] == expected_cudart_url


def test_resolve_cuda_pair_default_line_is_unchanged():
    """Omitting *line* must still resolve cuda-12 - the pre-existing default
    behavior, unaffected by adding the parameter."""
    assert sl._resolve_cuda_pair.__defaults__ == (sl._CUDA_LINE,)
    assert sl._CUDA_LINE == "cuda-12"


@pytest.mark.parametrize("line", ["cuda-12", "cuda-13"])
def test_resolve_backend_asset_cuda_line_selects_matching_asset(monkeypatch, line):
    monkeypatch.setattr(sl, "_platform_key", lambda: "win32")
    monkeypatch.setattr(sl, "_latest_tag", lambda: "b9870")
    monkeypatch.setattr(sl, "_release_assets", lambda tag: list(_BOTH_LINES_ASSETS))
    url, _sha, _tag = sl._resolve_backend_asset("cuda", cuda_line=line)
    assert "cudart" not in url
    if line == "cuda-12":
        assert "12.4" in url
    else:
        assert "13.3" in url


# --------------------------------------------------------------------------- #
# BOTH lines have a pinned fallback asset in the REAL                        #
# _PINNED_FALLBACK_SHA256 table, not a test fixture.                         #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("line, expected_fragment", [("cuda-12", "12.4"), ("cuda-13", "13.3")])
def test_pinned_fallback_offline_url_names_a_real_checksum_table_entry(monkeypatch, line, expected_fragment):
    """With the release API entirely unreachable (the templated-URL fallback
    inside _resolve_backend_asset), the guessed filename for EITHER line must
    still be a REAL key in the pinned checksum table.

    Uses _PINNED_TAG, the tag a default install actually resolves; no
    _latest_tag monkeypatch is needed because that path makes no network call."""
    monkeypatch.setattr(sl, "_platform_key", lambda: "win32")
    monkeypatch.setattr(sl, "_release_assets", lambda tag: [])
    url, sha, _tag = sl._resolve_backend_asset("cuda", cuda_line=line)
    assert sl._PINNED_TAG in url
    assert expected_fragment in url
    fname = url.rsplit("/", 1)[-1]
    assert fname in sl._PINNED_FALLBACK_SHA256, (
        f"{fname!r} (line={line}) must be a real pinned checksum-table entry")
    assert sha == sl._PINNED_FALLBACK_SHA256[fname]


def test_both_pinned_cuda_lines_share_the_same_upstream_tag():
    """Confirms the exact premise of this fix: both build lines - and both
    matching cudart bundles - are already pinned for the SAME upstream tag,
    so this was always a selection problem, never a missing-asset one."""
    tag = sl._PINNED_TAG
    for line, ver in (("cuda-12", "12.4"), ("cuda-13", "13.3")):
        build_name = f"llama-{tag}-bin-win-cuda-{ver}-x64.zip"
        cudart_name = f"cudart-llama-bin-win-cuda-{ver}-x64.zip"
        assert build_name in sl._PINNED_FALLBACK_SHA256, build_name
        assert cudart_name in sl._PINNED_FALLBACK_SHA256, cudart_name


# --------------------------------------------------------------------------- #
# Full matrix: card architecture x driver capability -> a build+cudart pair  #
# that always exists in the checksum table, and a too-old driver never gets  #
# handed the newer line's asset.                                             #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "gpu_name, compute_capability, cuda_capability, expect_line, expect_driver_ok",
    [
        ("RTX 3080", "8.6", "12.4", "cuda-12", True),
        ("RTX 3080", "8.6", "11.2", "cuda-12", False),
        ("RTX 4090", "8.9", "12.4", "cuda-12", True),
        ("H100", "9.0", "12.4", "cuda-12", True),
        ("B200", "10.0", "13.3", "cuda-13", True),
        ("B200", "10.0", "12.4", "cuda-13", False),      # arch needs 13.x, driver can't
        ("RTX 5090", "12.0", "13.3", "cuda-13", True),
        ("RTX 5090", "12.0", "12.4", "cuda-13", False),   # THE reported bug, exactly
        ("unknown NVIDIA GPU", "", "12.4", "cuda-12", True),
        ("unknown NVIDIA GPU", "N/A", "", "cuda-12", True),
    ],
)
def test_full_matrix_resolves_a_real_pair_and_never_gives_old_driver_new_line(
        monkeypatch, gpu_name, compute_capability, cuda_capability, expect_line, expect_driver_ok):
    info = sl.NvidiaInfo(present=True, gpu_name=gpu_name, driver_version="000.00",
                         cuda_capability=cuda_capability, compute_capability=compute_capability)
    assert info.cuda_line == expect_line
    assert info.driver_ok is expect_driver_ok

    # Whatever line the architecture needs, a build+cudart pair for it exists in
    # the REAL pinned checksum table for the PINNED tag.
    monkeypatch.setattr(sl, "_platform_key", lambda: "win32")
    monkeypatch.setattr(sl, "_release_assets", lambda t: [])
    url, sha, _tag = sl._resolve_backend_asset("cuda", cuda_line=info.cuda_line)
    fname = url.rsplit("/", 1)[-1]
    assert fname in sl._PINNED_FALLBACK_SHA256
    assert sha is not None

    if not expect_driver_ok:
        # The dialogue never hands this driver the too-new line's asset: it falls
        # back to vulkan regardless of why driver_ok is False.
        monkeypatch.setattr(sl.click, "confirm",
                            lambda *a, **k: pytest.fail("must not prompt when the driver is not ok"))
        assert sl._cuda_setup_dialogue(info, assume_yes=False) == ("vulkan", False)


# --------------------------------------------------------------------------- #
# main() end-to-end: a detected Blackwell GPU reaches _provision_backend with #
# cuda_line == "cuda-13", not the module default. The CliRunner-level tests   #
# of main()'s cuda dialogue wiring in test_cuda_setup.py use                  #
# NvidiaInfo(present=False), whose cuda_line is "cuda-12" anyway, so they     #
# cannot tell intact wiring from the hardcoded default.                       #
# --------------------------------------------------------------------------- #

def test_main_threads_blackwell_arch_into_cuda13_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(sl.sys, "platform", "win32")
    monkeypatch.setattr(sl, "nvidia_preflight", lambda: sl.NvidiaInfo(
        present=True, gpu_name="RTX 5090", driver_version="571.96",
        cuda_capability="13.3", compute_capability="12.0"))

    target = tmp_path / "lib"
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: target)
    seen = []

    def fake_provision_backend(backend, tgt, sha256, with_cudart, cuda_line=sl._CUDA_LINE, tag=None):
        seen.append((backend, with_cudart, cuda_line))
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
    assert seen == [("cuda", True, "cuda-13")], (
        "main() must thread the GPU-detected cuda_line (not the module default) "
        "all the way to _provision_backend")


def test_main_threads_pre_blackwell_arch_into_cuda12_fetch(monkeypatch, tmp_path):
    """The mirror case: an older NVIDIA card must still resolve to cuda-12
    through the SAME main()-level wiring - proves this is genuinely
    architecture-driven, not a hardcoded Blackwell special case."""
    monkeypatch.setattr(sl.sys, "platform", "win32")
    monkeypatch.setattr(sl, "nvidia_preflight", lambda: sl.NvidiaInfo(
        present=True, gpu_name="RTX 4090", driver_version="552.22",
        cuda_capability="12.4", compute_capability="8.9"))

    target = tmp_path / "lib"
    monkeypatch.setattr(sl, "_repo_runtime_lib", lambda: target)
    seen = []

    def fake_provision_backend(backend, tgt, sha256, with_cudart, cuda_line=sl._CUDA_LINE, tag=None):
        seen.append((backend, with_cudart, cuda_line))
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
    assert seen == [("cuda", True, "cuda-12")]


# --------------------------------------------------------------------------- #
# nvidia_preflight() against an older-driver error response: older nvidia-smi #
# builds reject --query-gpu=compute_cap with an error sentence rather than a  #
# version.                                                                    #
# --------------------------------------------------------------------------- #

def test_nvidia_preflight_handles_unsupported_compute_cap_field(monkeypatch):
    def fake_smi(*args):
        joined = " ".join(str(a) for a in args)
        if "compute_cap" in joined:
            return ('"compute_cap" is not a valid field to query, '
                    "--help-query-gpu for available fields.\n")
        if "--query-gpu=name" in joined:
            return "NVIDIA GeForce RTX 3080\n"
        return ("|  NVIDIA-SMI 470.182.03   Driver Version: 470.182.03   "
                "CUDA Version: 11.4  |\n")
    monkeypatch.setattr(sl, "_nvidia_smi", fake_smi)
    info = sl.nvidia_preflight()
    assert info.present
    # The raw error line ends up in compute_capability (nothing validates the
    # field); it degrades through _ver_tuple's broad except rather than crashing
    # or being mistaken for a Blackwell architecture.
    assert info.cuda_line == "cuda-12"
    assert info.driver_ok is False   # 11.4 genuinely too old for cuda-12's 12.4 minimum
