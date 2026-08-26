# SPDX-License-Identifier: AGPL-3.0-or-later
"""STAGE S3 (FRESH hardware-matched install) for the localm-managed ComfyUI feature.

When a user has NO usable ComfyUI to copy, `localm comfy setup` installs a
FRESH, hardware-matched ComfyUI under LOCALM_HOME:

  * torch per hardware: ONE map, comfy_torch_spec(det), reading hwdetect. AMD
    gfx1030 -> the ROCm gfx103X wheels localm itself runs on; NVIDIA -> CUDA;
    Intel -> XPU; no GPU -> CPU. ComfyUI ALWAYS needs torch, so a hardware combo
    with no verified GPU wheel degrades to CPU torch, never to nothing.
  * custom nodes: derived by scanning localm's OWN shipped workflow JSONs for
    non-core class_types, each mapped to a PINNED repo+commit. Today exactly
    {UnetLoaderGGUFAdvanced -> city96/ComfyUI-GGUF}.

The heavy end-to-end test exercises the FRESH mechanism against a MINIMAL fake:
a tiny throwaway git "ComfyUI" (stub main.py plus a requirements.txt holding one
tiny `name @ file://` wheel, NO torch) and a tiny git "custom node", with the
torch step skipped. It covers clone-at-pin, a fresh venv, pip -r requirements
and the custom-node clone-at-pin, offline. The multi-GB hardware-matched torch
install is a manual/integration check; the torch-map unit test covers that map.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

import localm.config as cfg
from localm.media import managed_comfy as mc


# --------------------------------------------------------------------------- #
#  Throwaway LOCALM_HOME (same wiring as the S1/S2 `home` fixture).            #
# --------------------------------------------------------------------------- #
@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    h.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    return h


def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, timeout=10)
        return True
    except Exception:
        return False


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _build_tiny_wheel(dest_dir: Path, name: str, version: str) -> Path:
    """A minimal dependency-free py3-none-any wheel pip can install offline."""
    dist_info = f"{name}-{version}.dist-info"
    files = {
        f"{name}/__init__.py": f'__version__ = "{version}"\n',
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\n"
            f"Name: {name}\nVersion: {version}\n"
            "Summary: localm S3 test fixture package\n"),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: localm-s3-test\n"
            "Root-Is-Purelib: true\nTag: py3-none-any\n"),
    }
    record = [f"{p},," for p in files] + [f"{dist_info}/RECORD,,"]
    files[f"{dist_info}/RECORD"] = "\n".join(record) + "\n"
    whl = dest_dir / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(whl, "w", zipfile.ZIP_DEFLATED) as z:
        for arcname, content in files.items():
            z.writestr(arcname, content)
    return whl


def _make_git_repo(workdir: Path, files: dict) -> str:
    """Write *files* into *workdir*, git-init + commit, return the HEAD sha. Uses a
    hermetic git config (no user global/system config, gpg off)."""
    workdir.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
    ident = ["-c", "user.email=s3@localhost", "-c", "user.name=s3-test",
             "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main"]

    def _git(*args):
        subprocess.run(["git", *ident, *args], cwd=str(workdir), env=env,
                       check=True, capture_output=True, text=True)

    _git("init")
    _git("add", "-A")
    _git("commit", "-m", "init")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(workdir),
                          capture_output=True, text=True, check=True).stdout.strip()


def _freeze(venv_python: Path) -> list:
    out = subprocess.run([str(venv_python), "-m", "pip", "freeze"],
                         capture_output=True, text=True, check=True).stdout
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


# --------------------------------------------------------------------------- #
#  Fake stubs for hardware detection.                                          #
# --------------------------------------------------------------------------- #
def _det(vendors, gpu_names=""):
    """A hwdetect.Detection-shaped object for comfy_torch_spec() to consume."""
    from localm import hwdetect
    return hwdetect.Detection(vendors=list(vendors), gpu_names=gpu_names.lower())


# =========================================================================== #
#  decision 4: hardware -> ComfyUI torch spec (ONE testable map).             #
# =========================================================================== #

def test_torch_spec_amd_gfx1030_is_rocm_wheels(monkeypatch):
    """AMD RX 6000 (gfx103X) on Windows -> the SAME ROCm torch wheels and AMD
    index localm itself runs on (pyproject [gpu]), never a plain PyPI torch."""
    from localm.media import managed_comfy_fresh as fresh
    monkeypatch.setattr(sys, "platform", "win32")
    spec = fresh.comfy_torch_spec(_det(["amd"], "AMD Radeon RX 6800"))
    assert spec.variant == "rocm"
    assert "torch==2.9.1+rocm7.13.0" in spec.packages
    assert "torchvision==0.24.0+rocm7.13.0" in spec.packages
    assert "rocm-sdk-libraries-gfx103x-all" in spec.packages
    assert spec.extra_index_url == "https://repo.amd.com/rocm/whl/gfx103X-all/"


def test_torch_spec_amd_gfx1030_pins_a_matching_torchaudio(monkeypatch):
    """ComfyUI's own requirements.txt lists torch/torchvision/torchaudio
    UNPINNED, and the later "install ComfyUI's requirements" step runs plain
    `pip install -r requirements.txt` with no --extra-index-url. torch and
    torchvision are already satisfied by the ROCm wheels this spec installs
    first, and pinning a matching ROCm torchaudio build (same repo, same "2.9"
    series and rocm7.13.0 tag as the torch pin) makes that bare requirement
    already-satisfied too - otherwise it resolves to a plain-PyPI build whose
    compiled C extension does not match this ROCm torch's ABI."""
    from localm.media import managed_comfy_fresh as fresh
    monkeypatch.setattr(sys, "platform", "win32")
    spec = fresh.comfy_torch_spec(_det(["amd"], "AMD Radeon RX 6900 XT"))
    assert "torchaudio==2.9.0+rocm7.13.0" in spec.packages


def test_torch_spec_nvidia_is_cuda(monkeypatch):
    from localm import hwdetect
    from localm.media import managed_comfy_fresh as fresh
    # cuda is platform-agnostic; pin the platform so the test is deterministic.
    monkeypatch.setattr(sys, "platform", "linux")
    # Deterministic regardless of the test-running machine's own hardware.
    monkeypatch.setattr(hwdetect, "_cuda_compute_capabilities", lambda: [])
    spec = fresh.comfy_torch_spec(_det(["nvidia"], "NVIDIA GeForce RTX 4090"))
    assert spec.variant == "cuda"
    assert spec.packages == ("torch", "torchvision")
    assert spec.index_url == "https://download.pytorch.org/whl/cu126"


def test_torch_spec_no_gpu_is_cpu(monkeypatch):
    """No GPU -> CPU torch (ComfyUI needs torch to run at all; never 'skip torch')."""
    from localm.media import managed_comfy_fresh as fresh
    monkeypatch.setattr(sys, "platform", "linux")
    spec = fresh.comfy_torch_spec(_det([], ""))
    assert spec.variant == "cpu"
    assert spec.index_url == "https://download.pytorch.org/whl/cpu"


def test_torch_spec_intel_is_xpu(monkeypatch):
    from localm.media import managed_comfy_fresh as fresh
    monkeypatch.setattr(sys, "platform", "linux")
    spec = fresh.comfy_torch_spec(_det(["intel"], "Intel Arc A770"))
    assert spec.variant == "xpu"
    assert spec.index_url.endswith("/xpu")


def test_torch_spec_amd_linux_is_upstream_rocm(monkeypatch):
    """Linux AMD -> upstream ROCm wheels (broad gfx). ComfyUI gets ROCm torch on
    AMD even though the llama runtime uses vulkan there."""
    from localm.media import managed_comfy_fresh as fresh
    monkeypatch.setattr(sys, "platform", "linux")
    spec = fresh.comfy_torch_spec(_det(["amd"], "AMD Radeon RX 6800"))
    assert spec.variant == "rocm"
    assert spec.index_url == "https://download.pytorch.org/whl/rocm6.2"


def test_torch_spec_amd_gfx110x_windows_is_rocm_win(monkeypatch):
    from localm.media import managed_comfy_fresh as fresh
    monkeypatch.setattr(sys, "platform", "win32")
    spec = fresh.comfy_torch_spec(_det(["amd"], "AMD Radeon RX 7900 XTX"))
    assert spec.variant == "rocm"
    assert spec.index_url == "https://download.pytorch.org/whl/rocm6.4"


def test_torch_spec_unknown_amd_windows_degrades_to_cpu_with_note(monkeypatch):
    """AMD on Windows with no verified prebuilt (e.g. RX 5000) -> CPU torch,
    with a note saying so."""
    from localm.media import managed_comfy_fresh as fresh
    monkeypatch.setattr(sys, "platform", "win32")
    spec = fresh.comfy_torch_spec(_det(["amd"], "AMD Radeon RX 5700 XT"))
    assert spec.variant == "cpu"
    assert spec.index_url == "https://download.pytorch.org/whl/cpu"
    assert spec.note and "cpu" in spec.note.lower()


def test_torch_install_args_amd_uses_extra_index(monkeypatch):
    """The pip args builder: AMD ROCm packages go in, with the AMD repo as an
    --extra-index-url so torch resolves there and everything else from PyPI."""
    from localm.media import managed_comfy_fresh as fresh
    monkeypatch.setattr(sys, "platform", "win32")
    spec = fresh.comfy_torch_spec(_det(["amd"], "AMD Radeon RX 6800"))
    args = fresh.comfy_torch_install_args(spec)
    assert "torch==2.9.1+rocm7.13.0" in args
    assert "--extra-index-url" in args
    assert "https://repo.amd.com/rocm/whl/gfx103X-all/" in args


def test_torch_install_args_cuda_uses_index_url(monkeypatch):
    from localm import hwdetect
    from localm.media import managed_comfy_fresh as fresh
    monkeypatch.setattr(sys, "platform", "linux")
    # Deterministic regardless of the test-running machine's own hardware.
    monkeypatch.setattr(hwdetect, "_cuda_compute_capabilities", lambda: [])
    spec = fresh.comfy_torch_spec(_det(["nvidia"], "NVIDIA GeForce RTX 4090"))
    args = fresh.comfy_torch_install_args(spec)
    assert args[:2] == ["torch", "torchvision"]
    assert "--index-url" in args
    assert "https://download.pytorch.org/whl/cu126" in args


def test_torch_install_args_cuda_blackwell_uses_cu130(monkeypatch):
    """ComfyUI's own torch install shares hwdetect.pytorch_index_url with the
    HF backend's, so a Blackwell card resolves to cu130 here too.
    managed_comfy_fresh.comfy_torch_spec is a SEPARATE call site from
    torch_pip_args."""
    from localm import hwdetect
    from localm.media import managed_comfy_fresh as fresh
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(hwdetect, "_cuda_compute_capabilities", lambda: [(12, 0)])
    spec = fresh.comfy_torch_spec(_det(["nvidia"], "NVIDIA RTX PRO 4000 Blackwell"))
    args = fresh.comfy_torch_install_args(spec)
    assert "https://download.pytorch.org/whl/cu130" in args


def test_hwdetect_pytorch_index_url_public_accessor(monkeypatch):
    """The shared index table is reachable via a public hwdetect accessor (single
    source of truth), including the newly-added cpu key."""
    from localm import hwdetect
    monkeypatch.setattr(hwdetect, "_cuda_compute_capabilities", lambda: [])
    assert hwdetect.pytorch_index_url("cuda") == "https://download.pytorch.org/whl/cu126"
    assert hwdetect.pytorch_index_url("cpu") == "https://download.pytorch.org/whl/cpu"
    assert hwdetect.pytorch_index_url("nonsense") is None


# =========================================================================== #
#  decision 5: derive required custom nodes from shipped workflows.           #
# =========================================================================== #

def test_required_custom_nodes_from_real_shipped_workflows():
    """Scanning localm's REAL shipped workflows yields EXACTLY the one non-core
    node (city96 GGUF) and classifies everything else as core (nothing unclassified)."""
    from localm.media import managed_comfy_fresh as fresh
    req = fresh.required_custom_nodes()
    assert len(req.nodes) == 1
    node = req.nodes[0]
    assert node.name == "ComfyUI-GGUF"
    assert "city96/ComfyUI-GGUF" in node.repo
    assert node.commit and len(node.commit) >= 7
    # Every other class_type in the shipped workflows is recognised core.
    assert req.unclassified == ()


def test_required_custom_nodes_surfaces_a_new_noncore_node(tmp_path):
    """A workflow using a class_type that is neither known-core nor mapped is
    SURFACED as unclassified, never silently omitted."""
    from localm.media import managed_comfy_fresh as fresh
    wf = tmp_path / "fake_workflow.json"
    wf.write_text(
        '{"1": {"class_type": "KSampler", "inputs": {}},'
        ' "2": {"class_type": "SomeBrandNewNode", "inputs": {}}}',
        encoding="utf-8")
    req = fresh.required_custom_nodes(workflow_files=[wf])
    assert "SomeBrandNewNode" in req.unclassified
    assert req.nodes == ()   # nothing installable for it (no pin yet)


def test_required_custom_nodes_maps_gguf_in_a_fake_workflow(tmp_path):
    """A fake workflow that uses the GGUF loader resolves to the pinned city96 repo."""
    from localm.media import managed_comfy_fresh as fresh
    wf = tmp_path / "fake_flux.json"
    wf.write_text('{"9": {"class_type": "UnetLoaderGGUFAdvanced", "inputs": {}}}',
                  encoding="utf-8")
    req = fresh.required_custom_nodes(workflow_files=[wf])
    assert len(req.nodes) == 1 and req.nodes[0].name == "ComfyUI-GGUF"
    assert req.unclassified == ()


def test_shipped_workflow_files_ignores_personal_overrides(tmp_path):
    """The shipped-workflow scan reads the COMMITTED templates only, never the
    gitignored personal overrides (flux_workflow.json / *_local.json), so the
    derivation is deterministic."""
    from localm.media import managed_comfy_fresh as fresh
    pkg = tmp_path / "localm"
    (pkg / "image_gen").mkdir(parents=True)
    (pkg / "music_gen").mkdir(parents=True)
    (pkg / "video_gen").mkdir(parents=True)
    # image: shipped example + a personal override that must be excluded.
    (pkg / "image_gen" / "flux_workflow.example.json").write_text("{}", encoding="utf-8")
    (pkg / "image_gen" / "flux_workflow.json").write_text("{}", encoding="utf-8")
    # music/video: shipped template + a *_local.json personal override.
    (pkg / "music_gen" / "ace_workflow.json").write_text("{}", encoding="utf-8")
    (pkg / "music_gen" / "ace_workflow_local.json").write_text("{}", encoding="utf-8")
    (pkg / "video_gen" / "wan_workflow.json").write_text("{}", encoding="utf-8")
    (pkg / "video_gen" / "wan_workflow_local.json").write_text("{}", encoding="utf-8")

    names = {p.name for p in fresh._shipped_workflow_files(pkg_dir=pkg)}
    assert names == {"flux_workflow.example.json", "ace_workflow.json",
                     "wan_workflow.json"}


# =========================================================================== #
#  The FRESH provisioning path, end to end and FOR REAL (core oracle).        #
# =========================================================================== #
@pytest.fixture(scope="session")
def fake_fresh_sources(tmp_path_factory):
    """A minimal fake 'ComfyUI' git repo (stub main.py + requirements.txt holding one
    tiny `name @ file://` wheel) and a tiny fake custom-node git repo. Built once."""
    if not _git_available():
        pytest.skip("git not on PATH (S3 fresh path clones ComfyUI + custom nodes)")
    root = tmp_path_factory.mktemp("fresh_src")

    pkg_name, pkg_version = "localms3pkg", "0.0.1"
    whl = _build_tiny_wheel(root, pkg_name, pkg_version)

    comfy = root / "ComfyUI"
    comfy_commit = _make_git_repo(comfy, {
        "main.py": "# fake ComfyUI entry point\nprint('comfy')\n",
        "requirements.txt": f"{pkg_name} @ {whl.as_uri()}\n",
        ".gitignore": "venv/\n.venv/\nmodels/\ncustom_nodes/\noutput/\n",
    })

    node = root / "FakeGGUFNode"
    node_commit = _make_git_repo(node, {
        "__init__.py": "# fake custom node\nNODE_CLASS_MAPPINGS = {}\n",
    })

    return types.SimpleNamespace(
        comfy=comfy, comfy_commit=comfy_commit, node=node, node_commit=node_commit,
        pkg_name=pkg_name, pkg_version=pkg_version)


def _yaml_base_paths(yaml_path: Path) -> set:
    import yaml
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return {e["base_path"] for e in data.values()
            if isinstance(e, dict) and "base_path" in e}


def test_provision_fresh_end_to_end(home, fake_fresh_sources, monkeypatch):
    """Fresh install against the minimal fake: clone at the pinned commit, a fresh
    localm venv, ComfyUI requirements installed, the pinned custom node cloned, and
    routing now targeting the managed instance. The torch install is SKIPPED (the
    map is unit-tested separately); everything else runs for real and offline."""
    from localm.media import managed_comfy_fresh as fresh
    monkeypatch.setenv("PIP_NO_INDEX", "1")

    node_pin = fresh.CustomNodePin(
        name="FakeGGUFNode", repo=str(fake_fresh_sources.node),
        commit=fake_fresh_sources.node_commit)

    result = fresh.provision_fresh(
        comfyui_repo=str(fake_fresh_sources.comfy),
        comfyui_commit=fake_fresh_sources.comfy_commit,
        custom_nodes=[node_pin],
        install_torch=False)
    assert result.ok, result.message

    paths = mc.managed_comfy_paths()
    # 1) ComfyUI checkout at the pinned commit.
    assert paths.main_py.is_file()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(paths.root),
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head == fake_fresh_sources.comfy_commit
    # 2) A fresh localm venv.
    assert paths.venv_python.is_file()
    # 3) ComfyUI requirements installed (the tiny wheel is present -> pip -r ran).
    assert any(fake_fresh_sources.pkg_name in ln for ln in _freeze(paths.venv_python))
    # 4) The derived custom node cloned at its pin.
    node_dir = paths.root / "custom_nodes" / "FakeGGUFNode"
    assert (node_dir / "__init__.py").is_file()
    node_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(node_dir),
                               capture_output=True, text=True, check=True).stdout.strip()
    assert node_head == fake_fresh_sources.node_commit
    # 5) extra_model_paths.yaml written, pointing at localm's managed models dir.
    assert paths.extra_model_paths.is_file()
    assert str(paths.models_dir) in _yaml_base_paths(paths.extra_model_paths)
    # 6) Provenance marker records a FRESH install.
    marker = paths.root / ".localm-comfy.json"
    assert marker.is_file()
    import json
    assert json.loads(marker.read_text(encoding="utf-8")).get("source") == "fresh"
    # 7) Installed -> S1 routing targets the managed instance (comfy_target=own).
    assert mc.is_managed_comfy_installed() is True
    cfg.save_config({**cfg.load_config(), "comfy_target": "own"})
    target = mc.resolve_comfy_target()
    assert target.managed is True
    assert target.workdir == str(paths.root)


def test_provision_fresh_refuses_when_managed_dir_exists(home, fake_fresh_sources):
    """A pre-existing managed dir is never silently overwritten (matches S2)."""
    from localm.media import managed_comfy_fresh as fresh
    paths = mc.managed_comfy_paths()
    paths.root.mkdir(parents=True)
    (paths.root / "sentinel").write_text("keep me\n", encoding="utf-8")
    result = fresh.provision_fresh(
        comfyui_repo=str(fake_fresh_sources.comfy),
        comfyui_commit=fake_fresh_sources.comfy_commit,
        custom_nodes=[], install_torch=False)
    assert result.ok is False
    assert "exists" in result.message.lower() or "remove" in result.message.lower()
    assert (paths.root / "sentinel").is_file()   # untouched


def test_provision_fresh_rolls_back_on_clone_failure(home):
    """A failed clone leaves NOTHING that reads as installed."""
    from localm.media import managed_comfy_fresh as fresh
    result = fresh.provision_fresh(
        comfyui_repo=str(home / "does-not-exist-repo"),
        comfyui_commit="0" * 40, custom_nodes=[], install_torch=False)
    assert result.ok is False
    assert mc.is_managed_comfy_installed() is False
    assert not mc.managed_comfy_paths().root.exists()


# =========================================================================== #
#  The public orchestrator + CLI dispatch: copy when present, fresh when not. #
# =========================================================================== #

def test_setup_managed_comfy_selects_fresh_when_no_user_comfy(home, monkeypatch):
    """No usable user ComfyUI -> the orchestrator runs the FRESH path (not copy)."""
    from localm.media import managed_comfy_fresh as fresh
    from localm.media import managed_comfy_provision as prov

    monkeypatch.setattr(fresh, "provision_fresh",
                        lambda **kw: prov.ProvisionResult(ok=True, status="fresh",
                                                          message="fresh ok"))
    monkeypatch.setattr(prov, "provision_by_copy",
                        lambda *a, **k: pytest.fail("copy path must not run"))
    res = fresh.setup_managed_comfy()
    assert res.ok and res.status == "fresh"


def test_setup_managed_comfy_selects_copy_when_user_comfy_present(home, monkeypatch,
                                                                 tmp_path):
    """A usable user ComfyUI -> the orchestrator uses the COPY path (S2), not fresh."""
    from localm.media import managed_comfy_fresh as fresh
    from localm.media import managed_comfy_provision as prov

    fake_stack = types.SimpleNamespace(workdir=tmp_path, commit="abc", version_marker="git:abc")
    monkeypatch.setattr(prov, "discover_user_comfy", lambda cfg=None: fake_stack)
    monkeypatch.setattr(prov, "count_user_custom_nodes", lambda wd: 0)
    monkeypatch.setattr(fresh, "provision_fresh",
                        lambda **kw: pytest.fail("fresh path must not run"))
    seen = {}

    def _copy_spy(stack, cfg=None, *, copy_custom_nodes, on_progress=None):
        seen["ran"] = True
        return prov.ProvisionResult(ok=True, status="copied", message="copied ok")

    monkeypatch.setattr(prov, "provision_by_copy", _copy_spy)
    res = fresh.setup_managed_comfy(copy_custom_nodes=False)
    assert res.ok and res.status == "copied"
    assert seen.get("ran") is True


def test_cli_setup_no_user_comfy_runs_fresh(cli_runner, monkeypatch):
    """`localm comfy setup` with no user ComfyUI routes to the fresh install.
    The heavy install is mocked; this asserts the WIRING selects fresh and
    reports its result."""
    from localm.cli import main
    from localm.media import managed_comfy_fresh as fresh
    from localm.media import managed_comfy_provision as prov

    monkeypatch.setattr(fresh, "provision_fresh",
                        lambda **kw: prov.ProvisionResult(
                            ok=True, status="fresh",
                            managed_root=mc.managed_comfy_paths().root,
                            message="localm's managed ComfyUI is ready (fresh)."))
    res = cli_runner.invoke(main, ["comfy", "setup"])
    assert res.exit_code == 0, res.output
    assert "fresh" in res.output.lower() or "ready" in res.output.lower()
