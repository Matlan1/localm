# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full round trip for the ComfyUI missing-model auto-download feature:
preflight detects a missing curated file -> resolve its HF source and ComfyUI
destination -> download it (HF network layer mocked) -> the file lands exactly
where ComfyUI would look for it -> re-running preflight now passes."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import localm.config as cfg
from localm import model_manager as mm
from localm.media import comfy_client
from localm.media import managed_comfy as mc
from localm.model_manager.registry import resolve_comfy_model_source


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    h.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    monkeypatch.setattr(mm, "MODELS_DIR", h / "models")
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    return h


def _object_info(unet_options):
    return {
        "UnetLoaderGGUF": {"input": {"required": {
            "unet_name": [list(unet_options), {}],
        }}},
    }


def _workflow():
    return {
        "1": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": "flux1-dev-Q8_0.gguf"}},
    }


def test_missing_model_detected_downloaded_and_preflight_now_passes(
        home, monkeypatch):
    workflow = _workflow()
    api_url = "http://127.0.0.1:8188"

    # 1. Preflight detects the file is missing.
    info_before = _object_info(["some-other-model.gguf"])
    with monkeypatch.context() as m:
        m.setattr(comfy_client, "comfy_object_info", lambda *a, **k: info_before)
        missing = comfy_client.describe_missing_models(workflow, api_url)
    assert len(missing) == 1
    slot = missing[0]
    assert slot.filename == "flux1-dev-Q8_0.gguf"

    # 2. Resolve it against the curated table.
    source = resolve_comfy_model_source(slot.filename)
    assert source is not None
    assert source.spec == "city96/FLUX.1-dev-gguf:flux1-dev-Q8_0.gguf"

    # 3. Resolve the real destination (external ComfyUI with a configured
    # workdir, mirroring what the GUI route does server-side).
    comfy_workdir = home.parent / "external-comfy"
    dest_dir = mc.comfy_models_dest_dir(
        source.comfy_subfolder,
        {"managed_comfy_enabled": False, "comfy_workdir": str(comfy_workdir)})
    assert dest_dir == comfy_workdir / "models" / "unet"

    # 4. Download it (HF network layer mocked), routed to dest_dir and not
    # registered in localm's own catalog.
    fake_bytes = b"pretend-this-is-a-real-gguf"

    def _fake_download(repo_id, filename, local_dir, **kw):
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(fake_bytes)
        return str(p)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)

    def _fake_head(url, allow_redirects=None, timeout=None):
        h = MagicMock()
        h.headers = {"content-length": str(len(fake_bytes))}
        return h

    monkeypatch.setattr("requests.head", _fake_head)

    ok = mm.pull_model(source.spec, model_type=source.model_type,
                       dest_dir=dest_dir, register=False)
    assert ok is True
    downloaded = dest_dir / "flux1-dev-Q8_0.gguf"
    assert downloaded.is_file()
    assert downloaded.read_bytes() == fake_bytes

    # 5. Re-running preflight against a /object_info that NOW includes the
    # filename (as it would after ComfyUI rescans its models folder) passes.
    info_after = _object_info(["flux1-dev-Q8_0.gguf"])
    with monkeypatch.context() as m:
        m.setattr(comfy_client, "comfy_object_info", lambda *a, **k: info_after)
        ok2, msg2 = comfy_client.preflight_models(workflow, api_url)
    assert ok2 is True
    assert msg2 == ""


def test_managed_destination_variant_of_the_same_round_trip(home, monkeypatch):
    """Same round trip, but with the MANAGED ComfyUI active instead of an
    external workdir - the file must land under <LOCALM_HOME>/comfyui-models,
    not the external path."""
    # is_managed_comfy_installed() also requires the provisioning completion
    # marker, not just main.py + venv.
    from localm.media.managed_comfy_provision import MARKER_FILENAME
    paths = mc.managed_comfy_paths()
    paths.main_py.parent.mkdir(parents=True, exist_ok=True)
    paths.main_py.write_text("# stand-in\n", encoding="utf-8")
    paths.venv_python.parent.mkdir(parents=True, exist_ok=True)
    paths.venv_python.write_text("", encoding="utf-8")
    (paths.root / MARKER_FILENAME).write_text("{}", encoding="utf-8")
    assert mc.is_managed_comfy_installed()

    source = resolve_comfy_model_source("clip_l.safetensors")
    dest_dir = mc.comfy_models_dest_dir(
        source.comfy_subfolder,
        {"managed_comfy_enabled": True, "comfy_target": "own"})
    assert dest_dir == home / "comfyui-models" / "clip"

    def _fake_download(repo_id, filename, local_dir, **kw):
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return str(p)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
    monkeypatch.setattr("requests.head",
                        lambda *a, **k: MagicMock(headers={"content-length": "1"}))

    ok = mm.pull_model(source.spec, model_type=source.model_type,
                       dest_dir=dest_dir, register=False)
    assert ok is True
    assert (home / "comfyui-models" / "clip" / "clip_l.safetensors").is_file()
