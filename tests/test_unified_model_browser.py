# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for localm model browser unified type support and ComfyUI model scan capabilities.
"""

import json
from unittest.mock import patch
import pytest

from localm import model_manager as mm
from localm.model_manager.scan import (
    get_comfy_api_url, get_comfy_workdir, preview_comfy_models, scan_comfy_models,
)


@pytest.fixture()
def comfy_home(tmp_path, monkeypatch):
    """Isolated LOCALM_HOME for get_comfy_workdir()/get_comfy_api_url(), which
    call load_config() directly - config.py's HOME_DIR/CONFIG_FILE are frozen
    at import time, so the autouse env-var isolation alone does not redirect
    them; patch the module attrs explicitly, matching test_comfy_models_dest_dir.py."""
    import localm.config as cfg
    h = tmp_path / ".localm"
    h.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(h))
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    return h


class TestGetComfyWorkdirManagedAwareness:
    """get_comfy_workdir()/get_comfy_api_url() must be managed-instance aware:
    resolving a per-plugin or global override regardless of whether localm's
    own ComfyUI is the active target makes the GUI's "Scan for ComfyUI models"
    button scan the wrong folder."""

    def _install_managed(self, home_dir):
        from localm.media import managed_comfy as mc
        from localm.media.managed_comfy_provision import MARKER_FILENAME
        paths = mc.managed_comfy_paths()
        paths.main_py.parent.mkdir(parents=True, exist_ok=True)
        paths.main_py.write_text("# stand-in for ComfyUI main.py\n", encoding="utf-8")
        paths.venv_python.parent.mkdir(parents=True, exist_ok=True)
        paths.venv_python.write_text("", encoding="utf-8")
        (paths.root / MARKER_FILENAME).write_text("{}", encoding="utf-8")
        assert mc.is_managed_comfy_installed()
        return paths

    def test_workdir_falls_back_through_plugins_then_global_when_not_managed(self, comfy_home):
        import localm.config as cfg
        cfg.save_config({**cfg.load_config(),
                         "comfy_workdir": r"D:\stale\global",
                         "plugins": {"video": {"comfy": {"workdir": r"D:\deliberate\video-comfy"}}}})
        assert get_comfy_workdir() == r"D:\deliberate\video-comfy"

    def test_workdir_routes_to_managed_instance_ignoring_stale_plugin_value(self, comfy_home):
        """Managed is active AND installed, so a per-plugin workdir left over
        from an unrelated custom install must not win."""
        import localm.config as cfg
        paths = self._install_managed(comfy_home)
        cfg.save_config({**cfg.load_config(), "comfy_target": "own",
                         "plugins": {"image": {"comfy": {"workdir": r"D:\some\other\comfy"}}}})
        assert get_comfy_workdir() == str(paths.root)

    def test_api_url_routes_to_managed_instance_ignoring_stale_plugin_value(self, comfy_home):
        import localm.config as cfg
        from localm.media import managed_comfy as mc
        self._install_managed(comfy_home)
        cfg.save_config({**cfg.load_config(), "comfy_target": "own",
                         "plugins": {"image": {"comfy": {"api_url": "http://127.0.0.1:8188"}}}})
        assert get_comfy_api_url() == mc.MANAGED_COMFY_API_URL

    def test_api_url_honours_plugin_override_when_not_managed(self, comfy_home):
        import localm.config as cfg
        cfg.save_config({**cfg.load_config(),
                         "plugins": {"music": {"comfy": {"api_url": "http://127.0.0.1:8199"}}}})
        assert get_comfy_api_url() == "http://127.0.0.1:8199"


class TestManagedScanFindsRealModels:
    """NEW-MODEL-SCAN-BLIND-TO-MANAGED-MODELS-DIR: get_comfy_workdir() correctly
    reports the managed checkout root (see TestGetComfyWorkdirManagedAwareness
    above - that value is still right for display purposes), but
    scan_comfy_models/preview_comfy_models blindly appended "/models" to
    whatever it returned. A managed ComfyUI's models live in the SIBLING
    comfyui-models dir, never inside a `models` subfolder under the checkout
    (managed_comfy_provision.py's copy step excludes "models"), so a real
    managed install with real downloaded files always previewed/scanned as
    empty - the "Scan for ComfyUI models" button found nothing even with a
    running managed instance serving real generations."""

    def _install_managed(self, home_dir):
        from localm.media import managed_comfy as mc
        from localm.media.managed_comfy_provision import MARKER_FILENAME
        paths = mc.managed_comfy_paths()
        paths.main_py.parent.mkdir(parents=True, exist_ok=True)
        paths.main_py.write_text("# stand-in for ComfyUI main.py\n", encoding="utf-8")
        paths.venv_python.parent.mkdir(parents=True, exist_ok=True)
        paths.venv_python.write_text("", encoding="utf-8")
        (paths.root / MARKER_FILENAME).write_text("{}", encoding="utf-8")
        assert mc.is_managed_comfy_installed()
        return paths

    def test_preview_finds_real_files_in_the_managed_models_dir(self, comfy_home):
        import localm.config as cfg
        paths = self._install_managed(comfy_home)
        cfg.save_config({**cfg.load_config(), "comfy_target": "own"})

        unet_dir = paths.models_dir / "unet"
        unet_dir.mkdir(parents=True)
        (unet_dir / "flux_unet.safetensors").write_bytes(b"UNET")

        with patch("localm.model_manager.scan.comfy_object_info", return_value=None):
            preview = preview_comfy_models(comfy_url="http://localhost:8188")

        assert preview.counts == {"diffusion-unet": 1}
        assert preview.already_registered == 0

    def test_workdir_override_of_the_managed_root_also_finds_it(self, comfy_home):
        """The "Use localm's own ComfyUI" quick-fill in the Import-from-ComfyUI
        modal hands back the managed checkout root as an explicit workdir
        override (models.js fetchManagedComfyPath(), fed by managed-status's
        `path` field) - that must resolve the same way as the no-override
        auto-detect path above, not scan a <root>/models that never exists."""
        import localm.config as cfg
        paths = self._install_managed(comfy_home)
        cfg.save_config({**cfg.load_config(), "comfy_target": "own"})

        lora_dir = paths.models_dir / "loras"
        lora_dir.mkdir(parents=True)
        (lora_dir / "style.safetensors").write_bytes(b"LORA")

        with patch("localm.model_manager.scan.comfy_object_info", return_value=None):
            preview = preview_comfy_models(comfy_url="http://localhost:8188",
                                           workdir=str(paths.root))

        assert preview.counts == {"lora": 1}

    def test_explicit_workdir_for_an_unrelated_folder_still_uses_models_subfolder(
        self, comfy_home, tmp_path
    ):
        """Guard against over-broadening the managed-root special case: an
        ordinary external ComfyUI folder (the common case for this override -
        picked via Browse or typed by hand) must still resolve to
        <workdir>/models exactly as before, even while a managed instance
        happens to be installed and active."""
        self._install_managed(comfy_home)
        import localm.config as cfg
        cfg.save_config({**cfg.load_config(), "comfy_target": "own"})

        external = tmp_path / "external-comfy"
        vae_dir = external / "models" / "vae"
        vae_dir.mkdir(parents=True)
        (vae_dir / "external_vae.safetensors").write_bytes(b"VAE")

        with patch("localm.model_manager.scan.comfy_object_info", return_value=None):
            preview = preview_comfy_models(comfy_url="http://localhost:8188",
                                           workdir=str(external))

        assert preview.counts == {"vae": 1}


@pytest.fixture()
def temp_registry(tmp_path, monkeypatch):
    """Temp registry storage and path monkeypatched."""
    store = {}
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    registry_file = tmp_path / "registry.json"
    
    monkeypatch.setattr(mm, "MODELS_DIR", models_dir)
    monkeypatch.setattr(mm, "REGISTRY_FILE", registry_file)
    monkeypatch.setattr(mm, "ensure_dirs", lambda: None)
    monkeypatch.setattr(mm, "_check_disk_space", lambda *a, **k: True)
    
    # Mock registry load/save
    def _load():
        if registry_file.exists():
            return json.loads(registry_file.read_text())
        return {}
        
    def _save(reg):
        registry_file.write_text(json.dumps(reg))
        
    def _update(mutator):
        reg = _load()
        mutator(reg)
        _save(reg)
        return reg

    monkeypatch.setattr("localm.config.load_registry", _load)
    monkeypatch.setattr("localm.config.save_registry", _save)
    monkeypatch.setattr("localm.config.update_registry", _update)
    monkeypatch.setattr(mm, "load_registry", _load)
    monkeypatch.setattr(mm, "save_registry", _save)
    monkeypatch.setattr(mm, "update_registry", _update)
    # scan.py did `from localm.config import load_registry` at ITS OWN import
    # time, so it holds an independent name bound to the original function and
    # patching localm.config.load_registry (above) does not reach it. Without
    # this, scan.py's dedup check reads the real, session-frozen registry instead
    # of this fixture's store.
    monkeypatch.setattr("localm.model_manager.scan.load_registry", _load)
    return store, models_dir, registry_file


def test_model_type_default(temp_registry):
    """Adding a model with no explicit type defaults to 'llm'."""
    _, models_dir, _ = temp_registry
    fake_file = models_dir / "my_model.gguf"
    fake_file.write_bytes(b"GGUF")

    # Add as local
    mm.add_local(str(fake_file))
    
    # Check registered type
    reg = mm.load_registry()
    assert len(reg) == 1
    assert "my_model" in reg
    assert reg["my_model"]["model_type"] == "llm"


def test_model_type_explicit(temp_registry):
    """Adding a model with an explicit type stores it correctly."""
    _, models_dir, _ = temp_registry
    fake_file = models_dir / "my_vae.gguf"
    fake_file.write_bytes(b"GGUF")

    mm.add_local(str(fake_file), model_type="vae")
    
    reg = mm.load_registry()
    assert len(reg) == 1
    assert "my_vae" in reg
    assert reg["my_vae"]["model_type"] == "vae"


def test_list_models_type_filter(temp_registry):
    """Filtering list_models by type retrieves only matching models."""
    _, models_dir, _ = temp_registry
    
    # Register an LLM
    f1 = models_dir / "model1.gguf"
    f1.write_bytes(b"GGUF 1")
    mm.add_local(str(f1), model_type="llm")

    # Register a VAE
    f2 = models_dir / "model2.gguf"
    f2.write_bytes(b"GGUF 2")
    mm.add_local(str(f2), model_type="vae")

    # Verify filtering behaves as expected via direct loader since list_models is for console print
    reg = mm.load_registry()
    assert len(reg) == 2

    # Verify type field values
    assert reg["model1"]["model_type"] == "llm"
    assert reg["model2"]["model_type"] == "vae"


def test_pull_type_forwarding(temp_registry, monkeypatch):
    """Verify pull_model forwards explicit type to registry."""
    _, models_dir, _ = temp_registry
    
    # Mock HF snapshots / GGUF download to just call register
    monkeypatch.setattr(mm.pull, "_pull_url", lambda url, name, expected_sha256=None, redownload=False, model_type="llm": 
        mm._register(name, models_dir / "url_model.gguf", url, model_type=model_type)
    )

    mm.pull.pull_model("https://example.com/test_model.gguf", name="pulled", model_type="diffusion-unet")
    
    reg = mm.load_registry()
    assert len(reg) == 1
    assert "pulled" in reg
    assert reg["pulled"]["model_type"] == "diffusion-unet"


def test_scan_folder_walk(tmp_path, temp_registry):
    """Walk ComfyUI folders and identify models by mapped folders."""
    _, _, registry_file = temp_registry
    comfy_dir = tmp_path / "comfy"
    
    # Create unet folder and mock model
    unet_dir = comfy_dir / "models" / "unet"
    unet_dir.mkdir(parents=True)
    model_file = unet_dir / "flux_unet.safetensors"
    model_file.write_bytes(b"UNET")

    with patch("localm.model_manager.scan.get_comfy_workdir", return_value=str(comfy_dir)):
        # Query with ComfyUI offline. This code path fetches /object_info via
        # comfy_object_info (urllib), not requests.get, so that is what has to be
        # patched to keep the test off the network.
        with patch("localm.model_manager.scan.comfy_object_info", return_value=None):
            res = scan_comfy_models(comfy_url="http://localhost:8188")

    assert res.added >= 1
    
    # Reload registry
    reg = mm.load_registry()
    assert "flux_unet" in reg
    assert reg["flux_unet"]["model_type"] == "diffusion-unet"


_UNET_OBJECT_INFO = {
    "UNETLoader": {
        "input": {"required": {"unet_name": [["some_model.safetensors"]]}}
    }
}


def test_scan_object_info_reconcile(tmp_path, temp_registry):
    """The /object_info reconcile pass classifies a file that folder-walk cannot.

    NOTE: the previous version put the file in a `checkpoints/` folder, which
    folder-walk already maps via SUBFOLDER_MAPPING, and patched `requests.get` -
    which this code path never calls (it fetches /object_info via urllib). So the
    reconcile was never exercised: the test passed on folder-walk alone. Here the
    file lives in a folder NOT in SUBFOLDER_MAPPING (folder-walk -> "unknown"), so
    only the reconcile can produce "diffusion-unet", and we patch the ACTUAL
    fetcher `comfy_object_info`.
    """
    _, _, registry_file = temp_registry
    comfy_dir = tmp_path / "comfy"
    generic_dir = comfy_dir / "models" / "my_custom_nodes"
    generic_dir.mkdir(parents=True)
    (generic_dir / "some_model.safetensors").write_bytes(b"CKPT")

    with patch("localm.model_manager.scan.get_comfy_workdir", return_value=str(comfy_dir)):
        with patch("localm.model_manager.scan.comfy_object_info",
                   return_value=_UNET_OBJECT_INFO):
            scan_comfy_models(comfy_url="http://localhost:8188")

    reg = mm.load_registry()
    assert "some_model" in reg
    assert reg["some_model"]["model_type"] == "diffusion-unet"


def test_scan_without_object_info_leaves_generic_file_unknown(tmp_path, temp_registry):
    """Guard proving the reconcile is load-bearing: with /object_info unavailable,
    the same generic-folder file stays 'unknown' (folder-walk cannot classify it)."""
    comfy_dir = tmp_path / "comfy"
    generic_dir = comfy_dir / "models" / "my_custom_nodes"
    generic_dir.mkdir(parents=True)
    (generic_dir / "some_model.safetensors").write_bytes(b"CKPT")

    with patch("localm.model_manager.scan.get_comfy_workdir", return_value=str(comfy_dir)):
        with patch("localm.model_manager.scan.comfy_object_info", return_value=None):
            scan_comfy_models(comfy_url="http://localhost:8188")

    reg = mm.load_registry()
    assert reg["some_model"]["model_type"] == "unknown"


# --------------------------------------------------------------------------- #
#  Guided Import-from-ComfyUI: workdir override + dry-run preview             #
# --------------------------------------------------------------------------- #

def _make_comfy_tree(root):
    """A folder tree covering every SUBFOLDER_MAPPING convention plus one
    unmapped folder, one file each - the fixture the workdir-override and
    preview tests scan."""
    layout = {
        "unet": "unet_model.safetensors",
        "clip": "clip_model.safetensors",
        "vae": "vae_model.safetensors",
        "loras": "lora_model.safetensors",
        "my_custom_nodes": "mystery_model.safetensors",
    }
    for sub, fname in layout.items():
        d = root / "models" / sub
        d.mkdir(parents=True)
        (d / fname).write_bytes(b"DATA")
    return layout


def test_scan_workdir_override_never_touches_configured_comfy_workdir(tmp_path, temp_registry):
    """A one-off scan with an explicit workdir must not even READ the configured
    comfy_workdir - `workdir or get_comfy_workdir()` short-circuits, so patching
    get_comfy_workdir to explode proves it is never consulted (a stronger
    guarantee than snapshotting config values, which could pass by coincidence)."""
    comfy_dir = tmp_path / "comfy"
    _make_comfy_tree(comfy_dir)

    def _boom():
        raise AssertionError("get_comfy_workdir() must not be called with an explicit workdir override")

    with patch("localm.model_manager.scan.get_comfy_workdir", side_effect=_boom):
        with patch("localm.model_manager.scan.comfy_object_info", return_value=None):
            res = scan_comfy_models(comfy_url="http://localhost:8188", workdir=str(comfy_dir))

    assert res.added == 5


def test_scan_workdir_override_categorizes_every_convention(tmp_path, temp_registry):
    """An arbitrary one-off folder scans and categorizes exactly like the
    configured-workdir path - unet/clip/vae/loras map correctly, the unmapped
    folder lands as 'unknown'."""
    comfy_dir = tmp_path / "comfy"
    _make_comfy_tree(comfy_dir)

    with patch("localm.model_manager.scan.comfy_object_info", return_value=None):
        res = scan_comfy_models(comfy_url="http://localhost:8188", workdir=str(comfy_dir))

    assert res.added == 5
    assert res.skipped == 0
    reg = mm.load_registry()
    assert reg["unet_model"]["model_type"] == "diffusion-unet"
    assert reg["clip_model"]["model_type"] == "text-encoder"
    assert reg["vae_model"]["model_type"] == "vae"
    assert reg["lora_model"]["model_type"] == "lora"
    assert reg["mystery_model"]["model_type"] == "unknown"


def test_preview_comfy_models_registers_nothing(tmp_path, temp_registry):
    """The dry-run preview must never write to the registry, whatever it finds."""
    comfy_dir = tmp_path / "comfy"
    _make_comfy_tree(comfy_dir)

    with patch("localm.model_manager.scan.comfy_object_info", return_value=None):
        preview = preview_comfy_models(comfy_url="http://localhost:8188", workdir=str(comfy_dir))

    assert mm.load_registry() == {}, "a preview must not register anything"
    assert preview.already_registered == 0
    assert preview.counts == {
        "diffusion-unet": 1, "text-encoder": 1, "vae": 1, "lora": 1, "unknown": 1,
    }


def test_preview_then_scan_totals_agree(tmp_path, temp_registry):
    """The preview's total_new (sum of counts) must match what a REAL scan of the
    same folder actually adds - a stale/mismatched preview would mislead the
    guided-import confirm step."""
    comfy_dir = tmp_path / "comfy"
    _make_comfy_tree(comfy_dir)

    with patch("localm.model_manager.scan.comfy_object_info", return_value=None):
        preview = preview_comfy_models(comfy_url="http://localhost:8188", workdir=str(comfy_dir))
        total_new = sum(preview.counts.values())
        res = scan_comfy_models(comfy_url="http://localhost:8188", workdir=str(comfy_dir))

    assert total_new == res.added == 5


def test_preview_excludes_already_registered_files(tmp_path, temp_registry):
    """A file the registry already knows about must count toward
    already_registered, not toward the per-type counts - re-previewing the same
    folder after a real scan should show nothing new left to import."""
    comfy_dir = tmp_path / "comfy"
    _make_comfy_tree(comfy_dir)

    with patch("localm.model_manager.scan.comfy_object_info", return_value=None):
        scan_comfy_models(comfy_url="http://localhost:8188", workdir=str(comfy_dir))
        preview = preview_comfy_models(comfy_url="http://localhost:8188", workdir=str(comfy_dir))

    assert preview.counts == {}
    assert preview.already_registered == 5


class TestScanProgressCallback:
    """scan_comfy_models's progress_cb: the unit the GUI's job-based real scan
    (models.py's gui_scan_models) wires to Job.progress() for a real
    "registering model N of M" count. Unit-level, no HTTP/job machinery."""

    def test_progress_cb_called_once_per_file_with_a_fixed_total(self, tmp_path, temp_registry):
        comfy_dir = tmp_path / "comfy"
        layout = _make_comfy_tree(comfy_dir)   # 5 files, one per SUBFOLDER_MAPPING convention

        calls = []
        with patch("localm.model_manager.scan.comfy_object_info", return_value=None):
            res = scan_comfy_models(comfy_url="http://localhost:8188", workdir=str(comfy_dir),
                                    progress_cb=lambda done, total, name: calls.append((done, total, name)))

        assert res.added == 5
        assert len(calls) == 5
        # done climbs 1..5 against a FIXED total (never a growing/shrinking one).
        assert [c[0] for c in calls] == [1, 2, 3, 4, 5]
        assert all(c[1] == 5 for c in calls)
        assert {c[2] for c in calls} == set(layout.values())

    def test_progress_cb_still_fires_for_an_already_registered_file(self, tmp_path, temp_registry):
        """A file that turns out to be a skip (already registered) is still a
        real item of work in the loop - the callback must count it too, or a
        rescan with mostly-skips would show a bar that never reaches its total."""
        comfy_dir = tmp_path / "comfy"
        _make_comfy_tree(comfy_dir)
        with patch("localm.model_manager.scan.comfy_object_info", return_value=None):
            scan_comfy_models(comfy_url="http://localhost:8188", workdir=str(comfy_dir))   # registers all 5

            calls = []
            res = scan_comfy_models(comfy_url="http://localhost:8188", workdir=str(comfy_dir),
                                    progress_cb=lambda done, total, name: calls.append((done, total, name)))

        assert res.added == 0 and res.skipped == 5
        assert len(calls) == 5, "every already-registered file still gets a progress tick"
        assert [c[0] for c in calls] == [1, 2, 3, 4, 5]

    def test_progress_cb_default_none_changes_nothing(self, tmp_path, temp_registry):
        """Every existing caller (the CLI has none; only the GUI route calls
        this) omits progress_cb - confirms that path is untouched."""
        comfy_dir = tmp_path / "comfy"
        _make_comfy_tree(comfy_dir)
        with patch("localm.model_manager.scan.comfy_object_info", return_value=None):
            res = scan_comfy_models(comfy_url="http://localhost:8188", workdir=str(comfy_dir))
        assert res.added == 5 and res.skipped == 0

    def test_preview_comfy_models_takes_no_progress_cb(self, tmp_path, temp_registry):
        """Deliberate: preview's directory walk has no honest total to report
        progress against (see the docstring on scan_comfy_models), so
        preview_comfy_models's signature was left unchanged rather than
        growing an unused parameter."""
        import inspect
        assert "progress_cb" not in inspect.signature(preview_comfy_models).parameters


def test_preview_missing_models_folder_reports_reason(tmp_path, temp_registry):
    """No models/ folder under the chosen workdir -> the same honest 'none (...)'
    reason scan_comfy_models gives, not a silent empty result."""
    empty_dir = tmp_path / "not-comfy"
    empty_dir.mkdir()

    preview = preview_comfy_models(workdir=str(empty_dir))

    assert preview.counts == {}
    assert preview.already_registered == 0
    assert preview.method.startswith("none (models folder not found under")
