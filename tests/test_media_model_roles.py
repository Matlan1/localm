# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm/plugins/media_roles.py`` joins ``Host.register_model_role`` (declared
by all three media plugins) to the registry's ``model_type`` slice; these pin that
contract.

The load-bearing properties, in the order they can go wrong:

* ONE node-name -> model_type inference (``comfy_client.model_type_for_node``),
  shared with the ComfyUI folder scanner. Two copies drift, and the picker then
  offers a file the scan filed under a different type.
* ONE "is this slot satisfied" rule (``comfy_client.slot_is_satisfied``), shared
  with ``describe_missing_models``. Two rules and the picker calls a slot fine
  that generation then refuses.
* the "could not ask ComfyUI" / "asked, nothing there" distinction survives every
  layer, instead of both collapsing to an empty list.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.media import comfy_client
from localm.plugins import media_roles


# --------------------------------------------------------------------------- #
#  the shared node-name -> model_type inference
# --------------------------------------------------------------------------- #

class TestModelTypeForNode:
    @pytest.mark.parametrize("class_type,expected", [
        # exactly the loaders the three SHIPPED workflows use, read out of the
        # tracked JSON
        ("UnetLoaderGGUFAdvanced", "diffusion-unet"),   # image / flux
        ("DualCLIPLoader", "text-encoder"),             # image / flux
        ("VAELoader", "vae"),                           # image + video
        ("CheckpointLoaderSimple", "diffusion-unet"),   # music / ACE
        ("UNETLoader", "diffusion-unet"),               # video / wan
        ("CLIPLoader", "text-encoder"),                 # video / wan
        # the rest of the surface
        ("LoraLoader", "lora"),
        ("LoraLoaderModelOnly", "lora"),
        ("CLIPTextEncode", "text-encoder"),
        ("KSampler", "unknown"),
        ("SaveImage", "unknown"),
    ])
    def test_mapping(self, class_type, expected):
        assert comfy_client.model_type_for_node(class_type) == expected

    def test_unclassifiable_is_unknown_not_a_guess(self):
        # "we could not tell" is a real answer and is not dressed up as a type.
        for junk in ("", None, "Reroute", "PrimitiveNode"):
            assert comfy_client.model_type_for_node(junk) == "unknown"

    def test_every_result_is_a_real_registry_type(self):
        from localm.model_manager.registry import MODEL_TYPES
        names = ["UNETLoader", "CLIPLoader", "VAELoader", "LoraLoader",
                 "CheckpointLoaderSimple", "KSampler"]
        for n in names:
            assert comfy_client.model_type_for_node(n) in MODEL_TYPES

    def test_checkpoint_agrees_with_the_scanner_folder_table(self):
        """The folder walk files models/checkpoints/* as 'diffusion-unet'. The
        /object_info pass must agree, or one file gets two types depending on
        whether ComfyUI happened to be up during the scan."""
        from localm.model_manager.scan import SUBFOLDER_MAPPING
        assert (comfy_client.model_type_for_node("CheckpointLoaderSimple")
                == SUBFOLDER_MAPPING["checkpoints"])


class TestShippedWorkflowsClassify:
    """Bound to the REAL tracked workflow files, not a fixture: a fixture can
    only ever contain the node types its author already thought of."""

    ROOT = Path(__file__).resolve().parents[1] / "localm"
    CASES = [
        ("image_gen/flux_workflow.example.json", "diffusion-unet"),
        ("music_gen/ace_workflow.json", "diffusion-unet"),
        ("video_gen/wan_workflow.json", "diffusion-unet"),
    ]

    @pytest.mark.parametrize("rel,needed", CASES, ids=[c[0] for c in CASES])
    def test_each_shipped_workflow_has_a_classifiable_unet_source(self, rel, needed):
        wf = json.loads((self.ROOT / rel).read_text(encoding="utf-8"))
        types = {comfy_client.model_type_for_node(n.get("class_type"))
                 for n in wf.values() if isinstance(n, dict)}
        assert needed in types, (
            f"{rel}: no node classifies as {needed}, so the plugin's own required "
            f"role can never be matched against its own default workflow")


# --------------------------------------------------------------------------- #
#  slot_is_satisfied: one rule, shared with preflight
# --------------------------------------------------------------------------- #

def _object_info():
    return {
        "UNETLoader": {"input": {"required": {
            "unet_name": [["wan_5B_fp16.safetensors"], {}]}}},
        "DualCLIPLoader": {"input": {"required": {
            "clip_name1": [["clip_l.safetensors", "clip_g.safetensors"], {}],
            "clip_name2": [["t5xxl_fp8_e4m3fn.safetensors", "t5xxl_fp16.safetensors"], {}],
        }}},
        "VAELoader": {"input": {"required": {
            "vae_name": [["ae.safetensors"], {}]}}},
    }


def _workflow():
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "wan_5B_fp16.safetensors"}},
        "2": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": "clip_l.safetensors",
                         "clip_name2": "t5xxl_fp8_e4m3fn.safetensors"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": "nowhere.safetensors"}},
    }


def _slots(workflow=None, info=None):
    from unittest.mock import patch
    with patch.object(comfy_client, "comfy_object_info",
                      return_value=info or _object_info()):
        return comfy_client.workflow_model_slots(workflow or _workflow(), "http://x")


class TestSlotIsSatisfied:
    def test_agrees_with_describe_missing_models_on_every_slot(self):
        """The anti-drift check: if these two disagree, the picker promises a
        model generation will refuse."""
        from unittest.mock import patch
        wf = _workflow()
        with patch.object(comfy_client, "comfy_object_info", return_value=_object_info()):
            slots = comfy_client.workflow_model_slots(wf, "http://x")
            missing = comfy_client.describe_missing_models(wf, "http://x")
        missing_keys = {(m.class_type, m.input_name) for m in missing}
        for slot in slots:
            key = (slot["class_type"], slot["input_name"])
            assert comfy_client.slot_is_satisfied(slot) is (key not in missing_keys)
        assert missing_keys, "the fixture produced no missing slot - it proves nothing"

    def test_exact_match_is_satisfied(self):
        assert comfy_client.slot_is_satisfied(
            {"current": "a.safetensors", "options": ["a.safetensors"]}) is True

    def test_unambiguous_precision_variant_is_satisfied(self):
        # preflight substitutes this one in, so the picker must not call it missing
        assert comfy_client.slot_is_satisfied(
            {"current": "wan_5B_fp16.safetensors",
             "options": ["wan_5B_fp8_scaled.safetensors"]}) is True

    def test_absent_is_not_satisfied(self):
        assert comfy_client.slot_is_satisfied(
            {"current": "a.safetensors", "options": ["b.safetensors"]}) is False

    def test_non_string_current_is_unsatisfied_not_an_exception(self):
        assert comfy_client.slot_is_satisfied({"current": None, "options": []}) is False
        assert comfy_client.slot_is_satisfied({}) is False


# --------------------------------------------------------------------------- #
#  role <-> slot pairing and slot annotation
# --------------------------------------------------------------------------- #

def _roles(*specs):
    return [{"role_id": rid, "label": lab, "model_type": mt,
             "plugin_name": "image", "required": req, "description": ""}
            for rid, lab, mt, req in specs]


IMAGE_ROLES = _roles(
    ("image-unet", "Diffusion model (UNet)", "diffusion-unet", True),
    ("image-clip1", "Text encoder 1 (CLIP-L)", "text-encoder", False),
    ("image-clip2", "Text encoder 2 (T5/CLIP-G)", "text-encoder", False),
    ("image-vae", "VAE", "vae", False),
    ("image-lora", "LoRA", "lora", False),
)


class TestAnnotateSlots:
    def test_none_survives_annotation(self):
        """Unreachable ComfyUI must not become an empty list anywhere on the
        path - that is the rule-5 distinction the whole surface rests on."""
        assert media_roles.annotate_slots(None, IMAGE_ROLES) is None

    def test_each_slot_gets_its_model_type(self):
        out = media_roles.annotate_slots(_slots(), IMAGE_ROLES)
        got = {(s["class_type"], s["input_name"]): s["model_type"] for s in out}
        assert got == {
            ("UNETLoader", "unet_name"): "diffusion-unet",
            ("DualCLIPLoader", "clip_name1"): "text-encoder",
            ("DualCLIPLoader", "clip_name2"): "text-encoder",
            ("VAELoader", "vae_name"): "vae",
        }

    def test_two_text_encoder_slots_pair_with_the_two_declared_roles_in_order(self):
        out = media_roles.annotate_slots(_slots(), IMAGE_ROLES)
        clips = [s for s in out if s["model_type"] == "text-encoder"]
        assert [s["role_id"] for s in clips] == ["image-clip1", "image-clip2"]
        assert [s["role_label"] for s in clips] == [
            "Text encoder 1 (CLIP-L)", "Text encoder 2 (T5/CLIP-G)"]

    def test_a_slot_with_no_declared_role_is_unpaired_not_mislabelled(self):
        # only a unet role declared: the clip/vae slots must NOT borrow it
        out = media_roles.annotate_slots(
            _slots(), _roles(("x-unet", "UNet", "diffusion-unet", True)))
        by_type = {s["model_type"]: s["role_id"] for s in out}
        assert by_type["diffusion-unet"] == "x-unet"
        assert by_type["text-encoder"] is None
        assert by_type["vae"] is None

    def test_installed_uses_the_shared_rule(self):
        out = media_roles.annotate_slots(_slots(), IMAGE_ROLES)
        by_input = {s["input_name"]: s["installed"] for s in out}
        assert by_input["unet_name"] is True
        assert by_input["vae_name"] is False        # nowhere.safetensors

    def test_originals_are_not_mutated(self):
        """workflow_model_slots is shared with preflight_models; a display
        concern must never edit what generation reads."""
        slots = _slots()
        before = json.dumps(slots, sort_keys=True)
        media_roles.annotate_slots(slots, IMAGE_ROLES)
        assert json.dumps(slots, sort_keys=True) == before

    def test_a_malformed_slot_is_passed_through_and_logged(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="localm"):
            out = media_roles.annotate_slots(["not-a-dict"], IMAGE_ROLES)
        assert out == ["not-a-dict"]
        assert any("not the documented dict shape" in r.getMessage()
                   for r in caplog.records), caplog.text


# --------------------------------------------------------------------------- #
#  registry slice
# --------------------------------------------------------------------------- #

REGISTRY = {
    "flux": {"path": "/models/unet/flux1-dev-Q8_0.gguf", "model_type": "diffusion-unet"},
    "clipl": {"path": "/models/clip/clip_l.safetensors", "model_type": "text-encoder"},
    "aevae": {"path": "/models/vae/ae.safetensors", "model_type": "vae"},
    "offpiste": {"path": "/elsewhere/my_own_vae.safetensors", "model_type": "vae"},
    "chat": {"path": "/models/qwen.gguf", "model_type": "llm"},
    "legacy": {"path": "/models/old.gguf"},                  # no model_type -> llm
    "broken": {"model_type": "vae"},                          # no path at all
    "notadict": "nope",
}


class TestRegistrySlice:
    def test_models_of_type_is_the_slice_list_models_renders(self):
        from localm.model_manager.registry import models_of_type
        assert set(models_of_type("vae", REGISTRY)) == {"aevae", "offpiste", "broken"}
        assert set(models_of_type("diffusion-unet", REGISTRY)) == {"flux"}

    def test_a_legacy_entry_with_no_model_type_counts_as_llm(self):
        from localm.model_manager.registry import models_of_type
        assert "legacy" in models_of_type("llm", REGISTRY)

    def test_filenames_only_no_paths_leak_to_the_api_surface(self):
        out = media_roles.registry_models_of_type("vae", REGISTRY)
        assert {m["filename"] for m in out} == {"ae.safetensors", "my_own_vae.safetensors"}
        for m in out:
            assert set(m) == {"name", "filename"}, (
                "/api/models does not expose registry paths and neither may this")

    def test_a_malformed_entry_is_skipped_not_fatal(self):
        out = media_roles.registry_models_of_type("vae", REGISTRY)
        assert "broken" not in {m["name"] for m in out}

    def test_by_type_covers_exactly_the_component_types(self):
        out = media_roles.registry_models_by_type(REGISTRY)
        assert set(out) == set(media_roles.COMPONENT_TYPES)
        assert "llm" not in out, "a chat model must never be offered as a component"


# --------------------------------------------------------------------------- #
#  per-role status
# --------------------------------------------------------------------------- #

class TestDescribeRoles:
    def test_in_workflow_is_none_when_comfyui_could_not_be_asked(self):
        out = media_roles.describe_roles(IMAGE_ROLES, None, REGISTRY)
        assert {r["in_workflow"] for r in out} == {None}
        assert {r["installed"] for r in out} == {None}

    def test_in_workflow_is_false_when_it_was_asked_and_the_role_is_absent(self):
        out = {r["role_id"]: r for r in
               media_roles.describe_roles(IMAGE_ROLES, _slots(), REGISTRY)}
        assert out["image-unet"]["in_workflow"] is True
        # the flux-shaped fixture has no LoraLoader node
        assert out["image-lora"]["in_workflow"] is False

    def test_registry_models_are_returned_even_when_comfyui_is_down(self):
        """The panel must stop being a dead end: the registry needs no ComfyUI."""
        out = {r["role_id"]: r for r in
               media_roles.describe_roles(IMAGE_ROLES, None, REGISTRY)}
        assert [m["name"] for m in out["image-vae"]["registry_models"]] == \
            ["aevae", "offpiste"]

    def test_registry_only_names_what_comfyui_is_not_offering(self):
        """The vae slot asks for nowhere.safetensors, which ComfyUI does not
        have, so the registered VAEs it is not offering ARE the actionable
        answer. ae is in the live options; my_own_vae is not."""
        out = {r["role_id"]: r for r in
               media_roles.describe_roles(IMAGE_ROLES, _slots(), REGISTRY)}
        assert out["image-vae"]["installed"] is False
        assert [m["name"] for m in out["image-vae"]["registry_only"]] == ["offpiste"]

    def test_a_slot_comfyui_serves_fine_gets_no_registry_noise(self):
        """Without this gate every same-type model you own is advertised under
        every WORKING slot too, burying the slots that are actually
        unsatisfied."""
        out = {r["role_id"]: r for r in
               media_roles.describe_roles(IMAGE_ROLES, _slots(), REGISTRY)}
        assert out["image-unet"]["installed"] is True
        assert out["image-unet"]["registry_only"] == []

    def test_registry_only_is_empty_without_a_slot_to_compare_against(self):
        out = {r["role_id"]: r for r in
               media_roles.describe_roles(IMAGE_ROLES, None, REGISTRY)}
        assert out["image-vae"]["registry_only"] == []

    def test_required_flag_and_current_value_survive(self):
        out = {r["role_id"]: r for r in
               media_roles.describe_roles(IMAGE_ROLES, _slots(), REGISTRY)}
        assert out["image-unet"]["required"] is True
        assert out["image-clip1"]["required"] is False
        assert out["image-unet"]["current"] == "wan_5B_fp16.safetensors"
        assert out["image-vae"]["installed"] is False


class TestResolveModelRoles:
    def test_unreachable_still_carries_the_registry_answers(self):
        out = media_roles.resolve_model_roles(None, IMAGE_ROLES, REGISTRY)
        assert out["reachable"] is False
        assert out["slots"] == []
        assert out["registry_models"]["vae"]
        assert len(out["roles"]) == len(IMAGE_ROLES)

    def test_reachable_annotates_and_describes(self):
        out = media_roles.resolve_model_roles(_slots(), IMAGE_ROLES, REGISTRY)
        assert out["reachable"] is True
        assert all("model_type" in s for s in out["slots"])
        assert len(out["roles"]) == len(IMAGE_ROLES)


# --------------------------------------------------------------------------- #
#  what the three plugins actually declare
# --------------------------------------------------------------------------- #

PICKERS = [
    ("image", "/api/imagine/comfy-models"),
    ("music", "/api/music/comfy-models"),
    ("video", "/api/video/comfy-models"),
]


def _media_app(tmp_path, monkeypatch, plugin):
    """A real app with the plugin INSTALLED, so the roles come from the real
    register() call, not a fixture restating them."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.web import attach_gui
    app = FastAPI()
    manager = PluginManager(app, external_root=tmp_path / "noplugins")
    manager.install(plugin)
    # What attach_engine() does on a real server (engine.py). The role lookup
    # reads the manager off app.state, so without this line every role assertion
    # below would pass vacuously against an empty list.
    app.state.plugin_manager = manager

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    return app


def _installed_backend(plugin: str):
    import sys
    mod = sys.modules.get(f"_localm_plugin_{plugin}.backend")
    assert mod is not None, (
        f"the installed {plugin} plugin's backend module was not found - the "
        "plugin loader's naming changed and this test would silently patch nothing")
    return mod


class TestDeclaredRolesCoverTheShippedWorkflow:
    """Joins each plugin's declared roles to its shipped workflow's real slots,
    so an under-declared role shows up for all three plugins."""

    WORKFLOWS = {
        "image": "image_gen/flux_workflow.example.json",
        "music": "music_gen/ace_workflow.json",
        "video": "video_gen/wan_workflow.json",
    }

    @pytest.mark.parametrize("plugin", ["image", "music", "video"])
    def test_every_component_type_in_the_shipped_workflow_has_a_role(
            self, tmp_path, monkeypatch, plugin):
        app = _media_app(tmp_path, monkeypatch, plugin)
        roles = media_roles.plugin_model_roles(app, plugin)
        assert roles, f"{plugin} declared no model roles at all"
        declared = {r["model_type"] for r in roles}

        root = Path(__file__).resolve().parents[1] / "localm"
        wf = json.loads((root / self.WORKFLOWS[plugin]).read_text(encoding="utf-8"))
        # Loader nodes only: CLIPTextEncode classifies as text-encoder but holds
        # prompt TEXT, not a model file, so it is not a slot and needs no role.
        used = {comfy_client.model_type_for_node(n.get("class_type"))
                for n in wf.values() if isinstance(n, dict)
                and any(k.endswith("_name") for k in (n.get("inputs") or {}))}
        used.discard("unknown")
        assert used <= declared, (
            f"{plugin}'s shipped workflow uses {sorted(used - declared)} with no "
            f"declared role, so those picker rows can never be labelled")

    @pytest.mark.parametrize("plugin", ["image", "music", "video"])
    def test_roles_are_scoped_to_the_asking_plugin(self, tmp_path, monkeypatch, plugin):
        app = _media_app(tmp_path, monkeypatch, plugin)
        roles = media_roles.plugin_model_roles(app, plugin)
        assert {r["plugin_name"] for r in roles} == {plugin}

    def test_no_manager_means_no_roles_not_a_crash(self):
        assert media_roles.plugin_model_roles(FastAPI(), "image") == []


class TestDocumentedRoleContract:
    """docs/plugins.md's "Model roles" section states these. A documented claim
    with no runnable check is exactly the kind that quietly stops being true (it
    sits next to the code, so readers trust it and nobody re-derives it)."""

    def test_an_unknown_model_type_raises_at_registration(self):
        """Documented as raising at register() time - the alternative is a role
        that silently never matches anything and reads as an empty picker."""
        from localm.plugins.contract import ModelRoleDescriptor
        from localm.plugins.engine import PluginHost, PluginSpec
        host = PluginHost(FastAPI(), object(), PluginSpec(name="demo"))
        with pytest.raises(ValueError, match="Invalid model_type"):
            host.register_model_role(
                ModelRoleDescriptor("demo-x", "X", "not-a-real-type"))

    def test_plugin_name_is_stamped_by_the_host_not_the_plugin(self):
        """Documented as "filled in by the host; do not set it" - the roles
        surface filters by it, so a plugin able to forge it could label its
        slots as another plugin's."""
        from localm.plugins.contract import ModelRoleDescriptor
        from localm.plugins.engine import PluginHost, PluginSpec
        host = PluginHost(FastAPI(), object(), PluginSpec(name="demo"))
        host.register_model_role(
            ModelRoleDescriptor("demo-vae", "VAE", "vae", plugin_name="someone-else"))
        assert host.model_roles[0].plugin_name == "demo"

    def test_a_disabled_plugin_stops_reporting_its_roles(self, tmp_path, monkeypatch):
        """A disabled or uninstalled plugin's roles stop being reported. Nothing
        clears host.model_roles on unmount; the engine only walks LOADED hosts.
        This pins the behaviour, not the mechanism."""
        app = _media_app(tmp_path, monkeypatch, "image")
        manager = app.state.plugin_manager
        assert media_roles.plugin_model_roles(app, "image")
        manager._unload("image")
        assert media_roles.plugin_model_roles(app, "image") == []

    def test_every_declared_role_uses_a_real_registry_type(self, tmp_path, monkeypatch):
        from localm.model_manager.registry import MODEL_TYPES
        for plugin in ("image", "music", "video"):
            app = _media_app(tmp_path, monkeypatch, plugin)
            for r in media_roles.plugin_model_roles(app, plugin):
                assert r["model_type"] in MODEL_TYPES


# --------------------------------------------------------------------------- #
#  the three routes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("plugin,route", PICKERS, ids=[p[0] for p in PICKERS])
def test_route_returns_roles_and_registry_models_when_reachable(
        tmp_path, monkeypatch, plugin, route):
    app = _media_app(tmp_path, monkeypatch, plugin)
    backend = _installed_backend(plugin)
    monkeypatch.setattr(backend, "_comfy_model_slots", lambda s: _slots())

    with TestClient(app) as client:
        body = client.get(route).json()

    assert body["reachable"] is True
    assert body["roles"], "the declared roles never reached the response"
    assert set(body["registry_models"]) == set(media_roles.COMPONENT_TYPES)
    unet = [s for s in body["slots"] if s["input_name"] == "unet_name"][0]
    assert unet["model_type"] == "diffusion-unet"
    assert unet["role_id"] and unet["role_label"]
    assert unet["installed"] is True


@pytest.mark.parametrize("plugin,route", PICKERS, ids=[p[0] for p in PICKERS])
def test_route_still_answers_with_roles_when_comfyui_is_down(
        tmp_path, monkeypatch, plugin, route):
    """The registry needs no ComfyUI, so with ComfyUI unreachable the roles and
    what could fill them still come back, alongside the "not running" message."""
    app = _media_app(tmp_path, monkeypatch, plugin)
    backend = _installed_backend(plugin)
    monkeypatch.setattr(backend, "_comfy_model_slots", lambda s: None)

    with TestClient(app) as client:
        body = client.get(route).json()

    assert body["reachable"] is False
    assert body["slots"] == []
    assert "not running" in body["message"]
    assert body["roles"], "an unreachable ComfyUI wiped the declared roles"
    assert {r["in_workflow"] for r in body["roles"]} == {None}, (
        "'ComfyUI could not be asked' was reported as 'this workflow has no "
        "such slot' - the exact collapse rule 5 forbids")


@pytest.mark.parametrize("plugin,route", PICKERS, ids=[p[0] for p in PICKERS])
def test_route_resolution_still_runs_off_the_event_loop(
        tmp_path, monkeypatch, plugin, route):
    """The /object_info fetch AND the registry read both sit behind
    _comfy_model_roles, so the event-loop offload has to cover it."""
    app = _media_app(tmp_path, monkeypatch, plugin)
    backend = _installed_backend(plugin)
    seen: dict = {}

    def _probe(s):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        return _slots()

    monkeypatch.setattr(backend, "_comfy_model_slots", _probe)
    with TestClient(app) as client:
        assert client.get(route).status_code == 200
    assert seen.get("on_loop") is False
