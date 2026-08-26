# SPDX-License-Identifier: AGPL-3.0-or-later
"""workflow_model_slots() / apply_model_overrides() (the model-picker feature):
enumerate every model-file combo slot in a workflow against ComfyUI's live
/object_info, and let a caller override any of them by node_id/input_name.
Shares the same node/input walk preflight_models() uses to validate model
choices - these tests also pin that preflight_models' own observable behaviour
is unchanged."""

from unittest.mock import patch

from localm.media import comfy_client


def _object_info():
    return {
        "UnetLoaderGGUFAdvanced": {"input": {"required": {
            "unet_name": [["flux1-dev-Q8_0.gguf", "flux1-dev-Q6_K.gguf"], {}],
            "dequant_dtype": [["default", "float32"]],
        }}},
        "DualCLIPLoader": {"input": {"required": {
            "clip_name1": [["clip_l.safetensors"], {}],
            "clip_name2": [["t5xxl_fp8_e4m3fn.safetensors", "t5xxl_fp16.safetensors"], {}],
        }}},
        "VAELoader": {"input": {"required": {
            "vae_name": [["ae.safetensors"], {}],
        }}},
        "KSampler": {"input": {"required": {
            "sampler_name": [["euler", "uni_pc"]],
            "seed": ["INT"],
        }}},
    }


def _workflow():
    return {
        "1": {"class_type": "UnetLoaderGGUFAdvanced",
              "inputs": {"unet_name": "flux1-dev-Q8_0.gguf", "dequant_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": "clip_l.safetensors",
                         "clip_name2": "t5xxl_fp8_e4m3fn.safetensors"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "8": {"class_type": "KSampler", "inputs": {"sampler_name": "euler", "seed": 0}},
    }


class TestWorkflowModelSlots:
    def test_none_when_object_info_unavailable(self):
        with patch.object(comfy_client, "comfy_object_info", return_value=None):
            assert comfy_client.workflow_model_slots(_workflow(), "http://x") is None

    def test_finds_every_model_file_slot_not_enums(self):
        with patch.object(comfy_client, "comfy_object_info", return_value=_object_info()):
            slots = comfy_client.workflow_model_slots(_workflow(), "http://x")
        assert slots is not None
        by_input = {s["input_name"]: s for s in slots}
        # Model-file combos are present...
        assert by_input["unet_name"]["current"] == "flux1-dev-Q8_0.gguf"
        assert by_input["unet_name"]["options"] == ["flux1-dev-Q8_0.gguf", "flux1-dev-Q6_K.gguf"]
        assert by_input["unet_name"]["node_id"] == "1"
        assert by_input["unet_name"]["class_type"] == "UnetLoaderGGUFAdvanced"
        assert by_input["clip_name1"]["current"] == "clip_l.safetensors"
        assert by_input["clip_name2"]["current"] == "t5xxl_fp8_e4m3fn.safetensors"
        assert by_input["vae_name"]["current"] == "ae.safetensors"
        # ...but non-model-file combos/links/numbers are NOT (dequant_dtype is an
        # enum, sampler_name/seed are on a node with no model-file inputs at all).
        assert "dequant_dtype" not in by_input
        assert "sampler_name" not in by_input
        assert "seed" not in by_input

    def test_empty_list_when_workflow_has_no_model_slots(self):
        wf = {"8": {"class_type": "KSampler", "inputs": {"sampler_name": "euler", "seed": 0}}}
        with patch.object(comfy_client, "comfy_object_info", return_value=_object_info()):
            assert comfy_client.workflow_model_slots(wf, "http://x") == []

    def test_slot_surfaces_when_comfyui_has_none_of_that_file_type_installed(self):
        """ComfyUI has ZERO files of a given type installed, so its live
        /object_info reports that combo's options as [] - which alone can never
        look like model files (_looks_like_model_files([]) is False). Without
        folding the node's current value into that heuristic, every slot in a
        workflow whose ComfyUI is missing all its models vanishes from the picker
        instead of surfacing exactly the you-are-missing-these case it exists
        for. VAELoader's vae_name is ComfyUI's own built-in latent-space
        pseudo-option ("pixel_space"), not a model file - it must still be
        recognized because the workflow's current value is a real filename."""
        info = _object_info()
        info["UnetLoaderGGUFAdvanced"]["input"]["required"]["unet_name"] = [[], {}]
        info["DualCLIPLoader"]["input"]["required"]["clip_name1"] = [[], {}]
        info["DualCLIPLoader"]["input"]["required"]["clip_name2"] = [[], {}]
        info["VAELoader"]["input"]["required"]["vae_name"] = [["pixel_space"], {}]
        with patch.object(comfy_client, "comfy_object_info", return_value=info):
            slots = comfy_client.workflow_model_slots(_workflow(), "http://x")
        assert slots is not None
        by_input = {s["input_name"]: s for s in slots}
        assert set(by_input) == {"unet_name", "clip_name1", "clip_name2", "vae_name"}
        assert by_input["unet_name"]["options"] == []
        assert by_input["unet_name"]["current"] == "flux1-dev-Q8_0.gguf"
        assert by_input["vae_name"]["options"] == ["pixel_space"]
        assert by_input["vae_name"]["current"] == "ae.safetensors"


class TestApplyModelOverrides:
    def test_applies_a_valid_override(self):
        wf = _workflow()
        changed = comfy_client.apply_model_overrides(
            wf, {"1": {"unet_name": "flux1-dev-Q6_K.gguf"}})
        assert changed == 1
        assert wf["1"]["inputs"]["unet_name"] == "flux1-dev-Q6_K.gguf"

    def test_applies_multiple_overrides_across_nodes(self):
        wf = _workflow()
        changed = comfy_client.apply_model_overrides(wf, {
            "1": {"unet_name": "flux1-dev-Q6_K.gguf"},
            "2": {"clip_name2": "t5xxl_fp16.safetensors"},
        })
        assert changed == 2
        assert wf["1"]["inputs"]["unet_name"] == "flux1-dev-Q6_K.gguf"
        assert wf["2"]["inputs"]["clip_name2"] == "t5xxl_fp16.safetensors"

    def test_same_value_is_not_counted_as_a_change(self):
        wf = _workflow()
        changed = comfy_client.apply_model_overrides(
            wf, {"1": {"unet_name": "flux1-dev-Q8_0.gguf"}})   # already this value
        assert changed == 0

    def test_unknown_node_id_is_a_noop_not_an_error(self):
        wf = _workflow()
        changed = comfy_client.apply_model_overrides(wf, {"999": {"unet_name": "x"}})
        assert changed == 0
        assert "999" not in wf

    def test_unknown_input_name_on_a_real_node_is_a_noop(self):
        """A stale/malformed override (naming a field that does not exist on
        that node) must never CREATE a new key - only ever write an EXISTING
        string input."""
        wf = _workflow()
        changed = comfy_client.apply_model_overrides(
            wf, {"1": {"not_a_real_field": "x"}})
        assert changed == 0
        assert "not_a_real_field" not in wf["1"]["inputs"]

    def test_cannot_overwrite_a_non_string_input(self):
        """A number/link input must never be clobbered by a string override -
        this is the graph-corruption guard, not just a UX nicety."""
        wf = _workflow()
        changed = comfy_client.apply_model_overrides(wf, {"8": {"seed": "999"}})
        assert changed == 0
        assert wf["8"]["inputs"]["seed"] == 0

    def test_empty_or_malformed_overrides_are_safe_noops(self):
        wf = _workflow()
        assert comfy_client.apply_model_overrides(wf, {}) == 0
        assert comfy_client.apply_model_overrides(wf, None) == 0
        assert comfy_client.apply_model_overrides(wf, {"1": "not a dict"}) == 0


class TestPreflightModelsStillWorksAfterRefactor:
    """preflight_models() builds on workflow_model_slots() internally - pin that
    its own observable behavior (substitution + missing-file message) is the
    same."""

    def test_substitutes_the_one_unambiguous_variant(self):
        wf = _workflow()
        wf["1"]["inputs"]["unet_name"] = "flux1-dev-Q8_0-fp16.safetensors"  # not installed
        info = _object_info()
        info["UnetLoaderGGUFAdvanced"]["input"]["required"]["unet_name"] = [
            ["flux1-dev-Q8_0.gguf"], {}]   # the fp16 variant "downgrades" to this
        with patch.object(comfy_client, "comfy_object_info", return_value=info):
            ok, msg = comfy_client.preflight_models(wf, "http://x")
        assert ok is True
        assert wf["1"]["inputs"]["unet_name"] == "flux1-dev-Q8_0.gguf"

    def test_reports_a_genuinely_missing_file(self):
        wf = _workflow()
        wf["3"]["inputs"]["vae_name"] = "totally-unrelated-vae.safetensors"
        with patch.object(comfy_client, "comfy_object_info", return_value=_object_info()):
            ok, msg = comfy_client.preflight_models(wf, "http://x")
        assert ok is False
        assert "totally-unrelated-vae.safetensors" in msg
        assert "VAELoader" in msg

    def test_noop_when_object_info_unavailable(self):
        wf = _workflow()
        with patch.object(comfy_client, "comfy_object_info", return_value=None):
            ok, msg = comfy_client.preflight_models(wf, "http://x")
        assert ok is True and msg == ""
