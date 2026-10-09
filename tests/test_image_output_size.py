# SPDX-License-Identifier: AGPL-3.0-or-later
"""Applying a requested output size to the image workflow."""

import copy
import json
from pathlib import Path

import pytest

from localm.image_gen import comfy

EXAMPLE = Path(comfy.__file__).with_name("flux_workflow.example.json")


@pytest.fixture
def workflow():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _build(workflow, **overrides):
    kwargs = dict(
        prompt="a cat", api_url="http://127.0.0.1:8188", guidance=None,
        negative_prompt=None, cfg=None, seed=1, clip_name1=None, clip_name2=None,
        lora_name=None, lora_strength_model=1.0, lora_strength_clip=0.5,
        input_image=None, denoise=None, fast_dequant=False, say=lambda t: None)
    kwargs.update(overrides)
    return comfy._build_image_workflow(workflow, **kwargs)


class TestApplyOutputSize:
    def test_sets_every_latent_and_flux_sampling_node(self, workflow):
        assert comfy.apply_output_size(workflow, 512, 768) is True
        for node_id in ("5", "28"):
            inputs = workflow[node_id]["inputs"]
            assert (inputs["width"], inputs["height"]) == (512, 768)

    def test_sd3_latent_node_is_covered(self):
        wf = {"1": {"class_type": "EmptySD3LatentImage",
                    "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}}
        assert comfy.apply_output_size(wf, 640, 384) is True
        assert (wf["1"]["inputs"]["width"], wf["1"]["inputs"]["height"]) == (640, 384)

    def test_a_workflow_without_a_latent_node_is_left_untouched(self, workflow):
        for node in list(workflow.values()):
            if node["class_type"] == "EmptyLatentImage":
                node["class_type"] = "SomethingElse"
        before = copy.deepcopy(workflow)
        assert comfy.apply_output_size(workflow, 512, 512) is False
        assert workflow == before

    def test_flux_sampling_node_alone_does_not_count(self):
        wf = {"28": {"class_type": "ModelSamplingFlux",
                     "inputs": {"width": 1024, "height": 1024}}}
        assert comfy.apply_output_size(wf, 512, 512) is False
        assert wf["28"]["inputs"]["width"] == 1024


class TestBuildImageWorkflowSize:
    def test_size_reaches_the_workflow(self, workflow):
        ok, msg, _ = _build(workflow, width=512, height=256)
        assert ok, msg
        assert (workflow["5"]["inputs"]["width"], workflow["5"]["inputs"]["height"]) == (512, 256)

    def test_no_size_keeps_the_workflow_size(self, workflow):
        ok, _msg, _ = _build(workflow)
        assert ok
        assert workflow["5"]["inputs"]["width"] == 1024

    def test_width_without_height_is_refused(self, workflow):
        ok, msg, _ = _build(workflow, width=512)
        assert not ok and "together" in msg

    def test_size_with_an_input_image_is_refused_before_any_upload(
            self, workflow, monkeypatch, tmp_path):
        monkeypatch.setattr(comfy, "_upload_image",
                            lambda *a, **k: pytest.fail("must not upload"))
        src = tmp_path / "in.png"
        src.write_bytes(b"x")
        ok, msg, uploaded = _build(workflow, width=512, height=512, input_image=src)
        assert not ok and uploaded is None
        assert "input image" in msg

    def test_workflow_without_a_latent_node_fails_with_a_message_naming_the_nodes(
            self, workflow):
        for node in workflow.values():
            if node["class_type"] == "EmptyLatentImage":
                node["class_type"] = "SomethingElse"
        ok, msg, _ = _build(workflow, width=512, height=512)
        assert not ok
        assert "EmptyLatentImage" in msg and "512x512" in msg
