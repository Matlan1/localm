# SPDX-License-Identifier: AGPL-3.0-or-later
"""An auto-typed pull of a ControlNet or textual-inversion safetensors file
registers with that type instead of crashing after the download, and no media
type a tensor probe can return is missing its console wording. Only the
network is faked; the typing and registration run for real."""

import pytest

import localm.model_manager as mm
from localm.model_manager import pull as pull_mod
from localm.model_manager.gguf import MEDIA_KIND_TYPES
from tests.test_pull_safetensors_content_type import (  # noqa: F401
    _entry, _safetensors_bytes, hf_network,
)

CONTROLNET = {"control_model.input_hint_block.0.weight": (16, 3, 3, 3),
              "control_model.zero_convs.0.0.weight": (320, 320, 1, 1),
              "control_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": (320, 768),
              "control_model.middle_block_out.0.weight": (1280, 1280, 1, 1)}
TEXTUAL_INVERSION = {"emb_params": (8, 768)}
CASES = [
    ("comfyanonymous/ControlNet-v1-1_fp16_safetensors:control_v11p_sd15_canny_fp16.safetensors",
     "control_v11p_sd15_canny_fp16.safetensors", CONTROLNET, "controlnet"),
    ("gsdf/EasyNegative:EasyNegative.safetensors",
     "EasyNegative.safetensors", TEXTUAL_INVERSION, "textual-inversion"),
]


@pytest.mark.parametrize("spec,filename,shapes,want", CASES)
def test_an_auto_typed_pull_registers_the_addon_type(hf_network, spec, filename, shapes, want):  # noqa: F811
    hf_network["body"] = _safetensors_bytes(shapes)
    assert mm.pull_model(spec) is True
    assert hf_network["downloads"] == 1
    assert _entry(filename)["model_type"] == want


@pytest.mark.parametrize("spec,filename,shapes,want", CASES)
def test_a_re_pull_of_an_already_downloaded_addon_registers_it(
        hf_network, spec, filename, shapes, want):  # noqa: F811
    hf_network["body"] = _safetensors_bytes(shapes)
    (hf_network["models"] / filename).write_bytes(hf_network["body"])
    assert mm.pull_model(spec) is True
    assert hf_network["downloads"] == 0
    assert _entry(filename)["model_type"] == want


@pytest.mark.parametrize("media_type", sorted(set(MEDIA_KIND_TYPES.values())))
def test_every_probe_media_type_has_console_wording(media_type):
    assert pull_mod._media_type_phrase(media_type)


def test_an_unlisted_media_type_still_has_wording():
    assert "future-type" in pull_mod._media_type_phrase("future-type")
