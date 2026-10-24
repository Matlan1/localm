# SPDX-License-Identifier: AGPL-3.0-or-later
"""Segment Anything checkpoints register as 'segmentation' (not 'unknown') when
a model-folder scan finds them, and the type is selectable everywhere a model
type is."""

from __future__ import annotations

import json
import pickle
import re
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from localm import model_manager as mm
from localm.model_manager.sam import is_sam_checkpoint
from localm.model_manager.scan import (
    SUBFOLDER_MAPPING, _discover_comfy_files, scan_comfy_models,
)
from tests.test_unified_model_browser import temp_registry  # noqa: F401

SAM1 = ["image_encoder.patch_embed.proj.weight", "image_encoder.blocks.0.attn.qkv.weight",
        "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
        "mask_decoder.iou_token.weight"]
SAM2 = ["image_encoder.trunk.patch_embed.proj.weight",
        "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
        "sam_mask_decoder.iou_token.weight"]
NOT_SAM = ["model.embed_tokens.weight", "lm_head.weight"]
REPO = Path(__file__).resolve().parents[1]


def write_pth(path: Path, keys: list[str], pickle_name: str = "archive/data.pkl") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(pickle_name, pickle.dumps({k: 0 for k in keys}, protocol=2))
        zf.writestr("archive/data/0", b"\0" * 8)
    return path


@pytest.fixture(autouse=True)
def _comfy_offline():
    with patch("localm.model_manager.scan.comfy_object_info", return_value=None):
        yield


def _discover(models: Path) -> dict[str, str]:
    found, _ = _discover_comfy_files(models)
    assert found is not None
    return {p.relative_to(models).as_posix(): t for p, t in found.items()}


@pytest.mark.parametrize("keys", [SAM1, SAM2])
@pytest.mark.parametrize("suffix", [".pth", ".PT"])
def test_a_sam_checkpoint_is_recognised_from_its_tensor_names(tmp_path, keys, suffix):
    assert is_sam_checkpoint(write_pth(tmp_path / f"sam{suffix}", keys)) is True


def test_a_checkpoint_missing_a_sam_part_is_not_sam(tmp_path):
    assert is_sam_checkpoint(write_pth(tmp_path / "a.pth", SAM1[:2])) is False
    assert is_sam_checkpoint(write_pth(tmp_path / "b.pth", NOT_SAM)) is False


def test_files_that_cannot_be_probed_are_not_sam(tmp_path):
    plain = tmp_path / "legacy.pth"
    plain.write_bytes(b"not a zip archive")
    wrong_suffix = write_pth(tmp_path / "sam.bin", SAM1)
    no_pickle = write_pth(tmp_path / "nopkl.pth", SAM1, pickle_name="archive/weights.bin")
    assert [is_sam_checkpoint(p) for p in (plain, wrong_suffix, no_pickle, tmp_path / "gone.pth")] \
        == [False] * 4


def test_the_pickle_is_never_executed(tmp_path):
    marker = tmp_path / "ran"

    class Evil:
        def __reduce__(self):
            return (marker.write_text, ("x",))

    path = tmp_path / "evil.pth"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", pickle.dumps({"k": Evil(), "image_encoder.patch_embed": 1,
                                                       "prompt_encoder": 1, "mask_decoder": 1},
                                                      protocol=2))
    assert is_sam_checkpoint(path) is True
    assert not marker.exists()


def test_scan_types_sam_by_content_and_by_folder(tmp_path):
    models = tmp_path / "models"
    write_pth(models / "checkpoints" / "sam_vit_b.pth", SAM1)
    write_pth(models / "Sams" / "opaque.pth", NOT_SAM)
    write_pth(models / "segmentation" / "opaque2.pth", NOT_SAM)
    write_pth(models / "sam" / "opaque3.pth", NOT_SAM)
    write_pth(models / "other" / "weights.pth", NOT_SAM)
    assert _discover(models) == {
        "checkpoints/sam_vit_b.pth": "segmentation",
        "Sams/opaque.pth": "segmentation",
        "segmentation/opaque2.pth": "segmentation",
        "sam/opaque3.pth": "segmentation",
        "other/weights.pth": "unknown",
    }


def test_scan_registers_a_stabilitymatrix_sam_file_as_segmentation(tmp_path, temp_registry):  # noqa: F811
    models = tmp_path / "StabilityMatrix" / "Data" / "Models"
    write_pth(models / "Sams" / "sam_vit_b_01ec64.pth", SAM1)
    res = scan_comfy_models(workdir=str(models.parent.parent))
    assert (res.added, res.skipped) == (1, 0)
    assert [e["model_type"] for e in mm.load_registry().values()] == ["segmentation"]


def test_segmentation_is_a_registry_type_and_every_folder_mapping_still_valid():
    assert "segmentation" in mm.MODEL_TYPES
    assert set(SUBFOLDER_MAPPING.values()) <= mm.MODEL_TYPES


def test_set_type_lists_segmentation():
    from click.testing import CliRunner

    from localm.cli import main
    assert "segmentation" in CliRunner().invoke(main, ["set-type", "--help"]).output


def test_gui_import_labels_exist_in_english_and_german():
    en = (REPO / "localm/plugins/gui/static/app/i18n-en.js").read_text(encoding="utf-8")
    de = json.loads((REPO / "localm/plugins/gui/static/i18n/de.json").read_text(encoding="utf-8"))
    key = "models.importComfy.type.segmentation"
    assert re.search(rf'"{re.escape(key)}":\s*"[^"]+"', en)
    assert de[key] and de[key] != "Segmentation models (SAM)"
    js = (REPO / "localm/plugins/gui/static/pages/models.js").read_text(encoding="utf-8")
    assert f'segmentation: "{key}"' in js
