# SPDX-License-Identifier: AGPL-3.0-or-later
"""Guards for the FLUX generator docs matching the code.

Two claims that must stay true:
  - docs/flux-setup.md names the T5 encoder filename the committed workflow
    actually loads; a mismatch is a guaranteed first-run ComfyUI error.
  - the generate_image negative-prompt description names the node the code
    actually builds: a real negative branch via CFGGuider, not
    "ConditioningConcat".
"""

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / "localm" / "image_gen" / "flux_workflow.example.json"
_DOC = _ROOT / "docs" / "flux-setup.md"


def _referenced_model_files() -> set:
    wf = json.loads(_WORKFLOW.read_text(encoding="utf-8"))
    files = set()
    for node in wf.values():
        for val in (node.get("inputs", {}) or {}).values():
            if isinstance(val, str) and val.endswith((".gguf", ".safetensors")):
                files.add(val)
    return files


def test_setup_doc_lists_every_model_file_the_workflow_loads():
    doc = _DOC.read_text(encoding="utf-8")
    referenced = _referenced_model_files()
    assert referenced, "no model files found in the example workflow"
    missing = sorted(f for f in referenced if f not in doc)
    assert not missing, (
        "docs/flux-setup.md must list the files the committed workflow loads, "
        f"missing: {missing}"
    )


def test_generate_image_negative_desc_describes_cfg_not_concat():
    from localm.plugins.coder.tools import TOOL_REGISTRY
    desc = TOOL_REGISTRY["generate_image"].params["negative_prompt"]["description"]
    assert "ConditioningConcat" not in desc
    assert "CFGGuider" in desc or "classifier-free" in desc.lower()
