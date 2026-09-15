# SPDX-License-Identifier: AGPL-3.0-or-later
"""Integration-test scaffold. Beyond the coarse `integration` marker, real
end-to-end paths are tagged with a resource-specific marker - `real_gguf`,
`real_comfy` - that conftest gates: a test carrying one is skipped (not
failed) unless its resource is actually available.

This module holds the always-runnable oracle (the markers are registered), a
comfy preflight check plus adapter-level integration tests gated on
`real_comfy`, and the gguf path is the existing
tests/test_gguf_smoke_integration.py, tagged `real_gguf`. Real-browser boot
coverage - loading the actual shipped ES-module graph in headless Chromium -
lives in tests-e2e/boot-and-click.spec.mjs, run with `npm run test:e2e`.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_real_resource_markers_registered():
    """The three resource markers are declared in pyproject so `-m real_gguf`
    (etc.) works and pytest does not warn about unknown markers."""
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    markers = cfg["tool"]["pytest"]["ini_options"]["markers"]
    names = {m.split(":", 1)[0].strip() for m in markers}
    for m in ("real_gguf", "real_comfy", "real_browser"):
        assert m in names, f"{m} marker is not registered in pyproject.toml"


def test_gguf_smoke_is_tagged_real_gguf():
    """The existing GGUF end-to-end smoke test carries the fine-grained marker so
    it can be selected (and gated) as a real_gguf test, not just `integration`."""
    text = (ROOT / "tests" / "test_gguf_smoke_integration.py").read_text(encoding="utf-8")
    assert "real_gguf" in text, "the GGUF smoke test should be tagged real_gguf"


# --------------------------------------------------------------------------- #
#  Gated skeletons. conftest skips these unless the resource is present, so in  #
#  CI they appear as skips, never failures.                                     #
# --------------------------------------------------------------------------- #

@pytest.mark.real_comfy
def test_comfy_preflight_server_reachable():
    """Preflight only: a real ComfyUI (set LOCALM_TEST_COMFY_URL) answers its own
    /system_stats. Exercises ComfyUI's endpoint directly, none of localm's own
    comfy_client code - see test_comfy_adapter_round_trip for that coverage. No
    `integration` marker."""
    import urllib.request

    base = os.environ["LOCALM_TEST_COMFY_URL"].rstrip("/")
    with urllib.request.urlopen(f"{base}/system_stats", timeout=10) as r:
        assert r.status == 200


@pytest.mark.integration
@pytest.mark.real_comfy
def test_comfy_adapter_round_trip(tmp_path):
    """localm's own comfy_client adapter - submission, job status, and artifact
    fetch - against a real ComfyUI, through the same functions the image/video/
    music backends call. A model-free EmptyImage -> SaveImage workflow keeps
    this runnable without any checkpoint installed on the target Comfy."""
    from localm.media import comfy_client as cc

    base = os.environ["LOCALM_TEST_COMFY_URL"].rstrip("/")
    workflow = {
        "1": {"class_type": "EmptyImage",
              "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0}},
        "2": {"class_type": "SaveImage",
              "inputs": {"images": ["1", 0],
                         "filename_prefix": "localm_integration_scaffold"}},
    }

    kind, value = cc.comfy_submit_prompt(base, workflow)
    assert kind == cc.SUBMIT_OK, f"workflow submission failed: {kind} {value}"

    status, result = cc.comfy_poll_until_done(base, value, max_poll_seconds=60)
    assert status == cc.POLL_FINISHED, (
        cc.comfy_exec_error_message(result, base) if status == cc.POLL_EXEC_ERROR
        else f"job did not finish: {status} {result}")

    info = cc.select_output_info(result, ("images",))
    assert info is not None, "no output artifact recorded for the finished job"

    out_path = tmp_path / "out.png"
    cc.comfy_fetch_output(base, info, out_path, timeout=10.0)
    assert out_path.stat().st_size > 0, "fetched output artifact is empty"


@pytest.mark.integration
@pytest.mark.real_comfy
def test_comfy_adapter_maps_submission_errors():
    """A workflow ComfyUI rejects (an unregistered node type) comes back through
    comfy_submit_prompt as a classified SUBMIT_HTTP_ERROR, not a silent
    SUBMIT_OK or an unhandled exception - the same mapping the media backends
    rely on to tell a user why generation refused to start."""
    from localm.media import comfy_client as cc

    base = os.environ["LOCALM_TEST_COMFY_URL"].rstrip("/")
    workflow = {
        "1": {"class_type": "LocalmIntegrationScaffoldDoesNotExist", "inputs": {}},
    }

    kind, value = cc.comfy_submit_prompt(base, workflow)
    assert kind == cc.SUBMIT_HTTP_ERROR, (
        f"expected a classified submit error, got {kind} {value!r}")
