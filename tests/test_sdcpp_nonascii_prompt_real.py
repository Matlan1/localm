# SPDX-License-Identifier: AGPL-3.0-or-later
"""A non-ASCII prompt through the real stable-diffusion.cpp worker and library:
the prompt reaches the library intact and, after ``families.clip_text``, the
library's CLIP tokenizer splits it into the same tokens as the reference CLIP
tokenizer. Runs only when LOCALM_TEST_SDCPP_RUNTIME (an installed runtime
folder) and LOCALM_TEST_SDCPP_SD_MODEL (an SD 1.x/2.x model file) are set; it
loads the model on that runtime's device."""

from __future__ import annotations

import os
import queue
import re
import threading

import pytest

from localm.media.sdcpp import _child, families

RUNTIME = os.environ.get("LOCALM_TEST_SDCPP_RUNTIME", "")
MODEL = os.environ.get("LOCALM_TEST_SDCPP_SD_MODEL", "")

pytestmark = pytest.mark.skipif(not (RUNTIME and MODEL),
                                reason="needs LOCALM_TEST_SDCPP_RUNTIME and "
                                       "LOCALM_TEST_SDCPP_SD_MODEL")

PROMPT = "Ein großer Ölbaum über dem Fluß"
REFERENCE_TOKENS = ["ein</w>", "gro", "Ã", "Ł", "er</w>", "Ã¶", "l", "baum</w>", "Ã¼",
                    "ber</w>", "dem</w>", "flu", "Ã", "Ł</w>"]
_SPLIT = re.compile(r'split prompt "(.*)" to \d+ tokens \[(.*)\]')


def _run_worker(monkeypatch, prompt: str) -> list:
    lines: list = []
    orig = _child._make_callbacks

    def make(resp_q):
        from localm.media.sdcpp import _binding as b
        real = b.SD_LOG_CB

        def factory(fn):
            def tee(level, text, data):
                lines.append((text or b"").decode("utf-8", "replace"))
                fn(level, text, data)
            return real(tee)
        monkeypatch.setattr(b, "SD_LOG_CB", factory)
        try:
            return orig(resp_q)
        finally:
            monkeypatch.setattr(b, "SD_LOG_CB", real)

    monkeypatch.setattr(_child, "_make_callbacks", make)
    monkeypatch.setattr(_child, "_arm_crash_trace", lambda path: None)
    import localm._mp_spawn as spawn
    import localm.debuglog as debuglog
    for name in ("install_parent_death_watchdog", "ignore_interrupt_signals",
                 "suppress_native_error_dialogs"):
        monkeypatch.setattr(spawn, name, lambda: None)
    monkeypatch.setattr(debuglog, "attach_child_logging", lambda: None)
    req, resp = queue.Queue(), queue.Queue()
    worker = threading.Thread(target=_child.worker_main, args=(req, resp, threading.Event()),
                              daemon=True)
    worker.start()

    def call(name, payload, timeout):
        req.put((name, payload))
        while True:
            kind, *rest = resp.get(timeout=timeout)
            value = rest[-1] if rest else None
            if kind in ("ok", "error", "budget_refused", "cancelled"):
                assert kind == "ok", value
                return value

    call("load", {"runtime_dir": RUNTIME, "extra_dirs": [], "ctx": {"model_path": MODEL}}, 600)
    call("generate_image", {
        "prompt": prompt, "negative_prompt": "", "width": 64, "height": 64, "seed": 7,
        "steps": 1, "cfg_scale": 1.0, "guidance": None, "clip_skip": None, "preview": False,
        "sample_method": None, "scheduler": None, "init_image": None, "strength": None,
        "loras": [], "hires": None, "mask_image": None, "ref_images": None,
        "control_image": None, "control_strength": None, "control_canny": False,
        "vae_tile_size": None}, 600)
    req.put(("shutdown", None))
    worker.join(60)
    return lines


def test_a_non_ascii_prompt_reaches_the_real_tokenizer_as_reference_clip_tokens(monkeypatch):
    sent = families.clip_text(PROMPT, "sd2")
    lines = _run_worker(monkeypatch, sent)
    split = [m for m in map(_SPLIT.search, lines) if m]
    assert split, "the library logged no tokenizer split"
    text, tokens = split[0].group(1), split[0].group(2)
    assert text == sent
    assert re.findall(r'"([^"]*)"', tokens) == REFERENCE_TOKENS
