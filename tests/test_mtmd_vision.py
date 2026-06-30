# SPDX-License-Identifier: AGPL-3.0-or-later
"""GGUF vision (C1) plumbing that is testable WITHOUT a model + GPU: the image ->
marker message rewrite, and that a text-only GGUF (no mmproj) still refuses images
while one with an mmproj advertises it is worth loading. The full mtmd image path
is verified end-to-end against a real vision model on an actual GPU (see
dev-notes/open-points-loop-worklog.md - it returned a correct image description)."""

import base64
import io

import pytest

# Pillow ships only with the [gpu] extra, which CI deliberately does not install
# (the ci.yml Tests step uses [dev,rag] only). Skip cleanly when it is absent so
# the suite still collects, matching the repo convention that gpu/gguf-tier tests
# importorskip themselves rather than break collection for everyone.
Image = pytest.importorskip("PIL.Image")


def _data_url(w=8, h=8):
    im = Image.new("RGB", (w, h), (10, 20, 30))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_messages_with_markers_extracts_images_and_inserts_marker():
    from localm.inference.backends.llamacpp.llama import LlamaCpp
    msgs = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _data_url()}},
            {"type": "text", "text": "what is this?"},
        ]},
    ]
    text_msgs, images = LlamaCpp._messages_with_markers(msgs, "<MARK>")
    assert len(images) == 1
    w, h, rgb = images[0]
    assert (w, h) == (8, 8)
    assert len(rgb) == 8 * 8 * 3                 # raw RGB, 3 bytes/pixel
    assert text_msgs[0]["content"] == "be terse"  # non-image message untouched
    assert "<MARK>" in text_msgs[1]["content"]    # marker spliced in where the image was
    assert "what is this?" in text_msgs[1]["content"]


def test_two_images_yield_two_markers_in_order():
    from localm.inference.backends.llamacpp.llama import LlamaCpp
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _data_url(4, 4)}},
        {"type": "image_url", "image_url": {"url": _data_url(6, 6)}},
        {"type": "text", "text": "compare"},
    ]}]
    text_msgs, images = LlamaCpp._messages_with_markers(msgs, "<M>")
    assert [(w, h) for (w, h, _) in images] == [(4, 4), (6, 6)]
    assert text_msgs[0]["content"].count("<M>") == 2


def test_gguf_without_mmproj_is_text_only_and_refuses_images():
    from localm.inference.backends.base import UnsupportedInputError
    from localm.inference.backends.gguf import GgufBackend
    be = GgufBackend("/fake/model.gguf")          # no mmproj, not loaded
    assert be.can_be_multimodal is False
    assert be.supports_images is False
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _data_url()}}]}]
    with pytest.raises(UnsupportedInputError):
        list(be.chat_stream(msgs))


def test_gguf_with_mmproj_advertises_can_be_multimodal():
    from localm.inference.backends.gguf import GgufBackend
    be = GgufBackend("/fake/model.gguf", mmproj_path="/fake/mmproj.gguf")
    assert be.can_be_multimodal is True           # worth loading to check vision
    assert be.supports_images is False            # not loaded yet -> not confirmed
