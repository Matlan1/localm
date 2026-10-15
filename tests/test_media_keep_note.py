# SPDX-License-Identifier: AGPL-3.0-or-later
"""The job-log line for a media job that keeps the chat model names the actual
reason, never claims both models fit when the policy or the numbers say else."""

from __future__ import annotations

from localm.vram import decide_media_swap, media_keep_note

GB = 1024 ** 3


def test_policy_never_says_so_and_does_not_claim_a_fit():
    s = {"swap_policy": "never", "vram_estimate_bytes": 12 * GB}
    assert decide_media_swap(s, read_free=lambda: 1 * GB) is False
    note = media_keep_note(s, read_free=lambda: 1 * GB)
    assert "never" in note and "fit" not in note


def test_auto_that_fits_names_the_numbers():
    s = {"swap_policy": "auto", "vram_estimate_bytes": 4 * GB}
    assert decide_media_swap(s, read_free=lambda: 10 * GB) is False
    note = media_keep_note(s, read_free=lambda: 10 * GB)
    assert "10.0 GB" in note and "4.0 GB" in note
    assert "both models" not in note.lower()
