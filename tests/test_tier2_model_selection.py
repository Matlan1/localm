# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline tests for scripts/tier2_gpu_split/model_selection.py's pure candidate-picking logic (the Tier 2 GPU-split harness - see scripts/tier2_gpu_split/README.md and issues/issues.txt's GPU-SPLIT-TESTING entry)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = (Path(__file__).resolve().parent.parent / "scripts" / "tier2_gpu_split"
          / "model_selection.py")
if not _PATH.is_file():
    # scripts/tier2_gpu_split/ is gitignored, maintainer-only tooling (AGENTS.md
    # rule 6) - never committed, so a fresh clone (or any worktree that did not
    # get it copied in) genuinely does not have this file. A tracked test must
    # never hard-crash COLLECTION over a file the repo itself excludes - that
    # breaks `pytest` for every external contributor, not just here. Skip with a
    # reason instead of importing; the test still runs normally once the
    # harness is present locally.
    pytest.skip(f"{_PATH} not present (gitignored maintainer-only harness, "
               "AGENTS.md rule 6) - skipping tests that need it",
               allow_module_level=True)
_spec = importlib.util.spec_from_file_location("model_selection", _PATH)
model_selection = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model_selection)

Candidate = model_selection.Candidate
pick_candidate = model_selection.pick_candidate

_GB = 1024 ** 3


def _c(name: str, gb: float) -> Candidate:
    return Candidate(f"org/{name}", (f"{name}.gguf",), int(gb * _GB))


def _split_c(name: str, part_gbs: tuple) -> Candidate:
    """A multi-part (split-GGUF) candidate, e.g. a Q6_K/Q8_0-class quant that Hugging Face stores as several sibling files - mirrors CANDIDATE_TABLE's real Q6_K/Q8_0 entries."""
    parts = tuple(f"{name}-{i + 1:05d}-of-{len(part_gbs):05d}.gguf"
                  for i in range(len(part_gbs)))
    return Candidate(f"org/{name}", parts, int(sum(part_gbs) * _GB))


class TestPickCandidate:
    def test_picks_smallest_candidate_spanning_the_gap(self):
        # single device: 16 GB free: combined: 40 GB free. 20 GB and 30 GB both
        # clear one device and fit combined - the smaller (20 GB) must win, not
        # merely the first in input order.
        candidates = [_c("thirty", 30), _c("twenty", 20), _c("eight", 8)]
        picked = pick_candidate(candidates, single_free_bytes=16 * _GB,
                                 combined_free_bytes=40 * _GB)
        assert picked is not None and picked.filename == "twenty.gguf"

    def test_none_when_nothing_exceeds_a_single_device(self):
        # every candidate fits on ONE device alone - none of them tests the
        # combined-budgeting path at all, so the honest answer is "no fit",
        # not a false-positive pick of the largest.
        candidates = [_c("eight", 8), _c("ten", 10)]
        picked = pick_candidate(candidates, single_free_bytes=16 * _GB,
                                 combined_free_bytes=40 * _GB)
        assert picked is None

    def test_none_when_nothing_fits_combined(self):
        # the only candidate that clears one device also blows the combined
        # ceiling - a real gap (this box's candidate table tops out too small,
        # or the rented GPUs are unusually large) that must surface as "no
        # candidate", never a forced pick that would silently fail to load.
        candidates = [_c("huge", 100)]
        picked = pick_candidate(candidates, single_free_bytes=16 * _GB,
                                 combined_free_bytes=40 * _GB)
        assert picked is None

    def test_margin_excludes_a_bare_exceed_of_single_device(self):
        # exactly at single_free (no margin) must NOT count as "too big for
        # one device" - measurement noise on a live box makes an exact-boundary
        # candidate an unreliable proof of the combined-budgeting path.
        candidates = [_c("exact", 16)]
        picked = pick_candidate(candidates, single_free_bytes=16 * _GB,
                                 combined_free_bytes=40 * _GB)
        assert picked is None

    def test_margin_excludes_a_bare_fit_of_combined_ceiling(self):
        # exactly at combined_free (no headroom) must NOT count as "fits
        # combined" - there would be nothing left for KV cache/buffers.
        candidates = [_c("exact-combined", 40)]
        picked = pick_candidate(candidates, single_free_bytes=16 * _GB,
                                 combined_free_bytes=40 * _GB)
        assert picked is None

    def test_empty_table_returns_none(self):
        assert pick_candidate([], single_free_bytes=16 * _GB,
                               combined_free_bytes=40 * _GB) is None

    def test_a_split_multi_part_candidate_is_sized_by_the_sum_of_its_parts(self):
        # A Q6_K/Q8_0-class quant ships as several sibling files on Hugging
        # Face (confirmed live against bartowski/Meta-Llama-3.1-70B-Instruct-
        # GGUF - every quant at Q6_K and above is 2-part there), so a
        # candidate's size must be the SUM across parts, not just its first
        # file - otherwise a split quant would look far smaller than it is
        # and never trigger the "too big for one device" floor it exists for.
        split = _split_c("big-split-quant", (37.0, 33.0))  # 70 GB combined
        assert split.size_bytes == int(70 * _GB)
        # .filename (what gets handed to GgufBackend/hf_hub_download) must be
        # the FIRST part, not the whole tuple.
        assert split.filename == "big-split-quant-00001-of-00002.gguf"

        picked = pick_candidate([split], single_free_bytes=48 * _GB,
                                 combined_free_bytes=96 * _GB)
        assert picked is split


class TestCandidateTableShape:
    def test_every_entry_is_a_nonempty_repo_and_nonempty_parts_tuple(self):
        for entry in model_selection.CANDIDATE_TABLE:
            assert len(entry) == 2
            repo_id, parts = entry
            assert repo_id.count("/") == 1 and repo_id.strip() == repo_id
            assert isinstance(parts, tuple) and len(parts) >= 1
            for part in parts:
                assert part.endswith(".gguf")

    def test_a_multi_part_entrys_parts_share_a_common_split_prefix(self):
        # Every split (2+-part) entry in the shipped table follows Hugging
        # Face's "NAME/NAME-NNNNN-of-MMMMM.gguf" split-GGUF convention, so
        # localm's own split-GGUF loader (localm/model_manager.py's
        # missing_split_parts) can find every sibling once all parts are
        # downloaded into the same directory.
        for _repo_id, parts in model_selection.CANDIDATE_TABLE:
            if len(parts) < 2:
                continue
            # Strip exactly the "-NNNNN-of-MMMMM.gguf" suffix (3 hyphen-
            # delimited segments) - rsplit's fixed maxsplit counts from the
            # right regardless of how many hyphens appear earlier in the
            # name (e.g. "Meta-Llama-3.1-70B-Instruct" has several), so this
            # isolates the part-number/total-count suffix correctly.
            prefixes = {p.rsplit("-", 3)[0] for p in parts}
            assert len(prefixes) == 1, (
                f"multi-part candidate {parts} does not share a common "
                f"split-name prefix")

    def test_no_duplicate_repo_parts_pairs(self):
        table = list(model_selection.CANDIDATE_TABLE)
        assert len(table) == len(set(table))
