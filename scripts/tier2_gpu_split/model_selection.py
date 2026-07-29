#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pick a real GGUF that is too big for one split device but fits the combined
free VRAM of the split - the model the Tier 2 combined-VRAM-budgeting test
(tests/test_gpu_split_real_hardware.py) needs, sized against whatever GPUs the
rented box actually granted (A6000 vs A100 vs MI300X all differ), not a
hardcoded guess.

``pick_candidate`` is pure (no I/O) and is what tests/test_tier2_model_selection.py
exercises with fabricated sizes - it can be proven correct without a network
connection or real hardware. ``resolve_live_sizes`` is the only impure part: it
asks the Hugging Face Hub for each candidate's REAL file size(s)
(files_metadata=True), so this never depends on a hardcoded size estimate going
stale. A candidate repo/file that 404s or omits a size is skipped with a
printed warning, not silently dropped (AGENTS.md rule 5) - the caller still
gets every candidate it could resolve.

SPLIT GGUFS: a quant big enough to genuinely exceed a single device's free
VRAM on the harness's recommended hardware (2x A6000, 48 GB each) usually
exceeds Hugging Face's practical single-file size too, so it ships as several
sibling parts (e.g. ``NAME/NAME-00001-of-00002.gguf``), not one file -
confirmed live against bartowski/Meta-Llama-3.1-70B-Instruct-GGUF: every quant
at Q6_K and above is 2-part. localm already loads split GGUFs in production
(``missing_split_parts()``, ``localm/model_manager.py``), so a ``Candidate``
here holds ALL of a quant's parts (a single-file quant is simply a
one-element tuple) and the caller downloads every part before loading the
first.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence


class Candidate(NamedTuple):
    repo_id: str
    parts: tuple  # repo-relative paths, in download/load order (1+ entries)
    size_bytes: int  # summed across all parts

    @property
    def filename(self) -> str:
        """The first part - what to hand to GgufBackend/hf_hub_download as
        the model path. For a single-file candidate this is its only file;
        for a split quant, llama.cpp's own split-GGUF loader finds the
        remaining parts alongside it once every part has been downloaded."""
        return self.parts[0]


# Ordered smallest-first (matters only for iteration order / early-exit below,
# not for correctness - pick_candidate always returns the smallest fitting
# entry regardless of input order). All bartowski quantizations - a prolific,
# long-established GGUF quantizer on Hugging Face; repo/file existence for the
# 8B and 70B entries was confirmed live 2026-07-29 (including the split-part
# layout of the Q6_K/Q8_0 entries, queried directly via the Hub API, not
# guessed from a rendered file-tree page), the middle single-file entries
# follow the same repo's established naming convention but were not each
# individually confirmed - resolve_live_sizes() finds out for real at run
# time either way, and a missing/renamed part just drops that one candidate.
#
# The table must reach ABOVE a single device's free VRAM on the harness's own
# recommended instance (2x A6000, 48 GB each - see README.md), not just look
# plausible: review caught that the table originally topped out at 39.6 GB
# (Q4_K_M) with no split-GGUF awareness at all, which never clears a ~48 GB
# single-device floor at the _EXCEEDS_SINGLE_MARGIN below - a silent
# pytest.skip on every real run against the recommended hardware. The Q6_K
# (~53.9 GB) and Q8_0 (~69.9 GB) split-quant entries below exist specifically
# to clear that floor while staying under a realistic combined ceiling (2x
# A6000 = 96 GB, 2x A100 40GB fallback = 80 GB).
CANDIDATE_TABLE = (
    ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
     ("Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",)),
    ("bartowski/Qwen2.5-14B-Instruct-GGUF",
     ("Qwen2.5-14B-Instruct-Q8_0.gguf",)),
    ("bartowski/Qwen2.5-32B-Instruct-GGUF",
     ("Qwen2.5-32B-Instruct-Q5_K_M.gguf",)),
    ("bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
     ("Meta-Llama-3.1-70B-Instruct-Q2_K.gguf",)),
    ("bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
     ("Meta-Llama-3.1-70B-Instruct-Q3_K_M.gguf",)),
    ("bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
     ("Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",)),
    ("bartowski/Meta-Llama-3.1-70B-Instruct-GGUF", (
        "Meta-Llama-3.1-70B-Instruct-Q6_K/Meta-Llama-3.1-70B-Instruct-Q6_K-00001-of-00002.gguf",
        "Meta-Llama-3.1-70B-Instruct-Q6_K/Meta-Llama-3.1-70B-Instruct-Q6_K-00002-of-00002.gguf",
    )),
    ("bartowski/Meta-Llama-3.1-70B-Instruct-GGUF", (
        "Meta-Llama-3.1-70B-Instruct-Q8_0/Meta-Llama-3.1-70B-Instruct-Q8_0-00001-of-00002.gguf",
        "Meta-Llama-3.1-70B-Instruct-Q8_0/Meta-Llama-3.1-70B-Instruct-Q8_0-00002-of-00002.gguf",
    )),
)

# A candidate must clear the single-device free figure by this margin (not just
# barely exceed it - measurement noise on a live box makes "barely over" an
# unreliable boundary) and stay under the combined free figure by this much
# headroom (KV cache + compute buffers + the other device's own base
# allocations still need room after the weights land).
_EXCEEDS_SINGLE_MARGIN = 1.05
_FITS_COMBINED_MARGIN = 0.85


def pick_candidate(
    sized_candidates: Sequence[Candidate],
    single_free_bytes: int,
    combined_free_bytes: int,
) -> Optional[Candidate]:
    """The smallest candidate that is unambiguously too big for one device
    (>= single_free * 1.05) but comfortably fits the combined pool
    (<= combined_free * 0.85). None if nothing in the table spans that range -
    an honest "no candidate fits this box", never a forced wrong-sized pick."""
    single_floor = single_free_bytes * _EXCEEDS_SINGLE_MARGIN
    combined_ceiling = combined_free_bytes * _FITS_COMBINED_MARGIN
    fitting = [
        c for c in sized_candidates
        if single_floor <= c.size_bytes <= combined_ceiling
    ]
    if not fitting:
        return None
    return min(fitting, key=lambda c: c.size_bytes)


def resolve_live_sizes(
    candidates: Sequence[tuple] = CANDIDATE_TABLE,
) -> list:
    """Query the Hugging Face Hub for each candidate's REAL total size (summed
    across every part). Groups lookups by repo_id so multiple candidates
    sharing one repo (most of CANDIDATE_TABLE does) cost one model_info() call,
    not one per candidate. Never raises on a single candidate's or repo's
    failure (a renamed/removed file, a network blip) - that candidate is
    skipped with a printed warning so the caller still gets every size it
    could resolve, and the skip is visible rather than silent (AGENTS.md
    rule 5)."""
    from huggingface_hub import HfApi

    api = HfApi()
    by_repo: dict = {}
    for repo_id, parts in candidates:
        by_repo.setdefault(repo_id, []).append(tuple(parts))

    resolved = []
    for repo_id, parts_list in by_repo.items():
        try:
            info = api.model_info(repo_id, files_metadata=True)
        except Exception as e:  # noqa: BLE001 - one repo's failure must not
            # abort resolution of every other repo's candidates.
            print(f"  (skipping every candidate in {repo_id}: {e})")
            continue
        sizes_by_name = {s.rfilename: s.size for s in info.siblings}
        for parts in parts_list:
            part_sizes = [sizes_by_name.get(p) for p in parts]
            if any(sz is None for sz in part_sizes):
                missing = [p for p, sz in zip(parts, part_sizes) if sz is None]
                print(f"  (skipping {repo_id}/{parts[0]}: part(s) not found "
                      f"or no size reported by the Hub: {missing})")
                continue
            resolved.append(Candidate(repo_id, parts, sum(part_sizes)))
    return resolved


def select_model_for_combined_test(
    single_free_bytes: int, combined_free_bytes: int,
) -> Optional[Candidate]:
    """Convenience wrapper: resolve real sizes, then pick. This is what the
    hardware-gated test in tests/test_gpu_split_real_hardware.py calls; the
    pure pick_candidate() above is what tests/test_tier2_model_selection.py
    proves correct without a network connection."""
    sized = resolve_live_sizes()
    return pick_candidate(sized, single_free_bytes, combined_free_bytes)
