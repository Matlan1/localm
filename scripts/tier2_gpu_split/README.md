# Tier 2 GPU-split real-hardware gate

Closes the still-open half of `issues/issues.txt`'s `GPU-SPLIT-TESTING` entry.
Tier 1 (`tests/test_gpu_split_native_vulkan.py`) proves the native split path
works using a real GPU plus a free *software* second Vulkan device (Mesa
lavapipe) - by design it cannot touch real VRAM pressure, real allocator/OOM
behavior, or the `amd-rocm`/HIP backend at all (no software HIP implementation
exists). Tier 2 is a real 2-GPU rental that closes exactly that gap, run as a
pre-release gate rather than an open-ended manual session.

**Full background**: `dev-notes/split-gpu-testing-research-2026-07-13.md` (the
original ranked research) and `dev-notes/cloud-gpu-testing-providers-2026-07-16.md`
(the provider correction below).

## A correction this harness makes

Both `issues/issues.txt` and the 2026-07-13 research doc still name **Vast.ai**
at ~$0.30-0.55/run. That is superseded by the 2026-07-16 doc: Vast.ai is a
peer-to-peer marketplace (a stranger's machine), unacceptable for a tool that
downloads models and runs local inference. This harness targets reputable,
datacenter-owned providers instead, at a correspondingly higher real cost
(~$2-6/run, not $0.30-0.55) - see the cost table below.

## Instance specs to rent

| Run | Provider | Instance | Rate (verified live 2026-07-29 - confirm before paying, prices drift) | Image |
|---|---|---|---|---|
| NVIDIA / `vulkan` backend | [Lambda Cloud](https://cloud.lambda.ai/) | 2x A6000 (primary); 2x A100 40GB (fallback if unavailable - Lambda capacity is often tight) | A6000 ~$0.80/GPU-hr ($1.60/hr for 2x); A100 ~$1.99/GPU-hr ($3.98/hr for 2x) | Lambda's standard on-demand Ubuntu 22.04 |
| AMD / `amd-rocm` backend | [Hot Aisle](https://hotaisle.xyz/) | 2x MI300X | $2.99/GPU-hr -> ~$5.98/hr (up from an earlier ~$4 estimate - noted as price drift) | Hot Aisle's Ubuntu ROCm image |

Both Linux-only. AMD-on-Windows-multi-GPU is not a gap this harness needs to
close: no datacenter rents a Windows-capable AMD card, and the maintainer
deprioritized that cell on 2026-07-16 (see the provider doc).

## Automation honesty - what is and is not scripted

- **Lambda (NVIDIA run)**: `run_gate.py --launch lambda` calls Lambda Cloud's
  launch/terminate REST API directly
  (`https://cloud.lambdalabs.com/api/v1/instance-operations/{launch,terminate}`,
  HTTP Basic auth with the API key as username). This shape is long-stable and
  publicly documented, but **it was not exercised against a live account in
  this session** (no key available while building this harness) - your first
  real run is also the first live check of this integration. It fails loudly
  (prints the raw non-2xx response body, exits non-zero) rather than silently
  guessing on drift.
- **Hot Aisle (AMD run)**: Hot Aisle's exact API shape could not be confidently
  verified in this session (search results were inconclusive). Rather than
  script a guessed integration against real money, launch/terminate is a
  **manual step** (their dashboard/quick-start) - `run_gate.py --launch manual
  --ssh-host <ip>` picks up from there. Provisioning, tests, the timeout, and
  the teardown reminder are identical either way; only the "click launch" /
  "click terminate" steps are manual for this provider.

## Usage

```bash
# Dry run first - real HF size lookups, real command rendering, NO ssh/cloud
# calls/charges. Do this before ever spending money.
python scripts/tier2_gpu_split/run_gate.py --backend vulkan --dry-run

# NVIDIA/vulkan run, Lambda launches+terminates the instance:
export LAMBDA_API_KEY=...
python scripts/tier2_gpu_split/run_gate.py --backend vulkan --launch lambda \
    --ssh-key-name my-lambda-key --local-ssh-key ~/.ssh/my-lambda-key \
    --ref master

# AMD/amd-rocm run - launch the Hot Aisle box by hand first, then:
python scripts/tier2_gpu_split/run_gate.py --backend amd-rocm --launch manual \
    --ssh-host <ip> --ssh-user ubuntu --local-ssh-key ~/.ssh/hotaisle_key \
    --ref master
```

Exit codes: `0` all tests passed, and teardown was either confirmed (Lambda)
or not applicable for this launch mode (manual/Hot Aisle always prints its
own "MANUAL TEARDOWN REQUIRED" reminder regardless - check the log); `1` same,
but a test failed; `2` a harness/provisioning error before a pass/fail could
be determined; `3` teardown was attempted (Lambda only) and definitively
**failed to confirm**, or the launch request may have created an instance but
never returned its id (a possible untracked orphan) - a real billing risk,
reported regardless of test outcome; `130` the run was interrupted by
Ctrl-C/SIGTERM (the conventional 128+SIGINT code) - teardown was attempted
before exiting, read the log for whether it confirmed. **Check the provider
dashboard yourself immediately on exit code 3 or 130.**

## Cost ceiling and timeout

`run_gate.py` prints the cost ceiling before doing anything, computed from the
**worst-case rate** - the pricier fallback instance type when one exists (Lambda
silently substitutes it whenever the primary type has no capacity, with no
re-confirmation - see the instance table above) times `--timeout-minutes / 60`
- and requires typing `y` (or `--yes`) to proceed. The `--timeout-minutes`
wall-clock budget (default 60) is the actual enforcement mechanism: every ssh
and Lambda-API step is bounded by the time actually remaining in that budget
(never a larger fixed literal - see `_bounded_timeout()` and its unit tests in
`tests/test_tier2_run_gate.py`, added after review caught a sub-call's fixed
timeout that could exceed its caller's own remaining deadline), and teardown
always runs afterward regardless of how the run ended (pass, fail, timeout,
harness error, or Ctrl-C/SIGTERM) - see `_run()`'s `try/except Exception`,
its idempotent/reentry-safe `teardown()`, and its `SIGINT`/`SIGTERM` handlers
in `run_gate.py`. A confirmed 45-60 minute run costs roughly $1.20-3.00
(NVIDIA at the primary A6000 rate; up to $3.98/hr if the pricier A100
fallback is used) or $4.50-6.00 (AMD) at the rates above.

## The test list and PASS/FAIL oracles (decided in advance)

All five live in `tests/test_gpu_split_real_hardware.py`, gated behind the
`real_multi_gpu_hardware` pytest marker and `LOCALM_TEST_REAL_MULTI_GPU=1` (see
`tests/conftest.py`) so they only ever run during an actual Tier 2 gate. Every
test reads real per-device VRAM via `localm.discover.list_gpus()`/
`vram_capacity()` - only `localm.config.load_config` is mocked (to inject
`gpu_split_indices`/`gpu_split_ratios`), exactly like
`tests/test_gpu_split_native_vulkan.py`'s established style. Tests 1/4/5
construct a real `LlamaCpp` directly (it applies the split itself, in
`__init__`); tests 2/3 deliberately construct `GgufBackend` instead and call
its `.load()` - `_check_vram()` (the preflight both tests exist to exercise)
is only ever invoked from there, never from a bare `LlamaCpp` construction, a
distinction review caught before either test ever ran for real (see each
test's docstring for the full story).

| # | Test | Proves | PASS | FAIL |
|---|---|---|---|---|
| 1 | `test_split_load_honors_configured_ratio` | Real allocator behavior: a lopsided 9:1 split actually lands on the configured devices | Device 0's independently-measured (nvidia-smi/rocm-smi) VRAM-delta share >= 70%; coherent completion | Even/reversed split, no delta, a hang/crash, or garbage output |
| 2 | `test_combined_vram_budgeting_admits_model_too_big_for_one_device` | The #770 combined-VRAM-budgeting fix holds under REAL VRAM pressure - the 2026-07-21 checkup's own words: "NOT live-verified on real 2-GPU hardware... this is exactly the Tier 2 gap" | A live-HF-size-picked model too big for one device but fitting combined loads without refusing, `applied_split_device_count() >= 2`, coherent completion | `RuntimeError` refusal, or a split that degraded to one device |
| 3 | `test_genuine_oom_refused_cleanly_at_split_boundary` | A genuinely over-combined-capacity request is refused safely, fast, cleanly | `RuntimeError` containing "cannot fit across this split" within 30s; GPU/driver still healthy afterward (a trivial follow-up load succeeds) | A hang past 30s, a crash, or a wrong/misleading message |
| 4 | `test_amd_rocm_hip_split_path_executes` | The amd-rocm/HIP multi-device path - never executed once, on any hardware, ever - does not immediately fail | Both real AMD devices show a nonzero VRAM delta; coherent completion | All layers on one device, a load error, or garbage output |
| 5 | `test_adversarial_uneven_ratio_and_short_device_refusal` | The research doc's requested adversarial configs: an extreme ratio, and `gpu_split_shortfall()`'s refuse-when-short path against real numbers | A 99:1 ratio skews harder than test 1's floor; a pinned split sized to exceed one device's real free VRAM produces a non-empty shortfall | No skew, or a silent proceed on a provably-too-small device |

Test 3 is deliberately scoped to the **Python-side preflight**
(`_check_vram()`, `localm/inference/backends/llamacpp/_sizing.py:591-642`), not
a forced native-allocator crash. The worker-subprocess isolation in
`localm/inference/backends/llamacpp/_worker.py` exists precisely so a real
native OOM/crash cannot take down the parent process, but deliberately trying
to trigger that on a *metered, rented* box is a materially higher-risk exercise
than proving the documented preflight refusal that already guards against it.
A follow-on test that forces a genuine native-level OOM through the worker
path (asserting the worker dies cleanly and the parent reports it, rather than
hanging) is a reasonable next step, not built here.

`tests/test_tier2_model_selection.py` unit-tests `model_selection.py`'s pure
candidate-picking logic with fabricated sizes - no network or GPU needed.
`tests/test_tier2_run_gate.py` unit-tests `run_gate.py`'s timeout-bounding
arithmetic and its Lambda-launch-orphan exit-code handling with a fake clock
and mocked ssh/Lambda calls - also no network or GPU. Both run in the normal
suite today and are the pieces of this harness's own logic provable in full
without the rental.

The model a Tier 2 run actually needs for test 2 is picked from
`model_selection.CANDIDATE_TABLE`, which includes real **split-GGUF** quants
(Q6_K, ~53.9 GB; Q8_0, ~69.8 GB, both of a 70B model) alongside single-file
ones - confirmed live that Hugging Face ships any 70B quant at Q6_K or above
as 2+ sibling files, not one, and localm already loads split GGUFs in
production (`localm/model_manager.py`'s `missing_split_parts`). A quant big
enough to genuinely exceed a single A6000's 48 GB free VRAM needed this: the
table originally topped out at 39.6 GB (Q4_K_M), which structurally could
never trigger the combined-budgeting test on the harness's own recommended
instance - caught by review, not by running the actual rental.

## What is proven right now, and what remains unexecuted

**Proven without the rental** (see the PR/commit this shipped in for the actual
command output):
- `run_gate.py --dry-run` for both backends: real HF candidate-size lookups
  (including the real, live-confirmed split-part sizes above), real
  remote-command rendering.
- `tests/test_tier2_model_selection.py` and `tests/test_tier2_run_gate.py`:
  real assertions on the pure selection logic and the timeout-bounding
  arithmetic/orphan-handling logic.
- `tests/test_gpu_split_real_hardware.py` collects cleanly and skips
  informatively without `LOCALM_TEST_REAL_MULTI_GPU` set (and would skip just
  as informatively on a real box with fewer than 2 detected GPUs).
- The existing Tier 1 lavapipe test still passes on this tree.
- The full local suite, `npm test`, and hygiene all pass.
- **Lint**: `ruff check` on every new/changed Python file passes. The new
  `provision_remote.sh` was syntax-checked with `bash -n` (passes) - shellcheck
  itself is not installed anywhere in this environment (not on PATH, not in
  either available WSL distro, not under any common install location checked),
  is not run in CI, and is not wired into `check_hygiene.py`, so shell-specific
  quoting/portability analysis was **not** performed. Said plainly rather than
  folded into an unscoped "lint passes" claim.
- A multi-dimension adversarial code review of this entire harness (correctness,
  security, cost-safety, hygiene, completeness-vs-spec) ran before this was
  considered done, and found - among other things - both the split-GGUF sizing
  gap and the `GgufBackend`-vs-`LlamaCpp` test-routing bug described above,
  plus the timeout-bounding and Lambda-orphan-instance fixes now reflected in
  `run_gate.py`. See `dev-notes/TIER2-GPU-SPLIT-HARNESS-2026-07-29.md` for the
  full finding list.

**Still unexecuted - needs the maintainer's cloud account and payment method**:
both real rental runs (NVIDIA/Lambda and AMD/Hot Aisle). Neither has ever been
run. Each costs roughly the table above's hourly rate times the run's
wall-clock time - budget ~$2-6 total for a first NVIDIA run and ~$5-6 for a
first AMD run (a bit more than a bare 45-60 minutes is realistic the very
first time, while confirming the Lambda API integration actually matches this
harness's assumptions and while ROCm-torch-index auto-detection in
`provision_remote.sh` is proven for real). The `amd-rocm`/HIP split path in
particular has never been executed on any hardware, ever, before this harness
runs it for the first time.
