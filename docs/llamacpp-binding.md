# llama.cpp ctypes Binding

`localm.inference.backends.llamacpp` is a pure-Python ctypes wrapper around the native `llama.dll`.  It replaces `llama-cpp-python` entirely: no C compiler, no Python wheel, no version lock.

## Module Layout

| File | Responsibility |
|---|---|
| `_loader.py` | DLL discovery, dependency-order loading, PATH extension |
| `_structs.py` | ctypes Structure definitions (sizes probed from the DLL) |
| `_abi.py` | Runtime ABI self-check: verifies the loaded DLL's struct layout before first use |
| `_api.py` | Low-level C API bindings (one Python function per C function) |
| `_symbols.py` | Resolves a C++-linkage export (MTP draft-head API) by reading the binary's own export table when a plain `getattr` lookup fails |
| `llama.py` | `LlamaCpp` public class + helpers |
| `mtmd.py` | Multimodal (vision) support: binds the bundled `mtmd.dll` for GGUF mmproj |
| `_runner.py` | Subprocess isolation for the whole GGUF model lifecycle (load, generate, tokenize, grammar-check, unload), so a native abort in the child kills only that child |
| `_worker.py` | `GgufWorker`: owns the real native model; runs only inside the isolated child process spawned by `_runner.py` |
| `_sizing.py` | VRAM measurement and load-sizing logic shared by `GgufBackend` (preflight checks before spawning) and `GgufWorker` (mid-generation context-grow checks) |
| `_vram_probe.py` | Standalone daemon entry point that answers the native ggml backend's VRAM-view query out-of-process, so a native abort inside the query cannot take down the caller |
| `__init__.py` | Exports `LlamaCpp` |

## DLL Loading (`_loader.py`)

The loader resolves the native binary directory from project-local locations
only - never a sibling folder elsewhere on disk - in this order:

1. `LLAMA_CPP_LIB` environment variable (explicit path to `llama.dll`, for
   one-off use)
2. the `binary_dir` config key in `<data dir>/config.json`
3. the `localm-llama-runtime` wheel bundled in this venv, populated by
   `localm setup-llama`

If none resolve, `load_lib()` raises with instructions to run `localm
setup-llama` (which downloads a prebuilt into the venv, or copies your own
build with `--from <dir>`).

Before loading `llama.dll`, the binary directory and the venv's bundled ROCm
runtime directories (the `rocm-sdk` wheels: amdhip64, rocm_kpack, rocblas, ...)
are added to the OS DLL search path, then all upstream ggml DLLs are pre-loaded
in dependency order so Windows symbol resolution succeeds:

```
ggml-base.dll → ggml-cpu.dll → ggml-hip.dll → ggml.dll → llama.dll
```

`load_lib()` is idempotent: it caches `_loaded_lib` and returns immediately on repeat calls.

## Struct Layouts (`_structs.py`)

Struct layouts were derived by probing `llama_model_default_params()` and `llama_context_default_params()` against known default values, then cross-referenced with `llama.h`.

upstream llama.cpp appends fields to the params structs several times a quarter
with no ABI or soname bump, so `_structs.py` stays safe two ways:

- it OVER-allocates the two by-value params structs (a named trailing field for
  what we know, plus a reserved pad), and the code round-trips
  `*_default_params()` (overwriting only the fields it names). A newer build
  therefore never reads past our buffer, and any field we do not name keeps its
  native default. A trailing field ADDITION is harmless.
- a mid-struct REORDER (the memory-corrupting kind of drift) is caught at load
  time by `_abi.verify_abi` (below), which refuses rather than corrupting memory.

The `sizeof` asserts in `_structs.py` are a self-consistency guard on our own
definitions; they do NOT validate against the loaded DLL.

### `LlamaModelParams` (72 bytes native on V1/V2, 80 on V3; all over-allocated to 104) - THREE layouts

upstream reordered this struct in place at an unchanged size (`main_gpu`
moved, `load_mode` inserted, three booleans folded into it), then inserted a
4-byte `lazy_mode` enum directly after `load_mode` at b10653 (spelled
`tensor_read_lazy` until b10679), so localm binds
`LlamaModelParamsV1` (<= lemonade b1288 / upstream b10103),
`LlamaModelParamsV2` (>= lemonade b1307 / upstream b10105..b10649) and
`LlamaModelParamsV3` (upstream >= b10653) and picks one per loaded library at
load time. There is deliberately no bare `LlamaModelParams` name - go through
`_abi.model_params_class()` / `_api.llama_model_default_params()`.

| Offset | Type | V1 field | V2 field | V3 field | Default |
|--------|------|----------|----------|----------|---------|
| 0 | ptr | `devices` | `devices` | `devices` | NULL |
| 8 | ptr | `tensor_buft_overrides` | `tensor_buft_overrides` | `tensor_buft_overrides` | NULL |
| 16 | i32 | `n_gpu_layers` | `n_gpu_layers` | `n_gpu_layers` | -1 (all) |
| 20 | i32 | `split_mode` | `split_mode` | `split_mode` | 1 (LAYER) |
| 24 | i32 | `main_gpu` | `load_mode` | `load_mode` | 0 / 1 (MMAP) / -1 (AUTO) |
| 28 | i32 | *(padding)* | `main_gpu` | `lazy_mode` | / 0 / 1 (AUTO) |
| 32 | ptr / i32 | `tensor_split` | `tensor_split` | `main_gpu` (+4 pad) | NULL / NULL / 0 |
| 40 | ptr | `progress_callback` | `progress_callback` | `tensor_split` | NULL |
| 48 | ptr | `progress_callback_user_data` | `progress_callback_user_data` | `progress_callback` | NULL |
| 56 | ptr | `kv_overrides` | `kv_overrides` | `progress_callback_user_data` | NULL |
| 64 | 8×bool / ptr | `vocab_only`, `use_mmap`, `use_direct_io`, `use_mlock`, `check_tensors`, `use_extra_bufts`, `no_host`, `no_alloc` | `vocab_only`, `check_tensors`, `use_extra_bufts`, `no_host`, `no_alloc`, `load_mtp` | `kv_overrides` | |
| 72-77 | 6×bool | | | `vocab_only`, `check_tensors`, `use_extra_bufts`, `no_host`, `no_alloc`, `load_mtp` | |

Use `_structs.set_use_mmap()` / `get_use_mmap()` rather than naming `use_mmap`
directly - it has no V2/V3 counterpart. `lazy_mode` (V3 only,
`LLAMA_LAZY_MODE_OFF/AUTO/ON` = 0/1/2) keeps the build's own default unless a
call site sets it.

### `LlamaContextParams` (152 bytes native on b1288; 160 on b9682+; 160 on
b10360+ with an inserted field; 168 on b11480+ with a second inserted field,
over-allocated to 224) - THREE layouts

upstream inserted a new `uint32_t` field, `n_outputs_max_per_seq`, directly
before `n_threads` sometime between lemonade b1307 (2026-08-04, confirmed
absent) and ggml-org b10360 (2026-08-11, confirmed present) - both are live in
production (the bundled AMD ROCm build vs. the fetched cuda/vulkan/cpu builds),
so localm binds `LlamaContextParamsV1` (no `n_outputs_max_per_seq`) and
`LlamaContextParamsV2` (with it) and picks one per loaded library, same
mechanism as `LlamaModelParams` above.

ggml-org b11480 then inserted a `size_t moe_cache_size` (device cache in bytes
for MoE experts kept in host memory, default 0 = disabled) directly after
`type_v`; b11479 is the last release without it. Every field from
`abort_callback` onward moved 8 bytes later, which is `LlamaContextParamsV3`.
No bare `LlamaContextParams` name - go through `_abi.context_params_class()` /
`_api.llama_context_default_params()`.

Key fields (every field is named identically in every layout that has it, so
call sites need no layout awareness at all):

| V1 offset | V2 offset | V3 offset | Type | Field | Default |
|-----------|-----------|-----------|------|-------|---------|
| 0 | 0 | 0 | u32 | `n_ctx` | 512 |
| 4 | 4 | 4 | u32 | `n_batch` | 2048 |
| - | 24 | 24 | u32 | `n_outputs_max_per_seq` | 1 |
| 24 | 28 | 28 | i32 | `n_threads` | -1 (auto) |
| 36 | 40 | 40 | i32 | `rope_scaling_type` | -1 (unspecified) |
| 48 | 52 | 52 | i32 | `flash_attn_type` | -1 (unspecified) |
| 80 | 84 | 84 | f32 | `defrag_thold` | -1.0 |
| - | - | 112 | size_t | `moe_cache_size` | 0 (disabled) |
| 112 | 112 | 120 | ptr | `abort_callback` | NULL |
| 128 | 128 | 136 | bool | `embeddings` | False |
| 129 | 129 | 137 | bool | `offload_kqv` | True |
| 131 | 131 | 139 | bool | `op_offload` | True |
| 152 | 152 | 160 | ptr | `ctx_other` | NULL |

b9682+ appended a trailing `ctx_other` (`struct llama_context *`), taking the
native struct to 160 bytes; localm names it and over-allocates to 224 for
headroom (unchanged by the V1/V2 split above: V2's extra 4-byte field exactly
offsets V1's now-unneeded 4-byte manual alignment pad before `cb_eval`; V3's
8-byte `moe_cache_size` comes out of the reserved pad, so all three layouts
total 224 bytes).

### `LlamaBatch` (56 bytes)

Matches the C layout exactly: `n_tokens` + 4 bytes padding + 6 pointers.

### `LlamaChatMessage` (16 bytes)

```c
typedef struct {
    const char * role;     // [0]
    const char * content;  // [8]
} llama_chat_message;
```

## Runtime ABI self-check (`_abi.py`)

`verify_abi(lib)` runs once inside `load_lib()`, right after the native library
loads and before any by-value struct crosses the FFI boundary. It first decides
WHICH of the three `LlamaModelParams` and (independently) WHICH of the three
`LlamaContextParams` layouts is loaded - `detect_model_params_layout()` uses two
independent signals: the `llama_load_mode_*` marker symbols split V1 from the
V2/V3 family (their absence with V2- or V3-shaped bytes is a refusal), and the
default-params value fingerprint splits V2 from V3, since the `lazy_mode`
insertion added no symbol; `detect_context_params_layout()` has no marker
symbol for either of its insertions, so it rests on value fingerprints alone:
the `ctx_type` / `-1` enum run splits V1 from the V2 family, and the boolean
flag block (at 128 on V2, 136 on V3) splits V2 from V3. Both fall back to their
historical V1 layout when inconclusive (the context axis falls back to V2 when
only the V2/V3 split is inconclusive), and callers must not treat that
fallback as a determination. `evaluate()` also refuses when the
`abort_callback`, `abort_callback_data` or `samplers` default reads non-NULL,
which is how a V2/V3 context misbind shows up. Under the `llama_load_mode_*`
symbols there is no fallback: default bytes that were read but match neither
V2 nor V3 are refused (every V2 build fingerprints conclusively, so such bytes
are a layout localm does not bind), while bytes that could not be read at all
bind V2 with a note.

It then calls `llama_model_default_params()` / `llama_context_default_params()`
(no model, no GPU needed) using the DETECTED classes and checks a structural
fingerprint of the returned defaults:

- the long-stable `*_UNSPECIFIED == -1` enums (`rope_scaling_type`,
  `pooling_type`, `attention_type`) - three consecutive `-1` int32s that a
  shifted layout essentially never reproduces (read at each layout's own
  correct offset, since the check is by field name, not raw offset);
- a valid `split_mode` (0/1/2/3 = NONE/LAYER/ROW/TENSOR) and ordered window sizes
  (`1 <= n_ubatch <= n_batch`, `n_ctx >= 1`, `n_seq_max >= 1`). Absolute size
  magnitudes are only a non-fatal diagnostic, so a future build that defaults
  higher is never refused.

The context_params LAYOUT DECISION itself (as opposed to the keystone check
above) scores each candidate layout out of 6: `ctx_type` at the position
immediately before the run is graded 0/1/2 (2 when it equals its own default
of 0, 1 when merely not -1, 0 when it is -1 and so falls inside the run
itself), plus up to 4 more points for however many of the FOUR consecutive
`rope_scaling_type`/`pooling_type`/`attention_type`/`flash_attn_type` reads
are exactly -1. Checking only three of the four (an earlier version of this
fingerprint) let a struct with exactly `rope_scaling_type` corrupted score
HIGHER under the wrong layout than the true one under its own - the field sits
at the position the other layout treats as `ctx_type`, and "not -1" is nearly
always true, so a wrong-but-plausible score could outscore a genuinely
partially-corrupted true one. All four fields close that gap.

On a proven mismatch it raises `AbiMismatch` (a reportable `LocalmError`) naming
the offending field, instead of letting a wrong layout corrupt memory. It is
deliberately false-positive-proof: only structural invariants and the `-1`
keystone gate the refusal, so a legitimate build whose *default values* drift
still loads (the drift is logged and shown by `localm doctor`). Two safety valves:

- it fails OPEN - if its own probe cannot run (a symbol missing on a very old
  build, a call raising), it logs and allows the load;
- `LOCALM_SKIP_ABI_CHECK=1` bypasses it entirely (logged), so a false alarm on an
  untested build can never permanently block a user.

The fingerprint was validated byte-for-byte against the cpu, vulkan, and amd-rocm
prebuilts localm provisions. Offsets for these POD fields are commit-determined,
not OS-determined, so a given build matches on every OS. Note that llama.cpp's
own build-tag namespaces collide: a bare `b1xxx` number can mean either the
lemonade-sdk/llamacpp-rocm AMD build or an unrelated ggml-org/llama.cpp tag - see
`_structs.py`'s docstring before quoting one. `localm doctor` surfaces the
verdict ("native ABI: ...") by running the check in a subprocess so a broken DLL
cannot crash the diagnostic.

## Checking against upstream (`scripts/check_llama_abi.py`)

A header-diff VERIFIER (not a generator). It parses `llama_model_params` /
`llama_context_params` / `llama_batch` out of a real `llama.h`, computes each
field's natural-alignment offset, and diffs them against `_structs.py`:

```
python scripts/check_llama_abi.py                 # ALL pinned refs (LLAMA_ABI_REFS["v1"], ["v2"], ["v3"], ["ctx_v3"])
python scripts/check_llama_abi.py --ref latest    # newest upstream release
python scripts/check_llama_abi.py --header path/to/llama.h
```

Checks `llama_model_params` and `llama_context_params` as two INDEPENDENT
layout axes (see above) - a header carrying, say, model_params v2 but
context_params v1 is diffed correctly against each struct's own matching
localm class, not assumed to move in lockstep. The no-arg default run fails
loudly if either axis's pinned refs stop straddling that axis's reorder.

A mid-struct reorder/insert exits non-zero; a purely trailing addition is a note
(it is absorbed by the reserved pad). A weekly CI job (`abi-check`) runs
`--ref latest` and also provisions the real cpu prebuilt to run `verify_abi`
against the actual binary.

### Enum domains

Layout and domain are different questions, and the layout half is structurally
blind to the other. An offset check reads WHERE a field sits; it cannot see
WHICH VALUES are legal in it. Upstream added `LLAMA_LOAD_MODE_AUTO = -1` between
b10361 and b10373 and made it the new default, so `llama_model_default_params()`
started returning -1 into a field whose offset had not moved by one byte.
localm's `_VALID_LOAD_MODES` did not list -1, so localm refused every build from
b10373 on while the weekly layout gate stayed green the whole time. It was not
broken; it was answering an adjacent question. **"Passes the ABI gate" is
therefore not the definition of a confirmed build.**

So the same run also diffs the DOMAIN of every enum localm binds, listed in
`_ENUM_BINDINGS`. Member values are read live out of the localm module that owns
them, never restated in the script, so the verifier compares against the same
constants the runtime uses. The two outcomes are deliberately distinct:

| upstream change | outcome | why |
|---|---|---|
| a NEW member localm does not bind | reported loudly, exit code unchanged | additive on its own. Hard-failing here would train people to widen localm's accept-sets to silence the gate, destroying the misaligned-read tripwire that reads them |
| a CHANGED VALUE for a member localm binds | non-zero exit | a number localm passes or accepts has changed meaning |

Two more cases it keeps apart, because a bare "the enum is not in this header"
cannot tell them apart: if the struct FIELD that uses the enum is also absent,
the header simply predates the feature and it is skipped (b9870 has neither
`llama_model_params.load_mode` nor `enum llama_load_mode`); if the field is
present and the enum is not, the domain is UNVERIFIED and that fails. A member
localm binds which the header lacks is a note rather than a failure, because the
pinned v2 ref b10360 legitimately predates `AUTO`.

The additive report is the b10373 detector, and it has a limit worth stating: a
new member is only dangerous once it becomes the DEFAULT, and a header cannot
show that (`llama.h` declares `llama_model_default_params` and never defines
it). On seeing that report, bind the member and then re-probe a real build's
`llama_*_default_params()`.

### Bumping the bundled build

When you change the prebuilt localm fetches (`DEFAULT_URL` or the pinned tag):

1. run `python scripts/check_llama_abi.py --ref <the build's tag>` and reconcile
   any reported field drift in `_structs.py`;
2. if a field was reordered or inserted mid-struct, update `_structs.py` to match
   (add a new layout + detection if a field's OFFSET moved for only some
   currently-shipped builds, not all - see `LlamaModelParamsV1`/`V2`/`V3` and
   `LlamaContextParamsV1`/`V2`/`V3` above for the pattern; teach EVERY site that
   discriminates layouts, including `_abi.model_params_class`, `evaluate`,
   `_structs.set_use_mmap` and `check_llama_abi.py`'s header classifier) and
   re-probe a real build; update the `_abi` anchors only if a keystone moved;
3. if it reports a NEW enum member, bind it (and add it to `_ENUM_BINDINGS`),
   then re-probe a real build's `llama_*_default_params()` - the header cannot
   tell you whether the new member became the default, which is the half that
   caused the b10373 outage;
4. bump the relevant entry in `LLAMA_ABI_REFS` in `scripts/check_llama_abi.py`.

## API Bindings (`_api.py`)

All functions are bound lazily via:

```python
def _bind(fn_name, restype, *argtypes):
    lib = load_lib()
    fn  = getattr(lib, fn_name)
    fn.restype  = restype
    fn.argtypes = list(argtypes)
    return fn
```

Covered functions (grouped):

**Lifecycle**: `llama_backend_init`, `llama_backend_free`  
**Model**: `llama_load_model_from_file`, `llama_free_model`, `llama_model_default_params`  
**Context**: `llama_init_from_model`, `llama_free`, `llama_context_default_params`  
**Accessors**: `llama_get_model`, `llama_n_ctx`, `llama_n_ctx_seq`, `llama_model_n_ctx_train`, `llama_model_n_embd`, `llama_model_n_layer`  
**Vocabulary**: `llama_model_get_vocab`, `llama_vocab_n_tokens`, `llama_n_vocab`, `llama_tokenize`, `llama_token_to_piece`, `llama_detokenize`, `llama_token_bos`, `llama_token_eos`, `llama_vocab_is_eog`, `llama_token_is_eog`  
**Chat template**: `llama_model_chat_template`, `llama_chat_apply_template`  
**Batch**: `llama_batch_get_one`, `llama_batch_init`, `llama_batch_free`  
**Inference**: `llama_decode`, `llama_get_logits_ith`, `llama_get_logits`  
**Embeddings** (export-probed via `has_embeddings_api()`): `llama_get_embeddings_seq`, `llama_get_embeddings_ith` - bound here but unused by `LlamaCpp`/`GgufBackend`; `llama_get_embeddings_seq` is called by the separate dedicated embedding-model loader (`localm.inference.embedder`, see Known Limitations), `llama_get_embeddings_ith` currently has no caller anywhere in the codebase  
**Model metadata**: `has_model_meta_api()` / `llama_model_meta_val_str`  
**Model introspection**: `has_kv_head_api()` / `llama_model_n_head` / `llama_model_n_head_kv`, `has_hybrid_api()` / `llama_model_is_recurrent` / `llama_model_is_hybrid`, `has_max_devices()` / `llama_max_devices`  
**Sampler chain**: `llama_sampler_chain_default_params`, `llama_sampler_chain_init`, `llama_sampler_chain_add`, `llama_sampler_free`, `llama_sampler_sample`, `llama_sampler_accept`, `llama_sampler_init_greedy`, `llama_sampler_init_dist`, `llama_sampler_init_top_k`, `llama_sampler_init_top_p`, `llama_sampler_init_min_p`, `llama_sampler_init_temp`, `llama_sampler_init_grammar`, `llama_sampler_init_grammar_lazy_patterns` (export-probed via `has_lazy_grammar()`), `llama_sampler_init_penalties` (export-probed via `has_penalties_sampler()`)  
**Memory (KV cache)**: `llama_get_memory`, `llama_memory_clear`, `llama_memory_seq_rm` (all probed at runtime via `has_memory_api()`), `llama_kv_cache_seq_rm` (a combined wrapper that prefers the memory API and falls back to the legacy call on an older DLL)  
**Multi-Token Prediction (MTP)**: `llama_model_mtp_support` / `llama_model_has_mtp` (plain export; whether an MTP draft context on this model would run a real draft head), `llama_set_embeddings_nextn`, `llama_get_embeddings_nextn` (declared without `extern "C"` in an internal header, resolved via `_symbols.py` rather than a plain `getattr`, and probed as a group via `mtp_hidden_state_available()`), `llama_get_embeddings_nextn_ith`, `llama_set_nextn_layer_offset` (same C++-linkage resolution, but each probed individually rather than as part of the group check) - see below  
**Diagnostics**: `llama_print_system_info`

## LlamaCpp Class (`llama.py`)

### Construction

```python
llm = LlamaCpp(
    model_path,
    n_ctx=4096,         # context window
    n_gpu_layers=99,    # layers to offload (99 = all)
    verbose=False,
    seed=0xFFFFFFFF,    # LLAMA_DEFAULT_SEED
    n_threads=None,     # None = auto
)
```

The constructor:
1. Calls `llama_model_default_params()`, sets `n_gpu_layers`, applies GPU-split/main-GPU placement
2. Calls `llama_backend_init()`, then loads the model
3. Calls `llama_context_default_params()`, sets `n_ctx`, `n_batch`, `offload_kqv`, creates context
4. Creates `_Tokenizer(model_ptr, ctx_ptr)`
5. If `mmproj_path` is given, best-effort loads it via `MtmdContext` (`mtmd.py`)
   for in-process vision; any failure leaves the model text-only rather than
   raising

### Chat template

`create_chat_completion` formats messages via `_apply_model_template(model_ptr, messages)`:

1. Calls `llama_model_chat_template(model_ptr)`: returns the Jinja template embedded in the GGUF
2. Builds a `LlamaChatMessage` ctypes array from the messages list
3. Calls `llama_chat_apply_template(tmpl, array, n, add_assistant=True, buf, buflen)`
4. If the template includes a BOS marker (`<bos>`, `<s>`, or a leading BOM character) at the start, skips `add_special` in tokenize to avoid doubling
5. Falls back to hardcoded ChatML if the model has no embedded template

### Generation loop (`_generate`)

KV cache strategy (probed at runtime via the `llama_memory_*` function
family):

- **Prefix reuse** (default on current DLLs): the common token prefix with
  the previous call stays in the KV cache; diverging cached tokens are
  removed with `llama_memory_seq_rm` and only the new suffix is prefilled.
  Follow-up chat turns skip re-evaluating the whole history.
  M-RoPE models (Qwen2-VL, Qwen2.5-VL, Qwen3-VL) reuse the prefix the same
  way; for an image prompt the cut never splits an image.
- **Full clear**: when the cache refuses to drop a range (a recurrent or
  hybrid model such as Qwen3.5), the cache is cleared and the whole prompt
  is prefilled into the same context.
- **Fresh rebuild** (old DLLs, or when the request outgrows the live context):
  the context is freed and re-created at the next dynamic-window size
  (`n_ctx_grow` steps up to `n_ctx_max`), then the full prompt is prefilled.
- Prefill is chunked to a fixed 2048-token constant, not the context's actual
  `n_batch` (which is `min(n_ctx, 2048)` and can be smaller on a small-context
  configuration): a single oversized `llama_decode` batch aborts the native
  process rather than returning an error.

```
loop:
    token = llama_sampler_sample(chain, ctx, -1)
    if llama_vocab_is_eog(vocab, token): break
    yield token
    feed token back (llama_batch_init + llama_decode)
```

(`llama_batch_get_one` is also bound in `_api.py`, but only the separate
embedding-model loader uses it - see Known Limitations below; the chat
generation loop always builds its one-token batch via `llama_batch_init`.)

Do NOT call `llama_sampler_accept` after `llama_sampler_sample`: sample()
already accepts the token into every stateful sampler in the chain. A second
accept advances the grammar sampler's parse state twice per token (it throws
`std::runtime_error` across the C ABI once its stacks empty - WinError
0xe06d7363) and double-counts the repetition-penalty window.

### Parallel slots (`_slots.py`)

`LlamaCpp(n_parallel=N)` with N above 1 creates its context with
`n_seq_max = N` and `kv_unified = true`: one KV cache of `n_ctx` cells shared
by N sequences, so the slots cost no extra KV memory (a model with recurrent
layers keeps one recurrent state per sequence, which the VRAM sizing charges).
The model keeps one slot when it is an encoder-decoder or diffusion model, when
a draft source is on, or when the runtime lacks the `llama_memory_*` API, and N
is lowered until it divides `n_ubatch`; `n_parallel` and `parallel_note` say
what it holds and why.

`_generate` then hands each text reply to the `SlotScheduler`: a reply gets a
free sequence (the one whose cached tokens share the longest prefix with its
prompt), its own sampler chain, and a reservation of `prompt + budget + 64`
cells. One scheduler thread makes every native call; each step decodes the
pending token of every running reply and prompt chunks of the replies still
prefilling, up to 2048 tokens, then samples each reply from its own row.

- A request that does not fit takes the cells of idle sequences first (least
  recently used), then grows the context in `n_ctx_grow` steps (re-decoding the
  running replies' tokens into the new context, and only when the VRAM check
  says the bigger cache fits on the GPU), and otherwise waits for the running
  replies, FIFO, with a `waiting` status every few seconds.
- A reply with no budget reserves 512 cells at a time and ends with `length`
  at the ceiling.
- Cancelling a reply (a closed stream, or a stop check that returns True)
  ends only that sequence.
- A failed combined decode is retried one sequence at a time, so only the
  failing reply ends.
- An image turn runs `_generate_image` inside `SlotScheduler.exclusive()`: it
  waits for the running replies, holds the model alone, and leaves the cache
  marked for clearing before the next text reply.

Replies decoded in one batch are not bit-identical to the same replies decoded
alone, and on a GPU the same batch can round differently from one run to the
next, so a greedy reply under concurrency can differ from the same request run
alone. A reply decoded alone, from a clean cache or from a prefix it decoded
alone, matches the one-slot model. A reply's logits do not depend on what the
replies beside it contain.

### Multi-Token Prediction (MTP) speculative decoding

Some models are trained with an extra "next-n" head that predicts more than
one token ahead (DeepSeek-V3/R1, the Qwen3.5/3.6 MTP family, Nemotron,
GLM-DSA, among others - see `MTP_GRAPH_ARCHITECTURES` below for the exact
set this runtime can drive). `LlamaCpp` can use that head to draft tokens
speculatively and verify them in the same pass as the next real token,
producing several tokens per verification when the drafts are accepted,
without a separate draft model.

**Off by default** - `mtp_enabled=False` (config key `mtp_enabled`, one
setting shared by `GgufBackend`/`GgufWorker`). Each step costs a small draft
decode per drafted token plus one verification batch, which pays only while
verifying several tokens costs about what verifying one does (the whole model
on the GPU). `localm bench-mtp` measures it per model. `mtp_draft_tokens`
(1-3, default 1) sets how many tokens one step drafts.

**Detection is a capability test, not a metadata test**
(`llama_model_mtp_support()` in `_api.py`). Both of these must hold:

- the GGUF declares MTP heads (`<arch>.nextn_predict_layers`, or a tolerated
  alias some third-party conversions use instead);
- the loaded llama.cpp build's model class for that architecture actually
  builds an MTP draft graph, per `MTP_GRAPH_ARCHITECTURES` in `_api.py` - a
  hand-derived allowlist pinned to the shipped runtime's build tag
  (`MTP_ARCH_SOURCE_TAG`) and kept honest by `scripts/check_mtp_arch_allowlist.py`,
  which fails when the pin moves without a re-derivation. Carrying the
  metadata does not imply this: GLM-4.5/4.5-Air/4.6 ship
  `nextn_predict_layers` and the NextN tensors, but the runtime's
  `build_arch_graph` ignores the MTP graph type for `glm4moe` and returns an
  ordinary decoder, so an MTP context there would be a second full decoder
  with its own VRAM-pinned KV cache rather than a draft head -
  `llama_model_mtp_support` refuses it instead of allocating one.

When both hold, `LlamaCpp` opens a second context, sized like the main one and
recreated whenever the main one grows, as the draft head's own KV cache. Two
more things gate whether it actually activates:

- **Feeding the hidden state.** The draft head predicts from the target
  model's hidden state at the previous position, not from the token
  embedding alone - fed only the embedding, fewer than one draft in ten was
  accepted, which does not repay the extra work. The API that supplies it
  (`llama_set_embeddings_nextn` / `llama_get_embeddings_nextn`) is declared
  in llama.cpp's internal `src/llama-ext.h` without `extern "C"`, so it is
  exported under a compiler-mangled name rather than a plain one.
  `_symbols.py` resolves it by reading the binary's own export table (MSVC
  or Itanium mangling) once a plain-name `getattr` lookup fails, rather than
  reading that failed lookup as "not exported" - the earlier mistake, which
  produced a written, incorrect conclusion that the shipped runtimes could
  not drive MTP at all. Fed correctly, a real MTP model accepts about half
  its drafts.
- **Rewinding a rejected draft.** Speculation writes a draft token into the
  cache and removes it again when the target rejects it. A cache holding
  recurrent state (the Qwen3.5/3.6 MTP family, Nemotron, DeepSeek V4) cannot
  be truncated at all unless it was asked to keep per-token snapshots, and
  rolling back r positions needs r snapshots. The context requests
  `max(2, mtp_draft_tokens)` of them (`n_rs_seq`, on builds whose context
  params struct has the field) whenever MTP is enabled, a no-op on a model
  with no recurrent layers. Each snapshot is one more copy of the recurrent
  state.

**Drafting and verification, per step.** The token just sampled sits at
position `pos`. One draft decode carries it, paired with the target's hidden
state for `pos - 1`, together with any accepted tokens the draft cache has not
seen yet (each paired with the hidden state of the position before it, with no
output row). Further drafts are decoded from the draft head's own next-n row,
up to `mtp_draft_tokens`, stopping at an end-of-generation token; the draft
cache is then trimmed back to `pos`. Drafts are picked greedily by a sampler
chain attached to the draft context with `llama_set_sampler`, so the choice is
made inside `llama_decode` and no vocabulary-sized logits row is copied out
(a runtime without backend sampling picks the same token on the CPU).

The token and its drafts are then decoded in ONE batch on the MAIN context,
and the REQUEST's own sampler chain - not a bare greedy sampler - samples each
row in turn: the drafts it agrees with are emitted, and at the first mismatch
its own token is emitted instead and the rejected rows are removed from the
main cache (`llama_memory_seq_rm`). Temperature, top-k/top-p and the
repetition penalty apply identically whether or not a draft is accepted, and
every token the sampler sees is a token that is emitted. Verifying several
tokens in one batch runs different kernels than decoding them one by one, and
their results can differ in the last bits, so where the two most likely tokens
are nearly tied (measured: a top-2 logit gap around 0.1) the reply can take the
other one. Up to three draft tokens no such divergence was seen on the test
models, which is why the setting stops at three. If the removal fails,
MTP is disabled for the rest of the loaded model's life (not just the current
generation) - speculation needs that rewind. **A grammar does not stop
drafting**: drafts are never accepted into the request's sampler, which only
ever samples (and so accepts) the tokens that are emitted, in order, so a
grammar or lazy grammar in that sampler sees exactly the sequence it would see
without MTP, and a draft the grammar forbids is simply rejected at
verification.

Prefill mirrors each main chunk into the draft cache with the hidden states
shifted by one position, and the first draft of a reply reads the hidden state
of the last prompt token.

**Pacing.** Whether speculation pays depends on the model (on a small model the
vocabulary-sized output layer makes a draft step and an extra verification row
relatively expensive), on how often drafts are accepted (lower when sampling
with a temperature) and on how busy the machine is. Each loaded model keeps a
`_DraftPacer` that measures the time per emitted token of speculative steps and
of plain one-token steps as the reply runs (one step in 24 runs plain to keep
that figure current, one in 3 until there are 6 plain figures). Over the last 16 steps of each kind, speculation costs the
median time of a speculative step divided by the mean number of tokens one
made available, and plain decoding the median time of a plain step. When
speculation is the slower of the two it is paused for 32 steps, doubling on
each consecutive pause up to 256, and then measured again. The pacer belongs
to the loaded model, so a pause can carry over into the next reply. Paused
steps of a reply that drafts do no draft-cache work; on the last paused step
the rows queued before the pause are flushed and the skipped positions are
mirrored, and a failing mirror stops drafting for that reply only. Paused steps
are reported per reply (`mtp_paused_steps`, `usage.mtp`).

**Why it declines**, recorded in `mtp_status`, logged
(`MTP: active=%s status=%s`) and returned as `usage.mtp.reason` when a reply
reports MTP `unavailable` or `stopped`:
`disabled` (config off); `native-refused` / `no-metadata-api` /
`no-mtp-metadata` / `unknown-architecture` / `no-mtp-graph:<arch>` (from
`llama_model_mtp_support` - the runtime or the GGUF's own declaration refuses
MTP for this model, distinct from the rewind case below);
`rewind-unsupported` (the draft KV cache could not be rewound after a
rejected draft, checked at load and again during generation);
`no-hidden-state-api` / `no-ctx-type-field` / `hidden-state-refused` /
`context-refused` (this runtime cannot build or feed a draft context);
`draft-context-full` (the conversation outgrew the draft context - ordinary
decoding continues, the reply does not stop); `draft-prefill-error:*`
/ `draft-prefill-failed:*` / `draft-trim-error:*` for a failed draft-side
prefill or cache trim; and `error:<ExceptionName>` for any other exception
raised while setting up the draft context. `Engine.supports_mtp` /
`GgufBackend.supports_mtp` reflect whether MTP is actually active for the
currently loaded model.

The child reports `mtp_status` and `mtp_active` at the end of EVERY call, not
just at load - `GgufBackend._record_mtp` takes that per-call state and, if
`mtp_status` is one of the statuses above that mean speculation has stopped for
the model's life, latches `supports_mtp` False from the next reply onward.
Without this, a session whose speculation stopped hours into a conversation
(draft context exhaustion, a failed rewind) would keep reporting the
capability it had at load time. `mtp_active` is the narrower, per-call answer:
False on a turn carrying an image even while `supports_mtp` stays True, since
the same model speculates normally on its next text turn.

Per call the child also reports `mtp_call_status` (why this reply stopped
drafting partway: `draft-decode-failed:*`, `draft-decode-error:*`,
`draft-catchup-failed:*`, `draft-catchup-error:*`, `draft-trim-failed`,
`draft-out-of-step`; the next reply drafts again) and `mtp_drafted` /
`mtp_accepted` (draft tokens sent to verification and kept).
`GgufBackend.last_mtp_usage` turns these into the `usage.mtp` object of the
chat API (see server-api.md), which the GUI shows next to the reply's tok/s.

### Draft sources: n-gram (prompt lookup) and draft-model speculative decoding

The decode loop in `LlamaCpp._generate` speculates through a `DraftSource`
(`_drafting.py`): `begin_call`, then per step `drafting` / `ready` / `budget` /
`propose`, then `after_verify` or `after_single_token`, `finish` and always
`end_call`. Verification is the same for every source: one batch of
`[token] + drafts`, each row sampled with the request's own sampler, the longest
matching prefix kept, the rest removed from the cache, and the first mismatch
carried to the next step as the token to emit. Output is the target model's,
whatever the source proposes.

`spec_source` chooses the source: `off`, `mtp` (the MTP head above), `ngram` or
`draft`.
Unset, it follows `mtp_enabled` (true is `mtp`, false is `off`); an explicit value
wins. One source is active per loaded model.

**`ngram`** (`_ngram.py`) needs no second model, no draft context and no extra
VRAM. For the token just sampled it looks up the longest trailing n-gram
(`NGRAM_N_MAX` 5 down to `NGRAM_N_MIN` 3 tokens) of the tokens in the cache plus
that token, finds its most recent earlier occurrence, and proposes what followed
it, up to `spec_draft_tokens` (default 8, at most 16). An end-of-generation token
is never proposed. The index follows the cache: it keeps the prefix a follow-up
turn shares with the previous one and indexes only what is new. A step with no
match decodes one token as a plain step, so a reply that never repeats itself
costs a dictionary lookup per token. It pays on text that repeats earlier text:
rewriting a file, quoting a passage, repeated tool-call JSON. At load its step
costs are measured as for a draft model (below, without the draft figures); a
target on which drafts accepted 90% of the time would not beat one-token
decoding by 5% turns it off with status `ngram-cannot-pay`. Each step looks
its candidate up first and then drafts the part of it with the most expected
tokens per second under the measured costs, none unless that beats a plain step
by 5%. Every candidate is checked token by token against the tokens the reply
goes on to hold, whether it was drafted or not, so how far candidates run before
a wrong token is learned without paying for drafts. The copy position is the
source position of the latest reply token a candidate token was found right
against, carried forward while each later reply token equals the next source
token. The counts are kept per draft position and per kind of candidate: where
its match ends relative to the copy position (on it: a copy continuing; 1 to 32
positions past it (`NGRAM_RESUME_GAP`): a copy picking up again after an edit,
such as a renamed identifier; otherwise a fresh match), and whether the match
lies in the context before the reply or in the reply itself. A kind starts from
a 10% chance per token of being wrong for a match in
the context and 40% for one in the reply; its counts over the model's life are
the prior of its counts in the current reply. So a copy that picks up again
after an edit drafts about as far as such copies have run before their next
edit (at most `spec_draft_tokens`), matches that keep going wrong (prose, code
written from scratch) are held back, and on a target whose verification
batches are dear, such as a Mixture-of-Experts model, a long draft is chosen
only where candidates are nearly always right. The speculation
report carries `runs` (per kind, the tokens a full-length candidate is expected
to yield) and `held_steps` (the reply's candidates held back) beside `costs`
and `observed_ms`. When the costs cannot be measured it proposes up to the cap
and the pacer alone decides.

**`draft`** (`_draftmodel.py`) drafts with a second, smaller GGUF named by
`spec_draft_model` (a registered model name or a path). The draft model must
share the target's vocabulary (`draft_vocab_mismatch`): the same tokenizer type,
the same add-BOS / add-EOS flags and the same BOS / EOS id where one is added,
sizes at most `DRAFT_VOCAB_SIZE_MAX_DIFFERENCE` (128) apart, and the same token
text for every id from `DRAFT_VOCAB_CHECK_START_ID` (5) up. It loads after the
target with every layer on the GPU, split over the same devices as the target
(the same `main_gpu` and split ratios), and drafts on its own context of the
main context's size, recreated when the main one grows. Its
cache follows the main cache lazily: each step keeps the prefix the two share,
removes the rest, decodes what is new plus the sampled token, then samples
drafts greedily, decoding each before sampling the next.

Once the draft model has loaded, its costs on this machine are measured
(`LlamaCpp._measure_step_costs`, a few dozen decodes): the target's one-token
decode, its verification batches of 2, 3, 5, 9 and 17 tokens up to
`spec_draft_tokens + 1` (others interpolated, the curve taken as never falling
as the batch grows), the draft model's one-token decode and its batched decode
per token. Each target figure is the median of 5 timed decodes after 3 warm-up
ones; a target whose one-token decode takes more than 20 ms gets 3 after 1, and
one taking more than 50 ms a single timed decode after 1 warm-up, of the
batches of 2 and `spec_draft_tokens + 1` only. A draft model that cannot beat
plain decoding at an acceptance of 0.85 (`DRAFT_GATE_ACCEPTANCE`) is freed with
status `draft-cannot-pay`. Each step then drafts the length k up to
`spec_draft_tokens` (default 8, at most 16) with the most expected tokens per
second, `(1 + p + ... + p^k) / cost(k)`; k 0 (a plain step) wins unless
drafting is expected to beat it by 5%. p is the fraction of drafts accepted,
each verification weighing 0.9 of the evidence before it, kept separately for a
step right after one whose drafts were all accepted, so a run of copied or easy
text drafts long; that estimate starts from the other one and ages while it
holds steps back. When the estimate says not to draft, a step after 32 held
steps since the last verification (and the first held step of a model)
drafts the length sized for p 0.9, if that length pays, so a draft that turns
good is noticed; each such probe that had a rejection and after which
drafting still does not pay doubles that interval for the rest of the reply,
up to 256. `cost(k)` follows
what steps actually take: the decode loop reports each step's seconds, and
each length keeps a running figure that starts at its modelled cost and moves
a fifth of the way to each new time, clipped to within 3 times the figure. A
step whose verification failed or that grew the context is not counted; nor,
for a draft model, is a step whose proposal failed, drafted nothing, or first
caught the draft cache up on more than 3 tokens (for an n-gram source such a
step counts as a plain step). A length not seen yet costs its measured
`k * draft + verify(k + 1)`, plus the overhead seen per plain step and a
drafting-step overhead fitted as fixed plus per draft to the lengths seen so
far. The verification cost curve is what makes this model aware: a
Mixture-of-Experts target, whose verification batch reads more experts per
extra token, gets shorter drafts than a dense one, and a target whose experts
sit in system RAM shorter still. A step skips drafting when catching the draft
cache up (for a CPU draft model after a long prompt, or a draft context
recreated for a grown main context) costs more than the rest of the reply is
expected to save. Without measurements a step drafts at most 2 tokens. The
figures are in the speculation report (`costs`, `observed_ms`, `acceptance`,
`acceptance_after_full_accept`, and `held_steps`, the reply's steps the costs
held back) and in the debug log. An end-of-generation draft ends the proposal. A
failed draft decode clears the draft cache and stops drafting for that reply; a
draft cache that cannot drop a rejected draft turns drafting off for the model.
The draft model is freed before the target. Its weights, its KV cache at the
main context's size, its logits buffer and a fixed margin are charged in the
VRAM estimate; under llama.cpp's implicit multi-GPU split that charge is
spread over the devices in proportion to their shares. The draft goes on the
GPU only when it fits beside the whole target (on an implicit split: when the
per-device plan with it keeps the target's split and fits every device);
otherwise it runs on the CPU, so it never costs the target layers or devices,
and the reply's usage reports `draft-on-cpu`. A draft model that is missing, is
not a causal chat model by its metadata (a draft head, a diffusion,
encoder-decoder, embedding, audio or image model; nothing is loaded), fails to
load, does not share the vocabulary, has recurrent layers, or whose context is
refused leaves the model working without drafting, with the status naming why
(`draft-model-missing`, `draft-unsupported-role`, `draft-load-failed`,
`draft-vocab-mismatch`, `draft-rewind-unsupported`, `draft-context-refused`).
`localm spec-drafts MODEL` lists the downloaded causal chat models whose
metadata passes the vocabulary rule.

A model with recurrent layers needs one state snapshot per draft token to drop
rejected drafts (`n_rs_seq`), so there the n-gram or draft-model draft length is
capped at 4 (`NGRAM_RECURRENT_DRAFT_TOKENS_MAX`) and the snapshots are charged in
the VRAM estimate. A cache that cannot drop a rejected draft at all is found at
load (status `rewind-unsupported`) or on the first rejection, after which
drafting stays off for that model. A turn with an image does not draft
(`skipped` `image`). `LlamaCpp.speculation_report()` carries the source, its
status and the reply's figures to the parent in the done envelope
(`speculation`), and `GgufBackend.last_speculation_usage` turns them into
`usage.speculation` (see server-api.md). `localm bench-spec MODEL` compares a
source against no speculation on this machine.

### Stop-string filter (`_filtered_stream`)

Many models signal end-of-turn with multi-token sequences (e.g. `<|im_end|>` → 6 tokens: `<`, `|`, `im`, `_`, `end`, `|>`). `llama_vocab_is_eog` can't catch these if they aren't registered as special tokens.

`_filtered_stream(pieces)` wraps the raw text piece stream:

- Buffers the last `max(len(s) for s in STOP_STRINGS) - 1` characters
- Yields the safe prefix immediately
- When a complete stop string appears anywhere in the buffer, yields the text before it and returns
- Flushes the remaining buffer at end-of-stream

Stop strings checked: `<|im_end|>`, `<end_of_turn>`, `<turn|>`, `<|eot_id|>`, `</s>`, `<|endoftext|>`, `[/INST]`, `<|end|>`.

### Sampler chain

`[grammar] → [penalties] → top_k(40) → top_p(0.95) → min_p(0.05) → temp(t) → dist(seed)`

- The GBNF grammar stage (`llama_sampler_init_grammar`) is added when a
  grammar string is supplied.
- A LAZY grammar (`llama_sampler_init_grammar_lazy_patterns`, export-probed via
  `has_lazy_grammar()`) is used instead when the caller passes `grammar_lazy=True`
  with `grammar_triggers`, and the DLL exports the lazy variant: generation
  stays unconstrained until the output matches a trigger pattern, then the
  grammar enforces from there (the "text-or-tool" mechanism, so a strict
  grammar never stalls a thinking model). Without triggers, or on an older
  DLL, the request is REFUSED with a `GrammarUnsupportedError` (HTTP 400)
  rather than silently generating unconstrained text - answering with a
  normal 200 the caller had every reason to believe was grammar-conformant
  would be worse than a clean refusal.
- The repetition-penalty stage is added when `repeat_penalty != 1.0` and
  the DLL exports `llama_sampler_init_penalties`.
- For `temperature <= 0` the stochastic stages are replaced by `greedy`
  (grammar and penalties still apply).

### Output filtering

`_filtered_stream` halts on stop strings (above); `_scrub_stream` then
removes internal model markers (thinking-channel tags, reserved placeholder
tokens). Chat output is ALWAYS scrubbed - debug mode does not skip this. In
debug mode (`LOCALM_DEBUG`) the raw, pre-scrub text is additionally written
to the debug log, except in privacy mode, where chat content is never
persisted.

## Known Limitations

- **Single sequence only**: `_create_batch` always calls `llama_batch_init(n, 0, 1)` -
  the hardcoded final `1` is `n_seq_max`
- **No embedding extraction via the chat class**: `LlamaCpp`/`GgufBackend` never
  call the embedding accessors, so `GgufBackend.embed` raises
  `NotImplementedError` and `GgufBackend.can_embed` is `False`. The low-level
  binding itself does have a working embeddings path (`has_embeddings_api()` /
  `llama_get_embeddings_seq` in `_api.py`) - it is used by a separate,
  dedicated on-device embedding-model loader (`localm.inference.embedder`),
  loaded independently of whatever chat model is active. HF-format models
  embed fine. (See server-api.md for the `/v1/embeddings` behavior.)
