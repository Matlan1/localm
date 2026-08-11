# llama.cpp ctypes Binding

`localm.inference.backends.llamacpp` is a pure-Python ctypes wrapper around the native `llama.dll`.  It replaces `llama-cpp-python` entirely: no C compiler, no Python wheel, no version lock.

## Module Layout

| File | Responsibility |
|---|---|
| `_loader.py` | DLL discovery, dependency-order loading, PATH extension |
| `_structs.py` | ctypes Structure definitions (sizes probed from the DLL) |
| `_abi.py` | Runtime ABI self-check: verifies the loaded DLL's struct layout before first use |
| `_api.py` | Low-level C API bindings (one Python function per C function) |
| `llama.py` | `LlamaCpp` public class + helpers |
| `mtmd.py` | Multimodal (vision) support: binds the bundled `mtmd.dll` for GGUF mmproj |
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

### `LlamaModelParams` (72 bytes native, over-allocated to 104) - TWO layouts

upstream reordered this struct in place at an unchanged size (`main_gpu`
moved, `load_mode` inserted, three booleans folded into it), so localm binds
`LlamaModelParamsV1` (<= lemonade b1288 / upstream b10103) and
`LlamaModelParamsV2` (>= lemonade b1307 / upstream b10105) and picks one per
loaded library at load time. There is deliberately no bare `LlamaModelParams`
name - go through `_abi.model_params_class()` / `_api.llama_model_default_params()`.

| Offset | Type | V1 field | V2 field | Default |
|--------|------|----------|----------|---------|
| 0 | ptr | `devices` | `devices` | NULL |
| 8 | ptr | `tensor_buft_overrides` | `tensor_buft_overrides` | NULL |
| 16 | i32 | `n_gpu_layers` | `n_gpu_layers` | -1 (all) |
| 20 | i32 | `split_mode` | `split_mode` | 1 (LAYER) |
| 24 | i32 | `main_gpu` | `load_mode` | 0 / 1 (MMAP) |
| 28 | i32 | *(padding)* | `main_gpu` | / 0 |
| 32 | ptr | `tensor_split` | `tensor_split` | static default |
| 40 | ptr | `progress_callback` | `progress_callback` | NULL |
| 48 | ptr | `progress_callback_user_data` | `progress_callback_user_data` | NULL |
| 56 | ptr | `kv_overrides` | `kv_overrides` | NULL |
| 64-71 | 8×bool | `vocab_only`, `use_mmap`, `use_direct_io`, `use_mlock`, `check_tensors`, `use_extra_bufts`, `no_host`, `no_alloc` | `vocab_only`, `check_tensors`, `use_extra_bufts`, `no_host`, `no_alloc`, `load_mtp` | |

Use `_structs.set_use_mmap()` / `get_use_mmap()` rather than naming `use_mmap`
directly - it has no V2 counterpart.

### `LlamaContextParams` (152 bytes native on b1288; 160 on b9682+; 160 on
b10360+ with an inserted field, over-allocated to 224) - TWO layouts

upstream inserted a new `uint32_t` field, `n_outputs_max_per_seq`, directly
before `n_threads` sometime between lemonade b1307 (2026-08-04, confirmed
absent) and ggml-org b10360 (2026-08-11, confirmed present) - both are live in
production (the bundled AMD ROCm build vs. the fetched cuda/vulkan/cpu builds),
so localm binds `LlamaContextParamsV1` (no `n_outputs_max_per_seq`) and
`LlamaContextParamsV2` (with it) and picks one per loaded library, same
mechanism as `LlamaModelParams` above. No bare `LlamaContextParams` name - go
through `_abi.context_params_class()` / `_api.llama_context_default_params()`.

Key fields (everything before `n_threads` and everything from `cb_eval`
onward is named identically in both, so most call sites need no V1/V2
awareness at all):

| V1 offset | V2 offset | Type | Field | Default |
|-----------|-----------|------|-------|---------|
| 0 | 0 | u32 | `n_ctx` | 512 |
| 4 | 4 | u32 | `n_batch` | 2048 |
| - | 24 | u32 | `n_outputs_max_per_seq` | 1 |
| 24 | 28 | i32 | `n_threads` | -1 (auto) |
| 36 | 40 | i32 | `rope_scaling_type` | -1 (unspecified) |
| 48 | 52 | i32 | `flash_attn_type` | -1 (unspecified) |
| 80 | 84 | f32 | `defrag_thold` | -1.0 |
| 128 | 128 | bool | `embeddings` | False |
| 129 | 129 | bool | `offload_kqv` | True |
| 131 | 131 | bool | `op_offload` | True |

b9682+ appended a trailing `ctx_other` (`struct llama_context *`), taking the
native struct to 160 bytes; localm names it and over-allocates to 224 for
headroom (unchanged by the V1/V2 split above: V2's extra 4-byte field exactly
offsets V1's now-unneeded 4-byte manual alignment pad before `cb_eval`, so
both layouts total 224 bytes).

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
WHICH of the two `LlamaModelParams` and (independently) WHICH of the two
`LlamaContextParams` layouts is loaded - `detect_model_params_layout()` uses two
independent signals (the `llama_load_mode_*` marker symbols, plus a value
fingerprint as corroboration); `detect_context_params_layout()` has no marker
symbol for its insertion, so it rests on a value fingerprint alone. Both fall
back to their historical V1 layout when inconclusive, and callers must not treat
that fallback as a determination.

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
above) scores each candidate layout out of 5: `ctx_type` at the position
immediately before the run is-not-(-1), plus all FOUR consecutive
`rope_scaling_type`/`pooling_type`/`attention_type`/`flash_attn_type` reads
being exactly -1. Checking only three of the four (an earlier version of this
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
python scripts/check_llama_abi.py                 # BOTH pinned refs (LLAMA_ABI_REFS["v1"], ["v2"])
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

### Bumping the bundled build

When you change the prebuilt localm fetches (`DEFAULT_URL` or the pinned tag):

1. run `python scripts/check_llama_abi.py --ref <the build's tag>` and reconcile
   any reported field drift in `_structs.py`;
2. if a field was reordered or inserted mid-struct, update `_structs.py` to match
   (add a V2 layout + detection if a field's OFFSET moved for only some
   currently-shipped builds, not all - see `LlamaContextParamsV1`/`V2` above for
   the pattern) and re-probe a real build; update the `_abi` anchors only if a
   keystone moved;
3. bump the relevant entry in `LLAMA_ABI_REFS` in `scripts/check_llama_abi.py`.

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
**Accessors**: `llama_get_model`, `llama_n_ctx`, `llama_model_n_ctx_train`, `llama_model_n_embd`  
**Vocabulary**: `llama_model_get_vocab`, `llama_vocab_n_tokens`, `llama_tokenize`, `llama_token_to_piece`, `llama_detokenize`, `llama_token_bos`, `llama_token_eos`, `llama_vocab_is_eog`  
**Chat template**: `llama_model_chat_template`, `llama_chat_apply_template`  
**Batch**: `llama_batch_get_one`, `llama_batch_init`, `llama_batch_free`  
**Inference**: `llama_decode`, `llama_get_logits_ith`, `llama_get_logits`  
**Embeddings** (export-probed via `has_embeddings_api()`): `llama_get_embeddings_seq`, `llama_get_embeddings_ith` - bound here but unused by `LlamaCpp`/`GgufBackend`; the only caller is the separate dedicated embedding-model loader (`localm.inference.embedder`, see Known Limitations)  
**Sampler chain**: `llama_sampler_chain_init`, `llama_sampler_chain_add`, `llama_sampler_free`, `llama_sampler_sample`, `llama_sampler_accept`, `llama_sampler_init_greedy`, `llama_sampler_init_dist`, `llama_sampler_init_top_k`, `llama_sampler_init_top_p`, `llama_sampler_init_min_p`, `llama_sampler_init_temp`, `llama_sampler_init_grammar`, `llama_sampler_init_grammar_lazy_patterns` (export-probed via `has_lazy_grammar()`), `llama_sampler_init_penalties` (export-probed via `has_penalties_sampler()`)  
**Memory (KV cache)**: `llama_get_memory`, `llama_memory_clear`, `llama_memory_seq_rm` (all probed at runtime via `has_memory_api()`)  
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
1. Calls `llama_backend_init()`
2. Calls `llama_model_default_params()`, sets `n_gpu_layers`, loads model
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
4. If the template includes a BOS marker (`<bos>`, `<s>`) at the start, skips `add_special` in tokenize to avoid doubling
5. Falls back to hardcoded ChatML if the model has no embedded template

### Generation loop (`_generate`)

KV cache strategy (probed at runtime via the `llama_memory_*` function
family):

- **Prefix reuse** (default on current DLLs): the common token prefix with
  the previous call stays in the KV cache; diverging cached tokens are
  removed with `llama_memory_seq_rm` and only the new suffix is prefilled.
  Follow-up chat turns skip re-evaluating the whole history.
- **Fresh rebuild** (old DLLs, or when the request outgrows the live
  context): the context is freed and re-created at the next dynamic-window
  size (`n_ctx_grow` steps up to `n_ctx_max`), then the full prompt is
  prefilled.
- Prefill is always chunked to `n_batch` (2048): a single oversized
  `llama_decode` batch aborts the native process rather than returning an
  error.

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
  DLL, a lazy request generates unconstrained rather than silently becoming a
  strict grammar.
- The repetition-penalty stage is added when `repeat_penalty != 1.0` and
  the DLL exports `llama_sampler_init_penalties`.
- For `temperature <= 0` the stochastic stages are replaced by `greedy`
  (grammar and penalties still apply).

### Output filtering

`_filtered_stream` halts on stop strings (above); `_scrub_stream` then
removes internal model markers (thinking-channel tags, reserved placeholder
tokens) unless debug mode is active.

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
- **No speculative decoding / draft models**
