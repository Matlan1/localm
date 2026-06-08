# llama.cpp ctypes Binding

`localm.inference.backends.llamacpp` is a pure-Python ctypes wrapper around the native `llama.dll`.  It replaces `llama-cpp-python` entirely — no C compiler, no Python wheel, no version lock.

## Module Layout

| File | Responsibility |
|---|---|
| `_loader.py` | DLL discovery, dependency-order loading, PATH extension |
| `_structs.py` | ctypes Structure definitions (sizes probed from the DLL) |
| `_api.py` | Low-level C API bindings (one Python function per C function) |
| `llama.py` | `LlamaCpp` public class + helpers |
| `__init__.py` | Exports `LlamaCpp` |

## DLL Loading (`_loader.py`)

The loader searches for `llama.dll` in this order:

1. `LLAMA_CPP_LIB` environment variable (explicit path)
2. `D:\projects\llama-gfx1030-prebuilt\` (AMD gfx1030 prebuilt)
3. `D:\projects\llama.cpp\build\bin\` (local build)

Before loading `llama.dll`, all upstream ggml DLLs are pre-loaded in dependency order so Windows symbol resolution succeeds:

```
ggml.dll → ggml-base.dll → ggml-cpu.dll → ggml-hip.dll → llama.dll
```

The binary directory is prepended to `os.environ["PATH"]` so Windows finds any further runtime DLLs (e.g. HIP runtime) automatically.

`load_lib()` is idempotent — it caches `_loaded_lib` and returns immediately on repeat calls.

## Struct Layouts (`_structs.py`)

Struct layouts were derived by probing `llama_model_default_params()` and `llama_context_default_params()` against known default values, then cross-referenced with `llama.h`.

### `LlamaModelParams` (72 bytes)

| Offset | Type | Field | Default |
|--------|------|-------|---------|
| 0 | ptr | `devices` | NULL |
| 8 | ptr | `tensor_buft_overrides` | NULL |
| 16 | i32 | `n_gpu_layers` | -1 (all) |
| 20 | i32 | `split_mode` | 1 (LAYER) |
| 24 | i32 | `main_gpu` | 0 |
| 28 | i32 | *(padding)* | |
| 32 | ptr | `tensor_split` | static default |
| 40 | ptr | `progress_callback` | NULL |
| 48 | ptr | `progress_callback_user_data` | NULL |
| 56 | ptr | `kv_overrides` | NULL |
| 64–71 | 8×bool | flags: `vocab_only`, `use_mmap`, `use_direct_io`, `use_mlock`, `check_tensors`, `use_extra_bufts`, `no_host`, `no_alloc` | |

### `LlamaContextParams` (152 bytes)

Key fields:

| Offset | Type | Field | Default |
|--------|------|-------|---------|
| 0 | u32 | `n_ctx` | 512 |
| 4 | u32 | `n_batch` | 2048 |
| 24 | i32 | `n_threads` | -1 (auto) |
| 48 | i32 | `flash_attn_type` | -1 (unspecified) |
| 80 | f32 | `defrag_thold` | -1.0 |
| 128 | bool | `embeddings` | False |
| 129 | bool | `offload_kqv` | True |
| 131 | bool | `op_offload` | True |

Size asserts in `_structs.py` will catch layout drift at import time.

### `LlamaBatch` (56 bytes)

Matches the C layout exactly — `n_tokens` + 4 bytes padding + 6 pointers.

### `LlamaChatMessage` (16 bytes)

```c
typedef struct {
    const char * role;     // [0]
    const char * content;  // [8]
} llama_chat_message;
```

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
**Sampler chain**: `llama_sampler_chain_init`, `llama_sampler_chain_add`, `llama_sampler_free`, `llama_sampler_sample`, `llama_sampler_accept`, `llama_sampler_init_greedy`, `llama_sampler_init_dist`, `llama_sampler_init_top_k`, `llama_sampler_init_top_p`, `llama_sampler_init_min_p`, `llama_sampler_init_temp`  
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

### Chat template

`create_chat_completion` formats messages via `_apply_model_template(model_ptr, messages)`:

1. Calls `llama_model_chat_template(model_ptr)` — returns the Jinja template embedded in the GGUF
2. Builds a `LlamaChatMessage` ctypes array from the messages list
3. Calls `llama_chat_apply_template(tmpl, array, n, add_assistant=True, buf, buflen)`
4. If the template includes a BOS marker (`<bos>`, `<s>`) at the start, skips `add_special` in tokenize to avoid doubling
5. Falls back to hardcoded ChatML if the model has no embedded template

### Generation loop (`_generate`)

The KV cache is cleared by re-creating a fresh context before each call (the prebuilt DLL lacks `llama_kv_self_clear`):

```
llama_free(ctx)
→ llama_init_from_model(model, params)
→ prefill prompt in one batch (llama_batch_get_one + llama_decode)
→ loop:
    token = llama_sampler_sample(chain, ctx, -1)
    llama_sampler_accept(chain, token)
    if llama_vocab_is_eog(vocab, token): break
    yield token
    feed token back (llama_batch_get_one + llama_decode)
```

### Stop-string filter (`_filtered_stream`)

Many models signal end-of-turn with multi-token sequences (e.g. `<|im_end|>` → 6 tokens: `<`, `|`, `im`, `_`, `end`, `|>`). `llama_vocab_is_eog` can't catch these if they aren't registered as special tokens.

`_filtered_stream(pieces)` wraps the raw text piece stream:

- Buffers the last `max(len(s) for s in STOP_STRINGS) - 1` characters
- Yields the safe prefix immediately
- When a complete stop string appears anywhere in the buffer, yields the text before it and returns
- Flushes the remaining buffer at end-of-stream

Stop strings checked: `<|im_end|>`, `<end_of_turn>`, `<|eot_id|>`, `</s>`, `<|endoftext|>`, `[/INST]`, `<|end|>`.

### Sampler chain

For `temperature > 0`:  `top_k(40) → top_p(0.95) → min_p(0.05) → temp(t) → dist(seed)`  
For `temperature == 0`: `greedy`

## Known Limitations

- **No KV cache reuse across calls** — context is recreated for each `_generate()` call. Cost is ~10–50 ms per call depending on n_ctx.
- **Single sequence only** — `n_seq_max=1` (the `llama_batch_get_one` path)
- **No grammar sampling** — `llama_sampler_init_grammar` not currently bound
- **No embedding extraction** — `embeddings=False` in context params
