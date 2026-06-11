# HTTP Server API

localm's inference server exposes an OpenAI-compatible REST API plus a small
set of localm-specific management endpoints.

```bash
localm serve <model>          # default 127.0.0.1:8642, auto-bumps when busy
```

Authentication: open when `LOCALM_API_KEY` is unset (local development).
When set, endpoints require `Authorization: Bearer <key>`. CORS is locked to
localhost by default; widen it with the `cors_origins` config key. See
[tls.md](tls.md) before exposing the server beyond 127.0.0.1.

## OpenAI-compatible endpoints

### `POST /v1/chat/completions`

Streaming and non-streaming chat. Standard OpenAI request body, plus localm
extras:

| Field | Notes |
|---|---|
| `top_k`, `repeat_penalty` | extra sampling controls |
| `seed` | reproducible generation |
| `grammar` | GBNF grammar constraining the output (local models) |

```json
{
  "model": "mymodel",
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "stream": true,
  "temperature": 0.8,
  "seed": 42
}
```

The final usage block carries exact token counts from the model's own
tokenizer plus performance numbers:

```json
{
  "usage": {
    "prompt_tokens": 14, "completion_tokens": 3, "total_tokens": 17,
    "ttft_ms": 181.4, "tokens_per_sec": 38.2
  }
}
```

Multimodal input uses the standard multipart content format with base64
data-URIs (`{"type": "image_url", "image_url": {"url": "data:image/..."}}`)
and requires a model loaded with `--mmproj`.

### `POST /v1/completions`

Raw text completion (streaming and non-streaming), same extras as chat.

### `POST /v1/embeddings`

```json
{"model": "mymodel", "input": ["text one", "text two"]}
```

Returns OpenAI-format embedding vectors. 422 when the loaded model cannot
embed.

### `GET /v1/models`

Lists the currently served model. `GET /v1/models/{id}` returns registry
detail for any registered model: path, source, size, SHA256, aliases, and
whether it is active and loaded.

### `GET /health`

200 with model name and load state; 503 when no engine is initialised.

## Model lifecycle

`POST /v1/models/unload` releases the model from VRAM (e.g. before image
generation hands the GPU to ComfyUI); `POST /v1/models/load` reloads it.
Unloading is implicit-recovery: the next chat request reloads automatically.

## Management endpoints

### `GET /v1/config` / `PATCH /v1/config`

Read and update `~/.localm/config.json`. PATCH accepts only known keys and
persists immediately; engine values (context sizes, GPU layers) apply on the
next model load.

### `GET /v1/plugins` / `POST /v1/plugins/install` / `DELETE /v1/plugins/{name}`

List installed external plugins (with manifest errors surfaced), install
from a local directory containing `plugin.toml`, remove by name.

## Client example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8642/v1", api_key="localm")

stream = client.chat.completions.create(
    model="mymodel",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## Behaviour notes

- **Concurrency**: inference is serialised through a semaphore; concurrent
  requests queue in order. GPU memory is shared and the KV cache is not
  concurrency-safe, so this is deliberate.
- **Context**: the window starts at `n_ctx` and grows on demand up to
  `n_ctx_max` (see the dynamic context section of the README). Conversations
  that outgrow the ceiling get a clear error instead of an OOM.
- **GUI endpoints**: `localm gui` adds `/api/*` routes (coder sessions,
  model switching, image jobs) on top of this API; see [gui.md](gui.md).
