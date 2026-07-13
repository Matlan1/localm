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

`ttft_ms` (time to first token) is reported on streaming responses only; a non-streaming response still returns the token counts and `tokens_per_sec`, but no `ttft_ms`.

When the memory plugin is active and recalls facts for a turn, the response also carries an `X-Localm-Memory` header (see [Memory endpoints](#memory-endpoints-memory-plugin)).

Multimodal input uses the standard multipart content format with base64
data-URIs (`{"type": "image_url", "image_url": {"url": "data:image/..."}}`)
and requires a vision-capable model (a GGUF paired with a multimodal projector
via `localm run --mmproj` / `localm pull --mmproj`, or a HuggingFace-format
vision model). A GGUF loaded without an mmproj is text-only and rejects an
attached image with a clear error.

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
Both wait for any in-flight generation to finish first. Unloading is
implicit-recovery: the next chat request reloads automatically.

## Management endpoints

### `GET /v1/config` / `PATCH /v1/config`

Read and update `~/.localm/config.json`. PATCH accepts only known keys and
persists immediately; engine values (context sizes, GPU layers) apply on the
next model load.

## Plugin management endpoints

These endpoints are present under `localm serve` too, not just `localm gui`.
They are scope-gated: `GET` requires `PLUGINS_READ`, the mutations require
`PLUGINS_ADMIN`.

### `GET /api/plugins`

Returns `{plugins: [...], errors: {...}}`. Each plugin entry carries the
state flags `installed`, `enabled`, `active`, `available`, and `loaded`,
plus `name`, `version`, `description`, `scope`, `builtin`, `protected`,
`tab`, `label`, `icon`, `group`, `client_entry`, `assets_base`, `requires`,
`requires_extras`, `extra`, and `error`.

### `POST /api/plugins/{name}/install` / `POST /api/plugins/{name}/uninstall`

Move a plugin between the available catalog and the installed set. Install
copies the plugin from the store into `~/.localm/plugins/` and enables it;
uninstall removes the installed directory and accepts `?delete_data=` to
also drop the plugin's stored data. Uninstalling a protected plugin (chat)
returns 409; an unknown plugin returns 404.

### `POST /api/plugins/refresh` / `POST /api/plugins/{name}/refresh`

Refresh plugin state from disk without a server restart. The bare route
refreshes every installed plugin and returns `{"status": "ok", "refreshed":
...}`; the `{name}` route refreshes just that plugin and returns
`{"status": "refreshed"|"up-to-date", "name": name}`. An unknown or
non-builtin plugin returns 404; any other refresh failure returns 400.

### `POST /api/plugins/{name}/enable` / `POST /api/plugins/{name}/disable`

Toggle an installed plugin active or inactive in config. Disabling a
protected plugin (chat) returns 409; enabling a plugin that is not installed
returns 409; an unknown plugin returns 404.

## Memory endpoints (memory plugin)

These routes exist only when the `memory` plugin is installed and enabled;
otherwise they `404`. Memory is owner-scoped and off in privacy mode: every
write returns `403` in privacy mode (no new traces). See [memory.md](memory.md).

| Route | Purpose |
|---|---|
| `GET /api/memory` | The current facts (text + per-item metadata) and whether writes are allowed. |
| `PUT /api/memory` | Bulk edit. A line matching an existing fact keeps that record; new lines are added; omitted lines are deleted. `413` past the 256-record or 64k-char cap. |
| `POST /api/memory/append` | Add one fact. |
| `PATCH /api/memory/{id}` | Edit one record's text or importance. |
| `DELETE /api/memory/{id}` | Delete one record. |
| `POST /api/memory/consolidate` | Distil durable facts from recent sessions now; `503` until a model is loaded. |
| `GET /api/memory/forgotten` | Archived (forgotten) records available for recovery - read-only, available even in privacy mode. |
| `POST /api/memory/forgotten/{id}/restore` | Recover one archived record back into the live store; `404` if no matching archive entry exists. |

When the memory plugin recalls facts for a chat turn, `POST /v1/chat/completions`
returns an `X-Localm-Memory` response header: a compact JSON blob with the count
of injected facts, the items, and a degrade reason when semantic recall could not
be used (for example `no_embedder` before `localm setup-embeddings`).

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
- **GUI endpoints**: `localm gui` adds further `/api/*` routes (coder sessions,
  model switching, image jobs) on top of this API; the `/api/plugins*`
  management routes above are already part of the base server. See [gui.md](gui.md).
