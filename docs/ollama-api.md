# Ollama-compatible API

localm answers the Ollama HTTP API on the same server and port as its
OpenAI-compatible API, so a tool that speaks Ollama can use a localm model
without an adapter. The routes are checked against the official `ollama`
Python client; other clients (Open WebUI, Home Assistant, Continue, Zed) use
the same wire format, but one that depends on tool calling or on a JSON-schema
`format` will not work yet (see the end of this page).

```bash
localm serve <model> --port 11434     # Ollama's default port, for clients that assume it
```

Point the client at `http://127.0.0.1:<port>` and use a model name from
`GET /api/tags`.

## How it is mounted

Each Ollama route is an exact path on the existing server: there is no prefix,
no second listener and no proxy. localm's own routes also live under `/api/`
(for example `/api/session`, `/api/plugins` and the GUI's `/api/models`); none
uses an Ollama path, and a test fails if one ever does. The origin and
open-mode exemptions are exact paths too, so `/api/embed` being callable by
local apps does not extend to the GUI's `/api/embedding/warmup`.

Chat and generate requests are translated to a `/v1/chat/completions` request
and answered by the same code path, so capability routing, routing to another
instance, context compaction, audit and the per-model request queue behave
exactly as they do on `/v1`.

## Authentication

The same API keys and scopes apply. With no key configured the server is open,
like the rest of the API.

| Route | Needs |
|---|---|
| `POST /api/chat`, `/api/generate`, `/api/embed`, `/api/embeddings`, `GET /api/version` | any valid key |
| `GET /api/tags`, `GET /api/ps`, `POST /api/show` | `models:read` |
| `POST /api/copy`, `POST /api/pull`, `/api/push`, `/api/create`, `DELETE /api/delete`, `/api/blobs/{digest}` | `models:write` |

In open mode the inference routes, `/api/show` and the three read-only GETs are
callable by any local app, like their `/v1` counterparts. The model-management
routes keep the open-mode requirement of the GUI shell token (or an API key).

Send the key as `Authorization: Bearer <key>`. Most Ollama clients support this
(`OLLAMA_API_KEY` for the Python client, the API key field in Open WebUI).

## Errors

Errors on these routes are `{"error": "<message>"}`, the shape Ollama clients
read. A malformed request is a 400, an unknown model a 404, a refused
credential a 401 or 403, and a failed generation a 500. A stream that fails
after it has started ends with one `{"error": "..."}` line. The one exception is
the open-mode origin and shell-token refusal of a model-management route, which
keeps the `{"detail": "..."}` body the rest of localm's API uses.

## Chat and generate

`POST /api/chat` and `POST /api/generate` stream NDJSON by default (`stream`
defaults to true, as in Ollama); send `"stream": false` for one JSON document.

| Request field | Behaviour |
|---|---|
| `model` | Required. A trailing `:latest` is dropped when the bare name is registered. |
| `messages`, `prompt`, `system` | Mapped to a chat conversation. `/api/generate` sends `system`, then `prompt`, as a chat. |
| `images` | Base64 images, on a message or on `/api/generate`. Needs a vision model. |
| `options.temperature`, `top_p`, `top_k`, `repeat_penalty`, `seed` | Passed to the sampler. |
| `options.num_predict` | A cap of 1 or more sets the maximum number of new tokens; -1 and -2 mean no cap. |
| `options.stop` | A string or a list. Sent as the request's `stop` (see [server-api.md](server-api.md)): the reply is cut before the first match, the generation ends there and `done_reason` is `stop`. |
| `format` | `"json"` constrains the reply to a JSON object. A JSON schema is refused with a 400. |
| `think` | `true` (or a level string) returns the model's reasoning in `message.thinking` (`thinking` on generate); `false` turns reasoning off. |
| `keep_alive` | On a request with no messages or prompt: `0` unloads the model, anything else loads it (a model served by another instance is left to that instance). Unloading needs `models:write`; with no API key configured it needs the GUI shell token, like `POST /v1/models/unload`. Ignored on a normal request; localm manages residency itself (`idle_unload_seconds`). |
| other `options` keys, unknown request keys | Accepted and ignored (named in the debug log). |

Refused with a 400, rather than silently dropped: `tools` and `tool_calls`,
`raw`, `suffix`, a request-level `template`, and a non-empty `context`.

Each reply object carries `model`, `created_at`, the content (`message` or
`response`) and `done`. The last one adds `done_reason` (`stop` or `length`)
and the timing fields `total_duration`, `prompt_eval_count`,
`prompt_eval_duration`, `eval_count` and `eval_duration` in nanoseconds.
`prompt_eval_duration` is the time to the first token and `eval_duration` the
decode time implied by the measured tokens per second; a field with no
measurement behind it is left out. `load_duration` is never sent. A reply cut by
a stop sequence reports `total_duration` only.

## Embeddings

`POST /api/embed` takes `input` (a string or a list) and returns `embeddings`.
`POST /api/embeddings` takes `prompt` and returns `embedding`. The vectors are
whatever the embedding model produces; they are not renormalised.
`prompt_eval_count` is sent only when the backend counted the input tokens. `dimensions`
is refused with a 400, and `truncate` is accepted and ignored.

## Listings

- `GET /api/tags` lists the registered chat and embedding models, with
  `name`, `model`, `size`, `modified_at`, `digest` (the recorded SHA-256, or
  empty) and `details` (`format`, and `family` when the model's architecture is
  recorded). `parameter_size` and `quantization_level` are empty.
- `GET /api/ps` lists the loaded models with `context_length` (the loaded
  window) and `expires_at` (the idle-unload time, or a date in 2318 when no idle
  timeout is set). `size_vram` is not reported.
- `POST /api/show` returns `details`, `model_info` (`general.architecture` and
  `<architecture>.context_length` when recorded), `capabilities`
  (`completion` or `embedding`, plus `vision` and `thinking` when known) and
  empty `modelfile`, `parameters` and `template`. `tools` is not advertised.
- `GET /api/version` returns localm's own version.

## Model management

`POST /api/copy` registers a second name for a model (an alias, the same file).
`/api/pull`, `/api/push`, `/api/create`, `/api/delete` and `/api/blobs/{digest}`
answer 501 with the localm equivalent in the message: `localm pull`,
`localm rm`, or the GUI.

## Not supported yet

Tool calling (`tools`, `tool_calls`) and `format` as a JSON schema.
