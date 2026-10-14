# Ollama-compatible API

localm answers the Ollama HTTP API on the same server and port as its
OpenAI-compatible API, so a tool that speaks Ollama can use a localm model
without an adapter. The routes are checked against the official `ollama`
Python client; other clients (Open WebUI, Home Assistant, Continue, Zed) use
the same wire format. Tool calling is supported on `/api/chat`.

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
read. A malformed request (including a body that fails validation) is a 400, an
unknown model a 404, a refused credential a 401 or 403, a taken name or a model
in use a 409, a model-management route localm does not serve a 501, and a failed
generation a 500. A chat, generate or embedding request that the OpenAI-shaped
route behind it refuses keeps that route's status (for example 413 for a prompt
longer than the context window, 422 for a model that cannot embed, 503 while a
model cannot be loaded). A reply from that route that is not JSON, and an
embedding model that returns the wrong number of vectors, are a 502. A stream
that fails after it has started ends with one `{"error": "..."}` line. The one
exception is the open-mode origin and shell-token refusal of a model-management
route, which keeps the `{"detail": "..."}` body the rest of localm's API uses.

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
| `format` | `"json"` constrains the reply to a JSON object. A JSON schema constrains it to that schema: `type`, `enum`, `const`, `properties`, `required`, `additionalProperties`, `items`, `prefixItems`, `minItems`, `maxItems`, `minLength`, `maxLength`, integer `minimum` and `maximum`, `format` (`date`, `time`, `date-time`, `uuid`), `anyOf`, `oneOf`, `allOf` and local `$ref` (recursive schemas included). Properties come out in the order the schema lists them, and an object with listed properties has no others unless `additionalProperties` allows it. A keyword the grammar cannot enforce (`pattern`, `multipleOf`, `not`, number bounds, ...) is a 400 naming it. |
| `tools` | `/api/chat` only. Function tools, as Ollama sends them. The model's calls come back in `message.tool_calls` (`function.name`, `function.arguments` as an object, `id`, `function.index`); a streamed call is its own line before the `done` line, and `done_reason` is `stop`. An earlier turn's `assistant` message with `tool_calls` and the `tool` messages after it are given back to the model; they are matched to the calls by order, so `tool_name` is accepted and not needed. The rules in [Tool calling](server-api.md#tool-calling) apply, with `tool_choice` always `auto`. |
| `think` | `true` (or a level string) returns the model's reasoning in `message.thinking` (`thinking` on generate); `false` turns reasoning off. |
| `keep_alive` | On `/api/chat` with no messages, or `/api/generate` with no prompt and no images: a zero (`0`, `"0"`, or a zero duration such as `"0s"`) unloads the model, anything else loads it. The reply is `{"model", "created_at", "message": {"role": "assistant", "content": ""}, "done": true, "done_reason": "unload"}` (`"load"` for a load; `"response": ""` in place of `message` on generate). A model that another instance serves is left to that instance, for a load and for an unload. Unloading needs `models:write` (a 403 without it; with no API key configured it needs the GUI shell token, like `POST /v1/models/unload`), an unknown model is a 404, and a model still in use after a short grace period a 409. Ignored on a normal request; localm manages residency itself (`idle_unload_seconds`). |
| other `options` keys, unknown request keys | Accepted and ignored (named in the debug log). |

Refused with a 400, rather than silently dropped: `raw`, `suffix`, a
request-level `template`, and a non-empty `context`. Also a 400: a missing
`model`, a `format` string other than `"json"`, an `options.stop` that is not a
string or a list of strings, a `think` that is not a boolean or a string, a
message with a role other than `system`, `user`, `assistant` or `tool`,
`tool_calls` on a message that is not from the assistant (or one without a
function name), and `images` that are not base64 image data.

Each reply object carries `model` (as the request named it), `created_at`, the
content (`message` or `response`) and `done`. A stream is `application/x-ndjson`,
one object per line. The last object adds `done_reason` (`stop` or `length`)
and the timing fields `total_duration`, `prompt_eval_count`,
`prompt_eval_duration`, `eval_count` and `eval_duration` (the durations in
nanoseconds). `prompt_eval_duration` is the time to the first token and
`eval_duration` the decode time implied by the measured tokens per second; a
field with no measurement behind it is left out. `load_duration` is never sent.
The `X-Localm-*` response headers of the underlying chat route (for example
`X-Localm-Model-Routing`) are forwarded as HTTP headers on both streamed and
non-streamed replies. When a request spends a long time being prepared, the
stream opens before those headers are known and they are not sent: NDJSON has
no place to carry them.

## Embeddings

`POST /api/embed` takes `input` (a string or a list) and returns `{"model",
"embeddings", "total_duration"}`, plus `prompt_eval_count` when the backend
counted the input tokens. `POST /api/embeddings` takes `prompt` and returns
`{"embedding": [...]}` only. The vectors are whatever the embedding model
produces; they are not renormalised. A missing `model`, `input` or `prompt` is a
400, as is `dimensions`; `truncate` is accepted and ignored. An empty `input` list
returns `{"model", "embeddings": []}` without loading the model, but it still
refuses `dimensions` and answers 404 for a model that is not registered (a model
served by another instance counts as registered).

## Listings

- `GET /api/tags` returns `{"models": [...]}`: the registered chat and embedding
  models (and the startup model when it is not registered), each with
  `name`, `model`, `size` (the weights' size in bytes, 0 when unreadable),
  `modified_at` (the weights file's modification time, or
  `1970-01-01T00:00:00Z` when it cannot be read), `digest` (the recorded SHA-256, or
  empty) and `details`: `parent_model` (empty), `format`, `family` (the model's
  architecture when recorded, else empty), `families` (a list holding `family`,
  or null), and empty `parameter_size` and `quantization_level`.
- `GET /api/ps` returns `{"models": [...]}`: the loaded models, each with the
  `name`, `model`, `size`, `digest` and `details` of a tag (no `modified_at`),
  `expires_at` (the idle-unload time, or a date in 2318 when no idle
  timeout is set) and `context_length` (the loaded window, when known).
  `size_vram` is not reported.
- `POST /api/show` takes `model` (or `name`; `verbose` is accepted and ignored)
  and returns `details`, `model_info` (`general.architecture` and
  `<architecture>.context_length` when recorded), `modified_at`, `capabilities`
  (`completion` and `tools` for a chat model, `embedding` for an embedding
  model, plus `vision` and `thinking` when known) and empty `modelfile`,
  `parameters` and `template`. A missing model name is a 400 and an unknown
  model a 404.
- `GET /api/version` returns `{"version": "<localm's own version>"}`.

## Model management

`POST /api/copy` takes `source` and `destination`, registers a second name for a
model (an alias, the same file) and returns `{"status": "success"}`. A missing
field or a destination that is not a usable name (letters, digits, `.`, `_` and
`-`) is a 400, an unknown source a 404 and a destination that already exists a
409.
`POST /api/pull`, `POST /api/push`, `POST /api/create`, `DELETE /api/delete` and
`POST` or `HEAD /api/blobs/{digest}` answer 501 with the localm equivalent in the
message: `localm pull`, `localm rm`, or the GUI.
