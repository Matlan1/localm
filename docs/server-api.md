# HTTP Server API

localm's inference server exposes an OpenAI-compatible REST API. Start it with:

```bash
localm serve <model> [--host 127.0.0.1] [--port 8080]
```

## Endpoints

### `GET /health`

Returns 200 if the model is loaded, 503 otherwise.

```json
{"status": "ok", "model": "gemma4-4b"}
```

### `GET /v1/models`

Lists the currently loaded model in OpenAI format.

```json
{
  "object": "list",
  "data": [
    {
      "id": "gemma4-4b",
      "object": "model",
      "created": 1749000000,
      "owned_by": "localm"
    }
  ]
}
```

### `POST /v1/chat/completions`

Chat completions — streaming or non-streaming.

**Request body:**

```json
{
  "model": "gemma4-4b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is the capital of France?"}
  ],
  "max_tokens": 1024,
  "temperature": 0.8,
  "top_p": 0.95,
  "top_k": 40,
  "repeat_penalty": 1.1,
  "stream": false
}
```

All fields except `messages` are optional.

**Non-streaming response:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1749000000,
  "model": "gemma4-4b",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Paris."},
      "finish_reason": "stop"
    }
  ]
}
```

**Streaming response** (`stream: true`) — Server-Sent Events:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1749000000,"model":"gemma4-4b","choices":[{"index":0,"delta":{"role":"assistant","content":null},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1749000000,"model":"gemma4-4b","choices":[{"index":0,"delta":{"role":null,"content":"Paris"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1749000000,"model":"gemma4-4b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### `POST /v1/models/unload`

Unloads the current model from GPU memory without stopping the server. Returns 200 immediately; the model is released asynchronously.

```json
{}
```

**Response:**
```json
{"status": "unloaded"}
```

Useful before running FLUX image generation so ComfyUI has the full VRAM budget. The model reloads automatically on the next `/v1/chat/completions` request (see `engine.py` — `chat_stream` calls `backend.load()` if the backend is unloaded).

### `POST /v1/models/load`

Reloads the previously-unloaded model. Blocks until the model is ready.

```json
{}
```

**Response:**
```json
{"status": "loaded"}
```

---

## Multimodal (image input)

Pass images as base64 data-URIs in the multipart content format:

```json
{
  "model": "gemma4-12b",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQ..."
          }
        },
        {"type": "text", "text": "Describe this image."}
      ]
    }
  ]
}
```

Image input requires a multimodal model loaded with `--mmproj <path>`.

## Python Client Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="localm",  # any non-empty string
)

# Non-streaming
response = client.chat.completions.create(
    model="gemma4-4b",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="gemma4-4b",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
```

## CORS

The server allows all origins (`*`) by default, making it usable from browser-based UIs without a proxy.

## Concurrency

Requests are serialised — the model generates one completion at a time. Concurrent requests queue and execute in order. This is intentional: GPU memory is shared and the KV cache is not concurrency-safe.
