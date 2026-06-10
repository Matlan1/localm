# The localm web GUI

`localm gui` starts the inference server with a browser frontend on top. Chat with your models and run the coder agent from one page. There is no build step, no Node, and no network dependency: the frontend is plain HTML/JS served by the same FastAPI process, with vendored libraries for markdown rendering and syntax highlighting.

```bash
localm gui              # first registered model, opens your default browser
localm gui mymodel      # pick a model
localm gui --no-browser # just start the server, open the URL yourself
localm gui -p 8650      # explicit port (auto-bumps when busy)
```

## Chat

- Model selector in the sidebar lists every registered model. Switching loads the new model and unloads the old one (the switch waits for any in-flight request to finish).
- Streaming responses with markdown and highlighted code blocks, copy buttons on messages and code.
- The parameters drawer sets temperature, top-p, max tokens, seed, and a system prompt per conversation.
- Conversations are stored in your browser's localStorage, never on the server. Deleting one removes it for good.
- The usage line under the composer shows total tokens, time to first token, and tokens per second for the last reply.

## Coder

Start a session by pointing the agent at a project directory. The agent gets the same tools as the terminal version: read, write, edit, patch, shell, search, tests, image generation, plus any MCP tools configured for that project.

What you see in the feed:

- The agent's reasoning streams live.
- Every tool call becomes a card. Click it to expand arguments and output.
- Destructive actions (file writes, shell commands) pause the agent and show an approval card with a unified diff of exactly what would change. Approve or reject from the browser. Unanswered approvals time out after 10 minutes and are rejected.
- Auto-approve can be enabled at session start if you trust the task.

Session persistence follows the coder's modes: `privacy` (default, nothing saved), `log` (JSONL audit trail), `full` (audit trail plus markdown transcript).

Stop asks the agent to halt at the next safe point. End session terminates it.

## Security notes

- The server binds to 127.0.0.1 by default and CORS is locked to localhost, so other websites you visit cannot call your API from browser JS.
- If `LOCALM_API_KEY` is set, the GUI prompts for the key once and stores it in localStorage.
- Binding to 0.0.0.0 without an API key triggers a CLI warning. Do not expose the GUI to a network you do not trust: the coder agent can write files and run shell commands on this machine.

## How it fits together

```
browser (static JS)
   │  /v1/chat/completions      streaming chat
   │  /api/models               registry + engine switching
   │  /api/coder/sessions/*     session lifecycle, SSE events, approvals
   ▼
localm FastAPI server ── Engine (GGUF/HF) ── your GPU
            └── coder Agent (per session, in-process thread)
```

The coder agent talks to the model through the server's own OpenAI-compatible endpoint, so chat and agent share one engine and inference is serialised cleanly between them.
