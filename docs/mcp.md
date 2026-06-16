# MCP support

> The MCP server is the `mcp` plugin. Install it with `localm plugin install
> mcp`; it is not active by default. The `generate_image` tool below works only
> when the `image` plugin and ComfyUI are also present.

localm speaks the Model Context Protocol in both directions:

- **Server**: `localm mcp` exposes your local models as tools to any MCP client (Claude Desktop, IDEs, other agents).
- **Client**: the coder agent can consume external MCP tool servers and use their tools like built-ins.

Both sides use stdio transport (JSON-RPC 2.0, newline-delimited). No network ports are opened.

## localm as an MCP server

```bash
localm mcp --print-config
```

This prints the standard `mcpServers` JSON block plus the path of Claude Desktop's config file on your system. Paste the entry there (or into any other MCP client's config) and restart the client. The client starts and stops the server process automatically; you never run `localm mcp` by hand.

Exposed tools:

| Tool | What it does |
|---|---|
| `chat` | Generate with a local model. Per-call `model`, `system`, `seed`, `temperature`, `max_tokens` |
| `list_models` | Your registry with sources and sizes |
| `embed` | Embedding vectors from the local model |
| `generate_image` | Local FLUX via ComfyUI (omit with `--no-images`) |

Options:

```bash
localm mcp --model NAME      # default model (else LOCALM_MODEL env, else first registered)
localm mcp --no-images       # hide the image tool
```

The model loads lazily on the first tool call, so client startup stays instant. All logging goes to stderr; stdout carries only protocol frames.

## The coder as an MCP client

Add servers to `.localcoder/config.toml` in your project:

```toml
[mcp.servers.fs]
command = "some-mcp-server"
args = ["--root", "."]

[mcp.servers.search]
command = "python"
args = ["my_search_server.py"]
trusted = true
```

On agent start, each server is spawned and its tools are registered as `mcp_<server>_<tool>`. The agent sees their descriptions in its system prompt and calls them like built-in tools.

Security defaults:

- Tools from servers without `trusted = true` are treated as destructive, which routes them through the normal confirmation gate.
- A server that fails to start produces a warning and is skipped. External servers can never break the agent.

Be deliberate about which servers you configure. An MCP server is arbitrary code running with your user account; review it the way you would review any dependency, and avoid `npx -y <anything>` style launchers that download and execute unreviewed code.
