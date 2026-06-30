# MCP support

> The MCP server is the `mcp` plugin. Install it with `localm plugin install
> mcp`; it is not active by default. The `generate_image` tool below needs
> ComfyUI running; it is exposed unless you pass `--no-images` and does not
> require the `image` plugin.

localm speaks the Model Context Protocol in both directions:

- **Server**: `localm mcp` exposes your local models as tools to any MCP client (Claude Desktop, IDEs, other agents).
- **Client**: the coder agent can consume external MCP tool servers and use their tools like built-ins.

Both sides use stdio transport (JSON-RPC 2.0, newline-delimited). No network ports are opened.

## Exposed tools

All MCP clients see the same tools from localm:

| Tool | What it does |
|---|---|
| `chat` | Generate with a local model. Per-call `model`, `system`, `seed`, `temperature`, `max_tokens` |
| `list_models` | Your registry with sources and sizes |
| `embed` | Embedding vectors from the local model |
| `generate_image` | Local FLUX via ComfyUI (omit with `--no-images`) |

Options for the server:

```bash
localm mcp --model NAME      # default model (else LOCALM_MODEL env, else first registered)
localm mcp --no-images       # hide the image tool
```

The model loads lazily on the first tool call, so client startup stays instant. All logging goes to stderr; stdout carries only protocol frames.

## Quick start: expose localm to Claude Desktop

### Step 1: Install the plugin

```bash
localm plugin install mcp
```

### Step 2: Get the config block

```bash
localm mcp --print-config
```

This prints a JSON block showing the command to add to Claude Desktop's MCP server list, plus the path to Claude Desktop's config file on your system:

- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: Claude Desktop has no official Linux build; see alternative MCP clients below

The output looks like:

```json
{
  "mcpServers": {
    "localm": {
      "command": "localm",
      "args": ["mcp"]
    }
  }
}
```

### Step 3: Paste into Claude Desktop config

Open the config file in your editor and merge the `mcpServers` block into the existing config (if the file is new, paste the entire block). Example config after setup:

```json
{
  "mcpServers": {
    "localm": {
      "command": "localm",
      "args": ["mcp"]
    }
  }
}
```

If you have other MCP servers configured, add localm as another entry under `mcpServers`.

### Step 4: Restart Claude Desktop

Close and reopen Claude Desktop. The app will launch localm on startup. You should see the localm tools (chat, list_models, embed, generate_image if ComfyUI is running) available in the tool menu.

### Step 5: Try a tool

Click the tool icon in the Claude Desktop message input and select `chat`. Enter a prompt and a model name. The local model runs offline and returns the response. For example:

- Model: `mistral-7b`
- Prompt: `Explain MCP in one sentence.`

The tool appears in the tool call panel with the model's response.

## Coder + external MCP servers

The coder agent (localm coder) can load external MCP servers alongside localm's tools. This lets you use specialized tools (file search, database query, API calls) without writing Python.

### Add a server to the project config

Create or edit `.localcoder/config.toml` in your project root:

```toml
[mcp.servers.fs]
command = "some-mcp-server"
args = ["--root", "."]

[mcp.servers.search]
command = "python"
args = ["my_search_server.py"]
trusted = true
```

Each `[mcp.servers.NAME]` table declares:

- `command`: the executable to run (must be on PATH or an absolute path)
- `args`: command-line arguments (optional)
- `trusted` (optional, default false): if true, the agent can call tools without confirmation; if false, destructive tools trigger a confirmation prompt

### How tools appear

When you run `localm coder`, it spawns each configured server and fetches its tool list. Tools are named `mcp_<server>_<tool>` and appear in the agent's system prompt. For example:

- Server `fs` with tool `search` becomes `mcp_fs_search`
- Server `search` with tool `query` becomes `mcp_search_query`

The agent sees the tool description and calls it like a built-in tool (e.g. `mcp_fs_search(pattern="*.py", path="src")`).

### Verify tools loaded

Run the coder in verbose mode to see MCP startup:

```bash
localm coder --debug
```

Look for messages like:

```
[localm-mcp] MCP server 'fs' started
[localm-mcp] registered: mcp_fs_search, mcp_fs_read, ...
```

If a server fails to start, you will see a warning like:

```
[localm-mcp] MCP server 'search' failed to start: [command not found]
```

The coder continues anyway - MCP problems never break the agent.

### Security: trusted vs. untrusted

- **untrusted (default)**: Tools that modify files, delete data, or call external APIs are gated by a confirmation prompt. You review each call and decide to allow or skip.
- **trusted**: Tools run without prompts. Use this only for servers you trust completely (e.g. internal tools you wrote, well-known open-source projects with security track records).

Do not set `trusted = true` for arbitrary code. MCP servers run with your user account and full file access.

### Example: add sqlite server

```toml
[mcp.servers.db]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-sqlite", "app.db"]
trusted = true
```

This spawns the official SQLite MCP server (downloaded by npx) and registers its tools. The agent can query your database without leaving localm.

## Troubleshooting

### Claude Desktop: tools not visible

**Check 1: is the plugin installed and active?**

```bash
localm plugin install mcp
localm info
```

Look for `mcp: enabled` in the output.

**Check 2: did you merge the config correctly?**

Run:

```bash
localm mcp --print-config
```

and copy the entire `mcpServers` block into Claude Desktop's config file. Do not edit the JSON manually - copy the exact output.

**Check 3: restart Claude Desktop**

Close and reopen the app. If you edited the config while it was running, the app does not reload automatically.

**Check 4: check logs**

macOS: run `log stream --predicate 'process == "Claude"'` in a terminal and restart Claude Desktop. Errors appear in the log.

Windows: check `%APPDATA%\Claude\logs\` for error files.

### Coder: MCP server fails to start

**Check the command:**

Make sure the executable exists and is on PATH:

```bash
which some-mcp-server
```

If not on PATH, use an absolute path in `.localcoder/config.toml`:

```toml
[mcp.servers.db]
command = "/usr/local/bin/my-server"
```

**Check dependencies:**

If the server is a Python script, verify it can run standalone:

```bash
python my_search_server.py
```

It should print nothing and wait for JSON-RPC input (then hang - kill it with Ctrl+C).

If it errors, install missing packages:

```bash
pip install -r requirements.txt
```

**Check the args:**

Some servers have required arguments. Consult the server's documentation:

```toml
[mcp.servers.search]
command = "python"
args = ["my_search_server.py", "--db-path", "app.db"]
```

### Tool not visible in coder

**Check the server is alive:**

In verbose mode:

```bash
localm coder --debug
```

Look for `[localm-mcp]` lines. If the server failed to start, a warning appears (see "MCP server fails to start" above).

**Check the tool name:**

The registered name is `mcp_<server>_<tool>`, where `<tool>` is the tool's name from the server (not your invention). Run the server directly to see its tool list:

```bash
python my_search_server.py
```

It prints JSON-RPC messages. Send an `initialize` request (you may have to script this; see the localm source at `localm/plugins/coder/mcp.py` for an example).

**Is the tool marked destructive?**

If the server is not marked `trusted = true` in the config, destructive tools require confirmation. The agent will offer to call the tool, and you must approve it in the console before it runs.

### Image generation fails

**Check ComfyUI is running:**

The `generate_image` tool requires ComfyUI. Start it:

```bash
python main.py
```

(download from https://github.com/comfyanonymous/ComfyUI). It prints its URL on startup, typically `http://127.0.0.1:8188`.

Then restart Claude Desktop to reload the MCP server. The `generate_image` tool should appear.

**Pass the URL if non-standard:**

If ComfyUI runs on a different machine or port, set the environment variable before launching Claude:

```bash
set COMFYUI_URL=http://192.168.1.100:8188
```

(on macOS/Linux, use `export` instead of `set`).

### Permission errors in MCP tools

If an MCP tool fails with "permission denied", the server may be trying to access files outside your project:

- **For `fs` / file-access servers**: check that the server's root is set to your project path, not the system root.
- **For `sqlite` / database servers**: ensure the database file is readable and writable by your user.
- **For external API servers**: check that API keys are set in the server's environment, not hardcoded in the config.

Pass environment variables to a server:

```toml
[mcp.servers.api]
command = "python"
args = ["api_server.py"]
env = { "API_KEY" = "sk-..." }
```

---

## Technical details

### Protocol

localm implements MCP 2025-03-26 over JSON-RPC 2.0 on newline-delimited JSON stdin/stdout. This matches the MCP spec and allows any MCP-compliant client and server to interoperate.

### Alternative MCP clients

MCP servers work with any client that speaks the protocol:

- **Claude Desktop** (Windows, macOS only; Linux builds are unofficial)
- **IDE plugins** (VS Code, JetBrains, etc. - check the marketplace)
- **LLM CLI tools** (aider, continue, and others support MCP)
- **Custom agents** that speak JSON-RPC

The localm MCP server (`localm mcp`) and the coder's client (`localm coder`) both use the same protocol, so they can also talk to each other or to any third-party server/client.
