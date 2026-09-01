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

localm exposes local-model operations and localm management as MCP tools. Most are
always present; three are conditional (marked below).

| Tool | What it does | Annotation |
|---|---|---|
| `chat` | Generate with a local model. Per-call `model`, `system`, `seed`, `temperature`, `max_tokens` | |
| `server_activity` | What any running localm server (GUI/HTTP) on this machine is doing right now - downloads, indexing, media generation - so a client can check before starting a long operation of its own | read-only |
| `list_models` | Your registry with sizes and sources | read-only |
| `system_stats` | Live CPU/RAM/VRAM/GPU load, for judging model/quant fit | read-only |
| `search_models` | Search HuggingFace for GGUF repos | read-only |
| `list_model_files` | A repo's GGUF files with quant, size, and VRAM-fit | read-only |
| `pull_model` | Download, register, and (by default) load a GGUF | |
| `setup_embeddings` | Install the on-device embedding model | |
| `remove_model` | Remove a model and delete its file if under the models dir | **destructive** |
| `run_doctor` | Run `localm doctor` and return the report | read-only |
| `list_plugins` | List engine plugins and whether each is active | read-only |
| `install_plugin` | Install and enable an engine plugin | |
| `enable_plugin` | Enable an installed plugin | |
| `disable_plugin` | Disable an installed plugin | |
| `uninstall_plugin` | Uninstall a plugin (and its data with `delete_data`) | **destructive** |
| `embed` | Embedding vectors from the local model (only when the backend can embed) | |
| `run_coder_task` | Delegate a whole coding task to the local coder agent (only when the coder plugin is active and not `--no-coder`) | |
| `generate_image` | Local FLUX via ComfyUI (omit with `--no-images`; needs a reachable ComfyUI) | |

### Tool annotations

Each tool carries MCP annotations: read-only tools are marked `readOnlyHint`, and
the two that delete things (`remove_model`, `uninstall_plugin`) are marked
`destructiveHint`, so an annotation-aware client can prompt for confirmation before
a destructive call. The annotations are advisory metadata the server advertises;
the server itself does not prompt. Coverage is deliberately narrow: only those two
carry `destructiveHint`. Other state-changing tools (for example `pull_model`,
`setup_embeddings`, the plugin install/enable/disable tools, and the conditional
`run_coder_task` and `generate_image`, which run code and write files) carry no
annotation, so an annotation-only client cannot tell they mutate. localm's own
coder client (below) does not read these hints from remote servers; it gates on the
`trusted` config flag instead.

Options for the server:

```bash
localm mcp --model NAME      # default model (else LOCALM_MODEL env, else the first chat-eligible registered model)
localm mcp --no-images       # do not expose generate_image
localm mcp --no-coder        # do not expose run_coder_task
```

The model loads on the first tool call, so client startup stays instant. All logging goes to stderr; stdout carries only protocol frames.

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

Open the config file in your editor and paste the block from Step 2 (if the file is new, paste it whole). If you have other MCP servers configured, add localm as another entry under `mcpServers`.

### Step 4: Restart Claude Desktop

Close and reopen Claude Desktop. The app will launch localm on startup. You should see the localm tools (chat, the model and plugin management tools, and generate_image unless you passed `--no-images`) available in the tool menu. See [Exposed tools](#exposed-tools) for the full list.

### Step 5: Try a tool

Click the tool icon in the Claude Desktop message input and select `chat`. Enter a prompt and a model name. The local model runs offline and returns the response. For example:

- Model: `mistral-7b`
- Prompt: `Explain MCP in one sentence.`

The tool appears in the tool call panel with the model's response.

## Quick start: Claude Code (Agent Plugin)

This repo carries a `plugin.json`, `.mcp.json`, and a `localm` skill at its
root, so Claude Code can load the MCP server as a plugin instead of a manual
config paste:

```bash
localm plugin install mcp
claude --plugin-dir /path/to/this/repo
```

`--plugin-dir` loads it for that one session, including the `localm` skill
(operational guidance on picking a model/quant, when to delegate a whole
task versus a single chat call, and what image generation needs). A
persistent install via `claude plugin install` needs the repo registered
with a marketplace, which is not set up yet.

## Coder + external MCP servers

The coder agent (localm coder) can load external MCP servers alongside localm's tools, adding specialized tools (file search, database query, API calls).

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
- `env` (optional): a table of environment variables to set for the server process, merged on top of the coder's own environment - see [Permission errors in MCP tools](#permission-errors-in-mcp-tools) below
- `trusted` (optional, default false): if true, every tool this server offers runs without confirmation; if false, every one of its tools (whatever it actually does) is treated as destructive and triggers a confirmation prompt - see [Security: trusted vs. untrusted](#security-trusted-vs-untrusted) below

### How tools appear

When you run `localm coder`, it spawns each configured server and fetches its tool list. Tools are named `mcp_<server>_<tool>` and appear in the agent's system prompt. For example:

- Server `fs` with tool `search` becomes `mcp_fs_search`
- Server `search` with tool `query` becomes `mcp_search_query`

The agent sees the tool description and calls it like a built-in tool (e.g. `mcp_fs_search(pattern="*.py", path="src")`).

### Verify tools loaded

Start the coder:

```bash
localm coder
```

MCP startup output does not depend on `--verbose` (that flag only affects how tool call results are printed later). A server that starts cleanly prints nothing here: its tools appear in the agent's tool list (named `mcp_<server>_<tool>` and listed in the system prompt). Confirm it loaded by checking its tools are offered, for example `mcp_fs_search`.

If a server fails to start, the coder prints a warning and continues - MCP problems never break the agent. For example:

```
MCP server 'search': command not found: npx
```

### Security: trusted vs. untrusted

`trusted` is a per-SERVER flag, not a per-tool one: the coder client does not
inspect a remote tool's own annotations or guess what it does, it gates ALL of
a server's tools the same way based on this one setting.

- **untrusted (default)**: every tool the server offers - including a
  read-only one like a search or query tool - is registered as destructive,
  so each call is gated by a confirmation prompt. You review each call and
  decide to allow or skip.
- **trusted**: every tool from that server runs without prompts. Use this only
  for servers you trust completely (e.g. internal tools you wrote, well-known
  open-source projects with security track records) - a single tool call you
  did not mean to allow cannot be stopped once the server is trusted.

Do not set `trusted = true` for arbitrary code. MCP servers run with your user account and full file access.

**`trusted` only governs confirmation, not how the result is read.** Separately
from that flag, the *output* of every `mcp_<server>_<tool>` call - trusted
server or not - is wrapped as untrusted external content before it reaches the
model: fenced, labeled, and accompanied by an instruction not to treat it as
commands. This is indirect-prompt-injection defense in depth (the same
handling `fetch_url` and `web_search` get), because a tool result can carry
attacker-influenceable text (a document, a search hit, a database row) that a
model would otherwise read as part of its own context. `trusted` decides
whether *calling* a tool needs your approval; the output fencing runs
regardless and is not configurable.

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
localm plugin status
```

Confirm `mcp` is listed as installed and enabled in the output.

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

If the server's own process wrote anything to stderr before it died - a
Python traceback, a missing-dependency message, an auth error - the coder
captures the last 20 lines and includes them in the warning it prints, so
read the full warning text before digging further; it often names the real
cause directly.

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

A healthy server waits for input (no output); if it errors instead, install its missing packages.

**Check the args:**

Some servers have required arguments. Consult the server's documentation:

```toml
[mcp.servers.search]
command = "python"
args = ["my_search_server.py", "--db-path", "app.db"]
```

### Tool not visible in coder

**Check the server is alive:**

Start the coder:

```bash
localm coder
```

If a server failed to start, a yellow warning line naming it appears at startup (see "MCP server fails to start" above) - this does not depend on `--verbose`. A clean server is silent (see "Verify tools loaded"), so confirm it with the tool-name check below.

**Check the tool name:**

The registered name is `mcp_<server>_<tool>`, where `<tool>` is the tool's name from the server (not your invention). Consult the server's own docs for its tool names.

**Is the tool marked destructive?**

Unless the server is marked `trusted = true` in the config, every one of its
tools is treated as destructive and needs confirmation - there is no
per-tool exception. The agent will offer to call the tool, and you must
approve it in the console before it runs.

### Image generation fails

**Check ComfyUI is running:**

The `generate_image` tool requires ComfyUI. Start it:

```bash
python main.py
```

(download from https://github.com/comfyanonymous/ComfyUI). It prints its URL on startup, typically `http://127.0.0.1:8188`.

The `generate_image` tool is already in the menu (it is hidden only by `--no-images`, never by ComfyUI's reachability); once ComfyUI is running, retry the call and it should succeed.

**Pass the URL if non-standard:**

If ComfyUI runs on a different machine or port, set the environment variable before launching Claude:

```bash
set FLUX_API_URL=http://192.168.1.100:8188
```

(on macOS/Linux, use `export` instead of `set`). To set it persistently, run `localm config comfy_api_url http://192.168.1.100:8188`.

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
