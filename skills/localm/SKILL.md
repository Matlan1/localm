---
name: localm
description: >
  Operational guidance for localm's MCP tools: offline chat, model search and
  download, checking capacity before loading a model, delegating a whole
  coding task instead of one-shot chat calls, and what image generation
  needs. Load this before calling any localm tool for the first time in a
  session, or when choosing between localm's tools for a task.
---

# localm

localm exposes a local, offline LLM engine as MCP tools: chat, model search
and management, diagnostics, and (when active) coding-agent delegation and
image generation. Every call runs on this machine's own hardware. Nothing is
sent anywhere except an explicit download (`pull_model`) or a HuggingFace
search (`search_models`).

The tool list is not fixed. `run_coder_task` only appears when the coder
plugin is active and the server was not started with `--no-coder`;
`generate_image` only appears unless the server was started with
`--no-images`; `embed` only appears when the active model backend can
produce embeddings. Check what is actually advertised for this session
rather than assuming a tool exists.

## Check capacity before loading or downloading a model

Call `system_stats` before picking a model or a quant for a task. If VRAM
is tight, prefer a smaller quant over skipping the task or accepting a
worse answer, and prefer evicting the currently loaded model over settling
for an under-sized one when the task genuinely needs the better model.

Before `pull_model`-ing something new, run `search_models` to find a repo,
then `list_model_files` on that repo: it returns each quant's size and a
fit badge (fits, tight, or too-big) against this machine's free VRAM, so
you can pick a quant that will actually load instead of discovering it
does not fit after a multi-gigabyte download.

Call `server_activity` before starting anything long-running (a pull, a
re-embed, a media generation). It reports what any localm server running on
this machine is doing right now, including work started from the GUI or
another client that this session cannot otherwise see, so you do not start
a redundant download or generation.

## `chat` vs `run_coder_task`

`chat` is a single generate call: one prompt in, one response out, no side
effects. Use it for a question, a transform, or anything that does not need
to touch files or run commands.

`run_coder_task` delegates a whole, self-contained coding task (reading and
editing files, running shell commands, git, tests) to localm's own offline
agent and blocks until it finishes or times out. Use it to hand off a
describable chunk of work rather than driving each step yourself. It needs
`task` and `cwd` (a real project directory); shell commands inside the
delegated task are denied by default, since there is no terminal to confirm
them, unless the call sets `yes: true`. `max_turns` (default 40) and
`timeout_seconds` (default 900) bound a run that goes wrong.

## `generate_image`

Local FLUX image generation via ComfyUI. The tool is advertised whenever the
server was not started with `--no-images`, independent of whether ComfyUI
is actually reachable; a call fails if ComfyUI is not running. If ComfyUI
runs on a non-default host or port, that is configured on the localm side
(the `FLUX_API_URL` environment variable, or the `comfy_api_url` setting),
not per call. A successful call returns the saved file path and the seed
used, so a follow-up call can reproduce or vary the same image.

## `install_plugin` / `enable_plugin` / `list_plugins` manage localm itself, not this bundle

These tools install and toggle localm's own engine plugins (`coder`, `rag`,
`image`, `memory`, and others) inside the running localm instance. They have
nothing to do with this Claude Code plugin: enabling localm's `coder` plugin
is what makes `run_coder_task` appear in the tool list at all, but neither
tool touches this MCP bundle's own installation.

## Destructive tools

`remove_model` and `uninstall_plugin` delete a model file or an installed
plugin (and its data, if `delete_data` is set) and carry a
`destructiveHint` annotation. Confirm the target is right before calling
either; there is no undo.

## Reference

The full tool table (every tool, its parameters, and its annotations) and
the coder agent's own MCP-client side (connecting external MCP servers to
`localm coder`) are in `docs/mcp.md`. The CLI equivalents of these tools,
and everything else localm can do outside MCP, are in `docs/cli.md`.
