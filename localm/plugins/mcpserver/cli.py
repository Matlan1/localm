# SPDX-License-Identifier: AGPL-3.0-or-later
"""CLI for the localm MCP server plugin: ``localm mcp``."""

from __future__ import annotations

import json
import sys

import click


@click.command("mcp")
@click.option("-m", "--model", default=None, envvar="LOCALM_MODEL",
              help="Default model for chat/embed tools "
                   "[default: LOCALM_MODEL env, else first registered].")
@click.option("--no-images", is_flag=True,
              help="Don't expose the generate_image tool.")
@click.option("--no-coder", is_flag=True,
              help="Don't expose run_coder_task even if the coder plugin is active.")
@click.option("--no-memory", is_flag=True,
              help="Don't expose memory_recall even if the memory plugin is active.")
@click.option("--memory-write", is_flag=True,
              help="Also expose memory_append, letting the client write to your "
                   "durable memory. Off by default: enabling the memory plugin is "
                   "not consent for an external client to write into it.")
@click.option("--print-config", is_flag=True,
              help="Print the mcpServers JSON block for your MCP client and exit.")
def main(model, no_images, no_coder, no_memory, memory_write,
         print_config):
    """Run localm as an MCP server (stdio transport).

    MCP clients launch this command on demand - add it to the client's
    server config and the client starts/stops it automatically:

    \b
      Claude Desktop (claude_desktop_config.json) and most clients:
        localm mcp --print-config

    \b
    Exposed tools: chat, model management (list_models, search_models,
    list_model_files, pull_model, remove_model, setup_embeddings),
    system_stats, run_doctor, plugin management (list/install/enable/
    disable/uninstall_plugin), and, conditionally, embed, generate_image
    (unless --no-images), run_coder_task (coder plugin active, unless
    --no-coder), and memory_recall (memory plugin active, unless --no-memory).
    remove_model and uninstall_plugin are marked destructive.

    memory_append is NOT exposed unless you pass --memory-write: enabling the
    memory plugin turns on localm's own chat memory, which is a separate
    decision from letting an external client write into it. Both memory tools
    are refused in privacy mode.
    All output except the protocol goes to stderr; logs never corrupt
    the JSON-RPC stream.
    """
    if print_config:
        args = ["mcp"]
        if model:
            args += ["--model", model]
        if no_images:
            args += ["--no-images"]
        if no_coder:
            args += ["--no-coder"]
        if no_memory:
            args += ["--no-memory"]
        if memory_write:
            args += ["--memory-write"]
        block = {
            "mcpServers": {
                "localm": {
                    "command": "localm",
                    "args": args,
                }
            }
        }
        import sys as _sys
        if _sys.platform == "win32":
            cfg_path = "%APPDATA%\\Claude\\claude_desktop_config.json"
        elif _sys.platform == "darwin":
            cfg_path = "~/Library/Application Support/Claude/claude_desktop_config.json"
        else:
            cfg_path = "~/.config/Claude/claude_desktop_config.json"
        click.echo(json.dumps(block, indent=2))
        click.echo(
            "\n# Paste the mcpServers entry into your MCP client's config.\n"
            f"# Claude Desktop: {cfg_path}\n"
            "# The client launches/stops the server automatically.\n"
            "# First install the plugin:  localm plugin install mcp",
            err=False,
        )
        return

    # The MCP server plugin ships disabled, so `localm mcp` refuses to serve
    # until the user opts in. --print-config above is exempt.
    from localm.plugins.engine import PluginManager
    if not PluginManager(None).is_active("mcp"):     # installed (on disk) AND enabled
        click.echo(
            "The MCP server plugin is not active.\n"
            "Install it with:  localm plugin install mcp\n"
            "(then your MCP client can launch this server).",
            err=True,
        )
        raise SystemExit(1)

    if sys.stdin.isatty():
        click.echo(
            "localm mcp speaks the MCP stdio protocol - it is meant to be "
            "launched BY an MCP client, not run interactively.\n"
            "Get the client config with:  localm mcp --print-config",
            err=True,
        )

    from .server import serve_stdio
    serve_stdio(model=model, enable_images=not no_images,
                enable_coder=not no_coder, enable_memory=not no_memory,
                enable_memory_write=memory_write)
