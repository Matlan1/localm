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
@click.option("--print-config", is_flag=True,
              help="Print the mcpServers JSON block for your MCP client and exit.")
def main(model, no_images, print_config):
    """Run localm as an MCP server (stdio transport).

    MCP clients launch this command on demand - add it to the client's
    server config and the client starts/stops it automatically:

    \b
      Claude Desktop (claude_desktop_config.json) and most clients:
        localm mcp --print-config

    \b
    Exposed tools: chat, list_models, embed, generate_image.
    All output except the protocol goes to stderr; logs never corrupt
    the JSON-RPC stream.
    """
    if print_config:
        args = ["mcp"]
        if model:
            args += ["--model", model]
        if no_images:
            args += ["--no-images"]
        block = {
            "mcpServers": {
                "localm": {
                    "command": "localm",
                    "args": args,
                }
            }
        }
        click.echo(json.dumps(block, indent=2))
        click.echo(
            "\n# Paste the mcpServers entry into your MCP client's config.\n"
            "# Claude Desktop: %APPDATA%\\Claude\\claude_desktop_config.json\n"
            "# The client launches/stops the server automatically.\n"
            "# First enable the plugin:  localm plugin enable mcp",
            err=False,
        )
        return

    # The MCP server is an optional plugin (Phase 3). It ships disabled, so a
    # client that launches `localm mcp` gets a clear refusal until the user opts
    # in - rather than silently exposing the models. --print-config above is
    # exempt so users can set up the client first, then enable.
    from localm.config import load_config
    if "mcp" not in load_config().get("plugins_enabled", []):
        click.echo(
            "The MCP server plugin is disabled.\n"
            "Enable it with:  localm plugin enable mcp\n"
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
    serve_stdio(model=model, enable_images=not no_images)
