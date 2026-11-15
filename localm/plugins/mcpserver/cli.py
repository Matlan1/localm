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
@click.option("--coder-shell", is_flag=True,
              help="Let run_coder_task auto-approve shell commands when the "
                   "client passes yes=true (or the project config sets "
                   "auto_approve). Off by default: without it the coder may "
                   "write files but every shell call is denied.")
@click.option("--coder-root", type=click.Path(file_okay=False, exists=True,
                                              path_type=str),
              default=None,
              help="Only let run_coder_task work in this folder or below "
                   "[default: any folder].")
@click.option("--share-loaded-models", "share_loaded", is_flag=True,
              help="Answer chat with a model another localm instance on this "
                   "machine already has loaded, instead of loading a second copy "
                   "into VRAM. Another install's instance is used only when it "
                   "needs no API key, and any instance only while it records "
                   "chat no more than this machine's chat persistence allows.")
@click.option("--print-config", is_flag=True,
              help="Print the mcpServers JSON block for your MCP client and exit.")
def main(model, no_images, no_coder, no_memory, memory_write, coder_shell,
         coder_root, share_loaded, print_config):
    """Run localm as an MCP server (stdio transport).

    MCP clients launch this command on demand - add it to the client's
    server config and the client starts/stops it automatically:

    \b
      Claude Desktop (claude_desktop_config.json) and most clients:
        localm mcp --print-config

    \b
    Exposed tools: chat, rerank, server_activity, model management (list_models, search_models,
    list_model_files, pull_model, remove_model, setup_embeddings),
    system_stats, run_doctor, plugin management (list_plugins,
    install_plugin, enable_plugin, disable_plugin, uninstall_plugin), and,
    conditionally, embed, generate_image
    (unless --no-images), run_coder_task (coder plugin active, unless
    --no-coder), and memory_recall (memory plugin active, unless --no-memory).
    remove_model and uninstall_plugin are marked destructive.

    run_coder_task never auto-approves shell commands unless you pass
    --coder-shell, and --coder-root DIR confines the folders it may work in.

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
        if coder_shell:
            args += ["--coder-shell"]
        if coder_root:
            args += ["--coder-root", coder_root]
        if share_loaded:
            args += ["--share-loaded-models"]
        # The command is this process's own interpreter running `-m localm`.
        # A LOCALM_HOME set in this shell is carried as an absolute path, so the
        # client starts the server on the same data folder. Only the JSON goes
        # to stdout; the notes after it go to stderr.
        import os

        server: dict = {"command": sys.executable, "args": ["-m", "localm", *args]}
        if os.environ.get("LOCALM_HOME", "").strip():
            from pathlib import Path

            from localm.config import home_dir
            server["env"] = {"LOCALM_HOME": str(Path(home_dir()).absolute())}
        block = {"mcpServers": {"localm": server}}
        if sys.platform == "win32":
            cfg_path = "%APPDATA%\\Claude\\claude_desktop_config.json"
        elif sys.platform == "darwin":
            cfg_path = "~/Library/Application Support/Claude/claude_desktop_config.json"
        else:
            cfg_path = "~/.config/Claude/claude_desktop_config.json"
        click.echo(json.dumps(block, indent=2))
        click.echo(
            "\n# Paste the mcpServers entry into your MCP client's config.\n"
            f"# Claude Desktop: {cfg_path}\n"
            "# The client launches/stops the server automatically.\n"
            "# First install the plugin:  localm plugin install mcp",
            err=True,
        )
        return

    # The MCP server plugin ships disabled, so `localm mcp` refuses to serve
    # until the user opts in. --print-config above is exempt.
    from localm.plugins.engine import PluginManager
    mgr = PluginManager(None)
    if not mgr.is_active("mcp"):     # installed (on disk) AND enabled
        remedy = ("Enable it with:  localm plugin enable mcp"
                  if mgr.is_installed("mcp") else
                  "Install it with:  localm plugin install mcp")
        click.echo(
            "The MCP server plugin is not active.\n"
            f"{remedy}\n"
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
                enable_memory_write=memory_write, share_loaded=share_loaded,
                coder_shell=coder_shell, coder_root=coder_root)
