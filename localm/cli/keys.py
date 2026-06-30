# SPDX-License-Identifier: AGPL-3.0-or-later
import os
import sys

import click

from ._core import console, main


# ------------------------------------------------------------------ #
#  API key management                                                  #
# ------------------------------------------------------------------ #

def _mask_key(key: str) -> str:
    """Show enough of a key to recognise it without leaking the whole secret to
    a terminal log or screen-share. Short keys are fully masked."""
    key = key.strip()
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"




@main.group("key")
def key_group():
    """View and manage the API key that protects localm's HTTP surface.

    localm has no accounts: "auth" is one shared owner key, presented as
    'Authorization: Bearer <key>'. With no key the server runs open (loopback
    dev); set one to require it. You can also mint named, scope-limited keys for
    clients that should only do part of what the owner can.

    \b
    Examples:
      localm key show
      localm key generate
      localm key create dashboard --scope models:read
    """




@key_group.command("show")
@click.option("--reveal", is_flag=True,
              help="Print the full key instead of a masked preview.")
def key_show(reveal):
    """Show the active owner key (masked) or 'open mode'."""
    from localm import auth
    active = auth.get_api_key()
    if not active:
        console.print("[yellow]open mode[/yellow] - no API key set; the server "
                      "accepts unauthenticated requests.")
        if auth.require_auth_enabled():
            console.print("[dim]require_auth is on, so protected routes refuse "
                          "every request until a key is set.[/dim]")
        return
    env_key = os.environ.get(auth.ENV_VAR)
    from_env = bool(env_key and env_key.strip())
    source = f"{auth.ENV_VAR} env" if from_env else "auth.key file"
    shown = active if reveal else _mask_key(active)
    console.print(f"API key [dim](from {source})[/dim]:")
    console.print(f"  [bold]{shown}[/bold]")
    if from_env:
        console.print(f"[dim]{auth.ENV_VAR} overrides the stored auth.key file "
                      "while it is set.[/dim]")




@key_group.command("generate")
def key_generate():
    """Generate a new random owner key, persist it, and print it once."""
    from localm import auth
    key = auth.regenerate_key()
    console.print("[green]New API key (shown once - copy it now):[/green]")
    console.print(f"  [bold]{key}[/bold]")
    console.print("[dim]Clients send it as: Authorization: Bearer <key>[/dim]")
    env_key = os.environ.get(auth.ENV_VAR)
    if env_key and env_key.strip():
        console.print(f"[yellow]Note:[/yellow] {auth.ENV_VAR} is set and "
                      "overrides this stored key until it is unset.")




@key_group.command("set")
@click.argument("key")
def key_set(key):
    """Persist a specific owner KEY (use 'generate' for a random one)."""
    from localm import auth
    if not key or not key.strip():
        raise click.ClickException("Key must be non-empty.")
    auth.set_api_key(key)
    console.print(f"[green]✓[/green] API key set "
                  f"[dim]({_mask_key(key)})[/dim]")




@key_group.command("clear")
@click.option("-y", "--yes", is_flag=True, help="Skip the confirmation prompt.")
def key_clear(yes):
    """Remove the owner key, returning the server to open mode."""
    from localm import auth
    if auth.get_api_key() is None:
        console.print("[dim]No API key set (already open mode).[/dim]")
        return
    if not yes and not click.confirm(
            "Clear the API key? The server will then accept unauthenticated "
            "requests (unless require_auth is on)"):
        console.print("[dim]Cancelled.[/dim]")
        return
    auth.clear_api_key()
    console.print("[green]✓[/green] API key cleared - open mode.")
    env_key = os.environ.get(auth.ENV_VAR)
    if env_key and env_key.strip():
        console.print(f"[yellow]Note:[/yellow] {auth.ENV_VAR} is still set, so a "
                      "key remains active from the environment.")




@key_group.command("list")
def key_list():
    """List named, scope-limited keys (metadata only - never the secret)."""
    from localm import auth
    keys = auth.list_keys()
    if not keys:
        console.print("[dim]No named keys. Mint one:  "
                      "localm key create <name> --scope <scope>[/dim]")
        return
    from rich.table import Table
    table = Table(header_style="bold cyan")
    table.add_column("ID", style="bold")
    table.add_column("Name")
    table.add_column("Scopes")
    for k in keys:
        table.add_row(str(k.get("id", "")), str(k.get("name", "")),
                      ", ".join(k.get("scopes", [])))
    console.print(table)




@key_group.command("create")
@click.argument("name")
@click.option("-s", "--scope", "scopes", multiple=True, required=True,
              help="A capability scope to grant (repeatable).")
def key_create(name, scopes):
    """Mint a named key limited to SCOPES; print the secret once.

    Privileged scopes (admin, keys:admin, plugins:admin, config:write) are
    refused here - only the owner key carries those, so a minted key can never
    escalate itself.
    """
    from localm import auth
    try:
        rec = auth.create_key(name, list(scopes), allow_privileged=False)
    except (ValueError, PermissionError) as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print(f"[green]New key '{rec['name']}' "
                  f"({', '.join(rec['scopes'])}) - shown once:[/green]")
    console.print(f"  [bold]{rec['key']}[/bold]")
    console.print(f"[dim]id {rec['id']}; revoke with: "
                  f"localm key rm {rec['id']}[/dim]")




@key_group.command("rm")
@click.argument("key_id")
@click.option("-y", "--yes", is_flag=True, help="Skip the confirmation prompt.")
def key_rm(key_id, yes):
    """Revoke a named key by ID (see 'localm key list')."""
    from localm import auth
    if not yes and not click.confirm(f"Revoke named key '{key_id}'?"):
        console.print("[dim]Cancelled.[/dim]")
        return
    if auth.revoke_key(key_id):
        console.print(f"[green]✓[/green] Revoked {key_id}.")
    else:
        console.print(f"[red]No such key:[/red] {key_id}")
        sys.exit(1)
