# SPDX-License-Identifier: AGPL-3.0-or-later
import os
import sys

import click

from ._core import console, main
from .errors import _note_env_override


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
    _note_env_override("is set and overrides this stored key until it is unset.")




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
    from localm import auth, sessions
    if auth.get_api_key() is None:
        console.print("[dim]No API key set (already open mode).[/dim]")
        return
    if not yes and not click.confirm(
            "Clear the API key? The server will then accept unauthenticated "
            "requests (unless require_auth is on)"):
        console.print("[dim]Cancelled.[/dim]")
        return
    auth.clear_api_key()
    # A browser owner session carries its own ADMIN scope snapshot and survives a
    # key roll (S1), so a leftover owner cookie would keep full access after the
    # key is gone - defeating the clear (dangerous when require_auth is on). Sign
    # every browser session out here, mirroring /api/auth/key/clear. Device bearer
    # KEYS live in the keystore and are untouched.
    revoked = sessions.revoke_all()
    console.print("[green]✓[/green] API key cleared - open mode.")
    if revoked:
        console.print("[dim]Browser sessions were signed out.[/dim]")
    _note_env_override("is still set, so a key remains active from the environment.")




@key_group.command("recover")
def key_recover():
    """Recover owner access after a lockout (run LOCALLY on the server machine).

    Mints a FRESH owner key and prints it once - use this when you have lost the
    owner key but still need to manage the server. Existing scoped DEVICE keys are
    untouched, so devices keep working; only the owner credential is rotated. Live
    browser (cookie) sessions are signed out too, so a captured owner cookie cannot
    outlive the recovery. The local CLI is the trusted recovery path, so this does
    not require the old key (SEC-3). To instead drop all auth and return to open
    mode, use 'key clear'."""
    from localm import auth, sessions
    had = auth.get_api_key() is not None
    key = auth.regenerate_key()
    # regenerate_key deliberately leaves browser sessions alone (S1: a GUI key roll
    # must not log the browser out), but recovery is the compromise path. An owner
    # cookie carries its own ADMIN snapshot and is exempt from the keystore recheck,
    # so it would survive the rotation unless dropped here. Revoke every session
    # (NOT inside regenerate_key, to keep the GUI's survive-a-roll behavior intact),
    # mirroring /api/auth/key/clear. Device bearer KEYS in the keystore are untouched.
    revoked = sessions.revoke_all()
    console.print("[green]Owner access recovered. New owner key (shown once - "
                  "copy it now):[/green]")
    console.print(f"  [bold]{key}[/bold]")
    console.print("[dim]Clients send it as: Authorization: Bearer <key>[/dim]")
    if had:
        console.print("[dim]The previous owner key no longer works; scoped device "
                      "keys are unchanged.[/dim]")
    if revoked:
        console.print("[dim]Browser sessions were reset; sign in again with the "
                      "new key.[/dim]")
    _note_env_override("is set and overrides this stored key until it is unset.")


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
    table.add_column("FS access")
    for k in keys:
        table.add_row(str(k.get("id", "")), str(k.get("name", "")),
                      ", ".join(k.get("scopes", [])),
                      str(k.get("fs_access", "none")))
    console.print(table)




@key_group.command("create")
@click.argument("name")
@click.option("-s", "--scope", "scopes", multiple=True, required=True,
              help="A capability scope to grant (repeatable).")
@click.option("--fs-access", "fs_access",
              type=click.Choice(["none", "host"]), default="none",
              show_default=True,
              help="Host filesystem reach: none (device upload only) or host (the "
                   "whole server disk). Defaults to none so a shared key cannot "
                   "browse your disk. (A confined 'shared'/designated-roots tier "
                   "is reserved but not yet enforced.)")
def key_create(name, scopes, fs_access):
    """Mint a named key limited to SCOPES; print the secret once.

    Privileged scopes (admin, keys:admin, plugins:admin, config:write) are
    refused here - only the owner key carries those, so a minted key can never
    escalate itself.

    --fs-access grants host filesystem reach (default none). The owner key always
    has full host access; this is how you let (or deny) a shared device key browse
    the server disk.
    """
    from localm import auth
    try:
        rec = auth.create_key(name, list(scopes), allow_privileged=False,
                              fs_access=fs_access)
    except (ValueError, PermissionError) as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print(f"[green]New key '{rec['name']}' "
                  f"({', '.join(rec['scopes'])}; fs-access {rec['fs_access']}) "
                  f"- shown once:[/green]")
    console.print(f"  [bold]{rec['key']}[/bold]")
    console.print(f"[dim]id {rec['id']}; revoke with: "
                  f"localm key rm {rec['id']}[/dim]")
    # Catch at grant: if this key unlocks a plugin the host cannot serve yet
    # (not installed, or missing its pip extras), say so right here.
    try:
        from localm.plugins.engine import PluginManager
        warns = PluginManager(None).scope_deps_warnings(list(scopes))
    except Exception:
        warns = []
    for w in warns:
        console.print(f"[yellow]Note:[/yellow] {w}")
    if warns:
        console.print("[dim]Install missing plugin packages on the host with:  "
                      "localm plugin install-deps --all[/dim]")




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
