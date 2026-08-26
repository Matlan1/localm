# SPDX-License-Identifier: AGPL-3.0-or-later
import os
import sys
import time
from datetime import datetime

import click

from ._core import console, main
from .errors import _note_env_override


# ------------------------------------------------------------------ #
#  API key management                                                  #
# ------------------------------------------------------------------ #

def _mask_key(key: str) -> str:
    """A key reduced to its first and last four characters, joined by '...'.

    The key is stripped first; one of 8 characters or fewer is masked in full."""
    key = key.strip()
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def _fmt_ts(ts) -> str:
    """A readable local timestamp, ``YYYY-MM-DD HH:MM``, for a single-line
    message. '-' when *ts* is None or cannot be read as a timestamp."""
    if ts is None:
        return "-"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return "-"


def _fmt_age(seconds) -> str:
    """A coarse human duration: Ns, Nm, NhMMm or Nd, whichever fits *seconds*.

    A negative value clamps to 0."""
    s = int(max(0, seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d"


def _fmt_since(ts) -> str:
    """Elapsed time since *ts* in _fmt_age's form, for a 'key list' table column.

    '-' when *ts* is None or cannot be read as a number. No 'ago' suffix: the
    column header (Age/Used) supplies that."""
    if ts is None:
        return "-"
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return "-"
    return _fmt_age(time.time() - ts)


def _print_wide_table(table) -> None:
    """Print a Rich table, widened when output is not a terminal.

    Piped or redirected output renders at least 120 columns instead of Rich's
    80-column fallback. An attached terminal keeps its own real width."""
    if console.is_terminal:
        console.print(table)
        return
    from rich.console import Console as _Console
    _Console(width=max(console.size.width, 120)).print(table)


def _fmt_expires(ts) -> str:
    """Compact expiry for the 'key list' table: 'never' (no deadline), 'in
    <age>' (still valid), 'expired' (past), or '-' when *ts* is unreadable.

    list_keys() does not filter expired keys out of its result."""
    if ts is None:
        return "never"
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return "-"
    delta = ts - time.time()
    if delta >= 0:
        return f"in {_fmt_age(delta)}"
    return "[red]expired[/red]"




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
    try:
        auth.set_api_key(key)
    except ValueError as e:
        # set_api_key rejects a key that is too short or uses characters an HTTP
        # header cannot carry. Re-raised as a CLI error so it does not reach the
        # crash/bug-report handler.
        raise click.ClickException(str(e)) from e
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
    failed = auth.clear_api_key()
    # A browser owner session carries its own ADMIN scope snapshot and survives
    # a key roll, so a leftover owner cookie would keep full access after the key
    # is gone. Every browser session is signed out here, mirroring
    # /api/auth/key/clear. Device bearer KEYS in the keystore are untouched.
    revoked = sessions.revoke_all()
    if failed:
        # A security step that FAILED never reports success. The underlying
        # warnings go to debuglog only, so the failure is named here.
        console.print("[red]x[/red] API key NOT fully cleared - credentials may "
                      "still grant access:")
        # The CLI is a LOCAL surface, so it prints the path and the OS error;
        # the HTTP route shows only "what".
        for item in failed:
            console.print(f"  [yellow]-[/yellow] {item['what']}: "
                          f"{item['path']} ({item['error']})")
        console.print("[dim]Delete the listed file(s) by hand, then run "
                      "'localm key clear' again to confirm.[/dim]")
    elif revoked is not None:
        console.print("[green]✓[/green] API key cleared - open mode.")
    if revoked is None:
        # revoke_all returns None only when the store could not be written,
        # which means live sessions REMAIN live and a leftover owner cookie keeps
        # full ADMIN access once any key is configured again. Local surface, so
        # it names the file and the fix.
        console.print("[red]x[/red] Browser sessions were NOT signed out - a "
                      "signed-in browser may still have access:")
        console.print(f"  [yellow]-[/yellow] {sessions.sessions_file()}")
        console.print("[dim]Re-run 'localm key clear' once that file is "
                      "writable, or delete it by hand.[/dim]")
    elif revoked:
        console.print("[dim]Browser sessions were signed out.[/dim]")
    _note_env_override("is still set, so a key remains active from the environment.")




@key_group.command("recover")
def key_recover():
    """Recover owner access after a lockout (run LOCALLY on the server machine).

    Mints a FRESH owner key and prints it once - use this when you have lost the
    owner key but still need to manage the server. Existing scoped DEVICE keys are
    untouched, so devices keep working; only the owner credential is rotated. Live
    browser (cookie) sessions are signed out too, so a captured owner cookie cannot
    outlive the recovery. The old key is not required. To instead drop all auth and
    return to open mode, use 'key clear'."""
    from localm import auth, sessions
    had = auth.get_api_key() is not None
    key = auth.regenerate_key()
    # regenerate_key leaves browser sessions alone, and an owner cookie carries
    # its own ADMIN snapshot exempt from the keystore recheck, so it survives the
    # rotation. Every session is revoked here rather than inside regenerate_key,
    # mirroring /api/auth/key/clear. Device bearer KEYS in the keystore are
    # untouched.
    revoked = sessions.revoke_all()
    console.print("[green]Owner access recovered. New owner key (shown once - "
                  "copy it now):[/green]")
    console.print(f"  [bold]{key}[/bold]")
    console.print("[dim]Clients send it as: Authorization: Bearer <key>[/dim]")
    if had:
        console.print("[dim]The previous owner key no longer works; scoped device "
                      "keys are unchanged.[/dim]")
    if revoked is None:
        # This command always configures a NEW key, so a surviving ADMIN cookie
        # resolves immediately against it and keeps full access. That failure is
        # reported rather than passed over, but it is not raised: the exit code
        # stays 0, matching the credential half of `key clear` and the HTTP route
        # (200 with cleared:false), and the new key printed above is usable.
        console.print("[red]x[/red] Browser sessions were NOT signed out, so a "
                      "captured session cookie may still have access:")
        console.print(f"  [yellow]-[/yellow] {sessions.sessions_file()}")
        console.print("[dim]Delete that file by hand, then re-run "
                      "'localm key recover' to confirm.[/dim]")
    elif revoked:
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
    table.add_column("RAG roots")
    table.add_column("Age")
    table.add_column("Expires")
    table.add_column("Used")
    for k in keys:
        rag_roots = k.get("rag_roots", [])
        table.add_row(str(k.get("id", "")), str(k.get("name", "")),
                      ", ".join(k.get("scopes", [])),
                      str(k.get("fs_access", "none")),
                      ", ".join(rag_roots) if rag_roots else "(unrestricted)",
                      _fmt_since(k.get("created")),
                      _fmt_expires(k.get("expires")),
                      _fmt_since(k.get("last_used")))
    _print_wide_table(table)




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
@click.option("--rag-root", "rag_roots", multiple=True,
              help="A folder this key may index/query via RAG (repeatable). "
                   "Setting this CONFINES the key's RAG reach to exactly these "
                   "folders instead of your home dir + working dir + configured "
                   "allowed roots. Omit to leave the key unrestricted (falls back "
                   "to the global RAG indexing policy, same as today).")
@click.option("--expires-in", "expires_in", type=float, default=None,
              help="Expire this key after N seconds from now (e.g. 3600 for "
                   "1 hour, 86400 for 1 day). Omit for a key that never expires.")
@click.option("--allow-privileged", "allow_privileged", is_flag=True, default=False,
              help="Allow granting privileged scopes (admin, keys:admin, "
                   "plugins:admin, config:write, coder:full) to this key. "
                   "Refused by default so a routine 'key create' cannot mint "
                   "an owner-equivalent credential; pass this to mint one "
                   "deliberately instead of always reaching for the wildcard "
                   "owner key.")
def key_create(name, scopes, fs_access, rag_roots, expires_in, allow_privileged):
    """Mint a named key limited to SCOPES; print the secret once.

    Privileged scopes (admin, keys:admin, plugins:admin, config:write,
    coder:full) are refused unless --allow-privileged is given.

    --fs-access grants host filesystem reach (default none). The owner key always
    has full host access; this is how you let (or deny) a shared device key browse
    the server disk.

    --rag-root grants a per-key RAG-indexing folder allowlist (repeatable, default
    unrestricted). The owner key is never confined by this; use it to hand a
    shared key access to specific document folders only.

    --expires-in sets a relative time-to-live in seconds from now. Omit for a
    key that never expires.
    """
    from localm import auth
    from localm import scopes as S
    expires = time.time() + expires_in if expires_in is not None else None
    try:
        rec = auth.create_key(name, list(scopes), allow_privileged=allow_privileged,
                              fs_access=fs_access,
                              rag_roots=list(rag_roots) or None,
                              expires=expires)
    except (ValueError, PermissionError) as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    rag_note = (f"; rag-roots {', '.join(rec['rag_roots'])}"
                if rec.get("rag_roots") else "")
    expires_note = (f"; expires {_fmt_ts(rec['expires'])}"
                     if rec.get("expires") else "")
    priv_granted = [s for s in rec["scopes"] if s in S.PRIVILEGED_SCOPES]
    priv_note = (f"; [yellow]privileged:[/yellow] {', '.join(priv_granted)}"
                 if priv_granted else "")
    console.print(f"[green]New key '{rec['name']}' "
                  f"({', '.join(rec['scopes'])}; fs-access {rec['fs_access']}"
                  f"{rag_note}{expires_note}{priv_note}) - shown once:[/green]")
    console.print(f"  [bold]{rec['key']}[/bold]")
    console.print(f"[dim]id {rec['id']}; revoke with: "
                  f"localm key rm {rec['id']}[/dim]")
    # Warn at grant time when this key unlocks a plugin the host cannot serve
    # yet: not installed, or missing its pip extras.
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
