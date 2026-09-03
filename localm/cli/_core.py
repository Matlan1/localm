# SPDX-License-Identifier: AGPL-3.0-or-later
import sys
from typing import Optional

# Force UTF-8 output on Windows so Rich's Unicode markup doesn't crash
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import click
from rich.console import Console
from ..bindhost import self_connect_host, url_host


console = Console()






def _exposed_bind_warning(host: str) -> Optional[str]:
    """
    Warning text when binding beyond loopback unsafely. Two unsafe cases, both of
    which serve a takeable LLM API to the whole network:
      * no API key set at all (unauthenticated), or
      * a key is set but is shorter than MIN_KEY_LEN (e.g. a 1-char
        LOCALM_API_KEY env var or a hand-edited auth.key); the floor is enforced
        only at SET time, so an env/file-sourced key bypasses it.
    Returns None when the configuration is safe (loopback, or a strong key).
    """
    from localm.auth import MIN_KEY_LEN, any_key_configured, get_api_key
    from localm.bindhost import is_loopback_host
    if is_loopback_host(host):
        return None
    if not any_key_configured():
        return (
            f"⚠ Binding to {host} WITHOUT authentication - anyone on the network "
            f"can use this server, unload your model, and read every response.\n"
            f"  Set an API key first:  $env:LOCALM_API_KEY = \"<secret>\"  "
            f"(clients send it as a Bearer token)"
        )
    key = get_api_key() or ""
    if len(key) < MIN_KEY_LEN:
        return (
            f"⚠ Binding to {host} with a WEAK API key ({len(key)} "
            f"char{'s' if len(key) != 1 else ''}) - a key this short is trivially "
            f"guessable, so the server is effectively unauthenticated on the "
            f"network.\n"
            f"  Generate one:  localm key generate\n"
            f"  ({MIN_KEY_LEN}+ characters silences this warning, but only a "
            f"randomly generated key is actually hard to guess.)"
        )
    return None




def _resolve_bind_host(cli_host: Optional[str]):
    """Resolve the effective bind host for a fresh server start. Returns
    ``(host, from_config)``.

    Precedence: an explicit ``-H/--host`` wins for that process, and survives
    an in-place restart, which re-execs the same argv. With no explicit flag,
    the GUI-settable ``bind_host`` config key applies (Applies.RESTART: it is
    read in the fresh process, so a Settings-driven bind takes effect across
    the Restart server button). Otherwise loopback.

    ``from_config`` is True when the value came from config, i.e. was possibly
    set from the GUI by a user with NO terminal. A failed precondition past
    that point (no strong API key, TLS unavailable) must degrade LOUDLY to a
    loopback bind rather than exit; explicit CLI binds keep their fail-hard
    behavior.

    A config value that is not even well-FORMED (possible only via a
    hand-edited config.json - PATCH /v1/config and `localm config` both
    validate at write time) is treated as unset, with a warning. Syntax is all
    this helper judges; whether the address is bindable RIGHT NOW is a runtime
    question answered by _bind_preflight_error at the call site."""
    if cli_host is not None:
        return str(cli_host), False
    from localm.config import load_config
    cfg_host = str(load_config().get("bind_host") or "").strip()
    if not cfg_host:
        return "127.0.0.1", False
    from localm.bindhost import is_valid_bind_host
    if not is_valid_bind_host(cfg_host):
        from rich.markup import escape

        from localm.debuglog import logger
        msg = (f"config 'bind_host' is not a bindable address: {cfg_host!r} - "
               f"ignoring it and binding 127.0.0.1 (use an IP literal like "
               f"0.0.0.0, or localhost)")
        logger.warning(msg)
        console.print(f"[yellow]{escape(msg)}[/yellow]")
        return "127.0.0.1", False
    return cfg_host, True


def _bind_preflight_error(host: str) -> Optional[str]:
    """Why *host* cannot be bound on this machine RIGHT NOW, or None when it
    can. Probes with a real throwaway bind to an ephemeral port.

    Catches a SPECIFIC interface IP that is no longer assigned (DHCP gave the
    machine a different address after the value was saved), which uvicorn
    would die on at the socket bind. Loopback and the wildcards bind
    trivially; the probe costs one socket.

    The address can still disappear in the window between this probe and the
    real bind, so a None result is not a guarantee."""
    import socket
    try:
        infos = socket.getaddrinfo(host, 0, type=socket.SOCK_STREAM,
                                   flags=socket.AI_PASSIVE)
        family, stype, proto, _name, addr = infos[0]
        with socket.socket(family, stype, proto) as s:
            s.bind(addr)
        return None
    except OSError as e:
        return str(e)


def _config_tls_pair(cfg: dict):
    """The usable custom TLS pair from config, or ``None``.

    ``(tls_cert, tls_key)`` when both keys are set, both files exist, and the
    pair actually loads as a certificate chain (``SSLContext.load_cert_chain``,
    which also proves the key matches the cert). Anything less returns None
    with a WARNING naming what is wrong, and the caller falls back to the
    built-in certificate, which keeps the bind encrypted rather than dead or
    cleartext.

    The CLI pair (--tls-cert/--tls-key) does NOT come through here: click
    checks existence at parse time and a broken pair then fails uvicorn's own
    startup."""
    import ssl
    from pathlib import Path

    from rich.markup import escape

    from localm.debuglog import logger
    cert = str(cfg.get("tls_cert") or "").strip()
    key = str(cfg.get("tls_key") or "").strip()
    if not cert and not key:
        return None
    problem = None
    if not (cert and key):
        problem = ("both tls_cert and tls_key must be set (only "
                   + ("tls_cert" if cert else "tls_key") + " is)")
    elif not Path(cert).is_file():
        problem = f"tls_cert file not found: {cert}"
    elif not Path(key).is_file():
        problem = f"tls_key file not found: {key}"
    else:
        try:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert, key)
            return cert, key
        except Exception as e:
            problem = f"the pair does not load as a certificate chain: {e}"
    msg = (f"configured TLS certificate pair is unusable ({problem}) - "
           f"using localm's built-in certificate instead")
    logger.warning(msg)
    console.print(f"[yellow]{escape(msg)}[/yellow]")
    return None


def _resolve_tls(host, *, no_tls, tls_cert, tls_key):
    """Decide TLS for a bind and return ``(ssl_certfile, ssl_keyfile)`` - or
    ``(None, None)`` for plain HTTP.

    Built-in TLS: on any non-loopback bind localm mints its own local-CA
    certificate, so the API key and all traffic are encrypted. ``--no-tls``
    forces plain HTTP; ``--tls-cert``/``--tls-key`` override with a
    user-supplied pair on any bind. Raises on a half-specified override; the
    cert-generation path may raise on a broken crypto stack, and the caller
    refuses to fall back to cleartext.

    The GUI-settable config keys are the persistent, flag-less forms and CLI
    flags win over all of them: ``tls_enabled`` False acts as --no-tls (only
    when --no-tls was not itself passed), and a usable ``tls_cert``/
    ``tls_key`` pair acts as the override pair (an UNUSABLE config pair falls
    back to the built-in cert with a warning - see _config_tls_pair; an
    explicit CLI pair keeps its fail-hard behavior and never reads config at
    all).
    """
    if tls_cert or tls_key:
        if not (tls_cert and tls_key):
            raise click.UsageError(
                "--tls-cert and --tls-key must be provided together.")
        return str(tls_cert), str(tls_key)
    from localm.bindhost import is_loopback_host
    from localm.config import load_config
    cfg = load_config()
    if not no_tls and cfg.get("tls_enabled", True) is False:
        no_tls = True
    if no_tls or is_loopback_host(host):
        return None, None
    pair = _config_tls_pair(cfg)
    if pair is not None:
        return pair
    from localm import netname, tls
    from localm.config import home_dir
    # Put the reachable NAMES (localm.local, <hostname>.local, the Tailscale
    # MagicDNS name) in the cert alongside the bind IP. Best-effort; never
    # blocks TLS.
    extra_names = netname.cert_hostnames()
    return tls.ensure_cert(home_dir(), hostnames=[host, *extra_names])




def _setup_tls_or_exit(host, *, no_tls, tls_cert, tls_key):
    """Resolve TLS for *host*, or exit(2) rather than silently serve a network
    bind in cleartext. Returns ``(ssl_certfile, ssl_keyfile)`` (cert is None for
    plain HTTP)."""
    from rich.markup import escape

    try:
        return _resolve_tls(host, no_tls=no_tls, tls_cert=tls_cert, tls_key=tls_key)
    except click.UsageError:
        raise
    except Exception as e:
        console.print(f"[bold red]Could not set up built-in TLS: {escape(str(e))}[/bold red]")
        console.print(
            "[bold red]Refusing to serve a network bind in cleartext. Re-run with "
            "--no-tls to force HTTP (the API key would cross the network "
            "unencrypted), or pass --tls-cert/--tls-key.[/bold red]")
        sys.exit(2)




def _complete_model_name(ctx, param, incomplete):
    """Shell-completion callback: registered model names matching the prefix."""
    try:
        from ..config import load_registry as _lr
        return sorted(n for n in _lr() if n.startswith(incomplete))
    except Exception:
        return []




def _stdout_is_a_pipe() -> bool:
    """True when stdout is a PIPE - the only shape in which a downstream consumer
    (``localm ... | head``, ``| findstr``) can close the far end under us.

    This is the discriminator for a bare EINVAL, which Windows raises for a broken
    pipe AND for a broad set of genuine I/O misuse. On Windows an early pipe close
    surfaces as OSError errno=22 with isinstance(e, BrokenPipeError) False, so the
    exception alone cannot be trusted, while os.fstat reports S_ISFIFO True for
    that same stdout at the handler and False for a console or a file redirect."""
    import os
    import stat
    try:
        return stat.S_ISFIFO(os.fstat(sys.stdout.fileno()).st_mode)
    except Exception:
        # No fileno (a wrapped/replaced stdout) or stdout is gone: not a pipe we
        # can confirm, so an EINVAL here is treated as a REAL error and reported.
        return False


class _GracefulGroup(click.Group):
    """Single, cross-cutting failure handler for the whole CLI.

    Every subcommand (run, gui, serve, coder, setup-llama, ...) is invoked
    through this group, so an unexpected crash anywhere is caught in ONE place:
    it prints "sorry, X went wrong because Y" and offers a prefilled, editable
    bug report. A command that hits a known problem raises bugreport.LocalmError
    with a summary/reason; it does no reporting itself.

    User errors (ClickException / bad usage), Ctrl+C, and clean exits pass
    through untouched."""

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except (SystemExit, KeyboardInterrupt, click.exceptions.Abort,
                click.exceptions.Exit, click.ClickException):
            raise
        except OSError as e:
            # A downstream consumer closing the pipe early (`localm ... | head`,
            # `| findstr`) surfaces as BrokenPipeError, or on Windows as an OSError
            # EPIPE/EINVAL from the stdout flush. Do not report that; exit as if
            # killed by SIGPIPE. A real OSError falls through to reporting.
            #
            # EINVAL counts as a pipe close ONLY when stdout is actually a pipe,
            # because Windows also raises errno 22 for genuine I/O misuse: reading
            # a directory as a file, an invalid or reserved path, a bad handle,
            # some native/ctypes and socket calls. EPIPE needs no such
            # qualification. A genuine EINVAL raised while stdout happens to be a
            # pipe is still read as a pipe close.
            import errno
            if (isinstance(e, BrokenPipeError)
                    or e.errno == errno.EPIPE
                    or (e.errno == errno.EINVAL and _stdout_is_a_pipe())):
                try:
                    sys.stdout.close()
                except Exception:
                    pass
                raise SystemExit(0)
            self._report_failure(ctx, e)
            raise SystemExit(1)
        except Exception as e:  # an actual, unexpected failure
            self._report_failure(ctx, e)
            raise SystemExit(1)

    @staticmethod
    def _report_failure(ctx, e):
        # Guarded so reporting can never itself crash the handler; a failure to
        # report falls back to a one-line message.
        try:
            from localm import bugreport
            interactive = bool(getattr(sys.stdin, "isatty", lambda: False)())
            if isinstance(e, bugreport.LocalmError):
                bugreport.report_failure(
                    summary=e.summary, reason=e.reason,
                    error=e.__cause__ or e, context=e.context,
                    interactive=interactive)
            else:
                bugreport.report_failure(
                    summary="localm hit an unexpected error",
                    reason=str(e), error=e,
                    context={"command": getattr(ctx, "invoked_subcommand", None)},
                    interactive=interactive)
        except Exception:
            # Broken-pipe is handled above, so stdout is live here for a real
            # failure.
            from rich.markup import escape
            console.print(f"[red]localm failed:[/red] {escape(str(e))}")




def running_server(*, allow_url_override: bool = True):
    """The localm server serving this directory: ``(url, headers)``, or None.

    ``url`` is a bare origin (no trailing slash), ``headers`` already carries
    the right credential: the owner key (``LOCALM_API_KEY`` env, else the
    persisted ``auth.key``) when one is configured, otherwise the discovered
    instance's own attach token - the 0600 per-instance registry field that
    the open-mode management gate accepts in place of a key. A caller that
    skips this and builds its own headers gets a 403 on the default, keyless
    install.

    Returns None rather than exiting, leaving the caller to decide what to do
    without a server.

    *allow_url_override*: honour ``LOCALM_URL`` for a different instance. An
    overridden URL has no registry entry, so there is no attach token to fall
    back on and a keyless server there needs ``LOCALM_API_KEY``.
    """
    import os

    from .. import instances
    from ..auth import resolve_bearer_headers
    from ..config import home_dir

    if allow_url_override:
        override = os.environ.get("LOCALM_URL", "").rstrip("/")
        if override:
            return override, resolve_bearer_headers(None)
    entry = instances.find_attachable(home_dir(), instances.resolve_root_dir())
    if entry is None:
        return None
    scheme = entry.get("scheme", "http")
    url = f"{scheme}://{url_host(self_connect_host(entry.get('host')))}:{entry.get('port')}"
    return url, resolve_bearer_headers(entry.get("token"))


def server_call(url, headers, method: str, path: str, *, timeout: float = 30.0,
                params=None, json_body=None, not_found: str = "unsupported") -> tuple:
    """Call *path* on a discovered localm server. Returns ``(state, payload)``.

    *state* is one of:
      ``"ok"``           - payload is the parsed body
      ``"unauthorized"`` - the server wants a credential this client lacks
      ``"unsupported"``  - this server has no such route (an older localm, or
                           a plugin whose routes are not mounted)
      ``"missing"``      - 404 on a route whose 404 means "no such object"
      ``"http"``         - some other HTTP status; payload is ``(code, detail)``
      ``"unreachable"``  - could not connect; payload is a short reason

    *not_found* names what a 404 MEANS on this particular path, because the
    status code alone cannot tell you. On ``/v1/comfy/status`` a 404 is an
    older server with no such route; on ``/api/jobs/<id>/cancel`` the route
    exists and 404s for an id that is not there (or not yours). Pass
    ``not_found="missing"`` where the object, not the route, is what is
    absent. A 405 is never *not_found*; it always maps to ``unsupported``.
    """
    import requests

    from .. import tls
    try:
        r = requests.request(method, f"{url}{path}", headers=headers,
                             params=params, json=json_body, timeout=timeout,
                             verify=tls.requests_verify(url))
    except requests.RequestException as e:
        return "unreachable", type(e).__name__
    if r.status_code in (401, 403):
        return "unauthorized", r.status_code
    if r.status_code == 404:
        return not_found, r.status_code
    if r.status_code == 405:
        # A POST to a path this server does not serve comes back 405, not 404
        # (only GET produces a 404 there), so 405 means the route is absent.
        # Always "unsupported", never *not_found*: a mounted route with a
        # genuinely absent object answers 404 with its own detail.
        return "unsupported", r.status_code
    if not r.ok:
        detail = ""
        try:
            detail = (r.json() or {}).get("detail", "")
        except ValueError:
            detail = (r.text or "")[:200]
        return "http", (r.status_code, detail)
    try:
        return "ok", r.json()
    except ValueError:
        # A 200 whose body is not JSON means something other than localm
        # answered on that port, not an empty/negative result.
        return "http", (r.status_code, "the reply was not JSON")


def report_server_failure(state, payload, what: str) -> None:
    """Print why *what* could not be done, naming WHICH failure happened.

    Never prints a negative RESULT - every branch here is "could not ask".

    *what* is caller-supplied and is escaped. *payload* is escaped for the
    "unreachable" branch (``type(e).__name__``) and for ``detail`` (the
    "http" branch, untrusted server-response text). ``code`` is
    ``r.status_code`` from ``requests``, always an ``int``, so it is left
    unescaped."""
    from rich.markup import escape

    what = escape(what)
    if state == "unreachable":
        console.print(f"[red]Could not reach the localm server[/red] to {what} "
                      f"({escape(str(payload))}).")
    elif state == "unauthorized":
        console.print(f"[red]The localm server refused this request[/red] "
                      f"({what}). Set LOCALM_API_KEY to the server's key, or "
                      "run from the install whose auth.key it uses.")
    elif state == "unsupported":
        console.print(f"[yellow]![/yellow]  This server has no route to {what} "
                      "(it predates this feature).")
    elif state == "missing":
        console.print(f"[red]Could not {what}[/red]: the server has no such "
                      "item (it may have already finished and been forgotten).")
    else:
        code, detail = payload if isinstance(payload, tuple) else (payload, "")
        console.print(f"[red]Could not {what} (HTTP {code})[/red]"
                      + (f": {escape(str(detail))}" if detail else ""))


def no_server_message(what: str) -> None:
    """Say that *what* needs a running localm server, and how to get one.

    *what* is caller-supplied (see report_server_failure); the directory is a
    real filesystem path from ``instances.resolve_root_dir()`` and can
    legitimately contain brackets. Both escaped."""
    from rich.markup import escape

    from .. import instances
    console.print(f"[red]No running localm server found for this directory[/red] "
                  f"- {escape(what)} needs one.")
    console.print(f"[dim]Directory:[/dim] {escape(instances.resolve_root_dir())}")
    console.print("[dim]Start one with[/dim] localm gui  [dim]or[/dim]  "
                  "localm serve <model>[dim], or set LOCALM_URL to target a "
                  "different instance.[/dim]")


def _read_version_for_cli() -> str:
    """Version string for ``localm --version``: the live VERSION file, falling
    back to a static string if unreadable."""
    try:
        from localm._version import read_version
        return read_version()
    except Exception:
        return "0.2.0"




@click.group(cls=_GracefulGroup, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(_read_version_for_cli(), prog_name="localm")
def main() -> None:
    """Run local LLMs offline - HuggingFace and GGUF models, AMD/NVIDIA/CPU."""
    # Install the process-wide graceful-failure net: a crash anywhere - a
    # background thread (preload, jobs, coder), or any uncaught main-thread
    # error - becomes a "sorry X for Y" + bug-report offer instead of a hard
    # crash. _GracefulGroup still handles per-command errors; this covers
    # everything the group does not wrap.
    from localm import bugreport
    from localm.debuglog import honor_env_debug, install_ring_buffer
    # Always-on, in-memory recent-activity buffer a bug report can carry, even
    # without --debug. INFO+ only, so chat content (logged at DEBUG) never
    # enters it.
    install_ring_buffer()
    # Honour LOCALM_DEBUG=1 by opening the log file, not just flipping verbose
    # semantics. No-op unless the env var is set.
    honor_env_debug()
    bugreport.install_global_handlers()


def console_main() -> None:
    """The ``localm`` console-script entry point (pyproject [project.scripts]).

    Guards that the process is inside the project venv, then runs the CLI
    group. Separate from ``main``, which in-process callers invoke directly and
    which applies no venv gate."""
    from localm._venvguard import require_venv
    require_venv()
    main()
