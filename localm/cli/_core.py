# SPDX-License-Identifier: AGPL-3.0-or-later
import sys
from typing import Optional

# Force UTF-8 output on Windows.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import click
from rich.console import Console
from ..bindhost import self_connect_host, url_host


console = Console()






def _exposed_bind_warning(host: str) -> Optional[str]:
    """
    Warning text when binding beyond loopback unsafely. Two unsafe cases:
      * no API key set at all (unauthenticated), or
      * a key is set but is shorter than MIN_KEY_LEN (an env-var or
        hand-edited auth.key value bypasses the set-time floor).
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
            f"  Set a strong key (>= {MIN_KEY_LEN} chars):  "
            f"$env:LOCALM_API_KEY = \"<secret>\""
        )
    return None




def _resolve_bind_host(cli_host: Optional[str]):
    """Resolve the effective bind host for a fresh server start. Returns
    ``(host, from_config)``.

    Precedence: an explicit ``-H/--host`` wins, then the ``bind_host`` config
    key, otherwise loopback.

    ``from_config`` is True when the value came from config. A caller meeting
    a failed precondition past that point (no strong API key, TLS unavailable)
    must degrade LOUDLY to a loopback bind rather than exit; explicit CLI
    binds keep their fail-hard behavior.

    A config value that is not well-FORMED is treated as unset, with a
    warning. Syntax is all this helper judges; whether the address is bindable
    RIGHT NOW is answered by _bind_preflight_error at the call site."""
    if cli_host is not None:
        return str(cli_host), False
    from localm.config import load_config
    cfg_host = str(load_config().get("bind_host") or "").strip()
    if not cfg_host:
        return "127.0.0.1", False
    from localm.bindhost import is_valid_bind_host
    if not is_valid_bind_host(cfg_host):
        from localm.debuglog import logger
        msg = (f"config 'bind_host' is not a bindable address: {cfg_host!r} - "
               f"ignoring it and binding 127.0.0.1 (use an IP literal like "
               f"0.0.0.0, or localhost)")
        logger.warning(msg)
        console.print(f"[yellow]{msg}[/yellow]")
        return "127.0.0.1", False
    return cfg_host, True


def _bind_preflight_error(host: str) -> Optional[str]:
    """Why *host* cannot be bound on this machine RIGHT NOW, or None when it
    can. Probes with a real throwaway bind to an ephemeral port.

    The address can still stop being bindable between this probe and the real
    bind; this is not a TOCTOU-free guarantee."""
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
    pair loads as a certificate chain (``SSLContext.load_cert_chain``, which
    also proves the key matches the cert). Anything less returns None with a
    WARNING naming what is wrong, and the caller falls back to the built-in
    certificate.

    The CLI pair (--tls-cert/--tls-key) does not come through here."""
    import ssl
    from pathlib import Path
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
    console.print(f"[yellow]{msg}[/yellow]")
    return None


def _resolve_tls(host, *, no_tls, tls_cert, tls_key):
    """Decide TLS for a bind and return ``(ssl_certfile, ssl_keyfile)`` - or
    ``(None, None)`` for plain HTTP.

    On any non-loopback bind localm mints its own local-CA certificate.
    ``--no-tls`` forces plain HTTP; ``--tls-cert``/``--tls-key`` override with
    a user-supplied pair on any bind. Raises on a half-specified override, and
    the cert-generation path may raise on a broken crypto stack.

    CLI flags win over the config keys: ``tls_enabled`` False acts as --no-tls
    (only when --no-tls was not itself passed), and a usable ``tls_cert``/
    ``tls_key`` pair acts as the override pair (an UNUSABLE config pair falls
    back to the built-in cert with a warning - see _config_tls_pair; an
    explicit CLI pair never reads config at all).
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
    # Put the reachable names (localm.local, <hostname>.local, the Tailscale
    # MagicDNS name) in the certificate alongside the bind IP. Best-effort;
    # never blocks TLS.
    extra_names = netname.cert_hostnames()
    return tls.ensure_cert(home_dir(), hostnames=[host, *extra_names])




def _setup_tls_or_exit(host, *, no_tls, tls_cert, tls_key):
    """Resolve TLS for *host*, or print an error and exit(2). Returns
    ``(ssl_certfile, ssl_keyfile)`` (cert is None for plain HTTP)."""
    try:
        return _resolve_tls(host, no_tls=no_tls, tls_cert=tls_cert, tls_key=tls_key)
    except click.UsageError:
        raise
    except Exception as e:
        console.print(f"[bold red]Could not set up built-in TLS: {e}[/bold red]")
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
    """True when stdout is a PIPE, from ``os.fstat``. False for a console, a
    file redirect, a wrapped/replaced stdout with no fileno, or a stdout that
    is gone. Qualifies a bare EINVAL in _GracefulGroup.invoke."""
    import os
    import stat
    try:
        return stat.S_ISFIFO(os.fstat(sys.stdout.fileno()).st_mode)
    except Exception:
        # No fileno (a wrapped/replaced stdout), or stdout is gone.
        return False


class _GracefulGroup(click.Group):
    """Click group that catches an unexpected crash in any subcommand, says
    "sorry, X went wrong because Y", and offers a prefilled, editable bug
    report. A command that hits a known problem raises bugreport.LocalmError
    with its own summary/reason.

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
            # EPIPE/EINVAL from the stdout flush: close stdout and exit 0 without
            # reporting. EINVAL counts only when stdout is ACTUALLY a pipe, since
            # Windows also raises errno 22 for genuine I/O misuse; EPIPE needs no
            # such qualification. Any other OSError falls through to reporting.
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
        # Guarded so a failure inside reporting cannot crash the handler.
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
            # A broken pipe is handled earlier, so stdout is live here.
            console.print(f"[red]localm failed:[/red] {e}")




def running_server(*, allow_url_override: bool = True):
    """The localm server serving this directory: ``(url, headers)``, or None.

    ``url`` is a bare origin (no trailing slash), ``headers`` already carries
    the right credential: the owner key (``LOCALM_API_KEY`` env, else the
    persisted ``auth.key``) when one is configured, otherwise the discovered
    instance's own attach token - the 0600 per-instance registry field that
    the open-mode management gate accepts in place of a key.

    Returns None rather than exiting when nothing serves this directory.

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

    *not_found* names what a 404 MEANS on this particular path. On
    ``/v1/comfy/status`` a 404 is an older server with no such route; on
    ``/api/jobs/<id>/cancel`` the route exists and 404s for an id that is not
    there (or not yours). Pass ``not_found="missing"`` where the object, not
    the route, is what is absent. 405 is never *not_found*.
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
        # A POST to a path this server does not serve comes back 405, not
        # 404; only GET produces a 404 there. Always "unsupported", never
        # *not_found*: a mounted route whose object is absent answers 404.
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
        # A 200 whose body is not JSON: something other than localm answered
        # on that port.
        return "http", (r.status_code, "the reply was not JSON")


def report_server_failure(state, payload, what: str) -> None:
    """Print why *what* could not be done, naming WHICH failure happened.

    Every branch says "could not ask"; none prints a negative RESULT.
    """
    if state == "unreachable":
        console.print(f"[red]Could not reach the localm server[/red] to {what} "
                      f"({payload}).")
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
                      + (f": {detail}" if detail else ""))


def no_server_message(what: str) -> None:
    """Say that *what* needs a running localm server, and how to get one."""
    from .. import instances
    console.print(f"[red]No running localm server found for this directory[/red] "
                  f"- {what} needs one.")
    console.print(f"[dim]Directory:[/dim] {instances.resolve_root_dir()}")
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
        return "0.1.5rc3"




@click.group(cls=_GracefulGroup, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(_read_version_for_cli(), prog_name="localm")
def main() -> None:
    """Run local LLMs offline - HuggingFace and GGUF models, AMD/NVIDIA/CPU."""
    # Install the process-wide graceful-failure net: a crash in a background
    # thread (preload, jobs, coder) or any uncaught main-thread error becomes a
    # "sorry X for Y" + bug-report offer instead of a hard crash.
    from localm import bugreport
    from localm.debuglog import honor_env_debug, install_ring_buffer
    # Always-on, in-memory recent-activity buffer for bug reports, with or
    # without --debug. INFO+ only, so chat content (logged at DEBUG) never
    # enters it.
    install_ring_buffer()
    # Honour LOCALM_DEBUG=1 by opening the log file, not just flipping verbose
    # semantics. No-op unless the env var is set.
    honor_env_debug()
    bugreport.install_global_handlers()


def console_main() -> None:
    """The ``localm`` console-script entry point (pyproject [project.scripts]).

    Guards that we are inside the project venv, then runs the CLI group.
    In-process callers invoke ``main`` directly and skip the venv gate."""
    from localm._venvguard import require_venv
    require_venv()
    main()
