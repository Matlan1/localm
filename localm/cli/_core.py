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
      * a key is set but is too weak (e.g. a 1-char LOCALM_API_KEY env var or a
        hand-edited auth.key) - the 8-char floor is only enforced at SET time, so
        an env/file-sourced key bypasses it; a trivially-guessable owner secret
        is no better than none (NEW-O).
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

    Precedence: an explicit ``-H/--host`` always wins for that process - and
    survives an in-place restart, which re-execs the same argv (see
    http_server._restart_argv). With no explicit flag, the GUI-settable
    ``bind_host`` config key applies (Applies.RESTART: this read, running in
    the fresh process, is what makes a Settings-driven bind take effect across
    the Restart server button). Otherwise loopback.

    ``from_config`` tells the caller the value came from config, i.e. was
    possibly set from the GUI by a user with NO terminal - a failed
    precondition past that point (no strong API key, TLS unavailable) must
    degrade LOUDLY to a loopback bind rather than exit, or the server dies
    with no terminal-free way back. Explicit CLI binds keep their fail-hard
    behavior (the operator typing -H is watching a terminal).

    A config value that is not even well-FORMED (possible only via a
    hand-edited config.json - PATCH /v1/config and `localm config` both
    validate at write time) is treated as unset, with a warning. Syntax is all
    this helper can judge; whether the address is bindable RIGHT NOW (a
    specific interface IP can go stale when DHCP reassigns the machine) is a
    runtime question answered by _bind_preflight_error at the call site in
    plugins/gui/cli.py - handing a stale address to uvicorn would kill the
    server at startup, the exact no-way-back failure above."""
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

    Exists because syntax validation cannot see the commonest real failure of
    the ``bind_host`` config key's own recommended use: a SPECIFIC interface
    IP that is no longer assigned, because DHCP gave the machine a different
    address some time after the value was saved. Handing such a host to the
    server kills the process at the socket bind (uvicorn exits on a failed
    bind, and portmux's uvicorn fallback re-tries the same host) - for a
    config-driven bind that is a locked-out user with no terminal, so the
    caller falls back to loopback instead. Loopback and the wildcards bind
    trivially; the probe costs one socket.

    Known residual, stated rather than glossed: the address can still
    disappear in the window between this probe and the real bind. The probe
    closes the common stale-at-boot case; it is not a TOCTOU-free guarantee,
    and the explicit-CLI path (-H) deliberately keeps today's fail-hard
    behavior in front of the operator who typed it."""
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
    with a WARNING naming what is wrong, so the caller falls back to the
    built-in certificate: a config-sourced pair is applied at a startup nobody
    may be watching (possibly set from the GUI, applied by the Restart
    button), so a broken pair must not become a dead server (uvicorn raises on
    an unloadable pair) - and must not become cleartext either. Falling back
    to the built-in cert keeps the bind encrypted and the server reachable;
    the warning keeps the substitution honest (clients pinning the custom cert
    will refuse the built-in one, which is the loud, safe direction).

    The CLI pair (--tls-cert/--tls-key) deliberately does NOT come through
    here: click checks existence at parse time and a broken pair then fails
    uvicorn's own startup in front of the operator who typed it."""
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

    Built-in TLS (NET-1): on any non-loopback bind localm mints its own local-CA
    certificate so HTTPS works out of the box (the API key and all traffic are
    encrypted). ``--no-tls`` forces plain HTTP (the key would cross the network
    in cleartext); ``--tls-cert``/``--tls-key`` override with a user-supplied
    pair on any bind. Raises on a half-specified override; the cert-generation
    path may raise on a broken crypto stack and the caller refuses to fall back
    to cleartext.

    The GUI-settable config keys are the persistent, flag-less forms and CLI
    flags win over all of them: ``tls_enabled`` False acts as --no-tls (only
    when --no-tls was not itself passed - there is no positive --tls flag, so
    an absent flag cannot veto the config), and a usable ``tls_cert``/
    ``tls_key`` pair acts as the override pair (an UNUSABLE config pair falls
    back to the built-in cert with a warning instead of dying or serving
    cleartext - see _config_tls_pair; an explicit CLI pair keeps its fail-hard
    behavior and never reads config at all).
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
    # Cover the reachable NAMES too (localm.local, <hostname>.local, the Tailscale
    # MagicDNS name) so name-based HTTPS has no cert-name-mismatch warning after
    # the one-time CA trust - not just the bind IP. Best-effort; never blocks TLS.
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
    pipe AND for a broad set of genuine I/O misuse (see _GracefulGroup.invoke). It
    is not a guess: on Windows a real early pipe close was MEASURED to surface as
    OSError errno=22 with isinstance(e, BrokenPipeError) False, so the exception
    alone cannot be trusted - while os.fstat reports S_ISFIFO True for that same
    stdout at the handler, and False for a console or a file redirect.

    Deliberately NOT "flush stdout and see if it fails": the flush that broke the
    pipe already discarded the buffer, so by the time this runs a second flush
    SUCCEEDS on a genuinely dead pipe (measured) - that probe reports healthy for
    the exact case it is meant to catch. A zero-byte write is no better (measured:
    it returns 0 on a broken pipe)."""
    import os
    import stat
    try:
        return stat.S_ISFIFO(os.fstat(sys.stdout.fileno()).st_mode)
    except Exception:
        # No fileno (a wrapped/replaced stdout) or stdout is gone: not a pipe we
        # can confirm, so an EINVAL here is treated as a REAL error and reported.
        # Failing toward reporting is the safe direction - a spurious report is
        # noise, a swallowed failure is a lie (rule 5).
        return False


class _GracefulGroup(click.Group):
    """Single, cross-cutting failure handler for the whole CLI.

    Every subcommand (run, gui, serve, coder, setup-llama, ...) is invoked
    through this group, so an unexpected crash anywhere is caught in ONE place:
    we say "sorry, X went wrong because Y" and offer a prefilled, editable bug
    report. A command that hits a known problem just raises bugreport.LocalmError
    with a good summary/reason; it does not know about reporting itself.

    User errors (ClickException / bad usage), Ctrl+C, and clean exits pass
    through untouched - those are not bugs."""

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except (SystemExit, KeyboardInterrupt, click.exceptions.Abort,
                click.exceptions.Exit, click.ClickException):
            raise
        except OSError as e:
            # A downstream consumer closing the pipe early (`localm ... | head`,
            # `| findstr`) surfaces as BrokenPipeError, or on Windows as an OSError
            # EPIPE/EINVAL from the stdout flush. That is NOT a bug: do not report it
            # (reporting would write to the same dead stdout and re-crash), and exit
            # as if killed by SIGPIPE. A real OSError falls through to reporting.
            #
            # EINVAL must be qualified by stdout ACTUALLY being a pipe (REG-555).
            # Windows raises errno 22 for a broad set of GENUINE I/O misuse - reading
            # a directory as a file, an invalid or reserved path, a bad handle, some
            # native/ctypes and socket calls. Accepting a bare EINVAL as "the pipe
            # closed" meant any of those exited 0, printed nothing and filed no
            # report: the command hard-failed and told the user, and every script
            # checking the exit code, that it had SUCCEEDED. That is the
            # fail-closed-to-silent-success shape rule 5 exists to forbid, and it hid
            # real bugs.
            #
            # EPIPE needs no such qualification (it means exactly "broken pipe", and
            # Python maps it to BrokenPipeError anyway - verified, not assumed).
            #
            # Known residual, stated rather than papered over: a genuine EINVAL
            # raised WHILE stdout happens to be a pipe is still read as a pipe close.
            # Nothing available at this layer separates those two (the measurements
            # in _stdout_is_a_pipe rule out the obvious probes), and the alternative -
            # a bug report on every legitimate `| head` - is the regression this
            # handler was written to fix. The common shapes (a terminal, a file
            # redirect, no pipe at all) are now reported correctly.
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
        # Reporting must never itself crash the handler (that would turn a caught bug
        # into the hard crash we are trying to avoid), so guard it - and if reporting
        # fails, write the fallback to STDERR, since STDOUT may be the very pipe that
        # just broke.
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
            # Broken-pipe is already handled above (before this point), so stdout is
            # live here for a real failure - the original fallback is safe.
            from rich.markup import escape
            console.print(f"[red]localm failed:[/red] {escape(str(e))}")




def running_server(*, allow_url_override: bool = True):
    """The localm server serving this directory: ``(url, headers)``, or None.

    ``url`` is a bare origin (no trailing slash), ``headers`` already carries
    the right credential: the owner key (``LOCALM_API_KEY`` env, else the
    persisted ``auth.key``) when one is configured, otherwise the discovered
    instance's own attach token - the 0600 per-instance registry field that
    the open-mode management gate accepts in place of a key (#953). A caller
    that skips this and builds its own headers gets a 403 on the default,
    keyless install, which is how localm runs out of the box.

    Returns None rather than exiting so a caller can decide: some verbs need a
    server and must say so, while ``comfy status`` still has an honest partial
    answer to give without one.

    *allow_url_override*: honour ``LOCALM_URL`` for a different instance, the
    same escape hatch ``localm unload`` offers. An overridden URL has no
    registry entry, so there is no attach token to fall back on and a keyless
    server there needs ``LOCALM_API_KEY``.
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

    The same five-outcome split ``localm.selfclient.read_activity`` keeps, for
    the same reason: folding any of them into the others reports an answer on
    the evidence of never having got one. ``unsupported`` matters most here -
    an older server with no ``/v1/comfy/status`` must not read as "ComfyUI is
    not running".

    *not_found* names what a 404 MEANS on this particular path, because the
    status code alone cannot tell you. On ``/v1/comfy/status`` a 404 is an
    older server with no such route; on ``/api/jobs/<id>/cancel`` the route
    exists and 404s for an id that is not there (or not yours). Same code, two
    unrelated answers, and a caller that printed "this server predates the
    feature" for a mistyped job id would be reporting the wrong one. Pass
    ``not_found="missing"`` where the object, not the route, is what is absent.
    405 is never *not_found* - see the comment at that branch.
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
        # MEASURED against a real running server, because the obvious
        # assumption is wrong: a POST to a path this server does not serve
        # comes back 405, NOT 404. Only GET produces a 404 there. So reading
        # 404 alone as "no such route" made `comfy start` report "Could not
        # start ComfyUI (HTTP 405)" on a server with no media plugin
        # installed, instead of falling through to the route that could
        # actually start it.
        #
        # Always "unsupported", never *not_found*: 405 is a statement about
        # the path and method, and can never mean "the object you named is
        # missing" - a mounted route with a genuinely absent object answers
        # 404 with its own detail (measured: POST /api/jobs/<unknown>/cancel
        # gives 404 "No such job: ...").
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

    Never prints a negative RESULT - every branch here is "could not ask", and
    a caller that turned any of them into "it is not running" or "nothing to
    do" would be stating an answer it never obtained.

    *what* is a caller-supplied description (every current caller happens to
    pass a hardcoded literal, but this is a shared, generic helper with no way
    to enforce that for a future or different caller - escape it regardless,
    the same defense-in-depth ``rag.py``'s already-validated collection names
    use). *payload* for the "unreachable" branch is ``type(e).__name__`` -
    not numeric, so escaped too rather than trusted to stay bracket-free.
    ``detail`` (the "http" branch) is server-response text and genuinely
    untrusted. ``code`` is ``r.status_code`` from ``requests`` - always a real
    ``int`` in every path that reaches here, so it is left unescaped."""
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
    """Version string for ``localm --version``: the live VERSION file (so it tracks
    a code-only self-update), falling back to a static string if unreadable."""
    try:
        from localm._version import read_version
        return read_version()
    except Exception:
        return "0.1.5rc3"




@click.group(cls=_GracefulGroup, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(_read_version_for_cli(), prog_name="localm")
def main() -> None:
    """Run local LLMs offline - HuggingFace and GGUF models, AMD/NVIDIA/CPU."""
    # Install the process-wide graceful-failure net so a crash anywhere - a
    # background thread (preload, jobs, coder), or any uncaught main-thread
    # error - becomes a "sorry X for Y" + bug-report offer, not a hard crash.
    # The _GracefulGroup above still handles per-command errors with nicer
    # context; this covers everything the group does not wrap.
    from localm import bugreport
    from localm.debuglog import honor_env_debug, install_ring_buffer
    # Always-on, in-memory recent-activity buffer so a bug report carries what the
    # app was doing before it broke - even without --debug (a tester has no log
    # file). INFO+ only, so chat content (logged at DEBUG) never enters it.
    install_ring_buffer()
    # Honour LOCALM_DEBUG=1 as a real debug request: open the log file, not just
    # flip verbose semantics (REC-DEBUGENV). No-op unless the env var is set.
    honor_env_debug()
    bugreport.install_global_handlers()


def console_main() -> None:
    """The ``localm`` console-script entry point (pyproject [project.scripts]).

    Guards that we are inside the project venv, then runs the CLI group. Kept
    SEPARATE from ``main`` so in-process callers - the test suite's CliRunner and
    the ``localm coder`` route - invoke the group directly without the venv gate;
    only the stray-global-exe path (a separate ``pip install``) hits it (NEW-J)."""
    from localm._venvguard import require_venv
    require_venv()
    main()
