# SPDX-License-Identifier: AGPL-3.0-or-later
import sys
from typing import Optional

# Force UTF-8 output on Windows so Rich's Unicode markup doesn't crash
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import click
from rich.console import Console


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
            f"  Set a strong key (>= {MIN_KEY_LEN} chars):  "
            f"$env:LOCALM_API_KEY = \"<secret>\""
        )
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
    """
    if tls_cert or tls_key:
        if not (tls_cert and tls_key):
            raise click.UsageError(
                "--tls-cert and --tls-key must be provided together.")
        return str(tls_cert), str(tls_key)
    from localm.bindhost import is_loopback_host
    if no_tls or is_loopback_host(host):
        return None, None
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
            console.print(f"[red]localm failed:[/red] {e}")




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
    url = f"{scheme}://{entry.get('host', '127.0.0.1')}:{entry.get('port')}"
    return url, resolve_bearer_headers(entry.get("token"))


def server_call(url, headers, method: str, path: str, *, timeout: float = 30.0,
                params=None, json_body=None, not_found: str = "unsupported") -> tuple:
    """Call *path* on a discovered localm server. Returns ``(state, payload)``.

    *state* is one of:
      ``"ok"``           - payload is the parsed body
      ``"unauthorized"`` - the server wants a credential this client lacks
      ``"unsupported"``  - 404: this server has no such route (an older localm)
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
