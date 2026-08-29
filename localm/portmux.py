# SPDX-License-Identifier: AGPL-3.0-or-later
"""Same-port HTTP -> HTTPS handling for network binds.

When localm binds past loopback it serves HTTPS (built-in TLS, see ``tls.py``).
A browser that opens the plain ``http://`` URL on that port then speaks cleartext
to a TLS socket, the handshake fails, and the user sees a bare "connection
reset". This module handles that case WITHOUT a reverse proxy: it owns the public
listening socket, peeks the first byte of every connection, and either

  * TLS (the handshake starts with byte ``0x16``) -> transparently relays the
    raw bytes to an internal loopback uvicorn that terminates TLS and serves the
    app (TLS still terminates at uvicorn; this module only shuffles bytes), or
  * plaintext (an HTTP request) -> answers with a ``308`` redirect to the
    ``https://`` URL plus a small HTML catch page.

BOTH BINDS COME THROUGH HERE, INCLUDING THE PLAIN-HTTP LOOPBACK DEFAULT. The TLS
path relays to an internal TLS uvicorn as described above; the plain-HTTP path
runs the SAME first-byte peek (``_serve_async_plain``) so a client that wrongly
opens a TLS connection on this HTTP port is closed cleanly at the socket layer
instead of feeding a ClientHello into uvicorn's HTTP parser. Any setup failure
falls back to a direct ``uvicorn.run``.

**CONSEQUENCE: the app never sees the client's socket.** Every accepted
connection is relayed over a fresh internal loopback connection, so the peer
uvicorn reports is PORTMUX'S OWN socket, owned by the server process, and
resolves to the SERVER'S OWN pid rather than the client's. Any local-identity or
peer-credential reasoning at the app layer is therefore invalid, not merely
imprecise: a check built on it would answer "trusted" for every caller while
appearing to ask.

Nothing trusts ``request.client.host`` for a security decision - the
loopback/network split is made from the configured bind host (see ``gui/web.py``
and ``http_server.py``'s ``bind_host`` gates) - and no such check may be added.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
import sys
from typing import Optional

from localm.netlisten import create_listen_socket

# Child of the "localm" logger so records flow through its handlers (the debug
# file handler and the fd-2-stable console handler).
_log = logging.getLogger("localm.portmux")

# A TLS record begins with the handshake content-type byte 0x16; a plaintext
# HTTP request begins with an ASCII method letter. One byte distinguishes them.
_TLS_FIRST_BYTE = 0x16

# Seconds a stop gives in-flight responses before cancelling them, passed to
# uvicorn as timeout_graceful_shutdown at every bind. Left unset, a stop waits
# for the longest open response. See test_the_bound_is_short_enough_to_feel_immediate.
GRACEFUL_SHUTDOWN_TIMEOUT = 3.0

# Host header shapes accepted before being reflected into a redirect:
# host[:port] for a DNS name or IPv4, or [v6]:port for IPv6.
_HOST_RE = re.compile(r"^[A-Za-z0-9.\-]+(:\d+)?$")
_HOST6_RE = re.compile(r"^\[[0-9A-Fa-f:]+\](:\d+)?$")
# A request target we are willing to preserve across the redirect: an absolute
# path of printable ASCII with no spaces or control characters.
_PATH_RE = re.compile(r"^/[!-~]*$")

_READ_CHUNK = 65536


def _crash_watchdog_disabled() -> bool:
    import os
    return os.environ.get("LOCALM_CRASH_WATCHDOG", "").strip().lower() in (
        "0", "off", "false", "no")


def _spawn_crash_recovery_watchdog(*, host: str, port: int, tls: bool,
                                   instance_id: Optional[str]) -> None:
    """Spawn ``scripts/crash_recovery_watchdog.py`` DETACHED, watching this
    process's own pid, so it can relaunch the server if this process dies
    without reaching ``disarm_crash_guard``. Never raises: a watchdog that
    fails to spawn must not block or fail server startup. Set
    ``LOCALM_CRASH_WATCHDOG=off`` to disable it."""
    if not instance_id or _crash_watchdog_disabled():
        return
    try:
        import json
        import os
        import subprocess

        from localm import bugreport
        from localm.bindhost import self_connect_host
        from localm.updater import repo_root

        script = repo_root() / "scripts" / "crash_recovery_watchdog.py"
        if not script.is_file():
            return
        crash_dir = bugreport._crash_dir(home=None)
        relaunch_argv = [sys.executable, "-m", "localm", *sys.argv[1:]]
        if port:
            relaunch_argv += ["-p", str(port)]
        argv = [sys.executable, str(script),
               "--pid", str(os.getpid()),
               "--host", self_connect_host(host),
               "--port", str(port),
               "--scheme", "https" if tls else "http",
               "--instance-id", str(instance_id),
               "--crash-dir", str(crash_dir),
               "--relaunch-argv", json.dumps(relaunch_argv)]
        # A prior watchdog's own relaunch of THIS process sets this so the
        # rolling crash-storm window (each watchdog is a short-lived, one-shot
        # process, so it cannot keep the count in its own memory) survives
        # across the whole relaunch chain rather than resetting to zero every
        # time a fresh watchdog is spawned.
        restart_history = os.environ.get("LOCALM_CRASH_WATCHDOG_HISTORY", "")
        if restart_history:
            argv += ["--restart-history", restart_history]
        kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL, close_fds=True)
        if sys.platform == "win32":
            kwargs["creationflags"] = (
                getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200))
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen(argv, **kwargs)
    except Exception as e:
        try:
            _log.warning("could not spawn the crash-recovery watchdog: %s", e)
        except Exception:
            pass


def run_server(
    app,
    host: str,
    port: int,
    ssl_certfile: Optional[str] = None,
    ssl_keyfile: Optional[str] = None,
    log_level: str = "warning",
) -> None:
    """Serve *app* on ``(host, port)``, blocking until interrupted.

    Without TLS this is a plain ``uvicorn.run``. With TLS it serves HTTPS and
    also catches a plain-HTTP request on the same port with an https redirect,
    falling back to a direct TLS bind if the demultiplexer cannot start.

    Pure transport: mDNS name advertising lives in the CLI that also prints the
    reachable URLs (``localm serve`` / ``localm gui``), so the advertised name and
    the printed name can never disagree - see ``localm/netname.py`` and the
    ``start_advertiser`` callers.
    """
    import uvicorn

    from localm import bugreport
    # Report a prior hard crash, then arm the crash guard for this run. Disarmed
    # in the finally on a clean exit. instance_id scopes the marker to this
    # instance. *app* is a generic ASGI callable, so .state is read via getattr.
    instance_id = getattr(getattr(app, "state", None), "instance_id", None)
    bugreport.check_and_report_prior_crash()
    bugreport.arm_crash_guard(context={"host": host, "port": port,
                                        "tls": bool(ssl_certfile)},
                              instance_id=instance_id)
    _spawn_crash_recovery_watchdog(host=host, port=port, tls=bool(ssl_certfile),
                                   instance_id=instance_id)

    try:
        if not ssl_certfile:
            # Plain-HTTP bind. Fronted with the same first-byte peek the TLS path
            # uses, so a TLS connection opened on this HTTP port is closed at the
            # socket layer instead of reaching uvicorn's HTTP parser.
            try:
                asyncio.run(_serve_async_plain(app, host, port, log_level))
            except KeyboardInterrupt:
                pass
            except Exception:   # pragma: no cover - defensive fallback
                # Fall back to a direct uvicorn.run if the peek layer fails.
                import traceback
                traceback.print_exc()
                _run_uvicorn_on_socket(uvicorn, app, host, port,
                                       log_level=log_level)
            return

        try:
            asyncio.run(_serve_async(app, host, port, ssl_certfile, ssl_keyfile,
                                     log_level))
        except KeyboardInterrupt:
            pass
        except Exception:   # pragma: no cover - defensive fallback
            # Fall back to a direct TLS bind if the peek layer fails.
            import traceback
            traceback.print_exc()
            _run_uvicorn_on_socket(uvicorn, app, host, port,
                                   log_level=log_level,
                                   ssl_certfile=ssl_certfile,
                                   ssl_keyfile=ssl_keyfile)
    finally:
        bugreport.disarm_crash_guard(instance_id=instance_id)


def _run_uvicorn_on_socket(uvicorn, app, host, port, *, log_level,
                           ssl_certfile=None, ssl_keyfile=None) -> None:
    """The last-resort direct uvicorn bind, on a socket built the same way the
    normal path builds it.

    ``uvicorn.run(host=..., port=...)`` reaches asyncio's create_server and
    silently re-applies IPV6_V6ONLY, which would serve IPv6 only on a ``::``
    bind and contradict the URLs already printed. ``Server.run(sockets=[...])``
    takes the prepared socket instead, so the reachable set is the same on the
    degraded path as on the normal one.

    If even the socket cannot be built, this falls back to uvicorn's own binding
    and logs a warning naming the failure."""
    config_kwargs = dict(app=app, log_level=log_level,
                         timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_TIMEOUT)
    if ssl_certfile:
        config_kwargs.update(ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)
    try:
        sock = create_listen_socket(host, port)
    except OSError as e:
        _log.warning("portmux: could not build the listening socket for %s:%s "
                     "(%s); falling back to uvicorn's own bind, which serves "
                     "IPv6 only for a :: host", host, port, e)
        uvicorn.run(host=host, port=port, **config_kwargs)
        return
    server = uvicorn.Server(uvicorn.Config(host=host, port=port, **config_kwargs))
    server.run(sockets=[sock])


def _track_conn_task(inflight: "set[asyncio.Task]", coro) -> None:
    """Schedule *coro* as a tracked, fire-and-forget per-connection task.

    ``asyncio.start_server``'s ``client_connected_cb`` gives no way to keep a
    reference to the Task it creates when the callback returns a coroutine
    (``asyncio.streams.StreamReaderProtocol`` just does ``loop.create_task(res)``
    and drops it) - so a connection still blocked in ``_relay``/``_pump`` at
    shutdown is invisible to ``Server.wait_closed()`` (which only waits for the
    LISTENING socket to stop accepting, never for handler tasks already running)
    and is destroyed mid-flight when the event loop tears down ("Task was
    destroyed but it is pending!"). Tracking the Task here lets shutdown cancel
    and await it instead."""
    task = asyncio.ensure_future(coro)
    inflight.add(task)
    task.add_done_callback(inflight.discard)


async def _cancel_inflight_conns(inflight: "set[asyncio.Task]") -> None:
    """Cancel and await every still-pending task tracked via
    :func:`_track_conn_task`, so shutdown never abandons a connection mid-relay.
    A task that already finished on its own (the common case) is not touched."""
    pending = [t for t in inflight if not t.done()]
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def _serve_async(app, host, port, ssl_certfile, ssl_keyfile, log_level) -> None:
    import uvicorn

    # Internal loopback uvicorn that terminates TLS and serves the real app on an
    # ephemeral port (0 -> the OS picks a free one).
    config = uvicorn.Config(
        app, host="127.0.0.1", port=0, log_level=log_level,
        ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_TIMEOUT,
    )
    server = uvicorn.Server(config)
    serve_task = asyncio.ensure_future(server.serve())

    # Wait until uvicorn is actually listening so we know the internal port.
    while not server.started and not serve_task.done():
        await asyncio.sleep(0.02)
    if serve_task.done():
        serve_task.result()   # re-raise uvicorn's startup error
        return
    internal_port = server.servers[0].sockets[0].getsockname()[1]
    _harden_uvicorn_logging()

    inflight: "set[asyncio.Task]" = set()

    def _on_conn(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        _track_conn_task(inflight, _handle_conn(reader, writer, internal_port, port))

    # Family-aware listening socket, not start_server(host=..., port=...):
    # asyncio forces IPV6_V6ONLY on every AF_INET6 socket it builds, which would
    # make `-H ::` IPv6-only.
    _lsock = create_listen_socket(host, port)
    try:
        demux = await asyncio.start_server(_on_conn, sock=_lsock)
    except BaseException:
        _lsock.close()
        raise

    # Keeps the Windows event loop waking up so it can process Ctrl+C.
    wakeup_task = None
    if sys.platform == "win32":
        async def _wakeup():
            while True:
                await asyncio.sleep(0.5)
        wakeup_task = asyncio.ensure_future(_wakeup())

    try:
        await serve_task          # runs until Ctrl+C / should_exit
    finally:
        if wakeup_task:
            wakeup_task.cancel()
        demux.close()
        # Cancel in-flight connections BEFORE wait_closed(): it blocks until every
        # accepted connection has detached, which never happens on its own for one
        # parked mid-_relay/_pump.
        await _cancel_inflight_conns(inflight)
        try:
            await demux.wait_closed()
        except Exception:         # pragma: no cover
            pass
        server.should_exit = True


async def _serve_async_plain(app, host, port, log_level) -> None:
    """Serve plain HTTP behind the same first-byte peek as the TLS path: real HTTP
    is relayed to an internal uvicorn; a TLS handshake on this HTTP port (a client
    that wrongly tried HTTPS) is closed cleanly and surfaced once, so uvicorn's
    HTTP parser never sees the TLS bytes and the 'Invalid HTTP request' flood
    cannot happen at the source."""
    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level=log_level,
                            timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_TIMEOUT)
    server = uvicorn.Server(config)
    serve_task = asyncio.ensure_future(server.serve())

    while not server.started and not serve_task.done():
        await asyncio.sleep(0.02)
    if serve_task.done():
        serve_task.result()   # re-raise uvicorn's startup error
        return
    internal_port = server.servers[0].sockets[0].getsockname()[1]
    _harden_uvicorn_logging()

    state = {"warned": False, "count": 0}
    inflight: "set[asyncio.Task]" = set()

    def _on_conn(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        _track_conn_task(
            inflight, _handle_conn_plain(reader, writer, internal_port, port, state))

    # Family-aware listening socket, not start_server(host=..., port=...):
    # asyncio forces IPV6_V6ONLY on every AF_INET6 socket it builds, which would
    # make `-H ::` IPv6-only.
    _lsock = create_listen_socket(host, port)
    try:
        demux = await asyncio.start_server(_on_conn, sock=_lsock)
    except BaseException:
        _lsock.close()
        raise

    # Keeps the Windows event loop waking up so it can process Ctrl+C.
    wakeup_task = None
    if sys.platform == "win32":
        async def _wakeup():
            while True:
                await asyncio.sleep(0.5)
        wakeup_task = asyncio.ensure_future(_wakeup())

    try:
        await serve_task          # runs until Ctrl+C / should_exit
    finally:
        if wakeup_task:
            wakeup_task.cancel()
        demux.close()
        # Cancel in-flight connections BEFORE wait_closed(); see _serve_async.
        await _cancel_inflight_conns(inflight)
        try:
            await demux.wait_closed()
        except Exception:         # pragma: no cover
            pass
        server.should_exit = True


async def _handle_conn_plain(reader, writer, internal_port, public_port, state) -> None:
    """Peek the first byte: a TLS ClientHello (wrong scheme on the HTTP port) is
    closed cleanly and noted; anything else is a real HTTP connection and is
    relayed to the internal uvicorn unchanged.

    One try/finally around the WHOLE body, not just the part after the first
    byte: this task can be cancelled by :func:`_cancel_inflight_conns` on
    shutdown, and cancellation can land anywhere including inside the initial
    ``readexactly``, which must still close the writer."""
    try:
        first = await reader.readexactly(1)
        if first[0] == _TLS_FIRST_BYTE:
            # A TLS handshake cannot be completed on a plain-HTTP port, so the
            # connection is closed and the cause surfaced once.
            _note_tls_on_http(public_port, state)
        else:
            await _relay(first, reader, writer, internal_port)
    except (asyncio.IncompleteReadError, ConnectionError, OSError):
        pass
    finally:
        _safe_close(writer)


def _note_tls_on_http(public_port, state) -> None:
    """Surface a wrong-scheme (HTTPS-on-the-HTTP-port) connection honestly without
    flooding: one prominent notice with the cause + fix, the rest counted at debug
    level. The first notice is written to a STABLE stderr duplicate so it is
    immune to the model-load fd-2 redirect; writing through the live stderr during
    that window turns a benign reject into a WinError-6 cascade."""
    state["count"] += 1
    if state["warned"]:
        # Already surfaced once this run; further occurrences are counted at debug
        # level only.
        try:
            _log.debug("TLS handshake on the plain-HTTP port %d again (count=%d)",
                       public_port, state["count"])
        except Exception:
            pass
        return
    state["warned"] = True
    msg = (
        "A client tried to use HTTPS on the plain-HTTP port %d. This is almost "
        "always a browser that cached an HTTPS upgrade for this address (HSTS, a "
        "service worker, or HTTPS-First/Only mode): it keeps opening TLS "
        "connections that this HTTP port cannot answer. Open http://127.0.0.1:%d "
        "explicitly, or clear this site's data. (Further occurrences this run are "
        "logged at debug level.)" % (public_port, public_port)
    )
    _safe_notice(msg)               # one-time, fd-2-safe console line
    try:
        _log.debug("%s", msg)        # also recorded in the debug log file
    except Exception:
        pass


async def _handle_conn(reader, writer, internal_port, public_port) -> None:
    """Peek the first byte and route: TLS -> relay to uvicorn, else -> redirect.

    One try/finally around the WHOLE body, including the initial ``readexactly``
    - see :func:`_handle_conn_plain`."""
    try:
        first = await reader.readexactly(1)
        if first[0] == _TLS_FIRST_BYTE:
            await _relay(first, reader, writer, internal_port)
        else:
            await _redirect_to_https(first, reader, writer, public_port)
    except (asyncio.IncompleteReadError, ConnectionError, OSError):
        pass
    finally:
        _safe_close(writer)


async def _relay(first, c_reader, c_writer, internal_port) -> None:
    """Transparently pipe a connection to the internal uvicorn. The already
    consumed first byte is written through first so the stream is intact. Generic
    over scheme: the TLS path relays the ClientHello onward; the plain path relays
    the HTTP request line onward."""
    try:
        u_reader, u_writer = await asyncio.open_connection("127.0.0.1", internal_port)
    except OSError:
        return
    try:
        u_writer.write(first)
        await u_writer.drain()
        await asyncio.gather(
            _pump(c_reader, u_writer),
            _pump(u_reader, c_writer),
        )
    finally:
        _safe_close(u_writer)


async def _pump(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
    """Copy bytes src -> dst until EOF, then half-close dst so the peer sees it."""
    try:
        while True:
            data = await src.read(_READ_CHUNK)
            if not data:
                break
            dst.write(data)
            await dst.drain()
    except (ConnectionError, OSError, asyncio.CancelledError):
        pass
    finally:
        try:
            if dst.can_write_eof():
                dst.write_eof()
        except (OSError, RuntimeError):
            pass


async def _redirect_to_https(first, c_reader, c_writer, public_port) -> None:
    """Answer a plaintext HTTP request with a 308 to the https:// equivalent on
    the same host:port, plus a small HTML catch page."""
    try:
        rest = await asyncio.wait_for(c_reader.readuntil(b"\r\n\r\n"), timeout=5)
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError,
            asyncio.TimeoutError, ConnectionError, OSError) as exc:
        rest = getattr(exc, "partial", b"") or b""

    head = (first + rest).decode("latin-1", "replace")
    lines = head.split("\r\n")
    request_line = lines[0] if lines else ""
    parts = request_line.split(" ")
    path = parts[1] if len(parts) >= 2 else "/"
    if not _PATH_RE.match(path):
        path = "/"

    host_header = ""
    for ln in lines[1:]:
        if ln.lower().startswith("host:"):
            host_header = ln.split(":", 1)[1].strip()
            break

    location = None
    if _HOST_RE.match(host_header) or _HOST6_RE.match(host_header):
        # Append the port when the Host header omits it, so the redirect targets
        # this non-standard port rather than :443. IPv6 keeps its port after the
        # closing bracket.
        bracketed = host_header.startswith("[")
        has_port = ("]:" in host_header) if bracketed else (":" in host_header)
        authority = host_header if has_port else f"{host_header}:{public_port}"
        location = f"https://{authority}{path}"

    if location:
        safe = html.escape(location, quote=True)
        body = (
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta http-equiv=\"refresh\" content=\"0;url={safe}\">"
            "<title>localm - secure connection</title></head>"
            "<body style=\"font-family:system-ui,Segoe UI,sans-serif;"
            "background:#0f1115;color:#d7dde7;padding:2rem\">"
            "<h2 style=\"color:#4f9cf9\">localm uses a secure connection</h2>"
            f"<p>This server speaks <b>https</b>. Redirecting to "
            f"<a style=\"color:#4f9cf9\" href=\"{safe}\">{safe}</a> ...</p>"
            "<p style=\"color:#8b94a5\">If your browser does not redirect, "
            "tap the link above.</p></body></html>"
        ).encode("utf-8")
        head_bytes = (
            "HTTP/1.1 308 Permanent Redirect\r\n"
            f"Location: {location}\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("latin-1")
    else:
        # No usable Host header (e.g. HTTP/1.0 without one): cannot build a
        # redirect target, so just explain the situation.
        body = (
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<title>localm - secure connection</title></head>"
            "<body style=\"font-family:system-ui,Segoe UI,sans-serif;"
            "background:#0f1115;color:#d7dde7;padding:2rem\">"
            "<h2 style=\"color:#4f9cf9\">localm uses a secure connection</h2>"
            "<p>This server speaks <b>https</b>, not http. Reopen this address "
            "with <b>https://</b> in front.</p></body></html>"
        ).encode("utf-8")
        head_bytes = (
            "HTTP/1.1 400 Bad Request\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("latin-1")

    c_writer.write(head_bytes + body)
    await c_writer.drain()


def _safe_close(writer: Optional[asyncio.StreamWriter]) -> None:
    if writer is None:
        return
    try:
        writer.close()
    except (OSError, RuntimeError):
        pass


_stable_stream = None
_stable_resolved = False


def _get_stable_stream():
    """A duplicate of stderr taken ONCE (cached) so it is immune to the model-load
    fd-2 juggling the llama.cpp backend uses to silence native output (the same
    mechanism debuglog uses for the console mirror). MUST be resolved early, while
    fd 2 is the real console - resolving it lazily during a redirect window would
    duplicate the redirected fd and lose the output. Returns None when stderr has
    no duplicable fd (a detached process); callers fall back."""
    global _stable_stream, _stable_resolved
    if not _stable_resolved:
        _stable_resolved = True
        try:
            from localm.debuglog import _stable_console_stream
            _stable_stream = _stable_console_stream()
        except Exception:
            _stable_stream = None
    return _stable_stream


def _safe_notice(msg: str) -> None:
    """Write a one-time operational notice through the stable stderr duplicate.

    Writing a log line through the LIVE stderr during the model-load fd-2 redirect
    window raises OSError [WinError 6] on Windows, and Python's logging then prints
    a multi-line '--- Logging error ---' traceback per line. A failure to emit the
    courtesy hint must never take down the server, so the write is best-effort;
    the connection itself is already handled by the caller."""
    try:
        stream = _get_stable_stream() or sys.stderr
        stream.write("WARNING " + msg + "\n")
        stream.flush()
    except Exception:
        pass


def _harden_uvicorn_logging() -> None:
    """Point uvicorn's console log handlers at the same stable stderr duplicate.

    uvicorn configures its own loggers (uvicorn / uvicorn.error / uvicorn.access)
    with StreamHandlers on the LIVE sys.stderr. During the model-load fd-2
    redirect window those writes raise WinError 6 and cascade into
    '--- Logging error ---' tracebacks. Redirecting the handlers to a stable fd-2
    duplicate makes uvicorn's own warnings/errors survive that window; this
    changes WHERE they are written and drops none of them. Resolves the stable
    stream early, while fd 2 is still clean at server startup. Best-effort and
    idempotent."""
    stable = _get_stable_stream()
    if stable is None:
        return
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi"):
        for h in logging.getLogger(name).handlers:
            if (isinstance(h, logging.StreamHandler)
                    and not isinstance(h, logging.FileHandler)):
                try:
                    h.setStream(stable)
                except Exception:
                    pass
