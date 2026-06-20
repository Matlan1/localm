# SPDX-License-Identifier: AGPL-3.0-or-later
"""Same-port HTTP -> HTTPS handling for network binds (NET-1 follow-up).

When localm binds past loopback it serves HTTPS (built-in TLS, see ``tls.py``).
A browser that opens the plain ``http://`` URL on that port then speaks cleartext
to a TLS socket, the handshake fails, and the user just sees a bare "connection
reset" - not seamless. This module makes that case graceful WITHOUT a reverse
proxy: it owns the public listening socket, peeks the first byte of every
connection, and either

  * TLS (the handshake starts with byte ``0x16``) -> transparently relays the
    raw bytes to an internal loopback uvicorn that terminates TLS and serves the
    app (TLS still terminates at uvicorn; we only shuffle bytes), or
  * plaintext (an HTTP request) -> answers with a ``308`` redirect to the
    ``https://`` URL plus a small HTML catch page, so opening ``http://`` either
    redirects automatically or at least explains itself.

This is used ONLY on a TLS bind (always a network bind in practice; loopback is
plain HTTP and never reaches here). Any setup failure falls back to a direct
``uvicorn.run`` TLS bind (today's behaviour), so the working HTTPS path is never
put at risk by this convenience.

Nothing in the server trusts ``request.client.host`` for a security decision
(the loopback/network split is made from the configured bind host, not the peer
address - see ``gui/web.py``), so the relayed connection appearing to uvicorn as
``127.0.0.1`` is safe.
"""

from __future__ import annotations

import asyncio
import html
import re
from typing import Optional

# A TLS record (and therefore the ClientHello that opens any HTTPS connection)
# begins with the handshake content-type byte 0x16. A plaintext HTTP request
# begins with an ASCII method letter ("G", "P", ...). One byte distinguishes them.
_TLS_FIRST_BYTE = 0x16

# Accept only a sane Host header before reflecting it into a redirect, so a
# crafted Host cannot turn the redirect into an open redirect / header smuggle.
# host[:port] for a DNS name or IPv4, or [v6]:port for IPv6.
_HOST_RE = re.compile(r"^[A-Za-z0-9.\-]+(:\d+)?$")
_HOST6_RE = re.compile(r"^\[[0-9A-Fa-f:]+\](:\d+)?$")
# A request target we are willing to preserve across the redirect: an absolute
# path of printable ASCII with no spaces or control characters.
_PATH_RE = re.compile(r"^/[!-~]*$")

_READ_CHUNK = 65536


def run_server(
    app,
    host: str,
    port: int,
    ssl_certfile: Optional[str] = None,
    ssl_keyfile: Optional[str] = None,
    log_level: str = "warning",
) -> None:
    """Serve *app* on ``(host, port)``, blocking until interrupted.

    Without TLS this is a plain ``uvicorn.run`` (the loopback default - unchanged).
    With TLS it serves HTTPS and also catches a plain-HTTP request on the same
    port with an https redirect (issue 8), falling back to a direct TLS bind if
    the demultiplexer cannot start.
    """
    import uvicorn

    if not ssl_certfile:
        uvicorn.run(app, host=host, port=port, log_level=log_level)
        return

    try:
        asyncio.run(_serve_async(app, host, port, ssl_certfile, ssl_keyfile, log_level))
    except KeyboardInterrupt:
        pass
    except Exception:   # pragma: no cover - defensive fallback
        # Never leave the user without a server because the convenience layer
        # failed: fall back to a direct TLS bind. The http-typo case then just
        # is not caught, which is no worse than before this module existed.
        import traceback
        traceback.print_exc()
        uvicorn.run(app, host=host, port=port, log_level=log_level,
                    ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)


async def _serve_async(app, host, port, ssl_certfile, ssl_keyfile, log_level) -> None:
    import uvicorn

    # Internal loopback uvicorn that terminates TLS and serves the real app on an
    # ephemeral port (0 -> the OS picks a free one).
    config = uvicorn.Config(
        app, host="127.0.0.1", port=0, log_level=log_level,
        ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile,
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

    async def _on_conn(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        await _handle_conn(reader, writer, internal_port, port)

    demux = await asyncio.start_server(_on_conn, host=host, port=port)
    try:
        await serve_task          # runs until Ctrl+C / should_exit
    finally:
        demux.close()
        try:
            await demux.wait_closed()
        except Exception:         # pragma: no cover
            pass
        server.should_exit = True


async def _handle_conn(reader, writer, internal_port, public_port) -> None:
    """Peek the first byte and route: TLS -> relay to uvicorn, else -> redirect."""
    try:
        first = await reader.readexactly(1)
    except (asyncio.IncompleteReadError, ConnectionError, OSError):
        _safe_close(writer)
        return

    try:
        if first[0] == _TLS_FIRST_BYTE:
            await _relay_tls(first, reader, writer, internal_port)
        else:
            await _redirect_to_https(first, reader, writer, public_port)
    except (ConnectionError, OSError, asyncio.IncompleteReadError):
        pass
    finally:
        _safe_close(writer)


async def _relay_tls(first, c_reader, c_writer, internal_port) -> None:
    """Transparently pipe a TLS connection to the internal uvicorn. The already
    consumed first byte is written through first so the handshake is intact."""
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
        # The client reached us ON public_port; if the Host header omits the port
        # (a non-compliant client), append it so the redirect targets the right
        # port - these binds use a non-standard port, so a bare host would resolve
        # to :443 and fail. IPv6 keeps its port after the closing bracket.
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
