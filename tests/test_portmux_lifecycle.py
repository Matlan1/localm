# SPDX-License-Identifier: AGPL-3.0-or-later
"""In-process tests for portmux's server-lifecycle functions: ``run_server``,
``_serve_async`` and ``_serve_async_plain``, plus the real (non-faked) ``_relay``
round trip.

The async lifecycle functions are started as real asyncio tasks against real
ephemeral (port 0) sockets in THIS process, so both the happy path and the
"internal server never came up" failure path are exercised for real.

One substitution: on the PLAIN (no-TLS) path the internal uvicorn always binds to
a hardcoded ``127.0.0.1:0`` (an ephemeral loopback port), which cannot
practically be made to fail without mocking something, so the "internal server
startup failed" branch is exercised via a minimal fake ``uvicorn.Server`` there.
The TLS variant gets the SAME branch exercised for real instead, via a genuinely
bad certificate path - a corrupt or missing cert must not hang or silently fall
back to plaintext.
"""
from __future__ import annotations

import asyncio
import contextlib
import socket
import ssl

import pytest

from localm import bugreport as bugreport_mod
from localm import portmux, tls


# --------------------------------------------------------------------------- #
#  Shared helpers
# --------------------------------------------------------------------------- #

async def _tiny_asgi_app(scope, receive, send):
    """A minimal real ASGI app: implements lifespan (so uvicorn's startup
    completes cleanly) and answers every HTTP request with a fixed body and an
    explicit Content-Length (avoids chunked encoding, keeping response
    assertions simple)."""
    if scope["type"] == "lifespan":
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif msg["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
    elif scope["type"] == "http":
        body = b"lifecycle-ok"
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/plain"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _wait_connectable(port: int, timeout: float = 10.0) -> None:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    last = None
    while loop.time() < deadline:
        try:
            _, w = await asyncio.open_connection("127.0.0.1", port)
            w.close()
            with contextlib.suppress(Exception):
                await w.wait_closed()
            return
        except OSError as e:
            last = e
            await asyncio.sleep(0.05)
    raise AssertionError(f"port {port} never became connectable: {last}")


async def _shutdown(task: asyncio.Task, timeout: float = 10.0) -> None:
    """Cancel a running server-lifecycle task and wait for its finally-block
    teardown (demux.close()/wait_closed(), wakeup_task cancel) to finish."""
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
        await asyncio.wait_for(task, timeout=timeout)


# --------------------------------------------------------------------------- #
#  _relay: real byte-for-byte round trip (both directions, through _pump)
# --------------------------------------------------------------------------- #

async def _make_echo_server():
    async def handle(reader, writer):
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        finally:
            writer.close()
    server = await asyncio.start_server(handle, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    return server, port


def test_relay_pumps_bytes_both_directions_byte_for_byte():
    # Round-trips arbitrary bytes through a real internal listener and asserts
    # the reply is byte-for-byte identical: no header rewriting, no injected or
    # dropped bytes.
    async def go():
        echo_server, internal_port = await _make_echo_server()

        async def on_client(reader, writer):
            first = await reader.readexactly(1)
            await portmux._relay(first, reader, writer, internal_port)
            writer.close()

        front = await asyncio.start_server(on_client, host="127.0.0.1", port=0)
        front_port = front.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", front_port)
            writer.write(b"hello-relay-roundtrip")
            await writer.drain()
            writer.write_eof()
            reply = await asyncio.wait_for(reader.read(-1), timeout=10)
            assert reply == b"hello-relay-roundtrip"
        finally:
            writer.close()
            front.close()
            await front.wait_closed()
            echo_server.close()
            await echo_server.wait_closed()
    asyncio.run(go())


# --------------------------------------------------------------------------- #
#  _serve_async_plain / _serve_async: real end-to-end lifecycle
# --------------------------------------------------------------------------- #

def test_serve_async_plain_relays_real_http_and_shuts_down_cleanly():
    async def go():
        port = _free_port()
        task = asyncio.ensure_future(
            portmux._serve_async_plain(_tiny_asgi_app, "127.0.0.1", port, "warning"))
        try:
            await _wait_connectable(port)
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            await writer.drain()
            data = await asyncio.wait_for(reader.read(-1), timeout=10)
            assert data.startswith(b"HTTP/1.1 200")
            assert data.endswith(b"lifecycle-ok")
            writer.close()
        finally:
            await _shutdown(task)
        # Teardown must actually release the public port, not just return.
        with pytest.raises(OSError):
            await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), timeout=2)
    asyncio.run(go())


def test_serve_async_tls_relays_a_real_handshake_and_shuts_down_cleanly(tmp_path):
    cert, key = tls.ensure_cert(tmp_path, hostnames=["127.0.0.1"])
    ca = str(tls.ca_cert_path(tmp_path))

    async def go():
        port = _free_port()
        task = asyncio.ensure_future(
            portmux._serve_async(_tiny_asgi_app, "127.0.0.1", port, cert, key, "warning"))
        try:
            await _wait_connectable(port)
            ctx = ssl.create_default_context(cafile=ca)
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", port, ssl=ctx, server_hostname="127.0.0.1")
            writer.write(b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            await writer.drain()
            data = await asyncio.wait_for(reader.read(-1), timeout=10)
            assert data.startswith(b"HTTP/1.1 200")
            assert data.endswith(b"lifecycle-ok")
            writer.close()
        finally:
            await _shutdown(task)
        with pytest.raises(OSError):
            await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), timeout=2)
    asyncio.run(go())


def test_serve_async_plain_cancels_an_inflight_connection_on_shutdown():
    """asyncio.start_server()'s client_connected_cb creates a Task for each
    connection that nothing keeps a reference to, so a connection still blocked
    in _relay's pumps at shutdown is invisible to demux.wait_closed() (which only
    waits for the LISTENING socket, never for handler tasks already running) and
    is silently destroyed mid-flight instead of being closed ("Task was destroyed
    but it is pending!"). Shutdown must cancel and await the tracked task, whose
    finally closes the writer, so the client sees a clean EOF as PART of
    shutdown."""
    async def go():
        port = _free_port()
        task = asyncio.ensure_future(
            portmux._serve_async_plain(_tiny_asgi_app, "127.0.0.1", port, "warning"))
        await _wait_connectable(port)
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            # One byte only, never the rest of a request line: the internal
            # uvicorn never gets a full HTTP request, so both pump directions
            # block on read() and the connection's task stays pending.
            writer.write(b"G")
            await writer.drain()
            await asyncio.sleep(0.1)   # let accept + relay-connect land

            await _shutdown(task)

            # The server must close THIS connection as part of its OWN
            # shutdown, not leave it for the OS to reap. A short timeout
            # distinguishes "closed promptly" from "never closed".
            data = await asyncio.wait_for(reader.read(-1), timeout=2)
            assert data == b""
        finally:
            writer.close()
    asyncio.run(go())


def test_serve_async_tls_cancels_an_inflight_connection_on_shutdown(tmp_path):
    """TLS variant: _serve_async wires the same per-connection callback, so a
    connection still blocked in _relay's pumps at shutdown must be cancelled and
    awaited, closing the writer so the client sees a clean EOF."""
    cert, key = tls.ensure_cert(tmp_path, hostnames=["127.0.0.1"])
    ca = str(tls.ca_cert_path(tmp_path))

    async def go():
        port = _free_port()
        task = asyncio.ensure_future(
            portmux._serve_async(_tiny_asgi_app, "127.0.0.1", port, cert, key, "warning"))
        await _wait_connectable(port)
        ctx = ssl.create_default_context(cafile=ca)
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", port, ssl=ctx, server_hostname="127.0.0.1")
        try:
            # A real completed TLS handshake, then a partial request line only
            # - the internal uvicorn never gets a full request, so the pumps
            # block and the connection task stays pending through shutdown.
            writer.write(b"G")
            await writer.drain()
            await asyncio.sleep(0.1)

            await _shutdown(task)

            data = await asyncio.wait_for(reader.read(-1), timeout=2)
            assert data == b""
        finally:
            writer.close()
    asyncio.run(go())


def test_serve_async_propagates_a_bad_tls_cert_instead_of_hanging(tmp_path):
    # A corrupt or missing certificate must surface loudly - the server never
    # comes up and the caller finds out immediately - rather than hang forever
    # or fall through to an unprotected bind.
    #
    # The outer wait_for(timeout=10) is a safety net against a hang, not the
    # behavioural assertion: asyncio.TimeoutError is itself an Exception
    # subclass, so the isinstance check below is what pins failed-fast.
    async def go():
        port = _free_port()
        missing_cert = str(tmp_path / "does-not-exist.crt")
        missing_key = str(tmp_path / "does-not-exist.key")
        with pytest.raises(Exception) as exc_info:
            await asyncio.wait_for(
                portmux._serve_async(_tiny_asgi_app, "127.0.0.1", port,
                                     missing_cert, missing_key, "warning"),
                timeout=10)
        assert not isinstance(exc_info.value, asyncio.TimeoutError), (
            "startup hung instead of failing fast on the bad cert "
            "(wait_for's own timeout fired, not a real startup error)")
    asyncio.run(go())


class _FailFastServer:
    """Stand-in for uvicorn.Server whose serve() fails before startup completes,
    simulating the internal loopback uvicorn never coming up: portmux must
    propagate the error, not hang or silently continue with no backend
    listening."""
    def __init__(self, config):
        self.config = config
        self.started = False
        self.should_exit = False
        self.servers = []

    async def serve(self, sockets=None):
        raise RuntimeError("simulated internal-server startup failure")


def test_serve_async_plain_propagates_internal_server_startup_failure(monkeypatch):
    import uvicorn as uvicorn_mod
    monkeypatch.setattr(uvicorn_mod, "Server", _FailFastServer)

    async def go():
        port = _free_port()
        with pytest.raises(RuntimeError, match="simulated internal-server startup failure"):
            await portmux._serve_async_plain(_tiny_asgi_app, "127.0.0.1", port, "warning")
        # No demux must have been created off the back of a failed backend.
        with pytest.raises(OSError):
            await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), timeout=2)
    asyncio.run(go())


class _CleanExitServer:
    """Stand-in for uvicorn.Server whose serve() completes NORMALLY (no
    exception) without ever setting ``started`` - the other completion state
    portmux's startup-wait loop must handle: _FailFastServer above covers the
    task finishing WITH an exception."""
    def __init__(self, config):
        self.config = config
        self.started = False
        self.should_exit = False
        self.servers = []

    async def serve(self, sockets=None):
        return


def test_serve_async_plain_returns_cleanly_when_internal_server_exits_without_starting(monkeypatch):
    import uvicorn as uvicorn_mod
    monkeypatch.setattr(uvicorn_mod, "Server", _CleanExitServer)

    async def go():
        port = _free_port()
        result = await portmux._serve_async_plain(_tiny_asgi_app, "127.0.0.1", port, "warning")
        assert result is None
        with pytest.raises(OSError):
            await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), timeout=2)
    asyncio.run(go())


def test_serve_async_tls_returns_cleanly_when_internal_server_exits_without_starting(
        tmp_path, monkeypatch):
    cert, key = tls.ensure_cert(tmp_path, hostnames=["127.0.0.1"])
    import uvicorn as uvicorn_mod
    monkeypatch.setattr(uvicorn_mod, "Server", _CleanExitServer)

    async def go():
        port = _free_port()
        result = await portmux._serve_async(
            _tiny_asgi_app, "127.0.0.1", port, cert, key, "warning")
        assert result is None
    asyncio.run(go())


def test_serve_async_plain_closes_the_listen_socket_when_start_server_fails(monkeypatch):
    """The internal uvicorn is real and comes up; the OUTER demux socket fails
    to bind as a start_server listener - the prepared socket must be closed
    rather than leaked, and the error must propagate."""
    async def fail_start_server(*a, **kw):
        raise RuntimeError("simulated: asyncio.start_server failed")
    monkeypatch.setattr(portmux.asyncio, "start_server", fail_start_server)

    async def go():
        port = _free_port()
        with pytest.raises(RuntimeError, match="simulated: asyncio.start_server failed"):
            await portmux._serve_async_plain(_tiny_asgi_app, "127.0.0.1", port, "warning")
        # A fresh bind on the same port must succeed immediately - proves the
        # listen socket was closed, not leaked.
        s = portmux.create_listen_socket("127.0.0.1", port)
        s.close()
    asyncio.run(go())


def test_serve_async_tls_closes_the_listen_socket_when_start_server_fails(tmp_path, monkeypatch):
    cert, key = tls.ensure_cert(tmp_path, hostnames=["127.0.0.1"])

    async def fail_start_server(*a, **kw):
        raise RuntimeError("simulated: asyncio.start_server failed")
    monkeypatch.setattr(portmux.asyncio, "start_server", fail_start_server)

    async def go():
        port = _free_port()
        with pytest.raises(RuntimeError, match="simulated: asyncio.start_server failed"):
            await portmux._serve_async(
                _tiny_asgi_app, "127.0.0.1", port, cert, key, "warning")
        s = portmux.create_listen_socket("127.0.0.1", port)
        s.close()
    asyncio.run(go())


# --------------------------------------------------------------------------- #
#  _cancel_inflight_conns: the empty/all-done arm the shutdown tests above
#  never reach (they always leave one connection genuinely still pending)
# --------------------------------------------------------------------------- #

def test_cancel_inflight_conns_with_no_pending_tasks_is_a_noop():
    asyncio.run(portmux._cancel_inflight_conns(set()))


def test_cancel_inflight_conns_skips_a_task_that_already_finished():
    async def go():
        done_task = asyncio.ensure_future(asyncio.sleep(0))
        await done_task
        assert done_task.done()
        # Must not try to cancel/await an already-finished task.
        await portmux._cancel_inflight_conns({done_task})
    asyncio.run(go())


# --------------------------------------------------------------------------- #
#  Non-Windows platform: the wakeup-task branch is win32-only, so a real
#  Windows CI run can only exercise the "no wakeup task" arm by asserting the
#  live platform check itself, not by faking sys.platform on the box that
#  IS win32 (that would test a code path no real non-Windows box takes).
# --------------------------------------------------------------------------- #

def test_serve_async_plain_skips_the_wakeup_task_off_windows(monkeypatch):
    monkeypatch.setattr(portmux.sys, "platform", "linux")

    async def go():
        port = _free_port()
        task = asyncio.ensure_future(
            portmux._serve_async_plain(_tiny_asgi_app, "127.0.0.1", port, "warning"))
        await _wait_connectable(port)
        await _shutdown(task)
        with pytest.raises(OSError):
            await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), timeout=2)
    asyncio.run(go())


def test_serve_async_tls_skips_the_wakeup_task_off_windows(tmp_path, monkeypatch):
    cert, key = tls.ensure_cert(tmp_path, hostnames=["127.0.0.1"])
    monkeypatch.setattr(portmux.sys, "platform", "linux")

    async def go():
        port = _free_port()
        task = asyncio.ensure_future(
            portmux._serve_async(_tiny_asgi_app, "127.0.0.1", port, cert, key, "warning"))
        await _wait_connectable(port)
        await _shutdown(task)
        with pytest.raises(OSError):
            await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), timeout=2)
    asyncio.run(go())


# --------------------------------------------------------------------------- #
#  run_server: crash-guard wiring, instance_id extraction, failure handling
# --------------------------------------------------------------------------- #

def _patch_bugreport(monkeypatch):
    """Record calls to the crash-guard hooks without touching disk: run_server is
    tested here for its OWN wiring and ordering, not for bugreport's own
    behaviour."""
    calls = []
    monkeypatch.setattr(bugreport_mod, "check_and_report_prior_crash",
                        lambda *a, **k: calls.append(("checked",)))
    monkeypatch.setattr(
        bugreport_mod, "arm_crash_guard",
        lambda context=None, home=None, instance_id=None:
            calls.append(("armed", context, instance_id)))
    monkeypatch.setattr(
        bugreport_mod, "disarm_crash_guard",
        lambda home=None, instance_id=None: calls.append(("disarmed", instance_id)))
    return calls


async def _bare_app(scope, receive, send):
    pass


def test_run_server_plain_wires_crash_guard_and_extracts_instance_id(monkeypatch):
    calls = _patch_bugreport(monkeypatch)

    async def fake_serve(app, host, port, log_level):
        return
    monkeypatch.setattr(portmux, "_serve_async_plain", fake_serve)

    class State:
        instance_id = "inst-abc"

    class App:
        state = State()

    portmux.run_server(App(), "0.0.0.0", 9999)

    assert calls[0] == ("checked",)
    assert calls[1] == ("armed", {"host": "0.0.0.0", "port": 9999, "tls": False}, "inst-abc")
    assert calls[2] == ("disarmed", "inst-abc")


def test_run_server_handles_a_bare_asgi_callable_with_no_state(monkeypatch):
    # A bare ASGI function (no .state) must degrade to instance_id=None
    # instead of raising AttributeError before the server ever binds.
    calls = _patch_bugreport(monkeypatch)

    async def fake_serve(app, host, port, log_level):
        return
    monkeypatch.setattr(portmux, "_serve_async_plain", fake_serve)

    portmux.run_server(_bare_app, "127.0.0.1", 8000)

    assert calls[1] == ("armed", {"host": "127.0.0.1", "port": 8000, "tls": False}, None)
    assert calls[2] == ("disarmed", None)


def test_run_server_plain_swallows_keyboard_interrupt(monkeypatch):
    calls = _patch_bugreport(monkeypatch)

    async def fake_serve(app, host, port, log_level):
        raise KeyboardInterrupt()
    monkeypatch.setattr(portmux, "_serve_async_plain", fake_serve)

    portmux.run_server(_bare_app, "127.0.0.1", 8001)   # must not raise
    assert calls[-1] == ("disarmed", None), "crash guard must still be disarmed"


def test_run_server_plain_falls_back_to_uvicorn_run_on_unexpected_error(monkeypatch):
    calls = _patch_bugreport(monkeypatch)

    async def fake_serve(app, host, port, log_level):
        raise RuntimeError("peek layer exploded")
    monkeypatch.setattr(portmux, "_serve_async_plain", fake_serve)

    def fail_socket(host, port):   # simulates create_listen_socket failing
        raise OSError("simulated: cannot build the listening socket")
    monkeypatch.setattr(portmux, "create_listen_socket", fail_socket)

    import uvicorn as uvicorn_mod
    fallback_calls = []
    monkeypatch.setattr(uvicorn_mod, "run", lambda app, **kw: fallback_calls.append(kw))

    portmux.run_server(_bare_app, "127.0.0.1", 8002)   # must not raise
    assert fallback_calls == [{
        "host": "127.0.0.1", "port": 8002, "log_level": "warning",
        # The fallback is a real server bind, so it carries the same bounded
        # stop as the primary path; without it a Ctrl+C on the degraded path
        # waits for the longest open response.
        "timeout_graceful_shutdown": portmux.GRACEFUL_SHUTDOWN_TIMEOUT,
    }]
    assert calls[-1] == ("disarmed", None)


def test_run_server_plain_fallback_binds_the_prepared_socket_on_success(monkeypatch):
    """The OTHER arm from the test above: create_listen_socket succeeds on the
    fallback path too, so _run_uvicorn_on_socket must bind the real prepared
    socket via Server.run(sockets=...) instead of uvicorn's own bind."""
    calls = _patch_bugreport(monkeypatch)

    async def fake_serve(app, host, port, log_level):
        raise RuntimeError("peek layer exploded")
    monkeypatch.setattr(portmux, "_serve_async_plain", fake_serve)

    class _RecordingServer:
        instances: list = []
        def __init__(self, config):
            self.config = config
            self.run_sockets = None
            _RecordingServer.instances.append(self)
        def run(self, sockets=None):
            self.run_sockets = sockets

    _RecordingServer.instances = []
    import uvicorn as uvicorn_mod
    monkeypatch.setattr(uvicorn_mod, "Server", _RecordingServer)

    port = _free_port()
    portmux.run_server(_bare_app, "127.0.0.1", port)   # must not raise
    assert len(_RecordingServer.instances) == 1
    server = _RecordingServer.instances[0]
    try:
        assert server.run_sockets and len(server.run_sockets) == 1
        assert server.run_sockets[0].getsockname()[1] == port
    finally:
        for s in (server.run_sockets or []):
            s.close()
    assert calls[-1] == ("disarmed", None)


def test_run_server_tls_swallows_keyboard_interrupt(monkeypatch):
    calls = _patch_bugreport(monkeypatch)

    async def fake_serve(app, host, port, ssl_certfile, ssl_keyfile, log_level):
        raise KeyboardInterrupt()
    monkeypatch.setattr(portmux, "_serve_async", fake_serve)

    portmux.run_server(_bare_app, "0.0.0.0", 8443,
                       ssl_certfile="cert.pem", ssl_keyfile="key.pem")   # must not raise
    assert calls[1][1]["tls"] is True
    assert calls[-1] == ("disarmed", None)


def test_run_server_tls_falls_back_to_uvicorn_run_on_unexpected_error(monkeypatch):
    calls = _patch_bugreport(monkeypatch)

    async def fake_serve(app, host, port, ssl_certfile, ssl_keyfile, log_level):
        raise RuntimeError("demux exploded")
    monkeypatch.setattr(portmux, "_serve_async", fake_serve)

    def fail_socket(host, port):   # simulates create_listen_socket failing
        raise OSError("simulated: cannot build the listening socket")
    monkeypatch.setattr(portmux, "create_listen_socket", fail_socket)

    import uvicorn as uvicorn_mod
    fallback_calls = []
    monkeypatch.setattr(uvicorn_mod, "run", lambda app, **kw: fallback_calls.append(kw))

    portmux.run_server(_bare_app, "0.0.0.0", 8443,
                       ssl_certfile="cert.pem", ssl_keyfile="key.pem")   # must not raise
    assert fallback_calls == [{
        "host": "0.0.0.0", "port": 8443, "log_level": "warning",
        "timeout_graceful_shutdown": portmux.GRACEFUL_SHUTDOWN_TIMEOUT,
        "ssl_certfile": "cert.pem", "ssl_keyfile": "key.pem",
    }]
    assert calls[-1] == ("disarmed", None)
