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
import signal
import socket
import ssl
import sys
import threading
import time

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


# --------------------------------------------------------------------------- #
#  Stop signals: while run_server() is active, SIGHUP/SIGTERM/SIGBREAK end
#  serving gracefully instead of ending the process, so it disarms itself.
# --------------------------------------------------------------------------- #

@pytest.fixture
def _stop_state(monkeypatch):
    """Fresh portmux stop state for one test, restored afterwards. A signal
    re-delivered through the default-disposition path is recorded instead of
    raised, so a wrongly taken default path fails an assertion rather than
    ending the test process. Yields that record."""
    monkeypatch.setattr(portmux, "_active_runs", 0)
    monkeypatch.setattr(portmux, "_stop_requested", False)
    monkeypatch.setattr(portmux, "_stopping", False)
    monkeypatch.setattr(portmux, "_stop_hooks", [])
    raised = []
    monkeypatch.setattr(portmux.signal, "raise_signal", raised.append)
    yield raised


@pytest.fixture
def _sigterm_default():
    """SIGTERM at SIG_DFL for the test, whatever the runner installed."""
    original = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    yield
    signal.signal(signal.SIGTERM, original if original is not None else signal.SIG_DFL)


def test_route_stop_signals_routes_a_default_signal_and_restores_it(_sigterm_default):
    with portmux.route_stop_signals(("SIGTERM", "SIGNOTAREALSIGNAL")) as routed:
        assert routed == [signal.SIGTERM]
        assert signal.getsignal(signal.SIGTERM) is portmux._on_stop_signal
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_route_stop_signals_leaves_an_ignored_signal_ignored(_sigterm_default):
    """nohup ignores SIGHUP; such a choice must survive run_server()."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with portmux.route_stop_signals(("SIGTERM",)) as routed:
        assert routed == []
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN


def test_route_stop_signals_leaves_a_custom_handler_alone(_sigterm_default):
    def custom(signum, frame):
        pass
    signal.signal(signal.SIGTERM, custom)
    with portmux.route_stop_signals(("SIGTERM",)) as routed:
        assert routed == []
        assert signal.getsignal(signal.SIGTERM) is custom
    assert signal.getsignal(signal.SIGTERM) is custom


def test_route_stop_signals_keeps_a_handler_replaced_inside_the_block(_sigterm_default):
    def later(signum, frame):
        pass
    with portmux.route_stop_signals(("SIGTERM",)):
        signal.signal(signal.SIGTERM, later)
    assert signal.getsignal(signal.SIGTERM) is later


def test_route_stop_signals_does_nothing_off_the_main_thread(_sigterm_default):
    out = {}

    def run():
        with portmux.route_stop_signals(("SIGTERM",)) as routed:
            out["routed"] = routed
            out["handler"] = signal.getsignal(signal.SIGTERM)

    t = threading.Thread(target=run)
    t.start()
    t.join(10)
    assert out == {"routed": [], "handler": signal.SIG_DFL}


def test_stop_signal_during_run_server_ends_serving_through_the_hooks(
        _stop_state, monkeypatch):
    monkeypatch.setattr(portmux, "_active_runs", 1)
    hits = []
    portmux._stop_hooks.extend([lambda: hits.append("a"), lambda: hits.append("b")])

    portmux._on_stop_signal(signal.SIGTERM, None)   # must not raise or exit

    assert _stop_state == [], "a stop signal during run_server() was re-delivered"
    assert hits == ["a", "b"]
    assert portmux._stop_requested is True


def test_stop_signal_while_run_server_disarms_only_records_the_request(
        _stop_state, monkeypatch):
    monkeypatch.setattr(portmux, "_active_runs", 1)
    monkeypatch.setattr(portmux, "_stopping", True)
    hits = []
    portmux._stop_hooks.append(lambda: hits.append(1))

    portmux._on_stop_signal(signal.SIGTERM, None)

    assert hits == []
    assert portmux._stop_requested is True


def test_stop_signal_with_no_run_server_active_gets_its_default_effect(
        _stop_state, monkeypatch):
    set_calls, raised, hits = [], [], []
    monkeypatch.setattr(portmux.signal, "signal", lambda s, h: set_calls.append((s, h)))
    monkeypatch.setattr(portmux.signal, "raise_signal", lambda s: raised.append(s))
    portmux._stop_hooks.append(lambda: hits.append(1))

    portmux._on_stop_signal(signal.SIGTERM, None)

    assert set_calls == [(signal.SIGTERM, signal.SIG_DFL)]
    assert raised == [signal.SIGTERM]
    assert hits == []
    assert portmux._stop_requested is False


def test_a_repeated_stop_signal_after_a_stop_request_is_absorbed(
        _stop_state, monkeypatch):
    """Closing a terminal can deliver SIGHUP twice; the second one, after the
    first already stopped serving, must not kill the process mid-teardown."""
    monkeypatch.setattr(portmux, "_stop_requested", True)
    set_calls = []
    monkeypatch.setattr(portmux.signal, "signal", lambda s, h: set_calls.append((s, h)))

    portmux._on_stop_signal(signal.SIGTERM, None)

    assert _stop_state == [], "a repeated stop signal was re-delivered"
    assert set_calls == []


def _native_sleep(seconds: float) -> None:
    """Block this thread in native code that runs no Python bytecode."""
    import ctypes
    if sys.platform == "win32":
        ctypes.windll.kernel32.Sleep(int(seconds * 1000))
    else:
        ctypes.CDLL(None).sleep(int(seconds))


_ROUTABLE = "SIGBREAK" if sys.platform == "win32" else "SIGTERM"
# The real raise_signal, captured before _stop_state replaces it with a recorder.
_REAL_RAISE_SIGNAL = signal.raise_signal


def test_a_stop_signal_reaches_the_hooks_while_the_main_thread_is_in_native_code(
        _stop_state, monkeypatch):
    """App-window mode: the main thread sits in the webview's native loop and
    runs no Python for a long time, while the server runs on another thread.
    The stop must still be requested promptly."""
    sig = getattr(signal, _ROUTABLE)
    original = signal.getsignal(sig)
    signal.signal(sig, signal.SIG_DFL)
    monkeypatch.setattr(portmux, "_active_runs", 1)
    stopped_at = []
    portmux._stop_hooks.append(lambda: stopped_at.append(time.monotonic()))
    sent_at = []

    def send():
        sent_at.append(time.monotonic())
        _REAL_RAISE_SIGNAL(sig)

    try:
        with portmux.route_stop_signals((_ROUTABLE,), serving_elsewhere=True):
            sender = threading.Timer(0.2, send)
            sender.start()
            _native_sleep(3)
            sender.join(5)
    finally:
        signal.signal(sig, original if original is not None else signal.SIG_DFL)

    assert sent_at and stopped_at, "the stop signal never reached the hooks"
    assert stopped_at[0] - sent_at[0] < 1.5, (
        f"the stop waited {stopped_at[0] - sent_at[0]:.2f}s for the main thread "
        "to leave native code")


def _current_wakeup_fd() -> int:
    current = signal.set_wakeup_fd(-1)
    signal.set_wakeup_fd(current)
    return current


def test_route_stop_signals_restores_the_wakeup_fd_and_ends_its_helper(
        _stop_state, _sigterm_default):
    before_fd = _current_wakeup_fd()
    before = {t.ident for t in threading.enumerate()}
    with portmux.route_stop_signals(("SIGTERM",), serving_elsewhere=True):
        helpers = [t for t in threading.enumerate()
                   if t.name == "localm-stop-signals" and t.ident not in before]
        assert len(helpers) == 1
        assert _current_wakeup_fd() != before_fd
    assert not helpers[0].is_alive(), "the stop-signal helper thread outlived the block"
    assert _current_wakeup_fd() == before_fd, "the previous wakeup fd was not restored"


def test_route_stop_signals_starts_no_helper_when_serving_on_this_thread(
        _stop_state, _sigterm_default):
    before_fd = _current_wakeup_fd()
    with portmux.route_stop_signals(("SIGTERM",)):
        assert not [t for t in threading.enumerate() if t.name == "localm-stop-signals"]
        assert _current_wakeup_fd() == before_fd


def test_the_last_resort_bind_is_interrupted_only_from_its_own_thread(
        _stop_state, monkeypatch):
    import uvicorn as uvicorn_mod
    monkeypatch.setattr(portmux, "_active_runs", 1)

    def fail_socket(host, port):
        raise OSError("simulated: cannot build the listening socket")
    monkeypatch.setattr(portmux, "create_listen_socket", fail_socket)
    seen = {}

    def fake_run(**kw):
        other = []

        def from_elsewhere():
            try:
                portmux._request_stop()
                other.append("returned")
            except KeyboardInterrupt:
                other.append("interrupted")
        t = threading.Thread(target=from_elsewhere)
        t.start()
        t.join(5)
        seen["other thread"] = other
        try:
            portmux._request_stop()
            seen["own thread"] = "returned"
        except KeyboardInterrupt:
            seen["own thread"] = "interrupted"
    monkeypatch.setattr(uvicorn_mod, "run", fake_run)

    portmux._run_uvicorn_on_socket(uvicorn_mod, _bare_app, "127.0.0.1", 9010,
                                   log_level="warning")

    assert seen == {"other thread": ["returned"], "own thread": "interrupted"}
    assert portmux._stop_hooks == []


def test_track_server_stops_that_server_until_its_serve_task_is_done(_stop_state):
    class _Server:
        should_exit = False
    server = _Server()

    async def go():
        serve_task = asyncio.get_running_loop().create_future()
        portmux._track_server(server, serve_task)
        assert server.should_exit is False
        assert len(portmux._stop_hooks) == 1
        portmux._stop_hooks[0]()
        assert server.should_exit is True
        serve_task.set_result(None)
        await asyncio.sleep(0)
        assert portmux._stop_hooks == []

    asyncio.run(go())


def test_track_server_applies_a_stop_requested_earlier_in_the_run(
        _stop_state, monkeypatch):
    monkeypatch.setattr(portmux, "_active_runs", 1)
    monkeypatch.setattr(portmux, "_stop_requested", True)

    class _Server:
        should_exit = False
    server = _Server()

    async def go():
        serve_task = asyncio.get_running_loop().create_future()
        portmux._track_server(server, serve_task)
        serve_task.set_result(None)

    asyncio.run(go())
    assert server.should_exit is True


def test_track_server_ignores_a_request_left_over_from_an_earlier_run(
        _stop_state, monkeypatch):
    monkeypatch.setattr(portmux, "_stop_requested", True)

    class _Server:
        should_exit = False
    server = _Server()

    async def go():
        serve_task = asyncio.get_running_loop().create_future()
        portmux._track_server(server, serve_task)
        serve_task.set_result(None)

    asyncio.run(go())
    assert server.should_exit is False


def test_run_server_routes_stop_signals_only_while_it_runs(
        _stop_state, _sigterm_default, monkeypatch):
    calls = _patch_bugreport(monkeypatch)
    seen = {}

    async def fake_serve(app, host, port, log_level):
        seen["handler"] = signal.getsignal(signal.SIGTERM)
        seen["active"] = portmux._active_runs
        seen["stopping"] = portmux._stopping
    monkeypatch.setattr(portmux, "_serve_async_plain", fake_serve)

    portmux.run_server(_bare_app, "127.0.0.1", 8005)

    assert seen == {"handler": portmux._on_stop_signal, "active": 1, "stopping": False}
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    assert portmux._active_runs == 0
    assert calls[-1] == ("disarmed", None)


def test_run_server_resets_a_stop_request_left_from_an_earlier_run(
        _stop_state, _sigterm_default, monkeypatch):
    calls = _patch_bugreport(monkeypatch)
    monkeypatch.setattr(portmux, "_stop_requested", True)
    monkeypatch.setattr(portmux, "_stopping", True)
    seen = {}

    async def fake_serve(app, host, port, log_level):
        seen["requested"] = portmux._stop_requested
        seen["stopping"] = portmux._stopping
    monkeypatch.setattr(portmux, "_serve_async_plain", fake_serve)

    portmux.run_server(_bare_app, "127.0.0.1", 8008)

    assert seen == {"requested": False, "stopping": False}, (
        "a stop left over from an earlier run_server() call reached this one")
    assert calls[-1] == ("disarmed", None)


def test_run_server_counts_itself_on_top_of_a_run_already_active(
        _stop_state, _sigterm_default, monkeypatch):
    """App-window mode can have the main thread routing signals while the
    server thread's run_server() is active; each call adds itself."""
    calls = _patch_bugreport(monkeypatch)
    monkeypatch.setattr(portmux, "_active_runs", 1)
    seen = {}

    async def fake_serve(app, host, port, log_level):
        seen["active"] = portmux._active_runs
    monkeypatch.setattr(portmux, "_serve_async_plain", fake_serve)

    portmux.run_server(_bare_app, "127.0.0.1", 8009)

    assert seen == {"active": 2}
    assert portmux._active_runs == 1
    assert calls[-1] == ("disarmed", None)


@pytest.mark.parametrize("tls_files", [None, ("cert.pem", "key.pem")])
def test_run_server_hands_its_own_arguments_to_the_watchdog_and_the_serve_step(
        _stop_state, _sigterm_default, monkeypatch, tls_files):
    _patch_bugreport(monkeypatch)
    spawned = []
    monkeypatch.setattr(portmux, "_spawn_crash_recovery_watchdog",
                        lambda **kw: spawned.append(kw))
    served = []

    async def fake_plain(app, host, port, log_level):
        served.append((app, host, port, log_level))

    async def fake_tls(app, host, port, ssl_certfile, ssl_keyfile, log_level):
        served.append((app, host, port, ssl_certfile, ssl_keyfile, log_level))
    monkeypatch.setattr(portmux, "_serve_async_plain", fake_plain)
    monkeypatch.setattr(portmux, "_serve_async", fake_tls)

    class State:
        instance_id = "inst-args"

    class App:
        state = State()

        async def __call__(self, scope, receive, send):
            pass

    app = App()
    cert, key = tls_files or (None, None)
    portmux.run_server(app, "0.0.0.0", 9011, ssl_certfile=cert, ssl_keyfile=key)

    assert spawned == [{"host": "0.0.0.0", "port": 9011, "tls": bool(cert),
                        "instance_id": "inst-args"}]
    if cert:
        assert served == [(app, "0.0.0.0", 9011, cert, key, "warning")]
    else:
        assert served == [(app, "0.0.0.0", 9011, "warning")]


def test_run_server_does_not_serve_after_a_stop_during_startup(
        _stop_state, _sigterm_default, monkeypatch):
    calls = _patch_bugreport(monkeypatch)

    def arm_then_stop(context=None, home=None, instance_id=None):
        calls.append(("armed", context, instance_id))
        portmux._on_stop_signal(signal.SIGTERM, None)
    monkeypatch.setattr(bugreport_mod, "arm_crash_guard", arm_then_stop)
    served = []

    async def fake_serve(app, host, port, log_level):
        served.append(1)
    monkeypatch.setattr(portmux, "_serve_async_plain", fake_serve)

    portmux.run_server(_bare_app, "127.0.0.1", 8006)

    assert served == [], "run_server served after a stop signal had already arrived"
    assert calls[-1] == ("disarmed", None)


def test_run_server_is_stopping_before_it_disarms_and_a_signal_then_is_absorbed(
        _stop_state, _sigterm_default, monkeypatch):
    """A second terminal-close SIGHUP can land inside the disarm itself; it must
    neither raise into it nor run a stop hook."""
    calls = _patch_bugreport(monkeypatch)
    seen = {}
    hits = []

    def disarm(home=None, instance_id=None):
        seen["stopping"] = portmux._stopping
        portmux._stop_hooks.append(lambda: hits.append(1))
        portmux._on_stop_signal(signal.SIGTERM, None)
        calls.append(("disarmed", instance_id))
    monkeypatch.setattr(bugreport_mod, "disarm_crash_guard", disarm)

    async def fake_serve(app, host, port, log_level):
        return
    monkeypatch.setattr(portmux, "_serve_async_plain", fake_serve)

    portmux.run_server(_bare_app, "127.0.0.1", 8007)

    assert seen == {"stopping": True}
    assert hits == []
    assert calls[-1] == ("disarmed", None)


def _stop_when_listening(port, outcome):
    """Wait until *port* accepts, then deliver a stop the way the handler would.
    If run_server() is still serving 15s later, interrupt the main thread so a
    missing stop hook fails the test instead of hanging it."""
    import _thread
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), 0.2):
                break
        except OSError:
            time.sleep(0.05)
    portmux._on_stop_signal(signal.SIGTERM, None)
    outcome["stop_sent"] = time.monotonic()
    while time.monotonic() < outcome["stop_sent"] + 15.0:
        if outcome.get("returned"):
            return
        time.sleep(0.05)
    outcome["interrupted"] = True
    _thread.interrupt_main()


@pytest.mark.parametrize("use_tls", [False, True])
def test_a_stop_signal_ends_a_real_serve_and_run_server_disarms(
        _stop_state, _sigterm_default, monkeypatch, tmp_path, use_tls):
    calls = _patch_bugreport(monkeypatch)
    port = _free_port()
    cert = key = None
    if use_tls:
        cert, key = tls.ensure_cert(tmp_path, hostnames=["127.0.0.1"])
    outcome = {}
    stopper = threading.Thread(target=_stop_when_listening, args=(port, outcome),
                                daemon=True)
    stopper.start()

    portmux.run_server(_tiny_asgi_app, "127.0.0.1", port,
                       ssl_certfile=cert, ssl_keyfile=key)
    outcome["returned"] = True
    stopper.join(20)

    assert not outcome.get("interrupted"), (
        "the stop signal did not end serving - run_server had to be interrupted")
    assert "stop_sent" in outcome
    assert calls[-1] == ("disarmed", None)
    assert portmux._stop_hooks == []


class _GatedLifespanApp:
    """An ASGI app whose lifespan startup waits for *release*, counting how many
    times each lifespan phase ran."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.counts = {"startup": 0, "shutdown": 0}

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    self.counts["startup"] += 1
                    self.started.set()
                    while not self.release.is_set():
                        await asyncio.sleep(0.01)
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    self.counts["shutdown"] += 1
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        else:
            await _tiny_asgi_app(scope, receive, send)


@pytest.mark.parametrize("use_tls", [False, True])
def test_a_stop_during_the_internal_servers_startup_ends_serving_cleanly(
        _stop_state, _sigterm_default, monkeypatch, tmp_path, use_tls):
    """uvicorn closes its sockets straight after a startup that was told to
    stop; run_server() must end there, without an error, a fallback bind or a
    second lifespan startup."""
    calls = _patch_bugreport(monkeypatch)
    fallback = []
    monkeypatch.setattr(portmux, "_run_uvicorn_on_socket",
                        lambda *a, **k: fallback.append(k))
    app = _GatedLifespanApp()
    cert = key = None
    if use_tls:
        cert, key = tls.ensure_cert(tmp_path, hostnames=["127.0.0.1"])

    def stop_mid_startup():
        if app.started.wait(15):
            portmux._on_stop_signal(signal.SIGTERM, None)
        app.release.set()

    threading.Thread(target=stop_mid_startup, daemon=True).start()
    portmux.run_server(app, "127.0.0.1", _free_port(),
                       ssl_certfile=cert, ssl_keyfile=key)

    assert fallback == [], "a stop during startup was treated as a failed bind"
    assert app.counts["startup"] == 1, (
        f"the app's lifespan startup ran {app.counts['startup']} times")
    assert calls[-1] == ("disarmed", None)
    assert portmux._stop_hooks == []
