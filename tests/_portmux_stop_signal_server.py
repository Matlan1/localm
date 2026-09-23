# SPDX-License-Identifier: AGPL-3.0-or-later
"""Helper subprocess for test_stop_signal_crash_guard: serve a tiny ASGI app
through localm.portmux.run_server on the main thread, with the crash guard armed
under LOCALM_HOME. NOT a test module (underscore prefix -> pytest does not
collect it).

Usage: python _portmux_stop_signal_server.py <port> <instance_id> [<signal name>]

With a signal name, the process raises that signal on itself once its crash
marker exists and the port accepts connections. Prints "run_server returned"
when run_server() returns.
"""
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

# Import localm from this worktree rather than the editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def _app(scope, receive, send):
    if scope["type"] == "lifespan":
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif msg["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
    elif scope["type"] == "http":
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"stop-signal-ok"})


class _State:
    pass


def _raise_when_serving(port: int, marker: Path, signame: str) -> None:
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if marker.exists():
            try:
                with socket.create_connection(("127.0.0.1", port), 0.2):
                    break
            except OSError:
                pass
        time.sleep(0.05)
    print(f"raising {signame} while serving (marker armed: {marker.exists()})",
          flush=True)
    signal.raise_signal(getattr(signal, signame))


if __name__ == "__main__":
    from localm import portmux
    from localm.config import HOME_DIR

    _port = int(sys.argv[1])
    _instance_id = sys.argv[2]
    _app.state = _State()
    _app.state.instance_id = _instance_id
    if len(sys.argv) > 3:
        _marker = Path(HOME_DIR) / "run" / f"server-crash.{_instance_id}.marker"
        threading.Thread(target=_raise_when_serving,
                         args=(_port, _marker, sys.argv[3]), daemon=True).start()
    print(f"pid {os.getpid()} serving", flush=True)
    portmux.run_server(_app, "127.0.0.1", _port, None, None, "warning")
    print("run_server returned", flush=True)
