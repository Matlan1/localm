# SPDX-License-Identifier: AGPL-3.0-or-later
"""Helper subprocess for test_portmux: run a localm.portmux server with a tiny
ASGI app. NOT a test module (underscore prefix -> pytest does not collect it).

Usage: python _portmux_server.py <port> [<certfile> <keyfile>]
       (omit cert/key, or pass "-" "-", to run a PLAIN-HTTP bind.)
"""
import sys
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
        await send({"type": "http.response.body", "body": b"hello-localm"})


if __name__ == "__main__":
    from localm import portmux
    _port = int(sys.argv[1])
    _cert = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    _key = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    portmux.run_server(_app, "127.0.0.1", _port, _cert, _key, "warning")
