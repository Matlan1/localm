# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real loopback HTTP servers standing in for other running localm instances.

``peer_server`` answers ``GET /whoami`` and ``GET /v1/instances/status`` (and,
when given handlers, ``POST /v1/instances/vouch`` and ``POST
/v1/instances/cooperate-unload``) on a throwaway loopback port, so the live
detection in ``localm.gpu_registry`` runs against a genuine socket and a genuine
HTTP exchange rather than patched ``requests``.

``enable_detection`` turns peer detection on for one test and narrows the port
range it scans to the test's own ports, so it never talks to a real localm
instance of the user running the suite.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

FOREIGN_PID = 1


class _Handler(BaseHTTPRequestHandler):
    server_version = "peer-test"

    def _send(self, status: int, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        spec = self.server.spec
        spec["requests"].append(("GET", self.path, None))
        if self.path == "/whoami":
            self._send(200, spec["whoami"])
        elif self.path == "/v1/instances/status" and spec["status"] is not None:
            self._send(200, spec["status"])
        else:
            self._send(404, {"detail": "Not Found"})

    def do_POST(self):
        spec = self.server.spec
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            body = {}
        spec["requests"].append(("POST", self.path, body))
        handler = spec["post"].get(self.path)
        if handler is None:
            self._send(404, {"detail": "Not Found"})
            return
        status, payload = handler(body)
        self._send(status, payload)

    def log_message(self, *args):
        return


def default_status(whoami: dict, **overrides) -> dict:
    """A coordination status matching *whoami*'s instance id."""
    status = {"instance_id": whoami.get("instance_id"), "pid": FOREIGN_PID,
              "model": None, "models": [], "vram_estimate_bytes": None,
              "gpu_index": 0}
    status.update(overrides)
    return status


@contextmanager
def peer_server(whoami: dict, *, status: Optional[dict] = "derive",
                post: Optional[dict] = None):
    """A loopback server for one fake instance. *status* is the body of ``GET
    /v1/instances/status`` (``"derive"`` builds one from *whoami*, None answers
    404). *post* maps a path to ``handler(body) -> (status_code, payload)``.
    Yields an object with ``.port`` and ``.requests`` (every request seen as
    ``(method, path, body)``)."""
    spec = {
        "whoami": whoami,
        "status": default_status(whoami) if status == "derive" else status,
        "post": post or {},
        "requests": [],
    }
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.spec = spec
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()

    class _Running:
        port = srv.server_address[1]
        requests = spec["requests"]

    try:
        yield _Running
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def enable_detection(monkeypatch, *ports: int) -> None:
    """Turn peer detection on for this test, scanning only *ports*."""
    monkeypatch.setenv("LOCALM_PEER_DETECTION", "on")
    monkeypatch.setattr("localm.config.PORT_RANGE", (min(ports), max(ports)))


def localm_whoami(instance_id: str, **extra) -> dict:
    return {"app": "localm", "instance_id": instance_id, "version": "test",
            "root_dir": None, "mode": "full", **extra}
