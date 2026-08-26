# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Managed lifecycle for localm serve.

localcoder can start and stop a localm inference server automatically so the
user only needs one command:  localcoder --model gemma4-4b

The server is started as a background subprocess, monitored until the port
responds, and stopped when localcoder exits.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from typing import Optional

from .display import (
    print_info,
    print_server_ready,
    print_server_starting,
    print_server_timeout,
    print_warning,
)
from .proc_tail import StderrTail


# localm's claimed range (see localm.config.PORT_RANGE) - stays clear of
# ComfyUI (8188), A1111 (7860), and the 8000/8080/8888 dev-server crowd.
_DEFAULT_PORT = 8642
_PORT_RANGE_END = 8741
_STARTUP_TIMEOUT = 90     # seconds to wait for the server to come up
_POLL_INTERVAL   = 0.5    # seconds between port probes


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


class ManagedServer:
    """
    Starts `localm serve <model>` as a child process and tears it down on exit.

    Use as a context manager:

        with ManagedServer("gemma4-4b", port=8642) as srv:
            backend = make_localm_backend("gemma4-4b", port=srv.port)
    """

    def __init__(
        self,
        model: str,
        port: int = _DEFAULT_PORT,
        host: str = "127.0.0.1",
        extra_args: Optional[list[str]] = None,
    ) -> None:
        self.model      = model
        self.port       = port
        self.host       = host
        self._extra     = extra_args or []
        self._proc: Optional[subprocess.Popen] = None
        self._stderr: Optional[StderrTail] = None

    # ------------------------------------------------------------------ #

    def start(self) -> bool:
        """
        Start the server if the port isn't already listening.

        Returns True if the server is ready, False on timeout.
        """
        if _port_open(self.host, self.port):
            print_info(f"Port {self.port} already open - using existing server")
            return True

        print_server_starting(self.model, self.port)

        # Launch the SAME localm/venv that is running the coder, not a bare
        # "localm" resolved via PATH. In the self-contained-clone model the
        # venv's Scripts dir is deliberately not on PATH, so a bare name finds a
        # different (or broken) global localm, or nothing at all. Using
        # sys.executable + "-m localm" guarantees this interpreter's localm.
        cmd = [
            sys.executable, "-m", "localm", "serve", self.model,
            "--host", self.host,
            "--port", str(self.port),
            *self._extra,
        ]

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
        except (FileNotFoundError, OSError):
            print_warning(
                "Could not launch the local model server with this interpreter.\n"
                "Point --url at an existing OpenAI-compatible server instead."
            )
            return False
        self._stderr = StderrTail(self._proc)

        deadline = time.monotonic() + _STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if _port_open(self.host, self.port):
                print_server_ready(self.port)
                # Expose the server URL so tools (e.g. generate_image) can call
                # /v1/models/unload before handing VRAM to an external process.
                os.environ["LOCALM_URL"] = f"http://{self.host}:{self.port}/v1"
                return True
            if self._proc.poll() is not None:
                msg = f"localm serve exited early (code {self._proc.returncode})"
                tail = self._stderr.tail()
                if tail:
                    msg += f":\n{tail}"
                print_warning(msg)
                return False
            time.sleep(_POLL_INTERVAL)

        print_server_timeout(self._stderr.tail())
        return False

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    # ------------------------------------------------------------------ #

    def __enter__(self) -> "ManagedServer":
        if not self.start():
            raise RuntimeError(f"Failed to start localm serve {self.model}")
        return self

    def __exit__(self, *_) -> None:
        self.stop()


def find_free_port(start: int = _DEFAULT_PORT, end: int = _PORT_RANGE_END) -> int:
    """Find the first free TCP port in [start, end]."""
    for port in range(start, end):
        if not _port_open("127.0.0.1", port):
            return port
    return start
