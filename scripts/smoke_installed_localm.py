#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Start an installed ``localm gui --no-model`` and check it answers, then stop it.

Run from a directory that is NOT the source checkout, with the interpreter of the
environment that holds the installed wheel, so the package under test is the
installed one:

    python scripts/smoke_installed_localm.py --localm /path/to/venv/bin/localm

Checks, in order: the imported ``localm`` lives in site-packages; the server
answers ``/whoami``, ``/health`` (200, or 503 "No engine initialised" with no model
loaded), ``/v1/models`` (an empty or populated list), and ``/`` (the packaged GUI
page); a termination request ends the process within ``--stop-timeout`` seconds.

Exit 0 when every check passes, 1 otherwise. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def fetch(url: str, timeout: float = 5.0) -> tuple[int, bytes]:
    """``(status, body)``; an HTTP error status is returned, a connection failure raises OSError."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def wait_ready(base: str, proc: subprocess.Popen, deadline: float) -> str | None:
    """None once ``/whoami`` answers 200; otherwise why it never did."""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return f"the server exited with code {proc.returncode} before it answered"
        try:
            status, _ = fetch(f"{base}/whoami")
            if status == 200:
                return None
        except OSError:
            pass
        time.sleep(1.0)
    return "the server did not answer /whoami in time"


def check_endpoints(base: str) -> list[str]:
    problems: list[str] = []
    status, body = fetch(f"{base}/health")
    if status not in (200, 503):
        problems.append(f"/health answered {status}")
    elif status == 503 and b"No engine initialised" not in body:
        problems.append(f"/health answered 503 with an unexpected body: {body[:200]!r}")
    status, body = fetch(f"{base}/v1/models")
    if status != 200:
        problems.append(f"/v1/models answered {status}")
    else:
        try:
            models = json.loads(body)
            if models.get("object") != "list" or not isinstance(models.get("data"), list):
                problems.append(f"/v1/models is not a model list: {body[:200]!r}")
        except ValueError:
            problems.append(f"/v1/models is not JSON: {body[:200]!r}")
    status, body = fetch(f"{base}/")
    if status != 200 or b"<html" not in body.lower():
        problems.append(f"/ did not serve the GUI page (status {status})")
    return problems


def imported_from_site_packages() -> str | None:
    """None when ``import localm`` resolves into site-packages; otherwise why not."""
    out = subprocess.run([sys.executable, "-c", "import localm; print(localm.__file__)"],
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        return f"import localm failed: {out.stderr.strip()[-300:]}"
    path = out.stdout.strip()
    if "site-packages" not in Path(path).parts:
        return f"localm was imported from {path}, not from the installed wheel"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--localm", required=True, help="path to the installed localm console script")
    ap.add_argument("--start-timeout", type=float, default=120.0)
    ap.add_argument("--stop-timeout", type=float, default=30.0)
    args = ap.parse_args(argv)

    problem = imported_from_site_packages()
    if problem:
        print(f"::error::{problem}")
        return 1

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="localm-smoke-") as home:
        env = dict(os.environ, LOCALM_HOME=home)
        log_path = Path(home) / "gui.log"
        with open(log_path, "wb") as log:
            proc = subprocess.Popen(
                [args.localm, "gui", "--no-model", "--no-browser", "--isolated", "--port", str(port)],
                stdout=log, stderr=subprocess.STDOUT, env=env, cwd=home)
            problems: list[str] = []
            try:
                why = wait_ready(base, proc, time.monotonic() + args.start_timeout)
                if why:
                    problems.append(why)
                else:
                    problems += check_endpoints(base)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                try:
                    proc.wait(timeout=args.stop_timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    problems.append(f"the server did not stop within {args.stop_timeout:.0f}s of a terminate request")
        if problems:
            for item in problems:
                print(f"::error::{item}")
            print(log_path.read_text(encoding="utf-8", errors="replace")[-4000:])
            return 1
    print(f"installed localm answered on {base} and stopped on request")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
