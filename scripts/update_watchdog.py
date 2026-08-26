#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-restart health watchdog: verify a just-applied localm update actually came
back up, and roll it back automatically if it did not.

``localm/updater.py``'s ``spawn_health_watchdog()`` launches this DETACHED right
before the server re-execs into a freshly swapped build (``_do_restart`` in
``localm/inference/http_server.py``). By the time it runs, the file swap has
ALREADY happened, so this script executes from the NEW (possibly broken) build's
copy on disk. It is stdlib-only, with ZERO ``localm`` package imports, so a
broken new build cannot break the thing meant to detect it.

It polls the restarted instance's own unauthenticated ``GET /whoami`` (never
``/health``, which 503s whenever no model is loaded yet) until it answers with
the expected VERSION, or a bounded timeout elapses. On timeout it loads
``scripts/rollback_update.py`` - a sibling file, by file path - and calls its
``main(["--yes"], install_root=...)`` to restore the previous build.

A rollback that did not happen is NEVER reported as success: three distinct exit
codes distinguish "came back up healthy" from "rolled back" from "rollback
itself failed", and only the middle two ever touch the install.

Run standalone for manual testing:
    python scripts/update_watchdog.py --host 127.0.0.1 --port 8642 \\
        --expect-version 0.2.0 --install-root .
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

EXIT_HEALTHY = 0
EXIT_ROLLED_BACK = 1
EXIT_ROLLBACK_FAILED = 2

DEFAULT_TIMEOUT_S = 90.0
DEFAULT_POLL_INTERVAL_S = 2.0
DEFAULT_REQUEST_TIMEOUT_S = 3.0


def _log(log_path: Optional[Path], msg: str) -> None:
    """Timestamped trace line to stdout and, best-effort, *log_path*. A write
    error is swallowed."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _insecure_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def probe_once(url: str, timeout: float) -> Optional[dict]:
    """One GET against *url* (the instance's own /whoami). Returns the parsed JSON
    body on a 200, else None - a refused connection, a timeout, a non-200 status, or
    an unparseable body are all just "not healthy yet", never raised."""
    try:
        ctx = _insecure_ssl_context() if url.startswith("https://") else None
        req = urllib.request.Request(url, headers={"User-Agent": "localm-update-watchdog"})
        # ProxyHandler({}) disables proxy use for this request. opener.open()
        # takes no context= kwarg, so the TLS context rides on an HTTPSHandler.
        handlers = [urllib.request.ProxyHandler({})]
        if ctx is not None:
            handlers.append(urllib.request.HTTPSHandler(context=ctx))
        opener = urllib.request.build_opener(*handlers)
        with opener.open(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def wait_for_healthy(url: str, expect_version: str, *, total_timeout: float,
                     poll_interval: float, request_timeout: float,
                     log_path: Optional[Path]) -> bool:
    """Poll *url* until it answers with ``version == expect_version`` or
    *total_timeout* elapses. A single wrong-version or unreachable response is
    NOT a failure: polling continues until the deadline and only ever succeeds
    early."""
    deadline = time.monotonic() + total_timeout
    attempt = 0
    while True:
        attempt += 1
        body = probe_once(url, request_timeout)
        if body is not None and body.get("version") == expect_version:
            _log(log_path, f"healthy after {attempt} attempt(s): version "
                           f"{body.get('version')!r} matches")
            return True
        if body is None:
            _log(log_path, f"attempt {attempt}: {url} not reachable yet")
        else:
            _log(log_path, f"attempt {attempt}: got version {body.get('version')!r}, "
                           f"want {expect_version!r}")
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))


def _load_rollback_module(install_root: Path):
    """Load scripts/rollback_update.py BY FILE PATH (a sibling of this script),
    so a broken localm package cannot stop the rollback. Returns the module, or
    None if it cannot be loaded (the caller then reports manual-recovery
    guidance)."""
    path = install_root / "scripts" / "rollback_update.py"
    if not path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            "_localm_update_watchdog_rollback", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def _invoke_rollback(install_root: Path, log_path: Optional[Path]) -> int:
    rb = _load_rollback_module(install_root)
    if rb is None:
        _log(log_path,
            f"could not load scripts/rollback_update.py from {install_root} - "
            "cannot auto-rollback. Restore the previous build by hand from "
            f"{install_root}'s data-home updates/backup dir, or run rollback.bat "
            "/ rollback.sh in the install folder once the helper is fixed.")
        return EXIT_ROLLBACK_FAILED
    try:
        rc = int(rb.main(["--yes"], install_root=install_root))
    except Exception as e:
        _log(log_path, f"rollback raised an unexpected error: {e}")
        return EXIT_ROLLBACK_FAILED
    if rc == 0:
        _log(log_path, "rolled back to the previous build successfully")
        return EXIT_ROLLED_BACK
    _log(log_path, f"rollback_update.py reported failure (exit {rc}) - the backup "
                   "is kept; manual recovery needed")
    return EXIT_ROLLBACK_FAILED


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Verify a just-restarted localm build is healthy; auto-roll "
                    "back if it is not.")
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", required=True, type=int)
    ap.add_argument("--scheme", default="http", choices=("http", "https"))
    ap.add_argument("--expect-version", required=True)
    ap.add_argument("--install-root", required=True, type=Path)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_S)
    ap.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    ap.add_argument("--log-file", type=Path, default=None)
    args = ap.parse_args(argv)

    install_root = args.install_root.resolve()
    # Bracket an IPv6 host so the URL authority parses.
    _host = args.host
    if ":" in _host and not _host.startswith("["):
        _host = f"[{_host}]"
    url = f"{args.scheme}://{_host}:{args.port}/whoami"
    _log(args.log_file, f"watching {url} for version {args.expect_version!r} "
                        f"(timeout {args.timeout}s, poll every {args.poll_interval}s)")

    healthy = wait_for_healthy(
        url, args.expect_version, total_timeout=args.timeout,
        poll_interval=args.poll_interval, request_timeout=args.request_timeout,
        log_path=args.log_file)

    if healthy:
        code = EXIT_HEALTHY
    else:
        _log(args.log_file, "health check failed - invoking automatic rollback")
        code = _invoke_rollback(install_root, args.log_file)

    _log(args.log_file, f"watchdog exiting with code {code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
